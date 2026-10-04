"""Bottom-to-top layer renderer for ``moviepy.ae.Composition``.

The renderer implements the specification's pipeline for the stages that
exist so far::

    for layer in comp.layers (bottom -> top) if layer contributes at t:
        buf = layer.source_buffer(source_time)        # footage / solid / precomp
        buf = layer.apply_masks(buf, t)                # WS-05 masks
        buf = warp(buf, view @ world_matrix, opacity)  # WS-02 transform
        buf = apply_track_matte(buf)                   # WS-05 track matte
        acc = blend(acc, buf, layer.blend_mode)        # WS-04
    return acc cropped to the region of interest

Masks (WS-05) are applied to the source pixels before the transform and
track mattes (WS-05) multiply the transformed layer by the matte layer's alpha
or luma just before the blend. Effects (WS-06) will slot in after the masks. Every layer is resampled directly into the requested region of interest,
so large or far off-screen layers cost only the pixels that can be seen.

Examples
--------
>>> from moviepy.ae import Composition
>>> from moviepy.ae.renderer import Renderer
>>> comp = Composition(size=(4, 2), fps=10, duration=1)
>>> _ = comp.add_solid("red", color=(255, 0, 0))
>>> Renderer().render(comp, 0.0).to_uint8_rgb()[0, 0].tolist()
[255, 0, 0]
"""

import math

import numpy as np

from moviepy.ae._geometry import validate_flag, validate_rectangle
from moviepy.ae.blend.modes import blend
from moviepy.ae.buffer import Buffer
from moviepy.ae.context import RenderContext
from moviepy.ae.warp import (
    _integer_translation,
    destination_bounds,
    resolve_interpolation,
    warp_buffer,
)


def view_matrix(scale):
    """Return the pixel-center preserving matrix for a resolution scale.

    Pixel centers sit at integer coordinates, so scaling the frame about its
    top-left *corner* maps ``x`` to ``(x + 0.5) * scale - 0.5``.
    """
    offset = 0.5 * scale - 0.5
    return np.array(
        [[scale, 0.0, offset], [0.0, scale, offset], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def scaled_size(size, scale):
    """Return the pixel size of a frame rendered at ``scale``."""
    return (max(1, math.ceil(size[0] * scale)), max(1, math.ceil(size[1] * scale)))


def _intersection(first, second):
    """Intersect half-open rectangles; ``None`` when they do not overlap."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _transparent(bounds):
    """Return a transparent Buffer covering a rectangle."""
    left, top, right, bottom = bounds
    array = np.zeros((bottom - top, right - left, 4), dtype=np.float32)
    return Buffer._publish(array, (left, top), "srgb", (np.float32(0), 0.0, True))


class Renderer:
    """Render a composition's layer stack into one premultiplied Buffer.

    Parameters
    ----------
    include_guides : bool, optional
        Render guide layers too. AE excludes guide layers from final output
        and from parent compositions, which is the default here.

    Notes
    -----
    Solo follows AE: when any enabled layer of a composition is soloed, only
    soloed layers render. Disabled (``enabled=False``) layers never render,
    but they can still act as parents. Null layers have no pixels.
    """

    def __init__(self, *, include_guides=False):
        self.include_guides = validate_flag(include_guides, "include_guides")

    def render(self, comp, t, context=None, *, bounds=None, clip=True):
        """Render ``comp`` at composition time ``t``.

        Parameters
        ----------
        comp : Composition
            Composition to render.
        t : float
            Composition time in seconds.
        context : RenderContext, optional
            Quality, resolution scale and random seed. Its time and fps are
            replaced by ``t`` and the composition fps.
        bounds : tuple of int, optional
            Region of interest in (scaled) composition pixels. Defaults to the
            whole frame.
        clip : bool, optional
            When false, render the union of all layer pixels without clipping
            to the frame (used by Collapse Transformations).

        Returns
        -------
        Buffer
            Premultiplied pixels. With ``clip`` the buffer covers exactly the
            region of interest; its alpha is the composition alpha.
        """
        ctx = self._context(comp, t, context)
        view = view_matrix(ctx.resolution_scale)
        if clip:
            frame = (0, 0, *scaled_size(comp.size, ctx.resolution_scale))
            roi = frame if bounds is None else validate_rectangle(bounds)
            accumulator = _transparent(roi)
        else:
            roi, accumulator = None, None
        for layer in self.contributing_layers(comp, ctx.t):
            rendered = self._matted_layer(layer, ctx, view, roi)
            if rendered is None:
                continue
            if accumulator is None:
                accumulator = _transparent(rendered.bounds)
            accumulator = blend(
                accumulator,
                rendered,
                layer.blend_mode,
                preserve_underlying_transparency=layer.preserve_transparency,
                context=ctx,
                layer_id=layer.name,
            )
        if accumulator is None:
            return _transparent((0, 0, 0, 0))
        if roi is None:
            return accumulator
        return accumulator.crop(roi).expand_to(roi)

    def contributing_layers(self, comp, t):
        """Return layers that render at ``t``, ordered bottom to top."""
        layers = [
            layer
            for layer in comp.layers
            if layer.enabled and (self.include_guides or not layer.guide)
        ]
        solo_active = any(layer.solo for layer in layers)
        return [
            layer
            for layer in reversed(layers)
            if layer.visible(t, solo_active=solo_active)
        ]

    def render_layer(self, layer, context, view, roi):
        """Transform one layer into (scaled) composition space.

        Returns ``None`` when the layer has no pixels inside ``roi``.
        """
        t = context.t
        source = layer.source_buffer(layer.source_time(t), context)
        if 0 in source.size:
            return None
        source = layer.apply_masks(source, t, context)
        matrix = view @ layer.world_matrix(t, context)
        opacity = layer.opacity_at(t, context)
        interpolation = resolve_interpolation(layer.transform.interpolation, context)
        natural = destination_bounds(
            matrix, source.size, source.offset, interpolation=interpolation
        )
        target = natural if roi is None else _intersection(natural, roi)
        if target is None or opacity <= 0.0:
            return None
        if target == natural or _integer_translation(matrix) is not None:
            # Whole layers and integer shifts keep warp_buffer's copy-free path;
            # cropping afterwards touches only the visible pixels.
            rendered = warp_buffer(source, matrix, opacity, interpolation=interpolation)
            rendered = rendered if target == natural else rendered.crop(target)
        else:
            rendered = warp_buffer(
                source, matrix, opacity, interpolation=interpolation, bounds=target
            )
        return None if 0 in rendered.size else rendered

    def _matted_layer(self, layer, context, view, roi):
        """Render one layer and apply its track matte, if any."""
        rendered = self.render_layer(layer, context, view, roi)
        if rendered is not None and layer.track_matte is not None:
            rendered = self.apply_track_matte(layer, rendered, context, view, roi)
        return rendered

    def apply_track_matte(self, layer, rendered, context, view, roi):
        """Multiply a transformed layer by its track matte's opacity.

        The matte layer renders through its own masks, transform, opacity and
        (recursively) its own track matte, even when its video switch is off,
        but only inside its time window. Where the matte has no pixels an
        alpha/luma matte hides the layer and an inverted matte shows it.
        """
        matte = layer.track_matte
        source = matte.layer
        t = context.t
        matte_buffer = None
        if source.in_point <= t < source.out_point:
            matte_buffer = self._matted_layer(source, context, view, roi)
        bounds = rendered.bounds
        if matte_buffer is None:
            matte_buffer = _transparent(bounds)
        aligned = matte_buffer.crop(bounds).expand_to(bounds)
        values = matte.values(aligned)
        rgba = rendered.rgba * values[..., None]
        return Buffer._publish(rgba, rendered.offset, rendered.color_space)

    @staticmethod
    def _context(comp, t, context):
        """Bind the composition time and frame rate into a render context."""
        if context is None:
            return RenderContext(t=t, fps=comp.fps)
        if not isinstance(context, RenderContext):
            raise TypeError("context must be a RenderContext")
        return RenderContext(
            t=t,
            fps=comp.fps,
            resolution_scale=context.resolution_scale,
            quality=context.quality,
            rng_seed=context.rng_seed,
            cache=context.cache,
            shutter=context.shutter,
        )
