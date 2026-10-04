"""Bottom-to-top layer renderer for ``moviepy.ae.Composition``.

The renderer implements the specification's pipeline (section 3.3)::

    for layer in comp.layers (bottom -> top) if layer contributes at t:
        buf = layer.source_buffer(source_time)        # footage / solid / precomp
        buf = layer.apply_masks(buf, t)                # WS-05 masks
        buf = layer.apply_effects(buf, t)              # WS-06 effect stack
        buf = warp(buf, view @ world_matrix, opacity)  # WS-02 transform
        buf = apply_track_matte(buf)                   # WS-05 track matte
        acc = blend(acc, buf, layer.blend_mode)        # WS-04
    return acc cropped to the region of interest

Adjustment layers (WS-06) are the exception: their effects run on the
accumulated result below them, and their transformed, masked, matted and
faded coverage limits where the effected pixels replace the backdrop.
Temporal effects on adjustment layers receive the composite below at other
times. Every layer is resampled directly into the requested region of
interest, so large or far off-screen layers cost only the visible pixels.

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
from contextlib import contextmanager
from dataclasses import replace

import numpy as np

from moviepy.ae._geometry import validate_flag, validate_rectangle
from moviepy.ae.blend.modes import blend
from moviepy.ae.buffer import Buffer
from moviepy.ae.context import RenderContext, _exact_time
from moviepy.ae.effects.adjustment import apply_adjustment
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


def _layers_under(comp, layer):
    """Return the composition layers stacked below ``layer``."""
    layers = list(comp.layers)
    for position, item in enumerate(layers):
        if item is layer:
            return layers[position + 1 :]
    return []


def _transparent(bounds):
    """Return a transparent Buffer covering a rectangle."""
    left, top, right, bottom = bounds
    array = np.zeros((bottom - top, right - left, 4), dtype=np.float32)
    return Buffer._publish(array, (left, top), "srgb", (np.float32(0), 0.0, True))


class _RenderCache:
    """Own intermediate results only while the outer render is active."""

    def __init__(self):
        self.values = {}
        self.active = True


@contextmanager
def _render_cache(context):
    """Share active nested work and retire results after the outer render."""
    cache = context.cache
    owner = not isinstance(cache, _RenderCache) or not cache.active
    if owner:
        cache = _RenderCache()
        context = replace(context, t=context._exact_time, cache=cache)
    try:
        yield context
    finally:
        if owner:
            cache.active = False
            cache.values.clear()


def _cached(context, key, compute):
    """Reuse a result, including None, inside a renderer-owned cache."""
    cache = context.cache
    if not isinstance(cache, _RenderCache) or not cache.active:
        return compute()
    if key not in cache.values:
        cache.values[key] = compute()
    return cache.values[key]


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
        with _render_cache(ctx) as ctx:
            return self._render(comp, ctx, bounds, clip)

    def _render(self, comp, ctx, bounds, clip):
        """Render inside the active cache scope."""
        view = view_matrix(ctx.resolution_scale)
        roi = None
        if clip:
            frame = (0, 0, *scaled_size(comp.size, ctx.resolution_scale))
            roi = frame if bounds is None else validate_rectangle(bounds)
        accumulator = self._composite(comp, ctx, view, roi)
        if accumulator is None:
            return _transparent((0, 0, 0, 0))
        if roi is None:
            return accumulator
        return accumulator.crop(roi).expand_to(roi)

    def _render_key(self, context, view, roi):
        """Distinguish every render setting that affects intermediate pixels."""
        return (
            id(self),
            self.include_guides,
            context.t,
            context._exact_time,
            context.fps,
            context.resolution_scale,
            context.quality,
            context.rng_seed,
            id(context.shutter),
            tuple(view.ravel()),
            roi,
        )

    def _composite(self, comp, ctx, view, roi, below=None):
        """Reuse the composite below a layer during this render."""
        key = ("composite", id(comp), id(below), self._render_key(ctx, view, roi))
        return _cached(
            ctx, key, lambda: self._composite_uncached(comp, ctx, view, roi, below)
        )

    def _composite_uncached(self, comp, ctx, view, roi, below):
        """Blend contributing layers bottom to top; stop under ``below``."""
        accumulator = None if roi is None else _transparent(roi)
        layers = self.contributing_layers(comp, ctx.t)
        if below is not None:
            allowed = {id(layer) for layer in _layers_under(comp, below)}
            layers = [layer for layer in layers if id(layer) in allowed]
        for layer in layers:
            if layer.is_adjustment:
                accumulator = self._adjust(comp, layer, accumulator, ctx, view, roi)
                continue
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
                layer_id=layer.id,
            )
        return accumulator

    def _adjust(self, comp, layer, accumulator, ctx, view, roi):
        """Apply an adjustment layer's effects to the accumulated backdrop."""
        if accumulator is None:
            return None
        coverage = self._matted_layer(layer, ctx, view, roi)

        def below_at(local_t):
            time = (
                _exact_time(layer.start_time)
                + _exact_time(local_t) * _exact_time(layer.stretch) / 100
            )
            result = self._composite(comp, ctx.with_time(time), view, roi, layer)
            return _transparent(accumulator.bounds) if result is None else result

        return apply_adjustment(
            layer, accumulator, coverage, ctx._exact_time, ctx, source_at=below_at
        )

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
        source = layer.prepared_source(t, context)
        if 0 in source.size:
            return None
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
        """Reuse a transformed and matted layer during this render."""
        key = ("layer", id(layer), self._render_key(context, view, roi))
        return _cached(
            context, key, lambda: self._matted_layer_uncached(layer, context, view, roi)
        )

    def _matted_layer_uncached(self, layer, context, view, roi):
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
