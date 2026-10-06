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

from moviepy.ae._accumulator import _Accumulator
from moviepy.ae._geometry import validate_flag, validate_rectangle
from moviepy.ae.buffer import Buffer
from moviepy.ae.context import RenderContext, _exact_time
from moviepy.ae.effects.adjustment import apply_adjustment
from moviepy.ae.effects.base import AEEffect
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


def _transparent(bounds, color_space="srgb"):
    """Return a transparent Buffer covering a rectangle."""
    left, top, right, bottom = bounds
    array = np.zeros((bottom - top, right - left, 4), dtype=np.float32)
    return Buffer._publish(array, (left, top), color_space, (np.float32(0), 0.0, True))


def _needs_full_effect_frame(effects):
    """Identify stacks whose nested temporal reads have unknown spatial demand."""
    temporal = sum(
        type(effect).temporal_window is not AEEffect.temporal_window
        for effect in effects
    )
    spatial = any(
        type(effect).input_margin is not AEEffect.input_margin
        or type(effect).bounds_expand is not AEEffect.bounds_expand
        for effect in effects
    )
    return temporal > 1 and spatial


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
            return _transparent((0, 0, 0, 0), ctx.working_space)
        if roi is None or accumulator.bounds == roi:
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
            context.working_space,
            context.rng_seed,
            id(context.shutter),
            context._footage_offsets,
            context._exposure_center,
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
        layers = self.contributing_layers(comp, ctx.t)
        if below is not None:
            allowed = {id(layer) for layer in _layers_under(comp, below)}
            layers = [layer for layer in layers if id(layer) in allowed]
        requested = roi
        roi = self._effect_roi(comp, layers, ctx, roi)
        accumulator = _Accumulator(roi, ctx.working_space)
        for layer in layers:
            if layer.is_adjustment:
                accumulator.replace(
                    self._adjust(comp, layer, accumulator.snapshot(), ctx, view, roi)
                )
                continue
            if (
                layer.blend_mode in ("stencil_alpha", "stencil_luma")
                and layer.motion_blur
                and ctx.shutter is not None
            ):
                accumulator.replace(
                    self._stencil_exposure(
                        layer, accumulator.snapshot(), ctx, view, roi
                    )
                )
                continue
            rendered = self._matted_layer(layer, ctx, view, roi)
            if rendered is None:
                continue
            accumulator.blend(
                rendered,
                layer.blend_mode,
                preserve=layer.preserve_transparency,
                context=ctx,
                layer_id=layer.id,
            )
        accumulator = accumulator.snapshot()
        if requested is not None and roi != requested:
            return accumulator.crop(requested).expand_to(requested)
        return accumulator

    def _stencil_exposure(self, layer, backdrop, context, view, roi):
        """Apply zero-opacity opt-out inside each Stencil exposure sample."""
        from moviepy.ae.blend import blend
        from moviepy.ae.time.motion_blur import sample_layer

        def render_one(sample):
            instant = replace(sample, t=sample._exact_time, shutter=None)
            source = self._matted_layer(layer, instant, view, roi)
            if source is None:
                return backdrop
            return blend(
                backdrop,
                source,
                layer.blend_mode,
                preserve_underlying_transparency=layer.preserve_transparency,
                context=instant,
                layer_id=layer.id,
            )

        return sample_layer(
            layer,
            context,
            render_one,
            outside=_transparent(backdrop.bounds, backdrop.color_space),
        )

    @staticmethod
    def _effect_roi(comp, layers, ctx, roi):
        """Include adjustment input dependencies, bounded by the canvas."""
        if roi is None:
            return None
        size = scaled_size(comp.size, ctx.resolution_scale)
        margins = [0, 0, 0, 0]
        for layer in layers:
            if not layer.is_adjustment:
                continue
            effects = layer.effects.active()
            if _needs_full_effect_frame(effects) or (
                layer.motion_blur and ctx.shutter is not None
            ):
                # A nested temporal stage may replay a spatial effect at a
                # time with a larger radius. The current margins cannot bound
                # that demand, including when the current radius is zero.
                return (
                    min(roi[0], 0),
                    min(roi[1], 0),
                    max(roi[2], size[0]),
                    max(roi[3], size[1]),
                )
            local_t = float(
                (ctx._exact_time - _exact_time(layer.start_time))
                * 100
                / _exact_time(layer.stretch)
            )
            for effect in effects:
                values = effect.values_at(
                    local_t,
                    ctx,
                    pixel_scale=ctx.resolution_scale,
                    **layer.expression_bindings,
                )
                needed = effect.input_margin(size, local_t, ctx, values)
                margins = [a + max(0, math.ceil(b)) for a, b in zip(margins, needed)]
        if not any(margins):
            return roi
        expanded = _intersection(
            (0, 0, *size),
            (
                roi[0] - margins[0],
                roi[1] - margins[1],
                roi[2] + margins[2],
                roi[3] + margins[3],
            ),
        )
        if expanded is None:
            return roi
        # Preserve explicitly requested pixels outside the canvas, if any.
        return (
            min(roi[0], expanded[0]),
            min(roi[1], expanded[1]),
            max(roi[2], expanded[2]),
            max(roi[3], expanded[3]),
        )

    def _adjust(self, comp, layer, accumulator, ctx, view, roi):
        """Apply an adjustment layer's effects to the accumulated backdrop."""
        if accumulator is None:
            return None
        if layer.motion_blur and ctx.shutter is not None:
            from moviepy.ae.time.motion_blur import sample_layer

            return sample_layer(
                layer,
                ctx,
                lambda sample: self._adjust_at(
                    comp, layer, accumulator, sample, view, roi, instantaneous=True
                ),
                outside=accumulator,
            )
        return self._adjust_at(comp, layer, accumulator, ctx, view, roi)

    def _adjust_at(
        self, comp, layer, accumulator, ctx, view, roi, *, instantaneous=False
    ):
        """Apply adjustment properties at one time to the current backdrop."""
        coverage_context = (
            replace(ctx, t=ctx._exact_time, shutter=None) if instantaneous else ctx
        )
        coverage = self._matted_layer(layer, coverage_context, view, roi)

        def below_at(local_t):
            time = (
                _exact_time(layer.start_time)
                + _exact_time(local_t) * _exact_time(layer.stretch) / 100
            )
            when = ctx.with_time(time)
            # Earlier effects in this same stack are re-run at the sampled
            # time. Their animated margins may exceed the current margins.
            sample_roi = self._effect_roi(comp, [layer], when, roi)
            result = self._composite(comp, when, view, sample_roi, layer)
            return (
                _transparent(accumulator.bounds, ctx.working_space)
                if result is None
                else result
            )

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
        if layer.motion_blur and context.shutter is not None:
            from moviepy.ae.time.motion_blur import sample_layer

            return sample_layer(
                layer,
                context,
                lambda sample: self._render_layer_at(layer, sample, view, roi),
            )
        return self._render_layer_at(layer, context, view, roi)

    def _render_layer_at(self, layer, context, view, roi):
        """Transform a single sample, without recursively applying its shutter."""
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
        if opacity <= 0.0:
            # Exact zero also disables Stencil under the selected project
            # policy; positive-opacity empty coverage still masks the backdrop.
            return None
        if target is None:
            if layer.blend_mode in ("stencil_alpha", "stencil_luma"):
                # An enabled stencil still masks every underlying pixel when
                # its coverage lies outside the requested rendering region.
                return _transparent((0, 0, 0, 0), context.working_space)
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
        if layer.motion_blur and context.shutter is not None:
            from moviepy.ae.time.motion_blur import sample_layer

            return sample_layer(
                layer,
                context,
                lambda sample: self._matted_layer_at(
                    layer,
                    replace(sample, t=sample._exact_time, shutter=None),
                    view,
                    roi,
                ),
            )
        return self._matted_layer_at(layer, context, view, roi)

    def _matted_layer_at(self, layer, context, view, roi):
        """Apply a track matte inside the same instantaneous temporal sample."""
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
            sample_context = context
            if context._exposure_center is not None:
                from moviepy.ae.time.motion_blur import _sample_context

                sample_context = _sample_context(
                    source,
                    context.with_time(context._exposure_center),
                    context._exact_time,
                )
            matte_buffer = self._matted_layer(source, sample_context, view, roi)
        bounds = rendered.bounds
        if matte_buffer is None:
            matte_buffer = _transparent(bounds, context.working_space)
        aligned = matte_buffer.crop(bounds).expand_to(bounds)
        values = matte.values(aligned)
        rgba = rendered.rgba * values[..., None]
        return Buffer._publish(rgba, rendered.offset, rendered.color_space)

    @staticmethod
    def _context(comp, t, context):
        """Bind the composition time and frame rate into a render context."""
        if context is None:
            context = RenderContext()
        if not isinstance(context, RenderContext):
            raise TypeError("context must be a RenderContext")
        shutter = None
        if comp.motion_blur:
            from moviepy.ae.time.motion_blur import Shutter

            shutter = Shutter.coerce(context.shutter)
        return RenderContext(
            t=t,
            fps=comp.fps,
            resolution_scale=context.resolution_scale,
            quality=context.quality,
            rng_seed=context.rng_seed,
            cache=context.cache,
            shutter=shutter,
            working_space=context.working_space,
            _footage_offsets=context._footage_offsets,
            _exposure_center=context._exposure_center,
        )
