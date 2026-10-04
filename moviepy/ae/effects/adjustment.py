"""Adjustment-layer compositing: effects on the backdrop, limited by coverage."""

import numpy as np

from moviepy.ae.blend.modes import BlendMode, blend
from moviepy.ae.buffer import Buffer


def apply_adjustment(layer, below, coverage, t, context=None, *, source_at=None):
    """Apply an adjustment layer to the accumulated pixels ``below`` it.

    Parameters
    ----------
    layer : Layer
        Layer with ``is_adjustment``; its ``effects`` run on ``below`` in
        composition space at the layer's own time.
    below : Buffer
        World-space accumulation of every layer under ``layer``.
    coverage : Buffer or None
        The adjustment layer rendered like a white solid: masks, transform,
        opacity and track matte. Its alpha is where the effects apply.
    t : float
        Composition time.
    context : RenderContext, optional
        Render settings and expression context.
    source_at : callable, optional
        ``source_at(layer_time)`` returns the composite below at another time,
        for temporal effects.

    Returns
    -------
    Buffer
        ``below`` with the effected pixels mixed in. With the Normal mode the
        mix is ``below + (effected - below) * coverage``; pixels with zero
        coverage, disabled effects or an empty stack keep ``below`` bit for
        bit. Other blend modes blend ``effected * coverage`` onto ``below``.
    """
    if coverage is None or 0 in coverage.size or 0 in below.size:
        return below
    bounds = below.bounds
    alpha = coverage.crop(bounds).expand_to(bounds).rgba[..., 3:]
    if not alpha.any() or not layer.effects.active():
        return below
    effected = layer.effects.apply(
        below,
        layer.source_time(t),
        context,
        bindings=dict(layer.expression_bindings),
        source_at=source_at,
    )
    return _mix(layer, below, effected.crop(bounds).expand_to(bounds), alpha, context)


def _mix(layer, below, effected, alpha, context):
    """Mix aligned effected pixels into ``below`` through ``alpha``."""
    mode = BlendMode.coerce(layer.blend_mode)
    if mode is BlendMode.NORMAL and not layer.preserve_transparency:
        rgba = below.rgba + (effected.rgba - below.rgba) * alpha
        return Buffer._publish(
            rgba.astype(np.float32, copy=False), below.offset, below.color_space
        )
    scaled = Buffer._publish(
        (effected.rgba * alpha).astype(np.float32, copy=False),
        below.offset,
        below.color_space,
    )
    return blend(
        below,
        scaled,
        mode,
        preserve_underlying_transparency=layer.preserve_transparency,
        context=context,
        layer_id=layer.id,
    )
