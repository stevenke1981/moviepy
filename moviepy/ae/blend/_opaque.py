"""Bounded SDR formulas that avoid repeated alpha and RGB conversions."""

import cv2
import numpy as np

from moviepy.ae.blend import _formulas as formulas
from moviepy.ae.buffer import Buffer, _owned_output


_TILE_PIXELS = 8192


def _multiply(backdrop, source, opacity, target):
    """Evaluate ``Cb * (Cs_premult + 1 - alpha_s)`` on opaque SDR."""
    weight = float(np.float32(1) - opacity)
    cv2.add(source, (weight,) * 4, dst=target)
    cv2.multiply(backdrop, target, dst=target)


def _overlay(backdrop, source, opacity, target):
    """Use ``Cb + min(Cb, 1-Cb) * (2*Cs_premult-alpha_s)``."""
    distance = np.subtract(np.float32(1), backdrop)
    np.minimum(backdrop, distance, out=distance)
    np.multiply(source, np.float32(2), out=target)
    target -= opacity
    target *= distance
    target += backdrop


def _planes(rgba):
    """Extract a contiguous RGB plane stack without an interleaved intermediate."""
    planes = np.empty((3, *rgba.shape[:2]), np.float32)
    cv2.mixChannels([rgba], list(planes), [0, 0, 1, 1, 2, 2])
    return planes


def _hue(backdrop, source, opacity, target):
    """Fuse the bounded W3C Hue affine steps on owned, contiguous planes."""
    cb, cs = _planes(backdrop), _planes(source)
    cs *= np.float32(1) / opacity
    np.clip(cs, 0, 1, out=cs)
    low = formulas._low(cs)
    span = formulas._high(cs) - low
    scale = formulas._safe_divide(formulas._saturation(cb), span, 0.0)
    cs -= low
    cs *= scale
    level, target_level = formulas._lum(cs), formulas._lum(cb)
    weight = _hue_weight(target_level, level, span * scale, opacity)
    cs -= level
    cs *= weight
    cs += opacity * target_level
    cb *= np.float32(1) - opacity
    cs += cb
    cv2.mixChannels(list(cs), [target], [0, 0, 1, 1, 2, 2])
    target[..., 3] = 1


def _hue_weight(target_level, level, high, opacity):
    """Combine ClipColor's two factors and source opacity for bounded Hue.

    SetSat gives colors S with minimum zero, maximum M and luminance Ls.
    SetLum gives C = S + Lb - Ls, so ClipColor's factors are Lb/Ls below
    zero and (1-Lb)/(M-Ls) above one. Both corrections have center Lb;
    their product f gives Lb + (S-Ls)*f. Keep both factors, including when
    float32 rounding activates both bounds. Only proven SDR enters here.
    """
    under = target_level < level
    over = high + (target_level - level) > np.float32(1)
    has_under, has_over = under.any(), over.any()
    if has_under:
        factor = formulas._safe_divide(target_level, level, 0.0)
        factor[~under] = 1
    else:
        factor = np.ones_like(level)
    if has_over:
        push = formulas._safe_divide(np.float32(1) - target_level, high - level, 0.0)
        push[~over] = 1
        factor *= push
    factor *= opacity
    return factor


def composite(base, source, formula):
    """Return an opaque SDR fast path, or None when its bounds do not apply.

    Multiply and Overlay stay in [0, 1] after each rounded operation. Their
    alpha channels evaluate to exactly one, so their summaries need no scan.
    Hue still checks the actual extrema: SetLum/ClipColor can round outside
    the unit interval and must not advertise unit-premultiplied metadata.
    """
    operation = {
        formulas.multiply: _multiply,
        formulas.overlay: _overlay,
        formulas.hue: _hue,
    }.get(formula)
    if (
        operation is None
        or base._uniform_alpha != 1
        or not base._unit_premultiplied
        or not source._unit_premultiplied
    ):
        return None
    opacity = source._uniform_alpha
    if opacity == 0:
        return base
    if opacity < np.finfo(np.float32).tiny:
        return None
    output = _owned_output(base.rgba.shape)
    tile_pixels = 32768 if operation is _overlay else _TILE_PIXELS
    rows = max(1, tile_pixels // base.size[0])
    bound, unit = 1.0, True
    for top in range(0, base.size[1], rows):
        target = output[top : top + rows]
        with np.errstate(all="ignore"):
            operation(
                base.rgba[top : top + rows],
                source.rgba[top : top + rows],
                opacity,
                target,
            )
        if operation is _hue:
            low, high = float(np.min(target)), float(np.max(target))
            if not np.isfinite(low) or not np.isfinite(high):
                raise ValueError("blend arithmetic must remain finite in float32")
            bound = max(bound, abs(low), abs(high))
            unit = unit and low >= 0 and high <= 1
    return Buffer._publish(
        output, base.offset, base.color_space, (np.float32(1), bound, unit)
    )
