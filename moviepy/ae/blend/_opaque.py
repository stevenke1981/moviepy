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
    """Extract contiguous RGB planes without an interleaved RGB intermediate."""
    planes = np.empty((3, *rgba.shape[:2]), np.float32)
    cv2.mixChannels([rgba], list(planes), [0, 0, 1, 1, 2, 2])
    return list(planes)


def _hue(backdrop, source, opacity, target):
    """Keep W3C SetSat/SetLum/ClipColor in planes through alpha compositing."""
    cb, cs = _planes(backdrop), _planes(source)
    inverse = np.float32(1) / opacity
    for plane in cs:
        plane *= inverse
        np.clip(plane, 0, 1, out=plane)
    low = formulas._low(cs)
    scale = formulas._safe_divide(
        formulas._saturation(cb), formulas._high(cs) - low, 0.0
    )
    for plane in cs:
        plane -= low
        plane *= scale
    delta = formulas._lum(cb) - formulas._lum(cs)
    for plane in cs:
        plane += delta
    _clip_hue(cs)
    for rgb, base in zip(cs, cb):
        rgb *= opacity
        rgb += (np.float32(1) - opacity) * base
    cv2.mixChannels(cs, [target], [0, 0, 1, 1, 2, 2])
    target[..., 3] = 1


def _clip_hue(planes):
    """Apply both W3C ClipColor corrections about the same luminance.

    Each correction is ``lum + (color - lum) * factor``. Their composition
    multiplies the factors, so the three mutable planes need only one pass.
    The extrema and luminance are those before either correction, as in the
    general formula; signed/HDR inputs never enter this bounded SDR path.
    """
    low, high = formulas._low(planes), formulas._high(planes)
    under, over = low < 0, high > 1
    has_under, has_over = under.any(), over.any()
    if not has_under and not has_over:
        return
    lum = formulas._lum(planes)
    if has_under:
        factor = formulas._safe_divide(lum, lum - low, 0.0)
        factor[~under] = 1
    else:
        factor = np.ones_like(lum)
    if has_over:
        push = formulas._safe_divide(np.float32(1) - lum, high - lum, 0.0)
        push[~over] = 1
        factor *= push
    for plane in planes:
        plane -= lum
        plane *= factor
        plane += lum


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
