"""Per-pixel blend functions ``B(Cb, Cs)`` on straight float32 RGB.

Every function receives the backdrop color ``cb`` and the source color ``cs``
as ``(H, W, 3)`` float32 arrays of *straight* (not premultiplied) values and
returns a new array of the same shape. Alpha handling lives in
``moviepy.ae.blend.alpha``; these functions only describe color mixing.

Formula sources
---------------
* W3C *Compositing and Blending Level 1* (§10 "Blending"): normal, multiply,
  screen, overlay, darken, lighten, color-dodge, color-burn, hard-light,
  soft-light, difference, exclusion and the non-separable hue, saturation,
  color and luminosity modes, including ``Lum``, ``ClipColor``, ``SetLum``,
  ``Sat`` and ``SetSat``.
* The widely published Photoshop-family definitions (also described in the
  Adobe After Effects user guide's "Blending mode reference" prose) for
  linear burn, linear dodge, vivid light, linear light, pin light, hard mix,
  subtract, divide, darker color and lighter color.
"""

import numpy as np


# W3C Compositing Level 1 luminosity coefficients for the HSL-family modes.
W3C_LUMA = np.array([0.3, 0.59, 0.11], dtype=np.float32)
# ITU-R BT.709 coefficients, used for luma-based stencil/silhouette modes.
REC709_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
REC601_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

_ONE = np.float32(1.0)
_HALF = np.float32(0.5)


def _safe_divide(numerator, denominator, fill):
    """Divide where the denominator is positive and use ``fill`` elsewhere."""
    result = np.full(np.broadcast(numerator, denominator).shape, fill, np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def luminance(rgb, coefficients=W3C_LUMA):
    """Return the weighted channel sum with a trailing singleton axis."""
    import cv2

    source = np.ascontiguousarray(rgb, dtype=np.float32)
    if not source.size:
        return np.zeros((*source.shape[:-1], 1), dtype=np.float32)
    weights = np.asarray(coefficients, dtype=np.float32).reshape(1, 3)
    return cv2.transform(source, weights).reshape(*source.shape[:-1], 1)


# -- separable modes ----------------------------------------------------------


def normal(cb, cs):
    """Return the source color."""
    return cs.copy()


def multiply(cb, cs):
    """Return ``Cb * Cs`` (W3C multiply)."""
    return cb * cs


def screen(cb, cs):
    """Return ``Cb + Cs - Cb * Cs`` (W3C screen)."""
    return cb + cs - cb * cs


def darken(cb, cs):
    """Return the channel minimum (W3C darken)."""
    return np.minimum(cb, cs)


def lighten(cb, cs):
    """Return the channel maximum (W3C lighten)."""
    return np.maximum(cb, cs)


def color_dodge(cb, cs):
    """Return W3C color-dodge: 0 if Cb=0, 1 if Cs=1, else min(1, Cb/(1-Cs))."""
    result = np.minimum(_ONE, _safe_divide(cb, _ONE - cs, 1.0))
    result[cb <= 0] = 0.0
    return result


def color_burn(cb, cs):
    """Return W3C color-burn: 1 if Cb=1, 0 if Cs=0, else 1-min(1,(1-Cb)/Cs)."""
    result = _ONE - np.minimum(_ONE, _safe_divide(_ONE - cb, cs, 1.0))
    result[cb >= 1] = 1.0
    return result


def classic_color_dodge(cb, cs):
    """Return the legacy dodge ``min(1, Cb / (1 - Cs))`` without the Cb=0 rule."""
    return np.minimum(_ONE, _safe_divide(cb, _ONE - cs, 1.0))


def classic_color_burn(cb, cs):
    """Return the legacy burn ``1 - min(1, (1 - Cb) / Cs)`` without the Cb=1 rule."""
    return _ONE - np.minimum(_ONE, _safe_divide(_ONE - cb, cs, 1.0))


def linear_burn(cb, cs):
    """Return ``max(0, Cb + Cs - 1)``."""
    return np.maximum(np.float32(0), cb + cs - _ONE)


def linear_dodge(cb, cs):
    """Return ``min(1, Cb + Cs)``."""
    return np.minimum(_ONE, cb + cs)


def add(cb, cs):
    """Return the unclipped sum ``Cb + Cs``; clipping happens only at export."""
    return cb + cs


def hard_light(cb, cs):
    """Return W3C hard-light: multiply below 0.5, screen above."""
    doubled = cs + cs
    return np.where(cs <= _HALF, cb * doubled, screen(cb, doubled - _ONE))


def overlay(cb, cs):
    """Return W3C overlay, i.e. ``HardLight(Cs, Cb)``."""
    return hard_light(cs, cb)


def soft_light(cb, cs):
    """Return the W3C soft-light formula with its ``D(Cb)`` helper."""
    curve = np.where(
        cb <= np.float32(0.25),
        ((np.float32(16) * cb - np.float32(12)) * cb + np.float32(4)) * cb,
        np.sqrt(np.maximum(cb, 0)),
    )
    darker = cb - (_ONE - cs - cs) * cb * (_ONE - cb)
    lighter = cb + (cs + cs - _ONE) * (curve - cb)
    return np.where(cs <= _HALF, darker, lighter)


def vivid_light(cb, cs):
    """Return color burn with ``2Cs`` below 0.5, color dodge with ``2Cs-1`` above."""
    doubled = cs + cs
    return np.where(
        cs <= _HALF, color_burn(cb, doubled), color_dodge(cb, doubled - _ONE)
    )


def linear_light(cb, cs):
    """Return ``clip(Cb + 2Cs - 1, 0, 1)``."""
    return np.clip(cb + cs + cs - _ONE, 0, 1)


def pin_light(cb, cs):
    """Return ``min(Cb, 2Cs)`` below 0.5 and ``max(Cb, 2Cs - 1)`` above."""
    doubled = cs + cs
    return np.where(
        cs <= _HALF, np.minimum(cb, doubled), np.maximum(cb, doubled - _ONE)
    )


def hard_mix(cb, cs):
    """Return 1 where ``Cb + Cs >= 1`` and 0 elsewhere (thresholded vivid light)."""
    return (cb + cs >= _ONE).astype(np.float32)


def difference(cb, cs):
    """Return ``|Cb - Cs|`` (W3C difference)."""
    return np.abs(cb - cs)


def exclusion(cb, cs):
    """Return ``Cb + Cs - 2 Cb Cs`` (W3C exclusion)."""
    return cb + cs - np.float32(2) * cb * cs


def subtract(cb, cs):
    """Return ``max(0, Cb - Cs)``."""
    return np.maximum(np.float32(0), cb - cs)


def divide(cb, cs):
    """Return ``min(1, Cb / Cs)``; a zero source yields 1 (or 0 for black Cb)."""
    result = np.minimum(_ONE, _safe_divide(cb, cs, 1.0))
    result[(cs <= 0) & (cb <= 0)] = 0.0
    return result


# -- whole-color selection modes ---------------------------------------------


def darker_color(cb, cs):
    """Keep whichever whole color has the lower W3C luminosity (ties keep Cb)."""
    return np.where(luminance(cs) < luminance(cb), cs, cb)


def lighter_color(cb, cs):
    """Keep whichever whole color has the higher W3C luminosity (ties keep Cb)."""
    return np.where(luminance(cs) > luminance(cb), cs, cb)


# -- non-separable W3C modes ---------------------------------------------------
# These work on planar (H, W) channels: reductions across a 3-wide inner axis
# are an order of magnitude slower in NumPy than elementwise plane arithmetic.


def _planes(rgb):
    """Split ``(H, W, 3)`` into three contiguous float32 planes."""
    import cv2

    if not rgb.size:
        return [np.zeros(rgb.shape[:-1], dtype=np.float32) for _ in range(3)]
    return list(cv2.split(np.ascontiguousarray(rgb, dtype=np.float32)))


def _stack(planes):
    """Merge three planes back into ``(H, W, 3)``."""
    import cv2

    if not planes[0].size:
        return np.zeros((*planes[0].shape, 3), dtype=np.float32)
    return cv2.merge(planes)


def _lum(planes):
    """Return W3C ``Lum`` of planar RGB."""
    red, green, blue = planes
    return W3C_LUMA[0] * red + W3C_LUMA[1] * green + W3C_LUMA[2] * blue


def _low(planes):
    return np.minimum(np.minimum(planes[0], planes[1]), planes[2])


def _high(planes):
    return np.maximum(np.maximum(planes[0], planes[1]), planes[2])


def _clip_color(planes):
    """Return W3C ``ClipColor``: pull out-of-gamut colors toward their luminance."""
    low, high = _low(planes), _high(planes)
    under, over = low < 0, high > 1
    if not under.any() and not over.any():
        return planes
    lum = _lum(planes)
    pull = _safe_divide(lum, lum - low, 0.0)
    push = _safe_divide(_ONE - lum, high - lum, 0.0)
    result = []
    for plane in planes:
        plane = np.where(under, lum + (plane - lum) * pull, plane)
        result.append(np.where(over, lum + (plane - lum) * push, plane))
    return result


def _set_lum(planes, lum):
    """Return W3C ``SetLum``: shift a color to a target luminosity."""
    delta = lum - _lum(planes)
    return _clip_color([plane + delta for plane in planes])


def _saturation(planes):
    """Return W3C ``Sat``: the channel range."""
    return _high(planes) - _low(planes)


def _set_sat(planes, saturation):
    """Return W3C ``SetSat``: rescale the channel range to ``saturation``."""
    low = _low(planes)
    scale = _safe_divide(saturation, _high(planes) - low, 0.0)
    return [(plane - low) * scale for plane in planes]


def hue(cb, cs):
    """Return W3C hue: source hue with backdrop saturation and luminosity."""
    backdrop, source = _planes(cb), _planes(cs)
    shifted = _set_sat(source, _saturation(backdrop))
    return _stack(_set_lum(shifted, _lum(backdrop)))


def saturation(cb, cs):
    """Return W3C saturation: source saturation, backdrop hue and luminosity."""
    backdrop, source = _planes(cb), _planes(cs)
    shifted = _set_sat(backdrop, _saturation(source))
    return _stack(_set_lum(shifted, _lum(backdrop)))


def color(cb, cs):
    """Return W3C color: source hue and saturation with backdrop luminosity."""
    return _stack(_set_lum(_planes(cs), _lum(_planes(cb))))


def luminosity(cb, cs):
    """Return W3C luminosity: backdrop hue and saturation, source luminosity."""
    return _stack(_set_lum(_planes(cb), _lum(_planes(cs))))
