"""Constant-coverage blending with bounded temporary storage."""

import cv2
import numpy as np

from moviepy.ae.blend import _formulas as formulas
from moviepy.ae.buffer import Buffer


TILE_PIXELS = 32768


def _premultiplied(base, source, formula, preserve):
    """Use bounded SDR identities; extended inputs retain the general formula."""
    if not base._unit_premultiplied or not source._unit_premultiplied:
        return None
    if formula not in (formulas.multiply, formulas.screen):
        return None
    ab, opacity = base._uniform_alpha, source._uniform_alpha
    if formula is formulas.multiply:
        rgb = cv2.multiply(base.rgba, source.rgba)
        cv2.addWeighted(rgb, 1, base.rgba, float(np.float32(1) - opacity), 0, dst=rgb)
        if not preserve and ab != 1:
            cv2.addWeighted(rgb, 1, source.rgba, float(np.float32(1) - ab), 0, dst=rgb)
    else:
        rgb = cv2.multiply(base.rgba, source.rgba, scale=-1)
        cv2.add(rgb, base.rgba, dst=rgb)
        cv2.addWeighted(rgb, 1, source.rgba, float(ab) if preserve else 1, 0, dst=rgb)
    alpha = ab if preserve else opacity + ab - opacity * ab
    alpha = np.float32(np.clip(alpha, 0, 1))
    rgb[..., 3] = alpha
    low, high = float(np.min(rgb)), float(np.max(rgb))
    metadata = (alpha, max(abs(low), abs(high)), low >= 0 and high <= float(alpha))
    return Buffer._publish(rgb, base.offset, base.color_space, metadata)


def _rgb(rgba):
    """Extract only RGB; a known constant alpha needs no replicated planes."""
    rgb = np.empty((*rgba.shape[:2], 3), np.float32)
    cv2.mixChannels([rgba], [rgb], [0, 0, 1, 1, 2, 2])
    return rgb


def _mix(cb, cs, ab, opacity, formula, clamp, classic, preserve):
    """Keep the general compositor's float32 operation order."""
    if classic:
        backdrop, source = cb.copy(), cs.copy()
    else:
        backdrop = cb * (np.float32(1) / ab if ab > 0 else np.float32(0))
        source = cs * (np.float32(1) / opacity if opacity > 0 else np.float32(0))
    if clamp:
        np.clip(backdrop, 0, 1, out=backdrop)
        np.clip(source, 0, 1, out=source)
    rgb = np.asarray(formula(backdrop, source), np.float32)
    rgb *= opacity
    rgb *= ab
    rgb += (np.float32(1) - opacity) * cb
    if not preserve:
        rgb += (np.float32(1) - ab) * cs
    return rgb


def composite(base, source, formula, *, clamp, classic, preserve):
    """Return a Buffer for aligned uniform coverage, or None for the fallback."""
    ab, opacity = base._uniform_alpha, source._uniform_alpha
    if base.bounds != source.bounds or ab is None or opacity is None:
        return None
    native = None if classic else _premultiplied(base, source, formula, preserve)
    if native is not None:
        return native
    alpha = ab if preserve else opacity + ab - opacity * ab
    alpha = np.float32(np.clip(alpha, 0, 1))
    output = np.empty_like(base.rgba)
    bound, unit = float(alpha), True
    rows = max(1, TILE_PIXELS // base.size[0])
    for top in range(0, base.size[1], rows):
        target = output[top : top + rows]
        cb, cs = _rgb(base.rgba[top : top + rows]), _rgb(source.rgba[top : top + rows])
        with np.errstate(all="ignore"):
            rgb = _mix(cb, cs, ab, opacity, formula, clamp, classic, preserve)
        if not np.isfinite(rgb).all():
            raise ValueError("blend arithmetic must remain finite in float32")
        if alpha == 0:
            rgb.fill(0)
        low, high = float(np.min(rgb)), float(np.max(rgb))
        bound = max(bound, abs(low), abs(high))
        unit = unit and low >= 0 and high <= float(alpha)
        cv2.mixChannels([rgb], [target], [0, 0, 1, 1, 2, 2])
        target[..., 3] = alpha
    return Buffer._publish(output, base.offset, base.color_space, (alpha, bound, unit))
