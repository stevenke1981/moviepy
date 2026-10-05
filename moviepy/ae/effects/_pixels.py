"""Shared pixel helpers for built-in effects (straight-color round trips)."""

import numpy as np

from moviepy.ae.blend._formulas import REC709_LUMA
from moviepy.ae.buffer import Buffer
from moviepy.ae.color import srgb_to_linear


_OPAQUE = (np.float32(1.0), 1.0, True)


def is_opaque(buffer):
    """Return whether every pixel of ``buffer`` has alpha exactly 1."""
    return buffer._uniform_alpha is not None and buffer._uniform_alpha == 1


def straight_rgb(buffer):
    """Return independent straight float32 RGB (0 where alpha is 0)."""
    rgba = buffer.rgba
    if is_opaque(buffer):
        return rgba[..., :3].copy()
    alpha = rgba[..., 3:]
    rgb = np.zeros(rgba.shape[:2] + (3,), dtype=np.float32)
    np.divide(rgba[..., :3], alpha, out=rgb, where=alpha > 0)
    return rgb


def publish_straight(rgb, alpha, like, opaque=False):
    """Clip straight ``rgb`` to [0, 1], premultiply by ``alpha`` and publish.

    ``alpha`` is an ``(H, W, 1)`` array in [0, 1] (all ones when ``opaque``);
    ``like`` supplies offset and color space.
    """
    rgba = np.empty(rgb.shape[:2] + (4,), dtype=np.float32)
    color = rgba[..., :3]
    np.maximum(rgb, np.float32(0.0), out=color)
    np.minimum(color, np.float32(1.0), out=color)
    if opaque:
        rgba[..., 3] = 1.0
        return Buffer._publish(rgba, like.offset, like.color_space, _OPAQUE)
    color *= alpha
    rgba[..., 3:] = alpha
    return Buffer._publish(rgba, like.offset, like.color_space)


def map_straight(src, function):
    """Apply ``function(straight_rgb) -> rgb`` and re-premultiply.

    Alpha is untouched; the result is clipped to [0, 1] like AE's 8/16 bpc
    working space. Empty buffers are returned as is.
    """
    if 0 in src.size:
        return src
    rgb = function(straight_rgb(src))
    return publish_straight(rgb, src.rgba[..., 3:], src, is_opaque(src))


def luma(rgb):
    """Return Rec.709 luma of straight RGB."""
    return rgb @ REC709_LUMA


def color01(code, color_space="srgb"):
    """Interpret authored RGB codes as sRGB, converting for linear projects."""
    color = np.asarray(code[:3], dtype=np.float32) / np.float32(255.0)
    return srgb_to_linear(color) if color_space == "linear" else color
