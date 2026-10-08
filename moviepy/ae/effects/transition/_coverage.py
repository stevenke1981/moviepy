"""Private helpers shared by the Transition effects (coverage masks on premultiplied RGBA)."""

from functools import lru_cache

import numpy as np

from moviepy.ae.buffer import Buffer


def pixel_centers(size):
    """Return broadcastable pixel-center coordinates ``(x (1,W), y (H,1))`` in layer pixels."""
    width, height = size
    x = np.arange(width, dtype=np.float64)[None, :] + 0.5
    y = np.arange(height, dtype=np.float64)[:, None] + 0.5
    return x, y


def edge_ramp(distance, feather):
    """Return visibility for a signed edge distance in px (positive = visible).

    ``feather`` is the soft-edge width in px. Values below 1 px use a 1 px
    smoothstep centred on the edge, which antialiases hard edges and is exact
    (0 or 1) for pixel-aligned edges.
    """
    width = max(float(feather), 1.0)
    t = np.clip((distance + 0.5 * width) / width, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def apply_visibility(src, visible):
    """Multiply premultiplied RGBA by a ``(H, W)`` visibility in [0, 1]."""
    rgba = src.rgba * visible.astype(np.float32)[..., None]
    return Buffer._publish(
        np.ascontiguousarray(rgba, dtype=np.float32), src.offset, src.color_space
    )


def fully_transparent(src):
    """Return a transparent buffer with the same shape, offset and color space."""
    zeros = np.zeros(src.rgba.shape, dtype=np.float32)
    return Buffer(zeros, src.offset, src.color_space)


@lru_cache(maxsize=8)
def random_table(seed):
    """Return 65536 uniform thresholds from ``numpy.random.default_rng(seed)``."""
    return np.random.default_rng(seed).random(65536)
