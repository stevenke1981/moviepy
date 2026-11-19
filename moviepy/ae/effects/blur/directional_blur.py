"""Directional Blur (Blur & Sharpen)."""

import math

import cv2
import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


MAX_SAMPLES = 64


@register
class DirectionalBlur(AEEffect):
    """Smear premultiplied pixels along a straight line through each pixel.

    ==============  ==================  ==========================================
    Parameter       AE panel name       Notes
    ==============  ==================  ==========================================
    direction       Direction           degrees, 0 = horizontal, 90 = downward
    blur_length     Blur Length         0..1000 px, default 0 (identity)
    ==============  ==================  ==========================================

    Notes
    -----
    The blur is a box average of ``blur_length`` pixels centered on each
    pixel, sampled with bilinear interpolation at no more than 64 positions
    (longer lengths therefore show sample spacing as faint ghosting). Adobe's
    exact sample distribution is not published, and this version is centered
    rather than one-sided. Premultiplied RGBA is averaged, so transparent
    pixels never add color and ``rgb <= alpha`` is preserved. The layer grows
    by half the blur length on every side so smeared content can spread out.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae import Buffer
    >>> row = np.zeros((1, 5, 4), dtype=np.float32); row[0, 2] = 1
    >>> out = DirectionalBlur(direction=0, blur_length=2).process(Buffer(row), 0.0)
    >>> out.size
    (9, 5)
    >>> [round(float(v), 2) for v in out.rgba[2, :, 3]]
    [0.0, 0.0, 0.0, 0.33, 0.33, 0.33, 0.0, 0.0, 0.0]
    """

    name = "Directional Blur"
    category = "Blur & Sharpen"
    PARAMS = (
        Param("direction", "float", 0.0, (-3600.0, 3600.0)),
        Param("blur_length", "float", 0.0, (0.0, 1000.0), unit="px"),
    )

    def bounds_expand(self, size, t, context=None, values=None):
        """Grow by half the blur length (plus one pixel) on every side."""
        values = self.values_at(t, context) if values is None else values
        length = values["blur_length"]
        margin = int(math.ceil(length / 2.0)) + 1 if length > 0 else 0
        return (margin,) * 4

    def render(self, src, t, context=None, values=None):
        """Average premultiplied samples along the blur line."""
        values = self.values_at(t, context) if values is None else values
        length = values["blur_length"]
        if 0 in src.size or length <= 0:
            return src
        radians = math.radians(values["direction"])
        step_x, step_y = math.cos(radians), math.sin(radians)
        count = min(MAX_SAMPLES, int(math.ceil(length)) + 1)
        height, width = src.size[1], src.size[0]
        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        rgba = np.ascontiguousarray(src.rgba, dtype=np.float32)
        accumulated = np.zeros((height, width, 4), dtype=np.float32)
        for offset in np.linspace(-length / 2.0, length / 2.0, count):
            map_x = (xs + np.float32(offset * step_x)).astype(np.float32)
            map_y = (ys + np.float32(offset * step_y)).astype(np.float32)
            accumulated += cv2.remap(
                rgba,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        accumulated /= np.float32(count)
        np.clip(accumulated, 0.0, 1.0, out=accumulated)
        np.minimum(accumulated[..., :3], accumulated[..., 3:], out=accumulated[..., :3])
        return Buffer._publish(accumulated, src.offset, src.color_space)
