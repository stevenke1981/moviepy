"""Radial Blur (Blur & Sharpen)."""

import math

import cv2
import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


TYPES = ("spin", "zoom")
MAX_SAMPLES = 32


@register
class RadialBlur(AEEffect):
    """Blur premultiplied pixels by rotating or scaling them about a center.

    ==========  ===========  ==============================================
    Parameter   AE panel     Notes
    ==========  ===========  ==============================================
    amount      Amount       0..100, default 0 (identity)
    center      Center       normalized layer position (0..1), default 0.5
    type        Type         ``spin`` (rotational) or ``zoom`` (radial)
    ==========  ===========  ==============================================

    Notes
    -----
    ``spin`` averages rotations spanning ``amount`` degrees centered on the
    original position; ``zoom`` averages scalings spanning ``amount`` percent
    of the distance to the center, centered on the original position. Each
    output pixel is the mean of up to 32 bilinear samples, scaled with
    ``amount``. Adobe's internal spin and zoom scales are not published, so
    the strength mapping is an approximation. Content rotated or scaled from
    outside the layer is transparent, not edge-repeated.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae import Buffer
    >>> flat = np.ones((3, 3, 4), dtype=np.float32)
    >>> out = RadialBlur(amount=0).process(Buffer(flat), 0.0)
    >>> bool(np.array_equal(out.rgba, flat))
    True
    """

    name = "Radial Blur"
    category = "Blur & Sharpen"
    PARAMS = (
        Param("amount", "float", 0.0, (0.0, 100.0)),
        Param("center", "vec2", (0.5, 0.5)),
        Param("type", "enum", TYPES[0], choices=TYPES),
    )

    def render(self, src, t, context=None, values=None):
        """Average premultiplied samples taken around the center."""
        values = self.values_at(t, context) if values is None else values
        amount = float(values["amount"])
        if 0 in src.size or amount <= 0:
            return src
        height, width = src.size[1], src.size[0]
        center_x = min(1.0, max(0.0, float(values["center"][0]))) * (width - 1)
        center_y = min(1.0, max(0.0, float(values["center"][1]))) * (height - 1)
        count = min(MAX_SAMPLES, 2 + int(amount // 3))
        fractions = np.linspace(-1.0, 1.0, count)
        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
        )
        dx, dy = xs - center_x, ys - center_y
        rgba = np.ascontiguousarray(src.rgba, dtype=np.float32)
        accumulated = np.zeros((height, width, 4), dtype=np.float32)
        for fraction in fractions:
            if values["type"] == "spin":
                radians = math.radians(fraction * amount / 2.0)
                cos, sin = math.cos(radians), math.sin(radians)
                map_x = center_x + dx * cos - dy * sin
                map_y = center_y + dx * sin + dy * cos
            else:
                scale = 1.0 + fraction * amount / 200.0
                map_x = center_x + dx * scale
                map_y = center_y + dy * scale
            accumulated += cv2.remap(
                rgba,
                map_x.astype(np.float32),
                map_y.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        accumulated /= np.float32(count)
        np.clip(accumulated, 0.0, 1.0, out=accumulated)
        np.minimum(accumulated[..., :3], accumulated[..., 3:], out=accumulated[..., :3])
        return Buffer._publish(accumulated, src.offset, src.color_space)
