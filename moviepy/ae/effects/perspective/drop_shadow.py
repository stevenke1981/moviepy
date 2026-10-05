"""Alpha-based drop shadow with animated, resolution-aware parameters."""

import math

import cv2
import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect, Param, sigma_margin
from moviepy.ae.effects.blur.gaussian_blur import filter_pixels
from moviepy.ae.effects.registry import register


@register
class DropShadow(AEEffect):
    """Place a colored soft shadow behind the source's actual alpha.

    ``direction`` is clockwise from up; 90 degrees points right. Softness
    uses sigma = softness / 2, matching GaussianBlur. This is an explicit
    Gaussian shadow model, not Adobe's proprietary implementation.
    ``distance`` and ``softness`` are authored pixels. ``opacity`` is percent,
    color uses RGB codes, and ``shadow_only`` hides the original pixels.

    Notes
    -----
    Shadow coverage is derived from alpha, including semitransparent edges.
    """

    name = "Drop Shadow"
    category = "Perspective"
    PARAMS = (
        Param("color", "color", (0, 0, 0)),
        Param("opacity", "float", 50.0, (0.0, 100.0)),
        Param("direction", "float", 135.0),
        Param("distance", "float", 5.0, (0.0, 10000.0), unit="px"),
        Param("softness", "float", 10.0, (0.0, 1000.0), unit="px"),
        Param("shadow_only", "bool", False),
    )

    @staticmethod
    def _offset(values):
        radians = math.radians(values["direction"] % 360)
        offsets = (
            values["distance"] * math.sin(radians),
            -values["distance"] * math.cos(radians),
        )
        return tuple(0.0 if abs(value) < 1e-10 else value for value in offsets)

    def bounds_expand(self, size, t, context=None, values=None):
        """Include the displaced three-sigma shadow, without losing the source."""
        values = self.values_at(t, context) if values is None else values
        if values["opacity"] == 0:
            return (0, 0, 0, 0)
        dx, dy = self._offset(values)
        radius = sigma_margin(values["softness"] / 2)
        return tuple(math.ceil(radius + max(0, v)) for v in (-dx, -dy, dx, dy))

    def input_margin(self, size, t, context=None, values=None):
        """Read input on the opposite side of the shadow's output growth."""
        left, top, right, bottom = self.bounds_expand(size, t, context, values)
        return right, bottom, left, top

    def render(self, src, t, context=None, values=None):
        """Filter coverage only, then composite the unchanged source over it."""
        values = self.values_at(t, context) if values is None else values
        if 0 in src.size or (values["opacity"] == 0 and not values["shadow_only"]):
            return src
        dx, dy = self._offset(values)
        matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
        alpha = filter_pixels(
            src.rgba[..., 3], values["softness"] / 2, values["softness"] / 2
        )
        alpha = cv2.warpAffine(alpha, matrix, src.size, flags=cv2.INTER_LINEAR)
        alpha = np.clip(alpha * (values["opacity"] / 100), 0, 1)[..., None]
        rgba = np.empty_like(src.rgba)
        rgba[..., :3] = np.asarray(values["color"], np.float32) / 255 * alpha
        rgba[..., 3:] = alpha
        if not values["shadow_only"]:
            rgba *= 1 - src.rgba[..., 3:]
            rgba += src.rgba
        return Buffer._publish(rgba, src.offset, src.color_space)
