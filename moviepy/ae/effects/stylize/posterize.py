"""Posterize (Stylize)."""

import numpy as np

from moviepy.ae.effects._pixels import map_straight
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Posterize(AEEffect):
    """Quantize straight color to a fixed number of levels per channel.

    ===========  ===============  ===========================
    Parameter    AE panel name    Notes
    ===========  ===============  ===========================
    level        Level            2..255, default 255
    ===========  ===============  ===========================

    Notes
    -----
    Each straight RGB channel ``c`` in 0..1 becomes
    ``round(c * (N - 1)) / (N - 1)`` with ``N`` the rounded level. Alpha is
    untouched. The default level 255 returns the input unchanged, which is
    exact for 8-bit codes; AE's exact quantization curve is not documented,
    so intermediate levels are a uniform-step approximation.
    """

    name = "Posterize"
    category = "Stylize"
    PARAMS = (Param("level", "float", 255.0, (2.0, 255.0)),)

    def render(self, src, t, context=None, values=None):
        """Snap each straight channel to the nearest of N evenly spaced levels."""
        values = self.values_at(t, context) if values is None else values
        levels = int(round(values["level"]))
        if levels >= 255 or 0 in src.size:
            return src
        steps = np.float32(levels - 1)

        def posterize(rgb):
            return np.round(rgb * steps) / steps

        return map_straight(src, posterize)
