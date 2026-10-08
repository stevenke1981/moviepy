"""Levels (Color Correction)."""

import numpy as np

from moviepy.ae.effects._pixels import map_straight
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Levels(AEEffect):
    """Remap input black/white points, apply gamma, then set output range.

    ==============  ===============  ==========================
    Parameter       AE panel name    Notes
    ==============  ===============  ==========================
    input_black     Input Black      0..255 code, default 0
    input_white     Input White      0..255 code, default 255
    gamma           Gamma            0.1..10, default 1
    output_black    Output Black     0..255 code, default 0
    output_white    Output White     0..255 code, default 255
    ==============  ===============  ==========================

    Notes
    -----
    Each straight channel ``c`` (codes divided by 255) becomes
    ``x = clip((c - in_black) / (in_white - in_black), 0, 1)``, then
    ``x ** (1 / gamma)``, then ``out_black + (out_white - out_black) * x``.
    Gamma above 1 brightens midtones, as in AE. A degenerate input span
    (white not above black) is widened to one code so the result stays
    finite. The math is applied per channel on the buffer's encoded values.
    """

    name = "Levels"
    category = "Color Correction"
    PARAMS = (
        Param("input_black", "float", 0.0, (0.0, 255.0)),
        Param("input_white", "float", 255.0, (0.0, 255.0)),
        Param("gamma", "float", 1.0, (0.1, 10.0)),
        Param("output_black", "float", 0.0, (0.0, 255.0)),
        Param("output_white", "float", 255.0, (0.0, 255.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Apply the input/gamma/output remap to straight color."""
        values = self.values_at(t, context) if values is None else values
        in_black = values["input_black"] / 255.0
        in_white = values["input_white"] / 255.0
        gamma = values["gamma"]
        out_black = values["output_black"] / 255.0
        out_white = values["output_white"] / 255.0
        if (in_black, in_white, gamma, out_black, out_white) == (
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
        ):
            return src
        span = max(in_white - in_black, 1.0 / 255.0)
        exponent = np.float32(1.0 / gamma)

        def remap(rgb):
            x = np.clip((rgb - np.float32(in_black)) / np.float32(span), 0.0, 1.0)
            x = np.power(x, exponent)
            return np.float32(out_black) + np.float32(out_white - out_black) * x

        return map_straight(src, remap)
