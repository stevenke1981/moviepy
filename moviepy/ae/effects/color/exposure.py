"""Exposure (Color Correction)."""

import numpy as np

from moviepy.ae.color import linear_to_srgb, srgb_to_linear
from moviepy.ae.effects._pixels import map_straight
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Exposure(AEEffect):
    """Scale light by stops, add an offset and apply gamma in linear light.

    ==================  ================  ==========================
    Parameter           AE panel name     Notes
    ==================  ================  ==========================
    exposure            Exposure          stops -20..20, default 0
    offset              Offset            -2..2 linear, default 0
    gamma_correction    Gamma Correction  0.1..10, default 1
    ==================  ================  ==========================

    Notes
    -----
    Straight color is decoded to linear light when the buffer is sRGB
    (``moviepy.ae.color``), then ``v = max(c * 2**exposure + offset, 0)`` and
    ``v ** (1 / gamma_correction)`` are applied, and the result is encoded
    back to sRGB. Linear buffers are processed directly. Values above
    display white are clipped to the 0..1 working range, like other effects.
    """

    name = "Exposure"
    category = "Color Correction"
    PARAMS = (
        Param("exposure", "float", 0.0, (-20.0, 20.0)),
        Param("offset", "float", 0.0, (-2.0, 2.0)),
        Param("gamma_correction", "float", 1.0, (0.1, 10.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Apply stops, offset and gamma in linear light."""
        values = self.values_at(t, context) if values is None else values
        gain = np.float32(2.0 ** values["exposure"])
        offset = np.float32(values["offset"])
        exponent = np.float32(1.0 / values["gamma_correction"])
        if gain == 1.0 and offset == 0.0 and exponent == 1.0:
            return src
        srgb = src.color_space == "srgb"

        def expose(rgb):
            linear = srgb_to_linear(rgb) if srgb else rgb
            light = np.maximum(linear * gain + offset, 0.0)
            light = np.power(light, exponent)
            return linear_to_srgb(light) if srgb else light

        return map_straight(src, expose)
