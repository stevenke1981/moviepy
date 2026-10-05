"""Tint (Color Correction)."""

import numpy as np

from moviepy.ae.effects._pixels import color01, luma, map_straight
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Tint(AEEffect):
    """Map luminance onto a black-to-white color ramp.

    ===============  ===============  ===========================
    Parameter        AE panel name    Notes
    ===============  ===============  ===========================
    map_black_to     Map Black To     RGB codes, default black
    map_white_to     Map White To     RGB codes, default white
    amount_to_tint   Amount to Tint   0..100 %, default 100
    ===============  ===============  ===========================

    Notes
    -----
    ``L`` is the Rec.709 luma of the straight color; the tinted color is
    ``black + (white - black) * L`` mixed with the input by the amount. With
    the defaults the result is a grayscale image.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae import Buffer
    >>> green = Buffer.from_uint8_rgb(np.array([[[0, 255, 0]]], dtype=np.uint8))
    >>> Tint().process(green, 0.0).to_uint8_rgb().tolist()
    [[[182, 182, 182]]]
    """

    name = "Tint"
    category = "Color Correction"
    PARAMS = (
        Param("map_black_to", "color", (0.0, 0.0, 0.0)),
        Param("map_white_to", "color", (255.0, 255.0, 255.0)),
        Param("amount_to_tint", "float", 100.0, (0.0, 100.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Replace straight color by the tinted luminance ramp."""
        values = self.values_at(t, context) if values is None else values
        amount = np.float32(values["amount_to_tint"] / 100.0)
        if amount == 0.0:
            return src
        black = color01(values["map_black_to"], src.color_space)
        white = color01(values["map_white_to"], src.color_space)

        def tint(rgb):
            mapped = black + (white - black) * luma(rgb)[..., None]
            return rgb + (mapped - rgb) * amount

        return map_straight(src, tint)
