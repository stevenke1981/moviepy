"""Brightness & Contrast (Color Correction)."""

from moviepy.ae.effects._pixels import map_straight
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class BrightnessContrast(AEEffect):
    """Shift brightness and scale contrast around mid gray.

    ==========  =============  =====================================
    Parameter   AE panel name  Notes
    ==========  =============  =====================================
    brightness  Brightness     -150..150 (8-bit code offset)
    contrast    Contrast       -100..100 (slope ``1 + contrast/100``)
    ==========  =============  =====================================

    Notes
    -----
    Straight color ``c`` becomes ``(c - 0.5) * (1 + contrast/100) + 0.5 +
    brightness/255``, clipped to [0, 1]. This is AE's *Use Legacy* linear
    model; the non-legacy curve is not published by Adobe.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae import Buffer
    >>> gray = Buffer.from_uint8_rgb(np.full((1, 1, 3), 100, dtype=np.uint8))
    >>> effect = BrightnessContrast(brightness=50)
    >>> effect.process(gray, 0.0).to_uint8_rgb().tolist()
    [[[150, 150, 150]]]
    """

    name = "Brightness & Contrast"
    category = "Color Correction"
    PARAMS = (
        Param("brightness", "float", 0.0, (-150.0, 150.0)),
        Param("contrast", "float", 0.0, (-100.0, 100.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Apply the linear brightness/contrast model to straight color."""
        values = self.values_at(t, context) if values is None else values
        slope = 1.0 + values["contrast"] / 100.0
        offset = values["brightness"] / 255.0
        if slope == 1.0 and offset == 0.0:
            return src
        return map_straight(src, lambda rgb: (rgb - 0.5) * slope + 0.5 + offset)
