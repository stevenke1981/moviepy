"""Vibrance (Color Correction)."""

import numpy as np

from moviepy.ae.effects._pixels import luma, map_straight
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Vibrance(AEEffect):
    """Scale color saturation, favoring pixels that are already dull.

    ============  ============  ==========================
    Parameter     AE panel name Notes
    ============  ============  ==========================
    vibrance      Vibrance      -100..100 %, default 0
    saturation    Saturation    -100..100 %, default 0
    ============  ============  ==========================

    Notes
    -----
    Chroma ``C = rgb - luma(rgb)`` (Rec.709 luma, so perceived brightness is
    kept) is multiplied by ``gain = (1 + saturation/100) *
    (1 + vibrance/100 * (1 - s))``, where ``s = (max - min) / max`` is the
    pixel's HSV saturation. Low-saturation pixels therefore change most with
    ``vibrance``; a fully saturated pixel is unaffected by it. Saturation
    -100 gives grayscale and negative vibrance reduces dull colors the most.
    Gains are floored at zero and results are clipped to 0..1. This is a
    documented model, not Adobe's exact curve.
    """

    name = "Vibrance"
    category = "Color Correction"
    PARAMS = (
        Param("vibrance", "float", 0.0, (-100.0, 100.0)),
        Param("saturation", "float", 0.0, (-100.0, 100.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Scale chroma around luma with pixel-dependent vibrance gain."""
        values = self.values_at(t, context) if values is None else values
        vibrance = values["vibrance"] / 100.0
        saturation = values["saturation"] / 100.0
        if vibrance == 0.0 and saturation == 0.0:
            return src

        def scale(rgb):
            high = rgb.max(axis=-1)
            low = rgb.min(axis=-1)
            sat = np.divide(high - low, high, out=np.zeros_like(high), where=high > 0)
            gain = (1.0 + saturation) * (1.0 + vibrance * (1.0 - sat))
            gain = np.maximum(gain, 0.0)[..., None]
            luma_rgb = luma(rgb)[..., None]
            return luma_rgb + (rgb - luma_rgb) * gain

        return map_straight(src, scale)
