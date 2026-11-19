"""Fill (Generate)."""

import numpy as np

from moviepy.ae.effects._pixels import color01, map_straight
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Fill(AEEffect):
    """Replace the layer's color with a solid color, keeping its alpha.

    =========  =============  ==============================
    Parameter  AE panel name  Notes
    =========  =============  ==============================
    color      Color          RGB codes, default red
    opacity    Opacity        0..100 %, default 100
    =========  =============  ==============================

    Notes
    -----
    Limit the fill to a region with the effect ``mask`` (AE's Fill Mask).

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae import Buffer
    >>> gray = Buffer.from_uint8_rgb(np.full((1, 1, 3), 100, dtype=np.uint8))
    >>> Fill(color=(0, 0, 255), opacity=50).process(gray, 0.0).to_uint8_rgb().tolist()
    [[[50, 50, 178]]]
    """

    name = "Fill"
    category = "Generate"
    PARAMS = (
        Param("color", "color", (255.0, 0.0, 0.0)),
        Param("opacity", "float", 100.0, (0.0, 100.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Mix the straight color towards the fill color."""
        values = self.values_at(t, context) if values is None else values
        amount = np.float32(values["opacity"] / 100.0)
        if amount == 0.0:
            return src
        color = color01(values["color"], src.color_space)
        return map_straight(src, lambda rgb: rgb + (color - rgb) * amount)
