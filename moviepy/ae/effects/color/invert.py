"""Invert (Color Correction; AE lists it under Channel)."""

import numpy as np

from moviepy.ae.effects._pixels import is_opaque, publish_straight, straight_rgb
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


CHANNELS = ("rgb", "red", "green", "blue", "alpha")
_CHANNELS = {
    "rgb": slice(0, 3),
    "red": slice(0, 1),
    "green": slice(1, 2),
    "blue": slice(2, 3),
}


@register
class Invert(AEEffect):
    """Invert straight color channels or the alpha channel.

    =========  =============  =====================================
    Parameter  AE panel name  Notes
    =========  =============  =====================================
    channel    Channel        ``rgb`` (default), ``red``, ``green``,
                              ``blue`` or ``alpha``
    =========  =============  =====================================

    Notes
    -----
    Color channels are inverted on straight (unpremultiplied) values, so
    transparent pixels stay transparent. Inverting ``alpha`` keeps the
    straight color and replaces alpha by ``1 - alpha``. Blend With Original
    is the base-class compositing option.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae import Buffer
    >>> red = Buffer.from_uint8_rgb(np.array([[[255, 0, 0]]], dtype=np.uint8))
    >>> Invert().process(red, 0.0).to_uint8_rgb().tolist()
    [[[0, 255, 255]]]
    """

    name = "Invert"
    category = "Color Correction"
    PARAMS = (Param("channel", "enum", "rgb", choices=CHANNELS),)

    def render(self, src, t, context=None, values=None):
        """Invert the selected straight channels."""
        values = self.values_at(t, context) if values is None else values
        if 0 in src.size:
            return src
        rgb = straight_rgb(src)
        alpha = src.rgba[..., 3:]
        if values["channel"] == "alpha":
            return publish_straight(rgb, np.float32(1.0) - alpha, src)
        channels = _CHANNELS[values["channel"]]
        np.subtract(np.float32(1.0), rgb[..., channels], out=rgb[..., channels])
        return publish_straight(rgb, alpha, src, is_opaque(src))
