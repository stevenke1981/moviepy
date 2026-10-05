"""Thresholded alpha-aware light bloom."""

import numpy as np

from moviepy.ae.buffer import Buffer, unpremultiply
from moviepy.ae.effects.base import AEEffect, Param, sigma_margin
from moviepy.ae.effects.blur.gaussian_blur import filter_pixels
from moviepy.ae.effects.registry import register


@register
class Glow(AEEffect):
    """Add a soft colored halo from visible pixels above a luma threshold.

    ``radius`` is authored pixels, ``threshold`` is a percent, ``intensity``
    controls halo coverage, and ``color`` is RGB codes. This bounded Gaussian
    bloom is not a physically based lighting simulation or Adobe's algorithm.

    Notes
    -----
    Transparent RGB cannot seed a halo; intensity zero is an exact identity.
    """

    name = "Glow"
    category = "Stylize"
    PARAMS = (
        Param("radius", "float", 20.0, (0.0, 1000.0), unit="px"),
        Param("threshold", "float", 60.0, (0.0, 100.0)),
        Param("intensity", "float", 1.0, (0.0, 10.0)),
        Param("color", "color", (255, 220, 150)),
    )

    def bounds_expand(self, size, t, context=None, values=None):
        """Grow by the Gaussian support unless the glow is disabled."""
        values = self.values_at(t, context) if values is None else values
        margin = sigma_margin(values["radius"] / 2) if values["intensity"] else 0
        return (margin,) * 4

    def render(self, src, t, context=None, values=None):
        """Screen the halo into visible pixels and extend transparent coverage."""
        values = self.values_at(t, context) if values is None else values
        if 0 in src.size or values["intensity"] == 0:
            return src
        rgb = unpremultiply(src.rgba)[..., :3]
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        threshold = values["threshold"] / 100
        gain = np.clip((luma - threshold) / max(1 - threshold, 1e-6), 0, 1)
        coverage = filter_pixels(
            gain * src.rgba[..., 3], values["radius"] / 2, values["radius"] / 2
        )
        alpha = -np.expm1(-values["intensity"] * np.maximum(coverage, 0))
        alpha = alpha[..., None]
        rgba = np.array(src.rgba, copy=True)
        halo = np.asarray(values["color"], np.float32) / 255 * alpha
        # Premultiplied Screen also illuminates an opaque dark backdrop.
        # Preserve values already above display white instead of dimming HDR.
        rgba[..., :3] += halo * np.maximum(1 - src.rgba[..., :3], 0)
        rgba[..., 3:] += alpha * (1 - src.rgba[..., 3:])
        return Buffer._publish(rgba, src.offset, src.color_space)
