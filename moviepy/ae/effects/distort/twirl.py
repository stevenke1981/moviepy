"""Rotational twirl around a center point (Distort)."""

import numpy as np
from scipy.ndimage import map_coordinates

from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Twirl(AEEffect):
    """Rotate pixels inside a circle, with rotation fading toward its rim.

    ==================  ==================  ===========================
    Parameter           AE panel name       Notes
    ==================  ==================  ===========================
    angle               Angle               degrees at the center, default 0
    twirl_radius        Twirl Radius        0..100 % of half the short side
    center_x            Center X            0..1 of width, default 0.5
    center_y            Center Y            0..1 of height, default 0.5
    ==================  ==================  ===========================

    Notes
    -----
    The rotation at radius ``r`` is ``angle * (1 - r / R) ** 2`` and is zero
    outside the circle ``R``. The quadratic fall-off is a calibrated choice;
    Adobe's exact profile is not published. The center defaults to the
    buffer center. ``angle`` 0 or a zero radius is an exact identity. Sampling
    is bilinear and premultiplied; the circle lies within the layer, so the
    output has the same bounds as the input.
    """

    name = "Twirl"
    category = "Distort"
    PARAMS = (
        Param("angle", "float", 0.0),
        Param("twirl_radius", "float", 75.0, (0.0, 100.0)),
        Param("center_x", "float", 0.5, (0.0, 1.0)),
        Param("center_y", "float", 0.5, (0.0, 1.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Rotate each destination pixel's offset back into the source."""
        values = self.values_at(t, context) if values is None else values
        if 0 in src.size or values["angle"] == 0 or values["twirl_radius"] == 0:
            return src
        height, width = src.rgba.shape[:2]
        cx = values["center_x"] * width
        cy = values["center_y"] * height
        radius = values["twirl_radius"] / 100.0 * min(width, height) / 2.0

        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float64) + 0.5,
            np.arange(height, dtype=np.float64) + 0.5,
        )
        dx = xs - cx
        dy = ys - cy
        dist = np.hypot(dx, dy)
        fade = np.clip(1.0 - dist / radius, 0.0, 1.0)
        theta = np.where(dist < radius, np.radians(values["angle"]) * fade**2, 0.0)
        cos, sin = np.cos(theta), np.sin(theta)
        # Inverse map: rotate the destination offset by -theta.
        sx = cx + cos * dx + sin * dy
        sy = cy - sin * dx + cos * dy
        rgba = _sample(src.rgba, sx, sy)
        return Buffer._publish(rgba, src.offset, src.color_space)


def _sample(rgba, xs, ys):
    """Bilinear, zero-filled premultiplied lookup at pixel-center coordinates."""
    # Round away float noise (e.g. cos(90 deg) = 6e-17): scipy "constant"
    # mode zeroes points a hair outside the array, even at the true edge.
    coords = np.round(np.stack((ys - 0.5, xs - 0.5)), 9)
    out = np.empty(rgba.shape, dtype=np.float32)
    for channel in range(4):
        out[..., channel] = map_coordinates(
            rgba[..., channel], coords, order=1, mode="constant", cval=0.0
        )
    return out
