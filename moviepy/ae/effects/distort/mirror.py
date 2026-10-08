"""Reflect one half-plane across a line (Distort)."""

import math

import numpy as np
from scipy.ndimage import map_coordinates

from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Mirror(AEEffect):
    """Replace the pixels on one side of a line by their mirror image.

    ==================  ==================  ===========================
    Parameter           AE panel name       Notes
    ==================  ==================  ===========================
    reflection_center_x Reflection Center X 0..1 of width, default 0.5
    reflection_center_y Reflection Center Y 0..1 of height, default 0.5
    reflection_angle    Reflection Angle    degrees, default 0
    ==================  ==================  ===========================

    Notes
    -----
    The line passes through the center with direction ``reflection_angle``
    (0 is horizontal). Pixels on the side the normal points to, that is
    ``(-sin a, cos a)`` (downward at 0), take the color mirrored from the
    other side; the other half is kept. With the defaults the bottom half is
    replaced by the reflected top half, so the result is not an identity,
    which differs from the AE default effect. The output bounds are unchanged;
    for a line that is not axis-aligned or off-center, a reflected sample can
    fall outside the layer and is transparent there.
    """

    name = "Mirror"
    category = "Distort"
    PARAMS = (
        Param("reflection_center_x", "float", 0.5, (0.0, 1.0)),
        Param("reflection_center_y", "float", 0.5, (0.0, 1.0)),
        Param("reflection_angle", "float", 0.0),
    )

    def render(self, src, t, context=None, values=None):
        """Sample mirrored positions for pixels on the reflected side."""
        values = self.values_at(t, context) if values is None else values
        if 0 in src.size:
            return src
        height, width = src.rgba.shape[:2]
        cx = values["reflection_center_x"] * width
        cy = values["reflection_center_y"] * height
        theta = math.radians(values["reflection_angle"])
        nx, ny = -math.sin(theta), math.cos(theta)

        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float64) + 0.5,
            np.arange(height, dtype=np.float64) + 0.5,
        )
        side = (xs - cx) * nx + (ys - cy) * ny
        reflected = side > 0
        mirror_x = xs - 2.0 * side * nx
        mirror_y = ys - 2.0 * side * ny
        sampled = _sample(src.rgba, mirror_x, mirror_y)
        rgba = np.where(reflected[..., None], sampled, src.rgba)
        return Buffer._publish(rgba.astype(np.float32), src.offset, src.color_space)


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
