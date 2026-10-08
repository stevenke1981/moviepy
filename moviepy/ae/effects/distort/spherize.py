"""Spherical bulge or pinch inside a circle (Distort)."""

import numpy as np
from scipy.ndimage import map_coordinates

from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Spherize(AEEffect):
    """Wrap the pixels inside a circle around a sphere, or pinch them.

    ==================  ==================  ===========================
    Parameter           AE panel name       Notes
    ==================  ==================  ===========================
    radius              Radius              0..2500 px, default 0
    amount              Amount              -100..100 %, default 0
    ==================  ==================  ===========================

    Notes
    -----
    With ``u = r / radius`` the source radius is ``u + a * (2/pi * asin(u) - u)``
    where ``a = amount / 100``. Positive amounts bulge (sphere projection at
    ``a = 1``); negative amounts pinch. The map is continuous at the rim, and
    pixels outside the circle are unchanged. The center is the buffer center.
    This is an analytic sphere model, not Adobe's proprietary algorithm.
    Radius 0 or amount 0 is an exact identity.
    """

    name = "Spherize"
    category = "Distort"
    PARAMS = (
        Param("radius", "float", 0.0, (0.0, 2500.0), unit="px"),
        Param("amount", "float", 0.0, (-100.0, 100.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Scale each destination offset by its radial source-to-dest ratio."""
        values = self.values_at(t, context) if values is None else values
        if 0 in src.size or values["radius"] == 0 or values["amount"] == 0:
            return src
        height, width = src.rgba.shape[:2]
        cx, cy = width / 2.0, height / 2.0
        radius = values["radius"]
        amount = values["amount"] / 100.0

        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float64) + 0.5,
            np.arange(height, dtype=np.float64) + 0.5,
        )
        dx = xs - cx
        dy = ys - cy
        u = np.hypot(dx, dy) / radius
        inside = u < 1.0
        uc = np.clip(u, 0.0, 1.0)
        target = uc + amount * (2.0 / np.pi * np.arcsin(uc) - uc)
        safe = np.where(uc > 0, uc, 1.0)
        # Limit of target / u at the center is 1 + a * (2/pi - 1).
        ratio = np.where(uc > 0, target / safe, 1.0 + amount * (2.0 / np.pi - 1.0))
        ratio = np.where(inside, ratio, 1.0)
        rgba = _sample(src.rgba, cx + dx * ratio, cy + dy * ratio)
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
