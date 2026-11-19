"""Vignette (Stylize)."""

import numpy as np

from moviepy.ae.effects._pixels import map_straight
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


@register
class Vignette(AEEffect):
    """Darken or lighten the edges of the frame around a movable center.

    ===========  ===============  ===========================
    Parameter    AE panel name    Notes
    ===========  ===============  ===========================
    amount       Amount           -100..100 %, default 0
    midpoint     Midpoint         0..100 %, default 50
    roundness    Roundness        -100..100 %, default 0
    feather      Feather          0..100 %, default 50
    ===========  ===============  ===========================

    Notes
    -----
    The center is the middle of the buffer and distances are measured in
    buffer pixels. The normalized distance ``d`` is 1 at the frame corner
    when roundness is 0 (an ellipse matching the frame), blends toward a
    circle (positive roundness) or a rectangle (negative roundness). Pixels
    inside ``midpoint`` are untouched; ``feather`` sets the smoothstep width
    beyond it. Negative amounts multiply straight color by ``1 - |amount| * s``
    (darken); positive amounts move it toward white by ``amount * s`` (lighten).
    Amount 0 is an exact identity. Adobe's falloff curve and its
    color-space handling are not reproduced exactly.
    """

    name = "Vignette"
    category = "Stylize"
    PARAMS = (
        Param("amount", "float", 0.0, (-100.0, 100.0)),
        Param("midpoint", "float", 50.0, (0.0, 100.0)),
        Param("roundness", "float", 0.0, (-100.0, 100.0)),
        Param("feather", "float", 50.0, (0.0, 100.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Apply the radial falloff to straight color; alpha is preserved."""
        values = self.values_at(t, context) if values is None else values
        amount = values["amount"] / 100.0
        if amount == 0.0 or 0 in src.size:
            return src
        height, width = src.size[1], src.size[0]
        ys = (np.arange(height, dtype=np.float32) + 0.5 - height / 2) / (height / 2)
        xs = (np.arange(width, dtype=np.float32) + 0.5 - width / 2) / (width / 2)
        nx, ny = np.meshgrid(xs, ys)
        d_ellipse = np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0)
        d_rect = np.maximum(np.abs(nx), np.abs(ny))
        radius = np.sqrt((nx * width / 2) ** 2 + (ny * height / 2) ** 2) / (
            min(width, height) / 2
        )
        roundness = values["roundness"] / 100.0
        if roundness >= 0:
            d = (1.0 - roundness) * d_ellipse + roundness * radius
        else:
            d = (1.0 + roundness) * d_ellipse - roundness * d_rect
        midpoint = values["midpoint"] / 100.0
        width_feather = max(values["feather"] / 100.0, 1e-6)
        u = np.clip((d - midpoint) / width_feather, 0.0, 1.0)
        s = (u * u * (3.0 - 2.0 * u)).astype(np.float32)[..., None]
        amount = np.float32(amount)

        def vignette(rgb):
            if amount < 0:
                return rgb * (1.0 + amount * s)
            return rgb + (1.0 - rgb) * (amount * s)

        return map_straight(src, vignette)
