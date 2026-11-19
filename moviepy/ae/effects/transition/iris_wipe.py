"""Iris Wipe (AE Transition category)."""

import math

import numpy as np

from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register
from moviepy.ae.effects.transition._coverage import (
    apply_visibility,
    edge_ramp,
    fully_transparent,
    pixel_centers,
)


@register
class IrisWipe(AEEffect):
    """Close a regular polygonal iris around a centre point.

    =====================  =================  ==========================================
    Parameter              AE panel name      Notes
    =====================  =================  ==========================================
    transition_completion  Transition Compl.  0..100 %, default 0
    iris_points            Iris Points        6..32 polygon sides, default 6
    iris_center_x          Iris Center X      normalized to layer width, default 0.5
    iris_center_y          Iris Center Y      normalized to layer height, default 0.5
    feather                Feather            0..1000 px soft edge, default 0
    =====================  =================  ==========================================

    Notes
    -----
    The iris is a regular polygon with one vertex at the top. Its apothem
    shrinks linearly from the farthest layer corner (completion 0) to zero
    (completion 100), so every pixel is visible at completion 0 and hidden at
    100. ``iris_points`` is rounded to the nearest integer. AE's iris also
    offers an inner-radius option that is not implemented here.
    """

    name = "Iris Wipe"
    category = "Transition"
    PARAMS = (
        Param("transition_completion", "float", 0.0, (0.0, 100.0)),
        Param("iris_points", "float", 6.0, (6.0, 32.0)),
        Param("iris_center_x", "float", 0.5),
        Param("iris_center_y", "float", 0.5),
        Param("feather", "float", 0.0, (0.0, 1000.0), unit="px"),
    )

    def render(self, src, t, context=None, values=None):
        """Keep the pixels inside the shrinking polygon."""
        values = self.values_at(t, context) if values is None else values
        completion = values["transition_completion"] / 100.0
        if 0 in src.size or completion <= 0.0:
            return src
        if completion >= 1.0:
            return fully_transparent(src)
        width, height = src.size
        sides = int(round(values["iris_points"]))
        cx = values["iris_center_x"] * width
        cy = values["iris_center_y"] * height
        corners = [(0.0, 0.0), (width, 0.0), (0.0, height), (width, height)]
        farthest = max(math.hypot(px - cx, py - cy) for px, py in corners)
        apothem = farthest * (1.0 - completion)
        x, y = pixel_centers(src.size)
        rx, ry = x - cx, y - cy
        radius = np.hypot(rx, ry)
        step = 2.0 * math.pi / sides
        normal0 = -math.pi / 2.0 + math.pi / sides
        k = np.mod(np.arctan2(ry, rx) - normal0, step)
        alpha = np.minimum(k, step - k)  # angle from the nearest edge normal
        distance = apothem - radius * np.cos(alpha)  # signed, positive inside
        visible = edge_ramp(distance, values["feather"])
        return apply_visibility(src, np.broadcast_to(visible, (height, width)))
