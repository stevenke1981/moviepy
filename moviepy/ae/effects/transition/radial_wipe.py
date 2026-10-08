"""Radial Wipe (AE Transition category)."""

import numpy as np

from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register
from moviepy.ae.effects.transition._coverage import (
    apply_visibility,
    edge_ramp,
    fully_transparent,
    pixel_centers,
)


WIPES = ("clockwise", "counterclockwise", "both")


@register
class RadialWipe(AEEffect):
    """Sweep a radial edge around a centre point to hide the layer.

    =====================  =================  ==========================================
    Parameter              AE panel name      Notes
    =====================  =================  ==========================================
    transition_completion  Transition Compl.  0..100 %, default 0
    start_angle            Start Angle        degrees, default 0 (12 o'clock)
    wipe_center_x          Wipe Center X      normalized to layer width, default 0.5
    wipe_center_y          Wipe Center Y      normalized to layer height, default 0.5
    wipe                   Wipe               ``clockwise`` (default),
                                              ``counterclockwise`` or ``both``
    feather                Feather            0..1000 px soft edge, default 0
    =====================  =================  ==========================================

    Notes
    -----
    Angles are measured clockwise on screen from ``start_angle``. ``clockwise``
    hides the first ``360 * completion`` degrees; ``counterclockwise`` hides
    the same arc in the opposite direction; ``both`` hides ``180 * completion``
    degrees on each side of the start. The edge antialias width is the arc
    length in px at each pixel. The start seam (the far side of the sweep) is
    a hard edge. Completion 0 returns the input; 100 is fully transparent.
    """

    name = "Radial Wipe"
    category = "Transition"
    PARAMS = (
        Param("transition_completion", "float", 0.0, (0.0, 100.0)),
        Param("start_angle", "float", 0.0),
        Param("wipe_center_x", "float", 0.5),
        Param("wipe_center_y", "float", 0.5),
        Param("wipe", "enum", "clockwise", choices=WIPES),
        Param("feather", "float", 0.0, (0.0, 1000.0), unit="px"),
    )

    def render(self, src, t, context=None, values=None):
        """Hide the angular sector swept from the start angle."""
        values = self.values_at(t, context) if values is None else values
        completion = values["transition_completion"] / 100.0
        if 0 in src.size or completion <= 0.0:
            return src
        if completion >= 1.0:
            return fully_transparent(src)
        width, height = src.size
        x, y = pixel_centers(src.size)
        cx = values["wipe_center_x"] * width
        cy = values["wipe_center_y"] * height
        rx, ry = x - cx, y - cy
        radius = np.maximum(np.hypot(rx, ry), 1.0)
        theta = np.degrees(np.arctan2(rx, -ry))  # 0 = up, clockwise positive
        start = values["start_angle"]
        clockwise = np.mod(theta - start, 360.0)
        if values["wipe"] == "clockwise":
            offset = clockwise - 360.0 * completion
        elif values["wipe"] == "counterclockwise":
            offset = np.mod(start - theta, 360.0) - 360.0 * completion
        else:
            side = np.minimum(clockwise, 360.0 - clockwise)
            offset = side - 180.0 * completion
        visible = edge_ramp(np.radians(offset) * radius, values["feather"])
        return apply_visibility(src, np.broadcast_to(visible, (height, width)))
