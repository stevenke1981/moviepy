"""Linear Wipe (AE Transition category)."""

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
class LinearWipe(AEEffect):
    """Wipe the layer away along a straight edge sweeping across it.

    =====================  =================  ==========================================
    Parameter              AE panel name      Notes
    =====================  =================  ==========================================
    transition_completion  Transition Compl.  0..100 %, default 0 (layer visible)
    wipe_angle             Wipe Angle         degrees, default 90; 0 wipes left to
                                              right, 90 wipes top to bottom
    feather                Feather            0..1000 px soft edge, default 0
    =====================  =================  ==========================================

    Notes
    -----
    The sweep runs over the layer's own pixel extent projected onto the wipe
    direction, so 50 % hides exactly half of a rectangle at angle 0 or 90.
    Feather 0 uses a 1 px antialiased edge. Completion 0 returns the input
    unchanged; 100 returns a fully transparent buffer of the same shape.
    """

    name = "Linear Wipe"
    category = "Transition"
    PARAMS = (
        Param("transition_completion", "float", 0.0, (0.0, 100.0)),
        Param("wipe_angle", "float", 90.0),
        Param("feather", "float", 0.0, (0.0, 1000.0), unit="px"),
    )

    def render(self, src, t, context=None, values=None):
        """Hide the part of the layer behind the moving wipe edge."""
        values = self.values_at(t, context) if values is None else values
        completion = values["transition_completion"] / 100.0
        if 0 in src.size or completion <= 0.0:
            return src
        if completion >= 1.0:
            return fully_transparent(src)
        width, height = src.size
        angle = math.radians(values["wipe_angle"])
        dx, dy = math.cos(angle), math.sin(angle)
        x, y = pixel_centers(src.size)
        projection = x * dx + y * dy
        low = min(0.0, width * dx) + min(0.0, height * dy)
        high = max(0.0, width * dx) + max(0.0, height * dy)
        edge = low + completion * (high - low)
        visible = edge_ramp(projection - edge, values["feather"])
        return apply_visibility(src, np.broadcast_to(visible, (height, width)))
