"""Venetian Blinds (AE Transition category)."""

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
class VenetianBlinds(AEEffect):
    """Hide the layer through a repeating set of slats.

    =====================  =================  ==========================================
    Parameter              AE panel name      Notes
    =====================  =================  ==========================================
    transition_completion  Transition Compl.  0..100 %, default 0
    direction              Direction          degrees, default 0; 0 gives horizontal
                                              slats, 90 gives vertical slats
    width                  Width              slat period in px, 2..1000, default 20
    feather                Feather            0..1000 px soft edge, default 0
    =====================  =================  ==========================================

    Notes
    -----
    Each period of ``width`` px is hidden from its start for
    ``completion * width`` px, so the hidden fraction equals the completion
    for any layer. The slat period is measured in layer pixels from the layer's
    own origin. The seam at the start of each period is a hard edge; the
    moving edge is antialiased.
    """

    name = "Venetian Blinds"
    category = "Transition"
    PARAMS = (
        Param("transition_completion", "float", 0.0, (0.0, 100.0)),
        Param("direction", "float", 0.0),
        Param("width", "float", 20.0, (2.0, 1000.0), unit="px"),
        Param("feather", "float", 0.0, (0.0, 1000.0), unit="px"),
    )

    def render(self, src, t, context=None, values=None):
        """Hide the leading part of each slat period."""
        values = self.values_at(t, context) if values is None else values
        completion = values["transition_completion"] / 100.0
        if 0 in src.size or completion <= 0.0:
            return src
        if completion >= 1.0:
            return fully_transparent(src)
        width, height = src.size
        angle = math.radians(values["direction"])
        nx, ny = -math.sin(angle), math.cos(angle)
        x, y = pixel_centers(src.size)
        projection = x * nx + y * ny
        low = min(0.0, width * nx) + min(0.0, height * ny)
        period = values["width"]
        phase = np.mod(projection - low, period)
        visible = edge_ramp(phase - completion * period, values["feather"])
        return apply_visibility(src, np.broadcast_to(visible, (height, width)))
