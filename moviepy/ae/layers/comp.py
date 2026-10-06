"""Pre-composition layer: a nested ``Composition`` used as a layer source."""

import math
from dataclasses import replace

from moviepy.ae.layers.base import Layer


_FRAME_EPSILON = 1e-9


class CompLayer(Layer):
    """Use a ``Composition`` as the source of a layer (AE pre-compose).

    Parameters
    ----------
    comp : Composition
        Nested composition. It keeps its own size, fps and duration.
    name : str, optional
        Layer name; defaults to the composition name.
    ``**kwargs``
        Every ``Layer`` keyword.

    Notes
    -----
    Source time is snapped to the child's own frame grid, so a 24 fps child
    inside a 30 fps parent repeats frames exactly as AE does. Times outside
    the child duration hold the first or last frame. Negative stretch plays
    the child backwards from its end.

    With ``collapse_transformations`` the child is rendered without clipping
    to its frame, so content outside the child's bounds stays visible after the
    layer transform, matching AE's collapsed pre-comps. Child blend modes are
    still resolved inside the child; they do not interact with parent layers.
    """

    def __init__(self, comp, name=None, **kwargs):
        from moviepy.ae.composition import Composition

        if not isinstance(comp, Composition):
            raise TypeError("comp must be a Composition")
        self.comp = comp
        super().__init__(comp.name if name is None else name, **kwargs)

    def default_out_point(self):
        """Return the composition end in parent time."""
        return self.start_time + self.comp.duration * abs(self.stretch) / 100.0

    def source_time(self, t):
        """Map parent time onto child time, reversing from its end if needed."""
        time = super().source_time(t)
        return time if self.stretch >= 0 else self.comp.duration + time

    @property
    def source_size(self):
        """Return the nested composition size."""
        return self.comp.size

    def child_time(self, t):
        """Snap a child time to the child's frame grid and clamp it."""
        fps = self.comp.fps
        frame = math.floor(float(t) * fps + _FRAME_EPSILON)
        last = max(0, math.ceil(self.comp.duration * fps - _FRAME_EPSILON) - 1)
        return min(max(frame, 0), last) / fps

    def source_buffer(self, t, context=None):
        """Render the nested composition at child time ``t``."""
        if context is not None:
            context = replace(
                context,
                t=context._exact_time,
                shutter=self.comp.context.shutter,
                _footage_offsets=(),
                _exposure_center=None,
            )
        return self.comp.render_buffer(
            self.child_time(t), context, clip=not self.collapse_transformations
        )
