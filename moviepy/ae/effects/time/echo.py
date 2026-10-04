"""Echo (Time): combine the current frame with earlier or later frames."""

from fractions import Fraction

import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.context import _TemporalTime
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


OPERATORS = (
    "add",
    "maximum",
    "minimum",
    "screen",
    "composite_in_back",
    "composite_in_front",
    "blend",
)


@register
class Echo(AEEffect):
    """Overlay time-shifted copies of the layer (AE Echo).

    ==================  ==================  ================================
    Parameter           AE panel name       Notes
    ==================  ==================  ================================
    echo_time           Echo Time (sec)     -10..10, negative = past frames
    number_of_echoes    Number Of Echoes    0..30 (rounded down)
    starting_intensity  Starting Intensity  0..1, weight of the first image
    decay               Decay               0..1, weight ratio per echo
    echo_operator       Echo Operator       see ``OPERATORS``
    ==================  ==================  ================================

    Notes
    -----
    Image ``k`` (``k = 0`` is the current frame) is taken at
    ``t + k * echo_time`` with weight ``starting_intensity * decay ** k``.
    Operators work on premultiplied RGBA: ``add`` sums, ``maximum`` /
    ``minimum`` take extremes, ``screen`` is ``1 - prod(1 - w F)``,
    ``composite_in_back`` puts older echoes behind, ``composite_in_front``
    in front, and ``blend`` averages. Results are clipped to [0, 1]. The
    renderer supplies the other frames through ``temporal_window``.
    """

    name = "Echo"
    category = "Time"
    PARAMS = (
        Param("echo_time", "float", -1.0 / 30.0, (-10.0, 10.0)),
        Param("number_of_echoes", "float", 1.0, (0.0, 30.0)),
        Param("starting_intensity", "float", 1.0, (0.0, 1.0)),
        Param("decay", "float", 1.0, (0.0, 1.0)),
        Param("echo_operator", "enum", "add", choices=OPERATORS),
    )

    def temporal_window(self, t, context=None, values=None):
        """Return the span of the echoes before or after ``t``."""
        values = self.values_at(t, context) if values is None else values
        span = abs(values["echo_time"]) * int(values["number_of_echoes"])
        if span == 0.0:
            return (0.0, 0.0)
        return (span, 0.0) if values["echo_time"] < 0 else (0.0, span)

    def render(self, src, t, context=None, values=None):
        """Without other frames only the weighted current image remains."""
        values = self.values_at(t, context) if values is None else values
        return self._combine([src], values)

    def render_temporal(self, src, t, context, values, source_at):
        """Fetch the echoes through ``source_at`` and combine them."""
        count = int(values["number_of_echoes"])
        frames = [src]
        step = Fraction(values["echo_time"])
        frames += [source_at(_TemporalTime(k * step)) for k in range(1, count + 1)]
        return self._combine(frames, values)

    @staticmethod
    def _combine(frames, values):
        """Weight and merge aligned frames with the selected operator."""
        frames = [frame for frame in frames if 0 not in frame.size] or frames[:1]
        if 0 in frames[0].size:
            return frames[0]
        bounds = _union([frame.bounds for frame in frames])
        stack = np.stack([f.crop(bounds).expand_to(bounds).rgba for f in frames])
        weights = values["starting_intensity"] * values["decay"] ** np.arange(
            len(frames), dtype=np.float32
        )
        weighted = stack * weights.astype(np.float32)[:, None, None, None]
        rgba = _OPERATIONS[values["echo_operator"]](weighted)
        rgba = np.clip(rgba, 0.0, 1.0).astype(np.float32)
        np.minimum(rgba[..., :3], rgba[..., 3:], out=rgba[..., :3])
        return Buffer._publish(rgba, bounds[:2], frames[0].color_space)


def _over(front, back):
    return front + back * (1.0 - front[..., 3:])


def _in_back(weighted):
    result = weighted[0]
    for frame in weighted[1:]:
        result = _over(result, frame)
    return result


def _in_front(weighted):
    result = weighted[0]
    for frame in weighted[1:]:
        result = _over(frame, result)
    return result


_OPERATIONS = {
    "add": lambda w: w.sum(axis=0),
    "maximum": lambda w: w.max(axis=0),
    "minimum": lambda w: w.min(axis=0),
    "screen": lambda w: 1.0 - np.prod(1.0 - w, axis=0),
    "composite_in_back": _in_back,
    "composite_in_front": _in_front,
    "blend": lambda w: w.mean(axis=0),
}


def _union(rectangles):
    lefts, tops, rights, bottoms = zip(*rectangles)
    return (min(lefts), min(tops), max(rights), max(bottoms))
