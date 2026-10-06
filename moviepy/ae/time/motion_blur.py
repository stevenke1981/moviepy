"""Deterministic midpoint integration of animated layer properties."""

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from numbers import Integral

import numpy as np

from moviepy.ae._geometry import validate_flag, validate_pixel_area
from moviepy.ae.buffer import Buffer
from moviepy.ae.context import _exact_time
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.values import finite_real


@dataclass(frozen=True)
class Shutter:
    """Exposure angles and bounded deterministic temporal samples.

    The interval starts at ``t + phase / (360 * fps)`` and lasts
    ``angle / (360 * fps)`` seconds. Defaults are a centered half-frame exposure.
    Fixed sampling uses ``samples_min``; adaptive sampling increases this up to
    ``samples_max`` using transform travel, and uses the maximum for effects,
    masks or unknown source types. Proven static solids need one sample.
    """

    angle: float = 180.0
    phase: float = -90.0
    samples_min: int = 8
    samples_max: int = 64
    adaptive: bool = True

    def __post_init__(self):
        for name in ("angle", "phase"):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if not 0 <= self.angle <= 720:
            raise ValueError("shutter angle must be within 0..720 degrees")
        for name in ("samples_min", "samples_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if not 1 <= value <= 256:
                raise ValueError(f"{name} must be within 1..256")
            object.__setattr__(self, name, int(value))
        if self.samples_max < self.samples_min:
            raise ValueError("samples_max must be at least samples_min")
        validate_flag(self.adaptive, "adaptive")

    @classmethod
    def coerce(cls, value):
        """Accept a Shutter or a mapping; ``samples`` requests a fixed count."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("shutter must be a Shutter or parameter mapping")
        fields = dict(value)
        if "samples" in fields:
            count = fields.pop("samples")
            if "samples_min" in fields or "samples_max" in fields:
                raise ValueError("samples cannot be combined with sample bounds")
            fields.update(samples_min=count, samples_max=count, adaptive=False)
        return cls(**fields)

    def interval(self, context):
        """Return exact endpoints without merging distinct subframe requests."""
        unit = 360 * _exact_time(context.fps)
        start = context._exact_time + _exact_time(self.phase) / unit
        return start, start + _exact_time(self.angle) / unit


def _constant_transform(layer):
    """Only classify properties whose lack of time dependence is proven."""
    for ancestor in layer.parent_chain():
        from moviepy.ae.transform import Transform

        if type(ancestor.transform) is not Transform:
            return False
        for value in ancestor.transform._fields.values():
            if not isinstance(value, Property):
                continue
            if (
                value.keyframes
                or value._callable is not None
                or value._expression is not None
                or value._dimensions is not None
                or value._projection is not None
            ):
                return False
    return True


def _sample_count(layer, context, settings, start, end):
    """Bound work while retaining a conservative path for unknown animation."""
    from moviepy.ae.layers.av import AVLayer
    from moviepy.ae.layers.solid import SolidLayer

    known = type(layer) is SolidLayer or type(layer) is AVLayer
    simple = (
        known
        and not len(layer.masks)
        and not layer.effects.active()
        and layer.track_matte is None
    )
    if (
        simple
        and _constant_transform(layer)
        and layer.in_point <= start
        and end < layer.out_point
    ):
        return 1
    if not settings.adaptive:
        return settings.samples_min
    if not simple:
        return settings.samples_max
    width, height = layer.source_size
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]]).T
    times = [start + (end - start) * Fraction(i, 4) for i in range(5)]
    points = [
        layer.world_matrix(float(t), context.with_time(t)) @ corners for t in times
    ]
    travel = sum(
        float(np.linalg.norm(b[:2] - a[:2], axis=0).max())
        for a, b in zip(points, points[1:])
    )
    count = math.ceil(travel * context.resolution_scale * 2)
    return min(settings.samples_max, max(settings.samples_min, count))


def _sample_context(layer, context, time):
    """Hold AV footage at the center while sampling its animated properties."""
    from moviepy.ae.layers.av import AVLayer

    sample = replace(context, t=time, _exposure_center=context._exact_time)
    if isinstance(layer, AVLayer):
        reference = (
            id(layer),
            layer.source_time(context.t),
            layer.source_time(sample.t),
            context._exact_time,
        )
        sample = replace(
            sample, t=time, _footage_offsets=(*context._footage_offsets, reference)
        )
    return sample


def sample_layer(layer, context, render_one, *, outside=None):
    """Average premultiplied samples; retain missing-window samples as zero.

    ``render_one`` receives a context at each sample and returns a Buffer or
    None. Only one output accumulator is kept; inputs are released as sampling
    proceeds. Float64 accumulation prevents HDR overflow before averaging.
    """
    settings = Shutter.coerce(context.shutter)
    if not layer.motion_blur or settings.angle == 0:
        return render_one(context)
    start, end = settings.interval(context)
    count = _sample_count(layer, context, settings, start, end)
    accumulator, bounds, empty, seen_active = None, None, None, False
    for i in range(count):
        time = start + (end - start) * Fraction(2 * i + 1, 2 * count)
        seen_active = seen_active or layer.in_point <= time < layer.out_point
        result = (
            render_one(_sample_context(layer, context, time))
            if layer.in_point <= time < layer.out_point
            else outside
        )
        if result is None:
            continue
        empty = result
        if 0 in result.size:
            continue
        current = result.bounds
        union = (
            current
            if bounds is None
            else (
                min(bounds[0], current[0]),
                min(bounds[1], current[1]),
                max(bounds[2], current[2]),
                max(bounds[3], current[3]),
            )
        )
        if bounds != union:
            validate_pixel_area(union[2] - union[0], union[3] - union[1], "motion blur")
            expanded = np.zeros(
                (union[3] - union[1], union[2] - union[0], 4), np.float64
            )
            if accumulator is not None:
                expanded[
                    bounds[1] - union[1] : bounds[3] - union[1],
                    bounds[0] - union[0] : bounds[2] - union[0],
                ] = accumulator
            accumulator, bounds = expanded, union
        accumulator[
            current[1] - bounds[1] : current[3] - bounds[1],
            current[0] - bounds[0] : current[2] - bounds[0],
        ] += result.rgba
    if accumulator is None:
        if empty is not None or seen_active:
            return empty
        return Buffer._publish(
            np.zeros((0, 0, 4), np.float32), (0, 0), context.working_space
        )
    accumulator /= count
    rgba = accumulator.astype(np.float32)
    rgba[rgba[..., 3] == 0, :3] = 0
    return Buffer._publish(rgba, bounds[:2], context.working_space)
