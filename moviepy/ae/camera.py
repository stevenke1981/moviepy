"""Subject-aware 2D framing built from the existing Property and Transform APIs."""

import math
from dataclasses import dataclass

from moviepy.ae._geometry import to_property, validate_pixel_size
from moviepy.ae.properties.easing import Ease
from moviepy.ae.properties.keyframe import Keyframe
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.values import finite_real
from moviepy.ae.transform import Transform


def _rectangle(value, name, extent):
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        raise ValueError(f"{name} requires left, top, right, bottom pixel centers")
    left, top, right, bottom = (finite_real(v, name) for v in value)
    if not (0 <= left < right <= extent[0] - 1):
        raise ValueError(f"{name} horizontal bounds must lie inside the source")
    if not (0 <= top < bottom <= extent[1] - 1):
        raise ValueError(f"{name} vertical bounds must lie inside the source")
    return (left, top, right, bottom)


@dataclass(frozen=True)
class FramingSample:
    """Resolved source center, uniform scale, sampled crop and clamp evidence."""

    center: tuple
    scale: float
    crop: tuple
    clamped: bool


class _FramingValue(Property):
    """Forward the renderer's expression bindings into both camera controls."""

    def __init__(self, framing, field):
        self.framing, self.field = framing, field
        # Keep the normal dynamic-property marker for source inspection/copies.
        super().__init__(self.base_value_at, value_type="vec2")

    def base_value_at(self, t):
        """Evaluate without expression context, as for other Property sources."""
        return self.value_at(t)

    def value_at(self, t, **bindings):
        """Evaluate camera controls with the active layer/context bindings."""
        sample = self.framing.sample(t, **bindings)
        return sample.center if self.field == "center" else (sample.scale * 100,) * 2


class CameraFraming:
    """Keep an opaque rectangular source covering a 2D output during pan/zoom.

    ``focus`` and ``pan`` are source-pixel ``vec2`` Properties; ``zoom`` is a
    scalar Property relative to cover scale. Values, keyframes and expressions
    use the existing AE evaluator. Positive pan X/Y moves the viewing window
    right/down, so the image moves left/up. No tracking or new view is inferred.

    ``safe_bounds`` is the allowed source rectangle in pixel-center coordinates,
    defaulting to the complete source. ``overscan`` (at least 1) reserves extra
    source coverage. ``subject_bounds`` optionally keeps a manually supplied
    source box inside the output's fractional ``safe_margin``. Impossible box,
    margin and cover constraints raise rather than reveal a border or crop the
    protected subject. Zoom and pan are clamped at every evaluation, including
    overshooting easing curves; ``sample`` exposes the actual crop and clamp.

    The generated ordinary Transform uses positive-weight linear resampling.
    Assign it to an unrotated, unparented footage layer of ``source_size`` in a
    composition of ``size``. Other geometry/effects can invalidate the coverage
    guarantee, as can transparent pixels already present in the source.
    """

    def __init__(
        self,
        source_size,
        size,
        *,
        focus=None,
        pan=(0, 0),
        zoom=1.0,
        overscan=1.04,
        safe_bounds=None,
        subject_bounds=None,
        safe_margin=0.08,
    ):
        self.source_size = validate_pixel_size(source_size)
        self.size = validate_pixel_size(size)
        if min(*self.source_size, *self.size) < 2:
            raise ValueError("camera framing dimensions must be at least two pixels")
        self.overscan = finite_real(overscan, "overscan")
        self.safe_margin = finite_real(safe_margin, "safe_margin")
        if self.overscan < 1 or not 0 <= self.safe_margin < 0.5:
            raise ValueError("overscan must be >= 1 and safe_margin within [0, 0.5)")
        full = (0, 0, self.source_size[0] - 1, self.source_size[1] - 1)
        self.safe_bounds = _rectangle(
            full if safe_bounds is None else safe_bounds,
            "safe_bounds",
            self.source_size,
        )
        self.subject_bounds = (
            None
            if subject_bounds is None
            else _rectangle(subject_bounds, "subject_bounds", self.source_size)
        )
        box = self.subject_bounds or self.safe_bounds
        if focus is None:
            focus = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self.focus, self.pan, self.zoom = focus, pan, zoom

    def __setattr__(self, name, value):
        if name in ("focus", "pan", "zoom"):
            value = to_property(value, name, "float" if name == "zoom" else "vec2")
        object.__setattr__(self, name, value)

    def sample(self, t, **bindings):
        """Resolve the constraints at layer-local time without reading any pixels."""
        focus = self.focus.value_at(t, **bindings)
        pan = self.pan.value_at(t, **bindings)
        zoom = self.zoom.value_at(t, **bindings)
        if zoom <= 0:
            raise ValueError("camera zoom must remain positive")
        requested = tuple(a + b for a, b in zip(focus, pan))
        if not all(math.isfinite(v) for v in requested):
            raise ValueError("camera focus plus pan must remain finite")
        left, top, right, bottom = self.safe_bounds
        # The core sampler forms float32 inverse coordinates. Reserve a few
        # source-coordinate ULPs so rounding cannot read transparent pixels
        # just beyond an otherwise mathematically exact coverage boundary.
        guard = min(
            (right - left) / 1000,
            (bottom - top) / 1000,
            max(1e-5, max(self.source_size) * 2**-21),
        )
        left, top, right, bottom = (
            left + guard,
            top + guard,
            right - guard,
            bottom - guard,
        )
        spans = (self.size[0] - 1, self.size[1] - 1)
        cover = max(spans[0] / (right - left), spans[1] / (bottom - top))
        minimum = cover * self.overscan
        scale = minimum * max(1.0, zoom)
        requested_scale = scale
        if not math.isfinite(scale):
            raise ValueError("camera scale exceeds finite geometry")
        if self.subject_bounds is not None:
            box = self.subject_bounds
            maximum = min(
                spans[axis] * (1 - 2 * self.safe_margin) / (box[axis + 2] - box[axis])
                for axis in (0, 1)
            )
            if maximum < minimum:
                raise ValueError("subject cannot fit: reduce safe_margin or overscan")
            scale = min(scale, maximum)
        half = tuple(span / (2 * scale) for span in spans)
        limits = [[left + half[0], right - half[0]], [top + half[1], bottom - half[1]]]
        if self.subject_bounds is not None:
            box = self.subject_bounds
            for axis in (0, 1):
                margin = spans[axis] * self.safe_margin / scale
                limits[axis][0] = max(
                    limits[axis][0], box[axis + 2] - half[axis] + margin
                )
                limits[axis][1] = min(limits[axis][1], box[axis] + half[axis] - margin)
        if any(low > high + 1e-9 for low, high in limits):
            raise ValueError("subject safe margin conflicts with source safe_bounds")
        center = tuple(
            min(high, max(low, value)) for value, (low, high) in zip(requested, limits)
        )
        crop = (
            center[0] - half[0],
            center[1] - half[1],
            center[0] + half[0],
            center[1] + half[1],
        )
        clamped = center != requested or scale != requested_scale or zoom < 1
        return FramingSample(center, scale, crop, clamped)

    def to_transform(self, *, opacity=100.0):
        """Build a live core Transform; edit this rig's Properties to animate it.

        Like other callable Properties, the generated projection is not JSON
        serializable. The rig owns focus, pan and zoom; changing the generated
        anchor/scale/position independently bypasses the framing constraints.
        Layer time, expression bindings, masks and the normal renderer remain
        those of the existing Transform pipeline.
        """
        return Transform(
            anchor_point=_FramingValue(self, "center"),
            position=tuple((value - 1) / 2 for value in self.size),
            scale=_FramingValue(self, "scale"),
            interpolation="linear",
            opacity=opacity,
        )


def pan_zoom(
    source_size,
    size,
    *,
    duration,
    direction="right",
    distance=80.0,
    zoom=(1.0, 1.08),
    easing=None,
    **framing,
):
    """Return a keyframed CameraFraming preset for left/right/up/down or a push.

    ``distance`` is nonnegative source pixels travelled by the viewing window.
    ``direction='none'`` permits a zoom-only push. Keyframes hold before zero
    and after ``duration``; pass the last frame's time to reach the endpoint on
    the final visible frame. The default curve is the core ``Ease.easy_ease``.
    """
    directions = {
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, -1),
        "down": (0, 1),
        "none": (0, 0),
    }
    if direction not in directions:
        raise ValueError("direction must be left, right, up, down or none")
    duration, distance = finite_real(duration, "duration"), finite_real(
        distance, "distance"
    )
    if duration <= 0 or distance < 0:
        raise ValueError("duration must be positive and distance nonnegative")
    if not isinstance(zoom, (tuple, list)) or len(zoom) != 2:
        raise ValueError("zoom requires start and end factors")
    zoom = tuple(finite_real(value, "zoom") for value in zoom)
    if min(zoom) <= 0:
        raise ValueError("zoom must remain positive")
    easing = Ease.easy_ease() if easing is None else easing
    if not isinstance(easing, Ease):
        raise TypeError("easing must be an Ease")
    pan = tuple(value * distance for value in directions[direction])
    return CameraFraming(
        source_size,
        size,
        pan=[Keyframe(0, (0, 0), out_ease=easing), Keyframe(duration, pan)],
        zoom=[Keyframe(0, zoom[0], out_ease=easing), Keyframe(duration, zoom[1])],
        **framing,
    )
