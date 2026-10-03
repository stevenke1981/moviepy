"""Cubic spatial paths evaluated by distance, with interior roving timing."""

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from moviepy.ae.properties.easing import finite_number


def _vector(value, name, dimension=None):
    try:
        vector = tuple(finite_number(v, name) for v in value)
    except TypeError as error:
        raise TypeError(f"{name} must be a numeric vector") from error
    if len(vector) not in (2, 3) or (dimension and len(vector) != dimension):
        raise ValueError(f"{name} must have matching two or three dimensions")
    return vector


@lru_cache(maxsize=512)
def _arc_table(points):
    controls = np.asarray(points, dtype=float)
    parameters = np.linspace(0.0, 1.0, 4097)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    mid = (parameters[1:] + parameters[:-1]) / 2
    half = np.diff(parameters) / 2
    u = mid[:, None] + half[:, None] * nodes
    try:
        with np.errstate(over="raise", invalid="raise"):
            derivative = 3 * (
                (1 - u)[..., None] ** 2 * (controls[1] - controls[0])
                + 2 * ((1 - u) * u)[..., None] * (controls[2] - controls[1])
                + u[..., None] ** 2 * (controls[3] - controls[2])
            )
            speed = np.hypot.reduce(derivative, axis=2)
            distances = np.cumsum(half * (speed @ weights))
    except FloatingPointError as error:
        raise ValueError(
            "spatial curve arithmetic and arc length must be finite"
        ) from error
    if not np.all(np.isfinite(distances)):
        raise ValueError("spatial arc length must be finite")
    lengths = np.concatenate(([0.0], distances))
    parameters.setflags(write=False)
    lengths.setflags(write=False)
    return parameters, lengths


@dataclass(frozen=True)
class SpatialSegment:
    """Represent a cubic position segment with endpoint-relative tangents.

    Parameters
    ----------
    start, end : sequence of float
        Matching 2D or 3D positions.
    out_tangent, in_tangent : sequence of float or None, optional
        Offsets from start and end. Missing tangents yield a straight path.

    Notes
    -----
    Arc length uses eight-point Gaussian quadrature in 4096 intervals, then
    monotone inversion. Constant-distance samples therefore follow physical
    distance rather than the nonlinear cubic parameter. Table data is cached.
    """

    start: tuple
    end: tuple
    out_tangent: object = None
    in_tangent: object = None
    _points: tuple = field(init=False, repr=False)

    def __post_init__(self):
        start = _vector(self.start, "start")
        end = _vector(self.end, "end", len(start))
        outgoing = (
            _vector(self.out_tangent, "out_tangent", len(start))
            if self.out_tangent is not None
            else (0.0,) * len(start)
        )
        incoming = (
            _vector(self.in_tangent, "in_tangent", len(start))
            if self.in_tangent is not None
            else (0.0,) * len(start)
        )
        points = _finite_controls(start, end, outgoing, incoming)
        _arc_table(points)
        for name, value in (
            ("start", start),
            ("end", end),
            ("out_tangent", outgoing),
            ("in_tangent", incoming),
            ("_points", points),
        ):
            object.__setattr__(self, name, value)

    @property
    def length(self):
        """Return the physical curve length in position units."""
        return float(_arc_table(self._points)[1][-1])

    def point(self, progress):
        """Return a tuple position at a clamped finite cubic parameter.

        Parameters
        ----------
        progress : float
            Cubic parameter, clamped to [0, 1].

        Returns
        -------
        tuple of float
            Position in the endpoint coordinate system.
        """
        u = min(1.0, max(0.0, finite_number(progress, "progress")))
        if u == 0:
            return self.start
        if u == 1:
            return self.end
        controls = np.asarray(self._points)
        weights = np.array(
            [(1 - u) ** 3, 3 * (1 - u) ** 2 * u, 3 * (1 - u) * u * u, u**3]
        )
        try:
            with np.errstate(over="raise", invalid="raise"):
                result = controls[0] + weights @ (controls - controls[0])
        except FloatingPointError as error:
            raise ValueError("spatial point arithmetic must remain finite") from error
        if not np.all(np.isfinite(result)):
            raise ValueError("spatial point must remain finite")
        return tuple(result)

    def point_at_fraction(self, fraction):
        """Return a position at a clamped fraction of total arc length.

        Parameters
        ----------
        fraction : float
            Distance divided by total curve length, clamped to [0, 1].

        Returns
        -------
        tuple of float
            Position determined by inversion of the arc-length table.
        """
        fraction = min(1.0, max(0.0, finite_number(fraction, "fraction")))
        parameters, lengths = _arc_table(self._points)
        if lengths[-1] == 0:
            return self.start
        return self.point(float(np.interp(fraction * lengths[-1], lengths, parameters)))


def resolve_roving_times(keys):
    """Return keys with interior roving times proportional to path distances.

    Parameters
    ----------
    keys : sequence of Keyframe
        Sorted keys with fixed first and last positions. Each consecutive run
        of roving keys is bracketed by fixed keys. Hold segments and zero-length
        segments inside partly nonzero runs are rejected to preserve distinct
        key times. Entirely zero-length runs receive uniform spacing.

    Returns
    -------
    tuple of Keyframe
        Immutable replacements. Zero-length runs receive uniform spacing.
    """
    keys = tuple(keys)
    if not keys:
        raise ValueError("keyframes must not be empty")
    if keys[0].roving or keys[-1].roving:
        raise ValueError("first and last keys cannot rove")
    result = list(keys)
    anchors = [i for i, key in enumerate(keys) if not key.roving]
    for start, end in zip(anchors, anchors[1:]):
        if end == start + 1:
            continue
        distances = _run_distances(keys, start, end)
        if any(distance == 0 for distance in distances) and any(
            distance > 0 for distance in distances
        ):
            raise ValueError(
                "zero-length segments in a partially nonzero roving run are invalid"
            )
        fractions = _roving_fractions(distances)
        duration = finite_number(keys[end].time - keys[start].time, "roving duration")
        for index in range(start + 1, end):
            result[index] = replace(
                keys[index],
                time=keys[start].time + duration * float(fractions[index - start - 1]),
            )
    if any(left.time >= right.time for left, right in zip(result, result[1:])):
        raise ValueError("resolved roving times must remain distinct and increasing")
    return tuple(result)


def _roving_fractions(distances):
    try:
        with np.errstate(over="raise", invalid="raise"):
            cumulative = np.cumsum(distances)
    except FloatingPointError as error:
        raise ValueError("total roving arc length must be finite") from error
    if not np.all(np.isfinite(cumulative)):
        raise ValueError("total roving arc length must be finite")
    if cumulative[-1]:
        return cumulative / cumulative[-1]
    return np.arange(1, len(distances) + 1) / len(distances)


def _run_distances(keys, start, end):
    distances = []
    for left, right in zip(keys[start:end], keys[start + 1 : end + 1]):
        ease = left.out_ease or right.in_ease
        if left.interp == "hold" or (ease and ease.kind == "hold"):
            raise ValueError("roving cannot use hold interpolation")
        distances.append(
            SpatialSegment(
                left.value, right.value, left.out_tangent, right.in_tangent
            ).length
        )
    return distances


def _finite_controls(start, end, outgoing, incoming):
    try:
        with np.errstate(over="raise", invalid="raise"):
            points = (
                start,
                tuple(np.add(start, outgoing)),
                tuple(np.add(end, incoming)),
                end,
            )
    except FloatingPointError as error:
        raise ValueError("spatial control coordinates must be finite") from error
    if not np.all(np.isfinite(points)):
        raise ValueError("spatial control coordinates must be finite")
    return points
