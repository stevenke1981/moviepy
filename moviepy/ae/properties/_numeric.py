"""Private numeric interpolation helpers, separate from value codecs."""

import numpy as np

from moviepy.ae.properties.easing import invert_cubic_x
from moviepy.ae.properties.spatial import SpatialSegment


def _velocity(keys, index, incoming, mode):
    key = keys[index]
    if mode == "auto_bezier":
        return _auto_velocity(keys, index)
    explicit = key.in_speed if incoming else key.out_speed
    if key.interp == "continuous_bezier":
        explicit = key.out_speed if key.out_speed is not None else key.in_speed
    if explicit is not None:
        return np.asarray(explicit, dtype=float)
    if mode == "bezier":
        return np.zeros_like(np.asarray(key.value, dtype=float))
    if 0 < index < len(keys) - 1:
        before, after = keys[index - 1], keys[index + 1]
    elif index == 0:
        before, after = keys[0], keys[1]
    else:
        before, after = keys[-2], keys[-1]
    return (np.asarray(after.value) - np.asarray(before.value)) / (
        after.time - before.time
    )


def _auto_velocity(keys, index):
    """Use shared weighted harmonic slopes to preserve monotone data."""
    if index == 0:
        left, right = keys[:2]
        slope = (np.asarray(right.value) - np.asarray(left.value)) / (
            right.time - left.time
        )
        return _cap_auto_velocity(keys, index, slope)
    if index == len(keys) - 1:
        left, right = keys[-2:]
        slope = (np.asarray(right.value) - np.asarray(left.value)) / (
            right.time - left.time
        )
        return _cap_auto_velocity(keys, index, slope)
    before, key, after = keys[index - 1 : index + 2]
    first_dt, second_dt = key.time - before.time, after.time - key.time
    first = (np.asarray(key.value) - np.asarray(before.value)) / first_dt
    second = (np.asarray(after.value) - np.asarray(key.value)) / second_dt
    same_sign = first * second > 0
    first_safe, second_safe = np.where(same_sign, first, 1.0), np.where(
        same_sign, second, 1.0
    )
    first_weight, second_weight = 2 * second_dt + first_dt, second_dt + 2 * first_dt
    harmonic = (first_weight + second_weight) / (
        first_weight / first_safe + second_weight / second_safe
    )
    return _cap_auto_velocity(keys, index, np.where(same_sign, harmonic, 0.0))


def _cap_auto_velocity(keys, index, velocity):
    """Apply both adjacent handle bounds to one shared key velocity."""
    key = keys[index]
    explicit = key.out_speed if key.out_speed is not None else key.in_speed
    if explicit is not None:
        velocity = np.asarray(explicit, dtype=float)
    limit = np.full_like(np.asarray(key.value, dtype=float), np.inf)
    for other_index, influence in (
        (index - 1, key.in_influence),
        (index + 1, key.out_influence),
    ):
        if not 0 <= other_index < len(keys) or influence == 0:
            continue
        other = keys[other_index]
        secant = (np.asarray(other.value) - np.asarray(key.value)) / (
            other.time - key.time
        )
        bound = np.abs(secant) / (influence / 100)
        limit = np.minimum(limit, bound)
        velocity = np.where(velocity * secant >= 0, velocity, 0.0)
    return np.sign(velocity) * np.minimum(np.abs(velocity), limit)


def _bezier_value(start, end, outgoing, incoming, left, right, progress):
    duration = right.time - left.time
    first, second = left.out_influence / 100, right.in_influence / 100
    u = invert_cubic_x(progress, first, 1 - second)
    p1 = start + outgoing * duration * first
    p2 = end - incoming * duration * second
    return (
        (1 - u) ** 3 * start
        + 3 * (1 - u) ** 2 * u * p1
        + 3 * (1 - u) * u * u * p2
        + u**3 * end
    )


def _spatial_fraction(keys, index, progress, path):
    left, right = keys[index : index + 2]
    if path.length == 0 or left.interp == "linear":
        return progress
    outgoing = _velocity(keys, index, False, left.interp)
    incoming = _velocity(keys, index + 1, True, left.interp)
    outgoing = (
        float(outgoing) if outgoing.ndim == 0 else float(np.linalg.norm(outgoing))
    )
    incoming = (
        float(incoming) if incoming.ndim == 0 else float(np.linalg.norm(incoming))
    )
    if outgoing < 0 or incoming < 0:
        raise ValueError("spatial speed must be nonnegative")
    return float(
        _bezier_value(0.0, path.length, outgoing, incoming, left, right, progress)
        / path.length
    )


def evaluate_segment(keys, index, t, spatial):
    """Evaluate one segment with physical speed and distance-aware easing."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _evaluate_segment(keys, index, t, spatial)
    except FloatingPointError as error:
        raise ValueError(
            "keyframe interpolation arithmetic must remain finite"
        ) from error


def _evaluate_segment(keys, index, t, spatial):
    left, right = keys[index : index + 2]
    if left.interp == "hold":
        return left.value
    progress = (t - left.time) / (right.time - left.time)
    ease = left.out_ease or right.in_ease
    if spatial:
        path = SpatialSegment(
            left.value, right.value, left.out_tangent, right.in_tangent
        )
        fraction = (
            ease(progress) if ease else _spatial_fraction(keys, index, progress, path)
        )
        return path.point_at_fraction(fraction)
    if isinstance(left.value, (bool, str)) or isinstance(right.value, (bool, str)):
        raise TypeError("discrete values require hold interpolation")
    start, end = np.asarray(left.value, dtype=float), np.asarray(
        right.value, dtype=float
    )
    if ease:
        fraction = ease(progress)
        value = start * (1 - fraction) + end * fraction
    elif left.interp == "linear":
        value = start * (1 - progress) + end * progress
    else:
        value = _bezier_value(
            start,
            end,
            _velocity(keys, index, False, left.interp),
            _velocity(keys, index + 1, True, left.interp),
            left,
            right,
            progress,
        )
    return float(value) if value.ndim == 0 else tuple(value)
