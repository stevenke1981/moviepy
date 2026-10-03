"""Immutable keyframes, temporal interpolation and metadata serialization."""

from dataclasses import dataclass, fields

import numpy as np

from moviepy.ae.properties.easing import Ease, finite_number


def _optional_numeric(value, name):
    if value is None:
        return None
    if np.isscalar(value):
        return finite_number(value, name)
    return tuple(finite_number(item, name) for item in value)


@dataclass(frozen=True)
class Keyframe:
    """Store a time/value pair and temporal or spatial interpolation metadata.

    Parameters
    ----------
    time : float
        Finite time in seconds.
    value : object
        Validated inert scalar, vector or path value.
    in_ease, out_ease : Ease or None, optional
        Explicit normalized curve; outgoing takes precedence for its segment.
    interp : str, optional
        ``linear``, ``hold``, ``bezier``, ``auto_bezier`` or
        ``continuous_bezier``. The left key selects segment interpolation.
    in_influence, out_influence : float, optional
        Temporal handle lengths as percentages in [0, 100], default 33.33.
    in_speed, out_speed : float or tuple or None, optional
        Signed value units per second. Explicit bezier defaults to zero;
        Auto defaults use shared shape-preserving harmonic velocities;
        continuous defaults derive shared centered velocities.
    in_tangent, out_tangent : tuple or None, optional
        Spatial handles as offsets from the endpoint, in position units.
    roving : bool, optional
        Resolve interior spatial key timing by adjacent curve lengths.

    Notes
    -----
    Auto uses shared weighted harmonic slopes with bounded segment controls;
    continuous uses centered velocities and allows user-defined overshoot.
    These are documented approximations of proprietary tangent heuristics.
    """

    time: float
    value: object
    in_ease: object = None
    out_ease: object = None
    interp: str = "linear"
    in_influence: float = 33.33
    out_influence: float = 33.33
    in_speed: object = None
    out_speed: object = None
    in_tangent: object = None
    out_tangent: object = None
    roving: bool = False

    def __post_init__(self):
        from moviepy.ae.properties.values import normalize_value

        object.__setattr__(self, "time", finite_number(self.time, "time"))
        object.__setattr__(self, "value", normalize_value(self.value))
        if self.interp not in (
            "linear",
            "hold",
            "bezier",
            "auto_bezier",
            "continuous_bezier",
        ):
            raise ValueError("unknown keyframe interpolation")
        if not isinstance(self.roving, bool):
            raise TypeError("roving must be bool")
        for name in ("in_ease", "out_ease"):
            if getattr(self, name) is not None and not isinstance(
                getattr(self, name), Ease
            ):
                raise TypeError(f"{name} must be Ease or None")
        for name in ("in_influence", "out_influence"):
            value = finite_number(getattr(self, name), name)
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100]")
            object.__setattr__(self, name, value)
        for name in ("in_speed", "out_speed", "in_tangent", "out_tangent"):
            value = _optional_numeric(getattr(self, name), name)
            if name.endswith("tangent") and value is not None and np.isscalar(value):
                raise TypeError(f"{name} must be a vector")
            object.__setattr__(self, name, value)

    def to_dict(self):
        """Return every metadata field as an embedded JSON dictionary."""
        from moviepy.ae.properties.values import encode_value

        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["value"] = encode_value(self.value)
        for name in ("in_ease", "out_ease"):
            result[name] = (
                getattr(self, name).to_dict() if getattr(self, name) else None
            )
        for name in ("in_speed", "out_speed", "in_tangent", "out_tangent"):
            if isinstance(result[name], tuple):
                result[name] = list(result[name])
        return result

    @classmethod
    def from_dict(cls, data):
        """Restore strictly validated inert keyframe data, without code imports.

        Parameters
        ----------
        data : dict
            Exact fields produced by ``to_dict`` with inert value encoding.

        Returns
        -------
        Keyframe
            Validated immutable keyframe with detached vector metadata.
        """
        from moviepy.ae.properties.serialization import validate_payload
        from moviepy.ae.properties.values import decode_value

        validate_payload(data)
        if type(data) is not dict or set(data) != {field.name for field in fields(cls)}:
            raise ValueError("invalid keyframe dictionary fields")
        result = dict(data)
        result["value"] = decode_value(result["value"])
        for name in ("in_ease", "out_ease"):
            if result[name] is not None:
                result[name] = Ease.from_dict(result[name])
        return cls(**result)


def normalize_keyframes(items):
    """Validate and sort keys or (time, value[, outgoing Ease]) shorthand.

    Parameters
    ----------
    items : iterable
        Keyframe instances or tuples. Duplicate/nonfinite times, inconsistent
        numeric dimensions and roving endpoints are rejected.

    Returns
    -------
    tuple of Keyframe
        Detached immutable keys sorted by distinct times. Roving times are
        resolved only during spatial evaluation or by resolve_roving_times.
    """
    keys = []
    for item in items:
        if isinstance(item, Keyframe):
            keys.append(item)
        elif isinstance(item, (tuple, list)) and len(item) in (2, 3):
            keys.append(
                Keyframe(item[0], item[1], out_ease=item[2] if len(item) == 3 else None)
            )
        else:
            raise TypeError("keys must be Keyframe or (time, value[, Ease])")
    keys.sort(key=lambda key: key.time)
    if not keys:
        raise ValueError("keyframes must not be empty")
    if any(left.time == right.time for left, right in zip(keys, keys[1:])):
        raise ValueError("keyframe times must be distinct")
    for left, right in zip(keys, keys[1:]):
        finite_number(right.time - left.time, "segment duration")
    if keys[0].roving or keys[-1].roving:
        raise ValueError("first and last keys cannot rove")
    _validate_numeric_shapes(keys)
    return tuple(keys)


def _validate_numeric_shapes(keys):
    numeric = [
        key
        for key in keys
        if isinstance(key.value, (float, int, tuple))
        and not isinstance(key.value, bool)
    ]
    if numeric and len(numeric) == len(keys):
        shape = np.asarray(numeric[0].value).shape
        for key in keys:
            if np.asarray(key.value).shape != shape:
                raise ValueError("keyframe numeric value dimensions must match")
            for name in ("in_speed", "out_speed"):
                speed = getattr(key, name)
                if (
                    speed is not None
                    and not np.isscalar(speed)
                    and np.asarray(speed).shape != shape
                ):
                    raise ValueError("speed dimensions must match the value")
            for name in ("in_tangent", "out_tangent"):
                tangent = getattr(key, name)
                if tangent is not None and np.asarray(tangent).shape != shape:
                    raise ValueError("tangent dimensions must match the value")


def evaluate_keyframes(keys, t, spatial=False):
    """Evaluate sorted keys, clamping endpoints and optionally using arc length.

    Parameters
    ----------
    keys : tuple of Keyframe
        Normalized keys.
    t : float
        Finite sample time in seconds.
    spatial : bool, optional
        Interpret vector values as 2D/3D positions and resolve roving timing.

    Returns
    -------
    object
        Scalar, immutable numeric tuple, or a held discrete value.
    """
    from moviepy.ae.properties._numeric import evaluate_segment
    from moviepy.ae.properties.spatial import resolve_roving_times

    t = finite_number(t, "time")
    if not keys:
        raise ValueError("keyframes must not be empty")
    if any(key.roving for key in keys):
        if not spatial:
            raise ValueError("roving keyframes require spatial=True")
        keys = resolve_roving_times(keys)
    if t <= keys[0].time:
        return keys[0].value
    if t >= keys[-1].time:
        return keys[-1].value
    index = int(np.searchsorted([key.time for key in keys], t, side="right")) - 1
    return evaluate_segment(keys, index, t, spatial)
