"""Immutable temporal easing curves with accurate cubic-bezier inversion."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import Tuple


def finite_number(value, name):
    """Validate a finite real scalar, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def cubic_coordinate(u, first, second):
    """Evaluate a normalized cubic coordinate."""
    return 3 * (1 - u) ** 2 * u * first + 3 * (1 - u) * u**2 * second + u**3


def invert_cubic_x(progress, first, second):
    """Solve a monotone unit cubic by safeguarded Newton and bisection."""
    if progress <= 0:
        return 0.0
    if progress >= 1:
        return 1.0
    low, high, u = 0.0, 1.0, progress
    for _ in range(48):
        residual = cubic_coordinate(u, first, second) - progress
        if abs(residual) < 1e-14:
            return u
        if residual > 0:
            high = u
        else:
            low = u
        derivative = 3 * (
            (1 - u) ** 2 * first
            + 2 * (1 - u) * u * (second - first)
            + u * u * (1 - second)
        )
        candidate = u - residual / derivative if derivative > 1e-12 else low
        u = candidate if low < candidate < high else (low + high) / 2
    return u


@dataclass(frozen=True)
class Ease:
    """Represent a normalized temporal easing curve.

    Parameters
    ----------
    kind : {"linear", "hold", "bezier"}, optional
        Curve family.
    control_points : tuple of float, optional
        CSS cubic controls ``x1, y1, x2, y2``. X controls lie in [0, 1];
        Y controls may overshoot. Factories provide common presets.

    Examples
    --------
    >>> Ease.easy_ease()(0.5)
    0.5
    """

    kind: str = "linear"
    control_points: Tuple[float, ...] = ()

    def __post_init__(self):
        if self.kind not in ("linear", "hold", "bezier"):
            raise ValueError("unknown easing kind")
        controls = tuple(finite_number(v, "control point") for v in self.control_points)
        if self.kind == "bezier":
            if len(controls) != 4 or not all(0 <= controls[i] <= 1 for i in (0, 2)):
                raise ValueError("bezier requires four controls with x in [0, 1]")
        elif controls:
            raise ValueError("only bezier easing accepts control points")
        object.__setattr__(self, "control_points", controls)

    def __call__(self, progress):
        """Evaluate a finite progress, clamping outside the unit interval.

        Parameters
        ----------
        progress : float
            Normalized segment time.

        Returns
        -------
        float
            Eased progress, potentially overshooting for custom Y controls.
        """
        progress = min(1.0, max(0.0, finite_number(progress, "progress")))
        if self.kind == "linear":
            return progress
        if self.kind == "hold":
            return float(progress == 1)
        x1, y1, x2, y2 = self.control_points
        return cubic_coordinate(invert_cubic_x(progress, x1, x2), y1, y2)

    @classmethod
    def linear(cls):
        """Return an identity time mapping."""
        return cls("linear")

    @classmethod
    def hold(cls):
        """Return a step at the segment end."""
        return cls("hold")

    @classmethod
    def bezier(cls, x1, y1, x2, y2):
        """Return a CSS cubic-bezier curve with four finite controls.

        Parameters
        ----------
        x1, y1, x2, y2 : float
            Control coordinates. X lies in [0, 1]; Y may overshoot.

        Returns
        -------
        Ease
            Immutable curve evaluated by inversion of its X coordinate.
        """
        return cls("bezier", (x1, y1, x2, y2))

    @classmethod
    def ease_in(cls):
        """Return CSS ease-in, cubic-bezier(.42, 0, 1, 1)."""
        return cls.bezier(0.42, 0, 1, 1)

    @classmethod
    def ease_out(cls):
        """Return CSS ease-out, cubic-bezier(0, 0, .58, 1)."""
        return cls.bezier(0, 0, 0.58, 1)

    @classmethod
    def ease_in_out(cls):
        """Return CSS ease-in-out, cubic-bezier(.42, 0, .58, 1)."""
        return cls.bezier(0.42, 0, 0.58, 1)

    @classmethod
    def easy_ease(cls):
        """Return zero-speed handles with exactly 33.33 percent influence."""
        return cls.bezier(0.3333, 0, 0.6667, 1)

    def to_dict(self):
        """Return an inert JSON-compatible embedded curve dictionary."""
        return {"kind": self.kind, "control_points": list(self.control_points)}

    @classmethod
    def from_dict(cls, data):
        """Restore a curve from a strictly validated embedded dictionary.

        Parameters
        ----------
        data : dict
            Exact ``kind`` and ``control_points`` fields from ``to_dict``.

        Returns
        -------
        Ease
            Validated immutable curve.
        """
        from moviepy.ae.properties.serialization import validate_payload

        validate_payload(data)
        if type(data) is not dict or set(data) != {"kind", "control_points"}:
            raise ValueError("invalid easing dictionary fields")
        return cls(data["kind"], data["control_points"])
