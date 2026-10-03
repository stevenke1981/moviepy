"""Finite immutable values and inert JSON codecs for animated properties."""

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Tuple

import numpy as np

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]
VALUE_TYPES = frozenset(("float", "vec2", "vec3", "color", "bool", "enum", "path"))
MAX_POINTS = 4096
_REAL_TYPES = frozenset(
    (
        int,
        float,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    )
)


def finite_real(value, name="value"):
    """Return a finite float, excluding booleans and arbitrary objects."""
    if type(value) not in _REAL_TYPES:
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sequence(value):
    if isinstance(value, np.ndarray):
        if type(value) is not np.ndarray:
            raise TypeError("array subclasses are not inert values")
        if value.dtype.kind not in "fiu" or value.ndim != 1:
            raise TypeError("values must be one-dimensional numeric arrays")
        if value.size > MAX_POINTS * 6:
            raise ValueError("numeric array exceeds component budget")
        return value.tolist()
    if type(value) not in (list, tuple):
        raise TypeError("value must be a list or tuple")
    return value


def _vector(value, lengths):
    items = _sequence(value)
    if len(items) not in lengths:
        raise ValueError(f"value must have length in {lengths}")
    return tuple(finite_real(item, "component") for item in items)


def _points(value, name):
    if isinstance(value, np.ndarray):
        if type(value) is not np.ndarray:
            raise TypeError("array subclasses are not inert values")
        if value.dtype.kind not in "fiu" or value.ndim != 2:
            raise TypeError(f"{name} must be numeric point pairs")
        if value.shape[0] > MAX_POINTS or value.shape[1] != 2:
            raise ValueError("path array exceeds point or dimension budget")
        value = value.tolist()
    if type(value) not in (list, tuple):
        raise TypeError(f"{name} must be a sequence")
    if len(value) > MAX_POINTS:
        raise ValueError("path exceeds point budget")
    return tuple(_vector(point, (2,)) for point in value)


@dataclass(frozen=True)
class PathValue:
    """Store detached two-dimensional vertices and relative Bezier tangents.

    Parameters
    ----------
    vertices : sequence of pairs
        Finite vertex coordinates, up to 4096 points.
    in_tangents, out_tangents : sequence of pairs, optional
        Relative tangents. Empty sequences become zero tangents.
    closed : bool, optional
        Whether the path joins its last vertex to its first.
    """

    vertices: Tuple[Vec2, ...]
    in_tangents: Tuple[Vec2, ...] = ()
    out_tangents: Tuple[Vec2, ...] = ()
    closed: bool = True

    def __post_init__(self):
        vertices = _points(self.vertices, "vertices")
        if not isinstance(self.closed, bool):
            raise TypeError("closed must be bool")
        object.__setattr__(self, "vertices", vertices)
        for name in ("in_tangents", "out_tangents"):
            points = _points(getattr(self, name), name)
            if not points:
                points = ((0.0, 0.0),) * len(vertices)
            if len(points) != len(vertices):
                raise ValueError("path tangent count must match vertices")
            object.__setattr__(self, name, points)


def infer_value_type(value):
    """Infer an explicit property kind without executing user callbacks."""
    if type(value) in (bool, np.bool_):
        return "bool"
    if type(value) in _REAL_TYPES:
        return "float"
    if type(value) is PathValue:
        return "path"
    if isinstance(value, (Enum, str)):
        return "enum"
    if isinstance(value, (list, tuple, np.ndarray)):
        items = _sequence(value)
        if len(items) in (2, 3):
            return f"vec{len(items)}"
    raise TypeError("cannot infer value_type; specify a supported kind")


def normalize_value(value, value_type=None):
    """Normalize a property value to finite detached immutable Python data."""
    if (
        value_type is None
        and type(value) in _REAL_TYPES
        and isinstance(value, Integral)
    ):
        finite_real(value)
        return int(value)
    if value_type is None and isinstance(value, (tuple, list, np.ndarray)):
        items = _sequence(value)
        if len(items) <= MAX_POINTS * 6:
            return tuple(finite_real(item, "component") for item in items)
        raise ValueError("numeric value exceeds component budget")
    kind = infer_value_type(value) if value_type is None else value_type
    if not isinstance(kind, str) or kind not in VALUE_TYPES:
        raise ValueError("unsupported value_type")
    if kind == "float":
        return finite_real(value)
    if kind in ("vec2", "vec3", "color"):
        lengths = (3, 4) if kind == "color" else (int(kind[-1]),)
        return _vector(value, lengths)
    if kind == "bool":
        if type(value) not in (bool, np.bool_):
            raise TypeError("bool property requires bool values")
        return bool(value)
    if kind == "enum":
        return _enum_token(value)
    if type(value) is not PathValue:
        raise TypeError("path property requires PathValue")
    return PathValue(
        value.vertices, value.in_tangents, value.out_tangents, value.closed
    )


def _enum_token(value):
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        if not value or len(value) > 256:
            raise ValueError("enum token must contain 1 to 256 characters")
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        if abs(value) > 2**53:
            raise ValueError("enum integer token exceeds exact JSON range")
        return int(value)
    raise TypeError("enum values must be string or integer tokens")


def encode_value(value):
    """Encode an already normalized value without restoring executable objects."""
    if isinstance(value, PathValue):
        return {
            "vertices": [list(p) for p in value.vertices],
            "in_tangents": [list(p) for p in value.in_tangents],
            "out_tangents": [list(p) for p in value.out_tangents],
            "closed": value.closed,
        }
    if isinstance(value, tuple):
        return [encode_value(item) for item in value]
    if isinstance(value, Enum):
        return _enum_token(value)
    if isinstance(value, (bool, str, int)):
        return value
    return finite_real(value)


def decode_value(payload, value_type=None):
    """Decode only inert value data, optionally enforcing a property kind."""
    if value_type == "path" or isinstance(payload, dict):
        fields = {"vertices", "in_tangents", "out_tangents", "closed"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("invalid path value fields")
        payload = PathValue(**payload)
    return normalize_value(payload, value_type)


def path_topology(value):
    """Return the shape and closure required for compatible path morphing."""
    return (len(value.vertices), value.closed)


def flatten_path(value):
    """Return numeric coordinates for all vertices and relative tangents."""
    return tuple(
        component
        for group in (value.vertices, value.in_tangents, value.out_tangents)
        for point in group
        for component in point
    )


def rebuild_path(flat, template):
    """Rebuild a path from interpolated coordinates and its topology template."""
    count = len(template.vertices)
    groups = []
    for group in range(3):
        offset = group * count * 2
        groups.append(
            tuple(
                tuple(flat[offset + 2 * i : offset + 2 * i + 2]) for i in range(count)
            )
        )
    return PathValue(*groups, closed=template.closed)


def default_value(value_type):
    """Return an inert typed base without evaluating an expression or callback."""
    defaults = {
        "float": 0.0,
        "vec2": (0.0, 0.0),
        "vec3": (0.0, 0.0, 0.0),
        "color": (0.0, 0.0, 0.0),
        "bool": False,
        "enum": "default",
        "path": PathValue(()),
    }
    if type(value_type) is not str or value_type not in defaults:
        raise ValueError("unsupported value_type")
    return defaults[value_type]


def expression_value(value):
    """Convert a path to inert (vertices, in_tangents, out_tangents, closed).

    Expression paths inherit the sandbox's 64-item sequence and finite-value
    limits. Ordinary PathValue properties may hold up to 4096 vertices.
    """
    if type(value) is PathValue:
        return (value.vertices, value.in_tangents, value.out_tangents, value.closed)
    return value


def path_from_expression(value):
    """Restore validated path geometry from an inert four-part tuple."""
    if type(value) not in (tuple, list) or len(value) != 4:
        raise ValueError("path expression must return a four-part tuple")
    return PathValue(value[0], value[1], value[2], value[3])


def restore_expression_value(value, value_type):
    """Restore typed path data and exact integral enum tokens from expression data."""
    if value_type == "path":
        return path_from_expression(value)
    if value_type == "enum" and type(value) is float and value.is_integer():
        return int(value)
    return value
