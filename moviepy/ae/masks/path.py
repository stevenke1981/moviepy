"""Bezier mask paths: shape constructors, SVG import and flattening.

Paths are ``moviepy.ae.properties.PathValue`` objects (vertices plus
*relative* in/out tangents), so they keyframe through ``Property`` exactly
like any other WS-01 value: paths with the same vertex count and closure are
interpolated point by point.

Coordinates follow the WS-02 pixel-center convention: layer pixel ``[row,
col]`` is centered on ``(col, row)`` and covers ``col - 0.5 .. col + 0.5``.
A rectangle centered on ``((W - 1) / 2, (H - 1) / 2)`` with size ``(W, H)``
therefore covers a ``W x H`` layer exactly.

Examples
--------
>>> from moviepy.ae.masks.path import rect, flatten
>>> path = rect(center=(1.5, 1.5), size=(4, 4))
>>> path.vertices
((-0.5, -0.5), (3.5, -0.5), (3.5, 3.5), (-0.5, 3.5))
>>> flatten(path).shape
(4, 2)
"""

import math
import re

import numpy as np

from moviepy.ae.properties.values import PathValue, finite_real


# Control-point distance that makes four cubic segments approximate a circle
# with a maximum radial error of about 0.027%.
KAPPA = 4.0 * (math.sqrt(2.0) - 1.0) / 3.0
_MAX_SVG_LENGTH = 100_000
_MAX_SEGMENT_STEPS = 512


def _pair(value, name):
    """Validate a two-component coordinate."""
    try:
        first, second = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two numbers") from error
    return (finite_real(first, name), finite_real(second, name))


def _positive_pair(value, name):
    """Validate a strictly positive (width, height)."""
    width, height = _pair(value, name)
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} must be positive")
    return width, height


def polygon(points, *, closed=True):
    """Return a straight-edged path through ``points``.

    Parameters
    ----------
    points : sequence of (x, y)
        At least two vertices.
    closed : bool, optional
        Join the last vertex to the first.
    """
    vertices = tuple(_pair(point, "points") for point in points)
    if len(vertices) < 2:
        raise ValueError("a path needs at least two vertices")
    return PathValue(vertices, closed=closed)


def rect(center, size):
    """Return a closed rectangle given its center and (width, height)."""
    x, y = _pair(center, "center")
    width, height = _positive_pair(size, "size")
    left, right = x - width / 2.0, x + width / 2.0
    top, bottom = y - height / 2.0, y + height / 2.0
    return polygon([(left, top), (right, top), (right, bottom), (left, bottom)])


def ellipse(center, size):
    """Return a closed four-vertex Bezier ellipse with the given bounding size."""
    x, y = _pair(center, "center")
    width, height = _positive_pair(size, "size")
    rx, ry = width / 2.0, height / 2.0
    kx, ky = rx * KAPPA, ry * KAPPA
    vertices = ((x, y - ry), (x + rx, y), (x, y + ry), (x - rx, y))
    in_tangents = ((-kx, 0.0), (0.0, -ky), (kx, 0.0), (0.0, ky))
    out_tangents = ((kx, 0.0), (0.0, ky), (-kx, 0.0), (0.0, -ky))
    return PathValue(vertices, in_tangents, out_tangents, closed=True)


def rounded_rect(center, size, roundness):
    """Return a rectangle whose corners are quarter ellipses of ``roundness`` px."""
    x, y = _pair(center, "center")
    width, height = _positive_pair(size, "size")
    radius = min(finite_real(roundness, "roundness"), width / 2.0, height / 2.0)
    if radius < 0:
        raise ValueError("roundness must be nonnegative")
    if radius == 0:
        return rect(center, size)
    left, right = x - width / 2.0, x + width / 2.0
    top, bottom = y - height / 2.0, y + height / 2.0
    k = radius * KAPPA
    vertices = (
        (left + radius, top),
        (right - radius, top),
        (right, top + radius),
        (right, bottom - radius),
        (right - radius, bottom),
        (left + radius, bottom),
        (left, bottom - radius),
        (left, top + radius),
    )
    zero = (0.0, 0.0)
    in_tangents = (
        (-k, 0.0),
        zero,
        (0.0, -k),
        zero,
        (k, 0.0),
        zero,
        (0.0, k),
        zero,
    )
    out_tangents = (
        zero,
        (k, 0.0),
        zero,
        (0.0, k),
        zero,
        (-k, 0.0),
        zero,
        (0.0, -k),
    )
    return PathValue(vertices, in_tangents, out_tangents, closed=True)


def star(center, points, outer_radius, inner_radius, rotation=0.0):
    """Return a straight-edged star (polystar) path.

    Parameters
    ----------
    center : (x, y)
        Star center.
    points : int
        Number of outer points, at least 2.
    outer_radius, inner_radius : float
        Positive radii of the outer and inner vertices.
    rotation : float, optional
        Clockwise degrees; 0 puts the first point straight up.
    """
    x, y = _pair(center, "center")
    if isinstance(points, bool) or not isinstance(points, int) or points < 2:
        raise ValueError("points must be an integer of at least 2")
    outer = finite_real(outer_radius, "outer_radius")
    inner = finite_real(inner_radius, "inner_radius")
    if outer <= 0 or inner <= 0:
        raise ValueError("radii must be positive")
    start = math.radians(finite_real(rotation, "rotation")) - math.pi / 2.0
    vertices = []
    for index in range(2 * points):
        radius = outer if index % 2 == 0 else inner
        angle = start + math.pi * index / points
        vertices.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    return polygon(vertices)


# -- SVG ------------------------------------------------------------------------

_TOKEN = re.compile(r"[MmLlHhVvCcSsQqZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_UNSUPPORTED = re.compile(r"[^MmLlHhVvCcSsQqZz0-9eE+\-.,\s]")
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "Z": 0}


def from_svg_path(d):
    """Parse one SVG path ``d`` string into a ``PathValue``.

    Supports ``M L H V C S Q Z`` in absolute and relative forms (quadratic
    segments are converted to exact cubics). Arcs and multiple subpaths are
    rejected with ``ValueError`` because an AE mask holds a single path.
    """
    if not isinstance(d, str):
        raise TypeError("d must be a string")
    if len(d) > _MAX_SVG_LENGTH:
        raise ValueError("SVG path exceeds the supported length")
    tokens = _TOKEN.findall(d)
    if not tokens or _UNSUPPORTED.search(d):
        raise ValueError("unsupported or empty SVG path")
    builder = _SvgBuilder()
    command, index = None, 0
    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command, index = token, index + 1
            if command in "Zz":
                builder.close()
                continue
        elif command is None:
            raise ValueError("SVG path must start with a command")
        arity = _ARITY[command.upper()]
        arguments = tokens[index : index + arity]
        if len(arguments) != arity or any(item.isalpha() for item in arguments):
            raise ValueError("malformed SVG path arguments")
        builder.segment(command, [float(item) for item in arguments])
        index += arity
        if command in "Mm":
            command = "L" if command == "M" else "l"
    return builder.build()


class _SvgBuilder:
    """Accumulate SVG segments as vertices with relative tangents."""

    def __init__(self):
        self.vertices, self.ins, self.outs = [], [], []
        self.current = (0.0, 0.0)
        self.last_control = None
        self.closed = False

    def segment(self, command, values):
        relative = command.islower()
        kind = command.upper()
        if self.closed:
            raise ValueError("only one SVG subpath is supported")
        if kind == "M":
            if self.vertices:
                raise ValueError("only one SVG subpath is supported")
            self._start(self._point(values, relative))
        elif kind in "LHV":
            self._line(kind, values, relative)
        elif kind in "CS":
            self._cubic(kind, values, relative)
        else:
            self._quadratic(values, relative)

    def _point(self, values, relative):
        x, y = values
        if relative:
            return (self.current[0] + x, self.current[1] + y)
        return (x, y)

    def _start(self, point):
        self.vertices.append(point)
        self.ins.append((0.0, 0.0))
        self.outs.append((0.0, 0.0))
        self.current, self.last_control = point, None

    def _require_start(self):
        if not self.vertices:
            raise ValueError("SVG path must start with M")

    def _line(self, kind, values, relative):
        self._require_start()
        x, y = self.current
        if kind == "H":
            point = (x + values[0] if relative else values[0], y)
        elif kind == "V":
            point = (x, y + values[0] if relative else values[0])
        else:
            point = self._point(values, relative)
        self._start(point)

    def _cubic(self, kind, values, relative):
        self._require_start()
        origin = self.current
        if kind == "S":
            last = self.last_control or origin
            first = (2 * origin[0] - last[0], 2 * origin[1] - last[1])
            second, end = (self._point(values[i : i + 2], relative) for i in (0, 2))
        else:
            first, second, end = (
                self._point(values[i : i + 2], relative) for i in (0, 2, 4)
            )
        self._curve(origin, first, second, end)

    def _quadratic(self, values, relative):
        self._require_start()
        origin = self.current
        control, end = (self._point(values[i : i + 2], relative) for i in (0, 2))
        first = tuple(o + 2.0 / 3.0 * (c - o) for o, c in zip(origin, control))
        second = tuple(e + 2.0 / 3.0 * (c - e) for e, c in zip(end, control))
        self._curve(origin, first, second, end)

    def _curve(self, origin, first, second, end):
        self.outs[-1] = (first[0] - origin[0], first[1] - origin[1])
        self._start(end)
        self.ins[-1] = (second[0] - end[0], second[1] - end[1])
        self.last_control = second

    def close(self):
        self._require_start()
        if len(self.vertices) > 1 and np.allclose(self.vertices[0], self.vertices[-1]):
            self.ins[0] = self.ins.pop()
            self.vertices.pop()
            self.outs.pop()
        self.closed = True

    def build(self):
        if len(self.vertices) < 2:
            raise ValueError("a path needs at least two vertices")
        return PathValue(
            tuple(self.vertices), tuple(self.ins), tuple(self.outs), closed=True
        )


# -- flattening -----------------------------------------------------------------


def flatten(path, scale=1.0):
    """Return polygon points approximating ``path`` as an ``(N, 2)`` array.

    Straight segments contribute their end vertex only; curved segments are
    subdivided uniformly with a step count that grows with their control
    polygon length measured in ``scale``-multiplied pixels (about one point
    per 2 output pixels), which keeps the polygon within a small fraction of
    a pixel of the true curve. Open paths are treated as closed for filling,
    like AE.
    """
    if not isinstance(path, PathValue):
        raise TypeError("path must be a PathValue")
    vertices = np.asarray(path.vertices, dtype=np.float64)
    ins = np.asarray(path.in_tangents, dtype=np.float64)
    outs = np.asarray(path.out_tangents, dtype=np.float64)
    count = len(vertices)
    points = [vertices[:1]]
    for index in range(count):
        following = (index + 1) % count
        if following == 0 and not path.closed:
            break
        segment = _segment_points(
            vertices[index],
            vertices[index] + outs[index],
            vertices[following] + ins[following],
            vertices[following],
            scale,
        )
        points.append(segment)
    result = np.concatenate(points)
    if path.closed and len(result) > 1 and np.allclose(result[0], result[-1]):
        result = result[:-1]
    return result


def _segment_points(p0, p1, p2, p3, scale):
    """Sample one cubic Bezier segment, excluding its start point."""
    if np.allclose(p1, p0) and np.allclose(p2, p3):
        return p3[None, :]
    length = (
        np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
    ) * scale
    steps = int(min(_MAX_SEGMENT_STEPS, max(4, math.ceil(length / 2.0))))
    t = np.linspace(0.0, 1.0, steps + 1)[1:, None]
    u = 1.0 - t
    return u**3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t**3 * p3
