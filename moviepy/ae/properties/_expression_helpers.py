"""Analytic vector, color, time and interpolation expression functions."""

import colorsys
import math

from moviepy.ae.properties._sandbox import ExpressionError, safe_number, sanitize


def vector(value, size=None):
    """Validate a finite vector and optional exact dimension."""
    value = sanitize(value)
    if type(value) is not tuple or not value or len(value) > 64:
        raise ExpressionError("expected a short numeric vector")
    if size is not None and len(value) != size:
        raise ExpressionError("incorrect vector dimension")
    return tuple(safe_number(item) for item in value)


def componentwise(function, *values):
    """Broadcast scalars only, without allocating multidimensional arrays."""
    clean = [sanitize(value) for value in values]
    sizes = {len(value) for value in clean if type(value) is tuple}
    if len(sizes) > 1:
        raise ExpressionError("vector shape mismatch")
    if not sizes:
        return safe_number(function(*(safe_number(value) for value in clean)))
    size = sizes.pop()
    rows = [vector(v) if type(v) is tuple else (safe_number(v),) * size for v in clean]
    return tuple(safe_number(function(*row)) for row in zip(*rows))


def clamp(value, minimum=0, maximum=1):
    """Clamp scalar or vector components to ordered bounds."""

    def bounded(number, low, high):
        if low > high:
            raise ExpressionError("clamp minimum exceeds maximum")
        return min(max(number, low), high)

    return componentwise(bounded, value, minimum, maximum)


def interpolate(t, tMin, tMax, v1, v2, mode="linear"):
    """Interpolate scalar or vector endpoints within a clamped interval."""
    t, start, end = map(safe_number, (t, tMin, tMax))
    if end <= start:
        raise ExpressionError("interpolation interval must be positive")
    fraction = min(max((t - start) / (end - start), 0), 1)
    curves = {
        "linear": lambda x: x,
        "ease": lambda x: x * x * (3 - 2 * x),
        "easeIn": lambda x: x * x * (2 - x),
        "easeOut": lambda x: x * (1 + x - x * x),
    }
    fraction = curves[mode](fraction)
    return componentwise(lambda a, b: a + (b - a) * fraction, v1, v2)


def length(a, b=None):
    """Return Euclidean vector magnitude or distance between two points."""
    first = vector(a)
    if b is not None:
        second = vector(b, len(first))
        first = tuple(x - y for x, y in zip(first, second))
    return math.hypot(*first)


def normalize(value):
    """Return unit direction, preserving a zero vector as zero."""
    value = vector(value)
    magnitude = length(value)
    return tuple(x / magnitude for x in value) if magnitude else (0.0,) * len(value)


def dot(a, b):
    """Return the dot product of equally sized finite vectors."""
    a, b = vector(a), vector(b)
    if len(a) != len(b):
        raise ExpressionError("vector shape mismatch")
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    """Return the right-handed cross product of two three-dimensional vectors."""
    a, b = vector(a, 3), vector(b, 3)
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def look_at(origin, target):
    """Return XYZ Euler degrees; positive Z points forward and roll is zero."""
    origin, target = vector(origin, 3), vector(target, 3)
    x, y, z = (b - a for a, b in zip(origin, target))
    if x == y == z == 0:
        return (0.0, 0.0, 0.0)
    return (
        math.degrees(-math.atan2(y, math.hypot(x, z))),
        math.degrees(math.atan2(x, z)),
        0.0,
    )


def rgb_to_hsl(value):
    """Convert normalized RGB to HSL while preserving optional alpha."""
    value = vector(value)
    if len(value) not in (3, 4):
        raise ExpressionError("color must have three or four channels")
    if any(x < 0 or x > 1 for x in value[:3]):
        raise ExpressionError("RGB channels must be in [0,1]")
    hue, lightness, saturation = colorsys.rgb_to_hls(*value[:3])
    return (hue, saturation, lightness) + value[3:]


def hsl_to_rgb(value):
    """Convert HSL to normalized RGB while preserving optional alpha."""
    value = vector(value)
    if len(value) not in (3, 4):
        raise ExpressionError("color must have three or four channels")
    hue, saturation, lightness = value[:3]
    if not (0 <= saturation <= 1 and 0 <= lightness <= 1):
        raise ExpressionError("HSL saturation/lightness must be in [0,1]")
    return colorsys.hls_to_rgb(hue % 1, lightness, saturation) + value[3:]


def frame_time(frames, fps):
    """Convert a frame position to seconds at a positive frame rate."""
    frames, fps = safe_number(frames), safe_number(fps)
    if fps <= 0:
        raise ExpressionError("fps must be positive")
    return frames / fps


def time_frames(t, fps):
    """Floor absolute time to a frame index at a positive frame rate."""
    t, fps = safe_number(t), safe_number(fps)
    if fps <= 0:
        raise ExpressionError("fps must be positive")
    return math.floor(safe_number(t * fps))


def numeric_min(*values):
    """Return the minimum of inert numeric arguments or one vector."""
    if len(values) == 1 and type(values[0]) is tuple:
        values = values[0]
    if not values:
        raise ExpressionError("min needs numeric arguments")
    return min(safe_number(x) for x in values)


def numeric_max(*values):
    """Return the maximum of inert numeric arguments or one vector."""
    if len(values) == 1 and type(values[0]) is tuple:
        values = values[0]
    if not values:
        raise ExpressionError("max needs numeric arguments")
    return max(safe_number(x) for x in values)


def registry(t, fps):
    """Build the fixed analytic helper registry for one evaluation."""
    helpers = {
        "clamp": clamp,
        "length": length,
        "normalize": normalize,
        "dot": dot,
        "cross": cross,
        "lookAt": look_at,
        "rgbToHsl": rgb_to_hsl,
        "hslToRgb": hsl_to_rgb,
        "min": numeric_min,
        "max": numeric_max,
        "framesToTime": lambda frames, fps=fps: frame_time(frames, fps),
        "timeToFrames": lambda t=t, fps=fps: time_frames(t, fps),
        "radiansToDegrees": lambda value: componentwise(math.degrees, value),
        "degreesToRadians": lambda value: componentwise(math.radians, value),
    }
    for name in ("sin", "cos", "tan", "sqrt", "floor", "ceil", "exp", "log"):
        function = getattr(math, name)
        helpers[name] = lambda value, fn=function: componentwise(fn, value)
    helpers["abs"] = lambda value: componentwise(abs, value)
    helpers["round"] = lambda value: componentwise(round, value)
    for name in ("linear", "ease", "easeIn", "easeOut"):
        helpers[name] = lambda t, tMin, tMax, v1, v2, mode=name: interpolate(
            t, tMin, tMax, v1, v2, mode
        )
    return helpers
