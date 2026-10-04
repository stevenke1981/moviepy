"""Typed animation properties, keyframes, easing and bounded expressions.

Examples
--------
>>> from moviepy.ae.properties import Expression, Property
>>> position = Property(0, keyframes=[(0, 0), (2, 10)])
>>> position.value_at(1)
5.0
>>> position.with_expression(Expression("value + time")).value_at(1)
6.0
"""

from importlib import import_module


_EXPORTS = {
    "Ease": "easing",
    "Keyframe": "keyframe",
    "SpatialSegment": "spatial",
    "Property": "property",
    "PathValue": "values",
    "Vec2": "values",
    "Vec3": "values",
    "Expression": "expression",
    "ExpressionError": "expression",
    "ExpressionSnapshot": "expression",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    """Load the requested public symbol without eagerly importing all modules."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    """Include lazy public symbols in module introspection."""
    return sorted(set(globals()) | set(__all__))
