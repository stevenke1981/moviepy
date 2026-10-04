"""Effect framework: ``AEEffect``, ``EffectStack``, registry and bridges.

Built-in effects live in category sub-packages (``blur``, ``color``,
``generate``, ``time``) and register themselves; use ``moviepy.ae.fx`` or
``registry.get`` to reach them.

Examples
--------
>>> from moviepy.ae.effects import registry
>>> sorted(registry.categories())[:2]
['Blur & Sharpen', 'Color Correction']
"""

from importlib import import_module


_EXPORTS = {
    "AEEffect": "base",
    "Param": "base",
    "EffectStack": "stack",
    "MoviePyEffect": "bridge",
    "from_moviepy_effect": "bridge",
    "apply_adjustment": "adjustment",
}
__all__ = [*_EXPORTS, "registry"]


def __getattr__(name):
    """Load the requested public symbol without eagerly importing all modules."""
    if name == "registry":
        value = import_module(f"{__name__}.registry")
    elif name in _EXPORTS:
        value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    """Include lazy public symbols in module introspection."""
    return sorted(set(globals()) | set(__all__))
