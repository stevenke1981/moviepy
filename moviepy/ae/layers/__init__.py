"""Layer objects with AE timing, switches, parenting and transform rendering.

Examples
--------
>>> from moviepy.ae.layers import NullLayer, SolidLayer
>>> solid = SolidLayer("bg", color=(255, 0, 0), size=(2, 2))
>>> controller = NullLayer("controller", size=(10, 10))
>>> solid.parent = controller
>>> solid.parent_chain()[1].name
'controller'
"""

from importlib import import_module


_EXPORTS = {
    "Layer": "base",
    "AVLayer": "av",
    "NullLayer": "null",
    "SolidLayer": "solid",
    "CompLayer": "comp",
    "AdjustmentLayer": "adjustment",
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
