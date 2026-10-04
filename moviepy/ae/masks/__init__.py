"""Layer masks (Bezier paths, modes, feather, expansion) and track mattes.

Examples
--------
>>> from moviepy.ae.masks import Mask, MaskMode
>>> mask = Mask.ellipse(center=(50, 50), size=(40, 20), feather=4, mode="subtract")
>>> mask.mode is MaskMode.SUBTRACT, mask.feather.value_at(0)
(True, (4.0, 4.0))
"""

from importlib import import_module


_EXPORTS = {
    "Mask": "mask",
    "MaskMode": "mask",
    "MaskStack": "mask",
    "MatteMode": "matte",
    "TrackMatte": "matte",
    "matte_values": "matte",
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
