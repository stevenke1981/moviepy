"""AE blend modes on premultiplied float32 buffers.

Examples
--------
>>> from moviepy.ae.blend import BlendMode
>>> BlendMode.coerce("Classic Color Burn")
<BlendMode.CLASSIC_COLOR_BURN: 'classic_color_burn'>
>>> BlendMode.SCREEN.label, BlendMode.SCREEN.category
('Screen', 'lighten')
>>> len(BlendMode)
38
"""

from importlib import import_module


_EXPORTS = {
    "BlendMode": "modes",
    "blend": "modes",
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
