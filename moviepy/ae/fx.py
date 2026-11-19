"""AE effect classes by Python name, e.g. ``moviepy.ae.fx.GaussianBlur``.

Attribute access resolves through ``moviepy.ae.effects.registry``, so custom
effects registered with ``registry.register`` appear here as well.

Examples
--------
>>> import moviepy.ae as ae
>>> blur = ae.fx.GaussianBlur(blurriness=20)
>>> blur.name, blur.blurriness.value_at(0)
('Gaussian Blur', 20.0)
"""

from moviepy.ae.effects import registry


def __getattr__(name):
    """Return a registered effect class by its class name."""
    try:
        return registry.get_class(name)
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    """List registered effect class names."""
    return sorted(cls.__name__ for cls in registry.list())
