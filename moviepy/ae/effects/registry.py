"""Effect registry: lookup by AE name or class name, categories and metadata.

Examples
--------
>>> from moviepy.ae.effects import registry
>>> registry.get("Gaussian Blur").__name__
'GaussianBlur'
>>> registry.get("gaussianblur") is registry.get("GaussianBlur")
True
>>> [cls.name for cls in registry.list(category="Blur & Sharpen")]
['Gaussian Blur']
"""

import json
from importlib import import_module

from moviepy.ae.effects.base import AEEffect, Param


_BUILTINS = (
    "moviepy.ae.effects.blur.gaussian_blur",
    "moviepy.ae.effects.color.brightness_contrast",
    "moviepy.ae.effects.color.invert",
    "moviepy.ae.effects.color.tint",
    "moviepy.ae.effects.generate.fill",
    "moviepy.ae.effects.perspective.drop_shadow",
    "moviepy.ae.effects.stylize.glow",
    "moviepy.ae.effects.time.echo",
)
_BY_NAME = {}
_BY_CLASS = {}
_STATE = {"loaded": False}


def _key(name):
    """Normalize a lookup key: case, spaces, ``&``, ``-`` and ``/`` ignored."""
    if not isinstance(name, str):
        raise TypeError("effect name must be a string")
    return "".join(ch for ch in name.lower() if ch.isalnum())


def register(cls):
    """Register an ``AEEffect`` subclass (usable as a class decorator).

    Raises
    ------
    TypeError
        If ``cls`` is not an ``AEEffect`` subclass or ``PARAMS`` is malformed.
    ValueError
        If metadata is missing or the name collides with another effect.
    """
    if not isinstance(cls, type) or not issubclass(cls, AEEffect):
        raise TypeError("only AEEffect subclasses can be registered")
    for field in ("name", "category"):
        value = getattr(cls, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{cls.__name__}.{field} must be a non-empty string")
    if not isinstance(cls.PARAMS, tuple) or not all(
        isinstance(param, Param) for param in cls.PARAMS
    ):
        raise TypeError(f"{cls.__name__}.PARAMS must be a tuple of Param")
    names = [param.name for param in cls.PARAMS]
    if len(set(names)) != len(names):
        raise ValueError(f"{cls.__name__} declares duplicate parameters")
    for table, key in ((_BY_NAME, _key(cls.name)), (_BY_CLASS, _key(cls.__name__))):
        existing = table.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(f"effect name {cls.name!r} is already registered")
    _BY_NAME[_key(cls.name)] = cls
    _BY_CLASS[_key(cls.__name__)] = cls
    return cls


def unregister(name):
    """Remove an effect registered under ``name`` (AE or class name)."""
    cls = get(name)
    _BY_NAME.pop(_key(cls.name), None)
    _BY_CLASS.pop(_key(cls.__name__), None)


def load_builtins():
    """Import the built-in effect modules once so they self-register."""
    if not _STATE["loaded"]:
        _STATE["loaded"] = True
        for module in _BUILTINS:
            import_module(module)


def get(name):
    """Return the effect class registered under an AE or class name."""
    load_builtins()
    key = _key(name)
    cls = _BY_NAME.get(key) or _BY_CLASS.get(key)
    if cls is None:
        raise KeyError(f"unknown effect {name!r}")
    return cls


def get_class(class_name):
    """Return the effect class whose Python class name is ``class_name``."""
    load_builtins()
    cls = _BY_CLASS.get(_key(class_name))
    if cls is None or cls.__name__ != class_name:
        raise KeyError(f"unknown effect class {class_name!r}")
    return cls


def list(category=None):  # noqa: A001 - spec API: registry.list(category=...)
    """Return registered effect classes sorted by name, optionally filtered."""
    load_builtins()
    classes = sorted(_BY_NAME.values(), key=lambda cls: cls.name)
    if category is None:
        return classes
    return [cls for cls in classes if cls.category == category]


def categories():
    """Return the sorted category names in use."""
    return sorted({cls.category for cls in list()})


def metadata(name=None):
    """Return metadata for one effect, or for all effects sorted by name."""
    if name is not None:
        return get(name).metadata()
    return [cls.metadata() for cls in list()]


def to_json(indent=2):
    """Return the registry metadata as a versioned JSON document."""
    return json.dumps({"schema": 1, "effects": metadata()}, indent=indent)
