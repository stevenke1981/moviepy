"""After Effects-style compositing for MoviePy.

The core values (``Buffer``, ``RenderContext``) load eagerly; everything else
is exported lazily so ``import moviepy.ae`` stays cheap.

Examples
--------
>>> import numpy as np
>>> from moviepy.ae import Buffer, RenderContext
>>> frame = np.array([[[255, 0, 0]]], dtype=np.uint8)
>>> source = Buffer.from_uint8_rgb(frame, mask=np.array([[0.5]]))
>>> source.to_uint8_rgb(bg=(0, 0, 1)).tolist()
[[[128, 0, 128]]]
>>> RenderContext(t=0.5, fps=24).frame_index
12
>>> import moviepy.ae as ae
>>> comp = ae.Composition(size=(2, 1), fps=24, duration=1)
>>> layer = comp.add_solid("bg", color=(0, 128, 0))
>>> layer.blend_mode = ae.BlendMode.SCREEN
>>> comp.get_frame(0).tolist()
[[[0, 128, 0], [0, 128, 0]]]
"""

from importlib import import_module

from moviepy.ae.buffer import Buffer, premultiply, unpremultiply
from moviepy.ae.context import RenderContext


_LAZY = {
    "Composition": "moviepy.ae.composition",
    "Marker": "moviepy.ae.composition",
    "from_moviepy": "moviepy.ae.composition",
    "Renderer": "moviepy.ae.renderer",
    "Shutter": "moviepy.ae.time.motion_blur",
    "write_master": "moviepy.ae.master",
    "MasterRender": "moviepy.ae.master",
    "BlendMode": "moviepy.ae.blend.modes",
    "blend": "moviepy.ae.blend.modes",
    "Mask": "moviepy.ae.masks.mask",
    "MaskMode": "moviepy.ae.masks.mask",
    "MaskStack": "moviepy.ae.masks.mask",
    "MatteMode": "moviepy.ae.masks.matte",
    "TrackMatte": "moviepy.ae.masks.matte",
    "Layer": "moviepy.ae.layers.base",
    "AVLayer": "moviepy.ae.layers.av",
    "CompLayer": "moviepy.ae.layers.comp",
    "AdjustmentLayer": "moviepy.ae.layers.adjustment",
    "AEEffect": "moviepy.ae.effects.base",
    "EffectStack": "moviepy.ae.effects.stack",
    "from_moviepy_effect": "moviepy.ae.effects.bridge",
    "NullLayer": "moviepy.ae.layers.null",
    "SolidLayer": "moviepy.ae.layers.solid",
    "TextLayer": "moviepy.ae.text.layer",
    "ParticleLayer": "moviepy.ae.particles.layer",
    "Transform": "moviepy.ae.transform",
    "Property": "moviepy.ae.properties.property",
    "Keyframe": "moviepy.ae.properties.keyframe",
    "Ease": "moviepy.ae.properties.easing",
    "Expression": "moviepy.ae.properties.expression",
}

_MODULES = {"fx": "moviepy.ae.fx", "effects": "moviepy.ae.effects"}

__all__ = [
    "Buffer",
    "RenderContext",
    "premultiply",
    "unpremultiply",
    *_LAZY,
    *_MODULES,
]


def __getattr__(name):
    """Load the requested public symbol without eagerly importing all modules."""
    if name in _MODULES:
        value = import_module(_MODULES[name])
    elif name in _LAZY:
        value = getattr(import_module(_LAZY[name]), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    """Include lazy public symbols in module introspection."""
    return sorted(set(globals()) | set(__all__))
