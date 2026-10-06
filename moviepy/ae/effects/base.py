"""Effect base class, parameter metadata and per-effect compositing options.

Every AE-style effect subclasses ``AEEffect``, declares its parameters as
``Param`` metadata and implements ``render``. Parameters are ``Property``
objects, so they accept constants, keyframes, callables and expressions.
The base class also implements the AE "Compositing Options": an effect mask,
an effect enable switch and ``blend_with_original``.
"""

import math
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar, Optional, Tuple

import numpy as np

from moviepy.ae._geometry import finite_real, to_property, validate_flag
from moviepy.ae.buffer import Buffer
from moviepy.Effect import Effect


_KINDS = ("float", "vec2", "vec3", "color", "bool", "enum")
_RESERVED = frozenset(("enabled", "mask", "blend_with_original", "name", "params"))


@dataclass(frozen=True)
class Param:
    """Describe one effect parameter for validation, GUIs and project files.

    Parameters
    ----------
    name : str
        snake_case parameter name, matching the AE effect panel.
    kind : {"float", "vec2", "vec3", "color", "bool", "enum"}
        Property value type. Colors are RGB codes in 0..255.
    default : object
        Default value.
    limits : tuple of float, optional
        Inclusive range; constants are validated and animated values clamped.
    choices : tuple of str, optional
        Allowed tokens for ``enum`` parameters.
    label : str, optional
        AE panel label; defaults to the title-cased name.
    unit : {None, "px"}, optional
        Pixel lengths are scaled to the effect's processing resolution.
        This runtime annotation does not change the version 1 JSON schema.
    """

    name: str
    kind: str
    default: object
    limits: Optional[Tuple[float, float]] = None
    choices: Optional[Tuple[str, ...]] = None
    label: Optional[str] = None
    unit: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("parameter name must be an identifier")
        if self.name in _RESERVED:
            raise ValueError(f"parameter name {self.name!r} is reserved")
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported parameter kind {self.kind!r}")
        if self.unit not in (None, "px"):
            raise ValueError(f"unsupported parameter unit {self.unit!r}")
        if self.unit == "px" and self.kind not in ("float", "vec2", "vec3"):
            raise ValueError("pixel units require a float or vector parameter")
        if self.kind == "enum" and not self.choices:
            raise ValueError("enum parameters need choices")
        if self.label is None:
            object.__setattr__(self, "label", self.name.replace("_", " ").title())

    def coerce(self, value):
        """Return a ``Property`` for ``value`` after validating constants."""
        if self.kind == "enum" and isinstance(value, str):
            self.check_choice(value)
        if self.kind == "color" and isinstance(value, (tuple, list)):
            if any(not 0.0 <= finite_real(v, self.name) <= 255.0 for v in value):
                raise ValueError(f"{self.name} components must be within 0..255")
        return to_property(value, self.name, self.kind, self.limits)

    def check_choice(self, token):
        """Raise ``ValueError`` unless ``token`` is an allowed enum value."""
        if token not in self.choices:
            allowed = ", ".join(self.choices)
            raise ValueError(f"{self.name} must be one of: {allowed}")
        return token

    def clamp(self, value):
        """Clamp an evaluated value into the declared range."""
        if self.kind == "enum":
            return self.check_choice(value)
        if self.kind == "color":
            return tuple(min(255.0, max(0.0, float(v))) for v in value[:3])
        if self.limits is None or not isinstance(value, Real):
            return value
        low, high = self.limits
        return min(high, max(low, float(value)))

    def scale(self, value, pixel_scale):
        """Convert an evaluated pixel length to the processing resolution."""
        if self.unit != "px" or pixel_scale == 1.0:
            return value
        if self.kind == "float":
            return value * pixel_scale
        return tuple(component * pixel_scale for component in value)

    def to_dict(self):
        """Return JSON-ready metadata."""
        return {
            "name": self.name,
            "label": self.label,
            "type": self.kind,
            "default": _json_value(self.default),
            "range": None if self.limits is None else list(self.limits),
            "choices": None if self.choices is None else list(self.choices),
        }


def _json_value(value):
    """Convert tuples to lists for JSON metadata."""
    return list(value) if isinstance(value, tuple) else value


class AEEffect(Effect):
    """Base class for AE-style effects living in a layer's ``EffectStack``.

    Subclasses set ``name`` (registry key, AE menu name), ``category`` (AE
    effect category), ``PARAMS`` and implement ``render``.

    Parameters
    ----------
    enabled : bool, optional
        The effect switch (``fx``). A disabled effect is skipped entirely.
    mask : Mask or MaskStack, optional
        Limits the effect to the masked area of its input (layer space for
        ordinary layers, composition space for adjustment layers).
    blend_with_original : float or Property, optional
        Percent 0..100 of the unprocessed input mixed back over the result
        (AE's Effect Opacity / Blend With Original); 0 keeps only the effect.
    ``**params``
        Values for the parameters declared in ``PARAMS``.

    Notes
    -----
    Effects are also MoviePy ``Effect`` objects: ``clip.with_effects([fx])``
    wraps the clip in a single-layer ``Composition`` running this effect.
    """

    name: ClassVar[str] = ""
    category: ClassVar[str] = ""
    PARAMS: ClassVar[Tuple[Param, ...]] = ()

    def __init__(self, *, enabled=True, mask=None, blend_with_original=0.0, **params):
        unknown = set(params) - {param.name for param in self.PARAMS}
        if unknown:
            raise TypeError(f"{type(self).__name__} got unknown parameters {unknown}")
        object.__setattr__(self, "params", {})
        for param in self.PARAMS:
            self.params[param.name] = param.coerce(
                params.get(param.name, param.default)
            )
        self.enabled = enabled
        self.mask = mask
        self.blend_with_original = blend_with_original

    # -- parameter access ------------------------------------------------------ #

    def __getattr__(self, attribute):
        params = self.__dict__.get("params")
        if params is not None and attribute in params:
            return params[attribute]
        raise AttributeError(f"{type(self).__name__} has no attribute {attribute!r}")

    def __setattr__(self, attribute, value):
        for param in type(self).PARAMS:
            if param.name == attribute:
                self.params[attribute] = param.coerce(value)
                return
        object.__setattr__(self, attribute, value)

    def copy(self):
        """Return an independent copy (parameter dict included)."""
        result = super().copy()
        object.__setattr__(result, "params", dict(self.params))
        return result

    @property
    def enabled(self):
        """Return the effect switch."""
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        object.__setattr__(self, "_enabled", validate_flag(value, "enabled"))

    @property
    def mask(self):
        """Return the effect mask (``MaskStack``) or None."""
        return self._mask

    @mask.setter
    def mask(self, value):
        from moviepy.ae.masks.mask import Mask, MaskStack

        if isinstance(value, Mask):
            value = MaskStack([value])
        elif value is not None and not isinstance(value, MaskStack):
            raise TypeError("mask must be a Mask, MaskStack or None")
        object.__setattr__(self, "_mask", value)

    @property
    def blend_with_original(self):
        """Return the Blend With Original Property (percent)."""
        return self._blend_with_original

    @blend_with_original.setter
    def blend_with_original(self, value):
        prop = to_property(value, "blend_with_original", "float", (0.0, 100.0))
        object.__setattr__(self, "_blend_with_original", prop)

    def values_at(self, t, context=None, *, pixel_scale=1.0, **bindings):
        """Evaluate, clamp and scale parameters at layer time ``t``."""
        pixel_scale = finite_real(pixel_scale, "pixel_scale")
        if pixel_scale <= 0:
            raise ValueError("pixel_scale must be positive")
        evaluation = dict(bindings)
        evaluation["context"] = context
        return {
            param.name: param.scale(
                param.clamp(self.params[param.name].value_at(t, **evaluation)),
                pixel_scale,
            )
            for param in self.PARAMS
        }

    # -- hooks for subclasses ------------------------------------------------ #

    def bounds_expand(self, size, t, context=None, values=None):
        """Return ``(left, top, right, bottom)`` pixels the effect grows by."""
        return (0, 0, 0, 0)

    def input_margin(self, size, t, context=None, values=None):
        """Return input pixels needed beyond an output region on each side.

        This is independent of output growth: an effect that repeats its
        input edges can read neighboring pixels without growing its output.
        """
        return self.bounds_expand(size, t, context, values)

    def temporal_window(self, t, context=None, values=None):
        """Return ``(before, after)`` seconds of input this effect may read."""
        return (0.0, 0.0)

    def render(self, src, t, context=None, values=None):
        """Return the processed premultiplied ``Buffer``.

        ``src`` is already padded by ``bounds_expand``; ``values`` holds the
        evaluated parameters (computed with ``values_at`` when omitted).
        """
        raise NotImplementedError("effects must implement render")

    def render_temporal(self, src, t, context, values, source_at):
        """Render with access to other times through ``source_at(dt)``.

        ``source_at(dt)`` returns this effect's input ``dt`` seconds (layer
        time) away; ``dt`` must lie inside ``temporal_window``.
        """
        return self.render(src, t, context, values)

    # -- stack entry point ----------------------------------------------------- #

    def process(
        self, src, t, context=None, *, bindings=None, source_at=None, pixel_scale=1.0
    ):
        """Run the effect with bounds growth, effect mask and blending.

        Parameters
        ----------
        src : Buffer
            Premultiplied input.
        t : float
            Layer time used to evaluate the parameters.
        context : RenderContext, optional
            Quality, seed and expression context.
        bindings : dict, optional
            Expression identity bindings (``index``, ``layer_id``).
        source_at : callable, optional
            Supplies inputs at other times for temporal effects.
        pixel_scale : float, optional
            Pixels per authored pixel: 1 for source-space effects and the
            render resolution scale for composition-space adjustments.
        """
        if not self._enabled:
            return src
        bindings = {} if bindings is None else dict(bindings)
        values = self.values_at(t, context, pixel_scale=pixel_scale, **bindings)
        margins = [
            max(0, int(v)) for v in self.bounds_expand(src.size, t, context, values)
        ]
        padded = src
        if any(margins) and 0 not in src.size:
            padded = src.pad(
                left=margins[0], top=margins[1], right=margins[2], bottom=margins[3]
            )
        if self.temporal_window(t, context, values) != (0.0, 0.0):
            if source_at is None:
                raise ValueError(f"{self.name} needs a temporal source")
            result = self.render_temporal(padded, t, context, values, source_at)
        else:
            result = self.render(padded, t, context, values)
        if not isinstance(result, Buffer):
            raise TypeError(f"{self.name} render must return a Buffer")
        return self._composite_on_original(
            src, result, t, context, bindings, pixel_scale
        )

    def _composite_on_original(
        self, original, result, t, context, bindings, pixel_scale
    ):
        """Mix the result with the original through mask and blend amount."""
        evaluation = dict(bindings, context=context)
        keep = self._blend_with_original.value_at(t, **evaluation)
        amount = 1.0 - min(1.0, max(0.0, keep / 100.0))
        if amount >= 1.0 and self._mask is None:
            return result
        bounds = _union(original.bounds, result.bounds, original.size, result.size)
        base = original.crop(bounds).expand_to(bounds)
        processed = result.crop(bounds).expand_to(bounds)
        factor = np.float32(amount)
        if self._mask is not None:
            coverage = self._mask.coverage(
                bounds, t, context, pixel_scale=pixel_scale, **bindings
            )
            if coverage is not None:
                factor = coverage[..., None] * factor
        rgba = base.rgba + (processed.rgba - base.rgba) * factor
        return Buffer._publish(
            rgba.astype(np.float32, copy=False), bounds[:2], original.color_space
        )

    # -- MoviePy interop -------------------------------------------------------- #

    def apply(self, clip):
        """Return ``clip`` as a single-layer ``Composition`` with this effect."""
        from moviepy.ae.composition import Composition

        if clip.duration is None:
            raise ValueError("AE effects need a clip with a duration")
        comp = Composition(
            size=clip.size,
            fps=getattr(clip, "fps", None) or 24,
            duration=clip.duration,
            transparent=clip.mask is not None,
            name=self.name or type(self).__name__,
        )
        layer = comp.add_clip(clip, "source")
        layer.effects.add(self.copy())
        return comp

    @classmethod
    def metadata(cls):
        """Return JSON-ready metadata: name, category, class and parameters."""
        return {
            "name": cls.name,
            "category": cls.category,
            "class": cls.__name__,
            "temporal": cls.temporal_window is not AEEffect.temporal_window,
            "params": [param.to_dict() for param in cls.PARAMS],
        }

    def __repr__(self):
        return f"{type(self).__name__}(enabled={self._enabled})"


def _union(first, second, first_size, second_size):
    """Return the union of two rectangles, ignoring empty ones."""
    if 0 in first_size:
        return second
    if 0 in second_size:
        return first
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def sigma_margin(sigma):
    """Return a padding (px) that holds a Gaussian of ``sigma`` (3 sigma)."""
    return 0 if sigma <= 0 else int(math.ceil(3.0 * sigma))
