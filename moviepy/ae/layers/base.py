"""Layer timing, switches, parenting and the shared render entry point.

A layer owns a ``Transform``, a composition-time
window, a source that produces premultiplied pixels in layer-local space, a
``MaskStack`` (WS-05), an optional ``TrackMatte`` (WS-05, applied by the
WS-03 renderer) and an ``EffectStack`` (WS-06). Time remapping (WS-07) is
still absent.
"""

import math
from numbers import Integral, Real

import numpy as np

from moviepy.ae._geometry import validate_flag
from moviepy.ae.blend.modes import BlendMode
from moviepy.ae.effects.base import AEEffect
from moviepy.ae.effects.stack import EffectStack
from moviepy.ae.masks.mask import Mask, MaskStack
from moviepy.ae.masks.matte import TrackMatte
from moviepy.ae.transform import Transform
from moviepy.ae.warp import resolve_interpolation, warp_buffer


_MAX_NAME = 256


def _name(value):
    """Validate a non-empty, bounded layer name."""
    if not isinstance(value, str):
        raise TypeError("name must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError("name must not be empty")
    if len(stripped) > _MAX_NAME:
        raise ValueError("name exceeds the supported length")
    return stripped


def _index(value):
    """Validate a positive integer layer index."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("index must be an integer, excluding bool")
    if value <= 0:
        raise ValueError("index must be positive")
    return int(value)


def _time(value, name):
    """Validate a finite composition or source time in seconds."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _blend_mode(value):
    """Normalize a blend mode member, token or AE label to its token."""
    return BlendMode.coerce(value).value


class Layer:
    """Base layer with AE timing, switches, parenting and transform rendering.

    Parameters
    ----------
    name : str, optional
        Non-empty layer name, also used as the deterministic expression identity.
    index : int, optional
        Positive AE-style layer index exposed to expressions. Defaults to 1.
    transform : Transform, optional
        Layer transform. A default ``Transform`` is created when omitted.
    in_point, out_point : float, optional
        Composition-time window, half open. ``out_point=None`` uses
        ``default_out_point``.
    start_time : float, optional
        Composition time at which playback begins at the source origin.
        Finite AV footage with negative stretch begins at the source end.
    stretch : float, optional
        Percent time stretch; negative values reverse the source.
    parent : Layer, optional
        Parent whose transform this layer inherits. Opacity is never inherited.
    enabled, solo, shy, locked, guide : bool, optional
        AE layer switches. ``enabled`` and ``solo`` affect rendering; ``shy``,
        ``locked`` and ``guide`` remain editor metadata in this workstream.
    blend_mode : BlendMode or str, optional
        Any of the 38 AE blend modes as a ``BlendMode`` member, token or AE
        label. Stored as the canonical token. Defaults to ``normal``.
    preserve_transparency : bool, optional
        AE's "Preserve Underlying Transparency" (T) switch.
    masks : MaskStack or sequence of Mask, optional
        Layer masks, applied to the source pixels before the transform.
    track_matte : TrackMatte, optional
        Matte layer and mode; applied by ``moviepy.ae.renderer.Renderer``.
    effects : EffectStack or sequence of AEEffect, optional
        Effects applied after the masks and before the transform.
    collapse_transformations, continuously_rasterize, motion_blur : bool, optional
        Flags stored for WS-03, WS-20 and WS-07 behavior.

    Notes
    -----
    Subclasses implement ``source_size`` and ``source_buffer``. Source
    time is ``(t - start_time) / (stretch / 100)``, so a 200 percent stretch
    plays the source at half speed and a negative stretch plays it backwards.

    Examples
    --------
    >>> from moviepy.ae.layers import Layer
    >>> layer = Layer(
    ...     "base", in_point=1.0, out_point=3.0, start_time=1.0, stretch=200.0
    ... )
    >>> layer.source_time(2.0)
    0.5
    >>> layer.is_active(3.0)
    False
    >>> layer.duration
    2.0
    """

    def __init__(
        self,
        name="Layer",
        *,
        index=1,
        transform=None,
        in_point=0.0,
        out_point=None,
        start_time=0.0,
        stretch=100.0,
        parent=None,
        enabled=True,
        solo=False,
        shy=False,
        locked=False,
        guide=False,
        blend_mode="normal",
        preserve_transparency=False,
        masks=None,
        track_matte=None,
        effects=None,
        collapse_transformations=False,
        continuously_rasterize=False,
        motion_blur=False,
    ):
        self.name = name
        self.index = index
        self.transform = Transform() if transform is None else transform
        self.start_time = start_time
        self.stretch = stretch
        self._out_point = None
        self.in_point = in_point
        self.out_point = out_point
        self._parent = None
        self.parent = parent
        self.enabled = enabled
        self.solo = solo
        self.shy = shy
        self.locked = locked
        self.guide = guide
        self.blend_mode = blend_mode
        self.preserve_transparency = preserve_transparency
        self.masks = MaskStack() if masks is None else masks
        self._track_matte = None
        self.track_matte = track_matte
        self.effects = EffectStack() if effects is None else effects
        self.collapse_transformations = collapse_transformations
        self.continuously_rasterize = continuously_rasterize
        self.motion_blur = motion_blur

    # -- validated attributes ----------------------------------------------- #

    @property
    def name(self):
        """Return the layer name used for expression identity."""
        return self._name

    @name.setter
    def name(self, value):
        self._name = _name(value)

    @property
    def index(self):
        """Return the positive AE-style layer index."""
        return self._index

    @index.setter
    def index(self, value):
        self._index = _index(value)

    @property
    def transform(self):
        """Return the layer transform."""
        return self._transform

    @transform.setter
    def transform(self, value):
        if not isinstance(value, Transform):
            raise TypeError("transform must be a Transform")
        self._transform = value

    @property
    def start_time(self):
        """Return the composition time of source time zero."""
        return self._start_time

    @start_time.setter
    def start_time(self, value):
        self._start_time = _time(value, "start_time")

    @property
    def stretch(self):
        """Return the percent time stretch; negative values reverse playback."""
        return self._stretch

    @stretch.setter
    def stretch(self, value):
        number = _time(value, "stretch")
        if number == 0.0:
            raise ValueError("stretch must not be zero")
        self._stretch = number

    @property
    def in_point(self):
        """Return the inclusive composition-time start."""
        return self._in_point

    @in_point.setter
    def in_point(self, value):
        number = _time(value, "in_point")
        # Mirror the out_point guard so an inverted window cannot be created by
        # choosing the other assignment order after construction.
        out_point = getattr(self, "_out_point", None)
        if out_point is not None and number > out_point:
            raise ValueError("in_point must not follow out_point")
        self._in_point = number

    @property
    def out_point(self):
        """Return the exclusive composition-time end."""
        return (
            self._out_point if self._out_point is not None else self.default_out_point()
        )

    @out_point.setter
    def out_point(self, value):
        resolved = None if value is None else _time(value, "out_point")
        if resolved is not None and resolved < self._in_point:
            raise ValueError("out_point must not precede in_point")
        self._out_point = resolved

    @property
    def parent(self):
        """Return the parent layer, or None."""
        return self._parent

    @parent.setter
    def parent(self, value):
        if value is not None and not isinstance(value, Layer):
            raise TypeError("parent must be a Layer or None")
        if value is not None:
            _reject_cycle(self, value)
        self._parent = value

    @property
    def enabled(self):
        """Return the video switch that gates rendering."""
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = validate_flag(value, "enabled")

    @property
    def solo(self):
        """Return whether this layer survives another layer's solo."""
        return self._solo

    @solo.setter
    def solo(self, value):
        self._solo = validate_flag(value, "solo")

    @property
    def shy(self):
        """Return the editor-only shy metadata flag."""
        return self._shy

    @shy.setter
    def shy(self, value):
        self._shy = validate_flag(value, "shy")

    @property
    def locked(self):
        """Return the editor-only locked metadata flag."""
        return self._locked

    @locked.setter
    def locked(self, value):
        self._locked = validate_flag(value, "locked")

    @property
    def guide(self):
        """Return the guide flag; render exclusion is a WS-03 renderer policy."""
        return self._guide

    @guide.setter
    def guide(self, value):
        self._guide = validate_flag(value, "guide")

    @property
    def blend_mode(self):
        """Return the canonical blend-mode token, e.g. ``"screen"``."""
        return self._blend_mode

    @blend_mode.setter
    def blend_mode(self, value):
        self._blend_mode = _blend_mode(value)

    @property
    def preserve_transparency(self):
        """Return the Preserve Underlying Transparency switch."""
        return self._preserve_transparency

    @preserve_transparency.setter
    def preserve_transparency(self, value):
        self._preserve_transparency = validate_flag(value, "preserve_transparency")

    @property
    def masks(self):
        """Return the layer's ``MaskStack``."""
        return self._masks

    @masks.setter
    def masks(self, value):
        if isinstance(value, MaskStack):
            self._masks = value
            return
        if isinstance(value, Mask):
            value = [value]
        self._masks = MaskStack(value)

    @property
    def track_matte(self):
        """Return the ``TrackMatte`` or None."""
        return self._track_matte

    @track_matte.setter
    def track_matte(self, value):
        if value is not None:
            if not isinstance(value, TrackMatte):
                raise TypeError("track_matte must be a TrackMatte or None")
            _reject_matte_cycle(self, value.layer)
        self._track_matte = value

    def set_track_matte(self, layer, mode="alpha", *, hide=True, **kwargs):
        """Use ``layer`` as this layer's track matte and return the binding.

        Like AE, the matte layer's video switch is turned off (``hide``) so it
        only acts as a matte; it still renders as a matte while switched off.
        Extra keywords (``coefficients``, ``linear``) go to ``TrackMatte``.
        """
        self.track_matte = TrackMatte(layer, mode, **kwargs)
        if hide:
            layer.enabled = False
        return self._track_matte

    def apply_masks(self, buffer, t, context=None):
        """Apply the mask stack to source pixels at composition time ``t``."""
        if not len(self._masks):
            return buffer
        bindings = dict(self.expression_bindings)
        return self._masks.apply(buffer, self.source_time(t), context, **bindings)

    @property
    def effects(self):
        """Return the layer's ``EffectStack``."""
        return self._effects

    @effects.setter
    def effects(self, value):
        if isinstance(value, EffectStack):
            self._effects = value
            return
        if isinstance(value, AEEffect):
            value = [value]
        self._effects = EffectStack(value)

    @property
    def is_adjustment(self):
        """Return whether effects apply to the layers below (adjustment layer)."""
        return False

    def apply_effects(self, buffer, t, context=None):
        """Run the effect stack on masked source pixels at composition time ``t``.

        Temporal effects fetch other times through the layer source and masks.
        """
        if not self._effects.active():
            return buffer
        bindings = dict(self.expression_bindings)
        return self._effects.apply(
            buffer,
            self.source_time(t),
            context,
            bindings=bindings,
            source_at=lambda local: self._masked_source(local, context),
        )

    def _masked_source(self, local_t, context=None):
        """Return source pixels with masks at layer (source-local) time."""
        buffer = self.source_buffer(local_t, context)
        if not len(self._masks):
            return buffer
        bindings = dict(self.expression_bindings)
        return self._masks.apply(buffer, local_t, context, **bindings)

    def prepared_source(self, t, context=None):
        """Return source pixels after masks and effects (before the transform).

        Adjustment layers return their masked coverage only: their effects act
        on the composite below and are run by the renderer.
        """
        buffer = self._masked_source(self.source_time(t), context)
        if self.is_adjustment or 0 in buffer.size:
            return buffer
        return self.apply_effects(buffer, t, context)

    @property
    def collapse_transformations(self):
        """Return the collapse flag whose behavior lands with WS-03."""
        return self._collapse_transformations

    @collapse_transformations.setter
    def collapse_transformations(self, value):
        self._collapse_transformations = validate_flag(
            value, "collapse_transformations"
        )

    @property
    def continuously_rasterize(self):
        """Return the vector rasterization flag whose behavior lands with WS-20."""
        return self._continuously_rasterize

    @continuously_rasterize.setter
    def continuously_rasterize(self, value):
        self._continuously_rasterize = validate_flag(value, "continuously_rasterize")

    @property
    def motion_blur(self):
        """Return the per-layer motion blur flag consumed by WS-07."""
        return self._motion_blur

    @motion_blur.setter
    def motion_blur(self, value):
        self._motion_blur = validate_flag(value, "motion_blur")

    # -- timing -------------------------------------------------------------- #

    def default_out_point(self):
        """Return the out point used when none was given explicitly."""
        return math.inf

    @property
    def duration(self):
        """Return the composition-time length of the layer window."""
        return self.out_point - self._in_point

    def source_time(self, t):
        """Map composition time onto source time with offset and stretch."""
        time = _time(t, "t")
        return (time - self._start_time) / (self._stretch / 100.0)

    def is_active(self, t):
        """Return whether the layer renders at composition time ``t``."""
        time = _time(t, "t")
        return self._enabled and self._in_point <= time < self.out_point

    def visible(self, t, *, solo_active=False):
        """Return whether the layer contributes when solo state is known."""
        if not self.is_active(t):
            return False
        return self._solo or not solo_active

    # -- hierarchy ----------------------------------------------------------- #

    @property
    def expression_bindings(self):
        """Return deterministic expression identity bindings for properties."""
        return {"index": self._index, "layer_id": self._name}

    def parent_chain(self):
        """Return ``(self, parent, ...)`` up to the root, rejecting cycles."""
        chain = []
        current = self
        seen = {id(current)}
        while current is not None:
            chain.append(current)
            current = current._parent
            if current is not None and id(current) in seen:
                raise ValueError("parent chain contains a cycle")
            if current is not None:
                seen.add(id(current))
        return tuple(chain)

    def local_matrix(self, t, context=None):
        """Return this layer's own matrix, ignoring parents.

        ``t`` is composition time. Like every other public layer entry point it
        is mapped through ``source_time`` first, because After Effects keys
        transform properties on the layer's own clock: moving ``start_time`` or
        ``stretch`` shifts and retimes the animation together with the footage.
        The source rectangle is assumed to start at its own origin, which holds
        for every layer type in this package.
        """
        bindings = dict(self.expression_bindings)
        bindings["context"] = context
        local_t = self.source_time(t)
        return self._transform.matrix_at(local_t, size=self.source_size, **bindings)

    def world_matrix(self, t, context=None):
        """Return the accumulated parent-to-child matrix at composition time.

        Each ancestor is evaluated at its own layer time, so a parent with an
        offset or a stretch stays in sync with the child it drives.
        """
        matrix = np.eye(3, dtype=np.float64)
        for layer in reversed(self.parent_chain()):
            matrix = matrix @ layer.local_matrix(t, context)
        return matrix

    def opacity_at(self, t, context=None):
        """Return this layer's clamped opacity factor, never inherited.

        ``t`` is composition time, mapped through ``source_time`` so a fade
        stays locked to the layer's own timeline.
        """
        bindings = dict(self.expression_bindings)
        bindings["context"] = context
        return self._transform.opacity_at(self.source_time(t), **bindings)

    # -- rendering ----------------------------------------------------------- #

    @property
    def source_size(self):
        """Return the nominal source (width, height) in pixels."""
        raise NotImplementedError("subclasses must define source_size")

    def source_buffer(self, t, context=None):
        """Return premultiplied source pixels at source-local time ``t``."""
        raise NotImplementedError("subclasses must define source_buffer")

    def render(self, t, context=None, *, bounds=None):
        """Return the transformed layer contribution, or None when inactive.

        Parameters
        ----------
        t : float
            Composition time in seconds.
        context : RenderContext, optional
            Supplies quality for ``auto`` interpolation and expression context.
        bounds : tuple of int, optional
            Exact destination rectangle for region-of-interest rendering.

        Returns
        -------
        Buffer or None
            Premultiplied world-space pixels. ``None`` means the layer window or
            its video switch excluded it; an empty buffer means it has no pixels.
        """
        if not self.is_active(t):
            return None
        buffer = self.prepared_source(t, context)
        if 0 in buffer.size:
            # Delegate so an explicit region of interest is honoured for empty
            # sources exactly as warp_buffer honours it for empty inputs.
            return warp_buffer(buffer, np.eye(3), 1.0, bounds=bounds)
        matrix = self.world_matrix(t, context)
        factor = self.opacity_at(t, context)
        interpolation = resolve_interpolation(self._transform.interpolation, context)
        return warp_buffer(
            buffer, matrix, factor, interpolation=interpolation, bounds=bounds
        )


def _reject_matte_cycle(layer, candidate):
    """Refuse a track matte that is the layer itself or loops back to it."""
    current, seen = candidate, set()
    while current is not None:
        if current is layer:
            raise ValueError("track matte chain contains a cycle")
        if id(current) in seen:
            return
        seen.add(id(current))
        matte = current.track_matte
        current = None if matte is None else matte.layer


def _reject_cycle(layer, candidate):
    """Refuse a parent assignment that would close the ancestor chain."""
    current = candidate
    seen = {id(layer)}
    while current is not None:
        if id(current) in seen:
            raise ValueError("parent chain contains a cycle")
        seen.add(id(current))
        current = current._parent
