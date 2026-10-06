"""AE-style compositions: a layer stack that is also a MoviePy ``VideoClip``.

Examples
--------
>>> from moviepy.ae import Composition
>>> comp = Composition(size=(4, 2), fps=10, duration=1, bg_color=(0, 0, 255))
>>> bg = comp.add_solid("bg", color=(255, 0, 0), size=(2, 2))
>>> comp.get_frame(0.0)[0].tolist()
[[255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]]
>>> [layer.name for layer in comp.layers], comp.layer(1) is bg
(['bg'], True)
"""

import copy as _copy
import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from moviepy.ae._geometry import validate_flag, validate_pixel_size
from moviepy.ae.color import display_rgba, quantize_rgba
from moviepy.ae.context import RenderContext
from moviepy.ae.layers.adjustment import AdjustmentLayer
from moviepy.ae.layers.av import AVLayer
from moviepy.ae.layers.base import Layer, _name
from moviepy.ae.layers.comp import CompLayer
from moviepy.ae.layers.null import NullLayer
from moviepy.ae.layers.solid import SolidLayer, _color
from moviepy.ae.properties.values import finite_real
from moviepy.ae.renderer import Renderer
from moviepy.video.VideoClip import VideoClip


_MAX_TEXT = 4096


def _text(value, name):
    """Validate a bounded marker text field."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if len(value) > _MAX_TEXT:
        raise ValueError(f"{name} exceeds the supported length")
    return value


@dataclass(frozen=True)
class Marker:
    """A composition marker, like AE's comp markers.

    Parameters
    ----------
    time : float
        Composition time in seconds.
    comment : str, optional
        Free text shown on the marker.
    duration : float, optional
        Nonnegative marker span in seconds.
    chapter : str, optional
        Chapter name exported to formats that support chapters.
    url : str, optional
        Web link associated with the marker.
    """

    time: float
    comment: str = ""
    duration: float = 0.0
    chapter: str = ""
    url: str = ""

    def __post_init__(self):
        object.__setattr__(self, "time", finite_real(self.time, "time"))
        duration = finite_real(self.duration, "duration")
        if duration < 0:
            raise ValueError("duration must be nonnegative")
        object.__setattr__(self, "duration", duration)
        for name in ("comment", "chapter", "url"):
            _text(getattr(self, name), name)

    @property
    def end(self):
        """Return ``time + duration``."""
        return self.time + self.duration


def _positive(value, name):
    """Validate a finite positive real number."""
    number = finite_real(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


class Composition(VideoClip):
    """A composition of AE layers that renders like any MoviePy clip.

    Parameters
    ----------
    size : tuple of int, optional
        Frame (width, height) in pixels.
    fps : float, optional
        Composition frame rate. Precomposed children sample time on their
        own frame grid.
    duration : float, optional
        Length in seconds.
    name : str, optional
        Composition name used by ``CompLayer`` defaults.
    bg_color : sequence of float, optional
        Background RGB codes (0..255). Like AE, the background shows in RGB
        output but is not part of the composition alpha.
    transparent : bool, optional
        When true, frames are straight (unmatted) RGB and ``mask`` carries the
        composition alpha, so ``write_videofile`` with an alpha codec or a
        parent ``CompositeVideoClip`` sees the transparency.
    work_area : tuple of float, optional
        (start, end) seconds; defaults to the whole duration.
    renderer : Renderer, optional
        Renderer used for frames; defaults to ``Renderer()``.
    context : RenderContext, optional
        Template supplying ``quality``, ``resolution_scale`` and ``rng_seed``
        for ``get_frame``. Frames are always full size; a reduced
        ``resolution_scale`` renders fewer pixels and is upscaled back.

    Notes
    -----
    ``layers`` lists layers top first, like the AE timeline: index 1 is the
    top layer. ``add_*`` methods insert the new layer on top, as AE does.
    Layers are mutable objects shared with clips derived through ``with_*``
    methods, mirroring how MoviePy shares ``frame_function``.
    """

    def __init__(
        self,
        size=(1920, 1080),
        fps=30.0,
        duration=10.0,
        *,
        name="Comp",
        bg_color=(0, 0, 0),
        transparent=False,
        work_area=None,
        renderer=None,
        context=None,
        motion_blur=None,
    ):
        super().__init__(duration=_positive(duration, "duration"))
        self.size = validate_pixel_size(size)
        self.fps = _positive(fps, "fps")
        self.name = _name(name)
        self.bg_color = bg_color
        self._layers = []
        self._markers = []
        self.renderer = Renderer() if renderer is None else renderer
        self.context = RenderContext() if context is None else context
        self.motion_blur = (
            self.context.shutter is not None if motion_blur is None else motion_blur
        )
        self.work_area = work_area
        self.frame_function = self._rgb_frame
        self.transparent = transparent
        self._auto_audio = True

    def __copy__(self):
        """Materialize audio before MoviePy changes a derived clip's timeline.

        MoviePy's decorators transform audio after editing video duration. The
        mixer must retain the original timeline until those audio transforms
        run, just like an ordinary VideoClip's attached track.
        """
        result = super().__copy__()
        result.audio = _copy.copy(self.audio)
        return result

    copy = __copy__

    # -- validated attributes ------------------------------------------------ #

    @property
    def audio(self):
        """Mix current AVLayer audio unless MoviePy supplied an explicit track."""
        if not getattr(self, "_auto_audio", False):
            return self._audio_override
        from moviepy.ae.audio import composition_audio

        return composition_audio(self)

    @audio.setter
    def audio(self, value):
        self._audio_override = value
        self._auto_audio = False

    def use_layer_audio(self):
        """Restore automatic layer mixing after a manual audio override."""
        self._auto_audio = True
        return self

    @property
    def motion_blur(self):
        """Return the composition shutter switch (off unless explicitly enabled).

        Providing a shutter in the initial context also opts in. Assign False
        to disable it while retaining all layer switches and shutter settings.
        """
        return self._motion_blur

    @motion_blur.setter
    def motion_blur(self, value):
        self._motion_blur = validate_flag(value, "motion_blur")

    @property
    def bg_color(self):
        """Return the background as RGB codes in 0..255."""
        return tuple(code * 255.0 for code in self._bg)

    @bg_color.setter
    def bg_color(self, value):
        self._bg = _color(value)

    @property
    def transparent(self):
        """Return whether frames carry the composition alpha as a mask."""
        return self._transparent

    @transparent.setter
    def transparent(self, value):
        self._transparent = validate_flag(value, "transparent")
        if self._transparent:
            mask = VideoClip(is_mask=True, duration=self.duration)
            mask.frame_function = self._alpha_frame
            mask.size, mask.fps = self.size, self.fps
            self.mask = mask
        else:
            self.mask = None

    @property
    def renderer(self):
        """Return the renderer used for frames."""
        return self._renderer

    @renderer.setter
    def renderer(self, value):
        if not isinstance(value, Renderer):
            raise TypeError("renderer must be a Renderer")
        self._renderer = value

    @property
    def context(self):
        """Return the render settings template used by ``get_frame``."""
        return self._context

    @context.setter
    def context(self, value):
        if not isinstance(value, RenderContext):
            raise TypeError("context must be a RenderContext")
        self._context = value

    @property
    def work_area(self):
        """Return the (start, end) work area in seconds."""
        return self._work_area

    @work_area.setter
    def work_area(self, value):
        if value is None:
            self._work_area = (0.0, float(self.duration))
            return
        start, end = (finite_real(item, "work_area") for item in value)
        if not 0.0 <= start < end <= self.duration:
            raise ValueError("work_area must satisfy 0 <= start < end <= duration")
        self._work_area = (start, end)

    # -- layers ---------------------------------------------------------------- #

    @property
    def layers(self):
        """Return the layers, top first (index 1 first)."""
        return tuple(self._layers)

    def add_layer(self, layer, index=1):
        """Insert an existing layer, by default on top, and return it."""
        if not isinstance(layer, Layer):
            raise TypeError("layer must be a Layer")
        if any(existing is layer for existing in self._layers):
            raise ValueError("layer is already in this composition")
        if isinstance(layer, CompLayer):
            _reject_comp_cycle(self, layer.comp)
        position = self._position(index, len(self._layers) + 1)
        self._layers.insert(position, layer)
        self._reindex()
        return layer

    def add_clip(self, clip, name=None, **kwargs):
        """Wrap a MoviePy clip as an ``AVLayer`` on top of the stack."""
        return self.add_layer(AVLayer(clip, name, **kwargs))

    def add_solid(self, name="Solid", *, color=(0, 0, 0), size=None, **kwargs):
        """Add a ``SolidLayer``; its size defaults to the composition size."""
        size = self.size if size is None else size
        return self.add_layer(SolidLayer(name, color=color, size=size, **kwargs))

    def add_adjustment(self, name="Adjustment Layer", *, size=None, **kwargs):
        """Add an ``AdjustmentLayer`` covering the composition by default."""
        size = self.size if size is None else size
        return self.add_layer(AdjustmentLayer(name, size=size, **kwargs))

    def add_text(self, text, name="Text", **kwargs):
        """Add a raster title with optional deterministic entrance animation."""
        from moviepy.ae.text import TextLayer

        return self.add_layer(TextLayer(text, name, **kwargs))

    def add_particles(self, name="Particles", *, size=None, **kwargs):
        """Add an analytic particle burst/spray on a fixed canvas."""
        from moviepy.ae.particles import ParticleLayer

        return self.add_layer(
            ParticleLayer(name, size=self.size if size is None else size, **kwargs)
        )

    def add_null(self, name="Null", **kwargs):
        """Add a ``NullLayer`` controller."""
        return self.add_layer(NullLayer(name, **kwargs))

    def add_comp(self, comp, name=None, **kwargs):
        """Nest another composition (pre-compose) as a ``CompLayer``."""
        return self.add_layer(CompLayer(comp, name, **kwargs))

    def layer(self, key):
        """Return a layer by 1-based index or by (first matching) name."""
        if isinstance(key, Integral) and not isinstance(key, bool):
            if not 1 <= key <= len(self._layers):
                raise IndexError("layer index out of range")
            return self._layers[key - 1]
        if isinstance(key, str):
            for layer in self._layers:
                if layer.name == key:
                    return layer
            raise KeyError(f"no layer named {key!r}")
        raise TypeError("layer key must be an index or a name")

    def remove_layer(self, key):
        """Remove a layer (object, index or name) and return it."""
        layer = key if isinstance(key, Layer) else self.layer(key)
        self._layers = [item for item in self._layers if item is not layer]
        if any(item.parent is layer for item in self._layers):
            for item in self._layers:
                if item.parent is layer:
                    item.parent = None
        self._reindex()
        return layer

    def move_layer(self, key, index):
        """Move a layer to a new 1-based index (1 = top)."""
        layer = key if isinstance(key, Layer) else self.layer(key)
        if not any(item is layer for item in self._layers):
            raise ValueError("layer is not in this composition")
        self._layers = [item for item in self._layers if item is not layer]
        self._layers.insert(self._position(index, len(self._layers) + 1), layer)
        self._reindex()
        return layer

    @staticmethod
    def _position(index, limit):
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("index must be an integer")
        if not 1 <= index <= limit:
            raise IndexError("layer index out of range")
        return int(index) - 1

    def _reindex(self):
        for number, layer in enumerate(self._layers, start=1):
            layer.index = number

    # -- markers ----------------------------------------------------------------- #

    @property
    def markers(self):
        """Return composition markers sorted by time."""
        return tuple(self._markers)

    def add_marker(self, time, comment="", **kwargs):
        """Create, store and return a ``Marker``."""
        marker = Marker(time, comment, **kwargs)
        self._markers.append(marker)
        self._markers.sort(key=lambda item: item.time)
        return marker

    def work_area_clip(self):
        """Return the composition trimmed to its work area."""
        return self.subclipped(*self._work_area)

    # -- rendering ----------------------------------------------------------------- #

    def render_buffer(self, t, context=None, *, bounds=None, clip=True):
        """Render premultiplied pixels at composition time ``t``.

        See ``Renderer.render`` for the parameters. ``context`` defaults to the
        composition's ``context`` template.
        """
        context = self._context if context is None else context
        return self._renderer.render(self, t, context, bounds=bounds, clip=clip)

    def _frame_buffer(self, t):
        """Render one full-size frame.

        Layers, properties and wrapped clips can change between RGB and mask
        requests, even at the same time. Each request therefore renders from
        current inputs; Renderer still reuses work within that single render.
        """
        time = float(t)
        buffer = self.render_buffer(time)
        scale = self._context.resolution_scale
        if scale != 1.0:
            buffer = _upscale(buffer, self.size)
        return buffer

    def _rgb_frame(self, t):
        buffer = self._frame_buffer(t)
        if buffer.color_space != "srgb":
            pixels = display_rgba(
                buffer, background=None if self._transparent else self._bg
            )
            return quantize_rgba(pixels, bits=8)[..., :3]
        if self._transparent:
            return buffer.to_uint8_rgb()
        return buffer.to_uint8_rgb(bg=self._bg)

    def _alpha_frame(self, t):
        alpha = self._frame_buffer(t).rgba[..., 3]
        return np.clip(alpha, 0.0, 1.0).astype(np.float64)


def _upscale(buffer, size):
    """Resize a reduced-resolution render back to the full frame size."""
    import cv2

    array = cv2.resize(buffer.rgba, size, interpolation=cv2.INTER_LINEAR)
    np.clip(array[..., 3], 0, 1, out=array[..., 3])
    return type(buffer)(array, color_space=buffer.color_space)


def _reject_comp_cycle(parent, child):
    """Refuse nesting that would make a composition contain itself."""
    pending, seen = [child], set()
    while pending:
        current = pending.pop()
        if current is parent:
            raise ValueError("nesting would make a composition contain itself")
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(
            layer.comp for layer in current.layers if isinstance(layer, CompLayer)
        )


def from_moviepy(clip, *, name="Comp"):
    """Convert a ``CompositeVideoClip`` into an equivalent ``Composition``.

    Each child clip becomes an ``AVLayer`` with its start time, end time,
    position (static, named or animated) and mask. Layer order, background
    color and transparency follow the source clip. This is a best-effort
    conversion: clip-level effects are kept because the clip frames are used
    as-is, but MoviePy-specific behaviors without an AE equivalent (for
    example a resizing background clip) are not translated.

    Parameters
    ----------
    clip : CompositeVideoClip
        Source composite.
    name : str, optional
        Name of the resulting composition.

    Returns
    -------
    Composition
    """
    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

    if not isinstance(clip, CompositeVideoClip):
        raise TypeError("clip must be a CompositeVideoClip")
    if clip.is_mask:
        raise ValueError("mask composites cannot be converted")
    duration = (
        clip.duration
        if clip.duration
        else max((child.end or 0.0 for child in clip.clips), default=1.0)
    )
    background = clip.bg_color if clip.created_bg else (0, 0, 0)
    transparent = clip.created_bg and len(background) == 4
    comp = Composition(
        size=tuple(int(item) for item in clip.size),
        fps=clip.fps or 30.0,
        duration=duration,
        name=name,
        bg_color=tuple(background[:3]),
        transparent=transparent or clip.mask is not None and not clip.created_bg,
    )
    children = list(clip.clips) if clip.created_bg else [clip.bg, *clip.clips]
    for child in children:
        comp.add_layer(_legacy_layer(child, comp.size))
    return comp


def _legacy_layer(child, comp_size):
    """Translate one legacy clip and its position function into an AVLayer."""
    from moviepy.ae.properties.property import Property
    from moviepy.ae.transform import Transform
    from moviepy.tools import compute_position

    width, height = (int(item) for item in child.size)

    def position(t):
        x, y = compute_position(
            (width, height), comp_size, child.pos(t), child.relative_pos
        )
        return (float(x), float(y))

    start = float(child.start or 0.0)
    end = child.end
    transform = Transform(
        anchor_point=(0.0, 0.0),
        position=Property(position, value_type="vec2"),
        interpolation="linear",
    )
    return AVLayer(
        child,
        getattr(child, "name", None) or None,
        transform=transform,
        in_point=start,
        out_point=None if end is None or not math.isfinite(end) else float(end),
        start_time=start,
    )
