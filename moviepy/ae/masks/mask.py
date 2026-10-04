"""Layer masks: animated Bezier paths combined with AE mask modes.

Examples
--------
>>> import numpy as np
>>> from moviepy.ae import Buffer
>>> from moviepy.ae.masks import Mask, MaskStack
>>> stack = MaskStack([Mask.rect(center=(1.5, 1.5), size=(2, 2))])
>>> source = Buffer(np.ones((4, 4, 4), dtype=np.float32))
>>> stack.apply(source, 0.0).rgba[..., 3].tolist()[1]
[0.0, 1.0, 1.0, 0.0]
"""

from enum import Enum
from functools import lru_cache

import numpy as np

from moviepy.ae._geometry import (
    finite_real,
    to_property,
    validate_flag,
    validate_pixel_size,
)
from moviepy.ae.buffer import Buffer
from moviepy.ae.masks import path as shapes
from moviepy.ae.masks.rasterize import rasterize
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.values import PathValue


_CACHE_SIZE = 8


class MaskMode(str, Enum):
    """AE mask modes, applied top to bottom through a ``MaskStack``."""

    NONE = "none"
    ADD = "add"
    SUBTRACT = "subtract"
    INTERSECT = "intersect"
    LIGHTEN = "lighten"
    DARKEN = "darken"
    DIFFERENCE = "difference"

    def __str__(self):
        return self.value

    @classmethod
    def coerce(cls, value):
        """Return the member for a member or case-insensitive token."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("mode must be a MaskMode or string")
        try:
            return cls(value.strip().lower())
        except ValueError:
            raise ValueError(f"unknown mask mode {value!r}") from None


def _is_static(prop):
    """Return whether a Property can only ever produce its constant value."""
    return (
        not prop.keyframes
        and getattr(prop, "_expression", None) is None
        and getattr(prop, "_callable", None) is None
        and getattr(prop, "_dimensions", None) is None
        and getattr(prop, "_projection", None) is None
    )


def _path_property(value):
    """Coerce a PathValue, keyframes, callable or Property into a path Property."""
    if isinstance(value, PathValue):
        return Property(value)
    return to_property(value, "path", "path")


@lru_cache(maxsize=_CACHE_SIZE)
def _cached_coverage(path, bounds, feather, expansion, supersample):
    """Rasterize once per distinct static input; results are read-only.

    At most ``_CACHE_SIZE`` coverages are kept (8 x 8 MiB for 1080p masks).
    """
    coverage = rasterize(
        path, bounds, feather=feather, expansion=expansion, supersample=supersample
    )
    coverage.setflags(write=False)
    return coverage


def _scale_path(path, scale):
    """Scale pixel-center vertices and relative tangents into render space."""
    if scale == 1.0:
        return path
    vertices = tuple(tuple((v + 0.5) * scale - 0.5 for v in p) for p in path.vertices)
    incoming = tuple(tuple(v * scale for v in p) for p in path.in_tangents)
    outgoing = tuple(tuple(v * scale for v in p) for p in path.out_tangents)
    return PathValue(vertices, incoming, outgoing, path.closed)


class Mask:
    """One AE layer mask.

    Parameters
    ----------
    path : PathValue, Property or keyframes
        Bezier path in layer pixel coordinates (pixel-center convention).
        Paths with equal vertex counts animate point by point.
    mode : MaskMode or str, optional
        ``add`` (default), ``subtract``, ``intersect``, ``lighten``,
        ``darken``, ``difference`` or ``none``.
    opacity : float or Property, optional
        Percent 0..100.
    feather : (float, float) or Property, optional
        Horizontal and vertical feather in pixels; ``sigma = feather / 2``.
    expansion : float or Property, optional
        Pixels to grow (positive) or shrink (negative) the shape.
    inverted : bool, optional
        Invert the mask after feathering.
    name : str, optional
        Display name.

    Notes
    -----
    All animated fields are evaluated at layer time (after ``start_time`` and
    ``stretch``), like transform properties. Static masks are rasterized once
    and cached.
    """

    def __init__(
        self,
        path,
        mode="add",
        *,
        opacity=100.0,
        feather=(0.0, 0.0),
        expansion=0.0,
        inverted=False,
        name="Mask",
    ):
        self.path = path
        self.mode = mode
        self.opacity = opacity
        self.feather = feather
        self.expansion = expansion
        self.inverted = inverted
        self.name = name

    # -- convenience constructors ----------------------------------------- #

    @classmethod
    def rect(cls, center, size, **kwargs):
        """Return a rectangular mask from its center and (width, height)."""
        return cls(shapes.rect(center, size), **kwargs)

    @classmethod
    def ellipse(cls, center, size, **kwargs):
        """Return an elliptical mask from its center and bounding size."""
        return cls(shapes.ellipse(center, size), **kwargs)

    @classmethod
    def rounded_rect(cls, center, size, roundness, **kwargs):
        """Return a rounded rectangle mask with corner radius ``roundness``."""
        return cls(shapes.rounded_rect(center, size, roundness), **kwargs)

    @classmethod
    def polygon(cls, points, **kwargs):
        """Return a straight-edged polygon mask."""
        return cls(shapes.polygon(points), **kwargs)

    @classmethod
    def star(cls, center, points, outer_radius, inner_radius, rotation=0.0, **kw):
        """Return a star mask (see ``moviepy.ae.masks.path.star``)."""
        return cls(
            shapes.star(center, points, outer_radius, inner_radius, rotation), **kw
        )

    @classmethod
    def from_svg_path(cls, d, **kwargs):
        """Return a mask from a single-subpath SVG ``d`` string."""
        return cls(shapes.from_svg_path(d), **kwargs)

    # -- validated fields ---------------------------------------------------- #

    @property
    def path(self):
        """Return the path Property."""
        return self._path

    @path.setter
    def path(self, value):
        self._path = _path_property(value)

    @property
    def mode(self):
        """Return the ``MaskMode``."""
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = MaskMode.coerce(value)

    @property
    def opacity(self):
        """Return the opacity Property (percent)."""
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        self._opacity = to_property(value, "opacity", "float", (0.0, 100.0))

    @property
    def feather(self):
        """Return the feather Property (x, y pixels)."""
        return self._feather

    @feather.setter
    def feather(self, value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = (float(value), float(value))
        self._feather = to_property(value, "feather", "vec2")

    @property
    def expansion(self):
        """Return the expansion Property (pixels)."""
        return self._expansion

    @expansion.setter
    def expansion(self, value):
        self._expansion = to_property(value, "expansion", "float")

    @property
    def inverted(self):
        """Return whether the mask is inverted."""
        return self._inverted

    @inverted.setter
    def inverted(self, value):
        self._inverted = validate_flag(value, "inverted")

    # -- evaluation ------------------------------------------------------------ #

    def coverage(self, bounds, t=0.0, context=None, *, pixel_scale=1.0, **bindings):
        """Return float32 mask values in [0, 1] over a world rectangle.

        The result includes feather, expansion, inversion and opacity.
        ``pixel_scale`` converts authored mask coordinates and lengths to
        output pixels; source-space masks use the default of 1.
        """
        pixel_scale = finite_real(pixel_scale, "pixel_scale")
        if pixel_scale <= 0:
            raise ValueError("pixel_scale must be positive")
        evaluation = dict(bindings)
        evaluation["context"] = context
        path = _scale_path(self._path.value_at(t, **evaluation), pixel_scale)
        feather = tuple(
            max(0.0, v) * pixel_scale for v in self._feather.value_at(t, **evaluation)
        )
        expansion = float(self._expansion.value_at(t, **evaluation)) * pixel_scale
        opacity = min(1.0, max(0.0, self._opacity.value_at(t, **evaluation) / 100.0))
        draft = context is not None and context.quality == "draft"
        arguments = (
            path,
            tuple(int(v) for v in bounds),
            feather,
            expansion,
            2 if draft else 4,
        )
        static = all(
            _is_static(prop) for prop in (self._path, self._feather, self._expansion)
        )
        # Only static masks are cached: animated ones would just churn memory.
        rasterizer = _cached_coverage if static else _cached_coverage.__wrapped__
        values = rasterizer(*arguments)
        if self._inverted:
            values = 1.0 - values
        if opacity != 1.0:
            values = values * np.float32(opacity)
        return values

    def to_clip(self, size, duration=None, fps=None):
        """Return a MoviePy mask clip (``is_mask=True``) rendering this mask."""
        return MaskStack([self]).to_clip(size, duration, fps)


def combine(accumulated, mask, mode):
    """Combine accumulated coverage with one mask using AE mask-mode rules."""
    if mode is MaskMode.ADD:
        return accumulated + mask - accumulated * mask
    if mode is MaskMode.SUBTRACT:
        return accumulated * (1.0 - mask)
    if mode is MaskMode.INTERSECT:
        return accumulated * mask
    if mode is MaskMode.LIGHTEN:
        return np.maximum(accumulated, mask)
    if mode is MaskMode.DARKEN:
        return np.minimum(accumulated, mask)
    if mode is MaskMode.DIFFERENCE:
        return np.abs(accumulated - mask)
    return accumulated


class MaskStack:
    """Ordered masks of one layer, combined top to bottom.

    The first active mask decides the starting state, as in AE: a stack
    starting with ``subtract``, ``intersect`` or ``darken`` starts from a
    fully opaque layer, otherwise from nothing. ``none`` masks are ignored,
    and a stack without active masks leaves the layer untouched. For binary
    coverage the modes reduce to the Boolean union, difference, intersection
    and symmetric difference.
    """

    _OPAQUE_START = (MaskMode.SUBTRACT, MaskMode.INTERSECT, MaskMode.DARKEN)

    def __init__(self, masks=()):
        self._masks = []
        for mask in masks:
            self.add(mask)

    def __iter__(self):
        return iter(tuple(self._masks))

    def __len__(self):
        return len(self._masks)

    def __getitem__(self, index):
        return self._masks[index]

    def add(self, mask):
        """Append a mask and return it."""
        if not isinstance(mask, Mask):
            raise TypeError("mask must be a Mask")
        self._masks.append(mask)
        return mask

    def remove(self, mask):
        """Remove a mask."""
        self._masks = [item for item in self._masks if item is not mask]

    def clear(self):
        """Remove every mask."""
        self._masks = []

    def active(self):
        """Return the masks whose mode is not ``none``."""
        return [mask for mask in self._masks if mask.mode is not MaskMode.NONE]

    def coverage(self, bounds, t=0.0, context=None, *, pixel_scale=1.0, **bindings):
        """Return combined float32 coverage, or ``None`` without active masks."""
        masks = self.active()
        if not masks:
            return None
        left, top, right, bottom = bounds
        start = 1.0 if masks[0].mode in self._OPAQUE_START else 0.0
        result = np.full((bottom - top, right - left), start, dtype=np.float32)
        for mask in masks:
            values = mask.coverage(
                bounds, t, context, pixel_scale=pixel_scale, **bindings
            )
            result = combine(result, values, mask.mode)
        return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)

    def apply(self, buffer, t=0.0, context=None, **bindings):
        """Multiply a premultiplied buffer by the combined mask coverage."""
        if not isinstance(buffer, Buffer):
            raise TypeError("buffer must be a Buffer")
        if 0 in buffer.size:
            return buffer
        values = self.coverage(buffer.bounds, t, context, **bindings)
        if values is None:
            return buffer
        rgba = buffer.rgba * values[..., None]
        return Buffer._publish(rgba, buffer.offset, buffer.color_space)

    def to_clip(self, size, duration=None, fps=None):
        """Return a MoviePy mask clip of the combined coverage over time."""
        from moviepy.video.VideoClip import VideoClip

        width, height = validate_pixel_size(size)

        def frame_function(t):
            values = self.coverage((0, 0, width, height), t)
            if values is None:
                return np.ones((height, width), dtype=np.float64)
            return values.astype(np.float64)

        clip = VideoClip(is_mask=True, duration=duration)
        clip.frame_function = frame_function
        clip.size = (width, height)
        if fps is not None:
            clip.fps = fps
        return clip
