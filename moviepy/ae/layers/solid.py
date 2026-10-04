"""Constant-color layer equivalent to an AE solid."""

import numpy as np

from moviepy.ae._geometry import validate_pixel_size
from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.base import Layer
from moviepy.ae.properties.values import finite_real


_COMPONENTS = 3
_CODE_RANGE = 255.0


def _color(value):
    """Validate 8-bit style RGB codes and return normalized components."""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise TypeError("color must be a sequence of three numbers")
    items = tuple(value)
    if len(items) != _COMPONENTS:
        raise ValueError("color must contain three components")
    normalized = []
    for item in items:
        code = finite_real(item, "color")
        if not 0.0 <= code <= _CODE_RANGE:
            raise ValueError("color components must be within 0..255")
        normalized.append(code / _CODE_RANGE)
    return tuple(normalized)


class SolidLayer(Layer):
    """Render one opaque constant color, like an AE solid layer.

    Parameters
    ----------
    name : str, optional
        Layer name. Defaults to ``Solid``.
    color : sequence of float, optional
        RGB codes in 0..255, matching the ``color=(20, 20, 30)`` convention of
        the specification examples. Defaults to black.
    size : tuple of int, optional
        Positive (width, height) in pixels. Defaults to ``(1920, 1080)``.
    ``**kwargs``
        Every ``Layer`` keyword.

    Notes
    -----
    Pixels are built once and cached because a solid never changes over time;
    assigning a new ``color`` or ``size`` rebuilds them. The cached buffer is
    immutable, so repeated renders share storage safely.

    Examples
    --------
    >>> from moviepy.ae.layers import SolidLayer
    >>> layer = SolidLayer("bg", color=(255, 0, 0), size=(2, 1))
    >>> layer.render(0.0).to_uint8_rgb().tolist()
    [[[255, 0, 0], [255, 0, 0]]]
    """

    def __init__(self, name="Solid", *, color=(0, 0, 0), size=(1920, 1080), **kwargs):
        super().__init__(name, **kwargs)
        self._cache = None
        self.size = size
        self.color = color

    @property
    def color(self):
        """Return the normalized RGB components in 0..1."""
        return self._color

    @color.setter
    def color(self, value):
        self._color = _color(value)
        self._cache = None

    @property
    def size(self):
        """Return the solid (width, height) in pixels."""
        return self._size

    @size.setter
    def size(self, value):
        self._size = validate_pixel_size(value)
        self._cache = None

    @property
    def source_size(self):
        """Return the solid size used for anchor resolution."""
        return self._size

    def source_buffer(self, t, context=None):
        """Return the cached opaque buffer for this solid."""
        if self._cache is None:
            self._cache = self._build()
        return self._cache

    def _build(self):
        """Materialize immutable premultiplied pixels for the current color."""
        width, height = self._size
        try:
            rgba = np.empty((height, width, 4), dtype=np.float32)
        except MemoryError as error:
            # validate_pixel_size already refuses areas past MAX_RENDER_PIXELS;
            # this only fires when a host raised that budget beyond the memory
            # actually available, and keeps the error contract clean.
            raise ValueError(
                f"solid of {width}x{height} does not fit in available memory"
            ) from error
        for channel in range(_COMPONENTS):
            rgba[..., channel] = np.float32(self._color[channel])
        rgba[..., 3] = np.float32(1.0)
        return Buffer(rgba)
