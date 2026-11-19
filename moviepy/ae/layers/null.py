"""Invisible controller layer used as a transform parent."""

import numpy as np

from moviepy.ae._geometry import validate_pixel_size
from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.base import Layer


class NullLayer(Layer):
    """Provide a transform controller that contributes no pixels.

    Parameters
    ----------
    name : str, optional
        Layer name. Defaults to ``Null``.
    size : tuple of int, optional
        Nominal (width, height) used to resolve an automatic anchor point.
        Defaults to ``(100, 100)``, matching the AE null object.
    ``**kwargs``
        Every ``Layer`` keyword.

    Notes
    -----
    ``render`` returns an empty buffer, which composites as an identity, so a
    null can sit anywhere in the layer stack. Children inherit its transform
    but never its opacity.

    Examples
    --------
    >>> from moviepy.ae.layers import NullLayer
    >>> null = NullLayer("controller", size=(50, 50))
    >>> null.source_size, null.render(0.0).size
    ((50, 50), (0, 0))
    >>> null.transform.resolve_anchor((50, 50))
    (24.5, 24.5)
    """

    def __init__(self, name="Null", *, size=(100, 100), **kwargs):
        super().__init__(name, **kwargs)
        self._size = validate_pixel_size(size)
        self._empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))

    @property
    def size(self):
        """Return the nominal (width, height) used for anchor resolution."""
        return self._size

    @size.setter
    def size(self, value):
        self._size = validate_pixel_size(value)

    @property
    def source_size(self):
        """Return the nominal null size."""
        return self._size

    def source_buffer(self, t, context=None):
        """Return a shared empty buffer because a null has no pixels."""
        return self._empty
