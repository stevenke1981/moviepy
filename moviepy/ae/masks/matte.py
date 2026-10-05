"""Track mattes: use another layer's alpha or luma as this layer's alpha.

Examples
--------
>>> import numpy as np
>>> from moviepy.ae import Buffer
>>> from moviepy.ae.masks.matte import MatteMode, matte_values
>>> gray = Buffer(np.full((1, 1, 4), 0.5, dtype=np.float32) * [1, 1, 1, 2])
>>> float(matte_values(gray, MatteMode.LUMA)[0, 0])
0.5
>>> round(float(matte_values(gray, "luma", linear=True)[0, 0]), 4)
0.214
"""

from enum import Enum

import numpy as np

from moviepy.ae.blend._formulas import REC601_LUMA, REC709_LUMA


_COEFFICIENTS = {"rec709": REC709_LUMA, "rec601": REC601_LUMA}


class MatteMode(str, Enum):
    """AE track matte modes."""

    ALPHA = "alpha"
    ALPHA_INVERTED = "alpha_inverted"
    LUMA = "luma"
    LUMA_INVERTED = "luma_inverted"

    def __str__(self):
        return self.value

    @classmethod
    def coerce(cls, value):
        """Return the member for a member, token or label like "Luma Inverted"."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("matte mode must be a MatteMode or string")
        token = value.strip().lower().replace(" ", "_").replace("-", "_")
        try:
            return cls(token)
        except ValueError:
            raise ValueError(f"unknown matte mode {value!r}") from None

    @property
    def inverted(self):
        """Return whether the matte is inverted."""
        return self.value.endswith("_inverted")


def srgb_to_linear(values):
    """Apply the sRGB electro-optical transfer function (IEC 61966-2-1)."""
    values = np.asarray(values, dtype=np.float32)
    low = values / np.float32(12.92)
    high = ((values + np.float32(0.055)) / np.float32(1.055)) ** np.float32(2.4)
    return np.where(values <= np.float32(0.04045), low, high).astype(np.float32)


def matte_values(buffer, mode, *, coefficients="rec709", linear=False):
    """Return ``(H, W)`` matte opacity derived from a premultiplied buffer.

    Parameters
    ----------
    buffer : Buffer
        The matte layer, already transformed into composition space.
    mode : MatteMode or str
        Alpha, alpha inverted, luma or luma inverted.
    coefficients : {"rec709", "rec601"}, optional
        Luma weights. Rec.709 is the default (0.2126, 0.7152, 0.0722).
    linear : bool, optional
        Linearize sRGB colors before taking luma (linear-light projects).
        Mid gray 0.5 then yields 0.214 instead of 0.5.

    Returns
    -------
    numpy.ndarray
        float32 values in [0, 1]. Luma is the luma of the straight color
        multiplied by the matte alpha, so transparent areas are black.
    """
    member = MatteMode.coerce(mode)
    rgba = buffer.rgba
    if member in (MatteMode.ALPHA, MatteMode.ALPHA_INVERTED):
        values = rgba[..., 3].astype(np.float32, copy=True)
    else:
        if linear and buffer.color_space not in ("srgb", "linear"):
            raise ValueError("linear luma requires sRGB or linear sRGB pixels")
        values = _luma(rgba, coefficients, linear and buffer.color_space != "linear")
    np.clip(values, 0.0, 1.0, out=values)
    return np.float32(1.0) - values if member.inverted else values


def _luma(rgba, coefficients, linear):
    """Return premultiplied luma, optionally computed in linear light."""
    if coefficients not in _COEFFICIENTS:
        raise ValueError("coefficients must be 'rec709' or 'rec601'")
    weights = _COEFFICIENTS[coefficients]
    alpha = rgba[..., 3]
    if not linear:
        return (rgba[..., :3] @ weights).astype(np.float32)
    straight = np.zeros(rgba.shape[:2] + (3,), dtype=np.float32)
    np.divide(rgba[..., :3], alpha[..., None], out=straight, where=alpha[..., None] > 0)
    linear_rgb = srgb_to_linear(np.clip(straight, 0.0, 1.0))
    return ((linear_rgb @ weights) * alpha).astype(np.float32)


class TrackMatte:
    """Bind a matte layer and mode to a target layer.

    Parameters
    ----------
    layer : Layer
        Matte source in the same composition. Like AE, it is evaluated even
        when its video switch is off, but only inside its own time window.
    mode : MatteMode or str, optional
        Defaults to ``alpha``.
    coefficients : {"rec709", "rec601"}, optional
        Luma weights for luma modes.
    linear : bool, optional
        Compute luma in linear light.
    """

    def __init__(self, layer, mode="alpha", *, coefficients="rec709", linear=False):
        from moviepy.ae.layers.base import Layer

        if not isinstance(layer, Layer):
            raise TypeError("track matte layer must be a Layer")
        if coefficients not in _COEFFICIENTS:
            raise ValueError("coefficients must be 'rec709' or 'rec601'")
        if not isinstance(linear, bool):
            raise TypeError("linear must be bool")
        self.layer = layer
        self.mode = MatteMode.coerce(mode)
        self.coefficients = coefficients
        self.linear = linear

    def values(self, buffer):
        """Return matte opacity from the rendered matte buffer."""
        return matte_values(
            buffer, self.mode, coefficients=self.coefficients, linear=self.linear
        )
