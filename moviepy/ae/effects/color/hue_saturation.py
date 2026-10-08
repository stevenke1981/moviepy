"""Hue/Saturation (Color Correction)."""

import numpy as np

from moviepy.ae.effects._pixels import map_straight
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


def _rgb_to_hsl(rgb):
    """Return hue (turns, 0..1), saturation and lightness of straight RGB."""
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    high = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    chroma = high - low
    light = (high + low) * np.float32(0.5)
    denom = 1.0 - np.abs(2.0 * light - 1.0)
    sat = np.divide(chroma, denom, out=np.zeros_like(chroma), where=denom > 1e-7)
    safe = np.where(chroma > 0, chroma, np.float32(1.0))
    hue = np.where(
        high == red,
        (green - blue) / safe,
        np.where(high == green, (blue - red) / safe + 2.0, (red - green) / safe + 4.0),
    )
    hue = np.where(chroma > 0, hue / 6.0, 0.0) % 1.0
    return hue, np.clip(sat, 0.0, 1.0), light


def _hsl_to_rgb(hue, sat, light):
    """Convert HSL (hue in turns) back to straight RGB."""
    chroma = (1.0 - np.abs(2.0 * light - 1.0)) * sat
    h6 = (hue % 1.0) * 6.0
    second = chroma * (1.0 - np.abs(h6 % 2.0 - 1.0))
    base = light - chroma * 0.5
    sector = np.floor(h6).astype(np.int64) % 6
    zero = np.zeros_like(chroma)
    table = (
        (chroma, second, zero),
        (second, chroma, zero),
        (zero, chroma, second),
        (zero, second, chroma),
        (second, zero, chroma),
        (chroma, zero, second),
    )
    conditions = [sector == k for k in range(6)]
    channels = [np.select(conditions, [entry[c] for entry in table]) for c in range(3)]
    return np.stack([channel + base for channel in channels], axis=-1)


def _adjust(value, percent):
    """AE-style signed adjustment: positive moves toward 1, negative toward 0."""
    factor = percent / 100.0
    if factor >= 0:
        return value + (1.0 - value) * factor
    return value * (1.0 + factor)


@register
class HueSaturation(AEEffect):
    """Rotate hue and adjust saturation and lightness, optionally colorizing.

    ====================  ====================  ===========================
    Parameter             AE panel name         Notes
    ====================  ====================  ===========================
    master_hue            Master Hue            degrees, default 0
    master_saturation     Master Saturation     -100..100 %, default 0
    master_lightness      Master Lightness      -100..100 %, default 0
    colorize              Colorize              bool, default off
    colorize_hue          Colorize Hue          degrees, default 0
    colorize_saturation   Colorize Saturation   0..100 %, default 25
    colorize_lightness    Colorize Lightness    -100..100 %, default 0
    ====================  ====================  ===========================

    Notes
    -----
    Pixels are converted to HSL on straight color. With ``colorize`` on,
    every pixel (including grays) takes ``colorize_hue`` and
    ``colorize_saturation``, then ``colorize_lightness`` is applied; the
    master adjustments act on that result. Positive saturation or lightness
    moves a channel toward 1 by the given fraction of the headroom, negative
    values scale it toward 0. The math runs on the buffer's encoded values
    (sRGB codes for sRGB projects), not on linear light, and results are
    clipped to the 0..1 working range. This is a documented HSL model, not
    Adobe's exact curve.
    """

    name = "Hue/Saturation"
    category = "Color Correction"
    PARAMS = (
        Param("master_hue", "float", 0.0, (-3600.0, 3600.0)),
        Param("master_saturation", "float", 0.0, (-100.0, 100.0)),
        Param("master_lightness", "float", 0.0, (-100.0, 100.0)),
        Param("colorize", "bool", False),
        Param("colorize_hue", "float", 0.0, (-3600.0, 3600.0)),
        Param("colorize_saturation", "float", 25.0, (0.0, 100.0)),
        Param("colorize_lightness", "float", 0.0, (-100.0, 100.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Apply the HSL adjustments to straight color."""
        values = self.values_at(t, context) if values is None else values
        colorize = bool(values["colorize"])
        master = (
            values["master_hue"],
            values["master_saturation"],
            values["master_lightness"],
        )
        if not colorize and not any(master):
            return src

        def adjust(rgb):
            hue, sat, light = _rgb_to_hsl(rgb)
            if colorize:
                hue = np.full_like(hue, (values["colorize_hue"] / 360.0) % 1.0)
                sat = np.full_like(sat, values["colorize_saturation"] / 100.0)
                light = _adjust(light, values["colorize_lightness"])
            hue = (hue + values["master_hue"] / 360.0) % 1.0
            sat = np.clip(_adjust(sat, values["master_saturation"]), 0.0, 1.0)
            light = np.clip(_adjust(light, values["master_lightness"]), 0.0, 1.0)
            return _hsl_to_rgb(hue, sat, light)

        return map_straight(src, adjust)
