"""Periodic wave displacement (Distort)."""

import math

import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.distort._sampling import sample_bilinear as _sample
from moviepy.ae.effects.registry import register


WAVE_TYPES = ("sine", "square", "triangle", "sawtooth")


def _waveform(kind, phase):
    """Return a unit-amplitude waveform in [-1, 1] for ``phase`` in radians."""
    if kind == "sine":
        return np.sin(phase)
    if kind == "square":
        return np.where(np.sin(phase) >= 0, 1.0, -1.0)
    if kind == "triangle":
        return (2.0 / np.pi) * np.arcsin(np.sin(phase))
    return 2.0 * np.mod(phase / (2.0 * np.pi), 1.0) - 1.0


@register
class WaveWarp(AEEffect):
    """Displace pixels along a wave that travels across the layer.

    ==================  ===================  ===========================
    Parameter           AE panel name        Notes
    ==================  ===================  ===========================
    wave_type           Wave Type            sine / square / triangle / sawtooth
    wave_height         Wave Height          0..1000 px peak-to-peak, default 0
    wave_width          Wave Width           1..10000 px wavelength, default 40
    direction           Direction            degrees, default 90
    wave_speed          Wave Speed           cycles per second, default 1
    phase               Phase                degrees, default 0
    ==================  ===================  ===========================

    ``direction`` sets the displacement axis: 0 displaces horizontally, 90
    vertically (clockwise from right). The wave travels perpendicular to it.
    Each destination pixel samples the source at its position minus the
    displacement evaluated at that pixel, with bilinear, premultiplied,
    zero-filled sampling.

    Notes
    -----
    Displacement amplitude is ``wave_height / 2``, so the height is the total
    peak-to-peak excursion. The layer grows by ``wave_height`` pixels on every
    side so displaced content is never clipped. Adobe's exact waveform and
    its height convention are not published; the sine, square, triangle and
    sawtooth shapes here are analytic. Wave phase uses world coordinates, so
    the pattern does not shift when the padding changes. ``wave_height`` 0 is
    an exact identity.
    """

    name = "Wave Warp"
    category = "Distort"
    PARAMS = (
        Param("wave_type", "enum", "sine", choices=WAVE_TYPES),
        Param("wave_height", "float", 0.0, (0.0, 1000.0), unit="px"),
        Param("wave_width", "float", 40.0, (1.0, 10000.0), unit="px"),
        Param("direction", "float", 90.0),
        Param("wave_speed", "float", 1.0),
        Param("phase", "float", 0.0),
    )

    def bounds_expand(self, size, t, context=None, values=None):
        """Grow by ``wave_height`` pixels on every side."""
        values = self.values_at(t, context) if values is None else values
        margin = int(math.ceil(values["wave_height"]))
        return (margin,) * 4

    def render(self, src, t, context=None, values=None):
        """Sample ``src`` at positions displaced by the animated wave."""
        values = self.values_at(t, context) if values is None else values
        if 0 in src.size or values["wave_height"] == 0:
            return src
        height, width = src.rgba.shape[:2]
        theta = math.radians(values["direction"])
        normal = (math.cos(theta), math.sin(theta))
        travel = (-math.sin(theta), math.cos(theta))
        amplitude = values["wave_height"] / 2.0

        xs, ys = np.meshgrid(
            np.arange(width, dtype=np.float64) + 0.5,
            np.arange(height, dtype=np.float64) + 0.5,
        )
        world_x = xs + src.offset[0]
        world_y = ys + src.offset[1]
        cycles = (world_x * travel[0] + world_y * travel[1]) / values["wave_width"]
        phase = 2.0 * np.pi * (cycles - values["wave_speed"] * t) + math.radians(
            values["phase"]
        )
        shift = amplitude * _waveform(values["wave_type"], phase)
        rgba = _sample(src.rgba, xs - shift * normal[0], ys - shift * normal[1])
        return Buffer._publish(rgba, src.offset, src.color_space)
