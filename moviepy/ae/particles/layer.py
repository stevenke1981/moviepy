"""Random-access point-emitter particles for sparks and stylized explosions."""

import math
from numbers import Integral

import numpy as np

from moviepy.ae._geometry import finite_real, validate_pixel_size
from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import sigma_margin
from moviepy.ae.effects.blur.gaussian_blur import filter_pixels
from moviepy.ae.layers.base import Layer
from moviepy.ae.layers.solid import _color


def _pair(value, name):
    values = tuple(finite_real(item, name) for item in value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain two numbers")
    return np.asarray(values, dtype=np.float64)


class ParticleLayer(Layer):
    """Emit a reproducible burst or short spray, with analytic gravity.

    ``count`` is the total particle count (up to 50,000), ``seed`` fixes the
    birth schedule and velocities. ``emission_duration=0`` is a burst.
    Positions use pixels, velocities pixels/second, gravity pixels/second².
    Direction is clockwise from right in degrees; spread is the full cone.
    All particles live ``lifetime`` seconds. Color and opacity fade with age.
    The soft circular sprites are a stylized explosion, not smoke or fluid
    dynamics. Parameters are fixed at construction; layer transforms/effects
    remain animatable. No playback history or cross-frame cache is required.
    """

    def __init__(
        self,
        name="Particles",
        *,
        size=(1920, 1080),
        count=300,
        seed=0,
        emitter=None,
        speed=(80, 240),
        direction=-90,
        spread=360,
        gravity=(0, 160),
        lifetime=1.5,
        emission_duration=0.0,
        radius=2.0,
        color_start=(255, 235, 150),
        color_end=(255, 55, 10),
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._size = validate_pixel_size(size)
        for value, label in ((count, "count"), (seed, "seed")):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{label} must be a nonnegative integer")
        if count > 50000:
            raise ValueError("count must not exceed 50000")
        self._lifetime = finite_real(lifetime, "lifetime")
        self._emission = finite_real(emission_duration, "emission_duration")
        self._radius = finite_real(radius, "radius")
        if self._lifetime <= 0 or self._emission < 0 or not 0 < self._radius <= 64:
            raise ValueError(
                "lifetime > 0, emission_duration >= 0 and 0 < radius <= 64 required"
            )
        self._origin = _pair(
            (size[0] / 2, size[1] / 2) if emitter is None else emitter, "emitter"
        )
        self._gravity = _pair(gravity, "gravity")
        limits = _pair(speed, "speed")
        direction = finite_real(direction, "direction")
        spread = finite_real(spread, "spread")
        if not 0 <= limits[0] <= limits[1] or not 0 <= spread <= 360:
            raise ValueError(
                "speed must be ordered and nonnegative; spread must be 0..360"
            )
        generator = np.random.default_rng(int(seed))
        angle = np.radians(
            direction + generator.uniform(-spread / 2, spread / 2, count)
        )
        magnitude = generator.uniform(*limits, count)
        self._velocity = (
            np.column_stack((np.cos(angle), np.sin(angle))) * magnitude[:, None]
        )
        self._birth = generator.uniform(0, self._emission, count)
        self._colors = np.asarray([_color(color_start), _color(color_end)], np.float32)
        self._birth.setflags(write=False)
        self._velocity.setflags(write=False)

    @property
    def source_size(self):
        """Return the fixed particle canvas size."""
        return self._size

    def state_at(self, t):
        """Return detached active positions and normalized ages at local time."""
        age = finite_real(t, "t") - self._birth
        active = (age >= 0) & (age < self._lifetime)
        age = age[active]
        positions = self._origin + self._velocity[active] * age[:, None]
        positions += 0.5 * self._gravity * age[:, None] ** 2
        return positions, age / self._lifetime

    def source_buffer(self, t, context=None):
        """Splat and soften visible particles; random seeks reproduce playback."""
        width, height = self._size
        margin = sigma_margin(self._radius / 2)
        positions, ages = self.state_at(t)
        positions = positions + margin
        shape = (height + 2 * margin, width + 2 * margin)
        visible = ((positions >= -1) & (positions < (shape[1], shape[0]))).all(axis=1)
        positions, ages = positions[visible], ages[visible]
        density = np.zeros((*shape, 4), np.float32)
        color = self._colors[0] + (self._colors[1] - self._colors[0]) * ages[:, None]
        weights = (
            np.column_stack((color, np.ones(len(ages)))) * (1 - ages[:, None]) ** 2
        )
        weights *= max(1, 2 * math.pi * (self._radius / 2) ** 2)
        self._splat(density, positions, weights)
        density = filter_pixels(density, self._radius / 2, self._radius / 2)
        density = density[margin : margin + height, margin : margin + width]
        amount = np.maximum(density[..., 3:], 0)
        alpha = -np.expm1(-amount)
        rgb = np.zeros_like(density[..., :3])
        np.divide(density[..., :3], amount, out=rgb, where=amount > 0)
        rgba = np.concatenate((np.clip(rgb, 0, 1) * alpha, alpha), axis=-1)
        return Buffer._publish(rgba, (0, 0), "srgb")

    @staticmethod
    def _splat(canvas, positions, weights):
        """Accumulate subpixel samples in a stable particle order."""
        corners = np.floor(positions).astype(np.int64)
        fraction = positions - corners
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            xy = corners + (dx, dy)
            inside = ((xy >= 0) & (xy < (canvas.shape[1], canvas.shape[0]))).all(axis=1)
            factor = fraction[:, 0] if dx else 1 - fraction[:, 0]
            factor = factor * (fraction[:, 1] if dy else 1 - fraction[:, 1])
            np.add.at(
                canvas,
                (xy[inside, 1], xy[inside, 0]),
                (weights * factor[:, None])[inside],
            )
