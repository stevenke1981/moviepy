"""Local deterministic random streams and continuous bounded noise."""

import itertools
import math

import numpy as np

from moviepy.ae.properties._expression_helpers import componentwise, vector
from moviepy.ae.properties._sandbox import ExpressionError, safe_number, sanitize


_UNSET = object()


def _integer(value, minimum, maximum, name):
    number = safe_number(value)
    if not number.is_integer() or not minimum <= number <= maximum:
        raise ExpressionError(name + " must be an integer in its allowed range")
    return int(number)


class NoiseState:
    """Own streams per evaluation; never alter the global NumPy RNG."""

    def __init__(self, context, layer_id, value, t):
        self.context, self.layer_id = context, layer_id
        self.value, self.t, self.seed = value, t, 0
        self.rng = context.rng_for(layer_id)
        self.stable_seed = self._stable_seed()

    def _stable_seed(self):
        identity = "expression-noise:" + repr(self.layer_id) + ":" + str(self.seed)
        stream = self.context.with_time(0).rng_for(identity)
        return int(stream.integers(0, 2**63))

    def seed_random(self, seed, timeless=False):
        """Restart local streams with an explicit seed and optional fixed time."""
        self.seed = _integer(seed, 0, 2**53, "seed")
        if type(timeless) is not bool:
            raise ExpressionError("timeless must be boolean")
        identity = "expression-random:" + repr(self.layer_id) + ":" + str(self.seed)
        ctx = self.context.with_time(0) if timeless else self.context
        self.rng = ctx.rng_for(identity)
        self.stable_seed = self._stable_seed()
        return 0.0

    def random(self, minimum=_UNSET, maximum=None):
        """Sample uniform scalar or vector values between ordered bounds."""
        if maximum is None:
            minimum, maximum = (0, 1) if minimum is _UNSET else (0, minimum)
        elif minimum is _UNSET:
            minimum = 0
        return componentwise(
            lambda low, high: self._uniform(low, high), minimum, maximum
        )

    def _uniform(self, low, high):
        if high < low:
            raise ExpressionError("random minimum exceeds maximum")
        return float(self.rng.uniform(low, high))

    def gaussian(self, minimum=_UNSET, maximum=None):
        """Sample normals with the given interval spanning six standard deviations."""
        if maximum is None:
            minimum, maximum = (0, 1) if minimum is _UNSET else (0, minimum)
        elif minimum is _UNSET:
            minimum = 0
        return componentwise(
            lambda low, high: self._gaussian(low, high), minimum, maximum
        )

    def _gaussian(self, low, high):
        if high < low:
            raise ExpressionError("gaussRandom minimum exceeds maximum")
        return float(self.rng.normal((low + high) / 2, (high - low) / 6))

    def _lattice(self, coordinates):
        code = self.stable_seed
        for axis, value in enumerate(coordinates):
            code ^= (value & ((1 << 64) - 1)) + (axis + 1) * 0x9E3779B97F4A7C15
            code = ((code ^ (code >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        code = (code ^ (code >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
        return ((code ^ (code >> 31)) / (2**64 - 1)) * 2 - 1

    def noise(self, coordinate):
        """Sample continuous quintic value noise in one to three dimensions."""
        coordinate = sanitize(coordinate)
        values = (
            vector(coordinate)
            if type(coordinate) is tuple
            else (safe_number(coordinate),)
        )
        if not 1 <= len(values) <= 3 or any(abs(x) > 1e12 for x in values):
            raise ExpressionError("noise requires one to three bounded coordinates")
        origins = tuple(math.floor(x) for x in values)
        fractions = tuple(x - origin for x, origin in zip(values, origins))
        weights = tuple(x * x * x * (x * (x * 6 - 15) + 10) for x in fractions)
        result = 0.0
        for corners in itertools.product((0, 1), repeat=len(values)):
            lattice = tuple(origin + corner for origin, corner in zip(origins, corners))
            weight = math.prod(w if c else 1 - w for w, c in zip(weights, corners))
            result += self._lattice(lattice) * weight
        return result

    def wiggle(self, freq, amp, octaves=1, amp_mult=0.5, t=None):
        """Add deterministic narrow-band noise to a scalar or vector base value."""
        freq, amp_mult = safe_number(freq), safe_number(amp_mult)
        octaves = _integer(octaves, 1, 8, "octaves")
        t = self.t if t is None else safe_number(t)
        if freq < 0 or not 0 <= amp_mult <= 1:
            raise ExpressionError("freq must be nonnegative; amp_mult must be in [0,1]")
        if freq == 0:
            return self.value
        phase_time = safe_number(freq * t)
        rng = np.random.Generator(np.random.PCG64(self.stable_seed))
        size = len(self.value) if type(self.value) is tuple else 1
        amplitudes = (
            vector(amp, size) if type(amp) is tuple else (safe_number(amp),) * size
        )
        deviations = []
        for amplitude in amplitudes:
            total = 0.0
            for octave in range(octaves):
                frequencies = rng.uniform(0.9, 1.1, 7)
                phases = rng.uniform(0, 2 * math.pi, 7)
                waves = np.sin(
                    2 * math.pi * phase_time * (2**octave) * frequencies + phases
                )
                total += float(waves.sum() / 7) * amplitude * amp_mult**octave
            deviations.append(total)
        offset = tuple(deviations) if type(self.value) is tuple else deviations[0]
        return componentwise(lambda base, delta: base + delta, self.value, offset)

    def registry(self):
        """Return the fixed local random and noise helper capabilities."""
        return {
            "random": self.random,
            "gaussRandom": self.gaussian,
            "noise": self.noise,
            "wiggle": self.wiggle,
            "seedRandom": self.seed_random,
        }
