"""Add Grain (Noise & Grain)."""

import numpy as np

from moviepy.ae.effects._pixels import is_opaque, publish_straight, straight_rgb
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register

NOISE_TYPES = ("color", "mono")
_SEED_LIMIT = 2**31 - 1


def _grain_generator(context, t, random_seed):
    """Return a PCG64 generator fixed by seed, time (ms) and context stream."""
    words = [int(random_seed), int(round(t * 1000.0)) % 2**63]
    if context is not None and hasattr(context, "rng_for"):
        stream = context.rng_for(f"add_grain:{int(random_seed)}")
        words.append(int(stream.integers(0, 2**63, dtype=np.uint64)))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))


def _reflect_unit(values):
    """Fold values into [0, 1] by mirroring at both ends (no saturation)."""
    folded = np.mod(values, 2.0)
    return np.where(folded > 1.0, 2.0 - folded, folded)


@register
class Noise(AEEffect):
    """Add random color or monochrome grain to the image.

    ====================  ===================  ===========================
    Parameter             AE panel name        Notes
    ====================  ===================  ===========================
    amount_of_noise       Amount of Noise      0..100 %, default 0
    noise_type            Noise Type           "color" or "mono"
    clipping              Clipping             bool, default on
    random_seed           Random Seed          integer 0..2^31-1, default 0
    ====================  ===================  ===========================

    Notes
    -----
    Noise is uniform in ``[-amount/100, +amount/100]`` on straight 0..1
    channels. ``color`` draws independent R, G and B samples; ``mono`` draws
    one sample per pixel for all channels. Grain is added to straight color
    and re-premultiplied, so transparent pixels stay transparent.

    The generator is a PCG64 stream seeded from ``random_seed``, the frame
    time in milliseconds and, when the render context provides ``rng_for``,
    a context stream. The same inputs give bit-identical output; a different
    time gives different grain. With clipping on, grained channels are
    clamped to 0..1. With clipping off they are mirrored back into 0..1 so
    no channel saturates. Adobe's grain distribution and its clipping are not
    reproduced exactly. Amount 0 returns the input unchanged.
    """

    name = "Noise"
    category = "Noise & Grain"
    PARAMS = (
        Param("amount_of_noise", "float", 0.0, (0.0, 100.0)),
        Param("noise_type", "enum", "color", choices=NOISE_TYPES),
        Param("clipping", "bool", True),
        Param("random_seed", "float", 0.0, (0.0, float(_SEED_LIMIT))),
    )

    def render(self, src, t, context=None, values=None):
        """Add seeded grain to straight color and republish."""
        values = self.values_at(t, context) if values is None else values
        strength = values["amount_of_noise"] / 100.0
        if strength == 0.0 or 0 in src.size:
            return src
        height, width = src.size[1], src.size[0]
        rng = _grain_generator(context, t, values["random_seed"])
        channels = 3 if values["noise_type"] == "color" else 1
        draw = rng.uniform(-1.0, 1.0, size=(height, width, channels))
        grain = (draw * strength).astype(np.float32)
        if channels == 1:
            grain = np.repeat(grain, 3, axis=2)
        rgb = straight_rgb(src) + grain
        if values["clipping"]:
            rgb = np.clip(rgb, 0.0, 1.0)
        else:
            rgb = _reflect_unit(rgb).astype(np.float32)
        return publish_straight(rgb, src.rgba[..., 3:], src, is_opaque(src))
