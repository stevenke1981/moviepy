"""Composition audio adapters and scalar, animatable dB levels.

Audio uses the footage's continuous time mapping, without video frame snapping
or pitch preservation. Mixing borrows the source clips and never closes them.
"""

import math

from moviepy.ae._geometry import to_property
from moviepy.ae.properties.values import finite_real


def levels_property(value):
    """Coerce scalar dB levels to the existing animated float Property type."""
    return to_property(value, "audio_levels", "float")


def db_to_amplitude(value):
    """Convert finite scalar decibels to gain, rejecting numeric overflow.

    Zero dB is unity; all channels share this gain. Exact mute is controlled by
    ``audio_enabled``, rather than a nonfinite Property value. The float mix is
    not limited to display or encoder amplitude ranges.
    """
    value = finite_real(value, "audio_levels")
    try:
        return math.pow(10.0, value / 20.0)
    except OverflowError as error:
        raise ValueError("audio_levels gain must remain finite") from error


def composition_audio(comp):
    """Build a MoviePy audio mix from the composition's current AV sources.

    Source membership is captured by each returned mix; layer timing, mute,
    levels and solo switches remain live. Fetch ``comp.audio`` again after
    adding/removing layers or replacing a source clip. Manual MoviePy audio
    overrides are handled by ``Composition.audio``.
    """
    from moviepy.ae.audio.mix import build_audio

    return build_audio(comp)
