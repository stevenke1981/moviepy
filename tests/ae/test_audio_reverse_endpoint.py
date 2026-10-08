"""Reverse playback must include the final PCM sample despite float rounding."""

import numpy as np

import pytest

from moviepy import AudioArrayClip, ColorClip
from moviepy.ae import Composition


@pytest.mark.parametrize(
    "fps,count",
    [(8000, 1001), (8000, 1002), (44100, 15), (44100, 16), (48000, 27), (48000, 28)],
)
def test_reverse_starts_with_last_pcm_sample(fps, count):
    pcm = np.zeros((count, 2))
    pcm[-1] = [0.125, -0.25]
    source = AudioArrayClip(pcm, fps=fps)
    clip = ColorClip((2, 2), (0, 0, 0), duration=source.duration).with_audio(source)
    comp = Composition(size=(2, 2), duration=source.duration)
    comp.add_clip(clip, stretch=-100)
    audio = comp.audio

    np.testing.assert_array_equal(audio.get_frame(0), pcm[-1])
    np.testing.assert_array_equal(
        audio.get_frame(np.array([source.duration, 0, 0])),
        [[0, 0], pcm[-1], pcm[-1]],
    )
