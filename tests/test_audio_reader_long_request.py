"""Regression: one get_frame call spanning more than half the reader buffer."""

import wave

import numpy as np

from moviepy.audio.io.AudioFileClip import AudioFileClip


def test_long_time_array_returns_samples_not_silence(tmp_path):
    fps, seconds = 44100, 6
    t = np.arange(fps * seconds) / fps
    tone = (0.5 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(fps)
        handle.writeframes(np.repeat(tone[:, None], 2, axis=1).tobytes())

    clip = AudioFileClip(str(path), fps=fps)
    try:
        # 5 s in one call: far beyond buffersize // 2 frames.
        times = np.arange(0, 5 * fps) / fps
        samples = clip.reader.get_frame(times)
        assert samples.shape == (len(times), 2)
        expected = np.repeat(tone[: len(times), None] / 32767, 2, axis=1)
        np.testing.assert_allclose(samples, expected, atol=2e-3)
        # Out-of-range tail entries stay silent at their own positions.
        mixed = np.concatenate([times, [7.0, 8.0]])
        out = clip.reader.get_frame(mixed)
        assert out.shape == (len(mixed), 2)
        np.testing.assert_allclose(out[:-2], expected, atol=2e-3)
        assert not out[-2:].any()
    finally:
        clip.close()
