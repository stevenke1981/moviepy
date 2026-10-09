"""Spectrum analysis and bars against a direct port of ``s13-spectrum.py``."""

import colorsys

import numpy as np

import pytest

from moviepy import VideoClip
from moviepy.ae import Composition, Transform
from moviepy.ae.templates._audio_io import AudioWriter
from moviepy.ae.templates.spectrum import (
    SpectrumLayer,
    analyze_spectrum,
    load_levels,
    save_levels,
)


FPS, RATE, FFT, BANDS = 24, 48000, 8192, 64
EDGES = np.geomspace(45, 10000, BANDS + 1)
FREQ = np.fft.rfftfreq(FFT, 1 / RATE)
WINDOW = np.hanning(FFT).astype(np.float32)
SLICES = [
    np.flatnonzero((FREQ >= a) & (FREQ < b)) for a, b in zip(EDGES[:-1], EDGES[1:])
]


def ref_bands(samples):
    power = np.mean(np.abs(np.fft.rfft(samples * WINDOW[:, None], axis=0)) ** 2, axis=1)
    amps = (
        np.array([np.sqrt(power[s].max()) if len(s) else 0 for s in SLICES])
        * 2
        / WINDOW.sum()
    )
    db = 20 * np.log10(np.maximum(amps, 1e-10))
    return np.clip((db + 65) / 53, 0, 1).astype(np.float32)


def ref_analyze(data, seconds):
    count = round(seconds * FPS)
    out = np.empty((count, BANDS), np.float32)
    previous = np.zeros(BANDS, np.float32)
    for i in range(count):
        center = round(i * RATE / FPS)
        start = center - FFT // 2
        block = np.zeros((FFT, data.shape[1]), np.float32)
        lo, hi = max(0, start), min(len(data), start + FFT)
        if hi > lo:
            block[lo - start : hi - start] = data[lo:hi]
        current = ref_bands(block)
        weight = np.where(current > previous, 0.55, 0.14)
        previous = previous + weight * (current - previous)
        out[i] = previous
    return out


def ref_composition(levels):
    width, height = 1088, 140
    palette = np.array(
        [
            colorsys.hsv_to_rgb(0.02 + 0.73 * i / (BANDS - 1), 0.72, 1)
            for i in range(BANDS)
        ]
    )
    colors = np.rint(palette * 255).astype(np.uint8)

    def pixels(t, mask=False):
        index = min(len(levels) - 1, max(0, round(t * FPS)))
        shape = (height, width) if mask else (height, width, 3)
        frame = np.zeros(shape, np.float32 if mask else np.uint8)
        for band, value in enumerate(levels[index]):
            h = round(float(value) * (height - 8))
            if h:
                x = band * 17 + 2
                frame[height - 4 - h : height - 4, x : x + 12] = (
                    1 if mask else colors[band]
                )
        return frame

    duration = len(levels) / FPS
    clip = VideoClip(lambda t: pixels(t), duration=duration).with_fps(FPS)
    clip = clip.with_mask(
        VideoClip(lambda t: pixels(t, True), is_mask=True, duration=duration)
    )
    comp = Composition(
        size=(width, height), fps=FPS, duration=duration, transparent=True
    )
    comp.add_clip(clip, name="ref", transform=Transform(opacity=30))
    return comp


def write_wav(path, data):
    with AudioWriter(path, rate=RATE) as writer:
        writer.write(np.asarray(data, np.float32))


def tone(hz, seconds, amp=0.1):
    return amp * np.sin(2 * np.pi * hz * np.arange(int(seconds * RATE)) / RATE)


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    path = tmp_path_factory.mktemp("spec") / "mix.wav"
    t = tone(440, 1.0) + tone(2000, 1.0, 0.05)
    data = np.column_stack([t, 0.5 * t[::-1]]).astype(np.float32)
    write_wav(path, data)
    return path, data


def test_analysis_matches_reference_port(mixed):
    path, data = mixed
    got = analyze_spectrum(path, seconds=1.0)
    want = ref_analyze(data, 1.0)
    assert got.shape == want.shape == (24, 64) and got.dtype == np.float32
    np.testing.assert_allclose(got, want, atol=2e-5)


def test_analysis_start_and_tail(mixed):
    path, data = mixed
    # Whole file, then a later slice: window content is global, smoothing restarts.
    full = analyze_spectrum(path)
    assert full.shape[0] == 24
    part = analyze_spectrum(path, start=0.5, seconds=0.5)
    assert part.shape[0] == 12
    with pytest.raises(ValueError):
        analyze_spectrum(path, seconds=5.0)


@pytest.mark.parametrize("hz", [110, 440, 2000])
def test_tone_peak_in_right_band(tmp_path, hz):
    t = tone(hz, 0.6)
    path = tmp_path / "t.wav"
    write_wav(path, np.column_stack([t, t]))
    levels = analyze_spectrum(path, seconds=0.5)
    peak = int(levels[-1].argmax())
    assert EDGES[max(0, peak - 1)] < hz < EDGES[min(BANDS, peak + 2)]
    assert levels[-1].max() > 0.7


def test_silence_is_zero_and_antiphase_not_cancelled(tmp_path):
    path = tmp_path / "s.wav"
    write_wav(path, np.zeros((RATE // 2, 2)))
    assert not analyze_spectrum(path, seconds=0.25).any()
    t = tone(440, 0.6)
    path2 = tmp_path / "a.wav"
    write_wav(path2, np.column_stack([t, -t]))
    assert analyze_spectrum(path2, seconds=0.5)[-1].max() > 0.7


def test_save_load_levels_exclusive(tmp_path):
    levels = np.random.RandomState(1).rand(5, 7).astype(np.float32)
    path = tmp_path / "l.npy"
    save_levels(path, levels)
    np.testing.assert_array_equal(load_levels(path), levels)
    with pytest.raises(FileExistsError):
        save_levels(path, levels)


def test_layer_matches_reference_composition():
    rng = np.random.RandomState(3)
    levels = rng.rand(4, BANDS).astype(np.float32)
    levels[1] = 1.0
    levels[2] = 0.0
    ref = ref_composition(levels)
    comp = Composition(size=(1088, 140), fps=FPS, duration=4 / FPS, transparent=True)
    comp.add_layer(SpectrumLayer(levels, fps=FPS))
    for i in range(4):
        t = i / FPS
        a, b = comp.render_buffer(t), ref.render_buffer(t)
        np.testing.assert_allclose(a.rgba, b.rgba, atol=1e-6)
    assert np.isclose(comp.render_buffer(1 / FPS).rgba[..., 3].max(), 0.3)


def test_layer_frame_index_clamped():
    layer = SpectrumLayer(np.zeros((3, 4), np.float32), size=(80, 20))
    assert layer.frame_index(-1) == 0 and layer.frame_index(99) == 2
    assert layer.frame_index(1.5 / FPS) in (1, 2)
    with pytest.raises(ValueError):
        SpectrumLayer(np.zeros(3))
