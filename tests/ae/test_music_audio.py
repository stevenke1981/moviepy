"""Tests for the streaming music-channel audio helpers."""

import numpy as np

import pytest

from moviepy.ae.templates._audio_io import AudioWriter, audio_info, read_audio
from moviepy.ae.templates.music_audio import (
    assemble_chapters,
    chime_tone,
    loop_extend,
    loop_extend_array,
    measure_loudness,
    mix_cues,
    normalize_loudness,
    write_chime,
)


RATE = 8000


def reference_chapter(data, seconds, rate, overlap_seconds):
    """Straightforward port of the S14 ``chapter()`` used as an oracle."""
    length, cross = round(seconds * rate), round(overlap_seconds * rate)
    out = np.zeros((length, 2), np.float32)
    ramp = np.linspace(0, 1, cross, dtype=np.float32)[:, None]
    start, cues = 0, []
    while start < length:
        count = min(len(data), length - start)
        piece = data[:count].copy()
        if start:
            blend = min(cross, count)
            out[start : start + blend] *= 1 - ramp[:blend]
            piece[:blend] *= ramp[:blend]
        out[start : start + count] += piece
        cues.append(start / rate)
        start += len(data) - cross
    return out, cues


def sine(seconds, freq, rate=RATE, amp=0.4):
    time = np.arange(round(seconds * rate)) / rate
    wave = (amp * np.sin(2 * np.pi * freq * time)).astype(np.float32)
    return np.stack([wave, wave * 0.9], axis=1)


def noise_master(seconds, seed, rate=RATE, amp=0.1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((round(seconds * rate), 2)) * amp).astype(np.float32)


@pytest.mark.parametrize("seconds", [0.7, 1.3, 2.0, 5.37, 6.0])
def test_loop_extend_matches_reference(seconds):
    data = noise_master(1.0, 1)
    blocks, starts = loop_extend(data, seconds, rate=RATE, overlap=0.25, block=777)
    got = np.concatenate(list(blocks))
    want, cues = reference_chapter(data, seconds, RATE, 0.25)
    assert got.shape == (round(seconds * RATE), 2)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, want)
    assert starts == cues


def test_loop_extend_array_and_validation():
    data = noise_master(1.0, 2)
    array, starts = loop_extend_array(data, 3.0, rate=RATE, overlap=0.25)
    assert len(array) == 3 * RATE and starts[0] == 0.0
    with pytest.raises(ValueError):
        loop_extend(data, 3.0, rate=RATE, overlap=0.5)  # not > 2 * overlap
    with pytest.raises(ValueError):
        loop_extend(data[:, :1], 3.0, rate=RATE, overlap=0.1)


def test_loop_seams_have_no_discontinuity():
    data = sine(1.0, 220.7)  # not a whole number of periods: phase jump at wrap
    step = 2 * np.pi * 220.7 / RATE * 0.4
    out, starts = loop_extend_array(data, 6.0, rate=RATE, overlap=0.25)
    assert len(starts) > 4
    jumps = np.abs(np.diff(out[:, 0]))
    assert jumps.max() < 1.25 * step
    # A plain concatenation would jump at the seam; make sure the test sees it.
    naive = np.concatenate([data, data])[:, 0]
    assert np.abs(np.diff(naive)).max() > 1.25 * step


def test_assemble_chapters(tmp_path):
    first = sine(1.0, 200.0)
    second = tmp_path / "second.wav"
    with AudioWriter(second, rate=RATE) as writer:
        writer.write(sine(1.0, 330.0))
    out = tmp_path / "all.flac"
    options = dict(rate=RATE, loop_overlap=0.25, chapter_crossfade=0.5, edge_fade=0.25)
    sheet = assemble_chapters([first, second], [2.0, 2.5], out, **options)
    total = round(2.0 * RATE) + round(2.5 * RATE) - round(0.5 * RATE)
    assert sheet["total_frames"] == total == audio_info(out)["frames"]
    assert sheet["human_listening"] == "NOT_RUN"
    assert sheet["clipped_samples"] == 0 and sheet["peak"] < 1
    one, two = sheet["chapters"]
    assert (one["id"], two["id"]) == ("G01", "G02")
    assert one["global_start_seconds"] == 0.0
    assert two["global_start_seconds"] == pytest.approx(1.5)
    assert two["crossfade_center_seconds"] == pytest.approx(1.75)
    assert two["source"] == str(second) and len(two["sha256"]) == 64
    assert one["source_loop_starts"][:2] == [0.0, 0.75]
    assert two["prepared_seconds"] == 2.5
    audio = read_audio(out, rate=RATE)
    assert len(audio) == total
    assert abs(audio[0, 0]) < 1e-4 and abs(audio[-1, 0]) < 1e-4
    assert np.abs(audio[: RATE // 4]).max() > 0.2  # fade-in has finished
    # Away from fades and the chapter seam the audio equals the looped chapter.
    expect, _ = reference_chapter(first, 2.0, RATE, 0.25)
    mid = slice(RATE // 4, 2 * RATE - 4000)
    np.testing.assert_allclose(audio[mid, 0], expect[mid, 0], atol=2e-6)
    with pytest.raises(FileExistsError):
        assemble_chapters([first], 2.0, out, **options)
    redo = assemble_chapters([first], 2.0, out, overwrite=True, **options)
    assert redo["total_frames"] == 2 * RATE == audio_info(out)["frames"]


def test_assemble_clipping(tmp_path):
    loud = sine(1.0, 100.0, amp=1.5)
    options = dict(rate=RATE, loop_overlap=0.25, chapter_crossfade=0.5, edge_fade=0.25)
    with pytest.raises(ValueError, match="clips"):
        assemble_chapters([loud, loud], 2.0, tmp_path / "x.wav", **options)
    assert not (tmp_path / "x.wav").exists()
    sheet = assemble_chapters(
        [loud, loud], 2.0, tmp_path / "y.wav", allow_clipping=True, **options
    )
    assert sheet["clipped_samples"] > 0 and sheet["total_frames"] == 7 * RATE // 2


def test_chime_shape_and_spectrum(tmp_path):
    tone = chime_tone()
    assert tone.shape == (98400, 2) and tone.dtype == np.float32
    np.testing.assert_array_equal(tone[:, 0], tone[:, 1])
    assert 20 * np.log10(np.abs(tone).max()) == pytest.approx(-30.0, abs=1e-3)
    assert tone[0, 0] == 0.0 and tone[-1, 0] == 0.0
    assert abs(tone[1, 0]) < 1e-4  # soft attack
    assert np.abs(np.diff(tone[:, 0])).max() < 0.2 * np.abs(tone).max()
    window = tone[:, 0].astype(np.float64) * np.hanning(len(tone))
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(tone), 1 / 48000)

    def peak_near(low, high):
        band = (freqs > low) & (freqs < high)
        return freqs[band][np.argmax(spectrum[band])], spectrum[band].max()

    d4, d4_level = peak_near(250, 340)
    a4, a4_level = peak_near(400, 480)
    assert d4 == pytest.approx(293.66, abs=2.0)
    assert a4 == pytest.approx(440.0, abs=2.0)
    far = spectrum[(freqs > 1000) & (freqs < 4000)].max()
    assert far < min(d4_level, a4_level) * 0.01
    facts = write_chime(tmp_path / "chime.wav")
    assert facts["frames"] == 98400 == audio_info(tmp_path / "chime.wav")["frames"]
    assert facts["human_listening"] == "NOT_RUN" and facts["clipped_samples"] == 0
    with pytest.raises(FileExistsError):
        write_chime(tmp_path / "chime.wav")
    with pytest.raises(ValueError):
        chime_tone(notes=(("X", 440.0, 1.5, 1.6),))


def test_mix_cues(tmp_path):
    music = noise_master(6.0, 3, amp=0.2)
    quiet = noise_master(6.0, 4, amp=0.002)
    both = np.concatenate([music[: 3 * RATE], quiet[3 * RATE :]])
    path = tmp_path / "music.wav"
    with AudioWriter(path, rate=RATE, subtype="float") as writer:
        writer.write(both)
    cue = chime_tone(
        duration=0.5,
        notes=(("A4", 440.0, 0.0, 0.5),),
        release=0.3,
        attack=0.05,
        rate=RATE,
    )
    out = tmp_path / "mixed.wav"
    report = mix_cues(path, cue, [0.5, 4.0], out, rate=RATE)
    loud_event, quiet_event = report["events"]
    assert loud_event["gain"] == 1.0  # never boosted
    assert quiet_event["gain"] < 1.0
    assert quiet_event["difference_db"] == pytest.approx(-10.0, abs=0.01)
    assert loud_event["difference_db"] < -10.0
    assert report["ducking"] is False and report["normalization"] is False
    assert report["human_listening"] == "NOT_RUN" and report["clipped_samples"] == 0
    mixed = read_audio(out, rate=RATE)
    assert len(mixed) == len(both)
    start = int(0.5 * RATE)
    np.testing.assert_allclose(mixed[:start], both[:start], atol=1e-6)
    stop = start + len(cue)
    np.testing.assert_allclose(mixed[stop : 4 * RATE], both[stop : 4 * RATE], atol=1e-6)
    np.testing.assert_allclose(mixed[start:stop] - both[start:stop], cue, atol=1e-5)
    with pytest.raises(FileExistsError):
        mix_cues(path, cue, [0.5], out, rate=RATE)
    with pytest.raises(ValueError):
        mix_cues(path, cue, [5.9], tmp_path / "late.wav", rate=RATE)
    with pytest.raises(ValueError):
        mix_cues(path, cue, [2.0, 1.0], tmp_path / "order.wav", rate=RATE)


def test_mix_cues_clipping_raises(tmp_path):
    path = tmp_path / "hot.wav"
    with AudioWriter(path, rate=RATE, subtype="float") as writer:
        writer.write(np.full((2 * RATE, 2), 0.999, np.float32))
    cue = np.full((100, 2), 0.5, np.float32)
    with pytest.raises(ValueError, match="clipped"):
        mix_cues(path, cue, [0.5], tmp_path / "o.wav", rate=RATE, below_music_db=-20)
    assert not (tmp_path / "o.wav").exists()


def pink_noise(frames, seed):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(frames))
    freqs = np.fft.rfftfreq(frames)
    spectrum[1:] /= np.sqrt(freqs[1:])
    spectrum[0] = 0
    wave = np.fft.irfft(spectrum, frames)
    wave = wave / np.abs(wave).max() * 0.12
    return np.stack([wave, wave[::-1]], axis=1).astype(np.float32)


def test_loudness_measure_and_normalize(tmp_path):
    """The explicit pure-FFmpeg backend keeps the original behaviour."""
    source = tmp_path / "noise.wav"
    with AudioWriter(source, rate=48000, subtype="float") as writer:
        writer.write(pink_noise(480000, 5))
    measured = measure_loudness(source, backend="ffmpeg")
    for key in ("integrated_lufs", "true_peak_dbtp", "lra", "threshold", "input_i"):
        assert key in measured
    assert -45 < measured["integrated_lufs"] < -20
    target = tmp_path / "norm.flac"
    facts = normalize_loudness(source, target, backend="ffmpeg")
    assert facts["output"]["frames"] == 480000 == audio_info(target)["frames"]
    assert facts["human_listening"] == "NOT_RUN"
    loudness = facts["output"]["loudness"]
    assert loudness["integrated_lufs"] == pytest.approx(-18.0, abs=1.0)
    assert loudness["true_peak_dbtp"] <= -1.0
    with pytest.raises(FileExistsError):
        normalize_loudness(source, target, backend="ffmpeg")
    longer = tmp_path / "long.wav"
    normalize_loudness(source, longer, frames=500000, backend="ffmpeg")
    assert audio_info(longer)["frames"] == 500000
    with pytest.raises(ValueError):
        normalize_loudness(source, tmp_path / "bad.mp3", backend="ffmpeg")
