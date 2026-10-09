"""Tests for the automated music QA helpers on small synthetic media."""

import json
import subprocess

import numpy as np

import pytest

from moviepy.ae.templates import music_qa as qa
from moviepy.ae.templates._audio_io import AudioWriter
from moviepy.ae.templates.music_qa import (
    clipping_report,
    flash_report,
    loudness_report,
    qa_report,
    silence_report,
    stream_facts,
)
from moviepy.config import FFMPEG_BINARY


WIDTH, HEIGHT = 160, 90
RATE = 48000


def _write_video(path, frames, fps, audio=None):
    """Encode RGB frames losslessly to ``path``, adding WAV audio if given."""
    command = [
        FFMPEG_BINARY,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    if audio is not None:
        command += ["-i", str(audio), "-map", "0:v", "-map", "1:a"]
        command += ["-c:a", "pcm_s16le"]
    command += ["-c:v", "ffv1", str(path)]
    data = np.ascontiguousarray(frames, dtype=np.uint8).tobytes()
    subprocess.run(command, input=data, check=True)


def _gray_frames(levels):
    """Build full-frame gray frames from a sequence of 0-255 levels."""
    out = np.empty((len(levels), HEIGHT, WIDTH, 3), np.uint8)
    for index, level in enumerate(levels):
        out[index] = level
    return out


def _sine(seconds, amplitude, freq=1000.0):
    """Return stereo sine samples at the module's test rate."""
    t = np.arange(int(seconds * RATE)) / RATE
    tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([tone, tone], axis=1)


def test_steady_video_has_no_flashes(tmp_path):
    path = tmp_path / "steady.mkv"
    _write_video(path, _gray_frames([128] * 48), 24)
    result = flash_report(path)
    assert result["max_flashes_per_second"] == 0
    assert result["verdict"] == "PASS"
    assert result["worst_window_start"] is None


def test_five_hz_black_white_flicker_fails(tmp_path):
    path = tmp_path / "flicker.mkv"
    levels = [255 if (index // 2) % 2 == 0 else 0 for index in range(200)]
    _write_video(path, _gray_frames(levels), 20)
    result = flash_report(path)
    assert result["max_flashes_per_second"] == 5
    assert result["flash_count"] >= 45
    assert result["verdict"] == "FAIL"
    assert result["red_flash"] == "NOT_COMPUTED"
    assert result["max_changing_area"] >= 0.99


def test_slow_fade_passes(tmp_path):
    path = tmp_path / "fade.mkv"
    levels = np.linspace(0, 255, 96).round().astype(int)
    _write_video(path, _gray_frames(levels), 24)
    result = flash_report(path)
    assert result["max_flashes_per_second"] == 0
    assert result["verdict"] == "PASS"
    assert result["max_luma_step"] < 0.1


def test_small_flickering_area_passes(tmp_path):
    frames = _gray_frames([128] * 80)
    for index in range(80):
        if (index // 2) % 2 == 0:
            frames[index, :40, :40] = 255
        else:
            frames[index, :40, :40] = 0
    path = tmp_path / "small.mkv"
    _write_video(path, frames, 20)
    result = flash_report(path)
    assert result["max_changing_area"] < 0.25
    assert result["max_flashes_per_second"] == 0
    assert result["verdict"] == "PASS"


def test_loudness_matches_sine_level_and_reports_true_peak(tmp_path):
    amplitude = 10 ** ((-18.0 + 0.691) / 20)
    path = tmp_path / "sine.wav"
    with AudioWriter(path, rate=RATE) as writer:
        writer.write(_sine(10, amplitude))
    result = loudness_report(path)
    expected = -0.691 + 20 * np.log10(amplitude)
    assert abs(result["integrated_lufs"] - expected) < 1.0
    assert abs(result["true_peak_dbtp"] - 20 * np.log10(amplitude)) < 0.5
    assert result["checks"]["true_peak"]["verdict"] == "PASS"
    assert result["verdict"] == "PASS"


def test_loud_sine_fails_loudness_target(tmp_path):
    path = tmp_path / "loud.wav"
    with AudioWriter(path, rate=RATE) as writer:
        writer.write(_sine(4, 0.5))
    result = loudness_report(path)
    assert result["checks"]["integrated"]["verdict"] == "FAIL"
    assert result["verdict"] == "FAIL"


def test_silence_detects_inserted_three_second_gap(tmp_path):
    audio = _sine(10, 0.1)
    audio[3 * RATE : 6 * RATE] = 0
    path = tmp_path / "gap.wav"
    with AudioWriter(path, rate=RATE) as writer:
        writer.write(audio)
    result = silence_report(path)
    assert result["count"] == 1
    gap = result["gaps"][0]
    assert abs(gap["start"] - 3.0) < 0.05
    assert abs(gap["duration"] - 3.0) < 0.05
    assert result["verdict"] == "REVIEW"


def test_clipping_counts_injected_clipped_samples(tmp_path):
    audio = _sine(1, 0.5)
    audio[100:104, 0] = 1.0
    audio[200:203, 1] = -1.0
    path = tmp_path / "clip.wav"
    with AudioWriter(path, rate=RATE) as writer:
        writer.write(audio)
    result = clipping_report(path)
    assert result["clipped_samples"] == 7
    assert result["verdict"] == "FAIL"


def test_clean_audio_is_not_clipped(tmp_path):
    path = tmp_path / "clean.wav"
    with AudioWriter(path, rate=RATE) as writer:
        writer.write(_sine(1, 0.5))
    result = clipping_report(path)
    assert result["clipped_samples"] == 0
    assert result["verdict"] == "PASS"


def test_stream_facts_reads_banner(tmp_path):
    path = tmp_path / "facts.mkv"
    tone = _sine(2, 0.1)
    wav = tmp_path / "tone.wav"
    with AudioWriter(wav, rate=RATE) as writer:
        writer.write(tone)
    _write_video(path, _gray_frames([64] * 48), 24, audio=wav)
    facts = stream_facts(path)
    assert facts["video"]["width"] == WIDTH
    assert facts["video"]["height"] == HEIGHT
    assert facts["video"]["fps"] == 24
    assert facts["audio"]["sample_rate"] == RATE
    assert facts["audio"]["channels"] == 2
    assert abs(facts["duration"] - 2.0) < 0.1


def test_qa_report_fields_and_exclusive_create(tmp_path):
    path = tmp_path / "show.mkv"
    wav = tmp_path / "show.wav"
    with AudioWriter(wav, rate=RATE) as writer:
        writer.write(_sine(2, 0.1368))
    _write_video(path, _gray_frames([90] * 48), 24, audio=wav)
    out = tmp_path / "qa" / "show-qa.json"
    result = qa_report(path, output_json=out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    for key in ("stream", "flash", "loudness", "silence", "clipping"):
        assert key in saved
    assert saved["human_listening"] == "NOT_RUN"
    assert saved["human_visual"] == "NOT_RUN"
    assert saved["verdicts"]["flash"] == "PASS"
    assert saved["overall"] in ("PASS", "REVIEW", "FAIL")
    assert result["stream"]["video"]["width"] == WIDTH
    with pytest.raises(FileExistsError):
        qa_report(path, output_json=out)
    replaced = qa_report(path, output_json=out, overwrite=True)
    assert replaced["schema"] == "music_qa/1"


def test_summary_lines_survive_interleaved_ffmpeg_logs():
    spliced = "Peak:      -17.3 dBFSEncoder thread received EOF"
    assert float(qa._PEAK_LINE.match(spliced).group(1)) == -17.3
    assert float(qa._LUFS_LINE.match("I:   -18.0 LUFS[out#0] done").group(1)) == -18.0
    assert float(qa._LRA_LINE.match("LRA:   4.2 LUTerminating").group(1)) == 4.2
