"""Tests for seamless loops, scene cycles and loop export."""

import re
import subprocess

import numpy as np

import pytest

from moviepy import VideoClip
from moviepy.ae.templates.visual_loop import export_loop, scene_cycle, seamless_loop
from moviepy.config import FFMPEG_BINARY


FPS = 12
SIZE = (160, 90)


def source_clip(duration=7.0, phase=0.0, size=(96, 54)):
    """Return a smooth moving pattern that is not periodic."""
    width, height = size
    xs = np.arange(width, dtype=np.float64)[None, :, None]
    ys = np.arange(height, dtype=np.float64)[:, None, None]
    shift = np.array([0.0, 2.0, 4.0])[None, None, :]

    def frame(t):
        wave = np.sin(0.15 * xs + 0.1 * ys + 1.3 * t + phase + shift)
        return np.broadcast_to(127.5 + 120 * wave, (height, width, 3)).astype(np.uint8)

    clip = VideoClip(frame, duration=duration)
    clip.fps = FPS
    return clip


def diff(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def test_seamless_loop_duration_and_cover_fit():
    comp = seamless_loop(
        source_clip(), length=5.0, crossfade=1.0, fps=FPS, size=SIZE, start=1.0
    )
    assert comp.duration == 4.0
    assert comp.size == SIZE
    assert comp.get_frame(0).shape == (90, 160, 3)


def test_seamless_loop_matches_xfade_definition():
    src = source_clip()
    comp = seamless_loop(src, length=5.0, crossfade=1.0, fps=FPS, size=(96, 54))
    # Before the dissolve the output is the source shifted by the crossfade.
    assert diff(comp.get_frame(1.0), src.get_frame(2.0)) < 2
    # Halfway through the dissolve it is an equal mix of tail and head.
    mix = (
        src.get_frame(1.0 + 3.5).astype(float) + src.get_frame(0.5).astype(float)
    ) / 2
    assert diff(comp.get_frame(3.5), mix) < 4


def test_seamless_loop_first_last_continuity():
    src = source_clip()
    comp = seamless_loop(src, length=5.0, crossfade=1.0, fps=FPS, size=(96, 54))
    frames = [comp.get_frame(i / FPS) for i in range(int(comp.duration * FPS))]
    steps = [diff(a, b) for a, b in zip(frames, frames[1:])]
    wrap = diff(frames[-1], frames[0])
    assert wrap <= 1.5 * max(steps)
    # The very next frame after the last one is the first: one more step.
    assert wrap <= 2 * float(np.median(steps)) + 1


def test_seamless_loop_rejects_bad_timing():
    with pytest.raises(ValueError):
        seamless_loop(source_clip(3.0), length=5.0, fps=FPS, size=SIZE)
    with pytest.raises(ValueError):
        seamless_loop(source_clip(), length=2.0, crossfade=1.0, fps=FPS, size=SIZE)


def test_seamless_loop_from_video_path(tmp_path):
    path = tmp_path / "src.mp4"
    subprocess.run(
        [FFMPEG_BINARY, "-v", "error", "-f", "lavfi", "-i",
         f"testsrc=size=96x54:rate={FPS}:duration=4", "-pix_fmt", "yuv420p",
         str(path)],
        check=True,
    )  # fmt: skip
    comp = seamless_loop(path, length=3.0, crossfade=1.0, fps=FPS, size=SIZE)
    assert comp.duration == 2.0
    assert comp.get_frame(0).shape == (90, 160, 3)


def test_scene_cycle_duration_and_seams():
    loops = [seamless_loop(source_clip(phase=p), length=4.0, crossfade=1.0, fps=FPS,
                           size=(96, 54)) for p in (0.0, 2.0)]  # fmt: skip
    cycle = scene_cycle(loops, segment=6.0, crossfade=1.0, fps=FPS, size=(96, 54))
    assert cycle.duration == 12.0
    frames = [cycle.get_frame(i / FPS) for i in range(int(12 * FPS))]
    steps = [diff(a, b) for a, b in zip(frames, frames[1:])]
    # Middle of a scene: no dissolve, scene identity visible.
    scene0 = loops[0].get_frame((3.0 + 1.0) % loops[0].duration)
    assert diff(cycle.get_frame(3.0), scene0) < 3
    # The scene seam and the wrap (last -> first) move no more than a step.
    assert diff(frames[-1], frames[0]) <= 1.5 * max(steps)
    assert max(steps) < 20


def test_scene_cycle_validation():
    with pytest.raises(ValueError):
        scene_cycle([], segment=6.0, fps=FPS, size=SIZE)
    with pytest.raises(ValueError):
        scene_cycle([source_clip()], segment=1.0, crossfade=2.0, fps=FPS, size=SIZE)


def decode(path, size):
    run = subprocess.run(
        [FFMPEG_BINARY, "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    data = np.frombuffer(run.stdout, np.uint8)
    return data.reshape(-1, size[1], size[0], 3)


def test_export_loop_gop_no_bframes_and_evidence(tmp_path):
    comp = seamless_loop(source_clip(), length=5.0, crossfade=1.0, fps=FPS, size=SIZE)
    path = tmp_path / "loop.mp4"
    evidence = export_loop(comp, path, encoder="libx264", quality="draft")
    assert evidence["frames"] == 48
    assert evidence["encoder"] == "libx264"
    assert evidence["human_visual"] == "NOT_RUN"
    assert re.fullmatch(r"[0-9a-f]{64}", evidence["frames_sha256"])
    seam = evidence["seam"]
    assert seam["last_to_first_mean_abs_diff"] <= 1.5 * seam["max_step_mean_abs_diff"]
    run = subprocess.run(
        [FFMPEG_BINARY, "-hide_banner", "-i", str(path), "-vf", "showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    infos = re.findall(r"n:\s*(\d+) .*?iskey:(\d) type:(\w)", run.stderr)
    assert len(infos) == 48
    assert not any(kind == "B" for _, _, kind in infos)
    keys = [int(n) for n, key, _ in infos if key == "1"]
    assert keys == [0, 12, 24, 36]
    assert "yuv420p" in run.stderr
    assert decode(path, SIZE).shape[0] == 48


def test_export_loop_exclusive_and_overwrite(tmp_path):
    comp = seamless_loop(source_clip(), length=3.0, crossfade=1.0, fps=FPS, size=SIZE)
    path = tmp_path / "loop.mp4"
    first = export_loop(comp, path, encoder="libx264", quality="draft")
    with pytest.raises(FileExistsError):
        export_loop(comp, path, encoder="libx264", quality="draft")
    again = export_loop(comp, path, encoder="libx264", quality="draft", overwrite=True)
    assert again["frames_sha256"] == first["frames_sha256"]
