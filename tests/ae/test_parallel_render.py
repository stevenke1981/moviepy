"""Tests for multi-process rendering and encoder selection (``moviepy.ae.parallel``)."""

import hashlib
import json
import multiprocessing
import shutil
import subprocess
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

import pytest

from moviepy import VideoClip
from moviepy.ae import Composition, parallel
from moviepy.ae.properties import Property
from moviepy.ae.templates import episode as ep
from tests.ae import test_templates_episode as base


SIZE = (160, 90)
FPS = 12
needs_ffprobe = pytest.mark.skipif(
    shutil.which("ffprobe") is None, reason="ffprobe not installed"
)


def scene(seconds):
    """A small keyframed composition (picklable factory for the workers)."""
    comp = Composition(size=SIZE, fps=FPS, duration=seconds)
    comp.add_solid("bg", color=(20, 40, 60))
    spin = comp.add_solid("spin", color=(250, 200, 10), size=(60, 30))
    spin.transform.position = Property(
        (0.0, 0.0), keyframes=[(0, (20.0, 10.0)), (seconds, (100.0, 50.0))]
    )
    spin.transform.rotation = Property(0.0, keyframes=[(0, 0.0), (seconds, 120.0)])
    spin.transform.opacity = 85.0
    return comp


def failing(seconds):
    """A clip whose frame function fails after half a second."""

    def frame(t):
        if t > 0.5:
            raise ValueError("boom at %.2f" % t)
        return np.zeros((SIZE[1], SIZE[0], 3), np.uint8)

    return VideoClip(frame, duration=seconds).with_fps(FPS)


def no_children():
    for _ in range(50):
        if not multiprocessing.active_children():
            return True
        time.sleep(0.1)
    return False


def segments_gone(names):
    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)
    return not parallel._SEGMENTS


@pytest.fixture
def media(tmp_path):
    for name, color in base.COLORS.items():
        base._image(tmp_path / f"{name}.png", color)
    base._image(tmp_path / "bg.png", (90, 90, 90))
    return tmp_path


def episode_spec(media):
    return base._mini(media, audio=base._audio(media, narration_s=2.0))


# --------------------------------------------------------------------------- #


def test_frames_identical_ordered_and_clean(monkeypatch):
    names = []
    original = shared_memory.SharedMemory

    def spy(*args, **kwargs):
        segment = original(*args, **kwargs)
        if kwargs.get("create"):
            names.append(segment.name)
        return segment

    monkeypatch.setattr(parallel.shared_memory, "SharedMemory", spy)
    monkeypatch.setattr(parallel, "_SLOT_BUDGET", 0)  # smallest ring: 3 x workers
    clip = scene(4.0)
    # uneven chunking, out-of-order and repeated times
    times = [k / FPS for k in range(37)] + [0.25, 3.0, 0.25]
    expected = [clip.get_frame(t) for t in times]
    got = list(parallel.iter_frames_parallel(scene, (4.0,), times, workers=3, chunk=5))
    assert len(got) == len(expected)
    for frame, reference in zip(got, expected):
        assert frame.dtype == np.uint8 and frame.tobytes() == reference.tobytes()
    assert no_children() and names and segments_gone(names)


def test_sequential_path_matches_and_uses_given_clip():
    clip = scene(2.0)
    times = [k / FPS for k in range(10)]
    got = list(parallel.iter_frames_parallel(scene, (2.0,), times, workers=1))
    again = list(parallel.iter_frames_parallel(None, (), times, workers=0, clip=clip))
    for a, b, t in zip(got, again, times):
        assert a.tobytes() == b.tobytes() == clip.get_frame(t).tobytes()
    assert list(parallel.iter_frames_parallel(scene, (2.0,), [], workers=2)) == []


def test_worker_exception_propagates_and_cleans_up():
    times = [k / FPS for k in range(30)]
    with pytest.raises(RuntimeError) as info:
        list(parallel.iter_frames_parallel(failing, (3.0,), times, workers=2))
    assert "ValueError: boom at" in str(info.value) and "Traceback" in str(info.value)
    assert no_children() and not parallel._SEGMENTS


def test_closing_the_generator_early_cleans_up():
    times = [k / FPS for k in range(60)]
    frames = parallel.iter_frames_parallel(scene, (6.0,), times, workers=2)
    next(frames)
    next(frames)
    assert multiprocessing.active_children() and parallel._SEGMENTS
    frames.close()
    assert no_children() and not parallel._SEGMENTS


@needs_ffprobe
def test_write_video_parallel_counts_audio_and_frame_identity(media, tmp_path):
    narration = base._wav(media / "voice.wav", 1.0, 0.3, 22050)  # shorter: padded
    out = tmp_path / "clip.mp4"
    hashes = []
    for workers in (1, 2):
        digest = hashlib.md5()
        result = parallel.write_video_parallel(
            scene, (2.5,), out, fps=FPS, duration=2.5, size=SIZE,
            audio_path=narration, workers=workers, encoder="libx264",
            frame_hook=lambda frame, digest=digest: digest.update(frame.tobytes()),
        )  # fmt: skip
        hashes.append(digest.hexdigest())
        assert result["encoder"] == "libx264" and result["frames"] == 30
        assert result["workers"] == workers and result["fps_achieved"] > 0
    assert hashes[0] == hashes[1]
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-show_entries",
         "stream=codec_type,codec_name,nb_read_frames,duration", "-of", "json",
         str(out)], capture_output=True, text=True, check=True,
    )  # fmt: skip
    streams = {s["codec_type"]: s for s in json.loads(run.stdout)["streams"]}
    assert streams["video"]["codec_name"] == "h264"
    assert int(streams["video"]["nb_read_frames"]) == 30
    assert streams["audio"]["codec_name"] == "aac"
    assert float(streams["audio"]["duration"]) == pytest.approx(2.5, abs=0.1)


def test_write_video_parallel_argument_errors(tmp_path):
    with pytest.raises(ValueError, match="duration or n_frames"):
        parallel.write_video_parallel(scene, (1.0,), tmp_path / "x.mp4", fps=FPS)
    with pytest.raises(ValueError, match="n_frames"):
        parallel.write_video_parallel(
            scene, (1.0,), tmp_path / "x.mp4", fps=FPS, n_frames=0
        )


# --------------------------------------------------------------------------- #
# encoders


class FakeRun:
    def __init__(self, calls, nvenc_ok, listed=("libx264", "libx265")):
        self.calls, self.nvenc_ok, self.listed = calls, nvenc_ok, listed

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if "-encoders" in command:
            text = "".join(f" V..... {name}  desc\n" for name in self.listed)
            return subprocess.CompletedProcess(command, 0, text, "")
        code = 0 if self.nvenc_ok or "nvenc" not in " ".join(command) else 1
        return subprocess.CompletedProcess(command, code, b"", b"no device")


def test_detect_encoder_falls_back_and_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(parallel, "_ENCODER_CACHE", {})
    monkeypatch.setattr(parallel.subprocess, "run", FakeRun(calls, nvenc_ok=False))
    assert parallel.detect_encoder() == "libx264"
    assert parallel.detect_encoder("auto") == "libx264"
    assert sum("h264_nvenc" in c for c in calls) == 1  # cached
    with pytest.raises(parallel.EncoderUnavailable, match="h264_nvenc"):
        parallel.detect_encoder("h264_nvenc")
    assert parallel.detect_encoder("libx265") == "libx265"
    with pytest.raises(parallel.EncoderUnavailable, match="unknown"):
        parallel.detect_encoder("vp9")


def test_detect_encoder_prefers_working_nvenc(monkeypatch):
    monkeypatch.setattr(parallel, "_ENCODER_CACHE", {})
    monkeypatch.setattr(parallel.subprocess, "run", FakeRun([], nvenc_ok=True))
    assert parallel.detect_encoder() == "h264_nvenc"
    assert parallel.detect_encoder("hevc_nvenc") == "hevc_nvenc"
    assert parallel.detect_encoder("libx264") == "libx264"


def test_detect_encoder_survives_a_missing_binary(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(parallel, "_ENCODER_CACHE", {})
    monkeypatch.setattr(parallel.subprocess, "run", boom)
    assert parallel.detect_encoder() == "libx264"
    with pytest.raises(parallel.EncoderUnavailable):
        parallel.detect_encoder("libx264")


def option(args, flag):
    return args[args.index(flag) + 1]


def test_encoder_args():
    x264 = parallel.encoder_args("libx264")
    assert option(x264, "-c:v") == "libx264" and option(x264, "-crf") == "18"
    assert option(x264, "-preset") == "medium"
    nv = parallel.encoder_args("h264_nvenc")
    assert option(nv, "-preset") == "p5" and option(nv, "-tune") == "hq"
    assert option(nv, "-rc") == "vbr" and option(nv, "-cq") == "19"
    assert option(nv, "-b:v") == "0" and option(nv, "-bf") == "2"
    hevc = parallel.encoder_args("hevc_nvenc", "balanced")
    assert option(hevc, "-c:v") == "hevc_nvenc" and "-bf" not in hevc
    assert option(hevc, "-tag:v") == "hvc1"
    assert option(parallel.encoder_args("libx265"), "-tag:v") == "hvc1"
    for args in (x264, nv, hevc):
        assert option(args, "-pix_fmt") == "yuv420p"
        assert option(args, "-movflags") == "+faststart"
    with pytest.raises(ValueError, match="quality"):
        parallel.encoder_args("libx264", "ultra")
    with pytest.raises(parallel.EncoderUnavailable):
        parallel.encoder_args("vp9")


def test_real_nvenc_encode_when_available(tmp_path):
    if parallel.detect_encoder() != "h264_nvenc":
        pytest.skip("no working NVENC on this machine")
    out = tmp_path / "nv.mp4"
    result = parallel.write_video_parallel(
        scene, (1.0,), out, fps=FPS, duration=1.0, size=SIZE, workers=1,
        encoder="h264_nvenc",
    )  # fmt: skip
    assert result["encoder"] == "h264_nvenc" and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# episode integration


def test_render_episode_workers_match_sequential(media, tmp_path, monkeypatch):
    spec = episode_spec(media)
    digests = {}
    original = parallel.write_video_parallel
    for workers in (1, 2):
        digest = hashlib.md5()

        def hooked(*args, digest=digest, **kwargs):
            kwargs["frame_hook"] = lambda frame: digest.update(frame.tobytes())
            return original(*args, **kwargs)

        monkeypatch.setattr(parallel, "write_video_parallel", hooked)
        out = tmp_path / f"w{workers}"
        result = ep.render_episode(
            spec, out, preview=True, stills=(), workers=workers, encoder="libx264"
        )
        digests[workers] = digest.hexdigest()
        assert result["workers"] == workers and result["encoder"] == "libx264"
        assert result["render_fps"] > 0
        assert not list(out.glob("*.tmp.wav"))
        assert Path(result["video"]).stat().st_size > 0
    assert digests[1] == digests[2]


def test_cli_flags(media, tmp_path, capsys):
    path = tmp_path / "episode.json"
    episode_spec(media).to_json(path)
    args = ["render", str(path), str(tmp_path / "o"), "--preview"]
    assert ep.main([*args, "--workers", "1", "--encoder", "libx264"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["workers"] == 1 and report["encoder"] == "libx264"
    with pytest.raises(SystemExit):
        ep.main([*args, "--encoder", "vp9"])
    with pytest.raises(SystemExit):
        ep.main([*args, "--workers", "many"])
