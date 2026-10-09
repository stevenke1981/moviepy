"""Tests for the long-form music renderer (loop + overlay + ASS + audio)."""

import json
import re
import subprocess

import numpy as np

import pytest

from moviepy.ae.composition import Composition
from moviepy.ae.templates._audio_io import AudioWriter
from moviepy.ae.templates.music_render import (
    frame_count,
    preview_window,
    render_music_video,
)
from moviepy.ae.templates.visual_loop import export_loop, seamless_loop
from moviepy.config import FFMPEG_BINARY
from tests.ae.test_music_visual import source_clip


FPS = 12
SIZE = (160, 90)
OPTIONS = dict(fps=FPS, size=SIZE, encoder="libx264", quality="draft", workers=2)


def red_box(width=40, height=20):
    """Overlay factory: an opaque red box (top level so workers can pickle it)."""
    comp = Composition(size=(width, height), fps=FPS, duration=60.0, transparent=True)
    comp.add_solid(color=(255, 0, 0), size=(width, height))
    return comp


def run_ffmpeg(*args):
    return subprocess.run(
        [FFMPEG_BINARY, "-hide_banner", *args], capture_output=True, check=True
    )


def decode(path):
    run = run_ffmpeg(
        "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    )
    return np.frombuffer(run.stdout, np.uint8).reshape(-1, SIZE[1], SIZE[0], 3)


def info(path):
    return subprocess.run(
        [FFMPEG_BINARY, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    ).stderr


def duration_of(path):
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info(path)).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    folder = tmp_path_factory.mktemp("music")
    comp = seamless_loop(
        source_clip(size=(96, 54)), length=4.0, crossfade=1.0, fps=FPS, size=SIZE
    )
    loop = folder / "loop.mp4"
    export_loop(comp, loop, encoder="libx264", quality="draft")  # 3 s, 36 frames
    t = np.arange(48000 * 4) / 48000
    tone = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav = folder / "tone.wav"
    with AudioWriter(wav, rate=48000, channels=2, subtype="PCM_16") as writer:
        writer.write(np.stack([tone, tone], axis=1))
    return {"folder": folder, "loop": loop, "wav": wav}


def render(media, name, **kw):
    options = dict(OPTIONS, background=media["loop"], audio=media["wav"])
    options.update(kw)
    render_music_video(media["folder"] / name, **options)
    return media["folder"] / name


def test_frame_count_alignment():
    assert frame_count(4.0, 12) == 48
    with pytest.raises(ValueError):
        frame_count(4.01, 12)


def test_exact_frames_duration_audio_and_evidence(media):
    out = render(media, "plain.mp4")
    assert decode(out).shape[0] == 48
    assert abs(duration_of(out) - 4.0) < 0.06
    assert "Audio: aac" in info(out) and "48000 Hz" in info(out)
    evidence = json.loads((media["folder"] / "plain.mp4.json").read_text("utf-8"))
    assert evidence["frames"] == 48 and evidence["seconds"] == 4.0
    for key in ("command", "frames", "seconds", "encoder", "overlay", "elapsed"):
        assert key in evidence
    assert evidence["encoder"] == "libx264" and evidence["overlay"] is None
    assert evidence["human_visual"] == "NOT_RUN"
    assert evidence["human_listening"] == "NOT_RUN"
    assert "-stream_loop" in evidence["command"]
    assert (media["folder"] / "plain.mp4.ffmpeg.log").is_file()


def test_background_loops_past_one_period(media):
    plain = decode(render(media, "loopcheck.mp4"))
    source = decode(media["loop"])
    assert len(source) == 36
    # Frame 40 of the render is frame 4 of the second pass of the loop.
    assert np.abs(plain[40].astype(int) - source[4].astype(int)).mean() < 12


def test_exclusive_create_and_overwrite(media):
    render(media, "once.mp4")
    with pytest.raises(FileExistsError):
        render(media, "once.mp4")
    render(media, "once.mp4", overwrite=True)
    assert decode(media["folder"] / "once.mp4").shape[0] == 48


def test_duration_must_be_frame_aligned(media):
    with pytest.raises(ValueError):
        render(media, "bad.mp4", duration=3.01)
    with pytest.raises(ValueError):
        render(media, "long.mp4", duration=9.0)
    assert not (media["folder"] / "bad.mp4").exists()
    wav = media["folder"] / "odd.wav"
    with AudioWriter(wav, rate=48000, channels=1, subtype="PCM_16") as writer:
        writer.write(np.zeros((48000 * 3 + 1000, 1), np.float32))
    with pytest.raises(ValueError):
        render(media, "odd.mp4", audio=wav)


def test_overlay_appears_at_position(media):
    base = decode(render(media, "nobox.mp4"))
    render(
        media,
        "box.mp4",
        overlay=(red_box, ()),
        overlay_position=(100, 50),
        duration=2.0,
    )
    frames = decode(media["folder"] / "box.mp4")
    assert len(frames) == 24
    inside = frames[10, 55:65, 110:130].astype(int)
    assert inside[..., 0].mean() > 200 and inside[..., 1].mean() < 70
    outside = frames[10, 0:40, 0:80].astype(int)
    assert np.abs(outside - base[10, 0:40, 0:80].astype(int)).mean() < 6
    evidence = json.loads((media["folder"] / "box.mp4.json").read_text("utf-8"))
    assert evidence["overlay"]["position"] == [100, 50]
    assert evidence["overlay"]["size"] == [40, 20]


def test_overlay_composition_object_sequential(media):
    render(media, "box2.mp4", overlay=red_box(), overlay_position=(0, 0), duration=1.0)
    frame = decode(media["folder"] / "box2.mp4")[5]
    assert frame[2:18, 2:38, 0].mean() > 200


ASS = (
    "[Script Info]\nScriptType: v4.00+\nPlayResX: 160\nPlayResY: 90\n\n"
    "[V4+ Styles]\n"
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
    "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
    "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
    "MarginR, MarginV, Encoding\n"
    "Style: Default,Arial,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,"
    "0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n\n"
    "[Events]\n"
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
    "Effect, Text\n"
    "Dialogue: 0,0:00:00.00,0:00:10.00,Default,,0,0,0,,"
    "{\\pos(0,0)\\1c&H00FF00&\\p1}m 0 0 l 50 0 50 30 0 30{\\p0}\n"
)


def test_ass_burn_changes_pixels(media):
    plain = decode(render(media, "noass.mp4", duration=1.0))
    ass = media["folder"] / "subs.ass"
    ass.write_text(ASS, encoding="utf-8")
    render(media, "ass.mp4", ass=ass, duration=1.0)
    burned = decode(media["folder"] / "ass.mp4")
    patch = burned[3, 5:25, 5:45].astype(int)
    assert patch[..., 1].mean() > 200 and patch[..., 0].mean() < 80
    assert np.abs(burned[3, 60:, 100:].astype(int) - plain[3, 60:, 100:]).mean() < 6


def test_ass_name_validation(media, tmp_path):
    bad = tmp_path / "a,b.ass"
    bad.write_text(ASS, encoding="utf-8")
    with pytest.raises(ValueError):
        render(media, "badass.mp4", ass=bad, duration=1.0)
    with pytest.raises(FileNotFoundError):
        render(media, "missing.mp4", ass=tmp_path / "none.ass", duration=1.0)


def test_chapters_present(media):
    meta = media["folder"] / "chapters.ffmeta"
    meta.write_text(
        ";FFMETADATA1\ntitle=Test\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=2000\ntitle=First\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=2000\nEND=4000\ntitle=Second\n",
        encoding="utf-8",
    )
    out = render(media, "chap.mp4", chapters=meta, metadata={"artist": "Haven"})
    text = run_ffmpeg("-v", "error", "-i", str(out), "-f", "ffmetadata", "-")
    text = text.stdout.decode("utf-8")
    assert text.count("[CHAPTER]") == 2
    assert "title=First" in text and "title=Second" in text
    assert "artist=Haven" in text


def test_chapters_rejected_with_start(media):
    meta = media["folder"] / "c2.ffmeta"
    meta.write_text(";FFMETADATA1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        render(media, "c2.mp4", chapters=meta, start=1.0, duration=1.0)


def test_preview_window_keeps_audio_and_loop_position(media):
    report = preview_window(
        media["folder"] / "prev.mp4",
        background=media["loop"],
        audio=media["wav"],
        start=2.0,
        seconds=2.0,
        **OPTIONS,
    )
    out = media["folder"] / "prev.mp4"
    assert report["frames"] == 24 and report["start"] == 2.0
    assert "Audio: aac" in info(out)
    assert abs(duration_of(out) - 2.0) < 0.06
    prev = decode(out)
    assert prev.shape[0] == 24
    full = decode(render(media, "full.mp4"))
    assert np.abs(prev[0].astype(int) - full[24].astype(int)).mean() < 10
    # The audio is the right slice: the tone fills the whole window.
    pcm = run_ffmpeg(
        "-v", "error", "-i", str(out), "-f", "s16le", "-ac", "1", "-ar", "48000", "-"
    ).stdout
    samples = np.frombuffer(pcm, np.int16)
    assert abs(len(samples) - 96000) < 2200
    assert np.abs(samples[5000:90000]).mean() > 1000


def test_still_image_background(media, tmp_path):
    from PIL import Image

    image = tmp_path / "still.png"
    Image.new("RGB", (320, 180), (10, 200, 30)).save(image)
    render(media, "still.mp4", background=image, duration=1.0)
    frame = decode(media["folder"] / "still.mp4")[3]
    assert abs(int(frame[..., 1].mean()) - 200) < 12
