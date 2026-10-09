"""Delivery packaging: ffmetadata escaping and a real tiny mux."""

import math
import struct
import wave

import pytest

from moviepy import ColorClip
from moviepy.ae.templates.delivery import (
    chapters_ffmetadata,
    iso639_2,
    mux_delivery,
)
from moviepy.ae.templates.subtitles import Cue, to_srt


BS = chr(92)


def test_ffmetadata_escaping():
    text = chapters_ffmetadata(
        [
            (0, 1.2345, "a=b;c#d" + BS + "e" + chr(10) + "f"),
            {"start": 1.2345, "end": 3, "title": "B"},
        ]
    )
    lines = text.split("\n")
    assert lines[0] == ";FFMETADATA1"
    assert "START=0" in lines and "END=1234" in lines or "END=1235" in lines
    assert "TIMEBASE=1/1000" in lines
    assert "title=a" + BS + "=b" + BS + ";c" + BS + "#d" + BS * 2 + "e" + BS in lines
    assert "f" in lines
    assert lines.count("[CHAPTER]") == 2
    assert "START=1234" in lines or "START=1235" in lines


@pytest.mark.parametrize(
    "bad", [(2, 1, "x"), (-1, 1, "x"), (0, 1, "  ")], ids=["order", "neg", "title"]
)
def test_ffmetadata_rejects_bad_chapters(bad):
    with pytest.raises(ValueError):
        chapters_ffmetadata([bad])


def test_language_codes():
    assert [iso639_2(x) for x in ("zh-TW", "zh-CN", "en", "ja", "FRA")] == [
        "zho",
        "zho",
        "eng",
        "jpn",
        "fra",
    ]
    with pytest.raises(ValueError):
        iso639_2("klingon")


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    folder = tmp_path_factory.mktemp("delivery")
    video = folder / "silent.mp4"
    ColorClip((64, 36), color=(200, 80, 40), duration=2).with_fps(10).write_videofile(
        str(video), codec="libx264", audio=False, logger=None
    )
    wav = folder / "tone.wav"
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * n / 16000)))
                for n in range(32000)
            )
        )
    srts = {}
    for lang, text in (("zh-TW", "你好"), ("en", "Hello")):
        path = folder / f"{lang}.srt"
        path.write_text(to_srt([Cue(0, 1.5, text)]), encoding="utf-8")
        srts[lang] = path
    return folder, video, wav, srts


def test_real_mux_reports_tracks_and_chapters(media):
    folder, video, wav, srts = media
    out = folder / "final.mp4"
    report = mux_delivery(
        video,
        out,
        audio=wav,
        subtitles=srts,
        chapters=[(0, 1, "Intro"), (1, 2, "Main")],
        default_subtitle="zh-TW",
    )
    assert out.is_file()
    assert not list(folder.glob("*.muxing.mp4"))
    assert not list(folder.glob("*.ffmetadata"))
    assert report["probe"] == "ffmpeg"
    assert report["chapters"] == 2
    assert report["duration"] == pytest.approx(2.0, abs=0.3)
    kinds = [(s["type"], s["codec"]) for s in report["streams"]]
    assert kinds == [
        ("video", "h264"),
        ("audio", "aac"),
        ("subtitle", "mov_text"),
        ("subtitle", "mov_text"),
    ]
    subs = [s for s in report["streams"] if s["type"] == "subtitle"]
    assert [s["language"] for s in subs] == ["zho", "eng"]
    assert [s["title"] for s in subs] == ["繁體中文（臺灣）", "English"]


def test_overwrite_refused_then_allowed(media):
    folder, video, wav, srts = media
    out = folder / "again.mp4"
    mux_delivery(video, out, audio=wav)
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        mux_delivery(video, out, audio=wav)
    assert out.read_bytes() == before
    report = mux_delivery(
        video, out, audio=wav, subtitles={"en": srts["en"]}, overwrite=True
    )
    assert [s["type"] for s in report["streams"]].count("subtitle") == 1


def test_bad_inputs(media):
    folder, video, wav, srts = media
    with pytest.raises(FileNotFoundError):
        mux_delivery(video, folder / "x.mp4", audio=folder / "missing.wav")
    with pytest.raises(ValueError):
        mux_delivery(video, folder / "y.mp4", default_subtitle="en")
    assert not (folder / "x.mp4").exists()
