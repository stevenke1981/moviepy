"""Tests for the declarative episode build (``moviepy.ae.templates.episode``)."""

import hashlib
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image

import pytest

from moviepy.ae.templates import episode as ep
from moviepy.ae.templates.episode import (
    EpisodeError,
    EpisodeSpec,
    build_episode,
    episode_report,
)


KAIU = Path("C:/Windows/Fonts/kaiu.ttf")
CONFIGS = Path(ep.__file__).parent / "configs"
needs_kaiu = pytest.mark.skipif(not KAIU.is_file(), reason="kaiu.ttf not installed")
COLORS = {"a": (220, 40, 40), "b": (40, 220, 40), "c": (40, 40, 220)}
SMALL = {"size": [320, 180], "fps": 12, "hold": 1.0}


def _image(path, color, size=(160, 90)):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), color, np.uint8)).save(path)
    return str(path)


def _wav(path, seconds, amplitude, rate=22050, freq=440.0, channels=2):
    """Write a sine WAV (16-bit PCM) and return its path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * rate)) / rate
    wave_ = np.sin(2 * np.pi * freq * t) * amplitude
    data = np.repeat(wave_[:, None], channels, 1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((data * 32767).astype("<i2").tobytes())
    return str(path)


@pytest.fixture
def media(tmp_path):
    """Generated placeholders: three flat shots, a background and an SRT."""
    for name, color in COLORS.items():
        _image(tmp_path / f"{name}.png", color)
    _image(tmp_path / "bg.png", (90, 90, 90))
    (tmp_path / "zh.srt").write_text(
        "1\n00:00:05,000 --> 00:00:07,000\nHELLO SUBTITLE\n", encoding="utf-8"
    )
    return tmp_path


def _data(media):
    """Latin-only episode (Pillow default fonts): intro, 3 shots, outro."""
    return {
        "preset": "nightlamp_story",
        "preset_overrides": dict(SMALL),
        "fonts": {"title": None, "body": None, "subtitle": None},
        "background": {"source": str(media / "bg.png"), "overlay_opacity": 0.5},
        "intro": {"title": "Open", "brand": "Lamp", "duration": 3},
        "outro": {"title": "Close", "duration": 3},
        "shots": [
            {"image": str(media / "a.png"), "duration": 4, "move": "push"},
            {
                "image": str(media / "b.png"),
                "duration": 4,
                "move": "push",
                "transition": "crossfade",
                "transition_duration": 0.5,
            },
            {
                "image": str(media / "c.png"),
                "duration": 4,
                "move": "pan-right",
                "transition": "dip",
                "transition_duration": 0.6,
            },
        ],
        "subtitles": {"primary": str(media / "zh.srt"), "primary_lang": "en"},
    }


def _spec(media, **extra):
    return EpisodeSpec.from_dict({**_data(media), **extra})


def _layers(comp):
    return {layer.name: layer for layer in comp.layers}


def _mean(comp, t):
    return comp.get_frame(t).reshape(-1, 3).mean(axis=0)


def _toggle_diff(comp, name, t):
    """Sum of absolute pixel change caused by layer ``name`` at time ``t``."""
    layer = _layers(comp)[name]
    shown = comp.get_frame(t).astype(int)
    layer.enabled = False
    hidden = comp.get_frame(t).astype(int)
    layer.enabled = True
    return np.abs(shown - hidden)


# --------------------------------------------------------------------------- #
# JSON and validation
# --------------------------------------------------------------------------- #


def test_json_round_trip(media):
    spec = _spec(
        media,
        chapters=[{"start": 5, "end": 9, "items": ["a", "b"], "number": 2}],
        quotes=[{"start": 10, "text": "x", "dynasty": "Tang"}],
    )
    again = EpisodeSpec.from_json(spec.to_json())
    assert again == spec
    assert again.to_json() == spec.to_json()
    assert again.shots[1].transition == "crossfade"
    inline = EpisodeSpec(
        shots=[{"video": "v.mp4", "in": 1, "out": 3}],
        subtitles={"primary": [{"start": 0, "end": 1, "text": "hi"}]},
    )
    assert EpisodeSpec.from_json(inline.to_json()) == inline
    assert json.loads(inline.to_json())["shots"][0]["in"] == 1.0


def test_shipped_configs_round_trip():
    for name in ("nightlamp_story", "nightlamp_history"):
        spec = EpisodeSpec.from_json(CONFIGS / f"{name}.json")
        assert spec.preset == name
        assert EpisodeSpec.from_json(spec.to_json()) == spec
        assert all("PLACEHOLDER" in s.source for s in spec.shots)
    history = EpisodeSpec.from_json(CONFIGS / "nightlamp_history.json")
    assert history.chapter_period == 5.5
    assert {s.zoom for s in history.shots} == {(1.0, 1.08)}
    assert {s.hold for s in history.shots} == {2.0}
    assert {s.transition_duration for s in history.shots[1:]} == {0.5}
    story = EpisodeSpec.from_json(CONFIGS / "nightlamp_story.json")
    assert story.background.overlay_opacity == 0.5
    assert story.resolve_preset().fps == 24


def test_relative_paths_resolve_against_json_folder(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    data = {
        "shots": [{"image": "media/a.png", "duration": 2}],
        "background": {"source": "media/bg.png"},
        "subtitles": {"primary": "subs/zh.srt"},
    }
    (folder / "ep.json").write_text(json.dumps(data), encoding="utf-8")
    spec = EpisodeSpec.from_json(folder / "ep.json")
    assert Path(spec.shots[0].image) == folder / "media" / "a.png"
    assert Path(spec.background.source) == folder / "media" / "bg.png"
    assert Path(spec.subtitles.primary) == folder / "subs" / "zh.srt"
    text = EpisodeSpec.from_json(json.dumps(data))  # text: left as written
    assert text.shots[0].image == "media/a.png"
    based = EpisodeSpec.from_json(json.dumps(data), base_dir=folder)
    assert Path(based.shots[0].image) == folder / "media" / "a.png"


SHOT = {"image": "a.png", "duration": 2}
CHAPTER = {"start": 1, "end": 5, "items": ["a"], "number": 1}


@pytest.mark.parametrize(
    "data, message",
    [
        ({"shots": [SHOT], "bogus": 1}, "unknown episode key"),
        ({"shots": []}, "at least one shot"),
        ({"shots": [{**SHOT, "zoomm": 1}]}, r"shots\[0\]: unknown key 'zoomm'"),
        ({"shots": [{"image": "a", "video": "b", "duration": 1}]}, "exactly one"),
        ({"shots": [{"image": "a"}]}, "needs a duration"),
        ({"shots": [{**SHOT, "move": "spin"}]}, "move must be one of"),
        ({"shots": [{**SHOT, "zoom": 0.5}]}, "zoom must be >= 1"),
        ({"shots": [{**SHOT, "transition": "wipe"}]}, "transition must be one of"),
        ({"shots": [{**SHOT, "transition_duration": 1}]}, "cut takes no"),
        ({"shots": [{**SHOT, "duration": -1}]}, "duration must be positive"),
        ({"shots": [{**SHOT, "duration": True}]}, "must be a number"),
        ({"shots": [{**SHOT, "move": "static", "zoom": 1.1}]}, "static shot"),
        ({"shots": [{"video": "v", "move": "push"}]}, "image shots only"),
        ({"shots": [{"video": "v", "in": 2, "out": 1}]}, "out must be greater"),
        ({"shots": [{**SHOT, "in": 1}]}, "video shots only"),
        ({"shots": [{**SHOT, "transition": "crossfade"}]}, "transition must be cut"),
        ({"shots": [SHOT], "preset": "nope"}, "unknown preset"),
        ({"shots": [SHOT], "preset_overrides": {"fps": -3}}, "preset_overrides"),
        ({"shots": [SHOT], "fonts": {"quote": None}}, "needs a real font"),
        ({"shots": [SHOT], "fonts": {"weird": "x"}}, "fonts key must be one of"),
        ({"shots": [SHOT], "chapters": [{**CHAPTER, "end": 0}]}, "end must be greater"),
        (
            {
                "shots": [SHOT],
                "chapters": [CHAPTER, {**CHAPTER, "start": 4, "end": 8, "number": 2}],
            },
            "chapters overlap",
        ),
        (
            {"shots": [SHOT], "chapters": [{"start": 1, "end": 2, "number": 1}]},
            "missing required key",
        ),
        ({"shots": [SHOT], "quotes": [{"start": 1}]}, "missing required key"),
        (
            {
                "shots": [SHOT],
                "subtitles": {"primary": [{"start": 2, "end": 1, "text": "x"}]},
            },
            "end must be greater",
        ),
        ({"shots": [SHOT], "subtitles": {"primary": 5}}, "SRT path or a list"),
        ({"shots": [SHOT], "intro": {"title": " "}}, "title must not be empty"),
        (
            {"shots": [SHOT], "background": {"source": "x", "overlay_opacity": 2}},
            "overlay_opacity",
        ),
    ],
)
def test_validation_errors(data, message):
    with pytest.raises(EpisodeError, match=message):
        EpisodeSpec.from_dict(data)


def test_invalid_json_and_missing_file(tmp_path):
    with pytest.raises(EpisodeError, match="invalid episode JSON"):
        EpisodeSpec.from_json("{not json")
    with pytest.raises(EpisodeError, match="cannot read"):
        EpisodeSpec.from_json(tmp_path / "missing.json")


# --------------------------------------------------------------------------- #
# build-time validation
# --------------------------------------------------------------------------- #


def test_build_rejects_transition_outside_focus_hold(media):
    data = _data(media)
    data["shots"][1]["transition_duration"] = 1.5  # the hold is 1.0 s
    with pytest.raises(EpisodeError, match="focus hold"):
        build_episode(EpisodeSpec.from_dict(data))


def test_build_rejects_bad_overlays(media):
    cases = {
        "outside the": {"start": 20, "end": 90, "items": ["a"], "number": 1},
        "overlaps the Intro": {"start": 0.5, "end": 2, "items": ["a"], "number": 1},
    }
    for message, chapter in cases.items():
        data = {**_data(media), "chapters": [chapter]}
        with pytest.raises(EpisodeError, match=message):
            build_episode(EpisodeSpec.from_dict(data))
    data = _data(media)
    data["subtitles"] = {"primary": [{"start": 30, "end": 200, "text": "late"}]}
    with pytest.raises(EpisodeError, match="outside the .* timeline"):
        build_episode(EpisodeSpec.from_dict(data))


def test_build_reports_missing_files(media):
    data = _data(media)
    data["shots"][0]["image"] = str(media / "nope.png")
    with pytest.raises(EpisodeError, match="not found"):
        build_episode(EpisodeSpec.from_dict(data))
    data = _data(media)
    data["subtitles"]["primary"] = str(media / "nope.srt")
    with pytest.raises(EpisodeError, match="subtitles.primary not found"):
        build_episode(EpisodeSpec.from_dict(data))


def test_ken_burns_errors_are_wrapped(media):
    data = _data(media)
    data["shots"][0]["duration"] = 0.5  # shorter than the 1 s hold
    with pytest.raises(EpisodeError, match=r"shots\[0\]"):
        build_episode(EpisodeSpec.from_dict(data))


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #


def test_end_to_end_frames(media):
    comp = build_episode(_spec(media))
    assert comp.size == (320, 180) and comp.fps == 12
    line = {seg["name"]: seg for seg in episode_report(comp)["timeline"]}
    intro, s1, s2, s3, outro = (
        line[n] for n in ("Intro", "Shot 01", "Shot 02", "Shot 03", "Outro")
    )
    # cut, crossfade (0.5 s overlap), dip (no overlap), cut.
    assert s1["start"] == pytest.approx(intro["end"])
    assert s2["start"] == pytest.approx(s1["end"] - 0.5)
    assert s3["start"] == pytest.approx(s2["end"])
    assert outro["start"] == pytest.approx(s3["end"])
    assert comp.duration == pytest.approx(outro["end"])

    # intro: an opaque card over everything.
    assert not np.allclose(_mean(comp, intro["start"] + 1.0), COLORS["a"], atol=30)
    # mid-shot 1: the plain red picture.
    assert np.allclose(_mean(comp, s1["start"] + 1.0), COLORS["a"], atol=3)
    # mid-crossfade: exactly half red, half green.
    half = (np.array(COLORS["a"]) + np.array(COLORS["b"])) / 2
    assert np.allclose(_mean(comp, s2["start"] + 0.25), half, atol=3)
    # the dip passes through its black; shot 3 is fully up half a second later.
    assert _mean(comp, s3["start"]).max() < 20
    assert np.allclose(_mean(comp, s3["start"] + 1.0), COLORS["c"], atol=3)
    # outro: a card again.
    assert not np.allclose(_mean(comp, outro["start"] + 1.5), COLORS["c"], atol=30)

    # subtitles are burned last: they change the lower half only, only in-window.
    change = _toggle_diff(comp, "Subtitles", 6.0).sum(axis=2)
    assert change.max() > 100 and change[: comp.size[1] // 2].max() == 0
    assert _toggle_diff(comp, "Subtitles", 8.0).max() == 0


def test_layer_order_and_report(media):
    comp = build_episode(_spec(media))
    names = [layer.name for layer in comp.layers]
    assert names == ["Subtitles", "Outro", "Intro", "Shot 03", "Shot 02", "Shot 01"]
    report = episode_report(comp)
    assert report["top_layer"] == "Subtitles"
    assert [layer["name"] for layer in report["layers"]] == names
    assert report["preset"] == "nightlamp_story"
    assert report["preset_resolved"]["fps"] == 12.0
    assert report["size"] == [320, 180]
    assert report["frames"] == int(np.ceil(report["duration"] * 12 - 1e-9))
    assert report["subtitles"]["primary"] == {
        "cues": 1,
        "rendered_cues": 1,
        "first": 5.0,
        "last": 7.0,
    }
    json.dumps(report)  # JSON-able
    sources = report["sources"]
    digest = hashlib.sha256((media / "a.png").read_bytes()).hexdigest()
    assert sources[str(media / "a.png")] == digest
    assert str(media / "zh.srt") in sources and str(media / "bg.png") in sources
    with pytest.raises(ValueError, match="not an episode"):
        episode_report(object())


def test_build_is_deterministic(media):
    spec = _spec(media)
    one, two = build_episode(spec), build_episode(spec)
    for t in (1.0, 4.5, 6.0, 8.4, 11.0, 14.0):
        assert np.array_equal(one.get_frame(t), two.get_frame(t))
    assert episode_report(one) == episode_report(two)


def test_optional_bookends_and_inline_cues(media):
    data = _data(media)
    for key in ("intro", "outro", "background"):
        data[key] = None
    data["subtitles"] = {
        "primary": [{"start": 1, "end": 2, "text": "inline", "lang": "en"}]
    }
    comp = build_episode(EpisodeSpec.from_dict(data))
    assert comp.layers[0].name == "Subtitles"
    assert comp.duration == pytest.approx(4 + 4 - 0.5 + 4)
    assert str(media / "a.png") in episode_report(comp)["sources"]


def test_static_and_video_shots(media, tmp_path):
    import moviepy

    frames = [np.full((36, 64, 3), (0, 100, 220), np.uint8) for _ in range(24)]
    video = tmp_path / "clip.mp4"
    try:
        clip = moviepy.ImageSequenceClip(frames, fps=12)
        clip.write_videofile(str(video), logger=None, audio=False)
        clip.close()
    except Exception as error:  # no ffmpeg / codec in this environment
        pytest.skip(f"cannot write a test video: {error}")
    data = {
        "preset_overrides": SMALL,
        "shots": [
            {"image": str(media / "a.png"), "duration": 2, "move": "static"},
            {
                "video": str(video),
                "in": 0.5,
                "out": 1.5,
                "transition": "crossfade",
                "transition_duration": 0.4,
            },
        ],
    }
    comp = build_episode(EpisodeSpec.from_dict(data))
    try:
        assert comp.duration == pytest.approx(2 + 1 - 0.4)
        assert np.allclose(_mean(comp, 1.0), COLORS["a"], atol=3)
        assert _mean(comp, 2.4)[2] > 150  # the blue-ish video is on screen
    finally:
        for clip in comp.episode_clips:
            clip.close()


@needs_kaiu
def test_chapters_and_quotes(media):
    data = _data(media)
    data["fonts"].update(chapter=str(KAIU), quote=str(KAIU))
    data["chapters"] = [{"start": 4.5, "end": 8, "items": ["一", "二"], "number": 1}]
    data["quotes"] = [{"start": 9, "text": "天行健君子以自強不息", "dynasty": "周"}]
    comp = build_episode(EpisodeSpec.from_dict(data))
    assert [layer.name for layer in comp.layers] == [
        "Subtitles",
        "Outro",
        "Intro",
        "Chapter 01",
        "Quote 01",
        "Shot 03",
        "Shot 02",
        "Shot 01",
    ]
    # the chapter tag is visible top-left while active, and only there.
    shown = _toggle_diff(comp, "Chapter 01", 6.0)
    assert shown.sum() > 1000
    assert shown[comp.size[1] // 2 :].max() == 0
    assert _toggle_diff(comp, "Chapter 01", 9.5).max() == 0
    # the quote is drawn over the picture while it writes.
    assert _toggle_diff(comp, "Quote 01", 12.0).sum() > 1000
    report = episode_report(comp)
    assert report["chapters"][0]["number"] == 1
    assert report["quotes"][0]["medium"] in ("bamboo", "paper")
    # overlapping quotes raise.
    data["quotes"].append({**data["quotes"][0], "start": 10})
    with pytest.raises(EpisodeError, match="quotes overlap"):
        build_episode(EpisodeSpec.from_dict(data))


@needs_kaiu
@pytest.mark.parametrize("name", ["nightlamp_story", "nightlamp_history"])
def test_shipped_configs_build(name, tmp_path):
    shutil.copy(CONFIGS / f"{name}.json", tmp_path / "ep.json")
    spec = EpisodeSpec.from_json(tmp_path / "ep.json")
    sources = [spec.background.source] + [s.image for s in spec.shots]
    for k, path in enumerate(sources):
        _image(path, list(COLORS.values())[k % 3])
    assert spec.audio is not None and spec.audio.music is None
    _wav(spec.audio.narration, 1.0, 0.3)
    small = {**spec.preset_overrides, **SMALL}

    # Pass 1 without subtitles gives the built timeline length for the SRTs.
    bare = EpisodeSpec.from_dict(
        {**spec.to_dict(), "preset_overrides": small, "subtitles": None}
    )
    total = build_episode(bare).duration
    last = int(total) - 2
    cues = f"1\n00:00:08,000 --> 00:00:10,000\n你好，世界。\n\n2\n00:00:{last:02d},000 --> 00:00:{last + 1:02d},000\n再會。\n"
    srt = Path(spec.subtitles.primary)
    srt.parent.mkdir(parents=True, exist_ok=True)
    srt.write_text(cues, encoding="utf-8")
    Path(spec.subtitles.secondary).write_text(cues, encoding="utf-8")

    full = EpisodeSpec.from_dict({**spec.to_dict(), "preset_overrides": small})
    comp = build_episode(full)
    report = episode_report(comp)
    assert report["preset"] == name
    assert report["layers"][0]["name"] == "Subtitles"
    assert report["subtitles"]["secondary"]["cues"] == 2
    shots = [s for s in report["timeline"] if s["kind"] == "shot"]
    for prev, cur in zip(shots, shots[1:]):
        assert cur["transition"] == "crossfade"
        assert prev["end"] - cur["start"] <= prev["hold"] + 1e-6
    if name == "nightlamp_history":
        assert [c["number"] for c in report["chapters"]] == [1, 2]
        assert len(report["quotes"]) == 1
    assert comp.get_frame(report["duration"] / 2).shape == (180, 320, 3)


# --------------------------------------------------------------------------- #
# audio, bookend timing, render_episode, init_episode and the CLI
# --------------------------------------------------------------------------- #

RATE = 22050


def _mini(media, **extra):
    """8 s Latin episode (fixed-length cards) cheap enough to render for real."""
    data = _data(media)
    card = {"duration": 3, "auto_extend": False, "title_start": 0.0,
            "title_duration": 0.2, "fade_out": 0.1}  # fmt: skip
    data["intro"] = {"title": "Open", **card}
    data["outro"] = {"title": "Close", **card}
    data["shots"] = [{"image": str(media / "a.png"), "duration": 2, "move": "push"}]
    data["subtitles"] = None
    data.update(extra)
    return EpisodeSpec.from_dict(data)


def _rms(comp, start, end):
    # Short windows: AudioFileClip.get_frame mis-handles spans over ~2.2 s.
    t = np.arange(int(start * RATE), int(end * RATE)) / RATE
    parts = [
        np.asarray(comp.audio.get_frame(t[k : k + RATE // 2]), dtype=float)
        for k in range(0, len(t), RATE // 2)
    ]
    return float(np.sqrt(np.mean(np.concatenate(parts) ** 2)))


def _audio(media, narration_s=2.0, **extra):
    section = {"narration": _wav(media / "voice.wav", narration_s, 0.5, RATE)}
    section.update(extra)
    return section


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _close(comp):
    for clip in comp.episode_clips:
        clip.close()


def test_audio_mix_duration_and_narration_gain(media):
    spec = _mini(media, audio=_audio(media, narration_offset=1.0, narration_gain_db=-6))
    comp = build_episode(spec)
    try:
        assert comp.audio.duration == pytest.approx(comp.duration)
        expected = 0.5 / np.sqrt(2) * 10 ** (-6 / 20)
        assert _rms(comp, 1.2, 2.8) == pytest.approx(expected, rel=0.03)
        assert _rms(comp, 0.0, 0.9) < 1e-4  # before the offset
        assert _rms(comp, 3.2, 7.9) < 1e-4  # after the narration
        report = episode_report(comp)
        assert report["audio"]["narration"]["duration"] == pytest.approx(2.0, abs=0.01)
        assert report["audio"]["narration"]["sha256"] == _sha(media / "voice.wav")
        assert report["audio"]["duration"] == pytest.approx(comp.duration)
        assert str(media / "voice.wav") in report["sources"]
    finally:
        _close(comp)


def test_music_gain_and_constant_duck_during_narration(media):
    music = _wav(media / "music.wav", 10.0, 0.5, RATE, freq=220.0)
    # Narration is muted by gain so only the music bed is measured.
    audio = _audio(
        media, narration_offset=2.0, music=music, music_gain_db=-6,
        music_duck_db=-12, narration_gain_db=-200,
    )  # fmt: skip
    comp = build_episode(_mini(media, audio=audio))
    try:
        bed = 0.5 / np.sqrt(2) * 10 ** (-6 / 20)
        assert _rms(comp, 0.2, 1.5) == pytest.approx(bed, rel=0.03)
        assert _rms(comp, 2.4, 3.8) == pytest.approx(bed * 10 ** (-12 / 20), rel=0.03)
        assert _rms(comp, 4.6, 7.9) == pytest.approx(bed, rel=0.03)
        info = episode_report(comp)["audio"]["music"]
        assert info["duck_span"] == [2.0, 4.0] and info["duck_db"] == -12
    finally:
        _close(comp)


def test_music_loop_and_no_duck(media):
    music = _wav(media / "music.wav", 1.0, 0.4, RATE)
    audio = _audio(media, music=music, music_duck_db=0, music_loop=True)
    comp = build_episode(_mini(media, audio=audio))
    try:
        assert _rms(comp, 6.5, 7.9) == pytest.approx(
            0.4 / 2**0.5 * 10 ** (-18 / 20), rel=0.05
        )
        assert episode_report(comp)["audio"]["music"]["duration"] == pytest.approx(8.0)
    finally:
        _close(comp)


def test_audio_fades(media):
    music = _wav(media / "music.wav", 10.0, 0.5, RATE)
    audio = _audio(
        media, music=music, fade_in=1.0, fade_out=1.0, narration_gain_db=-200
    )
    comp = build_episode(_mini(media, audio=audio))
    try:
        assert _rms(comp, 0.0, 0.2) < 0.3 * _rms(comp, 2.5, 2.7)
        assert _rms(comp, 7.8, 8.0) < 0.3 * _rms(comp, 2.5, 2.7)
    finally:
        _close(comp)


def test_narration_longer_than_timeline(media):
    long_audio = _audio(media, narration_s=11.0)
    with pytest.raises(EpisodeError, match="trim_audio"):
        build_episode(_mini(media, audio=long_audio))
    comp = build_episode(_mini(media, audio={**long_audio, "trim_audio": True}))
    try:
        assert comp.audio.duration == pytest.approx(comp.duration)
        assert _rms(comp, 7.0, 7.9) > 0.2
    finally:
        _close(comp)
    with pytest.raises(EpisodeError, match="past the"):
        build_episode(_mini(media, audio=_audio(media, narration_offset=8.5)))


@pytest.mark.parametrize(
    "audio, message",
    [
        ({}, "missing required key"),
        ({"narration": "nope.wav"}, "audio.narration not found"),
        ({"narration": "x", "bogus": 1}, "unknown key"),
        ({"narration": "x", "music_duck_db": 3}, "music_duck_db must be <= 0"),
        ({"narration": "x", "fade_in": -1}, "fade_in must be >= 0"),
        ({"narration": "x", "trim_audio": "yes"}, "trim_audio must be true"),
        ({"narration": "x", "narration_gain_db": "loud"}, "must be a number"),
    ],
)
def test_audio_validation_errors(media, audio, message):
    with pytest.raises(EpisodeError, match=message):
        build_episode(_mini(media, audio=audio))


def test_audio_missing_music_and_roundtrip(media):
    with pytest.raises(EpisodeError, match="audio.music not found"):
        build_episode(_mini(media, audio=_audio(media, music=str(media / "no.wav"))))
    spec = _mini(media, audio=_audio(media, music=None, fade_out=1.5))
    assert EpisodeSpec.from_json(spec.to_json()) == spec
    assert _mini(media).audio is None  # old specs stay valid


def test_audio_relative_paths_resolve_against_json_folder(tmp_path):
    folder = tmp_path / "proj"
    data = {
        "shots": [{"image": "a.png", "duration": 1}],
        "audio": {"narration": "v.wav", "music": "m.wav"},
    }
    spec = EpisodeSpec.from_dict(data, base_dir=folder)
    assert spec.audio.narration == str(folder / "v.wav")
    assert spec.audio.music == str(folder / "m.wav")


def test_bookend_timing_passthrough(media):
    timing = {
        "brand_start": 0.1, "brand_duration": 0.3, "title_start": 0.4,
        "title_duration": 0.5, "subtitle_start": 0.9, "subtitle_duration": 0.4,
        "fade_out": 0.2, "rule": False, "auto_extend": False,
        "title_keep_together": ["Open"], "subtitle_keep_together": ["x"],
    }  # fmt: skip
    data = _data(media)
    data["intro"] = {"title": "Open", "duration": 2, **timing}
    spec = EpisodeSpec.from_dict(data)
    card = spec.intro.card_spec("intro")
    for key, value in timing.items():
        assert getattr(card, key) == (
            tuple(value) if isinstance(value, list) else value
        )
    assert EpisodeSpec.from_json(spec.to_json()) == spec
    assert spec.outro.card_spec("outro").title_start == 0.55  # defaults kept
    data["intro"]["title_start"] = -1
    with pytest.raises(EpisodeError, match="title_start"):
        EpisodeSpec.from_dict(data)


def _ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,codec_type,"
         "pix_fmt,width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return json.loads(out)


def test_render_episode_preview_and_overwrite(media, tmp_path):
    narration = _wav(media / "voice.wav", 2.0, 0.3, RATE)
    spec = _mini(media, audio={"narration": narration})
    out = tmp_path / "out"
    result = ep.render_episode(spec, out, preview=True)
    assert result["preview"] is True and result["master"] is None
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert report["size"] == [480, 270] and report["fps"] == 12.0
    assert report["audio"]["narration"]["path"] == narration
    names = sorted(Path(p).name for p in result["stills"])
    assert len(names) == 2 and names[0].startswith("intro_")
    assert names[1].startswith("outro_")
    for path in result["stills"]:
        assert Image.open(path).size == (480, 270)
    assert {"build", "stills", "video", "total"} <= set(result["timings"])
    if shutil.which("ffprobe"):
        info = _ffprobe(result["video"])
        kinds = {s["codec_type"]: s for s in info["streams"]}
        assert kinds["video"]["codec_name"] == "h264"
        assert kinds["video"]["pix_fmt"] == "yuv420p"
        assert kinds["video"]["r_frame_rate"] == "12/1"
        assert kinds["audio"]["codec_name"] == "aac"
        assert float(info["format"]["duration"]) == pytest.approx(
            report["duration"], abs=0.15
        )
    with pytest.raises(EpisodeError, match="not empty"):
        ep.render_episode(spec, out, preview=True)
    again = ep.render_episode(spec, out, preview=True, stills=(1.0,), overwrite=True)
    assert [Path(p).name for p in again["stills"]] == ["t_0001000ms.png"]


def test_render_episode_from_json_path_without_audio(media, tmp_path):
    path = tmp_path / "episode.json"
    _mini(media).to_json(path)
    result = ep.render_episode(path, tmp_path / "o", preview=True, stills=())
    assert result["stills"] == []
    assert Path(result["video"]).stat().st_size > 0


def test_render_episode_master(media, tmp_path):
    spec = _mini(media, audio=_audio(media, narration_s=2.0))
    result = ep.render_episode(
        spec, tmp_path / "o", preview=True, master=True, stills=()
    )
    master = Path(result["master"])
    assert (master / "master.mkv").is_file() and (master / "manifest.json").is_file()
    with wave.open(str(tmp_path / "o" / "master_audio.wav")) as handle:
        assert handle.getnframes() == 96 * 48000 // 12


def test_render_episode_closes_clips_on_failure(media, tmp_path, monkeypatch):
    closed = []
    original = ep.build_episode

    def spy(spec):
        comp = original(spec)
        for clip in comp.episode_clips:
            clip.close = lambda c=clip: closed.append(c)

        def boom(*args, **kwargs):
            raise RuntimeError("x")

        monkeypatch.setattr("moviepy.ae.parallel.write_video_parallel", boom)
        return comp

    monkeypatch.setattr(ep, "build_episode", spy)
    spec = _mini(media, audio=_audio(media))
    with pytest.raises(RuntimeError):
        ep.render_episode(spec, tmp_path / "o", preview=True, stills=())
    assert closed  # the narration clip was closed


@pytest.mark.parametrize("channel", ["story", "history"])
def test_init_episode_tree(tmp_path, channel):
    target = tmp_path / "new" / "ep"
    config = ep.init_episode(target, channel)
    assert Path(config) == target / "episode.json"
    assert (target / "media").is_dir() and (target / "subtitles").is_dir()
    readme = (target / "README.txt").read_text(encoding="utf-8")
    assert "validate" in readme and "media/" in readme
    spec = EpisodeSpec.from_json(config)
    assert spec.preset == f"nightlamp_{channel}"
    assert spec.audio.narration == str(target / "media" / "PLACEHOLDER_narration.wav")
    assert spec.audio.music is None
    with pytest.raises(EpisodeError, match="not empty"):
        ep.init_episode(target, channel)
    empty = tmp_path / "empty"
    empty.mkdir()
    ep.init_episode(empty, channel)  # an empty directory is acceptable
    with pytest.raises(EpisodeError, match="channel"):
        ep.init_episode(tmp_path / "x", "news")


def test_cli_init_validate_render(media, tmp_path, capsys):
    path = tmp_path / "episode.json"
    _mini(media, audio=_audio(media)).to_json(path)
    assert ep.main(["validate", str(path)]) == 0
    assert capsys.readouterr().out.startswith("OK")
    assert ep.main(["validate", str(tmp_path / "missing.json")]) == 1
    assert "ERROR" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text('{"shots": []}', encoding="utf-8")
    assert ep.main(["validate", str(bad)]) == 1
    capsys.readouterr()
    assert ep.main(["init", str(tmp_path / "p"), "--channel", "story"]) == 0
    assert ep.main(["init", str(tmp_path / "p"), "--channel", "story"]) == 1
    # placeholders do not exist yet, so validating a fresh project must fail
    assert ep.main(["validate", str(tmp_path / "p" / "episode.json")]) == 1
    capsys.readouterr()
    out = tmp_path / "r"
    code = ep.main(["render", str(path), str(out), "--preview", "--still", "0.5", "3"])
    assert code == 0
    assert sorted(p.name for p in (out / "stills").iterdir()) == [
        "t_0000500ms.png",
        "t_0003000ms.png",
    ]
    assert ep.main(["render", str(path), str(out), "--preview"]) == 1


def test_cli_module_entry_point(media, tmp_path):
    path = tmp_path / "episode.json"
    _mini(media).to_json(path)
    repo = Path(ep.__file__).parents[3]
    base = [sys.executable, "-m", "moviepy.ae.templates", "validate"]
    run = subprocess.run([*base, str(path)], cwd=repo, capture_output=True, text=True)
    assert run.returncode == 0 and run.stdout.startswith("OK")
    run = subprocess.run([*base, "nope.json"], cwd=repo, capture_output=True, text=True)
    assert run.returncode == 1 and "ERROR" in run.stderr


# --------------------------------------------------------------------------- #
# Scene overlays, name tags and subtitle reflow
# --------------------------------------------------------------------------- #


def _overlay_data(media):
    data = _data(media)
    data["scene_overlay"] = {
        "chapters": [[5.0, "Part One"]],
        "logo": "Lamp",
        "watermark": "@lamp",
        "cta_text": "Subscribe",
        "layout": {"duration": 2.0, "fade": 0.4},
    }
    data["name_tags"] = [
        {
            "name": "Kong",
            "role": "Scholar",
            "subject_box": [1100, 150, 500, 600],
            "start": 7.0,
            "duration": 2.0,
            "orientation": "horizontal",
        }
    ]
    return data


def test_scene_overlay_and_name_tags_build(media):
    spec = EpisodeSpec.from_dict(_overlay_data(media))
    again = EpisodeSpec.from_dict(json.loads(spec.to_json()))
    assert again.to_dict() == spec.to_dict()
    plain = build_episode(_spec(media))
    comp = build_episode(spec)
    added = [layer.name for layer in comp.layers][
        1 : len(comp.layers) - len(plain.layers) + 1
    ]
    assert comp.layers[0].name == "Subtitles" and added
    report = episode_report(comp)
    assert report["scene_overlay"] == [{"start": 5.0, "title": "Part One"}]
    assert report["name_tags"] == [{"name": "Kong", "start": 7.0, "end": 9.0}]
    # Overlays draw only inside their windows.
    assert np.array_equal(plain.get_frame(4.5), comp.get_frame(4.5))
    assert not np.array_equal(plain.get_frame(6.0), comp.get_frame(6.0))
    assert not np.array_equal(plain.get_frame(8.0), comp.get_frame(8.0))


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"scene_overlay": {"chapters": []}}, "chapters"),
        ({"scene_overlay": {"chapters": [[1, "x"]], "layout": {"fade": 9}}}, "layout"),
        ({"name_tags": [{"name": ""}]}, "name_tags"),
        ({"subtitles": {"primary": [], "words": []}}, "overflow"),
    ],
)
def test_overlay_and_reflow_validation(media, patch, message):
    data = {**_data(media), **patch}
    with pytest.raises(EpisodeError, match=message):
        EpisodeSpec.from_dict(data)


def test_overlays_must_avoid_bookend_cards(media):
    data = _overlay_data(media)
    data["scene_overlay"]["chapters"] = [[1.0, "Too early"]]
    with pytest.raises(EpisodeError, match="scene_overlay"):
        build_episode(EpisodeSpec.from_dict(data))


def test_subtitle_overflow_split(media):
    data = _data(media)
    text = " ".join(["slowly"] * 30)
    data["subtitles"] = {
        "primary": [{"start": 4, "end": 9, "text": text, "lang": "en"}],
        "overflow": "split",
    }
    with pytest.raises(EpisodeError, match="subtitles"):
        build_episode(
            EpisodeSpec.from_dict(
                {**data, "subtitles": {**data["subtitles"], "overflow": "wrap"}}
            )
        )
    report = episode_report(build_episode(EpisodeSpec.from_dict(data)))
    lane = report["subtitles"]["primary"]
    assert lane["cues"] == 1 and lane["rendered_cues"] > 1
