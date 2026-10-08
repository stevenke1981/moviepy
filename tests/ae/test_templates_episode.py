"""Tests for the declarative episode build (``moviepy.ae.templates.episode``)."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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
    assert report["subtitles"]["primary"] == {"cues": 1, "first": 5.0, "last": 7.0}
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
