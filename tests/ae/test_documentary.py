"""Native-renderer regressions for the two documentary photo references."""

import json
import subprocess
import sys

import numpy as np
from PIL import Image

import pytest

from moviepy.ae import Composition
from moviepy.ae.layers.av import AVLayer
from moviepy.ae.templates.documentary import (
    CollageCard,
    add_documentary_subtitles,
    build_film_frame,
    build_news_collage,
    build_news_page,
)
from moviepy.ae.templates.subtitles import Cue
from moviepy.audio.AudioClip import AudioClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ColorClip, ImageClip, VideoClip


SIZE = (320, 180)
RED = (213, 44, 62)


def footage(duration=1):
    return ColorClip((200, 140), color=RED, duration=duration).with_fps(12)


@pytest.mark.parametrize("size", [SIZE, (480, 270), (360, 240)])
@pytest.mark.parametrize("kind", ["film", "news", "page"])
def test_native_builders_and_output(size, kind):
    clip = footage()
    if kind == "film":
        comp = build_film_frame(clip, size=size, push=0, pan=(0, 0))
    elif kind == "news":
        comp = build_news_collage(clip, [clip] * 4, size=size)
    else:
        comp = build_news_page(clip, headline="ARCHIVE", size=size)
    assert isinstance(comp, Composition)
    frame = comp.get_frame(0.7)
    assert frame.shape == (size[1], size[0], 3)
    assert frame.dtype == np.uint8
    assert all(isinstance(layer, AVLayer) for layer in comp.layers)


def test_film_opposite_bevels_and_fixed_shell():
    comp = build_film_frame(footage(), size=(640, 360), push=0, pan=(0, 0))
    x, y, w, h = comp.documentary_metadata["window_px"]
    frame = comp.get_frame(0.5)
    assert np.array_equal(frame[y + h // 2, x + w // 2], RED)
    assert not np.array_equal(frame[y + 2, x + 2], RED)
    assert not np.array_equal(frame[y + h - 3, x + w - 3], RED)
    # The other two corners are rounded, not bevelled by the same distance.
    assert np.array_equal(frame[y + 5, x + w - 6], RED)
    assert np.array_equal(frame[y + h - 6, x + 5], RED)
    assert len(comp.layer("film.viewport").masks) == 6
    assert np.array_equal(frame[:y], comp.get_frame(0.8)[:y])


def test_native_masks_remain_editable():
    comp = build_film_frame(footage(), size=SIZE, push=0, pan=(0, 0))
    x, y, _, _ = comp.documentary_metadata["window_px"]
    before = comp.get_frame(0.4)
    comp.layer("film.viewport").masks.clear()
    after = comp.get_frame(0.4)
    assert not np.array_equal(before[y, x], RED)
    assert np.array_equal(after[y, x], RED)


def test_cover_preserves_shape_aspect_ratio():
    image = np.zeros((200, 400, 3), np.uint8)
    yy, xx = np.ogrid[:200, :400]
    image[(xx - 200) ** 2 + (yy - 100) ** 2 <= 28**2] = 255
    comp = build_film_frame(
        ImageClip(image),
        size=(480, 360),
        duration=1,
        push=0,
        pan=(0, 0),
        hud=False,
    )
    frame = comp.get_frame(0)
    x, y, w, h = comp.documentary_metadata["window_px"]
    mask = frame[y : y + h, x : x + w, 0] > 200
    rows, cols = np.where(mask)
    assert abs((rows.max() - rows.min()) - (cols.max() - cols.min())) <= 2


def test_contain_keeps_whole_source_with_letterbox():
    comp = build_film_frame(
        ColorClip((400, 100), color=RED),
        size=(480, 360),
        duration=1,
        fit="contain",
        push=0,
        pan=(0, 0),
        notches=False,
    )
    x, y, w, h = comp.documentary_metadata["window_px"]
    frame = comp.get_frame(0.3)
    assert np.array_equal(frame[y + h // 2, x + w // 2], RED)
    assert np.array_equal(frame[y + 10, x + w // 2], (0, 0, 0))


def test_news_transparent_source_does_not_get_a_black_box():
    source = ImageClip(np.zeros((80, 100, 4), np.uint8))
    comp = build_news_collage(
        source, size=SIZE, duration=1, entrance=0, push=0, pan=(0, 0)
    )
    frame = comp.get_frame(0.4)
    background = comp.layer("news.background").clip.get_frame(0)
    assert np.array_equal(frame, background)


def test_news_order_and_editable_properties():
    main = footage()
    comp = build_news_collage(main, [main] * 4, size=SIZE, entrance=0)
    assert comp.layers[0].name == "news.main"
    assert comp.layers[-1].name == "news.background"
    layer = comp.layer("news.main")
    before = comp.get_frame(0.4)
    layer.transform.position = (-1000, -1000)
    assert not np.array_equal(before, comp.get_frame(0.4))
    assert layer.effects[0].name == "Drop Shadow"


def test_custom_cards_are_native_and_duplicates_rejected():
    cards = [
        CollageCard(footage(), center=(i / 8, 0.2), name=f"card.{i}") for i in range(6)
    ]
    comp = build_news_collage(footage(), cards, size=SIZE)
    assert len(comp.layers) == 8
    with pytest.raises(ValueError, match="unique"):
        build_news_collage(footage(), [cards[0], cards[0]], size=SIZE)
    with pytest.raises(ValueError, match="four"):
        build_news_collage(footage(), [footage()] * 5, size=SIZE)


def _tone(level, duration=1):
    return AudioClip(lambda t: np.zeros_like(t) + level, duration=duration, fps=8000)


@pytest.mark.parametrize("kind", ["news", "film", "page"])
def test_primary_audio_is_preserved_without_background_duplicates(kind):
    source = footage().with_audio(_tone(0.15))
    other = footage().with_audio(_tone(0.70))
    if kind == "news":
        comp = build_news_collage(source, [other] * 4, size=SIZE)
    elif kind == "page":
        comp = build_news_page(source, headline="A STORY", size=(400, 280))
    else:
        comp = build_film_frame(source, size=SIZE)
    assert comp.audio is not None
    assert np.allclose(comp.audio.get_frame(np.array([0.1, 0.5])), 0.15)
    assert source.audio is not None
    assert source.duration == 1


def test_short_source_holds_video_not_audio_and_never_reads_endpoint():
    times = []

    def frame(t):
        assert 0 <= t < 0.25
        times.append(t)
        return np.full((100, 160, 3), round(t * 200), np.uint8)

    source = VideoClip(frame, duration=0.25).with_fps(12).with_audio(_tone(0.2, 0.25))
    comp = build_film_frame(source, size=SIZE, duration=1, push=0, pan=(0, 0))
    a, b = comp.get_frame(0.5), comp.get_frame(0.9)
    assert np.array_equal(a, b)
    assert max(times) < 0.25
    assert np.allclose(comp.audio.get_frame(0.8), 0)


def test_embedded_page_video_keeps_playing_and_input_is_unchanged():
    clip = VideoClip(
        lambda t: np.full((100, 160, 3), round(20 + 200 * t), np.uint8), duration=1
    )
    page = build_news_page(clip, headline="A STORY", size=(480, 340))
    assert not np.array_equal(page.get_frame(0.1), page.get_frame(0.8))
    assert clip.mask is None
    assert clip.size == (160, 100)
    assert clip.duration == 1
    assert np.all(clip.get_frame(0.5) == 120)


@pytest.mark.parametrize("kind", ["film", "news"])
def test_random_access_deterministic_and_seeded(kind):
    build = build_film_frame if kind == "film" else build_news_collage
    comp = build(footage(), size=SIZE)
    first = comp.get_frame(0.6)
    comp.get_frame(0.1)
    comp.get_frame(0.85)
    assert np.array_equal(first, comp.get_frame(0.6))
    if kind == "news":
        assert not np.array_equal(first, comp.get_frame(0.1))
    assert np.array_equal(first, build(footage(), size=SIZE).get_frame(0.6))


@pytest.mark.parametrize("duration", [0.02, 0.15, 1])
def test_short_output_has_valid_entrance_keys(duration):
    comp = build_news_collage(footage(), size=SIZE, duration=duration)
    assert comp.get_frame(0).shape == (180, 320, 3)


@pytest.mark.parametrize(
    "options",
    [
        {"fps": 0},
        {"fps": True},
        {"duration": float("nan")},
        {"push": -1},
        {"push": float("inf")},
        {"pan": (0, float("nan"))},
        {"size": (320.5, 180)},
        {"size": (True, 180)},
        {"window": (0.9, 0, 0.5, 0.8)},
        {"window": (0, 0, 0, 1)},
        {"window": (0, 0)},
        {"hud": 1},
        {"fit": "stretch"},
        {"source_label": "line\nbreak"},
        {"notches": "yes"},
    ],
)
def test_invalid_film_options_fail_early(options):
    with pytest.raises((ValueError, TypeError)):
        build_film_frame(footage(), **options)


def test_page_text_is_not_silently_truncated():
    with pytest.raises(ValueError, match="headline"):
        build_news_page(footage(), headline="very long headline " * 50)
    with pytest.raises(ValueError, match="body"):
        build_news_page(footage(), headline="NEWS", body="story " * 500)
    with pytest.raises(FileNotFoundError, match="font"):
        build_news_page(footage(), headline="NEWS", font="missing-font.ttf")


def test_mask_input_is_not_mistaken_for_rgb():
    with pytest.raises(TypeError, match="RGB"):
        build_film_frame(ColorClip((100, 100), color=1.0, is_mask=True, duration=1))


def test_existing_subtitle_engine_is_used():
    from moviepy.ae.templates.documentary._art import font_path
    from moviepy.ae.templates.subtitles import SubtitleLayer

    font = font_path()
    if font is None:
        pytest.skip("no system font for native subtitle engine")
    comp = build_film_frame(footage(), size=SIZE, push=0, pan=(0, 0))
    add_documentary_subtitles(comp, [Cue(0.1, 0.5, "A letter", "en")], font=font)
    assert isinstance(comp.layers[0], SubtitleLayer)
    assert comp.layers[0].active_cues(0.5) == (None, None)


def test_cli_renders_real_mp4_and_utf8_paths(tmp_path):
    out = tmp_path / "輸出 documentary"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "moviepy.ae.templates.documentary",
            "--demo",
            "--output-dir",
            str(out),
            "--width",
            "320",
            "--duration",
            "0.5",
            "--fps",
            "4",
        ],
        check=True,
        timeout=90,
    )
    for name in ("news_collage", "film_frame"):
        assert (out / f"{name}.png").is_file()
        with VideoFileClip(str(out / f"{name}.mp4")) as clip:
            assert clip.size == [320, 180] or clip.size == (320, 180)
            assert clip.fps == 4
            assert abs(clip.duration - 0.5) < 0.05
            assert clip.get_frame(0.25).shape == (180, 320, 3)
    report = json.loads((out / "render-report.json").read_text(encoding="utf-8"))
    assert report["demo"] is True


def test_cli_config_paths_are_relative_and_main_screenshot_is_not_rewrapped(tmp_path):
    image = tmp_path / "新聞 圖.png"
    Image.new("RGB", (200, 130), RED).save(image)
    config = tmp_path / "project.json"
    config.write_text(
        json.dumps({"input": image.name, "main_is_page": True}), encoding="utf-8"
    )
    out = tmp_path / "render"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "moviepy.ae.templates.documentary",
            "--config",
            str(config),
            "--effect",
            "news",
            "--output-dir",
            str(out),
            "--width",
            "320",
            "--duration",
            "0.5",
            "--fps",
            "4",
        ],
        check=True,
        timeout=90,
    )
    assert (out / "news_collage.mp4").is_file()
    assert not (out / "film_frame.mp4").exists()


def test_shipped_json_cues_match_native_schema():
    from pathlib import Path

    from moviepy.ae.templates.documentary.__main__ import _load_config

    root = Path(__file__).resolve().parents[2]
    config = _load_config(root / "examples" / "ae_documentary.json")
    for name in ("subtitles_primary", "subtitles_secondary"):
        assert all(Cue(**item).duration > 0 for item in config[name])
