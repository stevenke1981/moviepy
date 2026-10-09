"""Verify the episode background and the intro/outro title card templates."""

import hashlib
import os

import numpy as np
from PIL import Image

import pytest

from moviepy import VideoClip
from moviepy.ae import Composition
from moviepy.ae.properties.keyframe import Keyframe
from moviepy.ae.templates.background import (
    contrast_report,
    cover_crop,
    media_background,
)
from moviepy.ae.templates.presets import get_preset
from moviepy.ae.templates.title_card import (
    NO_LINE_END,
    NO_LINE_START,
    TitleCardSpec,
    build_bookends,
    build_title_card,
    fit_lines,
    readable_duration,
)


KAIU = "C:/Windows/Fonts/kaiu.ttf"
LATIN = {"title": None, "body": None}


def small_preset(size=(64, 36), **changes):
    return get_preset("nightlamp_story", size=size, safe_margin=(4, 4), **changes)


def compose(layers, preset, duration=1.0):
    comp = Composition(preset.size, preset.fps, duration)
    for layer in layers:
        comp.add_layer(layer)
    return comp


def alpha_bbox(comp, t):
    alpha = comp.mask.get_frame(t)
    ys, xs = np.nonzero(alpha > 0.02)
    if not xs.size:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


# --------------------------------------------------------------------------- #
# cover crop and background
# --------------------------------------------------------------------------- #


def test_cover_crop_math():
    assert cover_crop((200, 100), (100, 100)) == (50.0, 0.0, 150.0, 100.0)
    assert cover_crop((100, 200), (100, 100)) == (0.0, 50.0, 100.0, 150.0)
    # Same aspect, any scale: the whole source.
    assert cover_crop((400, 200), (192, 108))[0] == pytest.approx(
        (400 - 200 * 192 / 108) / 2
    )
    box = cover_crop((400, 400), (1920, 1080))
    assert box[2] - box[0] == pytest.approx(400)
    assert (box[3] - box[1]) / (box[2] - box[0]) == pytest.approx(1080 / 1920)


def test_cover_crop_focal_clamps_inside_source():
    assert cover_crop((200, 100), (100, 100), (0.0, 0.5)) == (0.0, 0.0, 100.0, 100.0)
    assert cover_crop((200, 100), (100, 100), (1.0, 0.5)) == (
        100.0,
        0.0,
        200.0,
        100.0,
    )
    left, top, right, bottom = cover_crop((300, 500), (160, 90), (0.2, 0.99))
    assert 0 <= left < right <= 300 and 0 <= top < bottom <= 500
    with pytest.raises(ValueError):
        cover_crop((100, 100), (50, 50), (1.5, 0.5))


def test_overlay_opacity_is_exact_and_live():
    preset = small_preset()
    white = np.full((36, 64, 3), 255, np.uint8)
    background = media_background(white, preset)
    comp = compose(background.layers, preset)
    assert comp.get_frame(0.0)[5, 5].tolist() == [128, 128, 128]
    assert comp.get_frame(0.0)[0, 0].tolist() == [128, 128, 128]
    assert comp.get_frame(0.0)[-1, -1].tolist() == [128, 128, 128]
    assert background.overlay.transform.opacity.value_at(0.0) == 50.0
    background.overlay.transform.opacity = 100.0
    assert comp.get_frame(0.0)[5, 5].tolist() == [0, 0, 0]
    custom = media_background(
        white, preset, overlay_opacity=0.25, overlay_color="#FF0000"
    )
    pixel = compose(custom.layers, preset).get_frame(0.0)[5, 5]
    assert pixel.tolist() == [round(255 * 0.75 + 255 * 0.25)] + [round(255 * 0.75)] * 2


def test_cover_scaling_fills_the_canvas_edges():
    preset = small_preset()
    frame = np.zeros((40, 90, 3), np.uint8)
    frame[:, :45] = (200, 40, 40)
    frame[:, 45:] = (40, 40, 200)
    background = media_background(frame, preset, overlay_opacity=0.0)
    out = compose(background.layers, preset).get_frame(0.0)
    assert out[0, 0].tolist() == [200, 40, 40]
    assert out[-1, -1].tolist() == [40, 40, 200]
    assert out[18, 0].tolist() == [200, 40, 40]
    assert out[18, 63].tolist() == [40, 40, 200]
    with pytest.raises(ValueError):
        media_background(frame, preset, overlay_opacity=1.5)


def test_video_frame_time_is_held_static():
    def paint(t):
        value = int(round(t * 100))
        return np.full((20, 32, 3), value, np.uint8)

    clip = VideoClip(paint, duration=2.0)
    clip.fps = 10
    preset = small_preset()
    background = media_background(clip, preset, frame_time=1.0, overlay_opacity=0.0)
    comp = compose(background.layers, preset, duration=3.0)
    first, later = comp.get_frame(0.0), comp.get_frame(2.5)
    assert np.array_equal(first, later)
    assert first[10, 10, 0] == 100
    assert background.provenance["frame_time"] == pytest.approx(1.0)
    assert background.provenance["media_type"] == "video"
    snapped = media_background(clip, preset, frame_time=1.04)
    assert snapped.provenance["frame_time"] == pytest.approx(1.0)
    assert snapped.provenance["requested_frame_time"] == pytest.approx(1.04)
    with pytest.raises(ValueError):
        media_background(clip, preset, frame_time=2.0)


def test_missing_or_invalid_media_raises_never_substitutes(tmp_path):
    preset = small_preset()
    with pytest.raises(FileNotFoundError):
        media_background(tmp_path / "absent.png", preset)
    with pytest.raises(FileNotFoundError):
        media_background(str(tmp_path / "absent.mp4"), preset)
    with pytest.raises(TypeError):
        media_background(None, preset)
    with pytest.raises(ValueError):
        media_background(np.zeros((4, 4), np.uint8), preset)
    transparent = np.zeros((8, 8, 4), np.uint8)
    with pytest.raises(ValueError, match="opaque"):
        media_background(transparent, preset)
    path = tmp_path / "still.png"
    Image.new("RGB", (16, 9), (1, 2, 3)).save(path)
    with pytest.raises(ValueError, match="frame_time"):
        media_background(path, preset, frame_time=1.0)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        media_background(broken, preset)


def test_provenance_hash_and_crop_box(tmp_path):
    preset = small_preset()
    path = tmp_path / "episode.png"
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 255, (50, 120, 3), dtype=np.uint8)).save(path)
    background = media_background(path, preset, focal_point=(0.0, 0.5))
    assert (
        background.provenance["source_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert background.provenance["source_name"] == "episode.png"
    assert background.provenance["source_size"] == [120, 50]
    assert background.provenance["crop_box"] == cover_crop(
        (120, 50), preset.size, (0.0, 0.5)
    )
    assert background.provenance["crop_box"][0] == 0.0
    array_bg = media_background(np.zeros((9, 16, 3), np.uint8), preset)
    assert array_bg.provenance["source_sha256"] is None


def test_contrast_report_is_conservative():
    dark = np.zeros((20, 20, 3), np.uint8)
    report = contrast_report(dark, (0, 0, 20, 20), "#FFFFFF")
    assert report["minimum_estimated_contrast"] == pytest.approx(21.0)
    assert not report["needs_visual_review"]
    mixed = dark.copy()
    mixed[5:10, 5:10] = 255
    report = contrast_report(mixed, (0, 0, 20, 20), (255, 255, 255))
    assert report["minimum_estimated_contrast"] == pytest.approx(1.0)
    assert report["needs_visual_review"]
    # Darkening the bright patch with the 50 % overlay is applied before measuring.
    darkened = contrast_report(
        mixed, (5, 5, 10, 10), (255, 255, 255), overlay=((0, 0, 0), 0.5)
    )
    assert darkened["minimum_estimated_contrast"] > 3.0
    with pytest.raises(ValueError):
        contrast_report(dark, (30, 30, 40, 40), (255, 255, 255))


# --------------------------------------------------------------------------- #
# text fitting
# --------------------------------------------------------------------------- #


def test_fit_lines_shrinks_before_wrapping():
    lines, size = fit_lines("Night Lamp", None, 10_000, 40, 20)
    assert (lines, size) == (["Night Lamp"], 40)
    text = "Night Lamp Story"
    for width in range(60, 260, 20):
        try:
            lines, size = fit_lines(text, None, width, 40, 20, 2)
        except ValueError:
            continue
        if len(lines) > 1:
            assert size == 20
        else:
            assert size >= 20
    # Single line at an intermediate size when the minimum size is unnecessary.
    wide = fit_lines(text, None, 150, 60, 10)
    assert len(wide[0]) == 1 and 10 <= wide[1] < 60


def test_fit_lines_latin_punctuation_and_protected_terms():
    text = "alpha beta, gamma (delta) epsilon zeta"
    for width in range(40, 200, 7):
        try:
            lines, _ = fit_lines(text, None, width, 20, 20, 1, max_lines=4)
        except ValueError:
            continue
        assert " ".join(lines) == text
        for line in lines[1:]:
            assert line[0] not in NO_LINE_START and line[0] not in ",.;:!?"
        for line in lines[:-1]:
            assert line[-1] not in NO_LINE_END
    for width in range(60, 200, 5):
        try:
            lines, _ = fit_lines(
                "one two three four",
                None,
                width,
                20,
                20,
                0,
                keep_together=("two three",),
                max_lines=3,
            )
        except ValueError:
            continue
        assert any("two three" in line for line in lines)


def test_fit_lines_overflow_raises_without_truncating():
    with pytest.raises(ValueError, match="overflow"):
        fit_lines("Supercalifragilisticexpialidocious", None, 30, 20, 12)
    with pytest.raises(ValueError):
        fit_lines("a\nb\nc", None, 500, 20, 12)
    assert fit_lines("a\nb", None, 500, 20, 12)[0] == ["a", "b"]
    assert fit_lines("", None, 10, 20, 12) == ([], 20)


@pytest.mark.skipif(not os.path.isfile(KAIU), reason="kaiu.ttf not installed")
def test_fit_lines_cjk_punctuation_rules():
    text = "天地玄黃，宇宙洪荒。日月盈昃（辰宿）列張"
    seen_wrapped = 0
    for width in range(150, 560, 11):
        try:
            lines, size = fit_lines(text, KAIU, width, 60, 40, 4)
        except ValueError:
            continue
        assert "".join(lines) == text
        if len(lines) > 1:
            seen_wrapped += 1
            assert size == 40
        for line in lines[1:]:
            assert line[0] not in NO_LINE_START
        for line in lines[:-1]:
            assert line[-1] not in NO_LINE_END
    assert seen_wrapped
    term = "宇宙洪荒"
    for width in range(150, 560, 11):
        try:
            lines, _ = fit_lines(text, KAIU, width, 60, 40, 4, keep_together=(term,))
        except ValueError:
            continue
        assert any(term in line for line in lines)


def test_readable_duration_extends_instead_of_truncating():
    duration, report = readable_duration(1.0, "x" * 60, 1.7, 0.35, 24)
    assert duration == pytest.approx(
        round((1.7 + 10 + 0.35) * 24 + 0.5) / 24, abs=1 / 24
    )
    assert duration >= 1.7 + 10 + 0.35
    assert report["automatically_extended"] and report["reading_characters"] == 60
    assert round(duration * 24) == pytest.approx(duration * 24)
    short, report = readable_duration(5.0, "short", 1.7, 0.35, 24)
    assert short == 5.0 and not report["automatically_extended"]
    with pytest.raises(ValueError, match="insufficient"):
        readable_duration(1.0, "x" * 60, 1.7, 0.35, 24, auto_extend=False)
    with pytest.raises(ValueError):
        TitleCardSpec(title="")
    with pytest.raises(ValueError):
        TitleCardSpec(title="a", layout="diagonal")


# --------------------------------------------------------------------------- #
# title composition
# --------------------------------------------------------------------------- #

SPEC = TitleCardSpec(
    title="Night Lamp Story",
    brand="NIGHT LAMP",
    subtitle="A story about a lamp and a book.",
)


@pytest.mark.parametrize("layout", ["right_column", "center"])
def test_title_card_renders_inside_safe_area(layout):
    preset = get_preset("nightlamp_story")
    spec = TitleCardSpec(
        title=SPEC.title, brand=SPEC.brand, subtitle=SPEC.subtitle, layout=layout
    )
    comp = build_title_card(spec, preset, fonts=LATIN, transparent=True)
    assert comp.size == (1920, 1080)
    assert [layer.name for layer in comp.layers] == [
        "Subtitle",
        "Title",
        "Rule",
        "Brand",
    ]
    start = comp.get_frame(0.0)
    assert start.shape == (1080, 1920, 3)
    assert alpha_bbox(comp, 0.0) is None  # nothing has entered at t=0
    middle = comp.duration / 2
    assert comp.get_frame(middle).shape == (1080, 1920, 3)
    box = alpha_bbox(comp, middle)
    assert box is not None
    mx, my = preset.safe_margin
    assert box[0] >= mx - 1 and box[2] <= 1920 - mx + 1
    assert box[1] >= my - 1 and box[3] <= 1080 - my + 1
    if layout == "right_column":
        assert box[0] >= 770 - 2
    else:
        assert abs((box[0] + box[2]) / 2 - 960) < 80
    end = comp.mask.get_frame(comp.duration - 1e-3).max()
    assert end < 0.05  # faded out
    assert comp.title_report["reading"]["actual_seconds"] == comp.duration


def test_entrance_is_staggered_and_editable():
    preset = get_preset("nightlamp_story")
    comp = build_title_card(SPEC, preset, fonts=LATIN, transparent=True)
    brand, title = comp.layer("Brand"), comp.layer("Title")
    subtitle = comp.layer("Subtitle")
    opacity = [layer.transform.opacity.value_at for layer in (brand, title, subtitle)]
    assert all(f(0.0) == 0.0 for f in opacity)
    assert 0 < opacity[0](0.5) < 100 and opacity[1](0.5) == 0.0
    assert opacity[0](1.0) == 100.0 and opacity[2](0.9) == 0.0
    assert all(f(2.5) == 100.0 for f in opacity)
    assert all(f(comp.duration) == 0.0 for f in opacity)
    # Slide: starts below the resting position and settles on it.
    p0 = title.transform.position.value_at(0.0)
    p1 = title.transform.position.value_at(2.5)
    assert p0[1] - p1[1] == pytest.approx(18.0) and p0[0] == p1[0]
    # Editable: keyframes are plain Properties.
    assert len(title.transform.opacity.keyframes) >= 4


def test_determinism():
    preset = get_preset("nightlamp_story")
    frame = np.random.default_rng(5).integers(0, 255, (60, 100, 3), dtype=np.uint8)
    background = media_background(frame, preset)
    one = build_title_card(SPEC, preset, background, fonts=LATIN)
    two = build_title_card(SPEC, preset, background, fonts=LATIN)
    for t in (0.0, 1.0, 2.5):
        assert np.array_equal(one.get_frame(t), two.get_frame(t))


def test_title_card_over_background_and_bookends_share_overlay():
    preset = get_preset("nightlamp_story")
    frame = np.full((90, 160, 3), 250, np.uint8)
    background = media_background(frame, preset)
    intro, outro = build_bookends(
        SPEC,
        TitleCardSpec(title="See you soon", subtitle="Subscribe"),
        preset,
        background,
        fonts=LATIN,
    )
    assert intro.title_report["role"] == "intro"
    assert outro.title_report["role"] == "outro"
    assert outro.title_report["title_size"] <= 72
    assert intro.title_report["title_size"] > outro.title_report["title_size"]
    # The 50 % black overlay darkens white-ish media to ~125.
    assert intro.get_frame(0.0)[10, 10, 0] == 125
    assert outro.get_frame(0.0)[10, 10, 0] == 125
    # Keyframing the shared Property (not replacing it) drives both comps.
    background.overlay.transform.opacity.set_keyframes([Keyframe(0.0, 0.0)])
    assert intro.get_frame(0.0)[10, 10, 0] == 250
    assert outro.get_frame(0.0)[10, 10, 0] == 250
    assert intro.layers[-1].name == "Episode media"


def test_scales_to_1280x720():
    preset = get_preset("nightlamp_story", size=(1280, 720), safe_margin=(85.33, 60.0))
    comp = build_title_card(SPEC, preset, fonts=LATIN, transparent=True)
    assert comp.size == (1280, 720)
    frame = comp.get_frame(comp.duration / 2)
    assert frame.shape == (720, 1280, 3)
    box = alpha_bbox(comp, comp.duration / 2)
    mx, my = preset.safe_margin
    assert box[0] >= mx - 1 and box[2] <= 1280 - mx + 1
    assert box[1] >= my - 1 and box[3] <= 720 - my + 1
    full = build_title_card(
        SPEC, get_preset("nightlamp_story"), fonts=LATIN, transparent=True
    )
    ratio = comp.title_report["title_size"] / full.title_report["title_size"]
    assert ratio == pytest.approx(2 / 3, abs=0.03)


def test_long_hook_extends_duration_and_missing_font_raises(tmp_path):
    preset = get_preset("nightlamp_story")
    spec = TitleCardSpec(
        title="Night Lamp", subtitle="a long hook that needs time to read " * 3
    )
    comp = build_title_card(spec, preset, fonts=LATIN)
    assert comp.duration > spec.duration
    assert comp.title_report["reading"]["automatically_extended"]
    with pytest.raises(FileNotFoundError):
        build_title_card(
            SPEC, preset, fonts={"title": str(tmp_path / "missing.ttf"), "body": None}
        )
    with pytest.raises(ValueError, match="glyph|font"):
        build_title_card(TitleCardSpec(title="夜燈故事"), preset, fonts=LATIN)


@pytest.mark.skipif(
    not all(os.path.isfile(p) for p in get_preset("nightlamp_story").fonts.values()),
    reason="preset Source Han fonts not installed",
)
def test_cjk_card_with_preset_font():
    preset = get_preset("nightlamp_story")
    spec = TitleCardSpec(
        title="夜燈故事：第一集",
        brand="夜燈說書",
        subtitle="今晚的故事，關於一盞燈與一本書。",
        title_keep_together=("夜燈故事",),
    )
    comp = build_title_card(spec, preset, transparent=True)
    box = alpha_bbox(comp, comp.duration / 2)
    mx, my = preset.safe_margin
    assert box[0] >= mx and box[2] <= 1920 - mx
    assert box[1] >= my and box[3] <= 1080 - my
    for line in comp.title_report["title_lines"]:
        assert line[0] not in NO_LINE_START and line[-1] not in NO_LINE_END


# --------------------------------------------------------------------------- #
# review regressions
# --------------------------------------------------------------------------- #


def test_blank_lines_and_whitespace_copy_do_not_add_layers_or_height():
    preset = small_preset((320, 180))
    clean = build_title_card(TitleCardSpec(title="Night"), preset, fonts=LATIN)
    padded = build_title_card(
        TitleCardSpec(title="\nNight\n ", brand="  ", subtitle=" \n "),
        preset,
        fonts=LATIN,
    )
    assert padded.title_report["title_lines"] == ["Night"]
    assert padded.title_report["subtitle_lines"] == []
    assert [layer.name for layer in padded.layers] == [
        layer.name for layer in clean.layers
    ]
    assert padded.title_report["ink_boxes"] == clean.title_report["ink_boxes"]


def test_fit_lines_drops_blank_explicit_lines():
    assert fit_lines("a\n\n b \n", None, 1000, 20, 10) == (["a", "b"], 20)
    assert fit_lines("\n  \n", None, 1000, 20, 10) == ([], 20)


def test_entrance_ease_survives_a_zero_start_time():
    preset = small_preset((320, 180))
    spec = TitleCardSpec(title="Hi", title_start=0.0, title_duration=1.0)
    title = next(
        layer
        for layer in build_title_card(spec, preset, fonts=LATIN).layers
        if layer.name == "Title"
    )
    # Ease-out cubic is well ahead of linear a quarter of the way in.
    assert title.transform.opacity.value_at(0.25) > 60.0


def test_non_finite_float_frames_are_rejected():
    preset = small_preset()
    frame = np.full((36, 64, 3), 100.0)
    frame[3, 3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        media_background(frame, preset)


@pytest.mark.parametrize("size", [(1081, 607), (641, 361)])
def test_odd_canvas_with_background_stays_inside_scaled_margin(size):
    base = get_preset("nightlamp_story")
    preset = base.with_overrides(size=size)
    assert preset.safe_margin[1] == pytest.approx(base.safe_margin[1] * size[1] / 1080)
    frame = np.random.default_rng(1).integers(0, 255, (50, 120, 3), dtype=np.uint8)
    background = media_background(frame, preset)
    spec = TitleCardSpec(title="Night Lamp", brand="LAMP", subtitle="A quiet hook.")
    comp = build_title_card(spec, preset, background, fonts=LATIN)
    mx, my = preset.safe_margin
    for box in comp.title_report["ink_boxes"].values():
        assert box[0] >= mx - 1 and box[2] <= size[0] - mx + 1
        assert box[1] >= my - 1 and box[3] <= size[1] - my + 1
    out = comp.get_frame(2.0)
    assert out.shape == (size[1], size[0], 3)


def test_baked_cover_matches_transform_cover():
    preset = small_preset((96, 54))
    rng = np.random.default_rng(5)
    base = rng.integers(0, 255, (9, 16, 3), dtype=np.uint8)
    frame = np.kron(base, np.ones((20, 20, 1), np.uint8))  # 320x180 smooth blocks
    baked = media_background(frame, preset, overlay_opacity=0.0)
    live = media_background(frame, preset, overlay_opacity=0.0, bake=False)
    assert baked.footage.source_size == preset.size
    assert live.footage.source_size == (320, 180)
    a = compose(baked.layers, preset).get_frame(0.0).astype(int)
    b = compose(live.layers, preset).get_frame(0.0).astype(int)
    assert np.abs(a - b).mean() < 6.0
    # Same provenance either way; the bake is an implementation detail.
    assert baked.provenance["crop_box"] == live.provenance["crop_box"]
    assert baked.provenance["baked"] is True and live.provenance["baked"] is False


def test_overlay_column_stays_above_the_subtitle_zone():
    preset = get_preset("nightlamp_story")
    spec = TitleCardSpec(
        title=SPEC.title, brand=SPEC.brand, subtitle=SPEC.subtitle, layout="left_column"
    )
    overlay = build_title_card(spec, preset, fonts=LATIN, transparent=True)
    card = build_title_card(spec, preset, fonts=LATIN)
    bottom = max(box[3] for box in overlay.title_report["ink_boxes"].values())
    assert bottom <= 1080 * 0.6
    # An opaque card keeps the column centred in the whole safe area.
    card_bottom = max(box[3] for box in card.title_report["ink_boxes"].values())
    assert card_bottom > bottom
