"""Tests for the music-channel thumbnail generator (no GPU, network or music)."""

import os
import subprocess

import numpy as np
from PIL import Image

import pytest

from moviepy.ae.templates import thumbnail as th
from moviepy.config import FFMPEG_BINARY


MSJH = th.DEFAULT_FONTS["zh"]
YUGOTH = th.DEFAULT_FONTS["ja"]
HAS_FONTS = os.path.isfile(MSJH) and os.path.isfile(YUGOTH)
needs_fonts = pytest.mark.skipif(
    not HAS_FONTS, reason="msjh.ttc / YuGothM.ttc not installed"
)
DARK = np.full((360, 640, 3), (18, 36, 52), np.uint8)
W, H = 1280, 720


def _spec(**overrides):
    values = {
        "titles": {"zh": "水晶的聲音", "en": "Crystal Rain", "ja": "水晶の音"},
        "subtitle": "睡眠・放鬆",
        "duration_badge": 5220,
    }
    values.update(overrides)
    return th.ThumbnailSpec(**values)


def _inside(box, safe, tol=1e-6):
    return (
        box[0] >= safe[0] - tol
        and box[1] >= safe[1] - tol
        and box[2] <= safe[2] + tol
        and box[3] <= safe[3] + tol
    )


def _make_mp4(path):
    subprocess.run(
        [
            FFMPEG_BINARY,
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x336699:s=64x48:d=1:r=10",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_format_duration_text_and_seconds():
    assert th.format_duration("87 min") == "87 min"
    assert th.format_duration(5220) == "1:27:00"
    assert th.format_duration(640) == "10:40"
    assert th.format_duration(38400.4) == "10:40:00"
    with pytest.raises(ValueError):
        th.format_duration("  ")
    with pytest.raises(TypeError):
        th.format_duration(True)


def test_spec_validation_rejects_bad_input():
    with pytest.raises(ValueError, match="zh"):
        th.ThumbnailSpec(titles={"en": "Rain"})
    with pytest.raises(ValueError):
        th.ThumbnailSpec(titles={"zh": "雨", "fr": "pluie"})
    with pytest.raises(ValueError):
        th.ThumbnailSpec(titles={"zh": "雨"}, layout="corner")
    with pytest.raises(ValueError):
        th.ThumbnailSpec(titles={"zh": "雨"}, accent="teal")
    with pytest.raises(ValueError):
        th.ThumbnailSpec(titles={"zh": "雨"}, panel_opacity=1.5)
    spec = th.ThumbnailSpec(titles={"zh": "雨"}, duration_badge=600)
    assert spec.duration_badge == "10:00"
    assert spec.fonts["ja"] == YUGOTH


def test_grab_frame_from_array_and_image(tmp_path):
    frame = th.grab_frame(DARK)
    assert frame.shape == (360, 640, 3) and frame.dtype == np.uint8
    rgba = np.dstack([DARK, np.full(DARK.shape[:2], 255, np.uint8)])
    assert th.grab_frame(rgba).shape == (360, 640, 3)
    path = tmp_path / "still.png"
    Image.fromarray(DARK).save(path)
    assert np.array_equal(th.grab_frame(path), DARK)
    with pytest.raises(ValueError, match="time applies only"):
        th.grab_frame(path, time=0.5)
    with pytest.raises(FileNotFoundError):
        th.grab_frame(tmp_path / "missing.png")


def test_grab_frame_from_synthetic_video(tmp_path):
    path = tmp_path / "clip.mp4"
    _make_mp4(path)
    frame = th.grab_frame(path)
    assert frame.shape == (48, 64, 3) and frame.dtype == np.uint8
    expected = np.array([0x33, 0x66, 0x99])
    assert np.all(np.abs(frame.reshape(-1, 3).mean(axis=0) - expected) <= 8)
    assert th.grab_frame(path, time=0.5).shape == (48, 64, 3)
    with pytest.raises(ValueError, match="within"):
        th.grab_frame(path, time=5.0)


def test_contact_sheet_layout_and_errors():
    sheet = th.contact_sheet([DARK, Image.fromarray(DARK)] * 2, columns=2)
    assert sheet.mode == "RGB" and sheet.size == (664, 384)
    with pytest.raises(ValueError):
        th.contact_sheet([])
    with pytest.raises(ValueError):
        th.contact_sheet([DARK], columns=0)


def test_main_contrast_passes_on_dark_and_fails_on_pale():
    dark_box = (100.0, 100.0, 400.0, 200.0)
    dark = th._main_contrast(DARK, dark_box, "#FFFFFF", require=True)
    assert dark["passes"] and dark["minimum_estimated_contrast"] >= 4.5
    pale = np.full((360, 640, 3), 240, np.uint8)
    report = th._main_contrast(pale, dark_box, "#F5F5F5", require=False)
    assert not report["passes"] and report["required"] is False
    with pytest.raises(th.ContrastError):
        th._main_contrast(pale, dark_box, "#F5F5F5", require=True)


def test_exclusive_create_and_argument_checks(tmp_path):
    target = tmp_path / "thumb.jpg"
    target.write_bytes(b"existing")
    spec = _spec(fonts={"zh": str(tmp_path / "missing.ttc")})
    with pytest.raises(FileExistsError):
        th.render_thumbnail(spec, DARK, target)
    assert target.read_bytes() == b"existing"
    with pytest.raises(ValueError, match="suffix|jpg"):
        th.render_thumbnail(spec, DARK, tmp_path / "thumb.gif")
    with pytest.raises(ValueError, match="max_bytes"):
        th.render_thumbnail(spec, DARK, max_bytes=0)
    with pytest.raises(ValueError, match="size"):
        th.render_thumbnail(spec, DARK, size=(100, 100))


def test_missing_font_raises(tmp_path):
    spec = _spec(fonts={"zh": str(tmp_path / "nope.ttc")})
    with pytest.raises(FileNotFoundError, match="zh"):
        th.render_thumbnail(spec, DARK)


@needs_fonts
def test_render_size_format_and_bytes(tmp_path):
    jpg = tmp_path / "t.jpg"
    image, evidence = th.render_thumbnail(_spec(), DARK, jpg)
    assert image.size == (W, H) and image.mode == "RGB"
    assert Image.open(jpg).format == "JPEG"
    assert evidence["bytes"] == os.path.getsize(jpg) <= 2_000_000
    assert evidence["human_visual"] == "NOT_RUN"
    png = tmp_path / "t.png"
    _, evidence = th.render_thumbnail(_spec(), DARK, png)
    assert Image.open(png).format == "PNG"
    assert evidence["bytes"] == os.path.getsize(png)


@needs_fonts
def test_render_jpeg_steps_quality_down_to_byte_limit(tmp_path):
    noisy = np.random.default_rng(7).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    out = tmp_path / "small.jpg"
    # Noise is a busy background, so the contrast gate is opted out here; the
    # test targets only the byte limit.
    _, evidence = th.render_thumbnail(
        _spec(), noisy, out, max_bytes=400_000, require_contrast=False
    )
    assert evidence["bytes"] <= 400_000
    assert evidence["quality"] is not None and evidence["quality"] < 92
    with pytest.raises(th.ThumbnailError):
        th.render_thumbnail(_spec(), noisy, max_bytes=500, require_contrast=False)


@needs_fonts
@pytest.mark.parametrize("layout", ["left_panel", "bottom_band", "center"])
def test_text_boxes_stay_in_safe_area(layout):
    _, evidence = th.render_thumbnail(_spec(layout=layout), DARK)
    safe = evidence["safe_area"]
    assert safe == pytest.approx([64, 36, 1216, 684])
    for block in evidence["text_boxes"]:
        assert _inside(block["box"], safe), block
    assert _inside(evidence["badge"]["plate"], safe)


@needs_fonts
def test_badge_is_top_right_not_bottom_right():
    _, evidence = th.render_thumbnail(_spec(), DARK)
    x0, y0, x1, y1 = evidence["badge"]["plate"]
    assert evidence["badge"]["corner"] == "top_right"
    assert not (x0 >= 0.75 * W and y1 >= 0.75 * H)
    assert y0 < H / 2 and x0 > W / 2


@needs_fonts
def test_contrast_passes_on_dark_panel_and_fails_pale_panel_when_required():
    _, evidence = th.render_thumbnail(_spec(), DARK)
    assert evidence["contrast"]["passes"] is True
    assert evidence["contrast"]["main_title"]["minimum_estimated_contrast"] >= 4.5
    pale = _spec(panel_color="#F0F0F0", panel_opacity=1.0, text_color="#F5F5F5")
    with pytest.raises(th.ContrastError):
        th.render_thumbnail(pale, DARK)
    _, evidence = th.render_thumbnail(pale, DARK, require_contrast=False)
    assert evidence["contrast"]["passes"] is False


@needs_fonts
def test_fitting_raises_when_text_cannot_fit_at_min_scale():
    spec = _spec(titles={"zh": "很長的標題" * 120})
    with pytest.raises(th.ThumbnailFitError):
        th.render_thumbnail(spec, DARK)


@needs_fonts
def test_text_shrinks_to_fit_instead_of_truncating():
    spec = _spec(titles={"zh": "水晶森林雨聲海浪的深夜陪伴" * 3})
    _, evidence = th.render_thumbnail(spec, DARK)
    main = evidence["text_boxes"][0]
    assert main["font_size"] < th.BASE_SIZES["zh"]
    rendered = "".join(main["lines"])
    assert rendered == "水晶森林雨聲海浪的深夜陪伴" * 3


@needs_fonts
def test_missing_glyph_raises_instead_of_tofu():
    spec = _spec(titles={"zh": "雨夜書房\U0010fffd"})
    with pytest.raises(th.ThumbnailError, match="glyph"):
        th.render_thumbnail(spec, DARK)
