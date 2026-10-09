"""Tests for the historical source insert template."""

import hashlib

import numpy as np
from PIL import Image

import pytest

from moviepy.ae import Composition
from moviepy.ae.templates import source_insert as si
from moviepy.ae.templates.source_insert import (
    SourceInsert,
    add_source_inserts,
    citation_image,
    source_insert_image,
    source_insert_layers,
)


MAGENTA = (255, 0, 255)


def _solid(path, size, color=MAGENTA):
    Image.new("RGB", size, color).save(path)
    return str(path)


def _bbox(frame, color=MAGENTA):
    arr = np.asarray(frame)
    ys, xs = np.nonzero((arr == color).all(axis=2))
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _insert(path, **kw):
    kw.setdefault("start", 0.0)
    kw.setdefault("duration", 4.0)
    kw.setdefault("citation", "Museum")
    kw.setdefault("background", "dark")
    return SourceInsert(str(path), **kw)


# --------------------------------------------------------------------------- #
# Citation image.
# --------------------------------------------------------------------------- #


def test_citation_image_is_trimmed_to_text():
    img = citation_image("Wellcome Collection", px=12)
    alpha = np.asarray(img)[..., 3]
    assert img.mode == "RGBA"
    assert alpha[0].any() and alpha[-1].any()  # no empty rows at the edges
    assert alpha[:, 0].any() and alpha[:, -1].any()
    assert img.width > 3 * img.height


def test_citation_image_colour_and_size_scaling():
    small, big = citation_image("Source", px=12), citation_image("Source", px=24)
    assert big.height > small.height
    arr = np.asarray(small)
    solid = arr[arr[..., 3] == 255][:, :3]
    assert (solid == (230, 225, 213)).all(axis=1).any()


def test_citation_image_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        citation_image("   ")


# --------------------------------------------------------------------------- #
# Spec.
# --------------------------------------------------------------------------- #


def test_round_trip_and_validation():
    item = SourceInsert(
        "a.png", 1.5, 5, "Museum", title="T", layout="portrait", sha256="AB" * 32
    )
    assert SourceInsert.from_dict(item.to_dict()) == item
    assert item.sha256 == "ab" * 32
    with pytest.raises(ValueError, match="unknown source insert keys"):
        SourceInsert.from_dict({**item.to_dict(), "bogus": 1})
    with pytest.raises(ValueError, match="layout"):
        SourceInsert("a.png", 0, 4, "x", layout="wide")
    with pytest.raises(ValueError, match="citation"):
        SourceInsert("a.png", 0, 4, " ")
    with pytest.raises(ValueError, match="citation_corner"):
        SourceInsert("a.png", 0, 4, "x", citation_corner="middle")
    with pytest.raises(ValueError, match="twice the fade"):
        SourceInsert("a.png", 0, 0.7, "x")


# --------------------------------------------------------------------------- #
# Fit math.
# --------------------------------------------------------------------------- #


def test_full_layout_contains_image_above_subtitle_rect(tmp_path):
    path = _solid(tmp_path / "a.png", (400, 200))
    frame = source_insert_image(_insert(path), (960, 540))
    assert frame.size == (960, 540) and frame.mode == "RGB"
    x0, y0, x1, y1 = _bbox(frame)
    # box 820 x 380 px (1640 x 760 reference at scale 0.5): height limits
    assert abs((y1 - y0) - 380) <= 1 and abs((x1 - x0) - 760) <= 2
    assert abs((x1 - x0) / (y1 - y0) - 2.0) < 0.02  # aspect preserved
    assert y1 <= 800 * 0.5  # above the subtitle safe rect
    assert abs((x0 + x1) / 2 - 480) <= 1  # centred


def test_full_layout_never_crops_a_wide_image(tmp_path):
    path = _solid(tmp_path / "wide.png", (3000, 300))
    frame = source_insert_image(_insert(path), (1920, 1080))
    x0, y0, x1, y1 = _bbox(frame)
    assert x1 - x0 == 1640 and abs((y1 - y0) - 164) <= 1
    assert x0 >= 140 and x1 <= 1780


def test_full_layout_never_upscales_beyond_2x(tmp_path):
    path = _solid(tmp_path / "tiny.png", (20, 10))
    frame = source_insert_image(_insert(path), (1920, 1080))
    x0, y0, x1, y1 = _bbox(frame)
    assert (x1 - x0, y1 - y0) == (40, 20)


def test_portrait_keeps_native_pixels_and_draws_text_column(tmp_path):
    rng = np.random.default_rng(5)
    raw = rng.integers(1, 255, (300, 200, 3), dtype=np.uint8)
    path = tmp_path / "scan.png"
    Image.fromarray(raw).save(path)
    item = _insert(
        path, layout="portrait", title="Wu Zetian", caption="Portrait", note="c. 1690"
    )
    frame, rect, ink = si._compose(item, (1920, 1080), None, None)
    x, y, w, h = rect
    assert (w, h) == (200, 300) and x == 290
    assert (np.asarray(frame)[y : y + h, x : x + w] == raw).all()  # untouched
    column = np.asarray(frame)[:, x + w + 40 : 1800]
    assert (column == ink).all(axis=2).any()  # ink-coloured text pixels


def test_portrait_text_that_cannot_fit_raises(tmp_path):
    path = _solid(tmp_path / "a.png", (200, 300))
    item = _insert(path, layout="portrait", title="W" * 3000)
    with pytest.raises(ValueError, match="shorten"):
        source_insert_image(item, (1920, 1080))


def test_paper_background_is_light(tmp_path):
    path = _solid(tmp_path / "a.png", (200, 300))
    frame = source_insert_image(_insert(path, background="paper"), (480, 270))
    assert np.asarray(frame)[0, 0].mean() > 150


def test_missing_file_and_sha256(tmp_path):
    with pytest.raises(FileNotFoundError):
        source_insert_image(_insert(tmp_path / "none.png"), (480, 270))
    path = _solid(tmp_path / "a.png", (50, 50))
    with open(path, "rb") as handle:
        good = hashlib.sha256(handle.read()).hexdigest()
    source_insert_image(_insert(path, sha256=good), (480, 270))
    with pytest.raises(ValueError, match="sha256"):
        source_insert_image(_insert(path, sha256="0" * 64), (480, 270))


# --------------------------------------------------------------------------- #
# Citation placement.
# --------------------------------------------------------------------------- #


def test_citation_corner_avoids_subtitle_rect_and_image():
    frame = Image.new("RGB", (1920, 900), (40, 40, 40))  # safe rect reaches y=900
    item = SourceInsert("x.png", 0, 4, "c")
    cite = (200, 12)
    corner, rect = si._pick_corner(item, frame, (0, 0, 1, 1), cite, 1.0)
    assert corner.startswith("top")
    assert not si._hit(rect, (280, 800, 1400, 100))
    # an image covering the top right pushes the citation to the other corner
    corner, rect = si._pick_corner(item, frame, (1500, 0, 420, 300), cite, 1.0)
    assert corner == "top_left"
    with pytest.raises(ValueError, match="subtitle safe rect"):
        si._pick_corner(
            SourceInsert("x.png", 0, 4, "c", citation_corner="bottom_right"),
            frame,
            (0, 0, 1, 1),
            cite,
            1.0,
        )


def test_auto_citation_picks_the_darkest_corner(tmp_path):
    bg = Image.new("RGB", (1920, 1080), (240, 240, 240))
    bg.paste((10, 10, 10), (960, 0, 1920, 1080))
    bg_path = tmp_path / "bg.png"
    bg.save(bg_path)
    path = _solid(tmp_path / "a.png", (100, 100))
    comp = Composition(size=(1920, 1080), fps=5, duration=6)
    _, cite = source_insert_layers(comp, _insert(path, background=str(bg_path)))
    x, y = cite.transform.position.value_at(0.0)
    assert x > 960  # right half is the dark one
    assert not si._hit((x, y, 10, 10), (280, 800, 1400, 100))


# --------------------------------------------------------------------------- #
# Layers and fades.
# --------------------------------------------------------------------------- #


def test_fades_run_on_the_layer_clock_for_a_late_insert(tmp_path):
    path = _solid(tmp_path / "a.png", (80, 80))
    comp = Composition(size=(192, 108), fps=10, duration=12, bg_color=(0, 0, 0))
    item = _insert(path, start=5.0, duration=4.0, fade=0.4)
    picture, cite = source_insert_layers(comp, item)
    assert len(comp.layers) == 2
    for layer in (picture, cite):
        assert (layer.in_point, layer.out_point) == (5.0, 9.0)
        op = layer.transform.opacity
        assert op.value_at(layer.source_time(5.0)) == 0.0
        assert op.value_at(layer.source_time(5.2)) == pytest.approx(50.0)
        assert op.value_at(layer.source_time(7.0)) == 100.0
        assert op.value_at(layer.source_time(8.8)) == pytest.approx(50.0)
    assert comp.get_frame(2.0).max() == 0  # before: nothing
    assert comp.get_frame(7.0).max() > 0  # regression: a late insert is visible
    assert comp.get_frame(10.0).max() == 0  # after


def test_add_source_inserts_accepts_dicts_and_is_atomic(tmp_path):
    path = _solid(tmp_path / "a.png", (80, 80))
    comp = Composition(size=(192, 108), fps=10, duration=12)
    good = _insert(path, start=1.0).to_dict()
    layers = add_source_inserts(comp, [good, _insert(path, start=6.0)])
    assert len(layers) == 4 and len(comp.layers) == 4
    bad = _insert(tmp_path / "missing.png").to_dict()
    with pytest.raises(FileNotFoundError):
        add_source_inserts(comp, [good, bad])
    assert len(comp.layers) == 4
    with pytest.raises(TypeError):
        add_source_inserts(comp, [good], bogus=1)
