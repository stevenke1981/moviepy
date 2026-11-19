"""Tests for the chapter scene overlay pack (logo, watermark, vertical title, CTA)."""

import json

import numpy as np
from PIL import Image

import pytest

from moviepy.ae import Composition
from moviepy.ae.templates.music_render import _rgba_frame
from moviepy.ae.templates.scene_overlay import (
    SceneLayout,
    add_scene_overlays,
    cta_image,
    export_scene_overlay,
    logo_image,
    scene_overlay_layers,
    vertical_title_image,
    watermark_image,
)


def by_name(layers):
    return {layer.name: layer for layer in layers}


# --------------------------------------------------------------------------- #
# SceneLayout
# --------------------------------------------------------------------------- #


def test_layout_scaled_1920_to_1280x720():
    sc = SceneLayout().scaled((1280, 720))
    assert sc.scale == pytest.approx(2 / 3)
    assert sc.logo == (128, 83, 147, 73)
    assert sc.watermark == (976, 43, 239, 39)
    assert sc.title == (1013, 100, 117, 200)
    assert sc.cta == (851, 453, 280, 73)
    assert sc.cta_inset == 8
    assert (sc.title_font_px, sc.cta_font_px, sc.watermark_font_px) == (29, 24, 17)
    for rect in (sc.logo, sc.watermark, sc.title, sc.cta):
        x, y, w, h = rect
        assert x >= 0 and y >= 0 and x + w <= 1280 and y + h <= 720


def test_layout_scaled_identity_and_odd_aspect_stays_inside():
    layout = SceneLayout()
    assert layout.scaled((1920, 1080)).cta == layout.cta_rect
    sc = layout.scaled((1000, 1000))  # square: sizes follow the smaller ratio
    assert sc.scale == pytest.approx(1000 / 1920)
    for x, y, w, h in (sc.logo, sc.watermark, sc.title, sc.cta):
        assert x + w <= 1000 and y + h <= 1000


def test_layout_dict_round_trip_and_json():
    layout = SceneLayout(fade=0.5, cta_pop=((0, 70), (4, 110), (9, 100)))
    data = layout.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert SceneLayout.from_dict(data) == layout
    assert SceneLayout.from_dict(json.loads(layout.to_json())) == layout
    assert SceneLayout.from_dict({}).cta_rect == (1276, 680, 420, 110)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"logo_rect": (1900, 0, 100, 50)}, "outside"),
        ({"cta_rect": (0, 0, 0, 10)}, "outside"),
        ({"title_rect": (-1, 0, 10, 10)}, "outside"),
        ({"duration": 0.8, "fade": 0.4}, "twice the fade"),
        ({"duration": 4, "fade": -0.1}, "negative"),
        ({"cta_pop": ((0, 80), (0, 100))}, "increasing"),
        ({"cta_pop": ((0, 0),)}, "positive scale"),
        ({"cta_pop": ((0, 80), (200, 100))}, "before the fade-out"),
        ({"cta_inset": 60}, "cta_inset"),
        ({"watermark_opacity": 1.5}, "watermark_opacity"),
        ({"cta_color": (1, 2)}, "cta_color"),
        ({"title_floor_px": 99}, "title_floor_px"),
    ],
)
def test_layout_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SceneLayout(**kwargs)


def test_layout_from_dict_rejects_unknown_field():
    with pytest.raises(ValueError, match="unknown"):
        SceneLayout.from_dict({"fades": 1})


def test_pop_seconds_follow_pop_fps():
    layout = SceneLayout()
    assert layout.pop_seconds() == [(0.0, 80.0), (0.2, 104.0), (10 / 30, 100.0)]


# --------------------------------------------------------------------------- #
# Pictures
# --------------------------------------------------------------------------- #


def test_vertical_title_columns_run_right_to_left():
    # Column 0 (read first) is heavy, column 1 is a thin letter: the heavy
    # column must sit in the right half of the image.
    img = vertical_title_image("MMMM|iiii", (80, 120), font_px=22, panel=False)
    alpha = img[..., 3].astype(float)
    left, right = alpha[:, :40].sum(), alpha[:, 40:].sum()
    assert right > 2 * left > 0
    flipped = vertical_title_image("iiii|MMMM", (80, 120), font_px=22, panel=False)
    fa = flipped[..., 3].astype(float)
    assert fa[:, :40].sum() > 2 * fa[:, 40:].sum() > 0


def test_vertical_title_reads_top_to_bottom():
    img = vertical_title_image("Mi", (60, 140), font_px=24, panel=False)
    alpha = img[..., 3].astype(float)
    assert alpha[:70].sum() > 2 * alpha[70:].sum() > 0


def test_vertical_title_panel_and_alpha_contract():
    img = vertical_title_image("AB", (60, 90), font_px=20)
    assert img.shape == (90, 60, 4) and img.dtype == np.uint8
    assert img[45, 4, 3] in range(150, 165)  # panel body alpha 158
    assert img[0, 0, 3] == 0  # rounded corner is transparent
    assert img[45, 0, 3] > 100  # outline on the left edge


def test_vertical_title_auto_shrinks_but_keeps_every_character():
    text = "A" * 14
    big = vertical_title_image(text, (60, 200), font_px=44, floor_px=10, panel=False)
    # 14 stacked characters cannot fit at 44px (one cell ~ 54px); ink must stay
    # inside the box and is not clipped at the edges.
    alpha = big[..., 3]
    assert alpha[0].max() == 0 and alpha[-1].max() == 0
    assert alpha[:, 0].max() == 0 and alpha[:, -1].max() == 0
    assert alpha.sum() > 0


def test_vertical_title_raises_when_it_cannot_fit():
    with pytest.raises(ValueError, match="does not fit"):
        vertical_title_image("A" * 40, (60, 100), font_px=44, floor_px=20)
    with pytest.raises(ValueError, match="empty"):
        vertical_title_image("  ", (60, 100))


def test_vertical_title_cjk_punctuation_without_font_does_not_crash():
    img = vertical_title_image("「雷」，。", (80, 200), font_px=24)
    assert img.shape == (200, 80, 4)


def test_cta_pill_triangle_and_fit():
    img = cta_image("GO", (160, 48), font_px=20)
    assert img.shape == (48, 160, 4)
    assert img[24, 80, 3] == 255 or img[24, 80, 3] > 200
    assert img[0, 0, 3] == 0 and img[47, 159, 3] == 0  # pill corners are empty
    red = img[24, 6]
    assert red[0] > 180 and red[1] < 100 and red[2] < 100
    # the drawn triangle: white pixels to the right of the label, none at the edge
    white = (img[..., :3].min(axis=2) > 230) & (img[..., 3] > 200)
    xs = np.nonzero(white.any(axis=0))[0]
    assert xs.max() < 140
    assert cta_image("GO", (160, 48), font_px=20, triangle=False)[..., :3].shape[2] == 3
    with pytest.raises(ValueError, match="CTA"):
        cta_image("A very long subscribe label", (60, 30), font_px=20, floor_px=10)


def test_watermark_opacity_and_fit():
    img = watermark_image("@channel", (180, 36), opacity=0.55, font_px=20)
    assert 100 <= img[..., 3].max() <= 150
    assert watermark_image("@channel", (180, 36), opacity=0.0)[..., 3].max() == 0
    wide = watermark_image(
        "@a_much_longer_channel_name", (120, 30), font_px=26, floor_px=6
    )
    assert wide[..., 3].max() > 0
    assert wide[:, 0, 3].max() < 8 and wide[:, -1, 3].max() < 8  # shadow only
    with pytest.raises(ValueError, match="watermark"):
        watermark_image("@" + "x" * 80, (60, 20), floor_px=12)
    with pytest.raises(ValueError):
        watermark_image("@x", (60, 20), opacity=2)


def test_logo_text_image_and_array(tmp_path):
    text = logo_image("Logo", (90, 40), font_px=24)
    assert text.shape == (40, 90, 4) and text[..., 3].max() > 0
    path = tmp_path / "logo.png"
    Image.new("RGBA", (40, 20), (10, 200, 30, 255)).save(path)
    fitted = logo_image(str(path), (80, 80))
    assert fitted.shape == (80, 80, 4)
    assert fitted[40, 40].tolist() == [10, 200, 30, 255]
    assert fitted[2, 40, 3] == 0  # letterbox, not stretched
    assert logo_image(path, (80, 80)).shape == (80, 80, 4)
    assert logo_image(np.zeros((8, 8, 4), np.uint8), (16, 16)).shape == (16, 16, 4)


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #

SIZE = (320, 180)


def make_layers(start=1.0, **kw):
    options = dict(chapter_title="AB", logo="Logo", watermark="@x", start=start)
    options.update(kw)
    return scene_overlay_layers(SceneLayout(), SIZE, **options)


def test_layers_order_names_and_window():
    layers = make_layers(start=2.0)
    assert [layer.name for layer in layers] == [
        "logo",
        "watermark",
        "chapter title",
        "subscribe",
    ]
    for layer in layers:
        assert layer.in_point == 2.0 and layer.out_point == 6.0
        assert layer.is_active(2.0) and layer.is_active(5.99)
        assert not layer.is_active(1.99) and not layer.is_active(6.0)
    assert [x.name for x in scene_overlay_layers(SceneLayout(), SIZE)] == ["subscribe"]
    assert scene_overlay_layers(SceneLayout(), SIZE, cta_text=None) == []


def test_opacity_zero_before_one_mid_zero_after():
    # Transform properties run on the layer clock (t - start_time).
    layout = SceneLayout()
    start = 1.0
    for layer in make_layers(start=start):
        opacity = layer.transform.opacity
        assert opacity.value_at(layer.source_time(0.0)) == 0.0  # before the window
        assert opacity.value_at(layer.source_time(start)) == 0.0
        assert opacity.value_at(
            layer.source_time(start + layout.fade / 2)
        ) == pytest.approx(50.0)
        assert opacity.value_at(layer.source_time(start + 2.0)) == 100.0  # mid
        assert opacity.value_at(
            layer.source_time(start + layout.duration - layout.fade / 2)
        ) == (pytest.approx(50.0))
        assert opacity.value_at(layer.source_time(start + layout.duration)) == 0.0
        assert opacity.value_at(layer.source_time(start + 50.0)) == 0.0  # after


def test_cta_pop_scale_values_and_steady_state():
    layout = SceneLayout()
    start = 1.5
    cta = by_name(make_layers(start=start))["subscribe"]
    scale = cta.transform.scale
    fps = layout.pop_fps
    assert scale.value_at(cta.source_time(start + 0 / fps)) == pytest.approx(
        (80.0, 80.0)
    )
    assert scale.value_at(cta.source_time(start + 6 / fps)) == pytest.approx(
        (104.0, 104.0)
    )
    assert scale.value_at(cta.source_time(start + 10 / fps)) == pytest.approx(
        (100.0, 100.0)
    )
    assert 80.0 < scale.value_at(cta.source_time(start + 3 / fps))[0] < 104.0
    for t in (start + 0.5, start + 2.0, start + 3.9):
        assert scale.value_at(cta.source_time(t)) == pytest.approx((100.0, 100.0))
    # the other layers never scale
    assert by_name(make_layers())["chapter title"].transform.scale.value_at(0.5) == (
        pytest.approx((100.0, 100.0))
    )


def test_layers_render_with_fade_and_pop_in_a_composition():
    comp = Composition(size=SIZE, fps=10, duration=8, transparent=True)
    for layer in make_layers(start=1.0, watermark_span=(0.0, 8.0)):
        comp.add_layer(layer)
    alpha = lambda t: _rgba_frame(comp, t)[..., 3].astype(float)  # noqa: E731
    assert alpha(0.5).sum() > 0  # the watermark span starts earlier
    assert alpha(0.5).sum() < alpha(3.0).sum()  # then the rest joins in
    cta_x, cta_y, cta_w, cta_h = SceneLayout().scaled(SIZE).cta
    box = lambda t: alpha(t)[cta_y : cta_y + cta_h, cta_x : cta_x + cta_w]  # noqa: E731
    early = box(1.0 + 0.2 / 3.0 + 0.1)  # near the 80% start, mostly transparent
    full = box(3.0)
    assert full.sum() > early.sum()
    assert alpha(7.5).sum() < alpha(3.0).sum()


def test_watermark_span_must_be_longer_than_fades():
    with pytest.raises(ValueError, match="watermark_span"):
        make_layers(watermark_span=(0.0, 0.5))


def test_title_too_long_raises_before_any_layer_exists():
    comp = Composition(size=SIZE, fps=10, duration=8, transparent=True)
    with pytest.raises(ValueError, match="does not fit"):
        add_scene_overlays(comp, SceneLayout(), [(0, "A" * 200)])
    assert len(comp.layers) == 0


# --------------------------------------------------------------------------- #
# add_scene_overlays
# --------------------------------------------------------------------------- #


def test_add_scene_overlays_one_set_per_chapter_single_watermark():
    comp = Composition(size=SIZE, fps=10, duration=20, transparent=True)
    layers = add_scene_overlays(
        comp,
        SceneLayout(),
        [(8.0, "B"), (0.0, "A"), (4.0, "C")],  # touching windows are fine
        logo="Logo",
        watermark="@x",
    )
    names = [layer.name for layer in layers]
    assert names.count("watermark") == 1 and names.count("subscribe") == 3
    assert names.count("chapter title") == 3 and names.count("logo") == 3
    assert len(comp.layers) == len(layers)
    wm = by_name(layers)["watermark"]
    assert (wm.in_point, wm.out_point) == (0.0, 12.0)
    starts = sorted(x.in_point for x in layers if x.name == "subscribe")
    assert starts == [0.0, 4.0, 8.0]


def test_add_scene_overlays_watermark_per_chapter_and_explicit_span():
    comp = Composition(size=SIZE, fps=10, duration=20, transparent=True)
    layers = add_scene_overlays(
        comp,
        SceneLayout(),
        [(0, "A"), (5, "B")],
        watermark="@x",
        watermark_span="chapter",
    )
    assert [x.name for x in layers].count("watermark") == 2
    comp2 = Composition(size=SIZE, fps=10, duration=20, transparent=True)
    layers = add_scene_overlays(
        comp2, SceneLayout(), [(0, "A"), (5, "B")], watermark="@x",
        watermark_span=(0.0, 20.0),
    )  # fmt: skip
    wm = [x for x in layers if x.name == "watermark"]
    assert len(wm) == 1 and wm[0].out_point == 20.0


def test_add_scene_overlays_rejects_overlap_and_bad_windows():
    comp = Composition(size=SIZE, fps=10, duration=10, transparent=True)
    layout = SceneLayout()
    with pytest.raises(ValueError, match="overlap"):
        add_scene_overlays(comp, layout, [(0, "A"), (3.9, "B")])
    with pytest.raises(ValueError, match="overlap"):
        add_scene_overlays(comp, layout, [(5, "B"), (1, "A"), (2, "C")])
    with pytest.raises(ValueError, match="negative"):
        add_scene_overlays(comp, layout, [(-1, "A")])
    with pytest.raises(ValueError, match="composition"):
        add_scene_overlays(comp, layout, [(7, "A")])
    assert len(comp.layers) == 0
    assert add_scene_overlays(comp, layout, []) == []


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def test_export_scene_overlay_writes_alpha_mov_with_expected_frames(tmp_path):
    layout = SceneLayout(duration=2.0, fade=0.25)
    path = tmp_path / "chapter.mov"
    report = export_scene_overlay(
        path, layout, (128, 72), 4, chapter_title="A", cta_text="GO"
    )
    assert path.is_file() and (tmp_path / "chapter.mov.json").is_file()
    assert report["frames"] == report["expected_frames"] == 8
    assert report["roundtrip"]["frames_decoded"] == 8
    assert report["roundtrip"]["alpha_max_abs_diff"] == 0
    assert report["alpha"]["max"] > 0 and report["codec"] == "qtrle"
    saved = json.loads((tmp_path / "chapter.mov.json").read_text("utf-8"))
    assert saved["size"] == [128, 72]
    with pytest.raises(FileExistsError):
        export_scene_overlay(path, layout, (128, 72), 4, cta_text="GO")
    export_scene_overlay(path, layout, (128, 72), 4, cta_text="GO", overwrite=True)


def test_export_rejects_fractional_frame_count(tmp_path):
    with pytest.raises(ValueError, match="whole number"):
        export_scene_overlay(tmp_path / "x.mov", SceneLayout(), (128, 72), 2.3)


def test_space_between_cjk_is_a_preferred_column_break():
    from moviepy.ae.templates.scene_overlay import _tidy_title

    assert _tidy_title("第三章 雷劫守候", "|") == "第三章|雷劫守候"
    assert _tidy_title("第三章 雷劫守候") == "第三章雷劫守候"
    assert _tidy_title("Chapter  3") == "Chapter 3"


def test_late_chapter_renders_inside_its_window_only():
    # Regression: keys must run on the layer clock, not composition time.
    comp = Composition(size=(320, 180), fps=12, duration=10, bg_color=(220, 40, 40))
    layout = SceneLayout(duration=2.0, fade=0.4)
    add_scene_overlays(comp, layout, [(5.0, "Part")], logo="Lamp", cta_text="Go")

    def changed(t):
        return int(np.abs(comp.get_frame(t).astype(int) - (220, 40, 40)).sum())

    assert changed(4.9) == 0 and changed(7.1) == 0
    assert changed(5.2) < changed(6.0)
    assert changed(6.0) > 0
