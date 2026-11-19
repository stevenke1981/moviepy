"""Line plates, glossary highlighting and ``load_glossary`` (font independent)."""

import json

import numpy as np

import pytest

from moviepy.ae.templates.presets import ChannelPreset, SubtitleStyle, get_preset
from moviepy.ae.templates.subtitles import (
    Cue,
    _highlight_marks,
    _highlight_pattern,
    burn_subtitles,
    load_glossary,
    subtitle_layer,
    to_ass,
)


SIZE = (640, 360)
HL = (200, 30, 30)


def make_preset(**style):
    base = get_preset("nightlamp_story")
    values = dict(plate_opacity=0.0, plate_padding=0, highlight_color=None)
    values.update(style)
    return base.with_overrides(subtitles=values)


def buffer_of(preset, cues, secondary=None, **kw):
    layer = subtitle_layer(
        cues, preset, secondary=secondary, font="default", size=SIZE, **kw
    )
    return layer, layer.source_buffer(cues[0].start + 0.1)


def runs(mask):
    """Index ranges ``(first, last)`` of consecutive True values."""
    out, start = [], None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def hl_mask(rgba):
    rgb = rgba[..., :3]
    near = np.abs(rgb - np.asarray(HL, np.float32) / 255.0).max(axis=-1) < 0.02
    return near & (rgba[..., 3] > 0.99)


# -- style --------------------------------------------------------------- #


def test_style_defaults_and_validation():
    style = SubtitleStyle()
    assert style.plate_color == (0, 0, 0)
    assert style.plate_opacity == 0.0
    assert style.plate_padding == 0
    assert style.highlight_color is None
    assert SubtitleStyle(plate_color="#102030").plate_color == (16, 32, 48)
    assert SubtitleStyle(highlight_color=[1, 2, 3]).highlight_color == (1, 2, 3)
    for bad in (
        dict(plate_opacity=-0.1),
        dict(plate_opacity=1.5),
        dict(plate_opacity=True),
        dict(plate_opacity=float("nan")),
        dict(plate_padding=-1),
        dict(plate_color=(0, 0)),
        dict(plate_color=(0, 0, 300)),
        dict(highlight_color="red"),
        dict(highlight_color=(1, 2)),
    ):
        with pytest.raises(ValueError):
            SubtitleStyle(**bad)


def test_style_round_trip_and_unknown_key():
    style = SubtitleStyle(
        plate_color=(5, 6, 7),
        plate_opacity=0.25,
        plate_padding=8,
        highlight_color=(9, 8, 7),
    )
    data = json.loads(json.dumps(style.to_dict()))
    assert data["highlight_color"] == [9, 8, 7]
    assert SubtitleStyle.from_dict(data) == style
    assert SubtitleStyle.from_dict(SubtitleStyle().to_dict()) == SubtitleStyle()
    with pytest.raises(ValueError):
        SubtitleStyle.from_dict({"bogus": 1})
    preset = ChannelPreset("x", subtitles=style)
    assert ChannelPreset.from_json(preset.to_json()) == preset


@pytest.mark.parametrize("name", ["nightlamp_history", "nightlamp_story"])
def test_nightlamp_presets_use_r2b_look(name):
    style = get_preset(name).subtitles
    assert (style.primary_size, style.secondary_size) == (72, 42)
    assert style.plate_opacity == 0.05
    assert style.plate_padding == 0
    assert style.plate_color == (0, 0, 0)
    assert style.highlight_color == (0xC9, 0xA3, 0x5D)


def test_to_ass_ignores_plate_defaults():
    ass = to_ass([Cue(0, 1, "x")], SubtitleStyle())
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Primary" in ass


# -- plate --------------------------------------------------------------- #


def test_no_plate_by_default_has_no_faint_pixels():
    _, buf = buffer_of(make_preset(), [Cue(0, 2, "Hello")])
    alpha = buf.rgba[..., 3]
    faint = (alpha > 0) & (alpha < 0.04)
    assert not faint.any() or alpha[faint].min() > 0  # only antialiasing
    # nothing outside the glyph neighbourhood: corners are transparent
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0


def test_plate_alpha_inside_line_box_and_zero_between_blocks():
    cues, sec = [Cue(0, 2, "Hello World")], [Cue(0, 2, "Second language", "en")]
    _, plain = buffer_of(make_preset(), cues, sec)
    _, plated = buffer_of(make_preset(plate_opacity=0.05), cues, sec)
    a0, a1 = plain.rgba[..., 3], plated.rgba[..., 3]
    assert a0.shape == a1.shape
    boxes = runs((a1 > 0).any(axis=1))
    assert len(boxes) == 2  # one plate per language block, nothing between
    gap_rows = a1[boxes[0][1] + 1 : boxes[1][0]]
    assert gap_rows.size and gap_rows.max() == 0.0
    for top, bottom in boxes:
        cols = runs((a1[top : bottom + 1] > 0).any(axis=0))
        assert len(cols) == 1
        left, right = cols[0]
        box = a1[top : bottom + 1, left : right + 1]
        empty = a0[top : bottom + 1, left : right + 1] == 0
        assert empty.any()
        assert np.allclose(box[empty], 0.05, atol=1e-6)
        # outside the box the layer is untouched
        assert a1[:top].max() <= a1.max()
    outside = np.ones_like(a1, bool)
    for top, bottom in boxes:
        left = np.flatnonzero((a1[top : bottom + 1] > 0).any(axis=0))
        outside[top : bottom + 1, left[0] : left[-1] + 1] = False
    assert a1[outside].max() == 0.0


def test_plate_covers_exact_text_bbox_with_stroke():
    from PIL import Image, ImageDraw

    preset = make_preset(plate_opacity=0.05)
    layer, buf = buffer_of(preset, [Cue(0, 2, "Hello World")])
    font, stroke = layer._fonts["primary"], layer._stroke
    box = ImageDraw.Draw(Image.new("L", (4, 4))).textbbox(
        (0, 0), "Hello World", font=font, stroke_width=stroke
    )
    alpha = buf.rgba[..., 3]
    rows = np.flatnonzero((alpha > 0).any(axis=1))
    cols = np.flatnonzero((alpha > 0).any(axis=0))
    assert rows[-1] - rows[0] + 1 == box[3] - box[1]
    assert cols[-1] - cols[0] + 1 == box[2] - box[0]


def test_plate_padding_grows_plate_in_every_direction():
    def extent(padding):
        _, buf = buffer_of(
            make_preset(plate_opacity=0.05, plate_padding=padding),
            [Cue(0, 2, "Hello World")],
        )
        alpha = buf.rgba[..., 3]
        rows = np.flatnonzero((alpha > 0).any(axis=1))
        cols = np.flatnonzero((alpha > 0).any(axis=0))
        return rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1

    h0, w0 = extent(0)
    h1, w1 = extent(30)  # 30 px at 1080p -> 10 px at 360p
    assert (h1 - h0, w1 - w0) == (20, 20)


def test_text_pixels_composite_over_plate_premultiplied():
    cues = [Cue(0, 2, "Hello World")]
    _, plain = buffer_of(make_preset(), cues)
    _, plated = buffer_of(make_preset(plate_opacity=0.5, plate_color=(0, 0, 255)), cues)
    p0, p1 = plain.rgba, plated.rgba
    a0, a1 = p0[..., 3], p1[..., 3]
    assert (a1 >= a0 - 1e-6).all()
    assert (p1[..., :3] <= a1[..., None] + 1e-5).all()  # still premultiplied
    solid = a0 > 0.999
    assert solid.any()
    assert np.allclose(p1[solid], p0[solid], atol=1e-6)  # opaque text unchanged
    partial = (a0 > 0.05) & (a0 < 0.95)
    assert partial.any()
    expected = a0 + 0.5 * (1 - a0)
    inside = a1[partial]
    assert np.allclose(inside, expected[partial], atol=1e-5)
    blue_extra = p1[..., 2] - p0[..., 2]
    assert np.allclose(blue_extra[partial], (0.5 * (1 - a0))[partial], atol=1e-5)


def test_plate_pad_keeps_block_metrics_and_to_ass_lift_in_sync():
    from moviepy.ae.templates.subtitles import _block_height

    layer, buf = buffer_of(
        make_preset(plate_opacity=0.05, plate_padding=30), [Cue(0, 2, "Hello")]
    )
    font = layer._fonts["primary"]
    assert buf.size[1] == _block_height(font, layer._stroke, 1, 10)


# -- highlight ----------------------------------------------------------- #


def test_highlight_colours_only_matching_whole_words():
    preset = make_preset(highlight_color=HL)
    _, hit = buffer_of(preset, [Cue(0, 2, "Alpha Beta Gamma")], highlight=["Beta"])
    _, miss = buffer_of(preset, [Cue(0, 2, "Alpha Betamax Gamma")], highlight=["Beta"])
    _, off = buffer_of(preset, [Cue(0, 2, "Alpha Beta Gamma")])
    assert hl_mask(hit.rgba).sum() > 20
    assert hl_mask(miss.rgba).sum() == 0
    assert hl_mask(off.rgba).sum() == 0
    cols = np.flatnonzero(hl_mask(hit.rgba).any(axis=0))
    full = np.flatnonzero((hit.rgba[..., 3] > 0.99).any(axis=0))
    assert full[0] < cols[0] and cols[-1] < full[-1]  # only the middle word


def test_highlight_applies_to_both_lanes():
    preset = make_preset(highlight_color=HL)
    layer, buf = buffer_of(
        preset,
        [Cue(0, 2, "Wu Zetian rules")],
        [Cue(0, 2, "Wu Zetian rules", "en")],
        highlight=["Wu Zetian"],
    )
    mask = hl_mask(buf.rgba)
    rows = runs(mask.any(axis=1))
    assert len(rows) == 2  # one highlighted band per language block
    text_rows = runs((buf.rgba[..., 3] > 0.99).any(axis=1))
    assert len(text_rows) == 2


def test_cjk_substring_longest_first_non_overlapping():
    pattern = _highlight_pattern(["武則", "武則天", "天后"])
    assert _highlight_marks(("武則天后",), pattern, True) == (((0, 3),),)
    assert _highlight_marks(("天后武則",), pattern, True) == (((0, 2), (2, 4)),)


def test_english_terms_use_word_boundaries_and_longest_first():
    pattern = _highlight_pattern(["Wu", "Wu Zetian"])
    marks = _highlight_marks(("Wu Zetian and Wun",), pattern, False)
    assert marks == (((0, 9),),)


def test_term_split_by_line_break_is_marked_on_both_pieces():
    pattern = _highlight_pattern(["Wu Zetian", "李世民"])
    assert _highlight_marks(("Hello Wu", "Zetian rules"), pattern, False) == (
        ((6, 8),),
        ((0, 6),),
    )
    assert _highlight_marks(("李世", "民說"), pattern, True) == (((0, 2),), ((0, 1),))
    # a Latin term never matches across a CJK-style (no-space) join
    assert _highlight_marks(("Wu", "Zetian"), pattern, True) == ((), ())


def test_split_term_is_coloured_on_both_rendered_lines():
    preset = make_preset(highlight_color=HL)
    layer, buf = buffer_of(
        preset, [Cue(0, 2, "Hello Wu\nZetian", "en")], highlight=["Wu Zetian"]
    )
    assert layer._lanes["primary"]["lines"][0] == ("Hello Wu", "Zetian")
    assert [bool(m) for m in layer._lanes["primary"]["marks"][0]] == [True, True]
    rows = runs(hl_mask(buf.rgba).any(axis=1))
    assert len(rows) == 2  # coloured pixels on each of the two lines


def test_highlight_terms_are_protected_from_line_breaks():
    preset = make_preset(highlight_color=HL)
    layer = subtitle_layer(
        [
            Cue(
                0,
                2,
                "aaaa bbbb Wu Zetian cccc dddd eeee ffff gggg hhhh iiii jjjj",
                "en",
            )
        ],
        preset,
        font="default",
        size=(320, 180),
        highlight=["Wu Zetian"],
    )
    lines = layer._lanes["primary"]["lines"][0]
    assert len(lines) == 2
    assert not any(line.endswith("Wu") for line in lines)
    assert "Wu Zetian" in layer._protected
    assert layer._protected == ("Wu Zetian",)
    both = subtitle_layer(
        [Cue(0, 1, "x")], preset, font="default", protected=["x y"], highlight=["x y"]
    )
    assert both._protected == ("x y",)


def test_highlight_falls_back_to_palette_accent_or_raises():
    cues = [Cue(0, 2, "Alpha Beta")]
    story = make_preset()  # highlight_color None, palette has "accent"
    accent = story.palette["accent"]
    layer, buf = buffer_of(story, cues, highlight=["Beta"])
    assert layer._hl_color == accent
    bare = story.with_overrides(palette={})
    with pytest.raises(ValueError, match="accent"):
        subtitle_layer(cues, bare, font="default", size=SIZE, highlight=["Beta"])
    # no terms -> no colour needed
    subtitle_layer(cues, bare, font="default", size=SIZE)


def test_burn_subtitles_passes_highlight():
    from moviepy.ae.composition import Composition

    comp = Composition(size=SIZE, fps=24, duration=3, name="c")
    preset = make_preset(highlight_color=HL)
    burn_subtitles(
        comp, [Cue(0, 2, "Alpha Beta")], preset, font="default", highlight=["Beta"]
    )
    layer = comp.layers[-1]
    assert layer._hl_color == HL


# -- glossary ------------------------------------------------------------ #


GLOSSARY = {
    "人物": {
        "李治／唐高宗／高宗": "Emperor Gaozong",
        "武則天/武曌": "Wu Zetian",
        "王皇后": "Empress Wang",
        "狄仁傑、狄公": "Wu Zetian",
    },
    "地點": {"洛陽": "Luoyang"},
    "note": "ignored",
    "empty": {"空": ""},
}


def test_load_glossary_from_dict():
    terms = load_glossary(GLOSSARY)
    assert isinstance(terms, tuple)
    assert len(terms) == len(set(terms))
    for term in (
        "李治",
        "唐高宗",
        "高宗",
        "武則天",
        "武曌",
        "狄仁傑",
        "狄公",
        "Emperor Gaozong",
        "Wu Zetian",
        "Luoyang",
        "空",
    ):
        assert term in terms
    assert "" not in terms and "ignored" not in terms
    assert terms.count("Wu Zetian") == 1
    assert [len(t) for t in terms] == sorted((len(t) for t in terms), reverse=True)
    assert terms[0] == "Emperor Gaozong"


def test_load_glossary_from_path_and_text(tmp_path):
    path = tmp_path / "glossary.json"
    path.write_text(json.dumps(GLOSSARY, ensure_ascii=False), encoding="utf-8")
    expected = load_glossary(GLOSSARY)
    assert load_glossary(path) == expected
    assert load_glossary(str(path)) == expected
    assert load_glossary(json.dumps(GLOSSARY, ensure_ascii=False)) == expected
    with pytest.raises(ValueError):
        load_glossary([1, 2])


def test_missing_glyph_raises_instead_of_drawing_boxes(tmp_path, monkeypatch):
    from PIL import ImageFont

    from moviepy.ae.templates import paper, subtitles
    from moviepy.ae.templates.subtitles import Cue, LayoutError, SubtitleLayer

    font = tmp_path / "subset.otf"
    font.write_bytes(b"")
    monkeypatch.setattr(
        subtitles, "_load_font", lambda spec, px: ImageFont.load_default(size=20)
    )
    monkeypatch.setattr(paper, "has_glyph", lambda path, char: char != "Z")
    preset = get_preset("nightlamp_story")
    ok = [Cue(0, 1, "abc", "en")]
    SubtitleLayer(ok, preset, font=str(font), size=(640, 360))
    bad = [Cue(0, 1, "abc", "en"), Cue(1, 2, "xZy", "en")]
    with pytest.raises(LayoutError, match="no glyph for 'Z'.*1.00 s"):
        SubtitleLayer(bad, preset, font=str(font), size=(640, 360))
    # A PIL font object or Pillow's default font is not checked.
    SubtitleLayer(bad, preset, font="default", size=(640, 360))
