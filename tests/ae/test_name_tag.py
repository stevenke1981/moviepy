"""Tests for the character name tag template."""

import json

import numpy as np

import pytest

from moviepy import ImageClip
from moviepy.ae import Composition
from moviepy.ae.templates.name_tag import (
    SUBTITLE_SAFE_RECT,
    NameTag,
    add_name_tags,
    name_tag_image,
    name_tag_layer,
    place_name_tag,
)


CANVAS = (1920, 1080)


def _overlap(a, b):
    return (
        a[0] < b[0] + b[2]
        and b[0] < a[0] + a[2]
        and a[1] < b[1] + b[3]
        and b[1] < a[1] + a[3]
    )


# --------------------------------------------------------------------------- #
# Geometry.
# --------------------------------------------------------------------------- #


def test_auto_prefers_roomier_horizontal_side():
    card = (120, 300)
    x, y, side = place_name_tag((1200, 200, 400, 600), card, CANVAS)
    assert side == "left" and x + card[0] <= 1200
    x, y, side = place_name_tag((200, 200, 400, 600), card, CANVAS)
    assert side == "right" and x >= 600


def test_auto_falls_back_to_vertical_sides():
    # tall subject filling the width: only above/below can work
    x, y, side = place_name_tag((100, 400, 1700, 300), (300, 120), CANVAS)
    assert side in ("above", "below")
    assert not _overlap((x, y, 300, 120), (100, 400, 1700, 300))


def test_explicit_side_and_gap():
    x, y, side = place_name_tag((600, 300, 400, 400), (100, 200), CANVAS, side="left")
    assert (x, side) == (600 - 24 - 100, "left")
    x, y, side = place_name_tag(
        (600, 300, 400, 400), (100, 200), CANVAS, side="below", gap=10
    )
    assert (y, side) == (710, "below")


@pytest.mark.parametrize(
    "align,expect", [("top", 300), ("center", 400), ("bottom", 500)]
)
def test_align_along_edge(align, expect):
    _, y, _ = place_name_tag(
        (600, 300, 400, 400), (100, 200), CANVAS, side="right", align=align
    )
    assert y == expect


def test_clamped_inside_margin():
    # subject near the top: card would start above the margin, so it is clamped
    x, y, _ = place_name_tag(
        (900, 10, 300, 300), (100, 200), CANVAS, side="right", margin=48
    )
    assert y == 48
    x, y, _ = place_name_tag(
        (900, 700, 300, 380), (100, 200), CANVAS, side="right", align="bottom"
    )
    assert y + 200 <= 1080 - 48


def test_avoid_rect_respected():
    avoid = [SUBTITLE_SAFE_RECT]
    subject = (100, 200, 400, 600)
    x, y, side = place_name_tag(
        subject, (120, 300), CANVAS, side="right", avoid=avoid, align="bottom"
    )
    assert side == "right"
    assert not _overlap((x, y, 120, 300), SUBTITLE_SAFE_RECT)
    assert not _overlap((x, y, 120, 300), subject)
    # auto also keeps off the avoid rect
    x, y, side = place_name_tag(
        (1200, 400, 500, 650), (120, 300), CANVAS, avoid=avoid, align="bottom"
    )
    assert not _overlap((x, y, 120, 300), SUBTITLE_SAFE_RECT)


def test_no_room_raises():
    with pytest.raises(ValueError, match="no room"):
        place_name_tag((10, 10, 1900, 1060), (120, 300), CANVAS)
    with pytest.raises(ValueError, match="no room"):
        place_name_tag((900, 300, 200, 300), (120, 300), CANVAS, side="right", gap=900)
    # the avoid strip leaves no legal position on the only allowed side
    with pytest.raises(ValueError, match="avoids"):
        place_name_tag(
            (100, 100, 300, 700),
            (120, 900),
            (1920, 1000),
            side="right",
            avoid=[(0, 0, 1920, 1000)],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"side": "diagonal"},
        {"align": "middle"},
        {"gap": -1},
        {"margin": -1},
        {"avoid": [(0, 0, 0, 5)]},
    ],
)
def test_geometry_validation(kwargs):
    with pytest.raises(ValueError):
        place_name_tag((600, 300, 400, 400), (100, 200), CANVAS, **kwargs)


def test_geometry_rejects_bad_boxes():
    with pytest.raises(ValueError):
        place_name_tag((0, 0, -5, 10), (100, 200), CANVAS)
    with pytest.raises(ValueError):
        place_name_tag((0, 0, float("nan"), 10), (100, 200), CANVAS)
    with pytest.raises(ValueError):
        place_name_tag((0, 0, 10, 10), (0, 200), CANVAS)


def test_geometry_works_with_cjk_free_numbers_only():
    # pure numbers: no font, no image involved
    assert place_name_tag((1200, 200, 400, 600), (120, 300), CANVAS) == (
        1056,
        200,
        "left",
    )


# --------------------------------------------------------------------------- #
# NameTag spec.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"name": "   "},
        {"name": "A", "start": -1},
        {"name": "A", "start": float("inf")},
        {"name": "A", "duration": 0},
        {"name": "A", "duration": float("nan")},
        {"name": "A", "side": "up"},
        {"name": "A", "orientation": "diagonal"},
        {"name": "A", "subject_box": (0, 0, 0, 10)},
        {"name": "A", "subject_box": (1, 2, 3)},
        {"name": "A", "slide_px": -3},
        {"name": "A", "seal": "abc"},
        {"name": "A", "panel_color": (0, 0, 300)},
        {"name": "A", "panel_alpha": 1.5},
    ],
)
def test_nametag_validation(kwargs):
    with pytest.raises(ValueError):
        NameTag(**kwargs)


def test_dict_round_trip_through_json():
    tag = NameTag(
        "嬌娜",
        role="狐仙",
        subject_box=(1200, 200, 400, 600),
        start=2.5,
        duration=3.0,
        side="left",
        seal="誌",
        leader=True,
        name_color=(1, 2, 3),
    )
    data = json.loads(json.dumps(tag.to_dict(), ensure_ascii=False))
    assert NameTag.from_dict(data) == tag
    assert NameTag.from_dict({"name": "X"}).role is None
    with pytest.raises(ValueError, match="unknown"):
        NameTag.from_dict({"name": "X", "colour": 1})
    with pytest.raises(ValueError):
        NameTag.from_dict(["X"])


# --------------------------------------------------------------------------- #
# Image.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_image_has_panel_brackets_and_text(orientation):
    tag = NameTag("Jiaona", role="Fox", orientation=orientation, seal="S")
    img = name_tag_image(tag, scale=1.0)
    arr = np.asarray(img)
    assert arr.shape[2] == 4
    alpha = arr[..., 3]
    # panel is translucent but mostly opaque; the whole rectangle is covered
    assert np.count_nonzero(alpha >= 200) > 0.6 * alpha.size
    assert np.all(alpha[2:-2, 2:-2] > 0)
    # corner brackets: gold pixels exist near the top-left and bottom-right
    gold = (np.abs(arr[..., :3].astype(int) - tag.bracket_color).sum(-1) < 12) & (
        alpha > 250
    )
    h, w = alpha.shape
    assert gold[: h // 3, : w // 3].any() and gold[-h // 3 :, -w // 3 :].any()
    assert not gold[: h // 3, -w // 3 :].any() and not gold[-h // 3 :, : w // 3].any()
    # name colour appears (text is drawn)
    name = np.abs(arr[..., :3].astype(int) - tag.name_color).sum(-1) < 30
    assert (name & (alpha > 200)).sum() > 20


def test_image_background_outside_panel_is_transparent_in_layer():
    comp = Composition(size=(400, 300), fps=10, duration=1, transparent=True)
    tag = NameTag("Jiaona", subject_box=(10, 10, 40, 40), duration=1, fade_in=0.1)
    # canvas is small: cards scale with the canvas, rest of the frame stays empty
    comp.add_layer(name_tag_layer(tag, (400, 300), avoid=[])[-1])
    frame = comp.get_frame(0.5)
    mask = comp.mask.get_frame(0.5)
    assert mask[:, -20:].max() == 0
    assert mask.max() > 0.7
    assert frame.shape == (300, 400, 3)


def test_scale_changes_size_and_font_shrinks_to_fit():
    tag = NameTag("Jiaona", role="Fox")
    big = name_tag_image(tag, scale=1.0)
    small = name_tag_image(tag, scale=0.5)
    assert small.width < big.width and small.height < big.height
    long_tag = NameTag("A" * 14, role="Fox")
    natural = name_tag_image(long_tag, max_size=(10_000, 10_000))
    fitted = name_tag_image(long_tag, max_size=(10_000, 600))
    assert fitted.height <= 600 < natural.height
    with pytest.raises(ValueError, match="needs"):
        name_tag_image(long_tag, max_size=(10_000, 50))


def test_optional_parts_change_size():
    base = name_tag_image(NameTag("Jiaona"))
    with_role = name_tag_image(NameTag("Jiaona", role="Fox"))
    with_seal = name_tag_image(NameTag("Jiaona", seal="S"))
    assert with_role.width > base.width and with_seal.width > base.width


# --------------------------------------------------------------------------- #
# Layers.
# --------------------------------------------------------------------------- #


def _layer_pair(**kw):
    kw.setdefault("subject_box", (1250, 150, 420, 650))
    kw.setdefault("start", 1.0)
    kw.setdefault("duration", 4.0)
    return name_tag_layer(NameTag("Jiaona", role="Fox", **kw), CANVAS)


def test_opacity_envelope():
    (layer,) = _layer_pair()
    op = layer.transform.opacity_at
    assert op(0.0) == 0 and op(0.99) == 0
    assert op(3.0) == pytest.approx(1.0)
    assert op(5.0) == 0 and op(6.0) == 0
    assert 0 < op(1.2) < 1
    assert layer.in_point == 1.0 and layer.out_point == 5.0


def test_fades_squeezed_into_short_duration():
    (layer,) = _layer_pair(duration=0.5)
    assert layer.transform.opacity_at(1.0 + 0.45 * 0.5 / 1.1) == pytest.approx(1.0)
    assert layer.transform.opacity_at(1.5) == 0


@pytest.mark.parametrize(
    "side,axis,sign",
    [("left", 0, 1), ("right", 0, -1), ("above", 1, 1), ("below", 1, -1)],
)
def test_slide_moves_away_from_subject(side, axis, sign):
    box = (700, 560, 400, 100) if side == "above" else (700, 300, 400, 100)
    tag = NameTag("Jiaona", subject_box=box, side=side, start=0.0, duration=4.0)
    (layer,) = name_tag_layer(tag, CANVAS, avoid=[])
    pos = layer.transform.position.value_at
    start, rest = np.array(pos(0.0)), np.array(pos(2.0))
    delta = start - rest
    # the card begins closer to the subject, then drifts outward
    assert delta[axis] * sign > 0
    assert abs(delta[1 - axis]) < 1e-6
    assert abs(delta[axis]) <= 24.0 + 1e-6


def test_leader_layer_added_below_card():
    layers = _layer_pair(leader=True)
    assert [layer.name.split()[-1] for layer in layers] == ["leader", "tag"]


def test_layer_validation():
    with pytest.raises(ValueError, match="outside"):
        name_tag_layer(NameTag("A", subject_box=(1800, 100, 400, 400)), CANVAS)
    with pytest.raises(TypeError):
        name_tag_layer({"name": "A"}, CANVAS)
    with pytest.raises(ValueError, match="no room"):
        name_tag_layer(NameTag("A", subject_box=(10, 10, 1900, 1060)), CANVAS)


def _comp_with_subject(box, size=CANVAS, duration=6):
    comp = Composition(size=size, fps=10, duration=duration, bg_color=(0, 0, 0))
    w, h = size
    s = min(w / 1920, h / 1080)
    x, y, bw, bh = (int(v * s) for v in box)
    person = np.zeros((h, w, 3), np.uint8)
    person[y : y + bh, x : x + bw] = (120, 120, 125)
    comp.add_clip(ImageClip(person, duration=duration), name="person")
    return comp, (x, y, bw, bh)


def _card_pixels(frame, subject):
    """Boolean map of pixels that differ from the grey person and black bg."""
    x, y, w, h = subject
    pix = frame.astype(int)
    non_bg = pix.sum(-1) > 0
    person = np.zeros(non_bg.shape, bool)
    person[y : y + h, x : x + w] = True
    return non_bg & ~person


@pytest.mark.parametrize("size", [(1920, 1080), (960, 540)])
def test_rendered_card_sits_beside_subject(size):
    box = (1250, 150, 420, 650)
    comp, subject = _comp_with_subject(box, size)
    tag = NameTag("Jiaona", role="Fox", subject_box=box, start=0.0, duration=6.0)
    add_name_tags(comp, [tag])
    frame = comp.get_frame(3.0)
    # the subject box is untouched
    x, y, w, h = subject
    assert (frame[y : y + h, x : x + w] == (120, 120, 125)).all()
    card = _card_pixels(frame, subject)
    ys, xs = np.nonzero(card)
    assert xs.size > 500
    assert xs.max() < x  # entirely left of the subject
    assert xs.min() >= 48 * size[0] / 1920 - 1 and ys.min() >= 48 * size[1] / 1080 - 1
    # and it does not reach the subtitle safe rect
    sx, sy, sw, sh = (v * size[0] / 1920 for v in SUBTITLE_SAFE_RECT)
    assert not card[int(sy) : int(sy + sh), int(sx) : int(sx + sw)].any()


def test_card_absent_outside_its_time_window():
    box = (1250, 150, 420, 650)
    comp, subject = _comp_with_subject(box)
    add_name_tags(comp, [NameTag("Jiaona", subject_box=box, start=2.0, duration=2.0)])
    assert not _card_pixels(comp.get_frame(1.0), subject).any()
    assert _card_pixels(comp.get_frame(3.0), subject).any()
    assert not _card_pixels(comp.get_frame(4.5), subject).any()


def test_leader_renders_between_card_and_subject():
    box = (1250, 150, 420, 650)
    comp, subject = _comp_with_subject(box)
    tag = NameTag("Jiaona", subject_box=box, start=0.0, duration=6.0, leader=True)
    add_name_tags(comp, [tag])
    card = _card_pixels(comp.get_frame(3.0), subject)
    ys, xs = np.nonzero(card)
    assert xs.max() == 1250 - 1 or xs.max() >= 1250 - 6  # leader reaches the subject


# --------------------------------------------------------------------------- #
# add_name_tags.
# --------------------------------------------------------------------------- #


def test_add_name_tags_rejects_time_and_space_overlap():
    comp, _ = _comp_with_subject((1250, 150, 420, 650))
    a = NameTag("Jiaona", subject_box=(1250, 150, 420, 650), start=0, duration=4)
    b = NameTag("Ningzi", subject_box=(1250, 150, 420, 650), start=3, duration=4)
    with pytest.raises(ValueError, match="Jiaona.*Ningzi.*overlap"):
        add_name_tags(comp, [a, b])
    assert len(comp.layers) == 1  # nothing was added


def test_add_name_tags_allows_staggered_or_separate_tags():
    comp, _ = _comp_with_subject((1250, 150, 420, 650))
    a = NameTag("Jiaona", subject_box=(1250, 150, 420, 650), start=0, duration=4)
    later = NameTag("Ningzi", subject_box=(1250, 150, 420, 650), start=4, duration=2)
    apart = NameTag("Sage", subject_box=(150, 150, 420, 650), start=0, duration=4)
    layers = add_name_tags(comp, [a, later, apart])
    assert len(layers) == 3
    assert len(comp.layers) == 4
    # dict configs are accepted too
    comp2, _ = _comp_with_subject((1250, 150, 420, 650))
    add_name_tags(comp2, [{"name": "Jiaona", "subject_box": [1250, 150, 420, 650]}])
    assert len(comp2.layers) == 2


def test_add_name_tags_rejects_unknown_keywords():
    comp, _ = _comp_with_subject((1250, 150, 420, 650))
    with pytest.raises(TypeError):
        add_name_tags(comp, [], colour=1)


def test_card_shrinks_to_fit_a_narrow_gap():
    # 160 px between the subject and the right margin: too narrow for the
    # full card, wide enough once it is shrunk.
    tag = NameTag("AB", role="C", subject_box=(1500, 200, 212, 500), side="right")
    full = name_tag_image(tag, scale=1.0)
    assert full.width + 24 > 1920 - 48 - 1712
    layers = name_tag_layer(tag, CANVAS, avoid=[])
    card = layers[-1]
    assert card.source_size[0] < full.width
    assert card.source_size[0] <= 1920 - 48 - 1712 - 24


def test_vertical_card_falls_back_to_horizontal():
    # Only a short strip above the subject is free: a vertical card can't fit.
    tag = NameTag("ABCD", role="EF", subject_box=(100, 260, 1720, 500), side="above")
    layers = name_tag_layer(tag, CANVAS, avoid=[])
    w, h = layers[-1].source_size
    assert w > h and h <= 260 - 48 - 24
