"""Tests for the closing like / subscribe / share carousel card."""

import json

import numpy as np

import pytest

from moviepy.ae import Composition
from moviepy.ae.templates.end_card import (
    EndCard,
    add_end_card,
    end_card_layers,
    icon_image,
)


ACCENT = (221, 55, 55)
TEXT = (239, 220, 184)


def _comp(size=(960, 540), duration=20):
    return Composition(size=size, fps=10, duration=duration, bg_color=(0, 0, 0))


def _card(**kw):
    kw.setdefault("start", 10.0)
    kw.setdefault("duration", 6.0)
    kw.setdefault("channel", "Night Lamp")
    kw.setdefault("line", "Subscribe for more")
    kw.setdefault("actions", ("Like", "Subscribe", "Share"))
    return EndCard(**kw)


def _opacity(layer, t):
    return layer.transform.opacity.value_at(layer.source_time(t))


# --------------------------------------------------------------------------- #
# Spec and carousel timing.
# --------------------------------------------------------------------------- #


def test_defaults_follow_the_channel_brief():
    card = EndCard()
    assert card.channel == "夜燈說書" and card.actions == ("按讚", "訂閱", "分享")
    assert card.period == 1.6 and card.fade == 0.5
    assert card.panel_color == (29, 23, 19)
    assert round(card.panel_alpha * 255) == 185
    assert card.text_color == (239, 220, 184) and card.accent_color == ACCENT


def test_round_trip_through_json():
    card = _card(headline="Ep. 5", question="Who decides?", period=1.2)
    data = json.loads(json.dumps(card.to_dict()))
    assert EndCard.from_dict(data) == card
    with pytest.raises(ValueError, match="unknown end card keys"):
        EndCard.from_dict({"bogus": 1})
    with pytest.raises(ValueError, match="twice the fade"):
        EndCard(duration=0.9)
    with pytest.raises(ValueError, match="1 to 5"):
        EndCard(actions=())
    with pytest.raises(ValueError, match="icons"):
        EndCard(icons=("star",))


def test_active_index_cycles_through_actions_over_time():
    card = _card(duration=10.0)  # windows start at 0.5 + k * 1.6
    probe = {0.0: None, 0.4: None, 0.6: 0, 2.0: 0, 2.2: 1, 3.8: 2, 5.4: 0, 7.0: 1}
    for t, expected in probe.items():
        assert card.active_index(card.start + t) == expected, t
    assert card.active_index(card.start + 9.6) is None  # fading out
    assert card.active_index(card.start - 1.0) is None
    assert [i for i, _, _ in card.windows()][:5] == [0, 1, 2, 0, 1]


# --------------------------------------------------------------------------- #
# Layers: carousel gating on the layer clock.
# --------------------------------------------------------------------------- #


def test_layer_structure_and_clock():
    comp = _comp()
    card = _card()
    layers = end_card_layers(comp, card)
    assert len(layers) == 1 + 2 * 3 and len(comp.layers) == 7
    for layer in layers:
        assert (layer.in_point, layer.out_point) == (10.0, 16.0)
    panel = layers[0]
    assert _opacity(panel, 10.0) == 0.0  # late card is faded from zero...
    assert _opacity(panel, 10.25) == pytest.approx(50.0)
    assert _opacity(panel, 12.0) == 100.0  # ...and visible mid-card
    assert comp.get_frame(5.0).max() == 0 and comp.get_frame(13.0).max() > 0


def test_exactly_one_action_is_active_at_a_time():
    comp = _comp()
    card = _card()
    layers = end_card_layers(comp, card)
    dims, actives = layers[1::2], layers[2::2]
    assert "dim" in dims[0].name and "active" in actives[0].name
    for t in (10.6, 11.9, 12.4, 13.5, 14.9, 15.2):
        index = card.active_index(t)
        assert index is not None
        for i in range(3):
            assert _opacity(actives[i], t) == (100.0 if i == index else 0.0), (t, i)
            assert _opacity(dims[i], t) == (0.0 if i == index else 100.0), (t, i)
    # during the fade-in nothing is highlighted and every button is dim
    for i in range(3):
        assert _opacity(actives[i], 10.2) == 0.0
        assert _opacity(dims[i], 10.25) == pytest.approx(50.0)


def test_highlight_pops_with_scale_keys_at_each_window():
    comp = _comp(duration=30)
    card = _card(duration=12.0)  # action 0 is active at 0.5 and 5.3 (layer time)
    active0 = end_card_layers(comp, card)[2]
    scale = active0.transform.scale
    for begin in (0.5, 5.3):
        t = lambda frame: scale.value_at(begin + 1e-6 + frame / 30.0)  # noqa: E731
        assert t(0)[0] == pytest.approx(80.0, abs=0.2)
        assert t(6)[0] == pytest.approx(106.0, abs=0.2)
        assert t(10)[0] == pytest.approx(100.0, abs=0.2)
        assert t(20)[0] == pytest.approx(100.0, abs=0.2)
        assert 80.0 < t(3)[0] < 106.0
    keys = [(round(k.time, 4), k.value[0]) for k in scale.keyframes]
    assert keys[:3] == [(0.5, 80.0), (0.7, 106.0), (0.8333, 100.0)]
    # the anchor is the button centre so the pop grows from the middle
    assert active0.transform.anchor_point.value_at(0.0)[0] > 0


def test_panel_stays_above_the_subtitle_safe_rect():
    for size in ((1920, 1080), (1280, 720), (640, 480)):
        comp = _comp(size)
        panel = end_card_layers(comp, _card())[0]
        s = min(size[0] / 1920, size[1] / 1080)
        x, y = panel.transform.position.value_at(0.0)
        height = panel.clip.size[1]
        assert y + height <= 800 * s + 1
        assert 0 <= x and x + panel.clip.size[0] <= size[0]


# --------------------------------------------------------------------------- #
# Rasters.
# --------------------------------------------------------------------------- #


def test_active_state_is_accent_and_dim_state_is_translucent_text_colour():
    layers = end_card_layers(_comp((1920, 1080)), _card())
    dim, active = layers[1], layers[2]

    def solid(layer, threshold):
        rgb = np.asarray(layer.clip.get_frame(0))
        mask = np.asarray(layer.clip.mask.get_frame(0))
        return rgb[mask >= threshold], mask

    rgb, mask = solid(active, 0.99)
    assert (rgb == ACCENT).all(axis=1).any() and mask.max() == pytest.approx(1.0)
    rgb, mask = solid(dim, 0.2)
    assert (np.abs(rgb.astype(int) - TEXT).max(axis=1) <= 2).all()
    assert 0.3 < mask.max() < 0.5  # dimmed
    # each state is rasterised once and reused for every frame
    assert active.source_buffer(0.0) is active.source_buffer(1.0)


def test_icons_are_anti_aliased_polygons_with_a_cutout():
    for name in ("like", "subscribe", "share"):
        img = icon_image(name, 64, ACCENT)
        alpha = np.asarray(img)[..., 3]
        assert alpha.max() == 255 and 0 < (alpha > 0).mean() < 0.9
        assert ((alpha > 0) & (alpha < 255)).any()  # soft edges
    pill = icon_image("subscribe", 100, ACCENT)
    assert pill.getpixel((50, 50))[3] == 0  # play triangle is knocked out
    assert pill.getpixel((12, 50))[3] == 255
    assert icon_image("share", 32, ACCENT, 0.5).getpixel((24, 15))[3] in range(0, 129)
    with pytest.raises(ValueError, match="icon"):
        icon_image("star", 32, ACCENT)


def test_rendered_frame_shows_panel_colour():
    comp = _comp((480, 270))
    end_card_layers(comp, _card())
    frame = comp.get_frame(13.0)
    # inside the panel margin, far from text: panel colour * 185/255 over black
    x, y = int(360 * 0.25) + 6, int(60 * 0.25) + 6
    expected = np.array((29, 23, 19)) * 185 / 255
    assert np.abs(frame[y, x].astype(float) - expected).max() <= 3


# --------------------------------------------------------------------------- #
# Fit, add_end_card.
# --------------------------------------------------------------------------- #


def test_text_that_cannot_fit_raises_value_error():
    with pytest.raises(ValueError, match="does not fit"):
        end_card_layers(_comp(), _card(channel="W" * 300))
    with pytest.raises(ValueError, match="does not fit"):
        end_card_layers(_comp(), _card(actions=("X" * 80, "b", "c")))


def test_add_end_card_accepts_dict_and_optional_rows():
    comp = _comp()
    layers = add_end_card(comp, _card(headline="Ep. 5", question="Who?").to_dict())
    assert len(layers) == 7 and len(comp.layers) == 7
    one = add_end_card(_comp(), _card(actions=("Like",)))
    assert len(one) == 3
    with pytest.raises(TypeError):
        end_card_layers(_comp(), {"start": 0})
