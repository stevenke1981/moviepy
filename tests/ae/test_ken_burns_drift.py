"""Drift moves: pan and push in together, settle on the focus, hold, continue."""

import cv2
import numpy as np

import pytest

from moviepy.ae.templates.ken_burns import (
    KEN_BURNS_DEFAULTS,
    MOVES,
    check_motion,
    ken_burns,
)
from moviepy.ae.templates.presets import ChannelPreset


SIZE = (160, 90)
PRESET = ChannelPreset(name="drift-test", size=SIZE, fps=24, safe_margin=(8, 6))
DURATION = 3.0
HOLD = 0.5


def _texture(seed=5):
    width, height = SIZE
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 256, (height // 5, width // 5, 3), dtype=np.uint8)
    big = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(big, 40, 255).astype(np.uint8)


def _comp(move="drift-right", **kw):
    kw.setdefault("hold", HOLD)
    return ken_burns(_texture(), PRESET, duration=DURATION, move=move, **kw)


def test_defaults_and_auto_unchanged():
    assert "drift-left" in MOVES and "drift-right" in MOVES
    assert KEN_BURNS_DEFAULTS["drift_zoom"] == (1.02, 1.10)
    assert KEN_BURNS_DEFAULTS["drift_distance"] == 0.035
    assert KEN_BURNS_DEFAULTS["auto_cycle"] == (
        "push",
        "pan-right",
        "push",
        "pan-left",
        "pull",
    )


@pytest.mark.parametrize("move", ["drift-left", "drift-right"])
def test_no_border_every_frame(move):
    comp = _comp(move, focus=(0.9, 0.5))
    for k in range(int(DURATION * 24)):
        rgba = np.asarray(comp.render_buffer(k / 24).rgba)
        assert rgba[..., :3].min() > 0.05


@pytest.mark.parametrize("move,sign", [("drift-right", 1), ("drift-left", -1)])
def test_zoom_grows_centre_arrives_on_focus_and_holds(move, sign):
    comp = _comp(move, focus=(0.5, 0.5))
    framing = comp.framing
    first = framing.sample(0.0)
    end = framing.sample(DURATION - HOLD)
    final = framing.sample(DURATION - 0.01)
    assert end.scale > first.scale
    assert end.scale / first.scale == pytest.approx(1.10 / 1.02, rel=0.01)
    assert end.center == pytest.approx(final.center)
    assert final.scale == pytest.approx(end.scale)
    # window starts displaced against the direction and ends near the middle
    shift = end.center[0] - first.center[0]
    assert shift * sign > 0
    assert end.center[0] == pytest.approx((SIZE[0] - 1) / 2, abs=1.0)


def test_zoom_and_distance_overrides():
    comp = _comp(zoom=(1.3, 1.6), distance=0.1, focus=(0.5, 0.5))
    a, b = comp.framing.sample(0.0), comp.framing.sample(DURATION - HOLD)
    assert b.scale / a.scale == pytest.approx(1.6 / 1.3, rel=0.01)
    wide = b.center[0] - a.center[0]
    narrow_comp = _comp(zoom=(1.3, 1.6), distance=0.02, focus=(0.5, 0.5))
    n0, n1 = (narrow_comp.framing.sample(t) for t in (0.0, DURATION - HOLD))
    assert wide > n1.center[0] - n0.center[0] > 0


def test_focus_respected():
    comp = _comp(focus=(0.7, 0.3))
    end = comp.framing.sample(DURATION - HOLD)
    mid_x, mid_y = (SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2
    assert end.center[0] > mid_x + 5  # clamped by the window, but toward focus
    assert end.center[1] < mid_y
    other = _comp(focus=(0.3, 0.3)).framing.sample(DURATION - HOLD)
    assert other.center[0] == pytest.approx(2 * mid_x - end.center[0], abs=0.5)


def test_segments_continue_one_move():
    whole = _comp(focus=(0.5, 0.5))
    first = _comp(segment=(0, 0.5), focus=(0.5, 0.5))
    second = _comp(segment=(0.5, 1), focus=(0.5, 0.5))
    t_end = DURATION - HOLD
    f_end, s_start = first.framing.sample(t_end), second.framing.sample(0.0)
    assert f_end.center == pytest.approx(s_start.center, abs=1e-6)
    assert f_end.scale == pytest.approx(s_start.scale, abs=1e-9)
    mid = whole.framing.sample(t_end / 2)
    assert f_end.center == pytest.approx(mid.center, abs=0.05)
    assert second.framing.sample(t_end).center == pytest.approx(
        whole.framing.sample(t_end).center, abs=1e-6
    )


def test_segment_validation():
    with pytest.raises(ValueError):
        _comp(segment=(0.6, 0.4))
    with pytest.raises(ValueError):
        _comp("push", segment=(0, 0.5))
    with pytest.raises(ValueError):
        _comp(distance=0.9)


@pytest.mark.parametrize("move", ["drift-left", "drift-right"])
def test_drift_is_smooth(move):
    comp = _comp(move, focus=(0.5, 0.5))
    report = check_motion(comp, 0.0, DURATION - HOLD)
    assert report["verdict"] in ("SMOOTH", "STATIC")
    assert report["max_jerk"] < 3.0


def test_hold_phase_is_static():
    comp = _comp(focus=(0.5, 0.5))
    report = check_motion(comp, DURATION - HOLD + 0.1, HOLD - 0.15)
    assert report["median"] < 0.05
