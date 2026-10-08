"""Sub-pixel Ken Burns template: framing, direction, hold and motion verdicts."""

import cv2
import numpy as np
import pytest
from PIL import Image

from moviepy import ImageClip
from moviepy.ae.templates.ken_burns import (
    KEN_BURNS_DEFAULTS,
    MOVES,
    check_motion,
    ken_burns,
    motion_smoothness,
)
from moviepy.ae.templates.presets import ChannelPreset

SIZE = (160, 90)
PRESET = ChannelPreset(name="ken-burns-test", size=SIZE, fps=24, safe_margin=(8, 6))
DURATION = 3.0
LAST = DURATION - 1 / 24  # final rendered frame; t == duration is past the end


def _texture(size=SIZE, seed=3):
    """Smooth random blobs with no dark pixels, so a black border is detectable."""
    width, height = size
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 256, (height // 5, width // 5, 3), dtype=np.uint8)
    big = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(big, 40, 255).astype(np.uint8)


def _ramp(axis):
    """Brightness rising left-to-right (axis=1) or top-to-bottom (axis=0)."""
    width, height = SIZE
    if axis == 1:
        line = np.linspace(40, 250, width)
        plane = np.broadcast_to(line[None, :], (height, width))
    else:
        line = np.linspace(40, 250, height)
        plane = np.broadcast_to(line[:, None], (height, width))
    plane = np.rint(plane).astype(np.uint8)
    return np.repeat(plane[..., None], 3, axis=2)


def _render(comp, t):
    return np.asarray(comp.render_buffer(t).rgba)


@pytest.mark.parametrize("move", [*MOVES, "auto"])
def test_every_move_renders_size_and_no_border(move):
    comp = ken_burns(_texture(), PRESET, duration=DURATION, move=move, hold=0.5)
    assert comp.size == SIZE
    for t in (0.0, DURATION / 2, LAST):
        rgba = _render(comp, t)
        assert rgba.shape == (SIZE[1], SIZE[0], 4)
        np.testing.assert_allclose(rgba[..., 3], 1.0, atol=1e-6)
        assert rgba[..., :3].min() > 0.05  # a border would render as black


def test_pull_starts_zoomed_and_ends_at_cover():
    comp = ken_burns(_texture(), PRESET, duration=DURATION, move="pull", hold=0.5)
    start = comp.framing.sample(0).scale
    end = comp.framing.sample(DURATION).scale
    assert start / end == pytest.approx(1.08, rel=1e-3)


def test_push_zooms_in_on_the_centre():
    comp = ken_burns(_ramp(1), PRESET, duration=DURATION, move="push", hold=0.5)
    start, end = comp.framing.sample(0), comp.framing.sample(DURATION)
    crop_w0 = start.crop[2] - start.crop[0]
    crop_w1 = end.crop[2] - end.crop[0]
    assert crop_w0 / crop_w1 == pytest.approx(1.08, rel=1e-3)
    # The visible left edge moves inward, so it gets brighter as the ramp rises.
    left0 = _render(comp, 0)[:, 0, 0].mean()
    left1 = _render(comp, LAST)[:, 0, 0].mean()
    assert left1 > left0


@pytest.mark.parametrize(
    "move, ramp_axis, sign",
    [
        ("pan-right", 1, +1),
        ("pan-left", 1, -1),
        ("pan-down", 0, +1),
        ("pan-up", 0, -1),
    ],
)
def test_pan_direction_sign_and_full_distance(move, ramp_axis, sign):
    comp = ken_burns(_ramp(ramp_axis), PRESET, duration=DURATION, move=move, hold=0.5)
    start, end = _render(comp, 0), _render(comp, LAST)
    if ramp_axis == 1:
        edge0, edge1 = start[:, 0, 0].mean(), end[:, 0, 0].mean()
    else:
        edge0, edge1 = start[0, :, 0].mean(), end[0, :, 0].mean()
    assert np.sign(edge1 - edge0) == sign
    # Travel is distance times output width, measured in output pixels.
    rig = comp.framing
    a, b = rig.sample(0), rig.sample(DURATION)
    axis = 0 if move in ("pan-left", "pan-right") else 1  # centre index: x, y
    travelled = abs(b.center[axis] - a.center[axis]) * a.scale
    expected = KEN_BURNS_DEFAULTS["pan_distance"] * SIZE[0]
    assert travelled == pytest.approx(expected, rel=1e-3)


def test_hold_keeps_final_framing_identical():
    comp = ken_burns(_texture(), PRESET, duration=DURATION, move="push", hold=1.0)
    reference = _render(comp, 2.5)
    for t in (2.0, 2.4, 2.99):
        np.testing.assert_array_equal(_render(comp, t), reference)
    assert not np.array_equal(_render(comp, 1.0), reference)


def test_default_hold_comes_from_preset():
    comp = ken_burns(_texture(), PRESET, duration=DURATION, move="push")
    reference = _render(comp, 1.5 + 1e-6)  # preset.hold is 1.5 s
    np.testing.assert_array_equal(_render(comp, 2.9), reference)


def test_lead_keeps_starting_framing_identical():
    comp = ken_burns(_texture(), PRESET, duration=DURATION, move="push", lead=0.5)
    reference = _render(comp, 0.0)
    np.testing.assert_array_equal(_render(comp, 0.4), reference)
    assert not np.array_equal(_render(comp, 1.2), reference)


def test_push_motion_is_smooth_for_three_seconds():
    comp = ken_burns(_texture(), PRESET, duration=DURATION, move="push", hold=0.0)
    result = check_motion(comp, 0, DURATION)
    assert result["frames"] == 72
    assert result["verdict"] == "SMOOTH"
    assert result["max_jerk"] < 1.5  # 1920-wide px; integer steps reach 12
    assert result["still_ratio"] < 0.2


def test_integer_step_sequence_is_jitter():
    wide = _texture((SIZE[0] + 40, SIZE[1]))
    frames = [wide[:, int(0.4 * i) : int(0.4 * i) + SIZE[0]] for i in range(72)]
    result = motion_smoothness(frames)
    assert result["verdict"] == "JITTER"
    assert result["still_ratio"] >= 0.2


def test_subpixel_steps_of_same_speed_are_smooth():
    base = _texture()
    frames = []
    for i in range(72):
        matrix = np.float32([[1, 0, 0.4 * i], [0, 1, 0]])
        frames.append(
            cv2.warpAffine(
                base,
                matrix,
                SIZE,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    assert motion_smoothness(frames)["verdict"] == "SMOOTH"


def test_static_sequence_is_static():
    comp = ken_burns(_texture(), PRESET, duration=DURATION, move="push", zoom=(1, 1))
    assert check_motion(comp, 0, DURATION)["verdict"] == "STATIC"


def test_render_is_deterministic():
    first = ken_burns(_texture(), PRESET, duration=DURATION, move="pan-left")
    second = ken_burns(_texture(), PRESET, duration=DURATION, move="pan-left")
    for t in (0.3, 1.7):
        np.testing.assert_array_equal(_render(first, t), _render(second, t))


def test_path_array_and_clip_inputs_agree(tmp_path):
    texture = _texture()
    path = tmp_path / "texture.png"
    Image.fromarray(texture).save(path)
    renders = [
        _render(ken_burns(source, PRESET, duration=2, move="push", hold=0.5), 1.0)
        for source in (path, texture, ImageClip(texture))
    ]
    np.testing.assert_array_equal(renders[0], renders[1])
    np.testing.assert_array_equal(renders[1], renders[2])


def test_auto_cycles_through_moves_by_index():
    cycle = KEN_BURNS_DEFAULTS["auto_cycle"]
    texture = _texture()
    for index in range(len(cycle) + 1):
        auto = ken_burns(texture, PRESET, duration=2, move="auto", index=index)
        explicit = ken_burns(
            texture, PRESET, duration=2, move=cycle[index % len(cycle)]
        )
        np.testing.assert_array_equal(_render(auto, 0.7), _render(explicit, 0.7))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"move": "spin"},
        {"move": "push", "hold": DURATION},
        {"move": "push", "distance": 0.1},
        {"move": "pan-right", "zoom": 1.0, "distance": 0.5},
        {"move": "pan-right", "zoom": (1.1, 1.2)},
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        ken_burns(_texture(), PRESET, duration=DURATION, **kwargs)


def test_transparent_image_is_rejected():
    rgba = np.dstack([_texture(), np.full(SIZE[::-1], 128, np.uint8)])
    with pytest.raises(ValueError, match="opaque"):
        ken_burns(rgba, PRESET, duration=DURATION)
