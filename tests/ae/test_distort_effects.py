"""Distort effects: identity, analytic pixel checks and invariants."""

import numpy as np

import pytest

from moviepy.ae import Buffer
from moviepy.ae.effects.distort.mirror import Mirror
from moviepy.ae.effects.distort.spherize import Spherize
from moviepy.ae.effects.distort.twirl import Twirl
from moviepy.ae.effects.distort.wave_warp import WaveWarp


def buffer(rgba, offset=(0, 0), color_space="srgb"):
    """Premultiplied buffer from straight-color-premultiplied RGBA array."""
    return Buffer(np.asarray(rgba, dtype=np.float32), offset, color_space)


def random_rgba(width, height, seed=0):
    """Deterministic premultiplied RGBA with semitransparent pixels."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.2, 1.0, size=(height, width, 1)).astype(np.float32)
    color = rng.uniform(0.0, 1.0, size=(height, width, 3)).astype(np.float32)
    return np.concatenate([color * alpha, alpha], axis=-1)


def radial_rgba(size=33):
    """Opaque grey disc whose value depends only on distance from the center."""
    c = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(x - c, y - c)
    value = 0.5 + 0.3 * np.cos(r / 3.0)
    rgba = np.empty((size, size, 4), dtype=np.float32)
    rgba[..., :3] = value[..., None]
    rgba[..., 3] = 1.0
    return rgba


def all_effects():
    """Non-identity configurations of every Distort effect."""
    return [
        WaveWarp(wave_height=12, wave_width=20, wave_speed=0.7, direction=30),
        Twirl(angle=90, twirl_radius=60, center_x=0.4),
        Mirror(reflection_angle=25, reflection_center_x=0.6),
        Spherize(radius=12, amount=-40),
    ]


def test_defaults_and_zero_settings_are_identity():
    src = buffer(random_rgba(12, 10))
    assert WaveWarp().process(src, 0.3) is src
    assert Twirl().process(src, 0.3) is src
    assert Twirl(angle=0, twirl_radius=100).process(src, 0.3) is src
    assert Spherize().process(src, 0.3) is src
    assert Spherize(radius=50, amount=0).process(src, 0.3) is src


def test_twirl_full_turn_on_radially_symmetric_image_is_unchanged():
    rgba = radial_rgba(33)
    out = Twirl(angle=360, twirl_radius=75).process(buffer(rgba), 0.0)
    assert out.size == (33, 33)
    assert np.allclose(out.rgba, rgba, atol=0.02)
    # A genuine rotation is not a no-op for an asymmetric image.
    asym = radial_rgba(33)
    asym[:, :16, :3] = 0.0
    turned = Twirl(angle=90, twirl_radius=75).process(buffer(asym), 0.0)
    assert not np.allclose(turned.rgba, asym, atol=0.05)


def test_wave_warp_shifts_vertical_edge_as_a_wave():
    width = height = 40
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[:, 20:, :3] = 1.0
    rgba[..., 3] = 1.0
    amplitude = 5.0  # wave_height 10 is peak-to-peak
    effect = WaveWarp(wave_height=10, wave_width=40, wave_speed=0, direction=0, phase=0)
    out = effect.process(buffer(rgba), 0.0)
    # Margin of wave_height on each side; original rows are 10..49.
    assert out.size == (60, 60)
    red = out.rgba[..., 0]
    rows = np.arange(10, 50)
    # The stripe's total area is conserved, so measure its left edge: the
    # black coverage left of the stripe equals 30 + displacement (x >= 30 + D).
    left_black = (1.0 - red[rows, :40]).sum(axis=1)
    original_y = rows - 10 + 0.5
    expected = 30.0 + amplitude * np.sin(2 * np.pi * original_y / 40.0)
    assert np.allclose(left_black, expected, atol=1e-4)
    assert np.ptp(left_black) > 5.0


def test_wave_warp_animates_with_time():
    rgba = random_rgba(24, 24, seed=3)
    effect = WaveWarp(wave_height=6, wave_width=16, wave_speed=1, direction=90)
    early = effect.process(buffer(rgba), 0.0)
    later = effect.process(buffer(rgba), 0.25)
    assert not np.allclose(early.rgba, later.rgba)


def test_mirror_default_reflects_bottom_half_about_center_line():
    src = buffer(random_rgba(16, 16, seed=1))
    out = Mirror().process(src, 0.0)
    assert out.size == (16, 16)
    assert np.allclose(out.rgba[:8], src.rgba[:8], atol=1e-6)
    assert np.allclose(out.rgba[8:], src.rgba[7::-1], atol=1e-6)
    assert np.allclose(out.rgba[8:], out.rgba[7::-1], atol=1e-6)


def test_mirror_vertical_line_is_symmetric_left_right():
    src = buffer(random_rgba(16, 16, seed=2))
    out = Mirror(reflection_angle=90).process(src, 0.0)
    assert np.allclose(out.rgba[:, :8], src.rgba[:, 15:7:-1], atol=1e-6)
    assert np.allclose(out.rgba[:, 8:], src.rgba[:, 8:], atol=1e-6)
    assert np.allclose(out.rgba, out.rgba[:, ::-1], atol=1e-6)


def test_opaque_input_stays_opaque_for_non_growing_effects():
    rgba = radial_rgba(33)
    for effect in (
        Twirl(angle=180, twirl_radius=75),
        Mirror(reflection_angle=90),
        Spherize(radius=20, amount=60),
    ):
        out = effect.process(buffer(rgba), 0.0)
        assert out.size == (33, 33), effect.name
        assert np.allclose(out.rgba[..., 3], 1.0, atol=1e-6), effect.name


def test_alpha_is_mirrored_with_color():
    src = buffer(random_rgba(16, 16, seed=5))
    out = Mirror().process(src, 0.0)
    assert np.allclose(out.rgba[8:, :, 3], src.rgba[7::-1, :, 3], atol=1e-6)


@pytest.mark.parametrize("effect", all_effects(), ids=lambda e: type(e).__name__)
def test_output_is_premultiplied_and_deterministic(effect):
    src = buffer(random_rgba(24, 20, seed=7))
    first = effect.process(src, 0.4)
    second = effect.process(src, 0.4)
    assert np.array_equal(first.rgba, second.rgba)
    rgb, alpha = first.rgba[..., :3], first.rgba[..., 3:]
    assert np.all(rgb <= alpha + 1e-6)
    assert np.all(rgb >= -1e-6)
    assert np.all((alpha >= -1e-6) & (alpha <= 1 + 1e-6))


@pytest.mark.parametrize("effect", all_effects(), ids=lambda e: type(e).__name__)
def test_empty_buffer_and_color_space_are_kept(effect):
    empty = buffer(np.zeros((0, 4, 4), dtype=np.float32), offset=(3, 4))
    assert effect.process(empty, 0.0) is empty
    linear = buffer(random_rgba(10, 8), color_space="linear")
    assert effect.process(linear, 0.0).color_space == "linear"


def test_spherize_bulge_magnifies_center_and_keeps_rim_outside():
    rgba = np.zeros((33, 33, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    rgba[..., :3] = np.linspace(0, 1, 33, dtype=np.float32)[None, :, None]
    out = Spherize(radius=16, amount=100).process(buffer(rgba), 0.0)
    # Outside the circle nothing changes.
    assert np.allclose(out.rgba[0, :], rgba[0, :], atol=1e-6)
    # Bulge magnifies the center: the ramp's slope there is shallower than
    # the input's (about 2/pi of it at the center).
    center_row = out.rgba[16, :, 0]
    input_step = rgba[16, 17, 0] - rgba[16, 16, 0]
    output_step = center_row[17] - center_row[16]
    assert 0.0 < output_step < 0.8 * input_step
