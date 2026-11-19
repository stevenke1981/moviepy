"""Tests for Posterize, Mosaic, Vignette and Noise (Add Grain) effects."""

import numpy as np

import pytest

from moviepy.ae import Buffer, Property, RenderContext
from moviepy.ae.effects import registry
from moviepy.ae.effects.noise.add_grain import Noise
from moviepy.ae.effects.stylize.mosaic import Mosaic
from moviepy.ae.effects.stylize.posterize import Posterize
from moviepy.ae.effects.stylize.vignette import Vignette


def uniform(color, alpha=1.0, size=(4, 3), offset=(0, 0), space="srgb"):
    """Premultiplied buffer of straight ``color`` (0..1)."""
    width, height = size
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(color, dtype=np.float32) * alpha
    rgba[..., 3] = alpha
    return Buffer(rgba, offset, space)


def random_buffer(seed, size=(16, 12)):
    """Random premultiplied RGBA with some transparent pixels."""
    rng = np.random.default_rng(seed)
    width, height = size
    straight = rng.random((height, width, 4), dtype=np.float32)
    transparent = rng.random((height, width)) < 0.2
    straight[..., 3] = np.where(transparent, 0.0, straight[..., 3])
    rgba = straight.copy()
    rgba[..., :3] *= straight[..., 3:]
    return Buffer(rgba)


def assert_valid(buffer):
    """Premultiplied invariant: 0 <= rgb <= alpha <= 1."""
    rgb, alpha = buffer.rgba[..., :3], buffer.rgba[..., 3:]
    assert np.all(alpha >= 0) and np.all(alpha <= 1)
    assert np.all(rgb >= 0) and np.all(rgb <= alpha + 1e-6)


# -- registration ------------------------------------------------------------- #


def test_effects_are_registered_with_metadata():
    for name, cls, category in (
        ("Posterize", Posterize, "Stylize"),
        ("Mosaic", Mosaic, "Stylize"),
        ("Vignette", Vignette, "Stylize"),
        ("Noise", Noise, "Noise & Grain"),
    ):
        assert registry.get(name) is cls
        assert cls.category == category
        assert "Notes" in cls.__doc__
        for param in cls.PARAMS:
            assert param.name in cls.__doc__


# -- Posterize ----------------------------------------------------------------- #


def test_posterize_default_is_exact_identity():
    src = random_buffer(1)
    assert Posterize().process(src, 0.0) is src


def test_posterize_two_levels_snaps_to_0_or_1():
    src = uniform((0.3, 0.6, 0.9), alpha=0.5)
    out = Posterize(level=2).process(src, 0.0)
    np.testing.assert_allclose(out.rgba[0, 0], [0.0, 0.5, 0.5, 0.5], atol=1e-6)


def test_posterize_three_levels_known_values():
    # N=3 gives steps at 0, 0.5 and 1.
    src = uniform((0.3, 0.8, 0.0))
    out = Posterize(level=3).process(src, 0.0)
    np.testing.assert_allclose(out.rgba[0, 0], [0.5, 1.0, 0.0, 1.0], atol=1e-6)


def test_posterize_keyframed_level_animates():
    src = uniform((0.3, 0.3, 0.3))
    effect = Posterize(level=Property(255.0, keyframes=[(0, 255.0), (1, 2.0)]))
    assert effect.process(src, 0.0) is src
    assert effect.process(src, 1.0).rgba[0, 0, 0] == pytest.approx(0.0, abs=1e-6)


def test_posterize_linear_and_empty_buffers():
    lin = uniform((0.2, 0.4, 0.6), space="linear")
    out = Posterize(level=4).process(lin, 0.0)
    assert out.color_space == "linear"
    assert_valid(out)
    empty = Buffer(np.zeros((0, 3, 4), dtype=np.float32))
    assert 0 in Posterize(level=2).process(empty, 0.0).size


# -- Mosaic -------------------------------------------------------------------- #


def test_mosaic_averages_premultiplied_blocks():
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[0, 0] = [1.0, 0.0, 0.0, 1.0]  # opaque red; the other three are transparent
    out = Mosaic(horizontal_blocks=1, vertical_blocks=1).process(Buffer(rgba), 0.0)
    np.testing.assert_allclose(out.rgba[0, 0], [0.25, 0.0, 0.0, 0.25], atol=1e-6)
    np.testing.assert_allclose(out.rgba[1, 1], [0.25, 0.0, 0.0, 0.25], atol=1e-6)
    assert_valid(out)


def test_mosaic_two_by_two_blocks_on_four_by_four():
    rgba = np.zeros((4, 4, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    rgba[:2, :2, :3] = 1.0
    rgba[2:, 2:, :3] = 0.5
    out = Mosaic(horizontal_blocks=2, vertical_blocks=2).process(Buffer(rgba), 0.0)
    np.testing.assert_allclose(out.rgba[:2, :2, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(out.rgba[2:, 2:, 0], 0.5, atol=1e-6)
    np.testing.assert_allclose(out.rgba[:2, 2:, 0], 0.0, atol=1e-6)
    assert_valid(out)


def test_mosaic_one_pixel_blocks_is_identity_and_caps_counts():
    src = random_buffer(2, size=(5, 4))
    out = Mosaic(horizontal_blocks=5, vertical_blocks=4).process(src, 0.0)
    np.testing.assert_allclose(out.rgba, src.rgba, atol=1e-6)
    capped = Mosaic(horizontal_blocks=4000, vertical_blocks=4000).process(src, 0.0)
    np.testing.assert_allclose(capped.rgba, src.rgba, atol=1e-6)
    assert capped.size == src.size


def test_mosaic_sharp_colors_samples_block_centers():
    src = random_buffer(3, size=(8, 8))
    out = Mosaic(horizontal_blocks=2, vertical_blocks=2, sharp_colors=True).process(
        src, 0.0
    )
    center = np.broadcast_to(src.rgba[2, 2], (4, 4, 4))
    np.testing.assert_allclose(out.rgba[:4, :4], center, atol=1e-6)
    assert_valid(out)


def test_mosaic_random_buffer_output_is_valid_and_empty_passes_through():
    out = Mosaic(horizontal_blocks=3, vertical_blocks=7).process(random_buffer(4), 0.0)
    assert out.size == (16, 12)
    assert_valid(out)
    empty = Buffer(np.zeros((3, 0, 4), dtype=np.float32))
    assert 0 in Mosaic().process(empty, 0.0).size


# -- Vignette ------------------------------------------------------------------ #


def test_vignette_zero_amount_is_identity():
    src = random_buffer(5)
    assert Vignette().process(src, 0.0) is src


def test_vignette_negative_darkens_corners_not_center():
    src = uniform((0.8, 0.8, 0.8), size=(31, 31))
    out = Vignette(amount=-100, midpoint=50, feather=50).process(src, 0.0)
    center, edge, corner = out.rgba[15, 15, 0], out.rgba[0, 15, 0], out.rgba[0, 0, 0]
    assert center == pytest.approx(0.8, abs=1e-5)
    assert corner < edge < center
    assert_valid(out)


def test_vignette_positive_lightens_edges_toward_white():
    src = uniform((0.2, 0.2, 0.2), size=(31, 31))
    out = Vignette(amount=100, midpoint=0, feather=0).process(src, 0.0)
    assert out.rgba[15, 15, 0] == pytest.approx(0.2, abs=1e-5)
    assert out.rgba[0, 0, 0] > 0.9
    assert_valid(out)


def test_vignette_preserves_alpha_and_linear_space():
    src = random_buffer(6)
    out = Vignette(amount=-50).process(src, 0.0)
    np.testing.assert_allclose(out.rgba[..., 3], src.rgba[..., 3], atol=1e-6)
    lin = uniform((0.5, 0.5, 0.5), space="linear")
    assert Vignette(amount=-30).process(lin, 0.0).color_space == "linear"


def test_vignette_keyframed_amount_animates():
    src = uniform((0.8, 0.8, 0.8), size=(9, 9))
    effect = Vignette(amount=Property(0.0, keyframes=[(0, 0.0), (1, -100.0)]))
    assert effect.process(src, 0.0) is src
    assert effect.process(src, 1.0).rgba[0, 0, 0] < 0.8


def test_vignette_empty_buffer_passes_through():
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))
    assert Vignette(amount=50).process(empty, 0.0) is empty


# -- Noise --------------------------------------------------------------------- #


def test_noise_zero_amount_is_identity():
    src = random_buffer(7)
    assert Noise().process(src, 0.0) is src


def test_noise_same_time_is_bit_identical():
    src = random_buffer(8)
    fx = Noise(amount_of_noise=30, random_seed=5)
    first = fx.process(src, 0.5, RenderContext()).rgba
    second = fx.process(src, 0.5, RenderContext()).rgba
    assert np.array_equal(first, second)
    assert np.array_equal(fx.process(src, 0.5).rgba, fx.process(src, 0.5).rgba)


def test_noise_different_time_or_seed_gives_different_grain():
    src = uniform((0.5, 0.5, 0.5), size=(32, 32))
    fx = Noise(amount_of_noise=40)
    assert not np.array_equal(fx.process(src, 0.0).rgba, fx.process(src, 0.04).rgba)
    other = Noise(amount_of_noise=40, random_seed=9)
    assert not np.array_equal(fx.process(src, 0.0).rgba, other.process(src, 0.0).rgba)


def test_noise_mono_has_equal_channels_and_color_does_not():
    src = uniform((0.5, 0.5, 0.5), size=(16, 16))
    mono = Noise(amount_of_noise=50, noise_type="mono").process(src, 0.0).rgba
    np.testing.assert_allclose(mono[..., 0], mono[..., 1], atol=1e-6)
    np.testing.assert_allclose(mono[..., 1], mono[..., 2], atol=1e-6)
    color = Noise(amount_of_noise=50, noise_type="color").process(src, 0.0).rgba
    assert not np.allclose(color[..., 0], color[..., 1])


def test_noise_clipping_keeps_range_and_transparency():
    src = random_buffer(9)
    for clipping in (True, False):
        out = Noise(amount_of_noise=100, clipping=clipping).process(src, 0.0)
        assert_valid(out)
        transparent = src.rgba[..., 3] == 0
        assert np.all(out.rgba[transparent] == 0)


def test_noise_keyframed_amount_animates():
    src = uniform((0.5, 0.5, 0.5), size=(16, 16))
    fx = Noise(amount_of_noise=Property(0.0, keyframes=[(0, 0.0), (1, 80.0)]))
    assert fx.process(src, 0.0) is src
    assert not np.array_equal(fx.process(src, 1.0).rgba, src.rgba)


def test_noise_linear_and_empty_buffers():
    lin = uniform((0.4, 0.4, 0.4), size=(8, 8), space="linear")
    out = Noise(amount_of_noise=20).process(lin, 0.0)
    assert out.color_space == "linear"
    empty = Buffer(np.zeros((0, 4, 4), dtype=np.float32))
    assert 0 in Noise(amount_of_noise=50).process(empty, 0.0).size
