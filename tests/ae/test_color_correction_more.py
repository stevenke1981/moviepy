"""Analytic tests for Hue/Saturation, Levels, Exposure and Vibrance."""

import numpy as np

import pytest

from moviepy.ae import Buffer
from moviepy.ae.color import linear_to_srgb, srgb_to_linear
from moviepy.ae.effects.color.exposure import Exposure
from moviepy.ae.effects.color.hue_saturation import HueSaturation
from moviepy.ae.effects.color.levels import Levels
from moviepy.ae.effects.color.vibrance import Vibrance
from moviepy.ae.effects.registry import get


REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def uniform(color, alpha=1.0, size=(4, 3), offset=(0, 0), color_space="srgb"):
    """Premultiplied buffer of straight ``color`` (0..1)."""
    width, height = size
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(color, dtype=np.float32) * alpha
    rgba[..., 3] = alpha
    return Buffer(rgba, offset, color_space=color_space)


def pixel(buffer):
    """Return the RGBA of the first pixel."""
    return buffer.rgba[0, 0]


IDENTITY_CASES = [
    HueSaturation(),
    HueSaturation(colorize=False, colorize_saturation=25),
    Levels(),
    Exposure(),
    Exposure(exposure=0, offset=0, gamma_correction=1),
    Vibrance(),
]


@pytest.mark.parametrize("effect", IDENTITY_CASES, ids=lambda e: type(e).__name__)
def test_defaults_are_exact_identity(effect):
    source = uniform((0.2, 0.4, 0.6), alpha=0.5)
    assert effect.process(source, 0.0) is source


def test_registry_names_resolve():
    assert get("Hue/Saturation") is HueSaturation
    assert get("Levels") is Levels
    assert get("Exposure") is Exposure
    assert get("Vibrance") is Vibrance


# -- Hue/Saturation ---------------------------------------------------------- #


def test_hue_120_turns_red_into_green():
    out = HueSaturation(master_hue=120).process(uniform((1, 0, 0)), 0.0)
    np.testing.assert_allclose(pixel(out), [0, 1, 0, 1], atol=1e-6)


def test_hue_negative_120_turns_red_into_blue():
    out = HueSaturation(master_hue=-120).process(uniform((1, 0, 0)), 0.0)
    np.testing.assert_allclose(pixel(out), [0, 0, 1, 1], atol=1e-6)


def test_hue_360_is_identity_on_colored_pixels():
    source = uniform((0.2, 0.6, 0.9))
    out = HueSaturation(master_hue=360).process(source, 0.0)
    np.testing.assert_allclose(out.rgba, source.rgba, atol=1e-6)


def test_saturation_minus_100_gives_hsl_gray():
    # Red has L = 0.5, so full desaturation yields 50 % gray.
    out = HueSaturation(master_saturation=-100).process(uniform((1, 0, 0)), 0.0)
    np.testing.assert_allclose(pixel(out), [0.5, 0.5, 0.5, 1], atol=1e-6)


def test_lightness_plus_100_reaches_white():
    out = HueSaturation(master_lightness=100).process(uniform((0.25,) * 3), 0.0)
    np.testing.assert_allclose(pixel(out), [1, 1, 1, 1], atol=1e-6)


def test_lightness_minus_100_reaches_black():
    out = HueSaturation(master_lightness=-100).process(uniform((0.7, 0.2, 0.1)), 0.0)
    np.testing.assert_allclose(pixel(out), [0, 0, 0, 1], atol=1e-6)


def test_colorize_tints_gray_pixels():
    gray = uniform((0.5, 0.5, 0.5))
    blue = HueSaturation(
        colorize=True, colorize_hue=240, colorize_saturation=100
    ).process(gray, 0.0)
    np.testing.assert_allclose(pixel(blue), [0, 0, 1, 1], atol=1e-6)
    # Default colorize saturation (25 %) on mid gray gives a faint red cast.
    faint = HueSaturation(colorize=True).process(gray, 0.0)
    np.testing.assert_allclose(pixel(faint), [0.625, 0.375, 0.375, 1], atol=1e-6)


def test_hue_saturation_preserves_alpha_premultiplied():
    out = HueSaturation(master_hue=120).process(uniform((1, 0, 0), alpha=0.5), 0.0)
    np.testing.assert_allclose(pixel(out), [0, 0.5, 0, 0.5], atol=1e-6)
    assert np.all(out.rgba[..., :3] <= out.rgba[..., 3:] + 1e-6)


def test_hue_saturation_empty_buffer():
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))
    out = HueSaturation(master_hue=90, master_saturation=20).process(empty, 0.0)
    assert 0 in out.size


# -- Levels ------------------------------------------------------------------ #


def test_levels_input_white_128_scales_quarter_grey():
    # 0.25 is code 63.75; dividing by 128 gives ~0.498, i.e. almost doubling.
    out = Levels(input_white=128).process(uniform((0.25,) * 3), 0.0)
    np.testing.assert_allclose(pixel(out)[:3], 0.25 * 255 / 128, atol=1e-6)
    np.testing.assert_allclose(pixel(out)[:3], 0.5, atol=3e-3)


def test_levels_input_white_127_5_exactly_doubles_quarter_grey():
    out = Levels(input_white=127.5).process(uniform((0.25,) * 3), 0.0)
    np.testing.assert_allclose(pixel(out)[:3], 0.5, atol=1e-6)


def test_levels_input_black_clips_to_black():
    # Input black at code 64 (0.251) crushes a 0.25 grey to zero.
    out = Levels(input_black=64).process(uniform((0.25,) * 3), 0.0)
    np.testing.assert_allclose(pixel(out), [0, 0, 0, 1], atol=1e-6)


def test_levels_gamma_two_square_roots_midtones():
    out = Levels(gamma=2).process(uniform((0.25,) * 3), 0.0)
    np.testing.assert_allclose(pixel(out)[:3], 0.5, atol=1e-6)


def test_levels_output_range_maps_linearly():
    white = Levels(output_white=128).process(uniform((1, 1, 1)), 0.0)
    np.testing.assert_allclose(pixel(white)[:3], 128 / 255, atol=1e-6)
    black = Levels(output_black=51).process(uniform((0, 0, 0)), 0.0)
    np.testing.assert_allclose(pixel(black)[:3], 0.2, atol=1e-6)


def test_levels_preserves_alpha_and_empty_buffer():
    out = Levels(gamma=2).process(uniform((0.25,) * 3, alpha=0.5), 0.0)
    # Straight 0.25 -> sqrt(0.25) = 0.5, re-premultiplied by alpha 0.5.
    np.testing.assert_allclose(pixel(out), [0.25, 0.25, 0.25, 0.5], atol=1e-6)
    assert np.all(out.rgba[..., :3] <= out.rgba[..., 3:] + 1e-6)
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))
    assert 0 in Levels(gamma=2).process(empty, 0.0).size


# -- Exposure ---------------------------------------------------------------- #


def test_exposure_plus_one_doubles_linear_value():
    linear = uniform((0.25,) * 3, color_space="linear")
    out = Exposure(exposure=1).process(linear, 0.0)
    np.testing.assert_allclose(pixel(out), [0.5, 0.5, 0.5, 1], atol=1e-6)
    assert out.color_space == "linear"


def test_exposure_doubles_linear_light_of_srgb_input():
    srgb = uniform((0.5,) * 3)
    out = Exposure(exposure=1).process(srgb, 0.0)
    expected = linear_to_srgb(2.0 * srgb_to_linear(np.float32(0.5)))
    np.testing.assert_allclose(pixel(out)[:3], expected, atol=1e-5)
    assert out.color_space == "srgb"


def test_exposure_offset_adds_in_linear_light():
    out = Exposure(offset=0.1).process(uniform((0.4,) * 3, color_space="linear"), 0.0)
    np.testing.assert_allclose(pixel(out)[:3], 0.5, atol=1e-6)


def test_exposure_gamma_correction_in_linear_light():
    out = Exposure(gamma_correction=2).process(
        uniform((0.25,) * 3, color_space="linear"), 0.0
    )
    np.testing.assert_allclose(pixel(out)[:3], 0.5, atol=1e-6)


def test_exposure_preserves_alpha_and_empty_buffer():
    out = Exposure(exposure=1).process(
        uniform((0.25,) * 3, alpha=0.5, color_space="linear"), 0.0
    )
    np.testing.assert_allclose(pixel(out), [0.25, 0.25, 0.25, 0.5], atol=1e-6)
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32), color_space="linear")
    assert 0 in Exposure(exposure=2).process(empty, 0.0).size


# -- Vibrance ---------------------------------------------------------------- #


def test_vibrance_minus_100_saturation_gives_rec709_gray():
    out = Vibrance(saturation=-100).process(uniform((1, 0, 0)), 0.0)
    np.testing.assert_allclose(pixel(out), [0.2126, 0.2126, 0.2126, 1], atol=1e-6)


def test_saturation_doubles_chroma_and_keeps_luma():
    color = np.array([0.6, 0.5, 0.5], dtype=np.float32)
    out = Vibrance(saturation=100).process(uniform(color), 0.0)
    luma = float(color @ REC709)
    expected = luma + 2.0 * (color - luma)
    np.testing.assert_allclose(pixel(out)[:3], expected, atol=1e-6)
    np.testing.assert_allclose(float(pixel(out)[:3] @ REC709), luma, atol=1e-6)


def test_vibrance_boosts_dull_pixels_more_than_vivid_ones():
    # Chosen so neither pixel leaves the 0..1 range after scaling.
    dull = np.array([0.5, 0.5, 0.6], dtype=np.float32)  # HSV s = 1/6
    vivid = np.array([0.7, 0.5, 0.5], dtype=np.float32)  # HSV s = 2/7
    gains = []
    for color in (dull, vivid):
        out = Vibrance(vibrance=100).process(uniform(color), 0.0)
        luma = float(color @ REC709)
        chroma_in = color - luma
        chroma_out = pixel(out)[:3] - luma
        # Chroma scales by one common gain, so every component agrees.
        gain = chroma_out / chroma_in
        s = (color.max() - color.min()) / color.max()
        np.testing.assert_allclose(gain, 1.0 + (1.0 - s), atol=1e-5)
        gains.append(float(gain[0]))
    assert gains[0] > gains[1]


def test_vibrance_preserves_alpha_and_bounds():
    out = Vibrance(vibrance=100, saturation=50).process(
        uniform((0.5, 0.5, 0.6), alpha=0.5), 0.0
    )
    assert np.all(out.rgba[..., :3] <= out.rgba[..., 3:] + 1e-6)
    np.testing.assert_allclose(out.rgba[..., 3], 0.5, atol=1e-6)


def test_vibrance_empty_buffer():
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))
    assert 0 in Vibrance(vibrance=50).process(empty, 0.0).size
