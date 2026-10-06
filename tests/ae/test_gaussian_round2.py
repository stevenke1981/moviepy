"""Gaussian's verified SDR path preserves the reference RGBA projection."""

import cv2
import numpy as np

import pytest

from moviepy.ae import Buffer
from moviepy.ae.effects.blur.gaussian_blur import GaussianBlur, filter_pixels, gaussian


def _reference(pixels, sigma_x, sigma_y, border, unit_range):
    """Apply the established filter and independent channel projection."""
    out = filter_pixels(pixels, sigma_x, sigma_y, border)
    np.clip(out[..., 3], 0.0, 1.0, out=out[..., 3])
    if unit_range:
        np.maximum(out[..., :3], 0.0, out=out[..., :3])
        np.minimum(out[..., :3], out[..., 3:], out=out[..., :3])
    out[out[..., 3] == 0, :3] = 0
    return out


@pytest.mark.parametrize("sigma", [0.3, 3, 20, 37.3])
@pytest.mark.parametrize("alpha", [0, 0.3, 1])
@pytest.mark.parametrize(
    "dimensions", ["horizontal_and_vertical", "horizontal", "vertical"]
)
@pytest.mark.parametrize("repeat", [False, True])
def test_verified_gaussian_projection_matches_reference(
    sigma, alpha, dimensions, repeat
):
    pixels = np.random.default_rng(619).random((37, 49, 4), dtype=np.float32)
    pixels[..., 3] = alpha
    pixels[..., :3] *= alpha
    pixels[0, 0, :3] = alpha
    source = Buffer(pixels, offset=(-19, 27))
    assert source._unit_premultiplied
    effect = GaussianBlur(
        blurriness=sigma * 2,
        blur_dimensions=dimensions,
        repeat_edge_pixels=repeat,
    )
    values = effect.values_at(0)
    expected = _reference(
        source.rgba,
        *effect.sigmas(values),
        cv2.BORDER_REPLICATE if repeat else cv2.BORDER_CONSTANT,
        unit_range=True,
    )
    actual = effect.render(source, 0, values=values)
    np.testing.assert_array_equal(actual.rgba, expected)
    np.testing.assert_array_equal(source.rgba, pixels)
    assert actual.offset == source.offset
    assert actual.rgba.dtype == np.float32
    assert np.all(actual.rgba[..., :3] <= actual.rgba[..., 3:])
    assert not np.any(actual.rgba[..., :3][actual.rgba[..., 3] == 0])


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("color_space", ["srgb", "linear"])
def test_signed_hdr_gaussian_keeps_unbounded_channel_policy(repeat, color_space):
    pixels = np.zeros((151, 157, 4), np.float32)
    pixels[73, 79] = [3, -0.7, 0.3, 0.25]
    source = Buffer(pixels, offset=(-4, 9), color_space=color_space)
    assert not source._unit_premultiplied
    effect = GaussianBlur(blurriness=40, repeat_edge_pixels=repeat)
    actual = effect.render(source, 0)
    expected = _reference(
        source.rgba,
        20,
        20,
        cv2.BORDER_REPLICATE if repeat else cv2.BORDER_CONSTANT,
        unit_range=False,
    )
    np.testing.assert_array_equal(actual.rgba, expected)
    assert np.any(actual.rgba[..., 0] > actual.rgba[..., 3])
    assert np.min(actual.rgba[..., 1]) < 0
    assert not np.any(actual.rgba[:13])
    assert not np.any(actual.rgba[..., :3][actual.rgba[..., 3] == 0])
    assert actual.color_space == color_space


@pytest.mark.parametrize("sigmas", [(0, 0), (0.3, 8), (20, 20), (0, 20)])
@pytest.mark.parametrize("border", [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE])
@pytest.mark.parametrize("seed", [21, 202, 718])
def test_generic_unit_projection_remains_exact_for_unbounded_input(
    sigmas, border, seed
):
    pixels = np.random.default_rng(seed).uniform(-1, 3, (43, 51, 4)).astype(np.float32)
    pixels[10:33, 11:34] = 0
    expected = _reference(pixels, *sigmas, border, unit_range=True)
    actual = gaussian(pixels, *sigmas, border)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "level",
    [
        np.nextafter(np.float32(1), np.float32(0)),
        1,
        np.nextafter(np.float32(1), np.float32(2)),
    ],
)
@pytest.mark.parametrize("variable_alpha", [False, True])
@pytest.mark.parametrize("color_space", ["srgb", "linear"])
@pytest.mark.parametrize("repeat", [False, True])
def test_white_boundary_and_conservative_metadata_keep_previous_policy(
    level, variable_alpha, color_space, repeat
):
    pixels = np.ones((29, 31, 4), np.float32)
    if variable_alpha:
        pixels[::2, :, 3] = 0.5
    pixels[..., :3] = level * pixels[..., 3:]
    source = Buffer(pixels, color_space=color_space)
    if variable_alpha:
        assert not source._unit_premultiplied
    effect = GaussianBlur(blurriness=40, repeat_edge_pixels=repeat)
    expected = _reference(
        source.rgba,
        20,
        20,
        cv2.BORDER_REPLICATE if repeat else cv2.BORDER_CONSTANT,
        unit_range=source._unit_premultiplied,
    )
    actual = effect.render(source, 0)
    np.testing.assert_array_equal(actual.rgba, expected)
    assert actual.color_space == source.color_space
