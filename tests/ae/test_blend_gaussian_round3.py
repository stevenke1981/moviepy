"""Independent gamut and channel bounds checks across processing tiles."""

import cv2
import numpy as np

import pytest

from moviepy.ae import Buffer
from moviepy.ae.blend import blend
from moviepy.ae.effects.blur.gaussian_blur import filter_pixels, gaussian


def _hue_reference(backdrop, source):
    """Evaluate W3C SetSat, SetLum and sequential ClipColor in float64."""
    coefficients = np.array([0.3, 0.59, 0.11])

    def lum(rgb):
        return np.sum(rgb * coefficients, axis=-1, keepdims=True)

    low = source.min(axis=-1, keepdims=True)
    span = source.max(axis=-1, keepdims=True) - low
    scale = np.divide(
        np.ptp(backdrop, axis=-1, keepdims=True),
        span,
        out=np.zeros_like(span),
        where=span > 0,
    )
    shifted = (source - low) * scale
    mixed = shifted + lum(backdrop) - lum(shifted)
    level = lum(mixed)
    low, high = mixed.min(-1, keepdims=True), mixed.max(-1, keepdims=True)
    pull = np.divide(level, level - low, out=np.zeros_like(level), where=level > low)
    push = np.divide(
        1 - level, high - level, out=np.zeros_like(level), where=high > level
    )
    mixed = np.where(low < 0, level + (mixed - level) * pull, mixed)
    return np.where(high > 1, level + (mixed - level) * push, mixed)


@pytest.mark.parametrize("mode", ["hue", "overlay"])
@pytest.mark.parametrize("shape", [(53, 1), (7, 521), (23, 1921)])
@pytest.mark.parametrize("space", ["srgb", "linear"])
def test_gamut_extrema_gray_and_tile_tails_match_float64(mode, shape, space):
    rng = np.random.default_rng(630)
    palette = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0.5, 0.5, 0.5],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.01, 0.09, 0.02],
            [0.92, 0.96, 0.99],
            [0.5, np.nextafter(np.float32(0.5), np.float32(1)), 0.5],
        ],
        dtype=np.float32,
    )
    a, b = [np.ones((*shape, 4), np.float32) for _ in range(2)]
    a[..., :3] = palette[rng.integers(len(palette), size=shape)]
    b[..., :3] = palette[rng.integers(len(palette), size=shape)]
    cb, cs = a[..., :3].astype(np.float64), b[..., :3].astype(np.float64)
    expected = (
        _hue_reference(cb, cs)
        if mode == "hue"
        else np.where(cb <= 0.5, 2 * cb * cs, 1 - 2 * (1 - cb) * (1 - cs))
    )
    base, source = Buffer(a, (-13, 27), space), Buffer(b, (-13, 27), space)
    actual = blend(base, source, mode)
    np.testing.assert_allclose(actual.rgba[..., :3], expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(actual.rgba[..., 3], 1)
    np.testing.assert_array_equal(base.rgba, a)
    np.testing.assert_array_equal(source.rgba, b)
    assert actual.bounds == base.bounds
    assert actual.color_space == space
    assert not actual.rgba.flags.writeable
    assert not np.shares_memory(actual.rgba, base.rgba)
    assert not np.shares_memory(actual.rgba, source.rgba)
    assert np.max(np.abs(actual.rgba)) <= actual._value_bound
    if actual._unit_premultiplied:
        assert np.all(actual.rgba >= 0)
        assert np.all(actual.rgba[..., :3] <= actual.rgba[..., 3:])


@pytest.mark.parametrize("shape", [(173, 199), (23, 1921), (32769, 1)])
@pytest.mark.parametrize("unit_range", [False, True])
@pytest.mark.parametrize("scale", [1, 1e20])
def test_large_gaussian_projection_preserves_hdr_and_transparent_pixels(
    shape, unit_range, scale
):
    pixels = np.random.default_rng(603).uniform(-1, 3, (*shape, 4)).astype(np.float32)
    pixels[..., :3] *= np.float32(scale)
    pixels[::3, ..., 3] = 0
    pixels[1::5, ..., 3] = 1
    original = pixels.copy()
    expected = pixels.copy()
    np.clip(expected[..., 3], 0, 1, out=expected[..., 3])
    if unit_range:
        np.maximum(expected[..., :3], 0, out=expected[..., :3])
        np.minimum(expected[..., :3], expected[..., 3:], out=expected[..., :3])
    expected[expected[..., 3] == 0, :3] = 0
    actual = gaussian(pixels, 0, 0, cv2.BORDER_CONSTANT, unit_range=unit_range)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(pixels, original)
    assert actual.dtype == np.float32
    assert not np.shares_memory(actual, pixels)


@pytest.mark.parametrize("border", [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE])
@pytest.mark.parametrize("sigmas", [(0, 20), (1.5, 4.3), (20, 20)])
def test_large_filtered_channel_projection_matches_previous_policy(border, sigmas):
    pixels = np.random.default_rng(91).uniform(-1, 3, (173, 199, 4)).astype(np.float32)
    pixels[60:130, 60:130] = 0
    expected = filter_pixels(pixels, *sigmas, border)
    np.clip(expected[..., 3], 0, 1, out=expected[..., 3])
    np.maximum(expected[..., :3], 0, out=expected[..., :3])
    np.minimum(expected[..., :3], expected[..., 3:], out=expected[..., :3])
    expected[expected[..., 3] == 0, :3] = 0
    actual = gaussian(pixels, *sigmas, border)
    np.testing.assert_array_equal(actual, expected)
