"""Independent numerical and ownership checks for opaque blend acceleration."""

import numpy as np

import pytest

from moviepy.ae import Buffer
from moviepy.ae.blend import blend


def _lum(rgb):
    return np.sum(rgb * np.array([0.3, 0.59, 0.11]), axis=-1, keepdims=True)


def _hue64(backdrop, source):
    """Evaluate the W3C Hue definition in float64, including both ClipColor steps."""
    low = source.min(axis=-1, keepdims=True)
    span = source.max(axis=-1, keepdims=True) - low
    saturation = np.ptp(backdrop, axis=-1, keepdims=True)
    scale = np.divide(saturation, span, out=np.zeros_like(span), where=span > 0)
    shifted = (source - low) * scale
    mixed = shifted + _lum(backdrop) - _lum(shifted)
    lum, low, high = (
        _lum(mixed),
        mixed.min(-1, keepdims=True),
        mixed.max(-1, keepdims=True),
    )
    pull = np.divide(lum, lum - low, out=np.zeros_like(lum), where=lum > low)
    push = np.divide(1 - lum, high - lum, out=np.zeros_like(lum), where=high > lum)
    mixed = np.where(low < 0, lum + (mixed - lum) * pull, mixed)
    return np.where(high > 1, lum + (mixed - lum) * push, mixed)


@pytest.mark.parametrize("mode", ["multiply", "overlay", "hue"])
@pytest.mark.parametrize("opacity", [0.01, 0.2, 0.4, 0.75, 1])
@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("space", ["srgb", "linear"])
def test_opaque_uniform_blends_match_float64_reference(mode, opacity, preserve, space):
    rng = np.random.default_rng(4067)
    a = rng.random((41, 71, 4), dtype=np.float32)
    b = rng.random(a.shape, dtype=np.float32)
    a[..., 3], b[..., 3] = 1, opacity
    b[..., :3] *= b[..., 3:4]
    base, source = Buffer(a, (-6, 11), space), Buffer(b, (-6, 11), space)
    cb = a[..., :3].astype(np.float64)
    alpha = float(b[0, 0, 3])
    cs = b[..., :3].astype(np.float64) / alpha
    if mode == "multiply":
        mixed = cb * cs
    elif mode == "overlay":
        mixed = np.where(cb <= 0.5, 2 * cb * cs, 1 - 2 * (1 - cb) * (1 - cs))
    else:
        mixed = _hue64(cb, cs)
    expected = alpha * mixed + (1 - alpha) * cb
    result = blend(base, source, mode, preserve_underlying_transparency=preserve)
    np.testing.assert_allclose(result.rgba[..., :3], expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(result.rgba[..., 3], 1)
    np.testing.assert_array_equal(base.rgba, a)
    np.testing.assert_array_equal(source.rgba, b)
    assert result.bounds == base.bounds
    assert not result.rgba.flags.writeable
    assert not np.shares_memory(result.rgba, base.rgba)
    assert not np.shares_memory(result.rgba, source.rgba)


@pytest.mark.parametrize("mode", ["multiply", "overlay", "hue"])
def test_opaque_metadata_remains_conservative_at_float32_boundaries(mode):
    rng = np.random.default_rng(624)
    for alpha in [
        np.finfo(np.float32).tiny,
        0.001,
        0.4,
        np.nextafter(np.float32(1), np.float32(0)),
        1,
    ]:
        alpha = np.float32(alpha)
        a = rng.random((19, 67, 4), dtype=np.float32)
        b = rng.random(a.shape, dtype=np.float32)
        a[..., 3], b[..., 3] = 1, alpha
        a[0, :5, :3] = np.array(
            [
                0,
                np.nextafter(np.float32(0), np.float32(1)),
                0.5,
                np.nextafter(np.float32(1), np.float32(0)),
                1,
            ]
        )[:, None]
        b[..., :3] *= alpha
        b[0, :5, :3] = a[0, :5, :3] * alpha
        result = blend(Buffer(a), Buffer(b), mode)
        assert np.isfinite(result.rgba).all()
        assert float(np.max(np.abs(result.rgba))) <= result._value_bound
        if result._unit_premultiplied:
            assert np.all(result.rgba >= 0)
            assert np.all(result.rgba[..., :3] <= result.rgba[..., 3:4])
        assert np.all(result.rgba[..., 3] == result._uniform_alpha)


@pytest.mark.parametrize("mode", ["multiply", "overlay", "hue"])
def test_accelerated_output_survives_later_random_order_evaluations(mode):
    rng = np.random.default_rng(13)
    sources = []
    for _ in range(4):
        pixels = rng.random((97, 349, 4), dtype=np.float32)
        pixels[..., 3] = 0.4
        pixels[..., :3] *= 0.4
        sources.append(Buffer(pixels))
    backdrop = np.ones((97, 349, 4), np.float32)
    backdrop[..., :3] = rng.random(backdrop[..., :3].shape, dtype=np.float32)
    base = Buffer(backdrop)
    escaped = blend(base, sources[2], mode).rgba
    expected = escaped.copy()
    for index in [3, 0, 1, 2, 0]:
        result = blend(base, sources[index], mode)
        if index == 2:
            np.testing.assert_array_equal(result.rgba, escaped)
    np.testing.assert_array_equal(escaped, expected)
