"""Check fused Hue against the independent, sequential W3C definition."""

import numpy as np

import pytest

from moviepy.ae import Buffer
from moviepy.ae.blend import blend


def _reference_hue(backdrop, source):
    """Evaluate SetSat, SetLum and both ClipColor steps in float64."""
    coefficients = np.array([0.3, 0.59, 0.11], dtype=np.float64)

    def lum(rgb):
        return np.sum(rgb * coefficients, axis=-1, keepdims=True)

    low = source.min(axis=-1, keepdims=True)
    span = source.max(axis=-1, keepdims=True) - low
    factor = np.divide(
        np.ptp(backdrop, axis=-1, keepdims=True),
        span,
        out=np.zeros_like(span),
        where=span > 0,
    )
    scaled = (source - low) * factor
    mixed = scaled + lum(backdrop) - lum(scaled)
    level = lum(mixed)
    low, high = mixed.min(-1, keepdims=True), mixed.max(-1, keepdims=True)
    pull = np.divide(level, level - low, out=np.zeros_like(level), where=level > low)
    push = np.divide(
        1 - level, high - level, out=np.zeros_like(level), where=high > level
    )
    mixed = np.where(low < 0, level + (mixed - level) * pull, mixed)
    return np.where(high > 1, level + (mixed - level) * push, mixed)


@pytest.mark.parametrize("shape", [(35, 281), (2, 1), (1, 21)])
@pytest.mark.parametrize("space", ["srgb", "linear"])
@pytest.mark.parametrize(
    "alpha",
    [
        np.finfo(np.float32).tiny,
        1e-6,
        0.01,
        0.25,
        0.4,
        np.nextafter(np.float32(1), np.float32(0)),
        1,
    ],
)
def test_fused_hue_gamut_edges_and_grays_match_sequential_reference(
    shape, space, alpha
):
    rng = np.random.default_rng(640)
    palette = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0.5, 0.5, 0.5],
            [0.5, np.nextafter(np.float32(0.5), np.float32(1)), 0.5],
            [0.01, 0.09, 0.02],
            [0.91, 0.99, 0.96],
        ],
        dtype=np.float32,
    )
    a, b = [np.ones((*shape, 4), np.float32) for _ in range(2)]
    a[..., :3] = palette[rng.integers(len(palette), size=shape)]
    b[..., :3] = palette[rng.integers(len(palette), size=shape)]
    alpha = np.float32(alpha)
    b *= alpha
    # Preserve the public path's float32 source normalization before evaluating
    # the independent float64 color formula (important for near-gray sources).
    cs = np.clip(b[..., :3] * (np.float32(1) / alpha), 0, 1).astype(np.float64)
    cb = a[..., :3].astype(np.float64)
    expected = float(alpha) * _reference_hue(cb, cs) + (1 - float(alpha)) * cb
    base, source = Buffer(a, (-17, 31), space), Buffer(b, (-17, 31), space)
    result = blend(base, source, "hue")
    np.testing.assert_allclose(result.rgba[..., :3], expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(result.rgba[..., 3], 1)
    np.testing.assert_array_equal(base.rgba, a)
    np.testing.assert_array_equal(source.rgba, b)
    assert result.color_space == space and result.bounds == base.bounds
    assert not result.rgba.flags.writeable
    assert not np.shares_memory(result.rgba, base.rgba)
    assert not np.shares_memory(result.rgba, source.rgba)
    assert float(np.max(np.abs(result.rgba))) <= result._value_bound
    if result._unit_premultiplied:
        assert np.all(result.rgba >= 0)
        assert np.all(result.rgba[..., :3] <= result.rgba[..., 3:])


@pytest.mark.parametrize("space", ["srgb", "linear"])
@pytest.mark.parametrize("base_alpha", [0.2, 1])
@pytest.mark.parametrize("variable_alpha", [False, True])
def test_signed_hdr_hue_keeps_general_alpha_and_unbounded_composite(
    space, base_alpha, variable_alpha
):
    rng = np.random.default_rng(61)
    a, b = [rng.uniform(-0.5, 5, (9, 37, 4)).astype(np.float32) for _ in range(2)]
    a[..., 3], b[..., 3] = np.float32(base_alpha), np.float32(0.4)
    if variable_alpha:
        b[::2, ..., 3] = 0.25
    a[..., :3] *= a[..., 3:]
    b[..., :3] *= b[..., 3:]
    ab, opacity = a[..., 3:].astype(np.float64), b[..., 3:].astype(np.float64)
    cb, cs = a[..., :3].astype(np.float64), b[..., :3].astype(np.float64)
    mixed = _reference_hue(np.clip(cb / ab, 0, 1), np.clip(cs / opacity, 0, 1))
    expected = (1 - opacity) * cb + (1 - ab) * cs + opacity * ab * mixed
    base, source = Buffer(a, (-7, 19), space), Buffer(b, (-7, 19), space)
    result = blend(base, source, "hue")
    np.testing.assert_allclose(result.rgba[..., :3], expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        result.rgba[..., 3:], opacity + ab - opacity * ab, rtol=0, atol=1e-7
    )
    assert np.any(result.rgba[..., :3] > result.rgba[..., 3:])
    assert np.any(result.rgba[..., :3] < 0)
    np.testing.assert_array_equal(base.rgba, a)
    np.testing.assert_array_equal(source.rgba, b)
