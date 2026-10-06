"""Blend locality across tiles and constant-coverage fast paths."""

import numpy as np

import pytest

from moviepy.ae import Buffer, RenderContext
from moviepy.ae.blend import BlendMode, blend


@pytest.mark.parametrize("mode", list(BlendMode))
@pytest.mark.parametrize("preserve", [False, True])
def test_changing_one_pixel_does_not_change_other_tiles(mode, preserve):
    rng = np.random.default_rng(914)
    backdrop = rng.random((97, 347, 4), dtype=np.float32)
    source = rng.random(backdrop.shape, dtype=np.float32)
    backdrop[..., 3] = 1
    source[..., 3] = 0.4
    source[..., :3] *= source[..., 3:4]
    base = Buffer(backdrop, offset=(-7, 4))
    uniform = Buffer(source, offset=base.offset)
    source[-1, -1] = 0
    varying = Buffer(source, offset=base.offset)
    kwargs = dict(
        preserve_underlying_transparency=preserve,
        context=RenderContext(rng_seed=7),
        layer_id="tiles",
    )
    before = blend(base, uniform, mode, **kwargs)
    after = blend(base, varying, mode, **kwargs)
    np.testing.assert_allclose(before.rgba[:-1], after.rgba[:-1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        before.rgba[-1, :-1], after.rgba[-1, :-1], rtol=0, atol=1e-6
    )
    assert not before.rgba.flags.writeable


@pytest.mark.parametrize("mode", ["multiply", "screen"])
@pytest.mark.parametrize("ab", [0, 0.2, 0.7, 1])
@pytest.mark.parametrize("alpha_s", [0, 0.3, 1])
@pytest.mark.parametrize("preserve", [False, True])
def test_uniform_sdr_identities_match_float64_source_over(mode, ab, alpha_s, preserve):
    rng = np.random.default_rng(821)
    a = rng.random((11, 13, 4), dtype=np.float32)
    b = rng.random(a.shape, dtype=np.float32)
    a[..., 3], b[..., 3] = ab, alpha_s
    a[..., :3] *= a[..., 3:4]
    b[..., :3] *= b[..., 3:4]
    base, source = Buffer(a), Buffer(b)
    cb, cs = a[..., :3].astype(np.float64), b[..., :3].astype(np.float64)
    ab, alpha_s = float(a[0, 0, 3]), float(b[0, 0, 3])
    backdrop = cb / ab if ab else np.zeros_like(cb)
    front = cs / alpha_s if alpha_s else np.zeros_like(cs)
    mixed = (
        backdrop * front if mode == "multiply" else backdrop + front - backdrop * front
    )
    rgb = alpha_s * ab * mixed + (1 - alpha_s) * cb
    alpha = ab
    if not preserve:
        rgb += (1 - ab) * cs
        alpha = alpha_s + ab - alpha_s * ab
    actual = blend(base, source, mode, preserve_underlying_transparency=preserve)
    np.testing.assert_allclose(actual.rgba[..., :3], rgb, atol=3e-7, rtol=0)
    np.testing.assert_allclose(actual.rgba[..., 3], alpha, atol=1e-7, rtol=0)
