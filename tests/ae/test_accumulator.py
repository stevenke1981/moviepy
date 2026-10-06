"""Pixel equivalence and ownership boundaries for renderer accumulation."""

import numpy as np

import pytest

from moviepy.ae import Buffer, RenderContext
from moviepy.ae._accumulator import _Accumulator
from moviepy.ae.blend import BlendMode, blend


@pytest.mark.parametrize("mode", list(BlendMode))
@pytest.mark.parametrize("preserve", [False, True])
def test_mixed_accumulation_matches_immutable_blends_and_keeps_snapshots(
    mode, preserve
):
    rng = np.random.default_rng(9)
    base = Buffer(rng.random((8, 9, 4), dtype=np.float32))
    layer = Buffer(rng.random((3, 4, 4), dtype=np.float32), offset=(-2, 3))
    final = Buffer(rng.random((5, 5, 4), dtype=np.float32), offset=(4, -1))
    context = RenderContext(rng_seed=7)
    accumulator = _Accumulator(base.bounds, base.color_space)
    accumulator.replace(base)
    accumulator.blend(layer, mode, preserve=preserve, context=context, layer_id="same")
    snapshot = accumulator.snapshot()
    snapshot_pixels = snapshot.rgba.copy()
    expected = blend(
        base,
        layer,
        mode,
        preserve_underlying_transparency=preserve,
        context=context,
        layer_id="same",
    )
    np.testing.assert_array_equal(snapshot.rgba, expected.rgba)
    accumulator.blend(final, BlendMode.NORMAL)
    actual = accumulator.snapshot()
    expected = blend(expected, final, BlendMode.NORMAL)
    np.testing.assert_array_equal(actual.rgba, expected.rgba)
    assert actual.bounds == expected.bounds
    np.testing.assert_array_equal(snapshot.rgba, snapshot_pixels)
    assert not snapshot.rgba.flags.writeable


@pytest.mark.parametrize("space", ["srgb", "linear"])
def test_normal_batch_retains_signed_hdr_with_offsets_and_uniform_alpha(space):
    canvas = _Accumulator((-3, -2, 10, 9), space)
    reference = Buffer(
        np.zeros((11, 13, 4), np.float32), offset=(-3, -2), color_space=space
    )
    escaped = []
    for index in range(5):
        pixels = np.ones((3, 4, 4), np.float32)
        pixels[..., :3] = (2.0, -0.2, 0.7)
        pixels *= np.float32(0.2 + index * 0.1)
        source = Buffer(pixels, offset=(index, index - 1), color_space=space)
        canvas.blend(source, BlendMode.NORMAL)
        reference = blend(reference, source)
        if index == 2:
            snapshot = canvas.snapshot()
            escaped.append((snapshot, snapshot.rgba.copy()))
    result = canvas.snapshot()
    np.testing.assert_array_equal(result.rgba, reference.rgba)
    for snapshot, expected in escaped:
        np.testing.assert_array_equal(snapshot.rgba, expected)
