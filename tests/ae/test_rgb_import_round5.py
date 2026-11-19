"""Preserve RGB code precision and ownership through opaque footage imports."""

from types import SimpleNamespace

import numpy as np

import pytest

from moviepy.ae import Buffer


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.uint64, np.int64, np.float16, np.float32, np.float64],
)
@pytest.mark.parametrize("shape", [(0, 7), (17, 259), (67, 521), (1, 32769)])
@pytest.mark.parametrize("layout", ["contiguous", "reversed", "transposed"])
def test_opaque_footage_codes_keep_precision_and_readonly_ownership(
    dtype, shape, layout
):
    rng = np.random.default_rng(710)
    frame = rng.uniform(0, 255, (*shape, 3)).astype(dtype)
    if frame.size and np.issubdtype(dtype, np.floating):
        frame.reshape(-1)[:6] = [0, -0.0, 0.5, 127.5, 254.5, 255]
        if dtype == np.float64:
            frame.reshape(-1)[6:9] = np.nextafter([0.5, 127.5, 254.5], 255)
    if layout == "reversed":
        frame = frame[::-1, ::-1, ::-1]
    elif layout == "transposed":
        frame = frame.transpose(1, 0, 2)
    frame.setflags(write=False)
    original = frame.tobytes()
    expected = (frame.astype(np.float64) / 255).astype(np.float32)
    clip = SimpleNamespace(get_frame=lambda t: frame, mask=None)
    result = Buffer.from_clip(clip, 0.25, offset=(-13, 29))
    assert result.rgba[..., :3].tobytes() == expected.tobytes()
    np.testing.assert_array_equal(result.rgba[..., 3], 1)
    assert result.rgba.dtype == np.float32 and result.rgba.flags.c_contiguous
    assert not result.rgba.flags.writeable
    assert not np.shares_memory(result.rgba, frame)
    assert frame.tobytes() == original
    assert result.offset == (-13, 29) and result.color_space == "srgb"
    if frame.size:
        assert result._uniform_alpha == 1 and result._unit_premultiplied
        assert result._value_bound == 1
    if dtype == np.uint8:
        strict = Buffer.from_uint8_rgb(frame, offset=(-13, 29))
        assert strict.rgba.tobytes() == result.rgba.tobytes()
        np.testing.assert_array_equal(strict.to_uint8_rgb(), frame)


@pytest.mark.parametrize("mask_value", [0, 1e-20, 0.25, 1, None])
def test_masked_fractional_footage_retains_sequential_premultiplication(mask_value):
    rng = np.random.default_rng(715)
    frame = rng.uniform(0, 255, (67, 521, 3))
    mask = (
        rng.choice([0, 1e-20, 0.25, 1], (67, 521))
        if mask_value is None
        else np.full((67, 521), mask_value)
    )
    before_frame, before_mask = frame.copy(), mask.copy()
    expected = (frame / 255).astype(np.float32)
    expected *= mask.astype(np.float32)[..., None]
    clip = SimpleNamespace(
        get_frame=lambda t: frame,
        mask=SimpleNamespace(get_frame=lambda t: mask),
    )
    result = Buffer.from_clip(clip, 0)
    assert result.rgba[..., :3].tobytes() == expected.tobytes()
    np.testing.assert_array_equal(result.rgba[..., 3], mask.astype(np.float32))
    np.testing.assert_array_equal(frame, before_frame)
    np.testing.assert_array_equal(mask, before_mask)
    assert not np.shares_memory(result.rgba, frame)
    assert not np.shares_memory(result.rgba, mask)


def test_successive_dynamic_frames_do_not_reuse_published_pixels():
    codes = np.arange(256, dtype=np.uint8)
    first = np.broadcast_to(codes[None, :, None], (67, 256, 3)).copy()
    second = np.full((69, 521, 3), 127.5, dtype=np.float64)
    clip = SimpleNamespace(get_frame=lambda t: first if t == 0 else second, mask=None)
    retained = Buffer.from_clip(clip, 0)
    retained_bytes = retained.rgba.tobytes()
    result = Buffer.from_clip(clip, 1)
    first[:] = 0
    second[:] = 255
    assert retained.rgba.tobytes() == retained_bytes
    np.testing.assert_array_equal(result.rgba[..., :3], 0.5)
    np.testing.assert_array_equal(result.rgba[..., 3], 1)
    assert result.size == (521, 69)
    assert not np.shares_memory(result.rgba, retained.rgba)
