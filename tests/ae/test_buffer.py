"""Analytic contracts for premultiplied float32 image buffers."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np

import pytest

from moviepy.ae.buffer import Buffer, premultiply, unpremultiply
from moviepy.ae.context import RenderContext


def pixel(rgb=(0.2, 0.4, 0.8), alpha=0.5):
    """Return an independently calculated premultiplied pixel."""
    return np.array([[list(np.asarray(rgb) * alpha) + [alpha]]], dtype=np.float32)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("mask_value", [None, 1.0])
def test_all_rgb_codes_roundtrip(strided, mask_value):
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    frame = np.stack((codes, 255 - codes, np.roll(codes, 3)), axis=-1)
    if strided:
        frame = frame[::-1, ::-1]
    mask = None if mask_value is None else np.full(frame.shape[:2], mask_value)
    buffer = Buffer.from_uint8_rgb(frame, mask)
    np.testing.assert_array_equal(buffer.to_uint8_rgb(), frame)
    assert buffer.rgba.dtype == np.float32
    assert buffer.rgba.flags.c_contiguous
    assert not buffer.rgba.flags.writeable


def test_snapshot_frozen_and_no_ndarray_equality():
    source = pixel()
    buffer = Buffer(source, offset=(np.int64(-2), np.int32(3)))
    source[:] = 0
    np.testing.assert_array_equal(buffer.rgba, pixel())
    assert buffer.offset == (-2, 3)
    with pytest.raises(FrozenInstanceError):
        buffer.offset = (0, 0)
    with pytest.raises(ValueError):
        buffer.rgba[0, 0, 0] = 0
    assert buffer != Buffer(pixel())


def test_with_methods_and_exports_are_independent():
    original = Buffer(pixel())
    moved = original.with_offset((-4, 9))
    assert moved is not original
    assert moved.offset == (-4, 9)
    np.testing.assert_array_equal(moved.rgba, original.rgba)
    replacement = pixel((1, 0, 0), 1)
    replaced = original.with_rgba(replacement)
    replacement[:] = 0
    assert replaced.rgba[0, 0, 0] == 1
    straight = original.straight()
    straight[:] = 0
    exported = original.to_uint8_rgb()
    exported[:] = 0
    np.testing.assert_array_equal(original.rgba, pixel())


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 1e-20])
def test_helpers_preserve_hdr_and_tiny_alpha(alpha):
    straight = np.array([[[2.0, -1.5, 0.25, alpha]]], dtype=np.float64)
    before = straight.copy()
    multiplied = premultiply(straight)
    expected = straight.astype(np.float32)
    expected[..., :3] *= expected[..., 3:4]
    np.testing.assert_array_equal(multiplied, expected)
    restored = unpremultiply(multiplied)
    expected_straight = straight if alpha else np.zeros_like(straight)
    np.testing.assert_allclose(restored, expected_straight, rtol=1e-6)
    np.testing.assert_array_equal(straight, before)
    assert multiplied.dtype == restored.dtype == np.float32
    assert not np.shares_memory(multiplied, straight)
    assert not np.shares_memory(restored, multiplied)


def test_zero_alpha_canonical_and_no_divide_warning():
    buffer = Buffer(np.array([[[4.0, -2.0, 8.0, 0.0]]]))
    np.testing.assert_array_equal(buffer.rgba, np.zeros((1, 1, 4)))
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(buffer.straight(), buffer.rgba)
        np.testing.assert_array_equal(buffer.to_uint8_rgb(), [[[0, 0, 0]]])


@pytest.mark.parametrize("factory", [Buffer, premultiply, unpremultiply])
@pytest.mark.parametrize("shape", [(2, 3), (1, 2, 3), (1, 2, 5), (4,)])
def test_rgba_shape_errors(factory, shape):
    with pytest.raises(ValueError, match="rgba"):
        factory(np.zeros(shape))


@pytest.mark.parametrize("factory", [Buffer, premultiply, unpremultiply])
@pytest.mark.parametrize("dtype", [object, complex, str, bool])
def test_rgba_dtype_errors(factory, dtype):
    with pytest.raises(TypeError, match="rgba"):
        factory(np.ones((1, 1, 4), dtype=dtype))


@pytest.mark.parametrize("factory", [Buffer, premultiply, unpremultiply])
@pytest.mark.parametrize(
    "channel,value",
    [(0, np.nan), (1, np.inf), (2, -np.inf), (3, -0.1), (3, 1.1), (3, 1 + 1e-10)],
)
def test_rgba_nonfinite_and_alpha_range(factory, channel, value):
    rgba = np.ones((1, 1, 4), dtype=np.float64)
    rgba[0, 0, channel] = value
    with pytest.raises(ValueError, match="rgba"):
        factory(rgba)


def test_float32_overflow_fails_explicitly():
    with pytest.raises(ValueError, match="rgba"):
        Buffer(np.array([[[1e300, 0, 0, 1]]]))
    with pytest.raises(ValueError, match="rgba"):
        unpremultiply(np.array([[[1e30, 0, 0, 1e-30]]], dtype=np.float32))
    huge = Buffer(np.array([[[3e38, 0, 0, 0.5]]], dtype=np.float32))
    with pytest.raises(ValueError, match="composite"):
        huge.composite_over(huge)


@pytest.mark.parametrize("offset", [(1.0, 2), (True, 2), (1,), None, "xy"])
def test_offset_validation(offset):
    with pytest.raises((TypeError, ValueError), match="offset"):
        Buffer(pixel(), offset=offset)


@pytest.mark.parametrize("label", ["srgb", "linear", "ocio:ACEScg"])
def test_color_labels_export_raw_working_codes(label):
    buffer = Buffer(pixel((2, -1, 0.5), 1), color_space=label)
    np.testing.assert_array_equal(buffer.to_uint8_rgb(), [[[255, 0, 128]]])
    assert buffer.with_offset((2, 1)).color_space == label


@pytest.mark.parametrize("label", ["", "ocio:", "SRGB", 7, None])
def test_invalid_color_space(label):
    with pytest.raises((TypeError, ValueError), match="color_space"):
        Buffer(pixel(), color_space=label)


def test_strict_rgb_factory_mask_and_source_aliases():
    frame = np.array([[[128, 255, 0], [255, 0, 128]]], dtype=np.uint8)
    mask = np.array([[0.5, 0]])
    buffer = Buffer.from_uint8_rgb(frame, mask)
    expected = np.array([[[128 / 255 * 0.5, 0.5, 0, 0.5], [0, 0, 0, 0]]])
    np.testing.assert_allclose(buffer.rgba, expected, rtol=1e-6)
    frame[:] = 0
    mask[:] = 1
    np.testing.assert_allclose(buffer.rgba, expected, rtol=1e-6)
    with pytest.raises(TypeError, match="frame"):
        Buffer.from_uint8_rgb(frame.astype(float))
    with pytest.raises(ValueError, match="frame"):
        Buffer.from_uint8_rgb(np.zeros((1, 1, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="mask"):
        Buffer.from_uint8_rgb(frame, np.ones((2, 2)))


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[-0.1]]),
        np.array([[1.1]]),
        np.array([[1 + 1e-10]]),
        np.ones((1, 1, 1)),
        np.array([["x"]]),
        np.array([[1j]]),
    ],
)
def test_mask_validation(mask):
    with pytest.raises((TypeError, ValueError), match="mask"):
        Buffer.from_uint8_rgb(np.zeros((1, 1, 3), dtype=np.uint8), mask)


def test_bool_mask_supported():
    buffer = Buffer.from_uint8_rgb(
        np.full((1, 2, 3), 255, np.uint8), np.array([[True, False]])
    )
    np.testing.assert_array_equal(buffer.rgba, [[[1, 1, 1, 1], [0, 0, 0, 0]]])


def test_export_straight_background_and_nearest_even():
    buffer = Buffer(pixel((1, 0.5, 0), 0.5), offset=(-3, 7))
    np.testing.assert_array_equal(buffer.to_uint8_rgb(), [[[255, 128, 0]]])
    np.testing.assert_array_equal(buffer.to_uint8_rgb((0, 0, 1)), [[[128, 64, 128]]])
    ties = Buffer(pixel((0.5 / 255, 1.5 / 255, 2.5 / 255), 1))
    np.testing.assert_array_equal(ties.to_uint8_rgb(), [[[0, 2, 2]]])
    assert buffer.to_uint8_rgb().shape == (1, 1, 3)


@pytest.mark.parametrize(
    "bg", [(0, 0), (0, 0, 255), (-1, 0, 0), (np.inf, 0, 0), (np.nan, 0, 0), ("x", 0, 0)]
)
def test_background_validation(bg):
    with pytest.raises((TypeError, ValueError), match="bg"):
        Buffer(pixel()).to_uint8_rgb(bg)


def test_clip_reads_rgb_and_mask_once_at_source_local_time():
    reads = []

    def read_rgb(t):
        reads.append(("rgb", t))
        return np.full((2, 3, 3), 127.5)

    def read_mask(t):
        reads.append(("mask", t))
        return np.full((2, 3), 0.5)

    clip = SimpleNamespace(
        get_frame=read_rgb,
        start=100,
        is_mask=False,
        mask=SimpleNamespace(get_frame=read_mask, start=500),
    )
    buffer = Buffer.from_clip(clip, -0.25, offset=(-1, 2))
    assert reads == [("rgb", -0.25), ("mask", -0.25)]
    np.testing.assert_array_equal(buffer.rgba[..., :3], np.full((2, 3, 3), 0.25))
    assert buffer.offset == (-1, 2)


@pytest.mark.parametrize("shape", [(3, 1), (1, 3), (3, 3), (0, 0)])
def test_clip_masks_align_top_left_crop_and_transparent_pad(shape):
    mask = np.full(shape, 0.5)
    clip = SimpleNamespace(
        get_frame=lambda t: np.full((2, 2, 3), 255.0),
        mask=SimpleNamespace(get_frame=lambda t: mask),
    )
    buffer = Buffer.from_clip(clip, 0)
    expected = np.zeros((2, 2))
    expected[: min(2, shape[0]), : min(2, shape[1])] = 0.5
    np.testing.assert_array_equal(buffer.rgba[..., 3], expected)
    np.testing.assert_array_equal(buffer.rgba[..., 0], expected)


def test_clip_dynamic_sizes_and_float_codes_without_uint8_intermediate():
    clip = SimpleNamespace(
        get_frame=lambda t: np.full((1, int(t) + 1, 3), 0.5), mask=None
    )
    assert Buffer.from_clip(clip, 0).size == (1, 1)
    buffer = Buffer.from_clip(clip, 2)
    assert buffer.size == (3, 1)
    np.testing.assert_array_equal(
        buffer.rgba[..., 0], np.full((1, 3), np.float32(0.5 / 255))
    )


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((1, 1)),
        np.zeros((1, 1, 4)),
        np.full((1, 1, 3), -1),
        np.full((1, 1, 3), 255 + 1e-8),
        np.full((1, 1, 3), np.nan),
        np.full((1, 1, 3), "x"),
    ],
)
def test_clip_frame_validation(frame):
    clip = SimpleNamespace(get_frame=lambda t: frame, mask=None)
    with pytest.raises((TypeError, ValueError), match="frame"):
        Buffer.from_clip(clip, 0)


@pytest.mark.parametrize("t", [True, "0", np.nan, np.inf])
def test_clip_time_validation(t):
    clip = SimpleNamespace(get_frame=lambda t: np.zeros((1, 1, 3)), mask=None)
    with pytest.raises((TypeError, ValueError), match="t"):
        Buffer.from_clip(clip, t)


def test_mask_clip_and_invalid_discarded_mask_values_rejected():
    frame = np.zeros((1, 1, 3))
    clip = SimpleNamespace(get_frame=lambda t: frame, mask=None, is_mask=True)
    with pytest.raises(ValueError, match="clip"):
        Buffer.from_clip(clip, 0)
    clip.is_mask = False
    clip.mask = SimpleNamespace(get_frame=lambda t: np.array([[0, np.nan]]))
    with pytest.raises(ValueError, match="mask"):
        Buffer.from_clip(clip, 0)


def test_crop_world_intersection_and_empty_requested_offset():
    rgba = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    rgba[..., 3] = 1
    buffer = Buffer(rgba, offset=(-2, 3))
    assert buffer.size == (3, 2)
    assert buffer.bounds == (-2, 3, 1, 5)
    cropped = buffer.crop((-1, 4, 2, 6))
    assert cropped.bounds == (-1, 4, 1, 5)
    np.testing.assert_array_equal(cropped.rgba, rgba[1:2, 1:3])
    for bounds in [(9, 8, 10, 10), (-2, 3, -2, 5)]:
        empty = buffer.crop(bounds)
        assert empty.rgba.shape == (0, 0, 4)
        assert empty.offset == bounds[:2]


def test_asymmetric_pad_expand_and_world_pixel_preservation():
    buffer = Buffer(pixel((2, -1, 3), 1), offset=(-2, 3))
    padded = buffer.pad(left=2, top=1, right=3, bottom=4)
    assert padded.bounds == (-4, 2, 2, 8)
    np.testing.assert_array_equal(padded.rgba[1:2, 2:3], buffer.rgba)
    np.testing.assert_array_equal(padded.crop(buffer.bounds).rgba, buffer.rgba)
    expanded = buffer.expand_to((-4, 2, 2, 8))
    np.testing.assert_array_equal(expanded.rgba, padded.rgba)
    assert buffer.pad() is not buffer
    assert buffer.crop(buffer.bounds) is not buffer
    with pytest.raises(ValueError, match="bounds"):
        buffer.expand_to((-1, 3, 1, 4))


@pytest.mark.parametrize(
    "bounds", [(1, 0, 0, 1), (0, 2, 1, 1), (0, 0, 1), (0.0, 0, 1, 1), (False, 0, 1, 1)]
)
def test_bounds_validation(bounds):
    for method in (Buffer(pixel()).crop, Buffer(pixel()).expand_to):
        with pytest.raises((TypeError, ValueError), match="bounds"):
            method(bounds)


@pytest.mark.parametrize("margin", [-1, True, 0.5])
def test_pad_validation(margin):
    with pytest.raises((TypeError, ValueError), match="left"):
        Buffer(pixel()).pad(left=margin)


@pytest.mark.parametrize(
    "fg_alpha,bg_alpha", [(0, 0), (0, 0.4), (1, 0.4), (0.3, 1), (0.3, 0.6)]
)
def test_over_float64_analytic_formula_and_input_immutability(fg_alpha, bg_alpha):
    foreground = Buffer(pixel((2, -0.5, 0.8), fg_alpha))
    background = Buffer(pixel((-1, 1.5, 0.2), bg_alpha))
    fg_before = foreground.rgba.copy()
    bg_before = background.rgba.copy()
    expected = fg_before.astype(np.float64) + bg_before * (1 - fg_alpha)
    result = foreground.composite_over(background)
    np.testing.assert_allclose(result.rgba, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(foreground.rgba, fg_before)
    np.testing.assert_array_equal(background.rgba, bg_before)
    np.testing.assert_array_equal(
        result.rgba, foreground.composite_over(background).rgba
    )
    assert not result.rgba.flags.writeable


def test_over_union_offsets_disjoint_and_order():
    fg = Buffer(pixel((1, 0, 0), 0.5), offset=(-1, 2))
    bg = Buffer(pixel((0, 0, 1), 1), offset=(1, 3))
    result = fg.composite_over(bg)
    assert result.bounds == (-1, 2, 2, 4)
    np.testing.assert_array_equal(result.rgba[0, 0], fg.rgba[0, 0])
    np.testing.assert_array_equal(result.rgba[1, 2], bg.rgba[0, 0])
    np.testing.assert_array_equal(result.rgba[0, 1], np.zeros(4))
    overlap = bg.with_offset(fg.offset)
    np.testing.assert_array_equal(fg.composite_over(overlap).rgba, [[[0.5, 0, 0.5, 1]]])
    assert not np.array_equal(
        fg.composite_over(overlap).rgba, overlap.composite_over(fg).rgba
    )


def test_empty_identity_expand_and_color_mismatch():
    empty = Buffer(np.zeros((0, 0, 4)), offset=(100, -100))
    buffer = Buffer(pixel(), offset=(-1, 2))
    for result in (empty.composite_over(buffer), buffer.composite_over(empty)):
        assert result.bounds == buffer.bounds
        np.testing.assert_array_equal(result.rgba, buffer.rgba)
        assert result is not buffer
    other_empty = empty.with_offset((3, 4))
    assert empty.composite_over(other_empty).offset == empty.offset
    expanded = empty.expand_to((0, 0, 2, 3))
    assert expanded.bounds == (0, 0, 2, 3)
    np.testing.assert_array_equal(expanded.rgba, np.zeros((3, 2, 4)))
    with pytest.raises(ValueError, match="color_space"):
        buffer.composite_over(Buffer(pixel(), color_space="linear"))
    with pytest.raises(TypeError, match="background"):
        buffer.composite_over(np.zeros((1, 1, 4)))


def test_ten_layers_quantize_only_at_output():
    layer = Buffer.from_uint8_rgb(np.full((1, 1, 3), 3, np.uint8), np.full((1, 1), 0.1))
    accumulator = Buffer.from_uint8_rgb(np.zeros((1, 1, 3), np.uint8))
    oracle = accumulator.rgba.astype(np.float64)
    source = layer.rgba.astype(np.float64)
    for _ in range(10):
        accumulator = layer.composite_over(accumulator)
        oracle = source + oracle * (1 - source[..., 3:4])
    np.testing.assert_allclose(accumulator.rgba, oracle, rtol=1e-6)
    np.testing.assert_array_equal(accumulator.to_uint8_rgb(), [[[2, 2, 2]]])


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0, 1e-30])
def test_export_and_over_spatial_signed_hdr_regression(uniform, alpha):
    """Fast paths preserve analytic values across spatially varying pixels."""
    rng = RenderContext(rng_seed=180).rng_for("buffer-performance-regression")
    rgba = rng.uniform(-3, 4, (7, 9, 4)).astype(np.float32)
    rgba[..., 3] = alpha if uniform else rng.uniform(0, 1, (7, 9))
    rgba[0, 0, 3] = alpha if uniform else 0
    source = Buffer(rgba[::-1, ::-1])
    lower = Buffer(rgba)
    expected = source.rgba + lower.rgba * (np.float32(1) - source.rgba[..., 3:4])
    result = source.composite_over(lower)
    np.testing.assert_allclose(result.rgba, expected, rtol=1e-6, atol=1e-7)
    straight = np.zeros((7, 9, 3), np.float32)
    np.divide(
        source.rgba[..., :3],
        source.rgba[..., 3:4],
        out=straight,
        where=source.rgba[..., 3:4] > 0,
    )
    expected_codes = np.rint(np.clip(straight, 0, 1) * np.float32(255)).astype(np.uint8)
    np.testing.assert_array_equal(source.to_uint8_rgb(), expected_codes)
    for background in ((0, 0, 0), (1, 0.5, 0)):
        rgb = source.rgba[..., :3] + np.asarray(background, np.float32) * (
            np.float32(1) - source.rgba[..., 3:4]
        )
        expected_codes = np.rint(np.clip(rgb, 0, 1) * np.float32(255)).astype(np.uint8)
        np.testing.assert_array_equal(source.to_uint8_rgb(background), expected_codes)
    assert result.rgba.flags.c_contiguous
    assert not result.rgba.flags.writeable
    assert not np.shares_memory(result.rgba, source.rgba)
    assert not np.shares_memory(result.rgba, lower.rgba)


def test_export_straight_overflow_remains_explicit():
    """Export must reject division overflow before clipping finite HDR values."""
    source = Buffer(np.array([[[3e38, 0, 0, 1e-30]]], np.float32))
    with pytest.raises(ValueError, match="rgba"):
        source.to_uint8_rgb()


def test_export_all_nearest_even_ties_and_adjacent_float32_values():
    """Final conversion keeps float32 nearest-even ties and their neighbors."""
    ties = (np.arange(255, dtype=np.float32) + np.float32(0.5)) / np.float32(255)
    values = np.stack(
        (np.nextafter(ties, np.float32(0)), ties, np.nextafter(ties, np.float32(1))),
        axis=-1,
    )[None]
    rgba = np.ones((1, 255, 4), np.float32)
    rgba[..., :3] = values
    expected = np.rint(values * np.float32(255)).astype(np.uint8)
    np.testing.assert_array_equal(Buffer(rgba).to_uint8_rgb(), expected)


@pytest.mark.parametrize("uniform", [True, False])
def test_composite_overflow_spatial_alpha_remains_explicit(uniform):
    """Native arithmetic must retain float32 overflow validation."""
    rgba = np.full((3, 4, 4), np.float32(3e38))
    rgba[..., 3] = 0.5
    if not uniform:
        rgba[0, 0, 3] = 0.4
    source = Buffer(rgba)
    with pytest.raises(ValueError, match="composite"):
        source.composite_over(source)


@pytest.mark.parametrize("escape", ["slice", "asarray", "ndarray", "memoryview"])
def test_released_buffer_keeps_escaped_rgba_storage_alive(escape):
    """Retained NumPy and buffer-protocol views must outlive their Buffer."""
    foreground = Buffer(np.tile(pixel((0.8, -1, 3), 0.4), (5, 7, 1)))
    background = Buffer(np.tile(pixel((0.2, 0.3, 0.4), 0.8), (5, 7, 1)))
    result = foreground.composite_over(background)
    if escape == "slice":
        escaped = result.rgba[::2, ::2]
    elif escape == "asarray":
        escaped = np.asarray(result.rgba)
    elif escape == "ndarray":
        escaped = result.rgba.view(np.ndarray)
    else:
        escaped = memoryview(result.rgba)
    expected = np.asarray(escaped).copy()
    del result
    for index in range(12):
        changing = Buffer(np.tile(pixel((index, -index, 0), 0.3), (5, 7, 1)))
        output = changing.composite_over(background)
        del output
    np.testing.assert_array_equal(np.asarray(escaped), expected)
    assert not np.asarray(escaped).flags.writeable


def test_retained_composite_values_do_not_share_storage():
    """Multiple live composite results own independent immutable pixels."""
    background = Buffer(np.tile(pixel(), (5, 7, 1)))
    results = []
    expected = []
    for index in range(12):
        foreground = Buffer(np.tile(pixel((index, -index, 0), 0.3), (5, 7, 1)))
        result = foreground.composite_over(background)
        expected.append(result.rgba.copy())
        results.append(result)
    for index, result in enumerate(results):
        np.testing.assert_array_equal(result.rgba, expected[index])
        assert not any(
            np.shares_memory(result.rgba, other.rgba) for other in results[index + 1 :]
        )


def test_composite_base_chain_remains_readonly():
    """Allocator ownership must not expose a mutable base ndarray."""
    source = Buffer(np.tile(pixel(), (3, 4, 1)))
    result = source.composite_over(source)
    current = result.rgba
    while isinstance(current, np.ndarray):
        assert not current.flags.writeable
        current = current.base
    if current is not None and hasattr(current, "__array_interface__"):
        assert not np.asarray(current).flags.writeable


@pytest.mark.parametrize("varying", [False, True])
def test_recycled_union_pixels_are_fully_initialized(varying):
    """Fresh union output must clear holes even when allocation is recycled."""
    rgba = np.tile(pixel((2, -1, 3), 0.4), (2, 3, 1))
    if varying:
        rgba[0, 0] = pixel((2, -1, 3), 0.1)[0, 0]
    foreground = Buffer(rgba, offset=(-3, 2))
    background = Buffer(rgba, offset=(1, 4))
    expected = np.zeros((4, 7, 4), np.float32)
    expected[:2, :3] = foreground.rgba
    expected[2:, 4:] = background.rgba
    for _ in range(8):
        result = foreground.composite_over(background)
        np.testing.assert_array_equal(result.rgba, expected)
        del result


def test_large_opposite_hdr_channels_remain_finite_with_fallback():
    """Conservative native bounds must allow finite signed cancellation."""
    foreground = Buffer(np.array([[[3e38, -3e38, 0, 0.5]]], np.float32))
    background = Buffer(np.array([[[-3e38, 3e38, 0, 1]]], np.float32))
    expected = foreground.rgba + background.rgba * np.float32(0.5)
    result = foreground.composite_over(background)
    np.testing.assert_array_equal(result.rgba, expected)
    assert np.isfinite(result.rgba).all()


def test_normalized_composite_export_matches_float32_codebook():
    """Opaque normalized composites quantize every output code exactly once."""
    codes = np.arange(256, dtype=np.uint8).reshape(16, 16)
    frame = np.stack((codes, 255 - codes, np.roll(codes, 7)), axis=-1)
    result = Buffer.from_uint8_rgb(frame)
    for index in range(10):
        layer = Buffer.from_uint8_rgb(
            np.roll(frame, index + 1, axis=1),
            np.full((16, 16), 0.25 + 0.04 * index, np.float32),
        )
        result = layer.composite_over(result)
        straight = result.straight()[..., :3]
        expected = np.rint(np.clip(straight, 0, 1) * np.float32(255)).astype(np.uint8)
        np.testing.assert_array_equal(result.to_uint8_rgb(), expected)
        np.testing.assert_array_equal(result.to_uint8_rgb((0.9, 0.2, 1)), expected)
