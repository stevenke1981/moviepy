"""Fast paths in the AE core must match straightforward reference arithmetic."""

import zlib

import numpy as np

import pytest

from moviepy.ae import Buffer
from moviepy.ae._accumulator import _Accumulator
from moviepy.ae.blend.modes import BlendMode
from moviepy.ae.buffer import (
    _arithmetic,
    _background,
    _import_metadata,
    _nonzero_box,
    _pixel_metadata,
    _rgb_rgba,
)


def _premultiplied(rng, shape, *, hdr=False, alpha=None):
    """Random premultiplied RGBA with partial, zero and one alpha values."""
    height, width = shape
    if alpha is None:
        a = rng.choice([0.0, 1.0, 0.25, 0.5, 0.999], (height, width))
        a[rng.random((height, width)) < 0.3] = rng.random()
    else:
        a = np.full((height, width), alpha)
    rgb = rng.random((height, width, 3)) * (4.0 if hdr else 1.0)
    if not hdr:
        rgb *= a[..., None]
    else:
        rgb *= a[..., None] > 0
    return np.concatenate([rgb, a[..., None]], axis=-1).astype(np.float32)


def _reference_uint8(buffer, bg):
    """The original whole-frame export arithmetic."""
    import cv2

    source = buffer.rgba
    rgb = np.empty((*source.shape[:2], 3), np.float32)
    if not rgb.size:
        return rgb.astype(np.uint8)
    cv2.mixChannels([source], [rgb], [0, 0, 1, 1, 2, 2])
    alpha = buffer._uniform_alpha
    background = None if bg is None else _background(bg)
    with _arithmetic("rgba"):
        if background is None:
            if alpha is not None:
                if alpha > 0 and alpha != 1:
                    np.divide(rgb, alpha, out=rgb)
            else:
                np.divide(rgb, source[..., 3:4], out=rgb, where=source[..., 3:4] > 0)
        else:
            np.add(rgb, background * (1 - source[..., 3:4]), out=rgb)
    np.clip(rgb, 0, 1, out=rgb)
    return cv2.convertScaleAbs(rgb, alpha=255)


@pytest.mark.parametrize("shape", [(0, 5), (1, 1), (7, 13), (40, 1000), (97, 701)])
@pytest.mark.parametrize("kind", ["random", "hdr", "alpha0", "alpha1", "alpha_half"])
@pytest.mark.parametrize("bg", [None, (0.1, 0.5, 0.9), (0.0, 0.0, 0.0)])
@pytest.mark.parametrize("color_space", ["srgb", "linear"])
def test_uint8_export_matches_reference(shape, kind, bg, color_space):
    rng = np.random.default_rng(zlib.crc32(repr((shape, kind)).encode()))
    alpha = {"alpha0": 0.0, "alpha1": 1.0, "alpha_half": 0.5}.get(kind)
    array = _premultiplied(rng, shape, hdr=kind == "hdr", alpha=alpha)
    buffer = Buffer(array, (3, -2), color_space)
    if buffer._uniform_alpha == 1 and buffer._unit_premultiplied:
        return  # unchanged direct convertScaleAbs path
    np.testing.assert_array_equal(buffer.to_uint8_rgb(bg), _reference_uint8(buffer, bg))


def test_uint8_export_constant_zero_buffer_fills_background():
    buffer = Buffer(np.zeros((300, 400, 4), np.float32))
    assert buffer._uniform_alpha == 0 and buffer._value_bound == 0
    out = buffer.to_uint8_rgb((0.2, 0.4, 0.6))
    np.testing.assert_array_equal(out, _reference_uint8(buffer, (0.2, 0.4, 0.6)))
    assert out.shape == (300, 400, 3) and out.dtype == np.uint8


def test_uint8_export_overflow_still_reports():
    array = np.zeros((4, 4, 4), np.float32)
    array[..., 3] = 1e-30
    array[..., 0] = 1e10
    with pytest.raises(ValueError, match="rgba arithmetic"):
        Buffer(array).to_uint8_rgb()


def _reference_normal(layers):
    """Float32 source-over on the union of the layers' bounds."""
    left = min(b.offset[0] for b in layers)
    top = min(b.offset[1] for b in layers)
    right = max(b.bounds[2] for b in layers)
    bottom = max(b.bounds[3] for b in layers)
    canvas = np.zeros((bottom - top, right - left, 4), np.float32)
    for layer in layers:
        x, y = layer.offset[0] - left, layer.offset[1] - top
        w, h = layer.size
        region = canvas[y : y + h, x : x + w]
        region[...] = region * (np.float32(1) - layer.rgba[..., 3:4]) + layer.rgba
    return (left, top), canvas


def _accumulated(layers):
    acc = _Accumulator(layers[0].bounds, "srgb")
    for layer in layers:
        acc.blend(layer, BlendMode.NORMAL)
    return acc.snapshot()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("hdr", [False, True])
def test_accumulator_normal_matches_reference_stack(seed, hdr):
    rng = np.random.default_rng(900 + seed)
    full = (130, 270)
    sparse = np.zeros((*full, 4), np.float32)
    sparse[40:70, 30:200] = _premultiplied(rng, (30, 170), hdr=hdr)
    clear = np.zeros((*full, 4), np.float32)
    layers = [
        Buffer(_premultiplied(rng, full, hdr=hdr, alpha=1.0 if seed % 2 else None)),
        Buffer(_premultiplied(rng, full, hdr=hdr, alpha=0.5)),
        Buffer(sparse),
        Buffer(clear),
        Buffer(_premultiplied(rng, (31, 47), hdr=hdr), (-7 + seed, 91 - seed)),
        Buffer(_premultiplied(rng, full, hdr=hdr, alpha=0.0)),
        Buffer(_premultiplied(rng, (50, 80), hdr=hdr), (101, 3)),
    ]
    offset, expected = _reference_normal(layers)
    result = _accumulated(layers)
    assert result.offset == offset
    np.testing.assert_allclose(result.rgba, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("order", ["sparse_first", "opaque_then_sparse"])
def test_accumulator_sparse_overlay_over_opaque(order):
    rng = np.random.default_rng(31)
    base = Buffer(_premultiplied(rng, (260, 300), alpha=1.0))
    overlay = np.zeros((260, 300, 4), np.float32)
    overlay[200:240, 20:280] = _premultiplied(rng, (40, 260))
    layers = [Buffer(overlay), base]
    if order != "sparse_first":
        layers.reverse()
    _, expected = _reference_normal(layers)
    np.testing.assert_allclose(_accumulated(layers).rgba, expected, rtol=0, atol=1e-6)


def test_accumulator_overflow_guard_survives_fast_paths():
    big = np.zeros((300, 300, 4), np.float32)
    big[..., 0] = 3.0e38
    big[..., 3] = 0.5
    big[100:120, 100:150, 3] = 0.25
    layers = [Buffer(big), Buffer(big)]
    with pytest.raises(ValueError, match="composite arithmetic"):
        _accumulated(layers)


def test_accumulator_zero_source_keeps_metadata_and_pixels():
    rng = np.random.default_rng(5)
    base = Buffer(_premultiplied(rng, (20, 30), alpha=0.5))
    zero = Buffer(np.zeros((20, 30, 4), np.float32))
    result = _accumulated([base, zero])
    np.testing.assert_array_equal(result.rgba, base.rgba)
    assert result._uniform_alpha == 0.5


@pytest.mark.parametrize("box", [(5, 9, 3, 4), (0, 300, 0, 300), (7, 8, 250, 251)])
def test_nonzero_box_matches_bruteforce(box):
    top, bottom, left, right = box
    array = np.zeros((300, 300, 4), np.float32)
    rng = np.random.default_rng(2)
    array[top:bottom, left:right] = rng.random((bottom - top, right - left, 4)) + 1
    assert _nonzero_box(array) == box


def test_nonzero_box_of_empty_content():
    top, bottom, _, _ = _nonzero_box(np.zeros((300, 300, 4), np.float32))
    assert top == bottom


def test_nonzero_box_counts_hidden_rgb_at_zero_alpha():
    array = np.zeros((300, 300, 4), np.float32)
    array[10, 20, 0] = -1.0  # additive blend modes can leave rgb at alpha 0
    array[200, 40, 3] = 0.5
    assert _nonzero_box(array) == (10, 201, 20, 41)


def test_nonzero_box_small_arrays_are_whole():
    assert _nonzero_box(np.zeros((3, 4, 4), np.float32)) == (0, 3, 0, 4)


@pytest.mark.parametrize("mask_kind", [None, "uniform0", "uniform", "one", "random"])
@pytest.mark.parametrize("shape", [(0, 4), (9, 33), (64, 700)])
def test_import_metadata_equals_pixel_scan(mask_kind, shape):
    rng = np.random.default_rng(77)
    frame = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    mask = {
        None: None,
        "uniform0": np.zeros(shape, np.float32),
        "uniform": np.full(shape, 0.3, np.float32),
        "one": np.ones(shape, np.float32),
        "random": rng.random(shape).astype(np.float32),
    }[mask_kind]
    fast = _import_metadata(frame.shape, mask)
    slow = _pixel_metadata(_rgb_rgba(frame, mask))
    assert (fast[0] is None) == (slow[0] is None)
    if fast[0] is not None:
        assert fast[0] == slow[0]
    assert fast[1] == slow[1] and fast[2] == slow[2]


@pytest.mark.parametrize("shape", [(5, 7), (40, 900)])
def test_masked_import_matches_whole_frame_arithmetic(shape):
    rng = np.random.default_rng(8)
    frame = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    mask = rng.random(shape).astype(np.float32)
    expected = np.empty((*shape, 4), np.float32)
    expected[..., 3] = mask
    np.divide(frame, np.float32(255), out=expected[..., :3], casting="unsafe")
    expected[..., :3] *= expected[..., 3:4]
    assert _rgb_rgba(frame, mask).tobytes() == expected.tobytes()


def test_crop_of_whole_buffer_shares_protected_pixels():
    rng = np.random.default_rng(4)
    buffer = Buffer(_premultiplied(rng, (30, 40)), (5, 6))
    same = buffer.crop(buffer.bounds)
    assert same is not buffer and not same.rgba.flags.writeable
    np.testing.assert_array_equal(same.rgba, buffer.rgba)
    assert same.offset == buffer.offset
    assert same._value_bound == buffer._value_bound
    assert buffer.crop((0, 0, 100, 100)).bounds == buffer.bounds
    part = buffer.crop((10, 10, 20, 20))
    np.testing.assert_array_equal(part.rgba, buffer.rgba[4:14, 5:15])


def test_whole_buffer_crop_rescans_conservative_metadata():
    """A crossfade source must keep the exact summaries (and so the exact path).

    Metadata inherited from a mixed-bounds composite is conservative; reusing
    it selected a different float32 path and flipped rounding by one code.
    """
    rng = np.random.default_rng(26)
    base = Buffer(_premultiplied(rng, (64, 80), alpha=1.0))
    patch = Buffer(_premultiplied(rng, (20, 30), alpha=0.4), (10, 12))
    mixed = patch.composite_over(base)
    assert mixed._uniform_alpha is None  # conservative: bounds differ
    cropped = mixed.crop(mixed.bounds)
    exact = _pixel_metadata(mixed.rgba)
    assert cropped._uniform_alpha == exact[0] == 1
    assert cropped._value_bound == exact[1]
    assert cropped._unit_premultiplied == exact[2]
    fade = Buffer(_premultiplied(rng, (64, 80), alpha=0.37))
    got = _accumulated([cropped, fade]).rgba
    ref = _accumulated([Buffer(mixed.rgba.copy()), fade]).rgba
    assert got.tobytes() == ref.tobytes()
