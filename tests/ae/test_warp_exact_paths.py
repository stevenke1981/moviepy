"""Exact nearest gathers and quarter turns, with and without explicit bounds."""

import math

import numpy as np

import pytest

import moviepy.ae.warp as warp
from moviepy import VideoClip
from moviepy.ae import Buffer, Composition
from moviepy.ae.transform import Transform
from moviepy.ae.warp import destination_bounds, warp_buffer


def _pixels(height, width, seed=0):
    rng = np.random.default_rng(seed)
    pixels = rng.random((height, width, 4), dtype=np.float32)
    pixels[..., :3] *= pixels[..., 3:]
    return pixels


def _reference_nearest(buffer, matrix, bounds):
    """Gather floor(x + 0.5) source pixels pixel by pixel in world space."""
    inverse = np.linalg.inv(matrix)
    left, top, right, bottom = bounds
    output = np.zeros((bottom - top, right - left, 4), dtype=np.float32)
    for row in range(bottom - top):
        for column in range(right - left):
            point = inverse @ (left + column, top + row, 1.0)
            x = math.floor(point[0] - buffer.offset[0] + 0.5)
            y = math.floor(point[1] - buffer.offset[1] + 0.5)
            if 0 <= x < buffer.size[0] and 0 <= y < buffer.size[1]:
                output[row, column] = buffer.rgba[y, x]
    return output


def _rotation(degrees, center):
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    cx, cy = center
    return np.array(
        [
            [cos, -sin, cx - cos * cx + sin * cy],
            [sin, cos, cy - sin * cx - cos * cy],
            [0.0, 0.0, 1.0],
        ]
    )


NEAREST_MATRICES = {
    "upscale-half-pixel": np.array([[2.0, 0, 0.5], [0, 4.0, -1.5], [0, 0, 1]]),
    "downscale-ties": np.array([[0.5, 0, 0.25], [0, 0.5, 0.75], [0, 0, 1]]),
    "mirror-half-pixel": np.array([[-1.0, 0, 6.5], [0, 2.0, 0.5], [0, 0, 1]]),
    "half-pixel-shift": np.array([[1.0, 0, 2.5], [0, 1.0, -0.5], [0, 0, 1]]),
    "rotate-10": _rotation(10, (4.5, 3.0)),
    "rotate-33-scaled": _rotation(33, (2.0, 5.0)) @ np.diag([1.5, 0.75, 1.0]),
    "shear": np.array([[1.0, 0.5, 0.5], [0.25, 1.0, 0.0], [0, 0, 1]]),
}


@pytest.mark.parametrize("name", sorted(NEAREST_MATRICES))
def test_nearest_matches_floor_half_up_reference(name):
    matrix = NEAREST_MATRICES[name]
    buffer = Buffer(_pixels(7, 9), offset=(2, -3))
    result = warp_buffer(buffer, matrix, interpolation="nearest")
    expected = _reference_nearest(buffer, matrix, result.bounds)
    np.testing.assert_array_equal(result.rgba, expected)


@pytest.mark.parametrize("name", sorted(NEAREST_MATRICES))
def test_nearest_roi_matches_the_reference_inside_explicit_bounds(name):
    matrix = NEAREST_MATRICES[name]
    buffer = Buffer(_pixels(7, 9, seed=1), offset=(2, -3))
    natural = destination_bounds(
        matrix, buffer.size, buffer.offset, interpolation="nearest"
    )
    bounds = (natural[0] - 2, natural[1] + 1, natural[2] - 3, natural[3] + 2)
    result = warp_buffer(buffer, matrix, interpolation="nearest", bounds=bounds)
    assert result.bounds == bounds
    np.testing.assert_array_equal(
        result.rgba, _reference_nearest(buffer, matrix, bounds)
    )


@pytest.mark.parametrize(
    "matrix",
    [
        np.array([[3.0, 0, 0.5], [0, 3.0, -1.5], [0, 0, 1]]),
        np.array([[1 / 3, 0, 0.5], [0, 2 / 3, 0.25], [0, 0, 1]]),
        _rotation(30, (1.5, 2.5)) @ np.diag([3.0, 3.0, 1.0]),
    ],
)
def test_nearest_roi_equals_a_crop_of_the_full_render(matrix):
    buffer = Buffer(_pixels(7, 9, seed=5), offset=(-3, 4))
    full = warp_buffer(buffer, matrix, interpolation="nearest")
    left, top, right, bottom = full.bounds
    for bounds in [
        (left + 1, top + 2, right - 1, bottom - 2),
        (left + 3, top + 1, left + 5, bottom),
        (left - 2, top - 2, right + 2, bottom + 2),
    ]:
        part = warp_buffer(buffer, matrix, interpolation="nearest", bounds=bounds)
        np.testing.assert_array_equal(part.rgba, _padded(full, bounds))


def test_two_pixel_solid_at_half_pixel_stays_two_pixels_wide():
    solid = Buffer(np.ones((1, 2, 4), dtype=np.float32))
    shift = np.array([[1.0, 0, 0.5], [0, 1.0, 0], [0, 0, 1]])
    result = warp_buffer(solid, shift, interpolation="nearest")
    assert int(result.rgba[..., 3].sum()) == 2


def _large_nearest_cases():
    buffer = Buffer(_pixels(300, 420, seed=2), offset=(-7, 5))
    return buffer, [_rotation(10, (200.0, 150.0)), np.diag([2.0, 1.5, 1.0])]


def test_strips_threads_and_numpy_fallback_agree(monkeypatch):
    import cv2

    buffer, matrices = _large_nearest_cases()
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 4)
    threaded = [warp_buffer(buffer, m, interpolation="nearest") for m in matrices]
    monkeypatch.setattr(cv2, "getNumThreads", lambda: 1)
    serial = [warp_buffer(buffer, m, interpolation="nearest") for m in matrices]
    monkeypatch.setattr(warp, "_REMAP_LIMIT", 64)
    fallback = [warp_buffer(buffer, m, interpolation="nearest") for m in matrices]
    for first, second, third in zip(threaded, serial, fallback):
        assert first.bounds == second.bounds == third.bounds
        np.testing.assert_array_equal(first.rgba, second.rgba)
        np.testing.assert_array_equal(first.rgba, third.rgba)


def test_large_nearest_rotation_matches_reference_on_sampled_rows():
    buffer, matrices = _large_nearest_cases()
    result = warp_buffer(buffer, matrices[0], interpolation="nearest")
    left, top, right, _ = result.bounds
    for row in (0, 127, 128, 255, result.size[1] - 1):
        band = (left, top + row, right, top + row + 1)
        expected = _reference_nearest(buffer, matrices[0], band)
        np.testing.assert_array_equal(result.rgba[row : row + 1], expected)


def _forbid_resampling(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a quarter turn reached the resampler")

    monkeypatch.setattr(warp, "_resample", forbidden)


QUARTER_TURNS = [(1, False), (2, False), (3, False), (0, True), (1, True)]


def _quarter_matrix(turns, mirror):
    cos = round(math.cos(turns * math.pi / 2))
    sin = round(math.sin(turns * math.pi / 2))
    linear = np.array([[cos, -sin], [sin, cos]], dtype=float)
    if mirror:
        linear = linear @ np.diag([-1.0, 1.0])
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = (40.0, -11.0)
    return matrix


def _padded(full, bounds):
    """Crop a buffer into bounds, padding transparent pixels outside it."""
    output = np.zeros((bounds[3] - bounds[1], bounds[2] - bounds[0], 4), np.float32)
    x0, y0 = max(full.bounds[0], bounds[0]), max(full.bounds[1], bounds[1])
    x1, y1 = min(full.bounds[2], bounds[2]), min(full.bounds[3], bounds[3])
    if x0 < x1 and y0 < y1:
        output[
            y0 - bounds[1] : y1 - bounds[1], x0 - bounds[0] : x1 - bounds[0]
        ] = full.rgba[
            y0 - full.bounds[1] : y1 - full.bounds[1],
            x0 - full.bounds[0] : x1 - full.bounds[0],
        ]
    return output


@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
@pytest.mark.parametrize("turns, mirror", QUARTER_TURNS)
@pytest.mark.parametrize("opacity", [1.0, 0.5])
def test_quarter_turn_roi_is_an_exact_crop_or_pad(
    monkeypatch, interpolation, turns, mirror, opacity
):
    _forbid_resampling(monkeypatch)
    pixels = _pixels(9, 16, seed=3)
    buffer = Buffer(pixels, offset=(4, -2))
    matrix = _quarter_matrix(turns, mirror)
    full = warp_buffer(buffer, matrix, opacity, interpolation=interpolation)
    expected = pixels[:, ::-1] if mirror else pixels
    expected = np.rot90(expected, -turns) * np.float32(opacity)
    np.testing.assert_array_equal(full.rgba, expected)
    left, top, right, bottom = full.bounds
    regions = [
        (left + 1, top + 2, right - 3, bottom - 1),
        (left - 2, top - 1, right + 3, bottom + 2),
        (left - 5, top + 3, left + 4, top + 6),
        (right + 1, bottom + 1, right + 4, bottom + 3),
    ]
    for bounds in regions:
        part = warp_buffer(
            buffer, matrix, opacity, interpolation=interpolation, bounds=bounds
        )
        assert part.bounds == bounds
        np.testing.assert_array_equal(part.rgba, _padded(full, bounds))


def test_quarter_turn_padding_drops_the_uniform_alpha_summary():
    buffer = Buffer(np.ones((4, 6, 4), dtype=np.float32))
    matrix = _quarter_matrix(1, False)
    natural = warp_buffer(buffer, matrix, interpolation="linear")
    left, top = natural.bounds[:2]
    inside = warp_buffer(
        buffer, matrix, interpolation="linear", bounds=(left, top, left + 2, top + 2)
    )
    padded = warp_buffer(
        buffer,
        matrix,
        interpolation="linear",
        bounds=(left - 3, top - 3, left + 9, top + 9),
    )
    assert natural._uniform_alpha == 1.0
    assert inside._uniform_alpha == 1.0
    assert padded._uniform_alpha is None
    np.testing.assert_array_equal(padded.rgba, _padded(natural, padded.bounds))


def test_quarter_turn_output_does_not_alias_the_source():
    pixels = _pixels(3, 5)
    buffer = Buffer(pixels)
    result = warp_buffer(buffer, np.eye(3), interpolation="linear", bounds=(0, 0, 5, 3))
    np.testing.assert_array_equal(result.rgba, pixels)
    assert not np.shares_memory(result.rgba, buffer.rgba)


def _rotated_scene(interpolation):
    image = (np.random.default_rng(4).random((10, 16, 3)) * 255).astype(np.uint8)
    comp = Composition(size=(20, 12), fps=10, duration=1)
    layer = comp.add_clip(
        VideoClip(lambda t: image, duration=1),
        transform=Transform(rotation=90, interpolation=interpolation),
    )
    return comp, layer


@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
def test_layer_rotated_90_and_clipped_by_comp_and_roi_is_bit_exact(
    monkeypatch, interpolation
):
    _forbid_resampling(monkeypatch)
    comp, layer = _rotated_scene(interpolation)
    natural = layer.render(0)
    layer.transform.rotation = 0
    upright = layer.render(0)
    layer.transform.rotation = 90
    np.testing.assert_array_equal(natural.rgba, np.rot90(upright.rgba, -1))
    # The 10x16 rotated layer overhangs the 20x12 comp, so the renderer clips it
    # and passes explicit bounds to warp_buffer.
    comp_rect = (0, 0, *comp.size)
    assert natural.bounds != comp_rect
    full = comp.render_buffer(0)
    assert full.bounds == comp_rect
    np.testing.assert_array_equal(full.rgba, _padded(natural, comp_rect))
    for roi in [(3, 2, 17, 11), (0, 0, 7, 5), (13, 6, 20, 12), (8, 0, 9, 12)]:
        part = comp.render_buffer(0, bounds=roi)
        assert part.bounds == roi
        np.testing.assert_array_equal(part.rgba, full.crop(roi).rgba)
