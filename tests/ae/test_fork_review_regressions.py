"""Regression cases from the independent review before fork publication."""

import math

import numpy as np

import pytest

from moviepy import VideoClip
from moviepy.ae import Buffer, _geometry
from moviepy.ae.layers import AVLayer
from moviepy.ae.properties import Keyframe, Property
from moviepy.ae.transform import Transform
from moviepy.ae.warp import destination_bounds, warp_buffer


@pytest.mark.parametrize("interpolation", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("rotation", [0, 90, 180])
def test_offset_source_rotation_matches_origin_pixels(interpolation, rotation):
    pixels = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4) / 40
    pixels[..., 3] = 1
    transform = Transform(rotation=rotation, interpolation=interpolation)
    origin = transform.apply(Buffer(pixels))
    shifted = transform.apply(Buffer(pixels, offset=(10, -20)))
    assert shifted.offset == (origin.offset[0] + 10, origin.offset[1] - 20)
    np.testing.assert_allclose(shifted.rgba, origin.rgba, atol=1e-6)


def test_explicit_world_roi_uses_source_offset():
    source = Buffer(np.ones((3, 3, 4), dtype=np.float32), offset=(10, 20))
    result = warp_buffer(source, np.eye(3), bounds=(11, 21, 13, 23))
    assert result.bounds == (11, 21, 13, 23)
    np.testing.assert_array_equal(result.rgba, np.ones((2, 2, 4), np.float32))


@pytest.mark.parametrize("scale", [(0, 0), (0, 100), (100, 0)])
@pytest.mark.parametrize("interpolation", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("bounds", [None, (10, 20, 13, 23)])
def test_zero_scale_has_no_pixels_even_with_rotation_or_roi(
    scale, interpolation, bounds
):
    source = Buffer(np.ones((3, 3, 4), dtype=np.float32))
    transform = Transform(scale=scale, rotation=45, interpolation=interpolation)
    result = transform.apply(source, bounds=bounds)
    assert result.size == (0, 0)
    assert result.rgba.size == 0
    if bounds is not None:
        assert result.offset == bounds[:2]
    matrix = transform.matrix_at(0, size=source.size)
    rectangle = destination_bounds(matrix, source.size, interpolation=interpolation)
    assert rectangle[0] == rectangle[2] and rectangle[1] == rectangle[3]


def _coded_clip():
    calls = []
    mask_calls = []

    def frame(t):
        assert 0 <= t < 2
        calls.append(t)
        return np.full((1, 1, 3), 30 + int(t * 4) * 20, dtype=np.uint8)

    def mask(t):
        assert 0 <= t < 2
        mask_calls.append(t)
        return np.full((1, 1), 0.25 + t / 8)

    clip = VideoClip(frame, duration=2).with_fps(4)
    clip.mask = VideoClip(mask, is_mask=True, duration=2).with_fps(4)
    calls.clear()
    mask_calls.clear()
    return clip, calls, mask_calls


@pytest.mark.parametrize("stretch", [-100, -200])
@pytest.mark.parametrize("start", [-1, 0, 2])
def test_reverse_footage_reads_descending_frames_masks_and_animation(stretch, start):
    clip, calls, mask_calls = _coded_clip()
    position = Property((0, 0), keyframes=[(0, (0, 0)), (2, (20, 0))])
    layer = AVLayer(
        clip,
        start_time=start,
        in_point=start,
        stretch=stretch,
        transform=Transform(anchor_point=(0, 0), position=position),
    )
    assert layer.out_point == start + 2 * abs(stretch) / 100
    for elapsed, expected_time in [(0, 1.75), (0.5, 1.5), (1, 1), (1.5, 0.5)]:
        time = start + elapsed * abs(stretch) / 100
        result = layer.render(time)
        expected_pixel = 30 + int(expected_time * 4) * 20
        np.testing.assert_array_equal(result.to_uint8_rgb(), [[[expected_pixel] * 3]])
        assert result.rgba[0, 0, 3] == pytest.approx(0.25 + expected_time / 8)
        assert result.offset[0] == round((2 - elapsed) * 10)
    assert calls == mask_calls == [1.75, 1.5, 1, 0.5]
    assert layer.render(layer.out_point) is None


def test_reverse_window_and_stretch_edit_use_the_current_source_clock():
    clip, calls, _ = _coded_clip()
    layer = AVLayer(clip, start_time=1, in_point=1.5, out_point=2.5, stretch=-100)
    layer.render(1.5)
    assert calls[-1] == 1.5
    layer.stretch = -200
    layer.render(2)
    assert calls[-1] == 1.5
    layer.stretch = 100
    layer.render(2)
    assert calls[-1] == 1


def test_reverse_footage_requires_a_finite_source_end():
    clip = VideoClip(lambda t: np.zeros((1, 1, 3), dtype=np.uint8))
    layer = AVLayer(clip, stretch=-100, out_point=2)
    with pytest.raises(ValueError, match="finite.*duration"):
        layer.render(1)


@pytest.mark.parametrize("name", ["anchor_point", "position"])
def test_invalid_transform_assignment_keeps_automatic_state(name):
    transform = Transform()
    before = transform.to_dict()
    with pytest.raises((TypeError, ValueError)):
        setattr(transform, name, (1,))
    assert transform.to_dict() == before
    np.testing.assert_array_equal(transform.matrix_at(0, size=(3, 3)), np.eye(3))


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("opacity", [0, 0.1, 1])
def test_cubic_intermediate_overflow_is_rejected_before_opacity(sign, opacity):
    pixels = np.ones((8, 8, 4), dtype=np.float32)
    pixels[..., :3] = 0
    pixels[:, 4:, :3] = sign * np.finfo(np.float32).max * np.float32(0.99)
    matrix = np.array([[1, 0, 0.3], [0, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(ValueError, match="finite"):
        warp_buffer(Buffer(pixels), matrix, opacity, interpolation="cubic")


def test_curved_auto_orient_keeps_boundary_tangents_during_holds():
    position = Property(
        value_type="vec2",
        spatial=True,
        keyframes=[
            Keyframe(0, (0, 0), out_tangent=(0, 10)),
            Keyframe(1, (10, 10), in_tangent=(-10, 0), interp="hold"),
            Keyframe(2, (10, 10)),
        ],
    )
    transform = Transform(position=position, auto_orient=True)
    initial = transform.rotation_at(0)
    final = transform.rotation_at(math.nextafter(1, 0))
    assert initial == pytest.approx(90, abs=0.02)
    assert final == pytest.approx(0, abs=0.02)
    assert transform.rotation_at(-0.1) == pytest.approx(initial, abs=0.02)
    for time in [1, 1.001, 1.5, 2, 3]:
        assert transform.rotation_at(time) == pytest.approx(final, abs=0.02)


@pytest.mark.parametrize("row", [(0.1, 0, 1), (0, 0, 2), (0, 0, 0)])
def test_warp_rejects_non_affine_homogeneous_row(row):
    matrix = np.eye(3)
    matrix[2] = row
    with pytest.raises(ValueError, match="affine"):
        warp_buffer(Buffer(np.ones((1, 1, 4), np.float32)), matrix)


def test_translation_allocation_honours_the_pixel_budget(monkeypatch):
    source = Buffer(np.ones((2, 2, 4), dtype=np.float32))
    monkeypatch.setattr(_geometry, "MAX_RENDER_PIXELS", 3)
    with pytest.raises(ValueError, match="render budget"):
        warp_buffer(source, np.eye(3), opacity=0.5)


def test_translation_checks_the_whole_destination_rectangle():
    matrix = np.eye(3)
    matrix[0, 2] = _geometry.DIMENSION_LIMIT
    with pytest.raises(ValueError, match="representable"):
        warp_buffer(Buffer(np.ones((1, 1, 4), np.float32)), matrix)


def test_reverse_starts_on_the_last_frame_for_an_unaligned_duration():
    calls = []

    def frame(t):
        assert 0 <= t < 0.9
        calls.append(t)
        return np.full((1, 1, 3), int(t * 4) * 50, dtype=np.uint8)

    clip = VideoClip(frame, duration=0.9).with_fps(4)
    layer = AVLayer(clip, stretch=-100)
    calls.clear()
    for time in [0, 0.0001, 0.1]:
        np.testing.assert_array_equal(layer.render(time).to_uint8_rgb(), [[[150] * 3]])
    assert calls[0] == 0.75


def test_half_pixel_filter_support_is_invariant_under_large_world_origins():
    source = Buffer(np.ones((1, 3, 4), np.float32))
    small = np.array([[1, 0, 0.5], [0, 1, 0], [0, 0, 1]], dtype=float)
    large = small.copy()
    large[0, 2] += 10**9
    near = warp_buffer(source, small)
    far = warp_buffer(source, large)
    assert far.offset == (near.offset[0] + 10**9, near.offset[1])
    assert far.size == near.size == (4, 1)
    np.testing.assert_array_equal(far.rgba, near.rgba)


def test_nearest_final_bounds_are_range_checked_after_snapping():
    source = Buffer(np.ones((1, 1, 4), np.float32))
    matrix = np.eye(3)
    matrix[0, 2] = _geometry.DIMENSION_LIMIT - 0.5
    with pytest.raises(ValueError, match="representable"):
        destination_bounds(matrix, source.size, interpolation="nearest")
    with pytest.raises(ValueError, match="representable"):
        warp_buffer(source, matrix, interpolation="nearest")


def test_unrepresentable_inverse_is_rejected_before_opencv(monkeypatch):
    import cv2

    def forbidden(*args, **kwargs):
        raise AssertionError("an unsafe inverse reached OpenCV")

    monkeypatch.setattr(cv2, "warpAffine", forbidden)
    matrix = np.diag([1e-200, 1e-200, 1])
    source = Buffer(np.ones((1, 1, 4), np.float32))
    with pytest.raises(ValueError, match="inverse"):
        warp_buffer(source, matrix, interpolation="nearest", bounds=(0, 0, 3, 3))


def test_reciprocal_extreme_axes_reject_instead_of_silently_collapsing():
    matrix = np.diag([1e200, 1e-200, 1])
    source = Buffer(np.ones((1, 1, 4), np.float32))
    with pytest.raises(ValueError, match="inverse.*coordinate range"):
        warp_buffer(source, matrix, interpolation="nearest", bounds=(0, 0, 1, 1))
