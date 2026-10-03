"""WS-02 transform tests: matrix order, warping, opacity and validation."""

import math

import numpy as np
import pytest

from moviepy.ae import Buffer, RenderContext
from moviepy.ae.properties import Expression, Keyframe, Property
from moviepy.ae.transform import Transform, affine_matrix
from moviepy.ae.warp import warp_buffer


def make_buffer(width=6, height=6, offset=(0, 0), alpha=1.0):
    """Return an asymmetric opaque test Buffer with pixel-center coordinates."""
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            code = (row * width + col) % 7
            rgba[row, col] = (code / 7.0 + 0.01, 0.25, col / max(width - 1, 1), alpha)
    rgba[..., :3] *= rgba[..., 3:4]
    return Buffer(rgba, offset=offset)


def single_pixel(size=7):
    """Return an odd-sized Buffer holding one bright center pixel."""
    rgba = np.zeros((size, size, 4), dtype=np.float32)
    center = (size - 1) // 2
    rgba[center, center] = (1.0, 1.0, 1.0, 1.0)
    return Buffer(rgba), center


# --------------------------------------------------------------------------- #
# matrix construction


def test_matrix_matches_ae_order_by_hand():
    matrix = affine_matrix(
        position=(10.0, 20.0), rotation=90.0, scale=(50.0, 200.0), anchor=(4.0, 6.0)
    )
    radians = math.radians(90.0)
    cos, sin = math.cos(radians), math.sin(radians)
    sx, sy = 0.5, 2.0
    expected = np.array(
        [
            [sx * cos, -sy * sin, 10.0 - (sx * cos * 4.0 - sy * sin * 6.0)],
            [sx * sin, sy * cos, 20.0 - (sx * sin * 4.0 + sy * cos * 6.0)],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(matrix, expected, atol=1e-12)


def test_default_transform_is_identity_for_any_size():
    transform = Transform()
    for size in ((1, 1), (6, 4), (33, 17)):
        matrix = transform.matrix_at(0.0, size=size)
        np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)


def test_auto_anchor_is_the_pixel_center_convention():
    transform = Transform()
    assert transform.resolve_anchor((6, 5)) == (2.5, 2.0)
    assert transform.resolve_anchor((7, 7), offset=(10, 20)) == (13.0, 23.0)


def test_auto_position_resolves_to_the_anchor():
    transform = Transform(anchor_point=(3.0, 4.0))
    assert transform.resolve_position(0.0, (3.0, 4.0)) == (3.0, 4.0)
    transform.position = (11.0, 12.0)
    assert transform.resolve_position(0.0, (3.0, 4.0)) == (11.0, 12.0)


def test_matrix_requires_a_size_for_auto_anchor():
    with pytest.raises(ValueError, match="size"):
        Transform().matrix_at(0.0)


def test_rotation_maps_the_anchor_onto_the_position():
    transform = Transform(position=(40.0, 50.0), anchor_point=(3.0, 2.0), rotation=37.0)
    matrix = transform.matrix_at(0.0)
    mapped = matrix @ np.array([3.0, 2.0, 1.0])
    np.testing.assert_allclose(mapped[:2], (40.0, 50.0), atol=1e-9)


def test_rotation_over_360_degrees_is_supported():
    transform = Transform(rotation=725.0)
    np.testing.assert_allclose(
        transform.matrix_at(0.0, size=(6, 6)),
        Transform(rotation=5.0).matrix_at(0.0, size=(6, 6)),
        atol=1e-12,
    )


# --------------------------------------------------------------------------- #
# exact rotation, mirroring and translation


@pytest.mark.parametrize("degrees,k", ((90, -1), (180, 2), (270, 1)))
@pytest.mark.parametrize("interpolation", ("nearest", "linear"))
def test_exact_rotation_equals_numpy_rot90(degrees, k, interpolation):
    source = make_buffer()
    transform = Transform(rotation=degrees, interpolation=interpolation)
    result = transform.apply(source, 0.0)
    assert result.size == source.size
    assert result.bounds == source.bounds
    np.testing.assert_array_equal(result.rgba, np.rot90(source.rgba, k))


def test_cubic_rotation_matches_rot90_inside_the_support_margin():
    source = make_buffer()
    transform = Transform(rotation=90, interpolation="cubic")
    result = transform.apply(source, 0.0)
    assert result.bounds == (-1, -1, 7, 7)
    interior = result.crop((0, 0, 6, 6)).rgba
    expected = np.rot90(source.rgba, -1)
    # Cubic fixed-point taps leave denormal residue where the answer is zero,
    # so cubic is compared numerically and through the quantized export.
    np.testing.assert_allclose(interior, expected, atol=1e-6, rtol=0.0)
    np.testing.assert_array_equal(
        result.crop((0, 0, 6, 6)).to_uint8_rgb(),
        Buffer(expected).to_uint8_rgb(),
    )
    ring = np.concatenate(
        [result.rgba[0], result.rgba[-1], result.rgba[:, 0], result.rgba[:, -1]]
    )
    assert float(np.abs(ring).max()) == 0.0


def test_center_anchor_rotation_does_not_displace_the_center_pixel():
    source, center = single_pixel(9)
    for degrees in (0.0, 90.0, 180.0, 270.0, -90.0):
        result = Transform(rotation=degrees, interpolation="linear").apply(source, 0.0)
        assert result.bounds == source.bounds
        assert float(result.rgba[center, center, 3]) == pytest.approx(1.0, abs=1e-6)
    for degrees in (30.0, -45.0, 200.0):
        result = Transform(rotation=degrees, interpolation="linear").apply(source, 0.0)
        row = center - result.offset[1]
        column = center - result.offset[0]
        alpha = result.rgba[..., 3]
        # cv2 inverts the matrix in fixed point, so the impulse keeps 1.0 only
        # to bilinear precision rather than exactly.
        assert float(alpha[row, column]) == pytest.approx(1.0, abs=1e-5)
        # Rotation about the center keeps the four neighbour weights equal.
        neighbours = [
            alpha[row - 1, column],
            alpha[row + 1, column],
            alpha[row, column - 1],
            alpha[row, column + 1],
        ]
        np.testing.assert_allclose(neighbours, float(neighbours[0]), atol=1e-6)
        assert 1.0 < float(alpha.sum()) < 1.5


def test_negative_scale_mirrors_the_image():
    source = make_buffer()
    horizontal = Transform(scale=(-100, 100), interpolation="nearest").apply(source, 0)
    np.testing.assert_array_equal(horizontal.rgba, np.fliplr(source.rgba))
    vertical = Transform(scale=(100, -100), interpolation="nearest").apply(source, 0)
    np.testing.assert_array_equal(vertical.rgba, np.flipud(source.rgba))


def test_half_pixel_translation_matches_the_bilinear_analytic_solution():
    source = make_buffer(width=6, height=3)
    transform = Transform(
        position=(0.5, 0.0), anchor_point=(0.0, 0.0), interpolation="linear"
    )
    result = transform.apply(source, 0.0)
    assert result.size == (7, 3)
    assert result.bounds == (0, 0, 7, 3)
    np.testing.assert_allclose(result.rgba[:, 0], 0.5 * source.rgba[:, 0], atol=1e-6)
    for column in range(1, 6):
        expected = 0.5 * (source.rgba[:, column - 1] + source.rgba[:, column])
        np.testing.assert_allclose(result.rgba[:, column], expected, atol=1e-6)
    np.testing.assert_allclose(result.rgba[:, 6], 0.5 * source.rgba[:, 5], atol=1e-6)


def test_integer_translation_fast_path_shares_pixel_storage():
    source = make_buffer()
    result = Transform(position=(3.0, -2.0), anchor_point=(0.0, 0.0)).apply(source, 0.0)
    assert result.bounds == (3, -2, 9, 4)
    assert result.rgba is source.rgba
    np.testing.assert_array_equal(result.rgba, source.rgba)


def test_source_offset_participates_in_the_mapping():
    source = make_buffer(width=4, height=4, offset=(10, 20))
    transform = Transform(interpolation="nearest")
    result = transform.apply(source, 0.0)
    assert result.bounds == source.bounds
    np.testing.assert_array_equal(result.rgba, source.rgba)
    shifted = Transform(
        position=(11.0, 22.0), anchor_point=(10.0, 20.0), interpolation="nearest"
    )
    moved = shifted.apply(source, 0.0)
    assert moved.bounds == (11, 22, 15, 26)


def test_color_space_label_is_preserved():
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., 3] = 1.0
    source = Buffer(rgba, color_space="linear")
    result = Transform(rotation=90).apply(source, 0.0)
    assert result.color_space == "linear"


# --------------------------------------------------------------------------- #
# opacity


def test_opacity_scales_every_premultiplied_channel():
    source = make_buffer(alpha=0.8)
    result = Transform(opacity=50.0).apply(source, 0.0)
    np.testing.assert_allclose(result.rgba, source.rgba * np.float32(0.5), atol=1e-7)


def test_opacity_zero_is_fully_transparent_but_keeps_geometry():
    source = make_buffer()
    result = Transform(opacity=0.0).apply(source, 0.0)
    assert result.bounds == source.bounds
    assert float(np.abs(result.rgba).max()) == 0.0


def test_opacity_is_clamped_when_animation_overshoots():
    source = make_buffer()
    transform = Transform(opacity=Property(100.0, expression=Expression("value * 3")))
    result = transform.apply(source, 0.0)
    np.testing.assert_array_equal(result.rgba, source.rgba)


def test_opacity_out_of_range_is_rejected_on_assignment():
    with pytest.raises(ValueError, match="opacity"):
        Transform(opacity=150.0)
    with pytest.raises(ValueError, match="opacity"):
        Transform(opacity=-1.0)


# --------------------------------------------------------------------------- #
# animated properties


def test_keyframed_position_moves_the_layer():
    source = make_buffer(width=4, height=4)
    transform = Transform(
        position=Property((0.0, 0.0), keyframes=[(0, (0, 0)), (2, (10, 0))]),
        anchor_point=(0.0, 0.0),
        interpolation="nearest",
    )
    first = transform.apply(source, 0.0)
    middle = transform.apply(source, 1.0)
    last = transform.apply(source, 2.0)
    assert first.bounds == (0, 0, 4, 4)
    assert middle.bounds == (5, 0, 9, 4)
    assert last.bounds == (10, 0, 14, 4)


def test_keyframed_rotation_uses_the_keyframe_curve():
    source = make_buffer()
    transform = Transform(
        rotation=Property(
            0.0,
            keyframes=[
                Keyframe(0, 0, interp="linear"),
                Keyframe(1, 180, interp="linear"),
            ],
        ),
        interpolation="nearest",
    )
    result = transform.apply(source, 1.0)
    np.testing.assert_array_equal(result.rgba, np.rot90(source.rgba, 2))
    quarter = transform.apply(source, 0.5)
    np.testing.assert_array_equal(quarter.rgba, np.rot90(source.rgba, -1))


def test_separate_dimensions_drive_position_axes():
    source = make_buffer(width=4, height=4)
    position = Property((0.0, 0.0))
    axis_x, axis_y = position.separate_dimensions()
    axis_x.set_keyframes([(0, 0), (1, 8)])
    axis_y.set_keyframes([(0, 0), (1, -2)])
    transform = Transform(
        position=Property.from_dimensions((axis_x, axis_y)),
        anchor_point=(0.0, 0.0),
        interpolation="nearest",
    )
    assert transform.apply(source, 1.0).bounds == (8, -2, 12, 2)
    assert transform.apply(source, 0.5).bounds == (4, -1, 8, 3)


def test_expression_driven_scale_is_evaluated():
    source = make_buffer(width=4, height=4)
    transform = Transform(
        scale=Property(
            (100.0, 100.0), expression=Expression("[value[0] * 2, value[1] * 3]")
        ),
        anchor_point=(0.0, 0.0),
        interpolation="nearest",
    )
    assert transform.scale.value_at(0.0) == (200.0, 300.0)
    result = transform.apply(source, 0.0)
    # Nearest neighbor includes boundary pixels after H2 fix
    assert result.bounds == (-1, -1, 8, 11)


# --------------------------------------------------------------------------- #
# auto-orient


def test_auto_orient_follows_the_position_path():
    transform = Transform(
        position=Property((0.0, 0.0), keyframes=[(0, (0, 0)), (1, (0, 10))]),
        anchor_point=(0.0, 0.0),
        auto_orient=True,
    )
    assert transform.rotation_at(0.5) == pytest.approx(90.0, abs=1e-6)
    rightward = Transform(
        position=Property((0.0, 0.0), keyframes=[(0, (0, 0)), (1, (10, 0))]),
        anchor_point=(0.0, 0.0),
        auto_orient=True,
    )
    assert rightward.rotation_at(0.5) == pytest.approx(0.0, abs=1e-6)


def test_auto_orient_without_motion_falls_back_to_zero():
    transform = Transform(position=(4.0, 4.0), auto_orient=True)
    assert transform.rotation_at(0.0) == 0.0


def test_auto_orient_ignores_the_static_rotation_value():
    transform = Transform(
        position=Property((0.0, 0.0), keyframes=[(0, (0, 0)), (1, (0, -10))]),
        rotation=45.0,
        auto_orient=True,
    )
    assert transform.rotation_at(0.5) == pytest.approx(-90.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# explicit bounds, degenerate cases, determinism


def test_explicit_bounds_crop_and_pad_to_the_requested_rectangle():
    source = make_buffer(width=6, height=6)
    result = warp_buffer(
        source, np.eye(3), interpolation="nearest", bounds=(-2, -1, 4, 5)
    )
    assert result.bounds == (-2, -1, 4, 5)
    assert result.size == (6, 6)
    np.testing.assert_array_equal(result.rgba[1:, 2:], source.rgba[:5, :4])
    assert float(np.abs(result.rgba[0]).max()) == 0.0
    assert float(np.abs(result.rgba[:, :2]).max()) == 0.0


def test_zero_scale_returns_an_empty_buffer():
    source = make_buffer()
    result = Transform(scale=(0.0, 0.0)).apply(source, 0.0)
    assert result.size == (0, 0)
    assert result.rgba.shape == (0, 0, 4)


def test_empty_source_buffer_is_passed_through():
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))
    result = Transform(rotation=33.0, scale=(50.0, 50.0)).apply(empty, 0.0)
    assert result.size == (0, 0)


def test_apply_is_deterministic_and_repeatable():
    source = make_buffer(width=9, height=7)
    transform = Transform(position=(3.25, -1.75), rotation=27.5, scale=(113.0, 87.0))
    first = transform.apply(source, 0.25)
    second = transform.apply(source, 0.25)
    np.testing.assert_array_equal(first.rgba, second.rgba)
    assert first.bounds == second.bounds


def test_quality_selects_the_interpolation():
    source = make_buffer()
    transform = Transform(scale=(150.0, 150.0))
    best = transform.apply(source, 0.0, RenderContext(quality="best"))
    draft = transform.apply(source, 0.0, RenderContext(quality="draft"))
    assert not np.array_equal(best.rgba, draft.rgba)
    linear = transform.apply(
        source, 0.0, RenderContext(quality="best"), interpolation="linear"
    )
    np.testing.assert_array_equal(linear.rgba, draft.rgba)


def test_cubic_alpha_is_clipped_into_range():
    rgba = np.zeros((8, 8, 4), dtype=np.float32)
    rgba[:, 4:] = 1.0
    source = Buffer(rgba)
    result = warp_buffer(
        source,
        np.array([[1.0, 0.0, 0.3], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        interpolation="cubic",
    )
    assert float(result.rgba[..., 3].min()) >= 0.0
    assert float(result.rgba[..., 3].max()) <= 1.0


def test_nonfinite_transformation_is_rejected():
    source = make_buffer()
    matrix = np.array([[1.0, 0.0, np.inf], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        warp_buffer(source, matrix)


def test_invalid_matrix_shape_is_rejected():
    with pytest.raises(ValueError, match="3x3"):
        warp_buffer(make_buffer(), np.eye(2))


def test_unknown_interpolation_is_rejected():
    with pytest.raises(ValueError, match="interpolation"):
        Transform(interpolation="lanczos")
    with pytest.raises(ValueError, match="interpolation"):
        warp_buffer(make_buffer(), np.eye(3), interpolation="magic")


def test_scale_lock_preserves_the_aspect_ratio():
    transform = Transform(scale=(200.0, 100.0), constrain_proportions=True)
    transform.scale_axis("x", 400.0)
    assert transform.scale.value_at(0.0) == (400.0, 200.0)
    unlocked = Transform(scale=(200.0, 100.0))
    unlocked.scale_axis("y", 50.0)
    assert unlocked.scale.value_at(0.0) == (200.0, 50.0)
    with pytest.raises(ValueError, match="axis"):
        unlocked.scale_axis("z", 1.0)


def test_uniform_scale_setter_accepts_a_scalar():
    transform = Transform()
    transform.set_scale(75.0)
    assert transform.scale.value_at(0.0) == (75.0, 75.0)
    transform.set_scale(50.0, 25.0)
    assert transform.scale.value_at(0.0) == (50.0, 25.0)


# --------------------------------------------------------------------------- #
# 3D reserved fields


def test_reserved_three_d_fields_are_stored_but_not_applied():
    transform = Transform(position_z=12.0, scale_z=150.0, x_rotation=10.0)
    assert transform.position_z.value_at(0.0) == 12.0
    assert transform.scale_z.value_at(0.0) == 150.0
    assert transform.x_rotation.value_at(0.0) == 10.0
    assert transform.orientation.value_at(0.0) == (0.0, 0.0, 0.0)
    assert transform.three_d is False
    np.testing.assert_allclose(transform.matrix_at(0.0, size=(4, 4)), np.eye(3))


def test_three_d_layers_are_deferred_to_ws22():
    transform = Transform(three_d=True)
    with pytest.raises(NotImplementedError, match="WS-22"):
        transform.matrix_at(0.0, size=(4, 4))
    with pytest.raises(NotImplementedError, match="WS-22"):
        transform.apply(make_buffer(), 0.0)


# --------------------------------------------------------------------------- #
# validation and serialization


@pytest.mark.parametrize(
    "kwargs",
    (
        {"scale": (100.0, np.nan)},
        {"rotation": float("inf")},
        {"position": (0.0, float("nan"))},
        {"anchor_point": (1.0,)},
        {"scale": 100.0},
        {"opacity": "50"},
        {"auto_orient": "yes"},
        {"three_d": 1},
        {"constrain_proportions": None},
    ),
)
def test_invalid_transform_arguments_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        Transform(**kwargs)


def test_assigning_a_property_instance_is_preserved():
    property_ = Property(0.0, keyframes=[(0, 0), (1, 90)])
    transform = Transform(rotation=property_)
    assert transform.rotation is property_
    assert transform.matrix_at(1.0, size=(4, 4))[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_to_dict_round_trip_restores_static_and_keyed_values():
    transform = Transform(
        anchor_point=(1.0, 2.0),
        position=Property((0.0, 0.0), keyframes=[(0, (0, 0)), (1, (5, 6))]),
        scale=(120.0, 80.0),
        rotation=45.0,
        opacity=75.0,
        auto_orient=True,
        interpolation="cubic",
    )
    payload = transform.to_dict()
    assert payload["schema"] == 1
    restored = Transform.from_dict(payload)
    assert restored.auto_orient is True
    assert restored.interpolation == "cubic"
    for time in (0.0, 0.5, 1.0, 2.0):
        np.testing.assert_allclose(
            restored.matrix_at(time, size=(8, 8)),
            transform.matrix_at(time, size=(8, 8)),
            atol=1e-12,
        )
        assert restored.opacity_at(time) == transform.opacity_at(time)


def test_auto_anchor_and_position_serialize_as_null():
    payload = Transform().to_dict()
    assert payload["anchor_point"] is None
    assert payload["position"] is None
    restored = Transform.from_dict(payload)
    assert restored.anchor_point is None
    assert restored.position is None
    np.testing.assert_allclose(restored.matrix_at(0.0, size=(5, 5)), np.eye(3))


def test_from_dict_rejects_unknown_schema_and_fields():
    payload = Transform().to_dict()
    with pytest.raises(ValueError, match="schema"):
        Transform.from_dict({**payload, "schema": 99})
    with pytest.raises(ValueError, match="field"):
        Transform.from_dict({**payload, "unexpected": 1})


def test_warp_buffer_rejects_a_non_buffer_source():
    with pytest.raises(TypeError, match="Buffer"):
        warp_buffer(np.zeros((2, 2, 4), dtype=np.float32), np.eye(3))
