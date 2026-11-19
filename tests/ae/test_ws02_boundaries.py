"""WS-02 boundary tests: validation branches, degenerate geometry and getters."""

import math

import numpy as np

import pytest

from moviepy.ae import Buffer
from moviepy.ae.layers import AVLayer, Layer, NullLayer, SolidLayer
from moviepy.ae.properties import Property
from moviepy.ae.transform import Transform, affine_matrix
from moviepy.ae.warp import (
    destination_bounds,
    resolve_interpolation,
    warp_buffer,
)


def small_buffer(width=4, height=4):
    """Return an opaque float32 buffer for boundary exercises."""
    rgba = np.full((height, width, 4), 0.5, dtype=np.float32)
    return Buffer(rgba)


class StubClip:
    """Minimal clip double exposing only what AVLayer validates."""

    is_mask = False

    def __init__(self, size=(4, 4), duration=1.0):
        self.size = size
        self.duration = duration
        self.name = "stub"

    def get_frame(self, t):
        return np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# coordinate and matrix validation


def test_affine_matrix_rejects_wrong_length_coordinates():
    with pytest.raises(ValueError, match="two finite numbers"):
        affine_matrix((1.0, 2.0, 3.0), 0.0, (100.0, 100.0), (0.0, 0.0))


def test_integer_pairs_reject_bool_fractions_and_wrong_length():
    with pytest.raises(TypeError, match="integers"):
        destination_bounds(np.eye(3), (4, 4), offset=(True, 0))
    with pytest.raises(ValueError, match="two integers"):
        destination_bounds(np.eye(3), (4, 4), offset=(1,))
    with pytest.raises(ValueError, match="representable"):
        destination_bounds(np.eye(3), (4, 4), offset=(2**40, 0))


def test_negative_extent_size_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        Transform().resolve_anchor((-1, 4))


def test_property_kind_mismatch_is_rejected():
    with pytest.raises(ValueError, match="vec2 Property"):
        Transform(scale=Property(1.0))
    with pytest.raises(ValueError, match="float Property"):
        Transform(rotation=Property((1.0, 2.0)))


def test_bounds_validation():
    buffer = small_buffer()
    with pytest.raises(ValueError, match="four integers"):
        warp_buffer(buffer, np.eye(3), bounds=(1, 2, 3))
    with pytest.raises(ValueError, match="inverted"):
        warp_buffer(buffer, np.eye(3), bounds=(5, 0, 0, 5))
    with pytest.raises(ValueError, match="representable"):
        warp_buffer(buffer, np.eye(3), bounds=(-(2**31 - 1), 0, 2**31 - 1, 1))


def test_interpolation_names_are_validated():
    with pytest.raises(ValueError, match="interpolation"):
        resolve_interpolation(None)
    with pytest.raises(ValueError, match="interpolation"):
        resolve_interpolation("bicubic")
    with pytest.raises(ValueError, match="nearest, linear or cubic"):
        destination_bounds(np.eye(3), (4, 4), interpolation="auto")


def test_nonfinite_and_oversized_transformed_bounds_are_rejected():
    overflow = np.array([[1e308, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        destination_bounds(overflow, (10, 10))
    huge = np.array([[1e10, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="representable"):
        destination_bounds(huge, (10, 10))


def test_far_translation_falls_through_to_the_bounds_guard():
    far = np.array([[1.0, 0.0, float(2**40)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="representable"):
        warp_buffer(small_buffer(), far)


def test_empty_source_with_explicit_bounds_keeps_the_requested_origin():
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))
    result = warp_buffer(empty, np.eye(3), bounds=(3, 4, 7, 8))
    assert result.size == (0, 0)
    assert result.offset == (3, 4)


def test_opacity_is_applied_on_the_resampling_path():
    rgba = np.full((8, 8, 4), 0.5, dtype=np.float32)
    source = Buffer(rgba)
    result = Transform(rotation=30.0, opacity=25.0, interpolation="linear").apply(
        source, 0.0
    )
    interior = result.crop((3, 3, 5, 5)).rgba
    np.testing.assert_allclose(interior, 0.5 * 0.25, atol=1e-6)


# --------------------------------------------------------------------------- #
# transform getters, auto-orient and apply validation


def test_reserved_and_switch_getters_round_trip():
    transform = Transform(
        opacity=40.0,
        y_rotation=15.0,
        z_rotation=-25.0,
        constrain_proportions=True,
    )
    assert transform.opacity.value_at(0.0) == 40.0
    assert transform.y_rotation.value_at(0.0) == 15.0
    assert transform.z_rotation.value_at(0.0) == -25.0
    assert transform.constrain_proportions is True
    assert transform.interpolation == "auto"


def test_auto_orient_with_a_stationary_keyframed_position():
    transform = Transform(
        position=Property((0.0, 0.0), keyframes=[(0, (5, 5)), (1, (5, 5))]),
        auto_orient=True,
    )
    assert transform.rotation_at(0.5) == 0.0


def test_auto_orient_with_an_automatic_position_stays_level():
    transform = Transform(auto_orient=True)
    assert transform.rotation_at(0.0) == 0.0
    np.testing.assert_allclose(transform.matrix_at(0.0, size=(4, 4)), np.eye(3))


def test_apply_rejects_a_non_buffer():
    with pytest.raises(TypeError, match="Buffer"):
        Transform().apply(np.zeros((2, 2, 4), dtype=np.float32), 0.0)


def test_scale_lock_cannot_preserve_a_zero_axis():
    transform = Transform(scale=(0.0, 100.0), constrain_proportions=True)
    with pytest.raises(ValueError, match="zero axis"):
        transform.scale_axis("x", 50.0)


def test_scale_axis_requires_a_static_scale():
    keyed = Transform(
        scale=Property((100.0, 100.0), keyframes=[(0, (100, 100)), (1, (50, 50))])
    )
    with pytest.raises(ValueError, match="static"):
        keyed.scale_axis("x", 25.0)
    dynamic = Transform(scale=lambda t: (100.0, 100.0))
    with pytest.raises(ValueError, match="static"):
        dynamic.scale_axis("y", 25.0)


# --------------------------------------------------------------------------- #
# serialization boundaries


def test_from_dict_rejects_a_foreign_payload_type():
    payload = Transform().to_dict()
    with pytest.raises(ValueError, match="transform"):
        Transform.from_dict({**payload, "type": "layer"})


def test_from_dict_rejects_a_non_object_optional_property():
    payload = Transform().to_dict()
    with pytest.raises(ValueError, match="null or a property object"):
        Transform.from_dict({**payload, "anchor_point": 5})


# --------------------------------------------------------------------------- #
# layer validation boundaries


def test_layer_name_length_and_index_type_validation():
    with pytest.raises(ValueError, match="length"):
        Layer("x" * 400)
    with pytest.raises(TypeError, match="index"):
        Layer("layer", index=True)
    with pytest.raises(TypeError, match="real number"):
        Layer("layer", start_time=True)


def test_layer_getters_expose_validated_values():
    layer = Layer("layer", index=7, start_time=1.5, enabled=False)
    assert layer.name == "layer"
    assert layer.index == 7
    assert layer.start_time == 1.5
    assert layer.enabled is False
    assert layer.transform is not None


def test_parent_type_is_validated():
    with pytest.raises(TypeError, match="parent"):
        Layer("layer", parent="not-a-layer")


def test_parent_chain_detects_a_cycle_created_behind_the_setter():
    root = NullLayer("root", size=(4, 4))
    child = NullLayer("child", size=(4, 4), parent=root)
    # The validated setter already refuses cycles; this covers the defensive
    # check inside parent_chain itself.
    root._parent = child
    with pytest.raises(ValueError, match="cycle"):
        root.parent_chain()


def test_av_layer_without_a_finite_duration_stays_open_ended():
    layer = AVLayer(StubClip(duration=None))
    assert layer.out_point == math.inf
    assert layer.is_active(1e6)


def test_av_layer_rejects_a_degenerate_clip_size():
    layer = AVLayer(StubClip(size=(0, 4)))
    with pytest.raises(ValueError, match="positive pixel counts"):
        layer.source_size


def test_null_size_is_readable_and_writable():
    null = NullLayer("controller")
    assert null.size == (100, 100)
    null.size = (8, 6)
    assert null.size == (8, 6)
    assert null.source_size == (8, 6)
    assert null.transform.resolve_anchor((8, 6)) == (3.5, 2.5)
    with pytest.raises(ValueError, match="size"):
        null.size = (0, 6)


def test_solid_color_and_size_getters_and_validation():
    solid = SolidLayer("bg", color=(10, 20, 30), size=(3, 2))
    assert solid.size == (3, 2)
    assert solid.color == pytest.approx((10 / 255, 20 / 255, 30 / 255))
    with pytest.raises(ValueError, match="three components"):
        solid.color = (10, 20)
    with pytest.raises(TypeError, match="three numbers"):
        solid.color = None


def test_layers_package_exports_are_lazy_and_introspectable():
    import moviepy.ae.layers as package

    with pytest.raises(AttributeError, match="no attribute"):
        package.Missing
    listing = dir(package)
    for name in ("Layer", "AVLayer", "NullLayer", "SolidLayer"):
        assert name in listing
