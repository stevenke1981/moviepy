"""WS-02 layer tests: time model, switches, parenting and concrete sources."""

import math

import numpy as np
import pytest

from moviepy.ae import RenderContext
from moviepy.ae.layers import AVLayer, Layer, NullLayer, SolidLayer
from moviepy.ae.properties import Property
from moviepy.ae.transform import Transform


def make_clip(width=6, height=4, duration=2.0, fps=24):
    """Return a synthetic RGB clip whose pixels depend on time."""
    from moviepy.video.VideoClip import VideoClip

    def frame_function(t):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        code = int(min(max(t, 0.0) * 40, 200))
        frame[..., 0] = code
        frame[..., 1] = 128
        frame[..., 2] = 255 - code
        return frame

    clip = VideoClip(frame_function=frame_function, duration=duration)
    return clip.with_fps(fps)


# --------------------------------------------------------------------------- #
# time model


def test_source_time_applies_start_time_and_stretch():
    layer = SolidLayer("solid", color=(0, 0, 0), size=(2, 2), start_time=1.0)
    assert layer.source_time(1.0) == pytest.approx(0.0)
    assert layer.source_time(2.5) == pytest.approx(1.5)
    slowed = SolidLayer("solid", color=(0, 0, 0), size=(2, 2), stretch=200.0)
    assert slowed.source_time(4.0) == pytest.approx(2.0)
    sped = SolidLayer("solid", color=(0, 0, 0), size=(2, 2), stretch=50.0)
    assert sped.source_time(1.0) == pytest.approx(2.0)


def test_negative_stretch_reverses_time():
    layer = SolidLayer("solid", color=(0, 0, 0), size=(2, 2), stretch=-100.0)
    assert layer.source_time(0.0) == pytest.approx(0.0)
    assert layer.source_time(3.0) == pytest.approx(-3.0)


def test_zero_stretch_and_non_finite_times_are_rejected():
    with pytest.raises(ValueError, match="stretch"):
        SolidLayer("solid", color=(0, 0, 0), size=(2, 2), stretch=0.0)
    with pytest.raises(ValueError, match="finite"):
        SolidLayer("solid", color=(0, 0, 0), size=(2, 2), start_time=float("nan"))


def test_in_and_out_points_use_composition_time():
    layer = SolidLayer(
        "solid", color=(0, 0, 0), size=(2, 2), in_point=1.0, out_point=3.0
    )
    assert layer.duration == pytest.approx(2.0)
    assert not layer.is_active(0.999)
    assert layer.is_active(1.0)
    assert layer.is_active(2.999)
    assert not layer.is_active(3.0)


def test_open_out_point_defaults_to_infinity_for_a_solid():
    layer = SolidLayer("solid", color=(0, 0, 0), size=(2, 2))
    assert layer.out_point == math.inf
    assert layer.is_active(1e6)


def test_inverted_time_range_is_rejected():
    with pytest.raises(ValueError, match="out_point"):
        SolidLayer("solid", color=(0, 0, 0), size=(2, 2), in_point=2.0, out_point=1.0)


def test_av_layer_out_point_follows_the_source_duration_and_stretch():
    clip = make_clip(duration=2.0)
    layer = AVLayer(clip)
    assert layer.out_point == pytest.approx(2.0)
    stretched = AVLayer(clip, stretch=200.0)
    assert stretched.out_point == pytest.approx(4.0)
    explicit = AVLayer(clip, out_point=0.5)
    assert explicit.out_point == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# switches


def test_visibility_honours_enabled_and_solo():
    layer = SolidLayer(
        "solid", color=(0, 0, 0), size=(2, 2), in_point=0.0, out_point=4.0
    )
    assert layer.visible(0.0)
    layer.enabled = False
    assert not layer.visible(0.0)
    assert not layer.is_active(0.0)
    layer.enabled = True
    assert not layer.visible(0.0, solo_active=True)
    layer.solo = True
    assert layer.visible(0.0, solo_active=True)
    assert not layer.visible(5.0, solo_active=True)
    open_ended = SolidLayer("open", color=(0, 0, 0), size=(2, 2), solo=True)
    assert open_ended.visible(1e6, solo_active=True)


def test_metadata_switches_are_validated_and_inert():
    layer = SolidLayer("solid", color=(0, 0, 0), size=(2, 2), shy=True, locked=True)
    assert (layer.shy, layer.locked, layer.guide) == (True, True, False)
    assert layer.visible(0.0)
    with pytest.raises(TypeError, match="shy"):
        layer.shy = "yes"


def test_blend_mode_defaults_to_normal_and_rejects_empty_tokens():
    layer = SolidLayer("solid", color=(0, 0, 0), size=(2, 2))
    assert layer.blend_mode == "normal"
    layer.blend_mode = "screen"
    assert layer.blend_mode == "screen"
    with pytest.raises(ValueError, match="blend_mode"):
        layer.blend_mode = "  "
    with pytest.raises(TypeError, match="blend_mode"):
        layer.blend_mode = 3


def test_rendering_flags_are_stored_for_later_workstreams():
    layer = SolidLayer(
        "solid",
        color=(0, 0, 0),
        size=(2, 2),
        collapse_transformations=True,
        continuously_rasterize=True,
        motion_blur=True,
    )
    assert layer.collapse_transformations is True
    assert layer.continuously_rasterize is True
    assert layer.motion_blur is True


def test_invalid_layer_identity_is_rejected():
    with pytest.raises(ValueError, match="index"):
        SolidLayer("solid", color=(0, 0, 0), size=(2, 2), index=0)
    with pytest.raises(TypeError, match="name"):
        SolidLayer(3, color=(0, 0, 0), size=(2, 2))
    with pytest.raises(ValueError, match="name"):
        SolidLayer("  ", color=(0, 0, 0), size=(2, 2))
    with pytest.raises(TypeError, match="transform"):
        SolidLayer("solid", color=(0, 0, 0), size=(2, 2), transform="identity")


# --------------------------------------------------------------------------- #
# concrete sources


def test_solid_layer_renders_the_requested_color():
    layer = SolidLayer("bg", color=(20, 20, 30), size=(3, 2))
    buffer = layer.source_buffer(0.0)
    assert buffer.size == (3, 2)
    assert buffer.offset == (0, 0)
    np.testing.assert_array_equal(
        layer.render(0.0).to_uint8_rgb(),
        np.full((2, 3, 3), (20, 20, 30), dtype=np.uint8),
    )


def test_solid_color_validation():
    with pytest.raises(ValueError, match="color"):
        SolidLayer("bg", color=(300, 0, 0), size=(2, 2))
    with pytest.raises(ValueError, match="size"):
        SolidLayer("bg", color=(0, 0, 0), size=(0, 2))
    with pytest.raises(TypeError, match="color"):
        SolidLayer("bg", color="black", size=(2, 2))


def test_solid_layer_caches_its_pixel_storage():
    layer = SolidLayer("bg", color=(1, 2, 3), size=(4, 4))
    assert layer.source_buffer(0.0).rgba is layer.source_buffer(1.0).rgba
    layer.color = (9, 9, 9)
    np.testing.assert_array_equal(
        layer.source_buffer(0.0).to_uint8_rgb(), np.full((4, 4, 3), 9, dtype=np.uint8)
    )


def test_null_layer_contributes_no_pixels_but_can_parent():
    null = NullLayer("controller", size=(100, 100))
    assert null.source_size == (100, 100)
    rendered = null.render(0.0)
    assert rendered.size == (0, 0)
    child = SolidLayer("child", color=(0, 255, 0), size=(4, 4), parent=null)
    null.transform.position = (10.0, 20.0)
    null.transform.anchor_point = (50.0, 50.0)
    child.transform.position = (50.0, 50.0)
    child.transform.anchor_point = (0.0, 0.0)
    result = child.render(0.0)
    assert result.bounds == (10, 20, 14, 24)
    np.testing.assert_array_equal(
        result.to_uint8_rgb(), np.full((4, 4, 3), (0, 255, 0), dtype=np.uint8)
    )


def test_av_layer_reads_clip_frames_at_source_time():
    clip = make_clip(duration=2.0)
    layer = AVLayer(clip, start_time=0.5)
    assert layer.source_size == (6, 4)
    buffer = layer.source_buffer(0.5)
    np.testing.assert_array_equal(buffer.to_uint8_rgb(), clip.get_frame(0.5))
    shifted = layer.render(1.0)
    np.testing.assert_array_equal(shifted.to_uint8_rgb(), clip.get_frame(0.5))


def test_av_layer_rejects_mask_clips_and_non_clips():
    from moviepy.video.VideoClip import VideoClip

    mask = VideoClip(is_mask=True, duration=1.0)
    with pytest.raises(ValueError, match="mask"):
        AVLayer(mask)
    with pytest.raises(TypeError, match="clip"):
        AVLayer("not-a-clip")


def test_base_layer_source_is_abstract():
    layer = Layer("placeholder")
    with pytest.raises(NotImplementedError):
        layer.source_buffer(0.0)
    with pytest.raises(NotImplementedError):
        layer.source_size
    with pytest.raises(NotImplementedError):
        layer.render(0.0)
    assert layer.is_active(0.0)
    assert layer.duration == math.inf


# --------------------------------------------------------------------------- #
# parenting


def test_parent_rotation_and_child_position_match_a_hand_computed_matrix():
    parent = NullLayer("parent", size=(100, 100))
    parent.transform.anchor_point = (50.0, 50.0)
    parent.transform.position = (200.0, 100.0)
    parent.transform.rotation = 30.0
    child = SolidLayer(
        "child",
        color=(0, 255, 0),
        size=(10, 10),
        parent=parent,
        transform=Transform(position=(60.0, 50.0), anchor_point=(5.0, 5.0)),
    )
    radians = math.radians(30.0)
    cos, sin = math.cos(radians), math.sin(radians)
    parent_matrix = np.array(
        [
            [cos, -sin, 200.0 - (cos * 50.0 - sin * 50.0)],
            [sin, cos, 100.0 - (sin * 50.0 + cos * 50.0)],
            [0.0, 0.0, 1.0],
        ]
    )
    child_matrix = np.array([[1.0, 0.0, 55.0], [0.0, 1.0, 45.0], [0.0, 0.0, 1.0]])
    expected = parent_matrix @ child_matrix
    np.testing.assert_allclose(child.world_matrix(0.0), expected, atol=1e-9)
    rendered = child.render(0.0)
    alpha = rendered.rgba[..., 3].astype(np.float64)
    assert alpha.sum() == pytest.approx(100.0, rel=1e-3)
    rows, columns = np.mgrid[0 : alpha.shape[0], 0 : alpha.shape[1]]
    centroid = np.array(
        [(columns * alpha).sum() / alpha.sum(), (rows * alpha).sum() / alpha.sum()]
    )
    world_centroid = centroid + np.array(rendered.offset, dtype=np.float64)
    mapped_center = expected @ np.array([4.5, 4.5, 1.0])
    np.testing.assert_allclose(world_centroid, mapped_center[:2], atol=0.05)


def test_parent_chain_nests_and_detects_cycles():
    root = NullLayer("root", size=(10, 10))
    middle = NullLayer("middle", size=(10, 10), parent=root)
    leaf = NullLayer("leaf", size=(10, 10), parent=middle)
    assert leaf.parent_chain() == (leaf, middle, root)
    with pytest.raises(ValueError, match="cycle"):
        root.parent = leaf
        root.parent_chain()


def test_self_parenting_is_rejected():
    layer = NullLayer("self", size=(10, 10))
    with pytest.raises(ValueError, match="cycle"):
        layer.parent = layer


def test_opacity_is_not_inherited_from_the_parent():
    parent = NullLayer("parent", size=(10, 10))
    parent.transform.opacity = 25.0
    child = SolidLayer("child", color=(255, 255, 255), size=(2, 2), parent=parent)
    result = child.render(0.0)
    assert float(result.rgba[..., 3].max()) == pytest.approx(1.0)
    assert child.opacity_at(0.0) == pytest.approx(1.0)


def test_parent_scale_applies_to_child_geometry():
    parent = NullLayer("parent", size=(10, 10))
    parent.transform.scale = (200.0, 200.0)
    parent.transform.anchor_point = (0.0, 0.0)
    parent.transform.position = (0.0, 0.0)
    child = SolidLayer(
        "child",
        color=(255, 255, 255),
        size=(4, 4),
        parent=parent,
        transform=Transform(
            position=(0.0, 0.0), anchor_point=(0.0, 0.0), interpolation="linear"
        ),
    )
    result = child.render(0.0)
    # Source box [-1, 4] doubled by the parent, so world centers 0..6 stay full.
    assert result.bounds == (-1, -1, 8, 8)
    alpha = result.rgba[..., 3]

    def at(x, y):
        return float(alpha[y - result.offset[1], x - result.offset[0]])

    assert at(0, 0) == pytest.approx(1.0)
    assert at(6, 6) == pytest.approx(1.0)
    # Bilinear corner weights multiply: 0.5 per axis at the half-pixel edge.
    assert at(7, 7) == pytest.approx(0.25, abs=1e-6)
    assert at(-1, -1) == pytest.approx(0.25, abs=1e-6)


def test_invisible_layer_renders_none():
    layer = SolidLayer(
        "solid", color=(1, 2, 3), size=(2, 2), in_point=1.0, out_point=2.0
    )
    assert layer.render(0.5) is None
    assert layer.render(1.5) is not None
    layer.enabled = False
    assert layer.render(1.5) is None


def test_layer_render_is_deterministic_across_repeated_calls():
    clip = make_clip()
    layer = AVLayer(
        clip,
        transform=Transform(
            position=Property((0.0, 0.0), keyframes=[(0, (0, 0)), (2, (5, 3))]),
            rotation=Property(0.0, keyframes=[(0, 0), (2, 45)]),
            anchor_point=(3.0, 2.0),
        ),
    )
    first = layer.render(0.75)
    second = layer.render(0.75)
    np.testing.assert_array_equal(first.rgba, second.rgba)
    assert first.bounds == second.bounds


def test_render_accepts_explicit_bounds_for_roi_rendering():
    layer = SolidLayer("solid", color=(7, 7, 7), size=(8, 8))
    result = layer.render(0.0, bounds=(2, 2, 5, 6))
    assert result.bounds == (2, 2, 5, 6)
    assert result.size == (3, 4)


def test_layer_uses_its_own_identity_for_expression_randomness():
    first = SolidLayer("a", color=(255, 255, 255), size=(2, 2), index=1)
    second = SolidLayer("b", color=(255, 255, 255), size=(2, 2), index=2)
    assert first.expression_bindings["index"] == 1
    assert first.expression_bindings["layer_id"] == "a"
    assert second.expression_bindings["layer_id"] == "b"


def test_render_context_time_is_not_used_for_source_time():
    clip = make_clip(duration=4.0)
    layer = AVLayer(clip, start_time=1.0)
    context = RenderContext(t=9.0, fps=24)
    np.testing.assert_array_equal(
        layer.render(2.0, context).to_uint8_rgb(), clip.get_frame(1.0)
    )
