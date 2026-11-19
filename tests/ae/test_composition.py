"""WS-03 tests: Composition, Renderer, pre-compose and legacy interop."""

import os
import tracemalloc

import numpy as np

import pytest

import moviepy.ae as ae
from moviepy import ColorClip, CompositeVideoClip, ImageClip, VideoClip
from moviepy.ae import Composition, Marker, RenderContext, Renderer, from_moviepy
from moviepy.ae.layers import AVLayer, CompLayer, NullLayer, SolidLayer
from moviepy.ae.properties import Property
from moviepy.ae.transform import Transform
from moviepy.video.io.VideoFileClip import VideoFileClip


def corner(position):
    return Transform(position=position, anchor_point=(0.0, 0.0))


def frame_counter(fps, duration, size=(2, 2)):
    """Return a clip whose red code encodes its own frame number."""

    def frame_function(t):
        code = int(round(t * fps)) % 256
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[..., 0] = code
        return frame

    return VideoClip(frame_function, duration=duration).with_fps(fps)


def gradient_clip(width, height, duration=2.0):
    def frame_function(t):
        x = np.linspace(0, 255, width)[None, :, None]
        y = np.linspace(0, 255, height)[:, None, None]
        frame = np.concatenate(
            [x + 0 * y, y + 0 * x, np.full((height, width, 1), 40 * t)], axis=2
        )
        return frame.astype(np.uint8)

    return VideoClip(frame_function, duration=duration).with_fps(10)


# --------------------------------------------------------------------------- #
# equivalence with CompositeVideoClip


def legacy_scene():
    background = ColorClip(size=(16, 12), color=(10, 20, 30), duration=2.0)
    moving = gradient_clip(6, 4).with_position(lambda t: (int(2 + 4 * t), 3))
    masked = ImageClip(np.full((3, 5, 3), (200, 100, 50), dtype=np.uint8))
    masked = masked.with_mask(ImageClip(np.full((3, 5), 0.4), is_mask=True))
    masked = masked.with_position(("right", "bottom")).with_start(0.5)
    masked = masked.with_duration(1.0)
    offscreen = ColorClip(size=(6, 6), color=(0, 255, 0), duration=2.0)
    offscreen = offscreen.with_position((-3, -2))
    return CompositeVideoClip([background, moving, masked, offscreen], size=(16, 12))


@pytest.mark.parametrize("t", [0.0, 0.25, 0.75, 1.4])
def test_from_moviepy_matches_composite_video_clip(t):
    legacy = legacy_scene()
    comp = from_moviepy(legacy)
    assert comp.size == (16, 12)
    assert [layer.name for layer in comp.layers][-1] == comp.layer(4).name
    expected = legacy.get_frame(t).astype(int)
    actual = comp.get_frame(t).astype(int)
    assert np.abs(actual - expected).max() <= 1


def test_manual_position_only_stack_matches_composite_video_clip():
    foreground = gradient_clip(5, 3)
    legacy = CompositeVideoClip(
        [
            ColorClip(size=(10, 8), color=(90, 60, 30), duration=2.0),
            foreground.with_position((4, 2)),
        ],
        size=(10, 8),
    )
    comp = Composition(size=(10, 8), fps=10, duration=2.0)
    comp.add_solid("bg", color=(90, 60, 30))
    comp.add_clip(foreground, transform=corner((4.0, 2.0)))
    for t in (0.0, 0.5, 1.5):
        delta = comp.get_frame(t).astype(int) - legacy.get_frame(t).astype(int)
        assert np.abs(delta).max() <= 1


def test_from_moviepy_rejects_other_clips():
    with pytest.raises(TypeError, match="CompositeVideoClip"):
        from_moviepy(ColorClip(size=(2, 2), color=(0, 0, 0), duration=1))


# --------------------------------------------------------------------------- #
# nesting and time sampling


def test_nested_comps_sample_time_on_each_child_frame_grid():
    inner = Composition(size=(2, 2), fps=24, duration=2.0, name="inner")
    inner.add_clip(frame_counter(24, 2.0))
    middle = Composition(size=(2, 2), fps=30, duration=2.0, name="middle")
    middle.add_comp(inner)
    outer = Composition(size=(2, 2), fps=60, duration=2.0, name="outer")
    outer.add_comp(middle)
    for parent_frame in range(0, 100, 7):
        t = parent_frame / 60
        middle_frame = int(np.floor(t * 30 + 1e-9))
        inner_frame = int(np.floor(middle_frame / 30 * 24 + 1e-9))
        assert outer.get_frame(t)[0, 0, 0] == inner_frame


def test_comp_layer_timing_offset_stretch_and_reverse():
    inner = Composition(size=(2, 2), fps=10, duration=1.0)
    inner.add_clip(frame_counter(10, 1.0))
    outer = Composition(size=(2, 2), fps=10, duration=4.0)
    layer = outer.add_comp(inner, start_time=1.0)
    assert layer.out_point == pytest.approx(2.0)
    assert outer.get_frame(0.5)[0, 0, 0] == 0  # before in_point: background
    assert outer.get_frame(1.3)[0, 0, 0] == 3
    layer.stretch = 200.0
    assert layer.out_point == pytest.approx(3.0)
    assert outer.get_frame(1.6)[0, 0, 0] == 3
    layer.stretch = -100.0
    assert outer.get_frame(1.0)[0, 0, 0] == 9
    assert outer.get_frame(1.5)[0, 0, 0] == 5


def test_comp_layer_holds_the_last_frame_past_the_child_end():
    inner = Composition(size=(2, 2), fps=10, duration=1.0)
    inner.add_clip(frame_counter(10, 1.0))
    layer = CompLayer(inner, out_point=5.0)
    assert layer.child_time(3.0) == pytest.approx(0.9)
    assert layer.child_time(-1.0) == 0.0


def test_nesting_cycles_are_rejected():
    first = Composition(size=(2, 2), duration=1)
    second = Composition(size=(2, 2), duration=1)
    first.add_comp(second)
    with pytest.raises(ValueError, match="contain itself"):
        second.add_comp(first)
    with pytest.raises(ValueError, match="contain itself"):
        first.add_comp(first)
    with pytest.raises(TypeError, match="Composition"):
        CompLayer("comp")


def test_collapse_transformations_keeps_content_outside_the_child_frame():
    inner = Composition(size=(4, 4), fps=10, duration=1.0)
    inner.add_solid("wide", color=(255, 255, 255), size=(8, 4))
    outer = Composition(size=(12, 4), fps=10, duration=1.0)
    layer = outer.add_comp(inner, transform=corner((2.0, 0.0)))
    clipped = outer.get_frame(0)[0, :, 0]
    assert clipped.tolist() == [0, 0] + [255] * 4 + [0] * 6
    layer.collapse_transformations = True
    collapsed = outer.get_frame(0)[0, :, 0]
    assert collapsed.tolist() == [0, 0] + [255] * 8 + [0] * 2


# --------------------------------------------------------------------------- #
# layer management and switches


def test_add_methods_stack_new_layers_on_top_and_keep_indices():
    comp = Composition(size=(2, 2), fps=10, duration=1)
    bottom = comp.add_solid("bottom", color=(255, 0, 0))
    top = comp.add_solid("top", color=(0, 0, 255), size=(1, 2))
    assert comp.layers == (top, bottom)
    assert (top.index, bottom.index) == (1, 2)
    assert comp.layer("bottom") is bottom and comp.layer(1) is top
    assert comp.get_frame(0)[0].tolist() == [[0, 0, 255], [255, 0, 0]]
    comp.move_layer(top, 2)
    assert comp.layers == (bottom, top) and bottom.index == 1
    assert comp.get_frame(0)[0].tolist() == [[255, 0, 0], [255, 0, 0]]
    assert comp.remove_layer("bottom") is bottom
    assert comp.get_frame(0)[0].tolist() == [[0, 0, 255], [0, 0, 0]]
    with pytest.raises(KeyError):
        comp.layer("missing")
    with pytest.raises(IndexError):
        comp.layer(5)
    with pytest.raises(ValueError, match="already"):
        comp.add_layer(top)


def test_removing_a_parent_unparents_its_children():
    comp = Composition(size=(4, 4), fps=10, duration=1)
    controller = comp.add_null("ctrl")
    child = comp.add_solid("child", size=(1, 1), parent=controller)
    comp.remove_layer(controller)
    assert child.parent is None


def test_solo_enabled_and_guide_switches_affect_rendering():
    comp = Composition(size=(1, 1), fps=10, duration=1)
    red = comp.add_solid("red", color=(255, 0, 0))
    green = comp.add_solid("green", color=(0, 255, 0))
    assert comp.get_frame(0)[0, 0].tolist() == [0, 255, 0]
    green.enabled = False
    assert comp.get_frame(0)[0, 0].tolist() == [255, 0, 0]
    green.enabled = True
    red.solo = True
    assert comp.get_frame(0)[0, 0].tolist() == [255, 0, 0]
    red.solo = False
    green.guide = True
    assert comp.get_frame(0)[0, 0].tolist() == [255, 0, 0]
    comp.renderer = Renderer(include_guides=True)
    assert comp.get_frame(0)[0, 0].tolist() == [0, 255, 0]


def test_null_parent_drives_children_and_renders_nothing():
    comp = Composition(size=(6, 2), fps=10, duration=1)
    controller = comp.add_null("ctrl", size=(1, 1))
    controller.transform.anchor_point = (0.0, 0.0)
    controller.transform.position = Property(
        (0.0, 0.0), keyframes=[(0, (0.0, 0.0)), (1, (4.0, 0.0))]
    )
    comp.add_solid(
        "dot",
        color=(255, 255, 255),
        size=(1, 1),
        parent=controller,
        transform=corner((0.0, 0.0)),
    )
    comp.layer("dot").transform.interpolation = "nearest"
    assert np.flatnonzero(comp.get_frame(0.5)[0, :, 0]).tolist() == [2]


# --------------------------------------------------------------------------- #
# blend modes, alpha and background


def test_layer_blend_modes_are_applied_by_the_renderer():
    comp = Composition(size=(1, 1), fps=10, duration=1)
    comp.add_solid("bg", color=(128, 128, 128))
    top = comp.add_solid("top", color=(128, 128, 128))
    top.blend_mode = "multiply"
    assert comp.get_frame(0)[0, 0, 0] == round(128 * 128 / 255)
    top.blend_mode = ae.BlendMode.SCREEN
    expected = round(255 * (1 - (1 - 128 / 255) ** 2))
    assert abs(int(comp.get_frame(0)[0, 0, 0]) - expected) <= 1


def test_preserve_transparency_draws_only_over_existing_pixels():
    comp = Composition(size=(4, 1), fps=10, duration=1, transparent=True)
    comp.add_solid("half", color=(255, 0, 0), size=(2, 1))
    top = comp.add_solid("top", color=(0, 0, 255), preserve_transparency=True)
    assert top.preserve_transparency
    alpha = comp.mask.get_frame(0)[0]
    np.testing.assert_allclose(alpha, [1, 1, 0, 0])
    assert comp.get_frame(0)[0, 0].tolist() == [0, 0, 255]


def test_background_color_is_not_part_of_the_alpha():
    comp = Composition(size=(2, 1), fps=10, duration=1, bg_color=(0, 0, 255))
    comp.add_solid("half", color=(255, 0, 0), size=(1, 1))
    assert comp.get_frame(0)[0].tolist() == [[255, 0, 0], [0, 0, 255]]
    assert comp.mask is None
    buffer = comp.render_buffer(0)
    np.testing.assert_allclose(buffer.rgba[0, :, 3], [1, 0])


def test_transparent_comp_composites_correctly_in_moviepy():
    comp = Composition(size=(2, 1), fps=10, duration=1, transparent=True)
    layer = comp.add_solid("half", color=(255, 255, 255), size=(1, 1))
    layer.transform.opacity = 50.0
    np.testing.assert_allclose(comp.mask.get_frame(0)[0], [0.5, 0.0])
    assert comp.get_frame(0)[0, 0].tolist() == [255, 255, 255]
    background = ColorClip(size=(2, 1), color=(0, 0, 0), duration=1)
    composite = CompositeVideoClip([background, comp])
    assert abs(int(composite.get_frame(0)[0, 0, 0]) - 128) <= 1
    assert composite.get_frame(0)[0, 1].tolist() == [0, 0, 0]
    comp.transparent = False
    assert comp.mask is None


def test_dissolve_layers_are_deterministic_per_seed():
    comp = Composition(size=(16, 16), fps=10, duration=1)
    comp.add_solid("bg", color=(0, 0, 0))
    top = comp.add_solid("fg", color=(255, 255, 255))
    top.transform.opacity = 50.0
    top.blend_mode = "dissolve"
    first = comp.render_buffer(0).rgba
    np.testing.assert_array_equal(first, comp.render_buffer(0.5).rgba)
    other = comp.render_buffer(0, RenderContext(rng_seed=9)).rgba
    assert not np.array_equal(first, other)
    assert 0.3 < float((first[..., 0] > 0.5).mean()) < 0.7


# --------------------------------------------------------------------------- #
# region of interest, resolution and determinism


def scene():
    comp = Composition(size=(20, 12), fps=10, duration=2)
    comp.add_solid("bg", color=(20, 40, 60))
    spinner = comp.add_solid("spin", color=(250, 200, 10), size=(8, 4))
    spinner.transform.rotation = Property(0.0, keyframes=[(0, 0.0), (2, 90.0)])
    spinner.transform.opacity = 80.0
    over = comp.add_clip(gradient_clip(9, 7), transform=corner((13.5, 7.25)))
    over.blend_mode = "overlay"
    return comp


def test_region_of_interest_equals_a_crop_of_the_full_render():
    comp = scene()
    full = comp.render_buffer(0.7)
    roi = comp.render_buffer(0.7, bounds=(3, 2, 17, 11))
    assert roi.bounds == (3, 2, 17, 11)
    np.testing.assert_allclose(roi.rgba, full.crop((3, 2, 17, 11)).rgba, atol=1e-6)
    outside = comp.render_buffer(0.7, bounds=(30, 30, 33, 32))
    assert outside.bounds == (30, 30, 33, 32)


def test_render_is_bitwise_deterministic():
    comp = scene()
    np.testing.assert_array_equal(
        comp.render_buffer(1.1).rgba, scene().render_buffer(1.1).rgba
    )


def test_resolution_scale_renders_fewer_pixels_but_frames_stay_full_size():
    comp = scene()
    half = comp.render_buffer(0.0, RenderContext(resolution_scale=0.5))
    assert half.size == (10, 6)
    comp.context = RenderContext(resolution_scale=0.5, quality="draft")
    frame = comp.get_frame(0.0)
    assert frame.shape == (12, 20, 3)
    reference = scene().get_frame(0.0).astype(int)
    assert np.abs(frame.astype(int) - reference).mean() < 20


def test_rendering_does_not_leak_memory():
    comp = Composition(size=(160, 90), fps=30, duration=10)
    for index in range(20):
        layer = comp.add_solid(f"l{index}", color=(index * 10, 50, 200), size=(40, 30))
        layer.transform.position = Property(
            (0.0, 0.0), keyframes=[(0, (10.0 + index, 10.0)), (10, (150.0, 80.0))]
        )
        layer.transform.opacity = 60.0
    for frame in range(20):
        comp.get_frame(frame / 30)
    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    for frame in range(20, 220):
        comp.get_frame(frame / 30)
    growth = tracemalloc.get_traced_memory()[0] - baseline
    tracemalloc.stop()
    assert growth < 2 * 1024 * 1024


# --------------------------------------------------------------------------- #
# metadata, validation and export


def test_markers_and_work_area():
    comp = Composition(size=(2, 2), fps=10, duration=4)
    late = comp.add_marker(3.0, "outro", chapter="End")
    early = comp.add_marker(1.0, "intro", duration=0.5)
    assert comp.markers == (early, late) and early.end == 1.5
    assert isinstance(late, Marker) and late.chapter == "End"
    with pytest.raises(ValueError, match="duration"):
        Marker(1.0, duration=-1)
    assert comp.work_area == (0.0, 4.0)
    comp.work_area = (1.0, 2.5)
    assert comp.work_area_clip().duration == pytest.approx(1.5)
    with pytest.raises(ValueError, match="work_area"):
        comp.work_area = (3.0, 2.0)


def test_composition_validates_its_settings():
    with pytest.raises(ValueError, match="duration"):
        Composition(duration=0)
    with pytest.raises(ValueError, match="fps"):
        Composition(fps=-1)
    with pytest.raises(ValueError, match="size"):
        Composition(size=(0, 10))
    with pytest.raises(TypeError, match="renderer"):
        Composition(renderer="fast")
    with pytest.raises(TypeError, match="context"):
        Composition(context={})
    with pytest.raises(TypeError, match="Layer"):
        Composition().add_layer("layer")
    with pytest.raises(TypeError, match="context"):
        Composition(size=(2, 2), duration=1).render_buffer(0, "ctx")


def test_composition_writes_a_video_file(tmp_path):
    comp = Composition(size=(16, 16), fps=10, duration=0.5, bg_color=(0, 0, 0))
    comp.add_solid("bg", color=(0, 0, 255))
    box = comp.add_solid("box", color=(255, 255, 255), size=(8, 8))
    box.transform.position = Property(
        (8.0, 8.0), keyframes=[(0, (4.0, 8.0)), (0.5, (12.0, 8.0))]
    )
    path = str(tmp_path / "comp.mp4")
    comp.write_videofile(path, codec="libx264", logger=None)
    assert os.path.getsize(path) > 0
    with VideoFileClip(path) as reread:
        assert tuple(reread.size) == (16, 16)
        frame = reread.get_frame(0).astype(int)
    assert frame[8, 14, 2] > 200 and frame[8, 14, 0] < 60
    assert frame[8, 3].min() > 200


def test_video_clip_to_ae_layer_hook():
    clip = ColorClip(size=(3, 2), color=(1, 2, 3), duration=1)
    layer = clip.to_ae_layer("footage", start_time=0.5, blend_mode="add")
    assert isinstance(layer, AVLayer)
    assert (layer.name, layer.start_time, layer.blend_mode) == ("footage", 0.5, "add")


def test_facade_exports_are_lazy_and_complete():
    for name in ae.__all__:
        assert getattr(ae, name) is not None
    assert ae.SolidLayer is SolidLayer and ae.NullLayer is NullLayer
    with pytest.raises(AttributeError):
        ae.DoesNotExist  # noqa: B018
