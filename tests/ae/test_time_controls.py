"""AV source retiming contracts, independent of the layer-property clock."""

import numpy as np

import pytest

from moviepy import ColorClip, VideoClip
from moviepy.ae import Buffer, Composition, RenderContext, Transform
from moviepy.ae.effects.generate.fill import Fill
from moviepy.ae.effects.time.echo import Echo
from moviepy.ae.layers import AVLayer
from moviepy.ae.masks import Mask
from moviepy.ae.properties import Expression, Property


def coded_clip(*, fps=4, duration=2):
    """Return continuous footage with separately recorded RGB and alpha reads."""
    rgb_times, alpha_times = [], []

    def frame(t):
        rgb_times.append(t)
        assert 0 <= t < duration
        return np.full((1, 2, 3), 80 * t, dtype=np.float32)

    def alpha(t):
        alpha_times.append(t)
        return np.full((1, 2), 0.25 + t / 8)

    clip = VideoClip(frame, duration=duration)
    clip.mask = VideoClip(alpha, is_mask=True, duration=duration)
    if fps is not None:
        clip = clip.with_fps(fps)
    rgb_times.clear()
    alpha_times.clear()
    return clip, rgb_times, alpha_times


def test_time_remap_freezes_footage_but_keeps_layer_animation():
    clip, rgb_times, alpha_times = coded_clip()
    layer = AVLayer(
        clip,
        time_remap=0.25,
        transform=Transform(
            anchor_point=(0, 0),
            position=lambda t: (4 * t, 0),
            opacity=lambda t: 100 - 25 * t,
        ),
    )
    first, second = layer.render(0.5), layer.render(1)
    assert first.offset == (2, 0) and second.offset == (4, 0)
    np.testing.assert_array_equal(first.to_uint8_rgb(), second.to_uint8_rgb())
    assert rgb_times == alpha_times == [0.25, 0.25]
    assert first.rgba[0, 0, 3] == pytest.approx((0.25 + 0.25 / 8) * 0.875)
    assert second.rgba[0, 0, 3] == pytest.approx((0.25 + 0.25 / 8) * 0.75)
    assert layer.source_time(1) == 1


def test_time_remap_keeps_layer_mask_and_effect_on_the_layer_clock():
    clip, rgb_times, _ = coded_clip()
    layer = AVLayer(
        clip,
        time_remap=0.25,
        masks=[Mask.rect((0, 0), (20, 20), opacity=lambda t: 100 * t)],
        effects=[Fill(color=lambda t: (200 * t, 0, 0))],
    )
    for t in (0.25, 0.75):
        result = layer.prepared_source(t)
        np.testing.assert_allclose(result.straight()[0, 0, :3], (200 * t / 255, 0, 0))
        assert result.rgba[0, 0, 3] == pytest.approx((0.25 + 0.25 / 8) * t)
    assert rgb_times == [0.25, 0.25]


@pytest.mark.parametrize("stretch", [50, 100, 200, -100, -200])
def test_time_remap_identity_preserves_offset_and_stretch(stretch):
    clip, rgb_times, _ = coded_clip()
    layer = AVLayer(clip, start_time=1, stretch=stretch)
    times = [1 + value * abs(stretch) / 100 for value in (0, 0.2, 0.7, 1.8)]
    expected = [layer.prepared_source(t).rgba.copy() for t in times]
    expected_calls = rgb_times.copy()
    rgb_times.clear()
    layer.time_remap = Property(keyframes=[(0, 0), (2, 2)])
    for t, rgba in zip(times, expected):
        np.testing.assert_array_equal(layer.prepared_source(t).rgba, rgba)
    assert rgb_times == expected_calls


def test_time_remap_media_mapping_stays_continuous_unclamped_and_unsnapped():
    clip, _, _ = coded_clip()
    layer = AVLayer(clip, start_time=2, stretch=200, time_remap=lambda u: 3 * u - 1)
    assert layer.footage_time(0.1) == pytest.approx(-0.7)
    assert layer.media_time(2.2) == pytest.approx(-0.7)
    assert layer.media_time(4.5) == pytest.approx(2.75)
    assert layer.source_time(4.5) == 1.25
    assert layer.out_point == 6


def test_time_remap_render_holds_endpoints_without_extending_layer_window():
    clip, rgb_times, alpha_times = coded_clip()
    layer = AVLayer(clip, time_remap=-5)
    layer.render(0.3)
    layer.time_remap = 5
    layer.render(0.7)
    assert rgb_times == alpha_times == [0, 1.75]
    assert layer.render(layer.out_point) is None
    assert rgb_times == [0, 1.75]


def test_time_remap_temporal_effect_maps_each_neighbor_on_its_layer_clock():
    clip, rgb_times, alpha_times = coded_clip()
    layer = AVLayer(
        clip,
        time_remap=lambda u: u * u,
        effects=[Echo(echo_time=-0.25, number_of_echoes=2, echo_operator="blend")],
    )
    layer.prepared_source(0.75)
    assert rgb_times == alpha_times == [0.5625, 0.25, 0.0625]


def test_time_remap_round_trip_and_mutation_do_not_leave_stale_frames():
    clip, _, _ = coded_clip()
    comp = Composition(size=(2, 1), fps=10, duration=2)
    layer = comp.add_clip(clip, time_remap=Property(keyframes=[(0, 1.5), (2, 0)]))
    times = [1.5, 0.25, 1, 0.25]
    expected = [comp.render_buffer(t).rgba.copy() for t in times]
    layer.time_remap = Property.from_json(layer.time_remap.to_json())
    for t, rgba in zip(times, expected):
        np.testing.assert_array_equal(comp.render_buffer(t).rgba, rgba)
    layer.time_remap.set_keyframes([(0, 0.125), (2, 0.125)])
    assert not np.array_equal(comp.render_buffer(0.25).rgba, expected[1])
    layer.time_remap = None
    np.testing.assert_array_equal(
        layer.prepared_source(0.25).rgba, layer.source_buffer(0.25).rgba
    )


def test_time_remap_expression_receives_layer_clock_and_context_fps():
    clip, _, _ = coded_clip()
    layer = AVLayer(
        clip, start_time=1, time_remap=Expression("time + framesToTime(index)")
    )
    context = RenderContext(t=9, fps=20)
    assert layer.media_time(1.25, context) == pytest.approx(0.3)


@pytest.mark.parametrize(
    "value", [True, "1", (1, 2), float("nan"), float("inf"), Property((1, 2))]
)
def test_time_remap_invalid_assignment_is_atomic(value):
    clip, _, _ = coded_clip()
    layer = AVLayer(clip, time_remap=0.5)
    before = layer.time_remap
    with pytest.raises((TypeError, ValueError)):
        layer.time_remap = value
    assert layer.time_remap is before


def test_time_remap_callable_is_lazy_and_validates_evaluated_values():
    calls = []

    def remap(t):
        calls.append(t)
        return float("nan")

    clip, _, _ = coded_clip()
    layer = AVLayer(clip, time_remap=remap)
    assert not calls
    with pytest.raises((TypeError, ValueError)):
        layer.media_time(0.5)
    assert calls == [0.5]
    with pytest.raises(TypeError, match="callable"):
        layer.time_remap.to_json()


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf")])
def test_time_remap_mapping_rejects_invalid_times_even_when_disabled(value):
    clip, _, _ = coded_clip()
    layer = AVLayer(clip)
    with pytest.raises((TypeError, ValueError)):
        layer.footage_time(value)


class BufferFootage(AVLayer):
    """Exercise source controls with native float32 Buffers, including HDR."""

    def __init__(self, frames, **kwargs):
        self.frames, self.reads = frames, []
        clip = ColorClip((1, 1), color=(0, 0, 0), duration=1).with_fps(2)
        super().__init__(clip, **kwargs)

    def source_buffer(self, t, context=None):
        self.reads.append(t)
        return self.frames[0 if t < 0.5 else 1]


def test_frame_mix_fetches_source_grid_rgb_and_mask_before_interpolating():
    clip, rgb_times, alpha_times = coded_clip(fps=2)
    layer = AVLayer(clip, frame_blending="frame_mix", time_remap=0.25)
    result = layer.prepared_source(0.8)
    assert rgb_times == alpha_times == [0, 0.5]
    alpha0, alpha1 = 0.25, 0.3125
    expected = [(0 * alpha0 + (40 / 255) * alpha1) / 2] * 3
    expected.append((alpha0 + alpha1) / 2)
    np.testing.assert_allclose(result.rgba[0, 0], expected, rtol=1e-7)


def test_frame_mix_uses_premultiplied_pixels_without_hidden_color_bleed():
    first = Buffer(np.array([[[1, 0, 0, 1]]], dtype=np.float32))
    second = Buffer(np.array([[[0, 0, 1, 0]]], dtype=np.float32))
    layer = BufferFootage([first, second], frame_blending="frame_mix")
    result = layer.prepared_source(0.25)
    np.testing.assert_array_equal(result.rgba, [[[0.5, 0, 0, 0.5]]])
    np.testing.assert_array_equal(result.to_uint8_rgb(), [[[255, 0, 0]]])
    assert not result.rgba.flags.writeable
    np.testing.assert_array_equal(first.rgba, [[[1, 0, 0, 1]]])
    np.testing.assert_array_equal(second.rgba, [[[0, 0, 0, 0]]])


@pytest.mark.parametrize("color_space", ["srgb", "linear"])
def test_frame_mix_preserves_signed_hdr_and_handles_large_opposite_values(color_space):
    maximum = np.finfo(np.float32).max
    first = Buffer(
        np.array([[[maximum, -4, 6, 0.25]]], np.float32), color_space=color_space
    )
    second = Buffer(
        np.array([[[-maximum, 2, 2, 0.75]]], np.float32), color_space=color_space
    )
    layer = BufferFootage([first, second], frame_blending="frame_mix")
    result = layer.prepared_source(0.25)
    assert result.color_space == color_space
    np.testing.assert_array_equal(result.rgba, [[[0, -1, 4, 0.5]]])
    assert np.isfinite(result.rgba).all()


def test_frame_mix_converts_both_frames_before_linear_light_interpolation():
    black = Buffer(np.array([[[0, 0, 0, 1]]], np.float32))
    white = Buffer(np.ones((1, 1, 4), np.float32))
    layer = BufferFootage([black, white], frame_blending="frame_mix")
    linear = layer.prepared_source(0.25, RenderContext(working_space="linear"))
    np.testing.assert_array_equal(linear.rgba, [[[0.5, 0.5, 0.5, 1]]])
    assert linear.color_space == "linear"
    comp = Composition(
        size=(1, 1), duration=1, context=RenderContext(working_space="linear")
    )
    comp.add_layer(layer)
    np.testing.assert_array_equal(comp.get_frame(0.25), [[[188, 188, 188]]])
    comp.context = RenderContext(working_space="srgb")
    np.testing.assert_array_equal(comp.get_frame(0.25), [[[128, 128, 128]]])


def test_frame_mix_aligns_different_source_bounds_without_resizing():
    first = Buffer(np.array([[[1, 0, 0, 1]]], np.float32), offset=(-1, 0))
    second = Buffer(np.array([[[0, 0, 1, 1]]], np.float32), offset=(1, 0))
    layer = BufferFootage([first, second], frame_blending="frame_mix")
    result = layer.prepared_source(0.25)
    assert result.bounds == (-1, 0, 2, 1)
    np.testing.assert_array_equal(
        result.rgba, [[[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0.5, 0.5]]]
    )


def test_frame_mix_union_uses_existing_allocation_budget():
    first = Buffer(np.ones((1, 1, 4), np.float32), offset=(0, 0))
    second = Buffer(np.ones((1, 1, 4), np.float32), offset=(2**27, 0))
    layer = BufferFootage([first, second], frame_blending="frame_mix")
    with pytest.raises(ValueError, match="render budget"):
        layer.prepared_source(0.25)


def test_frame_mix_zero_sized_frames_are_transparent_without_expanding_empty_bounds():
    empty = Buffer(np.zeros((0, 0, 4), np.float32), offset=(2**27, 0))
    source = Buffer(np.ones((1, 1, 4), np.float32))
    layer = BufferFootage([empty, source], frame_blending="frame_mix")
    result = layer.prepared_source(0.25)
    assert result.bounds == source.bounds
    np.testing.assert_array_equal(result.rgba, np.full((1, 1, 4), 0.5))


@pytest.mark.parametrize("fps", [3, 24, 30000 / 1001])
def test_frame_mix_grid_boundaries_keep_adjacent_float_times_distinct(fps):
    clip, rgb_times, _ = coded_clip(fps=fps)
    layer = AVLayer(clip, frame_blending="frame_mix")
    boundary = 1 / fps
    layer.prepared_source(boundary)
    assert rgb_times == [boundary]
    rgb_times.clear()
    layer.prepared_source(np.nextafter(boundary, 0))
    assert rgb_times == [0, boundary]


@pytest.mark.parametrize("duration, expected_end", [(0.2, 0), (0.75, 0.5), (1, 0.5)])
def test_frame_mix_endpoint_holds_never_read_outside_the_source(duration, expected_end):
    clip, rgb_times, _ = coded_clip(fps=2, duration=duration)
    layer = AVLayer(clip, frame_blending="frame_mix")
    for time in (-5, 0, duration, 50):
        layer.sampled_source(time)
    assert rgb_times == [0, 0, expected_end, expected_end]


def test_frame_mix_defaults_off_keeps_continuous_reader_times_and_call_count():
    clip, rgb_times, alpha_times = coded_clip()
    layer = AVLayer(clip)
    assert layer.frame_blending == "off"
    layer.prepared_source(0.37)
    assert rgb_times == alpha_times == [0.37]
    layer.frame_blending = "frame_mix"
    layer.prepared_source(0.37)
    assert rgb_times == [0.37, 0.25, 0.5]
    layer.frame_blending = "off"
    layer.prepared_source(0.37)
    assert rgb_times[-1] == 0.37


def test_frame_mix_uses_source_fps_even_inside_a_different_comp_frame_rate():
    clip, rgb_times, _ = coded_clip(fps=2)
    comp = Composition(size=(2, 1), fps=30, duration=2)
    comp.add_clip(clip, frame_blending="frame_mix")
    comp.render_buffer(0.25)
    assert rgb_times == [0, 0.5]


@pytest.mark.parametrize("fps", [None, 0, -1, True, "24", float("nan"), float("inf")])
def test_frame_mix_requires_valid_source_fps_only_when_enabled(fps):
    clip, _, _ = coded_clip(fps=None)
    clip.fps = fps
    layer = AVLayer(clip)
    layer.prepared_source(0.37)
    with pytest.raises((TypeError, ValueError), match="fps"):
        layer.frame_blending = "frame_mix"
    assert layer.frame_blending == "off"


def test_frame_mix_revalidates_mutable_source_fps():
    clip, _, _ = coded_clip()
    layer = AVLayer(clip, frame_blending="frame_mix")
    clip.fps = 0
    with pytest.raises(ValueError, match="fps"):
        layer.prepared_source(0.25)


def test_frame_mix_accepts_open_ended_sources_and_rejects_empty_sources():
    clip = ColorClip((1, 1), color=(255, 0, 0)).with_fps(2)
    layer = AVLayer(clip, frame_blending="frame_mix")
    np.testing.assert_array_equal(
        layer.prepared_source(20.25).to_uint8_rgb(), [[[255, 0, 0]]]
    )
    empty = clip.with_duration(0)
    with pytest.raises(ValueError, match="duration"):
        AVLayer(empty, frame_blending="frame_mix")


@pytest.mark.parametrize("mode", [None, True, 1, "nearest", "FRAME_MIX"])
def test_frame_mix_mode_validation_is_atomic(mode):
    clip, _, _ = coded_clip()
    layer = AVLayer(clip)
    with pytest.raises((TypeError, ValueError), match="frame_blending"):
        layer.frame_blending = mode
    assert layer.frame_blending == "off"


def test_frame_mix_pixel_motion_has_an_explicit_unsupported_error():
    clip, _, _ = coded_clip()
    layer = AVLayer(clip)
    with pytest.raises(NotImplementedError, match="pixel_motion"):
        layer.frame_blending = "pixel_motion"
    assert layer.frame_blending == "off"


def test_frame_mix_random_seek_matches_sequential_output():
    clip, _, _ = coded_clip()
    layer = AVLayer(
        clip, frame_blending="frame_mix", time_remap=lambda t: 1.5 - t * t / 2
    )
    times = [0, 0.17, 0.5, 0.73, 1.2]
    expected = {t: layer.prepared_source(t).rgba.copy() for t in times}
    for t in (0.73, 0, 1.2, 0.17, 0.5, 0.73):
        np.testing.assert_array_equal(layer.prepared_source(t).rgba, expected[t])
