"""WS-02 integration tests: layers, transforms and the existing MoviePy pipeline."""

import numpy as np

import pytest

from moviepy import ColorClip, CompositeVideoClip
from moviepy.ae import Buffer, RenderContext
from moviepy.ae.layers import AVLayer, NullLayer, SolidLayer
from moviepy.ae.properties import Keyframe, Property
from moviepy.ae.transform import Transform


def render_stack(layers, t, *, bounds=None, context=None, solo_active=None):
    """Composite layers bottom to top exactly like the future WS-03 renderer."""
    if solo_active is None:
        solo_active = any(layer.solo for layer in layers)
    accumulator = None
    for layer in layers:
        rendered = (
            layer.render(t, context, bounds=bounds)
            if layer.visible(t, solo_active=solo_active)
            else None
        )
        if rendered is None or 0 in rendered.size:
            continue
        accumulator = (
            rendered if accumulator is None else rendered.composite_over(accumulator)
        )
    return accumulator


def test_position_only_stack_matches_composite_video_clip():
    background = ColorClip(size=(8, 6), color=(10, 20, 30), duration=1.0)
    foreground = ColorClip(size=(4, 3), color=(200, 100, 50), duration=1.0)
    legacy = CompositeVideoClip(
        [background, foreground.with_position((2, 1))], size=(8, 6)
    ).get_frame(0.5)

    layers = [
        SolidLayer("bg", color=(10, 20, 30), size=(8, 6)),
        SolidLayer(
            "fg",
            color=(200, 100, 50),
            size=(4, 3),
            transform=Transform(position=(2.0, 1.0), anchor_point=(0.0, 0.0)),
        ),
    ]
    actual = render_stack(layers, 0.5).to_uint8_rgb()
    assert np.abs(actual.astype(int) - legacy.astype(int)).max() <= 1


def test_semitransparent_stack_matches_the_float_analytic_solution():
    layers = [
        SolidLayer("bg", color=(0, 0, 0), size=(4, 4)),
        SolidLayer(
            "fg",
            color=(255, 255, 255),
            size=(2, 2),
            transform=Transform(position=(1.0, 1.0), anchor_point=(0.0, 0.0)),
        ),
    ]
    layers[1].transform.opacity = 50.0
    result = render_stack(layers, 0.0).to_uint8_rgb()
    expected = np.zeros((4, 4, 3), dtype=np.uint8)
    expected[1:3, 1:3] = 128
    np.testing.assert_array_equal(result, expected)


def test_solo_suppresses_every_other_layer():
    layers = [
        SolidLayer("bg", color=(255, 0, 0), size=(4, 4)),
        SolidLayer("fg", color=(0, 255, 0), size=(2, 2), solo=True),
    ]
    result = render_stack(layers, 0.0)
    assert result.bounds == (0, 0, 2, 2)
    np.testing.assert_array_equal(
        result.to_uint8_rgb(), np.full((2, 2, 3), (0, 255, 0), dtype=np.uint8)
    )


def test_keyframed_layer_animation_is_deterministic_and_moves():
    clip = ColorClip(size=(4, 4), color=(255, 255, 255), duration=2.0)
    layer = AVLayer(
        clip,
        transform=Transform(
            position=Property(
                (0.0, 0.0),
                keyframes=[
                    Keyframe(0, (0, 0), interp="linear"),
                    Keyframe(2, (12, 0), interp="linear"),
                ],
            ),
            anchor_point=(0.0, 0.0),
            interpolation="nearest",
        ),
    )
    background = SolidLayer("bg", color=(0, 0, 0), size=(20, 6))
    first = render_stack([background, layer], 1.0).to_uint8_rgb()
    second = render_stack([background, layer], 1.0).to_uint8_rgb()
    np.testing.assert_array_equal(first, second)
    moved = np.flatnonzero(first[0, :, 0] > 0)
    assert moved.min() == 6 and moved.max() == 9


def test_null_controller_drives_a_parented_footage_layer():
    clip = ColorClip(size=(4, 4), color=(20, 200, 90), duration=2.0)
    controller = NullLayer("controller", size=(100, 100))
    controller.transform.anchor_point = (50.0, 50.0)
    controller.transform.position = Property(
        (50.0, 50.0), keyframes=[(0, (50, 50)), (1, (70, 50))]
    )
    footage = AVLayer(
        clip,
        parent=controller,
        transform=Transform(position=(50.0, 50.0), anchor_point=(2.0, 2.0)),
    )
    early = footage.render(0.0)
    late = footage.render(1.0)
    assert early.bounds == (48, 48, 52, 52)
    assert late.bounds == (68, 48, 72, 52)
    np.testing.assert_array_equal(early.rgba, late.rgba)


def test_render_context_quality_reaches_the_layer_warp():
    clip = ColorClip(size=(6, 6), color=(255, 255, 255), duration=1.0)
    layer = AVLayer(
        clip,
        transform=Transform(
            position=(0.0, 0.0), anchor_point=(0.0, 0.0), scale=(150.0, 150.0)
        ),
    )
    best = layer.render(0.0, RenderContext(quality="best"))
    draft = layer.render(0.0, RenderContext(quality="draft"))
    # Cubic support widens the automatic rectangle by one pixel on each side.
    assert best.bounds == (-2, -2, 11, 11)
    assert draft.bounds == (-1, -1, 9, 9)
    assert not np.array_equal(best.crop(draft.bounds).rgba, draft.rgba)


def test_layer_time_window_and_stretch_reach_the_source_clip():
    frames = {}

    def frame_function(t):
        frames.setdefault(round(float(t), 6), 0)
        frames[round(float(t), 6)] += 1
        return np.full((2, 2, 3), 128, dtype=np.uint8)

    from moviepy.video.VideoClip import VideoClip

    clip = VideoClip(frame_function=frame_function, duration=4.0)
    layer = AVLayer(clip, in_point=1.0, out_point=3.0, start_time=1.0, stretch=200.0)
    assert layer.render(0.5) is None
    assert layer.render(1.0) is not None
    assert layer.render(2.0) is not None
    assert layer.render(3.0) is None
    assert set(frames) == {0.0, 0.5}


def test_guide_locked_and_shy_flags_do_not_change_rendering():
    layers = [
        SolidLayer("bg", color=(255, 255, 255), size=(2, 2), guide=True, shy=True),
        SolidLayer("fg", color=(0, 0, 0), size=(2, 2), locked=True),
    ]
    result = render_stack(layers, 0.0)
    np.testing.assert_array_equal(
        result.to_uint8_rgb(), np.zeros((2, 2, 3), dtype=np.uint8)
    )


def test_transform_result_feeds_buffer_compositing_without_quantization_loss():
    gradient = np.zeros((4, 4, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, 4, dtype=np.uint8)[None, :]
    source = Buffer.from_uint8_rgb(gradient)
    layer_transform = Transform(
        position=(1.5, 0.0), anchor_point=(0.0, 0.0), interpolation="linear"
    )
    warped = layer_transform.apply(source, 0.0)
    half = SolidLayer("half", color=(0, 0, 0), size=(8, 4))
    half.transform.opacity = 100.0
    combined = warped.composite_over(half.render(0.0))
    assert combined.rgba.dtype == np.float32
    assert float(combined.rgba[..., 3].max()) <= 1.0
    assert combined.to_uint8_rgb().shape == (4, 8, 3)
    # The half-pixel warp must keep sub-code float precision, not 1/255 steps.
    red = warped.rgba[..., 0][1]
    steps = np.unique(np.round(red * 255.0, 6))
    assert any(abs(step - round(step)) > 1e-4 for step in steps)


def test_ws02_guide_example_runs():
    """Mirror the documented example so the guide stays executable."""
    background = SolidLayer("bg", color=(20, 20, 30), size=(64, 36))
    footage = AVLayer(
        ColorClip(size=(32, 18), color=(240, 180, 60), duration=2.0),
        name="footage",
        transform=Transform(anchor_point=(16.0, 9.0)),
    )
    footage.transform.position = Property((16.0, 9.0))
    footage.transform.position.set_keyframes([(0, (16, 9)), (2, (48, 27))])
    footage.transform.scale = (100, 100)
    footage.transform.opacity = 80.0
    controller = NullLayer("controller", size=(100, 100))
    assert footage.parent is None
    footage.parent = controller
    frames = [
        render_stack([background, footage], time).to_uint8_rgb()
        for time in (0.0, 1.0, 2.0)
    ]
    assert all(frame.shape == (36, 64, 3) for frame in frames)
    assert not np.array_equal(frames[0], frames[-1])
    with pytest.raises(ValueError, match="cycle"):
        controller.parent = footage
        controller.parent_chain()
