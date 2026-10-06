"""Shutter sampling, actual trail geometry, coverage and compatibility."""

import numpy as np

import pytest

from moviepy import VideoClip
from moviepy.ae import Buffer, Composition, RenderContext, Transform
from moviepy.ae.effects.base import AEEffect, Param


def setup_shutter(**changes):
    values = {"angle": 180, "phase": -90, "samples": 64}
    values.update(changes)
    return RenderContext(shutter=values)


def moving_point(context=None, **kwargs):
    comp = Composition(size=(128, 16), fps=10, duration=2, context=context, **kwargs)
    layer = comp.add_solid(
        size=(1, 1),
        color=(255, 255, 255),
        motion_blur=True,
        transform=Transform(
            position=lambda t: (10 + 200 * t, 8), interpolation="linear"
        ),
    )
    return comp, layer


def test_actual_trail_width_matches_speed_times_exposure():
    comp, _ = moving_point(setup_shutter())
    alpha = comp.render_buffer(0.5).rgba[..., 3].sum(axis=0)
    x = np.arange(len(alpha))
    center = np.dot(x, alpha) / alpha.sum()
    width = np.sqrt(12 * np.dot((x - center) ** 2, alpha) / alpha.sum())
    assert center == pytest.approx(110, abs=0.02)
    assert width == pytest.approx(10, rel=0.05)
    assert alpha.sum() == pytest.approx(1, abs=1e-6)


def test_phase_shifts_center_by_the_exposure_half_width():
    comp, _ = moving_point(setup_shutter(phase=0))
    alpha = comp.render_buffer(0.5).rgba[..., 3].sum(axis=0)
    center = np.dot(np.arange(len(alpha)), alpha) / alpha.sum()
    assert center == pytest.approx(115, abs=0.02)


def test_defaults_and_layer_switch_keep_legacy_single_sample_pixels():
    comp, layer = moving_point()
    expected = comp.render_buffer(0.5).rgba.copy()
    assert np.count_nonzero(expected[..., 3]) == 1
    comp.context = setup_shutter()
    comp.motion_blur = True
    layer.motion_blur = False
    np.testing.assert_array_equal(comp.render_buffer(0.5).rgba, expected)
    layer.motion_blur = True
    comp.motion_blur = False
    np.testing.assert_array_equal(comp.render_buffer(0.5).rgba, expected)


def test_roi_and_random_seek_match_full_frame():
    comp, _ = moving_point(setup_shutter())
    expected = {t: comp.render_buffer(t) for t in (0.1, 0.25, 0.5)}
    for t in (0.5, 0.1, 0.25, 0.5):
        np.testing.assert_array_equal(comp.render_buffer(t).rgba, expected[t].rgba)
        roi = (45, 2, 116, 14)
        np.testing.assert_array_equal(
            comp.render_buffer(t, bounds=roi).rgba, expected[t].crop(roi).rgba
        )


def test_layer_half_open_window_clips_exposure_without_renormalizing():
    comp, layer = moving_point(setup_shutter(samples=8))
    layer.in_point = 0.5
    assert comp.render_buffer(0.5).rgba[..., 3].sum() == pytest.approx(0.5)
    assert not np.any(comp.render_buffer(np.nextafter(0.5, 0)).rgba)


class TimeColor(AEEffect):
    PARAMS = (Param("level", "float", 0.0),)

    def render(self, src, t, context=None, values=None):
        result = src.rgba.copy()
        result[..., :3] = values["level"] * result[..., 3:]
        return Buffer(result, src.offset, color_space=src.color_space)


def test_effect_properties_sample_shutter_and_keep_linear_hdr():
    comp = Composition(
        size=(1, 1),
        fps=1,
        duration=2,
        context=RenderContext(
            shutter={"angle": 360, "phase": -180, "samples": 8}, working_space="linear"
        ),
    )
    comp.add_solid(
        color=(255, 255, 255),
        motion_blur=True,
        effects=[TimeColor(level=lambda t: 2 + t * t)],
    )
    times = 0.5 + (np.arange(8) + 0.5) / 8
    value = comp.render_buffer(1).rgba[0, 0]
    np.testing.assert_allclose(value[:3], np.mean(2 + times**2), atol=1e-6, rtol=0)
    assert value[3] == 1


def test_av_footage_time_stays_at_center_while_transform_is_subsampled():
    calls = []

    def read(t):
        calls.append(float(t))
        return np.full((1, 1, 3), int(t * 100), np.uint8)

    clip = VideoClip(read, duration=2)
    comp = Composition(
        size=(32, 4), fps=10, duration=2, context=setup_shutter(samples=8)
    )
    comp.add_clip(
        clip,
        motion_blur=True,
        transform=Transform(
            position=lambda t: (10 + 10 * t, 1), interpolation="linear"
        ),
    )
    calls.clear()
    result = comp.render_buffer(0.5)
    assert np.count_nonzero(result.rgba[..., 3]) > 1
    assert calls
    np.testing.assert_allclose(calls, 0.5, atol=1e-15, rtol=0)


def test_standalone_layer_render_honors_explicit_shutter():
    _, layer = moving_point()
    actual = layer.render(0.5, setup_shutter(samples=8), bounds=(0, 0, 128, 16))
    assert np.count_nonzero(actual.rgba[..., 3]) > 1


def test_static_target_does_not_disable_a_moving_mattes_exposure():
    comp = Composition(
        size=(24, 2),
        fps=1,
        duration=3,
        context=setup_shutter(angle=360, phase=-180, samples=8),
    )
    target = comp.add_solid(color=(255, 255, 255))
    matte = comp.add_solid(
        size=(2, 2),
        color=(255, 255, 255),
        motion_blur=True,
        transform=Transform(
            anchor_point=(0, 0),
            position=lambda t: (int(t * 8), 0),
            interpolation="nearest",
        ),
    )
    target.set_track_matte(matte)
    expected = comp.render_buffer(1)
    assert np.count_nonzero(expected.rgba[..., 3]) > 4
    target.motion_blur = True
    np.testing.assert_array_equal(comp.render_buffer(1).rgba, expected.rgba)


def test_nested_av_matte_uses_child_time_during_parent_exposure():
    reads = []

    def pixels(t):
        reads.append(float(t))
        return np.full((2, 2, 3), int(t * 100), np.uint8)

    clip = VideoClip(pixels, duration=2).with_fps(10)
    child = Composition(size=(2, 2), fps=10, duration=2)
    target = child.add_solid(color=(255, 255, 255))
    target.set_track_matte(child.add_clip(clip), mode="luma")
    parent = Composition(
        size=(2, 2), fps=10, duration=8, context=setup_shutter(samples=4)
    )
    parent.add_comp(
        child,
        start_time=4,
        motion_blur=True,
        transform=Transform(anchor_point=(0, 0), position=(0, 0)),
    )
    reads.clear()
    parent.render_buffer(5)
    assert reads
    assert all(0.9 <= t <= 1.0 for t in reads)


def test_one_sample_is_at_exposure_midpoint_and_zero_angle_is_identity():
    comp, _ = moving_point(setup_shutter(samples=1, phase=0))
    alpha = comp.render_buffer(0.5).rgba[..., 3]
    assert alpha[8, 115] == 1
    comp.context = setup_shutter(angle=0)
    alpha = comp.render_buffer(0.5).rgba[..., 3]
    assert alpha[8, 110] == 1
    assert np.count_nonzero(alpha) == 1


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("angle", -1, ValueError),
        ("angle", 721, ValueError),
        ("angle", float("inf"), ValueError),
        ("phase", float("nan"), ValueError),
        ("samples_min", True, TypeError),
        ("samples_min", 0, ValueError),
        ("samples_max", 257, ValueError),
        ("samples_max", 4, ValueError),
        ("adaptive", "yes", TypeError),
    ],
)
def test_shutter_validation(field, value, error):
    from moviepy.ae import Shutter

    with pytest.raises(error):
        Shutter(**{field: value})


def test_proven_static_solid_uses_one_sample(monkeypatch):
    comp = Composition(size=(4, 3), motion_blur=True)
    comp.add_solid(color=(30, 120, 255), motion_blur=True)
    calls = []
    original = comp.renderer._render_layer_at

    def record(*args):
        calls.append(args[1].t)
        return original(*args)

    monkeypatch.setattr(comp.renderer, "_render_layer_at", record)
    comp.render_buffer(0.5)
    assert len(calls) == 1


def test_adaptive_samples_obey_bounds_and_parent_animation():
    from moviepy.ae import Shutter

    calls = []
    comp = Composition(
        size=(128, 16),
        fps=10,
        duration=2,
        motion_blur=True,
        context=RenderContext(shutter=Shutter(samples_min=8, samples_max=16)),
    )
    parent = comp.add_null(
        transform=Transform(anchor_point=(0, 0), position=lambda t: (200 * t, 0))
    )
    layer = comp.add_solid(
        size=(1, 1),
        color=(255, 255, 255),
        parent=parent,
        motion_blur=True,
        transform=Transform(position=(10, 8), interpolation="linear"),
    )
    original = layer.prepared_source
    layer.prepared_source = lambda t, ctx: (calls.append(t), original(t, ctx))[1]
    actual = comp.render_buffer(0.5)
    assert 8 <= len(calls) <= 16
    assert len(set(calls)) == len(calls)
    assert np.count_nonzero(actual.rgba[..., 3]) > 1


def test_shutter_outside_layer_window_returns_transparent_not_backdrop():
    comp = Composition(
        size=(4, 4),
        duration=1,
        fps=1,
        context=setup_shutter(angle=180, phase=360, samples=8),
    )
    comp.add_solid(color=(255, 255, 255))
    comp.add_solid(
        color=(255, 255, 255), blend_mode="stencil_alpha", motion_blur=True, out_point=1
    )
    assert not np.any(comp.render_buffer(0.5).rgba)


def test_adjustment_effect_properties_are_sampled_over_exposure():
    ctx = RenderContext(
        shutter={"angle": 360, "phase": -180, "samples": 8}, working_space="linear"
    )
    comp = Composition(size=(2, 2), fps=1, duration=2, context=ctx)
    comp.add_solid(color=(255, 255, 255))
    comp.add_adjustment(
        motion_blur=True, effects=[TimeColor(level=lambda t: 2 + t * t)]
    )
    expected = np.mean(2 + (0.5 + (np.arange(8) + 0.5) / 8) ** 2)
    np.testing.assert_allclose(
        comp.render_buffer(1).rgba[..., :3], expected, atol=1e-6, rtol=0
    )


def test_adjustment_outside_shutter_window_retains_backdrop():
    comp = Composition(
        size=(2, 2), fps=1, duration=1, context=setup_shutter(phase=360, samples=8)
    )
    comp.add_solid(color=(255, 255, 255))
    comp.add_adjustment(motion_blur=True, out_point=1, effects=[TimeColor(level=0)])
    np.testing.assert_array_equal(comp.render_buffer(0.5).rgba, np.ones((2, 2, 4)))


def test_animated_adjustment_radius_with_shutter_is_roi_invariant():
    from moviepy.ae.effects.blur.gaussian_blur import GaussianBlur

    comp = Composition(
        size=(48, 32), fps=1, duration=2, context=setup_shutter(samples=8)
    )
    comp.add_solid(
        size=(12, 12), color=(255, 255, 255), transform=Transform(position=(23, 15))
    )
    comp.add_adjustment(
        motion_blur=True, effects=[GaussianBlur(blurriness=lambda t: abs(t - 1) * 60)]
    )
    full = comp.render_buffer(1)
    roi = (6, 3, 25, 27)
    np.testing.assert_allclose(
        comp.render_buffer(1, bounds=roi).rgba, full.crop(roi).rgba, atol=1e-6, rtol=0
    )


@pytest.mark.parametrize("mode", ["stencil_alpha", "stencil_luma"])
def test_zero_opacity_stencil_policy_is_unchanged_by_motion_switch(mode):
    comp = Composition(size=(4, 4), context=setup_shutter(samples=8))
    comp.add_solid(color=(255, 255, 255))
    layer = comp.add_solid(
        color=(255, 255, 255), blend_mode=mode, transform=Transform(opacity=0)
    )
    original = comp.render_buffer(0.5)
    layer.motion_blur = True
    np.testing.assert_array_equal(comp.render_buffer(0.5).rgba, original.rgba)


def test_moving_matte_is_applied_inside_each_temporal_sample():
    comp = Composition(
        size=(48, 8), fps=1, duration=2, context=setup_shutter(samples=8)
    )
    transform = Transform(
        anchor_point=(0, 0), position=lambda t: (8 + 16 * t, 3), interpolation="nearest"
    )
    target = comp.add_solid(
        size=(2, 2), color=(255, 255, 255), motion_blur=True, transform=transform
    )
    matte = comp.add_solid(
        size=(2, 2), color=(255, 255, 255), motion_blur=True, transform=transform
    )
    target.set_track_matte(matte)
    actual = comp.render_buffer(1).rgba
    expected = np.zeros_like(actual, dtype=np.float64)
    for time in 0.75 + (np.arange(8) + 0.5) / 16:
        instantaneous = RenderContext(shutter={"angle": 0})
        expected += comp.render_buffer(float(time), instantaneous).rgba / 8
    np.testing.assert_array_equal(actual, expected)
    assert actual[..., 3].sum() == 4


def test_nested_composition_keeps_its_shutter_parameters():
    child, _ = moving_point(setup_shutter(phase=0, samples=8))
    parent = Composition(size=child.size, fps=child.fps, duration=child.duration)
    parent.add_comp(child)
    np.testing.assert_array_equal(
        parent.render_buffer(0.5).rgba, child.render_buffer(0.5).rgba
    )
