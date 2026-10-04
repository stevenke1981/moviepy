"""Regression tests from the 2026-10-04 review of WS-03..WS-06.

Every test here FAILS on commit 6d4c246 and encodes one item of
``AE_REVIEW_PLAN.md``. The item id is in each test name (R1, R2, ...).
"""

import numpy as np

import moviepy.ae as ae
from moviepy.ae import Buffer, Composition, RenderContext, Transform
from moviepy.ae.blend import blend
from moviepy.ae.renderer import Renderer
from moviepy.video.VideoClip import VideoClip


ROI = (37, 23, 121, 97)


def gradient_comp():
    """Return a 160x120 composition over a smooth opaque gradient."""
    yy, xx = np.mgrid[0:120, 0:160]
    image = np.stack(
        [xx * 255 // 160, yy * 255 // 120, (xx + yy) * 255 // 280], axis=-1
    ).astype(np.uint8)
    comp = Composition(size=(160, 120), fps=10, duration=2)
    comp.add_clip(VideoClip(lambda t: image, duration=2).with_fps(10), "bg")
    return comp


def assert_roi_matches_full(comp, t=0.5, atol=1e-5):
    full = comp.render_buffer(t).crop(ROI).rgba
    part = comp.render_buffer(t, bounds=ROI).rgba
    np.testing.assert_allclose(part, full, atol=atol)


def random_buffer(seed, shape=(40, 60)):
    rgba = np.random.default_rng(seed).random(shape + (4,)).astype(np.float32)
    rgba[..., :3] *= rgba[..., 3:4]
    return Buffer(rgba)


# -- R1: Dissolve noise must not depend on buffer bounds ---------------------- #


def test_r1_blend_dissolve_is_invariant_to_cropping():
    base, layer = random_buffer(1), random_buffer(2)
    region = (7, 5, 41, 33)
    for mode in ("dissolve", "dancing_dissolve"):
        whole = blend(base, layer, mode).crop(region).rgba
        part = blend(base.crop(region), layer.crop(region), mode).rgba
        np.testing.assert_array_equal(part, whole)


def test_r1_dissolve_layer_roi_equals_crop_of_full_render():
    comp = gradient_comp()
    layer = comp.add_solid("grain", color=(255, 255, 255))
    layer.transform = Transform(opacity=50)
    layer.blend_mode = "dissolve"
    assert_roi_matches_full(comp)


# -- R2: layers sharing a name must not share random streams ------------------ #


def test_r2_same_name_dissolve_layers_use_independent_noise():
    comp = Composition(size=(64, 64), fps=10, duration=1)
    for color in ((255, 0, 0), (0, 255, 0)):
        layer = comp.add_solid("Solid", color=color)
        layer.transform = Transform(opacity=50)
        layer.blend_mode = "dissolve"
    frame = comp.render_buffer(0).rgba
    red_visible = float((frame[..., 0] > 0.5).mean())
    # Independent 50% fields leave red where it was drawn and green was not.
    assert 0.15 < red_visible < 0.35


# -- R3: adjustment-layer effects and the region of interest ------------------ #


def test_r3_adjustment_blur_roi_equals_crop_of_full_render():
    comp = gradient_comp()
    comp.add_adjustment("adj").effects = [ae.fx.GaussianBlur(blurriness=12)]
    assert_roi_matches_full(comp, atol=1e-4)


# -- R4: pixel-unit effect parameters must follow resolution_scale ------------ #


def _edge_width(scale):
    comp = Composition(size=(160, 120), fps=10, duration=1)
    solid = comp.add_solid("s", color=(255, 255, 255), size=(80, 120))
    solid.transform = Transform(position=(40, 60))
    comp.add_adjustment("adj").effects = [ae.fx.GaussianBlur(blurriness=12)]
    context = RenderContext(resolution_scale=scale)
    row = comp.render_buffer(0, context).rgba[int(60 * scale), :, 3]
    return float(((row > 0.1) & (row < 0.9)).sum()) / scale


def test_r4_adjustment_blur_width_is_independent_of_resolution_scale():
    full = _edge_width(1.0)
    for scale in (0.5, 0.25):
        assert abs(_edge_width(scale) - full) <= 0.25 * full


# -- R5: cubic resampling into a region of interest --------------------------- #


def test_r5_rotated_layer_roi_equals_crop_at_best_quality():
    comp = Composition(size=(160, 120), fps=10, duration=2)
    solid = comp.add_solid("s", color=(255, 0, 0), size=(80, 60))
    solid.transform = Transform(position=(80, 60), rotation=30)
    assert_roi_matches_full(comp)


# -- R6: one render must not repeat identical (layer, time) work -------------- #


def _count_renders(comp, target, t):
    calls = []
    original = Renderer.render_layer

    def counting(self, layer, context, view, roi):
        if layer is target:
            calls.append(round(context.t, 6))
        return original(self, layer, context, view, roi)

    Renderer.render_layer = counting
    try:
        comp.get_frame(t)
    finally:
        Renderer.render_layer = original
    return calls


def test_r6_shared_track_matte_renders_once_per_frame():
    comp = Composition(size=(64, 36), fps=30, duration=2)
    matte = comp.add_solid("matte", color=(255, 255, 255), size=(30, 20))
    matte.transform = Transform(position=(32, 18), rotation=10)
    for index in range(3):
        comp.add_solid(f"fill{index}", color=(255, 0, 0)).set_track_matte(matte)
    assert len(_count_renders(comp, matte, 0.5)) == 1


def test_r6_stacked_temporal_adjustments_render_each_time_once():
    comp = Composition(size=(64, 36), fps=30, duration=2)
    solid = comp.add_solid("s", color=(255, 255, 0), size=(10, 10))
    for name in ("echo1", "echo2"):
        comp.add_adjustment(name).effects = [
            ae.fx.Echo(echo_time=-0.05, number_of_echoes=6)
        ]
    calls = _count_renders(comp, solid, 1.0)
    # Two stacked 6-echo layers need 13 distinct times (0..12 steps back).
    assert len(calls) == len(set(calls)) <= 13
