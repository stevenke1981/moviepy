"""Enabled Stencil outside an ROI still masks the accumulated backdrop."""

import numpy as np

import pytest

from moviepy.ae import Composition, RenderContext, Transform
from moviepy.ae.blend import blend


def scene(mode, space, position=(3, 3)):
    comp = Composition(size=(8, 8), duration=1, transparent=True)
    comp.context = RenderContext(working_space=space)
    comp.add_solid(color=(255, 255, 255))
    stencil = comp.add_solid(
        size=(2, 2),
        color=(255, 255, 255),
        blend_mode=mode,
        transform=Transform(anchor_point=(0, 0), position=position, opacity=50),
    )
    return comp, stencil


@pytest.mark.parametrize("mode", ["stencil_alpha", "stencil_luma"])
@pytest.mark.parametrize("space", ["srgb", "linear"])
@pytest.mark.parametrize("position", [(3, 3), (-1, -1), (20, 20)])
@pytest.mark.parametrize("roi", [(0, 0, 2, 2), (2, 2, 5, 5), (3, 3, 5, 5)])
def test_roi_is_exact_crop_of_full_stencil(mode, space, position, roi):
    comp, _ = scene(mode, space, position)
    full = comp.render_buffer(0.5)
    part = comp.render_buffer(0.5, bounds=roi)
    np.testing.assert_array_equal(part.rgba, full.crop(roi).rgba)
    if position == (20, 20):
        np.testing.assert_array_equal(full.rgba, np.zeros((8, 8, 4)))


@pytest.mark.parametrize("mode", ["stencil_alpha", "stencil_luma"])
@pytest.mark.parametrize("state", ["disabled", "inactive", "guide"])
def test_excluded_stencil_has_no_influence(mode, state):
    comp, stencil = scene(mode, "srgb", (20, 20))
    if state == "disabled":
        stencil.enabled = False
    elif state == "inactive":
        stencil.in_point = 0.75
    else:
        stencil.guide = True
    np.testing.assert_array_equal(comp.render_buffer(0.5).rgba, np.ones((8, 8, 4)))


@pytest.mark.parametrize("scale", [0.25, 0.5, 1.0])
def test_offscreen_stencil_survives_resolution_scaling(scale):
    comp, _ = scene("stencil_alpha", "linear", (20, 20))
    ctx = RenderContext(resolution_scale=scale, working_space="linear")
    actual = comp.render_buffer(0.5, ctx)
    assert not np.any(actual.rgba)


@pytest.mark.parametrize("mode", ["stencil_alpha", "stencil_luma"])
@pytest.mark.parametrize("space", ["srgb", "linear"])
@pytest.mark.parametrize("position", [(3, 3), (20, 20)])
@pytest.mark.parametrize("opacity", [0.0, 1e-8, 0.5, 1.0])
def test_stencil_opacity_policy_matches_direct_and_composition(
    mode, space, position, opacity
):
    comp, stencil = scene(mode, space, position)
    stencil.enabled = False
    base = comp.render_buffer(0.5)
    stencil.enabled = True
    stencil.color = (128, 128, 128)
    stencil.transform.opacity = 100
    source = stencil.render(0.5, comp.context)
    stencil.transform.opacity = opacity * 100
    direct = blend(base, source, mode, opacity)
    actual = comp.render_buffer(0.5)
    if opacity == 0:
        assert direct is base
        expected = base.rgba
    else:
        expected = np.zeros_like(base.rgba)
        if position == (3, 3):
            level = 128 / 255
            if space == "linear":
                level = ((level + 0.055) / 1.055) ** 2.4
            coverage = opacity * (level if mode == "stencil_luma" else 1)
            expected[3:5, 3:5] = base.rgba[3:5, 3:5] * coverage
            assert actual.rgba[..., 3].max() > 0
    np.testing.assert_allclose(direct.rgba, expected, rtol=1e-6, atol=1e-14)
    np.testing.assert_allclose(actual.rgba, expected, rtol=1e-6, atol=1e-14)
    np.testing.assert_allclose(actual.rgba, direct.rgba, rtol=1e-6, atol=1e-14)
    np.testing.assert_array_equal(
        comp.render_buffer(0.5, bounds=(0, 0, 2, 2)).rgba,
        actual.crop((0, 0, 2, 2)).rgba,
    )


@pytest.mark.parametrize("mode", ["stencil_alpha", "stencil_luma"])
@pytest.mark.parametrize("space", ["srgb", "linear"])
def test_stencil_keyframes_cross_zero_with_the_selected_discontinuity(mode, space):
    comp, stencil = scene(mode, space)
    stencil.enabled = False
    base = comp.render_buffer(0.5)
    stencil.enabled = True
    stencil.transform.opacity = 100
    source = stencil.render(0.5, comp.context)
    stencil.transform.opacity = [(0, 0.0001), (0.5, 0), (1, 0.0001)]
    results = {}
    for t in (0.499, 0.5, 0.501, 0.25, 0.75, 0.5):
        opacity = 1e-6 * abs(t - 0.5) * 2
        expected = blend(base, source, mode, opacity)
        actual = comp.render_buffer(t)
        np.testing.assert_allclose(actual.rgba, expected.rgba, rtol=1e-6, atol=1e-14)
        if t == 0.5:
            np.testing.assert_array_equal(actual.rgba, base.rgba)
        else:
            assert 0 < actual.rgba[..., 3].max() < 1e-6
            np.testing.assert_array_equal(actual.rgba[:2], 0)
        results[t] = actual.rgba.copy()
    assert results[0.5][3, 3, 3] == 1
    assert results[0.499][3, 3, 3] < 1e-8
    assert results[0.501][3, 3, 3] < 1e-8


@pytest.mark.parametrize("mode", ["stencil_alpha", "stencil_luma"])
@pytest.mark.parametrize("space", ["srgb", "linear"])
@pytest.mark.parametrize(
    "keys,opacities,outside_alpha",
    [
        ([(0, 0), (0.5, 0), (1, 100)], (0, 0, 0.25, 0.75), 0.5),
        ([(0, 25), (1, 75)], (0.3125, 0.4375, 0.5625, 0.6875), 0),
    ],
)
def test_stencil_shutter_averages_composited_keyframe_samples(
    mode, space, keys, opacities, outside_alpha
):
    comp, stencil = scene(mode, space)
    comp.fps = 1
    stencil.enabled = False
    base = comp.render_buffer(0.5)
    stencil.enabled = True
    stencil.color = (128, 128, 128)
    stencil.transform.opacity = 100
    source = stencil.render(0.5, comp.context)
    stencil.transform.opacity = keys
    stencil.motion_blur = True
    comp.motion_blur = True
    comp.context = RenderContext(
        working_space=space, shutter={"angle": 360, "phase": -180, "samples": 4}
    )
    expected = np.mean(
        [
            blend(base, source, mode, opacity).rgba.astype(np.float64)
            for opacity in opacities
        ],
        axis=0,
    )
    actual = comp.render_buffer(0.5)
    np.testing.assert_allclose(actual.rgba, expected, rtol=1e-6, atol=1e-14)
    np.testing.assert_array_equal(actual.rgba[:2], outside_alpha)
    np.testing.assert_array_equal(
        comp.render_buffer(0.5, bounds=(0, 0, 2, 2)).rgba,
        actual.crop((0, 0, 2, 2)).rgba,
    )
