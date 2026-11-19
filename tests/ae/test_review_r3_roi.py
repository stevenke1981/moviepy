"""Adjustment effects read neighbors outside the requested render region."""

import numpy as np

import pytest

import moviepy.ae as ae
from moviepy import VideoClip


REGIONS = ((37, 23, 121, 97), (0, 0, 60, 50), (130, 90, 160, 120))


def _scene():
    yy, xx = np.mgrid[0:120, 0:160]
    image = np.stack(
        [xx * 255 // 160, yy * 255 // 120, ((xx // 11 + yy // 9) % 2) * 255],
        axis=-1,
    ).astype(np.uint8)
    comp = ae.Composition(size=(160, 120), fps=10, duration=2)
    comp.add_clip(
        VideoClip(lambda t: np.roll(image, round(t * 20), axis=1), duration=2),
        "background",
    )
    return comp


def _assert_region(comp, roi, t=0.8):
    expected = comp.render_buffer(t).crop(roi)
    actual = comp.render_buffer(t, bounds=roi)
    assert actual.bounds == roi
    np.testing.assert_allclose(actual.rgba, expected.rgba, atol=1e-5)


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("roi", REGIONS)
def test_adjustment_blur_reads_roi_neighbors_at_canvas_edges(repeat, roi):
    comp = _scene()
    comp.add_adjustment().effects = [
        ae.fx.GaussianBlur(blurriness=12, repeat_edge_pixels=repeat)
    ]
    _assert_region(comp, roi)


@pytest.mark.parametrize("same_layer", [False, True])
@pytest.mark.parametrize("repeats", [(False, True), (True, False)])
@pytest.mark.parametrize("roi", REGIONS[:2])
def test_adjustment_blur_accumulates_stack_input_margins(same_layer, repeats, roi):
    comp = _scene()
    effects = [
        ae.fx.GaussianBlur(blurriness=12, repeat_edge_pixels=repeat)
        for repeat in repeats
    ]
    if same_layer:
        comp.add_adjustment().effects = effects
    else:
        for effect in effects:
            comp.add_adjustment().effects = [effect]
    _assert_region(comp, roi)


@pytest.mark.parametrize("same_layer", [False, True])
@pytest.mark.parametrize("repeat", [False, True])
def test_temporal_adjustment_recomputes_animated_input_margins(same_layer, repeat):
    comp = _scene()
    blur = ae.fx.GaussianBlur(blurriness=[(0, 24), (1, 2)], repeat_edge_pixels=repeat)
    echo = ae.fx.Echo(echo_time=-0.15, number_of_echoes=2, echo_operator="blend")
    if same_layer:
        comp.add_adjustment().effects = [blur, echo]
    else:
        comp.add_adjustment().effects = [blur]
        comp.add_adjustment().effects = [echo]
    comp.add_adjustment().effects = [
        ae.fx.GaussianBlur(blurriness=4, repeat_edge_pixels=True)
    ]
    _assert_region(comp, REGIONS[0])


def test_nested_temporal_reads_keep_another_times_larger_spatial_margin():
    comp = _scene()
    blur = ae.fx.GaussianBlur(
        blurriness=[(0, 0), (0.6, 0), (0.8, 24), (1, 0)],
        repeat_edge_pixels=True,
    )
    comp.add_adjustment().effects = [
        ae.fx.Echo(echo_time=-0.2, number_of_echoes=1, echo_operator="blend"),
        blur,
        ae.fx.Echo(echo_time=-0.2, number_of_echoes=1, echo_operator="blend"),
    ]
    # At t=1 the blur reads no neighbors, but the outer echo replays the
    # t=0.8 blur, which needs neighbors from both t=0.8 and t=0.6.
    assert blur.input_margin(comp.size, 1) == (0, 0, 0, 0)
    _assert_region(comp, REGIONS[0], t=1)


def test_adjustment_margin_uses_the_effects_exact_stretched_time():
    comp = ae.Composition(size=(160, 120), fps=10, duration=3)
    comp.add_solid("white", color=(255, 255, 255))
    adjustment = comp.add_adjustment(start_time=0.1, stretch=75)
    adjustment.effects = [
        ae.fx.GaussianBlur(blurriness=lambda t: 24 if t >= 1.6 else 0)
    ]
    # The effect evaluates at 1.6. An intermediate float conversion in the
    # planner maps to the preceding float and would omit the entire halo.
    _assert_region(comp, REGIONS[0], t=1.3)


@pytest.mark.parametrize("repeat", [False, True])
def test_adjustment_roi_preserves_transformed_masked_matted_coverage(repeat):
    comp = _scene()
    adjustment = comp.add_adjustment(size=(100, 90))
    adjustment.effects = [ae.fx.GaussianBlur(blurriness=12, repeat_edge_pixels=repeat)]
    adjustment.transform = ae.Transform(position=(75, 55), rotation=10, opacity=60)
    adjustment.masks = [ae.Mask.rect((49.5, 44.5), (80, 70), feather=4)]
    matte = comp.add_solid("matte", color=(255, 255, 255), size=(105, 90))
    matte.transform = ae.Transform(position=(80, 60), rotation=-8)
    adjustment.set_track_matte(matte)
    _assert_region(comp, REGIONS[0])
