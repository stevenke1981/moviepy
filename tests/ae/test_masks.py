"""WS-05 tests: mask paths, rasterization, mask modes and track mattes."""

import math

import numpy as np

import pytest

import moviepy.ae as ae
from moviepy.ae import Buffer, Composition, RenderContext
from moviepy.ae.layers import SolidLayer
from moviepy.ae.masks import (
    Mask,
    MaskMode,
    MaskStack,
    MatteMode,
    TrackMatte,
    matte_values,
    path as shapes,
)
from moviepy.ae.masks.matte import srgb_to_linear
from moviepy.ae.masks.rasterize import margin, rasterize
from moviepy.ae.properties import PathValue, Property
from moviepy.ae.transform import Transform


def uniform(color, alpha=1.0, size=(4, 4), offset=(0, 0)):
    """Return a premultiplied buffer filled with straight ``color`` (0..1)."""
    width, height = size
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(color, dtype=np.float32) * alpha
    rgba[..., 3] = alpha
    return Buffer(rgba, offset)


def corner(position=(0.0, 0.0)):
    return Transform(position=position, anchor_point=(0.0, 0.0))


def square(left, top, right, bottom, mode="add", **kwargs):
    center = ((left + right - 1) / 2.0, (top + bottom - 1) / 2.0)
    return Mask.rect(center, (right - left, bottom - top), mode=mode, **kwargs)


# --------------------------------------------------------------------------- #
# paths


def test_rect_path_has_four_straight_corners():
    path = shapes.rect((5.0, 3.0), (4.0, 2.0))
    assert isinstance(path, PathValue) and path.closed
    assert path.vertices == ((3.0, 2.0), (7.0, 2.0), (7.0, 4.0), (3.0, 4.0))
    assert all(t == (0.0, 0.0) for t in path.in_tangents + path.out_tangents)


def test_ellipse_uses_kappa_tangents():
    path = shapes.ellipse((0.0, 0.0), (2.0, 2.0))
    assert len(path.vertices) == 4
    lengths = [math.hypot(*t) for t in path.out_tangents]
    assert lengths == pytest.approx([shapes.KAPPA] * 4)


def test_star_alternates_radii():
    path = shapes.star((0.0, 0.0), 5, 10.0, 4.0)
    radii = [math.hypot(*v) for v in path.vertices]
    assert len(radii) == 10
    assert radii[0::2] == pytest.approx([10.0] * 5)
    assert radii[1::2] == pytest.approx([4.0] * 5)


def test_rounded_rect_clamps_roundness():
    path = shapes.rounded_rect((0.0, 0.0), (10.0, 4.0), 100.0)
    xs = [v[0] for v in path.vertices]
    ys = [v[1] for v in path.vertices]
    assert min(xs) == pytest.approx(-5.0) and max(ys) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        shapes.rounded_rect((0.0, 0.0), (10.0, 4.0), -1.0)


def test_polygon_needs_two_points():
    with pytest.raises(ValueError):
        shapes.polygon([(0, 0)])
    assert len(shapes.polygon([(0, 0), (1, 1)], closed=False).vertices) == 2


def test_svg_path_commands_match_shapes():
    path = shapes.from_svg_path("M 1 1 H 5 V 4 L 1 4 Z")
    assert path.vertices == ((1.0, 1.0), (5.0, 1.0), (5.0, 4.0), (1.0, 4.0))
    relative = shapes.from_svg_path("m1,1 h4 v3 h-4 z")
    assert relative.vertices == path.vertices


def test_svg_curves_flatten_close_to_circle():
    d = "M 10 0 C 10 5.5228 5.5228 10 0 10 S -10 5.5228 -10 0 Q -10 -10 0 -10 Z"
    points = shapes.flatten(shapes.from_svg_path(d))
    assert points.shape[1] == 2 and len(points) > 16
    assert np.abs(points).max() <= 10.0 + 1e-6


@pytest.mark.parametrize(
    "d",
    [
        "",
        "L 1 1",
        "M 0 0 L 1",
        "M 0 0 X 1 1",
        "M0 0 A 1 1 0 0 1 2 2",
        "M 0 0 L 1 1 M 3 3 L 4 4 Z",
    ],
)
def test_svg_path_errors(d):
    with pytest.raises(ValueError):
        shapes.from_svg_path(d)


# --------------------------------------------------------------------------- #
# rasterization


def test_circle_area_error_below_half_percent():
    radius = 40.0
    path = shapes.ellipse((50.0, 50.0), (2 * radius, 2 * radius))
    values = rasterize(path, (0, 0, 100, 100))
    expected = math.pi * radius**2
    assert abs(values.sum() - expected) / expected < 0.005


def test_axis_aligned_rect_is_exact():
    values = square(2, 1, 6, 4).coverage((0, 0, 8, 6))
    expected = np.zeros((6, 8), dtype=np.float32)
    expected[1:4, 2:6] = 1.0
    np.testing.assert_array_equal(values, expected)


def test_half_pixel_edge_gives_half_coverage():
    path = shapes.polygon([(-0.5, -0.5), (1.0, -0.5), (1.0, 0.5), (-0.5, 0.5)])
    values = rasterize(path, (0, 0, 3, 1))
    np.testing.assert_allclose(values[0], [1.0, 0.5, 0.0], atol=1e-6)


def test_feather_matches_gaussian_sigma_half_feather():
    feather = 8.0
    path = shapes.polygon([(-0.5, -50), (100, -50), (100, 50), (-0.5, 50)])
    values = rasterize(path, (-40, 0, 40, 1), feather=(feather, feather))
    row = values[0].astype(np.float64)
    xs = np.arange(-40, 40)
    sigma = feather / 2.0
    erf = np.vectorize(math.erf)
    expected = 0.5 * (1.0 + erf((xs + 0.5) / (sigma * math.sqrt(2.0))))
    assert np.abs(row - expected).max() < 0.02
    assert margin((feather, feather)) >= 3 * sigma


def test_expansion_grows_and_shrinks():
    base = square(10, 10, 20, 20)
    area = base.coverage((0, 0, 30, 30)).sum()
    grown = square(10, 10, 20, 20, expansion=2.0).coverage((0, 0, 30, 30)).sum()
    shrunk = square(10, 10, 20, 20, expansion=-2.0).coverage((0, 0, 30, 30)).sum()
    assert area == pytest.approx(100.0)
    assert grown == pytest.approx(14 * 14, abs=4.0 * 4)
    assert shrunk == pytest.approx(36.0, abs=1.0)


def test_coverage_honors_world_bounds_offset():
    mask = square(2, 2, 4, 4)
    values = mask.coverage((2, 2, 6, 6))
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0:2, 0:2] = 1.0
    np.testing.assert_array_equal(values, expected)


def test_draft_quality_still_close():
    mask = Mask.ellipse((20.0, 20.0), (30.0, 30.0))
    full = mask.coverage((0, 0, 40, 40))
    draft = mask.coverage((0, 0, 40, 40), context=RenderContext(quality="draft"))
    assert abs(full.sum() - draft.sum()) / full.sum() < 0.01


# --------------------------------------------------------------------------- #
# mask modes


BOUNDS = (0, 0, 4, 1)


def bit_masks(mode):
    """Left two pixels in A, pixels 1 and 2 in B -> truth table per column."""
    first = square(0, 0, 2, 1)
    second = square(1, 0, 3, 1, mode=mode)
    return MaskStack([first, second]).coverage(BOUNDS)[0].tolist()


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("add", [1, 1, 1, 0]),
        ("subtract", [1, 0, 0, 0]),
        ("intersect", [0, 1, 0, 0]),
        ("difference", [1, 0, 1, 0]),
        ("lighten", [1, 1, 1, 0]),
        ("darken", [0, 1, 0, 0]),
        ("none", [1, 1, 0, 0]),
    ],
)
def test_mode_boolean_truth_tables(mode, expected):
    assert bit_masks(mode) == expected


@pytest.mark.parametrize("mode", ["subtract", "intersect", "darken"])
def test_first_subtractive_mask_starts_opaque(mode):
    values = MaskStack([square(1, 0, 3, 1, mode=mode)]).coverage(BOUNDS)[0]
    expected = {"subtract": [1, 0, 0, 1], "intersect": [0, 1, 1, 0]}
    expected["darken"] = expected["intersect"]
    assert values.tolist() == expected[mode]


def test_partial_modes_follow_ae_formulas():
    first = square(0, 0, 4, 1, opacity=60.0)
    second = square(0, 0, 4, 1, opacity=50.0)
    rules = {
        "add": 0.6 + 0.5 - 0.3,
        "subtract": 0.6 * 0.5,
        "intersect": 0.3,
        "lighten": 0.6,
        "darken": 0.5,
        "difference": 0.1,
    }
    for mode, value in rules.items():
        second.mode = mode
        values = MaskStack([first, second]).coverage(BOUNDS)
        np.testing.assert_allclose(values, value, atol=1e-6)


def test_inverted_mask_and_empty_stack():
    values = square(1, 0, 3, 1, inverted=True).coverage(BOUNDS)[0]
    assert values.tolist() == [1, 0, 0, 1]
    stack = MaskStack([square(0, 0, 1, 1, mode="none")])
    assert stack.coverage(BOUNDS) is None
    buffer = uniform((1, 0, 0))
    assert stack.apply(buffer) is buffer


def test_mask_validation():
    with pytest.raises(ValueError):
        MaskMode.coerce("multiply")
    with pytest.raises(TypeError):
        MaskStack([object()])
    with pytest.raises(ValueError):
        square(0, 0, 1, 1, opacity=150.0)
    assert MaskMode.coerce(" Intersect ") is MaskMode.INTERSECT


def test_stack_apply_multiplies_premultiplied_pixels():
    buffer = uniform((1.0, 0.5, 0.0), alpha=0.8)
    masked = MaskStack([square(0, 0, 2, 4, opacity=50.0)]).apply(buffer)
    np.testing.assert_allclose(masked.rgba[0, 0], [0.4, 0.2, 0.0, 0.4], atol=1e-6)
    np.testing.assert_array_equal(masked.rgba[:, 2:], 0.0)
    assert masked.offset == buffer.offset


def test_animated_path_keyframes_interpolate():
    start = shapes.rect((1.5, 1.5), (2.0, 2.0))
    end = shapes.rect((5.5, 1.5), (2.0, 2.0))
    mask = Mask(Property(start, keyframes=[(0, start), (1, end)]))
    middle = mask.coverage((0, 0, 9, 5), t=0.5)
    assert middle.sum() == pytest.approx(4.0)
    assert middle[1:3, 3:5].min() == 1.0


def test_animated_feather_and_opacity():
    mask = square(0, 0, 4, 1)
    mask.opacity = Property(0.0, keyframes=[(0, 0.0), (2, 100.0)])
    assert mask.coverage(BOUNDS, t=1.0).max() == pytest.approx(0.5)


def test_mask_to_clip_matches_coverage():
    mask = Mask.ellipse((5.0, 4.0), (6.0, 6.0), feather=2.0)
    clip = mask.to_clip((10, 8), duration=1.0, fps=5)
    assert clip.is_mask and clip.size == (10, 8)
    frame = clip.get_frame(0.2)
    np.testing.assert_allclose(frame, mask.coverage((0, 0, 10, 8)), atol=1e-7)


def test_mask_rasterization_is_deterministic_and_cached():
    mask = Mask.star((16.0, 16.0), 5, 14.0, 6.0, feather=3.0)
    first = mask.coverage((0, 0, 32, 32))
    second = mask.coverage((0, 0, 32, 32))
    np.testing.assert_array_equal(first, second)
    assert not first.flags.writeable


# --------------------------------------------------------------------------- #
# mattes


def test_matte_mode_values():
    buffer = uniform((1.0, 1.0, 1.0), alpha=0.25)
    assert matte_values(buffer, "alpha")[0, 0] == pytest.approx(0.25)
    assert matte_values(buffer, "alpha_inverted")[0, 0] == pytest.approx(0.75)
    assert matte_values(buffer, "luma")[0, 0] == pytest.approx(0.25)
    assert matte_values(buffer, "Luma Inverted")[0, 0] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        MatteMode.coerce("silhouette")


@pytest.mark.parametrize("gray, expected", [(1.0, 1.0), (0.0, 0.0), (0.5, 0.5)])
def test_luma_matte_white_black_gray(gray, expected):
    buffer = uniform((gray, gray, gray))
    assert matte_values(buffer, "luma")[0, 0] == pytest.approx(expected, abs=1e-6)


def test_linear_luma_of_mid_gray():
    buffer = uniform((0.5, 0.5, 0.5), alpha=1.0)
    value = matte_values(buffer, "luma", linear=True)[0, 0]
    assert value == pytest.approx(0.214, abs=1e-3)
    assert srgb_to_linear(0.04) == pytest.approx(0.04 / 12.92)


def test_rec601_coefficients_differ_from_rec709():
    green = uniform((0.0, 1.0, 0.0))
    assert matte_values(green, "luma")[0, 0] == pytest.approx(0.7152)
    value = matte_values(green, "luma", coefficients="rec601")[0, 0]
    assert value == pytest.approx(0.587)
    with pytest.raises(ValueError):
        matte_values(green, "luma", coefficients="rec2020")


def test_track_matte_validation():
    target = SolidLayer("t", color=(255, 0, 0), size=(4, 4))
    with pytest.raises(TypeError):
        TrackMatte("layer")
    with pytest.raises(TypeError):
        target.track_matte = "matte"
    with pytest.raises(ValueError):
        target.set_track_matte(target)
    other = SolidLayer("o", color=(0, 0, 0), size=(4, 4))
    target.set_track_matte(other)
    with pytest.raises(ValueError):
        other.set_track_matte(target)
    with pytest.raises(TypeError):
        TrackMatte(other, linear="yes")


# --------------------------------------------------------------------------- #
# layer and renderer integration


def make_comp(size=(8, 6)):
    return Composition(size=size, fps=10, duration=1.0, bg_color=(0, 0, 0))


def test_layer_masks_render_in_layer_space():
    comp = make_comp()
    layer = comp.add_solid("red", color=(255, 0, 0), size=(4, 4))
    layer.transform = corner((3.0, 1.0))
    layer.masks = [square(0, 0, 2, 4)]
    frame = comp.get_frame(0)
    assert frame[1:5, 3:5, 0].min() == 255
    assert frame[1:5, 5:7].max() == 0
    assert isinstance(layer.masks, MaskStack) and len(layer.masks) == 1


def test_layer_masks_follow_layer_time():
    comp = make_comp()
    layer = comp.add_solid("red", color=(255, 0, 0), size=(8, 6), start_time=0.5)
    mask = square(0, 0, 8, 6)
    mask.opacity = Property(0.0, keyframes=[(0, 0.0), (0.5, 100.0)])
    layer.masks.add(mask)
    assert comp.get_frame(0.5)[0, 0, 0] == 0
    assert comp.get_frame(0.75)[0, 0, 0] == pytest.approx(128, abs=1)


def test_single_layer_render_applies_masks():
    layer = SolidLayer(
        "s", color=(255, 255, 255), size=(4, 4), masks=Mask.rect((0.5, 1.5), (2.0, 4.0))
    )
    layer.transform = corner()
    buffer = layer.render(0.0)
    np.testing.assert_array_equal(buffer.rgba[:, :2, 3], 1.0)
    np.testing.assert_array_equal(buffer.rgba[:, 2:, 3], 0.0)


def matte_scene(mode, matte_color=(255, 255, 255)):
    comp = make_comp()
    target = comp.add_solid("target", color=(255, 0, 0))
    matte = comp.add_solid("matte", color=matte_color, size=(4, 6))
    matte.transform = corner()
    target.set_track_matte(matte, mode)
    return comp, target, matte


@pytest.mark.parametrize(
    "mode, left, right",
    [
        ("alpha", 255, 0),
        ("alpha_inverted", 0, 255),
        ("luma", 255, 0),
        ("luma_inverted", 0, 255),
    ],
)
def test_track_matte_modes_in_renderer(mode, left, right):
    comp, _, matte = matte_scene(mode)
    assert matte.enabled is False
    frame = comp.get_frame(0)
    assert frame[:, :4, 0].min() == left and frame[:, :4, 0].max() == left
    assert frame[:, 4:, 0].min() == right and frame[:, 4:, 0].max() == right


def test_gray_luma_matte_halves_target():
    comp, _, _ = matte_scene("luma", matte_color=(128, 128, 128))
    frame = comp.get_frame(0)
    assert frame[0, 0, 0] == pytest.approx(128, abs=1)


def test_matte_respects_its_masks_and_time_window():
    comp, _, matte = matte_scene("alpha")
    matte.masks = [square(0, 0, 2, 6)]
    frame = comp.get_frame(0)
    assert frame[0, :2, 0].tolist() == [255, 255]
    assert frame[0, 2:, 0].max() == 0
    matte.in_point = 0.5
    assert comp.get_frame(0.2).max() == 0
    assert comp.get_frame(0.7)[0, 0, 0] == 255


def test_nested_track_mattes_multiply():
    comp = make_comp()
    target = comp.add_solid("target", color=(255, 0, 0))
    first = comp.add_solid("first", color=(255, 255, 255), size=(6, 6))
    second = comp.add_solid("second", color=(255, 255, 255), size=(4, 6))
    for layer in (first, second):
        layer.transform = corner()
    target.set_track_matte(first, "alpha")
    first.set_track_matte(second, "alpha_inverted")
    frame = comp.get_frame(0)
    assert frame[0].tolist() == [[0, 0, 0]] * 4 + [[255, 0, 0]] * 2 + [[0, 0, 0]] * 2


def test_disabled_matte_without_hide_still_renders():
    comp = make_comp()
    target = comp.add_solid("target", color=(255, 0, 0), size=(4, 6))
    target.transform = corner()
    matte = comp.add_solid("matte", color=(0, 0, 255), size=(4, 6))
    matte.transform = corner((4.0, 0.0))
    target.set_track_matte(matte, "alpha_inverted", hide=False)
    frame = comp.get_frame(0)
    assert frame[0, 0].tolist() == [255, 0, 0]
    assert frame[0, 7].tolist() == [0, 0, 255]


def test_facade_exports_ws05_names():
    for name in ["Mask", "MaskStack", "MaskMode", "TrackMatte", "MatteMode"]:
        assert hasattr(ae, name)
