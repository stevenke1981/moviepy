"""WS-04 blend-mode tests: analytic vectors, alpha rules and determinism."""

import numpy as np

import pytest

from moviepy.ae import Buffer, RenderContext
from moviepy.ae.blend import BlendMode, blend
from moviepy.ae.layers import SolidLayer


def pixel(rgb, alpha=1.0, offset=(0, 0), size=(1, 1)):
    """Return a uniform premultiplied buffer from a straight color."""
    width, height = size
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(rgb, dtype=np.float32) * np.float32(alpha)
    rgba[..., 3] = alpha
    return Buffer(rgba, offset)


def gray(value, alpha=1.0, **kwargs):
    return pixel((value, value, value), alpha, **kwargs)


def straight_rgb(buffer):
    return buffer.straight()[0, 0, :3]


# --------------------------------------------------------------------------- #
# catalogue


def test_catalogue_contains_all_38_ae_modes_with_unique_labels():
    assert len(BlendMode) == 38
    labels = [mode.label for mode in BlendMode]
    assert len(set(labels)) == 38
    assert BlendMode.CLASSIC_COLOR_BURN.label == "Classic Color Burn"
    assert BlendMode.STENCIL_LUMA.category == "matte"
    assert {mode.category for mode in BlendMode} == {
        "normal",
        "darken",
        "lighten",
        "contrast",
        "difference",
        "hsl",
        "matte",
        "utility",
    }


@pytest.mark.parametrize(
    "token", ["Color Burn", "color-burn", "COLOR_BURN", " color burn ", "color_burn"]
)
def test_coerce_accepts_tokens_and_labels(token):
    assert BlendMode.coerce(token) is BlendMode.COLOR_BURN
    assert BlendMode.coerce(BlendMode.COLOR_BURN) is BlendMode.COLOR_BURN
    assert BlendMode.COLOR_BURN == "color_burn"
    assert str(BlendMode.COLOR_BURN) == "color_burn"


def test_coerce_rejects_unknown_and_non_string_values():
    with pytest.raises(ValueError, match="blend_mode"):
        BlendMode.coerce("sparkle")
    with pytest.raises(ValueError, match="blend_mode"):
        BlendMode.coerce("   ")
    with pytest.raises(TypeError, match="blend_mode"):
        BlendMode.coerce(3)


def test_layer_blend_mode_is_validated_and_canonicalized():
    layer = SolidLayer("s", size=(1, 1))
    layer.blend_mode = "Linear Light"
    assert layer.blend_mode == "linear_light"
    layer.blend_mode = BlendMode.SCREEN
    assert layer.blend_mode == "screen" and type(layer.blend_mode) is str
    with pytest.raises(ValueError, match="blend_mode"):
        layer.blend_mode = "sparkle"
    assert layer.blend_mode == "screen"
    assert layer.preserve_transparency is False
    with pytest.raises(TypeError, match="preserve_transparency"):
        layer.preserve_transparency = 1


# --------------------------------------------------------------------------- #
# analytic vectors on opaque gray pixels: (mode, backdrop, source, expected)

VECTORS = [
    ("normal", 0.2, 0.7, 0.7),
    ("multiply", 0.5, 0.5, 0.25),
    ("screen", 0.5, 0.5, 0.75),
    ("overlay", 0.25, 0.5, 0.25),
    ("overlay", 0.75, 0.5, 0.75),
    ("overlay", 0.25, 0.8, 0.4),
    ("darken", 0.3, 0.6, 0.3),
    ("lighten", 0.3, 0.6, 0.6),
    ("color_dodge", 0.25, 0.5, 0.5),
    ("color_dodge", 0.0, 1.0, 0.0),
    ("color_dodge", 0.4, 1.0, 1.0),
    ("color_burn", 0.75, 0.5, 0.5),
    ("color_burn", 1.0, 0.0, 1.0),
    ("color_burn", 0.6, 0.0, 0.0),
    ("classic_color_burn", 0.75, 0.5, 0.5),
    ("classic_color_burn", 1.0, 0.0, 0.0),
    ("classic_color_dodge", 0.25, 0.5, 0.5),
    ("classic_color_dodge", 0.0, 1.0, 1.0),
    ("linear_burn", 0.3, 0.4, 0.0),
    ("linear_burn", 0.8, 0.6, 0.4),
    ("linear_dodge", 0.3, 0.4, 0.7),
    ("linear_dodge", 0.8, 0.6, 1.0),
    ("hard_light", 0.4, 0.25, 0.2),
    ("hard_light", 0.4, 0.75, 0.7),
    ("soft_light", 0.5, 0.25, 0.375),
    ("soft_light", 0.16, 0.75, 0.279168),
    ("soft_light", 0.64, 0.75, 0.72),
    ("vivid_light", 0.6, 0.25, 0.2),
    ("vivid_light", 0.6, 0.75, 1.0),
    ("linear_light", 0.4, 0.6, 0.6),
    ("linear_light", 0.9, 0.9, 1.0),
    ("pin_light", 0.6, 0.25, 0.5),
    ("pin_light", 0.2, 0.75, 0.5),
    ("pin_light", 0.4, 0.4, 0.4),
    ("hard_mix", 0.4, 0.6, 1.0),
    ("hard_mix", 0.4, 0.5, 0.0),
    ("difference", 0.2, 0.7, 0.5),
    ("classic_difference", 0.2, 0.7, 0.5),
    ("exclusion", 0.5, 0.5, 0.5),
    ("exclusion", 0.2, 0.7, 0.62),
    ("subtract", 0.7, 0.2, 0.5),
    ("subtract", 0.2, 0.7, 0.0),
    ("divide", 0.25, 0.5, 0.5),
    ("divide", 0.6, 0.5, 1.0),
    ("divide", 0.6, 0.0, 1.0),
    ("darker_color", 0.3, 0.6, 0.3),
    ("lighter_color", 0.3, 0.6, 0.6),
    ("luminescent_premul", 0.3, 0.6, 0.6),
    ("alpha_add", 0.3, 0.6, 0.6),
]


@pytest.mark.parametrize("mode,backdrop,source,expected", VECTORS)
def test_analytic_vector(mode, backdrop, source, expected):
    result = blend(gray(backdrop), gray(source), mode)
    np.testing.assert_allclose(result.rgba[0, 0], [expected] * 3 + [1.0], atol=1e-6)


def test_add_keeps_hdr_values_until_export():
    result = blend(gray(0.5), gray(0.75), "add")
    np.testing.assert_allclose(result.rgba[0, 0, :3], 1.25, atol=1e-6)
    assert result.to_uint8_rgb()[0, 0].tolist() == [255, 255, 255]


def test_whole_color_modes_select_by_luminosity():
    red, green = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    darker = blend(pixel(red), pixel(green), "darker_color")
    lighter = blend(pixel(red), pixel(green), "lighter_color")
    np.testing.assert_allclose(straight_rgb(darker), red)
    np.testing.assert_allclose(straight_rgb(lighter), green)


def lum(rgb):
    return float(np.dot(rgb, [0.3, 0.59, 0.11]))


def test_color_mode_matches_w3c_set_lum_with_clip_color():
    result = blend(gray(0.5), pixel((1.0, 0.0, 0.0)), "color")
    low = 0.5 - 0.3 * 0.5 / 0.7
    np.testing.assert_allclose(straight_rgb(result), [1.0, low, low], atol=1e-6)
    assert lum(straight_rgb(result)) == pytest.approx(0.5, abs=1e-6)


def test_hsl_modes_take_the_documented_components():
    backdrop, source = (0.2, 0.4, 0.6), (0.5, 0.45, 0.4)
    luminosity = straight_rgb(blend(pixel(backdrop), pixel(source), "luminosity"))
    assert lum(luminosity) == pytest.approx(lum(source), abs=1e-6)
    color = straight_rgb(blend(pixel(backdrop), pixel(source), "color"))
    assert lum(color) == pytest.approx(lum(backdrop), abs=1e-6)
    hue = straight_rgb(blend(pixel(backdrop), gray(0.9), "hue"))
    np.testing.assert_allclose(hue, [lum(backdrop)] * 3, atol=1e-6)
    saturation = straight_rgb(blend(pixel(backdrop), gray(0.9), "saturation"))
    np.testing.assert_allclose(saturation, [lum(backdrop)] * 3, atol=1e-6)
    hue = straight_rgb(blend(gray(0.5), pixel((1.0, 0.0, 0.0)), "hue"))
    np.testing.assert_allclose(hue, [0.5] * 3, atol=1e-6)


# --------------------------------------------------------------------------- #
# alpha rules


@pytest.mark.parametrize("mode", [mode for mode in BlendMode if "stencil" not in mode])
def test_zero_opacity_returns_the_base_and_full_opacity_is_the_default(mode):
    base = pixel((0.2, 0.5, 0.8), 0.75, size=(3, 2))
    layer = pixel((0.9, 0.3, 0.1), 0.5, offset=(1, 0), size=(3, 2))
    assert blend(base, layer, mode, 0.0) is base
    np.testing.assert_array_equal(
        blend(base, layer, mode, 1.0).rgba, blend(base, layer, mode).rgba
    )


@pytest.mark.parametrize("mode", ["multiply", "screen", "overlay", "hue", "divide"])
def test_opacity_scales_the_layer_like_a_premultiplied_factor(mode):
    base = pixel((0.2, 0.5, 0.8))
    layer = pixel((0.9, 0.3, 0.1), 0.8)
    half = blend(base, layer, mode, 0.5)
    expected = blend(base, pixel((0.9, 0.3, 0.1), 0.4), mode)
    np.testing.assert_allclose(half.rgba, expected.rgba, atol=1e-6)


def test_normal_is_bitwise_identical_to_composite_over():
    rng = np.random.default_rng(0)
    base = Buffer(_random_rgba(rng, (5, 4)))
    layer = Buffer(_random_rgba(rng, (3, 6)), offset=(2, -1))
    expected = layer.composite_over(base)
    result = blend(base, layer, "normal")
    assert result.bounds == expected.bounds
    np.testing.assert_array_equal(result.rgba, expected.rgba)


def _random_rgba(rng, shape):
    alpha = rng.uniform(0.1, 1.0, shape + (1,)).astype(np.float32)
    rgb = rng.uniform(0, 1, shape + (3,)).astype(np.float32)
    return np.concatenate([rgb * alpha, alpha], axis=-1)


def test_semitransparent_multiply_follows_the_w3c_composite():
    result = blend(gray(0.6), gray(0.5, 0.5), "multiply")
    expected = 0.5 * 0.6 + 0.5 * 0.6 * 0.5
    np.testing.assert_allclose(result.rgba[0, 0], [expected] * 3 + [1.0], atol=1e-6)
    over_transparent = blend(gray(0.6, 0.5), gray(0.5, 0.5), "multiply")
    rgb = 0.5 * 0.5 * 0.5 + 0.5 * 0.3 + 0.25 * 0.3
    np.testing.assert_allclose(
        over_transparent.rgba[0, 0], [rgb] * 3 + [0.75], atol=1e-6
    )


@pytest.mark.parametrize(
    "mode",
    [
        mode
        for mode in BlendMode
        if mode.category in ("darken", "lighten", "contrast", "difference", "hsl")
    ],
)
def test_blending_onto_transparency_shows_the_plain_layer(mode):
    base = pixel((0.0, 0.0, 0.0), 0.0, size=(2, 2))
    layer = pixel((0.9, 0.3, 0.1), 0.6, size=(2, 2))
    result = blend(base, layer, mode)
    np.testing.assert_allclose(result.rgba, layer.rgba, atol=1e-6)
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))
    np.testing.assert_allclose(blend(empty, layer, mode).rgba, layer.rgba, atol=1e-6)


def test_union_bounds_place_layer_pixels_outside_the_base():
    base = gray(0.5, size=(2, 2))
    layer = gray(0.5, offset=(1, 1), size=(2, 2))
    result = blend(base, layer, "multiply")
    assert result.bounds == (0, 0, 3, 3)
    np.testing.assert_allclose(result.rgba[1, 1], [0.25, 0.25, 0.25, 1.0])
    np.testing.assert_allclose(result.rgba[0, 0], [0.5, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(result.rgba[2, 2], [0.5, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(result.rgba[0, 2], 0.0)


def test_preserve_underlying_transparency_keeps_backdrop_alpha():
    base = pixel((0.5, 0.5, 0.5), 0.5, size=(2, 1))
    layer = pixel((1.0, 0.0, 0.0), 1.0, offset=(1, 0), size=(3, 1))
    result = blend(base, layer, "normal", preserve_underlying_transparency=True)
    assert result.bounds == base.bounds
    np.testing.assert_allclose(result.rgba[..., 3], base.rgba[..., 3])
    np.testing.assert_allclose(result.rgba[0, 0], base.rgba[0, 0])
    np.testing.assert_allclose(result.straight()[0, 1, :3], [1.0, 0.0, 0.0], atol=1e-6)
    screen = blend(base, layer, "screen", preserve_underlying_transparency=True)
    np.testing.assert_allclose(screen.rgba[..., 3], base.rgba[..., 3])
    with pytest.raises(TypeError, match="preserve"):
        blend(base, layer, preserve_underlying_transparency="yes")


def test_stencil_and_silhouette_alpha_cut_the_backdrop():
    base = gray(0.8, size=(3, 1))
    layer = pixel((0.1, 0.2, 0.3), 0.25, offset=(1, 0), size=(1, 1))
    stencil = blend(base, layer, "stencil_alpha")
    np.testing.assert_allclose(stencil.rgba[..., 3], [[0.0, 0.25, 0.0]])
    np.testing.assert_allclose(stencil.rgba[0, 1], base.rgba[0, 1] * 0.25)
    silhouette = blend(base, layer, "silhouette_alpha")
    np.testing.assert_allclose(silhouette.rgba[..., 3], [[1.0, 0.75, 1.0]])
    assert stencil.bounds == silhouette.bounds == base.bounds


@pytest.mark.parametrize("value,expected", [(1.0, 1.0), (0.0, 0.0), (0.5, 0.5)])
def test_luma_stencil_maps_white_opaque_and_black_transparent(value, expected):
    result = blend(gray(0.8), gray(value), "stencil_luma")
    assert result.rgba[0, 0, 3] == pytest.approx(expected, abs=1e-6)
    silhouette = blend(gray(0.8), gray(value), "silhouette_luma")
    assert silhouette.rgba[0, 0, 3] == pytest.approx(1 - expected, abs=1e-6)


def test_luma_stencil_uses_rec709_coefficients():
    result = blend(gray(1.0), pixel((0.0, 1.0, 0.0)), "stencil_luma")
    assert result.rgba[0, 0, 3] == pytest.approx(0.7152, abs=1e-6)


def test_stencil_with_an_empty_layer_hides_everything():
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))
    result = blend(gray(0.8, size=(2, 2)), empty, "stencil_alpha")
    np.testing.assert_array_equal(result.rgba, 0.0)
    kept = blend(gray(0.8, size=(2, 2)), empty, "silhouette_alpha")
    np.testing.assert_allclose(kept.rgba, gray(0.8, size=(2, 2)).rgba)


def test_alpha_add_joins_complementary_edges_seamlessly():
    color = (0.2, 0.6, 0.4)
    result = blend(pixel(color, 0.3), pixel(color, 0.7), "alpha_add")
    assert result.rgba[0, 0, 3] == pytest.approx(1.0)
    np.testing.assert_allclose(result.rgba[0, 0, :3], color, atol=1e-6)
    normal = blend(pixel(color, 0.3), pixel(color, 0.7), "normal")
    assert normal.rgba[0, 0, 3] < 0.8


def test_classic_modes_match_modern_modes_for_opaque_interior_values():
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0.05, 0.95, (4, 4, 3)).astype(np.float32)
    base = Buffer(np.concatenate([rgb, np.ones((4, 4, 1), np.float32)], -1))
    layer = Buffer(np.concatenate([rgb[::-1], np.ones((4, 4, 1), np.float32)], -1))
    for classic, modern in (
        ("classic_color_burn", "color_burn"),
        ("classic_color_dodge", "color_dodge"),
        ("classic_difference", "difference"),
    ):
        np.testing.assert_allclose(
            blend(base, layer, classic).rgba, blend(base, layer, modern).rgba, atol=1e-6
        )


def test_classic_difference_mixes_premultiplied_colors():
    result = blend(gray(0.8), gray(0.4, 0.5), "classic_difference")
    expected = 0.5 * 0.8 + 0.5 * abs(0.8 - 0.2)
    assert result.rgba[0, 0, 0] == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- #
# dissolve


def test_dissolve_reveals_a_fraction_equal_to_alpha_deterministically():
    base = gray(0.0, size=(100, 100))
    layer = pixel((1.0, 0.5, 0.0), 0.5, size=(100, 100))
    context = RenderContext(t=0.0, fps=24, rng_seed=5)
    first = blend(base, layer, "dissolve", context=context, layer_id="fx")
    again = blend(base, layer, "dissolve", context=context, layer_id="fx")
    np.testing.assert_array_equal(first.rgba, again.rgba)
    revealed = first.rgba[..., 0] > 0.5
    assert abs(revealed.mean() - 0.5) < 0.03
    colors = first.rgba[revealed][:, :3]
    np.testing.assert_allclose(colors, np.broadcast_to([1.0, 0.5, 0.0], colors.shape))
    np.testing.assert_allclose(first.rgba[..., 3], 1.0)
    later = blend(base, layer, "dissolve", context=context.with_time(1.0))
    np.testing.assert_array_equal(
        blend(base, layer, "dissolve", context=context).rgba, later.rgba
    )


def test_dancing_dissolve_changes_per_frame_but_not_within_a_frame():
    base = gray(0.0, size=(20, 20))
    layer = gray(1.0, 0.5, size=(20, 20))
    context = RenderContext(t=0.0, fps=10)
    frame0 = blend(base, layer, "dancing_dissolve", context=context)
    same = blend(base, layer, "dancing_dissolve", context=context.with_time(0.05))
    frame1 = blend(base, layer, "dancing_dissolve", context=context.with_time(0.1))
    np.testing.assert_array_equal(frame0.rgba, same.rgba)
    assert not np.array_equal(frame0.rgba, frame1.rgba)


def test_dissolve_with_preserved_transparency_keeps_backdrop_alpha():
    base = gray(0.2, 0.5, size=(10, 10))
    layer = gray(1.0, size=(10, 10))
    result = blend(base, layer, "dissolve", preserve_underlying_transparency=True)
    np.testing.assert_allclose(result.rgba[..., 3], 0.5)


# --------------------------------------------------------------------------- #
# validation


def test_blend_validates_inputs():
    with pytest.raises(TypeError, match="Buffer"):
        blend(np.zeros((1, 1, 4)), gray(0.5))
    linear = Buffer(np.ones((1, 1, 4), np.float32), color_space="linear")
    with pytest.raises(ValueError, match="color_space"):
        blend(gray(0.5), linear, "screen")
    with pytest.raises(TypeError, match="context"):
        blend(gray(0.5), gray(0.5, 0.5), "dissolve", context="ctx")
    huge = pixel((3e38, 3e38, 3e38))
    with pytest.raises(ValueError, match="finite"):
        blend(huge, huge, "add")


def test_blend_inputs_are_not_mutated():
    base, layer = gray(0.4, size=(2, 2)), gray(0.6, 0.5, size=(2, 2))
    before = base.rgba.copy(), layer.rgba.copy()
    for mode in BlendMode:
        blend(base, layer, mode)
    np.testing.assert_array_equal(base.rgba, before[0])
    np.testing.assert_array_equal(layer.rgba, before[1])
