"""Tests for Directional Blur, Radial Blur, Sharpen and Unsharp Mask."""

import numpy as np
import pytest

from moviepy.ae import Buffer, Property
from moviepy.ae.effects import registry
from moviepy.ae.effects.blur.directional_blur import DirectionalBlur
from moviepy.ae.effects.blur.radial_blur import RadialBlur
from moviepy.ae.effects.blur.sharpen import Sharpen, UnsharpMask


def uniform(color, alpha=1.0, size=(4, 3), offset=(0, 0), color_space="srgb"):
    """Premultiplied buffer of straight ``color`` (0..1)."""
    width, height = size
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(color, dtype=np.float32) * alpha
    rgba[..., 3] = alpha
    return Buffer(rgba, offset, color_space)


def impulse(size=(5, 5), at=(2, 2), color=(1.0, 1.0, 1.0)):
    """Opaque single pixel (x, y) on a transparent layer."""
    width, height = size
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[at[1], at[0]] = (*color, 1.0)
    return Buffer(rgba)


def step_edge(low=0.4, high=0.6):
    """Opaque 1x6 row: three ``low`` pixels then three ``high`` pixels."""
    rgba = np.ones((1, 6, 4), dtype=np.float32)
    rgba[:, :3, :3] = low
    rgba[:, 3:, :3] = high
    return Buffer(rgba)


def assert_valid_premultiplied(buffer):
    rgba = buffer.rgba
    assert np.all(rgba >= 0.0)
    assert np.all(rgba[..., :3] <= rgba[..., 3:] + 1e-6)
    assert np.all(rgba <= 1.0 + 1e-6)


# -- Directional Blur ------------------------------------------------------- #


def test_directional_blur_is_registered_under_ae_name():
    assert registry.get("Directional Blur") is DirectionalBlur
    assert DirectionalBlur.category == "Blur & Sharpen"


def test_directional_blur_zero_length_returns_source():
    src = impulse()
    assert DirectionalBlur().process(src, 0.0) is src
    assert DirectionalBlur(blur_length=0, direction=45).process(src, 0.0) is src


def test_directional_blur_horizontal_box_average():
    # Length 2 -> margin 2 px, samples at offsets -1, 0, +1 px. The impulse
    # sits at padded (x=4, y=4); each of its three neighbours gets 1/3.
    out = DirectionalBlur(direction=0, blur_length=2).process(impulse(), 0.0)
    assert out.size == (9, 9)
    np.testing.assert_allclose(
        out.rgba[4, :, 3], [0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0], atol=1e-6
    )
    assert np.all(out.rgba[0, :, 3] == 0)


def test_directional_blur_follows_direction_angle():
    out = DirectionalBlur(direction=90, blur_length=2).process(impulse(), 0.0)
    np.testing.assert_allclose(
        out.rgba[:, 4, 3], [0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0], atol=1e-6
    )
    # A vertical streak leaves the neighbouring column empty.
    assert np.all(out.rgba[:, 3, 3] == 0)


def test_directional_blur_grows_bounds_by_half_length():
    effect = DirectionalBlur(blur_length=10)
    assert effect.bounds_expand((5, 5), 0.0) == (6, 6, 6, 6)
    out = effect.process(impulse(), 0.0)
    assert out.size == (17, 17)
    assert out.offset == (-6, -6)


def test_directional_blur_keeps_premultiplied_alpha_valid():
    rgba = np.zeros((3, 5, 4), dtype=np.float32)
    rgba[1, 2] = (1.0, 0.0, 0.0, 1.0)  # opaque red
    rgba[1, 3] = (0.2, 0.0, 0.0, 0.2)  # premultiplied half-transparent red
    out = DirectionalBlur(direction=0, blur_length=4).process(Buffer(rgba), 0.0)
    assert_valid_premultiplied(out)


def test_directional_blur_keyframed_length_animates():
    length = Property(0.0, keyframes=[(0, 0.0), (1, 4.0)])
    effect = DirectionalBlur(direction=0, blur_length=length)
    src = impulse()
    # Length 0 at t=0 is the identity, so the source comes back untouched.
    assert effect.process(src, 0.0) is src
    # At t=1 the impulse spreads into a 5-sample average (1/5 per pixel).
    # Length 4 pads by 3 px, so the impulse sits at padded (x=5, y=5).
    spread = effect.process(src, 1.0).rgba[5, :, 3]
    assert spread.max() == pytest.approx(0.2, abs=1e-6)
    assert np.count_nonzero(spread) == 5


def test_directional_blur_preserves_linear_color_space_and_empty_buffer():
    linear = uniform((0.5, 0.5, 0.5), color_space="linear")
    out = DirectionalBlur(blur_length=3).process(linear, 0.0)
    assert out.color_space == "linear"
    empty = Buffer(np.zeros((0, 4, 4), dtype=np.float32))
    assert 0 in DirectionalBlur(blur_length=5).process(empty, 0.0).size


# -- Radial Blur ------------------------------------------------------------ #


def test_radial_blur_is_registered_and_identity_at_zero():
    assert registry.get("Radial Blur") is RadialBlur
    src = impulse()
    assert RadialBlur().process(src, 0.0) is src
    assert RadialBlur(amount=0, type="zoom").process(src, 0.0) is src


@pytest.mark.parametrize("kind", ["spin", "zoom"])
def test_radial_blur_center_pixel_is_fixed_point(kind):
    # Every sample maps the exact center onto itself, so it keeps its value.
    out = RadialBlur(amount=60, type=kind).process(impulse(), 0.0)
    assert out.rgba[2, 2, 3] == pytest.approx(1.0, abs=1e-6)
    assert out.rgba[2, 2, 0] == pytest.approx(1.0, abs=1e-6)


def test_radial_blur_spin_smears_content_around_center():
    out = RadialBlur(amount=60, type="spin").process(impulse(at=(4, 2)), 0.0)
    # The impulse two pixels right of center is spread along its arc.
    assert out.rgba[1, 4, 3] > 0.0 and out.rgba[3, 4, 3] > 0.0
    assert out.rgba[2, 4, 3] < 1.0
    assert_valid_premultiplied(out)


def test_radial_blur_zoom_smears_content_along_radius():
    out = RadialBlur(amount=60, type="zoom").process(impulse(at=(4, 2)), 0.0)
    # Zoom samples scale the ray through the center by 0.7..1.3. Column 4
    # (the impulse) and column 3 (bilinear spill from it) gain energy, while
    # column 1 lies outside the reachable range and stays empty.
    assert out.rgba[2, 4, 3] > 0.0
    assert out.rgba[2, 3, 3] > 0.0
    assert np.all(out.rgba[:, 1, 3] == 0.0)
    assert_valid_premultiplied(out)


def test_radial_blur_center_is_normalized_and_moves_the_fixed_point():
    src = impulse(at=(4, 4))
    default = RadialBlur(amount=40, type="spin").process(src, 0.0)
    corner = RadialBlur(amount=40, type="spin", center=(1.0, 1.0)).process(src, 0.0)
    # Around the default center (2, 2) the impulse at (4, 4) is smeared...
    assert default.rgba[4, 4, 3] < 1.0 - 1e-3
    # ...while centering on the impulse itself makes it a fixed point.
    assert corner.rgba[4, 4, 3] == pytest.approx(1.0, abs=1e-6)


def test_radial_blur_linear_color_space_and_empty_buffer():
    linear = uniform((0.3, 0.3, 0.3), color_space="linear")
    assert RadialBlur(amount=30).process(linear, 0.0).color_space == "linear"
    empty = Buffer(np.zeros((0, 4, 4), dtype=np.float32))
    assert 0 in RadialBlur(amount=30).process(empty, 0.0).size


# -- Sharpen and Unsharp Mask ----------------------------------------------- #


def test_sharpen_and_unsharp_are_registered():
    assert registry.get("Sharpen") is Sharpen
    assert registry.get("Unsharp Mask") is UnsharpMask


def test_sharpen_default_is_identity():
    src = uniform((0.2, 0.4, 0.6), alpha=0.5)
    assert Sharpen().process(src, 0.0) is src


def test_sharpen_uniform_color_is_unchanged():
    src = uniform((0.4, 0.4, 0.4))
    out = Sharpen(sharpen_amount=300).process(src, 0.0)
    np.testing.assert_allclose(out.rgba, src.rgba, atol=1e-5)


def test_sharpen_steepens_edge_and_keeps_alpha():
    src = step_edge()
    out = Sharpen(sharpen_amount=100).process(src, 0.0)
    assert out.rgba[0, 2, 0] < 0.4  # dark side pushed darker
    assert out.rgba[0, 3, 0] > 0.6  # bright side pushed brighter
    np.testing.assert_array_equal(out.rgba[..., 3], src.rgba[..., 3])
    assert_valid_premultiplied(out)


def test_sharpen_keeps_partial_alpha_and_clips_color():
    src = uniform((0.9, 0.9, 0.9), alpha=0.5)
    out = Sharpen(sharpen_amount=500).process(src, 0.0)
    np.testing.assert_array_equal(out.rgba[..., 3], src.rgba[..., 3])
    assert_valid_premultiplied(out)


def test_unsharp_mask_default_and_zero_radius_are_identity():
    src = uniform((0.2, 0.4, 0.6))
    assert UnsharpMask().process(src, 0.0) is src
    assert UnsharpMask(amount=200, radius=0).process(src, 0.0) is src


def test_unsharp_mask_threshold_suppresses_small_detail():
    rng = np.random.default_rng(0)
    rgba = np.ones((6, 6, 4), dtype=np.float32)
    rgba[..., :3] = 0.5 + rng.uniform(-0.002, 0.002, (6, 6, 3)).astype(np.float32)
    src = Buffer(rgba)
    # Threshold 255 demands |detail| >= 1.0, which this texture never reaches.
    out = UnsharpMask(amount=300, radius=1, threshold=255).process(src, 0.0)
    np.testing.assert_allclose(out.rgba, src.rgba, atol=1e-6)


def test_unsharp_mask_amplifies_detail_above_threshold():
    src = step_edge()
    out = UnsharpMask(amount=100, radius=1, threshold=0).process(src, 0.0)
    assert out.rgba[0, 2, 0] < 0.4 and out.rgba[0, 3, 0] > 0.6
    np.testing.assert_array_equal(out.rgba[..., 3], src.rgba[..., 3])
    assert_valid_premultiplied(out)


def test_unsharp_mask_keyframed_amount_animates():
    src = step_edge()
    amount = Property(0.0, keyframes=[(0, 0.0), (1, 200.0)])
    effect = UnsharpMask(amount=amount, radius=1)
    assert effect.process(src, 0.0) is src
    assert effect.process(src, 1.0).rgba[0, 2, 0] < 0.4


def test_sharpen_effects_preserve_linear_color_space_and_empty_buffers():
    linear = uniform((0.3, 0.3, 0.3), color_space="linear")
    assert UnsharpMask(amount=100).process(linear, 0.0).color_space == "linear"
    assert Sharpen(sharpen_amount=100).process(linear, 0.0).color_space == "linear"
    empty = Buffer(np.zeros((0, 4, 4), dtype=np.float32))
    assert 0 in UnsharpMask(amount=100).process(empty, 0.0).size
    assert 0 in Sharpen(sharpen_amount=100).process(empty, 0.0).size
