"""Regression contracts for mutable composition alpha and extended RGB."""

import numpy as np

import pytest

from moviepy import vfx
from moviepy.ae import Buffer, Composition, RenderContext
from moviepy.ae.effects import from_moviepy_effect
from moviepy.ae.effects.time.echo import Echo
from moviepy.Effect import Effect


class FloatIdentity(Effect):
    """A bridge fixture that does not itself quantize or clip its input."""

    def apply(self, clip):
        """Leave pixels and coverage unchanged."""
        return clip


@pytest.mark.parametrize("space", ["srgb", "linear"])
@pytest.mark.parametrize("mutation", ["opacity", "enabled"])
def test_alpha_reflects_layer_mutation_between_rgb_and_mask(space, mutation):
    comp = Composition(
        size=(8, 8),
        duration=1,
        transparent=True,
        context=RenderContext(working_space=space),
    )
    layer = comp.add_solid(color=(255, 255, 255))
    comp.get_frame(0)
    if mutation == "opacity":
        layer.transform.opacity = 0
    else:
        layer.enabled = False
    expected = comp.render_buffer(0).rgba[..., 3]
    assert not np.any(expected)
    np.testing.assert_array_equal(comp.mask.get_frame(0), expected)


@pytest.mark.parametrize("space", ["srgb", "linear"])
@pytest.mark.parametrize("red", [1.57, -0.25])
@pytest.mark.parametrize("alpha", [1.0, 0.5])
@pytest.mark.parametrize("kind", ["echo", "bridge"])
def test_identity_effect_retains_signed_hdr_pixels(space, red, alpha, kind):
    pixels = np.array([[[red * alpha, 0.4 * alpha, 0.2 * alpha, alpha]]], np.float32)
    source = Buffer(pixels, offset=(-2, 5), color_space=space)
    effect = (
        Echo(number_of_echoes=0)
        if kind == "echo"
        else from_moviepy_effect(FloatIdentity())
    )
    actual = effect.process(source, 0, RenderContext(working_space=space))
    np.testing.assert_allclose(actual.rgba, pixels, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(source.rgba, pixels)
    assert actual.offset == source.offset and actual.color_space == space
    assert not actual.rgba.flags.writeable


@pytest.mark.parametrize("kind", ["echo", "bridge"])
def test_normal_srgb_effects_keep_legacy_clipping_with_nonuniform_alpha(kind):
    pixels = np.array([[[0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1]]], np.float32)
    source = Buffer(pixels)
    if kind == "echo":
        actual = Echo(number_of_echoes=1).process(
            source, 0, source_at=lambda dt: source
        )
        expected = np.ones_like(pixels)
    else:
        actual = from_moviepy_effect(vfx.MultiplyColor(2)).process(source, 0)
        expected = pixels
    np.testing.assert_array_equal(actual.rgba, expected)


def test_linear_echo_can_create_hdr_from_unit_inputs():
    source = Buffer(np.array([[[0.8, 0.8, 0.8, 1]]], np.float32), color_space="linear")
    result = Echo(number_of_echoes=1).process(source, 0, source_at=lambda dt: source)
    np.testing.assert_allclose(result.rgba, [[[1.6, 1.6, 1.6, 1]]], atol=1e-6)


def test_extended_echo_rejects_float32_overflow():
    pixels = np.ones((1, 1, 4), np.float32)
    pixels[..., :3] = np.finfo(np.float32).max
    source = Buffer(pixels, color_space="linear")
    with pytest.raises(ValueError, match="finite"):
        Echo(number_of_echoes=1).process(source, 0, source_at=lambda dt: source)
