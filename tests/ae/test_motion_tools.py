"""Analytical and integration regression cases for the motion-tool increment."""

from pathlib import Path

import cv2
import numpy as np

import pytest

import moviepy.ae as ae
from moviepy.ae.ai import inpaint_classical, refine_matte
from moviepy.ae.effects.blur.gaussian_blur import filter_pixels
from moviepy.ae.text.clusters import clusters
from moviepy.ae.text.layer import PRESETS


FONT = Path(__file__).resolve().parents[2] / "media/doc_medias/example.ttf"


@pytest.mark.parametrize("sigmas", [(0, 0), (1.7, 0), (0, 2.3), (4, 6)])
@pytest.mark.parametrize("border", [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE])
def test_combined_gaussian_matches_independent_two_pass(sigmas, border):
    data = np.random.default_rng(20).random((23, 29, 4), dtype=np.float32)
    original = data.copy()
    expected = data.copy()
    for axis, sigma in enumerate(sigmas):
        if sigma:
            kernel = cv2.getGaussianKernel(
                2 * int(np.ceil(3 * sigma)) + 1, sigma, cv2.CV_32F
            )
            unit = np.ones((1, 1), np.float32)
            expected = cv2.sepFilter2D(
                expected,
                -1,
                kernel if axis == 0 else unit,
                kernel if axis == 1 else unit,
                borderType=border,
            )
    np.testing.assert_allclose(
        filter_pixels(data, *sigmas, border), expected, atol=2e-6
    )
    np.testing.assert_array_equal(data, original)


def test_gaussian_preserves_signed_hdr_at_constant_edges():
    source = ae.Buffer(
        np.tile(np.array([2.0, -0.25, 0.5, 0.5], np.float32), (12, 16, 1))
    )
    blurred = ae.fx.GaussianBlur(blurriness=8, repeat_edge_pixels=True).process(
        source, 0
    )
    np.testing.assert_allclose(blurred.rgba, source.rgba, atol=2e-6)


def test_drop_shadow_has_analytic_offset_alpha_and_source_over():
    source = ae.Buffer(np.array([[[0.5, 0, 0, 0.5]]], np.float32))
    result = ae.fx.DropShadow(direction=90, distance=3, softness=0, opacity=40).process(
        source, 0
    )
    assert result.bounds == (0, 0, 4, 1)
    np.testing.assert_allclose(result.rgba[0, 0], source.rgba[0, 0])
    np.testing.assert_allclose(result.rgba[0, 3], [0, 0, 0, 0.2])
    overlap = ae.fx.DropShadow(
        distance=0, softness=0, opacity=100, color=(0, 0, 255)
    ).process(source, 0)
    np.testing.assert_allclose(overlap.rgba[0, 0], [0.5, 0, 0.25, 0.75])


@pytest.mark.parametrize(
    "effect", [ae.fx.DropShadow(opacity=0), ae.fx.Glow(intensity=0)]
)
def test_disabled_light_strength_is_identity(effect):
    source = ae.Buffer(np.ones((3, 4, 4), np.float32))
    result = effect.process(source, 0)
    assert result.bounds == source.bounds
    np.testing.assert_array_equal(result.rgba, source.rgba)


@pytest.mark.parametrize("scale", [1, 0.5])
@pytest.mark.parametrize(
    "effect",
    [
        ae.fx.DropShadow(direction=315, distance=7.5, softness=8),
        ae.fx.Glow(radius=8, threshold=0),
    ],
)
def test_new_adjustment_effects_roi_matches_full(effect, scale):
    comp = ae.Composition(size=(96, 72), duration=1)
    source = comp.add_solid(color=(240, 230, 210), size=(25, 21))
    source.transform = ae.Transform(position=(46, 33))
    comp.add_adjustment(effects=[effect])
    context = ae.RenderContext(resolution_scale=scale)
    roi = tuple(round(v * scale) for v in (30, 20, 70, 54))
    full = comp.render_buffer(0, context).crop(roi)
    actual = comp.render_buffer(0, context, bounds=roi)
    np.testing.assert_allclose(actual.rgba, full.rgba, atol=1e-5)


def test_shadow_only_and_glow_preserve_valid_coverage():
    source = ae.Buffer(np.ones((1, 1, 4), np.float32))
    shadow = ae.fx.DropShadow(
        direction=90, distance=4, softness=0, shadow_only=True
    ).process(source, 0)
    assert shadow.rgba[0, 0, 3] == 0 and shadow.rgba[0, 4, 3] == 0.5
    glow = ae.fx.Glow(radius=6, threshold=0).process(source, 0)
    assert glow.size == (19, 19)
    assert np.all(glow.rgba[..., :3] <= glow.rgba[..., 3:] + 1e-6)
    assert np.count_nonzero(glow.rgba[..., 3]) > 1


def test_glow_spreads_highlights_across_an_opaque_image():
    pixels = np.zeros((9, 9, 4), np.float32)
    pixels[..., 3] = 1
    pixels[4, 4, :3] = 1
    source = ae.Buffer(pixels)
    result = (
        ae.fx.Glow(radius=2, threshold=50, color=(255, 255, 255))
        .process(source, 0)
        .crop(source.bounds)
    )
    assert result.rgba[4, 5, 0] > 0.01
    np.testing.assert_array_equal(result.rgba[4, 4, :3], 1)
    np.testing.assert_array_equal(result.rgba[..., 3], 1)


def test_particles_seek_determinism_and_analytic_gravity():
    layer = ae.ParticleLayer(
        size=(64, 48),
        count=10,
        seed=7,
        emitter=(20, 10),
        speed=(8, 8),
        direction=0,
        spread=0,
        gravity=(0, 6),
        lifetime=2,
    )
    points, ages = layer.state_at(0.5)
    np.testing.assert_allclose(points, np.tile([24, 10.75], (10, 1)))
    np.testing.assert_allclose(ages, 0.25)
    first = layer.source_buffer(0.5).rgba.copy()
    layer.source_buffer(1.2)
    np.testing.assert_array_equal(layer.source_buffer(0.5).rgba, first)
    assert not layer.source_buffer(-1).rgba.any()
    assert not layer.source_buffer(2).rgba.any()


def test_particles_seed_and_roi():
    comp = ae.Composition(size=(80, 60), duration=2)
    particles = comp.add_particles(count=70, seed=25, emitter=(40, 30), speed=(4, 24))
    particles.transform = ae.Transform(position=(40, 30))
    roi = (20, 15, 60, 45)
    np.testing.assert_allclose(
        comp.render_buffer(0.4, bounds=roi).rgba,
        comp.render_buffer(0.4).crop(roi).rgba,
        atol=1e-5,
    )
    other = ae.ParticleLayer(
        size=(80, 60), count=70, seed=26, emitter=(40, 30), speed=(4, 24)
    )
    assert not np.array_equal(
        particles.source_buffer(0.4).rgba, other.source_buffer(0.4).rgba
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": -1},
        {"count": True},
        {"count": 50001},
        {"radius": 0},
        {"lifetime": 0},
        {"speed": (5, 2)},
        {"emitter": (np.nan, 0)},
    ],
)
def test_particles_reject_invalid_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ae.ParticleLayer(size=(32, 24), **kwargs)


def test_typewriter_cluster_count_and_fixed_canvas():
    text = "Ae\u0301B"
    layer = ae.TextLayer(text, font=FONT, preset="typewriter", rate=2, delay=0.25)
    assert clusters("e\u0301👩‍💻🇹🇼") == ("e\u0301", "👩‍💻", "🇹🇼")
    assert layer.visible_text(0.24) == ""
    assert layer.visible_text(0.75) == "A"
    assert layer.visible_text(1.25) == "Ae\u0301"
    assert layer.visible_text(1.75) == text
    assert (
        layer.source_buffer(0).size == layer.source_buffer(2).size == layer.source_size
    )


@pytest.mark.parametrize("preset", PRESETS)
def test_text_preset_terminal_frame_and_random_seek(preset):
    layer = ae.TextLayer("AV TITLE", font=FONT, preset=preset, font_size=24)
    expected = ae.TextLayer("AV TITLE", font=FONT, preset="none", font_size=24)
    np.testing.assert_array_equal(
        layer.source_buffer(10).rgba, expected.source_buffer(10).rgba
    )
    middle = layer.source_buffer(0.3).rgba.copy()
    layer.source_buffer(0)
    np.testing.assert_array_equal(layer.source_buffer(0.3).rgba, middle)
    if preset != "none":
        assert not layer.source_buffer(0).rgba.any()


def test_text_validation_and_empty_title():
    with pytest.raises(ValueError, match="font"):
        ae.TextLayer("繁體中文")
    with pytest.raises(ValueError):
        ae.TextLayer("title", preset="not-a-preset")
    assert not ae.TextLayer("", font=FONT).source_buffer(1).rgba.any()


def test_guided_refinement_preserves_interiors_and_edges():
    image = np.zeros((32, 40, 3), np.uint8)
    image[:, 20:] = 255
    mask = np.zeros((32, 40), np.float32)
    mask[:, 20:] = 1
    result = refine_matte(image, mask, radius=3)
    np.testing.assert_array_equal(result[:, :10], 0)
    np.testing.assert_array_equal(result[:, 30:], 1)
    assert result[:, 20].mean() > 0.9 and result[:, 19].mean() < 0.1
    np.testing.assert_array_equal(refine_matte(image, mask, radius=0), mask)


@pytest.mark.parametrize("method", ["telea", "navier_stokes"])
def test_classical_inpainting_preserves_unmasked_pixels(method):
    image = np.full((24, 32, 3), (40, 90, 140), np.uint8)
    image[9:15, 12:20] = (255, 0, 0)
    mask = np.zeros((24, 32), np.float32)
    mask[8:16, 11:21] = 1
    result = inpaint_classical(image, mask, method=method)
    np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
    assert np.max(np.abs(result[10:14, 13:19].astype(int) - (40, 90, 140))) <= 3
    with pytest.raises(ValueError):
        inpaint_classical(image, np.ones_like(mask))
