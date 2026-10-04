"""Exercise public AE buffers against MoviePy and independent image references."""

import subprocess
import sys

import numpy as np

import pytest

from moviepy.ae import Buffer, RenderContext, premultiply, unpremultiply
from moviepy.video.VideoClip import ImageClip
from tests.ae.assets.generate_ws00 import RECIPE_IDS, recipe_inputs


@pytest.mark.parametrize("foreground_alpha", [None, 0.25, 0.5])
@pytest.mark.parametrize("background_alpha", [None, 0.4])
@pytest.mark.parametrize("position", [(0, 0), (-1, 1), (2, -1)])
def test_single_over_matches_moviepy(foreground_alpha, background_alpha, position):
    foreground = (np.arange(18).reshape(2, 3, 3) * 13).astype(np.uint8)
    background = (np.arange(36).reshape(3, 4, 3) * 7).astype(np.uint8)
    source = ImageClip(foreground).with_start(2).with_position(position)
    mask = None
    if foreground_alpha is not None:
        mask = np.full((2, 3), foreground_alpha)
        source = source.with_mask(ImageClip(mask, is_mask=True))
    background_mask = None
    if background_alpha is not None:
        background_mask = np.full((3, 4), background_alpha)
    legacy_rgb, legacy_alpha = source.compose_on(background, 2.25, background_mask)
    base = Buffer.from_uint8_rgb(background, background_mask)
    layer = Buffer.from_clip(source, 0.25, offset=position)
    rendered = layer.composite_over(base).crop(base.bounds)
    delta = rendered.to_uint8_rgb().astype(np.int16) - legacy_rgb.astype(np.int16)
    assert np.max(np.abs(delta)) <= 1
    expected_alpha = 1 if legacy_alpha is None else legacy_alpha
    np.testing.assert_allclose(rendered.rgba[..., 3], expected_alpha, atol=1e-7)


@pytest.mark.parametrize("recipe_id", RECIPE_IDS)
def test_public_buffer_golden(recipe_id, assert_image_close, ae_golden_dir):
    inputs = recipe_inputs(recipe_id)
    result = Buffer.from_uint8_rgb(
        inputs["foreground"], inputs["mask"], offset=inputs["offset"]
    )
    if inputs["background"] is not None:
        base = Buffer.from_uint8_rgb(inputs["background"])
        result = result.composite_over(base).crop(base.bounds)
    tol = 0 if recipe_id == "uint8_rgb_roundtrip" else 1 / 255
    assert_image_close(result.to_uint8_rgb(), ae_golden_dir / (recipe_id + ".png"), tol)


def test_public_api_example():
    frame = np.array([[[255, 0, 0]]], dtype=np.uint8)
    source = Buffer.from_uint8_rgb(frame, mask=np.array([[0.5]]))
    np.testing.assert_array_equal(source.to_uint8_rgb(bg=(0, 0, 1)), [[[128, 0, 128]]])
    np.testing.assert_array_equal(premultiply(unpremultiply(source.rgba)), source.rgba)
    context = RenderContext(t=0.5, fps=24, rng_seed=7)
    assert context.frame_index == 12
    np.testing.assert_array_equal(
        context.rng_for("foreground").random(5),
        context.rng_for("foreground").random(5),
    )


def test_ae_import_is_deferred_from_moviepy():
    code = (
        "import sys; import moviepy; "
        "assert 'moviepy.ae' not in sys.modules; "
        "import moviepy.ae as ae; "
        "assert {'Buffer', 'RenderContext', 'premultiply', 'unpremultiply'} "
        "<= set(ae.__all__); "
        "assert 'moviepy.ae.composition' not in sys.modules; "
        "assert 'moviepy.ae.blend.modes' not in sys.modules; "
        "assert ae.Composition.__module__ == 'moviepy.ae.composition'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
