"""Validation boundaries plus opt-in tests of real installed/model backends."""

import hashlib
import os

import numpy as np
from PIL import Image, ImageDraw

import pytest

from moviepy import ImageClip
from moviepy.ae import Buffer, Composition, Transform
from moviepy.ae.ai import U2NetBackgroundRemoval
from moviepy.ae.three_d import find_blender, render_scene
from moviepy.ae.three_d.scene import validate_scene


def test_scene_validation_is_detached_and_rejects_missing_capabilities(tmp_path):
    scene = {"objects": [{"type": "cube"}]}
    normalized = validate_scene(scene)
    assert scene == {"objects": [{"type": "cube"}]}
    assert normalized["objects"][0]["scale"] == [1, 1, 1]
    for invalid in (
        {"camera": {"location": [0, 0, 0], "target": [0, 0, 0]}},
        {"objects": [{"type": "mesh"}]},
        {"samples": 2.5},
        {"typo": True},
        {"objects": {}},
        {"lights": {}},
        {"materials": {2: {}}},
        {"objects": [{"type": "cube", "name": 23}]},
        {"objects": [{"type": "cube", "animation": [{"location": [1, 0, 0]}]}]},
        {"objects": [{"type": "cube", "scale": [0, 1, 1]}]},
    ):
        with pytest.raises((ValueError, TypeError)):
            validate_scene(invalid)
    with pytest.raises(ValueError, match="SBSAR"):
        validate_scene(
            {
                "materials": {
                    "paint": {"maps": {"base_color": str(tmp_path / "paint.sbsar")}}
                }
            }
        )


def test_scene_pbr_maps_and_animation(tmp_path):
    texture = tmp_path / "texture.png"
    Image.new("RGB", (4, 4), "white").save(texture)
    scene = validate_scene(
        {
            "materials": {
                "paint": {
                    "maps": {"roughness": str(texture)},
                    "normal_convention": "directx",
                }
            },
            "objects": [
                {
                    "type": "sphere",
                    "material": "paint",
                    "animation": [
                        {"time": 0, "rotation": [0, 0, 0]},
                        {"time": 1, "rotation": [0, 0, 90]},
                    ],
                }
            ],
        }
    )
    assert scene["materials"]["paint"]["maps"]["roughness"] == str(texture.resolve())
    with pytest.raises(ValueError, match="duplicate"):
        validate_scene(
            {"objects": [{"type": "cube", "animation": [{"time": 0}, {"time": 0}]}]}
        )
    with pytest.raises(FileNotFoundError):
        find_blender(tmp_path / "missing.exe")


def test_model_path_and_hash_fail_before_inference(tmp_path):
    with pytest.raises(FileNotFoundError):
        U2NetBackgroundRemoval(tmp_path / "absent.onnx")
    checkpoint = tmp_path / "not-a-model.onnx"
    checkpoint.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="SHA-256"):
        U2NetBackgroundRemoval(checkpoint, expected_sha256="0" * 64)


@pytest.mark.skipif(
    not os.environ.get("MOVIEPY_AE_TEST_BLENDER"),
    reason="opt-in installed Blender integration",
)
def test_actual_cycles_render_has_alpha_and_loads_as_clip(tmp_path):
    scene = {
        "size": [80, 60],
        "fps": 10,
        "duration": 0.2,
        "samples": 4,
        "camera": {"location": [0, -6, 2], "target": [0, 0, 0]},
        "materials": {"paint": {"base_color": [0.8, 0.08, 0.02, 1]}},
        "objects": [
            {
                "type": "cube",
                "material": "paint",
                "animation": [
                    {"time": 0, "rotation": [0, 0, 0]},
                    {"time": 0.2, "rotation": [0, 0, 35]},
                ],
            }
        ],
    }
    rendered = render_scene(scene, tmp_path / "render", timeout=120)
    assert len(rendered.frames) == 2
    rgba = np.array(Image.open(rendered.frames[0]))
    assert rgba.shape == (60, 80, 4)
    assert rgba[..., 3].max() == 255 and rgba[0, 0, 3] == 0
    assert rgba[30, 40, 0] > rgba[30, 40, 2]
    assert not np.array_equal(rgba, np.array(Image.open(rendered.frames[1])))
    with rendered.to_clip() as clip:
        assert clip.mask is not None and clip.duration == 0.2
        assert clip.get_frame(0.1).shape == (60, 80, 3)
    with pytest.raises(FileExistsError):
        render_scene(scene, tmp_path / "render")


@pytest.mark.skipif(
    not os.environ.get("MOVIEPY_AE_TEST_U2NET"),
    reason="opt-in approved local U2Net checkpoint",
)
def test_actual_u2net_checkpoint_inference_is_finite_and_repeatable():
    path = os.environ["MOVIEPY_AE_TEST_U2NET"]
    model = U2NetBackgroundRemoval(path, refine_radius=2)
    image = Image.new("RGB", (160, 128), (205, 220, 235))
    draw = ImageDraw.Draw(image)
    draw.ellipse((45, 15, 115, 85), fill=(210, 120, 30), outline=(100, 45, 15), width=3)
    draw.rounded_rectangle((65, 70, 95, 115), radius=5, fill=(80, 100, 60))
    frame = np.array(image)
    first = model.predict_mask(frame)
    assert first.shape == frame.shape[:2] and np.isfinite(first).all()
    assert 0 <= first.min() < first.max() <= 1
    assert first[35:65, 65:95].mean() > 0.95
    assert first[:15, :15].mean() < 0.05
    np.testing.assert_allclose(model.predict_mask(frame), first, atol=1e-6)
    source = Buffer.from_uint8_rgb(frame)
    result = model.process(source, 0)
    np.testing.assert_allclose(result.rgba[..., 3], first, atol=1e-6)
    with ImageClip(frame, duration=0.25).with_effects([model]) as cutout:
        assert cutout.mask is not None
        rgb = cutout.get_frame(0)
        np.testing.assert_array_equal(rgb[first > 0.99], frame[first > 0.99])
        np.testing.assert_allclose(cutout.mask.get_frame(0), first, atol=1e-6)
    assert len(model.model_sha256) == len(hashlib.sha256(b"").hexdigest())
    comp = Composition(size=(160, 128), duration=1)
    subject = comp.add_clip(ImageClip(frame))
    subject.transform = Transform(position=(80, 64))
    comp.add_adjustment().effects = [model]
    bounds = (30, 15, 125, 110)
    np.testing.assert_allclose(
        comp.render_buffer(0, bounds=bounds).rgba,
        comp.render_buffer(0).crop(bounds).rgba,
        atol=1e-6,
    )
