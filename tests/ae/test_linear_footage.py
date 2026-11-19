"""Exact color arithmetic and genuine scene-linear source/seek contracts."""

import hashlib
import json
import os

import numpy as np

import pytest

from moviepy.ae import Buffer, Composition, RenderContext, Transform
from moviepy.ae.buffer import unpremultiply
from moviepy.ae.color import convert_buffer, linear_to_srgb, srgb_to_linear
from moviepy.ae.effects.generate.fill import Fill
from moviepy.ae.three_d import render_godot_scene
from moviepy.ae.three_d._godot_linear import (
    LinearSequenceLayer,
    decode_linear_exr,
    resolve_linear_frames,
)


def transfer_reference(values, decode):
    values = np.asarray(values)
    magnitude = np.abs(values.astype(np.float64))
    result = (
        np.where(
            magnitude <= 0.04045,
            magnitude / 12.92,
            ((magnitude + 0.055) / 1.055) ** 2.4,
        )
        if decode
        else np.where(
            magnitude <= 0.0031308,
            magnitude * 12.92,
            1.055 * magnitude ** (1 / 2.4) - 0.055,
        )
    )
    return np.copysign(result, values).astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("decode", [True, False])
def test_transfer_optimization_preserves_float32_results(dtype, decode):
    rng = np.random.default_rng(47)
    values = np.concatenate(
        (
            rng.uniform(-8, 8, 10001),
            np.logspace(-42, 14, 101),
            [0, -0.0, 0.0031308, 0.04045, 1],
        )
    ).astype(dtype)
    function = srgb_to_linear if decode else linear_to_srgb
    # Byte comparison includes signed zero, unlike numeric array equality.
    assert function(values).tobytes() == transfer_reference(values, decode).tobytes()


@pytest.mark.parametrize("alpha", [0, 0.00001, 0.5, 1, None])
@pytest.mark.parametrize("target", ["srgb", "linear"])
def test_sparse_conversion_preserves_alpha_signed_hdr_and_hidden_rgb(alpha, target):
    rng = np.random.default_rng(23)
    pixels = rng.uniform(-4, 4, (17, 29, 4)).astype(np.float32)
    pixels[..., 3] = (
        rng.choice([0, 1e-8, 0.25, 0.5, 1], size=pixels.shape[:2])
        if alpha is None
        else alpha
    )
    pixels[..., :3] *= pixels[..., 3:4]
    source = Buffer(
        pixels, offset=(-3, 7), color_space="linear" if target == "srgb" else "srgb"
    )
    expected = unpremultiply(source.rgba)
    expected[..., :3] = (
        transfer_reference(expected[..., :3], target == "linear") * expected[..., 3:4]
    )
    actual = convert_buffer(source, target)
    assert actual.rgba.tobytes() == expected.tobytes()
    assert actual.offset == source.offset and not actual.rgba.flags.writeable


def test_sparse_conversion_rejects_overflowing_unpremultiply():
    source = Buffer(np.array([[[1.0, 1.0, 1.0, 1e-40]]], np.float32))
    with pytest.raises(ValueError):
        convert_buffer(source, "linear")


def test_partial_effect_mix_keeps_linear_encoding_and_hdr_values():
    source = Buffer(
        np.array([[[4, 0.25, 0.125, 0.5]]], np.float32), color_space="linear"
    )
    context = RenderContext(working_space="linear")
    full = Fill(color=(128, 64, 32)).process(source, 0, context)
    mixed = Fill(color=(128, 64, 32), blend_with_original=50).process(
        source, 0, context
    )
    assert mixed.color_space == "linear"
    np.testing.assert_array_equal(
        mixed.rgba, source.rgba + (full.rgba - source.rgba) * 0.5
    )
    assert mixed.rgba[0, 0, 0] > 1


def make_sequence(root):
    paths, arrays = [], []
    for index in range(3):
        pixels = np.zeros((2, 3, 4), np.float32)
        pixels[..., 3] = [[0, 0.25, 1], [1, 0.5, 0]]
        pixels[..., :3] = np.array([index + 2.25, 0.1234567, 0.75]) * pixels[..., 3:4]
        path = root / f"frame{index}.linear.npz"
        np.savez_compressed(path, rgba=pixels)
        paths.append(path)
        arrays.append(pixels)
    return paths, arrays


def test_linear_sequence_keeps_hdr_precision_and_exact_seek_boundaries(tmp_path):
    paths, arrays = make_sequence(tmp_path)
    layer = LinearSequenceLayer(paths, fps=24, duration=3 / 24)
    for t, index in [
        (2 / 24, 2),
        (0, 0),
        (1 / 24, 1),
        (np.nextafter(1 / 24, 0), 0),
        (-1, 0),
        (9, 2),
        (0, 0),
    ]:
        result = layer.source_buffer(t)
        assert result.color_space == "linear"
        np.testing.assert_array_equal(result.rgba, arrays[index])
    comp = Composition(
        size=(3, 2),
        fps=24,
        duration=3 / 24,
        context=RenderContext(working_space="linear"),
        transparent=True,
    )
    comp.add_layer(layer)
    np.testing.assert_array_equal(comp.render_buffer(1 / 24).rgba, arrays[1])


def test_linear_sequence_reverse_layer_clock_and_mask(tmp_path):
    paths, arrays = make_sequence(tmp_path)
    layer = LinearSequenceLayer(
        paths,
        fps=24,
        duration=3 / 24,
        start_time=1,
        stretch=-100,
        transform=Transform(opacity=50),
    )
    ctx = RenderContext(working_space="linear")
    np.testing.assert_array_equal(layer.render(1, ctx).rgba, arrays[2] * 0.5)
    np.testing.assert_array_equal(layer.render(1 + 2.5 / 24, ctx).rgba, arrays[0] * 0.5)
    assert layer.render(1 + 3 / 24, ctx) is None


@pytest.mark.parametrize("t", [float("inf"), float("nan"), True, "1"])
def test_linear_sequence_rejects_invalid_time(tmp_path, t):
    paths, _ = make_sequence(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        LinearSequenceLayer(paths, 24, 3 / 24).source_buffer(t)


def test_linear_resolve_averages_premultiplied_values_without_clipping(
    tmp_path, monkeypatch
):
    raw = tmp_path / "linear_raw"
    raw.mkdir()
    (raw / "frame00000000.exr").touch()
    (raw / "frame00000000.json").write_text(
        json.dumps(
            {
                "size": [2, 2],
                "frame": 0,
                "format": "RGBA16F",
                "stage": "post_transparent_before_postprocessing",
            }
        )
    )
    pixels = np.array(
        [[[8, 2, 1, 1], [9, 7, 4, 0]], [[2, 1, 0.5, 0.5], [9, 7, 4, 0]]], np.float32
    )
    monkeypatch.setattr(
        "moviepy.ae.three_d._godot_linear.decode_linear_exr", lambda path, size: pixels
    )
    frames, metadata = resolve_linear_frames(
        tmp_path, {"size": [1, 1], "fps": 24, "duration": 1 / 24}
    )
    with np.load(frames[0], allow_pickle=False) as data:
        np.testing.assert_array_equal(data["rgba"], [[[2.5, 0.75, 0.375, 0.375]]])
    assert metadata["peak_rgb"] == 2.5 and metadata["transfer"] == "linear"


@pytest.mark.skipif(
    not os.environ.get("MOVIEPY_AE_TEST_GODOT"),
    reason="requires opt-in installed Godot",
)
def test_real_godot_preserves_scene_linear_hdr_and_matches_sdr_coverage(tmp_path):
    from PIL import Image

    scene = {
        "size": [64, 48],
        "fps": 24,
        "duration": 3 / 24,
        "transparent": True,
        "materials": {
            "lamp": {
                "color": [0, 0, 0],
                "emission": [1, 0.3, 0.1],
                "emission_energy": 8,
            }
        },
        "objects": [
            {
                "type": "sphere",
                "material": "lamp",
                "scale": [2, 2, 2],
                "animation": [
                    {"time": 0, "location": [-0.5, 0, 0]},
                    {"time": 2 / 24, "location": [0.5, 0, 0]},
                ],
            }
        ],
    }
    result = render_godot_scene(
        scene,
        tmp_path / "render",
        executable=os.environ["MOVIEPY_AE_TEST_GODOT"],
        linear_output=True,
    )
    assert len(result.linear_frames) == 3 and result.metadata["linear"]["peak_rgb"] > 1
    layer = result.to_linear_layer()
    raw = decode_linear_exr(result.metadata["linear"]["exr_frames"][0], (128, 96))
    np.testing.assert_array_equal(raw[raw[..., 3] == 0, :3], 0)
    hashes = []
    for index in (2, 0, 1, 2):
        pixels = layer.source_buffer(index / 24).rgba
        assert pixels[..., :3].max() > 1
        with Image.open(result.frames[index]) as image:
            np.testing.assert_array_equal(
                np.rint(pixels[..., 3] * 255).astype(np.uint8), np.array(image)[..., 3]
            )
        hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
    assert hashes[0] == hashes[3] and len(set(hashes)) == 3
