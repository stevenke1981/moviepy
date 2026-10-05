"""Physical reference checks and real lossless-master round trips."""

import hashlib
import json
import subprocess
import wave

import numpy as np

import pytest

from moviepy import ColorClip, VideoClip, vfx
from moviepy.ae import Buffer, Composition, RenderContext, Transform, write_master
from moviepy.ae.color import (
    convert_buffer,
    display_rgba,
    linear_to_srgb,
    quantize_rgba,
    srgb_to_linear,
)
from moviepy.ae.effects import from_moviepy_effect
from moviepy.ae.effects.generate.fill import Fill
from moviepy.ae.effects.time.echo import Echo
from moviepy.ae.masks.matte import matte_values
from moviepy.ae.motion import transition_clip
from moviepy.config import FFMPEG_BINARY


def reference_decode(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def reference_encode(value):
    value = np.asarray(value, dtype=np.float64)
    return np.where(
        value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055
    )


@pytest.mark.parametrize("space", [None, [], True, "aces", "ocio:ACEScg"])
def test_working_space_requires_an_implemented_encoding(space):
    with pytest.raises((ValueError, TypeError), match="working_space"):
        RenderContext(working_space=space)


def test_transfer_breakpoints_signed_values_and_float_precision():
    values = np.array([0, 0.003, 0.04045, 0.04046, 0.18, 0.5, 1, 3])
    np.testing.assert_allclose(
        srgb_to_linear(values), reference_decode(values), rtol=2e-7
    )
    np.testing.assert_allclose(
        linear_to_srgb(srgb_to_linear(values)), values, atol=2e-7
    )
    np.testing.assert_array_equal(srgb_to_linear(-values), -srgb_to_linear(values))
    assert srgb_to_linear(values).dtype == np.float32
    assert linear_to_srgb([2])[0] > 1


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [True], ["x"]])
def test_transfer_rejects_invalid_channels(values):
    with pytest.raises((ValueError, TypeError)):
        srgb_to_linear(values)


def test_conversion_unpremultiplies_without_transforming_alpha():
    straight = np.array([[[0.6, 0.2, 0.9, 0.25], [1, 1, 1, 1e-8], [1, 0, 1, 0]]])
    premult = straight.copy()
    premult[..., :3] *= premult[..., 3:4]
    source = Buffer(premult, offset=(7, -3))
    result = convert_buffer(source, "linear")
    expected = reference_decode(straight[..., :3]) * straight[..., 3:4]
    np.testing.assert_allclose(result.rgba[..., :3], expected, atol=2e-8)
    np.testing.assert_array_equal(result.rgba[..., 3], source.rgba[..., 3])
    np.testing.assert_allclose(
        convert_buffer(result, "srgb").rgba, source.rgba, atol=3e-8
    )
    assert result.offset == (7, -3)
    assert not result.rgba.flags.writeable
    assert convert_buffer(result, "linear") is result
    with pytest.raises(ValueError, match="working_space"):
        convert_buffer(Buffer(premult, color_space="ocio:ACEScg"), "linear")


@pytest.mark.parametrize("background", [(0, 0, 0), (1, 1, 1), (0.12, 0.35, 0.8)])
def test_linear_over_matches_independent_radiometric_reference(background):
    alpha = np.array([[0, 0.001, 0.25, 0.5, 1]], np.float32)
    rgb = np.broadcast_to([0.8, 0.22, 0.06], (1, 5, 3))
    source = Buffer.from_clip(
        VideoClip(lambda t: rgb * 255).with_mask(
            VideoClip(lambda t: alpha, is_mask=True)
        ),
        0,
    )
    result = display_rgba(convert_buffer(source, "linear"), background=background)
    expected = reference_encode(
        reference_decode(rgb) * alpha[..., None]
        + reference_decode(background) * (1 - alpha[..., None])
    )
    np.testing.assert_allclose(result[..., :3], expected, atol=1.5e-7)
    np.testing.assert_array_equal(result[..., 3], 1)


def test_opt_in_linear_composition_keeps_legacy_output_and_nested_precision():
    child = Composition(size=(4, 3), fps=24, duration=1, transparent=True)
    child.add_solid(color=(255, 255, 255), transform=Transform(opacity=50))
    parent = Composition(size=(4, 3), fps=24, duration=1)
    parent.add_comp(child)
    np.testing.assert_array_equal(parent.get_frame(0), 128)
    parent.context = RenderContext(working_space="linear")
    np.testing.assert_array_equal(parent.get_frame(0), 188)
    assert parent.render_buffer(0).color_space == "linear"
    np.testing.assert_allclose(parent.render_buffer(0).rgba[..., 3], 0.5)
    # Context selection must not become a sticky cache or mutate the child.
    np.testing.assert_array_equal(child.get_frame(0), 255)
    parent.context = RenderContext()
    np.testing.assert_array_equal(parent.get_frame(0), 128)


def test_warp_filters_linear_premultiplied_color_before_export():
    source = VideoClip(lambda t: np.array([[[255, 255, 255], [0, 0, 0]]]), duration=1)
    comp = Composition(size=(3, 1), context=RenderContext(working_space="linear"))
    comp.add_clip(
        source,
        transform=Transform(
            anchor_point=(0, 0), position=(0.5, 0), interpolation="linear"
        ),
    )
    assert comp.get_frame(0)[0, 1, 0] == 188


def test_temporal_effect_reads_are_linear_and_seek_order_independent():
    source = VideoClip(
        lambda t: np.full((3, 4, 3), 255 if t >= 0.5 else 0, np.uint8), duration=1
    )
    comp = Composition(
        size=(4, 3), fps=24, duration=1, context=RenderContext(working_space="linear")
    )
    layer = comp.add_clip(source)
    layer.effects = [Echo(echo_time=-0.5, number_of_echoes=1, echo_operator="blend")]
    for t in (0.5, 0.9, 0.5):
        np.testing.assert_array_equal(comp.get_frame(t), 188)
    comp.context = RenderContext()
    np.testing.assert_array_equal(comp.get_frame(0.5), 128)


def test_color_controls_and_moviepy_bridge_interpret_authored_srgb():
    comp = Composition(size=(2, 2), context=RenderContext(working_space="linear"))
    layer = comp.add_solid(color=(0, 0, 0))
    layer.effects = [Fill(color=(64, 128, 192))]
    np.testing.assert_array_equal(comp.get_frame(0)[0, 0], [64, 128, 192])
    layer.effects = [
        Fill(color=(64, 128, 192)),
        from_moviepy_effect(vfx.InvertColors()),
    ]
    np.testing.assert_array_equal(comp.get_frame(0)[0, 0], [191, 127, 63])


def test_linear_matte_does_not_decode_already_linear_pixels_again():
    source = Buffer.from_uint8_rgb(np.full((2, 2, 3), 128, np.uint8))
    linear = convert_buffer(source, "linear")
    np.testing.assert_allclose(
        matte_values(linear, "luma", linear=True),
        reference_decode(128 / 255),
        atol=3e-8,
    )


def test_linear_transition_returns_fractional_display_codes():
    class HalfMask:
        metadata = {"effect": "transition", "size": [2, 2]}
        duration = 1
        fps = 24

        def to_mask_clip(self):
            return ColorClip((2, 2), 0.5, is_mask=True, duration=1)

    a = ColorClip((2, 2), (255, 255, 255), duration=1).with_opacity(0.25)
    b = ColorClip((2, 2), (0, 0, 0), duration=1).with_opacity(0.75)
    result = transition_clip(a, b, HalfMask(), working_space="linear")
    np.testing.assert_allclose(
        result.get_frame(0), reference_encode(0.25) * 255, atol=2e-5
    )
    np.testing.assert_allclose(result.mask.get_frame(0), 0.5)
    assert result.get_frame(0).dtype.kind == "f"


def test_16bit_quantization_retains_gradient_levels():
    values = np.linspace(0.03, 0.14, 2048)
    rgba = np.broadcast_to(values[None, :, None], (1, 2048, 4))
    eight, sixteen = quantize_rgba(rgba, bits=8), quantize_rgba(rgba)
    assert np.unique(sixteen[..., 0]).size == 2048
    assert np.unique(eight[..., 0]).size < 32
    assert np.max(np.abs(sixteen / 65535 - rgba)) <= 0.5 / 65535


def test_real_ffv1_preserves_rgba16_every_frame_and_exact_pcm(tmp_path):
    width, height, fps, count = 64, 16, 25, 7
    ramp = np.linspace(0.025, 0.45, width, dtype=np.float32)
    rgb = np.broadcast_to(ramp[None, :, None], (height, width, 3)) * 255
    alpha = np.broadcast_to(np.linspace(0, 1, width), (height, width))
    clip = VideoClip(lambda t: rgb + t, duration=count / fps).with_mask(
        VideoClip(lambda t: alpha, is_mask=True, duration=count / fps)
    )
    comp = Composition(
        size=(width, height),
        fps=fps,
        duration=count / fps,
        transparent=True,
        context=RenderContext(working_space="linear"),
    )
    comp.add_clip(clip)
    pcm = np.arange(count * (48000 // fps) * 2, dtype="<i2").tobytes()
    audio = tmp_path / "source.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        stream.writeframes(pcm)
    result = write_master(comp, tmp_path / "master", audio_path=audio)
    metadata = json.loads(result.manifest_path.read_text())
    assert metadata["frame_count"] == count
    assert metadata["pixel_format"] == "gbrap16le"
    decoded = subprocess.run(
        [
            FFMPEG_BINARY,
            "-hide_banner",
            "-i",
            str(result.path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba64le",
            "-vsync",
            "0",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    stride = width * height * 8
    assert len(decoded.stdout) == count * stride
    # Check the decoded stream, not just the flags requested by the writer.
    assert b"gbrap16le(pc, gbr/bt709/iec61966-2-1" in decoded.stderr
    for index in range(count):
        data = decoded.stdout[index * stride : (index + 1) * stride]
        assert hashlib.sha256(data).hexdigest() == metadata["rgba64le_sha256"][index]
        expected = quantize_rgba(display_rgba(comp.render_buffer(index / fps)))
        np.testing.assert_array_equal(
            np.frombuffer(data, "<u2").reshape(height, width, 4), expected
        )
    decoded_audio = subprocess.run(
        [
            FFMPEG_BINARY,
            "-v",
            "error",
            "-i",
            str(result.path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-c:a",
            "pcm_s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    assert decoded_audio.stdout == pcm
    with pytest.raises(FileExistsError):
        write_master(comp, result.path.parent)


def test_master_rejects_preview_resolution_and_wrong_audio_duration(tmp_path):
    comp = Composition(
        size=(16, 16), fps=24, duration=1, context=RenderContext(resolution_scale=0.5)
    )
    with pytest.raises(ValueError, match="full resolution"):
        write_master(comp, tmp_path / "half")
    comp.context = RenderContext()
    audio = tmp_path / "short.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 47999)
    with pytest.raises(ValueError, match="exact-duration"):
        write_master(comp, tmp_path / "short", audio_path=audio)
    assert not (tmp_path / "short").exists()


def test_master_does_not_silently_ignore_moviepy_time_wrappers(tmp_path):
    comp = Composition(size=(16, 16), fps=24, duration=2)
    with pytest.raises(ValueError, match="original AE timeline"):
        write_master(comp.subclipped(1, 2), tmp_path / "trimmed")
    assert not (tmp_path / "trimmed").exists()


@pytest.mark.parametrize("scale", [0.25, 0.5, 1])
def test_empty_linear_comp_and_preview_scale_keep_color_contract(scale):
    comp = Composition(
        size=(12, 8),
        bg_color=(31, 89, 153),
        context=RenderContext(working_space="linear", resolution_scale=scale),
    )
    np.testing.assert_array_equal(comp.get_frame(0)[0, 0], [31, 89, 153])
    # Positive linear weights preserve this constant-color reference even at
    # the source boundary. Cubic's documented ringing is tested separately.
    comp.add_solid(
        color=(255, 255, 255), transform=Transform(opacity=50, interpolation="linear")
    )
    expected = np.rint(
        reference_encode(0.5 + 0.5 * reference_decode(np.array([31, 89, 153]) / 255))
        * 255
    )
    np.testing.assert_array_equal(comp.get_frame(0)[0, 0], expected)


@pytest.mark.parametrize("space", ["srgb", "linear"])
def test_cubic_preserves_documented_rgb_overshoot_but_clamps_alpha(space):
    comp = Composition(
        size=(12, 8),
        bg_color=(31, 89, 153),
        context=RenderContext(working_space=space, resolution_scale=0.5),
    )
    comp.add_solid(
        color=(255, 255, 255), transform=Transform(opacity=50, interpolation="cubic")
    )
    buffer = comp.render_buffer(0)
    # Keys cubic (a=-0.75) has weight -3/32 at distance 1.5. At the
    # top-left edge that missing negative tap creates 35/32 gain per axis.
    # Existing API preserves RGB headroom but clips alpha before opacity.
    premultiplied = (35 / 32) ** 2 * 0.5
    np.testing.assert_array_equal(buffer.rgba[0, 0], [premultiplied] * 3 + [0.5])
    np.testing.assert_array_equal(buffer.rgba[1, 1], [0.5] * 4)
    bg = np.array([31, 89, 153], dtype=float) / 255
    expected = premultiplied + 0.5 * (reference_decode(bg) if space == "linear" else bg)
    if space == "linear":
        expected = reference_encode(expected)
    np.testing.assert_array_equal(comp.get_frame(0)[0, 0], np.rint(expected * 255))
