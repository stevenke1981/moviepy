"""Check external-render boundaries and opt-in real GPU movie generation."""

import copy
import json
import os
import wave

import numpy as np
from PIL import Image

import pytest

from moviepy.ae.three_d import find_godot, render_godot_scene
from moviepy.ae.three_d._godot_scene import validate_godot_scene
from moviepy.ae.three_d.godot import _resolve_audio, _resolve_frames, _verify_output


@pytest.mark.parametrize(
    "scene",
    [
        {"script": "arbitrary.gd"},
        {"fps": 24.5},
        {"fps": 7},
        {"transparent": 1},
        {"transparent": True, "glow": 1},
        {"camera": {"location": [0, 3, 0]}},
        {"objects": [{"type": "text", "text": ""}]},
        {"objects": [{"type": "cube", "font": "ignored.ttf"}]},
        {"objects": [{"type": "cube", "scale": [0, 1, 1]}]},
        {"objects": [{"type": "cube", "animation": [{"time": 0}]}]},
        {
            "objects": [
                {
                    "type": "cube",
                    "animation": [
                        {"time": 0, "location": [0, 0, 0]},
                        {"time": 0, "rotation": [0, 0, 0]},
                    ],
                }
            ]
        },
        {"particles": [{"speed": [3, 1]}]},
        {"particles": [{"start": 2}]},
        {"particles": [{"direction": [0, 0, 0]}]},
        {"particles": [{"seed": float("nan")}]},
        {"materials": {"x": {"shader": "void fragment() {}"}}},
    ],
)
def test_godot_rejects_unsupported_or_ambiguous_scene_data(scene):
    with pytest.raises((TypeError, ValueError)):
        validate_godot_scene(scene)


def test_godot_scene_does_not_mutate_input_and_sorts_keys():
    scene = {
        "objects": [
            {
                "type": "cube",
                "animation": [
                    {"time": 1, "rotation": [0, 90, 0]},
                    {"time": 0, "rotation": [0, 0, 0]},
                ],
            }
        ],
        "lights": [{"type": "omni", "location": [0, 4, 0]}],
    }
    original = copy.deepcopy(scene)
    normalized = validate_godot_scene(scene)
    assert scene == original
    assert normalized["objects"][0]["animation"][0]["time"] == 0
    normalized["objects"][0]["animation"].clear()
    assert len(scene["objects"][0]["animation"]) == 2


def test_godot_executable_resolution_and_existing_output(tmp_path, monkeypatch):
    binary = tmp_path / "Godot.exe"
    wrapper = tmp_path / "Godot_console.exe"
    binary.touch()
    wrapper.touch()
    monkeypatch.setenv("MOVIEPY_GODOT", str(wrapper))
    assert find_godot() == binary
    with pytest.raises(FileNotFoundError):
        find_godot(tmp_path / "missing.exe")
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        render_godot_scene({}, output)
    assert sentinel.read_text(encoding="utf-8") == "user data"


def test_godot_output_validation_detects_truncated_and_mismatched_sequences(tmp_path):
    scene = validate_godot_scene({"size": [4, 3], "fps": 10, "duration": 0.2})
    for n in range(2):
        Image.new("RGBA", (4, 3), (255, 0, 0, 255)).save(tmp_path / f"frame{n:08d}.png")
    with wave.open(str(tmp_path / "frame.wav"), "wb") as audio:
        audio.setparams((2, 4, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(9600 * 8))
    metadata = {
        "processed_frames": 2,
        "renderer": "forward_plus",
        "gpu": "test fixture",
    }
    (tmp_path / "runtime.json").write_text(json.dumps(metadata), encoding="utf-8")
    frames, _ = _verify_output(tmp_path, scene)
    assert len(frames) == 2
    Image.new("RGBA", (4, 3)).save(tmp_path / "frame00000002.png")
    with pytest.raises(RuntimeError, match="extra"):
        _verify_output(tmp_path, scene)
    (tmp_path / "frame00000002.png").unlink()
    Image.new("RGB", (4, 3)).save(tmp_path / "frame00000001.png")
    with pytest.raises(RuntimeError, match="unexpected frame"):
        _verify_output(tmp_path, scene)
    Image.new("RGBA", (4, 3)).save(tmp_path / "frame00000001.png")
    wav = tmp_path / "frame.wav"
    wav.write_bytes(wav.read_bytes()[:-4])
    with pytest.raises(RuntimeError, match="truncated"):
        _verify_output(tmp_path, scene)


def test_godot_edge_resolve_discards_transparent_background_color(tmp_path):
    scene = validate_godot_scene({"size": [2, 2], "fps": 10, "duration": 0.1})
    raw = tmp_path / "raw"
    raw.mkdir()
    pixels = np.zeros((4, 4, 4), np.uint8)
    pixels[..., :3] = [255, 255, 255]  # RGB behind alpha=0 must not bleed.
    pixels[0, 0] = [200, 90, 30, 255]
    Image.fromarray(pixels).save(raw / "frame00000000.png")
    with wave.open(str(raw / "frame.wav"), "wb") as audio:
        audio.setparams((2, 4, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(4800 * 8))
    _resolve_frames(tmp_path, scene)
    resolved = np.array(Image.open(tmp_path / "frame00000000.png"))
    np.testing.assert_array_equal(resolved[0, 0], [200, 90, 30, 64])
    np.testing.assert_array_equal(resolved[1, 1], [0, 0, 0, 0])


def test_godot_audio_cleanup_retains_exact_content_and_checks_discarded_tail(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pcm = np.arange(6400 * 2, dtype="<i4").tobytes()
    wav = raw / "frame.wav"
    with wave.open(str(wav), "wb") as audio:
        audio.setparams((2, 4, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)
    _resolve_audio(tmp_path, {"fps": 120}, 12, 4)
    with wave.open(str(tmp_path / "frame.wav"), "rb") as audio:
        assert audio.getnframes() == 4800
        assert audio.readframes(4800) == pcm[: 4800 * 8]
    wav.write_bytes(wav.read_bytes()[:-8])
    with pytest.raises(RuntimeError, match="truncated"):
        _resolve_audio(tmp_path, {"fps": 120}, 12, 4)


@pytest.mark.skipif(
    not os.environ.get("MOVIEPY_AE_TEST_GODOT"),
    reason="opt-in Godot 4.7+ with a Windows GPU",
)
def test_actual_godot_rgba_audio_events_and_arbitrary_clip_seeks(tmp_path):
    audio_path = tmp_path / "tone.wav"
    samples = (np.sin(np.arange(38400) / 48000 * 2 * np.pi * 220) * 5000).astype("<i2")
    with wave.open(str(audio_path), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())
    scene = {
        "size": [160, 120],
        "fps": 10,
        "duration": 0.8,
        "audio": str(audio_path),
        "materials": {"red": {"color": [0.9, 0.06, 0.01]}},
        "objects": [
            {
                "type": "cube",
                "material": "red",
                "animation": [
                    {"time": 0, "rotation": [0, 0, 0]},
                    {"time": 0.8, "rotation": [0, 75, 0]},
                ],
            }
        ],
        "lights": [{"type": "directional", "location": [-2, 5, 4]}],
        "particles": [{"location": [-1, 0, 0], "count": 64, "start": 0.3, "seed": 41}],
    }
    rendered = render_godot_scene(
        scene,
        tmp_path / "render",
        executable=os.environ["MOVIEPY_AE_TEST_GODOT"],
        timeout=120,
    )
    assert len(rendered.frames) == 8
    assert rendered.metadata["audio_released"] is True
    assert rendered.metadata["audio_cleanup_frames"] == 4
    assert "ObjectDB instances were leaked" not in rendered.log_path.read_text(
        encoding="utf-8"
    )
    assert rendered.metadata["burst_start_frames"] == [3]
    rgba = np.array(Image.open(rendered.frames[0]))
    assert rgba.shape == (120, 160, 4)
    assert rgba[0, 0, 3] == 0 and rgba[..., 3].max() == 255
    assert np.any((rgba[..., 3] > 0) & (rgba[..., 3] < 255))
    opaque = rgba[..., 3] == 255
    assert rgba[..., 0][opaque].mean() > rgba[..., 2][opaque].mean()
    assert not np.array_equal(rgba, np.array(Image.open(rendered.frames[6])))
    with wave.open(str(rendered.audio_path), "rb") as audio:
        values = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i4")
        assert np.max(np.abs(values)) > 100000
    clip = rendered.to_clip(with_audio=True)
    try:
        later = clip.get_frame(0.6).copy()
        first = clip.get_frame(0).copy()
        np.testing.assert_array_equal(clip.get_frame(0.6), later)
        np.testing.assert_array_equal(first, rgba[..., :3])
        np.testing.assert_allclose(clip.mask.get_frame(0), rgba[..., 3] / 255)
        assert clip.audio.duration == clip.duration == 0.8
    finally:
        clip.audio.close()
        clip.close()


@pytest.mark.skipif(
    not os.environ.get("MOVIEPY_AE_TEST_GODOT"),
    reason="opt-in Godot 4.7+ with a Windows GPU",
)
def test_actual_godot_releases_active_audio_at_maximum_frame_rate(tmp_path):
    source = tmp_path / "tone.wav"
    samples = (np.sin(np.arange(48000) / 48000 * 2 * np.pi * 220) * 12000).astype("<i2")
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(samples.tobytes())
    result = render_godot_scene(
        {"size": [16, 16], "fps": 120, "duration": 0.1, "audio": str(source)},
        tmp_path / "render",
        executable=os.environ["MOVIEPY_AE_TEST_GODOT"],
        timeout=120,
    )
    assert len(result.frames) == 12
    assert result.duration == 0.1
    assert result.metadata["audio_released"] is True
    assert result.metadata["audio_cleanup_frames"] == 4
    assert "ObjectDB instances were leaked" not in result.log_path.read_text(
        encoding="utf-8"
    )
    with wave.open(str(result.audio_path), "rb") as audio:
        assert audio.getnframes() == 4800
        pcm = audio.readframes(4800)
    with wave.open(str(result.audio_path.parent / "raw" / "frame.wav"), "rb") as audio:
        assert audio.getnframes() == 6400
        assert audio.readframes(4800) == pcm
    # A premature stop can fade the final samples even when the WAV length is right.
    values = np.frombuffer(pcm, "<i4")
    assert np.sqrt(np.mean(values[-800:].astype(float) ** 2)) > 1e8
