"""Render data-only scenes with Godot Movie Maker and a hidden Windows GPU window."""

import json
import math
import os
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from moviepy.ae.three_d._godot_scene import validate_godot_scene
from moviepy.ae.three_d._godot_window import hidden_render_window
from moviepy.ae.three_d._godot_worker import AUDIO_DRAIN_FRAMES, GDSCRIPT
from moviepy.ae.three_d.scene import number


def find_godot(executable=None):
    """Resolve an explicit executable, MOVIEPY_GODOT, or Godot on PATH."""
    candidate = executable or os.environ.get("MOVIEPY_GODOT")
    candidate = candidate or shutil.which("godot") or shutil.which("godot4")
    if not candidate:
        raise FileNotFoundError("Set MOVIEPY_GODOT to an installed Godot executable")
    path = Path(shutil.which(str(candidate)) or candidate).expanduser().resolve()
    if path.name.endswith("_console.exe"):
        # Run the engine itself so a timeout cannot orphan a wrapper's child.
        engine = path.with_name(path.name.replace("_console.exe", ".exe"))
        if engine.is_file():
            path = engine
    if not path.is_file():
        raise FileNotFoundError(f"Godot executable not found: {path}")
    return path


@dataclass(frozen=True)
class GodotRender:
    """A completed sequential simulation, cached RGBA frames and PCM audio."""

    frames: tuple
    fps: int
    duration: float
    audio_path: Path
    scene_path: Path
    log_path: Path
    metadata: dict
    linear_frames: tuple = ()

    def to_linear_layer(self, name="Godot Linear", **kwargs):
        """Read optional scene-linear frames into an AE layer without uint8."""
        from moviepy.ae.three_d._godot_linear import LinearSequenceLayer

        if not self.linear_frames:
            raise ValueError("render with linear_output=True for scene-linear footage")
        return LinearSequenceLayer(
            self.linear_frames, self.fps, self.duration, name, **kwargs
        )

    def to_clip(self, *, with_audio=False):
        """Open the cached frames for arbitrary seeking; close audio separately."""
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        clip = ImageSequenceClip(
            list(self.frames), fps=self.fps, with_mask=True
        ).with_duration(self.duration)
        if with_audio:
            clip.audio = AudioFileClip(str(self.audio_path)).with_duration(
                self.duration
            )
        return clip


def _write_project(output, scene, *, linear_output=False):
    # Resolve coverage ourselves: native MSAA resolves background RGB into
    # translucent pixels before tone mapping, producing fringes in a new comp.
    width, height = [value * 2 for value in scene["size"]]
    project = f"""config_version=5
[application]
config/name="MoviePy Godot render"
run/main_scene="res://main.tscn"
[display]
window/size/viewport_width={width}
window/size/viewport_height={height}
window/size/window_width_override={width}
window/size/window_height_override={height}
window/size/no_focus=true
window/stretch/mode="viewport"
[rendering]
renderer/rendering_method="forward_plus"
renderer/rendering_method.mobile="forward_plus"
rendering_device/fallback_to_opengl3=false
rendering_device/fallback_to_d3d12=false
anti_aliasing/quality/msaa_3d=0
anti_aliasing/quality/use_taa=false
transparent_background={str(scene["transparent"]).lower()}
[editor]
movie_writer/mix_rate=48000
movie_writer/speaker_mode=0
"""
    (output / "project.godot").write_text(project, encoding="utf-8")
    (output / "main.gd").write_text(GDSCRIPT, encoding="utf-8")
    if linear_output:
        from moviepy.ae.three_d._godot_capture import LINEAR_CAPTURE

        (output / "linear_raw").mkdir()
        (output / "linear_capture.gd").write_text(LINEAR_CAPTURE, encoding="utf-8")
    (output / "main.tscn").write_text(
        "[gd_scene load_steps=2 format=3]\n"
        '[ext_resource type="Script" path="res://main.gd" id="1"]\n'
        '[node name="Scene" type="Node3D"]\nscript = ExtResource("1")\n',
        encoding="utf-8",
    )


def _resolve_frames(output, scene):
    count = math.ceil(scene["fps"] * scene["duration"])
    cleanup = AUDIO_DRAIN_FRAMES if "audio" in scene else 0
    raw = output / "raw"
    expected = {raw / f"frame{n:08d}.png" for n in range(count + cleanup)}
    if set(raw.glob("frame*.png")) != expected:
        raise RuntimeError("Godot raw output has missing or extra frames")
    size = tuple(scene["size"])
    for index, path in enumerate(sorted(expected)):
        with Image.open(path) as image:
            if image.size != tuple(value * 2 for value in size):
                raise RuntimeError(f"Godot raw frame has an unexpected size: {path}")
            if scene["transparent"] and image.mode != "RGBA":
                raise RuntimeError("Godot omitted the requested alpha channel")
            pixels = np.array(image.convert("RGBA"), dtype=np.float32) / 255
        alpha = pixels[..., 3:4]
        if np.any((alpha != 0) & (alpha != 1)):
            raise RuntimeError("Godot raw coverage must be binary with MSAA disabled")
        if index >= count:
            continue
        rgb = pixels[..., :3]
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        premultiplied = cv2.resize(linear * alpha, size, interpolation=cv2.INTER_AREA)
        coverage = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)[..., None]
        linear = np.divide(
            premultiplied,
            coverage,
            out=np.zeros_like(premultiplied),
            where=coverage > 0,
        )
        rgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055,
        )
        rgba = np.rint(np.clip(np.concatenate((rgb, coverage), axis=2), 0, 1) * 255)
        Image.fromarray(rgba.astype(np.uint8)).save(output / path.name)
    _resolve_audio(output, scene, count, cleanup)


def _resolve_audio(output, scene, count, cleanup):
    """Validate the complete raw WAV and copy only the content PCM samples."""
    samples_per_frame = 48000 // scene["fps"]
    with wave.open(str(output / "raw" / "frame.wav"), "rb") as source:
        if (
            source.getframerate() != 48000
            or source.getnchannels() != 2
            or source.getsampwidth() != 4
            or source.getnframes() != (count + cleanup) * samples_per_frame
        ):
            raise RuntimeError("Godot raw audio does not match the recorded frames")
        with wave.open(str(output / "frame.wav"), "wb") as target:
            target.setparams(source.getparams())
            remaining = source.getnframes()
            content_remaining = count * samples_per_frame
            while remaining:
                chunk = min(remaining, 32768)
                pcm = source.readframes(chunk)
                if len(pcm) != chunk * 8:
                    raise RuntimeError("Godot raw audio payload is truncated")
                keep = min(content_remaining, chunk)
                if keep:
                    target.writeframesraw(pcm[: keep * 8])
                    content_remaining -= keep
                remaining -= chunk


def _verify_output(output, scene):
    count = math.ceil(scene["fps"] * scene["duration"])
    frames = tuple(str(output / f"frame{n:08d}.png") for n in range(count))
    if set(output.glob("frame*.png")) != {Path(name) for name in frames}:
        raise RuntimeError("Godot output has missing or extra frames")
    for name in frames:
        with Image.open(name) as image:
            if image.mode != "RGBA" or image.size != tuple(scene["size"]):
                raise RuntimeError(f"Godot produced an unexpected frame: {name}")
            image.verify()
    with wave.open(str(output / "frame.wav"), "rb") as audio:
        if (
            audio.getframerate() != 48000
            or audio.getnchannels() != 2
            or audio.getsampwidth() != 4
            or audio.getnframes() != count * (48000 // scene["fps"])
        ):
            raise RuntimeError("Godot audio does not match the recorded frames")
        remaining = audio.getnframes()
        while remaining:
            chunk = min(remaining, 32768)
            if len(audio.readframes(chunk)) != chunk * 8:
                raise RuntimeError("Godot audio payload is truncated")
            remaining -= chunk
    metadata = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    if (
        metadata.get("processed_frames") != count
        or metadata.get("renderer") != "forward_plus"
        or not metadata.get("gpu")
    ):
        raise RuntimeError("Godot did not complete the requested GPU simulation")
    if "audio" in scene and (
        metadata.get("audio_cleanup_frames") != AUDIO_DRAIN_FRAMES
        or metadata.get("audio_released") is not True
    ):
        raise RuntimeError("Godot did not release its audio resources")
    return frames, metadata


def render_godot_scene(
    scene, output_directory, *, executable=None, timeout=600, linear_output=False
):
    """Render a fresh RGBA sequence on Windows with Godot 4.7+ and Vulkan.

    Only the packaged GDScript runs. The scene cannot inject scripts, shaders
    or command-line arguments. Movie Maker advances sequentially at fixed fps;
    subsequent MoviePy seeks only read PNG files. This does not promise
    bit-identical particle results across GPUs or engine versions.

    ``linear_output=True`` additionally retains pre-postprocessing RGBA16F EXRs
    and resolved float32 linear frames. Use ``to_linear_layer`` to import them
    without a display transform. The existing PNG/to_clip path stays SDR.
    """
    if not isinstance(linear_output, bool):
        raise TypeError("linear_output must be a bool")
    scene = validate_godot_scene(scene)
    binary = find_godot(executable)
    timeout = number(timeout, "timeout", 0.001)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if os.name != "nt":
        raise NotImplementedError("hidden Godot rendering currently requires Windows")
    flags = subprocess.CREATE_NO_WINDOW
    version = (
        subprocess.check_output(
            [str(binary), "--version"], timeout=min(timeout, 15), creationflags=flags
        )
        .decode("utf-8", errors="replace")
        .strip()
    )
    parsed = re.match(r"4\.(\d+)\.", version)
    if not parsed or int(parsed.group(1)) < 7:
        raise ValueError(f"Godot 4.7+ is required; found {version}")
    output.mkdir(parents=True, exist_ok=True)
    scene_path = output / "scene.json"
    scene_path.write_text(
        json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_project(output, scene, linear_output=linear_output)
    runtime = output / "runtime"
    runtime.mkdir()
    (output / "raw").mkdir()
    env = dict(os.environ)
    for name in ("APPDATA", "LOCALAPPDATA", "TMP", "TEMP"):
        env[name] = str(runtime)
    log_path = output / "godot.log"
    with (
        hidden_render_window([value * 2 for value in scene["size"]]) as hwnd,
        log_path.open("wb") as log,
    ):
        command = [
            str(binary),
            "--path",
            str(output),
            "--wid",
            str(hwnd),
            "--display-driver",
            "windows",
            "--rendering-method",
            "forward_plus",
            "--rendering-driver",
            "vulkan",
            "--audio-driver",
            "Dummy",
            "--disable-vsync",
            "--log-file",
            str(output / "engine.log"),
            "--write-movie",
            str(output / "raw" / "frame.png"),
            "--fixed-fps",
            str(scene["fps"]),
            "--quit-after",
            str(
                math.ceil(scene["fps"] * scene["duration"])
                + (AUDIO_DRAIN_FRAMES if "audio" in scene else 0)
            ),
        ]
        try:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                env=env,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Godot timed out; diagnostics: {log_path}") from error
    diagnostics = log_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode or "ERROR:" in diagnostics:
        raise RuntimeError(
            f"Godot failed ({result.returncode}); {log_path}\n{diagnostics[-5000:]}"
        )
    try:
        _resolve_frames(output, scene)
        frames, metadata = _verify_output(output, scene)
        linear_frames = ()
        if linear_output:
            from moviepy.ae.three_d._godot_linear import resolve_linear_frames

            linear_frames, metadata["linear"] = resolve_linear_frames(output, scene)
            (output / "runtime.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
    except (OSError, ValueError, RuntimeError, wave.Error) as error:
        raise RuntimeError(
            f"Godot output validation failed; {log_path}: {error}"
        ) from error
    return GodotRender(
        frames,
        scene["fps"],
        scene["duration"],
        output / "frame.wav",
        scene_path,
        log_path,
        metadata,
        linear_frames,
    )
