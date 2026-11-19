"""Render validated JSON scenes with an installed Blender process.

Cycles evaluates real geometry, occlusion, lights and PBR materials. The
explicit interchange format is display-referred SDR RGBA PNG, not HDR/EXR.
No executable, texture, model or commercial SDK is downloaded automatically.
"""

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from moviepy.ae.three_d.scene import number, validate_scene


def find_blender(executable=None):
    """Resolve an explicit executable, MOVIEPY_BLENDER, PATH or usual install."""
    candidate = (
        executable or os.environ.get("MOVIEPY_BLENDER") or shutil.which("blender")
    )
    if candidate:
        path = Path(shutil.which(str(candidate)) or candidate).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Blender executable not found: {path}")
    if os.name == "nt":
        root = (
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Blender Foundation"
        )
        candidates = sorted(root.glob("Blender */blender.exe"), reverse=True)
        if candidates:
            return candidates[0]
    mac = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac.is_file():
        return mac
    raise FileNotFoundError(
        "Install Blender separately or set MOVIEPY_BLENDER to its executable"
    )


@dataclass(frozen=True)
class BlenderRender:
    """Verified image sequence and diagnostics from one completed 3D render."""

    frames: tuple
    fps: float
    duration: float
    log_path: Path
    scene_path: Path

    def to_clip(self):
        """Load the RGBA sequence as a MoviePy clip with its alpha mask."""
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        return ImageSequenceClip(
            list(self.frames), fps=self.fps, with_mask=True
        ).with_duration(self.duration)


def render_scene(scene, output_directory, *, executable=None, timeout=600):
    """Render a JSON scene into a new/empty directory, then verify every frame.

    Uses a hidden, factory-startup Blender process with automatic script
    execution disabled. Only the packaged worker runs. Existing output is
    never overwritten. Failures retain logs and raise instead of returning
    partial or stale frames. Material maps are explicit user-owned images.
    """
    scene = validate_scene(scene)
    binary = find_blender(executable)
    timeout = number(timeout, "timeout", 0.001)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scene_path = output / "scene.json"
    scene_path.write_text(
        json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log_path = output / "blender.log"
    worker = Path(__file__).with_name("_worker.py")
    command = [
        str(binary),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--python-exit-code",
        "1",
        "--python",
        str(worker),
        "--",
        str(scene_path),
        str(output),
    ]
    env = dict(os.environ, BLENDER_USER_CONFIG=str(output / "config"))
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with log_path.open("wb") as log:
        try:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=timeout,
                creationflags=flags,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Blender timed out; diagnostics: {log_path}") from error
    if process.returncode:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        )
        raise RuntimeError(f"Blender failed ({process.returncode}); {log_path}\n{tail}")
    count = math.ceil(scene["fps"] * scene["duration"])
    frames = tuple(str(output / f"frame_{n:06d}.png") for n in range(1, count + 1))
    for filename in frames:
        with Image.open(filename) as image:
            if image.size != tuple(scene["size"]) or image.mode != "RGBA":
                raise RuntimeError(f"Blender produced an unexpected frame: {filename}")
            image.verify()
    return BlenderRender(frames, scene["fps"], scene["duration"], log_path, scene_path)
