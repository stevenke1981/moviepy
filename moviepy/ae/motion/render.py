"""Cache complete motion sequences by content, parameters and engine version."""

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import PIL
from PIL import Image

from moviepy.ae.motion.captions import caption_frame, layout_captions
from moviepy.ae.motion.godot_effects import build_godot_motion_scene
from moviepy.ae.motion.spec import validate_motion_request
from moviepy.ae.motion.transitions import transition_mask
from moviepy.ae.three_d.godot import find_godot, render_godot_scene
from moviepy.ae.three_d.scene import number
from moviepy.video.VideoClip import VideoClip


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(request, engine_version):
    assets = {name: _sha256(path) for name, path in request["assets"].items()}
    canonical = dict(
        request,
        assets={
            name: {"sha256": assets[name], "format": Path(path).suffix.lower()}
            for name, path in request["assets"].items()
        },
    )
    module_root = Path(__file__).resolve().parent
    sources = sorted(module_root.glob("*.py"))
    sources += [
        module_root.parent / "three_d" / name
        for name in (
            "godot.py",
            "_godot_worker.py",
            "_godot_window.py",
            "_godot_scene.py",
            "scene.py",
        )
    ]
    sources.append(module_root.parent / "text" / "clusters.py")
    source_hash = hashlib.sha256(
        "".join(_sha256(path) for path in sources).encode("ascii")
    ).hexdigest()
    payload = {
        "schema": 1,
        "request": canonical,
        "engine": engine_version,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "implementation_sha256": source_hash,
        "pixels": "SDR sRGB straight RGBA8; alpha is linear coverage",
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    return digest, assets, payload


def motion_cache_key(value, *, engine_version=None):
    """Hash all rendering inputs, including local asset bytes and source code.

    For Godot effects, supply the exact engine version string. This pure lookup
    does not start an engine; render_motion_effect resolves the version itself.
    """
    request = validate_motion_request(value)
    if request["effect"] in ("title_card", "beat_particles"):
        if not isinstance(engine_version, str) or not engine_version:
            raise ValueError("Godot effects require an explicit engine_version")
    else:
        engine_version = "Pillow/NumPy"
    return _fingerprint(request, engine_version)[0]


@dataclass(frozen=True)
class MotionRender:
    """A verified, immutable-on-disk sequence suitable for arbitrary seeking."""

    frames: tuple
    fps: int
    metadata: dict
    directory: Path
    cache_hit: bool

    @property
    def duration(self):
        """Return the exact frame count divided by the frame rate."""
        return len(self.frames) / self.fps

    def to_clip(self):
        """Read straight RGB and coverage from cached integer-indexed frames."""

        @lru_cache(maxsize=4)
        def frame(index):
            with Image.open(self.frames[index]) as image:
                return np.asarray(image.convert("RGBA")).copy()

        def index(t):
            t = number(t, "clip time", 0)
            # Only correct floating error at a frame boundary, not half frames.
            return min(len(self.frames) - 1, math.floor(t * self.fps + 1e-9))

        clip = VideoClip(
            lambda t: frame(index(t))[..., :3], duration=self.duration
        ).with_fps(self.fps)
        clip.mask = VideoClip(
            lambda t: frame(index(t))[..., 3].astype(float) / 255,
            is_mask=True,
            duration=self.duration,
        ).with_fps(self.fps)
        return clip

    def to_mask_clip(self):
        """Expose the cached linear coverage for a MoviePy transition."""
        return self.to_clip().mask


def _read_cache(directory, request, key, cache_hit):
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        count, size = request["frame_count"], (request["width"], request["height"])
        metadata = manifest["metadata"]
        if (
            manifest["key"] != key
            or manifest["schema"] != 1
            or metadata["size"] != list(size)
            or metadata["fps"] != request["fps"]
            or metadata["effect"] != request["effect"]
        ):
            raise ValueError("cache manifest does not match the request")
        frames = tuple(directory / "render" / f"frame{n:08d}.png" for n in range(count))
        if (
            set((directory / "render").glob("frame*.png")) != set(frames)
            or len(manifest["frames_sha256"]) != count
        ):
            raise ValueError("cache sequence has missing or extra frames")
        for path, digest in zip(frames, manifest["frames_sha256"]):
            if _sha256(path) != digest:
                raise ValueError("cached frame hash mismatch")
            with Image.open(path) as image:
                if image.mode != "RGBA" or image.size != size:
                    raise ValueError("cached frame format mismatch")
                image.verify()
        return MotionRender(
            tuple(str(path) for path in frames),
            request["fps"],
            metadata,
            directory,
            cache_hit,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"motion cache is incomplete or corrupt: {directory}: {error}"
        ) from error


def render_motion_effect(value, cache_directory, *, godot=None, timeout=600):
    """Render sequentially once, then reuse a cache verified by SHA-256.

    Failed staging directories are retained for diagnosis. Existing cache
    entries are never overwritten, including corrupt entries.
    """
    request = validate_motion_request(value)
    uses_godot = request["effect"] in ("title_card", "beat_particles")
    binary, engine = None, "Pillow/NumPy"
    if uses_godot:
        binary = find_godot(godot)
        engine = (
            subprocess.check_output(
                [str(binary), "--version"],
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            .decode("utf-8")
            .strip()
        )
    key, assets, payload = _fingerprint(request, engine)
    cache = Path(cache_directory).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / key
    if destination.exists():
        return _read_cache(destination, request, key, True)
    # Inherit the caller's cache-directory permissions. mkdtemp's private
    # Windows ACL can otherwise survive publication and block another worker
    # that already has legitimate read access to the parent directory.
    # Keep the temporary path short: Godot appends long shader-cache names.
    # Exclusive mkdir detects a collision without weakening the final SHA key.
    while True:
        staging = cache / f".{uuid4().hex[:16]}"
        try:
            staging.mkdir()
            break
        except FileExistsError:
            continue
    metadata = {
        "effect": request["effect"],
        "fps": request["fps"],
        "size": [request["width"], request["height"]],
        "frame_count": request["frame_count"],
        "cache_key": key,
        "engine": engine,
        "asset_sha256": assets,
        "pixel_contract": payload["pixels"],
    }
    if uses_godot:
        scene = build_godot_motion_scene(request)
        result = render_godot_scene(
            scene, staging / "render", executable=binary, timeout=timeout
        )
        metadata["godot"] = result.metadata
        if len(result.frames) != request["frame_count"]:
            raise RuntimeError("Godot returned a different frame count")
        if request["effect"] == "beat_particles":
            expected = [event["frame"] for event in request["audio_events"]]
            if result.metadata["burst_start_frames"] != expected:
                raise RuntimeError(
                    "Godot particle triggers do not match the event frames"
                )
    else:
        (staging / "render").mkdir()
        layouts = layout_captions(request) if request["effect"] == "captions" else []
        if layouts:
            metadata["layout"] = [
                {
                    "font_size": item["font_size"],
                    "boxes": item["boxes"],
                    "start_frame": item["cue"]["start_frame"],
                    "end_frame": item["cue"]["end_frame"],
                }
                for item in layouts
            ]
        for index in range(request["frame_count"]):
            if layouts:
                rgba = caption_frame(request, layouts, index)
            else:
                mask = transition_mask(request, frame_index=index)
                rgba = np.full((*mask.shape, 4), 255, np.uint8)
                rgba[..., 3] = np.rint(mask * 255).astype(np.uint8)
            Image.fromarray(rgba).save(staging / "render" / f"frame{index:08d}.png")
    if assets != {name: _sha256(path) for name, path in request["assets"].items()}:
        raise RuntimeError(
            "an input asset changed during rendering; staging output retained"
        )
    manifest = {
        "schema": 1,
        "key": key,
        "inputs": payload,
        "metadata": metadata,
        "frames_sha256": [
            _sha256(staging / "render" / f"frame{n:08d}.png")
            for n in range(request["frame_count"])
        ],
    }
    (staging / "request.json").write_text(
        json.dumps(request, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _read_cache(staging, request, key, False)
    try:
        staging.rename(destination)
    except OSError:
        # Another process may have completed the same key in the meantime.
        # Keep our staging evidence and accept only a fully verified winner.
        if destination.exists():
            return _read_cache(destination, request, key, True)
        raise
    return _read_cache(destination, request, key, False)
