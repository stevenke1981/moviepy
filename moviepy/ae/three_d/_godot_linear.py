"""Read Godot's premultiplied scene-linear EXRs without a new codec dependency."""

import bisect
import json
import math
import os
import subprocess
from numbers import Real
from pathlib import Path

import cv2
import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.base import Layer
from moviepy.config import FFMPEG_BINARY


def decode_linear_exr(path, size):
    """Decode native float channels; FFmpeg performs no display transform."""
    width, height = size
    command = [
        FFMPEG_BINARY,
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gbrapf32le",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode or len(result.stdout) != width * height * 16:
        raise RuntimeError(
            f"invalid linear EXR {path}: {result.stderr.decode('utf-8', errors='replace')}"
        )
    planes = np.frombuffer(result.stdout, dtype="<f4").reshape(4, height, width)
    pixels = np.stack((planes[2], planes[0], planes[1], planes[3]), axis=2)
    if not np.isfinite(pixels).all() or np.any(
        (pixels[..., 3] < 0) | (pixels[..., 3] > 1)
    ):
        raise RuntimeError(f"invalid finite color/coverage in linear EXR {path}")
    return pixels


def resolve_linear_frames(output, scene):
    """Resolve 2x premultiplied linear coverage without clipping or encoding."""
    count = math.ceil(scene["fps"] * scene["duration"])
    raw = output / "linear_raw"
    paths = tuple(raw / f"frame{index:08d}.exr" for index in range(count))
    if set(raw.glob("frame*.exr")) != set(paths):
        raise RuntimeError("Godot linear output has missing or extra frames")
    size = tuple(scene["size"])
    native_size = tuple(value * 2 for value in size)
    frames = []
    peak = 0.0
    for index, path in enumerate(paths):
        metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (
            metadata.get("size") != list(native_size)
            or metadata.get("format") != "RGBA16F"
            or metadata.get("frame") != index
            or metadata.get("stage") != "post_transparent_before_postprocessing"
        ):
            raise RuntimeError(f"unexpected linear capture metadata: {path}")
        pixels = decode_linear_exr(path, native_size)
        pixels[pixels[..., 3] == 0, :3] = 0
        pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
        pixels[pixels[..., 3] == 0, :3] = 0
        peak = max(peak, float(np.max(pixels[..., :3])))
        destination = output / f"frame{index:08d}.linear.npz"
        np.savez_compressed(destination, rgba=pixels)
        frames.append(str(destination))
    return tuple(frames), {
        "format": "RGBA16F EXR; float32 premultiplied NPZ after 2x area resolve",
        "transfer": "linear",
        "primaries": "sRGB/BT.709 D65",
        "alpha": "premultiplied RGB; linear coverage",
        "stage": "post_transparent_before_postprocessing",
        "frames": count,
        "peak_rgb": peak,
        "exr_frames": [str(path) for path in paths],
    }


class LinearSequenceLayer(Layer):
    """Read an immutable resolved Godot sequence directly into linear Buffers.

    Use the layer's start_time/stretch for timing. Each request reads the
    selected frame, without an order-dependent decoder or persistent cache.
    """

    def __init__(self, frames, fps, duration, name="Godot Linear", **kwargs):
        self._frames = tuple(Path(path) for path in frames)
        if not self._frames:
            raise ValueError("linear frames must not be empty")
        self._fps, self._source_duration = float(fps), float(duration)
        if (
            not math.isfinite(self._fps)
            or self._fps <= 0
            or not math.isfinite(self._source_duration)
            or self._source_duration <= 0
            or len(self._frames) != math.ceil(self._fps * self._source_duration)
        ):
            raise ValueError("linear frame count must match finite fps and duration")
        self._times = tuple(index / self._fps for index in range(len(self._frames)))
        self._size = None
        self._size = self.source_buffer(0).size
        super().__init__(name, **kwargs)

    @property
    def source_size(self):
        """Return the resolved sequence dimensions."""
        return self._size

    def default_out_point(self):
        """End after the source duration scaled by the layer stretch."""
        return self.start_time + self._source_duration * abs(self.stretch) / 100

    def source_time(self, t):
        """Map composition time into forward or reverse sequence time."""
        time = super().source_time(t)
        return time if self.stretch >= 0 else self._source_duration + time

    def source_buffer(self, t, context=None):
        """Read the frame at source time, holding the sequence endpoints."""
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, Real):
            raise TypeError("source time must be a real number")
        if not math.isfinite(t):
            raise ValueError("source time must be finite")
        index = max(
            0, min(bisect.bisect_right(self._times, t) - 1, len(self._frames) - 1)
        )
        with np.load(self._frames[index], allow_pickle=False) as archive:
            pixels = archive["rgba"]
        if pixels.dtype != np.float32:
            raise ValueError("linear frame must contain float32 RGBA")
        buffer = Buffer(pixels, color_space="linear")
        if self._size is not None and buffer.size != self._size:
            raise ValueError("linear frame size changed")
        return buffer
