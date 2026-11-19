"""Lossless 16-bit RGBA intermediates from the floating-point AE renderer."""

import hashlib
import json
import math
import os
import subprocess
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from moviepy.ae.color import display_rgba, quantize_rgba
from moviepy.config import FFMPEG_BINARY


@dataclass(frozen=True)
class MasterRender:
    """A finished FFV1/Matroska master and its pixel/audio audit manifest."""

    path: Path
    manifest_path: Path
    metadata: dict


def _audio_info(path, count, fps):
    """Require an exact-duration PCM WAV, retaining its original sample bytes."""
    if path is None:
        return None
    source = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with wave.open(str(source), "rb") as audio:
        expected = count * audio.getframerate() / fps
        if (
            expected.denominator != 1
            or audio.getnframes() != expected.numerator
            or audio.getnchannels() not in (1, 2)
            or audio.getsampwidth() not in (2, 3, 4)
            or audio.getcomptype() != "NONE"
        ):
            raise ValueError(
                "audio must be exact-duration mono/stereo 16/24/32-bit PCM"
            )
        remaining = audio.getnframes()
        stride = audio.getnchannels() * audio.getsampwidth()
        while remaining:
            size = min(remaining, 32768)
            data = audio.readframes(size)
            if len(data) != size * stride:
                raise ValueError("audio PCM is truncated")
            digest.update(data)
            remaining -= size
        return {
            "path": str(source),
            "rate": audio.getframerate(),
            "channels": audio.getnchannels(),
            "sample_width": audio.getsampwidth(),
            "samples": audio.getnframes(),
            "pcm_sha256": digest.hexdigest(),
        }


def write_master(composition, output_directory, *, audio_path=None):
    """Write a lossless SDR RGBA16 master without passing through uint8 RGB.

    The directory must not exist. FFV1 preserves all quantized RGB and alpha
    codes (no chroma subsampling); sRGB/BT.709 primaries and the sRGB transfer
    are tagged explicitly. This is display-referred SDR, not an HDR export.
    RGB is straight/unassociated and alpha is linear coverage.

    A transparent composition retains alpha; an opaque one is flattened onto
    its sRGB background in its working space. Quantization happens once, at
    this output boundary. Out-of-range SDR RGB is clipped and counted.
    An optional PCM WAV must contain exactly the exported duration; it is
    stream-copied, without resampling, trimming, or a synthesized soundtrack.

    Rendering uses two encoder threads and no visible player. On failure the
    partial file and FFmpeg log remain for diagnosis; only success publishes
    ``master.mkv`` and ``manifest.json``. The caller owns all source clips.
    """
    from moviepy.ae.composition import Composition

    if not isinstance(composition, Composition):
        raise TypeError("composition must be an AE Composition")
    if (
        getattr(composition.frame_function, "__func__", None)
        is not Composition._rgb_frame
        or getattr(composition.frame_function, "__self__", None) is not composition
    ):
        raise ValueError("export the original AE timeline; apply time edits to layers")
    if composition.context.resolution_scale != 1:
        raise ValueError("master export requires full resolution_scale=1")
    fps = Fraction(str(composition.fps)).limit_denominator(1000000)
    count = math.ceil(math.nextafter(composition.duration * float(fps), -math.inf))
    if count < 1:
        raise ValueError("master export requires at least one frame")
    audio = _audio_info(audio_path, count, fps)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    partial, destination = output / "master.partial.mkv", output / "master.mkv"
    width, height = composition.size
    command = [
        FFMPEG_BINARY,
        "-hide_banner",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba64le",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps.numerator}/{fps.denominator}",
        "-i",
        "pipe:0",
    ]
    if audio:
        command += ["-i", audio["path"]]
    command += ["-map", "0:v:0"]
    if audio:
        command += ["-map", "1:a:0", "-c:a", "copy"]
    # Frame properties can override encoder-only flags in FFmpeg 7.1.
    # Set the actual filter-frame metadata so the container retains it too.
    command += [
        "-vf",
        "setparams=range=full:color_primaries=bt709:"
        "color_trc=iec61966-2-1:colorspace=gbr",
    ]
    command += [
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-coder",
        "1",
        "-context",
        "1",
        "-slicecrc",
        "1",
        "-threads",
        "2",
        "-pix_fmt",
        "gbrap16le",
        "-color_range",
        "pc",
        "-colorspace",
        "rgb",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "iec61966-2-1",
        "-frames:v",
        str(count),
        str(partial),
    ]
    metadata = {
        "schema": 1,
        "codec": "ffv1",
        "pixel_format": "gbrap16le",
        "color": "display-referred SDR sRGB / D65 / BT.709 primaries",
        "alpha": "straight RGB; linear 16-bit coverage",
        "working_space": composition.context.working_space,
        "size": list(composition.size),
        "fps": [fps.numerator, fps.denominator],
        "frame_count": count,
        "duration": float(count / fps),
        "audio": audio,
        "clipped_rgb_channels": 0,
        "rgba64le_sha256": [],
        "command": command,
    }
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (output / "ffmpeg.log").open("wb") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
            creationflags=flags,
        )
        try:
            for index in range(count):
                buffer = composition.render_buffer(float(index / fps))
                if buffer.size != composition.size or buffer.offset != (0, 0):
                    raise ValueError("master frame must cover the exact canvas")
                rgba = display_rgba(
                    buffer,
                    background=None if composition.transparent else composition._bg,
                )
                rgb = rgba[..., :3]
                metadata["clipped_rgb_channels"] += int(
                    np.count_nonzero((rgb < 0) | (rgb > 1))
                )
                packed = quantize_rgba(rgba).astype("<u2", copy=False).tobytes()
                metadata["rgba64le_sha256"].append(hashlib.sha256(packed).hexdigest())
                process.stdin.write(packed)
            process.stdin.close()
            if process.wait(timeout=120) != 0:
                raise RuntimeError(
                    f"FFmpeg master encode failed; see {output / 'ffmpeg.log'}"
                )
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            if not process.stdin.closed:
                process.stdin.close()
    partial.rename(destination)
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return MasterRender(destination, manifest, metadata)
