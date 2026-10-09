"""Streaming audio I/O through MoviePy's FFmpeg binary (no extra packages).

Long music masters (hours at 48 kHz) never fit in memory as float arrays, so
every reader and writer here works in bounded blocks. Samples are float32,
shape ``(frames, channels)``. Writers refuse to replace an existing file
unless ``overwrite=True``, matching the music workflow's exclusive-create
rule.

Examples
--------
>>> import numpy as np, tempfile, os
>>> folder = tempfile.mkdtemp()
>>> path = os.path.join(folder, "tone.wav")
>>> tone = np.zeros((4800, 2), np.float32)
>>> with AudioWriter(path, rate=48000) as writer:
...     writer.write(tone)
>>> audio_info(path)["frames"]
4800
>>> read_audio(path).shape
(4800, 2)
"""

import os
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np

from moviepy.config import FFMPEG_BINARY


DEFAULT_RATE = 48000
_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _flac_info(path):
    """Read rate, channels and total frames from a FLAC STREAMINFO block."""
    with open(path, "rb") as stream:
        if stream.read(4) != b"fLaC":
            return None
        header = stream.read(4)
        if len(header) < 4 or header[0] & 0x7F != 0:
            return None
        info = stream.read(18)
    if len(info) < 18:
        return None
    packed = int.from_bytes(info[10:18], "big")
    rate = packed >> 44
    channels = ((packed >> 41) & 0x7) + 1
    frames = packed & ((1 << 36) - 1)
    if not rate or not frames:
        return None
    return {"rate": rate, "channels": channels, "frames": frames}


def _wav_info(path):
    """Read WAV metadata, including IEEE float and extensible headers."""
    try:
        with wave.open(str(path), "rb") as handle:
            return {
                "rate": handle.getframerate(),
                "channels": handle.getnchannels(),
                "frames": handle.getnframes(),
            }
    except (wave.Error, EOFError):
        pass
    with open(path, "rb") as stream:
        if stream.read(4) != b"RIFF":
            return None
        stream.read(4)
        if stream.read(4) != b"WAVE":
            return None
        fmt = None
        while True:
            chunk = stream.read(8)
            if len(chunk) < 8:
                return None
            name, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if name == b"fmt ":
                body = stream.read(size)
                channels, rate = struct.unpack("<HI", body[2:8])
                bits = struct.unpack("<H", body[14:16])[0]
                fmt = (channels, rate, bits)
            elif name == b"data" and fmt is not None:
                channels, rate, bits = fmt
                return {
                    "rate": rate,
                    "channels": channels,
                    "frames": size // (channels * bits // 8),
                }
            else:
                stream.seek(size + (size & 1), os.SEEK_CUR)


def audio_info(path):
    """Return ``{"rate", "channels", "frames", "seconds"}`` with exact frames.

    WAV and FLAC headers are read directly. Other formats are decoded once
    through FFmpeg to count samples exactly (slow for very long files).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    info = None
    if path.suffix.lower() == ".flac":
        info = _flac_info(path)
    elif path.suffix.lower() in (".wav", ".wave"):
        info = _wav_info(path)
    if info is None:
        frames = 0
        rate = DEFAULT_RATE
        for block in iter_audio(path, rate=rate, channels=2):
            frames += len(block)
        info = {"rate": rate, "channels": 2, "frames": frames}
    info["seconds"] = info["frames"] / info["rate"]
    return info


def iter_audio(
    path, *, rate=DEFAULT_RATE, channels=2, start=0, frames=None, block=None
):
    """Yield float32 ``(n, channels)`` blocks decoded and resampled by FFmpeg.

    ``start`` and ``frames`` are sample counts at ``rate``. Seeking uses an
    accurate output-side trim, so block boundaries are sample exact.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    block = int(block or rate * 10)
    command = [FFMPEG_BINARY, "-hide_banner", "-nostdin", "-loglevel", "error"]
    command += ["-i", str(path), "-map", "0:a:0", "-vn"]
    # Resample before trimming so start/frames count samples at ``rate``.
    filters = [f"aresample={rate}"]
    trim = []
    if start:
        trim.append(f"start_sample={int(start)}")
    if frames is not None:
        trim.append(f"end_sample={int(start) + int(frames)}")
    if trim:
        filters.append("atrim=" + ":".join(trim))
    filters.append("asetpts=PTS-STARTPTS")
    command += ["-af", ",".join(filters), "-f", "f32le", "-acodec", "pcm_f32le"]
    command += ["-ar", str(rate), "-ac", str(channels), "-"]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_FLAGS
    )
    width = 4 * channels
    try:
        while True:
            data = process.stdout.read(block * width)
            if not data:
                break
            usable = len(data) - len(data) % width
            yield np.frombuffer(data[:usable], np.float32).reshape(-1, channels)
        error = process.stderr.read().decode("utf-8", "replace")
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg could not decode {path}: {error[-2000:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def read_audio(path, *, rate=DEFAULT_RATE, channels=2, start=0, frames=None):
    """Decode a whole (or ``start``/``frames`` slice of an) audio file."""
    blocks = list(
        iter_audio(path, rate=rate, channels=channels, start=start, frames=frames)
    )
    if not blocks:
        return np.zeros((0, channels), np.float32)
    return np.concatenate(blocks)


class AudioWriter:
    """Stream float32 blocks into a lossless file through FFmpeg.

    The codec follows the suffix: ``.flac`` (32-bit FLAC), ``.wav`` (24-bit
    PCM by default, ``subtype="float"`` for 32-bit float). Samples are not
    clipped here; callers check peaks. ``frames_written`` counts samples.
    """

    def __init__(
        self, path, *, rate=DEFAULT_RATE, channels=2, subtype="pcm24", overwrite=False
    ):
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(self.path)
        suffix = self.path.suffix.lower()
        if suffix == ".flac":
            codec = ["-c:a", "flac", "-sample_fmt", "s32"]
        elif suffix in (".wav", ".wave"):
            codec = ["-c:a", "pcm_f32le" if subtype == "float" else "pcm_s24le"]
        else:
            raise ValueError("AudioWriter supports .flac and .wav outputs")
        self.rate, self.channels, self.frames_written = rate, channels, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [FFMPEG_BINARY, "-hide_banner", "-nostdin", "-loglevel", "error"]
        command += ["-f", "f32le", "-ar", str(rate), "-ac", str(channels), "-i", "-"]
        command += [*codec, "-y" if overwrite else "-n", str(self.path)]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_FLAGS,
        )

    def write(self, block):
        """Append a ``(n, channels)`` block of float samples."""
        block = np.ascontiguousarray(block, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self.channels:
            raise ValueError(f"blocks must have shape (n, {self.channels})")
        if not np.isfinite(block).all():
            raise ValueError("audio blocks must be finite")
        self._process.stdin.write(block.tobytes())
        self.frames_written += len(block)

    def close(self):
        """Finish the file and raise if FFmpeg failed."""
        if self._process is None:
            return
        process, self._process = self._process, None
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", "replace")
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg could not write {self.path}: {error[-2000:]}")

    def abort(self):
        """Stop FFmpeg and remove the partial file."""
        if self._process is not None:
            self._process.kill()
            self._process.wait()
            self._process = None
        if self.path.exists():
            self.path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.close()
        else:
            self.abort()
        return False
