"""Automated technical QA for long music videos.

These reports are a machine pre-check, never a substitute for human QA. Every
report keeps ``human_listening`` and ``human_visual`` at ``"NOT_RUN"`` until a
person signs them. Hours-long inputs are streamed: video is decoded at a small
size (160x90 by default) through an FFmpeg rawvideo pipe, audio through
``iter_audio`` from ``moviepy.ae.templates._audio_io``, and only running statistics
are kept, so memory does not grow with duration.

The checks follow the music channel's ``s16-qa.py`` and ``s14-verify.py``:
exclusive-create evidence JSON, measured facts instead of assumed ones, and
exact stream facts. No ffprobe is needed; stream facts are parsed from the
``ffmpeg -i`` banner.

The flash check is an approximation of WCAG 2.3.1 (general flash and
red flash). It is not a certified Harding test. Red flash is reported as
``"NOT_COMPUTED"``.

Examples
--------
>>> flash_verdict(2)
'PASS'
>>> flash_verdict(3)
'FAIL'
"""

import json
import os
import re
import subprocess
from collections import deque
from pathlib import Path

import numpy as np

from moviepy.ae.templates._audio_io import iter_audio
from moviepy.config import FFMPEG_BINARY


__all__ = [
    "DEFAULT_TARGETS",
    "clipping_report",
    "flash_report",
    "flash_verdict",
    "loudness_report",
    "qa_report",
    "silence_report",
    "stream_facts",
]

DEFAULT_TARGETS = {"lufs": -18.0, "lufs_tolerance": 1.0, "true_peak_max": -1.5}
FLASH_LIMIT = 3
_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
_BUDGET = 16_000_000
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_SRGB = np.arange(256, dtype=np.float64) / 255.0
_SRGB_LUT = np.where(
    _SRGB <= 0.04045, _SRGB / 12.92, ((_SRGB + 0.055) / 1.055) ** 2.4
).astype(np.float32)
_DURATION = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_SIZE = re.compile(r",\s*(\d{2,5})x(\d{2,5})\b")
_FPS = re.compile(r"(\d+(?:\.\d+)?)\s*fps\b")
_TBR = re.compile(r"(\d+(?:\.\d+)?)\s*tbr\b")
_AUDIO = re.compile(r"Audio:\s*([^,\s]+).*?,\s*(\d+) Hz,\s*([^,\n]+)")
_FRAME_LOG = re.compile(r"\bM:\s*(-?[\d.]+|-?inf)\s+S:\s*(-?[\d.]+|-?inf)")
_LUFS_LINE = re.compile(r"I:\s*(-?[\d.]+|-?inf)\s*LUFS$")
_LRA_LINE = re.compile(r"LRA:\s*(-?[\d.]+)\s*LU$")
_PEAK_LINE = re.compile(r"Peak:\s*(-?[\d.]+|-?inf)\s*dBFS$")
_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def _ffmpeg_run(args):
    """Run FFmpeg with a fixed prefix and return its stderr text."""
    command = [FFMPEG_BINARY, "-hide_banner", "-nostdin", *args]
    done = subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_FLAGS
    )
    return done.stderr.decode("utf-8", "replace")


def _finite(value):
    """Return ``float(value)`` or ``None`` when the value is not finite."""
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _merge_targets(targets):
    """Return the default targets updated with the caller's overrides."""
    merged = dict(DEFAULT_TARGETS)
    for key, value in (targets or {}).items():
        if key not in DEFAULT_TARGETS:
            raise ValueError(f"unknown QA target: {key}")
        merged[key] = float(value)
    return merged


def _seconds(match):
    """Convert an ``HH:MM:SS.ss`` match from the FFmpeg banner to seconds."""
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_facts(text):
    """Parse the ``ffmpeg -i`` banner into stream facts.

    Parameters
    ----------
    text : str
        The stderr text printed by FFmpeg for an input file.

    Returns
    -------
    dict
        Duration, the first video stream and the first audio stream.
    """
    if "Input #0" not in text:
        raise ValueError("FFmpeg did not report an input stream")
    facts = {"duration": None, "video": None, "audio": None}
    duration = _DURATION.search(text)
    if duration:
        facts["duration"] = _seconds(duration)
    for line in text.splitlines():
        if "Stream #" not in line:
            continue
        if (
            ": Video:" in line
            and facts["video"] is None
            and "(attached pic)" not in line
        ):
            size = _SIZE.search(line)
            fps = _FPS.search(line) or _TBR.search(line)
            codec = line.split(": Video:", 1)[1].split()[0].rstrip(",")
            fps_value = float(fps.group(1)) if fps else None
            frames = None
            if facts["duration"] is not None and fps_value:
                frames = int(round(facts["duration"] * fps_value))
            facts["video"] = {
                "codec": codec,
                "width": int(size.group(1)) if size else None,
                "height": int(size.group(2)) if size else None,
                "fps": fps_value,
                "frames_estimate": frames,
            }
        elif ": Audio:" in line and facts["audio"] is None:
            found = _AUDIO.search(line)
            if found:
                layout = found.group(3).strip()
                channels = {"mono": 1, "stereo": 2}.get(layout)
                facts["audio"] = {
                    "codec": found.group(1),
                    "sample_rate": int(found.group(2)),
                    "layout": layout,
                    "channels": channels,
                }
    return facts


def stream_facts(media):
    """Return duration and first video and audio stream facts of a file.

    Parameters
    ----------
    media : str or pathlib.Path
        Video or audio file.

    Returns
    -------
    dict
        ``duration`` in seconds, ``video`` (codec, width, height, fps,
        frames_estimate) or ``None``, and ``audio`` (codec, sample_rate,
        layout, channels) or ``None``. ``frames_estimate`` is
        ``round(duration * fps)``, not a decoded count.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from moviepy.ae.templates._audio_io import AudioWriter
    >>> path = os.path.join(tempfile.mkdtemp(), "tone.wav")
    >>> with AudioWriter(path, rate=48000) as writer:
    ...     writer.write(np.zeros((4800, 2), np.float32))
    >>> facts = stream_facts(path)
    >>> facts["audio"]["sample_rate"], facts["audio"]["channels"]
    (48000, 2)
    """
    path = Path(media)
    if not path.is_file():
        raise FileNotFoundError(path)
    return _parse_facts(_ffmpeg_run(["-i", str(path)]))


def flash_verdict(max_flashes_per_second, *, limit=FLASH_LIMIT):
    """Return ``"PASS"`` when flashes per second stay below ``limit``.

    Parameters
    ----------
    max_flashes_per_second : int
        Largest number of flashes counted in any one-second window.
    limit : int
        Flash count that fails the check. The default is WCAG's three.

    Returns
    -------
    str
        ``"PASS"`` or ``"FAIL"``.

    Examples
    --------
    >>> flash_verdict(0), flash_verdict(3)
    ('PASS', 'FAIL')
    """
    return "PASS" if max_flashes_per_second < limit else "FAIL"


def _read_exact(stream, count):
    """Read ``count`` bytes, or fewer only at end of stream."""
    chunks, left = [], count
    while left > 0:
        data = stream.read(left)
        if not data:
            break
        chunks.append(data)
        left -= len(data)
    return b"".join(chunks)


def _decode_frames(media, *, start, seconds, fps, size, batch):
    """Yield ``(n, height, width, 3)`` uint8 RGB batches from a video stream."""
    width, height = size
    filters = [f"fps={float(fps):g}"] if fps else []
    filters += [f"scale={width}:{height}:flags=area", "format=rgb24"]
    command = [FFMPEG_BINARY, "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start:
        command += ["-ss", f"{float(start):.6f}"]
    command += ["-i", str(media)]
    if seconds is not None:
        command += ["-t", f"{float(seconds):.6f}"]
    command += [
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-vf",
        ",".join(filters),
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    frame_bytes = width * height * 3
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_FLAGS
    )
    try:
        while True:
            data = _read_exact(process.stdout, frame_bytes * batch)
            usable = len(data) - len(data) % frame_bytes
            if usable:
                yield np.frombuffer(data[:usable], np.uint8).reshape(
                    -1, height, width, 3
                )
            if len(data) < frame_bytes * batch:
                break
        error = process.stderr.read().decode("utf-8", "replace")
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg could not decode {media}: {error[-2000:]}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


class _FlashScanner:
    """Accumulate flash statistics over consecutive luminance batches."""

    def __init__(self, start, rate, series):
        self.start, self.rate = float(start), float(rate)
        self.prev = None
        self.prev_mean = None
        self.frames = 0
        self.flashes = 0
        self.max_step = 0.0
        self.max_area = 0.0
        self.mean_sum = 0.0
        self.pending = None
        self.window = deque()
        self.best = 0
        self.best_start = None
        self.means = [] if series else None

    def feed(self, luma):
        """Add a ``(n, height, width)`` batch of relative luminance."""
        means = luma.reshape(len(luma), -1).mean(axis=1, dtype=np.float64)
        self.mean_sum += float(means.sum())
        if self.means is not None:
            self.means.extend(float(value) for value in means)
        if self.prev is None:
            seq, base = luma, self.frames
            steps = np.diff(means)
        else:
            seq, base = np.concatenate([self.prev[None], luma]), self.frames - 1
            steps = np.diff(np.concatenate([[self.prev_mean], means]))
        if len(steps):
            self.max_step = max(self.max_step, float(np.abs(steps).max()))
        if len(seq) > 1:
            before, after = seq[:-1], seq[1:]
            delta = after - before
            changed = (np.abs(delta) >= 0.1) & (np.minimum(before, after) < 0.8)
            area = changed.mean(axis=(1, 2))
            up = (changed & (delta > 0)).mean(axis=(1, 2))
            down = (changed & (delta < 0)).mean(axis=(1, 2))
            self.max_area = max(self.max_area, float(area.max()))
            significant = np.nonzero((area >= 0.25) & (up != down))[0]
            for index in significant:
                sign = 1 if up[index] > down[index] else -1
                time = self.start + (base + index + 1) / self.rate
                self._change(sign, time)
        self.prev = luma[-1]
        self.prev_mean = float(means[-1])
        self.frames += len(luma)

    def _change(self, sign, time):
        """Pair opposing changes that happen within one second."""
        if self.pending is not None:
            pending_sign, pending_time = self.pending
            if sign != pending_sign and time - pending_time <= 1.0:
                self.pending = None
                self.flashes += 1
                self._count(time)
                return
        self.pending = (sign, time)

    def _count(self, time):
        """Add a flash at ``time`` to the one-second sliding window."""
        self.window.append(time)
        # The window is (time - 1, time]; the microsecond guards float edges.
        while self.window[0] <= time - 1.0 + 1e-6:
            self.window.popleft()
        if len(self.window) > self.best:
            self.best = len(self.window)
            self.best_start = self.window[0]


def flash_report(
    video, *, start=0.0, seconds=None, fps=None, size=(160, 90), series=False
):
    r"""Count WCAG-style luminance flashes per one-second window.

    A transition between two frames counts when at least 25 % of the frame
    changes by a relative luminance of 0.1 or more, and the darker of the two
    states is below 0.8. A flash is a pair of such transitions in opposite
    directions, both within one second of each other. Flashes are counted
    without overlap. The result reports the largest count in any one-second
    sliding window.

    Parameters
    ----------
    video : str or pathlib.Path
        Video file. Only the first video stream is decoded.
    start : float
        Start time in seconds.
    seconds : float or None
        Analysed duration in seconds. ``None`` decodes to the end.
    fps : float or None
        Sampling rate. ``None`` uses the source rate. Lower rates can alias
        fast flashes, so the default keeps the source rate.
    size : tuple of int
        ``(width, height)`` of the decoded luminance frames.
    series : bool
        Keep the per-frame mean relative luminance list in the result.

    Returns
    -------
    dict
        ``max_flashes_per_second``, ``worst_window_start`` (seconds or
        ``None``), ``frames``, ``flash_count``, ``verdict`` (``"PASS"`` when
        fewer than three flashes per second), ``max_changing_area``,
        ``max_luma_step``, ``mean_relative_luminance``, ``red_flash``
        (``"NOT_COMPUTED"``), and the method parameters.

    Examples
    --------
    >>> import os, subprocess, tempfile
    >>> from moviepy.config import FFMPEG_BINARY
    >>> path = os.path.join(tempfile.mkdtemp(), "steady.mkv")
    >>> _ = subprocess.run([FFMPEG_BINARY, "-hide_banner", "-loglevel", "error",
    ...     "-f", "lavfi", "-i", "color=c=gray:s=160x90:r=10:d=1", path])
    >>> flash_report(path)["verdict"]
    'PASS'
    """
    facts = stream_facts(video)
    if facts["video"] is None or not facts["video"]["fps"]:
        raise ValueError(f"no video stream with a frame rate in {video}")
    rate = float(fps) if fps else float(facts["video"]["fps"])
    if rate <= 0:
        raise ValueError("fps must be positive")
    width, height = int(size[0]), int(size[1])
    batch = max(1, min(256, _BUDGET // (width * height * 12)))
    scanner = _FlashScanner(start, rate, series)
    for rgb in _decode_frames(
        video, start=start, seconds=seconds, fps=fps, size=(width, height), batch=batch
    ):
        scanner.feed(_SRGB_LUT[rgb] @ _LUMA)
    if scanner.frames < 2:
        raise ValueError("flash_report needs at least two decoded frames")
    best = scanner.best
    result = {
        "frames": scanner.frames,
        "analysed_seconds": scanner.frames / rate,
        "sampling_fps": rate,
        "size": [width, height],
        "flash_count": scanner.flashes,
        "max_flashes_per_second": best,
        "worst_window_start": float(scanner.best_start) if best else None,
        "max_changing_area": scanner.max_area,
        "max_luma_step": scanner.max_step,
        "mean_relative_luminance": scanner.mean_sum / scanner.frames,
        "verdict": flash_verdict(best),
        "red_flash": "NOT_COMPUTED",
        "method": {
            "min_luma_change": 0.1,
            "darker_state_below": 0.8,
            "area_fraction": 0.25,
            "pair_max_seconds": 1.0,
            "window_seconds": 1.0,
            "limit_per_second": FLASH_LIMIT,
        },
    }
    if series:
        result["mean_luminance_series"] = scanner.means
    return result


def loudness_report(media, *, targets=None):
    """Measure EBU R128 loudness and true peak of the first audio stream.

    Parameters
    ----------
    media : str or pathlib.Path
        Audio or video file.
    targets : dict or None
        Overrides for ``DEFAULT_TARGETS``: ``lufs``, ``lufs_tolerance``
        and ``true_peak_max``.

    Returns
    -------
    dict
        ``integrated_lufs``, ``lra_lu``, ``true_peak_dbtp``,
        ``max_momentary_lufs``, ``max_short_term_lufs``, ``targets``,
        ``checks`` (per-check value, target and verdict) and ``verdict``.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from moviepy.ae.templates._audio_io import AudioWriter
    >>> path = os.path.join(tempfile.mkdtemp(), "tone.wav")
    >>> tone = np.zeros((48000, 2), np.float32)
    >>> with AudioWriter(path, rate=48000) as writer:
    ...     writer.write(tone)
    >>> loudness_report(path)["checks"]["true_peak"]["verdict"]
    'PASS'
    """
    limits = _merge_targets(targets)
    command = [
        FFMPEG_BINARY,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "verbose",
        "-i",
        str(media),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        "ebur128=peak=true:framelog=verbose",
        "-f",
        "null",
        "-",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_FLAGS
    )
    integrated = lra = true_peak = None
    momentary = short = float("-inf")
    in_true_peak = False
    tail = deque(maxlen=40)
    for raw in process.stderr:
        line = raw.decode("utf-8", "replace").rstrip()
        tail.append(line)
        found = _FRAME_LOG.search(line)
        if found:
            momentary = max(momentary, float(found.group(1)))
            short = max(short, float(found.group(2)))
            continue
        text = line.strip()
        if text == "True peak:":
            in_true_peak = True
        elif in_true_peak and _PEAK_LINE.match(text):
            true_peak = float(_PEAK_LINE.match(text).group(1))
            in_true_peak = False
        elif _LUFS_LINE.match(text) and integrated is None:
            integrated = float(_LUFS_LINE.match(text).group(1))
        elif _LRA_LINE.match(text):
            lra = float(_LRA_LINE.match(text).group(1))
    if process.wait() != 0 or integrated is None or true_peak is None:
        raise RuntimeError("FFmpeg ebur128 summary missing: " + "\n".join(tail))
    lufs_ok = abs(integrated - limits["lufs"]) <= limits["lufs_tolerance"]
    tp_ok = true_peak <= limits["true_peak_max"]
    checks = {
        "integrated": {
            "value": _finite(integrated),
            "target": limits["lufs"],
            "tolerance": limits["lufs_tolerance"],
            "verdict": "PASS" if lufs_ok else "FAIL",
        },
        "true_peak": {
            "value": _finite(true_peak),
            "max": limits["true_peak_max"],
            "verdict": "PASS" if tp_ok else "FAIL",
        },
    }
    verdict = "PASS" if lufs_ok and tp_ok else "FAIL"
    return {
        "integrated_lufs": _finite(integrated),
        "lra_lu": _finite(lra),
        "true_peak_dbtp": _finite(true_peak),
        "max_momentary_lufs": _finite(momentary),
        "max_short_term_lufs": _finite(short),
        "targets": limits,
        "checks": checks,
        "verdict": verdict,
    }


def silence_report(media, *, noise_db=-50, min_seconds=2.0):
    """List audio gaps quieter than ``noise_db`` lasting ``min_seconds``.

    Parameters
    ----------
    media : str or pathlib.Path
        Audio or video file.
    noise_db : float
        Level below which audio counts as silent, in dB.
    min_seconds : float
        Shortest silence reported, in seconds.

    Returns
    -------
    dict
        ``gaps`` (start, end, duration in seconds), ``count``,
        ``total_seconds`` and ``verdict``. ``"PASS"`` means no gap was found,
        and ``"REVIEW"`` asks a person to confirm that each gap is intended.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from moviepy.ae.templates._audio_io import AudioWriter
    >>> path = os.path.join(tempfile.mkdtemp(), "tone.wav")
    >>> tone = np.zeros((3 * 48000, 2), np.float32)
    >>> with AudioWriter(path, rate=48000) as writer:
    ...     writer.write(tone)
    >>> silence_report(path)["count"]
    1
    """
    facts = stream_facts(media)
    command = [
        FFMPEG_BINARY,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        str(media),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        f"silencedetect=noise={float(noise_db):g}dB:d={float(min_seconds):g}",
        "-f",
        "null",
        "-",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_FLAGS
    )
    gaps, opened = [], None
    tail = deque(maxlen=40)
    for raw in process.stderr:
        line = raw.decode("utf-8", "replace").rstrip()
        tail.append(line)
        start = _SILENCE_START.search(line)
        end = _SILENCE_END.search(line)
        if start:
            opened = float(start.group(1))
        elif end and opened is not None:
            stop = float(end.group(1))
            gaps.append({"start": opened, "end": stop, "duration": stop - opened})
            opened = None
    if process.wait() != 0:
        raise RuntimeError("FFmpeg silencedetect failed: " + "\n".join(tail))
    open_ended = opened is not None
    if open_ended:
        stop = facts["duration"]
        gaps.append(
            {
                "start": opened,
                "end": stop,
                "duration": None if stop is None else stop - opened,
            }
        )
    total = sum(gap["duration"] or 0.0 for gap in gaps)
    return {
        "noise_db": float(noise_db),
        "min_seconds": float(min_seconds),
        "gaps": gaps,
        "count": len(gaps),
        "total_seconds": total,
        "open_ended_at_end": open_ended,
        "verdict": "REVIEW" if gaps else "PASS",
    }


def clipping_report(media, *, rate=48000, channels=2, threshold=0.999, block=None):
    """Count samples whose magnitude reaches ``threshold``.

    Parameters
    ----------
    media : str or pathlib.Path
        Audio or video file, decoded with
        ``iter_audio`` from ``moviepy.ae.templates._audio_io``.
    rate : int
        Sample rate used for decoding. Resampling can change peaks slightly.
    channels : int
        Channel count used for decoding.
    threshold : float
        Magnitude that counts as clipped (full scale is 1.0).
    block : int or None
        Decoding block length in samples.

    Returns
    -------
    dict
        ``frames``, ``channels``, ``clipped_samples`` (counted per channel
        value), ``peak``, ``peak_dbfs`` and ``verdict``.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from moviepy.ae.templates._audio_io import AudioWriter
    >>> path = os.path.join(tempfile.mkdtemp(), "quiet.wav")
    >>> with AudioWriter(path, rate=48000) as writer:
    ...     writer.write(np.full((4800, 2), 0.5, np.float32))
    >>> clipping_report(path)["clipped_samples"]
    0
    """
    frames = clipped = 0
    peak = 0.0
    for data in iter_audio(media, rate=rate, channels=channels, block=block):
        magnitude = np.abs(data)
        if not np.isfinite(magnitude).all():
            raise ValueError("audio contains non-finite samples")
        clipped += int(np.count_nonzero(magnitude >= threshold))
        peak = max(peak, float(magnitude.max()))
        frames += len(data)
    peak_db = 20.0 * np.log10(peak) if peak > 0 else None
    return {
        "frames": frames,
        "channels": channels,
        "rate": rate,
        "threshold": float(threshold),
        "clipped_samples": clipped,
        "peak": peak,
        "peak_dbfs": _finite(peak_db),
        "verdict": "PASS" if clipped == 0 else "FAIL",
    }


def _write_json(path, value, overwrite):
    """Write evidence JSON, refusing to replace a file unless allowed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with open(path, mode, encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def qa_report(
    video,
    *,
    targets=None,
    output_json=None,
    overwrite=False,
    flash_seconds=None,
    flash_fps=None,
):
    """Run every automated check and optionally write the evidence JSON.

    Parameters
    ----------
    video : str or pathlib.Path
        Media file with optional video and audio streams. Video checks are
        ``NOT_RUN`` for audio-only input.
    targets : dict or None
        Overrides for ``DEFAULT_TARGETS``.
    output_json : str or pathlib.Path or None
        Evidence file to create. It is created exclusively.
    overwrite : bool
        Allow an existing ``output_json`` to be replaced.
    flash_seconds : float or None
        Limit the flash analysis to this many seconds.
    flash_fps : float or None
        Flash sampling rate. ``None`` keeps the source rate.

    Returns
    -------
    dict
        ``stream`` facts, every check, ``verdicts``, ``overall`` (``"FAIL"``
        if any check fails, ``"REVIEW"`` if any needs a person, otherwise
        ``"PASS"``), and ``human_listening`` and ``human_visual`` set to
        ``"NOT_RUN"``.

    Examples
    --------
    >>> import os, tempfile
    >>> import numpy as np
    >>> from moviepy.ae.templates._audio_io import AudioWriter
    >>> folder = tempfile.mkdtemp()
    >>> path = os.path.join(folder, "tone.wav")
    >>> with AudioWriter(path, rate=48000) as writer:
    ...     writer.write(np.zeros((4800, 2), np.float32))
    >>> qa_report(path)["human_listening"]
    'NOT_RUN'
    """
    path = Path(video)
    if not path.is_file():
        raise FileNotFoundError(path)
    out = Path(output_json) if output_json is not None else None
    if out is not None and out.exists() and not overwrite:
        raise FileExistsError(out)
    limits = _merge_targets(targets)
    facts = stream_facts(path)
    flash = None
    if facts["video"] is not None:
        flash = flash_report(path, seconds=flash_seconds, fps=flash_fps)
    loudness = loudness_report(path, targets=limits)
    silence = silence_report(path)
    clipping = clipping_report(path)
    verdicts = {
        "flash": flash["verdict"] if flash else "NOT_RUN",
        "loudness": loudness["verdict"],
        "silence": silence["verdict"],
        "clipping": clipping["verdict"],
    }
    values = set(verdicts.values())
    overall = "FAIL" if "FAIL" in values else "REVIEW" if "REVIEW" in values else "PASS"
    result = {
        "schema": "music_qa/1",
        "media": {"path": str(path.resolve()), "bytes": path.stat().st_size},
        "stream": facts,
        "targets": limits,
        "flash": flash,
        "loudness": loudness,
        "silence": silence,
        "clipping": clipping,
        "verdicts": verdicts,
        "overall": overall,
        "human_listening": "NOT_RUN",
        "human_visual": "NOT_RUN",
        "automated_only": True,
        "limitations": [
            "Flash check approximates WCAG 2.3.1; it is not a certified Harding test.",
            "Red flash is not computed.",
            "Frames are area-scaled to a small size; sub-pixel flashes can be missed.",
            "Loudness uses FFmpeg ebur128 (EBU R128); it is not the channel loudnorm.",
            "Silence REVIEW means a person must confirm that each gap is intended.",
            "Human listening and visual QA remain NOT_RUN.",
        ],
    }
    if out is not None:
        _write_json(out, result, overwrite)
    return result
