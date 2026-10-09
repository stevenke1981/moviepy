"""Final long-form renderer for music videos: looped picture, overlay, audio.

A 87-minute to 10-hour video must not push every background frame through
After Effects. AE renders only (a) a short seamless loop or scene cycle, once
(see ``moviepy.ae.templates.visual_loop``), and (b) a small moving overlay
region, such as the spectrum bars. FFmpeg repeats the background with
``-stream_loop -1`` (or ``-loop 1`` for a still), composites the overlay,
burns an ASS subtitle file, encodes the audio as AAC 48 kHz and attaches
chapters, in one pass.

The conventions are those of the music channel's ``s14-render.py``: outputs are
exclusive creates, the frame count and duration are exact (``-frames:v`` and
``-t``), the duration defaults to the exact audio length and must be
frame-aligned, and every render writes ``<output>.json`` evidence and
``<output>.ffmpeg.log``. Human listening and visual checks stay ``NOT_RUN``.

Examples
--------
>>> from moviepy.ae.templates.music_render import frame_count
>>> frame_count(5220.0, 24)
125280
"""

import hashlib
import json
import math
import os
import re
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from moviepy.ae.parallel import detect_encoder, encoder_args, iter_frames_parallel
from moviepy.ae.templates._audio_io import audio_info
from moviepy.config import FFMPEG_BINARY


__all__ = ["frame_count", "render_music_video", "preview_window"]

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_RATE = 48000


def frame_count(seconds, fps, *, tolerance=1e-6):
    """Return ``seconds * fps`` as an exact integer or raise ``ValueError``.

    Parameters
    ----------
    seconds : float
        Duration in seconds.
    fps : float
        Frame rate.
    tolerance : float
        Largest allowed distance from a whole number of frames.

    Examples
    --------
    >>> frame_count(2.5, 12)
    30
    >>> frame_count(2.51, 12)
    Traceback (most recent call last):
    ...
    ValueError: 2.51 s is not frame-aligned at 12 fps
    """
    exact = float(seconds) * float(fps)
    count = int(round(exact))
    if count < 1 or abs(exact - count) > tolerance:
        raise ValueError(f"{seconds:g} s is not frame-aligned at {fps:g} fps")
    return count


# --------------------------------------------------------------------------- #
# overlay frames as RGBA through the shared-memory iterator
# --------------------------------------------------------------------------- #


class _PackedRGBA:
    """Present an RGBA clip as a 3-channel clip for ``iter_frames_parallel``.

    ``iter_frames_parallel`` moves uint8 ``(h, w, 3)`` frames through shared
    memory. An ``(h, w, 4)`` frame holds ``4hw`` bytes, so it is stored in a
    ``(ceil(4h / 3), w, 3)`` array with a few unused trailing bytes and
    unpacked again by ``_unpack``.
    """

    def __init__(self, clip):
        self.clip = clip
        width, height = (int(v) for v in clip.size)
        self.shape = (height, width, 4)
        self.rows = -(-4 * height // 3)
        self.size = (width, self.rows)

    def get_frame(self, t):
        """Return the RGBA frame at ``t`` packed as ``(rows, w, 3)``."""
        rgba = _rgba_frame(self.clip, t)
        if rgba.shape != self.shape:
            raise ValueError(f"overlay frame has shape {rgba.shape}, not {self.shape}")
        packed = np.zeros(self.rows * self.shape[1] * 3, np.uint8)
        packed[: rgba.size] = rgba.reshape(-1)
        return packed.reshape(self.rows, self.shape[1], 3)

    def close(self):
        """Close the wrapped clip."""
        close = getattr(self.clip, "close", None)
        if close is not None:
            close()


def _rgba_frame(clip, t):
    """Return the straight-alpha uint8 ``(h, w, 4)`` frame of ``clip`` at ``t``.

    An AE composition is read through ``render_buffer`` (colour with
    ``to_uint8_rgb`` and alpha from the buffer, as the channel's spectrum step
    does); any other clip uses its frame and optional mask.
    """
    render = getattr(clip, "render_buffer", None)
    if render is not None:
        buffer = render(float(t))
        rgb = buffer.to_uint8_rgb()
        out = np.empty((*rgb.shape[:2], 4), np.uint8)
        out[..., :3] = rgb
        out[..., 3] = np.rint(np.clip(buffer.rgba[..., 3], 0.0, 1.0) * 255).astype(
            np.uint8
        )
        return out
    rgb = np.asarray(clip.get_frame(float(t)))
    out = np.empty((*rgb.shape[:2], 4), np.uint8)
    out[..., :3] = rgb[..., :3]
    mask = getattr(clip, "mask", None)
    if mask is None:
        out[..., 3] = 255
    else:
        alpha = np.asarray(mask.get_frame(float(t)), np.float64)
        out[..., 3] = np.rint(np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    return out


def _packed_factory(factory, args):
    """Build the overlay clip in a worker and wrap it for transport."""
    return _PackedRGBA(factory(*args))


def _unpack(frame, shape):
    flat = np.asarray(frame).reshape(-1)
    return flat[: shape[0] * shape[1] * 4].reshape(shape)


def _parse_overlay(overlay):
    """Return ``(factory, args, clip)``; exactly one of factory or clip is set."""
    if overlay is None:
        return None
    if isinstance(overlay, (tuple, list)):
        if len(overlay) != 2 or not callable(overlay[0]):
            raise TypeError("overlay tuple must be (picklable callable, args)")
        return overlay[0], tuple(overlay[1]), None
    if hasattr(overlay, "get_frame") or hasattr(overlay, "render_buffer"):
        return None, (), overlay
    if callable(overlay):
        return overlay, (), None
    raise TypeError("overlay must be (factory, args), a callable or a clip")


# --------------------------------------------------------------------------- #
# FFmpeg helpers
# --------------------------------------------------------------------------- #


def _looped_frames(path, fps):
    """Count the frames of video ``path`` after conversion to ``fps``."""
    run = subprocess.run(
        [FFMPEG_BINARY, "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0",
         "-vf", f"fps={Fraction(str(fps)).limit_denominator(1000000)}",
         "-f", "null", "-"],
        capture_output=True,
        text=True,
        creationflags=_NO_WINDOW,
    )  # fmt: skip
    found = re.findall(r"frame=\s*(\d+)", run.stderr)
    if run.returncode != 0 or not found:
        raise RuntimeError(f"cannot count frames of {path}:\n{run.stderr[-500:]}")
    return int(found[-1])


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_ass(ass):
    ass = Path(ass).resolve()
    if not ass.is_file():
        raise FileNotFoundError(ass)
    if not _SAFE_NAME.match(ass.name):
        raise ValueError(
            "ASS must have a simple filename (letters, digits, '_', '-', '.')"
        )
    return ass


def _build_command(
    *, plan, output, encoder, quality, audio_bitrate, metadata, has_chapters
):
    """Assemble the FFmpeg command for one render attempt."""
    fps = plan["fps"]
    rate = Fraction(str(fps)).limit_denominator(1000000)
    ratio = f"{rate.numerator}/{rate.denominator}"
    width, height = plan["size"]
    command = [FFMPEG_BINARY, "-y", "-hide_banner", "-nostdin", "-loglevel", "warning"]
    index = 0
    overlay_index = background_index = audio_index = chapter_index = None
    if plan["overlay"] is not None:
        ow, oh = plan["overlay"]["size"]
        command += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{ow}x{oh}",
                    "-r", ratio, "-i", "pipe:0"]  # fmt: skip
        overlay_index, index = index, index + 1
    if plan["background_kind"] == "video":
        command += ["-stream_loop", "-1"]
    else:
        command += ["-loop", "1", "-framerate", ratio]
    command += ["-i", plan["background"]]
    background_index, index = index, index + 1
    if plan["audio"] is not None:
        command += ["-i", plan["audio"]]
        audio_index, index = index, index + 1
    if has_chapters:
        command += ["-f", "ffmetadata", "-i", plan["chapters"]]
        chapter_index, index = index, index + 1

    chain = [f"fps={ratio}"]
    if plan["trim_frames"]:
        chain.append(f"trim=start_frame={plan['trim_frames']}")
        chain.append("setpts=PTS-STARTPTS")
    chain += [
        f"scale={width}:{height}:force_original_aspect_ratio=increase",
        f"crop={width}:{height}",
        "setsar=1",
    ]
    graph = f"[{background_index}:v:0]" + ",".join(chain) + "[bg]"
    top = "bg"
    if overlay_index is not None:
        x, y = plan["overlay_position"]
        graph += f";[bg][{overlay_index}:v:0]overlay={x}:{y}:shortest=1:format=auto[ov]"
        top = "ov"
    tail = []
    if plan["ass"] is not None:
        if plan["start"]:
            # Subtitle events carry global times; give the frames global PTS.
            tail.append(f"setpts=PTS+{plan['start']:.6f}/TB")
        tail.append(f"ass=filename={plan['ass_name']}")
        if plan["start"]:
            tail.append("setpts=PTS-STARTPTS")
    tail.append("format=yuv420p")
    graph += f";[{top}]" + ",".join(tail) + "[v]"
    if audio_index is not None:
        first = int(round(plan["start"] * _RATE))
        last = first + int(round(plan["seconds"] * _RATE))
        graph += (
            f";[{audio_index}:a:0]aresample={_RATE},"
            f"atrim=start_sample={first}:end_sample={last},"
            "asetpts=PTS-STARTPTS[a]"
        )
    command += ["-filter_complex", graph, "-map", "[v]"]
    if audio_index is not None:
        command += ["-map", "[a]"]
    command += [
        "-map_metadata", str(chapter_index) if has_chapters else "-1",
        "-map_chapters", str(chapter_index) if has_chapters else "-1",
    ]  # fmt: skip
    for key, value in sorted((metadata or {}).items()):
        command += ["-metadata", f"{key}={value}"]
    command += encoder_args(encoder, quality)
    command += ["-r", ratio, "-frames:v", str(plan["frames"]),
                "-t", f"{plan['frames'] / fps:.6f}"]  # fmt: skip
    if audio_index is not None:
        command += ["-c:a", "aac", "-b:a", str(audio_bitrate), "-ar", str(_RATE)]
    else:
        command += ["-an"]
    command.append(plan["output"])
    return command


def _run(command, cwd, log_path, mode, frames, count):
    """Run FFmpeg, feeding ``frames`` (or nothing) on stdin; return success."""
    with open(log_path, mode, encoding="utf-8") as log:
        log.write("$ " + " ".join(map(str, command)) + "\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.PIPE if frames is not None else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            creationflags=_NO_WINDOW,
        )
        failed = False
        try:
            if frames is not None:
                try:
                    done = 0
                    for frame in frames:
                        proc.stdin.write(np.ascontiguousarray(frame).data)
                        done += 1
                    if done != count:
                        failed = True
                except (BrokenPipeError, OSError):
                    failed = True
                finally:
                    try:
                        proc.stdin.close()
                    except OSError:
                        failed = True
            code = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    return code == 0 and not failed


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def render_music_video(
    output,
    *,
    background,
    audio,
    duration=None,
    overlay=None,
    overlay_position=(0, 0),
    ass=None,
    chapters=None,
    fps=24,
    size=(1280, 720),
    encoder="auto",
    quality="high",
    workers=None,
    start=0.0,
    audio_bitrate="256k",
    overwrite=False,
    metadata=None,
    loop_frames=None,
):
    """Render a long music video in one FFmpeg pass.

    Parameters
    ----------
    output : str or os.PathLike
        Output ``.mp4``. ``<output>.json`` (evidence) and
        ``<output>.ffmpeg.log`` are written next to it. Existing files raise
        ``FileExistsError`` unless ``overwrite`` is true.
    background : str or os.PathLike
        A seamless looped video (repeated with ``-stream_loop -1``; export it
        with ``visual_loop.export_loop``) or a still image (``-loop 1``). It is
        cover-fitted to ``size``.
    audio : str or os.PathLike or None
        Audio file, encoded as AAC 48 kHz. ``None`` renders without audio and
        then ``duration`` is required.
    duration : float, optional
        Seconds to render. Defaults to the exact audio duration after
        ``start``. It must be frame-aligned at ``fps``.
    overlay : tuple, callable, Composition or None
        What AE renders on top: ``(factory, args)`` with a picklable top-level
        factory returning a composition or clip (rendered by
        ``workers`` processes as RGBA with alpha), a bare factory callable, or
        a ready composition (rendered in this process). Overlay time is the
        global time ``start + i / fps``.
    overlay_position : tuple of int
        ``(x, y)`` of the overlay's top-left corner on the picture.
    ass : str or os.PathLike, optional
        ASS subtitles burned with the ``ass`` filter. FFmpeg runs with the
        file's folder as working directory and a validated plain filename, so
        Windows drive letters never reach the filter graph.
    chapters : str or os.PathLike, optional
        FFmetadata file mapped as chapters and global metadata. Only for
        ``start == 0``, where its times are the video's times.
    fps : float
        Output frame rate.
    size : tuple of int
        Output ``(width, height)``.
    encoder : str
        ``"auto"``, ``"libx264"``, ``"h264_nvenc"``, ``"hevc_nvenc"`` or
        ``"libx265"``; arguments come from ``parallel.encoder_args``.
    quality : str
        ``"high"``, ``"balanced"`` or ``"draft"``.
    workers : int, optional
        Overlay render processes (``None`` means ``parallel.default_workers``).
    start : float
        Global start in seconds; the audio is cut with ``atrim`` and a video
        background loop is advanced to the same place.
    audio_bitrate : str
        AAC bitrate, for example ``"256k"``.
    overwrite : bool
        Replace existing output and evidence files.
    metadata : dict, optional
        Extra global ``key=value`` tags.
    loop_frames : int, optional
        Frames in one background loop at ``fps``; measured by decoding when
        ``start`` is not zero and this is omitted.

    Returns
    -------
    dict
        The evidence written to ``<output>.json``: ``command``, ``frames``,
        ``seconds``, ``start``, ``encoder``, ``overlay`` info, ``elapsed``,
        ``output_sha256`` and ``human_visual`` / ``human_listening`` set to
        ``"NOT_RUN"``.

    Raises
    ------
    FileExistsError
        If an output or evidence file exists and ``overwrite`` is false.
    ValueError
        For a duration that is not frame-aligned or exceeds the audio.
    RuntimeError
        If FFmpeg fails (the log tail is included, the partial output removed).

    Examples
    --------
    >>> import inspect
    >>> inspect.signature(render_music_video).parameters["audio_bitrate"].default
    '256k'
    """
    output = Path(output).resolve()
    evidence_path = Path(str(output) + ".json")
    log_path = Path(str(output) + ".ffmpeg.log")
    if not overwrite:
        for path in (output, evidence_path, log_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists; preserve it")
    fps = float(fps)
    width, height = (int(v) for v in size)
    start = float(start)
    if not (math.isfinite(fps) and fps > 0 and math.isfinite(start) and start >= 0):
        raise ValueError("fps must be positive and start non-negative")
    if start and chapters:
        raise ValueError("chapters belong only on a start=0 render")
    if start:
        frame_count(start, fps)

    background = Path(background).resolve()
    if not background.is_file():
        raise FileNotFoundError(background)
    kind = "image" if background.suffix.lower() in _IMAGE_SUFFIXES else "video"

    audio_path = None
    info = None
    if audio is not None:
        audio_path = Path(audio).resolve()
        info = audio_info(audio_path)
        if duration is None:
            duration = info["seconds"] - start
    if duration is None:
        raise ValueError("duration is required without audio")
    duration = float(duration)
    frames = frame_count(duration, fps)
    if info is not None and start + duration > info["seconds"] + 0.5 / info["rate"]:
        raise ValueError(
            f"start + duration ({start + duration:g}s) exceeds the audio "
            f"({info['seconds']:g}s)"
        )

    ass_path = _check_ass(ass) if ass is not None else None
    chapter_path = None
    if chapters is not None:
        chapter_path = Path(chapters).resolve()
        if not chapter_path.is_file():
            raise FileNotFoundError(chapter_path)

    parsed = _parse_overlay(overlay)
    overlay_plan = None
    probe = None
    if parsed is not None:
        factory, factory_args, clip = parsed
        probe = clip if clip is not None else factory(*factory_args)
        overlay_plan = {
            "size": [int(probe.size[0]), int(probe.size[1])],
            "kind": "clip" if clip is not None else "factory",
        }
        if clip is None:
            close = getattr(probe, "close", None)
            if close is not None:
                close()
            probe = None
    x, y = (int(v) for v in overlay_position)

    trim = 0
    period = None
    if kind == "video" and start:
        period = loop_frames or _looped_frames(background, fps)
        trim = int(round(start * fps)) % int(period)
    plan = {
        "fps": fps,
        "size": (width, height),
        "background": str(background),
        "background_kind": kind,
        "audio": None if audio_path is None else str(audio_path),
        "overlay": overlay_plan,
        "overlay_position": (x, y),
        "ass": None if ass_path is None else str(ass_path),
        "ass_name": None if ass_path is None else ass_path.name,
        "chapters": None if chapter_path is None else str(chapter_path),
        "start": start,
        "seconds": frames / fps,
        "frames": frames,
        "trim_frames": trim,
        "output": str(output),
    }

    chosen = detect_encoder(encoder)
    output.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    mode = "w" if overwrite else "x"
    for attempt in range(2):
        command = _build_command(
            plan=plan,
            output=output,
            encoder=chosen,
            quality=quality,
            audio_bitrate=audio_bitrate,
            metadata=metadata,
            has_chapters=chapter_path is not None,
        )
        frames_iter = None
        shape = None
        if overlay_plan is not None:
            ow, oh = overlay_plan["size"]
            shape = (oh, ow, 4)
            times = [start + i / fps for i in range(frames)]
            if parsed[2] is not None:
                frames_iter = iter_frames_parallel(
                    None, (), times, workers=1,
                    size=(ow, -(-4 * oh // 3)), clip=_PackedRGBA(parsed[2]),
                )  # fmt: skip
            else:
                frames_iter = iter_frames_parallel(
                    _packed_factory, (parsed[0], parsed[1]), times,
                    workers=workers, size=(ow, -(-4 * oh // 3)),
                )  # fmt: skip
            frames_iter = (_unpack(f, shape) for f in frames_iter)
        try:
            good = _run(
                command,
                ass_path.parent if ass_path is not None else None,
                log_path,
                mode,
                frames_iter,
                frames,
            )
        finally:
            if frames_iter is not None:
                frames_iter.close()
        if good:
            break
        if output.exists():
            output.unlink()
        if attempt == 0 and encoder == "auto" and chosen != "libx264":
            chosen = "libx264"
            mode = "w"
            continue
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"ffmpeg ({chosen}) failed:\n{tail}")
    elapsed = time.perf_counter() - began

    if overlay_plan is not None:
        from moviepy.ae.parallel import default_workers

        overlay_plan = dict(
            overlay_plan,
            position=[x, y],
            frames=frames,
            alpha=True,
            workers=(
                1
                if parsed[2] is not None
                else min(frames, default_workers() if workers is None else workers)
            ),
        )
    report = {
        "output": str(output),
        "command": [str(c) for c in command],
        "frames": frames,
        "seconds": frames / fps,
        "start": start,
        "fps": fps,
        "size": [width, height],
        "encoder": chosen,
        "quality": quality,
        "background": {
            "path": str(background),
            "kind": kind,
            "trim_frames": trim,
            "loop_frames": period,
        },
        "audio": (
            None
            if audio_path is None
            else {
                "path": str(audio_path),
                "encoded": True,
                "codec": "aac",
                "bitrate": str(audio_bitrate),
                "rate": _RATE,
                "source_frames": info["frames"],
                "source_rate": info["rate"],
            }
        ),
        "overlay": overlay_plan,
        "ass": plan["ass"],
        "chapters": plan["chapters"],
        "metadata": dict(metadata or {}),
        "elapsed": elapsed,
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "log": str(log_path),
        "human_listening": "NOT_RUN",
        "human_visual": "NOT_RUN",
    }
    with open(evidence_path, "w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def preview_window(output, *, background, audio, start=0.0, seconds=60.0, **options):
    """Render the window ``[start, start + seconds)`` of a music video.

    The channel makes 60 s previews before each full render. Audio is cut
    exactly with ``atrim`` (sample accurate), the background loop is advanced
    to the same position, overlay times and ASS events are global.

    Parameters
    ----------
    output : str or os.PathLike
        Preview ``.mp4``; evidence files are written next to it.
    background : str or os.PathLike
        Looped video or still image, as in ``render_music_video``.
    audio : str or os.PathLike
        Full-length audio file; only the window is encoded.
    start : float
        Window start in seconds (frame-aligned).
    seconds : float
        Window length in seconds (frame-aligned).
    ``**options``
        Any other ``render_music_video`` keyword except ``duration``.

    Returns
    -------
    dict
        The evidence of ``render_music_video``.

    Examples
    --------
    >>> import inspect
    >>> inspect.signature(preview_window).parameters["seconds"].default
    60.0
    """
    if "duration" in options:
        raise TypeError("preview_window takes seconds, not duration")
    return render_music_video(
        output,
        background=background,
        audio=audio,
        start=start,
        duration=seconds,
        **options,
    )
