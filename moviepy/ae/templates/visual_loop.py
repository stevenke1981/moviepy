"""Seamless picture loops and scene cycles for long music videos.

An 87-minute (or 10-hour) video never pushes every background frame through
After Effects. AE renders one short seamless *loop* (or one *cycle* of several
loops) exactly once with these helpers; FFmpeg then repeats the exported file
with ``-stream_loop -1`` (see ``moviepy.ae.templates.music_render``).

``seamless_loop`` is the AE twin of the music channel's ``visual()`` step: take
``length`` seconds of footage, cross-dissolve the tail into the head and keep
``length - crossfade`` seconds, so the last frame flows into the first.
``scene_cycle`` plays several loops one after another, ``segment`` seconds each,
with a dissolve between neighbours and from the last scene back to the first,
so the whole cycle is itself seamless. ``export_loop`` writes the result as an
MP4 that is safe to stream-copy and loop: constant frame rate, a closed GOP of
one second and no B-frames.

Examples
--------
>>> import numpy as np
>>> from moviepy import VideoClip
>>> from moviepy.ae.templates.visual_loop import seamless_loop
>>> clip = VideoClip(lambda t: np.full((18, 32, 3), int(t * 10), np.uint8),
...                  duration=6.0)
>>> clip.fps = 12
>>> comp = seamless_loop(clip, length=5.0, crossfade=1.0, fps=12, size=(32, 18))
>>> comp.duration, comp.size
(4.0, (32, 18))
"""

import hashlib
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from moviepy import VideoClip, VideoFileClip
from moviepy.ae.composition import Composition
from moviepy.ae.layers import AVLayer
from moviepy.ae.parallel import detect_encoder, encoder_args, iter_frames_parallel
from moviepy.ae.properties.keyframe import Keyframe
from moviepy.ae.transform import Transform
from moviepy.config import FFMPEG_BINARY


__all__ = ["seamless_loop", "scene_cycle", "export_loop"]

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _number(value, name, low=0.0, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or (value <= low if strict else value < low):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {low}")
    return value


def _size(size):
    try:
        width, height = (int(v) for v in size)
    except (TypeError, ValueError) as error:
        raise ValueError("size must be (width, height)") from error
    if width < 2 or height < 2:
        raise ValueError("size must be at least 2x2")
    return width, height


def _as_clip(source):
    """Return a video clip for a path or an existing clip."""
    if isinstance(source, (str, os.PathLike)):
        if not Path(source).is_file():
            raise FileNotFoundError(source)
        return VideoFileClip(str(source), audio=False)
    if not callable(getattr(source, "get_frame", None)):
        raise TypeError("source must be a video path or a clip")
    return source


def _cover(clip, size):
    """Return a ``Transform`` that scales ``clip`` to cover ``size`` centred."""
    sw, sh = (int(v) for v in clip.size)
    width, height = size
    percent = max(width / sw, height / sh) * 100.0
    return dict(
        scale=(percent, percent),
        position=((width - 1) / 2.0, (height - 1) / 2.0),
    )


def _fade_in(begin, length):
    """Opacity keyframes (layer clock) fading 0 to 100 over ``length``."""
    return [Keyframe(float(begin), 0.0), Keyframe(float(begin + length), 100.0)]


class _Wrapped(VideoClip):
    """A clip that repeats ``clip`` forever (frame function based, no copy)."""

    def __init__(self, clip, duration, fps):
        period = float(clip.duration)
        step = 1.0 / fps

        def frame(t):
            local = float(t) % period
            if local > period - step * 1e-3:
                local = 0.0
            return clip.get_frame(local)

        super().__init__(frame, duration=duration)
        self.size = tuple(int(v) for v in clip.size)
        self.fps = fps
        self._inner = clip

    def close(self):
        """Close nothing: the wrapped clip is owned by the caller."""


def seamless_loop(
    source, *, length=8.0, crossfade=2.0, fps=24, size=(1280, 720), start=0.0
):
    """Return a composition whose last frame flows into its first.

    Take ``[start, start + length)`` of ``source`` and cross-dissolve the tail
    into the head, as the music channel's ``visual()`` step does with FFmpeg's
    ``xfade``: the result lasts ``length - crossfade`` seconds, plays
    ``source[start + crossfade ...]`` first and, during its last ``crossfade``
    seconds, dissolves into ``source[start ...]``, which is exactly the clip
    that continues at the loop point.

    Parameters
    ----------
    source : str, os.PathLike or VideoClip
        Video path or clip; it must last at least ``start + length`` seconds.
    length : float
        Seconds of footage used (default 8, the Flow clip length).
    crossfade : float
        Dissolve length in seconds; must be shorter than ``length / 2``.
    fps : float
        Composition frame rate.
    size : tuple of int
        Output ``(width, height)``; the footage is cover-fitted and centred.
    start : float
        Where in the source the used span begins.

    Returns
    -------
    Composition
        Duration ``length - crossfade``; two ``AVLayer`` layers (head on top).

    Raises
    ------
    ValueError
        For impossible timing or a source shorter than ``start + length``.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy import VideoClip
    >>> clip = VideoClip(lambda t: np.zeros((4, 8, 3), np.uint8), duration=3.0)
    >>> clip.fps = 10
    >>> seamless_loop(clip, length=2.0, crossfade=0.5, fps=10, size=(8, 4)).duration
    1.5
    """
    length = _number(length, "length", 0.0, True)
    crossfade = _number(crossfade, "crossfade", 0.0, True)
    start = _number(start, "start")
    fps = _number(fps, "fps", 0.0, True)
    size = _size(size)
    if crossfade * 2 >= length:
        raise ValueError("crossfade must be shorter than half of length")
    clip = _as_clip(source)
    duration = clip.duration
    if duration is None or start + length > float(duration) + 1e-6:
        raise ValueError(
            f"source lasts {duration}s but start + length needs {start + length}s"
        )
    out = length - crossfade
    # ``AVLayer.start_time`` places source time 0 on the composition clock, so
    # shifting by ``start`` selects the used span without copying the clip.
    comp = Composition(size=size, fps=fps, duration=out, name="seamless_loop")
    fit = _cover(clip, size)
    tail = AVLayer(
        clip,
        "tail",
        start_time=-(crossfade + start),
        in_point=0.0,
        out_point=out,
        transform=Transform(**fit),
    )
    head = AVLayer(
        clip,
        "head",
        start_time=out - crossfade - start,
        in_point=out - crossfade,
        out_point=out,
        # Opacity keys run on the layer clock, whose zero sits at ``start_time``:
        # ``start`` seconds before the head's in point.
        transform=Transform(**fit, opacity=_fade_in(start, crossfade)),
    )
    comp.add_layer(tail)
    comp.add_layer(head)
    return comp


def scene_cycle(loops, *, segment=120.0, crossfade=2.0, fps=24, size=(1280, 720)):
    """Return a seamless cycle that plays every loop for ``segment`` seconds.

    Mirrors the music channel's 480 s cycle (four 8 s loops, 120 s each, 2 s
    dissolves): scene ``i`` fades in during the ``crossfade`` seconds before
    ``i * segment``, and the last scene fades back into the first, so the
    cycle can be repeated without a visible seam.

    Parameters
    ----------
    loops : sequence
        One or more seamless loops: video paths, clips or compositions (for
        example the result of ``seamless_loop`` or an exported loop file).
        Each is repeated for as long as its scene lasts.
    segment : float
        Seconds each scene occupies in the cycle.
    crossfade : float
        Dissolve length in seconds; must be shorter than ``segment``.
    fps : float
        Frame rate of the composition.
    size : tuple of int
        Output size; every loop is cover-fitted to it.

    Returns
    -------
    Composition
        Duration ``len(loops) * segment``.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy import VideoClip
    >>> clip = VideoClip(lambda t: np.zeros((4, 8, 3), np.uint8), duration=1.0)
    >>> clip.fps = 10
    >>> cycle = scene_cycle([clip, clip], segment=4.0, crossfade=1.0,
    ...                     fps=10, size=(8, 4))
    >>> cycle.duration, len(cycle.layers)
    (8.0, 3)
    """
    loops = list(loops)
    if not loops:
        raise ValueError("loops must not be empty")
    segment = _number(segment, "segment", 0.0, True)
    crossfade = _number(crossfade, "crossfade", 0.0, True)
    fps = _number(fps, "fps", 0.0, True)
    size = _size(size)
    if crossfade >= segment:
        raise ValueError("crossfade must be shorter than segment")
    clips = [_as_clip(item) for item in loops]
    total = len(clips) * segment
    comp = Composition(size=size, fps=fps, duration=total, name="scene_cycle")

    def scene(index, clip, begin, length, fade):
        # Loop time 0 sits at ``begin``; scene 0 starts ``crossfade`` in, as
        # the source's trim=start=crossfade does.
        wrapped = _Wrapped(clip, length, fps)
        transform = Transform(
            **_cover(wrapped, size),
            opacity=_fade_in(0.0, crossfade) if fade else 100.0,
        )
        layer = AVLayer(
            wrapped,
            f"scene {index}",
            start_time=begin,
            in_point=max(begin, 0.0),
            out_point=min(begin + length, total),
            transform=transform,
        )
        comp.add_layer(layer)

    for index, clip in enumerate(clips):
        begin = index * segment - crossfade
        scene(index, clip, begin, segment + crossfade, fade=index > 0)
    # Closing dissolve: scene 0 again, entering during the last crossfade so the
    # cycle's last frame is followed by its own first frame.
    scene(0, clips[0], total - crossfade, crossfade, fade=True)
    return comp


def _loop_command(output, fps, size, encoder, quality, frames):
    gop = max(1, int(round(fps)))
    args = encoder_args(encoder, quality)
    # Loops are concatenated by stream copy: no B-frames and a closed GOP.
    while "-bf" in args:
        at = args.index("-bf")
        del args[at : at + 2]
    args += ["-bf", "0", "-g", str(gop), "-keyint_min", str(gop)]
    if encoder in ("libx264", "libx265"):
        args += ["-sc_threshold", "0"]
        if encoder == "libx264":
            args += ["-x264-params", "open-gop=0"]
        else:
            args += ["-x265-params", "open-gop=0:scenecut=0"]
    else:
        args += ["-forced-idr", "1", "-no-scenecut", "1"]
    return [
        FFMPEG_BINARY, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
        "-r", repr(float(fps)), "-i", "-", "-map", "0:v:0", "-an", *args,
        "-r", repr(float(fps)), "-fps_mode", "cfr", "-frames:v", str(frames),
        str(output),
    ]  # fmt: skip


def _pipe(command, frames, digest, stats):
    """Feed frames to FFmpeg, hashing them and tracking seam statistics."""
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
            creationflags=_NO_WINDOW,
        )
        failed = False
        try:
            try:
                for frame in frames:
                    frame = np.ascontiguousarray(frame)
                    digest.update(frame.tobytes())
                    if stats["first"] is None:
                        stats["first"] = frame.copy()
                    if stats["previous"] is not None:
                        diff = np.abs(
                            frame.astype(np.int16) - stats["previous"].astype(np.int16)
                        ).mean()
                        stats["steps"].append(float(diff))
                    stats["previous"] = frame.copy()
                    proc.stdin.write(frame.data)
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
        if code != 0 or failed:
            log.seek(0)
            return False, log.read().decode("utf-8", "replace")[-2000:]
    return True, ""


def export_loop(
    comp, path, *, encoder="auto", quality="high", overwrite=False, workers=None
):
    """Write ``comp`` as an MP4 that ``-stream_loop -1`` can repeat.

    The file has a constant frame rate, yuv420p, a closed GOP of one second
    (``-g fps``, no scene-cut keyframes) and no B-frames, exactly the recipe of
    the music channel's shared loop, so loops cut and concatenate by stream
    copy.

    Parameters
    ----------
    comp : Composition or tuple
        The composition to render, or ``(factory, args)`` of a picklable
        top-level factory returning one (needed for ``workers > 1``; a
        composition itself is never pickled).
    path : str or os.PathLike
        Output ``.mp4``.
    encoder : str
        ``"auto"`` (NVENC when a real test encode works, else ``libx264``) or
        an explicit encoder name.
    quality : str
        ``"high"``, ``"balanced"`` or ``"draft"``.
    overwrite : bool
        Replace an existing file; by default an existing file raises.
    workers : int, optional
        Render processes; ``None`` renders sequentially for a composition and
        uses all cores for a factory tuple.

    Returns
    -------
    dict
        Evidence: ``path``, ``frames``, ``fps``, ``size``, ``seconds``,
        ``encoder``, ``quality``, ``gop``, ``frames_sha256`` (hash of the raw
        frames), ``seam`` (first/last frame difference against the median step
        difference), ``command`` and ``human_visual`` set to ``"NOT_RUN"``.

    Raises
    ------
    FileExistsError
        If ``path`` exists and ``overwrite`` is false.
    RuntimeError
        If FFmpeg fails.

    Examples
    --------
    >>> import inspect
    >>> inspect.signature(export_loop).parameters["quality"].default
    'high'
    """
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists")
    if isinstance(comp, tuple):
        factory, args = comp
        probe = factory(*args)
        own = True
    else:
        factory, args, probe, own = None, (), comp, False
    try:
        fps = float(probe.fps)
        size = (int(probe.size[0]), int(probe.size[1]))
        frames = int(round(float(probe.duration) * fps))
    finally:
        if own:
            probe.close()
    if frames < 2:
        raise ValueError("a loop needs at least two frames")
    if workers is None:
        workers = 1 if factory is None else None
    if factory is None and workers not in (None, 1):
        raise ValueError("workers > 1 needs a (factory, args) tuple")
    chosen = detect_encoder(encoder)
    output.parent.mkdir(parents=True, exist_ok=True)
    times = [i / fps for i in range(frames)]
    began = time.perf_counter()
    for attempt in range(2):
        digest = hashlib.sha256()
        stats = {"first": None, "previous": None, "steps": []}
        command = _loop_command(output, fps, size, chosen, quality, frames)
        if factory is None:
            iterator = iter_frames_parallel(
                None, (), times, workers=1, size=size, clip=probe
            )
        else:
            iterator = iter_frames_parallel(
                factory, args, times, workers=workers, size=size
            )
        try:
            good, tail = _pipe(command, iterator, digest, stats)
        finally:
            iterator.close()
        if good:
            break
        if attempt == 0 and encoder == "auto" and chosen != "libx264":
            chosen = "libx264"
            continue
        if output.exists():
            output.unlink()
        raise RuntimeError(f"ffmpeg ({chosen}) failed:\n{tail}")
    first, last = stats["first"], stats["previous"]
    wrap = float(np.abs(first.astype(np.int16) - last.astype(np.int16)).mean())
    steps = stats["steps"]
    return {
        "path": str(output),
        "frames": frames,
        "fps": fps,
        "size": list(size),
        "seconds": frames / fps,
        "encoder": chosen,
        "quality": quality,
        "gop": max(1, int(round(fps))),
        "b_frames": 0,
        "frames_sha256": digest.hexdigest(),
        "seam": {
            "last_to_first_mean_abs_diff": wrap,
            "median_step_mean_abs_diff": float(np.median(steps)),
            "max_step_mean_abs_diff": float(np.max(steps)),
        },
        "elapsed": time.perf_counter() - began,
        "command": command,
        "human_visual": "NOT_RUN",
    }
