"""Multi-process frame rendering and hardware-accelerated video encoding.

Two independent speed-ups for long exports, neither of which changes a single
rendered pixel:

* ``iter_frames_parallel`` renders frames in several worker processes and
  yields them in order. Every worker builds its *own* clip from a picklable
  ``factory`` (a composition is never pickled, so this works with the Windows
  ``spawn`` start method) and renders contiguous chunks of frames so that its
  per-clip caches stay effective. Frames travel through one
  ``multiprocessing.shared_memory`` segment divided into slots instead of
  being pickled.
* ``detect_encoder`` / ``encoder_args`` choose an FFmpeg encoder,
  preferring NVIDIA NVENC when a real test encode succeeds and falling back to
  ``libx264``. ``write_video_parallel`` ties both together.
"""

import multiprocessing
import os
import queue
import subprocess
import tempfile
import time
import traceback
from fractions import Fraction
from multiprocessing import shared_memory

import numpy as np

from moviepy.config import FFMPEG_BINARY


__all__ = [
    "EncoderUnavailable",
    "ENCODERS",
    "default_workers",
    "detect_encoder",
    "encoder_args",
    "iter_frames_parallel",
    "write_video_parallel",
]

ENCODERS = ("libx264", "h264_nvenc", "hevc_nvenc", "libx265")
QUALITIES = ("high", "balanced", "draft")

# Names of the shared-memory segments currently owned by this process (a
# leak detector for tests; always empty when no render is running).
_SEGMENTS = set()
# encoder name -> bool, filled by real test encodes (once per process).
_ENCODER_CACHE = {}
_SLOT_BUDGET = 512 * 1024 * 1024  # bytes of shared memory the ring may use


class EncoderUnavailable(ValueError):
    """The requested FFmpeg encoder cannot be used on this machine."""


def default_workers():
    """Return the default worker count: ``min(cpu_count - 1, 12)``, at least 1."""
    return max(1, min((os.cpu_count() or 2) - 1, 12))


def _close_clip(clip):
    """Close ``clip`` and the source clips of an episode, ignoring errors."""
    clips = [clip, *getattr(clip, "episode_clips", ())]
    for item in clips:
        try:
            item.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


def _as_frame(frame, shape):
    """Return ``frame`` as a uint8 array of ``shape`` or raise ``ValueError``."""
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.shape != shape:
        raise ValueError(f"frame has shape {frame.shape}, expected {shape}")
    return frame


# --------------------------------------------------------------------------- #
# worker process
# --------------------------------------------------------------------------- #


def _single_thread():
    """Keep OpenCV from starting a thread pool per worker (oversubscription)."""
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:  # pragma: no cover - cv2 is an AE dependency
        pass


def _worker(factory, factory_args, name, slots, shape, tasks, results, gate):
    """Render chunks taken from ``tasks`` into shared-memory slots."""
    shm = ring = clip = None
    try:
        _single_thread()
        shm = shared_memory.SharedMemory(name=name)
        ring = np.ndarray((slots, *shape), dtype=np.uint8, buffer=shm.buf)
        clip = factory(*factory_args)
        while True:
            task = tasks.get()
            if task is None:
                break
            start, times = task
            for offset, t in enumerate(times):
                index = start + offset
                frame = _as_frame(clip.get_frame(t), shape)
                # Slot ``index % slots`` is free once frame ``index - slots``
                # was consumed. Waiting on the count of consumed frames (not
                # on a per-slot lock) keeps a far-ahead frame from taking a
                # slot before the earlier frame that shares it; the parent
                # consumes in order, so this cannot deadlock.
                with gate["cond"]:
                    gate["cond"].wait_for(lambda: gate["done"].value > index - slots)
                ring[index % slots] = frame
                results.put(("ok", index))
    except BaseException:  # noqa: BLE001 - reported to the parent
        results.put(("err", traceback.format_exc()))
    finally:
        ring = None
        if clip is not None:
            _close_clip(clip)
        if shm is not None:
            try:
                shm.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# public frame iterator
# --------------------------------------------------------------------------- #


def iter_frames_parallel(
    factory,
    factory_args,
    times,
    *,
    workers=None,
    chunk=None,
    size=None,
    clip=None,
    copy=True,
):
    """Yield the uint8 RGB frames of ``factory(*factory_args)`` at ``times``.

    Frames are yielded in the order of ``times`` and are byte-identical to
    ``clip.get_frame(t).astype("uint8")``, whatever ``workers`` is.

    Parameters
    ----------
    factory : callable
        Picklable top-level callable returning a clip or composition. Each
        worker process calls ``factory(*factory_args)`` once.
    factory_args : tuple
        Picklable arguments of ``factory``.
    times : sequence of float
        Frame times in seconds.
    workers : int, optional
        Number of processes; ``None`` means ``default_workers`` and
        ``workers <= 1`` renders sequentially in this process.
    chunk : int, optional
        Contiguous frames per task (default 4). Contiguity keeps per-worker
        caches useful; smaller chunks balance load better.
    size : tuple of int, optional
        ``(width, height)`` of the frames. When omitted it is read from a clip
        (``clip`` if given, else a temporary ``factory`` build).
    clip : optional
        An already built clip, used as is by the sequential path (and not
        closed) and to find ``size``.
    copy : bool
        Yield private copies (default). With ``False`` the yielded array is a
        view of shared memory, valid only until the next iteration.

    Yields
    ------
    numpy.ndarray
        ``(height, width, 3)`` uint8 frames.

    Notes
    -----
    A worker error is re-raised as ``RuntimeError`` carrying the worker's
    traceback text. On normal end, exceptions, ``KeyboardInterrupt`` or closing
    the generator, all workers are terminated and the shared memory released.
    """
    times = [float(t) for t in times]
    count = len(times)
    workers = default_workers() if workers is None else int(workers)
    if count == 0:
        return
    if workers <= 1:
        own = clip is None
        active = factory(*factory_args) if own else clip
        try:
            shape = (int(active.size[1]), int(active.size[0]), 3)
            for t in times:
                yield _as_frame(active.get_frame(t), shape)
        finally:
            if own:
                _close_clip(active)
        return

    if size is None:
        probe = clip if clip is not None else factory(*factory_args)
        try:
            size = tuple(probe.size)
        finally:
            if clip is None:
                _close_clip(probe)
    shape = (int(size[1]), int(size[0]), 3)
    frame_bytes = shape[0] * shape[1] * 3
    workers = min(workers, count)
    chunk = max(1, int(chunk) if chunk else 4)
    wanted = 2 * workers * chunk
    capped = max(3 * workers, _SLOT_BUDGET // frame_bytes)
    slots = max(1, min(count, max(3 * workers, min(wanted, capped))))

    ctx = multiprocessing.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    gate = {"cond": ctx.Condition(), "done": ctx.Value("q", 0, lock=False)}
    shm = ring = None
    procs = []
    try:
        shm = shared_memory.SharedMemory(create=True, size=slots * frame_bytes)
        _SEGMENTS.add(shm.name)
        ring = np.ndarray((slots, *shape), dtype=np.uint8, buffer=shm.buf)
        for start in range(0, count, chunk):
            tasks.put((start, times[start : start + chunk]))
        for _ in range(workers):
            tasks.put(None)
        for _ in range(workers):
            proc = ctx.Process(
                target=_worker,
                args=(factory, factory_args, shm.name, slots, shape, tasks, results,
                      gate),
                daemon=True,
            )  # fmt: skip
            proc.start()
            procs.append(proc)

        ready, following = set(), 0
        while following < count:
            try:
                message = results.get(timeout=0.2)
            except queue.Empty:
                for proc in procs:
                    if proc.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"render worker died with exit code {proc.exitcode}"
                        )
                continue
            if message[0] == "err":
                raise RuntimeError("render worker failed:\n" + message[1])
            ready.add(message[1])
            while following in ready:
                ready.discard(following)
                slot = following % slots
                yield ring[slot].copy() if copy else ring[slot]
                with gate["cond"]:
                    gate["done"].value = following + 1
                    gate["cond"].notify_all()
                following += 1
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(5)
        for channel in (tasks, results):
            channel.close()
            channel.cancel_join_thread()
        ring = None
        if shm is not None:
            _SEGMENTS.discard(shm.name)
            try:
                shm.close()
            except BufferError:  # a caller still holds a view; unlink anyway
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _ffmpeg_lists_encoder(name):
    """True when ``ffmpeg -encoders`` lists ``name``."""
    try:
        run = subprocess.run(
            [FFMPEG_BINARY, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return run.returncode == 0 and any(
        line.split()[1:2] == [name] for line in run.stdout.splitlines()
    )


def _test_encode(name):
    """Really encode a few frames with ``name``; True on success.

    Listing in ``-encoders`` is not enough for NVENC: the driver or the GPU may
    be missing. The test clip is 256x144 because NVENC rejects very small
    frames (HEVC needs at least 129 px on some GPUs).
    """
    command = [
        FFMPEG_BINARY, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "lavfi", "-i", "color=c=gray:s=256x144:r=10:d=0.5",
        "-frames:v", "3", "-c:v", name, "-pix_fmt", "yuv420p", "-f", "null", "-",
    ]  # fmt: skip
    try:
        run = subprocess.run(
            command,
            capture_output=True,
            timeout=60,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return run.returncode == 0


def _usable(name):
    """Cached availability of encoder ``name`` (per process)."""
    if name not in _ENCODER_CACHE:
        if name.endswith("_nvenc"):
            _ENCODER_CACHE[name] = _test_encode(name)
        else:
            _ENCODER_CACHE[name] = _ffmpeg_lists_encoder(name)
    return _ENCODER_CACHE[name]


def detect_encoder(preferred="auto"):
    """Return the name of an FFmpeg H.26x encoder that works here.

    Parameters
    ----------
    preferred : str
        ``"auto"`` (default) returns ``"h264_nvenc"`` if a real test encode
        with it succeeds, else ``"libx264"``. An explicit ``"libx264"``,
        ``"h264_nvenc"``, ``"hevc_nvenc"`` or ``"libx265"`` is validated and
        returned unchanged. Results are cached per process.

    Raises
    ------
    EncoderUnavailable
        For an unknown name or an encoder that is missing or fails to run.
    """
    if preferred == "auto":
        return "h264_nvenc" if _usable("h264_nvenc") else "libx264"
    if preferred not in ENCODERS:
        raise EncoderUnavailable(
            f"unknown encoder {preferred!r}; choose auto or one of {list(ENCODERS)}"
        )
    if not _usable(preferred):
        reason = (
            "the NVIDIA driver/GPU is unavailable or ffmpeg lacks NVENC"
            if preferred.endswith("_nvenc")
            else "this ffmpeg build does not include it"
        )
        raise EncoderUnavailable(f"encoder {preferred} is not usable: {reason}")
    return preferred


_QUALITY = {
    # quality: (x264/x265 preset, crf, nvenc preset, nvenc cq)
    "high": ("medium", 18, "p5", 19),
    "balanced": ("medium", 23, "p4", 25),
    "draft": ("veryfast", 28, "p3", 31),
}


def encoder_args(encoder, quality="high"):
    """Return the FFmpeg output arguments (video encoder section) for ``encoder``.

    ``quality`` is ``"high"`` (visually transparent: CRF 18 / NVENC CQ 19),
    ``"balanced"`` or ``"draft"``. Software encoders use constant-quality
    ``-crf``; NVENC uses ``-rc vbr -cq N -b:v 0`` (quality-targeted VBR) with
    the slow ``p5`` preset and ``-tune hq``. ``h264_nvenc`` adds ``-bf 2``
    (B-frames; HEVC B-frames are left to the driver default for compatibility
    with older GPUs). All use ``-pix_fmt yuv420p`` and ``-movflags +faststart``;
    HEVC adds ``-tag:v hvc1`` so Apple players accept the file.
    """
    if quality not in _QUALITY:
        raise ValueError(f"quality must be one of {list(QUALITIES)}, got {quality!r}")
    preset, crf, nv_preset, cq = _QUALITY[quality]
    if encoder in ("libx264", "libx265"):
        args = ["-c:v", encoder, "-preset", preset, "-crf", str(crf)]
    elif encoder in ("h264_nvenc", "hevc_nvenc"):
        args = ["-c:v", encoder, "-preset", nv_preset, "-tune", "hq", "-rc", "vbr",
                "-cq", str(cq), "-b:v", "0"]  # fmt: skip
        if encoder == "h264_nvenc":
            args += ["-bf", "2"]
    else:
        raise EncoderUnavailable(f"unknown encoder {encoder!r}")
    if encoder in ("libx265", "hevc_nvenc"):
        args += ["-tag:v", "hvc1"]
    return args + ["-pix_fmt", "yuv420p", "-movflags", "+faststart"]


# --------------------------------------------------------------------------- #
# video writer
# --------------------------------------------------------------------------- #


def _ffmpeg_command(output, fps, size, frames, encoder, quality, audio_path):
    """Build the FFmpeg command reading rgb24 frames from stdin."""
    rate = Fraction(str(fps)).limit_denominator(1000000)
    command = [
        FFMPEG_BINARY, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
        "-r", f"{rate.numerator}/{rate.denominator}", "-i", "-",
    ]  # fmt: skip
    if audio_path is not None:
        command += ["-i", str(audio_path)]
    command += ["-map", "0:v:0"]
    if audio_path is not None:
        command += ["-map", "1:a:0"]
    command += encoder_args(encoder, quality)
    length = Fraction(frames) / rate
    if audio_path is not None:
        # apad + an explicit output length trim or pad the audio to the video
        # duration exactly (no -shortest).
        command += ["-c:a", "aac", "-b:a", "192k", "-af", "apad"]
    command += ["-frames:v", str(frames), "-t", f"{float(length):.6f}", str(output)]
    return command


def _encode(command, frames, hook, bar):
    """Pipe ``frames`` into FFmpeg; return True when it exited cleanly."""
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
                for frame, _ in zip(frames, bar):
                    if hook is not None:
                        hook(frame)
                    proc.stdin.write(np.ascontiguousarray(frame).data)
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
            tail = log.read().decode("utf-8", "replace")[-2000:]
            return False, tail
        return True, ""


def write_video_parallel(
    factory,
    factory_args,
    output,
    *,
    fps,
    duration=None,
    n_frames=None,
    size=None,
    audio_path=None,
    workers=None,
    encoder="auto",
    quality="high",
    logger=None,
    clip=None,
    frame_hook=None,
):
    """Render ``factory(*factory_args)`` to ``output`` with FFmpeg.

    Frames come from ``iter_frames_parallel`` at ``t = i / fps`` for
    ``i in range(n_frames)``, where ``n_frames`` defaults to
    ``int(duration * fps)``: exactly the frame set of MoviePy's
    ``write_videofile``. They are piped to one FFmpeg process as raw rgb24.

    Parameters
    ----------
    output : path
        Output file (mp4 recommended; ``+faststart`` is set).
    fps : float
        Frame rate.
    duration, n_frames : float, int
        One of them is required.
    size : tuple of int, optional
        ``(width, height)``; read from the clip when omitted.
    audio_path : path, optional
        Audio file muxed as AAC 192k, padded with silence or trimmed to the
        video duration.
    workers : int, optional
        See ``iter_frames_parallel``.
    encoder : str
        See ``detect_encoder``. With ``"auto"``, if NVENC passes the test
        but the real encode fails (frame size or session limits), the whole
        render is retried once with ``libx264``.
    quality : str
        See ``encoder_args``.
    logger : optional
        ``None`` or a proglog logger / ``"bar"``.
    clip : optional
        An already built clip for the sequential path and size probing.
    frame_hook : callable, optional
        Called with every frame, in order, before it is encoded (for hashing).

    Returns
    -------
    dict
        ``encoder``, ``workers``, ``frames``, ``seconds`` and ``fps_achieved``.
    """
    import proglog

    if n_frames is None:
        if duration is None:
            raise ValueError("give duration or n_frames")
        n_frames = int(duration * fps)
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError("nothing to render: n_frames < 1")
    workers = default_workers() if workers is None else max(1, int(workers))
    chosen = detect_encoder(encoder)
    if size is None:
        probe = clip if clip is not None else factory(*factory_args)
        try:
            size = tuple(probe.size)
        finally:
            if clip is None:
                _close_clip(probe)
    size = (int(size[0]), int(size[1]))
    times = [i / fps for i in range(n_frames)]
    logger = proglog.default_bar_logger(logger)
    began = time.perf_counter()
    for attempt in range(2):
        command = _ffmpeg_command(
            output, fps, size, n_frames, chosen, quality, audio_path
        )
        frames = iter_frames_parallel(
            factory, factory_args, times, workers=workers, size=size, clip=clip
        )
        bar = logger.iter_bar(frame_index=range(n_frames))
        try:
            good, tail = _encode(command, frames, frame_hook, bar)
        finally:
            frames.close()
        if good:
            break
        if attempt == 0 and encoder == "auto" and chosen != "libx264":
            chosen = "libx264"
            continue
        raise RuntimeError(f"ffmpeg ({chosen}) failed:\n{tail}")
    seconds = time.perf_counter() - began
    return {
        "encoder": chosen,
        "workers": min(workers, n_frames),
        "frames": n_frames,
        "seconds": seconds,
        "fps_achieved": n_frames / seconds if seconds > 0 else float("inf"),
    }
