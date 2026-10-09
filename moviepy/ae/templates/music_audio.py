r"""Audio side of the long-form music-channel workflow.

Streaming ports of the S14/S06 recipes: two-pass linear ``loudnorm`` with an
exact-frame contract, looping one master to a chapter length with a linear
crossfade, joining chapters with crossfades and edge fades, a synthetic
two-note sine-bell cue, and quiet cue mixing that never boosts and never
ducks. Long audio is handled in blocks through ``._audio_io``; only one
source master (and two crossfade windows) is ever held in memory.

Every function that creates a file refuses to replace an existing one unless
``overwrite=True``. Evidence dictionaries carry ``"human_listening":
"NOT_RUN"``: no human has listened to anything these functions produce.

Examples
--------
>>> import numpy as np
>>> data = np.zeros((20, 2), np.float32)
>>> looped, starts = loop_extend_array(data, 3.0, rate=10, overlap=0.5)
>>> looped.shape, starts
((30, 2), [0.0, 1.5])
>>> chime_tone().shape
(98400, 2)
"""

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from moviepy.ae.templates._audio_io import (
    AudioWriter,
    audio_info,
    iter_audio,
    read_audio,
)
from moviepy.config import FFMPEG_BINARY


DEFAULT_RATE = 48000
DEFAULT_LUFS = -18.0
DEFAULT_TRUE_PEAK = -1.8
DEFAULT_LRA = 11.0
_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)

DEFAULT_NOTES = (("D4", 293.664768, 0.0, 1.6), ("A4", 440.0, 0.45, 1.6))


def _dbfs(value):
    """Convert a linear amplitude to dBFS with a -240 dB floor."""
    return float(20 * np.log10(max(float(value), 1e-12)))


def _chunks(array, block):
    """Yield independent copies of ``array`` in ``block``-frame pieces."""
    for index in range(0, len(array), block):
        yield np.array(array[index : index + block], dtype=np.float32)


def _load_master(source, rate):
    """Return a finite stereo float32 array from a path or an array."""
    if isinstance(source, (str, Path)):
        data = read_audio(source, rate=rate, channels=2)
    else:
        data = np.asarray(source, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Source audio must be stereo with shape (frames, 2)")
    if not np.isfinite(data).all():
        raise ValueError("Source audio contains non-finite samples")
    return data


def _sha256(source):
    """Return the SHA-256 of a file's bytes or of an array's samples."""
    digest = hashlib.sha256()
    if isinstance(source, (str, Path)):
        with open(source, "rb") as stream:
            for piece in iter(lambda: stream.read(1 << 20), b""):
                digest.update(piece)
    else:
        digest.update(np.ascontiguousarray(source, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _ffmpeg(args):
    """Run FFmpeg quietly and return its stderr text, raising on failure."""
    command = [FFMPEG_BINARY, "-hide_banner", "-nostdin", "-loglevel", "info", *args]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=_FLAGS,
    )
    text = result.stderr.decode("utf-8", "replace")
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed: {text[-2000:]}")
    return text


def _loudnorm_json(text):
    """Return the last loudnorm JSON object found in FFmpeg's log."""
    found = re.findall(r'\{[^{}]*"input_i"[^{}]*\}', text)
    if not found:
        raise RuntimeError("FFmpeg printed no loudnorm measurement")
    return json.loads(found[-1])


# --------------------------------------------------------------------------- #
# Loudness
# --------------------------------------------------------------------------- #


def measure_loudness(
    path, *, target_lufs=DEFAULT_LUFS, true_peak=DEFAULT_TRUE_PEAK, lra=DEFAULT_LRA
):
    """Measure loudness with the first pass of FFmpeg ``loudnorm``.

    Parameters
    ----------
    path : str or Path
        Any audio file FFmpeg can decode.
    target_lufs : float
        Integrated target handed to loudnorm (does not change the measurement).
    true_peak : float
        True-peak target in dBTP (does not change the measurement).
    lra : float
        Loudness-range target (does not change the measurement).

    Returns
    -------
    dict
        ``integrated_lufs``, ``true_peak_dbtp``, ``lra``, ``threshold`` and
        ``target_offset`` as floats, plus the raw ``loudnorm`` fields under
        their FFmpeg names (``input_i`` ...) as strings.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    filt = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:print_format=json"
    text = _ffmpeg(["-i", str(path), "-map", "0:a:0", "-af", filt, "-f", "null", "-"])
    raw = _loudnorm_json(text)
    result = {
        "integrated_lufs": float(raw["input_i"]),
        "true_peak_dbtp": float(raw["input_tp"]),
        "lra": float(raw["input_lra"]),
        "threshold": float(raw["input_thresh"]),
        "target_offset": float(raw["target_offset"]),
    }
    result.update(raw)
    return result


def normalize_loudness(
    source,
    target,
    *,
    target_lufs=DEFAULT_LUFS,
    true_peak=DEFAULT_TRUE_PEAK,
    lra=DEFAULT_LRA,
    rate=DEFAULT_RATE,
    frames=None,
    overwrite=False,
):
    """Normalize with a two-pass linear ``loudnorm`` and an exact frame count.

    The first pass measures ``source``; the second applies the measured
    values with ``linear=true`` and then ``aresample`` (loudnorm may run at
    192 kHz internally, so resampling comes before trimming), ``apad`` and
    ``atrim`` so the result has exactly ``frames`` samples at ``rate``.

    Parameters
    ----------
    source : str or Path
        Input audio.
    target : str or Path
        Output ``.flac`` (32-bit) or ``.wav`` (24-bit PCM), stereo.
    target_lufs : float
        Integrated loudness target.
    true_peak : float
        True-peak ceiling in dBTP.
    lra : float
        Loudness-range target.
    rate : int
        Output sample rate.
    frames : int, optional
        Exact output length in samples; defaults to the source length
        converted to ``rate``.
    overwrite : bool
        Replace an existing ``target`` instead of raising.

    Returns
    -------
    dict
        ``measured`` (input loudness), ``output`` (rate, channels, frames and
        loudness re-measured on the result), ``target`` and
        ``human_listening``.
    """
    source, target = Path(source), Path(target)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    suffix = target.suffix.lower()
    if suffix == ".flac":
        codec = ["-c:a", "flac", "-sample_fmt", "s32"]
    elif suffix in (".wav", ".wave"):
        codec = ["-c:a", "pcm_s24le"]
    else:
        raise ValueError("normalize_loudness writes .flac or .wav files")
    if frames is None:
        info = audio_info(source)
        frames = info["frames"]
        if info["rate"] != rate:
            frames = round(frames * rate / info["rate"])
    frames = int(frames)
    if frames <= 0:
        raise ValueError("frames must be positive")
    measured = measure_loudness(
        source, target_lufs=target_lufs, true_peak=true_peak, lra=lra
    )
    if not math.isfinite(measured["integrated_lufs"]):
        raise ValueError("Source is silent; loudness cannot be normalized")
    filt = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={lra}:linear=true"
    for key, field in (
        ("measured_I", "input_i"),
        ("measured_TP", "input_tp"),
        ("measured_LRA", "input_lra"),
        ("measured_thresh", "input_thresh"),
        ("offset", "target_offset"),
    ):
        filt += f":{key}={measured[field]}"
    filt += f",aresample={rate},apad=whole_len={frames},atrim=end_sample={frames}"
    target.parent.mkdir(parents=True, exist_ok=True)
    args = ["-y" if overwrite else "-n", "-i", str(source), "-map", "0:a:0"]
    args += ["-af", filt, "-ar", str(rate), "-ac", "2", *codec, str(target)]
    _ffmpeg(args)
    info = audio_info(target)
    if (info["frames"], info["rate"], info["channels"]) != (frames, rate, 2):
        target.unlink()
        raise ValueError(
            f"Normalized audio violated the exact-frame contract: {info} != "
            f"{frames} frames at {rate} Hz stereo"
        )
    return {
        "measured": measured,
        "output": {
            "path": str(target),
            "rate": rate,
            "channels": 2,
            "frames": frames,
            "loudness": measure_loudness(
                target, target_lufs=target_lufs, true_peak=true_peak, lra=lra
            ),
        },
        "target": {"lufs": target_lufs, "true_peak_dbtp": true_peak, "lra": lra},
        "human_listening": "NOT_RUN",
    }


def music_audio_report(path):
    """Return an evidence dictionary (format and loudness) for one audio file.

    Parameters
    ----------
    path : str or Path
        Audio file to describe.

    Returns
    -------
    dict
        ``path``, ``info`` (rate, channels, exact frames, seconds),
        ``loudness`` and ``human_listening``.
    """
    return {
        "path": str(path),
        "info": audio_info(path),
        "loudness": measure_loudness(path),
        "human_listening": "NOT_RUN",
    }


# --------------------------------------------------------------------------- #
# Looping and assembly
# --------------------------------------------------------------------------- #


def _loop_blocks(data, length, cross, block):
    """Generate the looped stream; see ``loop_extend``."""
    ramp = np.linspace(0, 1, cross, dtype=np.float32)[:, None]
    count_source = len(data)
    period = count_source - cross
    position, tail = 0, None
    while position < length:
        count = min(count_source, length - position)
        piece = data[:count]
        more = position + period < length
        begin = 0
        if tail is not None:
            blend = min(cross, count)
            mixed = tail[:blend] * (1 - ramp[:blend]) + piece[:blend] * ramp[:blend]
            yield from _chunks(mixed, block)
            begin = blend
        end = period if more else count
        yield from _chunks(piece[begin:end], block)
        tail = piece[period:count] if more else None
        position += period


def loop_extend(source, seconds, *, rate=DEFAULT_RATE, overlap=10.0, block=None):
    """Repeat ``source`` with a linear crossfade to an exact length.

    Streamed equivalent of the S14 ``chapter()``: the master is repeated with
    ``overlap`` seconds of linear crossfade between the end of one pass and
    the start of the next until exactly ``round(seconds * rate)`` samples.

    Parameters
    ----------
    source : str, Path or numpy.ndarray
        Stereo master (file, or float array shaped ``(frames, 2)``). It must
        be longer than twice ``overlap``.
    seconds : float
        Output length; the result has exactly ``round(seconds * rate)`` frames.
    rate : int
        Sample rate (files are resampled to it).
    overlap : float
        Crossfade length in seconds between repeats.
    block : int, optional
        Frames per yielded block (default ``10 * rate``).

    Returns
    -------
    blocks : iterator of numpy.ndarray
        Float32 ``(n, 2)`` blocks that concatenate to the looped audio.
    starts : list of float
        Start second of every pass of the source inside the output.
    """
    data = _load_master(source, rate)
    length, cross = round(seconds * rate), round(overlap * rate)
    if length <= 0 or cross <= 0:
        raise ValueError("seconds and overlap must be positive")
    if len(data) <= 2 * cross:
        raise ValueError("Stereo source must exceed twice the loop overlap")
    period = len(data) - cross
    starts = [start / rate for start in range(0, length, period)]
    return _loop_blocks(data, length, cross, int(block or rate * 10)), starts


def loop_extend_array(data, seconds, *, rate=DEFAULT_RATE, overlap=10.0):
    """In-memory convenience wrapper of ``loop_extend``.

    Parameters
    ----------
    data : numpy.ndarray
        Stereo float array shaped ``(frames, 2)``.
    seconds : float
        Output length in seconds.
    rate : int
        Sample rate.
    overlap : float
        Crossfade length in seconds.

    Returns
    -------
    array : numpy.ndarray
        The looped ``(round(seconds * rate), 2)`` float32 audio.
    starts : list of float
        Start second of every pass of the source.
    """
    blocks, starts = loop_extend(data, seconds, rate=rate, overlap=overlap)
    return np.concatenate(list(blocks)), starts


def assemble_chapters(
    sources,
    seconds,
    output,
    *,
    rate=DEFAULT_RATE,
    loop_overlap=10.0,
    chapter_crossfade=12.0,
    edge_fade=4.0,
    ids=None,
    allow_clipping=False,
    overwrite=False,
):
    """Loop each master to a chapter length and join chapters with crossfades.

    Each source is extended with ``loop_extend``; neighbouring chapters
    overlap by ``chapter_crossfade`` seconds (linear), so the total is
    ``sum(chapter lengths) - (n - 1) * chapter_crossfade``. The output starts
    with a linear fade-in and ends with a linear fade-out of ``edge_fade``
    seconds (both reach exactly zero). Only the crossfade windows are kept in
    memory besides one source master.

    Parameters
    ----------
    sources : sequence of str, Path or numpy.ndarray
        One stereo master per chapter.
    seconds : float or sequence of float
        Prepared length of every chapter (one number applies to all).
    output : str or Path
        Output ``.flac`` or ``.wav``.
    rate : int
        Sample rate.
    loop_overlap : float
        Crossfade between loop repeats within a chapter.
    chapter_crossfade : float
        Crossfade between chapters; each chapter must be longer than twice it.
    edge_fade : float
        Fade length at the start and end (at most ``chapter_crossfade``).
    ids : sequence of str, optional
        Chapter ids (default ``G01``, ``G02`` ...).
    allow_clipping : bool
        Do not raise when samples reach full scale.
    overwrite : bool
        Replace an existing ``output`` instead of raising.

    Returns
    -------
    dict
        Cue sheet: ``chapters`` (id, source, sha256, prepared seconds and
        frames, loop starts, global start and crossfade-centre start),
        ``total_frames``, ``total_seconds``, ``peak``, ``clipped_samples``
        and ``human_listening``.
    """
    sources = list(sources)
    count = len(sources)
    if not count:
        raise ValueError("At least one source is required")
    if isinstance(seconds, (int, float)):
        seconds = [seconds] * count
    seconds = list(seconds)
    if len(seconds) != count:
        raise ValueError("seconds must have one entry per source")
    ids = list(ids) if ids is not None else [f"G{i + 1:02}" for i in range(count)]
    if len(ids) != count:
        raise ValueError("ids must have one entry per source")
    cross, fade = round(chapter_crossfade * rate), round(edge_fade * rate)
    lengths = [round(value * rate) for value in seconds]
    if cross <= 0 or fade < 0 or fade > cross:
        raise ValueError("Need 0 <= edge_fade <= chapter_crossfade, crossfade > 0")
    if any(length < 2 * cross for length in lengths):
        raise ValueError("Every chapter must be at least twice the crossfade")
    if Path(output).exists() and not overwrite:
        raise FileExistsError(output)
    total = sum(lengths) - (count - 1) * cross
    ramp = np.linspace(0, 1, cross, dtype=np.float32)[:, None]
    state = {"peak": 0.0, "clipped": 0}
    sheet, offset = [], 0

    def emit(writer, array):
        if not len(array):
            return
        magnitude = np.abs(array)
        state["peak"] = max(state["peak"], float(magnitude.max()))
        state["clipped"] += int((magnitude >= 1).sum())
        if state["clipped"] and not allow_clipping:
            raise ValueError("Assembled audio clips (samples at or above full scale)")
        writer.write(array)

    with AudioWriter(output, rate=rate, channels=2, overwrite=overwrite) as writer:
        previous_tail = None
        for index, source in enumerate(sources):
            blocks, starts = loop_extend(
                source, seconds[index], rate=rate, overlap=loop_overlap
            )
            sheet.append(
                {
                    "id": ids[index],
                    "source": (
                        str(source) if isinstance(source, (str, Path)) else "array"
                    ),
                    "sha256": _sha256(source),
                    "prepared_seconds": lengths[index] / rate,
                    "prepared_frames": lengths[index],
                    "source_loop_starts": starts,
                    "global_start_seconds": offset / rate,
                    "crossfade_center_seconds": (offset + (cross / 2 if index else 0))
                    / rate,
                }
            )
            offset += lengths[index] - cross
            head, buffer = None, np.zeros((0, 2), np.float32)
            for piece in blocks:
                buffer = np.concatenate([buffer, piece]) if len(buffer) else piece
                if head is None:
                    if len(buffer) < cross:
                        continue
                    head, buffer = buffer[:cross].copy(), buffer[cross:]
                    if previous_tail is None:
                        if fade:
                            head[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[
                                :, None
                            ]
                        emit(writer, head)
                    else:
                        emit(writer, previous_tail * (1 - ramp) + head * ramp)
                if len(buffer) > cross:
                    emit(writer, buffer[:-cross])
                    buffer = buffer[-cross:]
            if len(buffer) != cross:
                raise RuntimeError("Chapter stream ended with an unexpected tail")
            previous_tail = buffer.copy()
            if index == count - 1:
                if fade:
                    previous_tail[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[
                        :, None
                    ]
                emit(writer, previous_tail)
        if writer.frames_written != total:
            raise ValueError(
                f"Assembled {writer.frames_written} frames, expected {total}"
            )
    if audio_info(output)["frames"] != total:
        raise ValueError("Assembled file does not have the exact expected length")
    return {
        "rate": rate,
        "chapters": sheet,
        "loop_overlap_seconds": loop_overlap,
        "chapter_crossfade_seconds": chapter_crossfade,
        "edge_fade_seconds": edge_fade,
        "total_frames": total,
        "total_seconds": total / rate,
        "peak": state["peak"],
        "peak_dbfs": _dbfs(state["peak"]),
        "clipped_samples": state["clipped"],
        "human_listening": "NOT_RUN",
    }


# --------------------------------------------------------------------------- #
# Chime
# --------------------------------------------------------------------------- #


def _note_envelope(count, attack, release, rate):
    """Return a smooth 0 -> 1 -> 0 envelope of ``count`` samples."""
    time = np.arange(count) / rate
    seconds = (count - 1) / rate
    attack = min(attack, seconds)
    release_start = max(attack, seconds - release)
    envelope = np.ones(count)
    rising = time < attack
    if attack > 0:
        envelope[rising] = np.sin(0.5 * np.pi * time[rising] / attack) ** 2
    falling = time > release_start
    if seconds > release_start:
        progress = (time[falling] - release_start) / (seconds - release_start)
        decay = (np.exp(-5.0 * progress) - np.exp(-5.0)) / (1 - np.exp(-5.0))
        envelope[falling] = decay * np.cos(0.5 * np.pi * progress)
    envelope[0] = 0.0
    envelope[-1] = 0.0
    return envelope


def chime_tone(
    *,
    notes=DEFAULT_NOTES,
    duration=2.05,
    attack=0.08,
    release=1.25,
    peak_dbfs=-30.0,
    rate=DEFAULT_RATE,
):
    """Synthesize a soft sine-bell cue (S14 ``CHIME-DESIGN.json`` default).

    Pure sines with a raised-sine attack and an exponential-like release that
    ends at exactly zero, so the first and last samples are 0 and nothing
    clicks. The mix is scaled so its peak equals ``peak_dbfs``.

    Parameters
    ----------
    notes : sequence of tuple
        ``(name, frequency_hz, start_seconds, duration_seconds)`` per note.
    duration : float
        Total length; the array has ``round(duration * rate)`` frames.
    attack : float
        Attack length of each note in seconds.
    release : float
        Release length of each note in seconds (clamped to the note).
    peak_dbfs : float
        Peak level of the result.
    rate : int
        Sample rate.

    Returns
    -------
    numpy.ndarray
        Float32 centred stereo array shaped ``(frames, 2)``.
    """
    total = round(duration * rate)
    mono = np.zeros(total)
    for _, frequency, start, length in notes:
        first, count = round(start * rate), round(length * rate)
        if count < 2 or first < 0 or first + count > total:
            raise ValueError("Every note must fit inside the cue duration")
        time = np.arange(count) / rate
        envelope = _note_envelope(count, attack, release, rate)
        mono[first : first + count] += np.sin(2 * np.pi * frequency * time) * envelope
    peak = np.abs(mono).max()
    if peak <= 0:
        raise ValueError("The cue is silent")
    mono *= 10 ** (peak_dbfs / 20) / peak
    return np.repeat(mono[:, None], 2, axis=1).astype(np.float32)


def write_chime(path, *, overwrite=False, **kwargs):
    r"""Render ``chime_tone`` into a 24-bit WAV file.

    Parameters
    ----------
    path : str or Path
        Output ``.wav`` (or ``.flac``); exclusive create by default.
    overwrite : bool
        Replace an existing file instead of raising.
    \*\*kwargs
        Passed to ``chime_tone``.

    Returns
    -------
    dict
        Facts: ``frames``, ``rate``, ``peak_dbfs``, ``rms_dbfs``, first and
        last sample, ``clipped_samples`` and ``human_listening``.
    """
    rate = kwargs.get("rate", DEFAULT_RATE)
    tone = chime_tone(**kwargs)
    with AudioWriter(path, rate=rate, channels=2, overwrite=overwrite) as writer:
        writer.write(tone)
    rms = float(np.sqrt(np.mean(tone.astype(np.float64) ** 2)))
    return {
        "path": str(path),
        "frames": len(tone),
        "rate": rate,
        "channels": 2,
        "peak_dbfs": _dbfs(np.abs(tone).max()),
        "rms_dbfs": _dbfs(rms),
        "first_sample": float(tone[0, 0]),
        "last_sample": float(tone[-1, 0]),
        "clipped_samples": int((np.abs(tone) >= 1).sum()),
        "human_listening": "NOT_RUN",
    }


# --------------------------------------------------------------------------- #
# Cue mixing
# --------------------------------------------------------------------------- #


def mix_cues(
    music,
    cue,
    times,
    output,
    *,
    below_music_db=10.0,
    rate=DEFAULT_RATE,
    overwrite=False,
):
    """Add a quiet cue at given times, never boosting and never ducking.

    Port of the S14 ``mix_chimes``: for each event the gain is
    ``min(1, music_rms * 10**(-below_music_db / 20) / cue_rms)`` with the music
    RMS measured in the cue window, so the cue keeps its designed level or is
    lowered. The music is otherwise untouched (no normalization, no ducking).
    Two streaming passes: measure the windows, then mix block by block.

    Parameters
    ----------
    music : str or Path
        Stereo music file at ``rate`` (``.wav`` or ``.flac``).
    cue : str, Path or numpy.ndarray
        Cue file or stereo float array at ``rate``.
    times : sequence of float
        Event start seconds; unique, ordered, and the cue must fit.
    output : str or Path
        Output ``.flac`` or ``.wav``.
    below_music_db : float
        Desired cue RMS below the local music RMS.
    rate : int
        Required sample rate of music and cue.
    overwrite : bool
        Replace an existing ``output`` instead of raising.

    Returns
    -------
    dict
        ``events`` (seconds, sample, gain, cue/music RMS dBFS, difference),
        ``peak``, ``clipped_samples``, ``frames``, ``normalization`` and
        ``ducking`` (both False) and ``human_listening``.
    """
    if Path(output).exists() and not overwrite:
        raise FileExistsError(output)
    info = audio_info(music)
    if info["rate"] != rate or info["channels"] != 2:
        raise ValueError("Music must be stereo at the required rate")
    if isinstance(cue, (str, Path)):
        cue = read_audio(cue, rate=rate, channels=2)
    cue = np.asarray(cue, dtype=np.float32)
    if cue.ndim != 2 or cue.shape[1] != 2 or not len(cue):
        raise ValueError("The cue must be non-empty stereo audio")
    if not np.isfinite(cue).all():
        raise ValueError("Invalid cue audio")
    starts = [round(t * rate) for t in times]
    if starts != sorted(set(starts)) or any(
        s < 0 or s + len(cue) > info["frames"] for s in starts
    ):
        raise ValueError("Cue positions must be unique, ordered and within the music")
    cue_rms = float(np.sqrt(np.mean(cue.astype(np.float64) ** 2)))
    if cue_rms <= 0:
        raise ValueError("Cue is silent")
    sums = [0.0] * len(starts)
    seen = 0
    for piece in iter_audio(music, rate=rate, channels=2):
        if not np.isfinite(piece).all():
            raise ValueError("Non-finite music samples")
        for slot, start in enumerate(starts):
            low, high = max(seen, start), min(seen + len(piece), start + len(cue))
            if low < high:
                window = piece[low - seen : high - seen].astype(np.float64)
                sums[slot] += float(np.sum(window**2))
        seen += len(piece)
    if seen != info["frames"]:
        raise ValueError("Music decoded to an unexpected number of frames")
    records = []
    for start, energy in zip(starts, sums):
        music_rms = math.sqrt(energy / (len(cue) * 2))
        gain = min(1.0, music_rms * 10 ** (-below_music_db / 20) / cue_rms)
        records.append(
            {
                "seconds": start / rate,
                "sample": start,
                "gain": gain,
                "cue_rms_dbfs": _dbfs(cue_rms * gain),
                "music_rms_dbfs": _dbfs(music_rms),
                "difference_db": _dbfs(cue_rms * gain) - _dbfs(music_rms),
            }
        )
    peak, clipped, offset = 0.0, 0, 0
    with AudioWriter(output, rate=rate, channels=2, overwrite=overwrite) as writer:
        for piece in iter_audio(music, rate=rate, channels=2):
            piece = np.array(piece, dtype=np.float32)
            for start, record in zip(starts, records):
                low = max(offset, start)
                high = min(offset + len(piece), start + len(cue))
                if low < high:
                    gain = np.float32(record["gain"])
                    piece[low - offset : high - offset] += (
                        cue[low - start : high - start] * gain
                    )
            magnitude = np.abs(piece)
            peak = max(peak, float(magnitude.max()))
            clipped += int((magnitude >= 1).sum())
            if clipped:
                raise ValueError("Cue mix clipped (samples at or above full scale)")
            writer.write(piece)
            offset += len(piece)
        if offset != info["frames"]:
            raise ValueError("Cue mix changed the music duration")
    if audio_info(output)["frames"] != info["frames"]:
        raise ValueError("Cue mix output has a different frame count")
    return {
        "events": records,
        "peak": peak,
        "clipped_samples": clipped,
        "frames": offset,
        "normalization": False,
        "ducking": False,
        "human_listening": "NOT_RUN",
    }
