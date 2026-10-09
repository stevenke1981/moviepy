r"""Declarative music-channel episodes: spec, resumable steps and command line.

A ``MusicEpisodeSpec`` describes a whole long-form music video (a
5 h 20 min sleep mix or an 87 min study companion) as data. Four resumable
steps build it; each writes its artifact next to an evidence file and, when
run again, skips the artifact after re-verifying its SHA-256 hash, so a
10-hour build can be resumed after a crash or a power cut:

``audio``
    Two-pass linear loudnorm of every track, loop-extension and crossfaded
    assembly into one chapter bed, and (study mode) a quiet chime mixed at
    the phase boundaries. The final length must equal the study schedule
    (study) or the chapter sum minus the crossfades (sleep) and be a whole
    number of video frames.
``visual``
    One seamless loop per clip and a scene cycle (``export_loop`` MP4s that
    FFmpeg repeats with ``-stream_loop -1``), or a prepared ``cycle_path``.
``overlays``
    Spectrum level table (``.npy``) analysed from the final audio, the study
    ASS file and the ffmetadata chapters (study chapters, or sleep chapters
    from the cue sheet).
``render``
    One FFmpeg pass (``render_music_video``) or a preview window.

Shipped recommended configs (``configs/music_*.json``) and the
``music_channel`` choices they come from:

``music_sleep_longform.json`` (S06 long-form sleep, ``build-season-06.ps1``)
    Six pieces of 3210 s joined with 12 s chapter crossfades give exactly
    19200 s (5 h 20 min); 10 s loop crossfades inside a chapter; 4 s edge
    fades; a single shared native 1280x720 / 24 fps loop (8 s clips, 2 s
    loop dissolve); loudness -18 LUFS, true peak -1.8 dBTP, LRA 11; no
    spectrum and no chime; chapters from the cue sheet; NVENC when present.
``music_study_pomodoro.json`` (S14 study companion, ``s14-assets.py``, ``s14-render.py``)
    A 5220 s schedule (3 x 25 min focus, 5 min breaks, 2 min closing; chimes
    at 1500, 1800, 3300, 3600 and 5100 s), six pieces of 880 s (6 x 880 - 5 x
    12 = 5220), chimes 10 dB under the local music RMS (never boosted, no
    ducking), spectrum bars at 30 % opacity at (96, 548) in a 1088x140
    layer, AAC 256k. S14 burned ``study-overlay-v2.ass``, the non-compact
    three-panel layout, so ``study_compact`` is ``false`` (the compact
    backplate layout appeared in later seasons).

Examples
--------
>>> spec = MusicEpisodeSpec.from_dict({
...     "mode": "sleep_longform",
...     "tracks": [{"id": "A", "source": "a.wav", "chapter_seconds": 60}],
...     "visual": {"clips": ["c.mp4"]}})
>>> spec.spectrum.enabled, spec.chime.enabled
(False, False)
>>> MusicEpisodeSpec.from_dict(spec.to_dict()) == spec
True
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

from moviepy.ae.templates._audio_io import audio_info
from moviepy.ae.templates.music_audio import (
    DEFAULT_RATE,
    assemble_chapters,
    chime_tone,
    mix_cues,
    normalize_loudness,
    write_chime,
)
from moviepy.ae.templates.music_render import (
    frame_count,
    preview_window,
    render_music_video,
)
from moviepy.ae.templates.spectrum import (
    SpectrumLayer,
    analyze_spectrum,
    load_levels,
    save_levels,
)
from moviepy.ae.templates.study import StudySchedule
from moviepy.ae.templates.visual_loop import export_loop, scene_cycle, seamless_loop


__all__ = [
    "MusicEpisodeError",
    "TrackSpec",
    "MusicAudioSpec",
    "VisualSpec",
    "SpectrumSpec",
    "ChimeSpec",
    "MusicEpisodeSpec",
    "MODES",
    "STEPS",
    "pomodoro_schedule",
    "music_paths",
    "validate_music_episode",
    "prepare_audio",
    "prepare_visual",
    "prepare_overlays",
    "render",
    "spectrum_overlay",
    "build_music_episode",
    "init_music_episode",
    "main",
]

MODES = ("sleep_longform", "study_pomodoro")
STEPS = ("audio", "visual", "overlays", "render")
ENCODERS = ("auto", "libx264", "h264_nvenc", "hevc_nvenc", "libx265")
QUALITIES = ("high", "balanced", "draft")
HUMAN_GATES = {
    "full_listening": "NOT_RUN",
    "full_visual_watch": "NOT_RUN",
    "rights": "NOT_RUN",
    "private_upload": "NOT_RUN",
}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_CHIME_KEYS = ("notes", "duration", "attack", "release", "peak_dbfs")


class MusicEpisodeError(ValueError):
    """A music episode specification, input or build step is invalid."""


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #


def _num(value, name, *, low=None, high=None, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MusicEpisodeError(f"{name} must be a number, got {value!r}")
    if integer and (isinstance(value, float) and not value.is_integer()):
        raise MusicEpisodeError(f"{name} must be a whole number, got {value!r}")
    value = int(value) if integer else float(value)
    if not math.isfinite(value):
        raise MusicEpisodeError(f"{name} must be finite")
    if positive and value <= 0:
        raise MusicEpisodeError(f"{name} must be positive")
    if low is not None and value < low:
        raise MusicEpisodeError(f"{name} must be >= {low}")
    if high is not None and value > high:
        raise MusicEpisodeError(f"{name} must be <= {high}")
    return value


def _str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise MusicEpisodeError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _bool(value, name):
    if not isinstance(value, bool):
        raise MusicEpisodeError(f"{name} must be true or false, got {value!r}")
    return value


def _choice(value, name, options):
    if value not in options:
        raise MusicEpisodeError(f"{name} must be one of {list(options)}, got {value!r}")
    return value


def _pair(value, name, **kw):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MusicEpisodeError(f"{name} must be a pair of numbers, got {value!r}")
    return tuple(_num(v, f"{name}[{i}]", **kw) for i, v in enumerate(value))


def _set(obj, **values):
    for key, value in values.items():
        object.__setattr__(obj, key, value)


def _make(cls, data, label):
    """Build dataclass ``cls`` from a mapping; unknown/missing keys raise."""
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        raise MusicEpisodeError(f"{label} must be an object")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise MusicEpisodeError(f"{label}: unknown key(s) {unknown}")
    missing = [
        f.name
        for f in fields(cls)
        if f.default is MISSING and f.default_factory is MISSING and f.name not in data
    ]
    if missing:
        raise MusicEpisodeError(f"{label}: missing required key(s) {missing}")
    try:
        return cls(**data)
    except MusicEpisodeError as error:
        raise MusicEpisodeError(f"{label}: {error}") from None


def _plain(value):
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


# --------------------------------------------------------------------------- #
# the specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrackSpec:
    """One music master that becomes one chapter of the bed.

    Parameters
    ----------
    id : str
        File-name safe id (``S01``); used for the normalized master.
    source : str
        Audio file (any format FFmpeg reads).
    chapter_seconds : float, optional
        Prepared chapter length (the master is looped to it). Required in
        sleep mode; in study mode either all tracks give it or none do (then
        the chapters share the schedule length equally).
    title : str, optional
        Chapter title in the sleep ffmetadata; defaults to ``id``.
    """

    id: str
    source: str
    chapter_seconds: float = None
    title: str = None

    def __post_init__(self):
        _str(self.id, "id")
        if not _SAFE_NAME.match(self.id):
            raise MusicEpisodeError(
                f"id must be a file-name safe word, got {self.id!r}"
            )
        _str(self.source, "source")
        if self.chapter_seconds is not None:
            _set(
                self,
                chapter_seconds=_num(
                    self.chapter_seconds, "chapter_seconds", positive=True
                ),
            )
        if self.title is not None:
            _str(self.title, "title")


@dataclass(frozen=True)
class MusicAudioSpec:
    """Loudness and assembly parameters (S14/S06 recipe by default).

    Parameters
    ----------
    target_lufs, true_peak, lra : float
        ``loudnorm`` targets: -18 LUFS, -1.8 dBTP, LRA 11.
    loop_overlap : float
        Crossfade between loop repeats inside a chapter (10 s).
    chapter_crossfade : float
        Crossfade between chapters (12 s).
    edge_fade : float
        Fade-in at the start and fade-out at the end (4 s).
    """

    target_lufs: float = -18.0
    true_peak: float = -1.8
    lra: float = 11.0
    loop_overlap: float = 10.0
    chapter_crossfade: float = 12.0
    edge_fade: float = 4.0

    def __post_init__(self):
        _set(
            self,
            target_lufs=_num(self.target_lufs, "target_lufs", low=-70, high=-5),
            true_peak=_num(self.true_peak, "true_peak", high=0),
            lra=_num(self.lra, "lra", positive=True),
            loop_overlap=_num(self.loop_overlap, "loop_overlap", positive=True),
            chapter_crossfade=_num(
                self.chapter_crossfade, "chapter_crossfade", positive=True
            ),
            edge_fade=_num(self.edge_fade, "edge_fade", low=0),
        )
        if self.edge_fade > self.chapter_crossfade:
            raise MusicEpisodeError("edge_fade must be <= chapter_crossfade")


@dataclass(frozen=True)
class VisualSpec:
    """The picture: Flow clips -> seamless loops -> a scene cycle.

    Parameters
    ----------
    clips : sequence of str
        Source clips (one scene each, in order).
    clip_length : float
        Seconds of footage used per clip (8, the Flow clip length).
    loop_crossfade : float
        Tail-to-head dissolve that makes a clip loop (2 s).
    segment : float
        Seconds each scene occupies in the cycle (120).
    scene_crossfade : float
        Dissolve between scenes (2 s).
    cycle_path : str, optional
        An already prepared cycle MP4 to reuse instead of building one.
    """

    clips: tuple = ()
    clip_length: float = 8.0
    loop_crossfade: float = 2.0
    segment: float = 120.0
    scene_crossfade: float = 2.0
    cycle_path: str = None

    def __post_init__(self):
        if isinstance(self.clips, (str, bytes)) or not isinstance(
            self.clips, (list, tuple)
        ):
            raise MusicEpisodeError("clips must be a list of paths")
        _set(
            self,
            clips=tuple(_str(c, f"clips[{i}]") for i, c in enumerate(self.clips)),
            clip_length=_num(self.clip_length, "clip_length", positive=True),
            loop_crossfade=_num(self.loop_crossfade, "loop_crossfade", positive=True),
            segment=_num(self.segment, "segment", positive=True),
            scene_crossfade=_num(
                self.scene_crossfade, "scene_crossfade", positive=True
            ),
        )
        if self.cycle_path is not None:
            _str(self.cycle_path, "cycle_path")
        if not self.clips and self.cycle_path is None:
            raise MusicEpisodeError("give clips or a prepared cycle_path")
        if self.clip_length <= 2 * self.loop_crossfade:
            raise MusicEpisodeError("clip_length must exceed 2 * loop_crossfade")
        if self.segment <= self.scene_crossfade:
            raise MusicEpisodeError("segment must exceed scene_crossfade")


@dataclass(frozen=True)
class SpectrumSpec:
    """Real-audio spectrum bars (S13 ``s13-spectrum`` defaults).

    Parameters
    ----------
    enabled : bool, optional
        ``None`` means on for study mode and off for sleep mode.
    position : sequence of int
        Top-left corner on the picture ((96, 548) on 1280x720).
    size : sequence of int
        Layer pixel size (1088, 140).
    opacity : float
        Layer opacity in percent (30).
    bands, fft, fmin, fmax, bar_width, pitch, attack, release
        ``analyze_spectrum`` / ``SpectrumLayer`` parameters: 64 log bands
        45 Hz - 10 kHz, FFT 8192, bars 12 px every 17 px, attack 0.55,
        release 0.14.
    """

    enabled: bool = None
    position: tuple = (96, 548)
    size: tuple = (1088, 140)
    opacity: float = 30.0
    bands: int = 64
    fft: int = 8192
    fmin: float = 45.0
    fmax: float = 10000.0
    bar_width: int = 12
    pitch: int = 17
    attack: float = 0.55
    release: float = 0.14

    def __post_init__(self):
        if self.enabled is not None:
            _bool(self.enabled, "enabled")
        _set(
            self,
            position=_pair(self.position, "position", low=0, integer=True),
            size=_pair(self.size, "size", low=1, integer=True),
            opacity=_num(self.opacity, "opacity", low=0, high=100),
            bands=_num(self.bands, "bands", low=1, integer=True),
            fft=_num(self.fft, "fft", low=16, integer=True),
            fmin=_num(self.fmin, "fmin", positive=True),
            fmax=_num(self.fmax, "fmax", positive=True, high=24000),
            bar_width=_num(self.bar_width, "bar_width", low=1, integer=True),
            pitch=_num(self.pitch, "pitch", low=1, integer=True),
            attack=_num(self.attack, "attack", low=0, high=1),
            release=_num(self.release, "release", low=0, high=1),
        )
        if self.fmin >= self.fmax:
            raise MusicEpisodeError("fmin must be below fmax")
        if self.pitch < self.bar_width:
            raise MusicEpisodeError("pitch must be >= bar_width")


@dataclass(frozen=True)
class ChimeSpec:
    r"""Phase-boundary cue for study mode (S14 ``CHIME-DESIGN.json``).

    Parameters
    ----------
    enabled : bool, optional
        ``None`` means on for study mode and off for sleep mode.
    below_music_db : float
        Cue RMS below the local music RMS (10 dB); never boosted.
    path : str, optional
        Existing stereo 48 kHz cue file; excludes ``design``.
    design : dict
        ``chime_tone`` keywords (``notes`` as ``[name, hz, start, length]``
        rows, ``duration``, ``attack``, ``release``, ``peak_dbfs``); empty
        means the S14 two-note D4/A4 bell (2.05 s, peak -30 dBFS).
    """

    enabled: bool = None
    below_music_db: float = 10.0
    path: str = None
    design: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.enabled is not None:
            _bool(self.enabled, "enabled")
        _set(self, below_music_db=_num(self.below_music_db, "below_music_db", low=0))
        if self.path is not None:
            _str(self.path, "path")
        if not isinstance(self.design, dict):
            raise MusicEpisodeError("design must be an object")
        unknown = sorted(set(self.design) - set(_CHIME_KEYS))
        if unknown:
            raise MusicEpisodeError(f"design: unknown key(s) {unknown}")
        if self.path and self.design:
            raise MusicEpisodeError("give a cue path or a design, not both")
        _set(self, design=json.loads(json.dumps(self.design)))

    def kwargs(self):
        """Return the ``chime_tone`` keyword arguments of ``design``."""
        values = dict(self.design)
        if "notes" in values:
            values["notes"] = tuple(tuple(row) for row in values["notes"])
        return values


@dataclass(frozen=True)
class MusicEpisodeSpec:
    """A whole music-channel episode as data.

    Parameters
    ----------
    mode : str
        ``"sleep_longform"`` or ``"study_pomodoro"``.
    name : str
        File-name safe episode name (output file names).
    size : sequence of int
        Output ``(width, height)`` (even numbers); default 1280x720.
    fps : float
        Output frame rate (24).
    tracks : sequence of TrackSpec
        Music masters, one chapter each.
    audio, visual, spectrum, chime
        Section specs (objects or dicts); see ``MusicAudioSpec``,
        ``VisualSpec``, ``SpectrumSpec`` and ``ChimeSpec``.
    study : str or dict, optional
        STUDY-MODE style document (path or inline); required for study mode.
    study_compact : bool
        Use the compact ASS layout (S14 used the non-compact one).
    chapters : bool
        Attach ffmetadata chapters (sleep: from the tracks, study: from the
        schedule).
    encoder, quality, workers, audio_bitrate
        Video encoder choice, quality, render processes (``None`` = all
        cores) and AAC bitrate.
    output_dir : str
        Folder for ``audio/``, ``visual/``, ``overlays/`` and ``render/``.
    """

    mode: str
    name: str = "music_episode"
    size: tuple = (1280, 720)
    fps: float = 24
    tracks: tuple = ()
    audio: MusicAudioSpec = None
    visual: VisualSpec = None
    spectrum: SpectrumSpec = None
    study: object = None
    study_compact: bool = False
    chime: ChimeSpec = None
    chapters: bool = True
    encoder: str = "auto"
    quality: str = "high"
    workers: int = None
    audio_bitrate: str = "256k"
    output_dir: str = "build"

    def __post_init__(self):
        _choice(self.mode, "mode", MODES)
        study_mode = self.mode == "study_pomodoro"
        if not _SAFE_NAME.match(_str(self.name, "name")):
            raise MusicEpisodeError(
                f"name must be a file-name safe word: {self.name!r}"
            )
        size = _pair(self.size, "size", low=2, integer=True)
        if size[0] % 2 or size[1] % 2:
            raise MusicEpisodeError(f"size must be even for yuv420p, got {size}")
        fps = _num(self.fps, "fps", positive=True)
        if not isinstance(self.tracks, (list, tuple)) or not self.tracks:
            raise MusicEpisodeError("tracks must be a non-empty list")
        tracks = tuple(
            _make(TrackSpec, t, f"tracks[{i}]") for i, t in enumerate(self.tracks)
        )
        ids = [t.id for t in tracks]
        if len(set(ids)) != len(ids):
            raise MusicEpisodeError(f"tracks: duplicate ids in {ids}")
        audio = _make(MusicAudioSpec, self.audio or {}, "audio")
        visual = _make(VisualSpec, self.visual or {}, "visual")
        spectrum = _make(SpectrumSpec, self.spectrum or {}, "spectrum")
        chime = _make(ChimeSpec, self.chime or {}, "chime")
        if spectrum.enabled is None:
            spectrum = dataclasses.replace(spectrum, enabled=study_mode)
        if chime.enabled is None:
            chime = dataclasses.replace(chime, enabled=study_mode)
        given = [t.chapter_seconds is not None for t in tracks]
        if not study_mode and not all(given):
            raise MusicEpisodeError("tracks: every track needs chapter_seconds")
        if study_mode and any(given) and not all(given):
            raise MusicEpisodeError(
                "tracks: give chapter_seconds on every track or on none"
            )
        for track in tracks:
            if (
                track.chapter_seconds is not None
                and track.chapter_seconds < 2 * audio.chapter_crossfade
            ):
                raise MusicEpisodeError(
                    f"tracks: {track.id}.chapter_seconds must be at least twice "
                    "audio.chapter_crossfade"
                )
        if study_mode:
            if self.study is None:
                raise MusicEpisodeError("study is required in study_pomodoro mode")
            if isinstance(self.study, dict):
                try:
                    StudySchedule.from_dict(self.study)
                except (ValueError, KeyError, TypeError) as error:
                    raise MusicEpisodeError(f"study: {error}") from None
                _set(self, study=json.loads(json.dumps(self.study)))
            else:
                _str(self.study, "study")
        elif self.study is not None:
            raise MusicEpisodeError("study is only valid in study_pomodoro mode")
        if chime.enabled:
            if not study_mode:
                raise MusicEpisodeError("chime.enabled needs study_pomodoro mode")
            if not chime.path:
                try:
                    chime_tone(**chime.kwargs())
                except (ValueError, TypeError) as error:
                    raise MusicEpisodeError(f"chime.design: {error}") from None
        if spectrum.enabled:
            x, y = spectrum.position
            w, h = spectrum.size
            if x + w > size[0] or y + h > size[1]:
                raise MusicEpisodeError(
                    f"spectrum: position {spectrum.position} + size "
                    f"{spectrum.size} exceeds the {size[0]}x{size[1]} picture"
                )
        for step, value in (
            (
                "clip_length - loop_crossfade",
                visual.clip_length - visual.loop_crossfade,
            ),
            ("segment", visual.segment),
        ):
            try:
                frame_count(value, fps)
            except ValueError:
                raise MusicEpisodeError(
                    f"visual: {step} ({value:g} s) is not a whole number of frames "
                    f"at {fps:g} fps"
                ) from None
        _choice(self.encoder, "encoder", ENCODERS)
        _choice(self.quality, "quality", QUALITIES)
        if self.workers is not None:
            _num(self.workers, "workers", low=1, integer=True)
        if not re.match(r"^[1-9][0-9]*k$", str(self.audio_bitrate)):
            raise MusicEpisodeError(
                f"audio_bitrate must look like '256k', got {self.audio_bitrate!r}"
            )
        _bool(self.chapters, "chapters")
        _bool(self.study_compact, "study_compact")
        _str(self.output_dir, "output_dir")
        _set(
            self,
            size=size,
            fps=fps,
            tracks=tracks,
            audio=audio,
            visual=visual,
            spectrum=spectrum,
            chime=chime,
        )

    # -- JSON ---------------------------------------------------------------- #

    def to_dict(self):
        """Return a JSON-compatible dictionary (``from_dict`` inverse)."""
        return {f.name: _plain(_asdict(getattr(self, f.name))) for f in fields(self)}

    @classmethod
    def from_dict(cls, data, base_dir=None):
        """Build a spec from ``to_dict`` output.

        Parameters
        ----------
        data : dict
            The document; unknown keys raise ``MusicEpisodeError``.
        base_dir : str or Path, optional
            Folder against which relative media, study, cue and output paths
            are resolved.

        Returns
        -------
        MusicEpisodeSpec
            The validated spec.
        """
        if not isinstance(data, dict):
            raise MusicEpisodeError("a music episode must be a JSON object")
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise MusicEpisodeError(f"unknown music episode key(s): {unknown}")
        data = json.loads(json.dumps(data))
        if base_dir is not None:
            _resolve_paths(data, Path(base_dir))
        return cls(**data)

    def to_json(self, path=None):
        """Serialize to JSON text, also writing UTF-8 ``path`` when given."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source, base_dir=None):
        """Load from a JSON file path or JSON text.

        A file's relative paths are resolved against its folder; for JSON
        text pass ``base_dir`` (otherwise paths stay as written).
        """
        if isinstance(source, Path) or (
            isinstance(source, str) and not source.lstrip().startswith("{")
        ):
            path = Path(source)
            try:
                text = path.read_text(encoding="utf-8-sig")
            except OSError as error:
                raise MusicEpisodeError(f"cannot read {path}: {error}") from None
            base_dir = path.parent if base_dir is None else base_dir
        else:
            text = source
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise MusicEpisodeError(f"invalid music episode JSON: {error}") from None
        return cls.from_dict(data, base_dir)

    def schedule(self):
        """Return the study ``StudySchedule`` (study mode only)."""
        if self.mode != "study_pomodoro":
            raise MusicEpisodeError("only study_pomodoro mode has a schedule")
        try:
            if isinstance(self.study, dict):
                return StudySchedule.from_dict(self.study)
            return StudySchedule.from_json(self.study)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise MusicEpisodeError(f"study: {error}") from None


def _asdict(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _asdict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_asdict(v) for v in value]
    return value


def _resolve_paths(data, base):
    def fix(value):
        if not isinstance(value, str) or not value:
            return value
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else base / path)

    for track in data.get("tracks") or ():
        if isinstance(track, dict) and "source" in track:
            track["source"] = fix(track["source"])
    visual = data.get("visual")
    if isinstance(visual, dict):
        if isinstance(visual.get("clips"), list):
            visual["clips"] = [fix(c) for c in visual["clips"]]
        if "cycle_path" in visual:
            visual["cycle_path"] = fix(visual["cycle_path"])
    chime = data.get("chime")
    if isinstance(chime, dict) and "path" in chime:
        chime["path"] = fix(chime["path"])
    if isinstance(data.get("study"), str):
        data["study"] = fix(data["study"])
    if "output_dir" in data:
        data["output_dir"] = fix(data["output_dir"])


# --------------------------------------------------------------------------- #
# study schedule helper
# --------------------------------------------------------------------------- #


def pomodoro_schedule(
    episode="Study Episode",
    *,
    title=("學習陪伴", "Study Companion", "学習のおとも"),
    focus=1500,
    rest=300,
    rounds=3,
    closing=120,
):
    """Return a STUDY-MODE style document with S14's rhythm and display.

    ``rounds`` focus blocks are separated by breaks, and the last "break" is
    the closing block, so the defaults give S14's 5220 s with chimes at 1500,
    1800, 3300, 3600 and 5100 s. Labels and tips are neutral trilingual text.

    Parameters
    ----------
    episode : str
        Episode title written into the ffmetadata.
    title : tuple of str
        Chinese, English and Japanese panel title.
    focus, rest, closing : int
        Seconds per focus block, break and closing block.
    rounds : int
        Number of focus blocks.

    Returns
    -------
    dict
        A document accepted by ``StudySchedule.from_dict``.

    Examples
    --------
    >>> schedule = StudySchedule.from_dict(pomodoro_schedule())
    >>> schedule.duration, schedule.chime_times()
    (5220, [1500, 1800, 3300, 3600, 5100])
    """
    minutes = rest // 60
    phases, chapters, cursor = [], [], 0
    for index in range(1, rounds + 1):
        last = index == rounds
        phases.append(
            {
                "id": f"F{index}",
                "type": "focus",
                "start": cursor,
                "end": cursor + focus,
                "label": f"專注 {index} / {rounds}",
                "label_en": f"Focus {index} / {rounds}",
                "label_ja": f"集中 {index} / {rounds}",
                "tip": "這一輪，先完成一件小事。",
                "tip_en": "Choose one small task for this round.",
                "tip_ja": "この時間は、小さな課題を一つ。",
            }
        )
        chapters.append({"start": cursor, "name": f"專注 {index}"})
        cursor += focus
        length = closing if last else rest
        phases.append(
            {
                "id": f"B{index}",
                "type": "closing" if last else "break",
                "start": cursor,
                "end": cursor + length,
                "label": "收尾" if last else f"休息 {minutes} 分鐘",
                "label_en": "Wrap up" if last else f"Break {minutes} minutes",
                "label_ja": "振り返り" if last else f"休憩 {minutes} 分",
                "tip": (
                    "記下下次要複習的一個重點。" if last else "抬頭看看遠處，喝口水。"
                ),
                "tip_en": (
                    "Note one key point to review next time."
                    if last
                    else "Look into the distance. Take a sip of water."
                ),
                "tip_ja": (
                    "次に復習する要点を一つ書こう。"
                    if last
                    else "遠くを眺めて、水を一口。"
                ),
            }
        )
        chapters.append({"start": cursor, "name": "收尾" if last else "休息"})
        cursor += length
    return {
        "version": 2,
        "episode": episode,
        "mode": "study_companion",
        "duration_seconds": cursor,
        "fps": 24,
        "focus_seconds": focus * rounds,
        "rest_seconds": cursor - focus * rounds,
        "display": {
            "width": 1280,
            "height": 720,
            "timer_anchor": "upper_right",
            "timer_font": "Consolas",
            "zh_font": "Microsoft JhengHei",
            "ja_font": "Yu Gothic",
            "timer_panel_opacity": 0.3,
            "title_panel_opacity": 0.3,
            "tip_panel_opacity": 0.3,
            "font_sizes": {
                "title_zh": 30,
                "title_en": 26,
                "title_ja": 26,
                "phase_zh": 35,
                "phase_en": 27,
                "phase_ja": 27,
                "timer": 68,
                "tip_zh": 37,
                "tip_en": 29,
                "tip_ja": 29,
            },
            "tip_display_seconds": 20,
            "no_flashing": True,
        },
        "title": {"zh": title[0], "en": title[1], "ja": title[2]},
        "phases": phases,
        "study_chapters": chapters,
    }


# --------------------------------------------------------------------------- #
# paths, hashing and the resumable-step helper
# --------------------------------------------------------------------------- #


def music_paths(spec):
    """Return every artifact path of ``spec`` as a dictionary of strings.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.

    Returns
    -------
    dict
        Keys ``masters`` (list), ``music``, ``chime``, ``audio`` (the final
        audio), ``loops`` (list), ``cycle`` (the background), ``levels``,
        ``ass``, ``chapters`` and ``video``.
    """
    root = Path(spec.output_dir)
    study = spec.mode == "study_pomodoro"
    music = root / "audio" / "music.flac"
    mixed = study and spec.chime.enabled
    return {
        "masters": [
            str(root / "audio" / "masters" / f"{t.id}.flac") for t in spec.tracks
        ],
        "music": str(music),
        "chime": (
            str(spec.chime.path or root / "audio" / "chime.wav") if mixed else None
        ),
        "audio": str(root / "audio" / "audio.flac") if mixed else str(music),
        "loops": [
            str(root / "visual" / "loops" / f"loop-{i + 1:02}.mp4")
            for i in range(len(spec.visual.clips))
        ],
        "cycle": str(spec.visual.cycle_path or root / "visual" / "cycle.mp4"),
        "levels": (
            str(root / "overlays" / "levels.npy") if spec.spectrum.enabled else None
        ),
        "ass": str(root / "overlays" / "study.ass") if study else None,
        "chapters": (
            str(root / "overlays" / "chapters.ffmeta") if spec.chapters else None
        ),
        "video": str(root / "render" / f"{spec.name}.mp4"),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    text = json.dumps(value, sort_keys=True, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence_path(path):
    return Path(str(path) + ".json")


def _retarget(value, old, new):
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, dict):
        return {k: _retarget(v, old, new) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_retarget(v, old, new) for v in value]
    return value


def _resumable(path, config, make, *, kind="audio"):
    """Create ``path`` with ``make(partial_path)`` or verify and skip it.

    The artifact is built under a ``.partial`` name and renamed only when
    complete, so an interrupted run never leaves a believable half file. The
    evidence ``<path>.json`` (exclusive create) stores the SHA-256 and a hash
    of ``config``; an existing artifact is skipped only if both still match.
    """
    path = Path(path)
    evidence = _evidence_path(path)
    config_hash = _digest(config)
    if path.exists():
        if not evidence.is_file():
            raise MusicEpisodeError(
                f"{path} exists without evidence; remove it to rebuild"
            )
        recorded = json.loads(evidence.read_text(encoding="utf-8"))
        if recorded.get("config_sha256") != config_hash:
            raise MusicEpisodeError(
                f"{path} was built from different settings or inputs; "
                "remove it (and later artifacts) to rebuild"
            )
        if recorded.get("sha256") != _sha256(path):
            raise MusicEpisodeError(f"{path} no longer matches its recorded hash")
        recorded["skipped"] = True
        return recorded
    if evidence.exists():
        raise MusicEpisodeError(f"{evidence} exists but {path} is missing")
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    try:
        facts = make(partial)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    if path.exists():
        partial.unlink()
        raise FileExistsError(path)
    os.replace(partial, path)
    facts = _retarget(dict(facts), str(partial), str(path))
    facts.update(
        path=str(path),
        sha256=_sha256(path),
        bytes=path.stat().st_size,
        config_sha256=config_hash,
    )
    facts.setdefault(
        "human_visual" if kind == "visual" else "human_listening", "NOT_RUN"
    )
    with open(evidence, "x", encoding="utf-8") as stream:
        json.dump(facts, stream, ensure_ascii=False, indent=2, default=_json_default)
    facts["skipped"] = False
    return facts


def _require_file(path, label):
    if not Path(path).is_file():
        raise MusicEpisodeError(f"{label} not found: {path}")
    return Path(path)


def _file_config(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size}


# --------------------------------------------------------------------------- #
# step 1: audio
# --------------------------------------------------------------------------- #


def _audio_plan(spec):
    """Return chapter frames and the final length, checking all contracts."""
    rate = DEFAULT_RATE
    cross = round(spec.audio.chapter_crossfade * rate)
    count = len(spec.tracks)
    study = spec.mode == "study_pomodoro"
    expected = None
    if study:
        expected = spec.schedule().duration * rate
    if all(t.chapter_seconds is None for t in spec.tracks):
        whole = expected + (count - 1) * cross
        lengths = [whole // count] * count
        lengths[-1] += whole - sum(lengths)
    else:
        lengths = [round(t.chapter_seconds * rate) for t in spec.tracks]
    if any(length < 2 * cross for length in lengths):
        raise MusicEpisodeError(
            "tracks: every chapter must be at least twice audio.chapter_crossfade"
        )
    total = sum(lengths) - (count - 1) * cross
    if study and total != expected:
        raise MusicEpisodeError(
            f"audio length {total / rate:g} s != study schedule "
            f"{expected / rate:g} s; chapter_seconds must sum to duration + "
            f"(n-1) * chapter_crossfade = {(expected + (count - 1) * cross) / rate:g} s"
        )
    try:
        video_frames = frame_count(total / rate, spec.fps)
    except ValueError:
        raise MusicEpisodeError(
            f"audio length {total / rate:.6f} s is not a whole number of frames "
            f"at {spec.fps:g} fps; adjust chapter_seconds"
        ) from None
    return {
        "rate": rate,
        "chapter_frames": lengths,
        "chapter_seconds": [length / rate for length in lengths],
        "total_frames": total,
        "total_seconds": total / rate,
        "video_frames": video_frames,
        "expected_frames": expected,
    }


def prepare_audio(spec):
    """Normalize, assemble and (study) chime-mix the audio; resumable.

    Writes ``audio/masters/<id>.flac`` (two-pass linear loudnorm to the
    target with an exact-frame contract), ``audio/music.flac`` (looped
    chapters joined with crossfades and edge fades), in study mode with chimes
    ``audio/chime.wav`` and ``audio/audio.flac`` (chimes mixed at the phase
    boundaries, ``below_music_db`` under the local music, no ducking), each
    with an exclusive-create ``<file>.json`` evidence. Existing artifacts are
    skipped after their hash and settings are re-verified.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.

    Returns
    -------
    dict
        ``masters``, ``music`` (with the cue sheet), ``chime``, ``mix``,
        ``audio`` (final audio path), ``frames``, ``seconds`` and
        ``video_frames``; every artifact entry has ``skipped``.

    Raises
    ------
    MusicEpisodeError
        For missing sources, a length that disagrees with the schedule or is
        not frame-aligned, or artifacts that no longer match their evidence.
    """
    plan = _audio_plan(spec)
    paths = music_paths(spec)
    audio = spec.audio
    masters = []
    for track, target in zip(spec.tracks, paths["masters"]):
        source = _require_file(track.source, f"track {track.id} source")
        config = {
            "source": _file_config(source),
            "lufs": audio.target_lufs,
            "tp": audio.true_peak,
            "lra": audio.lra,
        }

        def normalize(partial, source=source):
            return normalize_loudness(
                source,
                partial,
                target_lufs=audio.target_lufs,
                true_peak=audio.true_peak,
                lra=audio.lra,
            )

        masters.append(_resumable(target, config, normalize))

    config = {
        "masters": [m["sha256"] for m in masters],
        "ids": [t.id for t in spec.tracks],
        "seconds": plan["chapter_seconds"],
        "loop_overlap": audio.loop_overlap,
        "chapter_crossfade": audio.chapter_crossfade,
        "edge_fade": audio.edge_fade,
    }

    def assemble(partial):
        sheet = assemble_chapters(
            paths["masters"],
            plan["chapter_seconds"],
            partial,
            rate=plan["rate"],
            loop_overlap=audio.loop_overlap,
            chapter_crossfade=audio.chapter_crossfade,
            edge_fade=audio.edge_fade,
            ids=[t.id for t in spec.tracks],
        )
        if sheet["total_frames"] != plan["total_frames"]:
            raise MusicEpisodeError("assembled audio disagrees with the planned length")
        sheet["titles"] = [t.title or t.id for t in spec.tracks]
        return sheet

    music = _resumable(paths["music"], config, assemble)
    report = {"masters": masters, "music": music, "chime": None, "mix": None}
    if paths["chime"] is not None:
        schedule = spec.schedule()
        cue_path = paths["chime"]
        if spec.chime.path:
            _require_file(cue_path, "chime path")
            cue = {"path": cue_path, "external": True}
        else:
            cue = _resumable(
                cue_path,
                {"design": spec.chime.design},
                lambda partial: write_chime(partial, **spec.chime.kwargs()),
            )
        times = schedule.chime_times()
        config = {
            "music": music["sha256"],
            "cue": cue.get("sha256") or _sha256(cue_path),
            "times": times,
            "below_music_db": spec.chime.below_music_db,
        }

        def mix(partial):
            facts = mix_cues(
                paths["music"],
                cue_path,
                times,
                partial,
                below_music_db=spec.chime.below_music_db,
            )
            if facts["frames"] != plan["total_frames"]:
                raise MusicEpisodeError("chime mix changed the audio length")
            return facts

        report["chime"] = cue
        report["mix"] = _resumable(paths["audio"], config, mix)
    info = audio_info(paths["audio"])
    if info["frames"] != plan["total_frames"]:
        raise MusicEpisodeError(
            f"final audio has {info['frames']} frames, expected {plan['total_frames']}"
        )
    report.update(
        audio=paths["audio"],
        frames=info["frames"],
        seconds=info["frames"] / info["rate"],
        video_frames=plan["video_frames"],
    )
    return report


# --------------------------------------------------------------------------- #
# step 2: visual
# --------------------------------------------------------------------------- #


def _loop_factory(clip, length, crossfade, fps, size):
    """Return the seamless loop composition of one clip (picklable factory)."""
    return seamless_loop(
        clip, length=length, crossfade=crossfade, fps=fps, size=tuple(size)
    )


def _cycle_factory(loops, segment, crossfade, fps, size):
    """Return the scene-cycle composition of exported loops (picklable)."""
    return scene_cycle(
        list(loops), segment=segment, crossfade=crossfade, fps=fps, size=tuple(size)
    )


def prepare_visual(spec):
    """Export one seamless loop per clip and the scene cycle; resumable.

    Writes ``visual/loops/loop-NN.mp4`` (``seamless_loop`` of the first
    ``clip_length`` seconds, ``loop_crossfade`` dissolve) and
    ``visual/cycle.mp4`` (``scene_cycle``, ``segment`` seconds per scene,
    ``scene_crossfade`` dissolves, also from the last scene back to the first),
    both via ``export_loop`` (closed 1 s GOP, no B-frames) with evidence. With
    ``visual.cycle_path`` the prepared cycle is only verified and hashed into
    ``visual/cycle.external.json``.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.

    Returns
    -------
    dict
        ``loops`` (list), ``cycle`` and ``background`` (the cycle path).
    """
    paths = music_paths(spec)
    visual = spec.visual
    size = list(spec.size)
    if visual.cycle_path:
        cycle_file = _require_file(visual.cycle_path, "visual.cycle_path")
        evidence = Path(spec.output_dir) / "visual" / "cycle.external.json"
        digest = _sha256(cycle_file)
        if evidence.is_file():
            recorded = json.loads(evidence.read_text(encoding="utf-8"))
            if recorded.get("sha256") != digest:
                raise MusicEpisodeError(f"{cycle_file} changed since it was recorded")
            recorded["skipped"] = True
            return {"loops": [], "cycle": recorded, "background": str(cycle_file)}
        facts = {
            "path": str(cycle_file),
            "sha256": digest,
            "bytes": cycle_file.stat().st_size,
            "external": True,
            "human_visual": "NOT_RUN",
        }
        evidence.parent.mkdir(parents=True, exist_ok=True)
        with open(evidence, "x", encoding="utf-8") as stream:
            json.dump(facts, stream, ensure_ascii=False, indent=2)
        facts["skipped"] = False
        return {"loops": [], "cycle": facts, "background": str(cycle_file)}

    loops = []
    for clip, target in zip(visual.clips, paths["loops"]):
        source = _require_file(clip, "visual clip")
        config = {
            "source": _file_config(source),
            "length": visual.clip_length,
            "crossfade": visual.loop_crossfade,
            "fps": spec.fps,
            "size": size,
            "encoder": spec.encoder,
            "quality": spec.quality,
        }

        def export(partial, source=source):
            comp = _loop_factory(
                str(source),
                visual.clip_length,
                visual.loop_crossfade,
                spec.fps,
                size,
            )
            try:
                return export_loop(
                    comp, partial, encoder=spec.encoder, quality=spec.quality
                )
            finally:
                comp.close()

        loops.append(_resumable(target, config, export, kind="visual"))
    config = {
        "loops": [loop["sha256"] for loop in loops],
        "segment": visual.segment,
        "crossfade": visual.scene_crossfade,
        "fps": spec.fps,
        "size": size,
        "encoder": spec.encoder,
        "quality": spec.quality,
    }

    def export_cycle(partial):
        args = (
            tuple(paths["loops"]),
            visual.segment,
            visual.scene_crossfade,
            spec.fps,
            tuple(size),
        )
        return export_loop(
            (_cycle_factory, args),
            partial,
            encoder=spec.encoder,
            quality=spec.quality,
            workers=spec.workers,
        )

    cycle = _resumable(paths["cycle"], config, export_cycle, kind="visual")
    return {"loops": loops, "cycle": cycle, "background": paths["cycle"]}


# --------------------------------------------------------------------------- #
# step 3: overlays
# --------------------------------------------------------------------------- #


def _ffmeta_title(text):
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("\n", " ")
    )


def _sleep_ffmetadata(spec, sheet):
    """Return ffmetadata chapters from an assembly cue sheet."""
    chapters = sheet["chapters"]
    titles = sheet.get("titles") or [c["id"] for c in chapters]
    total_ms = round(sheet["total_frames"] * 1000 / sheet["rate"])
    lines = [";FFMETADATA1", f"title={_ffmeta_title(spec.name)}"]
    for index, chapter in enumerate(chapters):
        start = round(chapter["crossfade_center_seconds"] * 1000)
        end = (
            round(chapters[index + 1]["crossfade_center_seconds"] * 1000)
            if index + 1 < len(chapters)
            else total_ms
        )
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start}",
            f"END={end}",
            f"title={_ffmeta_title(titles[index])}",
        ]
    return "\n".join(lines) + "\n"


def _text_writer(text):
    def make(partial):
        with open(partial, "xb") as stream:
            stream.write(text.encode("utf-8"))
        return {"characters": len(text)}

    return make


def prepare_overlays(spec):
    """Write the spectrum levels, the study ASS and the chapters; resumable.

    ``overlays/levels.npy`` is ``analyze_spectrum`` of the final audio (one
    row per video frame; only when ``spectrum.enabled``), ``overlays/study.ass``
    is ``StudySchedule.to_ass()`` (study mode) and
    ``overlays/chapters.ffmeta`` holds the study chapters (study) or the
    chapters of the cue sheet at the crossfade centres (sleep). Requires the
    audio step to have run.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.

    Returns
    -------
    dict
        ``levels``, ``ass`` and ``chapters`` evidence entries (``None`` when
        not applicable).
    """
    paths = music_paths(spec)
    audio_path = Path(paths["audio"])
    music_evidence = _evidence_path(paths["music"])
    if not audio_path.is_file() or not music_evidence.is_file():
        raise MusicEpisodeError("overlays need the audio step first (no final audio)")
    audio_hash = json.loads(_evidence_path(audio_path).read_text(encoding="utf-8"))[
        "sha256"
    ]
    frames = frame_count(audio_info(audio_path)["seconds"], spec.fps)
    report = {"levels": None, "ass": None, "chapters": None}
    sp = spec.spectrum
    if sp.enabled:
        options = {
            "bands": sp.bands,
            "fft": sp.fft,
            "fmin": sp.fmin,
            "fmax": sp.fmax,
            "attack": sp.attack,
            "release": sp.release,
        }

        def analyse(partial):
            levels = analyze_spectrum(audio_path, fps=spec.fps, **options)
            if len(levels) != frames:
                raise MusicEpisodeError(
                    f"spectrum has {len(levels)} rows for {frames} video frames"
                )
            save_levels(partial, levels)
            return {
                "rows": int(levels.shape[0]),
                "bands": int(levels.shape[1]),
                "fps": spec.fps,
                "audio_sha256": audio_hash,
                "options": options,
            }

        report["levels"] = _resumable(
            paths["levels"],
            {"audio": audio_hash, "fps": spec.fps, **options},
            analyse,
        )
    ffmeta = None
    if spec.mode == "study_pomodoro":
        schedule = spec.schedule()
        text = schedule.to_ass(compact=spec.study_compact)
        report["ass"] = _resumable(
            paths["ass"],
            {"ass": hashlib.sha256(text.encode("utf-8")).hexdigest()},
            _text_writer(text),
        )
        if spec.chapters:
            ffmeta = schedule.chapters_ffmetadata()
    elif spec.chapters:
        sheet = json.loads(music_evidence.read_text(encoding="utf-8"))
        ffmeta = _sleep_ffmetadata(spec, sheet)
    if ffmeta is not None:
        report["chapters"] = _resumable(
            paths["chapters"],
            {"ffmeta": hashlib.sha256(ffmeta.encode("utf-8")).hexdigest()},
            _text_writer(ffmeta),
        )
    return report


# --------------------------------------------------------------------------- #
# step 4: render
# --------------------------------------------------------------------------- #


def spectrum_overlay(levels_path, fps, size, duration, options):
    """Build the spectrum overlay composition (picklable top-level factory).

    Parameters
    ----------
    levels_path : str
        ``.npy`` written by the overlays step.
    fps : float
        Video frame rate.
    size : sequence of int
        Overlay ``(width, height)``.
    duration : float
        Composition length in seconds.
    options : dict
        ``SpectrumLayer`` keywords (``opacity``, ``bar_width``, ``pitch``).

    Returns
    -------
    Composition
        A transparent composition with one ``SpectrumLayer``.
    """
    from moviepy.ae.composition import Composition

    size = (int(size[0]), int(size[1]))
    comp = Composition(size=size, fps=fps, duration=duration, transparent=True)
    comp.add_layer(
        SpectrumLayer(load_levels(levels_path), fps=fps, size=size, **options)
    )
    return comp


def _preview_window(preview, total_seconds, fps):
    if preview is None:
        return None
    try:
        start, seconds = preview
    except (TypeError, ValueError):
        raise MusicEpisodeError("preview must be (start, seconds)") from None
    start = _num(start, "preview start", low=0)
    seconds = _num(seconds, "preview seconds", positive=True)
    for label, value in (("start", start), ("seconds", seconds)):
        try:
            frame_count(value, fps)
        except ValueError:
            raise MusicEpisodeError(
                f"preview {label} {value:g} s is not a whole number of frames "
                f"at {fps:g} fps"
            ) from None
    if start + seconds > total_seconds + 1e-6:
        raise MusicEpisodeError(
            f"preview window ends at {start + seconds:g} s, after the "
            f"{total_seconds:g} s episode"
        )
    return start, seconds


def render(spec, *, preview=None):
    """Render the final video (or a preview window); resumable.

    One FFmpeg pass through ``render_music_video``: the looped cycle,
    the spectrum overlay at ``spectrum.position``, the burned study ASS, the
    final audio as AAC and the chapters. ``preview=(start, seconds)`` renders
    ``render/<name>-preview-<start>-<seconds>.mp4`` with ``preview_window``
    (chapters only when ``start`` is 0). If the output and its evidence
    already exist the file hash is verified and the render is skipped.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.
    preview : tuple of float, optional
        ``(start, seconds)``, both whole frames at ``fps``.

    Returns
    -------
    dict
        The ``render_music_video`` evidence plus ``skipped``.

    Raises
    ------
    MusicEpisodeError
        If an earlier step's artifact is missing or the output is incomplete.
    """
    paths = music_paths(spec)
    for key in ("audio", "cycle"):
        if not Path(paths[key]).is_file():
            raise MusicEpisodeError(f"render needs the {key} artifact: {paths[key]}")
    for key in ("levels", "ass", "chapters"):
        if paths[key] is not None and not Path(paths[key]).is_file():
            raise MusicEpisodeError(f"render needs the {key} artifact: {paths[key]}")
    info = audio_info(paths["audio"])
    window = _preview_window(preview, info["seconds"], spec.fps)
    output = Path(paths["video"])
    if window is not None:
        output = output.with_name(
            f"{spec.name}-preview-{window[0]:g}-{window[1]:g}.mp4"
        )
    evidence = _evidence_path(output)
    log = Path(str(output) + ".ffmpeg.log")
    if output.exists():
        if not evidence.is_file():
            raise MusicEpisodeError(f"{output} exists without evidence; remove it")
        recorded = json.loads(evidence.read_text(encoding="utf-8"))
        if recorded.get("output_sha256") != _sha256(output):
            raise MusicEpisodeError(f"{output} no longer matches its recorded hash")
        recorded["skipped"] = True
        return recorded
    if log.exists() and not evidence.exists():
        log.unlink()  # leftover of a failed attempt of this very render
    overlay = None
    if paths["levels"] is not None:
        sp = spec.spectrum
        args = (
            paths["levels"],
            spec.fps,
            tuple(sp.size),
            info["seconds"],
            {"opacity": sp.opacity, "bar_width": sp.bar_width, "pitch": sp.pitch},
        )
        overlay = (spectrum_overlay, args)
    cycle_evidence = _evidence_path(paths["cycle"])
    loop_frames = None
    if cycle_evidence.is_file():
        loop_frames = json.loads(cycle_evidence.read_text(encoding="utf-8")).get(
            "frames"
        )
    options = dict(
        background=paths["cycle"],
        audio=paths["audio"],
        overlay=overlay,
        overlay_position=tuple(spec.spectrum.position),
        ass=paths["ass"],
        chapters=paths["chapters"],
        fps=spec.fps,
        size=tuple(spec.size),
        encoder=spec.encoder,
        quality=spec.quality,
        workers=spec.workers,
        audio_bitrate=spec.audio_bitrate,
        loop_frames=loop_frames,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if window is None:
        report = render_music_video(output, **options)
    else:
        if window[0]:
            options["chapters"] = None
        report = preview_window(output, start=window[0], seconds=window[1], **options)
    report = dict(report)
    report["skipped"] = False
    return report


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #


def validate_music_episode(spec, *, check_files=True):
    """Check a spec beyond its syntax: lengths, frame alignment, input files.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.
    check_files : bool
        Also report missing track, clip, cue, cycle and study files.

    Returns
    -------
    dict
        ``ok``, ``missing`` (files), ``problems`` (messages), ``plan`` (audio
        lengths or ``None`` when the plan could not be made) and ``paths``.
    """
    missing, problems, plan = [], [], None
    if check_files:
        wanted = [t.source for t in spec.tracks]
        # A prepared cycle replaces the clips; they are not read at all then.
        if spec.visual.cycle_path:
            wanted.append(spec.visual.cycle_path)
        else:
            wanted += list(spec.visual.clips)
        wanted += [spec.chime.path] if spec.chime.path else []
        wanted += [spec.study] if isinstance(spec.study, str) else []
        missing = [str(p) for p in wanted if not Path(p).is_file()]
    try:
        plan = _audio_plan(spec)
    except MusicEpisodeError as error:
        if not (
            isinstance(spec.study, str) and not Path(spec.study).is_file()
        ):  # a missing study file is already reported as missing
            problems.append(str(error))
    return {
        "ok": not missing and not problems,
        "missing": missing,
        "problems": problems,
        "plan": plan,
        "paths": music_paths(spec),
    }


def build_music_episode(spec, *, steps=STEPS, preview=None):
    """Run the chosen steps in order and return a combined report.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.
    steps : sequence of str
        Any of ``"audio"``, ``"visual"``, ``"overlays"``, ``"render"``; they
        always run in that order.
    preview : tuple of float, optional
        ``(start, seconds)``; applies to the ``render`` step.

    Returns
    -------
    dict
        ``name``, ``mode``, ``steps`` (the step reports) and ``human_gates``
        (all ``"NOT_RUN"``: full listening, full visual watch, rights,
        private upload).
    """
    if isinstance(steps, str):
        steps = [s for s in steps.split(",") if s]
    steps = list(steps)
    unknown = [s for s in steps if s not in STEPS]
    if unknown or not steps:
        raise MusicEpisodeError(f"steps must be a subset of {list(STEPS)}, got {steps}")
    runners = {
        "audio": prepare_audio,
        "visual": prepare_visual,
        "overlays": prepare_overlays,
        "render": lambda s: render(s, preview=preview),
    }
    results = {}
    for step in STEPS:
        if step in steps:
            results[step] = runners[step](spec)
    return {
        "name": spec.name,
        "mode": spec.mode,
        "steps": results,
        "human_gates": dict(HUMAN_GATES),
    }


_README = """\
森息音界 music episode 專案（模式：{mode}）

製作清單
[ ] 1. 把音樂放入 tracks/（取代 PLACEHOLDER_*.wav），並在 music.json 的 tracks
       填 id、source、chapter_seconds。
[ ] 2. 把 Flow 影片片段放入 visual/（取代 PLACEHOLDER_clip_*.mp4），
       或在 visual.cycle_path 指向已備妥的循環影片。
{study}[ ] 3. 檢查：  python -m moviepy.ae.templates music validate music.json
[ ] 4. 建置素材：python -m moviepy.ae.templates music build music.json --steps audio,visual,overlays
[ ] 5. 預覽 60 秒：python -m moviepy.ae.templates music build music.json --steps render --preview 0 60
[ ] 6. 正式輸出：python -m moviepy.ae.templates music build music.json --steps render
所有輸出皆為獨佔建立（不覆寫）；每一步寫入證據 JSON，中斷後重跑會驗證雜湊並略過已完成者。
若修改設定，須先刪除對應產物與其 .json 證據才能重建。

人工驗收關卡（全部尚未執行，須由人簽核）
- 完整試聽（full listening）：NOT_RUN
- 完整畫面觀看（full visual watch）：NOT_RUN
- 版權與授權（rights）：NOT_RUN
- 私人上傳（private upload，先設為不公開）：NOT_RUN
"""

_STUDY_STEP = "[ ] 2b. 編輯 study.json（階段、三語標籤與提示）；總長須等於音訊總長。\n"


def init_music_episode(directory, mode):
    """Create a music episode project folder for ``mode``.

    ``directory`` must not exist or be empty. It receives ``music.json`` (the
    shipped config with PLACEHOLDER paths), empty ``tracks/`` and ``visual/``
    folders, ``study.json`` (study mode, from ``pomodoro_schedule``) and a
    Traditional Chinese ``README.txt`` with the human QA gates (all NOT_RUN).

    Parameters
    ----------
    directory : str or Path
        New project folder.
    mode : str
        ``"sleep_longform"`` or ``"study_pomodoro"``.

    Returns
    -------
    str
        Path of ``music.json``.

    Examples
    --------
    >>> init_music_episode("x", "tv")
    Traceback (most recent call last):
    ...
    moviepy.ae.templates.music_episode.MusicEpisodeError: mode must be one of ['sleep_longform', 'study_pomodoro'], got 'tv'
    """
    _choice(mode, "mode", MODES)
    target = Path(directory)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise MusicEpisodeError(f"directory exists and is not empty: {target}")
    source = Path(__file__).parent / "configs" / f"music_{mode}.json"
    target.mkdir(parents=True, exist_ok=True)
    (target / "tracks").mkdir()
    (target / "visual").mkdir()
    config = target / "music.json"
    config.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    if mode == "study_pomodoro":
        document = json.dumps(pomodoro_schedule(), ensure_ascii=False, indent=2)
        (target / "study.json").write_text(document + "\n", encoding="utf-8")
    (target / "README.txt").write_text(
        _README.format(
            mode=mode, study=_STUDY_STEP if mode == "study_pomodoro" else ""
        ),
        encoding="utf-8",
    )
    return str(config)


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #


def _parser():
    parser = argparse.ArgumentParser(
        prog="python -m moviepy.ae.templates music",
        description="Music-channel episode tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="create a project folder")
    p_init.add_argument("directory")
    p_init.add_argument("--mode", choices=MODES, required=True)
    p_val = sub.add_parser("validate", help="check the spec, lengths and inputs")
    p_val.add_argument("spec")
    p_val.add_argument("--allow-missing", action="store_true", help="skip file checks")
    p_build = sub.add_parser("build", help="run the resumable build steps")
    p_build.add_argument("spec")
    p_build.add_argument("--steps", default=",".join(STEPS))
    p_build.add_argument("--preview", type=float, nargs=2, metavar=("START", "SECONDS"))
    p_build.add_argument("--workers", type=int, default=None)
    p_build.add_argument("--encoder", choices=ENCODERS, default=None)
    return parser


def main(argv=None):
    """Command line of ``python -m moviepy.ae.templates music``.

    Subcommands are ``init DIR --mode``, ``validate SPEC`` and
    ``build SPEC [--steps ...] [--preview START SECONDS] [--workers N]
    [--encoder E]``.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments after ``music``; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status: 0 success, 1 on any error or failed validation.
    """
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            print(f"created {init_music_episode(args.directory, args.mode)}")
        elif args.command == "validate":
            spec = MusicEpisodeSpec.from_json(Path(args.spec))
            result = validate_music_episode(spec, check_files=not args.allow_missing)
            for path in result["missing"]:
                print(f"MISSING {path}", file=sys.stderr)
            for problem in result["problems"]:
                print(f"PROBLEM {problem}", file=sys.stderr)
            if not result["ok"]:
                return 1
            plan = result["plan"]
            length = (
                "length unchecked"
                if plan is None
                else f"{plan['total_seconds']:g} s, {plan['video_frames']} frames"
            )
            print(
                f"OK {spec.name} ({spec.mode}): {spec.size[0]}x{spec.size[1]} "
                f"@ {spec.fps:g} fps, {len(spec.tracks)} tracks, {length}"
            )
        else:
            spec = MusicEpisodeSpec.from_json(Path(args.spec))
            changes = {}
            if args.workers is not None:
                changes["workers"] = args.workers
            if args.encoder is not None:
                changes["encoder"] = args.encoder
            if changes:
                spec = dataclasses.replace(spec, **changes)
            report = build_music_episode(
                spec,
                steps=args.steps,
                preview=tuple(args.preview) if args.preview else None,
            )
            print(
                json.dumps(report, ensure_ascii=False, indent=2, default=_json_default)
            )
    except (MusicEpisodeError, OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
