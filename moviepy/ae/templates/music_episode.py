r"""Declarative music-channel episodes: spec, resumable steps and command line.

A ``MusicEpisodeSpec`` describes a whole long-form music video (a
5 h 20 min sleep mix or an 87 min study companion) as data. Six resumable
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
``thumbnail``
    ``publish/thumbnail.jpg`` from one cycle frame (``thumbnail.py``).
``qa``
    ``qa/<name>.qa.json`` from ``music_qa.qa_report`` on the rendered MP4.

Optional sections (all absent by default, so older specs build unchanged):
``ambience`` (``particles`` -> ``visual/particles.mov`` looped by FFmpeg;
``light_arc`` -> background ``video_filters``; ``sleep_fade`` -> whole-frame
``final_filters`` plus a music fade), ``cards`` (trilingual track-title cards
as ASS), ``study_ring`` / ``study_timeline`` (study ASS layers),
``thumbnail`` and ``qa``. Times in ``ambience`` may be negative to count back
from the end of the video.

Shipped recommended configs (``configs/music_*.json``) and the
``music_channel`` choices they come from:

``music_sleep_longform.json`` (S06 long-form sleep, ``build-season-06.ps1``)
    Six pieces of 3210 s joined with 12 s chapter crossfades give exactly
    19200 s (5 h 20 min); 10 s loop crossfades inside a chapter; 4 s edge
    fades; a single shared native 1280x720 / 24 fps loop (8 s clips, 2 s
    loop dissolve); loudness -18 LUFS, true peak -1.8 dBTP, LRA 11; no
    spectrum and no chime; chapters from the cue sheet; NVENC when present.
    Enrichment (low stimulation, sleep-friendly, nothing flashes): gentle
    ``fireflies`` (18, 60 % peak, seed 7, 20 s seamless loop; recorded in the
    evidence) over the picture; a ``day_to_night`` light arc across the whole
    5 h 20 min (peak change far below the 0.1/s guard); a ``sleep_fade`` over
    the last 30 minutes to a dim 0.15 glow with the music fading with it;
    trilingual track cards at every chapter (placeholder titles); a
    thumbnail with the auto duration badge; automated QA on the render.
``music_study_pomodoro.json`` (S14 study companion, ``s14-assets.py``, ``s14-render.py``)
    A 5220 s schedule (3 x 25 min focus, 5 min breaks, 2 min closing; chimes
    at 1500, 1800, 3300, 3600 and 5100 s), six pieces of 880 s (6 x 880 - 5 x
    12 = 5220), chimes 10 dB under the local music RMS (never boosted, no
    ducking), spectrum bars at 30 % opacity at (96, 548) in a 1088x140
    layer, AAC 256k. S14 burned ``study-overlay-v2.ass``, the non-compact
    three-panel layout, so ``study_compact`` is ``false`` (the compact
    backplate layout appeared in later seasons). Enrichment: track cards,
    the progress ring and the session timeline (all ASS, so no per-frame
    cost), a thumbnail and automated QA. No particles, light arc or sleep
    fade by default: a study session needs a still, focused picture (set
    ``ambience.particles`` to ``dust`` for a little life), and the sleep fade
    would dim a working screen.

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
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path

from moviepy.ae.templates._audio_io import audio_info
from moviepy.ae.templates.ambience import (
    _MAX_RATE,
    PARTICLE_KINDS,
    LightArc,
    ParticleLayer,
    SleepFade,
    _particle_factory,
    export_overlay_loop,
)
from moviepy.ae.templates.music_audio import (
    DEFAULT_RATE,
    assemble_chapters,
    chime_tone,
    mix_cues,
    normalize_loudness,
    write_chime,
)
from moviepy.ae.templates.music_cards import combine_study_ass, sleep_cards_ass
from moviepy.ae.templates.music_qa import DEFAULT_TARGETS, qa_report
from moviepy.ae.templates.music_render import (
    frame_count,
    preview_window,
    render_music_video,
)
from moviepy.ae.templates.soundx import check_soundx
from moviepy.ae.templates.spectrum import (
    SpectrumLayer,
    analyze_spectrum,
    load_levels,
    save_levels,
)
from moviepy.ae.templates.study import StudySchedule
from moviepy.ae.templates.thumbnail import (
    LAYOUTS,
    ThumbnailSpec,
    format_duration,
    grab_frame,
    render_thumbnail,
)
from moviepy.ae.templates.visual_loop import export_loop, scene_cycle, seamless_loop


__all__ = [
    "MusicEpisodeError",
    "TrackSpec",
    "MusicAudioSpec",
    "VisualSpec",
    "SpectrumSpec",
    "ChimeSpec",
    "ParticlesSpec",
    "LightArcSpec",
    "SleepFadeSpec",
    "AmbienceSpec",
    "CardsSpec",
    "ThumbnailStepSpec",
    "QaSpec",
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
    "make_thumbnail",
    "run_qa",
    "resolve_ambience",
    "spectrum_overlay",
    "build_music_episode",
    "init_music_episode",
    "main",
]

MODES = ("sleep_longform", "study_pomodoro")
STEPS = ("audio", "visual", "overlays", "render", "thumbnail", "qa")
LOUDNESS_BACKENDS = ("soundx", "ffmpeg")
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
_LIGHT_PRESETS = ("day_to_night", "dusk", "dawn")
_LIGHT_PARAMS = ("brightness", "saturation", "warmth", "contrast")
_CARD_POSITIONS = ("lower_left", "lower_right", "upper_left", "upper_right")
_LANGS = ("zh", "en", "ja")
_THUMBNAIL_SIZE = (1280, 720)


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
    loudness_backend : str
        ``"soundx"`` (default; the owner requires soundx for all level and
        loudness work: S14 staging, then LUFS by soundx >= 0.3.0 or, with
        0.2.0, FFmpeg ``loudnorm``) or ``"ffmpeg"`` (pure FFmpeg, explicit).
    """

    target_lufs: float = -18.0
    true_peak: float = -1.8
    lra: float = 11.0
    loop_overlap: float = 10.0
    chapter_crossfade: float = 12.0
    edge_fade: float = 4.0
    loudness_backend: str = "soundx"

    def __post_init__(self):
        _choice(self.loudness_backend, "loudness_backend", LOUDNESS_BACKENDS)
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
class ParticlesSpec:
    """A seeded, seamlessly looping particle layer over the picture.

    Rendered once as ``visual/particles.mov`` (``ambience.export_overlay_loop``)
    and repeated by FFmpeg; see ``ambience.particle_loop``.

    Parameters
    ----------
    kind : str
        ``fireflies``, ``dust``, ``petals``, ``leaves``, ``snow``,
        ``snowflakes`` or ``rain``.
    period : float
        Loop length in seconds; a whole number of frames at ``fps``.
    count : int, optional
        Particle count (per-kind default scaled by region area).
    seed : int
        Random seed; recorded in the evidence so variety is explicit.
    opacity : float, optional
        Peak opacity 0..1 (per-kind default, already gentle).
    speed : float
        Multiplier of fall speed and wander cycles.
    region : sequence of int, optional
        ``(x, y, width, height)`` confining the particles.
    """

    kind: str = "fireflies"
    period: float = 20.0
    count: int = None
    seed: int = 0
    opacity: float = None
    speed: float = 1.0
    region: tuple = None

    def __post_init__(self):
        _choice(self.kind, "kind", PARTICLE_KINDS)
        _set(
            self,
            period=_num(self.period, "period", positive=True),
            seed=_num(self.seed, "seed", low=0, integer=True),
            speed=_num(self.speed, "speed", positive=True),
        )
        if self.count is not None:
            _set(self, count=_num(self.count, "count", low=0, integer=True))
        if self.opacity is not None:
            _set(self, opacity=_num(self.opacity, "opacity", low=0, high=1))
        if self.region is not None:
            if not isinstance(self.region, (list, tuple)) or len(self.region) != 4:
                raise MusicEpisodeError(
                    f"region must be [x, y, width, height], got {self.region!r}"
                )
            region = tuple(
                _num(v, f"region[{i}]", low=0 if i < 2 else 1, integer=True)
                for i, v in enumerate(self.region)
            )
            _set(self, region=region)

    def options(self, size, fps):
        """Return the ``particle_loop`` keywords for a picture ``size``."""
        values = {
            "size": (int(size[0]), int(size[1])),
            "period": self.period,
            "fps": fps,
            "seed": self.seed,
            "speed": self.speed,
        }
        for key in ("count", "opacity", "region"):
            if getattr(self, key) is not None:
                values[key] = getattr(self, key)
        return values


def _keyframes(value):
    """Normalize LightArc keyframes to a tuple of ``(time, {param: v})``."""
    if not isinstance(value, (list, tuple)) or not value:
        raise MusicEpisodeError("keyframes must be a non-empty list")
    rows = []
    for index, item in enumerate(value):
        label = f"keyframes[{index}]"
        if isinstance(item, dict):
            params = dict(item)
            if "time" not in params:
                raise MusicEpisodeError(f"{label}: missing 'time'")
            when = params.pop("time")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            when, params = item[0], item[1]
            if not isinstance(params, dict):
                raise MusicEpisodeError(f"{label}: parameters must be an object")
        else:
            raise MusicEpisodeError(
                f"{label} must be [time, {{parameters}}] or an object with 'time'"
            )
        unknown = sorted(set(params) - set(_LIGHT_PARAMS))
        if unknown:
            raise MusicEpisodeError(f"{label}: unknown parameter(s) {unknown}")
        rows.append(
            (
                _num(when, f"{label} time"),
                {k: _num(v, f"{label} {k}") for k, v in sorted(params.items())},
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class LightArcSpec:
    """A slow light and colour curve over the whole video (FFmpeg ``eq``).

    Give a ``preset`` (spanning the whole video) or ``keyframes``, or neither
    for no light arc. Negative times count back from the end of the video.

    Parameters
    ----------
    preset : str, optional
        ``day_to_night``, ``dusk`` or ``dawn``.
    keyframes : sequence, optional
        ``[time, {"brightness": b, "saturation": s, "warmth": w,
        "contrast": c}]`` rows or ``{"time": t, ...}`` objects.
    max_rate : float, optional
        Largest allowed change per second (flicker guard); the default is
        the ``ambience`` limit of 0.1 per second. Raise it only for short
        test videos.
    """

    preset: str = None
    keyframes: tuple = None
    max_rate: float = None

    def __post_init__(self):
        if self.preset is not None:
            _choice(self.preset, "preset", _LIGHT_PRESETS)
        if self.keyframes is not None:
            _set(self, keyframes=_keyframes(self.keyframes))
        if self.preset is not None and self.keyframes is not None:
            raise MusicEpisodeError("give a preset or keyframes, not both")
        if self.max_rate is not None:
            _set(self, max_rate=_num(self.max_rate, "max_rate", positive=True))

    @property
    def active(self):
        """Whether a light arc is configured."""
        return self.preset is not None or self.keyframes is not None


@dataclass(frozen=True)
class SleepFadeSpec:
    """Fade the whole picture (and the music) to a floor near the end.

    Parameters
    ----------
    start : float
        Seconds where the fade starts; negative counts from the end
        (``-1800`` is the last 30 minutes).
    end : float, optional
        Seconds where ``floor`` is reached; negative counts from the end;
        ``None`` means the end of the video.
    floor : float
        Gain held at the end: 0 black / silent, 0.15 a dim glow.
    audio : bool
        Also fade the music with the same gain.
    max_rate : float, optional
        Flicker-guard ceiling, see ``LightArcSpec``.
    """

    start: float
    end: float = None
    floor: float = 0.0
    audio: bool = True
    max_rate: float = None

    def __post_init__(self):
        _set(self, start=_num(self.start, "start"))
        if self.end is not None:
            _set(self, end=_num(self.end, "end"))
        _set(self, floor=_num(self.floor, "floor", low=0))
        if self.floor >= 1:
            raise MusicEpisodeError("floor must be < 1")
        _bool(self.audio, "audio")
        if self.max_rate is not None:
            _set(self, max_rate=_num(self.max_rate, "max_rate", positive=True))


@dataclass(frozen=True)
class AmbienceSpec:
    """Slow, low-stimulation motion layers (nothing flashes).

    Parameters
    ----------
    particles : ParticlesSpec or dict, optional
        Particle loop (``visual/particles.mov``), or ``None``.
    light_arc : LightArcSpec or dict, optional
        Slow light and colour arc on the background (``video_filters``).
    sleep_fade : SleepFadeSpec or dict, optional
        Whole-frame fade at the end (``final_filters`` plus music fade).
    """

    particles: ParticlesSpec = None
    light_arc: LightArcSpec = None
    sleep_fade: SleepFadeSpec = None

    def __post_init__(self):
        if self.particles is not None:
            _set(self, particles=_make(ParticlesSpec, self.particles, "particles"))
        _set(self, light_arc=_make(LightArcSpec, self.light_arc or {}, "light_arc"))
        if self.sleep_fade is not None:
            _set(self, sleep_fade=_make(SleepFadeSpec, self.sleep_fade, "sleep_fade"))


@dataclass(frozen=True)
class CardsSpec:
    """Trilingual track-title cards at every chapter start (ASS).

    Parameters
    ----------
    enabled : bool
        Burn the cards (default off, so older specs are unchanged).
    titles : dict
        ``{track id: {"zh": ..., "en": ..., "ja": ...}}``, at least one
        language per track that gets a card.
    position : str
        ``lower_left``, ``lower_right``, ``upper_left`` or ``upper_right``.
    hold : float
        Seconds fully visible.
    offset : float
        Delay after the chapter start.
    skip_first : bool
        No card for the first track (the video's opening).
    """

    enabled: bool = False
    titles: dict = field(default_factory=dict)
    position: str = "lower_left"
    hold: float = 8.0
    offset: float = 3.0
    skip_first: bool = False

    def __post_init__(self):
        _bool(self.enabled, "enabled")
        _choice(self.position, "position", _CARD_POSITIONS)
        _set(
            self,
            hold=_num(self.hold, "hold", low=0),
            offset=_num(self.offset, "offset", low=0),
        )
        _bool(self.skip_first, "skip_first")
        if not isinstance(self.titles, dict):
            raise MusicEpisodeError("titles must be an object of track titles")
        titles = {}
        for track_id, names in self.titles.items():
            if not isinstance(names, dict):
                raise MusicEpisodeError(f"titles.{track_id} must be an object")
            unknown = sorted(set(names) - set(_LANGS))
            if unknown:
                raise MusicEpisodeError(
                    f"titles.{track_id}: unknown language {unknown}"
                )
            clean = {
                lang: _str(names[lang], f"titles.{track_id}.{lang}")
                for lang in _LANGS
                if lang in names
            }
            if not clean:
                raise MusicEpisodeError(f"titles.{track_id} needs zh, en or ja text")
            titles[str(track_id)] = clean
        _set(self, titles=titles)


@dataclass(frozen=True)
class ThumbnailStepSpec:
    """The ``publish/thumbnail.jpg`` step (see ``thumbnail.render_thumbnail``).

    Parameters
    ----------
    enabled : bool
        Run the step by default (default off).
    titles : dict
        ``{"zh": ..., "en": ..., "ja": ...}``; ``zh`` is required when enabled.
    subtitle : str, optional
        Accent line under the titles.
    duration_badge : str or float, optional
        Badge text, seconds, ``"auto"`` (the formatted total duration) or
        ``None`` for no badge.
    layout : str
        ``left_panel``, ``bottom_band`` or ``center``.
    time : float, optional
        Seconds into the background cycle for the frame (default: a third of
        the cycle).
    output : str
        File name inside ``publish/`` (``.jpg``, ``.jpeg`` or ``.png``).
    require_contrast : bool
        Refuse a main title below 4.5:1 contrast.
    """

    enabled: bool = False
    titles: dict = field(default_factory=dict)
    subtitle: str = None
    duration_badge: object = "auto"
    layout: str = "left_panel"
    time: float = None
    output: str = "thumbnail.jpg"
    require_contrast: bool = True

    def __post_init__(self):
        _bool(self.enabled, "enabled")
        if not isinstance(self.titles, dict):
            raise MusicEpisodeError("titles must be an object")
        _set(self, titles=dict(self.titles))
        if self.subtitle is not None:
            _str(self.subtitle, "subtitle")
        badge = self.duration_badge
        if badge is not None and badge != "auto":
            if isinstance(badge, str):
                _str(badge, "duration_badge")
            else:
                _num(badge, "duration_badge", low=0)
        _choice(self.layout, "layout", LAYOUTS)
        if self.time is not None:
            _set(self, time=_num(self.time, "time", low=0))
        if not _SAFE_NAME.match(_str(self.output, "output")) or Path(
            self.output
        ).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise MusicEpisodeError(
                "output must be a simple .jpg/.jpeg/.png file name, "
                f"got {self.output!r}"
            )
        _bool(self.require_contrast, "require_contrast")
        if self.enabled or self.titles:
            try:
                self.render_spec(0)
            except (ValueError, TypeError) as error:
                raise MusicEpisodeError(f"titles/subtitle/layout: {error}") from None

    def render_spec(self, total_seconds):
        """Return the ``ThumbnailSpec`` (``auto`` badge from ``total_seconds``)."""
        badge = self.duration_badge
        if badge == "auto":
            badge = format_duration(total_seconds)
        return ThumbnailSpec(
            titles=dict(self.titles),
            subtitle=self.subtitle,
            duration_badge=badge,
            layout=self.layout,
        )


@dataclass(frozen=True)
class QaSpec:
    """The automated QA step on the rendered MP4 (``music_qa.qa_report``).

    Parameters
    ----------
    enabled : bool
        Run the step by default (default off).
    targets : dict
        Overrides of ``music_qa.DEFAULT_TARGETS`` (``lufs``,
        ``lufs_tolerance``, ``true_peak_max``).
    flash_seconds : float, optional
        Limit the flash analysis to this many seconds (default: all).
    """

    enabled: bool = False
    targets: dict = field(default_factory=dict)
    flash_seconds: float = None

    def __post_init__(self):
        _bool(self.enabled, "enabled")
        if not isinstance(self.targets, dict):
            raise MusicEpisodeError("targets must be an object")
        unknown = sorted(set(self.targets) - set(DEFAULT_TARGETS))
        if unknown:
            raise MusicEpisodeError(f"targets: unknown key(s) {unknown}")
        _set(
            self,
            targets={k: _num(v, f"targets.{k}") for k, v in self.targets.items()},
        )
        if self.flash_seconds is not None:
            _set(
                self,
                flash_seconds=_num(self.flash_seconds, "flash_seconds", positive=True),
            )


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
        Folder for ``audio/``, ``visual/``, ``overlays/``, ``render/``,
        ``publish/`` and ``qa/``.
    ambience : AmbienceSpec or dict, optional
        Particles, light arc and sleep fade; absent means none.
    cards : CardsSpec or dict, optional
        Track title cards (off unless ``enabled``).
    study_ring, study_timeline : bool
        Study mode: add the progress ring / session timeline to the ASS.
    thumbnail : ThumbnailStepSpec or dict, optional
        The ``thumbnail`` step; off unless ``enabled``.
    qa : QaSpec or dict, optional
        The ``qa`` step; off unless ``enabled``.
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
    ambience: AmbienceSpec = None
    cards: CardsSpec = None
    study_ring: bool = False
    study_timeline: bool = False
    thumbnail: ThumbnailStepSpec = None
    qa: QaSpec = None

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
        _bool(self.study_ring, "study_ring")
        _bool(self.study_timeline, "study_timeline")
        if (self.study_ring or self.study_timeline) and not study_mode:
            raise MusicEpisodeError(
                "study_ring and study_timeline need study_pomodoro mode"
            )
        ambience = _make(AmbienceSpec, self.ambience or {}, "ambience")
        cards = _make(CardsSpec, self.cards or {}, "cards")
        thumbnail = _make(ThumbnailStepSpec, self.thumbnail or {}, "thumbnail")
        qa = _make(QaSpec, self.qa or {}, "qa")
        if ambience.particles is not None:
            try:
                ParticleLayer(
                    ambience.particles.kind, **ambience.particles.options(size, fps)
                )
            except (ValueError, TypeError) as error:
                raise MusicEpisodeError(f"ambience: particles: {error}") from None
        unknown = sorted(set(cards.titles) - {t.id for t in tracks})
        if unknown:
            raise MusicEpisodeError(f"cards: titles for unknown track id(s) {unknown}")
        if cards.enabled:
            wanted = [t.id for t in tracks][1 if cards.skip_first else 0 :]
            absent = [i for i in wanted if i not in cards.titles]
            if absent:
                raise MusicEpisodeError(
                    f"cards: titles missing for track id(s) {absent}"
                )
        if thumbnail.enabled and "zh" not in thumbnail.titles:
            raise MusicEpisodeError("thumbnail: titles.zh is required when enabled")
        _set(
            self,
            size=size,
            fps=fps,
            tracks=tracks,
            audio=audio,
            visual=visual,
            spectrum=spectrum,
            chime=chime,
            ambience=ambience,
            cards=cards,
            thumbnail=thumbnail,
            qa=qa,
        )
        total = self._known_total()
        if total is not None:
            try:
                resolve_ambience(self, total)
            except MusicEpisodeError as error:
                raise MusicEpisodeError(f"ambience: {error}") from None

    def _known_total(self):
        """Return the total seconds when the spec alone determines it."""
        try:
            if self.mode == "study_pomodoro":
                if not isinstance(self.study, dict):
                    return None
                return float(StudySchedule.from_dict(self.study).duration)
            cross = self.audio.chapter_crossfade
            return (
                sum(t.chapter_seconds for t in self.tracks)
                - (len(self.tracks) - 1) * cross
            )
        except (ValueError, KeyError, TypeError):
            return None

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
        audio), ``loops`` (list), ``cycle`` (the background), ``particles``,
        ``levels``, ``ass`` (the study ASS, or the sleep track cards),
        ``chapters``, ``video``, ``thumbnail`` and ``qa`` (``None`` when the
        part is not configured).
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
        "particles": (
            str(root / "visual" / "particles.mov") if spec.ambience.particles else None
        ),
        "ass": (
            str(root / "overlays" / "study.ass")
            if study
            else str(root / "overlays" / "cards.ass") if spec.cards.enabled else None
        ),
        "chapters": (
            str(root / "overlays" / "chapters.ffmeta") if spec.chapters else None
        ),
        "video": str(root / "render" / f"{spec.name}.mp4"),
        "thumbnail": (
            str(root / "publish" / spec.thumbnail.output)
            if spec.thumbnail.enabled
            else None
        ),
        "qa": str(root / "qa" / f"{spec.name}.qa.json") if spec.qa.enabled else None,
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


def _artifact_sha(path):
    """Return the recorded SHA-256 of an artifact (or hash the file)."""
    evidence = _evidence_path(path)
    if evidence.is_file():
        recorded = json.loads(evidence.read_text(encoding="utf-8")).get("sha256")
        if recorded:
            return recorded
    return _sha256(path)


def _at(value, total, name):
    """Resolve seconds (negative: from the end) inside ``[0, total]``."""
    seconds = value + total if value < 0 else value
    if not -1e-9 <= seconds <= total + 1e-9:
        raise MusicEpisodeError(
            f"{name} = {value:g} s is outside the {total:g} s video"
        )
    return max(0.0, seconds)


def resolve_ambience(spec, total):
    """Resolve the ambience times against the video length and build filters.

    Negative times count back from the end. A fade or arc that is too fast
    (a flicker risk) or lies outside the video raises.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.
    total : float
        Total video duration in seconds.

    Returns
    -------
    dict
        ``video_filters`` (light arc), ``final_filters`` (sleep fade video),
        ``audio_filters`` (sleep fade music), ``description`` (JSON-friendly,
        with the resolved times) and the objects ``light_arc`` / ``sleep_fade``
        (``None`` when absent).

    Raises
    ------
    MusicEpisodeError
        For times outside ``[0, total]`` or a curve faster than the limit.
    """
    total = float(total)
    ambience = spec.ambience
    result = {
        "light_arc": None,
        "sleep_fade": None,
        "video_filters": [],
        "final_filters": [],
        "audio_filters": [],
        "description": {"total_seconds": total},
    }
    arc_spec = ambience.light_arc
    if arc_spec.active:
        limit = _MAX_RATE if arc_spec.max_rate is None else arc_spec.max_rate
        try:
            if arc_spec.preset is not None:
                arc = getattr(LightArc, arc_spec.preset)(total, max_rate=limit)
            else:
                rows = [
                    (_at(when, total, f"light_arc: keyframes[{i}] time"), dict(values))
                    for i, (when, values) in enumerate(arc_spec.keyframes)
                ]
                arc = LightArc(rows, max_rate=limit)
        except ValueError as error:
            raise MusicEpisodeError(f"light_arc: {error}") from None
        result["light_arc"] = arc
        result["video_filters"].append(arc.to_ffmpeg_filter())
        result["description"]["light_arc"] = {
            "preset": arc_spec.preset,
            "keyframes": [[t, v] for t, v in arc.keyframes],
        }
    fade_spec = ambience.sleep_fade
    if fade_spec is not None:
        start = _at(fade_spec.start, total, "sleep_fade: start")
        end = (
            total
            if fade_spec.end is None
            else _at(fade_spec.end, total, "sleep_fade: end")
        )
        limit = _MAX_RATE if fade_spec.max_rate is None else fade_spec.max_rate
        try:
            fade = SleepFade(
                start, end, fade_spec.floor, fade_spec.audio, max_rate=limit
            )
            fade.validate(total)
        except ValueError as error:
            raise MusicEpisodeError(f"sleep_fade: {error}") from None
        result["sleep_fade"] = fade
        result["final_filters"].append(fade.video_filter())
        if fade.audio_filter() is not None:
            result["audio_filters"].append(fade.audio_filter())
        result["description"]["sleep_fade"] = fade.describe()
    if ambience.particles is not None:
        result["description"]["particles"] = _plain(
            ambience.particles.options(spec.size, spec.fps)
        )
        result["description"]["particles"]["kind"] = ambience.particles.kind
    return result


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

    Writes ``audio/masters/<id>.flac`` (S14 soundx staging and loudness to
    the target with an exact-frame contract; ``audio.loudness_backend``), ``audio/music.flac`` (looped
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
            "loudness_backend": audio.loudness_backend,
        }

        def normalize(partial, source=source):
            return normalize_loudness(
                source,
                partial,
                target_lufs=audio.target_lufs,
                true_peak=audio.true_peak,
                lra=audio.lra,
                backend=audio.loudness_backend,
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


def _prepare_cycle(spec):
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


def prepare_particles(spec):
    """Export the transparent particle loop ``visual/particles.mov``; resumable.

    Uses ``ambience.export_overlay_loop`` (lossless ``qtrle`` with alpha, a
    seamless seeded loop of ``ambience.particles.period`` seconds) and the
    usual exclusive-create evidence; the seed and every option are part of the
    settings hash, so a changed particle setting is detected as stale.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.

    Returns
    -------
    dict or None
        The artifact evidence (with ``skipped``), or ``None`` without
        ``ambience.particles``.
    """
    particles = spec.ambience.particles
    if particles is None:
        return None
    path = music_paths(spec)["particles"]
    options = particles.options(spec.size, spec.fps)
    config = {"kind": particles.kind, "options": _plain(options)}

    def export(partial):
        stale = Path(str(partial) + ".json")
        if stale.exists():
            stale.unlink()
        try:
            facts = export_overlay_loop(
                (_particle_factory, (particles.kind, options)),
                partial,
                workers=spec.workers,
            )
        finally:
            if stale.exists():
                stale.unlink()
        facts = dict(facts)
        facts["particles"] = config
        return facts

    return _resumable(path, config, export, kind="visual")


def prepare_visual(spec):
    """Export the loops, the scene cycle and the particle loop; resumable.

    See the cycle export below for the loops and the cycle; with
    ``ambience.particles`` the transparent ``visual/particles.mov`` is added
    (``prepare_particles``).

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.

    Returns
    -------
    dict
        ``loops`` (list), ``cycle``, ``background`` (the cycle path) and
        ``particles`` (evidence or ``None``).
    """
    report = _prepare_cycle(spec)
    report["particles"] = prepare_particles(spec)
    return report


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


def _ass_text(spec, sheet):
    """Return the ASS text of the episode (study layers and/or track cards)."""
    cards = spec.cards
    options = None
    if cards.enabled:
        options = {
            "titles": cards.titles,
            "position": cards.position,
            "hold": cards.hold,
            "offset": cards.offset,
            "skip_first": cards.skip_first,
        }
    try:
        if spec.mode == "study_pomodoro":
            schedule = spec.schedule()
            if not (spec.study_ring or spec.study_timeline or cards.enabled):
                return schedule.to_ass(compact=spec.study_compact)
            return combine_study_ass(
                schedule,
                ring=spec.study_ring,
                timeline=spec.study_timeline,
                cards=None if options is None else dict(options, cues=sheet),
                compact=spec.study_compact,
            )
        return sleep_cards_ass(sheet, **options)
    except ValueError as error:
        raise MusicEpisodeError(f"cards/ring/timeline: {error}") from None


def prepare_overlays(spec):
    """Write the spectrum levels, the ASS and the chapters; resumable.

    ``overlays/levels.npy`` is ``analyze_spectrum`` of the final audio (one
    row per video frame; only when ``spectrum.enabled``), ``overlays/study.ass``
    is ``StudySchedule.to_ass()`` merged with the progress ring, session
    timeline and track cards when configured (study mode), or
    ``overlays/cards.ass`` with the track cards (sleep mode, ``cards.enabled``);
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
    sheet = json.loads(music_evidence.read_text(encoding="utf-8"))
    if paths["ass"] is not None:
        text = _ass_text(spec, sheet)
        report["ass"] = _resumable(
            paths["ass"],
            {"ass": hashlib.sha256(text.encode("utf-8")).hexdigest()},
            _text_writer(text),
        )
    if spec.mode == "study_pomodoro":
        if spec.chapters:
            ffmeta = spec.schedule().chapters_ffmetadata()
    elif spec.chapters:
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
    for key in ("particles", "levels", "ass", "chapters"):
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
    ambience = resolve_ambience(spec, info["seconds"])
    config_hash = _digest(_render_config(spec, paths, window, ambience))
    if output.exists():
        if not evidence.is_file():
            raise MusicEpisodeError(f"{output} exists without evidence; remove it")
        recorded = json.loads(evidence.read_text(encoding="utf-8"))
        if recorded.get("output_sha256") != _sha256(output):
            raise MusicEpisodeError(f"{output} no longer matches its recorded hash")
        if recorded.get("config_sha256", config_hash) != config_hash:
            raise MusicEpisodeError(
                f"{output} was rendered from different settings or inputs; "
                "remove it (and its evidence) to render again"
            )
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
        overlay_loops=([(paths["particles"], 0, 0, 1.0)] if paths["particles"] else []),
        video_filters=ambience["video_filters"],
        audio_filters=ambience["audio_filters"],
        final_filters=ambience["final_filters"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if window is None:
        report = render_music_video(output, **options)
    else:
        if window[0]:
            options["chapters"] = None
        report = preview_window(output, start=window[0], seconds=window[1], **options)
    report = dict(report)
    report["config_sha256"] = config_hash
    report["ambience"] = ambience["description"]
    with open(evidence, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, default=_json_default)
    report["skipped"] = False
    return report


def _render_config(spec, paths, window, ambience):
    """Return the settings and input hashes that determine a render."""
    artifacts = {
        key: _artifact_sha(paths[key])
        for key in ("audio", "cycle", "particles", "levels", "ass", "chapters")
        if paths[key] is not None
    }
    return {
        "artifacts": artifacts,
        "fps": spec.fps,
        "size": list(spec.size),
        "encoder": spec.encoder,
        "quality": spec.quality,
        "audio_bitrate": spec.audio_bitrate,
        "spectrum": _asdict(spec.spectrum) if spec.spectrum.enabled else None,
        "ambience": ambience["description"],
        "window": None if window is None else list(window),
    }


# --------------------------------------------------------------------------- #
# step 5: thumbnail
# --------------------------------------------------------------------------- #


def make_thumbnail(spec):
    """Render ``publish/<thumbnail.output>`` from the cycle; resumable.

    One frame of the background cycle (``thumbnail.time`` seconds, default a
    third of the cycle) gets the trilingual titles, optional subtitle and the
    duration badge (``"auto"`` is the formatted total duration) through
    ``thumbnail.render_thumbnail`` at 1280x720; the evidence (text boxes,
    contrast, fonts) is written next to it. The text needs the Windows fonts
    of ``thumbnail.DEFAULT_FONTS``.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode (``thumbnail.enabled`` must be true).

    Returns
    -------
    dict
        The artifact evidence plus ``skipped``.

    Raises
    ------
    MusicEpisodeError
        If the cycle is missing, a font or glyph is missing, the text does
        not fit or the contrast is too low.
    """
    if not spec.thumbnail.enabled:
        raise MusicEpisodeError("thumbnail.enabled is false")
    paths = music_paths(spec)
    cycle = _require_file(paths["cycle"], "thumbnail needs the cycle")
    total = _audio_plan(spec)["total_seconds"]
    try:
        thumbnail = spec.thumbnail.render_spec(total)
    except (ValueError, TypeError) as error:
        raise MusicEpisodeError(f"thumbnail: {error}") from None
    config = {
        "cycle": _artifact_sha(cycle),
        "titles": thumbnail.titles,
        "subtitle": thumbnail.subtitle,
        "badge": thumbnail.duration_badge,
        "layout": thumbnail.layout,
        "time": spec.thumbnail.time,
        "size": list(_THUMBNAIL_SIZE),
        "require_contrast": spec.thumbnail.require_contrast,
    }

    def make(partial):
        frame = grab_frame(cycle, spec.thumbnail.time)
        _, facts = render_thumbnail(
            thumbnail,
            frame,
            partial,
            size=_THUMBNAIL_SIZE,
            require_contrast=spec.thumbnail.require_contrast,
        )
        facts["frame_time"] = spec.thumbnail.time
        facts["background_source"] = f"{cycle} @ {spec.thumbnail.time}"
        return facts

    try:
        return _resumable(paths["thumbnail"], config, make, kind="visual")
    except (ValueError, OSError) as error:
        if isinstance(error, MusicEpisodeError):
            raise
        raise MusicEpisodeError(f"thumbnail: {error}") from None


# --------------------------------------------------------------------------- #
# step 6: qa
# --------------------------------------------------------------------------- #


def run_qa(spec):
    """Run the automated QA on the rendered MP4 into ``qa/<name>.qa.json``.

    ``music_qa.qa_report`` checks flashing (WCAG 2.3.1 approximation),
    loudness, silence and clipping of the real encoded file. The JSON is an
    exclusive create that also records the render's SHA-256 and the QA
    settings; an existing report for the same render and settings is skipped,
    one for a different render raises. Human listening and visual checks stay
    ``NOT_RUN``.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode (``qa.enabled`` must be true).

    Returns
    -------
    dict
        ``path``, ``overall`` and ``verdicts`` of the report, ``render_sha256``
        and ``skipped``.

    Raises
    ------
    MusicEpisodeError
        If the render is missing, or the report belongs to another render.
    """
    if not spec.qa.enabled:
        raise MusicEpisodeError("qa.enabled is false")
    paths = music_paths(spec)
    video = Path(paths["video"])
    if not video.is_file() or not _evidence_path(video).is_file():
        raise MusicEpisodeError(f"qa needs the rendered video and evidence: {video}")
    render_hash = _sha256(video)
    recorded_hash = json.loads(_evidence_path(video).read_text(encoding="utf-8"))
    if recorded_hash.get("output_sha256") != render_hash:
        raise MusicEpisodeError(f"{video} no longer matches its recorded hash")
    settings = {"targets": spec.qa.targets, "flash_seconds": spec.qa.flash_seconds}
    settings_hash = _digest(settings)
    out = Path(paths["qa"])
    if out.exists():
        report = json.loads(out.read_text(encoding="utf-8"))
        if (
            report.get("render_sha256") != render_hash
            or report.get("config_sha256") != settings_hash
        ):
            raise MusicEpisodeError(
                f"{out} belongs to a different render or QA settings; remove it "
                "to run the QA again"
            )
        skipped = True
    else:
        try:
            report = qa_report(
                video,
                targets=spec.qa.targets or None,
                flash_seconds=spec.qa.flash_seconds,
            )
        except (ValueError, OSError) as error:
            raise MusicEpisodeError(f"qa: {error}") from None
        report.update(
            render_sha256=render_hash, config_sha256=settings_hash, settings=settings
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "x", encoding="utf-8") as stream:
            json.dump(
                report, stream, ensure_ascii=False, indent=2, default=_json_default
            )
        skipped = False
    return {
        "path": str(out),
        "overall": report["overall"],
        "verdicts": report["verdicts"],
        "render_sha256": render_hash,
        "human_listening": report["human_listening"],
        "human_visual": report["human_visual"],
        "skipped": skipped,
    }


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
        lengths or ``None`` when the plan could not be made), ``paths`` and
        ``soundx`` (the ``check_soundx`` result as a dict, or ``None`` when
        ``audio.loudness_backend`` is ``"ffmpeg"``). A missing or too old
        soundx is a problem when the backend is ``"soundx"``.
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
    if plan is not None:
        try:
            resolve_ambience(spec, plan["total_seconds"])
        except MusicEpisodeError as error:
            problems.append(f"ambience: {error}")
    soundx = None
    if spec.audio.loudness_backend == "soundx":
        check = check_soundx()
        soundx = asdict(check)
        problems.extend(f"soundx: {problem}" for problem in check.problems)
    return {
        "ok": not missing and not problems,
        "missing": missing,
        "problems": problems,
        "plan": plan,
        "paths": music_paths(spec),
        "soundx": soundx,
    }


def _step_enabled(spec, step):
    if step == "thumbnail":
        return spec.thumbnail.enabled
    if step == "qa":
        return spec.qa.enabled
    return True


def build_music_episode(spec, *, steps=None, preview=None):
    """Run the chosen steps in order and return a combined report.

    Parameters
    ----------
    spec : MusicEpisodeSpec
        The episode.
    steps : sequence of str or str, optional
        Any of ``"audio"``, ``"visual"``, ``"overlays"``, ``"render"``,
        ``"thumbnail"``, ``"qa"`` (a comma-separated string works too); they
        always run in that order. The default runs the four core steps and
        the ``thumbnail`` / ``qa`` steps when enabled in the spec; naming a
        disabled step explicitly reports it as skipped.
    preview : tuple of float, optional
        ``(start, seconds)``; applies to the ``render`` step.

    Returns
    -------
    dict
        ``name``, ``mode``, ``steps`` (the step reports) and ``human_gates``
        (all ``"NOT_RUN"``: full listening, full visual watch, rights,
        private upload).
    """
    if steps is None:
        steps = [s for s in STEPS if _step_enabled(spec, s)]
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
        "thumbnail": make_thumbnail,
        "qa": run_qa,
    }
    results = {}
    for step in STEPS:
        if step in steps:
            if _step_enabled(spec, step):
                results[step] = runners[step](spec)
            else:
                results[step] = {"enabled": False, "skipped": True}
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
[ ] 7. 縮圖與自動檢查：python -m moviepy.ae.templates music build music.json --steps thumbnail,qa
       （在 music.json 填好 thumbnail.titles 與 cards.titles；氛圍層在 ambience）
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
    p_build.add_argument(
        "--steps",
        default=None,
        help=f"comma-separated subset of {','.join(STEPS)} (default: all enabled)",
    )
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
