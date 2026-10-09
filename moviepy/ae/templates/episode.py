"""Declarative episode build: one JSON document -> one AE ``Composition``.

``EpisodeSpec`` is the "optimal configuration" layer that ties the individual
templates (``presets``, ``background``, ``title_card``, ``ken_burns``,
``chapter_tag``, ``quote``, ``subtitles``) together. ``build_episode`` turns a
spec into a single ``Composition`` and ``episode_report`` summarises it for QA.

Layer order (bottom to top)
---------------------------
``shots`` (picture, in time order) -> ``quotes`` -> ``chapters`` -> intro and
outro title cards -> ``Subtitles``. This is the order of the R23 and NLH video
workflows: picture first, alpha overlays above it, cards above those, and the
burned-in subtitles last so no overlay can ever hide a cue. Chapters and
quotes may not overlap the intro/outro (a card is opaque), and chapters may not
overlap each other; both raise ``EpisodeError``.

Timeline rules
--------------
* Segments run intro, shots, outro. ``cut`` abuts, ``crossfade`` overlaps the
  incoming segment onto the previous one by ``transition_duration`` (the
  incoming layer fades in with AE opacity keyframes), ``dip`` fades the
  outgoing layer to ``dip_color`` over half the duration and the incoming one
  up from it over the other half, with no overlap.
* A transition belongs to the segment it leads *into* and may only run inside
  the previous Ken Burns shot's focus hold, so the picture has settled before
  it dissolves (the NLH rule). Other previous segments are not constrained.
* Title cards may extend themselves for reading time (``auto_extend``), so the
  timeline is fixed when the cards are built; chapter, quote and cue times are
  validated against that built timeline.

Why the two shipped configs look the way they do
------------------------------------------------
``configs/nightlamp_story.json`` (R23 night-lamp story)
    * ``preset: nightlamp_story``: 24 fps previews, warm-white type, and
      the lamp accent; 24 fps keeps preview renders cheap while matching
      cinema cadence, and the preset renders 1920x1080.
    * Bookends use ``background`` media with ``overlay_opacity`` 0.5. R23 v5
      holds a single frozen frame of real episode media under a 50 % black
      overlay: text stays readable (see ``background.contrast_report``) while
      the picture still identifies the episode. There is no colour fallback.
    * Shots use Ken Burns ``auto`` (push, pan right, push, pan left, pull),
      so consecutive stills never repeat a move, with the preset's 1.5 s
      hold and a 0.6 s (``preset.fade``) crossfade inside that hold.
    * No chapters or quotes: the story channel carries only subtitles. Burned
      subtitles are zh-TW 72 px over English 42 px (1080p reference).
``configs/nightlamp_history.json`` (nightlamp-history / NLH)
    * ``preset: nightlamp_history``: rice-paper theme with ink and vermilion.
    * ``chapter_period`` 5.5 s: NLH's channel value, long enough to read a
      line of classical Chinese before the vertical carousel turns.
    * Ken Burns ``push`` 1.0 -> 1.08 with a 2 s ``hold``: 8 % reads as gentle
      emphasis (NLH's 12 % is stronger than stills need); the move is eased
      and always finishes before the hold, so the focus rests on the subject.
    * ``crossfade`` 0.5 s, only after that hold: it lies entirely within the
      2 s focus hold, so a dissolve never starts while the camera moves.
    * Quotes use ``dynasty`` so the bamboo/paper medium is chosen by era.
    * Burned subtitles are zh-TW 72 px plus English 42 px; chapter tags
      replace the old ASS "diamond" text.

Examples
--------
>>> from moviepy.ae.templates.episode import EpisodeSpec
>>> spec = EpisodeSpec(
...     shots=[{"image": "a.jpg", "duration": 4}, {"image": "b.jpg", "duration": 4,
...             "transition": "crossfade", "transition_duration": 0.5}])
>>> EpisodeSpec.from_json(spec.to_json()) == spec
True
>>> spec.shots[1].transition
'crossfade'
"""

import hashlib
import json
import math
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

from moviepy.ae.templates.name_tag import NameTag


__all__ = [
    "EpisodeError",
    "BackgroundSpec",
    "BookendSpec",
    "ShotSpec",
    "ChapterSpec",
    "QuoteSpec",
    "SubtitleSpec",
    "SceneOverlaySpec",
    "AudioSpec",
    "EpisodeSpec",
    "build_episode",
    "episode_report",
    "render_episode",
    "init_episode",
    "main",
]

TRANSITIONS = ("cut", "crossfade", "dip")
_EPS = 1e-6


class EpisodeError(ValueError):
    """An episode specification or timeline is invalid."""


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #


def _num(value, name, *, low=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EpisodeError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise EpisodeError(f"{name} must be finite")
    if positive and value <= 0:
        raise EpisodeError(f"{name} must be positive")
    if low is not None and value < low:
        raise EpisodeError(f"{name} must be >= {low}")
    return value


def _opt_num(value, name, **kw):
    return None if value is None else _num(value, name, **kw)


def _str(value, name, *, empty=False):
    if not isinstance(value, str):
        raise EpisodeError(f"{name} must be a string, got {value!r}")
    if not empty and not value.strip():
        raise EpisodeError(f"{name} must not be empty")
    return value


def _opt_str(value, name):
    return None if value is None else _str(value, name)


def _choice(value, name, options):
    if value not in options:
        raise EpisodeError(f"{name} must be one of {list(options)}, got {value!r}")
    return value


def _pair(value, name, low=None, high=None):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise EpisodeError(f"{name} must be a pair of numbers")
    pair = tuple(_num(v, f"{name}[{k}]") for k, v in enumerate(value))
    for v in pair:
        if (low is not None and v < low) or (high is not None and v > high):
            raise EpisodeError(f"{name} values must lie in [{low}, {high}]")
    return pair


def _jsonable(value, name):
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as error:
        raise EpisodeError(f"{name} must be JSON compatible: {error}") from None


def _from_dict(cls, data, label, renames=None):
    """Build dataclass ``cls`` from ``data``, rejecting unknown/missing keys."""
    if not isinstance(data, dict):
        raise EpisodeError(f"{label} must be an object")
    renames = renames or {}
    known = {f.name for f in fields(cls)}
    values = {}
    for key, value in data.items():
        name = renames.get(key, key)
        if name not in known:
            raise EpisodeError(f"{label}: unknown key {key!r}")
        values[name] = value
    missing = [
        f.name
        for f in fields(cls)
        if f.default is MISSING
        and f.default_factory is MISSING
        and f.name not in values
    ]
    if missing:
        raise EpisodeError(f"{label}: missing required key(s) {missing}")
    try:
        return cls(**values)
    except EpisodeError as error:
        raise EpisodeError(f"{label}: {error}") from None


def _to_plain(value):
    if isinstance(value, tuple):
        return [_to_plain(v) for v in value]
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    return value


def _spec_dict(obj, renames=None):
    inverse = {v: k for k, v in (renames or {}).items()}
    return {inverse.get(f.name, f.name): _to_plain(getattr(obj, f.name))
            for f in fields(obj)}  # fmt: skip


def _transition_fields(obj, name):
    _choice(obj.transition, f"{name}transition", TRANSITIONS)
    object.__setattr__(
        obj,
        "transition_duration",
        _opt_num(obj.transition_duration, f"{name}transition_duration", positive=True),
    )
    if obj.transition == "cut" and obj.transition_duration is not None:
        raise EpisodeError("a cut takes no transition_duration")


# --------------------------------------------------------------------------- #
# specification pieces
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BackgroundSpec:
    """Media behind the intro/outro cards (see ``background.media_background``).

    ``source`` is an image or video path; ``frame_time`` selects the frozen
    video frame; ``overlay_opacity`` (fraction, ``None`` = preset value) is the
    darkening overlay.
    """

    source: str
    frame_time: float = None
    focal_point: tuple = (0.5, 0.5)
    overlay_opacity: float = None

    def __post_init__(self):
        _str(self.source, "source")
        object.__setattr__(
            self, "frame_time", _opt_num(self.frame_time, "frame_time", low=0)
        )
        object.__setattr__(
            self, "focal_point", _pair(self.focal_point, "focal_point", 0, 1)
        )
        opacity = _opt_num(self.overlay_opacity, "overlay_opacity", low=0)
        if opacity is not None and opacity > 1:
            raise EpisodeError("overlay_opacity must lie in [0, 1]")
        object.__setattr__(self, "overlay_opacity", opacity)


_CARD_TIMING = (
    "brand_start",
    "brand_duration",
    "title_start",
    "title_duration",
    "subtitle_start",
    "subtitle_duration",
)


@dataclass(frozen=True)
class BookendSpec:
    """Intro or outro card: ``TitleCardSpec`` copy fields plus its transition.

    ``transition`` joins this card to the previous segment (default ``cut``);
    the card's own timing fields (``brand_*``, ``title_*``, ``subtitle_*``
    start/duration, ``fade_out``, ``auto_extend``) are those of
    ``TitleCardSpec`` and share its defaults (R23's entrance), so a channel can
    retime its cards from JSON without touching code.
    """

    title: str
    brand: str = ""
    subtitle: str = ""
    layout: str = "right_column"
    duration: float = 5.0
    fade_out: float = 0.35
    rule: bool = True
    auto_extend: bool = True
    title_keep_together: tuple = ()
    subtitle_keep_together: tuple = ()
    transition: str = "cut"
    transition_duration: float = None
    brand_start: float = 0.2
    brand_duration: float = 0.55
    title_start: float = 0.55
    title_duration: float = 0.85
    subtitle_start: float = 1.05
    subtitle_duration: float = 0.65

    def __post_init__(self):
        _str(self.title, "title")
        _str(self.brand, "brand", empty=True)
        _str(self.subtitle, "subtitle", empty=True)
        _choice(self.layout, "layout", ("right_column", "center"))
        object.__setattr__(
            self, "duration", _num(self.duration, "duration", positive=True)
        )
        object.__setattr__(self, "fade_out", _num(self.fade_out, "fade_out", low=0))
        for name in _CARD_TIMING:
            object.__setattr__(self, name, _num(getattr(self, name), name, low=0))
        for name in ("rule", "auto_extend"):
            if not isinstance(getattr(self, name), bool):
                raise EpisodeError(f"{name} must be true or false")
        for name in ("title_keep_together", "subtitle_keep_together"):
            terms = getattr(self, name)
            if not isinstance(terms, (list, tuple)):
                raise EpisodeError(f"{name} must be a list of strings")
            object.__setattr__(
                self, name, tuple(_str(t, f"{name} item") for t in terms)
            )
        _transition_fields(self, "")

    def card_spec(self, role):
        """Return the matching ``TitleCardSpec`` for ``role``."""
        from moviepy.ae.templates.title_card import TitleCardSpec

        kwargs = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("transition", "transition_duration")
        }
        try:
            return TitleCardSpec(role=role, **kwargs)
        except ValueError as error:
            raise EpisodeError(str(error)) from None


@dataclass(frozen=True)
class AudioSpec:
    """Narration and optional music mixed into the episode soundtrack.

    ``narration`` is required whenever the section exists: an episode with
    music but no voice is a different product and is left to ``Composition``
    layer audio. It starts at ``narration_offset`` seconds with
    ``narration_gain_db``. ``music`` (optional) is mixed at ``music_gain_db``
    (-18 dB keeps it a bed under speech) and is lowered a further
    ``music_duck_db`` (-8 dB, a constant, not a follower) while the narration
    plays, with 0.3 s ramps just outside the narration so speech onset is never
    attenuated. ``music_loop`` repeats a short track to the timeline length;
    otherwise it simply ends. ``fade_in`` / ``fade_out`` shape the whole mix.

    Audio longer than the timeline is a mistake (the tail would be cut off
    silently), so ``build_episode`` raises ``EpisodeError`` unless
    ``trim_audio`` is true, which then truncates the mix to the timeline.
    """

    narration: str
    narration_offset: float = 0.0
    narration_gain_db: float = 0.0
    music: str = None
    music_gain_db: float = -18.0
    music_duck_db: float = -8.0
    music_loop: bool = False
    fade_in: float = 0.0
    fade_out: float = 0.0
    trim_audio: bool = False

    def __post_init__(self):
        _str(self.narration, "narration")
        _opt_str(self.music, "music")
        object.__setattr__(
            self,
            "narration_offset",
            _num(self.narration_offset, "narration_offset", low=0),
        )
        for name in ("narration_gain_db", "music_gain_db"):
            object.__setattr__(self, name, _num(getattr(self, name), name))
        duck = _num(self.music_duck_db, "music_duck_db")
        if duck > 0:
            raise EpisodeError("music_duck_db must be <= 0 (a reduction)")
        object.__setattr__(self, "music_duck_db", duck)
        for name in ("fade_in", "fade_out"):
            object.__setattr__(self, name, _num(getattr(self, name), name, low=0))
        for name in ("music_loop", "trim_audio"):
            if not isinstance(getattr(self, name), bool):
                raise EpisodeError(f"{name} must be true or false")


@dataclass(frozen=True)
class ShotSpec:
    """One picture segment: a Ken Burns still, a static still or a video clip.

    Exactly one of ``image`` / ``video``. Images take ``move`` (``auto``,
    ``push``, ``pull``, ``pan-*`` or ``static``) with ``zoom``, ``focus``,
    ``distance``, ``hold``, ``lead`` (see ``ken_burns``). Videos take
    ``clip_in`` / ``clip_out`` (JSON keys ``in`` / ``out``); ``duration``
    defaults to ``clip_out - clip_in`` or the rest of the file.
    ``transition`` joins this shot to the previous segment.
    """

    image: str = None
    video: str = None
    duration: float = None
    move: str = None
    zoom: object = None
    focus: tuple = None
    distance: float = None
    hold: float = None
    lead: float = None
    clip_in: float = None
    clip_out: float = None
    transition: str = "cut"
    transition_duration: float = None

    def __post_init__(self):
        if (self.image is None) == (self.video is None):
            raise EpisodeError("give exactly one of image or video")
        _opt_str(self.image, "image")
        _opt_str(self.video, "video")
        object.__setattr__(
            self, "duration", _opt_num(self.duration, "duration", positive=True)
        )
        if self.image is not None:
            if self.duration is None:
                raise EpisodeError("an image shot needs a duration")
            if self.clip_in is not None or self.clip_out is not None:
                raise EpisodeError("in/out apply to video shots only")
            move = "auto" if self.move is None else self.move
            _choice(
                move,
                "move",
                ("auto", "static", "push", "pull")
                + ("pan-left", "pan-right", "pan-up", "pan-down"),
            )
            object.__setattr__(self, "move", move)
            zoom = self.zoom
            if zoom is not None:
                if isinstance(zoom, (list, tuple)):
                    zoom = _pair(zoom, "zoom")
                else:
                    zoom = _num(zoom, "zoom")
                if min(zoom if isinstance(zoom, tuple) else (zoom,)) < 1:
                    raise EpisodeError("zoom must be >= 1")
            object.__setattr__(self, "zoom", zoom)
            if self.focus is not None:
                object.__setattr__(self, "focus", _pair(self.focus, "focus", 0, 1))
            object.__setattr__(
                self, "distance", _opt_num(self.distance, "distance", positive=True)
            )
            object.__setattr__(self, "hold", _opt_num(self.hold, "hold", low=0))
            object.__setattr__(self, "lead", _opt_num(self.lead, "lead", low=0))
            if self.move == "static" and any(
                v is not None for v in (self.zoom, self.focus, self.distance)
            ):
                raise EpisodeError("a static shot takes no zoom/focus/distance")
        else:
            for name in ("move", "zoom", "focus", "distance", "hold", "lead"):
                if getattr(self, name) is not None:
                    raise EpisodeError(f"{name} applies to image shots only")
            object.__setattr__(self, "clip_in", _opt_num(self.clip_in, "in", low=0))
            object.__setattr__(
                self, "clip_out", _opt_num(self.clip_out, "out", positive=True)
            )
            start = self.clip_in or 0.0
            if self.clip_out is not None and self.clip_out <= start:
                raise EpisodeError("out must be greater than in")
            if (
                self.duration is not None
                and self.clip_out is not None
                and abs(self.duration - (self.clip_out - start)) > _EPS
            ):
                raise EpisodeError("duration contradicts out - in; give one of them")
        _transition_fields(self, "")

    @property
    def is_ken_burns(self):
        """True for an image shot that moves."""
        return self.image is not None and self.move != "static"

    @property
    def source(self):
        """The media path."""
        return self.image if self.image is not None else self.video


@dataclass(frozen=True)
class ChapterSpec:
    """A top-left chapter tag over ``[start, end)`` (see ``chapter_tag``).

    ``period`` overrides ``EpisodeSpec.chapter_period`` for this chapter.
    """

    start: float
    end: float
    items: tuple
    number: int
    period: float = None
    seal: str = None
    position: str = "top_left"

    def __post_init__(self):
        object.__setattr__(self, "start", _num(self.start, "start", low=0))
        object.__setattr__(self, "end", _num(self.end, "end"))
        if self.end <= self.start:
            raise EpisodeError("end must be greater than start")
        items = [self.items] if isinstance(self.items, str) else self.items
        if not isinstance(items, (list, tuple)) or not items:
            raise EpisodeError("items must be a non-empty list of strings")
        object.__setattr__(self, "items", tuple(_str(i, "items entry") for i in items))
        if isinstance(self.number, bool) or not isinstance(self.number, int):
            raise EpisodeError("number must be an integer")
        if self.number < 1:
            raise EpisodeError("number must be >= 1")
        object.__setattr__(
            self, "period", _opt_num(self.period, "period", positive=True)
        )
        _opt_str(self.seal, "seal")
        _choice(self.position, "position", ("top_left", "bottom_left"))


@dataclass(frozen=True)
class QuoteSpec:
    """A vertical classical quotation starting at ``start`` (see ``quote``).

    ``duration`` defaults to the template's computed length. ``dynasty`` or
    ``year`` selects bamboo versus paper through ``resolve_medium``.
    """

    start: float
    text: str
    source: str = None
    dynasty: str = None
    year: int = None
    medium: str = "auto"
    title: str = None
    seal: str = None
    per_column: int = 10
    duration: float = None
    layout: str = "center"
    size: float = 62
    speed: float = 9.0
    hold: float = 3.0
    fade_out: float = 0.6
    dim: float = 0.35
    seed: int = 7

    def __post_init__(self):
        object.__setattr__(self, "start", _num(self.start, "start", low=0))
        _str(self.text, "text")
        for name in ("source", "dynasty", "title", "seal"):
            _opt_str(getattr(self, name), name)
        _str(self.medium, "medium")
        if self.year is not None and (
            isinstance(self.year, bool) or not isinstance(self.year, int)
        ):
            raise EpisodeError("year must be an integer")
        if isinstance(self.per_column, bool) or not isinstance(self.per_column, int):
            raise EpisodeError("per_column must be an integer")
        if self.per_column < 1:
            raise EpisodeError("per_column must be >= 1")
        object.__setattr__(
            self, "duration", _opt_num(self.duration, "duration", positive=True)
        )
        _choice(self.layout, "layout", ("center", "right", "left"))
        for name in ("size", "speed"):
            object.__setattr__(
                self, name, _num(getattr(self, name), name, positive=True)
            )
        for name in ("hold", "fade_out"):
            object.__setattr__(self, name, _num(getattr(self, name), name, low=0))
        dim = _num(self.dim, "dim", low=0)
        if dim > 1:
            raise EpisodeError("dim must lie in [0, 1]")
        object.__setattr__(self, "dim", dim)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EpisodeError("seed must be an integer")


def _cue_list(value, name):
    if isinstance(value, str):
        _str(value, name)
        return value
    if not isinstance(value, (list, tuple)):
        raise EpisodeError(f"{name} must be an SRT path or a list of cues")
    cues = []
    for k, cue in enumerate(value):
        label = f"{name}[{k}]"
        if not isinstance(cue, dict):
            raise EpisodeError(f"{label} must be an object")
        extra = set(cue) - {"start", "end", "text", "lang"}
        if extra:
            raise EpisodeError(f"{label}: unknown key(s) {sorted(extra)}")
        if not {"start", "end", "text"} <= set(cue):
            raise EpisodeError(f"{label} needs start, end and text")
        start = _num(cue["start"], f"{label}.start", low=0)
        end = _num(cue["end"], f"{label}.end")
        if end <= start:
            raise EpisodeError(f"{label}: end must be greater than start")
        item = {"start": start, "end": end, "text": _str(cue["text"], f"{label}.text")}
        if "lang" in cue:
            item["lang"] = _str(cue["lang"], f"{label}.lang")
        cues.append(item)
    return tuple(cues)


@dataclass(frozen=True)
class SubtitleSpec:
    """Burned-in subtitles (see ``subtitles.SubtitleLayer``).

    ``primary`` / ``secondary`` are an SRT path or a list of
    ``{"start", "end", "text"[, "lang"]}``; ``protected`` terms are never split
    across lines. ``overflow="split"`` turns a cue that does not fit into
    several shorter cues (``subtitles.reflow_cues``), timed from ``words``
    (an ASR word-timing JSON path or list, see ``subtitles.load_words``)
    when given.
    """

    primary: object
    secondary: object = None
    protected: tuple = ()
    primary_lang: str = "zh-TW"
    secondary_lang: str = "en"
    overflow: str = "wrap"
    words: object = None

    def __post_init__(self):
        object.__setattr__(self, "primary", _cue_list(self.primary, "primary"))
        if self.secondary is not None:
            object.__setattr__(
                self, "secondary", _cue_list(self.secondary, "secondary")
            )
        if not isinstance(self.protected, (list, tuple)):
            raise EpisodeError("protected must be a list of strings")
        object.__setattr__(
            self,
            "protected",
            tuple(_str(t, "protected entry") for t in self.protected),
        )
        _str(self.primary_lang, "primary_lang")
        _str(self.secondary_lang, "secondary_lang")
        _choice(self.overflow, "overflow", ("wrap", "split"))
        if self.words is not None and not isinstance(self.words, (str, list, tuple)):
            raise EpisodeError("words must be a JSON path or a list of words")
        if self.words is not None and self.overflow != "split":
            raise EpisodeError("words are only used with overflow='split'")


@dataclass(frozen=True)
class SceneOverlaySpec:
    """Chapter overlay sets: logo, watermark, vertical title, subscribe pop.

    ``chapters`` is a list of ``[start_seconds, title]``; each starts one
    overlay set of ``layout`` duration (see ``scene_overlay``). ``logo`` is
    an image path or text, ``layout`` a ``SceneLayout.to_dict`` subset.
    """

    chapters: tuple
    logo: str = None
    watermark: str = None
    cta_text: str = "立即訂閱"
    layout: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.chapters, (list, tuple)) or not self.chapters:
            raise EpisodeError("chapters must be a non-empty list of [start, title]")
        rows = []
        for k, row in enumerate(self.chapters):
            if not isinstance(row, (list, tuple)) or len(row) != 2:
                raise EpisodeError(f"chapters[{k}] must be [start, title]")
            rows.append(
                (
                    _num(row[0], f"chapters[{k}] start", low=0),
                    _str(row[1], f"chapters[{k}] title"),
                )
            )
        object.__setattr__(self, "chapters", tuple(rows))
        _opt_str(self.logo, "logo")
        _opt_str(self.watermark, "watermark")
        _str(self.cta_text, "cta_text")
        if not isinstance(self.layout, dict):
            raise EpisodeError("layout must be an object")
        object.__setattr__(self, "layout", _jsonable(self.layout, "layout"))
        self.scene_layout()

    def scene_layout(self):
        """Return the validated ``SceneLayout``."""
        from moviepy.ae.templates.scene_overlay import SceneLayout

        try:
            return SceneLayout.from_dict(self.layout)
        except (TypeError, ValueError) as error:
            raise EpisodeError(f"scene_overlay.layout: {error}") from None


_FONT_ROLES = ("title", "body", "quote", "chapter", "subtitle")
_RENAMES = {"in": "clip_in", "out": "clip_out"}


@dataclass(frozen=True)
class EpisodeSpec:
    """The whole episode as data.

    Parameters
    ----------
    preset : str
        ``get_preset`` name.
    preset_overrides : dict
        ``ChannelPreset.with_overrides`` keywords (``size``, ``fps``,
        ``overlay_opacity``, ``hold``, ``fade``, ``subtitles`` ...). ``fonts``
        and ``palette`` entries are merged into the preset's.
    fonts : dict
        Per-role font paths: ``title``/``body`` (cards), ``quote``,
        ``chapter``, ``subtitle``. ``null`` means Pillow's default font and is
        only valid for ``title``, ``body`` and ``subtitle`` (Latin text).
    background : BackgroundSpec or None
        Media under the cards; without it cards sit on the dip colour.
    intro, outro : BookendSpec or None
        Opening and closing title cards.
    shots : sequence of ShotSpec
        Picture segments in time order.
    chapters, quotes : sequences of ChapterSpec / QuoteSpec
        Overlays placed on the built timeline.
    subtitles : SubtitleSpec or None
        Burned-in subtitle files, drawn above every other layer.
    scene_overlay : SceneOverlaySpec or None
        Per-chapter logo, watermark, vertical title and subscribe button.
    name_tags : sequence of NameTag
        Character name cards beside a portrait (see ``name_tag.NameTag``;
        ``subject_box`` in 1920x1080 reference pixels).
    audio : AudioSpec or None
        Narration/music mix attached to the composition (see ``AudioSpec``).
    chapter_period : float
        Default seconds per chapter item (NLH: 5.5).
    dip_color : tuple
        RGB behind the picture, and the colour of ``dip`` transitions.
    seed : int
        Seed for chapter textures.

    Dict members of the lists may be given instead of dataclass instances.
    Relative paths are resolved against the JSON folder by ``from_json``.
    """

    name: str = "episode"
    preset: str = "nightlamp_history"
    preset_overrides: dict = field(default_factory=dict)
    fonts: dict = field(default_factory=dict)
    background: BackgroundSpec = None
    intro: BookendSpec = None
    outro: BookendSpec = None
    shots: tuple = ()
    chapters: tuple = ()
    quotes: tuple = ()
    subtitles: SubtitleSpec = None
    chapter_period: float = 5.5
    dip_color: tuple = (0, 0, 0)
    seed: int = 7
    audio: AudioSpec = None
    scene_overlay: SceneOverlaySpec = None
    name_tags: tuple = ()

    def __post_init__(self):
        _str(self.name, "name")
        _str(self.preset, "preset")
        if not isinstance(self.preset_overrides, dict):
            raise EpisodeError("preset_overrides must be an object")
        object.__setattr__(
            self,
            "preset_overrides",
            _jsonable(self.preset_overrides, "preset_overrides"),
        )
        if not isinstance(self.fonts, dict):
            raise EpisodeError("fonts must be an object")
        for role, path in self.fonts.items():
            _choice(role, "fonts key", _FONT_ROLES)
            if path is None and role in ("quote", "chapter"):
                raise EpisodeError(f"fonts.{role} needs a real font file")
            if path is not None:
                _str(path, f"fonts.{role}")
        object.__setattr__(self, "fonts", dict(self.fonts))
        for name, cls in (
            ("background", BackgroundSpec),
            ("intro", BookendSpec),
            ("outro", BookendSpec),
            ("subtitles", SubtitleSpec),
            ("audio", AudioSpec),
            ("scene_overlay", SceneOverlaySpec),
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, cls):
                object.__setattr__(self, name, _from_dict(cls, value, name))
        for name, cls, renames in (
            ("shots", ShotSpec, _RENAMES),
            ("chapters", ChapterSpec, None),
            ("quotes", QuoteSpec, None),
        ):
            items = getattr(self, name)
            if not isinstance(items, (list, tuple)):
                raise EpisodeError(f"{name} must be a list")
            object.__setattr__(
                self,
                name,
                tuple(
                    (
                        item
                        if isinstance(item, cls)
                        else _from_dict(cls, item, f"{name}[{k}]", renames)
                    )
                    for k, item in enumerate(items)
                ),
            )
        if not self.shots:
            raise EpisodeError("an episode needs at least one shot")
        if not isinstance(self.name_tags, (list, tuple)):
            raise EpisodeError("name_tags must be a list")
        tags = []
        for k, item in enumerate(self.name_tags):
            try:
                tags.append(
                    item if isinstance(item, NameTag) else NameTag.from_dict(item)
                )
            except (TypeError, ValueError) as error:
                raise EpisodeError(f"name_tags[{k}]: {error}") from None
        object.__setattr__(self, "name_tags", tuple(tags))
        object.__setattr__(
            self,
            "chapter_period",
            _num(self.chapter_period, "chapter_period", positive=True),
        )
        if not isinstance(self.dip_color, (list, tuple)) or len(self.dip_color) != 3:
            raise EpisodeError("dip_color must be three 0..255 values")
        colour = tuple(_num(c, "dip_color", low=0) for c in self.dip_color)
        if max(colour) > 255:
            raise EpisodeError("dip_color must be three 0..255 values")
        object.__setattr__(self, "dip_color", colour)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise EpisodeError("seed must be an integer")
        if self.intro is None and self.shots[0].transition != "cut":
            raise EpisodeError("shots[0]: no previous segment, transition must be cut")
        if self.intro is None and self.shots[0].transition_duration is not None:
            raise EpisodeError("shots[0]: no previous segment")
        # Early preset check so mistakes surface when the spec is created.
        self.resolve_preset()
        # Chapters may not overlap each other.
        ordered = sorted(self.chapters, key=lambda c: c.start)
        for a, b in zip(ordered, ordered[1:]):
            if b.start < a.end - _EPS:
                raise EpisodeError(
                    f"chapters overlap: [{a.start:g}, {a.end:g}) and "
                    f"[{b.start:g}, {b.end:g})"
                )

    # -- preset ------------------------------------------------------------ #

    def resolve_preset(self):
        """Return the ``ChannelPreset`` with ``preset_overrides`` applied."""
        from moviepy.ae.templates.presets import PRESETS

        try:
            base = PRESETS[self.preset]
        except KeyError:
            raise EpisodeError(
                f"unknown preset {self.preset!r}; choose from {sorted(PRESETS)}"
            ) from None
        changes = dict(self.preset_overrides)
        for key in ("fonts", "palette"):
            if key in changes:
                changes[key] = {**getattr(base, key), **changes[key]}
        if "size" in changes:
            changes["size"] = tuple(changes["size"])
        try:
            return base.with_overrides(**changes) if changes else base
        except (TypeError, ValueError) as error:
            raise EpisodeError(f"preset_overrides: {error}") from None

    # -- JSON ---------------------------------------------------------------- #

    def to_dict(self):
        """Return a JSON-compatible dictionary (``from_dict`` inverse)."""
        value = {
            "name": self.name,
            "preset": self.preset,
            "preset_overrides": _to_plain(self.preset_overrides),
            "fonts": dict(self.fonts),
            "background": (
                None if self.background is None else _spec_dict(self.background)
            ),
            "intro": None if self.intro is None else _spec_dict(self.intro),
            "outro": None if self.outro is None else _spec_dict(self.outro),
            "shots": [_spec_dict(s, _RENAMES) for s in self.shots],
            "chapters": [_spec_dict(c) for c in self.chapters],
            "quotes": [_spec_dict(q) for q in self.quotes],
            "subtitles": None if self.subtitles is None else _spec_dict(self.subtitles),
            "chapter_period": self.chapter_period,
            "dip_color": list(self.dip_color),
            "seed": self.seed,
            "audio": None if self.audio is None else _spec_dict(self.audio),
            "scene_overlay": (
                None if self.scene_overlay is None else _spec_dict(self.scene_overlay)
            ),
            "name_tags": [tag.to_dict() for tag in self.name_tags],
        }
        return value

    @classmethod
    def from_dict(cls, data, base_dir=None):
        """Build a spec from ``to_dict`` output.

        Relative media, SRT and font paths are resolved against ``base_dir``
        (when given). Unknown keys raise ``EpisodeError``.
        """
        if not isinstance(data, dict):
            raise EpisodeError("an episode must be a JSON object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise EpisodeError(f"unknown episode key(s): {unknown}")
        data = json.loads(json.dumps(data))  # private deep copy
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

        A file's relative paths are resolved against its folder; for JSON text
        pass ``base_dir`` (otherwise paths are kept as written).
        """
        if isinstance(source, Path) or (
            isinstance(source, str) and not source.lstrip().startswith("{")
        ):
            path = Path(source)
            try:
                text = path.read_text(encoding="utf-8-sig")
            except OSError as error:
                raise EpisodeError(
                    f"cannot read episode JSON {path}: {error}"
                ) from None
            base_dir = path.parent if base_dir is None else base_dir
        else:
            text = source
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise EpisodeError(f"invalid episode JSON: {error}") from None
        return cls.from_dict(data, base_dir)


def _resolve_paths(data, base):
    def fix(value):
        if not isinstance(value, str) or not value:
            return value
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else base / path)

    def fix_in(mapping, keys):
        if isinstance(mapping, dict):
            for key in keys:
                if key in mapping:
                    mapping[key] = fix(mapping[key])

    fix_in(data.get("background"), ("source",))
    fix_in(data.get("fonts"), tuple(data.get("fonts") or ()))
    for shot in data.get("shots") or ():
        fix_in(shot, ("image", "video"))
    fix_in(data.get("audio"), ("narration", "music"))
    subs = data.get("subtitles")
    if isinstance(subs, dict):
        fix_in(subs, ("primary", "secondary", "words"))
    overlay = data.get("scene_overlay")
    if isinstance(overlay, dict) and isinstance(overlay.get("logo"), str):
        # A logo is an image path or plain text; only rebase existing files.
        candidate = base / Path(overlay["logo"]).expanduser()
        if not Path(overlay["logo"]).is_absolute() and candidate.is_file():
            overlay["logo"] = str(candidate)


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _opacity(origin, fade_in, fade_out):
    """Opacity ``Property`` (percent) with optional fade windows, or ``None``.

    Layer properties are evaluated in layer time (composition time minus the
    layer's ``start_time``), so the windows, given in composition time, are
    shifted by ``origin`` (the layer's ``start_time``).
    """
    from moviepy.ae.properties import Keyframe, Property

    keys = []
    if fade_in is not None:
        keys += [(fade_in[0], 0.0), (fade_in[1], 100.0)]
    if fade_out is not None:
        keys += [(fade_out[0], 100.0), (fade_out[1], 0.0)]
    if not keys:
        return None
    keys.sort(key=lambda k: k[0])
    unique = []
    for t, v in keys:
        if unique and t <= unique[-1][0] + 1e-9:
            if v < unique[-1][1] or t == unique[-1][0]:
                unique[-1] = (unique[-1][0], min(unique[-1][1], v))
            continue
        unique.append((t, v))
    return Property(100.0, keyframes=[Keyframe(t - origin, v) for t, v in unique])


def _cover_transform(size, source_size, opacity):
    from moviepy.ae.transform import Transform

    width, height = size
    sw, sh = source_size
    scale = max(width / sw, height / sh) * 100.0
    kwargs = {} if opacity is None else {"opacity": opacity}
    return Transform(
        anchor_point=(sw / 2, sh / 2),
        position=(width / 2, height / 2),
        scale=(scale, scale),
        **kwargs,
    )


def _check_file(path, label):
    if not Path(path).is_file():
        raise EpisodeError(f"{label} not found: {path}")


def _read_cues(value, lang, label, sources):
    from moviepy.ae.templates.subtitles import Cue, parse_srt

    if isinstance(value, str):
        _check_file(value, label)
        sources[value] = _sha256(value)
        try:
            return parse_srt(Path(value).read_text(encoding="utf-8-sig"), lang)
        except ValueError as error:
            raise EpisodeError(f"{label}: {error}") from None
    return [Cue(c["start"], c["end"], c["text"], c.get("lang", lang)) for c in value]


def _transition_window(kind, duration, previous):
    """Return ``(overlap, fade_in, fade_out)`` offsets for a transition.

    Offsets are relative to the incoming segment start: ``fade_in`` is
    ``(begin, end)`` for the incoming layer, ``fade_out`` for the outgoing
    one (relative to the incoming start as well).
    """
    if kind == "crossfade":
        return duration, (0.0, duration), None
    if kind == "dip":
        half = duration / 2
        return 0.0, (0.0, half), (-half, 0.0)
    return 0.0, None, None


_DUCK_RAMP = 0.3  # seconds of music ramp just outside the narration


def _db_gain(value):
    return 10.0 ** (value / 20.0)


def _audio_file(path, label, sources, clips):
    from moviepy import AudioFileClip

    _check_file(path, label)
    sources[str(path)] = _sha256(path)
    try:
        clip = AudioFileClip(str(path))
    except Exception as error:  # ffmpeg/moviepy raise assorted types
        raise EpisodeError(f"{label} cannot be read as audio: {error}") from None
    clips.append(clip)
    if not clip.duration or clip.duration <= 0:
        raise EpisodeError(f"{label} is empty")
    return clip


def _build_audio(audio, total, sources, clips):
    """Mix ``AudioSpec`` for a ``total`` second timeline -> ``(clip, info)``."""
    import numpy as np

    from moviepy import CompositeAudioClip
    from moviepy.audio import fx as afx

    narration = _audio_file(audio.narration, "audio.narration", sources, clips)
    start = audio.narration_offset
    end = start + narration.duration
    if start >= total - _EPS:
        raise EpisodeError(
            f"audio.narration_offset {start:g} s is past the {total:g} s timeline"
        )
    if end > total + 1e-3 and not audio.trim_audio:
        raise EpisodeError(
            f"audio.narration ends at {end:g} s, past the {total:g} s timeline; "
            "lengthen the shots or set audio.trim_audio to true"
        )
    voice = narration
    if audio.narration_gain_db:
        voice = voice.with_effects(
            [afx.MultiplyVolume(_db_gain(audio.narration_gain_db))]
        )
    voice = voice.with_start(start)
    parts = [voice]
    info = {
        "narration": {
            "path": str(audio.narration),
            "sha256": sources[str(audio.narration)],
            "duration": float(narration.duration),
            "offset": start,
            "gain_db": audio.narration_gain_db,
        },
        "music": None,
        "fade_in": audio.fade_in,
        "fade_out": audio.fade_out,
        "trim_audio": audio.trim_audio,
    }
    if audio.music is not None:
        music = _audio_file(audio.music, "audio.music", sources, clips)
        if audio.music_loop and music.duration < total:
            music = music.with_effects([afx.AudioLoop(duration=total)])
        if music.duration > total:
            music = music.subclipped(0, total)
        music = music.with_effects([afx.MultiplyVolume(_db_gain(audio.music_gain_db))])
        duck = _db_gain(audio.music_duck_db)
        duck_from, duck_to = start, min(end, total)
        if duck < 1.0:
            ramp = _DUCK_RAMP

            def ducked(get_frame, t, _a=duck_from, _b=duck_to, _g=duck):
                times = np.asarray(t, dtype=float)
                inside = np.minimum(
                    (times - (_a - ramp)) / ramp, ((_b + ramp) - times) / ramp
                )
                gain = 1.0 - (1.0 - _g) * np.clip(inside, 0.0, 1.0)
                frame = get_frame(t)
                return frame * (gain if times.ndim == 0 else gain[:, None])

            music = music.transform(ducked, keep_duration=True)
        parts.insert(0, music)
        info["music"] = {
            "path": str(audio.music),
            "sha256": sources[str(audio.music)],
            "duration": float(min(music.duration, total)),
            "gain_db": audio.music_gain_db,
            "duck_db": audio.music_duck_db,
            "duck_span": [duck_from, duck_to],
            "loop": audio.music_loop,
        }
    if audio.fade_in + audio.fade_out > total + _EPS:
        raise EpisodeError("audio.fade_in + audio.fade_out exceed the timeline")
    mix = CompositeAudioClip(parts)
    mix.fps = max(int(getattr(c, "fps", 0) or 0) for c in parts) or 44100
    mix = mix.with_duration(total)
    fades = []
    if audio.fade_in:
        fades.append(afx.AudioFadeIn(audio.fade_in))
    if audio.fade_out:
        fades.append(afx.AudioFadeOut(audio.fade_out))
    if fades:
        mix = mix.with_effects(fades)
    info["duration"] = float(total)
    return mix, info


def build_episode(spec):
    """Assemble ``spec`` into a ``Composition`` at the preset size and fps.

    Parameters
    ----------
    spec : EpisodeSpec

    Returns
    -------
    Composition
        Layers (top first): ``Subtitles``, ``Outro``, ``Intro``,
        ``Chapter NN``, ``Quote NN``, ``Shot NN``. With ``spec.audio`` the
        mixed soundtrack is attached as ``comp.audio``. ``comp.episode`` holds the
        timeline and provenance used by ``episode_report``; ``comp.preset`` is
        the resolved ``ChannelPreset`` and ``comp.episode_clips`` the opened
        media clips (close them with ``clip.close()`` when finished).

    Raises
    ------
    EpisodeError
        For missing files, transitions that leave the previous focus hold,
        overlays that leave the timeline or overlap a card, overlapping
        quotes, cues outside the timeline, and narration longer than the
        timeline (unless ``audio.trim_audio``).

    Examples
    --------
    >>> import numpy as np, tempfile, os
    >>> from PIL import Image
    >>> d = tempfile.mkdtemp()
    >>> p = os.path.join(d, "a.png")
    >>> Image.fromarray(np.zeros((9, 16, 3), np.uint8)).save(p)
    >>> spec = EpisodeSpec(
    ...     preset_overrides={"size": [32, 18], "fps": 4, "hold": 0.5},
    ...     shots=[{"image": p, "duration": 2}, {"image": p, "duration": 2,
    ...            "transition": "crossfade", "transition_duration": 0.5}])
    >>> comp = build_episode(spec)
    >>> comp.size, comp.duration
    ((32, 18), 3.5)
    """
    from moviepy import ImageClip, VideoFileClip
    from moviepy.ae.composition import Composition
    from moviepy.ae.layers.av import AVLayer
    from moviepy.ae.layers.comp import CompLayer
    from moviepy.ae.templates.background import media_background
    from moviepy.ae.templates.chapter_tag import chapter_tag
    from moviepy.ae.templates.ken_burns import ken_burns
    from moviepy.ae.templates.name_tag import add_name_tags
    from moviepy.ae.templates.quote import vertical_quote
    from moviepy.ae.templates.scene_overlay import add_scene_overlays
    from moviepy.ae.templates.subtitles import load_words, subtitle_layer
    from moviepy.ae.templates.title_card import build_title_card
    from moviepy.ae.transform import Transform

    if not isinstance(spec, EpisodeSpec):
        raise TypeError("spec must be an EpisodeSpec")
    preset = spec.resolve_preset()
    size = preset.size
    sources = {}
    clips = []

    def note(path, label):
        _check_file(path, label)
        sources[str(path)] = _sha256(path)

    # -- 1. segments: intro, shots, outro --------------------------------- #
    background = None
    if spec.background is not None:
        bg = spec.background
        note(bg.source, "background source")
        background = media_background(
            bg.source,
            preset,
            frame_time=bg.frame_time,
            focal_point=bg.focal_point,
            overlay_opacity=bg.overlay_opacity,
        )
    card_fonts = {k: spec.fonts[k] for k in ("title", "body") if k in spec.fonts}

    segments = []  # dicts: kind, name, source, duration, transition, td, make
    if spec.intro is not None:
        card = build_title_card(
            spec.intro.card_spec("intro"), preset, background, fonts=card_fonts
        )
        segments.append(
            dict(kind="intro", name="Intro", source=None, comp=card,
                 duration=card.duration, transition=spec.intro.transition,
                 td=spec.intro.transition_duration, hold=None)
        )  # fmt: skip
    kb_index = 0
    for number, shot in enumerate(spec.shots, 1):
        label = f"shots[{number - 1}]"
        name = f"Shot {number:02d}"
        note(shot.source, f"{label} {'image' if shot.image else 'video'}")
        if shot.image is not None and shot.move != "static":
            try:
                comp = ken_burns(
                    shot.image,
                    preset,
                    duration=shot.duration,
                    move=shot.move,
                    index=kb_index,
                    zoom=shot.zoom,
                    focus=shot.focus,
                    distance=shot.distance,
                    hold=shot.hold,
                    lead=0.0 if shot.lead is None else shot.lead,
                )
            except ValueError as error:
                raise EpisodeError(f"{label}: {error}") from None
            kb_index += 1
            hold = preset.hold if shot.hold is None else shot.hold
            seg = dict(comp=comp, duration=shot.duration, hold=hold)
        elif shot.image is not None:
            clip = ImageClip(shot.image).with_duration(shot.duration)
            clips.append(clip)
            seg = dict(clip=clip, duration=shot.duration, hold=None)
        else:
            clip = VideoFileClip(shot.video, audio=False)
            clips.append(clip)
            start_in = shot.clip_in or 0.0
            end_out = shot.clip_out if shot.clip_out is not None else clip.duration
            if shot.duration is not None and shot.clip_out is None:
                end_out = start_in + shot.duration
            if end_out > clip.duration + 1e-3:
                raise EpisodeError(
                    f"{label}: out {end_out:g} s is past the end of the video "
                    f"({clip.duration:g} s)"
                )
            seg = dict(clip=clip, duration=end_out - start_in, hold=None,
                       clip_in=start_in)  # fmt: skip
        segments.append(
            dict(kind="shot", name=name, source=shot.source,
                 transition=shot.transition, td=shot.transition_duration, **seg)
        )  # fmt: skip
    if spec.outro is not None:
        card = build_title_card(
            spec.outro.card_spec("outro"), preset, background, fonts=card_fonts
        )
        segments.append(
            dict(kind="outro", name="Outro", source=None, comp=card,
                 duration=card.duration, transition=spec.outro.transition,
                 td=spec.outro.transition_duration, hold=None)
        )  # fmt: skip

    # -- 2. timeline ------------------------------------------------------- #
    cursor = 0.0
    previous = None
    for k, seg in enumerate(segments):
        kind = seg["transition"]
        td = seg["td"]
        if kind != "cut" and td is None:
            td = preset.fade
        if kind != "cut" and previous is None:
            raise EpisodeError(f"{seg['name']}: no previous segment to {kind} from")
        overlap, fade_in, fade_out = (0.0, None, None)
        if previous is not None and kind != "cut":
            need = td / 2 if kind == "dip" else td
            if need > seg["duration"] + _EPS or need > previous["duration"] + _EPS:
                raise EpisodeError(
                    f"{seg['name']}: {kind} of {td:g} s is longer than a neighbour"
                )
            if previous["kind"] == "shot" and previous["hold"] is not None:
                if need > previous["hold"] + _EPS:
                    raise EpisodeError(
                        f"{seg['name']}: {kind} of {td:g} s must lie inside the "
                        f"{previous['name']} focus hold of {previous['hold']:g} s"
                    )
            overlap, fade_in, fade_out = _transition_window(kind, td, previous)
        seg["transition_duration"] = 0.0 if kind == "cut" else td
        seg["start"] = max(0.0, cursor - overlap)
        seg["end"] = seg["start"] + seg["duration"]
        if fade_in is not None:
            seg["fade_in"] = (seg["start"] + fade_in[0], seg["start"] + fade_in[1])
        if fade_out is not None:
            previous["fade_out"] = (
                seg["start"] + fade_out[0],
                seg["start"] + fade_out[1],
            )
        cursor = seg["end"]
        previous = seg
    total = cursor
    cards = [(s["start"], s["end"], s["name"]) for s in segments if s["kind"] != "shot"]

    # -- 3. overlays, validated against the built timeline ----------------- #
    def inside(label, start, end):
        if start < -_EPS or end > total + _EPS:
            raise EpisodeError(
                f"{label} [{start:g}, {end:g}) lies outside the {total:g} s timeline"
            )
        for c_start, c_end, c_name in cards:
            if start < c_end - _EPS and end > c_start + _EPS:
                raise EpisodeError(
                    f"{label} [{start:g}, {end:g}) overlaps the {c_name} card "
                    f"[{c_start:g}, {c_end:g})"
                )

    chapter_tags = []
    for k, chapter in enumerate(spec.chapters, 1):
        inside(f"chapters[{k - 1}]", chapter.start, chapter.end)
        tag = chapter_tag(
            chapter.items,
            chapter.number,
            chapter.end - chapter.start,
            preset,
            period=chapter.period or spec.chapter_period,
            font=spec.fonts.get("chapter"),
            position=chapter.position,
            seal=chapter.seal,
            seed=spec.seed,
        )
        chapter_tags.append((k, chapter, tag))

    quote_comps = []
    for k, quote in enumerate(spec.quotes, 1):
        try:
            comp = vertical_quote(
                quote.text,
                quote.source,
                preset,
                dynasty=quote.dynasty,
                medium=quote.medium,
                title=quote.title,
                seal=quote.seal,
                per_column=quote.per_column,
                duration=quote.duration,
                font=spec.fonts.get("quote"),
                year=quote.year,
                size=quote.size,
                layout=quote.layout,
                speed=quote.speed,
                hold=quote.hold,
                fade_out=quote.fade_out,
                dim=quote.dim,
                seed=spec.seed,
            )
        except ValueError as error:
            raise EpisodeError(f"quotes[{k - 1}]: {error}") from None
        inside(f"quotes[{k - 1}]", quote.start, quote.start + comp.duration)
        quote_comps.append((k, quote, comp))
    ordered = sorted(quote_comps, key=lambda q: q[1].start)
    for (_, a, ca), (_, b, _cb) in zip(ordered, ordered[1:]):
        if b.start < a.start + ca.duration - _EPS:
            raise EpisodeError(
                f"quotes overlap: one starting at {a.start:g} s runs to "
                f"{a.start + ca.duration:g} s, next starts at {b.start:g} s"
            )

    cue_sets = {}
    if spec.subtitles is not None:
        subs = spec.subtitles
        cue_sets["primary"] = _read_cues(
            subs.primary, subs.primary_lang, "subtitles.primary", sources
        )
        if subs.secondary is not None:
            cue_sets["secondary"] = _read_cues(
                subs.secondary, subs.secondary_lang, "subtitles.secondary", sources
            )
        if isinstance(subs.words, str):
            _check_file(subs.words, "subtitles.words")
            sources[subs.words] = _sha256(subs.words)
        for lane, cues in cue_sets.items():
            for cue in cues:
                if cue.end > total + _EPS:
                    raise EpisodeError(
                        f"subtitles.{lane} cue [{cue.start:g}, {cue.end:g}) "
                        f"{cue.text[:20]!r} lies outside the {total:g} s timeline"
                    )
    for k, tag in enumerate(spec.name_tags):
        inside(f"name_tags[{k}]", tag.start, tag.end)
    if spec.scene_overlay is not None:
        overlay = spec.scene_overlay
        layout = overlay.scene_layout()
        for start, _ in overlay.chapters:
            inside("scene_overlay", start, start + layout.duration)
        if overlay.logo and Path(overlay.logo).is_file():
            note(overlay.logo, "scene_overlay.logo")
    for role in spec.fonts:
        if spec.fonts[role] is not None:
            note(spec.fonts[role], f"fonts.{role}")

    mix, audio_info = (None, None)
    if spec.audio is not None:
        mix, audio_info = _build_audio(spec.audio, total, sources, clips)

    # -- 4. composition, bottom to top ------------------------------------- #
    comp = Composition(
        size=size,
        fps=preset.fps,
        duration=total,
        name=spec.name,
        bg_color=spec.dip_color,
    )

    def place(layer_cls, source, name, start, end, transform=None, **kw):
        kw.setdefault("start_time", start)
        layer = layer_cls(
            source, name, transform=transform, in_point=start, out_point=end, **kw
        )
        return comp.add_layer(layer)

    def segment_layer(seg):
        origin = seg["start"] - seg.get("clip_in", 0.0)
        opacity = _opacity(origin, seg.get("fade_in"), seg.get("fade_out"))
        if "comp" in seg:
            transform = None if opacity is None else Transform(opacity=opacity)
            return place(
                CompLayer, seg["comp"], seg["name"], seg["start"], seg["end"], transform
            )
        clip = seg["clip"]
        return place(
            AVLayer,
            clip,
            seg["name"],
            seg["start"],
            seg["end"],
            _cover_transform(size, clip.size, opacity),
            start_time=origin,
        )

    for seg in segments:
        if seg["kind"] == "shot":
            segment_layer(seg)
    for k, quote, qcomp in quote_comps:
        place(CompLayer, qcomp, f"Quote {k:02d}", quote.start,
              quote.start + qcomp.duration)  # fmt: skip
    for k, chapter, tag in chapter_tags:
        transform = Transform(
            anchor_point=(0, 0), position=(tag.meta["x"], tag.meta["y"])
        )
        place(CompLayer, tag.composition, f"Chapter {k:02d}", chapter.start,
              chapter.end, transform)  # fmt: skip
    for seg in segments:
        if seg["kind"] != "shot":
            segment_layer(seg)
    if spec.scene_overlay is not None:
        overlay = spec.scene_overlay
        logo = overlay.logo
        if logo and Path(logo).is_file():
            logo = Path(logo)
        try:
            add_scene_overlays(
                comp,
                overlay.scene_layout(),
                overlay.chapters,
                logo=logo,
                watermark=overlay.watermark,
                cta_text=overlay.cta_text,
                font=spec.fonts.get("title"),
            )
        except ValueError as error:
            raise EpisodeError(f"scene_overlay: {error}") from None
    if spec.name_tags:
        try:
            add_name_tags(
                comp,
                spec.name_tags,
                preset=preset,
                font=spec.fonts.get("title"),
                role_font=spec.fonts.get("body"),
            )
        except ValueError as error:
            raise EpisodeError(f"name_tags: {error}") from None
    subtitle = None
    if cue_sets:
        subs = spec.subtitles
        font = spec.fonts.get("subtitle", MISSING)
        kwargs = {"font": "default"} if font is None else {}
        if isinstance(font, str):
            kwargs = {"font": font}
        words = None
        if subs.words is not None:
            try:
                words = load_words(subs.words)
            except (KeyError, TypeError, ValueError) as error:
                raise EpisodeError(f"subtitles.words: {error}") from None
        try:
            subtitle = subtitle_layer(
                cue_sets["primary"],
                preset,
                secondary=cue_sets.get("secondary"),
                protected=subs.protected,
                size=size,
                overflow=subs.overflow,
                words=words,
                **kwargs,
            )
        except ValueError as error:
            raise EpisodeError(f"subtitles: {error}") from None
        subtitle.in_point = 0.0
        subtitle.out_point = total
        comp.add_layer(subtitle)

    # -- 5. QA record ------------------------------------------------------ #
    timeline = []
    for seg in segments:
        timeline.append(
            {
                "kind": seg["kind"],
                "name": seg["name"],
                "source": None if seg["source"] is None else Path(seg["source"]).name,
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "transition": seg["transition"],
                "transition_duration": seg["transition_duration"],
                "hold": seg["hold"],
            }
        )
    if mix is not None:
        comp.audio = mix
    comp.preset = preset
    comp.spec = spec
    comp.episode_clips = clips
    comp.episode = {
        "name": spec.name,
        "preset": spec.preset,
        "preset_resolved": preset.to_dict(),
        "duration": total,
        "timeline": timeline,
        "chapters": [
            {
                "name": f"Chapter {k:02d}",
                "start": c.start,
                "end": c.end,
                "number": c.number,
                "items": list(c.items),
            }
            for k, c, _ in chapter_tags
        ],
        "quotes": [
            {
                "name": f"Quote {k:02d}",
                "start": q.start,
                "end": q.start + qc.duration,
                "medium": qc.quote_meta["medium"],
            }
            for k, q, qc in quote_comps
        ],
        "subtitles": {
            lane: {
                "cues": len(cues),
                "rendered_cues": len(subtitle._lanes[lane]["cues"]),
                "first": cues[0].start if cues else None,
                "last": cues[-1].end if cues else None,
            }
            for lane, cues in cue_sets.items()
        },
        "scene_overlay": (
            None
            if spec.scene_overlay is None
            else [
                {"start": start, "title": title}
                for start, title in spec.scene_overlay.chapters
            ]
        ),
        "name_tags": [
            {"name": tag.name, "start": tag.start, "end": tag.end}
            for tag in spec.name_tags
        ],
        "sources": dict(sorted(sources.items())),
    }
    if audio_info is not None:
        comp.episode["audio"] = audio_info
    return comp


def episode_report(comp):
    """Return a JSON-able QA dictionary for a ``build_episode`` composition.

    Keys: ``name``, ``preset`` (name) and ``preset_resolved`` (full values),
    ``size``, ``fps``, ``duration``, ``frames``, ``timeline`` (segments with
    start/end/transition), ``chapters``, ``quotes``, ``subtitles`` (cue counts
    and span), ``layers`` (top first: name, type, in/out), ``sources`` (path ->
    sha256 of every media, SRT, audio and font file), ``audio`` (only when the
    spec has audio: narration/music path, sha256, duration, gains) and
    ``top_layer``.

    Examples
    --------
    >>> episode_report(object())
    Traceback (most recent call last):
    ...
    ValueError: not an episode composition; use build_episode()
    """
    info = getattr(comp, "episode", None)
    if info is None:
        raise ValueError("not an episode composition; use build_episode()")
    layers = [
        {
            "name": layer.name,
            "type": type(layer).__name__,
            "in_point": float(layer.in_point),
            "out_point": float(layer.out_point),
        }
        for layer in comp.layers
    ]
    report = json.loads(json.dumps(info))
    report.update(
        size=list(comp.size),
        fps=float(comp.fps),
        frames=int(math.ceil(comp.duration * comp.fps - 1e-9)),
        layers=layers,
        top_layer=layers[0]["name"] if layers else None,
    )
    return report


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

PREVIEW_SIZE = (480, 270)
PREVIEW_FPS = 12
_MASTER_RATE = 48000


def _default_stills(info, duration):
    """``[(label, time)]``: intro/outro middle, chapter start+1 s, quote middle."""
    out = []
    for seg in info["timeline"]:
        if seg["kind"] == "intro":
            out.append(("intro", (seg["start"] + seg["end"]) / 2))
    for k, chapter in enumerate(info["chapters"], 1):
        out.append((f"chapter{k:02d}", min(chapter["start"] + 1.0, chapter["end"])))
    for k, quote in enumerate(info["quotes"], 1):
        out.append((f"quote{k:02d}", (quote["start"] + quote["end"]) / 2))
    for seg in info["timeline"]:
        if seg["kind"] == "outro":
            out.append(("outro", (seg["start"] + seg["end"]) / 2))
    if not out:
        out.append(("mid", duration / 2))
    return out


def _export_wav(audio, path, count, fps, rate=_MASTER_RATE):
    """Write ``audio`` as exact-length 16-bit PCM for ``count`` frames at ``fps``.

    Returns False (writing nothing) when ``count`` frames hold no whole number
    of samples at ``rate``.
    """
    import wave
    from fractions import Fraction

    import numpy as np

    samples = Fraction(count) * rate / Fraction(str(fps)).limit_denominator(1000000)
    if samples.denominator != 1:
        return False
    samples = int(samples)
    channels = 2
    chunk = rate  # one second at a time keeps memory flat
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        for first in range(0, samples, chunk):
            last = min(first + chunk, samples)
            times = np.arange(first, last) / rate
            block = np.zeros((last - first, channels))
            live = times < audio.duration - 1e-9
            if live.any():
                data = np.asarray(audio.get_frame(times[live]), dtype=float)
                if data.ndim == 1:
                    data = data[:, None]
                block[live] = data if data.shape[1] == 2 else np.repeat(data, 2, 1)
            pcm = (np.clip(block, -1.0, 1.0) * 32767.0).astype("<i2")
            handle.writeframes(pcm.tobytes())
    return True


def _episode_factory(spec_dict):
    """Rebuild the composition of ``spec_dict`` (picklable worker factory)."""
    return build_episode(EpisodeSpec.from_dict(spec_dict))


def render_episode(
    spec_or_path,
    output_dir,
    *,
    preview=False,
    stills=None,
    master=False,
    overwrite=False,
    workers=None,
    encoder="auto",
):
    """Render an episode into ``output_dir`` and return what was written.

    Parameters
    ----------
    spec_or_path : EpisodeSpec, dict or path
        A spec, its ``to_dict`` form or an ``episode.json`` path.
    output_dir : path
        Created if missing. A non-empty directory is refused (``EpisodeError``)
        unless ``overwrite`` is true, which replaces files of the same name.
    preview : bool
        Draft render: size is forced to 480x270 and fps to 12 (a fraction of
        the pixels and half the frames of a 24 fps 1080p episode, so a layout
        check takes seconds). Fonts, timing and audio are unchanged.
    stills : sequence of float, optional
        Times (s) of ``stills/*.png``. ``None`` (default) picks the intro and
        outro middles, each chapter start + 1 s and each quote middle; ``()``
        writes none. Times are clamped inside the timeline.
    master : bool
        Also write the lossless RGBA16 master with ``moviepy.ae.write_master``
        into ``master/``. It needs an exact-length PCM WAV, so the mixed audio
        is exported to ``master_audio.wav`` (48 kHz stereo); if the frame rate
        makes an integral sample count impossible, the master is written
        silent and ``notes`` says so.
    overwrite : bool
        Allow writing into a non-empty ``output_dir``.
    workers : int, optional
        Frame-rendering processes (``None``: ``min(cpu_count - 1, 12)``;
        ``1``: render in this process). Frames are identical for any value.
    encoder : str
        ``"auto"`` (NVENC if a real test encode works, else libx264) or one of
        ``libx264``, ``h264_nvenc``, ``hevc_nvenc``, ``libx265``. See
        ``moviepy.ae.parallel.detect_encoder``.

    Returns
    -------
    dict
        ``video`` (``episode.mp4``: the chosen encoder, aac, yuv420p,
        ``comp.fps``), ``encoder``, ``workers`` and ``render_fps`` (frames
        per second of wall time of the video step),
        ``report`` (``report.json``), ``stills`` (list of paths), ``master``
        (directory or ``None``), ``notes``, ``preview`` and ``timings``
        (seconds for ``build``, ``stills``, ``video``, ``master``, ``total``).
        Every opened clip is closed, also on failure.
    """
    import dataclasses
    import shutil
    import time

    if isinstance(spec_or_path, EpisodeSpec):
        spec = spec_or_path
    elif isinstance(spec_or_path, dict):
        spec = EpisodeSpec.from_dict(spec_or_path)
    else:
        spec = EpisodeSpec.from_json(Path(spec_or_path))
    out = Path(output_dir)
    if out.exists() and not out.is_dir():
        raise EpisodeError(f"output path is not a directory: {out}")
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise EpisodeError(f"output directory is not empty: {out} (use overwrite)")
    if preview:
        spec = dataclasses.replace(
            spec,
            preset_overrides={
                **spec.preset_overrides,
                "size": list(PREVIEW_SIZE),
                "fps": PREVIEW_FPS,
            },
        )
    out.mkdir(parents=True, exist_ok=True)
    timings, notes = {}, []
    began = time.perf_counter()
    comp = None
    try:
        mark = time.perf_counter()
        comp = build_episode(spec)
        timings["build"] = time.perf_counter() - mark
        report = episode_report(comp)
        report_path = out / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        mark = time.perf_counter()
        if stills is None:
            wanted = _default_stills(comp.episode, comp.duration)
        else:
            wanted = [(None, _num(t, "still time", low=0)) for t in stills]
        last = max(comp.duration - 1.0 / comp.fps, 0.0)
        still_paths = []
        if wanted:
            (out / "stills").mkdir(exist_ok=True)
        for label, t in wanted:
            t = min(t, last)
            name = f"{label or 't'}_{int(round(t * 1000)):07d}ms.png"
            comp.save_frame(str(out / "stills" / name), t=t)
            still_paths.append(str(out / "stills" / name))
        timings["stills"] = time.perf_counter() - mark

        mark = time.perf_counter()
        from moviepy.ae.parallel import write_video_parallel

        video = out / "episode.mp4"
        frames = int(comp.duration * comp.fps)
        audio_tmp = None
        if comp.audio is not None:
            audio_tmp = out / "episode_audio.tmp.wav"
            if not _export_wav(comp.audio, audio_tmp, frames, comp.fps):
                comp.audio.write_audiofile(
                    str(audio_tmp), fps=_MASTER_RATE, logger=None
                )
        try:
            written = write_video_parallel(
                _episode_factory,
                (spec.to_dict(),),
                video,
                fps=comp.fps,
                n_frames=frames,
                size=comp.size,
                audio_path=audio_tmp,
                workers=workers,
                encoder=encoder,
                clip=comp,
            )
        finally:
            if audio_tmp is not None and audio_tmp.exists():
                audio_tmp.unlink()
        timings["video"] = time.perf_counter() - mark

        master_dir = None
        if master:
            from fractions import Fraction

            from moviepy.ae.master import write_master

            mark = time.perf_counter()
            audio = comp.audio
            wav = None
            if audio is None:
                notes.append("master: composition has no audio; master is silent")
            else:
                fps = Fraction(str(comp.fps)).limit_denominator(1000000)
                count = math.ceil(math.nextafter(comp.duration * float(fps), -math.inf))
                wav = out / "master_audio.wav"
                if not _export_wav(audio, wav, count, comp.fps):
                    wav = None
                    notes.append(
                        f"master: {comp.fps:g} fps gives no whole sample count at "
                        f"{_MASTER_RATE} Hz; master written without audio"
                    )
            master_dir = out / "master"
            if master_dir.exists():
                shutil.rmtree(master_dir)  # only reached with overwrite
            write_master(comp, master_dir, audio_path=wav)
            timings["master"] = time.perf_counter() - mark
    finally:
        for clip in getattr(comp, "episode_clips", ()):
            try:
                clip.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
    timings["total"] = time.perf_counter() - began
    return {
        "video": str(video),
        "report": str(report_path),
        "stills": still_paths,
        "master": None if master_dir is None else str(master_dir),
        "notes": notes,
        "preview": bool(preview),
        "timings": timings,
        "encoder": written["encoder"],
        "workers": written["workers"],
        "render_fps": written["fps_achieved"],
    }


# --------------------------------------------------------------------------- #
# project scaffolding and command line
# --------------------------------------------------------------------------- #

CHANNELS = ("story", "history")

_README = """\
夜燈 episode 專案（頻道：{channel}）

製作清單
[ ] 1. 把圖片放入 media/，背景圖亦然（取代 PLACEHOLDER_*.jpg）。
[ ] 2. 把旁白放入 media/（取代 PLACEHOLDER_narration.wav）；
       音樂（選用）放入 media/，並在 episode.json 的 audio.music 填路徑。
[ ] 3. 把字幕 SRT 放入 subtitles/（取代 PLACEHOLDER_*.srt）。
[ ] 4. 編輯 episode.json：標題、鏡頭秒數、章節、引文、audio。
       所有相對路徑皆相對於 episode.json 所在資料夾。
[ ] 5. 檢查：  python -m moviepy.ae.templates validate episode.json
[ ] 6. 預覽：  python -m moviepy.ae.templates render episode.json out --preview
[ ] 7. 正式輸出：python -m moviepy.ae.templates render episode.json final
       （加 --master 另存無損母帶）
注意：旁白若長於時間軸會報錯；請加長鏡頭，或於 audio 設 "trim_audio": true。
"""


def init_episode(directory, channel):
    """Create an episode project folder from the ``channel`` config.

    ``directory`` must not exist or be empty. It receives ``episode.json`` (the
    shipped config with PLACEHOLDER paths), empty ``media/`` and ``subtitles/``
    folders and a Traditional Chinese ``README.txt`` checklist. Returns the
    path of ``episode.json``.

    Examples
    --------
    >>> init_episode("x", "tv")
    Traceback (most recent call last):
    ...
    moviepy.ae.templates.episode.EpisodeError: channel must be one of ['story', 'history'], got 'tv'
    """
    _choice(channel, "channel", CHANNELS)
    target = Path(directory)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise EpisodeError(f"directory exists and is not empty: {target}")
    source = Path(__file__).parent / "configs" / f"nightlamp_{channel}.json"
    target.mkdir(parents=True, exist_ok=True)
    (target / "media").mkdir()
    (target / "subtitles").mkdir()
    config = target / "episode.json"
    config.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (target / "README.txt").write_text(
        _README.format(channel=channel), encoding="utf-8"
    )
    return str(config)


def main(argv=None):
    """Command line entry: ``init``, ``validate`` and ``render``.

    Returns the process exit status (0 success, 1 on any error), so it can be
    called from tests; ``python -m moviepy.ae.templates`` exits with it.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m moviepy.ae.templates", description="Episode project tools."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="create a project folder")
    p_init.add_argument("directory")
    p_init.add_argument("--channel", choices=CHANNELS, required=True)
    p_val = sub.add_parser("validate", help="load and build the timeline only")
    p_val.add_argument("episode")
    p_ren = sub.add_parser("render", help="render episode.mp4, report and stills")
    p_ren.add_argument("episode")
    p_ren.add_argument("output")
    p_ren.add_argument("--preview", action="store_true")
    p_ren.add_argument("--master", action="store_true")
    p_ren.add_argument("--overwrite", action="store_true")
    p_ren.add_argument("--workers", type=int, default=None, help="render processes")
    p_ren.add_argument(
        "--encoder",
        choices=("auto", "libx264", "h264_nvenc", "hevc_nvenc", "libx265"),
        default="auto",
    )
    p_ren.add_argument("--still", type=float, nargs="+", action="extend", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(f"created {init_episode(args.directory, args.channel)}")
        elif args.command == "validate":
            spec = EpisodeSpec.from_json(Path(args.episode))
            comp = build_episode(spec)
            try:
                report = episode_report(comp)
            finally:
                for clip in comp.episode_clips:
                    clip.close()
            print(
                f"OK {report['name']}: {report['size'][0]}x{report['size'][1]} "
                f"@ {report['fps']:g} fps, {report['duration']:.2f} s, "
                f"{len(report['timeline'])} segments, "
                f"{len(report['chapters'])} chapters, {len(report['quotes'])} quotes"
                + (", audio" if "audio" in report else ", no audio")
            )
        else:
            result = render_episode(
                args.episode,
                args.output,
                preview=args.preview,
                stills=args.still,
                master=args.master,
                overwrite=args.overwrite,
                workers=args.workers,
                encoder=args.encoder,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (EpisodeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
