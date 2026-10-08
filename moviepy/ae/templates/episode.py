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


__all__ = [
    "EpisodeError",
    "BackgroundSpec",
    "BookendSpec",
    "ShotSpec",
    "ChapterSpec",
    "QuoteSpec",
    "SubtitleSpec",
    "EpisodeSpec",
    "build_episode",
    "episode_report",
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


@dataclass(frozen=True)
class BookendSpec:
    """Intro or outro card: ``TitleCardSpec`` copy fields plus its transition.

    ``transition`` joins this card to the previous segment (default ``cut``);
    the card's own entrance/fade-out timing is that of ``TitleCardSpec``.
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

    def __post_init__(self):
        _str(self.title, "title")
        _str(self.brand, "brand", empty=True)
        _str(self.subtitle, "subtitle", empty=True)
        _choice(self.layout, "layout", ("right_column", "center"))
        object.__setattr__(
            self, "duration", _num(self.duration, "duration", positive=True)
        )
        object.__setattr__(self, "fade_out", _num(self.fade_out, "fade_out", low=0))
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
    across lines.
    """

    primary: object
    secondary: object = None
    protected: tuple = ()
    primary_lang: str = "zh-TW"
    secondary_lang: str = "en"

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
    shots : sequence of ShotSpec
    chapters, quotes : sequences of ChapterSpec / QuoteSpec
    subtitles : SubtitleSpec or None
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
                    item
                    if isinstance(item, cls)
                    else _from_dict(cls, item, f"{name}[{k}]", renames)
                    for k, item in enumerate(items)
                ),
            )
        if not self.shots:
            raise EpisodeError("an episode needs at least one shot")
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
            "background": None
            if self.background is None
            else _spec_dict(self.background),
            "intro": None if self.intro is None else _spec_dict(self.intro),
            "outro": None if self.outro is None else _spec_dict(self.outro),
            "shots": [_spec_dict(s, _RENAMES) for s in self.shots],
            "chapters": [_spec_dict(c) for c in self.chapters],
            "quotes": [_spec_dict(q) for q in self.quotes],
            "subtitles": None if self.subtitles is None else _spec_dict(self.subtitles),
            "chapter_period": self.chapter_period,
            "dip_color": list(self.dip_color),
            "seed": self.seed,
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
    subs = data.get("subtitles")
    if isinstance(subs, dict):
        fix_in(subs, ("primary", "secondary"))


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


def build_episode(spec):
    """Assemble ``spec`` into a ``Composition`` at the preset size and fps.

    Parameters
    ----------
    spec : EpisodeSpec

    Returns
    -------
    Composition
        Layers (top first): ``Subtitles``, ``Outro``, ``Intro``,
        ``Chapter NN``, ``Quote NN``, ``Shot NN``. ``comp.episode`` holds the
        timeline and provenance used by ``episode_report``; ``comp.preset`` is
        the resolved ``ChannelPreset`` and ``comp.episode_clips`` the opened
        media clips (close them with ``clip.close()`` when finished).

    Raises
    ------
    EpisodeError
        For missing files, transitions that leave the previous focus hold,
        overlays that leave the timeline or overlap a card, overlapping
        quotes and cues outside the timeline.

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
    from moviepy.ae.templates.quote import vertical_quote
    from moviepy.ae.templates.subtitles import subtitle_layer
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
        for lane, cues in cue_sets.items():
            for cue in cues:
                if cue.end > total + _EPS:
                    raise EpisodeError(
                        f"subtitles.{lane} cue [{cue.start:g}, {cue.end:g}) "
                        f"{cue.text[:20]!r} lies outside the {total:g} s timeline"
                    )
    for role in spec.fonts:
        if spec.fonts[role] is not None:
            note(spec.fonts[role], f"fonts.{role}")

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
    subtitle = None
    if cue_sets:
        subs = spec.subtitles
        font = spec.fonts.get("subtitle", MISSING)
        kwargs = {"font": "default"} if font is None else {}
        if isinstance(font, str):
            kwargs = {"font": font}
        subtitle = subtitle_layer(
            cue_sets["primary"],
            preset,
            secondary=cue_sets.get("secondary"),
            protected=subs.protected,
            size=size,
            **kwargs,
        )
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
        ],  # fmt: skip
        "quotes": [
            {
                "name": f"Quote {k:02d}",
                "start": q.start,
                "end": q.start + qc.duration,
                "medium": qc.quote_meta["medium"],
            }
            for k, q, qc in quote_comps
        ],  # fmt: skip
        "subtitles": {
            lane: {
                "cues": len(cues),
                "first": cues[0].start if cues else None,
                "last": cues[-1].end if cues else None,
            }
            for lane, cues in cue_sets.items()
        },
        "sources": dict(sorted(sources.items())),
    }
    return comp


def episode_report(comp):
    """Return a JSON-able QA dictionary for a ``build_episode`` composition.

    Keys: ``name``, ``preset`` (name) and ``preset_resolved`` (full values),
    ``size``, ``fps``, ``duration``, ``frames``, ``timeline`` (segments with
    start/end/transition), ``chapters``, ``quotes``, ``subtitles`` (cue counts
    and span), ``layers`` (top first: name, type, in/out), ``sources`` (path ->
    sha256 of every media, SRT and font file) and ``top_layer``.

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
