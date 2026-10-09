r"""Track title cards, study progress ring and session timeline as ASS graphics.

Everything here is plain ASS (libass) text, so nothing costs anything per
frame in Python: the 87 minute study video burns the layers with the FFmpeg
``ass`` filter exactly like the S14 countdown.  Layouts are in the 1280x720
PlayRes of ``study.StudySchedule.to_ass``.

Layout (all rectangles are ``(x, y, width, height)`` in PlayRes pixels)

- title panel ``(40, 38, 700, 148)``, timer panel ``(900, 38, 338, 247)``
  (clock text at ``(918, 179)``) and tip panel ``(40, 244, 838, 164)`` come
  from the S14 layout; the spectrum is ``(96, 548, 1088, 140)``.
- progress ring: centre ``(1186, 234)``, outer radius 37.5, right of the
  ``MM:SS`` clock and inside the timer panel.
- timeline bar ``(600, 524, 584, 6)`` and its "next" label
  ``(664, 496, 520, 22)`` sit in the gap between tip panel and spectrum.
- track card: lower left, bottom edge at y 540, left of the timeline.

Examples
--------
>>> doc = AssDocument()
>>> doc.add_style("Vec", "Arial", 20, outline=0)
>>> doc.add_event(1, 0, 1.5, "Vec", "{\\pos(0,0)}")
>>> doc.to_text().splitlines()[-1]
'Dialogue: 1,0:00:00.00,0:00:01.50,Vec,,0,0,0,,{\\pos(0,0)}'
>>> arc_drawing((100, 100), 40, 6, 0.0)
''
"""

import math

from moviepy.ae.templates.study import StudySchedule, ass_text, ass_time


__all__ = [
    "AssDocument",
    "SPECTRUM_RECT",
    "STUDY_RECTS",
    "RING_CENTER",
    "TIMELINE_RECT",
    "arc_beziers",
    "arc_drawing",
    "track_cards",
    "progress_ring",
    "timeline_segments",
    "session_timeline",
    "combine_study_ass",
    "sleep_cards_ass",
]

SPECTRUM_RECT = (96, 548, 1088, 140)
STUDY_RECTS = (
    (40, 38, 700, 148),  # title panel
    (900, 38, 338, 247),  # timer panel
    (40, 244, 838, 164),  # tip panel
)
RING_CENTER = (1186, 234)
TIMELINE_RECT = (600, 524, 584, 6)
_LABEL_WIDTH = 520
_LABEL_HEIGHT = 22
_PHASE_COLORS = {
    "focus": (0xA8, 0xD5, 0xC4),
    "break": (0xF4, 0xEB, 0xD1),
    "closing": (0xD9, 0xC8, 0xE8),
}
_STYLE_FORMAT = (
    "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, "
    "Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, "
    "Encoding"
)
_EVENT_FORMAT = (
    "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
)
_DEFAULT_FONT = "Microsoft JhengHei"
_JA_FONT = "Yu Gothic"
_TEXT_COLOR = "&H00F4F6EE"
_EDGE_COLOR = "&H00132619"
_CARD_KEYS = {"fonts", "sizes", "opacity", "pad", "text_color"}


def _bgr(rgb):
    """Return the ASS ``&HBBGGRR&`` colour for an ``(r, g, b)`` tuple."""
    r, g, b = rgb
    return f"&H{b:02X}{g:02X}{r:02X}&"


def _num(value):
    """Format a float compactly for ASS tags."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def _intersects(a, b):
    """Return True when two ``(x, y, w, h)`` rectangles overlap (area > 0)."""
    return (
        a[0] < b[0] + b[2]
        and b[0] < a[0] + a[2]
        and a[1] < b[1] + b[3]
        and b[1] < a[1] + a[3]
    )


def _cs(second):
    return round(second * 100)


# --------------------------------------------------------------------------- #
# ASS document builder
# --------------------------------------------------------------------------- #


class AssDocument:
    r"""Small ASS builder: styles, events, merge with another ASS document.

    Parameters
    ----------
    play_res : tuple of int, optional
        ``(PlayResX, PlayResY)``; merging a document with another PlayRes
        raises ``ValueError``.
    title : str, optional
        ``Title`` of the Script Info section.

    Examples
    --------
    >>> doc = AssDocument()
    >>> doc.add_style("Vec", "Arial", 20, outline=0)
    >>> doc.add_event(0, 1, 2, "Vec", "x")
    >>> other = AssDocument()
    >>> other.add_style("Vec", "Arial", 20, outline=0)
    >>> other.add_event(0, 0, 1, "Vec", "y")
    >>> doc.merge(other.to_text())
    >>> [e.rsplit(",", 1)[-1] for e in doc.to_text().splitlines()[-2:]]
    ['y', 'x']
    """

    def __init__(self, play_res=(1280, 720), *, title="Quiet music overlay"):
        self.play_res = (int(play_res[0]), int(play_res[1]))
        self.title = str(title)
        self._styles = {}
        self._events = []

    # -- construction -------------------------------------------------------- #

    @classmethod
    def from_ass(cls, text):
        """Parse an ASS document (as written by ``StudySchedule.to_ass``)."""
        title, res = "Quiet music overlay", (1280, 720)
        for line in text.splitlines():
            key, _, value = line.partition(":")
            if key == "Title":
                title = value.strip()
        res = _parse_play_res(text) or res
        doc = cls(res, title=title)
        doc.merge(text)
        return doc

    def add_style(
        self,
        name,
        fontname,
        size,
        *,
        primary=_TEXT_COLOR,
        outline_colour=_EDGE_COLOR,
        outline=0.65,
        alignment=7,
    ):
        """Add a style (same field layout as the S14 styles).

        Re-adding an identical style is a no-op; a different definition under
        the same name raises ``ValueError``.
        """
        if "," in name or not name.strip():
            raise ValueError("Invalid style name")
        line = (
            f"Style: {name},{fontname},{size},{primary},{primary},"
            f"{outline_colour},{outline_colour},0,0,0,0,100,100,0,0,1,"
            f"{outline},0,{alignment},0,0,0,1"
        )
        self._set_style(name, line)

    def _set_style(self, name, line):
        old = self._styles.get(name)
        if old is not None and old != line:
            raise ValueError(f"Conflicting definitions of style {name!r}")
        self._styles[name] = line

    def add_event(self, layer, start, end, style, text):
        """Add a ``Dialogue`` event; ``text`` is used verbatim (escape it first)."""
        if end <= start:
            raise ValueError("Event must end after it starts")
        if style not in self._styles:
            raise ValueError(f"Unknown style {style!r}")
        line = (
            f"Dialogue: {int(layer)},{ass_time(start)},{ass_time(end)},"
            f"{style},,0,0,0,,{text}"
        )
        self._events.append((_cs(start), len(self._events), line))

    def merge(self, other_ass_text):
        """Merge another ASS document (styles and events) into this one.

        Styles are unioned by name (conflicting definitions raise
        ``ValueError``); every event line is kept byte for byte.  Events are
        re-sorted by start time, stable for equal starts.
        """
        other_res = _parse_play_res(other_ass_text)
        if other_res is not None and other_res != self.play_res:
            raise ValueError("PlayRes differs between merged documents")
        section = None
        for raw in other_ass_text.splitlines():
            line = raw.rstrip("\r")
            if line.startswith("[") and line.rstrip().endswith("]"):
                section = line.strip().lower()
                continue
            if section == "[script info]":
                continue
            elif section == "[v4+ styles]" and line.startswith("Style:"):
                name = line[6:].split(",", 1)[0].strip()
                self._set_style(name, line)
            elif section == "[events]" and line.startswith(("Dialogue:", "Comment:")):
                fields = line.split(",", 9)
                if len(fields) < 10:
                    raise ValueError(f"Malformed event line: {line[:60]}")
                start = _parse_ass_time(fields[1])
                self._events.append((start, len(self._events), line))
        self._events.sort(key=lambda item: (item[0], item[1]))
        self._events = [(s, i, line) for i, (s, _, line) in enumerate(self._events)]

    # -- queries ------------------------------------------------------------- #

    @property
    def style_names(self):
        """Return the style names in insertion order."""
        return list(self._styles)

    @property
    def events(self):
        """Return the event lines sorted by start time."""
        return [line for _, _, line in sorted(self._events)]

    def to_text(self):
        """Return the ASS document."""
        head = (
            "[Script Info]\n"
            f"Title: {ass_text(self.title)}\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {self.play_res[0]}\n"
            f"PlayResY: {self.play_res[1]}\n"
            "WrapStyle: 2\n"
            "ScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\n"
            f"Format: {_STYLE_FORMAT}\n"
        )
        body = "\n".join(self._styles.values())
        events = "\n".join(self.events)
        return (
            f"{head}{body}\n\n[Events]\nFormat: {_EVENT_FORMAT}\n"
            f"{events}{chr(10) if events else ''}"
        )


def _parse_play_res(text):
    x = y = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key == "PlayResX":
            x = int(value)
        elif key == "PlayResY":
            y = int(value)
    return (x, y) if x and y else None


def _parse_ass_time(text):
    hours, minutes, rest = text.strip().split(":")
    seconds, _, fraction = rest.partition(".")
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 100 + int(
        (fraction + "00")[:2]
    )


# --------------------------------------------------------------------------- #
# Vector helpers
# --------------------------------------------------------------------------- #

_DRAW_SCALE = 5  # \p5: coordinates are 1/16 px


def _style_vec(doc):
    doc.add_style("Vec", _DEFAULT_FONT, 20, outline=0)


def arc_beziers(center, radius, start_angle, sweep):
    """Approximate a circular arc with cubic Bezier segments.

    Parameters
    ----------
    center : tuple of float
        ``(cx, cy)`` in pixels (y grows downwards).
    radius : float
        Arc radius in pixels.
    start_angle : float
        Start angle in radians; positive angles run clockwise on screen.
    sweep : float
        Signed sweep in radians; each segment covers at most 90 degrees.

    Returns
    -------
    list of tuple
        ``[(p0, p1, p2, p3), ...]`` with ``p`` as ``(x, y)``.

    Examples
    --------
    >>> len(arc_beziers((0, 0), 10, 0.0, 2 * math.pi))
    4
    >>> arc_beziers((0, 0), 10, 0.0, 0.0)
    []
    """
    if sweep == 0:
        return []
    count = max(1, math.ceil(abs(sweep) / (math.pi / 2) - 1e-9))
    step = sweep / count
    k = 4.0 / 3.0 * math.tan(step / 4.0)
    cx, cy = center
    segments = []
    for i in range(count):
        a0 = start_angle + i * step
        a1 = a0 + step
        p0 = (cx + radius * math.cos(a0), cy + radius * math.sin(a0))
        p3 = (cx + radius * math.cos(a1), cy + radius * math.sin(a1))
        p1 = (p0[0] - k * radius * math.sin(a0), p0[1] + k * radius * math.cos(a0))
        p2 = (p3[0] + k * radius * math.sin(a1), p3[1] - k * radius * math.cos(a1))
        segments.append((p0, p1, p2, p3))
    return segments


def _pt(point):
    scale = 1 << (_DRAW_SCALE - 1)
    return f"{round(point[0] * scale)} {round(point[1] * scale)}"


def _contour(segments):
    """Return ``m ... b ...`` for a chain of Bezier segments (no close)."""
    return f"m {_pt(segments[0][0])} b {_chain(segments)}"


def _chain(segments):
    return " ".join(f"{_pt(p1)} {_pt(p2)} {_pt(p3)}" for _, p1, p2, p3 in segments)


def _reverse(segments):
    return [(p3, p2, p1, p0) for p0, p1, p2, p3 in reversed(segments)]


def arc_drawing(center, radius, width, fraction, start_angle=-math.pi / 2):
    """Return the ASS drawing (scale ``p5``) of a clockwise ring sector.

    Parameters
    ----------
    center : tuple of float
        Ring centre ``(cx, cy)``.
    radius : float
        Radius of the ring's centre line.
    width : float
        Stroke width in pixels.
    fraction : float
        Swept fraction of the full circle; ``<= 0`` returns ``""`` and
        ``>= 1`` a complete annulus (outer and inner contour wound oppositely
        so the middle stays open).
    start_angle : float, optional
        Start angle in radians; the default is twelve o'clock.

    Returns
    -------
    str
        Drawing commands with absolute coordinates; use with ``pos(0,0)``.

    Examples
    --------
    >>> arc_drawing((100, 100), 40, 6, 0.0)
    ''
    >>> arc_drawing((100, 100), 40, 6, 1.0).count("m ")
    2
    """
    if width <= 0 or radius <= width / 2:
        raise ValueError("Need 0 < width < 2 * radius")
    if fraction <= 0:
        return ""
    outer_r, inner_r = radius + width / 2, radius - width / 2
    if fraction >= 1:
        outer = arc_beziers(center, outer_r, start_angle, 2 * math.pi)
        inner = arc_beziers(center, inner_r, start_angle, 2 * math.pi)
        return _contour(outer) + " " + _contour(_reverse(inner))
    sweep = fraction * 2 * math.pi
    outer = arc_beziers(center, outer_r, start_angle, sweep)
    inner = _reverse(arc_beziers(center, inner_r, start_angle, sweep))
    return f"{_contour(outer)} l {_pt(inner[0][0])} b {_chain(inner)}"


def _vec_body(drawing, color, alpha, extra=""):
    return (
        f"{{\\pos(0,0)\\an7\\p{_DRAW_SCALE}\\1c{_bgr(color)}\\1a&H{alpha:02X}&"
        f"{extra}}}{drawing}{{\\p0}}"
    )


def _rect_drawing(rects):
    scale = 1 << (_DRAW_SCALE - 1)
    parts = []
    for x, y, w, h in rects:
        x0, y0, x1, y1 = (round(v * scale) for v in (x, y, x + w, y + h))
        parts.append(f"m {x0} {y0} l {x1} {y0} {x1} {y1} {x0} {y1}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Track title cards (#6)
# --------------------------------------------------------------------------- #


def _text_width(text, size):
    """Conservative width estimate: wide glyphs 1.0 em, others 0.58 em."""
    return sum(size * (1.0 if ord(c) >= 0x2E80 else 0.58) for c in text)


def _cue_list(cues):
    """Return ``(chapters, total_seconds or None)`` from a cue sheet or list."""
    if isinstance(cues, dict):
        chapters, total = cues.get("chapters"), cues.get("total_seconds")
    else:
        chapters, total = cues, None
    if not chapters:
        raise ValueError("cues needs at least one chapter")
    return list(chapters), total


def _chapter_start(chapter):
    for key in ("crossfade_center_seconds", "global_start_seconds", "start"):
        if key in chapter:
            return float(chapter[key])
    raise ValueError("Chapter has no start time")


def track_cards(
    cues,
    *,
    titles,
    position="lower_left",
    hold=8.0,
    fade_in=1.5,
    fade_out=2.0,
    offset=3.0,
    style=None,
    max_width=540,
    reserved=None,
    intro_seconds=0.0,
    skip_first=False,
    play_res=(1280, 720),
):
    """Build slow-fading trilingual title cards at every chapter start.

    Parameters
    ----------
    cues : dict or sequence of dict
        ``music_audio.assemble_chapters`` cue sheet (or just its ``chapters``
        list).  A chapter starts at ``crossfade_center_seconds`` (the
        ffmetadata chapter mark), else ``global_start_seconds`` / ``start``.
    titles : mapping
        ``{chapter id: {"zh": ..., "en": ..., "ja": ...}}`` (at least one
        language per chapter; missing ids raise ``ValueError``).
    position : str, optional
        ``lower_left`` (default, between tip panel and spectrum),
        ``lower_right``, ``upper_left`` or ``upper_right``.
    hold : float, optional
        Seconds fully visible; the event lasts ``fade_in + hold + fade_out``
        (clipped at the next card and at ``total_seconds``).
    fade_in, fade_out : float, optional
        Fade lengths in seconds (``fad``).
    offset : float, optional
        Delay after the chapter start.
    style : dict, optional
        Overrides: ``fonts`` ({"zh","en","ja"} font names), ``sizes``
        (three ints), ``opacity`` (backplate, 0.35), ``pad`` (px, 14),
        ``text_color`` (ASS colour).
    max_width : int, optional
        Widest allowed card (``ValueError`` when a title is wider).
    reserved : sequence of rectangles, optional
        Forbidden ``(x, y, w, h)`` regions; default is the spectrum only.
    intro_seconds : float, optional
        Cards that would start before this time are skipped (intro guard).
    skip_first : bool, optional
        Always skip the first chapter.
    play_res : tuple of int, optional
        PlayRes of the document.

    Returns
    -------
    AssDocument
        Backplate (layer 0) and text (layer 2) events per card.

    Raises
    ------
    ValueError
        On a card overlapping a reserved rectangle or leaving the frame.

    Examples
    --------
    >>> cues = {"chapters": [{"id": "A", "crossfade_center_seconds": 0.0}]}
    >>> doc = track_cards(cues, titles={"A": {"en": "Rain"}}, intro_seconds=0)
    >>> doc.events[0].startswith("Dialogue: 0,0:00:03.00,0:00:14.50,CardPanel")
    True
    """
    if position not in ("lower_left", "lower_right", "upper_left", "upper_right"):
        raise ValueError(f"Unknown card position {position!r}")
    if min(hold, fade_in, fade_out, offset) < 0 or hold + fade_in + fade_out <= 0:
        raise ValueError("hold, fades and offset must be non-negative")
    options = dict(style or {})
    if set(options) - _CARD_KEYS:
        raise ValueError(f"Unknown card style keys {sorted(set(options) - _CARD_KEYS)}")
    fonts = {"zh": _DEFAULT_FONT, "en": _DEFAULT_FONT, "ja": _JA_FONT}
    fonts.update(options.get("fonts", {}))
    sizes = dict(zip(("zh", "en", "ja"), options.get("sizes", (30, 24, 24))))
    opacity = options.get("opacity", 0.35)
    pad = options.get("pad", 14)
    color = options.get("text_color", _TEXT_COLOR)
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be between zero and one")
    rects = [SPECTRUM_RECT] if reserved is None else [tuple(r) for r in reserved]
    chapters, total = _cue_list(cues)
    doc = AssDocument(play_res, title="Track cards")
    _style_vec(doc)
    doc.add_style("CardPanel", _DEFAULT_FONT, 20, outline=0)
    for lang, tag in (("zh", "CardZH"), ("en", "CardEN"), ("ja", "CardJA")):
        doc.add_style(tag, fonts[lang], sizes[lang], primary=color)
    starts = [_chapter_start(c) + offset for c in chapters]
    alpha = round((1 - opacity) * 255)
    for index, chapter in enumerate(chapters):
        if index == 0 and (skip_first or starts[0] < intro_seconds):
            continue
        cid = chapter.get("id")
        try:
            names = titles[cid]
        except (KeyError, TypeError):
            raise ValueError(f"No title for chapter {cid!r}") from None
        lines = [
            (lang, str(names[lang])) for lang in ("zh", "en", "ja") if names.get(lang)
        ]
        if not lines:
            raise ValueError(f"Chapter {cid!r} has no title text")
        heights = [round(sizes[lang] * 1.2) for lang, _ in lines]
        width = round(max(_text_width(t, sizes[lang]) for lang, t in lines) + 2 * pad)
        height = sum(heights) + 2 * pad
        if width > max_width:
            raise ValueError(f"Card for {cid!r} is {width}px wide (> {max_width})")
        x = 40 if position.endswith("left") else play_res[0] - 40 - width
        y = 38 if position.startswith("upper") else 540 - height
        rect = (x, y, width, height)
        if x < 0 or y < 0 or x + width > play_res[0] or y + height > play_res[1]:
            raise ValueError("Card leaves the frame")
        for other in rects:
            if _intersects(rect, other):
                raise ValueError(f"Card {cid!r} at {rect} overlaps reserved {other}")
        begin = starts[index]
        end = begin + fade_in + hold + fade_out
        if index + 1 < len(chapters):
            end = min(end, starts[index + 1])
        if total is not None:
            end = min(end, float(total))
        if end <= begin:
            continue
        fad = f"\\fad({round(fade_in * 1000)},{round(fade_out * 1000)})"
        doc.add_event(
            0,
            begin,
            end,
            "CardPanel",
            _vec_body(_rect_drawing([rect]), (0x19, 0x24, 0x17), alpha, fad),
        )
        ty = y + pad
        for (lang, text), line_h in zip(lines, heights):
            doc.add_event(
                2,
                begin,
                end,
                f"Card{lang.upper()}",
                f"{{\\pos({x + pad},{ty}){fad}}}" + ass_text(text),
            )
            ty += line_h
    return doc


# --------------------------------------------------------------------------- #
# Progress ring (#7)
# --------------------------------------------------------------------------- #


def _colors(colors):
    palette = dict(_PHASE_COLORS)
    palette.update(colors or {})
    return palette


def _check_timer_clearance(schedule, ring_box):
    """Raise when the ring box would touch the panel edge or the clock text."""
    timer = STUDY_RECTS[1]
    if not (
        timer[0] <= ring_box[0]
        and timer[1] <= ring_box[1]
        and ring_box[0] + ring_box[2] <= timer[0] + timer[2]
        and ring_box[1] + ring_box[3] <= timer[1] + timer[3]
    ):
        raise ValueError("Progress ring must lie inside the timer panel")
    size = schedule.display["font_sizes"]["timer"]
    clock = (918, 179, round(5 * 0.55 * size), round(size * 1.2))
    if _intersects(ring_box, clock):
        raise ValueError(f"Progress ring overlaps the MM:SS clock {clock}")


def progress_ring(
    schedule,
    *,
    center=RING_CENTER,
    radius=34,
    width=7,
    colors=None,
    step=1,
    track_alpha=0xC8,
    arc_alpha=0x30,
):
    """Draw the elapsed fraction of the current phase as a ring (one event/s).

    Parameters
    ----------
    schedule : StudySchedule
        The validated schedule.
    center : tuple of float, optional
        Ring centre; the default ``(1186, 234)`` sits right of the clock text
        (``pos(918,179)``, Consolas 68 px, about 187 px wide) inside the S14
        timer panel ``(900, 38, 338, 247)`` and below the phase labels.
    radius : float, optional
        Radius of the stroke's centre line (default 34).
    width : float, optional
        Stroke width in pixels (default 7).
    colors : dict, optional
        ``{phase type: (r, g, b)}`` overrides (sage for focus, cream for
        breaks, lavender for closing).
    step : int, optional
        Seconds per event (1 matches the clock).
    track_alpha, arc_alpha : int, optional
        ASS alpha (0 opaque .. 255 clear) of the faint full ring and the arc.

    Returns
    -------
    AssDocument
        Per phase one faint full ring, plus one arc event per ``step``
        seconds whose sweep is the fraction elapsed at the start of the
        interval (so it matches the ``MM:SS`` countdown, 0 at the phase start).

    Raises
    ------
    ValueError
        If the ring leaves the timer panel or overlaps the clock text.

    Examples
    --------
    >>> from moviepy.ae.templates.study import StudySchedule
    >>> def ph(i, kind, a, b):
    ...     return {"id": i, "type": kind, "start": a, "end": b, "label": "L",
    ...             "label_en": "L", "label_ja": "L", "tip": "T",
    ...             "tip_en": "T", "tip_ja": "T"}
    >>> schedule = StudySchedule.from_dict({
    ...     "episode": "D", "duration_seconds": 4, "focus_seconds": 4,
    ...     "rest_seconds": 0,
    ...     "display": {"timer_panel_opacity": 0.3, "title_panel_opacity": 0.3,
    ...                 "tip_panel_opacity": 0.3,
    ...                 "font_sizes": {"timer": 68}},
    ...     "phases": [ph("F", "focus", 0, 4)], "study_chapters": []})
    >>> len(progress_ring(schedule).events)  # track + 3 arcs (second 0 is empty)
    4
    """
    if not isinstance(schedule, StudySchedule):
        raise TypeError("schedule must be a StudySchedule")
    if step < 1 or int(step) != step:
        raise ValueError("step must be a positive whole number of seconds")
    cx, cy = center
    outer = radius + width / 2
    side = math.ceil(2 * outer) + 1
    box = (math.floor(cx - outer), math.floor(cy - outer), side, side)
    _check_timer_clearance(schedule, box)
    palette = _colors(colors)
    doc = AssDocument(title="Study progress ring")
    _style_vec(doc)
    for phase in schedule.phases:
        color = palette[phase["type"]]
        start, end = phase["start"], phase["end"]
        doc.add_event(
            1,
            start,
            end,
            "Vec",
            _vec_body(arc_drawing(center, radius, width, 1.0), color, track_alpha),
        )
        length = end - start
        for second in range(start, end, int(step)):
            fraction = (second - start) / length
            drawing = arc_drawing(center, radius, width, fraction)
            if drawing:
                doc.add_event(
                    1,
                    second,
                    min(end, second + int(step)),
                    "Vec",
                    _vec_body(drawing, color, arc_alpha),
                )
    return doc


# --------------------------------------------------------------------------- #
# Session timeline (#7)
# --------------------------------------------------------------------------- #


def timeline_segments(schedule, rect=TIMELINE_RECT, gap=1.0):
    """Return the bar segments ``(phase id, type, x0, x1)`` for ``rect``.

    Parameters
    ----------
    schedule : StudySchedule
        The validated schedule.
    rect : tuple, optional
        Bar rectangle ``(x, y, width, height)``.
    gap : float, optional
        Pixels removed from the right edge of every segment but the last.

    Examples
    --------
    >>> timeline_segments.__name__
    'timeline_segments'
    """
    x, _, width, _ = rect
    total = schedule.duration
    out = []
    for i, phase in enumerate(schedule.phases):
        x0 = x + width * phase["start"] / total
        x1 = x + width * phase["end"] / total
        if i + 1 < len(schedule.phases):
            x1 -= gap
        out.append((phase["id"], phase["type"], x0, x1))
    return out


def session_timeline(
    schedule,
    *,
    rect=TIMELINE_RECT,
    colors=None,
    playhead_step=10,
    next_seconds=60,
    label_prefix=("接下來：", "next: ", "次は："),
    bar_alpha=0x70,
    gap=1.0,
):
    """Draw a thin phase bar, a moving playhead and calm "next" labels.

    Parameters
    ----------
    schedule : StudySchedule
        The validated schedule.
    rect : tuple, optional
        Bar rectangle; default ``(600, 524, 584, 6)`` (gap between the tip
        panel and the spectrum, right of the track card).
    colors : dict, optional
        ``{phase type: (r, g, b)}`` overrides.
    playhead_step : int, optional
        Seconds per playhead event; each event glides linearly (``move``)
        to the position of the next one, so 10 s is already smooth.
    next_seconds : int, optional
        The "next: ..." label is shown during the last ``next_seconds`` of
        every phase except the last (``0`` disables it), fading 1.5 s in and out (no countdown).
    label_prefix : tuple of str, optional
        Prefixes of the zh / en / ja parts of the label.
    bar_alpha : int, optional
        ASS alpha of the segments.
    gap : float, optional
        Pixel gap between segments.

    Returns
    -------
    AssDocument
        One event per phase type for the segments (spanning the whole
        episode), one per ``playhead_step`` for the playhead and one label per
        phase.

    Examples
    --------
    >>> session_timeline.__name__
    'session_timeline'
    """
    if not isinstance(schedule, StudySchedule):
        raise TypeError("schedule must be a StudySchedule")
    if playhead_step < 1 or int(playhead_step) != playhead_step:
        raise ValueError("playhead_step must be a positive whole number")
    x, y, width, height = rect
    palette = _colors(colors)
    doc = AssDocument(title="Study session timeline")
    _style_vec(doc)
    doc.add_style("NextLabel", _DEFAULT_FONT, 18, outline=0.5, alignment=9)
    segments = timeline_segments(schedule, rect, gap)
    by_type = {}
    for _, kind, x0, x1 in segments:
        by_type.setdefault(kind, []).append((x0, y, x1 - x0, height))
    for kind in sorted(by_type):
        doc.add_event(
            1,
            0,
            schedule.duration,
            "Vec",
            _vec_body(_rect_drawing(by_type[kind]), palette[kind], bar_alpha),
        )
    # Playhead: a 3 px wide tick, 4 px taller than the bar on both sides.
    tick = (0, 0, 3, height + 8)
    step = int(playhead_step)
    for second in range(0, schedule.duration, step):
        nxt = min(schedule.duration, second + step)
        x_a = x + width * second / schedule.duration - 1.5
        x_b = x + width * nxt / schedule.duration - 1.5
        move = f"\\move({_num(x_a)},{y - 4},{_num(x_b)},{y - 4})"
        drawing = _rect_drawing([tick])
        doc.add_event(
            2,
            second,
            nxt,
            "Vec",
            f"{{{move}\\an7\\p{_DRAW_SCALE}\\1c{_bgr((0xEE, 0xF6, 0xF4))}\\1a&H30&}}"
            f"{drawing}{{\\p0}}",
        )
    zh, en, ja = label_prefix
    phases = schedule.phases
    for i, phase in enumerate(phases[:-1] if next_seconds > 0 else []):
        following = phases[i + 1]
        begin = max(phase["start"], phase["end"] - next_seconds)
        text = (
            f"{zh}{following['label']}  ·  {en}{following['label_en']}  ·  "
            f"{ja}{following['label_ja']}"
        )
        size = 18
        width_est = _text_width(text, size)
        if width_est > _LABEL_WIDTH:
            size = int(size * _LABEL_WIDTH / width_est)
            if size < 12:
                raise ValueError(f"Next-phase label too wide: {text!r}")
        shrink = f"\\fs{size}" if size != 18 else ""
        doc.add_event(
            2,
            begin,
            phase["end"],
            "NextLabel",
            f"{{\\pos({x + width},{y - 28})\\fad(1500,1500)\\1a&H20&{shrink}}}"
            + ass_text(text),
        )
    return doc


# --------------------------------------------------------------------------- #
# Combination
# --------------------------------------------------------------------------- #


def combine_study_ass(schedule, *, ring=True, timeline=True, cards=None, compact=False):
    """Return the S14 ASS merged with the progress ring, timeline and cards.

    Parameters
    ----------
    schedule : StudySchedule
        The validated schedule.
    ring, timeline : bool, optional
        Add the progress ring / session timeline layers.
    cards : dict, optional
        Keyword arguments of ``track_cards`` (``cues`` and ``titles`` are
        required); the reserved regions are the study panels, the spectrum
        and, when enabled, the timeline bar and its label.
    compact : bool, optional
        Passed to ``StudySchedule.to_ass`` (needs the Windows fonts).

    Returns
    -------
    str
        The combined ASS text; every original S14 event line is unchanged.

    Examples
    --------
    >>> combine_study_ass.__name__
    'combine_study_ass'
    """
    doc = AssDocument.from_ass(schedule.to_ass(compact=compact))
    if ring:
        doc.merge(progress_ring(schedule).to_text())
    if timeline:
        doc.merge(session_timeline(schedule).to_text())
    if cards is not None:
        options = dict(cards)
        reserved = [SPECTRUM_RECT, *STUDY_RECTS]
        if timeline:
            x, y, w, h = TIMELINE_RECT
            reserved += [
                (x, y - 4, w, h + 8),
                (x + w - _LABEL_WIDTH, y - 28, _LABEL_WIDTH, _LABEL_HEIGHT),
            ]
        reserved += [tuple(r) for r in options.pop("reserved", [])]
        doc.merge(track_cards(reserved=reserved, **options).to_text())
    return doc.to_text()


def sleep_cards_ass(cues, titles, **kwargs):
    r"""Return the ASS text of the track cards alone (sleep mode).

    Parameters
    ----------
    cues : dict or sequence of dict
        Cue sheet from ``assemble_chapters`` (or its ``chapters``).
    titles : mapping
        Chapter id to ``{"zh", "en", "ja"}`` titles.
    \*\*kwargs
        Passed to ``track_cards``.

    Examples
    --------
    >>> text = sleep_cards_ass([{"id": "A", "start": 0}], {"A": {"en": "Rain"}})
    >>> "CardEN" in text
    True
    """
    return track_cards(cues, titles=titles, **kwargs).to_text()
