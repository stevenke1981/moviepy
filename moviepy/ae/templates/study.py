"""Study-companion schedule and overlay (port of ``study-overlay.py``).

``StudySchedule`` validates the ``STUDY-MODE.json`` document (contiguous
focus / break / closing phases, trilingual labels and tips, panel opacities)
and derives chime times, ffmetadata chapters and the ASS overlay.
``StudyOverlayLayer`` draws the same overlay natively as an AE layer: three
translucent panels, a trilingual title, the phase label, an ``MM:SS``
countdown and tips that fade in over 0.5 s and out over 0.7 s.

Examples
--------
>>> from moviepy.ae.templates.study import StudySchedule
>>> def phase(i, kind, a, b):
...     return {"id": i, "type": kind, "start": a, "end": b,
...             "label": "L", "label_en": "L", "label_ja": "L",
...             "tip": "T", "tip_en": "T", "tip_ja": "T"}
>>> schedule = StudySchedule.from_dict({
...     "episode": "Demo", "duration_seconds": 90, "focus_seconds": 60,
...     "rest_seconds": 30,
...     "display": {"timer_panel_opacity": 0.3, "title_panel_opacity": 0.3,
...                 "tip_panel_opacity": 0.3},
...     "phases": [phase("F1", "focus", 0, 60), phase("B1", "break", 60, 90)],
...     "study_chapters": [{"start": 0, "name": "Focus"}]})
>>> schedule.remaining_at(59.5)[1]
1
>>> schedule.chime_times()
[60]
"""

import copy
import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.base import Layer


__all__ = ["StudySchedule", "StudyOverlayLayer", "clock_text", "ass_time", "ass_text"]

_TYPES = ("focus", "break", "closing")
_LANG_KEYS = ("label", "label_en", "label_ja", "tip", "tip_en", "tip_ja")
_OPACITIES = ("timer_panel_opacity", "title_panel_opacity", "tip_panel_opacity")
_WIN_FONTS = {
    "zh": "C:/Windows/Fonts/msjh.ttc",
    "ja": "C:/Windows/Fonts/YuGothM.ttc",
    "timer": "C:/Windows/Fonts/consola.ttf",
}
_ASS_FONT_FILES = {
    "Microsoft JhengHei": "msjh.ttc",
    "Yu Gothic": "YuGothM.ttc",
    "Consolas": "consola.ttf",
}
_FILL = (0xEE, 0xF6, 0xF4)  # ASS &H00F4F6EE is BGR
_FILL_REST = (0xD1, 0xEB, 0xF4)  # \1c&HF4EBD1& (RGB D1 EB F4 after the BGR swap)
_EDGE = (0x19, 0x26, 0x13)  # &H00132619
_PANEL = (0x19, 0x24, 0x17)  # &H172419


def clock_text(seconds):
    """Return ``MM:SS`` for a whole number of seconds.

    Parameters
    ----------
    seconds : int
        Non-negative whole seconds.

    Examples
    --------
    >>> clock_text(1500)
    '25:00'
    """
    return f"{seconds // 60:02}:{seconds % 60:02}"


def ass_time(second):
    """Format seconds as an ASS ``H:MM:SS.cc`` timestamp.

    Parameters
    ----------
    second : float
        Time in seconds.

    Examples
    --------
    >>> ass_time(61.5)
    '0:01:01.50'
    """
    centiseconds = round(second * 100)
    hours, rest = divmod(centiseconds, 360000)
    minutes, rest = divmod(rest, 6000)
    seconds, fraction = divmod(rest, 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{fraction:02}"


def ass_text(text):
    r"""Neutralise ASS override syntax so text cannot inject tags.

    Parameters
    ----------
    text : str
        Raw text; backslashes and braces are replaced by full-width forms and
        newlines become ``\N``.

    Examples
    --------
    >>> ass_text("a{\\b1}b")
    'a（／b1）b'
    """
    return (
        str(text)
        .replace("\\", "／")
        .replace("{", "（")
        .replace("}", "）")
        .replace("\n", r"\N")
    )


def _require(mapping, key, where):
    try:
        return mapping[key]
    except (KeyError, TypeError):
        raise ValueError(f"Missing required key {key!r} in {where}") from None


def _validate(config):
    if not isinstance(config, dict):
        raise ValueError("Study schedule must be a mapping")
    duration = _require(config, "duration_seconds", "schedule")
    if not isinstance(duration, int) or duration <= 0:
        raise ValueError("Duration must be a positive integer number of seconds")
    previous = 0
    ids = set()
    phases = _require(config, "phases", "schedule")
    for phase in phases:
        pid = _require(phase, "id", "phase")
        start = _require(phase, "start", "phase")
        end = _require(phase, "end", "phase")
        if pid in ids or start != previous:
            raise ValueError("Duplicate phase ID or non-contiguous schedule")
        if not all(isinstance(v, int) for v in (start, end)) or end <= start:
            raise ValueError("Invalid phase interval")
        if _require(phase, "type", "phase") not in _TYPES:
            raise ValueError("Unknown phase type")
        ids.add(pid)
        previous = end
    if previous != duration:
        raise ValueError("Schedule does not cover the whole episode")
    focus = sum(p["end"] - p["start"] for p in phases if p["type"] == "focus")
    if focus != _require(config, "focus_seconds", "schedule"):
        raise ValueError("Focus total does not match configuration")
    if duration - config["focus_seconds"] != _require(
        config, "rest_seconds", "schedule"
    ):
        raise ValueError("Rest total does not match configuration")
    display = _require(config, "display", "schedule")
    for key in _OPACITIES:
        value = _require(display, key, "display")
        if isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError("Panel opacity must be between zero and one")
    for phase in phases:
        if not all(phase.get(key) for key in _LANG_KEYS):
            raise ValueError(
                "All phase labels and tips need Chinese, English and Japanese"
            )


class StudySchedule:
    """Validated study-companion schedule (the ``STUDY-MODE.json`` document).

    Parameters
    ----------
    config : dict
        Parsed document; it is deep-copied and validated like the source
        ``load_config`` (``ValueError`` on any broken rule).

    Examples
    --------
    >>> StudySchedule.from_dict({"duration_seconds": 0, "phases": []})
    Traceback (most recent call last):
    ...
    ValueError: Duration must be a positive integer number of seconds
    """

    def __init__(self, config):
        _validate(config)
        self._config = copy.deepcopy(config)

    # -- construction / export ---------------------------------------------- #

    @classmethod
    def from_dict(cls, config):
        """Build a schedule from a dictionary."""
        return cls(config)

    @classmethod
    def from_json(cls, path):
        """Build a schedule from a UTF-8 JSON file."""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        """Return a deep copy of the validated document."""
        return copy.deepcopy(self._config)

    # -- queries ------------------------------------------------------------- #

    @property
    def duration(self):
        """Return the episode length in whole seconds."""
        return self._config["duration_seconds"]

    @property
    def phases(self):
        """Return the phase dictionaries (treat as read-only)."""
        return self._config["phases"]

    @property
    def display(self):
        """Return the ``display`` section (treat as read-only)."""
        return self._config["display"]

    def phase_at(self, second):
        """Return the phase active at ``second`` (half-open intervals)."""
        if (
            isinstance(second, bool)
            or not isinstance(second, (int, float))
            or not math.isfinite(second)
            or second < 0
            or second >= self.duration
        ):
            raise ValueError("Time is outside the episode")
        return next(p for p in self.phases if p["start"] <= second < p["end"])

    def remaining_at(self, second):
        """Return ``(phase, whole seconds left in the phase)`` at ``second``."""
        phase = self.phase_at(second)
        return phase, math.ceil(phase["end"] - second)

    def chime_times(self):
        """Return the phase boundaries (seconds) where a chime cue plays."""
        return [p["end"] for p in self.phases[:-1]]

    def chapters_ffmetadata(self):
        r"""Return the ffmetadata text for the study chapters.

        Returns
        -------
        str
            ``;FFMETADATA1`` document with millisecond chapters; ``\\``, ``=``,
            ``;`` and ``#`` in titles are escaped.
        """
        config = self._config
        lines = [";FFMETADATA1", f"title={_require(config, 'episode', 'schedule')}"]
        chapters = _require(config, "study_chapters", "schedule")
        for i, chapter in enumerate(chapters):
            start = chapter["start"]
            if i + 1 < len(chapters):
                end = chapters[i + 1]["start"]
            else:
                end = self.duration
            self.phase_at(start)
            title = (
                chapter["name"]
                .replace("\\", "\\\\")
                .replace("=", "\\=")
                .replace(";", "\\;")
                .replace("#", "\\#")
                .replace("\n", " ")
            )
            lines.extend(
                [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={start * 1000}",
                    f"END={end * 1000}",
                    f"title={title}",
                ]
            )
        return "\n".join(lines) + "\n"

    # -- ASS ----------------------------------------------------------------- #

    def to_ass(
        self, start=0, seconds=None, *, compact=False, demo=False, font_paths=None
    ):
        """Return the overlay as ASS text (PlayRes 1280x720).

        Parameters
        ----------
        start : int, optional
            First second of the rendered interval (event times are relative).
        seconds : int, optional
            Interval length; ``None`` runs to the end of the episode.
        compact : bool, optional
            Use the season-15 layout: dark text with a white outline on
            individual translucent backplates instead of three big panels.
            Backplates are sized with Pillow, so the fonts must exist.
        demo : bool, optional
            Add the ``preview.label`` caption panel.
        font_paths : dict, optional
            ``{ASS font name: file}`` overrides for ``compact`` measuring
            (defaults to ``C:/Windows/Fonts``).

        Returns
        -------
        str
            The ASS document.
        """
        text = self._build_ass(start, seconds, demo)
        if not compact:
            return text
        return _compact(text, font_paths or {})

    def _build_ass(self, start, duration, demo):
        config = self._config
        duration = self.duration - start if duration is None else duration
        if (
            not isinstance(start, int)
            or not isinstance(duration, int)
            or start < 0
            or duration <= 0
            or start + duration > self.duration
        ):
            raise ValueError(
                "Render interval must be a whole-second interval within the episode"
            )
        end = start + duration
        header = _ASS_HEADER
        display = config["display"]
        sizes = _require(display, "font_sizes", "display")
        styles = [
            ("Panel", display["zh_font"], 28),
            ("TitleZH", display["zh_font"], sizes["title_zh"]),
            ("TitleEN", display["zh_font"], sizes["title_en"]),
            ("TitleJA", display["ja_font"], sizes["title_ja"]),
            ("LabelZH", display["zh_font"], sizes["phase_zh"]),
            ("LabelEN", display["zh_font"], sizes["phase_en"]),
            ("LabelJA", display["ja_font"], sizes["phase_ja"]),
            ("Clock", display["timer_font"], sizes["timer"]),
            ("TipZH", display["zh_font"], sizes["tip_zh"]),
            ("TipEN", display["zh_font"], sizes["tip_en"]),
            ("TipJA", display["ja_font"], sizes["tip_ja"]),
            ("Demo", display["zh_font"], 20),
        ]
        header = header.replace(
            "__STYLES__",
            "\n".join(
                f"Style: {name},{font},{size},&H00F4F6EE,&H00F4F6EE,&H00132619,"
                f"&H00132619,0,0,0,0,100,100,0,0,1,"
                f'{0 if name == "Panel" else 0.65},0,7,0,0,0,1'
                for name, font, size in styles
            ),
        )
        events = []

        def event(layer, a, b, style, body):
            if b > a:
                events.append(
                    f"Dialogue: {layer},{ass_time(a - start)},{ass_time(b - start)},"
                    f"{style},,0,0,0,,{body}"
                )

        def panel(a, b, x, y, width, height, opacity, fade=""):
            alpha = round((1 - opacity) * 255)
            body = (
                f"{{\\pos({x},{y})\\p1\\1c&H172419&\\1a&H{alpha:02X}&{fade}}}"
                f"m 0 0 l {width} 0 {width} {height} 0 {height}{{\\p0}}"
            )
            event(0, a, b, "Panel", body)

        panel(start, end, 900, 38, 338, 247, display["timer_panel_opacity"])
        panel(start, end, 40, 38, 700, 148, display["title_panel_opacity"])
        for language, style, y in [
            ("zh", "TitleZH", 49),
            ("en", "TitleEN", 97),
            ("ja", "TitleJA", 137),
        ]:
            event(
                2,
                start,
                end,
                style,
                f"{{\\pos(58,{y})}}" + ass_text(config["title"][language]),
            )
        if demo:
            panel(start, end, 40, 474, 550, 35, display["title_panel_opacity"])
            event(
                3,
                start,
                end,
                "Demo",
                r"{\pos(54,478)}" + ass_text(config["preview"]["label"]),
            )
        for phase in config["phases"]:
            a, b = max(start, phase["start"]), min(end, phase["end"])
            color = r"\1c&HF4EBD1&" if phase["type"] != "focus" else ""
            for key, style, y in [
                ("label", "LabelZH", 52),
                ("label_en", "LabelEN", 102),
                ("label_ja", "LabelJA", 140),
            ]:
                event(
                    2, a, b, style, f"{{\\pos(918,{y}){color}}}" + ass_text(phase[key])
                )
            tip_end = min(phase["end"], phase["start"] + display["tip_display_seconds"])
            tip_a, tip_b = max(start, phase["start"]), min(end, tip_end)
            # A short preview may start inside a tip; do not manufacture another fade-in.
            fade_in = 500 if tip_a == phase["start"] else 0
            fade_out = 700 if tip_b == tip_end else 0
            fade = f"\\fad({fade_in},{fade_out})"
            panel(tip_a, tip_b, 40, 244, 838, 164, display["tip_panel_opacity"], fade)
            for key, style, y in [
                ("tip", "TipZH", 254),
                ("tip_en", "TipEN", 313),
                ("tip_ja", "TipJA", 358),
            ]:
                event(
                    2,
                    tip_a,
                    tip_b,
                    style,
                    f"{{\\pos(58,{y}){fade}}}" + ass_text(phase[key]),
                )
        for second in range(start, end):
            _, remaining = self.remaining_at(second)
            event(
                2,
                second,
                second + 1,
                "Clock",
                r"{\pos(918,179)}" + clock_text(remaining),
            )
        return header + "\n".join(events) + "\n"


_ASS_HEADER = """[Script Info]
Title: Quiet study companion
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
__STYLES__

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _compact(text, font_paths):
    """Season-15 layout: dark text, white outline, individual backplates."""
    from PIL import ImageFont

    lines = []
    for line in text.splitlines():
        if line.startswith("Style: ") and not line.startswith("Style: Panel,"):
            fields = line.split(",")
            fields[3:6] = ["&H00000000", "&H00000000", "&H00FFFFFF"]
            fields[15] = "1"
            fields[16] = "2.2"
            fields[17] = "0"
            line = ",".join(fields)
        lines.append(line.replace(r"\1c&HF4EBD1&", ""))
    styles = {}
    result = []
    for line in lines:
        if line.startswith("Style: "):
            fields = line.split(",")
            name, size = fields[1], int(fields[2])
            path = font_paths.get(name)
            if path is None:
                if name not in _ASS_FONT_FILES:
                    raise ValueError(
                        f"No font file known for {name!r}; pass font_paths"
                    )
                path = "C:/Windows/Fonts/" + _ASS_FONT_FILES[name]
            styles[fields[0][7:]] = (ImageFont.truetype(os.fspath(path), size), size)
        if line.startswith("Dialogue: "):
            fields = line.split(",", 9)
            if fields[3] == "Panel":
                continue
            match = re.search(r"\\pos\((\d+),(\d+)\)", fields[9])
            if not match:
                raise ValueError("Every text event needs an explicit position")
            x, y = map(int, match.groups())
            plain = re.sub(r"\{[^}]*\}", "", fields[9])
            font, size = styles[fields[3]]
            width, height = math.ceil(font.getlength(plain)) + 12, size + 4
            if x - 6 + width > 1260 or y + height > 710:
                raise ValueError("Compact text panel would exceed safe area")
            fade = re.search(r"\\fad\(\d+,\d+\)", fields[9])
            fading = fade.group(0) if fade else ""
            panel = fields.copy()
            panel[0], panel[3] = "Dialogue: 0", "Panel"
            panel[9] = (
                f"{{\\pos({x - 6},{y - 1})\\p1\\1c&H172419&\\1a&HB2&{fading}}}"
                f"m 0 0 l {width} 0 {width} {height} 0 {height}{{\\p0}}"
            )
            result.append(",".join(panel))
        result.append(line)
    return "\n".join(result) + "\n"


# --------------------------------------------------------------------------- #
# Native AE layer
# --------------------------------------------------------------------------- #


def _over(canvas, src, alpha, x, y):
    """Composite premultiplied ``src``/``alpha`` onto ``canvas`` at (x, y)."""
    height, width = canvas.shape[:2]
    h, w = alpha.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    dst = canvas[y0:y1, x0:x1]
    keep = 1.0 - alpha[sub][..., None]
    dst[..., :3] = src[sub] + dst[..., :3] * keep
    dst[..., 3:] = alpha[sub][..., None] + dst[..., 3:] * keep


class StudyOverlayLayer(Layer):
    r"""AE-native equivalent of the ASS study overlay, rendered with Pillow.

    The overlay is built for a 1280x720 design and scaled to ``size``.
    Static parts are rasterised once per phase, the countdown once per
    distinct string, and finished frames are cached per
    ``(second, tip-fade step)`` so an 87-minute render mostly reuses
    cached buffers.

    Parameters
    ----------
    schedule : StudySchedule
        The validated schedule.
    size : tuple of int, optional
        Canvas size ``(width, height)``.
    fonts : dict, optional
        ``{"zh": path, "ja": path, "timer": path}``. Defaults to
        ``msjh.ttc`` / ``YuGothM.ttc`` / ``consola.ttf`` in
        ``C:/Windows/Fonts``. A missing file raises ``FileNotFoundError``.
    fade_steps : int, optional
        Quantisation of the tip fade (12 steps is one step per frame at
        24 fps for the 0.5 s fade-in).
    cache_size : int, optional
        Number of finished frames kept (each is about 7 MB at 1280x720).
    name : str, optional
        Layer name.
    \*\*kwargs
        Passed to ``Layer``.

    Examples
    --------
    >>> StudyOverlayLayer.__name__
    'StudyOverlayLayer'
    """

    FADE_IN = 0.5
    FADE_OUT = 0.7

    def __init__(
        self,
        schedule,
        *,
        size=(1280, 720),
        fonts=None,
        fade_steps=12,
        cache_size=8,
        name="Study overlay",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        from PIL import ImageFont

        if not isinstance(schedule, StudySchedule):
            raise TypeError("schedule must be a StudySchedule")
        self.schedule = schedule
        self._size = (int(size[0]), int(size[1]))
        self._sx = self._size[0] / 1280.0
        self._sy = self._size[1] / 720.0
        paths = dict(_WIN_FONTS)
        paths.update(fonts or {})
        for key, path in paths.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"study overlay font {key!r} not found: {path}")
        sizes = schedule.display["font_sizes"]

        def font(kind, key):
            px = max(1, int(round(sizes[key] * self._sy)))
            return ImageFont.truetype(os.fspath(paths[kind]), px)

        self._fonts = {
            "title_zh": font("zh", "title_zh"),
            "title_en": font("zh", "title_en"),
            "title_ja": font("ja", "title_ja"),
            "phase_zh": font("zh", "phase_zh"),
            "phase_en": font("zh", "phase_en"),
            "phase_ja": font("ja", "phase_ja"),
            "timer": font("timer", "timer"),
            "tip_zh": font("zh", "tip_zh"),
            "tip_en": font("zh", "tip_en"),
            "tip_ja": font("ja", "tip_ja"),
        }
        self._stroke = int(round(0.65 * self._sy))
        self._fade_steps = max(1, int(fade_steps))
        self._capacity = max(1, int(cache_size))
        self._frames = OrderedDict()
        self._statics = OrderedDict()
        self._tips = OrderedDict()
        self._clocks = OrderedDict()
        self._empty = Buffer(np.zeros((1, 1, 4), np.float32))
        # Everything lives inside this design-space box (panels' union).
        x0, y0 = int(math.floor(40 * self._sx)), int(math.floor(38 * self._sy))
        x1, y1 = int(math.ceil(1238 * self._sx)), int(math.ceil(408 * self._sy))
        self._box = (x0, y0, min(x1, self._size[0]), min(y1, self._size[1]))
        self.frames_rendered = 0
        self.out_point = float(schedule.duration)

    @property
    def source_size(self):
        """Return the canvas size in pixels."""
        return self._size

    # -- drawing primitives -------------------------------------------------- #

    def _canvas(self):
        x0, y0, x1, y1 = self._box
        return np.zeros((y1 - y0, x1 - x0, 4), np.float32)

    def _panel(self, canvas, x, y, w, h, opacity):
        bx, by = self._box[:2]
        px, py = int(round(x * self._sx)) - bx, int(round(y * self._sy)) - by
        pw, ph = int(round(w * self._sx)), int(round(h * self._sy))
        alpha = np.full((ph, pw), float(opacity), np.float32)
        src = alpha[..., None] * np.asarray(_PANEL, np.float32) / 255.0
        _over(canvas, src, alpha, px, py)

    def _text(self, canvas, x, y, text, font, color):
        from PIL import Image, ImageDraw

        if not text:
            return
        stroke = self._stroke
        pad = stroke + 2
        width = int(math.ceil(font.getlength(text))) + 2 * pad
        ascent, descent = font.getmetrics()
        height = ascent + descent + 2 * pad
        fill_mask = Image.new("L", (width, height), 0)
        edge_mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(fill_mask).text((pad, pad), text, font=font, fill=255)
        ImageDraw.Draw(edge_mask).text(
            (pad, pad), text, font=font, fill=255, stroke_width=stroke
        )
        fa = np.asarray(fill_mask, np.float32) / 255.0
        oa = np.maximum(np.asarray(edge_mask, np.float32) / 255.0, fa)
        under = oa - fa
        fill = np.asarray(color, np.float32) / 255.0
        edge = np.asarray(_EDGE, np.float32) / 255.0
        src = fill * fa[..., None] + edge * under[..., None]
        bx, by = self._box[:2]
        px = int(round(x * self._sx)) - bx - pad
        py = int(round(y * self._sy)) - by - pad
        _over(canvas, src, oa, px, py)

    # -- cached pieces ------------------------------------------------------- #

    def _remember(self, cache, key, make, capacity):
        value = cache.get(key)
        if value is None:
            value = cache[key] = make()
        else:
            cache.move_to_end(key)
        while len(cache) > capacity:
            cache.popitem(last=False)
        return value

    def _static(self, index):
        def make():
            schedule, phase = self.schedule, self.schedule.phases[index]
            display, title = schedule.display, schedule._config["title"]
            canvas = self._canvas()
            self._panel(canvas, 900, 38, 338, 247, display["timer_panel_opacity"])
            self._panel(canvas, 40, 38, 700, 148, display["title_panel_opacity"])
            for key, lang, y in [
                ("title_zh", "zh", 49),
                ("title_en", "en", 97),
                ("title_ja", "ja", 137),
            ]:
                self._text(canvas, 58, y, title[lang], self._fonts[key], _FILL)
            color = _FILL if phase["type"] == "focus" else _FILL_REST
            for key, label, y in [
                ("phase_zh", "label", 52),
                ("phase_en", "label_en", 102),
                ("phase_ja", "label_ja", 140),
            ]:
                self._text(canvas, 918, y, phase[label], self._fonts[key], color)
            return canvas

        return self._remember(self._statics, index, make, 3)

    def _tip(self, index):
        def make():
            phase = self.schedule.phases[index]
            canvas = self._canvas()
            self._panel(
                canvas, 40, 244, 838, 164, self.schedule.display["tip_panel_opacity"]
            )
            for key, label, y in [
                ("tip_zh", "tip", 254),
                ("tip_en", "tip_en", 313),
                ("tip_ja", "tip_ja", 358),
            ]:
                self._text(canvas, 58, y, phase[label], self._fonts[key], _FILL)
            return canvas

        return self._remember(self._tips, index, make, 3)

    def _clock(self, text):
        def make():
            canvas = self._canvas()
            self._text(canvas, 918, 179, text, self._fonts["timer"], _FILL)
            return canvas

        return self._remember(self._clocks, text, make, 128)

    def tip_state(self, t):
        """Return the tip fade step ``0..fade_steps`` shown at time ``t``."""
        phase = self.schedule.phase_at(t)
        tip_end = min(
            phase["end"], phase["start"] + self.schedule.display["tip_display_seconds"]
        )
        if not phase["start"] <= t < tip_end:
            return 0
        factor = min(
            1.0, (t - phase["start"]) / self.FADE_IN, (tip_end - t) / self.FADE_OUT
        )
        return max(0, int(round(factor * self._fade_steps)))

    def cache_key(self, t):
        """Return the ``(second, tip step)`` key a frame at ``t`` is cached under."""
        return (int(math.floor(t)), self.tip_state(t))

    def source_buffer(self, t, context=None):
        """Return the premultiplied overlay for composition time ``t``."""
        if not (0 <= t < self.schedule.duration):
            return self._empty
        second, step = self.cache_key(t)
        key = (second, step)
        buffer = self._frames.get(key)
        if buffer is not None:
            self._frames.move_to_end(key)
            return buffer
        phase, remaining = self.schedule.remaining_at(second)
        index = self.schedule.phases.index(phase)
        canvas = self._static(index).copy()
        if step:
            tip = self._tip(index)
            factor = step / self._fade_steps
            alpha = tip[..., 3] * factor
            _over(canvas, tip[..., :3] * factor, alpha, 0, 0)
        clock = self._clock(clock_text(remaining))
        _over(canvas, clock[..., :3], clock[..., 3], 0, 0)
        canvas.setflags(write=False)
        self.frames_rendered += 1
        buffer = Buffer(canvas, offset=(self._box[0], self._box[1]))
        self._frames[key] = buffer
        while len(self._frames) > self._capacity:
            self._frames.popitem(last=False)
        return buffer
