"""Bilingual burned-in subtitles with protected line breaks.

Merges two production references: the R23 subtitle pipeline (timing profile,
SRT/VTT/ASS delivery) and the NLH ``breaks``/``subs`` commands (never split a
word or a glossary name, zh-TW 72 px above English 42 px at 1080p).

The text layer is pure Python and font independent: ``break_lines`` takes a
``measure`` callable (text -> pixels), so layout is testable without fonts.
``subtitle_layer`` turns a whole cue list into one AE layer that draws the
active cue(s) at time ``t`` from an LRU cache of rendered blocks.

Notes
-----
CJK text may only break at word boundaries. When ``jieba`` is importable it
supplies them (protected terms are added to its dictionary); otherwise a
deterministic fallback trusts punctuation, closing brackets, spaces and
CJK/Latin script changes only. A long unpunctuated sentence then raises
``LayoutError`` instead of being cut mid-word; pass ``allow_char_breaks=True``
to opt in to character-level cuts (kinsoku, digit runs and protected terms
are still respected).

Examples
--------
>>> from moviepy.ae.templates.subtitles import Cue, break_lines, to_srt
>>> break_lines("Lin Yuqing reads slowly tonight", len, 20, lang="en",
...             protected=["Lin Yuqing"])
['Lin Yuqing reads', 'slowly tonight']
>>> print(to_srt([Cue(0.5, 3.5, "Hello")]), end="")
1
00:00:00,500 --> 00:00:03,500
Hello
<BLANKLINE>
"""

import json
import math
import re
import unicodedata
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.base import Layer


__all__ = [
    "Cue",
    "LayoutError",
    "BadBreak",
    "SubtitleTiming",
    "TimingIssue",
    "SubtitleLayer",
    "NO_LINE_START",
    "NO_LINE_END",
    "break_lines",
    "find_bad_breaks",
    "check_timing",
    "grapheme_count",
    "to_srt",
    "to_vtt",
    "to_ass",
    "parse_srt",
    "subtitle_layer",
    "burn_subtitles",
]

# Kinsoku: characters that may not start / end a line (NLH + common ASCII).
NO_LINE_START = frozenset("，。、．；：！？」』）》〉】〕｝﹂﹄︶︾﹀︼︺…‥)]},.;:!?")
NO_LINE_END = frozenset("「『（《〈【〔｛([{")
_BREAK_AFTER = frozenset("，、；：。！？…—")
_CLOSERS = frozenset("」』）》〉】〕｝")
_PREFER_AFTER = _BREAK_AFTER | frozenset(",;:.!?")
_REPEATED = frozenset("…—‥")
_NUM_TOKEN = re.compile(
    r"[〇零一二兩三四五六七八九十百千萬0-9０-９]+(?:年|月|日|歲|次|人|萬|里|天)?"
)
_LATIN_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-_]*")
_CJK_RANGES = (
    (0x2E80, 0x2FDF),
    (0x3000, 0x30FF),
    (0x3100, 0x312F),
    (0x31A0, 0x31BF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF),
    (0xF900, 0xFAFF),
    (0xFE30, 0xFE4F),
    (0xFF00, 0xFFEF),
)
_CJK_LANGS = ("zh", "ja", "ko")


class LayoutError(ValueError):
    """Raised when text cannot be laid out without truncating or splitting words."""


def _is_cjk_char(ch):
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def _is_cjk_lang(lang):
    return str(lang).lower().replace("_", "-").split("-")[0] in _CJK_LANGS


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class Cue:
    """One timed subtitle text in a single language.

    Parameters
    ----------
    start, end : float
        Seconds on the final timeline, ``0 <= start < end``.
    text : str
        Non-empty text (NFC normalized); ``"\\n"`` separates lines.
    lang : str, optional
        BCP-47-like tag. Defaults to ``"zh-TW"``.

    Examples
    --------
    >>> Cue(1, 2.5, "Hi", "en").duration
    1.5
    >>> Cue(2, 1, "bad")
    Traceback (most recent call last):
    ...
    ValueError: end must be greater than start
    """

    start: float
    end: float
    text: str
    lang: str = "zh-TW"

    def __post_init__(self):
        start = _finite(self.start, "start")
        end = _finite(self.end, "end")
        if start < 0:
            raise ValueError("start must be >= 0")
        if end <= start:
            raise ValueError("end must be greater than start")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        text = unicodedata.normalize("NFC", self.text.replace("\r\n", "\n"))
        text = "\n".join(line.strip() for line in text.strip().split("\n"))
        if not text:
            raise ValueError("text must not be empty")
        if not isinstance(self.lang, str) or not self.lang.strip():
            raise ValueError("lang must be a non-empty string")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "lang", self.lang.strip())

    @property
    def duration(self):
        """Cue length in seconds."""
        return self.end - self.start


# --------------------------------------------------------------------------- #
# Break analysis


def _protected_spans(text, protected, ignore_case):
    flags = re.IGNORECASE if ignore_case else 0
    spans = []
    for term in protected:
        term = str(term).strip()
        if len(term) >= 2:
            spans += [m.span() for m in re.finditer(re.escape(term), text, flags)]
    return spans


def _jieba_bounds(text, protected):
    try:
        import jieba
    except ImportError:
        return None
    jieba.setLogLevel(60)
    for term in protected:
        term = str(term).strip()
        if len(term) >= 2:
            jieba.add_word(term, freq=2_000_000)
    bounds, pos = set(), 0
    for word in jieba.lcut(text, HMM=False):
        pos += len(word)
        bounds.add(pos)
    return bounds


def _fallback_bounds(text):
    bounds = set()
    for i, ch in enumerate(text):
        if ch in _BREAK_AFTER or ch in _CLOSERS or ch.isspace():
            bounds.add(i + 1)
        if i and (ch.isspace() or text[i - 1].isspace()):
            bounds.add(i)
        if i and _is_cjk_char(ch) != _is_cjk_char(text[i - 1]):
            bounds.add(i)
    return bounds


class _CJKAnalysis:
    """Classify every inner index of ``text`` as an allowed or blocked cut."""

    def __init__(self, text, protected, use_jieba=True, allow_char_breaks=False):
        self.text = text
        self.allow_char = allow_char_breaks
        self.protected = _protected_spans(text, protected, False)
        self.numbers = [
            m.span() for m in _NUM_TOKEN.finditer(text) if m.end() - m.start() > 1
        ]
        self.latin = [
            m.span() for m in _LATIN_RUN.finditer(text) if m.end() - m.start() > 1
        ]
        bounds = _jieba_bounds(text, protected) if use_jieba else None
        self.bounds = bounds if bounds is not None else set()
        self.bounds |= _fallback_bounds(text) if bounds is None else set()

    @staticmethod
    def _inside(spans, i):
        return any(s < i < e for s, e in spans)

    def reason(self, i):
        """Return why a cut before ``text[i]`` is bad, or None when allowed."""
        text = self.text
        if self._inside(self.protected, i):
            return "inside_protected"
        if self._inside(self.numbers, i):
            return "inside_number"
        if self._inside(self.latin, i):
            return "inside_word"
        if text[i] in NO_LINE_START:
            return "line_start_forbidden"
        if text[i - 1] in NO_LINE_END:
            return "line_end_forbidden"
        if text[i] == text[i - 1] and text[i] in _REPEATED:
            return "inside_ellipsis"
        if not self.allow_char and i not in self.bounds:
            return "not_word_boundary"
        return None

    def cuts(self):
        return [i for i in range(1, len(self.text)) if self.reason(i) is None]


def _en_reason(text, i, spans):
    if any(s < i < e for s, e in spans):
        return "inside_protected"
    prev, nxt = text[i - 1], text[i]
    if prev == "-" and i >= 2 and text[i - 2].isalnum() and nxt.isalnum():
        return "inside_word"
    return None


def _en_cuts(text, protected):
    spans = _protected_spans(text, protected, True)
    out = []
    for i in range(1, len(text)):
        if text[i - 1] == " " and text[i] != " ":
            if not any(s < i < e for s, e in spans):
                out.append(i)
    return out


# --------------------------------------------------------------------------- #
# Layout


def _balanced_cuts(text, cuts, measure, max_width, max_lines):
    """Return the cut tuple for the fewest lines that fit, balanced; or None."""

    @lru_cache(maxsize=None)
    def width(a, b):
        return float(measure(text[a:b].strip()))

    total = width(0, len(text))
    cut_set = tuple(cuts)
    for lines in range(1, max_lines + 1):
        ideal = total / lines

        @lru_cache(maxsize=None)
        def go(pos, remaining):
            if remaining == 1:
                w = width(pos, len(text))
                if w > max_width:
                    return None
                return (((w - ideal) / max_width) ** 2, ())
            best = None
            for c in cut_set:
                if c <= pos:
                    continue
                if not text[pos:c].strip():
                    continue
                w = width(pos, c)
                if w > max_width:
                    break
                rest = go(c, remaining - 1)
                if rest is None:
                    continue
                bonus = 0.03 if text[c - 1].strip() in _PREFER_AFTER else 0.0
                cost = ((w - ideal) / max_width) ** 2 - bonus + rest[0]
                if best is None or cost < best[0] - 1e-12:
                    best = (cost, (c,) + rest[1])
            return best

        found = go(0, lines)
        if found is not None:
            return found[1]
    return None


def break_lines(
    text,
    measure,
    max_width,
    *,
    lang="zh-TW",
    protected=(),
    max_lines=2,
    use_jieba=True,
    allow_char_breaks=False,
):
    """Break ``text`` into balanced lines without splitting words or names.

    Parameters
    ----------
    text : str
        Subtitle text; explicit ``"\\n"`` are treated as forced line ends.
    measure : callable
        ``measure(str) -> float`` pixel width of one line.
    max_width : float
        Maximum line width in the units of ``measure``.
    lang : str, optional
        CJK languages (zh/ja/ko) break after punctuation or at word
        boundaries; every other language breaks at spaces only.
    protected : iterable of str, optional
        Glossary terms that must never be split.
    max_lines : int, optional
        Maximum number of lines (default 2).
    use_jieba : bool, optional
        Use ``jieba`` word boundaries when installed.
    allow_char_breaks : bool, optional
        Allow CJK cuts between any two characters that kinsoku, digit runs
        and protected terms permit.

    Returns
    -------
    list of str

    Raises
    ------
    LayoutError
        If the text cannot fit in ``max_lines`` lines. Nothing is truncated.

    Examples
    --------
    >>> break_lines("今天天氣很好，我們去公園散步。", lambda s: 10 * len(s), 100)
    ['今天天氣很好，', '我們去公園散步。']
    >>> break_lines("short", len, 3, lang="en")
    Traceback (most recent call last):
    ...
    moviepy.ae.templates.subtitles.LayoutError: cannot fit 'short' in 2 lines of width 3
    """
    if not callable(measure):
        raise TypeError("measure must be callable")
    max_width = _finite(max_width, "max_width")
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    if isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 1:
        raise ValueError("max_lines must be a positive integer")
    protected = tuple(protected)
    cjk = _is_cjk_lang(lang)
    segments = [
        s.strip() for s in unicodedata.normalize("NFC", text).split("\n") if s.strip()
    ]
    if not segments:
        raise ValueError("text must not be empty")
    result = []
    for segment in segments:
        budget = max_lines - len(result) - (len(segments) - 1 - len(result))
        if budget < 1:
            raise LayoutError(
                f"cannot fit {text!r} in {max_lines} lines of width {max_width:g}"
            )
        if cjk:
            segment = re.sub(r"\s+", " ", segment)
            cuts = _CJKAnalysis(segment, protected, use_jieba, allow_char_breaks).cuts()
        else:
            segment = re.sub(r"\s+", " ", segment)
            cuts = _en_cuts(segment, protected)
        found = _balanced_cuts(segment, cuts, measure, max_width, budget)
        if found is None:
            raise LayoutError(
                f"cannot fit {segment!r} in {budget} lines of width {max_width:g}"
            )
        edges = (0, *found, len(segment))
        result += [
            segment[edges[i] : edges[i + 1]].strip() for i in range(len(edges) - 1)
        ]
    return result


@dataclass(frozen=True)
class BadBreak:
    """One line break that splits a word, a protected term or breaks kinsoku."""

    block: int
    line: int
    reason: str
    context: str

    def __str__(self):
        return f"block {self.block} line {self.line}: {self.reason} ({self.context})"


def _split_lines(block):
    return [part for part in re.split(r"\\N|\n", block)]


def _blocks(items):
    if isinstance(items, (str, Cue)):
        items = [items]
    items = list(items)
    if items and all(isinstance(i, str) for i in items):
        if not any(("\n" in i or "\\N" in i) for i in items):
            return [[line.strip() for line in items]]
    blocks = []
    for item in items:
        text = item.text if isinstance(item, Cue) else str(item)
        blocks.append([part.strip() for part in _split_lines(text)])
    return blocks


def find_bad_breaks(lines_or_cues, protected=(), *, use_jieba=True):
    """Report line breaks that split words, protected terms or break kinsoku.

    Port of the NLH ``breaks`` command. ``lines_or_cues`` is an iterable of
    ``Cue`` or text blocks (``"\\n"`` or ASS ``"\\N"`` separate lines). A plain
    list of strings without any newline is treated as the lines of one block.

    Returns
    -------
    list of BadBreak
        Empty when every break is acceptable.

    Examples
    --------
    >>> find_bad_breaks(["Lin Yuqing reads", "slowly"], ["Lin Yuqing"])
    []
    >>> find_bad_breaks(["Lin", "Yuqing reads"], ["Lin Yuqing"])[0].reason
    'inside_protected'
    """
    protected = tuple(protected)
    problems = []
    for b, lines in enumerate(_blocks(lines_or_cues)):
        lines = [line for line in lines if line]
        cuts = []  # (index in full, line number, cjk)
        full = lines[0] if lines else ""
        for n in range(1, len(lines)):
            prev, nxt = full[-1:], lines[n][:1]
            cjk = _is_cjk_char(prev) or _is_cjk_char(nxt)
            joiner = "" if cjk else " "
            full += joiner + lines[n]
            cuts.append((len(full) - len(lines[n]), n, cjk))
        cjk_text = _CJKAnalysis(full, protected, use_jieba) if full else None
        spans = _protected_spans(full, protected, True)
        for index, n, cjk in cuts:
            reason = cjk_text.reason(index) if cjk else _en_reason(full, index, spans)
            if reason:
                context = (
                    full[max(0, index - 4) : index] + "|" + full[index : index + 4]
                )
                problems.append(BadBreak(b, n, reason, context))
    return problems


# --------------------------------------------------------------------------- #
# Timing


def grapheme_count(text):
    """Approximate grapheme count: marks, joiners and newlines are not counted.

    >>> grapheme_count("é\\nab")
    3
    """
    text = unicodedata.normalize("NFC", text)
    return sum(
        1
        for ch in text
        if ch not in "\n\r\u200d"
        and not unicodedata.combining(ch)
        and not (0xFE00 <= ord(ch) <= 0xFE0F)
    )


@dataclass(frozen=True)
class TimingIssue:
    """One reading-speed or timing problem found by ``check_timing``."""

    kind: str  # too_short, too_long, cps, overlap
    index: int
    message: str
    value: float = 0.0

    def __str__(self):
        return f"cue {self.index}: {self.kind} - {self.message}"


@dataclass(frozen=True)
class SubtitleTiming:
    """Reading-speed limits; defaults are the R23 ``subtitle_profile.json``.

    ``max_cps`` maps a language prefix (``zh``, ``en``) to the maximum
    graphemes per second, counting punctuation and spaces.

    Examples
    --------
    >>> timing = SubtitleTiming()
    >>> timing.limit_cps("zh-TW"), timing.limit_cps("en"), timing.min_duration
    (12.0, 17.0, 1.0)
    >>> [i.kind for i in timing.check([Cue(0, 0.5, "x" * 20, "en")])]
    ['cps', 'too_short']
    """

    min_duration: float = 1.0
    max_duration: float = 7.0
    max_cps: dict = field(default_factory=lambda: {"zh": 12.0, "en": 17.0})
    default_cps: float = 17.0
    max_merge_gap: float = 0.2

    def __post_init__(self):
        for name in ("min_duration", "max_duration", "default_cps", "max_merge_gap"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not 0 < self.min_duration <= self.max_duration:
            raise ValueError("need 0 < min_duration <= max_duration")
        if self.default_cps <= 0:
            raise ValueError("default_cps must be positive")
        limits = {
            str(k).lower(): _finite(v, "max_cps") for k, v in self.max_cps.items()
        }
        if any(v <= 0 for v in limits.values()):
            raise ValueError("max_cps values must be positive")
        object.__setattr__(self, "max_cps", limits)

    @classmethod
    def from_profile(cls, source):
        """Build from an R23 ``subtitle_profile.json`` path, text or dict."""
        if isinstance(source, Path) or (
            isinstance(source, str) and not source.lstrip().startswith("{")
        ):
            source = Path(source).read_text(encoding="utf-8-sig")
        data = json.loads(source) if isinstance(source, str) else dict(source)
        cps = {}
        for lang, cfg in data.get("languages", {}).items():
            if "max_cps" in cfg:
                cps.setdefault(lang.lower().split("-")[0], float(cfg["max_cps"]))
        kwargs = {"max_cps": cps} if cps else {}
        for key in ("min_duration", "max_duration", "max_merge_gap"):
            if key in data:
                kwargs[key] = data[key]
        return cls(**kwargs)

    def limit_cps(self, lang):
        """Return the CPS limit for ``lang``."""
        base = str(lang).lower().replace("_", "-").split("-")[0]
        return self.max_cps.get(base, self.default_cps)

    def cps(self, cue):
        """Return graphemes per second of ``cue``."""
        return grapheme_count(cue.text) / cue.duration

    def check(self, cues):
        """Return a list of ``TimingIssue`` for ``cues`` (empty when clean)."""
        cues = list(cues)
        issues = []
        for i, cue in enumerate(cues):
            if cue.duration < self.min_duration - 1e-9:
                issues.append(
                    TimingIssue(
                        "too_short",
                        i,
                        f"{cue.duration:.3f}s < {self.min_duration:g}s",
                        cue.duration,
                    )
                )
            if cue.duration > self.max_duration + 1e-9:
                issues.append(
                    TimingIssue(
                        "too_long",
                        i,
                        f"{cue.duration:.3f}s > {self.max_duration:g}s",
                        cue.duration,
                    )
                )
            rate, limit = self.cps(cue), self.limit_cps(cue.lang)
            if rate > limit + 1e-9:
                issues.append(
                    TimingIssue("cps", i, f"{rate:.1f} cps > {limit:g}", rate)
                )
        by_lang = {}
        for i, cue in enumerate(cues):
            by_lang.setdefault(cue.lang, []).append(i)
        for indices in by_lang.values():
            indices.sort(key=lambda k: (cues[k].start, cues[k].end))
            for a, b in zip(indices, indices[1:]):
                over = cues[a].end - cues[b].start
                if over > 1e-9:
                    issues.append(
                        TimingIssue(
                            "overlap", b, f"overlaps cue {a} by {over:.3f}s", over
                        )
                    )
        return sorted(issues, key=lambda issue: (issue.index, issue.kind))


def check_timing(cues, timing=None):
    """Check CPS, minimum/maximum duration and same-language overlaps.

    Examples
    --------
    >>> check_timing([Cue(0, 2, "ok", "en"), Cue(1.5, 4, "overlap", "en")])[0].kind
    'overlap'
    """
    return (timing or SubtitleTiming()).check(cues)


# --------------------------------------------------------------------------- #
# Formats


def _ms(seconds):
    return int(round(seconds * 1000))


def _clock(seconds, sep):
    ms = _ms(seconds)
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_srt(cues, *, escape=False):
    """Serialize cues as SubRip text (millisecond times, UTF-8 friendly).

    >>> print(to_srt([Cue(0, 1.5, "a\\nb")]), end="")
    1
    00:00:00,000 --> 00:00:01,500
    a
    b
    <BLANKLINE>
    """
    out = []
    for n, cue in enumerate(sorted(cues, key=lambda c: (c.start, c.end)), 1):
        text = _escape(cue.text) if escape else cue.text
        out.append(
            f"{n}\n{_clock(cue.start, ',')} --> {_clock(cue.end, ',')}\n{text}\n\n"
        )
    return "".join(out)


def to_vtt(cues):
    """Serialize cues as WebVTT; ``&``, ``<`` and ``>`` are escaped.

    >>> print(to_vtt([Cue(0, 1, "a&b")]), end="")
    WEBVTT
    <BLANKLINE>
    00:00:00.000 --> 00:00:01.000
    a&amp;b
    <BLANKLINE>
    """
    out = ["WEBVTT\n\n"]
    for cue in sorted(cues, key=lambda c: (c.start, c.end)):
        out.append(
            f"{_clock(cue.start, '.')} --> {_clock(cue.end, '.')}\n"
            f"{_escape(cue.text)}\n\n"
        )
    return "".join(out)


_SRT_TIME = re.compile(
    r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _srt_seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text, lang="zh-TW"):
    """Parse SubRip text into cues (BOM, CRLF and missing indexes tolerated).

    >>> parse_srt("1\\n00:00:01,000 --> 00:00:02,500\\nHi\\n", "en")
    [Cue(start=1.0, end=2.5, text='Hi', lang='en')]
    """
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in block.split("\n")]
        if not any(line.strip() for line in lines):
            continue
        for k, line in enumerate(lines):
            match = _SRT_TIME.search(line)
            if match:
                g = match.groups()
                body = "\n".join(lines[k + 1 :]).strip()
                cues.append(Cue(_srt_seconds(*g[:4]), _srt_seconds(*g[4:]), body, lang))
                break
        else:
            raise ValueError(f"no timing line in SRT block: {block[:40]!r}")
    return cues


def _ass_clock(seconds):
    cs = int(round(seconds * 100))
    h, rest = divmod(cs, 360_000)
    m, rest = divmod(rest, 6000)
    s, cs = divmod(rest, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_color(rgb):
    r, g, b = (int(c) for c in rgb)
    return f"&H00{b:02X}{g:02X}{r:02X}"


def _ass_text(text):
    if any(ch in text for ch in "{}\\"):
        raise ValueError("ASS text must not contain '{', '}' or backslash")
    return text.replace("\n", "\\N")


def to_ass(
    cues,
    style,
    size=(1920, 1080),
    *,
    primary_lang=None,
    font_name="Microsoft JhengHei",
    secondary_font_name=None,
):
    """Serialize cues as an ASS script that mirrors the burned-in layout.

    Parameters
    ----------
    cues : iterable of Cue
        Primary and secondary language cues together.
    style : SubtitleStyle
        Geometry from ``preset.subtitles``; sizes scale to ``size[1]``.
    size : tuple of int
        ``PlayResX``/``PlayResY``.
    primary_lang : str, optional
        Language drawn large and above; defaults to the first cue's language.
        Every other language uses the secondary style, placed at the bottom.

    Examples
    --------
    >>> from moviepy.ae.templates.presets import get_preset
    >>> ass = to_ass([Cue(0, 1, "你好")], get_preset("nightlamp_story").subtitles)
    >>> "Dialogue: 0,0:00:00.00,0:00:01.00,Primary,,0,0,60,,你好" in ass
    True
    """
    cues = sorted(cues, key=lambda c: (c.start, c.end))
    if not cues:
        raise ValueError("cues must not be empty")
    width, height = (int(v) for v in size)
    scale = height / style.reference_height
    primary_lang = primary_lang or cues[0].lang
    secondary_font_name = secondary_font_name or font_name
    side = int(round(width * (1 - style.max_width) / 2))
    margin = int(round(style.margin_bottom * scale))
    secondary_px = style.secondary_size * scale
    gap = style.line_gap * scale

    def style_line(name, font, px, color):
        return (
            f"Style: {name},{font},{px:g},{_ass_color(color)},{_ass_color(color)},"
            f"{_ass_color(style.outline_color)},&H00000000,0,0,0,0,100,100,0,0,1,"
            f"{style.outline_width * scale:g},0,2,{side},{side},{margin},1"
        )

    secondary = [c for c in cues if c.lang != primary_lang]
    lines = [
        "[Script Info]",
        "; Generated by moviepy.ae.templates.subtitles",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        style_line(
            "Primary",
            font_name,
            round(style.primary_size * scale, 2),
            style.primary_color,
        ),
        style_line(
            "Secondary",
            secondary_font_name,
            round(secondary_px, 2),
            style.secondary_color,
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    for cue in cues:
        is_primary = cue.lang == primary_lang
        lift = 0
        if is_primary:
            for other in secondary:
                if other.start < cue.end and cue.start < other.end:
                    rows = other.text.count("\n") + 1
                    lift = max(lift, int(round(rows * secondary_px * 1.2 + gap)))
        lines.append(
            f"Dialogue: 0,{_ass_clock(cue.start)},{_ass_clock(cue.end)},"
            f"{'Primary' if is_primary else 'Secondary'},,0,0,{margin + lift},,"
            f"{_ass_text(cue.text)}"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Burn-in layer


def _load_font(spec, px):
    from PIL import ImageFont

    if spec is None or isinstance(spec, str) and spec == "default":
        return ImageFont.load_default(size=max(1, int(round(px))))
    if isinstance(spec, (str, Path)):
        path = Path(spec).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"subtitle font not found: {path}")
        return ImageFont.truetype(str(path), max(1, int(round(px))))
    return spec  # an already-built PIL font object


class SubtitleLayer(Layer):
    """AE layer that draws the cue(s) active at ``t`` from a cached raster set.

    The layer is a full-canvas, transparent-between-cues source. Each
    language block is rendered once (Pillow text with a real outline) and kept
    in an LRU cache; a frame costs a lookup plus the usual layer composite.

    Parameters
    ----------
    cues, secondary : sequence of Cue
        Primary language (large, above) and optional secondary (small, below).
    preset : ChannelPreset
        Supplies ``subtitles`` geometry, fonts, ``size`` and ``scaled_px``.
    font, secondary_font : str, path or PIL font, optional
        Font path, an ImageFont, or ``"default"`` for Pillow's built-in font.
        ``font=None`` resolves ``preset.font("body")``; ``secondary_font=None``
        reuses the primary font. Missing files raise, never swap silently.
    size : tuple of int, optional
        Canvas size; defaults to ``preset.size``.
    protected : iterable of str, optional
        Glossary terms that line breaking must keep whole.
    cache_size : int, optional
        LRU capacity for composed frames and rendered blocks.

    Notes
    -----
    All cues are laid out when the layer is built, so an unfittable cue raises
    ``LayoutError`` immediately. Lines already within ``max_width`` and
    ``max_lines`` keep the author's breaks; anything else is re-wrapped.
    """

    def __init__(
        self,
        cues,
        preset,
        *,
        secondary=None,
        font=None,
        secondary_font=None,
        size=None,
        protected=(),
        cache_size=64,
        name="Subtitles",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.preset = preset
        self._size = tuple(size or preset.size)
        self._style = style = preset.subtitles
        s = lambda v: preset.scaled_px(v, self._size)  # noqa: E731
        self._px = {
            "primary": s(style.primary_size),
            "secondary": s(style.secondary_size),
        }
        self._stroke = int(round(s(style.outline_width)))
        self._gap = int(round(s(style.line_gap)))
        self._margin = int(round(s(style.margin_bottom)))
        if font is None:
            font = preset.font("body")
        if secondary_font is None:
            secondary_font = font
        self._fonts = {
            "primary": _load_font(font, self._px["primary"]),
            "secondary": _load_font(secondary_font, self._px["secondary"]),
        }
        self._protected = tuple(protected)
        self._max_width = style.max_width * self._size[0]
        self._lanes = {
            "primary": self._prepare(cues, "primary"),
            "secondary": self._prepare(secondary or (), "secondary"),
        }
        self._cache = OrderedDict()
        self._blocks = OrderedDict()
        self._capacity = max(1, int(cache_size))
        self.cache_hits = 0
        self.cache_misses = 0
        self.rasters = 0
        times = [c.start for lane in self._lanes.values() for c in lane["cues"]]
        ends = [c.end for lane in self._lanes.values() for c in lane["cues"]]
        if times:
            self.in_point = min(times)
            self.out_point = max(ends)
        self._empty = Buffer(np.zeros((1, 1, 4), np.float32))

    # -- layout -------------------------------------------------------------- #

    def _measure(self, role):
        font, pad = self._fonts[role], 2 * self._stroke
        return lambda text: font.getlength(text) + pad

    def _lines_for(self, cue, role):
        measure = self._measure(role)
        lines = cue.text.split("\n")
        if len(lines) <= self._style.max_lines and all(
            measure(line) <= self._max_width for line in lines
        ):
            return tuple(lines)
        joined = "\n".join(lines) if len(lines) > 1 else cue.text
        flat = (
            joined.replace("\n", "")
            if _is_cjk_lang(cue.lang)
            else joined.replace("\n", " ")
        )
        return tuple(
            break_lines(
                flat,
                measure,
                self._max_width,
                lang=cue.lang,
                protected=self._protected,
                max_lines=self._style.max_lines,
            )
        )

    def _prepare(self, cues, role):
        cues = sorted(cues, key=lambda c: (c.start, c.end))
        for cue in cues:
            if not isinstance(cue, Cue):
                raise TypeError("cues must be Cue instances")
        return {
            "cues": cues,
            "starts": [c.start for c in cues],
            "lines": [self._lines_for(c, role) for c in cues],
        }

    # -- timing / lookup ----------------------------------------------------- #

    def active_index(self, role, t):
        """Return the index of the latest-starting cue active at ``t`` or None."""
        lane = self._lanes[role]
        i = bisect_right(lane["starts"], t) - 1
        while i >= 0:
            if lane["cues"][i].end > t:
                return i
            i -= 1
        return None

    def active_cues(self, t):
        """Return ``(primary, secondary)`` cues active at ``t`` (None if absent)."""
        out = []
        for role in ("primary", "secondary"):
            i = self.active_index(role, t)
            out.append(None if i is None else self._lanes[role]["cues"][i])
        return tuple(out)

    @property
    def source_size(self):
        """Canvas size in pixels."""
        return self._size

    # -- rendering ----------------------------------------------------------- #

    def _block(self, role, lines):
        key = (role, lines)
        block = self._blocks.get(key)
        if block is not None:
            self._blocks.move_to_end(key)
            return block
        block = self._raster(role, lines)
        self._blocks[key] = block
        while len(self._blocks) > self._capacity:
            self._blocks.popitem(last=False)
        return block

    def _raster(self, role, lines):
        from PIL import Image, ImageDraw

        self.rasters += 1
        font, stroke = self._fonts[role], self._stroke
        color = (
            self._style.primary_color
            if role == "primary"
            else (self._style.secondary_color)
        )
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        pad = stroke + 2
        step = int(round(line_h * 1.1))
        width = int(math.ceil(max(font.getlength(line) for line in lines))) + 2 * pad
        height = step * (len(lines) - 1) + line_h + 2 * pad
        outline_mask = Image.new("L", (width, height), 0)
        fill_mask = Image.new("L", (width, height), 0)
        d_out, d_fill = ImageDraw.Draw(outline_mask), ImageDraw.Draw(fill_mask)
        for n, line in enumerate(lines):
            x = (width - font.getlength(line)) / 2
            y = pad + n * step
            d_out.text((x, y), line, font=font, fill=255, stroke_width=stroke)
            d_fill.text((x, y), line, font=font, fill=255)
        oa = np.asarray(outline_mask, np.float32) / 255.0
        fa = np.asarray(fill_mask, np.float32) / 255.0
        oa = np.maximum(oa, fa)
        out = np.empty((height, width, 4), np.float32)
        fill = np.asarray(color, np.float32) / 255.0
        edge = np.asarray(self._style.outline_color, np.float32) / 255.0
        under = oa * (1.0 - fa)
        out[..., :3] = fill * fa[..., None] + edge * under[..., None]
        out[..., 3] = fa + under
        out.setflags(write=False)
        return out

    def _compose(self, p_lines, s_lines):
        blocks = []  # top to bottom
        if p_lines is not None:
            blocks.append(self._block("primary", p_lines))
        if s_lines is not None:
            blocks.append(self._block("secondary", s_lines))
        if not blocks:
            return self._empty
        gap = self._gap if len(blocks) == 2 else 0
        width = max(b.shape[1] for b in blocks)
        height = sum(b.shape[0] for b in blocks) + gap
        canvas = np.zeros((height, width, 4), np.float32)
        y = 0
        for block in blocks:
            h, w = block.shape[:2]
            x = (width - w) // 2
            canvas[y : y + h, x : x + w] = block
            y += h + gap
        # Trailing pad rows sit under the margin line, as in a text box.
        cw, ch = self._size
        x0 = (cw - width) // 2
        y0 = ch - self._margin - height
        return Buffer(canvas, offset=(x0, y0))

    def source_buffer(self, t, context=None):
        """Return the premultiplied subtitle block active at layer time ``t``."""
        keys = []
        lines = []
        for role in ("primary", "secondary"):
            i = self.active_index(role, t)
            keys.append(i)
            lines.append(None if i is None else self._lanes[role]["lines"][i])
        key = tuple(keys)
        buffer = self._cache.get(key)
        if buffer is not None:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return buffer
        self.cache_misses += 1
        buffer = self._compose(*lines)
        self._cache[key] = buffer
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
        return buffer

    def cache_info(self):
        """Return ``(hits, misses, cached_frames, rasters)``."""
        return (self.cache_hits, self.cache_misses, len(self._cache), self.rasters)


def subtitle_layer(
    cues, preset, *, secondary=None, font=None, secondary_font=None, **kw
):
    """Build a ``SubtitleLayer`` for a whole timeline.

    Returns
    -------
    SubtitleLayer
        An ordinary AE layer; add it to a ``Composition`` with ``add_layer`` and
        keyframe ``layer.transform`` as usual.
    """
    return SubtitleLayer(
        cues,
        preset,
        secondary=secondary,
        font=font,
        secondary_font=secondary_font,
        **kw,
    )


def burn_subtitles(
    clip_or_comp, cues, preset, *, secondary=None, font=None, secondary_font=None, **kw
):
    """Burn subtitles into a clip or composition and return a ``Composition``.

    A ``Composition`` gets the subtitle layer added on top **in place** and is
    returned; any other MoviePy clip is wrapped in a new ``Composition`` first.
    """
    from moviepy.ae.composition import Composition

    if isinstance(clip_or_comp, Composition):
        comp = clip_or_comp
    else:
        clip = clip_or_comp
        comp = Composition(
            size=tuple(int(v) for v in clip.size),
            fps=clip.fps or preset.fps,
            duration=clip.duration,
            name="Subtitled",
        )
        comp.add_clip(clip, name="Video")
    layer = subtitle_layer(
        cues,
        preset,
        secondary=secondary,
        font=font,
        secondary_font=secondary_font,
        size=comp.size,
        **kw,
    )
    comp.add_layer(layer)
    return comp
