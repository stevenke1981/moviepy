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
supplies them (its global dictionary is never mutated: protected terms are
segmented around, not added); otherwise a
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
    "Word",
    "load_words",
    "load_glossary",
    "reflow_cues",
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
NO_LINE_START = frozenset("，。、．；：！？」』）》”’〉】〕｝﹂﹄︶︾﹀︼︺…‥)]},.;:!?")
NO_LINE_END = frozenset("「『（《〈【〔｛([{“‘")
_BREAK_AFTER = frozenset("，、；：。！？…—")
_CLOSERS = frozenset("」』）》〉】〕｝”’")
_PREFER_AFTER = _BREAK_AFTER | frozenset(",;:.!?")
# Ends of a spoken pause: a cue may end here without splitting a phrase.
_PAUSE_END = _PREFER_AFTER | frozenset("」』”’")
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
    r"""One timed subtitle text in a single language.

    Parameters
    ----------
    start, end : float
        Seconds on the final timeline, ``0 <= start < end``.
    text : str
        Non-empty text (NFC normalized); ``"\n"`` separates lines.
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


def _merge_spans(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _jieba_bounds(text, spans):
    """Word boundaries from jieba, or None when it is not installed.

    jieba's dictionary is process-global, so protected terms are *not* added
    to it. Instead the text is cut at the protected spans and each remaining
    piece is segmented on its own; a span's edges are boundaries by
    construction, and cuts inside a span are rejected by the caller.
    """
    try:
        import jieba
    except ImportError:
        return None
    jieba.setLogLevel(60)
    bounds, pos = set(), 0
    for start, end in [*_merge_spans(spans), (len(text), len(text))]:
        offset = pos
        for word in jieba.lcut(text[pos:start], HMM=False):
            offset += len(word)
            bounds.add(offset)
        bounds.add(end)
        pos = end
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
        bounds = _jieba_bounds(text, self.protected) if use_jieba else None
        self.jieba = bounds is not None
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
    r"""Break ``text`` into balanced lines without splitting words or names.

    Parameters
    ----------
    text : str
        Subtitle text; explicit ``"\n"`` are treated as forced line ends.
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
    for n, segment in enumerate(segments):
        # Every segment still to come needs at least one line.
        budget = max_lines - len(result) - (len(segments) - 1 - n)
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


# --------------------------------------------------------------------------- #
# Reflow: split cues that cannot fit into several timed cues


@dataclass(frozen=True)
class Word:
    """One ASR word (or CJK token) with its time span in seconds.

    Examples
    --------
    >>> Word("嬌娜", 1.0, 1.4).text
    '嬌娜'
    """

    text: str
    start: float
    end: float

    def __post_init__(self):
        start = _finite(self.start, "start")
        end = _finite(self.end, "end")
        if start < 0 or end < start:
            raise ValueError("word times need 0 <= start <= end")
        if not isinstance(self.text, str):
            raise ValueError("word text must be a string")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


def load_words(source):
    """Read word timings from ASR JSON (Qwen3ASR and similar shapes).

    ``source`` is a path, a JSON string, a dict or a list. Accepted shapes are
    ``{"words": [...]}``, ``{"segments": [{"words": [...]}]}`` or a bare list;
    each item has ``text`` (or ``word``), ``start`` and ``end`` in seconds.

    Examples
    --------
    >>> load_words({"words": [{"text": "hi", "start": 0, "end": 0.4}]})
    [Word(text='hi', start=0.0, end=0.4)]
    """
    data = source
    if isinstance(source, Path) or (
        isinstance(source, str) and not source.lstrip().startswith(("{", "["))
    ):
        data = Path(source).read_text(encoding="utf-8-sig")
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict):
        if "words" in data:
            data = data["words"]
        elif "segments" in data:
            data = [w for seg in data["segments"] for w in seg.get("words", ())]
        else:
            raise ValueError("expected a 'words' or 'segments' list")
    words = []
    for item in data:
        text = item.get("text", item.get("word"))
        words.append(Word(text, item["start"], item["end"]))
    return words


def _is_key(ch):
    return unicodedata.category(ch)[0] in "LN"


def _word_owners(texts, words):
    """Map each character of each text to the index of its ASR word.

    Letters and digits of all texts are aligned against those of all words
    with ``difflib`` (equal runs and equal-length replacements map one to
    one); punctuation, spaces and unmatched characters map to ``None``.
    """
    from difflib import SequenceMatcher

    word_keys, word_owner = [], []
    for index, word in enumerate(words):
        for ch in unicodedata.normalize("NFC", word.text):
            if _is_key(ch):
                word_keys.append(ch.casefold())
                word_owner.append(index)
    text_keys, text_pos = [], []
    for t, text in enumerate(texts):
        for i, ch in enumerate(text):
            if _is_key(ch):
                text_keys.append(ch.casefold())
                text_pos.append((t, i))
    owners = [[None] * len(text) for text in texts]
    matcher = SequenceMatcher(None, text_keys, word_keys, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and a1 - a0 == b1 - b0):
            for k in range(a1 - a0):
                t, i = text_pos[a0 + k]
                owners[t][i] = word_owner[b0 + k]
    return owners


def _neighbours(owner, cut):
    """Return the word owners left and right of ``cut`` (skipping punctuation)."""
    left = next((o for o in reversed(owner[:cut]) if o is not None), None)
    right = next((o for o in owner[cut:] if o is not None), None)
    return left, right


def _piece_lines(text, a, b, cuts, measure, max_width, max_lines, weak=()):
    """Line cuts (relative to ``a``) for ``text[a:b]``, or None if it can't fit.

    Line breaks first use only strong cuts (punctuation, spaces, dictionary
    words); ``weak`` cuts (an ASR token edge and nothing else) are a fallback.
    """
    piece = text[a:b]
    if not piece.strip():
        return None
    inner = [c - a for c in cuts if a < c < b]
    if max_lines > 1 and weak:
        strong = [c for c in inner if c + a not in weak]
        if len(strong) < len(inner):
            found = _balanced_cuts(piece, strong, measure, max_width, max_lines)
            if found is not None:
                return found
    return _balanced_cuts(piece, inner, measure, max_width, max_lines)


def _split_cost(piece, width, max_width, max_lines, lines, last, weak):
    """Cost of one output cue; lower is better.

    Every cue costs about 1, so fewer cues win among cuts of the same kind.
    A cut inside a phrase (a word edge) costs more than one extra cue, so an
    extra cue that ends at punctuation is preferred; a cut that only an ASR
    token edge allows (no dictionary or punctuation evidence) costs most.
    """
    fill = min(1.0, width / (max_width * max_lines))
    cost = 1.0 + 0.35 * (1.0 - fill) ** 2 + 0.25 * len(lines)
    if not last:
        if grapheme_count(piece) < 5:
            cost += 0.2
        if piece[-1:] in _PAUSE_END:
            cost -= 0.25
        elif weak:
            cost += 2.5
        else:
            cost += 1.2
    return cost


def _flat_text(cue):
    if _is_cjk_lang(cue.lang):
        return cue.text.replace("\n", "")
    return re.sub(r"\s+", " ", cue.text.replace("\n", " ")).strip()


def _bad_join(left, right, left_owner, right_owner, lang, protected, use_jieba):
    """Return True when the boundary between two consecutive cues is a bad cut.

    A cue that does not end at punctuation stops mid-phrase, so it is always
    re-laid out together with its neighbour (the layout still keeps the old
    cut when nothing better fits). Otherwise the boundary is bad when it
    falls inside one ASR word, a protected name, a number or a Latin word, or
    breaks kinsoku. Only a window around the boundary is analysed.
    """
    if left[-1:] not in _PREFER_AFTER and left[-1:] not in _CLOSERS:
        return True
    if left_owner and right_owner:
        a, b = left_owner[-1], right_owner[0]
        if a is not None and a == b:
            return True
    if _is_cjk_lang(lang):
        window = 24
        joined = left[-window:] + right[:window]
        analysis = _CJKAnalysis(joined, protected, use_jieba)
        return analysis.reason(min(window, len(left))) is not None
    joined = f"{left} {right}"
    cut = len(left) + 1
    spans = _protected_spans(joined, protected, True)
    return any(s < cut < e for s, e in spans)


def _source_time(text, parts, cut):
    """Interpolate a time for ``cut`` inside the source cue that contains it."""
    k = max(i for i, (off, _) in enumerate(parts) if off <= cut)
    offset, cue = parts[k]
    stop = parts[k + 1][0] if k + 1 < len(parts) else len(text)
    keys = [i for i in range(offset, stop) if _is_key(text[i])]
    before = sum(1 for i in keys if i < cut)
    return cue.start + cue.duration * before / max(1, len(keys))


def reflow_cues(
    cues,
    measure,
    max_width,
    *,
    max_lines=1,
    protected=(),
    words=None,
    use_jieba=True,
    allow_char_breaks=False,
    max_gap=0.3,
    rejoin=True,
    rejoin_gap=0.6,
    report=None,
):
    r"""Split cues that overflow ``max_lines`` into consecutive, re-timed cues.

    Port of the R26 "semantic pixel width" subtitle layout: a cue that fits is
    only re-wrapped; one that does not is cut into the fewest pieces that each
    fit, never inside a word, a number or a ``protected`` name, preferring
    cuts after punctuation and pieces of balanced width. Text is never
    dropped, truncated or changed (``"".join`` of the pieces equals the
    original text without its line breaks).

    Parameters
    ----------
    cues : iterable of Cue
        Source cues; each may hold one or several sentences.
    measure : callable
        ``measure(str) -> float`` pixel width of one line (include the
        outline; ``SubtitleLayer`` measures with its own font).
    max_width : float
        Usable line width in the units of ``measure``.
    max_lines : int, optional
        Lines per output cue; the default ``1`` gives one-line captions.
    protected : iterable of str, optional
        Names and glossary terms that must stay whole.
    words : sequence of Word, optional
        ASR word timings for the same text (see ``load_words``). Their edges
        become extra cut points, cuts inside one ASR word are refused, and
        each new boundary is timed from the words around it. Without words the
        boundary time is proportional to the letters and digits before it.
    use_jieba, allow_char_breaks : bool, optional
        As in ``break_lines``.
    max_gap : float, optional
        When the pause between the words around a cut is at most this many
        seconds the next cue starts as the previous one ends (no flicker);
        longer pauses are kept as gaps.
    rejoin : bool, optional
        First merge consecutive cues whose shared boundary is itself a bad
        cut (inside a name, number, word or ASR word, or against kinsoku),
        then lay the merged text out again. The original boundary time is
        reused wherever a new cut lands on an old one.
    rejoin_gap : float, optional
        Largest gap in seconds between two cues that may still be merged.
    report : list, optional
        When given, one dict per output group is appended: ``index`` (first
        source cue), ``sources``, ``text``, ``pieces`` and ``timing``
        (``"unchanged"``, ``"rejoined"``, ``"source"``, ``"words"``,
        ``"proportional"`` or ``"mixed"``).

    Returns
    -------
    list of Cue
        Ordered output cues; lines inside a cue are joined with ``"\n"``.

    Raises
    ------
    LayoutError
        If some text cannot be cut into fitting pieces at an allowed point.

    Examples
    --------
    >>> cue = Cue(0.0, 4.0, "今天天氣很好，我們去公園玩。")
    >>> out = reflow_cues([cue], lambda s: 10 * len(s), 80)
    >>> [(c.text, c.start, c.end) for c in out]
    [('今天天氣很好，', 0.0, 2.0), ('我們去公園玩。', 2.0, 4.0)]
    """
    if not callable(measure):
        raise TypeError("measure must be callable")
    max_width = _finite(max_width, "max_width")
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    if isinstance(max_lines, bool) or not isinstance(max_lines, int) or max_lines < 1:
        raise ValueError("max_lines must be a positive integer")
    max_gap = _finite(max_gap, "max_gap")
    protected = tuple(protected)
    cues = sorted(cues, key=lambda c: (c.start, c.end))
    for cue in cues:
        if not isinstance(cue, Cue):
            raise TypeError("cues must be Cue instances")
    widths = {}

    def measured(line):
        width = widths.get(line)
        if width is None:
            width = widths[line] = float(measure(line))
        return width

    flats = [_flat_text(cue) for cue in cues]
    words = list(words or ())
    owners = _word_owners(flats, words) if words else [[None] * len(t) for t in flats]
    groups = []
    for i, cue in enumerate(cues):
        prev = cues[i - 1] if i else None
        if (
            rejoin
            and prev is not None
            and prev.lang == cue.lang
            and cue.start - prev.end <= rejoin_gap
            and _bad_join(
                flats[i - 1],
                flats[i],
                owners[i - 1],
                owners[i],
                cue.lang,
                protected,
                use_jieba,
            )
        ):
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    for group in groups:
        first, last = cues[group[0]], cues[group[-1]]
        cjk = _is_cjk_lang(first.lang)
        entry = {"index": group[0], "sources": list(group)}
        if len(group) == 1:
            lines = first.text.split("\n")
            if len(lines) <= max_lines and all(
                float(measure(line)) <= max_width for line in lines
            ):
                out.append(first)
                if report is not None:
                    entry.update(text=first.text, pieces=1, timing="unchanged")
                    report.append(entry)
                continue
        joiner = "" if cjk else " "
        text, owner, parts = "", [], []
        for i in group:
            if text and joiner:
                text += joiner
                owner.append(None)
            parts.append((len(text), cues[i]))
            text += flats[i]
            owner += owners[i]
        weak = set()
        if cjk:
            analysis = _CJKAnalysis(text, protected, use_jieba, allow_char_breaks)
            weak = set(range(len(text) + 1)) - analysis.bounds
            cuts = []
            for i in range(1, len(text)):
                reason = analysis.reason(i)
                left, right = _neighbours(owner, i)
                edge = left is not None and right is not None and left != right
                if reason is None or (reason == "not_word_boundary" and edge):
                    cuts.append(i)
        else:
            cuts = _en_cuts(text, protected)
        # A cut between two characters of one ASR word would split the word.
        cuts = [
            c
            for c in cuts
            if not (
                owner[c - 1] is not None
                and owner[c] is not None
                and owner[c - 1] == owner[c]
            )
        ]
        n = len(text)
        edges = [0, *cuts, n]
        best = {n: (0.0, None, None)}
        for a in reversed(edges[:-1]):
            choice = None
            for b in edges:
                if b <= a or b not in best:
                    continue
                piece = text[a:b].strip()
                width = measured(piece)
                if width > max_width * max_lines:
                    break
                found = _piece_lines(
                    text, a, b, cuts, measured, max_width, max_lines, weak
                )
                if found is None:
                    continue
                cost = best[b][0] + _split_cost(
                    piece, width, max_width, max_lines, found, b == n, b in weak
                )
                if choice is None or cost < choice[0] - 1e-12:
                    choice = (cost, b, found)
            if choice is not None:
                best[a] = choice
        if 0 not in best:
            raise LayoutError(
                f"cannot split {text!r} into cues of {max_lines} line(s) "
                f"of width {max_width:g}"
            )
        spans, a = [], 0
        while a < n:
            _, b, found = best[a]
            piece = text[a:b]
            marks = (0, *found, len(piece))
            body = "\n".join(
                piece[marks[i] : marks[i + 1]].strip() for i in range(len(marks) - 1)
            )
            spans.append((a, body))
            a = b
        times, methods = [first.start], set()
        for a, _ in spans[1:]:
            left, right = _neighbours(owner, a)
            source = next(
                (k for k, (off, _) in enumerate(parts) if k and off == a), None
            )
            if source is not None:
                end, start = parts[source - 1][1].end, parts[source][1].start
                methods.add("source")
            elif left is not None and right is not None and left != right:
                start = words[right].start
                end = min(words[left].end, start)
                if start - end <= max_gap:
                    end = start
                methods.add("words")
            else:
                end = start = _source_time(text, parts, a)
                methods.add("proportional")
            end = min(max(end, times[-1]), last.end)
            start = min(max(start, end), last.end)
            times += [end, start]
        times.append(last.end)
        pieces = []
        for k, (_, body) in enumerate(spans):
            start, end = times[2 * k], times[2 * k + 1]
            if end <= start:
                raise LayoutError(
                    f"split of {text!r} leaves an empty time span at {body!r}"
                )
            pieces.append(Cue(start, end, body, first.lang))
        out += pieces
        if report is not None:
            if not methods:
                timing = "rejoined"
            elif len(methods) == 1:
                timing = methods.pop()
            else:
                timing = "mixed"
            entry.update(text=text, pieces=len(pieces), timing=timing)
            report.append(entry)
    return out


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
    r"""Report line breaks that split words, protected terms or break kinsoku.

    Port of the NLH ``breaks`` command. ``lines_or_cues`` is an iterable of
    ``Cue`` or text blocks (``"\n"`` or ASS ``"\N"`` separate lines). A plain
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
    r"""Approximate grapheme count: marks, joiners and newlines are not counted.

    >>> grapheme_count("é\nab")
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
    r"""Serialize cues as SubRip text (millisecond times, UTF-8 friendly).

    SRT has no escape syntax and players (VLC, mpv, YouTube, Premiere) show
    the characters literally, so the text is written **verbatim** by default;
    HTML-escaping would display ``&amp;`` on screen. R23 always escapes
    (``html.escape(quote=False)``) because its SRT feeds an HTML-aware
    importer; pass ``escape=True`` for that behaviour. WebVTT *is* markup, so
    ``to_vtt`` always escapes, exactly like R23.

    >>> print(to_srt([Cue(0, 1.5, "a\nb")]), end="")
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
    r"""Parse SubRip text into cues (BOM, CRLF and missing indexes tolerated).

    >>> parse_srt("1\n00:00:01,000 --> 00:00:02,500\nHi\n", "en")
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
    font=None,
    secondary_font=None,
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
    font_name, secondary_font_name : str, optional
        Font *family names* written into the ASS styles (what the player loads).
    font, secondary_font : str, path or PIL font, optional
        Font *files* (or ``"default"``) used only to measure line heights. With
        a ``font`` the lift of a primary cue over an overlapping secondary cue
        equals the burned-in ``SubtitleLayer`` stack exactly (line height,
        1.1 leading, stroke padding and ``line_gap`` from the real metrics);
        without one a ``1.2 * size`` per row approximation is used.
        ``secondary_font`` defaults to ``font``.

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
    measured = None
    if font is not None:
        sec_font = _load_font(
            secondary_font if secondary_font is not None else font, secondary_px
        )
        extra = int(round(style.plate_padding * scale)) if style.plate_opacity else 0
        measured = (
            sec_font,
            int(round(style.outline_width * scale)),
            int(round(gap)),
            extra,
        )
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
                    if measured is None:
                        height = rows * secondary_px * 1.2 + gap
                    else:
                        height = (
                            _block_height(*measured[:2], rows, measured[3])
                            + measured[2]
                        )
                    lift = max(lift, int(round(height)))
        lines.append(
            f"Dialogue: 0,{_ass_clock(cue.start)},{_ass_clock(cue.end)},"
            f"{'Primary' if is_primary else 'Secondary'},,0,0,{margin + lift},,"
            f"{_ass_text(cue.text)}"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Glossary highlighting


def load_glossary(source):
    """Read glossary terms from the r2b ``glossary.json`` format.

    ``source`` is a path, a JSON string or a dict of categories, each a dict
    ``{zh: en}``. A zh key may hold alternatives separated by ``／``, ``/`` or
    ``、`` (``"李治／唐高宗"``); every alternative and every English value is a
    term. Non-dict categories are ignored.

    Returns
    -------
    tuple of str
        Unique terms, longest first (ties alphabetical), ready for the
        ``highlight`` argument of ``subtitle_layer``.

    Examples
    --------
    >>> load_glossary({"p": {"李治／唐高宗": "Emperor Gaozong", "王皇后": "Wang"}})
    ('Emperor Gaozong', 'Wang', '唐高宗', '王皇后', '李治')
    """
    data = source
    if isinstance(source, Path) or (
        isinstance(source, str) and not source.lstrip().startswith("{")
    ):
        data = Path(source).read_text(encoding="utf-8-sig")
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError("glossary must be a dict of categories")
    terms = set()
    for category in data.values():
        if not isinstance(category, dict):
            continue
        for zh, en in category.items():
            terms.update(t.strip() for t in re.split(r"[／/、]", str(zh)) if t.strip())
            if isinstance(en, str) and en.strip():
                terms.add(en.strip())
    return tuple(sorted(terms, key=lambda t: (-len(t), t)))


def _highlight_pattern(terms):
    """Compile glossary terms: longest first, Latin terms on word boundaries."""
    unique = {str(t).strip() for t in terms if str(t).strip()}
    if not unique:
        return None
    alternatives = [
        (
            rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])"
            if re.search(r"[A-Za-z]", t)
            else re.escape(t)
        )
        for t in sorted(unique, key=lambda t: (-len(t), t))
    ]
    return re.compile("|".join(alternatives))


def _highlight_marks(lines, pattern, cjk):
    """Per-line ``(start, end)`` spans of glossary matches in a wrapped block.

    Matching runs on the joined block, so a term split by a line break is
    found and each piece is marked on its own line.
    """
    if pattern is None:
        return tuple(() for _ in lines)
    joiner = "" if cjk else " "
    joined = joiner.join(lines)
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + len(joiner)
    marks = [[] for _ in lines]
    for match in pattern.finditer(joined):
        for n, line in enumerate(lines):
            a = max(match.start(), starts[n])
            b = min(match.end(), starts[n] + len(line))
            if a < b:
                marks[n].append((a - starts[n], b - starts[n]))
    return tuple(tuple(m) for m in marks)


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


def _block_metrics(font, stroke, extra=0):
    """Return ``(line_height, line_step, pad)`` of one rasterised text block.

    ``extra`` is the plate padding in pixels; it widens the block margin so a
    padded plate is never clipped.
    """
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    return line_h, int(round(line_h * 1.1)), stroke + 2 + extra


def _block_height(font, stroke, rows, extra=0):
    """Pixel height of a ``rows``-line block exactly as ``_raster`` draws it."""
    line_h, step, pad = _block_metrics(font, stroke, extra)
    return step * (rows - 1) + line_h + 2 * pad


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
    highlight : iterable of str, optional
        Glossary terms drawn in ``style.highlight_color`` in both lanes
        (Latin terms on word boundaries, CJK by substring, longest first,
        non-overlapping; a term split by a line break is coloured on both
        pieces). Highlight terms are also ``protected`` from line breaks.
        When the style has no ``highlight_color`` the preset palette role
        ``"accent"`` is used; if that is missing too, ``ValueError`` is raised.
    overflow : {"wrap", "split"}, optional
        ``"wrap"`` (default) re-wraps a cue within ``max_lines`` and raises
        ``LayoutError`` when it cannot fit. ``"split"`` runs ``reflow_cues``
        first, so a long cue becomes several consecutive cues that each fit;
        ``reflow_report`` then lists what was split and how it was timed.
    words : sequence of Word, optional
        ASR word timings for the primary cues, used by ``overflow="split"``.
    cache_size : int, optional
        LRU capacity for composed frames and rendered blocks. A composed
        1080p frame is about 2-7 MB (float32 RGBA of the text box only), so the
        default 16 bounds the layer near 100 MB while still covering sequential
        playback; raise it for scrubbing across many cues.

    Notes
    -----
    With ``style.plate_opacity > 0`` every rendered line gets its own
    rectangle (``plate_color`` at that opacity) covering the line's ink box
    including the outline, grown by the scaled ``plate_padding``; plates sit
    under the outline and fill. Raster and cache keys depend on the layer's
    constant style only, so they need no extra key part.

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
        highlight=(),
        overflow="wrap",
        words=None,
        cache_size=16,
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
        self._font_files = {
            role: (
                str(Path(spec).expanduser()) if isinstance(spec, (str, Path)) else None
            )
            for role, spec in (("primary", font), ("secondary", secondary_font))
        }
        if self._font_files["primary"] == "default":
            self._font_files["primary"] = None
        if self._font_files["secondary"] == "default":
            self._font_files["secondary"] = None
        self._highlight = tuple(dict.fromkeys(str(t) for t in highlight))
        self._protected = tuple(dict.fromkeys((*protected, *self._highlight)))
        self._hl_pattern = _highlight_pattern(self._highlight)
        self._hl_color = None
        if self._hl_pattern is not None:
            color = style.highlight_color or preset.palette.get("accent")
            if color is None:
                raise ValueError(
                    "highlight terms need style.highlight_color or a preset "
                    "palette role 'accent'"
                )
            self._hl_color = color
        self._plate_pad = (
            int(round(s(style.plate_padding))) if style.plate_opacity > 0 else 0
        )
        self._widths = {}
        self._max_width = style.max_width * self._size[0]
        if overflow not in ("wrap", "split"):
            raise ValueError("overflow must be 'wrap' or 'split'")
        self.reflow_report = []
        if overflow == "split":
            split = lambda items, role, timing: reflow_cues(  # noqa: E731
                items,
                self._measure(role),
                self._max_width,
                max_lines=style.max_lines,
                protected=self._protected,
                words=timing,
                report=self.reflow_report,
            )
            cues = split(cues, "primary", words)
            if secondary:
                secondary = split(secondary, "secondary", None)
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
        cache = self._widths.setdefault(role, {})

        def measure(text):
            width = cache.get(text)
            if width is None:
                width = cache[text] = font.getlength(text) + pad
            return width

        return measure

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

    def _check_glyphs(self, cue, role):
        """Raise ``LayoutError`` when the font file lacks a glyph of ``cue``.

        A missing glyph would be drawn as an empty box (豆腐字). The check
        needs fontTools; without it every glyph is assumed present.
        """
        from moviepy.ae.templates.paper import has_glyph

        path = self._font_files[role]
        if path is None:
            return
        missing = "".join(
            c
            for c in dict.fromkeys(cue.text)
            if not c.isspace() and not has_glyph(path, c)
        )
        if missing:
            raise LayoutError(
                f"subtitle font {Path(path).name} has no glyph for {missing!r} "
                f"(cue at {cue.start:.2f} s: {cue.text[:24]!r}); use a font that "
                "covers it, e.g. the full Source Han TC instead of the TW subset"
            )

    def _prepare(self, cues, role):
        cues = sorted(cues, key=lambda c: (c.start, c.end))
        for cue in cues:
            if not isinstance(cue, Cue):
                raise TypeError("cues must be Cue instances")
            self._check_glyphs(cue, role)
        lines = [self._lines_for(c, role) for c in cues]
        return {
            "cues": cues,
            "starts": [c.start for c in cues],
            "lines": lines,
            "marks": [
                _highlight_marks(ln, self._hl_pattern, _is_cjk_lang(c.lang))
                for ln, c in zip(lines, cues)
            ],
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

    def _block(self, role, lines, marks):
        key = (role, lines, marks)
        block = self._blocks.get(key)
        if block is not None:
            self._blocks.move_to_end(key)
            return block
        block = self._raster(role, lines, marks)
        self._blocks[key] = block
        while len(self._blocks) > self._capacity:
            self._blocks.popitem(last=False)
        return block

    def _raster(self, role, lines, marks):
        """Premultiplied RGBA block: plates, then outline, fill and highlights."""
        from PIL import Image, ImageDraw

        self.rasters += 1
        font, stroke = self._fonts[role], self._stroke
        color = (
            self._style.primary_color
            if role == "primary"
            else (self._style.secondary_color)
        )
        line_h, step, pad = _block_metrics(font, stroke, self._plate_pad)
        width = int(math.ceil(max(font.getlength(line) for line in lines))) + 2 * pad
        height = _block_height(font, stroke, len(lines), self._plate_pad)
        outline_mask = Image.new("L", (width, height), 0)
        fill_mask = Image.new("L", (width, height), 0)
        mark_mask = Image.new("L", (width, height), 0)
        d_out, d_fill = ImageDraw.Draw(outline_mask), ImageDraw.Draw(fill_mask)
        d_mark = ImageDraw.Draw(mark_mask)
        plate = np.zeros((height, width), np.float32)
        opacity = self._style.plate_opacity
        for n, line in enumerate(lines):
            x = (width - font.getlength(line)) / 2
            y = pad + n * step
            d_out.text((x, y), line, font=font, fill=255, stroke_width=stroke)
            d_fill.text((x, y), line, font=font, fill=255)
            for a, b in marks[n]:
                px = x + font.getlength(line[:a])
                d_mark.text((px, y), line[a:b], font=font, fill=255)
            if opacity > 0:
                left, top, right, bottom = d_out.textbbox(
                    (x, y), line, font=font, stroke_width=stroke
                )
                p = self._plate_pad
                x0, y0 = max(0, int(left) - p), max(0, int(top) - p)
                x1, y1 = min(width, int(right) + p), min(height, int(bottom) + p)
                plate[y0:y1, x0:x1] = opacity
        oa = np.asarray(outline_mask, np.float32) / 255.0
        fa = np.asarray(fill_mask, np.float32) / 255.0
        ha = np.minimum(np.asarray(mark_mask, np.float32) / 255.0, fa)
        oa = np.maximum(oa, fa)
        out = np.empty((height, width, 4), np.float32)
        fill = np.asarray(color, np.float32) / 255.0
        edge = np.asarray(self._style.outline_color, np.float32) / 255.0
        under = oa * (1.0 - fa)
        rgb = fill * (fa - ha)[..., None] + edge * under[..., None]
        if self._hl_color is not None:
            hl = np.asarray(self._hl_color, np.float32) / 255.0
            rgb = rgb + hl * ha[..., None]
        text_a = fa + under
        out[..., :3] = rgb
        out[..., 3] = text_a
        if opacity > 0:
            # Premultiplied "text over plate": T + P * (1 - alpha_T).
            plate_rgb = np.asarray(self._style.plate_color, np.float32) / 255.0
            behind = (plate * (1.0 - text_a))[..., None]
            out[..., :3] += plate_rgb * behind
            out[..., 3] += behind[..., 0]
        out.setflags(write=False)
        return out

    def _compose(self, p_lines, s_lines):
        blocks = []  # top to bottom
        if p_lines is not None:
            blocks.append(self._block("primary", *p_lines))
        if s_lines is not None:
            blocks.append(self._block("secondary", *s_lines))
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
            lane = self._lanes[role]
            lines.append(None if i is None else (lane["lines"][i], lane["marks"][i]))
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
    cues,
    preset,
    *,
    secondary=None,
    font=None,
    secondary_font=None,
    highlight=(),
    **kw,
):
    """Build a ``SubtitleLayer`` for a whole timeline.

    ``highlight`` lists glossary terms coloured with the style's
    ``highlight_color`` in both lanes (see ``SubtitleLayer``).

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
        highlight=highlight,
        **kw,
    )


def burn_subtitles(
    clip_or_comp,
    cues,
    preset,
    *,
    secondary=None,
    font=None,
    secondary_font=None,
    highlight=(),
    **kw,
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
        highlight=highlight,
        size=comp.size,
        **kw,
    )
    comp.add_layer(layer)
    return comp
