"""Episode intro/outro title cards built from AE layers.

A simplified port of R23 ``ae_title_base.make_title`` and
``render_sequence.build_sequence``: a brand line, an accent rule, a fitted title
and a hook (subtitle) are rasterised with Pillow (no SVG) into straight-alpha
footage layers, then staggered in with fade plus slide ``Property`` keyframes
(ease-out cubic) and faded out at the end. Every layer and ``Property`` is
returned on the ``Composition`` so callers can re-keyframe them.

Examples
--------
>>> from moviepy.ae.templates.title_card import fit_lines, readable_duration
>>> fit_lines("Hello", None, 400, 40, 20)
(['Hello'], 40)
>>> duration, report = readable_duration(1.0, "x" * 30, 1.7, 0.35, 24)
>>> round(duration, 3), report["automatically_extended"]
(7.083, True)
"""

import math
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.ae.composition import Composition
from moviepy.ae.layers import AVLayer, SolidLayer
from moviepy.ae.properties.easing import Ease
from moviepy.ae.properties.keyframe import Keyframe
from moviepy.ae.properties.property import Property
from moviepy.ae.text.clusters import clusters
from moviepy.ae.transform import Transform


NO_LINE_START = set("、。，．・：；？！)]}）】》〉」』〕〗〙〛…")
NO_LINE_END = set("([{（【《〈「『〔〖〘〚")
_ASCII_NO_START = set(",.;:!?%")

LAYOUTS = ("right_column", "center")
ROLES = ("intro", "outro")

#: R23 tokens.json (1080p reference pixels), by segment role.
TOKENS = {
    "intro": {
        "title_size": 124,
        "title_min_size": 86,
        "title_tracking": 5,
        "subtitle_gap": 64,
    },
    "outro": {
        "title_size": 72,
        "title_min_size": 52,
        "title_tracking": 3,
        "subtitle_gap": 54,
    },
    "common": {
        "brand_size": 38,
        "brand_tracking": 5,
        "title_leading": 1.3,
        "subtitle_size": 36,
        "subtitle_min_size": 30,
        "subtitle_tracking": 2.5,
        "subtitle_leading": 1.45,
        "brand_rule_gap": 30,
        "rule_title_gap": 40,
        "rule_width": 96,
        "rule_height": 3,
        "column_x": 770 / 1920,
        "column_width": 988 / 1920,
        "center_width": 0.6,
        "title_rise": 18,
        "subtitle_rise": 10,
        "brand_rise": 8,
        "characters_per_second": 6.0,
        "minimum_hold": 2.5,
    },
}

_OUT_CUBIC = Ease.bezier(0.215, 0.61, 0.355, 1.0)


# --------------------------------------------------------------------------- #
# text fitting
# --------------------------------------------------------------------------- #


def _font(font_path, size, index=0):
    if font_path is None:
        return ImageFont.load_default(size=int(size))
    return ImageFont.truetype(str(font_path), int(size), index=index)


def _is_word_char(char):
    return (char.isalnum() and ord(char) < 0x2E80) or char in "'’-_"


class _Measure:
    """Cached advance width of a line with letter tracking."""

    def __init__(self, font_path, size, tracking, index=0):
        self.font = _font(font_path, size, index)
        self.tracking = tracking
        self._cache = {}

    def __call__(self, line):
        if line not in self._cache:
            count = len(clusters(line))
            self._cache[line] = self.font.getlength(line) + max(0, count - 1) * (
                self.tracking
            )
        return self._cache[line]


def _break_points(units, text, keep_together):
    """Return unit indices where a line may end (exclusive upper bound)."""
    offsets = [0]
    for unit in units:
        offsets.append(offsets[-1] + len(unit))
    protected = []
    for term in keep_together:
        start = text.find(term)
        while term and start != -1:
            protected.append((start, start + len(term)))
            start = text.find(term, start + 1)
    points = []
    for k in range(1, len(units)):
        left, right = units[k - 1], units[k]
        if right == " ":
            continue
        if right[0] in NO_LINE_START or right[0] in _ASCII_NO_START:
            continue
        if left != " ":
            if left[-1] in NO_LINE_END:
                continue
            if _is_word_char(left[-1]) and _is_word_char(right[0]):
                continue
            if left[-1].isascii() and right[0].isascii() and left[-1] not in "-/":
                continue  # ASCII text only breaks at spaces and after - or /
        if any(a < offsets[k] < b for a, b in protected):
            continue
        points.append(k)
    return points


def _greedy(units, points, measure, limit):
    """Pack lines no wider than ``limit``; return the lines or None."""
    lines, start, allowed = [], 0, points + [len(units)]
    while start < len(units):
        best = None
        for end in allowed:
            if end <= start:
                continue
            line = "".join(units[start:end]).strip(" ")
            if line and measure(line) <= limit:
                best = end
            elif best is not None:
                break
        if best is None:
            return None
        lines.append("".join(units[start:best]).strip(" "))
        start = best
        while start < len(units) and units[start] == " ":
            start += 1
    return lines


def _wrap(text, measure, max_width, n_lines, keep_together):
    units = clusters(text)
    points = _break_points(units, text, keep_together)
    best = _greedy(units, points, measure, max_width)
    if best is None or len(best) > n_lines:
        return None
    low, high = 0.0, float(max_width)
    for _ in range(24):
        mid = (low + high) / 2
        trial = _greedy(units, points, measure, mid)
        if trial is not None and len(trial) <= n_lines:
            best, high = trial, mid
        else:
            low = mid
    return best


def fit_lines(
    text,
    font_path,
    max_width,
    size,
    min_size,
    tracking=0.0,
    *,
    keep_together=(),
    max_lines=2,
    font_index=0,
):
    """Fit ``text`` into ``max_width`` pixels, CJK-aware.

    The size shrinks from ``size`` towards ``min_size`` *before* any wrapping
    is tried; wrapping (at most ``max_lines`` lines, balanced) only happens at
    ``min_size``. Breaks obey the R23 ``NO_LINE_START``/``NO_LINE_END``
    punctuation rules, ASCII words break only at spaces, and ``keep_together``
    terms are never split. Explicit newlines are kept as given.

    Parameters
    ----------
    text : str
    font_path : str or None
        Font file; ``None`` uses Pillow's default (Latin-only) font.
    max_width : float
        Available line width in pixels, tracking included.
    size, min_size : int
        Preferred and smallest font size in pixels.
    tracking : float, optional
        Extra pixels between clusters.
    keep_together : sequence of str, optional
        Terms that must stay on one line.
    max_lines : int, optional
    font_index : int, optional
        Face index for ``.ttc`` collections.

    Returns
    -------
    tuple
        ``(lines, size)``.

    Raises
    ------
    ValueError
        If the text cannot fit at ``min_size`` in ``max_lines`` lines. Nothing
        is truncated.

    Examples
    --------
    >>> fit_lines("a b", None, 1000, 30, 20)
    (['a b'], 30)
    >>> lines, size = fit_lines("word " * 12, None, 220, 30, 20, max_lines=3)
    >>> size, len(lines) > 1
    (20, True)
    """
    if not text:
        return [], int(size)
    size, min_size = int(size), int(min_size)
    if min_size > size or min_size < 1:
        raise ValueError("min_size must be in [1, size]")
    explicit = text.split("\n")
    if len(explicit) > max_lines:
        raise ValueError(f"text has more than {max_lines} lines")
    for current in range(size, min_size - 1, -1):
        measure = _Measure(font_path, current, tracking, font_index)
        if all(measure(line) <= max_width for line in explicit):
            return explicit, current
    measure = _Measure(font_path, min_size, tracking, font_index)
    if len(explicit) == 1:
        for n_lines in range(2, max_lines + 1):
            lines = _wrap(text, measure, max_width, n_lines, tuple(keep_together))
            if lines is not None:
                return lines, min_size
    raise ValueError(
        f"text overflow: cannot fit the complete text at {min_size}px in "
        f"{max_lines} lines; shorten the copy or widen the column"
    )


def readable_duration(
    requested,
    text,
    entrance,
    fade_out,
    fps,
    *,
    characters_per_second=6.0,
    minimum_hold=2.5,
    auto_extend=True,
):
    """Return ``(duration, report)`` long enough to read ``text`` (R23 rule).

    Letters and digits are counted; the hold is
    ``max(minimum_hold, characters / characters_per_second)`` and the card needs
    ``entrance + hold + fade_out`` seconds. Long copy extends the duration (on
    the ``fps`` frame grid) instead of being truncated.

    Raises
    ------
    ValueError
        If ``auto_extend`` is false and ``requested`` is too short.

    Examples
    --------
    >>> readable_duration(5.0, "short", 1.7, 0.35, 24)[0]
    5.0
    """
    characters = sum(unicodedata.category(c)[0] in ("L", "N") for c in text)
    hold = max(minimum_hold, characters / characters_per_second)
    required = entrance + hold + fade_out
    if requested + 1e-9 < required and not auto_extend:
        raise ValueError(f"reading time insufficient: need at least {required:.2f}s")
    duration = math.ceil(max(requested, required) * fps - 1e-8) / fps
    return duration, {
        "requested_seconds": requested,
        "actual_seconds": duration,
        "reading_characters": characters,
        "required_hold_seconds": hold,
        "available_hold_seconds": duration - entrance - fade_out,
        "automatically_extended": duration > requested + 1e-9,
    }


# --------------------------------------------------------------------------- #
# specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TitleCardSpec:
    """Copy, layout and timing of one title card.

    Times are seconds; the defaults are R23's entrance (brand 0.2/0.55, title
    0.55/0.85, hook 1.05/0.65) with a 0.35 s fade-out. ``duration`` is a
    request: ``readable_duration`` may extend it when ``auto_extend`` is true.

    Parameters
    ----------
    brand, title, subtitle : str
        ``subtitle`` is the hook line(s); ``hook`` is an alias.
    layout : {"right_column", "center"}
    role : {"intro", "outro"}
        Chooses R23's intro or smaller outro title tokens.
    title_keep_together, subtitle_keep_together : tuple of str
        Terms never split across lines.

    Examples
    --------
    >>> spec = TitleCardSpec(title="Night")
    >>> spec.layout, spec.duration, spec.hook
    ('right_column', 5.0, '')
    """

    title: str
    brand: str = ""
    subtitle: str = ""
    layout: str = "right_column"
    role: str = "intro"
    duration: float = 5.0
    brand_start: float = 0.2
    brand_duration: float = 0.55
    title_start: float = 0.55
    title_duration: float = 0.85
    subtitle_start: float = 1.05
    subtitle_duration: float = 0.65
    fade_out: float = 0.35
    rule: bool = True
    auto_extend: bool = True
    title_keep_together: tuple = ()
    subtitle_keep_together: tuple = ()

    def __post_init__(self):
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        for name in (
            "duration",
            "brand_start",
            "brand_duration",
            "title_start",
            "title_duration",
            "subtitle_start",
            "subtitle_duration",
            "fade_out",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        for name in ("title_keep_together", "subtitle_keep_together"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def hook(self):
        """Alias of ``subtitle``."""
        return self.subtitle

    @property
    def entrance_end(self):
        """Time at which the last entrance finishes."""
        ends = [self.title_start + self.title_duration]
        if self.brand:
            ends.append(self.brand_start + self.brand_duration)
        if self.subtitle:
            ends.append(self.subtitle_start + self.subtitle_duration)
        return max(ends)


# --------------------------------------------------------------------------- #
# rasterising and layers
# --------------------------------------------------------------------------- #


def _resolve_font(preset, fonts, role):
    """Return a font path, or None when the caller explicitly chose Pillow's."""
    if fonts is not None and role in fonts:
        value = fonts[role]
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{role} font not found: {path}")
        return str(path)
    return preset.font(role)


def _check_glyphs(text, font_path, label):
    font = _font(font_path, 24)
    missing = font.getmask("￿")
    blank = bytes(missing)
    if not any(blank):
        return
    for char in dict.fromkeys(text):
        if char.isspace() or char == "\n":
            continue
        mask = font.getmask(char)
        if mask.size == missing.size and bytes(mask) == blank:
            raise ValueError(f"{label}: font has no glyph for {char!r}")


def _render_text(lines, font_path, size, tracking, leading, color, align, pad):
    """Rasterise ``lines`` to a straight-alpha RGB + alpha pair."""
    font = _font(font_path, size)
    ascent, descent = font.getmetrics()
    widths = []
    for line in lines:
        count = len(clusters(line))
        widths.append(font.getlength(line) + max(0, count - 1) * tracking)
    text_w = max(widths)
    pitch = size * leading
    text_h = (len(lines) - 1) * pitch + ascent + descent
    width, height = int(math.ceil(text_w)) + 2 * pad, int(math.ceil(text_h)) + 2 * pad
    alpha = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(alpha)
    for row, line in enumerate(lines):
        x0 = pad + ((text_w - widths[row]) / 2 if align == "center" else 0.0)
        baseline = pad + ascent + row * pitch
        units = clusters(line)
        prefix = ""
        for index, unit in enumerate(units):
            x = x0 + font.getlength(prefix) + index * tracking
            draw.text((x, baseline), unit, font=font, fill=255, anchor="ls")
            prefix += unit
    mask = np.asarray(alpha, dtype=np.float64) / 255.0
    rgb = np.empty((height, width, 3), np.uint8)
    rgb[:] = tuple(int(c) for c in color)
    return rgb, mask, (text_w, text_h)


def _text_layer(name, lines, font_path, size, tracking, leading, color, align, pad):
    from moviepy import ImageClip

    rgb, mask, box = _render_text(
        lines, font_path, size, tracking, leading, color, align, pad
    )
    clip = ImageClip(rgb).with_mask(ImageClip(mask, is_mask=True))
    layer = AVLayer(clip, name)
    ys, xs = np.nonzero(mask > 0.02)
    ink = (
        (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if xs.size
        else (pad, pad, pad, pad)
    )
    return layer, {"size": rgb.shape[1::-1], "text_box": box, "ink": ink}


def _keys(items):
    """Build strictly increasing keyframes, dropping duplicate times."""
    result, last = [], -math.inf
    for time, value, ease in items:
        if time <= last + 1e-9:
            continue
        result.append(Keyframe(time, value, out_ease=ease))
        last = time
    return result


def _animate(layer, anchor, position, rise, start, length, duration, fade_out):
    """Install fade + slide-in and fade-out keyframes on ``layer``."""
    x, y = position
    end = start + length
    out_start = max(end, duration - fade_out)
    opacity = _keys(
        [
            (0.0, 0.0, None),
            (start, 0.0, _OUT_CUBIC),
            (end, 100.0, None),
            (out_start, 100.0, Ease.ease_in_out()),
            (duration, 0.0, None),
        ]
    )
    slide = _keys(
        [
            (0.0, (x, y + rise), None),
            (start, (x, y + rise), _OUT_CUBIC),
            (end, (x, y), None),
        ]
    )
    layer.transform = Transform(
        anchor_point=anchor,
        position=Property(slide, value_type="vec2", spatial=False),
        opacity=Property(opacity, value_type="float"),
    )
    layer.out_point = duration


def build_title_card(spec, preset, background=None, *, fonts=None, transparent=False):
    """Build an intro/outro ``Composition`` from ``spec`` and ``preset``.

    Parameters
    ----------
    spec : TitleCardSpec
    preset : ChannelPreset
        Canvas, fps, palette, ``safe_margin`` (pixels at ``preset.size``) and
        fonts. R23 sizes (124/86 title, 38 brand, ...) are scaled by
        ``preset.scale``.
    background : MediaBackground, optional
        Fresh footage and overlay layers are made for this composition.
    fonts : dict, optional
        ``{"title": path, "body": path}`` overriding the preset; a value of
        ``None`` explicitly selects Pillow's default (Latin-only) font.
    transparent : bool, optional
        Make the composition carry its alpha as a mask (useful over footage).

    Returns
    -------
    Composition
        Layers are named ``Brand``, ``Rule``, ``Title`` and ``Subtitle``;
        ``comp.title_report`` holds fitted sizes, lines, ink boxes and the
        reading-time record.

    Examples
    --------
    >>> from moviepy.ae.templates.presets import get_preset
    >>> preset = get_preset("nightlamp_story")
    >>> comp = build_title_card(
    ...     TitleCardSpec(title="Night", brand="Lamp"), preset,
    ...     fonts={"title": None, "body": None}, transparent=True)
    >>> comp.size, [layer.name for layer in comp.layers]
    ((1920, 1080), ['Title', 'Rule', 'Brand'])
    """
    size_w, size_h = preset.size
    s = preset.scale
    margin_x, margin_y = preset.safe_margin
    token = {**TOKENS["common"], **TOKENS[spec.role]}
    center = spec.layout == "center"
    if center:
        col_w = min(size_w - 2 * margin_x, size_w * token["center_width"])
        col_x = size_w / 2
    else:
        left = max(size_w * token["column_x"], margin_x)
        col_w = min(size_w * token["column_width"], size_w - margin_x - left)
        col_x = left
    if col_w <= 0:
        raise ValueError("safe margins leave no room for the text column")

    title_font = _resolve_font(preset, fonts, "title")
    body_font = _resolve_font(preset, fonts, "body")
    _check_glyphs(spec.title, title_font, "title")
    _check_glyphs(spec.brand + spec.subtitle, body_font, "body text")

    def px(value):
        return value * s

    title_lines, title_size = fit_lines(
        spec.title,
        title_font,
        col_w,
        round(px(token["title_size"])),
        round(px(token["title_min_size"])),
        px(token["title_tracking"]),
        keep_together=spec.title_keep_together,
    )
    sub_lines, sub_size = [], 0
    if spec.subtitle:
        sub_lines, sub_size = fit_lines(
            spec.subtitle,
            body_font,
            col_w,
            round(px(token["subtitle_size"])),
            round(px(token["subtitle_min_size"])),
            px(token["subtitle_tracking"]),
            keep_together=spec.subtitle_keep_together,
            max_lines=3,
        )
    align = "center" if center else "left"
    pad = max(2, round(px(8)))
    ink = tuple(preset.color("ink", (243, 236, 220)))
    secondary = tuple(preset.color("secondary_ink", (232, 223, 204)))
    accent = tuple(preset.color("lamp", (217, 164, 95)))

    entries = []  # (name, layer, info, gap_before, timing, rise)
    if spec.brand:
        layer, info = _text_layer(
            "Brand",
            [spec.brand],
            body_font,
            round(px(token["brand_size"])),
            px(token["brand_tracking"]),
            1.0,
            secondary,
            align,
            pad,
        )
        entries.append(
            (
                "Brand",
                layer,
                info,
                0,
                (spec.brand_start, spec.brand_duration),
                px(token["brand_rise"]),
            )
        )
    rule_w, rule_h = (
        round(px(token["rule_width"])),
        max(1, round(px(token["rule_height"]))),
    )
    if spec.rule:
        gap = px(token["brand_rule_gap"]) if spec.brand else 0
        rule = SolidLayer("Rule", color=accent, size=(rule_w, rule_h))
        info = {
            "size": (rule_w, rule_h),
            "text_box": (rule_w, rule_h),
            "ink": (0, 0, rule_w, rule_h),
            "solid": True,
        }
        entries.append(
            ("Rule", rule, info, gap, (spec.brand_start, spec.brand_duration), 0.0)
        )
    layer, info = _text_layer(
        "Title",
        title_lines,
        title_font,
        title_size,
        px(token["title_tracking"]),
        token["title_leading"],
        ink,
        align,
        pad,
    )
    entries.append(
        (
            "Title",
            layer,
            info,
            px(token["rule_title_gap"]) if spec.rule else 0,
            (spec.title_start, spec.title_duration),
            px(token["title_rise"]),
        )
    )
    if sub_lines:
        layer, info = _text_layer(
            "Subtitle",
            sub_lines,
            body_font,
            sub_size,
            px(token["subtitle_tracking"]),
            token["subtitle_leading"],
            secondary,
            align,
            pad,
        )
        entries.append(
            (
                "Subtitle",
                layer,
                info,
                px(token["subtitle_gap"]),
                (spec.subtitle_start, spec.subtitle_duration),
                px(token["subtitle_rise"]),
            )
        )

    heights = [e[2]["text_box"][1] for e in entries]
    total = sum(heights) + sum(e[3] for e in entries)
    available = size_h - 2 * margin_y
    if total > available:
        raise ValueError(
            f"title card needs {total:.0f}px of height but the safe area has "
            f"{available:.0f}px"
        )

    duration, reading = readable_duration(
        spec.duration,
        spec.title + spec.subtitle,
        spec.entrance_end,
        spec.fade_out,
        preset.fps,
        characters_per_second=token["characters_per_second"],
        minimum_hold=token["minimum_hold"],
        auto_extend=spec.auto_extend,
    )
    comp = Composition(
        preset.size,
        preset.fps,
        duration,
        name=f"{spec.role.title()} title card",
        bg_color=(0, 0, 0),
        transparent=transparent,
    )
    if background is not None:
        footage, overlay = background.make_layers(preset.size, duration)
        comp.add_layer(footage)
        comp.add_layer(overlay)

    cursor = margin_y + (available - total) / 2
    report_boxes = {}
    for name, layer, info, gap, (start, length), rise in entries:
        cursor += gap
        tw, th = info["text_box"]
        solid = info.get("solid", False)
        if solid:
            anchor = (rule_w / 2, 0) if center else (0, 0)
        else:
            anchor = (pad + tw / 2, pad) if center else (pad, pad)
        position = (col_x, cursor)
        _animate(layer, anchor, position, rise, start, length, duration, spec.fade_out)
        comp.add_layer(layer)
        ix0, iy0, ix1, iy1 = info["ink"]
        box = (
            position[0] - anchor[0] + ix0,
            position[1] - anchor[1] + iy0,
            position[0] - anchor[0] + ix1,
            position[1] - anchor[1] + iy1,
        )
        if (
            box[0] < margin_x - 1
            or box[2] > size_w - margin_x + 1
            or box[1] < margin_y - 1
            or box[3] > size_h - margin_y + 1
        ):
            raise ValueError(f"{name} falls outside the preset safe margin")
        report_boxes[name] = tuple(float(v) for v in box)
        cursor += th
    comp.title_report = {
        "role": spec.role,
        "layout": spec.layout,
        "title_lines": title_lines,
        "title_size": title_size,
        "subtitle_lines": sub_lines,
        "subtitle_size": sub_size,
        "column": (col_x, col_w),
        "ink_boxes": report_boxes,
        "reading": reading,
        "fonts": {"title": title_font, "body": body_font},
    }
    return comp


def build_bookends(
    intro_spec, outro_spec, preset, background=None, *, fonts=None, transparent=False
):
    """Build the ``(intro, outro)`` compositions sharing one background.

    The specs' ``role`` is set to ``"intro"``/``"outro"`` so the outro uses
    R23's smaller title tokens; the background overlay opacity ``Property`` is
    shared, so one edit changes both.

    Examples
    --------
    >>> from moviepy.ae.templates.presets import get_preset
    >>> intro, outro = build_bookends(
    ...     TitleCardSpec(title="Open"), TitleCardSpec(title="Close"),
    ...     get_preset("nightlamp_story"), fonts={"title": None, "body": None})
    >>> intro.title_report["role"], outro.title_report["role"]
    ('intro', 'outro')
    """
    intro = build_title_card(
        replace(intro_spec, role="intro"),
        preset,
        background,
        fonts=fonts,
        transparent=transparent,
    )
    outro = build_title_card(
        replace(outro_spec, role="outro"),
        preset,
        background,
        fonts=fonts,
        transparent=transparent,
    )
    return intro, outro
