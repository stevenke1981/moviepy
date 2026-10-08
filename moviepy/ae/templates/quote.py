"""Vertical right-to-left historical quotation on bamboo slips or a paper scroll.

Port of ``nlh_motion.py quote`` onto AE primitives. Bamboo slips are one
``AVLayer`` each (slide plus fade keyframes, cord segments baked in); the paper
scroll is a width-cached layer unrolled right to left between two roller
layers whose position Property shares the sheet's easing. Characters are written
top to bottom, right column to left, through cached per-column ink states.

Examples
--------
>>> from moviepy.ae.templates.quote import resolve_medium, split_columns
>>> resolve_medium(dynasty="西漢")[0], resolve_medium(dynasty="唐")[0]
('bamboo', 'paper')
>>> [''.join(c) for c in split_columns("天地玄黃，宇宙洪荒", 4)]
['天地玄黃，', '宇宙洪荒']
"""

import math
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from moviepy.ae import Composition
from moviepy.ae.layers.solid import SolidLayer
from moviepy.ae.properties import Keyframe, Property
from moviepy.ae.properties.easing import Ease
from moviepy.ae.templates.paper import (
    CachedAVLayer,
    alpha_over,
    cached_rgba_clip,
    drop_shadow,
    ease_out_cubic,
    has_glyph,
    ink_bleed,
    load_font,
    mix_color,
    paper_texture,
    rect_shadow,
    resize_straight,
    rgba_still,
    seal_stamp,
)
from moviepy.ae.templates.presets import ChannelPreset, get_preset
from moviepy.ae.transform import Transform


__all__ = [
    "resolve_medium",
    "split_columns",
    "split_quote_pages",
    "vertical_form",
    "vertical_quote",
]

# --------------------------------------------------------------------------- #
# Medium: bamboo slips before paper, paper handscroll after (MEDIUM.md).
# --------------------------------------------------------------------------- #

_BAMBOO = [
    "夏", "商", "殷", "周", "西周", "東周", "春秋", "戰國", "先秦", "秦",
    "楚漢", "漢", "西漢", "新", "新莽", "東漢",
]  # fmt: skip
_PAPER = [
    "三國", "魏", "曹魏", "蜀", "蜀漢", "吳", "東吳", "孫吳", "晉", "西晉", "東晉",
    "十六國", "南北朝", "南朝", "北朝", "劉宋", "南齊", "齊", "梁", "陳", "北魏",
    "東魏", "西魏", "北齊", "北周", "隋", "唐", "武周", "五代", "十國", "後梁",
    "後唐", "後晉", "後漢", "後周", "遼", "宋", "北宋", "南宋", "金", "西夏", "元",
    "明", "南明", "清", "民國",
]  # fmt: skip
_S2T = str.maketrans("东战国晋辽齐陈后汉吴楚", "東戰國晉遼齊陳後漢吳楚")
PAPER_YEAR = 220  # the Eastern Han ended in 220: this year and earlier -> bamboo


def resolve_medium(dynasty=None, year=None, medium="auto"):
    """Decide ``"bamboo"`` or ``"paper"`` with the channel rule.

    Eras before paper circulated (through the Eastern Han, 220 CE) use bamboo
    slips; from the Three Kingdoms on, a paper handscroll. The era is the
    story's, not the source book's. An explicit ``medium`` wins, then ``year``
    (``<= 220`` is bamboo), then ``dynasty`` (simplified forms accepted;
    ``後漢`` is the Five Dynasties and means paper, ``東漢`` means bamboo).

    Parameters
    ----------
    dynasty : str, optional
        Dynasty or era name such as ``"西漢"`` or ``"明朝"``.
    year : int, optional
        Common-era year of the story; negative for BCE.
    medium : {"auto", "bamboo", "paper"}, optional
        Explicit override.

    Returns
    -------
    tuple of str
        ``(medium, reason)``.

    Raises
    ------
    ValueError
        If nothing decides the medium.

    Examples
    --------
    >>> resolve_medium(dynasty="東漢")[0], resolve_medium(dynasty="後漢")[0]
    ('bamboo', 'paper')
    >>> resolve_medium(year=221)[0], resolve_medium(medium="bamboo")[1]
    ('paper', 'explicit medium')
    """
    if medium in ("bamboo", "paper"):
        return medium, "explicit medium"
    if medium != "auto":
        raise ValueError("medium must be 'auto', 'bamboo' or 'paper'")
    if year is not None:
        kind = "bamboo" if int(year) <= PAPER_YEAR else "paper"
        return kind, f"year {year} vs threshold {PAPER_YEAR}"
    if dynasty:
        name = dynasty.strip().translate(_S2T)
        name = re.sub(r"(王朝|朝|代|時期|時代)$", "", name)
        if name in _PAPER:
            return "paper", f"dynasty {dynasty}"
        if name in _BAMBOO:
            return "bamboo", f"dynasty {dynasty}"
        for era in sorted(_BAMBOO + _PAPER, key=len, reverse=True):
            if name.startswith(era):
                return ("bamboo" if era in _BAMBOO else "paper"), (
                    f"dynasty {dynasty} ~ {era}"
                )
    raise ValueError(
        "cannot decide the medium: give dynasty (e.g. 西漢, 明), year, "
        "or medium='bamboo'|'paper'"
    )


# --------------------------------------------------------------------------- #
# Vertical typesetting.
# --------------------------------------------------------------------------- #

VMAP = {
    "「": "﹁", "」": "﹂", "『": "﹃", "』": "﹄", "（": "︵", "）": "︶",
    "(": "︵", ")": "︶", "《": "︽", "》": "︾", "〈": "︿", "〉": "﹀",
    "【": "︻", "】": "︼", "〔": "︹", "〕": "︺", "｛": "︷", "｝": "︸",
    "—": "︱", "─": "︱", "－": "︱", "-": "︱", "…": "︙", "～": "︴",
    "·": "・", "‧": "・", "．": "・",
}  # fmt: skip
PUNCT_CENTER = set("，。、；：！？,.;:!?・")
NO_LINE_START = set("，。、；：！？」』）》〉】〕﹂﹄︶︾﹀︼︺")
SENTENCE_END = set("。！？；")


def vertical_form(char):
    """Return the vertical presentation form of ``char`` (or ``char`` itself).

    Examples
    --------
    >>> ''.join(vertical_form(c) for c in "「《史記》」")
    '﹁︽史記︾﹂'
    """
    return VMAP.get(char, char)


def _vchar(char, fpath):
    """Return ``(glyph, rotate_90, center)`` for drawing ``char`` in a column."""
    if char in VMAP:
        glyph = VMAP[char]
        if has_glyph(fpath, glyph):
            return glyph, False, True
        return char, True, True
    return char, False, char in PUNCT_CENTER


def split_columns(text, per_column):
    """Split ``text`` into columns of at most about ``per_column`` characters.

    ``\\n`` or ``|`` force a column break. Closing punctuation never starts a
    column; it is pulled onto the previous one (up to two extra characters).
    Columns are listed in reading order, i.e. right to left on screen.

    Examples
    --------
    >>> [''.join(c) for c in split_columns("甲乙丙丁，戊己|庚", 4)]
    ['甲乙丙丁，', '戊己', '庚']
    """
    if per_column < 1:
        raise ValueError("per_column must be at least 1")
    columns = []
    for para in re.split(r"\n|\|", text.replace("\\n", "\n")):
        chars = list(para.strip())
        while chars:
            take = min(per_column, len(chars))
            while (
                take < len(chars)
                and chars[take] in NO_LINE_START
                and take < per_column + 2
            ):
                take += 1
            if take < len(chars) and chars[take] in NO_LINE_START:
                # More closers than the pull-in allowance: hand the trailing
                # run to the next column's predecessor instead, or take the whole
                # run when the column holds nothing else.
                back = take
                while back > 1 and chars[back - 1] in NO_LINE_START:
                    back -= 1
                if back > 1:
                    take = back - 1  # the char before the run moves down with it
                else:
                    while take < len(chars) and chars[take] in NO_LINE_START:
                        take += 1
            columns.append(chars[:take])
            chars = chars[take:]
    return columns


def split_quote_pages(text, per_column=10, max_columns=12):
    """Split a long quotation into pages of at most ``max_columns`` columns.

    A page break prefers the end of a sentence (。！？；) among the last three
    columns of a page. Pages are returned as text with ``\\n`` between columns,
    so each one can be passed straight back to ``vertical_quote``.

    Examples
    --------
    >>> split_quote_pages("甲乙。丙丁。戊己。庚辛。", per_column=3, max_columns=2)
    ['甲乙。\\n丙丁。', '戊己。\\n庚辛。']
    """
    if max_columns < 1:
        raise ValueError("max_columns must be at least 1")
    columns = [list(c) for c in split_columns(text, per_column)]
    pages = []
    while columns:
        n = min(max_columns, len(columns))
        if len(columns) > n:
            for k in range(n, max(0, n - 3), -1):
                if columns[k - 1][-1] in SENTENCE_END:
                    n = k
                    break
        pages.append("\n".join("".join(c) for c in columns[:n]))
        columns = columns[n:]
    return pages


def _render_column(chars, fpath, fsize, cell, color):
    """Render one vertical column as straight RGBA ``uint8`` ``(cell*n, cell)``."""
    ss = 2
    n = len(chars)
    img = Image.new("RGBA", (cell * ss, max(1, cell * n) * ss), (*color, 0))
    draw = ImageDraw.Draw(img)
    font = load_font(fpath, fsize * ss)
    for i, ch in enumerate(chars):
        cx, cy = cell * ss / 2, (i + 0.5) * cell * ss
        glyph, rotate, center = _vchar(ch, fpath)
        if rotate:
            tile = Image.new("RGBA", (cell * ss, cell * ss), (*color, 0))
            ImageDraw.Draw(tile).text(
                (cell * ss / 2, cell * ss / 2),
                glyph,
                font=font,
                fill=(*color, 255),
                anchor="mm",
            )
            tile = tile.rotate(-90, resample=Image.BICUBIC)
            img.alpha_composite(
                tile, (int(cx - tile.width / 2), int(cy - tile.height / 2))
            )
        elif center:
            box = draw.textbbox((0, 0), glyph, font=font)
            draw.text(
                (cx - (box[0] + box[2]) / 2, cy - (box[1] + box[3]) / 2),
                glyph,
                font=font,
                fill=(*color, 255),
            )
        else:
            draw.text((cx, cy), glyph, font=font, fill=(*color, 255), anchor="mm")
    small = cv2.resize(
        np.asarray(img), (cell, max(1, cell * n)), interpolation=cv2.INTER_AREA
    )
    return small


# --------------------------------------------------------------------------- #
# Bamboo slip and roller.
# --------------------------------------------------------------------------- #


def _noise(rng, shape, size, interp=cv2.INTER_CUBIC):
    base = rng.normal(0, 1, shape).astype(np.float32)
    return cv2.resize(base, size, interpolation=interp)


def _bamboo_slip(w, h, base_rgb, seed):
    """Return one slip (straight RGBA): cylinder shading, fibers, dark ends."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base_rgb, np.float32) * rng.uniform(0.9, 1.06)
    x = np.linspace(0, 1, w, dtype=np.float32)
    shade = 0.8 + 0.22 * np.clip(np.sin(np.pi * x), 0, 1) ** 0.6
    fib = _noise(rng, (1, w), (w, h), cv2.INTER_LINEAR)
    fib2 = _noise(rng, (h // 60 + 2, w), (w, h))
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    ends = 1 - 0.10 * (np.exp(-y * 30) + np.exp(-(1 - y) * 30))
    stain = _noise(rng, (6, 3), (w, h))
    value = (
        base[None, None, :] * (shade[None, :, None] * ends[..., None])
        + (fib * 7 + fib2 * 4 - np.abs(stain) * 9)[..., None]
    )
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.clip(value, 0, 255).astype(np.uint8)
    mask = Image.new("L", (w * 2, h * 2), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, w * 2 - 1, h * 2 - 1], radius=int(w * 0.5), fill=255
    )
    out[..., 3] = np.asarray(mask.resize((w, h), Image.LANCZOS))
    return out


def _cord_segment(width, height, y, thick, color, knots):
    """Return a cord crossing ``width`` px at row ``y`` (straight RGBA)."""
    ss = 2
    img = Image.new("RGBA", (width * ss, height * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pts = [
        (xx * ss, (y + math.sin(xx * 0.05) * 0.8) * ss) for xx in range(0, width + 6, 6)
    ]
    draw.line(pts, fill=(*color, 255), width=int(thick * ss), joint="curve")
    dark = tuple(max(0, c - d) for c, d in zip(color, (34, 28, 20)))
    for xx in np.arange(0, width, thick * 0.9):
        yy = y + math.sin(xx * 0.05) * 0.8
        draw.line(
            [((xx - thick * 0.3) * ss, (yy - thick * 0.45) * ss),
             ((xx + thick * 0.3) * ss, (yy + thick * 0.45) * ss)],
            fill=(*dark, 200),
            width=max(1, int(ss * thick * 0.16)),
        )  # fmt: skip
    for ex in knots:
        draw.ellipse(
            [(ex - thick * 0.45) * ss, (y - thick * 0.7) * ss,
             (ex + thick * 0.45) * ss, (y + thick * 0.7) * ss],
            fill=(*color, 235),
        )  # fmt: skip
    return resize_straight(np.asarray(img), (width, height))


def _roller(h, w, wood):
    """Return a wooden roller (straight RGBA): cylinder shading, rounded caps."""
    ss = 2
    W, H = int(w * ss), int(h * ss)
    x = np.linspace(-1, 1, W, dtype=np.float32)
    shade = (
        0.55
        + 0.45 * np.clip(np.cos(x * np.pi / 2), 0, 1) ** 1.5
        + 0.25 * np.exp(-((x + 0.35) ** 2) / 0.02)
    )
    rgb = np.asarray(wood, np.float32)[None, None, :] * shade[None, :, None]
    out = np.empty((H, W, 4), np.uint8)
    out[..., :3] = np.clip(np.repeat(rgb, H, axis=0), 0, 255).astype(np.uint8)
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    cap = int(W * 0.9)
    draw.rounded_rectangle([0, cap, W - 1, H - cap], radius=int(W * 0.2), fill=255)
    draw.rounded_rectangle(
        [int(W * 0.12), 0, int(W * 0.88), cap + 4], radius=int(W * 0.3), fill=255
    )
    draw.rounded_rectangle(
        [int(W * 0.12), H - cap - 4, int(W * 0.88), H - 1],
        radius=int(W * 0.3),
        fill=255,
    )
    out[..., 3] = np.asarray(mask)
    return cv2.resize(out, (int(w), int(h)), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
# Keyframe helpers.
# --------------------------------------------------------------------------- #

_OUT_CUBIC = Ease.bezier(0.33, 1.0, 0.68, 1.0)
_SINE = Ease.bezier(0.37, 0.0, 0.63, 1.0)


def _keys(pairs, ease=None):
    """Build keyframes from ``(time, value)`` pairs with strictly rising time.

    A later pair at (almost) the same time replaces the earlier value. ``ease``
    applies to every segment that changes value.
    """
    clean = []
    for time, value in pairs:
        if clean and time <= clean[-1][0] + 1e-9:
            clean[-1] = (clean[-1][0], value)
        elif not clean or time > clean[-1][0]:
            clean.append((time, value))
    keys = []
    for i, (time, value) in enumerate(clean):
        nxt = clean[i + 1][1] if i + 1 < len(clean) else value
        changes = ease is not None and nxt != value
        keys.append(
            Keyframe(time, value, interp="bezier", out_ease=ease)
            if changes
            else Keyframe(time, value)
        )
    return keys


def _opacity(t0, t1, hold_until, duration, peak=100.0):
    return Property(
        0.0,
        keyframes=_keys(
            [(0.0, 0.0), (t0, 0.0), (t1, peak), (hold_until, peak), (duration, 0.0)],
            _OUT_CUBIC,
        ),
    )


def _layer(clip, name, x, y, opacity, *, position=None, **transform):
    if position is None:
        # Whole-pixel positions keep the renderer on its copy-free blit path.
        position = (float(round(x)), float(round(y)))
    return CachedAVLayer(
        clip,
        name,
        transform=Transform(
            anchor_point=transform.pop("anchor_point", (0.0, 0.0)),
            position=position,
            opacity=opacity,
            **transform,
        ),
    )


# --------------------------------------------------------------------------- #
# The template.
# --------------------------------------------------------------------------- #


def vertical_quote(
    text,
    source,
    preset="nightlamp_history",
    *,
    dynasty=None,
    medium="auto",
    title=None,
    seal=None,
    per_column=10,
    duration=None,
    font=None,
    year=None,
    size=62,
    layout="center",
    speed=9.0,
    hold=3.0,
    fade_out=0.6,
    dim=0.35,
    seed=7,
):
    """Build a vertical right-to-left quotation composition.

    Columns read right to left: optional ``title`` (vermilion), the body
    columns, then ``source`` bottom-aligned in the leftmost column with the
    optional ``seal`` below it. Bamboo slips slide in one by one, or a paper
    handscroll unrolls from the right between two rollers; then each character
    is written top to bottom, right column first.

    Parameters
    ----------
    text : str
        Quotation; ``\\n`` or ``|`` force column breaks. Quote verbatim.
    source : str or None
        Attribution such as ``"《史記·項羽本紀》"``.
    preset : ChannelPreset or str, optional
        Palette roles ``paper``, ``paper_shadow``, ``ink``, ``seal``,
        ``bamboo``, ``bamboo_dark``; fonts ``quote`` (body) and ``title``.
    dynasty, year, medium : optional
        See ``resolve_medium``.
    title : str, optional
        Vermilion heading column on the right.
    seal : str, optional
        1-4 character seal stamped below the source at the end.
    per_column : int, optional
        Characters per column (NLH default 10).
    duration : float, optional
        Composition length; default is computed (unroll + writing + hold).
    font : str, optional
        Font file overriding the preset.
    size : float, optional
        Body size in pixels at 1080p.
    layout : {"center", "right", "left"}, optional
        Horizontal placement of the scroll.
    speed : float, optional
        Characters written per second.
    hold, fade_out : float, optional
        Seconds held after writing and faded at the end.
    dim : float, optional
        Backdrop darkening 0..1 (0 removes the layer, keeping alpha clean).
    seed : int, optional
        Seeds every texture.

    Returns
    -------
    Composition
        Transparent composition at ``preset.size``. ``comp.quote_meta`` holds
        medium, reason, columns, timing; layers are named ``slip N``,
        ``sheet``, ``column N``, ``seal``.

    Raises
    ------
    ValueError
        If the quotation does not fit on one page; split it with
        ``split_quote_pages`` first.

    Examples
    --------
    >>> vertical_quote.__name__
    'vertical_quote'
    """
    if isinstance(preset, str):
        preset = get_preset(preset)
    if not isinstance(preset, ChannelPreset):
        raise TypeError("preset must be a ChannelPreset or preset name")
    if layout not in ("center", "right", "left"):
        raise ValueError("layout must be 'center', 'right' or 'left'")
    kind, reason = resolve_medium(dynasty, year, medium)
    bamboo = kind == "bamboo"
    body_font = preset.font("quote", font)
    title_font = preset.font("title", font)
    W, H = preset.size
    S = H / 1080.0
    paper = preset.color("paper", (239, 228, 204))
    shadow_rgb = preset.color("paper_shadow", (201, 180, 140))
    ink = preset.color("ink", (30, 26, 22))
    seal_rgb = preset.color("seal", (176, 30, 28))
    bamboo_rgb = preset.color("bamboo", (200, 168, 106))
    bamboo_dark = preset.color("bamboo_dark", (138, 106, 52))

    fs = int(size * S)
    cell = int(fs * 1.12)
    body = split_columns(text, per_column)
    if not body:
        raise ValueError("text must not be empty")
    per = max(len(c) for c in body)
    fsrc = int(fs * 0.62)
    csrc = int(fsrc * 1.12)
    cols = []  # (chars, font size, cell, color, kind) right to left
    if title:
        cols.append((list(title), title_font, fs, cell, seal_rgb, "title"))
    for c in body:
        cols.append((c, body_font, fs, cell, ink, "body"))
    src_color = mix_color(ink, bamboo_dark, 0.5 if not bamboo else 0.15)
    if source:
        cols.append((list(source), body_font, fsrc, csrc, src_color, "source"))
    n = len(cols)
    text_h = max(
        per * cell,
        len(title) * cell if title else 0,
        len(source) * csrc if source else 0,
    )
    colw = int(cell * (1.55 if bamboo else 1.45))
    padv = int(cell * (1.05 if bamboo else 0.8))
    panel_h = text_h + 2 * padv
    panel_w = colw * n
    margin_x = int(cell * (0.4 if bamboo else 1.2))
    total_w = panel_w + 2 * margin_x
    if panel_h > H * 0.94 or total_w > W * 0.96:
        raise ValueError(
            f"quotation needs {total_w}x{panel_h}px, larger than the frame; "
            "split it with split_quote_pages() or lower per_column/size"
        )
    cx = {
        "center": W / 2,
        "right": W - total_w / 2 - W * 0.05,
        "left": total_w / 2 + W * 0.05,
    }[layout]
    x_right = cx + total_w / 2 - margin_x
    y_top = H / 2 - panel_h / 2
    seal_room = int(cell * 1.25) if seal else 0

    nchar = sum(len(c[0]) for c in cols)
    tin = 0.35 + (n * 0.09 if bamboo else 0.9)
    write_t = nchar / speed
    T = float(duration) if duration else tin + write_t + hold + fade_out
    fade_out = min(fade_out, T / 3)
    hold_until = max(0.0, T - fade_out)

    comp = Composition(
        size=(W, H), fps=preset.fps, duration=T, name="vertical_quote",
        transparent=True,
    )  # fmt: skip
    layers = {}

    def add(layer):
        layers[layer.name] = layer
        comp.add_layer(layer)

    if dim > 0:
        add(_dim_layer(W, H, dim, T, hold_until))

    # ---- backdrop: slips or scroll -------------------------------------- #
    if bamboo:
        sw = int(colw * 0.9)
        gap = colw - sw
        pad = int(24 * S)
        cord_ys = [padv * 0.45, panel_h - padv * 0.45]
        if text_h > cell * 9:
            cord_ys.insert(1, panel_h / 2)
        cord_color = mix_color(bamboo_dark, (0, 0, 0), 0.25)
        thick = max(4, int(9 * S))
        for i in range(n):
            slip = _bamboo_slip(sw, panel_h, bamboo_rgb, seed * 1000 + i)
            img = np.zeros((panel_h + 2 * pad, colw + 2 * pad, 4), np.uint8)
            img[...] = drop_shadow(
                slip[..., 3] / 255.0, (colw + 2 * pad, panel_h + 2 * pad),
                (pad + gap // 2, pad + int(8 * S)), 9 * S, 0.45,
            )  # fmt: skip
            alpha_over(img, slip, pad + gap // 2, pad)
            for y in cord_ys:
                seg = _cord_segment(colw, 2 * thick + 4, thick + 2, thick, cord_color,
                                    (gap // 2, gap // 2 + sw))  # fmt: skip
                alpha_over(img, seg, pad, pad + y - thick - 2)
            x1 = x_right - i * colw
            ximg = float(round(x1 - sw - gap / 2 - pad))
            yimg = float(round(y_top - pad))
            t0 = 0.15 + i * 0.09
            slide = float(40 * S)
            add(
                _layer(
                    rgba_still(img, T),
                    f"slip {i + 1}",
                    ximg,
                    yimg,
                    _opacity(t0, t0 + 0.35, hold_until, T),
                    position=Property(
                        (ximg, yimg),
                        value_type="vec2",
                        keyframes=_keys(
                            [
                                (0.0, (ximg + slide, yimg)),
                                (t0, (ximg + slide, yimg)),
                                (t0 + 0.35, (ximg, yimg)),
                            ],
                            _OUT_CUBIC,
                        ),  # fmt: skip
                    ),
                )
            )
        centers = [x_right - i * colw - sw / 2 for i in range(n)]
    else:
        full_w = panel_w + 2 * margin_x
        back = Image.fromarray(np.array(paper_texture(full_w, panel_h, paper, seed)))
        draw = ImageDraw.Draw(back)
        rule = mix_color(paper, seal_rgb, 0.59)
        lw = max(1, int(1.6 * S))
        r1, r2 = padv * 0.45, padv * 0.45 + 6 * S
        for yy in (r1, r2, panel_h - r1, panel_h - r2):
            draw.line(
                [(margin_x * 0.6, yy), (full_w - margin_x * 0.6, yy)],
                fill=rule,
                width=lw,
            )
        for i in range(n + 1):
            xx = margin_x + panel_w - i * colw
            draw.line([(xx, r2), (xx, panel_h - r2)], fill=rule, width=lw)
        back = np.asarray(back)
        pad = int(50 * S)
        right_edge = x_right + margin_x
        u0, u1 = 0.1, 1.0

        def vw_at(t):
            return int(round(full_w * _SINE((t - u0) / (u1 - u0))))

        shadow_px = np.asarray(mix_color(shadow_rgb, (0, 0, 0), 0.85), np.uint8)
        sheet_size = (full_w + 2 * pad, panel_h + 2 * pad)

        def sheet_build(vw):
            rgb = np.empty((sheet_size[1], sheet_size[0], 3), np.uint8)
            rgb[...] = shadow_px
            if vw <= 0:
                return rgb, np.zeros(sheet_size[::-1], np.float32)
            ox = pad + full_w - vw
            alpha = rect_shadow(
                sheet_size, (ox, pad + int(10 * S), vw, panel_h), 16 * S, 0.42
            )
            rgb[pad : pad + panel_h, ox : ox + vw] = back[:, full_w - vw :]
            alpha[pad : pad + panel_h, ox : ox + vw] = 1.0
            return rgb, alpha

        sheet_clip = cached_rgba_clip(
            (full_w + 2 * pad, panel_h + 2 * pad),
            T,
            lambda t: vw_at(t),
            sheet_build,
            max_cache=64,
        )
        add(
            _layer(
                sheet_clip,
                "sheet",
                right_edge - full_w - pad,
                y_top - pad,
                _opacity(0.0, 0.05, hold_until, T),
            )  # fmt: skip
        )
        rw = int(cell * 0.42)
        ext = int(cell * 0.35)
        roll_x = float(round(right_edge - rw / 2))
        roll_end = float(round(right_edge - full_w - rw / 2))
        wood = mix_color(bamboo_dark, (0, 0, 0), 0.35)
        roll = _roller(panel_h + 2 * ext, rw, wood)
        ry = float(round(y_top - ext))
        add(
            _layer(
                rgba_still(roll, T),
                "roller right",
                roll_x,
                ry,
                _opacity(u0 - 0.05, u0, hold_until, T),
            )  # fmt: skip
        )
        add(
            _layer(
                rgba_still(roll, T),
                "roller left",
                roll_x,
                ry,
                _opacity(u0 - 0.05, u0, hold_until, T),
                position=Property(
                    (roll_x, ry),
                    value_type="vec2",
                    keyframes=_keys(
                        [
                            (0.0, (roll_x, ry)),
                            (u0, (roll_x, ry)),
                            (u1, (roll_end, ry)),
                        ],
                        _SINE,
                    ),  # fmt: skip
                ),
            )
        )
        centers = [x_right - (i + 0.5) * colw for i in range(n)]

    # ---- ink columns ------------------------------------------------------ #
    starts = []
    k = 0
    for chars, *_ in cols:
        starts.append([tin + (k + j) / speed for j in range(len(chars))])
        k += len(chars)
    levels = 8
    for i, (chars, fpath, fsize, c, color, role) in enumerate(cols):
        base = ink_bleed(_render_column(chars, fpath, fsize, c, color), 0.5 * S)
        h_col, w_col = base.shape[:2]

        def key_fn(t, starts=starts[i]):
            return tuple(
                int(round(ease_out_cubic((t - t0) / 0.18) * levels)) for t0 in starts
            )

        def build(key, base=base, c=c):
            rows = np.repeat(np.asarray(key, np.float32) / levels, c)[: base.shape[0]]
            out = base.copy()
            out[..., 3] = (out[..., 3] * rows[:, None]).astype(np.uint8)
            return out[..., :3].copy(), out[..., 3] / 255.0

        clip = cached_rgba_clip(
            (w_col, h_col), T, key_fn, build, max_cache=len(chars) * 2 + 4
        )
        yoff = padv
        if role == "source":
            yoff = max(padv, padv + text_h - len(chars) * c - seal_room)
        add(
            _layer(
                clip,
                f"column {i + 1}",
                centers[i] - w_col / 2,
                y_top + yoff,
                _opacity(0.0, 0.01, hold_until, T),
            )  # fmt: skip
        )

    # ---- closing seal ------------------------------------------------------ #
    if seal:
        ssz = int(cell * 1.05)
        stamp = seal_stamp(seal, ssz, seal_rgb, seed, body_font)
        ts = tin + nchar / speed + 0.15
        te = ts + 0.28
        lx = centers[-1]
        ly = y_top + panel_h - padv * 0.55 - ssz * 0.5
        add(
            CachedAVLayer(
                rgba_still(stamp, T),
                "seal",
                transform=Transform(
                    anchor_point=(ssz / 2, ssz / 2),
                    position=(lx, ly),
                    rotation=-3.0,
                    interpolation="linear",  # cubic rings past alpha
                    scale=Property(
                        (150.0, 150.0),
                        value_type="vec2",
                        keyframes=_keys(
                            [
                                (0.0, (150.0, 150.0)),
                                (ts, (150.0, 150.0)),
                                (te, (100.0, 100.0)),
                            ],
                            _OUT_CUBIC,
                        ),
                    ),  # fmt: skip
                    opacity=_opacity(ts, te, hold_until, T),
                ),
            )
        )

    comp.quote_meta = {
        "kind": "quote",
        "medium": kind,
        "medium_reason": reason,
        "columns": ["".join(c[0]) for c in cols],
        "per_column": per_column,
        "write_start": tin,
        "write_end": tin + write_t,
        "duration": T,
        "x": 0,
        "y": 0,
        "w": W,
        "h": H,
    }
    comp.quote_layers = layers
    return comp


def _dim_layer(W, H, dim, T, hold_until):
    return SolidLayer(
        "dim",
        color=(8, 6, 4),
        size=(W, H),
        transform=Transform(
            opacity=Property(
                0.0,
                keyframes=_keys(
                    [
                        (0.0, 0.0),
                        (0.4, dim * 100.0),
                        (hold_until, dim * 100.0),
                        (T, 0.0),
                    ],
                    _OUT_CUBIC,
                ),
            )
        ),
    )
