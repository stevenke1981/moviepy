"""Chapter overlay pack: logo, watermark, vertical chapter title and a popping CTA.

Template form of the night-lamp R5 scene layout. Geometry lives on a 1920x1080
reference canvas (``SceneLayout``) and is scaled to any composition. Per chapter
the pack is four layers that fade together (0.4 s in and out of a 4 s window):
the channel logo at the top left, a semi-transparent watermark at the top
right, the chapter title in vertical columns (top to bottom, right to left) on
a dark rounded panel with a gold outline, and a red pill "subscribe" button
whose scale pops 80% -> 104% -> 100% at frames 0 / 6 / 10. The play triangle is
drawn, never taken from a font. Every picture is rasterised once and reused as
a still, so rendering costs only the layer transforms.

Examples
--------
>>> from moviepy.ae.templates.scene_overlay import SceneLayout
>>> layout = SceneLayout()
>>> layout.scaled((1280, 720)).cta
(851, 453, 280, 73)
>>> SceneLayout.from_dict(layout.to_dict()) == layout
True
"""

import json
import math
import os
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from moviepy.ae.composition import Composition
from moviepy.ae.properties import Keyframe, Property
from moviepy.ae.properties.easing import Ease
from moviepy.ae.templates.paper import CachedAVLayer, has_glyph, load_font, rgba_still
from moviepy.ae.templates.quote import PUNCT_CENTER, VMAP, split_columns
from moviepy.ae.transform import Transform


__all__ = [
    "SceneLayout",
    "ScaledLayout",
    "add_scene_overlays",
    "cta_image",
    "export_scene_overlay",
    "logo_image",
    "scene_overlay_layers",
    "vertical_title_image",
    "watermark_image",
]

_SS = 3  # supersampling factor for every raster
_RECTS = ("logo_rect", "watermark_rect", "title_rect", "cta_rect")
_OUT = Ease.bezier(0.33, 1.0, 0.68, 1.0)
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Layout.
# --------------------------------------------------------------------------- #


def _rgb(value, name, n):
    out = tuple(int(v) for v in value)
    if len(out) not in n or any(not 0 <= v <= 255 for v in out):
        raise ValueError(f"{name} must be {' or '.join(map(str, n))} ints in 0..255")
    return out


@dataclass(frozen=True)
class ScaledLayout:
    """Pixel geometry of a ``SceneLayout`` on one canvas.

    Rectangles are ``(x, y, w, h)`` ints. Origins scale by the per-axis ratio
    (so right and bottom anchored items keep their margins) and sizes by
    ``scale``, the smaller ratio, so a rectangle never leaves the canvas.
    """

    size: tuple
    scale: float
    logo: tuple
    watermark: tuple
    title: tuple
    cta: tuple
    cta_inset: int
    title_font_px: int
    logo_font_px: int
    cta_font_px: int
    watermark_font_px: int
    title_floor_px: int


@dataclass(frozen=True)
class SceneLayout:
    """Geometry, timing and colours of the chapter overlay pack.

    Rectangles are ``(x, y, w, h)`` on ``reference_size``; the defaults are the
    R26 channel policy. ``cta_pop`` holds ``(frame, scale_percent)`` keys at
    ``pop_fps`` (frames 0 / 6 / 10 at 30 fps), so the pop lasts the same
    seconds whatever the composition frame rate is.

    Examples
    --------
    >>> layout = SceneLayout()
    >>> layout.duration, layout.fade
    (4.0, 0.4)
    >>> layout.pop_seconds()
    [(0.0, 80.0), (0.2, 104.0), (0.3333333333333333, 100.0)]
    >>> SceneLayout(duration=0.5)
    Traceback (most recent call last):
    ...
    ValueError: duration must exceed twice the fade (0.8 s)
    """

    reference_size: tuple = (1920, 1080)
    logo_rect: tuple = (192, 124, 220, 110)
    watermark_rect: tuple = (1464, 65, 358, 58)
    title_rect: tuple = (1520, 150, 175, 300)
    cta_rect: tuple = (1276, 680, 420, 110)
    cta_inset: int = 12
    duration: float = 4.0
    fade: float = 0.4
    cta_pop: tuple = ((0, 80.0), (6, 104.0), (10, 100.0))
    pop_fps: float = 30.0
    title_font_px: int = 44
    title_floor_px: int = 20
    logo_font_px: int = 36
    cta_font_px: int = 36
    watermark_font_px: int = 26
    cta_color: tuple = (221, 55, 55)
    panel_fill: tuple = (9, 13, 21, 158)
    panel_outline: tuple = (201, 167, 97, 170)
    title_color: tuple = (240, 211, 146)
    logo_color: tuple = (255, 255, 255)
    watermark_color: tuple = (255, 255, 255)
    watermark_opacity: float = 0.55

    def __post_init__(self):
        ref = tuple(int(v) for v in self.reference_size)
        if len(ref) != 2 or min(ref) <= 0:
            raise ValueError("reference_size must be two positive ints")
        object.__setattr__(self, "reference_size", ref)
        for name in _RECTS:
            rect = tuple(int(v) for v in getattr(self, name))
            if len(rect) != 4:
                raise ValueError(f"{name} must be (x, y, w, h)")
            x, y, w, h = rect
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > ref[0] or y + h > ref[1]:
                raise ValueError(f"{name} {rect} is outside the {ref} reference canvas")
            object.__setattr__(self, name, rect)
        object.__setattr__(self, "duration", float(self.duration))
        object.__setattr__(self, "fade", float(self.fade))
        if self.fade < 0:
            raise ValueError("fade must not be negative")
        if self.duration <= 2 * self.fade:
            raise ValueError(
                f"duration must exceed twice the fade ({2 * self.fade:g} s)"
            )
        if self.duration <= 0 or float(self.pop_fps) <= 0:
            raise ValueError("duration and pop_fps must be positive")
        object.__setattr__(self, "pop_fps", float(self.pop_fps))
        pop = tuple((float(f), float(s)) for f, s in self.cta_pop)
        if not pop or any(s <= 0 for _, s in pop) or pop[0][0] < 0:
            raise ValueError("cta_pop needs (frame >= 0, positive scale) keys")
        if any(b[0] <= a[0] for a, b in zip(pop, pop[1:])):
            raise ValueError("cta_pop frames must be strictly increasing")
        if pop[-1][0] / self.pop_fps > self.duration - self.fade + _EPS:
            raise ValueError("the CTA pop must finish before the fade-out starts")
        object.__setattr__(
            self, "cta_pop", tuple((int(f) if f == int(f) else f, s) for f, s in pop)
        )
        if self.cta_inset < 0 or 2 * self.cta_inset >= min(self.cta_rect[2:]):
            raise ValueError("cta_inset must leave a non-empty button")
        for name in (
            "title_font_px",
            "title_floor_px",
            "logo_font_px",
            "cta_font_px",
            "watermark_font_px",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.title_floor_px > self.title_font_px:
            raise ValueError("title_floor_px must not exceed title_font_px")
        if not 0.0 <= float(self.watermark_opacity) <= 1.0:
            raise ValueError("watermark_opacity must be within 0..1")
        for name in ("cta_color", "title_color", "logo_color", "watermark_color"):
            object.__setattr__(self, name, _rgb(getattr(self, name), name, (3,)))
        for name in ("panel_fill", "panel_outline"):
            object.__setattr__(self, name, _rgb(getattr(self, name), name, (3, 4)))

    def pop_seconds(self):
        """Return the CTA pop as ``(seconds, scale_percent)`` pairs."""
        return [(f / self.pop_fps, s) for f, s in self.cta_pop]

    def scaled(self, size):
        """Return the pixel geometry for a ``(width, height)`` canvas.

        Parameters
        ----------
        size : tuple of int
            Composition size in pixels.

        Returns
        -------
        ScaledLayout
            Rectangles, font sizes and inset in pixels.

        Examples
        --------
        >>> SceneLayout().scaled((1280, 720)).logo
        (128, 83, 147, 73)
        """
        width, height = (int(v) for v in size)
        if width <= 0 or height <= 0:
            raise ValueError("canvas size must be positive")
        sx, sy = width / self.reference_size[0], height / self.reference_size[1]
        s = min(sx, sy)

        def rect(r):
            x, y, w, h = r
            w, h = max(1, round(w * s)), max(1, round(h * s))
            x, y = round(x * sx), round(y * sy)
            return (min(x, max(0, width - w)), min(y, max(0, height - h)), w, h)

        def px(v):
            return max(1, round(v * s))

        return ScaledLayout(
            size=(width, height),
            scale=s,
            logo=rect(self.logo_rect),
            watermark=rect(self.watermark_rect),
            title=rect(self.title_rect),
            cta=rect(self.cta_rect),
            cta_inset=round(self.cta_inset * s),
            title_font_px=px(self.title_font_px),
            logo_font_px=px(self.logo_font_px),
            cta_font_px=px(self.cta_font_px),
            watermark_font_px=px(self.watermark_font_px),
            title_floor_px=px(self.title_floor_px),
        )

    def to_dict(self):
        """Return a JSON-ready dict (tuples become lists)."""
        out = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, tuple):
                value = [list(v) if isinstance(v, tuple) else v for v in value]
            out[item.name] = value
        return out

    @classmethod
    def from_dict(cls, data):
        """Build a layout from ``to_dict`` output; unknown keys are rejected.

        Examples
        --------
        >>> SceneLayout.from_dict({"fade": 0.5}).fade
        0.5
        >>> SceneLayout.from_dict({"fades": 1})
        Traceback (most recent call last):
        ...
        ValueError: unknown SceneLayout field(s): fades
        """
        known = {item.name for item in fields(cls)}
        extra = sorted(set(data) - known)
        if extra:
            raise ValueError(f"unknown SceneLayout field(s): {', '.join(extra)}")
        return cls(**dict(data))

    def to_json(self):
        """Return ``to_dict`` as indented JSON text."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


# --------------------------------------------------------------------------- #
# Raster helpers.
# --------------------------------------------------------------------------- #


def _finish(image, size):
    """Downsample a supersampled RGBA image to ``size`` as straight ``uint8``."""
    small = image.resize(tuple(size), Image.BOX)
    return np.asarray(small, dtype=np.uint8).copy()


def _check_size(size, what):
    w, h = (int(v) for v in size)
    if w < 4 or h < 4:
        raise ValueError(f"{what} size must be at least 4x4 pixels, got {(w, h)}")
    return w, h


def _text_box(font, text, stroke=0):
    left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
    return right - left, bottom - top


def _fit_single_line(text, font_path, font_px, max_w, max_h, floor, what, stroke=0):
    """Return ``(font_px, ss_font)``: the largest size whose line fits the box."""
    if not text or not str(text).strip():
        raise ValueError(f"{what} text must not be empty")
    px = int(font_px)
    floor = max(1, round(px * 0.35)) if floor is None else min(int(floor), px)
    while True:
        font = load_font(font_path, px * _SS)
        w, h = _text_box(font, text, stroke * _SS)
        if w / _SS <= max_w and h / _SS <= max_h:
            return px, font
        if px <= floor:
            raise ValueError(
                f"{what} text {text!r} needs {w / _SS:.0f}x{h / _SS:.0f}px at the "
                f"{floor}px floor but only {max_w:.0f}x{max_h:.0f}px is available; "
                "shorten it or enlarge the rectangle"
            )
        px = max(floor, px - 1 if px <= 24 else int(px * 0.94))


def _draw_line(draw, xy, text, font, fill, stroke=0, stroke_fill=None, anchor="lm"):
    kw = {}
    if stroke:
        kw = {"stroke_width": stroke * _SS, "stroke_fill": stroke_fill}
    draw.text(
        (xy[0] * _SS, xy[1] * _SS), text, font=font, fill=fill, anchor=anchor, **kw
    )


# --------------------------------------------------------------------------- #
# Vertical title.
# --------------------------------------------------------------------------- #

_VPUNCT = {
    "，": "︐", "、": "︑", "。": "︒", "：": "︓", "；": "︔", "！": "︕", "？": "︖",
}  # fmt: skip
_NUDGE = set("，。、,.")  # horizontal forms sit low-left in the em box: lift them


def _font_has(font_path, char):
    """Whether a real font file is known to map ``char`` (False without fontTools)."""
    if not font_path:
        return False
    try:
        import fontTools  # noqa: F401
    except ImportError:
        return False
    return has_glyph(str(font_path), char)


def _glyph_plan(char, font_path):
    """Return ``(glyph, rotate, nudge)`` for drawing ``char`` in a vertical column."""
    glyph = VMAP.get(char) or _VPUNCT.get(char)
    if glyph is not None:
        if _font_has(font_path, glyph):
            return glyph, False, False
        if char in VMAP:  # brackets and dashes turn with the text direction
            return char, True, False
    return char, False, char in _NUDGE


def _tidy_title(text, space=""):
    """Collapse whitespace; replace a space between two wide characters.

    ``space=""`` drops it; ``space="|"`` turns it into a column break, so
    "第三章 雷劫守候" can read as two columns 第三章 / 雷劫守候.
    """
    chars = list(" ".join(str(text).split()))
    out = []
    for i, ch in enumerate(chars):
        if ch == " " and 0 < i < len(chars) - 1:
            if ord(chars[i - 1]) > 0x2E7F and ord(chars[i + 1]) > 0x2E7F:
                out.append(space)
                continue
        out.append(ch)
    return "".join(out)


def _layout_columns(text, rows, max_cols):
    """Pick the most balanced column split that fits ``rows`` x ``max_cols``."""
    if "\n" in text or "|" in text:
        cols = split_columns(text, rows)
        return (
            cols
            if len(cols) <= max_cols and all(len(c) <= rows for c in cols)
            else None
        )
    n = len(text)
    for ncols in range(max(1, math.ceil(n / rows)), max_cols + 1):
        cols = split_columns(text, max(1, math.ceil(n / ncols)))
        if len(cols) <= max_cols and all(len(c) <= rows for c in cols):
            return cols
    return None


def vertical_title_image(
    text,
    size,
    *,
    font=None,
    font_px=44,
    floor_px=20,
    color=(240, 211, 146),
    panel=True,
    panel_fill=(9, 13, 21, 158),
    panel_outline=(201, 167, 97, 170),
    outline_px=2,
    inset=None,
    stroke=1,
):
    r"""Render a chapter title as vertical columns on a rounded dark panel.

    Characters run top to bottom, columns right to left. The text is split into
    balanced columns (closing punctuation never starts one) and the font shrinks
    from ``font_px`` towards ``floor_px`` until everything fits; nothing is ever
    cut. ``\n`` or ``|`` force column breaks. Brackets and CJK punctuation use
    the font's vertical forms when it has them, otherwise they are rotated (or,
    for commas and full stops, lifted towards the upper right).

    Parameters
    ----------
    text : str
        Title. A space between two wide characters is a preferred column
        break (dropped when the title only fits without it).
    size : tuple of int
        Output ``(width, height)`` in pixels (the panel fills it).
    font : str or os.PathLike, optional
        Font file; ``None`` uses Pillow's default font.
    font_px : int, optional
        Starting font size in pixels.
    floor_px : int, optional
        Smallest font size tried before giving up.
    color : tuple of int, optional
        Text colour.
    panel : bool, optional
        Draw the rounded panel; ``False`` leaves only the text.
    panel_fill, panel_outline : tuple of int, optional
        Panel colours with alpha.
    outline_px : float, optional
        Panel outline width in pixels.
    inset : int, optional
        Padding inside the panel; default 5% of the width.
    stroke : float, optional
        Dark text stroke in pixels.

    Returns
    -------
    numpy.ndarray
        Straight-alpha ``uint8`` array of shape ``(h, w, 4)``.

    Raises
    ------
    ValueError
        If the text is empty or does not fit at ``floor_px``.

    Examples
    --------
    >>> vertical_title_image("AB", (60, 90), font_px=20).shape
    (90, 60, 4)
    """
    w, h = _check_size(size, "title")
    spaced = _tidy_title(text, "|")
    text = _tidy_title(text)
    candidates = [spaced, text] if spaced != text else [text]
    if not text.strip():
        raise ValueError("title text must not be empty")
    floor_px = min(int(floor_px), int(font_px))
    pad = max(2, round(w * 0.05)) if inset is None else int(inset)
    avail_w, avail_h = w - 2 * pad, h - 2 * pad
    flat = text.replace("\n", "").replace("|", "")
    px = int(font_px)
    while True:
        cell_w, cell_h = max(1, round(px * 1.18)), max(1, round(px * 1.22))
        rows, max_cols = avail_h // cell_h, avail_w // cell_w
        cols = None
        for candidate in candidates if rows and max_cols else ():
            cols = _layout_columns(candidate, rows, max_cols)
            if cols:
                break
        if cols:
            break
        if px <= floor_px:
            raise ValueError(
                f"title {flat!r} ({len(flat)} characters) does not fit "
                f"{w}x{h}px even at the {floor_px}px floor; shorten it, "
                "split it with '|' or enlarge the rectangle"
            )
        px = max(floor_px, px - 1)
    big = Image.new("RGBA", (w * _SS, h * _SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    if panel:
        radius = max(2, round(min(w, h) * 0.07))
        draw.rounded_rectangle(
            (0, 0, w * _SS - 1, h * _SS - 1),
            radius=radius * _SS,
            fill=tuple(panel_fill) if len(panel_fill) == 4 else (*panel_fill, 255),
        )
        inner = max(1, round(outline_px * _SS))
        mask = Image.new("L", big.size, 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle(
            (0, 0, w * _SS - 1, h * _SS - 1), radius=radius * _SS, fill=255
        )
        md.rounded_rectangle(
            (inner, inner, w * _SS - 1 - inner, h * _SS - 1 - inner),
            radius=max(1, radius * _SS - inner),
            fill=0,
        )
        oc = tuple(panel_outline) if len(panel_outline) == 4 else (*panel_outline, 255)
        ring = Image.new("RGBA", big.size, oc)
        ring.putalpha(mask.point(lambda v: v * oc[3] // 255))
        big = Image.alpha_composite(big, ring)
        draw = ImageDraw.Draw(big)
    fnt = load_font(font, px * _SS)
    fill = (*color[:3], 255)
    edge = (0, 0, 0, 230)
    block_w = len(cols) * cell_w
    block_h = max(len(c) for c in cols) * cell_h
    left = (w - block_w) / 2
    top = (h - block_h) / 2
    for ci, chars in enumerate(cols):  # ci 0 is the rightmost column
        cx = left + (len(cols) - 1 - ci + 0.5) * cell_w
        for ri, ch in enumerate(chars):
            if ch == " ":
                continue
            cy = top + (ri + 0.5) * cell_h
            glyph, rotate, nudge = _glyph_plan(ch, font)
            if rotate:
                tile = Image.new("RGBA", (cell_w * _SS, cell_w * _SS), (0, 0, 0, 0))
                ImageDraw.Draw(tile).text(
                    (cell_w * _SS / 2, cell_w * _SS / 2), glyph, font=fnt, fill=fill,
                    anchor="mm", stroke_width=round(stroke * _SS), stroke_fill=edge,
                )  # fmt: skip
                tile = tile.rotate(-90, resample=Image.BICUBIC)
                big.alpha_composite(
                    tile,
                    (
                        round(cx * _SS - tile.width / 2),
                        round(cy * _SS - tile.height / 2),
                    ),
                )
                continue
            x, y = cx * _SS, cy * _SS
            if _is_punct(ch):
                box = draw.textbbox((0, 0), glyph, font=fnt)
                x, y = x - (box[0] + box[2]) / 2, y - (box[1] + box[3]) / 2
                if nudge:
                    x, y = x + px * _SS * 0.18, y - px * _SS * 0.2
                anchor = "la"
            else:
                anchor = "mm"
            draw.text(
                (x, y), glyph, font=fnt, fill=fill, anchor=anchor,
                stroke_width=round(stroke * _SS), stroke_fill=edge,
            )  # fmt: skip
    return _finish(big, (w, h))


def _is_punct(char):
    """Whether ``char`` is centred by its ink box (punctuation) in vertical text."""
    return char in PUNCT_CENTER or char in _VPUNCT.values() or char in VMAP.values()


# --------------------------------------------------------------------------- #
# CTA, watermark, logo.
# --------------------------------------------------------------------------- #


def cta_image(
    text,
    size,
    *,
    font=None,
    font_px=36,
    floor_px=None,
    color=(221, 55, 55),
    text_color=(255, 255, 255),
    triangle=True,
):
    """Render the red pill call-to-action button with a drawn play triangle.

    The label and the triangle form one group centred in the pill. The label
    shrinks from ``font_px`` to ``floor_px`` to fit and is never truncated.

    Parameters
    ----------
    text : str
        Button label, for example ``"立即訂閱"``.
    size : tuple of int
        Output ``(width, height)`` in pixels.
    font : str or os.PathLike, optional
        Font file; ``None`` uses Pillow's default font.
    font_px, floor_px : int, optional
        Starting and smallest label size (default floor: 35% of ``font_px``).
    color, text_color : tuple of int, optional
        Pill and label colours.
    triangle : bool, optional
        Draw the play triangle after the label.

    Returns
    -------
    numpy.ndarray
        Straight-alpha ``uint8`` array of shape ``(h, w, 4)``.

    Raises
    ------
    ValueError
        If the label cannot fit at ``floor_px``.

    Examples
    --------
    >>> cta_image("GO", (120, 40), font_px=16).shape
    (40, 120, 4)
    """
    w, h = _check_size(size, "CTA")
    tri_h = h * 0.36 if triangle else 0.0
    tri_w = tri_h * 0.86
    gap = h * 0.22 if triangle else 0.0
    pad = h * 0.5
    px, fnt = _fit_single_line(
        text, font, font_px, w - 2 * pad - gap - tri_w, h * 0.68, floor_px, "CTA"
    )
    big = Image.new("RGBA", (w * _SS, h * _SS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    draw.rounded_rectangle(
        (0, 0, w * _SS - 1, h * _SS - 1), radius=h * _SS // 2, fill=(*color, 255)
    )
    # A soft top light keeps the flat red from looking like a sticker.
    light = np.zeros((h * _SS, w * _SS, 4), np.uint8)
    ramp = np.clip(1.0 - np.linspace(0, 2.0, h * _SS), 0, 1) ** 2 * 26
    light[..., :3] = 255
    light[..., 3] = ramp[:, None].astype(np.uint8)
    pill = np.asarray(big)[..., 3:4] / 255.0
    light[..., 3] = (light[..., 3] * pill[..., 0]).astype(np.uint8)
    big = Image.alpha_composite(big, Image.fromarray(light, "RGBA"))
    draw = ImageDraw.Draw(big)
    tw = _text_box(fnt, text)[0] / _SS
    left_box = fnt.getbbox(text)[0] / _SS
    group = tw + gap + tri_w
    x0 = (w - group) / 2
    _draw_line(draw, (x0 - left_box, h / 2), text, fnt, (*text_color, 255))
    if triangle:
        tx = x0 + tw + gap
        draw.polygon(
            [
                (tx * _SS, (h / 2 - tri_h / 2) * _SS),
                (tx * _SS, (h / 2 + tri_h / 2) * _SS),
                ((tx + tri_w) * _SS, h / 2 * _SS),
            ],
            fill=(*text_color, 255),
        )
    return _finish(big, (w, h))


def watermark_image(
    text,
    size,
    *,
    font=None,
    opacity=0.55,
    font_px=26,
    floor_px=None,
    color=(255, 255, 255),
    align="right",
):
    """Render a semi-transparent watermark line with a soft dark shadow.

    Parameters
    ----------
    text : str
        Watermark, for example ``"@my_channel"``.
    size : tuple of int
        Output ``(width, height)`` in pixels.
    font : str or os.PathLike, optional
        Font file; ``None`` uses Pillow's default font.
    opacity : float, optional
        Overall opacity 0..1 baked into the alpha channel.
    font_px, floor_px : int, optional
        Starting and smallest size (default floor: 35% of ``font_px``); the
        text shrinks to fit and is never truncated.
    color : tuple of int, optional
        Text colour.
    align : {"right", "left", "center"}, optional
        Horizontal alignment inside the box.

    Returns
    -------
    numpy.ndarray
        Straight-alpha ``uint8`` array of shape ``(h, w, 4)``.

    Raises
    ------
    ValueError
        If the text cannot fit at ``floor_px``.

    Examples
    --------
    >>> int(watermark_image("@ab", (80, 24), font_px=14)[..., 3].max()) <= 150
    True
    """
    w, h = _check_size(size, "watermark")
    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError("opacity must be within 0..1")
    if align not in ("right", "left", "center"):
        raise ValueError("align must be 'right', 'left' or 'center'")
    pad = max(1, round(h * 0.08))
    px, fnt = _fit_single_line(
        text, font, font_px, w - 2 * pad, h - 2 * pad, floor_px, "watermark"
    )
    tw = _text_box(fnt, text)[0] / _SS
    bx = fnt.getbbox(text)[0] / _SS
    x = {"left": pad, "right": w - pad - tw, "center": (w - tw) / 2}[align] - bx
    ink = Image.new("RGBA", (w * _SS, h * _SS), (0, 0, 0, 0))
    _draw_line(ImageDraw.Draw(ink), (x, h / 2), text, fnt, (*color, 255))
    alpha = np.asarray(ink)[..., 3].astype(np.float32) / 255.0
    blur = Image.fromarray((alpha * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(1.0, px * 0.08) * _SS)
    )
    shadow = np.asarray(blur).astype(np.float32) / 255.0 * 0.75
    total = alpha + shadow * (1.0 - alpha)  # text over a black shadow
    out = np.zeros((h * _SS, w * _SS, 4), np.float32)
    out[..., :3] = (
        np.asarray(color, np.float32)[None, None, :]
        * (alpha / np.maximum(total, 1e-6))[..., None]
    )
    out[..., 3] = total * 255.0 * float(opacity)
    img = Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGBA")
    return _finish(img, (w, h))


def logo_image(
    logo, size, *, font=None, font_px=36, floor_px=None, color=(255, 255, 255)
):
    """Fit a logo (image or text) into a box, centred, never cropped.

    Parameters
    ----------
    logo : str, os.PathLike, PIL.Image.Image or numpy.ndarray
        An image file path or image object, or plain text. A ``str`` naming an
        existing file is loaded; any other ``str`` is drawn as the channel name
        (a ``Path`` must exist).
    size : tuple of int
        Output ``(width, height)`` in pixels.
    font, font_px, floor_px, color
        Text logo font, starting/smallest size and colour.

    Returns
    -------
    numpy.ndarray
        Straight-alpha ``uint8`` array of shape ``(h, w, 4)``.

    Raises
    ------
    ValueError
        If a text logo cannot fit at ``floor_px``.
    FileNotFoundError
        If a ``Path`` logo does not exist.

    Examples
    --------
    >>> logo_image("Ab", (60, 30), font_px=16).shape
    (30, 60, 4)
    """
    w, h = _check_size(size, "logo")
    if isinstance(logo, (str, os.PathLike)) and (
        isinstance(logo, os.PathLike) or os.path.isfile(logo)
    ):
        source = Image.open(logo).convert("RGBA")
    elif isinstance(logo, Image.Image):
        source = logo.convert("RGBA")
    elif isinstance(logo, np.ndarray):
        source = Image.fromarray(np.ascontiguousarray(logo, dtype=np.uint8)).convert(
            "RGBA"
        )
    else:
        px, fnt = _fit_single_line(
            str(logo), font, font_px, w - 4, h - 4, floor_px, "logo", stroke=1
        )
        big = Image.new("RGBA", (w * _SS, h * _SS), (0, 0, 0, 0))
        tw = _text_box(fnt, str(logo))[0] / _SS
        bx = fnt.getbbox(str(logo))[0] / _SS
        _draw_line(
            ImageDraw.Draw(big), ((w - tw) / 2 - bx, h / 2), str(logo), fnt,
            (*color, 255), stroke=1, stroke_fill=(0, 0, 0, 200),
        )  # fmt: skip
        return _finish(big, (w, h))
    ratio = min(w / source.width, h / source.height)
    nw, nh = max(1, round(source.width * ratio)), max(1, round(source.height * ratio))
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.alpha_composite(
        source.resize((nw, nh), Image.LANCZOS), ((w - nw) // 2, (h - nh) // 2)
    )
    return np.asarray(canvas, dtype=np.uint8).copy()


# --------------------------------------------------------------------------- #
# Layers.
# --------------------------------------------------------------------------- #


def _keys(rows):
    keys = []
    for time, value, *ease in rows:
        if keys and time <= keys[-1].time + _EPS:
            continue
        if ease:
            keys.append(Keyframe(time, value, interp="bezier", out_ease=ease[0]))
        else:
            keys.append(Keyframe(time, value))
    return keys


def _fade(t0, t1, fade):
    """Opacity Property: 0 at ``t0``, 100 after ``fade``, back to 0 at ``t1``."""
    if fade <= 0:
        rows = [(t0, 100.0), (t1, 100.0)]
    else:
        rows = [(t0, 0.0), (t0 + fade, 100.0), (t1 - fade, 100.0), (t1, 0.0)]
    return Property(0.0, keyframes=_keys(rows))


def _still_layer(image, name, xy, t0, t1, fade, *, anchor=(0.0, 0.0), scale=None):
    transform = Transform(
        anchor_point=anchor,
        position=(float(xy[0]), float(xy[1])),
        opacity=_fade(0.0, t1 - t0, fade),  # layer clock: t - start_time
        interpolation="linear",  # cubic resampling rings past alpha
        **({} if scale is None else {"scale": scale}),
    )
    return CachedAVLayer(
        rgba_still(image, t1 - t0),
        name,
        transform=transform,
        in_point=t0,
        out_point=t1,
        start_time=t0,
    )


def _pop_scale(layout, t0):
    rows = []
    for i, (sec, pct) in enumerate(layout.pop_seconds()):
        value = (pct, pct)
        rows.append(
            (t0 + sec, value, _OUT)
            if i < len(layout.cta_pop) - 1
            else (t0 + sec, value)
        )
    return Property((100.0, 100.0), value_type="vec2", keyframes=_keys(rows))


def _static_parts(layout, sc, logo, watermark, cta_text, font):
    """Rasterise everything that is identical for every chapter."""
    parts = {}
    if logo is not None:
        parts["logo"] = logo_image(
            logo,
            sc.logo[2:],
            font=font,
            font_px=sc.logo_font_px,
            color=layout.logo_color,
        )
    if watermark:
        parts["watermark"] = watermark_image(
            watermark, sc.watermark[2:], font=font, opacity=layout.watermark_opacity,
            font_px=sc.watermark_font_px, color=layout.watermark_color,
        )  # fmt: skip
    if cta_text:
        inset = sc.cta_inset
        parts["cta"] = cta_image(
            cta_text, (sc.cta[2] - 2 * inset, sc.cta[3] - 2 * inset), font=font,
            font_px=sc.cta_font_px, color=layout.cta_color,
        )  # fmt: skip
    return parts


def _title_image(layout, sc, title, font):
    return vertical_title_image(
        title, sc.title[2:], font=font, font_px=sc.title_font_px,
        floor_px=sc.title_floor_px, color=layout.title_color,
        panel_fill=layout.panel_fill, panel_outline=layout.panel_outline,
    )  # fmt: skip


def _chapter_layers(layout, sc, parts, title, start, watermark_span):
    t0, t1 = float(start), float(start) + layout.duration
    layers = []
    if "logo" in parts:
        layers.append(
            _still_layer(parts["logo"], "logo", sc.logo[:2], t0, t1, layout.fade)
        )
    if "watermark" in parts and watermark_span is not False:
        a, b = (t0, t1) if watermark_span is None else watermark_span
        layers.append(
            _still_layer(
                parts["watermark"], "watermark", sc.watermark[:2], a, b, layout.fade
            )
        )
    if title is not None:
        layers.append(
            _still_layer(title, "chapter title", sc.title[:2], t0, t1, layout.fade)
        )
    if "cta" in parts:
        image = parts["cta"]
        ih, iw = image.shape[:2]
        x, y, w, h = sc.cta
        layers.append(
            _still_layer(
                image, "subscribe", (x + (w - 1) / 2, y + (h - 1) / 2), t0, t1, layout.fade,
                anchor=((iw - 1) / 2, (ih - 1) / 2), scale=_pop_scale(layout, 0.0),
            )
        )  # fmt: skip
    return layers


def scene_overlay_layers(
    layout,
    canvas_size,
    *,
    chapter_title=None,
    start=0.0,
    logo=None,
    watermark=None,
    cta_text="立即訂閱",
    font=None,
    watermark_span=None,
):
    """Build the AE layers of one chapter overlay.

    Layers are returned bottom to top: ``logo``, ``watermark``,
    ``chapter title`` and ``subscribe``; each exists only when its input is
    given. All of them are active during ``[start, start + layout.duration)``
    and fade together; the CTA also pops (scale keys at ``layout.cta_pop``).
    Keyframes are in composition time.

    Parameters
    ----------
    layout : SceneLayout
        Geometry and timing.
    canvas_size : tuple of int
        Composition ``(width, height)``.
    chapter_title : str, optional
        Vertical title; ``None`` omits the title layer.
    start : float, optional
        Composition time of the chapter overlay.
    logo : str, os.PathLike, PIL.Image.Image or numpy.ndarray, optional
        Image or text, see ``logo_image``.
    watermark : str, optional
        Watermark text.
    cta_text : str, optional
        Button label; ``None`` or ``""`` omits the button.
    font : str or os.PathLike, optional
        Font file for all text (``None``: Pillow's default font).
    watermark_span : tuple of float, optional
        ``(t0, t1)`` over which the watermark stays (fading at both ends);
        ``None`` ties it to the chapter window.

    Returns
    -------
    list of CachedAVLayer
        Not yet in any composition; add them with ``comp.add_layer``.

    Raises
    ------
    ValueError
        If a text does not fit its rectangle at its minimum font size.

    Examples
    --------
    >>> layers = scene_overlay_layers(SceneLayout(), (320, 180), watermark="@x")
    >>> [layer.name for layer in layers]
    ['watermark', 'subscribe']
    """
    sc = layout.scaled(canvas_size)
    if watermark_span is not None:
        watermark_span = (float(watermark_span[0]), float(watermark_span[1]))
        if watermark_span[1] - watermark_span[0] <= 2 * layout.fade:
            raise ValueError("watermark_span must be longer than twice the fade")
    parts = _static_parts(layout, sc, logo, watermark, cta_text, font)
    title = _title_image(layout, sc, chapter_title, font) if chapter_title else None
    return _chapter_layers(layout, sc, parts, title, start, watermark_span)


def _check_chapters(layout, chapters, comp_duration):
    rows = []
    for item in chapters:
        start, title = item
        rows.append((float(start), title))
    rows.sort(key=lambda r: r[0])
    for (s0, _), (s1, _) in zip(rows, rows[1:]):
        if s1 < s0 + layout.duration - _EPS:
            raise ValueError(
                f"chapter overlays overlap: the one at {s0:g}s runs until "
                f"{s0 + layout.duration:g}s but the next starts at {s1:g}s"
            )
    for s, title in rows:
        if s < 0:
            raise ValueError(f"chapter start {s:g}s is negative")
        if comp_duration is not None and s + layout.duration > comp_duration + _EPS:
            raise ValueError(
                f"chapter at {s:g}s needs until {s + layout.duration:g}s but the "
                f"composition is {comp_duration:g}s long"
            )
    return rows


def add_scene_overlays(
    comp,
    layout,
    chapters,
    *,
    logo=None,
    watermark=None,
    cta_text="立即訂閱",
    font=None,
    watermark_span="auto",
):
    """Add one overlay set per chapter to a composition.

    Parameters
    ----------
    comp : Composition
        Target; its size sets the geometry and its duration bounds the chapters.
    layout : SceneLayout
        Geometry and timing.
    chapters : sequence of (float, str)
        ``(start_seconds, title)`` per chapter. Windows of
        ``layout.duration`` seconds must not overlap (touching is fine).
    logo, watermark, cta_text, font
        See ``scene_overlay_layers``; the static pictures are rasterised once.
    watermark_span : {"auto", "chapter"} or tuple of float, optional
        ``"auto"`` keeps one watermark layer from the first chapter's start to
        the last chapter's end, ``"chapter"`` repeats it with every chapter,
        a ``(t0, t1)`` tuple sets the span explicitly.

    Returns
    -------
    list of CachedAVLayer
        Every added layer, in chapter order.

    Raises
    ------
    ValueError
        If chapters overlap, start before 0, run past the composition, or a
        title does not fit.

    Examples
    --------
    >>> comp = Composition(size=(320, 180), fps=10, duration=12, transparent=True)
    >>> layers = add_scene_overlays(comp, SceneLayout(), [(0, "A"), (5, "B")])
    >>> len(layers)
    4
    >>> add_scene_overlays(comp, SceneLayout(), [(0, "A"), (3, "B")])
    Traceback (most recent call last):
    ...
    ValueError: chapter overlays overlap: the one at 0s runs until 4s but the next starts at 3s
    """
    rows = _check_chapters(layout, chapters, float(comp.duration))
    if not rows:
        return []
    sc = layout.scaled(comp.size)
    parts = _static_parts(layout, sc, logo, watermark, cta_text, font)
    images = [_title_image(layout, sc, t, font) if t else None for _, t in rows]
    if watermark_span == "auto":
        span = (rows[0][0], rows[-1][0] + layout.duration)
    elif watermark_span == "chapter":
        span = None
    else:
        span = (float(watermark_span[0]), float(watermark_span[1]))
    added = []
    for i, ((start, _), image) in enumerate(zip(rows, images)):
        use = parts
        if span is not None and i:
            use = {k: v for k, v in parts.items() if k != "watermark"}
        for layer in _chapter_layers(layout, sc, use, image, start, span):
            comp.add_layer(layer)
            added.append(layer)
    return added


def export_scene_overlay(
    path,
    layout,
    canvas_size,
    fps,
    *,
    chapter_title=None,
    logo=None,
    watermark=None,
    cta_text="立即訂閱",
    font=None,
    codec="qtrle",
    overwrite=False,
):
    """Write one chapter overlay as a transparent MOV (qtrle/argb by default).

    The clip lasts ``layout.duration`` seconds and starts fully transparent,
    ready to be overlaid on the clean master at each chapter.

    Parameters
    ----------
    path : str or os.PathLike
        Output ``.mov``; ``<path>.json`` evidence is written beside it.
    layout : SceneLayout
        Geometry and timing.
    canvas_size : tuple of int
        Frame ``(width, height)``.
    fps : float
        Frame rate; ``layout.duration * fps`` must be a whole number.
    chapter_title, logo, watermark, cta_text, font
        See ``scene_overlay_layers``.
    codec : {"qtrle", "prores_4444", "png"}, optional
        See ``export_overlay_loop``.
    overwrite : bool, optional
        Replace existing files.

    Returns
    -------
    dict
        The ``export_overlay_loop`` evidence plus ``expected_frames``.

    Raises
    ------
    ValueError
        If the duration is not a whole number of frames.
    RuntimeError
        If FFmpeg fails, or the decoded frame count differs from
        ``duration * fps``, or the file has no visible overlay.
    """
    from moviepy.ae.templates.ambience import export_overlay_loop

    fps = float(fps)
    frames = int(round(layout.duration * fps))
    if fps <= 0 or abs(layout.duration * fps - frames) > 1e-6:
        raise ValueError(
            f"duration {layout.duration:g}s at {fps:g} fps is not a whole number of frames"
        )
    comp = Composition(
        size=tuple(int(v) for v in canvas_size), fps=fps, duration=layout.duration,
        name="scene_overlay", transparent=True,
    )  # fmt: skip
    try:
        for layer in scene_overlay_layers(
            layout, comp.size, chapter_title=chapter_title, start=0.0, logo=logo,
            watermark=watermark, cta_text=cta_text, font=font,
        ):  # fmt: skip
            comp.add_layer(layer)
        report = export_overlay_loop(comp, Path(path), codec=codec, overwrite=overwrite)
    finally:
        comp.close()
    trip = report["roundtrip"]
    if report["frames"] != frames or trip["frames_decoded"] != frames:
        raise RuntimeError(
            f"overlay has {trip['frames_decoded']} decoded frames, expected {frames}"
        )
    if report["alpha"]["max"] <= 0:
        raise RuntimeError("overlay is fully transparent; nothing was drawn")
    report["expected_frames"] = frames
    return report
