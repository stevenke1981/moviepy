"""Historical or document image inserts with a small source citation.

A story video shows a scan, painting or photograph for a few seconds while the
narrator talks about it. ``SourceInsert`` describes one such shot: the image is
fitted whole (never cropped, never enlarged beyond 2x, pixels never edited),
placed above the subtitle safe rect on a paper or dark background, and a tiny
citation line is set in the corner that is clear of the subtitles and of the
image and is the quietest part of the frame. ``layout="portrait"`` is the
catalogue-card look: the image on the left at native size and a title, a
caption and a note in a right column.

All geometry is authored on a 1920x1080 reference and scaled by
``min(width / 1920, height / 1080)``. Opacity keyframes live on the layer clock
(``t - start_time``), so an insert that starts late is visible.

Examples
--------
>>> insert = SourceInsert("scan.jpg", start=3.0, duration=5.0, citation="Wellcome")
>>> SourceInsert.from_dict(insert.to_dict()) == insert
True
>>> citation_image("Wellcome Collection", px=12).mode
'RGBA'
"""

import hashlib
import math
import os
from dataclasses import dataclass, fields
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from moviepy.ae.properties import Keyframe, Property
from moviepy.ae.templates.name_tag import REFERENCE_SIZE, SUBTITLE_SAFE_RECT
from moviepy.ae.templates.paper import (
    CachedAVLayer,
    load_font,
    paper_texture,
    rgba_still,
)
from moviepy.ae.transform import Transform


__all__ = [
    "SourceInsert",
    "add_source_inserts",
    "citation_image",
    "source_insert_image",
    "source_insert_layers",
]

_LAYOUTS = ("full", "portrait")
_CORNERS = ("auto", "top_right", "bottom_right", "top_left", "bottom_left")
_MAX_UPSCALE = 2.0
_PAPER_BASE = (226, 214, 190)
_PAPER_INK = (58, 44, 32)
_DARK_INK = (237, 223, 193)
_SHRINK = 0.92
_EPS = 1e-6
# reference geometry
_TOP = 20
_SIDE = 140
_GAP_ABOVE_SUBTITLE = 20
_CITE_MARGIN = 64
_PORTRAIT_X = 290
_COLUMN_GAP = 70
# (default px, floor px) of the portrait text blocks
_TITLE_PX = (96, 40)
_CAPTION_PX = (56, 24)
_NOTE_PX = (32, 16)


# --------------------------------------------------------------------------- #
# Citation.
# --------------------------------------------------------------------------- #


def citation_image(
    text, *, font=None, px=12, color=(230, 225, 213), stroke=(20, 20, 20, 210)
):
    """Draw a small citation line, trimmed to its text.

    Parameters
    ----------
    text : str
        The citation, for example ``"Wellcome Collection, CC BY"``.
    font : str, optional
        Font file path; ``None`` uses Pillow's default font.
    px : float, optional
        Text size in pixels. 12 is the reference size at 1080p; scale it with
        the composition.
    color : sequence of int, optional
        Fill colour, RGB.
    stroke : sequence of int, optional
        Outline colour, RGB or RGBA; a 1 px outline keeps small text readable
        on any picture.

    Returns
    -------
    PIL.Image.Image
        ``RGBA`` image exactly as large as the text and its outline.

    Raises
    ------
    ValueError
        If ``text`` is empty or ``px`` is not positive.

    Examples
    --------
    >>> img = citation_image("Source: Museum", px=12)
    >>> img.height < 30 and img.width > img.height
    True
    >>> citation_image("  ")
    Traceback (most recent call last):
    ...
    ValueError: citation text must not be empty
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("citation text must not be empty")
    px = float(px)
    if not math.isfinite(px) or px <= 0:
        raise ValueError("px must be positive")
    fill = _color(color, "color", 3)
    outline = _color(stroke, "stroke", 3, 4)
    if len(outline) == 3:
        outline += (255,)
    f = load_font(font, max(1, round(px)))
    width = 1 if px < 24 else round(px / 12)
    text = text.strip()
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
        (0, 0), text, font=f, stroke_width=width
    )
    img = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
    ImageDraw.Draw(img).text(
        (-box[0], -box[1]),
        text,
        font=f,
        fill=(*fill, 255),
        stroke_width=width,
        stroke_fill=outline,
    )
    return img.crop(img.getbbox() or (0, 0, 1, 1))  # trim to the inked pixels


# --------------------------------------------------------------------------- #
# Spec.
# --------------------------------------------------------------------------- #


def _color(value, name, *counts):
    try:
        out = tuple(int(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a sequence of 0-255 integers") from None
    if len(out) not in counts or any(not 0 <= v <= 255 for v in out):
        raise ValueError(f"{name} must be {' or '.join(map(str, counts))} 0-255 ints")
    return out


def _seconds(value, name, low, strict_low=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or (value <= low if strict_low else value < low):
        raise ValueError(f"{name} must be {'>' if strict_low else '>='} {low:g}")
    return value


@dataclass(frozen=True)
class SourceInsert:
    """One historical image insert with its citation.

    Attributes
    ----------
    image : str
        Path of the source image; it is only ever resized, never edited.
    start, duration : float
        Appearance time and length in seconds.
    citation : str
        Short source line set in a corner, required.
    title, caption, note : str
        Right-column texts of the ``"portrait"`` layout (large, medium, small).
        Ignored by ``"full"``.
    layout : {"full", "portrait"}
        ``"full"`` fits the whole image centred above the subtitle safe rect;
        ``"portrait"`` puts it left at native size with a text column.
    background : str
        ``"paper"`` (paper texture, dark ink), ``"dark"`` (subtle vignette) or
        the path of a background picture.
    fade : float
        Fade-in and fade-out length in seconds.
    citation_corner : {"auto", "top_right", "bottom_right", "top_left", "bottom_left"}
        ``"auto"`` picks the quietest corner that clears subtitles and image.
    sha256 : str
        Optional hex digest of the image file, checked when rendering.

    Examples
    --------
    >>> SourceInsert("a.png", 0, 4, "Museum").layout
    'full'
    >>> SourceInsert("a.png", 0, 0.5, "Museum")
    Traceback (most recent call last):
    ...
    ValueError: duration must exceed twice the fade (0.8 s)
    """

    image: str
    start: float
    duration: float
    citation: str
    title: str = ""
    caption: str = ""
    note: str = ""
    layout: str = "full"
    background: str = "paper"
    fade: float = 0.4
    citation_corner: str = "auto"
    sha256: str = ""

    def __post_init__(self):
        sset = object.__setattr__
        if not isinstance(self.image, (str, os.PathLike)) or not str(self.image):
            raise ValueError("image must be a path")
        sset(self, "image", os.fspath(self.image))
        sset(self, "start", _seconds(self.start, "start", 0.0))
        sset(self, "duration", _seconds(self.duration, "duration", 0.0, True))
        sset(self, "fade", _seconds(self.fade, "fade", 0.0))
        if not isinstance(self.citation, str) or not self.citation.strip():
            raise ValueError("citation must be a non-empty string")
        sset(self, "citation", self.citation.strip())
        for key in ("title", "caption", "note"):
            value = getattr(self, key)
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            sset(self, key, value.strip())
        if self.layout not in _LAYOUTS:
            raise ValueError(f"layout must be one of {_LAYOUTS}, got {self.layout!r}")
        if self.citation_corner not in _CORNERS:
            raise ValueError(f"citation_corner must be one of {_CORNERS}")
        if not isinstance(self.background, (str, os.PathLike)) or not str(
            self.background
        ):
            raise ValueError("background must be 'paper', 'dark' or an image path")
        sset(self, "background", os.fspath(self.background))
        if self.duration <= 2 * self.fade:
            raise ValueError(
                f"duration must exceed twice the fade ({2 * self.fade:g} s)"
            )
        if not isinstance(self.sha256, str):
            raise ValueError("sha256 must be a hex string")
        sset(self, "sha256", self.sha256.strip().lower())

    @property
    def end(self):
        """Time in seconds when the insert has fully faded out."""
        return self.start + self.duration

    def to_dict(self):
        """Return a JSON-friendly dict that ``from_dict`` reads back."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data):
        """Build an insert from a dict such as ``json.load`` returns.

        Raises
        ------
        ValueError
            On unknown keys or invalid values.
        """
        if not isinstance(data, dict):
            raise ValueError("a source insert config must be a mapping")
        known = {item.name for item in fields(cls)}
        extra = sorted(set(data) - known)
        if extra:
            raise ValueError(f"unknown source insert keys: {extra}")
        return cls(**data)


# --------------------------------------------------------------------------- #
# Frame composition.
# --------------------------------------------------------------------------- #


def _canvas(size):
    try:
        w, h = (int(v) for v in size)
    except (TypeError, ValueError):
        raise ValueError("size must be (w, h) integers") from None
    if w <= 0 or h <= 0:
        raise ValueError("size must be positive")
    return w, h, min(w / REFERENCE_SIZE[0], h / REFERENCE_SIZE[1])


def _hit(a, b):
    return (
        a[0] < b[0] + b[2]
        and b[0] < a[0] + a[2]
        and a[1] < b[1] + b[3]
        and b[1] < a[1] + a[3]
    )


def _load_source(insert):
    path = insert.image
    if not os.path.isfile(path):
        raise FileNotFoundError(f"source image not found: {path}")
    if insert.sha256:
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != insert.sha256:
            raise ValueError(
                f"source image {path} does not match its sha256 "
                f"(expected {insert.sha256[:12]}..., got {digest[:12]}...)"
            )
    with Image.open(path) as opened:
        return opened.convert("RGBA")


@lru_cache(maxsize=4)
def _vignette(w, h):
    """Subtle dark vignette with faint banding, after the r2b preview wall."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    v = np.clip(1 - ((x - w * 0.47) / w) ** 2 - ((y - h * 0.45) / h) ** 2, 0, 1)
    band = 3 * np.sin(x * (0.015 * 2400 / w)) + 2 * np.cos(y * (0.029 * 1350 / h))
    arr = np.stack(
        [34 + 25 * v + band, 37 + 19 * v + band, 35 + 10 * v + band], axis=-1
    )
    out = np.clip(arr, 0, 255).astype(np.uint8)
    out.flags.writeable = False
    return out


def _background(insert, w, h):
    """Return ``(RGB Image, ink colour)`` for the insert background."""
    kind = insert.background
    if kind == "paper":
        return Image.fromarray(np.array(paper_texture(w, h, _PAPER_BASE))), _PAPER_INK
    if kind == "dark":
        return Image.fromarray(np.array(_vignette(w, h))), _DARK_INK
    if not os.path.isfile(kind):
        raise FileNotFoundError(f"background image not found: {kind}")
    with Image.open(kind) as opened:
        pic = opened.convert("RGB")
    k = max(w / pic.width, h / pic.height)  # cover; a backdrop may be cropped
    pic = pic.resize(
        (max(w, round(pic.width * k)), max(h, round(pic.height * k))), Image.LANCZOS
    )
    left, top = (pic.width - w) // 2, (pic.height - h) // 2
    return pic.crop((left, top, left + w, top + h)), _DARK_INK


def _fit(iw, ih, box_w, box_h, cap):
    """Largest uniform scale of ``iw x ih`` inside the box, at most ``cap``."""
    return min(box_w / iw, box_h / ih, cap)


def _resized(image, scale):
    if abs(scale - 1.0) < _EPS:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def _wrap(text, font, width):
    """Greedy wrap: CJK breaks anywhere, Latin prefers spaces."""
    lines = []
    for para in text.split("\n"):
        line = ""
        for ch in para:
            if font.getlength(line + ch) <= width or not line:
                line += ch
                continue
            cut = line.rfind(" ")
            if ch != " " and cut > 0:
                lines.append(line[:cut].rstrip())
                line = line[cut + 1 :] + ch
            else:
                lines.append(line.rstrip())
                line = ch.lstrip()
        lines.append(line.rstrip())
    return lines


def _text_block(blocks, width, height, scale, gap_px):
    """Wrap and size ``(text, font_path, (px, floor), colour)`` blocks to fit.

    All blocks shrink together by one factor so their proportions stay.
    Returns ``(drawables, total_height)``; raises ``ValueError`` when the text
    does not fit even with every block at its floor size.
    """
    blocks = [b for b in blocks if b[0]]
    if not blocks:
        return [], 0
    factor = 1.0
    while True:
        rows, total, floored = [], 0, True
        for text, path, (px, floor), color in blocks:
            size = max(round(floor * scale), round(px * scale * factor))
            floored = floored and size <= round(floor * scale)
            font = load_font(path, size)
            lines = _wrap(text, font, width)
            asc, desc = font.getmetrics()
            line_h = round((asc + desc) * 1.12)
            rows.append((lines, font, line_h, color))
            total += line_h * len(lines)
        total += gap_px * (len(rows) - 1)
        too_wide = any(
            font.getlength(line) > width + _EPS
            for lines, font, _, _ in rows
            for line in lines
        )
        if total <= height and not too_wide:
            return rows, total
        if floored:
            raise ValueError(
                f"source insert text needs {total}px height / fits width "
                f"{not too_wide} in a {width}x{height}px column even at the "
                "smallest size; shorten title/caption/note"
            )
        factor *= _SHRINK


def _compose(insert, size, font, title_font):
    """Paint the full frame; return ``(frame, image_rect, ink)``."""
    W, H, s = _canvas(size)
    source = _load_source(insert)
    frame, ink = _background(insert, W, H)
    frame = frame.convert("RGBA")
    safe_top = SUBTITLE_SAFE_RECT[1] * s
    if insert.layout == "full":
        box_w = W - 2 * _SIDE * s
        box_h = safe_top - _GAP_ABOVE_SUBTITLE * s - _TOP * s
        scale = _fit(source.width, source.height, box_w, box_h, _MAX_UPSCALE)
        img = _resized(source, scale)
        x = round((W - img.width) / 2)
        y = round(_TOP * s + (box_h - img.height) / 2)
        texts = None
    else:
        col_x = _PORTRAIT_X * s
        # the image takes up to half the frame; the text column gets the rest
        box_w = W * 0.38
        box_h = safe_top - _GAP_ABOVE_SUBTITLE * s - 2 * _TOP * s
        scale = _fit(source.width, source.height, box_w, box_h, min(s, _MAX_UPSCALE))
        img = _resized(source, scale)
        x = round(min(col_x, W - _SIDE * s - img.width))
        y = round(_TOP * s + (box_h - img.height) / 2 + _TOP * s)
        texts = (x + img.width + _COLUMN_GAP * s, W - _SIDE * s)
    frame.alpha_composite(img, (x, y))
    rect = (x, y, img.width, img.height)
    if texts is not None:
        left, right = texts
        blocks = [
            (
                insert.title,
                title_font if title_font is not None else font,
                _TITLE_PX,
                ink,
            ),
            (insert.caption, font, _CAPTION_PX, ink),
            (insert.note, font, _NOTE_PX, tuple(round(c * 0.8) for c in ink)),
        ]
        rows, total = _text_block(
            blocks,
            max(1, int(right - left)),
            max(1, int(box_h)),
            s,
            round(28 * s),
        )
        draw = ImageDraw.Draw(frame)
        # Centred on the picture, but a small scan may carry a taller column.
        top = 2 * _TOP * s
        cy = min(max(y + (img.height - total) / 2, top), top + box_h - total)
        for lines, fnt, line_h, color in rows:
            for line in lines:
                draw.text((left, cy), line, font=fnt, fill=(*color, 255))
                cy += line_h
            cy += round(28 * s)
    return frame.convert("RGB"), rect, ink


def source_insert_image(insert, size, *, font=None, title_font=None):
    """Compose the full-frame picture of a source insert.

    Parameters
    ----------
    insert : SourceInsert
        Content and look.
    size : tuple of int
        Composition ``(width, height)``.
    font : str, optional
        Font path for caption, note (and title when ``title_font`` is None);
        ``None`` uses Pillow's default font.
    title_font : str, optional
        Font path of the portrait title.

    Returns
    -------
    PIL.Image.Image
        ``RGB`` image of exactly ``size``. The source image is placed as a
        resized (never cropped, never edited) copy; at scale 1 its pixels are
        identical to the file's.

    Raises
    ------
    FileNotFoundError
        If the image or a background picture does not exist.
    ValueError
        On a ``sha256`` mismatch, or text that does not fit the portrait column
        even at its smallest size.
    """
    return _compose(_check(insert), size, font, title_font)[0]


def _check(insert):
    if not isinstance(insert, SourceInsert):
        raise TypeError("insert must be a SourceInsert")
    return insert


# --------------------------------------------------------------------------- #
# Layers.
# --------------------------------------------------------------------------- #


def _corner_rect(corner, size, cite, margin):
    W, H = size
    w, h = cite
    x = margin if corner.endswith("left") else W - margin - w
    y = margin if corner.startswith("top") else H - margin - h
    return (round(x), round(y), w, h)


def _pick_corner(insert, frame, image_rect, cite_size, s):
    """Choose the corner for the citation; return ``(corner, rect)``."""
    W, H = frame.size
    margin = _CITE_MARGIN * s
    safe = tuple(v * s for v in SUBTITLE_SAFE_RECT)
    if insert.citation_corner != "auto":
        corner = insert.citation_corner
        rect = _corner_rect(corner, (W, H), cite_size, margin)
        if _hit(rect, safe):
            raise ValueError(
                f"citation corner {corner!r} overlaps the subtitle safe rect"
            )
        return corner, rect
    gray = np.asarray(frame.convert("L"), np.float32)
    clear, free = [], []
    for corner in _CORNERS[1:]:
        rect = _corner_rect(corner, (W, H), cite_size, margin)
        if _hit(rect, safe):
            continue
        x, y, w, h = rect
        patch = gray[max(0, y) : y + h, max(0, x) : x + w]
        score = float(patch.mean() + patch.std()) if patch.size else 0.0
        (free if _hit(rect, image_rect) else clear).append((score, corner, rect))
    pool = clear or free  # the image may cover a corner only as a last resort
    if not pool:
        raise ValueError(
            "no corner is clear of the subtitle safe rect for the citation"
        )
    _, corner, rect = min(pool, key=lambda item: item[0])
    return corner, rect


def _fade(fade, duration):
    """Opacity on the layer clock: 0, 100 after ``fade``, 0 at the end."""
    if fade <= 0:
        rows = [(0.0, 100.0), (duration, 100.0)]
    else:
        rows = [(0.0, 0.0), (fade, 100.0), (duration - fade, 100.0), (duration, 0.0)]
    keys = [Keyframe(t, v) for t, v in rows]
    return Property(0.0, keyframes=keys)


def _layer(image, name, xy, insert):
    duration = insert.duration
    return CachedAVLayer(
        rgba_still(np.asarray(image.convert("RGBA")), duration),
        name,
        transform=Transform(
            anchor_point=(0.0, 0.0),
            position=(float(xy[0]), float(xy[1])),
            opacity=_fade(insert.fade, duration),  # layer clock: t - start_time
            interpolation="linear",
        ),
        in_point=insert.start,
        out_point=insert.end,
        start_time=insert.start,
    )


def source_insert_layers(comp, insert, *, font=None, title_font=None):
    """Add the picture layer and citation layer of one source insert.

    Parameters
    ----------
    comp : Composition
        Target; its size is the canvas.
    insert : SourceInsert
        Content and timing.
    font, title_font : str, optional
        Font paths, see ``source_insert_image``.

    Returns
    -------
    list of CachedAVLayer
        ``[picture, citation]``, lowest first, already added to ``comp``.
        Both fade on the layer clock over ``[insert.start, insert.end]``.

    Raises
    ------
    FileNotFoundError, ValueError
        See ``source_insert_image``; also when an explicit citation corner
        overlaps the subtitle safe rect.
    """
    layers = _build(_check(insert), comp.size, font, title_font)
    for layer in layers:
        comp.add_layer(layer)
    return layers


def _build(insert, size, font, title_font):
    W, H, s = _canvas(size)
    frame, image_rect, ink = _compose(insert, size, font, title_font)
    look = {}
    if insert.background == "paper":  # light text would vanish on paper
        look = {"color": ink, "stroke": (*_PAPER_BASE, 200)}
    cite = citation_image(insert.citation, font=font, px=max(6.0, 12 * s), **look)
    _, rect = _pick_corner(insert, frame, image_rect, cite.size, s)
    return [
        _layer(frame, f"{insert.title or 'source'} insert", (0, 0), insert),
        _layer(cite, "source citation", rect[:2], insert),
    ]


def add_source_inserts(comp, inserts, **kw):
    """Add several source inserts to ``comp``.

    Parameters
    ----------
    comp : Composition
        Target composition.
    inserts : iterable of SourceInsert or dict
        Inserts (dicts go through ``SourceInsert.from_dict``).
    ``**kw``
        ``font`` and ``title_font``, passed to ``source_insert_layers``.

    Returns
    -------
    list of CachedAVLayer
        All added layers. Nothing is added when any insert is invalid.
    """
    unknown = set(kw) - {"font", "title_font"}
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
    items = [SourceInsert.from_dict(i) if isinstance(i, dict) else i for i in inserts]
    layers = [
        layer
        for item in items
        for layer in _build(
            _check(item), comp.size, kw.get("font"), kw.get("title_font")
        )
    ]
    for layer in layers:
        comp.add_layer(layer)
    return layers
