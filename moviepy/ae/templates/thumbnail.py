"""YouTube thumbnail generator for the music channel.

``render_thumbnail`` cover-crops a background still, applies a subtle vignette
and a translucent panel, then lays out trilingual titles (zh-TW, en, ja), an
optional subtitle and an optional duration badge. Text is shrunk until it fits
its box and is never truncated; when it cannot fit even at the minimum scale,
``ThumbnailFitError`` is raised.

The duration badge sits in the top-right corner. YouTube draws its own
timestamp over the bottom-right corner of a thumbnail, so nothing the channel
needs to read is placed there.

Defaults trace to the music channel sources: the panel color ``#03141A`` and
the 1280x720 canvas come from ``build-season-04-assets.py`` (text composited
over a master image in Pillow), the translucent title panel and the
top-right timer anchor come from ``STUDY-MODE.json`` (season 14), and the
trilingual font split (Microsoft JhengHei for zh/en, Yu Gothic for ja) comes
from the same study config. The contrast check uses WCAG relative luminance on
the composited pixels, because the channel's style rules favour dark, calm
panels and no flashing.

Examples
--------
>>> format_duration(5220)
'1:27:00'
>>> format_duration("87 min")
'87 min'
>>> spec = ThumbnailSpec(titles={"zh": "水晶的聲音", "en": "Crystal Rain"})
>>> spec.layout, spec.panel_opacity
('left_panel', 0.35)
"""

import io
import math
import numbers
import re
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from moviepy.ae.motion.captions import check_glyphs
from moviepy.ae.templates.background import (
    VIDEO_SUFFIXES,
    _opaque_rgb,
    contrast_report,
    cover_crop,
)


DEFAULT_FONTS = {
    "zh": "C:/Windows/Fonts/msjh.ttc",
    "en": "C:/Windows/Fonts/msjh.ttc",
    "ja": "C:/Windows/Fonts/YuGothM.ttc",
}
LAYOUTS = ("left_panel", "bottom_band", "center")
TITLE_LANGS = ("zh", "en", "ja")
BASE_SIZES = {"zh": 96, "en": 44, "ja": 44, "subtitle": 34}
BADGE_SIZE = 34
MIN_SCALE = 0.4
SCALE_STEPS = 31
MARGIN = 0.05
LINE_HEIGHT = 1.2
BLOCK_GAP = 0.35
MAIN_CONTRAST = 4.5
MIN_SIZE = (320, 180)
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


AUTO_PANEL_STEPS = (0.45, 0.55, 0.65, 0.75, 0.85)


class ThumbnailError(ValueError):
    """Base error for thumbnail rendering problems."""


class ThumbnailFitError(ThumbnailError):
    """Raised when trilingual text cannot fit its box at the minimum scale."""


class ContrastError(ThumbnailError):
    """Raised when the main title misses the contrast target and it is required."""


def format_duration(value):
    """Return the badge text for a duration given as text or seconds.

    Parameters
    ----------
    value : str or int or float
        Text such as ``"87 min"`` (returned stripped), or seconds, formatted as
        ``H:MM:SS`` when an hour or more and ``M:SS`` otherwise.

    Returns
    -------
    str
        The badge text.

    Examples
    --------
    >>> format_duration(640)
    '10:40'
    >>> format_duration(36000)
    '10:00:00'
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("duration_badge must not be empty")
        return text
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError("duration_badge must be text or a number of seconds")
    if not math.isfinite(value) or value < 0:
        raise ValueError("duration_badge seconds must be finite and non-negative")
    total = int(round(value))
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return float(value)


def _rgb(hex_color):
    return tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))


@dataclass
class ThumbnailSpec:
    """Text, layout and style for one thumbnail.

    Parameters
    ----------
    titles : dict
        Keys ``"zh"`` (required), ``"en"`` and ``"ja"``, each non-empty text.
    subtitle : str, optional
        Optional line drawn under the titles in ``accent``.
    duration_badge : str or int or float, optional
        Badge text, or seconds formatted by ``format_duration``.
    layout : str, optional
        ``"left_panel"`` (default), ``"bottom_band"`` or ``"center"``.
    accent : str, optional
        ``#RRGGBB`` color for the subtitle. Default ``"#7FD6C7"``.
    text_color : str, optional
        ``#RRGGBB`` color for the titles. Default ``"#FFFFFF"``.
    panel_color : str, optional
        ``#RRGGBB`` panel color. Default ``"#03141A"`` from the S04 shade.
    fonts : dict, optional
        Explicit font paths per language; missing keys use ``DEFAULT_FONTS``.
    panel_opacity : float, optional
        Panel opacity in ``[0, 1]``. Default ``0.35``.
    auto_panel : bool, optional
        Darken the panel in ``AUTO_PANEL_STEPS`` until the main title reaches
        4.5:1 contrast. Default ``True``.
    vignette : float, optional
        Vignette strength in ``[0, 1]``. Default ``0.25``.

    Examples
    --------
    >>> spec = ThumbnailSpec(titles={"zh": "雨聲", "ja": "雨の音"}, duration_badge=3600)
    >>> spec.duration_badge
    '1:00:00'
    """

    titles: Dict[str, str]
    subtitle: Optional[str] = None
    duration_badge: Optional[Union[str, int, float]] = None
    layout: str = "left_panel"
    accent: str = "#7FD6C7"
    text_color: str = "#FFFFFF"
    panel_color: str = "#03141A"
    fonts: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FONTS))
    panel_opacity: float = 0.35
    auto_panel: bool = True
    vignette: float = 0.25

    def __post_init__(self):
        """Validate every field so that rendering never sees bad input."""
        titles = {}
        for lang, text in dict(self.titles).items():
            if lang not in TITLE_LANGS:
                raise ValueError(f"unknown title language {lang!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"titles[{lang!r}] must be non-empty text")
            titles[lang] = text.strip()
        if "zh" not in titles:
            raise ValueError("titles['zh'] is required")
        self.titles = titles
        if self.subtitle is not None:
            if not isinstance(self.subtitle, str) or not self.subtitle.strip():
                raise ValueError("subtitle must be non-empty text when given")
            self.subtitle = self.subtitle.strip()
        if self.duration_badge is not None:
            self.duration_badge = format_duration(self.duration_badge)
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        for name in ("accent", "text_color", "panel_color"):
            if not _HEX.match(str(getattr(self, name))):
                raise ValueError(f"{name} must be #RRGGBB")
        unknown = set(self.fonts) - set(TITLE_LANGS)
        if unknown:
            raise ValueError(f"unknown font languages {sorted(unknown)}")
        self.fonts = {**DEFAULT_FONTS, **dict(self.fonts)}
        self.panel_opacity = _unit(self.panel_opacity, "panel_opacity")
        if not isinstance(self.auto_panel, bool):
            raise ThumbnailError("auto_panel must be true or false")
        self.vignette = _unit(self.vignette, "vignette")


class _Fonts:
    """Load and cache Pillow fonts per language and size."""

    def __init__(self, paths):
        self.paths = paths
        self._cache = {}

    def get(self, lang, size):
        key = (lang, size)
        if key not in self._cache:
            path = self.paths[lang]
            self._cache[key] = ImageFont.truetype(str(path), size, index=0)
        return self._cache[key]


def _required_fonts(spec):
    langs = {"zh"}
    langs.update(spec.titles)
    if spec.duration_badge is not None:
        langs.add("en")
    paths = {}
    for lang in sorted(langs):
        path = Path(spec.fonts[lang])
        if not path.is_file():
            raise FileNotFoundError(f"font for {lang!r} not found: {path}")
        paths[lang] = path
    return paths


def _wrap_paragraph(text, font, width):
    """Greedy wrap by words, falling back to characters; None if a glyph is too wide."""
    lines = []
    current = ""
    for word in [part for part in text.split(" ") if part]:
        candidate = word if not current else current + " " + word
        if font.getlength(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        piece = ""
        for char in word:
            trial = piece + char
            if font.getlength(trial) <= width:
                piece = trial
                continue
            if not piece or font.getlength(char) > width:
                return None
            lines.append(piece)
            piece = char
        current = piece
    if current:
        lines.append(current)
    return lines


def _wrap(text, font, width):
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            continue
        wrapped = _wrap_paragraph(paragraph.strip(), font, width)
        if wrapped is None:
            return None
        lines.extend(wrapped)
    return lines


def _layout_blocks(blocks, box, align, fonts, scale):
    """Place every block at ``scale``; return None if anything leaves ``box``."""
    bx0, by0, bx1, by1 = box
    bw, bh = bx1 - bx0, by1 - by0
    plans = []
    total = 0.0
    for index, block in enumerate(blocks):
        size = max(1, int(round(block["base"] * scale)))
        font = fonts.get(block["lang"], size)
        lines = _wrap(block["text"], font, bw)
        if not lines:
            return None
        step = size * LINE_HEIGHT
        if index:
            total += BLOCK_GAP * size
        total += step * len(lines)
        plans.append((block, font, size, lines, step))
    if total > bh:
        return None
    y = by0 + (bh - total) / 2.0
    placed = []
    for index, (block, font, size, lines, step) in enumerate(plans):
        if index:
            y += BLOCK_GAP * size
        rows = []
        for text in lines:
            width = font.getlength(text)
            x = bx0 if align == "left" else bx0 + (bw - width) / 2.0
            left, top, right, bottom = font.getbbox(text)
            ink = (x + left, y + top, x + right, y + bottom)
            if ink[0] < bx0 or ink[2] > bx1 or ink[1] < by0 or ink[3] > by1:
                return None
            rows.append({"text": text, "x": x, "y": y, "ink": ink})
            y += step
        ink_union = [
            min(r["ink"][0] for r in rows),
            min(r["ink"][1] for r in rows),
            max(r["ink"][2] for r in rows),
            max(r["ink"][3] for r in rows),
        ]
        placed.append(
            {
                "role": block["role"],
                "lang": block["lang"],
                "color": block["color"],
                "font": font,
                "font_size": size,
                "rows": rows,
                "box": ink_union,
            }
        )
    return placed


def _fit(blocks, box, align, fonts):
    """Return ``(placed, scale)`` for the largest scale that fits, else raise."""
    for scale in np.linspace(1.0, MIN_SCALE, SCALE_STEPS):
        placed = _layout_blocks(blocks, box, align, fonts, float(scale))
        if placed is not None:
            return placed, float(scale)
    raise ThumbnailFitError(
        "text cannot fit its box even at the minimum scale "
        f"{MIN_SCALE}; shorten the titles or use a larger layout"
    )


def _geometry(layout, size):
    """Return panel rect, text box, text alignment and the safe area."""
    width, height = size
    safe = (
        width * MARGIN,
        height * MARGIN,
        width * (1 - MARGIN),
        height * (1 - MARGIN),
    )
    if layout == "left_panel":
        panel = (0, 0, round(width * 0.56), height)
        box = (safe[0], safe[1], panel[2] - width * MARGIN, safe[3])
        align = "left"
    elif layout == "bottom_band":
        band_top = round(height * 0.56)
        panel = (0, band_top, width, height)
        box = (safe[0], band_top + height * MARGIN, safe[2], safe[3])
        align = "left"
    else:
        panel = (
            round(width * 0.2),
            round(height * 0.16),
            round(width * 0.8),
            round(height * 0.84),
        )
        box = (
            panel[0] + width * MARGIN,
            panel[1] + height * MARGIN,
            panel[2] - width * MARGIN,
            panel[3] - height * MARGIN,
        )
        align = "center"
    return panel, box, align, safe


def _cover_rgb(array, size):
    source = Image.fromarray(array)
    box = cover_crop(source.size, size)
    return source.resize(size, Image.Resampling.LANCZOS, box=box)


def _apply_vignette(rgb, strength):
    if strength <= 0:
        return rgb
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx + 0.5) / width * 2 - 1
    ny = (yy + 0.5) / height * 2 - 1
    radius = np.sqrt(nx**2 + ny**2) / math.sqrt(2)
    falloff = np.clip((radius - 0.45) / 0.55, 0, 1) ** 2
    factor = 1 - strength * 0.7 * falloff
    return np.clip(np.rint(rgb * factor[..., None]), 0, 255).astype(np.uint8)


def _main_contrast(rgb, box, text_color, require):
    """Return the WCAG contrast evidence for the main title; raise if required."""
    result = contrast_report(rgb, box, text_color, target=MAIN_CONTRAST)
    passes = not result["needs_visual_review"]
    result["passes"] = passes
    result["required"] = bool(require)
    if require and not passes:
        raise ContrastError(
            "main title contrast "
            f"{result['minimum_estimated_contrast']:.2f}:1 is below {MAIN_CONTRAST}:1"
        )
    return result


def _output_format(output):
    if output is None:
        return None
    suffix = Path(output).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    raise ValueError("output must end in .jpg, .jpeg or .png")


def _encode(image, fmt, max_bytes):
    """Return ``(data, quality, mode)``; JPEG quality steps down until it fits."""
    if fmt == "PNG":
        palette = image.quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.FLOYDSTEINBERG,
        )
        for mode, candidate in (("rgb", image), ("palette256", palette)):
            buffer = io.BytesIO()
            candidate.save(buffer, format="PNG", optimize=True, compress_level=9)
            if len(buffer.getvalue()) <= max_bytes:
                return buffer.getvalue(), None, mode
        raise ThumbnailError(f"PNG exceeds max_bytes={max_bytes} even at 256 colors")
    for quality in range(92, 39, -4):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, subsampling=0)
        if len(buffer.getvalue()) <= max_bytes:
            return buffer.getvalue(), quality, "rgb"
    raise ThumbnailError(f"JPEG exceeds max_bytes={max_bytes} even at quality 40")


def _no_time(time):
    if time is not None:
        raise ValueError("time applies only to video sources")


def _grab_video(path, time):
    from moviepy import VideoFileClip

    with VideoFileClip(str(path), audio=False) as clip:
        duration = float(clip.duration or 0.0)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("video has no finite positive duration")
        fps = float(clip.fps or 24.0)
        when = duration / 3.0 if time is None else float(time)
        if not math.isfinite(when) or when < 0 or when >= duration:
            raise ValueError(f"time {when} must lie within [0, {duration:.3f}s)")
        when = max(0.0, min(when, duration - 0.5 / fps))
        frame = clip.get_frame(when)
    return _opaque_rgb(np.array(frame), "video frame")


def grab_frame(source, time=None):
    """Return one RGB ``uint8`` frame from an image, a video or an array.

    Parameters
    ----------
    source : str or path-like or numpy.ndarray or PIL.Image.Image
        An image file, a video file (decoded with MoviePy's FFmpeg), or pixels.
    time : float, optional
        Video only. Seconds to decode. Defaults to one third of the duration.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8 RGB pixels.

    Raises
    ------
    FileNotFoundError
        If a path does not exist.
    ValueError
        If ``time`` is given for a still, or lies outside the video.

    Examples
    --------
    >>> import numpy as np
    >>> grab_frame(np.full((4, 6, 3), 200, np.uint8)).shape
    (4, 6, 3)
    """
    if isinstance(source, Image.Image):
        _no_time(time)
        return np.array(source.convert("RGB"))
    if isinstance(source, np.ndarray):
        _no_time(time)
        return _opaque_rgb(source, "array")
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"no such source: {path}")
    if path.suffix.lower() in VIDEO_SUFFIXES:
        return _grab_video(path, time)
    _no_time(time)
    try:
        with Image.open(path) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("animated images are unsupported; give a still")
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"cannot decode image {path.name}") from error
    return np.array(image)


def _to_background(background):
    if isinstance(background, (str, Path)):
        return grab_frame(background), str(background)
    if isinstance(background, Image.Image):
        return np.array(background.convert("RGB")), "image"
    return _opaque_rgb(np.asarray(background), "background"), "array"


def _render_thumbnail_once(
    spec,
    background,
    output=None,
    *,
    size=(1280, 720),
    max_bytes=2_000_000,
    overwrite=False,
    require_contrast=True,
):
    """Render a thumbnail and optionally write it (exclusive create by default).

    Parameters
    ----------
    spec : ThumbnailSpec
        Text, layout and style.
    background : str or path-like or numpy.ndarray or PIL.Image.Image
        Still image, video (one frame via ``grab_frame``) or pixels.
    output : str or path-like, optional
        ``.jpg``/``.jpeg`` (JPEG, quality stepped down to fit) or ``.png``.
        When ``None`` nothing is written; the size is measured as JPEG.
    size : tuple of int, optional
        Canvas ``(width, height)``. Default ``(1280, 720)``.
    max_bytes : int, optional
        Upper bound on the encoded size. Default ``2_000_000``.
    overwrite : bool, optional
        Replace an existing output file. Default ``False`` (exclusive create).
    require_contrast : bool, optional
        Raise ``ContrastError`` when the main title misses 4.5:1.
        Default ``True``; when ``False`` the evidence records the miss.

    Returns
    -------
    tuple
        ``(PIL.Image.Image, dict)``: the RGB image and an evidence dict with
        size, format, bytes, sha256, text boxes, badge box, safe area, fonts,
        contrast result and ``"human_visual": "NOT_RUN"``.

    Raises
    ------
    FileNotFoundError
        If a required font or the background file is missing, or the output
        exists and ``overwrite`` is false.
    ThumbnailFitError
        If text cannot fit at the minimum scale.
    ContrastError
        If the contrast is required and missed.

    Examples
    --------
    >>> import numpy as np
    >>> spec = ThumbnailSpec(titles={"zh": "雨聲"}, fonts={"zh": "missing.ttc"})
    >>> render_thumbnail(spec, np.zeros((72, 128, 3), np.uint8))  # doctest: +SKIP
    """
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or min(size) < min(MIN_SIZE)
        or size[0] < MIN_SIZE[0]
        or size[1] < MIN_SIZE[1]
    ):
        raise ValueError(f"size must be a (width, height) tuple of at least {MIN_SIZE}")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    fmt = _output_format(output)
    if output is not None and Path(output).exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")
    paths = _required_fonts(spec)
    fonts = _Fonts(paths)
    frame, source_label = _to_background(background)

    width, height = int(size[0]), int(size[1])
    canvas = _cover_rgb(frame, (width, height))
    rgb = _apply_vignette(np.asarray(canvas, dtype=np.float64), spec.vignette)
    base = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")

    panel, text_box, align, safe = _geometry(spec.layout, (width, height))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(
        [panel[0], panel[1], panel[2] - 1, panel[3] - 1],
        fill=(*_rgb(spec.panel_color), round(255 * spec.panel_opacity)),
    )
    composed = Image.alpha_composite(base, overlay)

    blocks = []
    for lang in TITLE_LANGS:
        if lang in spec.titles:
            blocks.append(
                {
                    "role": f"title_{lang}",
                    "lang": lang,
                    "text": spec.titles[lang],
                    "base": BASE_SIZES[lang],
                    "color": spec.text_color,
                }
            )
    if spec.subtitle is not None:
        blocks.append(
            {
                "role": "subtitle",
                "lang": "zh",
                "text": spec.subtitle,
                "base": BASE_SIZES["subtitle"],
                "color": spec.accent,
            }
        )
    # A missing glyph would render as a tofu box silently; refuse instead.
    for block in blocks:
        try:
            check_glyphs(fonts.get(block["lang"], 48), block["text"])
        except ValueError as error:
            raise ThumbnailError(f"{block['role']}: {error}") from None
    placed, scale = _fit(blocks, text_box, align, fonts)

    main = placed[0]
    contrast = _main_contrast(
        np.asarray(composed.convert("RGB")),
        main["box"],
        spec.text_color,
        require_contrast,
    )

    draw = ImageDraw.Draw(composed)
    badge = None
    if spec.duration_badge is not None:
        badge_font = fonts.get("en", BADGE_SIZE)
        left, top, right, bottom = badge_font.getbbox(spec.duration_badge)
        pad_x, pad_y = 18, 10
        text_w, text_h = right - left, bottom - top
        plate = (
            safe[2] - (text_w + 2 * pad_x),
            safe[1],
            safe[2],
            safe[1] + text_h + 2 * pad_y,
        )
        draw.rounded_rectangle(plate, radius=10, fill=(0, 0, 0, 170))
        origin_x = plate[0] + pad_x - left
        origin_y = plate[1] + pad_y - top
        draw.text(
            (origin_x, origin_y),
            spec.duration_badge,
            font=badge_font,
            fill=(*_rgb(spec.accent), 255),
        )
        badge = {
            "text": spec.duration_badge,
            "plate": [round(v, 2) for v in plate],
            "corner": "top_right",
        }

    for block in placed:
        for row in block["rows"]:
            draw.text(
                (row["x"], row["y"]),
                row["text"],
                font=block["font"],
                fill=(*_rgb(block["color"]), 255),
            )
    image = composed.convert("RGB")

    data, quality, mode = _encode(image, fmt or "JPEG", max_bytes)
    evidence = {
        "output": None if output is None else str(output),
        "format": fmt or "JPEG (measured only)",
        "quality": quality,
        "png_mode": mode if fmt == "PNG" else None,
        "size": [width, height],
        "bytes": len(data),
        "max_bytes": max_bytes,
        "sha256": sha256(data).hexdigest(),
        "layout": spec.layout,
        "accent": spec.accent,
        "text_color": spec.text_color,
        "panel": {
            "box": list(panel),
            "color": spec.panel_color,
            "opacity": spec.panel_opacity,
        },
        "vignette": spec.vignette,
        "background_source": source_label,
        "fonts": {lang: str(path) for lang, path in paths.items()},
        "scale": round(scale, 4),
        "safe_area": [round(v, 2) for v in safe],
        "text_boxes": [
            {
                "role": block["role"],
                "lang": block["lang"],
                "font_size": block["font_size"],
                "lines": [row["text"] for row in block["rows"]],
                "box": [round(v, 2) for v in block["box"]],
            }
            for block in placed
        ],
        "badge": badge,
        "contrast": {
            "main_title": contrast,
            "target": MAIN_CONTRAST,
            "passes": contrast["passes"],
        },
        "youtube_timestamp_note": (
            "YouTube overlays its own duration timestamp at the bottom-right; "
            "the badge is kept in the top-right corner"
        ),
        "human_visual": "NOT_RUN",
    }
    if output is not None:
        mode_flag = "wb" if overwrite else "xb"
        with open(output, mode_flag) as handle:
            handle.write(data)
    return image, evidence


def contact_sheet(images, columns=3):
    """Tile images into one preview sheet for comparing variants.

    Parameters
    ----------
    images : list
        PIL images, numpy arrays, or paths readable by ``grab_frame``.
    columns : int, optional
        Cells per row. Default ``3``.

    Returns
    -------
    PIL.Image.Image
        RGB sheet; each cell is 320x180 with the image fitted inside.

    Examples
    --------
    >>> import numpy as np
    >>> sheet = contact_sheet([np.zeros((90, 160, 3), np.uint8)] * 4, columns=2)
    >>> sheet.size
    (664, 384)
    """
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer")
    frames = []
    for item in images:
        if isinstance(item, Image.Image):
            frames.append(item.convert("RGB"))
        else:
            frames.append(Image.fromarray(grab_frame(item)))
    if not frames:
        raise ValueError("images must not be empty")
    cell_w, cell_h, gap = 320, 180, 8
    rows = -(-len(frames) // columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_w + (columns + 1) * gap, rows * cell_h + (rows + 1) * gap),
        (24, 24, 24),
    )
    for index, frame in enumerate(frames):
        fitted = ImageOps.contain(frame, (cell_w, cell_h))
        col, row = index % columns, index // columns
        x = gap + col * (cell_w + gap) + (cell_w - fitted.width) // 2
        y = gap + row * (cell_h + gap) + (cell_h - fitted.height) // 2
        sheet.paste(fitted, (x, y))
    return sheet


def render_thumbnail(spec, background, output=None, **options):
    """Render a thumbnail and optionally write it (exclusive create by default).

    Same parameters and return value as the single-pass renderer. When
    ``spec.auto_panel`` is true and the main title misses the 4.5:1 contrast
    target, the panel is darkened step by step (``AUTO_PANEL_STEPS``) until it
    passes; bright episode backgrounds therefore still yield a readable
    thumbnail. The opacity actually used is recorded in the evidence under
    ``panel["opacity"]`` and ``panel["auto_steps"]``. Nothing is written until
    an attempt passes.

    Examples
    --------
    >>> AUTO_PANEL_STEPS[0] <= AUTO_PANEL_STEPS[-1] <= 0.9
    True
    """
    if not spec.auto_panel or not options.get("require_contrast", True):
        return _render_thumbnail_once(spec, background, output, **options)
    tried = []
    levels = [spec.panel_opacity] + [
        o for o in AUTO_PANEL_STEPS if o > spec.panel_opacity
    ]
    for index, opacity in enumerate(levels):
        attempt = replace(spec, panel_opacity=opacity)
        tried.append(opacity)
        try:
            image, evidence = _render_thumbnail_once(
                attempt, background, output, **options
            )
        except ContrastError:
            if index == len(levels) - 1:
                raise
            continue
        evidence.setdefault("panel", {})
        if isinstance(evidence["panel"], dict):
            evidence["panel"]["auto_steps"] = tried
        return image, evidence
