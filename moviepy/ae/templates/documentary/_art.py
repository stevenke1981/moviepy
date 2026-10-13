"""Static, deterministic artwork for editable documentary compositions.

Only the artwork is rasterized. Footage, masks, camera transforms, shadows and
subtitles remain ordinary AE layers. No reference photograph or font is bundled.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.ae.templates.paper import paper_texture


def font_path(explicit=None, *, cjk=False):
    """Resolve an explicit/system font, failing rather than drawing CJK tofu."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"font not found: {path}")
        ImageFont.truetype(str(path), 20)
        return str(path)
    windir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    candidates = (
        windir / "msjh.ttc",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    )
    if not cjk:
        candidates += (
            windir / "arial.ttf",
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        )
    for path in candidates:
        if path.is_file():
            return str(path)
    if cjk:
        raise FileNotFoundError("CJK text requires --font / a font file path")
    return None


def _font(path, px):
    if path:
        return ImageFont.truetype(str(path), max(1, round(px)))
    try:
        return ImageFont.load_default(size=max(1, round(px)))
    except TypeError:  # Pillow 9.2 supports the templates' minimum dependency.
        return ImageFont.load_default()


def _wrap(text, font, width):
    """Wrap CJK/Latin text by measured glyph advances, preserving newlines."""
    result = []
    for paragraph in text.split("\n"):
        line = ""
        for char in paragraph:
            if line and font.getlength(line + char) > width:
                result.append(line.rstrip())
                line = char.lstrip()
            else:
                line += char
        result.append(line.rstrip())
    return result


def text_image(text, px, *, font=None, color=(243, 241, 235), max_width=None):
    """Return tightly bounded RGBA text with a small dark outline."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    f = _font(font, px)
    stroke = max(1, round(px / 24))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.textbbox((0, 0), text, font=f, stroke_width=stroke)
    if max_width is not None and box[2] - box[0] > max_width:
        raise ValueError("label exceeds its available width; shorten it")
    image = Image.new("RGBA", (box[2] - box[0] + 2, box[3] - box[1] + 2))
    ImageDraw.Draw(image).text(
        (1 - box[0], 1 - box[1]),
        text,
        font=f,
        fill=(*color, 255),
        stroke_width=stroke,
        stroke_fill=(8, 10, 14, 175),
    )
    return np.asarray(image)


def article_art(size, headline, body, source, font, accent):
    """Draw a newspaper/webpage with a blank, independently composited video slot.

    Return ``(RGB artwork, media_box)``. Text overflow raises, never truncates.
    """
    w, h = size
    image = Image.new("RGB", size, (249, 248, 245))
    draw = ImageDraw.Draw(image)
    s = w / 1280
    ink = (24, 26, 30)
    draw.rectangle((0, 0, w - 1, round(h * 0.07)), fill=(237, 237, 235))
    draw.text(
        (w * 0.045, h * 0.021), "ARCHIVE / NEWS", font=_font(font, 18 * s), fill=ink
    )
    draw.rounded_rectangle(
        (w * 0.72, h * 0.019, w * 0.955, h * 0.05),
        radius=max(1, round(8 * s)),
        outline=(172, 174, 173),
        width=max(1, round(s)),
    )
    draw.rectangle((0, h * 0.07, w, h * 0.168), fill=accent)
    navigation = (
        ("新聞", "紀錄", "人物", "影像", "文化", "故事")
        if any(ord(c) > 127 for c in headline)
        else ("NEWS", "ARCHIVE", "PEOPLE", "VIDEO", "CULTURE", "STORIES")
    )
    for i, word in enumerate(navigation):
        draw.text(
            (w * (0.065 + i * 0.156), h * 0.096),
            word,
            font=_font(font, 31 * s),
            fill=(255, 255, 255),
        )
    # Keep a single strong headline row, as in the photographed news webpage.
    for px in range(max(1, round(55 * s)), max(0, round(24 * s) - 1), -1):
        title_font = _font(font, px)
        if title_font.getlength(headline) <= w * 0.91:
            break
    else:
        raise ValueError("headline is too long for the page; shorten it")
    draw.text(
        (w * 0.045, h * 0.195),
        headline,
        font=title_font,
        fill=ink,
        stroke_width=max(1, round(s)),
        stroke_fill=ink,
    )
    source_font = _font(font, 19 * s)
    if source_font.getlength(source) > w * 0.91:
        raise ValueError("source is too long for the page")
    draw.text((w * 0.045, h * 0.291), source, font=source_font, fill=(83, 86, 88))
    box = (round(w * 0.315), round(h * 0.36), round(w * 0.955), round(h * 0.85))
    draw.rectangle(box, fill=(229, 227, 224))
    body_font = _font(font, 22 * s)
    lines = _wrap(body, body_font, w * 0.245)
    line_h = max(1, round(35 * s))
    room = int((box[3] - box[1]) / line_h)
    if len(lines) > room:
        raise ValueError(f"body needs {len(lines)} lines; this page fits {room}")
    for i, line in enumerate(lines):
        draw.text(
            (w * 0.045, box[1] + i * line_h), line, font=body_font, fill=(66, 68, 71)
        )
    draw.line((w * 0.045, h * 0.891, w * 0.955, h * 0.891), fill=(196, 192, 187))
    footer = "DOCUMENTARY ARCHIVE  /  EDITABLE FOOTAGE"
    draw.text(
        (w * 0.045, h * 0.922), footer, font=_font(font, 17 * s), fill=(96, 96, 96)
    )
    return np.asarray(image), box


def textured_background(size, seed, *, dark=False):
    """Bake fixed paper grain and a restrained vignette, without flicker."""
    w, h = size
    base = (28, 32, 41) if dark else (180, 173, 164)
    image = paper_texture(w, h, base, seed=seed, strength=0.3 if dark else 2.5)
    yy, xx = np.ogrid[-1 : 1 : complex(h), -1 : 1 : complex(w)]
    factor = 1 - (0.12 if dark else 0.16) * np.minimum(1, (xx * xx + yy * yy) / 2)
    return np.clip(np.asarray(image, np.float32) * factor[..., None], 0, 255).astype(
        np.uint8
    )


def frame_hud(size, iso, font):
    """Draw the photographed camera UI as transparent artwork (no logos)."""
    w, h = size
    s = min(w / 1920, h / 1080)
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)
    light = (224, 227, 232, 235)
    faint = (138, 151, 173, 145)
    line = max(1, round(2 * s))
    draw.polygon(
        [(w * 0.045, h * 0.047), (w * 0.045, h * 0.086), (w * 0.067, h * 0.066)],
        fill=light,
    )
    draw.line(
        (w * 0.465, h * 0.119, w * 0.535, h * 0.119),
        fill=light,
        width=max(1, round(3 * s)),
    )
    iso_font = _font(font, 23 * s)
    if iso_font.getlength(iso) > w * 0.30:
        raise ValueError("ISO label exceeds the top-right HUD area")
    draw.text(
        (w * 0.922 - iso_font.getlength(iso), h * 0.06), iso, font=iso_font, fill=light
    )
    # Partially cropped perforations at the extreme right are intentional.
    pitch = h * 0.071
    for i in range(-1, 16):
        y = i * pitch + h * 0.018
        draw.rounded_rectangle(
            (w * 0.987, y, w * 1.014, y + h * 0.039),
            radius=max(1, round(5 * s)),
            outline=faint,
            width=line,
        )
    for i in range(6):
        x = w * 0.018 + i * w * 0.003
        draw.line((x, h * 0.856, x, h * 0.938), fill=light, width=max(1, round(s)))
    cx, cy, r = w * 0.934, h * 0.974, 17 * s
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=faint, width=line)
    return np.asarray(image)


def demo_letter(size, font, *, chinese=False, seed=19):
    """Make original paper artwork for the explicit offline demo, not real evidence."""
    w, h = size
    image = Image.fromarray(paper_texture(w, h, (229, 216, 185), seed, 1.6))
    draw = ImageDraw.Draw(image)
    s = w / 1280
    ink = (68, 65, 55)
    draw.line(
        (w * 0.5, 0, w * 0.497, h), fill=(189, 175, 147), width=max(1, round(2 * s))
    )
    draw.line(
        (0, h * 0.53, w, h * 0.535), fill=(184, 170, 144), width=max(1, round(2 * s))
    )
    title = "一封遠方的信" if chinese else "A LETTER FROM FAR AWAY"
    draw.text((w * 0.095, h * 0.115), title, font=_font(font, 54 * s), fill=ink)
    lines = (
        (
            "親愛的朋友：",
            "窗外的光，照在泛黃的信紙上。",
            "那些走過的路，仍留在記憶裡。",
            "願重逢的那一天，我們都還記得。",
        )
        if chinese
        else (
            "Dear friend,",
            "The afternoon light falls across this page.",
            "The roads we travelled remain in memory.",
            "May we meet again, with the stories intact.",
        )
    )
    for i, text in enumerate(lines):
        draw.text(
            (w * 0.10, h * (0.31 + i * 0.092)), text, font=_font(font, 28 * s), fill=ink
        )
    draw.text(
        (w * 0.67, h * 0.73), "1961", font=_font(font, 95 * s), fill=(136, 58, 47)
    )
    draw.rectangle(
        (w * 0.78, h * 0.10, w * 0.91, h * 0.28),
        outline=(132, 71, 60),
        width=max(1, round(3 * s)),
    )
    for i in range(5):
        y = h * (0.15 + i * 0.015)
        draw.line(
            (w * 0.795, y, w * 0.893, y), fill=(144, 85, 71), width=max(1, round(s))
        )
    # These marks explicitly distinguish demonstration artwork from a historic scan.
    draw.text(
        (w * 0.10, h * 0.88),
        "ORIGINAL DEMO ART / NOT A HISTORICAL DOCUMENT",
        font=_font(font, 15 * s),
        fill=(125, 114, 96),
    )
    return np.asarray(image)
