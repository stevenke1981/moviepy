"""Lay out Chinese/Latin cues once, then animate fixed grapheme positions."""

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.ae.text.clusters import clusters


def load_font(request, size):
    """Load the explicitly supplied font or Pillow's Latin fallback."""
    filename = request["assets"].get("font")
    if filename:
        return ImageFont.truetype(filename, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError as error:
        raise ValueError(
            "this Pillow version requires an explicit font file"
        ) from error


def check_glyphs(font, text):
    """Reject missing-glyph boxes instead of silently substituting CJK text."""
    missing = font.getmask("\U0010ffff")
    missing = (missing.size, bytes(missing))
    for unit in clusters(text):
        if unit.strip():
            mask = font.getmask(unit)
            if (mask.size, bytes(mask)) == missing:
                raise ValueError(f"the supplied font lacks a glyph for {unit!r}")


def _wrap(text, font, width):
    lines, line, prefix, index = [], [], "", 0
    for unit in clusters(text.replace("\r\n", "\n")):
        if unit == "\n":
            lines.append(line)
            line, prefix = [], ""
            continue
        if font.getlength(unit) > width:
            return None
        if line and font.getlength(prefix + unit) > width:
            lines.append(line)
            line, prefix = [], ""
        line.append((index, unit, float(font.getlength(prefix))))
        prefix += unit
        index += 1
    lines.append(line)
    return lines


def _glyph_masks(font, unit, stroke):
    bbox = font.getbbox(unit, anchor="ls", stroke_width=stroke)
    size = (max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1]))
    masks = []
    for amount in (stroke, 0):
        image = Image.new("L", size)
        ImageDraw.Draw(image).text(
            (-bbox[0], -bbox[1]),
            unit,
            font=font,
            anchor="ls",
            fill=255,
            stroke_width=amount,
            stroke_fill=255,
        )
        masks.append(np.asarray(image, dtype=np.float32) / 255)
    return bbox, masks[0], masks[1]


def layout_captions(request):
    """Fit cues into their safe area without moving glyphs during animation.

    The input is a normalized motion request. Common grapheme clusters remain
    intact; full script shaping across separately animated clusters is outside
    this Chinese/Latin layout's scope.
    """
    width, height, params = request["width"], request["height"], request["params"]
    margin_x = math.ceil(width * params["safe_margin"])
    margin_y = math.ceil(height * params["safe_margin"])
    safe_width, safe_height = width - 2 * margin_x, height - 2 * margin_y
    pad = params["stroke_width"] + 3
    layouts = []
    for cue in request["cues"]:
        for size in range(params["font_size"], params["min_font_size"] - 1, -1):
            font = load_font(request, size)
            lines = _wrap(cue["text"], font, safe_width - 2 * pad)
            ascent, descent = font.getmetrics()
            line_height = math.ceil((ascent + descent) * params["line_spacing"])
            if (
                lines is not None
                and len(lines) <= params["max_lines"]
                and len(lines) * line_height + 2 * pad <= safe_height
            ):
                break
        else:
            raise ValueError("caption cannot fit the safe area at min_font_size")
        # Compare against FreeType's missing-glyph raster; never silently draw
        # a CJK cue as the same placeholder box repeated for every character.
        check_glyphs(font, cue["text"])
        block_height = len(lines) * line_height + 2 * pad
        top = round(params["position"][1] * height - block_height / 2)
        top = max(margin_y, min(top, height - margin_y - block_height))
        glyphs, boxes = [], []
        for row, line in enumerate(lines):
            line_width = float(font.getlength("".join(unit for _, unit, _ in line)))
            left = round(params["position"][0] * width - line_width / 2)
            left = max(
                margin_x + pad,
                min(left, width - margin_x - pad - math.ceil(line_width)),
            )
            baseline = top + pad + row * line_height + ascent
            for index, unit, x in line:
                bbox, outer, inner = _glyph_masks(font, unit, params["stroke_width"])
                position = (round(left + x + bbox[0]), baseline + bbox[1])
                box = [
                    position[0],
                    position[1],
                    position[0] + outer.shape[1],
                    position[1] + outer.shape[0],
                ]
                if (
                    box[0] < margin_x
                    or box[1] < margin_y
                    or box[2] > width - margin_x
                    or box[3] > height - margin_y
                ):
                    raise ValueError("glyph ink exceeds the configured safe area")
                glyphs.append(
                    {
                        "text": unit,
                        "position": position,
                        "outer": outer,
                        "inner": inner,
                        "start_frame": cue["unit_frames"][index],
                    }
                )
                boxes.append(box)
        layouts.append(
            {"cue": cue, "font_size": size, "glyphs": glyphs, "boxes": boxes}
        )
    return layouts


def caption_frame(request, layouts, frame_index):
    """Return one straight-alpha SDR RGBA frame at an integer frame index."""
    width, height, params = request["width"], request["height"], request["params"]
    canvas = np.zeros((height, width, 4), np.float32)
    for layout in layouts:
        cue = layout["cue"]
        if not cue["start_frame"] <= frame_index < cue["end_frame"]:
            continue
        for glyph in layout["glyphs"]:
            age = frame_index - glyph["start_frame"]
            pop = params["style"] == "pop"
            if pop and age < 0:
                continue
            fill = params["accent"] if age >= 0 else params["color"]
            outer, inner = glyph["outer"], glyph["inner"]
            rgb = (
                np.asarray(params["stroke_color"])
                * np.maximum(outer - inner, 0)[..., None]
            )
            rgb += np.asarray(fill) * inner[..., None]
            pixels = np.concatenate((rgb, outer[..., None]), axis=2).astype(np.float32)
            x, y = glyph["position"]
            if pop:
                entrance = min(
                    params["pop_frames"], cue["end_frame"] - cue["start_frame"]
                )
                phase = min(1, (age + 1) / entrance)
                ease = 1 - (1 - phase) ** 3
                scale = 0.65 + 0.35 * ease
                h, w = pixels.shape[:2]
                small = (max(1, round(w * scale)), max(1, round(h * scale)))
                pixels = cv2.resize(pixels, small, interpolation=cv2.INTER_AREA) * ease
                x += (w - small[0]) // 2
                y += (h - small[1]) // 2
            h, w = pixels.shape[:2]
            destination = canvas[y : y + h, x : x + w]
            destination[:] = pixels + destination * (1 - pixels[..., 3:4])
    alpha = canvas[..., 3:4]
    rgb = np.divide(
        canvas[..., :3], alpha, out=np.zeros_like(canvas[..., :3]), where=alpha > 0
    )
    return np.rint(np.clip(np.concatenate((rgb, alpha), axis=2), 0, 1) * 255).astype(
        np.uint8
    )
