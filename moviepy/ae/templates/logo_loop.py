"""Seamless 夜燈說書 logo title loop: a lamp, a breathing glow and rotating lines.

The channel's logo title for idle screens, live-stream waiting rooms, channel
trailers and the head or tail of an episode. Everything moves on integer
harmonics of ``period``: the halo and glow breathe once per loop, the flame
flickers on harmonics 3, 5 and 7, a light sweep crosses the title once, embers
rise exactly a whole number of frame heights, and the taglines rotate like a
carousel (the last one dissolves back into the first). The frame at
``period`` therefore equals the frame at ``0``, so the exported file repeats
with ``-stream_loop -1`` (or a player's loop switch) without a visible seam.

Every element is an ordinary AE layer (``Background``, ``Halo``, ``Embers``,
``Glow``, ``Lamp``, ``Flame glow``, ``Flame``, ``Title``, ``Shine``, ``Tagline 1`` ...) with its
``Transform`` ``Property`` objects, so a caller can re-keyframe any of them.

Examples
--------
>>> from moviepy.ae.templates.logo_loop import LogoLoopSpec, tagline_opacity_keys
>>> spec = LogoLoopSpec(title="Night Lamp", taglines=("one", "two"))
>>> spec.period, spec.layout
(6.0, 'stacked')
>>> [(k.time, k.value) for k in tagline_opacity_keys(0, 2, 6.0, 0.8)]
[(0.0, 100.0), (2.2, 100.0), (2.6, 0.0), (5.6, 0.0), (6.0, 100.0)]
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from moviepy.ae.composition import Composition
from moviepy.ae.layers import AVLayer
from moviepy.ae.properties.easing import Ease
from moviepy.ae.properties.keyframe import Keyframe
from moviepy.ae.properties.property import Property
from moviepy.ae.templates.title_card import (
    _check_glyphs,
    _render_text,
    _resolve_font,
    fit_lines,
)
from moviepy.ae.transform import Transform


__all__ = [
    "LogoLoopSpec",
    "tagline_opacity_keys",
    "build_logo_loop",
    "export_logo_loop",
    "main",
]

LAYOUTS = ("stacked", "inline")

#: 1080p reference pixels; scaled by ``preset.scale``.
TOKENS = {
    "title_size": 168,
    "title_min_size": 96,
    "title_tracking": 26,
    "tagline_size": 42,
    "tagline_min_size": 30,
    "tagline_tracking": 8,
    "lamp_height": 190,
    "lamp_title_gap": 34,
    "inline_gap": 44,
    "title_tagline_gap": 40,
    "tagline_rise": 14,
    "glow_radius": 18,
    "ember_size": 7,
}

_SS = 4  # supersampling of the drawn lamp


def _number(value, name, low=0.0, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or (value <= low if strict else value < low):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} {low}")
    return value


@dataclass(frozen=True)
class LogoLoopSpec:
    """Copy, layout and motion of one logo title loop.

    Parameters
    ----------
    title : str
        The logo word mark (default ``夜燈說書``).
    taglines : tuple of str
        Lines shown under the title one after another, each for
        ``period / len(taglines)`` seconds; empty for none.
    period : float
        Loop length in seconds; ``period * fps`` must be a whole number.
    crossfade : float
        Tagline hand-over in seconds: the old line fades out in the first
        half, the new one fades in during the second (at most a tagline's
        share of the loop).
    layout : {"stacked", "inline"}
        Lamp above the title, or lamp left of the title.
    lamp, shine : bool
        Draw the lamp (with halo and flame flicker) and the title light sweep.
    embers : int
        Number of rising glow particles (0 for none).
    breath : float
        Halo and glow breathing depth in ``[0, 1]``.
    seed : int
        Ember layout seed.

    Examples
    --------
    >>> LogoLoopSpec().title
    '夜燈說書'
    >>> LogoLoopSpec(taglines="一盞燈").taglines
    ('一盞燈',)
    """

    title: str = "夜燈說書"
    taglines: tuple = ("一盞夜燈・說一段故事",)
    period: float = 6.0
    crossfade: float = 0.8
    layout: str = "stacked"
    lamp: bool = True
    shine: bool = True
    embers: int = 28
    breath: float = 0.35
    seed: int = 7

    def __post_init__(self):
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("title must be a non-empty string")
        taglines = (self.taglines,) if isinstance(self.taglines, str) else self.taglines
        taglines = tuple(taglines or ())
        if not all(isinstance(line, str) and line.strip() for line in taglines):
            raise ValueError("taglines must be non-empty strings")
        object.__setattr__(self, "taglines", tuple(line.strip() for line in taglines))
        if self.layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        object.__setattr__(self, "period", _number(self.period, "period", strict=True))
        object.__setattr__(self, "crossfade", _number(self.crossfade, "crossfade"))
        if len(self.taglines) > 1 and self.crossfade > self.period / len(self.taglines):
            raise ValueError("crossfade is longer than one tagline's share")
        object.__setattr__(self, "breath", _number(self.breath, "breath"))
        if self.breath > 1:
            raise ValueError("breath must lie in [0, 1]")
        for name in ("embers", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


def _tagline_points(index, count, period, crossfade):
    """``(time, opacity, rise)`` points of tagline ``index`` of ``count``.

    ``rise`` is ``+1`` below the resting line, ``0`` on it and ``-1`` above.
    """
    if count <= 1:
        return [(0.0, 100.0, 0.0)]
    share, half = period / count, crossfade / 2
    begin, end = index * share, (index + 1) * share
    fade_out = [(end - crossfade, 100.0, 0.0), (end - half, 0.0, -1.0)]
    if index == 0:
        points = [(0.0, 100.0, 0.0)] + fade_out
        points += [(period - half, 0.0, 1.0), (period, 100.0, 0.0)]
    else:
        points = [(0.0, 0.0, 1.0), (begin - half, 0.0, 1.0), (begin, 100.0, 0.0)]
        points += fade_out
        if index < count - 1:
            points.append((period, 0.0, -1.0))
    result, last = [], -math.inf
    for time, value, rise in points:
        time = round(time, 9)
        if time > last + 1e-9:
            result.append((time, value, rise))
            last = time
    return result


def tagline_opacity_keys(index, count, period, crossfade):
    """Opacity keyframes (0..100) of tagline ``index`` of ``count``.

    Tagline ``i`` owns ``[i, i + 1) * period / count``. At each hand-over the
    outgoing line fades out (rising) during the first half of ``crossfade``
    and the incoming one fades in (rising into place) during the second half,
    ending exactly on the boundary; the first takes over from the last at
    ``period``. Values at ``0`` and ``period`` agree, so the loop is seamless.

    Examples
    --------
    >>> [(k.time, k.value) for k in tagline_opacity_keys(0, 1, 4.0, 0.5)]
    [(0.0, 100.0)]
    >>> [(k.time, k.value) for k in tagline_opacity_keys(1, 2, 6.0, 0.8)]
    [(0.0, 0.0), (2.6, 0.0), (3.0, 100.0), (5.2, 100.0), (5.6, 0.0)]
    """
    ease = Ease.ease_in_out()
    return [
        Keyframe(time, value, out_ease=ease)
        for time, value, _ in _tagline_points(index, count, period, crossfade)
    ]


def _tagline_position(index, count, period, crossfade, xy, rise):
    points = _tagline_points(index, count, period, crossfade)
    if len(points) == 1:
        return xy
    ease = Ease.ease_in_out()
    keys = [
        Keyframe(time, (xy[0], xy[1] + offset * rise), out_ease=ease)
        for time, _, offset in points
    ]
    return Property(keys, value_type="vec2", spatial=False)


# --------------------------------------------------------------------------- #
# rasters
# --------------------------------------------------------------------------- #


def _image_layer(name, rgb, alpha, **transform):
    from moviepy import ImageClip

    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    clip = ImageClip(rgb).with_mask(ImageClip(np.clip(alpha, 0, 1), is_mask=True))
    layer = AVLayer(clip, name)
    layer.transform = Transform(**transform)
    return layer


def _solid_rgb(shape, color):
    rgb = np.empty((shape[0], shape[1], 3), np.uint8)
    rgb[:] = tuple(int(c) for c in color)
    return rgb


def _teardrop(width, height, sharp=1.6, steps=96):
    """Polygon of a flame: a round bottom at y=height and the tip at y=0."""
    points = []
    for k in range(steps):
        t = 2 * math.pi * k / steps
        x = math.sin(t) * abs(math.sin(t / 2)) ** sharp
        y = math.cos(t)
        points.append((width / 2 + x * width / 2, height / 2 - y * height / 2))
    return points


def _lamp_rasters(height, gold, body, rim):
    """Return ``(base_rgb, base_alpha, flame_rgb, flame_alpha, geometry)``.

    The lamp is an oil-lamp bowl (``body`` with a ``gold`` rim) on a small
    foot; the flame is drawn separately so it can flicker around its base.
    """
    H = int(round(height))
    W = int(round(height * 0.92))
    S = _SS
    big = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    bowl_top = 0.60 * H
    bowl = (0.08 * W, bowl_top, 0.92 * W, bowl_top + 0.30 * H)
    # Lower half-ellipse bowl with a lip and a short spout to the right.
    draw.pieslice(
        [v * S for v in (bowl[0], bowl[1] - 0.15 * H, bowl[2], bowl[3])],
        0,
        180,
        fill=(*body, 255),
    )
    draw.polygon(
        [
            (0.80 * W * S, (bowl_top - 0.005 * H) * S),
            (1.00 * W * S, (bowl_top - 0.055 * H) * S),
            (0.97 * W * S, (bowl_top + 0.03 * H) * S),
            (0.78 * W * S, (bowl_top + 0.07 * H) * S),
        ],
        fill=(*body, 255),
    )
    lip = max(1, round(0.025 * H * S))
    draw.rounded_rectangle(
        [0.04 * W * S, bowl_top * S - lip, 0.84 * W * S, bowl_top * S + lip],
        radius=lip,
        fill=(*rim, 255),
    )
    foot_y = bowl[3] - 0.02 * H
    draw.polygon(
        [
            (0.40 * W * S, foot_y * S),
            (0.60 * W * S, foot_y * S),
            (0.70 * W * S, 0.985 * H * S),
            (0.30 * W * S, 0.985 * H * S),
        ],
        fill=(*body, 255),
    )
    draw.rounded_rectangle(
        [0.22 * W * S, 0.955 * H * S, 0.78 * W * S, 1.0 * H * S - 1],
        radius=max(1, round(0.02 * H * S)),
        fill=(*rim, 255),
    )
    # Wick.
    draw.rectangle(
        [0.485 * W * S, (bowl_top - 0.05 * H) * S, 0.515 * W * S, bowl_top * S],
        fill=(60, 44, 30, 255),
    )
    base = np.asarray(big.resize((W, H), Image.LANCZOS), dtype=np.float64)

    fw, fh = 0.30 * W, 0.52 * H
    FW, FH = int(math.ceil(fw)) + 4, int(math.ceil(fh)) + 4
    flame = Image.new("RGBA", (FW * S, FH * S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(flame)
    offset = 2 * S
    outer = [(x * S + offset, y * S + offset) for x, y in _teardrop(fw, fh)]
    draw.polygon(outer, fill=(*gold, 255))
    core = [
        ((fw * 0.22 + x) * S + offset, (fh * 0.38 + y) * S + offset)
        for x, y in _teardrop(fw * 0.56, fh * 0.6, sharp=1.2)
    ]
    draw.polygon(core, fill=(255, 246, 222, 255))
    flame = flame.filter(ImageFilter.GaussianBlur(0.9 * S))
    flame = np.asarray(flame.resize((FW, FH), Image.LANCZOS), dtype=np.float64)
    geometry = {
        "size": (W, H),
        "flame_size": (FW, FH),
        # Flame base (its pivot) in lamp pixels: just above the wick.
        "flame_base": (W / 2, bowl_top - 0.035 * H),
        "flame_anchor": (FW / 2, FH - 2 - 0.04 * fh),
    }
    return (
        base[..., :3].astype(np.uint8),
        base[..., 3] / 255.0,
        flame[..., :3].astype(np.uint8),
        flame[..., 3] / 255.0,
        geometry,
    )


def _blur(alpha, radius):
    image = Image.fromarray(np.rint(np.clip(alpha, 0, 1) * 255).astype(np.uint8))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(radius))) / 255.0


def _breathing(period, low, high, harmonic=1, phase=0.0):
    """Opacity (0..100) that breathes between ``low`` and ``high`` per loop."""
    omega = 2 * math.pi * harmonic / period

    def value(t):
        wave = 0.5 - 0.5 * math.cos(omega * t + phase)
        return 100.0 * (low + (high - low) * wave)

    return Property(value, value_type="float")


def _flicker(period, depth):
    """Flame scale (percent) on harmonics 3, 5 and 7 of the loop."""
    w = 2 * math.pi / period

    def value(t):
        x = 1 + depth * (0.55 * math.sin(3 * w * t) + 0.3 * math.sin(5 * w * t + 1.1))
        y = 1 + depth * (
            0.8 * math.sin(3 * w * t + 0.4) + 0.45 * math.sin(7 * w * t + 2.0)
        )
        return (100.0 * x, 100.0 * y)

    return Property(value, value_type="vec2")


def _sway(period, depth):
    w = 2 * math.pi / period
    return Property(
        lambda t: depth * (math.sin(2 * w * t) + 0.5 * math.sin(5 * w * t + 0.7)),
        value_type="float",
    )


def _procedural_layer(name, size, color, alpha, **transform):
    """Layer whose RGB is ``color`` and whose alpha is ``alpha(t)``.

    The clip has no duration: an AE layer holds a finite clip's last frame
    after its end, which would break ``alpha``'s periodicity at ``period``.
    """
    from moviepy import VideoClip

    w, h = size
    rgb = _solid_rgb((h, w), color)
    clip = VideoClip(lambda t: rgb).with_mask(VideoClip(alpha, is_mask=True))
    layer = AVLayer(clip, name)
    layer.transform = Transform(**transform)
    return layer


def _ember_alpha(size, count, period, seed, sprite_px, region):
    """Return ``alpha(t)`` of ``count`` embers rising in a periodic loop."""
    w, h = size
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = region
    radius = max(1.0, sprite_px)
    r = int(math.ceil(radius * 3))
    span = y1 - y0 + 2 * r
    embers = []
    for _ in range(count):
        embers.append(
            {
                "x": rng.uniform(x0, x1),
                "y": rng.uniform(0, span),
                "laps": int(rng.integers(1, 3)),  # whole spans per period
                "sway": rng.uniform(4, 22) * radius / 4,
                "sway_k": int(rng.integers(1, 4)),
                "phase": rng.uniform(0, 2 * math.pi),
                "twinkle_k": int(rng.integers(2, 6)),
                "gain": rng.uniform(0.35, 0.9) * rng.uniform(0.6, 1.0),
                "scale": rng.uniform(0.6, 1.3),
            }
        )
    sprites = {}

    def alpha(t):
        t = t % period  # t == period lands exactly on the first frame
        frame = np.zeros((h, w))
        phase = 2 * math.pi * t / period
        for e in embers:
            y = y1 + r - ((e["y"] + span * e["laps"] * t / period) % span)
            x = e["x"] + e["sway"] * math.sin(e["sway_k"] * phase + e["phase"])
            # Fade in at the bottom and out at the top of the region.
            u = (y1 + r - y) / span
            edge = min(1.0, 5 * u, 5 * (1 - u))
            twinkle = 0.65 + 0.35 * math.sin(e["twinkle_k"] * phase + e["phase"])
            value = e["gain"] * max(0.0, edge) * twinkle
            if value <= 0.003:
                continue
            key = round(e["scale"], 2)
            if key not in sprites:
                sr = max(1, int(round(r * key)))
                sy, sx = np.mgrid[-sr : sr + 1, -sr : sr + 1]
                sigma = radius * key / 1.6
                sprites[key] = np.exp(-(sx**2 + sy**2) / (2 * sigma**2))
            stamp = sprites[key]
            sr = stamp.shape[0] // 2
            cx, cy = int(round(x)), int(round(y))
            ax0, ay0 = max(0, cx - sr), max(0, cy - sr)
            ax1, ay1 = min(w, cx + sr + 1), min(h, cy + sr + 1)
            if ax0 >= ax1 or ay0 >= ay1:
                continue
            patch = stamp[ay0 - cy + sr : ay1 - cy + sr, ax0 - cx + sr : ax1 - cx + sr]
            region_view = frame[ay0:ay1, ax0:ax1]
            np.maximum(region_view, patch * value, out=region_view)
        return frame

    return alpha


def _shine_alpha(mask, period, start=0.12, length=0.42, width=0.07):
    """Return ``alpha(t)`` of a diagonal light band sweeping across ``mask``."""
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    slant = 0.45
    u = (xx + slant * (h - yy)) / (w + slant * h)

    def alpha(t):
        progress = ((t / period) % 1.0 - start) / length
        if progress <= 0 or progress >= 1:
            return np.zeros_like(mask)
        centre = -3 * width + progress * (1 + 6 * width)
        band = np.exp(-(((u - centre) / width) ** 2))
        return mask * band * 0.85

    return alpha


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #


def build_logo_loop(spec=None, preset=None, *, fonts=None, transparent=False):
    """Build the seamless logo title loop ``Composition``.

    Parameters
    ----------
    spec : LogoLoopSpec, optional
        Copy and motion; the default is the 夜燈說書 logo with one tagline.
    preset : ChannelPreset, optional
        Canvas, fps, palette and fonts; defaults to ``nightlamp_story``. Title
        and tagline colours follow ``card_ink``/``card_secondary_ink`` (else
        ``ink``/``secondary_ink``), the lamp and glow use ``lamp`` (else
        ``bamboo``).
    fonts : dict, optional
        ``{"title": path, "body": path}``; ``None`` selects Pillow's default
        (Latin-only) font.
    transparent : bool, optional
        Leave out the night background and carry alpha, for laying the logo
        over footage (export as a MOV with alpha).

    Returns
    -------
    Composition
        ``period`` seconds long; ``comp.logo_report`` records sizes, boxes,
        fonts and the frame count.

    Raises
    ------
    ValueError
        If ``period * fps`` is not a whole number of frames, a glyph is
        missing or the block does not fit in the safe area.

    Examples
    --------
    >>> from moviepy.ae.templates.presets import get_preset
    >>> preset = get_preset("nightlamp_story", size=(320, 180), safe_margin=(16, 12))
    >>> comp = build_logo_loop(
    ...     LogoLoopSpec(title="Night Lamp", taglines=("a", "b"), embers=4),
    ...     preset, fonts={"title": None, "body": None})
    >>> comp.duration, comp.logo_report["frames"]
    (6.0, 144)
    >>> [layer.name for layer in comp.layers][:3]
    ['Tagline 2', 'Tagline 1', 'Shine']
    """
    from moviepy.ae.templates.presets import get_preset

    spec = LogoLoopSpec() if spec is None else spec
    preset = get_preset("nightlamp_story") if preset is None else preset
    fps = float(preset.fps)
    frames = int(round(spec.period * fps))
    if abs(spec.period * fps - frames) > 1e-6 or frames < 2:
        raise ValueError(
            f"period {spec.period:g}s at {fps:g} fps is not a whole number of frames"
        )
    size_w, size_h = preset.size
    s = preset.scale
    margin_x, margin_y = preset.safe_margin

    def px(value):
        return value * s

    title_font = _resolve_font(preset, fonts, "title")
    body_font = _resolve_font(preset, fonts, "body")
    _check_glyphs(spec.title, title_font, "title")
    _check_glyphs("".join(spec.taglines), body_font, "tagline")

    ink = tuple(preset.color("card_ink", preset.color("ink", (243, 236, 220))))
    secondary = tuple(
        preset.color(
            "card_secondary_ink", preset.color("secondary_ink", (232, 223, 204))
        )
    )
    gold = tuple(preset.color("lamp", preset.color("bamboo", (217, 164, 95))))
    dark = preset.color("panel", preset.color("bamboo_dark", (73, 60, 45)))
    # Bronze bowl: halfway between the lamp gold and the dark panel colour.
    body_color = tuple(int(round((a + b) / 2)) for a, b in zip(gold, dark))
    outline = (max(1.0, 3 * s), (20, 16, 12)) if transparent else None
    pad = max(2, round(px(10)))

    lamp_h = px(TOKENS["lamp_height"]) if spec.lamp else 0.0
    lamp_w = lamp_h * 0.92
    avail_w = size_w - 2 * margin_x - 6 * s
    title_w = avail_w
    if spec.lamp and spec.layout == "inline":
        title_w -= lamp_w + px(TOKENS["inline_gap"])
    title_lines, title_size = fit_lines(
        spec.title,
        title_font,
        title_w,
        round(px(TOKENS["title_size"])),
        round(px(TOKENS["title_min_size"])),
        px(TOKENS["title_tracking"]),
        max_lines=1,
    )
    title_rgb, title_mask, title_box = _render_text(
        title_lines,
        title_font,
        title_size,
        px(TOKENS["title_tracking"]),
        1.2,
        ink,
        "center",
        pad,
        outline,
    )
    taglines = []
    for line in spec.taglines:
        lines, line_size = fit_lines(
            line,
            body_font,
            avail_w,
            round(px(TOKENS["tagline_size"])),
            round(px(TOKENS["tagline_min_size"])),
            px(TOKENS["tagline_tracking"]),
            max_lines=1,
        )
        taglines.append(
            _render_text(
                lines,
                body_font,
                line_size,
                px(TOKENS["tagline_tracking"]),
                1.2,
                secondary,
                "center",
                pad,
                outline,
            )
            + (line_size,)
        )
    tag_h = max((t[2][1] for t in taglines), default=0.0)

    # Block geometry (ink boxes, without the raster padding).
    tw, th = title_box
    if spec.layout == "stacked" or not spec.lamp:
        gap = px(TOKENS["lamp_title_gap"]) if spec.lamp else 0.0
        head_h = lamp_h + gap + th
        head_w = max(lamp_w, tw)
    else:
        gap = px(TOKENS["inline_gap"])
        head_h = max(lamp_h, th)
        head_w = lamp_w + gap + tw
    tag_gap = px(TOKENS["title_tagline_gap"]) if taglines else 0.0
    block_h = head_h + tag_gap + tag_h
    block_w = max(head_w, max((t[2][0] for t in taglines), default=0.0))
    if block_h > size_h - 2 * margin_y or block_w > size_w - 2 * margin_x + 1:
        raise ValueError(
            f"logo block {block_w:.0f}x{block_h:.0f}px does not fit the safe area"
        )
    cx = size_w / 2
    top = (size_h - block_h) / 2
    if spec.layout == "stacked" or not spec.lamp:
        lamp_xy = (cx - lamp_w / 2, top)
        title_xy = (cx - tw / 2, top + head_h - th)
    else:
        left = cx - head_w / 2
        lamp_xy = (left, top + (head_h - lamp_h) / 2)
        title_xy = (left + lamp_w + gap, top + (head_h - th) / 2)
    tag_y = top + head_h + tag_gap

    comp = Composition(
        preset.size,
        fps,
        spec.period,
        name="Logo loop",
        bg_color=(0, 0, 0),
        transparent=transparent,
    )
    period = spec.period
    boxes = {}

    if not transparent:
        yy = np.linspace(0.0, 1.0, size_h)[:, None]
        top_c = np.array([13, 11, 14], np.float64)
        bottom_c = np.array([34, 25, 18], np.float64)
        grad = top_c + (bottom_c - top_c) * yy[..., None]
        xx = np.linspace(-1.0, 1.0, size_w)[None, :]
        vignette = 1 - 0.45 * np.clip((xx**2 + ((yy - 0.5) * 2) ** 2) / 2, 0, 1)
        rgb = np.rint(grad * vignette[..., None]).clip(0, 255).astype(np.uint8)
        comp.add_layer(
            _image_layer(
                "Background",
                rgb,
                np.ones((size_h, size_w)),
                anchor_point=(0, 0),
                position=(0, 0),
            )
        )

    flame_centre = None
    if spec.lamp:
        base_rgb, base_alpha, flame_rgb, flame_alpha, geo = _lamp_rasters(
            lamp_h, gold, body_color, gold
        )
        fx = lamp_xy[0] + geo["flame_base"][0] * lamp_w / geo["size"][0]
        fy = lamp_xy[1] + geo["flame_base"][1] * lamp_h / geo["size"][1]
        flame_centre = (fx, fy - geo["flame_size"][1] * 0.35)
        # Halo: a wide radial lamp glow that breathes once per loop.
        halo_r = 0.42 * size_h
        hy, hx = np.mgrid[0:size_h, 0:size_w]
        d2 = ((hx - flame_centre[0]) ** 2 + (hy - flame_centre[1]) ** 2) / halo_r**2
        halo = np.exp(-2.2 * d2) * (0.30 if transparent else 0.42)
        comp.add_layer(
            _image_layer(
                "Halo",
                _solid_rgb(halo.shape, gold),
                halo,
                anchor_point=(0, 0),
                position=(0, 0),
                opacity=_breathing(period, 1 - spec.breath, 1.0),
            )
        )

    if spec.embers:
        region = (
            max(0.0, cx - 0.36 * size_w),
            max(0.0, top - 0.25 * size_h),
            min(float(size_w), cx + 0.36 * size_w),
            min(float(size_h), top + block_h + 0.18 * size_h),
        )
        comp.add_layer(
            _procedural_layer(
                "Embers",
                (size_w, size_h),
                gold,
                _ember_alpha(
                    (size_w, size_h),
                    spec.embers,
                    period,
                    spec.seed,
                    px(TOKENS["ember_size"]) / 2,
                    region,
                ),
                anchor_point=(0, 0),
                position=(0, 0),
            )
        )

    # Glow: blurred title (and flame) in lamp gold, screened, breathing.
    glow_r = px(TOKENS["glow_radius"])
    margin = int(math.ceil(glow_r * 3))
    gh, gw = title_mask.shape
    glow = np.zeros((gh + 2 * margin, gw + 2 * margin))
    glow[margin : margin + gh, margin : margin + gw] = title_mask
    glow = _blur(glow, glow_r) * 0.9
    comp.add_layer(
        _image_layer(
            "Glow",
            _solid_rgb(glow.shape, gold),
            glow,
            anchor_point=(0, 0),
            position=(title_xy[0] - pad - margin, title_xy[1] - pad - margin),
            opacity=_breathing(period, 0.55 * (1 - spec.breath), 0.75, phase=0.6),
        )
    )
    comp.layers[0].blend_mode = "screen"

    if spec.lamp:
        lamp_layer = _image_layer(
            "Lamp",
            base_rgb,
            base_alpha,
            anchor_point=(0, 0),
            position=lamp_xy,
        )
        comp.add_layer(lamp_layer)
        boxes["Lamp"] = (
            lamp_xy[0],
            lamp_xy[1],
            lamp_xy[0] + lamp_w,
            lamp_xy[1] + lamp_h,
        )
        flame_glow = _blur(np.pad(flame_alpha, margin), glow_r * 0.8)
        comp.add_layer(
            _image_layer(
                "Flame glow",
                _solid_rgb(flame_glow.shape, gold),
                np.clip(flame_glow * 1.6, 0, 1),
                anchor_point=(
                    margin + geo["flame_anchor"][0],
                    margin + geo["flame_anchor"][1],
                ),
                position=(fx, fy),
                scale=_flicker(period, 0.10),
                opacity=_breathing(period, 0.6, 1.0, harmonic=3),
            )
        )
        comp.layers[0].blend_mode = "screen"
        comp.add_layer(
            _image_layer(
                "Flame",
                flame_rgb,
                flame_alpha,
                anchor_point=geo["flame_anchor"],
                position=(fx, fy),
                scale=_flicker(period, 0.07),
                rotation=_sway(period, 2.5),
            )
        )

    comp.add_layer(
        _image_layer(
            "Title",
            title_rgb,
            title_mask,
            anchor_point=(pad, pad),
            position=title_xy,
        )
    )
    boxes["Title"] = (title_xy[0], title_xy[1], title_xy[0] + tw, title_xy[1] + th)
    if spec.shine:
        comp.add_layer(
            _procedural_layer(
                "Shine",
                title_mask.shape[::-1],
                (255, 206, 132),
                _shine_alpha(title_mask, period),
                anchor_point=(pad, pad),
                position=title_xy,
            )
        )

    for index, (rgb, mask, box, line_size) in enumerate(taglines):
        x = cx - box[0] / 2
        comp.add_layer(
            _image_layer(
                f"Tagline {index + 1}",
                rgb,
                mask,
                anchor_point=(pad, pad),
                position=_tagline_position(
                    index,
                    len(taglines),
                    period,
                    spec.crossfade,
                    (x, tag_y),
                    px(TOKENS["tagline_rise"]),
                ),
                opacity=Property(
                    tagline_opacity_keys(index, len(taglines), period, spec.crossfade),
                    value_type="float",
                ),
            )
        )
        boxes[f"Tagline {index + 1}"] = (x, tag_y, x + box[0], tag_y + box[1])

    for name, (x0, y0, x1, y1) in boxes.items():
        if (
            x0 < margin_x - 1
            or x1 > size_w - margin_x + 1
            or y0 < margin_y - 1
            or y1 > size_h - margin_y + 1
        ):
            raise ValueError(f"{name} falls outside the preset safe margin")
    comp.logo_report = {
        "title": spec.title,
        "taglines": list(spec.taglines),
        "layout": spec.layout,
        "period": period,
        "fps": fps,
        "frames": frames,
        "title_size": title_size,
        "tagline_sizes": [t[3] for t in taglines],
        "boxes": {k: tuple(float(v) for v in b) for k, b in boxes.items()},
        "flame_centre": flame_centre,
        "fonts": {"title": title_font, "body": body_font},
        "transparent": bool(transparent),
    }
    return comp


def export_logo_loop(comp, path, *, overwrite=False, codec="qtrle", **kwargs):
    """Write a logo loop: MP4 when opaque, a MOV with alpha when transparent.

    Opaque loops go through ``visual_loop.export_loop`` (closed one-second
    GOP, no B-frames, safe for ``-stream_loop -1``); transparent ones through
    ``ambience.export_overlay_loop`` with ``codec``. Extra keyword arguments
    are passed to the exporter.

    Returns
    -------
    dict
        The exporter's evidence.

    Examples
    --------
    >>> import inspect
    >>> inspect.signature(export_logo_loop).parameters["codec"].default
    'qtrle'
    """
    path = Path(path)
    if comp.transparent:
        from moviepy.ae.templates.ambience import export_overlay_loop

        return export_overlay_loop(
            comp, path, codec=codec, overwrite=overwrite, **kwargs
        )
    from moviepy.ae.templates.visual_loop import export_loop

    return export_loop(comp, path, overwrite=overwrite, **kwargs)


def main(argv=None):
    """``python -m moviepy.ae.templates logo OUTPUT``: render a logo loop.

    Examples
    --------
    >>> import pytest
    >>> with pytest.raises(SystemExit):
    ...     main(["--help"])  # doctest: +ELLIPSIS
    usage: ...
    """
    import argparse
    import json

    from moviepy.ae.templates.presets import get_preset

    parser = argparse.ArgumentParser(
        prog="python -m moviepy.ae.templates logo",
        description="Render the seamless 夜燈說書 logo title loop.",
    )
    parser.add_argument("output", help=".mp4 (opaque) or .mov (with --transparent)")
    parser.add_argument("--title", default=LogoLoopSpec.title)
    parser.add_argument(
        "--tagline",
        action="append",
        help="tagline shown in rotation; repeat for more (default: one line)",
    )
    parser.add_argument("--no-tagline", action="store_true")
    parser.add_argument("--period", type=float, default=LogoLoopSpec.period)
    parser.add_argument(
        "--crossfade",
        type=float,
        help="tagline hand-over seconds (default: 0.8, at most half a share)",
    )
    parser.add_argument("--layout", choices=LAYOUTS, default=LogoLoopSpec.layout)
    parser.add_argument("--preset", default="nightlamp_story")
    parser.add_argument("--size", help="WIDTHxHEIGHT, e.g. 1280x720")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--embers", type=int, default=LogoLoopSpec.embers)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    overrides = {}
    if args.size:
        width, _, height = args.size.lower().partition("x")
        overrides["size"] = (int(width), int(height))
    if args.fps:
        overrides["fps"] = args.fps
    preset = get_preset(args.preset, **overrides)
    if args.no_tagline:
        taglines = ()
    else:
        taglines = tuple(args.tagline or LogoLoopSpec.taglines)
    crossfade = args.crossfade
    if crossfade is None:
        crossfade = min(LogoLoopSpec.crossfade, args.period / max(1, len(taglines)) / 2)
    spec = LogoLoopSpec(
        title=args.title,
        taglines=taglines,
        period=args.period,
        crossfade=crossfade,
        layout=args.layout,
        embers=args.embers,
    )
    comp = build_logo_loop(spec, preset, transparent=args.transparent)
    try:
        report = export_logo_loop(comp, args.output, overwrite=args.overwrite)
    finally:
        comp.close()
    keep = ("path", "frames", "fps", "size", "seconds", "encoder", "codec", "seam")
    print(
        json.dumps(
            {k: report[k] for k in keep if k in report}, ensure_ascii=False, indent=2
        )
    )
    return 0
