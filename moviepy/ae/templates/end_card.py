"""Closing card: a like / subscribe / share carousel above the subtitle strip.

The last seconds of an episode show the channel name, a short line, a question
for the comments and three icon buttons. One button at a time is "active": it
pops (80 % -> 106 % -> 100 % over 10 frames at 30 fps, like the chapter
overlay's subscribe pop) and turns the accent colour while the others are
dimmed; the highlight moves to the next action every ``period`` seconds and
loops until the card fades out. Icons are drawn with Pillow polygons, so no
emoji or icon font is needed.

Each icon is rendered once per state (dim, active) and shown by gating layer
opacity with hold keyframes on the layer clock (``t - start_time``). Geometry is
authored at 1920x1080 and scaled by ``min(width / 1920, height / 1080)``.

Examples
--------
>>> card = EndCard(start=100.0, duration=6.0)
>>> [card.active_index(100.0 + t) for t in (0.0, 0.5, 2.1, 3.7, 5.6)]
[None, 0, 1, 2, None]
>>> EndCard.from_dict(card.to_dict()) == card
True
"""

import math
from dataclasses import dataclass, fields

import numpy as np
from PIL import Image, ImageDraw

from moviepy.ae.properties import Keyframe, Property
from moviepy.ae.properties.easing import Ease
from moviepy.ae.templates.name_tag import REFERENCE_SIZE, SUBTITLE_SAFE_RECT
from moviepy.ae.templates.paper import CachedAVLayer, load_font, rgba_still
from moviepy.ae.transform import Transform


__all__ = [
    "EndCard",
    "add_end_card",
    "end_card_layers",
    "icon_image",
]

_ICONS = ("like", "subscribe", "share")
_POP = ((0, 80.0), (6, 106.0), (10, 100.0))  # (frame, percent) at _POP_FPS
_POP_FPS = 30.0
_OUT = Ease.bezier(0.16, 1.0, 0.3, 1.0)
_PANEL_RECT = (360, 60, 1200, 720)  # bottom edge stays above the subtitle rect
_PAD = 40
_DIM = 0.42  # opacity of an inactive button
_SS = 4  # icon supersampling
_SHRINK = 0.92
_FLOOR = 0.55


# --------------------------------------------------------------------------- #
# Spec.
# --------------------------------------------------------------------------- #


def _rgb(value, name):
    try:
        out = tuple(int(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be three 0-255 integers") from None
    if len(out) != 3 or any(not 0 <= v <= 255 for v in out):
        raise ValueError(f"{name} must be three 0-255 integers")
    return out


def _num(value, name, low, high=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or (value <= low if strict else value < low):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {low:g}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be <= {high:g}")
    return value


@dataclass(frozen=True)
class EndCard:
    """The closing card of an episode.

    Attributes
    ----------
    start, duration : float
        Appearance time and length in seconds.
    channel : str
        Channel name, the large title.
    headline : str
        Optional line under the channel name.
    line : str
        Call-to-action line above the buttons.
    question : str
        Optional comment question at the bottom.
    actions : tuple of str
        Button labels, 1 to 5; the carousel highlights them in turn.
    icons : tuple of str
        Icon per action by position (cycled): ``"like"``, ``"subscribe"``,
        ``"share"``.
    period : float
        Seconds each action stays highlighted.
    panel_color : tuple
        Panel RGB.
    panel_alpha : float
        Panel opacity, 0-1 (default 185/255).
    text_color, accent_color : tuple
        Text RGB, and the highlight RGB of the active action.
    fade : float
        Fade-in and fade-out seconds; the carousel runs between them.

    Examples
    --------
    >>> EndCard(duration=1.0)
    Traceback (most recent call last):
    ...
    ValueError: duration must exceed twice the fade (1 s)
    """

    start: float = 0.0
    duration: float = 8.0
    channel: str = "夜燈說書"
    headline: str = ""
    line: str = "訂閱・下一回再走進歷史"
    question: str = ""
    actions: tuple = ("按讚", "訂閱", "分享")
    icons: tuple = _ICONS
    period: float = 1.6
    panel_color: tuple = (29, 23, 19)
    panel_alpha: float = 185 / 255
    text_color: tuple = (239, 220, 184)
    accent_color: tuple = (221, 55, 55)
    fade: float = 0.5

    def __post_init__(self):
        sset = object.__setattr__
        sset(self, "start", _num(self.start, "start", 0.0))
        sset(self, "duration", _num(self.duration, "duration", 0.0, strict=True))
        sset(self, "period", _num(self.period, "period", 0.0, strict=True))
        sset(self, "fade", _num(self.fade, "fade", 0.0))
        sset(self, "panel_alpha", _num(self.panel_alpha, "panel_alpha", 0.0, 1.0))
        for key in ("channel", "headline", "line", "question"):
            value = getattr(self, key)
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            sset(self, key, value.strip())
        if not self.channel:
            raise ValueError("channel must not be empty")
        for key in ("actions", "icons"):
            value = getattr(self, key)
            if isinstance(value, str) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{key} must be a sequence of strings")
            sset(self, key, tuple(value))
        if not 1 <= len(self.actions) <= 5 or not all(a.strip() for a in self.actions):
            raise ValueError("actions must hold 1 to 5 non-empty labels")
        if not self.icons or any(i not in _ICONS for i in self.icons):
            raise ValueError(f"icons must be a non-empty subset of {_ICONS}")
        for key in ("panel_color", "text_color", "accent_color"):
            sset(self, key, _rgb(getattr(self, key), key))
        if self.duration <= 2 * self.fade:
            raise ValueError(
                f"duration must exceed twice the fade ({2 * self.fade:g} s)"
            )

    @property
    def end(self):
        """Time in seconds when the card has fully faded out."""
        return self.start + self.duration

    def windows(self):
        """Return the highlight windows as ``(action_index, begin, end)``.

        Times are in seconds on the layer clock (``0`` is ``start``). The
        carousel runs between the fades and cycles through the actions.
        """
        out, t, k = [], self.fade, 0
        stop = self.duration - self.fade
        while t < stop - 1e-9:
            out.append((k % len(self.actions), t, min(t + self.period, stop)))
            t += self.period
            k += 1
        return out

    def active_index(self, t):
        """Return the highlighted action at composition time ``t``, or ``None``."""
        u = float(t) - self.start
        for index, begin, end in self.windows():
            if begin - 1e-9 <= u < end - 1e-9:
                return index
        return None

    def to_dict(self):
        """Return a JSON-friendly dict that ``from_dict`` reads back."""
        out = {}
        for item in fields(self):
            value = getattr(self, item.name)
            out[item.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data):
        """Build a card from a dict such as ``json.load`` returns.

        Raises
        ------
        ValueError
            On unknown keys or invalid values.
        """
        if not isinstance(data, dict):
            raise ValueError("an end card config must be a mapping")
        known = {item.name for item in fields(cls)}
        extra = sorted(set(data) - known)
        if extra:
            raise ValueError(f"unknown end card keys: {extra}")
        return cls(**data)


# --------------------------------------------------------------------------- #
# Icons.
# --------------------------------------------------------------------------- #


def _mask_like(d, u):
    # cuff + hand with a raised thumb, on a 100 unit grid
    d.rounded_rectangle([8 * u, 44 * u, 28 * u, 90 * u], radius=3 * u, fill=255)
    hand = [(34, 46), (50, 24), (56, 8), (66, 8), (70, 18), (62, 40), (90, 40),
            (95, 50), (90, 56), (93, 64), (87, 70), (89, 78), (82, 84), (82, 90),
            (34, 90)]  # fmt: skip
    d.polygon([(x * u, y * u) for x, y in hand], fill=255)


def _mask_subscribe(d, u):
    d.rounded_rectangle([4 * u, 24 * u, 96 * u, 76 * u], radius=26 * u, fill=255)
    d.polygon([(42 * u, 36 * u), (42 * u, 64 * u), (66 * u, 50 * u)], fill=0)


def _mask_share(d, u):
    d.line([(12 * u, 82 * u), (12 * u, 62 * u), (30 * u, 44 * u), (56 * u, 44 * u)],
           fill=255, width=round(11 * u), joint="curve")  # fmt: skip
    d.polygon([(54 * u, 14 * u), (94 * u, 44 * u), (54 * u, 74 * u)], fill=255)


_MASKS = {"like": _mask_like, "subscribe": _mask_subscribe, "share": _mask_share}


def icon_image(name, size, color, opacity=1.0):
    """Draw an icon with Pillow polygons.

    Parameters
    ----------
    name : {"like", "subscribe", "share"}
        Thumb-up, subscribe pill with a play triangle, or share arrow.
    size : int
        Edge of the square image in pixels.
    color : sequence of int
        RGB fill.
    opacity : float, optional
        Multiplies the alpha channel.

    Returns
    -------
    PIL.Image.Image
        ``RGBA`` square image; the icon is anti-aliased (4x supersampling) and
        everything outside it, including the play-triangle cut-out, is
        transparent.

    Examples
    --------
    >>> img = icon_image("subscribe", 40, (255, 0, 0))
    >>> img.size, img.getpixel((20, 20))[3], img.getpixel((8, 20))[3]
    ((40, 40), 0, 255)
    """
    if name not in _MASKS:
        raise ValueError(f"icon must be one of {_ICONS}, got {name!r}")
    size = int(size)
    if size < 2:
        raise ValueError("size must be at least 2")
    big = size * _SS
    mask = Image.new("L", (big, big), 0)
    _MASKS[name](ImageDraw.Draw(mask), big / 100.0)
    mask = mask.resize((size, size), Image.LANCZOS)
    alpha = (np.asarray(mask, np.float32) * float(opacity)).round().astype(np.uint8)
    out = np.zeros((size, size, 4), np.uint8)
    out[..., :3] = _rgb(color, "color")
    out[..., 3] = alpha
    return Image.fromarray(out, "RGBA")


# --------------------------------------------------------------------------- #
# Rasters.
# --------------------------------------------------------------------------- #


def _canvas(size):
    try:
        w, h = (int(v) for v in size)
    except (TypeError, ValueError):
        raise ValueError("size must be (w, h) integers") from None
    if w <= 0 or h <= 0:
        raise ValueError("size must be positive")
    return w, h, min(w / REFERENCE_SIZE[0], h / REFERENCE_SIZE[1])


def _centered(draw, text, font, cx, y, fill):
    width = font.getlength(text)
    draw.text((cx - width / 2, y), text, font=font, fill=fill)


def _plan(card, s, font, title_font, factor):
    """Lay the panel out at text ``factor``; return the plan or ``None``."""
    px, py, pw, ph = (v * s for v in _PANEL_RECT)
    inner_w = pw - 2 * _PAD * s
    gap = 22 * s
    f_chan = load_font(title_font, max(8, round(84 * s * factor)))
    f_head = load_font(font, max(8, round(44 * s * factor)))
    f_line = load_font(font, max(8, round(40 * s * factor)))
    f_ques = load_font(font, max(8, round(44 * s * factor)))
    f_label = load_font(font, max(8, round(48 * s * factor)))

    def height(f):
        a, d = f.getmetrics()
        return a + d

    pitch = min(340 * s, inner_w / len(card.actions))
    icon = round(130 * s * factor)
    cell_h = icon + round(14 * s) + height(f_label)
    rows = [("channel", card.channel, f_chan, height(f_chan))]
    if card.headline:
        rows.append(("headline", card.headline, f_head, height(f_head)))
    if card.line:
        rows.append(("line", card.line, f_line, height(f_line)))
    rows.append(("cells", None, None, cell_h))
    if card.question:
        rows.append(("question", card.question, f_ques, height(f_ques)))
    total = sum(r[3] for r in rows) + gap * (len(rows) - 1)
    widest = max(
        [f.getlength(t) for _, t, f, _ in rows if t]
        + [f_label.getlength(a) for a in card.actions]
    )
    too_wide = widest > inner_w + 1e-6 or any(
        f_label.getlength(a) > pitch * 0.94 for a in card.actions
    )
    if too_wide or total > ph - 2 * _PAD * s:
        return None
    return {
        "rows": rows, "gap": gap, "total": total, "pitch": pitch, "icon": icon,
        "cell_h": cell_h, "label": f_label, "rect": (px, py, pw, ph),
    }  # fmt: skip


def _fit_plan(card, s, font, title_font):
    factor = 1.0
    while factor >= _FLOOR - 1e-9:
        plan = _plan(card, s, font, title_font, factor)
        if plan is not None:
            return plan
        factor *= _SHRINK
    raise ValueError(
        "end card text does not fit the panel even at the smallest size; "
        "shorten channel/headline/line/question/actions"
    )


def _cell_image(card, plan, index, active):
    """Raster one button (icon over label) in the dim or active state."""
    pitch = round(plan["pitch"])
    icon, cell_h = plan["icon"], plan["cell_h"]
    color = card.accent_color if active else card.text_color
    opacity = 1.0 if active else _DIM
    img = Image.new("RGBA", (pitch, cell_h), (0, 0, 0, 0))
    glyph = icon_image(card.icons[index % len(card.icons)], icon, color, opacity)
    img.alpha_composite(glyph, ((pitch - icon) // 2, 0))
    label = Image.new("RGBA", img.size, (0, 0, 0, 0))
    font = plan["label"]
    ink = (*color, round(255 * opacity))
    _centered(
        ImageDraw.Draw(label),
        card.actions[index],
        font,
        pitch / 2,
        cell_h - sum(font.getmetrics()),
        ink,
    )
    img.alpha_composite(label)
    return img


def _panel_image(card, plan):
    px, py, pw, ph = plan["rect"]
    size = (round(pw), round(ph))
    img = Image.new("RGBA", size, (*card.panel_color, round(255 * card.panel_alpha)))
    draw = ImageDraw.Draw(img)
    fill = (*card.text_color, 255)
    y = (size[1] - plan["total"]) / 2
    slot = {}
    for kind, text, font, h in plan["rows"]:
        if kind == "cells":
            slot["y"] = y
        else:
            _centered(draw, text, font, size[0] / 2, y, fill)
        y += h + plan["gap"]
    return img, slot["y"]


# --------------------------------------------------------------------------- #
# Layers.
# --------------------------------------------------------------------------- #


def _opacity(rows):
    keys = []
    for t, value, interp in rows:
        if keys and t <= keys[-1].time + 1e-6:
            keys.pop()  # a later row at the same instant wins
        keys.append(Keyframe(t, value, interp=interp))
    return Property(0.0, keyframes=keys)


def _layer(image, name, card, transform):
    return CachedAVLayer(
        rgba_still(np.asarray(image), card.duration),
        name,
        transform=transform,
        in_point=card.start,
        out_point=card.end,
        start_time=card.start,
    )


def _gate(card, index, on):
    """Opacity rows: ``on`` is the value inside this action's windows.

    Outside the windows the opposite (0 or 100) applies, with the card fade
    applied to the dim state only; active windows lie between the fades.
    """
    inside = 100.0 if on else 0.0
    outside = 0.0 if on else 100.0
    d, f = card.duration, card.fade
    rows = [(0.0, 0.0 if (f > 0 or on) else 100.0, "hold" if on else "linear")]
    rows.append((f, outside, "hold"))
    for i, begin, end in card.windows():
        if i != index:
            continue
        # the first window starts right after the fade-in key; nudge the switch
        # so the dim button still fades in with the rest of the card
        begin = max(begin, f + 1e-3) if f > 0 else begin
        rows += [(begin, inside, "hold"), (end, outside, "hold")]
    rows.append((d - f, outside, "linear"))
    rows.append((d, 0.0, "linear"))
    return _opacity(rows)


def _pop_scale(card, index):
    rows = []
    for i, begin, _ in card.windows():
        if i != index:
            continue
        for n, (frame, pct) in enumerate(_POP):
            t = begin + frame / _POP_FPS
            if rows and t <= rows[-1].time + 1e-6:
                continue
            last = n == len(_POP) - 1
            rows.append(
                Keyframe(t, (pct, pct), interp="hold")
                if last
                else Keyframe(t, (pct, pct), interp="bezier", out_ease=_OUT)
            )
    return Property((100.0, 100.0), value_type="vec2", keyframes=rows)


def end_card_layers(comp, card, *, font=None, title_font=None):
    """Add the layers of the closing carousel card to ``comp``.

    Parameters
    ----------
    comp : Composition
        Target; its size is the canvas (geometry scales from 1920x1080).
    card : EndCard
        Content and timing.
    font : str, optional
        Font path for labels, headline, line and question; ``None`` uses
        Pillow's default font.
    title_font : str, optional
        Font path of the channel name; defaults to ``font``.

    Returns
    -------
    list of CachedAVLayer
        ``panel``, then per action a dim and an active layer, lowest first.
        The dim layer is hidden while its action is highlighted; the active
        layer shows only then and pops from 80 % to 106 % to 100 % at the start
        of each of its windows. All keyframes are on the layer clock.

    Raises
    ------
    ValueError
        If the text cannot fit the panel even after shrinking to 55 %.
    """
    if not isinstance(card, EndCard):
        raise TypeError("card must be an EndCard")
    if title_font is None:
        title_font = font
    W, H, s = _canvas(comp.size)
    plan = _fit_plan(card, s, font, title_font)
    px, py, pw, ph = plan["rect"]
    if py + ph > SUBTITLE_SAFE_RECT[1] * s + 1e-6:
        raise ValueError("end card panel would cover the subtitle safe rect")
    panel, cells_y = _panel_image(card, plan)
    layers = [
        _layer(
            panel,
            "end card panel",
            card,
            Transform(
                anchor_point=(0.0, 0.0),
                position=(float(round(px)), float(round(py))),
                opacity=_opacity(_fade_rows(card)),
                interpolation="linear",
            ),
        )
    ]
    n = len(card.actions)
    pitch, cell_h = plan["pitch"], plan["cell_h"]
    for i in range(n):
        cx = px + pw / 2 + (i - (n - 1) / 2) * pitch
        cy = py + cells_y + cell_h / 2
        w = round(pitch)
        # the anchor is the cell centre, so the pop grows from the middle
        base = dict(
            anchor_point=(w / 2, cell_h / 2),
            position=(float(round(cx)), float(round(cy))),
            interpolation="linear",
        )
        for active in (False, True):
            image = _cell_image(card, plan, i, active)
            extra = {"scale": _pop_scale(card, i)} if active else {}
            layers.append(
                _layer(
                    image,
                    f"end card {card.actions[i]} {'active' if active else 'dim'}",
                    card,
                    Transform(opacity=_gate(card, i, active), **base, **extra),
                )
            )
    for layer in layers:
        comp.add_layer(layer)
    return layers


def _fade_rows(card):
    d, f = card.duration, card.fade
    if f <= 0:
        return [(0.0, 100.0, "linear"), (d, 100.0, "linear")]
    return [(0.0, 0.0, "linear"), (f, 100.0, "linear"),
            (d - f, 100.0, "linear"), (d, 0.0, "linear")]  # fmt: skip


def add_end_card(comp, card, **kw):
    """Add the closing card to ``comp``.

    Parameters
    ----------
    comp : Composition
        Target composition.
    card : EndCard or dict
        The card (a dict goes through ``EndCard.from_dict``).
    ``**kw``
        ``font`` and ``title_font``, passed to ``end_card_layers``.

    Returns
    -------
    list of CachedAVLayer
        The added layers.
    """
    if isinstance(card, dict):
        card = EndCard.from_dict(card)
    return end_card_layers(comp, card, **kw)
