"""Name card placed beside a character portrait (never over the face or subtitles).

A story video shows a character illustration in part of the frame; this
template puts a small vertical name card (name, role, optional seal) next to
that person. ``place_name_tag`` is pure geometry: it picks the side with the
most free room, keeps the card off the subject box and off any ``avoid``
rectangle such as the subtitle safe rect, and clamps it inside the frame.
``name_tag_image`` draws the card (translucent panel, two gold corner brackets,
vertical name column, smaller role column, seal) and ``name_tag_layer`` turns
it into AE layers with eased fade and slide keyframes. All geometry is authored
on a 1920x1080 reference and scaled by ``min(width / 1920, height / 1080)``.

Examples
--------
>>> from moviepy.ae.templates.name_tag import place_name_tag
>>> place_name_tag((1200, 200, 400, 600), (120, 300), (1920, 1080))
(1056, 200, 'left')
"""

import math
from dataclasses import dataclass, fields, replace

import numpy as np
from PIL import Image, ImageDraw

from moviepy.ae.layers.av import AVLayer
from moviepy.ae.properties import Keyframe, Property
from moviepy.ae.properties.easing import Ease
from moviepy.ae.templates.paper import load_font, rgba_still, seal_stamp
from moviepy.ae.templates.presets import ChannelPreset, get_preset
from moviepy.ae.templates.quote import _render_column
from moviepy.ae.transform import Transform


__all__ = [
    "NameTag",
    "REFERENCE_SIZE",
    "SUBTITLE_SAFE_RECT",
    "add_name_tags",
    "name_tag_image",
    "name_tag_layer",
    "place_name_tag",
]

REFERENCE_SIZE = (1920, 1080)
SUBTITLE_SAFE_RECT = (280, 800, 1400, 100)
# Card scales tried in turn when the full-size card has no room.
_SHRINK_STEPS = (1.0, 0.9, 0.8, 0.7)
_SIDES = ("auto", "left", "right", "above", "below")
_ALIGNS = ("top", "center", "bottom", "left", "right")
_ORIENTATIONS = ("vertical", "horizontal")
_EASE_OUT = Ease.bezier(0.16, 1.0, 0.3, 1.0)
_EASE_IN = Ease.bezier(0.7, 0.0, 0.84, 0.0)
_MIN_PX = 14
# direction a card moves while sliding in: away from the subject
_AWAY = {"left": (-1, 0), "right": (1, 0), "above": (0, -1), "below": (0, 1)}


# --------------------------------------------------------------------------- #
# Geometry.
# --------------------------------------------------------------------------- #


def _rect(value, name):
    try:
        x, y, w, h = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be four numbers (x, y, w, h)") from None
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError(f"{name} must be finite")
    if w <= 0 or h <= 0:
        raise ValueError(f"{name} width and height must be positive")
    return x, y, w, h


def _hit(a, b):
    """Report whether two ``(x, y, w, h)`` rects overlap by more than an edge."""
    return (
        a[0] < b[0] + b[2]
        and b[0] < a[0] + a[2]
        and a[1] < b[1] + b[3]
        and b[1] < a[1] + a[3]
    )


def _along(pref, size, low, high, blockers, vertical):
    """Return positions along one axis in ``[low, high]``, nearest ``pref`` first."""
    cands = [pref]
    for bx, by, bw, bh in blockers:
        if vertical:
            cands += [by - size, by + bh]
        else:
            cands += [bx - size, bx + bw]
    cands = {min(max(c, low), high) for c in cands}
    return sorted(cands, key=lambda c: abs(c - pref))


def _preferred(align, start, extent, size):
    """Top-left coordinate of a ``size`` span aligned inside ``extent``."""
    if align in ("center",):
        return start + (extent - size) / 2
    if align in ("bottom", "right"):
        return start + extent - size
    return start


def place_name_tag(
    subject_box,
    card_size,
    canvas_size,
    *,
    side="auto",
    gap=24,
    margin=48,
    avoid=(),
    align="top",
):
    """Choose where a name card goes beside a subject.

    Parameters
    ----------
    subject_box : sequence of 4 numbers
        ``(x, y, w, h)`` pixels of the person in the frame.
    card_size : sequence of 2 numbers
        ``(w, h)`` of the card.
    canvas_size : sequence of 2 numbers
        ``(w, h)`` of the frame.
    side : {"auto", "left", "right", "above", "below"}, optional
        ``"auto"`` tries the side with the most free room first, left and
        right before above and below. An explicit side is never swapped.
    gap : float, optional
        Distance between the subject box and the card.
    margin : float, optional
        Minimum distance between the card and the frame edge.
    avoid : sequence of rects, optional
        ``(x, y, w, h)`` rectangles the card must not overlap, for example the
        subtitle safe rect. The card slides along the subject edge to dodge them
        but always keeps some overlap with that edge, so it stays beside the
        subject rather than drifting next to someone else.
    align : {"top", "center", "bottom", "left", "right"}, optional
        Where the card sits along the subject edge. ``"top"``/``"left"`` and
        ``"bottom"``/``"right"`` are interchangeable between the two axes.

    Returns
    -------
    tuple
        ``(x, y, side_used)`` with the card's top-left corner in integer pixels.

    Raises
    ------
    ValueError
        When the inputs are invalid or no side has room for the card.

    Examples
    --------
    >>> place_name_tag((1200, 200, 400, 600), (120, 300), (1920, 1080))
    (1056, 200, 'left')
    >>> place_name_tag((100, 200, 400, 600), (120, 300), (1920, 1080))
    (524, 200, 'right')
    >>> place_name_tag((100, 200, 400, 300), (120, 300), (1920, 1080), side="below")
    (100, 524, 'below')
    >>> # the subtitle strip pushes a tall card upward
    >>> place_name_tag(
    ...     (100, 200, 400, 600), (120, 300), (1920, 1080), side="right",
    ...     avoid=[(280, 800, 1400, 100)], align="bottom")
    (524, 500, 'right')
    >>> place_name_tag((10, 10, 1900, 1060), (120, 300), (1920, 1080))
    Traceback (most recent call last):
    ...
    ValueError: no room for a 120x300 card beside the subject box
    """
    sx, sy, sw, sh = subject = _rect(subject_box, "subject_box")
    try:
        cw, ch = (float(v) for v in card_size)
        W, H = (float(v) for v in canvas_size)
    except (TypeError, ValueError):
        raise ValueError("card_size and canvas_size must be (w, h) pairs") from None
    if not all(map(math.isfinite, (cw, ch, W, H))) or min(cw, ch, W, H) <= 0:
        raise ValueError("card_size and canvas_size must be positive and finite")
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    if align not in _ALIGNS:
        raise ValueError(f"align must be one of {_ALIGNS}, got {align!r}")
    gap, margin = float(gap), float(margin)
    if gap < 0 or margin < 0:
        raise ValueError("gap and margin must not be negative")
    blockers = [_rect(r, "avoid rect") for r in avoid]
    room = {
        "left": sx - margin,
        "right": W - margin - (sx + sw),
        "above": sy - margin,
        "below": H - margin - (sy + sh),
    }
    if side == "auto":
        order = sorted(("right", "left"), key=lambda s: -room[s]) + sorted(
            ("below", "above"), key=lambda s: -room[s]
        )
    else:
        order = [side]
    low_x, high_x = margin, W - margin - cw
    low_y, high_y = margin, H - margin - ch
    for used in order:
        if used in ("left", "right"):
            x = sx - gap - cw if used == "left" else sx + sw + gap
            if x < low_x or x > high_x or low_y > high_y:
                continue
            pref = _preferred(align, sy, sh, ch)
            for y in _along(pref, ch, low_y, high_y, blockers, True):
                if y >= sy + sh or y + ch <= sy:
                    continue  # slid past the subject: no longer beside it
                spot = (round(x), round(y), cw, ch)
                if not _hit(spot, subject) and not any(_hit(spot, b) for b in blockers):
                    return spot[0], spot[1], used
        else:
            y = sy - gap - ch if used == "above" else sy + sh + gap
            if y < low_y or y > high_y or low_x > high_x:
                continue
            pref = _preferred(align, sx, sw, cw)
            for x in _along(pref, cw, low_x, high_x, blockers, False):
                if x >= sx + sw or x + cw <= sx:
                    continue
                spot = (round(x), round(y), cw, ch)
                if not _hit(spot, subject) and not any(_hit(spot, b) for b in blockers):
                    return spot[0], spot[1], used
    where = "beside the subject box" if side == "auto" else f"on side {side!r}"
    extra = " that avoids the given rects" if blockers else ""
    raise ValueError(f"no room for a {cw:g}x{ch:g} card {where}{extra}")


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


def _num(value, name, low=None, high=None, strict_low=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if low is not None and (value <= low if strict_low else value < low):
        raise ValueError(f"{name} must be {'>' if strict_low else '>='} {low:g}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be <= {high:g}")
    return value


@dataclass(frozen=True)
class NameTag:
    """One name card: text, subject box, timing and look (1920x1080 reference).

    Attributes
    ----------
    name : str
        Character name, required.
    role : str, optional
        Smaller line or column beside the name, such as 狐仙 or 書生.
    subject_box : tuple
        ``(x, y, w, h)`` of the person on the 1920x1080 reference frame.
    start, duration : float
        Appearance time and length in seconds.
    side : {"auto", "left", "right", "above", "below"}
        Card side; ``"auto"`` picks the side with the most room.
    orientation : {"vertical", "horizontal"}
        Vertical columns suit CJK; horizontal puts the name on one line and the
        role below it.
    fade_in, fade_out : float
        Opacity ramps (R26 card: 0.45 s in, 0.65 s out).
    slide_px : float
        The card slides this far (reference px) away from the subject while
        fading in; limited to ``gap`` so it never crosses the subject box.
    seal : str, optional
        One or two characters stamped as a red seal, ``None`` to omit.
    leader : bool
        Add a thin line from the card edge to the subject box.
    font_px : float
        Name size in reference px (the image shrinks it when space is tight).
    gap, margin, align
        Placement parameters, see ``place_name_tag``.
    panel_color, panel_alpha, bracket_color, name_color, role_color, seal_color
        Colours: RGB triples, and panel opacity in 0-1.

    Examples
    --------
    >>> tag = NameTag("Jiaona", role="Fox", subject_box=(1200, 200, 400, 600))
    >>> NameTag.from_dict(tag.to_dict()) == tag
    True
    """

    name: str
    role: "str | None" = None
    subject_box: tuple = (0, 0, 1, 1)
    start: float = 0.0
    duration: float = 4.0
    side: str = "auto"
    orientation: str = "vertical"
    fade_in: float = 0.45
    fade_out: float = 0.65
    slide_px: float = 24.0
    seal: "str | None" = None
    leader: bool = False
    font_px: float = 56.0
    gap: float = 24.0
    margin: float = 48.0
    align: str = "top"
    panel_color: tuple = (12, 17, 23)
    panel_alpha: float = 0.8
    bracket_color: tuple = (216, 182, 147)
    name_color: tuple = (255, 228, 183)
    role_color: tuple = (242, 233, 221)
    seal_color: tuple = (176, 30, 28)

    def __post_init__(self):
        sset = object.__setattr__
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        sset(self, "name", self.name.strip())
        if self.role is not None:
            if not isinstance(self.role, str):
                raise ValueError("role must be a string or None")
            sset(self, "role", self.role.strip() or None)
        if self.seal is not None:
            if not isinstance(self.seal, str) or not 1 <= len(self.seal.strip()) <= 2:
                raise ValueError("seal must be one or two characters, or None")
            sset(self, "seal", self.seal.strip())
        sset(self, "subject_box", _rect(self.subject_box, "subject_box"))
        sset(self, "start", _num(self.start, "start", 0.0))
        sset(self, "duration", _num(self.duration, "duration", 0.0, strict_low=True))
        for key in ("fade_in", "fade_out", "slide_px", "gap", "margin"):
            sset(self, key, _num(getattr(self, key), key, 0.0))
        sset(self, "font_px", _num(self.font_px, "font_px", 0.0, strict_low=True))
        sset(self, "panel_alpha", _num(self.panel_alpha, "panel_alpha", 0.0, 1.0))
        if self.side not in _SIDES:
            raise ValueError(f"side must be one of {_SIDES}, got {self.side!r}")
        if self.align not in _ALIGNS:
            raise ValueError(f"align must be one of {_ALIGNS}, got {self.align!r}")
        if self.orientation not in _ORIENTATIONS:
            raise ValueError(f"orientation must be one of {_ORIENTATIONS}")
        sset(self, "leader", bool(self.leader))
        for key in (
            "panel_color",
            "bracket_color",
            "name_color",
            "role_color",
            "seal_color",
        ):
            sset(self, key, _rgb(getattr(self, key), key))

    @property
    def end(self):
        """Time in seconds when the card has fully faded out."""
        return self.start + self.duration

    def to_dict(self):
        """Return a JSON-friendly dict that ``from_dict`` reads back."""
        out = {}
        for item in fields(self):
            value = getattr(self, item.name)
            out[item.name] = list(value) if isinstance(value, tuple) else value
        return out

    @classmethod
    def from_dict(cls, data):
        """Build a tag from a dict such as ``json.load`` returns.

        Raises
        ------
        ValueError
            On unknown keys or invalid values.
        """
        if not isinstance(data, dict):
            raise ValueError("a name tag config must be a mapping")
        known = {item.name for item in fields(cls)}
        extra = sorted(set(data) - known)
        if extra:
            raise ValueError(f"unknown name tag keys: {extra}")
        return cls(**data)


# --------------------------------------------------------------------------- #
# Card image.
# --------------------------------------------------------------------------- #


def _column(text, cell, font, color):
    """Render ``text`` as one vertical column (straight RGBA, ``cell`` wide)."""
    array = _render_column(list(text), font, int(cell * 0.86), cell, color)
    return Image.fromarray(array)


def _brackets(draw, w, h, inset, length, thick, color):
    fill = (*color, 255)
    draw.rectangle([inset, inset, inset + length, inset + thick - 1], fill=fill)
    draw.rectangle([inset, inset, inset + thick - 1, inset + length], fill=fill)
    x1, y1 = w - 1 - inset, h - 1 - inset
    draw.rectangle([x1 - length, y1 - thick + 1, x1, y1], fill=fill)
    draw.rectangle([x1 - thick + 1, y1 - length, x1, y1], fill=fill)


def _panel(tag, size):
    return Image.new("RGBA", size, (*tag.panel_color, round(255 * tag.panel_alpha)))


def _stamp(tag, size, font):
    return Image.fromarray(seal_stamp(tag.seal, size, tag.seal_color, 3, font))


def _vertical_card(tag, fs, scale, font, role_font):
    rf = max(8, round(fs * 0.46))
    pad_x, pad_y = round(fs * 0.62), round(fs * 0.55)
    sep = round(fs * 0.34)
    seal_sz = round(fs * 0.62) if tag.seal else 0
    col1 = _column(tag.name, round(fs * 1.12), font, tag.name_color)
    col2 = None
    if tag.role:
        col2 = _column(tag.role, round(rf * 1.2), role_font, tag.role_color)
    role_h = col2.height if col2 else 0
    seal_gap = round(fs * 0.2) if (col2 and seal_sz) else 0
    side_h = role_h + seal_gap + seal_sz
    side_w = max(col2.width if col2 else 0, seal_sz)
    body_h = max(col1.height, side_h)
    card = _panel(
        tag,
        (pad_x * 2 + col1.width + (sep + side_w if side_w else 0), pad_y * 2 + body_h),
    )
    # Chinese reads right to left: name column right, role and seal column left.
    x1 = card.width - pad_x - col1.width
    card.alpha_composite(col1, (x1, pad_y + (body_h - col1.height) // 2))
    if side_w:
        if col2 is not None:
            card.alpha_composite(col2, (pad_x + (side_w - col2.width) // 2, pad_y))
        if seal_sz:
            card.alpha_composite(
                _stamp(tag, seal_sz, font),
                (pad_x + (side_w - seal_sz) // 2, pad_y + body_h - seal_sz),
            )
        rule = Image.new("RGBA", card.size, (0, 0, 0, 0))
        lx = x1 - sep // 2
        ImageDraw.Draw(rule).line(
            [(lx, pad_y), (lx, pad_y + body_h - 1)],
            fill=(*tag.bracket_color, 90),
            width=max(1, round(scale)),
        )
        card.alpha_composite(rule)
    return card


def _horizontal_card(tag, fs, font, role_font):
    rf = max(8, round(fs * 0.5))
    pad_x, pad_y = round(fs * 0.62), round(fs * 0.55)
    sep = round(fs * 0.34)
    f_name, f_role = load_font(font, fs), load_font(role_font, rf)
    a1, d1 = f_name.getmetrics()
    a2, d2 = f_role.getmetrics()
    line_gap = round(fs * 0.16) if tag.role else 0
    body_h = a1 + d1 + (line_gap + a2 + d2 if tag.role else 0)
    seal_sz = round(body_h * 0.78) if tag.seal else 0
    text_w = math.ceil(
        max(f_name.getlength(tag.name), f_role.getlength(tag.role or ""))
    )
    card = _panel(
        tag,
        (pad_x * 2 + text_w + (sep + seal_sz if seal_sz else 0), pad_y * 2 + body_h),
    )
    text = Image.new("RGBA", card.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text)
    draw.text((pad_x, pad_y), tag.name, font=f_name, fill=(*tag.name_color, 255))
    if tag.role:
        draw.text(
            (pad_x, pad_y + a1 + d1 + line_gap),
            tag.role,
            font=f_role,
            fill=(*tag.role_color, 255),
        )
    card.alpha_composite(text)
    if seal_sz:
        card.alpha_composite(
            _stamp(tag, seal_sz, font),
            (card.width - pad_x - seal_sz, pad_y + (body_h - seal_sz) // 2),
        )
    return card


def _build_card(tag, fs, scale, font, role_font):
    """Draw the card at name size ``fs`` px; return a PIL RGBA image."""
    if tag.orientation == "vertical":
        card = _vertical_card(tag, fs, scale, font, role_font)
    else:
        card = _horizontal_card(tag, fs, font, role_font)
    corners = Image.new("RGBA", card.size, (0, 0, 0, 0))
    _brackets(
        ImageDraw.Draw(corners),
        card.width,
        card.height,
        max(2, round(fs * 0.14)),
        max(6, round(fs * 0.5)),
        max(2, round(fs * 0.055)),
        tag.bracket_color,
    )
    card.alpha_composite(corners)
    return card


def name_tag_image(tag, *, font=None, role_font=None, scale=1.0, max_size=None):
    """Draw the name card as a transparent-background RGBA image.

    The panel is translucent dark, two gold corner brackets mark the top-left
    and bottom-right corners (as in the R26 ASS card), the name is a large
    vertical column with the smaller role column beside it, and an optional red
    seal sits under the role. The name font shrinks until the card fits
    ``max_size``; text is never cut.

    Parameters
    ----------
    tag : NameTag
        Card content and colours.
    font, role_font : str, optional
        Font file paths for name and role. ``None`` uses Pillow's default font;
        ``role_font`` defaults to ``font``.
    scale : float, optional
        Pixels per reference pixel (``min(width / 1920, height / 1080)``).
    max_size : tuple, optional
        ``(w, h)`` limit in output pixels; default ``(900, 620) * scale``.

    Returns
    -------
    PIL.Image.Image
        ``RGBA`` image; pixels outside the panel are fully transparent.

    Raises
    ------
    ValueError
        When the text does not fit even at the smallest font size.

    Examples
    --------
    >>> img = name_tag_image(NameTag("Jiaona", role="Fox"), scale=0.5)
    >>> img.mode, img.getpixel((img.width // 2, 1))[3] > 0
    ('RGBA', True)
    """
    scale = _num(scale, "scale", 0.0, strict_low=True)
    role_font = font if role_font is None else role_font
    if max_size is None:
        limit = (900 * scale, 620 * scale)
    else:
        limit = tuple(float(v) for v in max_size)
    fs = max(_MIN_PX, round(tag.font_px * scale))
    while True:
        card = _build_card(tag, fs, scale, font, role_font)
        if card.width <= limit[0] and card.height <= limit[1]:
            return card
        if fs <= _MIN_PX:
            raise ValueError(
                f"name card for {tag.name!r} needs {card.width}x{card.height}px even "
                f"at {_MIN_PX}px text, limit {limit[0]:.0f}x{limit[1]:.0f}; "
                "shorten the name/role or use orientation='horizontal'"
            )
        fs = max(_MIN_PX, int(fs * 0.92))


# --------------------------------------------------------------------------- #
# Layers.
# --------------------------------------------------------------------------- #


def _canvas(canvas_size):
    try:
        w, h = (int(v) for v in canvas_size)
    except (TypeError, ValueError):
        raise ValueError("canvas_size must be (w, h) integers") from None
    if w <= 0 or h <= 0:
        raise ValueError("canvas_size must be positive")
    return w, h, min(w / REFERENCE_SIZE[0], h / REFERENCE_SIZE[1])


def _keys(rows):
    keys = []
    for time, value, *ease in rows:
        if keys and time <= keys[-1].time + 1e-6:
            continue
        if ease:
            keys.append(Keyframe(time, value, interp="bezier", out_ease=ease[0]))
        else:
            keys.append(Keyframe(time, value))
    return keys


def _prepare(tag, canvas_size, preset, font, role_font, avoid):
    """Render the card and find its place; return everything layers need."""
    if not isinstance(tag, NameTag):
        raise TypeError("tag must be a NameTag")
    W, H, s = _canvas(canvas_size)
    if preset is not None:
        if isinstance(preset, str):
            preset = get_preset(preset)
        if not isinstance(preset, ChannelPreset):
            raise TypeError("preset must be a ChannelPreset or preset name")
        font = preset.font("title", font)
        if role_font is not None or "body" in preset.fonts:
            role_font = preset.font("body", role_font)
        else:
            role_font = font
    subject = tuple(v * s for v in tag.subject_box)
    if subject[0] + subject[2] > W + 1e-6 or subject[1] + subject[3] > H + 1e-6:
        raise ValueError(
            f"subject_box {tag.subject_box} of {tag.name!r} is outside the "
            f"{W}x{H} canvas (boxes use 1920x1080 reference pixels)"
        )
    if avoid is None:
        avoid = [SUBTITLE_SAFE_RECT]
    avoid = [tuple(v * s for v in _rect(r, "avoid rect")) for r in avoid]
    margin = tag.margin * s
    # In a crowded frame the full-size card may not fit beside the subject:
    # shrink it in steps (down to 70 %), then try a horizontal card, which is
    # much shorter, before giving up.
    variants = [tag]
    if tag.orientation == "vertical":
        variants.append(replace(tag, orientation="horizontal"))
    attempts = [(v, k) for v in variants for k in _SHRINK_STEPS]
    for n, (variant, shrink) in enumerate(attempts):
        image = name_tag_image(
            variant,
            font=font,
            role_font=role_font,
            scale=s * shrink,
            max_size=(W * 0.5, H - 2 * margin),
        )
        try:
            x, y, side = place_name_tag(
                subject,
                image.size,
                (W, H),
                side=tag.side,
                gap=tag.gap * s,
                margin=margin,
                avoid=avoid,
                align=tag.align,
            )
            tag = variant
            break
        except ValueError:
            if n == len(attempts) - 1:
                raise
    return {
        "tag": tag,
        "image": image,
        "x": x,
        "y": y,
        "side": side,
        "scale": s,
        "subject": subject,
        "rect": (x, y, image.width, image.height),
    }


def _leader_image(prep):
    """Draw the thin leader line; return ``(image, x, y)`` of its top-left."""
    tag, s = prep["tag"], prep["scale"]
    x, y, w, h = prep["rect"]
    sx, sy, sw, sh = prep["subject"]
    a = {
        "left": (x + w, y + h / 2),
        "right": (x, y + h / 2),
        "above": (x + w / 2, y + h),
        "below": (x + w / 2, y),
    }[prep["side"]]
    b = (min(max(a[0], sx), sx + sw), min(max(a[1], sy), sy + sh))
    pad = max(4, math.ceil(4 * s))
    x0 = math.floor(min(a[0], b[0])) - pad
    y0 = math.floor(min(a[1], b[1])) - pad
    x1 = math.ceil(max(a[0], b[0])) + pad
    y1 = math.ceil(max(a[1], b[1])) + pad
    ss = 4
    img = Image.new("RGBA", ((x1 - x0) * ss, (y1 - y0) * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pa = ((a[0] - x0) * ss, (a[1] - y0) * ss)
    pb = ((b[0] - x0) * ss, (b[1] - y0) * ss)
    color = (*tag.bracket_color, 220)
    draw.line([pa, pb], fill=color, width=max(1, round(2 * s * ss)))
    r = max(2.0, 3.5 * s) * ss
    draw.ellipse([pb[0] - r, pb[1] - r, pb[0] + r, pb[1] + r], fill=color)
    return img.resize((x1 - x0, y1 - y0), Image.LANCZOS), x0, y0


def _layers(prep):
    tag, s = prep["tag"], prep["scale"]
    t0, t1 = tag.start, tag.end
    fin, fout = tag.fade_in, tag.fade_out
    if fin + fout > tag.duration:
        k = tag.duration / (fin + fout)
        fin, fout = fin * k, fout * k
    rows = [(t0, 0.0), (t0 + fin, 100.0, _EASE_OUT), (t1 - fout, 100.0), (t1, 0.0)]

    def opacity():
        return Property(100.0, keyframes=_keys(rows))

    out = []
    if tag.leader:
        img, lx, ly = _leader_image(prep)
        out.append(
            AVLayer(
                rgba_still(np.asarray(img), t1),
                f"{tag.name} leader",
                in_point=t0,
                out_point=t1,
                transform=Transform(
                    anchor_point=(0.0, 0.0),
                    position=(float(lx), float(ly)),
                    opacity=opacity(),
                ),
            )
        )
    slide = min(tag.slide_px, tag.gap) * s
    dx, dy = _AWAY[prep["side"]]
    x, y = float(prep["x"]), float(prep["y"])
    # starts nearer the subject, drifts away to its rest place, eases back on exit
    position = Property(
        (x, y),
        value_type="vec2",
        keyframes=_keys(
            [
                (t0, (x - dx * slide, y - dy * slide), _EASE_OUT),
                (t0 + fin, (x, y)),
                (t1 - fout, (x, y), _EASE_IN),
                (t1, (x - dx * slide * 0.5, y - dy * slide * 0.5)),
            ]
        ),
    )
    out.append(
        AVLayer(
            rgba_still(np.asarray(prep["image"]), t1),
            f"{tag.name} name tag",
            in_point=t0,
            out_point=t1,
            transform=Transform(
                anchor_point=(0.0, 0.0), position=position, opacity=opacity()
            ),
        )
    )
    return out


def name_tag_layer(
    tag, canvas_size, *, preset=None, font=None, role_font=None, avoid=None
):
    """Build the AE layers of a name card beside its subject.

    Parameters
    ----------
    tag : NameTag
        Content, subject box (1920x1080 reference px) and timing.
    canvas_size : tuple
        ``(w, h)`` of the composition; sizes scale by ``min(w/1920, h/1080)``.
    preset : ChannelPreset or str, optional
        Supplies the ``title`` (and ``body``) font paths when given.
    font, role_font : str, optional
        Font file paths overriding the preset; ``None`` without a preset uses
        Pillow's default font.
    avoid : sequence of rects, optional
        Reference-pixel rectangles the card must not cover. ``None`` uses the
        R26 subtitle safe rect ``(280, 800, 1400, 100)``; pass ``[]`` for none.

    Returns
    -------
    list of AVLayer
        ``[leader, card]`` (leader only when ``tag.leader``), lowest first, for
        ``comp.add_layer``. Opacity and position are keyframed in composition
        time over ``[tag.start, tag.end]``; the card slides away from the
        subject while fading in.

    Raises
    ------
    ValueError
        When the card does not fit beside the subject even after shrinking
        to 70 percent and switching a vertical card to horizontal, or the box
        leaves the frame.
    """
    return _layers(_prepare(tag, canvas_size, preset, font, role_font, avoid))


def add_name_tags(comp, tags, **kw):
    """Add several name tags to ``comp``.

    Parameters
    ----------
    comp : Composition
        Target composition; its size is the canvas.
    tags : iterable of NameTag or dict
        Tags (dicts go through ``NameTag.from_dict``).
    ``**kw``
        Passed to ``name_tag_layer`` (``preset``, ``font``, ``role_font``,
        ``avoid``).

    Returns
    -------
    list of AVLayer
        All added layers.

    Raises
    ------
    ValueError
        If two tags overlap in time and their cards (or a card and the other
        tag's subject box) overlap in space. Nothing is added in that case.
    """
    tags = [NameTag.from_dict(t) if isinstance(t, dict) else t for t in tags]
    unknown = set(kw) - {"preset", "font", "role_font", "avoid"}
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
    preps = [
        _prepare(
            t,
            comp.size,
            kw.get("preset"),
            kw.get("font"),
            kw.get("role_font"),
            kw.get("avoid"),
        )
        for t in tags
    ]
    for i, a in enumerate(preps):
        for b in preps[i + 1 :]:
            ta, tb = a["tag"], b["tag"]
            if not (ta.start < tb.end and tb.start < ta.end):
                continue
            if (
                _hit(a["rect"], b["rect"])
                or _hit(a["rect"], b["subject"])
                or _hit(b["rect"], a["subject"])
            ):
                raise ValueError(
                    f"name tags {ta.name!r} and {tb.name!r} overlap in time "
                    f"({ta.start:g}-{ta.end:g}s vs {tb.start:g}-{tb.end:g}s) "
                    "and in space; change side/align/subject_box or stagger them"
                )
    layers = [layer for prep in preps for layer in _layers(prep)]
    for layer in layers:
        comp.add_layer(layer)
    return layers
