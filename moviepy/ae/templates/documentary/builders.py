"""Build the two photographic references as native, editable AE layer stacks."""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np

from moviepy.ae import (
    CameraFraming,
    Composition,
    Ease,
    Keyframe,
    Mask,
    Property,
    Transform,
)
from moviepy.ae.effects.perspective.drop_shadow import DropShadow
from moviepy.ae.masks.path import polygon, rounded_rect
from moviepy.ae.templates.documentary._art import (
    article_art,
    font_path,
    frame_hud,
    text_image,
    textured_background,
)
from moviepy.ae.templates.paper import CachedAVLayer, rgba_still
from moviepy.ae.templates.presets import ChannelPreset, SubtitleStyle
from moviepy.ae.templates.subtitles import subtitle_layer
from moviepy.video.VideoClip import ImageClip, VideoClip


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    return value


def _pair(value, name, low, high):
    try:
        pair = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain two numbers") from error
    if len(pair) != 2:
        raise ValueError(f"{name} must contain two numbers")
    return tuple(_number(v, name, low, high) for v in pair)


def _source(clip):
    if not isinstance(clip, VideoClip) or clip.is_mask:
        raise TypeError("source must be an RGB MoviePy VideoClip / ImageClip")
    if min(clip.size) < 2:
        raise ValueError("source dimensions must be at least 2 pixels")
    if clip.duration is not None:
        _number(clip.duration, "source duration", 1e-9, 86400)
    return clip


def _settings(size, duration, fps, source=None):
    try:
        size = tuple(size)
    except TypeError as error:
        raise ValueError("size must contain two integer pixel dimensions") from error
    if len(size) != 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 64
        for v in size
    ):
        raise ValueError("size must contain two integer dimensions >= 64")
    if duration is None:
        duration = getattr(source, "duration", None)
        if duration is None:
            raise ValueError("duration is required for a still / unbounded clip")
    duration = _number(duration, "duration", 1e-6, 86400)
    fps = _number(fps, "fps", 1, 240)
    return tuple(map(int, size)), duration, fps


def _keys(start, end, last, kind="float"):
    if last <= 0:
        return Property(start, value_type=kind)
    return Property(
        [Keyframe(0, start, out_ease=Ease.easy_ease()), Keyframe(last, end)],
        value_type=kind,
    )


def _still(comp, array, name, position=(0, 0)):
    clip = (
        rgba_still(array, comp.duration)
        if array.shape[2] == 4
        else ImageClip(np.asarray(array), duration=comp.duration)
    )
    layer = CachedAVLayer(
        clip,
        name,
        audio_enabled=False,
        transform=Transform(anchor_point=(0, 0), position=position),
    )
    return comp.add_layer(layer)


def _viewport(source, size, duration, fps, *, fit="cover", push=0, pan=(0, 0)):
    """Fit footage without stretching; preserve its audio through an AVLayer."""
    if fit not in ("cover", "contain"):
        raise ValueError("fit must be 'cover' or 'contain'")
    last = max(0, duration - 1 / fps)
    comp = Composition(size=size, duration=duration, fps=fps, name="Footage viewport")
    if fit == "cover":
        framing = CameraFraming(
            source.size,
            size,
            overscan=1.005,
            zoom=_keys(1.0, 1.0 + push, last),
            pan=_keys((0, 0), (pan[0] * source.w, pan[1] * source.h), last, "vec2"),
        )
        transform = framing.to_transform()
    else:
        if pan != (0, 0):
            raise ValueError("contain uses centered footage; pan must be (0, 0)")
        factor = min((size[0] - 1) / (source.w - 1), (size[1] - 1) / (source.h - 1))
        end = factor * 100
        transform = Transform(
            position=((size[0] - 1) / 2, (size[1] - 1) / 2),
            scale=_keys((end / (1 + push),) * 2, (end,) * 2, last, "vec2"),
        )
    comp.add_layer(
        CachedAVLayer(source, "footage", out_point=duration, transform=transform)
    )
    return comp


@dataclass(frozen=True)
class CollageCard:
    """One custom clipping, placed using normalized canvas coordinates.

    ``width`` is a fraction of canvas width; aspect ratio is preserved. ``depth``
    multiplies the shared pan/push to produce 2.5D parallax, not reconstructed 3D.
    ``clip`` is borrowed. Only the main card carries audio by default.
    """

    clip: VideoClip
    center: tuple = (0.2, 0.2)
    width: float = 0.32
    rotation: float = 0.0
    depth: float = 0.5
    name: str = "clipping"

    def __post_init__(self):
        _source(self.clip)
        object.__setattr__(self, "center", _pair(self.center, "center", -2, 3))
        for field, low, high in (
            ("width", 0.01, 4),
            ("rotation", -180, 180),
            ("depth", 0, 4),
        ):
            object.__setattr__(
                self, field, _number(getattr(self, field), field, low, high)
            )
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("card name must be non-empty")


def build_news_page(
    source,
    *,
    headline,
    body="",
    source_label="",
    size=(1280, 900),
    duration=None,
    fps=24,
    font=None,
    accent=(205, 27, 49),
    fit="cover",
):
    """Return a news-page precomposition with an independently playing video.

    Headline, left body column, red navigation and source label are newly drawn;
    supply an ImageClip directly to ``build_news_collage`` for a real screenshot.
    Overlong text raises instead of disappearing outside the page.
    """
    _source(source)
    size, duration, fps = _settings(size, duration, fps, source)
    for text, name in (
        (headline, "headline"),
        (body, "body"),
        (source_label, "source_label"),
    ):
        if not isinstance(text, str):
            raise TypeError(f"{name} must be a string")
    if not headline.strip():
        raise ValueError("headline must not be empty")
    accent = tuple(accent)
    if len(accent) != 3:
        raise ValueError("accent must contain three RGB codes")
    accent = tuple(round(_number(v, "accent", 0, 255)) for v in accent)
    font = font_path(
        font, cjk=any(ord(c) > 127 for c in headline + body + source_label)
    )
    art, box = article_art(size, headline, body, source_label, font, accent)
    comp = Composition(size=size, duration=duration, fps=fps, name="News page")
    _still(comp, art, "article.chrome")
    left, top, right, bottom = box
    viewport = _viewport(source, (right - left, bottom - top), duration, fps, fit=fit)
    comp.add_clip(
        viewport,
        "article.media",
        out_point=duration,
        transform=Transform(anchor_point=(0, 0), position=(left, top)),
    )
    comp.documentary_metadata = {"template": "news_page", "media_box": box}
    return comp


def _card_transform(card, size, duration, fps, push, pan, entrance):
    w, h = size
    x, y = card.center[0] * w, card.center[1] * h
    scale = card.width * w / card.clip.w * 100
    last = max(0, duration - 1 / fps)
    end_scale = scale * (1 + push * card.depth)
    end_position = (x + pan[0] * w * card.depth, y + pan[1] * h * card.depth)
    if entrance > 0 and last > 0:
        settle = min(entrance, last * 0.4)
        ease = Ease.easy_ease()
        position = Property(
            [
                Keyframe(0, (x, y + h * 0.05), out_ease=ease),
                Keyframe(settle, (x, y), out_ease=ease),
                Keyframe(last, end_position),
            ],
            value_type="vec2",
        )
        scaling = Property(
            [
                Keyframe(0, (scale * 0.97,) * 2, out_ease=ease),
                Keyframe(settle, (scale,) * 2, out_ease=ease),
                Keyframe(last, (end_scale,) * 2),
            ],
            value_type="vec2",
        )
        opacity = _keys(0.0, 100.0, settle)
    else:
        position = _keys((x, y), end_position, last, "vec2")
        scaling = _keys((scale,) * 2, (end_scale,) * 2, last, "vec2")
        opacity = 100
    return Transform(
        position=position, scale=scaling, rotation=card.rotation, opacity=opacity
    )


def build_news_collage(
    main,
    clippings=(),
    *,
    size=(1920, 1080),
    duration=None,
    fps=24,
    main_width=0.63,
    main_center=(0.52, 0.49),
    push=0.025,
    pan=(0.008, -0.006),
    entrance=0.55,
    seed=7,
    background=None,
):
    """Return the layered news montage from reference 1.

    ``clippings`` accepts up to four clips in the photographed default layout,
    or any number of ``CollageCard`` objects for explicit layouts. Order is back
    to front; the main page is always above them. Camera movement is configurable
    interpretation of a still reference, not a claim about its original motion.
    """
    _source(main)
    size, duration, fps = _settings(size, duration, fps, main)
    push = _number(push, "push", 0, 0.5)
    pan = _pair(pan, "pan", -0.25, 0.25)
    entrance = _number(entrance, "entrance", 0, 60)
    main_card = CollageCard(main, main_center, main_width, 0, 1, "news.main")
    clips = tuple(clippings)
    defaults = (
        ((0.17, 0.22), 0.36, -1.5),
        ((0.98, 0.22), 0.43, 0.8),
        ((0.12, 0.91), 0.37, -1.0),
        ((0.95, 0.88), 0.40, 1.3),
    )
    if len(clips) > 4 and not all(isinstance(c, CollageCard) for c in clips):
        raise ValueError(
            "more than four clippings requires explicit CollageCard layouts"
        )
    cards = []
    for i, clip in enumerate(clips):
        if isinstance(clip, CollageCard):
            cards.append(clip)
        else:
            center, width, rotation = defaults[i]
            cards.append(
                CollageCard(
                    clip,
                    center,
                    width,
                    rotation,
                    0.35 + i * 0.1,
                    f"news.clipping.{i + 1:02d}",
                )
            )
    names = [c.name for c in cards] + [main_card.name, "news.background"]
    if len(names) != len(set(names)):
        raise ValueError(
            "card names must be unique and not use reserved news layer names"
        )
    comp = Composition(size=size, duration=duration, fps=fps, name="News collage")
    if background is None:
        _still(comp, textured_background(size, seed), "news.background")
    else:
        _source(background)
        comp.add_clip(
            _viewport(background, size, duration, fps),
            "news.background",
            audio_enabled=False,
        )
    for card in cards + [main_card]:
        factor = card.width * size[0] / card.clip.w
        layer = CachedAVLayer(
            card.clip,
            card.name,
            out_point=duration,
            audio_enabled=card is main_card,
            transform=_card_transform(card, size, duration, fps, push, pan, entrance),
            effects=[
                DropShadow(
                    opacity=28,
                    distance=10 * min(size[0] / 1920, size[1] / 1080) / factor,
                    softness=20 * min(size[0] / 1920, size[1] / 1080) / factor,
                )
            ],
        )
        comp.add_layer(layer)
    comp.documentary_metadata = {
        "template": "news_collage",
        "size": size,
        "main_width": main_width,
        "motion_inferred": True,
        "input_ownership": "borrowed",
    }
    return comp


def film_window_masks(size, *, cut=40, radius=12, notches=True):
    """Return editable masks: rounded corners, opposing bevels and top notches.

    Pixel-center geometry keeps the outer pixel edges at -0.5 / size-0.5.
    Reference 2 has bevels at BOTH top-left and bottom-right, not four rounds.
    """
    w, h = size
    cut = _number(cut, "cut", 0, min(w, h) / 3)
    radius = _number(radius, "radius", 0, min(w, h) / 3)
    if not isinstance(notches, bool):
        raise ValueError("notches must be bool")
    right, bottom = w - 0.5, h - 0.5
    masks = [
        Mask(
            rounded_rect(((w - 1) / 2, (h - 1) / 2), size, radius),
            name="rounded window",
        )
    ]
    if cut:
        masks += [
            Mask(
                polygon(((-0.5, -0.5), (cut - 0.5, -0.5), (-0.5, cut - 0.5))),
                mode="subtract",
                name="top-left bevel",
            ),
            Mask(
                polygon(
                    ((right, bottom), (right - cut, bottom), (right, bottom - cut))
                ),
                mode="subtract",
                name="bottom-right bevel",
            ),
        ]
    if notches:
        diameter = max(2, w * 0.009)
        for i in range(3):
            masks.append(
                Mask.ellipse(
                    (w * (0.141 + i * 0.016), -0.5),
                    (diameter, diameter),
                    mode="subtract",
                    name=f"top notch {i + 1}",
                )
            )
    return masks


def build_film_frame(
    source,
    *,
    size=(1920, 1080),
    duration=None,
    fps=24,
    window=(0.056, 0.163, 0.894, 0.777),
    fit="cover",
    push=0.035,
    pan=(0.015, 0),
    source_label="",
    watermark="",
    iso="ISO 400",
    font=None,
    seed=11,
    cut=40,
    radius=12,
    notches=True,
    hud=True,
):
    """Return the archival camera/film frame from reference 2.

    The fixed shell/HUD is separate from the moving footage and native masks.
    No film tint, television reflection, scan moire or broadcaster mark is baked
    into the input. ``window`` is normalized (left, top, width, height).
    ``cut`` / ``radius`` are authored pixels on a 1920x1080 reference.
    For ``fit='contain'`` use ``pan=(0, 0)``; zoom then ends at the full-image fit.
    """
    _source(source)
    size, duration, fps = _settings(size, duration, fps, source)
    push = _number(push, "push", 0, 0.5)
    pan = _pair(pan, "pan", -0.25, 0.25)
    if not isinstance(hud, bool):
        raise ValueError("hud must be bool")
    window = tuple(window)
    if len(window) != 4:
        raise ValueError("window must be (left, top, width, height)")
    x, y, width, height = [_number(v, "window", 0, 1) for v in window]
    if min(width, height) <= 0 or x + width > 1 or y + height > 1:
        raise ValueError("window must fit inside the canvas and have positive extent")
    w, h = size
    left, top, vw, vh = round(x * w), round(y * h), round(width * w), round(height * h)
    if min(vw, vh) < 2:
        raise ValueError("window must be at least two pixels in each direction")
    for value, label in (
        (source_label, "source_label"),
        (watermark, "watermark"),
        (iso, "iso"),
    ):
        if not isinstance(value, str) or "\n" in value:
            raise ValueError(f"{label} must be a single-line string")
    font = font_path(
        font, cjk=any(ord(c) > 127 for c in source_label + watermark + iso)
    )
    comp = Composition(
        size=size, duration=duration, fps=fps, name="Archival film frame"
    )
    _still(comp, textured_background(size, seed, dark=True), "film.shell")
    viewport = _viewport(source, (vw, vh), duration, fps, fit=fit, push=push, pan=pan)
    s = min(w / 1920, h / 1080)
    cut = _number(cut, "cut", 0, 1000)
    radius = _number(radius, "radius", 0, 1000)
    comp.add_clip(
        viewport,
        "film.viewport",
        out_point=duration,
        transform=Transform(anchor_point=(0, 0), position=(left, top)),
        masks=film_window_masks(
            (vw, vh), cut=cut * s, radius=radius * s, notches=notches
        ),
    )
    if hud:
        _still(comp, frame_hud(size, iso, font), "film.hud")
    if source_label:
        art = text_image(source_label, 30 * s, font=font, max_width=vw * 0.76)
        _still(
            comp, art, "film.source", (left + vw - art.shape[1] - 25 * s, top + 26 * s)
        )
    if watermark:
        art = text_image(watermark, 74 * s, font=font, max_width=vw * 0.25)
        _still(
            comp,
            art,
            "film.watermark",
            (left + vw - art.shape[1] - 34 * s, top + vh - art.shape[0] - 65 * s),
        )
    comp.documentary_metadata = {
        "template": "film_frame",
        "window_px": (left, top, vw, vh),
        "motion_inferred": True,
        "input_ownership": "borrowed",
    }
    return comp


def add_documentary_subtitles(
    comp, primary, *, secondary=None, font, secondary_font=None
):
    """Add the existing bilingual SubtitleLayer, with the reference's plain style.

    ``primary`` / ``secondary`` are sequences of the existing template ``Cue``.
    The subtitle remains a separate top AE layer; timings stay half-open.
    """
    if not isinstance(comp, Composition):
        raise TypeError("comp must be an AE Composition")
    style = SubtitleStyle(
        primary_size=46,
        secondary_size=32,
        outline_width=2,
        margin_bottom=18,
        line_gap=7,
        max_lines=1,
        max_width=0.76,
        plate_opacity=0,
    )
    preset = ChannelPreset(
        name="documentary_reference",
        size=comp.size,
        fps=comp.fps,
        safe_margin=(0, 0),
        fonts={
            "subtitle": str(font),
            "subtitle_secondary": str(secondary_font or font),
        },
        subtitles=style,
    )
    comp.add_layer(
        subtitle_layer(
            primary,
            preset,
            secondary=secondary,
            font=font,
            secondary_font=secondary_font or font,
            size=comp.size,
            name="documentary.subtitles",
        )
    )
    return comp
