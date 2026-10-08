"""Top-left chapter tag: torn paper strip, vermilion numeral seal, item carousel.

Port of ``nlh_motion.py chapter`` onto AE primitives. The strip slides in from
the left with an eased position Property, the seal is a parented layer whose
scale and opacity are keyframed, and the text items rotate vertically every
``period`` seconds while the strip width eases between item widths. Strip
states (item pair plus quantized transition progress) are rendered once and
cached, so the long static stretches cost only a dictionary lookup per frame.

Examples
--------
>>> from moviepy.ae.templates.chapter_tag import ChapterTag
>>> ChapterTag.__name__
'ChapterTag'
"""

import json
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from moviepy.ae import Composition
from moviepy.ae.layers.av import AVLayer
from moviepy.ae.properties import Keyframe, Property
from moviepy.ae.properties.easing import Ease
from moviepy.ae.templates.paper import (
    alpha_over,
    cached_rgba_clip,
    chinese_numeral,
    drop_shadow,
    ease_out_cubic,
    ink_bleed,
    load_font,
    paper_texture,
    rgba_still,
    seal_stamp,
    torn_mask,
)
from moviepy.ae.templates.presets import ChannelPreset, get_preset
from moviepy.ae.transform import Transform


__all__ = ["ChapterTag", "chapter_tag"]

_EASE_OUT = Ease.bezier(0.16, 1.0, 0.3, 1.0)
_EASE_IN = Ease.bezier(0.7, 0.0, 0.84, 0.0)
_EASE_CUBIC = Ease.bezier(0.33, 1.0, 0.68, 1.0)
_STEPS = 24
_POSITIONS = ("top_left", "bottom_left")


@dataclass
class ChapterTag:
    """A chapter-tag composition, its editable layers and overlay metadata.

    Attributes
    ----------
    composition : Composition
        Transparent composition the size of the tag; render it as an overlay.
    strip, seal : AVLayer
        The paper strip (with text) and the parented seal; their ``transform``
        Properties can be re-keyframed.
    meta : dict
        Sidecar data like NLH's ``.json``: ``x``/``y`` overlay position on the
        full frame, ``w``/``h`` tag size, ``items``, ``period``, ``duration``.
    """

    composition: Composition
    strip: AVLayer
    seal: AVLayer
    meta: dict = field(default_factory=dict)

    def to_json(self, path=None):
        """Return the metadata as JSON text, also writing ``path`` if given."""
        text = json.dumps(self.meta, ensure_ascii=False, indent=2) + "\n"
        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text


def _faded(image, factor):
    out = image.copy()
    out[..., 3] = (out[..., 3] * max(0.0, min(1.0, factor))).astype(np.uint8)
    return out


def chapter_tag(
    items,
    number,
    duration,
    preset="nightlamp_history",
    *,
    period=5.5,
    font=None,
    position="top_left",
    seal=None,
    seed=7,
    size=44,
):
    """Build the animated chapter tag.

    Parameters
    ----------
    items : sequence of str or str
        Text items rotated vertically; a ``"a|b|c"`` string is split on ``|``.
    number : int
        Chapter number, stamped as a formal numeral (壹, 貳, ...).
    duration : float
        Composition length in seconds.
    preset : ChannelPreset or str, optional
        Palette roles ``paper``, ``ink``, ``seal`` and the ``title`` font.
    period : float, optional
        Seconds each item stays (NLH channel value 5.5).
    font : str, optional
        Font file overriding ``preset.font("title")``.
    position : {"top_left", "bottom_left"}, optional
        Where the overlay belongs on the full frame (written to ``meta``).
    seal : str, optional
        Seal text overriding the numeral (1-4 characters).
    seed : int, optional
        Seeds paper, tearing and seal wear.
    size : float, optional
        Text size in pixels at 1080p; scaled to the preset height.

    Returns
    -------
    ChapterTag
        ``.composition`` is transparent outside the strip.

    Examples
    --------
    >>> chapter_tag.__name__
    'chapter_tag'
    """
    if isinstance(preset, str):
        preset = get_preset(preset)
    if not isinstance(preset, ChannelPreset):
        raise TypeError("preset must be a ChannelPreset or preset name")
    if isinstance(items, str):
        items = items.replace("\\n", "\n").split("|")
    items = [str(s).strip() for s in items if str(s).strip()]
    if not items:
        raise ValueError("items must contain at least one non-empty string")
    if position not in _POSITIONS:
        raise ValueError(f"position must be one of {_POSITIONS}")
    duration = float(duration)
    period = float(period)
    if duration <= 0 or period <= 0:
        raise ValueError("duration and period must be positive")
    fpath = preset.font("title", font)
    S = preset.size[1] / 1080.0
    paper = preset.color("paper", (239, 228, 204))
    ink = preset.color("ink", (30, 26, 22))
    seal_rgb = preset.color("seal", (176, 30, 28))

    fs = int(size * S)
    fmain = load_font(fpath, fs)
    seal_sz = int(fs * 1.75)
    padl, padr, padv = int(fs * 0.5), int(fs * 0.75), int(fs * 0.42)
    strip_h = seal_sz + 2 * padv
    widths = [
        int(padl + seal_sz + fs * 0.55 + fmain.getlength(s) + padr) for s in items
    ]
    maxw = max(widths)
    m = int(28 * S)
    RW, RH = maxw + 2 * m + int(40 * S), strip_h + 2 * m
    tin = min(0.7, duration * 0.4)
    tout = min(0.45, duration * 0.3)
    hold_at = max(tin, duration - tout)
    sw = min(0.5, period * 0.5)
    text_x = m + padl + seal_sz + fs * 0.55

    mask_full = torn_mask(maxw, strip_h, max(1, int(3 * S)), seed, "tbr")
    tex_full = paper_texture(maxw, strip_h, paper, seed)
    text_imgs = []
    for s in items:
        img = Image.new("RGBA", (int(fmain.getlength(s)) + 8, int(fs * 1.4)), (*ink, 0))
        ImageDraw.Draw(img).text(
            (2, fs * 0.7), s, font=fmain, fill=(*ink, 255), anchor="lm"
        )
        text_imgs.append(ink_bleed(np.asarray(img), 0.4 * S))
    accent = np.zeros((strip_h - 5, max(2, int(4 * S)), 4), np.uint8)
    accent[...] = (*seal_rgb, 230)

    def phase(t):
        k = int(t // period)
        u = t - k * period
        cur, prev = k % len(items), (k - 1) % len(items)
        p = u / sw if (k > 0 and u < sw) else 1.0
        return k, cur, prev, min(p, 1.0)

    def key_fn(t):
        k, cur, prev, p = phase(t)
        if p < 1.0:
            return ("move", cur, prev, int(p * _STEPS))
        ta = ease_out_cubic((t - 0.45) / 0.4) if k == 0 and t < 0.9 else 1.0
        return ("rest", cur, int(round(ta * _STEPS)))

    def build(key):
        if key[0] == "move":
            _, cur, prev, q = key
            e = ease_out_cubic(q / _STEPS)
            w = int(round(widths[prev] + (widths[cur] - widths[prev]) * e))
        else:
            _, cur, qa = key
            prev, e, w = None, 1.0, widths[cur]
        w = max(1, min(w, maxw))
        alpha = mask_full[:, maxw - w :]
        out = drop_shadow(alpha, (RW, RH), (m, m + int(6 * S)), 8 * S, 0.38)
        strip = np.empty((strip_h, w, 4), np.uint8)
        strip[..., :3] = tex_full[:, maxw - w :]
        strip[..., 3] = np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8)
        alpha_over(out, strip, m, m)
        alpha_over(out, accent, m, m + 2)
        text = np.zeros((RH, RW, 4), np.uint8)
        cy = m + strip_h / 2
        if key[0] == "move":
            a, b = text_imgs[prev], text_imgs[cur]
            alpha_over(
                text, _faded(a, 1 - e), text_x, cy - a.shape[0] / 2 - e * fs * 0.6
            )
            alpha_over(
                text, _faded(b, e), text_x, cy - b.shape[0] / 2 + (1 - e) * fs * 0.6
            )
        elif qa > 0:
            ta = qa / _STEPS
            b = text_imgs[cur]
            alpha_over(
                text, _faded(b, ta), text_x, cy - b.shape[0] / 2 + (1 - ta) * fs * 0.25
            )
        clip_rect = np.zeros((RH, RW), np.float32)
        clip_rect[m + 3 : m + strip_h - 3, m : m + max(0, w - 6)] = 1.0
        text[..., 3] = (text[..., 3] * clip_rect).astype(np.uint8)
        alpha_over(out, text, 0, 0)
        return out[..., :3].copy(), out[..., 3] / 255.0

    clip = cached_rgba_clip((RW, RH), duration, key_fn, build, max_cache=256)

    strip_layer = AVLayer(
        clip,
        "chapter strip",
        transform=Transform(
            anchor_point=(0.0, 0.0),
            position=Property(
                (0.0, 0.0),
                value_type="vec2",
                keyframes=[
                    Keyframe(
                        0.0, (-float(RW), 0.0), interp="bezier", out_ease=_EASE_OUT
                    ),
                    Keyframe(tin, (0.0, 0.0), interp="bezier", out_ease=_EASE_IN),
                    Keyframe(hold_at, (0.0, 0.0), interp="bezier", out_ease=_EASE_IN),
                    Keyframe(duration, (-RW * 0.6, 0.0)),
                ],
            ),
            opacity=Property(
                100.0,
                keyframes=[
                    Keyframe(0.0, 0.0),
                    Keyframe(tin / 2, 100.0),
                    Keyframe(hold_at, 100.0),
                    Keyframe(duration, 0.0),
                ],
            ),
        ),
    )

    stamp = seal_stamp(seal or chinese_numeral(number), seal_sz, seal_rgb, seed, fpath)
    s0 = min(0.35, duration * 0.2)
    s1 = min(s0 + 0.3, hold_at)
    seal_layer = AVLayer(
        rgba_still(stamp, duration),
        "chapter seal",
        parent=strip_layer,
        transform=Transform(
            anchor_point=(seal_sz / 2, seal_sz / 2),
            position=(m + padl + seal_sz / 2, RH / 2),
            rotation=-4.0,
            scale=Property(
                (145.0, 145.0),
                value_type="vec2",
                keyframes=[
                    Keyframe(0.0, (145.0, 145.0)),
                    Keyframe(s0, (145.0, 145.0), interp="bezier", out_ease=_EASE_CUBIC),
                    Keyframe(s1, (100.0, 100.0)),
                ],
            ),
            opacity=Property(
                100.0,
                keyframes=[
                    Keyframe(0.0, 0.0),
                    Keyframe(s0, 0.0, interp="bezier", out_ease=_EASE_CUBIC),
                    Keyframe(s1, 100.0),
                    Keyframe(hold_at, 100.0),
                    Keyframe(duration, 0.0),
                ],
            ),
        ),
    )

    comp = Composition(
        size=(RW, RH),
        fps=preset.fps,
        duration=duration,
        name="chapter_tag",
        transparent=True,
    )
    comp.add_layer(strip_layer)
    comp.add_layer(seal_layer)
    y = int(22 * S)
    if position == "bottom_left":
        y = preset.size[1] - RH - int(22 * S)
    meta = {
        "kind": "chapter",
        "x": int(26 * S),
        "y": y,
        "w": RW,
        "h": RH,
        "items": items,
        "period": period,
        "duration": duration,
        "position": position,
        "number": number,
        "fps": preset.fps,
    }
    return ChapterTag(comp, strip_layer, seal_layer, meta)
