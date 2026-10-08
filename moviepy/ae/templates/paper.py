"""Deterministic paper, torn-edge, seal and shadow helpers for NLH templates.

Everything here is seeded numpy/Pillow work that returns plain arrays, so the
chapter tag and the quote scroll can bake it into cached layer states. Images
are straight-alpha ``uint8`` RGBA unless noted; masks are float32 in 0..1.

Examples
--------
>>> from moviepy.ae.templates.paper import chinese_numeral, paper_texture
>>> [chinese_numeral(n) for n in (1, 10, 11, 21)]
['壹', '拾', '拾壹', '貳拾壹']
>>> a = paper_texture(16, 8, (239, 228, 204), seed=1)
>>> bool((a == paper_texture(16, 8, (239, 228, 204), seed=1)).all())
True
"""

import math
from collections import OrderedDict
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.av import AVLayer, _clamped_source_time
from moviepy.video.VideoClip import ImageClip, VideoClip


__all__ = [
    "CachedAVLayer",
    "alpha_over",
    "cached_rgba_clip",
    "chinese_numeral",
    "drop_shadow",
    "ease_in_out_sine",
    "ease_out_back",
    "ease_out_cubic",
    "has_glyph",
    "ink_bleed",
    "load_font",
    "mix_color",
    "paper_texture",
    "rect_shadow",
    "resize_straight",
    "rgba_still",
    "scale_alpha",
    "seal_stamp",
    "torn_mask",
]

_DIGITS = "零壹貳參肆伍陸柒捌玖"


def clamp(x, low=0.0, high=1.0):
    """Clamp ``x`` into ``[low, high]``."""
    return max(low, min(high, x))


def ease_out_cubic(x):
    """Return cubic ease-out of ``x`` clamped to 0..1."""
    return 1 - (1 - clamp(x)) ** 3


def ease_in_out_sine(x):
    """Return sinusoidal ease-in-out of ``x`` clamped to 0..1."""
    return 0.5 - 0.5 * math.cos(math.pi * clamp(x))


def ease_out_back(x, s=1.4):
    """Return back ease-out (small overshoot) of ``x`` clamped to 0..1."""
    x = clamp(x) - 1
    return 1 + (s + 1) * x**3 + s * x**2


def mix_color(a, b, t):
    """Return the RGB blend ``a * (1 - t) + b * t`` as ints."""
    return tuple(int(round(x * (1 - t) + y * t)) for x, y in zip(a, b))


def chinese_numeral(n):
    """Return the formal (banknote) numeral used on seals: 壹, 貳, 拾壹, 貳拾.

    Parameters
    ----------
    n : int
        Positive number below 100; larger numbers fall back to digits.

    Examples
    --------
    >>> chinese_numeral(3), chinese_numeral(20), chinese_numeral(105)
    ('參', '貳拾', '105')
    """
    n = int(n)
    if n <= 0 or n >= 100:
        return str(n)
    if n < 10:
        return _DIGITS[n]
    if n == 10:
        return "拾"
    if n < 20:
        return "拾" + _DIGITS[n % 10]
    return _DIGITS[n // 10] + "拾" + (_DIGITS[n % 10] if n % 10 else "")


@lru_cache(maxsize=32)
def load_font(path, size):
    """Return a cached Pillow font; ``path=None`` uses Pillow's default font."""
    size = max(1, int(size))
    if path is None:
        return ImageFont.load_default(size)
    return ImageFont.truetype(str(path), size)


_CMAPS = {}


def has_glyph(path, char):
    """Report whether the font at ``path`` maps ``char``.

    Uses fontTools when present; without it every glyph is assumed present.
    """
    if not path:
        return False
    if path not in _CMAPS:
        try:
            from fontTools.ttLib import TTCollection, TTFont

            font = (
                TTCollection(path).fonts[0]
                if str(path).lower().endswith(".ttc")
                else TTFont(path)
            )
            _CMAPS[path] = set(font.getBestCmap())
        except Exception:
            _CMAPS[path] = None
    cmap = _CMAPS[path]
    return True if cmap is None else ord(char) in cmap


@lru_cache(maxsize=16)
def _paper_texture(w, h, base, seed, strength):
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 1, (h, w)).astype(np.float32) * 2.4 * strength
    low = rng.normal(0, 1, (h // 90 + 2, w // 90 + 2)).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC) * 2.2 * strength
    fibers = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(fibers)
    count = int(w * h / 7000)
    xs, ys = rng.uniform(0, w, count), rng.uniform(0, h, count)
    lengths, angles = rng.uniform(8, 34, count), rng.uniform(0, math.pi, count)
    shades = rng.integers(20, 71, count)
    for x, y, ln, an, sh in zip(xs, ys, lengths, angles, shades):
        draw.line([(x, y), (x + ln * math.cos(an), y + ln * math.sin(an))], int(sh))
    fib = np.asarray(fibers.filter(ImageFilter.GaussianBlur(0.6)), np.float32)
    fib = fib / 70 * 4 * strength
    arr = np.asarray(base, np.float32)[None, None, :] + (grain + low - fib)[..., None]
    out = np.clip(arr, 0, 255).astype(np.uint8)
    out.flags.writeable = False
    return out


def paper_texture(w, h, base, seed=7, strength=1.0):
    """Return a deterministic rice-paper texture (grain, blotches, fibers).

    Parameters
    ----------
    w, h : int
        Size in pixels.
    base : sequence of int
        Base RGB color.
    seed : int, optional
        Same seed and arguments give identical pixels.
    strength : float, optional
        Noise amplitude multiplier.

    Returns
    -------
    numpy.ndarray
        Read-only ``uint8`` array of shape ``(h, w, 3)`` (cached; copy to edit).

    Examples
    --------
    >>> paper_texture(8, 4, (200, 200, 200), seed=3).shape
    (4, 8, 3)
    """
    return _paper_texture(int(w), int(h), tuple(int(v) for v in base), seed, strength)


def torn_mask(w, h, amp, seed=0, sides="tblr", ss=2):
    """Return a float32 0..1 mask whose chosen sides have a torn, wavy edge.

    Parameters
    ----------
    w, h : int
        Mask size.
    amp : float
        Maximum jitter in pixels.
    seed : int, optional
        Seed for the jitter.
    sides : str, optional
        Any of ``t``, ``b``, ``l``, ``r``; others stay straight.
    ss : int, optional
        Supersampling factor for the anti-aliased polygon.

    Examples
    --------
    >>> m = torn_mask(20, 10, 2, seed=1)
    >>> m.shape, bool(m.max() <= 1.0)
    ((10, 20), True)
    """
    rng = np.random.default_rng(seed)
    W, H, A = w * ss, h * ss, amp * ss
    step = 11 * ss

    def jit(side, n):
        return rng.uniform(-A, A, n) if side in sides else np.zeros(n)

    xs = np.arange(0, W, step)
    ys = np.arange(0, H, step)
    pts = []
    pts += list(zip(xs, A + jit("t", len(xs))))
    pts += list(zip(W - A + jit("r", len(ys)), ys))
    xr = np.arange(W, 0, -step)
    pts += list(zip(xr, H - A + jit("b", len(xr))))
    yr = np.arange(H, 0, -step)
    pts += list(zip(A + jit("l", len(yr)), yr))
    pts = [(min(max(px, 0), W), min(max(py, 0), H)) for px, py in pts]
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    if ss != 1:
        mask = mask.resize((w, h), Image.LANCZOS)
    return np.asarray(mask, np.float32) / 255.0


def drop_shadow(alpha, size, offset, blur, opacity, color=(20, 14, 10)):
    """Return a blurred shadow of ``alpha`` as straight RGBA ``uint8``.

    Parameters
    ----------
    alpha : numpy.ndarray
        Float 0..1 mask of the casting shape.
    size : tuple of int
        Output ``(width, height)``.
    offset : tuple of int
        Where the mask's top-left lands, shadow offset included.
    blur : float
        Gaussian sigma in pixels.
    opacity : float
        Peak shadow opacity.
    color : sequence of int, optional
        Shadow RGB.

    Examples
    --------
    >>> sh = drop_shadow(np.ones((4, 4), np.float32), (12, 12), (4, 6), 1.0, 0.5)
    >>> sh.shape, int(sh[..., 3].max()) <= 128
    ((12, 12, 4), True)
    """
    width, height = size
    canvas = np.zeros((height, width), np.float32)
    ox, oy = int(offset[0]), int(offset[1])
    ah, aw = alpha.shape
    x0, y0 = max(ox, 0), max(oy, 0)
    x1, y1 = min(ox + aw, width), min(oy + ah, height)
    if x0 < x1 and y0 < y1:
        canvas[y0:y1, x0:x1] = alpha[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    if blur > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), float(blur))
    out = np.empty((height, width, 4), np.uint8)
    out[..., :3] = np.asarray(color, np.uint8)
    out[..., 3] = np.clip(canvas * opacity * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def rect_shadow(size, rect, blur, opacity):
    """Return the float32 0..1 alpha of a blurred rectangle's shadow.

    A Gaussian is separable, so the blur of a rectangle is the outer product of
    two blurred 1-D profiles: the cost is ``O(w + h)`` for the blur instead of a
    full-canvas filter. Equals ``drop_shadow`` of a solid rectangle.

    Parameters
    ----------
    size : tuple of int
        Canvas ``(width, height)``.
    rect : tuple of int
        ``(x, y, w, h)`` of the casting rectangle on the canvas.
    blur, opacity : float
        Gaussian sigma in pixels and peak opacity.

    Examples
    --------
    >>> a = rect_shadow((12, 12), (4, 4, 4, 4), 1.0, 0.5)
    >>> a.shape, bool(0.4 < a[6, 6] <= 0.5)
    ((12, 12), True)
    """
    width, height = size
    x, y, w, h = (int(v) for v in rect)

    def profile(n, start, length):
        line = np.zeros((1, n), np.float32)
        line[0, max(start, 0) : max(0, min(start + length, n))] = 1.0
        if blur > 0:
            line = cv2.GaussianBlur(line, (0, 0), float(blur))
        return line[0]

    gx, gy = profile(width, x, w), profile(height, y, h)
    return np.outer(gy, gx).astype(np.float32) * np.float32(opacity)


def alpha_over(dst, src, x, y):
    """Composite straight-alpha RGBA ``src`` over ``dst`` in place at ``(x, y)``.

    Both arrays are ``uint8`` ``(h, w, 4)``; ``src`` is cropped to ``dst``.

    Examples
    --------
    >>> dst = np.zeros((2, 2, 4), np.uint8)
    >>> src = np.full((1, 1, 4), 255, np.uint8)
    >>> alpha_over(dst, src, 1, 1)[1, 1].tolist()
    [255, 255, 255, 255]
    """
    x, y = int(round(x)), int(round(y))
    sh, sw = src.shape[:2]
    dh, dw = dst.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, dw), min(y + sh, dh)
    if x0 >= x1 or y0 >= y1:
        return dst
    part = src[y0 - y : y1 - y, x0 - x : x1 - x]
    if part[..., 3].min() == 255:  # opaque source: a plain copy, no blending
        dst[y0:y1, x0:x1] = part
        return dst
    s = part.astype(np.float32)
    d = dst[y0:y1, x0:x1].astype(np.float32)
    sa = s[..., 3:4] / 255.0
    da = d[..., 3:4] / 255.0
    oa = sa + da * (1 - sa)
    safe = np.where(oa > 0, oa, 1.0)
    rgb = (s[..., :3] * sa + d[..., :3] * da * (1 - sa)) / safe
    dst[y0:y1, x0:x1, :3] = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    dst[y0:y1, x0:x1, 3] = np.clip(oa[..., 0] * 255 + 0.5, 0, 255).astype(np.uint8)
    return dst


def scale_alpha(image, factor):
    """Return a copy of an RGBA ``uint8`` image with alpha times ``factor``."""
    if factor >= 0.999:
        return image
    out = image.copy()
    out[..., 3] = np.clip(out[..., 3] * factor + 0.5, 0, 255).astype(np.uint8)
    return out


def ink_bleed(image, amount=0.6, seed=11):
    """Soften RGBA ink: slight bleed plus uneven density (deterministic)."""
    a = image.astype(np.float32)
    h, w = a.shape[:2]
    alpha = a[..., 3]
    blurred = cv2.GaussianBlur(alpha, (0, 0), max(0.1, float(amount)))
    alpha = np.maximum(alpha * 0.92, blurred)
    rng = np.random.default_rng(seed)
    low = rng.normal(0, 1, (max(2, h // 40), max(2, w // 40))).astype(np.float32)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    a[..., 3] = np.clip(alpha * (0.9 + 0.08 * low), 0, 255)
    return a.astype(np.uint8)


def resize_straight(image, size):
    """Area-resize a straight-alpha RGBA ``uint8`` image without dark fringes.

    ``cv2.resize`` on straight RGBA averages the colour of transparent pixels
    into the edge. This averages premultiplied colour instead and divides the
    alpha back out, so a coloured shape keeps its colour at its soft edge.

    Parameters
    ----------
    image : numpy.ndarray
        Straight RGBA ``uint8`` ``(h, w, 4)``.
    size : tuple of int
        Output ``(width, height)``.

    Examples
    --------
    >>> img = np.zeros((4, 4, 4), np.uint8)
    >>> img[:, :2] = (200, 100, 50, 255)
    >>> out = resize_straight(img, (1, 1))
    >>> out[0, 0].tolist()
    [200, 100, 50, 128]
    """
    a = image.astype(np.float32)
    alpha = a[..., 3:4] / 255.0
    a[..., :3] *= alpha
    small = cv2.resize(a, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
    alpha = small[..., 3:4] / 255.0
    safe = np.where(alpha > 1e-6, alpha, 1.0)
    out = np.empty(small.shape, np.uint8)
    out[..., :3] = np.clip(small[..., :3] / safe + 0.5, 0, 255)
    out[..., 3] = np.clip(small[..., 3] + 0.5, 0, 255)
    return out


def seal_stamp(text, size, color=(176, 30, 28), seed=3, font=None):
    """Draw a vermilion relief seal (paper-coloured glyphs on red).

    Parameters
    ----------
    text : str
        One to four characters: 1 centered, 2 stacked, 3-4 in two columns
        read right to left (Chinese numerals such as 壹貳參 work well).
    size : int
        Output edge in pixels.
    color : sequence of int, optional
        Seal RGB.
    seed : int, optional
        Seeds the worn-ink pattern and the rough edge.
    font : str, optional
        Font file path; ``None`` uses Pillow's default font.

    Returns
    -------
    numpy.ndarray
        Straight RGBA ``uint8`` ``(size, size, 4)``.

    Examples
    --------
    >>> s = seal_stamp("壹", 24, seed=1)
    >>> s.shape, bool(s[..., 3].max() > 0)
    ((24, 24, 4), True)
    >>> bool((s == seal_stamp("壹", 24, seed=1)).all())
    True
    """
    ss = 3
    size = int(size)
    S = size * ss
    rng = np.random.default_rng(seed)
    base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    pad = int(S * 0.05)
    ImageDraw.Draw(base).rounded_rectangle(
        [pad, pad, S - pad, S - pad], radius=int(S * 0.08), fill=(*color, 255)
    )
    chars = list(text)[:4]
    if len(chars) == 1:
        cells = [(S / 2, S / 2, S * 0.62)]
    elif len(chars) == 2:
        cells = [(S / 2, S * 0.31, S * 0.38), (S / 2, S * 0.69, S * 0.38)]
    else:
        cells = [
            (S * 0.69, S * 0.31, S * 0.36),
            (S * 0.69, S * 0.69, S * 0.36),
            (S * 0.31, S * 0.31, S * 0.36),
            (S * 0.31, S * 0.69, S * 0.36),
        ]
    glyphs = Image.new("L", (S, S), 0)
    draw = ImageDraw.Draw(glyphs)
    for ch, (cx, cy, fs) in zip(chars, cells):
        draw.text(
            (cx, cy),
            ch,
            font=load_font(font, int(fs)),
            fill=255,
            anchor="mm",
            stroke_width=int(fs * 0.025),
            stroke_fill=255,
        )
    a = np.asarray(base, np.float32).copy()
    t = np.asarray(glyphs, np.float32)[..., None] / 255
    a[..., :3] = a[..., :3] * (1 - t) + np.array((246, 236, 220), np.float32) * t
    low = rng.normal(0, 1, (12, 12)).astype(np.float32)
    low = cv2.resize(low, (S, S), interpolation=cv2.INTER_CUBIC)
    hi = rng.normal(0, 1, (S, S)).astype(np.float32)
    wear = np.clip(1.0 - np.maximum(0, low * 0.35 + hi * 0.25 - 0.55), 0.15, 1)
    a[..., 3] *= wear
    a[..., 3] *= torn_mask(S, S, int(S * 0.012) + 1, seed, "tblr", ss=1)
    big = np.clip(a, 0, 255).astype(np.uint8)
    return resize_straight(big, (size, size))


def cached_rgba_clip(
    size, duration, key_fn, build_fn, *, max_cache=128, max_bytes=192 * 2**20
):
    """Return a ``VideoClip`` (with mask) whose frames are cached by state.

    ``key_fn(t)`` maps time to a hashable state; ``build_fn(key)`` renders it
    once as ``(rgb uint8 (h, w, 3), alpha float (h, w))``. Long static stretches
    therefore cost one dictionary lookup per frame.

    Parameters
    ----------
    size : tuple of int
        Frame ``(width, height)``; every built state must match.
    duration : float
        Clip length in seconds.
    key_fn, build_fn : callable
        State selector and renderer.
    max_cache : int, optional
        Least-recently-used limit on cached states.
    max_bytes : int, optional
        Least-recently-used limit on the memory of cached states; the newest
        state always stays.

    Returns
    -------
    VideoClip
        With ``mask`` set; ``clip.builds`` counts renders, ``clip.states`` is
        the cache and ``clip.state_key`` is ``key_fn`` (``CachedAVLayer`` uses
        it to import each state into the renderer once).

    Examples
    --------
    >>> clip = cached_rgba_clip(
    ...     (2, 2), 1.0, lambda t: 0,
    ...     lambda k: (np.zeros((2, 2, 3), np.uint8), np.ones((2, 2))))
    >>> _ = clip.get_frame(0.1), clip.get_frame(0.9)
    >>> clip.builds
    1
    """
    states = OrderedDict()
    width, height = size
    used = [0]

    def state(t):
        key = key_fn(t)
        hit = states.get(key)
        if hit is None:
            rgb, alpha = build_fn(key)
            if rgb.shape[:2] != (height, width):
                raise ValueError("built state does not match the clip size")
            hit = (rgb, np.asarray(alpha, np.float32))
            states[key] = hit
            used[0] += hit[0].nbytes + hit[1].nbytes
            clip.builds += 1
            while len(states) > 1 and (len(states) > max_cache or used[0] > max_bytes):
                _, old = states.popitem(last=False)
                used[0] -= old[0].nbytes + old[1].nbytes
        else:
            states.move_to_end(key)
        return hit

    clip = VideoClip.__new__(VideoClip)
    clip.builds = 0
    VideoClip.__init__(clip, lambda t: state(t)[0], duration=duration)
    clip.mask = VideoClip(lambda t: state(t)[1], is_mask=True, duration=duration)
    clip.states = states
    clip.state_key = key_fn
    return clip


def rgba_still(image, duration):
    """Wrap a straight RGBA ``uint8`` image as a constant clip with a mask.

    Examples
    --------
    >>> clip = rgba_still(np.full((2, 3, 4), 255, np.uint8), 1.0)
    >>> clip.size, clip.mask.get_frame(0).shape
    ((3, 2), (2, 3))
    """
    clip = ImageClip(np.ascontiguousarray(image[..., :3]), duration=duration)
    clip.mask = ImageClip(
        image[..., 3].astype(np.float32) / 255.0, is_mask=True, duration=duration
    )
    return clip


class CachedAVLayer(AVLayer):
    """``AVLayer`` that imports each distinct source state into a Buffer once.

    ``Buffer.from_clip`` converts the clip frame to premultiplied float32 on
    every render, which dominates the cost of large still sources. A still
    (``ImageClip``) or a ``cached_rgba_clip`` maps many times to one state, so
    this layer keeps the imported ``Buffer`` (immutable) in a small
    least-recently-used cache keyed by that state. Any other clip falls back to
    the plain import, so behaviour never changes, only the speed.

    Parameters
    ----------
    max_bytes : int, optional
        Memory limit for the cached Buffers of this layer.

    Examples
    --------
    >>> clip = rgba_still(np.full((2, 2, 4), 255, np.uint8), 1.0)
    >>> layer = CachedAVLayer(clip, "still")
    >>> layer.source_buffer(0.0) is layer.source_buffer(0.5)
    True
    """

    _STATIC = object()

    def __init__(self, *args, max_bytes=160 * 2**20, **kwargs):
        super().__init__(*args, **kwargs)
        self._buffers = OrderedDict()
        self._buffer_bytes = 0
        self._buffer_limit = int(max_bytes)

    def _state(self, time):
        key_fn = getattr(self.clip, "state_key", None)
        if key_fn is not None:
            return key_fn(time)
        if isinstance(self.clip, ImageClip):
            return self._STATIC
        return None

    def source_buffer(self, t, context=None):
        """Return the cached Buffer for the clip state at source time ``t``."""
        time = _clamped_source_time(self.clip, t)
        key = self._state(time)
        if key is None:
            return Buffer.from_clip(self.clip, time)
        hit = self._buffers.get(key)
        if hit is not None:
            self._buffers.move_to_end(key)
            return hit
        hit = Buffer.from_clip(self.clip, time)
        self._buffers[key] = hit
        self._buffer_bytes += hit.rgba.nbytes
        while len(self._buffers) > 1 and self._buffer_bytes > self._buffer_limit:
            _, old = self._buffers.popitem(last=False)
            self._buffer_bytes -= old.rgba.nbytes
        return hit
