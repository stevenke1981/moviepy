r"""Ambience for long relaxation videos: particles, light arc, sleep fade.

Three cheap ways to keep a 3-10 hour picture alive without rendering every
frame through After Effects (see ``music_render`` for the PERFORMANCE RULE):

``particle_loop``
    A short, *exactly periodic*, transparent particle composition (fireflies,
    dust, petals, leaves, snow, snowflakes, rain). ``export_overlay_loop``
    writes it once as a MOV with alpha; FFmpeg repeats it with ``-stream_loop -1`` and overlays it
    on the looped background (``render_music_video(overlay_loops=...)``).
``LightArc``
    A slow brightness / saturation / warmth / contrast curve over hours,
    compiled to one FFmpeg ``eq=...:eval=frame`` expression of the time ``t``.
    No Python runs per frame.
``SleepFade``
    A slow fade of the picture (and optionally the music) towards a floor,
    reaching it at ``end`` and holding it afterwards.

All three follow the channel's low-stimulation rules: nothing flashes, light
changes are cosine-eased and limited to ``0.1`` of the range per second
(``max_rate``), and particles drift slowly.

Examples
--------
>>> from moviepy.ae.templates.ambience import LightArc, SleepFade
>>> arc = LightArc([(0, {"brightness": 0.0}), (600, {"brightness": -0.2})])
>>> round(arc.value_at(300)["brightness"], 3)
-0.1
>>> SleepFade(60, 120).gain_at(120)
0.0
"""

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.composition import Composition
from moviepy.ae.layers.base import Layer
from moviepy.ae.parallel import iter_frames_parallel
from moviepy.ae.templates.music_render import (
    _NO_WINDOW,
    _PackedRGBA,
    _rgba_frame,
    _unpack,
)
from moviepy.config import FFMPEG_BINARY


__all__ = [
    "PARTICLE_KINDS",
    "ParticleLayer",
    "particle_loop",
    "export_overlay_loop",
    "LightArc",
    "SleepFade",
]

_ANGLES = 32  # orientation steps over 180 degrees
_SUBPIXEL = 8  # sprite sub-pixel steps per axis
_SUPERSAMPLE = 3

# Per-kind defaults. ``count`` is for a 1280x720 region and scales with area;
# sizes are for 720 px of height and scale with the region. ``wraps`` is the
# number of times a falling particle crosses the region per period at
# ``speed=1`` (its depth factor multiplies it); ``sway`` is the horizontal
# wander amplitude in px; ``twinkle`` the opacity modulation depth.
_KINDS = {
    "fireflies": dict(
        count=28, sizes=(2.6, 3.6, 4.8), shape=(1.0, 1.0), power=1.0,
        color=(255, 236, 150), opacity=0.85, fall=False, wraps=0,
        sway=(25, 80), sway_cycles=(1, 3), twinkle=0.65, rotate=0, slope=0.0,
    ),
    "dust": dict(
        count=60, sizes=(1.1, 1.5, 2.0), shape=(1.0, 1.0), power=1.0,
        color=(255, 244, 225), opacity=0.35, fall=False, wraps=0,
        sway=(15, 55), sway_cycles=(1, 2), twinkle=0.25, rotate=0, slope=0.0,
    ),
    "petals": dict(
        count=14, sizes=(4.0, 5.0, 6.0), shape=(1.0, 0.55), power=2.0,
        color=(255, 196, 212), opacity=0.7, fall=True, wraps=1,
        sway=(30, 90), sway_cycles=(1, 2), twinkle=0.0, rotate=1, slope=0.0,
    ),
    "leaves": dict(
        count=12, sizes=(5.0, 6.2, 7.4), shape=(1.0, 0.45), power=2.0,
        color=(196, 150, 72), opacity=0.65, fall=True, wraps=1,
        sway=(40, 110), sway_cycles=(1, 2), twinkle=0.0, rotate=1, slope=0.0,
    ),
    "snow": dict(
        count=90, sizes=(1.3, 1.9, 2.6), shape=(1.0, 1.0), power=1.5,
        color=(246, 249, 255), opacity=0.7, fall=True, wraps=1,
        sway=(10, 35), sway_cycles=(1, 3), twinkle=0.0, rotate=0, slope=0.0,
    ),
    "snowflakes": dict(
        count=26, sizes=(4.5, 6.5, 9.0), shape=(1.0, 1.0), power=1.0,
        color=(246, 249, 255), opacity=0.75, fall=True, wraps=1,
        sway=(12, 40), sway_cycles=(1, 2), twinkle=0.0, rotate=1, slope=0.0,
        glyph="crystal",
    ),
    "rain": dict(
        count=110, sizes=(16.0, 22.0, 30.0), shape=(1.0, 0.07), power=1.0,
        color=(200, 215, 235), opacity=0.22, fall=True, wraps=6,
        sway=(0, 0), sway_cycles=(1, 1), twinkle=0.0, rotate=0, slope=0.12,
    ),
}  # fmt: skip
PARTICLE_KINDS = tuple(_KINDS)


def _sprite(sigma_major, sigma_minor, power, angle, frac_x, frac_y):
    """Return a soft elliptical sprite centred at a sub-pixel offset."""
    half = int(math.ceil(3.0 * sigma_major)) + 1
    steps = (np.arange(_SUPERSAMPLE) + 0.5) / _SUPERSAMPLE - 0.5
    axis = np.arange(-half, half + 1, dtype=np.float64)
    gy, gx = np.meshgrid(axis, axis, indexing="ij")
    total = np.zeros_like(gx)
    cos, sin = math.cos(angle), math.sin(angle)
    for oy in steps:
        for ox in steps:
            dx = gx + ox - frac_x
            dy = gy + oy - frac_y
            major = dx * cos + dy * sin
            minor = -dx * sin + dy * cos
            rho2 = (major / sigma_major) ** 2 + (minor / sigma_minor) ** 2
            total += np.exp(-0.5 * rho2**power)
    return (total / _SUPERSAMPLE**2).astype(np.float32), half


def _segment_distance(px, py, ax, ay, bx, by):
    """Return the distance from points ``(px, py)`` to segment ``a``-``b``."""
    vx, vy = bx - ax, by - ay
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0.0, 1.0)
    return np.hypot(px - ax - t * vx, py - ay - t * vy)


def _crystal_segments(radius):
    """Return the line segments of a six-armed snowflake of ``radius`` px."""
    segments = []
    for arm in range(6):
        a = math.pi / 3 * arm
        ca, sa = math.cos(a), math.sin(a)
        segments.append((0.0, 0.0, radius * ca, radius * sa))
        for at, length in ((0.5, 0.32), (0.75, 0.2)):
            bx, by = at * radius * ca, at * radius * sa
            for side in (-1, 1):
                b = a + side * math.pi / 3
                tip = length * radius
                segments.append(
                    (bx, by, bx + tip * math.cos(b), by + tip * math.sin(b))
                )
    return segments


def _crystal_sprite(radius, angle, frac_x, frac_y):
    """Return an antialiased six-armed snowflake sprite (peak 1.0)."""
    width = max(0.45, 0.09 * radius)  # arm half-thickness (gaussian sigma)
    half = int(math.ceil(radius + 3.0 * width)) + 1
    steps = (np.arange(_SUPERSAMPLE) + 0.5) / _SUPERSAMPLE - 0.5
    axis = np.arange(-half, half + 1, dtype=np.float64)
    gy, gx = np.meshgrid(axis, axis, indexing="ij")
    cos, sin = math.cos(angle), math.sin(angle)
    segments = _crystal_segments(radius)
    total = np.zeros_like(gx)
    for oy in steps:
        for ox in steps:
            dx = gx + ox - frac_x
            dy = gy + oy - frac_y
            u = dx * cos + dy * sin
            v = -dx * sin + dy * cos
            near = np.full_like(gx, np.inf)
            for ax, ay, bx, by in segments:
                np.minimum(near, _segment_distance(u, v, ax, ay, bx, by), out=near)
            core = np.hypot(u, v) / max(1.0, 0.18 * radius)
            total += np.maximum(
                np.exp(-0.5 * (near / width) ** 2), np.exp(-0.5 * core**2)
            )
    return (total / _SUPERSAMPLE**2).astype(np.float32), half


class ParticleLayer(Layer):
    r"""Draw seeded, exactly periodic particles as a transparent AE layer.

    Positions and opacities are pure functions of the integer frame index
    modulo the loop length, built from integer cycle counts (sine wander,
    sine twinkle, wrapped fall), so frame ``n + frames`` equals frame ``n``
    bit for bit. Sprites are rendered once per (size, angle, sub-pixel offset)
    and added to a small float canvas.

    Parameters
    ----------
    kind : str
        One of ``PARTICLE_KINDS``.
    size : tuple of int, optional
        Layer size in pixels.
    period : float, optional
        Loop length in seconds; ``period * fps`` must be a whole number.
    fps : float, optional
        Frame rate used to quantise time.
    count : int, optional
        Particles; the default depends on the kind and the region area.
    seed : int, optional
        Random seed; equal arguments always give equal frames.
    color : tuple of int, optional
        Straight sRGB colour of all particles; per-kind default.
    opacity : float, optional
        Peak opacity in 0..1; per-kind default.
    speed : float, optional
        Multiplier of the fall speed (rounded to whole wraps per period, at
        least one) and of the wander cycles.
    region : tuple of int, optional
        ``(x, y, width, height)`` rectangle that confines the particles;
        everything outside stays fully transparent.
    name : str, optional
        Layer name.
    \*\*kwargs
        Passed to ``Layer``.

    Examples
    --------
    >>> layer = ParticleLayer("snow", size=(64, 36), period=2.0, fps=4, count=5)
    >>> layer.frames
    8
    >>> bool((layer.rgba_uint8(0) == layer.rgba_uint8(8)).all())
    True
    """

    def __init__(
        self,
        kind,
        *,
        size=(1280, 720),
        period=20.0,
        fps=24,
        count=None,
        seed=0,
        color=None,
        opacity=None,
        speed=1.0,
        region=None,
        name="Particles",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        if kind not in _KINDS:
            raise ValueError(f"unknown kind {kind!r}; use one of {PARTICLE_KINDS}")
        spec = _KINDS[kind]
        self.kind = kind
        self._size = (int(size[0]), int(size[1]))
        self.fps = float(fps)
        exact = float(period) * self.fps
        self.frames = int(round(exact))
        if self.frames < 2 or abs(exact - self.frames) > 1e-6:
            raise ValueError("period * fps must be a whole number of frames (>= 2)")
        self.period = self.frames / self.fps
        if kind == "fireflies" and self.period < 2.0:
            raise ValueError("fireflies need period >= 2 s to twinkle at <= 0.5 Hz")
        if not (math.isfinite(float(speed)) and speed > 0):
            raise ValueError("speed must be positive")
        if region is None:
            region = (0, 0, *self._size)
        rx, ry, rw, rh = (int(v) for v in region)
        if rx < 0 or ry < 0 or rw < 1 or rh < 1:
            raise ValueError("region must be (x, y, width, height) inside the layer")
        if rx + rw > self._size[0] or ry + rh > self._size[1]:
            raise ValueError("region must be (x, y, width, height) inside the layer")
        self.region = (rx, ry, rw, rh)
        self.color = tuple(int(c) for c in (spec["color"] if color is None else color))
        if len(self.color) != 3 or not all(0 <= c <= 255 for c in self.color):
            raise ValueError("color must be three 0..255 integers")
        self.opacity = float(spec["opacity"] if opacity is None else opacity)
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be in 0..1")
        scale = min(rw, rh) / 720.0
        area = rw * rh / (1280.0 * 720.0)
        n = max(1, int(round(spec["count"] * area))) if count is None else int(count)
        if n < 0:
            raise ValueError("count must be >= 0")
        self.count = n
        self.seed = int(seed)
        self.speed = float(speed)
        self._spec = spec
        self._scale = scale
        self._build(np.random.default_rng(self.seed), n, rw, rh, scale)
        self._cache = {}
        self._last = None

    # -- construction --------------------------------------------------- #

    def _build(self, rng, n, rw, rh, scale):
        spec = self._spec
        sizes = np.asarray(spec["sizes"], np.float64) * max(scale, 0.35)
        cls = rng.integers(0, len(sizes), n)
        self._cls = cls
        self._rep = {}
        for i, c in enumerate(cls):
            self._rep.setdefault(int(c), i)
        self._sigma_major = np.maximum(sizes[cls] * spec["shape"][0], 0.55)
        # the minor axis of a streak must stay a visible hairline
        self._sigma_minor = np.maximum(sizes[cls] * spec["shape"][1], 0.55)
        if spec["shape"][1] < 0.1:
            self._sigma_minor = np.full(n, max(0.7, 0.8 * scale))
            self._sigma_major = self._sigma_major / 2.4
        self.margin = (
            int(math.ceil(3 * float(np.max(self._sigma_major, initial=1)))) + 3
        )
        depth = (cls + 1) / len(sizes)  # bigger = nearer = faster, brighter
        self._alpha_base = (0.55 + 0.45 * depth) * self.opacity
        lo, hi = spec["sway"]
        self._sway = rng.uniform(lo, hi, n) * max(scale, 0.35) * depth
        k_lo, k_hi = spec["sway_cycles"]
        k_lo = max(1, int(round(k_lo * self.speed)))
        k_hi = max(k_lo, int(round(k_hi * self.speed)))
        self._kx = rng.integers(k_lo, k_hi + 1, n)
        self._ky = rng.integers(k_lo, k_hi + 1, n)
        self._phx = rng.random(n)
        self._phy = rng.random(n)
        self._cx = rng.uniform(0, rw, n)
        self._cy = rng.uniform(0, rh, n)
        cap = max(1, int(math.floor(0.5 * self.period)))  # <= 0.5 Hz
        top = max(1, min(cap, int(math.floor(0.25 * self.period)) or 1))
        self._kt = rng.integers(1, top + 1, n)
        self._pht = rng.random(n)
        self._twinkle = float(spec["twinkle"])
        wraps = spec["wraps"]
        if spec["fall"]:
            base = wraps * self.speed * (1.0 + depth)
            self._wraps = np.maximum(1, np.rint(base)).astype(np.int64)
            self._y0 = rng.uniform(0, 1, n)
        slope = spec["slope"]
        self._slope = slope
        if slope:
            theta = math.atan2(1.0, slope)
        else:
            theta = math.pi / 2 if spec["shape"][1] < 0.1 else 0.0
        self._angle0 = np.full(n, int(round(theta / math.pi * _ANGLES)) % _ANGLES)
        if spec["rotate"]:
            self._angle0 = rng.integers(0, _ANGLES, n)
            sign = rng.choice([-1, 1], n)
            self._rot = sign * np.maximum(1, np.rint(spec["rotate"] * self.speed))
            self._rot = self._rot.astype(np.int64)
        else:
            self._rot = np.zeros(n, np.int64)

    # -- rendering ------------------------------------------------------ #

    @property
    def source_size(self):
        """Return the layer size in pixels."""
        return self._size

    def _positions(self, n):
        """Return x, y, alpha and angle index of every particle at frame ``n``."""
        rx, ry, rw, rh = self.region
        phase = n / self.frames
        two_pi = 2.0 * math.pi
        sway_x = self._sway * np.sin(two_pi * (self._kx * phase + self._phx))
        sway_y = self._sway * np.sin(two_pi * (self._ky * phase + self._phy))
        m = self.margin
        lx, ly = rw + 2 * m, rh + 2 * m
        if self._spec["fall"]:
            fall = (self._y0 + self._wraps * phase) % 1.0
            yy = fall * ly - m
            x = self._cx + sway_x + self._slope * (yy + m)
            y = yy
        else:
            x = self._cx + sway_x
            y = self._cy + sway_y
        x = (x + m) % lx - m
        y = (y + m) % ly - m
        twinkle = 0.5 + 0.5 * np.sin(two_pi * (self._kt * phase + self._pht))
        alpha = self._alpha_base * (1.0 - self._twinkle * (1.0 - twinkle))
        angle = (self._angle0 + (self._rot * _ANGLES * n) // self.frames) % _ANGLES
        return x, y, alpha, angle

    def _get_sprite(self, key):
        found = self._cache.get(key)
        if found is None:
            index, angle, qx, qy = key
            if self._spec.get("glyph") == "crystal":
                found = _crystal_sprite(
                    float(self._sigma_major[self._rep[index]]),
                    math.pi * angle / _ANGLES,
                    qx / _SUBPIXEL,
                    qy / _SUBPIXEL,
                )
                self._cache[key] = found
                return found
            found = _sprite(
                float(self._sigma_major[self._rep[index]]),
                float(self._sigma_minor[self._rep[index]]),
                self._spec["power"],
                math.pi * angle / _ANGLES,
                qx / _SUBPIXEL,
                qy / _SUBPIXEL,
            )
            self._cache[key] = found
        return found

    def alpha_canvas(self, frame):
        """Return the float32 ``(h, w)`` alpha of the region at a frame index.

        Parameters
        ----------
        frame : int
            Frame index; taken modulo the loop length.
        """
        n = int(frame) % self.frames
        _, _, rw, rh = self.region
        canvas = np.zeros((rh, rw), np.float32)
        if self.count == 0:
            return canvas
        x, y, alpha, angle = self._positions(n)
        qx = np.rint(x * _SUBPIXEL).astype(np.int64)
        qy = np.rint(y * _SUBPIXEL).astype(np.int64)
        ix, fx = np.divmod(qx, _SUBPIXEL)
        iy, fy = np.divmod(qy, _SUBPIXEL)
        classes = self._cls
        for i in range(self.count):
            sprite, half = self._get_sprite(
                (int(classes[i]), int(angle[i]), int(fx[i]), int(fy[i]))
            )
            x0, y0 = int(ix[i]) - half, int(iy[i]) - half
            x1, y1 = x0 + sprite.shape[1], y0 + sprite.shape[0]
            sx0, sy0 = max(0, -x0), max(0, -y0)
            sx1 = sprite.shape[1] - max(0, x1 - rw)
            sy1 = sprite.shape[0] - max(0, y1 - rh)
            if sx0 >= sx1 or sy0 >= sy1:
                continue
            canvas[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1] += sprite[
                sy0:sy1, sx0:sx1
            ] * np.float32(alpha[i])
        np.minimum(canvas, 1.0, out=canvas)
        return canvas

    def frame_index(self, t):
        """Return the loop frame shown at time ``t`` (period-wrapped)."""
        return int(round(float(t) * self.fps)) % self.frames

    def rgba_uint8(self, frame):
        """Return straight-alpha uint8 ``(H, W, 4)`` pixels of a loop frame.

        Parameters
        ----------
        frame : int
            Frame index; taken modulo the loop length.
        """
        canvas = self.alpha_canvas(frame)
        rx, ry, rw, rh = self.region
        out = np.zeros((self._size[1], self._size[0], 4), np.uint8)
        alpha = np.rint(canvas * 255.0).astype(np.uint8)
        view = out[ry : ry + rh, rx : rx + rw]
        view[..., 3] = alpha
        mask = alpha > 0
        for channel, value in enumerate(self.color):
            view[..., channel][mask] = value
        return out

    def rgba_at(self, t):
        """Return ``rgba_uint8`` for time ``t`` (used by the overlay renderer)."""
        return self.rgba_uint8(self.frame_index(t))

    def source_buffer(self, t, context=None):
        """Return the premultiplied float RGBA buffer at time ``t``."""
        n = self.frame_index(t)
        if self._last is not None and self._last[0] == n:
            return self._last[1]
        canvas = self.alpha_canvas(n)
        rx, ry, rw, rh = self.region
        rgba = np.zeros((self._size[1], self._size[0], 4), np.float32)
        view = rgba[ry : ry + rh, rx : rx + rw]
        for channel, value in enumerate(self.color):
            view[..., channel] = canvas * np.float32(value / 255.0)
        view[..., 3] = canvas
        buffer = Buffer(rgba)
        self._last = (n, buffer)
        return buffer


def particle_loop(
    kind="fireflies",
    *,
    size=(1280, 720),
    period=20.0,
    fps=24,
    count=None,
    seed=0,
    color=None,
    opacity=None,
    speed=1.0,
    region=None,
):
    """Return a transparent, exactly periodic particle composition.

    The composition lasts exactly ``period`` seconds; frame ``n + period*fps``
    is identical to frame ``n``, so FFmpeg's ``-stream_loop -1`` repeats it with
    no seam. Defaults are gentle (low density, slow, no flashing): fireflies
    wander and twinkle at up to 0.25 Hz (hard limit 0.5 Hz), dust drifts, petals
    and leaves sway and turn, snow falls, snowflakes (six-armed crystals) fall
    while turning slowly, and rain is thin and translucent.

    Parameters
    ----------
    kind : str
        ``"fireflies"``, ``"dust"``, ``"petals"``, ``"leaves"``, ``"snow"``,
        ``"snowflakes"`` (six-armed crystals turning slowly)
        or ``"rain"``.
    size : tuple of int
        Composition size ``(width, height)``.
    period : float
        Loop length in seconds (``period * fps`` must be a whole number).
        A falling particle crosses the region at least once per period, so
        a longer period means slower fall.
    fps : float
        Frame rate.
    count : int, optional
        Number of particles (default per kind, scaled by region area).
    seed : int
        Random seed; the same arguments always give the same pixels.
    color : tuple of int, optional
        Straight RGB colour (per-kind default).
    opacity : float, optional
        Peak opacity in 0..1 (per-kind default).
    speed : float
        Multiplier of fall speed and wander cycles (whole cycles, at least 1).
    region : tuple of int, optional
        ``(x, y, width, height)`` confining all particles.

    Returns
    -------
    moviepy.ae.composition.Composition
        Transparent composition with one ``ParticleLayer``; the layer is also
        ``comp.particle_layer`` and ``comp.fast_rgba(t)`` returns the straight
        uint8 RGBA frame without the compositor.

    Examples
    --------
    >>> comp = particle_loop("dust", size=(64, 36), period=2.0, fps=4, count=4)
    >>> comp.duration, comp.size
    (2.0, (64, 36))
    """
    layer = ParticleLayer(
        kind,
        size=size,
        period=period,
        fps=fps,
        count=count,
        seed=seed,
        color=color,
        opacity=opacity,
        speed=speed,
        region=region,
        name=kind.capitalize(),
    )
    comp = Composition(
        size=size, fps=fps, duration=layer.period, transparent=True, name=f"{kind}"
    )
    comp.add_layer(layer)
    comp.particle_layer = layer
    comp.fast_rgba = layer.rgba_at
    return comp


def _particle_factory(kind, options):
    """Top-level factory so ``particle_loop`` can run in worker processes."""
    return particle_loop(kind, **dict(options))


# --------------------------------------------------------------------------- #
# overlay loop export
# --------------------------------------------------------------------------- #

_CODECS = {
    "qtrle": ["-c:v", "qtrle", "-pix_fmt", "argb"],
    "prores_4444": [
        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
        "-alpha_bits", "16",
    ],
    "png": ["-c:v", "png", "-pix_fmt", "rgba"],
}  # fmt: skip


def _decode_command(path, size):
    return [
        FFMPEG_BINARY, "-v", "error", "-nostdin", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgba", "-",
    ]  # fmt: skip


def export_overlay_loop(comp, path, *, codec="qtrle", overwrite=False, workers=None):
    """Write a transparent composition as a MOV with alpha for FFmpeg to loop.

    Parameters
    ----------
    comp : Composition or tuple
        A composition (for example ``particle_loop(...)``) or ``(factory,
        args)`` of a picklable top-level factory, for multi-process rendering;
        ``(_particle_factory, (kind, options))`` builds particles.
    path : str or os.PathLike
        Output ``.mov``; ``<path>.json`` evidence is written beside it.
    codec : str
        ``"qtrle"`` (lossless, default), ``"prores_4444"`` or ``"png"``.
    overwrite : bool
        Replace existing files; by default they raise ``FileExistsError``.
    workers : int, optional
        Render processes; ``None`` renders sequentially for a composition.

    Returns
    -------
    dict
        Evidence: ``path``, ``codec``, ``frames``, ``fps``, ``size``,
        ``seconds``, ``frames_sha256``, ``alpha`` (min/max, covered share),
        ``seam`` (frame 0 against the frame at ``t = period``, and the
        last-to-first step against the median step, both on RGBA),
        ``roundtrip`` (decoded again with alpha; first, middle and last frames
        compared) and ``human_visual`` set to ``"NOT_RUN"``.

    Raises
    ------
    FileExistsError
        If an output file exists and ``overwrite`` is false.
    RuntimeError
        If FFmpeg fails or a lossless codec does not return the alpha exactly.

    Examples
    --------
    >>> import inspect
    >>> inspect.signature(export_overlay_loop).parameters["codec"].default
    'qtrle'
    """
    if codec not in _CODECS:
        raise ValueError(f"codec must be one of {tuple(_CODECS)}")
    output = Path(path)
    evidence_path = Path(str(output) + ".json")
    if not overwrite:
        for item in (output, evidence_path):
            if item.exists():
                raise FileExistsError(f"{item} already exists")
    if isinstance(comp, tuple):
        factory, args = comp
        probe = factory(*args)
        own = True
    else:
        factory, args, probe, own = None, (), comp, False
    try:
        fps = float(probe.fps)
        width, height = (int(v) for v in probe.size)
        frames = int(round(float(probe.duration) * fps))
        try:
            wrap_frame = _rgba_frame(probe, frames / fps)
        except Exception:
            wrap_frame = None
    finally:
        if own:
            probe.close()
    if frames < 2:
        raise ValueError("a loop needs at least two frames")
    if workers is None:
        workers = 1 if factory is None else None
    if factory is None and workers not in (None, 1):
        raise ValueError("workers > 1 needs a (factory, args) tuple")
    times = [i / fps for i in range(frames)]
    rows = -(-4 * height // 3)
    if factory is None:
        source = (
            _unpack(f, (height, width, 4))
            for f in iter_frames_parallel(
                None, (), times, workers=1, size=(width, rows),
                clip=_PackedRGBA(comp),
            )
        )  # fmt: skip
    else:
        from moviepy.ae.templates.music_render import _packed_factory

        source = (
            _unpack(f, (height, width, 4))
            for f in iter_frames_parallel(
                _packed_factory, (factory, args), times, workers=workers,
                size=(width, rows),
            )
        )  # fmt: skip
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        FFMPEG_BINARY, "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}", "-i", "pipe:0", *_CODECS[codec], str(output),
    ]  # fmt: skip
    began = time.perf_counter()
    digest = hashlib.sha256()
    first = previous = None
    steps = []
    covered = 0.0
    alpha_max = 0
    keep = {}
    wanted = {0, frames // 2, frames - 1}
    proc = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, creationflags=_NO_WINDOW,
    )  # fmt: skip
    failed = False
    try:
        done = 0
        for frame in source:
            frame = np.ascontiguousarray(frame)
            digest.update(frame.data)
            if first is None:
                first = frame.copy()
            if previous is not None:
                steps.append(float(np.abs(frame.astype(np.int16) - previous).mean()))
            previous = frame.astype(np.int16)
            plane = frame[..., 3]
            covered += float((plane > 0).mean())
            alpha_max = max(alpha_max, int(plane.max()))
            if done in wanted:
                keep[done] = frame.copy()
            try:
                proc.stdin.write(frame.data)
            except (BrokenPipeError, OSError):
                failed = True
                break
            done += 1
        try:
            proc.stdin.close()
        except OSError:
            failed = True
        stderr = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        close = getattr(source, "close", None)
        if close is not None:
            close()
    if code != 0 or failed or done != frames:
        if output.exists():
            output.unlink()
        raise RuntimeError(f"ffmpeg failed ({codec}):\n{stderr[-1500:]}")
    last = previous.astype(np.uint8)
    wrap_step = float(np.abs(first.astype(np.int16) - last.astype(np.int16)).mean())
    median_step = float(np.median(steps)) if steps else 0.0
    seam = {
        "frame0_equals_frame_period": (
            None if wrap_frame is None else bool(np.array_equal(first, wrap_frame))
        ),
        "last_to_first_mean_abs_diff": wrap_step,
        "median_step_mean_abs_diff": median_step,
        "max_step_mean_abs_diff": float(np.max(steps)) if steps else 0.0,
    }
    roundtrip = _roundtrip(output, width, height, frames, keep, codec)
    if codec != "prores_4444" and roundtrip["alpha_max_abs_diff"] != 0:
        raise RuntimeError(f"{codec} did not return the alpha exactly: {roundtrip}")
    report = {
        "path": str(output),
        "codec": codec,
        "frames": frames,
        "fps": fps,
        "size": [width, height],
        "seconds": frames / fps,
        "frames_sha256": digest.hexdigest(),
        "alpha": {
            "max": alpha_max,
            "mean_covered_share": covered / frames,
        },
        "seam": seam,
        "roundtrip": roundtrip,
        "output_bytes": output.stat().st_size,
        "elapsed": time.perf_counter() - began,
        "command": command,
        "human_visual": "NOT_RUN",
    }
    with open(evidence_path, "w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def _roundtrip(path, width, height, frames, keep, codec):
    """Decode ``path`` with alpha; compare the kept source frames."""
    proc = subprocess.Popen(
        _decode_command(path, (width, height)), stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, creationflags=_NO_WINDOW,
    )  # fmt: skip
    size = width * height * 4
    alpha_diff = rgb_diff = 0
    count = 0
    while True:
        raw = proc.stdout.read(size)
        if len(raw) < size:
            break
        if count in keep:
            got = np.frombuffer(raw, np.uint8).reshape(height, width, 4)
            ref = keep[count]
            alpha_diff = max(
                alpha_diff,
                int(np.abs(got[..., 3].astype(np.int16) - ref[..., 3]).max()),
            )
            covered = ref[..., 3] > 0
            if covered.any():
                rgb_diff = max(
                    rgb_diff,
                    int(
                        np.abs(
                            got[..., :3][covered].astype(np.int16)
                            - ref[..., :3][covered]
                        ).max()
                    ),
                )
        count += 1
    proc.stdout.close()
    proc.stderr.read()
    proc.wait()
    return {
        "frames_decoded": count,
        "frames_match": count == frames,
        "compared_frames": sorted(keep),
        "alpha_max_abs_diff": alpha_diff,
        "rgb_max_abs_diff_where_visible": rgb_diff,
    }


# --------------------------------------------------------------------------- #
# light / colour arc
# --------------------------------------------------------------------------- #

_NEUTRAL = {"brightness": 0.0, "saturation": 1.0, "warmth": 0.0, "contrast": 1.0}
_LIMITS = {
    "brightness": (-1.0, 1.0),
    "saturation": (0.0, 3.0),
    "warmth": (-1.0, 1.0),
    "contrast": (0.0, 3.0),
}
_WARMTH_GAMMA = 0.25  # gamma_r = 1 + 0.25 * warmth, gamma_b = 1 - 0.25 * warmth
_LIMITED = 112 / 255  # eq pivot (128) minus the limited-range black code (16)
_MAX_RATE = 0.1  # of the unit range per second


def _num(value):
    text = f"{value:.6g}"
    return "0" if text in ("-0", "0") else text


def _ease(u):
    return (1.0 - math.cos(math.pi * min(1.0, max(0.0, u)))) / 2.0


class LightArc:
    """A slow light and colour curve compiled into one FFmpeg expression.

    Between keyframes every parameter follows a cosine ease (zero slope at the
    keys, so changes start and stop gently); before the first and after the last
    key the value is held. ``to_ffmpeg_filter`` writes each parameter as
    ``v0 + sum(delta_i * ease_i(t))``, a flat sum without ``if`` nesting, so
    the expression grows linearly (about 70 characters per changing segment and
    parameter) and FFmpeg parses 24, 100 or more keyframes without trouble.

    Parameters
    ----------
    keyframes : sequence
        ``(time_seconds, {"brightness": b, "saturation": s, "warmth": w,
        "contrast": c})`` with strictly increasing times. Omitted keys are
        neutral (brightness 0, saturation 1, warmth 0, contrast 1). Meaning
        follows FFmpeg ``eq``: brightness is added (-1..1), saturation and
        contrast multiply (1 = unchanged), warmth (-1 cool .. 1 warm) raises
        red gamma and lowers blue gamma by up to 25 %.
    max_rate : float or None
        Largest allowed change per second as a fraction of the parameter's
        unit (peak slope of the cosine); a faster arc raises ``ValueError``
        so nothing can flicker. ``None`` disables the check.

    Examples
    --------
    >>> arc = LightArc.day_to_night(3600)
    >>> arc.value_at(0)["brightness"], arc.value_at(3600)["warmth"] < 0
    (0.0, True)
    >>> arc.to_ffmpeg_filter().startswith("eq=eval=frame:")
    True
    """

    def __init__(self, keyframes, *, max_rate=_MAX_RATE):
        keys = []
        for item in keyframes:
            when, values = item
            when = float(when)
            full = dict(_NEUTRAL)
            for name, value in dict(values).items():
                if name not in _NEUTRAL:
                    raise ValueError(f"unknown parameter {name!r}")
                low, high = _LIMITS[name]
                if not low <= float(value) <= high:
                    raise ValueError(f"{name}={value} outside [{low}, {high}]")
                full[name] = float(value)
            if not math.isfinite(when) or when < 0:
                raise ValueError("keyframe times must be finite and >= 0")
            if keys and when <= keys[-1][0]:
                raise ValueError("keyframe times must strictly increase")
            keys.append((when, full))
        if not keys:
            raise ValueError("at least one keyframe is required")
        self.keyframes = tuple(keys)
        if max_rate is not None:
            for name, rate in self.max_rates().items():
                if rate > max_rate:
                    raise ValueError(
                        f"{name} changes at {rate:.3f}/s (limit {max_rate}); "
                        "lengthen the segment"
                    )

    # -- evaluation ------------------------------------------------------ #

    def max_rates(self):
        """Return the peak change per second of every parameter."""
        rates = dict.fromkeys(_NEUTRAL, 0.0)
        for (a, va), (b, vb) in zip(self.keyframes, self.keyframes[1:]):
            for name in rates:
                peak = math.pi / 2.0 * abs(vb[name] - va[name]) / (b - a)
                rates[name] = max(rates[name], peak)
        return rates

    def value_at(self, t):
        """Return the four parameter values at time ``t`` (seconds).

        Parameters
        ----------
        t : float
            Time in seconds.
        """
        keys = self.keyframes
        t = float(t)
        out = dict(keys[0][1])
        for (a, _), (b, vb), (_, va) in zip(keys, keys[1:], keys):
            u = _ease((t - a) / (b - a))
            for name in out:
                out[name] += (vb[name] - va[name]) * u
        return out

    # -- FFmpeg ----------------------------------------------------------- #

    def _expression(self, name, scale=1.0, offset=0.0):
        keys = self.keyframes
        base = offset + scale * keys[0][1][name]
        text = _num(base)
        for (a, va), (b, vb) in zip(keys, keys[1:]):
            delta = scale * (vb[name] - va[name])
            if delta == 0.0:
                continue
            text += (
                f"+({_num(delta / 2.0)})*(1-cos(PI*clip((t-{_num(a)})"
                f"/{_num(b - a)},0,1)))"
            )
        return text

    def to_ffmpeg_filter(self):
        """Return an ``eq=...:eval=frame`` filter applying the arc over ``t``.

        FFmpeg evaluates the expressions once per frame from the frame time, so
        hours of video cost nothing in Python. ``t`` is the filter's input
        time: when a render starts at ``S`` seconds, ``music_render`` gives the
        frames global timestamps first.
        """
        parts = [
            "eq=eval=frame",
            "brightness='" + self._expression("brightness") + "'",
            "contrast='" + self._expression("contrast") + "'",
            "saturation='" + self._expression("saturation") + "'",
            "gamma_r='" + self._expression("warmth", _WARMTH_GAMMA, 1.0) + "'",
            "gamma_b='" + self._expression("warmth", -_WARMTH_GAMMA, 1.0) + "'",
        ]
        return ":".join(parts)

    # -- presets ---------------------------------------------------------- #

    @classmethod
    def _preset(cls, duration, rows, max_rate):
        duration = float(duration)
        if not (math.isfinite(duration) and duration > 0):
            raise ValueError("duration must be positive")
        return cls([(duration * f, v) for f, v in rows], max_rate=max_rate)

    @classmethod
    def day_to_night(cls, duration, *, max_rate=_MAX_RATE):
        """Bright warm day through golden hour to a dim, cool night.

        Parameters
        ----------
        duration : float
            Length of the whole arc in seconds.
        max_rate : float or None
            Change-rate limit, see the class.
        """
        return cls._preset(duration, [
            (0.0, dict(brightness=0.0, saturation=1.0, warmth=0.15)),
            (0.35, dict(brightness=-0.02, saturation=1.05, warmth=0.45)),
            (0.65, dict(brightness=-0.09, saturation=0.9, warmth=0.1)),
            (1.0, dict(brightness=-0.22, saturation=0.6, warmth=-0.4, contrast=0.92)),
        ], max_rate)  # fmt: skip

    @classmethod
    def dusk(cls, duration, *, max_rate=_MAX_RATE):
        """Warm evening light that slowly dims and softens.

        Parameters
        ----------
        duration : float
            Length of the whole arc in seconds.
        max_rate : float or None
            Change-rate limit, see the class.
        """
        return cls._preset(duration, [
            (0.0, dict(brightness=0.0, saturation=1.0, warmth=0.1)),
            (0.5, dict(brightness=-0.05, saturation=0.97, warmth=0.5)),
            (1.0, dict(brightness=-0.15, saturation=0.8, warmth=0.2, contrast=0.95)),
        ], max_rate)  # fmt: skip

    @classmethod
    def dawn(cls, duration, *, max_rate=_MAX_RATE):
        """Dim cool night that warms and brightens into morning.

        Parameters
        ----------
        duration : float
            Length of the whole arc in seconds.
        max_rate : float or None
            Change-rate limit, see the class.
        """
        return cls._preset(duration, [
            (0.0, dict(brightness=-0.22, saturation=0.6, warmth=-0.4, contrast=0.92)),
            (0.5, dict(brightness=-0.08, saturation=0.85, warmth=0.35)),
            (1.0, dict(brightness=0.0, saturation=1.0, warmth=0.15)),
        ], max_rate)  # fmt: skip

    def describe(self):
        """Return a JSON-friendly description for evidence files."""
        return {
            "keyframes": [[t, dict(v)] for t, v in self.keyframes],
            "max_rates": self.max_rates(),
            "filter": self.to_ffmpeg_filter(),
        }


# --------------------------------------------------------------------------- #
# sleep fade
# --------------------------------------------------------------------------- #


class SleepFade:
    """Fade the picture, and optionally the music, to a floor and hold it.

    The gain follows a cosine ease from 1 at ``start`` to ``floor`` at ``end``
    and stays at ``floor`` afterwards. The video filter scales luma and chroma
    by the gain (``eq``: ``contrast=g`` with ``brightness=-(112/255)(1-g)`` maps
    limited-range luma code ``Y`` to ``16 + g (Y - 16)``, that is every RGB value
    times ``g``); the audio filter is a ``volume`` expression of the same
    gain (the music tail). Both read the global time ``t``.

    Parameters
    ----------
    start : float
        Seconds where the fade begins.
    end : float
        Seconds where ``floor`` is reached; must be greater than ``start``.
    floor : float
        Gain held from ``end`` on: 0 is black / silence, 0.2 a dim glow.
    audio : bool
        Whether ``audio_filter`` returns a filter (otherwise ``None``).
    max_rate : float or None
        Largest allowed change of the gain per second (peak of the cosine);
        shorter fades raise ``ValueError``. ``None`` disables the check.

    Examples
    --------
    >>> fade = SleepFade(100, 200, floor=0.1)
    >>> fade.gain_at(0), fade.gain_at(150), fade.gain_at(999)
    (1.0, 0.55, 0.1)
    >>> fade.validate(300)
    """

    def __init__(self, start, end, floor=0.0, audio=True, *, max_rate=_MAX_RATE):
        self.start, self.end = float(start), float(end)
        self.floor = float(floor)
        self.audio = bool(audio)
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError("start and end must be finite")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("need 0 <= start < end")
        if not 0.0 <= self.floor < 1.0:
            raise ValueError("floor must be in [0, 1)")
        rate = math.pi / 2.0 * (1.0 - self.floor) / (self.end - self.start)
        if max_rate is not None and rate > max_rate:
            raise ValueError(
                f"fade changes {rate:.3f}/s (limit {max_rate}); lengthen it"
            )

    def validate(self, duration):
        """Raise ``ValueError`` if the fade does not end inside ``duration``.

        Parameters
        ----------
        duration : float
            Video length in seconds.
        """
        if not self.end <= float(duration) + 1e-9:
            raise ValueError(f"fade ends at {self.end:g} s, after {duration:g} s")

    def gain_at(self, t):
        """Return the gain at time ``t`` (1 before ``start``, ``floor`` after).

        Parameters
        ----------
        t : float
            Time in seconds.
        """
        u = _ease((float(t) - self.start) / (self.end - self.start))
        return round(1.0 - (1.0 - self.floor) * u, 12)

    def _gain_expression(self):
        return (
            f"1-({_num(1.0 - self.floor)})*(1-cos(PI*clip((t-{_num(self.start)})"
            f"/{_num(self.end - self.start)},0,1)))/2"
        )

    def video_filter(self):
        """Return the ``eq`` filter that fades luma and chroma by the gain."""
        gain = self._gain_expression()
        return (
            f"eq=eval=frame:contrast='{gain}':brightness='-{_LIMITED}*(1-({gain}))'"
            f":saturation='{gain}'"
        )

    def audio_filter(self):
        """Return the ``volume`` filter for the music, or ``None``."""
        if not self.audio:
            return None
        return f"volume=eval=frame:volume='{self._gain_expression()}':precision=float"

    def describe(self):
        """Return a JSON-friendly description for evidence files."""
        return {
            "start": self.start,
            "end": self.end,
            "floor": self.floor,
            "video_filter": self.video_filter(),
            "audio_filter": self.audio_filter(),
        }
