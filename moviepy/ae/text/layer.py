"""Pillow text sources with deterministic entrance animation."""

import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.ae._geometry import finite_real, validate_pixel_size
from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.base import Layer
from moviepy.ae.layers.solid import _color
from moviepy.ae.text.clusters import clusters


PRESETS = ("none", "typewriter", "fade_up", "pop", "word_fade")


class TextLayer(Layer):
    """Render a fixed title with a reusable entrance preset.

    Pass an explicit local font for Chinese or other non-Latin text. The
    Pillow default is a portable Latin fallback. Kerning/shaping is handled
    by Pillow. ``rate`` is reveal units/second; typewriter displays floor
    (rate * max(0, t-delay)) units. ``word_fade`` staggers whitespace-delimited
    words; Chinese titles can use typewriter. Presets are not Adobe files.
    Styles are set at construction; normal layer transforms/effects animate
    the resulting source. No font is downloaded or redistributed.
    """

    def __init__(
        self,
        text,
        name="Text",
        *,
        font=None,
        font_size=48,
        color=(255, 255, 255),
        preset="none",
        rate=12,
        animate_duration=0.7,
        delay=0.0,
        stagger=0.12,
        padding=None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}")
        self._rate = finite_real(rate, "rate")
        self._entrance = finite_real(animate_duration, "animate_duration")
        self._delay = finite_real(delay, "delay")
        self._stagger = finite_real(stagger, "stagger")
        size = finite_real(font_size, "font_size")
        if (
            self._rate <= 0
            or self._entrance <= 0
            or self._stagger < 0
            or size < 1
            or size != int(size)
        ):
            raise ValueError(
                "rate/duration/font_size must be positive; font_size integral; stagger nonnegative"
            )
        if font is None:
            if any(ord(c) > 255 for c in text):
                raise ValueError("an explicit font file is required for non-Latin text")
            try:
                self._font = ImageFont.load_default(size=int(size))
            except TypeError as error:
                raise ValueError(
                    "this Pillow version needs an explicit font file for font_size"
                ) from error
        else:
            self._font = ImageFont.truetype(str(Path(font).expanduser()), int(size))
        self._text, self._preset = text, preset
        self._units = clusters(text)
        self._fill = tuple(round(c * 255) for c in _color(color)) + (255,)
        pad = int(size) if padding is None else finite_real(padding, "padding")
        if pad < 0 or pad != int(pad):
            raise ValueError("padding must be a nonnegative integer")
        self._pad, self._font_size = int(pad), size
        box = ImageDraw.Draw(Image.new("L", (1, 1))).multiline_textbbox(
            (0, 0), text or " ", font=self._font
        )
        self._origin = (self._pad - box[0], self._pad - box[1])
        self._size = validate_pixel_size(
            (
                max(1, box[2] - box[0] + 2 * self._pad),
                max(1, box[3] - box[1] + 2 * self._pad),
            )
        )
        self._cache = {}

    @property
    def source_size(self):
        """Return the full title canvas; typewriter never changes its anchor."""
        return self._size

    def visible_text(self, t):
        """Return the currently revealed text, grouping common graphemes."""
        time = finite_real(t, "t") - self._delay
        if time < 0:
            return ""
        if self._preset != "typewriter":
            return self._text
        return "".join(self._units[: max(0, math.floor(self._rate * time))])

    def _raster(self, text):
        if text not in self._cache:
            image = Image.new("RGBA", self._size)
            ImageDraw.Draw(image).multiline_text(
                self._origin, text, font=self._font, fill=self._fill
            )
            array = np.asarray(image)
            buffer = Buffer.from_uint8_rgb(
                array[..., :3], array[..., 3].astype(np.float32) / 255
            )
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = buffer
        return self._cache[text]

    def source_buffer(self, t, context=None):
        """Render source-local entrance motion without mutable playback state."""
        time = finite_real(t, "t") - self._delay
        if time < 0:
            return self._raster("")
        if self._preset == "word_fade":
            return self._words(time)
        buffer = self._raster(self.visible_text(t))
        if self._preset in ("none", "typewriter"):
            return buffer
        progress = min(1.0, time / self._entrance)
        ease = 1 - (1 - progress) ** 3
        if progress == 1:
            return buffer
        matrix = np.array(
            [[1, 0, 0], [0, 1, (1 - ease) * self._font_size * 0.6]], np.float64
        )
        if self._preset == "pop":
            scale = 0.7 + 0.3 * ease
            matrix = cv2.getRotationMatrix2D(
                ((self._size[0] - 1) / 2, (self._size[1] - 1) / 2), 0, scale
            )
        rgba = cv2.warpAffine(buffer.rgba, matrix, self._size, flags=cv2.INTER_LINEAR)
        rgba *= np.float32(ease)
        return Buffer._publish(rgba, (0, 0), "srgb")

    def _words(self, time):
        words = len(re.findall(r"\S+", self._text))
        if time >= self._entrance + max(0, words - 1) * self._stagger:
            return self._raster(self._text)
        image = Image.new("RGBA", self._size)
        drawing = ImageDraw.Draw(image)
        number, y = 0, self._origin[1]
        for line in self._text.split("\n"):
            prefix = ""
            for token in re.findall(r"\S+|\s+", line):
                if not token.isspace():
                    phase = np.clip(
                        (time - number * self._stagger) / self._entrance, 0, 1
                    )
                    ease = 1 - (1 - float(phase)) ** 3
                    x = self._origin[0] + self._font.getlength(prefix)
                    drawing.text(
                        (x, y + (1 - ease) * self._font_size * 0.6),
                        token,
                        font=self._font,
                        fill=(*self._fill[:3], round(255 * ease)),
                    )
                    number += 1
                prefix += token
            y += self._font.getbbox("A")[3] + 4
        array = np.asarray(image)
        return Buffer.from_uint8_rgb(
            array[..., :3], array[..., 3].astype(np.float32) / 255
        )
