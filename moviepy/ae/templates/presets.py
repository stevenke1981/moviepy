"""Validated channel presets shared by every production template.

A preset only stores numbers, colors and font *paths*; templates read it and
never mutate it. ``with_overrides`` returns a validated copy, and JSON round
trips keep projects reproducible without importing template code.

Examples
--------
>>> from moviepy.ae.templates.presets import get_preset
>>> preset = get_preset("nightlamp_story")
>>> preset.size, preset.fps, preset.overlay_opacity
((1920, 1080), 24.0, 0.5)
>>> preset.with_overrides(fps=30).fps
30.0
>>> preset.scaled_px(72, (1280, 720))
48.0
"""

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path


def _color(value, name):
    if isinstance(value, str):
        text = value.lstrip("#")
        if len(text) != 6:
            raise ValueError(f"{name} must be #RRGGBB")
        try:
            return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError as error:
            raise ValueError(f"{name} must be #RRGGBB") from error
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must be three 0..255 RGB codes")
    result = []
    for channel in value:
        if isinstance(channel, bool) or not isinstance(channel, (int, float)):
            raise ValueError(f"{name} must be three 0..255 RGB codes")
        if not 0 <= channel <= 255:
            raise ValueError(f"{name} must be three 0..255 RGB codes")
        result.append(int(round(channel)))
    return tuple(result)


def _contrast(a, b):
    """WCAG contrast ratio of two sRGB colours (1 .. 21)."""

    def lum(rgb):
        c = [v / 255.0 for v in rgb]
        c = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not (math.isfinite(value) and low <= value <= high):
        raise ValueError(f"{name} must lie in [{low}, {high}]")
    return value


@dataclass(frozen=True)
class SubtitleStyle:
    """Burned-in subtitle geometry at the preset's reference height.

    ``primary_size``/``secondary_size`` are pixels at ``reference_height``;
    ``margin_bottom`` is measured from the frame bottom to the last line.
    ``max_lines`` limits each language block and ``max_width`` is the line
    width as a fraction of the frame width.

    ``plate_opacity > 0`` draws a ``plate_color`` rectangle behind each
    rendered line (its ink box including the outline, grown by
    ``plate_padding`` pixels at ``reference_height``). ``highlight_color``
    colours glossary terms; ``None`` falls back to the preset palette role
    ``"accent"`` when highlight terms are given.

    Examples
    --------
    >>> SubtitleStyle(plate_opacity=0.05).plate_color
    (0, 0, 0)
    >>> style = SubtitleStyle(highlight_color="#C9A35D")
    >>> style.highlight_color
    (201, 163, 93)
    >>> SubtitleStyle.from_dict(style.to_dict()) == style
    True
    """

    primary_size: float = 72
    secondary_size: float = 42
    primary_color: tuple = (255, 255, 255)
    secondary_color: tuple = (235, 235, 235)
    outline_color: tuple = (0, 0, 0)
    outline_width: float = 4
    margin_bottom: float = 60
    line_gap: float = 10
    max_lines: int = 2
    max_width: float = 0.86
    reference_height: int = 1080
    plate_color: tuple = (0, 0, 0)
    plate_opacity: float = 0.0
    plate_padding: float = 0
    highlight_color: tuple = None

    def __post_init__(self):
        object.__setattr__(self, "plate_color", _color(self.plate_color, "plate_color"))
        object.__setattr__(
            self, "plate_opacity", _number(self.plate_opacity, "plate_opacity", 0, 1)
        )
        object.__setattr__(
            self, "plate_padding", _number(self.plate_padding, "plate_padding", 0, 1000)
        )
        if self.highlight_color is not None:
            object.__setattr__(
                self, "highlight_color", _color(self.highlight_color, "highlight_color")
            )
        for name in ("primary_size", "secondary_size", "outline_width"):
            object.__setattr__(self, name, _number(getattr(self, name), name, 0, 1000))
        for name in ("margin_bottom", "line_gap"):
            object.__setattr__(self, name, _number(getattr(self, name), name, 0, 4000))
        for name in ("primary_color", "secondary_color", "outline_color"):
            object.__setattr__(self, name, _color(getattr(self, name), name))
        object.__setattr__(
            self, "max_width", _number(self.max_width, "max_width", 0.1, 1)
        )
        if isinstance(self.max_lines, bool) or self.max_lines not in range(1, 6):
            raise ValueError("max_lines must be an integer in [1, 5]")
        if (
            isinstance(self.reference_height, bool)
            or not isinstance(self.reference_height, int)
            or self.reference_height <= 0
        ):
            raise ValueError("reference_height must be a positive integer")

    def to_dict(self):
        """Return a JSON-compatible dictionary (colors as RGB lists)."""
        return {
            k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, value):
        """Build a style from ``to_dict`` output, rejecting unknown keys."""
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown subtitle style keys: {sorted(unknown)}")
        return cls(**value)


@dataclass(frozen=True)
class ChannelPreset:
    """Canvas, timing, palette, fonts and subtitle defaults for one channel.

    Pixel quantities (``safe_margin``, subtitle sizes) are authored at
    ``SubtitleStyle.reference_height`` and scaled with ``scaled_px`` for other
    output sizes. ``fonts`` maps a role (``title``, ``body``, ``quote``) to a
    font file path; missing files are reported by templates, not silently
    replaced. ``palette`` maps role names to RGB codes.
    """

    name: str
    size: tuple = (1920, 1080)
    fps: float = 24.0
    safe_margin: tuple = (128, 90)
    overlay_color: tuple = (0, 0, 0)
    overlay_opacity: float = 0.5
    fade: float = 0.5
    hold: float = 1.5
    working_space: str = "srgb"
    palette: dict = field(default_factory=dict)
    fonts: dict = field(default_factory=dict)
    subtitles: SubtitleStyle = field(default_factory=SubtitleStyle)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        size = tuple(self.size)
        if len(size) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in size
        ):
            raise ValueError("size must be two positive integers")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "fps", _number(self.fps, "fps", 1, 240))
        margin = tuple(self.safe_margin)
        if len(margin) != 2:
            raise ValueError("safe_margin must be (x, y) pixels")
        margin = tuple(
            _number(v, "safe_margin", 0, s / 2) for v, s in zip(margin, size)
        )
        object.__setattr__(self, "safe_margin", margin)
        object.__setattr__(
            self, "overlay_color", _color(self.overlay_color, "overlay_color")
        )
        object.__setattr__(
            self,
            "overlay_opacity",
            _number(self.overlay_opacity, "overlay_opacity", 0, 1),
        )
        object.__setattr__(self, "fade", _number(self.fade, "fade", 0, 60))
        object.__setattr__(self, "hold", _number(self.hold, "hold", 0, 600))
        if self.working_space not in ("srgb", "linear"):
            raise ValueError("working_space must be 'srgb' or 'linear'")
        palette = {
            str(k): _color(v, f"palette[{k!r}]") for k, v in dict(self.palette).items()
        }
        object.__setattr__(self, "palette", palette)
        fonts = {str(k): str(v) for k, v in dict(self.fonts).items()}
        object.__setattr__(self, "fonts", fonts)
        subtitles = self.subtitles
        if isinstance(subtitles, dict):
            subtitles = SubtitleStyle(**subtitles)
        if not isinstance(subtitles, SubtitleStyle):
            raise TypeError("subtitles must be a SubtitleStyle or dict")
        object.__setattr__(self, "subtitles", subtitles)

    @property
    def scale(self):
        """Pixel scale of this canvas relative to the subtitle reference."""
        return self.size[1] / self.subtitles.reference_height

    def scaled_px(self, value, size=None):
        """Scale a reference-height pixel quantity to ``size`` (or own size)."""
        height = (size or self.size)[1]
        return float(value) * height / self.subtitles.reference_height

    def color(self, role, default=(255, 255, 255)):
        """Return palette RGB codes for ``role`` or ``default``."""
        return self.palette.get(role, _color(default, "default"))

    def paper_ink(self, paper=None, default=(30, 26, 22)):
        """Return the ink colour for text printed on paper (tags, scrolls).

        Uses the ``paper_ink`` role when set, else ``ink`` when it reads on
        ``paper`` (WCAG contrast at least 4.5), else ``default``. A channel
        whose ``ink`` is a light colour meant for footage would otherwise
        print almost invisible text on its paper strips.

        >>> get_preset("nightlamp_story").paper_ink()
        (30, 26, 22)
        >>> get_preset("nightlamp_history").paper_ink() == get_preset(
        ...     "nightlamp_history").color("ink")
        True
        """
        if "paper_ink" in self.palette:
            return self.palette["paper_ink"]
        paper = paper or self.color("paper", (239, 228, 204))
        ink = self.palette.get("ink")
        if ink is not None and _contrast(ink, paper) >= 4.5:
            return ink
        return _color(default, "default")

    def font(self, role, explicit=None):
        """Resolve a font path for ``role``; raise if it does not exist.

        ``explicit`` overrides the preset. Templates never fall back to a
        different typeface silently, because CJK glyph coverage differs.
        """
        path = explicit or self.fonts.get(role)
        if path is None:
            raise FileNotFoundError(f"preset {self.name!r} has no {role!r} font")
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{role} font not found: {path}")
        return str(path)

    def with_overrides(self, **changes):
        """Return a validated copy with ``changes`` applied."""
        if "size" in changes and "safe_margin" not in changes:
            # Margins are authored for the current canvas; keep them proportional.
            factor = tuple(changes["size"])[1] / self.size[1]
            changes["safe_margin"] = tuple(m * factor for m in self.safe_margin)
        if "subtitles" in changes and isinstance(changes["subtitles"], dict):
            changes["subtitles"] = replace(self.subtitles, **changes["subtitles"])
        return replace(self, **changes)

    def to_dict(self):
        """Return a JSON-compatible dictionary."""
        value = asdict(self)
        value["size"] = list(self.size)
        value["safe_margin"] = list(self.safe_margin)
        value["overlay_color"] = list(self.overlay_color)
        value["palette"] = {k: list(v) for k, v in self.palette.items()}
        value["subtitles"] = self.subtitles.to_dict()
        return value

    @classmethod
    def from_dict(cls, value):
        """Build a preset from ``to_dict`` output, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown preset keys: {sorted(unknown)}")
        return cls(**value)

    def to_json(self, path=None):
        """Serialize to JSON text, also writing UTF-8 ``path`` when given."""
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source):
        """Load from a JSON path or JSON text."""
        if isinstance(source, Path) or (
            isinstance(source, str) and not source.lstrip().startswith("{")
        ):
            source = Path(source).read_text(encoding="utf-8-sig")
        return cls.from_dict(json.loads(source))


_WINDOWS_FONTS = Path("C:/Windows/Fonts")


def _font_dirs():
    """System and per-user font folders searched for preset fonts."""
    dirs = [_WINDOWS_FONTS]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    home = Path.home()
    dirs += [
        home / ".local" / "share" / "fonts",
        home / ".fonts",
        home / "Library" / "Fonts",
    ]
    return dirs


def source_han_font(style="sans", weight="Bold"):
    """Return the path of an installed Source Han (思源) Traditional Chinese font.

    Looks for the Traditional Chinese (``TC``, full glyph set) and then the
    Taiwan subset (``TW``) static OpenType files of Source Han Sans (思源黑體, ``style="sans"``) or Source
    Han Serif (思源宋體, ``style="serif"``) in the Windows system and
    per-user font folders and the usual user font folders elsewhere. When
    none is installed the expected per-user path is returned anyway, so
    ``ChannelPreset.font`` reports exactly which file to install; no other
    typeface is substituted. A browser's duplicate-download name such as
    ``SourceHanSerifTW-Bold (1).otf`` is accepted when the plain name is
    absent, since Windows installs the file under the name it was given.

    Parameters
    ----------
    style : {"sans", "serif"}, optional
        Source Han Sans or Source Han Serif.
    weight : str, optional
        Weight in the file name, for example ``"Regular"``, ``"Medium"``,
        ``"SemiBold"`` (serif only) or ``"Bold"``.

    Examples
    --------
    >>> Path(source_han_font("serif", "Bold")).name.startswith(
    ...     ("SourceHanSerifTC-Bold", "SourceHanSerifTW-Bold"))
    True
    """
    if style not in ("sans", "serif"):
        raise ValueError("style must be 'sans' or 'serif'")
    family = "Sans" if style == "sans" else "Serif"
    names = [f"SourceHan{family}{region}-{weight}.otf" for region in ("TC", "TW")]
    dirs = _font_dirs()
    # The full TC set wins over the TW subset wherever each is installed.
    for name in names:
        for folder in dirs:
            if (folder / name).is_file():
                return str(folder / name)
        stem = name[: -len(".otf")]
        for folder in dirs:
            copies = sorted(folder.glob(f"{stem} (*).otf")) if folder.is_dir() else []
            for path in copies:
                if re.fullmatch(re.escape(stem) + r" \(\d+\)\.otf", path.name):
                    return str(path)
    return str(dirs[1 if len(dirs) > 1 else 0] / names[0])


PRESETS = {
    # night-lantern-workflow-r23 / night-lamp-story-title v5: episode media
    # under a 50 % black overlay, warm-white type, 24 fps template previews.
    "nightlamp_story": ChannelPreset(
        name="nightlamp_story",
        size=(1920, 1080),
        fps=24,
        safe_margin=(128, 90),
        overlay_color=(0, 0, 0),
        overlay_opacity=0.5,
        fade=0.6,
        hold=1.5,
        palette={
            "ink": "#F3ECDC",
            "secondary_ink": "#E8DFCC",
            "accent": "#D6A391",
            "lamp": "#D9A45F",
            "panel": "#493C2D",
        },
        # 思源宋體 for titles, 思源黑體 for body text and burned subtitles.
        fonts={
            "title": source_han_font("serif", "Bold"),
            "body": source_han_font("sans", "Bold"),
        },
        subtitles=SubtitleStyle(
            primary_size=72,
            secondary_size=42,
            plate_opacity=0.05,
            plate_padding=0,
            highlight_color=(0xC9, 0xA3, 0x5D),
        ),
    ),
    # nightlamp-history / nlh-motion: rice paper, dense ink and vermilion
    # seal; zh-TW 72 px plus English 42 px burned in at 1080p.
    "nightlamp_history": ChannelPreset(
        name="nightlamp_history",
        size=(1920, 1080),
        fps=24,
        safe_margin=(96, 72),
        overlay_color=(0, 0, 0),
        overlay_opacity=0.0,
        fade=0.5,
        hold=2.0,
        palette={
            "paper": "#EFE4CC",
            "paper_shadow": "#C9B48C",
            "ink": "#1E1A16",
            # r2b title over footage: lamp gold over warm white.
            "card_ink": "#EED6A4",
            "card_secondary_ink": "#F8EED5",
            "seal": "#B01E1C",
            "bamboo": "#C8A86A",
            "bamboo_dark": "#8A6A34",
        },
        fonts={
            "title": source_han_font("serif", "Bold"),
            "body": source_han_font("sans", "Bold"),
            "quote": source_han_font("serif", "SemiBold"),
        },
        subtitles=SubtitleStyle(
            primary_size=72,
            secondary_size=42,
            plate_opacity=0.05,
            plate_padding=0,
            highlight_color=(0xC9, 0xA3, 0x5D),
        ),
    ),
}


def get_preset(name, **overrides):
    """Return a named preset, optionally with validated ``overrides``."""
    try:
        preset = PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return preset.with_overrides(**overrides) if overrides else preset
