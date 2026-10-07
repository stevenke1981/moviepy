"""Optional pinned resvg adapter with explicit, snapshotted font resources."""

import hashlib
import importlib
import io
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

from moviepy.ae.buffer import Buffer


RESVG_VERSION = "0.2.6"
FONTTOOLS_VERSION = "4.59.2"
MAX_FONT_BYTES = 32 * 1024 * 1024


def read_local_file(path, budget):
    """Read one explicitly selected regular file, with no URI/UNC resolution."""
    value = str(path)
    if "://" in value or value.startswith(("\\\\", "//")):
        raise ValueError("SVG inputs and fonts must be explicit local files")
    selected = Path(path).expanduser().resolve()
    if str(selected).startswith(("\\\\", "//")):
        raise ValueError("SVG inputs and fonts must be explicit local files")
    if not selected.is_file():
        raise ValueError(f"SVG input file does not exist: {selected}")
    with selected.open("rb") as stream:
        content = stream.read(budget + 1)
    if len(content) > budget:
        raise ValueError("SVG input file exceeds its byte budget")
    return content


def font_snapshots(paths):
    """Hash the exact font bytes on every request, including same-mtime edits."""
    result = []
    total = 0
    for path in paths:
        data = read_local_file(path, MAX_FONT_BYTES)
        total += len(data)
        if total > 2 * MAX_FONT_BYTES:
            raise ValueError("SVG fonts exceed the combined 64 MiB budget")
        result.append((hashlib.sha256(data).hexdigest(), data))
    return tuple(result)


def _optional_module(name, expected):
    try:
        module = importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(
            "SVG rendering requires optional dependencies: "
            f"pip install resvg-py=={RESVG_VERSION} fonttools=={FONTTOOLS_VERSION}"
        ) from error
    if module.__version__ != expected:
        raise RuntimeError(
            f"SVG requires {name}=={expected}; found {module.__version__}"
        )
    return module


def font_faces(snapshots):
    """Read Unicode coverage and family names from supplied TTF/OTF/TTC bytes."""
    if not snapshots:
        return ()
    _optional_module("fontTools", FONTTOOLS_VERSION)
    from fontTools.ttLib import TTCollection, TTFont

    faces = []
    for _, data in snapshots:
        if data[:4] not in {b"ttcf", b"OTTO", b"\x00\x01\x00\x00", b"true"}:
            raise ValueError("SVG fonts must be uncompressed TTF, OTF or TTC files")
        stream = io.BytesIO(data)
        container = None
        fonts = []
        try:
            if data[:4] == b"ttcf":
                container = TTCollection(stream, lazy=True)
                fonts = container.fonts
            else:
                fonts = [TTFont(stream, lazy=True)]
            if len(fonts) > 32:
                raise ValueError("SVG font collection exceeds 32 faces")
            for font in fonts:
                names = tuple(
                    dict.fromkeys(
                        item.toUnicode()
                        for item in font["name"].names
                        if item.nameID in {1, 16}
                    )
                )
                coverage = frozenset(
                    code
                    for code, glyph in (font.getBestCmap() or {}).items()
                    if glyph != ".notdef"
                )
                if names and coverage:
                    faces.append((names, coverage))
        except Exception as error:
            raise ValueError(f"invalid SVG font: {error}") from error
        finally:
            if container is not None:
                container.close()
            else:
                for font in fonts:
                    font.close()
    if not faces:
        raise ValueError("SVG fonts contain no usable Unicode faces")
    return tuple(faces)


def validate_text(root, faces):
    """Fail on absent families/glyphs instead of silently dropping SVG text."""
    default = faces[0][0][0] if faces else None
    generic = {"serif", "sans-serif", "monospace", "cursive", "fantasy"}

    def check(text, family):
        if not text or not text.strip():
            return
        if not faces:
            raise ValueError(
                "SVG text requires explicit font_files, including Latin/CJK text"
            )
        names = [part.strip().strip("'\"") for part in (family or default).split(",")]
        names = [default if name in generic else name for name in names]
        candidates = [
            face
            for face in faces
            if any(name.casefold() in {n.casefold() for n in face[0]} for name in names)
        ]
        if not candidates:
            raise ValueError(f"SVG font family is unavailable in font_files: {family}")
        # This is a cmap preflight, not a guarantee about ligatures or shaping.
        missing = {
            ord(char)
            for char in text
            if not char.isspace()
            and unicodedata.category(char) != "Cf"
            and not (0xFE00 <= ord(char) <= 0xFE0F or 0xE0100 <= ord(char) <= 0xE01EF)
            and not any(ord(char) in coverage for _, coverage in candidates)
        }
        if missing:
            codes = ", ".join(f"U+{code:04X}" for code in sorted(missing)[:8])
            raise ValueError(f"SVG font is missing glyphs: {codes}")

    def visit(node, family=None, in_text=False):
        family = node.get("font-family", family)
        in_text = in_text or node.tag == "text"
        if in_text:
            check(node.text, family)
        for child in node:
            visit(child, family, in_text)
            if in_text:
                check(child.tail, family)

    visit(root)
    return default


def rasterize(svg, size, snapshots, family):
    """Decode resvg's straight-alpha PNG and premultiply exactly once."""
    module = _optional_module("resvg_py", RESVG_VERSION)
    options = {
        "svg_string": svg,
        "skip_system_fonts": True,
        "font_dirs": [],
        "resources_dir": None,
        "dpi": 96,
        "log_information": False,
    }
    if family is not None:
        for key in (
            "font_family",
            "serif_family",
            "sans_serif_family",
            "cursive_family",
            "fantasy_family",
            "monospace_family",
        ):
            options[key] = family
    # Passing our byte snapshots avoids a file edit between hashing and resvg.
    # TemporaryDirectory closes files before the native Windows reader opens them.
    with tempfile.TemporaryDirectory(prefix="moviepy-svg-") as directory:
        paths = []
        for index, (_, data) in enumerate(snapshots):
            suffix = ".ttc" if data[:4] == b"ttcf" else ".otf"
            path = Path(directory) / f"font-{index}{suffix}"
            path.write_bytes(data)
            paths.append(str(path))
        options["font_files"] = paths
        try:
            encoded = module.svg_to_bytes(**options)
        except ValueError as error:
            raise ValueError(f"resvg could not render SVG: {error}") from error
    with Image.open(io.BytesIO(encoded)) as image:
        if image.format != "PNG" or image.size != size:
            raise ValueError("resvg returned an unexpected image format or size")
        rgba = np.asarray(image.convert("RGBA"))
    # tiny-skia uses premultiplication internally; its encoded PNG is straight.
    # The source stays SDR sRGB. Layer.prepared_source performs working-space
    # conversion before masks/effects without changing other layers' HDR range.
    return Buffer.from_uint8_rgb(rgba[..., :3], rgba[..., 3].astype(np.float32) / 255)
