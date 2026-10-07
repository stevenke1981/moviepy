"""Editable SVG composition and cache comparison using the optional svg extra.

Run from the repository root::

    python -m examples.ae_svg --output svg.mp4
    python -m examples.ae_svg --benchmark 30
    python -m examples.ae_svg --font /path/to/font.ttf --text "MoviePy AE" --output title.mp4
"""

import argparse
import json
import platform
import statistics
import time

import numpy as np

from moviepy.ae import (
    Composition,
    Keyframe,
    Mask,
    Property,
    RenderContext,
    Transform,
    fx,
)


ARTWORK = """<svg viewBox="0 0 512 256">
  <defs><linearGradient id="accent" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#38d9a9"/>
    <stop offset="1" stop-color="#4263eb"/>
  </linearGradient></defs>
  <rect x="8" y="8" width="496" height="240" rx="28" fill="#172339"/>
  <circle id="orb" cx="100" cy="112" r="44" fill="url(#accent)"/>
  <path d="M 172 80 L 390 80 L 350 140 L 172 140 Z" fill="#364663"/>
  <rect x="48" y="194" width="416" height="12" rx="6" fill="#364663"/>
  <rect id="progress" x="48" y="194" width="24" height="12" rx="6" fill="#38d9a9"/>
</svg>"""


def build_scene(font=None, text="MoviePy AE", *, cache_entries=8):
    """Create a normal Composition whose SVG attributes remain editable."""
    comp = Composition(
        size=(640, 360),
        fps=24,
        duration=3,
        context=RenderContext(working_space="linear"),
    )
    comp.add_solid("background", color=(9, 14, 25))
    svg = ARTWORK
    if font:
        svg = svg.replace(
            "</svg>",
            '<text id="label" x="172" y="168" font-size="24" fill="white">SVG</text></svg>',
        )
    layer = comp.add_svg(
        svg,
        "Editable card",
        size=(512, 256),
        font_files=[font] if font else [],
        cache_entries=cache_entries,
    )
    layer.bind("progress", "width", Property(24, keyframes=[(0, 24), (2.5, 416)]))
    layer.bind("orb", "r", Property(36, keyframes=[(0, 36), (1.5, 52), (3, 36)]))
    if font:
        layer.bind(
            "label",
            "text",
            Property(
                "SVG",
                keyframes=[
                    Keyframe(0, "SVG", interp="hold"),
                    Keyframe(0.6, text, interp="hold"),
                ],
            ),
        )
    layer.transform = Transform(
        position=Property((64, 80), keyframes=[(0, (64, 80)), (0.6, (64, 52))]),
        anchor_point=(0, 0),
    )
    layer.masks = [
        Mask.rect(
            center=(255.5, 127.5),
            size=(512, 256),
            opacity=Property(0, keyframes=[(0, 0), (0.4, 100)]),
        )
    ]
    layer.effects = [fx.GaussianBlur(blurriness=0.6)]
    return comp, layer


def compare_cache(samples=30, *, font=None, text="MoviePy AE"):
    """Measure warm/cold equivalent frames and distinct animated source frames."""
    if samples < 3:
        raise ValueError("benchmark needs at least three samples")
    cached, cached_layer = build_scene(font, text)
    uncached, uncached_layer = build_scene(font, text, cache_entries=0)
    np.testing.assert_array_equal(cached.get_frame(1), uncached.get_frame(1))

    def measure(call):
        values = []
        for index in range(samples):
            start = time.perf_counter()
            call(index)
            values.append((time.perf_counter() - start) * 1000)
        return statistics.median(values)

    # Alternate paired measurements so one scene does not own the warm-up phase.
    paired = {"cached_composition_ms": [], "uncached_composition_ms": []}
    for index in range(samples):
        order = [
            ("cached_composition_ms", cached),
            ("uncached_composition_ms", uncached),
        ]
        for name, comp in order[:: 1 if index % 2 == 0 else -1]:
            start = time.perf_counter()
            comp.get_frame(1)
            paired[name].append((time.perf_counter() - start) * 1000)
    result = {key: statistics.median(values) for key, values in paired.items()}
    result["cached_source_ms"] = measure(lambda i: cached_layer.source_buffer(1))
    result["uncached_source_ms"] = measure(lambda i: uncached_layer.source_buffer(1))
    result["animated_source_ms"] = measure(
        lambda i: cached_layer.source_buffer(i / samples * 2.5)
    )
    result["composition_speedup"] = (
        result["uncached_composition_ms"] / result["cached_composition_ms"]
    )
    result["source_speedup"] = result["uncached_source_ms"] / result["cached_source_ms"]
    result.update(
        {
            "samples": samples,
            "size": [640, 360],
            "raster_size": [512, 256],
            "python": platform.python_version(),
            "platform": platform.platform(),
            "explicit_font": font is not None,
            "cache": cached_layer.cache_info,
        }
    )
    return result


def main():
    """Render the requested output or print measured cache comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument(
        "--font", help="Explicit local font containing the requested text"
    )
    parser.add_argument("--text", default="MoviePy AE")
    parser.add_argument("--benchmark", type=int, default=0, metavar="SAMPLES")
    args = parser.parse_args()
    if not args.output and not args.benchmark:
        parser.error("provide --output or --benchmark")
    if args.benchmark:
        print(
            json.dumps(
                compare_cache(args.benchmark, font=args.font, text=args.text), indent=2
            )
        )
    if args.output:
        comp, _ = build_scene(args.font, args.text)
        comp.write_videofile(args.output, codec="libx264", audio=False)


if __name__ == "__main__":
    main()
