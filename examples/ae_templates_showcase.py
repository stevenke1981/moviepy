"""Still-frame showcase for ``moviepy.ae.templates``.

Renders one PNG per template from synthetic media (a seeded gradient, so no
external footage is needed) and writes ``report.json`` with per-step timings::

    python -m examples.ae_templates_showcase OUTPUT_DIR [--size 960x540]

Outputs: ``title_card.png`` (over a media background), ``ken_burns_first.png``,
``ken_burns_last.png``, ``chapter_tag.png``, ``quote_bamboo.png``,
``quote_paper.png`` and ``subtitle.png``.

The presets use ``kaiu.ttf`` (``C:/Windows/Fonts``). When it is missing the
script prints the path it needs and exits with status 2; it never swaps in
another typeface. ``OUTPUT_DIR`` must be absent or empty, so earlier results
are never overwritten.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from moviepy.ae.buffer import Buffer
from moviepy.ae.templates.background import media_background
from moviepy.ae.templates.chapter_tag import chapter_tag
from moviepy.ae.templates.ken_burns import ken_burns
from moviepy.ae.templates.presets import get_preset
from moviepy.ae.templates.quote import vertical_quote
from moviepy.ae.templates.subtitles import Cue, burn_subtitles
from moviepy.ae.templates.title_card import TitleCardSpec, build_title_card

EXIT_MISSING_FONT = 2


def parse_size(text):
    """Parse ``WIDTHxHEIGHT`` into a tuple of positive ints."""
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("size must look like 960x540") from error
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size must be positive")
    return width, height


def synthetic_media(size, seed=7):
    """Return a deterministic RGB gradient with a little seeded grain."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            xx * 255 // max(width - 1, 1),
            yy * 255 // max(height - 1, 1),
            np.full_like(xx, 128),
        ],
        axis=-1,
    ).astype(np.int32)
    grain = np.random.default_rng(seed).integers(-6, 7, rgb.shape)
    return np.clip(rgb + grain, 0, 255).astype(np.uint8)


def save_frame(buffer, path, bg=(0, 0, 0)):
    """Write a rendered Buffer as an opaque RGB PNG."""
    Image.fromarray(buffer.to_uint8_rgb(bg=bg)).save(path)


def run(out_dir, size):
    """Render every still into ``out_dir`` and return the report dictionary."""
    story = get_preset("nightlamp_story").with_overrides(size=size)
    history = get_preset("nightlamp_history").with_overrides(size=size)

    fonts = {}
    for preset, roles in ((story, ("title", "body")), (history, ("quote", "title"))):
        for role in roles:
            fonts[f"{preset.name}.{role}"] = preset.font(role)

    media = synthetic_media(size)
    report = {
        "size": list(size),
        "presets": [story.name, history.name],
        "fonts": fonts,
        "steps": [],
    }

    def timed(name, file_name, action):
        started = time.perf_counter()
        result = action()
        seconds = time.perf_counter() - started
        report["steps"].append(
            {"name": name, "file": file_name, "seconds": round(seconds, 4)}
        )
        return result

    # 1. Title card over the media background (background is required media).
    background = timed(
        "media_background",
        None,
        lambda: media_background(media, story, duration=6.0),
    )
    title = timed(
        "build_title_card",
        "title_card.png",
        lambda: build_title_card(
            TitleCardSpec(
                title="夜燈說書",
                brand="Night Lamp Story",
                subtitle="一盞燈，照見舊書裡的人。",
            ),
            story,
            background=background,
        ),
    )
    timed(
        "render_title_card",
        "title_card.png",
        lambda: save_frame(title.render_buffer(2.0), out_dir / "title_card.png"),
    )

    # 2. Ken Burns: first and last frame of one push move.
    kb_duration = 4.0
    motion = timed(
        "ken_burns",
        None,
        lambda: ken_burns(media, story, duration=kb_duration, move="push"),
    )
    last_t = kb_duration - 1.0 / story.fps
    timed(
        "render_ken_burns",
        "ken_burns_first.png",
        lambda: save_frame(motion.render_buffer(0.0), out_dir / "ken_burns_first.png"),
    )
    timed(
        "render_ken_burns",
        "ken_burns_last.png",
        lambda: save_frame(
            motion.render_buffer(last_t), out_dir / "ken_burns_last.png"
        ),
    )

    # 3. Chapter tag composited over the Ken Burns frame at its own position.
    base = motion.render_buffer(1.0).to_uint8_rgb(bg=(0, 0, 0))

    def chapter_frame():
        tag = chapter_tag(["城守住了|本章人物：于謙"], 1, 4.0, preset=history, size=44)
        layer = tag.composition.render_buffer(1.0).with_offset(
            (int(tag.meta["x"]), int(tag.meta["y"]))
        )
        composed = layer.composite_over(Buffer.from_uint8_rgb(base))
        Image.fromarray(composed.to_uint8_rgb()).save(out_dir / "chapter_tag.png")

    timed("chapter_tag", "chapter_tag.png", chapter_frame)

    # 4. Bamboo slips (medium forced) and paper handscroll.
    quote_text = "學而時習之，不亦說乎。|有朋自遠方來，不亦樂乎。"
    for medium, file_name in (
        ("bamboo", "quote_bamboo.png"),
        ("paper", "quote_paper.png"),
    ):
        scroll = timed(
            f"vertical_quote[{medium}]",
            None,
            lambda m=medium: vertical_quote(
                quote_text,
                "《論語·學而》",
                preset=history,
                medium=m,
                duration=6.0,
                size=62,
                per_column=12,
            ),
        )
        timed(
            f"render_quote[{medium}]",
            file_name,
            lambda s=scroll, f=file_name: save_frame(s.render_buffer(5.0), out_dir / f),
        )

    # 5. Bilingual burned-in subtitle over the Ken Burns frame.
    zh = [Cue(0.0, 3.0, "夜燈照著舊書頁。", "zh-TW")]
    en = [Cue(0.0, 3.0, "The lamp lights old pages.", "en")]
    subtitled = timed(
        "burn_subtitles",
        None,
        lambda: burn_subtitles(
            ken_burns(media, story, duration=4.0, move="push"),
            zh,
            story,
            secondary=en,
        ),
    )
    timed(
        "render_subtitles",
        "subtitle.png",
        lambda: save_frame(subtitled.render_buffer(1.5), out_dir / "subtitle.png"),
    )

    report["total_seconds"] = round(sum(s["seconds"] for s in report["steps"]), 4)
    report["outputs"] = sorted(
        {step["file"] for step in report["steps"] if step["file"]}
    )
    return report


def main(argv=None):
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output_dir", type=Path, help="new or empty directory")
    parser.add_argument(
        "--size", type=parse_size, default=(960, 540), help="WIDTHxHEIGHT"
    )
    args = parser.parse_args(argv)

    out_dir = args.output_dir
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        print(
            f"refusing to write: {out_dir} is not an empty directory", file=sys.stderr
        )
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        report = run(out_dir, args.size)
    except FileNotFoundError as error:
        print(
            f"missing font: {error}. Install kaiu.ttf (Windows: C:/Windows/Fonts) "
            "or edit the preset fonts; no fallback font is used.",
            file=sys.stderr,
        )
        return EXIT_MISSING_FONT

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(report['outputs'])} stills and report.json to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
