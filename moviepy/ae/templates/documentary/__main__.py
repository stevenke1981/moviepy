"""CLI for reference-driven documentary effects; all demo media is procedural."""

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from moviepy.ae.templates.documentary import (
    add_documentary_subtitles,
    build_film_frame,
    build_news_collage,
    build_news_page,
)
from moviepy.ae.templates.documentary._art import _font, demo_letter, font_path
from moviepy.ae.templates.subtitles import Cue
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import ImageClip, VideoClip


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
_CONFIG_KEYS = {
    "input",
    "clippings",
    "main_is_page",
    "headline",
    "body",
    "source_label",
    "watermark",
    "font",
    "news",
    "film",
    "subtitles_primary",
    "subtitles_secondary",
}


def _read_media(path, stack):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"media not found: {path}")
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        with Image.open(path) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("animated images are not stills; use a video input")
            array = np.asarray(ImageOps.exif_transpose(image).convert("RGBA"))
        return ImageClip(array)
    clip = VideoFileClip(str(path))
    stack.callback(clip.close)
    return clip


def _demo_source(font, duration, fps, chinese):
    base = demo_letter((1280, 720), font, chinese=chinese)
    yy, xx = np.ogrid[:720, :1280]

    def frame(t):
        # A moving window shadow demonstrates LIVE footage inside both templates.
        position = 150 + 750 * float(t) / duration
        distance = xx - position - yy * 0.23
        shadow = 0.25 * np.exp(-np.square(distance / 95))
        shadow += 0.12 * np.exp(-np.square((distance - 360) / 65))
        return np.clip(base * (1 - shadow[..., None]), 0, 255).astype(np.uint8)

    return VideoClip(frame_function=frame, duration=duration).with_fps(fps)


def _demo_cards(source, font, chinese):
    titles = (
        (
            "時間留下的線索",
            "一段被重新讀起的往事",
            "那些未曾遺忘的日子",
            "在記憶裡，再次相逢",
        )
        if chinese
        else (
            "TRACES OF TIME",
            "A STORY REVISITED",
            "DAYS TO REMEMBER",
            "MEMORIES RETURN",
        )
    )
    cards = []
    for i, title in enumerate(titles):
        w, h = (580, 800) if i % 2 else (740, 600)
        image = Image.new("RGB", (w, h), (242, 240, 235))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, w, 63), fill=(181 + i * 7, 33, 46))
        draw.text((24, 18), "ARCHIVE / NEWS", font=_font(font, 23), fill="white")
        draw.text((25, 90), title, font=_font(font, 27), fill=(31, 33, 38))
        media = Image.fromarray(source.get_frame(0.3 + i * 0.2))
        media = ImageOps.fit(media, (w - 50, int(h * 0.53)))
        image.paste(media, (25, 152))
        draw.text((26, h - 99), "1961", font=_font(font, 65), fill=(147, 45, 47))
        cards.append(ImageClip(np.asarray(image)))
    return cards


def _load_config(path):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    unknown = set(config) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if not isinstance(config.get("input"), str) or not config["input"]:
        raise ValueError("config needs a non-empty input path")
    if not isinstance(config.get("main_is_page", False), bool):
        raise ValueError("main_is_page must be a boolean")
    for name in ("news", "film"):
        if not isinstance(config.get(name, {}), dict):
            raise ValueError(f"{name} must be an object of template options")
    for name in ("clippings", "subtitles_primary", "subtitles_secondary"):
        if not isinstance(config.get(name, []), list):
            raise ValueError(f"{name} must be an array")
    return config


def _parser():
    parser = argparse.ArgumentParser(
        description="MoviePy AE documentary reference effects"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--demo", action="store_true", help="use original procedural demo media"
    )
    inputs.add_argument("--input", type=Path, help="video or still to frame")
    inputs.add_argument(
        "--config", type=Path, help="UTF-8 JSON; paths relative to this file"
    )
    parser.add_argument("--clipping", type=Path, action="append", default=[])
    parser.add_argument(
        "--main-is-page",
        action="store_true",
        help="do not wrap input in another news page",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("documentary-previews"))
    parser.add_argument("--effect", choices=("both", "news", "film"), default="both")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--duration", type=float, default=6)
    parser.add_argument("--fps", type=float, default=24)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--language", choices=("en", "zh-TW"), default="en")
    parser.add_argument("--headline", default="A STORY ACROSS HALF A CENTURY")
    parser.add_argument("--source-label", default="")
    parser.add_argument("--watermark", default="")
    return parser


def main(argv=None):
    """Render both effects and still previews; return 2 for invalid inputs."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.width < 320 or args.width > 7680 or args.width % 2:
            raise ValueError("width must be an even integer between 320 and 7680")
        if not math.isfinite(args.duration) or not 0 < args.duration <= 86400:
            raise ValueError("duration must be positive, finite, and <= 86400 seconds")
        if not math.isfinite(args.fps) or not 1 <= args.fps <= 240:
            raise ValueError("fps must be between 1 and 240")
        return _render(args)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


def _render(args):
    config = _load_config(args.config) if args.config else {}
    base = args.config.resolve().parent if args.config else Path.cwd()
    chinese = args.language == "zh-TW"
    explicit_font = args.font or (base / config["font"] if config.get("font") else None)
    configured_text = "".join(
        str(config.get(key, ""))
        for key in ("headline", "body", "source_label", "watermark")
    )
    font = font_path(
        explicit_font, cjk=chinese or any(ord(c) > 127 for c in configured_text)
    )
    size = (args.width, max(2, round(args.width * 9 / 16 / 2) * 2))
    common = {"size": size, "duration": args.duration, "fps": args.fps}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        if args.demo:
            source = _demo_source(font, args.duration, args.fps, chinese)
            clips = _demo_cards(source, font, chinese)
            headline = (
                "跨越半個世紀，記憶仍留在信裡"
                if chinese
                else "A STORY ACROSS HALF A CENTURY"
            )
            body = (
                "一封泛黃的信，讓遠方的往事重新浮現。\n紙上的字跡、熟悉的日期，和那些未曾遺忘的人。\n時間走過半個世紀，故事仍在繼續。"
                if chinese
                else "An old letter brings a distant story back into view.\nHandwriting, a familiar date and people never forgotten.\nHalf a century later, the story continues."
            )
            source_label = (
                "來源：原創示範素材" if chinese else "Source: original demo artwork"
            )
            watermark = "1961"
        else:
            source = _read_media(
                base / config["input"] if config else args.input, stack
            )
            paths = [
                base / path for path in config.get("clippings", [])
            ] + args.clipping
            clips = [_read_media(path, stack) for path in paths]
            headline = config.get("headline", args.headline)
            body = config.get("body", "")
            source_label = config.get("source_label", args.source_label)
            watermark = config.get("watermark", args.watermark)
        comps = {}
        if args.effect in ("both", "news"):
            if args.main_is_page or config.get("main_is_page", False):
                page = source
            else:
                page = build_news_page(
                    source,
                    headline=headline,
                    body=body,
                    source_label=source_label,
                    duration=args.duration,
                    fps=args.fps,
                    font=font,
                )
            comps["news_collage"] = build_news_collage(
                page, clips, **common, **config.get("news", {})
            )
        if args.effect in ("both", "film"):
            comps["film_frame"] = build_film_frame(
                source,
                source_label=source_label,
                watermark=watermark,
                font=font,
                **common,
                **config.get("film", {}),
            )
        primary = [Cue(**cue) for cue in config.get("subtitles_primary", [])]
        secondary = [Cue(**cue) for cue in config.get("subtitles_secondary", [])]
        if args.demo and font:
            primary = [
                Cue(
                    0,
                    args.duration,
                    (
                        "一封信，讓遠方的故事再次被讀起。"
                        if chinese
                        else "A letter brings the story back."
                    ),
                    "zh-TW" if chinese else "en",
                )
            ]
            secondary = (
                [Cue(0, args.duration, "A letter brings a distant story back.", "en")]
                if chinese
                else []
            )
        if any(c.end > args.duration for c in primary + secondary):
            raise ValueError("subtitle end must not exceed output duration")
        report = {
            "version": "1.0.0",
            "size": size,
            "fps": args.fps,
            "duration": args.duration,
            "demo": args.demo,
            "font_path": font,
            "outputs": {},
        }
        for name, comp in comps.items():
            if primary or secondary:
                if font is None:
                    raise ValueError("subtitles require --font")
                add_documentary_subtitles(comp, primary, secondary=secondary, font=font)
            output = args.output_dir / f"{name}.mp4"
            comp.write_videofile(
                str(output),
                fps=args.fps,
                codec="libx264",
                audio_codec="aac",
                preset="medium",
                threads=2,
                ffmpeg_params=[
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ],
                logger=None,
            )
            t = min(args.duration * 0.65, max(0, args.duration - 1 / args.fps))
            comp.save_frame(str(args.output_dir / f"{name}.png"), t=t)
            report["outputs"][name] = {
                "video": output.name,
                "preview": f"{name}.png",
                "layers": [layer.name for layer in comp.layers],
                "metadata": comp.documentary_metadata,
            }
            print(f"Rendered {output} ({size[0]}x{size[1]}, {args.fps:g} fps)")
        (args.output_dir / "render-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
