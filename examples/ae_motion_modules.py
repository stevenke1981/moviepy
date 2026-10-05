"""Render eight seconds of original captions, beat particles and transitions.

Run: python -m examples.ae_motion_modules OUTPUT --godot PATH --font FONT
Use --preview for 640x360 at 12 fps; the default is 1280x720 at 24 fps.
No files, models, fonts or music are downloaded.
"""

import argparse
import json
import math
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy import AudioFileClip, CompositeVideoClip, VideoClip, vfx
from moviepy.ae.motion import analyze_audio, render_motion_effect, transition_clip


def make_rhythm(path, *, duration=8, rate=48000, seed=605):
    """Synthesize an original kick, click and chord pattern at 120 BPM."""
    t = np.arange(round(duration * rate)) / rate
    signal = np.zeros_like(t)
    rng = np.random.default_rng(seed)
    for index, start in enumerate(np.arange(0.5, duration, 0.5)):
        age = t - start
        positive = np.maximum(age, 0)
        gate = (age >= 0) & (age < 0.3)
        kick = np.sin(2 * np.pi * (55 * positive + 3.5 * (1 - np.exp(-positive * 28))))
        click = rng.uniform(-1, 1, len(t)) * np.exp(-positive * 95)
        chord = sum(
            np.sin(2 * np.pi * frequency * positive)
            for frequency in (220, 277.18, 329.63)
        )
        signal += gate * (
            0.42 * kick * np.exp(-positive * 24)
            + 0.09 * click
            + 0.035 * chord * np.exp(-positive * 15)
        )
    fade = np.minimum((duration - t) / 0.1, 1)
    left = signal * fade
    right = (0.92 * signal + 0.08 * np.roll(signal, round(rate * 0.004))) * fade
    stereo = np.column_stack((left, right))
    with wave.open(str(path), "wb") as handle:
        handle.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        handle.writeframes(
            np.rint(np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes()
        )


def backdrop(size, palette, label):
    """Build original geometric motion from arrays and simple line art."""
    width, height = size
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    x, y = x / width, y / height
    dark, light, accent = [np.asarray(color, dtype=np.float32) for color in palette]
    font = ImageFont.load_default(size=max(10, round(height * 0.024)))
    label_layer = Image.new("RGBA", size)
    drawing = ImageDraw.Draw(label_layer)
    margin = round(width * 0.08)
    drawing.text(
        (margin, round(height * 0.085)),
        "MOVIEPY / GODOT",
        font=font,
        fill=(188, 213, 235),
    )
    drawing.text(
        (margin, round(height * 0.135)),
        label,
        font=font,
        fill=tuple(int(c) for c in accent),
    )
    drawing.line(
        (margin, round(height * 0.205), width - margin, round(height * 0.205)),
        fill=(*[int(c) for c in accent], 65),
        width=max(1, height // 360),
    )
    overlay = np.asarray(label_layer, dtype=float)
    alpha = overlay[..., 3:4] / 255

    def frame(t):
        halo = np.exp(
            -(((x - 0.5 - 0.04 * math.sin(t)) / 0.6) ** 2 + ((y - 0.42) / 0.75) ** 2)
            * 2
        )
        rgb = dark + (light - dark) * halo[..., None]
        cx, cy = 0.5 + 0.1 * math.sin(t * 0.8), 0.45
        radius = np.sqrt(((x - cx) * width / height) ** 2 + (y - cy) ** 2)
        rings = (
            np.exp(-(((radius - 0.31) / 0.0025) ** 2))
            + np.exp(-(((radius - 0.36) / 0.0015) ** 2)) * 0.35
        )
        rgb += accent * rings[..., None] * 0.25
        columns = ((x + t * 0.006) * 32) % 1
        rows = (y * 18) % 1
        grid = (columns < 0.02) | (rows < 0.025)
        rgb += grid[..., None] * 3
        return np.rint(
            np.clip(rgb * (1 - alpha) + overlay[..., :3] * alpha, 0, 255)
        ).astype(np.uint8)

    return VideoClip(frame, duration=8)


def main(argv=None):
    """Render a new showcase directory and record each cold/warm cache timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--godot", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    width, height, fps = (640, 360, 12) if args.preview else (1280, 720, 24)
    count, seed = 8 * fps, 605
    sound = args.output / "original_rhythm.wav"
    make_rhythm(sound, seed=seed)
    analysis = analyze_audio(sound, fps=fps, frame_count=count)
    expected = [round(time * fps) for time in np.arange(0.5, 8, 0.5)]
    actual = [event["frame"] for event in analysis["events"]]
    if len(actual) != len(expected) or any(
        abs(a - b) > 1 for a, b in zip(actual, expected)
    ):
        raise RuntimeError(f"original rhythm onset acceptance failed: {actual}")
    common = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": count,
        "seed": seed,
    }
    font_asset = {"font": str(args.font)}
    requests = {
        "title": {
            **common,
            "effect": "title_card",
            "frame_count": round(2.5 * fps),
            "assets": font_asset,
            "cues": [{"text": "聲 音 有 形"}],
            "params": {"color": [0.1, 0.72, 0.95], "depth": 0.32, "turn_degrees": 24},
        },
        "pop_captions": {
            **common,
            "effect": "captions",
            "assets": font_asset,
            "params": {
                "style": "pop",
                "accent": [0.94, 0.98, 1],
                "font_size": round(height * 0.063),
                "position": [0.5, 0.82],
                "stroke_width": max(1, height // 360),
            },
            "cues": [
                {"text": "讓文字 跟著節奏跳動", "start_frame": 0, "end_frame": 2 * fps},
                {
                    "text": "換個畫面 讓節奏繼續",
                    "start_frame": 6 * fps,
                    "end_frame": count,
                },
            ],
        },
        "highlight_captions": {
            **common,
            "effect": "captions",
            "assets": font_asset,
            "params": {
                "style": "highlight",
                "font_size": round(height * 0.063),
                "position": [0.5, 0.82],
                "accent": [1, 0.76, 0.28],
                "stroke_width": max(1, height // 360),
            },
            "cues": [
                {
                    "text": "每一個字 都在固定的位置",
                    "start_frame": 2 * fps,
                    "end_frame": 4 * fps,
                },
                {
                    "text": "聲音峰值 驅動真實粒子",
                    "start_frame": 4 * fps,
                    "end_frame": 6 * fps,
                },
            ],
        },
        "particles": {
            **common,
            "effect": "beat_particles",
            "assets": {"audio": str(sound)},
            "audio_events": analysis["events"],
            "params": {
                "count": 220,
                "lifetime": 0.9,
                "location": [0, -0.25, 0],
                "speed": [1.4, 3.8],
                "spread": 100,
                "color": [0.14, 0.75, 1],
                "size": 0.027,
            },
        },
        "wipe": {
            **common,
            "effect": "transition",
            "frame_count": fps,
            "params": {"style": "wipe", "direction": "left_to_right", "softness": 0.12},
        },
        "iris": {
            **common,
            "effect": "transition",
            "frame_count": fps,
            "params": {"style": "iris", "direction": "out", "softness": 0.09},
        },
    }
    results, timings = {}, {}
    for name, request in requests.items():
        start = time.perf_counter()
        result = render_motion_effect(request, args.output / "cache", godot=args.godot)
        cold = time.perf_counter() - start
        start = time.perf_counter()
        reused = render_motion_effect(request, args.output / "cache", godot=args.godot)
        warm = time.perf_counter() - start
        if not reused.cache_hit:
            raise RuntimeError("second lookup did not reuse the validated cache")
        results[name] = result
        timings[name] = {
            "cold_seconds": cold,
            "warm_seconds": warm,
            "cache_key": result.metadata["cache_key"],
        }
        print(f"{name}: render {cold:.3f}s, verified cache {warm:.3f}s", flush=True)

    backgrounds = [
        backdrop(
            (width, height),
            ((4, 9, 19), (13, 31, 48), (40, 201, 240)),
            "01 / TYPE + DEPTH",
        ),
        backdrop(
            (width, height),
            ((16, 7, 22), (45, 17, 44), (245, 143, 99)),
            "02 / RHYTHM + PARTICLES",
        ),
        backdrop(
            (width, height),
            ((7, 8, 28), (24, 29, 72), (125, 159, 255)),
            "03 / MASK + MOTION",
        ),
    ]
    wipe = transition_clip(
        backgrounds[0].subclipped(3, 4),
        backgrounds[1].subclipped(3, 4),
        results["wipe"],
    )
    iris = transition_clip(
        backgrounds[1].subclipped(6, 7),
        backgrounds[2].subclipped(6, 7),
        results["iris"],
    )

    def base_frame(t):
        if 3 <= t < 4:
            return wipe.get_frame(t - 3)
        if 6 <= t < 7:
            return iris.get_frame(t - 6)
        return backgrounds[0 if t < 3 else 1 if t < 6 else 2].get_frame(t)

    base = VideoClip(base_frame, duration=8).with_fps(fps)
    title_layer = (
        results["title"]
        .to_clip()
        .with_effects([vfx.CrossFadeIn(0.15), vfx.CrossFadeOut(0.3)])
        .with_start(0.25)
    )
    layers = [base, results["particles"].to_clip(), title_layer]
    layers += [
        results[name].to_clip() for name in ("pop_captions", "highlight_captions")
    ]
    audio = AudioFileClip(str(sound))
    video = (
        CompositeVideoClip(layers, size=(width, height))
        .with_audio(audio)
        .with_duration(8)
        .with_fps(fps)
    )
    try:
        video.write_videofile(
            str(args.output / "motion_modules.mp4"),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            audio_fps=48000,
            preset="medium",
            temp_audiofile=str(args.output / "encoded_audio.m4a"),
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
        for index, t in enumerate((0.75, 2.6, 3.5, 4.8, 6.5, 7.6)):
            Image.fromarray(video.get_frame(t)).save(
                args.output / f"preview_{index}.png"
            )
        # Check the same transparent title over three contrasting backdrops.
        title = results["title"].to_clip()
        rgba = np.array(Image.open(results["title"].frames[fps]))
        alpha = rgba[..., 3:4] / 255
        for name, color in (
            ("black", [0, 0, 0]),
            ("white", [255, 255, 255]),
            ("purple", [145, 32, 180]),
        ):
            pixels = np.rint(
                rgba[..., :3] * alpha + np.array(color) * (1 - alpha)
            ).astype(np.uint8)
            Image.fromarray(pixels).save(args.output / f"alpha_{name}.png")
        title.close()
        report = {
            "size": [width, height],
            "fps": fps,
            "duration": 8,
            "frames": count,
            "analysis": analysis,
            "expected_onset_frames": expected,
            "timings": timings,
            "effects": {name: value.metadata for name, value in results.items()},
            "assets": "original synthesized rhythm and geometry; explicit local CJK font, not copied or uploaded",
            "composition": "sRGB SDR with straight-alpha caches and premultiplied mask transitions; no transparent glow assumption",
        }
        (args.output / "validation.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(str(args.output / "motion_modules.mp4"), flush=True)
    finally:
        video.close()
        audio.close()
        for clip in [*layers, *backgrounds, wipe, iris]:
            clip.close()


if __name__ == "__main__":
    main()
