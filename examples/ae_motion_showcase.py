"""Render a synthetic title, particle and optional real-3D/AI showcase.

Run from the repository: python -m examples.ae_motion_showcase OUTPUT_DIR
Add --blender for Cycles geometry/PBR/shadows; --font PATH for a Chinese title;
--u2net PATH only for an already approved/downloaded local U2Net ONNX model.
All imagery is synthetic. Models, fonts and production assets are not copied.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import moviepy.ae as ae
from moviepy import ColorClip, CompositeVideoClip, VideoClip
from moviepy.ae.ai import U2NetBackgroundRemoval, inpaint_classical
from moviepy.ae.three_d import render_scene


def make_textures(directory):
    """Create original synthetic PBR images; no Substance package is loaded."""
    directory.mkdir(parents=True)
    y, x = np.indices((64, 64))
    cells = ((x // 8 + y // 8) % 2)[..., None]
    base = np.where(cells, (35, 100, 210), (205, 225, 245)).astype(np.uint8)
    maps = {
        "base_color": base,
        "roughness": np.tile(np.linspace(60, 210, 64).astype(np.uint8), (64, 1)),
        "metallic": np.full((64, 64), 130, np.uint8),
        "normal": np.tile(np.array([128, 128, 255], np.uint8), (64, 64, 1)),
        "height": (cells[..., 0] * 60 + 100).astype(np.uint8),
    }
    paths = {}
    for name, pixels in maps.items():
        path = directory / f"{name}.png"
        Image.fromarray(pixels).save(path)
        paths[name] = str(path.resolve())
    return paths


def scene_3d(maps, *, size=(320, 180), fps=12, duration=2):
    """A real rotating textured cube, sphere, floor, key light and fill light."""
    return {
        "size": list(size),
        "fps": fps,
        "duration": duration,
        "samples": 12,
        "camera": {"location": [4, -7, 3.2], "target": [0, 0, 0.45], "lens": 48},
        "materials": {
            "checker": {"maps": maps, "metallic": 0.5, "roughness": 0.3},
            "floor": {"base_color": [0.13, 0.16, 0.22, 1], "roughness": 0.6},
            "gold": {
                "base_color": [0.8, 0.36, 0.04, 1],
                "metallic": 0.75,
                "roughness": 0.22,
            },
        },
        "objects": [
            {
                "type": "cube",
                "name": "PBR cube",
                "location": [-0.6, 0, 0.65],
                "scale": [0.6, 0.6, 0.6],
                "material": "checker",
                "animation": [
                    {"time": 0, "rotation": [0, 0, -20]},
                    {"time": duration, "rotation": [0, 0, 90]},
                ],
            },
            {
                "type": "sphere",
                "location": [1, 0.5, 0.6],
                "scale": [0.6, 0.6, 0.6],
                "material": "gold",
            },
            {
                "type": "plane",
                "location": [0, 0, 0],
                "scale": [3, 3, 3],
                "material": "floor",
            },
        ],
        "lights": [
            {"type": "area", "location": [-3, -4, 5], "energy": 650, "size": 3},
            {
                "type": "area",
                "location": [3, 1, 3],
                "energy": 350,
                "size": 2,
                "color": [0.35, 0.6, 1],
            },
        ],
    }


def build_scene(*, font=None, background_3d=None, duration=2, size=(640, 360), fps=12):
    """Composite title presets, analytic sparks and an optional 3D RGBA clip."""
    width, height = size
    comp = ae.Composition(size=size, fps=fps, duration=duration, name="Motion tools")
    yy, xx = np.indices((height, width))

    def gradient(t):
        wave = 0.5 + 0.5 * np.sin(xx / 150 + yy / 180 + t)
        return np.stack((8 + 8 * wave, 13 + 16 * wave, 26 + 28 * wave), axis=-1).astype(
            np.uint8
        )

    comp.add_clip(VideoClip(gradient, duration=duration), "synthetic background")
    burst = comp.add_particles(
        count=500,
        seed=605,
        emitter=(width / 2, height / 2),
        speed=(35, 190),
        gravity=(0, 30),
        lifetime=1.8,
        emission_duration=0.25,
        radius=2.5,
    )
    burst.transform = ae.Transform(position=(width / 2, height / 2))
    if background_3d is not None:
        layer = comp.add_clip(background_3d, "Cycles geometry and shadows")
        layer.transform = ae.Transform(
            position=(width / 2, height / 2),
            scale=(100 * width / background_3d.w, 100 * height / background_3d.h),
        )
    title = comp.add_text(
        "光影實驗室" if font else "LIGHT IN MOTION",
        font=font,
        font_size=36,
        preset="typewriter",
        rate=10,
        delay=0.1,
    )
    title.transform = ae.Transform(position=(width / 2, height * 0.19))
    title.effects = [
        ae.fx.DropShadow(distance=3, softness=6, opacity=80),
        ae.fx.Glow(radius=10, threshold=60, intensity=0.5),
    ]
    subtitle = comp.add_text(
        "PARTICLES  /  TYPOGRAPHY  /  CYCLES",
        font=font,
        font_size=15,
        preset="word_fade",
        animate_duration=0.35,
        stagger=0.13,
    )
    subtitle.transform = ae.Transform(position=(width / 2, height * 0.87))
    return comp


def ai_sample(directory, model_path):
    """Use the real supplied model on a synthetic object and export diagnostics."""
    image = Image.new("RGB", (320, 240), (200, 218, 232))
    draw = ImageDraw.Draw(image)
    draw.ellipse((90, 35, 230, 175), fill=(215, 120, 35), outline=(90, 40, 10), width=5)
    draw.rounded_rectangle((130, 155, 190, 225), radius=10, fill=(70, 95, 55))
    frame = np.array(image)
    remover = U2NetBackgroundRemoval(model_path, refine_radius=3)
    alpha = remover.predict_mask(frame)
    image.save(directory / "ai_input.png")
    Image.fromarray(np.rint(alpha * 255).astype(np.uint8)).save(
        directory / "ai_matte.png"
    )
    rgba = np.dstack((frame, np.rint(alpha * 255).astype(np.uint8)))
    Image.fromarray(rgba).save(directory / "ai_cutout.png")
    replacement = np.full_like(frame, (25, 45, 70))
    result = frame * alpha[..., None] + replacement * (1 - alpha[..., None])
    Image.fromarray(np.rint(result).astype(np.uint8)).save(
        directory / "ai_composite.png"
    )

    def moving_subject(t):
        offset = round(22 * np.sin(2 * np.pi * t / 2))
        moving = Image.new("RGB", image.size, (200, 218, 232))
        moving.paste(image, (offset, 0))
        return np.array(moving)

    source = VideoClip(moving_subject, duration=2).with_fps(8)
    cutout = source.with_effects([remover])
    background = ColorClip(image.size, (25, 45, 70), duration=2)
    with CompositeVideoClip([background, cutout], size=image.size) as video:
        video.write_videofile(
            str(directory / "ai_background.mp4"),
            fps=8,
            codec="libx264",
            audio=False,
            logger=None,
        )
        Image.fromarray(video.get_frame(0.5)).save(directory / "ai_video_preview.png")
    cutout.close()
    source.close()
    background.close()
    return {
        "model_sha256": remover.model_sha256,
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "foreground_mean": float(alpha[65:145, 115:205].mean()),
        "corner_mean": float(alpha[:30, :30].mean()),
        "video": "ai_background.mp4: 2 seconds, 8 fps, inference per frame",
        "temporal_consistency": "not provided; independent frame inference",
    }


def main(argv=None):
    """Write a new synthetic validation bundle and an actual H.264 movie."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--blender", action="store_true")
    parser.add_argument("--u2net", type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report, background = {}, None
    if args.blender:
        maps = make_textures(args.output / "textures")
        rendered = render_scene(scene_3d(maps), args.output / "cycles", timeout=600)
        background = rendered.to_clip()
        report["cycles_frames"] = len(rendered.frames)
    comp = build_scene(font=args.font, background_3d=background)
    comp.write_videofile(
        str(args.output / "motion_showcase.mp4"),
        fps=12,
        codec="libx264",
        audio=False,
        logger=None,
    )
    for index, t in enumerate((0.0, 0.6, 1.2, 1.8)):
        Image.fromarray(comp.get_frame(t)).save(args.output / f"preview_{index}.png")
    if args.u2net:
        report["ai_background"] = ai_sample(args.output, args.u2net)
    sample = np.full((96, 128, 3), (35, 100, 150), np.uint8)
    sample[40:55, 50:70] = (255, 0, 0)
    mask = np.zeros(sample.shape[:2], np.float32)
    mask[39:56, 49:71] = 1
    Image.fromarray(sample).save(args.output / "inpaint_input.png")
    Image.fromarray(inpaint_classical(sample, mask)).save(
        args.output / "inpaint_classical.png"
    )
    report["inpainting"] = "classical Telea; not AI or generative fill"
    report["substance"] = "exported PBR maps only; no native SBSAR"
    (args.output / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    comp.close()
    if background is not None:
        background.close()


if __name__ == "__main__":
    main()
