"""Render ECLIPSE: the same original plate through legacy and linear pipelines.

Run ``python -m examples.ae_cinema_quality OUTPUT --godot PATH --font PATH``.
All geometry, animation, gradient plates and sound are generated locally.
The supplied font is used in place and is not redistributed. Both versions
share the exact Godot PNGs. The legacy version takes the existing uint8
get_frame boundary; the linear version exports directly from float32.
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from moviepy import VideoClip
from moviepy.ae import Composition, Property, RenderContext, Transform
from moviepy.ae.effects.blur.gaussian_blur import GaussianBlur
from moviepy.ae.three_d import render_godot_scene
from moviepy.config import FFMPEG_BINARY
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


SIZE, FPS, COUNT = (1280, 544), 24, 96
DURATION = COUNT / FPS


def scene_spec():
    """Define a restrained copper orbit, dark sphere and three shaped lights."""
    keys = []
    for frame in range(COUNT + 1):
        u = frame / COUNT
        eased = u * u * (3 - 2 * u)
        keys.append(
            {"time": frame / FPS, "rotation": [79 + 7 * eased, -12 + 18 * eased, 24]}
        )
    return {
        "size": list(SIZE),
        "fps": FPS,
        "duration": DURATION,
        "transparent": True,
        "camera": {"location": [0.5, 0.45, 7], "target": [0, 0.25, 0], "fov": 37},
        "ambient": 0.18,
        "materials": {
            "obsidian": {
                "color": [0.035, 0.045, 0.055],
                "metallic": 0.65,
                "roughness": 0.25,
            },
            "copper": {
                "color": [0.55, 0.23, 0.07],
                "metallic": 0.65,
                "roughness": 0.22,
                "emission": [0.30, 0.09, 0.012],
                "emission_energy": 0.55,
            },
        },
        "objects": [
            {"type": "sphere", "material": "obsidian", "scale": [2.3, 2.3, 2.3]},
            {
                "type": "torus",
                "material": "copper",
                "scale": [1.6, 1.6, 1.6],
                "rotation": [79, -12, 24],
                "animation": keys,
            },
        ],
        "lights": [
            {
                "type": "spot",
                "location": [-3, 4, 5],
                "target": [0, 0, 0],
                "energy": 10,
                "angle": 37,
                "color": [1, 0.68, 0.35],
            },
            {
                "type": "omni",
                "location": [3, 0.4, 2],
                "energy": 2.2,
                "color": [0.12, 0.40, 0.65],
            },
            {
                "type": "spot",
                "location": [1.5, 2, -3],
                "target": [0, 0, 0],
                "energy": 14,
                "angle": 52,
                "color": [0.12, 0.55, 0.80],
            },
        ],
    }


def soundtrack(path):
    """Write a quiet original stereo chord with a deterministic breath of noise."""
    rate = 48000
    t = np.arange(COUNT * (rate // FPS)) / rate
    fade = np.sin(np.pi * np.clip(t / 0.7, 0, 1) / 2) ** 2
    fade *= np.sin(np.pi * np.clip((DURATION - t) / 1.1, 0, 1) / 2) ** 2
    chord = sum(
        np.sin(2 * np.pi * hz * t) * gain
        for hz, gain in ((55, 0.025), (110, 0.014), (164.81, 0.008))
    )
    rng = np.random.default_rng(104)
    noise = rng.standard_normal(t.size)
    breath = np.convolve(noise, np.ones(65) / 65, mode="same") * 0.025
    sound = (chord + breath * np.exp(-(((t - 1.4) / 0.65) ** 2))) * fade
    stereo = np.column_stack((sound, sound * 0.96))
    with wave.open(str(path), "wb") as stream:
        stream.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        stream.writeframes(np.rint(stereo * 32767).astype("<i2").tobytes())


def gradient_plate():
    """Generate a continuous floating-point sRGB plate, without uint8 steps."""
    width, height = SIZE
    y, x = np.mgrid[:height, :width].astype(np.float32)
    x, y = (x + 0.5) / width, (y + 0.5) / height
    cool = np.exp(-(((x - 0.59) / 0.43) ** 2 + ((y - 0.38) / 0.72) ** 2) * 2)
    warm = np.exp(-(((x - 0.30) / 0.25) ** 2 + ((y - 0.45) / 0.50) ** 2) * 2)
    rgb = np.full((height, width, 3), [0.018, 0.027, 0.037], np.float32)
    rgb += cool[..., None] * [0.032, 0.078, 0.100]
    rgb += warm[..., None] * [0.070, 0.021, 0.004]
    return rgb * 255


def make_composition(footage, font, space, *, foreground_only=False):
    """Keep artwork/settings identical; change only the working color encoding."""
    comp = Composition(
        size=SIZE,
        fps=FPS,
        duration=DURATION,
        transparent=foreground_only,
        context=(
            RenderContext() if space == "srgb" else RenderContext(working_space=space)
        ),
    )
    if not foreground_only:
        plate = gradient_plate()
        comp.add_clip(VideoClip(lambda t: plate, duration=DURATION))
    # A restrained two-dimensional push, not a claim of camera parallax.
    scale = Property(lambda t: (80 + 1.5 * t / DURATION,) * 2, value_type="vec2")
    hero = comp.add_clip(
        footage,
        name="Orbit",
        transform=Transform(
            scale=scale,
            position=(SIZE[0] / 2, SIZE[1] / 2 - 68),
            interpolation="linear",
        ),
    )
    if not foreground_only:
        # This glow is a separate dim blurred duplicate, with identical
        # geometry/energy in both versions; it exercises linear filtering.
        bloom = comp.add_clip(
            footage,
            name="Optical diffusion",
            transform=Transform(
                scale=scale,
                position=(SIZE[0] / 2, SIZE[1] / 2 - 68),
                opacity=12,
                interpolation="linear",
            ),
        )
        bloom.effects = [GaussianBlur(blurriness=24)]
        comp.move_layer(hero, 1)
        comp.add_text(
            "E C L I P S E",
            font=font,
            font_size=34,
            color=(211, 203, 183),
            transform=Transform(position=(SIZE[0] / 2, 456), interpolation="linear"),
        )
        comp.add_text(
            "A  S T U D Y  I N  L I G H T",
            font=font,
            font_size=11,
            color=(101, 131, 143),
            transform=Transform(position=(SIZE[0] / 2, 493), interpolation="linear"),
        )
    return comp


def preview(master, target):
    """Make a tagged SDR H.264/AAC viewing copy with fixed encoding settings."""
    command = [
        FFMPEG_BINARY,
        "-hide_banner",
        "-nostdin",
        "-n",
        "-i",
        str(master),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-crf",
        "15",
        "-preset",
        "slow",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,"
        "setparams=range=limited:color_primaries=bt709:"
        "color_trc=iec61966-2-1:colorspace=bt709",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "iec61966-2-1",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    with target.with_suffix(".log").open("wb") as log:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=log,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    return command


def main():
    """Render shared footage once, then export both comparison pipelines."""
    from moviepy.ae import write_master

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--font", type=Path, required=True)
    parser.add_argument(
        "--reuse-godot",
        type=Path,
        help="Reuse this example's unchanged verified Godot render",
    )
    args = parser.parse_args()
    font = args.font.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    total_start = start = time.perf_counter()
    scene = scene_spec()
    if args.reuse_godot:
        from moviepy.ae.three_d._godot_scene import validate_godot_scene
        from moviepy.ae.three_d.godot import _verify_output

        source = args.reuse_godot.resolve()
        if json.loads((source / "scene.json").read_text()) != validate_godot_scene(
            scene
        ):
            raise ValueError("reused scene must exactly match this example")
        frames, _ = _verify_output(source, scene)
    else:
        source = output / "godot"
        frames = render_godot_scene(scene, source, executable=args.godot).frames
    elapsed_godot = time.perf_counter() - start
    footage = ImageSequenceClip(list(frames), fps=FPS, with_mask=True)
    soundtrack(output / "sound.wav")
    report = {
        "size": SIZE,
        "fps": FPS,
        "frames": COUNT,
        "godot_directory": str(source),
        "godot_seconds": elapsed_godot,
        "font_sha256": hashlib.sha256(font.read_bytes()).hexdigest(),
        "sources_sha256": [
            hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in frames
        ],
        "scene": scene,
        "timings": {},
        "preview_commands": {},
    }
    for space in ("srgb", "linear"):
        comp = make_composition(footage, font, space)
        if space == "srgb":
            # This explicitly preserves the baseline's 8-bit get_frame boundary.
            baseline = comp
            comp = Composition(size=SIZE, fps=FPS, duration=DURATION)
            comp.add_clip(VideoClip(baseline.get_frame, duration=DURATION))
        start = time.perf_counter()
        master = write_master(comp, output / space, audio_path=output / "sound.wav")
        report["timings"][space + "_master_seconds"] = time.perf_counter() - start
        report["preview_commands"][space] = preview(
            master.path, output / f"{space}.mp4"
        )
        for index in (0, COUNT // 2, COUNT - 1):
            Image.fromarray(comp.get_frame(index / FPS)).save(
                output / f"{space}_{index:03d}.png"
            )
        print(f"{space}: master and preview complete", flush=True)
    foreground = make_composition(footage, font, "linear", foreground_only=True)
    start = time.perf_counter()
    write_master(foreground, output / "foreground")
    report["timings"]["foreground_master_seconds"] = time.perf_counter() - start
    report["total_seconds"] = time.perf_counter() - total_start
    (output / "settings.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    footage.close()
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
