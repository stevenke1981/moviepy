"""Render an original 3D title, seeded GPU sparks, lighting and synthetic sound.

Run: python -m examples.ae_godot_showcase OUTPUT --godot PATH
Godot renders geometry/shadows into transparent PNGs and a WAV. MoviePy adds
the background and 2D bloom, then encodes H.264/AAC. No scene assets are fetched.
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
from PIL import Image

import moviepy.ae as ae
from moviepy import VideoClip
from moviepy.ae.three_d import render_godot_scene


def make_sound(path, duration, fps=48000):
    """Write an original, seeded PCM chord and two short synthetic bursts."""
    t = np.arange(round(duration * fps)) / fps
    envelope = np.minimum(t / 0.15, 1) * np.minimum((duration - t) / 0.5, 1)
    sound = sum(np.sin(2 * np.pi * hz * t) for hz in (110, 164.81, 220)) * 0.025
    rng = np.random.default_rng(605)
    for start in (0.45, 1.6):
        age = np.maximum(t - start, 0)
        burst = (
            rng.uniform(-1, 1, t.size) * 0.12 + np.sin(1800 * np.exp(-age) * age) * 0.09
        )
        sound += burst * np.exp(-age * 9) * (t >= start)
    stereo = np.column_stack((sound, sound)) * envelope[:, None]
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, fps, 0, "NONE", "not compressed"))
        output.writeframes(
            np.rint(np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes()
        )


def showcase_scene(*, size=(1280, 720), fps=24, duration=4, audio=None):
    """Describe Y-up 3D meshes, lights, keyframes and two one-shot emitters."""
    scene = {
        "size": list(size),
        "fps": fps,
        "duration": duration,
        "camera": {"location": [2.6, 2.5, 9], "target": [0, 0.65, 0], "fov": 42},
        "ambient": 0.35,
        "materials": {
            "title": {"color": [0.16, 0.68, 0.87], "metallic": 0.45, "roughness": 0.24},
            "subtitle": {
                "color": [0.7, 0.82, 0.95],
                "emission": [0.25, 0.35, 0.6],
                "emission_energy": 0.3,
            },
            "stage": {
                "color": [0.035, 0.055, 0.08],
                "metallic": 0.2,
                "roughness": 0.55,
                "grid": True,
            },
            "rim": {
                "color": [0.04, 0.7, 1],
                "emission": [0.02, 0.6, 1],
                "emission_energy": 3,
            },
            "gold": {"color": [1, 0.35, 0.025], "metallic": 0.55, "roughness": 0.25},
        },
        "objects": [
            {
                "type": "text",
                "text": "IGNITE",
                "location": [0, 0.85, 0],
                "material": "title",
                "pixel_size": 0.025,
                "depth": 0.24,
                "animation": [
                    {"time": 0, "rotation": [0, -18, 0], "scale": [0.7, 0.7, 0.7]},
                    {"time": 0.7, "rotation": [0, 6, 0], "scale": [1, 1, 1]},
                    {"time": duration, "rotation": [0, -9, 0]},
                ],
            },
            {
                "type": "text",
                "text": "LIGHT  /  MOTION",
                "location": [0, 0.0, 0.15],
                "pixel_size": 0.005,
                "depth": 0.025,
                "material": "subtitle",
            },
            {
                "type": "plane",
                "location": [0, -0.5, 0],
                "scale": [4.6, 1, 2.8],
                "material": "stage",
            },
            {
                "type": "torus",
                "location": [0, -0.43, 0],
                "scale": [2.5, 0.5, 1.1],
                "material": "rim",
            },
            {
                "type": "cube",
                "location": [2.7, 0.1, -0.5],
                "scale": [0.5, 0.5, 0.5],
                "material": "gold",
                "animation": [
                    {"time": 0, "rotation": [20, 0, 25]},
                    {"time": duration, "rotation": [100, 220, 25]},
                ],
            },
        ],
        "lights": [
            {
                "type": "spot",
                "location": [-3, 5, 5],
                "target": [0, 0, 0],
                "energy": 6,
                "angle": 48,
            },
            {
                "type": "spot",
                "location": [3, 3, -2],
                "target": [0, 0.5, 0],
                "energy": 9,
                "angle": 60,
                "color": [0.4, 0.18, 1],
            },
            {
                "type": "directional",
                "location": [-3, 5, 2],
                "target": [0, 0, 0],
                "energy": 1.2,
                "color": [0.45, 0.7, 1],
            },
        ],
        "particles": [
            {
                "location": [-1.7, 0.15, -0.5],
                "count": 520,
                "start": 0.45,
                "lifetime": 2.3,
                "speed": [1.2, 3.0],
                "spread": 70,
                "gravity": [0, -2.2, 0],
                "size": 0.019,
                "seed": 605,
            },
            {
                "location": [1.6, 0.25, -0.4],
                "count": 420,
                "start": 1.6,
                "lifetime": 2.1,
                "speed": [0.8, 2.7],
                "spread": 75,
                "gravity": [0, -2, 0],
                "size": 0.016,
                "color": [0.1, 0.75, 1],
                "seed": 606,
            },
        ],
    }
    if audio is not None:
        scene["audio"] = str(audio)
    return scene


def main(argv=None):
    """Render and compose the demonstration into a new output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--godot", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    size, fps, duration = (args.width, args.width * 9 // 16), 24, 4
    sound = args.output / "original_sound.wav"
    make_sound(sound, duration)
    result = render_godot_scene(
        showcase_scene(size=size, fps=fps, duration=duration, audio=sound),
        args.output / "godot",
        executable=args.godot,
    )
    rendered = result.to_clip(with_audio=True)
    comp = ae.Composition(size=size, fps=fps, duration=duration)
    y, x = np.indices((size[1], size[0]))
    glow = np.exp(
        -(((x / size[0] - 0.5) / 0.55) ** 2 + ((y / size[1] - 0.42) / 0.6) ** 2)
    )
    backdrop = np.stack((5 + 9 * glow, 9 + 18 * glow, 19 + 31 * glow), axis=-1).astype(
        np.uint8
    )
    comp.add_clip(VideoClip(lambda t: backdrop, duration=duration))
    comp.add_clip(rendered)
    comp.add_adjustment().effects = [ae.fx.Glow(radius=14, threshold=70, intensity=0.7)]
    # AE frame generation and audio encoding remain explicit independent stages.
    video = (
        VideoClip(comp.get_frame, duration=duration)
        .with_fps(fps)
        .with_audio(rendered.audio)
    )
    try:
        video.write_videofile(
            str(args.output / "godot_showcase.mp4"),
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
        for index, t in enumerate((0, 0.8, 1.8, 2.6)):
            Image.fromarray(video.get_frame(t)).save(
                args.output / f"preview_{index}.png"
            )
        report = dict(
            result.metadata,
            size=size,
            fps=fps,
            duration=duration,
            frames=len(result.frames),
        )
        report["capabilities"] = {
            "godot": "extruded TextMesh, GPU particles, geometry occlusion, spot/directional shadows, emission, procedural grid",
            "moviepy": "2D background, glow after compositing, H.264/AAC encoding",
            "assets": "original geometry/shader/audio; Godot bundled Open Sans font (OFL-1.1)",
        }
        (args.output / "validation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    finally:
        video.close()
        comp.close()
        if rendered.audio is not None:
            rendered.audio.close()
        rendered.close()


if __name__ == "__main__":
    main()
