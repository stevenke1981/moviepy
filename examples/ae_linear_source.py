"""Capture ECLIPSE as scene-linear EXR, then explicitly map it to an SDR study.

Run ``python -m examples.ae_linear_source OUTPUT --godot PATH --font PATH``.
The EXRs/NPZ retain HDR values. The MP4/master are deliberately SDR, with a
documented exposure and simple Reinhard mapping applied in this example.
"""

import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
from PIL import Image

from examples.ae_cinema_quality import SIZE, gradient_plate, preview, scene_spec
from moviepy import VideoClip
from moviepy.ae import Buffer, Composition, RenderContext, Transform, write_master
from moviepy.ae.buffer import unpremultiply
from moviepy.ae.effects.base import AEEffect
from moviepy.ae.three_d import render_godot_scene


class StudyDisplayMap(AEEffect):
    """Example-only relative exposure/Reinhard display map, not an HDR grade."""

    def render(self, buffer, t, context=None, values=None):
        """Apply relative exposure and compress highlights for the SDR study."""
        rgba = unpremultiply(buffer.rgba)
        # -1 stop; a peak-channel Reinhard gain retains RGB ratios and keeps
        # this illustrative SDR preview within range, including saturated light.
        rgb = np.maximum(rgba[..., :3], 0) * 0.5
        peak = np.max(rgb, axis=2, keepdims=True)
        rgba[..., :3] = rgb / (1 + peak) * rgba[..., 3:4]
        return Buffer(rgba, offset=buffer.offset, color_space="linear")


def main():
    """Capture a native linear sequence and export a mapped SDR preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False, parents=True)
    scene = scene_spec()
    scene["duration"] = 2
    for item in scene["objects"]:
        for key in item.get("animation", []):
            key["time"] *= 0.5
    # Deliberately bright emission tests highlight retention. This changes
    # the study lighting and is separate from the unchanged performance case.
    scene["materials"]["copper"]["emission_energy"] = 8
    start = time.perf_counter()
    source = render_godot_scene(
        scene, output / "godot", executable=args.godot, linear_output=True
    )
    capture_seconds = time.perf_counter() - start
    comp = Composition(
        size=SIZE, fps=24, duration=2, context=RenderContext(working_space="linear")
    )
    plate = gradient_plate()
    comp.add_clip(VideoClip(lambda t: plate, duration=2))
    layer = comp.add_layer(
        source.to_linear_layer(
            transform=Transform(
                scale=(80, 80),
                position=(SIZE[0] / 2, SIZE[1] / 2 - 68),
                interpolation="linear",
            )
        )
    )
    layer.effects = [StudyDisplayMap()]
    comp.add_text(
        "E C L I P S E",
        font=args.font,
        font_size=34,
        color=(211, 203, 183),
        transform=Transform(position=(SIZE[0] / 2, 456), interpolation="linear"),
    )
    comp.add_text(
        "L I N E A R   S O U R C E",
        font=args.font,
        font_size=11,
        color=(101, 131, 143),
        transform=Transform(position=(SIZE[0] / 2, 493), interpolation="linear"),
    )
    t = np.arange(96000) / 48000
    sound = (
        np.sin(2 * np.pi * 55 * t) * 0.02 + np.sin(2 * np.pi * 110 * t) * 0.012
    ) * np.sin(np.pi * t / 2) ** 2
    with wave.open(str(output / "sound.wav"), "wb") as audio:
        audio.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
        audio.writeframes(
            np.rint(np.column_stack((sound, sound * 0.96)) * 32767)
            .astype("<i2")
            .tobytes()
        )
    start = time.perf_counter()
    master = write_master(comp, output / "master", audio_path=output / "sound.wav")
    export_seconds = time.perf_counter() - start
    command = preview(master.path, output / "linear_source.mp4")
    for index in (0, 24, 47):
        Image.fromarray(comp.get_frame(index / 24)).save(
            output / f"frame{index:03d}.png"
        )
    report = {
        "source": source.metadata,
        "scene": scene,
        "capture_seconds": capture_seconds,
        "master_seconds": export_seconds,
        "display": "-1 EV, peak-channel Reinhard, then SDR sRGB encoding",
        "master_manifest": str(master.manifest_path),
        "preview_command": command,
    }
    (output / "settings.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "capture_seconds": capture_seconds,
                "export_seconds": export_seconds,
                "peak_source_rgb": source.metadata["linear"]["peak_rgb"],
                "clipped_output_channels": master.metadata["clipped_rgb_channels"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
