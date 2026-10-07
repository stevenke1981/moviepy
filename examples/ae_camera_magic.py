"""Demonstrate core AE framing and native Godot spells on original ECLIPSE footage.

Reuse a completed original ``ae_cinema_quality`` Godot directory via
``--source-frames``. No formal episode, downloaded media, or paid asset is used.
All layer placement, alpha, glow, masks and camera geometry run through AE.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from examples.ae_cinema_quality import gradient_plate
from moviepy import ImageClip, VideoClip
from moviepy.ae import (
    CameraFraming,
    CompLayer,
    Composition,
    Mask,
    RenderContext,
    Transform,
    pan_zoom,
)
from moviepy.ae.color import linear_to_srgb
from moviepy.ae.motion.magic import PRESETS, render_magic
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


FPS, COUNT = 24, 48
SIZE = (1280, 720)
SOURCE_SIZE = (1600, 1000)
SUBJECT = (580, 270, 1030, 760)
CONTEXT = RenderContext(working_space="linear")


def composition(size, duration=2, *, transparent=False):
    """Create a native linear composition with a fixed frame clock."""
    return Composition(
        size=size, fps=FPS, duration=duration, transparent=transparent, context=CONTEXT
    )


def original_plate(paths):
    """Reuse the existing original geometry over its original gradient design."""
    source = composition(SOURCE_SIZE)
    plate = ImageClip(gradient_plate()).with_duration(2)
    source.add_clip(
        plate,
        transform=Transform(
            anchor_point=(0, 0),
            position=(-1, -1),
            scale=(100 * 1602 / 1279, 100 * 1002 / 543),
            interpolation="linear",
        ),
    )
    footage = ImageSequenceClip([str(path) for path in paths], fps=FPS, with_mask=True)
    source.add_clip(
        footage,
        transform=Transform(
            position=(800, 485), scale=(110, 110), interpolation="linear"
        ),
    )
    return source, footage, plate


def label(comp, text, position, font, *, size=24, color=(237, 222, 187)):
    """Place an unobtrusive core TextLayer outside the protected subject."""
    comp.add_text(
        text,
        font=font,
        font_size=size,
        color=color,
        transform=Transform(position=position, interpolation="linear"),
    )


def display_frame(comp, t, *, tone_map=True):
    """Apply an explicit SDR viewing map after native float compositing."""
    pixels = comp.render_buffer(t).rgba
    rgb = pixels[..., :3]
    mapped = (
        rgb / (1 + np.maximum(rgb.max(axis=2, keepdims=True), 0)) if tone_map else rgb
    )
    return np.rint(np.clip(linear_to_srgb(mapped), 0, 1) * 255).astype(np.uint8)


def encode(comp, target):
    """Write a small silent viewing copy; the source RGBA/HDR remains separate."""
    clip = VideoClip(lambda t: display_frame(comp, t), duration=comp.duration)
    try:
        clip.write_videofile(
            str(target),
            fps=FPS,
            codec="libx264",
            audio=False,
            preset="fast",
            threads=2,
            ffmpeg_params=[
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
            logger=None,
        )
    finally:
        clip.close()
    if target.stat().st_size >= 9_000_000:
        raise RuntimeError("preview exceeds the requested 9 MB budget")


def camera_demo(source, output, font):
    """Show the four viewing-window directions and a protected-subject push."""
    result = composition(SIZE, duration=10)
    evidence, stills = [], []
    directions = (
        ("left", (-1, 0)),
        ("right", (1, 0)),
        ("up", (0, -1)),
        ("down", (0, 1)),
        ("none", (0, 0)),
    )
    for index, (direction, vector) in enumerate(directions):
        focus = (805 - vector[0] * 45, 515 - vector[1] * 45)
        rig = pan_zoom(
            SOURCE_SIZE,
            SIZE,
            duration=(COUNT - 1) / FPS,
            direction=direction,
            distance=90,
            zoom=(1.18, 1.18) if direction != "none" else (1.0, 1.32),
            focus=focus,
            subject_bounds=SUBJECT,
            overscan=1.06,
            safe_margin=0.08,
        )
        shot = composition(SIZE)
        shot.add_layer(CompLayer(source, transform=rig.to_transform()))
        title = "SUBJECT PUSH" if direction == "none" else "CAMERA " + direction.upper()
        label(shot, title, (640, 49), font, size=30)
        label(
            shot,
            "Manual subject box  |  2D framing  |  Covered edges",
            (640, 679),
            font,
            size=20,
        )
        result.add_layer(
            CompLayer(
                shot,
                start_time=index * 2,
                in_point=index * 2,
                out_point=(index + 1) * 2,
            )
        )
        samples = []
        for frame in (0, 24, 47):
            t = frame / FPS
            buffer = shot.render_buffer(t)
            if not np.all(buffer.rgba[..., 3] == 1):
                raise RuntimeError("camera preview exposed transparent pixels")
            sample = asdict(rig.sample(t))
            samples.append({"frame": frame, **sample})
            stills.append((title + f" / {frame:02d}", display_frame(shot, t)))
        evidence.append({"direction": direction, "samples": samples})
    encode(result, output / "camera_directions_720p.mp4")
    contact_sheet(
        stills, output / "camera_contact.png", columns=3, tile=(400, 225), font=font
    )
    return evidence


def protected_spell(source, magic):
    """Mask after Glow using an outer CompLayer, preserving subject and captions."""
    view_size = (608, 512)
    rig = CameraFraming(
        SOURCE_SIZE,
        view_size,
        focus=(805, 515),
        zoom=1.1,
        subject_bounds=SUBJECT,
        overscan=1.04,
    )
    before = composition(view_size)
    before.add_layer(CompLayer(source, transform=rig.to_transform()))
    matrix = rig.to_transform().matrix_at(0)
    points = matrix @ np.array(
        [[SUBJECT[0], SUBJECT[2]], [SUBJECT[1], SUBJECT[3]], [1, 1]]
    )
    left, top = np.floor(points[:2, 0] - 10).astype(int)
    right, bottom = np.ceil(points[:2, 1] + 10).astype(int)
    bounds = (int(left), int(top), int(right), int(bottom))
    effect = composition(view_size, transparent=True)
    effect.add_layer(
        magic.to_layer(position=(510, 320), glow_radius=8, glow_intensity=0.55)
    )
    protected = composition(view_size, transparent=True)
    protected.add_layer(
        CompLayer(
            effect,
            masks=[
                Mask.rect(
                    ((left + right) / 2, (top + bottom) / 2),
                    (right - left + 4, bottom - top + 4),
                    mode="subtract",
                )
            ],
        )
    )
    after = composition(view_size)
    after.add_layer(CompLayer(before))
    after.add_layer(CompLayer(protected))
    checks = []
    for frame in (0, 1, 24, 47):
        t = frame / FPS
        alpha = protected.render_buffer(t).rgba[..., 3]
        if np.any(alpha[top : bottom + 1, left : right + 1]):
            raise RuntimeError("spell crossed the manually protected subject region")
        if frame in (0, 47) and np.any(alpha):
            raise RuntimeError("spell did not fade to transparent at its endpoints")
        a, b = before.render_buffer(t).rgba, after.render_buffer(t).rgba
        if not np.array_equal(
            a[top : bottom + 1, left : right + 1], b[top : bottom + 1, left : right + 1]
        ):
            raise RuntimeError("spell changed pixels inside the protected region")
        checks.append(
            {
                "frame": frame,
                "protected_max_alpha": 0.0,
                "visible_max_alpha": float(alpha.max()),
            }
        )
    return before, after, {"protected_subject_pixels": bounds, "checks": checks}


def magic_demo(source, output, executable, font):
    """Render real transparent scenes, then compare against identical source frames."""
    result = composition(SIZE, duration=6)
    evidence, stills, alpha_stills = [], [], []
    for index, preset in enumerate(PRESETS):
        magic = render_magic(
            preset,
            output / preset,
            executable=executable,
            size=(192, 192),
            fps=FPS,
            frame_count=COUNT,
            seed=607,
            emission=2,
            particle_count=24,
        )
        before, after, checks = protected_spell(source, magic)
        shot = composition(SIZE)
        shot.add_solid(color=(9, 11, 13))
        shot.add_layer(
            CompLayer(
                before, transform=Transform(anchor_point=(0, 0), position=(24, 144))
            )
        )
        shot.add_layer(
            CompLayer(
                after, transform=Transform(anchor_point=(0, 0), position=(648, 144))
            )
        )
        label(shot, preset.replace("_", " ").upper(), (640, 49), font, size=30)
        label(shot, "BEFORE", (328, 114), font, size=20)
        label(shot, "AFTER", (952, 114), font, size=20)
        label(
            shot,
            "Native RGBA + AE Glow  |  Manual subject protection",
            (640, 689),
            font,
            size=18,
        )
        result.add_layer(
            CompLayer(
                shot,
                start_time=index * 2,
                in_point=index * 2,
                out_point=(index + 1) * 2,
            )
        )
        stills.append((preset, display_frame(shot, 1)))
        for background in ("black", "white", "checker"):
            board = composition((320, 220))
            yy, xx = np.indices((220, 320))
            level = ((xx // 20 + yy // 20) % 2) * 0.5 + 0.25
            if background != "checker":
                level[:] = float(background == "white")
            board.add_clip(
                ImageClip(
                    (np.repeat(level[..., None], 3, axis=2) * 255).astype(np.float32)
                ).with_duration(2)
            )
            board.add_layer(
                magic.to_layer(position=(160, 110), glow_radius=8, glow_intensity=0.55)
            )
            alpha_stills.append(
                (preset + " / " + background, display_frame(board, 1, tone_map=False))
            )
        checks.update(
            {
                "preset": preset,
                "godot": magic.render.metadata,
                "native_frames": len(magic.render.linear_frames),
            }
        )
        evidence.append(checks)
    encode(result, output / "magic_before_after_720p.mp4")
    contact_sheet(
        stills, output / "magic_contact.png", columns=1, tile=(960, 540), font=font
    )
    contact_sheet(
        alpha_stills,
        output / "alpha_black_white_checker.png",
        columns=3,
        tile=(320, 220),
        font=font,
    )
    return evidence


def contact_sheet(items, path, *, columns, tile, font):
    """Assemble actual AE renders with clearly separated evidence labels."""
    width, height = tile
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + 32)), (15, 17, 18))
    draw = ImageDraw.Draw(sheet)
    face = (
        ImageFont.truetype(str(font), 16) if font else ImageFont.load_default(size=16)
    )
    for index, (title, pixels) in enumerate(items):
        x, y = (index % columns) * width, (index // columns) * (height + 32)
        draw.text((x + 8, y + 6), title, font=face, fill=(237, 222, 187))
        sheet.paste(
            Image.fromarray(pixels).resize(tile, Image.Resampling.LANCZOS), (x, y + 32)
        )
    sheet.save(path)


def main():
    """Render only local, explicitly supplied and existing original fixture frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-frames", type=Path, required=True)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    frames = sorted(args.source_frames.glob("frame????????.png"))
    if len(frames) < COUNT:
        raise ValueError("supply at least 48 original ECLIPSE Godot RGBA frames")
    for frame in frames[:COUNT]:
        with Image.open(frame) as image:
            if image.mode != "RGBA" or image.size != (1280, 544):
                raise ValueError("source contract is original 1280x544 RGBA, 24 fps")
    args.output.mkdir(parents=True, exist_ok=False)
    source, footage, plate = original_plate(frames[:COUNT])
    try:
        evidence = {
            "size": SIZE,
            "fps": FPS,
            "source": "Original ECLIPSE procedural geometry/gradient; examples/ae_cinema_quality.py",
            "source_sha256": [
                hashlib.sha256(p.read_bytes()).hexdigest() for p in frames[:COUNT]
            ],
            "source_unchanged": True,
            "manual_subject_box": SUBJECT,
            "display": "SDR peak-channel Reinhard after native linear compositing; silent MP4",
            "camera": camera_demo(source, args.output, args.font),
            "magic": magic_demo(source, args.output, args.godot, args.font),
        }
        (args.output / "evidence.json").write_text(
            json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        footage.close()
        plate.close()
        source.close()


if __name__ == "__main__":
    main()
