"""Render a short demo of the After Effects-style ``moviepy.ae`` compositor.

The scene exercises keyframes with easing, parenting to a null, a
pre-composition, blend modes, feathered and animated masks, a luma track
matte, the Dissolve mode, layer effects (keyframed Gaussian Blur, Echo,
Fill) and an adjustment layer (masked vignette). Run it with an output
path::

    python examples/ae_demo.py ae_demo.mp4
"""

import sys

import numpy as np

import moviepy.ae as ae
from moviepy import VideoClip
from moviepy.ae import Composition, Ease, Keyframe, Mask, Property, Transform
from moviepy.ae.masks import path as shapes


WIDTH, HEIGHT, FPS, DURATION = 640, 360, 24, 4.0


def eased(times, values):
    """Return a Property whose keyframes all use an ease-in-out curve."""
    curve = Ease.ease_in_out()
    keys = [
        Keyframe(time, value, in_ease=curve, out_ease=curve)
        for time, value in zip(times, values)
    ]
    return Property(values[0], keyframes=keys)


def background_clip():
    """Return a slowly shifting color gradient as a regular MoviePy clip."""
    x = np.linspace(0.0, 1.0, WIDTH)[None, :]
    y = np.linspace(0.0, 1.0, HEIGHT)[:, None]

    def frame_function(t):
        phase = 2 * np.pi * t / DURATION
        red = 40 + 50 * x + 0 * y
        green = 30 + 60 * y + 0 * x
        blue = 90 + 60 * (0.5 + 0.5 * np.sin(phase + 3 * x))
        return np.dstack([red, green, blue + 0 * y]).astype(np.uint8)

    return VideoClip(frame_function, duration=DURATION).with_fps(FPS)


def stripes_clip():
    """Return animated rainbow stripes (track-matte fill)."""
    x = np.arange(WIDTH)[None, :]
    y = np.arange(HEIGHT)[:, None]

    def frame_function(t):
        hue = (x + y + 160 * t) / 60.0
        rgb = [127 + 127 * np.sin(hue + offset) for offset in (0.0, 2.1, 4.2)]
        return np.dstack(rgb).astype(np.uint8)

    return VideoClip(frame_function, duration=DURATION).with_fps(FPS)


def badge_comp():
    """Return a transparent pre-comp: a feathered star over a ring."""
    badge = Composition(
        size=(160, 160), fps=FPS, duration=DURATION, transparent=True, name="Badge"
    )
    ring = badge.add_solid("ring", color=(255, 255, 255))
    ring.masks = [
        Mask.ellipse((79.5, 79.5), (150, 150)),
        Mask.ellipse((79.5, 79.5), (128, 128), mode="subtract"),
    ]
    star = badge.add_solid("star", color=(255, 170, 30))
    star.masks = [Mask.star((79.5, 79.5), 5, 62, 26, feather=3)]
    # The star spins inside the pre-comp so the outer layer's Echo sees it move.
    star.transform = Transform(
        rotation=Property(0.0, keyframes=[(0.0, 0.0), (DURATION, 360.0)])
    )
    return badge


def build_scene():
    """Assemble the demo composition."""
    comp = Composition(size=(WIDTH, HEIGHT), fps=FPS, duration=DURATION, name="AE demo")
    comp.add_clip(background_clip(), "background")

    # Blend modes: two feathered discs over the background.
    for name, color, center, mode in [
        ("screen disc", (40, 120, 255), (200, 180), "screen"),
        ("multiply disc", (255, 60, 160), (300, 180), "multiply"),
    ]:
        disc = comp.add_solid(name, color=color)
        disc.masks = [Mask.ellipse(center, (220, 220), feather=30)]
        disc.blend_mode = mode

    # Luma track matte: stripes revealed through an animated, eased window.
    fill = comp.add_clip(stripes_clip(), "stripes")
    window = comp.add_solid("window", color=(255, 255, 255))
    start = shapes.rounded_rect((100, 300), (120, 60), 20)
    end = shapes.rounded_rect((540, 300), (120, 60), 20)
    window.masks = [Mask(eased([0.0, DURATION - 0.5], [start, end]), feather=(8, 8))]
    fill.set_track_matte(window, "luma")

    # Parenting: the badge pre-comp follows an eased null and spins.
    rig = comp.add_null("rig")
    rig.transform = Transform(
        position=eased([0.0, 2.0, 4.0], [(470, 120), (520, 150), (470, 120)]),
        scale=eased([0.0, 2.0, 4.0], [(80, 80), (110, 110), (80, 80)]),
    )
    badge = comp.add_comp(badge_comp(), "badge")
    badge.parent = rig
    badge.transform = Transform(position=(0.0, 0.0))
    # Effects: the badge focuses in and leaves an echo trail while it spins.
    badge.effects = [
        ae.fx.GaussianBlur(blurriness=eased([0.0, 1.2], [16.0, 0.0])),
        ae.fx.Echo(
            echo_time=-0.08,
            number_of_echoes=4,
            starting_intensity=1.0,
            decay=0.55,
            echo_operator="composite_in_back",
        ),
    ]

    # Dissolve: a solid that grains in over the first two seconds.
    grain = comp.add_solid("grain", color=(255, 255, 255), size=(160, 70))
    grain.transform = Transform(
        position=(110, 70),
        opacity=Property(0.0, keyframes=[(0.0, 0.0), (2.0, 100.0)]),
    )
    grain.masks = [Mask.rounded_rect((79.5, 34.5), (160, 70), 18)]
    grain.blend_mode = "dissolve"
    grain.effects.add(
        ae.fx.Fill(color=(255, 230, 120), opacity=[(2.0, 0.0), (3.5, 100.0)])
    )

    # Adjustment layer: darken and warm everything outside a soft ellipse.
    vignette = comp.add_adjustment("vignette")
    vignette.masks = [
        Mask.ellipse((319.5, 179.5), (600, 380), feather=(140, 140), inverted=True)
    ]
    vignette.effects = [
        ae.fx.BrightnessContrast(brightness=-60, contrast=10),
        ae.fx.Tint(
            map_black_to=(30, 0, 50), map_white_to=(255, 225, 190), amount_to_tint=35
        ),
    ]
    return comp


def main(output="ae_demo.mp4"):
    """Render the demo to ``output``."""
    comp = build_scene()
    comp.write_videofile(output, fps=FPS, codec="libx264", audio=False, logger=None)


if __name__ == "__main__":
    main(*sys.argv[1:2])
