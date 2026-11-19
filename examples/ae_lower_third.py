"""M1 example: an animated "lower third" title built with ``moviepy.ae``.

A bar wipes in through an eased, animated mask; the title slides in behind
a track matte with a blurred drop shadow; an adjustment layer softens and
darkens the footage behind the band. Run it with an output path::

    python examples/ae_lower_third.py lower_third.mp4
"""

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import moviepy.ae as ae
from moviepy import ImageClip, VideoClip
from moviepy.ae import Composition, Ease, Keyframe, Mask, Property, Transform
from moviepy.ae.masks import path as shapes


WIDTH, HEIGHT, FPS, DURATION = 640, 360, 24, 4.0
BAR_LEFT, BAR_TOP, BAR_WIDTH, BAR_HEIGHT = 40, 262, 420, 64


def eased(times, values):
    """Return a Property with ease-in-out keyframes."""
    curve = Ease.ease_in_out()
    keys = [
        Keyframe(t, v, in_ease=curve, out_ease=curve) for t, v in zip(times, values)
    ]
    return Property(values[0], keyframes=keys)


def text_clip(text, size, color=(255, 255, 255)):
    """Render ``text`` with Pillow's bundled font into a transparent clip."""
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no scalable default font
        font = ImageFont.load_default()
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("RGBA", (right - left + 4, bottom - top + 4), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((2 - left, 2 - top), text, font=font, fill=color)
    rgba = np.asarray(image)
    clip = ImageClip(rgba[..., :3]).with_duration(DURATION)
    mask = ImageClip(rgba[..., 3] / 255.0, is_mask=True).with_duration(DURATION)
    return clip.with_mask(mask)


def footage_clip():
    """Return synthetic "footage": drifting diagonal color bands."""
    x = np.arange(WIDTH)[None, :]
    y = np.arange(HEIGHT)[:, None]

    def frame_function(t):
        phase = (x * 0.6 + y + 40 * t) / 90.0
        red = 90 + 70 * np.sin(phase)
        green = 110 + 60 * np.sin(phase * 0.7 + 1.5)
        blue = 150 + 80 * np.cos(phase * 0.4)
        return np.dstack([red, green, blue]).astype(np.uint8)

    return VideoClip(frame_function, duration=DURATION).with_fps(FPS)


def bar_wipe():
    """Return the bar mask: a rectangle that grows from the left and back."""

    def box(width):
        center = (width / 2.0 - 0.5, BAR_HEIGHT / 2.0 - 0.5)
        return shapes.rect(center, (max(width, 1.0), BAR_HEIGHT))

    widths = [1.0, BAR_WIDTH, BAR_WIDTH, 1.0]
    return Mask(eased([0.2, 0.9, 3.2, 3.8], [box(w) for w in widths]))


def corner(position, **kwargs):
    """Return a Transform that places the layer's top-left corner."""
    return Transform(position=position, anchor_point=(0, 0), **kwargs)


def add_title(comp, text, size, top, window, delay):
    """Add a sliding title plus its shadow, both matted by ``window``."""
    start, end = (BAR_LEFT - 300, top), (BAR_LEFT + 22, top)
    motion = eased([0.5 + delay, 1.2 + delay, 3.1, 3.7], [start, end, end, start])
    shadow = comp.add_clip(text_clip(text, size), f"{text} shadow")
    shadow.transform = corner(
        eased(
            [0.5 + delay, 1.2 + delay, 3.1, 3.7],
            [(x + 3, y + 3) for x, y in (start, end, end, start)],
        ),
        opacity=60,
    )
    shadow.effects = [ae.fx.Fill(color=(0, 0, 0)), ae.fx.GaussianBlur(blurriness=4)]
    shadow.track_matte = ae.TrackMatte(window, "alpha")
    title = comp.add_clip(text_clip(text, size), text)
    title.transform = corner(motion)
    title.track_matte = ae.TrackMatte(window, "alpha")
    return title


def build_scene(title="KE SHENG DA", subtitle="After Effects-style compositing"):
    """Assemble the lower-third composition."""
    comp = Composition(
        size=(WIDTH, HEIGHT), fps=FPS, duration=DURATION, name="Lower third"
    )
    comp.add_clip(footage_clip(), "footage")

    focus = comp.add_adjustment("band focus")
    focus.masks = [Mask.rect((319.5, 293.5), (WIDTH, 120), feather=(0, 40))]
    focus.effects = [
        ae.fx.GaussianBlur(blurriness=eased([0.0, 0.8], [0.0, 6.0])),
        ae.fx.BrightnessContrast(brightness=eased([0.0, 0.8], [0.0, -45.0])),
    ]

    bar = comp.add_solid("bar", color=(20, 24, 38), size=(BAR_WIDTH, BAR_HEIGHT))
    bar.transform = corner((BAR_LEFT, BAR_TOP), opacity=88)
    bar.masks = [bar_wipe()]
    accent = comp.add_solid("accent", color=(255, 170, 30), size=(8, BAR_HEIGHT))
    accent.transform = corner(
        eased(
            [0.1, 0.6, 3.3, 3.9],
            [(-10, BAR_TOP), (BAR_LEFT, BAR_TOP), (BAR_LEFT, BAR_TOP), (-10, BAR_TOP)],
        )
    )

    # The matte window is the bar shape itself (hidden), so text slides out
    # from under the accent stripe and is clipped by the wiping bar.
    window = comp.add_solid(
        "text window", color=(255, 255, 255), size=(BAR_WIDTH - 12, BAR_HEIGHT)
    )
    window.transform = corner((BAR_LEFT + 12, BAR_TOP))
    window.masks = [bar_wipe()]
    window.enabled = False
    add_title(comp, title, 26, BAR_TOP + 7, window, 0.0)
    add_title(comp, subtitle, 15, BAR_TOP + 40, window, 0.15)
    comp.move_layer(comp.layer("accent"), 1)
    return comp


def main(output="ae_lower_third.mp4"):
    """Render the lower third to ``output``."""
    build_scene().write_videofile(
        output, fps=FPS, codec="libx264", audio=False, logger=None
    )


if __name__ == "__main__":
    main(*sys.argv[1:2])
