"""M1 example: a "logo reveal" built with ``moviepy.ae``.

The logo pre-comp scales and spins in while a keyframed Gaussian Blur
focuses it and an Echo leaves a trail; a blurred Screen copy adds glow; a
light sweep passes over the logo through an alpha track matte with the Add
mode; an adjustment layer adds a vignette. Run it with an output path::

    python examples/ae_logo_reveal.py logo_reveal.mp4
"""

import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import moviepy.ae as ae
from moviepy import ImageClip
from moviepy.ae import Composition, Ease, Keyframe, Mask, Property, Transform
from moviepy.ae.masks import path as shapes


WIDTH, HEIGHT, FPS, DURATION = 640, 360, 24, 4.0
CENTER = (WIDTH / 2.0, HEIGHT / 2.0 - 10)


def eased(times, values, curve=None):
    """Return a Property whose keyframes share one ease curve."""
    curve = Ease.ease_out() if curve is None else curve
    keys = [
        Keyframe(t, v, in_ease=curve, out_ease=curve) for t, v in zip(times, values)
    ]
    return Property(values[0], keyframes=keys)


def wordmark(text="MOVIEPY  AE", size=30):
    """Render a transparent text clip with Pillow's bundled font."""
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no scalable default font
        font = ImageFont.load_default()
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("RGBA", (right - left + 4, bottom - top + 4), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((2 - left, 2 - top), text, font=font, fill="white")
    rgba = np.asarray(image)
    clip = ImageClip(rgba[..., :3]).with_duration(DURATION)
    return clip.with_mask(ImageClip(rgba[..., 3] / 255.0, is_mask=True))


def logo_comp():
    """Return the transparent logo pre-comp: ring, spinning star, wordmark."""
    logo = Composition(
        size=(260, 220), fps=FPS, duration=DURATION, transparent=True, name="Logo"
    )
    ring = logo.add_solid("ring", color=(255, 255, 255), size=(260, 220))
    ring.masks = [
        Mask.ellipse((129.5, 79.5), (150, 150)),
        Mask.ellipse((129.5, 79.5), (126, 126), mode="subtract"),
    ]
    star = logo.add_solid("star", color=(255, 170, 30), size=(160, 160))
    star.masks = [Mask.star((79.5, 79.5), 5, 58, 24, feather=1.5)]
    star.transform = Transform(
        position=(129.5, 79.5),
        rotation=eased([0.0, 1.6], [-144.0, 0.0]),
    )
    text = logo.add_clip(wordmark(), "wordmark")
    text.transform = Transform(position=(130, 190), opacity=eased([1.2, 2.0], [0, 100]))
    return logo


def add_logo(comp, name):
    """Add one instance of the animated logo pre-comp."""
    layer = comp.add_comp(logo_comp(), name)
    layer.transform = Transform(
        position=CENTER,
        scale=eased([0.0, 1.0, 1.4], [(30, 30), (108, 108), (100, 100)]),
        opacity=eased([0.0, 0.5], [0.0, 100.0]),
    )
    return layer


def light_sweep():
    """Return a diagonal soft band that travels left to right."""

    def band(x):
        return shapes.polygon([(x, -40), (x + 50, -40), (x - 70, 260), (x - 120, 260)])

    path = Property(band(-80), keyframes=[(1.8, band(-80)), (3.0, band(420))])
    return Mask(path, feather=(18, 18))


def build_scene():
    """Assemble the logo reveal composition."""
    comp = Composition(
        size=(WIDTH, HEIGHT),
        fps=FPS,
        duration=DURATION,
        name="Logo reveal",
        bg_color=(12, 14, 26),
    )
    backdrop = comp.add_solid("backdrop", color=(40, 46, 90))
    backdrop.masks = [Mask.ellipse(CENTER, (520, 360), feather=(160, 160))]

    glow = add_logo(comp, "logo glow")
    glow.blend_mode = "screen"
    glow.effects = [
        ae.fx.GaussianBlur(blurriness=18),
        ae.fx.Fill(color=(255, 190, 90), opacity=70),
    ]
    glow.transform.opacity = eased([0.8, 1.6, 4.0], [0.0, 90.0, 60.0])

    logo = add_logo(comp, "logo")
    logo.effects = [
        ae.fx.GaussianBlur(blurriness=eased([0.0, 1.2], [30.0, 0.0])),
        ae.fx.Echo(
            echo_time=-0.05,
            number_of_echoes=4,
            decay=0.5,
            echo_operator="composite_in_back",
        ),
    ]

    matte = add_logo(comp, "sweep matte")
    sweep = comp.add_solid("sweep", color=(255, 255, 255), size=(360, 220))
    sweep.transform = Transform(position=CENTER, opacity=80)
    sweep.masks = [light_sweep()]
    sweep.blend_mode = "add"
    sweep.set_track_matte(matte, "alpha")
    comp.move_layer(matte, 1)

    vignette = comp.add_adjustment("vignette")
    vignette.masks = [
        Mask.ellipse(CENTER, (560, 380), feather=(120, 120), inverted=True)
    ]
    vignette.effects = [ae.fx.BrightnessContrast(brightness=-50)]
    return comp


def main(output="ae_logo_reveal.mp4"):
    """Render the logo reveal to ``output``."""
    build_scene().write_videofile(
        output, fps=FPS, codec="libx264", audio=False, logger=None
    )


if __name__ == "__main__":
    main(*sys.argv[1:2])
