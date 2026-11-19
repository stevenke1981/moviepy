"""Build real 3D title cards and event-driven GPU particle scenes."""

import math

from moviepy.ae.motion.captions import check_glyphs, load_font


def build_godot_motion_scene(request):
    """Translate a normalized motion request into the existing Godot schema."""
    width, height, fps = request["width"], request["height"], request["fps"]
    duration = request["frame_count"] / fps
    # The shared contract counts frames. Avoid an accidental extra ceil frame
    # from e.g. 7 / 25 * 25 == 7.000000000000001 in the duration-based backend.
    while math.ceil(duration * fps) > request["frame_count"]:
        duration = math.nextafter(duration, 0)
    params = request["params"]
    scene = {
        "size": [width, height],
        "fps": fps,
        "duration": duration,
        "camera": {"location": [0, 0, 8], "target": [0, 0, 0], "fov": 35},
        "ambient": 0.45,
    }
    if request["effect"] == "beat_particles":
        scene["particles"] = [
            {
                **{
                    name: params[name]
                    for name in (
                        "lifetime",
                        "speed",
                        "spread",
                        "gravity",
                        "size",
                        "color",
                        "location",
                    )
                },
                "count": max(1, round(params["count"] * event["strength"])),
                "start": event["frame"] / fps,
                "seed": (request["seed"] + 104729 * (event["frame"] + 1)) % 2147483648,
                "emission_energy": 3,
            }
            for event in request["audio_events"]
        ]
        return scene
    if request["effect"] != "title_card":
        raise ValueError("only title_card and beat_particles use Godot")
    text = request["cues"][0]["text"]
    font = load_font(request, 64)
    check_glyphs(font, text)
    lines = text.split("\n")
    advance = max(float(font.getlength(line)) for line in lines)
    ascent, descent = font.getmetrics()
    visible_height = 16 * math.tan(math.radians(35) / 2)
    # Reserve space for perspective rotation, extrusion and differing glyph
    # metrics in Pillow and Godot; the camera and font remain explicit.
    available = visible_height * (1 - 2 * params["safe_margin"]) * 0.7
    pixel_size = min(
        available * width / height / max(1, advance),
        available / ((ascent + descent) * len(lines)),
        0.05,
    )
    turn = params["turn_degrees"]
    node = {
        "type": "text",
        "text": text,
        "material": "title",
        "font_size": 64,
        "pixel_size": pixel_size,
        "depth": params["depth"],
        "centered": True,
        "animation": [
            {"time": 0, "rotation": [0, -turn, 0], "scale": [0.72] * 3},
            {"time": duration * 0.35, "rotation": [0, 0, 0], "scale": [1] * 3},
            {"time": duration, "rotation": [0, turn * 0.5, 0], "scale": [1] * 3},
        ],
    }
    if "font" in request["assets"]:
        node["font"] = request["assets"]["font"]
    scene.update(
        {
            "materials": {
                "title": {"color": params["color"], "metallic": 0.45, "roughness": 0.22}
            },
            "objects": [node],
            "lights": [
                {"type": "directional", "location": [-3, 5, 4], "energy": 2.2},
                {
                    "type": "omni",
                    "location": [3, 1, 2],
                    "energy": 3,
                    "color": [0.4, 0.5, 1],
                },
            ],
        }
    )
    return scene
