"""Validate the deliberately small, data-only Godot scene format."""

import copy
import math
import wave
from pathlib import Path

from moviepy.ae._geometry import validate_pixel_size
from moviepy.ae.three_d.scene import keys, number, vector


def _integer(value, label, low, high):
    value = number(value, label, low, high)
    if value != int(value):
        raise ValueError(f"{label} must be integral")
    return int(value)


def _bool(value, label):
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be bool")
    return value


def _path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.as_posix()


def _look_at(item):
    item["location"] = vector(item.get("location", (0, 3, 8)), "location")
    item["target"] = vector(item.get("target", (0, 0, 0)), "target")
    if item["location"][0::2] == item["target"][0::2]:
        raise ValueError("look-at direction must not be parallel to the Y up axis")


def _material(item):
    keys(item, "color metallic roughness emission emission_energy grid", "material")
    item["color"] = vector(item.get("color", (0.7, 0.7, 0.7)), "color", low=0, high=1)
    item["emission"] = vector(
        item.get("emission", (0, 0, 0)), "emission", low=0, high=1
    )
    for field, default, maximum in (
        ("metallic", 0, 1),
        ("roughness", 0.4, 1),
        ("emission_energy", 0, 32),
    ):
        item[field] = number(item.get(field, default), field, 0, maximum)
    item["grid"] = _bool(item.get("grid", False), "grid")


def _object(item, scene):
    keys(
        item,
        "type location rotation scale material animation text font font_size pixel_size depth centered",
        "object",
    )
    if item.get("type") not in ("cube", "sphere", "plane", "cylinder", "torus", "text"):
        raise ValueError("unsupported Godot object type")
    if "material" in item and item["material"] not in scene["materials"]:
        raise ValueError("object refers to an unknown material")
    for field, default in (
        ("location", (0, 0, 0)),
        ("rotation", (0, 0, 0)),
        ("scale", (1, 1, 1)),
    ):
        item[field] = vector(item.get(field, default), field)
    if min(item["scale"]) <= 0:
        raise ValueError("Godot scale components must be positive")
    text_fields = {"text", "font", "font_size", "pixel_size", "depth", "centered"}
    if item["type"] == "text":
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError("text mesh requires nonempty text")
        if len(item["text"]) > 2048:
            raise ValueError("text exceeds 2048 characters")
        if "font" in item:
            item["font"] = _path(item["font"])
        item["font_size"] = _integer(item.get("font_size", 64), "font_size", 8, 512)
        item["pixel_size"] = number(
            item.get("pixel_size", 0.02), "pixel_size", 0.0001, 1
        )
        item["depth"] = number(item.get("depth", 0.15), "depth", 0, 10)
        item["centered"] = _bool(item.get("centered", False), "centered")
    elif set(item) & text_fields:
        raise ValueError("font/text fields require a text mesh")
    animation = item.setdefault("animation", [])
    if not isinstance(animation, list):
        raise TypeError("animation must be a list")
    times = set()
    for key in animation:
        keys(key, "time location rotation scale", "animation key")
        if "time" not in key or len(key) == 1:
            raise ValueError("animation keys require time and a transform")
        key["time"] = number(key["time"], "key time", 0, scene["duration"])
        if key["time"] in times:
            raise ValueError("duplicate animation key time")
        times.add(key["time"])
        for field in ("location", "rotation", "scale"):
            if field in key:
                key[field] = vector(key[field], field)
                if field == "scale" and min(key[field]) <= 0:
                    raise ValueError("animated scale must be positive")
    animation.sort(key=lambda key: key["time"])


def _light(item):
    keys(item, "type location target color energy range angle shadows", "light")
    if item.get("type") not in ("directional", "omni", "spot"):
        raise ValueError("light type must be directional, omni or spot")
    if item["type"] == "omni":
        item["location"] = vector(item.get("location", (0, 3, 0)), "location")
        item["target"] = vector(item.get("target", (0, 0, 0)), "target")
    else:
        _look_at(item)
    item["color"] = vector(item.get("color", (1, 1, 1)), "color", low=0, high=1)
    item["energy"] = number(item.get("energy", 2), "energy", 0, 100)
    item["range"] = number(item.get("range", 20), "range", 0.01, 10000)
    item["angle"] = number(item.get("angle", 45), "angle", 1, 89)
    item["shadows"] = _bool(item.get("shadows", True), "shadows")


def _particles(item, scene, index):
    keys(
        item,
        "location count start lifetime speed direction spread gravity size color emission_energy seed",
        "particles",
    )
    item["location"] = vector(item.get("location", (0, 1, 0)), "location")
    item["count"] = _integer(item.get("count", 400), "count", 1, 50000)
    item["seed"] = _integer(item.get("seed", index), "seed", 0, 2147483647)
    item["start"] = number(item.get("start", 0), "start", 0, scene["duration"])
    if item["start"] >= scene["duration"]:
        raise ValueError("particle start must precede the end of the scene")
    item["lifetime"] = number(item.get("lifetime", 2), "lifetime", 0.01, 3600)
    item["speed"] = vector(item.get("speed", (1, 3)), "speed", 2, 0, 1000)
    if item["speed"][0] > item["speed"][1]:
        raise ValueError("particle speed range is reversed")
    item["direction"] = vector(item.get("direction", (0, 1, 0)), "direction")
    if not any(item["direction"]):
        raise ValueError("particle direction must be nonzero")
    item["spread"] = number(item.get("spread", 80), "spread", 0, 180)
    item["gravity"] = vector(item.get("gravity", (0, -2, 0)), "gravity")
    item["size"] = number(item.get("size", 0.025), "particle size", 0.001, 10)
    item["color"] = vector(item.get("color", (1, 0.45, 0.06)), "color", low=0, high=1)
    item["emission_energy"] = number(
        item.get("emission_energy", 4), "emission_energy", 0, 32
    )


def validate_godot_scene(value):
    """Normalize an independent Y-up scene without executing supplied code."""
    keys(
        value,
        "size fps duration transparent camera materials objects lights particles ambient glow audio",
        "Godot scene",
    )
    scene = copy.deepcopy(value)
    scene["size"] = list(validate_pixel_size(scene.get("size", (640, 360))))
    scene["fps"] = _integer(scene.get("fps", 24), "fps", 1, 120)
    if 48000 % scene["fps"]:
        raise ValueError("fps must divide the 48000 Hz Movie Maker sample rate")
    scene["duration"] = number(scene.get("duration", 2), "duration", 0.001, 3600)
    if math.ceil(scene["fps"] * scene["duration"]) > 100000:
        raise ValueError("scene exceeds 100000 frames")
    scene["transparent"] = _bool(scene.get("transparent", True), "transparent")
    scene["ambient"] = number(scene.get("ambient", 0.25), "ambient", 0, 4)
    scene["glow"] = number(scene.get("glow", 0), "glow", 0, 2)
    if scene["transparent"] and scene["glow"]:
        raise ValueError(
            "use glow on an opaque scene, or apply 2D glow after RGBA compositing"
        )
    camera = scene.setdefault("camera", {})
    keys(camera, "location target fov", "camera")
    _look_at(camera)
    camera["fov"] = number(camera.get("fov", 45), "fov", 1, 179)
    materials = scene.setdefault("materials", {})
    if not isinstance(materials, dict):
        raise TypeError("materials must be an object")
    for name, item in materials.items():
        if not isinstance(name, str) or not name:
            raise ValueError("material names must be nonempty strings")
        _material(item)
    for name in ("objects", "lights", "particles"):
        scene.setdefault(name, [])
        if not isinstance(scene[name], list):
            raise TypeError(f"{name} must be a list")
    for item in scene["objects"]:
        _object(item, scene)
    for item in scene["lights"]:
        _light(item)
    for index, item in enumerate(scene["particles"]):
        _particles(item, scene, index)
    if "audio" in scene:
        scene["audio"] = _path(scene["audio"])
        with wave.open(scene["audio"], "rb") as audio:
            if audio.getsampwidth() != 2 or audio.getnchannels() not in (1, 2):
                raise ValueError("audio must be a mono/stereo 16-bit PCM WAV")
    return scene
