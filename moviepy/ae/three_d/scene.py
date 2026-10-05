"""Validate the portable JSON subset supported by the Blender backend."""

import copy
import math
from pathlib import Path

from moviepy.ae._geometry import finite_real, validate_pixel_size


MAPS = {
    "base_color",
    "metallic",
    "roughness",
    "normal",
    "height",
    "emission",
    "opacity",
}


def number(value, label, low=None, high=None):
    """Validate a finite scalar in an optional inclusive range."""
    value = finite_real(value, label)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{label} is outside its supported range")
    return value


def vector(value, label, count=3, low=None, high=None):
    """Validate a fixed-length numeric vector."""
    result = [number(v, label, low, high) for v in value]
    if len(result) != count:
        raise ValueError(f"{label} must contain {count} values")
    return result


def keys(value, allowed, label):
    """Reject typos and unsupported capabilities rather than ignoring them."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    unknown = set(value) - set(allowed.split())
    if unknown:
        raise ValueError(f"unsupported {label} fields: {sorted(unknown)}")


def _material(material):
    keys(
        material,
        "base_color metallic roughness emission_color emission_strength maps normal_convention height_scale",
        "material",
    )
    material["base_color"] = vector(
        material.get("base_color", (0.7, 0.7, 0.7, 1)), "base_color", 4, 0, 1
    )
    for field, default in (("metallic", 0), ("roughness", 0.4)):
        material[field] = number(material.get(field, default), field, 0, 1)
    material["emission_color"] = vector(
        material.get("emission_color", (0, 0, 0)), "emission_color", low=0
    )
    material["emission_strength"] = number(
        material.get("emission_strength", 1), "emission_strength", 0
    )
    material["height_scale"] = number(
        material.get("height_scale", 0.05), "height_scale", 0
    )
    material["normal_convention"] = material.get("normal_convention", "opengl")
    if material["normal_convention"] not in ("opengl", "directx"):
        raise ValueError("normal_convention must be opengl or directx")
    maps = material.setdefault("maps", {})
    if not isinstance(maps, dict) or set(maps) - MAPS:
        raise ValueError(
            f"maps must use only {sorted(MAPS)}; native SBSAR is unsupported"
        )
    for name, filename in maps.items():
        path = Path(filename).expanduser().resolve()
        if path.suffix.lower() == ".sbsar":
            raise ValueError("native SBSAR is unsupported; export PBR image maps first")
        if not path.is_file():
            raise FileNotFoundError(path)
        maps[name] = str(path)


def _transform(item):
    for name, default in (
        ("location", (0, 0, 0)),
        ("rotation", (0, 0, 0)),
        ("scale", (1, 1, 1)),
    ):
        if name in item:
            item[name] = vector(item[name], name)
        elif default is not None:
            item[name] = list(default)
    if any(value == 0 for value in item["scale"]):
        raise ValueError("3D scale components must be nonzero")


def _object(item, materials, duration):
    keys(item, "type name location rotation scale material animation", "object")
    if item.get("type") not in ("cube", "sphere", "plane"):
        raise ValueError("object type must be cube, sphere or plane")
    if "name" in item and (not isinstance(item["name"], str) or not item["name"]):
        raise ValueError("object name must be a nonempty string")
    if item.get("material") is not None and item["material"] not in materials:
        raise ValueError("object refers to an unknown material")
    _transform(item)
    animation = item.setdefault("animation", [])
    if not isinstance(animation, list):
        raise TypeError("animation must be a list")
    times = set()
    for key in animation:
        keys(key, "time location rotation scale", "animation key")
        if "time" not in key:
            raise ValueError("animation key requires time")
        key["time"] = number(key["time"], "key time", 0, duration)
        if key["time"] in times:
            raise ValueError("duplicate animation key time")
        times.add(key["time"])
        for name in ("location", "rotation", "scale"):
            if name in key:
                key[name] = vector(key[name], name)
                if name == "scale" and 0 in key[name]:
                    raise ValueError("animated scale must be nonzero")


def validate_scene(value):
    """Return an independent normalized scene; all paths are explicit and local."""
    keys(
        value,
        "size fps duration samples transparent camera materials objects lights world_strength seed",
        "scene",
    )
    scene = copy.deepcopy(value)
    scene["size"] = list(validate_pixel_size(scene.get("size", (320, 180))))
    scene["fps"] = number(scene.get("fps", 24), "fps", 0.001, 240)
    scene["duration"] = number(scene.get("duration", 1), "duration", 0.001, 3600)
    scene["samples"] = number(scene.get("samples", 16), "samples", 1, 4096)
    scene["seed"] = number(scene.get("seed", 0), "seed", 0, 2147483647)
    for name in ("samples", "seed"):
        if scene[name] != int(scene[name]):
            raise ValueError(f"{name} must be integral")
        scene[name] = int(scene[name])
    if math.ceil(scene["fps"] * scene["duration"]) > 100000:
        raise ValueError("scene exceeds 100000 frames")
    scene["transparent"] = scene.get("transparent", True)
    if not isinstance(scene["transparent"], bool):
        raise TypeError("transparent must be bool")
    scene["world_strength"] = number(
        scene.get("world_strength", 0.15), "world_strength", 0
    )
    _camera(scene.setdefault("camera", {}))
    materials = scene.setdefault("materials", {})
    if not isinstance(materials, dict):
        raise TypeError("materials must be an object")
    if any(not isinstance(name, str) or not name for name in materials):
        raise ValueError("material names must be nonempty strings")
    for material in materials.values():
        _material(material)
    objects = scene.setdefault("objects", [])
    if not isinstance(objects, list):
        raise TypeError("objects must be a list")
    for item in objects:
        _object(item, materials, scene["duration"])
    lights = scene.setdefault(
        "lights", [{"type": "area", "location": [2, -3, 5], "energy": 600, "size": 4}]
    )
    if not isinstance(lights, list):
        raise TypeError("lights must be a list")
    for light in lights:
        _light(light)
    return scene


def _camera(camera):
    keys(camera, "location target lens", "camera")
    camera["location"] = vector(camera.get("location", (4, -6, 3)), "camera location")
    camera["target"] = vector(camera.get("target", (0, 0, 0)), "camera target")
    camera["lens"] = number(camera.get("lens", 50), "camera lens", 1, 1000)
    if camera["location"] == camera["target"]:
        raise ValueError("camera location must differ from target")


def _light(light):
    keys(light, "type location target color energy size", "light")
    if light.get("type") not in ("area", "point", "sun"):
        raise ValueError("light type must be area, point or sun")
    light["location"] = vector(light.get("location", (2, -3, 5)), "light location")
    light["target"] = vector(light.get("target", (0, 0, 0)), "light target")
    if light["location"] == light["target"]:
        raise ValueError("light location must differ from target")
    light["color"] = vector(light.get("color", (1, 1, 1)), "light color", low=0, high=1)
    light["energy"] = number(light.get("energy", 500), "light energy", 0)
    light["size"] = number(light.get("size", 2), "light size", 0.001)
