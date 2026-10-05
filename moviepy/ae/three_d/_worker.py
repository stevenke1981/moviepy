"""Packaged Blender worker; run only in Blender's Python interpreter."""

import json
import math
import sys
from pathlib import Path


def _aim(obj, target):
    obj.rotation_euler = (
        (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


def _materials(definitions):
    result = {}
    for name, definition in definitions.items():
        material = bpy.data.materials.new(name)
        material.use_nodes = True
        nodes, links = material.node_tree.nodes, material.node_tree.links
        surface = nodes.get("Principled BSDF")
        surface.inputs["Base Color"].default_value = definition["base_color"]
        surface.inputs["Alpha"].default_value = definition["base_color"][3]
        for field in ("Metallic", "Roughness"):
            surface.inputs[field].default_value = definition[field.lower()]
        surface.inputs["Emission Color"].default_value = (
            *definition["emission_color"],
            1,
        )
        surface.inputs["Emission Strength"].default_value = definition[
            "emission_strength"
        ]
        _maps(nodes, links, surface, definition)
        result[name] = material
    return result


def _maps(nodes, links, surface, definition):
    textures = {}
    inputs = {
        "base_color": "Base Color",
        "metallic": "Metallic",
        "roughness": "Roughness",
        "emission": "Emission Color",
        "opacity": "Alpha",
    }
    for field, path in definition["maps"].items():
        texture = nodes.new("ShaderNodeTexImage")
        # Separate datablocks allow the same file to have distinct color roles.
        texture.image = bpy.data.images.load(path, check_existing=False)
        texture.image.colorspace_settings.name = (
            "sRGB" if field in ("base_color", "emission") else "Non-Color"
        )
        textures[field] = texture.outputs["Color"]
        if field in inputs:
            links.new(textures[field], surface.inputs[inputs[field]])
    normal = None
    if "normal" in textures:
        source = textures["normal"]
        if definition["normal_convention"] == "directx":
            multiply = nodes.new("ShaderNodeVectorMath")
            multiply.operation = "MULTIPLY"
            multiply.inputs[1].default_value = (1, -1, 1)
            links.new(source, multiply.inputs[0])
            add = nodes.new("ShaderNodeVectorMath")
            add.operation = "ADD"
            add.inputs[1].default_value = (0, 1, 0)
            links.new(multiply.outputs[0], add.inputs[0])
            source = add.outputs[0]
        node = nodes.new("ShaderNodeNormalMap")
        links.new(source, node.inputs["Color"])
        normal = node.outputs["Normal"]
    if "height" in textures:
        node = nodes.new("ShaderNodeBump")
        node.inputs["Distance"].default_value = definition["height_scale"]
        links.new(textures["height"], node.inputs["Height"])
        if normal is not None:
            links.new(normal, node.inputs["Normal"])
        normal = node.outputs["Normal"]
    if normal is not None:
        links.new(normal, surface.inputs["Normal"])


def _objects(definitions, materials, fps):
    for index, definition in enumerate(definitions):
        kind = definition["type"]
        if kind == "cube":
            bpy.ops.mesh.primitive_cube_add()
        elif kind == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16)
            bpy.ops.object.shade_smooth()
        else:
            bpy.ops.mesh.primitive_plane_add(size=2)
        obj = bpy.context.object
        obj.name = definition.get("name", f"{kind}_{index}")
        obj.location, obj.scale = definition["location"], definition["scale"]
        obj.rotation_euler = [math.radians(v) for v in definition["rotation"]]
        if definition.get("material") is not None:
            obj.data.materials.append(materials[definition["material"]])
        for key in definition["animation"]:
            frame = 1 + key["time"] * fps
            for name in ("location", "rotation", "scale"):
                if name in key:
                    attribute = "rotation_euler" if name == "rotation" else name
                    value = (
                        [math.radians(v) for v in key[name]]
                        if name == "rotation"
                        else key[name]
                    )
                    setattr(obj, attribute, value)
                    obj.keyframe_insert(attribute, frame=frame)


def _lights(definitions):
    for index, definition in enumerate(definitions):
        data = bpy.data.lights.new(f"light_{index}", definition["type"].upper())
        data.energy, data.color = definition["energy"], definition["color"]
        if definition["type"] == "area":
            data.size = definition["size"]
        elif definition["type"] == "point":
            data.shadow_soft_size = definition["size"] / 2
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = definition["location"]
        _aim(obj, definition["target"])


def render(spec, output):
    """Build one isolated Cycles scene and save explicit display RGBA frames."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples, scene.cycles.seed = spec["samples"], spec["seed"]
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = spec["size"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = spec["transparent"]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.fps = max(1, round(spec["fps"]))
    scene.render.fps_base = scene.render.fps / spec["fps"]
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = spec[
        "world_strength"
    ]
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    materials = _materials(spec["materials"])
    _objects(spec["objects"], materials, spec["fps"])
    _lights(spec["lights"])
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location, camera_data.lens = (
        spec["camera"]["location"],
        spec["camera"]["lens"],
    )
    _aim(camera, spec["camera"]["target"])
    scene.camera = camera
    for frame in range(1, math.ceil(spec["duration"] * spec["fps"]) + 1):
        scene.frame_set(frame)
        scene.render.filepath = str(output / f"frame_{frame:06d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    import bpy
    from mathutils import Vector

    args = sys.argv[sys.argv.index("--") + 1 :]
    render(json.loads(Path(args[0]).read_text(encoding="utf-8")), Path(args[1]))
