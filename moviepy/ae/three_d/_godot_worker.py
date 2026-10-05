"""Packaged GDScript source; it is data when imported by ordinary Python."""

# A stop needs a 512-sample mix and a subsequent main-thread cleanup. At the
# maximum 120 fps each movie frame mixes 400 samples; four frames cover both.
AUDIO_DRAIN_FRAMES = 4

GDSCRIPT = r'''
extends Node3D

var spec: Dictionary
var frame_index: int = 0
var bursts: Array = []
var animation_player: AnimationPlayer
var evidence: Dictionary = {}
var font_names: Array = []
var audio_player: AudioStreamPlayer
var audio_refs: Array = []

func vec(value) -> Vector3:
    return Vector3(value[0], value[1], value[2])

func rgb(value) -> Color:
    return Color(value[0], value[1], value[2], 1.0)

func material(data: Dictionary) -> Material:
    if data.grid:
        var shader = Shader.new()
        shader.code = """
shader_type spatial;
uniform vec4 paint : source_color;
uniform vec4 glow_color : source_color;
uniform float glow_energy;
uniform float metal;
uniform float rough;
void fragment() {
    vec2 cell = abs(fract(UV * 16.0) - 0.5);
    float line = smoothstep(0.465, 0.49, max(cell.x, cell.y));
    ALBEDO = paint.rgb * (1.0 + 0.6 * line);
    METALLIC = metal;
    ROUGHNESS = rough;
    EMISSION = glow_color.rgb * glow_energy + vec3(0.03, 0.18, 0.28) * line;
}
"""
        var grid = ShaderMaterial.new()
        grid.shader = shader
        grid.set_shader_parameter("paint", rgb(data.color))
        grid.set_shader_parameter("glow_color", rgb(data.emission))
        grid.set_shader_parameter("glow_energy", data.emission_energy)
        grid.set_shader_parameter("metal", data.metallic)
        grid.set_shader_parameter("rough", data.roughness)
        return grid
    var result = StandardMaterial3D.new()
    result.albedo_color = rgb(data.color)
    result.metallic = data.metallic
    result.roughness = data.roughness
    result.emission_enabled = data.emission_energy > 0
    result.emission = rgb(data.emission)
    result.emission_energy_multiplier = data.emission_energy
    return result

func make_mesh(item: Dictionary) -> Mesh:
    match item.type:
        "cube":
            return BoxMesh.new()
        "sphere":
            var sphere = SphereMesh.new()
            sphere.radius = 0.5
            sphere.height = 1.0
            return sphere
        "plane":
            return PlaneMesh.new()
        "cylinder":
            var cylinder = CylinderMesh.new()
            cylinder.height = 1.0
            cylinder.top_radius = 0.5
            cylinder.bottom_radius = 0.5
            return cylinder
        "torus":
            var torus = TorusMesh.new()
            torus.inner_radius = 0.92
            torus.outer_radius = 1.0
            return torus
        "text":
            var text = TextMesh.new()
            var font: Font = ThemeDB.fallback_font
            if item.has("font"):
                var custom = FontFile.new()
                if custom.load_dynamic_font(item.font) != OK:
                    push_error("Cannot load supplied font")
                    get_tree().quit(1)
                    return text
                custom.allow_system_fallback = false
                font = custom
            text.font = font
            text.text = item.text
            text.font_size = item.font_size
            text.pixel_size = item.pixel_size
            text.depth = item.depth
            if item.centered:
                text.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
            font_names.append(font.get_font_name())
            return text
    return null

func _ready():
    spec = JSON.parse_string(FileAccess.get_file_as_string("res://scene.json"))
    get_viewport().transparent_bg = spec.transparent
    var environment = Environment.new()
    environment.background_mode = Environment.BG_COLOR
    environment.background_color = Color(0.008, 0.013, 0.028)
    environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
    environment.ambient_light_color = Color(0.55, 0.67, 0.85)
    environment.ambient_light_energy = spec.ambient
    environment.tonemap_mode = Environment.TONE_MAPPER_FILMIC
    environment.glow_enabled = spec.glow > 0
    environment.glow_intensity = spec.glow
    var world = WorldEnvironment.new()
    world.environment = environment
    add_child(world)
    var camera = Camera3D.new()
    add_child(camera)
    camera.position = vec(spec.camera.location)
    camera.look_at(vec(spec.camera.target))
    camera.fov = spec.camera.fov
    camera.current = true

    var materials = {}
    for id in spec.materials:
        materials[id] = material(spec.materials[id])
    var animation = Animation.new()
    animation.length = spec.duration
    for index in range(spec.objects.size()):
        var item = spec.objects[index]
        var node = MeshInstance3D.new()
        node.name = "Object%d" % index
        node.mesh = make_mesh(item)
        add_child(node)
        node.position = vec(item.location)
        node.rotation_degrees = vec(item.rotation)
        node.scale = vec(item.scale)
        if item.has("material"):
            node.material_override = materials[item.material]
        for field in ["location", "rotation", "scale"]:
            var values = []
            for key in item.animation:
                if key.has(field):
                    values.append(key)
            if values.is_empty():
                continue
            var property_name = {"location": "position", "rotation": "rotation_degrees", "scale": "scale"}[field]
            var track = animation.add_track(Animation.TYPE_VALUE)
            animation.track_set_path(track, NodePath(str(node.name) + ":" + property_name))
            animation.track_insert_key(track, 0.0, vec(item[field]))
            for key in values:
                animation.track_insert_key(track, key.time, vec(key[field]))
    animation_player = AnimationPlayer.new()
    animation_player.process_mode = Node.PROCESS_MODE_DISABLED
    add_child(animation_player)
    var library = AnimationLibrary.new()
    library.add_animation("scene", animation)
    animation_player.add_animation_library("", library)
    animation_player.play("scene")
    animation_player.seek(0.0, true)

    for item in spec.lights:
        var light: Light3D
        match item.type:
            "directional":
                light = DirectionalLight3D.new()
                light.directional_shadow_max_distance = 30.0
            "omni":
                light = OmniLight3D.new()
                light.omni_range = item.range
            "spot":
                light = SpotLight3D.new()
                light.spot_range = item.range
                light.spot_angle = item.angle
        add_child(light)
        light.position = vec(item.location)
        if item.type != "omni":
            light.look_at(vec(item.target))
        light.light_color = rgb(item.color)
        light.light_energy = item.energy
        light.shadow_enabled = item.shadows

    for item in spec.particles:
        var particles = GPUParticles3D.new()
        particles.emitting = false
        particles.amount = item.count
        particles.one_shot = true
        particles.explosiveness = 1.0
        particles.lifetime = item.lifetime
        particles.fixed_fps = spec.fps
        particles.fract_delta = false
        particles.interpolate = false
        particles.use_fixed_seed = true
        particles.seed = item.seed
        var extent = item.speed[1] * item.lifetime + vec(item.gravity).length() * item.lifetime * item.lifetime + 1.0
        particles.visibility_aabb = AABB(Vector3.ONE * -extent, Vector3.ONE * extent * 2)
        particles.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
        var process_material = ParticleProcessMaterial.new()
        process_material.direction = vec(item.direction).normalized()
        process_material.spread = item.spread
        process_material.gravity = vec(item.gravity)
        process_material.initial_velocity_min = item.speed[0]
        process_material.initial_velocity_max = item.speed[1]
        process_material.scale_min = 0.6
        process_material.scale_max = 1.4
        particles.process_material = process_material
        var mesh = SphereMesh.new()
        mesh.radius = item.size
        mesh.height = item.size * 2
        mesh.radial_segments = 8
        mesh.rings = 4
        var spark = StandardMaterial3D.new()
        spark.albedo_color = rgb(item.color)
        spark.emission_enabled = true
        spark.emission = rgb(item.color)
        spark.emission_energy_multiplier = item.emission_energy
        mesh.material = spark
        particles.draw_pass_1 = mesh
        add_child(particles)
        particles.position = vec(item.location)
        bursts.append({"node": particles, "start": item.start, "triggered": false})

    if spec.has("audio"):
        var stream = AudioStreamWAV.load_from_file(spec.audio)
        if stream == null:
            push_error("Cannot load PCM WAV audio")
            get_tree().quit(1)
            return
        var audio = AudioStreamPlayer.new()
        audio_player = audio
        audio.stream = stream
        audio_refs.append(weakref(stream))
        add_child(audio)
        audio.play()
    evidence = {
        "version": Engine.get_version_info().string,
        "gpu": RenderingServer.get_video_adapter_name(),
        "renderer": RenderingServer.get_current_rendering_method(),
        "driver": RenderingServer.get_current_rendering_driver_name(),
        "user_data_dir": OS.get_user_data_dir(),
        "fonts": font_names,
        "render_size": [get_viewport().size.x, get_viewport().size.y],
        "processed_frames": 0,
        "audio_cleanup_frames": 0,
        "audio_released": not spec.has("audio"),
        "burst_start_frames": []
    }

func save_evidence():
    var file = FileAccess.open("res://runtime.json", FileAccess.WRITE)
    file.store_string(JSON.stringify(evidence, "  "))

func _process(_delta):
    var content_frames = int(ceil(spec.duration * spec.fps))
    if frame_index >= content_frames:
        # The previous movie iteration already wrote the last content samples.
        # AudioServer defers releasing playback until it mixes after stop().
        if frame_index == content_frames:
            if audio_player.has_stream_playback():
                audio_refs.append(weakref(audio_player.get_stream_playback()))
            audio_player.stop()
            audio_player.stream = null
            audio_player.queue_free()
            audio_player = null
        frame_index += 1
        evidence.audio_cleanup_frames = frame_index - content_frames
        if evidence.audio_cleanup_frames == __AUDIO_DRAIN_FRAMES__:
            evidence.audio_released = true
            for ref in audio_refs:
                if ref.get_ref() != null:
                    evidence.audio_released = false
            save_evidence()
            if not evidence.audio_released:
                push_error("Godot audio resources did not finish releasing")
                get_tree().quit(1)
        return
    var t = float(frame_index) / spec.fps
    animation_player.seek(t, true)
    for burst in bursts:
        if not burst.triggered and t + 0.000000001 >= burst.start:
            burst.node.emitting = true
            burst.node.restart(true)
            burst.triggered = true
            evidence.burst_start_frames.append(frame_index)
    frame_index += 1
    if frame_index == content_frames:
        evidence.processed_frames = frame_index
        if not spec.has("audio"):
            save_evidence()

func _exit_tree():
    if is_instance_valid(audio_player):
        audio_player.stop()
        audio_player.stream = null
'''.replace(
    "__AUDIO_DRAIN_FRAMES__", str(AUDIO_DRAIN_FRAMES)
)
