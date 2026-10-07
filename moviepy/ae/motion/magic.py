"""Small warm-light presets rendered by Godot and composited as native AE layers."""

import math
from dataclasses import dataclass

from moviepy.ae._geometry import validate_pixel_size
from moviepy.ae.blend.modes import BlendMode
from moviepy.ae.effects.stylize.glow import Glow
from moviepy.ae.layers.av import AVLayer
from moviepy.ae.properties.easing import Ease
from moviepy.ae.properties.keyframe import Keyframe
from moviepy.ae.properties.property import Property
from moviepy.ae.three_d._godot_scene import _integer, validate_godot_scene
from moviepy.ae.three_d.godot import render_godot_scene
from moviepy.ae.three_d.scene import number, vector
from moviepy.ae.transform import Transform


PRESETS = ("warm_aura", "flow_particles", "cast_ring")


def build_magic_scene(
    preset,
    *,
    size=(384, 384),
    fps=24,
    frame_count=48,
    seed=607,
    color=(1.0, 0.58, 0.16),
    emission=2.0,
    radius=0.85,
    particle_count=30,
    particle_size=0.03,
):
    """Build the existing Godot scene contract, with explicit baked timing.

    Geometry, color, emission, seed and particle count are render-time controls.
    Placement, opacity and AE Glow remain live Properties on ``to_layer()``.
    The scene contains no external script/shader, background or postprocess
    glow. RGBA coverage comes from the existing native transparent renderer.
    """
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    fps = _integer(fps, "fps", 1, 120)
    frame_count = _integer(frame_count, "frame_count", 3, 100000)
    size = validate_pixel_size(size)
    seed = _integer(seed, "seed", 0, 2147483647)
    particle_count = _integer(particle_count, "particle_count", 1, 5000)
    color = vector(color, "color", low=0, high=1)
    emission = number(emission, "emission", 0, 32)
    radius = number(radius, "radius", 0.05, 1.2)
    particle_size = number(particle_size, "particle_size", 0.001, 0.1)
    duration = frame_count / fps
    while math.ceil(duration * fps) > frame_count:
        duration = math.nextafter(duration, 0)
    last = (frame_count - 1) / fps
    scene = {
        "size": size,
        "fps": fps,
        "duration": duration,
        "transparent": True,
        "camera": {
            "location": (0, 0, 6 * max(1, size[1] / size[0])),
            "target": (0, 0, 0),
            "fov": 35,
        },
        "ambient": 0,
        "glow": 0,
        "materials": {
            "gold": {"color": color, "emission": color, "emission_energy": emission}
        },
    }
    if preset == "flow_particles":
        starts = sorted({0, frame_count // 6, frame_count // 3})[:particle_count]
        scene["particles"] = [
            {
                "count": particle_count // len(starts)
                + (index < particle_count % len(starts)),
                "start": frame / fps,
                "seed": (seed + 104729 * index) % 2147483648,
                "lifetime": duration * 0.85,
                "location": (-radius * 0.55, -radius * 0.65, 0),
                "direction": (0.25, 1, 0),
                "speed": (radius * 0.6, radius * 1.15),
                "spread": 42,
                "gravity": (0, 0.03, 0),
                "size": particle_size,
                "color": color,
                "emission_energy": emission,
            }
            for index, frame in enumerate(starts)
        ]
    else:
        start_scale, end_scale = (0.87, 1.0) if preset == "warm_aura" else (0.25, 1.12)
        scene["objects"] = [
            {
                "type": "torus",
                "material": "gold",
                "rotation": (76 if preset == "warm_aura" else 90, 0, -12),
                "scale": (radius * start_scale,) * 3,
                "animation": [
                    {"time": 0, "scale": (radius * start_scale,) * 3},
                    {
                        "time": last,
                        "scale": (radius * end_scale,) * 3,
                        "rotation": (90, 0, 12),
                    },
                ],
            }
        ]
    return validate_godot_scene(scene)


@dataclass(frozen=True)
class MagicRender:
    """Completed Godot footage with a live AE layer/effect-stack adapter."""

    render: object
    preset: str
    color: tuple

    def to_layer(
        self,
        name=None,
        *,
        position=None,
        scale=(100, 100),
        opacity=None,
        glow_radius=10.0,
        glow_intensity=0.7,
        blend_mode="normal",
        **layer_options,
    ):
        """Create a Layer with live Transform and Glow Properties.

        Native linear frames are used whenever present; no uint8 conversion
        precedes compositing. A linear working Composition preserves HDR with
        Normal/Add. Screen retains the core's documented SDR input clamp.
        The default opacity has zero coverage on first and last visible frames;
        an explicit opacity Property replaces that envelope. Timing, masks and
        track mattes use normal Layer semantics. Put a mask on an outer
        CompLayer when a protected face/text region must also exclude the halo.
        """
        mode = BlendMode.coerce(blend_mode)
        if mode not in (BlendMode.NORMAL, BlendMode.ADD, BlendMode.SCREEN):
            raise ValueError("magic layers support Normal, Add or Screen")
        if opacity is None:
            last = (len(self.render.frames) - 1) / self.render.fps
            opacity = Property(
                [
                    Keyframe(0, 0, out_ease=Ease.easy_ease()),
                    Keyframe(last * 0.18, 72),
                    Keyframe(last * 0.68, 72, out_ease=Ease.easy_ease()),
                    Keyframe(last, 0),
                ]
            )
        options = {
            "transform": Transform(
                position=position, scale=scale, opacity=opacity, interpolation="linear"
            ),
            "effects": [
                Glow(
                    radius=glow_radius,
                    intensity=glow_intensity,
                    threshold=25,
                    color=tuple(v * 255 for v in self.color),
                )
            ],
            "blend_mode": mode,
            **layer_options,
        }
        name = name or self.preset.replace("_", " ").title()
        if self.render.linear_frames:
            return self.render.to_linear_layer(name, **options)
        return AVLayer(self.render.to_clip(), name, audio_enabled=False, **options)


def render_magic(
    preset,
    output_directory,
    *,
    executable=None,
    timeout=600,
    linear_output=True,
    **parameters,
):
    """Bake one preset with the existing hidden Godot renderer, then adapt to AE.

    Output must be new/empty, as for ``render_godot_scene``. Fixed seed/time
    permits repeat rendering on the same engine/GPU; cross-driver bit identity
    is not promised. Random AE frame reads only access the completed sequence.
    The default retains native scene-linear HDR plus an explicit SDR PNG copy.
    Set ``linear_output=False`` only when that SDR source contract is sufficient.
    """
    scene = build_magic_scene(preset, **parameters)
    render = render_godot_scene(
        scene,
        output_directory,
        executable=executable,
        timeout=timeout,
        linear_output=linear_output,
    )
    return MagicRender(render, preset, tuple(scene["materials"]["gold"]["color"]))
