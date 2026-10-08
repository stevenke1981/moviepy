"""Check the additional magic presets at the scene-contract level (no Godot render)."""

import hashlib
import json
import math

import pytest

from moviepy.ae.motion.magic import PRESETS, build_magic_scene
from moviepy.ae.three_d._godot_scene import validate_godot_scene


NEW_PRESETS = ("spark_burst", "orbit_rings", "rising_embers", "pulse_orb")
PARTICLE_PRESETS = ("spark_burst", "rising_embers")
OBJECT_PRESETS = ("orbit_rings", "pulse_orb")
PARTICLE_COUNTS = (1, 3, 7, 12, 30)


def _digest(scene):
    return hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest()


def test_existing_presets_are_byte_identical_after_adding_new_ones():
    # Digests were recorded from the presets before the new branches existed.
    assert PRESETS[:3] == ("warm_aura", "flow_particles", "cast_ring")
    short = dict(fps=24, frame_count=7, seed=84, particle_count=7)
    expected = {
        ("warm_aura", "default"): (
            "113b72af16ef8e8d5afbe4b2491eaafadf2bd66b520e4a8d9ce884ba505faa02"
        ),
        ("flow_particles", "default"): (
            "71bfd8493483a69a3ef4adf76734137284e797aa515dd22ef8d7aef66ea586f9"
        ),
        ("cast_ring", "default"): (
            "3c03b63f6ef6fe14ed532c25f8faefee2b625017c0eb7eec6fbf6c17cbd16405"
        ),
        ("flow_particles", "short"): (
            "bb9bb1de20dca7b928bc7e865f5d07427c4d14135f3cbfd9d90f5b37f6d78907"
        ),
        ("cast_ring", "short"): (
            "cb4fdd5e26537dc56a64b4b9b1dd888f0c9e1bd4ca4af914d0932a63eac7b95c"
        ),
    }
    actual = {
        ("warm_aura", "default"): _digest(build_magic_scene("warm_aura")),
        ("flow_particles", "default"): _digest(build_magic_scene("flow_particles")),
        ("cast_ring", "default"): _digest(build_magic_scene("cast_ring")),
        ("flow_particles", "short"): _digest(
            build_magic_scene("flow_particles", **short)
        ),
        ("cast_ring", "short"): _digest(build_magic_scene("cast_ring", **short)),
    }
    assert actual == expected


@pytest.mark.parametrize("preset", NEW_PRESETS)
def test_new_presets_pass_the_godot_scene_validator(preset):
    scene = build_magic_scene(preset, fps=24, frame_count=48, particle_count=12)
    assert scene["transparent"] is True and scene["glow"] == 0
    assert validate_godot_scene(scene) == scene
    assert math.ceil(scene["duration"] * scene["fps"]) == 48
    assert scene["materials"]["gold"]["emission_energy"] == 2.0


@pytest.mark.parametrize("preset", PARTICLE_PRESETS)
@pytest.mark.parametrize("particle_count", PARTICLE_COUNTS)
def test_particle_counts_sum_exactly(preset, particle_count):
    scene = build_magic_scene(
        preset, fps=24, frame_count=48, particle_count=particle_count
    )
    assert sum(item["count"] for item in scene["particles"]) == particle_count
    assert all(item["count"] >= 1 for item in scene["particles"])


@pytest.mark.parametrize("preset", NEW_PRESETS)
def test_new_presets_are_deterministic_for_the_same_request(preset):
    first = build_magic_scene(preset, fps=24, frame_count=30, seed=11)
    assert first == build_magic_scene(preset, fps=24, frame_count=30, seed=11)


@pytest.mark.parametrize("preset", PARTICLE_PRESETS)
def test_particle_presets_change_with_seed(preset):
    a = build_magic_scene(preset, fps=24, frame_count=30, seed=11, particle_count=9)
    b = build_magic_scene(preset, fps=24, frame_count=30, seed=12, particle_count=9)
    assert a != b
    assert [item["seed"] for item in a["particles"]] != [
        item["seed"] for item in b["particles"]
    ]


@pytest.mark.parametrize("preset", OBJECT_PRESETS)
def test_object_presets_have_no_random_element(preset):
    # Tori and the orb are analytic keyframes, so the seed has no effect by design.
    a = build_magic_scene(preset, fps=24, frame_count=30, seed=11)
    b = build_magic_scene(preset, fps=24, frame_count=30, seed=12)
    assert a == b


@pytest.mark.parametrize("preset", NEW_PRESETS)
@pytest.mark.parametrize("fps,frame_count", [(24, 3), (24, 48), (120, 300)])
def test_new_presets_keep_all_timing_inside_duration(preset, fps, frame_count):
    scene = build_magic_scene(preset, fps=fps, frame_count=frame_count)
    duration = scene["duration"]
    assert 0 < duration <= frame_count / fps
    for item in scene.get("particles", []):
        assert 0 <= item["start"] < duration
        assert item["lifetime"] > 0
    for item in scene.get("objects", []):
        times = [key["time"] for key in item.get("animation", [])]
        assert all(0 <= time <= duration for time in times)
        assert times == sorted(times) and len(set(times)) == len(times)


def test_spark_burst_is_one_radial_burst_at_frame_zero():
    scene = build_magic_scene("spark_burst", particle_count=20)
    (group,) = scene["particles"]
    assert group["start"] == 0
    assert group["spread"] == 180
    assert group["gravity"][1] < 0  # pulls the sparks back down
    assert group["lifetime"] < scene["duration"]


def test_rising_embers_spawn_in_several_waves_from_the_bottom_band():
    scene = build_magic_scene("rising_embers", frame_count=48, particle_count=30)
    starts = sorted({item["start"] for item in scene["particles"]})
    xs = {item["location"][0] for item in scene["particles"]}
    assert len(starts) == 3
    assert len(xs) > 1
    assert all(item["location"][1] < 0 for item in scene["particles"])
    assert all(item["direction"][1] > 0 for item in scene["particles"])


def test_orbit_rings_use_three_distinct_tilts_with_staggered_spin():
    scene = build_magic_scene("orbit_rings", frame_count=48)
    assert len(scene["objects"]) == 3
    assert all(item["type"] == "torus" for item in scene["objects"])
    assert len({tuple(item["rotation"]) for item in scene["objects"]}) == 3
    starts = [item["animation"][0]["time"] for item in scene["objects"]]
    assert starts == sorted(starts) and len(set(starts)) == 3


def test_pulse_orb_is_an_emissive_sphere_with_in_out_scale():
    scene = build_magic_scene("pulse_orb", frame_count=48, radius=1.0)
    (orb,) = scene["objects"]
    assert orb["type"] == "sphere" and orb["material"] == "gold"
    scales = [key["scale"][0] for key in orb["animation"]]
    assert min(scales) > 0
    assert scales[0] == pytest.approx(0.55)
    assert max(scales) > scales[0] + 0.3
    # Starts and ends at the resting size; pulses rise and fall between them.
    assert scales[-1] == pytest.approx(0.55)
    assert scales.index(max(scales)) not in (0, len(scales) - 1)


@pytest.mark.parametrize("preset", PARTICLE_PRESETS)
def test_particle_presets_forward_color_size_and_emission(preset):
    scene = build_magic_scene(
        preset,
        color=(0.2, 0.7, 1.0),
        emission=3.5,
        particle_size=0.05,
        particle_count=6,
    )
    for item in scene["particles"]:
        assert item["color"] == [0.2, 0.7, 1.0]
        assert item["emission_energy"] == 3.5
        assert item["size"] == 0.05


@pytest.mark.parametrize("preset", NEW_PRESETS)
def test_invalid_preset_name_still_raises(preset):
    with pytest.raises(ValueError, match="preset must be one of"):
        build_magic_scene(preset + "_typo")
