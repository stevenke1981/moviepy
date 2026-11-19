"""Exercise framing coverage and the Godot-to-AE spell bridge without fake alpha."""

import math
import os
from pathlib import Path

import numpy as np

import pytest

from moviepy import ImageClip
from moviepy.ae import (
    CameraFraming,
    CompLayer,
    Composition,
    Expression,
    Mask,
    Property,
    RenderContext,
    pan_zoom,
)
from moviepy.ae.layers.av import AVLayer
from moviepy.ae.motion.magic import MagicRender, build_magic_scene, render_magic
from moviepy.ae.three_d.godot import GodotRender


@pytest.mark.parametrize(
    "direction,delta",
    [
        ("left", (-12, 0)),
        ("right", (12, 0)),
        ("up", (0, -12)),
        ("down", (0, 12)),
        ("none", (0, 0)),
    ],
)
def test_camera_directions_easing_and_endpoint_holds(direction, delta):
    rig = pan_zoom(
        (200, 140), (100, 70), duration=2, direction=direction, distance=12, zoom=(2, 2)
    )
    start, end = rig.sample(0), rig.sample(2)
    assert tuple(b - a for a, b in zip(start.center, end.center)) == pytest.approx(
        delta
    )
    assert rig.sample(-4) == start and rig.sample(8) == end
    progress = np.array(rig.pan.value_at(0.25))
    if direction != "none":
        assert np.linalg.norm(progress) < np.linalg.norm(delta) * 0.125


@pytest.mark.parametrize("size", [(80, 48), (43, 91), (120, 80)])
@pytest.mark.parametrize("overscan", [1, 1.04, 1.5])
def test_camera_keeps_all_output_pixels_opaque_under_large_pan(size, overscan):
    source = np.full((63, 97, 3), (70, 110, 150), np.uint8)
    rig = CameraFraming(
        (97, 63),
        size,
        pan=Property([(0, (-1e4, 1e4)), (1, (1e4, -1e4))], value_type="vec2"),
        zoom=Property([(0, 0.3), (1, 1.5)]),
        overscan=overscan,
    )
    comp = Composition(size=size, duration=2, fps=24, transparent=True)
    comp.add_layer(
        AVLayer(ImageClip(source).with_duration(2), transform=rig.to_transform())
    )
    for t in (1, 0, 0.17, 0.5, 1.25, 0):
        pixels = comp.render_buffer(t).rgba
        np.testing.assert_allclose(pixels[..., 3], 1, atol=1e-6)
        np.testing.assert_allclose(
            pixels[..., :3],
            np.broadcast_to(np.array((70, 110, 150)) / 255, pixels[..., :3].shape),
            atol=1e-6,
        )
        crop = rig.sample(t).crop
        assert crop[0] >= -1e-9 and crop[1] >= -1e-9
        assert crop[2] <= 96 + 1e-9 and crop[3] <= 62 + 1e-9


def test_camera_subject_box_stays_inside_margin_and_caps_zoom():
    rig = CameraFraming(
        (200, 140), (100, 70), subject_bounds=(65, 42, 134, 97), zoom=8, pan=(500, -500)
    )
    sample = rig.sample(0)
    assert sample.clamped
    matrix = rig.to_transform().matrix_at(0)
    mapped = matrix @ np.array([[65, 134], [42, 97], [1, 1]])
    assert np.all(mapped[:2, 0] >= np.array((99, 69)) * 0.08 - 1e-9)
    assert np.all(mapped[:2, 1] <= np.array((99, 69)) * 0.92 + 1e-9)


def test_camera_source_safe_bounds_and_live_properties_with_context():
    rig = CameraFraming(
        (200, 200),
        (80, 80),
        focus=Expression("[100 + index, 100]"),
        zoom=2,
        safe_bounds=(10, 20, 189, 179),
    )
    matrix = rig.to_transform().matrix_at(0, index=7)
    np.testing.assert_allclose((matrix @ [107, 100, 1])[:2], (39.5, 39.5))
    transform = rig.to_transform()
    rig.pan = [(0, (0, 0)), (1, (15, 0))]
    assert transform.matrix_at(1)[0, 2] < transform.matrix_at(0)[0, 2]
    with pytest.raises(TypeError, match="callable"):
        transform.to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"overscan": 0.99},
        {"safe_margin": 0.5},
        {"safe_bounds": (0, 0, 200, 140)},
        {"subject_bounds": (4, 6, 3, 7)},
        {"focus": (math.nan, 0)},
        {"zoom": True},
    ],
)
def test_camera_rejects_invalid_controls(kwargs):
    with pytest.raises((TypeError, ValueError)):
        CameraFraming((200, 140), (100, 70), **kwargs)


def test_camera_reports_impossible_subject_and_invalid_animated_zoom():
    rig = CameraFraming((200, 140), (100, 70), subject_bounds=(0, 0, 199, 139))
    with pytest.raises(ValueError, match="subject cannot fit"):
        rig.sample(0)
    rig = CameraFraming((200, 140), (100, 70), zoom=Property([(0, 1), (1, -1)]))
    with pytest.raises(ValueError, match="positive"):
        rig.sample(1)


@pytest.mark.parametrize("preset", ["warm_aura", "flow_particles", "cast_ring"])
def test_magic_contract_has_real_transparency_and_exact_timing(preset):
    scene = build_magic_scene(preset, fps=24, frame_count=7, seed=84, particle_count=7)
    assert scene["transparent"] is True and scene["glow"] == 0
    assert math.ceil(scene["duration"] * scene["fps"]) == 7
    assert scene == build_magic_scene(
        preset, fps=24, frame_count=7, seed=84, particle_count=7
    )
    if preset == "flow_particles":
        assert sum(item["count"] for item in scene["particles"]) == 7
        assert len({item["seed"] for item in scene["particles"]}) == len(
            scene["particles"]
        )
        assert scene != build_magic_scene(
            preset, fps=24, frame_count=7, seed=85, particle_count=7
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"preset": "fake"},
        {"fps": 7},
        {"frame_count": 1},
        {"seed": -1},
        {"emission": math.inf},
        {"color": (1, 0, -1)},
        {"particle_count": True},
        {"radius": 4},
    ],
)
def test_magic_rejects_ambiguous_or_unbounded_requests(parameters):
    with pytest.raises((ValueError, TypeError)):
        build_magic_scene(**{"preset": "warm_aura", **parameters})


def linear_fixture(tmp_path):
    frames = []
    for index in range(4):
        pixels = np.zeros((16, 16, 4), np.float32)
        pixels[5:10, 5:10] = (4, 1.5, 0.3, 1)
        path = tmp_path / f"{index}.npz"
        np.savez_compressed(path, rgba=pixels)
        frames.append(str(path))
    render = GodotRender(
        tuple("unused.png" for _ in frames),
        4,
        1,
        Path("unused.wav"),
        Path("unused.json"),
        Path("unused.log"),
        {},
        tuple(frames),
    )
    return MagicRender(render, "warm_aura", (1, 0.58, 0.16))


@pytest.mark.parametrize("mode", ["normal", "add"])
def test_magic_layer_retains_hdr_glow_alpha_and_random_access(tmp_path, mode):
    result = linear_fixture(tmp_path)
    layer = result.to_layer(opacity=100, glow_radius=2, blend_mode=mode)
    comp = Composition(
        size=(16, 16),
        fps=4,
        duration=1,
        transparent=True,
        context=RenderContext(working_space="linear"),
    )
    comp.add_layer(layer)
    first = comp.render_buffer(0.5).rgba.copy()
    for t in (0.75, 0, 0.5):
        np.testing.assert_array_equal(comp.render_buffer(t).rgba, first)
    assert first[..., :3].max() >= 4
    assert np.all(first[0, 0] == 0)
    assert np.any((first[..., 3] > 0) & (first[..., 3] < 1))
    assert layer.effects[0].radius.value_type == "float"


def test_magic_default_envelope_has_transparent_endpoints(tmp_path):
    layer = linear_fixture(tmp_path).to_layer()
    assert layer.transform.opacity_at(0) == 0
    assert layer.transform.opacity_at(0.75) == 0
    assert layer.transform.opacity_at(0.3) > 0
    with pytest.raises(ValueError, match="Normal, Add or Screen"):
        linear_fixture(tmp_path).to_layer(blend_mode="multiply")


def test_magic_protection_masks_the_complete_glow_after_precomposition(tmp_path):
    child = Composition(
        size=(16, 16),
        fps=4,
        duration=1,
        transparent=True,
        context=RenderContext(working_space="linear"),
    )
    child.add_layer(linear_fixture(tmp_path).to_layer(opacity=100, glow_radius=4))
    parent = Composition(
        size=(16, 16),
        fps=4,
        duration=1,
        transparent=True,
        context=RenderContext(working_space="linear"),
    )
    parent.add_layer(
        CompLayer(child, masks=[Mask.rect((7, 7), (7, 7), mode="subtract")])
    )
    pixels = parent.render_buffer(0.5).rgba
    assert not np.any(pixels[5:10, 5:10])
    assert np.any(pixels[..., 3])


def test_magic_portrait_scene_keeps_geometry_inside_the_shorter_axis():
    square = build_magic_scene("cast_ring", size=(128, 128))
    portrait = build_magic_scene("cast_ring", size=(64, 192))
    assert portrait["camera"]["location"][2] == square["camera"]["location"][2] * 3
    with pytest.raises(ValueError, match="frame_count"):
        build_magic_scene("cast_ring", frame_count=2)


@pytest.mark.skipif(
    not os.environ.get("MOVIEPY_AE_TEST_GODOT"), reason="real Godot render is opt-in"
)
def test_real_magic_rgba_repeatability_and_layer_endpoints(tmp_path):
    executable = os.environ["MOVIEPY_AE_TEST_GODOT"]
    results = {}
    for preset in ("warm_aura", "flow_particles", "cast_ring"):
        result = render_magic(
            preset,
            tmp_path / preset,
            executable=executable,
            size=(64, 64),
            fps=12,
            frame_count=12,
            emission=4,
        )
        results[preset] = result
        assert len(result.render.frames) == len(result.render.linear_frames) == 12
        assert result.render.metadata["linear"]["peak_rgb"] > 1
        partial = 0
        for filename in result.render.linear_frames:
            with np.load(filename) as source:
                rgba = source["rgba"]
                assert not np.any(rgba[rgba[..., 3] == 0, :3])
                partial += np.count_nonzero((rgba[..., 3] > 0) & (rgba[..., 3] < 1))
        assert partial > 0
        layer = result.to_layer()
        comp = Composition(
            size=(64, 64),
            fps=12,
            duration=1,
            transparent=True,
            context=RenderContext(working_space="linear"),
        )
        comp.add_layer(layer)
        middle = comp.render_buffer(0.5).rgba.copy()
        for t in (11 / 12, 0):
            assert not np.any(comp.render_buffer(t).rgba)
        np.testing.assert_array_equal(comp.render_buffer(0.5).rgba, middle)
        assert middle[..., 3].max() > 0
        assert not np.any(middle[0, 0])
    repeat = render_magic(
        "flow_particles",
        tmp_path / "repeat",
        executable=executable,
        size=(64, 64),
        fps=12,
        frame_count=12,
        emission=4,
    )
    for a, b in zip(
        results["flow_particles"].render.linear_frames, repeat.render.linear_frames
    ):
        with np.load(a) as first, np.load(b) as second:
            np.testing.assert_array_equal(first["rgba"], second["rgba"])
