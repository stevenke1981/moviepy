"""Smoke-test the ``examples/ae_*.py`` scenes so the examples keep working."""

import importlib.util
from pathlib import Path

import numpy as np


EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_scene_renders_expected_features():
    demo = load_example("ae_demo")
    comp = demo.build_scene()
    assert comp.size == (demo.WIDTH, demo.HEIGHT)
    names = [layer.name for layer in comp.layers]
    assert {"badge", "rig", "window", "stripes", "grain"} <= set(names)
    assert comp.layer("window").enabled is False
    assert comp.layer("badge").parent is comp.layer("rig")
    badge = comp.layer("badge")
    assert [e.name for e in badge.effects] == ["Gaussian Blur", "Echo"]
    vignette = comp.layer("vignette")
    assert vignette.is_adjustment and len(vignette.masks) == 1
    first, later = comp.get_frame(0.0), comp.get_frame(3.0)
    assert first.shape == (demo.HEIGHT, demo.WIDTH, 3) and first.dtype == np.uint8
    assert not np.array_equal(first, later)
    # The vignette darkens the corners; the feathered center barely changes.
    vignette.enabled = False
    plain = comp.get_frame(3.0)
    assert int(later[2, 2].sum()) < int(plain[2, 2].sum()) - 60
    center = later[180, 320].astype(int) - plain[180, 320].astype(int)
    assert np.abs(center).max() <= 1


def test_lower_third_reveals_bar_and_title():
    example = load_example("ae_lower_third")
    comp = example.build_scene()
    assert comp.layer("band focus").is_adjustment
    assert comp.layer("text window").enabled is False
    start, middle = comp.get_frame(0.0), comp.get_frame(1.6)
    y = example.BAR_TOP + example.BAR_HEIGHT // 2
    x = example.BAR_LEFT + example.BAR_WIDTH - 20
    # The bar is wiped in: dark at 1.6 s, absent at 0 s.
    assert middle[y, x].sum() < start[y, x].sum() - 100
    # White title pixels appear inside the bar once it has slid in.
    band = middle[example.BAR_TOP + 7 : example.BAR_TOP + 35, 60:300]
    assert band.min(axis=2).max() > 200


def test_logo_reveal_effects_and_sweep():
    example = load_example("ae_logo_reveal")
    comp = example.build_scene()
    assert comp.layer("sweep").track_matte.layer is comp.layer("sweep matte")
    assert comp.layer("sweep matte").enabled is False
    assert comp.layer("logo glow").blend_mode == "screen"
    early, settled = comp.get_frame(0.3), comp.get_frame(3.6)
    cx, cy = int(example.CENTER[0]), int(example.CENTER[1])
    # The star is blurred and faint at first, crisp orange later.
    assert settled[cy - 70, cx].tolist() != early[cy - 70, cx].tolist()
    assert settled[cy - 70, cx, 0] > 200 and settled[cy - 70, cx, 2] < 160
