"""Smoke-test the ``examples/ae_demo.py`` scene so the example keeps working."""

import importlib.util
from pathlib import Path

import numpy as np


DEMO = Path(__file__).resolve().parents[2] / "examples" / "ae_demo.py"


def load_demo():
    spec = importlib.util.spec_from_file_location("ae_demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_scene_renders_expected_features():
    demo = load_demo()
    comp = demo.build_scene()
    assert comp.size == (demo.WIDTH, demo.HEIGHT)
    names = [layer.name for layer in comp.layers]
    assert {"badge", "rig", "window", "stripes", "grain"} <= set(names)
    assert comp.layer("window").enabled is False
    assert comp.layer("badge").parent is comp.layer("rig")
    first, later = comp.get_frame(0.0), comp.get_frame(3.0)
    assert first.shape == (demo.HEIGHT, demo.WIDTH, 3) and first.dtype == np.uint8
    assert not np.array_equal(first, later)
    # The dissolve layer is fully opaque white once its opacity reaches 100.
    assert later[70, 110].tolist() == [255, 255, 255]
