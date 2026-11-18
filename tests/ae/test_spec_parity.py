"""Spec parity: the AE_PARITY_SPEC API-style example runs as written.

Covers ``ae.wiggle`` as a property value, ``ae.masks.MaskPath`` and lazy exports.
"""

import json
import subprocess
import sys

import numpy as np

import pytest

import moviepy.ae as ae
from moviepy.ae.properties.expression import Expression
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.values import PathValue
from moviepy.video.VideoClip import ColorClip


def _spec_comp(layer_id=None):
    """Build the spec example at 1/30 scale with a solid clip as footage."""
    comp = ae.Composition(size=(64, 36), fps=30, duration=2)
    comp.add_solid("bg", color=(20, 20, 30))
    footage = ColorClip((64, 36), color=(200, 120, 40)).with_duration(2)
    vid = comp.add_clip(footage, name="footage", id=layer_id)
    # Spec gap: position defaults to None (anchor-relative), so set it first.
    vid.transform.position = (32, 18)
    vid.transform.position.set_keyframes(
        [(0, (32, 18)), (2, (13, 10), ae.Ease.bezier(0.2, 0, 0.2, 1))]
    )
    vid.transform.scale = (100, 100)
    vid.blend_mode = ae.BlendMode.SCREEN
    vid.effects.add(ae.fx.GaussianBlur(blurriness=ae.wiggle(freq=2, amp=8)))
    vid.masks.add(
        ae.Mask.ellipse(center=(32, 18), size=(20, 20), feather=2, mode="add")
    )
    return comp, vid


def test_mask_path_is_path_value():
    """``ae.masks.MaskPath`` is the property path value type itself."""
    assert ae.masks.MaskPath is PathValue
    assert ae.MaskPath is PathValue
    assert "MaskPath" in dir(ae.masks)
    path = ae.masks.MaskPath([(0, 0), (4, 0), (4, 4)])
    assert path.closed and len(path.in_tangents) == 3


def test_wiggle_is_exported_and_returns_expression():
    """``ae.wiggle`` is listed in ``dir(ae)`` and builds the helper source."""
    assert "wiggle" in dir(ae) and "MaskPath" in dir(ae)
    expression = ae.wiggle(freq=2, amp=8)
    assert isinstance(expression, Expression)
    assert expression.source == "wiggle(2.0, 8.0, 1, 0.5)"
    vector = ae.wiggle(1, (3, 4), octaves=2, amp_mult=0.25)
    assert vector.source == "wiggle(1.0, [3.0, 4.0], 2, 0.25)"


def test_wiggle_matches_the_expression_helper():
    """The factory gives the same numbers as writing the expression by hand."""
    context = ae.RenderContext(rng_seed=5)
    manual = Property(10.0, expression=Expression("wiggle(2, 8)"))
    built = ae.wiggle(freq=2, amp=8, value=10.0)
    assert isinstance(built, Property) and built.value_type == "float"
    for t in (0.0, 0.25, 0.9, 1.7):
        expected = manual.value_at(t, context=context, layer_id="a")
        assert built.value_at(t, context=context, layer_id="a") == expected
        assert 2.0 <= expected <= 18.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"freq": -1, "amp": 1},
        {"freq": float("nan"), "amp": 1},
        {"freq": 1, "amp": 1, "octaves": 0},
        {"freq": 1, "amp": 1, "octaves": 1.5},
        {"freq": 1, "amp": 1, "amp_mult": 2},
        {"freq": 1, "amp": (1, 2, 3, 4)},
        {"freq": True, "amp": 1},
    ],
)
def test_wiggle_rejects_invalid_arguments_eagerly(kwargs):
    """Bad arguments fail when the factory is called, not at render time."""
    with pytest.raises(ValueError):
        ae.wiggle(**kwargs)


def test_effect_param_accepts_wiggle_and_clamps():
    """The blur Param takes the wiggle directly; negative blur clamps to zero."""
    blur = ae.fx.GaussianBlur(blurriness=ae.wiggle(freq=2, amp=8))
    context = ae.RenderContext(rng_seed=3)
    values = [
        blur.values_at(t / 30, context, layer_id="x")["blurriness"] for t in range(60)
    ]
    assert min(values) >= 0.0 and max(values) <= 8.0
    assert len(set(values)) > 10


def test_spec_example_renders_deterministically():
    """Same seed and time give identical frames; the wiggle changes over time."""
    layer_id = "0" * 31 + "1"
    first, vid = _spec_comp(layer_id)
    second, _ = _spec_comp(layer_id)
    context = ae.RenderContext(rng_seed=11)
    blur = vid.effects[0]
    blurs = {
        blur.values_at(t, context, layer_id=vid.id)["blurriness"]
        for t in (0.1, 0.4, 1.3)
    }
    assert len(blurs) > 1  # a negative wiggle clamps to 0, so two may tie
    for t in (0.0, 0.4, 1.3):
        a = first.render_buffer(t, context).rgba
        b = second.render_buffer(t, context).rgba
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first.get_frame(0.4), second.get_frame(0.4))
    assert first.get_frame(0.4).shape == (36, 64, 3)


def test_same_name_layers_wiggle_independently():
    """Two layers named alike still draw independent wiggle values."""
    comp = ae.Composition(size=(8, 8), fps=30, duration=2)
    blurs = []
    for _ in range(2):
        layer = comp.add_solid("Solid", color=(255, 255, 255))
        layer.effects.add(ae.fx.GaussianBlur(blurriness=ae.wiggle(2, 8, value=10)))
        blurs.append((layer, layer.effects[0]))
    context = ae.RenderContext(rng_seed=4)
    times = [t / 10 for t in range(1, 15)]
    series = [
        [effect.values_at(t, context, layer_id=layer.id)["blurriness"] for t in times]
        for layer, effect in blurs
    ]
    assert blurs[0][0].name == blurs[1][0].name
    assert series[0] != series[1]


def test_wiggle_property_roundtrips_through_json():
    """The wiggle is stored as expression source, so it survives JSON."""
    prop = ae.wiggle(freq=3, amp=(2, 5), value=(10, 20))
    restored = Property.from_dict(json.loads(json.dumps(prop.to_dict())))
    context = ae.RenderContext(rng_seed=2)
    assert restored.value_at(0.6, context=context, layer_id=7) == prop.value_at(
        0.6, context=context, layer_id=7
    )


def test_import_stays_lazy():
    """Importing ``moviepy.ae`` and listing it does not load the new exports."""
    code = (
        "import sys; import moviepy.ae as ae; names = dir(ae); "
        "assert 'wiggle' in names and 'MaskPath' in names; "
        "assert 'moviepy.ae.properties.expression' not in sys.modules; "
        "assert 'moviepy.ae.masks' not in sys.modules; "
        "assert 'moviepy.ae.properties.values' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
