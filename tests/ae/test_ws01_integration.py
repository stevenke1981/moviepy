"""Exercise the public WS-01 API across independently implemented modules."""

import json
from enum import Enum

import numpy as np

import pytest

from moviepy.ae import Buffer, RenderContext
from moviepy.ae.properties import (
    Ease,
    Expression,
    ExpressionError,
    Keyframe,
    PathValue,
    Property,
)


def test_property_drives_actual_float_buffer_pixels():
    """A typed color animation reaches the existing rendering core unchanged."""
    color = Property(
        (1, 0, 0),
        value_type="color",
        keyframes=[Keyframe(0, (1, 0, 0)), Keyframe(2, (0, 0, 1))],
    )
    frame = np.array([[(*color.value_at(1), 1)]], dtype=np.float32)
    assert Buffer(frame).to_uint8_rgb().tolist() == [[[128, 0, 128]]]


def test_expression_overlays_keyframes_and_json_roundtrips():
    """Expression sampling uses the animated base and survives safe JSON."""
    prop = Property(0, keyframes=[(0, 0), (2, 100)])
    prop = prop.with_expression(Expression("value + valueAtTime(1) + time"))
    assert prop.value_at(0.5) == pytest.approx(75.5)
    restored = Property.from_json(prop.to_json())
    assert restored.value_at(0.5) == pytest.approx(75.5)


def test_setter_mutates_and_with_keyframes_preserves_original():
    """Setter and MoviePy-style copy methods have distinct documented behavior."""
    prop = Property(0)
    assert prop.set_keyframes([(0, 0), (1, 10)]) is prop
    changed = prop.with_keyframes([(0, 0), (1, 20)])
    assert prop.value_at(0.5) == 5
    assert changed.value_at(0.5) == 10


def test_expression_context_wiggle_is_order_independent():
    """Caller context seed and layer identity reproduce subframe noise."""
    prop = Property(Expression("wiggle(2, 3)"))
    context = RenderContext(rng_seed=19, fps=60)
    times = [0.13, 1.12, 0.51, -0.25]
    first = {t: prop.value_at(t, context=context, layer_id="x") for t in times}
    for t in reversed(times):
        assert prop.value_at(t, context=context, layer_id="x") == first[t]


def test_separate_dimensions_have_independent_keyframe_times():
    """Independent scalar timelines recombine into a vector property."""
    x = Property(0, keyframes=[(0, 0), (1, 10)])
    y = Property(0, keyframes=[(0, 0), (2, 20)])
    combined = Property.from_dimensions((x, y))
    assert combined.value_at(1.5) == (10, 15)
    restored = Property.from_json(combined.to_json())
    assert restored.value_at(1.5) == (10, 15)


def test_enum_roundtrip_uses_tokens_without_class_imports():
    """Enum values keep explicit primitive-token semantics in JSON."""

    class Mode(Enum):
        NORMAL = "normal"

    prop = Property(Mode.NORMAL)
    assert prop.value_at(0) == "normal"
    assert Property.from_json(prop.to_json()).value_at(0) == "normal"


def test_path_static_roundtrip_and_color_easing_metadata():
    """Path geometry and numeric easing metadata have complete schemas."""
    path = PathValue(vertices=((0, 0), (2, 0)), closed=False)
    assert Property.from_json(Property(path).to_json()).value_at(0) == path
    prop = Property(
        0,
        keyframes=[Keyframe(0, 0, out_ease=Ease.ease_in()), Keyframe(1, 1)],
    )
    restored = Property.from_json(prop.to_json())
    assert restored.value_at(0.3) == pytest.approx(prop.value_at(0.3))


@pytest.mark.parametrize("source", ["__import__('os')", "open('x')", "().__class__"])
def test_public_expression_rejects_required_escape_cases(source):
    """The public entry point preserves the required sandbox boundary."""
    with pytest.raises(ExpressionError):
        Expression(source).evaluate(0)


def test_schema_rejects_callable_reconstruction():
    """Deserialization cannot turn an untrusted string into executable Python."""
    payload = {
        "schema": 1,
        "value_type": "float",
        "source": {"kind": "callable", "value": "lambda t: t"},
        "expression": None,
    }
    with pytest.raises((TypeError, ValueError)):
        Property.from_json(json.dumps(payload))


def test_python_callable_runs_but_is_not_silently_serialized():
    """Trusted user callables retain MoviePy use while JSON stays inert."""
    prop = Property(lambda t: 2 * t, value_type="float")
    assert prop.value_at(0.25) == 0.5
    with pytest.raises((TypeError, ValueError)):
        prop.to_json()
