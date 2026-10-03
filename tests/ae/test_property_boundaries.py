"""Expression interactions, geometry metadata and finite resource boundaries."""

import copy

import numpy as np
import pytest

from moviepy.ae import RenderContext
from moviepy.ae.properties.expression import Expression
from moviepy.ae.properties.easing import Ease
from moviepy.ae.properties.keyframe import Keyframe
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.serialization import dumps, loads, validate_payload
from moviepy.ae.properties.values import PathValue, decode_value, normalize_value


def test_expression_overlay_roundtrip_and_base_velocity():
    prop = Property(keyframes=[Keyframe(0, 0), Keyframe(1, 2)])
    animated = prop.with_expression(Expression("value + valueAtTime(time)"))
    assert animated.value_at(0.5) == 2
    assert animated.base_value_at(0.5) == 1
    assert animated.base_velocity_at(0) == pytest.approx(2)
    assert animated.base_velocity_at(1) == pytest.approx(2)
    assert animated.velocity_at(0.5) == pytest.approx(4)
    assert prop.value_at(0.5) == 1
    assert Property.from_json(animated.to_json()).value_at(0.5) == 2
    assert animated.with_expression(None).value_at(0.5) == 1
    direct = Property(Expression("time * 2"))
    assert direct.value_at(3) == 6
    context = RenderContext(rng_seed=4)
    wiggle = prop.with_expression(Expression("wiggle(2, 1)"))
    assert wiggle.value_at(0.4, context=context) == wiggle.value_at(
        0.4, context=context
    )


def test_expression_vectors_project_and_do_not_recurse():
    vector = Property((1, 2)).with_expression(Expression("value * 2"))
    x, y = vector.separate_dimensions()
    assert (x.value_at(0.5), y.value_at(0.5)) == (2, 4)
    with pytest.raises(TypeError, match="callable"):
        x.to_dict()
    scalar = Property(3).with_expression(
        Expression("valueAtTime(0) + velocityAtTime(0)")
    )
    assert scalar.value_at(0) == 3


def test_constructor_keys_and_discrete_integer_tokens():
    assert Property([(0, 0), (1, 2)]).value_at(0.5) == 1
    enum = Property(1, value_type="enum").with_keyframes(
        [Keyframe(0, 1, interp="hold"), Keyframe(1, 2, interp="hold")]
    )
    assert Property.from_json(enum.to_json()).value_at(0.5) == 1
    assert enum.value_at(1) == 2
    assert Property((1, 2, 3)).separate_dimensions()[2].value_at(0) == 3
    combined = Property.from_dimensions((Property(1), Property(2)))
    assert len(combined.separate_dimensions()) == 2
    assert combined.with_value((3, 4)).value_at(0) == (3, 4)


def test_spatial_roving_and_separated_speed_metadata():
    prop = Property((0, 0), spatial=True).with_keyframes(
        [Keyframe(0, (0, 0)), Keyframe(0.8, (1, 0), roving=True), Keyframe(1, (4, 0))]
    )
    assert prop.keyframes[1].time == pytest.approx(0.25)
    assert Property.from_json(prop.to_json()).value_at(0.5) == pytest.approx((2, 0))
    keyed = Property((0, 0)).with_keyframes(
        [Keyframe(0, (0, 0), out_speed=(1, 2)), Keyframe(1, (1, 2), in_speed=(1, 2))]
    )
    x, y = keyed.separate_dimensions()
    assert (x.keyframes[0].out_speed, y.keyframes[1].in_speed) == (1, 2)


def test_empty_path_and_tangents_detach():
    array = np.array([[0.0, 0.0], [1.0, 1.0]])
    tangent = [[0.2, 0.3], [0.4, 0.5]]
    path = PathValue(array, in_tangents=tangent)
    array[0] = 99
    tangent[0][0] = 99
    assert path.vertices[0] == (0, 0)
    assert path.in_tangents[0] == (0.2, 0.3)
    empty = PathValue(())
    assert (
        Property(empty)
        .with_keyframes([Keyframe(0, empty), Keyframe(1, empty)])
        .value_at(0.5)
        == empty
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Property(0, spatial=1),
        lambda: Property(0, spatial=True),
        lambda: Property(0, expression="time"),
        lambda: Property(Expression("time"), expression=Expression("value")),
        lambda: Property(0, value_type=[]),
        lambda: Property(keyframes=[]),
        lambda: Property([[1], [2]]),
        lambda: Property(0).set_keyframes(iter([])),
        lambda: Property(0).set_keyframes([Keyframe(0, 0)] * 4097),
        lambda: Property(0).with_keyframes(
            [Keyframe(0, 0), Keyframe(0.5, 1, roving=True), Keyframe(1, 2)]
        ),
        lambda: Property(0).separate_dimensions(),
        lambda: Property.from_dimensions((Property(0),)),
        lambda: Property.from_dimensions((Property(True), Property(0))),
        lambda: Property.from_audio_keyframes([], []),
        lambda: Property.from_audio_keyframes([0], [0, 1]),
        lambda: Property.from_audio_keyframes(range(4097), range(4097)),
        lambda: Property(0).base_velocity_at(0, delta=0),
        lambda: Property(True).velocity_at(0),
        lambda: Property(1).velocity_at(1e100, delta=1e-4),
    ],
)
def test_invalid_property_boundaries(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "value,kind",
    [
        (np.array([object()], dtype=object), "vec2"),
        (np.zeros((2, 2)), "vec2"),
        (np.zeros(24577), "vec2"),
        (np.zeros(24577), None),
        (np.zeros((4097, 2)), "path"),
        ("", "enum"),
        ("x" * 257, "enum"),
        (2**54, "enum"),
        (1.2, "enum"),
        (1, "bool"),
        (None, "path"),
        (1, "unknown"),
        ({}, "vec2"),
    ],
)
def test_value_codec_boundaries(value, kind):
    with pytest.raises((TypeError, ValueError)):
        normalize_value(value, kind)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PathValue(np.zeros((2, 3))),
        lambda: PathValue(np.zeros(4)),
        lambda: PathValue(None),
        lambda: PathValue([(0, 0)] * 4097),
        lambda: PathValue(((0, 0),), closed=1),
        lambda: PathValue(((0, 0),), in_tangents=((0, 0), (0, 0))),
        lambda: decode_value({"vertices": []}, "path"),
        lambda: normalize_value(10**1000),
    ],
)
def test_path_and_numeric_codec_boundaries(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "payload",
    [
        {1: 2},
        {"object": object()},
        {"text": "x" * 8193},
        {"huge": [None] * 100000},
        {"huge": {str(i): None for i in range(100000)}},
    ],
)
def test_payload_resource_boundaries(payload):
    with pytest.raises((TypeError, ValueError)):
        validate_payload(payload)


def test_payload_depth_and_json_input_boundaries():
    nested = None
    for _ in range(30):
        nested = [nested]
    with pytest.raises(ValueError, match="depth"):
        validate_payload(nested)
    with pytest.raises(TypeError):
        loads(b"{}")
    with pytest.raises(ValueError):
        loads("[" * 10000 + "]" * 10000)
    with pytest.raises(ValueError):
        dumps(["x" * 8192] * 123)


def test_malformed_dimension_and_keyframe_sources():
    base = Property.from_dimensions((Property(1), Property(2))).to_dict()
    for mutate in (
        lambda p: p["source"].update(properties=[]),
        lambda p: p.update(value_type="vec3"),
        lambda p: p.update(spatial=True),
        lambda p: p["source"].update(extra=0),
        lambda p: p.update(source=[]),
    ):
        payload = copy.deepcopy(base)
        mutate(payload)
        with pytest.raises((TypeError, ValueError)):
            Property.from_dict(payload)
    keyed = Property(keyframes=[Keyframe(0, 0)]).to_dict()
    keyed["source"]["keyframes"] = []
    with pytest.raises(ValueError):
        Property.from_dict(keyed)


def test_combined_dimensions_keep_context_and_snapshot_bindings():
    context = RenderContext(rng_seed=42, fps=24)
    x = Property(0).with_expression(Expression("random() + thisLayer.offset"))
    y = Property(0).with_expression(Expression("timeToFrames(time)"))
    combined = Property.from_dimensions((x, y))
    bindings = {"context": context, "this_layer": {"offset": 3}, "layer_id": "joined"}
    expected = (x.value_at(0.5, **bindings), y.value_at(0.5, **bindings))
    assert combined.value_at(0.5, **bindings) == expected
    overlay = combined.with_expression(Expression("valueAtTime(time) + value"))
    assert overlay.value_at(0.5, **bindings) == pytest.approx(
        tuple(2 * v for v in expected)
    )


def test_callback_subclasses_cannot_execute_conversion_protocols():
    calls = []

    class HostileNumber(np.float64):
        def __float__(self):
            calls.append("float")
            return 1.0

    class HostileList(list):
        def __iter__(self):
            calls.append("iterate")
            return super().__iter__()

    for value, kind in ((HostileNumber(1), "float"), (HostileList([1, 2]), "vec2")):
        with pytest.raises(TypeError):
            Property(lambda t, v=value: v, value_type=kind).value_at(0)
    assert calls == []


def test_hold_jumps_do_not_become_endpoint_velocity_or_continue_motion():
    prop = Property(keyframes=[Keyframe(0, 0, interp="hold"), Keyframe(1, 10)])
    assert prop.base_velocity_at(0) == 0
    assert prop.base_velocity_at(1) == 0
    assert prop.base_velocity_at(0.999999) == 0
    continued = prop.with_expression(Expression("loopOut(type='continue')"))
    assert continued.value_at(2) == 10
    interior = Property(
        keyframes=[Keyframe(0, 0, interp="hold"), Keyframe(1, 10), Keyframe(2, 12)]
    )
    assert interior.base_velocity_at(1) == pytest.approx(2)
    incoming = prop.with_expression(Expression("loopIn(type='continue')"))
    assert incoming.value_at(-1) == 0


def test_projected_expressions_keep_all_evaluation_bindings():
    context = RenderContext(rng_seed=42, fps=24)
    vector = Property((0, 0)).with_expression(
        Expression(
            "value+[thisLayer.offset+thisComp.fps+index, random()+timeToFrames(time)]"
        )
    )
    bindings = {
        "context": context,
        "this_layer": {"offset": 3},
        "this_comp": {"fps": 24},
        "index": 2,
        "layer_id": "projected",
    }
    expected = vector.value_at(0.5, **bindings)
    x, y = vector.separate_dimensions()
    assert (x.value_at(0.5, **bindings), y.value_at(0.5, **bindings)) == expected
    assert (
        x.with_expression(Expression("valueAtTime(time)")).value_at(0.5, **bindings)
        == expected[0]
    )
    assert x.with_value(7).value_at(0.5, **bindings) == 7
    assert y.with_keyframes([Keyframe(0, 4)]).value_at(0.5, **bindings) == 4
    with pytest.raises(TypeError, match="callable"):
        x.to_dict()


@pytest.mark.parametrize(
    "expression,kind,expected",
    [
        ("(1,2)", "vec2", (1.0, 2.0)),
        ("(1,2,3)", "vec3", (1.0, 2.0, 3.0)),
        ("True", "bool", True),
        ("'screen'", "enum", "screen"),
        ("2", "enum", 2),
        ("(.1,.2,.3)", "color", (0.1, 0.2, 0.3)),
        (
            "(((0,0),(1,1)),((0,0),(0,0)),((0,0),(0,0)),True)",
            "path",
            PathValue(((0, 0), (1, 1))),
        ),
    ],
)
def test_direct_expression_has_inert_typed_base(expression, kind, expected):
    prop = Property(Expression(expression), value_type=kind)
    assert prop.value_at(1) == expected
    assert Property.from_json(prop.to_json()).value_at(1) == expected


def test_path_expression_bridge_for_value_and_time_sampling():
    path = PathValue(((0, 0), (1, 1)))
    prop = Property(path).with_expression(Expression("valueAtTime(time)"))
    assert prop.value_at(0.5) == path
    assert prop.with_expression(Expression("value")).value_at(0.5) == path
    assert Property(Expression("value"), value_type="path").value_at(0) == PathValue(())
    with pytest.raises((TypeError, ValueError)):
        Property(path).with_expression(Expression("((),(),(),1)")).value_at(0)


@pytest.mark.parametrize(
    "keys",
    [
        [Keyframe(0, 0, out_ease=Ease.hold()), Keyframe(1, 10)],
        [Keyframe(0, 0), Keyframe(1, 10, in_ease=Ease.hold())],
    ],
)
def test_hold_ease_jumps_do_not_become_velocity(keys):
    prop = Property(keyframes=keys)
    assert prop.base_velocity_at(1) == 0
    assert (
        prop.with_expression(Expression("loopOut(type='continue')")).value_at(2) == 10
    )
