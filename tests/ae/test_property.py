"""Behavioral tests for typed property sources and immutable animation copies."""

from enum import Enum

import numpy as np

import pytest

from moviepy.ae.properties.keyframe import Keyframe
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.values import PathValue


@pytest.mark.parametrize(
    "value,kind,expected",
    [
        (2, "float", 2.0),
        ([1, 2], "vec2", (1.0, 2.0)),
        ([1, 2, 3], "vec3", (1.0, 2.0, 3.0)),
        ([2, 0.4, 0.5, 1], "color", (2.0, 0.4, 0.5, 1.0)),
        (True, "bool", True),
        ("screen", "enum", "screen"),
    ],
)
def test_static_typed_sources(value, kind, expected):
    assert Property(value, value_type=kind).value_at(3) == expected


def test_values_detach_and_setter_conventions():
    source = np.array([1.0, 2.0])
    prop = Property(source)
    source[0] = 99
    assert prop.value_at(0) == (1.0, 2.0)
    animated = prop.with_keyframes([Keyframe(0, (0, 0)), Keyframe(2, (2, 4))])
    assert prop.value_at(1) == (1.0, 2.0)
    assert animated.value_at(1) == (1.0, 2.0)
    assert animated.set_keyframes([Keyframe(0, (3, 4))]) is animated
    assert animated.value_at(1) == (3.0, 4.0)


def test_keyframe_interpolation_velocity_and_boundaries():
    prop = Property(keyframes=[Keyframe(0, 0), Keyframe(2, 6)])
    assert [prop.value_at(t) for t in (-1, 0, 1, 2, 3)] == [0, 0, 3, 6, 6]
    assert prop.base_velocity_at(1) == pytest.approx(3)
    assert prop.base_velocity_at(0) == pytest.approx(3)
    assert prop.base_velocity_at(2) == pytest.approx(3)
    assert prop.base_velocity_at(-1) == 0
    assert prop.with_value(7).value_at(1) == 7


def test_callable_type_validation_and_exception_propagation():
    assert Property(lambda t: (t, t + 1), value_type="vec2").value_at(2) == (2, 3)
    with pytest.raises(ValueError):
        Property(lambda t: float("nan")).value_at(0)
    with pytest.raises(ValueError):
        Property(lambda t: (1, 2, 3), value_type="vec2").value_at(0)

    def broken(t):
        raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        Property(broken).value_at(0)


def test_dimensions_are_independently_animated():
    vector = Property((0, 2)).with_keyframes([Keyframe(0, (0, 2)), Keyframe(2, (2, 6))])
    x, y = vector.separate_dimensions()
    y.set_keyframes([Keyframe(0, 10), Keyframe(1, 20)])
    combined = Property.from_dimensions((x, y))
    assert combined.value_at(1) == (1, 20)
    assert vector.value_at(1) == (1, 4)
    assert combined.to_dict()["source"]["kind"] == "dimensions"


def test_audio_hook_and_discrete_hold():
    prop = Property.from_audio_keyframes([0, 1, 2], [0, 4, 2])
    assert prop.value_at(0.5) == 2
    held = Property(True).with_keyframes(
        [Keyframe(0, True, interp="hold"), Keyframe(1, False, interp="hold")]
    )
    assert held.value_at(0.5) is True
    assert held.value_at(1) is False
    with pytest.raises(ValueError, match="hold"):
        Property(True).with_keyframes([Keyframe(0, True), Keyframe(1, False)])


def test_path_normalization_and_matching_topology():
    path = PathValue([[0, 0], [1, 1]])
    prop = Property(path).with_keyframes(
        [Keyframe(0, path), Keyframe(1, PathValue([[2, 0], [3, 1]]))]
    )
    assert prop.value_at(0.5).vertices == ((1, 0), (2, 1))
    with pytest.raises(ValueError, match="topology"):
        Property(path).with_keyframes(
            [Keyframe(0, path), Keyframe(1, PathValue([[2, 0]]))]
        )


@pytest.mark.parametrize(
    "value,kind",
    [
        (True, "float"),
        (float("inf"), "float"),
        ([1], "vec2"),
        ([1, False], "vec2"),
        (object(), None),
        ([[1, 2], [3]], None),
    ],
)
def test_invalid_values_reject(value, kind):
    with pytest.raises((TypeError, ValueError)):
        Property(value, value_type=kind)


@pytest.mark.parametrize("time", [True, "1", float("nan"), float("inf")])
def test_invalid_time(time):
    with pytest.raises((TypeError, ValueError)):
        Property(0).value_at(time)


def test_enum_uses_inert_token():
    class Mode(Enum):
        SCREEN = "screen"

    assert Property(Mode.SCREEN).value_at(0) == "screen"
