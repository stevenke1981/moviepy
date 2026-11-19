"""Schema-one round trips and malicious input rejection."""

import pytest

from moviepy.ae.properties.keyframe import Keyframe
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.values import PathValue


@pytest.mark.parametrize(
    "value,kind",
    [
        (2, "float"),
        ((1, 2), "vec2"),
        ((1, 2, 3), "vec3"),
        ((2, 0.4, 0.5), "color"),
        (True, "bool"),
        ("screen", "enum"),
        (PathValue(((0, 0), (2, 2))), "path"),
    ],
)
def test_static_roundtrip(value, kind):
    prop = Property(value, value_type=kind)
    restored = Property.from_json(prop.to_json())
    assert restored.value_at(1) == prop.value_at(1)
    assert restored.to_json() == prop.to_json()


def test_animation_and_dimensions_roundtrip():
    prop = Property((0, 0)).with_keyframes([Keyframe(0, (0, 0)), Keyframe(2, (2, 4))])
    restored = Property.from_dict(prop.to_dict())
    assert restored.value_at(0.5) == prop.value_at(0.5)
    separated = Property.from_dimensions(prop.separate_dimensions())
    assert Property.from_json(separated.to_json()).value_at(0.5) == (0.5, 1)


def test_callable_serialization_explicitly_rejects():
    with pytest.raises(TypeError, match="callable"):
        Property(lambda t: t).to_json()
    x, _ = Property(lambda t: (t, t), value_type="vec2").separate_dimensions()
    with pytest.raises(TypeError, match="callable"):
        x.to_dict()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.update(schema=True),
        lambda p: p.update(schema=2),
        lambda p: p.update(extra="field"),
        lambda p: p.update(value_type="python"),
        lambda p: p["source"].update(kind="callable"),
        lambda p: p["source"].update(module="os"),
        lambda p: p["source"].update(value=float("inf")),
    ],
)
def test_strict_schema_rejects(mutator):
    payload = Property(2).to_dict()
    mutator(payload)
    with pytest.raises((TypeError, ValueError)):
        Property.from_dict(payload)


def test_duplicate_json_and_size_limits():
    text = Property(2).to_json().replace('"schema": 1', '"schema": 1, "schema": 1')
    with pytest.raises(ValueError, match="duplicate"):
        Property.from_json(text)
    with pytest.raises(ValueError):
        Property.from_json(" " * 1000001)
    with pytest.raises(ValueError):
        Property.from_json('{"schema": NaN}')


def test_decoded_data_detaches_from_payload():
    payload = Property((1, 2)).to_dict()
    restored = Property.from_dict(payload)
    payload["source"]["value"][0] = 99
    assert restored.value_at(0) == (1, 2)


def test_missing_payload_and_top_level_type():
    for payload in (None, [], {}, {"schema": 1}):
        with pytest.raises((ValueError, TypeError)):
            Property.from_dict(payload)
