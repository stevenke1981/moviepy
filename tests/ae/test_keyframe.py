"""Numeric keyframe interpolation and metadata acceptance tests."""

import numpy as np

import pytest

from moviepy.ae.properties.easing import Ease
from moviepy.ae.properties.keyframe import (
    Keyframe,
    evaluate_keyframes,
    normalize_keyframes,
)


@pytest.mark.parametrize(
    "interp", ["linear", "bezier", "auto_bezier", "continuous_bezier"]
)
def test_interpolation_endpoints_midpoint(interp):
    keys = normalize_keyframes(
        [Keyframe(0, 0, interp=interp), Keyframe(2, 10, interp=interp)]
    )
    assert evaluate_keyframes(keys, -1) == 0
    assert evaluate_keyframes(keys, 1) == pytest.approx(5)
    assert evaluate_keyframes(keys, 3) == 10
    assert all(evaluate_keyframes(keys, t) >= 0 for t in np.linspace(0, 2, 50))


def test_hold_shorthand_and_ease_override():
    keys = normalize_keyframes([Keyframe(0, False, interp="hold"), Keyframe(1, True)])
    assert evaluate_keyframes(keys, 0.99) is False
    assert evaluate_keyframes(keys, 1) is True
    eased = normalize_keyframes([(0, 0, Ease.ease_in()), (1, 1)])
    assert evaluate_keyframes(eased, 0.5) == pytest.approx(Ease.ease_in()(0.5))


def test_temporal_speed_controls_derivative():
    keys = normalize_keyframes(
        [
            Keyframe(0, 0, interp="bezier", out_speed=3, out_influence=25),
            Keyframe(2, 10, in_speed=1, in_influence=40),
        ]
    )
    epsilon = 1e-5
    assert evaluate_keyframes(keys, epsilon) / epsilon == pytest.approx(3, rel=1e-4)
    assert (10 - evaluate_keyframes(keys, 2 - epsilon)) / epsilon == pytest.approx(
        1, rel=1e-4
    )


def test_vectors_and_continuous_derivative():
    keys = normalize_keyframes(
        [
            Keyframe(0, (0, 0), interp="continuous_bezier"),
            Keyframe(1, (1, 2), interp="continuous_bezier"),
            Keyframe(3, (4, 5), interp="continuous_bezier"),
        ]
    )
    h = 1e-5
    before = (
        np.asarray(evaluate_keyframes(keys, 1)) - evaluate_keyframes(keys, 1 - h)
    ) / h
    after = (
        np.asarray(evaluate_keyframes(keys, 1 + h)) - evaluate_keyframes(keys, 1)
    ) / h
    np.testing.assert_allclose(before, after, atol=1e-4)
    assert isinstance(evaluate_keyframes(keys, 0.5), tuple)


@pytest.mark.parametrize("values", [(0, 0.1, 100), (100, 0.1, 0), (0, 0, 10)])
def test_auto_bezier_preserves_monotone_uneven_values(values):
    keys = normalize_keyframes(
        [Keyframe(t, value, interp="auto_bezier") for t, value in enumerate(values)]
    )
    samples = np.array([evaluate_keyframes(keys, t) for t in np.linspace(0, 2, 501)])
    direction = 1 if values[-1] >= values[0] else -1
    assert np.all(direction * np.diff(samples) >= -1e-12)
    assert samples.min() >= min(values) - 1e-12
    assert samples.max() <= max(values) + 1e-12


def test_auto_shared_derivative_with_extreme_influence():
    keys = normalize_keyframes(
        [
            Keyframe(0, 0, interp="auto_bezier"),
            Keyframe(1, 0.1, interp="auto_bezier", in_influence=100),
            Keyframe(2, 100, interp="auto_bezier"),
        ]
    )
    h = 1e-6
    incoming = (0.1 - evaluate_keyframes(keys, 1 - h)) / h
    outgoing = (evaluate_keyframes(keys, 1 + h) - 0.1) / h
    assert incoming == pytest.approx(outgoing, rel=0.005)


def test_large_linear_values_use_finite_stable_interpolation():
    keys = normalize_keyframes([(0, 1e308), (1, -1e308)])
    assert evaluate_keyframes(keys, 0.5) == 0
    assert evaluate_keyframes(keys, 0.25) == pytest.approx(5e307)


def test_metadata_roundtrip_and_immutable_tangents():
    tangent = [2, 3]
    key = Keyframe(1, (2, 4), in_tangent=tangent, out_tangent=(-1, 0), in_speed=(1, 2))
    tangent[0] = 99
    assert key.in_tangent == (2, 3)
    assert Keyframe.from_dict(key.to_dict()) == key
    eased = Keyframe(0, (0, 1), out_ease=Ease.ease_out())
    assert Keyframe.from_dict(eased.to_dict()) == eased


def test_default_incoming_bezier_speed_and_vector_speed():
    keys = normalize_keyframes([Keyframe(0, 0, interp="bezier"), Keyframe(1, 1)])
    assert evaluate_keyframes(keys, 0.2) == pytest.approx(Ease.easy_ease()(0.2))
    keys = normalize_keyframes(
        [
            Keyframe(0, (0, 0), interp="bezier", out_speed=(1, 2)),
            Keyframe(1, (2, 4), in_speed=(1, 2)),
        ]
    )
    h = 1e-5
    np.testing.assert_allclose(
        np.asarray(evaluate_keyframes(keys, h)) / h, (1, 2), rtol=1e-4
    )


def test_continuous_explicit_speed_shares_both_sides():
    keys = normalize_keyframes(
        [
            Keyframe(0, 0, interp="continuous_bezier"),
            Keyframe(1, 1, interp="continuous_bezier", out_speed=2),
            Keyframe(2, 4, interp="continuous_bezier"),
        ]
    )
    h = 1e-5
    assert (1 - evaluate_keyframes(keys, 1 - h)) / h == pytest.approx(2, rel=1e-4)
    assert (evaluate_keyframes(keys, 1 + h) - 1) / h == pytest.approx(2, rel=1e-4)


def test_invalid_serialization_and_discrete_interpolation():
    with pytest.raises(ValueError):
        Keyframe.from_dict({"time": 0})
    with pytest.raises(TypeError):
        evaluate_keyframes(normalize_keyframes([(0, "a"), (1, "b")]), 0.5)
    with pytest.raises(ValueError):
        evaluate_keyframes((), 0)
    with pytest.raises(TypeError):
        normalize_keyframes([(0, 1, 2, 3)])
    with pytest.raises(ValueError):
        normalize_keyframes([Keyframe(0, (0, 0), in_speed=(1, 2, 3))])
    with pytest.raises(ValueError):
        normalize_keyframes([Keyframe(0, (0, 0), in_tangent=(1, 2, 3))])


@pytest.mark.parametrize("kwargs", [{"roving": 1}, {"in_ease": 2}, {"in_tangent": 2}])
def test_metadata_types(kwargs):
    with pytest.raises(TypeError):
        Keyframe(0, 0, **kwargs)


@pytest.mark.parametrize(
    "items",
    [
        [],
        [(0, 1), (0, 2)],
        [(float("nan"), 1)],
        [(0, (1, 2)), (1, (1, 2, 3))],
        [Keyframe(0, 1, roving=True), Keyframe(1, 2)],
    ],
)
def test_invalid_keys(items):
    with pytest.raises((ValueError, TypeError)):
        normalize_keyframes(items)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"interp": "bad"},
        {"out_influence": 101},
        {"in_speed": float("inf")},
        {"time": True},
        {"in_tangent": (1, float("nan"))},
    ],
)
def test_invalid_metadata(kwargs):
    parameters = {"time": 0, "value": 1}
    parameters.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        Keyframe(**parameters)
