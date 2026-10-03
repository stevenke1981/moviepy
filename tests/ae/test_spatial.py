"""Spatial arc-length and roving acceptance tests."""

import numpy as np
import pytest

from moviepy.ae.properties.keyframe import (
    Keyframe,
    evaluate_keyframes,
    normalize_keyframes,
)
from moviepy.ae.properties.spatial import SpatialSegment, resolve_roving_times


def test_curved_arc_length_equal_speed():
    path = SpatialSegment((0, 0), (100, 0), (0, 200), (0, -150))
    values = np.array([path.point_at_fraction(t) for t in np.linspace(0.05, 0.95, 901)])
    speeds = np.linalg.norm(np.diff(values, axis=0), axis=1)
    assert np.ptp(speeds) / np.mean(speeds) < 0.01
    assert path.length > 100
    assert path.point(0) == (0, 0)
    assert path.point(1) == (100, 0)


def test_spatial_evaluator_and_roving_lengths():
    keys = normalize_keyframes(
        [Keyframe(0, (0, 0)), Keyframe(0.5, (10, 0), roving=True), Keyframe(4, (40, 0))]
    )
    resolved = resolve_roving_times(keys)
    assert resolved[1].time == pytest.approx(1)
    assert evaluate_keyframes(resolved, 2, spatial=True) == pytest.approx((20, 0))
    with pytest.raises(ValueError):
        evaluate_keyframes(keys, 1)


def test_roving_degenerate_and_conflict():
    keys = normalize_keyframes(
        [Keyframe(0, (1, 1)), Keyframe(0.1, (1, 1), roving=True), Keyframe(2, (1, 1))]
    )
    assert resolve_roving_times(keys)[1].time == 1
    assert SpatialSegment((1, 1), (1, 1)).point_at_fraction(0.4) == (1, 1)
    bad = normalize_keyframes(
        [
            Keyframe(0, (0, 0), interp="hold"),
            Keyframe(1, (1, 1), roving=True),
            Keyframe(2, (2, 2)),
        ]
    )
    with pytest.raises(ValueError):
        resolve_roving_times(bad)


def test_spatial_temporal_ease_and_speed():
    keys = normalize_keyframes(
        [Keyframe(0, (0, 0), interp="bezier"), Keyframe(1, (10, 0))]
    )
    assert evaluate_keyframes(keys, 0.2, spatial=True)[0] == pytest.approx(
        10 * 0.104, abs=1e-3
    )
    moving = normalize_keyframes(
        [
            Keyframe(0, (0, 0), interp="bezier", out_speed=2),
            Keyframe(1, (10, 0), in_speed=2),
        ]
    )
    h = 0.001
    assert evaluate_keyframes(moving, h, spatial=True)[0] / h == pytest.approx(
        2, rel=0.02
    )
    negative = normalize_keyframes(
        [Keyframe(0, (0, 0), interp="bezier", out_speed=-1), Keyframe(1, (1, 1))]
    )
    with pytest.raises(ValueError):
        evaluate_keyframes(negative, 0.5, spatial=True)


def test_explicit_ease_and_three_dimensions():
    from moviepy.ae.properties.easing import Ease

    keys = normalize_keyframes([(0, (0, 0, 0), Ease.ease_in()), (1, (1, 2, 3))])
    np.testing.assert_allclose(
        evaluate_keyframes(keys, 0.5, spatial=True),
        np.array([1, 2, 3]) * Ease.ease_in()(0.5),
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        resolve_roving_times(())
    with pytest.raises(ValueError):
        resolve_roving_times((Keyframe(0, (0, 0), roving=True),))
    with pytest.raises(TypeError):
        SpatialSegment(3, (1, 2))


@pytest.mark.parametrize(
    "args", [((0,), (1,)), ((0, 0), (1, 1, 1)), ((0, 0), (1, 1), (float("nan"), 0))]
)
def test_invalid_spatial(args):
    with pytest.raises((TypeError, ValueError)):
        SpatialSegment(*args)


def test_partial_zero_length_roving_rejects_duplicate_times():
    keys = normalize_keyframes(
        [Keyframe(0, (0, 0)), Keyframe(1, (0, 0), roving=True), Keyframe(2, (10, 0))]
    )
    with pytest.raises(ValueError, match="zero-length"):
        resolve_roving_times(keys)


def test_spatial_control_and_length_overflow_rejected():
    with pytest.raises(ValueError, match="finite"):
        SpatialSegment((1e308, 0), (1e308, 1), (1e308, 0))
    with pytest.raises(ValueError, match="finite"):
        SpatialSegment((-1e308, 0), (1e308, 0))


def test_large_finite_spatial_endpoint_is_exact():
    path = SpatialSegment((1e150, 0), (1e150, 1))
    assert path.point(0) == path.start
    assert path.point(1) == path.end
    assert np.isfinite(path.length)
