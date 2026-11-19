"""Regression probes for final expression and numeric review findings."""

import sys

import pytest

from moviepy.ae.properties.expression import Expression
from moviepy.ae.properties.keyframe import Keyframe, normalize_keyframes
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.spatial import SpatialSegment, resolve_roving_times
from moviepy.ae.properties.values import PathValue


@pytest.mark.parametrize("name", ["random", "gaussRandom"])
def test_explicit_random_zero_bound(name):
    assert Expression(name + "(0)").evaluate(0) == 0
    assert Expression(name + "()").evaluate(0) != 0


def test_path_value_at_time_uses_inert_bridge():
    first = PathValue(((0, 0), (1, 1)))
    last = PathValue(((2, 0), (3, 1)))
    prop = Property(
        keyframes=[(0, first), (1, last)],
        value_type="path",
        expression=Expression("valueAtTime(.5)"),
    )
    assert prop.value_at(0.7).vertices == ((1, 0), (2, 1))


def test_nonfinite_segment_duration_rejects():
    with pytest.raises(ValueError, match="duration"):
        normalize_keyframes([(-1e308, 0), (1e308, 1)])


def test_maximum_finite_constant_spatial_is_exact():
    maximum = sys.float_info.max
    path = SpatialSegment((maximum, maximum), (maximum, maximum))
    assert path.point(0.074) == (maximum, maximum)
    assert path.point(0.09) == (maximum, maximum)


def test_roving_rounding_cannot_duplicate_time():
    keys = normalize_keyframes(
        [
            Keyframe(1e16, (0, 0)),
            Keyframe(1e16 + 2, (1e-200, 0), roving=True),
            Keyframe(1e16 + 4, (1, 0)),
        ]
    )
    with pytest.raises(ValueError, match="distinct"):
        resolve_roving_times(keys)
