"""Analytic temporal easing acceptance tests."""

import numpy as np
import pytest

from moviepy.ae.properties.easing import Ease


@pytest.mark.parametrize(
    "name", ["linear", "ease_in", "ease_out", "ease_in_out", "easy_ease"]
)
def test_easing_endpoints_monotonic(name):
    ease = getattr(Ease, name)()
    samples = np.array([ease(t) for t in np.linspace(0, 1, 101)])
    assert samples[0] == 0
    assert samples[-1] == 1
    assert np.all(np.diff(samples) >= 0)
    assert Ease.from_dict(ease.to_dict()) == ease


def test_easy_ease_exact_influence_and_rounded_css_reference():
    exact = Ease.bezier(0.3333, 0, 0.6667, 1)
    rounded = Ease.bezier(0.333, 0, 0.667, 1)
    for t in np.linspace(0, 1, 101):
        assert Ease.easy_ease()(t) == pytest.approx(exact(t), abs=1e-12)
        assert abs(exact(t) - rounded(t)) < 2e-4


def test_css_inversion_and_overshoot():
    ease = Ease.bezier(0, 0, 0, 1)
    assert ease(0.125) == pytest.approx(0.5)
    assert Ease.bezier(0.2, 2, 0.8, 2)(0.5) > 1
    assert Ease.hold()(0.999) == 0
    assert Ease.hold()(1) == 1


@pytest.mark.parametrize(
    "args", [(-0.1, 0, 0.5, 1), (0, 0, 1.1, 1), (0, float("nan"), 1, 1)]
)
def test_bad_controls(args):
    with pytest.raises((ValueError, TypeError)):
        Ease.bezier(*args)


def test_bad_progress_and_schema():
    with pytest.raises(ValueError):
        Ease.linear()(float("inf"))
    with pytest.raises(ValueError):
        Ease.from_dict({"kind": "unknown"})
    with pytest.raises(ValueError):
        Ease.from_dict(dict(Ease.linear().to_dict(), extra=1))
    with pytest.raises(ValueError):
        Ease.linear()(10**1000)
    with pytest.raises(ValueError):
        Ease("linear", (1,))
    with pytest.raises(ValueError):
        Ease.bezier(0, 0, 1, 1).from_dict(
            {"kind": "bezier", "control_points": [0, 0, 1]}
        )


def test_embedded_codec_rejects_live_object():
    class RealLike:
        def __float__(self):
            raise AssertionError("live object executed")

    with pytest.raises((TypeError, ValueError)):
        Ease.from_dict({"kind": "bezier", "control_points": [0, RealLike(), 1, 1]})
