"""Boundary regressions for helpers, inert values and controlled sampling."""

import numpy as np

import pytest

from moviepy.ae.properties.expression import (
    Expression,
    ExpressionError,
    ExpressionSnapshot,
)


@pytest.mark.parametrize(
    "source",
    [
        "length(1)",
        "cross([1,2],[1,2])",
        "clamp(1,3,2)",
        "linear(1,1,1,0,1)",
        "dot([1],[1,2])",
        "rgbToHsl([1,2])",
        "rgbToHsl([2,0,0])",
        "hslToRgb([1,2])",
        "hslToRgb([0,2,.5])",
        "framesToTime(1,fps=0)",
        "timeToFrames(1,fps=-1)",
        "min()",
        "max()",
        "seedRandom(1,timeless=3)",
        "random(2,1)",
        "gaussRandom(2,1)",
        "noise([1,2,3,4])",
        "wiggle(-1,1)",
        "wiggle(1,1,amp_mult=2)",
        "loopOut('wrong')",
        "loopIn(n=-1)",
        "thisLayer.missing",
        "thisLayer['_fields']",
        "thisLayer == thisComp",
        "value[.1]",
        "None",
        "thisComp",
        "length([])",
        "[1,[2]]+1",
    ],
)
def test_rejected_helper_edges(source):
    with pytest.raises(ExpressionError):
        Expression(source).evaluate(0)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("min([3,2,1])", 1),
        ("max([3,2,1])", 3),
        ("normalize([0,0])", (0, 0)),
        ("lookAt([0,0,0],[0,0,0])", (0, 0, 0)),
        ("True or unknown", True),
        ("False and unknown", False),
        ("+3", 3),
        ("degreesToRadians(180)", np.pi),
        ("radiansToDegrees(pi)", 180),
        ("velocityAtTime(0)", 0),
        ("valueAtTime(1)", 0),
        ("loopOut()", 0),
        ("thisLayer['name']", "hero"),
    ],
)
def test_valid_helper_edges(source, expected):
    actual = Expression(source).evaluate(0, this_layer={"name": "hero"})
    assert actual == (
        expected if isinstance(expected, str) else pytest.approx(expected)
    )


def test_numpy_subclasses_and_inert_bindings():
    calls = []

    class Hostile(np.float64):
        def __float__(self):
            calls.append("float")
            return 1

    for bindings in (
        {"t": Hostile(1)},
        {"index": Hostile(1)},
        {"value": Hostile(1)},
        {"context": {}},
        {"property": object()},
        {"layer_id": True},
        {"layer_id": -1},
        {"layer_id": ""},
        {"this_layer": 1},
    ):
        with pytest.raises(ExpressionError):
            Expression("value").evaluate(**{"t": 0, **bindings})
    assert not calls
    assert Expression("value").evaluate(0, value=np.array([1.0, 2.0])) == (1, 2)
    with pytest.raises(ExpressionError):
        Expression("value").evaluate(0, value=np.ones((2, 2)))


def test_snapshot_validation_and_immutability():
    snapshot = ExpressionSnapshot({"position": [1, 2]})
    assert Expression("thisLayer.position[1]").evaluate(0, this_layer=snapshot) == 2
    with pytest.raises(ExpressionError):
        snapshot.position = 3
    for data in ([], {"_private": 1}, {"x" * 257: 1}, {str(i): 1 for i in range(65)}):
        with pytest.raises(ExpressionError):
            ExpressionSnapshot(data)
    for _ in range(12):
        try:
            snapshot = ExpressionSnapshot({"child": snapshot})
        except ExpressionError:
            break
    else:
        pytest.fail("nested snapshots bypassed depth limit")


def test_callback_budget_and_hostile_results():
    from moviepy.ae.properties.property import Property

    source = "[" + ",".join(["valueAtTime(0)"] * 33) + "]"
    with pytest.raises(ExpressionError):
        Expression(source).evaluate(0, property=Property(1))
    assert (
        Expression("loopOut()").evaluate(0, property=Property(keyframes=[(0, 1)])) == 1
    )


def test_syntax_and_value_limits():
    for source in ("(", "random(x=1,x=2)", "1e101", "None.real"):
        with pytest.raises(ExpressionError):
            Expression(source).evaluate(0)
    with pytest.raises(ExpressionError):
        Expression("value").evaluate(0, value=10**101)
    with pytest.raises(ExpressionError):
        Expression("value").evaluate(0, value=tuple([0] * 65))


@pytest.mark.parametrize(
    "name,start_slope,end_slope",
    [
        ("ease", 0, 0),
        ("easeIn", 0, 1),
        ("easeOut", 1, 0),
    ],
)
def test_interpolation_endpoint_derivatives(name, start_slope, end_slope):
    expression = Expression(f"{name}(time,0,1,0,1)")
    step = 1e-6
    start = (expression.evaluate(step) - expression.evaluate(0)) / step
    end = (expression.evaluate(1) - expression.evaluate(1 - step)) / step
    assert start == pytest.approx(start_slope, abs=1e-5)
    assert end == pytest.approx(end_slope, abs=1e-5)
