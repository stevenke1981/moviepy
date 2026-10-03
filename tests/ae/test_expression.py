"""Analytic and safety acceptance for the finite expression language."""

import math

import numpy as np
import pytest

from moviepy.ae.context import RenderContext
from moviepy.ae.properties.expression import Expression, ExpressionError


@pytest.mark.parametrize(
    "source,expected",
    [
        ("time * 2 + value", 7),
        ("[1,2] + [3,4]", (4, 6)),
        ("[1,2] * 3", (3, 6)),
        ("value if time > 1 and index == 2 else 0", 3),
        ("linear(time,0,4,0,8)", 4),
        ("ease(time,0,4,0,8)", 4),
        ("easeIn(time,0,4,0,8)", 3),
        ("easeOut(time,0,4,0,8)", 5),
        ("clamp([0,2,4],1,3)", (1, 2, 3)),
        ("length([3,4])", 5),
        ("length([1,1],[4,5])", 5),
        ("normalize([3,4])", (0.6, 0.8)),
        ("dot([1,2],[3,4])", 11),
        ("cross([1,0,0],[0,1,0])", (0, 0, 1)),
        ("lookAt([0,0,0],[0,0,1])", (0, 0, 0)),
        ("rgbToHsl([1,0,0,.5])", (0, 1, 0.5, 0.5)),
        ("hslToRgb([0,1,.5])", (1, 0, 0)),
        ("timeToFrames(time)", 60),
        ("framesToTime(60)", 2),
        ("sin(pi/2) + cos(0) + sqrt(4)", 4),
        ("floor(1.5)+ceil(1.5)+round(1.5)", 5),
        ("max(1,2)+min(1,2)+abs(-3)", 6),
        ("[1,2,3][1]", 2),
        ("not False", True),
        ("2 ** 4 // 3 % 3", 2),
    ],
)
def test_expression_analytics(source, expected):
    actual = Expression(source).evaluate(2, value=3, index=2)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os')",
        "open('sentinel')",
        "().__class__",
        "value.__class__",
        "thisLayer.__dict__",
        "eval('1')",
        "getattr(value,'real')",
        "value()",
        "thisComp.layer('x')",
        "(lambda:1)()",
        "[x for x in [1]]",
        "(x for x in [1])",
        "(x:=1)",
        "f'{value}'",
        "random(*value)",
        "random(**thisComp)",
        "2 << 100",
        "{'a':1}",
        "random.__call__()",
    ],
)
def test_unsafe_syntax(source):
    with pytest.raises(ExpressionError):
        Expression(source).evaluate(0)


@pytest.mark.parametrize(
    "source",
    [
        "2 ** 1000000000",
        "2 ** (2 ** 100)",
        "'a'*1000000000",
        "wiggle(1,1,octaves=1000000000)",
        "1/0",
        "sqrt(-1)",
        "[1,2]+[1,2,3]",
        "unknown",
        "random(nope=2)",
        "value[100]",
        "10**32 * 10**32 * 10**32 * 10**32",
    ],
)
def test_runtime_rejection(source):
    with pytest.raises(ExpressionError):
        Expression(source).evaluate(0)


def test_snapshot_reads_and_detachment():
    data = {"name": "hero", "transform": {"position": [2, 3]}}
    expr = Expression("thisLayer.transform.position[0] + thisComp.width")
    assert expr.evaluate(0, this_layer=data, this_comp={"width": 10}) == 12
    assert Expression("thisLayer.name").evaluate(0, this_layer=data) == "hero"


def test_hostile_objects_never_invoke_protocols():
    calls = []

    class Hostile:
        def __float__(self):
            calls.append("float")
            raise RuntimeError

        def __add__(self, other):
            calls.append("add")
            raise RuntimeError

    for bindings in (
        {"value": Hostile()},
        {"this_layer": {"x": Hostile()}},
        {"value": np.array([Hostile()], dtype=object)},
    ):
        with pytest.raises(ExpressionError):
            Expression("value+1").evaluate(0, **bindings)
    assert not calls


def test_limits():
    for source in (
        "1" * 8193,
        "-" * 100 + "1",
        "[" + ",".join(["1"] * 65) + "]",
        "'" + "x" * 257 + "'",
        "1+" * 300 + "1",
    ):
        with pytest.raises(ExpressionError):
            Expression(source)
    with pytest.raises(ExpressionError):
        Expression("value").evaluate(math.inf)


def test_vector_multiplication_never_repeats_sequences():
    assert Expression("[0]*1000000000").evaluate(0) == (0.0,)


def test_schema_roundtrip():
    expr = Expression("time + value")
    data = {"schema": 1, "type": "expression", "source": "time + value"}
    assert expr.to_dict() == data
    assert Expression.from_dict(data).evaluate(2, value=3) == 5
    for invalid in (
        {**data, "schema": True},
        {**data, "extra": 1},
        {},
        {**data, "type": "callable"},
    ):
        with pytest.raises(ExpressionError):
            Expression.from_dict(invalid)


def test_random_noise_reproducibility_and_context():
    for source in (
        "random()",
        "gaussRandom(-2,2)",
        "noise([.2,.3,.4])",
        "wiggle(2,4)",
        "[seedRandom(5,True),random()][1]",
    ):
        expr = Expression(source)
        ctx = RenderContext(rng_seed=9)
        assert expr.evaluate(0.3, context=ctx) == expr.evaluate(0.3, context=ctx)
        assert expr.evaluate(0.3, context=ctx) != expr.evaluate(
            0.3, context=RenderContext(rng_seed=10)
        )
    expr = Expression("[seedRandom(5,True),random()][1]")
    assert expr.evaluate(0.1) == expr.evaluate(1.1)
    assert Expression("wiggle(0,3)").evaluate(0.2, value=5) == 5


def test_wiggle_frequency_and_continuity():
    expr = Expression("wiggle(3,10)")
    times = np.arange(2048) / 100
    samples = np.array([expr.evaluate(float(t)) for t in times])
    energy = np.abs(np.fft.rfft(samples)) ** 2
    frequencies = np.fft.rfftfreq(len(samples), 0.01)
    assert energy[(frequencies > 2.5) & (frequencies < 3.5)].sum() / energy.sum() > 0.9
    assert abs(expr.evaluate(0.999999) - expr.evaluate(1.000001)) < 0.001
    vector = Expression("wiggle(2,[3,4],octaves=2)").evaluate(0.3, value=(1, 2))
    assert len(vector) == 2


@pytest.mark.parametrize(
    "kind,t,expected",
    [
        ("cycle", 2.5, 5),
        ("pingpong", 2.5, 15),
        ("offset", 2.5, 25),
        ("continue", 2.5, 25),
    ],
)
def test_loop_out(kind, t, expected):
    from moviepy.ae.properties.property import Property

    prop = Property(keyframes=[(0, 0), (2, 20)])
    result = Expression(f"loopOut('{kind}')").evaluate(t, value=20, property=prop)
    assert result == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    "kind,t,expected",
    [
        ("cycle", -0.5, 15),
        ("pingpong", -0.5, 5),
        ("offset", -0.5, -5),
        ("continue", -0.5, -5),
    ],
)
def test_loop_in(kind, t, expected):
    from moviepy.ae.properties.property import Property

    prop = Property(keyframes=[(0, 0), (2, 20)])
    result = Expression(f"loopIn('{kind}')").evaluate(t, property=prop)
    assert result == pytest.approx(expected, abs=0.01)


def test_base_callbacks_bypass_expression():
    from moviepy.ae.properties.property import Property

    prop = Property(
        keyframes=[(0, 0), (2, 20)],
        expression=Expression("valueAtTime(time)+velocityAtTime(time)"),
    )
    assert prop.value_at(1) == pytest.approx(20)
