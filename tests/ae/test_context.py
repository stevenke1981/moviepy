"""Render context validation, immutability, and reproducible random streams."""

import doctest
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np

import pytest

import moviepy.ae.context as context_module
from moviepy.ae.context import RenderContext


def test_defaults_and_independent_opaque_defaults():
    first, second = RenderContext(), RenderContext()
    assert first == second
    assert (first.t, first.fps, first.resolution_scale) == (0.0, 30.0, 1.0)
    assert first.quality == "best"
    assert first.rng_seed == 0
    assert first.cache is None and first.shutter is None
    assert first.frame_index == 0


@pytest.mark.parametrize("field", ["t", "fps", "resolution_scale"])
@pytest.mark.parametrize("value", [True, np.bool_(False), "1", None, 1j, []])
def test_real_fields_reject_invalid_types(field, value):
    with pytest.raises(TypeError, match=field):
        RenderContext(**{field: value})


@pytest.mark.parametrize("field", ["t", "fps", "resolution_scale"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_real_fields_reject_nonfinite_values(field, value):
    with pytest.raises(ValueError, match=field):
        RenderContext(**{field: value})


@pytest.mark.parametrize("value", [0, -1, -0.0])
def test_fps_must_be_positive(value):
    with pytest.raises(ValueError, match="fps"):
        RenderContext(fps=value)


@pytest.mark.parametrize("value", [0, 0.3, 0.75, 2, -1])
def test_resolution_scale_is_restricted(value):
    with pytest.raises(ValueError, match="resolution_scale"):
        RenderContext(resolution_scale=value)


@pytest.mark.parametrize("scale", [0.25, 0.5, 1])
@pytest.mark.parametrize("quality", ["draft", "best"])
def test_supported_resolution_and_quality(scale, quality):
    ctx = RenderContext(resolution_scale=scale, quality=quality)
    assert ctx.resolution_scale == scale
    assert ctx.quality == quality


@pytest.mark.parametrize("quality", [None, True, [], 1])
def test_quality_rejects_invalid_types(quality):
    with pytest.raises(TypeError, match="quality"):
        RenderContext(quality=quality)


@pytest.mark.parametrize("quality", ["BEST", "", "preview"])
def test_quality_rejects_unknown_values(quality):
    with pytest.raises(ValueError, match="quality"):
        RenderContext(quality=quality)


@pytest.mark.parametrize("seed", [True, np.bool_(True), 1.0, "1", None, []])
def test_seed_rejects_invalid_types(seed):
    with pytest.raises(TypeError, match="rng_seed"):
        RenderContext(rng_seed=seed)


def test_seed_rejects_negative_values():
    with pytest.raises(ValueError, match="rng_seed"):
        RenderContext(rng_seed=-1)


@pytest.mark.parametrize("field", ["t", "fps", "resolution_scale"])
def test_unrepresentable_real_values_have_field_error(field):
    with pytest.raises(ValueError, match=field):
        RenderContext(**{field: 10**400})


@pytest.mark.parametrize("time", [1e308, -1e308])
def test_finite_time_and_fps_must_have_finite_product(time):
    with pytest.raises(ValueError, match="t.*fps"):
        RenderContext(t=time, fps=30)


def test_numpy_scalars_and_large_seed_are_normalized():
    ctx = RenderContext(
        t=np.float64(-0.5),
        fps=np.float32(24),
        resolution_scale=np.float32(0.5),
        rng_seed=np.uint64(2**63 + 1),
    )
    assert type(ctx.t) is float
    assert type(ctx.fps) is float
    assert type(ctx.resolution_scale) is float
    assert type(ctx.rng_seed) is int
    assert ctx.frame_index == -12
    assert np.array_equal(
        ctx.rng_for(np.int64(3)).integers(0, 100, 16),
        ctx.rng_for(3).integers(0, 100, 16),
    )
    assert RenderContext(rng_seed=10**100).rng_for("large").random(3).shape == (3,)


@pytest.mark.parametrize(
    "time,fps,expected",
    [
        (0, 30, 0),
        (1.5, 24, 36),
        (-0.01, 30, -1),
        (-0.5, 24, -12),
        (np.nextafter(0.5, 0), 2, 0),
        (0.5, 2, 1),
        (np.nextafter(0.5, 1), 2, 1),
        (np.nextafter(-0.5, -1), 2, -2),
        (-0.5, 2, -1),
        (np.nextafter(-0.5, 0), 2, -1),
    ],
)
def test_frame_index_floors_without_snapping(time, fps, expected):
    assert RenderContext(t=time, fps=fps).frame_index == expected


@pytest.mark.parametrize(
    "field", ["t", "fps", "resolution_scale", "quality", "rng_seed", "cache", "shutter"]
)
def test_context_fields_are_frozen(field):
    with pytest.raises(FrozenInstanceError):
        setattr(RenderContext(), field, None)


def test_with_time_preserves_settings_and_opaque_references():
    cache, shutter = {}, []
    original = RenderContext(1, 24, 0.5, "draft", 7, cache, shutter)
    changed = original.with_time(-0.5)
    assert changed is not original
    assert original.t == 1
    assert changed.t == -0.5 and changed.frame_index == -12
    assert (changed.fps, changed.resolution_scale, changed.quality) == (
        24,
        0.5,
        "draft",
    )
    assert changed.rng_seed == 7
    assert changed.cache is cache and changed.shutter is shutter
    with pytest.raises(ValueError, match="t"):
        original.with_time(float("nan"))


def test_opaque_transport_is_not_executed_or_compared():
    class Opaque:
        def __eq__(self, other):
            raise AssertionError("opaque transport must not be compared")

        def __hash__(self):
            raise AssertionError("opaque transport must not be hashed")

        def __call__(self, *args, **kwargs):
            raise AssertionError("opaque transport must not be executed")

    first = RenderContext(cache=Opaque(), shutter=Opaque())
    second = RenderContext(cache={}, shutter=[])
    assert first == second
    assert hash(first) == hash(second)
    assert first.with_time(1).cache is first.cache
    assert first.rng_for("layer").random(4).shape == (4,)


@pytest.mark.parametrize("layer_id", [True, np.bool_(True), None, 1.0, [], object()])
def test_layer_identity_rejects_invalid_types(layer_id):
    with pytest.raises(TypeError, match="layer_id"):
        RenderContext().rng_for(layer_id)


@pytest.mark.parametrize("layer_id", [-1, ""])
def test_layer_identity_rejects_invalid_values(layer_id):
    with pytest.raises(ValueError, match="layer_id"):
        RenderContext().rng_for(layer_id)


@pytest.mark.parametrize("layer_id", [0, 10**100, "foreground", "圖層\n𝄞"])
def test_rng_is_fresh_and_reproducible(layer_id):
    ctx = RenderContext(t=0.25, rng_seed=4)
    first, second = ctx.rng_for(layer_id), ctx.rng_for(layer_id)
    assert first is not second
    assert isinstance(first, np.random.Generator)
    assert isinstance(first.bit_generator, np.random.PCG64)
    expected = second.random(32)
    assert np.array_equal(first.random(32), expected)
    first.random(100)
    assert np.array_equal(ctx.rng_for(layer_id).random(32), expected)


def test_every_seed_input_separates_streams_and_integer_string_ids():
    baseline = RenderContext(t=0, rng_seed=1).rng_for(2).random(32)
    alternatives = [
        RenderContext(t=0, rng_seed=2).rng_for(2),
        RenderContext(t=0, rng_seed=1).rng_for(3),
        RenderContext(t=1 / 30, rng_seed=1).rng_for(2),
        RenderContext(t=0, rng_seed=2).rng_for(1),
        RenderContext(t=0, rng_seed=1).rng_for("2"),
    ]
    assert all(
        not np.array_equal(stream.random(32), baseline) for stream in alternatives
    )


def test_frame_and_layer_evaluation_order_has_no_effect():
    ctx = RenderContext(fps=2, rng_seed=19)
    requests = [(time, layer) for time in [-1, 0, 0.5, 2] for layer in [1, "smoke"]]
    forward = {
        key: ctx.with_time(key[0]).rng_for(key[1]).normal(size=24) for key in requests
    }
    backward = {
        key: ctx.with_time(key[0]).rng_for(key[1]).normal(size=24)
        for key in reversed(requests)
    }
    assert all(np.array_equal(forward[key], backward[key]) for key in requests)
    assert np.array_equal(
        ctx.with_time(0.1).rng_for("smoke").random(24),
        ctx.with_time(0.2).rng_for("smoke").random(24),
    )


def test_rng_does_not_use_or_change_global_numpy_random(monkeypatch):
    before = np.random.get_state()

    def forbidden(*args, **kwargs):
        raise AssertionError("global random API used")

    for name in [
        "seed",
        "random",
        "rand",
        "randint",
        "normal",
        "uniform",
        "default_rng",
    ]:
        monkeypatch.setattr(np.random, name, forbidden)
    RenderContext(t=-0.3, rng_seed=9).rng_for("noise").normal(size=16)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_stream_is_stable_across_process_hash_seeds():
    code = (
        "import json; from moviepy.ae.context import RenderContext; "
        "print(json.dumps(RenderContext(t=-1, rng_seed=9)"
        ".rng_for('圖層').integers(0, 1000000, 16).tolist()))"
    )
    results = []
    for hash_seed in ["0", "987654"]:
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONIOENCODING="utf-8")
        output = subprocess.check_output(
            [sys.executable, "-c", code], env=env, text=True, timeout=30
        )
        results.append(json.loads(output))
    assert results[0] == results[1]


def test_public_context_examples_execute():
    result = doctest.testmod(context_module, raise_on_error=True)
    assert result.failed == 0
    assert result.attempted >= 8
