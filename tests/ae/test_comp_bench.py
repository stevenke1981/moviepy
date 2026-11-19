"""Smoke tests for the WS-03..06 compositing benchmark CLI and its JSON."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "ae_comp_bench.py"


def _module():
    """Load the script without running it (no numeric imports happen)."""
    spec = importlib.util.spec_from_file_location("ae_comp_bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(*args, timeout=60):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def test_tiny_run_writes_complete_json(tmp_path):
    """Every case runs at 64x36 and the report has the documented schema."""
    output = tmp_path / "tiny.json"
    result = _run(
        *("--size", "64x36", "--warmup", "0", "--repeats", "1", "--frames", "1"),
        *("--modes", "multiply", "hue", "stencil_alpha"),
        *("--output", str(output)),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert report["benchmark"] == "ws03-06-compositing"
    assert (report["config"]["width"], report["config"]["height"]) == (64, 36)
    assert set(report["cases"]) == set(_module().CASES)
    modes = report["cases"]["blend_modes"]["modes"]
    assert set(modes) == {"normal", "multiply", "hue", "stencil_alpha"}
    assert modes["normal"]["ratio_vs_normal"] == 1.0
    assert modes["multiply"]["target"] == "<= 2.0x normal"
    assert modes["stencil_alpha"]["meets_target"] is None
    for name in _module().COMP_CASES:
        case = report["cases"][name]
        assert case["frame_shape"] == [36, 64, 3]
        assert case["deterministic"]
        assert case["samples"][0]["frames"] == 1
        assert case["median_ms"] > 0
    summary = report["summary"]
    assert summary["gated"] is False
    assert summary["targets_checked"] == 4  # multiply, hue, 20 layers, blur
    assert summary["targets_met"] + len(summary["targets_missed"]) == 4
    assert report["threads"]["environment"]["OMP_NUM_THREADS"] == "1"


def test_cli_rejects_existing_output_and_bad_size(tmp_path):
    """Prior evidence is never overwritten; invalid sizes exit with 2."""
    output = tmp_path / "prior.json"
    output.write_text("prior evidence", encoding="utf-8")
    assert _run("--output", str(output)).returncode == 2
    assert output.read_text(encoding="utf-8") == "prior evidence"
    bad = _run("--size", "8x8")
    assert bad.returncode == 2 and "at least 16x16" in bad.stderr


def test_targets_are_recorded_without_gating():
    """Ratios and pass/fail flags follow docs/ae/AE_REVIEW_PLAN.md section 3."""
    bench = _module()
    modes = {
        "normal": {"kind": "normal", "median_ms": 10.0},
        "multiply": {"kind": "separable", "median_ms": 20.0},
        "hue": {"kind": "separable", "median_ms": 20.1},
        "dissolve": {"kind": "dissolve", "median_ms": 50.0},
    }
    bench._blend_targets(modes)
    assert modes["multiply"]["meets_target"] is True
    assert modes["hue"]["meets_target"] is False
    assert modes["dissolve"]["meets_target"] is None
    cases = {
        "blend_modes": {"modes": modes},
        "comp_20_layers": {"meets_target": True},
        "echo_single": {"meets_target": None},
    }
    summary = bench._summary(cases)
    assert summary["targets_checked"] == 3
    assert summary["targets_missed"] == ["blend:hue"]
