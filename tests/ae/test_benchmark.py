"""Benchmark CLI checks keep diagnostic runs distinct from acceptance."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "ae_bench.py"


def _module():
    """Load the script without importing numeric libraries."""
    spec = importlib.util.spec_from_file_location("ae_bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptance_requires_full_protocol_and_primary_threshold():
    """A fast partial result cannot pass the WS-00 acceptance gate."""
    benchmark = _module()
    config = {
        "width": 1920,
        "height": 1080,
        "warmup": 5,
        "repeats": 5,
        "frames": 20,
        "cases": list(benchmark.CASES),
    }
    cases = {name: {"correct": True, "median_fps": 20.0} for name in benchmark.CASES}
    assert benchmark.acceptance(config, cases)["passed"]
    cases["normal_10"]["median_fps"] = 9.99
    assert not benchmark.acceptance(config, cases)["passed"]
    cases["normal_10"]["median_fps"] = 20.0
    config["frames"] = 1
    assert not benchmark.acceptance(config, cases)["passed"]
    config["frames"] = 20
    cases["normal_10"]["correct"] = False
    assert not benchmark.acceptance(config, cases)["passed"]
    cases.pop("gaussian_sigma20")
    assert not benchmark.acceptance(config, cases)["passed"]


def test_diagnostic_cli_writes_json_and_check_fails(tmp_path):
    """An actual 1080p one-frame run is correctly labelled diagnostic."""
    output = tmp_path / "diagnostic.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--check",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--frames",
            "1",
            "--cases",
            "single",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["width"] == 1920
    assert report["config"]["height"] == 1080
    assert report["cases"]["single"]["correct"]
    assert report["cases"]["single"]["samples"][0]["frames"] == 1
    assert not report["acceptance"]["passed"]
    assert report["threads"]["environment"]["OPENBLAS_NUM_THREADS"] == "1"


@pytest.mark.parametrize("flag", ["--frames", "--repeats"])
def test_cli_rejects_zero_work(flag):
    """Invalid timing work is rejected before imports or measurements."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), flag, "0"],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 2
    assert "must be positive" in result.stderr


def test_cli_preserves_existing_result(tmp_path):
    """A caller-selected existing result is never overwritten."""
    output = tmp_path / "prior.json"
    output.write_text("prior evidence", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 2
    assert "already exists" in result.stderr
    assert output.read_text(encoding="utf-8") == "prior evidence"


def test_provenance_timeout_is_explicit(monkeypatch):
    """A provenance timeout is bounded and marked unavailable in reports."""
    benchmark = _module()

    def timeout(args, **kwargs):
        assert kwargs["timeout"] == 10
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(benchmark.subprocess, "check_output", timeout)
    result = benchmark._command(["git", "status", "--porcelain=v1"])
    assert result == "unavailable: provenance command timed out after 10s"
