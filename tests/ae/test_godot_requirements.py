"""Check Godot requirement declarations and the check without a real Godot."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from moviepy.ae.three_d import requirements
from moviepy.ae.three_d.requirements import (
    GODOT_REQUIREMENTS,
    _parse_version,
    check_godot,
    main,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CREATE_NO_WINDOW = 0x08000000


class FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


@pytest.fixture
def fake_godot(monkeypatch, tmp_path):
    """Provide a Windows platform, an existing executable and a scripted version."""
    binary = tmp_path / "Godot_v4.7.2-stable_win64.exe"
    binary.touch()
    calls = []
    state = {"stdout": "4.7.2.stable.official.abc123", "returncode": 0}

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return FakeCompleted(stdout=state["stdout"], returncode=state["returncode"])

    monkeypatch.setattr(requirements, "_current_platform", lambda: "win32")
    monkeypatch.setattr(requirements, "_is_windows", lambda: True)
    monkeypatch.setattr(requirements, "find_godot", lambda executable=None: binary)
    monkeypatch.setattr(requirements.subprocess, "run", fake_run)

    class Handle:
        path = binary
        run_calls = calls

        def set_stdout(self, text):
            state["stdout"] = text

        def set_returncode(self, code):
            state["returncode"] = code

    return Handle()


def test_pyproject_godot_table_matches_requirements():
    try:
        import tomllib
    except ImportError:  # Python 3.9 and 3.10
        tomllib = pytest.importorskip("tomli")
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        table = tomllib.load(handle)["tool"]["moviepy"]["godot"]

    minimum = ".".join(map(str, GODOT_REQUIREMENTS["min_version"]))
    upper = ".".join(map(str, GODOT_REQUIREMENTS["max_version_exclusive"]))
    assert table["min_version"] == minimum
    assert table["max_version_exclusive"] == upper
    assert tuple(table["platforms"]) == GODOT_REQUIREMENTS["platforms"]
    assert table["renderer"] == GODOT_REQUIREMENTS["renderer_id"]
    assert table["env"] == GODOT_REQUIREMENTS["executable_env"]
    assert GODOT_REQUIREMENTS["max_major"] == 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4.7.2.stable.official.abc123", (4, 7, 2)),
        ("4.7.stable", (4, 7, 0)),
        ("3.6.2.stable.official", (3, 6, 2)),
        ("garbage", None),
        ("", None),
    ],
)
def test_parse_version(text, expected):
    assert _parse_version(text) == expected


def test_missing_executable_reports_problem_without_raising(monkeypatch):
    def missing(executable=None):
        raise FileNotFoundError("Set MOVIEPY_GODOT to an installed Godot executable")

    def no_run(*args, **kwargs):
        raise AssertionError("version must not be probed without an executable")

    monkeypatch.setattr(requirements, "_current_platform", lambda: "win32")
    monkeypatch.setattr(requirements, "find_godot", missing)
    monkeypatch.setattr(requirements.subprocess, "run", no_run)

    result = check_godot()
    assert result.ok is False
    assert result.executable is None
    assert result.version is None
    assert result.version_tuple is None
    assert result.platform_ok is True
    assert result.version_ok is False
    assert any("MOVIEPY_GODOT" in problem for problem in result.problems)


@pytest.mark.parametrize(
    ("version_text", "ok", "version_tuple"),
    [
        ("3.6.2.stable.official.abc123", False, (3, 6, 2)),
        ("4.6.1.stable.official.abc123", False, (4, 6, 1)),
        ("4.7.0.stable.official.abc123", True, (4, 7, 0)),
        ("4.7.2.stable.official.abc123", True, (4, 7, 2)),
        ("5.0.0.stable.official.abc123", False, (5, 0, 0)),
    ],
)
def test_version_range_is_enforced(fake_godot, version_text, ok, version_tuple):
    fake_godot.set_stdout(version_text + "\n")
    result = check_godot(fake_godot.path)
    assert result.ok is ok
    assert result.version_ok is ok
    assert result.version_tuple == version_tuple
    assert result.platform_ok is True
    assert result.executable == str(fake_godot.path)
    if ok:
        assert result.version == version_text
        assert result.problems == []
    else:
        assert result.version == version_text
        assert any("is required" in problem for problem in result.problems)


def test_unparseable_version_is_a_problem(fake_godot):
    fake_godot.set_stdout("not a godot")
    result = check_godot(fake_godot.path)
    assert result.ok is False
    assert result.version_tuple is None
    assert any("cannot parse" in problem for problem in result.problems)


def test_nonzero_exit_is_a_problem(fake_godot):
    fake_godot.set_returncode(1)
    result = check_godot(fake_godot.path)
    assert result.ok is False
    assert any("exited with code 1" in problem for problem in result.problems)


def test_version_probe_hides_window_and_passes_timeout(fake_godot):
    check_godot(fake_godot.path, timeout=7)
    args, kwargs = fake_godot.run_calls[0]
    assert args == [str(fake_godot.path), "--version"]
    assert kwargs["creationflags"] == CREATE_NO_WINDOW
    assert kwargs["timeout"] == 7


def test_timeout_is_reported(monkeypatch, tmp_path):
    binary = tmp_path / "godot.exe"
    binary.touch()

    def slow_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(requirements, "_current_platform", lambda: "win32")
    monkeypatch.setattr(requirements, "find_godot", lambda executable=None: binary)
    monkeypatch.setattr(requirements.subprocess, "run", slow_run)
    result = check_godot(timeout=3)
    assert result.ok is False
    assert any("timed out after 3" in problem for problem in result.problems)


def test_non_windows_platform_skips_probe(monkeypatch, tmp_path):
    binary = tmp_path / "godot"
    binary.touch()

    def no_run(*args, **kwargs):
        raise AssertionError("must not run Godot on a non-Windows platform")

    monkeypatch.setattr(requirements, "_current_platform", lambda: "linux")
    monkeypatch.setattr(requirements, "find_godot", lambda executable=None: binary)
    monkeypatch.setattr(requirements.subprocess, "run", no_run)
    result = check_godot(binary)
    assert result.ok is False
    assert result.platform_ok is False
    assert result.version is None
    assert result.version_ok is False
    assert any("win32" in problem for problem in result.problems)


def test_cli_json_success_exit_zero(fake_godot, capsys):
    code = main(["--json", "--executable", str(fake_godot.path)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["check"]["ok"] is True
    assert payload["check"]["version_tuple"] == [4, 7, 2]
    assert payload["requirements"]["min_version"] == [4, 7]
    assert payload["requirements"]["renderer"] == "Forward+ (Vulkan)"


def test_cli_json_old_version_exit_one(fake_godot, capsys):
    fake_godot.set_stdout("4.6.1.stable.official")
    code = main(["--json", "--executable", str(fake_godot.path)])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["check"]["ok"] is False
    assert payload["check"]["version_ok"] is False


def test_cli_text_output_for_missing_godot(monkeypatch, capsys):
    def missing(executable=None):
        raise FileNotFoundError("Set MOVIEPY_GODOT to an installed Godot executable")

    monkeypatch.setattr(requirements, "_current_platform", lambda: "win32")
    monkeypatch.setattr(requirements, "find_godot", missing)
    code = main([])
    out = capsys.readouterr().out
    assert code == 1
    assert "Godot requirements" in out
    assert "result:       FAILED" in out
    assert "MOVIEPY_GODOT" in out


def test_module_entry_point_reports_missing_godot(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["MOVIEPY_GODOT"] = str(tmp_path / "missing-godot.exe")
    completed = subprocess.run(
        [sys.executable, "-m", "moviepy.ae.three_d.requirements", "--json"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 1, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["check"]["ok"] is False
    assert payload["check"]["executable"] is None
