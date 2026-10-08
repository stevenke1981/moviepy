"""Declare and verify the external Godot runtime required by the 3D backend.

Run ``python -m moviepy.ae.three_d.requirements [--executable PATH] [--json]``
to print the requirements and check the Godot executable that would be used.
The check never raises for a missing or unsupported Godot; it reports problems.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass

from moviepy.ae.three_d.godot import find_godot

GODOT_REQUIREMENTS = {
    "engine": "Godot",
    "edition": "standard (not .NET/Mono)",
    "min_version": (4, 7),
    "max_major": 4,
    "max_version_exclusive": (5, 0),
    "platforms": ("win32",),
    "renderer": "Forward+ (Vulkan)",
    "renderer_id": "forward_plus",
    "gpu": "Vulkan 1.x capable GPU and driver",
    "executable_env": "MOVIEPY_GODOT",
    "path_names": ("godot", "godot4"),
    "download_url": "https://godotengine.org/download/windows/",
    "checksum": "Verify the download against SHA512-SUMS.txt of the same release.",
    "python_extras": (),
    "export_templates_required": False,
    "network_required": False,
}

# Matches "4.7.2.stable.official.abc123", "4.7.stable", "4.7.2"; the leading
# major.minor pair is required, the patch number is optional.
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")
_CREATE_NO_WINDOW = 0x08000000


@dataclass(frozen=True)
class GodotCheck:
    """Outcome of ``check_godot``; ``ok`` is True only when no problem exists."""

    ok: bool
    executable: str | None
    version: str | None
    version_tuple: tuple[int, ...] | None
    platform_ok: bool
    version_ok: bool
    problems: list[str]


def _current_platform():
    return sys.platform


def _is_windows():
    return os.name == "nt"


def _creation_flags():
    """Hide the console window of the probed child process on Windows."""
    if not _is_windows():
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW)


def _parse_version(text):
    """Return ``(major, minor, patch)`` from Godot's ``--version`` text, or None."""
    match = _VERSION_RE.match(text.strip())
    if not match:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch or 0))


def _version_in_range(version):
    """True for 4.7 <= version < 5.0 when the requirement table says so."""
    minimum = GODOT_REQUIREMENTS["min_version"]
    upper = GODOT_REQUIREMENTS["max_version_exclusive"]
    return (
        version[0] == GODOT_REQUIREMENTS["max_major"]
        and version[:2] >= minimum
        and version < upper
    )


def _probe_version(path, timeout):
    """Run ``<godot> --version``; return ``(text, error)`` with one of them None."""
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=_creation_flags(),
        )
    except subprocess.TimeoutExpired:
        return None, f"`{path} --version` timed out after {timeout} s"
    except OSError as exc:
        return None, f"cannot run `{path} --version`: {exc}"
    if completed.returncode != 0:
        return None, f"`{path} --version` exited with code {completed.returncode}"
    text = (completed.stdout or "").strip() or (completed.stderr or "").strip()
    if not text:
        return None, f"`{path} --version` printed no version"
    return text, None


def check_godot(executable=None, *, timeout=15):
    """Check the platform, executable and version without raising on bad setups.

    The executable is resolved with ``find_godot`` (explicit argument,
    ``MOVIEPY_GODOT``, then ``godot``/``godot4`` on PATH). The version is only
    probed on a supported platform.
    """
    problems = []
    platforms = GODOT_REQUIREMENTS["platforms"]
    platform = _current_platform()
    platform_ok = platform in platforms
    if not platform_ok:
        problems.append(
            f"Godot rendering requires {', '.join(platforms)}; current platform is {platform}"
        )

    path = None
    try:
        path = find_godot(executable)
    except (OSError, ValueError) as exc:
        problems.append(str(exc))

    version_text = None
    version_tuple = None
    version_ok = False
    if path is not None and platform_ok:
        version_text, error = _probe_version(path, timeout)
        if error:
            problems.append(error)
        elif version_text is not None:
            version_tuple = _parse_version(version_text)
            if version_tuple is None:
                problems.append(f"cannot parse Godot version from: {version_text}")
            else:
                version_ok = _version_in_range(version_tuple)
                if not version_ok:
                    minimum = ".".join(map(str, GODOT_REQUIREMENTS["min_version"]))
                    upper = ".".join(
                        map(str, GODOT_REQUIREMENTS["max_version_exclusive"])
                    )
                    problems.append(
                        f"Godot {minimum} <= version < {upper} is required; "
                        f"found {version_text}"
                    )

    ok = platform_ok and path is not None and version_ok and not problems
    return GodotCheck(
        ok=ok,
        executable=str(path) if path is not None else None,
        version=version_text if version_tuple is not None else None,
        version_tuple=version_tuple,
        platform_ok=platform_ok,
        version_ok=version_ok,
        problems=problems,
    )


def _format_text(result):
    requirements = GODOT_REQUIREMENTS
    minimum = ".".join(map(str, requirements["min_version"]))
    upper = ".".join(map(str, requirements["max_version_exclusive"]))
    lines = [
        "Godot requirements",
        f"  engine:       {requirements['engine']} {requirements['edition']}",
        f"  version:      {minimum} <= version < {upper}",
        f"  platforms:    {', '.join(requirements['platforms'])}",
        f"  renderer:     {requirements['renderer']}",
        f"  gpu:          {requirements['gpu']}",
        f"  executable:   set {requirements['executable_env']} or put "
        f"{' / '.join(requirements['path_names'])} on PATH",
        f"  download:     {requirements['download_url']}",
        f"  checksum:     {requirements['checksum']}",
        "  python extras: none; export templates not needed; network not needed",
        "",
        "Check",
        f"  executable:   {result.executable or 'not found'}",
        f"  version:      {result.version or 'unknown'}",
        f"  platform_ok:  {result.platform_ok}",
        f"  version_ok:   {result.version_ok}",
        f"  result:       {'OK' if result.ok else 'FAILED'}",
    ]
    lines.extend(f"  problem: {problem}" for problem in result.problems)
    return "\n".join(lines)


def main(argv=None):
    """Print requirements and the check result; return 0 when Godot is usable."""
    parser = argparse.ArgumentParser(
        prog="python -m moviepy.ae.three_d.requirements",
        description="Show the Godot requirements and check the installed executable.",
    )
    parser.add_argument(
        "--executable",
        help="Godot executable; defaults to MOVIEPY_GODOT, then godot/godot4 on PATH",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the requirements and check as JSON"
    )
    args = parser.parse_args(argv)
    result = check_godot(args.executable)
    if args.json:
        payload = {"requirements": GODOT_REQUIREMENTS, "check": asdict(result)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_format_text(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
