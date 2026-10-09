r"""Declare, find and drive the external ``soundx`` (sox-rs) audio tool.

All level and loudness processing of the music templates goes through
``soundx`` (https://github.com/urtiger101-tw/sox-rs). Two generations exist and
both are supported through capability detection:

``0.2.0``
    ``convert`` with ``--normalize --normalize-db=X --stat-json`` (peak
    staging) and ``info --json``. There is no LUFS support, so the integrated
    loudness step is done by FFmpeg ``loudnorm`` (the original S14 pipeline).
``0.3.0``
    Adds ``loudness`` (BS.1770-4 / EBU R128 measurement), ``convert
    --loudness-target`` and ``stream`` (bounded-memory two-pass for long WAV
    files). Detected by ``soundx loudness --help`` exiting 0.

Run ``python -m moviepy.ae.templates.soundx [--json]`` to print the
requirements and check the executable that would be used. The check never
raises for a missing or unsupported soundx; it reports problems.

Negative numbers are always passed as ``--option=-4`` because the command
line parser of soundx would otherwise read ``-4`` as a flag.

Examples
--------
>>> SOUNDX_REQUIREMENTS["executable_env"]
'MOVIEPY_SOUNDX'
>>> parse_version("soundx 0.2.0")
(0, 2, 0)
>>> option("normalize-db", -4.0)
'--normalize-db=-4.0'
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple


__all__ = [
    "SOUNDX_REQUIREMENTS",
    "SoundxError",
    "SoundxNotFoundError",
    "SoundxCheck",
    "find_soundx",
    "check_soundx",
    "require_soundx",
    "parse_version",
    "option",
    "soundx_info",
    "soundx_stage",
    "soundx_loudness",
    "soundx_normalize",
    "main",
]

SOUNDX_REQUIREMENTS = {
    "tool": "soundx (sox-rs)",
    "min_version": (0, 2, 0),
    "loudness_min_version": (0, 3, 0),
    "min_version_reason": "0.2.0: convert --normalize/--stat-json and info --json",
    "loudness_reason": "0.3.0: loudness, convert --loudness-target and stream (LUFS)",
    "executable_env": "MOVIEPY_SOUNDX",
    "path_names": ("soundx", "soundx.exe"),
    "windows_default": r"C:\Program Files\soundx\soundx.exe",
    "source_url": "https://github.com/urtiger101-tw/sox-rs",
    "python_extras": (),
    "network_required": False,
}

_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_CREATE_NO_WINDOW = 0x08000000
_STREAM_BYTES = 1 << 30
_CHECKS = {}


class SoundxError(RuntimeError):
    """A soundx executable is missing, unsuitable or failed."""


class SoundxNotFoundError(SoundxError):
    """No soundx executable could be located."""


@dataclass(frozen=True)
class SoundxCheck:
    """Outcome of ``check_soundx``; ``ok`` is True only when no problem exists."""

    ok: bool
    executable: Optional[str]
    version: Optional[str]
    version_tuple: Optional[Tuple[int, ...]]
    has_loudness: bool
    problems: List[str]


def _creation_flags():
    """Hide the console window of the child process on Windows."""
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW)


def _missing_message(detail):
    env = SOUNDX_REQUIREMENTS["executable_env"]
    url = SOUNDX_REQUIREMENTS["source_url"]
    return (
        f"{detail}. The music templates need soundx for all level and loudness "
        f"processing: install it from {url} and either put it on PATH or set "
        f"the {env} environment variable to the executable."
    )


def parse_version(text):
    """Return ``(major, minor, patch)`` from ``soundx --version`` text, or None.

    Parameters
    ----------
    text : str
        Output such as ``"soundx 0.2.0"``.

    Returns
    -------
    tuple of int or None
        The version, or ``None`` when no ``X.Y[.Z]`` number is present.
    """
    match = _VERSION_RE.search(text or "")
    if not match:
        return None
    major, minor, patch = match.groups()
    return (int(major), int(minor), int(patch or 0))


def option(name, value):
    """Format a ``--name=value`` argument (safe for negative numbers).

    Parameters
    ----------
    name : str
        Option name without leading dashes.
    value : object
        Option value; written with ``str``.

    Returns
    -------
    str
        The single-token argument.
    """
    return f"--{name}={value}"


def find_soundx(executable=None):
    r"""Locate the soundx executable.

    Order: the explicit ``executable`` argument, the ``MOVIEPY_SOUNDX``
    environment variable, ``soundx`` / ``soundx.exe`` on ``PATH`` and, on
    Windows, ``C:\Program Files\soundx\soundx.exe``.

    Parameters
    ----------
    executable : str or Path, optional
        Explicit path; it must exist.

    Returns
    -------
    pathlib.Path
        The executable.

    Raises
    ------
    SoundxNotFoundError
        If nothing usable is found; the message names ``MOVIEPY_SOUNDX`` and
        the source repository.
    """
    env = SOUNDX_REQUIREMENTS["executable_env"]
    if executable:
        path = Path(executable)
        if not path.is_file():
            raise SoundxNotFoundError(
                _missing_message(f"soundx executable not found: {path}")
            )
        return path
    given = os.environ.get(env)
    if given:
        path = Path(given)
        if not path.is_file():
            raise SoundxNotFoundError(
                _missing_message(f"{env} points to a missing file: {path}")
            )
        return path
    for name in SOUNDX_REQUIREMENTS["path_names"]:
        found = shutil.which(name)
        if found:
            return Path(found)
    if os.name == "nt":
        default = Path(SOUNDX_REQUIREMENTS["windows_default"])
        if default.is_file():
            return default
    raise SoundxNotFoundError(_missing_message("soundx was not found"))


def _run(executable, args, *, timeout=None):
    """Run ``soundx <args>`` and return the completed process (never raises on exit)."""
    try:
        return subprocess.run(
            [str(executable), *[str(a) for a in args]],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=_creation_flags(),
        )
    except subprocess.TimeoutExpired:
        raise SoundxError(f"`soundx {args[0]}` timed out after {timeout} s") from None
    except OSError as error:
        raise SoundxError(f"cannot run {executable}: {error}") from None


def _run_checked(executable, args, *, timeout=None):
    """Run soundx and raise ``SoundxError`` with stderr on a non-zero exit."""
    done = _run(executable, args, timeout=timeout)
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "").strip()
        raise SoundxError(
            f"soundx {args[0]} failed (exit code {done.returncode}): {detail}"
        )
    return done


def _json_from(text):
    """Parse the first JSON value in ``text`` (soundx may print extra lines)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char in "[{":
            try:
                return decoder.raw_decode(text[index:])[0]
            except ValueError:
                continue
    raise SoundxError(f"soundx printed no JSON: {text[-500:]}")


def _cache_key(path):
    """Return a key that changes when the executable file changes."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)


def check_soundx(executable=None, *, timeout=15, cache=True):
    """Check the executable, its version and the loudness capability.

    Parameters
    ----------
    executable : str or Path, optional
        Explicit executable (see ``find_soundx``).
    timeout : float
        Seconds allowed for each probe.
    cache : bool
        Reuse the probe of an unchanged executable file (same path, size and
        modification time) instead of starting it again.

    Returns
    -------
    SoundxCheck
        Never raises; missing or unsuitable setups are listed in ``problems``.
    """
    problems = []
    path = None
    try:
        path = find_soundx(executable)
    except SoundxError as error:
        problems.append(str(error))
    version_text, version_tuple, has_loudness = None, None, False
    key = _cache_key(path) if path is not None else None
    if cache and key is not None and key in _CHECKS:
        return _CHECKS[key]
    if path is not None:
        try:
            done = _run(path, ["--version"], timeout=timeout)
        except SoundxError as error:
            problems.append(str(error))
        else:
            text = (done.stdout or "").strip() or (done.stderr or "").strip()
            version_tuple = parse_version(text) if done.returncode == 0 else None
            if version_tuple is None:
                problems.append(f"cannot read the soundx version from `{path}`: {text}")
            else:
                version_text = text
                minimum = SOUNDX_REQUIREMENTS["min_version"]
                if version_tuple < minimum:
                    problems.append(
                        f"soundx >= {'.'.join(map(str, minimum))} is required; "
                        f"found {text}"
                    )
                else:
                    try:
                        probe = _run(path, ["loudness", "--help"], timeout=timeout)
                        has_loudness = probe.returncode == 0
                    except SoundxError:
                        has_loudness = False
    result = SoundxCheck(
        ok=path is not None and version_tuple is not None and not problems,
        executable=str(path) if path is not None else None,
        version=version_text,
        version_tuple=version_tuple,
        has_loudness=has_loudness,
        problems=problems,
    )
    if key is not None and result.ok:
        _CHECKS[key] = result
    return result


def require_soundx(executable=None):
    """Return a passing ``SoundxCheck`` or raise ``SoundxError``.

    Parameters
    ----------
    executable : str or Path, optional
        Explicit executable (see ``find_soundx``).

    Returns
    -------
    SoundxCheck
        The check, with ``ok`` True.

    Raises
    ------
    SoundxError
        With the problems joined; a missing soundx names ``MOVIEPY_SOUNDX``
        and the repository URL.
    """
    check = check_soundx(executable)
    if not check.ok:
        raise SoundxError(" ".join(check.problems) or "soundx is not usable")
    return check


# --------------------------------------------------------------------------- #
# wrappers
# --------------------------------------------------------------------------- #


def _guard_output(target, overwrite, suffixes=(".wav",)):
    """Enforce exclusive create and the WAV suffix; return the target path."""
    target = Path(target)
    if target.suffix.lower() not in suffixes:
        raise ValueError(f"soundx writes {'/'.join(suffixes)} files, got {target}")
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _convert(executable, args, target, existed):
    """Run a soundx command that writes ``target``; clean up after a failure."""
    try:
        done = _run_checked(executable, args)
    except SoundxError:
        if not existed and Path(target).exists():
            Path(target).unlink()
        raise
    return _json_from(done.stdout)


def soundx_info(path, *, executable=None):
    """Return ``soundx info --json`` for one file.

    Parameters
    ----------
    path : str or Path
        Audio file.
    executable : str or Path, optional
        Explicit soundx executable.

    Returns
    -------
    dict
        ``path``, ``spec`` (``sample_rate``, ``channels``), ``frames``,
        ``duration_seconds``, ``peak``, ``rms`` and ``clipped_samples``.
    """
    exe = require_soundx(executable).executable
    data = _json_from(_run_checked(exe, ["info", "--json", str(path)]).stdout)
    if isinstance(data, list):
        if not data:
            raise SoundxError("soundx info printed an empty list")
        data = data[0]
    return data


def soundx_stage(
    source,
    target,
    *,
    bits=24,
    rate=48000,
    channels=2,
    normalize_db=-4.0,
    overwrite=False,
    executable=None,
):
    """Stage a file: convert to 24-bit 48 kHz stereo WAV peak-normalized.

    The S14 recipe ``soundx convert IN OUT --bits 24 --rate 48000 --channels 2
    --normalize --normalize-db=-4 --stat-json``.

    Parameters
    ----------
    source : str or Path
        Input audio.
    target : str or Path
        Output ``.wav``; exclusive create unless ``overwrite``.
    bits : int
        WAV depth.
    rate : int
        Output sample rate.
    channels : int
        Output channel count.
    normalize_db : float
        Peak target in dBFS (negative values are passed as ``--opt=-4.0``).
    overwrite : bool
        Replace an existing ``target`` instead of raising.
    executable : str or Path, optional
        Explicit soundx executable.

    Returns
    -------
    dict
        The ``--stat-json`` object (``frames``, ``peak``, ``rms``,
        ``clipped_samples`` ...).
    """
    exe = require_soundx(executable).executable
    target = _guard_output(target, overwrite)
    existed = target.exists()
    args = ["convert", str(source), str(target)]
    args += [option("bits", int(bits)), option("rate", int(rate))]
    args += [option("channels", int(channels)), "--normalize"]
    args += [option("normalize-db", float(normalize_db)), "--stat-json"]
    stat = _convert(exe, args, target, existed)
    if not isinstance(stat, dict):
        raise SoundxError("soundx convert --stat-json printed no JSON object")
    return stat


def soundx_loudness(path, *, executable=None):
    """Measure integrated loudness with ``soundx loudness --json`` (>= 0.3.0).

    Parameters
    ----------
    path : str or Path
        Audio file.
    executable : str or Path, optional
        Explicit soundx executable.

    Returns
    -------
    dict
        ``integrated_lufs`` (``None`` for silence), ``loudness_range_lu``,
        ``true_peak_dbtp``, ``sample_peak_dbfs`` and friends.

    Raises
    ------
    SoundxError
        If this soundx has no ``loudness`` command (before 0.3.0).
    """
    check = require_soundx(executable)
    if not check.has_loudness:
        raise SoundxError(
            f"soundx {check.version} has no `loudness` command; upgrade to "
            f"soundx>=0.3.0 ({SOUNDX_REQUIREMENTS['source_url']})"
        )
    data = _json_from(
        _run_checked(check.executable, ["loudness", "--json", str(path)]).stdout
    )
    if isinstance(data, list):
        if not data:
            raise SoundxError("soundx loudness printed an empty list")
        data = data[0]
    return data


def soundx_normalize(
    source,
    target,
    *,
    target_lufs=-18.0,
    true_peak=-1.8,
    stream=None,
    bits=24,
    overwrite=False,
    executable=None,
):
    """Normalize integrated loudness with soundx (>= 0.3.0).

    Uses ``convert --loudness-target`` or, for a WAV above about 1 GB (or
    ``stream=True``), the bounded-memory ``stream`` command.

    Parameters
    ----------
    source : str or Path
        Input audio.
    target : str or Path
        Output ``.wav``; exclusive create unless ``overwrite``.
    target_lufs : float
        Integrated loudness target.
    true_peak : float
        True-peak ceiling in dBTP.
    stream : bool, optional
        Force or forbid the streaming command; ``None`` chooses by file size.
    bits : int
        WAV depth for ``convert`` (``stream`` keeps the input depth).
    overwrite : bool
        Replace an existing ``target`` instead of raising.
    executable : str or Path, optional
        Explicit soundx executable.

    Returns
    -------
    dict
        The printed JSON (``loudness`` input/output, ``gain_db``, ``limited``)
        or an empty dict when the command printed none.

    Raises
    ------
    SoundxError
        If this soundx predates 0.3.0 or the command fails.
    """
    check = require_soundx(executable)
    if not check.has_loudness:
        raise SoundxError(
            f"soundx {check.version} cannot normalize loudness; upgrade to "
            f"soundx>=0.3.0 ({SOUNDX_REQUIREMENTS['source_url']})"
        )
    source = Path(source)
    target = _guard_output(target, overwrite)
    existed = target.exists()
    if stream is None:
        stream = (
            source.suffix.lower() in (".wav", ".wave")
            and source.is_file()
            and source.stat().st_size > _STREAM_BYTES
        )
    goal = [option("loudness-target", float(target_lufs))]
    goal.append(option("true-peak", float(true_peak)))
    if stream:
        args = ["stream", str(source), str(target), *goal]
    else:
        args = ["convert", str(source), str(target), option("bits", int(bits))]
        args += [*goal, "--stat-json"]
    stat = _convert(check.executable, args, target, existed)
    return stat if isinstance(stat, dict) else {}


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #


def _format_text(result):
    requirements = SOUNDX_REQUIREMENTS
    minimum = ".".join(map(str, requirements["min_version"]))
    loud = ".".join(map(str, requirements["loudness_min_version"]))
    lines = [
        "soundx requirements",
        f"  tool:         {requirements['tool']}",
        f"  version:      >= {minimum} (staging); >= {loud} for LUFS",
        f"  executable:   set {requirements['executable_env']} or put "
        f"{' / '.join(requirements['path_names'])} on PATH",
        f"  windows:      {requirements['windows_default']}",
        f"  source:       {requirements['source_url']}",
        "",
        "Check",
        f"  executable:   {result.executable or 'not found'}",
        f"  version:      {result.version or 'unknown'}",
        f"  has_loudness: {result.has_loudness}",
        f"  result:       {'OK' if result.ok else 'FAILED'}",
    ]
    lines.extend(f"  problem: {problem}" for problem in result.problems)
    return "\n".join(lines)


def main(argv=None):
    """Print requirements and the check result; return 0 when soundx is usable."""
    parser = argparse.ArgumentParser(
        prog="python -m moviepy.ae.templates.soundx",
        description="Show the soundx requirements and check the installed tool.",
    )
    parser.add_argument(
        "--executable",
        help="soundx executable; defaults to MOVIEPY_SOUNDX, then PATH",
    )
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)
    result = check_soundx(args.executable)
    if args.json:
        payload = {"requirements": SOUNDX_REQUIREMENTS, "check": asdict(result)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_format_text(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
