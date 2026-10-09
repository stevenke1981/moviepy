"""A fake ``soundx`` executable for tests (works on Windows and Linux).

``install_fake_soundx(folder, version="0.2.0")`` writes a Python script and a
launcher (``soundx.cmd`` on Windows, ``soundx`` shell script elsewhere) and
returns the launcher path. Point ``MOVIEPY_SOUNDX`` at it. The fake emulates
``--version``, ``convert`` (``--normalize --normalize-db=X --stat-json`` and,
from 0.3.0, ``--loudness-target``), ``info --json``, ``loudness`` (0.3.0) and
``stream`` (0.3.0). Like clap it rejects a negative number passed as a
separate token. Every call is appended to the file named by the environment
variable ``FAKE_SOUNDX_LOG`` (JSON lines) when it is set.

Its "LUFS" is an ungated, unweighted RMS estimate; only the plumbing is
tested, never the loudness accuracy.
"""

import os
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SCRIPT = r"""
import json
import math
import os
import re
import sys

sys.path.insert(0, ROOT_DIR)

VERSION = "VERSION_TEXT"
LOUD = tuple(int(v) for v in VERSION.split(".")) >= (0, 3, 0)


def fail(message, code=2):
    sys.stderr.write("error: " + message + "\n")
    sys.exit(code)


def log(argv):
    path = os.environ.get("FAKE_SOUNDX_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(argv) + "\n")


def parse(tokens):
    positional, options = [], {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if re.match(r"^-\d", token):
            fail("unexpected argument '" + token + "' found")
        if token.startswith("--"):
            name, eq, value = token[2:].partition("=")
            if name in ("normalize", "stat-json", "json", "help"):
                options[name] = True
            elif eq:
                options[name] = value
            else:
                index += 1
                if index >= len(tokens) or re.match(r"^-\d", tokens[index]):
                    fail("a value is required for '--" + name + "'")
                if name in ("normalize-db", "loudness-target", "true-peak"):
                    fail("negative values need --" + name + "=VALUE")
                options[name] = tokens[index]
        else:
            positional.append(token)
        index += 1
    return positional, options


def load(path, rate=None, channels=None):
    from moviepy.ae.templates._audio_io import audio_info, read_audio

    info = audio_info(path)
    return read_audio(
        path, rate=rate or info["rate"], channels=channels or info["channels"]
    ), (rate or info["rate"])


def stats(data):
    import numpy as np

    peak = float(np.abs(data).max()) if len(data) else 0.0
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2))) if len(data) else 0.0
    return peak, rms, int((np.abs(data) >= 1).sum())


def lufs_of(data):
    import numpy as np

    power = float(np.sum(np.mean(data.astype(np.float64) ** 2, axis=0)))
    return None if power <= 0 else -0.691 + 10 * math.log10(power)


def loudness_of(path, data=None, rate=None):
    if data is None:
        data, rate = load(path)
    peak, _, _ = stats(data)
    peak_db = 20 * math.log10(max(peak, 1e-12))
    return {
        "path": str(path),
        "sample_rate": rate,
        "channels": data.shape[1],
        "frames": len(data),
        "duration_seconds": len(data) / rate,
        "integrated_lufs": lufs_of(data),
        "loudness_range_lu": 0.0,
        "true_peak_dbtp": peak_db,
        "sample_peak_dbfs": peak_db,
        "momentary_max_lufs": lufs_of(data),
        "short_term_max_lufs": lufs_of(data),
        "standard": "FAKE",
    }


def write(path, data, rate, bits):
    from moviepy.ae.templates._audio_io import AudioWriter

    with AudioWriter(path, rate=rate, channels=data.shape[1], overwrite=True) as w:
        w.write(data)


def convert(positional, options, stream=False):
    import numpy as np

    source, target = positional
    rate = int(options["rate"]) if "rate" in options else None
    channels = int(options["channels"]) if "channels" in options else None
    data, rate = load(source, rate, channels)
    extra = {}
    if "normalize" in options:
        db = float(options.get("normalize-db", "-1"))
        peak, _, _ = stats(data)
        if peak > 0:
            data = (data * (10 ** (db / 20) / peak)).astype(np.float32)
    if "loudness-target" in options:
        if not LOUD:
            fail("unexpected argument '--loudness-target' found")
        if "normalize" in options:
            fail("--loudness-target conflicts with --normalize")
        goal = float(options["loudness-target"])
        ceiling = float(options.get("true-peak", "-1.0"))
        before = loudness_of(source, data, rate)
        gain_db = goal - before["integrated_lufs"]
        data = (data * 10 ** (gain_db / 20)).astype(np.float32)
        peak, _, _ = stats(data)
        limited = 20 * math.log10(max(peak, 1e-12)) > ceiling
        if limited:
            data = (data * (10 ** (ceiling / 20) / peak)).astype(np.float32)
        extra["loudness"] = {
            "input": before,
            "output": loudness_of(target, data, rate),
            "gain_db": gain_db,
            "limited": limited,
            "limiter_gain_reduction_max_db": 0.0,
        }
    write(target, data, rate, int(options.get("bits", "16")))
    peak, rms, clipped = stats(data)
    if "stat-json" in options or stream:
        out = {
            "path": str(target),
            "spec": {"sample_rate": rate, "channels": data.shape[1]},
            "frames": len(data),
            "duration_seconds": len(data) / rate,
            "peak": peak,
            "rms": rms,
            "clipped_samples": clipped,
        }
        out.update(extra)
        print(json.dumps(out))


def main(argv):
    log(argv)
    if argv == ["--version"]:
        print("soundx " + VERSION)
        return
    if not argv:
        fail("a subcommand is required")
    command, rest = argv[0], argv[1:]
    positional, options = parse(rest)
    if command == "loudness":
        if not LOUD:
            sys.stderr.write("error: failed to read loudness: not found\n")
            sys.exit(1)
        if "help" in options:
            print("Measure loudness")
            return
        print(json.dumps([loudness_of(p) for p in positional]))
    elif command == "info":
        rows = []
        for path in positional:
            data, rate = load(path)
            peak, rms, clipped = stats(data)
            rows.append(
                {
                    "path": str(path),
                    "spec": {"sample_rate": rate, "channels": data.shape[1]},
                    "frames": len(data),
                    "duration_seconds": len(data) / rate,
                    "peak": peak,
                    "rms": rms,
                    "clipped_samples": clipped,
                }
            )
        print(json.dumps(rows))
    elif command == "convert":
        convert(positional, options)
    elif command == "stream":
        if not LOUD:
            fail("unrecognized subcommand 'stream'")
        convert(positional, options, stream=True)
    else:
        fail("unrecognized subcommand '" + command + "'")


main(sys.argv[1:])
"""


def install_fake_soundx(folder, version="0.2.0"):
    """Write the fake soundx into ``folder`` and return the launcher path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / "fake_soundx.py"
    script.write_text(
        SCRIPT.replace("ROOT_DIR", repr(str(ROOT))).replace("VERSION_TEXT", version),
        encoding="utf-8",
    )
    python = sys.executable
    if os.name == "nt":
        launcher = folder / "soundx.cmd"
        launcher.write_text(f'@"{python}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = folder / "soundx"
        launcher.write_text(f'#!/bin/sh\nexec "{python}" "{script}" "$@"\n')
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return launcher
