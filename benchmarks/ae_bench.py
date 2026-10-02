"""Measure real WS-00 1080p CPU work and emit a single JSON report.

Run from any working directory with ``python benchmarks/ae_bench.py --check``.
Numeric dependencies are loaded only after process-local thread limits are set.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ("single", "normal_10", "normal_10_end_to_end", "gaussian_sigma20")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)


def _positive(value):
    """Parse a strictly positive timing count."""
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _warmup(value):
    """Parse a nonnegative warmup count."""
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def _arguments(argv):
    """Validate CLI configuration before expensive dependencies load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=_warmup, default=5)
    parser.add_argument("--repeats", type=_positive, default=5)
    parser.add_argument("--frames", type=_positive, default=20)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args(argv)
    args.cases = list(dict.fromkeys(args.cases))
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def _dependencies():
    """Set process-only CPU limits, then load the actual public API."""
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"
    sys.path.insert(0, str(ROOT))
    import cv2
    import numpy as np

    from moviepy.ae import Buffer

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    return np, cv2, Buffer


def _sources(np, width, height, count):
    """Make deterministic spatial RGB patterns and semitransparent masks."""
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    sources = []
    for index in range(count + 1):
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        for channel in range(3):
            rgb[..., channel] = (
                x * (channel + 1) + y * (index + 1) + index * 19 + channel * 47
            ) % 256
        mask = (
            None
            if index == 0
            else np.full((height, width), 0.25 + index * 0.04, dtype=np.float32)
        )
        rgb.setflags(write=False)
        if mask is not None:
            mask.setflags(write=False)
        sources.append((rgb, mask))
    return sources


def _import_sources(Buffer, sources):
    """Create detached readonly Buffer inputs using the public importer."""
    return [Buffer.from_uint8_rgb(rgb, mask) for rgb, mask in sources]


def _normal(buffers):
    """Execute every layer's full-canvas source-over without caching."""
    result = buffers[0]
    for layer in buffers[1:]:
        result = layer.composite_over(result)
    return result


def _normal_rgb(buffers):
    """Composite in float32, then perform exactly one final RGB export."""
    return _normal(buffers).to_uint8_rgb()


def _reference(np, sources):
    """Compute independent source-over in float64 from original RGB/masks."""
    first = sources[0][0]
    result = np.ones((*first.shape[:2], 4), dtype=np.float64)
    result[..., :3] = first.astype(np.float64) / 255.0
    for rgb, mask in sources[1:]:
        alpha = mask.astype(np.float64)[..., None]
        result[..., :3] = rgb.astype(np.float64) / 255.0 * alpha + (
            result[..., :3] * (1.0 - alpha)
        )
        result[..., 3:4] = alpha + result[..., 3:4] * (1.0 - alpha)
    straight = np.zeros((*first.shape[:2], 3), dtype=np.float64)
    np.divide(
        result[..., :3], result[..., 3:4], out=straight, where=result[..., 3:4] != 0
    )
    rgb = np.rint(np.clip(straight, 0, 1) * 255.0).astype(np.uint8)
    return result, rgb


def _hash(array):
    """Hash complete contiguous output bytes outside the timing loop."""
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _normal_proof(np, buffers, sources):
    """Validate full float/output shapes, oracle values, and determinism."""
    expected, expected_rgb = _reference(np, sources)
    actual = _normal(buffers)
    rgb = actual.to_uint8_rgb()
    repeated = _normal(buffers)
    float_error = float(np.max(np.abs(actual.rgba.astype(np.float64) - expected)))
    code_error = int(np.max(np.abs(rgb.astype(np.int16) - expected_rgb)))
    deterministic = np.array_equal(actual.rgba, repeated.rgba) and (
        np.array_equal(rgb, repeated.to_uint8_rgb())
    )
    correct = actual.rgba.shape == expected.shape and rgb.shape == expected_rgb.shape
    correct = correct and actual.rgba.dtype == np.float32 and rgb.dtype == np.uint8
    correct = correct and float_error <= 2e-6 and code_error <= 1 and deterministic
    return {
        "correct": bool(correct),
        "float_max_error": float_error,
        "rgb_max_code_error": code_error,
        "deterministic": bool(deterministic),
        "float_shape": list(actual.rgba.shape),
        "rgb_shape": list(rgb.shape),
        "float_dtype": str(actual.rgba.dtype),
        "rgb_dtype": str(rgb.dtype),
        "output_sha256": _hash(rgb),
        "float_sha256": _hash(actual.rgba),
        "oracle": "independent float64 source-over from source RGB and masks",
    }


def _reflect(np, coordinates, length):
    """Map coordinates with OpenCV BORDER_REFLECT_101's exact convention."""
    period = 2 * (length - 1)
    mapped = coordinates % period
    return np.where(mapped < length, mapped, period - mapped)


def _gaussian_proof(np, cv2, source):
    """Check sigma20 against independent float64 convolution samples."""
    result = cv2.GaussianBlur(source, (0, 0), 20, borderType=cv2.BORDER_REFLECT_101)
    offsets = np.arange(-80, 81)
    kernel = np.exp(-(offsets.astype(np.float64) ** 2) / (2 * 20.0**2))
    kernel /= kernel.sum()
    height, width = source.shape[:2]
    points = (
        (0, 0),
        (0, width - 1),
        (height // 2, width // 2),
        (height - 1, 0),
        (height - 1, width - 1),
    )
    errors = []
    for y, x in points:
        rows = _reflect(np, y + offsets, height)
        columns = _reflect(np, x + offsets, width)
        patch = source[rows[:, None], columns[None, :]].astype(np.float64)
        expected = np.einsum("i,j,ijc->c", kernel, kernel, patch)
        errors.append(float(np.max(np.abs(result[y, x] - expected))))
    deterministic = np.array_equal(
        result, cv2.GaussianBlur(source, (0, 0), 20, borderType=cv2.BORDER_REFLECT_101)
    )
    correct = result.shape == source.shape and result.dtype == np.float32
    correct = correct and max(errors) <= 3e-6 and deterministic
    return {
        "correct": bool(correct),
        "sample_max_error": max(errors),
        "deterministic": bool(deterministic),
        "float_shape": list(result.shape),
        "float_dtype": str(result.dtype),
        "channels": "RGBA",
        "output_sha256": _hash(result),
        "sample_points_yx": list(points),
        "oracle": "independent float64 161-tap separable Gaussian samples",
    }


def _measure(operation, config):
    """Warm up, then report every full repeat and the median frame time."""
    for _ in range(config["warmup"]):
        operation()
    samples = []
    for repeat in range(config["repeats"]):
        started = time.perf_counter()
        for _ in range(config["frames"]):
            operation()
        elapsed = time.perf_counter() - started
        milliseconds = elapsed * 1000 / config["frames"]
        samples.append(
            {
                "repeat": repeat + 1,
                "frames": config["frames"],
                "seconds": elapsed,
                "ms_per_frame": milliseconds,
                "fps": config["frames"] / elapsed,
            }
        )
    median = statistics.median(sample["ms_per_frame"] for sample in samples)
    return {"samples": samples, "median_ms": median, "median_fps": 1000 / median}


def acceptance(config, cases):
    """Check the full WS-00 measurement protocol and Normal performance.

    Parameters
    ----------
    config : dict
        Resolution, warmup, repeat and frame counts, and selected case names.
    cases : dict
        Correctness and measured median fps by case name.

    Returns
    -------
    dict
        Pass status and explicit reasons; diagnostics never pass acceptance.
    """
    reasons = []
    full = config["width"] == 1920 and config["height"] == 1080
    full = full and config["warmup"] == 5 and config["repeats"] == 5
    full = full and config["frames"] == 20 and set(config["cases"]) == set(CASES)
    if not full or set(cases) != set(CASES):
        reasons.append(
            "requires all cases at 1920x1080, warmup=5, repeats=5, frames=20"
        )
    if any(not case["correct"] for case in cases.values()):
        reasons.append("correctness or determinism proof failed")
    normal = cases.get("normal_10")
    if normal is None or normal["median_fps"] < 10.0:
        reasons.append("prebuilt 10-layer Normal median must be >=10 fps")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "normal_minimum_fps": 10.0,
        "regression": "initial baseline; not measured",
    }


def _command(args):
    """Return provenance command output, explicitly marking unavailable data."""
    try:
        return subprocess.check_output(
            args,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except subprocess.TimeoutExpired as error:
        return f"unavailable: provenance command timed out after {error.timeout}s"
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def _machine(np, cv2):
    """Record machine, dependencies, current source state, and source hashes."""
    cpu = platform.processor()
    if os.name == "nt":
        cpu = _command(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
                "(Get-ItemProperty 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\"
                "CentralProcessor\\0').ProcessorNameString",
            ]
        )
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    for name in ("moviepy", "pillow", "imageio"):
        versions[name] = importlib.metadata.version(name)
    paths = (
        Path(__file__),
        ROOT / "moviepy/ae/buffer.py",
        ROOT / "moviepy/ae/context.py",
        ROOT / "moviepy/ae/__init__.py",
    )
    hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    dirty = _command(["git", "status", "--porcelain=v1"])
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "versions": versions,
        "commit": _command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(dirty),
        "git_status_porcelain": dirty,
        "source_sha256": hashes,
    }


def _cases(np, cv2, Buffer, sources, buffers, config):
    """Prepare oracle proofs outside timing and execute requested cases."""
    operations = {
        "single": lambda: buffers[0].to_uint8_rgb(),
        "normal_10": lambda: _normal_rgb(buffers),
        "normal_10_end_to_end": lambda: _normal_rgb(_import_sources(Buffer, sources)),
    }
    boundaries = {
        "single": "prebuilt Buffer; one final RGB export",
        "normal_10": "prebuilt readonly Buffers; 10 full-canvas over calls; "
        "one RGB export",
        "normal_10_end_to_end": "11 source imports; same 10 over calls; one RGB export",
        "gaussian_sigma20": "raw cv2 GaussianBlur float32 RGBA sigma=20; "
        "no WS-10 class",
    }
    proofs = {}
    if "single" in config["cases"]:
        proofs["single"] = _normal_proof(np, buffers[:1], sources[:1])
    if any(case.startswith("normal_10") for case in config["cases"]):
        proof = _normal_proof(np, buffers, sources)
        proofs.update(
            {name: proof for name in config["cases"] if name.startswith("normal_10")}
        )
    if "normal_10_end_to_end" in config["cases"]:
        proofs["normal_10_end_to_end"] = _normal_proof(
            np, _import_sources(Buffer, sources), sources
        )
    if "gaussian_sigma20" in config["cases"]:
        source = buffers[0].rgba
        proofs["gaussian_sigma20"] = _gaussian_proof(np, cv2, source)
        operations["gaussian_sigma20"] = lambda: cv2.GaussianBlur(
            source, (0, 0), 20, borderType=cv2.BORDER_REFLECT_101
        )
    return {
        name: {
            **proofs[name],
            **_measure(operations[name], config),
            "timed_boundary": boundaries[name],
        }
        for name in config["cases"]
    }


def _write_report(report, output):
    """Write intentional CLI JSON and preserve any existing result file."""
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    sys.stdout.write(payload)


def main(argv=None):
    """Execute the CLI, write JSON without overwriting, and return gate status.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments. None uses the process command line.

    Returns
    -------
    int
        One for a failed requested check; zero otherwise. Parser errors exit two.
    """
    args = _arguments(argv)
    np, cv2, Buffer = _dependencies()
    config = {
        "width": 1920,
        "height": 1080,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "frames": args.frames,
        "cases": args.cases,
        "foreground_layers": 10,
        "sigma": 20,
        "fixture_recipe": 1,
    }
    count = 10 if any(name.startswith("normal_10") for name in args.cases) else 0
    sources = _sources(np, config["width"], config["height"], count)
    buffers = _import_sources(Buffer, sources)
    machine = _machine(np, cv2)
    cases = _cases(np, cv2, Buffer, sources, buffers, config)
    report = {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "machine": machine,
        "cases": cases,
        "threads": {
            "environment": {name: os.environ[name] for name in THREAD_VARIABLES},
            "opencv_threads": cv2.getNumThreads(),
            "opencl": cv2.ocl.useOpenCL(),
            "scope": "process-local library CPU threads; no GPU/parallelism",
        },
        "input_buffers_readonly": all(not b.rgba.flags.writeable for b in buffers),
        "acceptance": acceptance(config, cases),
    }
    _write_report(report, args.output)
    return int(args.check and not report["acceptance"]["passed"])


if __name__ == "__main__":
    sys.exit(main())
