"""Measure equivalent Gaussian filtering and complete particle source rendering.

Run from the checkout: python -m benchmarks.ae_motion_bench --output NEW.json
This diagnostic records actual timings, not an AE parity or real-time claim.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from moviepy.ae import ParticleLayer
from moviepy.ae.effects.blur.gaussian_blur import filter_pixels


def measure(operation, repeats):
    """Warm up once and return every sample plus its median in milliseconds."""
    operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def run(threads, repeats):
    """Compare identical float32 kernels without resolution/quality shortcuts."""
    cv2.setNumThreads(threads)
    cv2.ocl.setUseOpenCL(False)
    pixels = np.random.default_rng(605).random((1080, 1920, 4), dtype=np.float32)
    pixels[..., :3] *= pixels[..., 3:]
    kernel = cv2.getGaussianKernel(121, 20, cv2.CV_32F)
    identity = np.ones((1, 1), np.float32)

    def previous():
        first = cv2.sepFilter2D(
            pixels, -1, kernel, identity, borderType=cv2.BORDER_REPLICATE
        )
        return cv2.sepFilter2D(
            first, -1, identity, kernel, borderType=cv2.BORDER_REPLICATE
        )

    def current():
        return filter_pixels(pixels, 20, 20, cv2.BORDER_REPLICATE)

    error = float(np.max(np.abs(previous() - current())))
    if error > 2e-6:
        raise AssertionError(f"equivalent Gaussian kernels disagree: {error}")
    filtering = {
        "previous_two_calls": measure(previous, repeats),
        "combined_call": measure(current, repeats),
        "max_absolute_error": error,
        "scope": "filter only, RGBA float32 1920x1080, sigma 20, replicate border",
    }
    particles = {}
    for count in (5000, 50000):
        layer = ParticleLayer(
            count=count,
            seed=605,
            size=(1920, 1080),
            speed=(80, 600),
            lifetime=2,
            emission_duration=0.5,
        )
        first = layer.source_buffer(1).rgba.copy()
        layer.source_buffer(0.25)
        if not np.array_equal(first, layer.source_buffer(1).rgba):
            raise AssertionError("particle random seek changed the output")
        particles[str(count)] = measure(lambda: layer.source_buffer(1), repeats)
        particles[str(count)]["rgba_sha256"] = hashlib.sha256(
            first.tobytes()
        ).hexdigest()
    return {
        "threads": cv2.getNumThreads(),
        "gaussian": filtering,
        "particles": particles,
        "particle_scope": "complete source_buffer, 1080p, including allocations; excludes composition/encoding",
    }


def main(argv=None):
    """Write a reproducible diagnostic to a new file without replacing evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--threads", nargs="+", type=int, default=[1, cv2.getNumThreads()]
    )
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    if args.repeats < 1 or any(t < 1 for t in args.threads):
        parser.error("repeats and threads must be positive")
    repo = Path(__file__).resolve().parents[1]
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "status": subprocess.check_output(
            ["git", "status", "--short"], cwd=repo, text=True
        ),
        "source_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                repo / "moviepy/ae/effects/blur/gaussian_blur.py",
                repo / "moviepy/ae/particles/layer.py",
            )
        },
        "warmup": 1,
        "repeats": args.repeats,
        "results": [run(t, args.repeats) for t in dict.fromkeys(args.threads)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
