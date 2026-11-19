"""Measure WS-03..06 compositing work at 1080p and emit one JSON report.

Run from the checkout::

    python benchmarks/ae_comp_bench.py --output benchmarks/results/NEW.json

Cases cover every blend mode against Normal, a 20-layer composition, a
feathered mask, an adjustment-layer Gaussian Blur and single and stacked Echo.
Targets from docs/ae/AE_REVIEW_PLAN.md section 3 are recorded as pass/fail but
never change the exit status: this is a measurement, not a gate.

Timing reuses the WS-00 protocol in ``ae_bench.py``: process-local single CPU
thread for NumPy/OpenCV, OpenCL off, warmup calls, then several repeats of N
frames, reporting the median repeat.
"""

import argparse
import hashlib
import importlib.util
import itertools
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMP_CASES = (
    "comp_backdrop_only",
    "comp_20_layers",
    "feathered_mask",
    "adjustment_gaussian_blur_40",
    "echo_single",
    "echo_stacked",
)
CASES = ("blend_modes",) + COMP_CASES
# Targets quoted from docs/ae/AE_REVIEW_PLAN.md section 3 (P1, P2, P3).
TARGETS = {
    "separable_ratio_vs_normal": 2.0,
    "comp_20_layers": 150.0,
    "adjustment_gaussian_blur_40": 150.0,
}
REFERENCE_WIDTH = 1920


def _load_ae_bench():
    """Import the WS-00 bench for its timing and provenance helpers."""
    spec = importlib.util.spec_from_file_location("ae_bench", HERE / "ae_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_ae_bench()


def _size(value):
    """Parse ``WIDTHxHEIGHT`` with both sides at least 16 pixels."""
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError("size must look like 1920x1080") from None
    if width < 16 or height < 16:
        raise argparse.ArgumentTypeError("size must be at least 16x16")
    return width, height


def _arguments(argv):
    """Validate the CLI before NumPy, OpenCV or MoviePy load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=_size, default=(1920, 1080))
    parser.add_argument("--warmup", type=BASE._warmup, default=2)
    parser.add_argument("--repeats", type=BASE._positive, default=5)
    parser.add_argument("--frames", type=BASE._positive, default=5)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument(
        "--modes", nargs="+", help="blend mode tokens (default: every mode)"
    )
    args = parser.parse_args(argv)
    args.cases = list(dict.fromkeys(args.cases))
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    return args


def _dependencies():
    """Apply the WS-00 thread limits, then import the compositing API."""
    np, cv2, Buffer = BASE._dependencies()
    import moviepy.ae as ae
    from moviepy.ae.blend import BlendMode, blend
    from moviepy.ae.blend.modes import _SPECS

    return {
        "np": np,
        "cv2": cv2,
        "Buffer": Buffer,
        "ae": ae,
        "BlendMode": BlendMode,
        "blend": blend,
        "kinds": {mode.value: spec[1] for mode, spec in _SPECS.items()},
    }


def _hash(array):
    """Hash the full contiguous output outside timing."""
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _textured(np, width, height, seed):
    """Deterministic uint8 RGB pattern; no constant-color shortcuts apply."""
    x = np.arange(width, dtype=np.uint32)[None, :]
    y = np.arange(height, dtype=np.uint32)[:, None]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[..., channel] = (x * (channel + 2) + y * (seed + 1) + 31 * seed) % 256
    return rgb


def _blend_inputs(deps, width, height):
    """Opaque textured backdrop and a source with spatially varying alpha."""
    np, Buffer = deps["np"], deps["Buffer"]
    ramp = np.linspace(0.2, 0.9, width, dtype=np.float32)[None, :]
    mask = np.ascontiguousarray(np.broadcast_to(ramp, (height, width)))
    base = Buffer.from_uint8_rgb(_textured(np, width, height, 1))
    layer = Buffer.from_uint8_rgb(_textured(np, width, height, 2), mask)
    return base, layer


def _blend_case(deps, config):
    """Time ``blend(base, layer, mode)`` for each mode at full canvas."""
    np, blend = deps["np"], deps["blend"]
    base, layer = _blend_inputs(deps, config["width"], config["height"])
    modes = config["modes"] or [mode.value for mode in deps["BlendMode"]]
    if "normal" not in modes:
        modes = ["normal"] + modes
    results = {}
    for mode in modes:
        first = blend(base, layer, mode).rgba
        again = blend(base, layer, mode).rgba
        measured = BASE._measure(lambda: blend(base, layer, mode), config)
        results[mode] = {
            "kind": deps["kinds"][mode],
            "deterministic": bool(np.array_equal(first, again)),
            "finite": bool(np.isfinite(first).all()),
            "output_sha256": _hash(first),
            **measured,
        }
    _blend_targets(results)
    return {
        "timed_boundary": "blend(base, layer, mode) on prebuilt full-canvas "
        "Buffers: opaque textured backdrop, textured source with alpha "
        "ramp 0.2..0.9 across x",
        "modes": results,
    }


def _blend_targets(results):
    """Annotate each mode with its ratio to Normal and the P1 target."""
    normal = results["normal"]["median_ms"]
    limit = TARGETS["separable_ratio_vs_normal"]
    for result in results.values():
        ratio = result["median_ms"] / normal
        result["ratio_vs_normal"] = ratio
        if result["kind"] == "separable":
            result["target"] = f"<= {limit}x normal"
            result["meets_target"] = ratio <= limit
        else:
            result["target"] = None
            result["meets_target"] = None


def _scale(config, value):
    """Scale a 1080p-defined pixel quantity to the requested canvas."""
    return max(1.0, value * config["width"] / REFERENCE_WIDTH)


def _layer_size(config, width, height):
    """Scale a 1080p layer size, keeping at least one pixel per side."""
    return (int(round(_scale(config, width))), int(round(_scale(config, height))))


def _new_comp(deps, config):
    """A 30 fps, 10 s composition with an opaque textured backdrop."""
    ae, np = deps["ae"], deps["np"]
    from moviepy import ImageClip

    size = (config["width"], config["height"])
    comp = ae.Composition(size=size, fps=30, duration=10)
    texture = _textured(np, size[0], size[1], 3)
    background = comp.add_clip(ImageClip(texture).with_duration(10), name="bg")
    background.transform = ae.Transform(position=(0.0, 0.0), anchor_point=(0, 0))
    return comp


def _moving_solid(deps, comp, config, index, size):
    """Add a solid whose position animates so every frame differs."""
    ae = deps["ae"]
    width, height = config["width"], config["height"]
    layer = comp.add_solid(
        f"solid{index}", color=(40 + index * 9, 120, 220 - index * 7), size=size
    )
    start = ((index * 0.173) % 0.8 * width, (index * 0.291) % 0.8 * height)
    end = (width - start[0], height - start[1])
    layer.transform.position = ae.Property(start, keyframes=[(0, start), (10, end)])
    layer.transform.opacity = 60.0
    return layer


def _scene_20_layers(deps, config):
    """Twenty 600x400 semitransparent moving solids over the backdrop."""
    comp = _new_comp(deps, config)
    size = _layer_size(config, 600, 400)
    for index in range(20):
        _moving_solid(deps, comp, config, index, size)
    return comp


def _scene_feather(deps, config):
    """One full-frame layer with an elliptical mask whose feather animates."""
    ae = deps["ae"]
    comp = _new_comp(deps, config)
    width, height = config["width"], config["height"]
    layer = _moving_solid(deps, comp, config, 0, (width, height))
    low, high = _scale(config, 40.0), _scale(config, 80.0)
    feather = ae.Property((low, low), keyframes=[(0, (low, low)), (10, (high, high))])
    layer.masks = [
        ae.Mask.ellipse(
            (width / 2, height / 2), (width * 0.6, height * 0.6), feather=feather
        )
    ]
    return comp


def _scene_blur(deps, config):
    """Backdrop plus one moving solid under a full-frame GaussianBlur(40)."""
    comp = _new_comp(deps, config)
    _moving_solid(deps, comp, config, 0, _layer_size(config, 600, 400))
    adjustment = comp.add_adjustment("blur")
    adjustment.effects = [deps["ae"].fx.GaussianBlur(blurriness=_scale(config, 40))]
    return comp


def _scene_echo(deps, config, stacked):
    """Moving solid under one or two Echo(number_of_echoes=6) adjustments."""
    comp = _new_comp(deps, config)
    _moving_solid(deps, comp, config, 0, _layer_size(config, 600, 400))
    for index in range(2 if stacked else 1):
        comp.add_adjustment(f"echo{index}").effects = [
            deps["ae"].fx.Echo(echo_time=-0.05, number_of_echoes=6)
        ]
    return comp


SCENES = {
    "comp_backdrop_only": (
        _new_comp,
        "only the 1080p textured ImageClip backdrop shared by every comp case",
    ),
    "comp_20_layers": (_scene_20_layers, "20 moving 600x400 solids, opacity 60"),
    "feathered_mask": (
        _scene_feather,
        "full-frame moving solid, ellipse mask, feather animating 40..80 px",
    ),
    "adjustment_gaussian_blur_40": (
        _scene_blur,
        "one moving 600x400 solid under a full-frame GaussianBlur(40) adjustment",
    ),
    "echo_single": (
        lambda deps, config: _scene_echo(deps, config, False),
        "one adjustment Echo(echo_time=-0.05, number_of_echoes=6)",
    ),
    "echo_stacked": (
        lambda deps, config: _scene_echo(deps, config, True),
        "two stacked adjustment Echo(echo_time=-0.05, number_of_echoes=6)",
    ),
}


def _comp_case(deps, config, name):
    """Time full ``Composition.get_frame`` calls at advancing times."""
    np = deps["np"]
    build, scene = SCENES[name]
    comp = build(deps, config)
    first, again = comp.get_frame(1.0), build(deps, config).get_frame(1.0)
    frames = itertools.count()
    measured = BASE._measure(lambda: comp.get_frame(next(frames) % 300 / 30), config)
    target = TARGETS.get(name)
    return {
        "scene": scene,
        "timed_boundary": "Composition.get_frame(t) including uint8 export; "
        "t advances one frame per call so no frame is reused",
        "layers": len(comp.layers),
        "deterministic": bool(np.array_equal(first, again)),
        "frame_shape": list(first.shape),
        "output_sha256": _hash(first),
        "target_ms": target,
        "meets_target": None if target is None else measured["median_ms"] < target,
        **measured,
    }


def _summary(cases):
    """Collect every recorded target outcome into one flat list."""
    rows = []
    for mode, result in cases.get("blend_modes", {}).get("modes", {}).items():
        if result["meets_target"] is not None:
            rows.append((f"blend:{mode}", result["meets_target"]))
    for name in COMP_CASES:
        if name in cases and cases[name]["meets_target"] is not None:
            rows.append((name, cases[name]["meets_target"]))
    failed = [name for name, met in rows if not met]
    return {
        "gated": False,
        "targets_checked": len(rows),
        "targets_met": len(rows) - len(failed),
        "targets_missed": failed,
        "source": "docs/ae/AE_REVIEW_PLAN.md section 3 (P1, P2, P3)",
    }


def _version(name):
    """Installed distribution version, or an explicit unavailable marker."""
    try:
        return BASE.importlib.metadata.version(name)
    except BASE.importlib.metadata.PackageNotFoundError:
        return "unavailable: not installed (imported from checkout)"


def _cpu():
    """CPU model name; on Windows read from the registry like ae_bench.py."""
    if BASE.os.name != "nt":
        return BASE.platform.processor()
    return BASE._command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "(Get-ItemProperty 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\"
                "CentralProcessor\\0').ProcessorNameString"
            ),
        ]
    )


def _machine(deps):
    """WS-00-style provenance plus hashes of the measured WS-03..06 code.

    MoviePy is usually imported from the checkout rather than installed, so
    distribution versions are looked up individually and may be unavailable.
    """
    np, cv2 = deps["np"], deps["cv2"]
    dirty = BASE._command(["git", "status", "--porcelain=v1"])
    machine = {
        "cpu": _cpu(),
        "logical_cpus": BASE.os.cpu_count(),
        "os": BASE.platform.platform(),
        "architecture": BASE.platform.machine(),
        "versions": {
            "python": BASE.platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            **{name: _version(name) for name in ("moviepy", "pillow", "imageio")},
        },
        "commit": BASE._command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(dirty),
        "git_status_porcelain": dirty,
        "source_sha256": {},
    }
    paths = [Path(__file__), HERE / "ae_bench.py"] + [
        ROOT / "moviepy/ae" / part
        for part in (
            "renderer.py",
            "composition.py",
            "_accumulator.py",
            "blend/modes.py",
            "masks/mask.py",
            "effects/blur/gaussian_blur.py",
            "effects/time/echo.py",
        )
    ]
    for path in paths:
        key = path.relative_to(ROOT).as_posix()
        machine["source_sha256"][key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return machine


def main(argv=None):
    """Run the selected cases and write JSON; returns 0 whatever the targets."""
    args = _arguments(argv)
    deps = _dependencies()
    config = {
        "width": args.size[0],
        "height": args.size[1],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "frames": args.frames,
        "cases": args.cases,
        "modes": args.modes,
    }
    cases = {}
    for name in args.cases:
        if name == "blend_modes":
            cases[name] = _blend_case(deps, config)
        else:
            cases[name] = _comp_case(deps, config, name)
    report = {
        "schema": 1,
        "benchmark": "ws03-06-compositing",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "machine": _machine(deps),
        "threads": {
            "opencv_threads": deps["cv2"].getNumThreads(),
            "opencl": deps["cv2"].ocl.useOpenCL(),
            "environment": {
                name: BASE.os.environ[name] for name in BASE.THREAD_VARIABLES
            },
        },
        "cases": cases,
        "summary": _summary(cases),
    }
    BASE._write_report(report, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
