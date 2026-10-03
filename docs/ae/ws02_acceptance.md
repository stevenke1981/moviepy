# WS-02 acceptance — independent verification, 2026-10-03

WS-02 delivers affine transforms, arc-path auto-orientation, AV/Solid/Null
layers, parenting, timing and render switches. Qwen's original delivery is
commit `16c4fe3`, following WS-01 `7aa9fd6` and WS-00 `2c6288e`.
The independent publication review found rendering defects that the original
tests missed. This document describes the repaired implementation and current
evidence, replacing obsolete test counts and the earlier 100% coverage claim.

Public modules are `moviepy.ae.transform`, `moviepy.ae.warp` and
`moviepy.ae.layers`. The guide is [ws02.rst](ws02.rst), and changes are recorded
in [ws02_change_notes.rst](ws02_change_notes.rst).

## Current validation

| Gate | Actual result |
| --- | --- |
| Complete existing and new tests | **1411 passed, 5 skipped, 1 xpassed**, one existing FFmpeg end-of-file warning; 150.88 s |
| AE suite | **808 passed**, 21.76 s |
| Entire AE coverage | **98.40%**, 2641 / 2684 statements |
| WS-02 runtime coverage | **99.19%**, 859 / 866 statements; requirement ≥80% |
| Added review regressions | **55 passed**; combined WS-02 scope **192 passed** |
| WS-02 module doctests | **11 passed** |
| Guide examples | **8 passed** in the shared example namespace |
| Black 23.7.0 | **191 tracked/new Python files unchanged** |
| Repository CI flake8, `--ignore=E501` | Passed |
| Strict flake8, all AE source/tests and benchmarks | Passed |
| Independent Python, correctness and security reviews | **APPROVE**, no unresolved findings in the reviewed scope |
| Fresh complete benchmark | Passed: Normal **12.43 fps**, all four cases within the ≤20% regression gate |

The test interpreter is `.tox/ae-ws00/Scripts/python.exe` (Python 3.11.14,
NumPy 2.4.6, OpenCV 5.0.0). The two existing Windows image-show tests use the
already-validated local Paint adapter: it opens the actual rendered PNG and
checks the process launch. No old test assertion, fixture, skip or OS file
association was changed. A window launch is not manual visual inspection.

The raw checkout-wide `black --check .` also finds an untracked Qwen diagnostic
helper under `.ag-artifacts/` that needs formatting. The publication check
instead covers every tracked Python file plus the new regression module, as
Git/pre-commit would. That local helper is preserved and is not published.
The 18 pre-existing E501 findings in unchanged legacy files remain outside this
workstream; CI and strict AE checks are reported separately.

Root evidence is retained under
`.ag-artifacts/runs/fork-push-review-20261003/`: `regressions-red.log`,
`numeric-boundaries-red.log`, `numeric-boundaries-green.log`,
`full-pytest-final.log`, `coverage-final.log/json`, `black-final.log`,
`flake8-ci-final.log`, `flake8-ae-final.log`, `doctests-final.log`,
`examples-final.log` and the three independent review reports.

## Fresh performance evidence

The complete run is published in
[`fork-review-20261003.json`](../../benchmarks/results/fork-review-20261003.json).
It uses the unchanged WS-00 protocol: 1920×1080, five warmups, five repeats,
20 measured frames per repeat, one CPU thread and OpenCL disabled. The baseline
is [`ws00-validation.json`](../../benchmarks/results/ws00-validation.json).

| Workload | WS-00 ms/frame | Current ms/frame | Current fps | Time change |
| --- | ---: | ---: | ---: | ---: |
| Single Buffer export | 15.815 | 15.512 | 64.464 | −1.915% |
| Ten Normal foreground layers, prebuilt Buffers | 91.547 | 80.438 | 12.432 | −12.135% |
| Ten Normal foreground layers, including source imports | 1020.306 | 983.454 | 1.017 | −3.612% |
| Raw RGBA GaussianBlur, sigma 20 | 326.100 | 299.192 | 3.342 | −8.251% |

Every case remains correct and deterministic, with the same output SHA-256 as
the baseline. The benchmark's `--check` gate verifies Normal ≥10 fps; comparing
the published case medians separately confirms the ≤20% regression gate.
The 12.43 fps figure measures prebuilt Buffers, not the source-import workload
or a future full Composition/effect renderer. Timing is specific to this machine
and run. Earlier WS-01 failed performance evidence is retained alongside the
subsequent recheck and this current verification.

## Review repairs and behavioral evidence

The first new regression run had **43 failures and 7 passes** before source
edits. Five further numerical regressions also failed before their fixes.
The final suite includes all 55 cases; no existing assertion was weakened.

- **Source offset:** world-to-local conversion now includes the source Buffer
  offset before calling OpenCV. Rotations across all filters and explicit
  world-space crops reproduce the same pixels as an origin-based source.
- **Collapsed transforms:** one or both zero-scale axes return an empty Buffer,
  including rotated and explicit-ROI cases. Singular matrices never enter
  OpenCV's inverse path that otherwise fills the ROI with the first pixel.
- **Reverse footage:** a finite AV source uses
  `duration + (t-start_time)/(stretch/100)` under negative stretch. Footage,
  masks, opacity and transform properties share that clock. The first frame is
  the final readable frame, including durations not divisible by `1/fps`.
  Tests cover negative starts, −100/−200 stretch, explicit/default windows and
  subsequent stretch edits. Reverse playback without a finite duration rejects.
- **Finite cubic results:** potential overflow is checked before opacity and
  alpha clipping. Signed/HDR data near the float32 limit raises clearly when
  OpenCV produces a nonfinite intermediate, even at opacity 0 or 0.1.
- **Held curved direction:** auto-orient samples the preceding segment's final
  tangent across holds/after motion, or the first initial tangent before motion.
  It does not substitute the curve's midpoint direction.
- **Transactional setters:** invalid anchor/position assignments preserve the
  prior flags, serialization and rendering rather than leaving missing fields.
- **Input and allocation boundaries:** matrices require an exact affine bottom
  row `(0,0,1)`. Every allocating translation observes the pixel budget; final
  bounds are range checked after snapping. Eight coordinate ulps preserve real
  half-pixel support at world positions ±1e9. Numerically unsupported inverse
  ranges reject before OpenCV/allocation rather than producing bad pixels.

Independent probes also confirmed matrix order `T(position) · R · S · T(-anchor)`,
hand-computed parent/child matrices, no inherited parent opacity, negative-scale
mirrors, exact bilinear half-pixel blends and deterministic repeated rendering.
Hostile JSON subclasses reject before protocol calls; the unchanged WS-01
expression sandbox continues to reject imports, files, dunders and resource
exhaustion expressions. Trusted application-supplied Python callbacks are
ordinary Python code, not isolated OS processes.

## Coordinate convention and scope

Pixels are centered at integer world coordinates, and the automatic anchor is
`((W-1)/2,(H-1)/2)` plus Buffer offset. Nearest/linear right-angle rotations
match `numpy.rot90` when dimensions have matching parity. For mixed-parity
width/height, a 90-degree rotation about the fixed world center falls on a
half-pixel grid and resamples; its bounds may grow. The implementation keeps
the fixed center rather than claiming unconditional bit-exact rotation.
Cubic uses numerical tolerance for denormal filter residue.

The implementation retains finite signed/HDR RGB overshoot, clips cubic alpha
to [0,1], and does not promise mass-conserving interpolation. Default AV out
points are recalculated from current start/stretch/duration; explicit ones stay
fixed. This stage stores collapse/continuous-rasterization/motion-blur flags;
their behavior belongs to WS-03/20/07. 3D fields are reserved and `three_d=True`
raises for WS-22. Blend formulas, masks, effect stacks, Composition/Renderer and
the VideoClip convenience hook remain subsequent workstreams.

No existing MoviePy runtime, dependency declaration, old test or WS-00 rendering
code was changed. No dependency was added. Existing reuse/licensing is listed
in [THIRD_PARTY.md](THIRD_PARTY.md). Every AE runtime module remains ≤800 lines,
every function <50 lines, and Python 3.9 grammar is checked; the actual execution
environment for this validation is Python 3.11.
