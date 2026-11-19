# WS-00 acceptance evidence

This records the final frozen implementation, measured acceptance boundaries,
and actual validation completed on 2026-10-03 Asia/Taipei. The full test suite,
AE coverage, Black, canonical CI flake8 and strict changed-scope flake8 passed.
Raw repository-wide flake8 retains eighteen pre-existing E501 violations in
unmodified files; that limitation is recorded below.

## Definition of Done (§9.3)

| Requirement | Status and evidence |
| --- | --- |
| Public numpydoc API and executable examples | Passed for Buffer/RenderContext exports; three actual Python examples in `ws00.rst` executed. |
| Parameters represented by Property and keyframes | N/A for WS-00 infrastructure; Property is WS-01. |
| Effect-specific common DoD | N/A; no effect class was added. Gaussian is a raw cv2 benchmark primitive. |
| Determinism | Passed: core tests and complete repeated benchmark output comparisons. |
| Explicit input validation; no hardcoded secrets | Passed in scoped core tests and independent reviews; metadata paths are relative to the checkout. |
| No debug output or swallowed operational errors | Passed in scoped reviews. Benchmark stdout intentionally emits JSON; provenance unavailability is explicit and bounded by a timeout. |
| Dependency process | No new dependency; existing NumPy/Pillow/OpenCV reuse and notices recorded in `THIRD_PARTY.md`. |
| New tests and coverage | Final frozen source: **318 passed**, **98.70% AE coverage**, exceeding the 80% gate; `.ag-artifacts/runs/ws00-plan-20261002/pytest-ae-coverage-final.log`. |
| Existing full test suite | **921 passed, 5 skipped, 1 xpassed, 1 warning in 186.77 s** on the final frozen source; `pytest-full.log`. The five skips and one xpass are unchanged from baseline. |
| Black | **Passed**, pinned Black 23.7 `--check .`: 158 files unchanged; `black-final.log`. |
| flake8 | **Canonical CI and strict changed scope passed**; raw full-tree check has eighteen unchanged pre-existing E501 violations, described below. |
| 1080p ten-layer single-thread CPU ≥10 fps | **Passed**, full protocol, median **10.923400 fps**, final JSON below. |
| Historical >20% benchmark regression guard | N/A: no historical same-machine baseline. Within-session unchanged Normal workload improved approximately 7.19×. |
| Files ≤800 lines / functions <50 lines | Passed in scoped syntax checks and both core review follow-ups; reviewed Buffer has 716 lines. |
| Independent Python/correctness review | Approved at final Buffer SHA256 below; no unresolved CRITICAL/HIGH findings. |

## Delivered file groups

The concrete delivery contains **36 new files and one allowed existing-file
change**, grouped as follows:

- Public core: `moviepy/ae/__init__.py`, `buffer.py`, `context.py`.
- Test infrastructure: `tests/ae/__init__.py`, `conftest.py`, `_golden.py`.
- Six test modules: `test_buffer.py`, `test_context.py`, `test_compatibility.py`,
  `test_golden.py`, `test_golden_cli.py`, `test_benchmark.py` under `tests/ae/`.
- Synthetic assets: `tests/ae/assets/__init__.py`, `generate_ws00.py`,
  `LICENSE-ASSETS.txt`, and seven generated source PNGs.
- Golden references: three PNGs and their three SHA/provenance JSON sidecars
  under `tests/ae/golden/` (`uint8_rgb_roundtrip`, `half_alpha_over`, `offset_over`).
- Benchmarks: `benchmarks/ae_bench.py`, `README.md`, and the preserved initial
  and final JSON results under `benchmarks/results/`.
- Documentation: `docs/ae/ws00.rst`, `change_notes.rst`, `THIRD_PARTY.md`, and
  this `ws00_acceptance.md` page.
- Allowed existing hook: `tests/conftest.py` registers `--update-golden` without
  changing the existing fixtures.

Run evidence under `.ag-artifacts/runs/ws00-plan-20261002/` is supplementary
validation material, separate from this source delivery inventory.

## Actual RED → GREEN evidence

All filenames below are relative to that run-evidence directory:

| Area | Observed RED | GREEN evidence |
| --- | --- | --- |
| Buffer | Missing new module, one collection error: `buffer-red.log`. | 119 initial owned tests passed: `buffer-green.log`; final expanded scope is included in the 318-test coverage result. |
| RenderContext | Missing `moviepy.ae`, one collection error: `context-red.log`. | 98 owned tests passed: `context-green.log`. |
| Golden | Missing helper, then appended three actual rollback fault-injection failures: `golden-red.log`. | Initial 47 tests and final 50 helper/CLI tests passed: `golden-green.log`. |
| Public exports and option hook | Missing Buffer export: `integration-red.log`; unrecognized `--update-golden`: `root-option-red.log`. | Integrated final 318-test run: `pytest-ae-coverage-final.log`. |
| Benchmark CLI | Five failures because the script was absent: `benchmark-red.log`. | Five initial CLI tests passed: `benchmark-green-final.log`; added timeout test is included in the final aggregate suite. |
| HDR optimization regression | Opposite signed HDR channels wrongly rejected: `performance-hdr-red.log`. | Fixed fallback and reviewed regression tests, included in the final AE and full-suite results. |
| Performance | Preserved full initial check failed at 1.518512 fps: `benchmark-initial.log`. | Final complete check passed at 10.923400 fps: `benchmark-final.log` and final JSON. |

The baseline was an existing-suite/environment observation, separate from
feature RED: `baseline-pytest.log` recorded 601 passed, two PNG-viewer failures,
five skips and one xpass. The final suite resolves the viewer execution boundary
with the real adapter described below.

## Final validation commands and lint boundary

The coordinator ran the following final checks using the validated environment:

```powershell
.tox\ae-ws00\Scripts\python.exe -m pytest tests/ae -q --cov=moviepy/ae --cov-report=term-missing --cov-fail-under=80
.tox\ae-ws00\Scripts\python.exe .ag-artifacts/runs/ws00-plan-20261002/run_tests_with_viewer.py tests -q
.tox\uv-cache\archive-v0\bIOCkZ-uJnOIztBLrgGU9\Scripts\python.exe -m black --check .
.tox\ae-ws00\Scripts\python.exe -m flake8 -v --show-source --ignore=E501 moviepy docs/conf.py examples tests benchmarks
.tox\ae-ws00\Scripts\python.exe -m flake8 moviepy/ae tests/ae benchmarks tests/conftest.py
.tox\ae-ws00\Scripts\python.exe benchmarks/ae_bench.py --check --output benchmarks/results/ws00-validation.json
```

The coverage command targets the `moviepy/ae` directory. This measures the
same AE source files while avoiding a legacy coverage 6.5/NumPy 2 issue in
which dotted-module targeting triggers a duplicate C-extension import.

The canonical CI flake8 invocation reports zero violations
(`flake8-ci-final.log`). Strict changed-scope flake8 is also clean
(`flake8-scope-final.log`). The raw full-tree invocation without CI's E501
ignore reports **18 pre-existing line-length violations**
(`flake8-strict-final.log`): twelve in `moviepy/video/VideoClip.py`, five in
`tests/test_ffmpeg_reader.py`, and one in `tests/test_issues.py`. These match
the original baseline, and the files remain unmodified under §9.4. This is an
explicit raw-lint limitation, not a claim that strict full-tree flake8 passed.

The one final pytest warning is the existing `test_slice_mirror` end-of-file
FFmpeg reader warning; the suite passed with the reader's last-valid-frame
fallback. The original five skips and one xpass were neither newly introduced
nor removed to change the acceptance count.

## Actual benchmark boundaries

`benchmarks/results/ws00-validation.json` is the preserved final full result,
timestamp 2026-10-02 15:55:34 UTC, `--check` exit 0. All cases use 1920×1080,
warmup 5, repeats 5 × frames 20, process-local numerical-library threads one,
OpenCV CPU threads one, OpenCL false, and readonly prebuilt Buffer inputs.

| Case | Median ms/frame | Median fps | Interpretation |
| --- | ---: | ---: | --- |
| single | 15.815345 | 63.229730 | One final RGB export from a prebuilt Buffer. |
| ten-layer Normal | 91.546585 | 10.923400 | Ten actual full-canvas semitransparent over calls, allocation/copy costs, and one final RGB export; acceptance passed. |
| import-and-compose | 1020.306100 | 0.980098 | Eleven source imports plus the same ten over calls and one final export; not a ten-fps claim. |
| raw Gaussian sigma20 | 326.099980 | 3.066544 | Actual float32 RGBA, four channels, raw cv2 primitive; no WS-10 effect implementation. |

All five primary repeats are ≥10 fps. All cases are correct and bitwise
deterministic. The Normal independent float64 oracle maximum error is
8.2445404e-8, with zero final RGB code error. Input fixture generation, imports
for the primary case, proofs, provenance reads and hashing are outside timers;
no frame-result cache or smaller canvas is used.

Initial `benchmarks/results/ws00-initial-20261002.json` remains intact. Its
Normal primary timing used the same workload and measured 1.518512 fps before
tuning. Its Gaussian timing used **RGB, three channels**; final Gaussian uses
**RGBA, four channels**. Those Gaussian numbers cannot establish a regression.

Reference CPU: AMD Ryzen 9 3900X 12-Core Processor; Windows 10.0.19045 x86-64.
Full versions, git dirty state, numerical thread controls, all repeats and
complete output/source hashes are in the JSON. Its Buffer source SHA256 is
`b2248f4a773c0f6214d644a33a6b819b23d589e4af287fb0af4b81f6f7c02ac4`, matching
both independent review follow-ups.

## Scope and deferred integration

The private storage allocator retains at most two released raw float32
allocations, totaling at most 64 MiB. A lease prevents storage reuse until the
original arrays and every escaped view die. Every operation still recomputes
and fully writes its output; public snapshots remain readonly and live results
are not mutated. This is bounded allocation reuse rather than a frame cache.

Root README, root CHANGELOG and Sphinx navigation integration are explicitly
deferred under the approved §9.4 existing-file whitelist. The new
`benchmarks/README.md`, `ws00.rst` and `change_notes.rst` document the feature,
machine and changes; RST pages use `:orphan:`. No Composition, Property,
WS-10 Gaussian effect, cache implementation or color-space conversion is claimed.

The final full suite ran the original tests with an actual Paint viewer adapter
to satisfy Pillow `show()` on this Windows environment. The adapter verifies
the actual rendered PNG and launches `mspaint.exe`, recording its process ID
and post-launch status in `run_tests_with_viewer.jsonl`. It did not edit the
existing tests, replace rendering with mocked success, or modify OS file
associations. PNG validity and real process launch were verified; no native
window visual-inspection proof is claimed. Repository publishing actions are
separate from the rendering and test acceptance recorded here.
