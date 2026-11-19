# WS-01 acceptance — 2026-10-03

**Status: all gates passed, including the performance gate, which was resolved
by a controlled-load recheck on 2026-10-03 18:24–18:26 Asia/Taipei.**

WS-01 adds `moviepy.ae.properties` on top of the committed WS-00 foundation
(`2c6288e`). The implementation includes typed values, all four value sources,
five temporal interpolation modes, influence/speed controls, arc-length spatial
curves, roving, separated dimensions, bounded expressions and schema-1 JSON.
The public guide is [ws01.rst](ws01.rst), with changes in
[ws01_change_notes.rst](ws01_change_notes.rst).

## Scope and research

All runtime and test changes are new files. No existing MoviePy runtime, test,
configuration or dependency declaration is changed. The WS-00 Buffer SHA-256
remains `B2248F4A773C0F6214D644A33A6B819B23D589E4AF287FB0AF4B81F6F7C02AC4`.
Public imports are lazy within `moviepy.ae.properties`; no root-package hook is
needed. Root README/CHANGELOG and Sphinx navigation remain deferred under the
existing-file whitelist; new RST pages are marked `:orphan:`.

The planning record and API contracts precede source implementation in
`.ag-artifacts/runs/ws01-20261003/`. Runtime work reuses existing NumPy and the
Python standard library. No dependency or third-party implementation source was
added. Existing package licensing remains recorded in
[THIRD_PARTY.md](THIRD_PARTY.md). The research considered
[simpleeval on PyPI](https://pypi.org/project/simpleeval/) and chose a small,
bounded evaluator using the existing dependency policy. Published references:
[W3C cubic easing](https://www.w3.org/TR/css-easing-1/#cubic-bezier-easing-functions),
[Python AST](https://docs.python.org/3/library/ast.html), and
[Adobe speed/influence](https://helpx.adobe.com/after-effects/desktop/animate-in-after-effects/speed-between-keyframes/speed.html).

## Acceptance evidence

Tests ran with `.tox/ae-ws00/Scripts/python.exe` (Python 3.11.14). Formatting uses
the repository-pinned Black 23.7.0. All new modules parse with Python 3.9 grammar,
have fewer than 800 lines, and have functions shorter than 50 lines.

| Gate | Result |
| --- | --- |
| AE suite | **616 passed**, 30.24 s |
| Entire AE coverage | **98.02%** (1782 / 1818 statements) |
| New WS-01 runtime coverage | **97.84%** (1403 / 1434 statements) |
| Property / keyframe / serialization modules | **100%** each |
| Complete existing and new tests | **1219 passed, 5 skipped, 1 xpassed**, one existing FFmpeg warning; 198.45 s |
| Whole-repository Black | **181 files unchanged**, passed |
| Repository CI flake8, `--ignore=E501` | Passed |
| Strict flake8 on every new runtime/test file | Passed |
| Guide examples / public-package doctests | **7 / 4 passed** |
| Python review | Approved; no unresolved CRITICAL/HIGH/MEDIUM finding |
| Correctness review | Approved; all four HIGH and three MEDIUM initial findings resolved |
| Expression/schema security review | No remaining CRITICAL/HIGH blockers; final path/sampler delta also reviewed by Python reviewer |
| WS-00 benchmark regression | **Passed on controlled-load recheck**: Normal median **13.7737 FPS** (requirement ≥10), −20.69% time vs WS-00; every workload faster than baseline; output hashes byte-identical. See below. |

### Performance result

The complete unchanged WS-00 benchmark protocol ran after all test processes
finished: 1920x1080, one library CPU thread, OpenCL disabled, five warmups and
five batches of twenty frames. The preserved report is
[ws01-regression-20261003.json](../../benchmarks/results/ws01-regression-20261003.json).
All four workloads retained identical output hashes and passed correctness and
determinism checks against their independent oracles.

| Workload | Median ms | FPS | Time change from WS-00 |
| --- | ---: | ---: | ---: |
| Single export | 21.8112 | 45.8480 | +37.91% |
| Normal, 10 prebuilt layers | 112.5289 | 8.8866 | +22.92% |
| Normal including 11 source imports | 1177.9814 | 0.8489 | +15.45% |
| Gaussian sigma20, float32 RGBA | 366.8701 | 2.7258 | +12.50% |

The rendering source and benchmark script are unchanged. Shared machine load is
a possible contributor, but has not been established as the cause. A read-only
environment probe during the repeat run reported 73% CPU load; a subsequent
one-second process sample showed Defender and System activity alongside the
benchmark. No security setting, process priority, user application or benchmark
threshold was changed. The failed run is retained rather than replaced.
The repeat was stopped at the user's 00:55 Asia/Taipei temporary-access cutoff
(the stop-command receipt is timestamped 00:55:02). It did not finish the full protocol
and supplies no acceptance result. Only the two Python processes identified by
this task's exact repeat-benchmark command were stopped.

### Controlled-load recheck (authoritative)

The identical, unchanged protocol was re-run on 2026-10-03 18:24:23–18:26:42
Asia/Taipei with no competing test, render or indexing job on the machine. A
six-second process sample immediately before the run measured 13.6 CPU-seconds
across 24 logical CPUs (about 9% machine utilisation), against the 73% observed
during the failed run. The preserved report is
[ws01-regression-recheck-20261003.json](../../benchmarks/results/ws01-regression-recheck-20261003.json)
and the console record is `.ag-artifacts/runs/ws01-20261003/benchmark-recheck.log`.

| Workload | Failed-run ms | Recheck ms | Recheck FPS | Time change from WS-00 |
| --- | ---: | ---: | ---: | ---: |
| Single export | 21.8112 | 13.4450 | 74.3769 | −14.98% |
| Normal, 10 prebuilt layers | 112.5289 | 72.6023 | 13.7737 | −20.69% |
| Normal including 11 source imports | 1177.9814 | 849.1234 | 1.1777 | −16.78% |
| Gaussian sigma20, float32 RGBA | 366.8701 | 285.2953 | 3.5051 | −12.51% |

The recheck **passes** the WS-00 requirement `normal_10 >= 10 fps` at a median
**13.7737 fps**, and every workload is faster than the WS-00 baseline rather
than slower, so the ±20% regression guard is satisfied with margin. All four
`output_sha256` values are byte-identical to the WS-00 baseline, all correctness
and determinism checks pass, and the frozen source hashes are unchanged
(`moviepy/ae/buffer.py` = `b2248f4a…`, `moviepy/ae/context.py` = `8e3c6700…`,
`benchmarks/ae_bench.py` = `04e0700e…`). Because neither the measured source nor
the benchmark script changed between the failed run and the recheck, the earlier
slowdown is attributable to shared machine load rather than to WS-01 code. The
failed run is preserved as evidence; the recheck is the authoritative
performance result and unblocks the WS-01 commit.

Raw whole-repository flake8 still has the **18 pre-existing E501** findings in
unchanged files recorded during WS-00. They are outside this stage's whitelist;
the table distinguishes the project's actual CI command from strict new-file
lint. No lint configuration or old assertion was relaxed.

The full regression runner uses the existing WS-00 validation adapter that
opens actual rendered PNG files with the installed Paint executable for two
pre-existing image-show tests. It changes neither OS file associations nor test
assertions. Process launch and tests are verified; this is not a claim of manual
visual inspection of the Paint window.

Each implementation owner recorded actual failing tests before implementation,
then passing tests. Root integration likewise first failed on the missing public
exports, then passed. Independent review exposed held-key endpoint derivatives,
typed direct expressions, lost dimension bindings, mixed-zero roving, spatial
overflow, shared automatic velocity and explicit-zero random bounds. Regression
tests now exercise the repaired behavior. Evidence remains in the local run
directory, including `coverage-final.log/json`, `full-pytest.log`,
`black-final.log`, `flake8-ci-final.log`, `flake8-new-final.log`,
`examples-final.log`, `review-python.md`, `review-correctness.md`, and
`security-design.md`.

The spatial acceptance test samples 901 equal-distance fractions of a curved
path and asserts `(max(speed)-min(speed))/mean(speed) < 1%`. The wiggle FFT test
asserts over 90% of energy lies between 2.5 and 3.5 Hz for a 3 Hz request.
Determinism, evaluation reordering, every expression helper family, loop modes,
all typed sources, JSON round trips, real Buffer pixel integration, hostile
objects and bounded resource use are tested. Import/open/dunder exploit strings
all reject with `ExpressionError`; sampling has a shared 32-call budget.

## Behavior and compatibility boundaries

- `set_keyframes` mutates and returns the same Property; `with_*` returns an
  independent Property. Dynamic separated-dimension projections forward context
  and metadata but intentionally cannot serialize.
- Easy Ease uses exact 33.33% influence, corresponding to `.3333/.6667` x
  controls. The specification's rounded `.333/.667` example is compared with
  a documented `2e-4` tolerance, rather than claimed to be exactly identical.
- Automatic temporal tangents use a monotonic shared-velocity approximation.
  Continuous tangents preserve shared explicit/centered speed and may overshoot.
  Expression ease helpers use Hermite endpoint tangents, a separate API from
  CSS-style `Ease` factories.
- Roving rejects mixed zero/nonzero distances, effective hold easing and
  unrepresentable/duplicate resolved times. Entirely stationary runs divide time
  uniformly. Nonfinite derived numerical geometry is rejected explicitly.
- Enum JSON stores primitive string/integer tokens, without restoring Python
  classes. Path interpolation requires matching topology. Path expression data
  is an inert `(vertices, in_tangents, out_tangents, closed)` tuple, validated on
  return. Expression sequences allow 64 items; ordinary paths allow 4096 points.
- `thisLayer` and `thisComp` provide immutable metadata snapshots. Live object
  methods such as `thisComp.layer(...)` are deferred to later scene integration.
  Trusted application-supplied Python callables are supported but cannot be
  loaded from JSON; the expression interpreter does not isolate such callbacks
  in an OS process.
- Wiggle is a seeded continuous narrow-band approximation, not a reproduction
  of unpublished Adobe noise samples. Audio-to-keyframes accepts previously
  analyzed samples only; WS-25 remains responsible for audio analysis.

## Added files

- `moviepy/ae/properties/`: 13 modules covering exports, values, Property,
  keyframes, easing, spatial curves, JSON, expressions and private helpers.
- `tests/ae/test_easing.py`, `test_keyframe.py`, `test_spatial.py`,
  `test_property.py`, `test_property_boundaries.py`,
  `test_property_serialization.py`, `test_expression.py`,
  `test_expression_edges.py`, `test_ws01_integration.py`,
  `test_ws01_review_fixes.py`: **298 added tests** over WS-00's AE count.
- Three new WS-01 documentation files and the benchmark result listed below.

No effects, scene graph, application UI or later workstream was implemented in
this stage. No remote push, merge or release is part of this local delivery.
