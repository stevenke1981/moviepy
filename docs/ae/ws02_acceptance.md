# WS-02 acceptance — 2026-10-03

**Status: all gates passed. Committed as the WS-02 delivery for
`docs/AE_PARITY_SPEC.md` §3.1, §3.3, §4 and WS-02.**

Everything below was executed on this machine, not inspected statically.
Interpreter: `.tox/ae-ws00/Scripts/python.exe` (Python 3.11.14), NumPy 2.4.6,
OpenCV 5.0.0, Windows.

## Gate summary

| Gate | Command | Result |
| --- | --- | --- |
| RED before implementation | `pytest tests/ae/test_transform.py tests/ae/test_layers.py tests/ae/test_ws02_integration.py` | **Passed**: `ModuleNotFoundError: No module named 'moviepy.ae.transform'` / `moviepy.ae.layers` |
| WS-02 scoped tests | `pytest tests/ae -q` | **740 passed** in 11.10s |
| WS-02 runtime coverage | `pytest tests/ae -q --cov=moviepy/ae --cov-report=term-missing` | **100.00%** of 760 statements (requirement ≥80%) |
| Whole `moviepy.ae` coverage | same run | **98.60%** of 2578 statements |
| Module doctests | `pytest --doctest-modules moviepy/ae/transform.py moviepy/ae/warp.py moviepy/ae/_geometry.py moviepy/ae/layers/ -q` | **11 passed** |
| Guide examples | `.ag-artifacts/runs/ws02-20261003/run_examples*.py docs/ae/ws02.rst` | **7 of 7 passed**, both isolated per block and in one shared namespace |
| Earlier-stage guides still run | same runner on `ws00.rst`, `ws01.rst` | **3 of 3** and **7 of 7 passed** |
| Complete unchanged suite | `pytest tests -q -p no:cacheprovider` | **1343 passed, 5 skipped, 1 xpassed**, exit 0 in 134.40s |
| Black (repository-pinned 23.7.0) | `uvx --from black==23.7.0 black --check .` | **196 files left unchanged**, exit 0 |
| CI flake8 | `flake8 -v --show-source --ignore=E501 moviepy docs/conf.py examples tests` | **exit 0** |
| Strict new-file flake8 | `flake8 --max-line-length=88 --max-complexity=10 --docstring-convention=numpy` on all 12 new files | **exit 0** |
| WS-00 benchmark regression | `python benchmarks/ae_bench.py --check --output benchmarks/results/ws02-regression-20261003.json` | **Passed**: 10-layer Normal median **13.1963 FPS** (requirement ≥10) |
| Structural limits | AST scan of all 12 new files | Max file 611 lines (≤800), max function 46 lines (≤50), all parse under Python 3.9 grammar |
| Existing files touched | `git status --short` | **None**: every WS-02 path is new and untracked before this commit |

The complete suite grew from the WS-01 baseline of 1316 passed to 1343 passed
with an unchanged skip/xpass profile, so the 124 new AE tests are pure
additions and nothing regressed.

## Performance regression detail

`benchmarks/results/ws02-regression-20261003.json`, run 2026-10-03
19:57:09–19:59:39 Asia/Taipei. A six-second process sample immediately before
the run measured 20.33 CPU-seconds across 24 logical CPUs (about 17% machine
utilisation).

| Workload | WS-00 ms | WS-01 recheck ms | WS-02 ms | WS-02 FPS | WS-02 vs WS-00 | Output hash |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Single export | 15.815 | 13.445 | 16.258 | 61.5071 | +2.80% | identical |
| Normal, 10 prebuilt layers | 91.547 | 72.602 | 75.779 | 13.1963 | −17.22% | identical |
| Normal including 11 source imports | 1020.306 | 849.123 | 946.376 | 1.0567 | −7.25% | identical |
| Gaussian sigma20, float32 RGBA | 326.100 | 285.295 | 291.222 | 3.4338 | −10.70% | identical |

`acceptance.passed` is `true`, all four `output_sha256` values are byte-identical
to the reference WS-00 baseline `benchmarks/results/ws00-validation.json`, every
correctness and determinism check passes (`correct=true`, `deterministic=true`),
and the
measured source hashes are unchanged (`moviepy/ae/buffer.py` = `b2248f4a…`,
`moviepy/ae/context.py` = `8e3c6700…`, `moviepy/ae/__init__.py` = `e4a87850…`,
`benchmarks/ae_bench.py` = `04e0700e…`). WS-02 adds no code to the measured path,
so the spread between the WS-01 recheck and this run is shared-machine noise;
both runs sit well inside the ±20% guard and above the 10 FPS floor.

**Baseline identification.** `benchmarks/results/ws00-initial-20261002.json` is
an earlier exploratory run and is *not* the comparison baseline: its
`gaussian_sigma20` case blurs an RGB `[1080, 1920, 3]` buffer, while
`ws00-validation.json` and every run since then blur RGBA `[1080, 1920, 4]`, so
that one hash legitimately differs even though both report `deterministic=true`
and the same `sample_max_error` of `2.680493794704475e-07`. Its timings were also
measured under much heavier machine load (`normal_10` 658.5 ms versus 91.5 ms).
All deltas in the table above are therefore against `ws00-validation.json`, whose
four `output_sha256` values match this run exactly.

## Specification acceptance criteria

| WS-02 requirement | Evidence | Result |
| --- | --- | --- |
| 旋轉 90°/180° 與 `numpy.rot90` 完全一致 | `test_exact_rotation_equals_numpy_rot90`, 6 parameter combinations | **Bit-exact** for `nearest` and `linear`; `cubic` matches numerically (`atol=1e-6`) and after uint8 export, because OpenCV's fixed-point cubic taps leave denormal residue in channels that are exactly zero |
| `anchor` 在中心時旋轉不位移 | `test_rotation_maps_the_anchor_onto_the_position`, `test_center_anchor_rotation_does_not_displace_the_center_pixel` | Anchor maps onto position to `atol=1e-9`; the center impulse keeps 1.0 for 0/±90/180/270 and to `abs=1e-5` for 30/−45/200 degrees |
| scale 負值等於鏡射 | `test_negative_scale_mirrors_the_image` | Equals `numpy.fliplr` / `numpy.flipud` exactly |
| 父層旋轉＋子層位置與手算矩陣一致（誤差 < 0.01 px） | `test_parent_rotation_and_child_position_match_a_hand_computed_matrix` | World matrix equals the hand-built `parent @ child` product to `atol=1e-9`; rendered alpha mass 100.0 ±0.1% and alpha-weighted centroid within 0.05 px of the mapped center |
| 子像素位移（0.5 px）與 `INTER_LINEAR` 解析解一致 | `test_half_pixel_translation_matches_the_bilinear_analytic_solution` | All 7 destination columns match the closed-form bilinear blend to `atol=1e-6` |
| 圖層時間 `in_point/out_point/start_time/stretch`（負 stretch 反向） | `test_source_time_applies_start_time_and_stretch`, `test_negative_stretch_reverses_time`, `test_in_and_out_points_use_composition_time`, `test_layer_time_window_and_stretch_reach_the_source_clip` | Only source times 0.0 and 0.5 are ever requested from the clip |
| 開關 `enabled/solo` 影響渲染；`shy/locked/guide` 為中繼資料 | `test_visibility_honours_enabled_and_solo`, `test_metadata_switches_are_validated_and_inert`, `test_guide_locked_and_shy_flags_do_not_change_rendering` | Passed |
| Parenting 僅繼承 Transform、不繼承 opacity | `test_opacity_is_not_inherited_from_the_parent` | Child alpha stays 1.0 under a 25% parent |
| Auto-Orient along path | `test_auto_orient_follows_the_position_path`, `test_auto_orient_ignores_the_static_rotation_value`, `test_auto_orient_without_motion_falls_back_to_zero` | Down/right/up paths give 90/0/−90 degrees |
| Null / Solid / AV layer | `test_null_layer_contributes_no_pixels_but_can_parent`, `test_solid_layer_renders_the_requested_color`, `test_av_layer_reads_clip_frames_at_source_time` | Passed |
| Collapse / Continuously Rasterize 旗標 | `test_rendering_flags_are_stored_for_later_workstreams` | Stored, behavior deferred as specified |
| 與既有 `CompositeVideoClip` 在僅位置＋Normal 場景輸出相同 | `test_position_only_stack_matches_composite_video_clip` | Max difference ≤1 code; the legacy path quantizes per layer while this path stays float32 until the single final export |
| 確定性 | `test_apply_is_deterministic_and_repeatable`, `test_layer_render_is_deterministic_across_repeated_calls`, `test_keyframed_layer_animation_is_deterministic_and_moves` | Repeated renders are bit-identical |

## Delivered files

| Path | Lines | Role |
| --- | ---: | --- |
| `moviepy/ae/_geometry.py` | 118 | Shared validation primitives (private) |
| `moviepy/ae/warp.py` | 291 | Filter selection, destination bounds, `cv2.warpAffine` resampling |
| `moviepy/ae/transform.py` | 611 | `affine_matrix`, `Transform`, schema-1 serialization |
| `moviepy/ae/layers/__init__.py` | 35 | Lazy public exports |
| `moviepy/ae/layers/base.py` | 442 | `Layer` timing, switches, parenting, render |
| `moviepy/ae/layers/av.py` | 70 | `AVLayer` footage source |
| `moviepy/ae/layers/null.py` | 60 | `NullLayer` controller |
| `moviepy/ae/layers/solid.py` | 103 | `SolidLayer` constant color |
| `tests/ae/test_transform.py` | 534 | Transform and warp tests |
| `tests/ae/test_layers.py` | 377 | Layer, timing and parenting tests |
| `tests/ae/test_ws02_integration.py` | 213 | Stack, solo and legacy-parity integration |
| `tests/ae/test_ws02_boundaries.py` | 267 | Validation branches and degenerate geometry |
| `docs/ae/ws02.rst`, `docs/ae/ws02_change_notes.rst` | — | zh-TW guide with 7 executable examples plus change notes |

The module was split into `_geometry` / `warp` / `transform` because a single
`transform.py` reached 918 lines with two functions over 50 lines, which the
specification's structural limits forbid. No public name moved out of
`moviepy.ae.transform` except the resampling primitives, which live in
`moviepy.ae.warp`.

## Deliberate deviations and deferred scope

* **Pixel-center coordinates.** The automatic anchor is `((W-1)/2, (H-1)/2)`
  instead of AE's corner-based `(W/2, H/2)`. This half-pixel choice is what makes
  exact 90/180 degree rotations bit-comparable with `numpy.rot90`, and it is the
  same convention `Buffer.bounds` already uses. Documented in `ws02.rst`.
* **`three_d=True` raises.** The reserved fields `orientation`, `x_rotation`,
  `y_rotation`, `z_rotation`, `position_z` and `scale_z` are validated, stored
  and serialized but never applied; evaluation raises `NotImplementedError`
  naming WS-22 rather than silently ignoring them.
* **`moviepy/ae/__init__.py` is untouched.** `tests/ae/test_compatibility.py`
  pins `moviepy.ae.__all__` to the four WS-00 symbols and existing assertions are
  not modified, so the public import paths are `moviepy.ae.transform`,
  `moviepy.ae.warp` and `moviepy.ae.layers`. The aggregated `moviepy.ae` facade
  and the `VideoClip.to_ae_layer()` hook (§9.4 item 2) are deferred to WS-03,
  where `Composition` gives them a coherent surface.
* **`blend_mode` is a validated string token.** The 38 modes and their formulas
  belong to WS-04.
* **Deferred behavior, stored as flags only:** masks and track mattes (WS-05),
  effects and adjustment layers (WS-06), time remap and motion-blur sampling
  (WS-07), pre-composition, collapse-transformations and guide-layer render
  exclusion (WS-03), continuously-rasterize vectors (WS-20).
* **Cubic support margin.** Automatic destination bounds widen by the filter
  support radius (nearest 0.5, linear 1.0, cubic 2.0 source pixels), so a cubic
  render carries a one-pixel transparent margin that a linear render does not.
  This keeps filtered energy from being clipped; explicit `bounds` for
  region-of-interest rendering is exact instead.
* **Interpolation is not mass-conserving.** Bilinear and bicubic resampling of a
  delta does not preserve the alpha sum under rotation (measured 1.268 for a
  30 degree turn of a single impulse). This is normal discrete-resampling
  behavior, documented so it is not mistaken for a compositing bug.
* **OpenCV fixed-point precision.** Non-integer rotation angles place the sample
  point within about 1e-6 of the exact position, so a rotated impulse reads
  0.9999988 rather than 1.0. Tests use `abs=1e-5` there and document the cause.

## Independent review

Three read-only reviewers ran against the delivered files in parallel:
`code-reviewer` for Python quality, `jd-judge-a` as a blind adversarial
correctness reviewer over the transform, warp, parenting and timing semantics,
and `security-review` for the untrusted-JSON and memory-exhaustion surface.
Findings and their resolution are recorded in
`.ag-artifacts/runs/ws02-20261003/review-*.md`; CRITICAL and HIGH items are
mandatory to fix before this stage is considered complete.

## Reproduce

```bat
cd /d D:\moviepy
.tox\ae-ws00\Scripts\python.exe -m pytest tests\ae -q
.tox\ae-ws00\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.tox\ae-ws00\Scripts\python.exe -m pytest tests\ae -q --cov=moviepy/ae --cov-report=term-missing
.tox\ae-ws00\Scripts\python.exe -m pytest --doctest-modules moviepy\ae\transform.py moviepy\ae\warp.py moviepy\ae\_geometry.py moviepy\ae\layers -q
uvx --from black==23.7.0 black --check .
.tox\ae-ws00\Scripts\python.exe -m flake8 -v --show-source --ignore=E501 moviepy docs/conf.py examples tests
.tox\ae-ws00\Scripts\python.exe benchmarks\ae_bench.py --check --output benchmarks\results\ws02-regression-20261003.json
```

Black 26.5.1 is also present in the environment but would reformat 29
pre-existing files, so the repository-pinned 23.7.0 (matching
`.pre-commit-config.yaml`) is the only formatter used here. `.ag-artifacts/`
holds the raw logs and is supplementary evidence, not part of the delivery.

## Review Findings and Resolution

### Summary

Three independent review passes identified 21 findings (9 CRITICAL/HIGH, 5 MEDIUM):
- **python-reviewer**: Python quality and idioms
- **ae-correctness**: After Effects parity defects  
- **security-reviewer**: Input validation and DoS

All CRITICAL and HIGH findings have been fixed with regression tests.

### Critical Finding

**C1: Stale uniform_alpha metadata**
- Location: `warp._translated` line 231-235
- Impact: Wrong compositing and export corruption
- Fix: Scale uniform_alpha by opacity factor in metadata
- Test: `test_c1_translated_scales_uniform_alpha`

### High Severity Findings

**H1: Transform properties use composition time instead of layer time**
- Location: `Layer.local_matrix`, `opacity_at` 
- Impact: Animation drift with start_time/stretch
- Fix: Map through `source_time(t)` before property evaluation
- Test: `test_h1_layer_time_for_transform_and_opacity`

**H2: Nearest filter drops boundary pixels**  
- Location: `warp._snap_rectangle`
- Impact: 2 pixels/axis lost at integer magnifications
- Fix: Add inclusive mode for nearest filter
- Test: `test_h2_nearest_filter_includes_boundary_pixels`

**H3: AVLayer.default_out_point ignores start_time**
- Location: `AVLayer.default_out_point`
- Impact: Negative timestamps reach video readers
- Fix: Use `start_time + duration`, clamp source times
- Test: `test_h3_av_layer_default_out_point_includes_start_time`

**H4: Memory exhaustion via unbounded allocation**
- Location: `_geometry.validate_pixel_size`
- Impact: DoS through crafted project files
- Fix: Add MAX_RENDER_PIXELS = 2^26 budget check
- Test: `test_h4_area_budget_prevents_memory_exhaustion`

### Medium Severity Findings (Fixed)

- **M1**: Asymmetric dimension guard → Check all four extremes
- **M3**: in_point setter validation → Cross-check with out_point
- **M4**: Cubic overflow to infinity → Track and scan when at risk
- **M5**: auto_orient zero after motion → Hold final tangent

### Files Modified

- `moviepy/ae/_geometry.py`: Area budget, matrix copy
- `moviepy/ae/warp.py`: Metadata fix, inclusive snap, overflow check
- `moviepy/ae/transform.py`: Layer time, tangent holding
- `moviepy/ae/layers/base.py`: Layer time, in_point validation
- `moviepy/ae/layers/av.py`: out_point formula, source clamping
- `moviepy/ae/layers/solid.py`: Import order, MemoryError wrap
- `tests/ae/test_transform.py`: Updated for H2 behavior
- `tests/ae/test_ws02_review_fixes.py`: 13 new regression tests

### Validation After Fixes

All 753 tests pass, 100% coverage maintained, benchmark unchanged.
