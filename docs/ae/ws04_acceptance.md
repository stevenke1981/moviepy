# WS-04 acceptance — verification, 2026-11-19

WS-04 delivers `moviepy.ae.blend`: 38 blend modes (`BlendMode`), `blend(base, layer, mode, opacity)` and the stencil/silhouette alpha modes, on premultiplied float32 buffers.
The guide is [ws04.rst](ws04.rst); changes are recorded in
[ws04_change_notes.rst](ws04_change_notes.rst). This document records freshly run evidence on branch
`feat/ae-parity` and lists the items that are not met.

## Current validation

| Gate | Actual result |
| --- | --- |
| Whole AE suite | **3579 passed, 44 skipped, 9 failed**, 170.57 s (`pytest tests/ae`, run before `test_mask_path_coverage.py` was added) |
| WS-04 test files | **508 passed** (`test_blend.py`, `test_blend_gaussian_round3.py`, `test_blend_performance_round2.py`, `test_blend_tiles.py`, `test_hue_round4.py`, `test_stencil_round2.py`) |
| Entire AE coverage | **89%** line+branch (20404 statements, 1830 missed; 6990 branches, 927 partial) |
| Per-module coverage (line+branch) | blend/modes.py **90%** (214 stmts, 19 missed); _formulas.py **96%**; _opaque.py **96%**; _uniform.py **98%**; alpha.py **96%**; blend/__init__.py **92%** |
| Guide examples | `tests/ae/test_guides.py`: 5 passed, 1 failed (`ws02.rst`, pre-existing; `ws04.rst` passes) |
| Independent review | not recorded |
| Performance | see benchmarks/results/ws03-06-2026-11-19.json (added in the same release) |

The coverage figures come from one run of
`python -m pytest tests/ae -q -p no:cacheprovider --cov=moviepy/ae --cov-branch --cov-report=term`
(Python 3.12.10, Windows). The nine failures in that run are listed below; none
is specific to this workstream's modules unless stated.

## Not met or open

- `flake8 moviepy/ae/blend/alpha.py` reports `alpha.py:5:89: E501` at this base (review plan section 5.4); a separate package is expected to fix it.
- Stencil/Silhouette at layer opacity 0 keeps the layer below (project-chosen semantics, documented in ws04.rst); it has not been compared against real After Effects.
- Independent reviewer sign-off (spec section 9.2): **not recorded**. No review
  report for this workstream exists in the repository.
- Whole-suite failures observed in the same run, owned by other packages:
  `test_guides[ws02.rst]`, `test_linear_master` (ffv1 real encode),
  `test_motion_blur_round2` (moving matte), `test_music_render` (long filter
  graph), `test_soundx` (lufs normalize), `test_transform` rot90 linear x3 and
  `test_benchmark::test_diagnostic_cli_writes_json_and_check_fails` (the last
  also fails when run alone: `python -m pytest tests/ae/test_benchmark.py`
  gives 1 failed, 5 passed).

## Reproduce

```
python -m pytest tests/ae/test_blend.py tests/ae/test_blend_gaussian_round3.py tests/ae/test_blend_performance_round2.py tests/ae/test_blend_tiles.py tests/ae/test_hue_round4.py tests/ae/test_stencil_round2.py -q -p no:cacheprovider
```
