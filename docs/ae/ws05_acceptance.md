# WS-05 acceptance — verification, 2026-11-19

WS-05 delivers `moviepy.ae.masks`: Bezier mask paths (`masks.path`), scanline rasterization with feather and expansion, `Mask`/`MaskStack` modes and Alpha/Luma track mattes.
The guide is [ws05.rst](ws05.rst); changes are recorded in
[ws05_change_notes.rst](ws05_change_notes.rst). This document records freshly run evidence on branch
`feat/ae-parity` and lists the items that are not met.

## Current validation

| Gate | Actual result |
| --- | --- |
| Whole AE suite | **3579 passed, 44 skipped, 9 failed**, 170.57 s (`pytest tests/ae`, run before `test_mask_path_coverage.py` was added) |
| WS-05 test files | **74 passed** (`test_masks.py`, `test_mask_path_coverage.py` (new, 17 tests)) |
| Entire AE coverage | **89%** line+branch (20404 statements, 1830 missed; 6990 branches, 927 partial) |
| Per-module coverage (line+branch) | masks/path.py **88%** in the full run above, **99%** (200 stmts, 0 missed; 66 branches, 1 partial) after adding `test_mask_path_coverage.py` (focused run of the masks tests with `--cov=moviepy/ae/masks --cov-branch`); rasterize.py **95%**; matte.py **92%**; masks/__init__.py **92%**; mask.py **88%** |
| Guide examples | `tests/ae/test_guides.py`: 5 passed, 1 failed (`ws02.rst`, pre-existing; `ws05.rst` passes) |
| Independent review | not recorded |
| Performance | see benchmarks/results/ws03-06-2026-11-19.json (added in the same release) |

The coverage figures come from one run of
`python -m pytest tests/ae -q -p no:cacheprovider --cov=moviepy/ae --cov-branch --cov-report=term`
(Python 3.12.10, Windows). The nine failures in that run are listed below; none
is specific to this workstream's modules unless stated.

## Not met or open

- `masks/mask.py` is at **88%** (22 statements missed), below the 90% bar that the plan set for `path.py`; no test was added for it in this package.
- `masks.path.flatten` subdivides curves uniformly (not adaptively as the spec says); area error is covered by tests, see ws05.rst.
- The `MaskPath` alias required by the spec is being added to `moviepy.ae.masks` in this release; it is not present at this base.
- Mask motion blur, variable-width feather and RotoBezier are not implemented.
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
python -m pytest tests/ae/test_masks.py tests/ae/test_mask_path_coverage.py -q -p no:cacheprovider --cov=moviepy/ae/masks --cov-branch --cov-report=term-missing
```
