# WS-03 acceptance — verification, 2026-11-19

WS-03 delivers `moviepy.ae.composition` (`Composition`, `Marker`, `from_moviepy`), `moviepy.ae.renderer` (`Renderer`, per-render `_RenderCache`) and `moviepy.ae.layers.CompLayer` (pre-compose). Layer audio is described in [composition_audio.rst](composition_audio.rst).
The guide is [ws03.rst](ws03.rst); changes are recorded in
[ws03_change_notes.rst](ws03_change_notes.rst). This document records freshly run evidence on branch
`feat/ae-parity` and lists the items that are not met.

## Current validation

| Gate | Actual result |
| --- | --- |
| Whole AE suite | **3579 passed, 44 skipped, 9 failed**, 170.57 s (`pytest tests/ae`, run before `test_mask_path_coverage.py` was added) |
| WS-03 test files | **193 passed** (`test_composition.py`, `test_composition_audio_round2.py`, `test_review_r2_identity.py`, `test_review_r3_roi.py`, `test_context.py`) |
| Entire AE coverage | **89%** line+branch (20404 statements, 1830 missed; 6990 branches, 927 partial) |
| Per-module coverage (line+branch) | composition.py **94%** (278 stmts, 11 missed); renderer.py **94%** (228 / 9); layers/comp.py **97%**; context.py **100%**; buffer.py **98%** |
| Guide examples | `tests/ae/test_guides.py`: 5 passed, 1 failed (`ws02.rst`, pre-existing; `ws03.rst` passes) |
| Independent review | not recorded |
| Performance | see benchmarks/results/ws03-06-2026-11-19.json (added in the same release) |

The coverage figures come from one run of
`python -m pytest tests/ae -q -p no:cacheprovider --cov=moviepy/ae --cov-branch --cov-report=term`
(Python 3.12.10, Windows). The nine failures in that run are listed below; none
is specific to this workstream's modules unless stated.

## Not met or open

- No module in this workstream is below 90% coverage.
- Collapse Transformations only skips cropping to the sub-composition frame; child blend modes still resolve inside the sub-composition and there is no resampling saving (see the known limits in ws03.rst).
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
python -m pytest tests/ae/test_composition.py tests/ae/test_composition_audio_round2.py tests/ae/test_review_r2_identity.py tests/ae/test_review_r3_roi.py tests/ae/test_context.py -q -p no:cacheprovider
```
