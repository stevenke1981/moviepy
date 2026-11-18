# WS-06 acceptance — verification, 2026-11-19

WS-06 delivers `moviepy.ae.effects`: `AEEffect`, `Param`, `EffectStack`, the registry, `moviepy.ae.fx`, `from_moviepy_effect` and adjustment layers (`Composition.add_adjustment`). The registry now lists 29 effects in 9 categories (measured with `registry.load_builtins(); len(registry.list())`).
The guide is [ws06.rst](ws06.rst); changes are recorded in
[ws06_change_notes.rst](ws06_change_notes.rst). This document records freshly run evidence on branch
`feat/ae-parity` and lists the items that are not met.

## Current validation

| Gate | Actual result |
| --- | --- |
| Whole AE suite | **3579 passed, 44 skipped, 9 failed**, 170.57 s (`pytest tests/ae`, run before `test_mask_path_coverage.py` was added) |
| WS-06 test files | **362 passed** (`test_effects.py`, `test_blur_sharpen_effects.py`, `test_gaussian_padding.py`, `test_gaussian_round2.py`, `test_distort_effects.py`, `test_stylize_noise_effects.py`, `test_transition_effects.py`, `test_color_correction_more.py`) |
| Entire AE coverage | **89%** line+branch (20404 statements, 1830 missed; 6990 branches, 927 partial) |
| Per-module coverage (line+branch) | effects/base.py **92%**; effects/stack.py **96%**; effects/registry.py **96%**; effects/bridge.py **92%**; effects/adjustment.py **95%**; layers/adjustment.py **100%**; effects/blur/gaussian_blur.py **100%** |
| Guide examples | `tests/ae/test_guides.py`: 5 passed, 1 failed (`ws02.rst`, pre-existing; `ws06.rst` passes) |
| Independent review | not recorded |
| Performance | see benchmarks/results/ws03-06-2026-11-19.json (added in the same release) |

The coverage figures come from one run of
`python -m pytest tests/ae -q -p no:cacheprovider --cov=moviepy/ae --cov-branch --cov-report=term`
(Python 3.12.10, Windows). The nine failures in that run are listed below; none
is specific to this workstream's modules unless stated.

## Not met or open

- Effect modules below 90%: `effects/distort/wave_warp.py` **80%** (5 statements missed) and `effects/distort/_sampling.py` **86%**.
- Gaussian blur still clamps its output to [0, 1] at this base (`gaussian_blur.py:91`), so HDR values above 1 do not survive an adjustment blur (review plan D4).
- The 40 ms / 150 ms single-thread Gaussian budgets quoted in ws06.rst are not met; see the benchmark file.
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
python -m pytest tests/ae/test_effects.py tests/ae/test_blur_sharpen_effects.py tests/ae/test_gaussian_padding.py tests/ae/test_gaussian_round2.py tests/ae/test_distort_effects.py tests/ae/test_stylize_noise_effects.py tests/ae/test_transition_effects.py tests/ae/test_color_correction_more.py -q -p no:cacheprovider
```
