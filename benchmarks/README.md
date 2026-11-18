# WS-00 CPU benchmarks

Run from the checkout with a Python environment that has NumPy and OpenCV
installed (`python` below is that interpreter):

```shell
python benchmarks/ae_bench.py --check --output benchmarks/results/ws00-validation-unique.json
```

The CLI writes one JSON object to stdout and, when requested, to a new output
file. An existing output file is rejected before measurements so prior evidence
is preserved. Choose a new filename for each run. Runtime errors are reported on
stderr and retain their traceback. Exit status is 0 for a successful requested
check, 1 for a failed acceptance check, and 2 for invalid CLI arguments.
Without `--check`, completed diagnostic measurements return 0 while the JSON
still records the actual acceptance status. Provenance child commands have a
ten-second timeout; unavailable or timed-out metadata is labelled explicitly.

All cases use **1920 × 1080**. Defaults are five warmup frames, five repeats of
20 frames each. Each repeat reports elapsed seconds, milliseconds per frame and
fps. The aggregate is the median repeat frame time and its reciprocal fps;
the fastest repeat is never substituted for the median.

| Case | Timed work |
| --- | --- |
| `single` | One final RGB export from a prebuilt opaque Buffer. |
| `normal_10` | Ten full-canvas semitransparent foreground `composite_over` calls over a base Buffer, including actual allocations and copies, followed by one final RGB export. All input Buffers are prebuilt and readonly. |
| `normal_10_end_to_end` | Import the base and all ten foreground sources using `Buffer.from_uint8_rgb`, then perform the same ten over calls and one final RGB export. This includes source conversion but excludes synthetic fixture generation and Python module loading. |
| `gaussian_sigma20` | Raw `cv2.GaussianBlur` on the base Buffer's readonly float32 RGBA (four channels), sigma 20, automatic kernel size, `BORDER_REFLECT_101`. No WS-10 effect class or GPU operation is involved. |

The primary WS-00 requirement is **`normal_10` median ≥10 fps**. The complete
protocol and all correctness checks must also pass. A partial run or changed
sample count cannot pass `--check`, even if its measured fps exceeds ten.
Fixture generation, correctness proofs, hashes and provenance reads occur
outside all timed sections. Every measured frame recomputes the actual work;
there is no output cache, ROI, reduced resolution, intermediate uint8 conversion
or opaque foreground shortcut. All ten foreground masks have alpha strictly
between zero and one. The opaque base does not skip any over call.

NumPy/OpenCV/MoviePy imports happen after process-local thread environment
variables are set to one. OpenCV is additionally configured with one CPU thread
and OpenCL disabled. This limits participating numerical-library CPU worker
threads and does not alter the user's persistent environment or pin the process
to a particular CPU core. Do not run competing tests or rendering jobs when
collecting comparative measurements.

Normal correctness uses an independent float64 source-over oracle built from
the original uint8 RGB and masks. It checks the full float RGBA and final RGB
shapes and values (float error ≤2e-6, final code error ≤one level), then recomputes
the complete image to check bitwise determinism. Gaussian correctness uses an
independent float64, 161-tap separable Gaussian calculation at five locations,
including the image corners, and checks full-output deterministic hashes. This
Gaussian check validates the cv2 primitive's samples rather than a future blur
effect's complete contract.

The JSON records CPU/OS/architecture, dependency versions, commit and dirty
working-tree status, source hashes, thread controls, all repeat measurements,
timing boundaries, oracle errors and output hashes. Git provenance includes
untracked files; source hashes identify the actual new implementation when the
commit alone does not. The first result establishes an initial baseline.
Historical ±20% regression checking requires a later comparable same-machine,
same-protocol result; a first measurement does not prove absence of regression.

The implementation may reuse at most two released raw float32 allocations,
bounded to 64 MiB in total. A storage lease prevents reuse until every array and
escaped view has died. This reuses storage, not rendered frames: each operation
fully writes its output, and live readonly Buffer values retain their pixels.
License and API reuse notes are in `docs/ae/THIRD_PARTY.md`.

For a shorter diagnostic or profiler run, select cases and counts:

```shell
python benchmarks/ae_bench.py --cases normal_10 --warmup 1 --repeats 1 --frames 2 --output benchmarks/results/ws00-diagnostic-unique.json
```

`--warmup` accepts nonnegative counts; `--repeats` and `--frames` require positive
counts. Resolution, foreground-layer count and Gaussian sigma are fixed, so a
diagnostic cannot silently substitute a small workload for the required workload.

## Reference machine and preserved evidence

The initial reference machine is an **AMD Ryzen 9 3900X 12-Core Processor**,
Windows x86-64. Actual version, thread and working-tree provenance is stored in
`results/ws00-initial-20261002.json`. The result file is the source of truth for
the measured initial fps and acceptance outcome.

The initial full run exited **1** under `--check`: the 10-layer Normal
performance requirement was not met. Every case's correctness and determinism
checks passed. These measurements precede performance tuning and are retained
without overwriting:

| Initial case | Median ms/frame | Median fps |
| --- | ---: | ---: |
| `single` | 195.883 | 5.105 |
| `normal_10` | 658.540 | 1.519 |
| `normal_10_end_to_end` | 1524.538 | 0.656 |
| `gaussian_sigma20` (initial RGB, three channels) | 248.783 | 4.020 |

Initial versions: Python 3.11.14, NumPy 2.4.6, OpenCV 5.0.0, MoviePy 2.2.0,
Pillow 12.3.0 and imageio 2.38.0; OS build Windows 10.0.19045. The Normal
float64 oracle maximum error was 9.10024e-8 with zero final RGB code error;
Gaussian sample maximum error was 2.68049e-7. Subsequent results must be recorded
in separate JSON files. The final Gaussian workload uses Buffer RGBA (four
channels); its timing must not be compared with the initial three-channel RGB
measurement as a performance regression. The Normal timed workload is unchanged.

## Final full-protocol result

`results/ws00-validation.json` records the final run at
2026-10-02 15:55:34 UTC. **`--check` passed with exit status 0**. The complete
1920×1080 protocol used five warmup frames and five repeats of twenty frames
for every case, one CPU numerical-library thread, and OpenCL disabled.

| Final case | Median ms/frame | Median fps |
| --- | ---: | ---: |
| `single` | 15.815 | 63.230 |
| `normal_10` | 91.547 | **10.923** |
| `normal_10_end_to_end` | 1020.306 | 0.980 |
| `gaussian_sigma20` (RGBA, four channels) | 326.100 | 3.067 |

All five primary Normal repeats exceeded ten fps: 10.923, 10.250, 10.280,
10.963 and 12.163. Every case passed its correctness and full-output determinism
checks. Normal's independent float64 maximum error was 8.24454e-8 and final RGB
code error was zero. The exact results and output hashes are retained in JSON.

The **10.923 fps acceptance applies to prebuilt readonly inputs**, including
ten real full-canvas over operations and one final RGB export. Including eleven
source imports costs 1020.306 ms/frame, or 0.980 fps, on the same machine.
The primary result therefore does not claim ten-fps throughput for repeated
import-and-compose workflows.

Compared with the preserved initial run, the unchanged primary Normal workload
improved from 1.519 to 10.923 fps, approximately 7.19× within this session.
This is a measured same-machine optimization comparison, not a historical CI
regression gate. No historical baseline exists, so that gate remains N/A.
Gaussian's initial three-channel RGB and final four-channel RGBA timings are
different workloads and are not used for a regression comparison.

The final JSON records Buffer source SHA256
`b2248f4a773c0f6214d644a33a6b819b23d589e4af287fb0af4b81f6f7c02ac4`, matching
the independently reviewed implementation. Numeric source hashes refer to
the measured snapshot; this README and the other documentation can be updated
without changing its rendering source or workload.

## WS-03..06 compositing benchmarks

`ae_comp_bench.py` measures the compositing work added by WS-03 to WS-06 and
records each result against the targets in `docs/ae/AE_REVIEW_PLAN.md` §3.
Targets are **recorded, not gated**: the CLI exits 0 whatever they say, so it
can run anywhere without failing a build.

```shell
python benchmarks/ae_comp_bench.py --output benchmarks/results/ws03-06-NEW.json
```

It reuses the WS-00 protocol by importing `ae_bench.py`: one numerical-library
CPU thread, OpenCL off, warmup calls, then repeats of N frames, reporting the
median repeat. To keep a full run under ten minutes the defaults are **two
warmup calls and five repeats of five frames** (WS-00 uses 5 / 5 / 20). The
committed run took 224 seconds. An existing output file is refused, as with
WS-00.

| Case | Timed work |
| --- | --- |
| `blend_modes` | `blend(base, layer, mode)` for all 38 modes on prebuilt full-canvas Buffers: opaque textured backdrop, textured source whose alpha ramps 0.2..0.9 across x (so no uniform-alpha fast path applies). |
| `comp_backdrop_only` | `Composition.get_frame(t)` with only the textured 1080p `ImageClip` backdrop. Every other comp case contains this backdrop, so this row is its fixed cost. |
| `comp_20_layers` | Backdrop plus twenty moving 600×400 solids at opacity 60. |
| `feathered_mask` | Backdrop plus one full-frame moving solid with an ellipse mask whose feather animates 40..80 px (animated so the mask cache cannot reuse coverage). |
| `adjustment_gaussian_blur_40` | Backdrop plus one moving 600×400 solid under a full-frame adjustment layer with `GaussianBlur(blurriness=40)`. |
| `echo_single` | The same solid under one adjustment layer with `Echo(echo_time=-0.05, number_of_echoes=6)`. |
| `echo_stacked` | The same solid under two such Echo adjustment layers. |

Comp cases time the full `get_frame(t)`, including the uint8 export. `t`
advances one frame per call, so no frame is reused. Each case also records
an output SHA-256 and checks that rebuilding the scene gives the same frame.

Useful options: `--cases` and `--modes` select a subset, and `--size 64x36`
scales every 1080p-defined size (layers, blur, feather) for a quick smoke run.
`tests/ae/test_comp_bench.py` runs it that way in about a second.

### Measured vs target (2026-11-19 local, 2026-11-18 23:51 UTC)

`results/ws03-06-2026-11-19.json` is the run below, at 1920×1080 on the
reference machine (AMD Ryzen 9 3900X 12-Core Processor, Windows 11, Python
3.12.10, NumPy 2.5.5, OpenCV 5.1.0). It was measured at commit
`09ddd64` plus the then-untracked bench and test files.
The JSON records them in `git_status_porcelain`, with source hashes. Every
case was deterministic, and every blend output was finite.

**Summary: 0 of 27 recorded targets met.** All
25 separable blend modes are above 2× Normal. The 20-layer comp and the
adjustment blur are both above 150 ms. No performance fixes were attempted in
this change. This is the first WS-03..06 result, so §9.3's 20% regression
comparison is not applicable yet. Later runs should be compared with this file
on the same machine.

| Comp case | Median ms/frame | Median fps | Target | Result | Deterministic |
| --- | ---: | ---: | --- | --- | --- |
| `adjustment_gaussian_blur_40` | 440.6 | 2.27 | < 150 ms | **miss** | true |
| `comp_20_layers` | 347.6 | 2.88 | < 150 ms | **miss** | true |
| `comp_backdrop_only` | 38.4 | 26.07 | — | n/a | true |
| `echo_single` | 669.6 | 1.49 | — | n/a | true |
| `echo_stacked` | 2563.5 | 0.39 | — | n/a | true |
| `feathered_mask` | 247.9 | 4.03 | — | n/a | true |

Because the comp cases include the 38.4 ms backdrop, the 20-layer case is
about 309 ms above the backdrop alone, which still misses the 150 ms target
by itself. Feathered mask and Echo have no numeric target in §3. Echo is
listed under R6 (render caching), so these rows are a baseline only.

| Blend mode | Kind | Median ms | vs Normal | Target | Result |
| --- | --- | ---: | ---: | --- | --- |
| `normal` | normal | 26.5 | 1.00× | — | n/a |
| `add` | unclamped | 89.8 | 3.39× | — | n/a |
| `alpha_add` | alpha_add | 66.3 | 2.50× | — | n/a |
| `classic_color_burn` | classic | 83.2 | 3.14× | — | n/a |
| `classic_color_dodge` | classic | 85.4 | 3.22× | — | n/a |
| `classic_difference` | classic | 73.4 | 2.77× | — | n/a |
| `color` | separable | 141.0 | 5.32× | ≤ 2.0× | **miss** |
| `color_burn` | separable | 98.2 | 3.71× | ≤ 2.0× | **miss** |
| `color_dodge` | separable | 98.4 | 3.71× | ≤ 2.0× | **miss** |
| `dancing_dissolve` | dissolve | 138.4 | 5.22× | — | n/a |
| `darken` | separable | 84.4 | 3.19× | ≤ 2.0× | **miss** |
| `darker_color` | separable | 102.6 | 3.87× | ≤ 2.0× | **miss** |
| `difference` | separable | 84.4 | 3.18× | ≤ 2.0× | **miss** |
| `dissolve` | dissolve | 140.4 | 5.30× | — | n/a |
| `divide` | separable | 96.1 | 3.63× | ≤ 2.0× | **miss** |
| `exclusion` | separable | 86.9 | 3.28× | ≤ 2.0× | **miss** |
| `hard_light` | separable | 101.9 | 3.85× | ≤ 2.0× | **miss** |
| `hard_mix` | separable | 87.5 | 3.30× | ≤ 2.0× | **miss** |
| `hue` | separable | 170.3 | 6.42× | ≤ 2.0× | **miss** |
| `lighten` | separable | 97.7 | 3.68× | ≤ 2.0× | **miss** |
| `lighter_color` | separable | 101.5 | 3.83× | ≤ 2.0× | **miss** |
| `linear_burn` | separable | 91.4 | 3.45× | ≤ 2.0× | **miss** |
| `linear_dodge` | separable | 85.9 | 3.24× | ≤ 2.0× | **miss** |
| `linear_light` | separable | 82.6 | 3.12× | ≤ 2.0× | **miss** |
| `luminescent_premul` | normal | 28.9 | 1.09× | — | n/a |
| `luminosity` | separable | 148.2 | 5.59× | ≤ 2.0× | **miss** |
| `multiply` | separable | 84.5 | 3.19× | ≤ 2.0× | **miss** |
| `overlay` | separable | 103.9 | 3.92× | ≤ 2.0× | **miss** |
| `pin_light` | separable | 89.8 | 3.39× | ≤ 2.0× | **miss** |
| `saturation` | separable | 166.0 | 6.26× | ≤ 2.0× | **miss** |
| `screen` | separable | 85.1 | 3.21× | ≤ 2.0× | **miss** |
| `silhouette_alpha` | silhouette | 69.6 | 2.63× | — | n/a |
| `silhouette_luma` | silhouette | 73.3 | 2.77× | — | n/a |
| `soft_light` | separable | 125.7 | 4.74× | ≤ 2.0× | **miss** |
| `stencil_alpha` | stencil | 70.2 | 2.65× | — | n/a |
| `stencil_luma` | stencil | 72.4 | 2.73× | — | n/a |
| `subtract` | separable | 89.2 | 3.36× | ≤ 2.0× | **miss** |
| `vivid_light` | separable | 123.2 | 4.65× | ≤ 2.0× | **miss** |

The ≤ 2× Normal target applies only to modes whose kind is `separable`. For
the other kinds (dissolve, classic, stencil, silhouette, alpha add, unclamped
Add) the ratio is recorded without a target. §3 also quotes an absolute
≤ 25 ms spec for blend modes; Normal measured 26.5 ms here.

### Slow memory test

`tests/ae/test_memory_slow.py` renders 20 moving 600×400 solids on a 1080p
comp for 20 warmup frames and then 200 measured frames. It asserts that
resident memory and `tracemalloc` traced memory each grow by less than 5%. It is
skipped unless `AE_SLOW=1` is set:

```shell
AE_SLOW=1 python -m pytest tests/ae/test_memory_slow.py -q -s
```

On the reference machine it passed in 34.94 s. RSS went from 149336064 to
150257664 bytes (+0.62%), and traced memory went from 84723808 to 84942262
bytes (+0.26%).
