# WS-00 CPU benchmarks

Run from the checkout with its validated Python environment:

```powershell
.tox\ae-ws00\Scripts\python.exe benchmarks/ae_bench.py --check --output benchmarks/results/ws00-validation-unique.json
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

```powershell
.tox\ae-ws00\Scripts\python.exe benchmarks/ae_bench.py --cases normal_10 --warmup 1 --repeats 1 --frames 2 --output benchmarks/results/ws00-diagnostic-unique.json
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
