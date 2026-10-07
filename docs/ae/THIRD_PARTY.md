# WS-00 third-party reuse

Checked on 2026-10-02 against the installed WS-00 environment and the projects'
official license pages. These packages already appear in MoviePy's existing
runtime dependencies; WS-00 adds no dependency. It calls public library APIs
rather than copying their implementation source. No GPL or proprietary source
code was copied into the new WS-00 modules, tests, assets, or benchmark.

| Existing package inspected | Project license and installed evidence | WS-00 use |
| --- | --- | --- |
| NumPy 2.4.6 | Project: BSD-3-Clause. Installed `METADATA` reports `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`; `licenses/LICENSE.txt` contains the NumPy notice and bundled-library notices. [Official project license](https://numpy.org/doc/stable/license.html), [versioned 2.4.6 license](https://raw.githubusercontent.com/numpy/numpy/v2.4.6/LICENSE.txt). | float32 RGBA storage, premultiplication, source-over fallback, validation, independent float64 reference calculations, array comparison, PCG64 RNG and synthetic assets. |
| Pillow 12.3.0 | MIT-CMU, recorded in `METADATA` and `licenses/LICENSE`. [Official license documentation](https://pillow.readthedocs.io/en/stable/about.html#license), [versioned license](https://github.com/python-pillow/Pillow/blob/12.3.0/LICENSE). | Golden PNG read/write, mode checks, independently generated synthetic reference images; existing MoviePy image compatibility. |
| opencv-python-headless 5.0.0.93 / cv2 5.0.0 | OpenCV library: Apache-2.0. Installed package `METADATA` labels Apache 2.0; its `LICENSE.txt` covers packaging scripts under MIT, and `LICENSE-3RD-PARTY.txt` contains the OpenCV and bundled-library notices. [Official OpenCV license](https://opencv.org/license/), [official packaging license](https://raw.githubusercontent.com/opencv/opencv-python/master/LICENSE.txt), [official packaging licensing notes](https://github.com/opencv/opencv-python#licensing). | Public CPU float32 `addWeighted` for safe uniform-alpha over, `mixChannels`/`convertScaleAbs` for final export, raw Gaussian sigma20 benchmark, and explicit benchmark thread/OpenCL controls. |

The metadata and license files were read from these installed distribution
directories under `.tox/ae-ws00/Lib/site-packages/`:

- `numpy-2.4.6.dist-info/`
- `pillow-12.3.0.dist-info/`
- `opencv_python_headless-5.0.0.93.dist-info/`

The project licenses above do not replace notices for components bundled in
binary wheels. For example, this NumPy wheel's license file lists OpenBLAS and
LAPACK notices, plus GCC runtime code under
`GPL-3.0-or-later WITH GCC-exception-3.1`. The OpenCV packaging notes and installed
third-party file list FFmpeg under LGPLv2.1. Those are existing packaged binary
components, distinct from copying their source into WS-00. Preserve the complete
installed notices when redistributing those wheels; consult the packaged files
and [upstream third-party notice list](https://github.com/opencv/opencv-python/blob/master/LICENSE-3RD-PARTY.txt)
for their component-specific terms. This page records WS-00's reuse and does not
claim a new audit of every transitive bundled component.

Normal source-over uses the published mathematical definition, with the
independent test oracle written for this project. The benchmark's Gaussian
reference is an independent float64 convolution; production timing calls the
installed cv2 primitive. Golden assets are synthetic and carry their own
`tests/ae/assets/LICENSE-ASSETS.txt` notice.

## Optional SVG backend (2026-10-07)

The `svg` extra adds the following exact pins; it does not change core runtime
dependencies. No upstream implementation, binary or font is copied into MoviePy.

| Package | Source and license | Use |
| --- | --- | --- |
| `resvg-py==0.2.6` | [Official release and wheels](https://pypi.org/project/resvg-py/0.2.6/); wrapper `LICENSE` in its source distribution is MIT, copyright 2024 baseplate-admin. Its source `Cargo.lock` fixes `resvg` and `usvg` at 0.46.0; resvg offers [MIT](https://github.com/linebender/resvg/blob/v0.46.0/LICENSE-MIT) or [Apache-2.0](https://github.com/linebender/resvg/blob/v0.46.0/LICENSE-APACHE). | Lazy optional static SVG-to-PNG rasterizer. The selected release provides Windows x64 and Linux x64 wheels for both Python 3.9 and 3.12. Newer wrapper releases drop part of that matrix. |
| `fonttools==4.59.2` | [Official release](https://pypi.org/project/fonttools/4.59.2/) and [versioned MIT license](https://github.com/fonttools/fonttools/blob/4.59.2/LICENSE). Supports Python >=3.9. No fontTools extras are selected. | Read supplied font family names and Unicode cmap coverage. Native font shaping remains in resvg. |

The wrapper's source distribution SHA-256 is
`dd8942159cbefd3f43389816e90065637dd1e89094f62bc9bd52e62513523444`.
The accompanying `svg_dependency_lock.json` records official artifact URLs,
SHA-256 digests for the four target wrapper wheels, the portable fontTools wheel,
and license metadata for all 80 crates in this source distribution's Cargo.lock.
Each crate license was checked against its exact-version crates.io record;
the source checksum is preserved. That list includes build/target dependencies,
so it is not a claim that every listed crate is linked into every wheel.

The metadata uses MIT, Apache-2.0, BSD, Zlib and related permissive alternatives,
plus Unicode-3.0 and Apache-2.0 WITH LLVM-exception where recorded. This is a
source-lock and registry-license review, not a binary composition audit or a
replacement for complete notices supplied with downloaded wheels. Inspect and
preserve those notices before redistributing native binaries. A version upgrade
must repeat provenance, platform, alpha and rendering checks.

The approved Windows CPython 3.12 wrapper wheel and portable fontTools wheel
were downloaded from the recorded official URLs and verified against their
SHA-256 digests before installation. The wrapper wheel contains its MIT
`licenses/LICENSE`; it does not contain a complete per-crate notice bundle.
The fontTools wheel contains `licenses/LICENSE` and `LICENSE.external`; the
latter includes SIL OFL-1.1 test-font notices, Adobe AGL/AGLFN BSD-3-Clause,
cu2qu Apache-2.0, and PyFilesystem2 MIT notices. Keep both files when packaging
that wheel. The integration does not redistribute either wheel or any of those
upstream test fonts.

SVG tests generate their small Latin/CJK test font from original geometric
glyphs at runtime. User-provided production fonts retain their own licenses and
are neither installed nor distributed by this integration. See `svg.rst` for
the supported input subset and font handling.
