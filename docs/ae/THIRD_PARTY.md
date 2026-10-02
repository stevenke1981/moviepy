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
