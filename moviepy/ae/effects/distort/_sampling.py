"""Shared inverse-mapping sampler for the Distort effects (NumPy only)."""

import numpy as np


def sample_bilinear(rgba, xs, ys):
    """Bilinear, zero-filled premultiplied lookup at pixel-center coordinates.

    ``xs``/``ys`` are source positions in pixel-edge coordinates (the center
    of pixel ``i`` is ``i + 0.5``). A sample is zero unless its pixel-index
    position lies inside ``[0, n - 1]`` on both axes; inside, the four
    neighbours are blended with exact float weights. This matches
    ``scipy.ndimage.map_coordinates(order=1, mode="constant", cval=0)``
    without requiring SciPy.

    Examples
    --------
    >>> import numpy as np
    >>> rgba = np.zeros((1, 2, 4), np.float32)
    >>> rgba[0, 1] = 1
    >>> sample_bilinear(rgba, np.array([[1.0, 2.5]]), np.array([[0.5, 0.5]]))[0, :, 0]
    array([0.5, 0. ], dtype=float32)
    """
    height, width = rgba.shape[:2]
    # Round away float noise (e.g. cos(90 deg) = 6e-17) so a point on the
    # true edge is not rejected as a hair outside the array.
    x = np.round(np.asarray(xs, dtype=np.float64) - 0.5, 9)
    y = np.round(np.asarray(ys, dtype=np.float64) - 0.5, 9)
    inside = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x0 = np.clip(np.floor(x), 0, max(width - 2, 0)).astype(np.intp)
    y0 = np.clip(np.floor(y), 0, max(height - 2, 0)).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = np.clip(x - x0, 0, 1).astype(np.float32)[..., None]
    fy = np.clip(y - y0, 0, 1).astype(np.float32)[..., None]
    top = rgba[y0, x0] * (1 - fx) + rgba[y0, x1] * fx
    bottom = rgba[y1, x0] * (1 - fx) + rgba[y1, x1] * fx
    out = top * (1 - fy) + bottom * fy
    out[~inside] = 0
    return out.astype(np.float32, copy=False)
