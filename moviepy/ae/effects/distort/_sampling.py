"""Shared inverse-mapping sampler for the Distort effects (OpenCV remap)."""

import cv2
import numpy as np


def sample_bilinear(rgba, xs, ys):
    """Bilinear, zero-filled premultiplied lookup at pixel-center coordinates.

    ``xs``/``ys`` are source positions in pixel-edge coordinates (the center
    of pixel ``i`` is ``i + 0.5``). A sample is zero unless its pixel-index
    position lies inside ``[0, n - 1]`` on both axes; inside, the four
    neighbours are blended bilinearly by ``cv2.remap``. This matches
    ``scipy.ndimage.map_coordinates(order=1, mode="constant", cval=0)``
    to float32 rounding without requiring SciPy, about 50x faster than a
    NumPy gather at 1080p.

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
    # Replicated borders keep in-range samples next to an edge exact; points
    # outside are zeroed below, matching SciPy's "constant" (no blending
    # with the border value), unlike cv2.BORDER_CONSTANT.
    out = cv2.remap(
        np.ascontiguousarray(rgba, dtype=np.float32),
        x.astype(np.float32),
        y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if out.ndim == 2:
        out = out[..., None]
    out[~inside] = 0
    return out
