"""Anti-aliased mask rasterization with feather and expansion.

``rasterize`` turns a ``PathValue`` into float32 coverage on an integer
world rectangle:

1. flatten the Bezier path (``moviepy.ae.masks.path.flatten``);
2. fill it at ``supersample`` x resolution with an exact scanline rule
   (a sample is inside when its pixel center is inside the polygon, even-odd
   rule), restricted to the path's bounding box;
3. optionally grow or shrink the hard shape by ``expansion`` pixels with an
   exact Euclidean distance transform at supersampled resolution;
4. box-average down to coverage (an area estimate of each pixel);
5. optionally blur with a Gaussian of ``sigma = feather / 2`` per axis.

The feather convention matches the specification's acceptance rule
(``feather = F`` gives ``sigma = F / 2``); AE does not publish its kernel, so
the visual softness can differ slightly from AE for the same number.
"""

import math

import numpy as np

from moviepy.ae.masks.path import flatten


_SUPERSAMPLE = 4


def _sigma(feather):
    return max(0.0, float(feather)) / 2.0


def margin(feather, expansion=0.0):
    """Return how far (px) feathering/expansion can move coverage outward."""
    sigma_x, sigma_y = (_sigma(value) for value in feather)
    grow = max(0.0, float(expansion))
    return int(math.ceil(3.0 * max(sigma_x, sigma_y) + grow)) + 1


def rasterize(
    path,
    bounds,
    *,
    feather=(0.0, 0.0),
    expansion=0.0,
    supersample=_SUPERSAMPLE,
):
    """Return float32 coverage in [0, 1] of ``path`` over ``bounds``.

    Parameters
    ----------
    path : PathValue
        Mask path in world (layer) coordinates, pixel-center convention.
    bounds : tuple of int
        Half-open (left, top, right, bottom) output rectangle.
    feather : tuple of float, optional
        Horizontal and vertical feather in pixels (``sigma = F / 2``).
    expansion : float, optional
        Positive values grow the shape, negative values shrink it (px).
    supersample : int, optional
        Linear supersampling factor for anti-aliasing (1 disables it).

    Returns
    -------
    numpy.ndarray
        ``(bottom - top, right - left)`` float32 coverage.
    """
    left, top, right, bottom = bounds
    pad = margin(feather, expansion)
    canvas = (left - pad, top - pad, right + pad, bottom + pad)
    coverage = _fill(path, canvas, float(expansion), int(supersample))
    sigma_x, sigma_y = (_sigma(value) for value in feather)
    if sigma_x > 0 or sigma_y > 0:
        coverage = _blur(coverage, sigma_x, sigma_y)
    return np.ascontiguousarray(
        coverage[pad : pad + bottom - top, pad : pad + right - left]
    )


def _fill(path, canvas, expansion, factor):
    """Fill the path inside ``canvas`` and return 1x float32 coverage."""
    left, top, right, bottom = canvas
    width, height = right - left, bottom - top
    coverage = np.zeros((height, width), dtype=np.float32)
    points = flatten(path, scale=factor)
    box = _box(points, max(0.0, expansion), canvas)
    if box is None:
        return coverage
    bx0, by0, bx1, by1 = box
    local = (points - (bx0, by0) + 0.5) * factor - 0.5
    size = ((bx1 - bx0) * factor, (by1 - by0) * factor)
    if expansion:
        hard = scanline_fill(local, *size)
        block = _downsample(_expand(hard, expansion * factor), factor)
    else:
        block = _span_coverage(
            scanline_spans(local, *size), bx1 - bx0, by1 - by0, factor
        )
    coverage[by0 - top : by1 - top, bx0 - left : bx1 - left] = block
    return coverage


def scanline_spans(points, width, height):
    """Return ``(rows, begin, stop)`` inside spans of a polygon.

    Pixel ``[row, col]`` is centered on ``(col, row)``. Each polygon edge
    covers the rows ``ymin <= row < ymax`` (half-open, so shared vertices are
    counted once); sorted crossings pair up with the even-odd rule, and a span
    holds the columns whose centers satisfy ``x_in <= col < x_out``. The test
    is exact for the flattened polygon, unlike ``cv2.fillPoly`` which also
    paints every pixel the outline touches and so inflates the area by about
    half a pixel along the whole perimeter.
    """
    empty = np.zeros(0, dtype=np.int64)
    if len(points) < 3:
        return empty, empty, empty
    start, end = points, np.roll(points, -1, axis=0)
    low = np.minimum(start[:, 1], end[:, 1])
    high = np.maximum(start[:, 1], end[:, 1])
    first = np.clip(np.ceil(low), 0, height).astype(np.int64)
    counts = np.clip(np.ceil(high), 0, height).astype(np.int64) - first
    if not counts.sum():
        return empty, empty, empty
    edge = np.repeat(np.arange(len(points)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    rows = offsets + first[edge]
    x0, y0 = start[edge, 0], start[edge, 1]
    crossings = x0 + (rows - y0) * (end[edge, 0] - x0) / (end[edge, 1] - y0)
    order = np.lexsort((crossings, rows))
    rows, crossings = rows[order], crossings[order]
    begin = np.clip(np.ceil(crossings[0::2]), 0, width).astype(np.int64)
    stop = np.clip(np.ceil(crossings[1::2]), 0, width).astype(np.int64)
    return rows[0::2], begin, stop


def scanline_fill(points, width, height):
    """Return a uint8 image with 1 where pixel centers lie inside the polygon."""
    rows, begin, stop = scanline_spans(points, width, height)
    difference = np.zeros((height, width + 1), dtype=np.int32)
    np.add.at(difference, (rows, begin), 1)
    np.add.at(difference, (rows, stop), -1)
    return (np.cumsum(difference[:, :width], axis=1) > 0).astype(np.uint8)


def _span_coverage(spans, width, height, factor):
    """Accumulate supersampled spans straight into 1x coverage.

    Summing the ``factor`` sub-rows of each output row before the cumulative
    sum keeps the work proportional to the output height, not ``factor``
    times it.
    """
    rows, begin, stop = spans
    columns = width * factor
    difference = np.zeros((height, columns + 1), dtype=np.int32)
    np.add.at(difference, (rows // factor, begin), 1)
    np.add.at(difference, (rows // factor, stop), -1)
    counts = np.cumsum(difference[:, :columns], axis=1)
    totals = counts.reshape(height, width, factor).sum(axis=2, dtype=np.int32)
    return totals.astype(np.float32) / np.float32(factor * factor)


def _box(points, grow, canvas):
    """Return the integer path bounding box (plus growth) clipped to canvas."""
    low = np.floor(points.min(axis=0) - grow) - 1
    high = np.ceil(points.max(axis=0) + grow) + 2
    x0, y0 = max(int(low[0]), canvas[0]), max(int(low[1]), canvas[1])
    x1, y1 = min(int(high[0]), canvas[2]), min(int(high[1]), canvas[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _expand(hard, distance):
    """Grow (positive) or shrink (negative) a binary shape by ``distance`` px."""
    import cv2

    if distance > 0:
        outside = cv2.distanceTransform(
            (hard == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        # Growth is measured from the boundary (half a sample beyond the last
        # inside center); the extra half sample keeps diagonal corners round.
        return (outside <= distance + 0.5).astype(np.uint8)
    inside = cv2.distanceTransform(hard, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (inside > -distance).astype(np.uint8)


def _downsample(hard, factor):
    """Average ``factor x factor`` blocks into coverage."""
    if factor == 1:
        return hard.astype(np.float32)
    height, width = hard.shape[0] // factor, hard.shape[1] // factor
    blocks = hard.reshape(height, factor, width, factor).astype(np.float32)
    return blocks.mean(axis=(1, 3), dtype=np.float32)


def _blur(coverage, sigma_x, sigma_y):
    """Gaussian blur with independent axes; zero outside the canvas."""
    import cv2

    def size(sigma):
        return 1 if sigma <= 0 else 2 * int(math.ceil(3.0 * sigma)) + 1

    return cv2.GaussianBlur(
        coverage,
        (size(sigma_x), size(sigma_y)),
        sigmaX=max(sigma_x, 1e-6),
        sigmaY=max(sigma_y, 1e-6),
        borderType=cv2.BORDER_CONSTANT,
    )
