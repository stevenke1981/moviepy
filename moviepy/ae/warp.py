"""Affine resampling of premultiplied float32 buffers.

This module owns the pixel-level half of the transform pipeline: filter
selection, destination rectangle derivation and the ``cv2.warpAffine`` call.
The animated transform model lives in ``moviepy.ae.transform``.

Examples
--------
>>> import numpy as np
>>> from moviepy.ae import Buffer
>>> from moviepy.ae.warp import warp_buffer
>>> source = Buffer(np.ones((2, 2, 4), dtype=np.float32))
>>> shift = np.array([[1, 0, 3], [0, 1, -1], [0, 0, 1]], dtype=float)
>>> warp_buffer(source, shift).bounds
(3, -1, 5, 1)
"""

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from moviepy.ae._geometry import (
    DIMENSION_LIMIT,
    applied_factor,
    validate_integers,
    validate_matrix,
    validate_pixel_area,
    validate_pixel_size,
    validate_rectangle,
)
from moviepy.ae.buffer import Buffer, _owned_output


INTERPOLATIONS = ("auto", "nearest", "linear", "cubic")
WARP_INTERPOLATIONS = ("nearest", "linear", "cubic")
_SUPPORT = {"nearest": 0.5, "linear": 1.0, "cubic": 2.0}
# OpenCV remap rejects source and destination axes from SHRT_MAX upward.
_REMAP_LIMIT = 32767
_STRIP_ROWS = 128
_POOL_LOCK = threading.Lock()
_POOL = {"pid": None, "workers": 0, "executor": None}


def resolve_interpolation(interpolation, context=None):
    """Map ``auto`` onto the context quality and reject unknown names.

    Parameters
    ----------
    interpolation : str
        ``auto``, ``nearest``, ``linear`` or ``cubic``.
    context : RenderContext, optional
        Provides ``quality``; ``best`` selects cubic and ``draft`` linear.

    Returns
    -------
    str
        A concrete ``nearest``, ``linear`` or ``cubic`` name.

    Examples
    --------
    >>> from moviepy.ae import RenderContext
    >>> from moviepy.ae.warp import resolve_interpolation
    >>> resolve_interpolation("auto", RenderContext(quality="draft"))
    'linear'
    """
    if not isinstance(interpolation, str) or interpolation not in INTERPOLATIONS:
        raise ValueError("interpolation must be auto, nearest, linear or cubic")
    if interpolation != "auto":
        return interpolation
    quality = "best" if context is None else context.quality
    return "cubic" if quality == "best" else "linear"


def destination_bounds(matrix, size, offset=(0, 0), *, interpolation="linear"):
    """Return the integer world rectangle receiving filtered source energy.

    Parameters
    ----------
    matrix : array_like
        Finite 3x3 source-to-destination affine matrix.
    size : tuple of int
        Positive source (width, height).
    offset : tuple of int, optional
        World coordinates of source pixel [0, 0].
    interpolation : str, optional
        Concrete filter name; its support radius widens the source box.

    Returns
    -------
    tuple of int
        Half-open (left, top, right, bottom). A degenerate matrix collapses to
        a zero-area rectangle instead of raising.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae.warp import destination_bounds
    >>> destination_bounds(np.eye(3), (4, 4))
    (0, 0, 4, 4)
    """
    transform = validate_matrix(matrix)
    width, height = validate_pixel_size(size)
    origin = validate_integers(offset, "offset")
    if interpolation not in WARP_INTERPOLATIONS:
        raise ValueError("interpolation must be nearest, linear or cubic")
    box = _support_box(width, height, origin, _SUPPORT[interpolation])
    corners = _transformed_corners(transform, box)
    rectangle = _snap_rectangle(
        corners[0], corners[1], inclusive=interpolation == "nearest"
    )
    target = (*rectangle[:2], *rectangle[:2]) if _singular(transform) else rectangle
    return validate_rectangle(target)


def _singular(matrix):
    """Recognize a collapsed affine plane without determinant overflow."""
    return np.linalg.slogdet(matrix[:2, :2])[0] == 0


def _support_box(width, height, origin, radius):
    """Widen the source pixel-center box by the filter support radius."""
    return (
        origin[0] - radius,
        origin[1] - radius,
        origin[0] + width - 1 + radius,
        origin[1] + height - 1 + radius,
    )


def _transformed_corners(transform, box):
    """Map the four box corners through the matrix, silencing overflow noise."""
    left, top, right, bottom = box
    with np.errstate(over="ignore", invalid="ignore"):
        return transform @ np.array(
            [
                [left, right, left, right],
                [top, top, bottom, bottom],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )


def _snap_rectangle(xs, ys, *, inclusive=False):
    """Snap a transformed box onto the whole destination pixels it can touch.

    Exact angles reconstructed from trigonometry leave floating-point residue,
    so eight coordinate ulps keep integer boundaries from adding a spare pixel
    without consuming real subpixel support at large world origins.

    ``inclusive`` selects the nearest-neighbour rule. A linear or cubic filter
    weighs a sample landing exactly on the support radius as zero, so a box edge
    on a pixel center excludes that pixel. Nearest samples source pixel
    ``floor(x + 0.5)``, so a destination center whose inverse lands exactly on
    the -0.5 support edge resolves up onto source pixel 0 and carries full
    source energy; those edge pixels must be kept, and excluding them cropped
    two pixels per axis from every integer-magnification upscale. The far
    edge, ``size - 0.5``, rounds up to the out-of-range pixel ``size`` and stays
    empty, which the extra inclusive pixel then simply leaves transparent.
    """
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError("transformed bounds must be finite")
    minimum_x, maximum_x = float(xs.min()), float(xs.max())
    minimum_y, maximum_y = float(ys.min()), float(ys.max())
    # Guard all four extremes: a scale-only matrix can push maximum_x past the
    # limit while minimum_x stays small, and the public destination_bounds API
    # must not hand a future region-of-interest planner an unusable rectangle.
    furthest = max(abs(minimum_x), abs(maximum_x), abs(minimum_y), abs(maximum_y))
    if furthest > DIMENSION_LIMIT:
        raise ValueError("transformed bounds exceed the representable pixel range")
    slack_x = 8 * math.ulp(max(1.0, abs(minimum_x), abs(maximum_x)))
    slack_y = 8 * math.ulp(max(1.0, abs(minimum_y), abs(maximum_y)))
    if inclusive:
        start_x = math.ceil(minimum_x - slack_x)
        start_y = math.ceil(minimum_y - slack_y)
        stop_x = max(math.floor(maximum_x + slack_x) + 1, start_x)
        stop_y = max(math.floor(maximum_y + slack_y) + 1, start_y)
        return (start_x, start_y, stop_x, stop_y)
    start_x = math.floor(minimum_x + slack_x) + 1
    start_y = math.floor(minimum_y + slack_y) + 1
    stop_x = max(math.ceil(maximum_x - slack_x), start_x)
    stop_y = max(math.ceil(maximum_y - slack_y), start_y)
    return (start_x, start_y, stop_x, stop_y)


def warp_buffer(buffer, matrix, opacity=1.0, *, interpolation="linear", bounds=None):
    """Resample a premultiplied buffer through an affine matrix and an opacity.

    Parameters
    ----------
    buffer : Buffer
        Premultiplied source pixels in world coordinates.
    matrix : array_like
        Finite 3x3 source-to-destination affine matrix.
    opacity : float, optional
        Applied factor in 0..1, multiplied into all four premultiplied channels
        and clamped instead of raising, so animated overshoot stays renderable.
    interpolation : {"nearest", "linear", "cubic"}, optional
        Resampling filter. Resolve ``auto`` with ``resolve_interpolation`` first.
    bounds : tuple of int, optional
        Exact destination rectangle for region-of-interest rendering. Omit it to
        derive the tight filtered extent from the matrix.

    Returns
    -------
    Buffer
        Resampled pixels at the destination rectangle. An exact integer
        translation at full opacity returns the source storage without copying.
        Cubic alpha overshoot is clipped to [0, 1]; RGB overshoot is retained
        because buffers may hold signed or HDR values.
    """
    if not isinstance(buffer, Buffer):
        raise TypeError("buffer must be a Buffer")
    transform = validate_matrix(matrix)
    factor = applied_factor(opacity)
    if interpolation not in WARP_INTERPOLATIONS:
        raise ValueError("interpolation must be nearest, linear or cubic")
    if 0 in buffer.size:
        return buffer if bounds is None else _empty_at(buffer, bounds)
    shift = None if bounds is not None else _integer_translation(transform)
    if shift is not None:
        return _translated(buffer, shift, factor)
    target = _target_rectangle(transform, buffer, interpolation, bounds)
    if target[2] <= target[0] or target[3] <= target[1] or _singular(transform):
        return _empty_result(buffer, target[:2])
    if interpolation != "cubic":
        exact = _quarter_turn(buffer, transform, factor, target)
        if exact is not None:
            return exact
    return _resample(buffer, transform, factor, interpolation, target)


def _empty_at(buffer, bounds):
    """Return a zero-pixel buffer at an explicitly requested origin."""
    return _empty_result(buffer, validate_rectangle(bounds)[:2])


def _empty_result(buffer, offset):
    """Return a zero-pixel buffer carrying the source color-space label."""
    array = np.zeros((0, 0, 4), dtype=np.float32)
    return Buffer._publish(array, offset, buffer.color_space, (None, 0.0, True))


def _integer_translation(matrix):
    """Return an exact integer shift for a pure translation, else None."""
    if not (
        matrix[0, 0] == 1.0
        and matrix[1, 1] == 1.0
        and matrix[0, 1] == 0.0
        and matrix[1, 0] == 0.0
        and matrix[2, 0] == 0.0
        and matrix[2, 1] == 0.0
        and matrix[2, 2] == 1.0
    ):
        return None
    shift_x, shift_y = float(matrix[0, 2]), float(matrix[1, 2])
    if shift_x != math.floor(shift_x) or shift_y != math.floor(shift_y):
        return None
    if abs(shift_x) > DIMENSION_LIMIT or abs(shift_y) > DIMENSION_LIMIT:
        return None
    return (int(shift_x), int(shift_y))


def _translated(buffer, shift, factor):
    """Copy or scale pixels without resampling, preserving exact summaries."""
    offset = (buffer.offset[0] + shift[0], buffer.offset[1] + shift[1])
    validate_rectangle(
        (*offset, offset[0] + buffer.size[0], offset[1] + buffer.size[1])
    )
    validate_pixel_area(*buffer.size, "destination bounds")
    if factor == 1.0:
        return buffer.with_offset(offset)
    array = _owned_output(buffer.rgba.shape)
    np.multiply(buffer.rgba, np.float32(factor), out=array)
    # Every channel is multiplied by ``factor``, so a uniform source alpha is
    # now uniformly ``alpha * factor``. Republishing the unscaled summary would
    # let ``_native_over`` weight the background by ``1 - 1`` and let the opaque
    # ``to_uint8_rgb`` fast path skip unpremultiplication. The float32 product
    # reproduces the per-pixel rounding above exactly, and ``applied_factor``
    # already clamps to 0..1 so premultiplication stays valid.
    return Buffer._publish(array, offset, buffer.color_space, _scaled(buffer, factor))


def _scaled(buffer, factor):
    """Return publish metadata for a pixel permutation scaled by ``factor``."""
    uniform = buffer._uniform_alpha
    return (
        None if uniform is None else uniform * np.float32(factor),
        buffer._value_bound * factor,
        buffer._unit_premultiplied,
    )


def _quarter_turn(buffer, transform, factor, target):
    """Rotate or mirror by whole quarter turns with numpy, bit for bit.

    A linear part that is a signed permutation with an integer translation maps
    every source pixel center onto a destination pixel center, so nearest and
    linear sampling both reduce to a pure reordering. OpenCV 5 evaluates those
    coordinates in floating point and can weigh a neighbour at about 1e-6,
    which broke the documented equality with ``numpy.rot90``. The reordering
    is computed on its natural rectangle and then cropped or zero-padded into
    ``target``, so an explicit region-of-interest render matches the
    corresponding crop of the full render exactly. Returns None for any other
    matrix.
    """
    linear = transform[:2, :2]
    if not (np.isin(linear, (-1.0, 0.0, 1.0)).all() and np.abs(linear).sum() == 2):
        return None
    if np.count_nonzero(linear, axis=0).min() == 0:
        return None
    width, height = buffer.size
    left, top = buffer.offset
    corners = transform[:2] @ np.array(
        [[left, left + width - 1], [top, top + height - 1], [1.0, 1.0]]
    )
    origin = corners.min(axis=1)
    if not np.array_equal(origin, np.floor(origin)):
        return None
    source = _pixel_records(buffer.rgba)
    signs = (linear[0, 0], linear[1, 1])
    if linear[0, 0] == 0.0:
        source = source.swapaxes(0, 1)
        signs = (linear[0, 1], linear[1, 0])
    source = source[:: int(signs[1]), :: int(signs[0])]
    natural_x, natural_y = int(origin[0]), int(origin[1])
    # Intersect the natural rectangle with the target, in target coordinates.
    x0, y0 = max(natural_x, target[0]), max(natural_y, target[1])
    x1 = min(natural_x + source.shape[1], target[2])
    y1 = min(natural_y + source.shape[0], target[3])
    array = _owned_output((target[3] - target[1], target[2] - target[0], 4))
    covered = x0 < x1 and y0 < y1
    full = covered and (x0, y0, x1, y1) == tuple(target)
    if not full:
        array.fill(0.0)
    if covered:
        destination = _pixel_records(array)[
            y0 - target[1] : y1 - target[1], x0 - target[0] : x1 - target[0]
        ]
        np.copyto(
            destination,
            source[y0 - natural_y : y1 - natural_y, x0 - natural_x : x1 - natural_x],
        )
        if factor != 1.0:
            array *= np.float32(factor)
    metadata = _scaled(buffer, factor)
    if not full:
        # Zero padding breaks a uniform alpha summary but keeps the bounds.
        metadata = (None, *metadata[1:])
    return Buffer._publish(array, target[:2], buffer.color_space, metadata)


def _pixel_records(pixels):
    """View RGBA pixels as 16-byte records so reordering copies move whole pixels.

    A transposed float32 view strides four scalars per element; copying one
    record per pixel instead runs about 1.6x faster. Falls back to the float
    view when the channel axis is not contiguous.
    """
    if pixels.strides[-1] != pixels.itemsize or pixels.shape[-1] != 4:
        return pixels
    return pixels.view(np.dtype((np.void, 4 * pixels.itemsize)))[..., 0]


def _target_rectangle(transform, buffer, interpolation, bounds):
    """Resolve and range-check the destination rectangle."""
    if bounds is not None:
        target = validate_rectangle(bounds)
    else:
        target = destination_bounds(
            transform, buffer.size, buffer.offset, interpolation=interpolation
        )
    width, height = target[2] - target[0], target[3] - target[1]
    if width > DIMENSION_LIMIT or height > DIMENSION_LIMIT:
        raise ValueError("destination bounds exceed the representable pixel range")
    validate_pixel_area(width, height, "destination bounds")
    return target


def _resample(buffer, transform, factor, interpolation, target):
    """Run the cv2 warp into pooled storage and publish the result."""
    import cv2

    width, height = target[2] - target[0], target[3] - target[1]
    forward = np.array(transform[:2], dtype=np.float64, copy=True)
    # cv2 indexes source pixels locally; the public matrix maps world points.
    forward[:, 2] += transform[:2, :2] @ np.asarray(buffer.offset) - target[:2]
    _validate_inverse_range(forward)
    flags = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }[interpolation]
    array = _owned_output((height, width, 4))
    if interpolation == "cubic":
        _cubic_remap(buffer, transform, target, array)
    elif interpolation == "nearest":
        _nearest_gather(buffer, transform, target, array)
    else:
        cv2.warpAffine(
            buffer.rgba,
            forward,
            (width, height),
            dst=array,
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0, 0.0),
        )
    return _finish(array, buffer, factor, interpolation, target)


def _nearest_gather(buffer, transform, target, array):
    """Copy the nearest source pixel with a fixed round-half-up rule.

    OpenCV 4 rounded nearest coordinates half up through fixed point, while
    OpenCV 5 resolves exact half-pixel ties differently, which widened a
    2-pixel solid at a half-pixel position to 3 pixels. The source index is
    therefore computed in float64 with numpy as ``floor(x + 0.5)``, the one
    rule kept across OpenCV releases, then clipped to ``[-1, size]``. Those
    integer-valued maps are exact in float32, so ``cv2.remap`` with
    ``INTER_NEAREST`` only gathers: no tie remains for it to resolve, and the
    out-of-range ``-1`` and ``size`` indices read the transparent border.
    Axis-aligned matrices build each map from one 1-D index vector; other
    matrices evaluate both coordinates per pixel. Like the cubic path, the
    coordinates are evaluated at world pixel positions, so a half-pixel tie
    resolves identically in a full render and in any region of interest. Rows
    run in strips, spread over as many threads as OpenCV may use.
    """
    import cv2

    forward = np.array(transform[:2], dtype=np.float64, copy=True)
    forward[:, 2] += transform[:2, :2] @ np.asarray(buffer.offset)
    inverse = cv2.invertAffineTransform(forward)
    if max(buffer.size) >= _REMAP_LIMIT or array.shape[1] >= _REMAP_LIMIT:
        _for_strips(
            array.shape[0],
            lambda start: _nearest_numpy(buffer, inverse, target, array, start),
        )
        return
    columns = np.arange(target[0], target[2], dtype=np.float64)
    if inverse[0, 1] == 0.0 and inverse[1, 0] == 0.0:
        # A zero cross term contributes an exact signed zero, so the 1-D sums
        # equal the per-pixel sums of the general branch bit for bit.
        rows = np.arange(target[1], target[3], dtype=np.float64)
        xs = _snap_indices(inverse[0, 0] * columns + inverse[0, 2], buffer.size[0])
        ys = _snap_indices(inverse[1, 1] * rows + inverse[1, 2], buffer.size[1])

        def maps(start, stop):
            map_x = np.empty((stop - start, xs.size), dtype=np.float32)
            map_y = np.empty_like(map_x)
            map_x[...] = xs
            map_y[...] = ys[start:stop, None]
            return map_x, map_y

    else:
        horizontal = (inverse[0, 0] * columns, inverse[1, 0] * columns)

        def maps(start, stop):
            rows = np.arange(target[1] + start, target[1] + stop, dtype=np.float64)
            scratch = np.empty((stop - start, columns.size), dtype=np.float64)
            result = []
            for axis in range(2):
                vertical = inverse[axis, 1] * rows + inverse[axis, 2]
                np.add(horizontal[axis][None, :], vertical[:, None], out=scratch)
                result.append(_snap_indices(scratch, buffer.size[axis]))
            return result

    def strip(start):
        stop = min(start + _STRIP_ROWS, array.shape[0])
        cv2.remap(
            buffer.rgba,
            *maps(start, stop),
            cv2.INTER_NEAREST,
            dst=array[start:stop],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0, 0.0),
        )

    _for_strips(array.shape[0], strip)


def _snap_indices(coordinates, limit):
    """Round mapped coordinates half up and clip them into float32 remap range.

    Modifies ``coordinates`` in place. Clipping to ``[-1, limit]`` keeps every
    out-of-range index out of range while making it exactly representable.
    """
    coordinates += 0.5
    np.floor(coordinates, out=coordinates)
    snapped = np.empty(coordinates.shape, dtype=np.float32)
    np.clip(coordinates, -1.0, float(limit), out=snapped)
    return snapped


def _nearest_numpy(buffer, inverse, target, array, start):
    """Gather one strip with numpy for sources or targets too wide for remap."""
    width, height = buffer.size
    stop = min(start + _STRIP_ROWS, array.shape[0])
    columns = np.arange(target[0], target[2], dtype=np.float64)
    rows = np.arange(target[1] + start, target[1] + stop, dtype=np.float64)
    source = []
    for axis in range(2):
        coordinates = inverse[axis, 0] * columns[None, :]
        coordinates = (
            coordinates + (inverse[axis, 1] * rows + inverse[axis, 2])[:, None]
        )
        coordinates += 0.5
        source.append(np.floor(coordinates, out=coordinates))
    inside = (source[0] >= 0) & (source[0] < width)
    inside &= (source[1] >= 0) & (source[1] < height)
    strip = array[start:stop]
    strip.fill(0.0)
    strip[inside] = buffer.rgba[
        source[1][inside].astype(np.intp), source[0][inside].astype(np.intp)
    ]


def _for_strips(height, work):
    """Run ``work(start)`` for every strip, threaded when OpenCV is threaded.

    Coordinate generation is plain numpy and would otherwise run on one core
    while OpenCV's own warps use them all. The worker count follows
    ``cv2.getNumThreads()``, so render worker processes that pin OpenCV to one
    thread also keep this path single threaded.
    """
    starts = range(0, height, _STRIP_ROWS)
    executor = _strip_executor() if len(starts) > 1 else None
    if executor is None:
        for start in starts:
            work(start)
        return
    for _ in executor.map(work, starts):
        pass


def _strip_executor():
    """Return a process-local thread pool sized to OpenCV, or None for one."""
    import cv2

    workers = min(cv2.getNumThreads(), 8)
    if workers <= 1:
        return None
    with _POOL_LOCK:
        # A forked child inherits the pool object but none of its threads.
        if _POOL["pid"] != os.getpid() or _POOL["workers"] != workers:
            if _POOL["pid"] == os.getpid():
                _POOL["executor"].shutdown(wait=False)
            _POOL.update(
                pid=os.getpid(),
                workers=workers,
                executor=ThreadPoolExecutor(workers, thread_name_prefix="ae-warp"),
            )
        return _POOL["executor"]


def _cubic_remap(buffer, transform, target, array):
    """Sample cubic pixels at fixed world coordinates, independent of the ROI.

    warpAffine rounds coordinates relative to its output origin. Explicit maps
    use the same inverse and arithmetic for each world pixel in every render.
    OpenCV remap requires each source and destination axis to be below 32767.
    """
    import cv2

    forward = np.array(transform[:2], dtype=np.float64, copy=True)
    forward[:, 2] += transform[:2, :2] @ np.asarray(buffer.offset)
    inverse = cv2.invertAffineTransform(forward)
    if max(buffer.size) >= 32767:
        _large_cubic_source(buffer, inverse, target, array)
        return
    for top in range(target[1], target[3], 32766):
        for left in range(target[0], target[2], 32766):
            right, bottom = min(left + 32766, target[2]), min(top + 32766, target[3])
            output = array[
                top - target[1] : bottom - target[1],
                left - target[0] : right - target[0],
            ]
            _remap_rectangle(
                buffer.rgba, inverse, (left, top, right, bottom), (0, 0), output
            )


def _remap_rectangle(pixels, inverse, bounds, source_origin, array):
    """Build only the requested maps, without full-size float64 temporaries."""
    import cv2

    xs = np.arange(bounds[0], bounds[2], dtype=np.float64)
    ys = np.arange(bounds[1], bounds[3], dtype=np.float64)
    maps = []
    for axis in range(2):
        mapping = np.empty(array.shape[:2], dtype=np.float32)
        horizontal = inverse[axis, 0] * xs
        vertical = inverse[axis, 1] * ys + inverse[axis, 2] - source_origin[axis]
        np.add(horizontal[None, :], vertical[:, None], out=mapping, casting="unsafe")
        maps.append(mapping)
    cv2.remap(pixels, *maps, cv2.INTER_CUBIC, dst=array, borderMode=cv2.BORDER_CONSTANT)


def _large_cubic_source(buffer, inverse, target, array):
    """Crop oversized sources on a fixed grid while sampling only the ROI."""
    steps = _cubic_tile_steps(inverse)
    for top in range(target[1] // steps[1] * steps[1], target[3], steps[1]):
        for left in range(target[0] // steps[0] * steps[0], target[2], steps[0]):
            tile = (left, top, left + steps[0], top + steps[1])
            bounds = (
                max(left, target[0]),
                max(top, target[1]),
                min(tile[2], target[2]),
                min(tile[3], target[3]),
            )
            output = array[
                bounds[1] - target[1] : bounds[3] - target[1],
                bounds[0] - target[0] : bounds[2] - target[0],
            ]
            pixels, origin = _cubic_tile_source(buffer, inverse, tile)
            if pixels is None:
                output.fill(0.0)
            else:
                _remap_rectangle(pixels, inverse, bounds, origin, output)


def _cubic_tile_steps(inverse):
    """Limit each axis to half the source span allowed after the cubic halo."""
    steps = []
    for axis in range(2):
        coefficient = float(np.max(np.abs(inverse[:, axis])))
        steps.append(
            4096 if coefficient <= 16380 / 4095 else int(16380 / coefficient) + 1
        )
    return tuple(steps)


def _cubic_tile_source(buffer, inverse, tile):
    """Choose a source crop from the full tile so ROI changes cannot move it."""
    box = (tile[0], tile[1], tile[2] - 1, tile[3] - 1)
    corners = _transformed_corners(inverse, box)
    left = max(0, math.floor(float(corners[0].min())) - 2)
    top = max(0, math.floor(float(corners[1].min())) - 2)
    right = min(buffer.size[0], math.floor(float(corners[0].max())) + 3)
    bottom = min(buffer.size[1], math.floor(float(corners[1].max())) + 3)
    if left >= right or top >= bottom:
        return None, None
    return buffer.rgba[top:bottom, left:right], (left, top)


def _validate_inverse_range(matrix):
    """Reject affine arithmetic outside OpenCV's float64 inverse range."""
    if not np.isfinite(matrix).all():
        raise ValueError("local affine inverse must remain representable in float64")
    a, b, c, d = (float(matrix[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)])
    determinant = a * d - b * c
    if (
        not math.isfinite(determinant)
        or determinant == 0
        or not math.isfinite(1.0 / determinant)
    ):
        raise ValueError("affine inverse exceeds the supported float64 range")
    inverse_axes = (abs(value / determinant) for value in (a, b, c, d))
    if max(inverse_axes) > DIMENSION_LIMIT:
        raise ValueError("affine inverse exceeds the supported pixel coordinate range")


def _finish(array, buffer, factor, interpolation, target):
    """Clamp cubic alpha, apply opacity and publish conservative metadata."""
    headroom = 2.0 if interpolation == "cubic" else 1.0
    if _may_overflow(buffer, factor, headroom) and not np.isfinite(array).all():
        raise ValueError("warped pixels must remain finite in float32")
    if interpolation == "cubic":
        np.clip(array[..., 3], 0.0, 1.0, out=array[..., 3])
    if factor != 1.0:
        array *= np.float32(factor)
    metadata = (
        None,
        buffer._value_bound * factor * headroom,
        buffer._unit_premultiplied and factor <= 1.0 and interpolation != "cubic",
    )
    return Buffer._publish(array, target[:2], buffer.color_space, metadata)


def _may_overflow(buffer, factor, headroom):
    """Decide in constant time whether resampled values could leave float32.

    Nearest and linear are convex combinations and cannot exceed the source
    magnitude; cubic overshoots by less than ``headroom``. Only a source already
    near the float32 ceiling can therefore overflow, which keeps the finiteness
    scan off the hot path for ordinary footage while still refusing to publish
    an infinite buffer that would detonate later inside an export.
    """
    # Opacity is applied after resampling and cannot make an overflow safe.
    projected = buffer._value_bound * headroom
    return not projected <= float(np.finfo(np.float32).max)
