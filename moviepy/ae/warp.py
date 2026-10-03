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

import numpy as np

from moviepy.ae._geometry import (
    BOUND_EPSILON,
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
    return _snap_rectangle(corners[0], corners[1], inclusive=interpolation == "nearest")


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

    Exact angles reconstructed from trigonometry leave about 1e-16 of residue,
    so a relative slack keeps integer boundaries from adding a spare pixel.

    ``inclusive`` selects the nearest-neighbour rule. A linear or cubic filter
    weighs a sample landing exactly on the support radius as zero, so a box edge
    on a pixel center excludes that pixel. Nearest resolves the same tie inward
    through round-half-even, so those edge pixels do carry full source energy
    and must be kept; excluding them cropped two pixels per axis from every
    integer-magnification upscale.
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
    slack_x = BOUND_EPSILON * max(1.0, abs(minimum_x), abs(maximum_x))
    slack_y = BOUND_EPSILON * max(1.0, abs(minimum_y), abs(maximum_y))
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
    if target[2] <= target[0] or target[3] <= target[1]:
        return _empty_result(buffer, target[:2])
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
    uniform = buffer._uniform_alpha
    metadata = (
        None if uniform is None else uniform * np.float32(factor),
        buffer._value_bound * factor,
        buffer._unit_premultiplied,
    )
    return Buffer._publish(array, offset, buffer.color_space, metadata)


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
    forward[0, 2] -= target[0]
    forward[1, 2] -= target[1]
    flags = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }[interpolation]
    array = _owned_output((height, width, 4))
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


def _finish(array, buffer, factor, interpolation, target):
    """Clamp cubic alpha, apply opacity and publish conservative metadata."""
    if interpolation == "cubic":
        np.clip(array[..., 3], 0.0, 1.0, out=array[..., 3])
    if factor != 1.0:
        array *= np.float32(factor)
    headroom = 2.0 if interpolation == "cubic" else 1.0
    if _may_overflow(buffer, factor, headroom) and not np.isfinite(array).all():
        raise ValueError("warped pixels must remain finite in float32")
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
    projected = buffer._value_bound * factor * headroom
    return not projected <= float(np.finfo(np.float32).max)
