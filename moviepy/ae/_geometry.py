"""Private validation primitives shared by transforms, warps and layers."""

from numbers import Integral, Real

import numpy as np

from moviepy.ae.properties.property import Property
from moviepy.ae.properties.values import finite_real


DIMENSION_LIMIT = 2**31 - 1
BOUND_EPSILON = 1e-9

# Upper bound on one allocated destination or source rectangle, in pixels.
# Per-axis checks alone admit products near 4.6e18 pixels, so an untrusted
# project file could request hundreds of gigabytes before any limit fires.
# 2**26 pixels is 1 GiB of float32 RGBA and covers compositions past 8K;
# a host application that really needs more may raise this constant.
MAX_RENDER_PIXELS = 2**26


def validate_flag(value, name):
    """Return a validated bool, rejecting truthy substitutes."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")
    return value


def validate_pair(value, name):
    """Return two finite floats from a coordinate-like sequence."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (finite_real(value[0], name), finite_real(value[1], name))
    raise ValueError(f"{name} must contain two finite numbers")


def validate_integers(value, name):
    """Return two plain integers, rejecting bool and fractions."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain two integers")
    items = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise TypeError(f"{name} must contain integers, excluding bool")
        if abs(int(item)) > DIMENSION_LIMIT:
            raise ValueError(f"{name} exceeds the representable pixel range")
        items.append(int(item))
    return tuple(items)


def validate_pixel_size(value):
    """Validate a positive integer (width, height) pair within the area budget."""
    width, height = validate_integers(value, "size")
    if width <= 0 or height <= 0:
        raise ValueError("size must contain positive pixel counts")
    return validate_pixel_area(width, height, "size")


def validate_pixel_area(width, height, name):
    """Return a validated (width, height) pair whose product stays allocatable.

    Raises
    ------
    ValueError
        If ``width * height`` exceeds ``MAX_RENDER_PIXELS``, so an untrusted
        rectangle is rejected before any allocation instead of raising
        ``MemoryError`` from deep inside NumPy or OpenCV.
    """
    if width * height > MAX_RENDER_PIXELS:
        raise ValueError(
            f"{name} of {width}x{height} exceeds the {MAX_RENDER_PIXELS}-pixel "
            "render budget"
        )
    return (width, height)


def validate_extent_size(value):
    """Validate a nonnegative (width, height) pair so empty sources resolve."""
    width, height = validate_integers(value, "size")
    if width < 0 or height < 0:
        raise ValueError("size must not be negative")
    return (width, height)


def validate_rectangle(value):
    """Validate an explicit half-open integer rectangle."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bounds must contain four integers")
    left, top = validate_integers(value[:2], "bounds")
    right, bottom = validate_integers(value[2:], "bounds")
    if right < left or bottom < top:
        raise ValueError("bounds must not be inverted")
    return (left, top, right, bottom)


def to_property(value, name, kind, limits=None):
    """Coerce a value, keyframe list, callable or expression into a Property.

    Parameters
    ----------
    value : object
        An existing ``Property`` of the requested kind, or any value the
        property constructor accepts.
    name : str
        Field name used in error messages.
    kind : str
        Required value type such as ``float`` or ``vec2``.
    limits : tuple of float, optional
        Inclusive range checked only for a bare real number, because animated
        sources are clamped when they are evaluated.

    Returns
    -------
    Property
        The supplied property, or a new one wrapping ``value``.
    """
    if isinstance(value, Property):
        if value.value_type != kind:
            raise ValueError(f"{name} requires a {kind} Property")
        return value
    if limits is not None and isinstance(value, Real) and not isinstance(value, bool):
        number = finite_real(value, name)
        if not limits[0] <= number <= limits[1]:
            raise ValueError(f"{name} must be within {limits[0]}..{limits[1]}")
    return Property(value, value_type=kind)


def validate_matrix(value):
    """Return a finite 3x3 float64 affine matrix that aliases no caller storage.

    Raises
    ------
    ValueError
        If the input is not a 3x3 array of finite numbers, including Python
        integers too large to represent as float64.
    """
    try:
        # copy=True keeps a float64 caller array from being handed back by
        # reference, so a validated matrix can never be mutated behind our back.
        array = np.array(value, dtype=np.float64, copy=True)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("matrix must be a 3x3 affine matrix") from error
    if array.shape != (3, 3):
        raise ValueError("matrix must be a 3x3 affine matrix")
    if not np.isfinite(array).all():
        raise ValueError("matrix must contain only finite values")
    if not np.array_equal(array[2], (0.0, 0.0, 1.0)):
        raise ValueError("matrix must be affine with last row (0, 0, 1)")
    return array


def opacity_factor(value):
    """Convert an opacity percentage into a clamped 0..1 factor."""
    return min(1.0, max(0.0, finite_real(value, "opacity") / 100.0))


def applied_factor(value):
    """Clamp an already normalized factor, never raising on overshoot."""
    return min(1.0, max(0.0, finite_real(value, "opacity")))
