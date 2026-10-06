"""Explicit sRGB transfer functions and linear-sRGB compositing.

``linear`` uses the sRGB/Rec.709 primaries and D65 white. It is not ACES,
an exposure calibration, a tone mapper, or an interpretation of an OCIO label.
Native scene-linear footage may retain HDR values in this working encoding.
Buffer's existing raw-code exports remain unchanged; use these functions
when a color-managed display/export boundary is wanted.
"""

import numpy as np

from moviepy.ae.buffer import Buffer, _arithmetic, _background, unpremultiply


def working_space(value):
    """Validate the two implemented working encodings."""
    if not isinstance(value, str):
        raise TypeError("working_space must be a string")
    if value not in ("srgb", "linear"):
        raise ValueError("working_space must be 'srgb' or 'linear'")
    return value


def _transfer(values, decode):
    """Convert straight channels, retaining signed and extended-range values."""
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError("color values must be real numbers")
    if not np.isfinite(array).all():
        raise ValueError("color values must be finite")
    # Float64 intermediates avoid overflow in the unused branch or before
    # validation. Publication is float32, matching the compositing pipeline.
    magnitude = np.array(array, dtype=np.float64, copy=True)
    np.abs(magnitude, out=magnitude)
    nonlinear = magnitude > (0.04045 if decode else 0.0031308)
    all_nonlinear = bool(np.all(nonlinear))
    high = magnitude if all_nonlinear else magnitude[nonlinear]
    if not all_nonlinear:
        if decode:
            magnitude /= 12.92
        else:
            magnitude *= 12.92
    if decode:
        high += 0.055
        high /= 1.055
        np.power(high, 2.4, out=high)
    else:
        np.power(high, 1 / 2.4, out=high)
        high *= 1.055
        high -= 0.055
    if not all_nonlinear:
        magnitude[nonlinear] = high
    with np.errstate(over="ignore", invalid="ignore"):
        np.copysign(magnitude, array, out=magnitude)
        result = magnitude.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("converted color must remain finite in float32")
    return result


def srgb_to_linear(values):
    """Decode straight sRGB channels; alpha must never enter this function."""
    return _transfer(values, True)


def linear_to_srgb(values):
    """Encode straight linear sRGB channels without clipping or quantization."""
    return _transfer(values, False)


def convert_buffer(buffer, target):
    """Convert straight RGB, then re-premultiply; preserve alpha and offset.

    Fully transparent hidden RGB stays zero, with no opacity epsilon.
    Identical encodings return the original immutable buffer.
    """
    working_space(target)
    if not isinstance(buffer, Buffer):
        raise TypeError("buffer must be a Buffer")
    working_space(buffer.color_space)
    if target == buffer.color_space:
        return buffer
    function = srgb_to_linear if target == "linear" else linear_to_srgb
    source = buffer.rgba
    result = np.zeros_like(source)
    result[..., 3] = source[..., 3]
    alpha = buffer._uniform_alpha
    if alpha == 1:
        result[..., :3] = function(source[..., :3])
    elif alpha is None or alpha > 0:
        covered = source[..., 3] > 0
        pixels = source[covered]
        with _arithmetic("rgba"):
            straight = pixels[:, :3] / pixels[:, 3:4]
        result[covered, :3] = function(straight) * pixels[:, 3:4]
    return Buffer._publish(result, buffer.offset, target)


def display_rgba(buffer, *, background=None):
    """Return straight float32 sRGB with linear alpha, without quantization.

    ``background`` is an optional normalized opaque sRGB triple. Flattening
    occurs in the buffer's working space before the display transfer.
    Values outside SDR [0, 1] are retained for the caller to inspect.
    """
    working_space(buffer.color_space)
    if background is None:
        result = unpremultiply(buffer.rgba)
    else:
        color = _background(background)
        if buffer.color_space == "linear":
            color = srgb_to_linear(color)
        result = np.array(buffer.rgba, copy=True)
        result[..., :3] += color * (1 - result[..., 3:4])
        result[..., 3] = 1
    if buffer.color_space == "linear":
        result[..., :3] = linear_to_srgb(result[..., :3])
    return result


def quantize_rgba(rgba, *, bits=16):
    """Round straight SDR RGBA once to 8 or 16 bits, clipping at this boundary."""
    if isinstance(bits, bool) or bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4 or array.dtype.kind not in "iuf":
        raise ValueError("rgba must be a real (H, W, 4) array")
    if not np.isfinite(array).all():
        raise ValueError("rgba must be finite")
    maximum = (1 << bits) - 1
    return np.rint(np.clip(array, 0, 1) * maximum).astype(
        np.uint8 if bits == 8 else np.uint16
    )
