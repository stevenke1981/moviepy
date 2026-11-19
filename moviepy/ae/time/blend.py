"""Source-frame interpolation in premultiplied working-space pixels."""

import math

import numpy as np

from moviepy.ae._geometry import validate_pixel_area, validate_rectangle
from moviepy.ae.buffer import Buffer
from moviepy.ae.color import convert_buffer
from moviepy.ae.properties.values import finite_real


def frame_blending_mode(value):
    """Validate the implemented source interpolation modes explicitly."""
    if not isinstance(value, str):
        raise TypeError("frame_blending must be a string")
    if value == "pixel_motion":
        raise NotImplementedError(
            "pixel_motion frame_blending requires optical flow and is not implemented"
        )
    if value not in ("off", "frame_mix"):
        raise ValueError("frame_blending must be 'off' or 'frame_mix'")
    return value


def source_timing(fps, duration):
    """Validate a frame grid and return its fps and optional last frame time."""
    fps = finite_real(fps, "source fps")
    if fps <= 0 or not math.isfinite(1 / fps):
        raise ValueError("source fps must define a finite positive frame interval")
    if duration is None:
        return fps, None
    duration = finite_real(duration, "source duration")
    if duration <= 0:
        raise ValueError("frame_mix requires a positive source duration")
    count = duration * fps
    if not math.isfinite(count):
        raise ValueError("source duration * fps must be finite")
    last = max(0, math.ceil(count) - 1)
    # Compare canonical frame timestamps, not an epsilon around rounded products.
    if last > 0 and last / fps >= duration:
        last -= 1
    if (last + 1) / fps < duration:
        last += 1
    return fps, min(last / fps, math.nextafter(duration, 0))


def _bracket(t, fps, last):
    """Return neighboring readable timestamps and their interpolation weight."""
    t = max(0.0, finite_real(t, "source time"))
    if last is not None:
        t = min(t, last)
    position = t * fps
    if not math.isfinite(position):
        raise ValueError("source time * fps must be finite")
    index = math.floor(position)
    if index > 0 and index / fps > t:
        index -= 1
    if (index + 1) / fps <= t:
        index += 1
    left = index / fps
    if left == t:
        return left, left, 0.0
    right = (index + 1) / fps
    if last is not None:
        right = min(right, last)
    if not math.isfinite(right) or not left < t < right:
        raise ValueError("source frame interval is not representable at this time")
    return left, right, (t - left) / (right - left)


def _aligned(first, second):
    """Align nonempty source pixels, checking allocation limits first."""
    frames = [frame for frame in (first, second) if 0 not in frame.size]
    if not frames:
        return first, second, first.offset
    bounds = validate_rectangle(
        (
            min(frame.bounds[0] for frame in frames),
            min(frame.bounds[1] for frame in frames),
            max(frame.bounds[2] for frame in frames),
            max(frame.bounds[3] for frame in frames),
        )
    )
    validate_pixel_area(bounds[2] - bounds[0], bounds[3] - bounds[1], "frame_mix")
    return (
        first if first.bounds == bounds else first.expand_to(bounds),
        second if second.bounds == bounds else second.expand_to(bounds),
        bounds[:2],
    )


def _mix(first, second, weight, target):
    """Average premultiplied pixels without SDR clipping or float32 overflow."""
    if first.color_space != target:
        first = convert_buffer(first, target)
    if second.color_space != target:
        second = convert_buffer(second, target)
    first, second, offset = _aligned(first, second)
    if 0 in first.size and 0 in second.size:
        return first
    rgba = first.rgba.astype(np.float64) * (1 - weight)
    rgba += second.rgba.astype(np.float64) * weight
    with np.errstate(over="ignore", invalid="ignore"):
        rgba = rgba.astype(np.float32)
    if not np.isfinite(rgba).all():
        raise ValueError("frame_mix arithmetic must remain finite in float32")
    rgba[rgba[..., 3] == 0, :3] = 0
    return Buffer._publish(rgba, offset, target)


def mix_frames(source_at, t, fps, duration, context=None):
    """Sample neighboring source frames and average them in the working space."""
    fps, last = source_timing(fps, duration)
    left, right, weight = _bracket(t, fps, last)
    first = source_at(left, context)
    target = first.color_space if context is None else context.working_space
    if weight == 0:
        return first if first.color_space == target else convert_buffer(first, target)
    second = source_at(right, context)
    return _mix(first, second, weight, target)
