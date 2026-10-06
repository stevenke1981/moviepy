"""Deterministic scalar sampling of borrowed MoviePy audio sources."""

import math
from numbers import Integral

import numpy as np

from moviepy.ae.properties.values import finite_real


def duration_of(value, name, *, allow_none=False):
    """Validate a finite nonnegative duration, optionally allowing open audio."""
    if value is None and allow_none:
        return None
    value = finite_real(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def source_format(source):
    """Return validated channel count, sampling rate and optional duration."""
    channels = getattr(source, "nchannels", None)
    if isinstance(channels, bool) or not isinstance(channels, Integral):
        raise TypeError("source audio nchannels must be an integer")
    if channels <= 0:
        raise ValueError("source audio nchannels must be positive")
    fps = getattr(source, "fps", None)
    fps = 44100.0 if fps is None else finite_real(fps, "source audio fps")
    if fps <= 0:
        raise ValueError("source audio fps must be positive")
    duration = duration_of(
        getattr(source, "duration", None), "source audio duration", allow_none=True
    )
    if duration is not None and not math.isfinite(duration * fps):
        raise ValueError("source audio duration * fps must remain finite")
    if not callable(getattr(source, "get_frame", None)):
        raise TypeError("source audio must provide get_frame(t)")
    return int(channels), fps, duration


def time_array(t):
    """Return a finite one-dimensional timestamp array and its scalar flag."""
    times = np.asarray(t)
    if times.dtype.kind not in "fiu":
        raise TypeError("audio time must contain real numbers, not bool")
    if times.ndim > 1:
        raise ValueError("audio time must be scalar or one-dimensional")
    scalar = times.ndim == 0
    times = np.atleast_1d(times).astype(np.float64, copy=False)
    if not np.isfinite(times).all():
        raise ValueError("audio time must be finite")
    return times, scalar


def read_source_audio(source, times, channels):
    """Read sorted unique legal times as scalars and restore requested order.

    File-audio ndarray reads assume ascending, narrowly spaced timestamps and
    use a different sample rounding rule from scalar reads. Scalar get_frame
    also retains effects and time transforms wrapped around a borrowed reader.
    No persistent sample cache or source-video frame-rate quantization is used.
    """
    unique, inverse = np.unique(times, return_inverse=True)
    samples = np.empty((len(unique), channels), dtype=np.float64)
    for index, time in enumerate(unique):
        sample = np.asarray(source.get_frame(float(time)))
        if sample.dtype.kind not in "fiu" or sample.shape not in ((channels,), ()):
            raise ValueError("source audio frame must have one value per channel")
        if sample.ndim == 0 and channels != 1:
            raise ValueError("source audio scalar frame requires one channel")
        if not np.isfinite(sample).all():
            raise ValueError("source audio samples must be finite")
        samples[index] = sample
    return samples[inverse]
