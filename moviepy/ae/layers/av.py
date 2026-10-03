"""Footage layer that reads frames from an existing MoviePy clip."""

import math

from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.base import Layer


def _validate_clip(clip):
    """Reject mask clips, unusable frame functions and malformed masks."""
    if getattr(clip, "is_mask", False):
        raise ValueError("clip must be an RGB clip, not a mask clip")
    if not callable(getattr(clip, "get_frame", None)):
        raise TypeError("clip must provide get_frame(t)")
    mask = getattr(clip, "mask", None)
    if mask is not None and not callable(getattr(mask, "get_frame", None)):
        raise TypeError("clip mask must provide get_frame(t)")


def _finite_duration(clip):
    """Return the clip duration in seconds, or None when it is not usable."""
    try:
        duration = float(getattr(clip, "duration", None))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(duration) or duration < 0:
        return None
    return duration


def _clamped_source_time(clip, t):
    """Clamp a source time into the readable range, holding the end frames."""
    time = max(0.0, float(t))
    duration = _finite_duration(clip)
    return time if duration is None else min(time, duration)


class AVLayer(Layer):
    """Wrap any MoviePy ``VideoClip`` as an AE footage layer.

    Parameters
    ----------
    clip : VideoClip
        RGB source. Mask clips are rejected; the clip's own mask, when present,
        becomes the layer alpha through ``Buffer.from_clip``.
    name : str, optional
        Layer name. Defaults to the clip name, else ``AVLayer``.
    ``**kwargs``
        Every ``Layer`` keyword, including
        ``transform``, ``in_point``, ``start_time``, ``stretch`` and ``parent``.

    Notes
    -----
    The default out point is ``start_time + clip.duration * abs(stretch) / 100``,
    so the whole source is reachable and the layer never asks the clip for a
    timestamp past its end. It is recomputed from the current ``in_point``,
    ``start_time`` and ``stretch`` on every access rather than frozen at
    construction, which matches After Effects retiming the layer bar when the
    stretch changes. A clip without a finite duration leaves the layer open
    ended. Source times before the first frame hold frame zero, so an
    ``in_point`` earlier than ``start_time`` cannot hand a negative timestamp to
    a real footage reader.

    Examples
    --------
    >>> from moviepy import ColorClip
    >>> from moviepy.ae.layers import AVLayer
    >>> clip = ColorClip(size=(4, 2), color=(10, 20, 30), duration=1.5)
    >>> layer = AVLayer(clip, start_time=0.5)
    >>> layer.source_size, round(layer.out_point, 6), layer.source_time(1.0)
    ((4, 2), 2.0, 0.5)
    """

    def __init__(self, clip, name=None, **kwargs):
        _validate_clip(clip)
        self.clip = clip
        resolved = name if name is not None else getattr(clip, "name", "") or "AVLayer"
        super().__init__(resolved, **kwargs)

    def default_out_point(self):
        """Return the trimmed source end in composition time."""
        duration = _finite_duration(self.clip)
        if duration is None:
            return math.inf
        return self.start_time + duration * abs(self.stretch) / 100.0

    @property
    def source_size(self):
        """Return the clip pixel size as (width, height)."""
        width, height = self.clip.size
        if width <= 0 or height <= 0:
            raise ValueError("clip size must contain positive pixel counts")
        return (int(width), int(height))

    def source_buffer(self, t, context=None):
        """Import one clip frame and its mask at source-local time ``t``."""
        return Buffer.from_clip(self.clip, _clamped_source_time(self.clip, t))
