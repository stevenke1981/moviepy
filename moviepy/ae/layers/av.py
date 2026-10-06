"""Footage layer that reads frames from an existing MoviePy clip."""

import math

from moviepy.ae._geometry import validate_flag
from moviepy.ae.audio import levels_property
from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.base import Layer
from moviepy.ae.time.blend import frame_blending_mode, mix_frames, source_timing
from moviepy.ae.time.remap import remap_property, remap_time


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
    if duration is None or time < duration or duration == 0:
        return time if duration is None else min(time, duration)
    fps = getattr(clip, "fps", None)
    if isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0:
        final_frame = max(0, math.ceil(duration * fps) - 1)
        return min(final_frame / fps, math.nextafter(duration, 0.0))
    return math.nextafter(duration, 0.0)


class AVLayer(Layer):
    """Wrap any MoviePy ``VideoClip`` as an AE footage layer.

    Parameters
    ----------
    clip : VideoClip
        RGB source. Mask clips are rejected; the clip's own mask, when present,
        becomes the layer alpha through ``Buffer.from_clip``.
    name : str, optional
        Layer name. Defaults to the clip name, else ``AVLayer``.
    time_remap : float, Property, callable or None, optional
        Source seconds evaluated on the layer's existing animation clock.
        ``None`` keeps the original timing. Remapping changes footage RGB and
        its clip mask; layer transforms, masks and effects keep their clocks.
    frame_blending : {"off", "frame_mix"}, optional
        ``off`` preserves original sampling. ``frame_mix`` averages neighboring
        source frames using the clip's finite positive ``fps``. Interpolation
        uses premultiplied RGBA in the render working space and preserves HDR.
        ``pixel_motion`` raises ``NotImplementedError``; optical flow is not
        implemented.
    audio_enabled : bool, optional
        Audio switch, independent of the video switch and opacity. Defaults
        to true; source clips without audio remain silent.
    audio_levels : float, Property or callable, optional
        Scalar decibels on the layer animation clock. Zero dB is unity; all
        channels share the gain. Use ``audio_enabled=False`` for exact mute.
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
    ended during forward playback. Negative stretch starts at the source end
    and uses ``duration + (t - start_time) / (stretch / 100)`` for footage and
    animated properties; it requires a finite source duration. The final frame
    is sampled inside the readable half-open source interval. Source times
    before the first frame hold frame zero, so an
    ``in_point`` earlier than ``start_time`` cannot hand a negative timestamp to
    a real footage reader.

    Time remapping does not extend the layer window automatically. Use an
    explicit ``out_point`` for a longer freeze. ``source_buffer(t)`` remains a
    raw source-time read; ``sampled_source(t)`` applies the footage controls at
    layer-local time. ``media_time(t)`` maps composition time to continuous
    source time for audio without video frame snapping or endpoint clamping.

    Examples
    --------
    >>> from moviepy import ColorClip
    >>> from moviepy.ae.layers import AVLayer
    >>> clip = ColorClip(size=(4, 2), color=(10, 20, 30), duration=1.5)
    >>> layer = AVLayer(clip, start_time=0.5)
    >>> layer.source_size, round(layer.out_point, 6), layer.source_time(1.0)
    ((4, 2), 2.0, 0.5)
    """

    def __init__(
        self,
        clip,
        name=None,
        *,
        time_remap=None,
        frame_blending="off",
        audio_enabled=True,
        audio_levels=0.0,
        **kwargs,
    ):
        _validate_clip(clip)
        self.clip = clip
        self.time_remap = time_remap
        self.frame_blending = frame_blending
        self.audio_enabled = audio_enabled
        self.audio_levels = audio_levels
        resolved = name if name is not None else getattr(clip, "name", "") or "AVLayer"
        super().__init__(resolved, **kwargs)

    @property
    def audio_enabled(self):
        """Return the audio switch, independent of the video switch."""
        return self._audio_enabled

    @audio_enabled.setter
    def audio_enabled(self, value):
        self._audio_enabled = validate_flag(value, "audio_enabled")

    @property
    def audio_levels(self):
        """Return the floating-point dB property on the layer animation clock."""
        return self._audio_levels

    @audio_levels.setter
    def audio_levels(self, value):
        self._audio_levels = levels_property(value)

    @property
    def time_remap(self):
        """Return the source-time Property, or None for original timing."""
        return self._time_remap

    @time_remap.setter
    def time_remap(self, value):
        self._time_remap = remap_property(value)

    @property
    def frame_blending(self):
        """Return the source interpolation mode: off or frame_mix."""
        return self._frame_blending

    @frame_blending.setter
    def frame_blending(self, value):
        mode = frame_blending_mode(value)
        if mode == "frame_mix":
            source_timing(getattr(self.clip, "fps", None), _finite_duration(self.clip))
        self._frame_blending = mode

    def footage_time(self, local_t, context=None):
        """Map layer-local seconds to source seconds without clamping or snapping.

        ``local_t`` is the unchanged clock used by transform and effect keys.
        The result can lie outside the source duration; the video reader holds
        endpoints, while an audio adapter can apply its own valid interval.
        """
        return remap_time(
            self._time_remap, local_t, context, bindings=self.expression_bindings
        )

    def media_time(self, t, context=None):
        """Map composition seconds through stretch and remap to source seconds."""
        return self.footage_time(self.source_time(t), context)

    def sampled_source(self, local_t, context=None):
        """Read remapped footage while leaving the layer-property clock intact."""
        time = (
            local_t if self._time_remap is None else self.footage_time(local_t, context)
        )
        if self._frame_blending == "off":
            return self.source_buffer(time, context)
        return mix_frames(
            self.source_buffer,
            time,
            getattr(self.clip, "fps", None),
            _finite_duration(self.clip),
            context,
        )

    def default_out_point(self):
        """Return the trimmed source end in composition time."""
        duration = _finite_duration(self.clip)
        if duration is None:
            return math.inf
        return self.start_time + duration * abs(self.stretch) / 100.0

    def source_time(self, t):
        """Map reverse playback from the finite source end on the layer clock."""
        time = super().source_time(t)
        if self.stretch >= 0:
            return time
        duration = _finite_duration(self.clip)
        if duration is None:
            raise ValueError("reverse footage requires a finite source duration")
        return duration + time

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
