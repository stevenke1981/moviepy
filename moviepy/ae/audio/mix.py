"""AV-layer audio timing and MoviePy composition mixing."""

import math
from dataclasses import replace

import numpy as np

from moviepy.ae.audio import db_to_amplitude
from moviepy.ae.audio._source import (
    duration_of,
    read_source_audio,
    source_format,
    time_array,
)
from moviepy.ae.layers.av import AVLayer
from moviepy.audio.AudioClip import AudioClip, CompositeAudioClip


def _candidates(comp):
    """Find sound-bearing AV layers independently of their visual switches."""
    result = []
    for layer in comp.layers:
        if not isinstance(layer, AVLayer):
            continue
        if layer.guide and not comp.renderer.include_guides:
            continue
        source = getattr(layer.clip, "audio", None)
        if source is not None:
            result.append((layer, source))
    return result


def _audible(layer, comp):
    """Honor the audio switch and audio-domain solo without visual opacity."""
    if not layer.audio_enabled:
        return False
    candidates = _candidates(comp)
    if not any(item is layer for item, _ in candidates):
        return False
    return layer.solo or not any(item.solo for item, _ in candidates)


def _mapped_times(layer, times, context, audio_duration, fps):
    """Map continuous audio time, including the legacy reverse endpoint."""
    mapped = np.array([layer.media_time(float(t), context) for t in times])
    if (
        layer.stretch < 0
        and layer.time_remap is None
        and audio_duration is not None
        and audio_duration > 0
        and getattr(layer.clip, "duration", None) == audio_duration
    ):
        endpoint = mapped == audio_duration
        # Match the source's finite audio sample grid, never its video fps.
        # A duration derived from count / fps can round to either side of count.
        sample_count = math.ceil(math.nextafter(audio_duration * fps, 0.0))
        last_index = max(0, sample_count - 1)
        last = last_index / fps
        # Scalar readers truncate time * fps; division can round below the index.
        if last * fps < last_index:
            last = math.nextafter(last, math.inf)
        mapped[endpoint] = min(last, math.nextafter(audio_duration, 0.0))
    return mapped


def _gains(layer, times, context):
    """Evaluate dB on the unchanged animation clock, not remapped footage."""
    bindings = layer.expression_bindings
    return np.array(
        [
            db_to_amplitude(
                layer.audio_levels.value_at(
                    layer.source_time(float(t)), context=context, **bindings
                )
            )
            for t in times
        ],
        dtype=np.float64,
    )


class _LayerAudio(AudioClip):
    """Borrow one source while gating and mapping composition-local times."""

    def __init__(self, layer, comp, source, duration):
        self.layer, self.comp, self.source = layer, comp, source
        self.nchannels, fps, self.source_duration = source_format(source)
        super().__init__(duration=duration, fps=fps)

    def frame_function(self, t):
        """Return mono/multichannel samples without changing source ownership."""
        times, scalar = time_array(t)
        output = np.zeros((len(times), self.nchannels), dtype=np.float64)
        layer, comp = self.layer, self.comp
        if not len(times) or not _audible(layer, comp):
            return output[0] if scalar else output
        active = (
            (times >= 0)
            & (times < self.duration)
            & (times >= layer.in_point)
            & (times < layer.out_point)
        )
        positions = np.flatnonzero(active)
        context = replace(comp.context, fps=comp.fps, cache=None, shutter=None)
        mapped = _mapped_times(
            layer, times[positions], context, self.source_duration, self.fps
        )
        valid = mapped >= 0
        if self.source_duration is not None:
            valid &= mapped < self.source_duration
        positions = positions[valid]
        if positions.size:
            samples = read_source_audio(self.source, mapped[valid], self.nchannels)
            with np.errstate(over="raise", invalid="raise"):
                try:
                    output[positions] = (
                        samples * _gains(layer, times[positions], context)[:, None]
                    )
                except FloatingPointError as error:
                    raise ValueError(
                        "audio samples and gain must remain finite"
                    ) from error
        return output[0] if scalar else output


class _CompositionAudio(CompositeAudioClip):
    """Use MoviePy mixing while accepting an empty sample request."""

    def frame_function(self, t):
        """Validate times before delegating the unclipped floating-point sum."""
        times, scalar = time_array(t)
        if not len(times):
            return np.zeros((0, self.nchannels), dtype=np.float64)
        with np.errstate(over="raise", invalid="raise"):
            try:
                return super().frame_function(float(times[0]) if scalar else times)
            except FloatingPointError as error:
                raise ValueError("mixed audio samples must remain finite") from error


def build_audio(comp):
    """Create a fresh mix, or None when no included AV source has audio."""
    duration = duration_of(comp.duration, "composition audio duration")
    clips = [
        _LayerAudio(layer, comp, source, duration)
        for layer, source in _candidates(comp)
    ]
    if not clips:
        return None
    channels = {clip.nchannels for clip in clips if clip.nchannels > 1}
    if len(channels) > 1:
        raise ValueError("audio mixes require matching non-mono channel counts")
    return _CompositionAudio(clips)
