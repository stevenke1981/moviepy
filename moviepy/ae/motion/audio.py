"""Analyze PCM volume onsets for offline, frame-quantized particle triggers."""

import hashlib
import math
import wave
from pathlib import Path

import numpy as np

from moviepy.ae.motion.spec import integer
from moviepy.ae.three_d.scene import number


def analyze_audio(filename, *, fps, frame_count, threshold=0.32, min_interval=0.18):
    """Extract volume onsets from mono/stereo 16-bit PCM WAV without playback.

    This is an energy-onset detector, not a musical beat-grid or tempo model.
    Events use the first video frame at or after the detected onset. Supply a
    hand-edited event list when the desired accents differ from energy peaks.
    """
    fps = integer(fps, "fps", 1, 120)
    frame_count = integer(frame_count, "frame_count", 1, 100000)
    if frame_count / fps > 3600:
        raise ValueError("audio analysis is limited to one hour")
    threshold = number(threshold, "threshold", 0.001, 1)
    min_interval = number(min_interval, "min_interval", 0.01, 10)
    path = Path(filename).expanduser().resolve()
    with wave.open(str(path), "rb") as source:
        rate, channels = source.getframerate(), source.getnchannels()
        if source.getsampwidth() != 2 or channels not in (1, 2) or rate <= 0:
            raise ValueError("audio analysis requires mono/stereo 16-bit PCM WAV")
        count = min(source.getnframes(), math.ceil(frame_count / fps * rate))
        hop = max(1, round(rate * 0.01))
        blocks = max(1, math.ceil(count / hop))
        rms = np.zeros(blocks, np.float64)
        offset, remaining = 0, count
        while remaining:
            take = min(remaining, hop * 256)
            payload = source.readframes(take)
            if len(payload) != take * channels * 2:
                raise ValueError("audio payload is truncated")
            samples = (
                np.frombuffer(payload, "<i2").reshape(-1, channels).astype(np.float64)
                / 32768
            )
            chunk_blocks = math.ceil(take / hop)
            energy = np.zeros(chunk_blocks * hop, np.float64)
            energy[:take] = np.mean(samples * samples, axis=1)
            rms[offset : offset + chunk_blocks] = np.sqrt(
                energy.reshape(chunk_blocks, hop).mean(axis=1)
            )
            remaining -= take
            offset += chunk_blocks
    history = np.concatenate(([0], np.cumsum(rms)))
    indices = np.arange(blocks)
    start = np.maximum(0, indices - 8)
    baseline = (history[indices] - history[start]) / np.maximum(1, indices - start)
    novelty = np.maximum(0, rms - baseline)
    peak = float(novelty.max())
    candidates = []
    if peak > 1e-5:
        for index, value in enumerate(novelty):
            before = novelty[index - 1] if index else -1
            after = novelty[index + 1] if index + 1 < blocks else -1
            if value >= threshold * peak and value > before and value >= after:
                candidates.append(index)
    selected = []
    for index in sorted(candidates, key=lambda i: (-novelty[i], i)):
        if all(abs(index - other) * hop / rate >= min_interval for other in selected):
            selected.append(index)
    events = {}
    for index in sorted(selected):
        frame = math.ceil(index * hop * fps / rate - 1e-9)
        if frame < frame_count:
            strength = float(rms[index] / max(float(rms.max()), 1e-12))
            events[frame] = max(events.get(frame, 0), strength)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "method": "10ms RMS rise above preceding 80ms mean",
        "source_sha256": digest.hexdigest(),
        "sample_rate": rate,
        "threshold": threshold,
        "min_interval": min_interval,
        "events": [
            {"frame": frame, "strength": strength}
            for frame, strength in sorted(events.items())
        ],
        "envelope": [float(value) for value in rms],
        "envelope_hop_seconds": hop / rate,
    }
