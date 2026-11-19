"""Real-audio spectrum bars for the music channel (port of ``s13-spectrum``).

``analyze_spectrum`` turns an audio file into a ``(frames, bands)`` float32
level table (log-spaced FFT bands, stereo power average, attack/release
smoothing) by decoding the file once, sequentially, in blocks.
``SpectrumLayer`` draws that table as vertical bars in an AE layer.

Examples
--------
>>> import numpy as np
>>> from moviepy.ae.templates.spectrum import SpectrumLayer
>>> layer = SpectrumLayer(np.ones((2, 64), np.float32), size=(1088, 140))
>>> layer.source_buffer(0.0).rgba.shape
(140, 1088, 4)
"""

import colorsys
import os
from pathlib import Path

import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.layers.base import Layer
from moviepy.ae.templates._audio_io import audio_info, iter_audio
from moviepy.ae.transform import Transform


__all__ = ["analyze_spectrum", "save_levels", "load_levels", "SpectrumLayer"]

_BATCH = 32


def _band_slices(fft, rate, bands, fmin, fmax):
    edges = np.geomspace(fmin, fmax, bands + 1)
    freq = np.fft.rfftfreq(fft, 1 / rate)
    return [
        np.flatnonzero((freq >= a) & (freq < b)) for a, b in zip(edges[:-1], edges[1:])
    ]


def _batch_levels(windows, window, slices, floor_db, range_db):
    """Map ``(n, fft, channels)`` windows to ``(n, bands)`` float32 levels."""
    spectrum = np.fft.rfft(windows * window[None, :, None], axis=1)
    power = np.mean(np.abs(spectrum) ** 2, axis=2)
    amps = np.zeros((len(windows), len(slices)))
    for band, index in enumerate(slices):
        if len(index):
            amps[:, band] = np.sqrt(power[:, index].max(axis=1))
    amps = amps * 2 / window.sum()
    db = 20 * np.log10(np.maximum(amps, 1e-10))
    return np.clip((db - floor_db) / range_db, 0, 1).astype(np.float32)


def analyze_spectrum(
    audio_path,
    *,
    fps=24,
    start=0.0,
    seconds=None,
    bands=64,
    fft=8192,
    fmin=45.0,
    fmax=10000.0,
    floor_db=-65.0,
    range_db=53.0,
    attack=0.55,
    release=0.14,
    rate=48000,
):
    """Return smoothed spectrum levels, one row per video frame.

    Frame ``i`` is analysed with a Hann window of ``fft`` samples centred on
    sample ``round(i * rate / fps)`` (zero padded before the start and after
    the end of the file). Stereo channels are averaged in *power*, so
    antiphase material is not cancelled. The band level is the loudest bin in
    the band, mapped from ``floor_db`` over ``range_db`` decibels to 0..1 and
    smoothed with ``attack`` when rising and ``release`` when falling. The
    file is decoded sequentially through FFmpeg with a sliding buffer.

    Parameters
    ----------
    audio_path : str or Path
        Any audio file FFmpeg can decode (resampled to ``rate``).
    fps : float, optional
        Video frame rate of the level table.
    start : float, optional
        First frame time in seconds (smoothing starts from silence there).
    seconds : float, optional
        Length in seconds; ``None`` analyses to the end of the file.
    bands : int, optional
        Number of log-spaced bands between ``fmin`` and ``fmax``.
    fft : int, optional
        Window and FFT length in samples.
    fmin, fmax : float, optional
        Frequency range in Hz.
    floor_db, range_db : float, optional
        A band at ``floor_db`` maps to 0 and at ``floor_db + range_db`` to 1.
    attack, release : float, optional
        Smoothing weights when a level rises / falls.
    rate : int, optional
        Analysis sample rate.

    Returns
    -------
    numpy.ndarray
        float32 array of shape ``(frames, bands)``.

    Examples
    --------
    >>> import numpy as np, tempfile, os
    >>> from moviepy.ae.templates._audio_io import AudioWriter
    >>> path = os.path.join(tempfile.mkdtemp(), "silence.wav")
    >>> with AudioWriter(path, rate=48000) as writer:
    ...     writer.write(np.zeros((48000, 2), np.float32))
    >>> analyze_spectrum(path, seconds=0.5).shape
    (12, 64)
    """
    if fps <= 0 or bands < 1 or fft < 2 or rate < 1:
        raise ValueError("fps, bands, fft and rate must be positive")
    if start < 0:
        raise ValueError("start must not be negative")
    first = int(round(start * fps))
    info = audio_info(audio_path)
    length = info["frames"] / info["rate"]
    if seconds is None:
        count = int(round(max(0.0, length - first / fps) * fps))
    else:
        count = int(round(seconds * fps))
        if length + 1e-9 < start + seconds:
            raise ValueError("audio does not cover the requested interval")
    slices = _band_slices(fft, rate, bands, fmin, fmax)
    window = np.hanning(fft).astype(np.float32)
    out = np.empty((count, bands), np.float32)
    if count == 0:
        return out
    half = fft // 2
    read_from = max(0, int(round(first * rate / fps)) - half)
    blocks = iter_audio(audio_path, rate=rate, channels=2, start=read_from, block=rate)
    buffer = np.zeros((0, 2), np.float32)
    buffer_start = read_from
    done = False
    previous = np.zeros(bands, np.float32)
    windows = np.zeros((_BATCH, fft, 2), np.float32)
    pending = 0
    row = 0

    def flush():
        nonlocal previous, row, pending
        levels = _batch_levels(windows[:pending], window, slices, floor_db, range_db)
        for current in levels:
            weight = np.where(current > previous, attack, release)
            previous = previous + weight * (current - previous)
            out[row] = previous
            row += 1
        pending = 0

    try:
        for i in range(count):
            begin = int(round((first + i) * rate / fps)) - half
            end = begin + fft
            while not done and buffer_start + len(buffer) < end:
                try:
                    buffer = np.concatenate([buffer, next(blocks)])
                except StopIteration:
                    done = True
            drop = min(len(buffer), max(0, begin - buffer_start))
            if drop:
                buffer = buffer[drop:]
                buffer_start += drop
            lo = max(begin, buffer_start)
            hi = min(end, buffer_start + len(buffer))
            windows[pending] = 0
            if hi > lo:
                windows[pending, lo - begin : hi - begin] = buffer[
                    lo - buffer_start : hi - buffer_start
                ]
            pending += 1
            if pending == _BATCH:
                flush()
        if pending:
            flush()
    finally:
        blocks.close()
    return out


def save_levels(path, levels):
    """Write a level table as ``.npy``; never replaces an existing file.

    Parameters
    ----------
    path : str or Path
        Destination; ``FileExistsError`` if it exists.
    levels : numpy.ndarray
        Table from ``analyze_spectrum``.

    Examples
    --------
    >>> import numpy as np, tempfile, os
    >>> path = os.path.join(tempfile.mkdtemp(), "levels.npy")
    >>> save_levels(path, np.zeros((2, 3), np.float32))
    >>> load_levels(path).shape
    (2, 3)
    """
    levels = np.ascontiguousarray(levels, dtype=np.float32)
    if levels.ndim != 2:
        raise ValueError("levels must be 2-D (frames, bands)")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as stream:
        np.save(stream, levels, allow_pickle=False)


def load_levels(path):
    """Read a level table written by ``save_levels`` (no pickles).

    Parameters
    ----------
    path : str or Path
        A ``.npy`` file.

    Returns
    -------
    numpy.ndarray
        float32 array of shape ``(frames, bands)``.
    """
    levels = np.load(os.fspath(path), allow_pickle=False)
    if levels.ndim != 2:
        raise ValueError("levels must be 2-D (frames, bands)")
    return levels.astype(np.float32, copy=False)


def _palette(bands, name):
    if name != "rainbow":
        raise ValueError("unknown palette %r (only 'rainbow')" % (name,))
    span = max(1, bands - 1)
    colors = [
        colorsys.hsv_to_rgb(0.02 + 0.73 * i / span, 0.72, 1) for i in range(bands)
    ]
    return np.rint(np.array(colors) * 255).astype(np.uint8)


class SpectrumLayer(Layer):
    r"""Vertical audio bars as an AE layer, one level row per video frame.

    Parameters
    ----------
    levels : numpy.ndarray
        ``(frames, bands)`` values in 0..1 (see ``analyze_spectrum``).
    fps : float, optional
        Frame rate of ``levels``; the row shown at ``t`` is
        ``round(t * fps)`` clamped to the table.
    size : tuple of int, optional
        Layer pixel size ``(width, height)``.
    bar_width, pitch : int, optional
        Bar width and distance between bar starts in pixels.
    margin : tuple of int, optional
        ``(left, bottom)`` margin; the tallest bar is ``height - 2 * bottom``.
    palette : str, optional
        ``"rainbow"``: HSV hue 0.02 to 0.75, saturation 0.72, value 1.
    opacity : float, optional
        Layer opacity in percent (the source composites at 30).
    name : str, optional
        Layer name.
    \*\*kwargs
        Passed to ``Layer`` (an explicit ``transform`` overrides ``opacity``).

    Examples
    --------
    >>> import numpy as np
    >>> layer = SpectrumLayer(np.zeros((1, 4), np.float32), size=(80, 20))
    >>> int(layer.source_buffer(0).rgba[..., 3].max())
    0
    """

    def __init__(
        self,
        levels,
        *,
        fps=24,
        size=(1088, 140),
        bar_width=12,
        pitch=17,
        margin=(2, 4),
        palette="rainbow",
        opacity=30,
        name="Spectrum",
        **kwargs,
    ):
        if kwargs.get("transform") is None:
            kwargs["transform"] = Transform(opacity=opacity)
        super().__init__(name, **kwargs)
        levels = np.asarray(levels, dtype=np.float32)
        if levels.ndim != 2 or len(levels) == 0:
            raise ValueError("levels must be a non-empty 2-D array")
        if fps <= 0 or bar_width < 1 or pitch < bar_width:
            raise ValueError("fps, bar_width and pitch must be valid")
        self.levels = levels
        self.fps = fps
        self._size = (int(size[0]), int(size[1]))
        width, height = self._size
        self._margin = (int(margin[0]), int(margin[1]))
        bands = levels.shape[1]
        self._colors = _palette(bands, palette)
        x = np.arange(width) - self._margin[0]
        band = np.where(x >= 0, x // pitch, -1)
        inside = (x >= 0) & (x % pitch < bar_width) & (band < bands)
        self._column_band = np.where(inside, band, -1)
        self._rows = np.arange(height)[:, None]
        self._last = None

    @property
    def source_size(self):
        """Return the layer size in pixels."""
        return self._size

    def frame_index(self, t):
        """Return the clamped level row shown at time ``t``."""
        return min(len(self.levels) - 1, max(0, int(round(t * self.fps))))

    def source_buffer(self, t, context=None):
        """Return the premultiplied bar image for the level row at ``t``."""
        index = self.frame_index(t)
        if self._last is not None and self._last[0] == index:
            return self._last[1]
        width, height = self._size
        bottom = height - self._margin[1]
        cap = height - 2 * self._margin[1]
        heights = np.rint(np.clip(self.levels[index].astype(np.float64), 0, 1) * cap)
        column = self._column_band
        column_height = np.where(column >= 0, heights[np.maximum(column, 0)], 0)
        mask = (self._rows >= bottom - column_height[None, :]) & (self._rows < bottom)
        mask &= (column >= 0)[None, :]
        rgba = np.zeros((height, width, 4), np.float32)
        colors = self._colors[np.maximum(column, 0)].astype(np.float32) / 255.0
        rgba[..., :3] = np.where(mask[..., None], colors[None, :, :], 0.0)
        rgba[..., 3] = mask
        buffer = Buffer(rgba)
        self._last = (index, buffer)
        return buffer
