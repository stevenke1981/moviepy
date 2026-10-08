"""Renderer-owned mutable pixels, published only at immutable boundaries."""

import numpy as np

from moviepy.ae.blend.modes import BlendMode, blend
from moviepy.ae.buffer import Buffer, _arithmetic


_TILE_PIXELS = 32768


def _over_varying(region, source):
    """Apply ``region * (1 - alpha) + source`` skipping transparent margins."""
    import cv2

    top, bottom, left, right = source._content_box()
    if top >= bottom:
        return
    pixels = source.rgba[top:bottom, left:right]
    target = region[top:bottom, left:right]
    height, width = pixels.shape[:2]
    rows = max(1, _TILE_PIXELS // width)
    count = min(rows, height)
    flat = np.empty((count, width), np.float32)
    weight = np.empty((count, width, 4), np.float32)
    for start in range(0, height, rows):
        n = min(rows, height - start)
        tile = target[start : start + n]
        np.subtract(1, pixels[start : start + n, :, 3], out=flat[:n])
        # Replicating the weight avoids a slow 4-wide broadcast inner loop.
        cv2.mixChannels([flat[:n]], [weight[:n]], [0, 0, 0, 1, 0, 2, 0, 3])
        np.multiply(tile, weight[:n], out=tile)
        np.add(tile, pixels[start : start + n], out=tile)


class _Accumulator:
    """Batch Normal layers without copying the entire backdrop per layer."""

    def __init__(self, bounds, color_space):
        self.color_space = color_space
        self.bounds = bounds
        self._buffer = None
        self._array = None if bounds is None else self._empty(bounds)
        self._metadata = (np.float32(0), 0.0, True)

    @staticmethod
    def _empty(bounds):
        return np.zeros((bounds[3] - bounds[1], bounds[2] - bounds[0], 4), np.float32)

    def snapshot(self):
        """Transfer ownership; future writes must detach from this snapshot."""
        if self._array is not None:
            self._buffer = Buffer._publish(
                self._array, self.bounds[:2], self.color_space, self._metadata
            )
            self._array = None
        return self._buffer

    def replace(self, buffer):
        """Accept an immutable effect or fallback result without copying it."""
        self._array, self._buffer = None, buffer
        self.bounds = None if buffer is None else buffer.bounds
        self._metadata = (
            (np.float32(0), 0.0, True)
            if buffer is None
            else (
                buffer._uniform_alpha,
                buffer._value_bound,
                buffer._unit_premultiplied,
            )
        )

    def blend(self, source, mode, *, preserve=False, context=None, layer_id="layer"):
        """Use owned Normal pixels; retain public semantics for other modes."""
        if self.bounds is None:
            self.bounds = source.bounds
            self._array = self._empty(self.bounds)
        if not preserve and mode in (BlendMode.NORMAL, BlendMode.LUMINESCENT_PREMUL):
            self._normal(source)
        else:
            self.replace(
                blend(
                    self.snapshot(),
                    source,
                    mode,
                    preserve_underlying_transparency=preserve,
                    context=context,
                    layer_id=layer_id,
                )
            )

    def _writable(self, source):
        bounds = (
            min(self.bounds[0], source.bounds[0]),
            min(self.bounds[1], source.bounds[1]),
            max(self.bounds[2], source.bounds[2]),
            max(self.bounds[3], source.bounds[3]),
        )
        if bounds != self.bounds:
            old = self._array if self._array is not None else self._buffer.rgba
            array = self._empty(bounds)
            left, top = self.bounds[0] - bounds[0], self.bounds[1] - bounds[1]
            array[top : top + old.shape[0], left : left + old.shape[1]] = old
            self._array, self.bounds = array, bounds
        elif self._array is None:
            self._array = np.array(self._buffer.rgba)
        self._buffer = None
        left, top = source.offset[0] - self.bounds[0], source.offset[1] - self.bounds[1]
        width, height = source.size
        return self._array[top : top + height, left : left + width]

    def _normal(self, source):
        same_bounds = self.bounds == source.bounds
        alpha, bound, unit = self._metadata
        region = self._writable(source)
        source_alpha = source._uniform_alpha
        if source_alpha is not None and source_alpha == 0 and not source._value_bound:
            pass  # A zero source leaves the backdrop untouched.
        elif (alpha is not None and alpha == 0 and not bound) or (
            source_alpha is not None and source_alpha == 1
        ):
            # An all-zero backdrop or an opaque source (weight 0) yields the
            # source exactly; skip the multiply-add over the backdrop.
            np.copyto(region, source.rgba)
        elif source_alpha is not None:
            weight = np.float32(1) - source_alpha
            native = same_bounds and source._value_bound + bound * float(
                weight
            ) <= float(np.finfo(np.float32).max)
            with _arithmetic("composite"):
                if native:
                    import cv2

                    cv2.addWeighted(
                        region, float(weight), source.rgba, 1, 0, dst=region
                    )
                else:
                    np.multiply(region, weight, out=region)
                    np.add(region, source.rgba, out=region)
        else:
            with _arithmetic("composite"):
                _over_varying(region, source)
        uniform = None
        if same_bounds and alpha is not None and source_alpha is not None:
            uniform = region[0, 0, 3] if region.size else None
        elif alpha == 0 and source_alpha == 0:
            uniform = np.float32(0)
        self._metadata = (
            uniform,
            bound + source._value_bound,
            unit and source._unit_premultiplied,
        )
