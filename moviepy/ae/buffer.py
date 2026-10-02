"""Immutable image values in a premultiplied float32 working color space.

RGB may contain finite signed or HDR values. Color-space labels describe the
existing numeric encoding; this module performs no color-space conversion.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from threading import Lock
from typing import Optional, Tuple
from weakref import finalize

import numpy as np

Offset = Tuple[int, int]
Bounds = Tuple[int, int, int, int]
RGB = Tuple[float, float, float]
_UNSET = object()
_FREE_STORAGE = []
_STORAGE_LOCK = Lock()
_STORAGE_BYTE_LIMIT = 64 * 1024 * 1024


def _release_storage(storage):
    """Retain at most two freed allocations, bounded to 64 MiB in total."""
    if storage.nbytes > _STORAGE_BYTE_LIMIT:
        return
    with _STORAGE_LOCK:
        _FREE_STORAGE.append(storage)
        while (
            len(_FREE_STORAGE) > 2
            or sum(item.nbytes for item in _FREE_STORAGE) > _STORAGE_BYTE_LIMIT
        ):
            _FREE_STORAGE.pop(0)


class _StorageLease:
    """Keep raw memory leased until the last ndarray or escaped view dies."""

    def __init__(self, storage):
        self.__array_interface__ = storage.__array_interface__
        retirement = finalize(self, _release_storage, storage)
        retirement.atexit = False

    def protect(self):
        """Expose read-only views through the public ndarray base chain."""
        pointer, _ = self.__array_interface__["data"]
        self.__array_interface__["data"] = (pointer, True)


def _protect_storage(array):
    """Protect every accessible NumPy base, including an allocation lease."""
    current = array
    while isinstance(current, np.ndarray):
        current.setflags(write=False)
        current = current.base
    if isinstance(current, _StorageLease):
        current.protect()


def _owned_output(shape):
    """Acquire storage for a completely overwritten, independently owned output."""
    size = shape[0] * shape[1] * shape[2]
    storage = None
    with _STORAGE_LOCK:
        for index, candidate in enumerate(_FREE_STORAGE):
            if candidate.size == size:
                storage = _FREE_STORAGE.pop(index)
                break
    if storage is None:
        storage = np.empty(size, dtype=np.float32)
    # ndarray.base retains the lease, including when views escape their Buffer.
    # The lease itself does not retain an ndarray that references the lease.
    return np.asarray(_StorageLease(storage)).reshape(shape)


def _pixel_metadata(array):
    """Summarize protected pixels once, without caching rendered outputs."""
    if not array.size:
        return None, 0.0, True
    alpha = array[0, 0, 3]
    uniform = alpha if np.all(array[..., 3] == alpha) else None
    minimum, maximum = float(np.min(array)), float(np.max(array))
    normalized = uniform is not None and minimum >= 0 and maximum <= float(uniform)
    return uniform, max(abs(minimum), abs(maximum)), normalized


def _native_over(foreground, background):
    """Use native float32 arithmetic when uniform alpha cannot overflow."""
    alpha = foreground._uniform_alpha
    if alpha is None:
        return None
    weight = np.float32(1) - alpha
    bound = foreground._value_bound + background._value_bound * float(weight)
    if bound > float(np.finfo(np.float32).max):
        return None
    import cv2

    result = _owned_output(background.rgba.shape)
    cv2.addWeighted(background.rgba, float(weight), foreground.rgba, 1, 0, dst=result)
    return result


def _over_metadata(foreground, background, array):
    """Propagate conservative summaries without scanning rendered pixels."""
    alpha = None
    if foreground.bounds == background.bounds:
        if (
            foreground._uniform_alpha is not None
            and background._uniform_alpha is not None
        ):
            alpha = array[0, 0, 3]
    elif foreground._uniform_alpha == 0 and background._uniform_alpha == 0:
        alpha = np.float32(0)
    return (
        alpha,
        foreground._value_bound + background._value_bound,
        foreground._unit_premultiplied and background._unit_premultiplied,
    )


def _export_rgb(buffer, background):
    """Extract owned contiguous RGB and apply working-space alpha arithmetic."""
    import cv2

    source = buffer.rgba
    rgb = np.empty((*source.shape[:2], 3), dtype=np.float32)
    if not rgb.size:
        return rgb
    cv2.mixChannels([source], [rgb], [0, 0, 1, 1, 2, 2])
    alpha = buffer._uniform_alpha
    with _arithmetic("rgba"):
        if background is None:
            if alpha is not None:
                if alpha > 0 and alpha != 1:
                    np.divide(rgb, alpha, out=rgb)
            else:
                np.divide(rgb, source[..., 3:4], out=rgb, where=source[..., 3:4] > 0)
        else:
            np.add(rgb, background * (1 - source[..., 3:4]), out=rgb)
    return rgb


def _numeric(value, name, shape_tail, *, allow_bool=False):
    """Validate real numeric channels before any float32 rounding."""
    array = np.asarray(value)
    kinds = "iufb" if allow_bool else "iuf"
    if array.dtype.kind not in kinds:
        raise TypeError(f"{name} must contain real numeric values")
    if array.ndim != len(shape_tail) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape_tail)
    ):
        raise ValueError(f"{name} must have shape {shape_tail}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _snapshot(value, name):
    """Detach numeric input, rejecting float32 overflow explicitly."""
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must remain finite in float32")
    return array


def _rgba_snapshot(value):
    """Validate RGBA and canonicalize fully transparent pixels."""
    array = _numeric(value, "rgba", (None, None, 4))
    if np.any(array[..., 3] < 0) or np.any(array[..., 3] > 1):
        raise ValueError("rgba alpha must be in [0, 1]")
    result = _snapshot(array, "rgba")
    result[..., :3][result[..., 3] == 0] = 0
    return result


def _mask_snapshot(mask, shape=None):
    """Validate mask values, optionally requiring an exact image size."""
    array = _numeric(mask, "mask", (None, None), allow_bool=True)
    if shape is not None and array.shape != shape:
        raise ValueError("mask shape must match frame height and width")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError("mask values must be in [0, 1]")
    return _snapshot(array, "mask")


def _coordinates(value, name, length):
    """Normalize integral coordinates while rejecting bool and fractions."""
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must contain {length} integers") from exc
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} integers")
    if any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Integral)
        for item in values
    ):
        raise TypeError(f"{name} must contain integers, excluding bool")
    return tuple(int(item) for item in values)


def _bounds(value):
    """Validate a half-open world rectangle."""
    bounds = _coordinates(value, "bounds", 4)
    if bounds[2] < bounds[0] or bounds[3] < bounds[1]:
        raise ValueError("bounds must not be inverted")
    return bounds


def _color_space(value):
    """Validate encoding labels without interpreting their values."""
    if not isinstance(value, str):
        raise TypeError("color_space must be a string")
    if value not in ("srgb", "linear") and not (
        value.startswith("ocio:") and value[5:].strip()
    ):
        raise ValueError("color_space must be srgb, linear or ocio:<name>")
    return value


@contextmanager
def _arithmetic(name):
    """Convert invalid/overflow arithmetic into a parameter-specific error."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            yield
    except FloatingPointError as exc:
        raise ValueError(f"{name} arithmetic must remain finite in float32") from exc


def _background(bg):
    """Validate a normalized constant opaque background."""
    values = _numeric(bg, "bg", (3,))
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("bg values must be normalized to [0, 1]")
    return _snapshot(values, "bg")


def _aligned_mask(mask, shape):
    """Crop or transparently pad a source mask from its top-left corner."""
    source = _mask_snapshot(mask)
    if source.shape == shape:
        return source
    output = np.zeros(shape, dtype=np.float32)
    height, width = min(shape[0], source.shape[0]), min(shape[1], source.shape[1])
    output[:height, :width] = source[:height, :width]
    return output


def _rgb_rgba(frame, mask):
    """Import numeric RGB codes once, retaining fractional code precision."""
    result = np.empty((*frame.shape[:2], 4), dtype=np.float32)
    result[..., 3] = 1 if mask is None else mask
    np.divide(frame, np.float32(255), out=result[..., :3], casting="unsafe")
    result[..., :3] *= result[..., 3:4]
    return result


def _over_arrays(foreground, background, bounds):
    """Allocate only the result and alpha weight for normal source-over."""
    if foreground.bounds == background.bounds:
        native = _native_over(foreground, background)
        if native is not None:
            return native
        result = _owned_output(background.rgba.shape)
        weight = np.float32(1) - foreground.rgba[..., 3:4]
        np.multiply(background.rgba, weight, out=result)
        np.add(result, foreground.rgba, out=result)
        return result
    left, top, right, bottom = bounds
    result = _owned_output((bottom - top, right - left, 4))
    result.fill(0)
    bx, by = background.offset[0] - left, background.offset[1] - top
    bw, bh = background.size
    result[by : by + bh, bx : bx + bw] = background.rgba
    fx, fy = foreground.offset[0] - left, foreground.offset[1] - top
    fw, fh = foreground.size
    region = result[fy : fy + fh, fx : fx + fw]
    np.multiply(region, np.float32(1) - foreground.rgba[..., 3:4], out=region)
    np.add(region, foreground.rgba, out=region)
    return result


def premultiply(straight_rgba: np.ndarray) -> np.ndarray:
    """Multiply straight RGB by alpha in a detached float32 array.

    Parameters
    ----------
    straight_rgba : array_like
        Finite ``(H, W, 4)`` real values, with alpha in [0, 1]. RGB may be
        signed or HDR. Fully transparent RGB is canonicalized to zero.

    Returns
    -------
    numpy.ndarray
        Independent float32 premultiplied RGBA values.

    Examples
    --------
    >>> premultiply(np.array([[[1., 0., 0., 0.5]]])).tolist()
    [[[0.5, 0.0, 0.0, 0.5]]]
    """
    result = _rgba_snapshot(straight_rgba)
    result[..., :3] *= result[..., 3:4]
    return result


def unpremultiply(premultiplied_rgba: np.ndarray) -> np.ndarray:
    """Recover straight RGB without dividing fully transparent pixels.

    Parameters
    ----------
    premultiplied_rgba : array_like
        Finite premultiplied ``(H, W, 4)`` values with alpha in [0, 1].

    Returns
    -------
    numpy.ndarray
        Detached float32 straight RGBA. RGB at zero alpha is zero. Small
        positive alpha is used exactly; no opacity epsilon is applied.

    Examples
    --------
    >>> unpremultiply(np.array([[[0.5, 0., 0., 0.5]]])).tolist()
    [[[1.0, 0.0, 0.0, 0.5]]]
    """
    source = _rgba_snapshot(premultiplied_rgba)
    result = np.zeros_like(source)
    result[..., 3] = source[..., 3]
    with _arithmetic("rgba"):
        np.divide(
            source[..., :3],
            source[..., 3:4],
            out=result[..., :3],
            where=source[..., 3:4] > 0,
        )
    return result


@dataclass(frozen=True, eq=False)
class Buffer:
    """Store a read-only snapshot of premultiplied float32 RGBA.

    Parameters
    ----------
    rgba : array_like
        Finite ``(H, W, 4)`` premultiplied values. Alpha is in [0, 1];
        signed and HDR RGB values are retained. Zero-alpha RGB becomes zero.
    offset : tuple of int, optional
        World coordinates of pixel [0, 0], in (x, y) order.
    color_space : str, optional
        Encoding label: ``srgb``, ``linear``, or ``ocio:<name>``.

    Notes
    -----
    Construction copies caller data. Fields are frozen and the array is
    read-only under normal NumPy use; deliberate WRITEABLE flag changes are
    outside this API contract. Equality uses identity rather than ndarray
    comparisons. No operation changes color encoding or quantizes RGB early.

    Examples
    --------
    >>> frame = np.array([[[255, 0, 0]]], dtype=np.uint8)
    >>> Buffer.from_uint8_rgb(frame).to_uint8_rgb().tolist()
    [[[255, 0, 0]]]
    """

    rgba: np.ndarray
    offset: Offset = (0, 0)
    color_space: str = "srgb"

    def __post_init__(self):
        array = _rgba_snapshot(self.rgba)
        array.setflags(write=False)
        object.__setattr__(self, "rgba", array)
        object.__setattr__(self, "offset", _coordinates(self.offset, "offset", 2))
        object.__setattr__(self, "color_space", _color_space(self.color_space))
        self._set_metadata(_pixel_metadata(array))

    def _set_metadata(self, metadata):
        """Attach private summaries to immutable, protected pixel storage."""
        object.__setattr__(self, "_uniform_alpha", metadata[0])
        object.__setattr__(self, "_value_bound", metadata[1])
        object.__setattr__(self, "_unit_premultiplied", metadata[2])

    @classmethod
    def _publish(cls, array, offset, color_space, metadata=_UNSET):
        """Publish owned or already protected storage without another copy."""
        _protect_storage(array)
        instance = object.__new__(cls)
        object.__setattr__(instance, "rgba", array)
        object.__setattr__(instance, "offset", offset)
        object.__setattr__(instance, "color_space", color_space)
        instance._set_metadata(
            _pixel_metadata(array) if metadata is _UNSET else metadata
        )
        return instance

    @classmethod
    def from_uint8_rgb(
        cls,
        frame: np.ndarray,
        mask: Optional[np.ndarray] = None,
        *,
        offset: Offset = (0, 0),
    ) -> "Buffer":
        """Import strictly uint8 RGB codes and an optional matching mask.

        Parameters
        ----------
        frame : numpy.ndarray
            uint8 image of shape (H, W, 3).
        mask : array_like, optional
            Matching (H, W) finite opacity values in [0, 1]; bool is allowed.
        offset : tuple of int, optional
            World (x, y) coordinates; does not pad the imported pixels.

        Returns
        -------
        Buffer
            Detached sRGB-labeled premultiplied values. Unmasked uint8 codes
            round-trip exactly; hidden colors at alpha zero are discarded.
        """
        array = _numeric(frame, "frame", (None, None, 3))
        if array.dtype != np.uint8:
            raise TypeError("frame must have dtype uint8")
        position = _coordinates(offset, "offset", 2)
        alpha = None if mask is None else _mask_snapshot(mask, array.shape[:2])
        return cls._publish(_rgb_rgba(array, alpha), position, "srgb")

    @classmethod
    def from_clip(cls, clip, t: float, *, offset: Offset = (0, 0)) -> "Buffer":
        """Read one legacy clip frame and mask at the same source-local time.

        Parameters
        ----------
        clip : VideoClip
            RGB source; mask clips and raw grayscale/RGBA are unsupported.
            RGB may be floating point, but always represents codes 0..255.
        t : float
            Finite local seconds. Clip/mask start, position and duration
            do not alter this time. Fractional RGB codes retain precision.
        offset : tuple of int, optional
            World coordinates of the returned pixels.

        Returns
        -------
        Buffer
            Premultiplied sRGB codes. Mismatched masks are cropped or padded
            transparently from the top left using the actual frame size.
        """
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, Real):
            raise TypeError("t must be a real numeric source-local time")
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        if getattr(clip, "is_mask", False):
            raise ValueError("clip must be an RGB clip, not a mask clip")
        if not callable(getattr(clip, "get_frame", None)):
            raise TypeError("clip must provide get_frame(t)")
        position = _coordinates(offset, "offset", 2)
        frame = _numeric(clip.get_frame(t), "frame", (None, None, 3))
        if np.any(frame < 0) or np.any(frame > 255):
            raise ValueError("frame RGB codes must be in [0, 255]")
        mask_clip = getattr(clip, "mask", None)
        mask = (
            None
            if mask_clip is None
            else _aligned_mask(mask_clip.get_frame(t), frame.shape[:2])
        )
        return cls._publish(_rgb_rgba(frame, mask), position, "srgb")

    @property
    def size(self) -> Tuple[int, int]:
        """Return pixel extent in (width, height) order.

        Returns
        -------
        tuple of int
            Pixel width and height, including zero-sized axes.
        """
        return (self.rgba.shape[1], self.rgba.shape[0])

    @property
    def bounds(self) -> Bounds:
        """Return half-open world (left, top, right, bottom) bounds.

        Returns
        -------
        tuple of int
            World rectangle derived from offset and pixel size.
        """
        x, y = self.offset
        width, height = self.size
        return (x, y, x + width, y + height)

    def straight(self) -> np.ndarray:
        """Return independent straight RGBA values with safe zero alpha.

        Returns
        -------
        numpy.ndarray
            float32 (H, W, 4) straight values. Zero-alpha RGB is zero.
            This array is not a Buffer, which always stores premultiplied RGB.
        """
        return unpremultiply(self.rgba)

    def to_uint8_rgb(self, bg: Optional[RGB] = None) -> np.ndarray:
        """Quantize working-space RGB once using nearest-even rounding.

        Parameters
        ----------
        bg : tuple of float, optional
            Opaque constant background of three normalized values in [0, 1],
            in this Buffer's encoding. If omitted, export straight RGB.

        Returns
        -------
        numpy.ndarray
            uint8 (H, W, 3) RGB. Values are clipped to [0, 1] only here.
            Offsets do not add a canvas. Non-sRGB labels export raw working
            codes, without a display color conversion or tone mapping.
        """
        background = None if bg is None else _background(bg)
        if self._uniform_alpha == 1 and self._unit_premultiplied:
            import cv2

            # This is the sole quantization; removing output alpha changes no codes.
            return cv2.convertScaleAbs(self.rgba, alpha=255)[..., :3].copy()
        rgb = _export_rgb(self, background)
        np.clip(rgb, 0, 1, out=rgb)
        if not rgb.size:
            return rgb.astype(np.uint8)
        import cv2

        return cv2.convertScaleAbs(rgb, alpha=255)

    def with_rgba(self, rgba: np.ndarray) -> "Buffer":
        """Replace pixel data with a new defensive snapshot.

        Parameters
        ----------
        rgba : array_like
            Valid premultiplied (H, W, 4) values; size may change.

        Returns
        -------
        Buffer
            New pixels with the current offset and color-space label.
        """
        return type(self)(rgba, self.offset, self.color_space)

    def with_offset(self, offset: Offset) -> "Buffer":
        """Return a new value with pixels placed at different world coordinates.

        Parameters
        ----------
        offset : tuple of int
            New (x, y) world coordinates.

        Returns
        -------
        Buffer
            New Buffer sharing only already protected pixel storage.
        """
        position = _coordinates(offset, "offset", 2)
        return self._publish(
            self.rgba,
            position,
            self.color_space,
            (self._uniform_alpha, self._value_bound, self._unit_premultiplied),
        )

    def crop(self, bounds: Bounds) -> "Buffer":
        """Take an intersection with a half-open world rectangle.

        Parameters
        ----------
        bounds : tuple of int
            Requested (left, top, right, bottom), without inverted axes.

        Returns
        -------
        Buffer
            Intersection pixels at their original world positions. A disjoint
            or zero-area crop returns (0, 0, 4) at the requested top-left.
        """
        requested = _bounds(bounds)
        left, top = max(requested[0], self.bounds[0]), max(requested[1], self.bounds[1])
        right = min(requested[2], self.bounds[2])
        bottom = min(requested[3], self.bounds[3])
        if right <= left or bottom <= top:
            array = np.zeros((0, 0, 4), dtype=np.float32)
            return self._publish(array, requested[:2], self.color_space)
        x, y = self.offset
        array = self.rgba[top - y : bottom - y, left - x : right - x].copy()
        return self._publish(array, (left, top), self.color_space)

    def pad(
        self, *, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0
    ) -> "Buffer":
        """Add transparent margins while preserving world pixel positions.

        Parameters
        ----------
        left, top, right, bottom : int, optional
            Nonnegative margin sizes. Bool and fractional values are rejected.

        Returns
        -------
        Buffer
            Expanded pixels with offset reduced by the left/top margins.
        """
        for name, value in (
            ("left", left),
            ("top", top),
            ("right", right),
            ("bottom", bottom),
        ):
            _coordinates((value,), name, 1)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        width, height = self.size
        array = np.zeros((height + top + bottom, width + left + right, 4), np.float32)
        array[top : top + height, left : left + width] = self.rgba
        position = (self.offset[0] - int(left), self.offset[1] - int(top))
        return self._publish(array, position, self.color_space)

    def expand_to(self, bounds: Bounds) -> "Buffer":
        """Expand transparently to a containing half-open world rectangle.

        Parameters
        ----------
        bounds : tuple of int
            Desired exact (left, top, right, bottom) containing current pixels.
            Empty source buffers can expand to any valid rectangle.

        Returns
        -------
        Buffer
            The requested extent, preserving world positions. A rectangle
            that would crop nonempty source pixels raises ValueError.
        """
        target = _bounds(bounds)
        if 0 in self.size:
            array = np.zeros(
                (target[3] - target[1], target[2] - target[0], 4), np.float32
            )
            return self._publish(array, target[:2], self.color_space)
        left, top, right, bottom = self.bounds
        if (
            target[0] > left
            or target[1] > top
            or target[2] < right
            or target[3] < bottom
        ):
            raise ValueError("bounds must contain the current buffer")
        return self.pad(
            left=left - target[0],
            top=top - target[1],
            right=target[2] - right,
            bottom=target[3] - bottom,
        )

    def composite_over(self, background: "Buffer") -> "Buffer":
        """Composite this foreground over another Buffer using normal over.

        Parameters
        ----------
        background : Buffer
            Lower pixels with an identical color-space label.

        Returns
        -------
        Buffer
            Union of nonempty world bounds, computed in float32 as
            ``foreground + background * (1 - foreground_alpha)`` for all
            four premultiplied channels. No clipping or quantization occurs.
            Empty inputs act as identities; two empty inputs use this offset.
        """
        if not isinstance(background, Buffer):
            raise TypeError("background must be a Buffer")
        if self.color_space != background.color_space:
            raise ValueError("color_space must match for compositing")
        if 0 in background.size:
            return self._publish(
                self.rgba,
                self.offset,
                self.color_space,
                (self._uniform_alpha, self._value_bound, self._unit_premultiplied),
            )
        if 0 in self.size:
            return self._publish(
                background.rgba,
                background.offset,
                self.color_space,
                (
                    background._uniform_alpha,
                    background._value_bound,
                    background._unit_premultiplied,
                ),
            )
        fg, bg = self.bounds, background.bounds
        bounds = (
            min(fg[0], bg[0]),
            min(fg[1], bg[1]),
            max(fg[2], bg[2]),
            max(fg[3], bg[3]),
        )
        with _arithmetic("composite"):
            array = _over_arrays(self, background, bounds)
        metadata = _over_metadata(self, background, array)
        return self._publish(array, bounds[:2], self.color_space, metadata)
