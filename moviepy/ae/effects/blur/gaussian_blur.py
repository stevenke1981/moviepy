"""Gaussian Blur (Blur & Sharpen)."""

from functools import lru_cache

import cv2
import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect, Param, sigma_margin
from moviepy.ae.effects.registry import register


DIMENSIONS = ("horizontal_and_vertical", "horizontal", "vertical")


@register
class GaussianBlur(AEEffect):
    """Blur premultiplied pixels with a separable Gaussian kernel.

    ==================  ========================  ===========================
    Parameter           AE panel name             Notes
    ==================  ========================  ===========================
    blurriness          Blurriness                0..1000 px, default 0
    blur_dimensions     Blur Dimensions           both / horizontal / vertical
    repeat_edge_pixels  Repeat Edge Pixels        replicate edges, no growth
    ==================  ========================  ===========================

    Notes
    -----
    ``sigma = blurriness / 2`` (the same convention as mask feather). Adobe
    does not publish its kernel; this calibration keeps the visible radius
    (about ``1.5 * blurriness``) close to AE. Without ``repeat_edge_pixels``
    the layer grows by ``3 sigma`` so the blur can spill outside its bounds.
    Blurring premultiplied RGBA is the correct (halo-free) operation.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae import Buffer
    >>> impulse = np.zeros((1, 1, 4), dtype=np.float32); impulse[...] = 1
    >>> out = GaussianBlur(blurriness=4).process(Buffer(impulse), 0.0)
    >>> out.size, round(float(out.rgba[..., 3].sum()), 4)
    ((13, 13), 1.0)
    """

    name = "Gaussian Blur"
    category = "Blur & Sharpen"
    PARAMS = (
        Param("blurriness", "float", 0.0, (0.0, 1000.0), unit="px"),
        Param("blur_dimensions", "enum", DIMENSIONS[0], choices=DIMENSIONS),
        Param("repeat_edge_pixels", "bool", False),
    )

    @staticmethod
    def sigmas(values):
        """Return ``(sigma_x, sigma_y)`` for evaluated parameters."""
        sigma = values["blurriness"] / 2.0
        dimensions = values["blur_dimensions"]
        sigma_x = sigma if dimensions != "vertical" else 0.0
        sigma_y = sigma if dimensions != "horizontal" else 0.0
        return sigma_x, sigma_y

    def bounds_expand(self, size, t, context=None, values=None):
        """Grow by three sigma per blurred axis unless edges repeat."""
        values = self.values_at(t, context) if values is None else values
        if values["repeat_edge_pixels"]:
            return (0, 0, 0, 0)
        return self.input_margin(size, t, context, values)

    def input_margin(self, size, t, context=None, values=None):
        """Read three sigma per axis even when output edges do not grow."""
        values = self.values_at(t, context) if values is None else values
        sigma_x, sigma_y = self.sigmas(values)
        x, y = sigma_margin(sigma_x), sigma_margin(sigma_y)
        return (x, y, x, y)

    def render(self, src, t, context=None, values=None):
        """Blur ``src`` (already padded by ``bounds_expand``)."""
        values = self.values_at(t, context) if values is None else values
        sigma_x, sigma_y = self.sigmas(values)
        if 0 in src.size or (sigma_x <= 0 and sigma_y <= 0):
            return src
        border = cv2.BORDER_REPLICATE
        if not values["repeat_edge_pixels"]:
            border = cv2.BORDER_CONSTANT
        rgba = gaussian(
            src.rgba, sigma_x, sigma_y, border, unit_range=src._unit_premultiplied
        )
        return Buffer._publish(rgba, src.offset, src.color_space)


@lru_cache(maxsize=64)
def _kernel(sigma):
    """Cache only small immutable kernels, never rendered frames."""
    if sigma <= 0:
        result = np.ones((1, 1), np.float32)
    else:
        result = cv2.getGaussianKernel(2 * sigma_margin(sigma) + 1, sigma, cv2.CV_32F)
    result.setflags(write=False)
    return result


def filter_pixels(pixels, sigma_x, sigma_y, border=cv2.BORDER_CONSTANT):
    """Filter a plane or channels in one separable operation, without clipping."""
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)
    if sigma_x <= 0 and sigma_y <= 0:
        return pixels.copy()
    kernel_x, kernel_y = _kernel(float(sigma_x)), _kernel(float(sigma_y))
    radius_x, radius_y = len(kernel_x) // 2, len(kernel_y) // 2
    if (
        border == cv2.BORDER_CONSTANT
        and max(radius_x, radius_y) >= 32
        and pixels.shape[0] > 4 * radius_y
        and pixels.shape[1] > 4 * radius_x
    ):
        # A full kernel radius of explicit zero padding makes the outer border
        # irrelevant to the retained pixels. OpenCV's replicated-edge path is
        # much faster for large RGBA kernels; the kernel and sampling stay exact.
        padded = cv2.copyMakeBorder(
            pixels, radius_y, radius_y, radius_x, radius_x, cv2.BORDER_CONSTANT
        )
        result = cv2.sepFilter2D(
            padded, -1, kernel_x, kernel_y, borderType=cv2.BORDER_REPLICATE
        )
        return np.ascontiguousarray(
            result[
                radius_y : radius_y + pixels.shape[0],
                radius_x : radius_x + pixels.shape[1],
            ]
        )
    return cv2.sepFilter2D(pixels, -1, kernel_x, kernel_y, borderType=border)


def gaussian(rgba, sigma_x, sigma_y, border, *, unit_range=True):
    """Blur RGBA; preserve signed/HDR color while keeping valid coverage."""
    out = filter_pixels(rgba, sigma_x, sigma_y, border)
    np.clip(out[..., 3], 0.0, 1.0, out=out[..., 3])
    if unit_range:
        np.maximum(out[..., :3], 0.0, out=out[..., :3])
        np.minimum(out[..., :3], out[..., 3:], out=out[..., :3])
    out[out[..., 3] == 0, :3] = 0
    return out
