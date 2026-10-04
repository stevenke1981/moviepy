"""Gaussian Blur (Blur & Sharpen)."""

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
        Param("blurriness", "float", 0.0, (0.0, 1000.0)),
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
        rgba = gaussian(src.rgba, sigma_x, sigma_y, border)
        return Buffer._publish(rgba, src.offset, src.color_space)


def gaussian(rgba, sigma_x, sigma_y, border):
    """Return a separable Gaussian blur of float32 RGBA (zero sigma = skip)."""
    out = np.ascontiguousarray(rgba, dtype=np.float32)
    for axis, sigma in ((1, sigma_x), (0, sigma_y)):
        if sigma <= 0:
            continue
        radius = sigma_margin(sigma)
        kernel = cv2.getGaussianKernel(2 * radius + 1, sigma, cv2.CV_32F)
        kx, ky = (kernel, np.ones((1, 1), np.float32))
        if axis == 0:
            kx, ky = ky, kernel
        out = cv2.sepFilter2D(out, -1, kx, ky, borderType=border)
    # float32 kernels can overshoot by an ulp; keep a valid premultiplied range.
    np.clip(out, 0.0, 1.0, out=out)
    np.minimum(out[..., :3], out[..., 3:], out=out[..., :3])
    return out
