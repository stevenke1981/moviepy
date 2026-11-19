"""Sharpen and Unsharp Mask (Blur & Sharpen)."""

import cv2
import numpy as np

from moviepy.ae.effects._pixels import is_opaque, publish_straight, straight_rgb
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.blur.gaussian_blur import filter_pixels
from moviepy.ae.effects.registry import register


SHARPEN_RADIUS = 1.0


def blurred_straight(rgb, alpha, sigma):
    """Blur straight color weighted by alpha (transparent edges do not darken).

    Premultiplied sums are normalized by the blurred alpha, so each pixel
    averages only the visible color around it.
    """
    stacked = np.concatenate([rgb * alpha, alpha], axis=-1).astype(np.float32)
    blurred = filter_pixels(stacked, sigma, sigma, cv2.BORDER_CONSTANT)
    coverage = blurred[..., 3:]
    safe = np.maximum(coverage, np.float32(1e-6))
    return np.where(coverage > 1e-6, blurred[..., :3] / safe, rgb)


def _unsharp(src, sigma, amount, threshold):
    """Straight color plus (optionally thresholded) high-pass detail."""
    rgb = straight_rgb(src)
    alpha = src.rgba[..., 3:]
    detail = rgb - blurred_straight(rgb, alpha, sigma)
    if threshold > 0:
        detail = detail * (np.abs(detail) * 255.0 >= threshold)
    result = rgb + detail * np.float32(amount)
    return publish_straight(result, alpha, src, is_opaque(src))


@register
class Sharpen(AEEffect):
    """Increase local contrast of straight color with a small unsharp mask.

    ==================  ==================  ==========================
    Parameter           AE panel name       Notes
    ==================  ==================  ==========================
    sharpen_amount      Sharpen Amount      0..500 %, default 0
    ==================  ==================  ==========================

    Notes
    -----
    The detail is ``color - blur(color)`` with a fixed 1 px Gaussian, and the
    output is ``color + detail * amount``. Adobe's Sharpen uses an unpublished
    kernel, and this version has no radius control (use Unsharp Mask for
    that). Color is blurred alpha-weighted, so transparent edges do not
    darken. Alpha is unchanged and color is clipped to 0..1.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae import Buffer
    >>> flat = np.full((4, 4, 4), 0.5, dtype=np.float32)
    >>> out = Sharpen(sharpen_amount=0).process(Buffer(flat), 0.0)
    >>> bool(np.array_equal(out.rgba, flat))
    True
    """

    name = "Sharpen"
    category = "Blur & Sharpen"
    PARAMS = (Param("sharpen_amount", "float", 0.0, (0.0, 500.0)),)

    def render(self, src, t, context=None, values=None):
        """Apply the fixed-radius unsharp mask to straight color."""
        values = self.values_at(t, context) if values is None else values
        amount = values["sharpen_amount"] / 100.0
        if 0 in src.size or amount == 0.0:
            return src
        return _unsharp(src, SHARPEN_RADIUS, amount, 0.0)


@register
class UnsharpMask(AEEffect):
    """Sharpen straight color by adding back thresholded high-pass detail.

    ==========  ===========  =========================================
    Parameter   AE panel     Notes
    ==========  ===========  =========================================
    amount      Amount       0..500 %, default 0 (identity)
    radius      Radius       0..100 px, Gaussian sigma, default 1
    threshold   Threshold    0..255 levels, default 0
    ==========  ===========  =========================================

    Notes
    -----
    ``detail = color - blur(color, radius)``. Only detail whose magnitude is
    at least ``threshold`` levels (the 0..1 value times 255) is amplified by
    ``amount``. Adobe's internal blur is not published, and threshold is
    applied to the working-space value, not to gamma-encoded display levels.
    Alpha is unchanged and color is clipped to 0..1.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae import Buffer
    >>> flat = np.full((4, 4, 4), 0.5, dtype=np.float32)
    >>> out = UnsharpMask(amount=200, radius=2).process(Buffer(flat), 0.0)
    >>> bool(np.allclose(out.rgba, flat))
    True
    """

    name = "Unsharp Mask"
    category = "Blur & Sharpen"
    PARAMS = (
        Param("amount", "float", 0.0, (0.0, 500.0)),
        Param("radius", "float", 1.0, (0.0, 100.0), unit="px"),
        Param("threshold", "float", 0.0, (0.0, 255.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Apply the thresholded unsharp mask to straight color."""
        values = self.values_at(t, context) if values is None else values
        amount = values["amount"] / 100.0
        if 0 in src.size or amount == 0.0 or values["radius"] <= 0:
            return src
        return _unsharp(src, values["radius"], amount, values["threshold"])
