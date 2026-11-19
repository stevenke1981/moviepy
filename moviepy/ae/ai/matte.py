"""Classical edge refinement and image inpainting; these are not AI models."""

import cv2
import numpy as np

from moviepy.ae._geometry import finite_real


def _inputs(frame, mask):
    image = np.asarray(frame)
    coverage = np.asarray(mask)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("frame must be uint8 RGB with shape (H, W, 3)")
    if 0 in image.shape[:2] or coverage.shape != image.shape[:2]:
        raise ValueError("mask must match a nonempty frame")
    if not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("mask values must be finite and within 0..1")
    return image, np.array(coverage, dtype=np.float32, copy=True)


def refine_matte(frame, mask, *, radius=4, epsilon=0.01):
    """Guide alpha edges by image luma while preserving confident interiors.

    This classical guided filter refines a supplied matte. It does not infer
    missing objects, recover hidden hair, decontaminate RGB or provide temporal
    consistency. ``mask`` is HxW coverage in 0..1, ``frame`` is uint8 RGB.
    """
    image, coverage = _inputs(frame, mask)
    radius = finite_real(radius, "radius")
    epsilon = finite_real(epsilon, "epsilon")
    if radius < 0 or radius > 128 or radius != int(radius) or epsilon <= 0:
        raise ValueError(
            "radius must be an integer in 0..128; epsilon must be positive"
        )
    if radius == 0 or np.ptp(coverage) == 0:
        return coverage
    kernel = (2 * int(radius) + 1,) * 2
    guide = (
        image.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32) / 255
    )

    def mean(array):
        return cv2.boxFilter(array, -1, kernel, borderType=cv2.BORDER_REFLECT)

    mean_i, mean_p = mean(guide), mean(coverage)
    variance = mean(guide * guide) - mean_i * mean_i
    a = (mean(guide * coverage) - mean_i * mean_p) / (np.maximum(variance, 0) + epsilon)
    b = mean_p - a * mean_i
    refined = np.clip(mean(a) * guide + mean(b), 0, 1)
    boundary = (
        cv2.morphologyEx(coverage, cv2.MORPH_GRADIENT, np.ones(kernel, np.uint8)) > 0
    )
    coverage[boundary] = refined[boundary]
    return coverage


def inpaint_classical(frame, mask, *, radius=3, method="telea"):
    """Remove marked pixels with OpenCV Telea/Navier-Stokes, not generative AI.

    Values greater than zero in the 0..1 mask mark pixels to replace. Pixels
    outside the mask remain byte-identical. No model, upload or network call
    is involved. Large missing objects may not be reconstructed convincingly.
    """
    image, coverage = _inputs(frame, mask)
    radius = finite_real(radius, "radius")
    if radius <= 0 or radius > 128:
        raise ValueError("radius must be within (0, 128]")
    methods = {"telea": cv2.INPAINT_TELEA, "navier_stokes": cv2.INPAINT_NS}
    if method not in methods:
        raise ValueError("method must be telea or navier_stokes")
    selected = (coverage > 0).astype(np.uint8)
    if not selected.any():
        return image.copy()
    if selected.all():
        raise ValueError("inpainting needs some unmasked context")
    result = cv2.inpaint(image, selected, radius, methods[method])
    result[selected == 0] = image[selected == 0]
    return result
