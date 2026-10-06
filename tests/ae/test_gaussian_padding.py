"""Large finite Gaussian kernels retain boundary and HDR semantics."""

import cv2
import numpy as np

import pytest

from moviepy import VideoClip
from moviepy.ae import Composition
from moviepy.ae.effects.blur.gaussian_blur import GaussianBlur, filter_pixels


@pytest.mark.parametrize("sigmas", [(20, 20), (20, 0), (0, 20), (12, 17.3)])
@pytest.mark.parametrize("border", [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE])
@pytest.mark.parametrize("channels", [None, 4])
def test_large_filter_matches_float64_finite_kernel(sigmas, border, channels):
    shape = (257, 263) if channels is None else (257, 263, channels)
    pixels = np.random.default_rng(682).uniform(-0.25, 3, shape).astype(np.float32)
    kernels = [
        (
            cv2.getGaussianKernel(2 * int(np.ceil(3 * sigma)) + 1, sigma, cv2.CV_32F)
            if sigma
            else np.ones((1, 1), np.float32)
        )
        for sigma in sigmas
    ]
    expected = cv2.sepFilter2D(
        pixels.astype(np.float64),
        -1,
        kernels[0].astype(np.float64),
        kernels[1].astype(np.float64),
        borderType=border,
    )
    actual = filter_pixels(pixels, *sigmas, border)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1.5e-6)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_array_equal(
        pixels, np.random.default_rng(682).uniform(-0.25, 3, shape).astype(np.float32)
    )


def test_large_adjustment_blur_roi_matches_full_frame():
    pixels = np.random.default_rng(205).integers(0, 256, (300, 320, 3), np.uint8)
    comp = Composition(size=(320, 300), fps=24, duration=1)
    comp.add_clip(VideoClip(lambda _: pixels, duration=1))
    comp.add_adjustment(effects=[GaussianBlur(blurriness=40)])
    complete = comp.render_buffer(0)
    bounds = (80, 73, 201, 186)
    roi = comp.render_buffer(0, bounds=bounds)
    np.testing.assert_allclose(roi.rgba, complete.crop(bounds).rgba, rtol=0, atol=1e-6)
