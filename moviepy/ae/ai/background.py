"""Local U2Net/U2Netp ONNX foreground segmentation via existing OpenCV DNN."""

import hashlib
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import Image

from moviepy.ae.ai.matte import refine_matte
from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect


class U2NetBackgroundRemoval(AEEffect):
    """Infer foreground coverage using an explicitly supplied U2Net ONNX file.

    No default model download or opaque model selection occurs. Supply a
    licensed U2Net/U2Netp checkpoint and optionally its expected SHA-256.
    The loaded CPU network is reused and serialized across threads. Inference
    is independent per frame, not video matting or temporal stabilization.
    Input to the model is SDR RGB; original source colors are retained and
    multiplied by the inferred coverage. ``refine_radius`` enables a separate
    classical guided-filter refinement, not another learned model.
    """

    name = "AI Background Removal (U2Net)"
    category = "Keying"

    def __init__(self, model_path, *, expected_sha256=None, refine_radius=0, **kwargs):
        super().__init__(**kwargs)
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and self.model_sha256 != expected_sha256.lower():
            raise ValueError("model SHA-256 does not match the expected checkpoint")
        if (
            isinstance(refine_radius, bool)
            or not isinstance(refine_radius, int)
            or not 0 <= refine_radius <= 128
        ):
            raise ValueError("refine_radius must be an integer in 0..128")
        self._refine_radius = refine_radius
        self._lock = Lock()
        self._net = cv2.dnn.readNetFromONNX(str(path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._outputs = self._net.getUnconnectedOutLayersNames()

    def input_margin(self, size, t, context=None, values=None):
        """Request the complete backdrop for global segmentation in an ROI."""
        return (size[0], size[1], size[0], size[1])

    def apply(self, clip):
        """Return an alpha-bearing cutout clip, preserving the source audio."""
        result = super().apply(clip)
        result.transparent = True
        result.audio = clip.audio
        return result

    def predict_mask(self, frame):
        """Infer a detached float32 HxW foreground mask from uint8 RGB."""
        image = np.asarray(frame)
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or 0 in image.shape[:2]
        ):
            raise ValueError("frame must be nonempty uint8 RGB")
        resized = np.asarray(
            Image.fromarray(image).resize((320, 320), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        resized /= max(1.0, float(resized.max()))
        resized -= np.array([0.485, 0.456, 0.406], np.float32)
        resized /= np.array([0.229, 0.224, 0.225], np.float32)
        tensor = np.ascontiguousarray(resized.transpose(2, 0, 1)[None])
        with self._lock:
            self._net.setInput(tensor)
            prediction = self._net.forward(self._outputs)[0].copy()
        if prediction.shape != (1, 1, 320, 320) or not np.isfinite(prediction).all():
            raise ValueError("checkpoint output is not a finite U2Net 1x1x320x320 mask")
        mask = prediction[0, 0].copy()
        low, high = float(mask.min()), float(mask.max())
        mask = (mask - low) / (high - low) if high > low else np.zeros_like(mask)
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        mask = np.clip(mask, 0, 1).astype(np.float32, copy=False)
        if self._refine_radius:
            mask = refine_matte(image, mask, radius=self._refine_radius)
        return mask

    def render(self, src, t, context=None, values=None):
        """Apply inferred coverage without inventing color behind the subject."""
        if 0 in src.size:
            return src
        mask = self.predict_mask(src.to_uint8_rgb())
        rgba = src.rgba * mask[..., None]
        return Buffer._publish(rgba, src.offset, src.color_space)
