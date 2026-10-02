"""Float32 premultiplied rendering buffers and deterministic render contexts.

Examples
--------
>>> import numpy as np
>>> from moviepy.ae import Buffer, RenderContext
>>> frame = np.array([[[255, 0, 0]]], dtype=np.uint8)
>>> source = Buffer.from_uint8_rgb(frame, mask=np.array([[0.5]]))
>>> source.to_uint8_rgb(bg=(0, 0, 1)).tolist()
[[[128, 0, 128]]]
>>> RenderContext(t=0.5, fps=24).frame_index
12
"""

from moviepy.ae.buffer import Buffer, premultiply, unpremultiply
from moviepy.ae.context import RenderContext

__all__ = ["Buffer", "RenderContext", "premultiply", "unpremultiply"]
