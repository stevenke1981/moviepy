"""Mosaic (Stylize)."""

import numpy as np

from moviepy.ae.buffer import Buffer
from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register


def _block_edges(length, blocks):
    """Return (starts, lengths) splitting ``length`` pixels into ``blocks``."""
    count = min(blocks, length)
    starts = (np.arange(count, dtype=np.int64) * length) // count
    ends = np.append(starts[1:], length)
    return starts, ends - starts


@register
class Mosaic(AEEffect):
    """Pixelate the image into a grid of uniform blocks.

    ====================  ===================  ===========================
    Parameter             AE panel name        Notes
    ====================  ===================  ===========================
    horizontal_blocks     Horizontal Blocks    1..4000, default 10
    vertical_blocks       Vertical Blocks      1..4000, default 10
    sharp_colors          Sharp Colors         bool, default off
    ====================  ===================  ===========================

    Notes
    -----
    Each block takes the average of its premultiplied RGBA, which is the
    correct way to average color with alpha: transparent pixels contribute
    no color, and the block alpha is the mean coverage. Block edges are
    distributed as evenly as integer pixels allow. When ``sharp_colors`` is
    on, each block instead takes the color of its center pixel, keeping
    edge-free colors; AE's exact sharp-color algorithm is not documented, so
    this is an approximation. The block count is capped at the buffer size
    along each axis. Mosaic is not an identity at its defaults, by design.
    """

    name = "Mosaic"
    category = "Stylize"
    PARAMS = (
        Param("horizontal_blocks", "float", 10.0, (1.0, 4000.0)),
        Param("vertical_blocks", "float", 10.0, (1.0, 4000.0)),
        Param("sharp_colors", "bool", False),
    )

    def render(self, src, t, context=None, values=None):
        """Replace each block by its premultiplied average (or center sample)."""
        values = self.values_at(t, context) if values is None else values
        if 0 in src.size:
            return src
        height, width = src.size[1], src.size[0]
        x_starts, x_lengths = _block_edges(
            width, int(round(values["horizontal_blocks"]))
        )
        y_starts, y_lengths = _block_edges(
            height, int(round(values["vertical_blocks"]))
        )
        rgba = src.rgba
        if values["sharp_colors"]:
            centers_x = x_starts + x_lengths // 2
            centers_y = y_starts + y_lengths // 2
            blocks = rgba[np.ix_(centers_y, centers_x)]
        else:
            sums = np.add.reduceat(rgba, x_starts, axis=1)
            sums = np.add.reduceat(sums, y_starts, axis=0)
            counts = (y_lengths[:, None] * x_lengths[None, :])[..., None]
            blocks = (sums / counts.astype(np.float32)).astype(np.float32)
        expanded = np.repeat(blocks, y_lengths, axis=0)
        expanded = np.repeat(expanded, x_lengths, axis=1)
        out = np.ascontiguousarray(expanded, dtype=np.float32)
        return Buffer._publish(out, src.offset, src.color_space)
