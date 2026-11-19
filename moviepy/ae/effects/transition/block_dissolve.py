"""Block Dissolve (AE Transition category)."""

import numpy as np

from moviepy.ae.effects.base import AEEffect, Param
from moviepy.ae.effects.registry import register
from moviepy.ae.effects.transition._coverage import (
    apply_visibility,
    fully_transparent,
    random_table,
)


@register
class BlockDissolve(AEEffect):
    """Dissolve the layer away in randomly ordered rectangular blocks.

    =====================  =================  ==========================================
    Parameter              AE panel name      Notes
    =====================  =================  ==========================================
    transition_completion  Transition Compl.  0..100 %, default 0
    block_width            Block Width        px, 1..1000, default 10
    block_height           Block Height       px, 1..1000, default 10
    feather                Feather            0..1000 px soft block edge, default 0
    random_seed            Random Seed        integer 0..2**31-1, default 0
    =====================  =================  ==========================================

    Notes
    -----
    Blocks sit on a lattice anchored at world pixel coordinates
    (``src.offset``), so a block keeps the same threshold whichever region is
    rendered. Each block gets a threshold from
    ``numpy.random.default_rng(random_seed)``, indexed by a hash of its lattice
    cell; a block is hidden once completion exceeds its threshold. Feather
    fades each visible block toward its edges over the given px width. Blocks
    switch on and off hard (no temporal antialiasing); the random table has
    65536 entries, so distant blocks can share a threshold.
    """

    name = "Block Dissolve"
    category = "Transition"
    PARAMS = (
        Param("transition_completion", "float", 0.0, (0.0, 100.0)),
        Param("block_width", "float", 10.0, (1.0, 1000.0), unit="px"),
        Param("block_height", "float", 10.0, (1.0, 1000.0), unit="px"),
        Param("feather", "float", 0.0, (0.0, 1000.0), unit="px"),
        Param("random_seed", "float", 0.0, (0.0, 2147483647.0)),
    )

    def render(self, src, t, context=None, values=None):
        """Hide the blocks whose random threshold the completion has passed."""
        values = self.values_at(t, context) if values is None else values
        completion = values["transition_completion"] / 100.0
        if 0 in src.size or completion <= 0.0:
            return src
        if completion >= 1.0:
            return fully_transparent(src)
        width, height = src.size
        block_w, block_h = values["block_width"], values["block_height"]
        ox, oy = src.offset
        wx = ox + np.arange(width, dtype=np.float64)[None, :] + 0.5
        wy = oy + np.arange(height, dtype=np.float64)[:, None] + 0.5
        col = np.floor(wx / block_w).astype(np.int64)
        row = np.floor(wy / block_h).astype(np.int64)
        table = random_table(int(round(values["random_seed"])))
        index = ((col * 73856093) ^ (row * 19349663)) & 0xFFFF
        visible = (table[index] >= completion).astype(np.float32)
        feather = values["feather"]
        if feather > 0.0:
            in_x = wx - col * block_w
            in_y = wy - row * block_h
            edge = np.minimum(
                np.minimum(in_x, block_w - in_x), np.minimum(in_y, block_h - in_y)
            )
            visible = visible * np.clip(edge / feather, 0.0, 1.0).astype(np.float32)
        return apply_visibility(src, np.broadcast_to(visible, (height, width)))
