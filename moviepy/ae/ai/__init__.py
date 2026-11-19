"""Explicit local model inference and separately labeled classical tools.

No models or runtimes are installed/downloaded on import or first use.
"""

from moviepy.ae.ai.background import U2NetBackgroundRemoval
from moviepy.ae.ai.matte import inpaint_classical, refine_matte


__all__ = ["U2NetBackgroundRemoval", "refine_matte", "inpaint_classical"]
