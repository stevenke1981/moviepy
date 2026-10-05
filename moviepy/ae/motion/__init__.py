"""Reusable caption, 3D title, beat-particle and mask-transition effects."""

from moviepy.ae.motion.audio import analyze_audio
from moviepy.ae.motion.render import (
    MotionRender,
    motion_cache_key,
    render_motion_effect,
)
from moviepy.ae.motion.spec import validate_motion_request
from moviepy.ae.motion.transitions import transition_clip, transition_mask


__all__ = [
    "MotionRender",
    "analyze_audio",
    "motion_cache_key",
    "render_motion_effect",
    "transition_clip",
    "transition_mask",
    "validate_motion_request",
]
