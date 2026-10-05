"""Generate endpoint-exact masks and composite straight-alpha video clips."""

from functools import lru_cache

import numpy as np

from moviepy.ae.motion.spec import integer
from moviepy.ae.three_d.scene import number
from moviepy.video.VideoClip import VideoClip


def transition_mask(request, *, frame_index=None, progress=None):
    """Return a float32 mask; zero selects A and one selects B.

    Use a normalized motion request. Explicit progress overrides the timeline.
    Pixel-center distances and exact endpoint branches prevent a residual rim.
    """
    if request["effect"] != "transition":
        raise ValueError("transition_mask requires a transition request")
    params = request["params"]
    if progress is None:
        index = integer(frame_index, "frame_index", 0, request["frame_count"] - 1)
        points = np.asarray(params["progress"], dtype=float)
        progress = float(np.interp(index, points[:, 0], points[:, 1]))
    progress = number(progress, "progress", 0, 1)
    width, height = request["width"], request["height"]
    if progress == 0 or progress == 1:
        return np.full((height, width), progress, np.float32)
    y, x = np.mgrid[0:height, 0:width]
    x, y = (x + 0.5) / width, (y + 0.5) / height
    direction = params["direction"]
    if params["style"] == "wipe":
        distance = {
            "left_to_right": x,
            "right_to_left": 1 - x,
            "top_to_bottom": y,
            "bottom_to_top": 1 - y,
        }[direction]
    else:
        cx, cy = params["center"]
        maximum = np.hypot(max(cx, 1 - cx) * width, max(cy, 1 - cy) * height)
        distance = np.hypot((x - cx) * width, (y - cy) * height) / maximum
        if direction == "in":
            distance = 1 - distance
    softness = params["softness"]
    if not softness:
        return (distance <= progress).astype(np.float32)
    edge = progress * (1 + softness) - softness / 2
    ramp = np.clip((edge - distance) / softness + 0.5, 0, 1)
    return (ramp * ramp * (3 - 2 * ramp)).astype(np.float32)


def transition_clip(first, second, rendered_mask):
    """Blend two same-sized local timelines using premultiplied SDR color.

    Input clips remain owned by the caller. The returned clip has no audio;
    attach the chosen soundtrack explicitly. RGB is sRGB-encoded SDR and mask
    values are linear coverage. Both input alpha masks are preserved.
    """
    if rendered_mask.metadata["effect"] != "transition":
        raise ValueError("rendered_mask must be a transition effect")
    size = tuple(rendered_mask.metadata["size"])
    duration = rendered_mask.duration
    if tuple(first.size) != size or tuple(second.size) != size:
        raise ValueError("both clips must match the transition size")
    if any(
        clip.duration is None or clip.duration < duration for clip in (first, second)
    ):
        raise ValueError("both clips must cover the transition duration")
    mask = rendered_mask.to_mask_clip()

    @lru_cache(maxsize=2)
    def sample(t):
        amount = mask.get_frame(t)[..., None]
        aa = (
            np.ones((*size[::-1], 1))
            if first.mask is None
            else first.mask.get_frame(t)[..., None]
        )
        ab = (
            np.ones((*size[::-1], 1))
            if second.mask is None
            else second.mask.get_frame(t)[..., None]
        )
        alpha = aa * (1 - amount) + ab * amount
        premultiplied = first.get_frame(t).astype(float) * aa * (1 - amount)
        premultiplied += second.get_frame(t).astype(float) * ab * amount
        rgb = np.divide(
            premultiplied, alpha, out=np.zeros_like(premultiplied), where=alpha > 0
        )
        return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8), np.clip(
            alpha[..., 0], 0, 1
        )

    clip = VideoClip(lambda t: sample(t)[0], duration=duration).with_fps(
        rendered_mask.fps
    )
    clip.mask = VideoClip(
        lambda t: sample(t)[1], is_mask=True, duration=duration
    ).with_fps(rendered_mask.fps)
    return clip
