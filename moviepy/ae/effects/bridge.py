"""Run classic ``moviepy.video.fx`` effects inside an AE effect stack."""

import numpy as np

from moviepy.ae.buffer import Buffer, premultiply
from moviepy.ae.color import convert_buffer
from moviepy.ae.effects._pixels import extended_color
from moviepy.ae.effects.base import AEEffect
from moviepy.Effect import Effect


class MoviePyEffect(AEEffect):
    """Wrap a MoviePy ``Effect`` so it can sit in an ``EffectStack``.

    Parameters
    ----------
    effect : moviepy.Effect.Effect
        Any MoviePy v2 effect, for example ``vfx.InvertColors()``.
    duration : float, optional
        Duration of the temporary clip the effect sees; needed by effects
        such as ``FadeOut`` that read ``clip.duration``.
    ``**kwargs``
        ``enabled``, ``mask`` and ``blend_with_original``.

    Notes
    -----
    The effect sees a constant clip whose frame is the straight RGB of the
    input (float 0..255) with its alpha as the mask, evaluated at layer time.
    Size changes are kept, anchored at the input's offset. The bridge preserves
    linear or already signed/HDR RGB; the wrapped effect can still clip itself.
    """

    name = "MoviePy Effect"
    category = "MoviePy"

    def __init__(self, effect, *, duration=None, **kwargs):
        if not isinstance(effect, Effect) or isinstance(effect, AEEffect):
            raise TypeError("effect must be a MoviePy Effect (not an AEEffect)")
        super().__init__(**kwargs)
        self.effect = effect
        self.duration = duration

    def render(self, src, t, context=None, values=None):
        """Apply the wrapped effect to a constant clip and read frame ``t``."""
        from moviepy.video.VideoClip import VideoClip

        if 0 in src.size:
            return src
        encoded = convert_buffer(src, "srgb") if src.color_space == "linear" else src
        straight = encoded.straight()
        frame = straight[..., :3].astype(np.float64) * 255.0
        alpha = straight[..., 3].astype(np.float64)
        clip = VideoClip(lambda _: frame, duration=self.duration)
        mask = VideoClip(lambda _: alpha, is_mask=True, duration=self.duration)
        processed = clip.with_mask(mask).with_effects([self.effect])
        time = max(0.0, float(t))
        rgb = np.asarray(processed.get_frame(time), dtype=np.float32) / 255.0
        out_alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        if processed.mask is not None:
            out_alpha = np.asarray(processed.mask.get_frame(time), dtype=np.float32)
        if not extended_color(src):
            np.clip(rgb, 0.0, 1.0, out=rgb)
        rgba = np.dstack([rgb, np.clip(out_alpha, 0.0, 1.0)])
        result = Buffer._publish(premultiply(rgba), src.offset, encoded.color_space)
        return (
            convert_buffer(result, "linear") if src.color_space == "linear" else result
        )

    def __repr__(self):
        return f"MoviePyEffect({self.effect!r})"


def from_moviepy_effect(effect, **kwargs):
    """Return an ``AEEffect`` running the MoviePy ``effect`` (see ``MoviePyEffect``).

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy import vfx
    >>> from moviepy.ae import Buffer
    >>> from moviepy.ae.effects import from_moviepy_effect
    >>> source = Buffer.from_uint8_rgb(np.array([[[255, 0, 0]]], dtype=np.uint8))
    >>> inverted = from_moviepy_effect(vfx.InvertColors()).process(source, 0.0)
    >>> inverted.to_uint8_rgb().tolist()
    [[[0, 255, 255]]]
    """
    return MoviePyEffect(effect, **kwargs)
