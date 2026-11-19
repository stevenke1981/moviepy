"""Required episode media as an AE background: cover crop plus a live overlay.

A port of the R23 ``media_background`` step. The footage is a single still (an
image, a numpy frame, or one frame of a video held static) placed with a
``Transform`` so it *covers* the canvas, and a ``SolidLayer`` overlay whose
``opacity`` is an ordinary animatable ``Property``. Missing media always
raises; nothing is ever substituted by a plain color.

Examples
--------
>>> from moviepy.ae.templates.background import cover_crop
>>> cover_crop((200, 100), (100, 100))
(50.0, 0.0, 150.0, 100.0)
>>> cover_crop((200, 100), (100, 100), focal_point=(0.0, 0.5))
(0.0, 0.0, 100.0, 100.0)
"""

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from moviepy.ae.layers import AVLayer, SolidLayer
from moviepy.ae.transform import Transform


VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg")


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1]")
    if not (math.isfinite(value) and 0 <= value <= 1):
        raise ValueError(f"{name} must be a number in [0, 1]")
    return float(value)


def _rgb(value, name="color"):
    if isinstance(value, str):
        text = value.lstrip("#")
        if len(text) != 6:
            raise ValueError(f"{name} must be #RRGGBB")
        try:
            return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))
        except ValueError as error:
            raise ValueError(f"{name} must be #RRGGBB") from error
    items = tuple(value)
    if len(items) != 3 or not all(0 <= c <= 255 for c in items):
        raise ValueError(f"{name} must be three 0..255 RGB codes")
    return tuple(int(round(c)) for c in items)


def cover_crop(source_size, target_size, focal_point=(0.5, 0.5)):
    """Return the source box that, scaled uniformly, exactly covers the target.

    Parameters
    ----------
    source_size, target_size : tuple of int
        ``(width, height)`` in pixels.
    focal_point : tuple of float, optional
        Normalized ``(x, y)`` source point the crop is centered on; the box is
        clamped so it never leaves the source.

    Returns
    -------
    tuple of float
        ``(left, top, right, bottom)`` in source pixel-edge coordinates.

    Examples
    --------
    >>> cover_crop((100, 200), (100, 100))
    (0.0, 50.0, 100.0, 150.0)
    >>> cover_crop((100, 200), (100, 100), (0.5, 1.0))
    (0.0, 100.0, 100.0, 200.0)
    """
    width, height = source_size
    tw, th = target_size
    if min(width, height, tw, th) <= 0:
        raise ValueError("sizes must be positive")
    fx = _unit(focal_point[0], "focal_point[0]")
    fy = _unit(focal_point[1], "focal_point[1]")
    scale = max(tw / width, th / height)
    cw, ch = tw / scale, th / scale
    left = min(max(fx * width - cw / 2, 0.0), width - cw)
    top = min(max(fy * height - ch / 2, 0.0), height - ch)
    return (left, top, left + cw, top + ch)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque_rgb(array, label):
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError(f"{label} must be an HxWx3 (or opaque HxWx4) frame")
    if array.shape[2] == 4:
        if np.any(array[..., 3] != (255 if array.dtype == np.uint8 else 1)):
            raise ValueError(
                f"{label} must be opaque; transparent pixels cannot silently "
                "become a solid background"
            )
        array = array[..., :3]
    if array.dtype != np.uint8:
        if array.dtype.kind not in "fiub":
            raise ValueError(f"{label} must have a numeric dtype")
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must contain only finite values")
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _load_image(path):
    try:
        with Image.open(path) as opened:
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError(
                    "animated images are unsupported; give a still or a video"
                )
            image = ImageOps.exif_transpose(opened).convert("RGBA")
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"cannot decode image {Path(path).name}") from error
    return _opaque_rgb(np.asarray(image), "episode image")


def _video_frame(clip, frame_time):
    """Return ``(frame, used_time, info)`` for ``frame_time`` of a video clip."""
    duration = clip.duration
    if duration is None or not math.isfinite(float(duration)) or duration <= 0:
        raise ValueError("video has no finite positive duration")
    requested = 0.0 if frame_time is None else float(frame_time)
    if not math.isfinite(requested) or requested < 0:
        raise ValueError("frame_time must be a finite, non-negative number")
    if requested >= duration:
        raise ValueError(
            f"frame_time {requested} must be below the video duration {duration:.6f}s"
        )
    fps = getattr(clip, "fps", None)
    used = requested
    info = {"source_duration": float(duration)}
    if fps and math.isfinite(fps) and fps > 0:
        index = int(math.floor(requested * fps + 1e-8))
        used = min(index / fps, math.nextafter(float(duration), 0.0))
        info.update(source_frame_index=index, source_fps=float(fps))
    return _opaque_rgb(clip.get_frame(used), "video frame"), used, info, requested


@dataclass
class MediaBackground:
    """Cover-fitted still footage plus a live overlay, with provenance.

    Attributes
    ----------
    footage : AVLayer
        The held still, scaled and positioned by its ``Transform`` to cover
        ``size`` (anchor at the source origin, scale in percent).
    overlay : SolidLayer
        Constant color layer; edit ``overlay.transform.opacity`` (percent) or
        keyframe it.
    provenance : dict
        ``source_sha256`` (files), ``frame_time`` actually used, ``crop_box``,
        focal point, sizes and overlay settings.
    frame : numpy.ndarray
        The uint8 RGB still, shared by every layer set built from it.
    bake : bool
        Whether layers carry the cover crop pre-resampled to the canvas (see
        ``media_background``); baked canvas-size frames are cached per size.
    """

    footage: AVLayer
    overlay: SolidLayer
    provenance: dict
    frame: np.ndarray = field(repr=False)
    size: tuple = (1920, 1080)
    focal_point: tuple = (0.5, 0.5)
    overlay_color: tuple = (0, 0, 0)
    bake: bool = True
    _baked: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def layers(self):
        """Return ``(footage, overlay)``, bottom first (add in this order)."""
        return (self.footage, self.overlay)

    @property
    def crop_box(self):
        """Return the source crop box used for ``size``."""
        return self.provenance["crop_box"]

    def make_layers(self, size=None, duration=None):
        """Build a fresh ``(footage, overlay)`` pair for ``size``.

        A layer belongs to one composition, so intro and outro each take their
        own pair. The overlay opacity ``Property`` object is shared with this
        object's ``overlay``: keyframe it in place (``set_keyframes``) and
        every composition follows; assigning a new value to
        ``overlay.transform.opacity`` only replaces this object's own.
        """
        size = tuple(self.size if size is None else size)
        footage, overlay = _build_layers(
            self.frame,
            size,
            self.focal_point,
            self.overlay_color,
            self.overlay.transform.opacity,
            duration,
            self._baked if self.bake else None,
        )
        return footage, overlay


def _axis_scale(dest, crop):
    if dest > 1 and crop > 1:
        return (dest - 1) / (crop - 1)
    return dest / crop


def _bake_cover(frame, box, size):
    """Resample the crop ``box`` of ``frame`` once to exactly ``size`` pixels.

    The box is widened to whole source pixels (under one pixel of drift) and
    resized with an area filter when shrinking, linear when enlarging. Doing
    this once instead of inside every rendered frame keeps a 4000x3000 still
    as cheap as a 1080p one.
    """
    import cv2

    height, width = frame.shape[:2]
    x0, y0 = (
        max(0, int(math.floor(box[0] + 1e-6))),
        max(0, int(math.floor(box[1] + 1e-6))),
    )
    x1 = min(width, max(x0 + 1, int(math.ceil(box[2] - 1e-6))))
    y1 = min(height, max(y0 + 1, int(math.ceil(box[3] - 1e-6))))
    region = frame[y0:y1, x0:x1]
    shrink = size[0] <= region.shape[1] and size[1] <= region.shape[0]
    out = cv2.resize(
        region,
        (int(size[0]), int(size[1])),
        interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR,
    )
    out = np.ascontiguousarray(out)
    out.setflags(write=False)
    return out


def _build_layers(frame, size, focal_point, color, opacity, duration, bake=None):
    from moviepy import ImageClip

    height, width = frame.shape[:2]
    box = cover_crop((width, height), size, focal_point)
    if bake is not None:
        key = (tuple(size), tuple(focal_point))
        if key not in bake:
            bake[key] = _bake_cover(frame, box, size)
        clip = ImageClip(bake[key], duration=duration)
        footage = AVLayer(
            clip, "Episode media", transform=Transform(anchor_point=(0, 0))
        )
        overlay = SolidLayer("Media overlay", color=color, size=size)
        overlay.transform.opacity = opacity
        if duration is not None:
            footage.out_point = duration
            overlay.out_point = duration
        return footage, overlay
    # Align the first and last destination pixel centres with the crop's first
    # and last source pixel centres, so edge pixels sample inside the footage
    # (no half-transparent border). The per-axis difference from the exact
    # cover scale is below one source pixel.
    sx = _axis_scale(size[0], box[2] - box[0])
    sy = _axis_scale(size[1], box[3] - box[1])
    position = (-box[0] * sx, -box[1] * sy)
    clip = ImageClip(frame, duration=duration)
    footage = AVLayer(
        clip,
        "Episode media",
        transform=Transform(
            anchor_point=(0, 0),
            position=position,
            scale=(sx * 100.0, sy * 100.0),
            # Cubic rings on the hard footage edge; linear stays exact on upscales.
            interpolation="linear" if min(sx, sy) >= 1 else "auto",
        ),
    )
    overlay = SolidLayer("Media overlay", color=color, size=size)
    overlay.transform.opacity = opacity
    if duration is not None:
        footage.out_point = duration
        overlay.out_point = duration
    return footage, overlay


def media_background(
    source,
    preset,
    *,
    frame_time=None,
    focal_point=(0.5, 0.5),
    overlay_opacity=None,
    overlay_color=None,
    duration=None,
    bake=True,
):
    """Build the episode background for ``preset`` from required media.

    Parameters
    ----------
    source : str, pathlib.Path, numpy.ndarray or VideoClip
        An image path, a video path (by suffix), an RGB uint8 frame, or a
        MoviePy clip. Video contributes the single frame at ``frame_time``,
        held static for the whole duration (R23 v5 behaviour).
    preset : ChannelPreset
        Supplies ``size`` and the overlay defaults.
    frame_time : float, optional
        Seconds into a video (snapped down to its frame grid); defaults to 0.
        Not allowed for still images.
    focal_point : tuple of float, optional
        Normalized crop center.
    overlay_opacity : float, optional
        Fraction in [0, 1]; defaults to ``preset.overlay_opacity``.
    overlay_color : sequence or str, optional
        RGB codes or ``#RRGGBB``; defaults to ``preset.overlay_color``.
    duration : float, optional
        Layer length in seconds; ``None`` leaves the layers open ended.
    bake : bool, optional
        ``True`` (default) resamples the cover crop once to the canvas size, so
        the footage layer has an identity ``Transform`` and every frame only
        blits canvas-sized pixels (about 2-4x faster per 1080p frame than
        re-scaling a 2000x1200 or 4000x3000 still each time). ``False`` keeps
        the full still and covers the canvas with ``Transform.scale`` and
        ``position``, which stay editable (for example to push in on the
        footage); ``frame`` always holds the original still either way.

    Returns
    -------
    MediaBackground

    Raises
    ------
    FileNotFoundError
        If a path does not exist. There is no color fallback.
    ValueError
        For undecodable, transparent or out-of-range media.

    Examples
    --------
    >>> import numpy as np
    >>> from moviepy.ae.templates.presets import get_preset
    >>> preset = get_preset("nightlamp_story", size=(64, 36), safe_margin=(4, 4))
    >>> frame = np.full((40, 40, 3), 255, np.uint8)
    >>> bg = media_background(frame, preset)
    >>> bg.overlay.transform.opacity.value_at(0.0)
    50.0
    >>> bg.provenance["crop_box"]
    (0.0, 8.75, 40.0, 31.25)
    """
    focal = (_unit(focal_point[0], "focal_point[0]"), _unit(focal_point[1], "y"))
    opacity = _unit(
        preset.overlay_opacity if overlay_opacity is None else overlay_opacity,
        "overlay_opacity",
    )
    color = _rgb(preset.overlay_color if overlay_color is None else overlay_color)
    provenance = {"frozen_frame": True, "focal_point": list(focal)}
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"episode background not found: {path}; supply actual media"
            )
        provenance.update(
            source_name=path.name, source_sha256=_sha256(path), source_type="file"
        )
        if path.suffix.lower() in VIDEO_SUFFIXES:
            from moviepy import VideoFileClip

            try:
                with VideoFileClip(str(path), audio=False) as clip:
                    frame, used, info, requested = _video_frame(clip, frame_time)
            except OSError as error:
                raise ValueError(f"cannot decode video {path.name}") from error
            provenance.update(
                media_type="video",
                requested_frame_time=requested,
                frame_time=used,
                **info,
            )
        else:
            if frame_time:
                raise ValueError("frame_time applies only to video sources")
            frame = _load_image(path)
            provenance.update(media_type="image", frame_time=None)
    elif isinstance(source, np.ndarray):
        if frame_time:
            raise ValueError("frame_time applies only to video sources")
        frame = _opaque_rgb(source, "frame")
        provenance.update(
            source_type="array", source_sha256=None, media_type="image", frame_time=None
        )
    elif callable(getattr(source, "get_frame", None)):
        frame, used, info, requested = _video_frame(source, frame_time)
        provenance.update(
            source_type="clip",
            source_sha256=None,
            media_type="video",
            requested_frame_time=requested,
            frame_time=used,
            **info,
        )
    else:
        raise TypeError("source must be a path, a numpy frame or a VideoClip")
    size = tuple(preset.size)
    height, width = frame.shape[:2]
    box = cover_crop((width, height), size, focal)
    provenance.update(
        source_size=[width, height],
        target_size=list(size),
        crop_box=box,
        frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
        overlay={"opacity": opacity, "color": list(color)},
    )
    bake = bool(bake)
    provenance["baked"] = bake
    cache = {}
    footage, overlay = _build_layers(
        frame, size, focal, color, opacity * 100.0, duration, cache if bake else None
    )
    return MediaBackground(
        footage, overlay, provenance, frame, size, focal, color, bake, cache
    )


def _luminance(rgb):
    value = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def contrast_report(background_rgb, box, text_color, *, overlay=None, target=3.0):
    """Estimate the worst WCAG-style contrast of text inside ``box``.

    The estimate is conservative: every pixel of the bounding box counts, so
    the minimum is a lower bound for any glyph placed in it. It is not an
    accessibility certification.

    Parameters
    ----------
    background_rgb : numpy.ndarray
        ``(H, W, 3)`` uint8 pixels the text sits on (already darkened, or
        pass ``overlay`` to darken here).
    box : tuple of float
        ``(x0, y0, x1, y1)`` pixels, clamped to the image.
    text_color : sequence or str
        RGB codes or ``#RRGGBB``.
    overlay : tuple, optional
        ``(color, opacity)`` blended onto the patch first (display RGB).
    target : float, optional
        Ratio below which ``needs_visual_review`` is set.

    Returns
    -------
    dict

    Examples
    --------
    >>> import numpy as np
    >>> bg = np.zeros((10, 10, 3), np.uint8)
    >>> r = contrast_report(bg, (0, 0, 10, 10), (255, 255, 255))
    >>> round(r["minimum_estimated_contrast"], 1), r["needs_visual_review"]
    (21.0, False)
    """
    pixels = np.asarray(background_rgb)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("background_rgb must be an HxWx3 array")
    height, width = pixels.shape[:2]
    x0, y0, x1, y1 = box
    x0, x1 = max(0, int(math.floor(x0))), min(width, int(math.ceil(x1)))
    y0, y1 = max(0, int(math.floor(y0))), min(height, int(math.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("box does not intersect the background")
    patch = pixels[y0:y1, x0:x1, :3].astype(np.float64)
    if overlay is not None:
        ocolor, oopacity = _rgb(overlay[0]), _unit(overlay[1], "overlay opacity")
        patch = np.rint(patch * (1 - oopacity) + np.array(ocolor) * oopacity)
    foreground = _luminance(_rgb(text_color, "text_color"))
    back = _luminance(patch)
    ratio = (np.maximum(back, foreground) + 0.05) / (
        np.minimum(back, foreground) + 0.05
    )
    minimum = float(ratio.min())
    return {
        "minimum_estimated_contrast": minimum,
        "mean_estimated_contrast": float(ratio.mean()),
        "target": float(target),
        "needs_visual_review": minimum < target,
        "box": [x0, y0, x1, y1],
        "method": "conservative full bounding-box estimate; not a certification",
    }
