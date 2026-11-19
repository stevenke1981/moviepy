"""Editable news-collage and archival camera-frame templates.

Use real MoviePy clips as inputs; the builders borrow them without closing or
modifying them. Stills should be ``ImageClip`` instances. All output is rendered
through the existing AE Composition/AVLayer/Mask/Transform pipeline.
"""

from moviepy.ae.templates.documentary.builders import (
    CollageCard,
    add_documentary_subtitles,
    build_film_frame,
    build_news_collage,
    build_news_page,
    film_window_masks,
)


__all__ = [
    "CollageCard",
    "add_documentary_subtitles",
    "build_film_frame",
    "build_news_collage",
    "build_news_page",
    "film_window_masks",
]
