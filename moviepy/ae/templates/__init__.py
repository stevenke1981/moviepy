"""Reusable production templates built on MoviePy AE primitives.

Each template returns ordinary AE layers or a ``Composition`` configured from a
``ChannelPreset``, so results stay editable (Properties, keyframes, effects).

Examples
--------
>>> from moviepy.ae import templates
>>> sorted(templates.PRESETS)
['nightlamp_history', 'nightlamp_story']
"""

from importlib import import_module


_LAZY = {
    "ChannelPreset": "presets",
    "SubtitleStyle": "presets",
    "PRESETS": "presets",
    "get_preset": "presets",
    "MediaBackground": "background",
    "cover_crop": "background",
    "media_background": "background",
    "contrast_report": "background",
    "TitleCardSpec": "title_card",
    "fit_lines": "title_card",
    "readable_duration": "title_card",
    "build_title_card": "title_card",
    "build_bookends": "title_card",
    "Cue": "subtitles",
    "LayoutError": "subtitles",
    "SubtitleTiming": "subtitles",
    "SubtitleLayer": "subtitles",
    "break_lines": "subtitles",
    "find_bad_breaks": "subtitles",
    "check_timing": "subtitles",
    "parse_srt": "subtitles",
    "to_srt": "subtitles",
    "to_vtt": "subtitles",
    "to_ass": "subtitles",
    "subtitle_layer": "subtitles",
    "burn_subtitles": "subtitles",
    "KEN_BURNS_DEFAULTS": "ken_burns",
    "ken_burns": "ken_burns",
    "motion_smoothness": "ken_burns",
    "check_motion": "ken_burns",
    "ChapterTag": "chapter_tag",
    "chapter_tag": "chapter_tag",
    "resolve_medium": "quote",
    "split_quote_pages": "quote",
    "vertical_quote": "quote",
    "EpisodeError": "episode",
    "EpisodeSpec": "episode",
    "build_episode": "episode",
    "episode_report": "episode",
}
__all__ = list(_LAZY)


def __getattr__(name):
    """Load the requested public symbol lazily."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    """Include lazy public symbols in module introspection."""
    return sorted(set(globals()) | set(__all__))
