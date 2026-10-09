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
    "source_han_font": "presets",
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
    "Word": "subtitles",
    "load_words": "subtitles",
    "reflow_cues": "subtitles",
    "NameTag": "name_tag",
    "place_name_tag": "name_tag",
    "name_tag_image": "name_tag",
    "name_tag_layer": "name_tag",
    "add_name_tags": "name_tag",
    "SceneLayout": "scene_overlay",
    "scene_overlay_layers": "scene_overlay",
    "add_scene_overlays": "scene_overlay",
    "export_scene_overlay": "scene_overlay",
    "vertical_title_image": "scene_overlay",
    "cta_image": "scene_overlay",
    "watermark_image": "scene_overlay",
    "logo_image": "scene_overlay",
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
    "SceneOverlaySpec": "episode",
    "build_episode": "episode",
    "episode_report": "episode",
    "measure_loudness": "music_audio",
    "normalize_loudness": "music_audio",
    "loop_extend": "music_audio",
    "assemble_chapters": "music_audio",
    "chime_tone": "music_audio",
    "write_chime": "music_audio",
    "mix_cues": "music_audio",
    "analyze_spectrum": "spectrum",
    "SpectrumLayer": "spectrum",
    "StudySchedule": "study",
    "StudyOverlayLayer": "study",
    "seamless_loop": "visual_loop",
    "scene_cycle": "visual_loop",
    "export_loop": "visual_loop",
    "render_music_video": "music_render",
    "preview_window": "music_render",
    "MusicEpisodeError": "music_episode",
    "MusicEpisodeSpec": "music_episode",
    "build_music_episode": "music_episode",
    "init_music_episode": "music_episode",
    "validate_music_episode": "music_episode",
    "particle_loop": "ambience",
    "export_overlay_loop": "ambience",
    "LightArc": "ambience",
    "SleepFade": "ambience",
    "AssDocument": "music_cards",
    "track_cards": "music_cards",
    "progress_ring": "music_cards",
    "session_timeline": "music_cards",
    "combine_study_ass": "music_cards",
    "sleep_cards_ass": "music_cards",
    "ThumbnailSpec": "thumbnail",
    "render_thumbnail": "thumbnail",
    "grab_frame": "thumbnail",
    "flash_report": "music_qa",
    "loudness_report": "music_qa",
    "silence_report": "music_qa",
    "clipping_report": "music_qa",
    "qa_report": "music_qa",
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
