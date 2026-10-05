"""Verify motion timing, alpha, audio events and content-addressed caches."""

import copy
import math
import os
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import pytest

from moviepy import ColorClip
from moviepy.ae.motion import (
    analyze_audio,
    motion_cache_key,
    render_motion_effect,
    transition_clip,
    transition_mask,
    validate_motion_request,
)
from moviepy.ae.motion.captions import caption_frame, layout_captions
from moviepy.ae.motion.godot_effects import build_godot_motion_scene


def request(effect="captions", **changes):
    value = {
        "effect": effect,
        "width": 160,
        "height": 96,
        "fps": 12,
        "frame_count": 12,
        "seed": 51,
    }
    if effect in ("captions", "title_card"):
        value["cues"] = [{"text": "MOTION", "start_frame": 0, "end_frame": 12}]
    if effect == "beat_particles":
        value["audio_events"] = [
            {"frame": 3, "strength": 1},
            {"frame": 8, "strength": 0.5},
        ]
    value.update(changes)
    return value


def write_pcm(path, samples, rate=48000):
    samples = np.asarray(samples, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setparams(
            (
                samples.shape[1] if samples.ndim == 2 else 1,
                2,
                rate,
                0,
                "NONE",
                "not compressed",
            )
        )
        handle.writeframes(samples.tobytes())


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": True},
        {"fps": 7},
        {"frame_count": 1.5},
        {"width": 0},
        {"cues": [{"text": "中文"}]},
        {"cues": [{"text": "abc", "start_frame": 5, "end_frame": 4}]},
        {"cues": [{"text": "abc", "unit_frames": [0, 2]}]},
        {"cues": [{"text": "abc", "unit_frames": [0, 5, 2]}]},
        {"cues": [{"text": "a", "end_frame": 8}, {"text": "b", "start_frame": 7}]},
        {"params": {"script": "x.gd"}},
        {"audio_events": [{"frame": 2, "strength": 1}]},
    ],
)
def test_motion_rejects_ambiguous_contracts(change):
    with pytest.raises((ValueError, TypeError)):
        validate_motion_request(request(**change))


def test_motion_normalization_preserves_input_and_grapheme_timing():
    original = request(cues=[{"text": "a\nbc", "start_frame": 2, "end_frame": 11}])
    before = copy.deepcopy(original)
    normalized = validate_motion_request(original)
    assert original == before
    assert normalized["cues"][0]["unit_frames"] == [2, 5, 8]


@pytest.mark.parametrize(
    "style,direction",
    [
        ("wipe", "left_to_right"),
        ("wipe", "right_to_left"),
        ("wipe", "top_to_bottom"),
        ("wipe", "bottom_to_top"),
        ("iris", "out"),
        ("iris", "in"),
    ],
)
def test_transition_has_exact_endpoints_and_monotonic_coverage(style, direction):
    value = validate_motion_request(
        request("transition", params={"style": style, "direction": direction})
    )
    np.testing.assert_array_equal(
        transition_mask(value, progress=0), np.zeros((96, 160))
    )
    np.testing.assert_array_equal(
        transition_mask(value, progress=1), np.ones((96, 160))
    )
    low, high = (transition_mask(value, progress=progress) for progress in (0.3, 0.7))
    assert np.isfinite(low).all() and np.all(low >= 0) and np.all(high <= 1)
    assert np.all(high >= low) and high.mean() > low.mean()
    assert np.any((low > 0) & (low < 1))
    if direction == "left_to_right":
        assert low[:, :10].mean() > low[:, -10:].mean()
    if direction == "right_to_left":
        assert low[:, :10].mean() < low[:, -10:].mean()


def test_progress_curve_can_reverse_without_mutating_request():
    value = validate_motion_request(
        request("transition", params={"progress": [[0, 1], [5, 0], [11, 1]]})
    )
    assert transition_mask(value, frame_index=0).min() == 1
    assert transition_mask(value, frame_index=5).max() == 0
    assert transition_mask(value, frame_index=11).min() == 1


def test_caption_highlight_keeps_positions_and_straight_alpha():
    value = validate_motion_request(
        request(
            params={
                "font_size": 20,
                "color": [1, 1, 1],
                "accent": [0.2, 1, 1],
                "stroke_width": 0,
                "position": [0.5, 0.65],
            }
        )
    )
    layout = layout_captions(value)
    first, last = (caption_frame(value, layout, frame) for frame in (0, 11))
    np.testing.assert_array_equal(first[..., 3], last[..., 3])
    assert np.any(first[..., :3] != last[..., :3])
    for box in layout[0]["boxes"]:
        assert box[0] >= 13 and box[2] <= 147 and box[1] >= 8 and box[3] <= 88
    value["params"]["style"] = "pop"
    value["params"]["accent"] = [1, 1, 1]
    popped = caption_frame(value, layout, 0)
    edge = (popped[..., 3] > 0) & (popped[..., 3] < 255)
    assert edge.any()
    np.testing.assert_array_equal(popped[..., :3][edge], np.full((edge.sum(), 3), 255))
    alpha = popped[..., 3:4] / 255
    for color in ((0, 0, 0), (255, 255, 255), (170, 30, 230)):
        composite = popped[..., :3] * alpha + np.asarray(color) * (1 - alpha)
        expected = 255 * alpha + np.asarray(color) * (1 - alpha)
        np.testing.assert_allclose(composite[edge], expected[edge])


def test_caption_cue_gaps_and_safe_area_overflow_are_explicit():
    value = validate_motion_request(
        request(cues=[{"text": "OK", "start_frame": 3, "end_frame": 8}])
    )
    layouts = layout_captions(value)
    assert not caption_frame(value, layouts, 2).any()
    assert caption_frame(value, layouts, 3)[..., 3].any()
    assert not caption_frame(value, layouts, 8).any()
    too_long = validate_motion_request(
        request(
            width=16,
            height=16,
            cues=[{"text": "cannot fit"}],
            params={"font_size": 20, "min_font_size": 20},
        )
    )
    with pytest.raises(ValueError, match="fit"):
        layout_captions(too_long)


def test_default_pop_timing_settles_all_glyphs_before_cue_end():
    value = validate_motion_request(request(params={"style": "pop", "pop_frames": 5}))
    layouts = layout_captions(value)
    popped = caption_frame(value, layouts, 11)
    value["params"]["style"] = "highlight"
    settled = caption_frame(value, layouts, 11)
    np.testing.assert_array_equal(popped, settled)


def test_energy_onsets_follow_original_pulses_and_handle_stereo_phase(tmp_path):
    rate = 48000
    samples = np.zeros(rate * 2)
    for time in (0.25, 0.75, 1.25, 1.75):
        start = round(time * rate)
        t = np.arange(2400) / rate
        samples[start : start + len(t)] = (
            12000 * np.sin(2 * np.pi * 180 * t) * np.exp(-t * 45)
        )
    source = tmp_path / "pulses.wav"
    write_pcm(source, np.column_stack((samples, -samples)))
    analyzed = analyze_audio(source, fps=24, frame_count=48)
    actual = [event["frame"] for event in analyzed["events"]]
    assert len(actual) == 4
    assert all(abs(got - wanted) <= 1 for got, wanted in zip(actual, [6, 18, 30, 42]))
    assert all(0 < event["strength"] <= 1 for event in analyzed["events"])
    write_pcm(source, np.zeros(rate))
    assert analyze_audio(source, fps=24, frame_count=24)["events"] == []
    source.write_bytes(source.read_bytes()[:-2])
    with pytest.raises(ValueError, match="truncated"):
        analyze_audio(source, fps=24, frame_count=24)


def test_cache_tracks_text_font_audio_seed_fps_size_params_and_engine(tmp_path):
    value = request()
    initial = motion_cache_key(value)
    changed = copy.deepcopy(value)
    changed["cues"][0]["text"] = "CHANGE"
    assert motion_cache_key(changed) != initial
    for key, other in (
        ("seed", 52),
        ("fps", 24),
        ("width", 192),
        ("frame_count", 24),
        ("params", {"style": "pop"}),
    ):
        assert motion_cache_key(dict(value, **{key: other})) != initial
    asset = tmp_path / "font.bin"
    asset.write_bytes(b"asset version 1")
    tracked = dict(value, assets={"font": str(asset)})
    before = motion_cache_key(tracked)
    asset.write_bytes(b"asset version 2")
    assert motion_cache_key(tracked) != before
    tracked = dict(value, assets={"audio": str(asset)})
    before = motion_cache_key(tracked)
    asset.write_bytes(b"asset version 3")
    assert motion_cache_key(tracked) != before
    gpu = request("beat_particles")
    assert motion_cache_key(gpu, engine_version="4.7.1") != motion_cache_key(
        gpu, engine_version="4.7.2"
    )


def test_cache_reuses_finished_frames_rejects_corruption_and_seeks_in_any_order(
    tmp_path, monkeypatch
):
    value = request(params={"font_size": 18, "style": "pop"})
    result = render_motion_effect(value, tmp_path)
    assert result.cache_hit is False
    import moviepy.ae.motion.render as module

    monkeypatch.setattr(
        module, "caption_frame", lambda *args: pytest.fail("cache was rerendered")
    )
    cached = render_motion_effect(value, tmp_path)
    assert cached.cache_hit and cached.frames == result.frames
    clip = cached.to_clip()
    try:
        for index in (11, 0, 7, 3, 11, 0):
            rgba = np.array(Image.open(result.frames[index]))
            np.testing.assert_array_equal(clip.get_frame(index / 12), rgba[..., :3])
            np.testing.assert_allclose(
                clip.mask.get_frame(index / 12), rgba[..., 3] / 255
            )
    finally:
        clip.close()
    Path(result.frames[3]).write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="corrupt"):
        render_motion_effect(value, tmp_path)
    assert Path(result.frames[3]).read_bytes() == b"broken"


def test_transition_compositor_preserves_source_alpha_at_endpoints_and_midpoint(
    tmp_path,
):
    result = render_motion_effect(
        request("transition", params={"progress": [[0, 0], [5, 0.5], [11, 1]]}),
        tmp_path,
    )
    first = ColorClip((160, 96), (200, 30, 10), duration=1).with_opacity(0.5)
    second = ColorClip((160, 96), (20, 40, 220), duration=1).with_opacity(0.25)
    clip = transition_clip(first, second, result)
    try:
        np.testing.assert_array_equal(clip.get_frame(0), first.get_frame(0))
        np.testing.assert_array_equal(clip.get_frame(11 / 12), second.get_frame(0))
        np.testing.assert_allclose(clip.mask.get_frame(0), 0.5)
        np.testing.assert_allclose(clip.mask.get_frame(11 / 12), 0.25)
        mask = np.array(Image.open(result.frames[5]))[..., 3:4] / 255
        alpha = 0.5 * (1 - mask) + 0.25 * mask
        expected = (
            first.get_frame(0) * 0.5 * (1 - mask) + second.get_frame(0) * 0.25 * mask
        ) / alpha
        np.testing.assert_array_equal(
            clip.get_frame(5 / 12), np.rint(expected).astype(np.uint8)
        )
        assert clip.audio is None
    finally:
        clip.close()
        first.close()
        second.close()


def test_concurrent_cache_completion_returns_one_verified_sequence(tmp_path):
    value = request("transition", width=32, height=24)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _: render_motion_effect(value, tmp_path), range(2))
        )
    assert results[0].directory == results[1].directory
    assert results[0].frames == results[1].frames
    assert render_motion_effect(value, tmp_path).cache_hit


def test_particle_builder_keeps_integer_frame_count_and_event_timing():
    for fps in (12, 24, 25, 30, 120):
        for count in (7, 13, 29):
            value = validate_motion_request(
                request(
                    "beat_particles",
                    fps=fps,
                    frame_count=count,
                    audio_events=[{"frame": count - 1}],
                )
            )
            scene = build_godot_motion_scene(value)
            assert math.ceil(scene["duration"] * fps) == count
            assert scene["particles"][0]["start"] == (count - 1) / fps


@pytest.mark.skipif(
    not os.environ.get("MOVIEPY_AE_TEST_GODOT"), reason="opt-in hidden Godot GPU"
)
def test_real_motion_particles_match_events_and_cached_nonsequential_seeks(tmp_path):
    value = request("beat_particles", params={"count": 50, "size": 0.06})
    result = render_motion_effect(
        value, tmp_path, godot=os.environ["MOVIEPY_AE_TEST_GODOT"]
    )
    assert result.metadata["godot"]["burst_start_frames"] == [3, 8]
    clip = result.to_clip()
    try:
        assert not clip.mask.get_frame(0).any()
        assert clip.mask.get_frame(7 / 12).any()
        late = clip.get_frame(11 / 12).copy()
        clip.get_frame(0)
        np.testing.assert_array_equal(clip.get_frame(11 / 12), late)
    finally:
        clip.close()
    assert render_motion_effect(
        value, tmp_path, godot=os.environ["MOVIEPY_AE_TEST_GODOT"]
    ).cache_hit


@pytest.mark.skipif(
    not os.environ.get("MOVIEPY_AE_TEST_GODOT")
    or not os.environ.get("MOVIEPY_AE_TEST_CJK_FONT"),
    reason="opt-in hidden Godot GPU and explicitly supplied CJK font",
)
def test_real_chinese_title_card_and_caption_layout_stay_inside_safe_area(tmp_path):
    font = os.environ["MOVIEPY_AE_TEST_CJK_FONT"]
    cues = [{"text": "節奏有形", "start_frame": 0, "end_frame": 12}]
    value = request(
        "title_card", width=256, height=144, assets={"font": font}, cues=cues
    )
    result = render_motion_effect(
        value, tmp_path, godot=os.environ["MOVIEPY_AE_TEST_GODOT"]
    )
    for path in result.frames:
        alpha = np.array(Image.open(path))[..., 3]
        y, x = np.where(alpha > 0)
        assert len(x) > 100
        assert x.min() >= 30 and x.max() < 226 and y.min() >= 17 and y.max() < 127
    captions = validate_motion_request(
        request(
            assets={"font": font},
            cues=[{"text": "逐字高亮\n穩定排版"}],
            params={"font_size": 22, "max_lines": 2},
        )
    )
    layouts = layout_captions(captions)
    a, b = caption_frame(captions, layouts, 0), caption_frame(captions, layouts, 11)
    np.testing.assert_array_equal(a[..., 3], b[..., 3])
    assert a[..., 3].any()
