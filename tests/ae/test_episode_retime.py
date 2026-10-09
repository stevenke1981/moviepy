"""Video shot speed / freeze and repeating chapter tags in ``EpisodeSpec``."""

from pathlib import Path

import numpy as np
from PIL import Image

import pytest

from moviepy.ae.templates.episode import (
    ChapterSpec,
    EpisodeError,
    EpisodeSpec,
    ShotSpec,
    build_episode,
)


SMALL = {"size": [64, 36], "fps": 12, "hold": 0.5}
FONT = Path("C:/Windows/Fonts/kaiu.ttf")
needs_fonts = pytest.mark.skipif(not FONT.is_file(), reason="kaiu.ttf is missing")


@pytest.fixture
def ramp_video(tmp_path):
    """A 2 s, 12 fps clip whose frame ``i`` is grey level ``10 * i``."""
    import moviepy

    frames = [np.full((36, 64, 3), 10 * i, np.uint8) for i in range(24)]
    path = tmp_path / "ramp.mp4"
    try:
        clip = moviepy.ImageSequenceClip(frames, fps=12)
        clip.write_videofile(
            str(path),
            logger=None,
            audio=False,
            ffmpeg_params=["-crf", "0", "-pix_fmt", "yuv444p"],
        )
        clip.close()
    except Exception as error:  # no ffmpeg / codec in this environment
        pytest.skip(f"cannot write a test video: {error}")
    return path


def _grey(comp, t):
    return float(comp.get_frame(t).mean())


def _close(comp):
    for clip in comp.episode_clips:
        clip.close()


def test_shot_speed_validation():
    with pytest.raises(EpisodeError, match="video shots only"):
        ShotSpec(image="a.png", duration=2, speed=0.5)
    with pytest.raises(EpisodeError, match="speed"):
        ShotSpec(video="a.mp4", speed=0)
    with pytest.raises(EpisodeError, match="later than in"):
        ShotSpec(video="a.mp4", clip_in=1, freeze_at=0.5)
    with pytest.raises(EpisodeError, match="not be later than out"):
        ShotSpec(video="a.mp4", clip_out=1, freeze_at=1.5)
    with pytest.raises(EpisodeError, match="contradicts"):
        ShotSpec(video="a.mp4", clip_in=0, clip_out=2, speed=0.5, duration=2)
    # (out - in) / speed agrees, and a freeze may run past the source.
    ShotSpec(video="a.mp4", clip_in=0, clip_out=2, speed=0.5, duration=4)
    ShotSpec(video="a.mp4", clip_in=0, clip_out=2, freeze_at=1, duration=6)
    assert ShotSpec(video="a.mp4").is_retimed is False
    assert ShotSpec(video="a.mp4", speed=2).is_retimed is True


def test_slowed_shot_plays_at_speed(ramp_video):
    spec = EpisodeSpec(
        preset_overrides=SMALL,
        shots=[{"video": str(ramp_video), "out": 1.5, "speed": 0.5}],
    )
    comp = build_episode(spec)
    try:
        # (out - in) / speed; ffmpeg builds disagree on the file's own length.
        assert comp.duration == pytest.approx(3.0)
        # Layer second 2 shows source second 1 (frame 12, grey 120).
        assert _grey(comp, 2.0) == pytest.approx(120, abs=6)
        assert _grey(comp, 1.0) == pytest.approx(60, abs=6)
    finally:
        _close(comp)


def test_freeze_holds_the_frame_after_a_late_start(tmp_path, ramp_video):
    still = tmp_path / "still.png"
    Image.fromarray(np.full((36, 64, 3), 255, np.uint8)).save(still)
    spec = EpisodeSpec(
        preset_overrides=SMALL,
        shots=[
            {"image": str(still), "duration": 1, "move": "static"},
            {"video": str(ramp_video), "in": 0.5, "freeze_at": 1.0, "duration": 3},
        ],
    )
    comp = build_episode(spec)
    try:
        assert comp.duration == pytest.approx(4.0)
        assert _grey(comp, 1.0) == pytest.approx(60, abs=6)  # source 0.5 s
        assert _grey(comp, 1.25) == pytest.approx(90, abs=6)  # source 0.75 s
        for t in (1.6, 2.5, 3.9):  # frozen on source 1.0 s
            assert _grey(comp, t) == pytest.approx(120, abs=6)
    finally:
        _close(comp)


def test_freeze_needs_no_source_beyond_the_freeze(ramp_video):
    spec = EpisodeSpec(
        preset_overrides=SMALL,
        shots=[{"video": str(ramp_video), "freeze_at": 1.5, "duration": 10}],
    )
    comp = build_episode(spec)
    try:
        assert comp.duration == pytest.approx(10)
    finally:
        _close(comp)
    too_long = EpisodeSpec(
        preset_overrides=SMALL,
        shots=[{"video": str(ramp_video), "speed": 2, "duration": 3}],
    )
    with pytest.raises(EpisodeError, match="past the end"):
        build_episode(too_long)


def test_chapter_appearances():
    plain = ChapterSpec(start=10, end=100, items=["a"], number=1)
    assert plain.appearances() == [(10, 100)]
    repeat = ChapterSpec(start=10, end=100, items=["a"], number=1, repeat_every=38)
    assert repeat.appearances() == [(10, 17.4), (48, 55.4), (86, 93.4)]
    clipped = ChapterSpec(
        start=0, end=77, items=["a"], number=1, repeat_every=38, visible=5
    )
    # The third slot would start at 76 s with only 1 s left: skipped.
    assert clipped.appearances() == [(0, 5), (38, 43)]
    short = ChapterSpec(start=0, end=1.5, items=["a"], number=1, repeat_every=38)
    assert short.appearances() == [(0, 1.5)]


def test_chapter_repeat_validation():
    with pytest.raises(EpisodeError, match="visible needs repeat_every"):
        ChapterSpec(start=0, end=9, items=["a"], number=1, visible=3)
    with pytest.raises(EpisodeError, match="must not exceed"):
        ChapterSpec(start=0, end=9, items=["a"], number=1, repeat_every=5, visible=6)
    with pytest.raises(EpisodeError, match="repeat_every"):
        ChapterSpec(start=0, end=9, items=["a"], number=1, repeat_every=-1)


@needs_fonts
def test_repeating_chapter_layers(tmp_path):
    still = tmp_path / "still.png"
    Image.fromarray(np.full((90, 160, 3), 90, np.uint8)).save(still)
    spec = EpisodeSpec(
        preset_overrides={"size": [320, 180], "fps": 12, "hold": 1.0},
        fonts={"chapter": str(FONT)},
        shots=[{"image": str(still), "duration": 30, "move": "static"}],
        chapters=[
            {
                "start": 0,
                "end": 30,
                "items": ["一"],
                "number": 1,
                "repeat_every": 12,
                "visible": 5,
            }
        ],
    )
    comp = build_episode(spec)
    try:
        names = [layer.name for layer in comp.layers if layer.name.startswith("Ch")]
        assert sorted(names) == ["Chapter 01", "Chapter 01.2", "Chapter 01.3"]
        assert comp.episode["chapters"][0]["appearances"] == [
            [0, 5],
            [12, 17],
            [24, 29],
        ]
        layer = {layer.name: layer for layer in comp.layers}["Chapter 01.2"]
        assert (layer.in_point, layer.out_point) == (12, 17)
        base = _grey(comp, 8.0)
        assert _grey(comp, 14.0) != pytest.approx(base, abs=0.01)
    finally:
        _close(comp)


def test_drift_shots_continue_one_move(tmp_path):
    image = tmp_path / "wide.png"
    ramp = np.tile(np.linspace(0, 255, 320).astype(np.uint8), (180, 1))
    Image.fromarray(np.dstack([ramp] * 3)).save(image)
    shot = {"image": str(image), "move": "drift-right", "hold": 0.5}
    spec = EpisodeSpec(
        preset_overrides=SMALL,
        shots=[
            {**shot, "duration": 2, "segment": [0, 0.5]},
            {**shot, "duration": 2, "segment": [0.5, 1]},
        ],
    )
    comp = build_episode(spec)
    # The second shot starts where the first one's move ended.
    assert (
        np.abs(
            comp.get_frame(1.99).astype(int) - comp.get_frame(2.0).astype(int)
        ).mean()
        < 3
    )
    with pytest.raises(EpisodeError, match="drift moves only"):
        ShotSpec(image="a.png", duration=2, move="push", segment=[0, 1])
    with pytest.raises(EpisodeError, match="a < b"):
        ShotSpec(image="a.png", duration=2, move="drift-left", segment=[0.6, 0.2])
