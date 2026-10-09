"""Tests for the ambience components: particles, light arc, sleep fade."""

import json
import math
import subprocess

import numpy as np

import pytest

from moviepy.ae.templates.ambience import (
    PARTICLE_KINDS,
    LightArc,
    ParticleLayer,
    SleepFade,
    _crystal_sprite,
    _particle_factory,
    export_overlay_loop,
    particle_loop,
)
from moviepy.config import FFMPEG_BINARY


SIZE = (160, 90)
FPS = 8
PERIOD = 3.0
FRAMES = 24


def layer_for(kind, **kw):
    options = dict(size=SIZE, period=PERIOD, fps=FPS, seed=3)
    options.update(kw)
    return particle_loop(kind, **options).particle_layer


def ffmpeg_means(vf, seconds, rate=1, size="32x18", color="0x808080"):
    """Return per-frame mean RGB of a flat colour passed through ``vf``."""
    run = subprocess.run(
        [FFMPEG_BINARY, "-v", "error", "-f", "lavfi", "-i",
         f"color=c={color}:s={size}:r={rate}:d={seconds}", "-vf", vf,
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True,
    )  # fmt: skip
    assert run.returncode == 0, run.stderr.decode()[-400:]
    w, h = (int(v) for v in size.split("x"))
    frames = np.frombuffer(run.stdout, np.uint8).reshape(-1, h * w, 3)
    return frames.mean(axis=1)


# --------------------------------------------------------------------------- #
# particles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", PARTICLE_KINDS)
def test_particles_are_exactly_periodic(kind):
    layer = layer_for(kind)
    assert layer.frames == FRAMES
    for n in (0, 5, FRAMES - 1):
        assert np.array_equal(layer.rgba_uint8(n), layer.rgba_uint8(n + FRAMES))
    comp = particle_loop(kind, size=SIZE, period=PERIOD, fps=FPS, seed=3)
    assert comp.duration == PERIOD
    assert np.array_equal(comp.fast_rgba(0.0), comp.fast_rgba(PERIOD))
    assert np.array_equal(comp.fast_rgba(1.0), comp.fast_rgba(1.0 + 2 * PERIOD))


@pytest.mark.parametrize("kind", PARTICLE_KINDS)
def test_particles_are_deterministic_by_seed(kind):
    a, b = layer_for(kind), layer_for(kind)
    other = layer_for(kind, seed=4)
    assert np.array_equal(a.rgba_uint8(7), b.rgba_uint8(7))
    assert not np.array_equal(a.rgba_uint8(7), other.rgba_uint8(7))


@pytest.mark.parametrize("kind", PARTICLE_KINDS)
def test_each_kind_has_alpha_and_stays_in_region(kind):
    region = (30, 20, 80, 40)
    layer = layer_for(kind, region=region, count=24)
    x, y, w, h = region
    seen = 0
    for n in range(FRAMES):
        frame = layer.rgba_uint8(n)
        alpha = frame[..., 3]
        seen = max(seen, int(alpha.max()))
        outside = alpha.copy()
        outside[y : y + h, x : x + w] = 0
        assert outside.max() == 0
        visible = alpha > 0
        assert (frame[..., :3][visible] == np.array(layer.color)).all()
    assert seen > 20


def test_composition_path_matches_fast_path():
    comp = particle_loop("snow", size=SIZE, period=PERIOD, fps=FPS, seed=1)
    fast = comp.fast_rgba(1.0)
    buffer = comp.render_buffer(1.0)
    alpha = np.rint(buffer.rgba[..., 3] * 255)
    assert np.abs(alpha - fast[..., 3]).max() <= 1
    assert comp.transparent


def test_no_flashing_and_gentle_twinkle():
    for kind in PARTICLE_KINDS:
        layer = layer_for(kind, size=(320, 180), count=40)
        means = np.array(
            [layer.rgba_uint8(n)[..., 3].mean() for n in range(FRAMES + 1)]
        )
        steps = np.abs(np.diff(means))
        assert steps.max() <= 0.2 * max(means.mean(), 1.0) + 0.3, kind
    fire = particle_loop("fireflies", period=20.0).particle_layer
    assert float(fire._kt.max()) / fire.period <= 0.5
    with pytest.raises(ValueError):
        particle_loop("fireflies", size=SIZE, period=1.0, fps=FPS)


def test_particle_validation():
    with pytest.raises(ValueError):
        particle_loop("lasers", size=SIZE, period=PERIOD, fps=FPS)
    with pytest.raises(ValueError):
        particle_loop("snow", size=SIZE, period=3.1, fps=FPS)
    with pytest.raises(ValueError):
        ParticleLayer("snow", size=SIZE, period=PERIOD, fps=FPS, region=(0, 0, 999, 9))
    assert layer_for("snow", count=0).rgba_uint8(0)[..., 3].max() == 0


@pytest.mark.parametrize("codec", ["qtrle", "png", "prores_4444"])
def test_export_overlay_loop_alpha_round_trip(tmp_path, codec):
    comp = particle_loop("fireflies", size=SIZE, period=PERIOD, fps=FPS, count=12)
    path = tmp_path / f"{codec}.mov"
    report = export_overlay_loop(comp, path, codec=codec)
    assert path.is_file() and (tmp_path / f"{codec}.mov.json").is_file()
    assert report["frames"] == FRAMES and report["size"] == list(SIZE)
    assert report["alpha"]["max"] > 20 and report["alpha"]["mean_covered_share"] < 0.5
    assert report["seam"]["frame0_equals_frame_period"] is True
    assert report["human_visual"] == "NOT_RUN"
    trip = report["roundtrip"]
    assert trip["frames_decoded"] == FRAMES and trip["frames_match"]
    if codec == "prores_4444":
        assert trip["alpha_max_abs_diff"] <= 2
    else:
        assert trip["alpha_max_abs_diff"] == 0
        assert trip["rgb_max_abs_diff_where_visible"] <= 1
    saved = json.loads((tmp_path / f"{codec}.mov.json").read_text("utf-8"))
    assert saved["frames_sha256"] == report["frames_sha256"]


def test_export_overlay_loop_exclusive_and_workers(tmp_path):
    comp = particle_loop("dust", size=SIZE, period=PERIOD, fps=FPS, count=10, seed=2)
    first = export_overlay_loop(comp, tmp_path / "a.mov")
    with pytest.raises(FileExistsError):
        export_overlay_loop(comp, tmp_path / "a.mov")
    options = tuple(
        sorted(dict(size=SIZE, period=PERIOD, fps=FPS, count=10, seed=2).items())
    )
    again = export_overlay_loop(
        (_particle_factory, ("dust", options)), tmp_path / "b.mov", workers=2
    )
    assert again["frames_sha256"] == first["frames_sha256"]
    export_overlay_loop(comp, tmp_path / "a.mov", overwrite=True)
    with pytest.raises(ValueError):
        export_overlay_loop(comp, tmp_path / "c.mov", codec="gif")


# --------------------------------------------------------------------------- #
# light arc
# --------------------------------------------------------------------------- #


def test_light_arc_value_at_and_presets():
    arc = LightArc(
        [
            (10, {"brightness": 0.0, "warmth": 0.2}),
            (110, {"brightness": -0.2, "saturation": 0.8, "warmth": 0.6}),
        ]
    )
    assert arc.value_at(0) == arc.value_at(10)
    mid = arc.value_at(60)
    assert mid["brightness"] == pytest.approx(-0.1)
    assert mid["saturation"] == pytest.approx(0.9)
    assert mid["contrast"] == 1.0
    assert arc.value_at(1e6)["warmth"] == pytest.approx(0.6)
    # cosine ease: zero slope at the keys, fastest in the middle
    assert (
        abs(arc.value_at(11)["brightness"]) < abs(arc.value_at(60)["brightness"]) / 50
    )
    for make in (LightArc.day_to_night, LightArc.dusk, LightArc.dawn):
        preset = make(7200)
        assert preset.keyframes[0][0] == 0 and preset.keyframes[-1][0] == 7200
        assert max(preset.max_rates().values()) <= 0.1
        assert preset.to_ffmpeg_filter().count("eq=") == 1
    night = LightArc.day_to_night(3600)
    assert night.value_at(3600)["brightness"] < night.value_at(0)["brightness"]
    assert night.value_at(3600)["warmth"] < 0 < night.value_at(1000)["warmth"]


def test_light_arc_validation():
    with pytest.raises(ValueError):
        LightArc([])
    with pytest.raises(ValueError):
        LightArc([(5, {}), (5, {})])
    with pytest.raises(ValueError):
        LightArc([(0, {"hue": 1})])
    with pytest.raises(ValueError):
        LightArc([(0, {"brightness": 2.0})])
    with pytest.raises(ValueError):
        LightArc([(0, {"brightness": 0}), (2, {"brightness": -0.5})])  # too fast
    LightArc([(0, {"brightness": 0}), (2, {"brightness": -0.5})], max_rate=None)


def test_light_arc_matches_ffmpeg_frames():
    arc = LightArc(
        [
            (0, {}),
            (4, {"brightness": -0.2, "saturation": 0.7, "warmth": 0.8}),
            (8, {"brightness": 0.05, "contrast": 1.2, "warmth": -0.6}),
            (12, {"brightness": -0.1, "saturation": 1.3}),
        ],
        max_rate=None,
    )
    times = [0, 2, 4, 6, 8, 10, 12, 13]
    got = ffmpeg_means(arc.to_ffmpeg_filter(), 14, color="0x808080")
    for t in times:
        v = arc.value_at(t)
        reference = ffmpeg_means(
            "eq=brightness={brightness}:contrast={contrast}:saturation={saturation}"
            ":gamma_r={gr}:gamma_b={gb}".format(
                gr=1 + 0.25 * v["warmth"], gb=1 - 0.25 * v["warmth"], **v
            ),
            1,
            color="0x808080",
        )[0]
        assert np.abs(got[t] - reference).max() <= 3, (t, got[t], reference)
    # the arc really changes the picture, and warmth moves red against blue
    assert abs(got[4].mean() - got[0].mean()) > 15
    assert (got[4][0] - got[4][2]) > (got[0][0] - got[0][2]) + 8
    assert (got[8][0] - got[8][2]) < (got[0][0] - got[0][2]) - 8


@pytest.mark.parametrize("count", [24, 40])
def test_light_arc_many_keyframes_accepted_by_ffmpeg(count):
    keys = [
        (
            i * 5.0,
            {"brightness": -0.1 * (i % 2), "warmth": 0.5 * ((i // 2) % 2)},
        )
        for i in range(count)
    ]
    arc = LightArc(keys, max_rate=None)
    expr = arc.to_ffmpeg_filter()
    assert len(expr) < 400 * count
    got = ffmpeg_means(expr, 3)
    assert got.shape[0] == 3 and np.isfinite(got).all()
    # last segment value at its own key
    later = ffmpeg_means(expr, 1, rate=1)
    assert later.shape[0] == 1


# --------------------------------------------------------------------------- #
# sleep fade
# --------------------------------------------------------------------------- #


def test_sleep_fade_video_monotonic_and_reaches_floor():
    fade = SleepFade(2, 10, floor=0.25, max_rate=None)
    luma = ffmpeg_means(fade.video_filter(), 14, color="0xc8c8c8").mean(axis=1)
    assert luma[0] == pytest.approx(luma[2], abs=1)  # held before start
    assert (np.diff(luma[2:11]) <= 0.5).all() and luma[2] - luma[10] > 40
    assert (np.diff(luma[2:11]) < -0.5).sum() >= 6
    assert luma[10] == pytest.approx(luma[13], abs=1)  # held after end
    # level at the floor is about floor * start level (limited-range YUV math)
    assert 0.22 < luma[13] / luma[0] < 0.29
    black = ffmpeg_means(SleepFade(1, 4, max_rate=None).video_filter(), 6)
    assert black[5].max() < 4


def test_sleep_fade_gain_audio_and_validation():
    fade = SleepFade(100, 200, floor=0.1)
    assert fade.gain_at(100) == 1.0 and fade.gain_at(200) == pytest.approx(0.1)
    assert fade.gain_at(150) == pytest.approx(0.55) and fade.gain_at(500) == 0.1
    gains = [fade.gain_at(t) for t in range(90, 210, 5)]
    assert all(a >= b for a, b in zip(gains, gains[1:]))
    fade.validate(300)
    with pytest.raises(ValueError):
        fade.validate(150)
    with pytest.raises(ValueError):
        SleepFade(10, 10)
    with pytest.raises(ValueError):
        SleepFade(10, 20)  # 16 s fade is faster than 0.1/s
    with pytest.raises(ValueError):
        SleepFade(0, 100, floor=1.0)
    assert SleepFade(0, 100, audio=False).audio_filter() is None
    assert "volume=eval=frame" in fade.audio_filter()
    assert json.dumps(fade.describe())


def test_sleep_fade_audio_filter_attenuates(tmp_path):
    fade = SleepFade(1, 5, floor=0.0, max_rate=None)
    t = np.arange(48000 * 7) / 48000
    pcm = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2").tobytes()
    out = subprocess.run(
        [FFMPEG_BINARY, "-v", "error", "-f", "s16le", "-ar", "48000", "-ac", "1",
         "-i", "-", "-af", fade.audio_filter(), "-f", "s16le", "-"],
        input=pcm, capture_output=True, check=True,
    ).stdout  # fmt: skip
    samples = np.frombuffer(out, np.int16).astype(np.float64)
    rms = [
        np.sqrt((samples[i * 48000 : (i + 1) * 48000] ** 2).mean()) for i in range(7)
    ]
    assert rms[0] > 10000 and rms[0] == pytest.approx(rms[1], rel=0.05)
    assert rms[1] > rms[2] > rms[3] > rms[4]
    assert rms[5] < 50 and rms[6] < 50


def test_snowflakes_are_six_armed_crystals():
    sprite, half = _crystal_sprite(8.0, 0.0, 0.0, 0.0)
    turned, _ = _crystal_sprite(8.0, math.pi / 3, 0.0, 0.0)
    assert np.abs(sprite - turned).max() < 1e-3  # six-fold symmetry
    assert sprite[half, half + 7] > 0.5  # on an arm
    assert sprite[half + 7, half] < 0.4 * sprite[half, half + 7]  # between arms
    layer = layer_for("snowflakes", count=6)
    assert layer.rgba_uint8(0)[..., 3].max() > 0
