"""Music episode spec, resumable steps, tiny end-to-end builds and the CLI."""

import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import pytest

from moviepy import VideoClip
from moviepy.ae.templates import music_episode as me, soundx as sx
from moviepy.ae.templates._audio_io import AudioWriter, audio_info
from moviepy.ae.templates.music_episode import (
    MusicEpisodeError,
    MusicEpisodeSpec,
    build_music_episode,
    music_paths,
    pomodoro_schedule,
    prepare_audio,
    validate_music_episode,
)
from moviepy.config import FFMPEG_BINARY
from tests.ae._fake_soundx import install_fake_soundx


FPS = 12
SIZE = (160, 90)
ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "moviepy" / "ae" / "templates" / "configs"


@pytest.fixture(scope="module", autouse=True)
def fake_soundx(tmp_path_factory):
    """Run every build in this module against a fake soundx 0.2.0 (CI has none)."""
    launcher = install_fake_soundx(tmp_path_factory.mktemp("fake_soundx"), "0.2.0")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MOVIEPY_SOUNDX", str(launcher))
        yield launcher


def tone(path, seconds, hertz, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * 48000)) / 48000
    wave = (
        0.25 * np.sin(2 * np.pi * hertz * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.7 * t))
    )
    wave = wave + 0.02 * rng.standard_normal(len(t))
    data = np.stack([wave, 0.9 * wave], axis=1).astype(np.float32)
    with AudioWriter(path, rate=48000, channels=2, subtype="PCM_16") as writer:
        writer.write(data)


def clip_file(path, phase):
    def frame(t):
        x = np.arange(96)[None, :, None]
        y = np.arange(54)[:, None, None]
        wave = np.sin(0.2 * x + 0.1 * y + 1.5 * t + phase + np.array([0, 2, 4]))
        return np.broadcast_to(127 + 120 * wave, (54, 96, 3)).astype(np.uint8)

    clip = VideoClip(frame, duration=3.0)
    clip.write_videofile(str(path), fps=FPS, codec="libx264", audio=False, logger=None)


def base_config(mode, folder):
    (folder / "tracks").mkdir()
    (folder / "visual").mkdir()
    tone(folder / "tracks" / "a.wav", 3.0, 330, 1)
    tone(folder / "tracks" / "b.wav", 3.0, 494, 2)
    clip_file(folder / "visual" / "c1.mp4", 0.0)
    clip_file(folder / "visual" / "c2.mp4", 2.0)
    config = {
        "mode": mode,
        "name": "tiny",
        "size": list(SIZE),
        "fps": FPS,
        "tracks": [
            {"id": "A", "source": "tracks/a.wav", "title": "First = One"},
            {"id": "B", "source": "tracks/b.wav", "title": "Second"},
        ],
        "audio": {
            "loop_overlap": 0.5,
            "chapter_crossfade": 1.0,
            "edge_fade": 0.5,
        },
        "visual": {
            "clips": ["visual/c1.mp4", "visual/c2.mp4"],
            "clip_length": 2.0,
            "loop_crossfade": 0.5,
            "segment": 2.0,
            "scene_crossfade": 0.5,
        },
        "encoder": "libx264",
        "quality": "draft",
        "workers": 2,
        "output_dir": "build",
    }
    return config


@pytest.fixture(scope="module")
def sleep_spec(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sleep")
    config = base_config("sleep_longform", folder)
    config["tracks"][0]["chapter_seconds"] = 4.0
    config["tracks"][1]["chapter_seconds"] = 4.0
    path = folder / "music.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return MusicEpisodeSpec.from_json(path)


@pytest.fixture(scope="module")
def study_spec(tmp_path_factory):
    folder = tmp_path_factory.mktemp("study")
    config = base_config("study_pomodoro", folder)
    config["study"] = pomodoro_schedule(
        "Tiny; study", focus=2, rest=2, rounds=2, closing=2
    )
    config["spectrum"] = {
        "position": [8, 56],
        "size": [144, 28],
        "bands": 16,
        "fft": 1024,
        "bar_width": 6,
        "pitch": 9,
    }
    config["chime"] = {
        "design": {
            "notes": [["A4", 440.0, 0.0, 0.4]],
            "duration": 0.5,
            "attack": 0.05,
            "release": 0.2,
        }
    }
    path = folder / "music.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return MusicEpisodeSpec.from_json(path)


@pytest.fixture(scope="module")
def sleep_report(sleep_spec):
    return build_music_episode(sleep_spec)


@pytest.fixture(scope="module")
def study_report(study_spec):
    return build_music_episode(study_spec)


def ffmpeg_info(path):
    return subprocess.run(
        [FFMPEG_BINARY, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    ).stderr


def frames_of(path):
    run = subprocess.run(
        [FFMPEG_BINARY, "-v", "error", "-i", str(path), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    )  # fmt: skip
    return np.frombuffer(run.stdout, np.uint8).reshape(-1, SIZE[1], SIZE[0], 3)


def duration_of(path):
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", ffmpeg_info(path)).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def hashes(spec):
    root = Path(spec.output_dir)
    return {
        str(p): me._sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.name.endswith(".log")
    }


# ---- spec ------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", me.MODES)
def test_shipped_configs_round_trip(mode, tmp_path):
    spec = MusicEpisodeSpec.from_json(CONFIGS / f"music_{mode}.json")
    assert MusicEpisodeSpec.from_dict(spec.to_dict()) == spec
    path = tmp_path / "again.json"
    spec.to_json(path)
    assert MusicEpisodeSpec.from_json(path) == spec
    assert Path(spec.tracks[0].source).is_absolute()
    assert spec.size == (1280, 720) and spec.fps == 24
    assert (spec.audio.target_lufs, spec.audio.true_peak) == (-18.0, -1.8)
    assert (spec.audio.loop_overlap, spec.audio.chapter_crossfade) == (10.0, 12.0)


def test_shipped_config_lengths_and_defaults():
    sleep = MusicEpisodeSpec.from_json(CONFIGS / "music_sleep_longform.json")
    plan = validate_music_episode(sleep, check_files=False)["plan"]
    assert plan["total_seconds"] == 19200.0 and plan["video_frames"] == 460800
    assert not sleep.spectrum.enabled and not sleep.chime.enabled
    study = MusicEpisodeSpec.from_json(CONFIGS / "music_study_pomodoro.json")
    inline = dataclasses.replace(study, study=pomodoro_schedule("S"))
    plan = validate_music_episode(inline, check_files=False)["plan"]
    assert plan["total_seconds"] == 5220.0 and plan["video_frames"] == 125280
    assert study.spectrum.enabled and study.spectrum.position == (96, 548)
    assert study.chime.enabled and study.chime.below_music_db == 10.0


def test_minimal_dict_gets_mode_defaults():
    track = [{"id": "A", "source": "a.wav", "chapter_seconds": 60}]
    sleep = MusicEpisodeSpec.from_dict(
        {"mode": "sleep_longform", "tracks": track, "visual": {"cycle_path": "c.mp4"}}
    )
    assert (sleep.spectrum.enabled, sleep.chime.enabled) == (False, False)
    study = MusicEpisodeSpec.from_dict(
        {
            "mode": "study_pomodoro",
            "tracks": track,
            "visual": {"cycle_path": "c.mp4"},
            "study": pomodoro_schedule(),
        }
    )
    assert (study.spectrum.enabled, study.chime.enabled) == (True, True)
    assert study.schedule().duration == 5220


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"mode": "party"}, "mode must be one of"),
        ({"size": [1281, 720]}, "size must be even"),
        ({"fps": 0}, "fps must be positive"),
        ({"fps": "24"}, "fps must be a number"),
        ({"tracks": []}, "tracks must be a non-empty list"),
        ({"tracks": [{"id": "A"}]}, r"tracks\[0\]: missing required key"),
        ({"tracks": [{"id": "A", "source": "a", "x": 1}]}, "unknown key"),
        (
            {"tracks": [{"id": "a b", "source": "a", "chapter_seconds": 60}]},
            "file-name",
        ),
        ({"audio": {"edge_fade": 20}}, "edge_fade must be <= chapter_crossfade"),
        ({"audio": {"lufs": -18}}, "audio: unknown key"),
        ({"audio": {"chapter_crossfade": 40}}, "at least twice"),
        ({"visual": {"clips": []}}, "give clips or a prepared cycle_path"),
        ({"visual": {"clips": ["c"], "clip_length": 3}}, "clip_length must exceed"),
        ({"visual": {"clips": ["c"], "segment": 120.01}}, "whole number of frames"),
        ({"spectrum": {"enabled": True, "position": [900, 600]}}, "exceeds"),
        ({"spectrum": {"fmin": 20000}}, "fmin must be below fmax"),
        ({"chime": {"enabled": True}}, "needs study_pomodoro"),
        ({"study": {}}, "only valid in study_pomodoro"),
        ({"encoder": "x264"}, "encoder must be one of"),
        ({"audio_bitrate": "fast"}, "audio_bitrate"),
        ({"workers": 0}, "workers must be >= 1"),
        ({"bogus": 1}, "unknown music episode key"),
    ],
)
def test_validation_errors_name_the_field(patch, message):
    data = {
        "mode": "sleep_longform",
        "tracks": [{"id": "A", "source": "a.wav", "chapter_seconds": 60}],
        "visual": {"clips": ["c.mp4"]},
    }
    data.update(patch)
    with pytest.raises(MusicEpisodeError, match=message):
        MusicEpisodeSpec.from_dict(data)


def test_study_validation_errors():
    base = {
        "mode": "study_pomodoro",
        "tracks": [{"id": "A", "source": "a.wav"}],
        "visual": {"clips": ["c.mp4"]},
    }
    with pytest.raises(MusicEpisodeError, match="study is required"):
        MusicEpisodeSpec.from_dict(base)
    broken = pomodoro_schedule()
    broken["duration_seconds"] += 1
    with pytest.raises(MusicEpisodeError, match="study:"):
        MusicEpisodeSpec.from_dict(dict(base, study=broken))
    mixed = dict(base, study=pomodoro_schedule())
    mixed["tracks"] = [
        {"id": "A", "source": "a", "chapter_seconds": 100},
        {"id": "B", "source": "b"},
    ]
    with pytest.raises(MusicEpisodeError, match="on every track or on none"):
        MusicEpisodeSpec.from_dict(mixed)
    cue = dict(base, study=pomodoro_schedule(), chime={"path": "x", "design": {"a": 1}})
    with pytest.raises(MusicEpisodeError, match="chime"):
        MusicEpisodeSpec.from_dict(cue)


def test_audio_length_must_match_schedule_and_frames(tmp_path):
    data = {
        "mode": "study_pomodoro",
        "fps": 24,
        "tracks": [
            {"id": "A", "source": "a.wav", "chapter_seconds": 3000},
            {"id": "B", "source": "b.wav", "chapter_seconds": 3000},
        ],
        "visual": {"clips": ["c.mp4"]},
        "study": pomodoro_schedule(),
    }
    spec = MusicEpisodeSpec.from_dict(data)
    with pytest.raises(MusicEpisodeError, match="5220"):
        me._audio_plan(spec)
    sleep = MusicEpisodeSpec.from_dict(
        {
            "mode": "sleep_longform",
            "fps": 24,
            "tracks": [{"id": "A", "source": "a", "chapter_seconds": 100.01}],
            "visual": {"clips": ["c.mp4"]},
        }
    )
    with pytest.raises(MusicEpisodeError, match="whole number of frames"):
        me._audio_plan(sleep)
    result = validate_music_episode(sleep, check_files=False)
    assert not result["ok"] and result["problems"]


def test_validate_reports_missing_files(sleep_spec, tmp_path):
    assert validate_music_episode(sleep_spec)["ok"]
    moved = dataclasses.replace(
        sleep_spec,
        tracks=(dataclasses.replace(sleep_spec.tracks[0], source=str(tmp_path / "n")),)
        + sleep_spec.tracks[1:],
    )
    result = validate_music_episode(moved)
    assert not result["ok"] and result["missing"] == [str(tmp_path / "n")]


# ---- end to end: sleep -------------------------------------------------------- #


def test_sleep_build_outputs(sleep_spec, sleep_report):
    steps = sleep_report["steps"]
    assert set(steps) == {"audio", "visual", "overlays", "render"}  # new steps off
    assert sleep_report["human_gates"] == {
        "full_listening": "NOT_RUN",
        "full_visual_watch": "NOT_RUN",
        "rights": "NOT_RUN",
        "private_upload": "NOT_RUN",
    }
    paths = music_paths(sleep_spec)
    assert paths["chime"] is None and paths["levels"] is None and paths["ass"] is None
    info = audio_info(paths["audio"])
    assert info["frames"] == 7 * 48000  # 4 + 4 - 1 s crossfade
    assert steps["audio"]["frames"] == info["frames"]
    assert steps["audio"]["music"]["human_listening"] == "NOT_RUN"
    video = Path(paths["video"])
    count = len(frames_of(video))
    assert count == 7 * FPS == steps["render"]["frames"]
    assert abs(duration_of(video) - 7.0) < 0.1
    text = ffmpeg_info(video)
    assert "Chapter #0:0" in text and "Chapter #0:1" in text
    assert "Audio: aac" in text and "160x90" in text
    assert steps["render"]["human_visual"] == "NOT_RUN"
    assert steps["render"]["overlay"] is None


def test_sleep_chapters_and_evidence(sleep_spec, sleep_report):
    paths = music_paths(sleep_spec)
    text = Path(paths["chapters"]).read_text(encoding="utf-8")
    assert text.count("[CHAPTER]") == 2
    assert "START=0\nEND=3500" in text and "START=3500\nEND=7000" in text
    assert "title=First \\= One" in text
    for key in ("music", "cycle", "chapters", "video"):
        evidence = json.loads(
            Path(str(paths[key]) + ".json").read_text(encoding="utf-8")
        )
        assert evidence["sha256" if key != "video" else "output_sha256"]
    for master in paths["masters"]:
        record = json.loads(Path(master + ".json").read_text(encoding="utf-8"))
        assert record["output"]["frames"] == 3 * 48000
        assert record["human_listening"] == "NOT_RUN"
    cycle = json.loads(Path(paths["cycle"] + ".json").read_text(encoding="utf-8"))
    assert cycle["frames"] == 4 * FPS and cycle["human_visual"] == "NOT_RUN"
    assert not list(Path(sleep_spec.output_dir).rglob("*.partial*"))


def test_resume_skips_and_keeps_hashes(sleep_spec, sleep_report):
    before = hashes(sleep_spec)
    again = build_music_episode(sleep_spec)
    steps = again["steps"]
    assert steps["audio"]["music"]["skipped"]
    assert all(m["skipped"] for m in steps["audio"]["masters"])
    assert steps["visual"]["cycle"]["skipped"]
    assert all(loop["skipped"] for loop in steps["visual"]["loops"])
    assert steps["overlays"]["chapters"]["skipped"]
    assert steps["render"]["skipped"]
    assert hashes(sleep_spec) == before
    assert (
        steps["render"]["output_sha256"]
        == sleep_report["steps"]["render"]["output_sha256"]
    )


def test_preview_window(sleep_spec, sleep_report):
    report = build_music_episode(sleep_spec, steps=["render"], preview=(1.0, 2.0))
    out = Path(report["steps"]["render"]["output"])
    assert out.name == "tiny-preview-1-2.mp4"
    assert len(frames_of(out)) == 2 * FPS
    again = build_music_episode(sleep_spec, steps=("render",), preview=(1.0, 2.0))
    assert again["steps"]["render"]["skipped"]
    with pytest.raises(MusicEpisodeError, match="after the"):
        build_music_episode(sleep_spec, steps=["render"], preview=(6.0, 2.0))
    with pytest.raises(MusicEpisodeError, match="whole number of frames"):
        build_music_episode(sleep_spec, steps=["render"], preview=(0.0, 0.05))


def test_exclusive_create_and_tamper_detection(sleep_spec, sleep_report):
    paths = music_paths(sleep_spec)
    spec = sleep_spec
    # stale settings are refused instead of silently reused
    changed = dataclasses.replace(
        spec, audio=dataclasses.replace(spec.audio, edge_fade=0.25)
    )
    with pytest.raises(MusicEpisodeError, match="different settings"):
        prepare_audio(changed)
    # an artifact without evidence is never overwritten
    stray = Path(paths["masters"][0])
    evidence = Path(str(stray) + ".json")
    saved = evidence.read_text(encoding="utf-8")
    evidence.unlink()
    with pytest.raises(MusicEpisodeError, match="without evidence"):
        prepare_audio(spec)
    with pytest.raises(FileExistsError):
        with open(evidence, "x", encoding="utf-8") as stream:
            stream.write(saved)
            raise FileExistsError  # evidence is opened with "x" in the code
    evidence.write_text(saved, encoding="utf-8")
    # a modified artifact fails the hash check
    cycle = Path(paths["cycle"])
    original = cycle.read_bytes()
    cycle.write_bytes(original + b"\0")
    try:
        with pytest.raises(MusicEpisodeError, match="recorded hash"):
            build_music_episode(spec, steps=["visual"])
    finally:
        cycle.write_bytes(original)
    # a missing earlier artifact is an error, not a silent rebuild elsewhere
    with pytest.raises(MusicEpisodeError, match="audio step first"):
        me.prepare_overlays(
            dataclasses.replace(spec, output_dir=str(spec.output_dir) + "x")
        )


def test_failed_step_leaves_no_partial(tmp_path):
    target = tmp_path / "a" / "x.bin"

    def boom(partial):
        partial.write_bytes(b"half")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        me._resumable(target, {"k": 1}, boom)
    assert list(target.parent.iterdir()) == []

    def good(partial):
        partial.write_bytes(b"ok")
        return {}

    facts = me._resumable(target, {"k": 1}, good)
    assert not facts["skipped"] and target.read_bytes() == b"ok"
    assert me._resumable(target, {"k": 1}, boom)["skipped"]


# ---- end to end: study -------------------------------------------------------- #


def test_study_build_outputs(study_spec, study_report):
    steps = study_report["steps"]
    paths = music_paths(study_spec)
    info = audio_info(paths["audio"])
    assert info["frames"] == 8 * 48000  # the schedule, not the chapter sum
    assert audio_info(paths["music"])["frames"] == 8 * 48000
    events = steps["audio"]["mix"]["events"]
    assert [e["seconds"] for e in events] == [2.0, 4.0, 6.0]
    assert all(e["gain"] <= 1 and e["difference_db"] <= -9.9 for e in events)
    assert steps["audio"]["mix"]["ducking"] is False
    video = Path(paths["video"])
    assert len(frames_of(video)) == 8 * FPS == steps["render"]["frames"]
    assert abs(duration_of(video) - 8.0) < 0.1
    text = ffmpeg_info(video)
    assert text.count("Chapter #0:") == 4
    assert steps["render"]["overlay"] is not None and steps["render"]["ass"]
    levels = np.load(paths["levels"])
    assert levels.shape == (8 * FPS, 16)
    ass = Path(paths["ass"]).read_text(encoding="utf-8")
    assert "[Events]" in ass and "PlayResX: 1280" in ass
    meta = Path(paths["chapters"]).read_text(encoding="utf-8")
    assert meta.startswith(";FFMETADATA1\ntitle=Tiny; study\n")
    assert meta.count("[CHAPTER]") == 4


def test_study_overlay_is_visible(study_spec, study_report):
    frames = frames_of(music_paths(study_spec)["video"])
    plain = frames[:, 56:84, 8:152].astype(int)
    assert plain.std(axis=0).mean() > 0.5  # bars move over the moving picture


def test_study_resume_and_chime_hash(study_spec, study_report):
    before = hashes(study_spec)
    again = build_music_episode(study_spec)["steps"]
    assert again["audio"]["mix"]["skipped"] and again["audio"]["chime"]["skipped"]
    assert (
        again["overlays"]["levels"]["skipped"] and again["overlays"]["ass"]["skipped"]
    )
    assert again["render"]["skipped"]
    assert hashes(study_spec) == before


def test_reused_cycle_path(study_spec, study_report, tmp_path):
    cycle = music_paths(study_spec)["cycle"]
    spec = dataclasses.replace(
        study_spec,
        output_dir=str(tmp_path / "other"),
        visual=dataclasses.replace(study_spec.visual, clips=(), cycle_path=cycle),
    )
    first = me.prepare_visual(spec)
    assert first["background"] == cycle and first["cycle"]["external"]
    assert not first["cycle"]["skipped"]
    assert me.prepare_visual(spec)["cycle"]["skipped"]
    assert not (tmp_path / "other" / "visual" / "loops").exists()


# ---- command line ------------------------------------------------------------- #


@pytest.mark.parametrize("mode", me.MODES)
def test_cli_init_and_validate(mode, tmp_path, capsys):
    target = tmp_path / "proj"
    assert me.main(["init", str(target), "--mode", mode]) == 0
    assert (target / "tracks").is_dir() and (target / "visual").is_dir()
    readme = (target / "README.txt").read_text(encoding="utf-8")
    for gate in ("full listening", "full visual watch", "rights", "private upload"):
        assert gate in readme
    assert readme.count("NOT_RUN") == 4
    assert (target / "study.json").exists() == (mode == "study_pomodoro")
    assert me.main(["init", str(target), "--mode", mode]) == 1  # not empty
    capsys.readouterr()
    assert me.main(["validate", str(target / "music.json")]) == 1  # placeholders
    err = capsys.readouterr().err
    assert "MISSING" in err and "PLACEHOLDER" in err
    assert me.main(["validate", str(target / "music.json"), "--allow-missing"]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_errors_and_build(sleep_spec, sleep_report, tmp_path, capsys):
    folder = Path(sleep_spec.output_dir).parent
    assert me.main(["validate", str(tmp_path / "none.json")]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert me.main(["validate", str(bad)]) == 1
    assert "ERROR" in capsys.readouterr().err
    assert me.main(["build", str(folder / "music.json"), "--steps", "bogus"]) == 1
    capsys.readouterr()
    code = me.main(
        [
            "build",
            str(folder / "music.json"),
            "--steps",
            "audio,visual",
            "--workers",
            "1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0 and json.loads(out)["steps"]["audio"]["music"]["skipped"]
    assert me.main(["validate", str(folder / "music.json")]) == 0


def test_python_dash_m_dispatch(tmp_path):
    folder = tmp_path / "p"
    run = subprocess.run(
        [sys.executable, "-m", "moviepy.ae.templates", "music", "init", str(folder),
         "--mode", "sleep_longform"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONPATH": str(ROOT)},
    )  # fmt: skip
    assert run.returncode == 0, run.stderr
    assert (folder / "music.json").exists()


# ---- ambience, cards, ring, timeline, thumbnail, qa ------------------------------- #

from moviepy.ae.templates.thumbnail import DEFAULT_FONTS  # noqa: E402


FONTS = all(Path(p).is_file() for p in DEFAULT_FONTS.values())
needs_fonts = pytest.mark.skipif(not FONTS, reason="Windows fonts not installed")
TITLES = {
    "A": {"zh": "第一首", "en": "First", "ja": "一曲目"},
    "B": {"zh": "第二首", "en": "Second", "ja": "二曲目"},
}


def rich_sections(**extra):
    sections = {
        "ambience": {
            "particles": {
                "kind": "fireflies",
                "period": 2,
                "count": 6,
                "seed": 3,
                "opacity": 0.9,
            },
            "light_arc": {
                "keyframes": [[0, {}], [-1, {"brightness": -0.2}]],
                "max_rate": 5,
            },
            "sleep_fade": {"start": -3, "floor": 0.3, "max_rate": 5},
        },
        "cards": {"enabled": True, "titles": TITLES},
        "thumbnail": {
            "enabled": True,
            "titles": {"zh": "測試", "en": "Test"},
            "time": 1,
            "require_contrast": False,
        },
        "qa": {"enabled": True, "flash_seconds": 5},
    }
    sections.update(extra)
    return sections


def seeded_copy(source_spec, target, **changes):
    """Spec with extra sections whose audio/visual artifacts are copied over."""
    import shutil

    for part in ("audio", "visual"):
        shutil.copytree(Path(source_spec.output_dir) / part, target / part)
    data = source_spec.to_dict()
    data.update(output_dir=str(target), **changes)
    return MusicEpisodeSpec.from_dict(data)


@pytest.fixture(scope="module")
def rich_sleep(sleep_spec, sleep_report, tmp_path_factory):
    folder = tmp_path_factory.mktemp("rich_sleep")
    spec = seeded_copy(sleep_spec, folder / "build", **rich_sections())
    path = folder / "music.json"
    spec.to_json(path)
    report = build_music_episode(spec, steps=["audio", "visual", "overlays", "render"])
    return spec, report, path


@pytest.fixture(scope="module")
def rich_study(study_spec, study_report, tmp_path_factory):
    folder = tmp_path_factory.mktemp("rich_study")
    spec = seeded_copy(
        study_spec,
        folder / "build",
        cards={"enabled": True, "titles": TITLES, "offset": 0.5},
        study_ring=True,
        study_timeline=True,
    )
    return spec, build_music_episode(spec)


BASE = {
    "mode": "sleep_longform",
    "tracks": [{"id": "A", "source": "a.wav", "chapter_seconds": 60}],
    "visual": {"clips": ["c.mp4"]},
}
FAST = {"keyframes": [[0, {}], [1, {"brightness": -0.5}]]}


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"ambience": {"x": 1}}, "ambience: unknown key"),
        ({"ambience": {"particles": {"kind": "lava"}}}, "kind must be one of"),
        (
            {"ambience": {"particles": {"period": 0.51}}},
            "ambience: particles: .*whole number",
        ),
        (
            {"ambience": {"particles": {"region": [0, 0, 5000, 10]}}},
            "ambience: particles: .*inside the layer",
        ),
        ({"ambience": {"particles": {"region": [0, 0]}}}, "region must be"),
        ({"ambience": {"particles": {"count": -1}}}, "count must be >= 0"),
        ({"ambience": {"particles": {"opacity": 2}}}, "opacity must be <= 1"),
        ({"ambience": {"particles": {"seed": 1.5}}}, "seed must be a whole number"),
        (
            {"ambience": {"light_arc": {"preset": "dusk", "keyframes": [[0, {}]]}}},
            "preset or keyframes, not both",
        ),
        ({"ambience": {"light_arc": {"preset": "noon"}}}, "preset must be one of"),
        (
            {"ambience": {"light_arc": {"keyframes": [[0, {"hue": 1}]]}}},
            r"keyframes\[0\]: unknown parameter",
        ),
        (
            {"ambience": {"light_arc": {"keyframes": [[0, {}], [-100, {}]]}}},
            r"keyframes\[1\] time = -100 s is outside the 60 s video",
        ),
        (
            {"ambience": {"light_arc": FAST}},
            "ambience: light_arc: brightness changes",
        ),
        (
            {"ambience": {"light_arc": {"keyframes": [[5, {}], [5, {}]]}}},
            "strictly increase",
        ),
        ({"ambience": {"sleep_fade": {}}}, "sleep_fade: missing required key"),
        ({"ambience": {"sleep_fade": {"start": -30, "floor": 1}}}, "floor must be < 1"),
        (
            {"ambience": {"sleep_fade": {"start": 30, "end": 100}}},
            "sleep_fade: end = 100 s is outside the 60 s video",
        ),
        (
            {"ambience": {"sleep_fade": {"start": 0, "end": 5}}},
            "ambience: sleep_fade: fade changes",
        ),
        (
            {"ambience": {"sleep_fade": {"start": 40, "end": 20}}},
            "ambience: sleep_fade: need 0 <= start < end",
        ),
        ({"cards": {"enabled": True}}, "cards: titles missing for track id"),
        ({"cards": {"titles": {"Z": {"en": "x"}}}}, "unknown track id"),
        ({"cards": {"titles": {"A": {"fr": "x"}}}}, "unknown language"),
        ({"cards": {"titles": {"A": {}}}}, "needs zh, en or ja"),
        ({"cards": {"position": "middle"}}, "position must be one of"),
        ({"cards": {"hold": -1}}, "hold must be >= 0"),
        ({"study_ring": True}, "need study_pomodoro mode"),
        ({"study_timeline": "yes"}, "study_timeline must be true or false"),
        ({"thumbnail": {"enabled": True}}, r"titles\['zh'\] is required"),
        ({"thumbnail": {"layout": "wide"}}, "layout must be one of"),
        ({"thumbnail": {"output": "a/b.jpg"}}, "output must be a simple"),
        ({"thumbnail": {"output": "t.gif"}}, "output must be a simple"),
        ({"thumbnail": {"duration_badge": -3}}, "duration_badge must be >= 0"),
        ({"thumbnail": {"titles": {"en": "x"}}}, "zh"),
        ({"qa": {"targets": {"loud": 1}}}, "targets: unknown key"),
        ({"qa": {"flash_seconds": 0}}, "flash_seconds must be positive"),
        ({"qa": {"enabled": "yes"}}, "qa: enabled must be true or false"),
    ],
)
def test_new_section_validation(patch, message):
    data = json.loads(json.dumps(BASE))
    data.update(patch)
    with pytest.raises(MusicEpisodeError, match=message):
        MusicEpisodeSpec.from_dict(data)


def test_absent_sections_keep_old_behaviour():
    spec = MusicEpisodeSpec.from_dict(BASE)
    assert spec.ambience.particles is None and not spec.ambience.light_arc.active
    assert spec.ambience.sleep_fade is None
    assert not (spec.cards.enabled or spec.thumbnail.enabled or spec.qa.enabled)
    paths = music_paths(spec)
    assert paths["particles"] is paths["ass"] is paths["thumbnail"] is None
    assert paths["qa"] is None
    assert MusicEpisodeSpec.from_dict(spec.to_dict()) == spec
    assert me.resolve_ambience(spec, 60)["video_filters"] == []


def test_shipped_configs_enrichment_defaults():
    sleep = MusicEpisodeSpec.from_json(CONFIGS / "music_sleep_longform.json")
    amb = sleep.ambience
    assert amb.particles.kind == "fireflies" and amb.light_arc.preset == "day_to_night"
    assert (amb.sleep_fade.start, amb.sleep_fade.floor) == (-1800, 0.15)
    assert sleep.cards.enabled and sleep.thumbnail.enabled and sleep.qa.enabled
    assert sleep.thumbnail.duration_badge == "auto"
    resolved = me.resolve_ambience(sleep, 19200)
    fade = resolved["sleep_fade"]
    assert (fade.start, fade.end) == (17400.0, 19200.0)
    assert len(resolved["audio_filters"]) == 1
    assert resolved["light_arc"].keyframes[-1][0] == 19200
    study = MusicEpisodeSpec.from_json(CONFIGS / "music_study_pomodoro.json")
    assert study.ambience.particles is None and study.ambience.sleep_fade is None
    assert not study.ambience.light_arc.active
    assert study.study_ring and study.study_timeline and study.cards.enabled
    assert study.thumbnail.enabled and study.qa.enabled
    assert set(study.cards.titles) == {t.id for t in study.tracks}
    for spec in (sleep, study):
        assert MusicEpisodeSpec.from_dict(spec.to_dict()) == spec


def test_negative_times_resolve_against_total():
    spec = MusicEpisodeSpec.from_dict(
        dict(
            BASE,
            ambience={
                "light_arc": {
                    "keyframes": [{"time": 0}, {"time": -10, "brightness": -0.1}]
                },
                "sleep_fade": {"start": -30, "floor": 0.2},
            },
        )
    )
    resolved = me.resolve_ambience(spec, 60)
    assert [t for t, _ in resolved["light_arc"].keyframes] == [0.0, 50.0]
    assert (resolved["sleep_fade"].start, resolved["sleep_fade"].end) == (30.0, 60.0)
    with pytest.raises(MusicEpisodeError, match="outside the 20 s video"):
        me.resolve_ambience(spec, 20)  # start -30 lies before the beginning
    assert resolved["description"]["sleep_fade"]["start"] == 30.0


def test_inline_study_total_is_checked_in_the_spec():
    data = dict(
        BASE,
        mode="study_pomodoro",
        study=pomodoro_schedule("S", focus=60, rest=30, rounds=2, closing=30),
        ambience={"sleep_fade": {"start": 400, "end": 500, "max_rate": 1}},
    )
    with pytest.raises(MusicEpisodeError, match="outside the 180 s video"):
        MusicEpisodeSpec.from_dict(data)


def test_rich_sleep_outputs(rich_sleep):
    spec, report, _ = rich_sleep
    paths = music_paths(spec)
    steps = report["steps"]
    assert set(steps) == {"audio", "visual", "overlays", "render"}
    assert Path(paths["particles"]).is_file()
    particles = steps["visual"]["particles"]
    assert particles["particles"]["options"]["seed"] == 3 and not particles["skipped"]
    assert particles["human_visual"] == "NOT_RUN"
    assert not list(Path(paths["particles"]).parent.glob("*.partial*"))
    ass = Path(paths["ass"]).read_text(encoding="utf-8")
    assert paths["ass"].endswith("cards.ass") and "CardEN" in ass
    render = steps["render"]
    assert render["overlay_loops"][0]["path"] == str(Path(paths["particles"]).resolve())
    assert len(render["video_filters"]) == 1 and len(render["final_filters"]) == 1
    assert len(render["audio_filters"]) == 1 and render["config_sha256"]
    assert render["ambience"]["sleep_fade"]["start"] == 4.0
    frames = frames_of(paths["video"])
    assert len(frames) == 7 * FPS
    # the sleep fade (floor 0.3 plus the arc) darkens the picture at the end
    assert frames[-1].mean() < 0.6 * frames[2].mean()
    assert frames[5 * FPS].mean() < frames[2].mean()


def test_rich_sleep_audio_fades(rich_sleep):
    spec, _, _ = rich_sleep
    run = subprocess.run(
        [FFMPEG_BINARY, "-v", "error", "-i", music_paths(spec)["video"],
         "-f", "s16le", "-ac", "1", "-ar", "48000", "-"],
        capture_output=True,
        check=True,
    )  # fmt: skip
    pcm = np.abs(np.frombuffer(run.stdout, np.int16).astype(float))
    early = pcm[48000 : int(1.5 * 48000)].mean()
    late = pcm[int(6.0 * 48000) : int(6.4 * 48000)].mean()
    assert late < 0.7 * early


def test_rich_sleep_resume_and_stale_detection(rich_sleep):
    spec, _, _ = rich_sleep
    again = build_music_episode(spec, steps="audio,visual,overlays,render")
    assert again["steps"]["visual"]["particles"]["skipped"]
    assert again["steps"]["overlays"]["ass"]["skipped"]
    assert again["steps"]["render"]["skipped"]
    seed = dataclasses.replace(
        spec.ambience.particles, seed=spec.ambience.particles.seed + 1
    )
    changed = dataclasses.replace(
        spec, ambience=dataclasses.replace(spec.ambience, particles=seed)
    )
    with pytest.raises(MusicEpisodeError, match="different settings"):
        me.prepare_visual(changed)
    fade = dataclasses.replace(spec.ambience.sleep_fade, floor=0.5)
    dimmer = dataclasses.replace(
        spec, ambience=dataclasses.replace(spec.ambience, sleep_fade=fade)
    )
    with pytest.raises(MusicEpisodeError, match="different settings"):
        me.render(dimmer)
    titles = {"A": {"en": "Changed"}, "B": TITLES["B"]}
    retitled = dataclasses.replace(
        spec, cards=dataclasses.replace(spec.cards, titles=titles)
    )
    with pytest.raises(MusicEpisodeError, match="different settings"):
        me.prepare_overlays(retitled)


def test_rich_sleep_preview_window_uses_global_time(rich_sleep):
    spec, _, _ = rich_sleep
    me.render(spec, preview=(3, 4))
    window = frames_of(Path(spec.output_dir) / "render" / "tiny-preview-3-4.mp4")
    full = frames_of(music_paths(spec)["video"])
    assert len(window) == 4 * FPS
    # the preview's last frame is the same instant of the fade as the full one
    assert abs(float(window[-1].mean()) - float(full[7 * FPS - 1].mean())) < 6


def test_disabled_steps_are_skipped_not_run(sleep_spec, sleep_report):
    report = build_music_episode(sleep_spec, steps="thumbnail,qa")
    assert report["steps"] == {
        "thumbnail": {"enabled": False, "skipped": True},
        "qa": {"enabled": False, "skipped": True},
    }
    with pytest.raises(MusicEpisodeError, match="thumbnail.enabled is false"):
        me.make_thumbnail(sleep_spec)
    with pytest.raises(MusicEpisodeError, match="qa.enabled is false"):
        me.run_qa(sleep_spec)


def test_qa_step_on_real_render(rich_sleep):
    spec, _, _ = rich_sleep
    first = build_music_episode(spec, steps=["qa"])["steps"]["qa"]
    assert not first["skipped"] and first["overall"] in ("PASS", "REVIEW", "FAIL")
    path = Path(music_paths(spec)["qa"])
    assert path.name == "tiny.qa.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["render_sha256"] == me._sha256(music_paths(spec)["video"])
    assert saved["human_listening"] == saved["human_visual"] == "NOT_RUN"
    assert saved["flash"]["verdict"] in ("PASS", "REVIEW", "FAIL")
    before = path.read_bytes()
    second = build_music_episode(spec, steps=["qa"])["steps"]["qa"]
    assert second["skipped"] and path.read_bytes() == before
    other = dataclasses.replace(spec, qa=dataclasses.replace(spec.qa, flash_seconds=3))
    with pytest.raises(MusicEpisodeError, match="different render or QA settings"):
        me.run_qa(other)


def test_qa_needs_a_render(rich_sleep, tmp_path):
    spec, _, _ = rich_sleep
    empty = dataclasses.replace(spec, output_dir=str(tmp_path / "none"))
    with pytest.raises(MusicEpisodeError, match="qa needs the rendered video"):
        me.run_qa(empty)


@needs_fonts
def test_thumbnail_step_and_resume(rich_sleep):
    spec, _, _ = rich_sleep
    first = build_music_episode(spec, steps=["thumbnail"])["steps"]["thumbnail"]
    path = Path(music_paths(spec)["thumbnail"])
    assert path.parts[-2:] == ("publish", "thumbnail.jpg") and path.is_file()
    assert not first["skipped"] and first["size"] == [1280, 720]
    assert first["badge"]["text"] == "0:07"  # auto badge = the formatted duration
    assert first["human_visual"] == "NOT_RUN" and first["sha256"] == me._sha256(path)
    assert not list(path.parent.glob("*.partial*"))
    again = build_music_episode(spec, steps=["thumbnail"])["steps"]["thumbnail"]
    assert again["skipped"]
    th = dataclasses.replace(spec.thumbnail, time=2)
    with pytest.raises(MusicEpisodeError, match="different settings"):
        me.make_thumbnail(dataclasses.replace(spec, thumbnail=th))


def test_rich_study_outputs(study_spec, study_report, rich_study):
    spec, report = rich_study
    paths = music_paths(spec)
    plain = music_paths(study_spec)
    text = Path(paths["ass"]).read_text(encoding="utf-8")
    base = Path(plain["ass"]).read_text(encoding="utf-8")
    assert text.count("Dialogue") > base.count("Dialogue") + 8
    for line in base.splitlines():
        if line.startswith("Dialogue"):
            assert line in text  # every original S14 event is unchanged
    assert "CardEN" in text
    assert report["steps"]["render"]["frames"] == len(frames_of(paths["video"]))
    assert set(report["steps"]) == {"audio", "visual", "overlays", "render"}


def test_rich_study_ring_is_visible(study_spec, study_report, rich_study):
    spec, _ = rich_study
    rich = frames_of(music_paths(spec)["video"])
    plain = frames_of(music_paths(study_spec)["video"])
    assert rich.shape == plain.shape
    # ring centre (1186, 234) on 1280x720 -> about (148, 29) at 160x90
    diff = np.abs(rich[12, 22:37, 138:159].astype(int) - plain[12, 22:37, 138:159])
    assert diff.max() > 8


def test_rich_study_resume(rich_study):
    spec, _ = rich_study
    again = build_music_episode(spec)
    assert again["steps"]["overlays"]["ass"]["skipped"]
    assert again["steps"]["render"]["skipped"]


def test_study_cards_failure_is_reported(rich_study):
    spec, _ = rich_study
    wide = {"A": {"en": "x" * 400}, "B": TITLES["B"]}
    bad = dataclasses.replace(spec, cards=dataclasses.replace(spec.cards, titles=wide))
    with pytest.raises(MusicEpisodeError, match="cards/ring/timeline"):
        me.prepare_overlays(bad)


def test_cli_steps_accept_new_names(rich_sleep, capsys):
    _, _, path = rich_sleep
    assert me.main(["build", str(path), "--steps", "qa", "--workers", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert list(out["steps"]) == ["qa"]
    assert me.main(["build", str(path), "--steps", "render,qa", "--workers", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert list(out["steps"]) == ["render", "qa"] and out["steps"]["qa"]["skipped"]
    assert me.main(["validate", str(path)]) == 0
    assert me.STEPS[-2:] == ("thumbnail", "qa")


# ---- soundx loudness backend ---------------------------------------------------- #


def test_loudness_backend_field(sleep_spec):
    assert sleep_spec.audio.loudness_backend == "soundx"
    data = sleep_spec.to_dict()
    assert data["audio"]["loudness_backend"] == "soundx"
    assert MusicEpisodeSpec.from_dict(data) == sleep_spec
    data["audio"]["loudness_backend"] = "ffmpeg"
    assert MusicEpisodeSpec.from_dict(data).audio.loudness_backend == "ffmpeg"
    data["audio"]["loudness_backend"] = "sox"
    with pytest.raises(MusicEpisodeError, match="loudness_backend"):
        MusicEpisodeSpec.from_dict(data)


@pytest.mark.parametrize("name", ["music_sleep_longform", "music_study_pomodoro"])
def test_shipped_configs_select_soundx(name):
    config = json.loads((CONFIGS / f"{name}.json").read_text(encoding="utf-8"))
    assert config["audio"]["loudness_backend"] == "soundx"


def test_validate_reports_soundx_status(sleep_spec, monkeypatch, tmp_path):
    result = validate_music_episode(sleep_spec)
    assert result["ok"] and result["soundx"]["ok"]
    assert result["soundx"]["version"] == "soundx 0.2.0"
    assert result["soundx"]["has_loudness"] is False
    monkeypatch.delenv("MOVIEPY_SOUNDX")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setitem(
        sx.SOUNDX_REQUIREMENTS, "windows_default", str(tmp_path / "nowhere.exe")
    )
    result = validate_music_episode(sleep_spec)
    assert not result["ok"] and not result["soundx"]["ok"]
    assert any(
        "MOVIEPY_SOUNDX" in p and p.startswith("soundx:") for p in result["problems"]
    )
    ffmpeg_only = dataclasses.replace(
        sleep_spec,
        audio=dataclasses.replace(sleep_spec.audio, loudness_backend="ffmpeg"),
    )
    result = validate_music_episode(ffmpeg_only)
    assert result["ok"] and result["soundx"] is None


def test_backend_is_part_of_the_settings_hash(sleep_spec, sleep_report, tmp_path):
    """Changing the backend invalidates the finished masters (hash mismatch)."""
    other = dataclasses.replace(
        sleep_spec,
        audio=dataclasses.replace(sleep_spec.audio, loudness_backend="ffmpeg"),
    )
    with pytest.raises(MusicEpisodeError, match="different settings"):
        prepare_audio(other)


def test_audio_step_records_soundx_evidence(sleep_spec, sleep_report):
    master = sleep_report["steps"]["audio"]["masters"][0]
    assert master["backend"] == "soundx"
    assert master["lufs_backend"].startswith("ffmpeg-loudnorm")
    assert master["soundx"]["version"] == "soundx 0.2.0"
    assert master["soundx"]["stage"]["frames"] == master["output"]["frames"]
    evidence = json.loads(Path(master["path"] + ".json").read_text(encoding="utf-8"))
    assert evidence["soundx"]["backend_chain"]
