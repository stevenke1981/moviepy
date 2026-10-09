"""Tests for the soundx wrapper and the soundx loudness backend of music_audio.

CI has no soundx, so a fake one (``_fake_soundx``) emulates the 0.2.0 and
0.3.0 command lines. Tests against the real binary skip when it is absent.
"""

import json
import math

import numpy as np

import pytest

from moviepy.ae.templates import soundx as sx
from moviepy.ae.templates._audio_io import AudioWriter, audio_info, read_audio
from moviepy.ae.templates.music_audio import (
    LUFS_FALLBACK,
    measure_loudness,
    music_audio_report,
    normalize_loudness,
)
from tests.ae._fake_soundx import install_fake_soundx


RATE = 48000


@pytest.fixture
def fake_020(tmp_path, monkeypatch):
    launcher = install_fake_soundx(tmp_path / "fake020", "0.2.0")
    monkeypatch.setenv("MOVIEPY_SOUNDX", str(launcher))
    monkeypatch.setenv("FAKE_SOUNDX_LOG", str(tmp_path / "calls020.log"))
    return launcher


@pytest.fixture
def fake_030(tmp_path, monkeypatch):
    launcher = install_fake_soundx(tmp_path / "fake030", "0.3.0")
    monkeypatch.setenv("MOVIEPY_SOUNDX", str(launcher))
    monkeypatch.setenv("FAKE_SOUNDX_LOG", str(tmp_path / "calls030.log"))
    return launcher


def calls(tmp_path, name):
    path = tmp_path / name
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def noise(frames, seed=3, amp=0.1, channels=2):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((frames, channels)) * amp).astype(np.float32)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "in.wav"
    with AudioWriter(path, rate=44100, subtype="float") as writer:
        writer.write(noise(44100))
    return path


def peak_db(path):
    return 20 * math.log10(float(np.abs(read_audio(path)).max()))


# ---- discovery and checks ------------------------------------------------------ #


def test_requirements_table():
    req = sx.SOUNDX_REQUIREMENTS
    assert req["min_version"] == (0, 2, 0)
    assert req["loudness_min_version"] == (0, 3, 0)
    assert req["executable_env"] == "MOVIEPY_SOUNDX"
    assert req["path_names"] == ("soundx", "soundx.exe")
    assert req["source_url"] == "https://github.com/urtiger101-tw/sox-rs"
    assert req["windows_default"] == r"C:\Program Files\soundx\soundx.exe"


def test_parse_version_and_option():
    assert sx.parse_version("soundx 0.3.1") == (0, 3, 1)
    assert sx.parse_version("0.2") == (0, 2, 0)
    assert sx.parse_version("nothing") is None
    assert sx.option("normalize-db", -4.0) == "--normalize-db=-4.0"


@pytest.fixture
def no_soundx(tmp_path, monkeypatch):
    monkeypatch.delenv("MOVIEPY_SOUNDX", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setitem(
        sx.SOUNDX_REQUIREMENTS, "windows_default", str(tmp_path / "nowhere.exe")
    )


def test_missing_soundx_names_env_and_repo(no_soundx):
    with pytest.raises(sx.SoundxNotFoundError) as caught:
        sx.find_soundx()
    message = str(caught.value)
    assert "MOVIEPY_SOUNDX" in message
    assert "https://github.com/urtiger101-tw/sox-rs" in message
    check = sx.check_soundx()
    assert not check.ok and check.executable is None and not check.has_loudness
    assert "MOVIEPY_SOUNDX" in check.problems[0]
    with pytest.raises(sx.SoundxError, match="MOVIEPY_SOUNDX"):
        sx.require_soundx()


def test_env_pointing_nowhere_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("MOVIEPY_SOUNDX", str(tmp_path / "gone.exe"))
    with pytest.raises(sx.SoundxNotFoundError, match="missing file"):
        sx.find_soundx()
    assert not sx.check_soundx().ok


def test_explicit_executable_wins_over_env(fake_020, tmp_path):
    other = install_fake_soundx(tmp_path / "other", "0.3.0")
    assert sx.find_soundx(other) == other
    assert sx.check_soundx(other).has_loudness
    assert not sx.check_soundx(fake_020).has_loudness


def test_check_detects_capabilities(fake_020, fake_030):
    old = sx.check_soundx(fake_020)
    assert old.ok and old.version == "soundx 0.2.0"
    assert old.version_tuple == (0, 2, 0) and not old.has_loudness
    new = sx.check_soundx(fake_030)
    assert new.ok and new.version_tuple == (0, 3, 0) and new.has_loudness
    assert new.problems == []


def test_too_old_version_is_a_problem(tmp_path):
    ancient = install_fake_soundx(tmp_path / "old", "0.1.5")
    check = sx.check_soundx(ancient)
    assert not check.ok and "0.2.0" in check.problems[0]


def test_cli_json_and_text(fake_030, capsys):
    assert sx.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["check"]["ok"] and payload["check"]["has_loudness"]
    assert payload["requirements"]["executable_env"] == "MOVIEPY_SOUNDX"
    assert sx.main([]) == 0
    assert "has_loudness: True" in capsys.readouterr().out


def test_cli_reports_missing(no_soundx, capsys):
    assert sx.main([]) == 1
    assert "FAILED" in capsys.readouterr().out


# ---- wrappers ------------------------------------------------------------------ #


def test_info_wrapper(fake_020, source):
    info = sx.soundx_info(source)
    assert info["frames"] == 44100 and info["spec"]["sample_rate"] == 44100


def test_stage_recipe_and_negative_argument_format(fake_020, source, tmp_path):
    target = tmp_path / "stage.wav"
    stat = sx.soundx_stage(source, target)
    argv = calls(tmp_path, "calls020.log")[-1]
    assert argv[0] == "convert"
    assert "--normalize-db=-4.0" in argv and "-4.0" not in argv
    for flag in ("--bits=24", "--rate=48000", "--channels=2"):
        assert flag in argv
    assert "--normalize" in argv and "--stat-json" in argv
    assert stat["frames"] == round(44100 * RATE / 44100) == audio_info(target)["frames"]
    assert stat["spec"] == {"sample_rate": RATE, "channels": 2}
    assert peak_db(target) == pytest.approx(-4.0, abs=0.05)


def test_stage_is_exclusive_and_cleans_up(fake_020, source, tmp_path):
    target = tmp_path / "stage.wav"
    sx.soundx_stage(source, target)
    with pytest.raises(FileExistsError):
        sx.soundx_stage(source, target)
    sx.soundx_stage(source, target, overwrite=True, normalize_db=-6.0)
    assert peak_db(target) == pytest.approx(-6.0, abs=0.05)
    with pytest.raises(ValueError, match=r"\.wav"):
        sx.soundx_stage(source, tmp_path / "stage.flac")
    broken = tmp_path / "broken.wav"
    with pytest.raises(sx.SoundxError, match="soundx convert failed"):
        sx.soundx_stage(tmp_path / "missing.wav", broken)
    assert not broken.exists()


def test_failure_carries_stderr(fake_020, tmp_path):
    with pytest.raises(sx.SoundxError) as caught:
        sx.soundx_info(tmp_path / "missing.wav")
    assert "exit code" in str(caught.value) and "FileNotFoundError" in str(caught.value)
    with pytest.raises(sx.SoundxError) as caught:
        sx.soundx_loudness(tmp_path / "x.wav")
    assert "no `loudness` command" in str(caught.value)
    assert "soundx>=0.3.0" in str(caught.value)


def test_loudness_and_normalize_need_030(fake_020, source, tmp_path):
    with pytest.raises(sx.SoundxError, match=r"soundx>=0\.3\.0"):
        sx.soundx_normalize(source, tmp_path / "n.wav")


def test_loudness_and_normalize_wrappers(fake_030, source, tmp_path):
    staged = tmp_path / "stage.wav"
    sx.soundx_stage(source, staged)
    measured = sx.soundx_loudness(staged)
    assert measured["integrated_lufs"] is not None
    target = tmp_path / "level.wav"
    stat = sx.soundx_normalize(staged, target, target_lufs=-18.0, true_peak=-1.8)
    argv = calls(tmp_path, "calls030.log")[-1]
    assert argv[0] == "convert"
    assert "--loudness-target=-18.0" in argv and "--true-peak=-1.8" in argv
    assert "--normalize" not in argv
    assert stat["loudness"]["output"]["integrated_lufs"] == pytest.approx(
        -18.0, abs=0.05
    )
    with pytest.raises(FileExistsError):
        sx.soundx_normalize(staged, target)
    streamed = tmp_path / "streamed.wav"
    sx.soundx_normalize(staged, streamed, stream=True, target_lufs=-20.0)
    argv = calls(tmp_path, "calls030.log")[-1]
    assert argv[0] == "stream" and "--loudness-target=-20.0" in argv
    assert audio_info(streamed)["frames"] == audio_info(staged)["frames"]


# ---- music_audio: soundx backend ------------------------------------------------ #


def test_normalize_with_020_uses_ffmpeg_for_lufs(fake_020, source, tmp_path):
    target = tmp_path / "master.flac"
    facts = normalize_loudness(source, target)
    expected = round(44100 * RATE / 44100)
    assert facts["backend"] == "soundx"
    assert facts["lufs_backend"] == LUFS_FALLBACK
    assert (
        facts["lufs_backend"]
        == "ffmpeg-loudnorm (soundx lacks loudness; upgrade to soundx>=0.3.0)"
    )
    evidence = facts["soundx"]
    assert evidence["version"] == "soundx 0.2.0" and not evidence["has_loudness"]
    assert evidence["executable"] == str(fake_020)
    assert evidence["stage"]["frames"] == expected
    assert evidence["normalize"] is None
    assert evidence["backend_chain"][0].startswith("soundx convert --normalize")
    assert any("ffmpeg loudnorm" in step for step in evidence["backend_chain"])
    assert facts["output"]["frames"] == expected == audio_info(target)["frames"]
    assert facts["output"]["loudness"]["integrated_lufs"] == pytest.approx(
        -18.0, abs=1.0
    )
    assert facts["measured"]["lufs_backend"] == LUFS_FALLBACK
    assert facts["human_listening"] == "NOT_RUN"
    assert (
        sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".soundx")) == []
    )
    json.dumps(facts)


def test_normalize_with_030_uses_soundx_lufs(fake_030, source, tmp_path):
    target = tmp_path / "master.flac"
    facts = normalize_loudness(source, target, frames=50000)
    assert facts["lufs_backend"] == "soundx"
    evidence = facts["soundx"]
    assert evidence["has_loudness"] and evidence["normalize"]["loudness"]
    assert evidence["backend_chain"][1].startswith("soundx convert --loudness-target")
    assert facts["output"]["frames"] == 50000 == audio_info(target)["frames"]
    assert facts["measured"]["lufs_backend"] == "soundx"
    assert facts["output"]["loudness"]["backend"] == "soundx"
    assert facts["output"]["loudness"]["integrated_lufs"] == pytest.approx(
        -18.0, abs=0.3
    )
    assert not any("ffmpeg loudnorm" in step for step in evidence["backend_chain"])
    names = [argv[0] for argv in calls(tmp_path, "calls030.log")]
    assert "convert" in names and "loudness" in names


def test_normalize_exclusive_and_wav_target(fake_030, source, tmp_path):
    target = tmp_path / "master.wav"
    facts = normalize_loudness(source, target)
    assert facts["output"]["loudness"]["measured_on"] == "final file"
    with pytest.raises(FileExistsError):
        normalize_loudness(source, target)
    with pytest.raises(ValueError):
        normalize_loudness(source, tmp_path / "bad.mp3")
    with pytest.raises(ValueError, match="backend"):
        normalize_loudness(source, tmp_path / "x.flac", backend="sox")


def test_normalize_silent_source_is_refused(fake_030, tmp_path):
    silent = tmp_path / "silent.wav"
    with AudioWriter(silent, rate=RATE) as writer:
        writer.write(np.zeros((RATE, 2), np.float32))
    target = tmp_path / "out.flac"
    with pytest.raises(ValueError, match="silent"):
        normalize_loudness(silent, target)
    assert not target.exists()


def test_missing_soundx_has_no_silent_fallback(no_soundx, source, tmp_path):
    target = tmp_path / "master.flac"
    with pytest.raises(sx.SoundxError) as caught:
        normalize_loudness(source, target)
    assert "MOVIEPY_SOUNDX" in str(caught.value)
    assert "github.com/urtiger101-tw/sox-rs" in str(caught.value)
    assert not target.exists()
    with pytest.raises(sx.SoundxError, match="MOVIEPY_SOUNDX"):
        measure_loudness(source)
    with pytest.raises(sx.SoundxError, match="MOVIEPY_SOUNDX"):
        music_audio_report(source)


def test_ffmpeg_backend_needs_no_soundx(no_soundx, source, tmp_path):
    target = tmp_path / "master.flac"
    facts = normalize_loudness(source, target, backend="ffmpeg")
    assert facts["backend"] == "ffmpeg" and "soundx" not in facts
    assert facts["lufs_backend"] == "ffmpeg-loudnorm"
    assert audio_info(target)["frames"] == facts["output"]["frames"]
    assert facts["output"]["loudness"]["integrated_lufs"] == pytest.approx(
        -18.0, abs=1.0
    )
    measured = measure_loudness(source, backend="ffmpeg")
    assert measured["backend"] == "ffmpeg" and "input_i" in measured
    assert music_audio_report(source, backend="ffmpeg")["loudness"]["backend"] == (
        "ffmpeg"
    )


def test_measure_backends(fake_020, fake_030, source):
    # fake_030 was installed last, so MOVIEPY_SOUNDX points at it
    new = measure_loudness(source)
    assert new["backend"] == "soundx" and new["lufs_backend"] == "soundx"
    assert new["soundx"]["standard"] == "FAKE"
    old = measure_loudness(source, soundx_executable=fake_020)
    assert old["backend"] == "soundx" and old["lufs_backend"] == LUFS_FALLBACK
    assert "input_i" in old


# ---- the real soundx (skipped when it is not installed) -------------------------- #


@pytest.fixture
def real():
    check = sx.check_soundx()
    if not check.ok:
        pytest.skip("real soundx not installed")
    return check


def test_real_soundx_stage_peak_and_frames(real, tmp_path):
    source = tmp_path / "tone.wav"
    time = np.arange(44100) / 44100
    wave = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    with AudioWriter(source, rate=44100, channels=1, subtype="float") as writer:
        writer.write(wave[:, None])
    target = tmp_path / "staged.wav"
    stat = sx.soundx_stage(source, target)
    info = audio_info(target)
    assert info["rate"] == RATE and info["channels"] == 2
    assert info["frames"] == stat["frames"] == round(44100 * RATE / 44100)
    assert peak_db(target) == pytest.approx(-4.0, abs=0.05)
    assert sx.soundx_info(target)["frames"] == info["frames"]
    with pytest.raises(FileExistsError):
        sx.soundx_stage(source, target)


def test_real_soundx_normalize_loudness(real, tmp_path):
    source = tmp_path / "noise.wav"
    with AudioWriter(source, rate=RATE, subtype="float") as writer:
        writer.write(noise(RATE * 4, amp=0.05))
    target = tmp_path / "norm.flac"
    facts = normalize_loudness(source, target)
    assert audio_info(target)["frames"] == RATE * 4
    assert facts["soundx"]["version"].startswith("soundx ")
    assert facts["output"]["loudness"]["integrated_lufs"] == pytest.approx(
        -18.0, abs=1.0
    )
    if real.has_loudness:
        assert facts["lufs_backend"] == "soundx"
    else:
        assert facts["lufs_backend"] == LUFS_FALLBACK
