"""Tests for the Source Han (思源) default fonts of the channel presets."""

from pathlib import Path

import pytest

from moviepy.ae.templates import presets
from moviepy.ae.templates.presets import PRESETS, source_han_font


def test_presets_default_to_source_han():
    for preset in PRESETS.values():
        names = {Path(path).name for path in preset.fonts.values()}
        assert all(name.startswith("SourceHan") for name in names)
        assert Path(preset.fonts["title"]).name.startswith("SourceHanSerif")
        assert Path(preset.fonts["body"]).name.startswith("SourceHanSans")


def test_source_han_font_searches_user_folder_first_match(tmp_path, monkeypatch):
    system, user = tmp_path / "system", tmp_path / "user"
    system.mkdir()
    user.mkdir()
    (user / "SourceHanSerifTW-Bold.otf").write_bytes(b"")
    monkeypatch.setattr(presets, "_font_dirs", lambda: [system, user])
    assert source_han_font("serif") == str(user / "SourceHanSerifTW-Bold.otf")
    # The full TC set wins over the TW subset, in any folder.
    (user / "SourceHanSerifTC-Bold.otf").write_bytes(b"")
    assert source_han_font("serif") == str(user / "SourceHanSerifTC-Bold.otf")
    (system / "SourceHanSerifTC-Bold.otf").write_bytes(b"")
    assert source_han_font("serif") == str(system / "SourceHanSerifTC-Bold.otf")


def test_missing_source_han_reports_expected_path(tmp_path, monkeypatch):
    system, user = tmp_path / "system", tmp_path / "user"
    monkeypatch.setattr(presets, "_font_dirs", lambda: [system, user])
    path = source_han_font("sans", "Medium")
    assert path == str(user / "SourceHanSansTC-Medium.otf")
    preset = PRESETS["nightlamp_story"].with_overrides(fonts={"body": path})
    with pytest.raises(FileNotFoundError, match="SourceHanSansTC-Medium.otf"):
        preset.font("body")
    with pytest.raises(ValueError):
        source_han_font("mono")


def test_source_han_font_accepts_duplicate_download_names(tmp_path, monkeypatch):
    user = tmp_path / "user"
    user.mkdir()
    (user / "SourceHanSerifTW-Bold (1).otf").write_bytes(b"")
    (user / "SourceHanSerifTW-Bold copy.otf").write_bytes(b"")
    monkeypatch.setattr(presets, "_font_dirs", lambda: [tmp_path / "none", user])
    assert source_han_font("serif") == str(user / "SourceHanSerifTW-Bold (1).otf")
    # The plain name wins over a numbered copy.
    (user / "SourceHanSerifTW-Bold.otf").write_bytes(b"")
    assert source_han_font("serif") == str(user / "SourceHanSerifTW-Bold.otf")
