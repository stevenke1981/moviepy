"""Checks that ``docs/ae/episode_workflow.rst`` matches the episode code.

The guide is the user's manual for ``python -m moviepy.ae.templates``. These
tests keep it honest: code blocks must parse, CLI subcommands and flags must
exist in the real argparse parser, every ``episode.json`` field in the tables
must exist in its dataclass (and every dataclass field must be documented),
and the guide must be reachable from ``templates.rst``.
"""

import ast
import inspect
import json
import re
import shlex
import textwrap
from dataclasses import fields
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "ae" / "episode_workflow.rst"
TEMPLATES_DOC = REPO / "docs" / "ae" / "templates.rst"
CONFIGS = REPO / "moviepy" / "ae" / "templates" / "configs"
PREFIX = "python -m moviepy.ae.templates"

# Dataclass behind each heading key in the field tables. ``shots`` and friends
# drop the ``[]`` suffix used in prose; ``episode.json`` is the top level.
SECTION_CLASSES = {
    "episode.json": "EpisodeSpec",
    "background": "BackgroundSpec",
    "intro": "BookendSpec",
    "shots": "ShotSpec",
    "chapters": "ChapterSpec",
    "quotes": "QuoteSpec",
    "subtitles": "SubtitleSpec",
    "audio": "AudioSpec",
}
# JSON key -> dataclass field for the renamed keys (``in``/``out`` on shots).
JSON_RENAMES = {"in": "clip_in", "out": "clip_out"}


@pytest.fixture(scope="module")
def doc_text():
    return DOC.read_text(encoding="utf-8")


def _code_blocks(text, language):
    """Return the body of every ``.. code-block:: <language>`` directive."""
    lines = text.splitlines()
    bodies = []
    marker = f".. code-block:: {language}"
    for index, line in enumerate(lines):
        if line.strip() != marker:
            continue
        indent = len(line) - len(line.lstrip())
        body = []
        for follow in lines[index + 1 :]:
            if follow.strip() and len(follow) - len(follow.lstrip()) <= indent:
                break
            body.append(follow)
        bodies.append(textwrap.dedent("\n".join(body)).strip("\n"))
    return bodies


def _cli_help(capsys):
    """Return ``{"": top_help, "init": ..., "validate": ..., "render": ...}``."""
    from moviepy.ae.templates.episode import main

    helps = {}
    for sub in ("", "init", "validate", "render"):
        argv = ["--help"] if not sub else [sub, "--help"]
        capsys.readouterr()
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 0
        helps[sub] = capsys.readouterr().out
    return helps


def _field_tables(text):
    """Yield ``(heading, [first-column names])`` for each ``list-table``.

    A heading is a text line directly followed by a ``-``, ``~`` or ``=``
    underline. Only the first column of each row is read.
    """
    lines = text.splitlines()
    heading = ""
    tables = []
    for index, line in enumerate(lines):
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        if line.strip() and re.fullmatch(r"[-~=]{3,}", nxt.strip() or "x"):
            heading = line.strip()
            continue
        if line.strip() != ".. list-table::":
            continue
        names = []
        for follow in lines[index + 1 :]:
            if follow.strip() and not follow.startswith(" "):
                break
            match = re.match(r"^\s*\* - (.*)$", follow)
            if match:
                names.append(re.findall(r"``([^`]+)``", match.group(1)))
        tables.append((heading, names))
    return tables


# --------------------------------------------------------------------------- #
# structure and links
# --------------------------------------------------------------------------- #


def test_doc_is_orphan_and_has_title(doc_text):
    assert doc_text.splitlines()[0] == ":orphan:"
    assert "新集影片製作流程" in doc_text.splitlines()[2]


def test_templates_rst_links_to_guide():
    text = TEMPLATES_DOC.read_text(encoding="utf-8")
    head = text.splitlines()[:30]
    assert "新影片製作流程" in head
    assert ":doc:`episode_workflow`" in text
    assert (REPO / "docs" / "ae" / "templates.rst").is_file()


def test_guide_links_back_to_templates(doc_text):
    assert ":doc:`templates`" in doc_text


def test_section_headings_are_chinese(doc_text):
    lines = doc_text.splitlines()
    headings = [
        lines[i]
        for i in range(len(lines) - 1)
        if lines[i].strip() and re.fullmatch(r"[-~=]{3,}", lines[i + 1].strip() or "x")
    ]
    assert len(headings) >= 10
    for heading in headings:
        assert re.search(r"[一-鿿]", heading), heading


# --------------------------------------------------------------------------- #
# code blocks
# --------------------------------------------------------------------------- #


def test_python_blocks_parse(doc_text):
    blocks = _code_blocks(doc_text, "python")
    assert len(blocks) >= 4
    for body in blocks:
        ast.parse(body)


def test_python_call_keywords_exist_in_signatures(doc_text):
    from moviepy.ae.templates import subtitles
    from moviepy.ae.templates.episode import build_episode, render_episode
    from moviepy.ae.templates.ken_burns import check_motion

    known = {
        "render_episode": render_episode,
        "build_episode": build_episode,
        "check_motion": check_motion,
        "find_bad_breaks": subtitles.find_bad_breaks,
        "check_timing": subtitles.check_timing,
        "parse_srt": subtitles.parse_srt,
    }
    checked = 0
    for body in _code_blocks(doc_text, "python"):
        for node in ast.walk(ast.parse(body)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            func = known.get(node.func.id)
            if func is None:
                continue
            checked += 1
            params = inspect.signature(func).parameters
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                continue
            for keyword in node.keywords:
                assert keyword.arg in params, f"{node.func.id}({keyword.arg}=)"
    assert checked >= 4


# --------------------------------------------------------------------------- #
# CLI: subcommands and flags exist in the real parser
# --------------------------------------------------------------------------- #


def test_cli_subcommands_exist(doc_text, capsys):
    helps = _cli_help(capsys)
    choices = set(re.search(r"\{([a-z,]+)\}", helps[""]).group(1).split(","))
    assert choices == {"init", "validate", "render"}
    mentioned = set(
        re.findall(r"python -m moviepy\.ae\.templates\s+([a-z]+)", doc_text)
    )
    assert mentioned == choices


def test_cli_console_lines_use_known_flags(doc_text, capsys):
    helps = _cli_help(capsys)
    lines = []
    for body in _code_blocks(doc_text, "console"):
        lines += [ln.strip() for ln in body.splitlines() if PREFIX in ln]
    assert len(lines) >= 5
    for line in lines:
        tokens = shlex.split(line.split(PREFIX, 1)[1].replace("\\", "/"))
        sub = tokens[0]
        assert sub in ("init", "validate", "render"), line
        for token in tokens:
            if token.startswith("--"):
                assert token in helps[sub], f"{token} not accepted by {sub}: {line}"


def test_every_flag_in_doc_exists_in_some_subcommand(doc_text, capsys):
    helps = _cli_help(capsys)
    known = " ".join(helps.values())
    flags = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", doc_text))
    assert flags, "the guide should mention the CLI flags"
    for flag in flags:
        assert flag in known, flag


def test_init_channels_match_cli(doc_text):
    from moviepy.ae.templates.episode import CHANNELS

    assert set(CHANNELS) == {"story", "history"}
    assert "``history``" in doc_text and "``story``" in doc_text


# --------------------------------------------------------------------------- #
# episode.json field tables
# --------------------------------------------------------------------------- #


def _dataclass_json_names(cls):
    inverse = {v: k for k, v in JSON_RENAMES.items()}
    return {inverse.get(f.name, f.name) for f in fields(cls)}


def test_field_tables_match_dataclasses(doc_text):
    from moviepy.ae.templates import episode

    tables = _field_tables(doc_text)
    seen = set()
    for heading, rows in tables:
        keys = re.findall(r"``([^`]+)``", heading)
        if not keys:
            continue
        section = keys[0]
        if section not in SECTION_CLASSES:
            continue
        cls = getattr(episode, SECTION_CLASSES[section])
        names = [name for row in rows for name in row]
        assert len(names) == len(set(names)), f"duplicate rows under {heading}"
        expected = _dataclass_json_names(cls)
        unknown = set(names) - expected
        assert (
            not unknown
        ), f"{section}: not fields of {cls.__name__}: {sorted(unknown)}"
        missing = expected - set(names)
        assert not missing, f"{section}: undocumented fields: {sorted(missing)}"
        seen.add(section)
    assert seen == set(SECTION_CLASSES), set(SECTION_CLASSES) - seen


def test_preset_override_keys_exist_in_channel_preset(doc_text):
    from moviepy.ae.templates.presets import ChannelPreset, SubtitleStyle

    rows = None
    for heading, names in _field_tables(doc_text):
        if "preset_overrides" in heading:
            rows = names
    assert rows, "preset_overrides table missing"
    keys = {name for row in rows for name in row}
    preset_fields = {f.name for f in fields(ChannelPreset)}
    assert keys <= preset_fields, sorted(keys - preset_fields)
    assert {"size", "fps", "fade", "hold", "overlay_opacity"} <= keys
    style_fields = {f.name for f in fields(SubtitleStyle)}
    assert {"primary_size", "secondary_size"} <= style_fields
    assert "primary_size" in doc_text and "secondary_size" in doc_text


def test_font_roles_documented(doc_text):
    from moviepy.ae.templates.episode import _FONT_ROLES

    for role in _FONT_ROLES:
        assert re.search(rf"(?<![A-Za-z_]){role}(?![A-Za-z_])", doc_text), role


def test_shipped_configs_use_documented_keys():
    from moviepy.ae.templates.episode import EpisodeSpec, ShotSpec

    known = {f.name for f in fields(EpisodeSpec)}
    shot_keys = _dataclass_json_names(ShotSpec)
    for path in sorted(CONFIGS.glob("nightlamp_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) <= known, path.name
        for shot in data["shots"]:
            assert set(shot) <= shot_keys, (path.name, sorted(set(shot) - shot_keys))


def test_presets_named_in_guide_exist(doc_text):
    from moviepy.ae.templates.presets import PRESETS

    assert set(PRESETS) == {"nightlamp_story", "nightlamp_history"}
    for name in PRESETS:
        assert f"``{name}``" in doc_text or name in doc_text


# --------------------------------------------------------------------------- #
# API names the guide points at
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "module, name",
    [
        ("moviepy.ae.templates.episode", "EpisodeSpec"),
        ("moviepy.ae.templates.episode", "build_episode"),
        ("moviepy.ae.templates.episode", "episode_report"),
        ("moviepy.ae.templates.episode", "render_episode"),
        ("moviepy.ae.templates.episode", "init_episode"),
        ("moviepy.ae.templates.episode", "EpisodeError"),
        ("moviepy.ae.templates.subtitles", "parse_srt"),
        ("moviepy.ae.templates.subtitles", "find_bad_breaks"),
        ("moviepy.ae.templates.subtitles", "check_timing"),
        ("moviepy.ae.templates.ken_burns", "check_motion"),
        ("moviepy.ae.templates.title_card", "build_title_card"),
        ("moviepy.ae.templates.chapter_tag", "chapter_tag"),
        ("moviepy.ae.templates.quote", "vertical_quote"),
        ("moviepy.ae.templates.ken_burns", "ken_burns"),
        ("moviepy.ae.templates.subtitles", "burn_subtitles"),
    ],
)
def test_named_api_exists_and_is_mentioned(doc_text, module, name):
    mod = __import__(module, fromlist=[name])
    assert hasattr(mod, name)
    assert name in doc_text


def test_master_api_and_outputs_mentioned(doc_text):
    from moviepy.ae.master import write_master

    assert callable(write_master)
    for item in ("episode.mp4", "report.json", "stills/", "master/", "--master"):
        assert item in doc_text, item
