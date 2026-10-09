"""Checks that ``docs/ae/music_workflow.rst`` matches the music episode code.

The guide is the user's manual for ``python -m moviepy.ae.templates music``.
These tests keep it honest: code blocks must parse, CLI subcommands and flags
must exist in the real argparse parser, every ``music.json`` field in the
tables must exist in its dataclass (and every dataclass field must be
documented), the literal defaults must equal the dataclass defaults, the
quoted error messages must exist in the source, and the guide must be
reachable from ``templates.rst``.
"""

import ast
import dataclasses
import inspect
import json
import re
import shlex
import textwrap
from dataclasses import MISSING, fields
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "ae" / "music_workflow.rst"
TEMPLATES_DOC = REPO / "docs" / "ae" / "templates.rst"
CONFIGS = REPO / "moviepy" / "ae" / "templates" / "configs"
SOURCE = REPO / "moviepy" / "ae" / "templates" / "music_episode.py"
PREFIX = "python -m moviepy.ae.templates"
SUBCOMMANDS = ("init", "validate", "build")

# Dataclass behind each heading key in the field tables.
SECTION_CLASSES = {
    "music.json": "MusicEpisodeSpec",
    "tracks[]": "TrackSpec",
    "audio": "MusicAudioSpec",
    "visual": "VisualSpec",
    "spectrum": "SpectrumSpec",
    "chime": "ChimeSpec",
    "ambience": "AmbienceSpec",
    "ambience.particles": "ParticlesSpec",
    "ambience.light_arc": "LightArcSpec",
    "ambience.sleep_fade": "SleepFadeSpec",
    "cards": "CardsSpec",
    "thumbnail": "ThumbnailStepSpec",
    "qa": "QaSpec",
}

# Object-valued sections whose keys must exist in their dataclass.
NESTED_CLASSES = {
    "audio": "MusicAudioSpec",
    "visual": "VisualSpec",
    "spectrum": "SpectrumSpec",
    "chime": "ChimeSpec",
    "ambience": "AmbienceSpec",
    "cards": "CardsSpec",
    "thumbnail": "ThumbnailStepSpec",
    "qa": "QaSpec",
}
AMBIENCE_CLASSES = {
    "particles": "ParticlesSpec",
    "light_arc": "LightArcSpec",
    "sleep_fade": "SleepFadeSpec",
}


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
    """Return ``{"": top_help, "init": ..., "validate": ..., "build": ...}``.

    The help is read through the real ``python -m moviepy.ae.templates``
    dispatcher, so the ``music`` prefix is exercised as users type it.
    """
    from moviepy.ae.templates.__main__ import run

    helps = {}
    for sub in ("",) + SUBCOMMANDS:
        argv = ["music", "--help"] if not sub else ["music", sub, "--help"]
        capsys.readouterr()
        with pytest.raises(SystemExit) as exit_info:
            run(argv)
        assert exit_info.value.code == 0
        helps[sub] = capsys.readouterr().out
    return helps


def _tables(text):
    """Yield ``(heading, rows)`` for each ``list-table``.

    A heading is a text line directly followed by a ``-``, ``~`` or ``=``
    underline. ``rows`` holds every row (header included) as a list of cell
    strings; continuation lines are joined onto their cell.
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
        rows = []
        for follow in lines[index + 1 :]:
            if follow.strip() and not follow.startswith(" "):
                break
            match = re.match(r"^\s*\* - (.*)$", follow)
            if match:
                rows.append([match.group(1).strip()])
                continue
            match = re.match(r"^\s+- (.*)$", follow)
            if match and rows:
                rows[-1].append(match.group(1).strip())
                continue
            if rows and follow.strip() and follow.startswith("      "):
                rows[-1][-1] += " " + follow.strip()
        tables.append((heading, rows))
    return tables


def _field_tables(text):
    """Return ``[(heading, [first-column names])]`` for the field tables.

    The header row is skipped; only ``None`` names (no backticks) are dropped.
    """
    result = []
    for heading, rows in _tables(text):
        names = []
        for row in rows[1:]:
            names.append(re.findall(r"``([^`]+)``", row[0]))
        result.append((heading, names))
    return result


def _normal(value):
    """Compare JSON-style: tuples become lists, so defaults can be compared."""
    if isinstance(value, (tuple, list)):
        return [_normal(v) for v in value]
    if isinstance(value, dict):
        return {k: _normal(v) for k, v in value.items()}
    return value


def _dataclass_default(field):
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return MISSING


def _assert_nested_keys(data, label):
    """Every key of an object-valued section must be a field of its class."""
    from moviepy.ae.templates import music_episode

    for key, class_name in NESTED_CLASSES.items():
        section = data.get(key)
        if isinstance(section, dict):
            known = _dataclass_names(getattr(music_episode, class_name))
            assert set(section) <= known, (label, key, sorted(set(section) - known))
    ambience = data.get("ambience")
    if isinstance(ambience, dict):
        for key, class_name in AMBIENCE_CLASSES.items():
            section = ambience.get(key)
            if isinstance(section, dict):
                known = _dataclass_names(getattr(music_episode, class_name))
                assert set(section) <= known, (label, key, sorted(set(section) - known))


# --------------------------------------------------------------------------- #
# structure and links
# --------------------------------------------------------------------------- #


def test_doc_is_orphan_and_has_title(doc_text):
    assert doc_text.splitlines()[0] == ":orphan:"
    assert "音樂頻道長片製作流程" in doc_text.splitlines()[2]


def test_templates_rst_links_to_guide():
    text = TEMPLATES_DOC.read_text(encoding="utf-8")
    assert "音樂頻道製作流程" in text
    assert ":doc:`music_workflow`" in text


def test_guide_links_back_to_templates(doc_text):
    assert ":doc:`templates`" in doc_text


def test_section_headings_are_chinese(doc_text):
    lines = doc_text.splitlines()
    headings = [
        lines[i]
        for i in range(len(lines) - 1)
        if lines[i].strip() and re.fullmatch(r"[-~=]{3,}", lines[i + 1].strip() or "x")
    ]
    assert len(headings) >= 20
    for heading in headings:
        assert re.search(r"[一-鿿]", heading), heading


# --------------------------------------------------------------------------- #
# code blocks
# --------------------------------------------------------------------------- #


def test_python_blocks_parse(doc_text):
    blocks = _code_blocks(doc_text, "python")
    assert len(blocks) >= 6
    for body in blocks:
        ast.parse(body)


def test_json_blocks_parse(doc_text):
    blocks = _code_blocks(doc_text, "json")
    for body in blocks:
        json.loads(body)


def test_python_call_keywords_exist_in_signatures(doc_text):
    from moviepy.ae.templates import (
        music_audio,
        music_episode,
        music_render,
        spectrum,
        study,
        visual_loop,
    )
    from moviepy.ae.templates.study import StudySchedule

    known = {
        "normalize_loudness": music_audio.normalize_loudness,
        "assemble_chapters": music_audio.assemble_chapters,
        "mix_cues": music_audio.mix_cues,
        "write_chime": music_audio.write_chime,
        "analyze_spectrum": spectrum.analyze_spectrum,
        "save_levels": spectrum.save_levels,
        "load_levels": spectrum.load_levels,
        "SpectrumLayer": spectrum.SpectrumLayer,
        "StudyOverlayLayer": study.StudyOverlayLayer,
        "StudySchedule.from_json": StudySchedule.from_json,
        "to_ass": StudySchedule.to_ass,
        "seamless_loop": visual_loop.seamless_loop,
        "scene_cycle": visual_loop.scene_cycle,
        "export_loop": visual_loop.export_loop,
        "render_music_video": music_render.render_music_video,
        "preview_window": music_render.preview_window,
        "frame_count": music_render.frame_count,
        "build_music_episode": music_episode.build_music_episode,
        "validate_music_episode": music_episode.validate_music_episode,
    }
    checked = 0
    for body in _code_blocks(doc_text, "python"):
        for node in ast.walk(ast.parse(body)):
            if not isinstance(node, ast.Call):
                continue
            target = _call_target(node, known)
            if target is None:
                continue
            checked += 1
            params = inspect.signature(target).parameters
            if any(p.kind == p.VAR_KEYWORD for p in params.values()):
                continue
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                assert keyword.arg in params, f"{ast.dump(node.func)}({keyword.arg}=)"
    assert checked >= 12


def _call_target(node, known):
    func = node.func
    if isinstance(func, ast.Name):
        return known.get(func.id)
    if isinstance(func, ast.Attribute):
        owner = func.value.id if isinstance(func.value, ast.Name) else ""
        key = f"{owner}.{func.attr}"
        if key in known:
            return known[key]
        if func.attr == "to_ass":
            return known["to_ass"]
    return None


# --------------------------------------------------------------------------- #
# CLI: subcommands and flags exist in the real parser
# --------------------------------------------------------------------------- #


def test_cli_subcommands_exist(doc_text, capsys):
    helps = _cli_help(capsys)
    choices = set(re.search(r"\{([a-z,]+)\}", helps[""]).group(1).split(","))
    assert choices == set(SUBCOMMANDS)
    mentioned = set(re.findall(r"templates[ \t]+music[ \t]+([a-z]+)", doc_text))
    assert mentioned == choices


def test_cli_console_lines_use_known_flags(doc_text, capsys):
    helps = _cli_help(capsys)
    lines = []
    for body in _code_blocks(doc_text, "console"):
        lines += [ln.strip() for ln in body.splitlines() if PREFIX in ln]
    assert len(lines) >= 5
    for line in lines:
        tokens = shlex.split(line.split(PREFIX, 1)[1].replace("\\", "/"))
        assert tokens[0] == "music", line
        sub = tokens[1]
        assert sub in SUBCOMMANDS, line
        for token in tokens:
            if token.startswith("--"):
                assert token in helps[sub], f"{token} not accepted by {sub}: {line}"


def test_every_flag_in_doc_exists_in_some_subcommand(doc_text, capsys):
    helps = _cli_help(capsys)
    known = " ".join(helps.values())
    flags = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", doc_text))
    assert {"--steps", "--preview", "--workers", "--encoder", "--mode"} <= flags
    for flag in flags:
        assert flag in known, flag


def test_modes_and_steps_match_code(doc_text):
    from moviepy.ae.templates.music_episode import ENCODERS, MODES, STEPS

    assert set(MODES) == {"sleep_longform", "study_pomodoro"}
    for mode in MODES:
        assert f"``{mode}``" in doc_text or f'"{mode}"' in doc_text
    assert ",".join(STEPS) == "audio,visual,overlays,render,thumbnail,qa"
    assert "audio,visual,overlays,render" in doc_text
    for step in STEPS:
        assert f"``{step}``" in doc_text, step
    for encoder in ENCODERS:
        assert f"``{encoder}``" in doc_text or f'"{encoder}"' in doc_text, encoder


def test_step_table_lists_every_step_in_order(doc_text):
    from moviepy.ae.templates.music_episode import STEPS

    for _, rows in _tables(doc_text):
        if rows and rows[0][0] == "步驟":
            names = [re.findall(r"``([^`]+)``", row[0])[0] for row in rows[1:]]
            assert names == list(STEPS)
            return
    raise AssertionError("the steps table (header 步驟) is missing")


def test_optional_steps_are_opt_in():
    from moviepy.ae.templates.music_episode import MusicEpisodeSpec, _step_enabled

    spec = MusicEpisodeSpec.from_dict(
        {
            "mode": "sleep_longform",
            "tracks": [{"id": "A", "source": "a.wav", "chapter_seconds": 60}],
            "visual": {"clips": ["c.mp4"]},
        }
    )
    assert not _step_enabled(spec, "thumbnail")
    assert not _step_enabled(spec, "qa")
    assert "預設關閉" in DOC.read_text(encoding="utf-8")


def test_ring_and_timeline_need_study_mode():
    from moviepy.ae.templates.music_episode import MusicEpisodeError, MusicEpisodeSpec

    data = {
        "mode": "sleep_longform",
        "tracks": [{"id": "A", "source": "a.wav", "chapter_seconds": 60}],
        "visual": {"clips": ["c.mp4"]},
    }
    for key in ("study_ring", "study_timeline"):
        with pytest.raises(MusicEpisodeError, match="need study_pomodoro mode"):
            MusicEpisodeSpec.from_dict(dict(data, **{key: True}))


def test_enumerated_options_are_documented(doc_text):
    from moviepy.ae.templates.ambience import _MAX_RATE, PARTICLE_KINDS
    from moviepy.ae.templates.music_episode import (
        _CARD_POSITIONS as POSITIONS,
        _LIGHT_PRESETS,
    )
    from moviepy.ae.templates.music_qa import DEFAULT_TARGETS
    from moviepy.ae.templates.thumbnail import LAYOUTS, MAIN_CONTRAST

    for kind in PARTICLE_KINDS:
        assert f"``{kind}``" in doc_text, kind
    for preset in _LIGHT_PRESETS:
        assert f"``{preset}``" in doc_text, preset
    for position in POSITIONS:
        assert f"``{position}``" in doc_text, position
    for layout in LAYOUTS:
        assert f"``{layout}``" in doc_text, layout
    for target in DEFAULT_TARGETS:
        assert f"``{target}``" in doc_text, target
    assert f"{MAIN_CONTRAST}:1" in doc_text
    assert f"{_MAX_RATE:g}" in doc_text
    assert "-18" in doc_text and "-1.5" in doc_text


# --------------------------------------------------------------------------- #
# music.json field tables
# --------------------------------------------------------------------------- #


def _dataclass_names(cls):
    return {f.name for f in fields(cls)}


def test_field_tables_match_dataclasses(doc_text):
    from moviepy.ae.templates import music_episode

    seen = set()
    for heading, names in _field_tables(doc_text):
        keys = re.findall(r"``([^`]+)``", heading)
        if not keys or keys[0] not in SECTION_CLASSES:
            continue
        section = keys[0]
        cls = getattr(music_episode, SECTION_CLASSES[section])
        flat = [name for row in names for name in row]
        assert len(flat) == len(set(flat)), f"duplicate rows under {heading}"
        expected = _dataclass_names(cls)
        unknown = set(flat) - expected
        assert (
            not unknown
        ), f"{section}: not fields of {cls.__name__}: {sorted(unknown)}"
        missing = expected - set(flat)
        assert not missing, f"{section}: undocumented fields: {sorted(missing)}"
        seen.add(section)
    assert seen == set(SECTION_CLASSES), set(SECTION_CLASSES) - seen


def test_field_table_defaults_equal_dataclass_defaults(doc_text):
    from moviepy.ae.templates import music_episode

    checked = 0
    for heading, rows in _tables(doc_text):
        keys = re.findall(r"``([^`]+)``", heading)
        if not keys or keys[0] not in SECTION_CLASSES:
            continue
        cls = getattr(music_episode, SECTION_CLASSES[keys[0]])
        by_name = {f.name: f for f in fields(cls)}
        for row in rows[1:]:
            name = re.findall(r"``([^`]+)``", row[0])[0]
            literal = re.fullmatch(r"``(.+)``", row[2].strip())
            if not literal:
                continue
            documented = json.loads(literal.group(1))
            default = _dataclass_default(by_name[name])
            assert default is not MISSING, name
            assert _normal(documented) == _normal(default), (keys[0], name)
            checked += 1
    assert checked >= 30


def test_shipped_configs_use_documented_keys():
    from moviepy.ae.templates.music_episode import MusicEpisodeSpec

    known = _dataclass_names(MusicEpisodeSpec)
    for path in sorted(CONFIGS.glob("music_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) <= known, path.name
        for track in data["tracks"]:
            assert set(track) <= _dataclass_names(_track_class()), path.name
        _assert_nested_keys(data, path.name)


def _track_class():
    from moviepy.ae.templates.music_episode import TrackSpec

    return TrackSpec


def test_json_example_keys_exist(doc_text):
    from moviepy.ae.templates.music_episode import MusicEpisodeSpec

    known = _dataclass_names(MusicEpisodeSpec)
    blocks = _code_blocks(doc_text, "json")
    assert blocks, "the guide should show a music.json example"
    assert len(blocks) >= 8
    for body in blocks:
        data = json.loads(body)
        assert set(data) <= known, sorted(set(data) - known)
        for track in data.get("tracks", []):
            assert set(track) <= _dataclass_names(_track_class())
        _assert_nested_keys(data, "guide json example")


def test_shipped_configs_total_lengths_match_guide():
    from moviepy.ae.templates.music_episode import (
        MusicEpisodeSpec,
        pomodoro_schedule,
        validate_music_episode,
    )

    sleep = json.loads((CONFIGS / "music_sleep_longform.json").read_text("utf-8"))
    spec = MusicEpisodeSpec.from_dict(sleep)
    plan = validate_music_episode(spec, check_files=False)["plan"]
    assert plan["total_seconds"] == 19200
    assert plan["video_frames"] == 460800

    study = json.loads((CONFIGS / "music_study_pomodoro.json").read_text("utf-8"))
    study["study"] = pomodoro_schedule()
    spec = MusicEpisodeSpec.from_dict(study)
    plan = validate_music_episode(spec, check_files=False)["plan"]
    assert plan["total_seconds"] == 5220
    assert plan["expected_frames"] == 5220 * 48000


def test_config_defaults_match_guide_table_values():
    from moviepy.ae.templates.music_episode import MusicEpisodeSpec

    spec = MusicEpisodeSpec.from_json(CONFIGS / "music_study_pomodoro.json")
    assert spec.spectrum.enabled is True
    assert spec.spectrum.position == (96, 548)
    assert spec.spectrum.size == (1088, 140)
    assert spec.chime.enabled is True and spec.chime.below_music_db == 10.0
    assert spec.study_compact is False
    assert spec.ambience.particles is None and not spec.ambience.light_arc.active
    assert spec.ambience.sleep_fade is None
    assert spec.cards.enabled is True and spec.cards.skip_first is False
    assert spec.study_ring is True and spec.study_timeline is True
    assert spec.thumbnail.enabled is True and spec.thumbnail.time == 60
    assert spec.qa.enabled is True
    sleep = MusicEpisodeSpec.from_json(CONFIGS / "music_sleep_longform.json")
    assert sleep.spectrum.enabled is False and sleep.chime.enabled is False
    assert sleep.visual.segment == 120.0 and sleep.visual.clip_length == 8.0
    assert sleep.ambience.particles.kind == "fireflies"
    assert sleep.ambience.particles.count == 18 and sleep.ambience.particles.seed == 7
    assert sleep.ambience.particles.opacity == 0.6
    assert sleep.ambience.light_arc.preset == "day_to_night"
    assert sleep.ambience.sleep_fade.start == -1800
    assert sleep.ambience.sleep_fade.floor == 0.15
    assert sleep.ambience.sleep_fade.audio is True
    assert sleep.cards.enabled is True and sleep.thumbnail.enabled is True
    assert sleep.study_ring is False and sleep.study_timeline is False
    assert sleep.qa.enabled is True


# --------------------------------------------------------------------------- #
# error messages quoted in the guide
# --------------------------------------------------------------------------- #


def test_quoted_error_fragments_exist_in_source(doc_text):
    source = SOURCE.read_text(encoding="utf-8")
    error_rows = []
    for heading, rows in _tables(doc_text):
        if "MusicEpisodeError" in heading:
            error_rows += rows[1:]
    assert len(error_rows) >= 12
    checked = 0
    for row in error_rows:
        for fragment in re.findall(r"``([^`]+)``", row[0]):
            assert fragment in source, fragment
            checked += 1
    assert checked >= 12


# --------------------------------------------------------------------------- #
# API names the guide points at
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "module, name",
    [
        ("moviepy.ae.templates.music_episode", "MusicEpisodeSpec"),
        ("moviepy.ae.templates.music_episode", "MusicEpisodeError"),
        ("moviepy.ae.templates.music_episode", "build_music_episode"),
        ("moviepy.ae.templates.music_episode", "validate_music_episode"),
        ("moviepy.ae.templates.music_episode", "init_music_episode"),
        ("moviepy.ae.templates.music_episode", "pomodoro_schedule"),
        ("moviepy.ae.templates.music_episode", "music_paths"),
        ("moviepy.ae.templates.music_audio", "normalize_loudness"),
        ("moviepy.ae.templates.music_audio", "assemble_chapters"),
        ("moviepy.ae.templates.music_audio", "mix_cues"),
        ("moviepy.ae.templates.music_audio", "write_chime"),
        ("moviepy.ae.templates.music_audio", "chime_tone"),
        ("moviepy.ae.templates.spectrum", "analyze_spectrum"),
        ("moviepy.ae.templates.spectrum", "SpectrumLayer"),
        ("moviepy.ae.templates.spectrum", "save_levels"),
        ("moviepy.ae.templates.spectrum", "load_levels"),
        ("moviepy.ae.templates.study", "StudySchedule"),
        ("moviepy.ae.templates.study", "StudyOverlayLayer"),
        ("moviepy.ae.templates.visual_loop", "seamless_loop"),
        ("moviepy.ae.templates.visual_loop", "scene_cycle"),
        ("moviepy.ae.templates.visual_loop", "export_loop"),
        ("moviepy.ae.templates.music_render", "render_music_video"),
        ("moviepy.ae.templates.music_render", "preview_window"),
        ("moviepy.ae.templates.music_render", "frame_count"),
        ("moviepy.ae.templates.music_episode", "make_thumbnail"),
        ("moviepy.ae.templates.music_episode", "run_qa"),
        ("moviepy.ae.templates.music_episode", "resolve_ambience"),
        ("moviepy.ae.templates.music_episode", "prepare_particles"),
        ("moviepy.ae.templates.ambience", "particle_loop"),
        ("moviepy.ae.templates.ambience", "export_overlay_loop"),
        ("moviepy.ae.templates.ambience", "LightArc"),
        ("moviepy.ae.templates.ambience", "SleepFade"),
        ("moviepy.ae.templates.music_cards", "track_cards"),
        ("moviepy.ae.templates.music_cards", "combine_study_ass"),
        ("moviepy.ae.templates.music_cards", "sleep_cards_ass"),
        ("moviepy.ae.templates.thumbnail", "ThumbnailSpec"),
        ("moviepy.ae.templates.thumbnail", "render_thumbnail"),
        ("moviepy.ae.templates.thumbnail", "grab_frame"),
        ("moviepy.ae.templates.music_qa", "qa_report"),
        ("moviepy.ae.templates.music_qa", "flash_verdict"),
    ],
)
def test_named_api_exists_and_is_mentioned(doc_text, module, name):
    mod = __import__(module, fromlist=[name])
    assert hasattr(mod, name)
    assert name in doc_text, name


def test_study_schedule_methods_exist_and_are_mentioned(doc_text):
    from moviepy.ae.templates.study import StudySchedule

    for name in ("from_json", "to_ass", "chime_times", "chapters_ffmetadata"):
        assert hasattr(StudySchedule, name), name
        assert name in doc_text, name


def test_output_names_mentioned(doc_text):
    for item in (
        "music.flac",
        "audio.flac",
        "chime.wav",
        "loop-01.mp4",
        "cycle.mp4",
        "levels.npy",
        "study.ass",
        "chapters.ffmeta",
        "render/",
        ".ffmpeg.log",
        "NOT_RUN",
        "NVENC",
        "particles.mov",
        "cards.ass",
        "publish/",
        "thumbnail.jpg",
        "thumbnail.jpg.json",
        "qa/",
        ".qa.json",
        "render_sha256",
        "partial",
    ):
        assert item in doc_text, item


# --------------------------------------------------------------------------- #
# init creates what the guide describes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["sleep_longform", "study_pomodoro"])
def test_init_creates_documented_layout(tmp_path, doc_text, mode):
    from moviepy.ae.templates.music_episode import init_music_episode

    target = tmp_path / "ep"
    config = Path(init_music_episode(target, mode))
    assert config == target / "music.json"
    assert (target / "tracks").is_dir() and (target / "visual").is_dir()
    assert (target / "README.txt").is_file()
    assert (target / "study.json").is_file() == (mode == "study_pomodoro")
    assert "``study.json``" in doc_text
    readme = (target / "README.txt").read_text(encoding="utf-8")
    for gate in ("完整試聽", "完整畫面觀看", "版權與授權", "私人上傳"):
        assert gate in readme and gate in doc_text, gate
    assert readme.count("NOT_RUN") == 4


def test_human_gates_are_the_four_documented_ones(doc_text):
    from moviepy.ae.templates.music_episode import HUMAN_GATES

    assert set(HUMAN_GATES) == {
        "full_listening",
        "full_visual_watch",
        "rights",
        "private_upload",
    }
    assert all(value == "NOT_RUN" for value in HUMAN_GATES.values())
    for key in HUMAN_GATES:
        assert f"``{key}``" in doc_text, key


def test_dataclass_field_counts_documented():
    from moviepy.ae.templates.music_episode import (
        AmbienceSpec,
        CardsSpec,
        ChimeSpec,
        LightArcSpec,
        MusicAudioSpec,
        MusicEpisodeSpec,
        ParticlesSpec,
        QaSpec,
        SleepFadeSpec,
        SpectrumSpec,
        ThumbnailStepSpec,
        TrackSpec,
        VisualSpec,
    )

    assert len(dataclasses.fields(MusicEpisodeSpec)) == 23
    assert len(dataclasses.fields(TrackSpec)) == 4
    assert len(dataclasses.fields(MusicAudioSpec)) == 7
    assert len(dataclasses.fields(VisualSpec)) == 6
    assert len(dataclasses.fields(SpectrumSpec)) == 12
    assert len(dataclasses.fields(ChimeSpec)) == 4
    assert len(dataclasses.fields(ParticlesSpec)) == 7
    assert len(dataclasses.fields(LightArcSpec)) == 3
    assert len(dataclasses.fields(SleepFadeSpec)) == 5
    assert len(dataclasses.fields(AmbienceSpec)) == 3
    assert len(dataclasses.fields(CardsSpec)) == 6
    assert len(dataclasses.fields(ThumbnailStepSpec)) == 8
    assert len(dataclasses.fields(QaSpec)) == 3
