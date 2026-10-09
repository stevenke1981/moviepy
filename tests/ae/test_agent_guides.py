"""Checks that the agent guides in ``docs/ae/agents`` match the real code.

The guides are operating manuals that an AI agent follows without a human.
A wrong flag, a misspelled spec field, a quoted error message that the code
never raises, or a dead doc link would send the agent in the wrong direction.
These tests keep the guides honest:

* every ``python -m moviepy.ae.templates ...`` line parses with the real
  argparse help (subcommand, positional count, ``--flag`` and choices);
* every spec field named in a quick-reference table exists on its dataclass;
* every error fragment quoted in a failure table occurs in the source;
* every ``docs/...`` and ``moviepy/...`` path mentioned exists;
* each guide has front matter with ``name`` and ``description`` and the
  eight required sections.
"""

import contextlib
import dataclasses
import importlib
import io
import re
from pathlib import Path

import pytest

from moviepy.ae.templates import episode as episode_mod, music_episode as music_mod
from moviepy.ae.templates.__main__ import run as cli_run


REPO = Path(__file__).resolve().parents[2]
AGENTS = REPO / "docs" / "ae" / "agents"
PREFIX = "python -m moviepy.ae.templates"

GUIDES = {
    "music_channel.md": {
        "name": "music-channel-episode",
        "module": music_mod,
        "triggers": ["森息音界", "睡眠長片", "讀書片"],
    },
    "nightlamp_episode.md": {
        "name": "nightlamp-episode",
        "module": episode_mod,
        "triggers": ["夜燈說書", "夜燈史話", "新集"],
    },
}

REQUIRED_SECTIONS = [
    "## 1. ",
    "## 2. ",
    "## 3. ",
    "## 4. ",
    "## 5. ",
    "## 6. ",
    "## 7. ",
    "## 8. ",
]

# Values accepted by flags, checked against the code's own constants.
VALUE_CHOICES = {
    "--mode": music_mod.MODES,
    "--channel": episode_mod.CHANNELS,
    "--encoder": music_mod.ENCODERS,
}
STEPS = music_mod.STEPS


def _read(name):
    return (AGENTS / name).read_text(encoding="utf-8")


def _source_text():
    chunks = []
    for path in sorted((REPO / "moviepy" / "ae").rglob("*.py")):
        chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


SOURCE = _source_text()


def _front_matter(text):
    match = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if match is None:
        return {}
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def _section(text, prefix):
    """Return the text of the ``## <prefix>`` section up to the next ``##``."""
    start = text.find("\n" + prefix)
    if start < 0:
        return ""
    start += 1
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def _table_rows(section):
    """Return the cells of each data row of the markdown tables in a section."""
    rows = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or set(stripped) <= set("|- "):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        rows.append(cells)
    return rows[1:] if rows else rows  # first row is the header


def _command_lines(text):
    """Return (line number, command) for every command in code blocks or inline code."""
    body = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
    found = []
    in_fence = False
    for number, line in enumerate(body.splitlines(), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            if line.strip().startswith(PREFIX):
                found.append((number, line.strip()))
            continue
        for span in re.findall(r"`(python -m moviepy\.ae\.templates[^`]*)`", line):
            found.append((number, span.strip()))
    return found


def _help(argv):
    """Return the help text printed by the real CLI for ``argv``."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            cli_run(list(argv) + ["--help"])
        except SystemExit as exit_info:
            assert exit_info.code in (0, None), argv
        else:  # pragma: no cover - argparse always exits on --help
            raise AssertionError(f"--help did not exit for {argv}")
    return buffer.getvalue()


def _subcommands(help_text):
    match = re.search(r"\{([\w,]+)\}", help_text)
    assert match, help_text
    return match.group(1).split(",")


def _positional_names(help_text):
    names, inside = [], False
    for line in help_text.splitlines():
        if line.startswith("positional arguments:"):
            inside = True
            continue
        if line.startswith("options:"):
            break
        if inside:
            match = re.match(r"^  ([A-Za-z_]\w*)\s*$", line)
            if match:
                names.append(match.group(1))
    return names


def _nargs(group, flag):
    if flag == "--preview" and group == "music":
        return 2
    if flag == "--still":
        return "*"
    if flag in VALUE_CHOICES or flag in ("--steps", "--workers"):
        return 1
    return 0


def _number(value, line, flag):
    try:
        float(value)
    except ValueError:
        raise AssertionError(f"{line!r}: {flag} needs numbers, got {value!r}") from None


def _check_module_command(line, tokens):
    """Check ``python -m moviepy.ae.templates.<module> --flags`` against its help."""
    module = importlib.import_module("moviepy.ae.templates" + tokens[0])
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as stop:
        module.main(["--help"])
    assert stop.value.code in (0, None), f"{line!r}: --help failed"
    for token in tokens[1:]:
        if token.startswith("--"):
            flag = token.split("=", 1)[0]
            assert flag in out.getvalue(), f"{line!r}: {flag} not in help"


def _check_command(line):
    """Parse one command line with the real CLI help; fail with a clear message."""
    command = re.split(r"\s+\d?>", line, maxsplit=1)[0]
    tokens = command[len(PREFIX) :].split()
    if command[len(PREFIX) :].startswith("."):
        _check_module_command(line, tokens)
        return
    group = []
    if tokens and tokens[0] == "music":
        group = ["music"]
        tokens = tokens[1:]
    if not tokens or tokens == ["--help"]:
        _help(group)  # the bare module or group help must print and exit 0
        return
    top = _help(group)
    assert tokens[0] in _subcommands(top), f"{line!r}: unknown subcommand"
    sub_argv = group + [tokens[0]]
    sub_help = _help(sub_argv)
    group_name = group[0] if group else ""
    positional = []
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            assert (
                token == "--help" or token in sub_help
            ), f"{line!r}: flag {token} not in help of {' '.join(sub_argv)}"
            count = _nargs(group_name, token)
            values = []
            if count == "*":
                index += 1
                while index < len(tokens) and not tokens[index].startswith("--"):
                    values.append(tokens[index])
                    index += 1
                assert values, f"{line!r}: {token} needs at least one value"
            else:
                values = tokens[index + 1 : index + 1 + count]
                assert len(values) == count, f"{line!r}: {token} needs {count} value(s)"
                index += 1 + count
            for value in values:
                if token == "--preview":
                    _number(value, line, token)
                elif token == "--steps":
                    for step in value.split(","):
                        assert step in STEPS, f"{line!r}: unknown step {step!r}"
                elif token == "--workers":
                    assert value.isdigit(), f"{line!r}: {token} needs an integer"
                elif token in VALUE_CHOICES:
                    assert (
                        value in VALUE_CHOICES[token]
                    ), f"{line!r}: {value!r} is not a choice of {token}"
            continue
        positional.append(token)
        index += 1
    expected = _positional_names(sub_help)
    assert len(positional) == len(
        expected
    ), f"{line!r}: {len(positional)} positional(s), help expects {expected}"


# --------------------------------------------------------------------------- #
# front matter and structure
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("guide", sorted(GUIDES))
def test_front_matter_has_name_and_description(guide):
    fields = _front_matter(_read(guide))
    assert fields.get("name") == GUIDES[guide]["name"]
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", fields["name"])
    assert len(fields.get("description", "")) >= 120


@pytest.mark.parametrize("guide", sorted(GUIDES))
def test_description_contains_trigger_phrases(guide):
    description = _front_matter(_read(guide))["description"]
    for phrase in GUIDES[guide]["triggers"]:
        assert phrase in description, f"{guide}: trigger {phrase!r} missing"


@pytest.mark.parametrize("guide", sorted(GUIDES))
def test_required_sections_present_in_order(guide):
    text = _read(guide)
    positions = []
    for prefix in REQUIRED_SECTIONS:
        index = text.find("\n" + prefix)
        assert index >= 0, f"{guide}: missing section {prefix!r}"
        positions.append(index)
    assert positions == sorted(positions), f"{guide}: sections out of order"


@pytest.mark.parametrize("guide", sorted(GUIDES))
def test_guide_has_no_future_import_line(guide):
    assert "from __future__ import annotations" not in _read(guide)


def test_readme_indexes_both_guides_and_install_path():
    readme = _read("README.md")
    for guide in GUIDES:
        assert guide in readme
        assert GUIDES[guide]["name"] in readme
    assert ".claude/skills/" in readme
    assert "SKILL.md" in readme


# --------------------------------------------------------------------------- #
# command lines
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("guide", sorted(GUIDES) + ["README.md"])
def test_every_command_line_parses_with_real_cli(guide):
    lines = _command_lines(_read(guide))
    if guide == "README.md":
        return
    assert lines, f"{guide}: no command lines found"
    for number, line in lines:
        try:
            _check_command(line)
        except AssertionError as error:
            raise AssertionError(f"{guide}:{number}: {error}") from None


def test_music_guide_mentions_every_music_subcommand_and_step():
    text = _read("music_channel.md")
    for word in ("init", "validate", "build", "--preview", "--workers", "--encoder"):
        assert word in text
    for step in STEPS:
        assert step in text


# --------------------------------------------------------------------------- #
# spec quick reference
# --------------------------------------------------------------------------- #


def _spec_rows(guide):
    section = _section(_read(guide), "## 4.")
    rows = []
    for cells in _table_rows(section):
        assert len(cells) >= 2, cells
        rows.append(cells)
    return rows


@pytest.mark.parametrize("guide", sorted(GUIDES))
def test_quick_reference_fields_exist_on_dataclasses(guide):
    module = GUIDES[guide]["module"]
    rows = _spec_rows(guide)
    assert len(rows) >= 10, f"{guide}: quick reference too short"
    for cells in rows:
        class_name = cells[1]
        cls = getattr(module, class_name, None)
        assert cls is not None and dataclasses.is_dataclass(
            cls
        ), f"{guide}: {class_name!r} is not a dataclass of the module"
        names = {field.name for field in dataclasses.fields(cls)}
        for token in re.findall(r"`([^`]+)`", cells[0]):
            leaf = token.replace("[]", "").split(".")[-1]
            assert (
                leaf in names
            ), f"{guide}: {token} is not a field of {class_name} ({sorted(names)})"


# --------------------------------------------------------------------------- #
# failure table
# --------------------------------------------------------------------------- #


def _failure_fragments(guide):
    section = _section(_read(guide), "## 7.")
    fragments = []
    for cells in _table_rows(section):
        for token in re.findall(r"`([^`]+)`", cells[0]):
            # Placeholders such as <path> are cut; the fixed text must match.
            fixed = re.split(r"<", token, maxsplit=1)[0].rstrip()
            if fixed:
                fragments.append(fixed)
    return fragments


@pytest.mark.parametrize("guide", sorted(GUIDES))
def test_quoted_error_messages_exist_in_source(guide):
    fragments = _failure_fragments(guide)
    assert len(fragments) >= 15, f"{guide}: failure table too short"
    missing = [frag for frag in fragments if frag not in SOURCE]
    assert not missing, f"{guide}: not raised by the source: {missing}"


# --------------------------------------------------------------------------- #
# referenced paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("guide", sorted(GUIDES) + ["README.md"])
def test_referenced_repo_paths_exist(guide):
    text = _read(guide)
    paths = set(re.findall(r"(?<![\w./-])((?:docs|moviepy|tests)/[\w./-]+)", text))
    assert paths or guide == "README.md"
    for raw in sorted(paths):
        path = REPO / raw.rstrip(".")
        assert path.exists(), f"{guide}: {raw} does not exist"
