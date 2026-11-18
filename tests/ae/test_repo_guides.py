"""Check entry-point guides: links resolve, quickstart runs, no drive paths."""

import re
from pathlib import Path

import pytest


def _repo():
    return Path(__file__).resolve().parents[2]


GUIDES = ["AGENTS.md", "CLAUDE.md", "README.md", "docs/ae/README.md", "CONTRIBUTING.md"]
LINKED = ["AGENTS.md", "docs/ae/README.md"]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
DRIVE = re.compile(r"(?<![A-Za-z])[A-Za-z]:\\")

REPO = _repo()


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("rel", LINKED)
def test_relative_links_exist(rel):
    base = (REPO / rel).parent
    missing = []
    for target in LINK.findall(_read(rel)):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        if not (base / target.split("#")[0]).exists():
            missing.append(target)
    assert not missing, f"dead links in {rel}: {missing}"


def test_readme_quickstart_runs(capsys):
    text = _read("README.md")
    section = text.split("# AE-style compositing (moviepy.ae)", 1)[1]
    blocks = re.findall(r"``` python\n(.*?)```", section, re.S)
    assert blocks, "no python quickstart in the README ae section"
    exec(compile(blocks[0], "README-ae-quickstart", "exec"), {})
    assert "[[[0, 128, 0], [0, 128, 0]]]" in capsys.readouterr().out


@pytest.mark.parametrize("rel", GUIDES)
def test_no_drive_letter_paths(rel):
    assert not DRIVE.search(_read(rel)), f"drive-letter path in {rel}"
