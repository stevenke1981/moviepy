"""Checks that ``docs/ae/templates.rst`` and the showcase example stay runnable."""

import ast
import importlib
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "ae" / "templates.rst"
EXAMPLE = REPO / "examples" / "ae_templates_showcase.py"
KAIU = Path("C:/Windows/Fonts/kaiu.ttf")
EXPECTED_PNGS = {
    "title_card.png",
    "ken_burns_first.png",
    "ken_burns_last.png",
    "chapter_tag.png",
    "quote_bamboo.png",
    "quote_paper.png",
    "subtitle.png",
}


def _python_blocks(text):
    """Return the body of every ``.. code-block:: python`` directive."""
    lines = text.splitlines()
    blocks = []
    for index, line in enumerate(lines):
        if line.strip() != ".. code-block:: python":
            continue
        indent = len(line) - len(line.lstrip())
        body = []
        for follow in lines[index + 1 :]:
            if follow.strip() and len(follow) - len(follow.lstrip()) <= indent:
                break
            body.append(follow)
        blocks.append("\n".join(body))
    return blocks


def _resolve(dotted):
    """Import ``dotted`` as a module, or as an attribute of its parent module."""
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError:
        parent, _, leaf = dotted.rpartition(".")
        return getattr(importlib.import_module(parent), leaf)


def test_doc_is_orphan_and_has_code_blocks():
    text = DOC.read_text(encoding="utf-8")
    assert text.startswith(":orphan:")
    assert len(_python_blocks(text)) >= 8


def test_every_python_code_block_parses():
    blocks = _python_blocks(DOC.read_text(encoding="utf-8"))
    assert blocks, "no python code blocks found"
    for number, block in enumerate(blocks, 1):
        source = textwrap.dedent(block)
        try:
            ast.parse(source)
        except SyntaxError as error:  # pragma: no cover - failure detail only
            pytest.fail(f"code block {number} does not parse: {error}\n{source}")


def test_every_referenced_template_name_exists():
    text = DOC.read_text(encoding="utf-8")
    names = sorted(set(re.findall(r"moviepy\.ae\.templates\.([A-Za-z_][\w.]*)", text)))
    assert names, "the doc should reference moviepy.ae.templates names"
    missing = []
    for name in names:
        try:
            _resolve(f"moviepy.ae.templates.{name}")
        except (ImportError, AttributeError):
            missing.append(name)
    assert not missing, f"documented names do not exist: {missing}"


def test_showcase_refuses_non_empty_output_dir(tmp_path):
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "examples.ae_templates_showcase", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 1
    assert "not an empty directory" in result.stderr
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "x"


def test_showcase_renders_stills_at_small_size(tmp_path):
    if not KAIU.is_file():
        pytest.skip(f"font not found: {KAIU}")
    assert EXAMPLE.is_file()
    out = tmp_path / "stills"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.ae_templates_showcase",
            str(out),
            "--size",
            "320x180",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 0, result.stderr
    produced = {path.name for path in out.glob("*.png")}
    assert EXPECTED_PNGS <= produced

    from PIL import Image

    with Image.open(out / "title_card.png") as image:
        assert image.size == (320, 180)

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["size"] == [320, 180]
    assert report["total_seconds"] >= 0
    assert {step["file"] for step in report["steps"] if step["file"]} == EXPECTED_PNGS
