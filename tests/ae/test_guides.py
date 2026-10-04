"""Execute every ``.. code-block:: python`` example in the AE guides."""

import re
from pathlib import Path

import pytest


GUIDE_DIR = Path(__file__).resolve().parents[2] / "docs" / "ae"
GUIDES = ["ws02.rst", "ws04.rst"]
_BLOCK = re.compile(r"^\.\. code-block:: python\n\n((?:(?:    .*)?\n)+)", re.MULTILINE)


def guide_blocks(name):
    text = (GUIDE_DIR / name).read_text(encoding="utf-8")
    blocks = []
    for match in _BLOCK.finditer(text):
        lines = match.group(1).splitlines()
        blocks.append("\n".join(line[4:] for line in lines))
    return blocks


@pytest.mark.parametrize("name", GUIDES)
def test_guide_examples_run(name):
    blocks = guide_blocks(name)
    assert blocks, f"{name} has no python examples"
    for index, source in enumerate(blocks):
        namespace = {"__name__": f"guide_{index}"}
        exec(compile(source, f"{name}[{index}]", "exec"), namespace)
