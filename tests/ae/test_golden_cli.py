"""Exercise the real root option and nested fixtures in isolated pytest trees."""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import pytest

from tests.ae.assets.generate_ws00 import reference_image


@pytest.fixture
def probe_tree(tmp_path):
    repository = Path(__file__).parents[2]
    shutil.copyfile(repository / "tests" / "conftest.py", tmp_path / "conftest.py")
    ae = tmp_path / "ae"
    ae.mkdir()
    shutil.copyfile(repository / "tests" / "ae" / "conftest.py", ae / "conftest.py")
    (ae / "golden").mkdir()
    return tmp_path


def run_probe(tree, source, *options):
    (tree / "ae" / "test_probe.py").write_text(source, encoding="utf-8")
    environment = os.environ.copy()
    repository = str(Path(__file__).parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [repository, environment.get("PYTHONPATH")])
    )
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "ae/test_probe.py", "-q", *options],
        cwd=tree,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def reference_probe():
    return (
        "from tests.ae.assets.generate_ws00 import reference_image\n"
        "def test_reference(assert_image_close, ae_golden_dir):\n"
        "    actual = reference_image('half_alpha_over')\n"
        "    assert_image_close(actual, ae_golden_dir / 'half_alpha_over.png', 0)\n"
    )


def test_default_is_compare_only(probe_tree):
    result = run_probe(probe_tree, reference_probe())
    assert result.returncode == 1, result.stdout + result.stderr
    assert "missing" in result.stdout
    assert list((probe_tree / "ae" / "golden").iterdir()) == []


def test_update_generates_independent_reference_and_rechecks(probe_tree):
    result = run_probe(probe_tree, reference_probe(), "--update-golden")
    assert result.returncode == 0, result.stdout + result.stderr
    path = probe_tree / "ae" / "golden" / "half_alpha_over.png"
    with Image.open(path) as image:
        actual = np.asarray(image).copy()
    np.testing.assert_array_equal(actual, reference_image(path.stem))
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.parent.iterdir()
    }
    result = run_probe(probe_tree, reference_probe())
    assert result.returncode == 0, result.stdout + result.stderr
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.parent.iterdir()
    }


def test_update_cannot_bless_broken_actual(probe_tree):
    source = reference_probe().replace(
        "actual = reference_image('half_alpha_over')",
        "actual = reference_image('half_alpha_over') * 0",
    )
    result = run_probe(probe_tree, source, "--update-golden")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "max error" in result.stdout
    path = probe_tree / "ae" / "golden" / "half_alpha_over.png"
    with Image.open(path) as image:
        actual = np.asarray(image).copy()
    np.testing.assert_array_equal(actual, reference_image(path.stem))


def test_unknown_recipe_fails_without_writing(probe_tree):
    source = reference_probe().replace("'half_alpha_over.png'", "'unknown.png'")
    result = run_probe(probe_tree, source, "--update-golden")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "unknown" in result.stdout and "recipe" in result.stdout
    assert list((probe_tree / "ae" / "golden").iterdir()) == []


def test_update_validates_actual_before_writing(probe_tree):
    source = reference_probe().replace(
        "actual = reference_image('half_alpha_over')",
        "actual = reference_image('half_alpha_over')[:1]",
    )
    result = run_probe(probe_tree, source, "--update-golden")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "actual shape" in result.stdout
    assert list((probe_tree / "ae" / "golden").iterdir()) == []


def test_update_rejects_path_outside_golden_root(probe_tree):
    source = reference_probe().replace(
        "ae_golden_dir / 'half_alpha_over.png'",
        "ae_golden_dir.parent / 'half_alpha_over.png'",
    )
    result = run_probe(probe_tree, source, "--update-golden")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "golden root" in result.stdout
    assert not (probe_tree / "ae" / "half_alpha_over.png").exists()


def test_option_is_registered_once_for_nested_collection(probe_tree):
    result = run_probe(probe_tree, reference_probe(), "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("--update-golden") == 1
