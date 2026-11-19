"""Validate independent image references and compare-only golden assertions."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import pytest

from tests.ae._golden import _publish_reference, assert_image_close, regenerate_golden
from tests.ae.assets.generate_ws00 import RECIPE_IDS, recipe_inputs, reference_image


@pytest.fixture
def sample():
    return np.array([[[0, 127, 255], [255, 17, 0]]], dtype=np.uint8)


@pytest.fixture
def golden_file(tmp_path, sample):
    path = tmp_path / "sample.png"
    Image.fromarray(sample).save(path)
    metadata = {
        "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": "RGB",
        "size": [2, 1],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def test_exact_comparison_does_not_modify_inputs(golden_file, sample):
    before = sample.copy()
    files = {p.name: p.read_bytes() for p in golden_file.parent.iterdir()}
    assert_image_close(sample, golden_file, 0)
    np.testing.assert_array_equal(sample, before)
    assert files == {p.name: p.read_bytes() for p in golden_file.parent.iterdir()}


@pytest.mark.parametrize(
    "change,tol,passes", [(1, 1 / 255, True), (1, 0, False), (2, 1 / 255, False)]
)
def test_normalized_tolerance(golden_file, sample, change, tol, passes):
    actual = sample.copy()
    actual[0, 0, 0] = change
    if passes:
        assert_image_close(actual, golden_file, tol)
    else:
        with pytest.raises(AssertionError, match=r"sample.*max error.*channel"):
            assert_image_close(actual, golden_file, tol)


def test_uint8_subtraction_does_not_wrap(golden_file, sample):
    actual = sample.copy()
    actual[0, 0, 2] = 0
    with pytest.raises(AssertionError, match="max error=1"):
        assert_image_close(actual, golden_file, 1 / 255)


@pytest.mark.parametrize("tol", [-1, 1.1, np.nan, np.inf, True, "0", [0]])
def test_invalid_tolerance(golden_file, sample, tol):
    with pytest.raises((TypeError, ValueError), match="tol"):
        assert_image_close(sample, golden_file, tol)


@pytest.mark.parametrize(
    "actual",
    [
        np.zeros((1, 2, 3), dtype=float),
        np.zeros((1, 2), dtype=np.uint8),
        np.zeros((1, 2, 2), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
    ],
)
def test_invalid_actual(golden_file, actual):
    with pytest.raises((TypeError, ValueError), match="actual"):
        assert_image_close(actual, golden_file, 0)


def test_shape_mismatch_is_explicit(golden_file):
    with pytest.raises(AssertionError, match="shape"):
        assert_image_close(np.zeros((2, 2, 3), dtype=np.uint8), golden_file, 0)


def test_mode_mismatch_is_explicit(golden_file):
    with pytest.raises(AssertionError, match="mode"):
        assert_image_close(np.zeros((1, 2, 4), dtype=np.uint8), golden_file, 0)


def test_missing_png_does_not_create_files(tmp_path, sample):
    with pytest.raises(AssertionError, match="missing.*--update-golden"):
        assert_image_close(sample, tmp_path / "missing.png", 0)
    assert list(tmp_path.iterdir()) == []


def test_missing_sidecar(golden_file, sample):
    golden_file.with_suffix(".json").unlink()
    with pytest.raises(AssertionError, match="metadata.*missing"):
        assert_image_close(sample, golden_file, 0)


def test_malformed_sidecar(golden_file, sample):
    golden_file.with_suffix(".json").write_text("not JSON", encoding="utf-8")
    with pytest.raises(AssertionError, match="metadata"):
        assert_image_close(sample, golden_file, 0)


def test_hash_mismatch(golden_file, sample):
    metadata_path = golden_file.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["png_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(AssertionError, match="SHA256"):
        assert_image_close(sample, golden_file, 0)


def test_corrupt_png(golden_file, sample):
    golden_file.write_bytes(b"invalid PNG")
    metadata_path = golden_file.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["png_sha256"] = hashlib.sha256(golden_file.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(AssertionError, match="corrupt.*PNG"):
        assert_image_close(sample, golden_file, 0)


def test_metadata_dimensions_must_match_png(golden_file, sample):
    metadata_path = golden_file.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["size"] = [99, 99]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(AssertionError, match="metadata.*size"):
        assert_image_close(sample, golden_file, 0)


def test_rgba_reference_preserves_alpha(tmp_path):
    path = tmp_path / "alpha.png"
    actual = np.array([[[1, 2, 3, 0], [4, 5, 6, 128]]], dtype=np.uint8)
    Image.fromarray(actual).save(path)
    metadata = {
        "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": "RGBA",
        "size": [2, 1],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    assert_image_close(actual, path, 0)
    actual[0, 1, 3] += 2
    with pytest.raises(AssertionError, match="channel=3"):
        assert_image_close(actual, path, 1 / 255)


@pytest.mark.parametrize("recipe_id", RECIPE_IDS)
def test_reference_recipes_are_deterministic(recipe_id):
    first = reference_image(recipe_id)
    second = reference_image(recipe_id)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)
    assert set(recipe_inputs(recipe_id)) == {
        "foreground",
        "background",
        "mask",
        "offset",
    }


def test_reference_generator_is_independent():
    source = Path(__file__).parent / "assets" / "generate_ws00.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported = [
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    imported += [
        name.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for name in node.names
    ]
    assert not any(name and name.startswith("moviepy") for name in imported)


def test_reference_known_pixels_and_offset_clipping():
    half = reference_image("half_alpha_over")
    assert half[0, 0, 0] == 125
    assert half[0, 0, 2] == 40
    inputs = recipe_inputs("offset_over")
    actual = reference_image("offset_over")
    np.testing.assert_array_equal(actual[0], inputs["background"][0])
    np.testing.assert_array_equal(actual[:, 2:], inputs["background"][:, 2:])
    expected = (
        inputs["foreground"][0, 1].astype(float)
        + inputs["background"][1, 0].astype(float)
    ) / 2
    np.testing.assert_array_equal(actual[1, 0], np.rint(expected).astype(np.uint8))


@pytest.mark.parametrize("recipe_id", RECIPE_IDS)
def test_regeneration_records_source_and_pixel_provenance(tmp_path, recipe_id):
    path = tmp_path / (recipe_id + ".png")
    regenerate_golden(path)
    assert_image_close(reference_image(recipe_id), path, 0)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["recipe"]["id"] == recipe_id
    source = Path(__file__).parents[2] / metadata["generator"]["path"]
    assert (
        metadata["generator"]["sha256"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert metadata["reference_formula"]
    assert metadata["versions"]["numpy"]
    assert metadata["versions"]["pillow"]


def test_regeneration_replaces_existing_pixels_with_reference(tmp_path):
    path = tmp_path / "half_alpha_over.png"
    path.write_bytes(b"broken")
    regenerate_golden(path)
    assert_image_close(reference_image(path.stem), path, 0)


def test_unknown_recipe_cannot_write(tmp_path):
    with pytest.raises(ValueError, match="unknown.*recipe"):
        regenerate_golden(tmp_path / "unknown.png")
    assert list(tmp_path.iterdir()) == []


def test_writer_error_propagates_and_does_not_publish(tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(OSError, match="synthetic write failure"):
        regenerate_golden(tmp_path / "half_alpha_over.png")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", [False, True])
def test_second_replace_failure_restores_original_pair(tmp_path, monkeypatch, existing):
    path = tmp_path / "half_alpha_over.png"
    if existing:
        regenerate_golden(path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    original_replace = Path.replace
    publication_error = OSError("metadata replacement failed")
    calls = 0

    def fail_second_replace(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise publication_error
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_replace)
    with pytest.raises(OSError) as caught:
        _publish_reference(path, b"new PNG bytes", {"new": "metadata"})
    assert caught.value is publication_error
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_rollback_failure_reports_error_and_retains_recovery_backup(
    tmp_path, monkeypatch
):
    path = tmp_path / "half_alpha_over.png"
    regenerate_golden(path)
    original_png = path.read_bytes()
    original_metadata = path.with_suffix(".json").read_bytes()
    original_replace = Path.replace
    publication_error = OSError("metadata replacement failed")
    calls = 0

    def fail_publication_and_restore(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise publication_error
        if calls == 3:
            raise OSError("original PNG restoration failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publication_and_restore)
    with pytest.raises(OSError, match="rollback failed.*recovery backup") as caught:
        _publish_reference(path, b"new PNG bytes", {"new": "metadata"})
    assert caught.value.__cause__ is publication_error
    assert "original PNG restoration failed" in str(caught.value)
    assert path.with_suffix(".json").read_bytes() == original_metadata
    backups = [
        p for p in tmp_path.iterdir() if p not in (path, path.with_suffix(".json"))
    ]
    assert len(backups) == 1
    assert backups[0].read_bytes() == original_png


@pytest.mark.parametrize("recipe_id", RECIPE_IDS)
def test_registered_reference_fixture(recipe_id, assert_image_close, ae_golden_dir):
    assert_image_close(
        reference_image(recipe_id), ae_golden_dir / (recipe_id + ".png"), 0
    )


def test_tracked_references_and_asset_license():
    root = Path(__file__).parent
    for recipe_id in RECIPE_IDS:
        assert_image_close(
            reference_image(recipe_id), root / "golden" / (recipe_id + ".png"), 0
        )
        metadata = json.loads(
            (root / "golden" / (recipe_id + ".json")).read_text(encoding="utf-8")
        )
        source = root / "assets" / "generate_ws00.py"
        assert (
            metadata["generator"]["sha256"]
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )
        assert metadata["recipe"]["id"] == recipe_id
    assert "MIT License" in (root / "assets" / "LICENSE-ASSETS.txt").read_text(
        encoding="utf-8"
    )
