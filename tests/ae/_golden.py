"""Compare final image bytes and publish independently generated references."""

import hashlib
import io
import json
import platform
import tempfile
from numbers import Real
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

from tests.ae.assets.generate_ws00 import (
    RECIPE_IDS,
    REFERENCE_FORMULA,
    reference_image,
)


def validate_actual(actual, tol):
    """Validate the final image and normalized maximum channel tolerance."""
    if not isinstance(actual, np.ndarray) or actual.dtype != np.uint8:
        raise TypeError("actual must be a uint8 numpy array")
    if actual.ndim != 3 or actual.shape[-1] not in (3, 4):
        raise ValueError("actual must have shape (H, W, 3) or (H, W, 4)")
    if not actual.shape[0] or not actual.shape[1]:
        raise ValueError("actual image dimensions must be nonempty")
    if isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real):
        raise TypeError("tol must be a normalized real scalar")
    if not np.isfinite(tol) or not 0 <= tol <= 1:
        raise ValueError("tol must be finite and in [0, 1]")


def _read_metadata(path):
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise AssertionError("golden metadata is missing: %s" % metadata_path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise AssertionError("invalid golden metadata: %s" % metadata_path) from error
    if not isinstance(metadata, dict):
        raise AssertionError("golden metadata must be a JSON object: %s" % path)
    return metadata


def _read_reference(path):
    if not path.exists():
        raise AssertionError("golden PNG missing: %s; use --update-golden" % path)
    metadata = _read_metadata(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata.get("png_sha256"):
        raise AssertionError("golden PNG SHA256 mismatch: %s" % path)
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PNG":
                raise ValueError("not PNG")
            image.load()
            mode, size = image.mode, image.size
            reference = np.asarray(image).copy()
    except (OSError, ValueError) as error:
        raise AssertionError("corrupt golden PNG: %s" % path) from error
    if metadata.get("mode") != mode or metadata.get("size") != list(size):
        raise AssertionError("golden metadata mode/size mismatch: %s" % path)
    return reference, mode


def assert_image_close(actual, golden_path, tol):
    """Compare a final uint8 RGB/RGBA image to an integrity-checked PNG.

    Parameters
    ----------
    actual : numpy.ndarray
        Nonempty final uint8 image with three or four channels.
    golden_path : path-like
        PNG path with a JSON sidecar recording SHA256, mode and size.
    tol : float
        Maximum normalized channel error; ``1 / 255`` permits one byte level.

    Raises
    ------
    AssertionError
        Reference integrity, mode, dimensions or pixel tolerance do not match.
    TypeError, ValueError
        The image or tolerance violates the input contract.
    """
    validate_actual(actual, tol)
    path = Path(golden_path)
    reference, mode = _read_reference(path)
    actual_mode = "RGB" if actual.shape[2] == 3 else "RGBA"
    if mode != actual_mode:
        raise AssertionError("%s: mode actual=%s golden=%s" % (path, actual_mode, mode))
    if actual.shape != reference.shape:
        raise AssertionError(
            "%s: shape actual=%s golden=%s" % (path, actual.shape, reference.shape)
        )
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64)) / 255
    maximum = float(np.max(difference))
    if maximum > tol:
        y, x, channel = np.argwhere(difference > tol)[0]
        raise AssertionError(
            "%s: mode=%s shape=%s max error=%.9g tol=%.9g "
            "first mismatch y=%d x=%d channel=%d"
            % (path, mode, actual.shape, maximum, tol, y, x, channel)
        )


def _provenance(recipe_id, expected, png_bytes):
    source = Path(__file__).parent / "assets" / "generate_ws00.py"
    return {
        "png_sha256": hashlib.sha256(png_bytes).hexdigest(),
        "mode": "RGB" if expected.shape[2] == 3 else "RGBA",
        "size": [expected.shape[1], expected.shape[0]],
        "recipe": {"id": recipe_id, "parameters": {"version": 1, "seed": None}},
        "generator": {
            "path": "tests/ae/assets/generate_ws00.py",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "reference_formula": REFERENCE_FORMULA,
        "versions": {
            "numpy": np.__version__,
            "pillow": pillow_version,
            "python": platform.python_version(),
        },
        "license": "MIT; tests/ae/assets/LICENSE-ASSETS.txt",
    }


def _stage_reference_bytes(target, content, temporary):
    with tempfile.NamedTemporaryFile(
        dir=target.parent, suffix=target.suffix, delete=False
    ) as stream:
        staged = Path(stream.name)
        temporary.append(staged)
        stream.write(content)
    return staged


def _rollback_publication(published, preserve):
    failures = []
    for target, backup in reversed(published):
        try:
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                backup.replace(target)
        except OSError as error:
            if backup is not None:
                preserve.add(backup)
            failures.append("%s: %s; recovery backup=%s" % (target, error, backup))
    if failures:
        raise OSError("golden publication rollback failed: " + "; ".join(failures))


def _publish_reference(path, png_bytes, metadata):
    temporary, published, preserve = [], [], set()
    targets = (path, path.with_suffix(".json"))
    contents = (png_bytes, json.dumps(metadata, indent=2).encode("utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        backups = [
            (
                _stage_reference_bytes(target, target.read_bytes(), temporary)
                if target.exists()
                else None
            )
            for target in targets
        ]
        staged = [
            _stage_reference_bytes(target, content, temporary)
            for target, content in zip(targets, contents)
        ]
        try:
            for target, replacement, backup in zip(targets, staged, backups):
                replacement.replace(target)
                published.append((target, backup))
        except OSError as publication_error:
            try:
                _rollback_publication(published, preserve)
            except OSError as rollback_error:
                raise rollback_error from publication_error
            raise
    finally:
        for staging_path in temporary:
            if staging_path not in preserve:
                staging_path.unlink(missing_ok=True)


def regenerate_golden(golden_path):
    """Publish a PNG/metadata pair from its registered independent recipe.

    Parameters
    ----------
    golden_path : path-like
        PNG filename whose stem must be one of ``RECIPE_IDS``.

    Raises
    ------
    ValueError
        The filename does not identify a registered recipe.
    OSError
        Encoding or publishing the reference fails.
    """
    path = Path(golden_path)
    if path.suffix != ".png" or path.stem not in RECIPE_IDS:
        raise ValueError("unknown reference recipe for golden path: %s" % path)
    expected = reference_image(path.stem)
    stream = io.BytesIO()
    Image.fromarray(expected).save(stream, format="PNG")
    png_bytes = stream.getvalue()
    _publish_reference(path, png_bytes, _provenance(path.stem, expected, png_bytes))
