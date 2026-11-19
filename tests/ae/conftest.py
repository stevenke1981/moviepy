"""Provide AE-only image fixtures without changing existing test fixtures."""

from pathlib import Path

import pytest

from tests.ae._golden import (
    assert_image_close as compare_image,
    regenerate_golden,
    validate_actual,
)
from tests.ae.assets.generate_ws00 import RECIPE_IDS, reference_image


@pytest.fixture
def ae_golden_dir():
    """Return the directory containing registered independent reference PNGs."""
    return Path(__file__).parent / "golden"


@pytest.fixture
def assert_image_close(request, ae_golden_dir):
    """Compare images, optionally regenerating a registered independent recipe."""
    update = request.config.getoption("--update-golden", default=False)

    def compare(actual, golden_path, tol):
        validate_actual(actual, tol)
        path = Path(golden_path)
        if not path.is_absolute():
            path = ae_golden_dir / path
        if update:
            if path.parent.resolve() != ae_golden_dir.resolve():
                raise ValueError("updated PNG must be inside the golden root")
            if path.suffix != ".png" or path.stem not in RECIPE_IDS:
                raise ValueError("unknown reference recipe: %s" % path)
            expected = reference_image(path.stem)
            if expected.shape != actual.shape:
                raise AssertionError("actual shape does not match reference recipe")
            regenerate_golden(path)
        compare_image(actual, path, tol)

    return compare
