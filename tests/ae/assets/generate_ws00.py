"""Generate synthetic WS-00 inputs and float64 analytical image references."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _half_alpha_frames():
    foreground = np.array(
        [
            [[240, 21, 0], [10, 101, 250], [88, 22, 77]],
            [[0, 255, 17], [255, 0, 255], [23, 47, 89]],
        ],
        dtype=np.uint8,
    )
    background = np.array(
        [
            [[10, 200, 80], [120, 0, 12], [200, 99, 4]],
            [[99, 17, 240], [5, 123, 77], [201, 111, 45]],
        ],
        dtype=np.uint8,
    )
    return foreground, background


RECIPE_IDS = ("uint8_rgb_roundtrip", "half_alpha_over", "offset_over")
REFERENCE_FORMULA = (
    "float64: Cpremult=Cstraight*alpha; "
    "Cout=Cforeground+Cbackground*(1-alpha_foreground); "
    "aout=alpha_foreground+alpha_background*(1-alpha_foreground); "
    "final uint8 RGB=np.rint(np.clip(Cout/aout,0,1)*255)"
)


def recipe_inputs(recipe_id):
    """Return fresh synthetic arrays for a registered analytical recipe.

    Parameters
    ----------
    recipe_id : str
        One of ``RECIPE_IDS``.

    Returns
    -------
    dict
        Foreground/background uint8 RGB, float64 mask, and integer offset.
        The round-trip recipe has no background or mask.
    """
    if recipe_id == "uint8_rgb_roundtrip":
        ramp = np.arange(256, dtype=np.uint16)
        foreground = np.stack([ramp, 255 - ramp, (ramp * 37) % 256], axis=-1)
        return {
            "foreground": foreground[None].astype(np.uint8),
            "background": None,
            "mask": None,
            "offset": (0, 0),
        }
    if recipe_id == "half_alpha_over":
        foreground, background = _half_alpha_frames()
        offset = (0, 0)
    elif recipe_id == "offset_over":
        grid = np.arange(36, dtype=np.uint16).reshape(3, 4, 3)
        background = ((grid * 7 + 13) % 256).astype(np.uint8)
        foreground = np.array(
            [
                [[255, 0, 9], [40, 240, 100], [80, 20, 230]],
                [[17, 99, 255], [210, 70, 10], [11, 180, 200]],
            ],
            dtype=np.uint8,
        )
        offset = (-1, 1)
    else:
        raise ValueError("unknown reference recipe: %r" % recipe_id)
    return {
        "foreground": foreground,
        "background": background,
        "mask": np.full(foreground.shape[:2], 0.5, dtype=np.float64),
        "offset": offset,
    }


def reference_image(recipe_id):
    """Render a registered reference without calling product rendering code.

    Parameters
    ----------
    recipe_id : str
        Identifier of an independent recipe.

    Returns
    -------
    numpy.ndarray
        Final uint8 RGB image; over recipes crop to opaque background bounds.
    """
    inputs = recipe_inputs(recipe_id)
    foreground = inputs["foreground"].astype(np.float64) / 255
    if inputs["background"] is None:
        return np.rint(foreground * 255).astype(np.uint8)
    background = inputs["background"].astype(np.float64) / 255
    x, y = inputs["offset"]
    height, width = foreground.shape[:2]
    left, top = max(0, x), max(0, y)
    right = min(background.shape[1], x + width)
    bottom = min(background.shape[0], y + height)
    source = foreground[top - y : bottom - y, left - x : right - x]
    alpha = inputs["mask"][top - y : bottom - y, left - x : right - x, None]
    destination = background[top:bottom, left:right]
    background[top:bottom, left:right] = source * alpha + destination * (1 - alpha)
    return np.rint(np.clip(background, 0, 1) * 255).astype(np.uint8)


def generate_assets(directory):
    """Write independently generated synthetic source PNGs.

    Parameters
    ----------
    directory : path-like
        Destination for synthetic RGB frames and 8-bit mask illustrations.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for recipe_id in RECIPE_IDS:
        inputs = recipe_inputs(recipe_id)
        for name in ("foreground", "background", "mask"):
            array = inputs[name]
            if array is None:
                continue
            if name == "mask":
                array = np.rint(array * 255).astype(np.uint8)
            Image.fromarray(array).save(directory / (recipe_id + "_" + name + ".png"))


def main():
    """Generate licensed synthetic input assets from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    generate_assets(parser.parse_args().output)


if __name__ == "__main__":
    main()
