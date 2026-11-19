"""WS-05 mask path: validation, SVG error paths and flattening edge cases."""

import numpy as np

import pytest

from moviepy.ae.masks import path as shapes
from moviepy.ae.properties import PathValue


@pytest.mark.parametrize(
    "call",
    [
        lambda: shapes.rect(5, (2, 2)),
        lambda: shapes.rect((0, 0), (0, 2)),
        lambda: shapes.rounded_rect((0, 0), (4, 4), -1),
        lambda: shapes.star((0, 0), 1, 2, 1),
        lambda: shapes.star((0, 0), 5, 0, 1),
    ],
)
def test_invalid_shape_arguments_raise(call):
    with pytest.raises(ValueError):
        call()


def test_zero_roundness_is_a_plain_rectangle():
    rounded = shapes.rounded_rect((5, 5), (4, 2), 0)
    assert rounded.vertices == shapes.rect((5, 5), (4, 2)).vertices


@pytest.mark.parametrize(
    "text, error",
    [
        (3, TypeError),
        ("M 0 0 " + "L 1 1 " * 20000, ValueError),
        ("", ValueError),
        ("M 0 0 A 1 1 0 0 0 2 2", ValueError),
        ("1 2 3 4", ValueError),
        ("M 0 0 L 1", ValueError),
        ("M 0 0 L 4 0 L 4 4 Z L 0 0", ValueError),
        ("M 0 0 Z", ValueError),
    ],
)
def test_svg_path_errors(text, error):
    with pytest.raises(error):
        shapes.from_svg_path(text)


def test_svg_close_merges_a_coincident_end_vertex():
    path = shapes.from_svg_path("M 0 0 L 4 0 L 4 4 L 0 0 Z")
    assert len(path.vertices) == 3


def test_flatten_rejects_non_paths_and_handles_open_paths():
    with pytest.raises(TypeError):
        shapes.flatten([(0, 0), (1, 1)])
    open_path = PathValue(
        ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0)),
        ((0.0, 0.0),) * 3,
        ((0.0, 0.0),) * 3,
        closed=False,
    )
    points = shapes.flatten(open_path)
    assert np.allclose(points, [(0, 0), (4, 0), (4, 4)])


def test_flatten_drops_the_duplicated_closing_point():
    points = shapes.flatten(shapes.rect((0, 0), (4, 4)))
    assert len(points) == 4
