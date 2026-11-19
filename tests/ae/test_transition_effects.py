"""Tests for the AE Transition effects (linear, radial, venetian, iris, block dissolve)."""

import numpy as np

import pytest

from moviepy.ae import Buffer, Property
from moviepy.ae.effects.registry import get
from moviepy.ae.effects.transition.block_dissolve import BlockDissolve
from moviepy.ae.effects.transition.iris_wipe import IrisWipe
from moviepy.ae.effects.transition.linear_wipe import LinearWipe
from moviepy.ae.effects.transition.radial_wipe import RadialWipe
from moviepy.ae.effects.transition.venetian_blinds import VenetianBlinds


def opaque(size=(40, 40), offset=(0, 0), color=(0.8, 0.4, 0.2)):
    """Opaque premultiplied buffer (alpha 1) of a constant straight color."""
    width, height = size
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(color, dtype=np.float32)
    rgba[..., 3] = 1.0
    return Buffer(rgba, offset)


def gradient(size=(37, 23), offset=(5, -3)):
    """Semi-transparent, non-uniform premultiplied buffer for invariant checks."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    alpha = (0.2 + 0.8 * ((xx + yy) % 7) / 6.0).astype(np.float32)
    rgb = np.stack([xx / width, yy / height, np.full_like(alpha, 0.5)], axis=-1)
    rgba = np.concatenate([rgb * alpha[..., None], alpha[..., None]], axis=-1)
    return Buffer(rgba.astype(np.float32), offset)


ALL_EFFECTS = (LinearWipe, RadialWipe, VenetianBlinds, IrisWipe, BlockDissolve)


# --------------------------------------------------------------------------- #
# Registration and metadata


@pytest.mark.parametrize("cls", ALL_EFFECTS)
def test_effects_register_with_notes_and_parameter_table(cls):
    assert cls.category == "Transition"
    assert get(cls.name) is cls
    assert "Notes" in cls.__doc__
    for param in cls.PARAMS:
        assert param.name in cls.__doc__


def _default(cls, name):
    return cls().params[name].value_at(0.0)


def test_documented_defaults():
    assert _default(LinearWipe, "wipe_angle") == 90.0
    assert _default(LinearWipe, "feather") == 0.0
    assert _default(RadialWipe, "wipe") == "clockwise"
    assert _default(RadialWipe, "start_angle") == 0.0
    assert _default(RadialWipe, "wipe_center_x") == 0.5
    assert _default(VenetianBlinds, "width") == 20.0
    assert _default(IrisWipe, "iris_points") == 6.0
    assert _default(BlockDissolve, "block_width") == 10.0
    assert _default(BlockDissolve, "random_seed") == 0.0


def test_radial_wipe_rejects_unknown_wipe_mode():
    with pytest.raises(ValueError):
        RadialWipe(wipe="spiral")


# --------------------------------------------------------------------------- #
# Endpoints


@pytest.mark.parametrize("cls", ALL_EFFECTS)
def test_completion_zero_returns_source_unchanged(cls):
    src = gradient()
    assert cls(transition_completion=0.0).process(src, 0.0) is src


@pytest.mark.parametrize("cls", ALL_EFFECTS)
def test_completion_hundred_is_fully_transparent_with_same_shape(cls):
    src = gradient()
    out = cls(transition_completion=100.0).process(src, 0.0)
    assert out.rgba.shape == src.rgba.shape
    assert out.offset == src.offset
    assert out.color_space == src.color_space
    assert not np.any(out.rgba)


@pytest.mark.parametrize("cls", ALL_EFFECTS)
def test_empty_buffer_passes_through(cls):
    empty = Buffer(np.zeros((0, 5, 4), dtype=np.float32), (2, 3))
    assert cls(transition_completion=50.0).process(empty, 0.0) is empty


@pytest.mark.parametrize("cls", ALL_EFFECTS)
def test_output_keeps_rgb_below_alpha_and_linear_color_space(cls):
    src = gradient()
    for completion in (1.0, 33.3, 50.0, 87.5):
        for feather in (0.0, 4.0):
            kwargs = {"transition_completion": completion}
            if any(p.name == "feather" for p in cls.PARAMS):
                kwargs["feather"] = feather
            out = cls(**kwargs).process(src, 0.0)
            rgba = out.rgba
            assert np.all(rgba[..., :3] <= rgba[..., 3:] + 1e-6)
            assert np.all(rgba >= 0.0)
    linear = Buffer(src.rgba, src.offset, "linear")
    out = LinearWipe(transition_completion=40.0).process(linear, 0.0)
    assert out.color_space == "linear"


# --------------------------------------------------------------------------- #
# Analytic checks


def test_linear_wipe_50_percent_hides_left_half_at_angle_zero():
    out = LinearWipe(transition_completion=50.0, wipe_angle=0.0).process(
        opaque((40, 4)), 0.0
    )
    alpha = out.rgba[..., 3]
    assert np.all(alpha[:, :20] == 0.0)
    assert np.all(alpha[:, 20:] == 1.0)
    assert alpha.mean() == pytest.approx(0.5)


def test_linear_wipe_default_angle_sweeps_top_down():
    out = LinearWipe(transition_completion=25.0).process(opaque((4, 40)), 0.0)
    alpha = out.rgba[..., 3]
    assert np.all(alpha[:10, :] == 0.0)
    assert np.all(alpha[10:, :] == 1.0)


def test_linear_wipe_feather_softens_edge():
    out = LinearWipe(transition_completion=50.0, wipe_angle=0.0, feather=10.0).process(
        opaque((40, 4)), 0.0
    )
    row = out.rgba[0, :, 3]
    assert 0.0 < row[15] < 1.0 or 0.0 < row[16] < 1.0
    assert np.all(np.diff(row) >= -1e-6)


def test_radial_wipe_25_percent_hides_top_right_quadrant():
    size = 40
    out = RadialWipe(transition_completion=25.0).process(opaque((size, size)), 0.0)
    alpha = out.rgba[..., 3]
    top_right = alpha[: size // 2, size // 2 :]
    assert np.all(top_right < 1e-6)
    assert np.all(alpha[: size // 2, : size // 2] > 1.0 - 1e-6)
    assert np.all(alpha[size // 2 :, :] > 1.0 - 1e-6)


def test_radial_wipe_counterclockwise_hides_top_left_quadrant():
    size = 40
    out = RadialWipe(transition_completion=25.0, wipe="counterclockwise").process(
        opaque((size, size)), 0.0
    )
    alpha = out.rgba[..., 3]
    assert np.all(alpha[: size // 2, : size // 2] < 1e-6)
    assert np.all(alpha[: size // 2, size // 2 :] > 1.0 - 1e-6)


def test_radial_wipe_both_hides_symmetric_wedge_about_start():
    size = 40
    out = RadialWipe(transition_completion=25.0, wipe="both").process(
        opaque((size, size)), 0.0
    )
    alpha = out.rgba[..., 3]
    # 180 * 0.25 = 45 degrees on each side of 12 o'clock: the top wedge.
    assert alpha[0, size // 2] < 1e-6
    assert alpha[0, size // 2 - 1] < 1e-6
    assert alpha[size - 1, size // 2] > 1.0 - 1e-6


def test_venetian_blinds_hidden_fraction_matches_completion():
    out = VenetianBlinds(transition_completion=30.0).process(opaque((100, 100)), 0.0)
    assert 1.0 - out.rgba[..., 3].mean() == pytest.approx(0.30, abs=0.02)


def test_venetian_blinds_vertical_slats_hidden_fraction():
    out = VenetianBlinds(transition_completion=70.0, direction=90.0).process(
        opaque((100, 100)), 0.0
    )
    assert 1.0 - out.rgba[..., 3].mean() == pytest.approx(0.70, abs=0.02)


def test_venetian_blinds_slat_period_is_width():
    out = VenetianBlinds(transition_completion=50.0, width=10.0).process(
        opaque((4, 40)), 0.0
    )
    column = out.rgba[:, 0, 3]
    assert np.array_equal(column[:5], np.zeros(5, dtype=np.float32))
    assert np.array_equal(column[5:10], np.ones(5, dtype=np.float32))
    assert np.array_equal(column[10:15], np.zeros(5, dtype=np.float32))


def test_iris_wipe_keeps_center_and_hides_corners_at_half():
    out = IrisWipe(transition_completion=50.0).process(opaque((40, 40)), 0.0)
    alpha = out.rgba[..., 3]
    assert alpha[20, 20] == 1.0
    assert alpha[0, 0] == 0.0
    assert alpha[0, 39] == 0.0
    assert alpha[39, 0] == 0.0
    assert alpha[39, 39] == 0.0


def test_iris_wipe_shrinks_monotonically():
    base = opaque((40, 40))
    coverage = [
        IrisWipe(transition_completion=c).process(base, 0.0).rgba[..., 3].sum()
        for c in (10.0, 30.0, 60.0, 90.0)
    ]
    assert all(a > b for a, b in zip(coverage, coverage[1:]))


def test_iris_points_selects_polygon_sides_with_analytic_area():
    # Apothem = farthest corner distance * (1 - completion); area = n a^2 tan(pi/n).
    farthest = 32.0 * np.sqrt(2.0)
    apothem = farthest * 0.6
    for sides in (6, 32):
        out = IrisWipe(transition_completion=40.0, iris_points=sides).process(
            opaque((64, 64)), 0.0
        )
        expected = sides * apothem**2 * np.tan(np.pi / sides)
        assert out.rgba[..., 3].sum() == pytest.approx(expected, rel=0.03)


def test_block_dissolve_is_deterministic_per_seed():
    src = gradient((60, 45), (0, 0))
    a = BlockDissolve(transition_completion=40.0, random_seed=3).process(src, 0.0)
    b = BlockDissolve(transition_completion=40.0, random_seed=3).process(src, 0.0)
    c = BlockDissolve(transition_completion=40.0, random_seed=4).process(src, 0.0)
    assert np.array_equal(a.rgba, b.rgba)
    assert not np.array_equal(a.rgba, c.rgba)


def test_block_dissolve_blocks_are_uniform_and_anchored_to_world_pixels():
    whole = BlockDissolve(
        transition_completion=50.0, block_width=7.0, block_height=5.0, random_seed=1
    ).process(opaque((60, 40), (0, 0)), 0.0)
    shifted = BlockDissolve(
        transition_completion=50.0, block_width=7.0, block_height=5.0, random_seed=1
    ).process(opaque((20, 40), (30, 0)), 0.0)
    assert np.array_equal(whole.rgba[:, 30:50, 3], shifted.rgba[:, :20, 3])
    alpha = whole.rgba[..., 3]
    assert set(np.unique(alpha)) <= {0.0, 1.0}
    assert np.all(alpha[0:5, 0:7] == alpha[0, 0])


def test_block_dissolve_hidden_fraction_tracks_completion():
    out = BlockDissolve(transition_completion=60.0, random_seed=9).process(
        opaque((200, 200)), 0.0
    )
    assert 1.0 - out.rgba[..., 3].mean() == pytest.approx(0.60, abs=0.08)


def test_block_dissolve_feather_fades_block_edges():
    hard = BlockDissolve(transition_completion=30.0, random_seed=2).process(
        opaque((40, 40)), 0.0
    )
    soft = BlockDissolve(
        transition_completion=30.0, random_seed=2, feather=3.0
    ).process(opaque((40, 40)), 0.0)
    assert soft.rgba[..., 3].sum() < hard.rgba[..., 3].sum()
    assert np.all(soft.rgba[..., 3] <= hard.rgba[..., 3] + 1e-6)


# --------------------------------------------------------------------------- #
# Keyframes


def test_keyframed_completion_animates_linear_wipe():
    completion = Property(0.0, keyframes=[(0.0, 0.0), (1.0, 100.0)])
    effect = LinearWipe(transition_completion=completion, wipe_angle=0.0)
    src = opaque((40, 4))
    means = []
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = effect.process(src, t)
        means.append(float(out.rgba[..., 3].mean()))
    assert means[0] == pytest.approx(1.0)
    assert means[-1] == pytest.approx(0.0)
    assert means[2] == pytest.approx(0.5, abs=0.05)
    assert all(a > b for a, b in zip(means, means[1:]))
