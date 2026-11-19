"""The 夜燈說書 logo title loop is seamless, editable and exportable."""

import functools

import numpy as np

import pytest

from moviepy.ae.templates import logo_loop
from moviepy.ae.templates.logo_loop import (
    LogoLoopSpec,
    build_logo_loop,
    export_logo_loop,
    tagline_opacity_keys,
)
from moviepy.ae.templates.presets import get_preset


LATIN = {"title": None, "body": None}


def _preset(size=(320, 180), fps=24):
    return get_preset("nightlamp_story", size=size, fps=fps)


def _spec(**changes):
    values = {
        "title": "Night Lamp",
        "taglines": ("one", "two", "three"),
        "crossfade": 0.2,
    }
    values.update(changes)
    return LogoLoopSpec(**values)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"title": " "}, "title"),
        ({"taglines": ("ok", "")}, "taglines"),
        ({"layout": "diagonal"}, "layout"),
        ({"period": 0}, "period"),
        ({"crossfade": 2.1}, "crossfade"),
        ({"breath": 1.5}, "breath"),
        ({"embers": -1}, "embers"),
        ({"seed": True}, "seed"),
    ],
)
def test_spec_validation(changes, message):
    with pytest.raises(ValueError, match=message):
        _spec(**changes)


def test_tagline_hand_over_is_periodic_and_never_overlaps():
    count, period, crossfade = 3, 6.0, 0.8
    from moviepy.ae.properties.property import Property

    curves = [
        Property(tagline_opacity_keys(i, count, period, crossfade), value_type="float")
        for i in range(count)
    ]
    for curve in curves:
        assert curve.value_at(0.0) == pytest.approx(curve.value_at(period))
    for t in np.linspace(0.0, period, 241):
        values = [curve.value_at(float(t)) for curve in curves]
        assert sum(v > 1e-6 for v in values) <= 1, (t, values)
    # Each line is fully shown for most of its share.
    for i, curve in enumerate(curves):
        middle = (i + 0.5) * period / count
        assert curve.value_at(middle) == pytest.approx(100.0)


def test_build_layers_report_and_seamless_frames():
    spec = _spec(period=2.0, embers=6)
    comp = build_logo_loop(spec, _preset(), fonts=LATIN)
    names = [layer.name for layer in comp.layers]
    assert names[-1] == "Background"
    for name in ("Halo", "Embers", "Glow", "Lamp", "Flame", "Title", "Shine"):
        assert name in names
    assert [n for n in names if n.startswith("Tagline")] == [
        "Tagline 3",
        "Tagline 2",
        "Tagline 1",
    ]
    report = comp.logo_report
    assert report["frames"] == 48 and report["taglines"] == ["one", "two", "three"]
    first = comp.get_frame(0.0).astype(np.int16)
    # Every layer is periodic: the frame at ``period`` is the first frame.
    assert np.array_equal(first, comp.get_frame(spec.period).astype(np.int16))
    middle = comp.get_frame(spec.period / 2).astype(np.int16)
    assert np.abs(first - middle).mean() > 0.05  # it actually moves


def test_inline_layout_puts_the_lamp_left_of_the_title():
    comp = build_logo_loop(
        _spec(layout="inline", taglines=()), _preset((640, 360)), fonts=LATIN
    )
    boxes = comp.logo_report["boxes"]
    assert boxes["Lamp"][2] <= boxes["Title"][0]
    assert not [n for n in boxes if n.startswith("Tagline")]


def test_transparent_loop_has_alpha_and_no_background():
    comp = build_logo_loop(_spec(period=1.0), _preset(), fonts=LATIN, transparent=True)
    assert "Background" not in [layer.name for layer in comp.layers]
    alpha = comp.mask.get_frame(0.5)
    assert alpha.max() > 0.9 and alpha.min() < 0.01


def test_period_must_be_whole_frames_and_block_must_fit():
    with pytest.raises(ValueError, match="whole number of frames"):
        build_logo_loop(_spec(period=1.01), _preset(), fonts=LATIN)
    with pytest.raises(ValueError):
        build_logo_loop(
            _spec(title="An extremely long channel title " * 4),
            _preset(),
            fonts=LATIN,
        )


def test_export_mp4_loop(tmp_path):
    comp = build_logo_loop(
        _spec(period=1.0, embers=3), _preset((64, 36), fps=12), fonts=LATIN
    )
    report = export_logo_loop(comp, tmp_path / "logo.mp4", encoder="libx264")
    assert report["frames"] == 12 and (tmp_path / "logo.mp4").stat().st_size > 0
    with pytest.raises(FileExistsError):
        export_logo_loop(comp, tmp_path / "logo.mp4")


def test_cli_renders_a_transparent_mov(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        logo_loop,
        "build_logo_loop",
        functools.partial(logo_loop.build_logo_loop, fonts=LATIN),
    )
    out = tmp_path / "logo.mov"
    code = logo_loop.main(
        [str(out), "--title", "Night Lamp", "--tagline", "a", "--tagline", "b"]
        + ["--size", "64x36", "--fps", "12", "--period", "1", "--transparent"]
    )
    assert code == 0 and out.is_file()
    assert '"frames": 12' in capsys.readouterr().out
