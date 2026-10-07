"""SVG source contracts, security boundary and optional real resvg regressions."""

import io
import os
import types
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

import pytest

from moviepy.ae import (
    Buffer,
    Composition,
    Expression,
    Keyframe,
    Mask,
    Property,
    RenderContext,
    SVGLayer,
    Transform,
)
from moviepy.ae.color import srgb_to_linear
from moviepy.ae.svg import backend
from moviepy.ae.svg.document import MAX_SVG_BYTES, evaluate_svg


BOX = (
    '<svg viewBox="0 0 16 16"><rect id="box" width="16" height="16" fill="red"/></svg>'
)
TEXT = '<svg viewBox="0 0 64 32"><text id="label" x="2" y="24" font-size="24">A</text></svg>'


@pytest.fixture
def contract_raster(monkeypatch):
    """A counted stub isolates AE cache semantics from the optional native engine."""
    calls = []

    def render(svg, size, snapshots, family):
        calls.append(svg)
        node = next(
            item for item in ET.fromstring(svg).iter() if item.get("id") == "box"
        )
        opacity = float(node.get("opacity", "1"))
        alpha = np.full((size[1], size[0]), opacity, dtype=np.float32)
        rgb = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        rgb[..., 0] = 255
        return Buffer.from_uint8_rgb(rgb, alpha)

    monkeypatch.setattr(backend, "rasterize", render)
    return calls


@pytest.fixture
def native():
    module = pytest.importorskip(
        "resvg_py", reason="install the optional moviepy[svg] extra"
    )
    assert module.__version__ == backend.RESVG_VERSION
    return module


def test_public_layer_factory_and_snapshot_file(tmp_path):
    from moviepy.ae.layers import SVGLayer as Exported

    path = tmp_path / "source.svg"
    path.write_text(BOX, encoding="utf-8")
    layer = SVGLayer.from_file(path, size=(16, 16))
    assert isinstance(layer, Exported)
    assert layer.svg == BOX
    path.write_text("broken", encoding="utf-8")
    assert layer.svg == BOX
    comp = Composition(size=(16, 16), duration=1)
    added = comp.add_svg(BOX)
    assert added is comp.layer(1) and added.source_size == (16, 16)


@pytest.mark.parametrize(
    "svg",
    [
        "",
        "<svg>",
        "<html/>",
        "<svg><script>alert(1)</script></svg>",
        '<svg onload="anything()"/>',
        '<svg><animate attributeName="x"/></svg>',
        "<svg><foreignObject/></svg>",
        "<svg><style>rect { fill: red }</style></svg>",
        '<svg><image href="file:///private.png"/></svg>',
        '<svg><image href="https://example.test/image.png"/></svg>',
        '<svg><image href="data:image/svg+xml;base64,AAAA"/></svg>',
        '<svg><use href="#box"/></svg>',
        '<svg xmlns:x="http://www.w3.org/1999/xlink" x:href="secret"/>',
        '<svg xml:base="file:///private/"/>',
        '<svg><rect fill="url(file:///private.png)"/></svg>',
        '<svg><rect style="fill: url(https://example.test/a)"/></svg>',
        r'<svg><rect style="fill: u\72l(secret)"/></svg>',
        '<svg><rect style="fill: u/**/rl(secret)"/></svg>',
        '<svg><rect style="@import: secret"/></svg>',
        '<svg><rect style="background: red"/></svg>',
        '<svg><rect mask="https://example.test/mask"/></svg>',
        '<svg><rect fill="url(#absent)"/></svg>',
        '<svg><g id="same"/><g id="same"/></svg>',
        '<svg><g id="two words"/></svg>',
        '<?xml-stylesheet href="secret"?><svg/>',
        '<!DOCTYPE svg SYSTEM "file:///secret"><svg/>',
        '<!DOCTYPE svg [<!ENTITY content "secret">]><svg>&content;</svg>',
        '<svg xmlns="urn:foreign"/>',
    ],
)
def test_unsafe_or_unsupported_svg_fails_before_backend(svg, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid document reached the native backend")

    monkeypatch.setattr(backend, "rasterize", unexpected)
    with pytest.raises(ValueError):
        SVGLayer(svg, size=(16, 16))


@pytest.mark.parametrize(
    "svg",
    [
        "<svg>" + "<g>" * 64 + "</g>" * 64 + "</svg>",
        "<svg>" + "<g/>" * 10000 + "</svg>",
        "<svg><desc>" + "x" * MAX_SVG_BYTES + "</desc></svg>",
    ],
    ids=["depth", "nodes", "bytes"],
)
def test_svg_input_budgets(svg):
    with pytest.raises(ValueError, match="budget"):
        SVGLayer(svg, size=(1, 1))


def test_atomic_source_replacement_and_binding_validation():
    layer = SVGLayer(BOX, size=(16, 16))
    prop = layer.bind("box", "opacity", 0.5)
    assert layer.bindings["box", "opacity"] is prop
    with pytest.raises(TypeError):
        layer.bindings["box", "opacity"] = Property(1)
    with pytest.raises(ValueError, match="not found"):
        layer.svg = "<svg/>"
    assert layer.svg == BOX
    for attribute in ("id", "href", "style", "onload"):
        with pytest.raises(ValueError):
            layer.bind("box", attribute, "anything")
    layer.unbind("box", "opacity")
    layer.svg = "<svg/>"
    assert not layer.bindings


def test_evaluated_text_is_xml_escaped():
    layer = SVGLayer(TEXT, size=(64, 32))
    text = '<script href="file:///secret"> & 中'
    layer.bind("label", "text", text)
    serialized, root = evaluate_svg(
        layer._template, layer.bindings, layer.size, 0, None, {}
    )
    assert "&lt;script" in serialized
    assert [node.tag for node in root.iter()] == ["svg", "text"]
    assert root[0].text == text


def test_evaluated_resource_cannot_bypass_validation(contract_raster):
    layer = SVGLayer(BOX, size=(16, 16))
    layer.bind(
        "box", "fill", Property(lambda t: "url(file:///secret)", value_type="enum")
    )
    with pytest.raises(ValueError, match="local fragment"):
        layer.source_buffer(0)
    assert not contract_raster


def test_same_time_key_edit_updates_rgb_and_mask(contract_raster):
    comp = Composition(size=(16, 16), duration=1, transparent=True)
    layer = comp.add_svg(BOX)
    opacity = layer.bind("box", "opacity", Property(keyframes=[(0, 1), (1, 1)]))
    first = comp.get_frame(0)
    opacity.set_keyframes([(0, 0), (1, 0)])
    assert not comp.mask.get_frame(0).any()
    assert not comp.get_frame(0).any()
    opacity.set_keyframes([(0, 1), (1, 1)])
    np.testing.assert_array_equal(comp.get_frame(0), first)
    assert comp.mask.get_frame(0).min() == 1
    assert len(contract_raster) == 2


def test_source_local_keys_and_context_affect_cached_inputs(contract_raster):
    layer = SVGLayer(BOX, size=(16, 16), start_time=2, stretch=200)
    layer.bind("box", "opacity", Property(keyframes=[(0, 0), (1, 1)]))
    np.testing.assert_allclose(layer.prepared_source(3).rgba[..., 3], 0.5)
    prop = Property(0, expression=Expression("index / 10"))
    layer.bind("box", "opacity", prop)
    layer.index = 2
    np.testing.assert_allclose(layer.source_buffer(0).rgba[..., 3], 0.2)
    layer.index = 4
    np.testing.assert_allclose(layer.source_buffer(0).rgba[..., 3], 0.4)
    assert len(contract_raster) == 3


@pytest.mark.parametrize(
    "limits,expected",
    [
        ({"cache_entries": 2}, 2),
        ({"cache_entries": 8, "cache_bytes": 16 * 16 * 16}, 1),
        ({"cache_entries": 0}, 0),
        ({"cache_bytes": 1}, 0),
    ],
)
def test_cache_limits_and_eviction(limits, expected, contract_raster):
    layer = SVGLayer(BOX, size=(16, 16), **limits)
    layer.bind("box", "opacity", Property(lambda t: t))
    for time in (0.1, 0.2, 0.3, 0.1):
        layer.source_buffer(time)
    info = layer.cache_info
    assert info["entries"] == expected
    assert info["bytes"] <= info["max_bytes"]
    assert len(contract_raster) == 4
    layer.clear_cache()
    assert layer.cache_info["bytes"] == 0


def test_cache_hits_promote_lru_and_size_changes_clear(contract_raster):
    layer = SVGLayer(BOX, size=(16, 16), cache_entries=2)
    layer.bind("box", "opacity", Property(lambda t: t))
    for time in (0.1, 0.2, 0.1, 0.3, 0.1):
        layer.source_buffer(time)
    assert len(contract_raster) == 3
    layer.size = (32, 16)
    assert layer.cache_info["entries"] == 0
    assert layer.source_buffer(0.1).size == (32, 16)
    layer.svg = BOX.replace('fill="red"', 'fill="blue"')
    layer.source_buffer(0.1)
    assert len(contract_raster) == 5


@pytest.mark.parametrize("text", ["Latin", "中文", "日本語", "한국어"])
def test_text_requires_explicit_fonts(text):
    layer = SVGLayer(TEXT.replace(">A<", f">{text}<"), size=(64, 32))
    with pytest.raises(ValueError, match="explicit font_files"):
        layer.source_buffer(0)


def test_missing_backend_is_actionable_and_core_still_imports(monkeypatch):
    original = backend.importlib.import_module

    def absent(name):
        if name == "resvg_py":
            raise ImportError("not installed")
        return original(name)

    monkeypatch.setattr(backend.importlib, "import_module", absent)
    with pytest.raises(RuntimeError, match="resvg-py==0.2.6"):
        SVGLayer(BOX, size=(16, 16)).source_buffer(0)
    assert Composition(size=(4, 4), duration=1).size == (4, 4)


def test_backend_contract_is_straight_png_and_no_implicit_io(monkeypatch):
    output = io.BytesIO()
    Image.new("RGBA", (2, 1), (255, 128, 64, 128)).save(output, format="PNG")

    def encode(**kwargs):
        assert kwargs["skip_system_fonts"] is True
        assert kwargs["font_files"] == kwargs["font_dirs"] == []
        assert kwargs["resources_dir"] is None
        assert "svg_path" not in kwargs
        return output.getvalue()

    module = types.SimpleNamespace(
        __version__=backend.RESVG_VERSION, svg_to_bytes=encode
    )
    monkeypatch.setattr(backend.importlib, "import_module", lambda name: module)
    result = SVGLayer(BOX, size=(2, 1)).source_buffer(0)
    expected = np.array([1, 128 / 255, 64 / 255, 1]) * (128 / 255)
    np.testing.assert_allclose(result.rgba[0, 0], expected, atol=1e-7)
    assert result.color_space == "srgb"


def test_native_edges_premultiply_once_and_composite(native):
    svg = '<svg><circle cx="8" cy="8" r="6.2" fill="white" opacity="0.5"/></svg>'
    layer = SVGLayer(svg, size=(16, 16))
    rgba = layer.source_buffer(0).rgba
    alpha = rgba[..., 3]
    edge = (alpha > 0) & (alpha < 0.49)
    assert edge.any()
    np.testing.assert_allclose(rgba[..., :3], np.repeat(alpha[..., None], 3, axis=2))
    for bg in ((0, 0, 0), (255, 255, 255), (30, 80, 180)):
        comp = Composition(size=(16, 16), duration=1, bg_color=bg)
        comp.add_layer(layer)
        expected = np.rint(
            255 * alpha[..., None] + np.array(bg) * (1 - alpha[..., None])
        )
        np.testing.assert_allclose(comp.get_frame(0), expected, atol=1)


def test_native_linear_boundary_preserves_sdr_and_existing_hdr(native):
    from moviepy.ae.layers.base import Layer

    class HDRLayer(Layer):
        source_size = (16, 16)

        def source_buffer(self, t, context=None):
            return Buffer(
                np.tile([4, 2, 1, 1], (16, 16, 1)).astype(np.float32),
                color_space="linear",
            )

    comp = Composition(
        size=(16, 16), duration=1, context=RenderContext(working_space="linear")
    )
    comp.add_layer(HDRLayer())
    layer = comp.add_svg(BOX.replace('fill="red"', 'fill="#808080" opacity="0.5"'))
    source = layer.source_buffer(0).rgba[8, 8]
    alpha = source[3]
    expected = srgb_to_linear(np.full(3, 128 / 255)) * alpha + np.array([4, 2, 1]) * (
        1 - alpha
    )
    result = comp.render_buffer(0, clip=False).rgba[8, 8]
    np.testing.assert_allclose(result[:3], expected, atol=2e-6)
    assert result[0] > 1 and result[3] == 1


def test_native_numeric_and_color_keys(native):
    comp = Composition(size=(16, 16), duration=2, transparent=True)
    layer = comp.add_svg(BOX)
    layer.bind("box", "width", Property(keyframes=[(0, 4), (1, 12)]))
    color = layer.bind(
        "box",
        "fill",
        Property(
            (1, 0, 0), value_type="color", keyframes=[(0, (1, 0, 0)), (1, (0, 0, 1))]
        ),
    )
    frame = comp.get_frame(0.5)
    assert tuple(frame[4, 4]) == (128, 0, 128)
    assert comp.mask.get_frame(0.5)[4, 10] == 0
    color.set_keyframes([(0, (0, 1, 0)), (1, (0, 1, 0))])
    assert tuple(comp.get_frame(0.5)[4, 4]) == (0, 255, 0)
    layer.bind("box", "opacity", 0)
    assert not comp.mask.get_frame(0.5).any()
    assert not comp.get_frame(0.5).any()


def test_native_layer_mask_effect_transform_and_parent(native):
    from moviepy.ae.effects.color.invert import Invert

    comp = Composition(size=(32, 24), duration=1, transparent=True)
    parent = comp.add_null("controller")
    parent.transform = Transform(position=(3, 2), anchor_point=(0, 0))
    layer = comp.add_svg(BOX, size=(16, 16), parent=parent)
    layer.transform = Transform(position=(2, 1), anchor_point=(0, 0), opacity=50)
    layer.masks = [Mask.rect(center=(3.5, 7.5), size=(8, 16))]
    layer.effects = [Invert()]
    frame, alpha = comp.get_frame(0), comp.mask.get_frame(0)
    assert tuple(frame[8, 8]) == (0, 255, 255)
    assert alpha[8, 8] == 0.5
    assert alpha[8, 18] == alpha[0, 0] == 0


def test_native_local_gradient_and_clip_path(native):
    svg = """<svg viewBox="0 0 16 16"><defs>
      <linearGradient id="paint"><stop offset="0" stop-color="red"/>
      <stop offset="1" stop-color="blue"/></linearGradient>
      <clipPath id="window"><rect width="8" height="16"/></clipPath>
      </defs><rect width="16" height="16" style="fill:url(#paint);clip-path:url(#window)"/></svg>"""
    rgba = SVGLayer(svg, size=(16, 16)).source_buffer(0).rgba
    assert rgba[8, 2, 0] > rgba[8, 6, 0]
    assert rgba[8, 2, 2] < rgba[8, 6, 2]
    assert rgba[8, 2, 3] == 1
    assert not rgba[:, 9:].any()


@pytest.mark.parametrize(
    "path",
    [
        "https://example.test/font.ttf",
        "file:///private/font.ttf",
        r"\\server\fonts\font.ttf",
        "//server/fonts/font.ttf",
    ],
)
def test_external_font_path_rejected_without_open(path, monkeypatch):
    from pathlib import Path

    def unexpected(*args, **kwargs):
        pytest.fail("external font path reached filesystem access")

    monkeypatch.setattr(Path, "resolve", unexpected)
    with pytest.raises(ValueError, match="explicit local"):
        SVGLayer(BOX, size=(16, 16), font_files=[path])


def test_missing_font_file_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        SVGLayer(TEXT, size=(64, 32), font_files=[tmp_path / "absent.ttf"])


@pytest.mark.parametrize(
    "changes",
    [
        {"cache_entries": -1},
        {"cache_bytes": True},
        {"cache_bytes": 0.5},
        {"size": (0, 10)},
    ],
)
def test_invalid_layer_budgets(changes):
    options = {"size": (16, 16), **changes}
    with pytest.raises((ValueError, TypeError)):
        SVGLayer(BOX, **options)


def test_svg_color_rejects_hdr_without_changing_property(contract_raster):
    prop = Property((3, 0, 0), value_type="color")
    layer = SVGLayer(BOX, size=(16, 16))
    layer.bind("box", "fill", prop)
    with pytest.raises(ValueError, match="SDR"):
        layer.source_buffer(0)
    assert prop.value_at(0) == (3, 0, 0)


def test_context_fps_rechecks_expression_at_same_time(contract_raster):
    layer = SVGLayer(BOX, size=(16, 16))
    layer.bind(
        "box", "opacity", Property(0, expression=Expression("timeToFrames(time) / 100"))
    )
    a = layer.source_buffer(0.5, RenderContext(fps=24))
    b = layer.source_buffer(0.5, RenderContext(fps=60))
    np.testing.assert_allclose(a.rgba[..., 3], 0.12)
    np.testing.assert_allclose(b.rgba[..., 3], 0.3)
    assert len(contract_raster) == 2


def _make_font(path, glyph_width=600):
    """Generate a tiny original Latin/CJK test font without redistributed assets."""
    pytest.importorskip("fontTools")
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen

    builder = FontBuilder(1000, isTTF=True)
    names = [".notdef", "space", "A", "B", "zhong"]
    builder.setupGlyphOrder(names)
    builder.setupCharacterMap({32: "space", 65: "A", 66: "B", 0x4E2D: "zhong"})
    glyphs = {}
    for name in names:
        pen = TTGlyphPen(None)
        if name != "space":
            pen.moveTo((0, 0))
            pen.lineTo((glyph_width, 0))
            pen.lineTo((glyph_width, 700))
            pen.lineTo((0, 700))
            pen.closePath()
        glyphs[name] = pen.glyph()
    builder.setupGlyf(glyphs)
    builder.setupHorizontalMetrics({name: (1000, 0) for name in names})
    builder.setupHorizontalHeader(ascent=800, descent=-200)
    builder.setupNameTable(
        {
            "familyName": "MoviePy SVG Test",
            "styleName": "Regular",
            "uniqueFontIdentifier": "MoviePySVGTest-Regular",
            "fullName": "MoviePy SVG Test Regular",
            "psName": "MoviePySVGTest-Regular",
        }
    )
    builder.setupOS2(
        sTypoAscender=800, sTypoDescender=-200, usWinAscent=800, usWinDescent=200
    )
    builder.setupPost()
    builder.setupMaxp()
    builder.save(path)


def test_native_text_keys_cjk_missing_glyph_and_family(native, tmp_path):
    font = tmp_path / "test.ttf"
    _make_font(font)
    layer = SVGLayer(TEXT, size=(64, 32), font_files=[font])
    prop = layer.bind(
        "label",
        "text",
        Property(
            "A",
            keyframes=[
                Keyframe(0, "A", interp="hold"),
                Keyframe(1, "中中", interp="hold"),
            ],
        ),
    )
    first = layer.source_buffer(0).rgba
    second = layer.source_buffer(1).rgba
    assert second[..., 3].sum() > first[..., 3].sum() * 1.8
    prop.set_keyframes([Keyframe(0, "缺", interp="hold")])
    with pytest.raises(ValueError, match="U\\+7F3A"):
        layer.source_buffer(0)
    layer.bind("label", "text", "A")
    layer.bind("label", "font-family", "Not Installed")
    with pytest.raises(ValueError, match="family is unavailable"):
        layer.source_buffer(0)


def test_native_font_content_edit_with_preserved_stat_invalidates(native, tmp_path):
    font = tmp_path / "test.ttf"
    _make_font(font, 600)
    layer = SVGLayer(TEXT, size=(64, 32), font_files=[font])
    before = layer.source_buffer(0).rgba.copy()
    stat = font.stat()
    _make_font(font, 300)
    assert font.stat().st_size == stat.st_size
    os.utime(font, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after = layer.source_buffer(0).rgba
    assert after[..., 3].sum() < before[..., 3].sum() * 0.6
    assert layer.cache_info["entries"] == 1
