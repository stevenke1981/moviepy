"""WS-06 tests: effect base, stack, registry, bridge, temporal effects."""

import json

import numpy as np

import pytest

import moviepy.ae as ae
from moviepy import ColorClip, VideoClip, vfx
from moviepy.ae import Buffer, Composition, Mask, Property, RenderContext
from moviepy.ae.effects import (
    AEEffect,
    EffectStack,
    MoviePyEffect,
    Param,
    from_moviepy_effect,
    registry,
)
from moviepy.ae.layers import SolidLayer
from moviepy.ae.properties import Expression
from moviepy.ae.transform import Transform


def uniform(color, alpha=1.0, size=(4, 3), offset=(0, 0)):
    """Premultiplied buffer of straight ``color`` (0..1)."""
    width, height = size
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., :3] = np.asarray(color, dtype=np.float32) * alpha
    rgba[..., 3] = alpha
    return Buffer(rgba, offset)


def frame_counter(fps=10, duration=2.0, size=(4, 4)):
    """Clip whose red code is 10 * frame index."""

    def frame_function(t):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[..., 0] = int(round(t * fps)) * 10 % 256
        return frame

    return VideoClip(frame_function, duration=duration).with_fps(fps)


class Scale(AEEffect):
    """Test effect: multiply premultiplied RGBA by ``gain``."""

    name = "Test Scale"
    category = "Test"
    PARAMS = (Param("gain", "float", 1.0, (0.0, 4.0)),)

    def render(self, src, t, context=None, values=None):
        values = self.values_at(t, context) if values is None else values
        rgba = np.clip(src.rgba * values["gain"], 0, 1)
        return Buffer._publish(rgba.astype(np.float32), src.offset, "srgb")


class Greedy(AEEffect):
    """Test temporal effect that asks for more than it declares."""

    name = "Test Greedy"
    category = "Test"

    def temporal_window(self, t, context=None, values=None):
        return (0.1, 0.0)

    def render_temporal(self, src, t, context, values, source_at):
        return source_at(-0.5)


# --------------------------------------------------------------------------- #
# Param and AEEffect basics


def test_param_validation_and_metadata():
    with pytest.raises(ValueError):
        Param("enabled", "float", 0.0)
    with pytest.raises(ValueError):
        Param("x", "path", None)
    with pytest.raises(ValueError):
        Param("mode", "enum", "a")
    with pytest.raises(ValueError):
        Param("not valid", "float", 0.0)
    param = Param("blur_dimensions", "enum", "a", choices=("a", "b"))
    assert param.label == "Blur Dimensions"
    assert param.to_dict()["choices"] == ["a", "b"]
    assert param.to_dict()["range"] is None


def test_effect_parameters_are_properties():
    blur = ae.fx.GaussianBlur(blurriness=10)
    assert isinstance(blur.blurriness, Property)
    blur.blurriness = [(0, 0), (1, 20)]
    assert blur.values_at(0.5)["blurriness"] == pytest.approx(10.0)
    with pytest.raises(TypeError):
        ae.fx.GaussianBlur(radius=3)
    with pytest.raises(ValueError):
        ae.fx.GaussianBlur(blurriness=-1)
    with pytest.raises(ValueError):
        ae.fx.GaussianBlur(blur_dimensions="diagonal")
    with pytest.raises(ValueError):
        ae.fx.Fill(color=(300, 0, 0))
    with pytest.raises(AttributeError):
        blur.missing


def test_animated_values_are_clamped_to_range():
    fill = ae.fx.Fill(opacity=Property(0.0, keyframes=[(0, 0.0), (1, 200.0)]))
    assert fill.values_at(1.0)["opacity"] == 100.0
    tint = ae.fx.Tint(map_white_to=Property((0, 0, 0), value_type="color"))
    assert tint.values_at(0)["map_white_to"] == (0.0, 0.0, 0.0)


def test_expression_driven_parameter():
    effect = Scale(gain=Property(1.0, expression=Expression("time * 2")))
    assert effect.values_at(0.75)["gain"] == pytest.approx(1.5)
    out = effect.process(uniform((0.25, 0.25, 0.25)), 0.75)
    assert out.rgba[0, 0, 0] == pytest.approx(0.375)


def test_copy_is_independent():
    blur = ae.fx.GaussianBlur(blurriness=5)
    clone = blur.copy()
    clone.blurriness = 9
    assert blur.values_at(0)["blurriness"] == 5.0
    assert clone.values_at(0)["blurriness"] == 9.0


def test_disabled_effect_is_identity_and_switchable():
    source = uniform((0.2, 0.4, 0.6))
    effect = ae.fx.Invert(enabled=False)
    assert effect.process(source, 0.0) is source
    with pytest.raises(TypeError):
        effect.enabled = "yes"


@pytest.mark.parametrize(
    "effect",
    [
        ae.fx.GaussianBlur(),
        ae.fx.BrightnessContrast(),
        ae.fx.Tint(amount_to_tint=0),
        ae.fx.Fill(opacity=0),
        ae.fx.Echo(number_of_echoes=0),
    ],
    ids=lambda effect: effect.name,
)
def test_default_or_zero_parameters_are_identity(effect):
    source = uniform((0.2, 0.4, 0.6), alpha=0.5)
    out = effect.process(source, 0.0)
    np.testing.assert_allclose(out.rgba, source.rgba, atol=1e-6)
    assert out.bounds == source.bounds


def test_double_invert_round_trips():
    source = uniform((0.2, 0.4, 0.6), alpha=0.5)
    twice = EffectStack([ae.fx.Invert(), ae.fx.Invert()]).apply(source, 0.0)
    np.testing.assert_allclose(twice.rgba, source.rgba, atol=1e-6)


# --------------------------------------------------------------------------- #
# compositing options


def test_blend_with_original():
    source = uniform((1.0, 0.0, 0.0))
    full = ae.fx.Invert(blend_with_original=100).process(source, 0.0)
    np.testing.assert_array_equal(full.rgba, source.rgba)
    half = ae.fx.Invert(blend_with_original=50).process(source, 0.0)
    np.testing.assert_allclose(half.rgba[0, 0], [0.5, 0.5, 0.5, 1.0], atol=1e-6)
    animated = ae.fx.Invert(blend_with_original=[(0, 0), (1, 100)])
    quarter = animated.process(source, 0.25)
    np.testing.assert_allclose(quarter.rgba[0, 0, 0], 0.25, atol=1e-6)


def test_effect_mask_limits_the_effect():
    source = uniform((1.0, 0.0, 0.0), size=(4, 1))
    effect = ae.fx.Invert(mask=Mask.rect((0.5, 0), (2, 1)))
    out = effect.process(source, 0.0)
    np.testing.assert_array_equal(out.rgba[0, 2:], source.rgba[0, 2:])
    np.testing.assert_allclose(out.rgba[0, :2, :3], [[0, 1, 1]] * 2)
    with pytest.raises(TypeError):
        effect.mask = "left half"


# --------------------------------------------------------------------------- #
# stack


def test_stack_order_changes_result():
    source = uniform((0.2, 0.2, 0.2))
    fill = ae.fx.Fill(color=(255, 255, 255), opacity=50)
    a_then_b = EffectStack([ae.fx.Invert(), fill]).apply(source, 0.0)
    b_then_a = EffectStack([fill.copy(), ae.fx.Invert()]).apply(source, 0.0)
    assert a_then_b.rgba[0, 0, 0] == pytest.approx(0.9)
    assert b_then_a.rgba[0, 0, 0] == pytest.approx(0.4)


def test_disabled_effect_in_stack_equals_absent():
    source = uniform((0.3, 0.6, 0.9), alpha=0.7)
    tint = ae.fx.Tint(map_black_to=(0, 0, 128))
    with_disabled = EffectStack([tint, ae.fx.Invert(enabled=False)])
    without = EffectStack([tint.copy()])
    np.testing.assert_array_equal(
        with_disabled.apply(source, 0.0).rgba, without.apply(source, 0.0).rgba
    )


def test_stack_management():
    first, second, third = ae.fx.Invert(), ae.fx.Tint(), ae.fx.Fill()
    stack = EffectStack([first, second])
    stack.add(third, 0)
    assert list(stack) == [third, first, second]
    stack.move(third, -1)
    assert list(stack) == [first, second, third]
    assert stack.find("Tint") is second and stack.find("Fill") is third
    second.enabled = False
    assert stack.active() == [first, third]
    stack.remove(first)
    assert len(stack) == 2
    with pytest.raises(ValueError):
        stack.remove(first)
    with pytest.raises(ValueError):
        stack.add(third)
    with pytest.raises(TypeError):
        stack.add(vfx.InvertColors())
    with pytest.raises(IndexError):
        stack.move(third, 5)
    with pytest.raises(KeyError):
        stack.find("Glow")
    stack.clear()
    assert len(stack) == 0


# --------------------------------------------------------------------------- #
# built-in effects


def test_gaussian_blur_profile_matches_sigma():
    impulse = np.zeros((1, 1, 4), dtype=np.float32)
    impulse[...] = 1.0
    blur = ae.fx.GaussianBlur(blurriness=8, blur_dimensions="horizontal")
    out = blur.process(Buffer(impulse, (10, 5)), 0.0)
    assert out.size == (25, 1) and out.offset == (-2, 5)
    row = out.rgba[0, :, 3].astype(np.float64)
    x = np.arange(-12, 13)
    expected = np.exp(-(x**2) / (2 * 4.0**2))
    expected /= expected.sum()
    np.testing.assert_allclose(row, expected, atol=1e-6)


def test_gaussian_blur_dimensions_and_edges():
    source = uniform((1, 1, 1), size=(9, 9))
    vertical = ae.fx.GaussianBlur(blurriness=2, blur_dimensions="vertical")
    out = vertical.process(source, 0.0)
    assert out.size == (9, 15)
    repeat = ae.fx.GaussianBlur(blurriness=6, repeat_edge_pixels=True)
    same = repeat.process(source, 0.0)
    assert same.bounds == source.bounds
    np.testing.assert_allclose(same.rgba, 1.0, atol=1e-6)


def test_brightness_contrast_formula():
    gray = uniform((0.25, 0.5, 0.75))
    flat = ae.fx.BrightnessContrast(contrast=-100).process(gray, 0.0)
    np.testing.assert_allclose(flat.rgba[0, 0, :3], 0.5, atol=1e-6)
    steep = ae.fx.BrightnessContrast(contrast=100).process(gray, 0.0)
    np.testing.assert_allclose(steep.rgba[0, 0, :3], [0.0, 0.5, 1.0], atol=1e-6)
    bright = ae.fx.BrightnessContrast(brightness=51).process(gray, 0.0)
    np.testing.assert_allclose(bright.rgba[0, 0, :3], [0.45, 0.7, 0.95], atol=1e-6)


def test_tint_maps_luma_to_ramp():
    source = uniform((1.0, 1.0, 1.0), alpha=0.5)
    tint = ae.fx.Tint(map_black_to=(0, 0, 255), map_white_to=(255, 0, 0))
    out = tint.process(source, 0.0)
    np.testing.assert_allclose(out.rgba[0, 0], [0.5, 0.0, 0.0, 0.5], atol=1e-6)
    half = ae.fx.Tint(amount_to_tint=50).process(uniform((1, 0, 0)), 0.0)
    np.testing.assert_allclose(half.rgba[0, 0, 0], 0.5 + 0.5 * 0.2126, atol=1e-6)


def test_invert_channels_on_premultiplied_input():
    source = uniform((1.0, 0.25, 0.0), alpha=0.5)
    rgb = ae.fx.Invert().process(source, 0.0)
    np.testing.assert_allclose(rgb.rgba[0, 0], [0.0, 0.375, 0.5, 0.5], atol=1e-6)
    green = ae.fx.Invert(channel="green").process(source, 0.0)
    np.testing.assert_allclose(green.rgba[0, 0], [0.5, 0.375, 0.0, 0.5], atol=1e-6)
    alpha = ae.fx.Invert(channel="alpha").process(uniform((1, 0, 0), 0.25), 0.0)
    np.testing.assert_allclose(alpha.rgba[0, 0], [0.75, 0, 0, 0.75], atol=1e-6)


def test_fill_keeps_alpha():
    out = ae.fx.Fill(color=(0, 255, 0)).process(uniform((1, 0, 0), 0.4), 0.0)
    np.testing.assert_allclose(out.rgba[0, 0], [0, 0.4, 0, 0.4], atol=1e-6)


class FakeSource:
    """Temporal source returning a gray level that encodes ``dt``."""

    def __init__(self):
        self.requests = []

    def __call__(self, dt):
        self.requests.append(round(dt, 6))
        return uniform((0.1 * (1 + round(-dt * 10)),) * 3)


@pytest.mark.parametrize(
    "operator, expected",
    [
        ("add", 0.1 + 0.2 * 0.5 + 0.3 * 0.25),
        ("maximum", 0.1),
        ("minimum", 0.075),
        ("screen", 1 - 0.9 * 0.9 * 0.925),
        ("blend", (0.1 + 0.1 + 0.075) / 3),
    ],
)
def test_echo_operators(operator, expected):
    echo = ae.fx.Echo(
        echo_time=-0.1, number_of_echoes=2, decay=0.5, echo_operator=operator
    )
    source = FakeSource()
    values = echo.values_at(0.0)
    assert echo.temporal_window(0.0, None, values) == pytest.approx((0.2, 0.0))
    out = echo.render_temporal(uniform((0.1,) * 3), 0.0, None, values, source)
    assert source.requests == [-0.1, -0.2]
    assert out.rgba[0, 0, 0] == pytest.approx(expected, abs=1e-6)
    assert out.rgba[0, 0, 3] <= 1.0


def test_echo_composite_operators_order_layers():
    echo = ae.fx.Echo(echo_time=0.5, number_of_echoes=1, echo_operator="add")
    assert echo.temporal_window(0.0) == (0.0, 0.5)
    front = uniform((1, 0, 0), 0.5, size=(1, 1))
    back = uniform((0, 0, 1), 1.0, size=(1, 1))
    for operator, expected in [
        ("composite_in_back", [0.5, 0, 0.5, 1]),
        ("composite_in_front", [0, 0, 1, 1]),
    ]:
        echo.echo_operator = operator
        out = echo.render_temporal(front, 0, None, echo.values_at(0), lambda dt: back)
        np.testing.assert_allclose(out.rgba[0, 0], expected, atol=1e-6)


def test_temporal_effect_requires_a_source_and_respects_window():
    with pytest.raises(ValueError):
        ae.fx.Echo().process(uniform((1, 1, 1)), 0.0)
    layer = SolidLayer("s", color=(255, 255, 255), size=(2, 2), effects=[Greedy()])
    with pytest.raises(ValueError):
        layer.render(1.0)


# --------------------------------------------------------------------------- #
# layers and compositions


def make_comp(size=(8, 6), duration=2.0):
    return Composition(size=size, fps=10, duration=duration, bg_color=(0, 0, 0))


def test_layer_effects_run_before_transform_and_grow_bounds():
    comp = make_comp(size=(20, 10))
    layer = comp.add_solid("w", color=(255, 255, 255), size=(4, 4))
    layer.transform = Transform(position=(8, 3), anchor_point=(0, 0))
    layer.effects.add(ae.fx.GaussianBlur(blurriness=2))
    frame = comp.get_frame(0)
    assert frame[5, 6, 0] > 0 and frame[5, 13, 0] > 0
    assert frame[5, 9, 0] > frame[5, 7, 0]
    assert frame[5, 2].max() == 0


def test_layer_effect_keyframes_follow_layer_time():
    comp = make_comp()
    layer = comp.add_solid("red", color=(255, 0, 0), start_time=0.5)
    layer.effects = ae.fx.Fill(color=(0, 0, 255), opacity=[(0, 0), (1, 100)])
    assert comp.get_frame(0.5)[0, 0].tolist() == [255, 0, 0]
    assert comp.get_frame(1.0)[0, 0].tolist() == [128, 0, 128]
    assert len(layer.effects) == 1


def test_layer_render_and_masks_then_effects():
    layer = SolidLayer(
        "s",
        color=(255, 0, 0),
        size=(4, 1),
        masks=Mask.rect((0.5, 0), (2, 1)),
        effects=[ae.fx.Invert()],
        transform=Transform(position=(0, 0), anchor_point=(0, 0)),
    )
    buffer = layer.render(0.0)
    np.testing.assert_allclose(buffer.rgba[0, :2], [[0, 1, 1, 1]] * 2)
    np.testing.assert_array_equal(buffer.rgba[0, 2:], 0.0)


def test_echo_on_layer_reads_earlier_frames():
    comp = make_comp(size=(4, 4))
    layer = comp.add_clip(frame_counter(), "counter")
    layer.effects.add(
        ae.fx.Echo(echo_time=-0.1, number_of_echoes=2, echo_operator="add")
    )
    # frame 5 -> red codes 50 + 40 + 30
    assert comp.get_frame(0.5)[0, 0, 0] == 120


def test_temporal_provider_reruns_earlier_effects():
    comp = make_comp(size=(4, 4))
    layer = comp.add_clip(frame_counter(), "counter")
    layer.effects = [
        ae.fx.Invert(channel="red"),
        ae.fx.Echo(echo_time=-0.1, number_of_echoes=1, echo_operator="minimum"),
    ]
    # inverted frame 5 = 205, inverted frame 4 = 215 -> minimum 205
    assert comp.get_frame(0.5)[0, 0, 0] == 205


def test_adjustment_layer_inverts_only_inside_mask_bit_identically():
    def scene(with_adjustment):
        comp = make_comp(size=(16, 8))
        comp.add_clip(frame_counter(size=(16, 8)), "counter")
        blur = comp.add_solid("dot", color=(30, 200, 90), size=(6, 4))
        blur.effects.add(ae.fx.GaussianBlur(blurriness=3))
        if with_adjustment:
            adjust = comp.add_adjustment(effects=[ae.fx.Invert()])
            adjust.masks = [Mask.rect((3.5, 3.5), (8, 8))]
        return comp

    plain = scene(False).render_buffer(0.7)
    adjusted = scene(True).render_buffer(0.7)
    np.testing.assert_array_equal(adjusted.rgba[:, 8:], plain.rgba[:, 8:])
    expected = plain.rgba[:, :8].copy()
    expected[..., :3] = expected[..., 3:] - expected[..., :3]
    np.testing.assert_allclose(adjusted.rgba[:, :8], expected, atol=1e-6)


def test_adjustment_opacity_blend_mode_and_disabled_effects():
    comp = make_comp(size=(4, 2))
    comp.add_solid("gray", color=(200, 200, 200))
    adjust = comp.add_adjustment(effects=[ae.fx.Invert()])
    adjust.transform.opacity = 50
    assert comp.get_frame(0)[0, 0].tolist() == [128, 128, 128]
    adjust.transform.opacity = 100
    adjust.blend_mode = "multiply"
    assert comp.get_frame(0)[0, 0].tolist() == [43, 43, 43]
    adjust.blend_mode = "normal"
    adjust.effects[0].enabled = False
    assert comp.get_frame(0)[0, 0].tolist() == [200, 200, 200]


def test_adjustment_affects_only_layers_below():
    comp = make_comp(size=(4, 1))
    comp.add_solid("red", color=(255, 0, 0))
    comp.add_adjustment(effects=[ae.fx.Invert()])
    top = comp.add_solid("blue", color=(0, 0, 255), size=(2, 1))
    top.transform = Transform(position=(0, 0), anchor_point=(0, 0))
    frame = comp.get_frame(0)
    assert frame[0].tolist() == [[0, 0, 255]] * 2 + [[0, 255, 255]] * 2


def test_adjustment_respects_time_window_track_matte_and_empty_backdrop():
    comp = make_comp(size=(4, 1))
    adjust = comp.add_adjustment(effects=[ae.fx.Fill()])
    assert comp.get_frame(0).max() == 0
    comp.add_layer(SolidLayer("white", color=(255, 255, 255), size=(4, 1)), 2)
    assert comp.layers[0] is adjust
    adjust.in_point = 1.0
    assert comp.get_frame(0.5)[0, 0].tolist() == [255, 255, 255]
    assert comp.get_frame(1.5)[0, 0].tolist() == [255, 0, 0]
    matte = comp.add_solid("matte", color=(255, 255, 255), size=(1, 1))
    matte.transform = Transform(position=(3, 0), anchor_point=(0, 0))
    adjust.set_track_matte(matte, "alpha")
    frame = comp.get_frame(1.5)
    assert frame[0].tolist() == [[255, 255, 255]] * 3 + [[255, 0, 0]]


def test_temporal_effect_on_adjustment_layer_sees_composite_below():
    comp = make_comp(size=(4, 4))
    comp.add_clip(frame_counter(), "counter")
    comp.add_adjustment(
        effects=[ae.fx.Echo(echo_time=-0.2, number_of_echoes=1, decay=0.5)]
    )
    # frame 6 (60) + 0.5 * frame 4 (40) = 80
    assert comp.get_frame(0.6)[0, 0, 0] == 80


def test_adjustment_layer_standalone_and_flags():
    adjust = ae.AdjustmentLayer(size=(4, 1), effects=[ae.fx.Invert()])
    adjust.transform = Transform(position=(0, 0), anchor_point=(0, 0))
    assert adjust.is_adjustment and not SolidLayer().is_adjustment
    out = adjust.apply_to(uniform((1, 0, 0), size=(4, 1)), 0.0)
    np.testing.assert_allclose(out.rgba[0, 0], [0, 1, 1, 1])
    coverage = adjust.render(0.0)
    np.testing.assert_array_equal(coverage.rgba, 1.0)


def test_adjustment_inside_precomp():
    inner = Composition(size=(4, 1), fps=10, duration=1, transparent=True)
    inner.add_solid("red", color=(255, 0, 0))
    inner.add_adjustment(effects=[ae.fx.Tint()])
    outer = make_comp(size=(4, 1), duration=1)
    outer.add_comp(inner)
    assert outer.get_frame(0)[0, 0].tolist() == [54, 54, 54]


def test_effect_rendering_is_deterministic():
    comp = make_comp(size=(16, 8))
    comp.add_clip(frame_counter(size=(16, 8)), "counter")
    adjust = comp.add_adjustment(effects=[ae.fx.GaussianBlur(blurriness=4)])
    adjust.effects.add(ae.fx.Echo(echo_time=-0.1, number_of_echoes=2))
    first = comp.render_buffer(1.0, RenderContext(rng_seed=3))
    second = comp.render_buffer(1.0, RenderContext(rng_seed=3))
    np.testing.assert_array_equal(first.rgba, second.rgba)


# --------------------------------------------------------------------------- #
# registry


def test_registry_lookup():
    assert registry.get("Gaussian Blur") is ae.fx.GaussianBlur
    assert registry.get("gaussian-blur") is registry.get("GaussianBlur")
    assert registry.get("brightness & contrast") is ae.fx.BrightnessContrast
    with pytest.raises(KeyError):
        registry.get("Glow")
    with pytest.raises(AttributeError):
        ae.fx.Glow
    assert "GaussianBlur" in dir(ae.fx)
    names = [cls.name for cls in registry.list(category="Color Correction")]
    assert names == ["Brightness & Contrast", "Invert", "Tint"]
    assert registry.categories() == [
        "Blur & Sharpen",
        "Color Correction",
        "Generate",
        "Time",
    ]


@pytest.mark.parametrize("cls", registry.list(), ids=lambda cls: cls.__name__)
def test_every_registered_effect_has_complete_metadata(cls):
    meta = cls.metadata()
    assert meta["name"] and meta["category"] and meta["class"] == cls.__name__
    assert cls.__doc__ and "Notes" in cls.__doc__
    instance = cls()
    for param in cls.PARAMS:
        assert param.label and param.kind
        assert param.name in cls.__doc__
        value = instance.values_at(0.0)[param.name]
        if param.limits is not None:
            assert param.limits[0] <= value <= param.limits[1]
        if param.choices is not None:
            assert value in param.choices
    source = uniform((0.2, 0.5, 0.8), alpha=0.6)
    source_at = (lambda dt: source) if meta["temporal"] else None
    out = instance.process(source, 0.0, source_at=source_at)
    assert isinstance(out, Buffer)
    assert np.all(out.rgba[..., :3] <= out.rgba[..., 3:] + 1e-6)


def test_registry_json_export():
    document = json.loads(registry.to_json())
    assert document["schema"] == 1
    blur = next(e for e in document["effects"] if e["name"] == "Gaussian Blur")
    assert blur["params"][0] == {
        "name": "blurriness",
        "label": "Blurriness",
        "type": "float",
        "default": 0.0,
        "range": [0.0, 1000.0],
        "choices": None,
    }
    assert registry.metadata("Echo")["temporal"] is True


def test_registering_custom_effects():
    registry.register(Scale)
    try:
        assert registry.get("Test Scale") is Scale
        assert ae.fx.Scale is Scale
        assert registry.register(Scale) is Scale
    finally:
        registry.unregister("Test Scale")
    with pytest.raises(KeyError):
        registry.get("Test Scale")
    with pytest.raises(TypeError):
        registry.register(int)

    class Nameless(AEEffect):
        category = "Test"

    with pytest.raises(ValueError):
        registry.register(Nameless)

    class Clash(Scale):
        name = "Gaussian Blur"

    with pytest.raises(ValueError):
        registry.register(Clash)

    class Doubled(Scale):
        name = "Doubled"
        PARAMS = (Param("gain", "float", 1.0), Param("gain", "float", 2.0))

    with pytest.raises(ValueError):
        registry.register(Doubled)


# --------------------------------------------------------------------------- #
# MoviePy interop


def test_bridge_runs_moviepy_effects():
    source = uniform((1.0, 0.5, 0.0), alpha=0.5)
    bridged = from_moviepy_effect(vfx.InvertColors())
    native = ae.fx.Invert()
    np.testing.assert_allclose(
        bridged.process(source, 0.0).rgba, native.process(source, 0.0).rgba, atol=1e-6
    )
    assert isinstance(bridged, MoviePyEffect)
    with pytest.raises(TypeError):
        from_moviepy_effect(ae.fx.Invert())
    with pytest.raises(TypeError):
        from_moviepy_effect("InvertColors")


def test_bridge_time_dependent_effect_and_layer_use():
    comp = make_comp(size=(2, 1), duration=1.0)
    layer = comp.add_solid("white", color=(255, 255, 255))
    layer.effects.add(from_moviepy_effect(vfx.FadeIn(1.0), duration=1.0))
    assert comp.get_frame(0.0)[0, 0, 0] == 0
    assert comp.get_frame(0.5)[0, 0, 0] == pytest.approx(128, abs=1)


def test_with_effects_shortcut_wraps_clip():
    clip = ColorClip((4, 2), color=(255, 0, 0), duration=1.0).with_fps(10)
    result = clip.with_effects([ae.fx.Invert()])
    assert isinstance(result, Composition)
    assert result.size == (4, 2) and result.duration == 1.0
    assert result.get_frame(0)[0, 0].tolist() == [0, 255, 255]
    masked = clip.with_mask(ColorClip((4, 2), color=0.5, is_mask=True, duration=1))
    out = masked.with_effects([ae.fx.Fill(color=(0, 0, 255))])
    assert out.mask is not None
    assert out.mask.get_frame(0)[0, 0] == pytest.approx(0.5, abs=1 / 255)
    with pytest.raises(ValueError):
        ColorClip((2, 2), color=(0, 0, 0)).with_effects([ae.fx.Invert()])


def test_facade_exports_ws06_names():
    for name in ["AEEffect", "EffectStack", "AdjustmentLayer", "fx", "effects"]:
        assert hasattr(ae, name)
    assert ae.effects.registry is registry
