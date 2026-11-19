"""Tests for the NLH chapter tag, vertical quote and shared paper helpers."""

import time
from pathlib import Path

import numpy as np

import pytest

from moviepy.ae.templates.chapter_tag import chapter_tag
from moviepy.ae.templates.paper import (
    chinese_numeral,
    paper_texture,
    seal_stamp,
    torn_mask,
)
from moviepy.ae.templates.presets import get_preset
from moviepy.ae.templates.quote import (
    resolve_medium,
    split_columns,
    split_quote_pages,
    vertical_form,
    vertical_quote,
)


KAIU = Path("C:/Windows/Fonts/kaiu.ttf")
PRESET_FONTS = [
    Path(path)
    for name in ("nightlamp_story", "nightlamp_history")
    for path in get_preset(name).fonts.values()
]
needs_kaiu = pytest.mark.skipif(
    not KAIU.is_file() or not all(path.is_file() for path in PRESET_FONTS),
    reason="kaiu.ttf or the preset Source Han fonts are not installed",
)

QUOTE = "「天下熙熙，皆為利來；天下攘攘，皆為利往。」夫千乘之王，萬家之侯。"


@pytest.fixture(scope="module")
def preset():
    return get_preset("nightlamp_history")


# ---- medium ------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"dynasty": "秦"}, "bamboo"),
        ({"dynasty": "西漢"}, "bamboo"),
        ({"dynasty": "东汉"}, "bamboo"),
        ({"dynasty": "戰國時代"}, "bamboo"),
        ({"dynasty": "三國"}, "paper"),
        ({"dynasty": "後漢"}, "paper"),
        ({"dynasty": "唐朝"}, "paper"),
        ({"dynasty": "明"}, "paper"),
        ({"dynasty": "南宋"}, "paper"),
        ({"year": 220}, "bamboo"),
        ({"year": 221}, "paper"),
        ({"year": -200}, "bamboo"),
        ({"dynasty": "唐", "medium": "bamboo"}, "bamboo"),
        ({"year": 100, "dynasty": "唐"}, "bamboo"),
    ],
)
def test_resolve_medium(kwargs, expected):
    assert resolve_medium(**kwargs)[0] == expected


def test_resolve_medium_errors():
    with pytest.raises(ValueError):
        resolve_medium()
    with pytest.raises(ValueError):
        resolve_medium(dynasty="火星")
    with pytest.raises(ValueError):
        resolve_medium(medium="stone")


# ---- column splitting and vertical punctuation ------------------------- #


def test_split_columns_basic_and_forced_breaks():
    cols = split_columns("甲乙丙丁戊己庚辛", 3)
    assert ["".join(c) for c in cols] == ["甲乙丙", "丁戊己", "庚辛"]
    cols = split_columns("甲乙|丙丁\n戊", 10)
    assert ["".join(c) for c in cols] == ["甲乙", "丙丁", "戊"]


def test_split_columns_no_closing_punctuation_at_column_start():
    cols = ["".join(c) for c in split_columns("甲乙丙，丁戊己。庚", 3)]
    assert cols[0] == "甲乙丙，"
    assert all(c[0] not in "，。」" for c in cols)


def test_vertical_punctuation_mapping():
    assert vertical_form("「") == "﹁" and vertical_form("」") == "﹂"
    assert vertical_form("『") == "﹃" and vertical_form("』") == "﹄"
    assert vertical_form("《") == "︽" and vertical_form("》") == "︾"
    assert vertical_form("（") == "︵" and vertical_form("）") == "︶"
    assert vertical_form("史") == "史"


def test_split_quote_pages_prefers_sentence_ends():
    pages = split_quote_pages("甲乙。丙丁。戊己。庚辛。壬癸。", 3, 2)
    assert pages == ["甲乙。\n丙丁。", "戊己。\n庚辛。", "壬癸。"]
    long = "甲" * 50
    assert all(len(p.split("\n")) <= 3 for p in split_quote_pages(long, 5, 3))
    assert "".join(split_quote_pages(long, 5, 3)).replace("\n", "") == long


# ---- seal and paper ----------------------------------------------------- #


def test_chinese_numerals():
    assert [chinese_numeral(n) for n in range(1, 10)] == list("壹貳參肆伍陸柒捌玖")
    assert chinese_numeral(10) == "拾"
    assert chinese_numeral(11) == "拾壹"
    assert chinese_numeral(20) == "貳拾"
    assert chinese_numeral(35) == "參拾伍"
    assert chinese_numeral(100) == "100"


def test_paper_texture_deterministic_by_seed():
    base = (239, 228, 204)
    a = paper_texture(64, 40, base, seed=5)
    b = paper_texture(64, 40, base, seed=5)
    c = paper_texture(64, 40, base, seed=6)
    assert a.dtype == np.uint8 and a.shape == (40, 64, 3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert abs(float(a.mean(axis=(0, 1))[0]) - 239) < 8


def test_torn_mask_and_seal_deterministic():
    m1, m2 = torn_mask(80, 30, 3, seed=2), torn_mask(80, 30, 3, seed=2)
    assert np.array_equal(m1, m2) and 0 <= m1.min() and m1.max() <= 1
    assert not np.array_equal(m1, torn_mask(80, 30, 3, seed=3))
    s1 = seal_stamp("壹", 48, seed=4)
    assert np.array_equal(s1, seal_stamp("壹", 48, seed=4))
    assert not np.array_equal(s1, seal_stamp("壹", 48, seed=5))
    assert s1.shape == (48, 48, 4) and s1[..., 3].max() > 200


@needs_kaiu
def test_seal_glyphs_are_drawn_with_cjk_font():
    plain = seal_stamp("壹", 64, seed=1, font=str(KAIU))
    blank = seal_stamp("", 64, seed=1, font=str(KAIU))
    # relief seal: light glyph pixels on a red body
    assert (plain[..., :3].sum(-1) > blank[..., :3].sum(-1) + 300).any()


# ---- chapter tag -------------------------------------------------------- #


def _alpha(comp, t):
    return comp.mask.get_frame(t)


@needs_kaiu
def test_chapter_tag_transparent_outside_strip_and_metadata(preset):
    tag = chapter_tag(["第一章", "本章人物：項羽", "楚漢"], 1, 8, preset)
    comp = tag.composition
    assert comp.transparent
    assert _alpha(comp, 0.0).max() == 0.0
    a = _alpha(comp, 3.0)
    assert a.max() > 0.9 and a[0, 0] == 0 and a[-1, -1] == 0
    assert (a > 0.05).mean() < 0.9
    assert tag.meta["x"] == 26 and tag.meta["y"] == 22
    assert tag.meta["w"] == comp.size[0] and tag.meta["items"][0] == "第一章"
    assert '"period": 5.5' in tag.to_json()
    # frame RGB is straight (unmatted) with alpha carried in the mask
    assert comp.get_frame(3.0).shape == (comp.size[1], comp.size[0], 3)


@needs_kaiu
def test_chapter_strip_enters_from_left_then_stable(preset):
    comp = chapter_tag(["第一章", "楚漢"], 1, 8, preset).composition
    right_edges = []
    for t in np.arange(0.0, 1.5, 0.05):
        cols = np.where(_alpha(comp, float(t)).max(axis=0) > 0.3)[0]
        right_edges.append(int(cols.max()) if len(cols) else -1)
    edges = np.array(right_edges)
    assert edges[0] == -1
    rise = np.diff(edges[edges >= 0])
    assert (rise >= 0).all() and edges.max() > edges[1]
    assert (edges[-6:] == edges[-1]).all()  # stable once in
    # left edge reaches its resting place too: strip is not clipped any more
    left = np.where(_alpha(comp, 1.5).max(axis=0) > 0.3)[0].min()
    assert left < comp.size[0] // 4


@needs_kaiu
def test_chapter_items_rotate_after_period_and_static_is_cached(preset):
    tag = chapter_tag(["第一章", "本章人物：項羽與劉邦", "楚漢相爭"], 2, 14, preset)
    comp = tag.composition
    clip = tag.strip.clip
    first = comp.get_frame(2.0).copy()
    w_first = np.where(_alpha(comp, 2.0).max(axis=0) > 0.3)[0].max()
    second = comp.get_frame(5.5 + 2.0).copy()
    w_second = np.where(_alpha(comp, 5.5 + 2.0).max(axis=0) > 0.3)[0].max()
    assert not np.array_equal(first, second)
    assert w_second > w_first  # strip widens to the longer item
    # smooth transition: intermediate widths between the two items
    mid = np.where(_alpha(comp, 5.5 + 0.25).max(axis=0) > 0.3)[0].max()
    assert w_first < mid < w_second + 1
    builds = clip.builds
    for t in np.arange(2.0, 5.0, 1 / 24):
        comp.get_frame(float(t))
    assert clip.builds == builds  # static stretch renders no new strip state


@needs_kaiu
def test_chapter_seal_parented_and_keyframed(preset):
    tag = chapter_tag("甲|乙", 3, 6, preset)
    assert tag.seal.parent is tag.strip
    s = tag.seal.transform.scale
    assert s.value_at(0.0)[0] == pytest.approx(145.0)
    assert s.value_at(2.0)[0] == pytest.approx(100.0)
    with pytest.raises(ValueError):
        chapter_tag([], 1, 5, preset)
    with pytest.raises(ValueError):
        chapter_tag("a", 1, 5, preset, position="center")


@needs_kaiu
def test_chapter_tag_1080p_frame_timing(preset):
    tag = chapter_tag(
        ["第一章 鴻門宴", "本章人物：項羽、劉邦", "楚漢相爭"], 1, 30, preset
    )
    comp = tag.composition
    comp.get_frame(3.0)  # warm: first build of this state
    samples = []
    for t in (3.0, 3.04, 3.08, 20.0):
        start = time.perf_counter()
        comp.get_frame(t)
        samples.append(time.perf_counter() - start)
    print(
        f"\nchapter tag {comp.size} @1080p preset: "
        f"cached frame RGB {np.mean(samples) * 1000:.1f} ms"
    )
    assert np.mean(samples) < 0.5


# ---- quote -------------------------------------------------------------- #


@needs_kaiu
def test_quote_bamboo_vs_paper_layers(preset):
    bamboo = vertical_quote(QUOTE, "《史記》", preset, dynasty="西漢", seal="史")
    paper = vertical_quote(QUOTE, "《明史》", preset, dynasty="明")
    assert bamboo.quote_meta["medium"] == "bamboo"
    assert paper.quote_meta["medium"] == "paper"
    assert "slip 1" in bamboo.quote_layers and "sheet" not in bamboo.quote_layers
    assert "sheet" in paper.quote_layers and "roller left" in paper.quote_layers
    assert "seal" in bamboo.quote_layers and "seal" not in paper.quote_layers
    assert bamboo.size == preset.size


@needs_kaiu
def test_quote_columns_right_to_left_with_source_last(preset):
    comp = vertical_quote(
        QUOTE, "《史記》", preset, dynasty="明", title="太史公曰", per_column=8
    )
    cols = comp.quote_meta["columns"]
    assert cols[0] == "太史公曰" and cols[-1] == "《史記》"
    assert cols[1].startswith("「") and len(cols) >= 4
    xs = [
        comp.quote_layers[f"column {i + 1}"].transform.position.value_at(0)[0]
        for i in range(len(cols))
    ]
    assert all(a > b for a, b in zip(xs, xs[1:]))  # strictly right to left
    # source column is bottom aligned: its top sits lower than the body columns
    ys = [
        comp.quote_layers[f"column {i + 1}"].transform.position.value_at(0)[1]
        for i in range(len(cols))
    ]
    assert ys[-1] > ys[1]


@needs_kaiu
def test_quote_reveal_is_monotonic(preset):
    comp = vertical_quote(QUOTE, "《史記》", preset, dynasty="明", dim=0)
    meta = comp.quote_meta
    layers = [v for k, v in comp.quote_layers.items() if k.startswith("column")]
    times = np.linspace(0.0, meta["write_end"] + 0.5, 14)
    ink = []
    for t in times:
        ink.append(
            sum(float(col.clip.mask.get_frame(float(t)).sum()) for col in layers)
        )
    assert ink[0] == 0.0
    assert all(b >= a - 1e-6 for a, b in zip(ink, ink[1:]))
    assert ink[-1] > ink[len(ink) // 2] > 0
    # writing order: right column starts before the left one
    first, last = layers[0], layers[-1]
    t_mid = meta["write_start"] + 1.0
    assert (
        first.clip.mask.get_frame(t_mid).sum() > last.clip.mask.get_frame(t_mid).sum()
    )


@needs_kaiu
def test_quote_scroll_unrolls_right_to_left(preset):
    comp = vertical_quote(QUOTE, "《明史》", preset, dynasty="明", dim=0)
    right_edge = comp.quote_layers["roller right"].transform.position.value_at(0)[0]
    left_x = [
        comp.quote_layers["roller left"].transform.position.value_at(t)[0]
        for t in (0.0, 0.4, 0.7, 1.0)
    ]
    assert left_x[0] == pytest.approx(right_edge)
    assert left_x[0] > left_x[1] > left_x[2] > left_x[3]
    widths = []
    for t in (0.2, 0.5, 0.8, 1.2):
        cols = np.where(comp.quote_layers["sheet"].clip.mask.get_frame(t).max(0) > 0.9)[
            0
        ]
        widths.append(0 if len(cols) == 0 else cols.max() - cols.min())
    assert widths[0] < widths[1] < widths[2] <= widths[3]


@needs_kaiu
def test_quote_too_long_raises_and_pages_render(preset):
    long = "天下熙熙皆為利來。" * 20
    with pytest.raises(ValueError, match="split_quote_pages"):
        vertical_quote(long, "《史記》", preset, dynasty="明")
    pages = split_quote_pages(long, 10, 8)
    assert len(pages) > 1
    comp = vertical_quote(pages[0], "《史記》", preset, dynasty="明")
    assert comp.get_frame(comp.duration * 0.6).shape == (1080, 1920, 3)


@needs_kaiu
def test_quote_deterministic(preset):
    a = vertical_quote(QUOTE, None, preset, dynasty="唐", seed=3)
    b = vertical_quote(QUOTE, None, preset, dynasty="唐", seed=3)
    t = a.quote_meta["write_end"] + 0.5
    assert np.array_equal(a.get_frame(t), b.get_frame(t))


# ---- review round: correctness and speed ---------------------------------- #


def test_split_columns_never_starts_a_column_with_closing_punctuation():
    for text, per in [
        ("甲。。。。", 1),
        ("甲乙。。。。。丙丁", 2),
        ("甲乙丙，」』丁", 3),
    ]:
        cols = ["".join(c) for c in split_columns(text, per)]
        assert "".join(cols) == text
        assert not any(c[0] in "，。、；：！？」』）》" for c in cols), cols
    assert ["".join(c) for c in split_columns("，，，", 2)] == ["，，，"]


def test_split_quote_pages_survive_punctuation_runs():
    text = "甲。。。。。乙丙。。。丁"
    pages = split_quote_pages(text, per_column=2, max_columns=2)
    assert "".join(p.replace("\n", "") for p in pages) == text


def test_resize_straight_keeps_colour_at_soft_edges():
    from moviepy.ae.templates.paper import resize_straight

    img = np.zeros((8, 8, 4), np.uint8)  # transparent black surround
    img[2:6, 2:6] = (200, 90, 30, 255)
    out = resize_straight(img, (3, 3))
    edge = (out[..., 3] > 0) & (out[..., 3] < 255)
    assert edge.any()
    # a naive straight-RGBA area resize darkens the edge; premultiplied does not
    assert np.abs(out[..., :3][edge].astype(int) - (200, 90, 30)).max() <= 1


def test_cord_and_seal_edges_have_no_dark_fringe():
    from moviepy.ae.templates.quote import _cord_segment

    color = (150, 120, 70)
    cord = _cord_segment(60, 12, 6, 4, color, (10,))
    soft = (cord[..., 3] > 8) & (cord[..., 3] < 240)
    assert soft.any()
    # edge pixels carry the cord colour (the knot shading is at most 34 darker)
    floor = np.array(color) - (34, 28, 20) - 1  # darkest tick colour
    assert (cord[..., :3][soft].astype(int) >= floor).all()
    seal = seal_stamp("壹", 40, (176, 30, 28), seed=2)
    edge = (seal[..., 3] > 8) & (seal[..., 3] < 200)
    assert edge.any()
    assert seal[..., 0][edge].min() >= 100  # red body or paper glyph, never black


def test_rect_shadow_matches_blurred_drop_shadow():
    from moviepy.ae.templates.paper import drop_shadow, rect_shadow

    size, rect = (80, 60), (20, 15, 30, 25)
    fast = rect_shadow(size, rect, 4.0, 0.5)
    ref = drop_shadow(np.ones((25, 30), np.float32), size, rect[:2], 4.0, 0.5)
    assert np.abs(fast - ref[..., 3] / 255.0).max() < 0.02


def test_cached_av_layer_imports_each_state_once():
    from moviepy.ae.templates.paper import CachedAVLayer, cached_rgba_clip

    clip = cached_rgba_clip(
        (4, 4),
        2.0,
        lambda t: int(t),
        lambda k: (np.full((4, 4, 3), 10 * k, np.uint8), np.ones((4, 4))),
    )
    layer = CachedAVLayer(clip, "x")
    first = layer.source_buffer(0.2)
    assert layer.source_buffer(0.9) is first
    assert layer.source_buffer(1.2) is not first
    assert layer.source_buffer(1.2).to_uint8_rgb()[0, 0, 0] == 10
    assert clip.builds == 2


def test_cached_rgba_clip_respects_byte_budget():
    from moviepy.ae.templates.paper import cached_rgba_clip

    clip = cached_rgba_clip(
        (16, 16),
        10.0,
        lambda t: int(t),
        lambda k: (np.zeros((16, 16, 3), np.uint8), np.ones((16, 16))),
        max_bytes=2 * (16 * 16 * 3 + 16 * 16 * 4),
    )
    for t in range(6):
        clip.get_frame(float(t))
    assert len(clip.states) == 2


@needs_kaiu
def test_chapter_single_item_never_flickers_at_period_boundaries(preset):
    comp = chapter_tag(["本章人物"], 1, 14, preset, period=2.0).composition
    ref, ref_mask = comp.get_frame(1.0), comp.mask.get_frame(1.0)
    for t in (2.0, 2.1, 2.25, 2.4, 4.05, 8.1):
        assert np.array_equal(comp.get_frame(t), ref), t
        assert np.array_equal(comp.mask.get_frame(t), ref_mask), t


@needs_kaiu
@pytest.mark.parametrize("duration", [0.05, 0.1, 0.5, 1.0, 1.15, 1.5])
def test_chapter_very_short_durations_build_and_render(preset, duration):
    tag = chapter_tag(["甲", "乙"], 2, duration, preset, period=5.5)
    last = max(0.0, duration - 1 / 24)
    for t in (0.0, last, duration * 0.5):
        assert tag.composition.get_frame(t).shape[2] == 3


@needs_kaiu
def test_chapter_duration_shorter_than_period_never_rotates(preset):
    tag = chapter_tag(["甲", "乙乙乙乙"], 1, 3.0, preset, period=10.0)
    comp = tag.composition
    ref = comp.get_frame(1.5)
    assert np.array_equal(ref, comp.get_frame(2.0))
    assert tag.strip.clip.builds <= 30


@needs_kaiu
def test_chapter_very_long_item_shrinks_to_fit_frame(preset):
    tag = chapter_tag(["本章人物與事件" * 12], 1, 4, preset)
    assert tag.meta["x"] + tag.meta["w"] <= preset.size[0]
    assert tag.meta["font_px"] < 44
    wide = chapter_tag(["本章人物與事件" * 12], 1, 4, preset, shrink_to_fit=False)
    assert wide.meta["x"] + wide.meta["w"] > preset.size[0]


@needs_kaiu
def test_chapter_hold_frames_are_pure_cache_hits(preset):
    tag = chapter_tag(["第一章 鴻門宴", "楚漢相爭"], 1, 20, preset)
    comp, clip = tag.composition, tag.strip.clip
    comp.render_buffer(3.0)
    builds = clip.builds
    start = time.perf_counter()
    for i in range(24):
        comp.render_buffer(3.0 + i / 24)
    per_frame = (time.perf_counter() - start) / 24
    assert clip.builds == builds
    assert per_frame < 0.1


@needs_kaiu
def test_chapter_deterministic(preset):
    a = chapter_tag(["甲", "乙"], 3, 8, preset, seed=5).composition
    b = chapter_tag(["甲", "乙"], 3, 8, preset, seed=5).composition
    for t in (0.3, 2.0, 5.7):
        assert np.array_equal(a.get_frame(t), b.get_frame(t))
        assert np.array_equal(a.mask.get_frame(t), b.mask.get_frame(t))


@needs_kaiu
@pytest.mark.parametrize("dynasty", ["西漢", "明"])
def test_quote_frame_time_boundaries(preset, dynasty):
    comp = vertical_quote(QUOTE, "《史記》", preset, dynasty=dynasty, seal="史")
    start = comp.render_buffer(0.0)
    assert start.rgba[..., 3].max() < 0.5
    end = comp.render_buffer(comp.duration - 1 / 24)
    assert end.rgba.shape == (1080, 1920, 4)
    assert np.isfinite(end.rgba).all()
    assert end.rgba[..., 3].max() < 0.35  # faded out
    held = comp.render_buffer(comp.quote_meta["write_end"] + 0.5).rgba
    # premultiplied invariant: colour never exceeds alpha
    assert (held[..., :3] <= held[..., 3:4] + 1e-4).all()
    assert held[..., 3].max() > 0.9


@needs_kaiu
def test_quote_punctuation_only_column_renders(preset):
    text = "「" + "。" * 7 + "」"
    comp = vertical_quote(text, None, preset, dynasty="唐", dim=0)
    frame = comp.render_buffer(comp.quote_meta["write_end"] + 0.5).rgba
    assert frame[..., 3].max() > 0.5


@needs_kaiu
def test_quote_layers_sit_on_whole_pixels_when_holding(preset):
    for dynasty in ("西漢", "明"):
        comp = vertical_quote(QUOTE, "《史記》", preset, dynasty=dynasty)
        t = comp.quote_meta["write_end"] + 0.5
        for name, layer in comp.quote_layers.items():
            if name in ("dim", "seal"):
                continue
            x, y = layer.transform.position.value_at(t)
            assert x == int(x) and y == int(y), name


def _builds(comp):
    return sum(
        getattr(layer.clip, "builds", 0)
        for layer in comp.quote_layers.values()
        if hasattr(layer, "clip")
    )


@needs_kaiu
@pytest.mark.parametrize("dynasty", ["西漢", "明"])
def test_quote_holding_frames_are_fast_and_cached(preset, dynasty):
    comp = vertical_quote(QUOTE, "《史記》", preset, dynasty=dynasty, title="貨殖")
    t = comp.quote_meta["write_end"] + 0.5
    comp.render_buffer(t)
    builds = _builds(comp)
    start = time.perf_counter()
    for i in range(24):
        comp.render_buffer(t + i / 24)
    per_frame = (time.perf_counter() - start) / 24
    print(f"\nquote {dynasty} holding 1080p: {per_frame * 1000:.1f} ms/frame")
    assert per_frame < 0.25  # measured ~45 ms; loose so slow machines do not flake
    assert _builds(comp) == builds
