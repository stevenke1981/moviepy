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
needs_kaiu = pytest.mark.skipif(not KAIU.is_file(), reason="kaiu.ttf not installed")

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
