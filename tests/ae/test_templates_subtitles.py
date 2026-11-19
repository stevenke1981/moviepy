"""Tests for moviepy.ae.templates.subtitles."""

import importlib.util
from pathlib import Path

import pytest

from moviepy.ae.templates.presets import get_preset
from moviepy.ae.templates.subtitles import (
    Cue,
    LayoutError,
    SubtitleTiming,
    Word,
    break_lines,
    burn_subtitles,
    check_timing,
    find_bad_breaks,
    load_words,
    parse_srt,
    reflow_cues,
    subtitle_layer,
    to_ass,
    to_srt,
    to_vtt,
)


def m10(text):
    return 10.0 * len(text)


KAIU = Path("C:/Windows/Fonts/kaiu.ttf")


# -- Cue ----------------------------------------------------------------- #


@pytest.mark.parametrize(
    "args",
    [(2, 1, "x"), (-1, 1, "x"), (0, 1, "  "), (float("nan"), 1, "x"), (0, True, "x")],
)
def test_cue_validation(args):
    with pytest.raises(ValueError):
        Cue(*args)


def test_cue_lang_validation_and_normalization():
    with pytest.raises(ValueError):
        Cue(0, 1, "x", lang=" ")
    assert Cue(0, 1, " a \n b ").text == "a\nb"


# -- line breaking --------------------------------------------------------- #


def test_cjk_break_after_punctuation_and_balanced():
    for jieba in (True, False):
        lines = break_lines("今天天氣很好，我們去公園散步。", m10, 100, use_jieba=jieba)
        assert lines == ["今天天氣很好，", "我們去公園散步。"]


def test_cjk_fits_one_line():
    assert break_lines("你好，世界。", m10, 500) == ["你好，世界。"]


def test_protected_term_never_split():
    term = "臺北故宮博物院"
    text = "他在臺北故宮博物院裡等候直到夜深了才離開"
    lines = break_lines(text, m10, 130, protected=[term], allow_char_breaks=True)
    assert "".join(lines) == text
    assert any(term in line for line in lines)
    problems = find_bad_breaks(lines, [term])
    assert not [p for p in problems if p.reason == "inside_protected"]
    if importlib.util.find_spec("jieba") is not None:
        # Without jieba, character breaks cannot be confirmed as word
        # boundaries, so only the protected-term rule is checked above.
        assert problems == []
    bad = find_bad_breaks(["他在臺北故宮", "博物院裡等候"], [term])
    assert bad and bad[0].reason == "inside_protected"


def test_digit_and_year_runs_not_split():
    lines = break_lines(
        "事情發生於2024年十二月之後無人再提", m10, 110, allow_char_breaks=True
    )
    joined = "|".join(lines)
    assert "2024年" in joined and "十二月" in joined


def test_kinsoku():
    text = "「你好」，他說。「再見」，她答。"
    lines = break_lines(text, m10, 90, allow_char_breaks=True)
    for line in lines:
        assert line[0] not in "，。」』"
        assert line[-1] not in "「『"
    assert find_bad_breaks(["他說「", "你好」"])[0].reason == "line_end_forbidden"
    assert find_bad_breaks(["你好", "，他說"])[0].reason == "line_start_forbidden"


def test_cjk_without_boundary_raises_unless_opted_in():
    text = "這是一段沒有任何標點的很長句子必須要換行才行"
    with pytest.raises(LayoutError):
        break_lines(text, m10, 130, use_jieba=False)
    assert len(break_lines(text, m10, 130, allow_char_breaks=True)) == 2


def test_english_breaks_at_spaces_only():
    text = "the quick brown fox jumps over the lazy dog"
    lines = break_lines(text, len, 25, lang="en")
    assert " ".join(lines) == text
    assert all(len(line) <= 25 for line in lines)
    assert len(lines) == 2
    assert abs(len(lines[0]) - len(lines[1])) <= 6


def test_english_protected_name_and_hyphen_kept():
    lines = break_lines(
        "she met Lin Yuqing near the well-known bridge",
        len,
        24,
        lang="en",
        protected=["Lin Yuqing"],
    )
    assert any("Lin Yuqing" in line for line in lines)
    assert any("well-known" in line for line in lines)
    assert find_bad_breaks(["she met Lin", "Yuqing today"], ["Lin Yuqing"])


def test_layout_error_never_truncates():
    with pytest.raises(LayoutError):
        break_lines("aaa bbb ccc ddd eee", len, 7, lang="en")
    with pytest.raises(LayoutError):
        break_lines("supercalifragilistic", len, 10, lang="en")
    with pytest.raises(LayoutError):
        break_lines("a b c d", len, 3, lang="en", max_lines=1)


def test_forced_newlines_respected():
    assert break_lines("one\ntwo", len, 20, lang="en") == ["one", "two"]
    with pytest.raises(LayoutError):
        break_lines("one\ntwo\nthree", len, 20, lang="en")


def test_find_bad_breaks_accepts_cues_and_blocks():
    cues = [Cue(0, 2, "你好，\n世界"), Cue(2, 4, "ab\ncd", "en")]
    assert find_bad_breaks(cues) == []
    assert find_bad_breaks(["Lin\\NYuqing"], ["Lin Yuqing"])[0].block == 0
    assert find_bad_breaks([]) == []


# -- formats --------------------------------------------------------------- #

FIXTURE = [Cue(0.5, 3.5, "你好，\n世界", "zh-TW"), Cue(3.5, 5.0, "A & B", "en")]


def test_srt_exact():
    assert to_srt(FIXTURE) == (
        "1\n00:00:00,500 --> 00:00:03,500\n你好，\n世界\n\n"
        "2\n00:00:03,500 --> 00:00:05,000\nA & B\n\n"
    )
    assert "A &amp; B" in to_srt(FIXTURE, escape=True)


def test_vtt_exact():
    assert to_vtt(FIXTURE) == (
        "WEBVTT\n\n"
        "00:00:00.500 --> 00:00:03.500\n你好，\n世界\n\n"
        "00:00:03.500 --> 00:00:05.000\nA &amp; B\n\n"
    )


def test_ass_exact_lines():
    style = get_preset("nightlamp_story").subtitles
    lines = to_ass(FIXTURE, style, (1920, 1080)).splitlines()
    assert "PlayResX: 1920" in lines and "PlayResY: 1080" in lines
    assert (
        "Style: Primary,Microsoft JhengHei,72,&H00FFFFFF,&H00FFFFFF,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,4,0,2,134,134,60,1"
    ) in lines
    assert (
        "Style: Secondary,Microsoft JhengHei,42,&H00EBEBEB,&H00EBEBEB,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,4,0,2,134,134,60,1"
    ) in lines
    assert "Dialogue: 0,0:00:00.50,0:00:03.50,Primary,,0,0,60,,你好，\\N世界" in lines
    assert "Dialogue: 0,0:00:03.50,0:00:05.00,Secondary,,0,0,60,,A & B" in lines
    half = to_ass(FIXTURE, style, (1280, 720))
    assert "Style: Primary,Microsoft JhengHei,48," in half


def test_ass_lifts_primary_over_overlapping_secondary():
    style = get_preset("nightlamp_story").subtitles
    cues = [Cue(0, 2, "主", "zh-TW"), Cue(0, 2, "sub", "en")]
    ass = to_ass(cues, style)
    assert "Primary,,0,0,60,," not in ass
    assert "Secondary,,0,0,60,,sub" in ass


def test_ass_rejects_control_codes():
    style = get_preset("nightlamp_story").subtitles
    with pytest.raises(ValueError):
        to_ass([Cue(0, 1, "a{b}")], style)


def test_srt_round_trip():
    back = parse_srt(to_srt(FIXTURE), "zh-TW")
    assert [(c.start, c.end, c.text) for c in back] == [
        (c.start, c.end, c.text) for c in FIXTURE
    ]
    crlf = "\ufeff" + to_srt(FIXTURE).replace("\n", "\r\n")
    assert len(parse_srt(crlf)) == 2
    with pytest.raises(ValueError):
        parse_srt("1\nnot a time\ntext\n")


# -- timing ---------------------------------------------------------------- #


def test_timing_checks():
    cues = [
        Cue(0, 0.5, "short", "en"),
        Cue(1, 9, "long one", "en"),
        Cue(10, 11, "x" * 20, "en"),
        Cue(10.5, 13, "overlap", "en"),
        Cue(10.5, 13, "不同語言可以重疊", "zh-TW"),
        Cue(20, 21, "這是一個很長的字幕超過十二字每秒", "zh-TW"),
    ]
    kinds = {(i.index, i.kind) for i in check_timing(cues)}
    assert (0, "too_short") in kinds
    assert (1, "too_long") in kinds
    assert (2, "cps") in kinds
    assert (3, "overlap") in kinds
    assert (5, "cps") in kinds
    assert not any(i == 4 for i, _ in kinds)


def test_timing_defaults_match_r23_profile():
    t = SubtitleTiming()
    assert (t.min_duration, t.max_duration) == (1.0, 7.0)
    assert (t.limit_cps("zh-TW"), t.limit_cps("zh-CN"), t.limit_cps("en")) == (
        12.0,
        12.0,
        17.0,
    )
    prof = {"min_duration": 0.8, "languages": {"en": {"max_cps": 15}}}
    assert SubtitleTiming.from_profile(prof).limit_cps("en") == 15.0
    assert check_timing([Cue(0, 2, "fine", "en")]) == []


# -- layer ------------------------------------------------------------------ #


@pytest.fixture
def preset():
    return get_preset("nightlamp_story")


def test_layer_transparent_between_cues_and_opaque_when_active(preset):
    size = (640, 360)
    layer = subtitle_layer(
        [Cue(1, 2, "Hello there"), Cue(3, 4, "Second line")],
        preset,
        font="default",
        size=size,
    )
    assert layer.render(0.5) is None  # outside the layer window
    gap = layer.render(2.5)
    assert gap is not None and gap.rgba[..., 3].max() == 0.0
    active = layer.render(1.5)
    assert active.rgba[..., 3].max() == 1.0
    x0, y0 = active.offset
    assert y0 >= size[1] // 2  # bottom region
    assert y0 + active.size[1] <= size[1]
    assert 0 <= x0 and x0 + active.size[0] <= size[0]
    solid = active.rgba[..., 3] > 0.99
    brightness = active.rgba[..., :3].max(axis=-1)
    assert (brightness[solid] < 0.1).any()  # black outline
    assert (brightness > 0.99).any()  # white fill


def test_bilingual_primary_above_secondary(preset):
    size = (640, 360)
    layer = subtitle_layer(
        [Cue(0, 2, "Primary text")],
        preset,
        secondary=[Cue(0, 2, "Secondary text", "en")],
        font="default",
        size=size,
    )
    both = layer.render(1.0)
    solo = subtitle_layer(
        [Cue(0, 2, "Primary text")], preset, font="default", size=size
    ).render(1.0)
    assert both.size[1] > solo.size[1]
    assert both.offset[1] < solo.offset[1]  # taller block grows upward
    assert both.offset[1] + both.size[1] == solo.offset[1] + solo.size[1]


def test_layer_cache_reuse(preset):
    layer = subtitle_layer(
        [Cue(0, 4, "Cached cue")], preset, font="default", size=(640, 360)
    )
    first = layer.source_buffer(1.0)
    for t in (1.1, 2.0, 3.9):
        assert layer.source_buffer(t) is first
    assert layer.rasters == 1
    hits, misses, _, _ = layer.cache_info()
    assert (hits, misses) == (3, 1)


def test_layer_lru_eviction(preset):
    cues = [Cue(i, i + 0.5, f"cue {i}") for i in range(6)]
    layer = subtitle_layer(cues, preset, font="default", size=(640, 360), cache_size=2)
    for i in range(6):
        layer.source_buffer(i + 0.1)
    assert layer.cache_info()[2] == 2


def test_layer_wraps_long_cue_and_raises_when_impossible(preset):
    text = "word " * 12
    layer = subtitle_layer(
        [Cue(0, 3, text.strip(), "en")], preset, font="default", size=(640, 360)
    )
    assert 1 < len(layer._lanes["primary"]["lines"][0]) <= 2
    with pytest.raises(LayoutError):
        subtitle_layer(
            [Cue(0, 3, "x" * 400, "en")], preset, font="default", size=(640, 360)
        )


def test_missing_font_raises(preset):
    with pytest.raises(FileNotFoundError):
        subtitle_layer([Cue(0, 1, "x")], preset, font="Z:/nope/missing.ttf")


def test_burn_subtitles_renders_frames(preset):
    from moviepy.ae.composition import Composition

    comp = Composition(size=(320, 180), fps=10, duration=3, bg_color=(0, 0, 0))
    out = burn_subtitles(comp, [Cue(1, 2, "Burned in")], preset, font="default")
    assert out is comp
    blank = comp.get_frame(0.2)
    shown = comp.get_frame(1.5)
    assert blank.max() == 0
    assert shown[90:].max() > 200 and shown[:90].max() == 0


@pytest.mark.skipif(not KAIU.is_file(), reason="kaiu.ttf not installed")
def test_cjk_layer_with_kaiu(preset):
    layer = subtitle_layer(
        [Cue(0, 3, "林雨晴慢慢讀信，窗外的燈火一盞一盞亮起。")],
        preset,
        secondary=[Cue(0, 3, "Lin Yuqing reads the letter slowly.", "en")],
        font=str(KAIU),
        size=(1280, 720),
    )
    buf = layer.render(1.0)
    assert buf.rgba[..., 3].max() == 1.0
    assert buf.offset[1] > 360


# -- review regressions ------------------------------------------------------ #


def test_protected_terms_do_not_mutate_global_jieba():
    jieba = pytest.importorskip("jieba")
    jieba.setLogLevel(60)
    term = "測試專屬詞彙甲乙丙"
    before = jieba.lcut("今天" + term + "很好", HMM=False)
    freq_before = len(jieba.dt.FREQ) if jieba.dt.initialized else None
    lines = break_lines(
        "今天" + term + "很好，我們去公園散步。",
        m10,
        120,
        protected=[term],
    )
    assert any(term in line for line in lines)  # kept whole
    assert term not in jieba.dt.FREQ
    assert jieba.lcut("今天" + term + "很好", HMM=False) == before
    if freq_before is not None:
        assert len(jieba.dt.FREQ) == freq_before
    find_bad_breaks(["今天" + term, "很好"], [term])
    assert term not in jieba.dt.FREQ


def test_protected_term_edges_are_word_boundaries():
    # The term is glued to neighbours that jieba would otherwise merge with it.
    text = "他說林雨晴慢慢讀信，窗外的燈火亮起。"
    lines = break_lines(text, m10, 110, protected=["林雨晴"])
    assert "".join(lines) == text
    assert all("林雨晴" in line or "林" not in line for line in lines)


def test_forced_newlines_never_exceed_total_line_budget():
    text = "aaaa bbbb cccc dddd\neeee ffff gggg hhhh"
    with pytest.raises(LayoutError):
        break_lines(text, len, 10, lang="en", max_lines=3)
    ok = break_lines("aaaa bbbb cccc dddd\neeee", len, 10, lang="en", max_lines=3)
    assert len(ok) <= 3


def test_srt_text_is_literal_by_default_and_escape_is_opt_in():
    cue = Cue(0, 1, "Tom & Jerry <3 >_<")
    assert "Tom & Jerry <3 >_<\n" in to_srt([cue])
    assert "Tom &amp; Jerry &lt;3 &gt;_&lt;\n" in to_srt([cue], escape=True)
    assert "Tom &amp; Jerry &lt;3 &gt;_&lt;" in to_vtt([cue])  # VTT is markup


def test_ass_lift_follows_font_metrics():
    from moviepy.ae.templates.subtitles import SubtitleLayer  # noqa: F401

    preset = get_preset("nightlamp_story")
    size = (1920, 1080)
    cues = [Cue(0, 2, "Primary text")]
    sec = [Cue(0, 2, "Secondary text\nsecond row", "en")]
    both = subtitle_layer(cues, preset, secondary=sec, font="default", size=size)
    solo = subtitle_layer(cues, preset, font="default", size=size)
    raised = solo.render(1.0).offset[1] - both.render(1.0).offset[1]
    ass = to_ass(cues + sec, preset.subtitles, size, font="default")
    margin = next(
        int(line.split(",")[7])
        for line in ass.splitlines()
        if line.startswith("Dialogue") and ",Primary," in line
    )
    base = int(round(preset.subtitles.margin_bottom))
    assert margin - base == raised
    # Without a font the documented approximation is kept.
    approx = to_ass(cues + sec, preset.subtitles, size)
    assert approx != ass


# -- reflow ----------------------------------------------------------------- #


def test_reflow_keeps_fitting_cue_untouched():
    cue = Cue(1, 2, "短句。")
    report = []
    assert reflow_cues([cue], m10, 200, report=report) == [cue]
    assert report[0]["timing"] == "unchanged"


def test_reflow_splits_at_punctuation_and_keeps_all_text():
    text = "嬌娜伸手按住孔生的胸口，紅丸在掌心轉動，傷處漸漸癒合。"
    out = reflow_cues([Cue(0, 9, text)], m10, 130, protected=["孔生"])
    assert "".join(c.text for c in out) == text
    assert all(m10(c.text) <= 130 for c in out)
    assert [c.text[-1] for c in out] == ["，", "，", "。"]
    assert out[0].start == 0 and out[-1].end == 9
    assert all(a.end == b.start for a, b in zip(out, out[1:]))


def test_reflow_never_cuts_protected_name():
    text = "我們今天要介紹的人物是皇甫公子與嬌娜姑娘"
    out = reflow_cues(
        [Cue(0, 6, text)],
        m10,
        110,
        protected=["皇甫公子", "嬌娜姑娘"],
        allow_char_breaks=True,
    )
    joined = "|".join(c.text for c in out)
    assert "皇甫公子" in joined and "嬌娜姑娘" in joined
    assert all(m10(c.text) <= 110 for c in out)


def test_reflow_uses_word_timings_and_refuses_cuts_inside_words():
    # No punctuation: only ASR word edges make the cut legal.
    text = "松娘輕聲說道她會一直守在門外"
    tokens = [
        ("松娘", 0.0, 0.5),
        ("輕聲", 0.5, 1.0),
        ("說道", 1.0, 1.5),
        ("她會", 1.5, 2.0),
        ("一直", 2.6, 3.0),
        ("守在", 3.0, 3.5),
        ("門外", 3.5, 4.0),
    ]
    words = [Word(*t) for t in tokens]
    report = []
    out = reflow_cues(
        [Cue(0, 4, text)], m10, 80, words=words, use_jieba=False, report=report
    )
    assert "".join(c.text for c in out) == text
    assert all(len(c.text) % 2 == 0 for c in out)
    assert report[0]["timing"] == "words"
    assert len(out) == 2
    ends = {2: 0.5, 4: 1.0, 6: 1.5, 8: 2.0, 10: 3.0, 12: 3.5}
    starts = {2: 0.5, 4: 1.0, 6: 1.5, 8: 2.6, 10: 3.0, 12: 3.5}
    cut = len(out[0].text)
    if starts[cut] - ends[cut] > 0.3:  # a real pause stays a gap
        assert (out[0].end, out[1].start) == (ends[cut], starts[cut])
    else:  # a short one is closed so the caption does not flicker
        assert out[0].end == out[1].start == starts[cut]


def test_reflow_word_pause_is_kept_as_gap():
    text = "她會一直守在門外。我知道"
    words = [Word("她會一直守在門外", 0.0, 2.0), Word("我知道", 2.8, 3.6)]
    out = reflow_cues([Cue(0, 3.6, text)], m10, 90, words=words)
    assert [c.text for c in out] == ["她會一直守在門外。", "我知道"]
    assert (out[0].end, out[1].start) == (2.0, 2.8)


def test_reflow_without_words_or_breaks_raises():
    with pytest.raises(LayoutError):
        reflow_cues(
            [Cue(0, 4, "松娘輕聲說道她會一直守在門外")], m10, 80, use_jieba=False
        )


def test_reflow_english_two_lines_per_cue():
    text = "Kong Xueli woke in the garden and found the fox girl watching over him"
    out = reflow_cues(
        [Cue(0, 8, text, "en")], m10, 200, max_lines=2, protected=["Kong Xueli"]
    )
    assert " ".join(c.text.replace("\n", " ") for c in out) == text
    assert all(len(c.text.split("\n")) <= 2 for c in out)
    assert "Kong Xueli" in out[0].text


def test_load_words_shapes(tmp_path):
    path = tmp_path / "asr.json"
    path.write_text(
        '{"segments": [{"words": [{"word": "a", "start": 0, "end": 1}]}]}',
        encoding="utf-8",
    )
    assert load_words(path) == [Word("a", 0, 1)]
    assert load_words([{"text": "b", "start": 1, "end": 2}])[0].start == 1.0
    with pytest.raises(ValueError):
        load_words({"x": []})


def test_layer_split_overflow_makes_more_cues(preset):
    text = " ".join(["slowly"] * 30)
    cues = [Cue(0, 6, text, "en")]
    with pytest.raises(LayoutError):
        subtitle_layer(cues, preset, font="default", size=(640, 360))
    layer = subtitle_layer(
        cues, preset, font="default", size=(640, 360), overflow="split"
    )
    lane = layer._lanes["primary"]["cues"]
    assert len(lane) > 1 and lane[0].start == 0 and lane[-1].end == 6
    assert layer.reflow_report[0]["timing"] == "proportional"
    with pytest.raises(ValueError):
        subtitle_layer(cues, preset, font="default", overflow="cut")


def test_reflow_line_breaks_prefer_punctuation_over_asr_edges():
    text = "替你尋一位合適的妻子。若真能娶到像香奴這樣的女子"
    words = [Word(ch, i * 0.2, i * 0.2 + 0.2) for i, ch in enumerate(text)]
    out = reflow_cues(
        [Cue(0, 6, text)], m10, 160, max_lines=2, words=words, use_jieba=False
    )
    assert out[0].text.split("\n")[0].endswith("。")
