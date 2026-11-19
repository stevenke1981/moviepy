"""Study schedule, ASS/ffmetadata export and overlay layer vs the source port."""

import math
import os

import numpy as np

import pytest

from moviepy.ae import Composition
from moviepy.ae.templates.study import StudyOverlayLayer, StudySchedule


FONTS = ["msjh.ttc", "YuGothM.ttc", "consola.ttf"]
HAVE_FONTS = all(os.path.isfile("C:/Windows/Fonts/" + f) for f in FONTS)
needs_fonts = pytest.mark.skipif(not HAVE_FONTS, reason="Windows fonts missing")


def toy():
    def phase(pid, kind, a, b):
        data = {"id": pid, "type": kind, "start": a, "end": b}
        for key in ("label", "label_en", "label_ja", "tip", "tip_en", "tip_ja"):
            data[key] = f"{key}-{pid}"
        return data

    return {
        "episode": "Toy; ep=1 #a\\b",
        "duration_seconds": 10,
        "focus_seconds": 4,
        "rest_seconds": 6,
        "display": {
            "timer_font": "Consolas",
            "zh_font": "Microsoft JhengHei",
            "ja_font": "Yu Gothic",
            "timer_panel_opacity": 0.3,
            "title_panel_opacity": 0.3,
            "tip_panel_opacity": 0.4,
            "tip_display_seconds": 3,
            "font_sizes": {
                "title_zh": 30,
                "title_en": 26,
                "title_ja": 26,
                "phase_zh": 35,
                "phase_en": 27,
                "phase_ja": 27,
                "timer": 68,
                "tip_zh": 37,
                "tip_en": 29,
                "tip_ja": 29,
            },
        },
        "title": {"zh": "標題", "en": "Title {\\b1}", "ja": "題"},
        "phases": [
            phase("F1", "focus", 0, 4),
            phase("B1", "break", 4, 7),
            phase("C1", "closing", 7, 10),
        ],
        "study_chapters": [
            {"start": 0, "name": "One"},
            {"start": 4, "name": "a=b;c#d\\e\nf"},
            {"start": 7, "name": "End"},
        ],
        "preview": {"label": "Demo {x}"},
    }


# ---- direct port of the source functions (reference) ------------------------ #


def ref_phase_at(config, second):
    return next(p for p in config["phases"] if p["start"] <= second < p["end"])


def ref_clock(seconds):
    return f"{seconds // 60:02}:{seconds % 60:02}"


def ref_ass_time(second):
    cs = round(second * 100)
    hours, rest = divmod(cs, 360000)
    minutes, rest = divmod(rest, 6000)
    seconds, fraction = divmod(rest, 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{fraction:02}"


def ref_ass_text(text):
    return (
        str(text)
        .replace("\\", "／")
        .replace("{", "（")
        .replace("}", "）")
        .replace("\n", r"\N")
    )


REF_HEADER = """[Script Info]
Title: Quiet study companion
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
__STYLES__

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ref_build_ass(config, start=0, duration=None, demo=False):
    duration = config["duration_seconds"] - start if duration is None else duration
    end = start + duration
    display = config["display"]
    sizes = display["font_sizes"]
    zh, ja = display["zh_font"], display["ja_font"]
    styles = [
        ("Panel", zh, 28),
        ("TitleZH", zh, sizes["title_zh"]),
        ("TitleEN", zh, sizes["title_en"]),
        ("TitleJA", ja, sizes["title_ja"]),
        ("LabelZH", zh, sizes["phase_zh"]),
        ("LabelEN", zh, sizes["phase_en"]),
        ("LabelJA", ja, sizes["phase_ja"]),
        ("Clock", display["timer_font"], sizes["timer"]),
        ("TipZH", zh, sizes["tip_zh"]),
        ("TipEN", zh, sizes["tip_en"]),
        ("TipJA", ja, sizes["tip_ja"]),
        ("Demo", zh, 20),
    ]
    header = REF_HEADER.replace(
        "__STYLES__",
        "\n".join(
            f"Style: {n},{f},{s},&H00F4F6EE,&H00F4F6EE,&H00132619,&H00132619,0,0,0,0,"
            f'100,100,0,0,1,{0 if n == "Panel" else 0.65},0,7,0,0,0,1'
            for n, f, s in styles
        ),
    )
    events = []

    def event(layer, a, b, style, text):
        if b > a:
            events.append(
                f"Dialogue: {layer},{ref_ass_time(a - start)},"
                f"{ref_ass_time(b - start)},{style},,0,0,0,,{text}"
            )

    def panel(a, b, x, y, width, height, opacity, fade=""):
        alpha = round((1 - opacity) * 255)
        text = (
            f"{{\\pos({x},{y})\\p1\\1c&H172419&\\1a&H{alpha:02X}&{fade}}}"
            f"m 0 0 l {width} 0 {width} {height} 0 {height}{{\\p0}}"
        )
        event(0, a, b, "Panel", text)

    panel(start, end, 900, 38, 338, 247, display["timer_panel_opacity"])
    panel(start, end, 40, 38, 700, 148, display["title_panel_opacity"])
    for language, style, y in [
        ("zh", "TitleZH", 49),
        ("en", "TitleEN", 97),
        ("ja", "TitleJA", 137),
    ]:
        event(
            2,
            start,
            end,
            style,
            f"{{\\pos(58,{y})}}" + ref_ass_text(config["title"][language]),
        )
    if demo:
        panel(start, end, 40, 474, 550, 35, display["title_panel_opacity"])
        event(
            3,
            start,
            end,
            "Demo",
            r"{\pos(54,478)}" + ref_ass_text(config["preview"]["label"]),
        )
    for phase in config["phases"]:
        a, b = max(start, phase["start"]), min(end, phase["end"])
        color = r"\1c&HF4EBD1&" if phase["type"] != "focus" else ""
        for key, style, y in [
            ("label", "LabelZH", 52),
            ("label_en", "LabelEN", 102),
            ("label_ja", "LabelJA", 140),
        ]:
            event(
                2, a, b, style, f"{{\\pos(918,{y}){color}}}" + ref_ass_text(phase[key])
            )
        tip_end = min(phase["end"], phase["start"] + display["tip_display_seconds"])
        tip_a, tip_b = max(start, phase["start"]), min(end, tip_end)
        fade_in = 500 if tip_a == phase["start"] else 0
        fade_out = 700 if tip_b == tip_end else 0
        fade = f"\\fad({fade_in},{fade_out})"
        panel(tip_a, tip_b, 40, 244, 838, 164, display["tip_panel_opacity"], fade)
        for key, style, y in [
            ("tip", "TipZH", 254),
            ("tip_en", "TipEN", 313),
            ("tip_ja", "TipJA", 358),
        ]:
            event(
                2,
                tip_a,
                tip_b,
                style,
                f"{{\\pos(58,{y}){fade}}}" + ref_ass_text(phase[key]),
            )
    for second in range(start, end):
        phase = ref_phase_at(config, second)
        remaining = math.ceil(phase["end"] - second)
        event(2, second, second + 1, "Clock", r"{\pos(918,179)}" + ref_clock(remaining))
    return header + "\n".join(events) + "\n"


def ref_chapters(config):
    lines = [";FFMETADATA1", f"title={config['episode']}"]
    chapters = config["study_chapters"]
    for i, chapter in enumerate(chapters):
        start = chapter["start"]
        if i + 1 < len(chapters):
            end = chapters[i + 1]["start"]
        else:
            end = config["duration_seconds"]
        title = (
            chapter["name"]
            .replace("\\", "\\\\")
            .replace("=", "\\=")
            .replace(";", "\\;")
            .replace("#", "\\#")
            .replace("\n", " ")
        )
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start * 1000}",
                f"END={end * 1000}",
                f"title={title}",
            ]
        )
    return "\n".join(lines) + "\n"


# ---- validation ------------------------------------------------------------- #


def mutated(fn):
    config = toy()
    fn(config)
    return config


BAD = [
    (lambda c: c.update(duration_seconds=0), "positive integer"),
    (lambda c: c.update(duration_seconds=10.0), "positive integer"),
    (lambda c: c["phases"][1].update(start=5), "non-contiguous"),
    (lambda c: c["phases"][1].update(id="F1"), "Duplicate"),
    (lambda c: c["phases"][0].update(end=0), "Invalid phase interval"),
    (lambda c: c["phases"][0].update(type="nap"), "Unknown phase type"),
    (lambda c: c["phases"][2].update(end=11), "whole episode"),
    (lambda c: c.update(focus_seconds=5), "Focus total"),
    (lambda c: c.update(rest_seconds=5), "Rest total"),
    (lambda c: c["display"].update(tip_panel_opacity=1.5), "opacity"),
    (lambda c: c["display"].update(timer_panel_opacity=-0.1), "opacity"),
    (lambda c: c["phases"][1].update(tip_ja=""), "Japanese"),
    (lambda c: c["phases"][1].pop("label_en"), "Japanese"),
    (lambda c: c.pop("phases"), "Missing"),
]


@pytest.mark.parametrize("fn,message", BAD)
def test_validation_errors(fn, message):
    with pytest.raises(ValueError, match=message):
        StudySchedule.from_dict(mutated(fn))


def test_dict_roundtrip_and_isolation():
    config = toy()
    schedule = StudySchedule.from_dict(config)
    out = schedule.to_dict()
    assert out == config
    out["phases"].clear()
    config["phases"].clear()
    assert len(schedule.phases) == 3


def test_from_json(tmp_path):
    import json

    path = tmp_path / "s.json"
    path.write_text(json.dumps(toy(), ensure_ascii=False), encoding="utf-8")
    assert StudySchedule.from_json(path).to_dict() == toy()


def test_phase_and_remaining_boundaries():
    s = StudySchedule.from_dict(toy())
    assert s.phase_at(0)["id"] == "F1" and s.phase_at(3.999)["id"] == "F1"
    assert s.phase_at(4)["id"] == "B1" and s.phase_at(9.5)["id"] == "C1"
    assert s.remaining_at(0)[1] == 4 and s.remaining_at(3.5)[1] == 1
    assert s.remaining_at(4)[1] == 3 and s.remaining_at(9)[1] == 1
    for bad in (-1, 10, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            s.phase_at(bad)


def test_chime_times():
    assert StudySchedule.from_dict(toy()).chime_times() == [4, 7]


def test_real_style_chimes():
    config = toy()
    config.update(duration_seconds=5220, focus_seconds=4500, rest_seconds=720)
    spans = [
        (0, 1500, "focus"),
        (1500, 1800, "break"),
        (1800, 3300, "focus"),
        (3300, 3600, "break"),
        (3600, 5100, "focus"),
        (5100, 5220, "closing"),
    ]
    config["phases"] = [
        dict(config["phases"][0], id=f"P{i}", type=kind, start=a, end=b)
        for i, (a, b, kind) in enumerate(spans)
    ]
    config["study_chapters"] = [{"start": 0, "name": "x"}]
    schedule = StudySchedule.from_dict(config)
    assert schedule.chime_times() == [1500, 1800, 3300, 3600, 5100]


# ---- exports ---------------------------------------------------------------- #


def test_ffmetadata_exact():
    text = StudySchedule.from_dict(toy()).chapters_ffmetadata()
    assert text == ref_chapters(toy())
    assert "title=a\\=b\\;c\\#d\\\\e f" in text
    assert "START=4000\nEND=7000" in text and text.endswith("title=End\n")


@pytest.mark.parametrize(
    "start,seconds,demo",
    [(0, None, False), (0, None, True), (2, 5, False), (5, 5, True)],
)
def test_ass_exact(start, seconds, demo):
    s = StudySchedule.from_dict(toy())
    want = ref_build_ass(toy(), start, seconds, demo)
    assert s.to_ass(start, seconds, demo=demo) == want


def test_ass_injection_escaped_and_interval_checks():
    s = StudySchedule.from_dict(toy())
    text = s.to_ass()
    assert "Title （／b1）" in text and "{\\b1}" not in text
    assert "Demo {x}" not in s.to_ass(demo=True)
    for args in ((0, 0), (-1, 3), (0, 11), (9, 2), (0.5, 2)):
        with pytest.raises(ValueError):
            s.to_ass(*args)


@needs_fonts
def test_compact_ass_has_individual_backplates():
    text = StudySchedule.from_dict(toy()).to_ass(0, 3, compact=True)
    assert text.count("\\1a&HB2&") > 10
    assert "&H00000000,&H00000000,&H00FFFFFF" in text
    assert "\\1c&HF4EBD1&" not in text
    assert "m 0 0 l 338 0" not in text  # the big grouped panels are gone


# ---- native layer ----------------------------------------------------------- #


@needs_fonts
def test_overlay_layer_renders_and_ticks():
    schedule = StudySchedule.from_dict(toy())
    layer = StudyOverlayLayer(schedule, size=(640, 360))
    comp = Composition(size=(640, 360), fps=24, duration=10, transparent=True)
    comp.add_layer(layer)
    a = comp.render_buffer(1.0).rgba
    assert a.shape == (360, 640, 4)
    # Transparent outside every panel, translucent inside the timer panel.
    assert a[300:, :, 3].max() == 0
    assert 0.25 < a[138, 612, 3] < 0.35

    def clock(t):
        return comp.render_buffer(t).rgba[90:130, 459:620]

    assert not np.array_equal(clock(1.0), clock(2.0))
    assert np.array_equal(clock(1.0), clock(1.5))

    def tip(t):
        return comp.render_buffer(t).rgba[125:200, 20:440, 3].max()

    assert 0 < tip(0.25) < tip(1.0)
    assert tip(1.0) > 0.35 and tip(3.5) == 0
    assert layer.tip_state(0.0) == 0 and layer.tip_state(1.0) == 12
    assert layer.cache_key(1.2) == layer.cache_key(1.9)


@needs_fonts
def test_overlay_cache_reuse_and_missing_font(tmp_path):
    schedule = StudySchedule.from_dict(toy())
    layer = StudyOverlayLayer(schedule, size=(640, 360))
    layer.source_buffer(1.0)
    layer.source_buffer(1.4)
    assert layer.frames_rendered == 1
    with pytest.raises(FileNotFoundError):
        StudyOverlayLayer(schedule, fonts={"zh": str(tmp_path / "none.ttc")})
