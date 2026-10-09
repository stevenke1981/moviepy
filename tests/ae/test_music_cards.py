"""Track cards, progress ring and session timeline as ASS graphics."""

import math
import os
import subprocess
import time

import numpy as np

import pytest

from moviepy.ae.templates.music_cards import (
    SPECTRUM_RECT,
    TIMELINE_RECT,
    AssDocument,
    arc_beziers,
    arc_drawing,
    combine_study_ass,
    progress_ring,
    session_timeline,
    sleep_cards_ass,
    timeline_segments,
    track_cards,
)
from moviepy.ae.templates.study import StudySchedule
from moviepy.config import FFMPEG_BINARY


FONTS = ["msjh.ttc", "YuGothM.ttc", "consola.ttf"]
HAVE_FONTS = all(os.path.isfile("C:/Windows/Fonts/" + f) for f in FONTS)
needs_fonts = pytest.mark.skipif(not HAVE_FONTS, reason="Windows fonts missing")


def phase(pid, kind, a, b):
    data = {"id": pid, "type": kind, "start": a, "end": b}
    for key in ("label", "label_en", "label_ja", "tip", "tip_en", "tip_ja"):
        data[key] = f"{key}-{pid}"
    return data


def make_schedule(phases, **extra):
    duration = phases[-1]["end"]
    focus = sum(p["end"] - p["start"] for p in phases if p["type"] == "focus")
    config = {
        "episode": "Toy",
        "duration_seconds": duration,
        "focus_seconds": focus,
        "rest_seconds": duration - focus,
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
        "title": {"zh": "標題", "en": "Title", "ja": "題"},
        "phases": phases,
        "study_chapters": [{"start": 0, "name": "One"}],
    }
    config.update(extra)
    return StudySchedule.from_dict(config)


@pytest.fixture(scope="module")
def toy():
    return make_schedule(
        [
            phase("F1", "focus", 0, 4),
            phase("B1", "break", 4, 7),
            phase("C1", "closing", 7, 10),
        ]
    )


def burn_frame(ass_path, frame, rate=2, color="black", size="1280x720"):
    """Return one decoded RGB frame (720, 1280, 3) of the burned colour source."""
    command = [
        FFMPEG_BINARY,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:r={rate}:d=12",
        "-vf",
        f"ass=filename={ass_path.name},select=eq(n\\,{frame})",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    done = subprocess.run(
        command, cwd=str(ass_path.parent), capture_output=True, check=False
    )
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    return np.frombuffer(done.stdout, np.uint8).reshape(720, 1280, 3)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("sweep", [0.3, math.pi / 2, 3.0, 2 * math.pi, -2.0])
def test_bezier_points_lie_on_circle(sweep):
    centre, radius = (200.0, 150.0), 40.0
    segments = arc_beziers(centre, radius, -math.pi / 2, sweep)
    assert len(segments) == max(1, math.ceil(abs(sweep) / (math.pi / 2) - 1e-9))
    for p0, p1, p2, p3 in segments:
        assert math.dist(p0, centre) == pytest.approx(radius, abs=1e-6)
        assert math.dist(p3, centre) == pytest.approx(radius, abs=1e-6)
        for t in np.linspace(0, 1, 21):
            x = sum(
                c * p[0]
                for c, p in zip(
                    ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3),
                    (p0, p1, p2, p3),
                )
            )
            y = sum(
                c * p[1]
                for c, p in zip(
                    ((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t**2 * (1 - t), t**3),
                    (p0, p1, p2, p3),
                )
            )
            assert abs(math.dist((x, y), centre) - radius) < 0.5


def test_sweep_zero_draws_nothing_and_one_is_full_annulus():
    assert arc_beziers((0, 0), 10, 0, 0.0) == []
    assert arc_drawing((100, 100), 30, 6, 0.0) == ""
    assert arc_drawing((100, 100), 30, 6, -1.0) == ""
    full = arc_drawing((100, 100), 30, 6, 1.0)
    assert full.count("m ") == 2  # outer and inner contour
    assert full.count(" b ") == 2
    assert full == arc_drawing((100, 100), 30, 6, 2.0)
    half = arc_drawing((100, 100), 30, 6, 0.5)
    assert half.count("m ") == 1 and " l " in half


def test_drawing_coordinates_are_on_the_two_radii():
    centre = (300.0, 200.0)
    text = arc_drawing(centre, 34, 7, 1.0)
    outer, inner = text.split(" m ")
    for chunk, radius in ((outer, 37.5), (inner, 30.5)):
        numbers = [int(v) / 16 for v in chunk.replace("m", "").replace("b", "").split()]
        points = list(zip(numbers[0::2], numbers[1::2]))
        # on-curve points are the first and every third after it
        on_curve = points[0::3]
        for p in on_curve:
            assert abs(math.dist(p, centre) - radius) < 0.5


# --------------------------------------------------------------------------- #
# Document / merge
# --------------------------------------------------------------------------- #


def event_lines(text):
    return [
        line for line in text.splitlines() if line.startswith(("Dialogue:", "Comment:"))
    ]


def test_merge_keeps_s14_events_and_styles(toy):
    base = toy.to_ass()
    merged = combine_study_ass(toy, cards=None)
    base_events = event_lines(base)
    merged_events = event_lines(merged)
    for line in base_events:
        assert merged_events.count(line) == base_events.count(line)
    base_styles = [x for x in base.splitlines() if x.startswith("Style:")]
    styles = [x for x in merged.splitlines() if x.startswith("Style:")]
    names = [x[7:].split(",")[0] for x in styles]
    assert len(names) == len(set(names))
    assert set(base_styles) <= set(styles)
    assert {"Vec", "NextLabel"} <= set(names)
    assert len(merged_events) > len(base_events)
    starts = [line.split(",")[1] for line in merged_events]
    assert starts == sorted(starts)
    assert "PlayResX: 1280" in merged and "PlayResY: 720" in merged


def test_merge_rejects_conflicts_and_other_playres():
    a, b = AssDocument(), AssDocument()
    a.add_style("X", "Arial", 20)
    b.add_style("X", "Arial", 21)
    with pytest.raises(ValueError, match="Conflicting"):
        a.merge(b.to_text())
    with pytest.raises(ValueError, match="PlayRes"):
        a.merge(AssDocument((640, 360)).to_text())
    with pytest.raises(ValueError, match="Unknown style"):
        a.add_event(0, 0, 1, "Nope", "t")
    with pytest.raises(ValueError):
        a.add_event(0, 2, 2, "X", "t")


def test_merge_is_idempotent_for_identical_styles():
    a = AssDocument()
    a.add_style("X", "Arial", 20)
    a.add_event(0, 1, 2, "X", "t")
    again = AssDocument.from_ass(a.to_text())
    again.merge(a.to_text())
    assert again.style_names == ["X"]
    assert len(again.events) == 2


# --------------------------------------------------------------------------- #
# Progress ring
# --------------------------------------------------------------------------- #


def test_ring_event_timing_is_exact(toy):
    doc = progress_ring(toy)
    events = doc.events
    # 3 track rings + arcs for seconds 1,2,3 | 5,6 | 8,9 (second 0 of a phase is empty)
    assert len(events) == 3 + 3 + 2 + 2
    prefixes = [e.split(",,")[0] for e in events]
    assert "Dialogue: 1,0:00:00.00,0:00:04.00,Vec" in prefixes
    assert "Dialogue: 1,0:00:04.00,0:00:07.00,Vec" in prefixes
    for second in (1, 2, 3, 5, 6, 8, 9):
        assert (
            f"Dialogue: 1,0:00:{second:02}.00,0:00:{second + 1:02}.00,Vec" in prefixes
        )
    assert not any(p.startswith("Dialogue: 1,0:00:00.00,0:00:01.00") for p in prefixes)
    # sweep grows with elapsed fraction: more drawing data later in a phase
    arcs = [e for e in events if "0:00:01.00,0:00:02.00" in e or "0:00:03.00" in e]
    assert len(arcs[0]) < len(arcs[-1])


def test_ring_uses_phase_colours_and_stays_inside_timer_panel(toy):
    text = progress_ring(toy).to_text()
    assert r"\1c&HC4D5A8&" in text  # focus sage (BGR)
    assert r"\1c&HD1EBF4&" in text  # break cream, as the S14 rest label
    # every coordinate (1/16 px) lies inside the timer panel (900,38,338,247)
    for line in event_lines(text):
        body = line.split("}", 1)[1].replace("{", "")
        numbers = [int(v) for v in body.split() if v.lstrip("-").isdigit()]
        xs, ys = numbers[0::2], numbers[1::2]
        assert min(xs) / 16 >= 900 and max(xs) / 16 <= 1238
        assert min(ys) / 16 >= 38 and max(ys) / 16 <= 285


def test_ring_rejects_clock_and_panel_collisions(toy):
    with pytest.raises(ValueError, match="clock"):
        progress_ring(toy, center=(1000, 234))
    with pytest.raises(ValueError, match="inside the timer panel"):
        progress_ring(toy, center=(1230, 234))
    with pytest.raises(ValueError):
        progress_ring(toy, step=0)


def test_event_count_and_size_of_the_87_minute_schedule():
    rows = [
        ("F1", "focus", 0, 1500),
        ("B1", "break", 1500, 1800),
        ("F2", "focus", 1800, 3300),
        ("B2", "break", 3300, 3600),
        ("F3", "focus", 3600, 5100),
        ("C1", "closing", 5100, 5220),
    ]
    schedule = make_schedule([phase(*r) for r in rows])
    started = time.perf_counter()
    doc = progress_ring(schedule)
    text = doc.to_text()
    elapsed = time.perf_counter() - started
    assert len(doc.events) == 5220 - 6 + 6
    assert len(text.encode("utf-8")) < 2_500_000
    assert elapsed < 5


# --------------------------------------------------------------------------- #
# Timeline
# --------------------------------------------------------------------------- #


def test_timeline_segment_proportions(toy):
    x, _, width, _ = TIMELINE_RECT
    segments = timeline_segments(toy, TIMELINE_RECT, gap=1.0)
    widths = [x1 - x0 for _, _, x0, x1 in segments]
    assert widths[0] == pytest.approx(width * 0.4 - 1)
    assert widths[1] == pytest.approx(width * 0.3 - 1)
    assert widths[2] == pytest.approx(width * 0.3)
    assert segments[0][2] == x and segments[-1][3] == pytest.approx(x + width)


def test_timeline_events(toy):
    doc = session_timeline(toy, playhead_step=2)
    events = doc.events
    bars = [e for e in events if "0:00:00.00,0:00:10.00,Vec" in e]
    assert len(bars) == 3  # one static event per phase type
    heads = [e for e in events if "\\move(" in e]
    assert len(heads) == 5
    assert heads[0].startswith("Dialogue: 2,0:00:00.00,0:00:02.00,Vec")
    assert heads[-1].startswith("Dialogue: 2,0:00:08.00,0:00:10.00,Vec")
    # the playhead of the last event ends where the bar ends
    end_x = TIMELINE_RECT[0] + TIMELINE_RECT[2] - 1.5
    assert f"\\move({end_x - TIMELINE_RECT[2] * 0.2:g}," in heads[-1].replace(
        ".0,", ","
    )
    labels = [e for e in events if ",NextLabel," in e]
    assert len(labels) == 2
    assert labels[0].startswith("Dialogue: 2,0:00:00.00,0:00:04.00,NextLabel")
    assert "label_en-B1" in labels[0] and "label_ja-C1" in labels[1]
    assert "\\fad(1500,1500)" in labels[0]
    assert not [
        e for e in session_timeline(toy, next_seconds=0).events if "NextLabel," in e
    ]


def test_next_label_only_in_last_seconds():
    schedule = make_schedule(
        [phase("F1", "focus", 0, 300), phase("B1", "break", 300, 400)]
    )
    labels = [e for e in session_timeline(schedule).events if ",NextLabel," in e]
    assert len(labels) == 1
    assert labels[0].startswith("Dialogue: 2,0:04:00.00,0:05:00.00,NextLabel")


# --------------------------------------------------------------------------- #
# Track cards
# --------------------------------------------------------------------------- #

CUES = {
    "total_seconds": 100.0,
    "chapters": [
        {"id": "A", "global_start_seconds": 0.0, "crossfade_center_seconds": 0.0},
        {"id": "B", "global_start_seconds": 40.0, "crossfade_center_seconds": 46.0},
    ],
}
TITLES = {
    "A": {"zh": "雨聲", "en": "Rain", "ja": "雨音"},
    "B": {"zh": "森林", "en": "Forest", "ja": "森"},
}


def test_card_timing_fades_and_clipping():
    doc = track_cards(CUES, titles=TITLES, intro_seconds=0)
    events = doc.events
    panels = [e for e in events if ",CardPanel," in e]
    assert panels[0].startswith("Dialogue: 0,0:00:03.00,0:00:14.50,CardPanel")
    assert panels[1].startswith("Dialogue: 0,0:00:49.00,0:01:00.50,CardPanel")
    assert all("\\fad(1500,2000)" in e for e in events)
    assert len([e for e in events if ",CardZH," in e]) == 2
    short = track_cards(CUES, titles=TITLES, hold=100, intro_seconds=0).events
    first = [e for e in short if ",CardPanel," in e][0]
    assert ",0:00:49.00,CardPanel" in first  # clipped at the next card
    last = [e for e in short if ",CardPanel," in e][1]
    assert ",0:01:40.00,CardPanel" in last  # clipped at total_seconds


def test_intro_guard_and_skip_first():
    default = track_cards(CUES, titles=TITLES)  # offset 3 < intro 0 never skips
    assert len([e for e in default.events if ",CardPanel," in e]) == 2
    guarded = track_cards(CUES, titles=TITLES, intro_seconds=10)
    assert len([e for e in guarded.events if ",CardPanel," in e]) == 1
    skipped = track_cards(CUES, titles=TITLES, skip_first=True)
    assert len([e for e in skipped.events if ",CardPanel," in e]) == 1


def test_cards_reject_reserved_overlap_and_bad_input():
    with pytest.raises(ValueError, match="overlaps reserved"):
        track_cards(CUES, titles=TITLES, reserved=[(0, 0, 1280, 720)])
    with pytest.raises(ValueError, match="overlaps reserved"):
        track_cards(
            CUES,
            titles=TITLES,
            position="upper_left",
            reserved=[(40, 38, 700, 148)],
        )
    with pytest.raises(ValueError, match="overlaps reserved"):
        # lower cards collide with the spectrum when the card is made taller
        track_cards(
            CUES,
            titles=TITLES,
            style={"sizes": (60, 50, 50)},
            reserved=[SPECTRUM_RECT, (40, 244, 838, 164)],
        )
    with pytest.raises(ValueError, match="wide"):
        track_cards(CUES, titles={"A": {"en": "x" * 80}, "B": TITLES["B"]})
    with pytest.raises(ValueError, match="No title"):
        track_cards(CUES, titles={"A": TITLES["A"]})
    with pytest.raises(ValueError, match="position"):
        track_cards(CUES, titles=TITLES, position="middle")
    with pytest.raises(ValueError, match="style"):
        track_cards(CUES, titles=TITLES, style={"colour": 1})


def test_default_card_clears_all_study_regions_and_timeline():
    schedule = make_schedule(
        [phase("F1", "focus", 0, 60), phase("B1", "break", 60, 90)]
    )
    text = combine_study_ass(schedule, cards={"cues": CUES, "titles": TITLES})
    assert ",CardPanel," in text and ",NextLabel," in text
    with pytest.raises(ValueError, match="overlaps reserved"):
        combine_study_ass(
            schedule,
            cards={"cues": CUES, "titles": TITLES, "position": "upper_left"},
        )


def test_ass_escaping_of_injected_braces_and_backslashes():
    titles = {
        "A": {"zh": "a{\\pos(0,0)}b", "en": "c\\Nd{x}", "ja": "e"},
        "B": TITLES["B"],
    }
    text = sleep_cards_ass(CUES, titles, intro_seconds=0)
    body = [e for e in event_lines(text) if ",CardZH," in e][0]
    assert body.count("{") == 1  # only our own override block
    assert "a（／pos(0,0)）b" in body
    en = [e for e in event_lines(text) if ",CardEN," in e][0]
    assert "c／Nd（x）" in en
    schedule = make_schedule(
        [phase("F1", "focus", 0, 5), phase("B1", "break", 5, 8)],
    )
    schedule._config["phases"][1]["label_en"] = "x{\\fs99}"
    label = [e for e in session_timeline(schedule).events if ",NextLabel," in e][0]
    assert "\\fs99" not in label


def test_sleep_cards_have_only_card_styles():
    text = sleep_cards_ass(CUES, TITLES)
    names = [x[7:].split(",")[0] for x in text.splitlines() if x.startswith("Style:")]
    assert names == ["Vec", "CardPanel", "CardZH", "CardEN", "CardJA"]


# --------------------------------------------------------------------------- #
# FFmpeg / libass
# --------------------------------------------------------------------------- #


def vector_only(text):
    """Return the ASS with every text (non-drawing) event removed."""
    keep = [
        line
        for line in text.splitlines()
        if not line.startswith("Dialogue:") or "\\p5" in line
    ]
    return "\n".join(keep) + "\n"


def test_ring_and_timeline_pixels_in_a_decoded_frame(toy, tmp_path):
    doc = AssDocument()
    doc.merge(progress_ring(toy).to_text())
    doc.merge(session_timeline(toy, next_seconds=0, playhead_step=10).to_text())
    path = tmp_path / "v.ass"
    path.write_text(doc.to_text(), encoding="utf-8")
    frame = burn_frame(path, 5)  # t = 2.5 s, second 2 of F1: fraction 0.5
    cx, cy = 1186, 234
    colour = np.array([0xA8, 0xD5, 0xC4], float)
    right = frame[cy, cx + 34].astype(float)  # inside the swept half
    left = frame[cy, cx - 34].astype(float)  # only the faint track
    assert np.allclose(right, colour * (1 - 0x30 / 255), atol=14)
    assert np.allclose(left, colour * (1 - 0xC8 / 255), atol=14)
    assert frame[cy, cx].sum() == 0  # open middle
    assert frame[cy, cx + 60].sum() == 0 and frame[cy + 60, cx].sum() == 0
    assert frame[cy - 34, cx].astype(float)[1] > 150  # 12 o'clock: arc start
    # timeline: segment colours and playhead (move 0 -> 10 s, at 25 %)
    y = TIMELINE_RECT[1] + 3
    x0, width = TIMELINE_RECT[0], TIMELINE_RECT[2]
    focus_px = frame[y, int(x0 + width * 0.1)].astype(float)
    break_px = frame[y, int(x0 + width * 0.55)].astype(float)
    scale = 1 - 0x70 / 255
    assert np.allclose(focus_px, [0xA8 * scale, 0xD5 * scale, 0xC4 * scale], atol=14)
    assert np.allclose(break_px, [0xF4 * scale, 0xEB * scale, 0xD1 * scale], atol=14)
    head_x = int(x0 + width * 0.25)
    assert frame[y, head_x].astype(float).min() > 150
    assert frame[y, head_x + 40].astype(float).max() < 140
    assert frame[y - 6, int(x0 + width * 0.1)].sum() == 0  # above the bar: empty
    assert frame[y + 40, int(x0 + width * 0.1)].sum() == 0


def test_card_backplate_darkens_the_expected_rectangle(tmp_path):
    text = track_cards(CUES, titles=TITLES, intro_seconds=0).to_text()
    path = tmp_path / "c.ass"
    path.write_text(vector_only(text), encoding="utf-8")
    frame = burn_frame(path, 12, color="0x808080")  # t = 6 s: fully faded in
    assert (
        frame[600, 300].tolist() == [128, 128, 128]
        or abs(int(frame[600, 300][0]) - 128) < 3
    )
    inside = frame[470, 50:60].astype(int)  # card spans y 420..540 from x 40
    assert (inside[:, 0] < 122).all()
    assert abs(int(frame[470, 900, 0]) - 128) < 3
    spectrum = SPECTRUM_RECT
    assert abs(int(frame[spectrum[1] + 5, spectrum[0] + 5, 0]) - 128) < 3


@needs_fonts
def test_combined_ass_burns_without_error(toy, tmp_path):
    schedule = make_schedule([phase("F1", "focus", 0, 8), phase("B1", "break", 8, 12)])
    text = combine_study_ass(schedule, cards={"cues": CUES, "titles": TITLES})
    path = tmp_path / "all.ass"
    path.write_text(text, encoding="utf-8")
    frame = burn_frame(path, 10, rate=2)  # t = 5 s: card fully visible
    assert frame.shape == (720, 1280, 3)
    assert frame[234, 1186 + 34].max() > 100  # ring arc present
    assert frame[470, 50].max() > 0  # card backplate / text region


def test_libass_speed_of_the_87_minute_overlay_window(tmp_path):
    rows = [
        ("F1", "focus", 0, 1500),
        ("B1", "break", 1500, 1800),
        ("F2", "focus", 1800, 3300),
        ("B2", "break", 3300, 3600),
        ("F3", "focus", 3600, 5100),
        ("C1", "closing", 5100, 5220),
    ]
    schedule = make_schedule([phase(*r) for r in rows])
    doc = AssDocument()
    doc.merge(progress_ring(schedule).to_text())
    doc.merge(session_timeline(schedule, next_seconds=0).to_text())
    path = tmp_path / "long.ass"
    path.write_text(doc.to_text(), encoding="utf-8")
    command = [
        FFMPEG_BINARY,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1280x720:r=24:d=5",
        "-vf",
        "setpts=PTS+1498/TB,ass=filename=long.ass",
        "-f",
        "null",
        "-",
    ]
    done = subprocess.run(command, cwd=str(tmp_path), capture_output=True, check=False)
    assert done.returncode == 0, done.stderr.decode(errors="replace")
