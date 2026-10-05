"""Validate shared, data-only requests for cached motion effects."""

import copy
from pathlib import Path

from moviepy.ae.text.clusters import clusters
from moviepy.ae.three_d.scene import keys, number, vector


EFFECTS = ("captions", "title_card", "beat_particles", "transition")


def integer(value, label, low, high):
    """Require a finite integer in the stated inclusive range."""
    value = number(value, label, low, high)
    if value != int(value):
        raise ValueError(f"{label} must be integral")
    return int(value)


def _color(params, name, default):
    params[name] = vector(params.get(name, default), name, 3, 0, 1)


def _cues(request):
    previous_end = 0
    for cue in request["cues"]:
        keys(cue, "start_frame end_frame text unit_frames", "caption cue")
        cue["start_frame"] = integer(
            cue.get("start_frame", 0), "cue start_frame", 0, request["frame_count"] - 1
        )
        cue["end_frame"] = integer(
            cue.get("end_frame", request["frame_count"]),
            "cue end_frame",
            cue["start_frame"] + 1,
            request["frame_count"],
        )
        if cue["start_frame"] < previous_end:
            raise ValueError("cues must be ordered and may not overlap")
        previous_end = cue["end_frame"]
        text = cue.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 2048:
            raise ValueError("cue text must contain 1–2048 characters")
        text = text.replace("\r\n", "\n")
        if any(ord(char) < 32 and char != "\n" for char in text):
            raise ValueError(
                "caption text supports line breaks, not control characters"
            )
        cue["text"] = text
        if any(ord(char) > 255 for char in text) and "font" not in request["assets"]:
            raise ValueError("non-Latin captions require an explicit local font")
        units = [unit for unit in clusters(text.replace("\r\n", "\n")) if unit != "\n"]
        span = cue["end_frame"] - cue["start_frame"]
        denominator = len(units)
        if request["effect"] == "captions" and request["params"]["style"] == "pop":
            span = max(0, span - request["params"]["pop_frames"])
            denominator = max(1, len(units) - 1)
        starts = cue.setdefault(
            "unit_frames",
            [
                cue["start_frame"] + index * span // denominator
                for index in range(len(units))
            ],
        )
        if not isinstance(starts, list) or len(starts) != len(units):
            raise ValueError("unit_frames must match the non-newline grapheme units")
        cue["unit_frames"] = [
            integer(value, "unit frame", cue["start_frame"], cue["end_frame"] - 1)
            for value in starts
        ]
        if starts != sorted(starts):
            raise ValueError("unit_frames must be ordered")


def _caption_params(params, request):
    keys(
        params,
        "style font_size min_font_size safe_margin position max_lines line_spacing color accent stroke_color stroke_width pop_frames",
        "caption params",
    )
    params.setdefault("style", "highlight")
    if params["style"] not in ("highlight", "pop"):
        raise ValueError("caption style must be highlight or pop")
    params["font_size"] = integer(
        params.get("font_size", max(8, round(request["height"] * 0.09))),
        "font_size",
        8,
        512,
    )
    params["min_font_size"] = integer(
        params.get("min_font_size", max(8, params["font_size"] // 2)),
        "min_font_size",
        8,
        params["font_size"],
    )
    params["safe_margin"] = number(
        params.get("safe_margin", 0.08), "safe_margin", 0, 0.4
    )
    params["position"] = vector(params.get("position", [0.5, 0.8]), "position", 2, 0, 1)
    params["max_lines"] = integer(params.get("max_lines", 2), "max_lines", 1, 6)
    params["line_spacing"] = number(
        params.get("line_spacing", 1.25), "line_spacing", 1, 2
    )
    params["stroke_width"] = integer(
        params.get("stroke_width", 2), "stroke_width", 0, 16
    )
    params["pop_frames"] = integer(
        params.get("pop_frames", max(1, round(request["fps"] * 0.18))),
        "pop_frames",
        1,
        240,
    )
    for name, default in (
        ("color", [0.92, 0.96, 1]),
        ("accent", [0.14, 0.9, 1]),
        ("stroke_color", [0.02, 0.03, 0.07]),
    ):
        _color(params, name, default)


def _title_params(params, request):
    keys(params, "color depth turn_degrees safe_margin", "title card params")
    _color(params, "color", [0.15, 0.65, 0.95])
    params["depth"] = number(params.get("depth", 0.16), "depth", 0.01, 0.5)
    params["turn_degrees"] = number(
        params.get("turn_degrees", 16), "turn_degrees", 0, 30
    )
    params["safe_margin"] = number(
        params.get("safe_margin", 0.12), "safe_margin", 0.05, 0.4
    )
    if (
        len(request["cues"]) != 1
        or request["cues"][0]["start_frame"] != 0
        or request["cues"][0]["end_frame"] != request["frame_count"]
    ):
        raise ValueError("title_card requires one cue spanning the entire effect")


def _particle_params(params, request):
    keys(
        params,
        "count lifetime speed spread gravity size color location",
        "particle params",
    )
    params["count"] = integer(params.get("count", 180), "count", 1, 5000)
    params["lifetime"] = number(params.get("lifetime", 0.7), "lifetime", 0.01, 30)
    params["speed"] = vector(params.get("speed", [1, 3]), "speed", 2, 0, 100)
    if params["speed"][0] > params["speed"][1]:
        raise ValueError("speed bounds are reversed")
    params["spread"] = number(params.get("spread", 100), "spread", 0, 180)
    params["gravity"] = vector(params.get("gravity", [0, -1.5, 0]), "gravity")
    params["location"] = vector(params.get("location", [0, -0.6, 0]), "location")
    params["size"] = number(params.get("size", 0.035), "size", 0.001, 1)
    _color(params, "color", [0.1, 0.75, 1])
    if len(request["audio_events"]) * params["count"] > 200000:
        raise ValueError("particle request exceeds 200000 allocated particles")


def _transition_params(params, request):
    keys(params, "style direction softness center progress", "transition params")
    params.setdefault("style", "wipe")
    if params["style"] not in ("wipe", "iris"):
        raise ValueError("transition style must be wipe or iris")
    params.setdefault(
        "direction", "out" if params["style"] == "iris" else "left_to_right"
    )
    if params["direction"] not in (
        "left_to_right",
        "right_to_left",
        "top_to_bottom",
        "bottom_to_top",
        "out",
        "in",
    ):
        raise ValueError("unsupported transition direction")
    if (params["style"] == "iris") != (params["direction"] in ("out", "in")):
        raise ValueError("iris uses out/in; wipe uses a horizontal/vertical direction")
    params["softness"] = number(params.get("softness", 0.035), "softness", 0, 0.5)
    params["center"] = vector(params.get("center", [0.5, 0.5]), "center", 2, 0, 1)
    points = params.setdefault(
        "progress", [[0, 0], [max(1, request["frame_count"] - 1), 1]]
    )
    if request["frame_count"] < 2 or not isinstance(points, list) or len(points) < 2:
        raise ValueError("a transition needs at least two frames and progress points")
    previous = -1
    for point_index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("progress points must be [frame, value]")
        index = integer(point[0], "progress frame", 0, request["frame_count"] - 1)
        progress = number(point[1], "progress", 0, 1)
        if index <= previous:
            raise ValueError("progress frames must be strictly increasing")
        previous = index
        points[point_index] = [index, progress]
    if points[0][0] != 0 or points[-1][0] != request["frame_count"] - 1:
        raise ValueError("progress must cover the first and last frames")


def validate_motion_request(value):
    """Normalize a request with exact integer frame timing and local assets."""
    keys(
        value,
        "effect params width height fps frame_count seed cues audio_events assets",
        "motion request",
    )
    request = copy.deepcopy(value)
    if request.get("effect") not in EFFECTS:
        raise ValueError(f"effect must be one of {EFFECTS}")
    for name, default, low, high in (
        ("width", 640, 16, 4096),
        ("height", 360, 16, 4096),
        ("fps", 24, 1, 120),
        ("frame_count", 48, 1, 100000),
        ("seed", 0, 0, 2147483647),
    ):
        request[name] = integer(request.get(name, default), name, low, high)
    if 48000 % request["fps"] or request["frame_count"] / request["fps"] > 3600:
        raise ValueError(
            "fps must divide 48000 and duration may not exceed 3600 seconds"
        )
    assets = request.setdefault("assets", {})
    keys(assets, "font audio", "assets")
    for name, filename in assets.items():
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        assets[name] = path.as_posix()
    for name in ("cues", "audio_events"):
        request.setdefault(name, [])
        if not isinstance(request[name], list) or len(request[name]) > 1000:
            raise ValueError(f"{name} must be a list of at most 1000 items")
    params = request.setdefault("params", {})
    if request["effect"] == "captions":
        _caption_params(params, request)
    _cues(request)
    frames = set()
    for event in request["audio_events"]:
        keys(event, "frame strength", "audio event")
        event["frame"] = integer(
            event.get("frame"), "event frame", 0, request["frame_count"] - 1
        )
        event["strength"] = number(event.get("strength", 1), "strength", 1e-9, 1)
        if event["frame"] in frames:
            raise ValueError("audio events may not share a frame")
        frames.add(event["frame"])
    request["audio_events"].sort(key=lambda event: event["frame"])
    effect = request["effect"]
    if effect in ("captions", "title_card") and not request["cues"]:
        raise ValueError("caption effects require cues")
    if effect not in ("captions", "title_card") and request["cues"]:
        raise ValueError("cues only apply to captions and title_card")
    if effect != "beat_particles" and request["audio_events"]:
        raise ValueError("audio_events only apply to beat_particles")
    if effect == "beat_particles" and not request["audio_events"]:
        raise ValueError("beat_particles requires analyzed or explicit audio_events")
    {
        "captions": _caption_params,
        "title_card": _title_params,
        "beat_particles": _particle_params,
        "transition": _transition_params,
    }[effect](params, request)
    return request
