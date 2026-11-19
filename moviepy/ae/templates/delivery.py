r"""Final packaging: mux video, narration, subtitle tracks and chapters into one MP4.

``mux_delivery`` wraps the one ffmpeg call the 武則天 r2b film used by hand: the
video stream is copied, audio is encoded to AAC, every SRT becomes a labelled
``mov_text`` track and chapters travel as FFMETADATA. The file is written under
a temporary name and renamed, and an existing deliverable is never replaced
unless asked.

Examples
--------
>>> from moviepy.ae.templates.delivery import chapters_ffmetadata
>>> print(chapters_ffmetadata([(0, 1.5, "A=B")]), end="")
;FFMETADATA1
[CHAPTER]
TIMEBASE=1/1000
START=0
END=1500
title=A\=B
"""

import os
import re
import subprocess
from pathlib import Path


DEFAULT_SUBTITLE_TITLES = {
    "zh-TW": "繁體中文（臺灣）",
    "zh-CN": "简体中文",
    "en": "English",
}

_LANGUAGE_CODES = {
    "zh": "zho",
    "zh-tw": "zho",
    "zh-cn": "zho",
    "zh-hk": "zho",
    "zh-hant": "zho",
    "zh-hans": "zho",
    "en": "eng",
    "ja": "jpn",
    "ko": "kor",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "pt": "por",
    "ru": "rus",
    "it": "ita",
    "vi": "vie",
    "th": "tha",
    "id": "ind",
    "ar": "ara",
}


def iso639_2(lang):
    """Return the ISO 639-2 code ffmpeg expects for a language tag.

    Parameters
    ----------
    lang : str
        ``"zh-TW"``, ``"zh-CN"``, ``"en"``, ``"ja"`` or an already three-letter
        code, which passes through lower-cased.

    Returns
    -------
    str
        Three-letter code.

    Raises
    ------
    ValueError
        For a tag that is neither known nor three letters.

    Examples
    --------
    >>> iso639_2("zh-TW"), iso639_2("en"), iso639_2("JPN"), iso639_2("ja")
    ('zho', 'eng', 'jpn', 'jpn')
    """
    key = str(lang).strip().replace("_", "-").lower()
    if key in _LANGUAGE_CODES:
        return _LANGUAGE_CODES[key]
    if re.fullmatch(r"[a-z]{3}", key):
        return key
    raise ValueError(f"unknown language tag {lang!r}")


def _escape_meta(text):
    out = str(text).replace("\\", "\\\\")
    for ch in "=;#":
        out = out.replace(ch, "\\" + ch)
    return out.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\\n")


def _chapter_fields(chapter):
    if isinstance(chapter, dict):
        return chapter["start"], chapter["end"], chapter["title"]
    start, end, title = chapter
    return start, end, title


def chapters_ffmetadata(chapters):
    r"""Render chapters as FFMETADATA1 text, escaping ``= ; # \`` and newlines.

    Parameters
    ----------
    chapters : iterable
        ``(start, end, title)`` tuples or dicts with those keys; times in
        seconds (millisecond precision).

    Returns
    -------
    str
        Text for ``ffmpeg -i file.ffmetadata -map_chapters N``.

    Raises
    ------
    ValueError
        If a chapter has a negative start, ``end <= start``, or an empty title.

    Examples
    --------
    >>> text = chapters_ffmetadata([{"start": 0, "end": 2, "title": "Intro; #1"}])
    >>> text.splitlines()[-1]
    'title=Intro\\; \\#1'
    """
    parts = [";FFMETADATA1\n"]
    for chapter in chapters:
        start, end, title = _chapter_fields(chapter)
        start_ms, end_ms = round(float(start) * 1000), round(float(end) * 1000)
        if start_ms < 0 or end_ms <= start_ms:
            raise ValueError(f"chapter {title!r} needs 0 <= start < end")
        if not str(title).strip():
            raise ValueError("chapter title must not be empty")
        parts.append(
            f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_ms}\nEND={end_ms}\n"
            f"title={_escape_meta(title)}\n"
        )
    return "".join(parts)


def find_ffmpeg(ffmpeg=None):
    """Resolve the ffmpeg executable: argument, ``FFMPEG_BINARY``, imageio-ffmpeg.

    Parameters
    ----------
    ffmpeg : str or os.PathLike, optional
        Explicit path.

    Returns
    -------
    str
        Executable path or name.
    """
    if ffmpeg:
        return os.fspath(ffmpeg)
    env = os.environ.get("FFMPEG_BINARY")
    if env and env != "auto-detect":
        return env
    from moviepy.config import FFMPEG_BINARY

    return FFMPEG_BINARY


_STREAM = re.compile(r"^\s*Stream #0:(\d+)(?:\[[^\]]*\])?(?:\((\w+)\))?: (\w+): (\w+)")
_CHAPTER = re.compile(r"^\s*Chapter #0:\d+")
_DURATION = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_delivery(path, ffmpeg=None):
    """Summarize a media file by parsing ``ffmpeg -i`` (no ffprobe needed).

    Parameters
    ----------
    path : str or os.PathLike
        File to inspect.
    ffmpeg : str, optional
        ffmpeg executable.

    Returns
    -------
    dict
        ``streams`` (list of ``{index, type, codec, language, title}``),
        ``chapters`` (count) and ``duration`` (seconds, or ``None``).

    Examples
    --------
    >>> probe_delivery("missing-file.mp4")["streams"]
    []
    """
    done = subprocess.run(
        [find_ffmpeg(ffmpeg), "-hide_banner", "-i", os.fspath(path)],
        capture_output=True,
        check=False,
    )
    lines = (done.stderr or b"").decode("utf-8", "replace").splitlines()
    streams, chapters, duration = [], 0, None
    current, in_meta = None, False
    for line in lines:
        m = _STREAM.match(line)
        if m:
            current = {
                "index": int(m.group(1)),
                "type": m.group(3).lower(),
                "codec": m.group(4),
                "language": m.group(2),
                "title": None,
            }
            streams.append(current)
            in_meta = False
            continue
        if _CHAPTER.match(line):
            chapters += 1
            current, in_meta = None, False
            continue
        dm = _DURATION.search(line)
        if dm and duration is None:
            h, mi, sec = dm.groups()
            duration = int(h) * 3600 + int(mi) * 60 + float(sec)
        if current is not None:
            if line.strip() == "Metadata:":
                in_meta = True
            elif in_meta and line.strip().startswith("title"):
                current["title"] = line.split(":", 1)[1].strip()
            elif in_meta and line.strip().startswith("handler_name"):
                name = line.split(":", 1)[1].strip()
                if current["title"] is None and not name.endswith("Handler"):
                    current["title"] = name
    return {"streams": streams, "chapters": chapters, "duration": duration}


def mux_delivery(
    video,
    output,
    *,
    audio=None,
    subtitles=None,
    chapters=None,
    overwrite=False,
    ffmpeg=None,
    default_subtitle=None,
    subtitle_titles=None,
    audio_codec="aac",
    audio_bitrate="256k",
    duration=None,
):
    """Mux video, narration, SRT tracks and chapters into a faststart MP4.

    Parameters
    ----------
    video : str or os.PathLike
        Rendered video; its stream is copied untouched.
    output : str or os.PathLike
        Destination ``.mp4``. Written as ``<stem>.muxing.mp4`` first, then renamed.
    audio : str or os.PathLike, optional
        Narration or master mix, encoded to AAC (``audio_codec="aac"``) at
        ``audio_bitrate`` or copied (``audio_codec="copy"``). Without it the
        video's own audio, if any, is kept.
    subtitles : mapping, optional
        Ordered ``language -> SRT path``; each becomes a ``mov_text`` track whose
        language is the ISO 639-2 code and whose title comes from
        ``subtitle_titles`` (defaults for ``zh-TW``, ``zh-CN`` and ``en``).
    chapters : iterable, optional
        Chapters for ``chapters_ffmetadata``.
    overwrite : bool
        Replace an existing ``output``. Default refuses with ``FileExistsError``.
    ffmpeg : str, optional
        Executable; else ``FFMPEG_BINARY``, else the one bundled by imageio-ffmpeg.
    default_subtitle : str, optional
        Language key of the track flagged default; all others are not.
    subtitle_titles : mapping, optional
        Overrides of the track titles by language key.
    audio_codec : str
        ``"aac"`` or ``"copy"``.
    audio_bitrate : str
        AAC bitrate.
    duration : float, optional
        Cap on the output length in seconds (``-t``).

    Returns
    -------
    dict
        ``output``, ``streams`` (type, codec, language, title per stream),
        ``chapters`` (count), ``duration`` and ``probe`` (``"ffmpeg"`` or
        ``"unavailable"``).

    Raises
    ------
    FileExistsError
        If ``output`` exists and ``overwrite`` is false, or a stale temporary exists.
    FileNotFoundError
        For a missing input.
    ValueError
        For an unknown ``default_subtitle``, ``audio_codec`` or language tag.
    RuntimeError
        If ffmpeg fails; the message carries its stderr tail.
    """
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass overwrite=True to replace it")
    if audio_codec not in ("aac", "copy"):
        raise ValueError("audio_codec must be 'aac' or 'copy'")
    subs = dict(subtitles or {})
    if default_subtitle is not None and default_subtitle not in subs:
        raise ValueError(f"default_subtitle {default_subtitle!r} is not in subtitles")
    titles = {**DEFAULT_SUBTITLE_TITLES, **dict(subtitle_titles or {})}
    codes = {lang: iso639_2(lang) for lang in subs}
    inputs = [Path(video)] + ([Path(audio)] if audio else [])
    inputs += [Path(p) for p in subs.values()]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    chapter_text = chapters_ffmetadata(chapters) if chapters else None
    suffix = output.suffix or ".mp4"
    pending = output.with_name(output.stem + ".muxing" + suffix)
    meta_path = output.with_name(output.stem + ".chapters.ffmetadata")
    if pending.exists():
        raise FileExistsError(f"stale temporary {pending}; remove it first")
    exe = find_ffmpeg(ffmpeg)

    args = [exe, "-hide_banner", "-y"]
    for path in inputs:
        args += ["-i", os.fspath(path)]
    meta_index = len(inputs)
    if chapter_text is not None:
        args += ["-i", os.fspath(meta_path)]
    args += ["-map", "0:v:0"]
    first_sub = 1
    if audio:
        args += ["-map", "1:a:0"]
        first_sub = 2
    else:
        args += ["-map", "0:a?"]
    for k in range(len(subs)):
        args += ["-map", str(first_sub + k)]
    if chapter_text is not None:
        args += ["-map_metadata", str(meta_index), "-map_chapters", str(meta_index)]
    else:
        args += ["-map_chapters", "-1"]
    args += ["-c:v", "copy"]
    args += (
        ["-c:a", "aac", "-b:a", audio_bitrate]
        if audio_codec == "aac"
        else [
            "-c:a",
            "copy",
        ]
    )
    if subs:
        args += ["-c:s", "mov_text"]
    for k, lang in enumerate(subs):
        args += [
            f"-metadata:s:s:{k}",
            f"language={codes[lang]}",
            f"-metadata:s:s:{k}",
            f"title={titles.get(lang, lang)}",
            f"-metadata:s:s:{k}",
            # mp4 has no portable per-track title box; the handler name is the
            # field players and ffmpeg read back as the track label.
            f"handler_name={titles.get(lang, lang)}",
            f"-disposition:s:{k}",
            "default" if lang == default_subtitle else "0",
        ]
    if duration is not None:
        args += ["-t", str(float(duration))]
    args += ["-movflags", "+faststart", os.fspath(pending)]

    try:
        if chapter_text is not None:
            meta_path.write_text(chapter_text, encoding="utf-8")
        done = subprocess.run(args, capture_output=True, check=False)
        if done.returncode != 0 or not pending.is_file():
            tail = (done.stderr or b"").decode("utf-8", "replace")[-1500:]
            raise RuntimeError(f"ffmpeg failed ({done.returncode}):\n{tail}")
        pending.replace(output)
    finally:
        if pending.exists():
            pending.unlink()
        if meta_path.exists():
            meta_path.unlink()

    report = {"output": str(output)}
    try:
        info = probe_delivery(output, exe)
    except OSError:  # pragma: no cover - ffmpeg vanished after the mux
        info = None
    if info is None or not info["streams"]:
        report.update(
            probe="unavailable", streams=[], chapters=len(chapters or []), duration=None
        )
    else:
        # The mp4 muxer adds a text "data" track for chapters; it is not content.
        info["streams"] = [x for x in info["streams"] if x["type"] != "data"]
        report.update(probe="ffmpeg", **info)
    return report
