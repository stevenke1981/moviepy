---
name: nightlamp-episode
description: Build and verify narrated 夜燈說書 (nightlamp_story) and 夜燈史話 (nightlamp_history) episodes with `python -m moviepy.ae.templates` (init, validate, render). Use for 夜燈說書, 夜燈史話, 新集 (a new episode), new episode folder, episode.json, Ken Burns still shots, chapter labels, vertical quotes, burned subtitles, narration and music mix, intro and outro title cards, preview render, master render. Trigger phrases: 夜燈說書, 夜燈史話, 新集, 新一集, nightlamp, nightlamp_story, nightlamp_history, episode.json, 說書影片, 史話影片. Do NOT use for music-channel videos (森息音界, 睡眠長片, 讀書片, music.json); use music-channel-episode instead.
---

# Nightlamp narrated episode (story / history)

Operating manual for an AI agent. Follow the steps in order. Do not improvise commands. Stop and report when a gate fails.

Background and full tables: `docs/ae/episode_workflow.rst`. Read it only when a field is not in section 4 below.

## 1. When to use / when NOT to use

USE when the human asks for any of:
- a new 夜燈說書 (`--channel story`) or 夜燈史話 (`--channel history`) episode, including 新集 or 新一集;
- an `episode.json` change, a preview, or a final render of such an episode;
- a single still-image Ken Burns shot, chapter label, vertical quote or burned subtitle that belongs to such an episode.

DO NOT use when:
- the job is a music-channel video (森息音界, 睡眠長片, 讀書片, `music.json`): use `music-channel-episode`;
- the human asks you to write the script, narration, quotes, or source citations. You may format text the human supplies. You do not author facts, quotes or narration;
- the human asks to upload, publish or share the video. Never do this (section 6);
- the human asks for a fact-check verdict. You can only report what is checked mechanically; history accuracy is a human gate.

## 2. Inputs required

Ask the human for every missing item. Never invent, substitute or generate media, text, quotes, sources or fonts.

| Input | Exact location / field | If missing |
|---|---|---|
| Channel | `init --channel story` or `init --channel history` | Ask. Only these two values exist. |
| Episode folder | new empty folder, e.g. `E:\episodes\ep012` | Ask. Do not reuse a non-empty folder. |
| Still images (approved) | `media/<file>` used by `shots[].image`; at least 1920x1080 px, never upscaled by you | Stop. Ask for approved images. |
| Video shots (optional) | `media/<file>` used by `shots[].video` | Ask. Video audio is never used. |
| Narration | `media/<file>` in `audio.narration`; stereo WAV preferred (mono is converted and about 3 dB quieter) | Stop. Ask for the approved narration. |
| Music bed (optional) | `audio.music` file; the human confirms its rights | Leave `audio.music` null and report. |
| Subtitles | `subtitles/zh-TW.srt` and `subtitles/en.srt` (UTF-8, BOM and CRLF accepted), in `subtitles.primary` and `subtitles.secondary` | Ask for the SRT files. Do not write subtitles yourself. |
| Titles and brand | `intro.title`, `intro.brand`, `intro.subtitle`, `outro.*`, `chapters[].items`, `chapters[].number` | Ask. Do not invent titles or chapter text. |
| Character names (optional) | `name_tags[].name`, `role`, `subject_box` (the person's box in 1920x1080 pixels), `start` | Ask for the names and the times they appear. Do not guess who is in a picture. |
| Glossary (optional) | `subtitles.glossary` (path to an r2b `glossary.json`) and `subtitles.highlight` (extra terms) | Ask for the glossary file or the terms. Never write proper nouns or translations yourself. |
| Title overlay (optional) | `title_overlay.title`, `title_overlay.subtitle`, `title_overlay.start` | Ask for the text and the start second. Text over the opening footage, not a card. |
| Source images (optional) | `source_inserts[].image`, `source_inserts[].citation`, and `title` / `caption` / `note` for `portrait` | Ask for each image and its citation. Never invent a source or a citation. |
| End card (optional) | `end_card.channel`, `end_card.line`, `end_card.question`, `end_card.actions` | Defaults are in section 4. Ask before changing the wording. |
| Chapter overlay (optional) | `scene_overlay.chapters` (`[start, title]`), `logo`, `watermark` | Ask for the chapter titles and the logo. |
| ASR word timing (optional) | `subtitles.words` (Qwen3ASR JSON with `words`) | Use only with `subtitles.overflow: "split"`. |
| Quotes (optional) | `quotes[].text`, `source`, `dynasty`, `year`, `title` as supplied | Only the text the human supplies. Never create a classical quote or source. |
| Font | `fonts` role paths; default Source Han Serif TW Bold (titles) and Source Han Sans TW Bold (body, subtitles), found by `source_han_font` in `C:/Windows/Fonts` or `%LOCALAPPDATA%/Microsoft/Windows/Fonts` | Check they exist. Missing font means stop and ask the human to install it. Never substitute a font silently. |
| Tools | FFmpeg (imageio-ffmpeg default; `FFMPEG_BINARY=auto-detect` for PATH ffmpeg) | Run `python -m moviepy.ae.templates --help`. If it fails, stop. |

Text rules for every title, subtitle, chapter item and quote: no medical or health claims (section 6). Historical claims are the human's responsibility; list them as open questions.

## 3. Exact command sequence

Run from the repo root. Use PowerShell.

```console
cd E:\moviepy_ae
```

Step 1. Create the episode folder (once; refuses a non-empty folder):

```console
python -m moviepy.ae.templates init E:\episodes\ep012 --channel history
```

Expected stdout: `created <path>\episode.json`. Then:

1. Copy approved images, narration and music into `E:\episodes\ep012\media\`.
2. Copy SRT files into `E:\episodes\ep012\subtitles\`.
3. Edit `episode.json`: replace every `PLACEHOLDER` value (`PLACEHOLDER_*.jpg`, `PLACEHOLDER_narration.wav`, `PLACEHOLDER_*.srt`, titles, quotes). Keep `preset` as `nightlamp_story` or `nightlamp_history` to match the channel.

Step 2. Validate. Must exit 0 before any render:

```console
python -m moviepy.ae.templates validate E:\episodes\ep012\episode.json
```

Expected stdout: `OK <name>: 1920x1080 @ 24 fps, <duration> s, <n> segments, <n> chapters, <n> quotes, audio` (or `no audio`). Record `<duration>`.

Step 3. Preview render (draft: 480x270 at 12 fps; layout and timing check only). Use its own empty folder:

```console
python -m moviepy.ae.templates render E:\episodes\ep012\episode.json E:\episodes\ep012\preview --preview
```

Step 4. Final render. Use a new empty folder. Do not pass `--overwrite`:

```console
python -m moviepy.ae.templates render E:\episodes\ep012\episode.json E:\episodes\ep012\final --workers 8 --encoder auto
```

Long runs: launch in the background and log to a file, for example `... > E:\episodes\ep012\logs\final.out.txt 2>&1` (create `logs` first with `New-Item -ItemType Directory -Force`). Poll the log and report progress.

Step 5. Master render (only if the human asks for a lossless master). Run this INSTEAD of Step 4, not after it: `--master` writes into the same output folder, and a folder that already holds a render is refused:

```console
python -m moviepy.ae.templates render E:\episodes\ep012\episode.json E:\episodes\ep012\final --master --workers 8 --encoder auto
```

Stills: by default the render writes stills for the intro and outro middles, each chapter start + 1 s, and each quote middle. To choose times, pass seconds: `--still 3.0 12.5`. Pass `--still` with at least one number.

Flags used above, all real: `init --channel {story,history}`, `render --preview`, `--master`, `--overwrite` (never used by this manual), `--workers N`, `--encoder {auto,libx264,h264_nvenc,hevc_nvenc,libx265}`, `--still T [T ...]`.

Expected render stdout: JSON with keys `video`, `encoder`, `workers`, `render_fps`, `report`, `stills`, `master`, `notes`, `preview`, `timings`.

## 4. Spec quick reference (`episode.json`)

Only the fields an agent usually edits. Full tables: `docs/ae/episode_workflow.rst` section 4. The JSON key `in` maps to Python field `clip_in`; `out` maps to `clip_out`.

| Field (JSON path) | Class | Default | Edit when |
|---|---|---|---|
| `name` | EpisodeSpec | `episode` | Episode name, written into `report.json`. |
| `preset` | EpisodeSpec | `nightlamp_history` | `nightlamp_story` or `nightlamp_history`. Match the channel. |
| `preset_overrides` | EpisodeSpec | `{}` | Only if the human asks. Overrides preset values. |
| `fonts` | EpisodeSpec | `{}` | Per-role font file paths. Must be real files. |
| `chapter_period` | EpisodeSpec | `5.5` | Seconds each chapter item stays on screen. Must be > 0. |
| `dip_color` | EpisodeSpec | `[0, 0, 0]` | Background and dip colour, three values 0 to 255. |
| `seed` | EpisodeSpec | `7` | Paper and bamboo texture seed. Same seed, same picture. |
| `background.source` | BackgroundSpec | required | Background image under the title cards. |
| `background.overlay_opacity` | BackgroundSpec | none | Value in [0, 1]. |
| `intro.title` | BookendSpec | required | Intro title text (human-supplied). |
| `intro.brand` | BookendSpec | `""` | Brand line. |
| `intro.subtitle` | BookendSpec | `""` | Subtitle line. |
| `intro.duration` | BookendSpec | `5.0` | Seconds on screen. |
| `outro.title` | BookendSpec | required | Outro title text (human-supplied). |
| `title_overlay.title` | TitleOverlaySpec | required | Opening title text drawn over the footage, with no card behind it (human-supplied). Same keys as `intro`. |
| `title_overlay.start` | TitleOverlaySpec | `0.0` | Timeline second when the title appears. `transition` must stay `cut`. |
| `title_overlay.layout` | TitleOverlaySpec | `left_column` | `left_column` for the 武則天 r2b look; `right_column` and `center` also exist. |
| `shots[].image` | ShotSpec | none | Still image path. Give exactly one of `image` or `video`. |
| `shots[].video` | ShotSpec | none | Video path. Give exactly one of `image` or `video`. |
| `shots[].duration` | ShotSpec | required for image | Seconds. Image shots need it. For video shots, never give a `duration` that contradicts `out - in`. |
| `shots[].move` | ShotSpec | none | One of `auto`, `static`, `push`, `pull`, `pan-left`, `pan-right`, `pan-up`, `pan-down`, `drift-left`, `drift-right`. `static` takes no `zoom`, `focus` or `distance`. Image shots only. |
| `shots[].zoom` | ShotSpec | none | Pair, start and end zoom, each >= 1. |
| `shots[].hold` | ShotSpec | none | Seconds held still. The motion check (G4) covers `duration - hold`. |
| `shots[].transition` | ShotSpec | `cut` | `cut`, `crossfade` or `dip`. The first shot must be `cut`. |
| `shots[].transition_duration` | ShotSpec | none | Seconds. Not used with `cut`. |
| `shots[].clip_in` | ShotSpec | none | JSON key `in`. Video shots only. |
| `shots[].clip_out` | ShotSpec | none | JSON key `out`. Video shots only. Must be greater than `in`. |
| `shots[].speed` | ShotSpec | none | Video shots only. Playback rate; `0.75` stretches a 6 s clip to 8 s. Without `duration`, the length is `(out - in) / speed`. |
| `shots[].freeze_at` | ShotSpec | none | Video shots only. Source second after which the frame holds still while the shot runs on. |
| `shots[].segment` | ShotSpec | `[0, 1]` | Image shots with a `drift-left` or `drift-right` move only. `[a, b]` with `0 <= a < b <= 1` plays part of the motion, so one still can cover two shots. |
| `chapters[].start` | ChapterSpec | required | Start second. |
| `chapters[].end` | ChapterSpec | required | End second. Must be greater than `start`. |
| `chapters[].items` | ChapterSpec | required | Non-empty list of strings (human-supplied). |
| `chapters[].number` | ChapterSpec | required | Integer >= 1. |
| `chapters[].repeat_every` | ChapterSpec | none | Seconds between re-appearances of the chapter tag, from `start` until `end`. The 武則天 r2b value is 38. |
| `chapters[].visible` | ChapterSpec | `7.4` | Seconds each appearance stays on screen. Needs `repeat_every`; must not exceed it. |
| `quotes[].start` | QuoteSpec | required | Start second. Quotes must not overlap. |
| `quotes[].text` | QuoteSpec | required | Quote text, as supplied by the human. |
| `quotes[].source` | QuoteSpec | none | Source title, as supplied by the human. |
| `quotes[].dynasty` | QuoteSpec | none | Dynasty, as supplied. |
| `quotes[].year` | QuoteSpec | none | Integer year, as supplied. |
| `subtitles.primary` | SubtitleSpec | required | Path to the primary SRT (zh-TW). |
| `subtitles.secondary` | SubtitleSpec | none | Path to the secondary SRT (en). |
| `subtitles.primary_lang` | SubtitleSpec | `zh-TW` | Language tag for the primary SRT. |
| `subtitles.secondary_lang` | SubtitleSpec | `en` | Language tag for the secondary SRT. |
| `subtitles.overflow` | SubtitleSpec | `wrap` | `split` turns a cue that does not fit into several shorter cues, cut at punctuation first. Use it when a render fails with a layout error. |
| `subtitles.words` | SubtitleSpec | none | ASR word-timing JSON, used to time the split cues. Only with `overflow: "split"`. |
| `subtitles.glossary` | SubtitleSpec | none | Path to a glossary JSON in the r2b `glossary.json` format (categories of `{zh: en}`). Its terms are drawn in the highlight colour. Human-supplied only. |
| `subtitles.highlight` | SubtitleSpec | `[]` | Extra terms drawn in the highlight colour, added to the glossary terms. Both nightlamp presets use `#C9A35D`. |
| `scene_overlay.chapters` | SceneOverlaySpec | required | List of `[start, title]`. Each chapter overlay lasts `layout.duration` (4 s) and must not touch the intro or outro card. |
| `scene_overlay.logo` / `watermark` / `cta_text` | SceneOverlaySpec | none / none / `立即訂閱` | Logo image path or text, watermark text, subscribe button text. |
| `name_tags[].name` / `role` / `subject_box` / `start` / `duration` / `side` / `seal` / `leader` | NameTag | duration 4 s, side `auto` | One card per character appearance (EpisodeSpec field `name_tags`). The card is placed beside `subject_box`, never over it or over the subtitles. |
| `source_inserts[].image` | SourceInsert | required | Source image path. Only resized, never edited (human-supplied). |
| `source_inserts[].start` / `duration` | SourceInsert | required | Timeline seconds the insert is on screen. Must lie inside the episode. |
| `source_inserts[].citation` | SourceInsert | required | Short source line in a corner. Human-supplied; never invented. |
| `source_inserts[].layout` | SourceInsert | `full` | `full` centres the whole image above the subtitles. `portrait` is a catalogue card: image at native size on the left, `title`, `caption` and `note` in a right column. |
| `source_inserts[].citation_corner` | SourceInsert | `auto` | `auto`, `top_right`, `bottom_right`, `top_left` or `bottom_left`. A corner that overlaps the subtitle area is an error. |
| `end_card.start` | EndCard | `0.0` | Seconds. `null` places the card at the end of the timeline, lasting `duration`. |
| `end_card.actions` | EndCard | `按讚`, `訂閱`, `分享` | One to five button labels. Each is highlighted in turn every `period` seconds. |
| `end_card.period` | EndCard | `1.6` | Seconds each button stays highlighted. |
| `end_card.channel` | EndCard | `夜燈說書` | Channel name shown as the large title. Must not be empty. |
| `audio.narration` | AudioSpec | required | Narration file path. |
| `audio.narration_gain_db` | AudioSpec | `0.0` | Narration gain. |
| `audio.music` | AudioSpec | none | Music bed path (rights confirmed by human). |
| `audio.music_gain_db` | AudioSpec | `-18.0` | Music bed level. |
| `audio.music_duck_db` | AudioSpec | `-8.0` | Fixed reduction while narration plays. Must be <= 0. |
| `audio.fade_in` / `audio.fade_out` | AudioSpec | `0.0` | Seconds. |

### 4.1 武則天 r2b 版式 checklist

Use this when the human asks for the 武則天 r2b look (reference film: https://www.youtube.com/watch?v=SA2kFayJ8Ok). `init` copies `nightlamp_history.json` or `nightlamp_story.json`, which already switch on most of these fields. Check each item against the episode JSON. Leave out what the human did not supply; do not invent it.

- Subtitles: the preset gives 72 px Chinese, 42 px English, a 5 % black plate and `#C9A35D` highlight. Put proper nouns the human supplies in `subtitles.glossary` or `subtitles.highlight`. Use `subtitles.overflow` `split` only if a render fails with a layout error.
- Opening: `title_overlay` puts the text over the first footage with no card. Use it only if the human wants the r2b opening; keep `transition` at `cut`.
- Opening footage: a video shot with `speed` `0.75`. Its `duration` is `(out - in) / speed`. Use `freeze_at` for a held frame.
- Camera: image shots with `move` `drift-left` or `drift-right` and `hold` `1.15`. Use `segment` to split one still over two shots. Check every such shot with G4.
- Chapter tag: `chapters[].repeat_every` `38` and `chapters[].visible` `7.4`. The chapter `items` are human-supplied.
- Source images: `source_inserts` with `layout` `full`, or `portrait` for a catalogue card. Each needs a human-supplied `citation`.
- End card: `end_card` with `start` `null`, so it sits at the end of the timeline. The default buttons are 按讚, 訂閱, 分享.
- Audio and delivery are separate steps outside the CLI. `build_episode` does not normalise loudness. If the human asks for the r2b audio finish, normalise the mix with `normalize_loudness` (soundx, target -16 LUFS, true peak -1.5 dBTP) and package the deliverable with `mux_delivery`. Both are Python APIs; see `docs/ae/templates.rst` (sections 成片封裝 and 響度標準化). If they were not run, report them as NOT RUN.

## 5. Verification gates

Each gate needs a pass criterion, evidence, and a line in the final report. A mechanical PASS does not mean the episode is approved. Human gates stay `NOT_RUN`.

| # | Gate | Command or evidence | Pass criterion |
|---|---|---|---|
| G1 | validate | `validate episode.json` | Exit 0. Stdout starts with `OK`. Duration matches the sum of the shot timeline you expect (report it). |
| G2 | subtitle breaks | Python: `find_bad_breaks(cues, protected=[...])` per SRT (zh-TW and en) | Returns `[]` for each file. |
| G3 | subtitle timing | Python: `check_timing(cues)` per SRT | Returns `[]` for each file. |
| G4 | motion smoothness | Python: `check_motion(comp, start, duration - hold)` for every image shot with `hold` | Every verdict is `SMOOTH`. `STATIC` and `JITTER` are FAIL. |
| G5 | preview | `preview\episode.mp4` exit 0; CLI JSON | Exit 0. `video` exists. `preview` is true. Report `timings`. |
| G6 | final render | `final\report.json`, CLI JSON | Exit 0. `report.json` has keys `timeline`, `chapters`, `quotes`, `subtitles`, `layers`, `sources`, `audio`. Each `sources` entry has a SHA-256 that matches the file on disk. `duration` in `report.json` equals G1 duration. |
| G7 | master (only if requested) | `final\master\manifest.json`, `final\master_audio.wav` | Both exist. `notes` in the CLI JSON does not say the master is silent. If `notes` says silent, FAIL the master gate and report the note. |
| G8 | stills | `final\stills\*.png` | Files exist (one per still time). Agent may open them to look for obvious cut-off text; the visual check is still `NOT_RUN` for the human. |
| G9 | full watch | human | `NOT_RUN`. Only the human can watch the whole episode. |
| G10 | fact check (history) | human | `NOT_RUN`. Historical claims, quotes and sources are the human's responsibility. |

Preview before final: G5 must pass before G6. Do not run the final render for a layout you have not previewed.

Python snippets for G2 to G4 (run from the repo root; adjust paths):

```python
from moviepy.ae.templates.subtitles import find_bad_breaks, parse_srt, check_timing

text = open(r"E:\episodes\ep012\subtitles\zh-TW.srt", encoding="utf-8-sig").read()
cues = parse_srt(text, "zh-TW")
print(find_bad_breaks(cues, protected=["夜燈"]))  # [] means pass
print(check_timing(cues))  # [] means pass
```

```python
from moviepy.ae.templates.episode import EpisodeSpec, build_episode
from moviepy.ae.templates.ken_burns import check_motion

spec = EpisodeSpec.from_json(r"E:\episodes\ep012\episode.json")
comp = build_episode(spec)
try:
    for seg in comp.episode["timeline"]:
        if seg["kind"] == "shot" and seg["hold"] is not None:
            print(seg["name"], check_motion(comp, seg["start"], seg["duration"] - seg["hold"])["verdict"])
finally:
    for clip in comp.episode_clips:
        clip.close()
```

## 6. Hard rules

1. Never overwrite an existing output. Never pass `--overwrite`. Use a new empty folder for each render. To re-render into an existing folder, ask the human first; if approved, delete that folder deliberately, state it in the report, then render.
2. Never write `human_visual`, `human_listening`, or any human gate to anything other than `NOT_RUN`.
3. No medical, health or therapeutic claims in any text: titles, subtitles, chapter items, quotes, descriptions. Do not write "cures", "helps sleep", "treats anxiety" or similar. Use only human-approved wording.
4. Do not author quotes, sources, dynasties, years or historical facts. Use only what the human supplies. If a required fact is missing, ask.
5. No upload, publish, share, schedule, or pin. Do not call any upload tool.
6. Never substitute media, fonts or narration. A missing file is a stop condition.
7. Long renders: run in the background, log to a file, poll the log, and report progress with timestamps. Do not kill a running render unless the human asks.
8. Do not edit `docs/` or `moviepy/` as part of a production run.
9. Godot is not used by this template. Only if a Godot step is requested, run `python -m moviepy.ae.three_d.requirements` and report its output.

## 7. Failure handling

Match the message fragment. The full message may add values after the fragment.

| Message fragment | Cause | Fix |
|---|---|---|
| `output directory is not empty` | `render` target folder already has files. | Use a new folder. Do not pass `--overwrite`. See hard rule 1. |
| `output path is not a directory` | Output path is an existing file. | Use a folder path. |
| `directory exists and is not empty` | `init` target not empty. | Use a new folder. |
| `not found:` | A referenced media or subtitle file does not exist (for example a `PLACEHOLDER_*` name was not replaced). The message starts with the label, e.g. `background source not found: <path>`. | Ask the human for the file. Do not substitute. |
| `font not found` | A font path in `fonts` (or the default) is missing. | Ask for the font, or ask the human to install the default Source Han Serif/Sans TW fonts. Never substitute silently. |
| `has no glyph for` | The title text contains a character the font lacks. | Ask the human to change the text or supply a font that covers it. |
| `text has more than` | Title or chapter text exceeds the line limit. | Ask the human to shorten it. |
| `reading time insufficient` | A title card is on screen too briefly to read. | Lengthen `duration` for that card (ask if it changes pacing). |
| `falls outside the preset safe margin` | A layout element is outside the safe area. | Adjust layout values or ask the human. |
| `safe margins leave no room` | Margins too large for the text column. | Adjust preset overrides with human approval. |
| `unknown preset` | `preset` is not `nightlamp_story` or `nightlamp_history`. | Use one of the two values. |
| `unknown episode key(s)` | Typo or unsupported key at top level of `episode.json`. | Remove or correct it. |
| `unknown key` | Typo or unsupported key inside a section. | Remove or correct it. |
| `missing required key(s)` | A required section field is absent. | Add it (ask the human for the content). |
| `an episode needs at least one shot` | `shots` is empty. | Add shots with approved media. |
| `give exactly one of image or video` | A shot has both or neither. | Keep one. |
| `an image shot needs a duration` | Image shot without `duration`. | Add `duration`. |
| `duration contradicts (out - in) / speed` | Video shot: `duration` disagrees with `(out - in) / speed` (no `freeze_at`). | Give one of them only. |
| `shots[0]: no previous segment, transition must be cut` | First shot uses a transition. | Set `transition` to `cut`. |
| `quotes overlap` | Two quotes share screen time. | Move one `start`. |
| `lies outside the` | A subtitle cue is outside the timeline. | Fix the SRT timing or the shot durations. |
| `invalid episode JSON` | `episode.json` is not valid JSON. | Fix the syntax (commas, quotes, encoding). |
| `an episode must be a JSON object` | Top level is not an object. | Wrap the content in `{ }`. |
| `a title overlay takes no transition` | `title_overlay` has a `transition` other than `cut`. | Remove `transition` from `title_overlay`. |
| `in/out apply to video shots only` | `in` or `out` set on an image shot. | Remove them, or make it a video shot. |
| `speed/freeze_at apply to video shots only` | `speed` or `freeze_at` set on an image shot. | Remove them, or make it a video shot. |
| `a static shot takes no zoom/focus/distance` | `move` is `static` with `zoom`, `focus` or `distance`. | Remove those fields, or change `move`. |
| `segment applies to drift moves only` | `segment` on a shot whose `move` is not `drift-left` or `drift-right`. | Remove `segment`, or change `move`. |
| `segment must satisfy` | `segment` is not `[a, b]` with `0 <= a < b <= 1`. | Fix the two numbers. |
| `visible needs repeat_every` | `chapters[].visible` given without `repeat_every`. | Add `repeat_every`, or remove `visible`. |
| `visible must not exceed repeat_every` | `chapters[].visible` is larger than `repeat_every`. | Lower `visible`. |
| `citation must be a non-empty string` | A `source_inserts` entry has an empty `citation`. | Ask the human for the citation. Never invent one. |
| `citation_corner must be one of` | `citation_corner` is not one of the five listed values. | Use one of the listed values. |
| `ERROR:` (any exit 1) | Generic wrapper for the messages above. | Read the text after `ERROR:` and match the table. |
| `ffmpeg (` followed by `failed:` | The encoder failed. | Read the log printed in the error. Try `--encoder libx264`. Report the log tail. |

## 8. Reporting template

Fill every line at the end. Write `NOT RUN` or `NOT VERIFIED` where it applies. Never imply a check ran.

```text
NIGHTLAMP EPISODE REPORT
Episode: <name>   Channel: <story|history>   Preset: <nightlamp_story|nightlamp_history>
Folder: <path to episode folder>   Spec: <path to episode.json>

Commands run (in order, with exit codes):
- <command>  -> exit <n>
- ...

Gates:
- G1 validate:        <PASS|FAIL>  duration=<s>  segments=<n>
- G2 subtitle breaks: <PASS|FAIL>  zh-TW=<[] or list>  en=<[] or list>
- G3 subtitle timing: <PASS|FAIL>  zh-TW=<[] or list>  en=<[] or list>
- G4 motion:          <PASS|FAIL>  shots=<n>  non-SMOOTH=<list>
- G5 preview:         <PASS|FAIL|NOT RUN>  file=<path>
- G6 final render:    <PASS|FAIL|NOT RUN>  file=<path>  sources_sha256=<checked|not checked>
- G7 master:          <PASS|FAIL|NOT REQUESTED>  notes=<text>
- G8 stills:          <count> files; agent look-over: <note or NOT RUN>

Evidence files: <list of absolute paths: episode.mp4, report.json, stills, logs>
Deleted or re-rendered (hard rule 1): <none, or list with reason>

Human gates (all NOT_RUN unless a human signed off):
- full watch: NOT_RUN
- fact check (history): NOT_RUN
- rights (narration, music, images): NOT_RUN
- private upload: NOT_RUN

Not done / not verified: <list>
Open questions for the human: <missing media, missing titles or quotes, font coverage, pacing, facts>
```
