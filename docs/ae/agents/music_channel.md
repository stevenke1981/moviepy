---
name: music-channel-episode
description: Build and verify long-form music-channel videos with `python -m moviepy.ae.templates music` (init, validate, build). Use for 森息音界 (Biophilic Haven) episodes, 睡眠長片 (sleep_longform, six-track ambient sleep video, about 5h20m), 讀書片 (study_pomodoro, 87-minute Pomodoro study video with chime and spectrum), music.json, study.json, Flow loop clips, thumbnail and QA steps. Trigger phrases: 森息音界, 睡眠長片, 讀書片, 番茄讀書, 音樂頻道, music channel episode, sleep longform, study pomodoro video, Biophilic Haven, music.json. Do NOT use for narrated story or history episodes (夜燈說書, 夜燈史話, 新集); use nightlamp-episode instead.
---

# Music channel episode (sleep_longform / study_pomodoro)

Operating manual for an AI agent. Follow the steps in order. Do not improvise commands. Stop and report when a gate fails.

Background and full tables: `docs/ae/music_workflow.rst`. Read it only when a field is not in section 4 below.

## 1. When to use / when NOT to use

USE when the human asks for any of:
- a 森息音界 episode, a 睡眠長片 (`sleep_longform`), or a 讀書片 (`study_pomodoro`);
- a long loop video built from approved music masters and approved visual clips;
- `music.json` or `study.json` work, or the `music init|validate|build` steps.

DO NOT use when:
- the job is a narrated story or history episode (夜燈說書, 夜燈史話, 新集): use `nightlamp-episode`;
- the human asks you to generate music, audio, images or video. This manual only assembles media the human supplies;
- the human asks to upload, publish, schedule or share anything. Never do this (section 6);
- the job needs Godot or 3D. Only then check `python -m moviepy.ae.three_d.requirements`.

## 2. Inputs required

Ask the human for every missing item. Never invent, substitute, generate or "placeholder-fill" media, titles or rights.

| Input | Exact location / field | If missing |
|---|---|---|
| Mode | `music init --mode sleep_longform` or `study_pomodoro` | Ask. |
| Project folder | new empty folder, e.g. `E:\music_projects\ep01` | Ask. Do not reuse a non-empty folder. |
| Music masters (one per track) | `tracks/<file>` referenced by `tracks[].source`; rights confirmed by human | Stop. Ask for the files and written rights confirmation. |
| Track ids and chapter lengths | `tracks[].id`, `tracks[].chapter_seconds` (sleep mode: required on every track) | Ask. |
| Visual clips (approved Flow clips, at least `clip_length` = 8.0 s each) | `visual/<file>` listed in `visual.clips`, or a prepared loop at `visual.cycle_path` | Stop. Ask for approved clips. |
| Track titles (cards, optional) | `cards.titles.<id>.zh/en/ja` | Leave `cards.enabled` false and report. Do not invent titles. |
| Thumbnail titles (optional) | `thumbnail.titles.zh` (required when enabled), `en`, `ja` | Leave `thumbnail.enabled` false and report. |
| Study schedule (`study_pomodoro` only) | `study.json` (created by `init`; human edits phases, three-language labels and hints) | Ask for label and hint text. Keep the generated schedule only if the human approves it. |
| Chime (optional, study mode) | `chime.path` (stereo 48 kHz) OR `chime.design`, not both | Use the default design only if the human approves it. |
| Fonts | `C:/Windows/Fonts/msjh.ttc` (zh, en) and `C:/Windows/Fonts/YuGothM.ttc` (ja) for thumbnail | Check existence first. If missing, stop and ask. |
| Tools | FFmpeg on PATH or imageio-ffmpeg; Python with repo importable | Run `python -m moviepy.ae.templates music --help`. If it fails, stop. |
| soundx (sox-rs) | `MOVIEPY_SOUNDX`, PATH, or `C:\Program Files\soundx\soundx.exe`; >= 0.3.0 for LUFS | Run `python -m moviepy.ae.templates.soundx --json`. Stop if `ok` is false. If `has_loudness` is false (0.2.0), LUFS falls back to FFmpeg loudnorm: report it, do not hide it. |

Text rules for every title, subtitle, hint and label: no medical or health claims (see section 6).

## 3. Exact command sequence

Run from the repo root. Use PowerShell.

```console
cd E:\moviepy_ae
```

Step 1. Create the project (once; refuses a non-empty folder):

```console
python -m moviepy.ae.templates music init E:\music_projects\ep01 --mode sleep_longform
```

Expected stdout: `created <path>\music.json`. Then:

1. Copy each approved master into `E:\music_projects\ep01\tracks\`.
2. Copy each approved clip into `E:\music_projects\ep01\visual\`.
3. Edit `music.json`: replace every `PLACEHOLDER` value (track paths, clip paths, track titles, `name`). If the human has no card or thumbnail titles, set `cards.enabled` and `thumbnail.enabled` to `false`. Never leave a `PLACEHOLDER` string in a field that is enabled.
4. Study mode only: edit `study.json`.

Step 2. Validate. Must exit 0 before any build:

```console
cd E:\music_projects\ep01
python -m moviepy.ae.templates music validate music.json
```

Step 3. Build the source artifacts (no full render yet):

```console
python -m moviepy.ae.templates music build music.json --steps audio,visual,overlays
```

Step 4. Preview first (60 seconds, from 0):

```console
python -m moviepy.ae.templates music build music.json --steps render --preview 0 60
```

Output: `build\render\<name>-preview-0-60.mp4` and its `.json` evidence. Check the preview gates in section 5 before Step 5.

Step 5. Full render. Long runs: launch in the background and log to a file (section 6):

```console
New-Item -ItemType Directory -Force build\logs | Out-Null
python -m moviepy.ae.templates music build music.json --steps render --workers 8 --encoder auto > build\logs\render.out.txt 2>&1
```

Use `--encoder libx264` to force software encoding. Use `--workers N` to set render processes (default: all cores).

Step 6. Thumbnail and QA (only if enabled in `music.json`):

```console
python -m moviepy.ae.templates music build music.json --steps thumbnail,qa
```

Step 7. Re-check the build as a whole (no `--steps` means every enabled step; all artifacts already built are verified by hash and skipped):

```console
python -m moviepy.ae.templates music build music.json
```

Expected: exit 0, JSON report with `steps` and `human_gates`. Artifacts that already exist and verify are reported with `"skipped": true`. If any step tries to rebuild, or reports a hash or settings mismatch, stop and read the error (section 7). Report whether this re-check ran; do not assume it.

Flags used above, all real: `music init --mode`, `music validate [--allow-missing]`, `music build --steps`, `--preview START SECONDS`, `--workers`, `--encoder {auto,libx264,h264_nvenc,hevc_nvenc,libx265}`.

Do not use `--allow-missing` for any build. It only lets `validate` run before files exist, to check lengths.

Build step names, in fixed order: `audio`, `visual`, `overlays`, `render`, `thumbnail`, `qa`. `--steps` takes a comma-separated subset in that order. `render` needs `audio`, `visual` and `overlays` to have run first.

## 4. Spec quick reference (`music.json`)

Only the fields an agent usually edits. Full tables: `docs/ae/music_workflow.rst` section 5.

| Field (JSON path) | Class | Default | Edit when |
|---|---|---|---|
| `mode` | MusicEpisodeSpec | required | Choose `sleep_longform` or `study_pomodoro`. Set at init. Do not change later. |
| `name` | MusicEpisodeSpec | `music_episode` | Output file name stem. Safe characters only. |
| `size` | MusicEpisodeSpec | `[1280, 720]` | Keep unless the human asks. Must be even. |
| `fps` | MusicEpisodeSpec | `24` | Keep 24. Total length must be a whole number of frames. |
| `tracks[].id` | TrackSpec | required | Unique track id, e.g. `G01` or `S01`. Used by cards and file names. |
| `tracks[].source` | TrackSpec | required | Path to the master, relative to `music.json`. |
| `tracks[].chapter_seconds` | TrackSpec | none | Sleep mode: required on every track. Study mode: give on all tracks or none. |
| `tracks[].title` | TrackSpec | none | Chapter title for the ffmetadata chapters. |
| `audio.target_lufs` | MusicAudioSpec | `-18.0` | Loudness target. QA checks it within 1.0 LU. |
| `audio.true_peak` | MusicAudioSpec | `-1.8` | True-peak target for loudnorm. |
| `audio.chapter_crossfade` | MusicAudioSpec | `12.0` | Seconds of overlap between chapters. Each chapter must be at least twice this. |
| `audio.edge_fade` | MusicAudioSpec | `4.0` | Must be <= `chapter_crossfade`. |
| `visual.clips` | VisualSpec | `[]` | Approved clip paths; each at least `clip_length` seconds. One scene per clip. |
| `visual.clip_length` | VisualSpec | `8.0` | Length of each clip used. Must exceed `2 * loop_crossfade`. |
| `visual.segment` | VisualSpec | `120.0` | Seconds per scene in the cycle. Must exceed `scene_crossfade`. |
| `visual.cycle_path` | VisualSpec | `null` | A prepared loop MP4. When set, `clips` is not read. |
| `spectrum.enabled` | SpectrumSpec | `null` (on in study mode) | Spectrum bar overlay. Keep the default for each mode. |
| `spectrum.position` | SpectrumSpec | `[96, 548]` | Overlay top-left in pixels. |
| `spectrum.opacity` | SpectrumSpec | `30.0` | Bar opacity. |
| `chime.enabled` | ChimeSpec | `null` (on in study mode) | Only valid in study mode. |
| `chime.path` | ChimeSpec | `null` | Stereo 48 kHz cue file. Not together with `design`. |
| `chime.below_music_db` | ChimeSpec | `10.0` | Cue level reduction under the music. |
| `study` | MusicEpisodeSpec | `study.json` | Required in study mode; forbidden in sleep mode. |
| `chapters` | MusicEpisodeSpec | `true` | Write ffmetadata chapters. |
| `encoder` | MusicEpisodeSpec | `auto` | Same choices as the CLI `--encoder`. |
| `workers` | MusicEpisodeSpec | `null` | Render processes. CLI `--workers` overrides it. |
| `output_dir` | MusicEpisodeSpec | `build` | Keep `build`. All artifacts go under it. |
| `cards.enabled` | CardsSpec | `false` | Show track title cards (sleep configs ship with it on). |
| `cards.titles` | CardsSpec | required when enabled | Object keyed by track id; each value needs `zh`, `en` or `ja`. |
| `thumbnail.enabled` | ThumbnailStepSpec | `false` | Enable only with human-approved titles. |
| `thumbnail.titles` | ThumbnailStepSpec | required when enabled | `zh` is required. Also `en`, `ja`. |
| `thumbnail.time` | ThumbnailStepSpec | `null` | Frame time in seconds; default is a third of the cycle. |
| `thumbnail.require_contrast` | ThumbnailStepSpec | `true` | Keep `true`. A failed contrast check must be reported, not bypassed. |
| `qa.enabled` | QaSpec | `false` | Run the automated QA on the render. |
| `qa.targets` | QaSpec | `{}` | Only `lufs`, `lufs_tolerance`, `true_peak_max`. Keep `{}` unless the human sets targets. |
| `qa.flash_seconds` | QaSpec | `null` | Limit flash analysis to N seconds. Keep `null` for the full check. |
| `ambience` | MusicEpisodeSpec | none | Particles, light arc and sleep fade. Only if the human asks. |

Study mode timing: `study.json` `duration_seconds` (5220 = 87 min) sets the audio length. `chapter_seconds` must sum to `duration + (n - 1) * chapter_crossfade`. The build refuses any other length.

## 5. Verification gates

Each gate needs a pass criterion, evidence, and a line in the final report. A technical PASS does not mean the content is approved. Human gates stay `NOT_RUN`.

Frame reference (for checks): 87 min at 24 fps = 125,280 video frames. 5 h 20 min (19200 s) at 24 fps = 460,800 video frames. Audio is 48 kHz: frames = seconds x 48000.

| # | Gate | Command or evidence | Pass criterion |
|---|---|---|---|
| G1 | validate | `music validate music.json` | Exit code 0. Stdout starts with `OK <name> (<mode>)`. Stderr has no `MISSING` or `PROBLEM` line. Record the `F frames` value: it is the expected `video_frames`. |
| G2 | audio | stdout JSON of `build --steps audio`, `steps.audio` | `steps.audio.video_frames` equals G1 frames. `steps.audio.frames` equals G1 seconds x 48000. Evidence file `build\audio\*.flac.json` exists next to the audio. `human_listening` is `NOT_RUN`. |
| G3 | visual | `steps.visual` | `steps.visual.cycle` present. `build\visual\cycle.mp4.json` (or `cycle.external.json`) has a `frames` key. Record its value. |
| G4 | overlays | `steps.overlays.levels.rows` (sleep: spectrum off, `levels` is null) | If the spectrum is on, `rows` equals G1 frames exactly (the code enforces it). Study mode: `build\overlays\study.ass` exists. |
| G5 | preview | `build\render\<name>-preview-0-60.mp4.json` | `frames` equals 1440 (60 x 24). `output_sha256` matches the file. |
| G6 | full render | `build\render\<name>.mp4.json` and its stdout | `frames` equals G1 frames. `output_sha256` matches the file. Study mode: `duration_seconds` in `study.json` equals the audio seconds. The `.ffmpeg.log` file has no error line. |
| G7 | thumbnail (if enabled) | `build\publish\thumbnail.jpg.json` | `contrast.passes` is `true`. `human_visual` is `NOT_RUN`. If `contrast.passes` is false, FAIL and report. Do not lower `require_contrast`. |
| G8 | QA (if enabled) | `build\qa\<name>.qa.json` | `overall` is not `FAIL`. `verdicts.flash` is `PASS` (flash PASS is required). `verdicts.loudness` is `PASS`: `integrated_lufs` within -19.0 to -17.0 and `true_peak_dbtp` <= -1.5. `verdicts.clipping` is `PASS` (`clipped_samples` 0). `verdicts.silence` is `PASS`, or `REVIEW` which needs a human to confirm each gap is intended. Report `REVIEW` as open, never as PASS. |
| G9 | human gates | `human_gates` in the build JSON | All four are `NOT_RUN`: `full_listening`, `full_visual_watch`, `rights`, `private_upload`. Report them as NOT_RUN. |

Order: G1 before any build. G2 to G4 before G5. G5 must pass before G6 (preview first, full render second). G6 before G7 and G8.

Note: `validate` prints `MISSING` for absent files and `PROBLEM` for spec errors. Either one means stop.

## 6. Hard rules

1. Never overwrite an existing artifact. Every artifact is an exclusive create. If a step reports that an artifact "was built from different settings" or "no longer matches its recorded hash", do not edit the evidence JSON. To rebuild, delete the artifact and its `.json` evidence, delete every later artifact that depends on it, state exactly which files you deleted and why, then rebuild. Ask the human first if the rebuild changes a finished render.
2. Never write `human_listening`, `human_visual`, `human_gates` or any `*_NOT_RUN` gate to anything other than `NOT_RUN`. Never mark a human check as done.
3. No medical, health or therapeutic claims in any text: titles, subtitles, hints, descriptions, labels, thumbnail text. Do not write "helps sleep", "treats insomnia", "reduces anxiety", "improves focus" or similar. Use only human-approved wording.
4. No upload, publish, share, schedule, or pin. Do not call any upload tool. The `private_upload` gate is for the human.
5. Long renders (87 min to 10 h): run in the background, write stdout and stderr to `build\logs\*.txt`, poll the log tail, and report progress with timestamps. Do not stop a running render unless the human asks.
6. For 87 min to 10 h videos, never push full frames through After Effects. Use only the provided steps (`audio`, `visual`, `overlays`, `render`). Do not write custom per-frame render scripts or run the episode CLI on these projects.
7. Do not use `--allow-missing` for a build. Do not invent, generate, or substitute tracks, clips, cues, fonts, titles or rights text.
8. Do not edit files under `docs/` or code under `moviepy/` as part of a production run.
9. Godot: this channel does not use Godot. Only if a Godot step is requested, run `python -m moviepy.ae.three_d.requirements` and report its output.

## 7. Failure handling

Match the message fragment (it comes from the code; the full text may add values after it).

| Message fragment | Cause | Fix |
|---|---|---|
| `MISSING <path>` (validate stderr) | Track, clip, cue, cycle or study file not found. | Ask the human for the file. Do not substitute. Re-run validate. |
| `PROBLEM <text>` (validate stderr) | A spec problem was found. | Fix the named field. Re-run validate. |
| `The music templates need soundx` | soundx not found or not runnable. | Ask the human to install soundx >= 0.3.0 from https://github.com/urtiger101-tw/sox-rs or set `MOVIEPY_SOUNDX`. Do not switch `loudness_backend` to `ffmpeg` without approval. |
| `tracks: every track needs chapter_seconds` | Sleep mode, a track has no `chapter_seconds`. | Add `chapter_seconds` to every track. |
| `tracks: give chapter_seconds on every track or on none` | Study mode, only some tracks have it. | Give it on all tracks or remove it from all. |
| `chapter_seconds must sum to duration` | Study mode: total audio does not match `study.json` `duration_seconds`. | Set chapter lengths so the total equals `duration + (n - 1) * chapter_crossfade`. |
| `is not a whole number of frames` | Total length gives a fractional frame count at `fps`. | Adjust `chapter_seconds` (whole seconds work at 24 fps). |
| `must be at least twice` | A chapter is shorter than `2 * audio.chapter_crossfade`. | Lengthen that chapter or lower `chapter_crossfade`. |
| `edge_fade must be <= chapter_crossfade` | Bad audio fade settings. | Lower `edge_fade`. |
| `study is required in study_pomodoro mode` | Study mode without `study`. | Set `"study": "study.json"`. |
| `study is only valid in study_pomodoro mode` | `study` set in sleep mode. | Remove it. |
| `chime.enabled needs study_pomodoro mode` | Chime on in sleep mode. | Turn it off. |
| `give clips or a prepared cycle_path` | Visual section empty. | Add approved clips or a cycle path. |
| `clip_length must exceed 2 * loop_crossfade` | Clip too short for the loop fade. | Use longer clips or a smaller `loop_crossfade`. |
| `segment must exceed scene_crossfade` | Scene shorter than its fade. | Increase `segment`. |
| `give a cue path or a design, not both` | Chime has both `path` and `design`. | Keep one. |
| `unknown music episode key(s)` | Typo or unsupported key in `music.json`. | Remove or correct the key. |
| `cards: titles for unknown track id(s)` | Card title keyed by an id that is not a track. | Use the track ids from `tracks[].id`. |
| `thumbnail: titles.zh is required when enabled` | Thumbnail on, no Chinese title. | Ask the human for `zh` title or disable thumbnail. |
| `directory exists and is not empty` | `init` target not empty. | Use a new folder. Do not delete the existing one without asking. |
| `steps must be a subset of` | Wrong `--steps` value. | Use names from `audio,visual,overlays,render,thumbnail,qa`. |
| `render needs the` | `render` run before its inputs exist. | Run `--steps audio,visual,overlays` first. |
| `overlays need the audio step first` | Overlays before audio. | Run `audio` first. |
| `preview must be (start, seconds)` | Bad `--preview` arguments. | Pass two numbers: `--preview 0 60`. |
| `is not a whole number of frames` (preview) | Preview start or length is not a whole number of frames. | Use times that are multiples of 1/fps (for example `--preview 0 60` at 24 fps). |
| `preview window ends at` | Preview runs past the end. | Shorten the window. |
| `exists without evidence; remove it` | An artifact exists but its `.json` is missing. | Do not guess. Report it. Delete only after the human agrees. |
| `exists but` `is missing` | Evidence exists, artifact is missing (interrupted run). | Delete the orphan `.json`, then rebuild that step. Say so in the report. |
| `was built from different settings or inputs` | `music.json` or an input changed after the build. | Follow hard rule 1 (delete artifact, evidence and downstream; report). |
| `no longer matches its recorded hash` | An artifact was edited or replaced after the build. | Restore it, or follow hard rule 1. Report which file. |
| `qa needs the rendered video and evidence` | QA run before render. | Run `render` first. |
| `qa.enabled is false` / `thumbnail.enabled is false` | Step named explicitly but disabled. | Enable it in `music.json` (only with approved inputs) or drop the step. |
| `ffmpeg (` followed by `failed:` | FFmpeg encode failed. | Read `build\render\<name>.mp4.ffmpeg.log`. Try `--encoder libx264`. Report the log tail. |
| `ERROR:` (any exit 1) | Generic wrapper for the messages above. | Read the text after `ERROR:` and match the table. |

## 8. Reporting template

Fill every line at the end. Write `NOT RUN` or `NOT VERIFIED` where it applies. Never imply a check ran.

```text
MUSIC CHANNEL REPORT
Episode: <name> (<mode>)   Project: <folder>
Spec: <path to music.json>   Config hash: <value from evidence, if known>

Commands run (in order, with exit codes):
- <command>  -> exit <n>
- ...

Gates:
- G1 validate:   <PASS|FAIL>  frames=<F> seconds=<S>
- G2 audio:      <PASS|FAIL>  frames=<..> video_frames=<..>
- G3 visual:     <PASS|FAIL>  cycle frames=<..>
- G4 overlays:   <PASS|FAIL|NOT APPLICABLE>  rows=<..>
- G5 preview:    <PASS|FAIL>  frames=<..>  file=<path>
- G6 full:       <PASS|FAIL|NOT RUN>  frames=<..>  file=<path>
- G7 thumbnail:  <PASS|FAIL|NOT ENABLED>  contrast=<..>
- G8 QA:         overall=<..> flash=<..> loudness=<..> clipping=<..> silence=<..>

Evidence files: <list of absolute paths>
Deleted or rebuilt (hard rule 1): <none, or list with reason>

Human gates (all NOT_RUN unless a human signed off):
- full_listening: NOT_RUN
- full_visual_watch: NOT_RUN
- rights: NOT_RUN
- private_upload: NOT_RUN

Not done / not verified: <list>
Open questions for the human: <list, e.g. silence REVIEW, missing titles>
```
