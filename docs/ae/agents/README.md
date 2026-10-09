# Agent guides for the video templates

Operating manuals for AI agents (Claude Code or Codex sub-agents) that produce videos with the
`moviepy.ae.templates` pipelines, without a human explaining the steps. Each guide is written in
English, in checklist form, with a YAML front matter block that tells an agent when to load it.

| Guide | Skill name | Use for | Trigger phrases (examples) |
|---|---|---|---|
| [music_channel.md](music_channel.md) | `music-channel-episode` | Long-form music-channel videos through the `music` subcommands (init, validate, build). Modes `sleep_longform` (about 5h20m) and `study_pomodoro` (87 min). | 森息音界, 睡眠長片, 讀書片, 番茄讀書, music.json, study.json |
| [nightlamp_episode.md](nightlamp_episode.md) | `nightlamp-episode` | Narrated episodes through the top-level subcommands (init, validate, render). Channels `story` (夜燈說書) and `history` (夜燈史話). | 夜燈說書, 夜燈史話, 新集, episode.json |

Both guides share the same eight sections: when to use, inputs, command sequence, spec quick
reference, verification gates, hard rules, failure handling and a reporting template.

## What the guides assume

- The repository is `E:\moviepy_ae`. Run commands from the repo root, or set
  `PYTHONPATH=E:\moviepy_ae` first.
- Media (music, clips, images, narration, subtitles, titles, fonts) is supplied by a human. The
  guides forbid an agent from inventing or substituting it.
- Nothing is uploaded or published. Human gates (listening, visual watch, rights, private upload,
  fact check) are recorded as `NOT_RUN`.

## Checking the guides

The test module `tests/ae/test_agent_guides.py` checks that the guides stay true to the code:

- front matter has `name` and `description`;
- every command line that names the `moviepy.ae.templates` module parses: the subcommand exists in the real argparse
  help, every `--flag` appears in that help, and `--mode`, `--channel`, `--steps` and `--encoder`
  values are valid choices;
- every spec field in a quick-reference table exists on its dataclass;
- every error message quoted in a failure table appears verbatim in the source;
- every `docs/...` and `moviepy/...` path mentioned in a guide exists.

Run it from the repo root:

```console
cd E:\moviepy_ae
$env:PYTHONUTF8 = "1"
python -m pytest tests/ae/test_agent_guides.py -q -p no:cacheprovider
```

## Installing a guide as a Claude Code skill

Claude Code loads project skills from `.claude/skills/<name>/SKILL.md`. To install a guide, copy
it into a folder named after its `name` field and rename it to `SKILL.md`. For example, for the
music guide, the target is `.claude/skills/music-channel-episode/SKILL.md`. The front matter must
stay at the top of the file. A sub-agent can also be given the guide's path, or its full text, as
its instructions.

Do not install a guide until it has passed the test module above.
