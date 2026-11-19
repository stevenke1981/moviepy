"""``python -m moviepy.ae.templates``: episode and music project command line."""

import difflib
import sys

from moviepy.ae.templates.episode import main


PROG = "python -m moviepy.ae.templates"
EPISODE_COMMANDS = ("init", "validate", "render")
GROUPS = (
    (
        "{init,validate,render}",
        "episode projects: scaffold, check and render an episode",
    ),
    ("music", "music-video projects (init, validate, build)"),
    ("library", "browse templates and recipes (list, show, recipes)"),
    ("logo", "render a looping logo animation"),
    ("documentary", "documentary news-collage and archival-film templates"),
)
KNOWN = EPISODE_COMMANDS + ("music", "library", "logo", "documentary")


def overview():
    """Return the top-level help text naming every command group."""
    lines = [f"usage: {PROG} <group> [options]", "", "command groups:"]
    for name, purpose in GROUPS:
        lines.append(f"  {name:<24}{purpose}")
    lines += ["", f"Run '{PROG} <group> --help' for the options of a group."]
    return "\n".join(lines) + "\n"


def _unknown(token):
    """Report an unknown command with a close-match suggestion; return 2."""
    near = difflib.get_close_matches(token, KNOWN, n=1, cutoff=0.5)
    msg = f"{PROG}: unknown command '{token}'"
    if near:
        msg += f" -- did you mean '{near[0]}'?"
    sys.stderr.write(msg + "\n\n" + overview())
    return 2


def run(argv=None):
    """Dispatch each command group to its CLI.

    ``music``, ``library``, ``logo`` and ``documentary`` go to their own CLIs;
    ``init``, ``validate`` and ``render`` go to the episode CLI.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        sys.stdout.write(overview())
        return 0
    if argv[0] in ("-h", "--help"):
        sys.stdout.write(overview())
        raise SystemExit(0)
    if argv[0] == "documentary":
        from moviepy.ae.templates.documentary.__main__ import main as documentary_main

        return documentary_main(argv[1:])
    if argv[0] == "library":
        from moviepy.ae.templates.library import main as library_main

        return library_main(argv[1:])
    if argv[0] == "logo":
        from moviepy.ae.templates.logo_loop import main as logo_main

        return logo_main(argv[1:])
    if argv[0] == "music":
        from moviepy.ae.templates.music_episode import main as music_main

        return music_main(argv[1:])
    if argv[0] in EPISODE_COMMANDS:
        return main(argv)
    return _unknown(argv[0])


if __name__ == "__main__":
    raise SystemExit(run())
