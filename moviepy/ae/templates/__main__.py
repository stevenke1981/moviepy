"""``python -m moviepy.ae.templates``: episode and music project command line."""

import sys

from moviepy.ae.templates.episode import main


def run(argv=None):
    """Dispatch ``music``/``library``/``logo`` to their CLIs, the rest to episodes."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "library":
        from moviepy.ae.templates.library import main as library_main

        return library_main(argv[1:])
    if argv and argv[0] == "logo":
        from moviepy.ae.templates.logo_loop import main as logo_main

        return logo_main(argv[1:])
    if argv and argv[0] == "music":
        from moviepy.ae.templates.music_episode import main as music_main

        return music_main(argv[1:])
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run())
