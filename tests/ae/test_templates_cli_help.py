"""Unified templates CLI: overview, unknown-command hint, unchanged delegation."""

import re

import pytest

from moviepy.ae.templates.__main__ import run


GROUPS = ("{init,validate,render}", "music", "library", "logo", "documentary")
HINT = "python -m moviepy.ae.templates <group> --help"


@pytest.mark.parametrize("argv", [[], ["-h"], ["--help"]])
def test_overview_lists_every_group(argv, capsys):
    if argv:
        with pytest.raises(SystemExit) as exc:
            run(argv)
        assert exc.value.code == 0
    else:
        assert run(argv) == 0
    out = capsys.readouterr().out
    for name in GROUPS:
        assert name in out
    assert "<group> --help" in out
    assert HINT in out


def test_unknown_command_suggests_and_exits_2(capsys):
    assert run(["rendr"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'rendr'" in err
    assert "did you mean 'render'?" in err
    assert "documentary" in err


def test_unknown_command_without_close_match(capsys):
    assert run(["zzzzqq"]) == 2
    err = capsys.readouterr().err
    assert "did you mean" not in err
    assert "music" in err


@pytest.mark.parametrize("group", ["music", "library"])
def test_overview_subcommands_exist_in_group_help(group, capsys):
    """Every subcommand named in the overview is a real subcommand of the group."""
    from moviepy.ae.templates.__main__ import GROUPS

    purpose = dict(GROUPS)[group]
    named = re.search(r"\(([a-z, ]+)\)", purpose).group(1).split(", ")
    assert named
    with pytest.raises(SystemExit):
        run([group, "--help"])
    out = capsys.readouterr().out
    choices = re.search(r"\{([a-z,]+)\}", out).group(1).split(",")
    assert sorted(named) == sorted(choices)


def test_episode_group_names_are_real_commands():
    from moviepy.ae.templates.__main__ import EPISODE_COMMANDS, GROUPS

    assert dict(GROUPS)["{init,validate,render}"]
    assert EPISODE_COMMANDS == ("init", "validate", "render")


@pytest.mark.parametrize(
    "argv",
    [
        ["init"],
        ["validate"],
        ["render"],
        ["music"],
        ["library"],
        ["logo"],
        ["documentary"],
    ],
)
def test_group_help_is_delegated(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        run(argv + ["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage:")
    assert "command groups:" not in out
