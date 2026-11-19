"""The template catalogue stays in step with the code it describes."""

import json
import re
from dataclasses import fields

import pytest

from moviepy.ae.templates import library
from moviepy.ae.templates.episode import EpisodeSpec
from moviepy.ae.templates.library import (
    RECIPES,
    TEMPLATES,
    catalog,
    get_recipe,
    get_template,
    list_templates,
)


def test_names_are_unique_and_entries_load():
    names = [t.name for t in TEMPLATES]
    assert len(names) == len(set(names))
    for info in TEMPLATES:
        assert info.load() is not None, info.name


def test_episode_fields_exist():
    known = {f.name for f in fields(EpisodeSpec)}
    for info in TEMPLATES:
        for part in re.split(r"\s*/\s*", info.episode_field):
            if not part:
                continue
            assert part.split(".")[0].split("[")[0] in known, info.name


def test_recipes_reference_templates_and_build_specs():
    for recipe in RECIPES:
        assert set(recipe.templates) <= {t.name for t in TEMPLATES}
        EpisodeSpec.from_dict(recipe.load())


def test_lookup_errors_and_tags():
    with pytest.raises(KeyError, match="known"):
        get_template("nope")
    with pytest.raises(KeyError, match="known"):
        get_recipe("nope")
    assert {t.name for t in list_templates("jiaona")} >= {"name_tag", "scene_overlay"}
    assert "end_card" in catalog("cta")


def test_cli(capsys):
    assert library.main(["list", "--tag", "audio"]) == 0
    assert "loudness" in capsys.readouterr().out
    assert library.main(["show", "end_card"]) == 0
    assert json.loads(capsys.readouterr().out)["episode_field"] == "end_card"
    assert library.main(["show", "history"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["config_json"]["preset"] == "nightlamp_history"
    assert library.main(["recipes"]) == 0
    assert "story" in capsys.readouterr().out
