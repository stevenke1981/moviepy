"""Template library: one catalogue of every 夜燈說書 editing template.

Each ``TemplateInfo`` names a template, the function that builds it, the
``episode.json`` field that drives it (``""`` when it is used from Python
only) and the episode it was extracted from, so an editor can look up "how
was the 武則天 subtitle plate made" without reading the renderers. Recipes are
whole-episode starting points: ``python -m moviepy.ae.templates init DIR
--channel history`` copies the matching recipe JSON.

Command line::

    python -m moviepy.ae.templates library list [--tag TAG]
    python -m moviepy.ae.templates library show NAME
    python -m moviepy.ae.templates library recipes

Examples
--------
>>> from moviepy.ae.templates.library import get_template, list_templates
>>> get_template("subtitles").episode_field
'subtitles'
>>> "end_card" in [t.name for t in list_templates("wuzetian")]
True
>>> callable(get_template("ken_burns").load())
True
"""

import importlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


__all__ = [
    "TemplateInfo",
    "Recipe",
    "TEMPLATES",
    "RECIPES",
    "list_templates",
    "get_template",
    "get_recipe",
    "catalog",
    "main",
]

_CONFIGS = Path(__file__).parent / "configs"


@dataclass(frozen=True)
class TemplateInfo:
    """One catalogue entry.

    Parameters
    ----------
    name : str
        Library key.
    title : str
        Short Traditional Chinese name shown to editors.
    module, entry : str
        ``moviepy.ae.templates`` module and the public function or class that
        builds the template.
    episode_field : str
        ``EpisodeSpec`` field driving it from ``episode.json`` (``""`` when the
        template is used from Python only).
    origin : str
        Episode and revision the look was taken from.
    summary : str
        What it does, in one or two sentences.
    tags : tuple of str
        Search tags (``subtitle``, ``overlay``, ``camera``, ``audio`` ...).
    """

    name: str
    title: str
    module: str
    entry: str
    episode_field: str
    origin: str
    summary: str
    tags: tuple = ()

    def load(self):
        """Import and return the template's entry point."""
        module = importlib.import_module(f"moviepy.ae.templates.{self.module}")
        return getattr(module, self.entry)

    def to_dict(self):
        """Return a JSON-compatible dictionary."""
        value = asdict(self)
        value["tags"] = list(self.tags)
        return value


@dataclass(frozen=True)
class Recipe:
    """A whole-episode starting point: a shipped ``configs/*.json`` file.

    Parameters
    ----------
    name : str
        Recipe key (also the ``--channel`` of ``init``).
    title : str
        Traditional Chinese name.
    config : str
        File name inside ``moviepy/ae/templates/configs``.
    look : str
        The film whose look the recipe reproduces.
    templates : tuple of str
        ``TEMPLATES`` keys the recipe switches on.
    """

    name: str
    title: str
    config: str
    look: str
    templates: tuple = ()

    @property
    def path(self):
        """Absolute path of the recipe JSON."""
        return _CONFIGS / self.config

    def load(self):
        """Return the recipe JSON as a dictionary."""
        return json.loads(self.path.read_text(encoding="utf-8"))


TEMPLATES = (
    TemplateInfo(
        "presets",
        "頻道預設",
        "presets",
        "get_preset",
        "preset",
        "夜燈說書全頻道",
        "畫布、fps、配色、思源字型與字幕版式。nightlamp_history / nightlamp_story "
        "採用武則天 r2b 的字幕：繁中 72、英文 42、5% 貼字黑底、專名土黃 #C9A35D。",
        ("preset", "font", "subtitle"),
    ),
    TemplateInfo(
        "subtitles",
        "雙語燒錄字幕",
        "subtitles",
        "subtitle_layer",
        "subtitles",
        "武則天 r2b；嬌娜 R5 斷行",
        "繁中在上、英文在下；每行各自一塊貼字半透明黑底（plate_opacity / "
        "plate_padding），詞表專名上土黃色且不被斷行拆開；過長字幕依標點與 ASR "
        "字時間拆成多則（overflow='split'）。",
        ("subtitle", "wuzetian", "jiaona"),
    ),
    TemplateInfo(
        "glossary",
        "專名詞表",
        "subtitles",
        "load_glossary",
        "subtitles.highlight",
        "武則天 r2b",
        "讀 glossary.json（類別 → {中文: 英文}，中文可用 ／ 、 分隔別名），"
        "回傳要上色並保護的專名清單。",
        ("subtitle", "wuzetian"),
    ),
    TemplateInfo(
        "ken_burns",
        "靜圖運鏡",
        "ken_burns",
        "ken_burns",
        "shots",
        "武則天 r2b",
        "次像素推拉搖移；drift-left / drift-right 為 r2b 的「左右移動同時放大，"
        "停在主體上再切換」，同一張圖拆成數段時以 segment 接續同一運鏡。",
        ("camera", "wuzetian"),
    ),
    TemplateInfo(
        "video_retime",
        "影片變速與定格",
        "episode",
        "ShotSpec",
        "shots[].speed / shots[].freeze_at",
        "武則天 r2b",
        "影片鏡頭以 speed 放慢（6 秒片頭 0.75 倍成 8 秒），freeze_at 讓尾段走樣的"
        "素材停在最佳畫格直到該段結束。",
        ("camera", "video", "wuzetian"),
    ),
    TemplateInfo(
        "chapter_tag",
        "左上章節紙條輪播",
        "chapter_tag",
        "chapter_tag",
        "chapters",
        "夜燈史話 NLH；武則天 r2b",
        "撕紙條加朱印，項目上下輪播；repeat_every / visible 讓長章節每隔一段"
        "時間（r2b 為 38 秒、每次 7.4 秒）再滑入一次。",
        ("overlay", "chapter", "wuzetian"),
    ),
    TemplateInfo(
        "title_overlay",
        "片頭標題疊字",
        "title_card",
        "build_title_card",
        "title_overlay",
        "武則天 r2b",
        "不遮畫面的透明標題（主標、副標、一句鉤子），疊在片頭影片上淡入淡出。",
        ("overlay", "title", "wuzetian"),
    ),
    TemplateInfo(
        "title_card",
        "片頭片尾字卡",
        "title_card",
        "build_title_card",
        "intro / outro",
        "R23 夜燈",
        "全畫面字卡：品牌、分隔線、自動縮字的標題與副標，依閱讀時間自動延長。",
        ("title",),
    ),
    TemplateInfo(
        "quote",
        "直式史書引文",
        "quote",
        "vertical_quote",
        "quotes",
        "夜燈史話 NLH；武則天 r2b",
        "右起直書的宣紙或竹簡引文，依朝代自動選材質，附出處。",
        ("overlay", "quote", "wuzetian"),
    ),
    TemplateInfo(
        "source_insert",
        "史料圖插入與引用",
        "source_insert",
        "add_source_inserts",
        "source_inserts",
        "武則天 r2b",
        "網路找到的史料原圖完整呈現、不裁切不改色，角落以 12 號小字標示引用，"
        "自動避開字幕區與主體。",
        ("overlay", "source", "wuzetian"),
    ),
    TemplateInfo(
        "end_card",
        "片尾按讚訂閱分享輪播",
        "end_card",
        "add_end_card",
        "end_card",
        "武則天 r1/r2b",
        "片尾面板：頻道名、下一回預告、提問，按讚・訂閱・分享三個圖示輪流放大亮起。",
        ("overlay", "cta", "wuzetian"),
    ),
    TemplateInfo(
        "scene_overlay",
        "章節疊圖套組",
        "scene_overlay",
        "add_scene_overlays",
        "scene_overlay",
        "嬌娜 R5",
        "每章 4 秒：左上 logo、右上浮水印、直式章名、紅色訂閱鈕彈出。",
        ("overlay", "chapter", "cta", "jiaona"),
    ),
    TemplateInfo(
        "name_tag",
        "人物名牌",
        "name_tag",
        "add_name_tags",
        "name_tags",
        "嬌娜 R5",
        "人物圖旁的直式或橫式姓名卡（身分、印章、引線），自動找不擋臉也不擋字幕的位置。",
        ("overlay", "character", "jiaona"),
    ),
    TemplateInfo(
        "background",
        "背景媒體",
        "background",
        "media_background",
        "background",
        "R23 夜燈",
        "字卡底下的封面裁切圖或影片定格，加可動畫的壓暗層。",
        ("background",),
    ),
    TemplateInfo(
        "music_bed",
        "章節配樂組接",
        "music_audio",
        "assemble_chapters",
        "",
        "武則天 r2b",
        "兩首以上配樂依章節排段，循環接點 8 秒交叉淡化、段落切換 6 秒、首尾 3 秒淡入淡出。",
        ("audio", "wuzetian"),
    ),
    TemplateInfo(
        "loudness",
        "響度標準化",
        "music_audio",
        "normalize_loudness",
        "",
        "武則天 r2b",
        "以 soundx 把配樂調到 −18 LUFS／−2 dBTP 並重測；成片 −16 LUFS、峰值 ≤ −1.5 dBTP。",
        ("audio", "wuzetian"),
    ),
    TemplateInfo(
        "delivery",
        "成片封裝",
        "delivery",
        "mux_delivery",
        "",
        "武則天 r2b",
        "無聲成片加混音、繁中／英文／簡中三條軟字幕與章節點封裝成 MP4，回讀串流核對。",
        ("delivery", "subtitle", "wuzetian"),
    ),
    TemplateInfo(
        "logo_loop",
        "夜燈說書輪播 logo 標題",
        "logo_loop",
        "build_logo_loop",
        "",
        "夜燈說書頻道識別",
        "油燈加「夜燈說書」標題的無縫循環：光暈呼吸、火苗閃動、標題掃光、餘燼上升，"
        "副標依序輪播；可輸出 MP4 或帶 alpha 的 MOV。",
        ("logo", "loop", "brand"),
    ),
)

RECIPES = (
    Recipe(
        "history",
        "夜燈說書・歷史人物（武則天 r2b 版式）",
        "nightlamp_history.json",
        "https://www.youtube.com/watch?v=SA2kFayJ8Ok",
        (
            "presets",
            "subtitles",
            "glossary",
            "ken_burns",
            "video_retime",
            "chapter_tag",
            "title_overlay",
            "quote",
            "source_insert",
            "end_card",
        ),
    ),
    Recipe(
        "story",
        "夜燈說書・聊齋故事（嬌娜 R5 加武則天 r2b 版式）",
        "nightlamp_story.json",
        "嬌娜 R5／R6 加武則天 r2b 字幕與片尾",
        (
            "presets",
            "subtitles",
            "glossary",
            "ken_burns",
            "video_retime",
            "scene_overlay",
            "name_tag",
            "end_card",
        ),
    ),
)


def list_templates(tag=None):
    """Return the catalogue, optionally only entries carrying ``tag``.

    Parameters
    ----------
    tag : str, optional
        Keep entries whose ``tags`` contain it.

    Returns
    -------
    list of TemplateInfo
    """
    return [t for t in TEMPLATES if tag is None or tag in t.tags]


def get_template(name):
    """Return the ``TemplateInfo`` called ``name``.

    Raises
    ------
    KeyError
        For an unknown name; the message lists the known ones.
    """
    for info in TEMPLATES:
        if info.name == name:
            return info
    raise KeyError(f"unknown template {name!r}; known: {[t.name for t in TEMPLATES]}")


def get_recipe(name):
    """Return the ``Recipe`` called ``name`` (``history`` or ``story``)."""
    for recipe in RECIPES:
        if recipe.name == name:
            return recipe
    raise KeyError(f"unknown recipe {name!r}; known: {[r.name for r in RECIPES]}")


def catalog(tag=None):
    """Return the catalogue as aligned plain text, one template per line."""
    rows = [
        (t.name, t.title, t.episode_field or "-", t.origin) for t in list_templates(tag)
    ]
    if not rows:
        return ""
    widths = [max(_width(r[i]) for r in rows) for i in range(3)]
    return "\n".join(
        "  ".join(cell + " " * (w - _width(cell)) for cell, w in zip(row, widths))
        + "  "
        + row[3]
        for row in rows
    )


def _width(text):
    """Terminal columns of ``text`` (East Asian wide characters count 2)."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def main(argv=None):
    """Run ``python -m moviepy.ae.templates library ...``; return an exit code."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m moviepy.ae.templates library",
        description="夜燈說書 template library",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list", help="list templates")
    p_list.add_argument("--tag")
    p_show = sub.add_parser("show", help="show one template or recipe as JSON")
    p_show.add_argument("name")
    sub.add_parser("recipes", help="list whole-episode recipes")
    args = parser.parse_args(argv)
    if args.command == "list":
        print(catalog(args.tag))
    elif args.command == "recipes":
        for recipe in RECIPES:
            print(f"{recipe.name:<8}  {recipe.title}  ({recipe.config})")
            print(f"          {', '.join(recipe.templates)}")
    else:
        try:
            info = get_template(args.name).to_dict()
        except KeyError:
            try:
                recipe = get_recipe(args.name)
            except KeyError as error:
                parser.exit(2, f"{error.args[0]}\n")
            info = {**asdict(recipe), "templates": list(recipe.templates)}
            info["config_json"] = recipe.load()
        print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0
