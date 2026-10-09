:orphan:

``moviepy.ae.templates`` 生產模板
=================================

這一組模板由兩個既有影片專案的製作流程整理而成：「夜燈說書」R23 的片頭片尾字卡與背景，以及 ``nightlamp-history`` 的 ``nlh-motion`` 動態工具。原本的做法是 Pillow 逐幀合成加上 ffmpeg 命令列；這裡改為以 ``moviepy.ae`` 的 AE 原語重新表達，讓每個模板回傳的都是一般的 layer、``Property`` 或 ``Composition``，之後可以加上關鍵影格、效果或再次合成。

模板只保存數字、顏色與字型**路徑**，不會替你決定內容文字。所有模板都要求明確的字型，缺字型時直接報錯，不會靜默換字。

新影片製作流程
--------------

要用 ``python -m moviepy.ae.templates`` 製作一集新影片（建立專案、撰寫 ``episode.json``、驗證、預覽與正式輸出），請直接閱讀 :doc:`episode_workflow`。本頁說明的是各個單一模板的 API。

音樂頻道製作流程
----------------

要製作長篇音樂影片（睡眠長片、番茄讀書片；``python -m moviepy.ae.templates music``），請閱讀 :doc:`music_workflow`。其中說明 ``music.json`` 的欄位、時長規則、續跑機制與人工驗收關卡。

預設值：``ChannelPreset``
-------------------------

``get_preset(name, **overrides)`` 回傳不可變的 ``ChannelPreset``。目前內建兩組：``nightlamp_story``（夜燈說書）與 ``nightlamp_history``（夜燈史話）。``with_overrides`` 會回傳經驗證的副本；改變 ``size`` 時，安全邊距會依比例縮放。

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - 欄位
     - ``nightlamp_story``
     - ``nightlamp_history``
     - 為何這樣設定
   * - ``size`` / ``fps``
     - 1920×1080 / 24
     - 1920×1080 / 24
     - 1080p 為頻道輸出規格；24 fps 沿用兩個來源專案的預覽速率。
   * - ``safe_margin``
     - (128, 90)
     - (96, 72)
     - 沿用各來源的版面邊距。故事版的 R23 值已對照 ``tokens.json``；史料版的值由 preset 指定，尚未另行驗證。
   * - ``overlay_opacity``
     - 0.5（黑色）
     - 0.0
     - 故事版以實拍或圖片作底，需要 50% 黑色遮罩才能讓暖白字在任何畫面上都清楚。史料版底色是宣紙，不需要壓暗。
   * - ``fade`` / ``hold``
     - 0.6 s / 1.5 s
     - 0.5 s / 2.0 s
     - 史料字卡文字較密，需要較長的停留時間才讀得完。
   * - 字級（``SubtitleStyle``）
     - 主字 72 px、副字 42 px
     - 主字 72 px、副字 42 px
     - 以 1080 高為參考，燒錄時依 ``scaled_px`` 等比縮放；中文 72 px、英文 42 px 是 NLH 字幕規格。
   * - 配色（``palette``）
     - ink ``#F3ECDC``，accent ``#D6A391``，lamp ``#D9A45F``，panel ``#493C2D``
     - paper ``#EFE4CC``，ink ``#1E1A16``，seal ``#B01E1C``，bamboo ``#C8A86A``，bamboo_dark ``#8A6A34``
     - 故事版取暖白與燈火色；史料版取宣紙、墨與朱印，與 NLH 的視覺一致。
   * - 字型（``fonts``）
     - title 思源宋體 Bold，body 思源黑體 Bold
     - title 思源宋體 Bold，body 思源黑體 Bold，quote 思源宋體 SemiBold
     - 思源體（TW 子集）完整涵蓋繁體字與標點；黑體用於字幕與內文，宋體用於標題與引文。

.. code-block:: python

    from moviepy.ae.templates.presets import get_preset

    preset = get_preset("nightlamp_story", size=(1280, 720))
    print(preset.size, preset.safe_margin, preset.scaled_px(72))
    print(preset.color("lamp"), preset.font("title"))

``preset.font(role, explicit=None)`` 會檢查檔案是否存在；不存在時丟出 ``FileNotFoundError``。``preset.to_json(path)`` 與 ``ChannelPreset.from_json(path)`` 可以把預設存成 JSON，讓專案不必 import 模板程式碼。

``media_background``：必要的背景素材
------------------------------------

``media_background(source, preset, *, frame_time=None, focal_point=(0.5, 0.5), overlay_opacity=None, overlay_color=None, duration=None)``

來源可以是圖片路徑、影片路徑（取單一格並靜止保持）、RGB ``uint8`` 陣列或 MoviePy clip。素材會以 ``cover_crop`` 等比放大到完全覆蓋畫面，再以 ``Transform`` 定位；上方的 ``SolidLayer`` 遮罩其 ``opacity`` 是可動畫的 ``Property``。

.. code-block:: python

    import numpy as np

    from moviepy.ae.templates.background import media_background
    from moviepy.ae.templates.presets import get_preset

    preset = get_preset("nightlamp_story")
    frame = np.full((720, 1280, 3), 90, np.uint8)  # 任何 RGB 陣列或圖片路徑
    bg = media_background(frame, preset, duration=6.0, overlay_opacity=0.4)
    bg.overlay.transform.opacity  # 可直接 keyframe 的 Property

素材不存在時一律拋出 ``FileNotFoundError``，不會用純色代替。``bg.provenance`` 記錄來源的 SHA-256、實際使用的影格時間與裁切框，方便重現。``contrast_report(background_rgb, box, text_color, *, overlay=None, target=3.0)`` 可估算文字區域最差的對比，但它是保守下界，不是無障礙認證。

``build_title_card``：片頭片尾字卡
---------------------------------

``TitleCardSpec`` 描述內容與時序；``build_title_card(spec, preset, background=None, *, fonts=None, transparent=False)`` 回傳一個 ``Composition``，各 layer 名稱為 ``Brand``、``Rule``、``Title``、``Subtitle``。字卡以淡入加上緩出的位移 ``Property`` 進場，結尾淡出。

.. code-block:: python

    from moviepy.ae.templates.background import media_background
    from moviepy.ae.templates.presets import get_preset
    from moviepy.ae.templates.title_card import (
        TitleCardSpec,
        build_bookends,
        build_title_card,
    )

    preset = get_preset("nightlamp_story")
    bg = media_background(frame, preset)  # 見上一節
    intro = build_title_card(
        TitleCardSpec(title="夜燈說書", brand="夜燈", subtitle="一盞燈，照見舊書"),
        preset,
        background=bg,
    )
    intro_comp, outro_comp = build_bookends(
        TitleCardSpec(title="開場"), TitleCardSpec(title="結語"), preset, background=bg
    )
    print(intro.title_report["title_lines"])

``build_bookends`` 讓片頭與片尾共用同一個背景遮罩 ``Property``，改一次即同時生效。``fit_lines(text, font_path, max_width, size, min_size, ...)`` 會先縮小字級、最後才換行，並遵守中文標點禁則；放不下時拋出 ``ValueError``，不做截斷。``readable_duration(requested, text, entrance, fade_out, fps)`` 依字數決定最短停留時間，文字較長時會延長而非截斷。

``fonts`` 參數預設讀取 preset 的字型。若要使用 Pillow 內建的拉丁字型，必須明確傳入 ``{"title": None, "body": None}``。

``subtitles``：雙語字幕
-----------------------

字幕模組負責三件事：解析與輸出字幕檔、檢查斷行與時序，以及燒錄成 AE layer。

.. code-block:: python

    from moviepy.ae.templates.presets import get_preset
    from moviepy.ae.templates.subtitles import (
        Cue,
        break_lines,
        burn_subtitles,
        find_bad_breaks,
        parse_srt,
        to_ass,
    )
    from moviepy.ae.templates.ken_burns import ken_burns

    preset = get_preset("nightlamp_story")
    zh = [Cue(0.0, 3.0, "夜燈照著舊書頁。", "zh-TW")]
    en = [Cue(0.0, 3.0, "The lamp lights old pages.", "en")]
    comp = ken_burns("still.png", preset, duration=4.0)
    comp = burn_subtitles(comp, zh, preset, secondary=en)  # 原地加入字幕層
    ass_text = to_ass(zh + en, preset.subtitles, preset.size, primary_lang="zh-TW")
    print(find_bad_breaks(zh))

``burn_subtitles(clip_or_comp, cues, preset, *, secondary=None, font=None, secondary_font=None)`` 若傳入 ``Composition`` 會原地修改並回傳它；若傳入一般 clip，會先包成新的 ``Composition``。``subtitle_layer`` 回傳 ``SubtitleLayer``，可用 ``add_layer`` 自行加入。``find_bad_breaks`` 會找出斷在詞中或拆開專有名詞的行；``check_timing(cues, timing=None)`` 檢查字幕時長與閱讀速度（``SubtitleTiming``）。``parse_srt``、``to_srt``、``to_vtt`` 處理標準字幕格式。

字幕過長：拆句而非截斷
~~~~~~~~~~~~~~~~~~~~~~

``reflow_cues(cues, measure, max_width, *, max_lines=1, protected=(), words=None, use_jieba=True, allow_char_breaks=False, max_gap=0.3, rejoin=True, rejoin_gap=0.6, report=None)`` 移植自夜燈 R26《嬌娜》R5 的「語意＋像素寬度」字幕版面。單行放得下的字幕原樣保留；放不下的字幕會拆成數句連續字幕，每句都放得下：

- 只在標點、空白或詞邊界切，不切開 ``protected`` 人名、數字或英文單字，並遵守行首行尾禁則（含彎引號 ``“”``）。
- 標點後的切點最優先。在詞中間切的代價高於多一句字幕，所以寧可多一句在逗號處結束的字幕。
- ``rejoin=True`` 會先把切壞的相鄰字幕併回再重排，例如沒在標點處結束的「…痛得連水｜都喝不下」或拆開人名的「孔雪｜笠」。新切點若落在原本的切點上，沿用原本的時間碼。
- 提供 ``words``（ASR 逐字時間，``load_words`` 可讀 Qwen3ASR 的 ``{"words": [...]}`` JSON）時，不會在同一個 ASR 詞的中間切，新切點的時間取自前後詞的邊界。停頓短於 ``max_gap`` 時兩句首尾相接以免閃爍，較長的停頓則保留為空檔。沒有 ``words`` 時，按原字幕內的字數比例內插。
- 文字一字不漏：所有輸出字幕串起來等於原文。真的切不開時拋出 ``LayoutError``。

.. code-block:: python

    from PIL import ImageFont

    from moviepy.ae.templates.presets import source_han_font
    from moviepy.ae.templates.subtitles import load_words, parse_srt, reflow_cues

    font = ImageFont.truetype(source_han_font("sans", "Bold"), 74)

    def measure(text):
        left, _, right, _ = font.getbbox(text, stroke_width=4)
        return right - left

    cues = parse_srt(open("zh.srt", encoding="utf-8").read(), "zh-TW")
    report = []
    one_line = reflow_cues(
        cues,
        measure,
        1352,  # R26 字幕安全框 1400 px 減左右留白 24 px
        protected=["孔雪笠", "皇甫公子", "嬌娜"],
        words=load_words("voice-full.json"),
        report=report,
    )

燒錄時改用 ``subtitle_layer(..., overflow="split", words=words)`` 或 ``burn_subtitles(..., overflow="split")``，圖層會用自己的字型量測後再拆句，``layer.reflow_report`` 記錄哪些字幕被拆開、用哪種方式對時。用《嬌娜》全片 532 句實測：輸出 616 句，文字完全一致，最寬一句 1340 px（上限 1352），沒有重疊，只有 5 句因整句沒有標點而必須在詞間切開。

``ken_burns``：靜圖次像素運鏡
-------------------------------

``ken_burns(image, preset, *, duration, move="auto", index=0, zoom=None, focus=None, distance=None, hold=None, lead=0.0, easing=None)``

回傳一個 ``Composition``，內含一個不透明的影像 layer（``comp.layers[0]``），其 ``Transform`` 由 ``CameraFraming`` 產生。位置與縮放都是小數座標，因此不會出現整數裁切造成的抖動。``move`` 可為 ``push``、``pull``、``pan-left``、``pan-right``、``pan-up``、``pan-down`` 或 ``auto``。預設 push 放大 8%、pan 放大 8% 並平移 6% 寬度，見 ``KEN_BURNS_DEFAULTS``。

.. code-block:: python

    from moviepy.ae.templates.ken_burns import check_motion, ken_burns
    from moviepy.ae.templates.presets import get_preset

    preset = get_preset("nightlamp_history")
    comp = ken_burns("still.png", preset, duration=5.0, move="push", zoom=(1.0, 1.1))
    first = comp.render_buffer(0.0).to_uint8_rgb(bg=(0, 0, 0))
    last = comp.render_buffer(4.9).to_uint8_rgb(bg=(0, 0, 0))
    rig = comp.framing  # 可編輯 focus / pan / zoom

``check_motion(clip_or_comp, start, duration, *, box=None)`` 以相位相關法量測位移，回傳 ``verdict``（``SMOOTH``、``STATIC`` 或 ``JITTER``）。這是 NLH ``motion-check`` 的對應功能，門檻也相同。

``chapter_tag``：章節標籤
-------------------------

``chapter_tag(items, number, duration, preset="nightlamp_history", *, period=5.5, font=None, position="top_left", seal=None, seed=7, size=44)``

回傳 ``ChapterTag``：``composition`` 為透明的標籤畫布，``strip``、``seal`` 是可獨立 keyframe 的 layer，``meta`` 含 ``x``、``y`` 等定位資訊。文字輪播週期預設 5.5 秒（NLH 頻道既定值），印章數字為大寫。

.. code-block:: python

    from moviepy.ae.templates.chapter_tag import chapter_tag

    tag = chapter_tag(
        ["城守住了", "本章人物：于謙"], number=1, duration=12.0, period=5.5
    )
    x, y = int(tag.meta["x"]), int(tag.meta["y"])
    # 疊到影片上時，把 tag.composition 放在 (x, y)

``vertical_quote``：直式引文
----------------------------

``vertical_quote(text, source, preset="nightlamp_history", *, dynasty=None, medium="auto", title=None, seal=None, per_column=10, duration=None, font=None, year=None, size=62, layout="center", speed=9.0, hold=3.0, fade_out=0.6, dim=0.35, seed=7)``

由右至左書寫，``\n`` 或 ``|`` 強制換欄。``medium="auto"`` 依 ``resolve_medium`` 判斷竹簡或紙卷：明確指定優先，其次看 ``year``（西元 220 年及以前為竹簡），最後看朝代名。``split_quote_pages`` 可把長引文切成多頁，切點優先落在句末。

.. code-block:: python

    from moviepy.ae.templates.quote import (
        resolve_medium,
        split_quote_pages,
        vertical_quote,
    )

    medium, reason = resolve_medium(year=-400)  # 竹簡
    pages = split_quote_pages("學而時習之，不亦說乎。" * 8, per_column=10, max_columns=12)
    scroll = vertical_quote(
        "學而時習之|不亦說乎", "《論語·學而》", medium=medium, duration=6.0
    )
    frame = scroll.render_buffer(5.0).to_uint8_rgb(bg=(0, 0, 0))

``name_tag``：人物名牌
----------------------

在人物圖旁邊顯示姓名與身分，例如「嬌娜／狐仙」。這是 R26 ASS 人物介紹卡的 AE 版：半透明深色底、金色對角括線、直書姓名、身分小字，可加印章（如「誌」）與指向人物的引線。

``place_name_tag(subject_box, card_size, canvas_size, *, side="auto", gap=24, margin=48, avoid=(), align="top")`` 是純幾何計算：``subject_box`` 是人物在畫面中的框 ``(x, y, w, h)``。``side="auto"`` 會先比較左右兩側的空間，再考慮上下，選出能放下名牌、且不壓到人物與 ``avoid`` 區域（預設為字幕安全框）的位置，名牌不會超出畫面邊界。真的沒有位置時拋出 ``ValueError``。

``NameTag(name, role=None, subject_box=..., start=0, duration=4, side="auto", orientation="vertical", seal=None, leader=False, ...)`` 描述一張名牌。``name_tag_layer(tag, canvas_size, *, preset=None, font=None, role_font=None, avoid=None)`` 回傳 AE layer：名牌從人物那一側滑出（0.45 秒）並淡入，結束前淡出（0.65 秒）。``add_name_tags(comp, tags)`` 會一次加入多張名牌，若兩張在同一時間互相重疊則拒絕。座標以 1920×1080 為基準，自動縮放到合成尺寸。字放不下時先縮小字級，縮到下限仍放不下才拋錯，不會截字。

.. code-block:: python

    from moviepy.ae.templates.name_tag import NameTag, add_name_tags
    from moviepy.ae.templates.presets import source_han_font

    tag = NameTag(
        "嬌娜",
        role="狐仙",
        subject_box=(1250, 150, 420, 650),  # 人物在 1920x1080 畫面中的位置
        start=12.0,
        duration=4.0,
        seal="誌",
        leader=True,
    )
    add_name_tags(
        comp,
        [tag],
        font=source_han_font("serif", "Bold"),  # 姓名：思源宋體
        role_font=source_han_font("sans", "Medium"),  # 身分：思源黑體
    )

``scene_overlay``：章節疊圖套組
-------------------------------

R5 的章節疊圖：左上頻道 logo（圖檔或文字）、右上半透明浮水印、右側直式章名（由上而下、由右而左，深色圓角底加金框）、右下紅色「立即訂閱 ▶」按鈕，按鈕在第 0、6、10 格依序縮放 80%→104%→100%。每章 4 秒，淡入淡出各 0.4 秒。版面預設值即 R26 policy 的矩形（``SceneLayout``），會依合成尺寸縮放。

``add_scene_overlays(comp, layout, chapters, *, logo=None, watermark=None, cta_text="立即訂閱", font=None, watermark_span="auto")`` 的 ``chapters`` 為 ``[(開始秒數, 章名), ...]``，章節時間互相重疊時拒絕。章名中的空格是優先換欄點，例如「第三章 雷劫守候」排成「第三章」與「雷劫守候」兩欄。``export_scene_overlay`` 可把一章的疊圖輸出成帶透明通道的 qtrle MOV，交給 FFmpeg 疊加。

.. code-block:: python

    from moviepy.ae.templates.scene_overlay import SceneLayout, add_scene_overlays

    add_scene_overlays(
        comp,
        SceneLayout(),
        [(95.0, "第一章 菩提寺"), (420.0, "第三章 雷劫守候")],
        logo="夜燈說書",
        watermark="夜燈說書@NanDayDream",
        font=source_han_font("sans", "Bold"),
    )

``EpisodeSpec`` 也接受這三項：``subtitles.overflow`` / ``subtitles.words``、``scene_overlay``（``chapters``、``logo``、``watermark``、``cta_text``、``layout``），以及 ``name_tags``（``NameTag.to_dict`` 格式的清單）。``build_episode`` 會檢查它們不與片頭片尾字卡重疊，並寫入 ``episode_report``。

``paper``：紙張與印章輔助
-------------------------

``moviepy.ae.templates.paper`` 提供 ``paper_texture``、``seal_stamp``、``drop_shadow``、``ink_bleed``、``torn_mask`` 等函式，以及 ``rgba_still`` 與 ``cached_rgba_clip``。它們是確定性（seeded）的 numpy 與 Pillow 運算，回傳陣列或 clip，供 ``chapter_tag`` 與 ``vertical_quote`` 內部使用；一般專案通常不需直接呼叫。

.. code-block:: python

    from moviepy.ae.templates.paper import paper_texture, seal_stamp

    sheet = paper_texture(320, 180, (239, 228, 204), seed=7)  # 只讀陣列，要改請 copy
    seal = seal_stamp("壹", 96, seed=3)  # RGBA uint8

字型政策
--------

- **明確指定**：``preset.font(role)`` 只接受 preset 中登錄的路徑，或呼叫端傳入的 ``explicit`` 路徑。
- **不靜默替換**：檔案不存在時丟出 ``FileNotFoundError``，錯誤訊息包含角色名與路徑。CJK 字形覆蓋各字型不同，不能悄悄換成別的字。
- **預設思源體**：兩套 preset 以 ``source_han_font(style, weight)`` 找出已安裝的 ``SourceHan{Sans,Serif}TW-{weight}.otf``（亦接受 TC 版），依序搜尋 ``C:/Windows/Fonts``、``%LOCALAPPDATA%/Microsoft/Windows/Fonts`` 與常見的使用者字型資料夾；找不到時回傳預期路徑，由 ``preset.font`` 報錯。要用其他字型請用 ``get_preset(name, fonts={...})`` 或 ``with_overrides``。
- **拉丁字型例外**：要用 Pillow 內建字型時，必須明確傳入 ``None``（例如 ``fonts={"title": None}``、``fit_lines(text, None, ...)``）。這是刻意的選擇，不是後備機制。

效能說明
--------

- **快取狀態**：``chapter_tag`` 與 ``vertical_quote`` 把靜止段落的狀態渲染一次並快取（``cached_rgba_clip``，預設最多 128 個狀態），靜止期間每幀只是字典查詢。
- **向量化**：遮罩、紙張紋理與陰影都是 numpy 運算；``paper_texture`` 與 ``seal_stamp`` 以種子產生，相同參數回傳相同像素。
- **字幕**：``SubtitleLayer`` 使用有上限的光柵快取（``cache_size=64``），每條字幕只光柵化一次。
- **實測**：``python -m examples.ae_templates_showcase`` 在本機 1920×1080 下，7 張靜態圖的總計時間約 2.0 秒（每幀渲染約 0.2 至 0.3 秒，含首次字型載入與快取建立）；數字依機器而異，僅供量級參考。
- **輸出時**：每次 ``render_buffer(t)`` 會產生新的 premultiplied float32 buffer；大量輸出時請逐幀寫入，不要全部保留在記憶體中。

限制
----

- ``media_background`` 只取**單一影格**並靜止保持，不做影片播放；需要動態背景請自行組合 ``AVLayer``。
- ``ken_burns`` 的來源必須是不透明影像；含透明像素會拋出 ``ValueError``。運鏡是二維裁切、平移與等比縮放，不會產生新的視角或視差。
- ``fit_lines`` 不截斷文字：放不下時直接拋出錯誤，需由呼叫端縮短文案或增加 ``max_lines``。
- ``vertical_quote`` 的引文應照原文輸入，模板不做繁簡轉換或校對。
- ``to_ass`` 預設字型名稱為 ``Microsoft JhengHei``，與燒錄用的思源黑體不同；若要 ASS 與燒錄外觀一致，請傳入 ``font_name="Source Han Sans TW"``。
- 本模組不依賴 SVG 渲染（``resvg_py``）；若環境沒有 ``jieba``，斷詞退回確定性的字元規則。

遷移對照
--------

**NLH 命令列 → AE 模板函式**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - NLH 命令（``nlh_motion.py``）
     - AE 模板函式
     - 備註
   * - ``chapter``
     - ``chapter_tag``
     - ``--items`` 以 ``|`` 分隔即 ``items``；``--num`` 對應 ``number``。
   * - ``quote``
     - ``vertical_quote``
     - ``--medium`` 對應 ``medium``；``--col-chars`` 對應 ``per_column``。
   * - ``kenburns``
     - ``ken_burns``
     - 輸出改為 ``Composition``，不直接寫 H.264。
   * - ``motion-check``
     - ``check_motion``
     - 門檻與 NLH 相同；``motion_smoothness`` 可直接處理影格陣列。
   * - ``breaks``
     - ``find_bad_breaks``
     - 輸入為 ``Cue`` 或文字行，不再直接讀 ASS／SRT 檔；檔案請先用 ``parse_srt``。
   * - ``subs``
     - ``burn_subtitles`` / ``to_ass``
     - 燒錄用 ``burn_subtitles``；只要字幕檔用 ``to_ass``。
   * - ``medium``（補充）
     - ``resolve_medium``
     - 回傳 ``(medium, reason)``。
   * - ``card``（近似）
     - ``build_title_card``
     - 版面與參數不是逐項對應，需自行檢查字級與位置。
   * - ``transition`` / ``join`` / ``overlay`` / ``batch``
     - 無對應模板
     - 尚未遷移，仍需使用原工具。

**R23 腳本 → 模板**

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - R23 腳本
     - 模板
     - 備註
   * - ``scripts/media_background.py``
     - ``media_background``
     - 只保留必要素材規則；不再有純色後備。
   * - ``scripts/ae_title_base.py`` 的 ``make_title``
     - ``build_title_card``（片頭片尾用 ``build_bookends``）
     - 文字以 Pillow 光柵化，不依賴 SVG；字級以 ``preset.scale`` 縮放。
   * - ``scripts/subtitle_pipeline.py``
     - ``subtitles``（``burn_subtitles``、``to_ass``、``parse_srt``、``find_bad_breaks``）
     - 斷行與時序檢查皆在模組內完成。

**R26 夜燈工作流程 → 模板**

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - R26 工具
     - 模板
     - 備註
   * - ``story_subtitle_delivery.py`` 的 ``reflow_with_pixel_policy``
     - ``reflow_cues`` / ``subtitle_layer(overflow="split")``
     - 不再要求字幕與 Qwen 文字等長；改以字元對齊，對不上的切點退回字數內插。
   * - ``story_subtitle_delivery.py`` 的 ``character_title_events``
     - ``name_tag``
     - ASS 改為 AE layer，並自動避開人物框與字幕框。
   * - ``story_ae_scene_layout.py``
     - ``scene_overlay``
     - logo 不再強制使用 Godot；需要立體 logo 時，可先用 ``motion.title_card`` 輸出圖片再傳入。

範例與測試
----------

- 範例：``python -m examples.ae_templates_showcase OUTPUT_DIR [--size 960x540]``。輸出目錄必須不存在或為空；缺少預設思源字型時以結束碼 2 退出。
- 測試：``tests/ae/test_templates_docs.py`` 會檢查本文件的程式碼可以通過語法解析、所引用的 ``moviepy.ae.templates`` 名稱都存在，並以 320×180 執行範例。
