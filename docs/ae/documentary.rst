:orphan:

紀錄片參考效果 R1
=================

本版依 2026-10-13 使用者提供的兩張電視翻拍照片，實作兩個可重用的
MoviePy AE 模板。它們回傳原生 ``Composition``，並非一張不可編輯的合成圖，
也不需要 After Effects、Blender、Godot 或額外模型服務。

參考圖可確認的是版式；入場時間、視差、推近幅度是本版的可調設計，不能據此
宣稱還原原影片的運鏡。翻拍的紫偏色、摩爾紋、電視邊框、電線與反射人影不加入效果。
使用者照片、節目截圖、字型二進位檔均不隨程式發布。

快速執行
--------

在 ``feat/ae-parity`` 分支的專案目錄執行：

.. code-block:: console

    python -m pip install -e .
    python -m moviepy.ae.templates documentary --demo --language zh-TW --width 1920 --duration 6 --fps 24 --output-dir output/documentary

Windows 自動尋找微軟正黑體；必要時明確指定
``--font C:\Windows\Fonts\msjh.ttc``。其他平台可提供自行安裝、可涵蓋所用字元的
TTF/TTC/OTF 路徑。中文字型不存在時直接報錯，不用方框替代。
示範信紙是程式自製素材，非歷史證物；移動窗影用來確認頁面內確實播放影片。
``--demo`` 預設無音訊；使用自有影片時保留主素材音訊。

輸出包含兩支 H.264 MP4、各一張預覽 PNG 與 ``render-report.json``。
預設為 1920×1080、24 fps、6 秒；``--width`` 採偶數且高度取最接近的 16:9 偶數。
快速驗證可改 ``--width 640 --duration 2 --fps 12``。

替換實際素材
------------

.. code-block:: console

    python -m moviepy.ae.templates documentary --input media/interview.mp4 --clipping media/news1.png --clipping media/news2.png --headline "跨越半個世紀的故事" --source-label "來源：自有素材" --language zh-TW --duration 12 --output-dir output/documentary
    python -m moviepy.ae.templates documentary --input media/news-page.png --main-is-page --effect news --duration 8 --output-dir output/news
    python -m moviepy.ae.templates documentary --config examples/ae_documentary.json --duration 8 --output-dir output/custom

第一種輸入會建立帶有獨立影片區塊的新聞頁；已有完整新聞網頁截圖時使用
``--main-is-page``，避免再套一層新聞頁。``--effect`` 可選 ``news``、``film`` 或
``both``。圖片與影片可混用；預設最多四張背景卡片，Python API 可自訂更多。
完整截圖本身是靜態素材；要讓頁內影片持續播放，請用 ``build_news_page`` 建立。

範例 JSON 內 ``media/...`` 是待替換的路徑，不附帶素材。
JSON 的 ``input``、``clippings``、``font`` 均相對於 JSON 所在目錄解析；命令列
``--clipping`` 路徑仍相對於目前工作目錄。建議不要把不同來源的私人素材推到公開庫。
字幕 ``start`` / ``end`` 單位是秒，``text`` 為完整文字，區間為左閉右開。
本版版式每個語言一行，超長字幕請拆成多個 cue；不靜默裁字。

效果一：多層新聞拼貼
--------------------

``build_news_collage(main, clippings, ...)`` 建立 ``news.background``、
``news.clipping.01`` 至 ``04``、``news.main``。
前景寬度預設畫布的 63%，四周卡片刻意超出畫框，並有少量傾斜與柔和投影。
``push``、``pan``、``entrance`` 分別控制推近、位移及入場時間；將它們設為
``0``、``(0, 0)``、``0`` 即可鎖定卡片，不停止素材播放。

``build_news_page`` 內有 ``article.chrome`` 與 ``article.media``：
導覽列、標題與文章底圖是靜態美術；內嵌的影片是獨立 AE 視窗。
文字、配色由建構參數調整後重建底圖，不是把每個字變成 AE TextLayer。
現成截圖也不會被自動 OCR 或重建成網站 DOM。

.. code-block:: python

    from moviepy import VideoFileClip, ImageClip
    from moviepy.ae.templates import build_news_page, build_news_collage
    from moviepy.ae.templates.documentary import CollageCard

    with VideoFileClip("media/interview.mp4") as source:
        page = build_news_page(
            source, headline="跨越半個世紀的故事", body="記憶仍在，故事繼續。",
            source_label="來源：自有素材", font="C:/Windows/Fonts/msjh.ttc",
            duration=8,
        )
        card = CollageCard(
            ImageClip("media/news1.png"), center=(0.16, 0.24), width=0.36,
            rotation=-2, depth=0.4, name="news.archive",
        )
        comp = build_news_collage(page, [card], duration=8)
        comp.layer("news.main").transform.rotation = 1
        comp.write_videofile("news.mp4", codec="libx264", audio_codec="aac")

中心座標與寬度使用畫布比例，旋轉單位為度。
``depth`` 是 2.5D 視差倍率，不是從照片推算的真實 3D 深度。
背景卡片預設靜音，只保留主卡片音訊；需要自行混音時可編輯原生 layer。

效果二：相機／底片取景框
------------------------

``build_film_frame`` 分離 ``film.shell``、``film.viewport``、``film.hud``，
以及選用的 ``film.source``、``film.watermark``。
預設視窗比例為 ``(0.056, 0.163, 0.894, 0.777)``，依序為左、上、寬、高。
左上與右下削角、另外兩角圓角，以及上緣三個缺口，使用六個可編輯的原生遮罩。
播放三角形、ISO 標籤、頂部短橫線、右側底片孔、左下條碼與右下圓環是獨立 HUD 圖層。

.. code-block:: python

    from moviepy.ae.templates import build_film_frame

    comp = build_film_frame(
        source, duration=8, source_label="來源：自有素材", watermark="1961",
        font="C:/Windows/Fonts/msjh.ttc", push=0.035, pan=(0.015, 0),
    )
    viewport = comp.layer("film.viewport")
    # viewport.clip is the editable footage precomposition.
    # viewport.masks contains the bevel / rounding / notch masks.

``cut`` / ``radius`` 以 1920×1080 的參考像素設定，再隨輸出縮放。
``notches=False`` 移除三缺口；``hud=False`` 關閉裝飾資訊。
``source_label`` 與 ``watermark`` 預設空白，不自動複製原節目識別。
``fit='cover'`` 等比裁切填滿，``fit='contain', pan=(0, 0)`` 保留全幅並留黑邊；
兩者皆不拉伸。素材內的推近不會帶動外框、來源標示或字幕一起移動。

音訊、資源與已知限制
--------------------

輸入的 clip 為借用資源，建構函式不修改或關閉它；在外部關閉來源前完成輸出。
CLI 會自行關閉其打開的影片讀取器。嵌套合成透過可攜帶音訊的 AVLayer 連接，
不是丟失聲音的純圖層替身。來源影片短於成品時，影片保持最後影格，音訊到尾端
後為靜音；不假造循環、不變速拉長聲音。示範素材沒有音軌。

這不是 After Effects ``.aep`` 專案，也未在本版加入 GUI 控制面板、MCP 新工具、
新聞網站擷取、OCR、影片下載或聲音生成。
固定紙紋/HUD 的美術結果可快取；影音、變換、遮罩與字幕仍依時間渲染。
相同 seed 和時間的結果可重現，跳播順序不影響畫面。高解析度 CPU 合成不是即時預覽承諾。

驗證
----

.. code-block:: console

    python -m pytest tests/ae/test_documentary.py -q
    python -m pytest tests/ae -q

專用測試涵蓋實際 MP4 編碼解碼、動態內嵌影片、原生遮罩可編輯、RGBA 邊緣、
比例填滿/完整容納、音訊保留及背景卡片靜音、短片尾端、隨機跳播、Unicode 路徑、
JSON 相對路徑與非法參數。現有 AE CI 使用 Windows/Linux × Python 3.9/3.12。
文件僅描述測試範圍；實際成功/失敗以本次提交的 CI 與測試報告為準。

實作參照
--------

* ``moviepy/ae/composition.py``、``layers/av.py``、``camera.py``：既有 AE 合成、聲音與鏡頭 API。
* ``moviepy/ae/templates/paper.py``、``subtitles.py``：既有靜態快取、紙紋與雙語字幕。
* ``moviepy/ae/masks``、``effects/perspective/drop_shadow.py``：原生遮罩與投影。
* 兩張使用者提供的翻拍照片：僅作構圖參考，不公開附帶、不辨識人物、不將推測運鏡視為已知事實。
