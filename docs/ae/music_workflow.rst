:orphan:

音樂頻道長片製作流程（``moviepy.ae.templates`` music）
======================================================

本文說明如何用 ``python -m moviepy.ae.templates music`` 製作長篇音樂影片：從建立專案資料夾、準備母帶與畫面素材、撰寫 ``music.json``，到驗證、預覽與正式輸出，以及人工驗收。流程由「森息音界 / Biophilic Haven」（``E:\music_channel``）的製作方式整理而成，對應兩種模式：

* ``sleep_longform``：長篇放鬆／睡眠片，即 S06 型（``build-season-06.ps1``），六首各 3210 秒、全長 5 小時 20 分。
* ``study_pomodoro``：番茄讀書片，即 S14 型（``s14-assets.py``、``s14-render.py``、``study-overlay.py``），全長 87 分鐘，附階段提示音與頻譜條。

所有欄位名稱、指令參數與預設值皆取自 ``moviepy/ae/templates/music_episode.py``、``configs/music_*.json`` 與各模組。若程式碼改變，請以原始碼為準並同步更新本文。

新影片（非音樂）的製作流程見 :doc:`episode_workflow`；各單一模板的 API 見 :doc:`templates`。

一、用途與設計
--------------

長片的難點在於畫面與聲音都很長：87 分鐘約為 125,280 個影格，10 小時約為 864,000 個影格。若每一幀都交給 After Effects 渲染，時間與磁碟都不可行。本模組的做法是**只讓 AE 渲染短循環與小疊層，其餘交給 FFmpeg 循環**：

* 背景：AE 只渲染一段無縫循環（``seamless_loop``，片段 8 秒、首尾溶接 2 秒），再把數段循環排成一個場景循環（``scene_cycle``，每段 120 秒、場景之間溶接 2 秒），並以 ``export_loop`` 輸出成 CFR、封閉 GOP（1 秒）、無 B 影格的 MP4。之後 FFmpeg 以 ``-stream_loop -1`` 重複播放這個檔案，背景不會逐幀進入 AE。
* 疊層：頻譜條（``SpectrumLayer``）只佔 1088×140 像素，字幕（ASS）由 FFmpeg 的 ``ass`` 濾鏡燒入。AE 僅需逐幀渲染這個小區域；頻譜的音訊分析則預先做成 ``.npy`` 表格，各 worker 讀取即可。
* 聲音：母帶以串流方式處理（``_audio_io`` 分塊讀寫），做兩段式 ``loudnorm``；章節循環與接合也逐塊進行，記憶體只保留一首母帶與交叉淡化窗口。

因此長度主要影響 FFmpeg 的編碼時間與磁碟空間，而非 AE 的記憶體。87 分鐘（S14）與 5 小時 20 分（S06）是頻道已實際製作的長度；10 小時屬於同一設計的延伸，本 repo 尚未實測。

二、環境
--------

Python 套件
~~~~~~~~~~~

本 repo 的 ``moviepy`` 必須優先於 site-packages 中的舊版。兩種做法擇一：

.. code-block:: console

    pip install -e E:\moviepy_ae

.. code-block:: console

    $env:PYTHONPATH = "E:\moviepy_ae"
    cd E:\moviepy_ae

``pip install -e`` 為永久設定；``PYTHONPATH`` 只對目前的 PowerShell 視窗有效。本模組依賴 numpy（頻譜分析）與 Pillow（精簡版字幕量字寬）。CI 以 Python 3.9 與 3.12 測試本文件。

FFmpeg 與編碼器
~~~~~~~~~~~~~~~

音訊讀寫與渲染都使用 ``moviepy.config.FFMPEG_BINARY``。在本機它指向 imageio-ffmpeg 內附的 ffmpeg 7.1，已內含本流程需要的功能（2026-10-09 以該執行檔檢查）：

* 濾鏡：``ass`` 與 ``subtitles`` （libass）、``loudnorm``、``xfade``、``atrim``、``overlay``。
* 編碼器：``libx264``、``libx265``、``h264_nvenc``、``hevc_nvenc``、``aac``。

不需要另外安裝系統 FFmpeg。若要改用其他版本，請確認上述濾鏡與編碼器都存在。

字型
~~~~

睡眠片不燒錄字幕，不需要字型。讀書片的 S14 版面使用下列字型（``study.py`` 的預設名稱）：

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - 用途
     - 字型
     - 檔案（``C:/Windows/Fonts``）
   * - 中文標題與階段
     - 微軟正黑體（Microsoft JhengHei）
     - ``msjh.ttc``
   * - 日文
     - Yu Gothic
     - ``YuGothM.ttc``
   * - 計時數字
     - Consolas
     - ``consola.ttf``

缺字型時的行為依版面而定：精簡版（``study_compact: true``）以 Pillow 量測底板大小，缺檔會報錯；原生疊層 ``StudyOverlayLayer`` 缺檔會拋出 ``FileNotFoundError``。燒入的 ASS（非精簡版）只保存字型名稱，由 libass 在系統字型中解析，本模組不檢查其存在，缺字型時可能靜默替代，請以預覽確認。

三、建立新集
------------

.. code-block:: console

    python -m moviepy.ae.templates music init E:\music_projects\ep01 --mode sleep_longform

.. code-block:: console

    python -m moviepy.ae.templates music init E:\music_projects\s14 --mode study_pomodoro

``--mode`` 為必填，只接受 ``sleep_longform`` 或 ``study_pomodoro``。目標資料夾若不存在或為空，會產生：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 路徑
     - 內容
   * - ``music.json``
     - 複製自 ``configs/music_<mode>.json``，所有素材路徑為 ``PLACEHOLDER_*``，需逐一替換。
   * - ``tracks/``
     - 放音樂母帶。空資料夾。
   * - ``visual/``
     - 放 Flow 影片片段或既有循環影片。空資料夾。
   * - ``study.json``
     - 僅 ``study_pomodoro``。由 ``pomodoro_schedule()`` 產生的 5220 秒排程，需依內容改寫。
   * - ``README.txt``
     - 繁體中文製作清單與人工驗收關卡，所有關卡皆為 ``NOT_RUN``。

目標資料夾若已存在且非空，``init`` 會拒絕執行。成功時印出 ``created <music.json 的路徑>``。命令列對應的 Python 函式為 ``init_music_episode(directory, mode)``。

README 內的製作清單依序為：放入母帶並填寫 ``tracks``；放入畫面素材；（讀書片）編輯 ``study.json``；執行 ``validate``；執行 ``build --steps audio,visual,overlays``；執行 ``build --steps render --preview 0 60`` 預覽；最後執行正式輸出。

四、素材準備
------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 素材
     - 說明
   * - 曲目母帶（``tracks[].source``）
     - 建議 48 kHz 立體聲 WAV 或 FLAC。任何 FFmpeg 可讀的格式都可以；本流程輸出的母帶為 48 kHz FLAC。每首母帶會被循環延長到 ``chapter_seconds``，因此應是可無縫接回的音樂。
   * - 畫面片段（``visual.clips``）
     - Flow 生成的 8 秒片段（``clip_length`` 預設 8.0）。每個片段至少要有 ``clip_length`` 秒，只使用開頭的 ``clip_length`` 秒。每個片段成為一個場景。
   * - 既有循環（``visual.cycle_path``）
     - 已備妥的循環 MP4。給了它就不再由 ``clips`` 產生循環，只計算雜湊並寫入 ``visual/cycle.external.json``。
   * - 讀書排程（``study.json``，讀書片）
     - STUDY-MODE 文件：階段（focus、break、closing）首尾相接、三語標籤與提示、面板透明度。總長決定音訊總長；鈴聲時間取自階段邊界（``chime_times()``）。
   * - 提示音（``chime.path``，選用）
     - 立體聲 48 kHz 的現成提示音。與 ``chime.design`` 擇一。

``music.json`` 中所有相對路徑都相對於 ``music.json`` 所在資料夾解析（``from_json`` 會處理）。

五、music.json 欄位總表
-----------------------

最小範例（睡眠片；其餘欄位省略即用預設值）：

.. code-block:: json

    {
      "mode": "sleep_longform",
      "name": "ep01_sleep",
      "tracks": [
        {"id": "G01", "source": "tracks/G01.wav", "chapter_seconds": 3210, "title": "第一章"}
      ],
      "visual": {"clips": ["visual/clip_01.mp4"]}
    }

以下欄位名稱即 JSON 的鍵。「預設」欄中用 JSON 字面值（``null``、``true``、``"auto"``、``[1280, 720]``）表示；寫「必填」或說明文字者沒有預設值。未知的鍵會報 ``MusicEpisodeError``，數值欄位不接受布林值。

5.1 ``music.json`` 頂層
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 16 14 18 52

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``mode``
     - 字串
     - 必填
     - ``"sleep_longform"`` 或 ``"study_pomodoro"``。決定章節規則、是否需要 ``study``，以及頻譜與提示音的預設開關。
   * - ``name``
     - 字串
     - ``"music_episode"``
     - 檔名安全的集名，用於輸出檔名 ``render/<name>.mp4``。
   * - ``size``
     - 整數清單 ``[寬, 高]``
     - ``[1280, 720]``
     - 輸出畫面大小。寬高皆須為偶數（yuv420p）。
   * - ``fps``
     - 數字
     - ``24``
     - 輸出影格率。音訊長度與循環片段長度都必須是整數幀。
   * - ``tracks``
     - 物件清單
     - 必填
     - 音樂母帶，每首一章，至少一首。見 5.2。
   * - ``audio``
     - 物件
     - 省略即用 5.3 的預設值
     - 響度與接合參數。見 5.3。
   * - ``visual``
     - 物件
     - 必填（``clips`` 或 ``cycle_path`` 其一）
     - 畫面素材與循環參數。見 5.4。
   * - ``spectrum``
     - 物件
     - 省略即依模式決定
     - 頻譜條。見 5.5。
   * - ``study``
     - 字串或物件
     - ``null``
     - 僅 ``study_pomodoro`` 使用，且必填：``study.json`` 的路徑（相對於 ``music.json``），或內嵌的 STUDY-MODE 物件。``sleep_longform`` 必須省略此鍵。
   * - ``study_compact``
     - 布林
     - ``false``
     - ``true`` 使用精簡版字幕（深色字加半透明底板）；``false`` 為 S14 所用的三面板版面。
   * - ``chime``
     - 物件
     - 省略即依模式決定
     - 階段提示音。見 5.6。
   * - ``chapters``
     - 布林
     - ``true``
     - 是否寫出 ffmetadata 章節。睡眠片依母帶切點，讀書片依排程階段。
   * - ``encoder``
     - 字串
     - ``"auto"``
     - ``"auto"``、``"libx264"``、``"h264_nvenc"``、``"hevc_nvenc"`` 或 ``"libx265"``。``auto`` 會先做一次真實的 NVENC 測試編碼，可用才採用，否則改用 libx264。CLI 的 ``--encoder`` 可覆寫。
   * - ``quality``
     - 字串
     - ``"high"``
     - ``"high"``、``"balanced"`` 或 ``"draft"``，交由 ``parallel.encoder_args`` 決定編碼參數。
   * - ``workers``
     - 整數或 null
     - ``null``
     - 覆疊渲染與循環輸出的程序數。``null`` 表示 ``min(CPU 核心數 − 1, 12)``，至少 1。CLI 的 ``--workers`` 可覆寫。
   * - ``audio_bitrate``
     - 字串
     - ``"256k"``
     - AAC 位元率，格式須為整數加 ``k``，例如 ``"256k"``。
   * - ``output_dir``
     - 字串
     - ``"build"``
     - 產物根目錄，相對於 ``music.json`` 所在資料夾。

5.2 ``tracks[]`` 每首曲目
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 14 22 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``id``
     - 字串
     - 必填
     - 檔名安全字（如 ``S01``、``G01``），不可重複。用於 ``audio/masters/<id>.flac``。
   * - ``source``
     - 字串
     - 必填
     - 母帶檔案路徑，任何 FFmpeg 可讀的格式。
   * - ``chapter_seconds``
     - 數字
     - 睡眠片必填；讀書片可全部省略
     - 該章的準備長度（母帶循環延長到此長度）。必須至少為 ``2 × audio.chapter_crossfade``。讀書片若有一首給出，就必須每首都給。
   * - ``title``
     - 字串
     - 省略則用 ``id``
     - 睡眠片章節標題，寫入 ffmetadata。

5.3 ``audio`` 響度與接合
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``target_lufs``
     - 數字
     - ``-18.0``
     - 目標積分響度（LUFS），範圍 −70 至 −5。
   * - ``true_peak``
     - 數字
     - ``-1.8``
     - true peak 上限（dBTP），不得大於 0。
   * - ``lra``
     - 數字
     - ``11.0``
     - 響度範圍（LU），必須為正數。
   * - ``loop_overlap``
     - 數字
     - ``10.0``
     - 章內循環接縫的交叉淡化秒數。
   * - ``chapter_crossfade``
     - 數字
     - ``12.0``
     - 相鄰章節之間的交叉淡化秒數。
   * - ``edge_fade``
     - 數字
     - ``4.0``
     - 全片開頭淡入與結尾淡出的秒數，不得大於 ``chapter_crossfade``。

5.4 ``visual`` 畫面循環
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``clips``
     - 字串清單
     - 必填（或給 ``cycle_path``）
     - 來源片段，每個片段一個場景，依序排列。
   * - ``clip_length``
     - 數字
     - ``8.0``
     - 每個片段使用的秒數。必須大於 ``2 × loop_crossfade``，且 ``clip_length − loop_crossfade`` 必須是整數幀。
   * - ``loop_crossfade``
     - 數字
     - ``2.0``
     - 讓片段尾接回頭的溶接秒數。
   * - ``segment``
     - 數字
     - ``120.0``
     - 循環中每個場景佔用的秒數。必須是整數幀，且大於 ``scene_crossfade``。
   * - ``scene_crossfade``
     - 數字
     - ``2.0``
     - 相鄰場景之間的溶接秒數，循環最後一段也會溶回第一段。
   * - ``cycle_path``
     - 字串或 null
     - ``null``
     - 已備妥的循環 MP4。給了它，就只驗證並雜湊，不再產生 ``visual/loops`` 與 ``visual/cycle.mp4``。

5.5 ``spectrum`` 頻譜條
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 16 20 44

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``enabled``
     - 布林或 null
     - 依模式：讀書片開、睡眠片關
     - 是否繪製頻譜條。
   * - ``position``
     - 整數 ``[x, y]``
     - ``[96, 548]``
     - 頻譜層左上角在畫面上的位置，``x + 寬``、``y + 高`` 不得超出畫面。
   * - ``size``
     - 整數 ``[寬, 高]``
     - ``[1088, 140]``
     - 頻譜層的像素大小。
   * - ``opacity``
     - 數字
     - ``30``
     - 不透明度（百分比，0 至 100）。
   * - ``bands``
     - 整數
     - ``64``
     - 對數頻帶數。
   * - ``fft``
     - 整數
     - ``8192``
     - FFT 視窗長度，至少 16。
   * - ``fmin``
     - 數字
     - ``45.0``
     - 最低頻率（Hz）。
   * - ``fmax``
     - 數字
     - ``10000.0``
     - 最高頻率（Hz），必須大於 ``fmin``，且不超過 24000。
   * - ``bar_width``
     - 整數
     - ``12``
     - 每根柱子的寬度（像素）。
   * - ``pitch``
     - 整數
     - ``17``
     - 柱距（像素），不得小於 ``bar_width``。
   * - ``attack``
     - 數字
     - ``0.55``
     - 上升平滑係數，0 至 1。
   * - ``release``
     - 數字
     - ``0.14``
     - 下降平滑係數，0 至 1。

5.6 ``chime`` 階段提示音
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``enabled``
     - 布林或 null
     - 依模式：讀書片開、睡眠片關
     - 是否在階段邊界混入提示音。睡眠片開啟會報錯。
   * - ``below_music_db``
     - 數字
     - ``10.0``
     - 提示音比當地音樂 RMS 低多少 dB。只會降低，不會放大，也不做閃避（ducking）。
   * - ``path``
     - 字串或 null
     - ``null``
     - 現成的立體聲 48 kHz 提示音檔。與 ``design`` 擇一。
   * - ``design``
     - 物件
     - ``{}``
     - ``chime_tone`` 的參數：``notes`` （每列 ``[音名, 頻率, 起始, 長度]``）、``duration``、``attack``、``release``、``peak_dbfs``。空物件即 S14 的 D4／A4 雙音鈴聲（2.05 秒，峰值 −30 dBFS）。

5.7 兩種模式的預設值與理由
~~~~~~~~~~~~~~~~~~~~~~~~~~

下表的數值取自 ``configs/music_*.json``。理由取自 ``music_episode.py`` 模組說明。

.. list-table::
   :header-rows: 1
   :widths: 18 30 30 22

   * - 項目
     - ``sleep_longform`` （S06）
     - ``study_pomodoro`` （S14）
     - 理由
   * - 曲目與章長
     - 6 首，各 3210 秒；扣除 5 次 12 秒交叉淡化後共 19200 秒（5 小時 20 分）
     - 6 首，各 880 秒；扣除 5 次 12 秒交叉淡化後共 5220 秒，與排程相符
     - S06 的長度來自「6 × 3210 − 5 × 12」；S14 的長度來自「6 × 880 − 5 × 12」，即 3 × 25 分專注、5 分休息與 2 分收尾。
   * - 響度
     - −18 LUFS、−1.8 dBTP、LRA 11
     - 同左
     - 兩片沿用同一套 S14／S06 配方。
   * - 循環接縫與淡入淡出
     - 循環接縫 10 秒、章節交叉淡化 12 秒、邊緣淡化 4 秒
     - 同左
     - 同左。
   * - 畫面
     - 1280×720、24 fps；8 秒片段、2 秒循環溶接、120 秒段落、2 秒場景溶接
     - 同左
     - 所有影片共用單一循環，片段長與溶接長沿用 S14 的 Flow 設定。
   * - 頻譜條
     - 關閉（``enabled`` 為 false）
     - 開啟；30% 不透明度，位於 (96, 548)，層大小 1088×140
     - 睡眠片不需要頻譜；讀書片的頻譜是 S13／S14 的視覺元素。
   * - 提示音
     - 關閉
     - 開啟；比音樂低 10 dB，不放大、不閃避
     - 提示音只出現在讀書片的階段邊界，且不得蓋過音樂。
   * - 章節
     - ``chapters``：true，依母帶切點
     - ``chapters``：true，依排程階段
     - 章節讓播放器可以跳章。
   * - 字幕版面
     - 無（不燒錄字幕）
     - ``study_compact``：省略，即預設 false
     - S14 使用的是非精簡的三面板版面；精簡版屬於之後的季。
   * - 輸出
     - ``audio_bitrate`` 256k；``encoder`` auto；``quality`` high；``workers`` null
     - 同左
     - AAC 256k 與高品質為頻道的發佈設定；NVENC 可用時自動採用。

六、時長規則與常見錯誤
----------------------

6.1 時長公式
~~~~~~~~~~~~

**睡眠片**：全長 = 各章 ``chapter_seconds`` 之和 − （章數 − 1）× ``audio.chapter_crossfade``。例如六首 3210 秒、交叉淡化 12 秒：6 × 3210 − 5 × 12 = 19200 秒。

**讀書片**：全長必須等於排程長度（``study.json`` 的 ``duration_seconds``，預設範例為 5220 秒）。

* 若每首都給 ``chapter_seconds``，兩者必須相符：6 × 880 − 5 × 12 = 5220。不相符時會報 ``audio length ... != study schedule ...``，並列出應有的章長和。
* 若全部省略 ``chapter_seconds``，程式會以排程長度反推：章長總和 = 全長 + （章數 − 1）× 交叉淡化，平均分配，最後一章補足餘數。

**幀對齊**：全長（秒）乘以 ``fps`` 必須是整數幀（容差 1e-6）。例如 19200 秒 × 24 fps = 460800 幀，合格；章長若為 3210.02 秒則不合格。循環的 ``clip_length − loop_crossfade`` （預設 6 秒）與 ``segment`` （預設 120 秒）也必須是整數幀。

**章長下限**：每章至少為 ``2 × audio.chapter_crossfade`` （預設 24 秒）。

6.2 命令列驗證
~~~~~~~~~~~~~~

.. code-block:: console

    python -m moviepy.ae.templates music validate E:\music_projects\ep01\music.json

通過時印出 ``OK <name> (<mode>): 1280x720 @ 24 fps, 6 tracks, 19200 s, 460800 frames``。缺檔以 ``MISSING`` 列出，問題以 ``PROBLEM`` 列出，兩者皆使結束碼為 1。只想檢查時長、暫時沒有素材時，加上 ``--allow-missing`` 跳過檔案檢查。

6.3 常見 MusicEpisodeError 與修正
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 26 34

   * - 錯誤訊息（節錄）
     - 原因
     - 修正
   * - ``tracks: every track needs chapter_seconds``
     - 睡眠片有曲目沒給章長。
     - 每首填 ``chapter_seconds``。
   * - ``tracks: give chapter_seconds on every track or on none``
     - 讀書片只給了部分曲目的章長。
     - 全部填上，或全部省略。
   * - ``chapter_seconds must sum to duration``
     - 讀書片章長總和與排程長度不符。
     - 依 6.1 的公式調整章長，或修改 ``study.json``。
   * - ``is not a whole number of frames``
     - 全長、循環段長或預覽起訖不是整數幀。
     - 改成 ``fps`` 的整數倍（例如 24 fps 時取 1/24 秒的倍數）。
   * - ``must be at least twice``
     - 某章短於 ``2 × chapter_crossfade``。
     - 加長該章，或降低 ``chapter_crossfade``。
   * - ``edge_fade must be <= chapter_crossfade``
     - 淡入淡出比章節交叉淡化還長。
     - 縮短 ``edge_fade``。
   * - ``study is required in study_pomodoro mode``
     - 讀書片缺少 ``study``。
     - 加上 ``"study": "study.json"``。
   * - ``study is only valid in study_pomodoro mode``
     - 睡眠片含有 ``study``。
     - 刪除該鍵。
   * - ``chime.enabled needs study_pomodoro mode``
     - 睡眠片開啟提示音。
     - 設定 ``"chime": {"enabled": false}``。
   * - ``give clips or a prepared cycle_path``
     - ``visual`` 既沒有片段也沒有循環檔。
     - 填 ``clips``，或指定 ``cycle_path``。
   * - ``clip_length must exceed 2 * loop_crossfade``
     - 片段太短，無法做首尾溶接。
     - 加長 ``clip_length``，或縮短 ``loop_crossfade``。
   * - ``give a cue path or a design, not both``
     - 提示音同時給了 ``path`` 與 ``design``。
     - 只保留一個。
   * - ``spectrum: position`` 與 ``exceeds the``
     - 頻譜層超出畫面。
     - 調整 ``position`` 或 ``size``。
   * - ``unknown music episode key(s)``
     - 鍵名拼錯或不存在。
     - 對照 5.1 至 5.6 修正。
   * - ``directory exists and is not empty``
     - ``init`` 的目標資料夾非空。
     - 換一個新資料夾。

七、步驟與續跑
--------------

建置分為四個步驟，依序為 ``audio``、``visual``、``overlays``、``render``。``build`` 的 ``--steps`` 可挑選其中幾個，但仍依此順序執行。

.. list-table::
   :header-rows: 1
   :widths: 14 46 40

   * - 步驟
     - 成品（相對於 ``output_dir``）
     - 證據與檢核
   * - ``audio``
     - ``audio/masters/<id>.flac`` （每首）；``audio/music.flac``；讀書片且提示音開啟時另有 ``audio/chime.wav`` 與 ``audio/audio.flac``
     - 每個成品旁有 ``<檔名>.json``，含 SHA-256、位元組數、設定雜湊與 ``human_listening: NOT_RUN``。音訊長度必須等於計畫的影格數。
   * - ``visual``
     - ``visual/loops/loop-NN.mp4`` （每個片段）；``visual/cycle.mp4``。若給 ``cycle_path``，則為 ``visual/cycle.external.json``
     - 證據含 ``human_visual: NOT_RUN``。
   * - ``overlays``
     - ``overlays/levels.npy`` （頻譜開啟時）；``overlays/study.ass`` （讀書片）；``overlays/chapters.ffmeta`` （章節開啟時）
     - 頻譜列數必須等於音訊的影格數。需要 ``audio`` 步驟已完成。
   * - ``render``
     - ``render/<name>.mp4``，另有 ``render/<name>.mp4.json`` 與 ``render/<name>.mp4.ffmpeg.log``
     - 證據含 ``output_sha256``、``human_listening: NOT_RUN``、``human_visual: NOT_RUN``。

7.1 證據與雜湊核對
~~~~~~~~~~~~~~~~~~

所有成品路徑由 ``music_paths(spec)`` 決定。證據檔的命名規則是「成品路徑 + ``.json``」，例如 ``audio/music.flac.json``。音訊、循環與疊層步驟的證據都含 ``config_sha256``，即該步驟設定的雜湊。再次執行時：

* 成品存在、證據存在、設定雜湊相符、成品雜湊相符：**略過**。回傳的報告中 ``skipped`` 為 true（證據檔本身不會改寫）。
* 成品存在但證據不存在：報錯 ``exists without evidence``。請確認成品來源後手動刪除。
* 設定雜湊不符：報錯 ``was built from different settings or inputs``。
* 成品雜湊與證據不符（檔案被改過）：報錯 ``no longer matches its recorded hash``。

7.2 .partial 機制
~~~~~~~~~~~~~~~~~

音訊與循環先寫入 ``<stem>.partial<suffix>``，例如 ``music.partial.flac``、``loop-01.partial.mp4``。完成後才以 ``os.replace`` 改為正式檔名，因此正式檔名不會出現半成品。程式出錯時會刪除 partial；若程序被強制結束而留下 partial，下次執行會先刪除它再重建。

7.3 改設定後的重建
~~~~~~~~~~~~~~~~~~

若修改了會影響某步驟的設定，須**手動刪除該步驟的成品與證據，以及其後所有步驟的成品與證據**，再重跑。程式不會自動判斷哪些下游成品過期。

* 改 ``audio`` 相關設定（響度、交叉淡化、章長、母帶、提示音）：刪除 ``audio/`` 中受影響的成品與證據，並刪除 ``overlays/`` 與 ``render/`` 的全部成品與證據（頻譜與章節都依賴最終音訊）。
* 改 ``visual`` 相關設定或 ``encoder``、``quality``、``size``：刪除 ``visual/``，並刪除 ``render/`` 的全部成品與證據。頻譜與字幕不依賴循環，可保留 ``overlays/``。
* 改 ``render`` 相關設定（``spectrum.position``、``encoder``、``audio_bitrate``、``study_compact`` 等）：**render 步驟只比對成品雜湊，不比對設定**，因此必須手動刪除 ``render/<name>.mp4``、``.json`` 與 ``.ffmpeg.log``，否則會直接略過。

設定雜湊記錄的是來源檔的路徑與位元組數，不計算來源內容的雜湊。若替換同名同大小的母帶，程式不會察覺，請自行刪除對應成品。

八、預覽與正式輸出
------------------

8.1 指令
~~~~~~~~

.. code-block:: console

    python -m moviepy.ae.templates music build E:\music_projects\ep01\music.json --steps audio,visual,overlays

.. code-block:: console

    python -m moviepy.ae.templates music build E:\music_projects\ep01\music.json --steps render --preview 0 60

.. code-block:: console

    python -m moviepy.ae.templates music build E:\music_projects\ep01\music.json --steps render

.. code-block:: console

    python -m moviepy.ae.templates music build E:\music_projects\ep01\music.json --workers 4 --encoder libx264

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 參數
     - 說明
   * - ``--steps``
     - 預設 ``audio,visual,overlays,render``。以逗號分隔，可挑選步驟。
   * - ``--preview START SECONDS``
     - 只對 ``render`` 步驟有效：輸出 ``render/<name>-preview-<START>-<SECONDS>.mp4``，例如 ``sleep_longform_episode-preview-0-60.mp4``。``START`` 與 ``SECONDS`` 都必須是整數幀，且 START + SECONDS 不得超過全長。章節只在 START 為 0 時附上。
   * - ``--workers N``
     - 覆寫 ``workers``：覆疊渲染與循環輸出的程序數，必須為正整數。
   * - ``--encoder E``
     - 覆寫 ``encoder``，可選值同 5.1。

預覽請在 ``audio``、``visual``、``overlays`` 完成之後執行，因為 ``render`` 需要這些成品。

8.2 輸出目錄結構
~~~~~~~~~~~~~~~~

.. code-block:: text

    build/
      audio/
        masters/S01.flac            (每首母帶，含 .json 證據)
        music.flac                  (章節接合後的音樂)
        chime.wav                   (讀書片、提示音開啟)
        audio.flac                  (讀書片、混入提示音後的最終音訊)
      visual/
        loops/loop-01.mp4           (每個片段一個循環)
        cycle.mp4                   (場景循環；或 cycle.external.json)
      overlays/
        levels.npy                  (頻譜表，頻譜開啟時)
        study.ass                   (讀書片的字幕)
        chapters.ffmeta             (章節，章節開啟時)
      render/
        <name>.mp4
        <name>.mp4.json             (渲染證據，含 output_sha256)
        <name>.mp4.ffmpeg.log
        <name>-preview-0-60.mp4     (預覽時)

正式輸出的 ``render/<name>.mp4`` 不會覆寫既有檔案；若檔案、證據或日誌已存在，會拒絕執行。

8.3 Python 介面
~~~~~~~~~~~~~~~

.. code-block:: python

    from moviepy.ae.templates.music_episode import MusicEpisodeSpec, build_music_episode, validate_music_episode

    spec = MusicEpisodeSpec.from_json("E:/music_projects/ep01/music.json")
    check = validate_music_episode(spec, check_files=True)
    if check["ok"]:
        report = build_music_episode(spec, steps=("audio", "visual", "overlays"))

九、驗收
--------

技術 PASS 只代表：雜湊與證據一致、長度與幀數正確、FFmpeg 成功結束。它**不代表內容通過**。證據中的 ``human_listening`` 與 ``human_visual`` 在人工簽核前一律為 ``NOT_RUN``，程式不會自行把它改為通過。

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 人工關卡（``human_gates``）
     - 要確認的事
   * - ``full_listening``：完整試聽
     - 從頭到尾聆聽完整音訊，包括所有章節接縫、淡入淡出，以及讀書片每個鈴聲時點的音量是否合適。
   * - ``full_visual_watch``：完整畫面觀看
     - 從頭到尾觀看成片，確認循環接縫、場景溶接、字幕與頻譜條位置，以及長時間播放時的畫面重複感。
   * - ``rights``：版權與授權
     - 音樂母帶與 Flow 生成內容的授權與使用條款，須由製作者確認，並保存紀錄。
   * - ``private_upload``：私人上傳
     - 上傳時先設為不公開，完成平台端檢查後才改為公開。

以上四關為獨立關卡。預覽（60 秒）通過不代表全長通過；全長試聽與全片觀看必須另外完成。簽核紀錄由製作者自行保存，程式不會產生。

十、單一部件入口
----------------

以下部件可單獨使用，不必經過 ``MusicEpisodeSpec``。每個範例皆為最小用法，路徑請換成自己的檔案。

10.1 響度與接合（``music_audio``）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from moviepy.ae.templates.music_audio import assemble_chapters, mix_cues, normalize_loudness, write_chime

    normalize_loudness("tracks/S01.wav", "build/audio/masters/S01.flac", target_lufs=-18.0, true_peak=-1.8, lra=11.0)
    sheet = assemble_chapters(
        ["build/audio/masters/S01.flac", "build/audio/masters/S02.flac"],
        [880, 880],
        "build/audio/music.flac",
        chapter_crossfade=12.0,
        edge_fade=4.0,
        ids=["S01", "S02"],
    )
    write_chime("build/audio/chime.wav", peak_dbfs=-30.0)
    mix_cues("build/audio/music.flac", "build/audio/chime.wav", [1500, 1800], "build/audio/audio.flac", below_music_db=10.0)

``normalize_loudness`` 做兩段式線性 ``loudnorm`` 並保證精確的影格數；``assemble_chapters`` 循環並以交叉淡化接合各章；``mix_cues`` 只會降低提示音，不會放大音樂，也不做閃避。

10.2 頻譜（``spectrum``）
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from moviepy.ae.templates.spectrum import SpectrumLayer, analyze_spectrum, load_levels, save_levels

    levels = analyze_spectrum("build/audio/audio.flac", fps=24, bands=64, fft=8192, fmin=45.0, fmax=10000.0)
    save_levels("build/overlays/levels.npy", levels)
    layer = SpectrumLayer(load_levels("build/overlays/levels.npy"), fps=24, size=(1088, 140), opacity=30)

``analyze_spectrum`` 以分塊方式讀取音訊，輸出 ``(影格數, 頻帶數)`` 的表格；``SpectrumLayer`` 將表格畫成 AE 圖層。

10.3 讀書排程與字幕（``study``）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from moviepy.ae.templates.study import StudyOverlayLayer, StudySchedule

    schedule = StudySchedule.from_json("study.json")
    ass_text = schedule.to_ass(compact=False)
    chime_at = schedule.chime_times()
    layer = StudyOverlayLayer(schedule, size=(1280, 720))

``to_ass`` 產生 ``study.ass`` 的內容；``chime_times`` 回傳提示音的秒數；``StudyOverlayLayer`` 是以 AE 原生繪製的同一套疊層。

10.4 循環畫面（``visual_loop``）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from moviepy.ae.templates.visual_loop import export_loop, scene_cycle, seamless_loop

    loop = seamless_loop("visual/PLACEHOLDER_clip_01.mp4", length=8.0, crossfade=2.0, fps=24, size=(1280, 720))
    export_loop(loop, "build/visual/loops/loop-01.mp4", encoder="auto", quality="high")
    cycle = scene_cycle(
        ["build/visual/loops/loop-01.mp4", "build/visual/loops/loop-02.mp4"],
        segment=120.0,
        crossfade=2.0,
        fps=24,
        size=(1280, 720),
    )
    export_loop(cycle, "build/visual/cycle.mp4")

``seamless_loop`` 讓影片首尾相接；``scene_cycle`` 將多段循環排成一個首尾相接的循環；``export_loop`` 輸出可由 ``-stream_loop -1`` 重複的 MP4。

10.5 最終渲染（``music_render``）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from moviepy.ae.templates.music_render import frame_count, preview_window, render_music_video

    frames = frame_count(19200, 24)
    render_music_video(
        "build/render/demo.mp4",
        background="build/visual/cycle.mp4",
        audio="build/audio/audio.flac",
        fps=24,
        size=(1280, 720),
    )
    preview_window(
        "build/render/demo-preview.mp4",
        background="build/visual/cycle.mp4",
        audio="build/audio/audio.flac",
        start=0.0,
        seconds=60.0,
    )

``frame_count`` 回傳整數幀數，非整數幀時報錯；``render_music_video`` 以一次 FFmpeg 完成循環背景、疊層、字幕、AAC 音訊與章節；``preview_window`` 渲染一段時間窗。

十一、與 music_channel 舊腳本對照
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 36 30

   * - ``E:\music_channel`` 舊腳本
     - 新的 API
     - 備註
   * - ``s14-assets.py`` ``normalize()``
     - ``music_audio.normalize_loudness``
     - 兩段式線性 ``loudnorm`` 與精確影格數。
   * - ``s14-assets.py`` ``chapter()``
     - ``music_audio.loop_extend`` （單章）與 ``assemble_chapters`` （多章接合）
     - 章內循環以 ``loop_overlap`` 交叉淡化。
   * - ``s14-assets.py`` ``mix_chimes()``
     - ``music_audio.mix_cues`` （提示音由 ``write_chime`` 或 ``chime_tone`` 產生）
     - 只降低、不放大，不閃避。
   * - ``s13-spectrum.py`` ``analyze()``
     - ``spectrum.analyze_spectrum`` （再以 ``save_levels`` 存檔）
     - 已驗證：前 60 秒與 S14 的 ``fft-levels.npy`` 逐位元相同。
   * - ``s13-spectrum.py`` ``composition()``
     - ``spectrum.SpectrumLayer``
     - 同一套柱狀繪製。
   * - ``study-overlay.py``
     - ``study.StudySchedule`` （``to_ass``、``chime_times``、``chapters_ffmetadata``）與 ``study.StudyOverlayLayer``
     - 已驗證：``to_ass()`` 與 S14 的 ``study-overlay-v2.ass`` 逐字相同；``chapters_ffmetadata()`` 與 ``study-chapters.ffmeta`` 相同。
   * - ``s13-assets.py`` 與 ``s14-assets.py`` 的 ``visual()``
     - ``visual_loop.seamless_loop``、``visual_loop.scene_cycle``、``visual_loop.export_loop``
     - 循環規格與 S14 相同：8 秒片段、2 秒溶接、120 秒段落。
   * - ``s14-render.py`` ``render()``
     - ``music_render.render_music_video`` （預覽用 ``preview_window``）
     - 一次 FFmpeg 渲染；成品與證據的寫法相同。
   * - ``build-season-06.ps1``
     - ``music_episode`` 的 ``sleep_longform`` 模式（``configs/music_sleep_longform.json``）
     - 章長、交叉淡化與全長 19200 秒相同。

十二、限制
----------

* **續跑粒度**：續跑以整個成品為單位，例如一首母帶、一個循環、整條音訊、整支渲染。章內或段內無法續跑，單一大型成品中斷就要從它重做。
* **渲染中斷需重來**：``render`` 直接寫入 ``render/<name>.mp4``，沒有 ``.partial``。FFmpeg 回報錯誤時會刪除半成品；但若程序被強制結束或斷電，會留下沒有證據的 MP4，下次執行會報 ``exists without evidence``。請手動刪除該 MP4 與 ``.ffmpeg.log`` 後重跑。
* **每個 worker 載入整個頻譜表**：``spectrum_overlay`` 在每個渲染程序中以 ``np.load`` 讀入整個 ``levels.npy``。記憶體約為 ``workers × 影格數 × 64 × 4`` 位元組：87 分鐘約 32 MB／worker，10 小時約 221 MB／worker。工作數多時請注意記憶體。
* **疊層仍逐幀進入 AE**：頻譜條雖小，仍需對每個影格渲染，因此 AE 的時間與長度成正比（區域小，成本較低）。背景則不受此限制。
* **字型由 libass 解析**：非精簡版的 ASS 只保存字型名稱，本模組不檢查是否存在，缺字型時可能靜默替代。請以預覽確認。精簡版與 ``StudyOverlayLayer`` 缺字型則會報錯。
* **設定雜湊不含來源內容**：來源檔以路徑與位元組數記錄，同大小的替換不會被察覺；render 步驟不比對設定（見 7.3）。
* **不做閃避**：提示音與音樂的關係只有降低提示音，沒有閃避或包絡跟隨。
* **人工關卡不自動化**：試聽、觀看、授權與私人上傳都需要人工，程式只產生 ``NOT_RUN`` 紀錄。
* **10 小時以上未實測**：本設計的長度上限來自 FFmpeg 循環與分塊讀寫；本 repo 內的實測為 87 分鐘（S14）與 5 小時 20 分（S06 的配方）。
