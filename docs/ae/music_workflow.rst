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
   * - ``ambience``
     - 物件或 null
     - ``null``
     - 環境粒子、光色弧線與睡眠漸暗。見 5.7。省略即不加任何環境效果。
   * - ``cards``
     - 物件或 null
     - ``null``
     - 每首開頭的三語曲目卡。見 5.11。省略即不燒入曲目卡。
   * - ``study_ring``
     - 布林
     - ``false``
     - 讀書片專用：加入進度環（ASS）。睡眠片設為 ``true`` 會報錯。見 5.11。
   * - ``study_timeline``
     - 布林
     - ``false``
     - 讀書片專用：加入課程時間軸（ASS）。睡眠片設為 ``true`` 會報錯。見 5.11。
   * - ``thumbnail``
     - 物件或 null
     - ``null``
     - ``thumbnail`` 步驟（發佈用縮圖）。見 5.12。省略或 ``enabled`` 為 false 即不執行。
   * - ``qa``
     - 物件或 null
     - ``null``
     - ``qa`` 步驟（成片的自動檢查）。見 5.13。省略或 ``enabled`` 為 false 即不執行。

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
   * - ``loudness_backend``
     - 字串
     - ``"soundx"``
     - 響度後端：``soundx``（預設；先以 soundx 峰值正規化至 -4 dBFS，soundx 0.3.0 以上再由 soundx 做 LUFS 標準化，0.2.0 則改用 FFmpeg loudnorm 並記入證據）或 ``ffmpeg``（僅 FFmpeg loudnorm）。找不到 soundx 會報錯，不會自動退回。

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

5.7 ``ambience`` 環境氛圍
~~~~~~~~~~~~~~~~~~~~~~~~

``ambience`` 收納三種慢速、低刺激的效果：環境粒子（``particles``）、長時間光色弧線（``light_arc``）與睡眠漸暗（``sleep_fade``）。三者都不必由 AE 逐幀渲染：粒子是一支預先做好的短循環，光色與漸暗則是 FFmpeg 濾鏡。FFmpeg 在編碼時仍要逐幀計算濾鏡，因此長片會增加編碼時間，但不增加 AE 的負擔。所有效果都不閃爍，光色變化以餘弦曲線緩入緩出。

時間可以用負數表示「從片尾倒數」，適用於 ``light_arc`` 的關鍵影格，以及 ``sleep_fade`` 的 ``start`` 與 ``end``。例如在 19200 秒的影片中，``-1800`` 即第 17400 秒。超出 ``[0, 全長]`` 的時間會報錯。

.. code-block:: json

    {
      "ambience": {
        "particles": {"kind": "dust", "period": 20, "seed": 3},
        "light_arc": {"preset": "dusk"},
        "sleep_fade": {"start": -1800, "floor": 0.15}
      }
    }

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``particles``
     - 物件或 null
     - ``null``
     - 粒子層，見 5.8。``null`` 表示沒有粒子。
   * - ``light_arc``
     - 物件或 null
     - ``null``
     - 光色弧線，見 5.9。``null`` 表示沒有光色變化。
   * - ``sleep_fade``
     - 物件或 null
     - ``null``
     - 睡眠漸暗，見 5.10。``null`` 表示不淡出。

5.8 ``ambience.particles`` 環境粒子
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

粒子層共六種：``fireflies``（螢火蟲，緩慢飄移並輕微明滅）、``dust``（塵埃，幾乎不動）、``petals``（花瓣）、``leaves``（葉片）、``snow``（雪）與 ``rain``（細雨，半透明）。各種的預設數量與透明度都已調低，讓畫面保持安靜。

粒子層是**無縫循環**：``particle_loop`` 產生的組合恰好為 ``period`` 秒，第 ``n + period × fps`` 幀與第 ``n`` 幀完全相同。``prepare_particles`` 把它寫成帶透明通道的 ``visual/particles.mov``（無損 qtrle）。渲染時 FFmpeg 以 ``-stream_loop -1`` 重複播放並疊在背景上，接縫不會被看見。

為什麼做成短循環？5 小時 20 分共 460800 幀，若每幀都由 AE 繪製，時間與磁碟都不可行。20 秒循環只需 480 幀，渲染一次即可，之後由 FFmpeg 重複。代價是粒子的運動每 ``period`` 秒重複一次；``period`` 越長，越不容易察覺重複。螢火蟲的 ``period`` 至少為 2 秒，確保明滅不超過 0.5 Hz。

.. code-block:: json

    {
      "ambience": {
        "particles": {"kind": "fireflies", "period": 20, "count": 18, "seed": 7, "opacity": 0.6}
      }
    }

.. list-table::
   :header-rows: 1
   :widths: 20 16 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``kind``
     - 字串
     - ``"fireflies"``
     - ``fireflies``、``dust``、``petals``、``leaves``、``snow`` 或 ``rain``。
   * - ``period``
     - 數字
     - ``20.0``
     - 循環長度（秒）。``period × fps`` 必須是整數幀。螢火蟲需至少 2 秒。
   * - ``count``
     - 整數或 null
     - ``null``
     - 粒子數。``null`` 用該種的預設值，並依區域面積縮放。
   * - ``seed``
     - 整數
     - ``0``
     - 隨機種子，寫入證據。改變種子會得到不同的粒子排列。
   * - ``opacity``
     - 數字或 null
     - ``null``
     - 峰值不透明度，0 至 1。``null`` 用該種的預設值（已經很淡）。
   * - ``speed``
     - 數字
     - ``1.0``
     - 落下與飄移速度的倍率，必須為正數。
   * - ``region``
     - 整數清單 ``[x, y, 寬, 高]`` 或 null
     - ``null``
     - 限制粒子出現的區域（像素）。``null`` 為整張畫面。

5.9 ``ambience.light_arc`` 長時間光色弧線
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

光色弧線是橫跨整部影片的慢速亮度、飽和度、色溫與對比曲線。它被編譯成一條 FFmpeg ``eq`` 表達式（``LightArc.to_ffmpeg_filter``），作用範圍只有**背景**（``video_filters``）：粒子、AE 疊層與字幕都不受影響。

三種預設（``preset``）都橫跨全長：

* ``day_to_night``：明亮溫暖的白天，經黃金時刻，到微暗偏冷的夜晚。睡眠片使用此預設。
* ``dusk``：溫暖的傍晚，慢慢變暗、變柔和。
* ``dawn``：微暗偏冷的夜，逐漸轉暖、變亮成清晨。

變化速率上限為每秒 0.1（``max_rate`` 的預設值）。這是防閃爍的界線：任一參數的變化若超過此速度，``resolve_ambience`` 會拒絕，並提示 ``lengthen the segment``（加長該段）。只有短測試影片才建議調高。

自訂 ``keyframes`` 時，每列為 ``[時間, {參數: 值}]``。參數只能是 ``brightness``、``saturation``、``warmth`` 與 ``contrast``；未列出者為中性值（亮度 0、飽和 1、色溫 0、對比 1）。``preset`` 與 ``keyframes`` 不可同時給。

.. code-block:: json

    {
      "ambience": {
        "light_arc": {
          "keyframes": [
            [0, {"brightness": 0.0, "warmth": 0.2}],
            [-600, {"brightness": -0.1, "saturation": 0.8}]
          ]
        }
      }
    }

.. list-table::
   :header-rows: 1
   :widths: 20 16 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``preset``
     - 字串或 null
     - ``null``
     - ``day_to_night``、``dusk`` 或 ``dawn``。與 ``keyframes`` 擇一。
   * - ``keyframes``
     - 清單或 null
     - ``null``
     - 自訂關鍵影格，時間須嚴格遞增（負數從片尾倒數）。與 ``preset`` 擇一。
   * - ``max_rate``
     - 數字或 null
     - ``null``
     - 每秒最大變化量。``null`` 採用 0.1。只有短測試影片才建議調高。

5.10 ``ambience.sleep_fade`` 睡眠漸暗
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

睡眠漸暗把畫面與音樂慢慢降到 ``floor``，並在 ``end`` 之後保持。增益由 1 經餘弦曲線降到 ``floor``。``floor`` 為 0 是全黑與靜音；0.15 則保留微亮，讓人醒來時不會被突然的黑屏嚇到。

兩段濾鏡的範圍不同：

* **畫面**：``final_filters`` 套用在**整個合成後的畫面**，因此字幕、頻譜條與粒子也會一起變暗。
* **音樂**：``audio_filters`` 以同一個增益淡出音量（``audio`` 為 true 時）。
* 對照：``light_arc`` 的 ``video_filters`` 只作用背景。

這也是讀書片預設不開睡眠漸暗的原因：它會讓字幕與計時一起變暗。

``start`` 與 ``end`` 可用負數從片尾倒數；``end`` 必須晚於 ``start``。淡出的最大變化速率超過 ``max_rate``（``null`` 採用 0.1 每秒）時會報錯。

.. code-block:: json

    {
      "ambience": {
        "sleep_fade": {"start": -1800, "end": null, "floor": 0.15, "audio": true}
      }
    }

.. list-table::
   :header-rows: 1
   :widths: 20 16 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``start``
     - 數字
     - 必填
     - 開始淡出的秒數。負數從片尾倒數，``-1800`` 即最後 30 分鐘。
   * - ``end``
     - 數字或 null
     - ``null``
     - 到達 ``floor`` 的秒數，負數同樣從片尾倒數。``null`` 為片尾。
   * - ``floor``
     - 數字
     - ``0.0``
     - 最終增益：0 為全黑與靜音，0.15 為微亮。必須小於 1。
   * - ``audio``
     - 布林
     - ``true``
     - 音樂是否以同一增益淡出。
   * - ``max_rate``
     - 數字或 null
     - ``null``
     - 增益每秒最大變化量。``null`` 採用 0.1。

5.11 ``cards`` 曲目卡、進度環與時間軸
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

曲目卡在每首開頭顯示三語標題（``zh``、``en``、``ja``，每首至少一種），淡入約 1.5 秒、保持 ``hold`` 秒、再淡出約 2 秒。``offset`` 是章節開始後延遲的秒數。卡片是 ASS 文字，由 ``music_cards.track_cards`` 生成，再由 FFmpeg 的 ``ass`` 濾鏡燒入，不需要逐幀的 Python 運算。位置固定在四個角之一，寬度上限 540 像素。

.. code-block:: json

    {
      "cards": {
        "enabled": true,
        "position": "lower_left",
        "titles": {"G01": {"zh": "雨夜", "en": "Rain Night", "ja": "雨の夜"}}
      }
    }

上例只是片段：``enabled`` 為 true 時，每首未跳過的曲目都必須在 ``titles`` 中有對應的標題，否則報 ``cards: titles missing for track id(s)``。

讀書片（``study_pomodoro``）的版面有保留區。``music_cards.combine_study_ass`` 會避開 S14 的三面板、頻譜條 ``(96, 548, 1088, 140)``，以及時間軸與其「下一首」標籤，因此卡片不會蓋住學習資訊。進度環（``study_ring``）位於計時面板內，中心 ``(1186, 234)``；時間軸（``study_timeline``）位於 ``(600, 524, 584, 6)``。合併後，S14 原有的字幕事件逐行不變，只追加新的事件。睡眠片沒有學習面板，卡片直接由 ``sleep_cards_ass`` 生成，不需保留區。

進度環與時間軸的 ASS 文字較大：S14 排程的進度環約 1.5 MB（實測 1,483,064 位元組），時間軸約 72 KB。它們同樣是純文字，逐幀成本為零。

.. list-table::
   :header-rows: 1
   :widths: 20 16 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``enabled``
     - 布林
     - ``false``
     - 是否燒入曲目卡。
   * - ``titles``
     - 物件
     - ``{}``
     - ``{"曲目 id": {"zh": ..., "en": ..., "ja": ...}}``。每首至少一種語言，且 id 必須存在於 ``tracks``。
   * - ``position``
     - 字串
     - ``"lower_left"``
     - ``lower_left``、``lower_right``、``upper_left`` 或 ``upper_right``。
   * - ``hold``
     - 數字
     - ``8.0``
     - 卡片完全顯示的秒數。
   * - ``offset``
     - 數字
     - ``3.0``
     - 章節開始後，延遲多少秒才出現卡片。
   * - ``skip_first``
     - 布林
     - ``false``
     - 為 true 時第一首不出卡（片頭已有標題時使用）。

5.12 ``thumbnail`` 縮圖步驟
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``thumbnail`` 步驟從背景循環取一張畫面（``time`` 秒；預設為循環的三分之一處），疊上三語標題、副標與時長徽章，輸出 1280×720 的 ``publish/<output>``（預設 ``publish/thumbnail.jpg``）。排版規則如下：

* **安全區**：四周各留 5% 的邊界，文字不貼邊。文字過長時會縮小字級，最小為原字級的 40%；仍放不下就報錯，絕不截斷。
* **徽章位置**：時長徽章在**右上角**。YouTube 會在縮圖右下角疊上自己的時間戳，因此右下角不放任何需要閱讀的內容。
* **對比**：主標題與底色的 WCAG 相對亮度對比須達 4.5:1。不足時，面板會自動加深（不透明度由 0.35 逐步加到最深 0.85）；仍不足且 ``require_contrast`` 為 true 時報錯。
* **缺字**：字型缺少某個字的字形時報錯（訊息以 ``thumbnail:`` 開頭），不會以方框替代。中文預設使用 ``C:/Windows/Fonts/msjh.ttc``，日文使用 ``YuGothM.ttc``。

``duration_badge`` 為 ``"auto"`` 時，全長 19200 秒顯示為 ``5:20:00``。

.. code-block:: json

    {
      "thumbnail": {
        "enabled": true,
        "titles": {"zh": "雨夜入眠", "en": "Rain Sleep", "ja": "雨の夜"},
        "layout": "left_panel",
        "time": 60
      }
    }

.. list-table::
   :header-rows: 1
   :widths: 20 16 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``enabled``
     - 布林
     - ``false``
     - 是否執行 ``thumbnail`` 步驟。
   * - ``titles``
     - 物件
     - ``{}``
     - ``{"zh": ..., "en": ..., "ja": ...}``。啟用時 ``zh`` 必填。
   * - ``subtitle``
     - 字串或 null
     - ``null``
     - 標題下方的強調行。
   * - ``duration_badge``
     - 字串、數字或 null
     - ``"auto"``
     - ``"auto"`` 為全長的時分秒格式；字串直接使用；數字視為秒數；``null`` 不顯示徽章。
   * - ``layout``
     - 字串
     - ``"left_panel"``
     - ``left_panel``、``bottom_band`` 或 ``center``。
   * - ``time``
     - 數字或 null
     - ``null``
     - 取背景循環第幾秒的畫面。``null`` 為循環的三分之一處。
   * - ``output``
     - 字串
     - ``"thumbnail.jpg"``
     - ``publish/`` 內的檔名，副檔名須為 ``.jpg``、``.jpeg`` 或 ``.png``。
   * - ``require_contrast``
     - 布林
     - ``true``
     - 為 true 時，對比不足（且無法靠加深面板達標）即拒絕輸出。為 false 時不檢查、不加深，只記錄於證據。

5.13 ``qa`` 自動檢查
~~~~~~~~~~~~~~~~~~~~

``qa`` 步驟對最終的 MP4 做技術檢查，寫入 ``qa/<name>.qa.json``（``music_qa.qa_report``）。它檢查四項，每項給出 PASS、FAIL 或 REVIEW：

* **閃光**：近似 WCAG 2.3.1。任一秒內出現 3 次或以上的閃光即 FAIL（判定由 ``flash_verdict`` 完成）；紅色閃光不計算（``NOT_COMPUTED``）。這不是經認證的 Harding 測試。
* **響度**：以 FFmpeg ebur128 量測積分響度，須落在 -18 ± 1 LUFS；true peak 須不大於 -1.5 dBTP。這是檢查成片，不是 ``loudnorm`` 的設定值。
* **靜音**：低於 -50 dB 且持續 2 秒以上的間隙列為 REVIEW，請人工確認是否為刻意的留白。
* **削波**：任一取樣的絕對值達 0.999 即 FAIL。

整體結果（``overall``）為：任一項 FAIL 則 FAIL；否則任一項 REVIEW 則 REVIEW；其餘為 PASS。``human_listening`` 與 ``human_visual`` 固定為 ``NOT_RUN``，QA 不取代人工關卡。

QA 會解碼媒體約 4 次（閃光、響度、靜音與削波各一次），成本隨影片長度線性增加。``flash_seconds`` 可限定只分析前幾秒，適合試跑。

.. code-block:: json

    {
      "qa": {"enabled": true, "targets": {"lufs_tolerance": 0.5}, "flash_seconds": 600}
    }

.. list-table::
   :header-rows: 1
   :widths: 20 16 18 46

   * - 欄位
     - 型別
     - 預設
     - 意義
   * - ``enabled``
     - 布林
     - ``false``
     - 是否執行 ``qa`` 步驟。
   * - ``targets``
     - 物件
     - ``{}``
     - 覆寫 ``lufs``（預設 -18）、``lufs_tolerance``（預設 1.0）與 ``true_peak_max``（預設 -1.5）。
   * - ``flash_seconds``
     - 數字或 null
     - ``null``
     - 只分析閃光的前幾秒。``null`` 為全長。

5.14 兩種模式的預設值與理由
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
   * - 環境粒子（``ambience.particles``）
     - ``fireflies``：18 隻、峰值不透明度 0.6、seed 7、20 秒循環
     - 無（``particles`` 為 ``null``）
     - 睡眠片要有一點生命，螢火蟲慢而輕柔。讀書片需要靜止、專注的畫面；若想要一點動態，可改設 ``dust``。
   * - 光色弧線（``ambience.light_arc``）
     - ``day_to_night``，橫跨全長 5 小時 20 分
     - 無（``light_arc`` 為 ``null``）
     - 5 小時的緩慢變化幾乎察覺不到，適合入睡。讀書片的畫面不應隨時間改變。
   * - 睡眠漸暗（``ambience.sleep_fade``）
     - 最後 30 分鐘（``start`` 為 -1800），降至 0.15，音樂同步淡出
     - 無（``sleep_fade`` 為 ``null``）
     - 0.15 保留微光，避免突然黑屏。讀書片的字幕與計時需要持續可讀，因此不淡出。
   * - 曲目卡（``cards``）
     - 開啟；六首全部出卡；左下角；``hold`` 8 秒，``offset`` 3 秒
     - 開啟；同左
     - 讓觀眾知道目前是哪一首。標題目前為 ``PLACEHOLDER``，需逐集改寫。
   * - 進度環與時間軸（``study_ring``、``study_timeline``）
     - 關閉（均為 ``false``）
     - 開啟（均為 ``true``）
     - 進度環與時間軸是讀書片的計時資訊；睡眠片沒有計時需求，且睡眠片開啟會報錯。
   * - 縮圖（``thumbnail``）
     - 開啟；標題為 ``PLACEHOLDER``；``time`` 60 秒；徽章 ``auto``
     - 同左（標題為讀書用的 ``PLACEHOLDER``）
     - 發佈所需的封面。兩份配置都只給佔位標題，正式發佈前必須改寫。
   * - 自動 QA（``qa``）
     - 開啟
     - 開啟
     - 渲染後的技術預檢，結果為 PASS、REVIEW 或 FAIL；人工關卡仍需完成。

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
     - 對照 5.1 至 5.13 修正。
   * - ``cards: titles missing for track id(s)``
     - 曲目卡已啟用，但某首（未跳過的）沒有 ``titles``。
     - 補上該曲目 id 的 ``zh``、``en`` 或 ``ja`` 其中一種，或設 ``skip_first``。
   * - ``cards: titles for unknown track id(s)``
     - ``cards.titles`` 的鍵不存在於 ``tracks``。
     - 修正 id 拼寫，使其與 ``tracks[].id`` 相符。
   * - ``study_ring and study_timeline need study_pomodoro mode``
     - 睡眠片開啟了進度環或時間軸。
     - 刪除這兩個鍵，或設為 ``false``。
   * - ``thumbnail: titles.zh is required when enabled``
     - 縮圖已啟用，但沒有 ``zh`` 標題。
     - 補上 ``thumbnail.titles.zh``。
   * - ``ambience:``
     - 光色變化或睡眠漸暗的速率超過 ``max_rate``，或時間超出 ``[0, 全長]``（前綴可能是 ``light_arc:`` 或 ``sleep_fade:``）。
     - 加長該段、降低幅度、修正負數時間，或確認全長。
   * - ``particles:``
     - 粒子參數不合法，例如未知的 ``kind``，或 ``period × fps`` 不是整數幀。
     - 依 5.8 修正。
   * - ``thumbnail:``
     - 縮圖排版、缺字或對比不足（後接原因）。
     - 縮短標題、改用較大的 ``layout``、補齊字型，或調整背景 ``time``。
   * - ``qa:``
     - QA 步驟無法完成，例如 FFmpeg 無法解碼成片。
     - 先確認成片可以播放，再重跑 ``qa`` 步驟。
   * - ``belongs to a different render or QA settings``
     - ``qa/<name>.qa.json`` 屬於另一支成片，或另一組 ``targets`` 與 ``flash_seconds``。
     - 刪除該報告後重跑 ``qa`` 步驟。
   * - ``directory exists and is not empty``
     - ``init`` 的目標資料夾非空。
     - 換一個新資料夾。

七、步驟與續跑
--------------

建置分為六個步驟，依序為 ``audio``、``visual``、``overlays``、``render``、``thumbnail``、``qa``。``build`` 的 ``--steps`` 可挑選其中幾個，但仍依此順序執行。``thumbnail`` 與 ``qa`` 預設關閉：只有在 ``music.json`` 中設為 ``enabled: true``，不指定 ``--steps`` 時才會執行。若明確列出而未啟用，報告中該步驟會標示 ``skipped``。

每個步驟寫出的成品與證據如下。

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
     - ``visual/loops/loop-NN.mp4`` （每個片段）；``visual/cycle.mp4``。若給 ``cycle_path``，則為 ``visual/cycle.external.json``。啟用粒子時另有 ``visual/particles.mov``
     - 證據含 ``human_visual: NOT_RUN``。粒子循環的證據記錄種類、種子與所有選項。
   * - ``overlays``
     - ``overlays/levels.npy`` （頻譜開啟時）；``overlays/study.ass`` （讀書片，含進度環、時間軸與曲目卡）；``overlays/cards.ass`` （睡眠片且曲目卡開啟時）；``overlays/chapters.ffmeta`` （章節開啟時）
     - 頻譜列數必須等於音訊的影格數。需要 ``audio`` 步驟已完成。ASS 與章節的設定雜湊取自文字內容，因此改動任何卡片文字都會被偵測。
   * - ``render``
     - ``render/<name>.mp4``，另有 ``render/<name>.mp4.json`` 與 ``render/<name>.mp4.ffmpeg.log``。粒子疊層、光色弧線與睡眠漸暗都在此步驟套用。
     - 證據含 ``output_sha256``、``config_sha256``、``human_listening: NOT_RUN``、``human_visual: NOT_RUN``，並以 ``ambience`` 欄記錄實際使用的濾鏡與解析後的時間。
   * - ``thumbnail``
     - ``publish/<output>`` （預設 ``publish/thumbnail.jpg``）
     - 證據為 ``publish/thumbnail.jpg.json``，含文字框、徽章框、安全區、字型、對比度與背景取樣時間，以及 ``human_visual: NOT_RUN``。
   * - ``qa``
     - ``qa/<name>.qa.json``
     - 報告本身即證據：記錄 ``render_sha256``、設定雜湊與 ``overall``（PASS、REVIEW 或 FAIL）；``human_listening`` 與 ``human_visual`` 為 ``NOT_RUN``。

7.1 證據與雜湊核對
~~~~~~~~~~~~~~~~~~

所有成品路徑由 ``music_paths(spec)`` 決定。證據檔的命名規則是「成品路徑 + ``.json``」，例如 ``audio/music.flac.json``。音訊、循環、粒子、疊層、縮圖與 render 步驟的證據都含 ``config_sha256``，即該步驟設定的雜湊；QA 報告則記錄 ``config_sha256`` 與 ``render_sha256``。再次執行時：

* 成品存在、證據存在、設定雜湊相符、成品雜湊相符：**略過**。回傳的報告中 ``skipped`` 為 true（證據檔本身不會改寫）。
* 成品存在但證據不存在：報錯 ``exists without evidence``。請確認成品來源後手動刪除。
* 設定雜湊不符：報錯 ``was built from different settings or inputs``。
* 成品雜湊與證據不符（檔案被改過）：報錯 ``no longer matches its recorded hash``。

7.2 .partial 機制
~~~~~~~~~~~~~~~~~

音訊、循環、粒子、疊層、縮圖都先寫入 ``<stem>.partial<suffix>``，例如 ``music.partial.flac``、``loop-01.partial.mp4``、``particles.partial.mov``、``thumbnail.partial.jpg``。完成後才以 ``os.replace`` 改為正式檔名，因此正式檔名不會出現半成品。程式出錯時會刪除 partial；若程序被強制結束而留下 partial，下次執行會先刪除它再重建。

``render`` 與 ``qa`` 不使用 partial：``render`` 直接寫入 MP4（見十二），``qa`` 報告以獨占方式寫入。若寫入中途斷電，請手動刪除不完整的檔案後重跑。

7.3 改設定後的重建
~~~~~~~~~~~~~~~~~~

若修改了會影響某步驟的設定，須**手動刪除該步驟的成品與證據，以及其後所有步驟的成品與證據**，再重跑。程式不會自動判斷哪些下游成品過期。

* 改 ``audio`` 相關設定（響度、交叉淡化、章長、母帶、提示音）：刪除 ``audio/`` 中受影響的成品與證據，並刪除 ``overlays/`` 與 ``render/`` 的全部成品與證據（頻譜與章節都依賴最終音訊）。
* 改 ``visual`` 相關設定或 ``encoder``、``quality``、``size``：刪除 ``visual/``，並刪除 ``render/`` 的全部成品與證據。頻譜與字幕不依賴循環，可保留 ``overlays/``。
* 改 ``ambience``：光色弧線與睡眠漸暗只影響 render，刪除 ``render/`` 的全部成品與證據即可。若改的是粒子（``kind``、``period``、``count``、``seed`` 等），還要刪除 ``visual/particles.mov`` 與其證據，因為它的設定雜湊已記錄在證據中。
* 改 ``cards``、``study_ring``、``study_timeline`` 或讀書片的字幕：刪除 ``overlays/cards.ass`` 或 ``overlays/study.ass`` 與其證據，並刪除 ``render/`` 的全部成品與證據。
* 改 ``thumbnail``：刪除 ``publish/`` 中的縮圖與其證據（``thumbnail.jpg`` 與 ``thumbnail.jpg.json``）。
* 改 ``render`` 相關設定（``spectrum.position``、``encoder``、``audio_bitrate``、``study_compact`` 等）：render 步驟會比對設定雜湊，設定不符即報錯 ``was rendered from different settings``，但不會自動重算。必須手動刪除 ``render/<name>.mp4``、``.json`` 與 ``.ffmpeg.log`` 後重跑。
* 改 ``qa``（``targets``、``flash_seconds``）或重新渲染：刪除 ``qa/<name>.qa.json``。報告記錄了成片的 ``render_sha256``，成片改變時會報 ``belongs to a different render``。

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
     - 以逗號分隔，可挑選步驟，例如 ``--steps thumbnail,qa``。不指定時的預設為 ``audio,visual,overlays,render``，並加上已啟用的 ``thumbnail`` 與 ``qa``。
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
        particles.mov               (環境粒子，啟用時)
      overlays/
        levels.npy                  (頻譜表，頻譜開啟時)
        study.ass                   (讀書片的字幕，含進度環、時間軸與曲目卡)
        cards.ass                   (睡眠片的曲目卡，啟用時)
        chapters.ffmeta             (章節，章節開啟時)
      render/
        <name>.mp4
        <name>.mp4.json             (渲染證據，含 output_sha256)
        <name>.mp4.ffmpeg.log
        <name>-preview-0-60.mp4     (預覽時)
      publish/
        thumbnail.jpg               (縮圖，thumbnail 啟用時；證據為 thumbnail.jpg.json)
      qa/
        <name>.qa.json              (自動檢查報告，qa 啟用時)

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

``qa`` 步驟的結果是機器預檢，不是驗收。``overall`` 為 PASS 也不代表聽感或畫面通過；REVIEW（例如靜音間隙）要由人工確認是否為刻意；FAIL 則須修正後重做，不得以人工略過。

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

10.6 環境粒子與漸暗（``ambience``）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from moviepy.ae.templates.ambience import LightArc, SleepFade, export_overlay_loop, particle_loop

    loop = particle_loop("fireflies", size=(1280, 720), period=20.0, fps=24, count=18, seed=7, opacity=0.6)
    export_overlay_loop(loop, "build/visual/particles.mov")
    video_filter = LightArc.day_to_night(19200).to_ffmpeg_filter()
    fade = SleepFade(17400, 19200, floor=0.15, audio=True)
    final_filter, audio_filter = fade.video_filter(), fade.audio_filter()

``particle_loop`` 與 ``export_overlay_loop`` 即 ``prepare_particles`` 所用的函式。``LightArc.day_to_night`` 等三個預設各回傳一個 ``LightArc``；``SleepFade.video_filter`` 與 ``audio_filter`` 回傳的 FFmpeg 字串，分別供 ``render_music_video`` 的 ``final_filters`` 與 ``audio_filters`` 使用。``resolve_ambience`` 會把 ``music.json`` 中的負數時間解析為這些數值，並報告實際使用的濾鏡。

10.7 縮圖、曲目卡與 QA（``thumbnail``、``music_cards``、``music_qa``）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from moviepy.ae.templates.music_qa import qa_report
    from moviepy.ae.templates.thumbnail import ThumbnailSpec, grab_frame, render_thumbnail

    spec = ThumbnailSpec(titles={"zh": "雨夜入眠", "en": "Rain Sleep", "ja": "雨の夜"}, duration_badge="5:20:00", layout="left_panel")
    frame = grab_frame("build/visual/cycle.mp4", 60)
    render_thumbnail(spec, frame, "build/publish/thumbnail.jpg", size=(1280, 720))
    report = qa_report("build/render/sleep_longform_episode.mp4", output_json="build/qa/sleep_longform_episode.qa.json")
    print(report["overall"], report["verdicts"])

``render_thumbnail`` 在面板未達 4.5:1 時會逐步加深面板，直到通過或用盡步數；成功時 ``panel["auto_steps"]`` 記錄嘗試過的不透明度。``qa_report`` 寫出的報告與 ``qa`` 步驟的報告內容相同，只是 ``qa`` 步驟另外記錄 ``render_sha256`` 與設定雜湊；``output_json`` 以獨占方式建立，既有檔案會拒絕覆寫。曲目卡則由 ``music_cards.sleep_cards_ass`` （睡眠片）或 ``music_cards.combine_study_ass`` （讀書片，可合併進度環與時間軸）生成 ASS 文字。``make_thumbnail`` 與 ``run_qa`` 則是 ``thumbnail`` 與 ``qa`` 步驟的 Python 函式，``build_music_episode`` 會依序呼叫它們。

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
* **光色弧線只作用背景**：``light_arc`` 經 ``video_filters`` 套用，粒子層、AE 疊層與字幕都不受影響。要讓字幕與頻譜一起變化，只有睡眠漸暗的 ``final_filters`` 會這樣做。
* **睡眠漸暗假設 limited range**：``SleepFade`` 的公式以 limited range（亮度碼 16 至 235）為前提。來源若為 full range，漸暗後的亮度會偏離預期；本 repo 未針對此情況實測。
* **粒子循環有週期**：粒子每 ``period`` 秒重複一次。長時間觀看時，節奏可能被察覺；較長的 ``period`` 與不同的 ``seed`` 可降低此風險。
* **進度環的 ASS 體積大**：S14 排程的 ``study_ring`` 約產生 1.5 MB 的 ASS（實測 1,483,064 位元組）。本 repo 未在播放器中實測其解析與燒入時間。
* **QA 是預檢，不是認證測試**：閃光檢查是 WCAG 2.3.1 的近似，不是認證的 Harding 測試。紅色閃光未計算，小於縮小取樣尺寸的閃光可能漏檢；靜音與削波的結論也須由人工確認。
* **QA 的成本隨長度增加**：媒體被解碼約 4 次（閃光、響度、靜音與削波）。本 repo 未量測整段耗時；試跑時可用 ``flash_seconds`` 限縮閃光分析的範圍。
* **縮圖的字型為 Windows 路徑**：``thumbnail`` 步驟使用 ``C:/Windows/Fonts`` 的字型，music.json 目前沒有字型欄位。在其他系統上需要修改 ``thumbnail.DEFAULT_FONTS``。
* **10 小時以上未實測**：本設計的長度上限來自 FFmpeg 循環與分塊讀寫；本 repo 內的實測為 87 分鐘（S14）與 5 小時 20 分（S06 的配方）。
