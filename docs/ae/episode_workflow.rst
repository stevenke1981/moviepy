:orphan:

新集影片製作流程（``moviepy.ae.templates`` episode）
====================================================

本文說明如何用 ``moviepy.ae.templates`` 製作一集全新的影片：從建立專案資料夾、準備素材、撰寫 ``episode.json``，到驗證、預覽與正式輸出。流程對應「夜燈說書」（``nightlamp_story``，故事頻道）與「夜燈史話」（``nightlamp_history``，史料頻道）兩套設定。

所有欄位名稱、指令參數與預設值皆取自 ``moviepy/ae/templates/episode.py``、``configs/*.json`` 與各模板模組。若程式碼改變，請以原始碼為準並同步更新本文。

一、環境
--------

Python 套件
~~~~~~~~~~~

本 repo 的 ``moviepy`` 必須優先於 site-packages 中的舊版。若 site-packages 有舊版，直接執行 ``python -m moviepy.ae.templates`` 可能會載入舊版而找不到 episode 模組。兩種做法擇一：

.. code-block:: console

    pip install -e E:\moviepy_ae

.. code-block:: console

    $env:PYTHONPATH = "E:\moviepy_ae"
    cd E:\moviepy_ae

``pip install -e`` 為永久設定；``PYTHONPATH`` 只對目前的 PowerShell 視窗有效。

字型
~~~~

預設字型為思源體：標題用思源宋體 ``SourceHanSerifTW-Bold.otf``，內文與燒錄字幕用思源黑體 ``SourceHanSansTW-Bold.otf``，史話引文用 ``SourceHanSerifTW-SemiBold.otf``。字型從 Adobe 官方 GitHub（``adobe-fonts/source-han-serif``、``adobe-fonts/source-han-sans`` 的 TW 子集）下載，安裝到 ``C:/Windows/Fonts`` 或使用者字型資料夾 ``%LOCALAPPDATA%/Microsoft/Windows/Fonts`` 皆可；``presets.source_han_font`` 會在這些資料夾尋找。模板不會在字型缺失或缺字時換成其他字型：

* 字型檔不存在：拋出 ``FileNotFoundError``（CLI 會顯示 ``ERROR:`` 並以結束碼 1 離開）。
* 字卡文字含有字型沒有的字：``title_card`` 會拋出 ``ValueError``，訊息為 ``font has no glyph for '…'``。請換字、改字，或在 ``fonts`` 指定另一個完整涵蓋的字型檔，不要期待自動替換。

FFmpeg（影片編碼）
~~~~~~~~~~~~~~~~~~

``render`` 使用 moviepy 的 ``FFMPEG_BINARY`` 編碼 ``episode.mp4``；``--master`` 另外呼叫 FFmpeg 寫入母帶。預設值 ``ffmpeg-imageio`` 會使用 imageio-ffmpeg 內附的執行檔。若要改用系統 PATH 上的 ``ffmpeg``，請設定環境變數 ``FFMPEG_BINARY=auto-detect``，或直接指向一個 ``ffmpeg.exe`` 路徑。母帶失敗時，輸出目錄內會留下 ``ffmpeg.log`` 供除錯。

二、建立新集
------------

.. code-block:: console

    python -m moviepy.ae.templates init E:\episodes\ep012 --channel history

``--channel`` 只接受 ``history`` 或 ``story``（必填）。目標資料夾若不存在或為空，會產生：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 路徑
     - 內容
   * - ``episode.json``
     - 複製自 ``configs/nightlamp_<channel>.json``，所有素材路徑為 ``PLACEHOLDER_*``，需逐一替換。
   * - ``media/``
     - 放圖片、背景、旁白與配樂。空資料夾。
   * - ``subtitles/``
     - 放 SRT 字幕。空資料夾。
   * - ``README.txt``
     - 繁體中文製作清單，與本文的步驟相同。

目標資料夾若已存在且非空，``init`` 會拒絕執行。

三、準備素材
------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 素材
     - 說明
   * - 圖片（``shots``）
     - 放入 ``media/``。圖片需足夠大，Ken Burns 會放大到覆蓋 1920×1080 畫面，解析度不足時放大後會模糊。
   * - 影片（``shots``，選用）
     - 以 ``video`` 指定，可用 ``in`` / ``out`` 裁切；``audio=False`` 讀取，影片原聲不會進入成品。
   * - 背景（``background``，選用）
     - 字卡底下的媒體，建議與第一張畫面同一素材。遮罩深淺由 preset 或 ``overlay_opacity`` 決定。
   * - 旁白（``audio.narration``）
     - 建議**立體聲** WAV。``AudioFileClip`` 會把單聲道轉成立體聲並降低約 3 dB，成品音量會偏小；立體聲可避免此情形。取樣率以 48 kHz 較為合適，與母帶音訊一致（非必要）。
   * - 配樂（``audio.music``，選用）
     - 固定以 ``music_gain_db`` 壓低為背景床，並在旁白期間再降 ``music_duck_db``。它不會跟隨旁白的音量起伏（見第十節）。
   * - 字幕（``subtitles``）
     - 建議 zh-TW 與 en 各一份 SRT（UTF-8，BOM 與 CRLF 皆可）。時間軸以成品時間為準，且不得超出時間軸。

所有相對路徑都相對於 ``episode.json`` 所在資料夾解析。

四、episode.json 欄位總表
-------------------------

以下欄位名稱即 JSON 的鍵。型別欄中「物件」代表 JSON object，「清單」代表 JSON array。除非註明，數值欄位皆為數字（JSON number），不接受布林值。未知的鍵會報 ``EpisodeError``；必填欄位缺漏時亦會報錯。

4.1 ``episode.json`` 頂層
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 14 20 48

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``name``
     - 字串
     - ``"episode"``
     - 集名，寫入 ``Composition`` 名稱與 ``report.json``。
   * - ``preset``
     - 字串
     - ``"nightlamp_history"``
     - 頻道預設值名稱，只能是 ``nightlamp_story`` 或 ``nightlamp_history``。
   * - ``preset_overrides``
     - 物件
     - ``{}``
     - 覆寫預設值，見 4.2。
   * - ``fonts``
     - 物件
     - ``{}``
     - 各角色字型路徑，見 4.2。
   * - ``background``
     - 物件或 null
     - ``null``
     - 字卡底下的媒體，見 4.3。null 時字卡坐在 ``dip_color`` 上。
   * - ``intro``
     - 物件或 null
     - ``null``
     - 片頭字卡，見 4.4。
   * - ``outro``
     - 物件或 null
     - ``null``
     - 片尾字卡，見 4.4。
   * - ``shots``
     - 清單
     - 必填
     - 畫面段，依時間順序，至少一個，見 4.5。
   * - ``chapters``
     - 清單
     - ``[]``
     - 左上角章節標籤，見 4.6。
   * - ``quotes``
     - 清單
     - ``[]``
     - 直式引文，見 4.7。
   * - ``subtitles``
     - 物件或 null
     - ``null``
     - 燒錄字幕，見 4.8。null 表示不燒字幕。
   * - ``audio``
     - 物件或 null
     - ``null``
     - 旁白與配樂，見 4.9。null 表示成品無聲。
   * - ``scene_overlay``
     - 物件或 null
     - ``null``
     - 章節疊圖套組（logo、浮水印、直式章名、訂閱鈕），見 4.10。
   * - ``name_tags``
     - 清單
     - ``[]``
     - 人物名牌，見 4.11。
   * - ``title_overlay``
     - 物件或 null
     - null
     - 疊在畫面上的透明片頭標題，見 4.12。
   * - ``source_inserts``
     - 清單
     - ``[]``
     - 史料圖插入與角落引用，見 4.13。
   * - ``end_card``
     - 物件或 null
     - null
     - 片尾「按讚・訂閱・分享」輪播，見 4.14。
   * - ``chapter_period``
     - 數字
     - ``5.5``
     - 章節標籤每個項目停留秒數，須大於 0。單一章節可用其 ``period`` 覆寫。
   * - ``dip_color``
     - 清單（3 個數字）
     - ``[0, 0, 0]``
     - 畫面底色，也是 ``dip`` 轉場的顏色，每個值為 0–255。
   * - ``seed``
     - 整數
     - ``7``
     - 章節紙紋與引文竹簡紋理的亂數種子。同一種子可重現同一畫面。

4.2 ``preset_overrides`` 與 ``fonts``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``preset_overrides`` 的鍵即 ``ChannelPreset`` 的欄位，值會經過驗證後套用：

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 鍵
     - 說明
   * - ``size``
     - ``[寬, 高]``，改變尺寸時安全邊距（``safe_margin``）會依比例縮放。
   * - ``fps``
     - 成品幀率，須為正數（範圍 1–240）。兩套 config 都設為 24。
   * - ``safe_margin``
     - ``[x, y]`` 安全邊距（像素，以 1080 高為準）。
   * - ``overlay_color``
     - ``[r, g, b]`` 遮罩顏色。
   * - ``overlay_opacity``
     - 0–1，字卡背景遮罩的不透明度。
   * - ``fade``
     - 秒，轉場預設長度（``transition_duration`` 為 null 時採用）。
   * - ``hold``
     - 秒，Ken Burns 靜圖的聚焦停留時間（``shot.hold`` 為 null 時採用）。
   * - ``palette``
     - 物件，配色角色名到 ``#RRGGBB``，與 preset 的 ``palette`` **合併**。
   * - ``fonts``
     - 物件，字型角色到路徑，與 preset 的 ``fonts`` **合併**。
   * - ``subtitles``
     - 物件，覆寫字幕樣式（``SubtitleStyle``），例如 ``primary_size``（預設 72）、``secondary_size``（預設 42）。

``fonts`` 的鍵只能是 ``title``、``body``、``quote``、``chapter``、``subtitle``。值為字型檔路徑；``null`` 只允許用於 ``title``、``body`` 與 ``subtitle``（使用 Pillow 預設字型，僅適合拉丁字母），``quote`` 與 ``chapter`` 必須指定真實字型檔。未指定的角色沿用 preset 的字型。

4.3 ``background``（背景媒體）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 14 20 48

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``source``
     - 字串
     - 必填
     - 圖片或影片路徑。影片取單一格靜止保持。
   * - ``frame_time``
     - 數字或 null
     - ``null``
     - 影片素材取哪一秒的影格（≥0）。null 時取第 0 秒。
   * - ``focal_point``
     - 清單（2 個數字）
     - ``[0.5, 0.5]``
     - 裁切焦點，兩值皆在 0–1 之間。
   * - ``overlay_opacity``
     - 數字或 null
     - ``null``
     - 遮罩不透明度（0–1）。null 採用 preset 值。

4.4 ``intro`` 與 ``outro``（字卡）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

字卡欄位與 ``TitleCardSpec`` 的文字與時序欄位相同，並多出 ``transition`` 與 ``transition_duration``。時序的預設值即夜燈說書 R23 的進場節奏。

.. list-table::
   :header-rows: 1
   :widths: 24 16 18 42

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``title``
     - 字串
     - 必填
     - 主標題，不可為空字串。
   * - ``brand``
     - 字串
     - ``""``
     - 品牌字，例如頻道名稱。
   * - ``subtitle``
     - 字串
     - ``""``
     - 副標題。
   * - ``layout``
     - ``"right_column"``、``"left_column"`` 或 ``"center"``
     - ``"right_column"``
     - 版面配置。
   * - ``duration``
     - 數字（>0）
     - ``5.0``
     - 字卡秒數。``auto_extend`` 為 true 時，可能會為了閱讀時間而延長。
   * - ``fade_out``
     - 數字（≥0）
     - ``0.35``
     - 字卡結尾淡出秒數。
   * - ``rule``
     - 布林
     - ``true``
     - 是否畫分隔線。
   * - ``auto_extend``
     - 布林
     - ``true``
     - 依文字量自動延長字卡，以保證有足夠閱讀時間。
   * - ``title_keep_together``
     - 字串清單
     - ``[]``
     - 標題中不得被斷開的詞。
   * - ``subtitle_keep_together``
     - 字串清單
     - ``[]``
     - 副標題中不得被斷開的詞。
   * - ``transition``
     - ``"cut"``、``"crossfade"`` 或 ``"dip"``
     - ``"cut"``
     - 接到前一段的方式。片頭沒有前一段，必須為 ``cut``。
   * - ``transition_duration``
     - 數字（>0）或 null
     - ``null``
     - 轉場秒數。``cut`` 不可設定；null 時採用 ``preset.fade``。
   * - ``brand_start`` / ``brand_duration``
     - 數字（≥0）
     - ``0.2`` / ``0.55``
     - 品牌字進場的起點與長度（秒，相對字卡開始）。
   * - ``title_start`` / ``title_duration``
     - 數字（≥0）
     - ``0.55`` / ``0.85``
     - 主標題進場的起點與長度。
   * - ``subtitle_start`` / ``subtitle_duration``
     - 數字（≥0）
     - ``1.05`` / ``0.65``
     - 副標題進場的起點與長度。

4.5 ``shots``（畫面段）
~~~~~~~~~~~~~~~~~~~~~~

每個 shot 必須**恰好**指定 ``image`` 或 ``video`` 其中之一。

.. list-table::
   :header-rows: 1
   :widths: 18 14 20 48

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``image``
     - 字串
     - 無
     - 圖片路徑。圖片 shot 必須有 ``duration``。
   * - ``video``
     - 字串
     - 無
     - 影片路徑。
   * - ``duration``
     - 數字（>0）
     - 圖片必填；影片可省略
     - 此段秒數。影片 shot 省略時，為 ``out`` 減 ``in``，或直到檔案結尾。
   * - ``move``
     - 字串
     - ``"auto"``
     - 圖片的運鏡：``auto``、``static``、``push``、``pull``、``pan-left``、``pan-right``、``pan-up``、``pan-down``、``drift-left``、``drift-right``。``drift-*`` 為武則天 r2b 的「左右移動同時放大、停在 ``focus`` 主體上再切換」（放大 1.02→1.10，左右各 3.5%），建議 ``hold`` 1.15。``auto`` 會依序輪替，使相鄰的靜圖不重複同一動作。僅限圖片。
   * - ``zoom``
     - 數字或 ``[起, 終]``
     - null
     - 縮放倍率，每個值 ≥1。``push``／``pull`` 用 ``[起, 終]`` 一對，pan 用單一數字。null 採用 ``ken_burns`` 預設值。僅限圖片。
   * - ``focus``
     - ``[x, y]``
     - null
     - 主體焦點，占來源圖寬高的比例（0–1）。僅限圖片。
   * - ``distance``
     - 數字（>0）
     - null
     - 平移距離，占輸出寬度的比例。僅 pan 運鏡使用。
   * - ``hold``
     - 數字（≥0）
     - null
     - 運鏡結束後保持最終構圖的秒數。null 採用 ``preset.hold``。``static`` 不可設定 ``zoom``、``focus`` 或 ``distance``。僅限圖片。
   * - ``lead``
     - 數字（≥0）
     - null
     - 運鏡開始前保持靜止的秒數。null 等同 0。僅限圖片。
   * - ``segment``
     - ``[a, b]``（0≤a<b≤1）或 null
     - null
     - 僅限 ``drift-left`` / ``drift-right``。同一張圖拆成數段時，各段取整個運鏡的一段（例如 ``[0, 0.5]`` 接 ``[0.5, 1]``），畫面連續不跳。
   * - ``in``
     - 數字（≥0）
     - null
     - 影片裁切起點（秒）。僅限影片。JSON 鍵為 ``in``，對應程式欄位 ``clip_in``。
   * - ``out``
     - 數字（>0）
     - null
     - 影片裁切終點（秒），必須大於 ``in``。僅限影片。JSON 鍵為 ``out``，對應程式欄位 ``clip_out``。
   * - ``speed``
     - 數字（>0）或 null
     - null
     - 影片播放速率。``0.75`` 把 6 秒素材放慢成 8 秒；未給 ``duration`` 時，秒數為 ``(out - in) / speed``。僅限影片。
   * - ``freeze_at``
     - 數字或 null
     - null
     - 素材秒數，到達後停在該格直到此段結束；用於素材尾段走樣時停在最佳畫格。須晚於 ``in``、不晚於 ``out``。僅限影片。
   * - ``transition``
     - ``"cut"``、``"crossfade"`` 或 ``"dip"``
     - ``"cut"``
     - 接到前一段的方式。第一段且無片頭時必須為 ``cut``。
   * - ``transition_duration``
     - 數字（>0）或 null
     - null
     - 轉場秒數。null 採用 ``preset.fade``。

4.6 ``chapters``（章節標籤）
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 14 20 48

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``start``
     - 數字（≥0）
     - 必填
     - 成品時間軸上的開始秒數（**絕對時間**，已含片頭）。
   * - ``end``
     - 數字
     - 必填
     - 結束秒數，必須大於 ``start``。
   * - ``items``
     - 字串清單（非空）
     - 必填
     - 依序輪播的項目。也可只給一個字串。
   * - ``number``
     - 整數（≥1）
     - 必填
     - 章節號碼。
   * - ``period``
     - 數字（>0）或 null
     - null
     - 此章節每個項目停留秒數。null 採用頂層 ``chapter_period``。
   * - ``seal``
     - 字串或 null
     - null
     - 朱印文字。
   * - ``position``
     - ``"top_left"`` 或 ``"bottom_left"``
     - ``"top_left"``
     - 標籤位置。
   * - ``repeat_every``
     - 數字（>0）或 null
     - null
     - 設定後標籤不再整章常駐，而是在 ``start`` 及之後每隔此秒數滑入一次（武則天 r2b 為 38 秒）。章末不足 2 秒的出現會略過。
   * - ``visible``
     - 數字（>0）或 null
     - null
     - 每次出現的秒數，null 為 7.4 秒。需搭配 ``repeat_every``，且不得大於它。

4.7 ``quotes``（直式引文）
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 14 20 48

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``start``
     - 數字（≥0）
     - 必填
     - 開始秒數（絕對時間）。
   * - ``text``
     - 字串
     - 必填
     - 引文內容，不可為空。
   * - ``source``
     - 字串或 null
     - null
     - 出處，例如 ``《周易》``。
   * - ``dynasty``
     - 字串或 null
     - null
     - 朝代，用來決定竹簡或宣紙（``resolve_medium``）。
   * - ``year``
     - 整數或 null
     - null
     - 年份，同樣用於決定媒介。
   * - ``medium``
     - 字串
     - ``"auto"``
     - 強制媒介；``auto`` 依 ``dynasty`` 或 ``year`` 判定。
   * - ``title``
     - 字串或 null
     - null
     - 引文標題。
   * - ``seal``
     - 字串或 null
     - null
     - 朱印文字。
   * - ``per_column``
     - 整數（≥1）
     - ``10``
     - 每直行字數。
   * - ``duration``
     - 數字（>0）或 null
     - null
     - 引文顯示秒數。null 採用模板計算的長度。
   * - ``layout``
     - ``"center"``、``"right"`` 或 ``"left"``
     - ``"center"``
     - 版面位置。
   * - ``size``
     - 數字（>0）
     - ``62``
     - 字級。
   * - ``speed``
     - 數字（>0）
     - ``9.0``
     - 每秒寫入的字數。
   * - ``hold``
     - 數字（≥0）
     - ``3.0``
     - 寫完後停留的秒數。
   * - ``fade_out``
     - 數字（≥0）
     - ``0.6``
     - 結尾淡出秒數。
   * - ``dim``
     - 數字（0–1）
     - ``0.35``
     - 背景壓暗程度（0–1；0 表示不壓暗）。
   * - ``seed``
     - 整數
     - ``7``
     - 紋理亂數種子。

引文不可與片頭、片尾字卡重疊，引文之間也不可重疊。

4.8 ``subtitles``（燒錄字幕）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 16 18 44

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``primary``
     - SRT 路徑或 cue 清單
     - 必填
     - 主字幕（通常為 zh-TW）。cue 清單的每項為 ``{"start", "end", "text"[, "lang"]}``。
   * - ``secondary``
     - SRT 路徑、cue 清單或 null
     - null
     - 副字幕（通常為 en），位於主字幕下方。
   * - ``protected``
     - 字串清單
     - ``[]``
     - 不得被拆到兩行的詞。
   * - ``primary_lang``
     - 字串
     - ``"zh-TW"``
     - 主字幕語言代碼，影響斷行規則。
   * - ``secondary_lang``
     - 字串
     - ``"en"``
     - 副字幕語言代碼。
   * - ``overflow``
     - ``"wrap"`` 或 ``"split"``
     - ``"wrap"``
     - ``wrap``：放不下時換行，超過行數就報錯。``split``：放不下的字幕拆成數句連續字幕（``reflow_cues``），優先在標點處切，人名與詞不切開，文字一字不漏。
   * - ``words``
     - JSON 路徑、清單或 null
     - null
     - ASR 逐字時間（例如 Qwen3ASR 的 ``{"words": [...]}``），用來為拆出的字幕對時；只能搭配 ``overflow: "split"``。
   * - ``glossary``
     - JSON 路徑或 null
     - null
     - 專名詞表（武則天 r2b 的 ``glossary.json`` 格式：類別 → ``{中文: 英文}``，中文可用 ``／``、``/``、``、`` 分隔別名）。詞表中的中英文專名在兩種語言的字幕都以 ``highlight_color``（夜燈預設土黃 ``#C9A35D``）上色，且不會被斷行拆開。
   * - ``highlight``
     - 字串清單
     - ``[]``
     - 額外要上色的專名，與 ``glossary`` 合併。

4.9 ``audio``（旁白與配樂）
~~~~~~~~~~~~~~~~~~~~~~~~~~

有 ``audio`` 區段時一定要有 ``narration``。

.. list-table::
   :header-rows: 1
   :widths: 24 14 18 44

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``narration``
     - 字串
     - 必填
     - 旁白檔案路徑（WAV 或 AudioFileClip 可讀的格式）。
   * - ``narration_offset``
     - 數字（≥0）
     - ``0.0``
     - 旁白開始的成品秒數。
   * - ``narration_gain_db``
     - 數字
     - ``0.0``
     - 旁白增益（dB）。
   * - ``music``
     - 字串或 null
     - null
     - 配樂檔案路徑。null 表示沒有配樂。
   * - ``music_gain_db``
     - 數字
     - ``-18.0``
     - 配樂增益（dB）。-18 dB 讓配樂保持在語音之下的背景床。
   * - ``music_duck_db``
     - 數字（≤0）
     - ``-8.0``
     - 旁白播放期間，配樂再降低的固定分貝數。這是常數，不是跟隨音量的壓縮器。
   * - ``music_loop``
     - 布林
     - ``false``
     - 配樂短於成品時是否循環。false 時配樂播完即止。
   * - ``fade_in``
     - 數字（≥0）
     - ``0.0``
     - 整體混音的淡入秒數。
   * - ``fade_out``
     - 數字（≥0）
     - ``0.0``
     - 整體混音的淡出秒數。兩者相加不可超過成品長度。
   * - ``trim_audio``
     - 布林
     - ``false``
     - 旁白或混音超過時間軸時，是否截掉超出的部分。false 時會報錯。

4.10 ``scene_overlay``（章節疊圖套組）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每章一組透明疊圖：左上 logo、右上浮水印、右側直式章名、右下「立即訂閱」按鈕（縮放 80%→104%→100%）。版面預設值即夜燈 R26 的 1920×1080 policy，會依成品尺寸縮放。疊圖不可與片頭、片尾字卡重疊。

.. list-table::
   :header-rows: 1
   :widths: 22 16 18 44

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``chapters``
     - ``[[秒, 章名], ...]``
     - 必填
     - 每章疊圖的開始秒數與章名。章名中的空格是優先換欄點（「第三章 雷劫守候」排成兩欄）。章節時間不可重疊。
   * - ``logo``
     - 字串或 null
     - null
     - logo 圖檔路徑（相對於 JSON 資料夾），或直接寫文字。
   * - ``watermark``
     - 字串或 null
     - null
     - 浮水印文字，半透明，從第一章顯示到最後一章結束。
   * - ``cta_text``
     - 字串
     - ``"立即訂閱"``
     - 訂閱按鈕文字；播放三角形是畫出來的，不依賴字型。
   * - ``layout``
     - 物件
     - ``{}``
     - ``SceneLayout`` 欄位的覆寫，例如 ``{"duration": 4.0, "fade": 0.4}``。

4.11 ``name_tags``（人物名牌）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在人物圖旁邊顯示姓名與身分。名牌放在 ``subject_box`` 旁邊（自動選左右上下有空間的一側），不壓到人物與字幕安全框，也不超出畫面。空間不夠時，先把名牌縮到 70%，再改成橫式，仍放不下才報錯，不截字。名牌從人物那一側滑出並淡入，結束前淡出。

.. list-table::
   :header-rows: 1
   :widths: 22 16 18 44

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``name``
     - 字串
     - 必填
     - 姓名，例如「嬌娜」。
   * - ``role``
     - 字串或 null
     - null
     - 身分小字，例如「狐仙」。
   * - ``subject_box``
     - ``[x, y, 寬, 高]``
     - 必填
     - 人物在 1920×1080 畫面中的框（像素），其他尺寸自動縮放。
   * - ``start`` / ``duration``
     - 數字
     - 必填 / ``4.0``
     - 出現時間與秒數。
   * - ``side``
     - 字串
     - ``"auto"``
     - ``auto``、``left``、``right``、``above`` 或 ``below``。
   * - ``orientation``
     - 字串
     - ``"vertical"``
     - ``vertical``（直書）或 ``horizontal``。
   * - ``align``
     - 字串
     - ``"top"``
     - 名牌沿人物邊緣的對齊：``top``、``center``、``bottom``、``left``、``right``。
   * - ``fade_in`` / ``fade_out``
     - 數字
     - ``0.45`` / ``0.65``
     - 淡入、淡出秒數。
   * - ``slide_px``
     - 數字
     - ``24``
     - 滑入距離（1080p 像素）。
   * - ``seal`` / ``leader``
     - 字串或 null / 布林
     - null / ``false``
     - 單字印章（如「誌」）；是否畫指向人物的引線。
   * - ``font_px`` / ``gap`` / ``margin``
     - 數字
     - ``56`` / ``24`` / ``48``
     - 姓名字級、名牌與人物的間距、與畫面邊緣的距離（1080p 像素）。
   * - ``panel_color`` / ``panel_alpha`` / ``bracket_color`` / ``name_color`` / ``role_color`` / ``seal_color``
     - 顏色
     - R26 夜燈配色
     - 底色與不透明度、括線、姓名、身分與印章顏色。

4.12 ``title_overlay``（片頭標題疊字）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

武則天 r2b 片頭的做法：標題、副標與一句鉤子直接疊在片頭影片上淡入淡出，不蓋整張字卡。欄位與 ``intro`` 相同（見 4.4），另加 ``start``；不可設轉場。不可與片頭片尾字卡重疊。

.. list-table::
   :header-rows: 1
   :widths: 22 16 18 44

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``start``
     - 數字（≥0）
     - ``0``
     - 成品時間軸上的開始秒數。r2b 為 8.2 秒。
   * - ``title``
     - 字串
     - 必填
     - 主標，例如「武則天」。
   * - ``brand``
     - 字串
     - ``""``
     - 主標上方的小字。
   * - ``subtitle``
     - 字串
     - ``""``
     - 副標或鉤子，例如「從才人到皇帝・權力的代價」。
   * - ``layout``
     - ``"right_column"``、``"left_column"`` 或 ``"center"``
     - ``"left_column"``
     - 文字欄位置。
   * - ``duration``
     - 數字（>0）
     - ``5.0``
     - 顯示秒數（含淡出）。r2b 約 8.3 秒。
   * - ``fade_out``
     - 數字（≥0）
     - ``0.35``
     - 結尾淡出秒數。
   * - ``rule``
     - 布林
     - ``true``
     - 是否畫分隔線。
   * - ``auto_extend``
     - 布林
     - ``true``
     - 字多時依閱讀時間自動延長。
   * - ``title_keep_together``
     - 字串清單
     - ``[]``
     - 主標中不可拆行的詞。
   * - ``subtitle_keep_together``
     - 字串清單
     - ``[]``
     - 副標中不可拆行的詞。
   * - ``transition``
     - ``"cut"``
     - ``"cut"``
     - 固定為 ``cut``，其他值報錯。
   * - ``transition_duration``
     - null
     - null
     - 必須為 null。
   * - ``brand_start``
     - 數字
     - ``0.2``
     - 各元素進場時間，同 4.4（另有 ``brand_duration``、``title_start``、``title_duration``、``subtitle_start``、``subtitle_duration``）。
   * - ``brand_duration``
     - 數字
     - ``0.55``
     - 品牌字進場秒數。
   * - ``title_start``
     - 數字
     - ``0.55``
     - 主標進場時間。
   * - ``title_duration``
     - 數字
     - ``0.85``
     - 主標進場秒數。
   * - ``subtitle_start``
     - 數字
     - ``1.05``
     - 副標進場時間。
   * - ``subtitle_duration``
     - 數字
     - ``0.65``
     - 副標進場秒數。

4.13 ``source_inserts``（史料圖插入）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

網路找到的史料原圖（畫像、文物照片）完整呈現：只縮放、不裁切、不改色，最多放大 2 倍。插入期間蓋住原本的畫面，角落以 12 號（1080p 基準）小字標示引用，自動挑不壓到圖與字幕、且最暗最乾淨的角落。給了 ``sha256`` 時會先核對原圖未被改動。

.. list-table::
   :header-rows: 1
   :widths: 22 16 18 44

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``image``
     - 路徑
     - 必填
     - 史料圖檔（相對於 JSON 資料夾）。
   * - ``start``
     - 數字（≥0）
     - 必填
     - 開始秒數。
   * - ``duration``
     - 數字（>0）
     - 必填
     - 顯示秒數。
   * - ``citation``
     - 字串
     - 必填
     - 角落引用，例如「金古良編、朱圭刻《無雙譜》（約1690）｜Wikimedia Commons／PD」。
   * - ``title``
     - 字串
     - ``""``
     - ``portrait`` 版式右欄的大標，例如「武則天」。
   * - ``caption``
     - 字串
     - ``""``
     - 右欄中字，例如「《無雙譜》」。
   * - ``note``
     - 字串
     - ``""``
     - 右欄小字，例如「約1690年・後世刻本圖像」。
   * - ``layout``
     - ``"full"`` 或 ``"portrait"``
     - ``"full"``
     - ``full``：整張圖置中、位於字幕區之上。``portrait``：圖在左（原尺寸），右欄放標題與說明，即 r2b 的《無雙譜》畫像卡。
   * - ``background``
     - ``"paper"``、``"dark"`` 或路徑
     - ``"paper"``
     - 底：宣紙紋、深色暈影，或一張圖（封面裁切）。
   * - ``fade``
     - 數字（≥0）
     - ``0.4``
     - 淡入淡出秒數。
   * - ``citation_corner``
     - 字串
     - ``"auto"``
     - ``auto``、``top_right``、``bottom_right``、``top_left``、``bottom_left``。指定的角落壓到字幕區會報錯。
   * - ``sha256``
     - 字串
     - ``""``
     - 原圖雜湊；給了就核對，不符即報錯。

4.14 ``end_card``（片尾按讚訂閱分享輪播）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

片尾半透明面板：頻道名、一句預告、一個提問，下方「按讚・訂閱・分享」三個圖示輪流放大（80%→106%→100%）並亮成紅色，其餘變暗。圖示以多邊形畫出，不依賴表情符號字型。``start`` 為 null 或省略時，自動放在時間軸最後 ``duration`` 秒。面板底緣在字幕安全框之上。

.. list-table::
   :header-rows: 1
   :widths: 22 16 18 44

   * - 欄位
     - 型別
     - 預設值
     - 意義
   * - ``start``
     - 數字或 null
     - null（JSON）
     - 開始秒數；null 為片尾。
   * - ``duration``
     - 數字（>0）
     - ``8.0``
     - 顯示秒數。
   * - ``channel``
     - 字串
     - ``"夜燈說書"``
     - 頻道名（大字）。
   * - ``headline``
     - 字串
     - ``""``
     - 頻道名下方的標題，可空。
   * - ``line``
     - 字串
     - ``"訂閱・下一回再走進歷史"``
     - 預告句。
   * - ``question``
     - 字串
     - ``""``
     - 留言提問，例如「武則天哪個選擇，最讓你在意？」。
   * - ``actions``
     - 字串清單
     - ``["按讚", "訂閱", "分享"]``
     - 按鈕文字，依序輪播。
   * - ``icons``
     - 字串清單
     - ``["like", "subscribe", "share"]``
     - 對應圖示：``like``、``subscribe``、``share``。
   * - ``period``
     - 數字（>0）
     - ``1.6``
     - 每個按鈕亮起的秒數。
   * - ``panel_color``
     - ``[R, G, B]``
     - ``[29, 23, 19]``
     - 面板顏色。
   * - ``panel_alpha``
     - 數字（0–1）
     - ``0.725``
     - 面板不透明度（185/255）。
   * - ``text_color``
     - ``[R, G, B]``
     - ``[239, 220, 184]``
     - 文字顏色。
   * - ``accent_color``
     - ``[R, G, B]``
     - ``[221, 55, 55]``
     - 亮起按鈕的顏色。
   * - ``fade``
     - 數字（≥0）
     - ``0.5``
     - 淡入淡出秒數。

4.15 兩頻道預設值與理由
~~~~~~~~~~~~~~~~~~~~~~~~

兩份 config 的數值取自 ``episode.py`` 模組說明文件。下表說明它們為何如此設定，改動前請先確認理由是否仍成立。

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - 項目
     - ``nightlamp_story``（夜燈說書）
     - ``nightlamp_history``（夜燈史話）
   * - 預設值（``preset``）
     - 24 fps，1920×1080；暖白字與燈火色。預覽保持 24 fps 讓渲染成本低，同時符合電影幀率。
     - 24 fps，1920×1080；宣紙、墨與朱印。
   * - 遮罩
     - 字卡底圖為真實畫面，``overlay_opacity`` 0.5，確保暖白字在任何畫面上都清楚。
     - 宣紙為底，``overlay_opacity`` 0.0，不需壓暗。
   * - 字卡背景
     - ``background`` 使用單一凍結影格，搭配 0.5 遮罩；不設純色備援。
     - 同樣可使用 ``background``；config 內有 ``background`` 區段。
   * - 運鏡
     - ``move`` 為 ``auto``，順序為推、右移、推、左移、拉，相鄰靜圖不重複；``hold`` 1.5 s。
     - ``push``，``zoom`` ``[1.0, 1.08]``，``hold`` 2.0 s。8% 的推進足以表現強調；12% 對靜圖而言過強。
   * - 轉場
     - ``crossfade`` 0.6 s（即 ``preset.fade``），落在 1.5 s 聚焦停留之內。
     - ``crossfade`` 0.5 s，完全落在 2.0 s 聚焦停留之內，溶接不會在鏡頭還在移動時開始。
   * - 字幕
     - zh-TW 72 px 加 English 42 px（1080p 基準）。
     - 同左。
   * - 章節與引文
     - 無章節、無引文，頻道只帶字幕。
     - 章節 ``chapter_period`` 5.5 s，為頻道值，足夠讀完一行古文再輪播。引文 ``dynasty`` 決定竹簡或宣紙。
   * - 片頭、片尾
     - 各 6.0 s，``right_column``。
     - 同左。
   * - 旁白
     - ``fade_in`` 0.5 s，``fade_out`` 2.0 s。
     - 同左。

五、時間規則
------------

時間軸由前到後依序為：片頭（若有）、各個 shot、片尾（若有）。

* **段落與轉場**：``cut`` 直接接上；``crossfade`` 讓新段與前一段重疊 ``transition_duration`` 秒；``dip`` 將前段淡出到 ``dip_color``，再把新段淡入，前後各佔一半時間，不重疊。
* **轉場位於聚焦停留內**：``crossfade`` 或 ``dip`` 的轉場長度（``dip`` 取一半）不得超過**前一張 Ken Burns 靜圖**的 ``hold``。靜圖、影片段不受此限制，但仍不得超過前後段的長度。
* **片頭限制**：沒有片頭時，第一段 shot 必須為 ``cut``。片頭本身也必須為 ``cut``。
* **字卡時間可能延長**：``auto_extend`` 為 true 時，字卡可能比 ``duration`` 長。章節、引文與字幕的時間都以建置後的時間軸為準，因此要先用 ``validate`` 讀出實際時間。
* **章節與引文的時間是絕對時間**：``start`` 從成品第 0 秒算起，包含片頭。要知道某張圖在哪一秒，請查 ``report.json`` 的 ``timeline``。
* **字卡不可被覆蓋**：章節與引文不可與片頭、片尾字卡重疊；章節彼此不可重疊；引文彼此不可重疊。
* **字幕與旁白不可超出**：字幕 cue 的結束時間不得晚於成品長度。旁白的結束時間不得晚於成品長度，除非設定 ``trim_audio``。

六、常見 EpisodeError 與修法
----------------------------

``EpisodeError`` 是 ``ValueError`` 的子類別。CLI 會印出 ``ERROR: <訊息>`` 並以結束碼 1 離開。下表列出最常見的訊息（節錄自程式碼）。

.. list-table::
   :header-rows: 1
   :widths: 40 26 34

   * - 訊息（節錄）
     - 原因
     - 修法
   * - ``background source not found: …``、``shots[0] image not found: …``
     - 路徑錯誤，或 ``PLACEHOLDER_*`` 未替換。
     - 把素材放進 ``media/``，並更正 ``episode.json`` 的檔名（大小寫也要相同）。
   * - ``shots[0]: no previous segment, transition must be cut``
     - 沒有片頭，但第一段 shot 設定了轉場。
     - 第一段改為 ``"transition": "cut"``，或加上片頭。
   * - ``Shot 02: crossfade of 0.6 s must lie inside the Shot 01 focus hold of 0.5 s``
     - 轉場比前一張的 ``hold`` 長。
     - 縮短 ``transition_duration``，或加長前一張的 ``hold``。
   * - ``Shot 03: crossfade of … is longer than a neighbour``
     - 轉場長度超過前段或後段的總長。
     - 加長 ``duration``，或縮短轉場。
   * - ``chapters[0] […) lies outside the … s timeline``、``chapters[1] […) overlaps the Intro card […)``
     - 章節時間超出成品，或覆蓋片頭/片尾。
     - 以 ``validate`` 或 ``report.json`` 讀出時間軸，把章節移到字卡之後。
   * - ``chapters overlap: […) and […)``
     - 兩個章節時間重疊。
     - 調整 ``start`` / ``end``。
   * - ``quotes overlap: …``
     - 引文的顯示期間相互重疊。
     - 把後一則引文的 ``start`` 移到前一則結束之後。
   * - ``quotes[0] […) overlaps the … card …``
     - 引文覆蓋字卡。
     - 同上，移到字卡之外。
   * - ``subtitles.primary cue […) … lies outside the … s timeline``
     - SRT 的時間超出成品長度，常見於片尾漏調。
     - 修正 SRT 或加長成品。
   * - ``audio.narration ends at … s, past the … s timeline; lengthen the shots or set audio.trim_audio to true``
     - 旁白比畫面長。
     - 加長 shot 的 ``duration``；若尾巴可以截掉，設 ``trim_audio`` 為 true。
   * - ``audio.narration_offset … s is past the … s timeline``
     - 旁白的起點已超出畫面。
     - 調整 ``narration_offset``。
   * - ``audio.fade_in + audio.fade_out exceed the timeline``
     - 淡入淡出加總超過成品長度。
     - 縮短淡入或淡出。
   * - ``video out … s is past the end of the video (…)``
     - 影片 shot 的 ``out`` 超過檔案長度。
     - 改小 ``out``。
   * - ``… must be a number, got …``、``… must be a string …``
     - JSON 型別錯誤，例如把數字寫成字串。
     - 修正型別；布林值不會被當作數字接受。
   * - ``unknown episode key(s): […]`` 或 ``…: unknown key '…'``
     - 欄位名稱拼錯或放錯區段。
     - 對照第四節的欄位表。
   * - ``… missing required key(s) […]``
     - 必填欄位缺漏（例如 ``title``、``narration``、``items``）。
     - 補上該欄位。
   * - ``unknown preset '…'; choose from […]``
     - ``preset`` 拼錯。
     - 改為 ``nightlamp_story`` 或 ``nightlamp_history``。
   * - ``fonts.quote needs a real font file``
     - ``quote`` 或 ``chapter`` 設成 null。
     - 指定真實字型檔。
   * - ``output directory is not empty: … (use overwrite)``
     - 輸出目錄已有檔案。
     - 換一個目錄，或加上 ``--overwrite``。

七、驗證
--------

.. code-block:: console

    python -m moviepy.ae.templates validate E:\episodes\ep012\episode.json

``validate`` 只讀取 JSON、檢查素材並建置時間軸，不會渲染。成功時輸出一行摘要，包含尺寸、幀率、總長、段落數、章節數、引文數與是否有音訊。其中的**總長**是之後撰寫章節與字幕時間的對照基準。

建議每次修改 ``episode.json`` 後都先 ``validate``。

八、預覽 → 正式輸出
-------------------

預覽
~~~~

.. code-block:: console

    python -m moviepy.ae.templates render E:\episodes\ep012\episode.json E:\episodes\ep012\out_preview --preview

``--preview`` 強制以 480×270、12 fps 渲染，字型、時間與音訊皆不變，適合檢查版面與時間，耗時短。預覽只用來檢查排版與時序；畫面細節、字型清晰度與運鏡平滑度要在正式輸出中確認。

指定靜圖時間
~~~~~~~~~~~~

.. code-block:: console

    python -m moviepy.ae.templates render E:\episodes\ep012\episode.json E:\episodes\ep012\out_still --preview --still 0.5 12.0 --still 40.0

``--still`` 可以重複使用或接多個數字（單位：秒），指定要輸出靜圖的時間點，寫入 ``stills/*.png``。影片仍會完整渲染，``episode.mp4`` 照常產生。不指定時，程式會自動選擇：片頭與片尾中點、每個章節開始後 1 秒、每則引文中點。檔名格式為 ``<標籤>_<毫秒>ms.png``（自訂時間的標籤為 ``t``）。

正式 1080p
~~~~~~~~~~

.. code-block:: console

    python -m moviepy.ae.templates render E:\episodes\ep012\episode.json E:\episodes\ep012\final

正式輸出使用 preset 的尺寸與幀率（1920×1080，24 fps）。

並行渲染與硬體編碼
~~~~~~~~~~~~~~~~~~

``render`` 預設以多個程序並行渲染各幀，並由單一 FFmpeg 依序編碼，因此輸出畫面與單程序渲染逐幀相同。``--workers`` 預設為 ``min(CPU 核心數 - 1, 12)``；``--workers 1`` 回到單程序。

``--encoder auto``（預設）會實際試編一小段影片確認 NVIDIA NVENC 可用，可用時採 ``h264_nvenc``，否則退回 ``libx264``。也可明確指定 ``libx264``、``h264_nvenc``、``hevc_nvenc`` 或 ``libx265``。

.. code-block:: console

    python -m moviepy.ae.templates render E:\episodes\ep012\episode.json E:\episodes\ep012\final --workers 8 --encoder h264_nvenc

1080p 實測（16 核心、RTX 3070 Ti、51.6 秒成片）：單程序 libx264 約 75 秒，並行＋NVENC 約 38 秒，約快 2 倍。單幀合成已大量使用記憶體頻寬，工作程序超過約 8 個後幾乎不再加速；每個工作程序啟動時會各自組裝一次 episode（約 2 秒）。``libx264`` 的品質預設為 CRF 18（較舊版的 23 檔案大、畫質高）。

無損母帶
~~~~~~~~

.. code-block:: console

    python -m moviepy.ae.templates render E:\episodes\ep012\episode.json E:\episodes\ep012\final --master

``--master`` 另外寫入 ``master/``（FFV1 無損 RGBA16，內含 ``master.mkv`` 與 ``manifest.json``）。母帶需要長度精確的 PCM WAV，因此混音會先輸出為 ``master_audio.wav``（48 kHz 立體聲）。

若幀率使得取樣數無法為整數，母帶會**無音訊**寫出，並在結果的 ``notes`` 中說明。29.97 這類分數幀率通常無法得到整數取樣數；24、25、30 fps 皆可。

覆寫
~~~~

輸出目錄若非空，會拒絕執行，除非加上 ``--overwrite``。加上後會取代同名檔案；母帶目錄會被整個刪除後重寫。

輸出目錄內容
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - 路徑
     - 內容
   * - ``episode.mp4``
     - 成品，libx264、aac、yuv420p，幀率為 preset 的值。
   * - ``report.json``
     - QA 報告，包含 ``timeline``、``chapters``、``quotes``、``subtitles``、``layers``、``sources``（每個素材的 SHA-256）與 ``audio``。
   * - ``stills/``
     - 靜圖 PNG，見上文。
   * - ``master/``
     - 僅在 ``--master`` 時產生。內含 ``master.mkv`` 與 ``manifest.json``。
   * - ``master_audio.wav``
     - 僅在 ``--master`` 且有音訊、取樣數為整數時產生。

CLI 會把結果以 JSON 印到終端機，包含 ``video``、``report``、``stills``、``master``、``notes`` 與 ``timings``（各階段秒數）。``timings`` 可用來估計下次渲染所需的時間。

Python 介面
~~~~~~~~~~~

CLI 與下列 Python 寫法相同：

.. code-block:: python

    from moviepy.ae.templates.episode import render_episode

    result = render_episode(
        r"E:\episodes\ep012\episode.json",
        r"E:\episodes\ep012\final",
        preview=False,
        master=True,
        overwrite=False,
    )
    print(result["video"], result["notes"], result["timings"])

``render_episode`` 也接受 ``EpisodeSpec`` 物件或 dict；``stills`` 參數對應 CLI 的 ``--still``。其他相關函式：``init_episode(directory, channel)`` 與 CLI 的 ``init`` 相同，回傳 ``episode.json`` 的路徑；``build_episode(spec)`` 回傳 ``Composition``，``episode_report(comp)`` 回傳與 ``report.json`` 相同的 QA 字典。

九、驗收清單
------------

成品完成前，逐項確認。機械檢查通過**不代表**版面好看或史實正確。

1. ``validate`` 輸出 OK，且總長與預期相符。
2. 字幕斷行：對 zh-TW 與 en 的 SRT 各執行 ``find_bad_breaks``，結果應為空。

   .. code-block:: python

       from moviepy.ae.templates.subtitles import find_bad_breaks, parse_srt

       text = open(r"E:\episodes\ep012\subtitles\zh-TW.srt", encoding="utf-8-sig").read()
       cues = parse_srt(text, "zh-TW")
       print(find_bad_breaks(cues, protected=["夜燈"]))  # [] 為通過

3. 字幕時序：對同一份 cue 執行 ``check_timing``，結果應為空（檢查字速、最短／最長秒數與重疊）。

   .. code-block:: python

       from moviepy.ae.templates.subtitles import check_timing

       print(check_timing(cues))  # [] 為通過

4. 運鏡平滑度：每個 Ken Burns 靜圖的移動段（即段長扣除 ``hold``）都必須為 ``SMOOTH``。``STATIC``（沒有動）與 ``JITTER``（抖動）都不通過。

   .. code-block:: python

       from moviepy.ae.templates.episode import EpisodeSpec, build_episode
       from moviepy.ae.templates.ken_burns import check_motion

       spec = EpisodeSpec.from_json(r"E:\episodes\ep012\episode.json")
       comp = build_episode(spec)
       try:
           for seg in comp.episode["timeline"]:
               if seg["kind"] == "shot" and seg["hold"] is not None:
                   window = seg["duration"] - seg["hold"]
                   result = check_motion(comp, seg["start"], window)
                   print(seg["name"], result["verdict"])
       finally:
           for clip in comp.episode_clips:
               clip.close()

5. 目視靜圖：檢查 ``stills/`` 內的每張 PNG。字卡文字不可被遮罩或裁切，章節標籤不可壓住字幕區域，引文不可與字幕重疊。
6. 完整看片：從頭到尾看一次成品 ``episode.mp4``，檢查音量、字幕同步、轉場與結尾。這一項無法被任何指令取代。

十、只要單一部件時的入口
------------------------

若只需要某一個元件，而不是整集，請參考 :doc:`templates`：

* ``build_title_card``：片頭片尾字卡（``TitleCardSpec``）。
* ``chapter_tag``：章節標籤。
* ``vertical_quote``：直式引文。
* ``ken_burns``：單張靜圖的次像素運鏡。
* ``burn_subtitles``：把字幕燒錄到任意 clip 或 composition。

十一、限制
----------

* **效能**：1080p 正式輸出的每幀時間依機器、素材解析度與媒介而異，故以預覽與 ``timings`` 估算。長片的渲染時間約與片長成正比，建議先完整預覽，再安排正式輸出。
* **配樂不跟隨旁白**：``music_duck_db`` 是固定的常數壓低，不會依旁白的實際音量動態調整。若旁白音量差異大，請在剪輯端另行處理。
* **旁白音量**：單聲道旁白會被轉為立體聲並降低約 3 dB（``AudioFileClip`` 的行為）。建議直接提供立體聲檔案。
* **字型**：不會自動換字。缺字即報錯，需要使用者自行確認字型涵蓋所需字元。
* **母帶是 SDR**：``--master`` 是顯示用的 sRGB/BT.709 無損母帶，不是 HDR 輸出。
* **預覽不等於成品**：``--preview`` 的解析度與幀率與正式輸出不同，不可用來確認畫質。
