:orphan:

字幕、節奏粒子與遮罩轉場
========================================

``moviepy.ae.motion`` 提供同一個影格契約下的四種 effect：
``captions``、``title_card``、``beat_particles``、``transition``。
它們組成三個可重用模組：字幕／字卡、音樂粒子、遮罩轉場。
既有 Composition、MoviePy clip 和 Godot 場景 API 保持可用。

字幕以 Pillow 排版並按固定位置產生影格；Godot 負責真實擠出文字與
GPUParticles3D；遮罩用 NumPy 計算，MoviePy 負責兩段素材及音訊的合成。
沒有新增 Python 依賴、前景視窗、素材下載或自動音訊播放。
Godot 的環境需求及限制另見 :doc:`godot_backend`。

快速示範
----------------------------------------

從 repo 根目錄執行，output 必須不存在。font 請指定自己有權使用且包含
中文字形的本機字型，程式不會複製或上傳字型檔：

.. code-block:: powershell

   python -m examples.ae_motion_modules output/preview --preview --godot C:/tools/Godot.exe --font C:/Windows/Fonts/msjhbd.ttc
   python -m examples.ae_motion_modules output/full --godot C:/tools/Godot.exe --font C:/Windows/Fonts/msjhbd.ttc

preview 是 640×360、12 fps；full 是 1280×720、24 fps，皆為八秒。
輸出包括 motion_modules.mp4、六張預覽、黑／白／紫背景的 alpha 檢查圖、
原創合成節奏 WAV、分析事件、驗證摘要與 effect 快取。
節奏聲音由 NumPy 合成，不使用商業歌曲或第三方音訊片段。

共同資料契約
----------------------------------------

.. code-block:: python

   request = {
       "effect": "captions",
       "params": {"style": "highlight", "font_size": 42},
       "width": 1280, "height": 720,
       "fps": 24, "frame_count": 96, "seed": 605,
       "assets": {"font": "/absolute/path/to/authorized-font.ttf"},
       "cues": [
           {"start_frame": 0, "end_frame": 48, "text": "逐字高亮"},
           {"start_frame": 48, "end_frame": 96, "text": "穩定排版"},
       ],
       "audio_events": [],
   }

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 欄位
     - 定義
   * - effect
     - 上述四種名稱之一，未知欄位與未知 effect 會報錯。
   * - width／height
     - 16–4096 的整數像素，preview 使用另一個明確尺寸。
   * - fps
     - 1–120 的整數且須整除 48000，所有模組共用這項限制。
   * - frame_count
     - 精確影格數，1–100000，總長度不可超過一小時。
   * - seed
     - 0–2147483647，影響 GPU 粒子種子，也納入所有 effect 的快取識別。
   * - assets
     - 明確的本機 font／audio 路徑，檔案內容 SHA-256 會進入快取識別。
   * - cues
     - 依時間排序且不重疊，start_frame 含該幀，end_frame 不含該幀。
   * - audio_events
     - 有唯一 frame 及 0 < strength <= 1 的粒子事件。

時間固定為 ``frame_index / fps``，不依牆鐘或即時播放器的更新速度。
所有效果先依序產生完整 PNG，再讓 MoviePy 任意取幀；倒序或跳讀不會
重跑模擬。長度永遠是 ``frame_count / fps``，轉場至少兩幀。

.. code-block:: python

   from moviepy.ae.motion import render_motion_effect

   result = render_motion_effect(request, "effect-cache")
   clip = result.to_clip()
   try:
       later = clip.get_frame(2.0)
       first = clip.get_frame(0.0)
       again = clip.get_frame(2.0)  # 與 later 相同的快取影格
   finally:
       clip.close()

result 提供 frames、fps、duration、metadata、directory、cache_hit。
to_clip() 回傳有 mask、無 audio 的 MoviePy clip；音軌由呼叫端明確選擇。

字幕與立體字卡
----------------------------------------

``captions`` 支援 ``style="highlight"`` 或 ``style="pop"``。
前者保留整句並累積變更已到時間的字色；後者按字縮放與淡入。
每個 cue 只排版一次，動畫不改變字距、換行或畫布大小。
pop 的預設時間會預留彈入時間，使最後一個字在 cue 結束前完成入場。

params 還可指定：

* font_size、min_font_size：超過安全區時逐級縮小，仍放不下就報錯。
* safe_margin：四周保留的尺寸比例，預設 0.08。
* position：文字區塊中心的畫面比例座標，預設 [0.5, 0.8]，會夾到安全區內。
* max_lines、line_spacing：最多行數及行距比例，預設 2 和 1.25。
* color、accent、stroke_color：0–1 的 sRGB 三通道顏色。
* stroke_width：外框像素；pop_frames：彈入所需影格數。

cue 可明確給 ``unit_frames``，每個非換行的 grapheme unit 一個絕對影格編號，
空白也算一個 unit。例如「聲音有形」可用 [0, 6, 12, 18]。
未指定時均勻分配；單位不切開常見組合音標與 ZWJ 序列，但不是完整
Unicode UAX #29 或跨字形複雜文字塑形引擎。
此版針對中文與拉丁文字，換行可明確指定，超長句會按 unit 換行。
缺字會報錯，避免把中文字幕靜默顯示為方框。

``title_card`` 使用一個覆蓋完整 effect 的 cue，產生真正擠出的 TextMesh，
帶固定攝影機、燈光及入場縮放／旋轉。params 可設 color、depth、
turn_degrees、safe_margin。中文同樣要指定 assets.font。
字體尺寸依字型量測與畫面比例保守縮放，保留旋轉與擠出空間；
很長的文字建議明確換行或改用字幕，不會提供通用字形排版編輯器。
Godot 的原生 text 物件新增可選 ``centered=True``；
預設 False 保留舊版文字對齊行為。
字卡與粒子各自渲染為獨立 PNG；MoviePy 疊合時不共用 3D 深度。
需要兩者互相遮擋時，請用既有 render_godot_scene 放進同一個場景。

音量分析與 GPU 粒子
----------------------------------------

.. code-block:: python

   from moviepy.ae.motion import analyze_audio, render_motion_effect

   analysis = analyze_audio(
       "rhythm.wav", fps=24, frame_count=192,
       threshold=0.32, min_interval=0.18,
   )
   particles = render_motion_effect({
       "effect": "beat_particles",
       "width": 1280, "height": 720,
       "fps": 24, "frame_count": 192, "seed": 605,
       "assets": {"audio": "rhythm.wav"},
       "audio_events": analysis["events"],
       "params": {
           "count": 220, "lifetime": 0.9,
           "location": [0, -0.25, 0],
           "speed": [1.4, 3.8], "color": [0.14, 0.75, 1],
       },
   }, "effect-cache", godot="C:/tools/Godot.exe")

分析接受 mono／stereo 16-bit PCM WAV，以 10 ms RMS 與前 80 ms 均值
差距找音量上升峰，再用 threshold 和 min_interval 篩選。讀取按固定小區塊
進行，不把整段 PCM 轉成常駐浮點陣列；左右聲道先平方，反相立體聲不會抵消。
事件量化到偵測時間之後的第一個影格，靜音回傳空事件。

這是音量 onset 偵測，不是節拍網格、BPM 推估或語音對齊模型。
可檢視 analysis 的 envelope、events、source_sha256 後自行編輯事件。
beat_particles 要求至少一個事件；相同影格不能有重複事件，
Godot 回報的 burst_start_frames 必須與事件完全相符才算成功。

每個事件建立一組固定種子的 GPUParticles3D；strength 會縮放粒子數。
count、lifetime、speed、spread、gravity、size、color、location 可調。
單次最多配置 200000 個粒子，效果在片段末尾截斷，不會延長時間軸。
音訊檔只供分析與快取識別，粒子輸出不混入聲音；最後用 MoviePy 加音軌。
固定 seed 不承諾跨 GPU、driver 或引擎版本逐像素一致。

遮罩轉場
----------------------------------------

``style="wipe"`` 提供 left_to_right、right_to_left、
top_to_bottom、bottom_to_top；``style="iris"`` 提供 out、in，
並用 center 設圓心比例座標。softness 是正規化的柔邊寬度。
圓形按實際像素距離計算，因此寬螢幕上的 iris 不會變成橢圓。

progress 預設從第一幀的 0 到最後一幀的 1，兩端明確全零／全一，
不會留一圈半透明殘邊。可提供 ``[[frame, value], ...]`` 的分段線性
曲線，允許暫停或倒退；必須覆蓋首尾影格。
若要直接算某個進度，用 validate_motion_request 正規化後呼叫
``transition_mask(request, progress=0.5)``。

.. code-block:: python

   from moviepy.ae.motion import render_motion_effect, transition_clip

   mask = render_motion_effect({
       "effect": "transition", "width": 1280, "height": 720,
       "fps": 24, "frame_count": 24, "seed": 0,
       "params": {"style": "iris", "direction": "out", "softness": 0.09},
   }, "effect-cache")
   transition = transition_clip(clip_a, clip_b, mask)
   # clip_a／clip_b 尺寸須一致，且各自的本地時間軸至少一秒。
   # 呼叫端保留兩個輸入 clip 的所有權；transition 不自行選擇音軌。

mask 的 0 選 A，1 選 B。中間值在 sRGB 編碼的 SDR RGB 上作
premultiplied 混合，再除以結果 alpha，兩段素材本身的 mask 都保留。
這是常用的顯示編碼混合，不是線性光混合、HDR 或景深／折射合成。

快取、透明及穩定性
----------------------------------------

快取 key 包含正規化契約、字幕與事件、seed、FPS、尺寸、素材檔案 SHA-256、
精確 Godot 版本、Pillow／NumPy／OpenCV 版本及相關實作原始碼雜湊。
改字、改字型檔內容、改音訊、改參數或換引擎都不會誤用舊快取。
``motion_cache_key(request, engine_version="...")`` 可做純查詢；
render_motion_effect 會自行讀取真實 Godot 版本。

先在唯一 staging 目錄完成，核對所有影格格式、尺寸和雜湊後才原子改名。
讀取快取仍核對整個 PNG 序列的 SHA-256。損壞或未完成的 cache 會報錯，
不覆寫原檔；失敗 staging 保留證據。兩個程序同時完成同一 key 時，
後完成者只接受已完整驗證的結果，保留自己那份 staging 供檢查。
快取含本機資產路徑與引擎 log，分享影片時不需要分享整個 cache。

PNG 契約為 8-bit SDR、sRGB、straight alpha；mask 是線性 coverage。
字幕縮放與轉場合成先用 premultiplied RGB，避免透明邊緣黑邊。
Godot 延用兩倍尺寸渲染及 alpha 正確縮小的既有後端。
示範有黑、白、彩色背景檢查圖；沒有假設透明 Godot glow 或折射能完整保留。
這一版沒有提供透明折射效果；若需要光暈，先合成背景再加 MoviePy Glow。

測試
----------------------------------------

.. code-block:: powershell

   $env:MOVIEPY_AE_TEST_GODOT = "C:/tools/Godot.exe"
   $env:MOVIEPY_AE_TEST_CJK_FONT = "C:/Windows/Fonts/msjhbd.ttc"
   python -m pytest tests/ae/test_motion_templates.py tests/ae/test_godot_backend.py -q

未設定 opt-in 環境時，真實引擎／字型測試會明確略過；CPU 契約、字幕、
音訊分析、遮罩及快取測試仍執行。完整驗收應加上其他 AE 測試、正式 lint、
FFmpeg 完整解碼及人工檢查預覽。效能紀錄把首次渲染和有完整雜湊驗證的
cache hit 分開，單機數值不可當作所有硬體的即時渲染保證。
