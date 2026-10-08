:orphan:

定時字幕、節奏粒子、遮罩轉場及快取契約另見 :doc:`motion_modules`。

Godot 立體字與粒子後端
========================================

``render_godot_scene`` 將 JSON 相容的 Python dict 交給獨立 Godot 程序，
以 Movie Maker 固定影格率產生 RGBA PNG 序列及 48 kHz stereo PCM WAV。
MoviePy 再接手剪輯、字幕、音訊、2D 後製與 MP4 輸出。此後端不需要 Adobe、
Substance、原生 SBSAR、額外 Python 套件或網路模型。

目前實測環境是 Windows 11、Godot 4.7.2 standard MIT 版、Forward+ Vulkan、
RTX 3060。程式要求 Godot 4.7 以上的 4.x；其他版本及 GPU 尚未逐一驗證。
目前隱藏 GPU 視窗實作限 Windows，其他平台會明確報錯。
這是離線輸出介面，沒有提供互動編輯器或即時預覽 UI。

取得與執行
----------

從 `Godot 官方下載頁 <https://godotengine.org/download/windows/>`_ 取得 standard
可攜版；不需要 .NET 版、export templates 或插件。自行核對對應 release 的
``SHA512-SUMS.txt``。套件不會自動下載或安裝引擎，也不修改 PATH 或系統設定。

.. code-block:: powershell

   $env:MOVIEPY_GODOT = 'C:\Tools\Godot\Godot_v4.7.2-stable_win64.exe'
   python -m examples.ae_godot_showcase .\output-godot

輸出目錄必須不存在。範例預設 1280×720、24 fps、4 秒，包含擠出文字、
兩組固定 seed 的 GPUParticles3D、真正 3D 遮擋及陰影、原創程式格線材質和
合成音效。最後的背景及光暈由 MoviePy 在 2D 合成後產生。

``godot/`` 內保留場景 JSON、GDScript、project.godot、PNG、WAV、
引擎 log 與 runtime.json。``godot_showcase.mp4`` 是壓到背景上的 H.264/AAC
交付影片，MP4 本身不含 alpha；透明資訊仍在 PNG 與 MoviePy mask 中。
``preview_*.png`` 可供靜態檢視。

最小使用範例
------------

.. code-block:: python

   from moviepy import ColorClip, CompositeVideoClip
   from moviepy.ae.three_d import render_godot_scene

   scene = {
       "size": [640, 360], "fps": 24, "duration": 2,
       "materials": {
           "blue": {"color": [0.1, 0.6, 0.9], "roughness": 0.3}
       },
       "objects": [{
           "type": "text", "text": "HELLO", "material": "blue",
           "depth": 0.2,
           "animation": [
               {"time": 0, "rotation": [0, -20, 0]},
               {"time": 2, "rotation": [0, 20, 0]},
           ],
       }],
       "lights": [{
           "type": "directional", "location": [-3, 5, 4],
           "target": [0, 0, 0], "energy": 2, "shadows": True,
       }],
       "particles": [{"start": 0.3, "count": 250, "seed": 605}],
   }
   result = render_godot_scene(scene, "render-hello")
   layer = result.to_clip()
   background = ColorClip((640, 360), (8, 15, 30), duration=2)
   with CompositeVideoClip([background, layer]) as video:
       video.write_videofile("hello.mp4", fps=24, audio=False)
   layer.close()
   background.close()

座標採 Godot 的 **Y 向上**，與既有 Blender 場景格式分開。旋轉是角度；
TRS 關鍵幀逐軸線性插值，旋轉不會自動選擇最短路徑。
相機 location、target 與 fov 控制視角，觀看方向不可與 Y 軸平行。
objects 支援 cube、sphere、plane、cylinder、torus、擠出的 text。
完整 schema 由 ``validate_godot_scene`` 驗證，未知欄位、任意 shader／script
和無效參數會報錯，不會被靜默忽略。

materials 提供 color、metallic、roughness、emission、emission_energy
及原創 grid 程式材質。lights 提供 directional、omni、spot 與 shadows 開關。
發光材質和粒子不等於新增光源，照亮其他物體需要明確加入 light。
地面要有 mesh 才能接收 3D 陰影；MoviePy 的其他 2D 圖層不會參與此場景的深度或陰影。

文字預設使用 Godot 內建 Open Sans SemiBold。中文可在 text 物件指定
font 為已獲使用授權且具備中文字形的本機 TTF／OTF／TTC 檔案。
字型只在執行時讀取，不會複製到 repo 或其他服務。
使用自訂字型時停用系統 fallback，避免無意使用其他字型。

粒子、透明與音訊邊界
--------------------

每組粒子是一次爆發：count、start、lifetime、speed、direction、spread、
gravity、size、color 與 seed 可調。
事件在第一個時間大於等於 start 的影格觸發。程式由 0 依序模擬，
完整輸出後才建立 clip；倒序取幀和跳到未來都只讀取快取，不會跳過粒子事件。
固定 seed 不代表不同 GPU、driver 或 Godot 版本逐位元一致。
此版本沒有流體、煙霧、粒子碰撞或每粒子燈光。

影格率須是 1–120 的整數，且可整除 48000，例如 24、25、30、50、60、120。
輸出影格數為 ``ceil(fps * duration)``；clip 裁到指定 duration，
原始 WAV 長度則對齊完整影格數。
audio 可指定 mono／stereo 16-bit PCM WAV，Godot 在離線渲染時混音，
不向喇叭播放。未提供音訊時依然輸出等長靜音 WAV。
使用 ``result.to_clip(with_audio=True)`` 時，結束後須關閉 clip.audio 與 clip。

預設 transparent=True，PNG 是 display-referred 8-bit SDR，不是 HDR／EXR。
引擎先以兩倍寬高、關閉 MSAA 渲染到 raw/；Python 在 linear RGB
以 premultiplied alpha 縮小，再輸出 straight-alpha PNG，避免原生 MSAA
把背景 RGB 混到透明邊緣。raw/ 保留原始影格供診斷，因此需要額外磁碟空間。
透明輸出刻意不提供 Godot 的畫面光暈，避免把輪廓外發光與表面 alpha 混為一談；
請在 MoviePy 合成背景後加 Glow。若使用不透明場景，可指定
``transparent=False, glow=0.7`` 啟用 Godot Environment glow。

後端使用不可見的原生父視窗加 ``--wid``，保留 Vulkan 圖形環境但不顯示
編輯器、搶焦點或操作桌面。一般 ``--headless`` 只有 dummy renderer，
不能當成此後端的 GPU 渲染。每次呼叫的 user data、cache、temp 及 log
都放在該次輸出目錄；逾時會終止本次引擎程序，失敗保留證據，不回傳半套成果。
成功前會核對所有 PNG 格式、尺寸、幀數、WAV sample count 及引擎完成紀錄。
Godot 的音訊停止會延後至後續混音才釋放播放資源。使用 WAV 時，後端在最後
一個內容影格寫入後停止音訊，再讓引擎執行四個清理影格，並以弱引用確認資源
已釋放；失敗會報錯。raw/ 保留清理影格與延長的原始 WAV 供診斷，但公開 PNG
序列和 frame.wav 僅包含原本要求的內容；音訊直接複製 PCM，沒有重取樣或淡出。
runtime.json 記錄 audio_cleanup_frames 和 audio_released。未提供音訊時不加
清理影格。所有引擎警告原樣保留於 log，沒有過濾或隱藏。

依賴需求 (Requirements)
----------------------

以下需求同時記錄於 ``moviepy/ae/three_d/requirements.py`` 的
``GODOT_REQUIREMENTS``，並與 ``pyproject.toml`` 的 ``[tool.moviepy.godot]``
保持一致（測試會比對兩者）。

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 項目
     - 需求
   * - 引擎與版本
     - Godot 4，且版本 **4.7 以上、5.0 以下**（例如 4.7.2）。3.x 與 5.x 會被拒絕。
   * - 版本
     - standard 版（非 .NET／Mono）；不需要 export templates 或插件。
   * - 作業系統
     - Windows（``sys.platform == "win32"``）。隱藏 GPU 視窗僅實作於 Windows。
   * - 繪製器
     - Forward+（Vulkan）。
   * - 顯示卡
     - 支援 Vulkan 1.x 的 GPU 與驅動程式。
   * - 執行檔指定
     - 環境變數 ``MOVIEPY_GODOT``；未設定時依序尋找 PATH 上的 ``godot``、``godot4``。
   * - 下載位置
     - `Godot 官方下載頁 <https://godotengine.org/download/windows/>`_。
       請以同一 release 的 ``SHA512-SUMS.txt`` 核對檔案完整性。
   * - Python 額外套件
     - 無；不需要額外 Python 套件。
   * - 網路
     - 渲染不需要網路；套件不會自動下載或安裝 Godot。

檢查指令會列出需求、找到的執行檔與版本，並以結束碼表示結果
（0 為可用，1 為不可用；不會因找不到 Godot 而拋出例外）：

.. code-block:: powershell

   python -m moviepy.ae.three_d.requirements
   python -m moviepy.ae.three_d.requirements --executable 'C:\Tools\Godot\Godot_v4.7.2-stable_win64.exe'
   python -m moviepy.ae.three_d.requirements --json

``--json`` 會輸出 ``requirements`` 與 ``check`` 兩個物件，方便其他工具讀取。

驗證與授權來源
--------------

一般 AE 測試不需要 Godot；具備以上環境時可啟用真實整合測試：

.. code-block:: powershell

   $env:MOVIEPY_AE_TEST_GODOT = $env:MOVIEPY_GODOT
   python -m pytest tests/ae/test_godot_backend.py -q

* Godot 引擎採 `MIT license <https://godotengine.org/license/>`_。
  已驗證可攜版的 SHA-256 是
  ``731980f9608d61333e5baf54a2ef17210acc7a538446c0cb9969f002aca1e953``。
  官方 release 是
  `4.7.2-stable <https://github.com/godotengine/godot-builds/releases/tag/4.7.2-stable>`_。
* 預設字型的選擇見
  `Godot default_theme.cpp <https://github.com/godotengine/godot/blob/4.7.2-stable/scene/theme/default_theme.cpp>`_；
  Open Sans 採 OFL-1.1，通知見
  `Godot COPYRIGHT.txt <https://github.com/godotengine/godot/blob/4.7.2-stable/COPYRIGHT.txt>`_。
* 範例幾何、格線 shader、動畫與音效由本專案程式產生，沿用 MoviePy MIT 授權。
  沒有第三方場景模型、貼圖素材或 AI 權重。
* 官方行為參考：
  `Movie Maker <https://docs.godotengine.org/en/stable/tutorials/animation/creating_movies.html>`_、
  `GPUParticles3D <https://docs.godotengine.org/en/stable/classes/class_gpuparticles3d.html>`_、
  `AnimationPlayer <https://docs.godotengine.org/en/stable/classes/class_animationplayer.html>`_。
