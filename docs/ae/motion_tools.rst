:orphan:

AE 動態工具：下一版預覽
========================================

Godot 的立體字、GPU 粒子與透明序列輸出另見 :doc:`godot_backend`。
可重用的定時字幕、音量驅動粒子與遮罩轉場另見 :doc:`motion_modules`。
Substance 不再是達成這類視覺效果的必要條件；AI 內容填補仍是獨立能力。

這是 ``feat/ae-parity`` 上的增量功能，沿用現有 MoviePy 2.2.0 套件版本。
尚未發布新版本，也沒有完整 After Effects 相容性的承諾。
既有 Composition、Layer、Effect 與 MoviePy clip API 保留。
本次不增加 Python 執行依賴；Blender 與模型檔由使用者另行提供。

實際提供的能力
----------------------------------------

* ``TextLayer``／``Composition.add_text``：固定文字畫布、透明文字與五種
  ``none``、``typewriter``、``fade_up``、``pop``、``word_fade`` 預設。
  打字動畫按常見組合字元分組，支援組合音標、ZWJ、膚色與旗幟；
  不是完整 Unicode UAX #29 實作。中文必須指定含中文字形的本機字型。
  舊 Pillow 若不支援可縮放預設字型，也須提供字型檔。
* ``ParticleLayer``／``Composition.add_particles``：最多 50,000 個有固定種子的
  2D 點發射器粒子、速度、方向、散布、重力、壽命與顏色衰減。
  任意時間取樣與倒退取樣結果一致；適合火花與風格化爆炸，沒有流體／煙霧模擬。
* ``fx.DropShadow``：方向、距離、柔化、顏色、不透明度與僅陰影模式。
  ``fx.Glow``：閾值、半徑、強度與顏色。兩者可用在一般圖層與調整圖層，
  支援現有效果遮罩、關鍵影格與縮小預覽。此 Glow 以 Screen 混合有界高斯光暈，
  在不透明影像內也能擴散高光，且保留已高於 display white 的 HDR 通道。
* ``three_d.render_scene``：獨立 Blender Cycles 程序真正計算立方體、球體、
  平面、攝影機、面光／點光／太陽光、遮擋、陰影與 PBR 材質。
  物體位置／旋轉／縮放可設線性關鍵影格。
  MoviePy 載入渲染後的 RGBA 影格序列；AE Composition 本身仍是 2D 合成器，
  其中的其他圖層不會直接參與 Blender 的深度遮擋或投影。
* ``U2NetBackgroundRemoval``：以現有 OpenCV DNN 執行使用者提供的
  U2Net／U2Netp ONNX 模型；這部分是真正的學習模型推論。
  模型只載入一次，CPU 網路操作序列化，調整圖層的區域渲染會取得完整背景。
* ``refine_matte`` 是傳統導引濾波修邊；``inpaint_classical`` 是傳統
  Telea／Navier–Stokes 填補。沒有生成式 AI 填補、毛髮重建或影片時序模型。

文字、粒子與光影
----------------------------------------

從 repo 根目錄執行範例，使用新的輸出目錄：

.. code-block:: console

   python -m examples.ae_motion_showcase output/motion
   python -m examples.ae_motion_showcase output/motion-3d --blender --font C:/Windows/Fonts/msjhbd.ttc --u2net C:/models/u2netp.onnx

範例以合成素材建立兩秒 H.264 影片、預覽圖、AI alpha／換背景圖片與影片、
傳統填補前後圖與 ``validation.json``。加上 ``--blender`` 才會產生真正 3D；
``--u2net`` 只讀取既有檔案，不會下載模型。字型與模型不會複製進 repo。

.. code-block:: python

   import moviepy.ae as ae

   comp = ae.Composition(size=(640, 360), duration=3, fps=24)
   title = comp.add_text("MOTION", font_size=48, preset="pop")
   title.transform = ae.Transform(position=(320, 120))
   title.effects = [ae.fx.DropShadow(), ae.fx.Glow(intensity=0.4)]
   sparks = comp.add_particles(count=1000, seed=42, emitter=(320, 180))
   sparks.transform = ae.Transform(position=(320, 180))
   comp.write_videofile("motion.mp4", codec="libx264", audio=False)

文字內容、字型、樣式與粒子發射參數在建立時設定；圖層的 transform、opacity、
effect 仍可動畫化。``word_fade`` 以空白分詞，中文逐字顯示請使用 typewriter。
``rate`` 是每秒顯示單位數，``delay`` 是開始時間；文字畫布使用完整標題大小，
不會因打字字數變化而移動 anchor。

Blender 與 Substance 貼圖
----------------------------------------

``render_scene(scene_dict, new_directory, executable=..., timeout=600)``
接受可 JSON 序列化的資料；不接受任意 Python 腳本或 .blend 檔。
可由 ``MOVIEPY_BLENDER``、PATH 或標準安裝位置尋找程式。
以 background、factory startup 與 disable-autoexec 啟動，保留 ``scene.json``
與 ``blender.log``；每張影格都驗證大小與 RGBA 格式，失敗會拋出例外。
既有非空輸出目錄不會被覆蓋。

場景範例見 ``examples/ae_motion_showcase.py:scene_3d``。
``materials`` 可指定 ``base_color``、``metallic``、``roughness``、
``emission_color``、``emission_strength``，以及 ``maps`` 中的本機影像路徑：
base_color、metallic、roughness、normal、height、emission、opacity。
Base Color／Emission 使用 sRGB；資料貼圖使用 Non-Color。
``normal_convention`` 接受 opengl 或 directx；height 產生 bump，不改變幾何外形。
Substance 必須先匯出這些 PBR 圖像，原生 ``.sbsar`` 會明確拒絕。

輸出交換格式是 Standard view 的 **8-bit SDR RGBA PNG**，不是線性 HDR／EXR。
``result.to_clip()`` 保留 alpha 與秒數；攝影機採 Blender 的右手座標與 Z 向上，
角度輸入為度。模型尺寸單位、燈光與色彩管理不等同 Adobe 校準。

模型與填補的界線
----------------------------------------

.. code-block:: python

   from moviepy.ae.ai import U2NetBackgroundRemoval, inpaint_classical

   remover = U2NetBackgroundRemoval(
       "C:/models/u2netp.onnx",
       expected_sha256="309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
       refine_radius=3,
   )
   # frame: uint8 RGB，mask: HxW float32 0..1
   mask = remover.predict_mask(frame)
   cutout_clip = clip.with_effects([remover])
   repaired_frame = inpaint_classical(frame, removal_mask, method="telea")

模型輸入縮放至 320x320 SDR RGB，輸出 alpha 回到來源尺寸；保留來源顏色。
每幀獨立推論，沒有光流或時序一致性，半透明物體與髮絲需要另行評估。
``refine_radius`` 是處理影格中的像素半徑，只對 matte 邊緣做導引濾波。
填補遮罩大於零的像素會被取代，未遮罩像素保持逐位元相同；
大型物體後面的內容不會被學習模型重建。

本機驗證所用檔案來自 `rembg U2Netp release
<https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx>`_：
4,574,861 bytes，SHA-256 如上，MD5 ``8e83ca70e441ab06c318d82300c84806``。
此 hash 僅對該檔案生效；其他模型請填自己的 hash。
`rembg 的 U2Netp session <https://github.com/danielgatis/rembg/blob/main/rembg/sessions/u2netp.py>`_
記載相同來源與 MD5。`U²-Net 原始碼的 Apache-2.0 授權
<https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE>`_ 不代表已確認每個轉換後
checkpoint、訓練素材與再散布情境的權利。repo 不包含或自動下載權重。

驗證方式與尚缺項目
----------------------------------------

.. code-block:: console

   python -m pytest tests/ae -q
   python -m black --check --diff .
   python -m isort --check-only --diff moviepy docs/conf.py examples tests benchmarks/ae_motion_bench.py
   python -m flake8 --show-source --ignore=E501 moviepy docs/conf.py examples tests benchmarks/ae_motion_bench.py
   python -m benchmarks.ae_motion_bench --output output/benchmark.json

實際後端測試需自行設定 ``MOVIEPY_AE_TEST_BLENDER=1`` 及
``MOVIEPY_AE_TEST_U2NET`` 為模型檔路徑，再跑 ``tests/ae/test_local_backends.py``。
預設 AE suite 清楚標記這兩項為 skipped；CI 不會暗中下載 Blender 或模型。
新增 AE workflow 覆蓋 Linux／Windows 和 Python 3.9／3.12，遠端執行狀態須以
發布後對應 commit 的 Actions 結果確認，本機成功不等於 CI 已通過。

高斯濾波改為單次 separable 呼叫並快取小型 immutable kernel；SDR 等價測試
與 signed／HDR 保留測試分開驗證。benchmark 報告完整取樣與中位數，
Gaussian 數字只涵蓋濾波；粒子數字涵蓋完整 source_buffer，均不含影片編碼。
不要據此宣稱固定 FPS 或完整影片加速比例。

尚未提供生成式 AI 內容感知填補、原生 SBSAR、通用 3D 模型匯入、
AE 文字 selector／完整排版、物理煙火或完整 AE 專案相容性。
AI 填補仍需要另行選定且獲授權的模型與執行環境；沒有用傳統填補測試替代。
