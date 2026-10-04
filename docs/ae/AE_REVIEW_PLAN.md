# AE_REVIEW_PLAN.md — feat/ae-parity 審查與待辦（2026-10-04）

審查對象：`feat/ae-parity` @ `6d4c246`（WS-03 / 04 / 05 / 06 + 範例）。
規格依據：`docs/AE_PARITY_SPEC.md`。
隨附 `test_review_20261004.py`：8 個測試，目前**全部 RED**，對應下方 R1–R6。

## 0. 現況

| 項目 | 結果（Linux sandbox、Python 3.12、單核） |
|---|---|
| `pytest tests/ae` | 1108 passed |
| `pytest tests --ignore=tests/ae` | 606 passed, 2 skipped, 1 xpassed |
| 覆蓋率（`moviepy/ae` 整體，含 branch） | 96% |
| black / isort | 通過 |
| flake8（嚴格） | 1 項：`moviepy/ae/blend/alpha.py:5` E501 |
| 函式 > 50 行 / 檔案 > 800 行 | 無 |
| 既有檔案修改 | 僅 `VideoClip.to_ae_layer`、`tests/conftest.py`，符合 §9.4 |

整體架構符合 §3.3 管線，可直接往下做。下列問題依優先序處理。

## 1. 執行規則

1. 先把 `test_review_20261004.py` 放進 `tests/ae/`，確認 8 個 RED。
2. 每個 R 項一個 commit（`fix(ae): ...`），該項測試轉 GREEN，其餘測試不得退步。
3. 不得為了通過而修改這 8 個測試的斷言；R6 的計數方式若因重構失效，可改寫計數手法，但「次數上限」不變。
4. 每個 commit 後跑：`pytest tests/ae -q`、`black --check`、`isort --check`、`flake8`。
5. P / D 項完成後更新對應 `docs/ae/wsXX.rst`。

---

## 2. 正確性缺陷（必修，已有 RED 測試）

### R1 — Dissolve 雜訊隨 buffer 邊界改變
- **現象**：`blend(base, layer, "dissolve")` 先裁切再混合，與先混合再裁切的結果不同（最大差 1.0）。合成層級上，ROI 渲染 ≠ 完整渲染的裁切。
- **原因**：`blend/modes.py::_noise` 以 `generator.random((h, w))` 產生與來源矩形等大的雜訊；Renderer 會先把圖層裁到 ROI，所以雜訊原點跟著移動。圖層部分移出畫面時圖樣也會跳動。
- **修法**：改成世界座標的整數 hash，$n(x, y) = \mathrm{hash}(x, y, \text{seed})$，seed 只從 `ctx.rng_for(layer_id)` 取一個整數。刪除 `_crop_noise`。
- **驗收**：`test_r1_*`（2 個）。

### R2 — 同名圖層共用亂數流
- **現象**：兩個都叫 `"Solid"` 的 50% Dissolve 圖層，下層可見比例為 0.000（獨立雜訊應約 0.25）。同名圖層的 `wiggle(2, 20)` 也輸出完全相同的位移。
- **原因**：`renderer.py` 與 `effects/adjustment.py` 傳 `layer_id=layer.name`；`Layer.expression_bindings` 的 `layer_id` 也是 name。`add_solid` 預設名稱固定，極易重複。
- **修法**：`Layer` 新增穩定且唯一的 `id`（建立時指派，重排圖層不變，可序列化）。上述三處改用它。不要用 `index`，重排會改變圖樣。
- **驗收**：`test_r2_*`；另補一個同名 wiggle 圖層位移不同的測試。

### R3 — Adjustment Layer 效果在 ROI 邊緣錯誤
- **現象**：Adjustment + `GaussianBlur(12)`，ROI 渲染與完整渲染裁切最大差 0.7156；`repeat_edge_pixels=True` 時差 0.0431。
- **原因**：`Renderer._composite` 先把 accumulator 裁到 ROI 再套效果，模糊在 ROI 邊界讀到透明。
- **修法**：合成前掃描 adjustment layer 的 active effects，以 `bounds_expand()` 累加外擴量，將下方圖層的 ROI 外擴（上限為畫面範圍），效果套完再裁回。邊緣複製必須發生在畫面邊界而非 ROI 邊界。
- **驗收**：`test_r3_*`；補 `repeat_edge_pixels=True` 的版本。

### R4 — Adjustment Layer 效果未隨 `resolution_scale` 縮放
- **現象**：同一個 `GaussianBlur(12)`，換算回全解析度的 10–90% 邊緣寬度：scale 1.0 = 23 px、0.5 = 48 px、0.25 = 112 px。半解析度預覽會比成品模糊兩倍。
- **原因**：圖層效果在來源空間（全解析度）執行所以正確；adjustment 效果在縮放後的合成空間執行，但像素單位參數沒有乘上 scale。
- **修法**：`AEEffect.process` 接收「像素縮放係數」（圖層空間 = 1.0，合成空間 = `ctx.resolution_scale`）；`Param` 增加 `unit="px"` 標記，由基底類別統一縮放，不要逐效果手寫。效果自帶的 mask 一併處理。
- **驗收**：`test_r4_*`（容差 25%）。

### R5 — `best` 品質下旋轉圖層的 ROI 不一致
- **現象**：80×60 solid 旋轉 30°，`quality="best"` 時 ROI 渲染與完整渲染在 12 個像素上差 0.018–0.035；`draft` 為 0。
- **線索**：`Renderer.render_layer` 走 `warp_buffer(..., bounds=target)` 與「整張 warp 再 crop」兩條路徑，cubic 下結果不同。先查 `warp.py` 中 cubic 的 clamp / 邊界處理是否依賴輸出範圍。
- **驗收**：`test_r5_*`（`atol=1e-5`）。

### R6 — 單次 render 內重複計算
- **現象**：
  - 一個 matte 圖層被 3 個圖層引用 → 每格渲染 3 次。
  - 兩層疊加的 `Echo(number_of_echoes=6)` adjustment：最底層圖層每格渲染 49 次，實際只有 13 個不同時間點。1080p 單層 echo 2142 ms，兩層 7971 ms。
- **原因**：規格 WS-03 要求「同一 (layer, t) 結果在單次 render 內快取」，目前未實作；`ctx.cache` 未使用。
- **修法**：`Renderer.render` 建立一次性的快取物件，往下傳遞。key 至少含 `(id(layer), t, roi, resolution_scale, quality)`，涵蓋 `_matted_layer` 與 `_composite(below=...)` 的結果。render 結束即丟棄（圖層可變，不可跨 render 保留）。
- **驗收**：`test_r6_*`（2 個）。

---

## 3. 效能（P1）

量測皆為 1920×1080、sandbox 單核，數字只供相對比較，請在參考機重測。

### P1 — 混合模式未達規格 ≤ 25 ms
| 模式 | ms |
|---|---:|
| normal | 24 |
| stencil_alpha | 56 |
| alpha_add | 117 |
| dissolve | 144 |
| multiply | 164 |
| screen | 177 |
| overlay | 239 |
| soft_light | 284 |
| hue | 401 |

方向：(a) `alpha.replicate` 把 alpha 複製成 3 通道、`split` 再複製一次，改用 channel-first 平面讓 `(H, W)` alpha 直接廣播；(b) 減少暫存陣列，改就地運算；(c) Multiply / Add / Darken / Lighten 有不需除 alpha 的 premultiplied 形式；(d) 仍不足時走 §3.6 的 `ae-fast`（numba）。
目標：separable 模式 ≤ 2× Normal。

### P2 — 每個圖層都重配整張畫面
- 20 個 600×400 圖層疊在 1080p：507 ms（約 25 ms／層，與圖層面積無關）；20 個全幅圖層 644 ms。
- 原因：`blend()` 每次 `np.array(base.expand_to(bounds).rgba)` 複製整個 accumulator。
- 修法：Renderer 持有可變 accumulator，新增內部 `blend_into(canvas, layer, ...)` 只改寫交集區；公開的 `blend()` 維持不可變語意並包一層。
- 目標：成本與圖層面積成正比；上述 20 小圖層 < 150 ms。

### P3 — GaussianBlur 大半徑過慢
- Adjustment + `GaussianBlur(40)`：1048 ms。`sepFilter2D` 在大 kernel 的 4 通道 float32 上很慢。
- 修法：大 σ 時先降採樣再模糊再升採樣，或改 `cv2.GaussianBlur` / 三次 box blur 近似；`draft` 品質可更激進。需保留 σ 剖面測試。
- 目標：≤ 150 ms。

### P4 — Benchmark 未涵蓋 WS-03–06
`benchmarks/ae_bench.py` 仍只有 4 個 WS-00 案例，`benchmarks/results/` 無新結果。新增案例：各混合模式、20 層合成、羽化遮罩、adjustment blur、echo；在參考機產出 JSON 並套用 §9.3 的 20% 回歸門檻。

---

## 4. 行為與設計決策（P2）

### D1 — Stencil / Silhouette 在 opacity 0 不連續
`stencil_alpha` 圖層 opacity 100 / 50 / 1 / 0 時，下方結果 alpha 為 1.0 / 0.5 / 0.01 / **1.0**。opacity 動畫到 0 會整個畫面跳回來。原因是 `render_layer` 在 `opacity <= 0` 直接略過圖層。請先在 AE 實測該行為，再決定：改成連續，或保留並寫進 `ws04.rst`。我沒有 AE 可對照，無法判定哪個正確。

### D2 — Composition 沒有音訊
`comp.add_clip(VideoFileClip(...))` 後 `comp.audio` 為 `None`，`write_videofile` 輸出無聲。完整功能屬 WS-25，但目前文件未提。至少在 `ws03.rst` 註明；可選做最小版本：stretch = 100 的 AVLayer 依 `start_time` 組成 `CompositeAudioClip`。

### D3 — alpha 快取只比對時間
`Composition._frame_buffer(reuse=True)` 以時間為唯一 key。`get_frame(t)` 後修改圖層再取 `mask.get_frame(t)` 會拿到舊 alpha。加一個圖層 / 設定變更的 revision 計數納入 key。

### D4 — HDR 被效果裁掉
`effects/blur/gaussian_blur.py::gaussian` 結尾 `np.clip(out, 0, 1)`，Add 模式產生的 > 1 數值經 adjustment blur 後會被截斷。WS-26 前可接受，但請寫進已知限制；修 P3 時改成只修正 ulp 誤差而非硬裁。

### D5 — 與規格的落差
- 規格的 `MaskPath` 實際是 `properties.values.PathValue`；請在 `moviepy.ae.masks` 匯出 `MaskPath` 別名。
- 規格範例的 `ae.wiggle(...)` 不存在於 `moviepy.ae`。
- `masks/path.py::flatten` 為等距細分，規格寫「自適應細分」；面積誤差測試已過，補文件說明即可。
- 記憶體測試為 160×90 + `tracemalloc`，規格為 20 層 1080p、200 格、RSS 成長 < 5%。補一個標記為 slow 的版本。

---

## 5. 流程缺口（P3）

1. `docs/ae/ws03–ws06_acceptance.md` 不存在（WS-00–02 都有）。每份需列：測試數、覆蓋率、效能量測、未達標項目。
2. `masks/path.py` 覆蓋率 88%，低於 WS-05 要求的 90%。
3. `CHANGELOG.md` 未更新。
4. `moviepy/ae/blend/alpha.py:5` E501。
5. §9.2 要求的獨立 reviewer 審查，WS-03–06 尚無記錄。

---

## 6. 建議順序

`R2 → R1 → R6 → R3 → R4 → R5 → P2 → P1 → P3 → P4 → D1–D5 → 流程缺口`

R2 先做是因為 R1 的 seed 依賴穩定的圖層 id；R6 的快取物件會被 R3 的 ROI 外擴用到；P2 會改動 `blend` 內部，放在 P1 之前可避免重工。

完成 R1–R6 與 P1–P3 後再開 WS-07（Motion Blur 會把每格成本乘上取樣數，現有效能會被放大）。
