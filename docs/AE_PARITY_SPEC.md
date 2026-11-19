# MoviePy → After Effects 級功能對齊規格書 (AE Parity Spec)

> 版本：v1.0　日期：2026-10-02　基準 commit：`211e4b1`（MoviePy 2.2.0）
> 用途：給 Agents 實作的工作規格。每個 Workstream（WS）可獨立分派，依賴關係見 §7。
> 語言：說明用中文；識別字、API、檔名維持英文。

---

## 0. 目標、原則、非目標

### 0.1 目標
讓 MoviePy 具備 **「程式化版 After Effects」** 的核心能力：
關鍵影格動畫、圖層／合成（Composition）巢狀、變形、遮罩、Track Matte、混合模式、
效果堆疊、調整圖層、運動模糊、時間重映射、形狀／文字動畫、3D 圖層與攝影機、追蹤、粒子、
色彩管理、渲染佇列。

### 0.2 設計原則
1. **不破壞既有 API**：現有 `VideoClip` / `CompositeVideoClip` / `Effect` / `with_effects` 行為與測試全部維持通過。新功能放在新套件 `moviepy/ae/`，既有檔案只做最小的「掛鉤」修改。
2. **可互通**：任何現有 `VideoClip` 可包成 AE 圖層；`ae.Composition` 本身是 `VideoClip` 子類，可丟回 `write_videofile`、`CompositeVideoClip`。
3. **Pull 模型保留**：維持 `frame_function(t) -> ndarray` 的惰性求值，不引入全域狀態。
4. **不可變語意**：與 MoviePy 的 `outplace` 風格一致，所有 `with_*` 回傳新物件。
5. **功能對齊而非二進位對齊**：只實作行為（參考公開文件與數學定義），**不複製 Adobe 程式碼／素材／預設資源**，不支援 `.aep` 讀寫。`Adobe`、`After Effects` 為商標，程式碼與文件中只做「對齊目標」描述，不用於套件名稱。
6. **可驗證**：每個效果都有解析解或 golden image 測試，不以「看起來像」當驗收。

### 0.3 非目標（本規格不做）
- 圖形化 GUI 時間軸編輯器（只提供程式 API、RAM 預覽與簡易預覽視窗）。
- Content-Aware Fill、Roto Brush 等需大型 ML 模型的功能（僅在 WS-23 留「外掛接口」）。
- Adobe 專屬第三方外掛（Trapcode、Element 3D 等）。

---

## 1. MoviePy 現況盤點（已讀程式碼確認）

| 領域 | 現況 | 位置 |
|---|---|---|
| 影格模型 | `Clip.get_frame(t) -> ndarray`，**uint8 RGB**；遮罩是**獨立 float [0,1]** 的 `VideoClip(is_mask=True)` | `moviepy/Clip.py`, `moviepy/video/VideoClip.py` |
| 合成 | `CompositeVideoClip.frame_function` 逐 clip 呼叫 `clip.compose_on(bg, t, bg_mask)`；僅有 **Normal（over）** 混合，每層都量化回 uint8 | `moviepy/video/compositing/CompositeVideoClip.py`, `VideoClip.compose_on` (~L720) |
| 變形 | `with_position`（可為 `t -> (x,y)`）、`resized`、`rotated`、`cropped`；**無 anchor point、無獨立 scale 軸、無 parenting** | `VideoClip.py`, `video/fx/Resize.py`, `Rotate.py` |
| 效果 | `Effect` 抽象類 + `@dataclass` 子類 + `clip.with_effects([...])`；約 33 個影像 fx、7 個音訊 fx（淡入淡出、模糊 HeadBlur、色彩 LumContrast／Gamma／Invert、時間 MultiplySpeed／TimeMirror、Loop、Freeze 等） | `moviepy/Effect.py`, `moviepy/video/fx/`, `moviepy/audio/fx/` |
| 動畫 | 只有「參數接受 `t -> value` 函式」，**無關鍵影格、無緩動曲線、無運算式、無 graph editor 概念** | — |
| 遮罩 | 灰階 mask clip、`MaskColor`、`MasksAnd/Or`；**無向量遮罩、羽化、擴展、遮罩模式、Track Matte** | — |
| 文字／形狀 | `TextClip`（Pillow）；`video/tools/drawing.py` 有基本繪圖；**無形狀圖層、無文字動畫器** | — |
| 時間 | `with_speed_scaled`/`MultiplySpeed`、`Loop`、`Freeze`、`TimeMirror`；**無 Time Remap 曲線、無 Frame Blending、無運動模糊** | — |
| 3D／攝影機／粒子／追蹤 | 無 | — |
| 色彩 | 8-bit sRGB 假設；無線性光、無 LUT、無 ICC/OCIO、無 16/32bpc | — |
| I/O | ffmpeg 讀寫、ImageSequence、GIF；無 EXR、無 alpha 輸出編碼預設（ProRes4444／PNG seq 需手動）、無 Render Queue | `moviepy/video/io/` |
| 預覽 | `ffplay` 預覽；無快取式 RAM Preview | `video/io/ffplay_previewer.py` |
| 依賴 | numpy, pillow, opencv-python-headless, imageio(-ffmpeg), proglog, decorator | `pyproject.toml` |
| 測試 | pytest；`tests/test_fx.py` 等；`conftest.py` 會掃描 doc examples | `tests/` |
| 風格 | black、flake8（absolute import、docstring）、numpydoc | `pyproject.toml` |

### 1.1 必須先解的結構性限制
1. **8-bit 逐層量化**：每次 `compose_on` 都 `astype(uint8)`，多層疊加（混合模式、發光、陰影）會產生 banding。→ WS-00 引入 float32 管線。
2. **Mask 與 RGB 分離**：AE 的圖層是 RGBA（含 alpha）並有 straight/premultiplied 之分。→ WS-00 定義內部 `Buffer`（premultiplied RGBA float32）。
3. **無邊界外像素概念**：AE 圖層可大於／偏離合成範圍，效果（Glow、Blur）會擴張邊界。→ `Buffer` 帶 `offset`。
4. **無圖層屬性系統**：參數是 Python 函式，無法做 graph editor、keyframe 資料序列化、表達式。→ WS-01。

---

## 2. 差距總表（AE 功能 → 對應 WS → 優先級）

優先級：**P0** 基礎（必先）／**P1** 核心（缺了不像 AE）／**P2** 廣度（效果庫）／**P3** 進階。

| AE 功能領域 | MoviePy 現況 | WS | 優先 |
|---|---|---|---|
| 渲染核心 float32 premult RGBA、ROI、快取 | 無 | WS-00 | P0 |
| 屬性／關鍵影格／緩動／貝茲曲線／表達式 | 僅 `t->value` | WS-01 | P0 |
| 圖層變形（Anchor/Position/Scale/Rotation/Opacity）、Parenting、Null | 部分 | WS-02 | P1 |
| Composition 巢狀（Pre-compose）、Work area、Markers、圖層時間偏移／延展 | 部分（CompositeVideoClip） | WS-03 | P1 |
| 混合模式（~38 種）、Alpha 模式 | 僅 Normal | WS-04 | P1 |
| 遮罩（貝茲路徑、模式、羽化、擴展、不透明度）、Track Matte（Alpha/Luma/反相） | 僅灰階 mask | WS-05 | P1 |
| 效果堆疊、調整圖層、效果參數動畫、效果註冊表 | `with_effects` | WS-06 | P1 |
| Time Remap、Frame Blending、Motion Blur、Time-stretch | 部分 | WS-07 | P1 |
| 模糊與銳化 | 僅 HeadBlur | WS-10 | P2 |
| 色彩校正（Curves、Levels、Hue/Sat、Color Balance、LUT…） | 少量 | WS-11 | P2 |
| 扭曲（Corner Pin、Wave Warp、Displacement、Mesh Warp…） | 無 | WS-12 | P2 |
| 生成（Gradient Ramp、Fractal Noise、Grid、Lens Flare…） | 無 | WS-13 | P2 |
| 去背／摳像（Keylight 類、Luma Key、Spill Suppress、Choker） | 僅 MaskColor | WS-14 | P2 |
| 風格化（Glow、Mosaic、Posterize、Find Edges、Motion Tile…） | 少量 | WS-15 | P2 |
| 雜訊與顆粒 | 無 | WS-16 | P2 |
| 轉場（Linear/Radial/Gradient/Iris Wipe、Block Dissolve、Card Wipe…） | CrossFade | WS-17 | P2 |
| 時間類效果（Echo、Posterize Time、Time Difference） | 無 | WS-18 | P2 |
| 透視類（Drop Shadow、Bevel Alpha、Radial Shadow） | 無 | WS-19 | P2 |
| 形狀圖層（Rect/Ellipse/Star/Path、Fill/Stroke/Gradient、Trim Paths、Repeater…） | 無 | WS-20 | P3→實務 P1.5 |
| 文字動畫器（Range Selector、逐字元屬性、Text on Path） | 無 | WS-21 | P3→實務 P1.5 |
| 3D 圖層、攝影機、燈光、陰影、景深 | 無 | WS-22 | P3 |
| 追蹤（Point/Planar）、Warp Stabilizer | 無 | WS-23 | P3 |
| 粒子系統 | 無 | WS-24 | P3 |
| 音訊圖層（Levels 關鍵影格、波形、Audio→Keyframes、EQ/Delay/Reverb） | 部分 | WS-25 | P2 |
| 色彩管理（16/32bpc、線性光、LUT、OCIO、EXR） | 無 | WS-26 | P2 |
| 專案檔（JSON）、Lottie 匯入、Render Queue、Proxy、多程序渲染 | 無 | WS-27 | P2 |
| 預覽（RAM Preview、解析度分數、Draft） | ffplay | WS-28 | P2 |
| 文件、範例、基準測試 | 部分 | WS-29 | 全程 |

---

## 3. 架構決策（所有 WS 必須遵守）

### 3.1 套件配置
```
moviepy/ae/
├── __init__.py            # 公開 API 匯出（延遲載入重依賴）
├── buffer.py              # Buffer：premultiplied RGBA float32 + offset
├── context.py             # RenderContext：time, fps, resolution_scale, quality, rng_seed, cache
├── color/                 # 色彩空間、LUT、ICC/OCIO、bit depth（WS-26）
├── properties/            # Property, Keyframe, Easing, Expression（WS-01）
├── layers/                # Layer, AVLayer, NullLayer, SolidLayer, AdjustmentLayer,
│                          # ShapeLayer, TextLayer, CameraLayer, LightLayer（WS-02/20/21/22）
├── composition.py         # Composition（VideoClip 子類）、Renderer（WS-03）
├── transform.py           # Transform（WS-02）
├── blend/                 # blend modes（WS-04）
├── masks/                 # Mask, MaskPath, TrackMatte（WS-05）
├── effects/               # 效果基底與註冊表 + 子目錄 blur/ color/ distort/ generate/
│   │                      #   keying/ stylize/ noise/ transition/ time/ perspective/（WS-06, 10–19）
│   ├── base.py            # AEEffect
│   └── registry.py
├── time/                  # time remap, frame blend, motion blur（WS-07）
├── vector/                # 向量引擎、形狀運算（WS-20）
├── text/                  # 文字動畫器（WS-21）
├── three_d/               # 攝影機、燈光、投影（WS-22）
├── tracking/              # 追蹤、穩定（WS-23）
├── particles/             # 粒子（WS-24）
├── audio/                 # 音訊圖層、Audio→Keyframes（WS-25）
├── io/                    # 專案 JSON、Lottie、EXR、Render Queue（WS-27）
└── preview/               # RAM Preview、快取（WS-28）
tests/ae/                  # 對應測試；golden 圖放 tests/ae/golden/
docs/ae/                   # 使用者文件（rst，併入 Sphinx）
```
**檔案大小**：單檔 ≤ 400 行為佳、上限 800；函式 < 50 行；巢狀 ≤ 4 層（遵守專案 coding-style）。

### 3.2 影像資料模型（WS-00 實作，其餘 WS 只使用）
```python
@dataclass(frozen=True)
class Buffer:
    rgba: np.ndarray      # (H, W, 4) float32，**premultiplied**，線性或 sRGB 由 color_space 標示
    offset: tuple[int, int] = (0, 0)   # 相對合成左上角，可為負；允許超出合成邊界
    color_space: str = "srgb"          # "srgb" | "linear" | "ocio:<name>"
    # 工具：from_uint8_rgb(frame, mask=None), to_uint8_rgb(bg=None), straight(), with_(…)
```
- 所有 AE 效果、混合、遮罩皆在 `Buffer` 上運算；僅在**最終輸出**量化為 uint8（或 16-bit）。
- 現有 `VideoClip` 轉 `Buffer`：`Buffer.from_clip(clip, t)`（RGB uint8 / 255 + mask → straight→premult）。
- 預設工作空間 `srgb`（與 AE 8bpc 預設相同、先求行為一致）；`project.linear_light=True` 時於 `color/` 轉線性運算（WS-26）。

### 3.3 渲染管線
```
Composition.frame_function(t)
  → Renderer.render(comp, t, ctx)
      for layer in comp.layers_sorted(bottom→top) if layer.is_active(t):
          buf  = layer.source_buffer(t, ctx)          # 素材 / 子 Composition / 形狀 / 文字
          buf  = layer.apply_masks(buf, t, ctx)       # WS-05
          buf  = layer.apply_effects(buf, t, ctx)     # WS-06（adjustment layer 例外：作用於其下方累積結果）
          buf  = layer.transform.apply(buf, t, ctx)   # WS-02（含 motion blur 子採樣 WS-07）
          buf  = layer.apply_track_matte(buf, t, ctx) # WS-05
          acc  = blend(acc, buf, layer.blend_mode)    # WS-04
      return acc.to_uint8_rgb()  (+ 輸出 mask 供 Composition.mask)
```
- `RenderContext` 傳遞：`t`、`fps`、`resolution_scale`（0.25/0.5/1）、`quality`（draft/best）、`rng_seed`（確定性雜訊）、`cache`（WS-28）、`shutter`（WS-07）。
- **確定性**：相同 (comp, t, ctx) 必須輸出位元相同結果（隨機效果以 `rng_seed + layer_id + frame_index` 播種）。

### 3.4 屬性系統（WS-01 實作）
```python
Prop = Property[T]            # 靜態值 / Keyframes / callable / Expression 皆可
prop.value_at(t: float) -> T
prop.set_keyframes([Keyframe(t, value, in_ease, out_ease, interp="bezier"|"linear"|"hold", ...)])
```
- 所有效果參數、Transform、Mask、Shape 屬性一律型別為 `Property`。接受裸值（自動包成 constant）、`t -> value` 函式（向下相容）、`Keyframes`、`Expression`。
- 可序列化為 JSON（WS-27 專案檔使用）。

### 3.5 效果基底（WS-06 實作）
```python
class AEEffect(ABC):
    name: ClassVar[str]                 # 如 "Gaussian Blur"，註冊表 key
    category: ClassVar[str]             # "Blur & Sharpen" 等，對齊 AE 分類
    params: dict[str, Property]         # 皆為可動畫屬性
    def bounds_expand(self, buf_shape, t, ctx) -> tuple[int,int,int,int]   # 效果擴張邊界（Glow、Blur）
    @abstractmethod
    def render(self, src: Buffer, t: float, ctx: RenderContext) -> Buffer
    # 可選：render_gpu（P3），temporal_window()（需要前後影格的效果，如 Echo／Frame Blend）
```
- **橋接現有 Effect**：提供 `ae.effects.from_moviepy_effect(Effect)`，讓既有 `video/fx/*` 可放進 AE 效果堆疊。
- 同時提供 MoviePy 風格捷徑：`clip.with_effects([ae.fx.GaussianBlur(blurriness=20)])` 對單一 clip 直接可用（內部包成單圖層 Composition）。

### 3.6 依賴政策
- 核心（WS-00～07）**只用** numpy、opencv（已有）、pillow（已有）。
- 可選依賴以 extras 宣告於 `pyproject.toml`：`ae-vector`（`skia-python` 或 `pycairo`，向量引擎首選 skia；無則退回 cv2 + 自實作貝茲展平）、`ae-fast`（`numba`）、`ae-gpu`（`moderngl`）、`ae-exr`（`OpenEXR` 或 imageio 插件）、`ae-color`（`PyOpenColorIO`）、`ae-track`（`scipy`）。
- 新增依賴前須在 PR 說明：理由、授權、Windows 安裝可行性。缺可選依賴時必須給出明確錯誤並有純 numpy 退路（效能可較差）。

### 3.7 API 風格
- 沿用 MoviePy 慣例：效果類 `PascalCase` dataclass；方法 `with_*` 回傳新物件；numpydoc docstring；絕對 import。
- 對 AE 使用者友善的命名：參數名盡量與 AE 效果面板相同（snake_case 化），如 `blurriness`、`blur_dimensions`、`edge_behavior`。
- 範例：
```python
import moviepy.ae as ae
comp = ae.Composition(size=(1920,1080), fps=30, duration=10)
bg   = comp.add_solid("bg", color=(20,20,30))
vid  = comp.add_clip(VideoFileClip("a.mp4"), name="footage")
vid.transform.position.set_keyframes([(0,(960,540)), (2,(400,300), ae.Ease.bezier(0.2,0,0.2,1))])
vid.transform.scale = (100,100)
vid.blend_mode = ae.BlendMode.SCREEN
vid.effects.add(ae.fx.GaussianBlur(blurriness=ae.wiggle(freq=2, amp=8)))
vid.masks.add(ae.Mask.ellipse(center=(960,540), size=(600,600), feather=40, mode="add"))
comp.write_videofile("out.mp4")
```

---

## 4. Workstreams — Phase 0：基礎（必須先完成並合併）

### WS-00　核心渲染模型與測試基建（P0）
**範圍**：`ae/buffer.py`, `ae/context.py`, `ae/__init__.py`, `tests/ae/conftest.py`, golden 測試工具。
**交付**：
- `Buffer`（見 §3.2）＋ 轉換：`from_uint8_rgb`、`from_clip`、`to_uint8_rgb`、`premultiply/unpremultiply`、`crop/pad/expand_to`、`composite_over`（float32，作為 Normal 基準實作）。
- `RenderContext`（見 §3.3）。
- 測試工具 `assert_image_close(actual, golden_path, tol)`；`--update-golden` pytest 旗標；golden 以 PNG 存放，附 SHA 與產生程式碼註記。
- 效能基準腳本 `benchmarks/ae_bench.py`（1080p 單層、10 層 Normal、模糊 σ=20）。
**驗收**：
- round-trip `uint8 → Buffer → uint8` 位元相同；`over` 與 `VideoClip.compose_on` 在不透明／半透明案例誤差 ≤ 1/255。
- 1080p Normal 合成 10 層 ≥ 10 fps（單執行緒 CPU，參考機型須在 README 註記）。
**依賴**：無。

### WS-01　屬性、關鍵影格、緩動、表達式（P0）
**範圍**：`ae/properties/{property,keyframe,easing,expression,spatial}.py`
**交付**：
1. `Property[T]`：型別含 float、`Vec2/Vec3`、color、bool、enum、path。靜態／關鍵影格／callable／表達式四種來源。
2. `Keyframe`：`time, value, interp ∈ {linear, hold, bezier, auto_bezier, continuous_bezier}`，**時間插值**（`in_influence/out_influence/in_speed/out_speed`，對齊 AE speed graph 的 influence 模型）；**空間插值**（位置屬性的貝茲運動路徑：`in_tangent/out_tangent`，弧長參數化使速度正確）。
3. `Ease` 預設：`linear, ease_in, ease_out, ease_in_out`、`bezier(x1,y1,x2,y2)`（CSS 同構）、`easy_ease`（33.33% influence，等同 AE Easy Ease）、`hold`。
4. 多維屬性可 **Separate Dimensions**（x、y、z 各自關鍵影格）。
5. **Roving keyframes**（依路徑自動調整時間以保持等速）。
6. **Expression 引擎**（Python 版，非 JS）：沙盒化 `eval`（白名單 AST，禁 `import`、`__`、檔案／網路），提供 AE 慣用函式：`time, value, index, thisLayer, thisComp, valueAtTime(t), velocityAtTime(t), wiggle(freq, amp, octaves=1, amp_mult=0.5, t=time), loopOut(type="cycle"|"pingpong"|"offset"|"continue", n=0), loopIn(...), linear(t,tMin,tMax,v1,v2), ease/easeIn/easeOut, clamp, interp helpers, length, normalize, dot, cross, lookAt, rgbToHsl/hslToRgb, random/gaussRandom/noise(seeded), timeToFrames/framesToTime`。
7. **Audio→Keyframes**介面預留（WS-25 實作）。
8. JSON 序列化／反序列化（含版本欄位 `"schema": 1`）。
**驗收**：
- 單元測試涵蓋：每種插值的端點、中點、單調性；bezier influence=33.33% 時等同 CSS `cubic-bezier(.333,0,.667,1)`；空間路徑等速時 `|dP/dt|` 變異 < 1%。
- `wiggle` 同 seed 同 t 必須相同；頻譜能量集中於 `freq` 附近（測試用 FFT 斷言）。
- 表達式沙盒安全測試：`__import__('os')`、`open()`、`().__class__` 全部拋 `ExpressionError`。
- 覆蓋率 ≥ 90%。
**依賴**：無（可與 WS-00 並行）。

---

## 5. Phase 1：核心合成（P0 完成後；WS-02/04/05/07 可並行，WS-03 先於 WS-06）

### WS-02　圖層與變形（P1）
**範圍**：`ae/layers/{base,av,null,solid}.py`, `ae/transform.py`
**交付**：
- `Transform`：`anchor_point, position(separable), scale(%, 可負值=翻轉, 可鎖比例), rotation(度, 可超過 360 圈數), opacity(0–100)`；3D 預留欄位 `orientation, x/y/z_rotation, position_z, scale_z`（WS-22 啟用）。
- 變換矩陣組合順序與 AE 相同：`translate(position) · rotate · scale · translate(-anchor)`；以 3×3 仿射矩陣＋`cv2.warpAffine`（float32、`INTER_CUBIC`/`LINEAR` 可選、`best/draft` 品質）。
- **Parenting**（圖層父子，含「僅繼承 Transform、不繼承 opacity」的 AE 行為）、**Null Layer**、**Solid Layer**、**Auto-Orient along path**。
- 圖層時間：`in_point, out_point, start_time(偏移), stretch(%)`（負 stretch = 反向）。
- 圖層開關：`enabled(visible), solo, shy, locked, guide`（`solo`/`enabled` 影響渲染）。
- **Collapse Transformations / Continuously Rasterize** 對子 comp 與向量層的語意（先實作旗標，行為在 WS-03/20 完成）。
**驗收**：
- 旋轉 90°/180° 與 numpy `rot90` 完全一致；`anchor` 在中心時旋轉不位移；scale 負值等於鏡射；父層旋轉 + 子層位置的數值與手算矩陣一致（誤差 < 0.01 px）。
- 子像素位移（0.5 px）與 `INTER_LINEAR` 解析解一致。

### WS-03　Composition 與渲染器（P1）
**範圍**：`ae/composition.py`, `ae/renderer.py`
**交付**：
- `Composition(VideoClip)`：`size, fps, duration, bg_color, work_area, markers, layers`；`add_clip/add_solid/add_null/add_comp/add_adjustment/add_text/add_shape`；`layers` 可重排；`Composition` 可作為另一個 `Composition` 的圖層來源（**Pre-compose**），支援 `collapse_transformations` 與 **獨立解析度／fps 的子 comp**（時間取樣以子 comp 自身 fps）。
- `Renderer`（§3.3 管線）：ROI 渲染（只算輸出需要的矩形）、`resolution_scale`、圖層邊界裁剪、同一 (layer,t) 結果在單次 render 內快取。
- 輸出 mask 供 `Composition.mask`（讓 `write_videofile` 輸出含 alpha：PNG seq、ProRes 4444、WebM VP9 alpha）。
- `from_moviepy(CompositeVideoClip) -> Composition`（盡力轉換）。
- Markers（含 chapter / comment）與 `comp.work_area`。
**驗收**：
- 與現有 `CompositeVideoClip` 在「僅位置＋Normal」場景輸出相同（± 1/255）。
- 巢狀 3 層 comp、不同 fps（24 / 30）時間取樣正確（以標記幀號 clip 驗證）。
- 20 層 1080p comp 單幀渲染不記憶體洩漏（連續 200 幀 RSS 成長 < 5%）。

### WS-04　混合模式與 Alpha 模式（P1）
**範圍**：`ae/blend/{modes,alpha}.py`
**交付**：**全部** AE 混合模式（含 Classic 變體與 Stencil/Silhouette 系列）：
Normal, Dissolve, Dancing Dissolve｜Darken, Multiply, Color Burn, Classic Color Burn, Linear Burn, Darker Color｜Add, Lighten, Screen, Color Dodge, Classic Color Dodge, Linear Dodge, Lighter Color｜Overlay, Soft Light, Hard Light, Vivid Light, Linear Light, Pin Light, Hard Mix｜Difference, Classic Difference, Exclusion, Subtract, Divide｜Hue, Saturation, Color, Luminosity｜Stencil Alpha/Luma, Silhouette Alpha/Luma, Alpha Add, Luminescent Premul。
- 介面 `BlendMode` enum + `blend(base: Buffer, layer: Buffer, mode, opacity) -> Buffer`，皆對 premultiplied 輸入正確反預乘後運算；支援 `preserve_underlying_transparency`（AE 的 "Preserve Underlying Transparency" 開關）。
- Dissolve 使用確定性雜訊（`ctx.rng_seed`）。
**驗收**：
- 每個模式有解析測試向量（例如 Multiply(0.5,0.5)=0.25；Screen(0.5,0.5)=0.75；Overlay 於 base=0.25/0.75 兩分支），對照公開的 W3C Compositing and Blending Level 1 公式；AE 獨有模式（Classic*、Linear Light、Pin Light、Hard Mix、Add/Subtract/Divide）以文件公式實作並寫明公式出處於 docstring。
- 不透明度 0 → 不改變 base；100 → 等同無 opacity。
- 向量化，1080p 單一混合 ≤ 25 ms（參考機型）。

### WS-05　遮罩與 Track Matte（P1）
**範圍**：`ae/masks/{path,mask,rasterize,matte}.py`
**交付**：
- `MaskPath`：貝茲頂點（`vertex, in_tangent, out_tangent`）、`closed`；**可關鍵影格**（頂點數相同時做逐點插值；`Property[MaskPath]`）。
- `Mask`：`path, mode ∈ {none, add, subtract, intersect, lighten, darken, difference}, opacity, feather(x,y 分離), expansion, inverted`；`MaskStack` 依序以 AE 規則結合。
- 光柵化：貝茲展平（自適應細分）→ 超取樣抗鋸齒（4×4 或 skia 若可用）→ 高斯羽化（含 `feather_x != feather_y`）→ expansion（距離變換／形態學）。
- 便利建構：`Mask.rect/ellipse/rounded_rect/polygon/star`、`Mask.from_svg_path(d)`。
- **Track Matte**：`Alpha`, `Alpha Inverted`, `Luma`, `Luma Inverted`（使用 Rec.709 luma 係數 0.2126/0.7152/0.0722，可選 Rec.601）；matte 圖層自身可隱藏。
- 與 `VideoClip.mask` 互通（`mask.to_clip(size)`）。
- 預留 **variable mask feather**（P3）。
**驗收**：
- 圓形遮罩面積誤差 < 0.5%（解析 πr²）；feather σ 與高斯核參數符合：feather=F 時等效 σ = F/2（需註明對齊 AE 的經驗定義與誤差範圍）。
- 多遮罩 add/subtract/intersect 與布林代數一致（逐像素測試 α∈{0,1} 的情形）。
- Luma matte：白→不透明、黑→透明，中灰 0.5 → α=0.5（線性空間差異需有旗標與測試）。

### WS-06　效果框架、效果堆疊、調整圖層（P1）
**範圍**：`ae/effects/{base,registry,stack,adjustment,bridge}.py`
**交付**：
- `AEEffect`（§3.5）、`EffectStack`（順序、啟用/停用、`add/remove/move`、每個效果可有自己的 mask 與 `blend_with_original`（AE 的 Effect Opacity / Composite on Original））。
- **Adjustment Layer**：對其下方已累積的結果套用效果，並以自身 mask/opacity/blend 控制範圍。
- 邊界管理：呼叫 `bounds_expand()` 先擴張 `Buffer`（例如 Glow 外擴）。
- **註冊表**：`ae.fx.<Name>` 與 `registry.get("Gaussian Blur")`；`registry.list(category=...)`；效果 metadata（參數型別、範圍、預設值）可匯出 JSON，供未來 GUI／專案檔使用。
- `from_moviepy_effect()` 橋接（§3.5）。
- 需要時間鄰域的效果（Echo、Frame Blend、Time Difference）透過 `temporal_window() -> (before, after)` 宣告，由 Renderer 供應其他時刻的 `Buffer`。
- 效果檔案模板與「新增效果 checklist」寫入 `docs/ae/writing_effects.rst`。
**驗收**：
- 效果順序影響結果（A→B ≠ B→A）的測試；停用效果等同未加；調整圖層範圍以 mask 限制後，範圍外像素位元相同。
- 註冊表至少含 WS-10～19 全部效果；每個效果有 `name/category/params` metadata 完整性測試（自動掃描）。

### WS-07　時間：Time Remap、Frame Blending、Motion Blur（P1）
**範圍**：`ae/time/{remap,blend,motion_blur}.py`
**交付**：
- **Time Remap**：圖層屬性 `time_remap: Property[float]`（關鍵影格化的來源時間曲線，可倒轉／凍結／變速）。
- **Frame Blending**：`off | frame_mix | pixel_motion`（pixel motion 使用 `cv2.calcOpticalFlowFarneback` / DIS 光流做插幀，P2 品質；frame_mix 為線性混合）。
- **Motion Blur**：comp 層級開關 + 圖層開關；參數 `shutter_angle(預設180), shutter_phase(預設-90), samples_min(8), samples_max(64), adaptive`；在 shutter 區間內多次評估圖層 **Transform 與效果屬性**並累積平均（線性光選項）。僅在屬性有變化時才多重取樣以省時。
- `Stretch` 與 `Reverse` 與 WS-02 協作。
**驗收**：
- 水平勻速移動的白點，motion blur 後的拖尾長度 = `speed · shutter_angle/360 · frame_duration`（誤差 < 5%）。
- Time Remap 線性 0→duration 等同原片；常數值＝凍結；反向曲線＝時間反轉。
- Pixel Motion 對平移測試序列的 PSNR ≥ 35 dB（對比真實中間幀）。

---

## 6. Phase 2：效果庫與周邊（WS-06 完成後全部可並行）

> **每個效果的共同 DoD**：① 繼承 `AEEffect`、有 metadata；② 所有數值參數為 `Property`（可關鍵影格）；③ premultiplied 正確（先反預乘再運算再預乘）；④ 至少 1 個解析／golden 測試 + 1 個「參數 0 / 預設時為恆等」測試；⑤ docstring 含參數表與對齊 AE 面板名稱；⑥ draft 品質時可降階（例如減少取樣）；⑦ 1080p 單幀預算見各表。
> 下列以「效果名 — 關鍵參數 — 實作提示」列出，先做 ★ 項目。

### WS-10　Blur & Sharpen（`effects/blur/`）
| 效果 | 關鍵參數 | 提示 |
|---|---|---|
| ★Gaussian Blur | blurriness, blur_dimensions(H/V/both), repeat_edge_pixels | `cv2.GaussianBlur`，σ 與 blurriness 的換算寫入 docstring；大 σ 用降採樣金字塔 |
| ★Fast Box Blur | blur_radius, iterations, dimensions | 積分影像／可分離盒式；迭代 3 次近似高斯 |
| ★Directional Blur | direction, length | 線性核旋轉；或沿方向多採樣 |
| ★Radial Blur | amount, center, type(spin/zoom), quality | 極座標映射 + 沿角／徑向取樣 |
| Camera Lens Blur | radius, iris_shape(blades,curvature,rotation), blur_map, highlights | 圓盤／多邊形卷積核（FFT 卷積）；blur map 驅動分層混合 |
| Channel Blur | red/green/blue/alpha blurriness | 逐通道 |
| Compound Blur | blur_layer, max_blur, stretch_map | 以另一層亮度為模糊量 |
| Bilateral / Smart Blur | radius, threshold | `cv2.bilateralFilter` |
| Unsharp Mask / Sharpen | amount, radius, threshold | 標準 USM |
| Reduce Interlace Flicker | softness | 垂直小核模糊 |
預算：Gaussian σ=20 @1080p ≤ 40 ms。

### WS-11　Color Correction（`effects/color/`）
★Brightness & Contrast（含 legacy 公式）、★Levels（input black/white、gamma、output black/white，逐通道）、★Curves（主／RGB 控制點，樣條插值 → 256/65536 LUT）、★Hue/Saturation（master + 通道範圍 + colorize）、★Color Balance（shadow/midtone/highlight × RGB，preserve luminosity）、★Exposure（exposure、offset、gamma，線性光）、★Tint、★Tritone、Vibrance、Photo Filter、Selective Color、Change Color／Change to Color、Leave Color、Colorama（循環調色盤）、Gamma/Pedestal/Gain、Shadow/Highlight、Auto Levels/Contrast/Color、Equalize、Invert、Black & White、Channel Mixer、**Apply LUT**（`.cube` 1D/3D，三線性／四面體插值）、**Lumetri 類**（basic correction：temperature、tint、exposure、contrast、highlights、shadows、whites、blacks、saturation + LUT；以明確公式實作並註明非 Adobe 實作）。
**驗收**：Levels／Curves 恆等設定下位元相同；3D LUT 對恆等 LUT 誤差 < 0.5/255；Hue 旋轉 360° 回原值。

### WS-12　Distort（`effects/distort/`）
★Transform（效果版）、★Corner Pin（4 點透視，`cv2.getPerspectiveTransform`）、★Wave Warp（waveform: sine/square/triangle/sawtooth/noise…、direction、speed、phase）、★Displacement Map（map_layer, max_horizontal/vertical, channel 選擇, edge behavior）、★Turbulent Displace、★Bulge、★Twirl、★Spherize、★Ripple、★Polar Coordinates、★Mirror、Offset(Wrap)、Magnify、Optics Compensation、**Mesh Warp**（網格貝茲／雙線性）、**Puppet Pin**（P3，基於 MLS 或 TPS 變形）、Liquify（不做）、Reshape（P3）。
實作統一走「反向映射表 `map_x,map_y` + `cv2.remap`」，共用 `distort/_remap.py`（含 edge_behavior：transparent / clamp / mirror / wrap）。
**驗收**：Mirror 與 numpy flip 位元相同；Corner Pin 以單位矩形為輸入時為恆等；Twirl angle=0 恆等；Displacement 以中灰 map 恆等。

### WS-13　Generate（`effects/generate/`）
★Gradient Ramp（linear/radial、起訖點、色、ramp_scatter）、★4-Color Gradient、★Fill、★Grid、★Checkerboard、★Fractal Noise（fractal_type、noise_type、invert、contrast、brightness、scale、complexity、evolution、`evolution_options`：cycle、random_seed、**可循環**；以 Perlin/Simplex 多倍頻 + 時間維度為第三軸）、Cell Pattern、Ellipse、Radio Waves、Lens Flare（多元件鏡頭光斑）、Beam、Lightning、Audio Spectrum / Audio Waveform（需 WS-25）、Stroke / Vegas（沿遮罩路徑，依賴 WS-05）、Paint Bucket（P3）。
**驗收**：相同 seed 與 evolution 輸出位元相同；`evolution` 循環 N 後首尾幀 PSNR ≥ 50 dB；Gradient Ramp 端點色精確。

### WS-14　Keying & Matte（`effects/keying/`）
★Linear Color Key、★Color Range、★Luma Key、★Difference Matte、★**Color Key（Keylight 類）**（screen_colour、screen_gain、screen_balance、clip_black/white、screen_pre_blur、despill bias、**View 模式**：final/screen_matte/status/inside mask/outside mask）、★Spill Suppressor（綠／藍）、★Matte Choker／Simple Choker／Refine Soft Matte／Refine Hard Matte（shrink、grow、feather、decontaminate edge colors）、Key Cleaner、Inner/Outer Key、Extract。
實作：色度距離（YCbCr / Lab）、軟閾值、despill（`G = min(G, (R+B)/2)` 類並可調）、matte 邊緣淨化（`decontaminate`）。
**驗收**：合成綠幕測試（程式生成：已知前景 + 純綠 + 漸層半透明邊）→ 還原 α 的 MAE < 0.03；spill 抑制後綠偏量（R,B 平均 − G）趨近 0。

### WS-15　Stylize（`effects/stylize/`）
★Glow（threshold、radius、intensity、colors A/B、composite_original(on top/behind/none)、glow_operation）、★Mosaic、★Posterize、★Threshold、★Find Edges（Sobel/Scharr）、★Motion Tile（tile_center、tile_w/h、output_w/h、mirror_edges、phase、horizontal_phase_shift）、★Roughen Edges、Strobe Light、Cartoon、Brush Strokes、Scatter、CC Block Load（P3）、Emboss、Color Emboss。
**驗收**：Glow 對單點亮源擴散輪廓呈高斯分布（相關係數 ≥ 0.98）；Motion Tile 輸出尺寸與平鋪邊界正確；Posterize 級數 N 的唯一值數 ≤ N。

### WS-16　Noise & Grain（`effects/noise/`）
★Noise（amount、type：uniform/gaussian、color/mono、clipping）、★Add Grain／Match Grain（以參考樣本估計顆粒統計；P3）、Dust & Scratches、Median、Remove Grain（非局部均值 `cv2.fastNlMeansDenoisingColored` 或時間域去噪）、Noise Alpha/HLS。
**驗收**：Gaussian noise amount=σ 時輸出殘差 σ 誤差 < 3%；相同 seed 確定性。

### WS-17　Transition（`effects/transition/`）
★Linear Wipe（transition_completion、wipe_angle、feather）、★Radial Wipe（start_angle、wipe_center、clockwise/counter/both、feather）、★Gradient Wipe（gradient_layer、softness、invert）、★Iris Wipe（形狀、半徑、旋轉）、★Block Dissolve（block 大小、feather、確定性隨機）、★Card Wipe（需 WS-22 之前可用 2.5D 近似）、Venetian Blinds、CC Glass Wipe／Grid Wipe／Radial Scale Wipe（類型類似的自行命名實作，避免侵權名稱可加後綴 `Like`），並提供 **`Transition` 高階 API**：`ae.transitions.apply(clip_a, clip_b, kind="linear_wipe", duration=1.0, **params)`，回傳可直接串接的 clip（對應 MoviePy 的 `concatenate_videoclips` 流程）。
**驗收**：completion=0／100 時分別等於 A／B 完整輸出；Linear Wipe 邊界位置 = completion × 對角長度。

### WS-18　Time Effects（`effects/time/`）
★Echo（echo_time、number_of_echoes、starting_intensity、decay、echo_operator）、★Posterize Time（frame_rate）、★Time Difference、★Timewarp（speed／source_frame 兩種模式 + 插幀方法）、Pixel Motion Blur、CC Force Motion Blur、Frame Hold。
依賴 WS-06 的 `temporal_window` 與 WS-07。
**驗收**：Posterize Time=10fps 於 30fps comp 時每 3 幀輸出重複；Echo 衰減序列與公式一致。

### WS-19　Perspective／陰影類（`effects/perspective/`）
★Drop Shadow（shadow_color、opacity、direction、distance、softness、shadow_only）、★Radial Shadow、★Bevel Alpha、Bevel Edges、Layer Styles：Inner/Outer Glow、Stroke、Satin、Color/Gradient Overlay（Layer Styles 作為圖層屬性群，非效果）。
**驗收**：Drop Shadow 的 offset 向量 = distance·(sin,−cos)(direction)；softness=0 時陰影邊緣硬。

### WS-25　音訊圖層（P2）
**範圍**：`ae/audio/`
**交付**：音訊圖層（`audio_levels` 可關鍵影格，dB 單位）、波形資料提取、**Audio→Keyframes**（把音訊振幅轉成 `Property` 的 keyframes／可驅動表達式 `thisComp.layer("audio").audioLevels`）、音訊效果：Bass & Treble、Parametric EQ、Delay、Reverb（Schroeder 簡易實作）、Stereo Mixer、High/Low-Pass；與現有 `moviepy.audio.fx` 橋接。Comp 輸出以 `CompositeAudioClip` 混音。
**驗收**：dB↔線性互轉誤差 < 1e-6；Levels 關鍵影格斜坡下輸出 RMS 與解析值誤差 < 0.5 dB。

### WS-26　色彩管理與位元深度（P2）
**範圍**：`ae/color/`
**交付**：`project.bit_depth ∈ {8, 16, 32}`（內部皆 float32，差異在輸入量化與輸出）、`linear_light` 旗標、sRGB↔linear 轉換、Rec.709／Rec.2020／Display-P3 矩陣、ICC（Pillow `ImageCms`）、可選 OCIO（`ae-color` extra）、LUT 載入（`.cube`、`.3dl`）、**EXR** 讀寫（`ae-exr` extra）、16-bit PNG/TIFF 輸出、**HDR 輸出**（PQ／HLG 轉換 + ffmpeg 旗標；P3）。
**驗收**：sRGB→linear→sRGB 往返誤差 < 1e-6；線性光下 50% 灰 Screen/Multiply 與解析結果一致；32bpc 中超過 1.0 的亮度（如 Add 疊兩次）不被截斷直到最終輸出。

### WS-27　I/O、專案檔、Render Queue（P2）
**範圍**：`ae/io/`
**交付**：
- **專案 JSON**（`.mpyproj`）：序列化 Composition／圖層／屬性／效果／遮罩（schema 版本化、向前相容遷移函式）；素材以相對路徑 + 校驗和；`ae.load_project()` / `project.save()`。
- **Lottie 匯入**（bodymovin JSON → Composition，涵蓋 shape layers、transform、基本 trim paths、precomps；不支援的功能列警告清單，不 silently ignore）。
- **Render Queue**：`RenderQueue.add(comp, output_module, settings)`，設定含：resolution(full/half/third/quarter)、quality、time span（work area/custom）、frame blending、motion blur、codec 預設（H.264、ProRes 422/4444、PNG seq、WebM alpha、GIF）、多程序分段渲染（以 `multiprocessing` 切分時間區段再用 ffmpeg concat，維持確定性）、失敗重試、進度回報（`proglog` 相容）、`--dry-run`。
- **Proxy／Pre-render**：對耗時圖層預渲染成暫存影像序列並透明替換（可失效偵測：以內容雜湊）。
- CLI：`python -m moviepy.ae render project.mpyproj --comp Main --out out.mp4`。
**驗收**：save→load→render 與原始 comp 的輸出位元相同；多程序渲染輸出與單程序位元相同；Lottie 官方樣例（至少 5 個公開授權樣例，放入 tests 並標註授權）可匯入並出現不支援清單。

### WS-28　預覽與快取（P2）
**範圍**：`ae/preview/`
**交付**：**RAM Preview**（記憶體上限可設、LRU 逐幀快取，鍵 = comp 結構雜湊 + t + ctx）、**屬性變更的精準失效**（只失效受影響時間範圍）、`resolution_scale` 與 `draft` 預覽、`comp.preview(start,end)`（以 OpenCV 視窗，鍵盤：空白播放/暫停、方向鍵逐幀）、Notebook 內嵌（沿用 `display_in_notebook` 機制）、**ROI** 縮放預覽。
**驗收**：重播已快取區段時 CPU 渲染呼叫數為 0（以 mock 計數）；改一個圖層的 Transform 後只重算該圖層以上的快取。

---

## 7. Phase 3：進階

### WS-20　形狀圖層與向量（`ae/vector/`, `ae/layers/shape.py`）
**交付**：
- 向量引擎：路徑（貝茲）、布林運算（Merge Paths：add/subtract/intersect/exclude；以 `skia-pathops` 或自實作掃描線／clipper）、描邊（寬度、cap、join、miter、dashes、offset）、填色（純色／線性／放射狀漸層，含 keyframes）。
- Shape contents：Rectangle、Rounded Rect、Ellipse、Polystar（star/polygon）、Path；**Group**（有自己的 transform）；**Path Operators**：Trim Paths（start、end、offset、simultaneous/individually）、Repeater（copies、offset、transform、composite above/below）、Offset Paths、Pucker & Bloat、Round Corners、Twist、Wiggle Paths、Zig Zag、Merge Paths。
- 渲染器：優先 `skia-python`（`ae-vector`），退路 cv2 + 自行展平；一致性測試在兩個後端間比對（SSIM ≥ 0.98）。
- Continuously rasterize：縮放時重新向量化而非放大點陣。
**驗收**：Trim Paths 0–50% 的描邊長度 = 總長一半（誤差 < 1%）；Repeater 3 copies 的位置序列與解析一致；SVG path 匯入渲染與 `cairosvg`／skia 參考圖 SSIM ≥ 0.97。

### WS-21　文字圖層與文字動畫器（`ae/text/`）
**交付**：
- TextLayer：source text（可關鍵影格）、字型、大小、字距（tracking）、行距、對齊、段落（point／box text）、**逐字／逐詞／逐行** 分解（用 Pillow／`fonttools` 取得 glyph 輪廓與 advance）。
- **Animator**：Range Selector（start、end、offset、units、based_on(chars/chars excl. spaces/words/lines)、mode、amount、shape(square/ramp up/ramp down/triangle/round/smooth)、ease_high/low、randomize_order、seed）、Wiggly Selector、Expression Selector；可動畫屬性：position、scale、rotation、opacity、tracking、fill/stroke color、blur、anchor point、skew、line anchor。
- **Text on Path**：沿遮罩路徑（reversed、perpendicular、force alignment、first/last margin）。
- 文字進出場預設（Typewriter、Fade Up by word、Bounce）作為 **Animation Presets**（JSON，可套用到任意圖層）。
- 3D 逐字元（依賴 WS-22）。
**驗收**：Range Selector 0–50% 影響前一半字元（逐字計數）；Randomize 同 seed 同序；Typewriter 預設在 t 時顯示字元數 = ⌊rate·t⌋。

### WS-22　3D 圖層、攝影機、燈光（`ae/three_d/`）
**交付**：
- 圖層 `three_d=True`：position(xyz)、anchor_point(xyz)、scale(xyz)、orientation、x/y/z rotation；`material_options`（casts_shadows、accepts_shadows、accepts_lights、ambient/diffuse/specular/shininess/metal）。
- **CameraLayer**：one-node／two-node（point of interest）、zoom／focal_length／angle_of_view／film_size、`depth_of_field`（focus_distance、aperture、blur_level，採 Layer-by-layer 模糊近似）、cam 動畫。
- 投影：透視變換 3D→2D（每層一個四邊形，`cv2.warpPerspective`）；**深度排序**（painter's；交叉面以 z-buffer 逐像素選項，P3）。
- **LightLayer**：ambient/point/spot/parallel；intensity、color、cone angle/feather、falloff；陰影（shadow map，darkness、diffusion）。
- Layer-based 3D 與 **Ray-traced**（不做）。
**驗收**：單位長度物件在距離 d、焦距 f 下投影尺寸 = f·size/d（誤差 < 0.5 px）；兩層相交時排序正確；DOF 在 focus 平面模糊半徑 = 0。

### WS-23　追蹤與穩定（`ae/tracking/`）
**交付**：
- **Point Tracker**（Lucas-Kanade 金字塔 `cv2.calcOpticalFlowPyrLK`，含 search/feature region、confidence、前後向一致性檢驗、sub-pixel refine）→ 輸出 `Property[Vec2]` keyframes，可直接 `apply_to(layer.transform.position / null)`；Position／Rotation／Scale 多點模式。
- **Planar Tracker**（ORB/AKAZE + RANSAC homography，輸出 Corner Pin 四點）。
- **Warp Stabilizer**：位移／縮放／旋轉模式、平滑度、裁切／自動縮放、rolling shutter 的簡易補償（P3）。
- **Camera Solver**（P3，Structure-from-Motion，可選 `ae-track`；輸出 3D 攝影機與 null 點雲）。
- **Roto／分割外掛接口**：`ae.tracking.Segmenter` 協議（輸入 frame → 輸出 mask），可接 SAM／RVM 等外部模型，核心不內建模型。
**驗收**：合成平移＋旋轉序列的追蹤誤差 RMS < 0.3 px；Planar Tracker 對已知 homography 的角點誤差 < 1 px；穩定後殘餘抖動（高頻位移能量）降低 ≥ 80%。

### WS-24　粒子系統（`ae/particles/`）
**交付**：確定性 CPU 粒子引擎（發射器：point/line/box/disc/mesh/layer-based、速率、壽命、初速／方向／擴散、重力、風、湍流、碰撞平面、sprite／圓點／運動模糊拖尾、大小／顏色／不透明度隨壽命曲線）、**預熱（pre-roll）**、以 `Property` 驅動參數、輸出為 `Buffer`；可作為「圖層來源」放入 Composition。numba 加速為可選。
**驗收**：相同 seed 與相同 t 結果位元相同、且對任意 t 隨機存取與順序播放結果一致（透過解析式或快照式狀態重建）；粒子數 50k 的 1080p 單幀 ≤ 2 s（無 numba）。

### WS-29　文件、範例、品質保證（全程並行）
**交付**：
- Sphinx 文件 `docs/ae/`：Getting started、概念（Buffer／Property／渲染管線）、每個 WS 的 API 參考與範例；**對照表**「AE 功能 → MoviePy AE API」。
- 範例腳本 `examples/ae/`（至少 12 個：Lower third、Logo reveal、Kinetic typography、Green screen、Parallax 3D、Particle burst、Lottie 匯入、Time remap、Motion blur、Corner pin 追蹤貼圖、調整圖層調色、轉場）；`tests/test_doc_examples.py` 的機制須能涵蓋新範例。
- 基準測試 `benchmarks/`，CI 以閾值守住回歸（容許 ±20%）。
- `CHANGELOG.md` 每個 WS 合併時新增條目。

---

## 8. 依賴圖與里程碑

```
WS-00 ─┬─► WS-02 ─┐
WS-01 ─┤          ├─► WS-03 ─► WS-06 ─┬─► WS-10..19（全部並行）
       ├─► WS-04 ─┤                    ├─► WS-07 ─► WS-18
       └─► WS-05 ─┘                    ├─► WS-20 / WS-21 ─► WS-22 ─► WS-24
                                        ├─► WS-23
WS-26 / WS-25 / WS-27 / WS-28 ◄─ WS-03 之後即可啟動；WS-29 全程
```

| 里程碑 | 內容 | 完成標準 |
|---|---|---|
| **M0** 基礎 | WS-00、WS-01 | Buffer 與屬性系統測試全綠、golden 工具可用 |
| **M1** 合成核心 | WS-02、03、04、05、06 | 範例「Lower third」與「Logo reveal」可輸出；與舊 `CompositeVideoClip` 回歸等價 |
| **M2** 動態與時間 | WS-07、WS-28 | Motion blur、Time remap、RAM preview 可用 |
| **M3** 效果庫 v1 | WS-10～17 的 ★ 項目、WS-19 ★ | ≥ 60 個效果、metadata 完整、各自測試 |
| **M4** 向量與文字 | WS-20、21 | Lottie 匯入樣例可渲染；Kinetic typography 範例 |
| **M5** 色彩／I/O | WS-26、27、25 | 32bpc 線性光、專案檔往返、Render Queue、音訊圖層 |
| **M6** 3D／追蹤／粒子 | WS-22、23、24 | Parallax 3D、貼圖追蹤、粒子範例 |
| **M7** 效果庫 v2 | 其餘非 ★ 效果、WS-18 | 覆蓋 §2 表中全部 P2 項目 |

建議並行度：M0 內 2 個 agent；M1 內 4 個（WS-02/04/05 並行，WS-03 隨後，WS-06 最後）；M3 內可 6–10 個 agent 同時做不同效果類別。

---

## 9. Agent 作業規範（每個 WS 必讀）

### 9.1 分派建議
| 工作 | 建議 agent（`~/.claude/agents/`） |
|---|---|
| 每個 WS 開工前出計畫 | `planner`（輸出檔案級計畫；大型 WS 先經 `architect` 審核） |
| 實作 | `tdd-guide`（先寫測試）→ 一般實作 agent |
| 實作後審查 | `python-reviewer` + `code-reviewer`（強制） |
| 讀取使用者輸入／檔案／網路（Lottie、LUT、專案檔、表達式） | `security-reviewer`（強制） |
| 建置／型別／lint 失敗 | `build-error-resolver` |
| 效能熱點 | `performance-optimizer` |
| 文件 | `doc-updater` |

**平行規則**：同一 WS 內不同檔案才可並行；不同 agent 不得同時修改同一檔案。共享檔案（`ae/__init__.py`、`effects/registry.py`、`pyproject.toml`、`CHANGELOG.md`）只能以「追加一行」方式修改，並在合併前 rebase。

### 9.2 流程（強制）
0. **Research & Reuse**：先 `gh search code`／PyPI 查是否有可採用的成熟實作（例：`skia-python`、`pycairo`、`pyfastnoisesimd`／`noise`、`colour-science`、`lottie`、`opencv` 現成函式）；有授權相容（MIT/BSD/Apache）者優先採用或包裝，並在 PR 說明。**不得複製 GPL／專有碼。**
1. **Plan**：產出該 WS 的檔案清單、類別／函式簽名、測試清單、風險。
2. **TDD**：RED → GREEN → REFACTOR；覆蓋率 ≥ 80%（WS-01、WS-04、WS-05 ≥ 90%）。
3. **Review**：`python-reviewer`；CRITICAL／HIGH 必修。
4. **驗證**：`pytest tests/ae -q`、`pytest tests -q`（**既有測試必須全數通過**）、`black --check`、`flake8`。
5. **Commit**：Conventional Commits（`feat(ae): ...`）；一個 WS 可拆多個 commit，每個 commit 可獨立通過測試。
6. **文件與 CHANGELOG** 同步更新。

### 9.3 Definition of Done（每個 PR）
- [ ] 公開 API 有 numpydoc docstring 與至少一個可執行範例
- [ ] 所有參數為 `Property`（若適用）且有關鍵影格測試
- [ ] 通過 §6 效果共同 DoD（若為效果）
- [ ] 確定性（相同輸入位元相同）有測試
- [ ] 無硬編碼路徑／密鑰；輸入驗證完整（尺寸、範圍、型別），錯誤訊息明確
- [ ] 無 `print`／debug 殘留；無吞掉例外
- [ ] 新增依賴已走 §3.6 流程，缺依賴時有退路或明確錯誤
- [ ] 既有測試與新測試全綠；基準測試無 > 20% 回歸
- [ ] 檔案 ≤ 800 行、函式 < 50 行

### 9.4 對既有程式碼的允許修改範圍
僅允許以下「掛鉤」修改（其餘一律不改既有檔）：
1. `moviepy/__init__.py`：延遲匯出 `ae`（`import moviepy.ae` 時才載入，不增加 `import moviepy` 時間）。
2. `moviepy/video/VideoClip.py`：新增 `VideoClip.to_ae_layer()`（薄包裝）。
3. `pyproject.toml`：新增 extras 與 `packages` 包含 `moviepy.ae*`。
4. `tests/conftest.py`：新增 `--update-golden` 選項（不改既有 fixture 行為）。

### 9.5 測試規範
- 單元（解析解）、整合（多層 comp）、golden（影像）、屬性測試（可用 `hypothesis`：恆等、單調、範圍）、效能（benchmarks）。
- golden 圖必須由程式產生（不得引用受版權素材）；測試素材一律合成或自製，放 `tests/ae/assets/` 並附 `LICENSE-ASSETS.txt`。
- 隨機性一律透過 `ctx.rng_seed`；禁止使用全域 `np.random`。

---

## 10. 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| 純 CPU 效能不足（1080p 多層＋效果） | 渲染過慢 | ROI、快取、draft 降階、`numba` 可選、多程序分段、預留 GPU 後端接口（`render_gpu`，P3 以 `moderngl` 實作少數熱點） |
| 與 AE 視覺不完全一致（模糊半徑、羽化、混合邊界） | 使用者期待落差 | 每個效果 docstring 明寫「與 AE 的差異與換算」；提供校準參數；以公開文件公式為準 |
| 向量引擎依賴（skia）在 Windows／ARM 安裝困難 | 功能缺失 | 自實作退路 + 後端一致性測試 |
| 表達式沙盒被繞過 | 任意程式碼執行 | AST 白名單、禁 attribute 底線、資源上限（時間／節點數）、`security-reviewer` 強制審查、模糊測試 |
| Lottie／專案檔／LUT 等外部輸入造成崩潰 | 穩定性、安全 | 嚴格 schema 驗證、大小上限、解析錯誤給出行號、fuzz 測試 |
| 破壞既有 API | 回歸 | §9.4 限制修改範圍；CI 全量跑既有測試 |
| 範圍過大 | 延期 | 嚴守里程碑；每個 WS 先做 ★ 項目，非 ★ 延後 |
| 商標／授權 | 法律 | 不使用 Adobe 名稱作為套件／類別名（僅文件對照）；不複製專有資源；第三方依賴授權逐一登記於 `docs/ae/THIRD_PARTY.md` |

---

## 11. 附錄 A：AE 效果對照清單（完整追蹤表）

> 實作時在此表勾選；格式：`[ ] 效果名（WS-編號）`。★ = 首批。

**Blur & Sharpen**：[ ]★Gaussian Blur [ ]★Fast Box Blur [ ]★Directional Blur [ ]★Radial Blur [ ] Camera Lens Blur [ ] Channel Blur [ ] Compound Blur [ ] Bilateral Blur [ ] Smart Blur [ ] Unsharp Mask [ ] Sharpen [ ] Reduce Interlace Flicker　（WS-10）

**Color Correction**：[ ]★Brightness & Contrast [ ]★Levels [ ]★Curves [ ]★Hue/Saturation [ ]★Color Balance [ ]★Exposure [ ]★Tint [ ]★Tritone [ ] Vibrance [ ] Photo Filter [ ] Selective Color [ ] Change Color [ ] Change to Color [ ] Leave Color [ ] Colorama [ ] Gamma/Pedestal/Gain [ ] Shadow/Highlight [ ] Auto Levels [ ] Auto Contrast [ ] Auto Color [ ] Equalize [ ] Invert [ ] Black & White [ ] Channel Mixer [ ] Apply LUT [ ] Basic Correction（Lumetri 類）　（WS-11）

**Distort**：[ ]★Transform [ ]★Corner Pin [ ]★Wave Warp [ ]★Displacement Map [ ]★Turbulent Displace [ ]★Bulge [ ]★Twirl [ ]★Spherize [ ]★Ripple [ ]★Polar Coordinates [ ]★Mirror [ ] Offset [ ] Magnify [ ] Optics Compensation [ ] Mesh Warp [ ] Puppet Pin [ ] Reshape　（WS-12）

**Generate**：[ ]★Gradient Ramp [ ]★4-Color Gradient [ ]★Fill [ ]★Grid [ ]★Checkerboard [ ]★Fractal Noise [ ] Cell Pattern [ ] Ellipse [ ] Radio Waves [ ] Lens Flare [ ] Beam [ ] Lightning [ ] Audio Spectrum [ ] Audio Waveform [ ] Stroke [ ] Vegas [ ] Paint Bucket　（WS-13）

**Keying / Matte**：[ ]★Linear Color Key [ ]★Color Range [ ]★Luma Key [ ]★Difference Matte [ ]★Color Key（Keylight 類）[ ]★Spill Suppressor [ ]★Matte Choker [ ]★Simple Choker [ ]★Refine Soft Matte [ ] Refine Hard Matte [ ] Key Cleaner [ ] Inner/Outer Key [ ] Extract　（WS-14）

**Stylize**：[ ]★Glow [ ]★Mosaic [ ]★Posterize [ ]★Threshold [ ]★Find Edges [ ]★Motion Tile [ ]★Roughen Edges [ ] Strobe Light [ ] Cartoon [ ] Brush Strokes [ ] Scatter [ ] Emboss [ ] Color Emboss　（WS-15）

**Noise & Grain**：[ ]★Noise [ ] Add Grain [ ] Match Grain [ ] Dust & Scratches [ ] Median [ ] Remove Grain　（WS-16）

**Transition**：[ ]★Linear Wipe [ ]★Radial Wipe [ ]★Gradient Wipe [ ]★Iris Wipe [ ]★Block Dissolve [ ]★Card Wipe [ ] Venetian Blinds [ ] Grid Wipe 類 [ ] Glass Wipe 類 [ ] Radial Scale Wipe 類　（WS-17）

**Time**：[ ]★Echo [ ]★Posterize Time [ ]★Time Difference [ ]★Timewarp [ ] Pixel Motion Blur [ ] Force Motion Blur [ ] Frame Hold　（WS-18）

**Perspective**：[ ]★Drop Shadow [ ]★Radial Shadow [ ]★Bevel Alpha [ ] Bevel Edges　**Layer Styles**：[ ] Drop Shadow [ ] Inner Shadow [ ] Outer Glow [ ] Inner Glow [ ] Bevel & Emboss [ ] Satin [ ] Color Overlay [ ] Gradient Overlay [ ] Stroke　（WS-19）

**核心功能**：[ ] Keyframes/Easing（01） [ ] Expressions（01） [ ] Transform+Parenting（02） [ ] Pre-compose（03） [ ] 38 Blend modes（04） [ ] Masks（05） [ ] Track Mattes（05） [ ] Effect stack/Adjustment（06） [ ] Time Remap（07） [ ] Motion Blur（07） [ ] Shape layers（20） [ ] Text animators（21） [ ] 3D/Camera/Lights（22） [ ] Tracking/Stabilize（23） [ ] Particles（24） [ ] Audio layers（25） [ ] Color mgmt（26） [ ] Project/Lottie/Render Queue（27） [ ] RAM Preview（28）

---

## 12. 附錄 B：第一個 Sprint 建議（可直接派工）

| Agent 任務 | 內容 | 預估 |
|---|---|---|
| A1 | WS-00：Buffer、RenderContext、golden 工具、benchmark 腳本 | 1 個 sprint |
| A2 | WS-01-a：Property／Keyframe／Easing／Spatial path／JSON | 1 個 sprint |
| A3 | WS-01-b：Expression 引擎（沙盒＋函式庫＋安全測試） | 1 個 sprint |
| A4 | WS-04（先行開發混合公式與測試向量；對 `Buffer` 只依賴 A1 的介面草案） | 1 個 sprint |
| 之後 | WS-02、WS-05（A1/A2 合併後）→ WS-03 → WS-06 | 依 §8 |

**第一步請 agent 做的事**：依 §9.2 流程，對 WS-00 產出計畫（`planner`）→ 經 `architect` 對 §3 提出修正意見（若有）→ 以 TDD 實作 → `python-reviewer` 審查 → 合併後解鎖 M0 其他任務。
