:orphan:

WS-00 變更註記
==============

新增 ``moviepy.ae`` 子套件，提供預乘 float32 ``Buffer``、RGB/clip 匯入與
一次最終 uint8 匯出、全域矩形操作及 Normal source-over。RGB 在合成中
保留 signed/HDR 值；數值編碼由色彩空間標籤描述，尚未執行色彩轉換。

新增 ``RenderContext`` 驗證設定、來源時間與 frame index；每圖層／每幀
以 fresh PCG64 產生可重算隨機串流。cache 與 shutter 為後續工作保留。

新增解析／相容性測試、合成 golden PNG 與 provenance sidecar，以及
1080p 單執行緒 CPU 基準。Gaussian σ=20 僅量測既有 cv2 原語；本次尚未
提供 WS-10 模糊效果。首次效能結果建立 baseline，不表示已驗證歷史回歸率。

CPU 最佳化重用既有 OpenCV 公開運算 API；內部原始配置回收池最多兩筆、
合計最多 64 MiB，須等原陣列和全部 view 都釋放才可重用。它不保存或
跳過渲染結果；Buffer 的公開不可變與唯讀契約保持一致。第三方授權及
既有相依套件的使用範圍記錄於 ``docs/ae/THIRD_PARTY.md``。

保留的初始 Gaussian σ=20 結果使用三通道 RGB；最終基準使用四通道 RGBA，
兩者工作量不同，因此不作效能回歸比較。Normal 仍使用相同十次全畫幅
半透明 over 和一次最終 RGB 輸出的驗收邊界。

依 ``docs/AE_PARITY_SPEC.md`` §9.4 的既有檔案修改白名單，本次新增此頁
及 ``benchmarks/README.md`` 保存變更和基準說明。根 ``CHANGELOG.md``、
根 ``README.md`` 與 Sphinx 導覽整合明確延後；這兩份 RST 使用
``:orphan:``，可以獨立建置而不修改既有 Sphinx index。
