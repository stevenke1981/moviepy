:orphan:

WS-02 變更註記
==============

新增 ``moviepy.ae.transform``（AE 順序的變形矩陣與動畫欄位）、
``moviepy.ae.warp``（濾波支援半徑、destination bounds 與
``cv2.warpAffine`` 重採樣）、``moviepy.ae._geometry``（共用驗證原語）
與 ``moviepy.ae.layers``（``Layer``／``AVLayer``／``NullLayer``／
``SolidLayer``）。既有 MoviePy 執行期、測試與依賴宣告完全未變更。

座標使用像素中心慣例，預設 ``Transform()`` 對任何來源尺寸都是
identity。整數平移且 opacity 100 時共用來源儲存體，不複製像素；
其他情況寫入 WS-00 的儲存池並以 ``Buffer._publish`` 發布保守 metadata，
避免每格重新掃描全張畫面的 min／max。

推送前的獨立審查修復了非零 Buffer offset 的取樣、零縮放錯誤填滿 ROI、
負 stretch 影片／mask／動畫倒放、opacity 套用前的 cubic 非有限值檢查，
以及曲線停止後的終點朝向。驗證失敗的屬性賦值會保留原物件狀態；
矩陣與所有平移／裁切路徑都有一致的形狀、bounds 及像素面積檢查。
大座標的半像素支援不再被相對容差裁掉；不可表示的反矩陣明確拒絕。

圖層提供 AE 時間模型（``in_point``／``out_point``／``start_time``／
``stretch``，負 stretch 為反向）、開關（``enabled``／``solo`` 影響渲染，
``shy``／``locked``／``guide`` 僅中繼資料）、Parenting（世界矩陣由根往下
累乘，opacity 不繼承，循環在賦值時拒絕）與 Auto-Orient。

3D 欄位、``collapse_transformations``、``continuously_rasterize`` 與
``motion_blur`` 只驗證、儲存與序列化，行為分別延後到 WS-22／WS-03／
WS-20／WS-07；``three_d=True`` 會拋出 ``NotImplementedError`` 而不是
靜默忽略。``blend_mode`` 仍是字串 token，38 種模式屬於 WS-04。
遮罩（WS-05）、特效（WS-06）、Time Remap（WS-07）與
``Composition``／``Renderer``（WS-03）不在本階段。

未新增依賴：重用既有 NumPy、OpenCV 與 Python 標準函式庫。
``moviepy/ae/__init__.py`` 維持 WS-00 內容不動，因為既有相容性測試
固定了 ``moviepy.ae.__all__``；公開匯入路徑為 ``moviepy.ae.transform``、
``moviepy.ae.warp`` 與 ``moviepy.ae.layers``，聚合 facade 延到 WS-03。
``VideoClip.to_ae_layer()`` 掛鉤同樣延到 WS-03。

公開功能與實際差異見 ``ws02.rst``，驗收證據見 ``ws02_acceptance.md``。

依原專案既有檔案修改白名單，本階段以新增文件記錄變更。
根 CHANGELOG、README 及 Sphinx 導覽整合維持延後；本頁使用 ``:orphan:``。
