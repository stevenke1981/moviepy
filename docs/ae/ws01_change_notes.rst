:orphan:

WS-01 變更註記
==============

新增 ``moviepy.ae.properties`` 子套件，提供 typed Property、完整時間
關鍵影格與 Ease、弧長空間路徑、roving、獨立維度、Path 值與 schema-1
JSON。``set_keyframes`` 修改目前物件；``with_*`` 取得新物件。

新增受資源限制的 Python 表達式 AST 解譯器、不可變 metadata 快照、
base 採樣／循環、向量／色彩／時間函式及確定性隨機／wiggle。
沙盒不公開任意 Python builtin、物件方法、檔案或網路能力。

未新增依賴，重用既有 NumPy 與 Python 標準函式庫。
音訊介面僅接收已完成分析的樣本；沒有 WS-25 音訊分析器。
公開功能與實際差異見 ``ws01.rst``，驗收證據見 ``ws01_acceptance.md``。

依原專案既有檔案修改白名單，本階段以新增文件記錄變更。
根 CHANGELOG、README 及 Sphinx 導覽整合維持延後；本頁使用 ``:orphan:``。
