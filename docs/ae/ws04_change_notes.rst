:orphan:

WS-04 變更註記
==============

新增 ``moviepy.ae.blend``：``BlendMode``（38 種 AE 模式的字串列舉，含
AE 選單名稱 ``label`` 與分類 ``category``）、``blend()``（premultiplied
float32、W3C source-over 帶混合、opacity、Preserve Underlying Transparency、
確定性 Dissolve／Dancing Dissolve）、``_formulas``（逐像素 ``B(Cb, Cs)``，
公式出處寫在 docstring）與 ``alpha``（alpha 合成規則）。

``Layer.blend_mode`` 從自由字串 token 改為驗證 ``BlendMode`` 並儲存標準
token（相容既有 ``"normal"``／``"screen"`` 用法）；新增
``Layer.preserve_transparency`` 旗標。未新增依賴。
