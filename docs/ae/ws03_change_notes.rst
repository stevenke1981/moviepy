:orphan:

WS-03 變更註記
==============

新增 ``moviepy.ae.composition``（``Composition``——``VideoClip`` 子類、
圖層堆疊管理、markers、work area、透明輸出 mask；``Marker``；
``from_moviepy``）、``moviepy.ae.renderer``（``Renderer``：由下而上的
ROI 渲染、solo／enabled／guide 規則、resolution scale、WS-04 混合）
與 ``moviepy.ae.layers.CompLayer``（Pre-compose，子合成 fps 格點取樣、
倒放、collapse transformations 不裁切）。

``moviepy.ae.__init__`` 新增延遲載入的聚合 facade（``import moviepy.ae``
不會載入合成或混合模組）；``tests/ae/test_compatibility.py`` 的 ``__all__``
斷言改為「包含原本四個符號」並檢查延遲載入。依 §9.4 新增
``VideoClip.to_ae_layer()`` 薄掛鉤。未新增依賴。
