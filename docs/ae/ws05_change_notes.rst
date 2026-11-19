:orphan:

WS-05 變更註記
==============

新增 ``moviepy.ae.masks``：``path``（rect／ellipse／rounded_rect／polygon／
star／``from_svg_path``／``flatten``，輸出 ``PathValue``）、``rasterize``
（精確掃描線超取樣、高斯羽化 ``sigma = F/2``、距離轉換擴張）、``mask``
（``MaskMode``、``Mask``、``MaskStack``、``combine``、``to_clip``）與
``matte``（``MatteMode``、``matte_values``、``TrackMatte``、
``srgb_to_linear``）。

``Layer`` 新增 ``masks``、``track_matte``、``set_track_matte()`` 與
``apply_masks()``；``Layer.render`` 與 ``Renderer.render_layer`` 在變換前套用
遮罩，``Renderer.apply_track_matte`` 在混合前套用軌道遮罩（含巢狀）。
``moviepy.ae`` 延遲匯出 ``Mask``、``MaskStack``、``MaskMode``、
``TrackMatte``、``MatteMode``。未新增依賴。
