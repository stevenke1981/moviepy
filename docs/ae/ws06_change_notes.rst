:orphan:

WS-06 變更註記
==============

新增 ``moviepy.ae.effects``：``base``（``Param``、``AEEffect``：參數皆為
``Property``、``bounds_expand``、``temporal_window``／``render_temporal``、
效果遮罩、``blend_with_original``、MoviePy ``Effect`` 介面）、``stack``
（``EffectStack``：add／remove／move／find／active，時間鄰域來源）、
``registry``（``register``／``get``／``list``／``categories``／``metadata``／
``to_json``）、``adjustment``（``apply_adjustment``）、``bridge``
（``from_moviepy_effect``）與內建參考效果 Gaussian Blur、Brightness &
Contrast、Tint、Invert、Fill、Echo。新增 ``moviepy.ae.fx`` 與
``moviepy.ae.layers.AdjustmentLayer``。

``Layer`` 新增 ``effects``、``is_adjustment``、``apply_effects()``、
``prepared_source()``；``Renderer`` 在遮罩後、變換前套用效果，並對調整圖層
套用下方累積結果（含時間鄰域）。``Composition.add_adjustment()``。
``moviepy.ae`` 延遲匯出 ``AEEffect``、``EffectStack``、``AdjustmentLayer``、
``from_moviepy_effect``、``fx``、``effects``。新增文件
``docs/ae/writing_effects.rst``。未新增依賴。
