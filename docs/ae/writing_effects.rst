:orphan:

撰寫新效果（模板與 Checklist）
==============================

所有 AE 效果都繼承 ``moviepy.ae.effects.AEEffect``，放在
``moviepy/ae/effects/<category>/<snake_name>.py``，並以 ``@register`` 註冊；
內建效果還需把模組路徑加進 ``moviepy/ae/effects/registry.py`` 的
``_BUILTINS``（只能追加一行）。

--------------

模板
--------------

.. code-block:: python

    import numpy as np

    from moviepy.ae import Buffer
    from moviepy.ae.effects import AEEffect, Param, registry


    @registry.register
    class ChannelGain(AEEffect):
        """Multiply straight color by a per-channel gain.

        ==========  =============  ========================
        Parameter   AE panel name  Notes
        ==========  =============  ========================
        gain        Gain           0..4, default 1
        channel     Channel        ``rgb`` / ``red`` ...
        ==========  =============  ========================

        Notes
        -----
        Differences from AE and calibration notes go here.
        """

        name = "Channel Gain"            # registry key / AE menu name
        category = "Color Correction"    # AE category
        PARAMS = (
            Param("gain", "float", 1.0, (0.0, 4.0)),
            Param("channel", "enum", "rgb", choices=("rgb", "red")),
        )

        def render(self, src, t, context=None, values=None):
            values = self.values_at(t, context) if values is None else values
            if values["gain"] == 1.0 or 0 in src.size:
                return src                                   # identity at default
            rgba = src.rgba.copy()
            channels = slice(0, 3) if values["channel"] == "rgb" else slice(0, 1)
            rgba[..., channels] *= values["gain"]
            np.minimum(rgba[..., :3], rgba[..., 3:], out=rgba[..., :3])
            return Buffer._publish(rgba, src.offset, src.color_space)


    gray = Buffer(np.full((1, 1, 4), 0.25, dtype=np.float32) + [0, 0, 0, 0.75])
    out = ChannelGain(gain=2).process(gray, 0.0)
    assert np.allclose(out.rgba[0, 0], [0.5, 0.5, 0.5, 1.0])
    assert registry.get("channel gain") is ChannelGain
    registry.unregister("Channel Gain")       # 範例結束時移除

需要外擴邊界時覆寫 ``bounds_expand(size, t, context, values)``，回傳
``(left, top, right, bottom)`` 像素；需要其他時刻影像時覆寫
``temporal_window`` 與 ``render_temporal``（見 :doc:`ws06`）。

--------------

新增效果 Checklist
------------------

* [ ] 繼承 ``AEEffect``，``name``／``category`` 對齊 AE 選單，``PARAMS`` 完整
  （型別、範圍、預設值、enum 選項）。
* [ ] 數值參數皆為 ``Property``（基底類別自動處理），有關鍵影格測試。
* [ ] premultiplied 正確：色彩運算前反預乘、運算後預乘，輸出 ``rgb <= alpha``。
* [ ] 至少一個解析或 golden 測試，以及「預設／參數為 0 時為恆等」測試。
* [ ] docstring 含參數表（AE 面板名稱）與「與 AE 的差異」說明。
* [ ] 隨機性只用 ``context.rng_for(layer_id)``；相同輸入位元相同（確定性測試）。
* [ ] draft 品質可降階（若適用），1080p 單幀時間記錄於文件。
* [ ] 檔案 ≤ 800 行、函式 < 50 行、無新依賴（或依 §3.6 流程申請）。
* [ ] ``tests/ae/test_effects.py`` 的 metadata 自動掃描測試通過。
