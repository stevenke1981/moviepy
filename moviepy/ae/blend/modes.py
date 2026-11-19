"""The AE blend-mode catalogue and the ``blend`` entry point.

Examples
--------
>>> import numpy as np
>>> from moviepy.ae import Buffer
>>> from moviepy.ae.blend import BlendMode, blend
>>> gray = Buffer(np.full((1, 1, 4), 0.5, dtype=np.float32) * [1, 1, 1, 2])
>>> blend(gray, gray, BlendMode.MULTIPLY).rgba[0, 0].tolist()
[0.25, 0.25, 0.25, 1.0]
>>> blend(gray, gray, "screen").rgba[0, 0].tolist()
[0.75, 0.75, 0.75, 1.0]
"""

from enum import Enum

import numpy as np

from moviepy.ae.blend import _formulas as formulas, alpha as rules
from moviepy.ae.blend._uniform import TILE_PIXELS, composite as uniform_composite
from moviepy.ae.buffer import Buffer
from moviepy.ae.context import RenderContext
from moviepy.ae.properties.values import finite_real


_TOKEN_LIMIT = 64


class BlendMode(str, Enum):
    """Every After Effects layer blend mode, as a string-valued enum.

    Members compare equal to their snake_case token (``BlendMode.SCREEN ==
    "screen"``). ``label`` gives the AE menu name and ``category`` the AE menu
    group. Use ``BlendMode.coerce`` to accept tokens, labels or members.
    """

    NORMAL = "normal"
    DISSOLVE = "dissolve"
    DANCING_DISSOLVE = "dancing_dissolve"
    DARKEN = "darken"
    MULTIPLY = "multiply"
    COLOR_BURN = "color_burn"
    CLASSIC_COLOR_BURN = "classic_color_burn"
    LINEAR_BURN = "linear_burn"
    DARKER_COLOR = "darker_color"
    ADD = "add"
    LIGHTEN = "lighten"
    SCREEN = "screen"
    COLOR_DODGE = "color_dodge"
    CLASSIC_COLOR_DODGE = "classic_color_dodge"
    LINEAR_DODGE = "linear_dodge"
    LIGHTER_COLOR = "lighter_color"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    VIVID_LIGHT = "vivid_light"
    LINEAR_LIGHT = "linear_light"
    PIN_LIGHT = "pin_light"
    HARD_MIX = "hard_mix"
    DIFFERENCE = "difference"
    CLASSIC_DIFFERENCE = "classic_difference"
    EXCLUSION = "exclusion"
    SUBTRACT = "subtract"
    DIVIDE = "divide"
    HUE = "hue"
    SATURATION = "saturation"
    COLOR = "color"
    LUMINOSITY = "luminosity"
    STENCIL_ALPHA = "stencil_alpha"
    STENCIL_LUMA = "stencil_luma"
    SILHOUETTE_ALPHA = "silhouette_alpha"
    SILHOUETTE_LUMA = "silhouette_luma"
    ALPHA_ADD = "alpha_add"
    LUMINESCENT_PREMUL = "luminescent_premul"

    def __str__(self):
        return self.value

    @property
    def label(self):
        """Return the AE menu label, for example ``"Classic Color Burn"``."""
        return _SPECS[self][2]

    @property
    def category(self):
        """Return the AE menu group such as ``"darken"`` or ``"matte"``."""
        return _SPECS[self][3]

    @classmethod
    def coerce(cls, value):
        """Return the member named by a member, token or AE label.

        Matching ignores case and treats spaces, hyphens and slashes like
        underscores, so ``"Color Burn"``, ``"color-burn"`` and
        ``"COLOR_BURN"`` all name ``BlendMode.COLOR_BURN``.

        Raises
        ------
        TypeError
            If ``value`` is not a string or ``BlendMode``.
        ValueError
            If the token names no blend mode.
        """
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("blend_mode must be a BlendMode or string token")
        token = value.strip().lower()
        if not token or len(token) > _TOKEN_LIMIT:
            raise ValueError("blend_mode must be a nonempty token")
        for separator in (" ", "-", "/"):
            token = token.replace(separator, "_")
        try:
            return cls(token)
        except ValueError:
            raise ValueError(f"unknown blend_mode {value!r}") from None


F = formulas
# member -> (formula or None, kind, AE label, AE menu category)
_SPECS = {
    BlendMode.NORMAL: (F.normal, "normal", "Normal", "normal"),
    BlendMode.DISSOLVE: (None, "dissolve", "Dissolve", "normal"),
    BlendMode.DANCING_DISSOLVE: (None, "dissolve", "Dancing Dissolve", "normal"),
    BlendMode.DARKEN: (F.darken, "separable", "Darken", "darken"),
    BlendMode.MULTIPLY: (F.multiply, "separable", "Multiply", "darken"),
    BlendMode.COLOR_BURN: (F.color_burn, "separable", "Color Burn", "darken"),
    BlendMode.CLASSIC_COLOR_BURN: (
        F.classic_color_burn,
        "classic",
        "Classic Color Burn",
        "darken",
    ),
    BlendMode.LINEAR_BURN: (F.linear_burn, "separable", "Linear Burn", "darken"),
    BlendMode.DARKER_COLOR: (F.darker_color, "separable", "Darker Color", "darken"),
    BlendMode.ADD: (F.add, "unclamped", "Add", "lighten"),
    BlendMode.LIGHTEN: (F.lighten, "separable", "Lighten", "lighten"),
    BlendMode.SCREEN: (F.screen, "separable", "Screen", "lighten"),
    BlendMode.COLOR_DODGE: (F.color_dodge, "separable", "Color Dodge", "lighten"),
    BlendMode.CLASSIC_COLOR_DODGE: (
        F.classic_color_dodge,
        "classic",
        "Classic Color Dodge",
        "lighten",
    ),
    BlendMode.LINEAR_DODGE: (F.linear_dodge, "separable", "Linear Dodge", "lighten"),
    BlendMode.LIGHTER_COLOR: (
        F.lighter_color,
        "separable",
        "Lighter Color",
        "lighten",
    ),
    BlendMode.OVERLAY: (F.overlay, "separable", "Overlay", "contrast"),
    BlendMode.SOFT_LIGHT: (F.soft_light, "separable", "Soft Light", "contrast"),
    BlendMode.HARD_LIGHT: (F.hard_light, "separable", "Hard Light", "contrast"),
    BlendMode.VIVID_LIGHT: (F.vivid_light, "separable", "Vivid Light", "contrast"),
    BlendMode.LINEAR_LIGHT: (F.linear_light, "separable", "Linear Light", "contrast"),
    BlendMode.PIN_LIGHT: (F.pin_light, "separable", "Pin Light", "contrast"),
    BlendMode.HARD_MIX: (F.hard_mix, "separable", "Hard Mix", "contrast"),
    BlendMode.DIFFERENCE: (F.difference, "separable", "Difference", "difference"),
    BlendMode.CLASSIC_DIFFERENCE: (
        F.difference,
        "classic",
        "Classic Difference",
        "difference",
    ),
    BlendMode.EXCLUSION: (F.exclusion, "separable", "Exclusion", "difference"),
    BlendMode.SUBTRACT: (F.subtract, "separable", "Subtract", "difference"),
    BlendMode.DIVIDE: (F.divide, "separable", "Divide", "difference"),
    BlendMode.HUE: (F.hue, "separable", "Hue", "hsl"),
    BlendMode.SATURATION: (F.saturation, "separable", "Saturation", "hsl"),
    BlendMode.COLOR: (F.color, "separable", "Color", "hsl"),
    BlendMode.LUMINOSITY: (F.luminosity, "separable", "Luminosity", "hsl"),
    BlendMode.STENCIL_ALPHA: (None, "stencil", "Stencil Alpha", "matte"),
    BlendMode.STENCIL_LUMA: (None, "stencil", "Stencil Luma", "matte"),
    BlendMode.SILHOUETTE_ALPHA: (None, "silhouette", "Silhouette Alpha", "matte"),
    BlendMode.SILHOUETTE_LUMA: (None, "silhouette", "Silhouette Luma", "matte"),
    BlendMode.ALPHA_ADD: (None, "alpha_add", "Alpha Add", "utility"),
    BlendMode.LUMINESCENT_PREMUL: (None, "normal", "Luminescent Premul", "utility"),
}
del F


def blend(
    base,
    layer,
    mode="normal",
    opacity=1.0,
    *,
    preserve_underlying_transparency=False,
    context=None,
    layer_id="layer",
):
    """Blend ``layer`` onto ``base`` with an AE blend mode.

    Parameters
    ----------
    base : Buffer
        Backdrop: the accumulated result of every layer below.
    layer : Buffer
        Premultiplied source layer, already transformed into world space.
    mode : BlendMode or str, optional
        Blend mode member, token (``"color_burn"``) or AE label.
    opacity : float, optional
        Extra 0..1 factor applied to the layer, clamped like layer opacity.
        Opacity 0 returns ``base`` unchanged, including Stencil modes;
        opacity 1 is identical to omitting it.
    preserve_underlying_transparency : bool, optional
        AE's "Preserve Underlying Transparency" switch: the layer is drawn
        only where the backdrop is opaque and the result keeps backdrop alpha.
    context : RenderContext, optional
        Supplies ``rng_seed`` and the frame index for the deterministic noise
        used by Dissolve (fixed pattern) and Dancing Dissolve (per frame).
    layer_id : int or str, optional
        Stable identity mixed into the Dissolve noise seed.

    Returns
    -------
    Buffer
        New premultiplied pixels covering the union of both bounds (the base
        bounds for Stencil/Silhouette and preserved transparency).

    Notes
    -----
    Colors are unpremultiplied before mixing and premultiplied after, using the
    W3C source-over rule described in ``moviepy.ae.blend.alpha``. Except for
    Normal, Add, Alpha Add and Luminescent Premul, straight colors are clamped
    to [0, 1] before mixing, matching 8/16 bpc AE projects. Luminescent Premul
    equals Normal here because this float pipeline never clips premultiplied
    color that exceeds alpha (the reason the mode exists in AE). Classic modes
    mix premultiplied colors directly; they equal the modern modes for opaque
    pixels apart from the Color Burn/Dodge pure-white/black backdrop rules.

    Stencil opacity zero disables the matte under the user-selected project
    policy. Positive opacity still multiplies matte coverage, so reaching zero
    can produce a discontinuity. This policy has not been verified in Adobe AE.
    """
    member = BlendMode.coerce(mode)
    _check_buffers(base, layer)
    factor = min(1.0, max(0.0, finite_real(opacity, "opacity")))
    preserve = _flag(preserve_underlying_transparency)
    formula, kind = _SPECS[member][:2]
    if factor == 0.0 or kind == "silhouette" and 0 in layer.size:
        return base
    if kind in ("stencil", "silhouette"):
        return _matte(base, layer, factor, member, kind)
    if 0 in layer.size:
        return base
    source = _scaled(layer, factor)
    if kind == "normal" and not preserve:
        return source.composite_over(base)
    if 0 in base.size:
        if preserve:
            return base
        base = _transparent(source.bounds, base.color_space)
    if kind == "dissolve":
        noise = lambda clipped: _noise(clipped, member, context, layer_id)
        return _region(base, source, preserve, rules.dissolve, noise)
    if kind == "alpha_add" and not preserve:
        return _region(base, source, False, _alpha_add)
    clamp = kind not in ("unclamped", "normal", "alpha_add")
    formula = formula or formulas.normal
    uniform = uniform_composite(
        base, source, formula, clamp=clamp, classic=kind == "classic", preserve=preserve
    )
    if uniform is not None:
        return uniform
    if _opaque_overlap(base, source):
        return _opaque_region(base, source, formula, clamp)
    composite = _separable(formula, clamp, kind == "classic")
    return _region(base, source, preserve, composite)


def _check_buffers(base, layer):
    """Require two Buffers with the same color-space label."""
    if not isinstance(base, Buffer) or not isinstance(layer, Buffer):
        raise TypeError("base and layer must be Buffers")
    if base.color_space != layer.color_space:
        raise ValueError("color_space must match for blending")


def _flag(value):
    """Validate a strict boolean switch."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("preserve_underlying_transparency must be bool")
    return bool(value)


def _scaled(layer, factor):
    """Apply an opacity factor to every premultiplied channel."""
    if factor == 1.0:
        return layer
    return Buffer._publish(
        layer.rgba * np.float32(factor), layer.offset, layer.color_space
    )


def _transparent(bounds, color_space):
    """Return a fully transparent Buffer covering ``bounds``."""
    left, top, right, bottom = bounds
    array = np.zeros((bottom - top, right - left, 4), dtype=np.float32)
    return Buffer._publish(array, (left, top), color_space)


def _union(first, second):
    """Return the union of two half-open world rectangles."""
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _separable(formula, clamp, classic):
    """Bind a color formula into a region compositor."""

    def composite(cb, alpha_b, cs, alpha_s, preserve):
        return rules.separable(
            cb,
            alpha_b,
            cs,
            alpha_s,
            formula,
            preserve=preserve,
            clamp=clamp,
            classic=classic,
        )

    return composite


def _alpha_add(cb, alpha_b, cs, alpha_s, preserve):
    """Adapt ``rules.alpha_add`` to the region compositor signature."""
    return rules.alpha_add(cb, alpha_b, cs, alpha_s)


def _region(base, source, preserve, composite, noise=None):
    """Expand the backdrop, composite inside the source rectangle and publish."""
    bounds = base.bounds if preserve else _union(base.bounds, source.bounds)
    canvas = np.array(base.expand_to(bounds).rgba)
    clipped = source.crop(bounds)
    if 0 in clipped.size:
        return Buffer._publish(canvas, bounds[:2], base.color_space)
    left, top = clipped.offset[0] - bounds[0], clipped.offset[1] - bounds[1]
    width, height = clipped.size
    region = canvas[top : top + height, left : left + width]
    noise_values = None if noise is None else noise(clipped)
    rows = max(1, TILE_PIXELS // width)
    for top in range(0, height, rows):
        target = region[top : top + rows]
        arguments = (*rules.split(target), *rules.split(clipped.rgba[top : top + rows]))
        with np.errstate(all="ignore"):
            if noise is not None:
                rgb, alpha = composite(
                    *arguments, noise_values[top : top + rows], preserve=preserve
                )
            else:
                rgb, alpha = composite(*arguments, preserve)
        np.clip(alpha, 0, 1, out=alpha)
        _merge_into(target, rgb, alpha)
    return _publish(canvas, bounds[:2], base.color_space)


def _opaque_overlap(base, source):
    """Return whether every source pixel lands on an opaque backdrop pixel."""
    inside = (
        source.bounds[0] >= base.bounds[0]
        and source.bounds[1] >= base.bounds[1]
        and source.bounds[2] <= base.bounds[2]
        and source.bounds[3] <= base.bounds[3]
    )
    return inside and base._uniform_alpha == 1 and source._uniform_alpha == 1


def _opaque_region(base, source, formula, clamp):
    """Blend opaque pixels: with both alphas 1 the composite is just ``B``."""
    canvas = np.array(base.rgba)
    left, top = source.offset[0] - base.offset[0], source.offset[1] - base.offset[1]
    width, height = source.size
    region = canvas[top : top + height, left : left + width]
    backdrop, _ = rules.split(region)
    layer, _ = rules.split(source.rgba)
    if clamp:
        np.clip(backdrop, 0, 1, out=backdrop)
        np.clip(layer, 0, 1, out=layer)
    with np.errstate(all="ignore"):
        rgb = np.asarray(formula(backdrop, layer), dtype=np.float32)
    _merge_into(region, rgb, np.ones((height, width), dtype=np.float32))
    return _publish(canvas, base.offset, base.color_space)


def _merge_into(region, rgb, alpha):
    """Write planar RGB and alpha back into a (possibly strided) RGBA view."""
    import cv2

    if region.flags.c_contiguous:
        cv2.mixChannels([rgb, alpha], [region], [0, 0, 1, 1, 2, 2, 3, 3])
        return
    region[..., :3] = rgb
    region[..., 3] = alpha


def _publish(array, offset, color_space):
    """Canonicalize transparent RGB, reject non-finite values and publish."""
    if not np.isfinite(array).all():
        raise ValueError("blend arithmetic must remain finite in float32")
    hidden = array[..., 3] == 0
    if hidden.any():
        array[..., :3][hidden] = 0
    return Buffer._publish(array, offset, color_space)


def _matte(base, layer, factor, member, kind):
    """Multiply the backdrop by the layer's alpha/luma (or its complement)."""
    if 0 in base.size:
        return base
    luma = member in (BlendMode.STENCIL_LUMA, BlendMode.SILHOUETTE_LUMA)
    weight = np.zeros((base.size[1], base.size[0], 1), dtype=np.float32)
    clipped = layer.crop(base.bounds)
    if 0 not in clipped.size:
        left = clipped.offset[0] - base.offset[0]
        top = clipped.offset[1] - base.offset[1]
        width, height = clipped.size
        cs, alpha_s = rules.split(clipped.rgba)
        values = rules.matte_factor(cs, alpha_s, luma) * np.float32(factor)
        weight[top : top + height, left : left + width, 0] = values
    if kind == "silhouette":
        weight = np.float32(1) - weight
    return _publish(base.rgba * weight, base.offset, base.color_space)


def _noise(source, member, context, layer_id):
    """Hash world pixel coordinates with one seed from the layer's stream."""
    ctx = RenderContext() if context is None else context
    if not isinstance(ctx, RenderContext):
        raise TypeError("context must be a RenderContext")
    if member is BlendMode.DISSOLVE:
        ctx = ctx.with_time(0.0)
    width, height = source.size
    seed = ctx.rng_for(layer_id).integers(0, 2**64, dtype=np.uint64)
    x = np.arange(width, dtype=np.uint64) + np.uint64(source.offset[0] % 2**64)
    y = np.arange(height, dtype=np.uint64) + np.uint64(source.offset[1] % 2**64)
    # Arithmetic intentionally wraps modulo 2**64, including negative coordinates.
    values = (
        x[None, :] * np.uint64(0x9E3779B97F4A7C15)
        ^ y[:, None] * np.uint64(0xD1B54A32D192ED03)
        ^ seed
    )
    values = (values ^ (values >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> 27)) * np.uint64(0x94D049BB133111EB)
    values ^= values >> 31
    return (values >> 40).astype(np.float32) * np.float32(2**-24)
