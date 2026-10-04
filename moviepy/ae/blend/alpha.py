"""Alpha rules that turn a ``B(Cb, Cs)`` color mix into premultiplied pixels.

All functions operate on aligned premultiplied float32 regions: ``cb`` and
``alpha_b`` for the backdrop (the accumulated layers below) and ``cs`` and
``alpha_s`` for the source layer, with opacity already applied to the source. Alpha arrays are
replicated to three channels so every operation is same-shape and contiguous.

The general rule is the W3C *Compositing and Blending Level 1* source-over
composite with blending (§9.1.4 / §10)::

    co = (1 - ab) * Cs_p + (1 - as) * Cb_p + as * ab * B(Cb, Cs)
    ao = as + ab - as * ab

With "Preserve Underlying Transparency" enabled the source is composited
*atop* the backdrop instead, so the backdrop alpha is kept::

    co = (1 - as) * Cb_p + as * ab * B(Cb, Cs)
    ao = ab

Here ``as``/``ab`` are the source/backdrop alpha, ``Cs_p``/``Cb_p`` the
premultiplied colors and ``Cs``/``Cb`` the straight colors.
"""

import numpy as np

from moviepy.ae.blend._formulas import REC709_LUMA, luminance


_ONE = np.float32(1.0)


def split(rgba):
    """Return contiguous premultiplied RGB and a 3-channel replicated alpha.

    Same-shape contiguous float32 operands are several times faster than
    broadcasting a ``(H, W, 1)`` alpha against strided RGB views.
    """
    import cv2

    height, width = rgba.shape[:2]
    rgb = np.empty((height, width, 3), dtype=np.float32)
    alpha = np.empty((height, width), dtype=np.float32)
    if rgb.size:
        cv2.mixChannels([rgba], [rgb, alpha], [0, 0, 1, 1, 2, 2, 3, 3])
    return rgb, replicate(alpha)


def replicate(alpha):
    """Stack one ``(H, W)`` plane into a contiguous ``(H, W, 3)`` array."""
    import cv2

    if not alpha.size:
        return np.empty((*alpha.shape, 3), dtype=np.float32)
    return cv2.merge([alpha, alpha, alpha])


def straight(rgb, alpha3):
    """Unpremultiply RGB, returning zero wherever alpha is zero."""
    inverse = np.zeros(alpha3.shape[:2], dtype=np.float32)
    plane = np.ascontiguousarray(alpha3[..., 0])
    np.divide(_ONE, plane, out=inverse, where=plane > 0)
    return rgb * replicate(inverse)


def separable(
    cb, alpha_b, cs, alpha_s, formula, *, preserve, clamp=True, classic=False
):
    """Composite one region with a separable or whole-color blend formula.

    Parameters
    ----------
    cb, cs : numpy.ndarray
        Premultiplied backdrop and source RGB of identical ``(H, W, 3)`` shape.
    alpha_b, alpha_s : numpy.ndarray
        Matching alpha replicated to ``(H, W, 3)``.
    formula : callable
        ``B(Cb, Cs)`` on straight colors.
    preserve : bool
        Keep the backdrop alpha (source-atop) instead of source-over.
    clamp : bool, optional
        Clamp straight inputs to [0, 1] before mixing (8/16 bpc semantics).
    classic : bool, optional
        Feed *premultiplied* colors to ``formula``. This reproduces the legacy
        "Classic" modes, which mixed colors without unpremultiplying first;
        for fully opaque pixels the result equals the modern mode.

    Returns
    -------
    tuple of numpy.ndarray
        Premultiplied ``(H, W, 3)`` RGB and ``(H, W)`` alpha.
    """
    if classic:
        backdrop, source = cb.copy(), cs.copy()
    else:
        backdrop, source = straight(cb, alpha_b), straight(cs, alpha_s)
    if clamp:
        np.clip(backdrop, 0, 1, out=backdrop)
        np.clip(source, 0, 1, out=source)
    rgb = np.asarray(formula(backdrop, source), dtype=np.float32)
    rgb *= alpha_s
    rgb *= alpha_b
    weight = _ONE - alpha_s
    weight *= cb
    rgb += weight
    alpha = alpha_b[..., 0].copy()
    if preserve:
        return rgb, alpha
    weight = _ONE - alpha_b
    weight *= cs
    rgb += weight
    source_alpha = alpha_s[..., 0]
    return rgb, source_alpha + alpha - source_alpha * alpha


def dissolve(cb, alpha_b, cs, alpha_s, noise, *, preserve):
    """Show whole, fully opaque source pixels where ``noise < source alpha``.

    ``noise`` is a deterministic ``(H, W)`` uniform field in [0, 1). The
    proportion of revealed pixels therefore equals the source alpha (which
    already includes layer opacity), matching the AE Dissolve description.
    """
    reveal = noise < alpha_s[..., 0]
    color = straight(cs, alpha_s)
    mask = reveal[..., None]
    if preserve:
        return np.where(mask, color * alpha_b, cb), alpha_b[..., 0].copy()
    return np.where(mask, color, cb), np.where(reveal, _ONE, alpha_b[..., 0])


def matte_factor(cs, alpha_s, luma):
    """Return the ``(H, W)`` stencil factor from source alpha or luma.

    Luma is Rec.709 luminance of the straight source color multiplied by its
    alpha, which equals the luminance of the premultiplied color.
    """
    if not luma:
        return alpha_s[..., 0].copy()
    return np.clip(luminance(cs, REC709_LUMA)[..., 0], 0, 1)


def alpha_add(cb, alpha_b, cs, alpha_s):
    """Keep the Normal straight color but add the two alpha values.

    Complementary edges (``as + alpha_b = 1`` with equal colors) become seamless and
    fully opaque, which is the purpose of AE's Alpha Add mode.
    """
    rgb = cs + (_ONE - alpha_s) * cb
    source_alpha, backdrop_alpha = alpha_s[..., 0], alpha_b[..., 0]
    normal_alpha = source_alpha + backdrop_alpha - source_alpha * backdrop_alpha
    summed = np.minimum(_ONE, source_alpha + backdrop_alpha)
    scale = np.zeros_like(summed)
    np.divide(summed, normal_alpha, out=scale, where=normal_alpha > 0)
    return rgb * replicate(scale), summed
