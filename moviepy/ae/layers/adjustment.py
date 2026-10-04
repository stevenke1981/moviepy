"""Adjustment layer: applies its effect stack to the layers below it."""

from moviepy.ae.effects.adjustment import apply_adjustment
from moviepy.ae.layers.solid import SolidLayer


class AdjustmentLayer(SolidLayer):
    """Apply effects to the composite below, like an AE adjustment layer.

    Parameters
    ----------
    name : str, optional
        Layer name. Defaults to ``Adjustment Layer``.
    size : tuple of int, optional
        Coverage (width, height); ``Composition.add_adjustment`` uses the
        composition size.
    ``**kwargs``
        Every ``Layer`` keyword (``effects``, ``masks``, ``transform``,
        ``blend_mode``, ``track_matte`` ...).

    Notes
    -----
    The layer contributes no pixels of its own. Its rectangle, after masks,
    transform, opacity and track matte, is the area where the effected
    composite replaces the backdrop. Effects (and effect masks) run in
    composition space at the layer's time; temporal effects see the composite
    below at other times. Rendering the layer alone (``render``) returns
    that white coverage.

    Examples
    --------
    >>> import moviepy.ae as ae
    >>> comp = ae.Composition(size=(4, 1), fps=10, duration=1)
    >>> _ = comp.add_solid("red", color=(255, 0, 0))
    >>> adjust = comp.add_adjustment(effects=[ae.fx.Invert()])
    >>> adjust.masks = [ae.Mask.rect((0.5, 0), (2, 1))]
    >>> comp.get_frame(0).tolist()
    [[[0, 255, 255], [0, 255, 255], [255, 0, 0], [255, 0, 0]]]
    """

    def __init__(self, name="Adjustment Layer", *, size=(1920, 1080), **kwargs):
        kwargs.pop("color", None)
        super().__init__(name, color=(255, 255, 255), size=size, **kwargs)

    @property
    def is_adjustment(self):
        """Return True: effects apply to the composite below."""
        return True

    def apply_to(self, below, t, context=None):
        """Apply this layer to a world-space ``below`` buffer outside a comp."""
        coverage = self.render(t, context, bounds=below.bounds)
        return apply_adjustment(self, below, coverage, t, context)
