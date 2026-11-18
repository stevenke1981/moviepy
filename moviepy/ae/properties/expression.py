"""Safe finite Python expressions for animated property values."""

import math
from dataclasses import dataclass, field

from moviepy.ae.context import RenderContext
from moviepy.ae.properties._expression_helpers import registry
from moviepy.ae.properties._expression_sampling import PropertySampler
from moviepy.ae.properties._noise import NoiseState
from moviepy.ae.properties._sandbox import (
    ExpressionError,
    ExpressionSnapshot,
    Interpreter,
    parse,
    safe_number,
    sanitize,
)


__all__ = ["Expression", "ExpressionError", "ExpressionSnapshot", "wiggle"]
_OPERATION_ERRORS = (
    ArithmeticError,
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    RecursionError,
)


@dataclass(frozen=True)
class Expression:
    """Evaluate a bounded subset of Python with AE-style numeric helpers.

    Parameters
    ----------
    source : str
        One expression, at most 8192 characters. Statements, comprehensions,
        arbitrary object methods and all private identifiers are forbidden.

    Notes
    -----
    Scalars and short vectors support componentwise arithmetic. Metadata is
    copied into immutable snapshots. No Python builtins, modules, operating
    system, file or network capabilities are exposed. Per-expression AST,
    operation, sampling and value bounds reject excessive work.

    ``wiggle`` is seeded narrow-band oscillatory noise with continuous subframe
    sampling, rather than a reproduction of unpublished Adobe noise samples.
    ``ease`` is smoothstep. ``easeIn`` / ``easeOut`` use cubic Hermite curves
    with one zero endpoint slope and one linear endpoint slope. These are
    documented mathematical approximations rather than unpublished sample parity.

    Examples
    --------
    >>> Expression("value + time * 2").evaluate(0.5, value=3)
    4.0
    >>> Expression("normalize([3,4])").evaluate(0)
    (0.6, 0.8)
    """

    source: str
    _tree: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_tree", parse(self.source))

    def evaluate(
        self,
        t,
        *,
        value=0,
        property=None,
        context=None,
        index=1,
        this_layer=None,
        this_comp=None,
        layer_id="expression",
    ):
        """Return a deterministic expression value using inert bindings.

        Parameters
        ----------
        t : float
            Finite time in seconds.
        value : scalar or sequence, optional
            Pre-expression base value.
        property : Property, optional
            Internal sampling capability, always bypassing its expression.
        context : RenderContext, optional
            Frame rate and deterministic seed; never exposed to the expression.
        index : int, optional
            Positive layer index, default 1.
        this_layer, this_comp : dict or ExpressionSnapshot, optional
            Detached public metadata exposed as ``thisLayer`` / ``thisComp``.
        layer_id : int or str, optional
            Stable identity for independent random streams.

        Returns
        -------
        float, tuple, bool or str
            Finite inert value. Invalid expressions raise ``ExpressionError``.
        """
        try:
            return self._evaluate(
                t, value, property, context, index, this_layer, this_comp, layer_id
            )
        except ExpressionError:
            raise
        except _OPERATION_ERRORS as error:
            raise ExpressionError(
                "invalid expression operation: " + type(error).__name__
            ) from error

    def _evaluate(self, t, value, prop, context, index, layer, comp, layer_id):
        t, index = safe_number(t), safe_number(index)
        if not index.is_integer() or index < 1:
            raise ExpressionError("index must be a positive integer")
        if context is not None and type(context) is not RenderContext:
            raise ExpressionError("context must be RenderContext")
        if type(layer_id) not in (str, int):
            raise ExpressionError("layer_id must be an inert string or integer")
        if type(layer_id) is str and (not layer_id or len(layer_id) > 256):
            raise ExpressionError("layer_id must be a short nonempty string")
        if type(layer_id) is int and not 0 <= layer_id <= 2**53:
            raise ExpressionError("layer_id integer must be nonnegative and bounded")
        for snapshot in (layer, comp):
            if snapshot is not None and type(snapshot) not in (
                dict,
                ExpressionSnapshot,
            ):
                raise ExpressionError("metadata requires a plain dict or snapshot")
        context = RenderContext(t=t) if context is None else context.with_time(t)
        value = sanitize(value)
        environment = {
            "time": t,
            "value": value,
            "index": index,
            "pi": math.pi,
            "thisLayer": sanitize({} if layer is None else layer),
            "thisComp": sanitize({} if comp is None else comp),
        }
        helpers = registry(t, context.fps)
        helpers.update(PropertySampler(prop, value, t).registry())
        helpers.update(NoiseState(context, layer_id, value, t).registry())
        result = sanitize(Interpreter(environment, helpers).evaluate(self._tree))
        if type(result) in (ExpressionSnapshot, type(None)):
            raise ExpressionError("expression must return a scalar or vector")
        return result

    def to_dict(self):
        """Return a schema-1 source-only JSON-compatible dictionary.

        Returns
        -------
        dict
            Schema, fixed type tag and original validated source.
        """
        return {"schema": 1, "type": "expression", "source": self.source}

    @classmethod
    def from_dict(cls, data):
        """Validate schema-1 expression data without restoring capabilities.

        Parameters
        ----------
        data : dict
            Exactly ``schema``, ``type`` and ``source`` fields.

        Returns
        -------
        Expression
            Validated expression; unsupported data raises ``ExpressionError``.
        """
        if (
            type(data) is not dict
            or len(data) != 3
            or set(data) != {"schema", "type", "source"}
        ):
            raise ExpressionError("invalid expression schema fields")
        if type(data["schema"]) is not int or data["schema"] != 1:
            raise ExpressionError("unsupported expression schema")
        if type(data["type"]) is not str or data["type"] != "expression":
            raise ExpressionError("invalid expression type")
        return cls(data["source"])


def _literal(number, name):
    value = float(number)
    if isinstance(number, bool) or not math.isfinite(value):
        raise ValueError(name + " must be a finite real number")
    return value


def _wiggle_source(freq, amp, octaves, amp_mult):
    freq, amp_mult = _literal(freq, "freq"), _literal(amp_mult, "amp_mult")
    if freq < 0 or not 0 <= amp_mult <= 1:
        raise ValueError("freq must be nonnegative and amp_mult within [0, 1]")
    if isinstance(octaves, bool) or octaves != int(octaves) or not 1 <= octaves <= 8:
        raise ValueError("octaves must be an integer in 1..8")
    if isinstance(amp, (tuple, list)):
        if not 1 <= len(amp) <= 3:
            raise ValueError("amp vectors need one to three components")
        amp_text = "[" + ", ".join(repr(_literal(a, "amp")) for a in amp) + "]"
    else:
        amp_text = repr(_literal(amp, "amp"))
    return f"wiggle({freq!r}, {amp_text}, {int(octaves)}, {amp_mult!r})"


def wiggle(freq, amp, octaves=1, amp_mult=0.5, *, value=None):
    """Build an AE-style ``wiggle`` animation usable directly as a property value.

    This is the Python factory form of the expression helper
    ``wiggle(freq, amp, octaves, amp_mult)``. The noise is seeded per layer
    (from the layer's stable ``id``) and per ``RenderContext.rng_seed``, so the
    same seed and time always give the same value, while two layers, even with
    the same name, wiggle independently.

    Parameters
    ----------
    freq : float
        Wiggles per second, nonnegative. ``0`` returns the base value.
    amp : float or sequence of float
        Peak deviation in the property's units; a sequence gives one amplitude
        per vector component.
    octaves : int, optional
        Number of noise octaves, 1..8. Default 1.
    amp_mult : float, optional
        Amplitude multiplier per extra octave, within [0, 1]. Default 0.5.
    value : scalar or sequence, optional
        Base value to wiggle around. When omitted, an ``Expression`` is
        returned and the base is the property's typed default (0 for floats);
        effect parameters clamp the result into their declared range.

    Returns
    -------
    Expression or Property
        An ``Expression`` when ``value`` is omitted, else a ``Property`` holding
        ``value`` with the wiggle expression applied on top. Both serialize to
        JSON because the wiggle is stored as expression source.

    Examples
    --------
    >>> import moviepy.ae as ae
    >>> ae.wiggle(freq=2, amp=8).source
    'wiggle(2.0, 8.0, 1, 0.5)'
    >>> blur = ae.fx.GaussianBlur(blurriness=ae.wiggle(freq=2, amp=8, value=10))
    >>> ctx = ae.RenderContext(rng_seed=1)
    >>> a = blur.blurriness.value_at(0.5, context=ctx, layer_id="clip")
    >>> a == blur.blurriness.value_at(0.5, context=ctx, layer_id="clip")
    True
    >>> 2.0 <= a <= 18.0
    True
    """
    expression = Expression(_wiggle_source(freq, amp, octaves, amp_mult))
    if value is None:
        return expression
    from moviepy.ae.properties.property import Property

    return Property(value, expression=expression)
