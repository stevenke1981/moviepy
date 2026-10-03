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

__all__ = ["Expression", "ExpressionError", "ExpressionSnapshot"]
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
