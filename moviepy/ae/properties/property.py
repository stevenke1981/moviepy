"""Typed static, keyed, callable and expression-driven animation properties."""

from dataclasses import replace
from typing import Generic, TypeVar

import numpy as np

from moviepy.ae.properties.keyframe import (
    Keyframe,
    evaluate_keyframes,
    normalize_keyframes,
)
from moviepy.ae.properties.spatial import resolve_roving_times
from moviepy.ae.properties.serialization import (
    MAX_KEYFRAMES,
    dumps,
    loads,
    require_fields,
    validate_payload,
)
from moviepy.ae.properties.values import (
    VALUE_TYPES,
    decode_value,
    default_value,
    encode_value,
    expression_value,
    finite_real,
    flatten_path,
    infer_value_type,
    normalize_value,
    path_topology,
    restore_expression_value,
    rebuild_path,
)

T = TypeVar("T")


class Property(Generic[T]):
    """Evaluate a validated property from a static, keyed or dynamic source.

    Parameters
    ----------
    value : object, optional
        Static value, time-only callable, keyframe sequence or Expression.
        Enum members normalize to inert string/integer tokens.
    value_type : str, optional
        ``float``, ``vec2``, ``vec3``, ``color``, ``bool``, ``enum`` or ``path``.
        Colors require an explicit type and may contain HDR components.
    keyframes : sequence, optional
        Keyframe objects or numeric worker supported shorthand records.
    expression : Expression or None, optional
        Expression overlay; its ``value`` observes the underlying source.
    spatial : bool, optional
        Enable arc-length interpolation of Vec2/Vec3 motion paths.

    Notes
    -----
    ``set_keyframes`` mutates and returns self. All ``with_*`` methods return
    independent properties. Dynamic callbacks execute only during evaluation;
    they cannot be restored from JSON. Values are detached immutable data.
    Direct expressions receive inert typed defaults, without execution during
    construction. Path expressions use ``(vertices, in_tangents, out_tangents,
    closed)`` tuples and the sandbox's 64-point sequence limit.
    """

    def __init__(
        self,
        value=0.0,
        *,
        value_type=None,
        keyframes=None,
        expression=None,
        spatial=False,
    ):
        from moviepy.ae.properties.expression import Expression

        if not isinstance(spatial, bool):
            raise TypeError("spatial must be bool")
        if isinstance(value, Expression):
            if expression is not None:
                raise ValueError("expression specified twice")
            expression, value = value, default_value(
                "float" if value_type is None else value_type
            )
        if keyframes is None and _is_keyframes(value):
            keyframes, value = value, 0.0
        sample = _first_value(keyframes) if keyframes is not None else value
        self._value_type = (
            ("float" if callable(sample) else infer_value_type(sample))
            if value_type is None
            else value_type
        )
        if not isinstance(self._value_type, str) or self._value_type not in VALUE_TYPES:
            raise ValueError("unsupported value_type")
        self._spatial = spatial
        if spatial and self._value_type not in ("vec2", "vec3"):
            raise ValueError("spatial properties require vec2 or vec3")
        self._expression = _check_expression(expression)
        self._keys = ()
        self._dimensions = None
        self._projection = None
        self._callable = value if callable(value) else None
        self._constant = (
            None
            if callable(value)
            else _initial_value(value, self._value_type, keyframes)
        )
        if keyframes is not None:
            self.set_keyframes(keyframes)

    @property
    def value_type(self):
        """Return the validated value kind used for every evaluated result."""
        return self._value_type

    @property
    def keyframes(self):
        """Return an immutable chronological tuple of detached Keyframes."""
        return self._keys

    @property
    def spatial(self):
        """Return whether vector animation follows an arc-length motion path."""
        return self._spatial

    def set_keyframes(self, keyframes):
        """Validate and replace keys in place, returning this Property.

        Parameters
        ----------
        keyframes : sequence
            Distinct finite-time keys. Discrete values require hold interpolation.

        Returns
        -------
        Property
            This same object, after successful atomic validation.
        """
        if not isinstance(keyframes, (list, tuple)):
            raise TypeError("keyframes must be a list or tuple")
        if len(keyframes) > MAX_KEYFRAMES:
            raise ValueError("keyframe count exceeds budget")
        keys = normalize_keyframes(keyframes)
        keys = tuple(
            replace(key, value=normalize_value(key.value, self.value_type))
            for key in keys
        )
        _check_key_values(keys, self.value_type)
        if any(key.roving for key in keys):
            if not self.spatial:
                raise ValueError("roving requires a spatial property")
            keys = resolve_roving_times(keys)
        self._keys, self._callable, self._dimensions, self._projection = (
            keys,
            None,
            None,
            None,
        )
        self._constant = keys[0].value
        return self

    def with_keyframes(self, keyframes):
        """Return an independent copy with new keys, retaining its expression."""
        return self._copy().set_keyframes(keyframes)

    def with_value(self, value):
        """Return a constant or callable copy while retaining its expression."""
        return Property(
            value,
            value_type=self.value_type,
            expression=self._expression,
            spatial=self.spatial,
        )

    def with_expression(self, expression):
        """Return a copy with a validated overlay, or remove it with None."""
        result = self._copy()
        result._expression = _check_expression(expression)
        return result

    def _copy(self):
        result = object.__new__(Property)
        result.__dict__ = self.__dict__.copy()
        if self._dimensions is not None:
            result._dimensions = tuple(item._copy() for item in self._dimensions)
        if self._projection is not None:
            source, dimension = self._projection
            result._projection = (source._copy(), dimension)
        return result

    def base_value_at(self, t):
        """Evaluate the underlying source without recursively applying expressions."""
        time = finite_real(t, "time")
        if self._projection is not None:
            source, dimension = self._projection
            bindings = getattr(self, "_evaluation_bindings", {})
            value = source.value_at(time, **bindings)[dimension]
        elif self._dimensions is not None:
            bindings = getattr(self, "_evaluation_bindings", {})
            value = tuple(
                dimension.value_at(time, **bindings) for dimension in self._dimensions
            )
        elif self._callable is not None:
            value = self._callable(time)
        elif self._keys:
            value = self._keyed_value(time)
        else:
            value = self._constant
        return normalize_value(value, self.value_type)

    def _keyed_value(self, time):
        if self.value_type != "path":
            return evaluate_keyframes(self._keys, time, spatial=self.spatial)
        template = self._keys[0].value
        if not template.vertices:
            return template
        numeric_keys = tuple(
            replace(key, value=flatten_path(key.value)) for key in self._keys
        )
        result = evaluate_keyframes(numeric_keys, time, spatial=False)
        return rebuild_path(result, template)

    def _expression_base_value_at(self, t):
        """Provide inert pre-expression data, including four-part path tuples."""
        return expression_value(self.base_value_at(t))

    def value_at(
        self,
        t,
        *,
        context=None,
        index=1,
        this_layer=None,
        this_comp=None,
        layer_id="property",
    ):
        """Evaluate the source and optional controlled expression bindings.

        Parameters
        ----------
        t : float
            Finite source time in seconds.
        context : RenderContext, optional
            Explicit frame rate and deterministic random seed.
        index : int, optional
            Layer index visible to the expression.
        this_layer, this_comp : mapping, optional
            Inert snapshots validated by the expression engine.
        layer_id : int or str, optional
            Stable identity for deterministic expression randomness.

        Returns
        -------
        object
            Finite, detached value conforming to ``value_type``.
        """
        time = finite_real(t, "time")
        source = self._evaluation_source(
            context, index, this_layer, this_comp, layer_id
        )
        value = source.base_value_at(time)
        if self._expression is not None:
            value = self._expression.evaluate(
                time,
                value=expression_value(value),
                property=source,
                context=context,
                index=index,
                this_layer=this_layer,
                this_comp=this_comp,
                layer_id=layer_id,
            )
            value = restore_expression_value(value, self.value_type)
        return normalize_value(value, self.value_type)

    def _evaluation_source(self, context, index, this_layer, this_comp, layer_id):
        if self._dimensions is None and self._projection is None:
            return self
        source = self._copy()
        source._evaluation_bindings = {
            "context": context,
            "index": index,
            "this_layer": this_layer,
            "this_comp": this_comp,
            "layer_id": layer_id,
        }
        return source

    def base_velocity_at(self, t, delta=1e-4):
        """Return base velocity, excluding hold jumps at exact key times.

        Interior keys use the outgoing segment; the final key uses its incoming
        segment. Hold segments have zero velocity, including their boundaries.
        Numeric differences remain inside the selected segment.
        """
        return self._velocity(t, self.base_value_at, delta, base=True)

    def velocity_at(self, t, *, context=None, delta=1e-4, **bindings):
        """Return evaluated velocity, including the expression overlay if present."""
        return self._velocity(
            t, lambda time: self.value_at(time, context=context, **bindings), delta
        )

    def _velocity(self, t, evaluator, delta, base=False):
        time, step = finite_real(t, "time"), finite_real(delta, "delta")
        if step <= 0:
            raise ValueError("delta must be positive")
        if self.value_type not in ("float", "vec2", "vec3", "color"):
            raise TypeError("velocity requires a numeric property")
        left, right = time - step, time + step
        if self._keys and (base or self._expression is None):
            window = self._base_difference_window(time, step)
            if window is None:
                return _zero(self.base_value_at(time))
            left, right = window
        duration = right - left
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("finite difference times are not representable")
        result = (np.asarray(evaluator(right)) - np.asarray(evaluator(left))) / duration
        return normalize_value(
            result.item() if result.ndim == 0 else result, self.value_type
        )

    def _base_difference_window(self, time, step):
        first, last = self._keys[0].time, self._keys[-1].time
        if time < first or time > last or first == last:
            return None
        index = (
            int(np.searchsorted([key.time for key in self._keys], time, side="right"))
            - 1
        )
        index = min(index, len(self._keys) - 2)
        left, right = self._keys[index : index + 2]
        ease = left.out_ease if left.out_ease is not None else right.in_ease
        if left.interp == "hold" or (ease is not None and ease.kind == "hold"):
            return None
        return max(time - step, left.time), min(time + step, right.time)

    def separate_dimensions(self):
        """Return independent scalar Properties for each Vec2/Vec3 component."""
        if self.value_type not in ("vec2", "vec3"):
            raise TypeError("separate dimensions requires vec2 or vec3")
        if self._dimensions is not None and self._expression is None:
            return tuple(item._copy() for item in self._dimensions)
        result = []
        for dimension in range(int(self.value_type[-1])):
            if self._callable is not None or self._expression is not None:
                source = self._copy()
                projected = Property(0)
                projected._projection = (source, dimension)
                result.append(projected)
            elif self._keys:
                keys = tuple(_dimension_key(key, dimension) for key in self._keys)
                result.append(Property(keyframes=keys))
            else:
                result.append(Property(self._constant[dimension]))
        return tuple(result)

    @classmethod
    def from_dimensions(cls, dimensions):
        """Combine two or three independent scalar Property curves into a vector."""
        if not isinstance(dimensions, (tuple, list)) or len(dimensions) not in (2, 3):
            raise ValueError("dimensions must contain two or three Properties")
        if any(
            not isinstance(item, Property) or item.value_type != "float"
            for item in dimensions
        ):
            raise TypeError("each dimension must be a float Property")
        result = cls(tuple(0 for _ in dimensions))
        result._dimensions = tuple(item._copy() for item in dimensions)
        return result

    @classmethod
    def from_audio_keyframes(cls, times, values):
        """Adapt sampled audio values into keys without decoding or analyzing audio."""
        if len(times) != len(values) or not len(times):
            raise ValueError("audio times and values must have equal nonzero length")
        if len(times) > MAX_KEYFRAMES:
            raise ValueError("audio sample count exceeds keyframe budget")
        return cls(
            keyframes=[Keyframe(time, value) for time, value in zip(times, values)]
        )

    def to_dict(self):
        """Serialize schema-one data; runtime callables raise an explicit TypeError."""
        if self._callable is not None or self._projection is not None:
            raise TypeError("callable properties cannot be serialized")
        if self._dimensions is not None:
            source = {
                "kind": "dimensions",
                "properties": [item.to_dict() for item in self._dimensions],
            }
        elif self._keys:
            source = {
                "kind": "keyframes",
                "keyframes": [key.to_dict() for key in self._keys],
            }
        else:
            source = {"kind": "constant", "value": encode_value(self._constant)}
        payload = {
            "schema": 1,
            "value_type": self.value_type,
            "source": source,
            "expression": (
                None if self._expression is None else self._expression.to_dict()
            ),
            "spatial": self.spatial,
        }
        validate_payload(payload)
        return payload

    @classmethod
    def from_dict(cls, payload):
        """Restore only strict inert schema-one data without importing named classes."""
        validate_payload(payload)
        return cls._decode(payload)

    @classmethod
    def _decode(cls, payload):
        require_fields(
            payload,
            ("schema", "value_type", "source", "expression", "spatial"),
            "property",
        )
        if type(payload["schema"]) is not int or payload["schema"] != 1:
            raise ValueError("unsupported property schema")
        kind = payload["value_type"]
        if type(kind) is not str or kind not in VALUE_TYPES:
            raise ValueError("unsupported property value_type")
        result = _decode_source(cls, payload["source"], kind, payload["spatial"])
        expression = payload["expression"]
        if expression is not None:
            from moviepy.ae.properties.expression import Expression

            result._expression = Expression.from_dict(expression)
        return result

    def to_json(self):
        """Return deterministic bounded schema-one JSON with finite numeric data."""
        return dumps(self.to_dict())

    @classmethod
    def from_json(cls, text):
        """Load strict JSON, rejecting duplicate fields and nonfinite data."""
        return cls.from_dict(loads(text))


def _is_keyframes(value):
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and (isinstance(value[0], Keyframe) or isinstance(value[0], (list, tuple)))
    )


def _first_value(keys):
    if not isinstance(keys, (tuple, list)) or not keys:
        raise ValueError("keyframes must be a nonempty list or tuple")
    first = keys[0]
    if isinstance(first, Keyframe):
        return first.value
    if not isinstance(first, (tuple, list)) or len(first) not in (2, 3):
        raise ValueError("keyframe shorthand requires time and value")
    return first[1]


def _initial_value(value, kind, keys):
    return normalize_value(_first_value(keys) if keys is not None else value, kind)


def _check_expression(expression):
    from moviepy.ae.properties.expression import Expression

    if expression is not None and not isinstance(expression, Expression):
        raise TypeError("expression must be Expression or None")
    return expression


def _check_key_values(keys, kind):
    if kind in ("bool", "enum") and any(key.interp != "hold" for key in keys):
        raise ValueError("bool and enum keyframes require hold interpolation")
    if kind == "path" and any(
        path_topology(key.value) != path_topology(keys[0].value) for key in keys
    ):
        raise ValueError("path keyframes must have matching topology")


def _zero(value):
    return tuple(0.0 for _ in value) if isinstance(value, tuple) else 0.0


def _dimension_key(key, dimension):
    metadata = {
        "value": key.value[dimension],
        "in_tangent": None,
        "out_tangent": None,
        "roving": False,
    }
    for name in ("in_speed", "out_speed"):
        value = getattr(key, name)
        if isinstance(value, (tuple, list)):
            metadata[name] = value[dimension]
    return replace(key, **metadata)


def _decode_source(cls, source, kind, spatial):
    if type(source) is not dict or type(source.get("kind")) is not str:
        raise ValueError("invalid property source")
    source_kind = source["kind"]
    if source_kind == "constant":
        require_fields(source, ("kind", "value"), "constant source")
        return cls(
            decode_value(source["value"], kind), value_type=kind, spatial=spatial
        )
    if source_kind == "keyframes":
        require_fields(source, ("kind", "keyframes"), "keyframe source")
        keys = source["keyframes"]
        if type(keys) is not list or not 0 < len(keys) <= MAX_KEYFRAMES:
            raise ValueError("invalid keyframe count")
        return cls(
            keyframes=[Keyframe.from_dict(key) for key in keys],
            value_type=kind,
            spatial=spatial,
        )
    if source_kind == "dimensions":
        require_fields(source, ("kind", "properties"), "dimension source")
        items = source["properties"]
        if type(items) is not list or len(items) not in (2, 3):
            raise ValueError("invalid dimension count")
        result = cls.from_dimensions([cls._decode(item) for item in items])
        if result.value_type != kind or spatial is not False:
            raise ValueError("dimensions must have matching nonspatial vector type")
        return result
    raise ValueError("unsupported property source kind")
