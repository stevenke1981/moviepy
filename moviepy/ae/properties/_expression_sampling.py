"""Controlled pre-expression Property sampling and loop semantics."""

import math

from moviepy.ae.properties._expression_helpers import componentwise
from moviepy.ae.properties._sandbox import ExpressionError, safe_number, sanitize


class PropertySampler:
    """Bind only the library's Property type and limit trusted base sampling."""

    def __init__(self, prop, value, t):
        if prop is not None:
            from moviepy.ae.properties.property import Property

            if type(prop) is not Property:
                raise ExpressionError("property binding must be a library Property")
        self.prop, self.value, self.t = prop, value, t
        self.remaining = 32

    def sample(self, t):
        """Read one bounded base value without evaluating its expression."""
        t = safe_number(t)
        self._consume()
        return (
            self.value
            if self.prop is None
            else sanitize(self.prop._expression_base_value_at(t))
        )

    def velocity(self, t):
        """Read base velocity, charging both finite-difference samples."""
        t = safe_number(t)
        self._consume(2)
        if self.prop is None:
            return componentwise(lambda _: 0, self.value)
        return sanitize(self.prop.base_velocity_at(t))

    def _consume(self, amount=1):
        self.remaining -= amount
        if self.remaining < 0:
            raise ExpressionError("property sample budget exceeded")

    def loop(self, type="cycle", n=0, incoming=False):
        """Apply cycle, pingpong, offset or continue to selected key intervals."""
        if type not in ("cycle", "pingpong", "offset", "continue"):
            raise ExpressionError("unknown loop type")
        n = safe_number(n)
        if n < 0 or not n.is_integer():
            raise ExpressionError("loop interval count must be a nonnegative integer")
        keys = () if self.prop is None else self.prop.keyframes
        if len(keys) < 2:
            return self.sample(self.t)
        count = len(keys) - 1 if n == 0 else min(int(n), len(keys) - 1)
        first, last = (
            (keys[0], keys[count]) if incoming else (keys[-count - 1], keys[-1])
        )
        start, end = first.time, last.time
        if (incoming and self.t >= start) or (not incoming and self.t <= end):
            return self.sample(self.t)
        if type == "continue":
            boundary = start if incoming else end
            return componentwise(
                lambda a, b: a + b * (self.t - boundary),
                self.sample(boundary),
                self.velocity(boundary),
            )
        period = end - start
        relative = (self.t - start) / period
        cycles = math.floor(relative)
        fraction = relative - cycles
        if type == "pingpong" and cycles % 2:
            fraction = 1 - fraction
        result = self.sample(start + fraction * period)
        if type == "offset":
            delta = componentwise(
                lambda a, b: b - a, self.sample(start), self.sample(end)
            )
            result = componentwise(lambda a, b: a + b * cycles, result, delta)
        return result

    def registry(self):
        """Return only the explicit pre-expression sampling capabilities."""
        return {
            "valueAtTime": self.sample,
            "velocityAtTime": self.velocity,
            "loopOut": lambda type="cycle", n=0: self.loop(type, n),
            "loopIn": lambda type="cycle", n=0: self.loop(type, n, incoming=True),
        }
