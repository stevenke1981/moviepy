"""Ordered, switchable effect stack owned by a layer (AE's Effect Controls)."""

from moviepy.ae.effects.base import AEEffect


_EPSILON = 1e-9


class EffectStack:
    """Apply effects top to bottom, like the AE Effect Controls panel.

    Parameters
    ----------
    effects : iterable of AEEffect, optional
        Initial effects in processing order.

    Notes
    -----
    Order matters: each effect receives the previous effect's output. A
    disabled effect is skipped exactly as if it were absent. Temporal effects
    (``temporal_window`` other than ``(0, 0)``) receive inputs at other times
    through a ``source_at`` provider that re-runs the earlier effects.

    Examples
    --------
    >>> from moviepy.ae.effects import EffectStack, registry
    >>> stack = EffectStack([registry.get("Invert")(), registry.get("Tint")()])
    >>> [effect.name for effect in stack]
    ['Invert', 'Tint']
    >>> stack.move(stack[1], 0) and [effect.name for effect in stack]
    ['Tint', 'Invert']
    """

    def __init__(self, effects=()):
        self._effects = []
        for effect in effects:
            self.add(effect)

    def __iter__(self):
        return iter(tuple(self._effects))

    def __len__(self):
        return len(self._effects)

    def __getitem__(self, index):
        return self._effects[index]

    def __repr__(self):
        return f"EffectStack({self._effects!r})"

    def add(self, effect, index=None):
        """Insert ``effect`` (at the end by default) and return it."""
        if not isinstance(effect, AEEffect):
            raise TypeError("effect must be an AEEffect")
        if any(item is effect for item in self._effects):
            raise ValueError("effect is already in this stack")
        if index is None:
            self._effects.append(effect)
        else:
            self._effects.insert(self._position(index, insert=True), effect)
        return effect

    def remove(self, effect):
        """Remove ``effect``; raise ``ValueError`` when it is absent."""
        for position, item in enumerate(self._effects):
            if item is effect:
                del self._effects[position]
                return
        raise ValueError("effect is not in this stack")

    def move(self, effect, index):
        """Move ``effect`` to position ``index`` and return the stack."""
        self.remove(effect)
        self._effects.insert(self._position(index, insert=True), effect)
        return self

    def clear(self):
        """Remove every effect."""
        self._effects = []

    def active(self):
        """Return the enabled effects in processing order."""
        return [effect for effect in self._effects if effect.enabled]

    def find(self, name):
        """Return the first effect whose ``name`` or class name matches."""
        for effect in self._effects:
            if name in (effect.name, type(effect).__name__):
                return effect
        raise KeyError(name)

    def _position(self, index, insert=False):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        size = len(self._effects) + (1 if insert else 0)
        if not -size <= index < size:
            raise IndexError("effect index out of range")
        return index % size if index < 0 else index

    def apply(self, buffer, t, context=None, *, bindings=None, source_at=None):
        """Run the enabled effects on ``buffer`` at layer time ``t``.

        Parameters
        ----------
        buffer : Buffer
            Premultiplied stack input (after masks).
        t : float
            Layer time.
        context : RenderContext, optional
            Render settings and expression context.
        bindings : dict, optional
            Expression identity bindings of the owning layer.
        source_at : callable, optional
            ``source_at(time)`` returns the stack input at another layer time;
            needed only by temporal effects.
        """
        effects = self.active()
        return self._run(effects, buffer, t, context, bindings, source_at)

    def _run(self, effects, buffer, t, context, bindings, source_at):
        """Apply ``effects`` in order, wiring temporal providers."""
        for position, effect in enumerate(effects):
            provider = None
            if source_at is not None:
                provider = self._provider(
                    effects[:position], effect, t, context, bindings, source_at
                )
            buffer = effect.process(
                buffer, t, context, bindings=bindings, source_at=provider
            )
        return buffer

    def _provider(self, earlier, effect, t, context, bindings, source_at):
        """Return ``dt -> input of effect at t + dt`` limited to its window."""
        cache = {}

        def provide(dt):
            values = effect.values_at(t, context, **(bindings or {}))
            before, after = effect.temporal_window(t, context, values)
            if not -before - _EPSILON <= dt <= after + _EPSILON:
                raise ValueError("requested time is outside the temporal window")
            key = round(float(dt), 9)
            if key not in cache:
                time = t + dt
                when = None if context is None else context.with_time(context.t + dt)
                inner = source_at(time)
                cache[key] = self._run(earlier, inner, time, when, bindings, source_at)
            return cache[key]

        return provide
