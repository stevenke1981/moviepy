"""Immutable render settings and deterministic per-frame random streams."""

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from fractions import Fraction
from numbers import Integral, Real
from typing import Optional, Union

from numpy.random import PCG64, Generator

from moviepy.ae.color import working_space


def _finite_real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _nonnegative_integer(value: Integral, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def _exact_time(value):
    """Preserve binary input values before temporal arithmetic is rounded."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, _TemporalTime):
        return value._exact_time
    return Fraction(_finite_real(value, "t"))


class _TemporalTime(float):
    """Pass a float-compatible time while retaining its exact arithmetic."""

    def __new__(cls, value):
        exact = _exact_time(value)
        result = super().__new__(cls, float(exact))
        result._exact_time = exact
        return result


@dataclass(frozen=True)
class RenderContext:
    """Carry validated render settings and reproducible random stream inputs.

    Parameters
    ----------
    t : float, optional
        Finite source time in seconds, including negative times. Defaults to 0.
    fps : float, optional
        Finite positive frame rate. Defaults to 30. The product ``t * fps``
        must also be finite.
    resolution_scale : {0.25, 0.5, 1.0}, optional
        Resolution setting passed to future rendering stages. Defaults to 1.
    quality : {"draft", "best"}, optional
        Quality setting passed to future rendering stages. Defaults to "best".
    rng_seed : int, optional
        Nonnegative integer random seed. Defaults to 0. Booleans are rejected.
    cache : object or None, optional
        Opaque cache reference. No cache operations are performed.
    shutter : object or None, optional
        Opaque shutter reference. No temporal sampling is performed.
    working_space : {"srgb", "linear"}, optional
        Blend, filter and resample encoded sRGB (legacy default) or decoded
        linear sRGB. Alpha is linear coverage in both modes. Nested comps
        inherit the active render context.

    Notes
    -----
    Fields cannot be reassigned, but opaque references may point to mutable
    objects. Cache and shutter are excluded from equality and hashing; a context
    hash is therefore not a complete rendering cache key. Random streams are
    reproducible for fixed inputs and dependencies, without consuming global
    NumPy random state.

    Examples
    --------
    >>> from moviepy.ae.context import RenderContext
    >>> ctx = RenderContext(t=0.5, fps=24, rng_seed=7)
    >>> ctx.frame_index
    12
    >>> ctx.with_time(-0.01).frame_index
    -1
    >>> ctx.t
    0.5
    >>> first = ctx.rng_for("foreground").random(3)
    >>> second = ctx.rng_for("foreground").random(3)
    >>> bool((first == second).all())
    True
    """

    t: float = 0.0
    fps: float = 30.0
    resolution_scale: float = 1.0
    quality: str = "best"
    rng_seed: int = 0
    cache: Optional[object] = field(default=None, compare=False, hash=False)
    shutter: Optional[object] = field(default=None, compare=False, hash=False)
    working_space: str = "srgb"
    _exact_time: Fraction = field(init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        working_space(self.working_space)
        original_time = self.t
        for name in ("t", "fps", "resolution_scale"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))
        object.__setattr__(self, "_exact_time", _exact_time(original_time))
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not math.isfinite(self.t * self.fps):
            raise ValueError("t * fps must be finite")
        if self.resolution_scale not in (0.25, 0.5, 1.0):
            raise ValueError("resolution_scale must be 0.25, 0.5, or 1.0")
        if not isinstance(self.quality, str):
            raise TypeError("quality must be a string")
        if self.quality not in ("draft", "best"):
            raise ValueError("quality must be 'draft' or 'best'")
        object.__setattr__(
            self, "rng_seed", _nonnegative_integer(self.rng_seed, "rng_seed")
        )

    @property
    def frame_index(self) -> int:
        """Return the frame containing ``t`` using floor without epsilon.

        Returns
        -------
        int
            ``floor(t * fps)``; negative fractional frame positions round down.
        """
        return math.floor(self.t * self.fps)

    def with_time(self, t: float) -> "RenderContext":
        """Return a new context at source time ``t``.

        Parameters
        ----------
        t : float
            Finite source time in seconds, including negative values.

        Returns
        -------
        RenderContext
            Validated context preserving all other settings and opaque references.

        Examples
        --------
        >>> original = RenderContext(fps=2)
        >>> later = original.with_time(1.5)
        >>> (original.t, later.t, later.frame_index)
        (0.0, 1.5, 3)
        """
        return replace(self, t=t)

    def rng_for(self, layer_id: Union[int, str]) -> Generator:
        """Create a fresh deterministic PCG64 stream for a stable layer identity.

        Parameters
        ----------
        layer_id : int or str
            Stable nonnegative integer or nonempty string. Booleans are rejected.
            Integer and string identities remain distinct.

        Returns
        -------
        numpy.random.Generator
            Independent generator seeded from ``rng_seed``, ``layer_id`` and
            ``frame_index``. Repeated calls restart the stream, so frame and layer
            evaluation order cannot change samples. Times within one frame share
            a stream; no subframe random sampling is defined here.

        Examples
        --------
        >>> ctx = RenderContext(rng_seed=3)
        >>> a, b = ctx.rng_for(1), ctx.rng_for(1)
        >>> a is b
        False
        >>> bool((a.integers(0, 10, 5) == b.integers(0, 10, 5)).all())
        True
        """
        if isinstance(layer_id, str):
            if not layer_id:
                raise ValueError("layer_id must be a nonempty string")
        else:
            layer_id = _nonnegative_integer(layer_id, "layer_id")
        payload = json.dumps(
            [self.rng_seed, layer_id, self.frame_index],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest = hashlib.blake2b(payload, digest_size=16).digest()
        return Generator(PCG64(int.from_bytes(digest, "big")))
