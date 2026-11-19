"""Footage-time properties, separate from the owning layer's animation clock."""

from moviepy.ae._geometry import to_property
from moviepy.ae.properties.values import finite_real


def remap_property(value):
    """Coerce an optional source-time curve without evaluating callbacks."""
    return None if value is None else to_property(value, "time_remap", "float")


def remap_time(prop, local_t, context=None, *, bindings=None):
    """Map layer-local seconds to continuous footage seconds without clamping."""
    time = finite_real(local_t, "local_t")
    if prop is None:
        return time
    return prop.value_at(time, context=context, **(bindings or {}))
