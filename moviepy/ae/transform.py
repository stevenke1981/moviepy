"""After Effects ordered affine transforms for animated layer geometry.

The coordinate space uses pixel centers: pixel ``[row, col]`` of a buffer sits
at world ``(offset_x + col, offset_y + row)``. Matrices therefore map source
world coordinates onto destination world coordinates, and a default
``Transform`` is the identity for every source size. Pixel resampling itself
lives in ``moviepy.ae.warp``.

Examples
--------
>>> from moviepy.ae.transform import Transform
>>> transform = Transform(position=(10, 20), anchor_point=(0, 0), scale=(50, 50))
>>> float(transform.matrix_at(0.0)[0, 0])
0.5
>>> transform.opacity_at(0.0)
1.0
"""

import bisect
import math

import numpy as np

from moviepy.ae._geometry import (
    opacity_factor,
    to_property,
    validate_extent_size,
    validate_flag,
    validate_integers,
    validate_pair,
)
from moviepy.ae.buffer import Buffer
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.serialization import require_fields, validate_payload
from moviepy.ae.properties.values import finite_real
from moviepy.ae.warp import INTERPOLATIONS, resolve_interpolation, warp_buffer

_SCHEMA = 1
_ANIMATED_FIELDS = (
    "scale",
    "rotation",
    "opacity",
    "orientation",
    "x_rotation",
    "y_rotation",
    "z_rotation",
    "position_z",
    "scale_z",
)
_TRANSFORM_KEYS = (
    "schema",
    "type",
    "anchor_point",
    "position",
    "scale",
    "rotation",
    "opacity",
    "auto_orient",
    "interpolation",
    "constrain_proportions",
    "three_d",
    "orientation",
    "x_rotation",
    "y_rotation",
    "z_rotation",
    "position_z",
    "scale_z",
)


def affine_matrix(position, rotation, scale, anchor):
    """Return the AE matrix ``T(position) · R(rotation) · S(scale) · T(-anchor)``.

    Parameters
    ----------
    position : sequence of float
        Destination world coordinates of the anchor point.
    rotation : float
        Clockwise-on-screen rotation in degrees; multiples of 360 are allowed.
    scale : sequence of float
        Percent scale per axis. Negative values mirror the layer.
    anchor : sequence of float
        Source coordinates of the point that ``position`` places.

    Returns
    -------
    numpy.ndarray
        float64 (3, 3) matrix mapping source world coordinates to destination
        world coordinates in pixel-center space.

    Examples
    --------
    >>> affine_matrix((0, 0), 0, (100, 100), (0, 0)).tolist()
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    """
    position_x, position_y = validate_pair(position, "position")
    anchor_x, anchor_y = validate_pair(anchor, "anchor_point")
    percent_x, percent_y = validate_pair(scale, "scale")
    angle = math.radians(finite_real(rotation, "rotation"))
    cos, sin = math.cos(angle), math.sin(angle)
    factor_x, factor_y = percent_x / 100.0, percent_y / 100.0
    m00, m01 = factor_x * cos, -factor_y * sin
    m10, m11 = factor_x * sin, factor_y * cos
    matrix = np.array(
        [
            [m00, m01, position_x - (m00 * anchor_x + m01 * anchor_y)],
            [m10, m11, position_y - (m10 * anchor_x + m11 * anchor_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    # Adding zero normalizes the negative zeros that exact angles produce.
    matrix += 0.0
    return matrix


def _tangent(position, t, bindings):
    """Return the position velocity as (x, y), or None when stationary."""
    velocity = position.velocity_at(t, **bindings)
    delta = (float(velocity[0]), float(velocity[1]))
    return None if delta == (0.0, 0.0) else delta


def _segment_index(times, t):
    """Return the index of the keyframe segment containing ``t``."""
    index = bisect.bisect_right(times, t) - 1
    return min(max(index, 0), len(times) - 2)


def _segment_tangent(position, times, index, bindings):
    """Return the tangent at the midpoint of one keyframe segment."""
    midpoint = (times[index] + times[index + 1]) * 0.5
    return _tangent(position, midpoint, bindings)


def _held_tangent(position, t, bindings):
    """Return the tangent of the nearest moving segment, searching back first.

    ``Property.velocity_at`` reports zero outside the keyframe range and across
    a hold segment. Auto-orient must not read that as "face right", so the
    search walks backwards to the segment that last moved, then forwards, and
    gives up only when the whole path is stationary.
    """
    times = [float(key.time) for key in position.keyframes]
    if len(times) < 2:
        return None
    last = len(times) - 2
    start = _segment_index(times, t)
    for index in range(start, -1, -1):
        delta = _segment_tangent(position, times, index, bindings)
        if delta is not None:
            return delta
    for index in range(start + 1, last + 1):
        delta = _segment_tangent(position, times, index, bindings)
        if delta is not None:
            return delta
    return None


class Transform:
    """Animate anchor point, position, scale, rotation and opacity of a layer.

    Parameters
    ----------
    anchor_point : Property or sequence or None, optional
        Source point placed at ``position``. ``None`` (default) resolves to the
        pixel-center middle of the source, ``((w-1)/2, (h-1)/2)`` plus its offset.
    position : Property or sequence or None, optional
        Destination world coordinates. ``None`` (default) resolves to the anchor
        point, which makes an untouched transform the identity.
    scale : Property or sequence, optional
        Percent per axis; negative values mirror. Defaults to ``(100, 100)``.
    rotation : Property or float, optional
        Degrees, clockwise on screen, unlimited in magnitude.
    opacity : Property or float, optional
        Percent in 0..100. A bare number is range checked at assignment; a
        supplied ``Property`` is not, because animated values are clamped when
        they are applied.
    auto_orient : bool, optional
        Replace ``rotation`` with the direction of the position path.
    interpolation : {"auto", "nearest", "linear", "cubic"}, optional
        ``auto`` follows ``RenderContext.quality``: best selects cubic and draft
        selects linear.
    constrain_proportions : bool, optional
        Make ``scale_axis`` preserve the current aspect ratio.
    three_d : bool, optional
        Reserved for WS-22. When true, evaluation raises ``NotImplementedError``
        instead of silently ignoring the 3D fields.
    orientation, x_rotation, y_rotation, z_rotation, position_z, scale_z
        Reserved 3D fields. They are validated, stored and serialized, and are
        never applied before WS-22.

    Notes
    -----
    Attribute assignment accepts a bare value, a keyframe sequence, a callable,
    an ``Expression`` or an existing ``Property``.
    Matrices follow the AE order
    ``translate(position) · rotate · scale · translate(-anchor)``.

    Examples
    --------
    >>> from moviepy.ae.transform import Transform
    >>> transform = Transform(position=(10, 20), anchor_point=(0, 0), scale=(50, 50))
    >>> float(transform.matrix_at(0.0)[0, 0])
    0.5
    >>> transform.opacity_at(0.0)
    1.0
    """

    def __init__(
        self,
        *,
        anchor_point=None,
        position=None,
        scale=(100.0, 100.0),
        rotation=0.0,
        opacity=100.0,
        auto_orient=False,
        interpolation="auto",
        constrain_proportions=False,
        three_d=False,
        orientation=(0.0, 0.0, 0.0),
        x_rotation=0.0,
        y_rotation=0.0,
        z_rotation=0.0,
        position_z=0.0,
        scale_z=100.0,
    ):
        self._fields = {}
        self._anchor_auto = True
        self._position_auto = True
        self.anchor_point = anchor_point
        self.position = position
        self.scale = scale
        self.rotation = rotation
        self.opacity = opacity
        self.orientation = orientation
        self.x_rotation = x_rotation
        self.y_rotation = y_rotation
        self.z_rotation = z_rotation
        self.position_z = position_z
        self.scale_z = scale_z
        self.auto_orient = auto_orient
        self.interpolation = interpolation
        self.constrain_proportions = constrain_proportions
        self.three_d = three_d

    # -- animated 2D fields ------------------------------------------------- #

    @property
    def anchor_point(self):
        """Return the anchor Property, or None for the automatic source center."""
        return None if self._anchor_auto else self._fields["anchor_point"]

    @anchor_point.setter
    def anchor_point(self, value):
        if value is None:
            self._anchor_auto = True
            self._fields.pop("anchor_point", None)
            return
        self._anchor_auto = False
        self._fields["anchor_point"] = to_property(value, "anchor_point", "vec2")

    @property
    def position(self):
        """Return the position Property, or None for anchor-relative placement."""
        return None if self._position_auto else self._fields["position"]

    @position.setter
    def position(self, value):
        if value is None:
            self._position_auto = True
            self._fields.pop("position", None)
            return
        self._position_auto = False
        self._fields["position"] = to_property(value, "position", "vec2")

    @property
    def scale(self):
        """Return the percent scale Property."""
        return self._fields["scale"]

    @scale.setter
    def scale(self, value):
        self._fields["scale"] = to_property(value, "scale", "vec2")

    @property
    def rotation(self):
        """Return the degrees rotation Property."""
        return self._fields["rotation"]

    @rotation.setter
    def rotation(self, value):
        self._fields["rotation"] = to_property(value, "rotation", "float")

    @property
    def opacity(self):
        """Return the percent opacity Property."""
        return self._fields["opacity"]

    @opacity.setter
    def opacity(self, value):
        self._fields["opacity"] = to_property(value, "opacity", "float", (0.0, 100.0))

    # -- reserved 3D fields ------------------------------------------------- #

    @property
    def orientation(self):
        """Return the reserved orientation Property, unused before WS-22."""
        return self._fields["orientation"]

    @orientation.setter
    def orientation(self, value):
        self._fields["orientation"] = to_property(value, "orientation", "vec3")

    @property
    def x_rotation(self):
        """Return the reserved x rotation Property, unused before WS-22."""
        return self._fields["x_rotation"]

    @x_rotation.setter
    def x_rotation(self, value):
        self._fields["x_rotation"] = to_property(value, "x_rotation", "float")

    @property
    def y_rotation(self):
        """Return the reserved y rotation Property, unused before WS-22."""
        return self._fields["y_rotation"]

    @y_rotation.setter
    def y_rotation(self, value):
        self._fields["y_rotation"] = to_property(value, "y_rotation", "float")

    @property
    def z_rotation(self):
        """Return the reserved z rotation Property, unused before WS-22."""
        return self._fields["z_rotation"]

    @z_rotation.setter
    def z_rotation(self, value):
        self._fields["z_rotation"] = to_property(value, "z_rotation", "float")

    @property
    def position_z(self):
        """Return the reserved position z Property, unused before WS-22."""
        return self._fields["position_z"]

    @position_z.setter
    def position_z(self, value):
        self._fields["position_z"] = to_property(value, "position_z", "float")

    @property
    def scale_z(self):
        """Return the reserved z scale Property, unused before WS-22."""
        return self._fields["scale_z"]

    @scale_z.setter
    def scale_z(self, value):
        self._fields["scale_z"] = to_property(value, "scale_z", "float")

    # -- switches ----------------------------------------------------------- #

    @property
    def auto_orient(self):
        """Return whether rotation follows the position path direction."""
        return self._auto_orient

    @auto_orient.setter
    def auto_orient(self, value):
        self._auto_orient = validate_flag(value, "auto_orient")

    @property
    def interpolation(self):
        """Return the resampling filter name, possibly ``auto``."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value):
        if not isinstance(value, str) or value not in INTERPOLATIONS:
            raise ValueError("interpolation must be auto, nearest, linear or cubic")
        self._interpolation = value

    @property
    def constrain_proportions(self):
        """Return whether ``scale_axis`` preserves the aspect ratio."""
        return self._constrain_proportions

    @constrain_proportions.setter
    def constrain_proportions(self, value):
        self._constrain_proportions = validate_flag(value, "constrain_proportions")

    @property
    def three_d(self):
        """Return the reserved 3D flag; evaluation raises until WS-22."""
        return self._three_d

    @three_d.setter
    def three_d(self, value):
        self._three_d = validate_flag(value, "three_d")

    # -- evaluation --------------------------------------------------------- #

    def resolve_anchor(self, size, offset=(0, 0), t=0.0, **bindings):
        """Return the source point that ``position`` places.

        Parameters
        ----------
        size : tuple of int
            Positive source (width, height) used by the automatic center. An
            empty source may pass ``(0, 0)``.
        offset : tuple of int, optional
            World coordinates of source pixel [0, 0].
        t : float, optional
            Evaluation time for an animated anchor point.
        ``**bindings``
            Forwarded to ``Property.value_at``.

        Returns
        -------
        tuple of float
            Source coordinates in pixel-center space.
        """
        if self._anchor_auto:
            width, height = validate_extent_size(size)
            origin = validate_integers(offset, "offset")
            return (origin[0] + (width - 1) / 2.0, origin[1] + (height - 1) / 2.0)
        return self._fields["anchor_point"].value_at(t, **bindings)

    def resolve_position(self, t, anchor, **bindings):
        """Return the destination coordinates for the resolved anchor point."""
        if self._position_auto:
            return (float(anchor[0]), float(anchor[1]))
        return self._fields["position"].value_at(t, **bindings)

    def rotation_at(self, t, **bindings):
        """Return the applied rotation in degrees, honouring ``auto_orient``."""
        if self._auto_orient:
            return self._path_rotation(t, bindings)
        return finite_real(self._fields["rotation"].value_at(t, **bindings), "rotation")

    def _path_rotation(self, t, bindings):
        """Derive rotation from the position velocity, holding the last direction.

        After Effects keeps an auto-oriented layer aimed along the final motion
        path tangent once the path ends and holds the previous direction across
        a hold segment, so a zero instantaneous velocity falls back to the
        nearest moving segment instead of snapping the layer level.
        """
        if self._position_auto:
            return 0.0
        position = self._fields["position"]
        delta = _tangent(position, t, bindings)
        if delta is None:
            delta = _held_tangent(position, t, bindings)
        if delta is None:
            return 0.0
        return math.degrees(math.atan2(delta[1], delta[0]))

    def opacity_at(self, t, **bindings):
        """Return the clamped 0..1 opacity factor applied to every channel."""
        return opacity_factor(self._fields["opacity"].value_at(t, **bindings))

    def matrix_at(self, t, *, size=None, offset=(0, 0), **bindings):
        """Return the 3x3 source-to-destination matrix at time ``t``.

        Parameters
        ----------
        t : float
            Finite evaluation time in seconds.
        size : tuple of int, optional
            Source (width, height); required when the anchor point or the
            position is automatic.
        offset : tuple of int, optional
            World coordinates of source pixel [0, 0].
        ``**bindings``
            Forwarded to the animated properties, including ``context``.

        Returns
        -------
        numpy.ndarray
            float64 (3, 3) matrix in AE order.

        Raises
        ------
        NotImplementedError
            When ``three_d`` is true, because WS-22 owns 3D evaluation.
        ValueError
            When ``size`` is omitted while the anchor point is automatic, or
            when an animated property yields a non-finite value.
        """
        if self._three_d:
            raise NotImplementedError("three-dimensional transforms arrive with WS-22")
        if size is None and self._anchor_auto:
            raise ValueError("size is required to resolve an automatic anchor_point")
        anchor = self.resolve_anchor(size, offset, t, **bindings)
        position = self.resolve_position(t, anchor, **bindings)
        scale = self._fields["scale"].value_at(t, **bindings)
        rotation = self.rotation_at(t, **bindings)
        return affine_matrix(position, rotation, scale, anchor)

    def apply(
        self,
        buffer,
        t=0.0,
        context=None,
        *,
        bounds=None,
        interpolation=None,
        **bindings,
    ):
        """Resample one buffer through this transform at time ``t``.

        Parameters
        ----------
        buffer : Buffer
            Source pixels; their world offset participates in the mapping.
        t : float, optional
            Evaluation time in seconds.
        context : RenderContext, optional
            Supplies ``quality`` for ``auto`` interpolation.
        bounds : tuple of int, optional
            Exact destination rectangle for region-of-interest rendering.
        interpolation : str, optional
            Override the transform filter for this call only.
        ``**bindings``
            Expression bindings forwarded to every animated property.

        Returns
        -------
        Buffer
            Transformed premultiplied pixels at the destination rectangle.

        Notes
        -----
        ``context`` always wins over a ``context`` entry supplied through
        ``bindings``, because the same object must reach both the filter
        selection and the expression evaluation.
        """
        if not isinstance(buffer, Buffer):
            raise TypeError("buffer must be a Buffer")
        evaluation = dict(bindings)
        evaluation["context"] = context
        matrix = self.matrix_at(t, size=buffer.size, offset=buffer.offset, **evaluation)
        factor = self.opacity_at(t, **evaluation)
        selected = self._interpolation if interpolation is None else interpolation
        return warp_buffer(
            buffer,
            matrix,
            factor,
            interpolation=resolve_interpolation(selected, context),
            bounds=bounds,
        )

    # -- scale helpers ------------------------------------------------------ #

    def set_scale(self, x, y=None):
        """Replace the scale with a static value pair.

        Parameters
        ----------
        x : float
            Horizontal percent, or both axes when ``y`` is omitted.
        y : float, optional
            Vertical percent.
        """
        first = finite_real(x, "scale")
        second = first if y is None else finite_real(y, "scale")
        self.scale = (first, second)

    def scale_axis(self, axis, value):
        """Set one scale axis, optionally preserving the aspect ratio.

        Parameters
        ----------
        axis : {"x", "y"}
            Axis to change.
        value : float
            New percent for that axis. With ``constrain_proportions`` the other
            axis is scaled by the same ratio.

        Raises
        ------
        ValueError
            When the axis is unknown, the current scale is animated, or a locked
            zero axis would require a division.
        """
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        number = finite_real(value, "scale")
        current = list(_static_pair(self._fields["scale"], "scale"))
        index = 0 if axis == "x" else 1
        if self._constrain_proportions:
            if current[index] == 0.0:
                raise ValueError("constrain_proportions cannot preserve a zero axis")
            ratio = number / current[index]
            current[1 - index] *= ratio
        current[index] = number
        self.scale = tuple(current)

    # -- serialization ------------------------------------------------------ #

    def to_dict(self):
        """Return strict schema-one data for every stored field."""
        payload = {
            "schema": _SCHEMA,
            "type": "transform",
            "anchor_point": _encode_optional(self._anchor_auto, "anchor_point", self),
            "position": _encode_optional(self._position_auto, "position", self),
            "auto_orient": self._auto_orient,
            "interpolation": self._interpolation,
            "constrain_proportions": self._constrain_proportions,
            "three_d": self._three_d,
        }
        for name in _ANIMATED_FIELDS:
            payload[name] = self._fields[name].to_dict()
        validate_payload(payload)
        return payload

    @classmethod
    def from_dict(cls, payload):
        """Restore a transform from strict schema-one data.

        Parameters
        ----------
        payload : dict
            Output of ``to_dict``. Unknown or missing fields are rejected.

        Returns
        -------
        Transform
            Independent transform with equal evaluated matrices.
        """
        validate_payload(payload)
        require_fields(payload, _TRANSFORM_KEYS, "transform")
        if type(payload["schema"]) is not int or payload["schema"] != _SCHEMA:
            raise ValueError("unsupported transform schema")
        if payload["type"] != "transform":
            raise ValueError("payload type must be 'transform'")
        result = cls(
            anchor_point=_decode_optional(payload["anchor_point"], "anchor_point"),
            position=_decode_optional(payload["position"], "position"),
            auto_orient=payload["auto_orient"],
            interpolation=payload["interpolation"],
            constrain_proportions=payload["constrain_proportions"],
            three_d=payload["three_d"],
            **{name: Property.from_dict(payload[name]) for name in _ANIMATED_FIELDS},
        )
        return result


def _static_pair(prop, name):
    """Return the constant vec2 of a Property, rejecting animated sources."""
    try:
        payload = prop.to_dict()
    except TypeError as error:
        raise ValueError(f"{name} must be static for scale_axis") from error
    if payload["source"]["kind"] != "constant" or payload["expression"] is not None:
        raise ValueError(f"{name} must be static for scale_axis")
    return tuple(payload["source"]["value"])


def _encode_optional(auto, name, transform):
    """Serialize an optional Property as None when it is automatic."""
    return None if auto else transform._fields[name].to_dict()


def _decode_optional(payload, name):
    """Restore an optional Property, keeping None automatic."""
    if payload is None:
        return None
    if type(payload) is not dict:
        raise ValueError(f"{name} must be null or a property object")
    return Property.from_dict(payload)
