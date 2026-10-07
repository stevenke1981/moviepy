"""Editable SVG artwork as an ordinary AE layer with a bounded source cache."""

import hashlib
from collections import OrderedDict
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

from moviepy.ae._geometry import validate_pixel_size
from moviepy.ae.layers.base import Layer
from moviepy.ae.properties.property import Property
from moviepy.ae.properties.values import finite_real
from moviepy.ae.svg import backend
from moviepy.ae.svg.document import (
    MAX_SVG_BYTES,
    binding_target,
    evaluate_svg,
    parse_svg,
)


class SVGLayer(Layer):
    """Render a static SVG document with AE Property-driven element attributes.

    ``size`` is the explicit raster viewport (width, height). Bind element ids
    with ``bind(id, attribute, Property(...))``; keys use source-local seconds.
    String Properties use hold interpolation, numeric/color Properties can
    interpolate. Layer transforms, parenting, masks and effects work normally.

    The optional ``svg`` extra supplies resvg and fontTools. Only explicit local
    ``font_files`` are used; all nonempty text needs a supplied matching font.
    External resources, scripts, stylesheets and native SVG animation are refused.
    This is SDR SVG rasterization at ``size``, not HDR or continuous vector scaling.
    """

    def __init__(
        self,
        svg,
        name="SVG",
        *,
        size,
        font_files=(),
        cache_entries=8,
        cache_bytes=64 * 1024 * 1024,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        for value in (cache_entries, cache_bytes):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("SVG cache limits must be nonnegative integers")
        self._max_entries, self._max_bytes = int(cache_entries), int(cache_bytes)
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._bindings = {}
        self._font_key = None
        self._faces = ()
        self.svg = svg
        self.size = size
        self.font_files = font_files

    @classmethod
    def from_file(cls, path, name="SVG", **kwargs):
        """Read one explicit UTF-8 local file as an editable in-memory snapshot."""
        data = backend.read_local_file(path, MAX_SVG_BYTES)
        return cls(data.decode("utf-8-sig"), name, **kwargs)

    @property
    def svg(self):
        """Return the source XML text before Property bindings are evaluated."""
        return self._svg

    @svg.setter
    def svg(self, value):
        template = parse_svg(value)
        for element_id, attribute in self._bindings:
            binding_target(template, element_id, attribute)
        self._svg, self._template = value, template
        self.clear_cache()

    @property
    def size(self):
        """Return the source raster viewport in pixels."""
        return self._size

    @size.setter
    def size(self, value):
        self._size = validate_pixel_size(value)
        self.clear_cache()

    @property
    def source_size(self):
        """Return the fixed source rectangle used for anchor resolution."""
        return self.size

    @property
    def font_files(self):
        """Return the explicit font paths, reread and hashed at render time."""
        return self._font_files

    @font_files.setter
    def font_files(self, value):
        if not isinstance(value, (list, tuple)) or len(value) > 8:
            raise ValueError(
                "font_files must be a list/tuple of at most eight local paths"
            )
        # Validate immediately; actual render reads its own byte snapshot.
        backend.font_snapshots(value)
        paths = tuple(Path(path).expanduser().absolute() for path in value)
        self._font_files = paths
        self._font_key, self._faces = None, ()
        self.clear_cache()

    @property
    def bindings(self):
        """Return a read-only mapping of (element id, attribute) to Property."""
        return MappingProxyType(self._bindings)

    def bind(self, element_id, attribute, value):
        """Bind an element attribute (or leaf ``text``) and return its Property."""
        binding_target(self._template, element_id, attribute)
        prop = value if isinstance(value, Property) else Property(value)
        if prop.value_type not in {"float", "color", "enum"}:
            raise TypeError(
                "SVG bindings require float, color or string enum Properties"
            )
        self._bindings[element_id, attribute] = prop
        self.clear_cache()
        return prop

    def unbind(self, element_id, attribute):
        """Remove a binding, restoring the value from the source SVG."""
        del self._bindings[element_id, attribute]
        self.clear_cache()

    def clear_cache(self):
        """Release all retained source buffers without changing any Properties."""
        self._cache.clear()
        self._cache_bytes = 0

    @property
    def cache_info(self):
        """Return current buffer count/bytes and the immutable retention limits."""
        return {
            "entries": len(self._cache),
            "bytes": self._cache_bytes,
            "max_entries": self._max_entries,
            "max_bytes": self._max_bytes,
        }

    def source_buffer(self, t, context=None):
        """Evaluate the current document and reuse only identical raster inputs."""
        time = finite_real(t, "t")
        svg, root = evaluate_svg(
            self._template,
            self._bindings,
            self.size,
            time,
            context,
            self.expression_bindings,
        )
        snapshots = backend.font_snapshots(self.font_files)
        font_key = tuple(digest for digest, _ in snapshots)
        if font_key != self._font_key:
            faces = backend.font_faces(snapshots)
            self._font_key, self._faces = font_key, faces
            self.clear_cache()
        family = backend.validate_text(root, self._faces)
        key = (
            backend.RESVG_VERSION,
            self.size,
            hashlib.sha256(svg.encode("utf-8")).digest(),
            font_key,
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        buffer = backend.rasterize(svg, self.size, snapshots, family)
        byte_count = self.size[0] * self.size[1] * 4 * 4
        if self._max_entries and byte_count <= self._max_bytes:
            while self._cache and (
                len(self._cache) >= self._max_entries
                or self._cache_bytes + byte_count > self._max_bytes
            ):
                self._cache.popitem(last=False)
                self._cache_bytes -= byte_count
            self._cache[key] = buffer
            self._cache_bytes += byte_count
        return buffer
