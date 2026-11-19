Editable SVG layers
===================

``SVGLayer`` is a normal AE layer. Its source is a validated SVG document whose
element attributes can be driven by ``Property``, keyframes or expressions.
The result follows the existing source -> working-space conversion -> masks ->
effects -> transform -> matte -> blend pipeline. ``Composition.add_svg`` adds
the layer directly; there is no intermediate video or separate scene format.

Install the optional backend into the environment that runs MoviePy::

    python -m pip install --only-binary=:all: resvg-py==0.2.6 fonttools==4.59.2

Alternatively, install this checkout with ``python -m pip install ".[svg]"``.
The core MoviePy dependencies are unchanged. The versions are pinned to retain
Python 3.9 wheels on Windows and Linux. See ``THIRD_PARTY.md`` for provenance.

Create and edit
---------------

.. code-block:: python

    from moviepy.ae import Composition, Property

    svg = '''<svg viewBox="0 0 160 80">
      <rect id="bar" x="10" y="20" width="20" height="40" fill="red"/>
    </svg>'''
    comp = Composition(size=(320, 180), fps=24, duration=2, transparent=True)
    layer = comp.add_svg(svg, "Progress", size=(160, 80))
    width = layer.bind("bar", "width", Property(keyframes=[(0, 20), (2, 140)]))
    layer.bind("bar", "fill", Property((0.2, 0.8, 1), value_type="color"))
    frame = comp.get_frame(1)
    alpha = comp.mask.get_frame(1)
    width.set_keyframes([(0, 40), (2, 100)])  # Same-time requests see this edit.

``bind`` returns the actual Property. Numeric attributes interpolate normally;
string attributes and leaf ``text``/``tspan`` content use enum Properties with
hold keyframes. Paint colors accept RGB/RGBA color Properties in 0..1.
``unbind(id, attribute)`` restores the document's original value. Assigning
``layer.svg`` replaces the document atomically and preserves valid bindings.
Missing bound ids reject the replacement. ``bindings`` exposes a read-only map.

Bindings use layer-local time, so ``start_time`` and ``stretch`` retime them
together with transforms and effects. Expressions receive the normal context,
layer index and deterministic layer identity. SVG-native SMIL/CSS animations
are unsupported; animation belongs to these AE Properties.

``size`` is the fixed raster viewport, not an animated property. ``viewBox`` and
``preserveAspectRatio`` control artwork fitting inside it. The viewport overrides
root SVG width/height; use ``layer.size = (w, h)`` to change it. Increasing layer
transform scale resamples this raster. Continuous vector rasterization at the
transformed scale is not implemented.

``SVGLayer.from_file(path, size=...)`` explicitly reads a UTF-8 file once. It
creates an editable in-memory snapshot; to reload, assign freshly read XML to
``layer.svg`` or create a new layer. A plain ``svg`` string is always XML, never
an implicit filename or URL.

Fonts and CJK
-------------

All nonempty SVG text, including Latin, requires explicit ``font_files``.
System font discovery is disabled. Supply local TTF, OTF or TTC files whose
family matches the document; if no family is specified, the first supplied
face is the default. Generic family names map to this same explicit default.
No fonts are downloaded, installed or redistributed by the layer.

fontTools checks requested family names and Unicode cmap coverage before
rendering. Missing files, absent families and missing characters (reported as
``U+XXXX``) raise an error instead of silently dropping text. This check does
not certify complex shaping, ligatures, emoji sequences or exact font-style
selection. Use a suitable CJK font for Chinese/Japanese/Korean text and visually
review typography needed for production.

Resources and supported SVG subset
----------------------------------

Supported elements are svg, g, defs, rect, circle, ellipse, line, polyline,
polygon, path, linearGradient, radialGradient, stop, clipPath, mask, text,
tspan, title and desc. Presentation/geometry attributes and a restricted
inline presentation ``style`` are supported. Local ``url(#id)`` paint, clip
and mask references work; unknown attributes/elements fail explicitly.

Scripts, event handlers, external/embedded images, href/use, foreignObject,
stylesheets, CSS escapes/imports, SVG animation, DTDs, entities and processing
instructions are rejected before resvg sees the document. SVG filters are
unsupported; use AE effects. No resources directory or implicit file/network
lookup is exposed. Convert unsupported exported artwork into this subset first.

The document limit is 2 MiB, 10,000 nodes and depth 64. Raster allocation uses
the existing AE pixel budget. Up to eight explicitly selected local font files
are allowed, at most 32 MiB each and 64 MiB combined. These limits and validation
reduce accidental or hostile resource use; this in-process native renderer is
not an OS sandbox for arbitrary untrusted files.

Color, alpha and cache
----------------------

resvg 0.46.0 renders static SDR SVG to PNG. tiny-skia's internal pixels are
premultiplied, while its PNG output is straight-alpha RGBA. The adapter decodes
PNG with Pillow and premultiplies exactly once into an sRGB ``Buffer``.
The ordinary layer pipeline converts that source to linear sRGB when requested.
It does not clip or reinterpret other native linear/HDR layers. SVG input itself
does not provide HDR, wide-gamut or native animated SVG support.

The source LRU defaults to eight buffers and 64 MiB. Both limits apply; an
oversized single frame renders without being retained. Set ``cache_entries=0``
to disable retention. ``cache_info`` reports entries/bytes/limits and
``clear_cache()`` releases retained buffers.

Every source request reevaluates Properties. Keys include serialized evaluated
SVG, raster size, backend version and SHA-256 of the actual font bytes. Font
content is read on each request, so even same-size/same-mtime replacements
invalidate the cache. The native call reads temporary copies of those exact
bytes. Changing an effect or transform uses the existing layer pipeline and
does not need to invalidate source pixels. Cache hits still pay for Property,
XML and font validation; distinct animated artwork normally needs a new raster.

Example and measurement
-----------------------

``examples/ae_svg.py`` builds an editable composition with geometry keyframes,
a gradient, layer mask, blur and transform. Its optional text mode accepts an
explicit font, including a suitable CJK font::

    python -m examples.ae_svg --output svg.mp4
    python -m examples.ae_svg --font /path/to/font.ttf --text "MoviePy AE" --output title.mp4
    python -m examples.ae_svg --benchmark 30

Run these module commands from the repository root to use the current checkout
even when the environment also has an older MoviePy installation.

The benchmark first checks identical output, then reports medians for paired
cached/uncached full compositions, static source rasterization and distinct
animated source samples. These are relative measurements of this small scene,
not a claim of real-time rendering or comparison with After Effects.

Run ``pytest tests/ae/test_svg_layer.py -q`` for contracts and native regressions.
The native tests are explicitly skipped when the optional backend is absent;
those skips do not establish rendering correctness.

Measured cache behavior
------------------------

On R93900X (Windows 11, Python 3.12.10), the example used a 640x360 composition,
512x256 SVG source, linear working space, mask, blur and transform. Each pair
had 40 alternating samples after three warmups, with OpenCV set to one thread
and OpenCL disabled in the measuring process. Median milliseconds were:

.. list-table:: Median milliseconds
   :header-rows: 1

   * - Scene / operation
     - No retention
     - Warm cache
     - Speedup
   * - Shapes / source
     - 8.35
     - 0.34
     - 24.72x
   * - Shapes / composition
     - 96.17
     - 88.47
     - 1.09x
   * - CJK / source
     - 59.03
     - 20.08
     - 2.94x
   * - CJK / composition
     - 150.05
     - 108.81
     - 1.38x

Here ``No retention`` means ``cache_entries=0`` with the process and filesystem
already warm; it is not a cold-start/disk benchmark. The CJK case used the local
Microsoft JhengHei TTC and the same visual scene. Source-cache hits made zero
resvg calls across three repeated composition requests; disabled retention made
three. Other composition work still runs on every request.

Distinct animated SVG samples took 8.32 ms for shapes and 58.50 ms with CJK text
at the source stage. Rehashing the supplied font bytes remains part of a cache
hit, and resvg reloads supplied fonts for each new raster. These measurements
show reuse of identical artwork; they do not make changing text/geometry free
or establish real-time rendering. The machine was not exclusively reserved;
ambient CPU snapshots around these runs were 0--0.26 percent.
