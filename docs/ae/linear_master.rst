Linear compositing and 16-bit masters
=====================================

This opt-in path removes two avoidable losses: blending encoded sRGB values,
and reducing a floating-point composition to 8 bits before intermediate export.
Existing projects keep their default sRGB working space and pixel results.

.. code-block:: python

   from moviepy.ae import Composition, RenderContext, Transform, write_master

   comp = Composition(
       size=(1920, 1080), fps=24, duration=4,
       context=RenderContext(working_space="linear"),
   )
   comp.add_solid(color=(255, 255, 255), transform=Transform(opacity=50))
   # White at 50% coverage over black displays near code 188, not code 128.
   master = write_master(comp, "output/shot_master", audio_path="sound.wav")
   print(master.path)  # output/shot_master/master.mkv

The output directory must be new. Audio is optional. If supplied, it must be
mono/stereo 16/24/32-bit PCM WAV with exactly the exported frame duration.
The writer copies PCM samples without resampling. It leaves source clips open.
Apply timing edits to AE layers before export: wrappers such as MoviePy's
``subclipped`` change a clip frame function, not the underlying AE layer timeline,
and are rejected instead of silently exporting the untrimmed composition.

Pixel contract
--------------

At the source boundary, footage and authored color controls are interpreted as
sRGB. A linear project converts straight RGB to linear sRGB, then premultiplies
by coverage. Layer filters, transforms, normal source-over and temporal reads
operate on float32 premultiplied pixels. Alpha is never gamma-encoded. Nested
compositions inherit the active context. A linear luma matte does not decode
pixels twice. Classic MoviePy effects are given floating sRGB codes and converted
back afterward; individual classic effects can still quantize or clip internally.

``linear`` means sRGB/Rec.709 primaries with D65 white, after the sRGB transfer
function is decoded. It does not select an exposure calibration, ACES, wide
gamut, or an OCIO transform. Native scene-linear footage can retain HDR values
in this encoding. OCIO labels are rejected at this conversion boundary.
RGB transfer functions use the signed extension described by
`W3C CSS Color 4 <https://www.w3.org/TR/css-color-4/#color-conversion-code>`_.

The final display boundary flattens the background in the selected working
space, then encodes straight sRGB. Transparent compositions retain coverage.
``Composition.get_frame`` remains uint8 for MoviePy compatibility. Use
``render_buffer`` with ``moviepy.ae.color.display_rgba`` for floating display
values; ``Buffer.to_uint8_rgb`` still exports raw working codes, as before.

``write_master`` bypasses ``get_frame`` and rounds once to 16-bit RGBA. It uses
the installed FFmpeg FFV1 encoder in Matroska, with ``gbrap16le`` pixels, full
range RGB, sRGB transfer and BT.709 primaries tags. There is no chroma subsampling.
The manifest records each packed RGBA64LE frame SHA-256, PCM SHA-256, sample and
frame counts, timing, the command and the number of clipped RGB channels.
Out-of-range display values are clipped at this SDR export boundary.
See the `FFmpeg FFV1 options <https://ffmpeg.org/ffmpeg-codecs.html#ffv1>`_.

The writer uses two encoder threads. On failure it retains the partial file
and log; the final master and manifest are published only after encoder success.
Master export requires full rendering resolution. FFV1 is an interchange/archive
master: some editing applications need a separately validated delivery transcode.

Reproducible comparison
-----------------------

.. code-block:: console

   python -m examples.ae_cinema_quality output/eclipse --godot PATH --font FONT_PATH
   python -m pytest tests/ae/test_linear_master.py -q

ECLIPSE is a four-second, 1280x544, 24 fps original lighting study. The Godot
geometry, light settings, object animation, gradient plate and synthetic sound
are generated locally. There is a small 2D push, with a fixed 3D camera.
Both versions use the exact same 96 Godot frames. The ``srgb`` version uses
the existing encoded-space renderer and its uint8 boundary; ``linear`` keeps
floating-point color until the RGBA16 master. A separate transparent foreground
master exercises alpha. The font stays at the explicit local path.

Each run saves settings, source hashes, all masters, fixed-setting SDR H.264/AAC
viewing copies and frame previews. ``--reuse-godot DIRECTORY`` can reuse the
same scene after validation, without re-running the GPU engine. It does not
download models or assets. The viewing copies are 8-bit and cannot demonstrate
all precision retained in the masters; evaluate decoded master pixels too.

Native scene-linear Godot footage
---------------------------------

Godot rendering can additionally retain its native pre-postprocessing color
buffer. The default PNG output and ``to_clip`` behavior remain SDR:

.. code-block:: python

   from moviepy.ae.three_d import render_godot_scene

   shot = render_godot_scene(scene, "output/shot", executable=godot_path,
                             linear_output=True)
   comp = Composition(size=scene["size"], fps=scene["fps"],
                      duration=scene["duration"],
                      context=RenderContext(working_space="linear"))
   comp.add_layer(shot.to_linear_layer())

This opt-in route copies the Forward+ RGBA16F color buffer in
``POST_TRANSPARENT``, before Godot's built-in glow, tone mapping and output
encoding. It saves native half-float EXRs in ``linear_raw`` and losslessly
compressed float32 NPZ files after the same 2x area resolve. RGB is already
premultiplied; coverage is averaged with RGB, without a transfer curve or
clipping. Transparent linear captures use a black clear color, so translucent
RGB is not contaminated by a nonblack background before capture. Hidden RGB
is also discarded before area filtering. The color encoding is linear
sRGB/BT.709 with D65, not ACES or PQ.
The relative scene units are not an absolute luminance calibration.

``to_linear_layer`` reads the float32 sequence directly as immutable Buffers.
Arbitrary seeks, endpoint holds and reverse stretch use the layer's source
clock. Apply timing and masks to the layer. The existing ``to_clip`` deliberately
continues to read the SDR PNG sequence. Partial effect blending also preserves
the input working-space label.

The installed FFmpeg EXR decoder reads floating channels without a display
conversion; no additional Python codec is required. The compositor API is
experimental, the capture synchronizes GPU readback, and enabling it adds
render time, storage and memory costs. The renderer must provide native
RGBA16F; an unavailable capture fails instead of upgrading PNGs and calling
them HDR. Godot's postprocessing is not included in these source EXRs.

The implementation follows the official
`CompositorEffect stage contract <https://docs.godotengine.org/en/stable/classes/class_compositoreffect.html>`_,
`RenderingDevice texture readback <https://docs.godotengine.org/en/stable/classes/class_renderingdevice.html#class-renderingdevice-method-texture-get-data>`_
and `Image EXR output <https://docs.godotengine.org/en/stable/classes/class_image.html#class-image-method-save-exr>`_.

``python -m examples.ae_linear_source OUTPUT --godot PATH --font FONT_PATH``
renders a separate two-second highlight study. Its stronger emission is
explicitly separate from the unchanged ECLIPSE performance comparison. The
example uses -1 stop exposure and a peak-channel Reinhard gain before sRGB
encoding. Its MP4 and FFV1 master are SDR; the native EXR/NPZ sources retain the
HDR values. This example display map is not a calibrated production grade.

Limits and performance
----------------------

Default Godot PNG footage is tone-mapped SDR RGBA8. Linearizing it cannot
recover clipped highlights or missing source precision. Existing blend modes
and color effects retain their documented clipping rules and are not all
physically based. Operations performed inside an imported source clip are not
retroactively linearized. Accurate camera motion blur, lens distortion, depth
passes, production assets, calibrated grading and an HDR output transform are
not added by this round.

The existing cubic filter retains signed/HDR RGB overshoot while clipping
alpha. Its negative lobes can create bright edge ringing, including in a linear
project. Dedicated tests preserve this documented behavior. The comparison
explicitly uses ``Transform(interpolation="linear")`` for positive-weight,
non-ringing edge resampling; higher-order filters are not automatically safer.

Color conversion skips fully transparent RGB and avoids a redundant
unpremultiply for opaque buffers. Piecewise transfer arithmetic retains its
float64 intermediates and final float32 rounding, with fewer temporary arrays.
No cross-frame cache, lower-precision lookup table or resolution reduction is
used. Performance depends on source coverage and color distribution; measure
the actual scene along with startup, peak memory and pixel equivalence.

These numerical and export improvements support better compositing. They do
not by themselves establish a finished Hollywood production standard. A
calibrated display transform and artist-reviewed production assets remain
necessary for a finished film workflow.
