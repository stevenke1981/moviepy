Subject framing and transparent magic
=====================================

These helpers compose through the existing AE Property, Transform, Layer,
EffectStack and Composition APIs. Camera framing is a two-dimensional crop,
translation and uniform scale. It does not track a subject automatically or
create a different viewpoint, parallax, or newly visible image content.

Framing and four camera directions
---------------------------------

.. code-block:: python

    from moviepy import VideoFileClip
    from moviepy.ae import Composition, pan_zoom

    clip = VideoFileClip("authorized_plate.mp4")
    comp = Composition(size=(1280, 720), fps=24, duration=2)
    rig = pan_zoom(
        clip.size, comp.size,
        duration=47 / 24, direction="left", distance=90,
        zoom=(1.1, 1.2), overscan=1.04,
        focus=(960, 540),
    )
    layer = comp.add_clip(clip, transform=rig.to_transform())
    print(rig.sample(1).crop, rig.sample(1).clamped)
    # Close the source clip when the composition has finished using it.

``left``, ``right``, ``up`` and ``down`` describe the viewing window's movement
through source pixels. Content moves in the opposite screen direction.
``none`` produces a zoom-only push. ``distance`` is a nonnegative number of
source pixels. Default timing uses ``Ease.easy_ease()``; supply any core Ease
for another curve. Keys hold before zero and after duration. Use the final
visible frame's timestamp as duration when the move must reach its endpoint
on that frame, rather than the half-open end of a clip.

For a custom shot, construct ``CameraFraming(source_size, size, ...)`` directly.
``focus`` and ``pan`` accept vec2 Properties; ``zoom`` accepts a float Property.
Each accepts a constant, core Keyframes, a time callable or an Expression.
Renderer expression bindings, including layer index and RenderContext, pass
through the generated Transform. For example:

.. code-block:: python

    from moviepy.ae import CameraFraming, Ease, Keyframe

    rig = CameraFraming(
        (1920, 1080), (1280, 720),
        subject_bounds=(730, 280, 1190, 810),
        safe_margin=0.08, overscan=1.06,
        zoom=[
            Keyframe(0, 1.0, out_ease=Ease.easy_ease()),
            Keyframe(2, 1.25),
        ],
    )
    transform = rig.to_transform()
    rig.pan.set_keyframes([Keyframe(0, (0, 0)), Keyframe(2, (80, 0))])

Focus, pan and rectangles use source pixel centers: a 1920x1080 source extends
from (0, 0) to (1919, 1079). Rectangles contain left, top, right, bottom.
The optional static ``subject_bounds`` defaults the focus to its center and
keeps that whole box within the fractional output ``safe_margin``. A changing
manual focus can use keyframes; this helper does not detect or update a box.
Without a subject box the helper aims at the focus, then clamps to coverage.

``safe_bounds`` restricts the usable source rectangle, for example to exclude
an existing border. ``overscan >= 1`` increases the cover scale. Positive zoom
values below one clamp to cover; zero/negative animated zoom is an error.
Zoom above the safe subject limit is reduced, and pan is clamped at every
evaluation, including custom easing overshoot. Incompatible source/subject
constraints raise ValueError. The ``FramingSample`` records actual center,
scale, sampled crop and whether a clamp occurred.

Coverage includes a tiny source-coordinate float32 rounding guard. Linear
resampling avoids the negative lobes and bright ringing of cubic filtering.
The coverage guarantee applies to opaque rectangular footage at the declared
size, with no extra parent transform, rotation or geometry-changing effect.
It cannot fill transparent holes that were already in a source.

Animate the rig controls, rather than independently editing its generated
anchor, position or scale. The result is an ordinary Transform evaluated by
the existing renderer. Its derived Properties are dynamic runtime values and,
like other callable Properties, cannot be serialized to project JSON. There
is no second renderer or independently rendered camera PNG sequence.

Three Godot presets, one AE bridge
---------------------------------

``moviepy.ae.motion.render_magic`` provides three small presets:

* ``warm_aura``: a slowly opening, tilted warm-gold ring with a restrained halo.
* ``flow_particles``: three small upward streams of fixed-seed GPU particles.
* ``cast_ring``: a brief expanding ring, with an AE opacity envelope.

.. code-block:: python

    from moviepy.ae import Composition, RenderContext
    from moviepy.ae.motion import render_magic

    spell = render_magic(
        "flow_particles", "new-spell-directory",
        executable="path/to/Godot.exe",
        size=(384, 384), fps=24, frame_count=48, seed=607,
        color=(1.0, 0.58, 0.16), emission=2.0, particle_count=24,
    )
    comp = Composition(
        size=(1280, 720), fps=24, duration=2,
        context=RenderContext(working_space="linear"),
    )
    # Add authorized footage first, then place this layer outside the face area.
    layer = spell.to_layer(position=(1040, 510), blend_mode="normal")
    comp.add_layer(layer)
    layer.effects[0].radius.set_keyframes([(0, 6), (1, 12)])

The bridge reuses ``render_godot_scene`` and its packaged scene validator,
hidden Windows GPU window, actual Forward+ rendering, 2x coverage resolve,
PNG verification and native float capture. It requires the already installed
Godot 4.7+ Vulkan backend; no engine or Python dependency is installed by these
helpers. Unsupported platforms/backends fail through the existing bridge.

``build_magic_scene`` exposes the exact normalized scene for inspection.
Size, fps, frame_count, seed, color, emission, radius, particle_count and
particle_size are
baked before rendering. At least three frames are required for the default
visible envelope. FPS must divide 48000, following the existing Movie Maker
contract. Portrait scenes increase camera distance to retain the same clear
geometry margins. Unknown parameters/presets and invalid values are rejected.

After rendering, ``MagicRender.to_layer`` returns an ordinary AE Layer.
Position, scale, opacity, Glow radius and Glow intensity accept the existing
Properties/keyframes. The default opacity reaches zero on both first and last
visible frames; explicitly supplying opacity replaces that envelope. Layer
start_time, stretch, effects, masks and track mattes retain core semantics.
No stateful GPU simulation runs during random frame reads: those read only
the completed sequence. Repeatability is tested on the same engine/GPU; exact
cross-driver or cross-version particle equality is not promised.

Alpha and color
---------------

Default ``linear_output=True`` retains native pre-postprocessing RGBA16F EXR
and resolved float32 premultiplied NPZ alongside an explicit SDR PNG copy.
The layer reads the float sequence directly. RGB is not clipped to display
white during the bridge or AE Glow; alpha remains linear coverage. Godot's
full-scene postprocess glow is deliberately disabled on a transparent scene.
The halo is the existing alpha-aware AE Glow after the native source is read.
There is no black-key removal and no reinterpretation of black RGB as alpha.

Normal and Add in a linear Composition retain source HDR values. Screen uses
the core's established SDR input clamp; choosing Screen does not claim HDR
preservation. ``linear_output=False`` explicitly chooses the existing SDR
PNG/AVLayer path. Neither a PNG nor an MP4 is an HDR master. The demonstration
uses an explicit peak-channel Reinhard SDR viewing map only after compositing.

Protecting faces and subtitles
-----------------------------

Placement and protection are manual. Choose the region from the actual shot;
do not treat a focus point as face detection. Layer masks normally precede
effects, so a Glow can grow outside an earlier mask. To exclude the complete
halo, pre-compose the effect, then mask an outer CompLayer:

.. code-block:: python

    from moviepy.ae import CompLayer, Composition, Mask, RenderContext

    effects = Composition(
        size=(1280, 720), fps=24, duration=2, transparent=True,
        context=RenderContext(working_space="linear"),
    )
    effects.add_layer(layer)
    protected = CompLayer(effects, masks=[
        Mask.rect((640, 260), (300, 300), mode="subtract"),
        Mask.rect((640, 665), (1280, 110), mode="subtract"),
    ])
    # Add protected over the footage in the final composition.

For a moving face or moving framing, manually keyframe the core mask path or
apply the same appropriate transform. There is no automatic motion/occlusion
tracking. The original ECLIPSE demonstration has an object, not a person's
face; it verifies zero effect coverage and unchanged pixels inside its
declared subject region. Labels are placed outside that region.

Reproduction and checks
-----------------------

.. code-block:: console

    python -m examples.ae_camera_magic output/camera-magic --source-frames EXISTING_ECLIPSE_GODOT_DIR --godot GODOT_EXE --font FONT_FILE
    python -m pytest tests/ae/test_camera_magic.py -q

The example reuses the first 48 original 1280x544 RGBA frames from
``examples.ae_cinema_quality`` at 24 fps. It builds its plate and all comparisons
with native AE compositions. It writes a 10-second four-direction/subject-push
preview, a 6-second before/after magic preview, black/white/checkerboard views,
contact sheets, source hashes, source/region coordinates and engine evidence.
Each MP4 is 1280x720, silent, and must remain below 9 MB. Original episode media
and font files are neither changed nor copied into the repository.

Set ``MOVIEPY_AE_TEST_GODOT`` to opt into actual GPU regression renders. The
CPU suite tests directional timing, expression bindings, portrait/landscape
coverage, extreme pan/zoom, impossible boxes, retained HDR, transparent
endpoints and random frame access. The GPU test renders all three presets
and repeats the fixed-seed particle sequence for frame-by-frame equality.

Run the existing AE suite and the repository's official Black, isort and
Flake8 commands as well. Numeric tests and decoded video checks complement
actual still-image inspection; they do not establish real-episode acceptance,
face tracking, physical lens optics or a calibrated production grade.
