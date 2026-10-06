Time controls and shutter sampling
=================================

Footage remapping
-----------------

``AVLayer.time_remap`` is an optional floating-point ``Property``. It maps the
existing layer animation clock to source seconds. Transform, layer-mask and
effect animation keep their original clock::

    from moviepy.ae import Composition, Property

    comp = Composition(size=clip.size, duration=clip.duration)
    layer = comp.add_clip(clip)
    layer.time_remap = Property(0.25)  # hold footage, keep layer animation
    layer.time_remap = Property(keyframes=[(0, 0), (1, 0.5)])
    layer.time_remap = None           # restore normal footage timing

The layer window stays half open, ``in_point <= t < out_point``. Remapping does
not extend the out point automatically. Video reads outside source duration
hold readable endpoint frames. ``source_buffer(t)`` remains a raw source-time
read; ``sampled_source(local_t)`` includes the new footage controls.
``media_time(comp_t)`` returns continuous, unclamped source seconds for audio.
Temporal effects re-evaluate the remap at each requested layer time.

Frame interpolation
-------------------

``layer.frame_blending = "frame_mix"`` averages neighboring source frames in
premultiplied RGBA and the active working color space. The source must declare
a finite positive frame rate, for example with ``clip.with_fps(24)``. Source
RGB and its clip mask use the same timestamps. Signed/HDR RGB is retained.
``"off"`` is the default and preserves existing procedural-source sampling.

``"pixel_motion"`` raises ``NotImplementedError``; optical-flow interpolation
has not been implemented. These controls currently apply to AVLayer, including
subclasses that provide their own raw Buffer source. CompLayer and the separate
native linear-sequence layer do not yet expose them. This is not complete WS-07
or verified Adobe pixel parity.

Motion blur
-----------

Enable both the composition and the desired layers::

    from moviepy.ae import RenderContext, Shutter

    comp.motion_blur = True
    comp.context = RenderContext(
        shutter=Shutter(angle=180, phase=-90,
                        samples_min=8, samples_max=64, adaptive=True),
        working_space="linear",
    )
    layer.motion_blur = True

Compositions default to motion blur off. Supplying an explicit shutter in the
initial Composition context also opts in; an explicit ``motion_blur=False``
overrides that choice. Standalone ``layer.render`` uses an explicitly supplied
context shutter. ``RenderContext`` continues to transport opaque references,
but rendering with motion blur validates the shutter parameters.

The exposure interval starts at ``t + phase/(360*fps)`` and ends at
``t + (phase+angle)/(360*fps)``. Deterministic midpoint samples average
premultiplied pixels with a float64 accumulator and a float32 result. Angle zero
uses the original instantaneous render. Angles are within 0..720 degrees;
sample bounds are integers within 1..256, with maximum at least minimum.

Fixed sampling uses ``samples_min``. Adaptive sampling uses transform travel to
increase that count up to ``samples_max``; effects, masks, track mattes and
unknown source types use the maximum. Proven static solids and ordinary AV
transforms without those dependencies can use one sample. This is bounded
numerical integration, not a guarantee for arbitrary
high-frequency animation. A mapping such as ``{"angle": 180, "phase": -90,
"samples": 16}`` requests a fixed sample count.

Transforms, parent transforms, masks and effect parameters are evaluated at
subframe times. Ordinary AV footage stays at the center footage time; layer
motion blur does not synthesize motion inside the source video. Source requests
by temporal effects retain their relative offsets. Adjustment effects also
sample the exposure and use the current backdrop; their spatial dependencies
conservatively render the whole composition to preserve ROI equivalence.
Track mattes apply inside each target shutter sample, before averaging, so a
matte moving with its target retains its coverage. Their own motion-blur switch
also works when the target is instantaneous. Nested compositions retain their
own shutter settings while inheriting the parent's working space and scale.
Entering a child composition also starts a new exposure clock in child time;
parent footage offsets are not reused in that different time domain.

The center time must be inside the layer window. Subsamples outside that window
contribute transparent coverage without renormalization; inactive adjustment
subsamples retain the backdrop. Existing defaults take the original render path.

Adobe's `animation-tools documentation
<https://helpx.adobe.com/after-effects/desktop/animate-in-after-effects/assorted-animation-tools/assorted-animation-tools.html>`_
defines shutter angle and phase and distinguishes layer animation from internal
footage motion. The interval above follows from those definitions. The library's
sample placement, boundary behavior and adaptive heuristic are explicit local
contracts, not an Adobe-rendered equivalence claim.
