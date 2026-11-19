Composition layer audio
=======================

``Composition.audio`` automatically returns a MoviePy ``CompositeAudioClip``
for sound-bearing ``AVLayer`` sources, or ``None`` when there are none. The
ordinary ``write_videofile`` path includes this mix::

    from moviepy.ae import Composition

    comp = Composition(size=clip.size, duration=4)
    layer = comp.add_clip(clip, start_time=0.5, in_point=0.75, out_point=3.5)
    layer.audio_levels = [(0, -12), (1, 0)]
    comp.write_videofile("output.mp4", codec="libx264", audio_codec="aac")

Layer controls
--------------

``audio_enabled`` defaults to true and controls exact mute. It is independent
of the visual ``enabled`` switch, opacity, masks and motion blur. Scalar
``audio_levels`` uses the existing floating-point Property implementation;
zero dB is unity and ``10 ** (dB / 20)`` applies equally to all channels.
Animated levels follow the layer animation clock, even when footage is frozen.
Nonfinite levels or overflow are rejected. Floating-point mixing adds samples
without clipping them; the chosen output format may limit their final range.

The layer's half-open ``in_point <= t < out_point`` interval gates audio.
``start_time``, ``stretch`` and ``time_remap`` use the same continuous source
mapping as video, without snapping audio to the video frame rate. Source audio
outside its own duration is silent. Negative stretch starts on the last audio
sample when the source audio and video durations agree. Frame blending does not
change audio. Retiming changes pitch; no pitch-preserving processing is provided.
A constant remap holds one sample value, which can be nonzero DC.

Solo considers included AV layers with an audio source; a silent SolidLayer or
NullLayer does not suppress sound. A muted solo audio layer still suppresses
other audio layers. Guide audio follows ``renderer.include_guides``. Mono is
broadcast when mixed with multichannel audio; differing non-mono channel counts
are rejected because their channel layouts are not defined here.

MoviePy integration and ownership
--------------------------------

Each access to ``comp.audio`` captures the current source membership. Layer
timing, mute, levels and solo remain live. Fetch the property again after adding
or removing layers or replacing a source clip. Audio readers are borrowed; the
mix never closes them. Wrapped source effects are preserved through the source's
public ``get_frame`` method. Random, reversed and repeated requests keep the
source reader's scalar sample semantics.

MoviePy's ``with_audio`` and ``without_audio`` create explicit overrides.
``comp.use_layer_audio()`` restores automatic mixing. Derived operations such as
``subclipped``, ``with_start`` and ``with_duration`` snapshot the original mix
before transforming its timeline, so duration changes do not prematurely trim
the source audio. Layer objects remain shared, as with the existing video API.

Only AVLayer source audio is mixed in this implementation. Nested CompLayer
audio, channel routing, pan, audio effects and pitch preservation are outside
this increment. This is not a claim of complete WS-25 or Adobe audio parity.
