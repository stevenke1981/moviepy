"""Composition audio timing, switches, levels and real-reader regressions."""

import wave

import numpy as np

import pytest

from moviepy import AudioClip, AudioFileClip, ColorClip
from moviepy.ae import Composition, Expression, RenderContext, Renderer, Transform
from moviepy.audio.AudioClip import CompositeAudioClip


def _source(duration=2, channels=1, fps=8000):
    """Make a continuous signal whose timestamps and channels are observable."""

    def frame(t):
        times = np.asarray(t)
        first = 0.1 + times * 0.05
        if channels == 1:
            return first
        return np.stack((first, -0.2 + times * 0.025), axis=-1)

    return AudioClip(frame, duration=duration, fps=fps)


def _video(audio, duration=2):
    """Attach sound to deliberately low-frame-rate visual footage."""
    return (
        ColorClip((2, 2), (60, 90, 120), duration=duration)
        .with_fps(3)
        .with_audio(audio)
    )


def test_automatic_audio_exposes_real_samples_and_composite_type():
    comp = Composition(size=(2, 2), duration=2)
    source = _source()
    comp.add_clip(_video(source))
    audio = comp.audio
    assert isinstance(audio, CompositeAudioClip)
    assert audio.duration == comp.duration
    times = np.array([0, 0.13337, 0.71591, 1.41238])
    expected = np.array([[source.get_frame(float(t))] for t in times])
    np.testing.assert_allclose(audio.get_frame(times), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        audio.get_frame(times[1]), expected[1], rtol=0, atol=1e-14
    )


def test_audio_switch_is_independent_of_visual_switch_and_opacity():
    comp = Composition(size=(2, 2), duration=2)
    layer = comp.add_clip(
        _video(_source()), enabled=False, transform=Transform(opacity=0)
    )
    assert comp.audio is not None
    np.testing.assert_allclose(comp.audio.get_frame(0.5), [0.125])
    layer.audio_enabled = False
    audio = comp.audio
    assert audio is None or not np.any(audio.get_frame(0.5))
    layer.audio_enabled = True
    np.testing.assert_allclose(comp.audio.get_frame(0.5), [0.125])


def test_offset_trim_and_source_audio_duration_are_half_open():
    comp = Composition(size=(2, 2), duration=3)
    comp.add_clip(
        _video(_source(duration=0.75), duration=2),
        start_time=0.5,
        in_point=0.75,
        out_point=2.25,
    )
    assert comp.audio is not None
    times = np.array([-0.1, 0.5, 0.75, 1.0, 1.25, 2.25, 3.0])
    expected = np.array([[0], [0], [0.1125], [0.125], [0], [0], [0]])
    np.testing.assert_allclose(
        comp.audio.get_frame(times), expected, rtol=0, atol=1e-14
    )


def test_db_levels_are_animated_on_layer_clock_and_mix_without_clipping():
    from moviepy.ae.audio import db_to_amplitude

    np.testing.assert_allclose(db_to_amplitude(-6), 10 ** (-6 / 20), rtol=0, atol=1e-14)
    comp = Composition(size=(2, 2), duration=2)
    first = comp.add_clip(
        _video(_source()), start_time=0.25, audio_levels=[(0, 0), (1, -20)]
    )
    first.time_remap = 0.5
    comp.add_clip(_video(AudioClip(lambda t: 1.2, duration=2, fps=8000)))
    assert comp.audio is not None
    times = np.array([0.25, 0.75, 1.25])
    expected = (1.2 + 0.125 * 10 ** (-np.array([0, 10, 20]) / 20))[:, None]
    np.testing.assert_allclose(
        comp.audio.get_frame(times), expected, rtol=0, atol=1e-14
    )


@pytest.mark.parametrize("stretch", [50, 200, -100])
def test_stretch_uses_continuous_audio_time_and_no_video_frame_quantization(stretch):
    comp = Composition(size=(2, 2), duration=3)
    source = _source()
    layer = comp.add_clip(_video(source), stretch=stretch, start_time=0.25)
    assert comp.audio is not None
    times = np.array([0.32117, 0.40733, 0.71119])
    expected = np.array(
        [[source.get_frame(layer.media_time(float(t), comp.context))] for t in times]
    )
    np.testing.assert_allclose(
        comp.audio.get_frame(times), expected, rtol=0, atol=1e-14
    )


@pytest.mark.parametrize("operation", ["with_start", "with_duration", "subclipped"])
def test_moviepy_derived_operations_keep_audio_time_mapping(operation):
    comp = Composition(size=(2, 2), duration=2)
    comp.add_clip(_video(_source()))
    times = np.array([0.0, 0.125, 0.499])
    if operation == "with_start":
        derived = comp.with_start(5)
        assert derived.audio.start == 5
        source_times = times
    elif operation == "with_duration":
        derived = comp.with_duration(0.5)
        assert derived.audio.duration == 0.5
        source_times = times
    else:
        derived = comp.subclipped(0.75, 1.5)
        assert derived.audio.duration == 0.75
        source_times = times + 0.75
    np.testing.assert_allclose(
        derived.audio.get_frame(times),
        (0.1 + source_times * 0.05)[:, None],
        rtol=0,
        atol=1e-14,
    )
    assert comp.duration == 2 and comp.start == 0


def test_manual_audio_and_without_audio_override_automatic_mix():
    comp = Composition(size=(2, 2), duration=2)
    comp.add_clip(_video(_source()))
    manual = AudioClip(lambda _: -0.7, duration=2, fps=8000)
    replacement = comp.with_audio(manual)
    assert replacement.audio is manual
    assert replacement.without_audio().audio is None
    assert comp.without_audio().audio is None
    np.testing.assert_allclose(comp.audio.get_frame(0.5), [0.125])
    replacement.use_layer_audio()
    np.testing.assert_allclose(replacement.audio.get_frame(0.5), [0.125])


def test_audio_solo_only_counts_sound_sources_and_respects_guide_policy():
    comp = Composition(size=(2, 2), duration=2)
    comp.add_clip(_video(AudioClip(lambda _: 0.1, duration=2, fps=8000)))
    second = comp.add_clip(_video(AudioClip(lambda _: 0.2, duration=2, fps=8000)))
    comp.add_solid(solo=True)
    np.testing.assert_allclose(comp.audio.get_frame(0.5), [0.3])
    second.solo = True
    np.testing.assert_allclose(comp.audio.get_frame(0.5), [0.2])
    second.audio_enabled = False
    np.testing.assert_array_equal(comp.audio.get_frame(0.5), [0])
    second.audio_enabled = True
    second.guide = True
    np.testing.assert_allclose(comp.audio.get_frame(0.5), [0.1])
    comp.renderer = Renderer(include_guides=True)
    np.testing.assert_allclose(comp.audio.get_frame(0.5), [0.2])


def test_mono_stereo_mixing_preserves_channel_shapes_and_random_order():
    comp = Composition(size=(2, 2), duration=2)
    mono, stereo = _source(), _source(channels=2)
    comp.add_clip(_video(mono))
    comp.add_clip(_video(stereo))
    audio = comp.audio
    times = np.array([0.7017, 0.0019, 1.3013, 0.0019, 0.5017])
    expected = np.array(
        [stereo.get_frame(float(t)) + mono.get_frame(float(t)) for t in times]
    )
    assert audio.nchannels == 2
    np.testing.assert_allclose(audio.get_frame(times), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        np.array([audio.get_frame(float(t)) for t in times]),
        expected,
        rtol=0,
        atol=1e-14,
    )
    assert audio.get_frame(np.array([])).shape == (0, 2)


def test_mapping_and_levels_expressions_use_composition_fps_and_seed():
    comp = Composition(
        size=(2, 2), duration=2, fps=10, context=RenderContext(fps=30, rng_seed=17)
    )
    layer = comp.add_clip(
        _video(_source()),
        time_remap=Expression("framesToTime(1)"),
        audio_levels=Expression("timeToFrames(time)"),
    )
    times = np.array([0.12, 0.29, 0.41])
    expected = (0.105 * 10 ** (np.floor(times * 10) / 20))[:, None]
    actual = comp.audio.get_frame(times)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    layer.frame_blending = "frame_mix"
    layer.motion_blur = True
    comp.motion_blur = True
    np.testing.assert_array_equal(comp.audio.get_frame(times), actual)


def test_legal_source_times_are_sorted_and_deduplicated_without_frame_rounding():
    requests = []

    def frame(time):
        assert np.ndim(time) == 0
        assert 0 <= time < 1
        requests.append(float(time))
        return 0.1 + float(time) * 0.05

    source = AudioClip(frame, duration=1, fps=8000)
    comp = Composition(size=(2, 2), duration=2)
    comp.add_clip(_video(source), time_remap=lambda t: t)
    audio = comp.audio
    requests.clear()
    times = np.array([0.80003125, -0.1, 0.2000625, 0.80003125, 1, 2])
    expected = np.array([[0.1400015625], [0], [0.110003125], [0.1400015625], [0], [0]])
    np.testing.assert_allclose(audio.get_frame(times), expected, rtol=0, atol=1e-14)
    assert requests == [0.2000625, 0.80003125]


@pytest.fixture
def real_audio(tmp_path):
    """Write known stereo PCM, independent of MoviePy's audio decoder."""
    fps = 8000
    indices = np.arange(fps, dtype=np.int32)
    pcm = np.column_stack(
        ((indices * 37) % 24001 - 12000, (indices * 53) % 20001 - 10000)
    ).astype("<i2")
    path = tmp_path / "source.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(fps)
        stream.writeframes(pcm.tobytes())
    return path, fps, pcm.astype(np.float64) / 32768


@pytest.mark.parametrize("mapping", ["forward", "reverse", "oscillating", "freeze"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_real_file_random_reads_keep_scalar_semantics_and_wrapped_effects(
    real_audio, mapping, wrapped
):
    path, fps, pcm = real_audio
    with AudioFileClip(path, fps=fps, buffersize=64) as original:
        source = (
            original.transform(lambda get, time: get(time) * 0.25)
            if wrapped
            else original
        )
        comp = Composition(size=(2, 2), duration=1.25)
        layer = comp.add_clip(
            _video(source, duration=1), stretch=-100 if mapping == "reverse" else 100
        )
        if mapping == "oscillating":
            layer.time_remap = lambda t: 0.5 + 0.45 * np.sin(19 * t)
        elif mapping == "freeze":
            layer.time_remap = 0.31253125
        audio = comp.audio
        times = np.array(
            [0.813371, 0, 0.10234, 0.99975, 0.813371, 0.5000625, -0.1, 1.0]
        )
        expected = np.zeros((len(times), 2))
        for index, time in enumerate(times):
            if not 0 <= time < 1:
                continue
            source_time = layer.media_time(float(time), comp.context)
            if mapping == "reverse" and source_time == 1:
                source_time = (fps - 1) / fps
            if 0 <= source_time < 1:
                expected[index] = pcm[int(source_time * fps)] * (0.25 if wrapped else 1)
        np.testing.assert_array_equal(audio.get_frame(times), expected)
        np.testing.assert_array_equal(
            np.array([audio.get_frame(float(t)) for t in times]), expected
        )
        audio.close()
        np.testing.assert_array_equal(original.get_frame(0.5), pcm[4000])


def test_reverse_endpoint_only_applies_to_aligned_unremapped_sources():
    comp = Composition(size=(2, 2), duration=2)
    short = comp.add_clip(_video(_source(duration=1), duration=2), stretch=-100)
    np.testing.assert_array_equal(
        comp.audio.get_frame(np.array([0, 0.5, 1])), [[0], [0], [0]]
    )
    np.testing.assert_allclose(comp.audio.get_frame(1.5), [0.125])
    short.time_remap = 1
    np.testing.assert_array_equal(
        comp.audio.get_frame(np.array([0, 0.5, 1.5])), [[0], [0], [0]]
    )
    short.time_remap = -0.1
    np.testing.assert_array_equal(
        comp.audio.get_frame(np.array([0, 0.5, 1.5])), [[0], [0], [0]]
    )


def test_source_membership_and_property_changes_are_visible_on_fresh_audio_access():
    comp = Composition(size=(2, 2), duration=2)
    assert comp.audio is None
    layer = comp.add_clip(_video(_source()))
    old_audio = comp.audio
    layer.audio_levels = -20
    np.testing.assert_allclose(old_audio.get_frame(0.5), [0.0125])
    comp.remove_layer(layer)
    assert comp.audio is None
    comp.add_clip(_video(_source(channels=2)))
    assert comp.audio.nchannels == 2


@pytest.mark.parametrize(
    "field,bad",
    [
        ("fps", 0),
        ("fps", -1),
        ("fps", float("inf")),
        ("duration", -1),
        ("duration", float("nan")),
        ("nchannels", 0),
        ("nchannels", True),
    ],
)
def test_invalid_source_audio_metadata_is_rejected(field, bad):
    source = _source()
    setattr(source, field, bad)
    comp = Composition(size=(2, 2), duration=2)
    comp.add_clip(_video(source))
    with pytest.raises((TypeError, ValueError), match="source audio"):
        _ = comp.audio


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "loud", 100000])
def test_invalid_db_values_are_rejected_without_silent_clamping(value):
    from moviepy.ae.audio import db_to_amplitude

    with pytest.raises((TypeError, ValueError), match="audio_levels"):
        db_to_amplitude(value)


def test_open_procedural_audio_uses_default_output_rate_and_comp_window():
    source = AudioClip(lambda t: 0.1 + t * 0.01)
    comp = Composition(size=(2, 2), duration=1.5)
    comp.add_clip(_video(source))
    audio = comp.audio
    assert audio.fps == 44100
    np.testing.assert_allclose(audio.get_frame(1.4), [0.114])
    np.testing.assert_array_equal(audio.get_frame(1.5), [0])


def test_matching_multichannel_sources_and_mono_broadcast_are_supported():
    comp = Composition(size=(2, 2), duration=2)
    channels = np.arange(6) / 20
    comp.add_clip(_video(AudioClip(lambda _: channels, duration=2, fps=8000)))
    comp.add_clip(_video(AudioClip(lambda _: 0.1, duration=2, fps=8000)))
    audio = comp.audio
    assert audio.nchannels == 6
    np.testing.assert_allclose(
        audio.get_frame(np.array([0.1, 0.2])), np.tile(channels + 0.1, (2, 1))
    )
    comp.add_clip(_video(_source(channels=2)))
    with pytest.raises(ValueError, match="matching non-mono channel counts"):
        _ = comp.audio


def test_reader_errors_propagate_and_nonfinite_audio_is_rejected():
    comp = Composition(size=(2, 2), duration=2)
    source = _source()
    layer = comp.add_clip(_video(source))

    def failed(_):
        raise OSError("source reader unavailable")

    source.frame_function = failed
    with pytest.raises(OSError, match="source reader unavailable"):
        comp.audio.get_frame(0.5)
    source.frame_function = lambda _: float("nan")
    with pytest.raises(ValueError, match="samples must be finite"):
        comp.audio.get_frame(0.5)
    source.frame_function = lambda _: 1e308
    layer.audio_levels = 20
    with pytest.raises(ValueError, match="must remain finite"):
        comp.audio.get_frame(0.5)


def test_mix_overflow_is_rejected_instead_of_publishing_infinity():
    comp = Composition(size=(2, 2), duration=2)
    for _ in range(2):
        comp.add_clip(_video(AudioClip(lambda _: 1e308, duration=2, fps=8000)))
    with pytest.raises(ValueError, match="mixed audio.*finite"):
        comp.audio.get_frame(np.array([0.1, 0.2]))


def test_real_wav_export_matches_animated_db_and_offset_samples(tmp_path):
    fps = 8000
    tone = AudioClip(lambda t: 0.2 * np.sin(2 * np.pi * 125 * t), duration=1, fps=fps)
    comp = Composition(size=(2, 2), duration=1.5)
    comp.add_clip(
        _video(tone, duration=1),
        start_time=0.125,
        in_point=0.125,
        audio_levels=[(0, 0), (1, -20)],
    )
    path = tmp_path / "mixed.wav"
    comp.audio.write_audiofile(
        path, fps=fps, nbytes=2, buffersize=128, codec="pcm_s16le", logger=None
    )
    with wave.open(str(path), "rb") as stream:
        assert stream.getnchannels() == 1
        assert stream.getframerate() == fps
        assert stream.getnframes() == int(comp.duration * fps)
        decoded = (
            np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(
                np.float64
            )
            / 32768
        )
    local = np.arange(len(decoded)) / fps - 0.125
    expected = np.zeros_like(local)
    valid = (local >= 0) & (local < 1)
    expected[valid] = (
        0.2 * np.sin(2 * np.pi * 125 * local[valid]) * 10 ** (-local[valid])
    )
    np.testing.assert_allclose(decoded, expected, rtol=0, atol=1 / 32768)
    error_db = 20 * np.log10(
        np.sqrt(np.mean(decoded**2)) / np.sqrt(np.mean(expected**2))
    )
    assert abs(error_db) < 0.5
