"""Regression tests for WS-02 review findings.

This module covers defects found during independent review passes and ensures
they stay fixed. Each test targets a specific CRITICAL, HIGH or MEDIUM finding
confirmed by the verification script.
"""

import numpy as np
import pytest

from moviepy.ae._geometry import MAX_RENDER_PIXELS, validate_matrix, validate_pixel_size
from moviepy.ae.buffer import Buffer
from moviepy.ae.layers import AVLayer, NullLayer, SolidLayer
from moviepy.ae.properties import Property
from moviepy.ae.transform import Transform
from moviepy.ae.warp import destination_bounds, warp_buffer


class MockClip:
    """Minimal clip for testing AVLayer behavior."""

    def __init__(self, duration=1.0):
        self.duration = duration
        self.get_frame_calls = []

    def get_frame(self, t):
        self.get_frame_calls.append(t)
        return np.ones((2, 2, 3), dtype=np.uint8) * 100


def test_c1_translated_scales_uniform_alpha():
    """C1: _translated must scale _uniform_alpha when applying opacity."""
    # Create a uniform alpha buffer
    array = np.ones((2, 2, 4), dtype=np.float32)
    array[..., :3] = 0.5  # RGB
    array[..., 3] = 0.8  # Alpha
    source = Buffer._publish(array, (0, 0), "srgb", (0.8, 1.0, True))

    # Apply opacity=50% via the integer translation fast path
    result = warp_buffer(source, np.eye(3), 0.5)

    # The metadata uniform_alpha should be 0.4 (0.8 * 0.5)
    assert abs(result._uniform_alpha - 0.4) < 1e-6
    # The actual pixel alpha should also be 0.4
    assert abs(result.rgba[0, 0, 3] - 0.4) < 1e-6


def test_h1_layer_time_for_transform_and_opacity():
    """H1: Transform properties evaluated at layer time, not composition time."""
    # Layer with start_time=5, position animates 0->10 over layer time 0..1
    layer = SolidLayer(
        "test",
        color=(255, 255, 255),
        size=(2, 2),
        start_time=5.0,
        transform=Transform(
            position=Property((0.0, 0.0), keyframes=[(0, (0, 0)), (1, (10, 0))]),
            opacity=Property(100.0, keyframes=[(0, 100), (1, 0)]),
            anchor_point=(0.0, 0.0),
        ),
    )

    # At comp t=5.5, layer time = 0.5
    # Position should be (5, 0), opacity should be 50%
    assert layer.source_time(5.5) == 0.5
    matrix = layer.world_matrix(5.5)
    assert abs(matrix[0, 2] - 5.0) < 1e-9
    assert abs(layer.opacity_at(5.5) - 0.5) < 1e-9

    # Parent chain test: parent with stretch=200
    parent = NullLayer(
        "parent",
        size=(2, 2),
        stretch=200.0,
        transform=Transform(
            position=Property((0.0, 0.0), keyframes=[(0, (0, 0)), (2, (20, 0))]),
            anchor_point=(0.0, 0.0),
        ),
    )
    child = SolidLayer(
        "child",
        color=(255, 255, 255),
        size=(2, 2),
        parent=parent,
        transform=Transform(position=(3.0, 0.0), anchor_point=(0.0, 0.0)),
    )

    # At comp t=1, parent layer time = 0.5, parent position = 5
    # Child world position = 5 + 3 = 8
    assert parent.source_time(1.0) == 0.5
    child_matrix = child.world_matrix(1.0)
    assert abs(child_matrix[0, 2] - 8.0) < 1e-9


def test_h2_nearest_filter_includes_boundary_pixels():
    """H2: Nearest filter must include boundary pixels with full energy."""
    # 5x5 source, 2x upscale
    scale_matrix = np.diag([2.0, 2.0, 1.0])

    # Nearest filter should include boundaries
    bounds = destination_bounds(scale_matrix, (5, 5), interpolation="nearest")
    assert bounds == (-1, -1, 10, 10)  # 11x11 pixels

    # Linear filter also expands to include its support
    bounds_linear = destination_bounds(scale_matrix, (5, 5), interpolation="linear")
    assert bounds_linear == (
        -1,
        -1,
        10,
        10,
    )  # Also 11x11 due to linear's 1.0 support radius


def test_h3_av_layer_default_out_point_includes_start_time():
    """H3: AVLayer.default_out_point must include start_time."""
    clip = MockClip(duration=1.5)
    layer = AVLayer(clip, start_time=0.5)

    # out_point should be start_time + duration = 2.0
    assert layer.out_point == 2.0

    # Source times should never be negative
    for comp_t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _ = layer.source_buffer(layer.source_time(comp_t))

    # All requested source times should be >= 0
    assert all(t >= 0 for t in clip.get_frame_calls)


def test_h4_area_budget_prevents_memory_exhaustion():
    """H4: Pixel area budget prevents unbounded allocations."""
    # Direct size validation
    with pytest.raises(ValueError, match="render budget"):
        validate_pixel_size((100000, 100000))

    # SolidLayer construction
    with pytest.raises(ValueError, match="render budget"):
        SolidLayer("huge", color=(255, 0, 0), size=(100000, 100000))

    # Warp destination bounds
    huge_scale = np.diag([10000.0, 10000.0, 1.0])
    with pytest.raises(ValueError, match="render budget"):
        warp_buffer(Buffer(np.ones((100, 100, 4), dtype=np.float32)), huge_scale, 1.0)

    # Budget constant is exposed
    assert MAX_RENDER_PIXELS == 2**26  # ~67M pixels


def test_m1_snap_rectangle_symmetric_guard():
    """M1: _snap_rectangle guards all four extremes symmetrically."""
    from moviepy.ae.warp import _snap_rectangle

    # Large positive x and y
    xs = np.array([0, 2**31])
    ys = np.array([0, 2**31])
    with pytest.raises(ValueError, match="exceed the representable"):
        _snap_rectangle(xs, ys)

    # Large negative x
    xs = np.array([-(2**31), 0])
    ys = np.array([0, 10])
    with pytest.raises(ValueError, match="exceed the representable"):
        _snap_rectangle(xs, ys)

    # Large negative y
    xs = np.array([0, 10])
    ys = np.array([-(2**31), 0])
    with pytest.raises(ValueError, match="exceed the representable"):
        _snap_rectangle(xs, ys)


def test_m3_in_point_setter_cross_validates():
    """M3: in_point setter prevents creating an inverted window."""
    layer = SolidLayer("test", color=(255, 0, 0), size=(2, 2))
    layer.out_point = 1.0

    # Cannot set in_point after out_point
    with pytest.raises(ValueError, match="must not follow out_point"):
        layer.in_point = 2.0

    # Can set equal (zero-duration window)
    layer.in_point = 1.0
    assert layer.in_point == 1.0


def test_m4_overflow_tracking_in_metadata():
    """M4: Value bounds are tracked correctly to prevent infinity propagation."""
    from moviepy.ae.warp import _may_overflow

    # Create a buffer with values near float32 max
    max_val = float(np.finfo(np.float32).max)
    huge_value = max_val * 0.6
    array = np.full((2, 2, 4), huge_value, dtype=np.float32)
    array[..., 3] = 1.0  # Valid alpha in [0, 1]
    source = Buffer(array)

    # Verify the overflow detection triggers for cubic with headroom
    assert _may_overflow(source, factor=1.0, headroom=2.0) is True

    # Test with actual resampling (non-identity) to trigger full path
    scale_matrix = np.diag([1.1, 1.1, 1.0])  # Force resampling
    result = warp_buffer(source, scale_matrix, 1.0, interpolation="cubic")

    # The result is finite (OpenCV handles large values gracefully)
    assert np.isfinite(result.rgba).all()

    # The metadata tracks potential overflow: value_bound * factor * headroom
    # For cubic interpolation, headroom=2.0
    expected_bound = source._value_bound * 1.0 * 2.0
    assert abs(result._value_bound - expected_bound) < 1e-6

    # The bound exceeds float32 max, indicating overflow risk
    assert result._value_bound > max_val


def test_m5_auto_orient_holds_final_tangent():
    """M5: auto_orient holds the final tangent direction after motion ends."""
    # Position moves right (0->10) then stops
    transform = Transform(
        position=Property((0.0, 0.0), keyframes=[(0, (0, 0)), (1, (10, 0))]),
        auto_orient=True,
    )

    # During motion at t=0.5: facing right (0 degrees)
    rotation = transform.rotation_at(0.5)
    assert abs(rotation - 0.0) < 1e-9

    # After motion at t=2.0: still facing right (AE behavior)
    rotation_after = transform.rotation_at(2.0)
    assert abs(rotation_after - 0.0) < 1e-9

    # Stationary position: zero rotation
    static_transform = Transform(position=(5.0, 5.0), auto_orient=True)
    assert static_transform.rotation_at(0.0) == 0.0


def test_validate_matrix_prevents_mutation():
    """validate_matrix returns a copy that cannot be mutated by the caller."""
    original = np.eye(3)
    validated = validate_matrix(original)

    # Modify the original
    original[0, 0] = 999.0

    # Validated copy should be unchanged
    assert validated[0, 0] == 1.0


def test_empty_source_returns_empty_at_bounds():
    """Empty sources return an empty buffer positioned at the explicit bounds."""
    # Create an empty buffer directly (size validation prevents 0x0 SolidLayer)
    empty = Buffer(np.zeros((0, 0, 4), dtype=np.float32))
    explicit_bounds = (10, 20, 30, 40)

    # warp_buffer with identity and explicit bounds
    # For empty sources, the bounds collapse to the top-left corner
    result = warp_buffer(empty, np.eye(3), 1.0, bounds=explicit_bounds)
    # Empty sources produce a degenerate rectangle at the requested position
    assert result.bounds == (10, 20, 10, 20)  # Collapsed to zero area
    assert result.size == (0, 0)  # Empty remains empty


def test_av_layer_clamps_source_time():
    """Test that AVLayer clamps negative source times to zero."""
    clip = MockClip(duration=1.0)
    layer = AVLayer(clip, in_point=-0.5, start_time=0.0)

    # At comp t=0, source_time = -0.5, but get_frame receives 0.0
    _ = layer.source_buffer(layer.source_time(0.0))
    assert clip.get_frame_calls[-1] == 0.0


def test_av_layer_validate_clip_hardening():
    """Test that AVLayer._validate_clip checks mask.get_frame."""

    class BadMask:
        """Mask without get_frame."""

        pass

    class ClipWithBadMask:
        """Clip with invalid mask."""

        def get_frame(self, t):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        mask = BadMask()

    with pytest.raises(TypeError, match="mask must provide get_frame"):
        AVLayer(ClipWithBadMask())
