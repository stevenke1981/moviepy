"""Slow WS-03 memory check: 20 layers at 1080p over 200 frames.

Skipped unless the environment variable AE_SLOW=1 is set::

    AE_SLOW=1 python -m pytest tests/ae/test_memory_slow.py -q

The fast variant in test_composition.py uses 160x90; this one follows the
WS-03 specification (docs/ae/AE_REVIEW_PLAN.md, D5): resident memory and
traced Python allocations must grow by less than 5% across 200 frames.
"""

import ctypes
import os
import sys
import tracemalloc

import pytest

from moviepy.ae import Composition
from moviepy.ae.properties import Property


pytestmark = pytest.mark.skipif(
    os.environ.get("AE_SLOW") != "1", reason="slow; set AE_SLOW=1 to run"
)

WARMUP_FRAMES = 20
MEASURED_FRAMES = 200
LIMIT = 0.05


class _Counters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _rss():
    """Current resident set size in bytes, or None when unsupported."""
    if sys.platform == "win32":
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = ctypes.c_void_p
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
        if query(current(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize
        return None
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _comp():
    """Twenty moving, semitransparent 600x400 solids on a 1080p comp."""
    comp = Composition(size=(1920, 1080), fps=30, duration=10)
    for index in range(20):
        layer = comp.add_solid(
            f"l{index}", color=(index * 10, 50, 200), size=(600, 400)
        )
        start = (100.0 + index * 40, 100.0 + index * 20)
        layer.transform.position = Property(
            start, keyframes=[(0, start), (10, (1800.0, 1000.0))]
        )
        layer.transform.opacity = 60.0
    return comp


def test_rendering_1080p_20_layers_does_not_grow_memory():
    rss_available = _rss() is not None
    if not rss_available:
        pytest.skip("no RSS probe on this platform")
    comp = _comp()
    tracemalloc.start()
    try:
        for frame in range(WARMUP_FRAMES):
            comp.get_frame(frame / 30)
        traced_before = tracemalloc.get_traced_memory()[0]
        rss_before = _rss()
        for frame in range(WARMUP_FRAMES, WARMUP_FRAMES + MEASURED_FRAMES):
            comp.get_frame(frame / 30)
        traced_after = tracemalloc.get_traced_memory()[0]
        rss_after = _rss()
    finally:
        tracemalloc.stop()
    rss_growth = (rss_after - rss_before) / rss_before
    traced_growth = (traced_after - traced_before) / traced_before
    print(
        f"rss {rss_before} -> {rss_after} ({rss_growth:+.2%}); "
        f"traced {traced_before} -> {traced_after} ({traced_growth:+.2%})"
    )
    assert rss_growth < LIMIT
    assert traced_growth < LIMIT
