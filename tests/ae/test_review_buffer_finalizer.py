"""A storage finalizer must not block or change an allocation in progress."""

import subprocess
import sys
import textwrap


def test_gc_during_pool_checkout_preserves_storage_and_completes():
    """Collect a retired lease after selecting a pool slot, before removing it."""
    script = textwrap.dedent(
        """
        import gc
        import inspect
        import sys

        import numpy as np
        import moviepy.ae.buffer as buffers

        gc.collect()
        gc.disable()
        doomed = buffers._owned_output((1, 3, 4))
        cycle = [doomed]
        cycle.append(cycle)
        del cycle, doomed
        expected = np.arange(4, dtype=np.float32)
        other = np.empty(8, dtype=np.float32)
        buffers._FREE_STORAGE[:] = [expected, other]
        lines, start = inspect.getsourcelines(buffers._owned_output)
        pop_line = start + next(
            i for i, line in enumerate(lines)
            if "storage = _FREE_STORAGE.pop(index)" in line
        )
        collected = []

        def collect_at_checkout(frame, event, arg):
            if (event == "line"
                    and frame.f_code is buffers._owned_output.__code__
                    and frame.f_lineno == pop_line):
                sys.settrace(None)
                gc.collect()
                collected.append(True)
            return collect_at_checkout

        sys.settrace(collect_at_checkout)
        try:
            result = buffers._owned_output((1, 1, 4))
        finally:
            sys.settrace(None)
        assert collected == [True]
        assert result.shape == (1, 1, 4)
        assert np.shares_memory(result, expected)
        np.testing.assert_array_equal(result.ravel(), expected)
        assert len(buffers._FREE_STORAGE) == 1
        assert buffers._FREE_STORAGE[0] is other
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
