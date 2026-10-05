"""Keep a Windows GPU surface invisible while its message loop stays alive."""

import os
import threading
from contextlib import contextmanager


@contextmanager
def hidden_render_window(size):
    """Yield a hidden native parent handle, then release only our own window."""
    if os.name != "nt":
        raise NotImplementedError("hidden Godot rendering currently requires Windows")
    import ctypes
    from ctypes import wintypes as wt

    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    user.CreateWindowExW.argtypes = [
        wt.DWORD,
        wt.LPCWSTR,
        wt.LPCWSTR,
        wt.DWORD,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wt.HWND,
        wt.HMENU,
        wt.HINSTANCE,
        wt.LPVOID,
    ]
    user.CreateWindowExW.restype = wt.HWND
    user.GetMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT]
    user.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
    user.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
    user.DispatchMessageW.restype = ctypes.c_ssize_t
    user.DestroyWindow.argtypes = [wt.HWND]
    user.IsWindowVisible.argtypes = [wt.HWND]
    user.PostThreadMessageW.argtypes = [wt.DWORD, wt.UINT, wt.WPARAM, wt.LPARAM]
    state, ready = {}, threading.Event()

    def run():
        state["thread"] = kernel.GetCurrentThreadId()
        hwnd = user.CreateWindowExW(
            0x08000080,  # WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
            "STATIC",
            "MoviePy Godot render surface",
            0x80000000,  # WS_POPUP without WS_VISIBLE
            0,
            0,
            size[0],
            size[1],
            None,
            None,
            None,
            None,
        )
        state["hwnd"] = hwnd
        state["error"] = ctypes.get_last_error()
        ready.set()
        if not hwnd:
            return
        message = wt.MSG()
        try:
            while user.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                user.TranslateMessage(ctypes.byref(message))
                user.DispatchMessageW(ctypes.byref(message))
        finally:
            user.DestroyWindow(hwnd)

    thread = threading.Thread(target=run, daemon=True, name="MoviePy Godot surface")
    thread.start()
    try:
        if not ready.wait(10) or not state.get("hwnd"):
            raise RuntimeError(
                f"cannot create hidden GPU surface: {state.get('error')}"
            )
        if user.IsWindowVisible(state["hwnd"]):
            raise RuntimeError("Godot parent surface must remain hidden")
        yield state["hwnd"]
    finally:
        if state.get("thread"):
            user.PostThreadMessageW(state["thread"], 0x0012, 0, 0)  # WM_QUIT
        thread.join(5)
