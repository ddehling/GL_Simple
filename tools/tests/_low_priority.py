"""Import FIRST in every DJ gate: the test process - and every helper it spawns - runs at the lowest priority the
machine gives, so a gate may run beside the live planner (user 2026-09-21: "stutters are understandable. It's
just when the system decides to simply not play anything that I criticise"). A child created by an IDLE-class
process inherits the class on Windows, so this must run before the decode pool exists. DJ_TEST_PRIORITY=normal
turns it off (a timing measurement wants the normal class)."""
import os
import sys


def lower():
    if os.environ.get("DJ_TEST_PRIORITY", "").lower() == "normal":
        return "normal"
    try:
        if sys.platform.startswith("win"):
            import ctypes
            from ctypes import wintypes
            IDLE_PRIORITY_CLASS = 0x00000040
            k32 = ctypes.windll.kernel32
            # (the pseudo-handle is pointer-sized: without these, -1 truncated to 32 bits was an invalid handle)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
            k32.GetPriorityClass.restype = wintypes.DWORD
            h = k32.GetCurrentProcess()
            ok = k32.SetPriorityClass(h, IDLE_PRIORITY_CLASS)
            return "idle" if (ok and k32.GetPriorityClass(h) == IDLE_PRIORITY_CLASS) else f"normal (SetPriorityClass failed, err {ctypes.get_last_error()})"
        os.nice(19)
        return "nice 19"
    except Exception as e:  # noqa: BLE001
        return f"normal ({type(e).__name__})"


PRIORITY = lower()
