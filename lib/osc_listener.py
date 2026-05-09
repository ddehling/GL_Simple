"""OSC listener for Weight_Of_Light receiver events.

Receives OSC messages on a UDP port (default 9001 — the destination port the
Weight_Of_Light boxes target by default) and prints each one. Pure
observability for now: no mapping table, no event-bus integration. Once the
shape of the incoming traffic is well-understood we can route specific
addresses into the web controller / scheduler from here.

Runs in a daemon thread so the main render loop is unaffected. Messages are
parsed by ``python-osc``; we register a single catch-all dispatcher so every
incoming address surfaces through the same log line regardless of pattern.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    _PYTHONOSC_AVAILABLE = True
except ImportError:
    _PYTHONOSC_AVAILABLE = False


class OscListener:
    """Print-only OSC listener. Start once at app boot, stop at shutdown.

    Usage:
        listener = OscListener(port=9001)
        listener.start()
        ...
        listener.stop()
    """

    def __init__(self, port: int = 9001, bind_ip: str = "0.0.0.0"):
        self.port = port
        self.bind_ip = bind_ip
        self._server: Optional[ThreadingOSCUDPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

    def start(self) -> bool:
        """Bind the socket and start the listener thread. Returns False if
        python-osc isn't installed (warns but doesn't raise — keeps the rest
        of GL_Simple running so a missing dep doesn't stop the show)."""
        if not _PYTHONOSC_AVAILABLE:
            print("[OSC] python-osc not installed; listener disabled. "
                  "Run: pip install -r requirements.txt")
            return False

        dispatcher = Dispatcher()
        # Catch-all: every incoming address routes through _on_message. The
        # default handler fires for any address not explicitly mapped, and
        # we never explicitly map anything, so it sees everything.
        dispatcher.set_default_handler(self._on_message)

        try:
            self._server = ThreadingOSCUDPServer(
                (self.bind_ip, self.port), dispatcher)
        except OSError as e:
            # Most common failure: port already in use (another GL_Simple
            # instance running, or a leftover process).
            print(f"[OSC] Could not bind {self.bind_ip}:{self.port} — {e}")
            return False

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="osc-listener",
            daemon=True)
        self._thread.start()
        print(f"[OSC] listening on {self.bind_ip}:{self.port}")
        return True

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        print("[OSC] listener stopped")

    @staticmethod
    def _on_message(address: str, *args: Any) -> None:
        # Format args compactly so a flood of high-rate messages
        # (e.g. analog at 50Hz) stays one line each.
        if not args:
            payload = ""
        elif len(args) == 1:
            payload = _fmt_arg(args[0])
        else:
            payload = " ".join(_fmt_arg(a) for a in args)
        ts = time.strftime("%H:%M:%S")
        print(f"[OSC {ts}] {address} {payload}")


def _fmt_arg(a: Any) -> str:
    if isinstance(a, float):
        # 3 decimals is enough resolution for analog/temp without making
        # the line ugly when the value is integer-ish (e.g. 23.000).
        return f"{a:.3f}".rstrip('0').rstrip('.')
    return str(a)
