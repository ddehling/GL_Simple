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
from typing import Any, Callable, List, Optional, Tuple

try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    _PYTHONOSC_AVAILABLE = True
except ImportError:
    _PYTHONOSC_AVAILABLE = False


# Type alias: a route handler is called with the full address + raw args.
RouteHandler = Callable[..., None]


class OscListener:
    """OSC listener with optional prefix-route handlers.

    Default behavior is print-only — every message hits the catch-all log.
    Projects (via their ``button_router`` hook or similar) can install
    prefix routes that override the log for matching addresses, so the
    same listener serves both observability and routed handling.

    Usage::

        listener = OscListener(port=9001)
        listener.start()
        listener.register_prefix("/wol/", on_wol_message)
        ...
        listener.stop()

    Handler signature: ``handler(address: str, *args)`` — same shape as
    the underlying python-osc default handler. Handlers run in the OSC
    server's worker thread; they should be fast and non-blocking.
    """

    def __init__(self, port: int = 9001, bind_ip: str = "0.0.0.0"):
        self.port = port
        self.bind_ip = bind_ip
        self._server: Optional[ThreadingOSCUDPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stopped = False
        # First-match-wins prefix routes. Threading: appended only via
        # register_prefix() which is normally called at boot from the
        # main thread; reads happen in the OSC worker thread. The list
        # is replaced atomically (Python list .append is thread-safe
        # for the single-writer case), so no lock needed.
        self._prefix_routes: List[Tuple[str, RouteHandler]] = []

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

    def register_prefix(self, prefix: str, handler: RouteHandler) -> None:
        """Route any message whose address starts with ``prefix`` to
        ``handler(address, *args)``. First-match-wins if multiple
        prefixes register; longer / more specific prefixes should be
        registered first.

        Routed messages skip the catch-all log so the operator's
        terminal isn't flooded with high-rate routed traffic. Use
        ``register_prefix("/", h)`` to capture everything (and disable
        the log)."""
        self._prefix_routes.append((prefix, handler))

    def _on_message(self, address: str, *args: Any) -> None:
        # First, try registered prefix routes. First match wins.
        for prefix, handler in self._prefix_routes:
            if address.startswith(prefix):
                try:
                    handler(address, *args)
                except Exception as e:
                    # Don't let one buggy handler take down the whole
                    # listener thread; log and keep serving.
                    print(f"[OSC] handler for {prefix!r} raised on "
                          f"{address!r}: {e}")
                return

        # Catch-all: format args compactly so a flood of high-rate
        # unrouted messages stays one line each.
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
