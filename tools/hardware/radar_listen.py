"""Standalone listener for EtherNode radar OSC traffic.

Use this to verify the firmware → network → host data path before
wiring the engine up. Doesn't depend on Stories_OGL, the renderer, or
any of GL_Simple's runtime — just python-osc and a UDP socket.

Usage:
    python tools/hardware/radar_listen.py                # bind 0.0.0.0:9001
    python tools/hardware/radar_listen.py --port 9000    # different port
    python tools/hardware/radar_listen.py -v             # also dump every raw message

The firmware emits an OSC bundle every ~1/rateHz seconds (default
5 Hz, configurable in the EtherNode GUI's LD2412 tab). Each bundle contains:

    /en/<host>/radar       <int state 0..3>
    /en/<host>/radar/m     <int moving distance cm>
    /en/<host>/radar/s     <int stationary distance cm>
    /en/<host>/radar/d     <int closest detected distance cm>
    /en/<host>/radar/me    <int moving energy 0..100>
    /en/<host>/radar/se    <int stationary energy 0..100>
    /en/<host>/radar/gm    <14 ints, gates 0..13 — moving>
    /en/<host>/radar/gs    <14 ints, gates 0..13 — stationary>

Set each device's OSC remote host to this machine's IP and remote port
to the value above (default 9001) in the EtherNode GUI's OSC tab.

Output: one block per host every second showing the latest values from
that host, plus the message rate (sanity check: should be 8 × rateHz —
8 messages per bundle × the configured rate). Hosts silent for 5 s+
flip to OFFLINE.
"""
from __future__ import annotations

import argparse
import threading
import time

try:
    from pythonosc import dispatcher, osc_server
except ImportError:
    raise SystemExit("python-osc not installed. Run: pip install python-osc")


N_GATES = 14
OFFLINE_AFTER_S = 5.0
PRINT_PERIOD_S  = 1.0

# Two-letter state labels keep alignment tidy in the per-host line.
_STATE_LABELS = {0: 'none', 1: 'mov ', 2: 'stat', 3: 'both'}


# ---------------------------------------------------------------------------
# Per-host record
# ---------------------------------------------------------------------------

class HostState:
    """Holds the most recent values from one device. Mutated under
    ``_lock`` because OSC dispatch and the print loop run on different
    threads."""

    def __init__(self, host: str):
        self.host = host
        self.state = 0
        self.m  = self.me = 0
        self.s  = self.se = 0
        self.d  = 0
        self.gm = [0] * N_GATES
        self.gs = [0] * N_GATES
        self.last_t = 0.0
        # Counter is reset every print cycle so we can compute msg/sec.
        self.msg_count = 0
        self.msg_window_t = time.time()


_lock = threading.Lock()
_hosts: dict[str, HostState] = {}
_verbose = False


def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# OSC dispatch — one handler per address shape we care about
# ---------------------------------------------------------------------------

def _on_osc(address: str, *args) -> None:
    parts = address.strip('/').split('/')
    # Expect /en/<host>/radar[/<sub>]
    if len(parts) < 3 or parts[0] != 'en' or parts[2] != 'radar':
        return

    host = parts[1]
    sub  = parts[3:] if len(parts) > 3 else []

    if _verbose:
        print(f"  RX  {address}  {args}")

    with _lock:
        h = _hosts.get(host)
        if h is None:
            h = HostState(host)
            _hosts[host] = h
            print(f"[new host] {host}")
        h.last_t = time.time()
        h.msg_count += 1

        if not sub:
            # Bare /en/<host>/radar = state int
            if args:
                h.state = _safe_int(args[0])
        elif sub == ['m']:
            if args: h.m  = _safe_int(args[0])
        elif sub == ['s']:
            if args: h.s  = _safe_int(args[0])
        elif sub == ['d']:
            if args: h.d  = _safe_int(args[0])
        elif sub == ['me']:
            if args: h.me = _safe_int(args[0])
        elif sub == ['se']:
            if args: h.se = _safe_int(args[0])
        elif sub == ['gm']:
            for i in range(min(N_GATES, len(args))):
                h.gm[i] = _safe_int(args[i])
        elif sub == ['gs']:
            for i in range(min(N_GATES, len(args))):
                h.gs[i] = _safe_int(args[i])


# ---------------------------------------------------------------------------
# Periodic per-host summary printer
# ---------------------------------------------------------------------------

def _format_gates(arr: list[int]) -> str:
    """One-line, fixed-column gate dump. Three chars per gate so values
    up to 100 align cleanly."""
    return ' '.join(f'{v:3d}' for v in arr)


def _print_loop() -> None:
    while True:
        time.sleep(PRINT_PERIOD_S)
        now = time.time()

        with _lock:
            if not _hosts:
                continue
            print()
            print(f"--- {time.strftime('%H:%M:%S')} -------------------------------------------")
            for h in sorted(_hosts.values(), key=lambda x: x.host):
                age = now - h.last_t
                window = max(now - h.msg_window_t, 0.001)
                rate = h.msg_count / window
                # Reset counter window for next sample period.
                h.msg_count = 0
                h.msg_window_t = now

                if age > OFFLINE_AFTER_S:
                    print(f"  {h.host:20s}  OFFLINE  (last seen {age:5.1f}s ago)")
                    continue

                state = _STATE_LABELS.get(h.state, '?   ')
                print(f"  {h.host:20s}  {state}  "
                      f"d={h.d:4d}cm  "
                      f"mov={h.m:4d}cm/E{h.me:3d}  "
                      f"stat={h.s:4d}cm/E{h.se:3d}  "
                      f"{rate:4.1f} msg/s")
                print(f"     gm: {_format_gates(h.gm)}")
                print(f"     gs: {_format_gates(h.gs)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    global _verbose

    parser = argparse.ArgumentParser(
        description="Listen for EtherNode LD2412 radar OSC traffic; "
                    "print a per-host summary every second.")
    parser.add_argument('--host', default='0.0.0.0',
                        help='bind address (default 0.0.0.0 = all interfaces)')
    parser.add_argument('--port', type=int, default=9001,
                        help='UDP port to listen on (default 9001, matches '
                             'the EtherNode firmware default OSC remote_port)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='also print every raw OSC message as it arrives '
                             '(loud at 5 Hz × 8 msgs/bundle × N hosts)')
    args = parser.parse_args()

    _verbose = args.verbose

    disp = dispatcher.Dispatcher()
    # /en/* matches everything under /en/. Our handler then filters
    # for the radar subset; non-radar traffic is silently ignored.
    disp.map('/en/*', _on_osc)

    try:
        server = osc_server.ThreadingOSCUDPServer((args.host, args.port), disp)
    except OSError as e:
        print(f"Could not bind {args.host}:{args.port} — {e}")
        print("(Is something else listening on that port? "
              "Stop the GL_Simple engine first if it's running.)")
        return 2

    print(f"Listening on {args.host}:{args.port}")
    print("Set each device's OSC remote host = this machine's IP, "
          f"remote port = {args.port}")
    print("(EtherNode GUI → OSC tab → 'OSC target' fields, then Save + Reboot.)")
    print()
    print("Ctrl-C to quit.")

    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _print_loop()
    except KeyboardInterrupt:
        print("\nDone.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
