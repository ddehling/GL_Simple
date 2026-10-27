# Standalone tool: find Pixelblazes on the LAN and check they're ready for the mirror.
"""Usage:
    python tools/hardware/pixelblaze_probe.py              # list what beacons
    python tools/hardware/pixelblaze_probe.py --test       # + push a moving rainbow
    python tools/hardware/pixelblaze_probe.py --host 192.168.68.50 --test

For each device: name, pixel count, map shape, the running pattern, and
whether the "GL_Simple Mirror" pattern is installed. --test switches to
that pattern for ~5 s, streams a rainbow, then switches back.
"""

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.pixelblaze_mirror import (PING, PixelblazeLink, SamplePlan, discover,  # noqa: E402
                                   find_segment)


def probe(ip: str, pattern: str, test: bool, interpolate: bool = False):
    link = PixelblazeLink(ip)
    try:
        link.connect()
        settings, active = link.get_config()
        n = int(settings.get("pixelCount") or 0)
        programs = link.list_programs()
        coords = link.get_map(n)
        mirror_id = next((pid for pid, name in programs.items()
                          if name.strip().lower() == pattern.lower()), None)
        print(f"{ip}  {settings.get('name', '?')}")
        print(f"   pixels:  {n}")
        print(f"   map:     {'none (strip)' if coords is None else f'{coords.shape[1]}D'}")
        print(f"   running: {active.get('name', '?')}")
        print(f"   mirror pattern: {'installed' if mirror_id else 'MISSING'}")
        plan = (SamplePlan(n, find_segment(coords, n), interpolate=interpolate)
                if n else None)
        if plan:
            print(f"   would send: {-(-n // plan.seg)} segment(s) x {plan.k} samples "
                  f"(of {plan.seg} px each)")
        if not (test and mirror_id and n):
            return
        prior = active.get("activeProgramId")
        link.set_active_program(mirror_id)
        print("   streaming a rainbow for 5 s...")
        try:
            stream_rainbow(link, n, plan)
        finally:
            if prior and prior != mirror_id:
                link.close()
                link.connect()   # the stream may have broken the socket
                link.set_active_program(prior)
                print("   restored its own pattern")
    except Exception as e:
        print(f"{ip}  error: {e.__class__.__name__}: {e}")
    finally:
        link.close()


def stream_rainbow(link, n, plan):
    """Ack-paced like the mirror: a frame, wait for its ack, the next."""
    fps = []
    acks = [0]
    cv = threading.Condition()

    def read_stats():
        while link.ws is not None:
            try:
                msg = link.ws.recv()
            except Exception:
                return
            if isinstance(msg, str) and msg.startswith('{"fps"'):
                fps.append(json.loads(msg)["fps"])
            elif isinstance(msg, str) and '"ack"' in msg:
                with cv:
                    acks[0] += 1
                    cv.notify_all()

    link.ws.settimeout(5.0)
    threading.Thread(target=read_stats, daemon=True).start()
    t0 = time.monotonic()
    seq = 0
    rtts = []
    while time.monotonic() - t0 < 5.0:
        hue = (np.arange(n) / max(n, 1) + (time.monotonic() - t0) * 0.3) % 1.0
        rgb = (np.stack([np.abs(hue * 6 - 3) - 1,
                         2 - np.abs(hue * 6 - 2),
                         2 - np.abs(hue * 6 - 4)], axis=1).clip(0, 1)
               * 255).astype(np.uint8)
        seq += 1
        with cv:
            before, t = acks[0], time.monotonic()
            link.ws.send(plan.message(rgb, seq))
            link.ws.send(PING)
            if not cv.wait_for(lambda: acks[0] > before, timeout=5.0):
                raise TimeoutError("no ack for 5 s")
            rtts.append(time.monotonic() - t)
    print(f"   {seq} frames acked ({seq / 5.0:.1f} fps, frame->ack "
          f"{np.median(rtts) * 1000:.0f} ms); device rendering at "
          f"{', '.join('%.1f' % f for f in fps[1:]) or '?'} fps")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", action="append", default=[],
                    help="probe this IP too (repeatable)")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="how long to listen for beacons")
    ap.add_argument("--pattern", default="GL_Simple Mirror")
    ap.add_argument("--interp", action="store_true",
                    help="--test with blending between samples (slower)")
    ap.add_argument("--test", action="store_true",
                    help="stream a rainbow to each device for 5 s")
    args = ap.parse_args()

    ips = set(args.host)
    try:
        ips |= discover(args.seconds)
    except OSError as e:
        print(f"can't listen for beacons on UDP 1889: {e}")
    if not ips:
        print("No Pixelblazes found. Check they're on this subnet, and that "
              "the firewall lets python receive UDP 1889.")
        return
    for ip in sorted(ips):
        probe(ip, args.pattern, args.test, args.interp)


if __name__ == "__main__":
    main()
