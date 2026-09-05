"""Gate for the SuperCollider backend (lib/gen/backends/sc.py) - Spike SC-0.

Renders the same composer output through scsynth NON-REALTIME (no audio
device, hermetic) and checks:
  1. It renders (supriya + scsynth present; otherwise SKIP, exit 0).
  2. CLEAN: finite, limited (peak <= 0.96), not silent, stereo.
  3. KICKS LAND ON THE GRID at the composer's sample times -> the OSC
     bundle timestamps are right (this is the clock contract the live
     path relies on).
  4. SPEED: NRT faster than realtime.

Usage: python tools/tests/_gen_sc_test.py [out.wav]
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer                      # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    try:
        import supriya  # noqa: F401
        from lib.gen.backends.sc import render_nrt
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: supriya not importable ({e})")
        return 0
    if not shutil.which("scsynth"):
        print("SKIP: scsynth not on PATH (apt install supercollider-server)")
        return 0
    seconds = 16
    c = Composer("groove", bpm=124, key="8A", seed=2)
    kicks, events = [], []
    for p in c.phrases_until(seconds * RATE):
        events += p.events
        kicks += [e.at for e in p.events if e.slot == "kick"]
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "gen_sc_test.wav")
    t0 = time.time()
    path = render_nrt(events, seconds, out, c.style)
    dt = time.time() - t0
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32")
    check(sr == RATE and x.ndim == 2 and x.shape[1] == 2, f"rendered {x.shape} @ {sr}")
    check(np.isfinite(x).all(), "finite")
    pk = float(np.abs(x).max())
    check(pk <= 0.96, f"limited: peak {pk:.3f}")
    rms = float(np.sqrt(np.mean(x ** 2)))
    check(rms > 0.02, f"not silent (rms {20 * np.log10(rms + 1e-9):.1f} dBFS)")
    from tools.tests._gen_synth_test import kick_alignment
    # kick-only NRT render for the clock contract (bass would mask onsets)
    kick_events = [e for e in events if e.slot == "kick"]
    kpath = render_nrt(kick_events, seconds, os.path.join(tempfile.gettempdir(), "gen_sc_kicks.wav"), c.style)
    kx, _ = sf.read(kpath, dtype="float32")
    hit, total = kick_alignment(kx[: seconds * RATE], kicks)
    check(total > 15 and hit >= 0.95 * total, f"{hit}/{total} kicks land on the composer's grid (kick-only render)")
    check(seconds / dt > 1.0, f"NRT {seconds / dt:.1f}x realtime")
    print(f"wrote {path}")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
