"""Gate for the Scope plugin (tools/gen/console/plugins/scope.py) and the
rack's monitor tap it reads.

  1. TAP: rack.recent(n) returns the last n rendered samples, oldest first,
     across the ring's wrap; LocalBackend.audio_tap forwards it; the
     remote backend answers None.
  2. MATH: a 1 kHz tone through ScopeMath peaks at 1 kHz; smoothing holds
     state; capture/clear baseline; difference against the baseline is ~0
     for the same signal and shows the change for a different one.
  3. TAB: the plugin adds a Scope tab; it paints (grab) with and without a
     backend; baseline controls drive the canvas; the message states why
     nothing is shown when idle or remote.

Usage: QT_QPA_PLATFORM=offscreen python tools/tests/_gen_scope_test.py [out.png]
"""
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer                      # noqa: E402
from lib.gen.synth import SynthRack                        # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    print("== tap")
    c = Composer("groove", bpm=124, key="8A", seed=1)
    rack = SynthRack(c.style, c.bpm, seed=1)
    rack.schedule(c.next_phrase().events)
    blocks = [rack.render(2048) for _ in range(4)]
    last = rack.recent(3000)
    check(last.shape == (3000, 2) and np.array_equal(last[-2048:], blocks[-1]) and np.array_equal(last[:952], blocks[-2][-952:]), "recent(n) = last n samples, oldest first")
    for _ in range(int(RATE * 4 / 2048) + 5):          # wrap the 4 s ring
        rack.render(2048)
    b = rack.render(2048); r = rack.recent(2048)
    check(np.array_equal(r, b), "correct across the ring wrap")
    from tools.gen.console.backend import LocalBackend, RemoteBackend
    be = LocalBackend({"style": "groove", "bpm": 124, "key": "8A", "seed": 2, "log_dir": tempfile.mkdtemp()}, audio=False)
    check(be.audio_tap(100) is None, "audio_tap None while idle")
    be.start(); be.pump(2.0)
    t = be.audio_tap(4096)
    check(t is not None and t.shape == (4096, 2) and float(np.abs(t).max()) > 0.01, "audio_tap returns live audio")
    check(RemoteBackend("http://127.0.0.1:1").audio_tap(10) is None, "remote backend has no tap")

    print("== math")
    from tools.gen.console.plugins.scope import FFT_N, ScopeMath
    m = ScopeMath(); m.smooth = 0.0
    tt = np.arange(FFT_N * 2) / RATE
    tone = np.stack([np.sin(2 * np.pi * 1000 * tt)] * 2, axis=1).astype(np.float32) * 0.5
    db = m.spectrum(tone)
    check(abs(ScopeMath.peak_hz(m.freqs, db) - 1000.0) < 15, f"1 kHz tone peaks at {ScopeMath.peak_hz(m.freqs, db):.0f} Hz")
    check(-8 < db.max() < 0, f"amplitude calibrated ({db.max():.1f} dBFS for a 0.5 sine ≈ -6)")
    check(m.capture() and m.baseline is not None, "baseline captured")
    diff = m.spectrum(tone) - m.baseline
    check(np.abs(diff).max() < 1e-6, "same signal minus baseline ≈ 0")
    tone2 = np.stack([np.sin(2 * np.pi * 3000 * tt)] * 2, axis=1).astype(np.float32) * 0.5
    d2 = m.spectrum(tone2) - m.baseline
    check(abs(ScopeMath.peak_hz(m.freqs, d2) - 3000.0) < 30 and d2.max() > 40, "difference shows the new tone")
    m.smooth = 0.9; a = m.spectrum(tone2); b2 = m.spectrum(tone)
    check(ScopeMath.peak_hz(m.freqs, b2) == ScopeMath.peak_hz(m.freqs, a), "smoothing holds the previous picture")
    m.clear(); check(m.baseline is None, "baseline cleared")

    print("== tab")
    from PyQt6.QtWidgets import QApplication
    from tools.gen.console.app import ConsoleWindow
    app = QApplication.instance() or QApplication([])
    win = ConsoleWindow(None, refresh_ms=10 ** 7); win.show()
    names = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    check("Scope" in names and "scope" in win.plugins, f"Scope tab registered {names}")
    tab = win.pages["scope"]; win.tabs.setCurrentWidget(tab); app.processEvents(); tab.tick()
    check("connect" in tab.canvas.message, f"idle message: {tab.canvas.message!r}")
    win.set_backend(be); win.tabs.setCurrentWidget(tab); app.processEvents(); tab.tick()
    check(tab.canvas.message == "" and tab.canvas.wave is not None and tab.canvas.spec is not None, "live: waveform + spectrum populated")
    tab.gain.setValue(6); tab.ref.setValue(-12); tab.floor.setValue(-70); tab.diff.setChecked(True); tab.capture()
    check(tab.canvas.gain_db == 6 and tab.canvas.ref_db == -12 and tab.canvas.floor_db == -70 and tab.canvas.diff and tab.canvas.baseline is not None, "baseline controls drive the canvas")
    win.resize(1100, 900); app.processEvents(); img = win.grab()
    check(not img.isNull(), "paints")
    if len(sys.argv) > 1:
        img.save(sys.argv[1]); print(f"  wrote {sys.argv[1]}")
    win.close()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
