"""The start smoke: READY BEFORE START (lib/dj/director.Director.prepare) - the tab's path, headless.

    python tools/tests/_dj_start_smoke.py --music D:/Devel/music

Builds the Director with no engine (as the tab does while its window sits open), prepares (helper warmed, the
opener picked and decoded), then attaches the engine and STARTS: the opener must open within two seconds of
START (it was 8-16 s, 2026-09-21). Also the plain path (no prepare) for the number. Never opens a device.
"""
import os
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.tests._low_priority import PRIORITY  # noqa: E402
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def run(music, prepared):
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    engine = AudioEngine()
    t0 = time.time()
    d = D.Director(music, engine=None, theme="groove")           # (as the tab: no engine before START)
    d.set_dial("layers", "two")
    t_built = time.time() - t0
    t_prep = 0.0
    if prepared:
        t1 = time.time()
        first = d.prepare()
        t_prep = time.time() - t1
        print(f"  prepared in {t_prep:.1f} s: {first.title[:36] if first else 'nothing'}")
    d.engine, d._attach = engine, engine.attach_track
    t_start = time.time()
    if not d.start(threaded=False):
        print(f"FAIL: did not start: {d.last_error}")
        return None
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    rc = d.rc
    opened = None
    while time.time() - t_start < 60.0:
        np.frombuffer(gen.send(BLOCK), dtype=np.float32)
        d.step()
        if rc.master is not None and (rc._tel_deck(rc.master) or {}).get("playing"):
            opened = time.time() - t_start
            break
        time.sleep(0.01)
    d.stop(0.1)
    print(f"  {'prepared' if prepared else 'plain   '}: Director built in {t_built:.1f} s, START → first note {opened if opened is not None else float('nan'):.1f} s")
    return opened


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    print(f"priority: {PRIORITY}")
    plain = run(music, prepared=False)
    prep = run(music, prepared=True)
    if prep is None or plain is None:
        print("FAIL: a start did not open")
        return 1
    if prep > 2.0:
        print(f"FAIL: the prepared START took {prep:.1f} s to the first note (two allowed)")
        return 1
    print(f"ALL OK  (plain {plain:.1f} s → prepared {prep:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
