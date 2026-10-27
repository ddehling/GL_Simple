"""ENGINE SWITCH smoke: the Director runs headless (the mixer pulled by hand), LAYERS is turned record -> two -> one
-> record while it plays; each turn must swap the engine (mode, the engine object, the audible submix) within a
few seconds, with no error and no dead air beyond the fade.

    python tools/tests/_dj_engine_switch_smoke.py --music D:/Devel/music

Exit 0 = OK, 1 = FAIL.
"""
import os
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from tools.tests._low_priority import PRIORITY  # noqa: E402,F401
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    for k in list(getattr(D, "PLAY_MIN_S", {}) or {}):
        D.PLAY_MIN_S[k] = 60.0
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    if not d.start(threaded=False):
        print("did not start:", d.last_error)
        return 1
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    fails = []

    def render(seconds, label):
        """Render `seconds` of audio stepping the Director; returns (seconds audible, longest silence)."""
        n = int(seconds * RATE / BLOCK)
        aud = 0
        longest = cur = 0
        for _ in range(n):
            x = np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2)
            d.step()
            rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
            if rms > 0.004:
                aud += 1
                cur = 0
            else:
                cur += 1
                longest = max(longest, cur)
        return aud * BLOCK / RATE, longest * BLOCK / RATE

    def engine_of():
        return "record" if d.player is not None else ("one" if d.system is not None else ("layered" if d.rc is not None else "none"))

    # the first record must be audible first
    t0 = time.time()
    while time.time() - t0 < 180:
        a, _ = render(1.0, "warm")
        if a > 0.5:
            break
    a, _ = render(8.0, "record")
    print(f"record mode: engine {engine_of()}, mode {d.mode}, audible {a:.1f} s of 8")
    if a < 6.0 or engine_of() != "record":
        fails.append("record mode did not play")
    # record <-> the others: the engine is swapped (a fade, a fresh start); one <-> layered: the warm handover.
    # The new engine's first song must be heard within LIMIT s of the press (the conductor decodes stems: slow)
    LIMIT = 120.0
    for want, expect in (("two", "layered"), ("one", "one"), ("record", "record")):
        was = d.mode
        ok = d.set_dial("layers", want)
        ev = d.events[-1][1] if d.events else ""
        t_press = time.time()
        heard_after = None
        rendered = 0.0
        # paced to REAL time: a decode is wall seconds (the control gate waits on the clock for the same reason)
        while time.time() - t_press < LIMIT:
            a, _gap = render(1.0, want)
            rendered += 1.0
            lag = rendered - (time.time() - t_press)
            if lag > 0.05:
                time.sleep(min(lag, 1.0))
            if a >= 0.5 and engine_of() == expect:
                heard_after = time.time() - t_press
                break
        a2, _ = render(6.0, want)
        # THE PICTURE must be there for every engine: the record, the decks and the maps, JSON-clean
        pic = ""
        try:
            import json
            st = d.status()
            maps = d.deck_maps()
            json.dumps(st); json.dumps(maps)
            rec, decks = st.get("record"), st.get("decks") or {}
            pic = f"picture: record {'yes' if rec else 'NO'} ({(rec or {}).get('bar')} bars), decks {sorted(decks)}, roles {[v.get('role') for v in decks.values()]}, maps {sorted(maps)}, plan {(st.get('plan') or {}).get('kind')}"
            if not rec or not decks or not maps or st.get("error"):
                fails.append(f"LAYERS {want}: the picture is incomplete - {pic} - error {st.get('error')}")
        except Exception as e:  # noqa: BLE001
            fails.append(f"LAYERS {want}: status/deck_maps raised {type(e).__name__}: {e}")
        print(f"LAYERS {want} (from {was}): said '{ev[:60]}' | engine {engine_of()}, mode {d.mode}, first sound after "
              f"{'%.0f s' % heard_after if heard_after is not None else 'never (%.0f s)' % (time.time() - t_press)}, then {a2:.1f} s audible of 6 | error {d.last_error}\n   {pic}")
        if not ok or engine_of() != expect or d.mode != expect:
            fails.append(f"LAYERS {want}: engine {engine_of()} / mode {d.mode}, expected {expect}")
        if d.last_error:
            fails.append(f"LAYERS {want}: error {d.last_error}")
        if heard_after is None:
            fails.append(f"LAYERS {want}: nothing heard within {LIMIT:.0f} s of the press")
        elif a2 < 4.0:
            fails.append(f"LAYERS {want}: {a2:.1f} s audible of 6 after the first sound")
    d.stop(0.1)
    for f in fails:
        print("FAIL:", f)
    print("OK" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code or 0)
