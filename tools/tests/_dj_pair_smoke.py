"""THE PAIR SMOKE (docs/DJ_MORE_THAN_A_MIXER.md 1): the pairs path driven end to end, headless - the Director starts
layered, the best pair whose record is not the record playing is taken (PLAY THIS PAIR), and the run is pumped
until that record plays and the pair's vocal comes over its groove. Three rounds of fixing the path from the
user's logs after the fact (2026-09-20) is what this replaces: it fails on the first thing that does not happen.

    python tools/tests/_dj_pair_smoke.py --music D:/Devel/music [--seed 1] [--seconds 900]

No audio device: the mixer is pulled by hand, paced at 1.5 x real time (the decodes are real seconds).
"""
import os
import random
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410
PACE = 1.5


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    music = arg("--music", "D:/Devel/music")
    seed = arg("--seed", 1, int)
    seconds = arg("--seconds", 900.0, float)
    D.PLAY_MIN_S["short"] = 60.0
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    d.set_dial("layers", "two")
    d.set_dial("pace", "short")
    d.set_dial("moments", "rare")                     # the DJ proposes nothing: only the pair
    if not d.start(threaded=False):
        print(f"FAIL: did not start: {d.last_error}")
        return 1
    rc = d.rc
    rc.rng = random.Random(seed)
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    audio = []
    pace = {"t0": None, "n0": 0}

    def pump(n):
        for _ in range(n):
            audio.append(np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2).copy())
            d.step()
            if rc.master is not None:
                if pace["t0"] is None:
                    pace["t0"], pace["n0"] = time.time(), len(audio)
                lag = (len(audio) - pace["n0"]) * BLOCK / RATE / PACE - (time.time() - pace["t0"])
                if lag > 0.05:
                    time.sleep(min(lag, 0.2))

    def bar_blocks():
        return max(1, int(rc._bar_clock() / BLOCK)) if rc.master else 20

    t_wait = time.time()
    while time.time() - t_wait < 120.0:
        pump(1)
        if rc.master is not None and (rc._tel_deck(rc.master) or {}).get("playing"):
            break
    if rc.master is None:
        print("FAIL: no first song")
        return 1
    pump(2 * bar_blocks())
    rec0 = rc.songs[rc.record_deck].track
    pairs = [p for p in d.pairs(n=40, fit_clock=True) if p["kind"] == "vocal" and p["r"].id != rec0.id and not p["played"]]
    if not pairs:
        print("FAIL: no vocal pair inside the wall")
        return 1
    p = pairs[0]
    print(f"record now: {rec0.title[:30]} ({rec0.bpm:.0f})")
    print(f"pair: {p['words']}")
    msg = d.drop_pair(p)
    print(f"PLAY THIS PAIR -> {msg}")
    t0 = len(audio)
    seen = {"record": None, "event_plan": None, "armed": None, "vocal_in": None, "fail": None}
    n_moves = len(rc.move_log)
    t_end = t0 + int(seconds * RATE / BLOCK)
    while len(audio) < t_end and seen["vocal_in"] is None and seen["fail"] is None:
        pump(bar_blocks())
        for m in rc.move_log[n_moves:]:
            txt = m["text"]
            clk = f"{m.get('clock_s', 0):.0f}s"
            if m["kind"] in ("bed_to", "open", "moment") or "your pair" in txt or "event" in txt or "the seam" in txt or "ACAPELLA" in txt or "IDEA" in txt:
                print(f"  {clk:>6} {m['kind']:10s} {txt[:120]}")
            if m["kind"] == "bed_to" and p["r"].title[:20] in txt and seen["record"] is None:
                seen["record"] = clk
                t_rec = float((rc._tel_deck(rc.record_deck) or {}).get("time_s") or 0.0)
                print(f"         -> {p['r'].title[:24]} is the record at its {int(t_rec) // 60}:{int(t_rec) % 60:02d} (the pair's groove {int(p['r_start']) // 60}:{int(p['r_start']) % 60:02d}-{int(p['r_end'] or 0) // 60}:{int(p['r_end'] or 0) % 60:02d}); decks: "
                      + ", ".join(f"{k}={s.track.title[:12]}{'(staged)' if not s.entered else ''}" for k, s in rc.songs.items()))
            if txt.startswith("event in") and "your pair" in txt:
                seen["event_plan"] = clk
            if txt.startswith("the event:") and "ACAPELLA" in txt:
                seen["armed"] = clk
            if txt.startswith("ACAPELLA") and ("vocal" in txt.lower()) and m["kind"] == "moment":
                seen["vocal_in"] = clk
            if "your pair could not be placed" in txt or "your pair missed" in txt or "the event is dropped" in txt or "the event refused" in txt:
                seen["fail"] = txt[:140]
        n_moves = len(rc.move_log)
        if d.last_error or rc.last_error:
            seen["fail"] = f"error: {d.last_error or rc.last_error}"
    d.stop(0.1)
    mono = np.concatenate(audio[t0:]).mean(axis=1) if len(audio) > t0 + 10 else np.zeros(1)
    nb = int(bar_blocks() * BLOCK)
    rms = [20 * np.log10(np.sqrt(np.mean(mono[i:i + nb] ** 2)) + 1e-9) for i in range(0, len(mono) - nb, nb)]
    sil = [s["clock_s"] for s in getattr(rc, "silence_log", []) if s.get("clock_s") is not None]
    dead = [i for i, r in enumerate(rms) if r < -45.0 and not any(abs((t0 * BLOCK + (i + 1) * nb) / RATE - c) < 2.0 * nb / RATE for c in sil)]
    print(f"\nrecord played at {seen['record']}, the pair planned at {seen['event_plan']}, armed at {seen['armed']}, the vocal in at {seen['vocal_in']}; dead bars {len(dead)} of {len(rms)}")
    if seen["fail"]:
        print(f"FAIL: {seen['fail']}")
        return 1
    if seen["vocal_in"] is None:
        print("FAIL: the pair's vocal never came in" + ("" if seen["record"] else " (its record never played)"))
        return 1
    if dead:
        print(f"FAIL: {len(dead)} dead bars")
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
