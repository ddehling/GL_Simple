"""THE PREVIEW SMOKE (docs/DJ_MORE_THAN_A_MIXER.md 1): the best pair of the pairing file with stems on both sides is
rendered offline - eight bars, the vocal (or bassline) stretched and shifted onto the record's groove - and written
to logs/preview_smoke.wav; its length and peak are checked. No device is opened, ever.

    python tools/tests/_dj_preview_smoke.py --music D:/Devel/music
"""
import os
import sys
import time

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_DECODE_INPROC"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    import json
    import numpy as np
    from lib.dj.db import LibraryDB
    from lib.dj import brain as B
    from lib.dj.preview import render_pair, write_wav
    from lib.dj.submix import RATE
    music = arg("--music", "D:/Devel/music")
    db = LibraryDB(music)
    lib = {t.id: t for t in B.load_library(db) if getattr(t, "has_stems", False)}
    p = json.load(open(os.path.join(music, ".stems", "pairings.json"), encoding="utf-8"))
    best = None
    for rid, r in (p.get("records") or {}).items():
        R = lib.get(int(rid))
        if R is None:
            continue
        for kind, key in (("vocal", "vocal_over"), ("bass", "bass_under")):
            for x in r.get(key) or []:
                S = lib.get(int(x["id"]))
                if S is None or S.id == R.id or S.title == R.title:
                    continue
                sc = float(x.get("score") or 0.0)
                if best is None or sc > best[0]:
                    g0 = float((r.get("grooves") or [[30.0]])[0][0])
                    r_start = float((x.get("groove") or [g0])[0]) if kind == "vocal" else g0
                    s_start = float((x.get("sung") or [0.0])[0]) if kind == "vocal" else float(x.get("landmark") or 0.0)
                    best = (sc, kind, R, S, r_start, s_start, int(x.get("shift") or 0))
    if best is None:
        print("no pair with stems on both sides")
        return 1
    sc, kind, R, S, r_start, s_start, shift = best
    print(f"pair: {kind} - {S.title[:30]} ({S.bpm:.0f}) onto {R.title[:30]} ({R.bpm:.0f}) from {r_start:.0f}s / {s_start:.0f}s, shift {shift:+d}, score {sc:.2f}")
    t0 = time.time()
    buf = render_pair(music, kind, R, S, r_start, s_start, bars=8, shift=shift)
    dt = time.time() - t0
    want = 8 * 4 * 60.0 / R.bpm
    got = len(buf) / RATE
    peak = float(np.max(np.abs(buf)))
    rms = float(np.sqrt(np.mean(buf ** 2)))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "preview_smoke.wav")
    write_wav(out, buf)
    ok = abs(got - want) < 0.05 * want and 0.5 <= peak <= 0.91 and rms > 0.01
    print(f"rendered {got:.2f} s (want {want:.2f}) in {dt:.1f} s; peak {peak:.3f}, rms {rms:.3f}; wrote {out}")
    print("ALL OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
