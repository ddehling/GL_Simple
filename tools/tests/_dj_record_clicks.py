"""CLICK HUNT at the record player's events: a short headless record session (play_min 30 s, a NEXT pressed once so a
short blend and a hurried swap both happen), the bus rendered to a wav, and at every scheduled event (the next record
starting, the bass swap, the old record out, its stop and unload, the end of the sync) the largest single-sample jump
within +-80 ms is compared with the signal's own jumps over the surrounding second (its 99.9th percentile). A click is
a jump well beyond what the music itself does. Also scans the whole render for the worst isolated jumps.

    python tools/tests/_dj_record_clicks.py --music D:/Devel/music [--wav out.wav] [--minutes 4]

Exit 0 = OK (no event spot over 2.5x the music's own jumps), 1 = FAIL.
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
    minutes = arg("--minutes", 4.0, float)
    wav = arg("--wav", None)
    seed = arg("--seed", None, int)
    old_eq = arg("--old-eq", None)
    if old_eq:
        # A/B: the committed EQ (git show HEAD:lib/dj/eq.py > file) swapped in before any deck is built
        import importlib.util
        import lib.dj.eq as _eqmod
        spec = importlib.util.spec_from_file_location("eq_old", old_eq)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _eqmod.ThreeBandEQ = mod.ThreeBandEQ
        import lib.dj.deck as _deckmod
        _deckmod.ThreeBandEQ = mod.ThreeBandEQ
        print("EQ: the committed version from", old_eq)
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    for k in list(D.PLAY_MIN_S):
        D.PLAY_MIN_S[k] = 30.0
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    if not d.start(threaded=False):
        print("did not start:", d.last_error)
        return 1
    p = d.player
    p.play_min_s = 30.0
    if seed is not None and getattr(p, "brain", None) is not None:
        import random
        p.brain.rng = random.Random(seed)       # the same picks every run (the opener is deterministic already)
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    pcm = []
    events = {}                      # name -> list of (clock, label)
    seen = set()
    clock0 = None
    t0 = time.time()
    nb = 0
    pressed = False
    t_play0 = None
    while True:
        x = np.frombuffer(gen.send(BLOCK), dtype=np.float32).reshape(-1, 2)
        nb += 1
        if clock0 is None:
            clock0 = p.submix.clock - BLOCK
        pcm.append(x.copy())
        d.step()
        # paced to REAL time once the first record sounds (decodes are wall seconds); before that, wait on the clock
        if t_play0 is None:
            if p.record is not None and ((p.submix.telemetry or {}).get("decks") or {}).get(p.record, {}).get("playing"):
                t_play0 = time.time()
                nb0 = nb
            elif time.time() - t0 > 180:
                print("no first record in 180 s:", p.last_error)
                return 1
            else:
                time.sleep(0.02)
                continue
        lag = (nb - nb0) * BLOCK / RATE - (time.time() - t_play0)
        if lag > 0.05:
            time.sleep(min(lag, 0.2))
        c = p.cut
        if c is not None and (c["seq"], c["state"]) not in seen:
            seen.add((c["seq"], c["state"]))
            for key, lbl in (("clock", "next record starts"), ("swap_clock", "the bass swap / the one"), ("end_clock", "old record out, end_sync, rate glide"),
                             ("stop_clock", "old deck stop + unload")):
                if c.get(key) is not None:
                    events.setdefault(f"{c['kind']}#{c['seq']} {lbl}", []).append((int(c[key]), c["state"]))
            if c.get("end_clock") is not None and c["kind"] != "cut":
                from lib.dj.recordplay import GLIDE_S
                events.setdefault(f"{c['kind']}#{c['seq']} rate glide ends (stretcher back to 1.0: a bypass flip?)", []).append((int(c["end_clock"] + GLIDE_S * RATE), c["state"]))
        # one NEXT, pressed while the first blend is under way (a hurried swap) - after 70 s of play
        played = (nb - nb0) * BLOCK / RATE
        if not pressed and played > 70.0:
            pressed = True
            print("NEXT ->", d.next())
        if played >= minutes * 60.0:
            break
        if time.time() - t0 > 900:
            print("timeout")
            break
    d.stop(0.1)
    audio = np.concatenate(pcm)
    m = audio.mean(axis=1).astype(np.float64)
    dif = np.abs(np.diff(m))
    print(f"rendered {len(m) / RATE:.0f} s; {len(events)} event kinds seen")
    if wav:
        import wave
        with wave.open(wav, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(RATE)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        print("wrote", wav)
    fails = []
    rows = []
    for name, lst in events.items():
        clk = lst[-1][0]                      # the last clock the plan held for this event (moves re-post)
        i = clk - clock0
        if i < 0 or i >= len(dif):
            continue
        lo, hi = max(0, i - int(0.08 * RATE)), min(len(dif), i + int(0.08 * RATE))
        wlo, whi = max(0, i - int(0.6 * RATE)), min(len(dif), i + int(0.6 * RATE))
        spot = float(dif[lo:hi].max()) if hi > lo else 0.0
        ref = float(np.percentile(dif[wlo:whi], 99.9)) if whi > wlo else 1.0
        ratio = spot / max(ref, 1e-6)
        rows.append((name, i / RATE, spot, ref, ratio))
        if ratio > 2.5 and spot > 0.02:
            fails.append(f"{name} at {i / RATE:.1f} s: jump {spot:.3f} vs the music's {ref:.3f} ({ratio:.1f}x)")
    for name, t, spot, ref, ratio in sorted(rows, key=lambda r: r[1]):
        print(f"  {t:7.1f} s  {name:48s} jump {spot:.3f}  music {ref:.3f}  {ratio:4.1f}x  {'CLICK?' if ratio > 2.5 and spot > 0.02 else 'ok'}")
    # the whole render: the worst isolated jumps (a kick's attack spans samples; a click is one)
    win = int(0.002 * RATE)
    cs = np.cumsum(np.concatenate([[0.0], dif ** 2]))
    loc = np.sqrt(np.maximum((cs[win:] - cs[:-win]) / win, 1e-12))
    loc = np.concatenate([np.full(win, loc[0]), loc])[:len(dif)]
    ratio = dif / np.maximum(loc, 1e-6)
    worst = np.argsort(-ratio)[:400]
    picks = []
    last = -10 ** 9
    for i in sorted(worst):
        if i - last > int(0.05 * RATE) and dif[i] > 0.05 and ratio[i] > 12:
            picks.append(i)
        last = i
    print(f"whole render: {len(picks)} isolated jumps over 12x their 2 ms surroundings and 0.05 full scale" + (": " + ", ".join(f"{i / RATE:.1f}s ({dif[i]:.2f}, {ratio[i]:.0f}x)" for i in picks[:12]) if picks else ""))
    for f in fails:
        print("FAIL:", f)
    print("OK" if not fails else f"FAIL ({len(fails)})")
    return 0 if not fails else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code or 0)
