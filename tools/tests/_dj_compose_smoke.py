"""The compose smoke (docs/DJ_THE_COMPOSER.md): the Director left alone at MOMENTS some for a rendered stretch -
every record must get a score, the score's events must fire (a guest vocal, a guest bassline, a build, the
takeover exit) or end with a reason, nothing silently; no dead bars.

    python tools/tests/_dj_compose_smoke.py --music D:/Devel/music [--seed 1] [--seconds 720] [--moments some|lots]
"""
import os
import random
import sys
import time

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.tests._low_priority import PRIORITY  # noqa: E402
from lib.dj.submix import RATE  # noqa: E402

BLOCK = 4410
PACE = 1.5


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from lib.audio_engine import AudioEngine
    from lib.dj import director as D
    music = arg("--music", "D:/Devel/music")
    seed = arg("--seed", 1, int)
    seconds = arg("--seconds", 720.0, float)
    moments = arg("--moments", "some")
    D.PLAY_MIN_S["short"] = 90.0
    engine = AudioEngine()
    d = D.Director(music, engine=None, theme="groove", attach=engine.attach_track)
    d.set_dial("layers", "two")
    d.set_dial("pace", "short")
    d.set_dial("moments", moments)
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
    t0 = len(audio)
    n_moves = 0
    t_end = t0 + int(seconds * RATE / BLOCK)
    fired = {"acapella": 0, "guest": 0, "build": 0, "takeover": 0}
    while len(audio) < t_end:
        pump(bar_blocks())
        for m in rc.move_log[n_moves:]:
            txt = m["text"]
            if m["kind"] in ("score", "order", "event_plan", "bed_to", "open") or txt.startswith(("ACAPELLA", "GUEST BASS", "BUILD", "TAKEOVER", "the event")):
                print(f"  {m.get('clock_s', 0):6.0f}s {m['kind']:10s} {txt[:130]}")
            if m["kind"] == "moment":
                for k, word in (("acapella", "ACAPELLA:"), ("guest", "GUEST BASS:"), ("build", "BUILD:"), ("takeover", "TAKEOVER:")):
                    if txt.startswith(word) and "over" not in txt.split(":")[0]:
                        fired[k] += 1
        n_moves = len(rc.move_log)
        if d.last_error or rc.last_error:
            print(f"  error: {d.last_error or rc.last_error}")
            break
    d.stop(0.1)
    mono = np.concatenate(audio[t0:]).mean(axis=1) if len(audio) > t0 + 10 else np.zeros(1)
    nb = int(bar_blocks() * BLOCK)
    rms = [20 * np.log10(np.sqrt(np.mean(mono[i:i + nb] ** 2)) + 1e-9) for i in range(0, len(mono) - nb, nb)]
    sil = [s["clock_s"] for s in getattr(rc, "silence_log", []) if s.get("clock_s") is not None]
    dead = [i for i, r in enumerate(rms) if r < -45.0 and not any(abs((t0 * BLOCK + (i + 1) * nb) / RATE - c) < 2.0 * nb / RATE for c in sil)]
    orders = getattr(rc, "orders", []) or []
    scores = [m for m in rc.move_log if m["kind"] == "score" and m["text"].startswith("the score of")]
    ended = {st: [o for o in orders if o["state"] == st] for st in ("done", "failed", "refused", "fixed", "staging", "running")}
    records = sum(1 for m in rc.move_log if m["kind"] in ("open", "bed_to") and ("takes the clock" in m["text"] or "opens on" in m["text"] or "the bed passes" in m["text"] or "the record is its" in m["text"]))
    # the event kinds that actually SOUNDED (moment moves), from the move log
    fired = {"acapella": sum(1 for m in rc.move_log if m["kind"] == "moment" and m["text"].startswith("ACAPELLA: ")),
             "guest": sum(1 for m in rc.move_log if m["kind"] == "moment" and m["text"].startswith("GUEST BASS: the")),
             "build": sum(1 for m in rc.move_log if m["kind"] == "moment" and m["text"].startswith("BUILD: the eight bars")),
             "takeover": sum(1 for m in rc.move_log if m["kind"] == "moment" and m["text"].startswith("TAKEOVER: "))}
    print(f"\nrecords {records}, scores composed {len(scores)}, orders: " + ", ".join(f"{k} {len(v)}" for k, v in ended.items() if v))
    print(f"events sounded: {fired}; dead bars {len(dead)} of {len(rms)}")
    for o in ended["failed"] + ended["refused"]:
        print(f"  {o['state']}: {o['kind']} on {d.title(o['record_tid'])[:22]}: {o.get('why', '')[:110]}")
    silent = [o for o in ended["failed"] + ended["refused"] if not o.get("why")]
    fails = []
    if not scores:
        fails.append("no record got a score")
    if sum(fired.values()) == 0:
        fails.append("no composed event sounded")
    if silent:
        fails.append(f"{len(silent)} orders ended without a reason")
    if dead:
        fails.append(f"{len(dead)} dead bars")
    if d.last_error or rc.last_error:
        fails.append(f"error: {d.last_error or rc.last_error}")
    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
