"""Estimate execution-knob values from RENDERED AUDIO, no ears needed.

The Seam Lab asks a human about the ~9 knobs that are matters of taste.
The other ~30 mostly have measurable consequences - level holes, abrupt
lurches, low-end pile-ups, clipping - which is exactly what this sweeps:

  for each knob: render the SAME handful of seams at several values
  across its range, score every render with objective metrics, and see
  which value the measurements prefer.

Because the same seams are rendered at every value, the pair-to-pair
variance that made verdict statistics hopeless cancels exactly - this is
the A/B design that is impractical for a listener but free for a machine.

A knob whose scores are FLAT across its range is reported as such: the
metrics cannot decide it, so either leave the default or promote it to
the Seam Lab's taste questions (seamprobe.promote).

Usage:
    python tools/dj/dj_knobsweep.py --music D:/Devel/music --knob fade_out_ramp
    python tools/dj/dj_knobsweep.py --music D:/Devel/music --deferred
    ... --apply     # write winners into logs/seam_tuning.json
Results land in logs/knob_sweep.json either way.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

RATE = 44100
N_VALUES = 5           # points across the knob's range
N_SEAMS = 4            # fixed seams rendered per point
FLAT_DB = 0.8          # score spread below this = "metrics can't decide"
APPLY_MARGIN = 0.5     # winner must beat the default by this to --apply


def metrics(x):
    """Objective badness of one rendered seam, in comparable dB-ish units.

    dead   - deepest 0.5s hole vs the median level (0 = none)
    lurch  - largest second-to-second level step (abrupt = bad)
    lowex  - low-band (<120 Hz) pile-up over its own median (double bass)
    clip   - samples at full scale
    """
    mono = x.mean(axis=1)
    n = len(mono)
    w = RATE // 2
    hops = max((n - w) // (RATE // 4), 1)
    rms = np.array([np.sqrt(np.mean(mono[i * (RATE // 4):
                                         i * (RATE // 4) + w] ** 2)) + 1e-9
                    for i in range(hops)])
    med = np.median(rms)
    dead = max(0.0, 20 * np.log10(med / rms.min()))
    sec = np.array([np.sqrt(np.mean(mono[i * RATE:(i + 1) * RATE] ** 2))
                    + 1e-9 for i in range(max(n // RATE, 1))])
    lurch = float(np.abs(np.diff(20 * np.log10(sec))).max()) \
        if len(sec) > 1 else 0.0
    # low band per second, via rFFT
    lows = []
    for i in range(max(n // RATE, 1)):
        seg = mono[i * RATE:(i + 1) * RATE]
        if len(seg) < RATE // 2:
            continue
        sp = np.abs(np.fft.rfft(seg))
        cut = int(120 * len(seg) / RATE)
        lows.append(np.sqrt(np.mean(sp[:cut] ** 2)) + 1e-9)
    lows = np.array(lows) if lows else np.array([1.0])
    lowex = max(0.0, 20 * np.log10(lows.max() / np.median(lows)) - 6.0)
    clip = int(np.sum(np.abs(x) >= 0.999))
    return {"dead": round(float(dead), 2), "lurch": round(lurch, 2),
            "lowex": round(float(lowex), 2), "clip": clip}


def badness(m):
    """One number, lower is better. Weights are coarse on purpose - this
    ranks values of ONE knob against each other on the SAME seams, so
    only differences matter."""
    return (max(0.0, m["dead"] - 8.0)          # holes deeper than 8 dB
            + 0.8 * max(0.0, m["lurch"] - 6.0)  # steps sharper than 6 dB
            + 0.6 * m["lowex"]
            + 0.1 * min(m["clip"], 50))


def build_seams(db, lib, knob, rng):
    """Fixed (a, b, plan) trios whose style actually reads `knob`."""
    from lib.dj.brain import Brain
    from lib.dj.themes import get_theme
    from tools.dj.planner.seamtune import styles_reading
    brain = Brain(lib, get_theme("groove"), seed=rng.randrange(1 << 30))
    styles = styles_reading(knob)
    out = []
    tries = 0
    while len(out) < N_SEAMS and tries < 300:
        tries += 1
        cur = rng.choice(lib)
        if cur.duration_s < 150:
            continue
        cand, meta = brain.choose_next(cur, 0.6, cur.bpm)
        if cand is None:
            continue
        want = rng.choice(styles)
        plan = brain.plan_transition(cur, cand, meta,
                                     after_s=cur.duration_s * 0.5,
                                     force_style=want, test_gates=True)
        if plan["style"] in styles:
            out.append((cur, cand, plan))
    return out


def sweep_knob(db, lib, knob, rng, log=print):
    from lib.dj.audition import render_seam
    from tools.dj.planner.seamtune import RANGES, apply_plan_knobs
    from lib.dj.brain import TUNE_DEFAULTS
    lo, hi = RANGES[knob]
    values = [round(lo + (hi - lo) * i / (N_VALUES - 1), 4)
              for i in range(N_VALUES)]
    default = TUNE_DEFAULTS[knob]
    seams = build_seams(db, lib, knob, rng)
    if not seams:
        log(f"  {knob}: no seams landed a style that reads it - skipped")
        return None
    log(f"  {knob}: {len(seams)} seams x {N_VALUES} values "
        f"({values[0]}..{values[-1]}, default {default})")
    scores = {}
    for v in values:
        tot = 0.0
        for (a, b, plan) in seams:
            p = dict(plan, tune={knob: v})
            apply_plan_knobs(p)
            audio = render_seam(db, a, b, p)
            tot += badness(metrics(audio))
        scores[v] = round(tot / len(seams), 3)
        log(f"     {knob}={v:<8} badness {scores[v]}")
    best = min(scores, key=scores.get)
    spread = max(scores.values()) - min(scores.values())
    # score at (or nearest to) the default, for the apply margin
    near_def = min(scores, key=lambda v: abs(v - default))
    flat = spread < FLAT_DB
    log(f"     -> best {best} (spread {spread:.2f} "
        + ("FLAT - metrics can't decide; taste question or leave default"
           if flat else f"vs default-ish {near_def}={scores[near_def]}") + ")")
    return {"knob": knob, "scores": {str(k): v for k, v in scores.items()},
            "best": best, "default": default, "flat": flat,
            "spread": round(spread, 3),
            "gain_vs_default": round(scores[near_def] - scores[best], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", required=True)
    ap.add_argument("--knob")
    ap.add_argument("--deferred", action="store_true",
                    help="sweep every knob the Seam Lab does not ask about")
    ap.add_argument("--apply", action="store_true",
                    help="write clear winners into the live tuning")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    from lib.dj.brain import load_library
    from lib.dj.db import LibraryDB
    from tools.dj.planner.seamprobe import PRIORITY, QUESTIONS
    from tools.dj.planner.seamtune import RANGES
    db = LibraryDB(args.music)
    lib = [t for t in load_library(db) if not t.excluded]
    print(f"library: {len(lib)} tracks")

    if args.knob:
        knobs = [args.knob]
    elif args.deferred:
        knobs = [k for k in RANGES if k in QUESTIONS and k not in PRIORITY]
    else:
        ap.error("--knob NAME or --deferred")
    rng = random.Random(args.seed)
    results, t0 = [], time.time()
    for i, k in enumerate(knobs):
        print(f"[{i + 1}/{len(knobs)}] sweeping {k} "
              f"({(time.time() - t0) / 60:.0f} min elapsed)")
        r = sweep_knob(db, lib, k, rng)
        if r:
            results.append(r)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "logs", "knob_sweep.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"t": time.time(), "results": results}, f, indent=2)
    print(f"\nwrote {out}")

    decisive = [r for r in results if not r["flat"]
                and r["gain_vs_default"] >= APPLY_MARGIN]
    flat = [r for r in results if r["flat"]]
    print(f"\n{len(decisive)} knobs where the metrics prefer a non-default "
          f"value; {len(flat)} flat (candidates for taste or leave-alone)")
    for r in decisive:
        print(f"   {r['knob']}: {r['default']} -> {r['best']} "
              f"(saves {r['gain_vs_default']} badness)")
    if args.apply and decisive:
        from lib.dj import tuning
        for r in decisive:
            tuning.set_value(r["knob"], r["best"],
                             why=f"knob sweep: badness "
                                 f"-{r['gain_vs_default']}")
            print(f"   applied {r['knob']} = {r['best']}")


if __name__ == "__main__":
    main()
