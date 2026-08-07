"""A/B the EXIT-POINT choice: does the play-time budget push seams into the
comedown, and does softening it help?

Two candidate changes, measured separately and together:

  soft   the drawn play budget stops being a HARD filter on best_pair's
         out candidates and becomes a penalty. A great exit 40s early
         beats a dead one on time. (Today: `outs = [o for o in
         cur.mix_outs if o["time_s"] >= after_s]`, and when nothing
         survives - 18-35% of seams on this library - plan_transition
         falls back to `duration - 35s`, which is the comedown by
         construction.)

  win    the "leave while the music is alive" energy factor measures the
         section AFTER out_s. The blend COMPLETES at out_s, so A actually
         plays the section BEFORE it; measured correlation between the
         two on this library is -0.06. This variant scores the real blend
         window off the 2 Hz energy curve instead.

NOTHING IN THE LIVE PATH IS TOUCHED. The variants are built by rewriting
brain.best_pair's own source at import time (three anchored substitutions,
each asserted to match exactly once) - so this harness tracks brain.py
instead of drifting from a hand-copied body. If an anchor stops matching,
this file fails loudly rather than measuring stale logic.

    # silent, whole-library, seconds - how big is the change?
    python tools/tests/_dj_exit_ab.py --music D:/Devel/music

    # then put it on your ears: same pair, same style, two exit points
    python tools/tests/_dj_exit_ab.py --music D:/Devel/music --render 6

Renders land in logs/exit_ab/ as matched pairs plus a manifest. Play them
back to back - this is a SAME-PAIR comparison seconds apart, so it does
not have the cross-seam memory problem seamprobe.py warns about.
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

from lib.dj.brain import Brain, load_library          # noqa: E402
from lib.dj.db import LibraryDB                       # noqa: E402
from lib.dj.themes import get_theme                   # noqa: E402
# The variant machinery lives in lib/dj/exitvariants.py so the planner's
# Exit Compare tab and this harness cannot drift apart.
from lib.dj.exitvariants import (BASELINE, BLEND_S, CLAMP_FRAC,   # noqa: E402
                                 VARIANTS,
                                 blend_rel as _blend_rel,
                                 build_best_pair as build_variant,
                                 fallback_out, last_groove_end)

RATE = 44100
THROTTLE_S = 2.0        # breathing room between renders (see knobsweep)


# ---------------------------------------------------------------- measuring
def outro_share(t, out_s):
    """Fraction of the blend window sitting in outro/breakdown material."""
    a, b = out_s - BLEND_S, out_s
    tot = bad = 0.0
    for s in t.sections or []:
        ov = min(s["end_s"], b) - max(s["start_s"], a)
        if ov <= 0:
            continue
        tot += ov
        if s.get("kind") in ("outro", "breakdown"):
            bad += ov
    return (bad / tot) if tot > 0 else 0.0


def describe(t, out_s):
    lge = last_groove_end(t)
    return {"pos": out_s / t.duration_s,
            "past_groove": (out_s - lge) if lge else None,
            "outro_share": outro_share(t, out_s),
            "blend_rel": _blend_rel(t, out_s)}


def draw_budget(theme, heat, rng):
    """system.py _draw_exit, arc-coupled."""
    span = max(theme.max_play_s - theme.min_play_s, 1.0)
    frac = min(1.0, rng.random() * (1.0 - 0.55 * heat) + 0.25 * (1.0 - heat))
    return theme.min_play_s + frac * span


def run_sweep(lib, theme, heats, n_per, seed, want_shifts=0):
    fns = {k: (Brain.best_pair if not flags else build_variant(flags))
           for k, (flags, _) in VARIANTS.items()}
    rng = random.Random(seed)
    brain = Brain(lib, theme, seed=seed)
    acc = {k: [] for k in VARIANTS}
    fb = {k: 0 for k in VARIANTS}
    shifts = []
    pool = [t for t in lib if t.duration_s >= 150 and t.mix_outs and t.sections]
    n = 0
    for heat in heats:
        for _ in range(n_per):
            cur = rng.choice(pool)
            cand, meta = brain.choose_next(cur, heat, cur.bpm)
            if cand is None:
                continue
            budget = draw_budget(theme, heat, rng)
            # system.py: after = pos + max(budget - played, MIN_LEAD),
            # capped at deadline = duration - 25
            after = min(budget, cur.duration_s - 25.0)
            n += 1
            row = {}
            for name, fn in fns.items():
                aft = min(after, cur.duration_s * CLAMP_FRAC) \
                    if VARIANTS[name][1] else after
                p = fn(brain, cur, cand, after_s=aft)
                if p is None:
                    fb[name] += 1
                    out_s = fallback_out(cur, aft)
                else:
                    out_s = p["out_s"]
                row[name] = out_s
                acc[name].append(describe(cur, out_s))
            if want_shifts:
                d = row[BASELINE] - row["current"]
                if d > 1.0:
                    shifts.append({"a": cur, "b": cand, "meta": meta,
                                   "after": after, "shift": d,
                                   "cur_out": row[BASELINE],
                                   "new_out": row["current"]})
    shifts.sort(key=lambda s: -s["shift"])
    return acc, fb, n, shifts[:want_shifts]


def report(acc, fb, n):
    print(f"\n{'variant':<10} {'pos med':>8} {'pos p90':>8} "
          f"{'past-groove':>12} {'outro>50%':>10} {'blend/body':>11} "
          f"{'fallback':>9}")
    print("-" * 74)
    for name in VARIANTS:
        rs = acc[name]
        if not rs:
            continue
        pos = np.array([r["pos"] for r in rs])
        pg = np.array([r["past_groove"] for r in rs
                       if r["past_groove"] is not None], dtype=float)
        osh = np.array([r["outro_share"] for r in rs])
        br = np.array([r["blend_rel"] for r in rs
                       if r["blend_rel"] is not None], dtype=float)
        print(f"{name:<10} {np.median(pos):8.2f} {np.percentile(pos,90):8.2f} "
              f"{100*np.mean(pg > 0):11.1f}% {100*np.mean(osh > 0.5):9.1f}% "
              f"{np.median(br):11.2f} {100*fb[name]/max(n,1):8.1f}%")
    print(f"\n  ({n} seams; 'past-groove' = exit lands after the track's last "
          f"groove/build section)")
    # the COST: softening the budget shortens play time. Quantify it.
    cur_pos = np.array([r["pos"] for r in acc[BASELINE]])
    new_pos = np.array([r["pos"] for r in acc["current"]])
    print(f"\n  play-time cost of the live build vs {BASELINE}: exits "
          f"move earlier by a median "
          f"{100*(np.median(cur_pos) - np.median(new_pos)):.1f}% of track "
          f"length\n  (that is the pacing the budget was buying - judge it "
          f"against the comedown it was spending)")


# ---------------------------------------------------------------- rendering
def render_pairs(db, lib, theme, shifts, outdir, seed):
    from lib.dj.audition import render_seam
    import soundfile as sf
    os.makedirs(outdir, exist_ok=True)
    brain = Brain(lib, theme, seed=seed)
    manifest = []
    for i, s in enumerate(shifts, 1):
        a, b = s["a"], s["b"]
        slug = "".join(c if c.isalnum() else "_" for c in a.title)[:34]
        # ONE plan, so style/beats/rate are identical - the ONLY difference
        # between the two renders is out_s. Anything else and this stops
        # being an A/B of the exit point.
        base = brain.plan_transition(a, b, s["meta"], after_s=s["after"])
        print(f"\n[{i}/{len(shifts)}] {a.title} -> {b.title}")
        print(f"    style={base['style']} beats={base['beats']}  "
              f"exit moves {s['cur_out']:.0f}s -> {s['new_out']:.0f}s "
              f"({s['shift']:.0f}s earlier, "
              f"{100*s['cur_out']/a.duration_s:.0f}% -> "
              f"{100*s['new_out']/a.duration_s:.0f}% of track)")
        rec = {"n": i, "a": a.title, "b": b.title, "style": base["style"],
               "shift_s": round(s["shift"], 1),
               "cur_out_s": round(s["cur_out"], 1),
               "new_out_s": round(s["new_out"], 1),
               "files": {}}
        for tag, out_s in (("A_current", s["cur_out"]),
                           ("B_proposed", s["new_out"])):
            plan = dict(base)
            plan["out_s"] = a.nearest_phrase(out_s)
            name = f"{i:02d}_{slug}_{tag}.wav"
            try:
                print(f"    rendering {tag}...", flush=True)
                audio = render_seam(db, a, b, plan)
                sf.write(os.path.join(outdir, name), audio, RATE)
                rec["files"][tag] = name
            except Exception as e:
                print(f"    !! {tag} failed: {e}")
                rec["files"][tag] = f"FAILED: {e}"
            time.sleep(THROTTLE_S)
        manifest.append(rec)
    mf = os.path.join(outdir, "manifest.json")
    with open(mf, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(manifest)} pairs to {outdir}")
    print("Play each NN_*_A_current.wav then NN_*_B_proposed.wav back to "
          "back.\nThe only difference is where A leaves.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", default="D:/Devel/music")
    ap.add_argument("--theme", default="groove")
    ap.add_argument("--seams", type=int, default=200,
                    help="seams per arc position (3 positions sampled)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--render", type=int, default=0,
                    help="render this many A/B pairs (biggest exit shifts)")
    args = ap.parse_args()

    db = LibraryDB(args.music)
    lib = load_library(db)
    theme = get_theme(args.theme)
    print(f"library: {len(lib)} tracks   theme: {theme.name} "
          f"(min_play {theme.min_play_s:.0f}s, max_play {theme.max_play_s:.0f}s)")

    acc, fb, n, shifts = run_sweep(lib, theme, (0.15, 0.5, 0.9), args.seams,
                                   args.seed, want_shifts=args.render)
    report(acc, fb, n)

    if args.render:
        outdir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "logs", "exit_ab")
        render_pairs(db, lib, theme, shifts, outdir, args.seed)


if __name__ == "__main__":
    main()
