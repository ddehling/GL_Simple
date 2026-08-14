"""cut_at_drop ENTRY GATE: is the style actually cutting into real drops?

The style's whole premise is that B's drop lands bare at the cut. Three
separate defects have hidden behind that claim, each invisible to the check
that came before it:

  PRECISION  drop_moments() labels a boundary from SECTION MEANS, which is
             the segmenter's opinion. Measured over 1534 labelled drops on
             the reference library: median x1.57 and 13% with no audible
             step at all. "The cut aligns with a labelled drop" was true
             and meant nothing.
  RECALL     the entry used to come from a `pre_drop` MIX-IN hint, and mix
             points answer "where can this track be brought IN" - 61% sit
             in the first quarter of the song, none past three quarters,
             while the real steps have a median position of 0.47. The style
             could not reach a mid-song drop, so it took an intro build.
  SHAPE      a step RATIO cannot tell a drop from a swell: breakdown->slam
             and quiet->mid score the same. A drop lands HOT, off a real
             dip, with the KICK coming back - that last one is beat power,
             not broadband loudness, and it was rising on none of the
             entries when it was first measured (x1.02, falling on 32%).

So this gate asserts the SHAPE of the entries the brain actually plans,
not the label it aligned to. Run it after touching _drop_entries, the
cut_at_drop gate, drop_step, or anything in the analysis that feeds
energy_curve / beat power.

Usage:
    python tools/tests/_dj_cutdrop_test.py --music D:/Devel/music
    python tools/tests/_dj_cutdrop_test.py --music ... --want 40
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from lib.dj import beatpower as bp
from lib.dj.brain import (_CUT_DROP_MAX_BEFORE, _CUT_DROP_MIN_AFTER,
                          _CUT_DROP_MIN_STEP, Brain, drop_levels, drop_step,
                          load_library)
from lib.dj.db import LibraryDB
from lib.dj.themes import get_theme

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


# levels come from the ENGINE (brain.drop_levels), never recomputed here:
# an independent reimplementation disagreed in the last digit and failed a
# landing at 0.6499 that the engine had passed at 0.65.
levels = drop_levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", required=True)
    ap.add_argument("--want", type=int, default=25,
                    help="entries to collect before judging")
    ap.add_argument("--tries", type=int, default=1200)
    args = ap.parse_args()

    lib = load_library(LibraryDB(args.music))
    print(f"library {len(lib)} tracks | collecting {args.want} planned "
          f"cut_at_drop entries\n")
    rng = random.Random(5)
    brain = Brain(lib, get_theme("groove"), seed=5)
    brain.veto_ids = set()

    rows = []
    for _ in range(args.tries):
        if len(rows) >= args.want:
            break
        cur = rng.choice(lib)
        if cur.duration_s < 120:
            continue
        brain.note_played(cur)
        cand, meta = brain.choose_next(cur, 0.6, cur.bpm)
        if cand is None:
            continue
        brain.note_played(cand)
        plan = brain.plan_transition(
            cur, cand, meta, after_s=cur.duration_s * rng.uniform(.35, .65),
            force_style="cut_at_drop")
        if plan["style"] != "cut_at_drop":
            continue
        at = plan["in_s"]
        lv = levels(cand, at)
        if lv is None:
            continue
        per = max(cand.period_s, 0.3)
        rows.append({
            "pair": f"{cur.title[:20]} -> {cand.title[:20]}",
            "step": drop_step(cand, at) or 0.0,
            "before": lv[0], "after": lv[1],
            "pos": at / max(cand.duration_s, 1e-6),
            "pw_b": bp.power_at(cand.id, max(at - 4 * per, 0.0)),
            "pw_a": bp.power_at(cand.id, at + 4 * per),
        })

    if not rows:
        print("  [FAIL] no cut_at_drop seam could be planned at all")
        sys.exit(1)

    st = np.array([r["step"] for r in rows])
    bf = np.array([r["before"] for r in rows])
    af = np.array([r["after"] for r in rows])
    pos = np.array([r["pos"] for r in rows])
    pw = [(r["pw_b"], r["pw_a"]) for r in rows
          if r["pw_b"] is not None and r["pw_a"] is not None]

    print(f"  {len(rows)} entries | step med x{np.median(st):.2f} | "
          f"before {np.median(bf):.2f} | after {np.median(af):.2f} | "
          f"position {np.median(pos):.2f} of track\n")

    check("every entry is a real step up",
          bool((st >= _CUT_DROP_MIN_STEP).all()),
          f"weakest x{st.min():.2f} (bar x{_CUT_DROP_MIN_STEP})")
    check("every entry LANDS hot",
          bool((af >= _CUT_DROP_MIN_AFTER).all()),
          f"coldest landing {af.min():.2f} of the track's p95 "
          f"(bar {_CUT_DROP_MIN_AFTER})")
    check("every entry comes off a real dip",
          bool((bf <= _CUT_DROP_MAX_BEFORE).all()),
          f"loudest run-up {bf.max():.2f} (bar {_CUT_DROP_MAX_BEFORE})")
    # RECALL: the failure mode was every entry sitting in the intro. A
    # median past a third of the track is the cheap witness that the scan
    # reaches where drops actually live.
    check("entries reach past the intro region",
          float(np.median(pos)) > 0.33,
          f"median entry at {np.median(pos):.2f} of the track "
          f"(pre_drop hints sat at 0.20)")
    if pw:
        ratio = np.array([a / max(b, 1e-3) for b, a in pw])
        check("the kick comes back",
              bool((ratio >= 1.0).all()),
              f"worst beat-power ratio x{ratio.min():.2f} over "
              f"{len(pw)} measured entries (med x{np.median(ratio):.2f})")
    else:
        print("  [warn] no beat-power measurements - kick check skipped")

    print()
    for r in sorted(rows, key=lambda r: r["step"])[:5]:
        print(f"    weakest: {r['pair']:<44} x{r['step']:.2f}  "
              f"{r['before']:.2f}->{r['after']:.2f}  @{r['pos']:.2f}")

    if failures:
        print(f"\nFAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("\nALL PASS")


if __name__ == "__main__":
    main()
