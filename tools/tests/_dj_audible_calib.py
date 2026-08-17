"""Calibrate the live audible-flam meter WITHOUT playing a night at 1x.

Born 2026-08-16 from the operator's question: "why isn't this the kind
of thing we can simulate?" It is. The submix renders offline much
faster than realtime with the PLL and the wide audible meter running
identically (the 08-16 disaster was reproduced exactly this way), so
the meter can be judged against this repo's independent offline
instrument - the env-xcorr kick lag _dj_quality_test.render_seam has
always measured - over as many brain-planned seams as we care to
render. What live nights (or a Lab listen) still uniquely supply is
TASTE; this harness reduces that to a shortlist: the seams where the
two instruments DISAGREE are the only ones worth ears.

Each rendered seam logs one row to logs/audible_calib.jsonl (append -
runs accumulate), then a summary prints the confusion between the
meter's flag (aud_max/aud_n at review.py's bars) and the ground truth
(settled env-xcorr lag med), plus the disagreement shortlist.

Usage:
    python tools/tests/_dj_audible_calib.py --music D:/Devel/music
        [--n 40]         seams to attempt (default 40)
    Ground-truth buckets: aligned < 40ms, grey 40-80ms, flam > 80ms
    (percussive flam turns audible ~25-35ms; 80ms is unambiguous).
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj.brain import Brain, load_library
from lib.dj.db import LibraryDB
from lib.dj.review import AUDIBLE_WIDE_BEATS, AUDIBLE_WIDE_N
from lib.dj.themes import get_theme

import tools.tests._dj_quality_test as Q

LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "logs", "audible_calib.jsonl")

# Every 3rd seam forces a kit-overlay style so the tier the meter was
# built for is represented; the rest are the brain's own picks.
_FORCE = ("stem_drum_swap", "drum_bridge", "stem_bass_swap")


def truth_bucket(lag_med):
    if lag_med is None:
        return None
    return ("aligned" if lag_med < 40.0
            else "grey" if lag_med <= 80.0 else "flam")


def main():
    n_want = int(sys.argv[sys.argv.index("--n") + 1]) \
        if "--n" in sys.argv else 40
    music = Q.MUSIC
    db = LibraryDB(music)
    library = load_library(db)
    eligible = [t for t in library if t.duration_s > 120]
    rng = random.Random(int(time.time()))
    theme = get_theme("groove")

    rows, attempts = [], 0
    while len(rows) < n_want and attempts < n_want * 6:
        attempts += 1
        cur = rng.choice(eligible)
        # Every 3rd seam forces a kit style, ROTATING through the tier
        # (v1 indexed with len%3==0 -> always _FORCE[0]: an all-
        # stem_drum_swap batch).
        force = _FORCE[(len(rows) // 3) % len(_FORCE)] \
            if len(rows) % 3 == 0 else None
        try:
            if force:
                m = Q.render_seam(library, cur, force)
            else:
                # The brain's own pick: plan first (cheap), skip fades -
                # the meter only runs under sync, a fade has none.
                brain = Brain(library, theme, seed=attempts)
                brain.note_played(cur)
                cand, meta = brain.choose_next(cur, 0.6, cur.bpm)
                if cand is None:
                    continue
                plan = brain.plan_transition(
                    cur, cand, meta, after_s=cur.duration_s * 0.45)
                if plan["style"] in ("long_fade",):
                    continue
                m = Q.render_seam(library, cur, plan["style"],
                                  pair=(cand, meta))
        except Exception as e:
            print(f"  render failed ({cur.title[:30]}): {e}")
            continue
        if m is None or m.get("dual_s", 0.0) < 3.0:
            continue
        row = {
            # v2: aud collection mirrors the live settled window
            # (blend+6s) and carries the full sample series - v1 rows
            # counted PLL convergence and are excluded from summaries.
            # v3: force_style's shared-theme leak fixed - v2 "unforced"
            # rows secretly planned from a one-style kit menu, so only
            # v3+ rows represent a natural style mix.
            "v": 3,
            "t": time.time(), "pair": m["pair"], "style": m["style"],
            "aud_series": m.get("aud_series") or [],
            "forced": bool(force),
            "lag_med_ms": m.get("lag_med"), "lag_max_ms": m.get("lag_max"),
            "grid_med_ms": (sorted(l for _, l in m["grid_lags"])
                            [len(m["grid_lags"]) // 2]
                            if m.get("grid_lags") else None),
            "aud_max": m.get("aud_max"), "aud_n": m.get("aud_n"),
            "dual_s": round(m.get("dual_s", 0.0), 1),
            "truth": truth_bucket(m.get("lag_med")),
            "flagged": (m.get("aud_max", 0.0) >= AUDIBLE_WIDE_BEATS
                        and m.get("aud_n", 0) >= AUDIBLE_WIDE_N),
        }
        rows.append(row)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [{len(rows):3d}/{n_want}] {row['style']:15} "
              f"truth={str(row['truth']):7} lag_med="
              f"{row['lag_med_ms'] if row['lag_med_ms'] is not None else '—'}"
              f"  meter aud_max={row['aud_max']} x{row['aud_n']}"
              f"{'  FLAG' if row['flagged'] else ''}")

    # ---- summary over EVERY accumulated v2 row, not just this run -------
    allrows = []
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("v", 1) >= 3:
                    allrows.append(r)
    print(f"\n=== calibration over {len(allrows)} accumulated seams "
          f"(bar: aud_max>={AUDIBLE_WIDE_BEATS}, n>={AUDIBLE_WIDE_N}) ===")
    print(f"  {'truth':8} {'n':>4} {'flagged':>8}")
    for b in ("aligned", "grey", "flam", None):
        grp = [r for r in allrows if r.get("truth") == b]
        if not grp:
            continue
        fl = sum(1 for r in grp if r.get("flagged"))
        print(f"  {str(b):8} {len(grp):4d} {fl:4d} ({100*fl/len(grp):3.0f}%)")
    dis = [r for r in allrows
           if (r.get("truth") == "aligned" and r.get("flagged"))
           or (r.get("truth") == "flam" and not r.get("flagged"))]
    if dis:
        print(f"\n  DISAGREEMENTS ({len(dis)}) - the only seams worth "
              "ears (find the pair in the Lab):")
        for r in dis[:12]:
            kind = ("meter flags an aligned seam" if r["truth"] == "aligned"
                    else "meter misses a measured flam")
            print(f"    {r['pair'][:52]:52} {r['style']:15} "
                  f"lag {r['lag_med_ms']}ms aud {r['aud_max']} "
                  f"x{r['aud_n']}  <- {kind}")
    else:
        print("\n  no disagreements: the meter tracks the offline "
              "ground truth on everything rendered so far.")

    # ---- threshold sweep from the stored series: no re-render needed ----
    # For each candidate (beats bar, sustain count) print detect% on
    # measured flams vs false-flag% on aligned seams. The operating
    # point wants near-0% aligned false flags first, then max detect.
    have = [r for r in allrows if r.get("aud_series")]
    if have:
        def flags(r, th, n_min):
            return sum(1 for _, v in r["aud_series"] if v > th) >= n_min
        print("\n  threshold sweep (detect% on flam / false% on aligned):")
        print("           " + "".join(f"  n>={n_:<4}" for n_ in (2, 3, 4, 6)))
        for th in (0.12, 0.15, 0.20, 0.25):
            cells = []
            for n_ in (2, 3, 4, 6):
                fl = [r for r in have if r["truth"] == "flam"]
                al = [r for r in have if r["truth"] == "aligned"]
                d = (100 * sum(flags(r, th, n_) for r in fl)
                     / max(len(fl), 1))
                fa = (100 * sum(flags(r, th, n_) for r in al)
                      / max(len(al), 1))
                cells.append(f"{d:3.0f}/{fa:3.0f}")
            print(f"    th {th:.2f}:" + "  ".join(cells))


if __name__ == "__main__":
    main()
