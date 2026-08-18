"""Anchor the offline flam instruments to the operator's actual ears.

The night-sim census (2026-08-17) reads ~46% of synced seams as >80ms
kick flam - a rate that cannot be literally true (live nights rated by
the same operator are mostly fine). Either normal operation really is
that rough, or the env-xcorr instrument overreads at scale. This tool
settles it with taste data that already exists: the Seam Lab ratings
(logs/seam_lab_ratings.jsonl) carry (a_id, b_id, style, verdict) for
seams the operator listened to and judged. Re-render those exact pairs
through the harness and put the instruments' numbers next to the
verdicts.

If operator-GOOD seams measure high lag, the instrument lies at scale
and the census bars need recalibrating against this anchoring. If
verdicts order the lag cleanly, the census is real.

Usage:
    python tools/tests/_dj_ear_anchor.py --music D:/Devel/music \
        [--limit 60]
"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj.brain import Brain, load_library
from lib.dj.db import LibraryDB
from lib.dj.themes import get_theme

import tools.tests._dj_quality_test as Q

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
RATINGS = os.path.join(ROOT, "logs", "seam_lab_ratings.jsonl")
OUT = os.path.join(ROOT, "logs", "ear_anchor.jsonl")


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) \
        if "--limit" in sys.argv else 60
    db = LibraryDB(Q.MUSIC)
    library = load_library(db)
    by_id = {t.id: t for t in library}

    rated = []
    seen = set()
    for line in open(RATINGS, encoding="utf-8"):
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("verdict") in ("good", "passable", "bad") \
                and not e.get("gate_test"):
            key = (e.get("a_id"), e.get("b_id"), e.get("style"))
            if key in seen:
                continue
            seen.add(key)
            rated.append(e)
    rated = rated[-limit:]
    print(f"re-rendering {len(rated)} operator-rated seams...")

    rows = []
    for i, e in enumerate(rated):
        a = by_id.get(e.get("a_id"))
        b = by_id.get(e.get("b_id"))
        if a is None or b is None:
            continue
        try:
            brain = Brain(library, get_theme("groove"), seed=7)
            s, meta = brain.score(a, b, 0.6, a.bpm, bpm_target=None)
            if meta is None:
                continue
            # allow_benched: a BENCH is a decision about what to play
            # LIVE, not a reason to throw away the operator's past
            # verdicts - and this tool exists to order those verdicts.
            # Without it the anchor silently under-samples exactly the
            # styles that earned the bench, i.e. the BAD end of the
            # scale it is meant to span: after drum_bridge and
            # stem_bass_swap were benched 2026-08-17, an 80-seam run
            # yielded 34 rows of which only 2 were 'bad', which cannot
            # validate anything. Skipped seams were silent, so the
            # sample looked fine until the verdict counts were read.
            m = Q.render_seam(library, a, e["style"], pair=(b, meta),
                              allow_benched=True)
        except Exception as ex:
            print(f"  [{i}] render failed: {ex}")
            continue
        if m is None:
            continue
        ki = m.get("kick_iso") or {}
        row = {"verdict": e["verdict"], "style": e["style"],
               "pair": m["pair"], "lag_med_ms": m.get("lag_med"),
               "grid_med_ms": (sorted(l for _, l in m["grid_lags"])
                               [len(m["grid_lags"]) // 2]
                               if m.get("grid_lags") else None),
               "aud_max": m.get("aud_max"), "aud_n": m.get("aud_n"),
               "kick_flam_ms": ki.get("flam_med_ms"),
               "kick_off_a_ms": ki.get("off_a_ms"),
               "kick_off_b_ms": ki.get("off_b_ms"),
               "dual_s": round(m.get("dual_s", 0.0), 1)}
        rows.append(row)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [{i + 1:2d}/{len(rated)}] {e['verdict']:8} "
              f"{e['style']:15} lag={row['lag_med_ms']} "
              f"aud={row['aud_max']}x{row['aud_n']} "
              f"kick_iso={row['kick_flam_ms']}")

    print("\n=== the instruments vs the operator's ears ===")
    for v in ("good", "passable", "bad"):
        grp = [r for r in rows if r["verdict"] == v
               and r["lag_med_ms"] is not None]
        if not grp:
            continue
        lags = [r["lag_med_ms"] for r in grp]
        auds = [r["aud_max"] or 0 for r in grp]
        over = sum(1 for l in lags if l > 80.0)
        print(f"  {v:8} n={len(grp):2d}  lag med {st.median(lags):6.1f}ms "
              f"(p25 {sorted(lags)[len(lags)//4]:.0f} / "
              f"p75 {sorted(lags)[3*len(lags)//4]:.0f})  "
              f">80ms: {over} ({100*over/len(grp):.0f}%)  "
              f"aud_max med {st.median(auds):.3f}")
    print("\n  ISOLATED-KICK instrument (per-deck, no cross-correlation):")
    for v in ("good", "passable", "bad"):
        grp = [r for r in rows if r["verdict"] == v
               and r.get("kick_flam_ms") is not None]
        if not grp:
            continue
        kf = [r["kick_flam_ms"] for r in grp]
        over = sum(1 for k in kf if k > 45.0)
        print(f"  {v:8} n={len(grp):2d}  kick flam med "
              f"{st.median(kf):6.1f}ms  >45ms: {over} "
              f"({100*over/len(grp):.0f}%)")
    unm = sum(1 for r in rows if r.get("kick_flam_ms") is None)
    print(f"  (declined to measure: {unm} of {len(rows)} - diffuse "
          "material or too few confident kicks)")
    good_hi = [r for r in rows if r["verdict"] == "good"
               and (r["lag_med_ms"] or 0) > 80.0]
    if good_hi:
        print(f"\n  operator-GOOD seams the instrument calls flam "
              f"({len(good_hi)}) - the overread, itemized:")
        for r in good_hi[:8]:
            print(f"    {r['pair'][:52]:52} {r['style']:15} "
                  f"lag {r['lag_med_ms']}ms grid {r['grid_med_ms']}")


if __name__ == "__main__":
    main()
