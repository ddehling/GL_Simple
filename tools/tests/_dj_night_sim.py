"""Simulate whole DJ nights offline and census what goes wrong.

Born 2026-08-17 ("simulate 1000 mixes as if they were from a real
night - I want to see what issue we have in normal operation"). Unlike
_dj_audible_calib.py (independent random pairs, kit tier oversampled),
this CHAINS seams the way a night does: one Brain per night keeps its
recency memory, each pick's B becomes the next A, the arc cycles like
the live 90-minute set, personas rotate, and long_fades are rendered
too - the sample IS a night, minus the waiting.

Every seam renders through the real submix (stems attached for stem
styles) and is measured on every axis the QA gate knows: kick lag,
grid lag, the live audible meter, level lurch, dead air, double bass,
clipping. Rows append to logs/night_sim/<shard>.jsonl.

Usage:
  worker (one process, run several in parallel):
    python tools/tests/_dj_night_sim.py --worker 0 --nights 3 \
        --seams 84 --music D:/Devel/music
  report (aggregate all shards):
    python tools/tests/_dj_night_sim.py --report
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
from lib.dj.persona import PERSONAS, ROTATION
from lib.dj.review import AUDIBLE_WIDE_BEATS, AUDIBLE_WIDE_N
from lib.dj.themes import get_theme

import tools.tests._dj_quality_test as Q

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
SHARD_DIR = os.path.join(ROOT, "logs", "night_sim")

SET_CYCLE_S = 90 * 60.0
AVG_PLAY_S = 240.0          # arc advance per seam (a ~4-minute play)


def _arg(name, dflt):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else dflt


# ---------------------------------------------------------------- worker

def worker():
    w = int(_arg("--worker", "0"))
    nights = int(_arg("--nights", "3"))
    seams = int(_arg("--seams", "84"))
    os.makedirs(SHARD_DIR, exist_ok=True)
    shard = os.path.join(SHARD_DIR, f"w{w}.jsonl")
    db = LibraryDB(Q.MUSIC)
    library = load_library(db)
    eligible = [t for t in library if t.duration_s > 120]
    rng = random.Random(1000 + w)

    def emit(row):
        with open(shard, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    for night in range(nights):
        persona = ROTATION[(w + night) % len(ROTATION)]
        brain = Brain(library, get_theme("groove"), seed=w * 100 + night)
        brain.persona = PERSONAS[persona]
        cur = rng.choice(eligible)
        brain.note_played(cur)
        decoded = {}
        elapsed = 0.0
        for i in range(seams):
            arc = (elapsed % SET_CYCLE_S) / SET_CYCLE_S
            base = {"night": f"w{w}n{night}", "seam": i,
                    "persona": persona, "arc": round(arc, 3),
                    "a": cur.title[:40]}
            try:
                cand, meta = brain.choose_next(cur, arc, cur.bpm)
            except Exception as e:
                emit({**base, "skip": f"choose_next: {e}"})
                break
            if cand is None:
                emit({**base, "skip": "no_candidate"})
                break
            try:
                plan = brain.plan_transition(
                    cur, cand, meta, arc=arc,
                    after_s=cur.duration_s * 0.45)
            except Exception as e:
                emit({**base, "skip": f"plan: {e}"})
                brain.note_played(cand)
                cur = cand
                continue
            diag = plan.get("diag") or {}
            base.update({
                "b": cand.title[:40], "style": plan["style"],
                "fade_reason": diag.get("fade_reason"),
                "gated": sorted((diag.get("gated") or {}).items()),
                "conf": [round(cur.bpm_conf or 0, 2),
                         round(cand.bpm_conf or 0, 2)],
                "kick_agreement": (plan.get("rhythm")
                                   or {}).get("kick_agreement"),
            })
            try:
                m = Q.render_seam(library, cur, plan["style"],
                                  pair=(cand, meta), decoded=decoded)
            except Exception as e:
                emit({**base, "skip": f"render: {e}"})
                m = None
            if m is None:
                if "skip" not in base:
                    emit({**base, "skip": "render_refused"})
            else:
                emit({**base,
                      "rate": round(m.get("rate", 1.0), 4),
                      "dual_s": round(m.get("dual_s", 0.0), 1),
                      "lag_med_ms": m.get("lag_med"),
                      "lag_max_ms": m.get("lag_max"),
                      "grid_med_ms": (sorted(
                          l for _, l in m["grid_lags"])
                          [len(m["grid_lags"]) // 2]
                          if m.get("grid_lags") else None),
                      "aud_max": m.get("aud_max"),
                      "aud_n": m.get("aud_n"),
                      "lurch_db": round(m.get("lurch_db", 0.0), 2),
                      "lurch_solo_db": round(
                          m.get("lurch_solo_db", 0.0), 2),
                      "rms_min_ratio": round(
                          m.get("rms_min_ratio", 1.0), 3),
                      "bass_bump_db": round(
                          m.get("bass_bump_db", 0.0), 2),
                      "peak": round(m.get("peak", 0.0), 3),
                      "clipped": m.get("clipped", 0),
                      })
            # Chain: B is the next A. Keep only its decode in the cache.
            brain.note_played(cand)
            decoded = {cand.id: decoded[cand.id]} \
                if cand.id in decoded else {}
            cur = cand
            elapsed += AVG_PLAY_S
    open(shard + ".done", "w").close()
    print(f"worker {w} done")


# ---------------------------------------------------------------- report

def report():
    rows = []
    import glob
    for p in sorted(glob.glob(os.path.join(SHARD_DIR, "w*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    rendered = [r for r in rows if "skip" not in r and "style" in r]
    skipped = [r for r in rows if "skip" in r]
    n = len(rendered)
    print(f"=== night sim: {n} rendered seams, {len(skipped)} skipped "
          f"({len(set(r['night'] for r in rows))} simulated nights) ===")
    if skipped:
        from collections import Counter
        why = Counter(str(r["skip"])[:40] for r in skipped)
        print("  skips: " + "  ".join(f"{k} x{v}"
                                      for k, v in why.most_common(6)))

    from collections import Counter, defaultdict
    styles = Counter(r["style"] for r in rendered)
    print("\n  style census: " + "  ".join(
        f"{k}={v}" for k, v in styles.most_common()))
    fades = [r for r in rendered if r["style"] == "long_fade"]
    if fades:
        why = Counter(r.get("fade_reason") or "dice" for r in fades)
        print(f"  fade share {100*len(fades)/max(n,1):.0f}%: " + "  ".join(
            f"{k} x{v}" for k, v in why.most_common()))

    # Issue bars. Synced-only where the axis needs sync.
    def synced(r):
        return r["style"] != "long_fade"

    issues = {
        "kick flam >80ms (env-xcorr, synced)": lambda r: synced(r)
            and (r.get("lag_med_ms") or 0) > 80.0,
        "audible-meter verdict (live bar)": lambda r:
            (r.get("aud_max") or 0) >= AUDIBLE_WIDE_BEATS
            and (r.get("aud_n") or 0) >= AUDIBLE_WIDE_N,
        # The QA gate's rule: the transition must not lurch harder than
        # the pair's own solo dynamics - an absolute bar overcounts on
        # dynamic material (and fades DIP by design).
        "level lurch (louder than the music's own)": lambda r:
            (r.get("lurch_db") or 0)
            > max(6.0, (r.get("lurch_solo_db") or 0) + 1.5),
        "dead air (rms floor <15% of median)": lambda r:
            (r.get("rms_min_ratio") or 1) < 0.15,
        "double bass >+3.5dB over solo peaks": lambda r:
            (r.get("bass_bump_db") or 0) > 3.5,
        "clipping": lambda r: (r.get("clipped") or 0) > 0,
    }
    print(f"\n  {'issue':44} {'seams':>6}  share")
    flagged_any = set()
    per_style = defaultdict(Counter)
    for name, f in issues.items():
        hit = [r for r in rendered if f(r)]
        for r in hit:
            flagged_any.add((r["night"], r["seam"]))
            per_style[r["style"]][name] += 1
        print(f"  {name:44} {len(hit):6d}  {100*len(hit)/max(n,1):4.1f}%")
    print(f"  {'ANY issue':44} {len(flagged_any):6d}  "
          f"{100*len(flagged_any)/max(n,1):4.1f}%")

    print(f"\n  per-style issue rates (issues/seams):")
    for st, cnt in styles.most_common():
        tot = sum(per_style[st].values())
        worst = per_style[st].most_common(1)
        print(f"    {st:16} {tot:3d}/{cnt:3d}"
              + (f"   worst: {worst[0][0][:34]} x{worst[0][1]}"
                 if worst else ""))

    # The individual worst seams - what an operator would actually hear.
    def badness(r):
        return ((r.get("lag_med_ms") or 0) / 80.0 if synced(r) else 0) \
            + (r.get("aud_max") or 0) / 0.12 \
            + (r.get("lurch_db") or 0) / 6.0 \
            + (r.get("bass_bump_db") or 0) / 3.5
    worst = sorted(rendered, key=badness, reverse=True)[:10]
    print("\n  worst 10 seams:")
    for r in worst:
        print(f"    {r['a'][:24]:24} -> {r.get('b','?')[:24]:24} "
              f"{r['style']:14} lag={r.get('lag_med_ms')} "
              f"aud={r.get('aud_max')}x{r.get('aud_n')} "
              f"lurch={r.get('lurch_db')} bass={r.get('bass_bump_db')} "
              f"conf={r.get('conf')}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        worker()
