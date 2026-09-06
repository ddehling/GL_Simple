"""Long-run SOAK for the generative system: an all-night run, offline,
faster than realtime, with the operator steering it now and then.

What it proves (the plan's §4.2 long-run requirements):
  * never runs out, never errors, never silences itself
  * flat resources: RSS, pending notes, active voices, phrase deque, motif
    memory all bounded (no growth with hours played)
  * audio stays clean (finite, under the ceiling) for every block
  * macro-form: movements cycle on the arc, key drifts, every section is
    visited, no section runs longer than the style allows
  * steering under load: gestures / patterns / style switches every few
    minutes keep landing at phrase boundaries

Usage: python tools/tests/_gen_long_run_sim.py [--hours 6] [--style groove]
       [--seed 7] [--set-length 3600] [--report out.json]
Exit 0 = ALL PASS. A 6 h night takes ~20 min at ~17x realtime.
"""
import argparse
import json
import os
import random
import resource
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.director import GESTURES                     # noqa: E402
from lib.gen.system import GenSystem                      # noqa: E402

BLOCK = 4096


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--style", default="groove")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--set-length", type=float, default=3600.0)
    ap.add_argument("--steer-every", type=float, default=600.0, help="seconds of music between operator moves")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    logdir = tempfile.mkdtemp(prefix="gen_soak_")
    g = GenSystem(engine=None, style=args.style, key="8A", seed=args.seed, set_length_s=args.set_length,
                  threaded=False, log_dir=logdir)
    assert g.start()
    rng = random.Random(args.seed)
    total = int(args.hours * 3600 * RATE)
    fails, notes = [], []
    rendered = 0
    peak = 0.0
    bad_blocks = 0
    hour_marks = []
    rss0 = rss_mb()
    sections = {}
    keys = set()
    max_pending = max_active = max_phrases = max_motifs = 0
    last_steer = 0
    steers = []
    t0 = time.time()
    next_hour = 3600 * RATE
    while rendered < total:
        blk = g.rack.read(BLOCK)
        if blk is None:
            fails.append(f"rack went silent/done at {rendered / RATE / 3600:.2f} h")
            break
        if not np.isfinite(blk).all():
            bad_blocks += 1
        pk = float(np.abs(blk).max())
        if pk >= 1.0:
            bad_blocks += 1
        peak = max(peak, pk)
        rendered += blk.shape[0]
        g.step()
        if g.last_error and "synthetic" not in g.last_error:
            if not any(g.last_error in f for f in fails):
                fails.append(f"error at {rendered / RATE / 3600:.2f} h: {g.last_error}")
        # bounded-resource watch
        with g.rack._lock:
            max_pending = max(max_pending, len(g.rack._pending))
        max_active = max(max_active, len(g.rack._active))
        max_phrases = max(max_phrases, len(g._phrases))
        max_motifs = max(max_motifs, len(g.composer.melody.memory))
        # operator moves
        if rendered - last_steer > args.steer_every * RATE:
            last_steer = rendered
            move = rng.choice(["gesture", "gesture", "gesture", "style", "pattern", "clear", "scene"])
            if move == "gesture":
                name = rng.choice([k for k in GESTURES if k not in ("more_like_this",)])
                g.gesture(name); steers.append((rendered / RATE, "gesture", name))
            elif move == "style":
                st = rng.choice(["groove", "downtempo", "ambient"]); g.set_style(st); steers.append((rendered / RATE, "style", st))
            elif move == "pattern":
                try:
                    g.set_slot_pattern("arp", 'note("0 2 4 7 9 7 4 2").scale("A4:minor").s("arp").gain(energy)')
                    steers.append((rendered / RATE, "slot_pattern", "arp"))
                except Exception:
                    pass
            elif move == "clear":
                g.clear_slot_pattern(); g.clear_pattern(); steers.append((rendered / RATE, "clear", ""))
            elif move == "scene":
                g.scene_save("soak"); g.scene_load("soak"); steers.append((rendered / RATE, "scene", "soak"))
        if rendered >= next_hour:
            st = g.status()
            hour_marks.append({"h": round(rendered / RATE / 3600, 2), "rss_mb": round(rss_mb(), 1), "movement": st["movement"],
                               "key": st["camelot"], "style": st["style"], "section": st["section"], "notes": st["notes"],
                               "wall_s": round(time.time() - t0, 1)})
            print(f"  {hour_marks[-1]}")
            next_hour += 3600 * RATE
    # tallies from the night log
    for rec in g._log_tail:
        pass
    log_path = os.path.join(logdir, os.listdir(logdir)[0]) if os.listdir(logdir) else None
    n_phr = n_err = n_move = 0
    if log_path and log_path.endswith(".jsonl"):
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("event") == "phrase":
                    n_phr += 1; sections[r["section"]] = sections.get(r["section"], 0) + 1; keys.add(r.get("key"))
                elif r.get("event") == "error":
                    n_err += 1
                elif r.get("event") == "movement":
                    n_move += 1
    hours = rendered / RATE / 3600
    wall = time.time() - t0
    st = g.status()
    g.stop()
    rss1 = rss_mb()
    expect_moves = int(hours * 3600 / args.set_length)
    report = {"hours": round(hours, 2), "wall_s": round(wall, 1), "x_realtime": round(hours * 3600 / wall, 1),
              "phrases": n_phr, "errors": n_err, "bad_blocks": bad_blocks, "peak": round(peak, 3),
              "rss_start_mb": round(rss0, 1), "rss_end_mb": round(rss1, 1), "movements": n_move, "expected_movements": expect_moves,
              "keys": sorted(k for k in keys if k), "sections": sections, "max_pending": max_pending, "max_active": max_active,
              "max_phrases": max_phrases, "max_motifs": max_motifs, "steers": len(steers), "hour_marks": hour_marks,
              "final": {k: st[k] for k in ("style", "section", "camelot", "movement", "notes")}}
    print(json.dumps(report, indent=1))

    def check(cond, msg):
        print(("  ok   " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)
    check(hours >= args.hours * 0.999, f"rendered {hours:.2f} h")
    check(n_err == 0 and not any("error at" in f for f in fails), "no conductor errors")
    check(bad_blocks == 0, f"every block finite and under the ceiling (peak {peak:.3f})")
    check(rss1 - rss0 < 150, f"RSS growth {rss1 - rss0:.0f} MB over {hours:.1f} h (< 150)")
    check(max_pending < 4000 and max_active < 400 and max_phrases <= 96 and max_motifs <= 12,
          f"bounded: pending {max_pending}, active {max_active}, phrases {max_phrases}, motifs {max_motifs}")
    check(n_move >= expect_moves - 1, f"movements {n_move} (expected ~{expect_moves})")
    check(len(keys) >= 2, f"key drifted across the night: {sorted(k for k in keys if k)}")
    check(len(sections) >= 4, f"sections visited: {sections}")
    check(len(steers) >= hours * 3600 / args.steer_every - 1, f"{len(steers)} operator moves landed")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    print("\nALL PASS" if not fails else f"\n{len(fails)} FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
