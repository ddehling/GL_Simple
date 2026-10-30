"""Key backfill (key v2, 2026-10-28): re-read every track's key + chroma with
lib/dj/features.estimate_key_audio WITHOUT a full re-analysis.

The old reader took frame_track's 10.8 Hz-bin raw-power chroma, so kick/bass
energy piled into F: 34 % of the library was stored as 7B and the stored
tracks.chroma made every pair look alike. New scans get the v2 key inline from
analyze_samples; this tool exists so the existing library doesn't need an
hours-long rescan (bumping ANALYSIS_VERSION would also re-run grid
refinement). Per track it is decode + one STFT, ~1 s, no GPU, no demucs, no
beat/structure analysis.

Writes ONLY key_pc, key_mode, camelot, key_conf, key_name, chroma and
key_version (LibraryDB.set_key), committed per track, so it is interruptible
and resumes where it stopped (rows already at features.KEY_VERSION are
skipped unless --force). The DB is per machine: run it on every machine that
has a library, the deploy box included. Afterwards run /dj-verify (key labels
and the chroma refinement change seam scoring) and restart any long-running
planner so it reloads the library.

Usage:
    python tools/dj/dj_keys.py [--dir MUSIC_DIR] [--limit N] [--ids 12,40]
                               [--dry-run] [--force]
"""
import os
import sys

# Heavy numeric work on a machine where music may be playing live: two
# threads, below-normal priority - set BEFORE numpy is imported.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
           "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("DJ_NO_PLAY_LOG", "1")
os.environ.setdefault("DJ_NO_MIDI", "1")

import argparse
import gc
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj import resolve_music_dir


def _below_normal():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception as e:                       # never fatal, but say so
        print(f"warning: could not lower priority ({type(e).__name__}: {e})")


def _dist_line(counter, total):
    if not total:
        return "(none)"
    top = ", ".join(f"{k} {v} ({100.0 * v / total:.1f}%)"
                    for k, v in counter.most_common(6))
    minor = sum(v for k, v in counter.items() if str(k).endswith("A"))
    return (f"{top} | 7B {100.0 * counter.get('7B', 0) / total:.1f}% | "
            f"minor {100.0 * minor / total:.1f}% | {len(counter)} keys")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N tracks (0 = all)")
    ap.add_argument("--ids", default="",
                    help="comma-separated track ids (default: every track)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print old -> new key per track, write nothing "
                         "(the DB is opened read-only)")
    ap.add_argument("--force", action="store_true",
                    help="re-read even tracks already at KEY_VERSION")
    args = ap.parse_args()
    _below_normal()

    from lib.dj.db import LibraryDB
    from lib.dj.features import (KEY_VERSION, _NOTE_NAMES, decode_file,
                                 estimate_key_audio, key_chroma_profile)

    root = resolve_music_dir(args.dir)
    db = LibraryDB(root, readonly=args.dry_run)
    want = {int(t) for t in args.ids.split(",") if t.strip()}
    rows = [r for r in db.all_tracks()
            if not r.get("missing") and not r.get("error")
            and (not want or r["id"] in want)
            and (args.force or (r.get("key_version") or 0) < KEY_VERSION)]
    rows.sort(key=lambda r: r["id"])
    if args.limit:
        rows = rows[:args.limit]
    total = len(rows)
    print(f"{total} tracks need key v{KEY_VERSION}"
          f"{' (DRY RUN: nothing is written)' if args.dry_run else ''}")
    before, after = Counter(), Counter()
    done = failed = same = 0
    t0 = time.time()
    try:
        for r in rows:
            path = db.abs(r["path"])
            samples = None
            try:
                samples = decode_file(path)
                pc, mode, cam, conf, ch = estimate_key_audio(samples)
                name = f"{_NOTE_NAMES[pc]} {mode}"
                prof = key_chroma_profile(ch) if ch is not None else None
                old = r.get("camelot") or "?"
                before[old] += 1
                after[cam] += 1
                same += old == cam
                if args.dry_run:
                    print(f"  {r['id']:5d} {old:>4s} -> {cam:<4s} conf "
                          f"{float(r.get('key_conf') or 0):.2f} -> {conf:.2f}"
                          f"  {(r.get('title') or r['path'])[:50]}")
                else:
                    db.set_key(r["id"], pc, mode, cam, round(conf, 3), name,
                               prof, KEY_VERSION)
                done += 1
            except Exception as e:
                failed += 1
                print(f"FAIL {r['path']}: {type(e).__name__}: {e}")
            finally:
                samples = None
                if (done + failed) % 25 == 0:
                    gc.collect()
            print(f"PROGRESS {done + failed} {total} {done} {failed} "
                  f"{(r.get('title') or r['path'])[:50]}", flush=True)
    except KeyboardInterrupt:
        print("interrupted: every finished track is committed; rerun to resume")
    n = sum(after.values())
    print(f"\nkeys before ({n}): {_dist_line(before, n)}")
    print(f"keys after  ({n}): {_dist_line(after, n)}")
    print(f"unchanged camelot {same}/{n}")
    print(f"done: {done} {'read' if args.dry_run else 'written'}, {failed} failed"
          f" in {time.time() - t0:.0f}s")
    db.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
