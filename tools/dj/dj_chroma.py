"""Chroma backfill: compute the 12-bin harmonic fingerprint (DB v12
tracks.chroma) for tracks analyzed before v12, WITHOUT a full re-analysis.

New/changed tracks get chroma inline from analyze_samples on a normal scan;
this tool exists so the existing library doesn't need a full (hours-long)
rescan just for one cheap descriptor. Per track it is decode + STFT only -
a few seconds, no GPU, no demucs.

Usage:
    python tools/dj/dj_chroma.py [--dir MUSIC_DIR] [--force] [--limit N]
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj import resolve_music_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--force", action="store_true",
                    help="recompute even when chroma already stored")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N tracks (0 = all)")
    args = ap.parse_args()

    import numpy as np
    from lib.dj.db import LibraryDB
    from lib.dj.features import decode_file, frame_track, chroma_profile

    root = resolve_music_dir(args.dir)
    db = LibraryDB(root)
    rows = [r for r in db.all_tracks()
            if not r.get("missing") and not r.get("error")
            and (args.force or not r.get("chroma"))]
    if args.limit:
        rows = rows[:args.limit]
    total = len(rows)
    print(f"{total} tracks need chroma")
    done = failed = 0
    for r in rows:
        path = db.abs(r["path"])
        try:
            samples = decode_file(path)
            bands, chroma = frame_track(samples)
            frame_energy = (bands / np.maximum(bands.mean(axis=0), 1e-10)
                            ).mean(axis=1)
            prof = chroma_profile(chroma, frame_energy)
            db.set_chroma(r["id"], prof)
            done += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {r['path']}: {type(e).__name__}: {e}")
        finally:
            samples = bands = chroma = None
            if (done + failed) % 25 == 0:
                gc.collect()
        print(f"PROGRESS {done + failed} {total} {done} {failed} "
              f"{(r.get('title') or r['path'])[:50]}", flush=True)
    print(f"done: {done} written, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
