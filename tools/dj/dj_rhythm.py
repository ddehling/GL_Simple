"""Rhythm backfill: compute the beat-sync rhythm signature (DB v13
tracks.rhythm) for tracks analyzed before v13, WITHOUT a full re-analysis -
and UPGRADE mix-derived signatures to stem-derived ones where the htdemucs
drum stem is on disk (a clean rhythm section beats guessing kicks from the
full mix's low-band flux).

New/changed tracks get a mix-derived signature inline from analyze_samples
on a normal scan; this tool exists so the existing library doesn't need a
full rescan, and so stem owners get the better measurement. Per track it
is decode + STFT + a beat-grid fold - a few seconds, no GPU, no demucs
(it only READS stems already rendered by tools/dj/dj_stems.py).

Usage:
    python tools/dj/dj_rhythm.py [--dir MUSIC_DIR] [--force] [--limit N]
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
                    help="recompute even when a signature is already stored")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N tracks (0 = all)")
    args = ap.parse_args()

    from lib.dj.db import LibraryDB
    from lib.dj.features import (decode_file, frame_track, FPS,
                                 primary_mix_point)
    from lib.dj.rhythm import SIG_VERSION, rhythm_signature
    from lib.dj.stems import stem_paths

    root = resolve_music_dir(args.dir)
    db = LibraryDB(root)

    def wants(r):
        if r.get("missing") or r.get("error") or not r.get("beat_grid"):
            return False
        if args.force or not r.get("rhythm"):
            return True
        sig = r["rhythm"] or {}
        if (sig.get("v") or 0) < SIG_VERSION:    # format upgrade (regions,
            return True                          # meter) - recompute
        # mix-derived signature + stems on disk -> upgrade to stem-derived
        return (sig.get("source") == "mix"
                and stem_paths(root, r["id"]) is not None)

    rows = [r for r in db.all_tracks() if wants(r)]
    if args.limit:
        rows = rows[:args.limit]
    total = len(rows)
    print(f"{total} tracks need a rhythm signature")
    done = failed = 0
    for r in rows:
        samples = bands = None
        try:
            stems = stem_paths(root, r["id"])
            if stems is not None:
                samples = decode_file(stems["drums"])
                source = "stem"
            else:
                samples = decode_file(db.abs(r["path"]))
                source = "mix"
            bands, _chroma = frame_track(samples)
            pts = db.mix_points_for(r["id"])
            sig = rhythm_signature(bands, r["beat_grid"],
                                   r.get("downbeat_offset") or 0,
                                   fps=FPS, source=source,
                                   mix_in_s=primary_mix_point(pts, "in"),
                                   mix_out_s=primary_mix_point(pts, "out"))
            if sig is None:
                raise ValueError("grid too short/unstable for a fold")
            db.set_rhythm(r["id"], sig)
            done += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {r['path']}: {type(e).__name__}: {e}")
        finally:
            samples = bands = None
            if (done + failed) % 25 == 0:
                gc.collect()
        print(f"PROGRESS {done + failed} {total} {done} {failed} "
              f"{(r.get('title') or r['path'])[:50]}", flush=True)
    print(f"done: {done} written, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
