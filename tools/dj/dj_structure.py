"""ML structure pass - label every track's sections with allin1.

Runs the All-In-One Music Structure Analyzer (PyTorch, same torch+CUDA
stack as the demucs vocal pass) over each track and stores functional
segment labels (intro/verse/chorus/bridge/inst/solo/break/outro) on the
track (tracks.structure, DB v12). The brain folds these into mix-in/out
fit so seams land on real outros/intros and never enter mid-chorus by
accident; the internal SSM sections stay the fallback when the pass
hasn't run.

Deps are optional: pip install -r requirements-dj-structure.txt (see its
header for the NATTEN-on-Windows note). Incremental - already-labeled
tracks are skipped unless --force; each track commits, so it's resumable.
A few seconds per track on GPU; much slower on CPU.

This is the subprocess the planner's "Structure (ML)" button spawns; it
prints `PROGRESS <done> <total> <matched> <missed> <title>` lines the GUI
parses.

Usage:
    python tools/dj/dj_structure.py                 # label all unlabeled
    python tools/dj/dj_structure.py --limit 20
    python tools/dj/dj_structure.py --force         # re-label everything
    python tools/dj/dj_structure.py --stats         # coverage report
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj import resolve_music_dir
from lib.dj.db import LibraryDB
from lib.dj import structure_ml


def cmd_label(args):
    root = os.path.abspath(resolve_music_dir(args.dir))
    db = LibraryDB(root)
    rows = db.all_tracks()
    todo = []
    for r in rows:
        if r.get("missing") or r.get("error"):
            continue
        if not args.force and r.get("structure"):
            continue
        todo.append((r["id"], db.abs(r["path"]),
                     (r.get("title") or r["path"])[:40]))
    if args.limit:
        todo = todo[:args.limit]
    total = len(todo)
    print(f"{len(rows)} tracks, {total} to label", flush=True)
    if not total:
        db.close()
        return 0

    if not structure_ml.available():
        print("structure pass unavailable: pip install -r "
              "requirements-dj-structure.txt (torch + allin1)", flush=True)
        db.close()
        return 2

    analyzer = structure_ml.StructureAnalyzer()
    matched = missed = 0
    for i, (tid, path, title) in enumerate(todo, 1):
        blob = None
        try:
            if os.path.isfile(path):
                blob = analyzer.analyze(path)
        except Exception as e:
            print(f"  [{i}/{total}] {title:40s} -> "
                  f"{type(e).__name__}: {e}", flush=True)
        if blob and blob.get("segments"):
            matched += 1
            db.set_structure(tid, blob)
            labels = " ".join(s[2] for s in blob["segments"][:10])
            print(f"  [{i}/{total}] {title:40s} {labels}", flush=True)
        else:
            missed += 1
            db.set_structure(tid, {"matched": False, "segments": []})
            print(f"  [{i}/{total}] {title:40s} -> no result", flush=True)
        print(f"PROGRESS {i} {total} {matched} {missed} {title}", flush=True)
        if i % 25 == 0:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    print(f"\nlabeled {matched}, failed {missed}", flush=True)
    db.close()
    return 0


def cmd_stats(args):
    root = os.path.abspath(resolve_music_dir(args.dir))
    db = LibraryDB(root)
    rows = db.all_tracks()
    labeled = [r for r in rows if r.get("structure")]
    good = [r for r in labeled if (r["structure"] or {}).get("segments")]
    counts = {}
    for r in good:
        for _, _, label in r["structure"]["segments"]:
            counts[label] = counts.get(label, 0) + 1
    print(f"library: {len(rows)} tracks")
    print(f"structure-labeled: {len(labeled)}  with-segments: {len(good)}  "
          f"failed: {len(labeled) - len(good)}")
    if counts:
        print("\nsegment labels:")
        for m, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:5d}  {m}")
    db.close()
    return 0


def cmd_batch(args):
    """SQLITE-FREE batch mode for cross-boundary runs (WSL): the planner
    holds the library DB open in WAL mode on the Windows side, and WAL's
    shared memory cannot span /mnt/c - opening it here dies with 'disk
    I/O error'. So the planner EXPORTS a track list (id, title, path),
    this mode appends {id, structure} JSONL results, and the planner
    imports them into the DB it already owns. Resumable: already-present
    result ids are skipped."""
    with open(args.tracklist, encoding="utf-8") as f:
        tracks = json.load(f)
    done_ids = set()
    if os.path.isfile(args.results):
        with open(args.results, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
    todo = [t for t in tracks if t["id"] not in done_ids]
    total = len(todo)
    print(f"{len(tracks)} in batch, {total} to label", flush=True)
    if not total:
        return 0
    if not structure_ml.available():
        print("structure pass unavailable: torch + allin1 missing in this "
              "environment", flush=True)
        return 2
    analyzer = structure_ml.StructureAnalyzer()
    matched = missed = 0
    with open(args.results, "a", encoding="utf-8") as out:
        for i, t in enumerate(todo, 1):
            blob = None
            try:
                if os.path.isfile(t["path"]):
                    blob = analyzer.analyze(t["path"])
            except Exception as e:
                print(f"  [{i}/{total}] {t.get('title', '')[:40]} -> "
                      f"{type(e).__name__}: {e}", flush=True)
            if blob and blob.get("segments"):
                matched += 1
            else:
                missed += 1
                blob = {"matched": False, "segments": []}
            out.write(json.dumps({"id": t["id"], "structure": blob}) + "\n")
            out.flush()
            print(f"PROGRESS {i} {total} {matched} {missed} "
                  f"{t.get('title', '')[:40]}", flush=True)
            if i % 25 == 0:
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
    print(f"\nlabeled {matched}, failed {missed}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="", help="music library directory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="re-label already-labeled tracks")
    ap.add_argument("--stats", action="store_true",
                    help="coverage report only")
    ap.add_argument("--tracklist", default="",
                    help="batch mode: JSON track list (no DB access)")
    ap.add_argument("--results", default="",
                    help="batch mode: JSONL results path")
    args = ap.parse_args()
    if args.tracklist:
        return cmd_batch(args)
    return cmd_stats(args) if args.stats else cmd_label(args)


if __name__ == "__main__":
    sys.exit(main())
