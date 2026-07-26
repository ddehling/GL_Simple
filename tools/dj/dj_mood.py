"""ML mood/valence pass - score the library with Music2Emo descriptors.

Runs the Music2Emotion model (PyTorch, same torch+CUDA stack as the demucs
vocal pass) over each track and stores normalized valence/arousal (0..1) +
predicted mood tags on the track (tracks.mood_ml). character.py then PREFERS
these over its heuristic derivation, so danceable/dark/uplifting tags,
valence steering and energy arcs get real ML signal.

Needs a Music2Emotion clone (see lib/dj/mood_ml.find_model_dir): set
$DJ_MOOD_MODEL_DIR or place it as a sibling `../Music2Emotion`. Incremental -
already-scored tracks are skipped unless --force; each track commits, so it's
resumable. ~3-5 s/track on GPU.

This is the subprocess the planner's "Mood (ML)" button spawns; it prints
`PROGRESS <done> <total> <matched> <missed> <title>` lines the GUI parses.

Usage:
    python tools/dj/dj_mood.py                 # score all un-scored tracks
    python tools/dj/dj_mood.py --limit 20
    python tools/dj/dj_mood.py --force         # re-score everything
    python tools/dj/dj_mood.py --stats         # coverage report
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj import resolve_music_dir
from lib.dj.db import LibraryDB
from lib.dj import mood_ml


def cmd_score(args):
    root = os.path.abspath(resolve_music_dir(args.dir))
    db = LibraryDB(root)
    rows = db.all_tracks()
    # resolve absolute paths BEFORE MoodExtractor chdir's into the model repo
    todo = []
    for r in rows:
        if r.get("missing") or r.get("error"):
            continue
        if not args.force and r.get("mood_ml"):
            continue
        todo.append((r["id"], db.abs(r["path"]),
                     (r.get("title") or r["path"])[:40]))
    if args.limit:
        todo = todo[:args.limit]
    total = len(todo)
    print(f"{len(rows)} tracks, {total} to score "
          f"(~{total * 4.0 / 60:.0f} min on GPU)", flush=True)
    if not total:
        db.close()
        return 0

    if not mood_ml.available():
        print("mood pass unavailable: Music2Emotion clone not found "
              "(set $DJ_MOOD_MODEL_DIR or place ../Music2Emotion) or torch "
              "missing", flush=True)
        db.close()
        return 2

    ext = mood_ml.MoodExtractor(args.model_dir or None)  # chdir + load model
    matched = missed = 0
    for i, (tid, path, title) in enumerate(todo, 1):
        blob = None
        if os.path.isfile(path):
            blob = ext.analyze(path)
        if blob and blob.get("valence") is not None:
            matched += 1
            db.set_mood_ml(tid, blob)
            tag = ", ".join(blob["moods"][:4]) or "-"
            print(f"  [{i}/{total}] {title:40s} v{blob['valence9']} "
                  f"a{blob['arousal9']}  {tag}", flush=True)
        else:
            missed += 1
            db.set_mood_ml(tid, {"matched": False})
            print(f"  [{i}/{total}] {title:40s} -> no result", flush=True)
        # machine-parseable line for the GUI worker
        print(f"PROGRESS {i} {total} {matched} {missed} {title}", flush=True)
        # Keep memory bounded across a long library: release per-track buffers
        # and the torch CUDA cache periodically so the pass (and the rest of
        # the machine) doesn't creep toward OOM.
        if i % 25 == 0:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    print(f"\nscored {matched}, failed {missed}", flush=True)
    db.close()
    return 0


def cmd_stats(args):
    root = os.path.abspath(resolve_music_dir(args.dir))
    db = LibraryDB(root)
    rows = db.all_tracks()
    scored = [r for r in rows if r.get("mood_ml")]
    good = [r for r in scored if (r["mood_ml"] or {}).get("valence") is not None]
    moods = {}
    for r in good:
        for m in r["mood_ml"].get("moods", []):
            moods[m] = moods.get(m, 0) + 1
    print(f"library: {len(rows)} tracks")
    print(f"mood-scored: {len(scored)}  with-values: {len(good)}  "
          f"failed: {len(scored) - len(good)}")
    if good:
        vs = [r["mood_ml"]["valence"] for r in good]
        ars = [r["mood_ml"]["arousal"] for r in good]
        print(f"valence  range {min(vs):.2f}-{max(vs):.2f}  "
              f"mean {sum(vs)/len(vs):.2f}")
        print(f"arousal  range {min(ars):.2f}-{max(ars):.2f}  "
              f"mean {sum(ars)/len(ars):.2f}")
    if moods:
        print("\ntop moods:")
        for m, n in sorted(moods.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {n:4d}  {m}")
    db.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="", help="music library directory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="re-score already-scored tracks")
    ap.add_argument("--model-dir", default="",
                    help="path to the Music2Emotion clone")
    ap.add_argument("--stats", action="store_true",
                    help="coverage report only")
    args = ap.parse_args()
    if args.limit and not args.stats:
        pass  # limit applied in cmd_score via slice below
    return cmd_stats(args) if args.stats else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
