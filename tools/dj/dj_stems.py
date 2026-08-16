"""Stem render pass - pre-separate every track for stem-aware mixing.

Runs htdemucs (same torch+CUDA stack as the vocal pass) over each track
ONCE and keeps the separated audio: four encoded stereo stems
(drums/bass/other/vocals) under <music_root>/.stems/<track_id>/, ~20 MB
per track. With stems on disk, the mixing brain unlocks the
stem_drum_swap and acapella_out transition styles (drums-only entries,
vocal tails riding the incoming instrumental); tracks without stems keep
the classic styles - everything degrades gracefully.

Deps: pip install -r requirements-dj-vocals.txt (torch + demucs).
Incremental and resumable: tracks whose four stem files exist are
skipped unless --force. ~10-30 s/track on GPU; slow on CPU.

This is the subprocess the planner's "Stems (render)" button spawns; it
prints `PROGRESS <done> <total> <matched> <missed> <title>` lines.

Usage:
    python tools/dj/dj_stems.py                 # render all missing
    python tools/dj/dj_stems.py --limit 10
    python tools/dj/dj_stems.py --force         # re-render everything
    python tools/dj/dj_stems.py --stats         # coverage + disk usage
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj import resolve_music_dir
from lib.dj.db import LibraryDB
from lib.dj import stems as ST


def cmd_render(args):
    root = os.path.abspath(resolve_music_dir(args.dir))
    db = LibraryDB(root)
    rows = db.all_tracks()
    todo = []
    from lib.dj.scan import MAX_SCAN_MB, MAX_SCAN_MIN
    for r in rows:
        if r.get("missing") or r.get("error"):
            continue
        if args.track and r["id"] != args.track:
            continue                     # single-track mode (Analysis tab)
        if not args.track and (
                r.get("excluded")        # do-not-use: never auto-rendered
                or (r.get("duration_s") or 0) > MAX_SCAN_MIN * 60.0
                or (r.get("file_size") or 0) > MAX_SCAN_MB * 1e6):
            continue                     # (crash guard - see lib/dj/scan.py)
        if not args.force and not args.track \
                and ST.has_stems(root, r["id"]):
            continue
        todo.append((r["id"], db.abs(r["path"]),
                     (r.get("title") or r["path"])[:40]))
    if args.limit:
        todo = todo[:args.limit]
    total = len(todo)
    print(f"{len(rows)} tracks, {total} to render "
          f"(~{total * 20 / 1024.0:.1f} GB of stems)", flush=True)
    if not total:
        db.close()
        return 0

    from lib.dj import vocals
    if not vocals.available():
        print("stem pass unavailable: pip install -r "
              "requirements-dj-vocals.txt (torch + demucs)", flush=True)
        db.close()
        return 2

    from lib.dj.features import decode_file_stereo
    renderer = ST.StemRenderer(model=args.model)
    matched = missed = 0
    for i, (tid, path, title) in enumerate(todo, 1):
        try:
            samples = decode_file_stereo(path)
            renderer.render(samples, root, tid)
            matched += 1
            print(f"  [{i}/{total}] {title:40s} ok", flush=True)
        except Exception as e:
            missed += 1
            print(f"  [{i}/{total}] {title:40s} -> "
                  f"{type(e).__name__}: {e}", flush=True)
        finally:
            samples = None
        print(f"PROGRESS {i} {total} {matched} {missed} {title}", flush=True)
        if i % 10 == 0:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    print(f"\nrendered {matched}, failed {missed}", flush=True)
    db.close()
    return 0


def cmd_stats(args):
    root = os.path.abspath(resolve_music_dir(args.dir))
    db = LibraryDB(root)
    rows = [r for r in db.all_tracks()
            if not r.get("missing") and not r.get("error")]
    have = [r for r in rows if ST.has_stems(root, r["id"])]
    size = 0
    base = os.path.join(root, ST.STEMS_DIRNAME)
    if os.path.isdir(base):
        for dirpath, _, files in os.walk(base):
            for f in files:
                try:
                    size += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
    print(f"library: {len(rows)} tracks, stems rendered: {len(have)}")
    print(f"stem storage: {size / 1024 ** 3:.2f} GB in {base}")
    db.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="", help="music library directory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--track", type=int, default=0,
                    help="render ONE track by library id (re-renders even "
                    "if stems exist - the Analysis tab's per-song button)")
    ap.add_argument("--model", default=ST.DEFAULT_STEM_MODEL,
                    choices=list(ST.STEM_MODELS),
                    help="htdemucs_ft (fine-tuned bag: cleaner stems, "
                    "~4x render time; DEFAULT) or htdemucs (fast)")
    ap.add_argument("--force", action="store_true",
                    help="re-render even when stems exist")
    ap.add_argument("--stats", action="store_true",
                    help="coverage + disk usage report")
    args = ap.parse_args()
    return cmd_stats(args) if args.stats else cmd_render(args)


if __name__ == "__main__":
    sys.exit(main())
