"""DJ library scanner CLI.

Incrementally analyzes every audio file in the music library (default:
<repo_parent>/music) into music/dj_library.sqlite3. Already-analyzed,
unchanged files are skipped; add songs and only the new ones get scanned.

Usage:
    python tools/dj_scan.py                     # incremental scan, default dir
    python tools/dj_scan.py --dir D:/music      # explicit library
    python tools/dj_scan.py --force             # re-analyze everything
    python tools/dj_scan.py --refine-grids      # + re-run low-conf beat grids
    python tools/dj_scan.py --workers 4
    python tools/dj_scan.py --track "song.mp3"  # single-file debug report
    python tools/dj_scan.py --list              # dump the library table
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.dj import resolve_music_dir


def _bar(done, total, name):
    width = 30
    fill = int(width * done / max(total, 1))
    sys.stdout.write(f"\r[{'#' * fill}{'.' * (width - fill)}] "
                     f"{done}/{total}  {name[:44]:44s}")
    sys.stdout.flush()


def cmd_scan(args):
    from lib.dj.scan import scan_library
    root = resolve_music_dir(args.dir)
    if not os.path.isdir(root):
        print(f"Music directory not found: {root}")
        print("Create it (or pass --dir) and drop audio files inside.")
        return 1
    print(f"Scanning {root}")
    s = scan_library(root, workers=args.workers, force=args.force,
                     progress_cb=_bar, vocals_pass=not args.no_vocals,
                     refine_grids=args.refine_grids)
    print(f"\n\nfound {s['found']} files | scanned {s['scanned']} "
          f"| skipped {s['skipped']} (unchanged) | errors {s['errors']} "
          f"| missing {s['missing']} | {s['elapsed_s']}s")
    g = s.get("grid_refine")
    if g:
        print(f"grid refine: {g['refined']} low-conf tracks re-run | "
              f"{g['improved']} improved, {g['regressed']} regressed "
              f"(mean {g['mean_delta']:+.3f}) | "
              f"{g['promoted_070']} promoted past the 0.70 precision gate")
    v = s.get("vocals") or {}
    if v.get("status") == "unavailable":
        print("vocal pass skipped: torch/demucs not installed "
              "(pip install -r requirements-dj-vocals.txt)")
    elif v:
        print(f"vocal pass: measured {v.get('measured', 0)} tracks"
              + (f", {v['errors']} errors" if v.get("errors") else ""))
    c = s["db_counts"]
    print(f"library now: {c['total']} tracks ({c['errors']} errored, "
          f"{c['missing']} missing)")
    return 0


def cmd_track(args):
    from lib.dj.features import analyze_file
    path = args.track
    if not os.path.isfile(path):
        path = os.path.join(resolve_music_dir(args.dir), args.track)
    print(f"Analyzing {path} ...")
    r = analyze_file(path, deep=True)
    curve = r.pop("energy_curve", [])
    print(json.dumps(r, indent=1))
    if curve:
        # Coarse energy sketch: one char per 5 s.
        marks = " .:-=+*#%@"
        step = 10
        line = "".join(marks[min(int(v * (len(marks) - 1)), len(marks) - 1)]
                       for v in curve[::step])
        print(f"\nenergy (5s/char): |{line}|")
    return 0


def cmd_list(args):
    from lib.dj.db import LibraryDB
    root = resolve_music_dir(args.dir)
    db = LibraryDB(root)
    tracks = db.all_tracks(include_missing=True, include_errors=True)
    print(f"{'title':40s} {'artist':20s} {'bpm':>6s} {'cf':>4s} "
          f"{'key':>4s} {'dur':>6s} flags")
    print("-" * 92)
    for t in sorted(tracks, key=lambda x: (x["artist"] or "", x["title"] or "")):
        flags = []
        if t["error"]:
            flags.append("ERR")
        if t["missing"]:
            flags.append("MISSING")
        print(f"{(t['title'] or '?')[:40]:40s} {(t['artist'] or '')[:20]:20s} "
              f"{t['bpm'] or 0:6.1f} {t['bpm_conf'] or 0:4.2f} "
              f"{t['camelot'] or '?':>4s} {t['duration_s'] or 0:6.1f} "
              f"{' '.join(flags)}")
    db.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="", help="music library directory")
    ap.add_argument("--force", action="store_true", help="re-analyze all")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--track", default="", help="single-file debug report")
    ap.add_argument("--list", action="store_true", help="dump the library")
    ap.add_argument("--no-vocals", action="store_true",
                    help="skip the ML vocal-measurement pass")
    ap.add_argument("--refine-grids", action="store_true",
                    help="also re-analyze unchanged tracks with bpm_conf"
                         " < 0.75 (vote-weighted grid confidence)")
    args = ap.parse_args()
    if args.track:
        return cmd_track(args)
    if args.list:
        return cmd_list(args)
    return cmd_scan(args)


if __name__ == "__main__":
    sys.exit(main())
