"""DJ library scanner CLI.

Incrementally analyzes every audio file in the music library (default:
<repo_parent>/music) into music/dj_library.sqlite3. Already-analyzed,
unchanged files are skipped; add songs and only the new ones get scanned.

Usage:
    python tools/dj/dj_scan.py                     # incremental scan, default dir
    python tools/dj/dj_scan.py --dir D:/music      # explicit library
    python tools/dj/dj_scan.py --force             # re-analyze everything
    python tools/dj/dj_scan.py --refine-grids      # + re-run low-conf beat grids
    python tools/dj/dj_scan.py --workers 4
    python tools/dj/dj_scan.py --track "song.mp3"  # single-file debug report
    python tools/dj/dj_scan.py --list              # dump the library table
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
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
    _phase_fill(root)
    _grid_health(root)
    return 0


def _phase_fill(root):
    """Fill the music-vs-grid PHASE profile for tracks that lack it.

    This belongs in the scan and was missing from it (2026-08-18).
    `beatpower.compute_phase` shipped 2026-08-04 behind its own CLI
    (`python lib/dj/beatpower.py --phase`) and nothing ever wired it
    into the scan, so every track added since arrived with NO phase
    profile - and phase is what the sync bias is computed from. An
    unprofiled track cannot be corrected, so its kicks land wherever
    its stored grid happens to sit.

    Measured before this existed: 40% of a 1,206-track library had
    coverage under 0.6 and 21% had none at all, and the tracks that
    flammed on EVERY seam (20 of them, vs 0.1% expected by chance)
    carried high grid confidence but only 50% phase coverage, with a
    flam that was a fixed fraction of a beat - the signature of a grid
    sitting off the real kicks with nothing measured to correct it.
    Filling this is the fix for that class, and doing it here is what
    keeps it fixed as the library grows.

    Incremental and resumable: only tracks with no profile are read.

    Delegates to beatpower's own CLI rather than re-implementing the
    read-merge-save of the shared profile file: that loop is already
    incremental, already skips what is profiled, and is the code the
    manual pass has always used. A second implementation of the same
    merge is how two writers end up disagreeing about one file.
    """
    import subprocess
    import sys as _sys
    from lib.dj import beatpower as bp
    from lib.dj.brain import load_library
    from lib.dj.db import LibraryDB
    db = LibraryDB(root)
    try:
        lib = load_library(db)
        todo = [t for t in lib if bp.profile_coverage(t.id) <= 0.0]
        n_lib = len(lib)
    finally:
        db.close()
    if not todo:
        print(f"phase      : all {n_lib} tracks profiled")
        return
    print(f"phase      : {len(todo)} of {n_lib} track(s) unprofiled - "
          f"measuring (one audio pass each)", flush=True)
    script = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        "lib", "dj", "beatpower.py")
    r = subprocess.run([_sys.executable, script, "--music", root,
                        "--phase"])
    if r.returncode != 0:
        print(f"phase      : pass exited {r.returncode} - re-run with "
              f"`python lib/dj/beatpower.py --music {root} --phase`")


def _grid_health(root):
    """Report the LOOSE-GRID TAIL, because it silently sets the ceiling on
    everything the DJ can do and nothing used to say so.

    A track under bpm_conf 0.5 forces long_fade on BOTH its seams - it can
    never be beat-matched - and one under 0.7 takes the whole precision
    style tier off the menu for every pair it touches. On the real library
    that was 200 of 680 tracks (29%) and 253 (37%) respectively, which is
    the actual reason 18% of logged seams were fades. `--refine-grids`
    re-runs exactly these and promotes the ones that were only unlucky."""
    from lib.dj.db import LibraryDB
    db = LibraryDB(root)
    try:
        row = db.conn.execute(
            "SELECT COUNT(*) n,"
            " SUM(CASE WHEN bpm_conf < 0.5 THEN 1 ELSE 0 END) loose,"
            " SUM(CASE WHEN bpm_conf < 0.7 THEN 1 ELSE 0 END) soft"
            " FROM tracks WHERE error IS NULL AND bpm_conf IS NOT NULL"
        ).fetchone()
    finally:
        db.close()
    n = row["n"] or 0
    if not n:
        return
    loose, soft = row["loose"] or 0, row["soft"] or 0
    print(f"beat grids : {n - soft} confident | {soft - loose} soft "
          f"(<0.70, no precision styles) | {loose} loose "
          f"(<0.50, fade-only)")
    if loose > n * 0.12:
        # Pair math: with a fraction f of the library fade-only, the odds
        # that a random pair touches one is 1-(1-f)^2 - it compounds fast.
        f = loose / n
        print(f"  -> {100*f:.0f}% fade-only means ~{100*(1-(1-f)**2):.0f}% of "
              "PAIRS can never beat-match.")
        print("     `python tools/dj/dj_scan.py --refine-grids` re-runs them.")


def cmd_retag(args):
    """Re-derive the library-relative auto tags from data already stored.

    No audio is touched - scan._recalibrate_tags just re-runs the
    percentile thresholds over the analysis in the DB, so this is seconds
    even on a big library. Worth running on its own whenever the tagging
    rules change: the 2026-07-24 NaN fix, for instance, restored the
    'driving' and 'mellow' tags that ten poisoned energy axes had silently
    erased from EVERY track, and themes that lean on those words stay inert
    until the stored tags catch up.

    Only `auto_tags` is rewritten. User tags are a separate table and are
    never touched."""
    from lib.dj.db import LibraryDB
    from lib.dj.scan import _recalibrate_tags
    import json as _json
    from collections import Counter
    root = resolve_music_dir(args.dir)
    db = LibraryDB(root)
    before = Counter()
    for r in db.conn.execute("SELECT auto_tags FROM tracks"):
        for t in _json.loads(r["auto_tags"] or "[]"):
            before[t] += 1
    print(f"Re-deriving auto tags in {root}", flush=True)
    _recalibrate_tags(db)
    after = Counter()
    n = 0
    for r in db.conn.execute("SELECT auto_tags FROM tracks"):
        n += 1
        for t in _json.loads(r["auto_tags"] or "[]"):
            after[t] += 1
    db.close()
    changed = [(t, before.get(t, 0), after.get(t, 0))
               for t in sorted(set(before) | set(after))
               if before.get(t, 0) != after.get(t, 0)]
    print(f"{n} tracks | {len(after)} distinct tags | "
          f"{len(changed)} tags changed count")
    for t, b, a in sorted(changed, key=lambda r: -abs(r[2] - r[1]))[:20]:
        mark = "  NEW" if not b else ("  GONE" if not a else "")
        print(f"  {t:16} {b:4d} -> {a:4d}{mark}")
    return 0


def cmd_revocals(args):
    from lib.dj.db import LibraryDB
    from lib.dj.scan import vocal_pass
    root = resolve_music_dir(args.dir)
    db = LibraryDB(root)
    print(f"Refining vocal curves in {root}", flush=True)

    def cb(d, t, n):
        _bar(d, t, n)
        # Machine-parseable line for the planner pipeline label (same
        # shape as the mood/structure/stems passes emit).
        print(f"\nPROGRESS {d} {t} {d} 0 {n[:40]}", flush=True)

    v = vocal_pass(db, progress_cb=cb, refine=True)
    print()
    if v.get("status") == "unavailable":
        print("torch/demucs not installed "
              "(pip install -r requirements-dj-vocals.txt)")
        return 1
    print(f"measured {v.get('measured', 0)} tracks"
          + (f", {v['errors']} errors" if v.get("errors") else ""))
    db.close()
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
    ap.add_argument("--retag", action="store_true",
                    help="ONLY re-derive the library-relative auto tags "
                         "from stored analysis (seconds, no audio decode) - "
                         "run after the tagging rules change")
    ap.add_argument("--revocals", action="store_true",
                    help="ONLY re-run the ML vocal pass, upgrading tracks "
                         "measured at the old coarse 24s hop to the fine "
                         "8s vocal curve (no audio re-analysis)")
    args = ap.parse_args()
    if args.track:
        return cmd_track(args)
    if args.list:
        return cmd_list(args)
    if args.retag:
        return cmd_retag(args)
    if args.revocals:
        return cmd_revocals(args)
    return cmd_scan(args)


if __name__ == "__main__":
    sys.exit(main())
