#!/usr/bin/env python
"""Headless "Analyze all": the planner's full analysis pipeline as a CLI.

Runs the same stage list the Library tab's ⟳ Analyze all builds (via
lib/dj/analyze.py): scan -> chroma -> [stems] -> rhythm -> vocal curves ->
enrich -> mood -> structure. Every stage skips already-analyzed tracks, so
re-running is cheap and an interrupted run resumes where it left off.

    python tools/dj/dj_analyze.py                     # default music dir
    python tools/dj/dj_analyze.py --dir D:/music
    python tools/dj/dj_analyze.py --stems             # include stem render
    python tools/dj/dj_analyze.py --only rhythm       # one stage
    python tools/dj/dj_analyze.py --list              # show the stage plan

Use this to analyze a library overnight (or from a script/cron) without
holding a Qt window open. Grid refinement stays deliberate:
dj_scan.py --refine-grids.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj import resolve_music_dir
from lib.dj.analyze import build_stages, run_stages
from lib.dj.db import LibraryDB


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the full DJ analysis pipeline headless.")
    ap.add_argument("--dir", default="", help="music library directory")
    ap.add_argument("--stems", action="store_true",
                    help="include the stem render stage (disk-hungry, "
                    "gates the stem transition styles)")
    ap.add_argument("--only", metavar="STAGE",
                    help="run a single stage by name (scan, chroma, stems, "
                    "rhythm, 'vocal curves', enrich, mood, structure)")
    ap.add_argument("--list", action="store_true",
                    help="print the stage plan and exit")
    a = ap.parse_args(argv)

    music_dir = resolve_music_dir(a.dir)
    stages = build_stages(music_dir, include_stems=a.stems or bool(a.only),
                          headless=True)
    if a.only:
        stages = [s for s in stages if s["name"] == a.only]
        if not stages:
            print(f"unknown stage '{a.only}'")
            return 2
    if a.list:
        for k, s in enumerate(stages, 1):
            print(f"  {k}. {s['name']}"
                  + (f"  (skip: {s['skip']})" if s.get("skip") else ""))
        return 0

    print(f"analyzing {music_dir} ({len(stages)} stages)")
    db = LibraryDB(music_dir)
    try:
        summary = run_stages(
            music_dir, db, stages,
            on_stage=lambda k, n, name: print(f"\n[{k}/{n}] {name}"),
            on_line=lambda line: print("  " + line))
    finally:
        db.close()

    print("\npipeline " + ("complete" if not summary["failed"]
                           else "finished WITH FAILURES"))
    for name, why in summary["skipped"]:
        print(f"  skipped: {name} ({why})")
    for name, why in summary["failed"]:
        print(f"  FAILED: {name} ({why})")
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
