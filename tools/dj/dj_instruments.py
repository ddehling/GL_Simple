"""Instrument pass - identify every track's OWN instruments and the note
each plays per beat, from its rendered stems (lib/dj/instruments.py).

Runs after the stem render: for each track with stems on disk and no
current instruments.json, the four stems are decoded and read - drum
sounds by onset-timbre clustering with coincidences resolved, plucked /
struck sounds by harmonic profile with a drone-subtracted pitch per
onset, held notes beat by beat on the harmonic part - and the result is
stored next to the stems (<music_root>/.stems/<id>/instruments.json).
Pure numpy / librosa / scikit-learn: no torch, no GPU; ~16 s per minute
of audio on one core (measured over a library sample). Incremental: tracks with a current result are
skipped unless --force.

This is the subprocess the planner's pipeline stage and the Analysis
tab's button spawn; it prints `PROGRESS <done> <total> <ok> <failed>
<title>` lines.

Usage:
    python tools/dj/dj_instruments.py                  # every track with stems
    python tools/dj/dj_instruments.py --track 42       # one track (re-runs)
    python tools/dj/dj_instruments.py --limit 20
    python tools/dj/dj_instruments.py --force
    python tools/dj/dj_instruments.py --stats          # coverage
    python tools/dj/dj_instruments.py --show 42        # print a track's instruments + a few beats
    python tools/dj/dj_instruments.py --render 42 --ids bass.1,drums.1 --out recon.wav   # play the reading back
    python tools/dj/dj_instruments.py --export-gen 42  # gen-console material + script.yaml under logs/analysis/<title>/
    python tools/dj/dj_instruments.py --program 42 --render-program   # the SongProgram (lib/dj/songprogram.py) + a render
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj import resolve_music_dir
from lib.dj.db import LibraryDB
from lib.dj import stems as ST
from lib.dj import instruments as INS


def _rows(db, args):
    rows = db.all_tracks()
    root = db.music_root
    todo = []
    for r in rows:
        if r.get("missing") or r.get("error"):
            continue
        if args.track and r["id"] != args.track:
            continue
        if not ST.has_stems(root, r["id"]):
            continue
        if not args.force and not args.track and INS.has_instruments(root, r["id"]):
            continue
        todo.append(r)
    return rows, todo


def cmd_run(args):
    root = os.path.abspath(resolve_music_dir(args.dir))
    db = LibraryDB(root)
    rows, todo = _rows(db, args)
    if args.limit:
        todo = todo[:args.limit]
    total = len(todo)
    print(f"{len(rows)} tracks, {total} to read", flush=True)
    ok = failed = 0
    for i, r in enumerate(todo, 1):
        title = (r.get("title") or r["path"])[:40]
        t0 = time.time()
        try:
            res = INS.identify_track(root, r["id"], r.get("beat_grid") or [], r.get("downbeat_offset") or 0,
                                     r.get("duration_s") or 0.0, bpm=r.get("bpm"))
            n = len(INS.instruments(res))
            ok += 1
            print(f"  [{i}/{total}] {title:40s} {n:2d} instruments  {time.time() - t0:4.0f} s", flush=True)
            for why in res.get("reasons") or []:
                print(f"        ! {why}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [{i}/{total}] {title:40s} -> {type(e).__name__}: {e}", flush=True)
        print(f"PROGRESS {i} {total} {ok} {failed} {title}", flush=True)
    print(f"\nread {ok}, failed {failed}", flush=True)
    db.close()
    return 0


def cmd_stats(args):
    root = os.path.abspath(resolve_music_dir(args.dir))
    db = LibraryDB(root)
    rows = [r for r in db.all_tracks() if not r.get("missing") and not r.get("error")]
    with_stems = [r for r in rows if ST.has_stems(root, r["id"])]
    have = [r for r in with_stems if INS.has_instruments(root, r["id"])]
    stale = sum(1 for r in with_stems if not INS.has_instruments(root, r["id"]) and INS.load(root, r["id"], any_version=True))
    print(f"library: {len(rows)} tracks, stems: {len(with_stems)}, instruments read: {len(have)}"
          + (f" ({stale} from an older version)" if stale else ""))
    db.close()
    return 0


def cmd_show(args):
    root = os.path.abspath(resolve_music_dir(args.dir))
    res = INS.load(root, args.show, any_version=True)
    if res is None:
        print("no instruments for that track (run the pass first)")
        return 1
    for line in INS.summary(res):
        print(line)
    for why in res.get("reasons") or []:
        print("!", why)
    table = INS.beat_table(res)
    n = len(res["beats"])
    k0 = max(0, min(n - 8, args.beat))
    print(f"\nbeats {k0}-{k0 + 7}:")
    for k in range(k0, min(n, k0 + 8)):
        bar, bib = INS.bar_beat(res, k)
        what = []
        for inst, ev in table.get(k, []):
            note = f":{INS.note_name(ev[2])}" if ev[2] is not None else ""
            step = f"+{ev[1]}" if ev[1] else ""
            what.append(f"{inst['id']}{step}{note}")
        print(f"  bar {bar:3d}.{bib}  {' '.join(what)}")
    return 0


def cmd_render(args):
    from lib.dj import resynth as RS
    import soundfile as sf
    root = os.path.abspath(resolve_music_dir(args.dir))
    res = INS.load(root, args.render)
    if res is None:
        print("no current instruments for that track (run the pass first)")
        return 1
    stems = RS.load_stems_mono(root, args.render)
    ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else None
    audio = RS.render(res, stems, ids=ids, t0=args.t0 or None, t1=args.t1 or None,
                      progress=lambda i, n, name: print(f"  {i}/{n} {name}", flush=True))
    out = args.out or f"recon_{args.render}.wav"
    sf.write(out, audio, INS.RATE)
    print(f"wrote {out} ({len(audio) / INS.RATE:.0f} s, {len(ids) if ids else len(INS.instruments(res))} instruments)")
    return 0


def cmd_program(args):
    """Build the SongProgram for a track, print its summary, save it next
    to the gen material, optionally render it."""
    from lib.dj import resynth as RS, songprogram as SP, gen_link as GL
    root = os.path.abspath(resolve_music_dir(args.dir))
    res = INS.load(root, args.program)
    if res is None:
        print("no current instruments for that track (run the pass first)")
        return 1
    db = LibraryDB(root)
    row = next((r for r in db.all_tracks() if r["id"] == args.program), None)
    secs = db.sections_for(args.program) if row else []
    db.close()
    stems = RS.load_stems_mono(root, args.program)
    t0 = time.time()
    prog = SP.build(res, stems, sections=[{k: v for k, v in s.items() if k in ("kind", "start_s", "end_s", "energy")} for s in secs],
                    progress=print)
    print(f"built in {time.time() - t0:.0f} s")
    print(SP.describe(prog))
    folder = args.out or GL.folder_for(row.get("title") if row else "", args.program)
    print("saved:", SP.save(prog, folder))
    if args.render_program:
        import soundfile as sf
        audio = SP.mixdown(prog, stems)
        out = os.path.join(folder, "program_render.wav")
        sf.write(out, audio, INS.RATE)
        print("rendered:", out)
    return 0


def cmd_export_gen(args):
    from lib.dj import resynth as RS
    import re
    root = os.path.abspath(resolve_music_dir(args.dir))
    db = LibraryDB(root)
    row = next((r for r in db.all_tracks() if r["id"] == args.export_gen), None)
    secs = db.sections_for(args.export_gen) if row else None
    db.close()
    res = INS.load(root, args.export_gen)
    if row is None or res is None:
        print("no such track or no current instruments (run the pass first)")
        return 1
    from lib.dj import gen_link as GL
    folder = GL.link(root, args.export_gen, out_dir=args.out or None, progress=print)
    print("folder:", folder, "(gen console -> Analysis -> Open, or DJ track)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="", help="music library directory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--track", type=int, default=0, help="read ONE track by library id (re-reads)")
    ap.add_argument("--force", action="store_true", help="re-read tracks that already have a current result")
    ap.add_argument("--stats", action="store_true", help="coverage report")
    ap.add_argument("--show", type=int, default=0, help="print a track's instruments and a few beats")
    ap.add_argument("--beat", type=int, default=64, help="with --show: first beat to print")
    ap.add_argument("--render", type=int, default=0, help="reconstruct a track from its reading -> wav (--ids, --t0, --t1, --out)")
    ap.add_argument("--ids", default="", help="with --render: comma-separated instrument ids (default all)")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=0.0)
    ap.add_argument("--out", default="", help="output wav (--render) or directory (--export-gen)")
    ap.add_argument("--export-gen", type=int, default=0, help="write gen-console material + script.yaml for a track")
    ap.add_argument("--program", type=int, default=0, help="build + save the SongProgram of a track (patterns, voices, chords, phrases)")
    ap.add_argument("--render-program", action="store_true", help="with --program: also render it to program_render.wav")
    args = ap.parse_args()
    if args.stats:
        return cmd_stats(args)
    if args.show:
        return cmd_show(args)
    if args.render:
        return cmd_render(args)
    if args.export_gen:
        return cmd_export_gen(args)
    if args.program:
        return cmd_program(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
