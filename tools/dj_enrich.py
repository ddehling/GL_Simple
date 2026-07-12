"""Enrich the library with MusicBrainz metadata: genre, release year/era,
label, canonical identity. Fills the gap the audio scanner leaves - it
learns bpm/key/energy from the waveform but never a track's genre or era.

Genres + decade become tags (in memory, via TrackInfo.all_tags) that steer
selection, the flavor system, and the Set Copilot; the full blob is stored
on the track. Incremental: already-enriched tracks are skipped unless
--force. MusicBrainz is rate-limited to ~1 req/s, so a big library takes a
while - it's resumable (each track commits).

Usage:
    python tools/dj_enrich.py                 # enrich all un-enriched tracks
    python tools/dj_enrich.py --limit 50
    python tools/dj_enrich.py --force         # re-enrich everything
    python tools/dj_enrich.py --stats         # coverage report
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dj import resolve_music_dir
from lib.dj.db import LibraryDB
from lib.dj.enrich import MusicBrainzClient, enrich_track


def cmd_enrich(args):
    root = resolve_music_dir(args.dir)
    db = LibraryDB(root)
    rows = db.all_tracks()
    todo = [r for r in rows
            if args.force or not r.get("enrichment")]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} tracks, {len(todo)} to enrich "
          f"(~{len(todo) * 1.1 / 60:.0f} min at 1 req/s)\n")
    mb = MusicBrainzClient()
    matched = missed = 0
    for i, r in enumerate(todo):
        title = r.get("title") or ""
        artist = r.get("artist") or ""
        track = {"title": title, "artist": artist,
                 "duration_s": r.get("duration_s")}
        try:
            blob = enrich_track(track, mb=mb)
        except Exception as e:
            blob = None
            print(f"  ! {title[:30]}: {type(e).__name__}: {e}")
        db.set_enrichment(r["id"], blob or {"source": "musicbrainz",
                                            "matched": False})
        if blob:
            matched += 1
            g = ", ".join(blob["genres"][:3]) or "-"
            print(f"  [{i+1}/{len(todo)}] {title[:34]:34s} -> "
                  f"{blob.get('year') or '????'}  {g}  (conf "
                  f"{blob['confidence']})")
        else:
            missed += 1
            print(f"  [{i+1}/{len(todo)}] {title[:34]:34s} -> no match")
    print(f"\nmatched {matched}, no-match {missed}")
    db.close()
    return 0


def cmd_stats(args):
    root = resolve_music_dir(args.dir)
    db = LibraryDB(root)
    rows = db.all_tracks()
    enriched = [r for r in rows if r.get("enrichment")]
    matched = [r for r in enriched
               if (r["enrichment"] or {}).get("mbid")]
    genres = {}
    decades = {}
    for r in matched:
        for g in r["enrichment"].get("genres", []):
            genres[g] = genres.get(g, 0) + 1
        d = r["enrichment"].get("decade")
        if d:
            decades[d] = decades.get(d, 0) + 1
    print(f"library: {len(rows)} tracks")
    print(f"enriched: {len(enriched)}  matched: {len(matched)}  "
          f"unmatched: {len(enriched) - len(matched)}")
    if genres:
        print("\ntop genres:")
        for g, n in sorted(genres.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {n:4d}  {g}")
    if decades:
        print("\ndecades:")
        for d, n in sorted(decades.items()):
            print(f"  {n:4d}  {d}")
    db.close()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default="", help="music library directory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="re-enrich already-enriched tracks")
    ap.add_argument("--stats", action="store_true",
                    help="coverage/genre report only")
    args = ap.parse_args()
    return cmd_stats(args) if args.stats else cmd_enrich(args)


if __name__ == "__main__":
    sys.exit(main())
