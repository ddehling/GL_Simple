"""Gate for MusicBrainz enrichment (lib/dj/enrich.py) + DB v9.

Runs against a canned transport - no network, matching MusicBrainz's real
response shape (verified against the live API). Covers: search parsing,
match scoring (title/artist/duration disambiguation + rejection), the full
enrichment blob (mbid/year/decade/era/genres/tags/confidence), the DB
round-trip, and TrackInfo surfacing genres/year into all_tags.

Usage: python tools/_dj_enrich_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


# Canned MusicBrainz responses (real shape).
SEARCH = {"recordings": [
    {"id": "mbid-kerala", "score": 100, "title": "Kerala",
     "length": 225000, "first-release-date": "2016-01-15",
     "artist-credit": [{"name": "Bonobo",
                        "artist": {"id": "a1", "name": "Bonobo"}}],
     "releases": [{"date": "2017-01-13",
                   "label-info": [{"label": {"name": "Ninja Tune"}}]}],
     "isrcs": ["GBCEN1600001"], "tags": [{"name": "electronic", "count": 3}]},
    # A decoy: same title, different artist, very different length.
    {"id": "mbid-other", "score": 80, "title": "Kerala",
     "length": 120000,
     "artist-credit": [{"name": "Someone Else",
                        "artist": {"id": "a2", "name": "Someone Else"}}]},
]}
GENRES = {"genres": [{"name": "downtempo", "count": 5},
                     {"name": "electronic", "count": 3}],
          "tags": [{"name": "chill", "count": 2}],
          "artist-credit": [{"artist": {"genres": [
              {"name": "trip hop", "count": 4}]}}]}


def fake_transport(method, url, headers=None, params=None, timeout=20):
    if "recording/mbid-kerala" in url:
        return 200, GENRES
    if url.rstrip("/").endswith("recording"):
        return 200, SEARCH
    return 404, None


def main():
    from lib.dj import enrich as EN
    print("MusicBrainz enrichment test\n" + "=" * 40 + "\n")

    mb = EN.MusicBrainzClient(transport=fake_transport)

    # -- search + match --------------------------------------------------------
    recs = mb.search_recording("Bonobo", "Kerala")
    check("search parses recordings", len(recs) == 2
          and recs[0]["id"] == "mbid-kerala", f"{len(recs)} recs")
    rec, conf = EN.best_match(recs, "Bonobo", "Kerala", duration_s=225)
    check("match picks the right recording by duration",
          rec is not None and rec["id"] == "mbid-kerala" and conf > 0.7,
          f"id={rec['id'] if rec else None} conf={conf}")
    # No artist given -> match on title alone.
    rec2, c2 = EN.best_match(recs, "", "Kerala")
    check("matches on title alone when no artist given",
          rec2 is not None, f"conf={c2}")
    # A wrong artist must be rejected (disambiguation guard).
    wrong, cw = EN.best_match(recs, "Nobody At All", "Kerala")
    check("wrong artist is rejected", wrong is None, f"got {wrong}")
    none, c3 = EN.best_match(recs, "X", "Totally Different Song")
    check("no title match -> None", none is None, f"got {none}")

    # -- full enrich blob ------------------------------------------------------
    blob = EN.enrich_track({"title": "Kerala", "artist": "Bonobo",
                            "duration_s": 225}, mb=mb)
    check("enrich blob complete",
          blob and blob["mbid"] == "mbid-kerala" and blob["year"] == 2016
          and blob["decade"] == "2010s" and blob["era"] == "modern"
          and "downtempo" in blob["genres"]
          and blob["isrc"] == "GBCEN1600001"
          and blob["label"] == "Ninja Tune",
          f"year={blob.get('year')} genres={blob.get('genres')[:3]} "
          f"label={blob.get('label')}")
    check("enrich derives genre+decade tags",
          "downtempo" in blob["tags"] and "2010s" in blob["tags"],
          f"tags={blob['tags']}")
    check("artist genres broaden coverage",
          "trip hop" in blob["genres"], f"genres={blob['genres']}")

    # No confident match -> None (don't write garbage).
    empty = EN.MusicBrainzClient(
        transport=lambda *a, **k: (200, {"recordings": []}))
    check("no match returns None",
          EN.enrich_track({"title": "Ghost", "artist": "Nobody"},
                          mb=empty) is None, "empty search -> None")

    # -- DB round-trip + TrackInfo ---------------------------------------------
    from lib.dj.db import LibraryDB
    from lib.dj.brain import TrackInfo
    tmp = tempfile.mkdtemp(prefix="gl_enrich_")
    db = LibraryDB(tmp)
    # Insert a minimal track row (upsert stats the file, so create it).
    track_path = os.path.join(tmp, "x.mp3")
    open(track_path, "wb").close()
    db.upsert_track(
        track_path,
        {"analysis_version": 1, "duration_s": 225, "bpm": 120,
         "title": "Kerala", "artist": "Bonobo"})
    tid = db.all_tracks()[0]["id"]
    db.set_enrichment(tid, blob)
    got = db.enrichment_for(tid)
    check("enrichment persists to DB",
          got and got["mbid"] == "mbid-kerala"
          and got["genres"] == blob["genres"],
          f"stored genres={got.get('genres') if got else None}")
    # Reload through the library and check TrackInfo surfaces it.
    row = db.all_tracks()[0]
    t = TrackInfo(row, [], [], [])
    check("TrackInfo exposes genres + year",
          t.genres == blob["genres"] and t.year == 2016
          and t.decade == "2010s",
          f"genres={t.genres} year={t.year}")
    check("genres + decade fold into all_tags",
          "downtempo" in t.all_tags and "2010s" in t.all_tags,
          f"all_tags has {[x for x in t.all_tags if x in ('downtempo','2010s')]}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
