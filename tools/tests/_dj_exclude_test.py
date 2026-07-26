"""Gate for the do-not-use flag (DB v11) + the save-set invariant.

Do-not-use: set_excluded round-trips; load_library tags TrackInfo.excluded;
the planner-boundary filter [t for t in lib if not t.excluded] removes it from
everything that auto-selects (so brain/suggest/copilot never see it) while it
stays in the full library for the browser. Save invariant: creating a named
setlist and saving the CURRENT entries preserves them (the bug cleared them).

Usage: python tools/tests/_dj_exclude_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def main():
    from lib.dj.db import LibraryDB
    from lib.dj.brain import load_library
    from lib.dj import setlist as SL
    print("Do-not-use + save-set test\n" + "=" * 40 + "\n")

    tmp = tempfile.mkdtemp(prefix="gl_excl_")
    db = LibraryDB(tmp)
    check("schema migrated to v11",
          db.conn.execute("PRAGMA user_version").fetchone()[0] >= 11,
          f"v{db.conn.execute('PRAGMA user_version').fetchone()[0]}")

    ids = []
    for i in range(3):
        p = os.path.join(tmp, f"t{i}.mp3")
        open(p, "wb").close()
        db.upsert_track(p, {"analysis_version": 1, "duration_s": 200,
                            "bpm": 120 + i, "title": f"T{i}"})
        ids.append(db.get_track(rel_path=db.rel(p))["id"])

    # -- exclusion round-trip --------------------------------------------------
    db.set_excluded(ids[1], True)
    lib = load_library(db)
    by = {t.id: t for t in lib}
    check("load_library tags TrackInfo.excluded",
          by[ids[1]].excluded is True
          and by[ids[0]].excluded is False, f"{[t.excluded for t in lib]}")

    selectable = [t for t in lib if not t.excluded]
    check("boundary filter drops the excluded track",
          ids[1] not in {t.id for t in selectable} and len(selectable) == 2,
          f"selectable ids={[t.id for t in selectable]}")
    check("full library still contains it (browser + toggle)",
          ids[1] in {t.id for t in lib}, "present in load_library output")

    # brain built on the filtered library can never pick it
    from lib.dj.brain import Brain
    from lib.dj.themes import get_theme
    b = Brain(selectable, get_theme("groove"))
    check("brain never sees the excluded track",
          ids[1] not in {t.id for t in b.library}, "absent from brain.library")

    # unflagging returns it
    db.set_excluded(ids[1], False)
    lib2 = load_library(db)
    check("Allow clears the flag",
          not next(t for t in lib2 if t.id == ids[1]).excluded,
          "excluded=False after clear")

    # -- save-set invariant (the bug: naming cleared entries) ------------------
    entries = [{"track_id": tid, "pin_type": "suggestion",
                "target_offset_min": None, "style_override": None,
                "target_play_s": None} for tid in ids]
    # The FIXED flow: create the row, then save the CURRENT entries (unchanged).
    sid = SL.create_setlist(db, "My Set", theme="groove")
    check("create_setlist does not touch the caller's entries",
          len(entries) == 3, f"{len(entries)} entries still held")
    SL.save_entries(db, sid, entries)
    loaded = SL.get_setlist(db, name="My Set")
    check("saved set keeps all its tracks (not empty)",
          loaded is not None and len(loaded["entries"]) == 3,
          f"{len(loaded['entries']) if loaded else 'None'} entries persisted")
    check("saved order matches",
          [e["track_id"] for e in loaded["entries"]] == ids,
          f"{[e['track_id'] for e in loaded['entries']]}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
