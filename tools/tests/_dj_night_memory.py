"""One night's memory survives a restart (Director._seed_night): 'played' rows written by an earlier session are
read at construction, the songs count as played tonight (the opener, the chain, the lists avoid them), and rows
older than NIGHT_HOURS do not. Runs in seconds, no audio.

    python tools/tests/_dj_night_memory.py --music D:/Devel/music
"""
import json
import os
import sys
import tempfile
import time

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.tests._low_priority import PRIORITY  # noqa: E402


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    from lib.dj import director as D
    d0 = D.Director(music, engine=None, theme="groove")
    stems = [t for t in d0.library if getattr(t, "has_stems", False) and t.bpm]
    if len(stems) < 4:
        print("FAIL: too few stem songs")
        return 1
    tmp = tempfile.mkdtemp(prefix="dj_night_")
    now = time.time()
    a, b, c = stems[0], stems[1], stems[2]
    rows = [{"t": now - 600, "sid": "old1", "event": "played", "track_id": a.id, "title": a.title},          # ten minutes ago: tonight
            {"t": now - 3 * 3600, "sid": "old1", "event": "played", "track_id": b.id, "title": b.title},     # three hours ago: tonight
            {"t": now - (D.NIGHT_HOURS + 1) * 3600, "sid": "old0", "event": "played", "track_id": c.id, "title": c.title},  # nine hours ago: not tonight
            {"t": now - 300, "sid": "old1", "event": "flag", "track_id": a.id}]                                 # (not a play)
    with open(os.path.join(tmp, f"dj_{time.strftime('%Y%m%d')}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    d = D.Director(music, engine=None, theme="groove", played_log_dir=tmp)
    played = d._played_ids()
    print(f"seeded {d._night_seeded} rows; played tonight: {sorted(played)} (expected {sorted({a.id, b.id})})")
    ok = played == {a.id, b.id}
    # the opener never comes from tonight's songs: the conductor's avoid list is the seeded set
    rc = d._new_conductor(attach=False)
    if not set(rc.avoid_ids or ()) >= {a.id, b.id}:
        print(f"FAIL: the conductor's avoid list lacks the night's songs: {rc.avoid_ids}")
        ok = False
    # the SONGS list says it
    rows_r = {r["id"]: r for r in d.rank(n=100000)}
    for t in (a, b):
        r = rows_r.get(t.id)
        if r is None or r["ok"] or "played" not in r["why"]:
            print(f"FAIL: SONGS does not mark {t.title[:30]} as played: {r and r['why'][:60]}")
            ok = False
    # a headless Director reads nothing by default (the gates stay deterministic)
    d2 = D.Director(music, engine=None, theme="groove")
    if d2._night_seeded:
        print(f"FAIL: a headless Director seeded {d2._night_seeded} rows from the real log")
        ok = False
    print("ALL OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
