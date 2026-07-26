"""Gate for the Music2Emo mood pass wiring (lib/dj/mood_ml, character.py,
brain.py, DB v10). Does NOT load torch or the real model - it feeds canned
mood blobs and checks the plumbing: 1..9 normalization, the DB round-trip
(mood_ml + file_genre), TrackInfo surfacing ml_valence/arousal/moods,
all_tags folding ML moods + the embedded genre tag, character.valence_raw
PREFERRING ML valence over the heuristic, the danceability mood-nudge, and
rank_library copying arousal through.

Usage: python tools/tests/_dj_mood_test.py
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


def make_row(**over):
    row = {
        "id": 1, "path": "a.mp3", "title": "T", "artist": "",
        "duration_s": 200.0, "bpm": 120.0, "bpm_conf": 0.8,
        "downbeat_offset": 0, "downbeat_conf": 0.0, "camelot": "8A",
        "key_mode": None, "mood_hist": {}, "rhythm_density": 6.0,
        "spectral": {"bass_share": 0.33, "high_share": 0.25},
        "energy_curve": [], "mood_ml": None, "file_genre": "",
    }
    row.update(over)
    return row


def main():
    from lib.dj import mood_ml as MM
    from lib.dj import character as CH
    from lib.dj.brain import TrackInfo, load_library
    print("Music2Emo mood-pass wiring test\n" + "=" * 40 + "\n")

    # -- normalization ---------------------------------------------------------
    check("norm9 maps 1..9 -> 0..1",
          MM._norm9(1) == 0.0 and MM._norm9(9) == 1.0 and MM._norm9(5) == 0.5,
          f"1->{MM._norm9(1)} 5->{MM._norm9(5)} 9->{MM._norm9(9)}")
    check("norm9 clamps out-of-range",
          MM._norm9(0) == 0.0 and MM._norm9(10) == 1.0
          and MM._norm9(None) is None, "0/10/None handled")

    # -- discovery helpers never crash ----------------------------------------
    md = MM.find_model_dir()
    check("find_model_dir returns str or None", md is None or isinstance(md, str),
          f"{md!r}")
    check("available() returns a bool", isinstance(MM.available(), bool),
          f"{MM.available()}")

    sections = [{"kind": "groove", "repetitiveness": 0.8}]

    # -- valence PREFERS ML over heuristic ------------------------------------
    t_ml = TrackInfo(make_row(mood_ml={"valence": 0.9, "arousal": 0.3,
                                       "moods": ["party"]}), sections, [], [])
    check("ml_valence surfaced on TrackInfo",
          t_ml.ml_valence == 0.9 and t_ml.ml_arousal == 0.3
          and t_ml.ml_moods == ["party"], f"val={t_ml.ml_valence}")
    check("valence_raw uses ML valence when present",
          CH.valence_raw(t_ml) == 0.9, f"{CH.valence_raw(t_ml)}")

    t_dark = TrackInfo(make_row(key_mode="minor",
                               spectral={"bass_share": 0.6, "high_share": 0.05}),
                       sections, [], [])
    t_bright = TrackInfo(make_row(key_mode="major",
                                 spectral={"bass_share": 0.2, "high_share": 0.35}),
                         sections, [], [])
    check("heuristic valence used when no ML (dark < bright)",
          CH.valence_raw(t_dark) < 0.5 < CH.valence_raw(t_bright),
          f"dark={CH.valence_raw(t_dark):.2f} bright={CH.valence_raw(t_bright):.2f}")

    # -- danceability nudged by ML mood tags ----------------------------------
    base = make_row()
    t_floor = TrackInfo(dict(base, mood_ml={"valence": 0.5, "arousal": 0.5,
        "moods": ["party", "groovy", "energetic"]}), sections, [], [])
    t_couch = TrackInfo(dict(base, mood_ml={"valence": 0.5, "arousal": 0.5,
        "moods": ["ambient", "soundscape", "sad"]}), sections, [], [])
    check("party moods raise danceability over ambient moods",
          CH.danceability_raw(t_floor) > CH.danceability_raw(t_couch),
          f"floor={CH.danceability_raw(t_floor):.2f} "
          f"couch={CH.danceability_raw(t_couch):.2f}")

    # -- all_tags folds ML moods + embedded genre -----------------------------
    t_tags = TrackInfo(make_row(mood_ml={"valence": 0.2, "arousal": 0.8,
                                         "moods": ["dark", "epic"]},
                                file_genre="Electronic/Dance"),
                       sections, [], [])
    tags = t_tags.all_tags
    check("ML moods fold into all_tags",
          "dark" in tags and "epic" in tags, f"{[x for x in tags if x in ('dark','epic')]}")
    check("embedded genre splits into all_tags",
          "electronic" in tags and "dance" in tags,
          f"{[x for x in tags if x in ('electronic','dance')]}")

    # -- rank_library copies arousal through ----------------------------------
    lib = [TrackInfo(make_row(id=i, path=f"{i}.mp3",
                    mood_ml={"valence": i / 10.0, "arousal": (i % 5) / 5.0,
                             "moods": []}), sections, [], [])
           for i in range(1, 7)]
    CH.rank_library(lib)
    check("rank_library sets arousal from ml_arousal",
          all(t.arousal == t.ml_arousal for t in lib),
          f"arousals={[t.arousal for t in lib]}")
    vals = sorted(t.valence for t in lib)
    check("valence percentile-ranks span 0..1",
          vals[0] == 0.0 and vals[-1] == 1.0, f"{vals}")

    # -- DB v10 round-trip -----------------------------------------------------
    from lib.dj.db import LibraryDB
    tmp = tempfile.mkdtemp(prefix="gl_mood_")
    db = LibraryDB(tmp)
    check("schema migrated to v10",
          db.conn.execute("PRAGMA user_version").fetchone()[0] >= 10,
          f"v{db.conn.execute('PRAGMA user_version').fetchone()[0]}")
    track_path = os.path.join(tmp, "x.mp3")
    open(track_path, "wb").close()
    db.upsert_track(track_path, {"analysis_version": 1, "duration_s": 200,
                                 "bpm": 120, "title": "X", "genre": "Techno"})
    tid = db.all_tracks()[0]["id"]
    check("file_genre persists from scan analysis",
          db.all_tracks()[0].get("file_genre") == "Techno",
          f"{db.all_tracks()[0].get('file_genre')}")
    blob = {"valence": 0.7, "arousal": 0.6, "moods": ["dark", "groovy"],
            "valence9": 6.6, "arousal9": 5.8}
    db.set_mood_ml(tid, blob)
    got = db.mood_ml_for(tid)
    check("mood_ml persists + round-trips",
          got and got["valence"] == 0.7 and got["moods"] == ["dark", "groovy"],
          f"{got}")
    row = db.all_tracks()[0]
    t = TrackInfo(row, [], [], [])
    check("reloaded TrackInfo carries ML + genre into all_tags",
          t.ml_valence == 0.7 and "dark" in t.all_tags and "techno" in t.all_tags,
          f"ml_val={t.ml_valence} tags={[x for x in t.all_tags if x in ('dark','techno')]}")

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    # -- brain USES the ML mood (score factors fire only when scored) ----------
    from tools.tests._dj_brain_test import fake_track
    from lib.dj.brain import Brain
    from lib.dj.themes import get_theme

    def scored(t, valence, arousal=0.5, dance=0.5):
        t.ml_valence, t.ml_arousal = valence, arousal
        t.arousal_rank, t.danceability = arousal, dance
        return t

    # Build a FULLY mood-scored library so the coverage gate (>=80%) opens.
    cur = scored(fake_track(10, 122, "8A"), 0.2)
    near = scored(fake_track(11, 122, "8A"), 0.25, dance=0.65)
    far = scored(fake_track(12, 122, "8A"), 0.9, dance=0.65)
    d_hi = scored(fake_track(13, 122, "8A"), 0.5, dance=0.9)
    d_lo = scored(fake_track(14, 122, "8A"), 0.5, dance=0.2)
    b = Brain([cur, near, far, d_hi, d_lo], get_theme("groove"))  # dance_t 0.65
    b.rng.uniform = lambda a, c: 1.0            # kill the ±10% jitter
    check("brain enables mood steering on a scored library", b._use_mood,
          f"_use_mood={b._use_mood}")

    # a partially-scored library must NOT enable it (the regression guard)
    half = Brain([cur, near, fake_track(99, 122, "8A")], get_theme("groove"))
    check("partial-scored library keeps mood steering OFF",
          not half._use_mood, f"_use_mood={half._use_mood}")

    # arousal blends into the arc energy; absent -> energy_proxy unchanged
    t_ar = fake_track(1, 122, "8A")
    t_ar.energy_rank, t_ar.arousal_rank = 0.2, 0.9
    check("arc energy blends arousal when present",
          abs(b._arc_energy(t_ar) - (0.6 * 0.2 + 0.4 * 0.9)) < 1e-9,
          f"{b._arc_energy(t_ar):.3f}")
    t_no = fake_track(2, 122, "8A")
    t_no.energy_rank = 0.2
    check("arc energy falls back to energy_proxy when unscored",
          abs(b._arc_energy(t_no) - 0.2) < 1e-9, f"{b._arc_energy(t_no):.3f}")

    # VALENCE CONTINUITY: from a dark current track, a mood-near candidate
    # outscores a bright one (all else equal).
    s_near = b.score(cur, near, 0.5, 122.0)[0]
    s_far = b.score(cur, far, 0.5, 122.0)[0]
    check("valence continuity favors the mood-near track",
          s_near > s_far * 1.2, f"near={s_near:.3f} far={s_far:.3f}")

    # DANCEABILITY TARGET: same valence, groove pulls toward high danceability
    check("dance_target favors the more danceable track",
          b.score(cur, d_hi, 0.5, 122.0)[0]
          > b.score(cur, d_lo, 0.5, 122.0)[0],
          f"hi={b.score(cur, d_hi, 0.5, 122.0)[0]:.3f} "
          f"lo={b.score(cur, d_lo, 0.5, 122.0)[0]:.3f}")

    # GATING: an UNSCORED candidate is untouched by valence/dance factors
    plain = fake_track(15, 122, "8A")           # no ml_valence
    s_plain_dark = b.score(cur, plain, 0.5, 122.0)[0]
    bright_cur = scored(fake_track(17, 122, "8A"), 0.95)
    s_plain_bright = b.score(bright_cur, plain, 0.5, 122.0)[0]
    check("unscored candidate ignores mood factors (gating)",
          abs(s_plain_dark - s_plain_bright) < 1e-6,
          f"dark-cur={s_plain_dark:.4f} bright-cur={s_plain_bright:.4f}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
