"""Phase E gate: setlists - CRUD, compiler, autofill, plan-following, and a
headless smoke of the PyQt6 planner.

Uses the same synthetic-track + real-scanner path as the Phase C e2e so
the compiler runs against genuine analysis rows, then follows a saved
setlist through DJSystem on the hand-pumped engine and asserts the play
order honors the plan.

Usage: python tools/_dj_setlist_test.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RATE = 44100
failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def build_temp_library(tmp):
    from scipy.io import wavfile
    from tools._dj_brain_test import synth_structured
    from lib.dj.scan import scan_library
    specs = [(122.0, 5, 54.0, 220.0), (126.0, 6, 58.0, 262.0),
             (124.0, 7, 50.0, 294.0), (85.0, 8, 46.0, 330.0)]
    for i, (bpm, seed, bass, pad) in enumerate(specs):
        wavfile.write(os.path.join(tmp, f"song_{i}_{int(bpm)}.wav"), RATE,
                      (synth_structured(bpm, 150.0, seed, bass, pad)
                       * 32767).astype(np.int16))
    s = scan_library(tmp, workers=1)
    assert s["errors"] == 0, s
    return s


def main():
    from lib.dj.db import LibraryDB
    from lib.dj.brain import load_library
    from lib.dj.themes import Theme
    from lib.dj import setlist as SL

    print("DJ setlist test\n" + "=" * 40 + "\n")
    tmp = tempfile.mkdtemp(prefix="gl_dj_sl_")
    try:
        build_temp_library(tmp)
        db = LibraryDB(tmp)
        lib = load_library(db)
        lib.sort(key=lambda t: t.path)
        theme = Theme("t", bpm_range=(80.0, 135.0), min_play_s=40.0,
                      max_play_s=65.0,
                      mood_weights={"groove": 1.0, "peak": 0.5})
        ids = [t.id for t in lib]        # song_0..song_3 order

        # -- CRUD round-trip ------------------------------------------------
        sid = SL.create_setlist(db, "friday", theme="groove", notes="test")
        SL.save_entries(db, sid, [
            {"track_id": ids[2], "pin_type": "anchor",
             "target_offset_min": None},
            {"track_id": ids[0], "pin_type": "suggestion"},
            {"track_id": ids[1], "pin_type": "anchor",
             "target_offset_min": 6.0},
        ])
        sl = SL.get_setlist(db, name="friday")
        check("crud round-trip", sl is not None and len(sl["entries"]) == 3
              and sl["entries"][0]["track_id"] == ids[2]
              and sl["entries"][2]["pin_type"] == "anchor"
              and sl["entries"][2]["target_offset_min"] == 6.0,
              f"entries={[(e['track_id'], e['pin_type']) for e in sl['entries']]}")
        names = [s["name"] for s in SL.list_setlists(db)]
        check("setlist listed", "friday" in names, f"names={names}")

        # -- Compiler ---------------------------------------------------------
        plan = SL.compile_plan(lib, sl["entries"], theme)
        slots = plan["slots"]
        check("compiler resolves all slots", len(slots) == 3
              and [s["track"].id for s in slots] ==
              [ids[2], ids[0], ids[1]],
              f"order={[s['track'].title for s in slots]}")
        check("every seam planned", all(s["transition"] for s in slots[:-1])
              and slots[-1]["transition"] is None,
              f"styles={[s['transition']['style'] if s['transition'] else None for s in slots]}")
        check("timeline monotonic", all(
            slots[i + 1]["start_offset_s"] > slots[i]["start_offset_s"]
            for i in range(len(slots) - 1))
            and abs(plan["total_s"] - sum(s["play_s"] for s in slots)) < 1.0,
            f"offsets={[round(s['start_offset_s'], 1) for s in slots]} "
            f"total={plan['total_s']:.0f}s")

        # Tempo clash: 122 -> 85 bpm has no legal stretch; the compiler must
        # warn and fall back rather than fail.
        clash = SL.compile_plan(lib, [
            {"track_id": ids[0], "pin_type": "anchor",
             "target_offset_min": None},
            {"track_id": ids[3], "pin_type": "anchor",
             "target_offset_min": None},
        ], theme)
        check("tempo clash warned", any("tempo clash" in w or "long_fade" in w
                                        for w in clash["warnings"]),
              f"warnings={clash['warnings']}")

        # -- Autofill ------------------------------------------------------------
        anchors = [
            {"track_id": ids[0], "pin_type": "anchor",
             "target_offset_min": None, "position": 0},
            {"track_id": ids[1], "pin_type": "anchor",
             "target_offset_min": 4.0, "position": 1},
        ]
        filled = SL.autofill(lib, anchors, theme)
        a_order = [e["track_id"] for e in filled
                   if e["pin_type"] == "anchor"]
        n_sugg = sum(1 for e in filled if e["pin_type"] == "suggestion")
        check("autofill keeps anchors in order",
              a_order == [ids[0], ids[1]], f"anchors={a_order}")
        check("autofill inserts suggestions", n_sugg >= 1,
              f"{n_sugg} suggestions inserted "
              f"({[e['track_id'] for e in filled]})")

        # -- Plan-following through the live system ---------------------------------
        from lib.dj.system import DJSystem
        from lib.audio_engine import AudioEngine
        engine = AudioEngine()
        dj = DJSystem(tmp, engine=engine, theme="groove", seed=7,
                      threaded=False, log_dir=tmp)
        assert dj.start()
        dj.brain.theme = theme
        dj.load_setlist("friday")
        gen = engine._mixer()
        next(gen)
        played, prev = [], None
        for i in range(int(260.0 * RATE) // 4410):
            gen.send(4410)
            dj.step()
            cur = (dj.status()["current"] or {}).get("id")
            if cur != prev and cur is not None:
                played.append(cur)
                prev = cur
        dj.stop()
        check("setlist order honored", played[:3] == [ids[2], ids[0], ids[1]],
              f"played={played} want={[ids[2], ids[0], ids[1]]}")

        # -- Planner UI headless smoke ---------------------------------------------
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication
            sys.path.insert(0, os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), "tools"))
            from tools.dj_planner import Planner
            app = QApplication.instance() or QApplication([])
            w = Planner(tmp)
            w.entries = [{"track_id": ids[0], "pin_type": "suggestion",
                          "target_offset_min": None, "style_override": None},
                         {"track_id": ids[1], "pin_type": "anchor",
                          "target_offset_min": 5.0, "style_override": None}]
            w._rebuild_set_list()
            from lib.dj.themes import get_theme
            w._compiled(SL.compile_plan(w.library, w.entries,
                                        get_theme("groove")))
            n_rows = w.plan_list.count()
            w.close()
            check("planner builds headless", n_rows >= 3,
                  f"{len(w.library)} tracks loaded, {n_rows} plan rows")
        except Exception as e:
            check("planner builds headless", False,
                  f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
