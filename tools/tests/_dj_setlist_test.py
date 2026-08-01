"""Phase E gate: setlists - CRUD, compiler, autofill, plan-following, and a
headless smoke of the PyQt6 planner.

Uses the same synthetic-track + real-scanner path as the Phase C e2e so
the compiler runs against genuine analysis rows, then follows a saved
setlist through DJSystem on the hand-pumped engine and asserts the play
order honors the plan.

Usage: python tools/tests/_dj_setlist_test.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

RATE = 44100
failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def build_temp_library(tmp):
    from scipy.io import wavfile
    from tools.tests._dj_brain_test import synth_structured
    from lib.dj.scan import scan_library
    specs = [(122.0, 5, 54.0, 220.0), (126.0, 6, 58.0, 262.0),
             (124.0, 7, 50.0, 294.0), (85.0, 8, 46.0, 330.0),
             # A tempo CHAIN for the beam/bridge gates: 120 <-> 112 <-> 104
             # are pairwise reachable (<8% stretch) but 120 <-> 104 is not.
             (120.0, 9, 52.0, 246.0), (112.0, 10, 48.0, 208.0),
             (104.0, 11, 56.0, 233.0)]
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

        # -- v2 analysis payloads: axes / auto tags / auto cues / user cues ----
        t0 = lib[0]
        check("axes computed", isinstance(t0.axes, dict)
              and set(t0.axes) >= {"vocal", "speed", "hardness", "energy"},
              f"axes={t0.axes}")
        check("auto tags derived", isinstance(t0.auto_tags, list),
              f"auto_tags={t0.auto_tags}")
        check("no confetti sections",
              all((s["end_beat"] - s["start_beat"]) >= 16
                  for t in lib for s in t.sections
                  if s["end_beat"] and s["start_beat"] is not None),
              "every section >= 16 beats (v2 anti-chop)")
        db.add_tag(t0.id, "Opener")
        db.add_tag(t0.id, "opener")            # dedup via lowercase
        check("user tags stored", db.tags_for(t0.id) == ["opener"],
              f"tags={db.tags_for(t0.id)}")
        # A user OUT cue must override the analyzer's mix-outs in planning.
        cue_t = t0.nearest_downbeat(70.0)
        db.add_cue(t0.id, "out", cue_t, label="my exit")
        from lib.dj.brain import load_library as _ll
        lib2 = _ll(db)
        t0b = next(t for t in lib2 if t.id == t0.id)
        check("user out-cue overrides mix points",
              len(t0b.mix_outs) == 1
              and abs(t0b.mix_outs[0]["time_s"] - cue_t) < 0.01
              and t0b.mix_outs[0]["score"] == 1.0,
              f"mix_outs={t0b.mix_outs}")
        db.remove_cue(db.cues_for(t0.id, kind="out")[0]["id"])

        # -- Plan mode: suggest_set + optimize_order ---------------------------
        from lib.dj.themes import Theme as _T
        sugg = SL.suggest_set(lib, theme, minutes=6.0, seed=3)
        check("suggest_set builds a set", len(sugg) >= 3
              and len({e["track_id"] for e in sugg}) >= 3,
              f"{len(sugg)} entries: {[e['track_id'] for e in sugg]}")
        # A PLANNED set must never repeat a track, even asking for a long set
        # from a small pool - it just ends short instead of looping.
        longask = SL.suggest_set(lib, theme, minutes=120.0, seed=1)
        lids = [e["track_id"] for e in longask]
        check("suggest_set never repeats a track (unique)",
              len(lids) == len(set(lids)),
              f"{len(lids)} entries, {len(set(lids))} unique")
        mixed = [
            {"track_id": ids[3], "pin_type": "suggestion"},
            {"track_id": ids[0], "pin_type": "anchor",
             "target_offset_min": None},
            {"track_id": ids[1], "pin_type": "suggestion"},
        ]
        opt = SL.optimize_order(lib, mixed, theme, seed=1)
        check("optimize keeps anchors placed",
              len(opt) == 3 and opt[1]["track_id"] == ids[0]
              and opt[1]["pin_type"] == "anchor",
              f"order={[e['track_id'] for e in opt]}")

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
        # v3 TIMING SOLVER: the timed anchor must actually LAND near its
        # target in the compiled timeline (v2 just filled and hoped).
        fplan = SL.compile_plan(lib, filled, theme)
        a_slot = next(s for s in fplan["slots"]
                      if s["entry"].get("pin_type") == "anchor"
                      and s["entry"].get("target_offset_min"))
        err_s = abs(a_slot["start_offset_s"] - 4.0 * 60.0)
        check("autofill lands the timed anchor", err_s <= 90.0,
              f"anchor at {a_slot['start_offset_s']:.0f}s vs target 240s "
              f"(err {err_s:.0f}s)")
        check("autofill stamps play hints",
              all(e.get("target_play_s") for e in filled
                  if e["pin_type"] == "suggestion"),
              f"hints={[e.get('target_play_s') for e in filled]}")

        # -- Shape ordering: the set follows a tempo curve ---------------------
        by_id = {t.id: t for t in lib}
        shape_pool = [{"track_id": t.id, "pin_type": "suggestion"}
                      for t in sorted(lib, key=lambda x: x.bpm)]
        ris = SL.order_by_shape(lib, shape_pool, theme, metric="tempo",
                                shape="rise")
        rb = [by_id[e["track_id"]].bpm for e in ris]
        check("order_by_shape rise climbs in tempo",
              len(rb) == len(shape_pool) and rb[0] <= rb[-1]
              and rb == sorted(rb) or rb[0] < rb[-1],
              f"first={rb[0]:.0f} last={rb[-1]:.0f}")
        pk = SL.order_by_shape(lib, shape_pool, theme, metric="tempo",
                               shape="peak")
        pb = [by_id[e["track_id"]].bpm for e in pk]
        mid = len(pb) // 2
        check("order_by_shape peak crests in the middle",
              pb[mid] >= pb[0] and pb[mid] >= pb[-1],
              f"ends {pb[0]:.0f}/{pb[-1]:.0f}, mid {pb[mid]:.0f}")

        # -- Beam ordering: no dead seam when a live order exists ---------------
        by_id = {t.id: t for t in lib}
        b120 = next(t for t in lib if abs(t.bpm - 120.0) < 1).id
        b112 = next(t for t in lib if abs(t.bpm - 112.0) < 1).id
        b104 = next(t for t in lib if abs(t.bpm - 104.0) < 1).id
        bad_order = [{"track_id": i, "pin_type": "suggestion"}
                     for i in (b120, b104, b112)]     # 120->104 is dead
        from lib.dj.brain import Brain as _Brain
        _b = _Brain(lib, theme)

        def dead_seams(order):
            n = 0
            for i in range(len(order) - 1):
                a, c = by_id[order[i]["track_id"]], \
                    by_id[order[i + 1]["track_id"]]
                r, _ = _b.rate_for(a.bpm, c)
                n += r is None
            return n
        opt2 = SL.optimize_order(lib, bad_order, theme, seed=1)
        check("beam ordering avoids the dead seam",
              len(opt2) == 3 and dead_seams(opt2) == 0
              and dead_seams(bad_order) == 1,
              f"in={[by_id[e['track_id']].bpm for e in bad_order]} -> "
              f"out={[by_id[e['track_id']].bpm for e in opt2]}")

        # -- Bridge finder ---------------------------------------------------------
        chain, score = SL.bridge(lib, by_id[b120], by_id[b104], theme)
        check("bridge connects the tempo islands",
              len(chain) >= 1 and any(t.id == b112 for t in chain),
              f"chain={[round(t.bpm) for t in chain]} score={score:.3f}")

        # -- new stem styles script their stem automation --------------------------
        from lib.dj.brain import Brain as _Brain
        br3 = _Brain(lib, theme, seed=1)
        a3, b3 = by_id[ids[0]], by_id[ids[1]]
        in3 = b3.mix_ins[0]["time_s"] if b3.mix_ins else 0.0
        base3 = {"rate": 1.0, "in_s": in3, "pair_score": 0.2,
                 "cand_id": b3.id, "pitch_st": 0, "a_rate": 1.0, "diag": {}}
        for st3, nmin in (("stem_bass_swap", 3), ("drum_bridge", 3),
                          ("acapella_in", 2), ("melody_carry", 1)):
            plan3 = dict(base3, style=st3, beats=16, tail_beats=16,
                         out_s=a3.nearest_phrase(a3.duration_s * 0.6))
            ev3, sw3, b03 = br3.preview_events(plan3, a3, b3)
            n3 = sum(1 for e in ev3 if e["cmd"] == "stem_gains")
            check(f"{st3} scripts stem automation",
                  n3 >= nmin and sw3 > b03,
                  f"{n3} stem_gains events, swap>{'blend' if sw3 > b03 else 'BAD'}")
        plan3 = dict(base3, style="long_blend", beats=32,
                     out_s=a3.nearest_phrase(a3.duration_s * 0.6),
                     duck_vocal_a=True)
        ev3, _sw, _b0 = br3.preview_events(plan3, a3, b3)
        duck_ev = [e for e in ev3 if e["cmd"] == "stem_gains"
                   and e["deck"] == "a"
                   and e["gains"].get("vocals") == 0.0]
        check("vocal duck scripts A's vocal stem out", bool(duck_ev),
              f"{len(duck_ev)} duck events")
        # Gates: every stem style refuses politely on a stem-less library.
        pplan3 = SL.compile_plan(lib, [
            {"track_id": ids[2], "pin_type": "anchor"},
            {"track_id": ids[0], "pin_type": "suggestion",
             "style_override": "drum_bridge"}], theme)
        check("drum_bridge gated without stems",
              any("no_stems" in w for w in pplan3["warnings"]),
              f"warnings={pplan3['warnings']}")

        # -- target_play_s survives the DB round-trip ------------------------------
        sid2 = SL.create_setlist(db, "timed", theme="groove")
        SL.save_entries(db, sid2, [
            {"track_id": ids[0], "pin_type": "suggestion",
             "target_play_s": 123.4}])
        got = SL.get_setlist(db, name="timed")["entries"][0]
        check("play hint round-trips", got.get("target_play_s") == 123.4,
              f"target_play_s={got.get('target_play_s')}")

        # -- Style pins go THROUGH plan_transition (not a label overwrite) -----
        # A pinned style must either be the planned style or be refused with
        # a visible warning (safety gates outrank the pin).
        pplan = SL.compile_plan(lib, [
            {"track_id": ids[2], "pin_type": "anchor"},
            {"track_id": ids[0], "pin_type": "suggestion",
             "style_override": "long_blend"},
        ], theme)
        tr = pplan["slots"][0]["transition"]
        pin_diag = (tr.get("diag") or {}).get("style_pin") or {}
        check("compile honors style pin",
              (tr["style"] == "long_blend" and pin_diag.get("honored"))
              or (pin_diag.get("honored") is False
                  and any("style pin" in w for w in pplan["warnings"])),
              f"style={tr['style']} pin={pin_diag}")
        # A pin the gates zero (no stems on disk here) must be refused, warn,
        # and fall back to a normal roll - never play the impossible style.
        pplan2 = SL.compile_plan(lib, [
            {"track_id": ids[2], "pin_type": "anchor"},
            {"track_id": ids[0], "pin_type": "suggestion",
             "style_override": "stem_drum_swap"},
        ], theme)
        tr2 = pplan2["slots"][0]["transition"]
        check("gated style pin refused with warning",
              tr2["style"] != "stem_drum_swap"
              and any("style pin" in w for w in pplan2["warnings"]),
              f"style={tr2['style']} warnings={pplan2['warnings']}")

        # -- Plan-following through the live system ---------------------------------
        # Re-save friday with a pinned style on the 2nd seam + plan meta, so
        # this run also proves the pin and the set's own arc clock reach the
        # live side.
        SL.save_entries(db, sid, [
            {"track_id": ids[2], "pin_type": "anchor",
             "target_offset_min": None},
            {"track_id": ids[0], "pin_type": "suggestion",
             "style_override": "long_blend"},
            {"track_id": ids[1], "pin_type": "anchor",
             "target_offset_min": 6.0},
        ])
        db.set_setlist_meta(sid, total_s=1234.0)
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
        saw_total = None
        for i in range(int(360.0 * RATE) // 4410):
            gen.send(4410)
            dj.step()
            if saw_total is None and dj._setlist_total_s is not None:
                saw_total = dj._setlist_total_s
            cur = (dj.status()["current"] or {}).get("id")
            if cur != prev and cur is not None:
                played.append(cur)
                prev = cur
        dj.stop()
        check("setlist order honored", played[:3] == [ids[2], ids[0], ids[1]],
              f"played={played} want={[ids[2], ids[0], ids[1]]}")
        check("setlist arc clock armed", saw_total == 1234.0,
              f"saw_total={saw_total}")
        # The live order-mode must have run the pin through plan_transition
        # and logged the outcome (honored or gate-refused - both are fine,
        # silence is the bug).
        import glob as _glob
        pins = []
        for lf in _glob.glob(os.path.join(tmp, "dj_*.jsonl")):
            with open(lf, encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if ev.get("event") == "style_pin":
                        pins.append(ev)
        check("live style pin logged",
              any(p.get("want") == "long_blend" for p in pins),
              f"pins={pins}")

        # -- Planner UI headless smoke (tabbed v2) ------------------------------
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        try:
            from PyQt6.QtWidgets import QApplication
            from tools.dj_planner import Planner
            from lib.dj.themes import get_theme
            from lib.dj.brain import Brain
            app = QApplication.instance() or QApplication([])
            w = Planner(tmp)
            st = w.set_tab
            st.entries = [{"track_id": ids[0], "pin_type": "suggestion",
                           "target_offset_min": None, "style_override": None},
                          {"track_id": ids[1], "pin_type": "anchor",
                           "target_offset_min": 5.0, "style_override": None}]
            st._rebuild()
            compiled = SL.compile_plan(w.library, st.entries,
                                       get_theme("groove"))
            st._compiled(compiled)
            n_rows = st.plan_list.count()
            # Mix timeline builds seams + envelopes from the same plan.
            w.mix_tab.timeline.set_plan(compiled,
                                        Brain(w.library, get_theme("groove")))
            n_seams = len(w.mix_tab.timeline.seams)
            env_ok = all(sm["env_a"] and sm["env_b"]
                         for sm in w.mix_tab.timeline.seams)
            # Analysis waveform accepts a decoded track headlessly.
            from lib.dj.features import decode_file_stereo
            t = w.library[0]
            mono = decode_file_stereo(db.abs(t.path)).mean(axis=1)
            w.analysis_tab.wave.set_track(t, mono, db.cues_for(t.id))
            wave_ok = len(w.analysis_tab.wave._pyramid) >= 2
            # v3 cockpit: arc strip carries the compiled shape, the report
            # card summarizes it, and the repair affordances compute.
            strip_ok = (len(st.arc_strip.slots) == len(compiled["slots"])
                        and not st.arc_strip.grab().isNull())
            report_txt = st.status.text()      # capture BEFORE bridge runs
            report_ok = "beat-matched" in report_txt or "tracks" in report_txt
            seam_info_ok = all(
                s["seam_info"] is not None for s in compiled["slots"][:-1])
            alts = st._slot_alternatives(0)
            # Transition options panel: every style listed with odds or a
            # gate reason for the first seam; clicking a style row pins it.
            from PyQt6.QtCore import Qt as _Qt
            _role = _Qt.ItemDataRole.UserRole + 1
            st._update_style_options(0)
            opts_ok = st.style_opts.count() >= 10
            row = next((st.style_opts.item(i)
                        for i in range(st.style_opts.count())
                        if st.style_opts.item(i).data(_role) == "long_fade"),
                       None)
            if row is not None:
                st._style_option_clicked(row)
                opts_pin_ok = (compiled["slots"][1]["entry"]
                               .get("style_override") == "long_fade")
                st._undo_edit()
                if st._worker is not None:      # let the recompile land
                    st._worker.wait(20000)      # before the window closes
            else:
                opts_pin_ok = False
            st.set_list.setCurrentRow(0)
            n_before = len(st.entries)
            st._insert_bridge()      # 122 -> 126 is mixable; may no-op
            bridge_ran = len(st.entries) >= n_before
            w.close()
            check("planner v2 builds headless",
                  n_rows >= 3 and w.tabs.count() >= 4,
                  f"{len(w.library)} tracks, {n_rows} plan rows, "
                  f"{w.tabs.count()} tabs")
            check("v3 cockpit renders", strip_ok and report_ok
                  and seam_info_ok,
                  "strip slots=%d, report='%s'" % (
                      len(st.arc_strip.slots),
                      report_txt[:60].encode("ascii", "replace").decode()))
            check("v3 repair affordances compute",
                  isinstance(alts, list) and bridge_ran,
                  f"{len(alts)} alternatives for slot 0")
            check("transition options panel lists + pins styles",
                  opts_ok and opts_pin_ok,
                  f"{st.style_opts.count()} rows, pin round-trip "
                  f"{opts_pin_ok}")
            check("mix timeline seams + envelopes",
                  n_seams == 1 and env_ok, f"{n_seams} seams, env={env_ok}")
            check("waveform pyramid built", wave_ok,
                  f"levels={len(w.analysis_tab.wave._pyramid)}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            check("planner v2 builds headless", False,
                  f"{type(e).__name__}: {e}")

        # -- Set Copilot backend (scripted fake client, no network) -----------
        try:
            from types import SimpleNamespace as NS
            from tools.dj.planner.copilot import SetCopilot

            def tu(name, args, tid="t1"):
                return NS(type="tool_use", name=name, input=args, id=tid)

            def resp(stop, blocks):
                return NS(stop_reason=stop, content=blocks, usage=None)

            script = [
                resp("tool_use", [NS(type="text", text="searching"),
                                  tu("search_library",
                                     {"bpm_min": 100, "bpm_max": 130})]),
                resp("tool_use", [tu("add_tracks",
                                     {"track_ids": [ids[0], ids[1]]})]),
                resp("tool_use", [tu("add_tracks",
                                     {"track_ids": [999999]})]),  # bad id
                resp("tool_use", [tu("pin_anchor",
                                     {"position": 1, "offset_min": 6})]),
                resp("tool_use", [tu("get_set", {})]),
                resp("end_turn", [NS(type="text",
                                     text="Two tracks in, slot 1 anchored "
                                          "at 6 min.")]),
            ]
            fake = NS(messages=NS(create=lambda **kw: script.pop(0)))
            cp = SetCopilot(lib, theme_name="groove", client=fake)
            events = []
            reply = cp.run_turn("build me a tiny set",
                                on_event=lambda k, p: events.append((k, p)))
            check("copilot turn completes", "anchored" in reply
                  and len(events) == 5,
                  f"reply='{reply[:40]}' events={len(events)}")
            check("copilot edits the set",
                  [e["track_id"] for e in cp.entries] == [ids[0], ids[1]]
                  and cp.entries[1]["pin_type"] == "anchor"
                  and cp.entries[1]["target_offset_min"] == 6.0,
                  f"entries={[(e['track_id'], e['pin_type']) for e in cp.entries]}")
            bad = [r for m in cp.messages if isinstance(m.get("content"),
                                                        list)
                   for r in m["content"]
                   if isinstance(r, dict) and r.get("is_error")]
            check("copilot guards bad ids", len(bad) == 1
                  and "999999" in bad[0]["content"],
                  f"{len(bad)} error results")
            state = cp.run_tool("get_set", {})
            check("copilot reads compiled state",
                  len(state["slots"]) == 2 and "beat_matched" in state
                  and state["slots"][1]["anchor"],
                  f"slots={len(state['slots'])} "
                  f"bm={state.get('beat_matched')}")

            # CLI transport: text command-protocol loop, no subprocess.
            cli = SetCopilot(lib, theme_name="groove")
            cli._mode = "cli"
            cli._claude_exe = "fake"
            cli_script = [
                '```json\n{"tool": "search_library", '
                '"input": {"bpm_min": 118}}\n```',
                "Found matching tracks in the library. The set looks good.",
            ]
            cli._call_claude = lambda sysp, prompt: cli_script.pop(0)
            cli_ev = []
            cli_reply = cli.run_turn(
                "find some tracks",
                on_event=lambda k, p: cli_ev.append(p.get("name")))
            check("copilot CLI transport runs tools + returns text",
                  cli_ev == ["search_library"]
                  and "looks good" in cli_reply
                  and "```" not in cli_reply,
                  f"tools={cli_ev} reply='{cli_reply[:40]}'")

            # build_set confines the brain to a filtered pool (the fix for
            # "it just returns random songs").
            cp2 = SetCopilot(lib, theme_name="groove", client=object())
            bpms = sorted(t.bpm for t in lib)
            lo, hi = bpms[0], bpms[len(bpms) // 2]
            res = cp2.run_tool("build_set",
                               {"minutes": 10, "bpm_min": lo, "bpm_max": hi})
            got = [next(x for x in lib if x.id == e["track_id"]).bpm
                   for e in cp2.entries]
            check("build_set confines the set to the pool",
                  res["ok"] and got and all(lo <= b <= hi for b in got),
                  f"pool={res.get('pool_size')} {len(got)} tracks, all in "
                  f"[{lo:.0f},{hi:.0f}]? {all(lo <= b <= hi for b in got)}")
            narrow = cp2.run_tool("build_set",
                                  {"minutes": 10, "bpm_min": 999,
                                   "bpm_max": 1000})
            check("build_set degrades gracefully on empty pool",
                  not narrow["ok"] and narrow["pool_size"] == 0,
                  f"ok={narrow['ok']} pool={narrow['pool_size']}")

            # pin_style writes the seam-INTO convention (entry i+1) and
            # validates the vocabulary; night_history reads the injected
            # log evidence; save/push queue deferred UI actions.
            t0, t1 = lib[0], lib[1]
            nv = {(t0.title, t1.title): [
                {"date": "20260725", "style": "long_blend",
                 "verdict": "flam", "max_err_beats": 0.21, "hole_s": 0.0,
                 "urgent": False, "rough": True}]}
            lp = {t0.id: __import__("time").time() - 3 * 86400}
            cp3 = SetCopilot(lib, theme_name="groove", client=object(),
                             night_verdicts=nv, last_played=lp)
            cp3.sync([{"track_id": t0.id, "pin_type": "suggestion"},
                      {"track_id": t1.id, "pin_type": "suggestion"}],
                     "groove")
            r = cp3.run_tool("pin_style", {"position": 0,
                                           "style": "long_fade"})
            pin_ok = (r["ok"] and cp3.entries[1]["style_override"]
                      == "long_fade")
            cp3.run_tool("pin_style", {"position": 0})       # clear
            pin_ok = pin_ok and cp3.entries[1]["style_override"] is None
            try:
                cp3.run_tool("pin_style", {"position": 0, "style": "nope"})
                pin_ok = False
            except ValueError:
                pass
            check("copilot pin_style round-trips + validates", pin_ok,
                  f"override={cp3.entries[1]['style_override']}")
            nh = cp3.run_tool("night_history", {})
            check("copilot night_history reads log evidence",
                  nh["live_seams"] and nh["live_seams"][0]["rough"]
                  and any(t["played_days_ago"] and 2.5 < t["played_days_ago"]
                          < 3.5 for t in nh["tracks"]),
                  f"seams={len(nh['live_seams'])} tracks={nh['tracks']}")
            cp3.run_tool("save_set", {"name": "cp-test"})
            cp3.run_tool("push_to_live", {"mode": "pool"})
            check("copilot save/push queue deferred UI actions",
                  cp3.pending_ui == [("save", {"name": "cp-test"}),
                                     ("push", {"mode": "pool"})],
                  f"pending={cp3.pending_ui}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            check("copilot turn completes", False, f"{type(e).__name__}: {e}")

        # -- PlanPreview: the preview must EXECUTE the compiled plan ----------
        try:
            import threading as _th
            import time as _time
            from tools.dj.planner.player import PlanPreview
            from lib.dj.themes import get_theme as _gt
            from lib.dj.brain import Brain as _B
            compiled = SL.compile_plan(lib, sl["entries"], theme)
            pv = PlanPreview(db)
            pv.set_plan(compiled, _B(lib, _gt("groove")))
            actions, anchors = pv._build_script(0, 30.0)
            loads = [a for a in actions if a[1] == "load"]
            posts = [a for a in actions if a[1] == "post"]
            check("preview script covers the plan",
                  len(loads) == 3 and len(posts) == 2
                  and len(anchors) == 2
                  and all(a[0] >= 0 for a in actions),
                  f"{len(loads)} loads, {len(posts)} event batches, "
                  f"{len(anchors)} seam anchors")
            # Device-free producer: render through the scripted submix and
            # confirm audio + a seam handover actually happen.
            pv.compiled = compiled
            pv._stop.clear()
            pv.playing = True
            pv._anchors = [(0, 0.0)] + anchors
            th = _th.Thread(target=pv._produce, args=(actions,), daemon=True)
            th.start()
            peak, seam_seen = 0.0, False
            t0 = _time.time()
            while _time.time() - t0 < 25:
                blk = pv._fetch(2205)
                peak = max(peak, float(np.abs(blk).max()))
                ph = pv.playhead()
                if ph is not None and anchors \
                        and ph > compiled["slots"][1]["start_offset_s"]:
                    seam_seen = True
                    break               # crossed the first drawn seam
                _time.sleep(0.001)      # drain far faster than realtime
            pv._stop.set()
            th.join(timeout=3)
            check("preview plays the drawn plan", peak > 0.1 and seam_seen,
                  f"peak={peak:.2f}, crossed first drawn seam={seam_seen}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            check("preview plays the drawn plan", False,
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
