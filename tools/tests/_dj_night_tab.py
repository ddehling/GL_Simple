"""THE NIGHT TAB, gated headless (tools/dj/planner/director.py + nighttab.py, docs/DJ_SETS.md section 4; 2026-10-29).

Builds the real Director tab offscreen on the real library, with the Director the tab prepares before START (no engine,
no audio device), and drives it the way a hand would:

  PICKER      the presets, your saved themes and the brain's themes are listed; picking a preset plans it
  LENGTH      180 plans the night again to about 180 minutes
  BUILD       a new seed, and another order
  CRATE       a narrower tempo range: the crate count changes and every planned record sits inside the range
  EDITS       a lean, a chapter's energy and a keys mode each plan again; veto plans today's night exactly for the seed
              (the same request with no keys mode at all, as before the keys modes existed)
  SAVE        saved, the tab closed and opened again: the theme is in the picker and plans the same night; delete removes it
  LIKE THESE  3 records give a crate that holds all 3, and a plan; saved and picked again, it is inferred again from its
              examples and plans the same night
  QT THREAD   the event loop never stalls more than 50 ms while a re-plan runs (a 5 ms heartbeat)
  FIXES 10-29 (the review's defects and the layout list, each checked):
              SAVE never raises out of the slot (a locked DB is words, the button unlights), the default name never
              replaces an older theme, a preset added later with a saved theme's name leaves that theme deletable;
              a slow "like these" answer never overwrites a newer pick; a hidden ARC never rewrites the STORY and the
              LENGTH knob goes through the LENGTH box; LAYERS round trip lands on NIGHT / DIALS; the strip keeps the
              last plan, dimmed, "planning…", through every re-plan (never the arc); PACE / ENERGY turned while the
              Director loaded reach the plan, the crate count follows PACE; ≈ LIKE THESE unlights when nothing started;
              two crate counts out of order settle on the newer; a failed preparation closes its planner; "like these"
              examples without stems do not make the crate instrumental; lift leaves flat chapters alone; chapter
              warnings need 2 records and a third of the chapter (red: over half), say "2/5 off", name the records and
              outline them; the strip's hover tells a chapter from a record by height; CRATE has its source; ENERGY's
              lean shows beside the STORY; the picker shows its whole name; the lanes are thin in RECORD mode and the
              strip tall; CRATE needs no scroll at 1280x800; the SONGS filter row fits at 1280; a night action's words
              leave the status line
  UNIT GATES  tools/tests/_dj_setplan_unit.py --quick and _dj_nightplan_unit.py still pass (headless, not rendered)

Screenshots (QWidget.grab at 1280x800 and 1600x1000) of the NIGHT tab with Desert Night planned, mid-edit, and the night
strip mid-night (a staged state on the plan's rows: no rendered audio here) go to --shots.

    python tools/tests/_dj_night_tab.py --music D:/Devel/music [--shots DIR] [--suffix _r1] [--no-units]

Exit 0 = ALL OK, 1 = FAIL. The planner's processes (about 400 MB each) are closed before the unit gates run.
"""
import json
import os
import subprocess
import sys
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
os.environ["DJ_DECODE_INPROC"] = "1"           # (the opener decodes in this process: no helper pool)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

GATE_SAVED = "_gate night tab (listed)"
GATE_SAVE = "_gate night tab (saved)"
GATE_LIKE = "_gate night tab (like)"
GATE_SHADOW = "_gate night tab (shadow)"
GATE_NAMES = (GATE_SAVED, GATE_SAVE, GATE_LIKE, GATE_SHADOW, "_gate night tab (saved 2)")
FAILS = []


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def check(ok, what, extra=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"  ({extra})" if extra else ""), flush=True)
    if not ok:
        FAILS.append(what)
    return ok


def pure_checks():
    """The fixes that need no tab: the store's guards, "like these" vocals, lift, the warning words (no planner, no Qt)."""
    import sqlite3
    from types import SimpleNamespace as NS
    from lib.dj import themestore as TS
    from lib.dj import setplanner as SP
    from tools.dj.planner.nighttab import off_marks, warning_words
    print("PURE", flush=True)

    class LockedDB:
        def list_themes(self):
            return []

        def save_theme(self, *_a, **_k):
            raise sqlite3.OperationalError("database is locked")
    try:
        ok, words = TS.save(LockedDB(), "a night", {"chapters": []})
        check(ok is False and words.startswith("not saved") and "locked" in words, "TS.save on a locked DB returns words, never raises", words)
    except Exception as e:  # noqa: BLE001
        check(False, "TS.save on a locked DB returns words, never raises", f"raised {type(e).__name__}: {e}")
    # "like these": examples without stems say nothing about vocals (no "none" crate from three unmeasured singers)
    lib = {}
    for i in range(1, 41):
        lib[i] = NS(id=i, bpm=118.0 + (i % 7), e=0.4 + 0.01 * (i % 20), words=frozenset({"house"}), genres=frozenset({"house"}),
                    sung=0.0 if i <= 3 else 0.3, vox=None if i <= 3 else [0.3], title=f"record {i}")
    out = TS.infer(lib, [1, 2, 3], lambda c: list(lib), 45, 210.0)      # (a crate big enough: no widening hides the rule)
    th = out.get("theme") or {}
    check(th.get("crate", {}).get("vocals") == "any" and th.get("inferred", {}).get("vox_unknown") == 3,
          "like these: examples without stems leave vocals 'any' and are counted", f"{th.get('crate', {}).get('vocals')}, unknown {th.get('inferred', {}).get('vox_unknown')}")
    # lift: a flat chapter is wander there (an unknown key is not scored down); a rising one keeps the unknown factor
    P = SP.Planner.__new__(SP.Planner)
    a, b = NS(t=NS(camelot=None)), NS(t=NS(camelot="8A"))
    tgt = [(0, 0.5, 120.0, {})] * (SP.Planner.TGT_STEPS + 1)
    flat = P._s_key({"keys": "lift", "rise": [False], "tgt": tgt}, a, b, None, 0.5)
    rise = P._s_key({"keys": "lift", "rise": [True], "tgt": tgt}, a, b, None, 0.5)
    check(flat == 1.0 and rise == SP.KEY_OFF["unknown"], "lift: unknown keys are scored down only where a chapter rises", f"flat {flat}, rising {rise}")
    # the warning words and the outlines from a chapter warning that carries its counts
    ws = [{"bad": False, "chapter": 0, "off": 2, "of": 5, "off_ids": [11, 12], "text": "“Dusk”: 2 of its 5 records ..."},
          {"bad": True, "chapter": 1, "off": 4, "of": 6, "off_ids": [13], "text": "x"}, {"bad": True, "text": "The route runs out at 1:10."}]
    ww = warning_words(ws, [{"name": "Dusk"}, {"name": "Sandstorm"}])
    check([w[0] for w in ww] == ["Dusk 2/5 off", "Sandstorm 4/6 off", "runs out"] and [w[2] for w in ww] == ["amber", "red", "red"],
          "chapter warnings read 'Dusk 2/5 off' (amber) / red over half", ", ".join(w[0] for w in ww))
    check(off_marks({"warnings": ws}) == {11: "amber", 12: "amber", 13: "red"}, "the records a warning names are the strip's outlines")
    check(SP.WARN_MIN_OFF == 2 and abs(SP.WARN_MIN_SHARE - 1 / 3) < 1e-9, "the warning thresholds: 2 records and a third of the chapter")


def check_warnings(resp_or_plan, label):
    """Every chapter warning a real plan gave meets the thresholds and names its records."""
    ws = [w for w in (getattr(resp_or_plan, "warnings", None) or []) if w.get("chapter") is not None]
    bad = [w for w in ws if not (w.get("off", 0) >= 2 and 3 * w["off"] >= w.get("of", 0) and bool(w.get("bad")) == (2 * w["off"] > w["of"])
                                 and len(w.get("off_ids") or []) == w["off"])]
    check(not bad, f"{label}: every chapter warning has 2+ records and a third of the chapter off, its records named",
          f"{len(ws)} chapter warnings" + (f"; wrong: {bad[0]}" if bad else ""))


def main():
    from PyQt6.QtCore import QEventLoop, QTimer, Qt
    from PyQt6.QtGui import QColor, QFont, QPalette
    from PyQt6.QtWidgets import QApplication, QTabWidget
    music = arg("--music", "D:/Devel/music")
    shots = arg("--shots", os.path.join(os.environ.get("TEMP", "."), "nighttab_shots"))
    os.makedirs(shots, exist_ok=True)
    app = QApplication([])
    # the planner app's own look (tools/dj_planner.py main): Fusion, the dark palette, the system font
    app.setStyle("Fusion")
    pal = QPalette()
    for role, c in ((QPalette.ColorRole.Window, (30, 30, 34)), (QPalette.ColorRole.WindowText, (220, 220, 220)),
                    (QPalette.ColorRole.Base, (24, 24, 28)), (QPalette.ColorRole.AlternateBase, (36, 36, 40)),
                    (QPalette.ColorRole.Text, (220, 220, 220)), (QPalette.ColorRole.Button, (45, 45, 50)),
                    (QPalette.ColorRole.ButtonText, (220, 220, 220)), (QPalette.ColorRole.Highlight, (50, 110, 160)),
                    (QPalette.ColorRole.PlaceholderText, (120, 120, 130))):
        pal.setColor(role, QColor(*c))
    app.setPalette(pal)
    app.setFont(QFont("Segoe UI", 9))

    from lib.dj import brain as B
    from lib.dj import themestore as TS
    from lib.dj.db import LibraryDB
    from lib.dj.setplanner import preset_names, get_preset
    from lib.dj.themes import PICKER_THEMES
    from tools.dj.planner.director import DirectorTab

    class FakePlanner:
        """What the tab reads of the planner app: the music folder, its DB and library."""
        music_dir = music

        def __init__(self):
            self.db = LibraryDB(music)
            self.library = [t for t in B.load_library(self.db) if not t.excluded]

        def render_stems(self, *_a, **_k):
            return False

    # -- the Qt thread's heartbeat: a 0 ms timer runs whenever the event loop is free, so a gap between two beats is how
    # long the Qt thread was busy (a 5 ms timer fires every 15.6 ms on Windows, and its gaps came in 16 ms steps) --------
    hb = {"last": None, "max": 0.0, "on": False, "where": "", "over20": 0}

    def beat():
        now = time.perf_counter()
        if hb["on"] and hb["last"] is not None:
            gap = now - hb["last"]
            if gap > 0.020:
                hb["over20"] += 1
            if gap > hb["max"]:
                hb["max"], hb["at"] = gap, hb["where"]
        hb["last"] = now
    heart = QTimer()
    heart.setInterval(0)
    heart.timeout.connect(beat)
    heart.start()

    def measure(on, where=""):
        hb["on"], hb["where"], hb["last"] = on, where, None

    def wait(cond, timeout, what):
        loop = QEventLoop()
        t_end = time.time() + timeout
        res = {"ok": False}

        def chk():
            try:
                ok = bool(cond())
            except Exception:
                ok = False
            if ok or time.time() > t_end:
                res["ok"] = ok
                loop.quit()
        t = QTimer()
        t.setInterval(20)
        t.timeout.connect(chk)
        t.start()
        loop.exec()
        t.stop()
        if not res["ok"] and what:
            print(f"  (timed out: {what})", flush=True)
        return res["ok"]

    def idle(sec):
        wait(lambda: False, sec, "")

    def answered(tab, old=None):
        """A plan made after `old` (a new NightPlan object per change), answered and applied, nothing waiting."""
        d = tab._night_dir()
        pl = getattr(d, "plan", None)
        return (pl is not None and pl is not old and pl.pending is None and not pl.want and bool(pl.changes)
                and not tab._night_debounce.isActive() and not pl._inbox)

    def order(tab):
        return [r["id"] for r in tab._night_dir().plan.rows]

    def plan_of(tab):
        return tab._night_dir().plan

    def pick(tab, kind, name):
        for i in range(tab.night_pick.count()):
            if tab.night_pick.itemData(i) == (kind, name):
                tab.night_picked_index = i
                tab._night_picked(i)
                return True
        return False

    def kinds(tab):
        out = {"preset": [], "saved": [], "brain": []}
        for i in range(tab.night_pick.count()):
            d = tab.night_pick.itemData(i)
            if d and d[0] in out:
                out[d[0]].append(d[1])
        return out

    def strip_watch(tab):
        """Samples the strip every 15 ms: how often it drew the arc (never, while a night is wanted) or a dimmed plan."""
        st = {"n": 0, "arc": 0, "planning": 0}

        def s():
            st["n"] += 1
            nt = tab.arc_strip.night
            if nt is None:
                st["arc"] += 1
            elif nt.get("planning"):
                st["planning"] += 1
        t = QTimer()
        t.setInterval(15)
        t.timeout.connect(s)
        t.start()
        st["timer"] = t
        return st

    def fits(widgets):
        """The widgets that are narrower than their own text needs (a clipped face)."""
        return [f"{w.text() if hasattr(w, 'text') else type(w).__name__} {w.width()}<{w.sizeHint().width()}"
                for w in widgets if w.isVisible() and w.width() < w.sizeHint().width()]

    def shot(host, name, sizes=((1280, 800), (1600, 1000))):
        was = hb["on"]
        measure(False)
        t = host.widget(0)
        t._say_timer.stop()
        t._night_unsay()                              # (a night action's words: gone, as they are 8 s later in use)
        check("_gate" not in t.moment_lbl.text(), f"{name}: no gate words left in the status line", t.moment_lbl.text())
        paths = []
        for w, h in sizes:
            host.resize(w, h)
            idle(0.4)
            p = os.path.join(shots, f"{name}{arg('--suffix', '')}_{w}x{h}.png")
            host.grab().save(p)
            # the crate on screen is the crate that plans: a kept chip shown again wears its own state, never an old theme's
            from tools.dj.planner.nighttab import Chip
            stale = [f"{c.word}: looks {c.look!r}, is {c.state!r}" for d in t.night_editor.chips.values() for c in d.values()
                     if isinstance(c, Chip) and c.isVisible() and (c.look or "") != (c.state or "")]
            check(not stale, f"{name} {w}x{h}: every chip shown looks like its state", "; ".join(stale[:4]))
            paths.append(p)
        host.resize(1280, 800)
        idle(0.2)
        measure(was, hb["where"])
        print("  shot " + ", ".join(paths), flush=True)

    def crate_n(tab):
        txt = tab.night_editor.count_lbl.text()
        try:
            return int(txt.split(" records")[0])
        except ValueError:
            return None

    def new_tab(planner):
        host = QTabWidget()
        tab = DirectorTab(planner)
        host.addTab(tab, "Director")
        host.resize(1280, 800)
        host.show()
        t0 = time.time()
        ok = wait(lambda: tab._prep is not None and not tab._prep_busy, 240, "the prepared Director")
        print(f"  the tab's Director prepared in {time.time() - t0:.0f} s" + (f" ({getattr(tab, '_prep_err', '')})" if not ok else ""), flush=True)
        return host, tab

    def close_tab(host, tab):
        measure(False)
        try:
            for d in (tab.director, tab._prep):
                if d is not None:
                    d.close_planner()
        finally:
            tab._night_timer.stop()
            tab._timer.stop()
            tab._prep = None
            host.close()
            host.deleteLater()
            idle(0.3)

    pure_checks()
    db = LibraryDB(music)
    for nm in GATE_NAMES:
        db.delete_theme(nm)
    listed = get_preset("Deep Space")
    listed["night"], listed["length"] = 11, 90.0
    TS.save(db, GATE_SAVED, listed)
    print("loading the library...", flush=True)
    planner = FakePlanner()
    host, tab = new_tab(planner)
    tabs_open = [(host, tab)]
    try:
        # the floor: the same heartbeat with nothing happening (the OS takes the thread away now and then on a busy machine)
        measure(True, "idle")
        idle(5.0)
        print(f"  (idle baseline: the longest Qt-thread gap {hb['max'] * 1000:.1f} ms, {hb['over20']} over 20 ms in 5 s)", flush=True)
        hb["max"], hb["over20"] = 0.0, 0
        measure(False)
        # -- PICKER ---------------------------------------------------------------------------------------------------
        print("PICKER", flush=True)
        k = kinds(tab)
        check(k["preset"] == preset_names(), "the presets are listed, in order", ", ".join(k["preset"]))
        check(GATE_SAVED in k["saved"], "a saved theme is listed", ", ".join(k["saved"]))
        check(k["brain"] == list(PICKER_THEMES), "the brain's themes are listed (no plan, the DJ picks)")
        check(tab.right_tabs.currentWidget() is tab.night_tab and tab.right_tabs.isTabVisible(0)
              and not tab.right_tabs.isTabVisible(1), "RECORD mode: the NIGHT tab first, DIALS hidden")
        # a hidden ARC left over (a strip drag before any theme, a DIALS button, the controller): it must not rewrite the STORY
        tab._night_dir().dials["arc"] = "yours"
        measure(True, "picking Desert Night")
        pick(tab, "preset", "Desert Night")
        idle(0.3)
        check(tab.arc_strip.night is not None and tab.arc_strip.night.get("planning"),
              "right after the pick the strip draws the night's chapters, 'planning…' (never the arc)")
        ok = wait(lambda: answered(tab) and crate_n(tab) is not None, 180, "Desert Night planned")
        pl = plan_of(tab)
        check(ok and pl.name.startswith("Desert Night") and len(pl.rows) >= 10, "picking a preset plans it",
              f"{len(pl.rows) if pl else 0} records, crate {crate_n(tab)}")
        check([c["name"] for c in pl.chapters] == [c["name"] for c in get_preset("Desert Night")["chapters"]], "its chapters are the preset's")
        check(tab._night_dir().dials.get("arc") == "theme" and [[round(x, 3) for x in c["energy"]] for c in pl.chapters]
              == [[round(x, 3) for x in c["energy"]] for c in get_preset("Desert Night")["chapters"]],
              "a hidden ARC ('yours') is set back to 'theme': the chapters plan the STORY's energies",
              str([c["energy"] for c in pl.chapters]))
        check(tab.arc_strip.night is not None and not tab.arc_strip.night.get("planning"), "the strip draws the night")
        check_warnings(pl, "Desert Night")
        # LAYOUT at 1280x800: the lanes thin and the strip tall in RECORD mode, CRATE without a scroll, the SONGS row fits
        host.resize(1280, 800)
        idle(0.4)
        sa = tab.night_editor.stack.widget(0)
        check(tab.canvas.height() <= 110 and tab.arc_strip.height() >= 110, "RECORD mode: the lanes are thin, the night strip tall",
              f"lanes {tab.canvas.height()} px, strip {tab.arc_strip.height()} px")
        check(sa.verticalScrollBar().maximum() == 0, "CRATE needs no scroll at 1280x800", f"scroll range {sa.verticalScrollBar().maximum()} px")
        frow = [b for g in tab.filter_groups.values() for b in g.buttons()] + [tab.f_hook, tab.f_unplayed, tab.sort_box, tab.queue_btn, tab.view_btn,
                                                                              tab.like_btn, tab.build_btn, tab.save_btn]
        bad = fits(frow)
        check(not bad, "the SONGS filter row fits at 1280: no button narrower than its face", "; ".join(bad))
        check(tab.minimumSizeHint().height() <= 800, "the tab's minimum height still fits 800", f"{tab.minimumSizeHint().height()} px")
        # the CRATE chips sized to the room: a taller pane shows more of the library's words, never a scroll, never the theme changed
        from tools.dj.planner.nighttab import Chip
        ed = tab.night_editor

        def chip_counts():
            return {k: len([c for c in d.values() if isinstance(c, Chip)]) for k, d in ed.chips.items()}

        def chip_slack():
            """The empty room under the last chip row of the CRATE pane (px)."""
            last = max((c.geometry().bottom() for c in ed.chips["tags"].values()), default=0)
            return sa.viewport().height() - last
        n12, th12 = chip_counts(), ed.theme()
        check(ed._lim == ed.LIMITS, "at 1280x800 the CRATE chips are the floor (LIMITS)", f"{ed._lim} {n12}")
        host.resize(1600, 1000)
        idle(0.8)
        n16, slack16 = chip_counts(), chip_slack()
        allv = all(len(ed.vocab[k]) <= ed._lim[k] for k in ed.LIMITS)
        row = max(c.height() for d in ed.chips.values() for c in d.values()) + 3
        check(sum(n16.values()) > sum(n12.values()) and all(n16[k] >= n12[k] for k in n16),
              "at 1600x1000 CRATE shows more of the library's words than at 1280", f"{n12} -> {n16}, caps {ed._lim}")
        check(sa.verticalScrollBar().maximum() == 0, "CRATE needs no scroll at 1600x1000 either",
              f"scroll range {sa.verticalScrollBar().maximum()} px")
        check(allv or slack16 < 3 * row, "at 1600x1000 the CRATE pane leaves less than three chip rows empty under the chips",
              f"{slack16} px empty, a row {row} px")
        check(ed.theme() == th12, "sizing the chips to the room never changes the theme")
        host.resize(1280, 800)
        idle(0.5)
        check(ed._lim == ed.LIMITS and chip_counts() == n12 and sa.verticalScrollBar().maximum() == 0,
              "back at 1280x800: the floor again, CRATE unscrolled", f"{ed._lim} {chip_counts()}")
        pool0 = {k: len(p) for k, p in ed._pool.items()}
        shot(host, "1_desert_night_planned")
        check({k: len(p) for k, p in ed._pool.items()} == pool0,
              "a window resized to and fro builds no chip twice (the hidden ones are kept)", f"{pool0}")
        saved = (hb["max"], hb.get("at"))
        hb["max"] = 0.0
        measure(True, "resizing 1280 <-> 1600")
        for _ in range(2):
            host.resize(1600, 1000)
            idle(0.8)
            host.resize(1280, 800)
            idle(0.5)
        check(hb["max"] < 0.050, "resizing the window to and fro: the chip refills never stall the Qt thread over 50 ms",
              f"longest {hb['max'] * 1000:.0f} ms")
        if saved[0] >= hb["max"]:
            hb["max"], hb["at"] = saved
        measure(True, "picking Desert Night")

        # -- LENGTH ----------------------------------------------------------------------------------------------------
        print("LENGTH", flush=True)
        old = pl
        i180 = tab.len_box.findData(180.0)
        tab.len_box.setCurrentIndex(i180)
        measure(True, "LENGTH 180")
        sw = strip_watch(tab)
        tab._night_length(i180)
        ok = wait(lambda: answered(tab, old), 180, "LENGTH 180")
        idle(0.3)
        sw["timer"].stop()
        check(sw["arc"] == 0 and sw["planning"] > 0, "while LENGTH re-plans the strip keeps the last plan, dimmed, 'planning…'",
              f"{sw['n']} samples: {sw['planning']} planning, {sw['arc']} the arc")
        pl = plan_of(tab)
        end_min = max(float(r.get("end_s") or 0.0) for r in pl.rows) / 60.0
        check(ok and abs(pl.length_min - 180.0) < 0.5 and 170.0 <= end_min <= 192.0, "LENGTH 180 plans about 180 minutes",
              f"{len(pl.rows)} records, ends at {end_min:.0f} min")
        old = pl
        i90 = tab.len_box.findData(90.0)
        tab.len_box.setCurrentIndex(i90)
        tab._night_length(i90)
        wait(lambda: answered(tab, old), 180, "LENGTH 90")
        # the controller's LENGTH knob (8) on a planned night goes through the LENGTH box; ARC (7) does nothing
        print("MIDI KNOBS", flush=True)

        class Knobs:
            def __init__(self, ch):
                self.ch = ch

            def update(self):
                ch, self.ch = self.ch, {}
                return ch
        d0 = tab._night_dir()
        old = plan_of(tab)
        tab.midi, tab.director = Knobs({"knob_7": 0.5, "knob_8": 0.6}), d0      # (knob 7 -> 'waves', knob 8 -> '3h')
        try:
            tab._midi_poll()
        finally:
            tab.director, tab.midi = None, None
        ok = wait(lambda: answered(tab, old), 180, "the LENGTH knob")
        pl = plan_of(tab)
        check(ok and abs(tab._nlen - 180.0) < 1e-6 and tab.len_box.currentData() == 180.0 and abs(pl.length_min - 180.0) < 0.5
              and d0.dials.get("arc") == "theme" and d0.dials.get("length") == "90m",
              "LENGTH knob on a planned night: the box, the night and later edits agree (180); the ARC knob is ignored",
              f"box {tab.len_box.currentText()}, night {pl.length_min:.0f} min, arc {d0.dials.get('arc')}")
        old = pl
        tab.len_box.setCurrentIndex(i90)
        tab._night_length(i90)
        wait(lambda: answered(tab, old), 180, "LENGTH 90 again")

        # -- BUILD -----------------------------------------------------------------------------------------------------
        print("BUILD", flush=True)
        old, seed0, ord0 = plan_of(tab), tab._nseed, order(tab)
        measure(True, "BUILD")
        sw = strip_watch(tab)
        tab._night_build()
        lit = tab.build_btn.isChecked()
        ok = wait(lambda: answered(tab, old), 180, "BUILD")
        idle(0.4)
        sw["timer"].stop()
        check(sw["arc"] == 0 and sw["planning"] > 0, "while BUILD re-plans the strip keeps the last plan, dimmed, 'planning…'",
              f"{sw['n']} samples: {sw['planning']} planning, {sw['arc']} the arc")
        pl = plan_of(tab)
        check(ok and tab._nseed != seed0 and pl.seed == tab._nseed, "BUILD changes the seed", f"{seed0} -> {tab._nseed}")
        check(order(tab) != ord0, "BUILD changes the order")
        check(lit and not tab.build_btn.isChecked(), "BUILD is lit while it plans, then not")

        # -- CRATE: a narrower tempo range -----------------------------------------------------------------------------
        print("CRATE", flush=True)
        n0 = crate_n(tab)
        old = plan_of(tab)
        ed = tab.night_editor
        measure(True, "the tempo range")
        ed.bpm_lo.setValue(112.0)
        ed.bpm_hi.setValue(122.0)
        check(tab._night_debounce.isActive() and tab._nsel.get("edited"), "an edit waits for the debounce and marks the theme edited")
        check("(edited)" in tab.night_pick.currentText(), "the picker says (edited)", tab.night_pick.currentText())
        ok = wait(lambda: answered(tab, old) and crate_n(tab) not in (None, n0), 180, "the narrower crate")
        pl = plan_of(tab)
        bpms = [float(r["bpm"]) for r in pl.rows if r.get("bpm")]
        check(ok and crate_n(tab) is not None and crate_n(tab) < (n0 or 0), "a narrower tempo range changes the crate count", f"{n0} -> {crate_n(tab)}")
        check(bpms and all(112.0 <= b <= 122.0 for b in bpms), "every planned record sits inside 112-122 bpm",
              f"{min(bpms):.1f}-{max(bpms):.1f}" if bpms else "none")
        # the crate's SOURCE: a playlist picked in CRATE is the theme's crate source (and back to the library)
        src = ed.src_box
        if src.count() > 1:
            src.setCurrentIndex(1)
            src.activated.emit(1)
            s1 = ed.theme()["crate"].get("source")
            src.setCurrentIndex(0)
            src.activated.emit(0)
            check(s1 == src.itemData(1) and ed.theme()["crate"].get("source") is None, "CRATE › from: a playlist is the crate's source",
                  f"{src.itemText(1)}")
        else:
            check(src.count() == 1 and ed.theme()["crate"].get("source") is None, "CRATE › from: the whole library (no playlists here)")
        wait(lambda: not tab._night_debounce.isActive(), 5, "")
        # two crate counts asked back to back (the second the newer): the count settles on the newer one, whatever order
        # the answers land in (the night timer held, so only these two are asked)
        class Rec(dict):
            """The tab's answer box, recording every crate answer written into it."""
            written = []

            def __setitem__(self, k, v):
                if k == "crate":
                    Rec.written.append(v[0])
                super().__setitem__(k, v)
        tab._night_timer.stop()
        box0 = tab._nbox
        try:
            tab._nbox = Rec(box0)
            d0 = tab._night_dir()
            th1 = json.loads(json.dumps(pl.theme))
            th2 = json.loads(json.dumps(pl.theme))
            th1["crate"]["bpm"], th2["crate"]["bpm"] = [100.0, 110.0], [104.0, 116.0]
            tab._night_count(d0, th1)
            k1 = tab._nbox.get("crate_key")
            tab._night_count(d0, th2)
            k2 = tab._nbox.get("crate_key")
            want2 = d0._planner_proc().request({"op": "crate", "theme": th2, "play_min_s": 210.0})
            ok = wait(lambda: "crate" in tab._nbox and tab._nbox["crate"][0] == k2, 60, "the newer crate count")
            idle(1.0)                                 # (the older answer, had it been kept, lands by now)
            got = tab._nbox.get("crate")
            check(ok and got and got[0] == k2 and got[1].get("crate") == want2.get("crate") and k1 not in Rec.written,
                  "two crate counts in a row: the older answer is dropped, the newer one lands",
                  f"{got[1].get('crate') if got else None} vs {want2.get('crate')}; written {len(Rec.written)}")
        finally:
            box0.update({k: v for k, v in tab._nbox.items() if k not in ("crate", "crate_key")})
            box0.pop("crate", None)
            box0.pop("crate_key", None)
            tab._nbox = box0
            tab._night_timer.start()
        wait(lambda: crate_n(tab) is not None, 60, "the crate counted again")

        # -- EDITS: a lean, a chapter's energy, a keys mode --------------------------------------------------------------
        print("EDITS", flush=True)
        old = plan_of(tab)
        ed.sec_group.button(1).click()
        ed._add_lean("hypnotic", 1.5)
        measure(True, "a lean")
        ed.touch()
        ok = wait(lambda: answered(tab, old), 120, "a lean")
        check(ok and plan_of(tab).theme.get("leans", {}).get("hypnotic") == 1.5, "adding a lean plans again")
        old = plan_of(tab)
        ed.sec_group.button(2).click()
        measure(True, "a chapter's energy")
        ed.ch_rows[2].e1.setValue(0.95)
        idle(0.1)
        shot(host, "2_mid_edit")                      # (the debounce still running: "(edited)", the STORY page)
        ok = wait(lambda: answered(tab, old), 120, "a chapter's energy")
        pl = plan_of(tab)
        check(ok and abs(float(pl.chapters[2]["energy"][-1]) - 0.95) < 1e-6, "editing a chapter's energy plans again",
              f"{pl.chapters[2]['name']} {pl.chapters[2]['energy']}")
        veto_order = order(tab)
        old = plan_of(tab)
        measure(True, "keys lift")
        ed.keys_btns["lift"].click()
        ok = wait(lambda: answered(tab, old), 120, "keys lift")
        pl = plan_of(tab)
        check(ok and pl.theme.get("keys_mode") == "lift", "a keys mode (lift) plans again",
              "the order moved" if order(tab) != veto_order else "the same order")
        old = pl
        ed.keys_btns["veto"].click()
        ok = wait(lambda: answered(tab, old), 120, "keys veto")
        pl = plan_of(tab)
        check(ok and order(tab) == veto_order, "veto plans the same night again for the same seed")
        # today's request: the same night asked with no keys mode at all (the planner before the keys modes)
        measure(False)
        req, *_rest = pl._req()
        req["theme"].pop("keys_mode", None)
        resp = tab._night_dir()._planner_proc().request(req)
        check(resp.get("ok") and [int(s["id"]) for s in resp.get("slots") or []] == order(tab),
              "veto reproduces today's plan exactly (no keys mode in the request)", resp.get("error") or resp.get("keys"))
        # the crate's pair count is the crate's own: a plan with a record from outside the crate fixed (its row built on
        # demand) leaves the count of the cached table as it was
        prc = tab._night_dir()._planner_proc()
        th_now = json.loads(json.dumps(pl.theme))
        c0 = prc.request({"op": "crate", "theme": th_now, "play_min_s": 210.0})
        inside = set(int(i) for i in pl.crate_ids)
        outs = [t for t in planner.library if t.id not in inside and 112 <= float(t.bpm or 0) <= 122 and getattr(t, "has_stems", False)]
        r_out = prc.request({"op": "replan", "theme": th_now, "length_min": 90, "fixed": [{"id": outs[0].id}], "seed": 5,
                             "play_min_s": 210.0}) if outs else {}
        c1 = prc.request({"op": "crate", "theme": th_now, "play_min_s": 210.0})
        check(c0.get("ok") and c1.get("ok") and c0.get("pairs") == c1.get("pairs") and c0.get("crate") == c1.get("crate"),
              "the crate count counts the crate's own pairs only (a fixed record outside it adds none)",
              f"{c0.get('pairs')} -> {c1.get('pairs')} pairs; replan with {outs[0].title[:24] if outs else '-'}: {r_out.get('ok')}")
        # PACE long and ENERGY amp lit without reaching the Director (as when turned while it loaded): the night timer
        # hands them over, the plan is asked again with them, the crate is counted again for the new PACE, and the STORY
        # rows show what the plan really asks of each chapter
        print("PACE / ENERGY", flush=True)
        d0 = tab._night_dir()
        e_old = [list(c["energy"]) for c in plan_of(tab).chapters]

        def settled(pl, leaned):
            """The same NightPlan asked again (on_dial): no answer out, and its chapters leaned (or not)."""
            return (pl.pending is None and not pl.want and not pl._inbox
                    and ([list(c["energy"]) for c in pl.chapters] != e_old) == leaned)

        def light(name, opt):
            from tools.dj.planner.director import DIAL_LABELS
            for b in tab.dial_groups[name].buttons():
                if b.text() == DIAL_LABELS[name][1][opt]:
                    b.setChecked(True)
        light("pace", "long")
        light("energy", "amp")
        ok = wait(lambda: d0.dials.get("pace") == "long" and d0.dials.get("energy") == "amp" and settled(plan_of(tab), True)
                  and '300.0' in (tab._nbox.get("crate_key") or ""), 180, "PACE long, ENERGY amp")
        pl = plan_of(tab)
        idle(0.6)
        effs = [r.eff.text() for r in ed.ch_rows]
        from tools.dj.planner.nighttab import _f2
        want_e = [f"plans {_f2(c['energy'][0])}→{_f2(c['energy'][-1])}" for c in pl.chapters]
        check(ok and all(effs) and all(e.startswith(w) for e, w in zip(effs, want_e)),
              "PACE / ENERGY reach a prepared night; the count follows PACE; STORY shows the leaned courses", " | ".join(effs))
        light("pace", "normal")
        light("energy", "hold")
        wait(lambda: d0.dials.get("pace") == "normal" and d0.dials.get("energy") == "hold" and settled(plan_of(tab), False), 180, "PACE / ENERGY back")
        idle(0.6)
        check(not any(r.eff.text() for r in ed.ch_rows), "ENERGY hold: no leaned course beside the STORY")
        check(hb["max"] <= 0.050, "the Qt thread never stalled over 50 ms during the re-plans",
              f"longest {hb['max'] * 1000:.0f} ms ({hb.get('at', '')})")
        print(f"  (the longest Qt-thread gap so far: {hb['max'] * 1000:.1f} ms during {hb.get('at', '')}; {hb['over20']} gaps over 20 ms)", flush=True)

        # -- SAVE, reopen, delete ---------------------------------------------------------------------------------------
        print("SAVE", flush=True)
        saved_order, saved_seed = order(tab), tab._nseed
        # a locked library while the night plays: SAVE says so, the button unlights, nothing escapes the slot
        import sqlite3
        real_save = planner.db.save_theme

        def locked(*_a, **_k):
            raise sqlite3.OperationalError("database is locked")
        planner.db.save_theme = locked
        try:
            r = tab._night_save(GATE_SAVE)
            raised = None
        except Exception as e:  # noqa: BLE001
            r, raised = None, e
        finally:
            planner.db.save_theme = real_save
        idle(0.7)
        check(raised is None and r is False and not tab.save_btn.isChecked() and "not saved" in tab.moment_lbl.text(),
              "SAVE on a locked library: words, the button unlit, no exception out of the slot",
              f"{tab.moment_lbl.text()[:70]}" + (f"; raised {raised}" if raised else ""))
        check(tab._night_save(GATE_SAVE), "SAVE")
        check(GATE_SAVE in kinds(tab)["saved"] and tab.night_pick.currentData() == ("saved", GATE_SAVE),
              "the saved theme is picked in the picker", tab.night_pick.currentText())
        nm2 = TS.free_name(planner.db, GATE_SAVE)
        check(nm2 == "_gate night tab (saved 2)" and TS.save(planner.db, GATE_SAVE, TS.load(planner.db, GATE_SAVE))[1].startswith("replaced"),
              "SAVE's default never takes an older theme's name (… 2); a save over a name says 'replaced'", nm2)
        # a preset added later with your theme's name: yours can still be saved and deleted
        real_pn = TS.preset_names
        TS.preset_names = lambda: list(real_pn()) + [GATE_SHADOW]
        try:
            planner.db.save_theme(GATE_SHADOW, {"name": GATE_SHADOW, "chapters": []})
            ok1, w1 = TS.save(planner.db, GATE_SHADOW, {"chapters": []})
            ok2 = TS.delete(planner.db, GATE_SHADOW)
            ok3, w3 = TS.save(planner.db, GATE_SHADOW, {"chapters": []})
        finally:
            TS.preset_names = real_pn
            planner.db.delete_theme(GATE_SHADOW)
        check(ok1 and ok2 and not ok3, "a saved theme a preset's name shadows can be saved and deleted; a new one under it is refused",
              f"{w1}; deleted {ok2}; {w3}")
        close_tab(host, tab)
        tabs_open.clear()
        host, tab = new_tab(planner)
        tabs_open.append((host, tab))
        check(GATE_SAVE in kinds(tab)["saved"], "reopened: the saved theme is in the picker")
        measure(True, "the saved theme")
        pick(tab, "saved", GATE_SAVE)
        ok = wait(lambda: answered(tab), 180, "the saved theme planned")
        check(ok and tab._nseed == saved_seed and order(tab) == saved_order, "the saved theme plans the same night",
              f"{len(order(tab)) if ok else 0} records, seed {tab._nseed}")
        check(tab._night_delete(GATE_SAVE) and GATE_SAVE not in kinds(tab)["saved"], "delete removes it from the picker")

        # -- LIKE THESE ------------------------------------------------------------------------------------------------
        print("LIKE THESE", flush=True)
        tab.found.clearSelection()
        tab.like_btn.click()                           # (nothing selected: a hint, and the button is not left lit)
        check(not tab.like_btn.isChecked(), "≈ LIKE THESE with nothing selected is not left lit")
        tab._night_say("deleted the saved theme _gate words", 0.2)
        idle(0.5)
        check(tab.moment_lbl.text() == "", "a night action's words leave the status line on their own")
        old = plan_of(tab)
        rows = [tab.found.item(i) for i in range(tab.found.count())]
        lib = {t.id: t for t in planner.library}
        cands = [it for it in rows if lib.get(it.data(Qt.ItemDataRole.UserRole)) is not None
                 and 112 <= float(lib[it.data(Qt.ItemDataRole.UserRole)].bpm or 0) <= 126]
        three = [cands[0], cands[len(cands) // 2], cands[-1]] if len(cands) >= 3 else rows[:3]
        tab.found.clearSelection()
        for it in three:
            it.setSelected(True)
        ids = [it.data(Qt.ItemDataRole.UserRole) for it in three]
        measure(True, "like these")
        tab._like_selected()
        ok = wait(lambda: answered(tab, old) and tab._nsel.get("src") == "like", 180, "like these")
        pl = plan_of(tab)
        inf = (tab._nsel.get("theme") or {}).get("inferred") or {}
        check(ok and all(int(i) in set(pl.crate_ids) for i in ids), "like these from 3 records: the crate holds all 3",
              f"crate {len(pl.crate_ids)}, need {inf.get('need')}, widened {len(inf.get('widened') or [])}x")
        check(ok and len(pl.rows) >= 6, "like these: a plan", f"{len(pl.rows)} records, {pl.name}")
        check(hb["max"] <= 0.050, "the Qt thread never stalled over 50 ms (reopened tab, like these)",
              f"longest {hb['max'] * 1000:.0f} ms ({hb.get('at', '')})")
        print(f"  (the longest Qt-thread gap: {hb['max'] * 1000:.1f} ms during {hb.get('at', '')}; {hb['over20']} gaps over 20 ms)", flush=True)
        fm = tab.night_pick.fontMetrics()
        check(fm.horizontalAdvance(tab.night_pick.currentText()) + 24 <= tab.night_pick.width()
              and tab.night_pick.currentText().strip() in tab.night_pick.toolTip(),
              "the picker shows its whole name, '(unsaved)' included (and in its tooltip)", tab.night_pick.currentText())
        shot(host, "3_like_these")
        # saved, then picked again: inferred again from its examples (nothing planned until the inference answers)
        like_order, like_seed = order(tab), tab._nseed
        check(tab._night_save(GATE_LIKE), "SAVE a like-these theme")
        old = plan_of(tab)
        pick(tab, "preset", "Deep Space")
        wait(lambda: answered(tab, old), 180, "Deep Space")
        old = plan_of(tab)
        pick(tab, "saved", GATE_LIKE)
        waiting = bool(tab._nsel.get("waiting"))
        ok = wait(lambda: answered(tab, old) and not tab._nsel.get("waiting"), 180, "the saved like-these theme")
        pl = plan_of(tab)
        check(waiting and ok and all(int(i) in set(pl.crate_ids) for i in ids) and order(tab) == like_order and tab._nseed == like_seed,
              "a saved like-these theme is inferred again on load and plans the same night", f"{len(order(tab))} records")
        tab._night_delete(GATE_LIKE)
        # A SLOW ANSWER NEVER WINS: LIKE THESE, then at once a preset - the inference that lands later is dropped
        print("STALE INFERENCE", flush=True)

        class Box(dict):
            seen = []

            def __setitem__(self, k, v):
                if k == "infer":
                    Box.seen.append(v[0])
                super().__setitem__(k, v)
        tab._nbox = Box(tab._nbox)
        old = plan_of(tab)
        tab.found.clearSelection()
        for i in range(tab.found.count()):
            if tab.found.item(i).data(Qt.ItemDataRole.UserRole) in ids:
                tab.found.item(i).setSelected(True)
        tab._like_selected()
        check(tab.like_btn.isChecked(), "≈ LIKE THESE is lit while its inference is out")
        pick(tab, "preset", "Golden Hour")
        ok = wait(lambda: Box.seen and answered(tab, old), 180, "the late inference and Golden Hour")
        idle(1.0)
        pl = plan_of(tab)
        check(ok and tab._nsel.get("src") == "preset" and tab._nsel.get("name") == "Golden Hour" and pl.name.startswith("Golden Hour")
              and not tab.like_btn.isChecked(), "a 'like these' answer landing after a newer pick is dropped (the pick stands)",
              f"{tab._nsel.get('name')} / {pl.name}; answers seen {len(Box.seen)}")
        check_warnings(pl, "Golden Hour")
        tab._nbox = dict(tab._nbox)
        # LAYERS round trip: record -> one -> record lands on DIALS, then NIGHT (never PARTS); the lanes follow the mode
        print("LAYERS", flush=True)

        def layers(opt):
            for b in tab.dial_groups["layers"].buttons():
                if b.text() == opt:
                    b.click()
            idle(0.2)
        layers("one")
        on_dials = tab.right_tabs.currentWidget() is tab.dials_tab
        wide = tab.canvas.height() > 150
        layers("record")
        check(on_dials and tab.right_tabs.currentWidget() is tab.night_tab and wide and tab.canvas.height() <= 110,
              "LAYERS record -> one -> record lands on DIALS, then NIGHT; full lanes in 'one', thin in RECORD",
              f"{tab.right_tabs.tabText(tab.right_tabs.currentIndex())}")
        # a preparation that fails closes the planner process it started
        print("PREPARE FAILS", flush=True)
        import lib.dj.director as LD

        class FailD:
            closed = []

            def __init__(self, *_a, **_k):
                self.dials, self.plan = {}, None

            def set_theme_plan(self, *_a, **_k):
                pass

            def plan_wait(self, *_a):
                pass

            def prepare(self):
                raise RuntimeError("gate: a preparation that fails")

            def close_planner(self):
                FailD.closed.append(1)
        real_d, prep0 = LD.Director, tab._prep
        LD.Director, tab._prep = FailD, None
        try:
            tab._prepare()
            wait(lambda: not tab._prep_busy, 30, "the failing preparation")
        finally:
            LD.Director, tab._prep = real_d, prep0
        check(FailD.closed and "gate" in str(getattr(tab, "_prep_err", "")), "a failed preparation closes its planner process",
              str(getattr(tab, "_prep_err", "")))

        # -- the night strip mid-night (a staged state on the plan's rows: no rendered audio in this gate) ----------------
        print("STRIP", flush=True)
        old = plan_of(tab)
        pick(tab, "preset", "Desert Night")
        wait(lambda: answered(tab, old), 180, "Desert Night again")
        pl = plan_of(tab)
        k = min(7, len(pl.rows) - 3)
        for j, r in enumerate(pl.rows):
            r["state"] = "played" if j < k else ("now" if j == k else ("next" if j == k + 1 else "planned"))
        pl.offset_s = float(pl.rows[k]["start_s"]) + 75.0
        idle(0.6)
        check(tab.arc_strip.night is not None and any(s.get("state") == "played" for s in tab.arc_strip.night["slots"]),
              "the strip shows the played records solid and the rest hollow")
        tip = ctip = None
        try:
            from tools.dj.planner.nighttab import night_tip, strip_frame
            from PyQt6.QtCore import QPointF
            sp = tab.arc_strip.night
            W, H = tab.arc_strip.width(), tab.arc_strip.height()
            top, bot, span, blocks, X, Y = strip_frame(W, H, sp, 10)
            s, a, b = blocks[k + 2]
            xm = 10 + (W - 20) * (a + b) / 2
            tip = night_tip(tab.arc_strip, sp, tab.arc_strip.night_rows, QPointF(xm, Y(s.get("energy") or 0.5)), 10)
            ch = next(c for c in sp["chapters"] if float(c["start_s"]) <= float(s["start_s"]) < float(c["end_s"]))
            ctip = night_tip(tab.arc_strip, sp, tab.arc_strip.night_rows, QPointF(xm, top + 5), 10)
        except Exception as e:  # noqa: BLE001
            tip = f"{type(e).__name__}: {e}"
            ch = {}
        check(bool(tip) and (pl.rows[k + 2]["title"] or "") in tip, "a block's tooltip names the record, its chapter and why",
              (tip or "").replace("\n", " | ")[:160])
        check(bool(ctip) and ctip.startswith(f"{ch.get('name')}:"), "the band's top row (its name, its warning mark) tells the chapter, not a record",
              (ctip or "").replace("\n", " | ")[:120])
        shot(host, "4_night_strip_mid_night")
    finally:
        for h, t in tabs_open:
            close_tab(h, t)
        for nm in GATE_NAMES:
            try:
                db.delete_theme(nm)
            except Exception:
                pass
        left = [r["name"] for r in db.list_themes() if r["name"].startswith("_gate")]
        check(not left, "the gate leaves no saved theme of its own behind", ", ".join(left))
    if "--no-units" not in sys.argv:
        print("UNIT GATES", flush=True)
        for script, extra in (("_dj_setplan_unit.py", ["--music", music, "--quick"]), ("_dj_nightplan_unit.py", [])):
            t0 = time.time()
            p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "tests", script)] + extra, cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            tail = [ln for ln in (p.stdout or "").splitlines() if ln.strip()][-3:]
            check(p.returncode == 0, f"{script} passes", f"exit {p.returncode}, {time.time() - t0:.0f} s: " + " | ".join(tail)[:220])
    print("\n" + ("ALL OK" if not FAILS else f"FAIL ({len(FAILS)}): " + "; ".join(FAILS)), flush=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":                       # (a helper process re-imports __main__: nothing may run on import)
    import faulthandler
    sys.stdout.reconfigure(encoding="utf-8")
    faulthandler.enable()                        # (a native crash in Qt says where)
    code = main()
    faulthandler.disable()
    sys.stdout.flush()
    os._exit(code)                               # (skip the native libraries' exit-time noise)
