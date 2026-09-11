"""Perform tab: the autonomous DJ, steered, and SEEN.

Two modes. AUTOMIX is the night: whole songs, one seam at a time, planned
and executed through the gates. REMIX (lib/dj/remix.py) is the system
playing PARTS of songs together: two or three songs live on one clock, the
four stem lanes each assigned to a song, a move every phrase - a lane
crosses, a song enters through one lane, a song leaves lane by lane. Its
controls: BLEND (0 = song follows song as a morph, 1 = free recombination),
CHANGE (the mix speed box: a move every 4 / 8 / 16 / 32 bars), VOCALS (how
freely the vocal lane crosses), MIX NOW (a move on the next bar), HOLD
(freeze the combination), theme and energy as in Automix.

The system plays your library the way a night runs (lib/dj/system.py:
the brain picks, plans and executes every seam through the gates on the
real engine); you steer it from a few controls and never touch a track:

  THEME       what it plays and how it moves (track choice, energy arc,
              the style dice)
  MIX TYPE    the transition every seam uses - auto (the dice) or a style
              pinned through the same gates (a gated-off pin falls back)
  MIX SPEED   how long the blends run: short / normal / long / marathon
  MIX NOW     the next transition, now
  HOLD        one more phrase of this track first
  REROLL      a different next track
  DROP / NEXT DROP   the operator moments the night already has
  ENERGY      a lean on the arc's target, -0.4 .. +0.4

And the readout a performer needs: the playing track's map (sections,
energy, where we are, where the exit is planned), the next track's map
with its entry, the seam as planned (style, length, tempo, key shift,
pair score, whether a pin was honoured, the morph schedule, the seconds
to the blend and its "why" chips), the horizon of coming tracks, the
history of what played and how, the night's energy arc with the current
position, the engine (decks, lock, level, state) and the system's own
event log as it happens.
"""
import time

from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QPolygonF
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QListWidget, QPushButton,
                             QSlider, QSplitter, QVBoxLayout, QWidget, QGroupBox)

SPEEDS = (("short", 0.5), ("normal", 1.0), ("long", 2.0), ("marathon", 3.0))
REMIX_BARS = {"short": 4, "normal": 8, "long": 16, "marathon": 32}      # bars between the conductor's moves
MODES = ("Automix", "Remix")
LANES = ("drums", "bass", "other", "vocals")
STYLE_MENU = ("auto", "stem_morph", "long_blend", "bass_swap", "stem_bass_swap", "stem_drum_swap",
              "filter_sweep", "drum_bridge", "breakdown_swap", "loop_in", "loop_roll_exit", "cut_at_drop",
              "phrase_cut", "spinback_cut", "echo_out", "acapella_out", "acapella_in", "melody_carry",
              "long_fade")
KIND_COLORS = {"intro": QColor(90, 110, 160), "groove": QColor(70, 160, 110), "build": QColor(200, 150, 60),
               "drop": QColor(210, 80, 80), "break": QColor(120, 90, 170), "breakdown": QColor(120, 90, 170),
               "outro": QColor(110, 110, 120), "verse": QColor(80, 140, 150), "chorus": QColor(190, 100, 120)}


def _mmss(s):
    s = int(max(0, s or 0))
    return f"{s // 60}:{s % 60:02d}"


class TrackMap(QWidget):
    """One track's geography: sections coloured by kind, the energy curve over them, the playhead,
    the planned exit (current) or entry (next)."""

    def __init__(self, title):
        super().__init__()
        self.setMinimumHeight(64)
        self.title = title
        self.brief = None
        self.map = None
        self.pos = None
        self.window = None            # (from_s, to_s) of the blend on this track, when known

    def set(self, brief, tmap, pos=None, window=None):
        self.brief, self.map, self.pos, self.window = brief, tmap, pos, window
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(18, 18, 22))
        p.setPen(QColor(200, 200, 210))
        if not self.brief or not self.map:
            p.drawText(QRectF(6, 0, W - 12, H), Qt.AlignmentFlag.AlignVCenter, f"{self.title}: -")
            return
        dur = max(float(self.map.get("duration") or self.brief.get("duration_s") or 1.0), 1.0)
        x0, x1, y0, y1 = 6, W - 6, 18, H - 6
        sx = (x1 - x0) / dur
        for s0, s1, kind, voc in self.map.get("sections") or []:
            col = KIND_COLORS.get(kind, QColor(100, 100, 110))
            p.fillRect(QRectF(x0 + s0 * sx, y0, max(1.0, (s1 - s0) * sx), y1 - y0), col)
            if voc and voc > 0.5:
                p.fillRect(QRectF(x0 + s0 * sx, y1 - 4, max(1.0, (s1 - s0) * sx), 3), QColor(255, 230, 140))
        curve = self.map.get("energy") or []
        if len(curve) > 2:
            pts = [QPointF(x0 + (x1 - x0) * i / (len(curve) - 1), y1 - (y1 - y0) * min(1.0, float(v))) for i, v in enumerate(curve)]
            p.setPen(QPen(QColor(240, 240, 250, 200), 1.5))
            p.drawPolyline(QPolygonF(pts))
        if self.window:
            a, b = self.window
            p.fillRect(QRectF(x0 + a * sx, y0, max(2.0, (b - a) * sx), y1 - y0), QColor(255, 255, 255, 50))
        ex = self.map.get("exit_s")
        if ex is not None:
            p.setPen(QPen(QColor(255, 120, 90), 2))
            p.drawLine(QPointF(x0 + ex * sx, y0 - 3), QPointF(x0 + ex * sx, y1 + 3))
        en = self.map.get("entry_s")
        if en is not None:
            p.setPen(QPen(QColor(120, 220, 120), 2))
            p.drawLine(QPointF(x0 + en * sx, y0 - 3), QPointF(x0 + en * sx, y1 + 3))
        if self.pos is not None:
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(QPointF(x0 + self.pos * sx, y0 - 4), QPointF(x0 + self.pos * sx, y1 + 4))
        p.setPen(QColor(220, 220, 230))
        head = f"{self.title}: {self.brief.get('title')} - {self.brief.get('artist') or ''}   {self.brief.get('bpm', 0):.1f} bpm  {self.brief.get('camelot') or '?'}"
        if self.pos is not None:
            head += f"   {_mmss(self.pos)} / {_mmss(dur)}"
        if ex is not None:
            head += f"   exit {_mmss(ex)}"
        if en is not None:
            head += f"   entry {_mmss(en)}"
        p.drawText(QRectF(6, 0, W - 12, 16), Qt.AlignmentFlag.AlignVCenter, head)


class ArcView(QWidget):
    """The night's energy arc and where we are on it."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(54)
        self.curve, self.phase, self.heat, self.nudge = [], 0.0, 0.0, 0.0

    def set(self, curve, phase, heat, nudge):
        self.curve, self.phase, self.heat, self.nudge = curve or [], float(phase or 0), float(heat or 0), float(nudge or 0)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(18, 18, 22))
        x0, x1, y0, y1 = 6, W - 6, 16, H - 6
        if len(self.curve) > 2:
            pts = [QPointF(x0 + (x1 - x0) * i / (len(self.curve) - 1), y1 - (y1 - y0) * min(1.0, float(v))) for i, v in enumerate(self.curve)]
            p.setPen(QPen(QColor(200, 170, 90), 2))
            p.drawPolyline(QPolygonF(pts))
        x = x0 + (x1 - x0) * min(1.0, max(0.0, self.phase))
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(QPointF(x, y0), QPointF(x, y1))
        p.setPen(QColor(220, 220, 230))
        p.drawText(QRectF(6, 0, W - 12, 14), Qt.AlignmentFlag.AlignVCenter,
                   f"arc: {100 * self.phase:.0f}% through the set, target energy {self.heat:.2f}" + (f", lean {self.nudge:+.2f}" if abs(self.nudge) > 0.01 else ""))


class PerformTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.engine = None
        self.system = None
        self.remix = None
        self._seen_events = 0
        self._seen_moves = 0
        v = QVBoxLayout(self)
        # -- controls ------------------------------------------------------------------------
        row = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setToolTip("open the audio device and let the system play the library")
        self.start_btn.clicked.connect(self._toggle)
        row.addWidget(self.start_btn)
        self.mode_box = QComboBox()
        self.mode_box.addItems(list(MODES))
        self.mode_box.setToolTip("Automix: whole songs, one seam at a time.  Remix: parts of two or three songs together, changing every phrase")
        self.mode_box.currentTextChanged.connect(self._mode_changed)
        row.addWidget(self.mode_box)
        row.addWidget(QLabel("theme"))
        self.theme_box = QComboBox()
        from lib.dj.themes import PICKER_THEMES
        self.theme_box.addItems(list(PICKER_THEMES))
        self.theme_box.setCurrentText("groove")
        self.theme_box.currentTextChanged.connect(self._theme)
        row.addWidget(self.theme_box)
        row.addWidget(QLabel("mix type"))
        self.style_box = QComboBox()
        self.style_box.addItems(list(STYLE_MENU))
        self.style_box.currentTextChanged.connect(self._style)
        row.addWidget(self.style_box)
        row.addWidget(QLabel("mix speed"))
        self.speed_box = QComboBox()
        self.speed_box.addItems([s for s, _f in SPEEDS])
        self.speed_box.setCurrentText("normal")
        self.speed_box.currentTextChanged.connect(self._speed)
        row.addWidget(self.speed_box)
        row.addWidget(QLabel("   energy"))
        self.energy = QSlider(Qt.Orientation.Horizontal)
        self.energy.setRange(-40, 40)
        self.energy.setValue(0)
        self.energy.setMaximumWidth(160)
        self.energy.setToolTip("lean on the arc's energy target, -0.4 .. +0.4")
        self.energy.valueChanged.connect(self._energy)
        row.addWidget(self.energy)
        # remix-only: blend and vocal freedom
        self.blend_lbl = QLabel("   blend")
        row.addWidget(self.blend_lbl)
        self.blend = QSlider(Qt.Orientation.Horizontal)
        self.blend.setRange(0, 100)
        self.blend.setValue(70)
        self.blend.setMaximumWidth(160)
        self.blend.setToolTip("0: every move hands a lane to the newest song (song follows song, a morph)\n1: every move recombines freely across three songs")
        self.blend.valueChanged.connect(lambda x: self.remix and self.remix.set_blend(x / 100.0))
        row.addWidget(self.blend)
        self.vocals_lbl = QLabel("   vocals")
        row.addWidget(self.vocals_lbl)
        self.vocals = QSlider(Qt.Orientation.Horizontal)
        self.vocals.setRange(0, 100)
        self.vocals.setValue(40)
        self.vocals.setMaximumWidth(160)
        self.vocals.setToolTip("how freely the vocal lane crosses between songs")
        self.vocals.valueChanged.connect(lambda x: self.remix and self.remix.set_vocal_freedom(x / 100.0))
        row.addWidget(self.vocals)
        row.addStretch(1)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.buttons = {}
        for label, fn, tip in (("MIX NOW", self._mix_now, "the next transition, now"),
                               ("HOLD", self._hold, "one more phrase of this track before the seam"),
                               ("REROLL", self._reroll, "a different next track"),
                               ("DROP", lambda: self._moment("drop"), "the drop moment: build and land on the bar"),
                               ("NEXT DROP", lambda: self._moment("nextdrop"), "land the next track's drop on this one's bar"),
                               ("ABORT MIX", self._abort, "recall an armed transition before its point of no return")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row2.addWidget(b)
            self.buttons[label] = b
        row2.addStretch(1)
        self.state_lbl = QLabel("")
        row2.addWidget(self.state_lbl)
        v.addLayout(row2)
        # -- the maps ---------------------------------------------------------------------------
        self.map_now = TrackMap("PLAYING")
        self.map_next = TrackMap("NEXT")
        self.map_c = TrackMap("C")
        v.addWidget(self.map_now)
        v.addWidget(self.map_next)
        v.addWidget(self.map_c)
        self.map_c.hide()
        self._mode_changed(self.mode_box.currentText())
        self.seam_lbl = QLabel("")
        self.seam_lbl.setWordWrap(True)
        v.addWidget(self.seam_lbl)
        self.arc = ArcView()
        v.addWidget(self.arc)
        # -- lists --------------------------------------------------------------------------------
        split = QSplitter(Qt.Orientation.Horizontal)
        for name, attr in (("coming up", "horizon_list"), ("played", "history_list"), ("the system's log", "event_list")):
            box = QGroupBox(name)
            lay = QVBoxLayout(box)
            lst = QListWidget()
            lay.addWidget(lst)
            setattr(self, attr, lst)
            split.addWidget(box)
        split.setSizes([260, 320, 520])
        v.addWidget(split, 1)
        self.engine_lbl = QLabel("")
        self.engine_lbl.setWordWrap(True)
        v.addWidget(self.engine_lbl)
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._tick)

    # -- engine ---------------------------------------------------------------------
    def _toggle(self):
        if self.system is None and self.remix is None:
            try:
                self._start()
            except Exception as e:  # noqa: BLE001
                self.engine_lbl.setText(f"could not start: {type(e).__name__}: {e}")
                self.close()
        else:
            self.close()
            self.start_btn.setText("▶ Start")

    def _start(self):
        from lib.audio_engine import AudioEngine
        try:
            self.planner.analysis_tab.player.stop()
        except Exception:
            pass
        self.engine = AudioEngine()
        self.engine.start()
        if self.mode_box.currentText() == "Remix":
            from lib.dj.db import LibraryDB
            from lib.dj import brain as B
            from lib.dj.remix import RemixConductor
            db = LibraryDB(self.planner.music_dir)
            lib = [t for t in B.load_library(db) if not t.excluded]
            self.remix = RemixConductor(db, self.planner.music_dir, lib, theme=self.theme_box.currentText())
            self.remix.set_blend(self.blend.value() / 100.0)
            self.remix.set_vocal_freedom(self.vocals.value() / 100.0)
            self.remix.set_change_bars(REMIX_BARS.get(self.speed_box.currentText(), 8))
            self._energy(self.energy.value())
            self.engine.attach_track("dj_remix", self.remix.submix)
            ok = self.remix.start(threaded=True)
            if not ok:
                self.engine_lbl.setText(self.remix.last_error or "could not start")
                self.close()
                return
        else:
            from lib.dj.system import DJSystem
            self.system = DJSystem(self.planner.music_dir, engine=self.engine,
                                   theme=self.theme_box.currentText(), threaded=True)
            ok = self.system.start()
            if not ok:
                self.engine_lbl.setText(self.system.last_error or "could not start")
                self.close()
                return
            self._style(self.style_box.currentText())
            self._speed(self.speed_box.currentText())
        self.mode_box.setEnabled(False)
        self.start_btn.setText("■ Stop")
        self._seen_events = self._seen_moves = 0
        for lst in (self.horizon_list, self.history_list, self.event_list):
            lst.clear()
        self.buttons["HOLD"].setChecked(False)
        self._timer.start()

    def close(self):
        self._timer.stop()
        if self.system is not None:
            try:
                self.system.stop(fade_s=1.0)
            except Exception:
                pass
            self.system = None
        if self.remix is not None:
            try:
                self.remix.stop(fade_s=1.0)
                time.sleep(1.2)                 # let the fade play before the device closes
            except Exception:
                pass
            self.remix = None
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None
        self.mode_box.setEnabled(True)

    # -- controls -------------------------------------------------------------------
    def _theme(self, name):
        if self.system is not None:
            self.system.set_theme(name)
        if self.remix is not None:
            self.remix.set_theme(name)

    def _style(self, name):
        if self.system is not None:
            self.system.set_mix_style(None if name == "auto" else name)

    def _speed(self, name):
        if self.system is not None:
            self.system.set_mix_speed(dict(SPEEDS).get(name, 1.0))
        if self.remix is not None:
            self.remix.set_change_bars(REMIX_BARS.get(name, 8))

    def _energy(self, x):
        if self.system is not None:
            self.system.set_energy_nudge(x / 100.0)
        if self.remix is not None:
            self.remix.set_energy(0.6 + x / 100.0)

    def _mix_now(self):
        if self.system is not None:
            self.system.request_skip()
        if self.remix is not None:
            self.remix.next_move()

    def _hold(self):
        if self.system is not None:
            self.system.request_hold()
        if self.remix is not None:
            self.remix.set_hold(self.buttons["HOLD"].isChecked())

    def _reroll(self):
        if self.system is not None:
            self.system.request_reroll()

    def _moment(self, flavor):
        if self.system is not None:
            self.system.moment(flavor)

    def _abort(self):
        if self.system is not None:
            self.system.abort_transition()

    # -- readout --------------------------------------------------------------------
    def _tick(self):
        # An exception inside a Qt timer slot aborts the whole process with no Python trace (found
        # 2026-09-10: a readout bug took the planner down silently). The readout may never do that.
        try:
            self._tick_inner()
        except Exception as e:  # noqa: BLE001
            self.engine_lbl.setText(f"readout error: {type(e).__name__}: {e}")

    def _tick_inner(self):
        if self.remix is not None:
            self._tick_remix()
            return
        if self.system is None:
            return
        st = self.system.status()
        cur, nxt, plan = st.get("current") or {}, st.get("next") or {}, st.get("plan") or {}
        self.state_lbl.setText(f"{st.get('state')}   theme {st.get('theme')}   persona {st.get('persona') or '-'}   "
                               f"mix {st.get('mix_style') or 'auto'} x{st.get('mix_speed', 1.0):g}   pool {st.get('eligible_pool')}")
        # maps
        win_now = (plan["out_s"] - plan["beats"] * 60.0 / max(cur.get("bpm") or 120.0, 1.0), plan["out_s"]) if plan and cur else None
        self.map_now.set(cur, st.get("track_map"), pos=cur.get("pos_s"), window=win_now)
        win_next = (plan["in_s"], plan["in_s"] + plan["beats"] * 60.0 / max(nxt.get("bpm") or 120.0, 1.0)) if plan and nxt else None
        self.map_next.set(nxt, st.get("next_map"), window=win_next)
        # the seam
        bits = []
        if plan:
            bits.append(f"SEAM: {plan['style']}  {plan['beats']} beats")
            if st.get("blend_in_s") is not None:
                bits.append(f"blend in {st['blend_in_s']:.0f} s")
            bits.append(f"A exits {_mmss(plan['out_s'])}, B enters {_mmss(plan['in_s'])}")
            if abs(plan.get("rate", 1.0) - 1.0) > 1e-3 or abs(plan.get("a_rate", 1.0) - 1.0) > 1e-3:
                bits.append(f"tempo B x{plan['rate']:.3f}" + (f", A x{plan['a_rate']:.3f}" if abs(plan.get('a_rate', 1.0) - 1.0) > 1e-3 else ""))
            if plan.get("pitch_st"):
                bits.append(f"key shift {plan['pitch_st']:+d} st")
            if plan.get("pair_score") is not None:
                bits.append(f"pair {plan['pair_score']:.2f}")
            if st.get("mix_style"):
                bits.append(("pin honoured" + (f" (crossed {plan['pin_waived']})" if plan.get("pin_waived") else ""))
                            if plan.get("pin") else f"pin NOT honoured ({plan.get('pin_why_not') or 'gated'})")
            if plan.get("speed"):
                bits.append(f"length x{plan['speed']['factor']:g}: {plan['speed']['beats']} -> {plan['speed']['to']} beats")
            if plan.get("morph"):
                bits.append("morph " + ", ".join(f"{s} @{b:g}" for s, b in plan["morph"]))
            if st.get("seam_chips"):
                bits.append("why: " + ", ".join(str(c) for c in st["seam_chips"][:5]))
        elif st.get("state") == "playing":
            bits.append("SEAM: not planned yet (the brain plans when the exit comes into range)")
        if st.get("moment_flavor"):
            bits.append(f"moment {st['moment_flavor']} in {st.get('moment_eta')} s")
        if st.get("moment_denied"):
            bits.append(f"{st['moment_denied']['flavor']} refused: {st['moment_denied']['why']}")
        if st.get("layer"):
            bits.append(f"layer {st['layer']['label']} ({st['layer']['left_s']:.0f} s left)")
        self.seam_lbl.setText("   ·   ".join(bits))
        self.buttons["ABORT MIX"].setEnabled(bool(st.get("abortable")))
        # arc
        self.arc.set(st.get("arc_curve"), st.get("arc_phase"), st.get("arc_heat"), st.get("energy_nudge"))
        # lists
        hz = st.get("horizon") or []
        if self.horizon_list.count() != len(hz) or any(self.horizon_list.item(i).text().split("  ")[0] != str(h.get("title", ""))[:40] for i, h in enumerate(hz)):
            self.horizon_list.clear()
            for h in hz:
                self.horizon_list.addItem(f"{str(h.get('title', ''))[:40]}  {h.get('bpm', 0) or 0:.0f} bpm {h.get('camelot') or ''}"
                                          + (f"  {h.get('why')}" if h.get("why") else ""))
        hist = st.get("history") or []
        if self.history_list.count() != len(hist):
            self.history_list.clear()
            for h in hist:
                verdict = h.get("verdict") or h.get("fb") or ""
                self.history_list.addItem(f"{h.get('t')}  {str(h.get('title', ''))[:36]}  via {h.get('via')}"
                                          + (f"  energy {h.get('energy')}" if h.get("energy") is not None else "")
                                          + (f"  [{verdict}]" if verdict else ""))
            self.history_list.scrollToBottom()
        evs = st.get("recent_events") or []
        if len(evs) < self._seen_events:
            self.event_list.clear()
            self._seen_events = 0
        for e in evs[self._seen_events:]:
            kind = e.get("event", "?")
            rest = {k: v for k, v in e.items() if k not in ("t", "clock_s", "event") and not isinstance(v, (dict, list))}
            txt = ", ".join(f"{k} {v}" for k, v in list(rest.items())[:6])
            self.event_list.addItem(f"{time.strftime('%H:%M:%S', time.localtime(e.get('t', time.time())))}  {kind}  {txt}"[:180])
        self._seen_events = len(evs)
        if evs:
            self.event_list.scrollToBottom()
        # engine
        decks = st.get("deck_tel") or {}
        d = "   ".join(f"{k.upper()}{'*' if k == st.get('active_deck') else ''}: {'playing' if v.get('playing') else 'idle'} {_mmss(v.get('time_s'))} gain {float(v.get('gain') or 0):.2f} x{float(v.get('rate') or 1):.3f}"
                       + (" loop" if v.get("loop") else "") for k, v in decks.items())
        sync = st.get("sync") or {}
        ss = st.get("sync_stats") or {}
        lock = ""
        if sync:
            lock = f"   sync {sync.get('slave')}->{sync.get('master')} bias {sync.get('bias_beats')} beats"
            if sync.get("audible_err_beats") is not None:
                lock += f", audible {sync['audible_err_beats']:+.3f}"
            lock += f", resnaps {ss.get('resnaps', 0)} nudges {ss.get('nudges', 0)}"
        err = getattr(self.system, "last_error", None)
        self.engine_lbl.setText(f"{d}{lock}   level {st.get('level', 0):.2f}" + (f"   ERROR {err}" if err else ""))

    # -- remix mode -------------------------------------------------------------------------------
    def _mode_changed(self, mode):
        remix = mode == "Remix"
        for w in (self.blend_lbl, self.blend, self.vocals_lbl, self.vocals):
            w.setVisible(remix)
        for w in (self.style_box,):
            w.setEnabled(not remix)
        self.style_box.setToolTip("Remix mode has no seam styles: the lanes are the mix" if remix else "")
        self.speed_box.setToolTip("Remix: a move every 4 / 8 / 16 / 32 bars" if remix else "how long the blends run")
        self.map_c.setVisible(remix)
        self.map_now.title, self.map_next.title = ("A", "B") if remix else ("PLAYING", "NEXT")
        for name in ("REROLL", "DROP", "NEXT DROP", "ABORT MIX"):
            self.buttons[name].setEnabled(not remix)
        self.buttons["HOLD"].setCheckable(remix)
        self.buttons["HOLD"].setToolTip("freeze the lane map (songs that run out still loop and leave)" if remix else "one more phrase of this track before the seam")
        self.buttons["MIX NOW"].setToolTip("a move on the next bar" if remix else "the next transition, now")
        arc = getattr(self, "arc", None)
        if arc is not None:
            arc.setVisible(not remix)
        for m in (self.map_now, self.map_next):
            m.update()

    def _tick_remix(self):
        rc = self.remix
        st = rc.status()
        songs = st.get("songs") or {}
        lanes = st.get("lanes") or {}
        # the lane map, the clock, the phrase
        def who(d):
            return str((songs.get(d) or {}).get("title") or "?")[:28] if d else "-"
        lane_txt = "   ".join(f"{ln.upper()}: {who(lanes.get(ln))}" for ln in LANES)
        self.state_lbl.setText(f"clock {st.get('master_bpm') or 0:.1f} bpm  key {st.get('key_centre') or '?'}   "
                               f"bar {st.get('phrase_bars')}/{st.get('change_bars')}"
                               + ("   HOLD" if st.get("hold") else "")
                               + f"   blend {st.get('blend', 0):.2f}  vocals {st.get('vocal_freedom', 0):.2f}  energy {st.get('energy', 0):.2f}")
        self.seam_lbl.setText(lane_txt)
        # a map per deck: the song, its lanes, where it is, its loop
        for deck, widget in (("a", self.map_now), ("b", self.map_next), ("c", self.map_c)):
            s = songs.get(deck)
            if not s:
                widget.title = deck.upper() + (f"  decoding {st['decoding'][deck][:30]}" if deck in (st.get("decoding") or {}) else "")
                widget.set(None, None)
                continue
            tags = ",".join(s.get("lanes") or []) or ("staged" if not s.get("entered") else "-")
            flags = (" master" if deck == st.get("master") else "") + (" leaving" if s.get("leaving") else "") + (" loop" if s.get("loop") else "")
            lock = f" lock {s['lock_ms']:.0f} ms" if s.get("lock_ms") is not None else ""
            widget.title = f"{deck.upper()} [{tags}]{flags}{lock}"
            lp = s.get("loop")
            win = (lp[0] / 44100.0, lp[1] / 44100.0) if lp else None
            widget.set(s, s.get("map"), pos=s.get("time_s"), window=win)
        # lists: coming (decoding / staged), left (from the moves), the moves
        coming = [f"decoding {t}" for t in (st.get("decoding") or {}).values()]
        coming += [f"{deck.upper()} staged: {songs[deck]['title'][:40]} (rate {songs[deck]['rate']:.3f}, shift {songs[deck]['shift']:+d})"
                   for deck in songs if not songs[deck].get("entered")]
        if [self.horizon_list.item(i).text() for i in range(self.horizon_list.count())] != coming:
            self.horizon_list.clear()
            self.horizon_list.addItems(coming)
        moves = rc.moves
        if len(moves) < self._seen_moves:
            self.event_list.clear()
            self._seen_moves = 0
        for t, m in moves[self._seen_moves:]:
            self.event_list.addItem(f"{t}  {m}"[:180])
            if m.endswith(" leaves") or "enters through" in m:
                self.history_list.addItem(f"{t}  {m}"[:120])
        if len(moves) > self._seen_moves:
            self.event_list.scrollToBottom()
            self.history_list.scrollToBottom()
        self._seen_moves = len(moves)
        # engine line
        parts = []
        for deck, s in songs.items():
            parts.append(f"{deck.upper()}: {'playing' if s.get('playing') else 'idle'} {_mmss(s.get('time_s'))} x{s.get('rate', 1):.3f} {s.get('shift', 0):+d}st"
                         + (f" fit {s['compat']:.2f}" if s.get("compat") is not None else ""))
        err = st.get("error")
        self.engine_lbl.setText("   ".join(parts) + (f"   ERROR {err}" if err else ""))
