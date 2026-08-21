"""Exit Compare: where the DJ leaves a track, and where it could leave.

The complaint this exists to answer is "songs mix out deep into their
comedown, so they're just noodling around for a while". That is a claim
about a POSITION on a song, so it should be a picture of the song.

  Lane A - the outgoing track, whole, with its section bands and energy
  curve. Every mix-out candidate the analyzer bookmarked is a tick whose
  height is its score. The play-time budget's floor is drawn as a hatched
  "can't exit before here" region: when it swallows the good candidates,
  the cause of a bad exit is visible rather than inferred. The last groove
  section's end is marked - past that line is the comedown.

  Two exits are drawn on the same lane: the pre-2026-08-07 engine (red)
  and the selected build (green), each with its blend window bracketed - the
  blend COMPLETES at out_s - the bracket is the stretch of A you actually
  hear underneath the incoming track.

  Lane B - the incoming track's head on its own time base, with its entry
  point marked, so the material arriving over A's exit is visible too.

Both sides render on demand through the shared offline renderer, so what
you hear is what a live night would make. Same pair, same style, same
beats - only out_s differs.

The budget fixes this tab was built to judge went LIVE on 2026-08-07
(brain.BUDGET_TAU_S, system.EXIT_MAX_FRAC), so the red line is now the
old engine, reconstructed by lib/dj/exitvariants.py, and green is what
ships. The tab stays useful for the same reason it was built: to see a
selection rule as a position on a song.
"""
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

from lib.dj.exitvariants import (BASELINE, blend_rel, choose_exit,
                                 last_groove_end)

RATE = 44100
BG = QColor(20, 20, 25)
TXT = QColor(160, 160, 170)
DIM = QColor(120, 120, 130)
SECTION_COLORS = {
    "intro": QColor(70, 90, 130), "outro": QColor(70, 90, 130),
    "groove": QColor(55, 115, 85), "build": QColor(185, 150, 55),
    "breakdown": QColor(95, 70, 135),
}
CUR = QColor(240, 100, 90)          # the BASELINE exit (old engine)
NEW = QColor(90, 220, 120)          # the selected variant (live by default)
CAND = QColor(150, 160, 200)
FLOOR = QColor(230, 80, 70)
ENERGY = QColor(120, 150, 200)


class ExitLanes(QWidget):
    """The picture. Fed by set_seam(); draws nothing until then."""
    seekRequested = pyqtSignal(float)     # seconds into the RENDER

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(300)
        self.seam = None
        self.simple = False
        self.playhead = None              # seconds into the render, or None
        self._render_span = None          # (t0_a, t1_a) of A covered by audio

    def set_seam(self, seam, simple=False):
        """`simple` drops the exit-budget layer (hatch, candidate ticks,
        the second exit line) and draws just the songs and where the seam
        lands - what the Gate Check panel needs, same picture otherwise."""
        self.seam = seam
        self.simple = simple
        self.playhead = None
        self.update()

    def set_playhead(self, t, key="cur"):
        """`t` is seconds into the RENDER; the lane draws in A's own time,
        so it maps back through the renderer's pre-roll."""
        self.playhead = (t, key)
        self.update()

    # -- geometry ----------------------------------------------------------
    def _lanes(self):
        H = self.height()
        pad = 18
        lane_h = (H - 3 * pad) // 2
        return pad, lane_h, pad * 2 + lane_h

    def _t2x(self, t, dur):
        W = max(self.width() - 70, 10)
        return 55 + W * max(0.0, min(t / max(dur, 1e-6), 1.0))

    @staticmethod
    def _label_right(p, placed, x_right, y, text, step=13):
        """Right-edge-anchored label that drops a row instead of printing
        over an already-placed one. The budget note, both exit tags and
        'comedown' all crowd the same corner on a late exit (lab-reported:
        overlapping text made the lens unreadable)."""
        w = p.fontMetrics().horizontalAdvance(text)
        x_right = max(x_right, 57 + w)       # keep it on the plot
        r = QRectF(x_right - w, y, w + 1, step)
        for _ in range(6):
            if not any(r.intersects(q) for q in placed):
                break
            r.translate(0, step)
        p.drawText(r, Qt.AlignmentFlag.AlignLeft, text)
        placed.append(r)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        if self.seam is None:
            p.setPen(TXT)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Pick a seam to see where it leaves.")
            return
        s = self.seam
        y_a, lane_h, y_b = self._lanes()
        self._draw_a(p, s, y_a, lane_h)
        self._draw_b(p, s, y_b, lane_h)

    # -- lane A: the outgoing track ---------------------------------------
    def _draw_a(self, p, s, y, h):
        a = s["a"]
        dur = a.duration_s
        strip = 13
        p.setPen(TXT)
        p.drawText(QRectF(2, y, 50, strip), Qt.AlignmentFlag.AlignVCenter,
                   "A/OUT")

        # sections
        for sec in a.sections or []:
            x0, x1 = self._t2x(sec["start_s"], dur), self._t2x(sec["end_s"], dur)
            col = SECTION_COLORS.get(sec.get("kind"), QColor(85, 85, 90))
            p.fillRect(QRectF(x0, y, x1 - x0, strip), col)
            if x1 - x0 > 44:
                p.setPen(QColor(235, 235, 235, 200))
                p.drawText(QRectF(x0 + 3, y, x1 - x0 - 4, strip),
                           Qt.AlignmentFlag.AlignVCenter, sec.get("kind", ""))
            wash = QColor(col)
            wash.setAlpha(30)
            p.fillRect(QRectF(x0, y + strip, x1 - x0, h - strip), wash)

        body_top = y + strip
        body_h = h - strip
        self._draw_energy(p, a, dur, body_top, body_h)

        # BUDGET FLOOR: everything left of it is unreachable today. This is
        # the whole point of the picture - when the block covers the good
        # candidates, that IS the bug. DARKEN rather than tint: a red wash
        # over half the lane just reads as background.
        aft = None if self.simple else s["cur"]["after_s"]
        xf = None
        if aft and aft > 0:
            xf = self._t2x(aft, dur)
            p.fillRect(QRectF(55, body_top, xf - 55, body_h),
                       QColor(0, 0, 0, 105))
            hatch = QColor(FLOOR)
            hatch.setAlpha(38)
            p.fillRect(QRectF(55, body_top, xf - 55, body_h),
                       QBrush(hatch, Qt.BrushStyle.BDiagPattern))
            p.setPen(QPen(FLOOR, 2, Qt.PenStyle.DashLine))
            p.drawLine(int(xf), int(body_top), int(xf), int(body_top + body_h))

        # last groove end - past here is comedown
        placed = []                 # label rects; later text dodges them
        lge = last_groove_end(a)
        if lge:
            xg = self._t2x(lge, dur)
            p.setPen(QPen(QColor(230, 200, 90, 190), 1, Qt.PenStyle.DotLine))
            p.drawLine(int(xg), int(body_top), int(xg), int(body_top + body_h))
            p.setPen(QColor(230, 200, 90, 210))
            r = QRectF(xg + 3, body_top + body_h - 16,
                       p.fontMetrics().horizontalAdvance("comedown →") + 2, 13)
            p.drawText(r, Qt.AlignmentFlag.AlignLeft
                       | Qt.AlignmentFlag.AlignBottom, "comedown →")
            placed.append(r)

        # Candidate mix-outs: the pool best_pair actually chooses from, tick
        # height by score. Drawn AFTER the floor so the ones it swallows are
        # still visible - seeing a tall tick inside the hatch is the whole
        # diagnosis in one glance.
        mx = max([o["score"] for o in a.mix_outs] or [1.0]) or 1.0
        base = body_top + body_h - 6
        if self.simple:
            mx = None
        if mx is not None:
            p.setPen(QPen(QColor(CAND.red(), CAND.green(), CAND.blue(), 90), 1))
            p.drawLine(55, int(base), int(self._t2x(dur, dur)), int(base))
        for o in (() if mx is None else a.mix_outs):
            x = self._t2x(o["time_s"], dur)
            hh = 10 + 34 * (o["score"] / mx)
            killed = xf is not None and x < xf
            col = QColor(FLOOR) if killed else QColor(CAND)
            p.setPen(QPen(col, 2))
            p.drawLine(int(x), int(base - hh), int(x), int(base))
            p.setBrush(QBrush(col if not killed else BG))
            p.drawRect(QRectF(x - 2.5, base - hh - 5, 5, 5))
            p.setBrush(Qt.BrushStyle.NoBrush)
        # the two exits
        lanes = ((("cur", NEW, "seam"),) if self.simple
                 else (("cur", CUR, "before"), ("new", NEW, "now")))
        for key, col, label in lanes:
            self._draw_exit(p, s, key, col, label, dur, body_top, body_h,
                            placed)

        # Budget note LAST so it drops below the exit tags when they crowd:
        # the floor often lands right beside the old exit, and both texts
        # sat on the same top row.
        if aft and xf is not None:
            p.setPen(FLOOR)
            self._label_right(p, placed, xf - 4, body_top + 1,
                              f"budget forbids exiting before "
                              f"{aft/dur*100:.0f}% →")

        # Playhead, mapped from render time back onto A's own timeline.
        if self.playhead is not None:
            t_r, key = self.playhead
            t_a = self._render_t0(s, key) + t_r
            if 0 <= t_a <= dur:
                xp = self._t2x(t_a, dur)
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.drawLine(int(xp), int(body_top), int(xp),
                           int(body_top + body_h))

        self._ruler(p, dur, body_top + body_h, y + h)

    @staticmethod
    def _render_t0(s, key):
        """A-time at render second 0 - render_seam cues A a whole blend
        plus 6s before out_s (audition.py), so the two must agree."""
        pre = max(12.0, s["beats"] * 60.0 / max(s["a"].bpm, 60.0) + 6.0)
        return s[key]["out_s"] - pre

    def _draw_exit(self, p, s, key, col, label, dur, top, h, placed):
        e = s[key]
        out_s = e["out_s"]
        x = self._t2x(out_s, dur)
        blend = s["blend_s"]
        xb = self._t2x(max(out_s - blend, 0.0), dur)
        band = QColor(col)
        band.setAlpha(46)
        # The blend window: the stretch of A heard under the incoming track.
        p.fillRect(QRectF(xb, top, x - xb, h), band)
        p.setPen(QPen(col, 2))
        p.drawLine(int(x), int(top), int(x), int(top + h))
        p.setPen(QPen(col, 1))
        p.drawLine(int(xb), int(top + h - 4), int(x), int(top + h - 4))
        # arrow head at out_s
        p.setBrush(QBrush(col))
        p.drawPolygon(QPolygonF([QPointF(x, top + h - 4),
                                 QPointF(x - 7, top + h - 9),
                                 QPointF(x - 7, top + h + 1)]))
        p.setBrush(Qt.BrushStyle.NoBrush)
        tag = f"{label} {out_s/dur*100:.0f}%"
        if e["fallback"]:
            tag += " (fallback)"
        p.setPen(col)
        yt = top + (2 if key == "cur" else 15)
        self._label_right(p, placed, x - 3, yt, tag)

    # -- lane B: the incoming track ---------------------------------------
    def _draw_b(self, p, s, y, h):
        b = s["b"]
        dur = b.duration_s
        strip = 13
        p.setPen(TXT)
        p.drawText(QRectF(2, y, 50, strip), Qt.AlignmentFlag.AlignVCenter,
                   "B/IN")
        for sec in b.sections or []:
            x0, x1 = self._t2x(sec["start_s"], dur), self._t2x(sec["end_s"], dur)
            col = SECTION_COLORS.get(sec.get("kind"), QColor(85, 85, 90))
            p.fillRect(QRectF(x0, y, x1 - x0, strip), col)
            if x1 - x0 > 44:
                p.setPen(QColor(235, 235, 235, 200))
                p.drawText(QRectF(x0 + 3, y, x1 - x0 - 4, strip),
                           Qt.AlignmentFlag.AlignVCenter, sec.get("kind", ""))
            wash = QColor(col)
            wash.setAlpha(30)
            p.fillRect(QRectF(x0, y + strip, x1 - x0, h - strip), wash)
        body_top, body_h = y + strip, h - strip
        self._draw_energy(p, b, dur, body_top, body_h)
        in_s = s["in_s"]
        if in_s is not None:
            x = self._t2x(in_s, dur)
            p.setPen(QPen(NEW, 2))
            p.drawLine(int(x), int(body_top), int(x), int(body_top + body_h))
            band = QColor(NEW)
            band.setAlpha(40)
            p.fillRect(QRectF(x, body_top,
                              self._t2x(in_s + s["blend_s"], dur) - x, body_h),
                       band)
            p.setPen(NEW)
            p.drawText(QRectF(x + 4, body_top + 2, 160, 13),
                       Qt.AlignmentFlag.AlignLeft,
                       f"enters {in_s/dur*100:.0f}%")
        self._ruler(p, dur, body_top + body_h, y + h)

    def _draw_energy(self, p, track, dur, top, h):
        c = np.asarray(track.row.get("energy_curve") or [], dtype=np.float64)
        if len(c) < 8:
            return
        W = int(self.width() - 70)
        pts = []
        for i in range(W):
            t = dur * i / max(W - 1, 1)
            j = min(int(t * 2), len(c) - 1)
            v = float(np.clip(c[j] / 1.2, 0.0, 1.0))
            pts.append(QPointF(55 + i, top + h - 4 - v * (h - 10)))
        p.setPen(QPen(ENERGY, 1))
        p.drawPolyline(QPolygonF(pts))

    def _ruler(self, p, dur, y0, y1):
        p.setPen(DIM)
        step = 60.0 if dur < 600 else 120.0
        t = step
        while t < dur:
            x = self._t2x(t, dur)
            p.drawLine(int(x), int(y1) - 4, int(x), int(y1))
            p.drawText(int(x) + 2, int(y1) - 1, f"{int(t//60)}:00")
            t += step

    # -- interaction -------------------------------------------------------
    def mousePressEvent(self, ev):
        if self.seam is None:
            return
        y_a, lane_h, _ = self._lanes()
        if ev.position().y() > y_a + lane_h:
            return
        dur = self.seam["a"].duration_s
        W = max(self.width() - 70, 10)
        t = (ev.position().x() - 55) / W * dur
        self.seekRequested.emit(max(0.0, t))


class _AudioWorker(QThread):
    """Render ONE variant's audio off the UI thread. The picture never waits
    on this - it is drawn from the plan the moment a setting changes, and
    audio is only produced when a Play button asks for it."""
    ready = pyqtSignal(str, object)          # key, audio
    failed = pyqtSignal(str)

    def __init__(self, db, a, b, plan, key):
        super().__init__()
        self.db, self.a, self.b = db, a, b
        self.plan, self.key = plan, key

    def run(self):
        try:
            from lib.dj.audition import render_seam
            self.ready.emit(self.key,
                            render_seam(self.db, self.a, self.b, self.plan))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# (variant key, menu label, what it does). The RED line is always
# exitvariants.BASELINE - the pre-2026-08-07 engine - so this dropdown
# only ever chooses what the GREEN line is solved with.
FIX_CHOICES = [
    ("current", "Live now: cap budget + soften floor",
     "What the DJ does today. The duration-35s fallback stops firing "
     "entirely (25% -> 0% of seams) and exits landing past the last groove "
     "section drop 19.2% -> 9.4%, at a cost of a median 3% of track length "
     "in play time."),
    ("soft_only", "Soften the floor only (no cap)",
     "Half of the live change: the play-time budget stops DELETING earlier "
     "exit candidates and penalises them instead, but the drawn budget is "
     "still allowed to exceed the record."),
    ("clamp_only", "Cap the budget only (old hard filter)",
     "The other half: the budget is capped at 72% of the track, but "
     "best_pair still deletes every candidate below it rather than "
     "penalising them."),
    ("win", "Live + fix the energy-window bug",
     "best_pair judges an exit by the section AFTER out_s, but the blend "
     "COMPLETES at out_s so the track actually plays the section BEFORE it "
     "(measured correlation -0.06). A real defect - but on its own it moved "
     "nothing measurable, so it is NOT applied. This is where to disprove "
     "that by ear."),
]
ARC_CHOICES = [
    (0.15, "valley (0.15)",
     "Quiet stretch of the night. The budget is drawn LONGEST here, which is "
     "why valley exits land deepest into the comedown."),
    (0.50, "climb (0.50)", "Mid-night. Middling budget draw."),
    (0.90, "peak (0.90)",
     "Peak heat. Records rotate fastest, so the budget is drawn shortest and "
     "exits are least likely to be forced late."),
]


class ExitCompareTab(QWidget):
    """Hold one seam still and vary the settings on it.

    The pair is only re-picked by "New seam". Every dropdown re-solves the
    SAME pair and redraws immediately, because the point of the tool is to
    see one song's exit move as the rule changes - re-rolling the pair on a
    settings change would make the comparison impossible.
    """

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        import random
        from tools.dj.planner.player import TrackPlayer
        self.player = TrackPlayer()
        self.rng = random.Random()
        self.brain = None
        self.seam = None
        self._pair = None            # (a, b, meta) - survives setting changes
        self._budget_r = 0.5         # the draw's uniform, held so the floor
        #                              moves only because a SETTING moved
        self._audio = {}             # key -> rendered ndarray, cleared on solve
        self._worker = None
        self._pending_seek = None
        self._playing = "cur"

        v = QVBoxLayout(self)
        row = QHBoxLayout()

        row.addWidget(QLabel("Theme:"))
        from lib.dj.themes import BUILTIN_THEMES
        self.theme_box = QComboBox()
        self.theme_box.addItems(sorted(BUILTIN_THEMES))
        self.theme_box.setCurrentText("groove")
        self.theme_box.setToolTip(
            "The night's theme. Sets the style menu AND the play-time budget "
            "range (min_play_s..max_play_s) that decides how late the exit is "
            "allowed to be - so changing it moves the hatched region.")
        self.theme_box.currentTextChanged.connect(self._theme_changed)
        row.addWidget(self.theme_box)

        row.addWidget(QLabel("Arc:"))
        self.heat_box = QComboBox()
        for heat, label, tip in ARC_CHOICES:
            self.heat_box.addItem(label, heat)
        for i, (heat, label, tip) in enumerate(ARC_CHOICES):
            self.heat_box.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
        self.heat_box.setToolTip(
            "Where in the night this seam happens. The budget is arc-coupled: "
            "valleys let records breathe (latest exits), peak rotates fast "
            "(earliest). Changing this moves the budget floor on the SAME "
            "pair.")
        self.heat_box.currentIndexChanged.connect(self._resolve)
        row.addWidget(self.heat_box)

        row.addWidget(QLabel("Fix:"))
        self.var_box = QComboBox()
        for key, label, tip in FIX_CHOICES:
            self.var_box.addItem(label, key)
        for i, (key, label, tip) in enumerate(FIX_CHOICES):
            self.var_box.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
        self.var_box.setToolTip(
            "Which candidate fix the GREEN exit is solved with. The red "
            "'current' exit never changes - it is what ships today. Hover an "
            "entry for what it does.")
        self.var_box.setCurrentIndex(0)   # FIX_CHOICES[0] is the live build
        self.var_box.currentIndexChanged.connect(self._resolve)
        row.addWidget(self.var_box)

        self.gen_btn = QPushButton("⟳ New seam")
        self.gen_btn.setToolTip(
            "Pick a DIFFERENT pair of tracks. The dropdowns re-solve the "
            "current pair instead, so you can watch one song's exit move.")
        self.gen_btn.clicked.connect(self._generate)
        row.addWidget(self.gen_btn)
        row.addStretch(1)
        v.addLayout(row)

        self.card = QLabel(
            "Press “New seam”: the brain picks a pair and plans it "
            "exactly like a live night. Then change the dropdowns - the same "
            "seam re-solves and redraws instantly.")
        self.card.setWordWrap(True)
        self.card.setStyleSheet("font-size: 14px; padding: 8px;")
        v.addWidget(self.card)

        self.lanes = ExitLanes()
        self.lanes.seekRequested.connect(self._seek_a)
        v.addWidget(self.lanes, 1)

        legend = QLabel(
            "<span style='color:#f0645a'>▌ before this change</span> &nbsp; "
            "<span style='color:#5adc78'>▌ the selected build</span> &nbsp; "
            "the shaded band behind each = the blend window, the stretch of "
            "the outgoing track you hear underneath the incoming one "
            "(the blend <i>completes</i> at the line). &nbsp;"
            "<span style='color:#f0645a'>▨ hatched</span> = the "
            "play-time budget forbids exiting there; ticks are the analyzer's "
            "exit candidates, tall = better, "
            "<span style='color:#f0645a'>red = killed by the budget</span>. "
            "<span style='color:#e6c85a'>┊</span> = end of the last "
            "groove section; past it is comedown.")
        legend.setWordWrap(True)
        legend.setStyleSheet("color:#8a9098; font-size:11px; padding:2px 8px;")
        v.addWidget(legend)

        row2 = QHBoxLayout()
        self.play_cur = QPushButton("▶ Play the OLD exit")
        self.play_cur.setStyleSheet("color:#f0645a;")
        self.play_cur.clicked.connect(lambda: self._play("cur"))
        self.play_new = QPushButton("▶ Play the NEW exit")
        self.play_new.setStyleSheet("color:#5adc78;")
        self.play_new.clicked.connect(lambda: self._play("new"))
        for b in (self.play_cur, self.play_new):
            b.setEnabled(False)
            row2.addWidget(b)
        stop = QPushButton("■ Stop")
        stop.clicked.connect(lambda: self.planner.stop_all_playback())
        row2.addWidget(stop)
        row2.addStretch(1)
        v.addLayout(row2)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color:#9aa0a8; padding: 0 8px;")
        v.addWidget(self.detail)

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._follow)
        self._tick.start(60)

    # -- settings ----------------------------------------------------------
    @property
    def heat(self):
        return self.heat_box.currentData()

    @property
    def variant(self):
        return self.var_box.currentData()

    def _theme_changed(self):
        self.brain = None
        self._resolve()

    def _ensure_brain(self):
        if self.brain is None:
            from lib.dj.brain import Brain
            from lib.dj.themes import get_theme
            self.brain = Brain(self.planner.library,
                               get_theme(self.theme_box.currentText()),
                               seed=self.rng.randrange(1 << 30))
            try:
                self.brain.load_pair_memory(self.planner.db)
            except Exception:
                pass
        return self.brain

    # -- pair choice -------------------------------------------------------
    def _generate(self):
        if not self.planner.library:
            self.card.setText("No library loaded.")
            return
        brain = self._ensure_brain()
        pool = [t for t in self.planner.library
                if t.duration_s >= 150 and t.mix_outs and t.sections]
        for _ in range(40):
            a = self.rng.choice(pool)
            brain.note_played(a)
            b, meta = brain.choose_next(a, self.heat, a.bpm)
            if b is None:
                continue
            self._pair = (a, b, meta)
            self._budget_r = self.rng.random()
            self._resolve()
            return
        self.card.setText("Couldn't find a mixable pair - try another theme.")

    # -- solving (cheap, synchronous, redraws immediately) -----------------
    def _budget_floor(self, a):
        """system.py's arc-coupled _draw_exit, capped at the planning
        deadline. The uniform is HELD across setting changes so the floor
        moves only because the theme or the arc moved, never because a
        fresh random number was drawn."""
        th = self._ensure_brain().theme
        span = max(th.max_play_s - th.min_play_s, 1.0)
        heat = self.heat
        frac = min(1.0, self._budget_r * (1.0 - 0.55 * heat)
                   + 0.25 * (1.0 - heat))
        return min(th.min_play_s + frac * span, a.duration_s - 25.0)

    def _resolve(self):
        """Re-solve the CURRENT pair under the current settings and redraw.
        No audio: the picture must never wait on a render."""
        if self._pair is None:
            return
        a, b, meta = self._pair
        brain = self._ensure_brain()
        try:
            after = self._budget_floor(a)
            cur = choose_exit(brain, a, b, after, BASELINE)
            new = choose_exit(brain, a, b, after, self.variant)
            base = brain.plan_transition(a, b, meta, after_s=after)
        except Exception as e:
            self.detail.setText(f"could not plan this pair: {e}")
            return
        # long_fade carries beats=0 - its overlap is set by the fade knobs,
        # not a beat count, so beats*period would draw a zero-width bracket
        # on exactly the style that fades longest.
        blend_s = base["beats"] * a.period_s
        if blend_s <= 0:
            from lib.dj.brain import TUNE_DEFAULTS as _T
            blend_s = _T["fade_lead_a"] + _T["fade_stop_lead"]
        # ONE plan for both exits, so style/beats/rate are identical and the
        # only difference is out_s. Snap each to a phrase as the live path
        # does, so the drawn line is the line that would be played.
        for e in (cur, new):
            e["out_s"] = a.nearest_phrase(e["out_s"])
        self._audio.clear()          # exits moved: any render is now stale
        self.seam = {"a": a, "b": b, "cur": cur, "new": new,
                     "in_s": base["in_s"], "style": base["style"],
                     "beats": base["beats"], "blend_s": blend_s,
                     "plan": base, "audio": self._audio,
                     "variant": self.variant}
        self.player.pause()
        self.lanes.set_seam(self.seam)
        self.play_cur.setEnabled(True)
        self.play_new.setEnabled(True)
        self._describe()

    def _describe(self):
        s = self.seam
        a, cur, new = s["a"], s["cur"], s["new"]
        lge = last_groove_end(a)

        def verdict(e):
            bits = [f"exits at {e['out_s']/a.duration_s*100:.0f}% "
                    f"({e['out_s']/60:.1f} min)"]
            if lge:
                past = e["out_s"] - lge
                bits.append(f"{past:+.0f}s vs the last groove section"
                            + (" — in the comedown" if past > 0 else ""))
            rel = blend_rel(a, e["out_s"])
            if rel is not None:
                bits.append(f"blend window at {rel:.2f}× the track's body")
            if e["fallback"]:
                bits.append("<b>no candidate survived the budget → "
                            "duration-35s fallback</b>")
            return ", ".join(bits)

        self.card.setText(f"{a.title}  →  {s['b'].title}      "
                          f"[{s['style']}, {s['beats']} beats, "
                          f"{s['blend_s']:.0f}s blend]")
        moved = cur["out_s"] - new["out_s"]
        if abs(moved) < 1.0:
            verdict_line = ("this build changes nothing on this seam — "
                            "the budget wasn't the binding constraint here")
        else:
            verdict_line = (f"it moves the exit <b>{abs(moved):.0f}s "
                            f"{'earlier' if moved > 0 else 'later'}</b>")
        self.detail.setText(
            f"<b style='color:#f0645a'>before (legacy)</b>: {verdict(cur)}<br>"
            f"<b style='color:#5adc78'>"
            f"{self.var_box.currentText()}</b>: {verdict(new)}<br>"
            f"{verdict_line}. Budget floor at {cur['after_s']/60:.1f} min of "
            f"{a.duration_s/60:.1f} min "
            f"({cur['after_s']/a.duration_s*100:.0f}% of the track). "
            f"Press a Play button to render and hear it; click lane A to "
            f"start from a point.")

    # -- playback (renders on demand) --------------------------------------
    def _plan_for(self, key):
        plan = dict(self.seam["plan"])
        plan["out_s"] = self.seam[key]["out_s"]
        return plan

    def _play(self, key, seek_to=None):
        if self.seam is None:
            return
        self._playing = key
        if key in self._audio:
            self._start(key, seek_to)
            return
        if self._worker is not None and self._worker.isRunning():
            self.detail.setText("still rendering - one moment...")
            return
        self._pending_seek = seek_to
        self.play_cur.setEnabled(False)
        self.play_new.setEnabled(False)
        self.detail.setText(f"rendering the {key} exit...")
        a, b, _ = self._pair
        self._worker = _AudioWorker(self.planner.db, a, b,
                                    self._plan_for(key), key)
        self._worker.ready.connect(self._audio_ready)
        self._worker.failed.connect(self._failed)
        self._worker.start()

    def _audio_ready(self, key, audio):
        self._audio[key] = audio
        self.play_cur.setEnabled(True)
        self.play_new.setEnabled(True)
        self._describe()
        self._start(key, self._pending_seek)
        self._pending_seek = None

    def _start(self, key, seek_to=None):
        self.planner.claim_playback("exitcompare")
        self.player.load(self._audio[key])
        if seek_to is not None:
            self.player.seek(seek_to)
        self.player.play()

    def _failed(self, msg):
        self.play_cur.setEnabled(True)
        self.play_new.setEnabled(True)
        self.detail.setText("render failed: " + msg)

    def _seek_a(self, t_a):
        """Click on lane A: play from that point of A, in whichever variant
        was last played."""
        if self.seam is None:
            return
        key = self._playing
        off = t_a - ExitLanes._render_t0(self.seam, key)
        if off < 0:
            self.detail.setText(
                "that point is before the rendered window — the render "
                "covers the blend, not the whole song")
            return
        self._play(key, seek_to=off)

    def _follow(self):
        if self.seam is None or not self.player.playing:
            return
        self.lanes.set_playhead(self.player.time_s(), self._playing)

    def stop_playback(self):
        try:
            self.player.pause()
        except Exception:
            pass

    def close_audio(self):
        try:
            self.player.close()
        except Exception:
            pass
