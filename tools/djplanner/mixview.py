"""Mix timeline: how the planned set actually mixes, DJ-software style.

Tracks sit on two alternating lanes so each seam's OVERLAP is visible.
Inside every block: the track's energy curve; at high zoom, beat ticks
(downbeats stronger) from the analyzed grid. Across every seam: the real
automation envelopes (white = gain; red/green/blue = low/mid/high EQ)
extracted from the exact events the engine will run (brain.preview_events).

Wheel = cursor-centered zoom, drag = pan, click a seam = select it
(seamSelected(i)); a playhead can be driven by the set preview player.
"""
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QWidget

RATE = 44100
LANE_COLORS = [QColor(45, 75, 105), QColor(50, 95, 75)]
EQ_COLORS = {"low": QColor(235, 105, 90), "mid": QColor(120, 210, 120),
             "high": QColor(110, 170, 240)}


def envelopes_from_events(events, blend_at, deck, init_gain, init_eq):
    """Breakpoint polylines for one deck: {'gain'|'low'|'mid'|'high':
    [(t_seconds_rel_blend_start, value), ...]}."""
    cur = {"gain": init_gain, "low": init_eq, "mid": init_eq,
           "high": init_eq}
    out = {k: [(-1e9, v)] for k, v in cur.items()}
    for e in sorted(events, key=lambda e: e["at"]):
        if e.get("deck") != deck:
            continue
        t = (e["at"] - blend_at) / RATE
        ramp = float(e.get("ramp_s", 0.05))
        if e["cmd"] == "gain":
            out["gain"] += [(t, cur["gain"]), (t + ramp, e["value"])]
            cur["gain"] = e["value"]
        elif e["cmd"] == "eq":
            for band in ("low", "mid", "high"):
                if e.get(band) is not None:
                    out[band] += [(t, cur[band]), (t + ramp, e[band])]
                    cur[band] = e[band]
        elif e["cmd"] == "stop":
            out["gain"] += [(t, cur["gain"]), (t, 0.0)]
            cur["gain"] = 0.0
    for k, v in out.items():
        v.append((1e9, cur[k]))
    return out


class MixTimeline(QWidget):
    seamSelected = pyqtSignal(int)
    timeClicked = pyqtSignal(float)          # output-time seconds

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(240)
        self.slots = []              # compiled slots
        self.seams = []              # per seam: {"start_s","blend_s","env_a","env_b"}
        self.total_s = 1.0
        self.view_t0 = 0.0
        self.view_t1 = 1.0
        self.playhead = None
        self.selected_seam = None
        self.live_bpm = None         # what the system is tracking right now
        self.bpm_curve = []          # [(t, bpm)] planned output tempo
        self._drag = None

    def set_plan(self, compiled, brain):
        """compiled = setlist.compile_plan result; brain for envelope math."""
        self.slots = compiled["slots"] if compiled else []
        self.seams = []
        self.selected_seam = None
        for i, s in enumerate(self.slots[:-1]):
            plan = s["transition"]
            if plan is None:
                continue
            cur, nxt = s["track"], self.slots[i + 1]["track"]
            beat_out = cur.period_s
            blend_s = (plan["beats"] * beat_out if plan["beats"]
                       else 20.0)
            try:
                events, swap_at, blend_at = brain.preview_events(
                    plan, cur, nxt)
                env_a = envelopes_from_events(events, blend_at, "a", 1.0, 1.0)
                env_b = envelopes_from_events(events, blend_at, "b", 0.0, 1.0)
            except Exception:
                env_a = env_b = None
            self.seams.append({
                "index": i,
                "start_s": self.slots[i + 1]["start_offset_s"],
                "blend_s": blend_s, "style": plan["style"],
                "env_a": env_a, "env_b": env_b,
            })
        self.total_s = max(compiled["total_s"] if compiled else 1.0, 1.0)
        # Planned output tempo: during a blend the incoming track is
        # stretched to the outgoing tempo, then glides home to its own bpm
        # (0.15%/s) - the strip shows exactly what the room will feel.
        self.bpm_curve = []
        GLIDE = 0.0015
        for i, s in enumerate(self.slots):
            t = s["track"]
            S = s["start_offset_s"]
            if i == 0:
                self.bpm_curve.append((S, t.bpm))
            else:
                prev = self.slots[i - 1]["track"]
                plan = self.slots[i - 1]["transition"]
                blend = next((sm["blend_s"] for sm in self.seams
                              if sm["index"] == i - 1), 10.0)
                swap_t = S + blend
                self.bpm_curve.append((swap_t, prev.bpm))
                rate = plan["rate"] if plan else 1.0
                glide_s = abs(rate - 1.0) / GLIDE
                self.bpm_curve.append((swap_t + glide_s, t.bpm))
        if self.slots:
            self.bpm_curve.append((self.total_s,
                                   self.slots[-1]["track"].bpm))
        self.view_t0, self.view_t1 = 0.0, self.total_s
        self.update()

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    # -- coords ------------------------------------------------------------
    def _t2x(self, t):
        span = max(self.view_t1 - self.view_t0, 1e-6)
        return (t - self.view_t0) / span * self.width()

    def _x2t(self, x):
        span = max(self.view_t1 - self.view_t0, 1e-6)
        return self.view_t0 + x / max(self.width(), 1) * span

    # -- painting -------------------------------------------------------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 16, 20))
        if not self.slots:
            p.setPen(QColor(120, 120, 130))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "compile a set (Set tab) to see the mix")
            return
        W, H = self.width(), self.height()
        strip_h = 26                 # planned-tempo strip on top
        lane_top = strip_h + 4
        lane_h = (H - lane_top - 22) // 2
        span = self.view_t1 - self.view_t0

        # -- BPM strip: the tempo the system tracks across the set ----------
        if self.bpm_curve:
            bpms = [b for _, b in self.bpm_curve]
            lo = min(bpms) - 3
            hi = max(bpms) + 3
            p.fillRect(QRectF(0, 0, W, strip_h), QColor(26, 26, 32))
            xs = np.array([b[0] for b in self.bpm_curve])
            vs = np.array([b[1] for b in self.bpm_curve])
            n_pts = max(W // 3, 16)
            ts = np.linspace(self.view_t0, self.view_t1, n_pts)
            ys = np.interp(ts, xs, vs)
            pts = [QPointF(self._t2x(t), 2 + (strip_h - 6)
                           * (1.0 - (v - lo) / max(hi - lo, 1e-6)))
                   for t, v in zip(ts, ys)]
            p.setPen(QPen(QColor(250, 200, 90), 2))
            p.drawPolyline(QPolygonF(pts))
            p.setPen(QColor(170, 170, 180))
            p.drawText(3, 11, f"tempo  {hi - 3:.0f}")
            p.drawText(3, strip_h - 3, f"{lo + 3:.0f} bpm")
            if self.playhead is not None:
                v = float(np.interp(self.playhead, xs, vs))
                label = (f"{self.live_bpm:.1f} bpm (live)"
                         if self.live_bpm else f"{v:.1f} bpm")
                x = self._t2x(self.playhead)
                p.setPen(QColor(255, 240, 200))
                p.drawText(int(min(max(x + 6, 60), W - 130)), 11, label)

        for i, s in enumerate(self.slots):
            t = s["track"]
            S = s["start_offset_s"]
            # Draw each block over its EXCLUSIVE drawn extent [S, S+play_s]
            # so blocks tile without overlap - clicking a block always maps
            # to that track (overlapping blocks caused seek-to-wrong-track).
            # The blend overlap is shown by the seam envelopes below.
            E = S + s["play_s"]
            x0, x1 = self._t2x(S), self._t2x(E)
            if x1 < 0 or x0 > W:
                continue
            lane = i % 2
            y0 = lane_top + lane * (lane_h + 4)
            r = QRectF(x0, y0, x1 - x0, lane_h)
            p.fillRect(r, LANE_COLORS[lane])
            p.setPen(QPen(QColor(0, 0, 0, 120), 1))
            p.drawRect(r)

            # Spectral content mapped from track-time [in_s, out_s+blend]:
            # three stacked band area-plots (low bottom / mid / high top),
            # each drawn as a FILLED HEIGHT (not a shade) so intensity is
            # actually readable - across a seam you SEE the incoming track's
            # bass stay low until the swap while its highs are already high.
            in_s = t.mix_ins[0]["time_s"] if t.mix_ins else 0.0
            seg_dur = E - S
            bc = t.row.get("band_curve") or {}
            n_cols = max(int(x1 - x0), 8)          # per-pixel temporal detail
            if all(k in bc and bc[k] for k in ("low", "mid", "high")):
                band_h = lane_h / 3.0
                for ri, (key, col) in enumerate(
                        (("high", EQ_COLORS["high"]),
                         ("mid", EQ_COLORS["mid"]),
                         ("low", EQ_COLORS["low"]))):
                    curve = bc[key]
                    base_y = y0 + (ri + 1) * band_h
                    p.fillRect(QRectF(x0, y0 + ri * band_h, x1 - x0,
                                      band_h), QColor(24, 24, 30))
                    poly = [QPointF(x0, base_y)]
                    for k in range(n_cols + 1):
                        ts = in_s + seg_dur * k / n_cols
                        ci = min(int(ts * 2), len(curve) - 1)
                        v = min(float(curve[ci]) / 1.1, 1.0)
                        poly.append(QPointF(x0 + (x1 - x0) * k / n_cols,
                                            base_y - (band_h - 1) * v))
                    poly.append(QPointF(x1, base_y))
                    fill = QColor(col)
                    fill.setAlpha(150)
                    p.setPen(Qt.PenStyle.NoPen)
                    p.setBrush(fill)
                    p.drawPolygon(QPolygonF(poly))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(col, 1))
                    p.drawPolyline(QPolygonF(poly[1:-1]))

            # Beat ticks at high zoom.
            if t.grid and span > 0:
                period = t.grid[0]["period_s"]
                px_beat = W / (span / period)
                if px_beat >= 6:
                    in_s = t.mix_ins[0]["time_s"] if t.mix_ins else 0.0
                    g = t.grid[0]
                    first_down = g["first_beat_s"] \
                        + t.downbeat_offset * g["period_s"]
                    tt = in_s - ((in_s - g["first_beat_s"]) % period)
                    while tt < in_s + (E - S):
                        out_t = S + (tt - in_s)
                        if self.view_t0 <= out_t <= self.view_t1:
                            x = self._t2x(out_t)
                            down = round((tt - first_down) / period) % 4 == 0
                            p.setPen(QPen(QColor(255, 255, 255,
                                                 80 if down else 30), 1))
                            p.drawLine(int(x), y0 + (2 if down else lane_h // 3),
                                       int(x), y0 + lane_h - 2)
                        tt += period
            if x1 - x0 > 60:
                p.setPen(QColor(240, 240, 240))
                p.drawText(QRectF(x0 + 4, y0 + 2, x1 - x0 - 8, 16),
                           Qt.AlignmentFlag.AlignLeft,
                           f"{t.title[:36]}  ({t.bpm:.0f} bpm {t.camelot})")

        # Seam envelopes + selection.
        for si, sm in enumerate(self.seams):
            S, blend = sm["start_s"], sm["blend_s"]
            x0, x1 = self._t2x(S - 2.0), self._t2x(S + blend + 2.0)
            if x1 < 0 or x0 > W:
                continue
            if si == self.selected_seam:
                p.fillRect(QRectF(x0, 0, x1 - x0, H),
                           QColor(255, 255, 255, 18))
            p.setPen(QColor(230, 230, 240))
            p.drawText(int(self._t2x(S)) + 2, lane_top + 12,
                       f"↳ {sm['style']}  ({sm['blend_s']:.0f}s)")
            for env, lane in ((sm["env_a"], 0), (sm["env_b"], 1)):
                if env is None:
                    continue
                y0 = lane_top + (lane if self.slots and lane < 2 else 0) \
                    * (lane_h + 4)
                # Envelope time 0 == blend start == seam start_s in out-time.
                for key, colw in (("gain", QColor(255, 255, 255)),
                                  ("low", EQ_COLORS["low"]),
                                  ("mid", EQ_COLORS["mid"]),
                                  ("high", EQ_COLORS["high"])):
                    pts = _sample_env(env[key], -2.0, blend + 2.0, 60)
                    poly = [QPointF(self._t2x(S + t_),
                                    y0 + lane_h * (1.0 - 0.85 * min(v, 1.2)))
                            for t_, v in pts]
                    p.setPen(QPen(colw, 2 if key == "gain" else 1))
                    p.drawPolyline(QPolygonF(poly))

        if self.playhead is not None:
            x = self._t2x(self.playhead)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(int(x), 0, int(x), H)

        # Ruler.
        p.setPen(QColor(150, 150, 160))
        step = 60 if span > 600 else (10 if span > 90 else 2)
        t = (int(self.view_t0 / step) + 1) * step
        while t < self.view_t1:
            x = self._t2x(t)
            p.drawText(int(x) + 2, H - 4, f"{int(t // 60)}:{int(t % 60):02d}")
            t += step

    # -- interaction --------------------------------------------------------
    def wheelEvent(self, ev):
        if not self.slots:
            return
        t_cursor = self._x2t(ev.position().x())
        factor = 0.8 if ev.angleDelta().y() > 0 else 1.25
        span = max(min((self.view_t1 - self.view_t0) * factor,
                       self.total_s), 4.0)
        frac = (t_cursor - self.view_t0) / max(self.view_t1 - self.view_t0,
                                               1e-6)
        self.view_t0 = max(0.0, t_cursor - frac * span)
        self.view_t1 = min(self.total_s, self.view_t0 + span)
        self.view_t0 = max(0.0, self.view_t1 - span)
        self.update()

    def mousePressEvent(self, ev):
        self._drag = (ev.position().x(), self.view_t0)
        self._moved = False

    def mouseMoveEvent(self, ev):
        if self._drag is not None:
            dx = ev.position().x() - self._drag[0]
            if abs(dx) > 4:
                self._moved = True
                span = self.view_t1 - self.view_t0
                t0 = float(np.clip(
                    self._drag[1] - dx / max(self.width(), 1) * span,
                    0, self.total_s - span))
                self.view_t0, self.view_t1 = t0, t0 + span
                self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag is not None and not self._moved:
            t = self._x2t(ev.position().x())
            self.timeClicked.emit(t)         # live preview seeks here
            best, best_d = None, 1e9
            for si, sm in enumerate(self.seams):
                mid = sm["start_s"] + sm["blend_s"] / 2
                d = abs(t - mid)
                if d < best_d:
                    best, best_d = si, d
            if best is not None and best_d < max(
                    self.seams[best]["blend_s"], 20.0):
                self.selected_seam = best
                self.seamSelected.emit(self.seams[best]["index"])
                self.update()
        self._drag = None


def _sample_env(bps, t0, t1, n):
    """Sample breakpoint list at n points across [t0, t1]."""
    ts = np.linspace(t0, t1, n)
    xs = np.array([b[0] for b in bps])
    vs = np.array([b[1] for b in bps])
    order = np.argsort(xs)
    return list(zip(ts, np.interp(ts, xs[order], vs[order])))
