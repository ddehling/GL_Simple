"""Seam scope: the audition's mechanics, drawn on the mix timeline.

Zoomed to the rendered seam (not the two whole songs), with everything
the blend actually does to the sound:

  * per-deck GAIN and EQ (low/mid/high) envelopes, plus stem gains when a
    stem style diverges one - the "volumes used in mixing"
  * both decks' BEAT GRIDS meeting in the middle, bar-numbered relative
    to the seam, so relative beat location and phase are visible; the
    scripted sync snap is simulated, and each blend downbeat pair that
    lands in the audible flam window gets bridged
  * a marker rail for every other command the script issues (cue, start,
    loop, brake, echo, filter, stop) plus the seam, the swap and the
    point of no return
  * ONE continuous playhead down the whole picture

It draws from the EXACT event list the renderer ran (`render_seam`'s
`info` out-param), so the picture cannot drift from the audio. The one
approximation is the PLL: the submix trims phase continuously after the
sync snap, which this simulates only as the initial snap.
"""
import math

from PyQt6.QtCore import Qt, QEvent, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QToolTip, QWidget

RATE = 44100

BG = QColor(18, 18, 23)
PANEL = QColor(26, 26, 32)
EDGE = QColor(58, 58, 68)
TXT = QColor(150, 150, 160)
TXT_HI = QColor(225, 225, 235)
A_COL = QColor(90, 150, 220)         # outgoing deck
B_COL = QColor(224, 170, 70)         # incoming deck
EQ_LOW = QColor(232, 105, 105)
EQ_MID = QColor(150, 200, 110)
EQ_HIGH = QColor(110, 195, 232)
STEM = QColor(205, 130, 235)
SEAM = QColor(90, 200, 145)
NO_RET = QColor(230, 90, 90)
ARM = QColor(250, 200, 90)
HEAD = QColor(248, 248, 255)
FLAM = QColor(235, 120, 90)
BEAT = QColor(96, 96, 110)
BAR = QColor(150, 150, 168)


def _mmss(s):
    s = max(int(s), 0)
    return f"{s // 60}:{s % 60:02d}"


class _Ramp:
    """One automated scalar: set(t, target, ramp_s) then at(t)."""

    __slots__ = ("v0", "v1", "t0", "dur")

    def __init__(self, v):
        self.v0 = self.v1 = float(v)
        self.t0, self.dur = -1e9, 0.0

    def set(self, t, target, ramp_s):
        self.v0 = self.at(t)
        self.v1 = float(target)
        self.t0 = t
        self.dur = max(float(ramp_s or 0.0), 0.0)

    def at(self, t):
        if self.dur <= 0.0 or t >= self.t0 + self.dur:
            return self.v1
        if t <= self.t0:
            return self.v0
        return self.v0 + (self.v1 - self.v0) * (t - self.t0) / self.dur


class _Deck:
    def __init__(self, pos, playing):
        self.gain = _Ramp(1.0)
        self.eq = {"low": _Ramp(1.0), "mid": _Ramp(1.0), "high": _Ramp(1.0)}
        self.rate = _Ramp(1.0)
        self.stems = {}                  # name -> _Ramp
        self.pos = float(pos)
        self.playing = bool(playing)
        self.loop = None                 # (start_s, end_s)


def _grid_seg(track, pos):
    for seg in (getattr(track, "grid", None) or []):
        if seg["start_s"] <= pos <= seg["end_s"]:
            return seg
    g = getattr(track, "grid", None) or []
    return g[0] if g else None


def _beat_index(track, pos):
    """(beat index, is_downbeat) at a source position, or (None, False)."""
    seg = _grid_seg(track, pos)
    if seg is None or not seg.get("period_s"):
        return None, False
    k = math.floor((pos - seg["first_beat_s"]) / seg["period_s"])
    off = getattr(track, "downbeat_offset", 0) or 0
    return k, ((k - off) % 4 == 0)


def _beat_phase(track, pos):
    """Position within the beat, 0..1, or None."""
    seg = _grid_seg(track, pos)
    if seg is None or not seg.get("period_s"):
        return None
    return ((pos - seg["first_beat_s"]) / seg["period_s"]) % 1.0


def simulate(info, a, b, t_lo, t_hi, n):
    """Replay the event script over n columns spanning [t_lo, t_hi].

    Returns {"t": [...], "a"/"b": {"gain", "low", "mid", "high", "pos",
    "playing", "stems"}, "marks": [(t, label, deck)], "sync_offset_ms"}.

    `t_lo` must be render second 0 - deck positions are integrated from
    the renderer's cue points forward, not wound in from an earlier time.
    """
    t0c = info["t0_clock"]
    evs = sorted(info.get("events") or [], key=lambda e: e.get("at", 0))
    for e in evs:
        e["_t"] = (e.get("at", 0) - t0c) / RATE
    decks = {"a": _Deck(info["cue_a"], True),
             "b": _Deck((info.get("plan") or {}).get("in_s", 0.0), False)}
    tracks = {"a": a, "b": b}
    out = {k: {"gain": [], "low": [], "mid": [], "high": [], "pos": [],
               "playing": [], "stems": {}} for k in decks}
    marks, sync_ms = [], None
    dt = (t_hi - t_lo) / max(n - 1, 1)
    ei = 0
    # Wind the script forward to t_lo first so the state at the left edge
    # is right even when events fired before it.
    ts = [t_lo + i * dt for i in range(n)]
    prev_t = t_lo                        # t_lo is render second 0
    for i, t in enumerate(ts):
        while ei < len(evs) and evs[ei]["_t"] <= t:
            e = evs[ei]
            ei += 1
            te = e["_t"]
            d = decks.get(e.get("deck", ""))
            cmd = e.get("cmd")
            if cmd in ("gain", "eq", "rate", "stem_gains", "cue", "jump",
                       "loop", "clear_loop", "release_loop", "brake",
                       "start", "stop", "unload") and d is None:
                continue
            if cmd == "gain":
                d.gain.set(te, e["value"], e.get("ramp_s", 0.05))
            elif cmd == "eq":
                for band in ("low", "mid", "high"):
                    if e.get(band) is not None:
                        d.eq[band].set(te, e[band], e.get("ramp_s", 0.05))
            elif cmd == "rate":
                d.rate.set(te, e["value"], e.get("ramp_s", 0.0))
            elif cmd == "stem_gains":
                for name, v in (e.get("gains") or {}).items():
                    r = d.stems.get(name)
                    if r is None:
                        r = d.stems[name] = _Ramp(1.0)
                    r.set(te, v, e.get("ramp_s", 0.05))
            elif cmd == "start":
                d.playing = True
                marks.append((te, "start", e.get("deck")))
            elif cmd in ("stop", "unload"):
                d.playing = False
                marks.append((te, "stop", e.get("deck")))
            elif cmd == "cue":
                d.pos = float(e["time_s"])
                marks.append((te, "cue", e.get("deck")))
            elif cmd == "jump":
                d.pos = float(e["time_s"])
                marks.append((te, "jump", e.get("deck")))
            elif cmd == "loop":
                d.loop = (float(e["start_s"]), float(e["end_s"]))
                marks.append((te, "loop", e.get("deck")))
            elif cmd in ("clear_loop", "release_loop"):
                d.loop = None
                marks.append((te, "loop off", e.get("deck")))
            elif cmd == "brake":
                d.rate.set(te, 0.0, e.get("duration_s", 1.5))
                marks.append((te, "brake", e.get("deck")))
            elif cmd == "sync":
                # phase_snap: the incoming deck is shifted onto the
                # master's beat phase at gain ~0 (see submix).
                sl, ma = decks.get(e["slave"]), decks.get(e["master"])
                ts_, tm = tracks.get(e["slave"]), tracks.get(e["master"])
                pa, pb = (_beat_phase(tm, ma.pos) if ma else None,
                          _beat_phase(ts_, sl.pos) if sl else None)
                if pa is not None and pb is not None:
                    dphase = (pa - pb + 0.5) % 1.0 - 0.5
                    seg = _grid_seg(ts_, sl.pos)
                    if seg:
                        sync_ms = abs(dphase) * seg["period_s"] * 1000.0
                        sl.pos += dphase * seg["period_s"]
                marks.append((te, "sync", e.get("slave")))
            elif cmd in ("echo", "filter", "fx_play", "duck", "end_sync"):
                lbl = {"end_sync": "sync off"}.get(cmd, cmd)
                if cmd == "filter" and e.get("mode") in (None, "off"):
                    lbl = "filter off"
                marks.append((te, lbl, e.get("deck")))
        # advance positions to t
        for k, d in decks.items():
            step = t - prev_t
            if d.playing and step > 0:
                d.pos += d.rate.at(t) * step
                if d.loop and d.pos >= d.loop[1]:
                    span = max(d.loop[1] - d.loop[0], 1e-6)
                    d.pos = d.loop[0] + (d.pos - d.loop[1]) % span
            o = out[k]
            o["gain"].append(d.gain.at(t) if d.playing else 0.0)
            for band in ("low", "mid", "high"):
                o[band].append(d.eq[band].at(t))
            o["pos"].append(d.pos)
            o["playing"].append(d.playing)
            for name, r in d.stems.items():
                o["stems"].setdefault(name, [1.0] * i).append(r.at(t))
        prev_t = t
    for k in out:                       # pad stems discovered mid-run
        for name, vals in out[k]["stems"].items():
            while len(vals) < n:
                vals.append(vals[-1] if vals else 1.0)
    out["t"] = ts
    out["marks"] = marks
    out["sync_offset_ms"] = sync_ms
    return out


class SeamScope(QWidget):
    """set_seam(a, b, info, after_s, render_s) / clear() /
    set_playhead(render_seconds or None). seekRequested(render_s)."""

    seekRequested = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(238)
        self.a = self.b = None
        self.info = None
        self.plan = {}
        self.after_s = 0.0
        self.render_s = 0.0
        self.play_t = None
        self._sim = None
        self._sim_key = None
        self._seam_id = 0
        self._lo = self._hi = 0.0
        self._px0 = self._sc = 0.0
        self._drag = False

    def clear(self):
        self.a = self.b = self.info = self._sim = self._sim_key = None
        self.plan, self.play_t = {}, None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_seam(self, a, b, info, after_s, render_s):
        self.a, self.b = a, b
        self.info = info if (info or {}).get("events") is not None else None
        self.plan = (info or {}).get("plan") or {}
        self.after_s = float(after_s or 0.0)
        self.render_s = float(render_s or 0.0)
        self.play_t = None
        self._sim = self._sim_key = None
        self._seam_id += 1
        if self.info is not None:
            t0c = info["t0_clock"]
            self._t_blend = (info["blend_at"] - t0c) / RATE
            self._t_swap = (info["swap_at"] - t0c) / RATE
            # A is cued at cue_a and runs at rate 1.0 into the seam, so
            # its exit point lands here in render time.
            self._t_out = self.plan.get("out_s", 0.0) - info["cue_a"]
            nr = self.plan.get("no_return_at")
            self._t_nr = None if nr is None else (nr - t0c) / RATE
            # Zoom: the whole rendered seam plus a short tail past the
            # swap - the two songs' full lengths are not the subject.
            self._lo = 0.0
            self._hi = max(min(self.render_s, self._t_swap + 10.0),
                           self._t_swap + 1.0)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()

    def set_playhead(self, t):
        t = None if t is None else round(float(t), 2)
        if t != self.play_t:
            self.play_t = t
            self.update()

    def window_end(self):
        """Render second where the drawn analysis region ends, or None.
        Past this the render is just the incoming track playing on - there
        is nothing left to look at, so playback stops here."""
        if self.info is None:
            return None
        return min(self._hi, self.render_s) if self.render_s else self._hi

    # -- geometry ----------------------------------------------------------
    def _x(self, t):
        return self._px0 + (t - self._lo) * self._sc

    def _t_at(self, x):
        if self._sc <= 0:
            return None
        return self._lo + (x - self._px0) / self._sc

    def _ensure_sim(self, n):
        key = (self._seam_id, n, round(self._lo, 3), round(self._hi, 3))
        if self._sim_key != key:
            self._sim = simulate(self.info, self.a, self.b,
                                 self._lo, self._hi, n)
            self._sim_key = key
        return self._sim

    # -- interaction -------------------------------------------------------
    def mousePressEvent(self, ev):
        if self.info is None:
            return
        t = self._t_at(ev.position().x())
        if t is not None:
            self._drag = True
            self.seekRequested.emit(
                min(max(t, 0.0), max(self.render_s - 0.05, 0.0)))

    def mouseMoveEvent(self, ev):
        if self._drag:
            t = self._t_at(ev.position().x())
            if t is not None:
                self.seekRequested.emit(
                    min(max(t, 0.0), max(self.render_s - 0.05, 0.0)))

    def mouseReleaseEvent(self, _ev):
        self._drag = False

    def event(self, ev):
        if ev.type() == QEvent.Type.ToolTip and self._sim is not None:
            t = self._t_at(ev.pos().x())
            if t is None:
                QToolTip.hideText()
                return True
            sim = self._sim
            i = max(0, min(int((t - self._lo) / max(
                self._hi - self._lo, 1e-9) * (len(sim["t"]) - 1)),
                len(sim["t"]) - 1))
            lines = [f"{t - self._t_out:+.1f}s from the seam "
                     f"(render {t:.1f}s)"]
            for k, track in (("a", self.a), ("b", self.b)):
                d = sim[k]
                if not d["playing"][i]:
                    lines.append(f"{track.title[:30]}: silent")
                    continue
                kb, _ = _beat_index(track, d["pos"][i])
                lines.append(
                    f"{track.title[:30]}: {_mmss(d['pos'][i])} · "
                    f"gain {d['gain'][i]:.2f} · eq "
                    f"{d['low'][i]:.2f}/{d['mid'][i]:.2f}/{d['high'][i]:.2f}"
                    + (f" · beat {kb}" if kb is not None else ""))
            lines.append("click to seek the audition here")
            QToolTip.showText(ev.globalPos(), "\n".join(lines), self)
            return True
        return super().event(ev)

    # -- painting helpers --------------------------------------------------
    def _curve(self, p, rect, vals, color, width=1.5, dash=None, fill=False,
               vmax=1.0, mask=None):
        """One automated curve. `mask` (per-column bools) breaks the line
        where the deck is silent - a deck's EQ while it is not playing is
        state, not sound, and drawing it flat across reads as a signal."""
        n = len(vals)
        if n < 2:
            return
        pts = [QPointF(self._px0 + i * (rect.width() / (n - 1)),
                       rect.bottom() - max(float(v), 0.0) / vmax
                       * (rect.height() - 2))
               for i, v in enumerate(vals)]
        if fill:
            poly = QPolygonF([QPointF(pts[0].x(), rect.bottom())] + pts
                             + [QPointF(pts[-1].x(), rect.bottom())])
            c = QColor(color)
            c.setAlpha(58)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawPolygon(poly)
        pen = QPen(color, width)
        if dash:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern(dash)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        runs, cur = [], []
        for i, pt in enumerate(pts):
            if mask is None or mask[i]:
                cur.append(pt)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        for run in runs:
            if len(run) >= 2:
                p.drawPolyline(QPolygonF(run))

    def _beat_row(self, p, track, positions, y, h, up, plot_w, label_ref):
        """Beat ticks for one deck. Returns [(x, beat_k)] of downbeats."""
        downs = []
        n = len(positions)
        if n < 2:
            return downs
        prev_k, _ = _beat_index(track, positions[0])
        seg = _grid_seg(track, positions[0])
        per_beat_px = ((seg["period_s"] * self._sc) if seg else 99)
        for i in range(1, n):
            k, is_down = _beat_index(track, positions[i])
            if k is None or k == prev_k:
                continue
            prev_k = k
            if not is_down and per_beat_px < 3.0:
                continue
            x = self._px0 + i * (plot_w / (n - 1))
            if is_down:
                downs.append((x, k))
                p.setPen(QPen(BAR, 1.4))
                p.drawLine(int(x), int(y), int(x), int(y + h))
            else:
                p.setPen(QPen(BEAT, 1))
                p.drawLine(int(x), int(y + (0 if up else h * 0.45)),
                           int(x), int(y + (h * 0.55 if up else h)))
        # Bar numbers relative to the reference downbeat.
        if label_ref is not None and downs:
            f = p.font()
            f.setPointSizeF(max(f.pointSizeF() - 2.0, 6.0))
            p.setFont(f)
            spacing = (downs[1][0] - downs[0][0]) if len(downs) > 1 else 99
            every = (1 if spacing >= 44 else 2 if spacing >= 24
                     else 4 if spacing >= 12 else 8)
            p.setPen(TXT)
            for x, k in downs:
                bar = round((k - label_ref) / 4.0)
                if bar % every:
                    continue
                p.drawText(QRectF(x + 2, y + (0 if up else h - 11), 30, 11),
                           Qt.AlignmentFlag.AlignLeft, f"{bar:+d}")
        return downs

    # -- paint -------------------------------------------------------------
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, BG)
        if self.info is None or self.a is None or self.b is None:
            p.setPen(TXT)
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter,
                       "seam scope - rate a seam to see its mechanics")
            p.end()
            return
        self._px0 = 34.0
        plot_w = w - self._px0 - 8
        self._sc = plot_w / max(self._hi - self._lo, 1e-6)
        n = max(int(plot_w), 32)
        sim = self._ensure_sim(n)

        # Rows grow with the widget - the curves are the point, so give
        # them whatever height the tab spares.
        hdr, rail_h, axis_h = 15.0, 16.0, 16.0
        avail = max(h - hdr - 6 - rail_h - axis_h, 70.0)
        ruler_h = min(max(avail * 0.10, 14.0), 26.0)
        lane_h = max((avail - 2 * ruler_h) / 2.0, 26.0)
        y_a = hdr + 2
        y_ar = y_a + lane_h
        y_br = y_ar + ruler_h
        y_b = y_br + ruler_h
        y_rail = y_b + lane_h + 2
        y_axis = y_rail + rail_h

        a_rect = QRectF(self._px0, y_a, plot_w, lane_h)
        b_rect = QRectF(self._px0, y_b, plot_w, lane_h)
        for r in (a_rect, b_rect):
            p.fillRect(r, PANEL)
            p.setPen(QPen(EDGE, 1))
            p.drawRect(r)
        # A quiet intro is ridden up to +3 dB (the entry trim), so unity is
        # not always the ceiling - scale to what the script actually asks
        # for, or the boost would be drawn as a flat top.
        gmax = 1.0
        for key in ("a", "b"):
            if sim[key]["gain"]:
                gmax = max(gmax, max(sim[key]["gain"]))
        gmax = min(gmax, 2.0)
        for r in (a_rect, b_rect):
            for frac in (0.5, 1.0):
                yy = r.bottom() - (frac / gmax) * (r.height() - 2)
                p.setPen(QPen(QColor(70, 70, 82), 1,
                              Qt.PenStyle.DashLine if frac == 1.0
                              else Qt.PenStyle.DotLine))
                p.drawLine(int(r.x()), int(yy), int(r.right()), int(yy))

        # -- overlap band + blend / seam / A-out / no-return -----------------
        # The band is the whole two-deck overlap: staged blends START a
        # blend-length BEFORE the exit point, so anchoring it at out_s
        # drew only the tail.
        bx0, bx1 = self._x(self._t_blend), self._x(self._t_swap)
        band = QColor(SEAM)
        band.setAlpha(26)
        p.fillRect(QRectF(bx0, y_a, max(bx1 - bx0, 2.0),
                          y_b + lane_h - y_a), band)
        # The seam itself is named on the time axis; the rest get text
        # where there is room for it, inside the lanes. Every lane label
        # records its rect in `placed` so later text (the stem names)
        # dodges it instead of printing on top.
        placed = []
        for tv, col, lbl, ly in ((self._t_blend, SEAM, "blend", y_a + 2),
                                 (self._t_out, SEAM, "", 0.0),
                                 (self._t_swap, SEAM, "A out", y_b + 2),
                                 (self._t_nr, NO_RET, "no return",
                                  y_b + lane_h - 14)):
            if tv is None or not (self._lo <= tv <= self._hi):
                continue
            x = self._x(tv)
            p.setPen(QPen(col, 1, Qt.PenStyle.DashLine))
            p.drawLine(int(x), int(y_a - 2), int(x), int(y_b + lane_h + 2))
            if lbl:
                p.setPen(col)
                r = QRectF(x + 3, ly,
                           p.fontMetrics().horizontalAdvance(lbl) + 2, 12)
                p.drawText(r, Qt.AlignmentFlag.AlignLeft, lbl)
                placed.append(r)

        # -- automation lanes ------------------------------------------------
        for key, rect, col, name in (("a", a_rect, A_COL, "A/OUT"),
                                     ("b", b_rect, B_COL, "B/IN")):
            d = sim[key]
            live = d["playing"]
            self._curve(p, rect, d["gain"], col, 2.0, fill=True, vmax=gmax)
            self._curve(p, rect, d["low"], EQ_LOW, 1.2, [5, 3], vmax=gmax,
                        mask=live)
            self._curve(p, rect, d["mid"], EQ_MID, 1.2, [2, 3], vmax=gmax,
                        mask=live)
            self._curve(p, rect, d["high"], EQ_HIGH, 1.2, [1, 3], vmax=gmax,
                        mask=live)
            stem_lbls = []
            for sname, vals in sorted(d["stems"].items()):
                if max(vals) - min(vals) < 0.02 and abs(vals[0] - 1.0) < 0.02:
                    continue
                self._curve(p, rect, vals, STEM, 1.4, [4, 2], vmax=gmax,
                            mask=live)
                j = min(range(len(vals)), key=lambda q: vals[q])
                stem_lbls.append((self._px0 + j * (rect.width()
                                                   / max(len(vals) - 1, 1))
                                  + 2, sname))
            # A stem style ducks several stems at the same beat, so their
            # minima - where the names are anchored - usually coincide.
            # Stack colliding names upward instead of printing them on top
            # of one another, and dodge the seam labels ('no return' lives
            # in the same bottom row; lab-reported: unreadable overlap).
            p.setPen(STEM)
            fm = p.fontMetrics()
            for x_l, sname in sorted(stem_lbls):
                r = QRectF(x_l, rect.bottom() - 14,
                           fm.horizontalAdvance(sname) + 2, 12)
                for _ in range(max(int(lane_h // 12) - 1, 0)):
                    if not any(r.intersects(q) for q in placed):
                        break
                    r.translate(0, -12)
                p.drawText(r, Qt.AlignmentFlag.AlignLeft
                           | Qt.AlignmentFlag.AlignBottom, sname)
                placed.append(r)
            p.setPen(TXT_HI)
            f = p.font()
            f.setPointSizeF(max(f.pointSizeF() - 1.5, 6.5))
            p.setFont(f)
            p.save()
            p.rotate(-90)
            p.drawText(int(-(rect.y() + lane_h - 3)), int(self._px0 - 22),
                       name)
            p.restore()
            p.setFont(self.font())

        # -- beat rulers (they meet in the middle) ---------------------------
        k_seam_a, _ = _beat_index(self.a, self.plan.get("out_s", 0.0))
        k_in_b, _ = _beat_index(self.b, self.plan.get("in_s", 0.0))
        off_a = getattr(self.a, "downbeat_offset", 0) or 0
        off_b = getattr(self.b, "downbeat_offset", 0) or 0
        ref_a = (None if k_seam_a is None
                 else k_seam_a - ((k_seam_a - off_a) % 4))
        ref_b = (None if k_in_b is None
                 else k_in_b - ((k_in_b - off_b) % 4))
        p.fillRect(QRectF(self._px0, y_ar, plot_w, ruler_h * 2),
                   QColor(22, 22, 28))
        downs_a = self._beat_row(p, self.a, sim["a"]["pos"], y_ar, ruler_h,
                                 True, plot_w, ref_a)
        downs_b = self._beat_row(p, self.b, sim["b"]["pos"], y_br, ruler_h,
                                 False, plot_w, ref_b)
        p.setPen(QPen(EDGE, 1))
        p.drawLine(int(self._px0), int(y_br), int(self._px0 + plot_w),
                   int(y_br))
        # Downbeat pairs that land in the audible flam window, during the
        # overlap only - the same 15-80 ms story the seam inspector tells.
        period_ms = (self.a.period_s * 1000.0) if self.a.bpm else 500.0
        b_play = sim["b"]["playing"]
        flams = 0
        if downs_a and downs_b:
            for xa, _k in downs_a:
                if not (bx0 - 1 <= xa <= bx1 + 1):
                    continue
                xb, _ = min(downs_b, key=lambda q: abs(q[0] - xa))
                dms = abs(xb - xa) / max(self._sc, 1e-6) * 1000.0
                if 15.0 <= dms <= 80.0:
                    flams += 1
                    p.setPen(QPen(FLAM, 2))
                    p.drawLine(int(min(xa, xb)), int(y_br),
                               int(max(xa, xb)), int(y_br))

        # -- marker rail ------------------------------------------------------
        p.fillRect(QRectF(self._px0, y_rail, plot_w, rail_h), PANEL)
        f = p.font()
        f.setPointSizeF(max(f.pointSizeF() - 2.0, 6.0))
        p.setFont(f)
        # One sorted pass over marks + the arm point: every tick is drawn,
        # but a label only prints where the PREVIOUS drawn label actually
        # ended. (The old check measured the current label's width against
        # the previous tick's x, so a short label after a long one - 'sync
        # B' after 'filter off B' - printed on top of it, and 'armed' was
        # drawn with no check at all.)
        rail = [(self._x(t), f"{lbl}" + (f" {deck.upper()}" if deck else ""),
                 A_COL if deck == "a" else B_COL if deck == "b" else TXT)
                for t, lbl, deck in sim["marks"]
                if self._lo <= t <= self._hi]
        t_arm = self.after_s - self.info["cue_a"]
        if self._lo <= t_arm <= self._hi:
            rail.append((self._x(t_arm), "armed", ARM))
        last_end = -1e9
        for x, text, col in sorted(rail, key=lambda q: q[0]):
            p.setPen(QPen(col, 1))
            p.drawLine(int(x), int(y_rail), int(x), int(y_rail + rail_h))
            if x + 2 >= last_end + 6:
                p.setPen(col)
                p.drawText(QRectF(x + 2, y_rail + 1, 120, rail_h - 2),
                           Qt.AlignmentFlag.AlignLeft, text)
                last_end = x + 2 + p.fontMetrics().horizontalAdvance(text)

        # -- time axis (seconds from the seam) --------------------------------
        span = self._hi - self._lo
        step = next(s for s in (1, 2, 5, 10, 15, 30, 60)
                    if span / s <= plot_w / 52.0)
        p.setPen(TXT)
        k0 = math.ceil((self._lo - self._t_out) / step)
        while True:
            rel = (k0) * step
            t = self._t_out + rel
            if t > self._hi:
                break
            x = self._x(t)
            p.setPen(QPen(QColor(70, 70, 82), 1))
            p.drawLine(int(x), int(y_axis), int(x), int(y_axis + 3))
            p.setPen(TXT)
            p.drawText(QRectF(x - 24, y_axis + 3, 48, 12),
                       Qt.AlignmentFlag.AlignHCenter,
                       "seam" if abs(rel) < 1e-6 else f"{rel:+.0f}s")
            k0 += 1

        # -- header numbers ----------------------------------------------------
        p.setFont(self.font())
        beats = self.plan.get("beats") or 0
        rate = self.plan.get("rate", 1.0)
        bits = [f"{self.plan.get('style', '?')}",
                f"{beats} beats ({beats / 4:.0f} bars, "
                f"{self._t_swap - self._t_out:.1f}s)",
                f"B {(rate - 1) * 100:+.1f}%"]
        sms = sim.get("sync_offset_ms")
        if sms is not None:
            bits.append(f"grid off {sms:.0f} ms at sync"
                        + (" (snapped)" if sms > 1 else ""))
        if flams:
            bits.append(f"{flams} downbeat flam{'s' if flams > 1 else ''}")
        if gmax > 1.02:
            bits.append(f"entry trim +{20 * math.log10(gmax):.1f} dB")
        if not any(b_play):
            bits.append("B never starts")
        # Legend first (right-aligned), so the header can be elided to
        # whatever room is left instead of running underneath it.
        lx = w - 8
        for col, name in ((EQ_HIGH, "high"), (EQ_MID, "mid"),
                          (EQ_LOW, "low"), (TXT_HI, "gain")):
            tw = p.fontMetrics().horizontalAdvance(name)
            lx -= tw
            p.setPen(col)
            p.drawText(int(lx), int(hdr - 3), name)
            lx -= 14
        p.setPen(TXT_HI)
        hw = max(lx - 10 - self._px0, 60.0)
        p.drawText(QRectF(self._px0, 0, hw, hdr),
                   Qt.AlignmentFlag.AlignLeft,
                   p.fontMetrics().elidedText("   ".join(bits),
                                              Qt.TextElideMode.ElideRight,
                                              int(hw)))
        # -- playhead: one line through everything ----------------------------
        if self.play_t is not None:
            t = self.play_t
            clamped = t > self._hi
            x = self._x(min(t, self._hi))
            c = QColor(HEAD)
            if clamped:
                c.setAlpha(90)
            p.setPen(QPen(c, 2))
            p.drawLine(int(x), int(y_a - 6), int(x), int(y_rail + rail_h))
            if not clamped:
                p.setBrush(HEAD)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(QPolygonF([QPointF(x - 4, y_a - 6),
                                         QPointF(x + 4, y_a - 6),
                                         QPointF(x, y_a - 1)]))
        p.end()
