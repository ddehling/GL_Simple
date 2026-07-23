"""Arc strip: the set's SHAPE at a glance while you build it.

One compact, time-true panel above the entry list: per-slot energy bars
against the theme's arc target (the dashed curve the night is supposed to
follow), the BPM path, anchor flags, and a colored seam marker at every
boundary - green = clean beat-matched seam, orange = warned, red = fade.
Clicking a bar selects that entry in the set list, so a dip you can SEE
becomes a slot you can fix.
"""
from PyQt6.QtCore import Qt, QEvent, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QToolTip, QWidget

BG = QColor(20, 20, 25)
BAR = QColor(45, 75, 105)
BAR_ANCHOR = QColor(80, 120, 160)
ARC = QColor(230, 230, 240, 150)
BPM = QColor(250, 200, 90)
SEAM_OK = QColor(70, 170, 120)
SEAM_WARN = QColor(220, 150, 40)
SEAM_FADE = QColor(190, 85, 85)
TXT = QColor(150, 150, 160)


class ArcStrip(QWidget):
    slotClicked = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(92)
        self.slots = []      # [{off, play, energy, bpm, anchor, seam, warn,
                             #   title?, chips?, warnings?}]
        self.arc = []        # [(time_frac 0..1, target 0..1)]
        self.target_s = 0.0  # intended set length; 0 = scale to content
        self._hit = []       # [(x0, x1, index)] from the last paint
        self._seam_hit = []  # [(x, index)] seam tick centers, last paint

    def set_data(self, slots, arc, target_s=0.0):
        self.slots = list(slots or [])
        self.arc = list(arc or [])
        self.target_s = float(target_s or 0.0)
        self.update()

    def mousePressEvent(self, ev):
        x = ev.position().x()
        for x0, x1, i in self._hit:
            if x0 <= x <= x1:
                self.slotClicked.emit(i)
                return

    def event(self, ev):
        # Hover tooltips: near a seam tick -> that seam's chips/warnings
        # (the weakest-term words, same vocabulary as the plan list); over
        # a bar -> the track. The whole set's rhythm health is scannable
        # without reading the compiled text.
        if ev.type() == QEvent.Type.ToolTip:
            x = ev.pos().x()
            for sx, i in self._seam_hit:
                if abs(x - sx) <= 5:
                    s = self.slots[i]
                    # Full seam explainer when the planner provided one
                    # (verdict + judged factors); chips fallback otherwise.
                    tip = s.get("tip")
                    if not tip:
                        lines = [f"seam: {s.get('title', '?')} → " +
                                 self.slots[i + 1].get("title", "?")]
                        lines += s.get("chips") or []
                        lines += s.get("warnings") or []
                        if len(lines) == 1:
                            lines.append("clean")
                        tip = "\n".join(lines)
                    QToolTip.showText(ev.globalPos(), tip, self)
                    return True
            for x0, x1, i in self._hit:
                if x0 <= x <= x1:
                    s = self.slots[i]
                    QToolTip.showText(
                        ev.globalPos(),
                        f"{s.get('title', '')}\n{s['bpm']:.0f} bpm,"
                        f" energy {s['energy']:.2f}", self)
                    return True
            QToolTip.hideText()
            return True
        return super().event(ev)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, BG)
        self._hit = []
        self._seam_hit = []
        if not self.slots:
            p.setPen(TXT)
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter,
                       "arc strip - add tracks or Suggest a set")
            p.end()
            return
        pad = 8
        top, bot = 16.0, h - 16.0
        span = bot - top
        content = max(self.slots[-1]["off"] + self.slots[-1]["play"], 1.0)
        # The x-axis is the INTENDED night, not the songs so far: a
        # half-built set occupies the left of the target timeline and the
        # theme curve spans the whole thing, so the shape you see is the
        # shape the arc anchoring actually scores against. Content longer
        # than the target extends the axis honestly.
        total = max(content, self.target_s or 0.0)

        def tx(t_s):
            return pad + (t_s / total) * (w - 2 * pad)

        # Unfilled remainder of the target: shaded, with an end-of-content
        # tick so the build edge is obvious.
        if total > content + 1.0:
            xc = tx(content)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 255, 255, 10))
            p.drawRect(QRectF(xc, top, (w - pad) - xc, bot - top))
            p.setPen(QPen(QColor(200, 200, 210, 90), 1,
                          Qt.PenStyle.DotLine))
            p.drawLine(int(xc), int(top), int(xc), int(bot))
            cm = int(content // 60)
            p.setPen(TXT)
            p.drawText(int(xc) + 3, int(top) + 10, f"{cm}:00")

        # Theme arc target: the shape the night is SUPPOSED to follow.
        if len(self.arc) >= 2:
            pen = QPen(ARC, 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            pts = [(pad + f * (w - 2 * pad), bot - t * span)
                   for f, t in self.arc]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))

        # Energy bars per slot (time-true widths) + BPM path.
        bpms = [s["bpm"] for s in self.slots if s["bpm"]]
        blo, bhi = (min(bpms), max(bpms)) if bpms else (0, 1)
        brange = max(bhi - blo, 1.0)
        bpm_pts = []
        for i, s in enumerate(self.slots):
            x0, x1 = tx(s["off"]), tx(s["off"] + s["play"])
            self._hit.append((x0, x1, i))
            e = max(0.05, min(float(s["energy"]), 1.0))
            r = QRectF(x0 + 1, bot - e * span, max(x1 - x0 - 2, 2), e * span)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(BAR_ANCHOR if s["anchor"] else BAR)
            p.drawRect(r)
            if s["anchor"]:
                p.setPen(QPen(QColor(240, 240, 250), 1))
                p.drawText(int(x0 + 2), int(top - 3), "⚓")
            if s["bpm"]:
                bpm_pts.append(((x0 + x1) / 2,
                                bot - ((s["bpm"] - blo) / brange) * span))
            seam = s.get("seam")
            if i < len(self.slots) - 1:
                c = (SEAM_FADE if (seam or {}).get("fade")
                     else SEAM_WARN if s.get("warn") else SEAM_OK)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c)
                p.drawRect(QRectF(x1 - 2.5, top - 8, 5, 6))
                self._seam_hit.append((x1, i))
        p.setPen(QPen(BPM, 1))
        for a, b in zip(bpm_pts, bpm_pts[1:]):
            p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        p.setPen(TXT)
        p.drawText(2, h - 3, "0:00")
        mins = int(total // 60)
        p.drawText(w - 46, h - 3, f"{mins // 60}:{mins % 60:02d}:00"
                   if mins >= 60 else f"{mins}:00")
        if bpms:
            p.setPen(QPen(BPM))
            p.drawText(2, int(top) + 2, f"{bhi:.0f}")
            p.drawText(2, int(bot), f"{blo:.0f}")
        p.end()
