"""Timeline tab: the song as a strip - what has played (left of the
cursor), what is already composed and waiting in the rack (right of it,
solid), and what the form knows beyond that (hatched: the rest of the
current section, the drop it is counting down to, the most likely next
sections and the energy arc).

Rows, top to bottom:
  sections   coloured blocks per phrase, section name, key changes
  energy     the phrase energy curve, and the arc beyond the composed edge
  chords     chord labels per bar (when there is room)
  marks      drops (flag), theme statements (T), automation lane moves
  below      a table of the phrases around the cursor

Data: state["timeline"] from GenSystem.timeline() (also over the remote
backend, since it rides in /api/gen/status)."""
from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QLabel, QHBoxLayout, QSlider

SECTION_COLOURS = {
    "intro": "#4a5a7a", "groove": "#3f7a5a", "build": "#a8772a", "drop": "#b03a3a", "break": "#5a4a8a",
    "outro": "#4a4a4a", "flow": "#3f7a7a", "swell": "#a85a2a", "calm": "#4a6a8a",
}


class TimelineStrip(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(170)
        self.state = {}
        self.window_s = 300.0        # seconds shown (past + future)
        self.past_frac = 0.35        # share of the window left of the cursor

    def set_window(self, seconds):
        self.window_s = float(seconds)
        self.update()

    def refresh(self, state):
        self.state = state or {}
        self.update()

    def _x(self, t, now, w):
        left = now - self.window_s * self.past_frac
        return (t - left) / self.window_s * w

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QColor("#16181d"))
        tl = (self.state or {}).get("timeline") or {}
        now = float(tl.get("now_s", 0.0))
        phrases = tl.get("phrases") or []
        hz = tl.get("horizon") or {}
        row_sec = QRectF(0, 8, w, 34)
        row_en = QRectF(0, 50, w, 52)
        row_ch = QRectF(0, 108, w, 20)
        row_mk = QRectF(0, 130, w, 22)
        font = QFont(); font.setPointSize(8); qp.setFont(font)
        # time grid: a tick per minute
        qp.setPen(QPen(QColor("#2a2e36"), 1))
        left = now - self.window_s * self.past_frac
        t = int(left // 60) * 60
        while t < left + self.window_s:
            x = self._x(t, now, w)
            qp.drawLine(int(x), 0, int(x), h)
            qp.setPen(QPen(QColor("#5a606c"))); qp.drawText(int(x) + 3, h - 4, f"{int(t // 60)}:{int(t % 60):02d}")
            qp.setPen(QPen(QColor("#2a2e36"), 1))
            t += 60
        # known future beyond the composed edge: hatched section-to-end, then likely next
        comp_to = float(hz.get("composed_to_s", now))
        sec_end = float(hz.get("section_end_s", comp_to))
        if sec_end > comp_to:
            x0, x1 = self._x(comp_to, now, w), self._x(sec_end, now, w)
            col = QColor(SECTION_COLOURS.get(hz.get("section", ""), "#555"))
            col.setAlpha(90)
            qp.setPen(Qt.PenStyle.NoPen); qp.setBrush(QBrush(col, Qt.BrushStyle.BDiagPattern))
            qp.drawRect(QRectF(x0, row_sec.top(), max(1.0, x1 - x0), row_sec.height()))
            qp.setPen(QPen(QColor("#c8ccd4"))); qp.drawText(int(x0) + 4, int(row_sec.top()) + 14, f"{hz.get('section', '')} ({hz.get('bars_left', 0)} bars left)")
            # likely next sections, proportional to weight, as thin bands
            xs = x1
            for name, wgt in (hz.get("next") or [])[:3]:
                span = max(20.0, 90.0 * float(wgt))
                col = QColor(SECTION_COLOURS.get(name, "#555")); col.setAlpha(60)
                qp.setPen(Qt.PenStyle.NoPen); qp.setBrush(QBrush(col, Qt.BrushStyle.Dense6Pattern))
                qp.drawRect(QRectF(xs, row_sec.top() + 6, span, row_sec.height() - 12))
                qp.setPen(QPen(QColor("#9aa0ac"))); qp.drawText(int(xs) + 3, int(row_sec.top()) + 26, f"{name} {int(float(wgt) * 100)}%")
                xs += span + 2
        # phrases
        prev_key = None
        pts = []
        for p in phrases:
            x0, x1 = self._x(p["start_s"], now, w), self._x(p["end_s"], now, w)
            if x1 < 0 or x0 > w:
                pts.append((x0, x1, p["energy"]))
                continue
            col = QColor(SECTION_COLOURS.get(p["section"], "#555"))
            if not p.get("played") and p["end_s"] > now:
                col = col.lighter(115)
            if p.get("played"):
                col.setAlpha(150)
            qp.setPen(QPen(QColor("#0e1013"), 1)); qp.setBrush(QBrush(col))
            qp.drawRect(QRectF(x0, row_sec.top(), max(1.0, x1 - x0 - 1), row_sec.height()))
            if x1 - x0 > 46:
                qp.setPen(QPen(QColor("#f0f2f5"))); qp.drawText(int(x0) + 4, int(row_sec.top()) + 14, p["section"])
                qp.setPen(QPen(QColor("#c8ccd4"))); qp.drawText(int(x0) + 4, int(row_sec.top()) + 28, f"bar {p['bar0']}")
            if p.get("key") and p["key"] != prev_key and prev_key is not None:
                qp.setPen(QPen(QColor("#ffd166"), 2)); qp.drawLine(int(x0), int(row_sec.top()), int(x0), int(row_mk.bottom()))
                qp.drawText(int(x0) + 3, int(row_mk.bottom()) - 2, p["key"])
            prev_key = p.get("key")
            # chords per bar
            nb = max(1, int(p.get("nbars", 4)))
            bw = (x1 - x0) / nb
            if bw > 24:
                qp.setPen(QPen(QColor("#9aa0ac")))
                for i, ch in enumerate(p.get("chords") or []):
                    qp.drawText(int(x0 + i * bw) + 2, int(row_ch.bottom()) - 4, str(ch))
            # marks
            for d in p.get("drops") or []:
                xd = self._x(d, now, w)
                qp.setPen(QPen(QColor("#ff5c5c"), 2)); qp.drawLine(int(xd), int(row_sec.top()), int(xd), int(row_mk.bottom()))
                qp.drawText(int(xd) + 3, int(row_mk.top()) + 10, "DROP")
            if p.get("lead") in ("theme", "theme_make"):
                qp.setPen(QPen(QColor("#ffd166"))); qp.drawText(int(x0) + 3, int(row_mk.top()) + 20, "T" if p["lead"] == "theme" else "t")
            for a in (p.get("auto") or [])[:3]:
                qp.setPen(QPen(QColor("#6cc3ff"))); qp.drawText(int(x0) + 16, int(row_mk.top()) + 20, f"{a.get('lane')}")
            pts.append((x0, x1, p["energy"]))
        # energy curve
        qp.setPen(QPen(QColor("#7fd1a8"), 2))
        last = None
        for x0, x1, e in pts:
            y = row_en.bottom() - float(e) * row_en.height()
            if last is not None:
                qp.drawLine(int(last[0]), int(last[1]), int(x0), int(y))
            qp.drawLine(int(x0), int(y), int(x1), int(y))
            last = (x1, y)
        # arc beyond the composed edge
        arc = hz.get("arc") or []
        qp.setPen(QPen(QColor("#7fd1a8"), 1, Qt.PenStyle.DashLine))
        prev = None
        for t_s, e in arc:
            x = self._x(float(t_s), now, w)
            y = row_en.bottom() - float(e) * row_en.height()
            if prev is not None and x <= w:
                qp.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)
        # the drop the build counts down to
        if hz.get("drop_s"):
            xd = self._x(float(hz["drop_s"]), now, w)
            qp.setPen(QPen(QColor("#ff5c5c"), 1, Qt.PenStyle.DashLine)); qp.drawLine(int(xd), int(row_sec.top()), int(xd), int(row_mk.bottom()))
            qp.drawText(int(xd) + 3, int(row_mk.top()) + 10, f"drop in {max(0.0, float(hz['drop_s']) - now):.0f}s")
        # cursor
        xc = self._x(now, now, w)
        qp.setPen(QPen(QColor("#ffffff"), 2)); qp.drawLine(int(xc), 0, int(xc), h)
        qp.setPen(QPen(QColor("#c8ccd4"))); qp.drawText(int(xc) + 4, 12, f"now {int(now // 60)}:{int(now % 60):02d}")
        if hz.get("ending"):
            qp.setPen(QPen(QColor("#ff9f43"))); qp.drawText(w - 90, 12, "ENDING")
        if hz.get("hold"):
            qp.setPen(QPen(QColor("#ff9f43"))); qp.drawText(w - 160, 12, "HOLD")
        qp.end()


class TimelinePage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("window"))
        self.zoom = QSlider(Qt.Orientation.Horizontal); self.zoom.setRange(60, 1800); self.zoom.setValue(300)
        self.zoom.setMaximumWidth(220); top.addWidget(self.zoom)
        self.lbl = QLabel("5:00"); top.addWidget(self.lbl); top.addStretch(1)
        self.info = QLabel(""); top.addWidget(self.info)
        lay.addLayout(top)
        self.strip = TimelineStrip()
        lay.addWidget(self.strip)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["at", "bar", "section", "energy", "chords", "lead", "layers"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)
        self.zoom.valueChanged.connect(self._zoom)

    def _zoom(self, v):
        self.strip.set_window(v)
        self.lbl.setText(f"{v // 60}:{v % 60:02d}")

    def refresh(self, state):
        self.strip.refresh(state)
        tl = (state or {}).get("timeline") or {}
        hz = tl.get("horizon") or {}
        now = float(tl.get("now_s", 0.0))
        self.info.setText(f"{hz.get('section', '-')}  bars left {hz.get('bars_left', '-')}  composed to +{max(0.0, float(hz.get('composed_to_s', now)) - now):.0f}s"
                          f"  next {', '.join(f'{n} {int(float(w) * 100)}%' for n, w in (hz.get('next') or [])[:2])}"
                          f"  movement {hz.get('movement', '-')}")
        phrases = tl.get("phrases") or []
        self.table.setRowCount(len(phrases))
        cur_row = None
        for i, p in enumerate(phrases):
            vals = [f"{int(p['start_s'] // 60)}:{int(p['start_s'] % 60):02d}", str(p["bar0"]), p["section"], f"{p['energy']:.2f}",
                    " ".join(p.get("chords") or []), str(p.get("lead") or ""), ",".join(p.get("layers") or [])]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if p["start_s"] <= now < p["end_s"]:
                    it.setBackground(QColor("#2f3a4a")); cur_row = i
                elif p.get("played"):
                    it.setForeground(QColor("#8a909c"))
                self.table.setItem(i, j, it)
        if cur_row is not None:
            self.table.scrollToItem(self.table.item(cur_row, 0))


def register(console):
    page = TimelinePage()
    console.add_tab("Timeline", page)
