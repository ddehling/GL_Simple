"""Perform tab: the autonomous DJ, steered, and SEEN - laid out as a stage and a deck.

Two modes. AUTOMIX is the night: whole songs, one seam at a time, planned
and executed through the gates (lib/dj/system.py). REMIX (lib/dj/remix.py)
is the system playing PARTS of songs together: two or three songs live on
one clock, the four stem lanes each assigned to a song, a move every
phrase. In both, the system does the technical work and you steer.

THE STAGE (left): the lane board - which song plays each lane, in the
song's colour, with how long it has been there - the phrase bar counting
down to the next move (or the seam), and one strip per deck: sections by
kind, energy, playhead, loop, exit and entry. THE DECK (right): the big
performance buttons (MIX NOW, HOLD, DROP, BREAK, LOOP 4 / 8), the sliders
with their values (energy, blend, vocals, tempo; the automixer's mix type
and mix speed), GOOD / BAD, and the memory row (SAVE, RECALL, REC). THE
FEED (bottom): every move newest first, with a thumbs-up / thumbs-down on
the move itself, and the SAY line - words the Perform Copilot maps onto
the same controls.

SONGS: the "songs" box confines the music to one of your saved setlists
(the Set tab's lists) - in Remix the conductor picks only from it, in
Automix it is the night's pool; "whole library" lifts it.

Keys (when the tab has focus): Space MIX NOW · H HOLD · D DROP · B BREAK ·
4 / 8 LOOP · ← BAD · → GOOD · S SAVE · R RECALL. The nanoKONTROL2 drives
the same controls (see _midi_poll). Every timer/slot body is guarded: an
exception in a Qt slot kills the process silently otherwise.
"""
import threading
import time

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QPolygonF, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget, QToolButton, QSizePolicy)

SPEEDS = (("short", 0.5), ("normal", 1.0), ("long", 2.0), ("marathon", 3.0))
REMIX_BARS = {"short": 4, "normal": 8, "long": 16, "marathon": 32}      # bars between the conductor's moves
MODES = ("Automix", "Remix")
LANES = ("drums", "bass", "other", "vocals")
STYLE_MENU = ("auto", "stem_morph", "long_blend", "bass_swap", "stem_bass_swap", "stem_drum_swap",
              "filter_sweep", "drum_bridge", "breakdown_swap", "loop_in", "loop_roll_exit", "cut_at_drop",
              "phrase_cut", "spinback_cut", "echo_out", "acapella_out", "acapella_in", "melody_carry",
              "long_fade")
KIND_COLORS = {"intro": QColor(80, 95, 140), "groove": QColor(60, 140, 95), "build": QColor(180, 135, 55),
               "drop": QColor(190, 70, 70), "break": QColor(110, 80, 160), "breakdown": QColor(110, 80, 160),
               "outro": QColor(95, 95, 105), "verse": QColor(70, 125, 135), "chorus": QColor(170, 90, 110)}
SONG_PALETTE = [QColor(120, 180, 255), QColor(255, 165, 80), QColor(130, 225, 140), QColor(240, 120, 210),
                QColor(245, 225, 100), QColor(120, 220, 230)]
BG = QColor(18, 18, 22)
FG = QColor(230, 230, 236)
DIM = QColor(150, 150, 165)

STYLE = """
QWidget#perform { background: #121216; color: #e6e6ea; font-size: 11pt; }
QWidget#perform QPushButton { background: #2a2a33; color: #e6e6ea; border: 1px solid #3a3a46; border-radius: 7px;
                              padding: 9px 8px; font-size: 12pt; font-weight: 600; }
QWidget#perform QPushButton:hover { background: #363642; }
QWidget#perform QPushButton:pressed { background: #4a4a5a; }
QWidget#perform QPushButton:checked { background: #3b5b8a; border-color: #7da2e3; }
QWidget#perform QPushButton:disabled { color: #5a5a66; background: #1a1a20; border-color: #24242c; }
QWidget#perform QPushButton[kind="go"] { background: #2f6b3f; border-color: #43975a; }
QWidget#perform QPushButton[kind="go"]:hover { background: #3a8250; }
QWidget#perform QPushButton[kind="hot"] { background: #7a2c2c; border-color: #b04444; }
QWidget#perform QPushButton[kind="hot"]:hover { background: #953838; }
QWidget#perform QPushButton[kind="good"] { background: #245c3a; border-color: #3a8a58; }
QWidget#perform QPushButton[kind="bad"] { background: #5c2424; border-color: #8a3a3a; }
QWidget#perform QPushButton[kind="small"] { padding: 5px 6px; font-size: 10pt; font-weight: 500; }
QWidget#perform QComboBox, QWidget#perform QLineEdit { background: #1e1e25; color: #e6e6ea; border: 1px solid #3a3a46;
                                                       border-radius: 5px; padding: 5px 7px; font-size: 11pt; }
QWidget#perform QComboBox QAbstractItemView { background: #1e1e25; color: #e6e6ea; selection-background-color: #3b5b8a; }
QWidget#perform QListWidget { background: #16161b; color: #dcdce2; border: 1px solid #2a2a33; border-radius: 6px; font-size: 10.5pt; }
QWidget#perform QSlider::groove:horizontal { height: 10px; background: #2a2a33; border-radius: 5px; }
QWidget#perform QSlider::sub-page:horizontal { background: #4a7ad9; border-radius: 5px; }
QWidget#perform QSlider::handle:horizontal { width: 20px; margin: -6px 0; background: #b9cdf5; border-radius: 10px; }
QWidget#perform QLabel[dim="true"] { color: #9a9aa6; }
QWidget#perform QLabel[chip="true"] { color: #e6e6ea; background: #22222a; border: 1px solid #33333e; border-radius: 5px; padding: 3px 8px; }
QWidget#perform QToolButton { background: transparent; border: 1px solid transparent; border-radius: 5px; padding: 1px 5px; font-size: 12pt; }
QWidget#perform QToolButton:hover { border-color: #3a3a46; }
QWidget#perform QToolButton:checked { background: #3b5b8a; }
QWidget#perform QSplitter::handle { background: #22222a; }
"""


def _mmss(s):
    s = int(max(0, s or 0))
    return f"{s // 60}:{s % 60:02d}"


# ------------------------------------------------------------------------------------------------------
# painted widgets
# ------------------------------------------------------------------------------------------------------
class TrackMap(QWidget):
    """One deck's strip: sections coloured by kind, the energy curve, the playhead, the loop, the planned
    exit or entry, the blend window; a head line in the song's colour and status chips."""

    def __init__(self, title):
        super().__init__()
        self.setMinimumHeight(72)
        self.title = title
        self.color = FG
        self.brief = None
        self.map = None
        self.pos = None
        self.window = None
        self.chips = []

    def set(self, brief, tmap, pos=None, window=None, chips=(), color=None):
        self.brief, self.map, self.pos, self.window = brief, tmap, pos, window
        self.chips = list(chips)
        self.color = color or FG
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, BG)
        p.setPen(QColor(80, 80, 92))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 6, 6)
        f = QFont(self.font())
        f.setPointSizeF(10.5)
        f.setBold(True)
        p.setFont(f)
        if not self.brief or not self.map:
            p.setPen(DIM)
            p.drawText(QRectF(10, 0, W - 20, H), Qt.AlignmentFlag.AlignVCenter, f"{self.title}   -")
            return
        dur = max(float(self.map.get("duration") or self.brief.get("duration_s") or 1.0), 1.0)
        x0, x1, y0, y1 = 8, W - 8, 24, H - 6
        sx = (x1 - x0) / dur
        stems = self.map.get("stems") or {}
        # the section band (kind colours) - the top strip when stems are drawn below it
        band_h = (y1 - y0) if not stems else max(10.0, (y1 - y0) * 0.30)
        for s0, s1, kind, voc in self.map.get("sections") or []:
            col = KIND_COLORS.get(kind, QColor(90, 90, 100))
            p.fillRect(QRectF(x0 + s0 * sx, y0, max(1.0, (s1 - s0) * sx), band_h), col)
            if voc and voc > 0.5 and not stems:
                p.fillRect(QRectF(x0 + s0 * sx, y1 - 4, max(1.0, (s1 - s0) * sx), 3), QColor(255, 230, 140))
        if stems:
            # four stem rows: each stem's half-second presence as a heat row in the stem's colour
            rows = [("drums", QColor(235, 235, 240)), ("bass", QColor(255, 170, 90)), ("other", QColor(140, 190, 255)), ("vocals", QColor(255, 225, 120))]
            ry0 = y0 + band_h + 2
            rh = (y1 - ry0) / len(rows)
            for k, (name, col) in enumerate(rows):
                vals = stems.get(name) or []
                if not vals:
                    continue
                n = len(vals)
                wpt = (x1 - x0) / n
                yy = ry0 + k * rh
                for i, v in enumerate(vals):
                    if v <= 0.03:
                        continue
                    c = QColor(col)
                    c.setAlpha(int(40 + 215 * min(1.0, float(v))))
                    p.fillRect(QRectF(x0 + i * wpt, yy + 1, wpt + 0.5, rh - 2), c)
                p.setPen(QColor(20, 20, 24, 200))
                f3 = QFont(self.font())
                f3.setPointSizeF(7.5)
                p.setFont(f3)
                p.drawText(QRectF(x0 + 3, yy, 60, rh), Qt.AlignmentFlag.AlignVCenter, name[:1].upper())
        curve = self.map.get("energy") or []
        if len(curve) > 2 and not stems:
            pts = [QPointF(x0 + (x1 - x0) * i / (len(curve) - 1), y1 - (y1 - y0) * min(1.0, float(v))) for i, v in enumerate(curve)]
            p.setPen(QPen(QColor(240, 240, 250, 190), 1.5))
            p.drawPolyline(QPolygonF(pts))
        # the analyser's entry (green, from the top) and exit (red, from the bottom) points
        for t_in in self.map.get("ins") or []:
            x = x0 + t_in * sx
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(120, 220, 120)))
            p.drawPolygon(QPolygonF([QPointF(x - 5, y0 - 1), QPointF(x + 5, y0 - 1), QPointF(x, y0 + 7)]))
        for t_out in self.map.get("outs") or []:
            x = x0 + t_out * sx
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(255, 120, 90)))
            p.drawPolygon(QPolygonF([QPointF(x - 5, y1 + 1), QPointF(x + 5, y1 + 1), QPointF(x, y1 - 7)]))
        if self.window:
            a, b = self.window
            p.fillRect(QRectF(x0 + a * sx, y0, max(2.0, (b - a) * sx), y1 - y0), QColor(255, 255, 255, 55))
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
        # head: deck + song in the song's colour, then the numbers dim, then chips
        p.setPen(self.color)
        head = f"{self.title}  {self.brief.get('title')}"
        art = self.brief.get("artist") or ""
        fm = p.fontMetrics()
        p.drawText(QRectF(10, 2, W - 20, 20), Qt.AlignmentFlag.AlignVCenter, head)
        x = 10 + fm.horizontalAdvance(head) + 10
        f2 = QFont(f)
        f2.setBold(False)
        p.setFont(f2)
        p.setPen(DIM)
        nums = (f"{art}   " if art else "") + f"{self.brief.get('bpm', 0) or 0:.1f} bpm  {self.brief.get('camelot') or '?'}"
        if self.pos is not None:
            nums += f"   {_mmss(self.pos)} / {_mmss(dur)}"
        if ex is not None:
            nums += f"   exit {_mmss(ex)}"
        if en is not None:
            nums += f"   entry {_mmss(en)}"
        p.drawText(QRectF(x, 2, W - x - 10, 20), Qt.AlignmentFlag.AlignVCenter, nums)
        x += p.fontMetrics().horizontalAdvance(nums) + 14
        for text, col in self.chips:
            w = p.fontMetrics().horizontalAdvance(text) + 12
            if x + w > W - 10:
                break
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(col))
            p.drawRoundedRect(QRectF(x, 4, w, 16), 4, 4)
            p.setPen(QColor(20, 20, 24))
            p.drawText(QRectF(x, 4, w, 16), Qt.AlignmentFlag.AlignCenter, text)
            x += w + 6


class LaneBoard(QWidget):
    """The four stem lanes: which song plays each, in the song's colour, and how long it has been there."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(176)
        self.rows = []                 # [(lane, title, color, age_frac, note)]
        self.title = "LANES"

    def set(self, rows, title="LANES"):
        self.rows, self.title = list(rows), title
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, BG)
        p.setPen(QColor(80, 80, 92))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 6, 6)
        n = max(1, len(self.rows))
        rh = (H - 8) / n
        for i, (lane, title, color, age, note) in enumerate(self.rows):
            y = 4 + i * rh
            if color is not None:
                band = QColor(color)
                band.setAlpha(40)
                p.fillRect(QRectF(6, y + 3, W - 12, rh - 6), band)
                if age is not None:
                    fill = QColor(color)
                    fill.setAlpha(70)
                    p.fillRect(QRectF(6, y + 3, (W - 12) * max(0.0, min(1.0, age)), rh - 6), fill)
            f = QFont(self.font())
            f.setPointSizeF(10.5)
            f.setBold(True)
            p.setFont(f)
            p.setPen(DIM)
            p.drawText(QRectF(16, y, 110, rh), Qt.AlignmentFlag.AlignVCenter, lane.upper())
            f.setPointSizeF(15)
            p.setFont(f)
            p.setPen(color if color is not None else DIM)
            p.drawText(QRectF(130, y, W - 140, rh), Qt.AlignmentFlag.AlignVCenter, title or "resting")
            if note:
                f.setPointSizeF(10)
                f.setBold(False)
                p.setFont(f)
                p.setPen(DIM)
                p.drawText(QRectF(130, y, W - 146, rh), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, note)


class PhraseBar(QWidget):
    """The countdown: bars to the next move (Remix) or seconds to the seam (Automix)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(34)
        self.setMaximumHeight(34)
        self.frac, self.text, self.color, self.ticks = 0.0, "", QColor(74, 122, 217), 0

    def set(self, frac, text, color=None, ticks=0):
        self.frac, self.text, self.ticks = max(0.0, min(1.0, frac or 0.0)), text, ticks
        self.color = color or QColor(74, 122, 217)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(38, 38, 46)))
        p.drawRoundedRect(QRectF(0, 4, W, H - 8), 6, 6)
        p.setBrush(QBrush(self.color))
        p.drawRoundedRect(QRectF(0, 4, W * self.frac, H - 8), 6, 6)
        if self.ticks:
            p.setPen(QPen(QColor(18, 18, 22, 150), 1))
            for i in range(1, self.ticks):
                x = W * i / self.ticks
                p.drawLine(QPointF(x, 6), QPointF(x, H - 6))
        f = QFont(self.font())
        f.setPointSizeF(11)
        f.setBold(True)
        p.setFont(f)
        p.setPen(FG)
        p.drawText(QRectF(12, 0, W - 24, H), Qt.AlignmentFlag.AlignVCenter, self.text)


class PadGrid(QWidget):
    """The instrument: lanes down, songs across. A cell puts that lane on that song on the next bar (the
    conductor does the crossfade, the shapes, the hygiene); a lit cell is where the lane is; a grey cell
    says why the guard refuses it. Each column heads with the song, its fit, its section, a fader, LOOP
    and OUT; the last column rests the lane."""

    def __init__(self, tab):
        super().__init__()
        self.tab = tab
        self.grid = QGridLayout(self)
        self.grid.setSpacing(6)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self._decks = ()
        self._cells = {}          # (lane, deck) -> button
        self._heads = {}          # deck -> (title label, info label, fader, loop btn, out btn)
        self._rest = {}           # lane -> button

    def rebuild(self, decks):
        """Lay the grid out for these deck names (called only when the set of songs changes)."""
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)          # off the grid NOW (deleteLater alone leaves ghosts until the loop runs)
                w.deleteLater()
        self._cells, self._heads, self._rest = {}, {}, {}
        self._decks = tuple(decks)
        lab = QLabel("")
        self.grid.addWidget(lab, 0, 0)
        for j, d in enumerate(self._decks):
            head = QWidget()
            hv = QVBoxLayout(head)
            hv.setContentsMargins(4, 2, 4, 2)
            hv.setSpacing(2)
            title = QLabel(d.upper())
            f = title.font()
            f.setPointSizeF(12.5)
            f.setBold(True)
            title.setFont(f)
            title.setWordWrap(True)
            info = QLabel("")
            info.setProperty("dim", "true")
            info.setWordWrap(True)
            row = QHBoxLayout()
            row.setSpacing(4)
            fader = QSlider(Qt.Orientation.Horizontal)
            fader.setRange(0, 150)
            fader.setValue(100)
            fader.setToolTip("this song's level")
            fader.valueChanged.connect(lambda v, d=d: self.tab._song_level(d, v / 100.0))
            lp = QPushButton("LOOP 4")
            lp.setProperty("kind", "small")
            lp.setCheckable(True)
            lp.setToolTip("loop this song 4 bars (again: release)")
            lp.clicked.connect(lambda _c, d=d: self.tab._song_loop(d, 4))
            out = QPushButton("OUT")
            out.setProperty("kind", "small")
            out.setToolTip("take this song off: its lanes move on, it leaves a bar later")
            out.clicked.connect(lambda _c, d=d: self.tab._eject_deck(d))
            row.addWidget(fader, 1)
            row.addWidget(lp)
            row.addWidget(out)
            hv.addWidget(title)
            hv.addWidget(info)
            hv.addLayout(row)
            self.grid.addWidget(head, 0, 1 + j)
            self._heads[d] = (title, info, fader, lp, out)
        rest_h = QLabel("rest")
        rest_h.setProperty("dim", "true")
        rest_h.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grid.addWidget(rest_h, 0, 1 + len(self._decks))
        for i, ln in enumerate(LANES):
            name = QLabel(ln.upper())
            name.setProperty("dim", "true")
            f = name.font()
            f.setPointSizeF(11)
            f.setBold(True)
            name.setFont(f)
            self.grid.addWidget(name, 1 + i, 0)
            for j, d in enumerate(self._decks):
                b = QPushButton("")
                b.setMinimumHeight(44)
                b.setCheckable(True)
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                b.clicked.connect(lambda _c, ln=ln, d=d: self.tab._cell(ln, d))
                self.grid.addWidget(b, 1 + i, 1 + j)
                self._cells[(ln, d)] = b
            r = QPushButton("–")
            r.setMinimumHeight(44)
            r.setMaximumWidth(64)
            r.setCheckable(True)
            r.setToolTip(f"rest the {ln} lane")
            r.clicked.connect(lambda _c, ln=ln: self.tab._cell(ln, None))
            self.grid.addWidget(r, 1 + i, 1 + len(self._decks))
            self._rest[ln] = r
        self.grid.setColumnStretch(0, 0)
        for j in range(len(self._decks)):
            self.grid.setColumnStretch(1 + j, 1)

    def render(self, st, colors):
        songs = st.get("songs") or {}
        decks = tuple(d for d in ("a", "b", "c", "d") if d in songs and not (songs[d].get("leaving")))
        if decks != self._decks:
            self.rebuild(decks)
        lanes = st.get("lanes") or {}
        grid = st.get("grid") or {}
        for d in decks:
            s = songs[d]
            title, info, fader, lp, out = self._heads[d]
            col = colors(s.get("id") or s.get("title"))
            title.setText(f"{d.upper()}  {s.get('title')}")
            title.setStyleSheet(f"color: rgb({col.red()},{col.green()},{col.blue()});")
            bits = [f"{s.get('bpm', 0) or 0:.0f} bpm ×{s.get('rate', 1):.3f}", f"{s.get('camelot') or '?'}{f' {s['shift']:+d}st' if s.get('shift') else ''}"]
            if s.get("compat") is not None:
                bits.append(f"fit {s['compat']:.2f}")
            if s.get("section"):
                bits.append(s["section"] + (" · singing" if s.get("singing") else ""))
            if d == st.get("master"):
                bits.append("CLOCK")
            if not s.get("staged"):
                bits.append("decoding…")
            elif not s.get("entered"):
                bits.append("staged, silent")
            if s.get("lock_ms") is not None:
                bits.append(f"lock {s['lock_ms']:.0f} ms")
            info.setText("  ·  ".join(bits) + f"   {_mmss(s.get('time_s'))}/{_mmss(s.get('duration_s'))}")
            if not fader.isSliderDown():
                fader.blockSignals(True)
                fader.setValue(int(round(100 * (s.get("level") if s.get("level") is not None else 1.0))))
                fader.blockSignals(False)
            lp.setChecked(bool(s.get("user_loop")))
            for ln in LANES:
                b = self._cells[(ln, d)]
                why = (grid.get(d) or {}).get(ln)
                on = lanes.get(ln) == d
                b.setChecked(on)
                b.setEnabled(why is None or on)
                b.setText(("● " if on else "") + (s.get("title") or "")[:22] + (f"\n{why[:34]}" if why and not on else ""))
                b.setToolTip(why or f"put the {ln} lane on {s.get('title')} on the next bar")
                b.setStyleSheet(f"QPushButton:checked {{ background: rgba({col.red()},{col.green()},{col.blue()},110); border-color: rgb({col.red()},{col.green()},{col.blue()}); }}")
        for ln in LANES:
            r = self._rest.get(ln)
            if r is not None:
                r.setChecked(lanes.get(ln) is None)


class ArcView(QWidget):
    """The night's energy arc and where we are on it (Automix)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(54)
        self.setMaximumHeight(64)
        self.curve, self.phase, self.heat, self.nudge = [], 0.0, 0.0, 0.0

    def set(self, curve, phase, heat, nudge):
        self.curve, self.phase, self.heat, self.nudge = curve or [], float(phase or 0), float(heat or 0), float(nudge or 0)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, BG)
        p.setPen(QColor(80, 80, 92))
        p.drawRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 6, 6)
        x0, x1, y0, y1 = 8, W - 8, 18, H - 6
        if len(self.curve) > 2:
            pts = [QPointF(x0 + (x1 - x0) * i / (len(self.curve) - 1), y1 - (y1 - y0) * min(1.0, float(v))) for i, v in enumerate(self.curve)]
            p.setPen(QPen(QColor(200, 170, 90), 2))
            p.drawPolyline(QPolygonF(pts))
        x = x0 + (x1 - x0) * min(1.0, max(0.0, self.phase))
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(QPointF(x, y0), QPointF(x, y1))
        p.setPen(DIM)
        p.drawText(QRectF(10, 0, W - 20, 16), Qt.AlignmentFlag.AlignVCenter,
                   f"the night's arc: {100 * self.phase:.0f}% through the set, target energy {self.heat:.2f}"
                   + (f", lean {self.nudge:+.2f}" if abs(self.nudge) > 0.01 else ""))


# ------------------------------------------------------------------------------------------------------
# the copilot's bridge
# ------------------------------------------------------------------------------------------------------
class SteerBridge:
    """What the Perform Copilot may touch. The copilot runs on its own thread; it READS state here and
    QUEUES control changes and actions, and the tab drains the queue on its timer - no widget and no
    engine call ever happens off the GUI thread."""

    def __init__(self, tab):
        self.tab = tab
        self.q = []
        self.lock = threading.Lock()
        self.controls = {}

    def snapshot_controls(self):
        t = self.tab
        self.controls = {"mode": t.mode_box.currentText(), "theme": t.theme_box.currentText(),
                         "blend": t.blend.value() / 100.0, "vocals": t.vocals.value() / 100.0,
                         "change_bars": REMIX_BARS.get(t.speed_box.currentText(), 8), "mix_speed": t.speed_box.currentText(),
                         "mix_style": t.style_box.currentText(), "energy_lean": t.energy.value() / 100.0,
                         "tempo": t.tempo.value() / 1000.0, "hold": t.buttons["HOLD"].isChecked(),
                         "songs": t.songs_box.currentText()}

    def state(self):
        out = {"controls": dict(self.controls)}
        rc, sysm = self.tab.remix, self.tab.system
        try:
            if rc is not None:
                st = rc.status()
                songs = st.get("songs") or {}
                out.update({"mode": "Remix",
                            "lanes": {ln: (songs.get(d) or {}).get("title") if d else None for ln, d in (st.get("lanes") or {}).items()},
                            "songs": [{"title": s.get("title"), "lanes": s.get("lanes"), "section": s.get("section"),
                                       "singing": s.get("singing"), "entered": s.get("entered")} for s in songs.values()],
                            "clock_bpm": st.get("master_bpm"), "key": st.get("key_centre"), "arc_phase": st.get("arc_phase"),
                            "energy_target": st.get("energy"), "hold": st.get("hold"), "loop": st.get("user_loop"),
                            "breaking": st.get("breaking"), "n_snapshots": st.get("n_snapshots"), "recording": st.get("recording"),
                            "pool": st.get("pool_name"), "last_moves": [m for _, m in (st.get("moves") or [])[-6:]],
                            "last_rated": st.get("last_rated")})
            elif sysm is not None:
                st = sysm.status()
                out.update({"mode": "Automix", "state": st.get("state"), "playing": (st.get("current") or {}).get("title"),
                            "next": (st.get("next") or {}).get("title"), "plan": st.get("plan"),
                            "arc_phase": st.get("arc_phase"), "energy_target": st.get("arc_heat"), "setlist": st.get("setlist")})
            else:
                out["mode"] = "idle (the operator has not pressed Start)"
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
        return out

    def steer(self, kw):
        with self.lock:
            self.q.append(("steer", dict(kw)))
        return {"ok": True, "set": kw}

    def act(self, action):
        with self.lock:
            self.q.append(("act", action))
        return {"ok": True, "action": action}

    def rate(self, good):
        with self.lock:
            self.q.append(("rate", bool(good)))
        return {"ok": True, "good": bool(good)}

    def drain(self):
        with self.lock:
            items, self.q = self.q, []
        return items


class SetlistBox(QComboBox):
    """The songs box: 'whole library' or one of the saved setlists; the list refreshes when opened."""

    def __init__(self, tab):
        super().__init__()
        self.tab = tab
        self.addItem("songs: whole library", None)

    def showPopup(self):
        self.refresh()
        super().showPopup()

    def refresh(self):
        cur = self.currentData()
        db = self.tab._db()
        names = []
        if db is not None:
            try:
                from lib.dj import setlist as SL
                names = [(f"{r['name']}  ({r['n_tracks']} tracks)", r["name"]) for r in SL.list_setlists(db)]
            except Exception:
                names = []
        self.blockSignals(True)
        self.clear()
        self.addItem("songs: whole library", None)
        for label, name in names:
            self.addItem(label, name)
        i = self.findData(cur)
        self.setCurrentIndex(i if i >= 0 else 0)
        self.blockSignals(False)


# ------------------------------------------------------------------------------------------------------
# the tab
# ------------------------------------------------------------------------------------------------------
class PerformTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.setObjectName("perform")
        self.setStyleSheet(STYLE)
        self.planner = planner
        self.engine = None
        self.system = None
        self.remix = None
        self._own_db = None
        self._seen_events = 0
        self._feed_sig = None
        self._song_colors = {}
        self.midi = None
        self._midi_note = ""
        self._recall_idx = 0
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # -- the top bar: transport, mode, theme, songs, the state chips ------------------------------
        top = QHBoxLayout()
        self.start_btn = QPushButton("▶  START")
        self.start_btn.setProperty("kind", "go")
        self.start_btn.setMinimumWidth(130)
        self.start_btn.setToolTip("open the audio device and let the system play")
        self.start_btn.clicked.connect(self._toggle)
        top.addWidget(self.start_btn)
        self.mode_box = QComboBox()
        self.mode_box.addItems(list(MODES))
        self.mode_box.setToolTip("Automix: whole songs, one seam at a time.  Remix: parts of two or three songs together, changing every phrase")
        self.mode_box.currentTextChanged.connect(self._mode_changed)
        top.addWidget(self.mode_box)
        self.theme_box = QComboBox()
        from lib.dj.themes import PICKER_THEMES
        self.theme_box.addItems(list(PICKER_THEMES))
        self.theme_box.setCurrentText("groove")
        self.theme_box.setToolTip("theme: what it plays and how the night moves")
        self.theme_box.currentTextChanged.connect(self._theme)
        top.addWidget(self.theme_box)
        self.songs_box = SetlistBox(self)
        self.songs_box.setMinimumWidth(220)
        self.songs_box.setToolTip("confine the music to one of your saved setlists (the Set tab's lists); whole library lifts it")
        self.songs_box.currentIndexChanged.connect(self._pool_changed)
        top.addWidget(self.songs_box)
        top.addSpacing(12)
        self.state_lbl = QLabel("")
        self.state_lbl.setProperty("dim", "true")
        top.addWidget(self.state_lbl, 1)
        root.addLayout(top)

        # -- stage + deck ---------------------------------------------------------------------------------
        split = QSplitter(Qt.Orientation.Horizontal)
        stage = QWidget()
        sv = QVBoxLayout(stage)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(6)
        self.pads = PadGrid(self)
        sv.addWidget(self.pads)
        self.phrase = PhraseBar()
        sv.addWidget(self.phrase)
        # the crate: songs that fit the clock now, ranked, with why; pick one and it stages on a free deck
        crate = QHBoxLayout()
        crate.setSpacing(6)
        self.crate_lbl = QLabel("CRATE")
        self.crate_lbl.setProperty("dim", "true")
        crate.addWidget(self.crate_lbl)
        self.crate_search = QLineEdit()
        self.crate_search.setPlaceholderText("search the crate (title / artist)")
        self.crate_search.setMaximumWidth(260)
        crate.addWidget(self.crate_search)
        self.crate_stage = QPushButton("STAGE →")
        self.crate_stage.setProperty("kind", "small")
        self.crate_stage.setToolTip("decode the chosen song onto a free deck, beat-locked and key-fitted, silent until you give it a lane")
        self.crate_stage.clicked.connect(self._stage_pick)
        crate.addWidget(self.crate_stage)
        self.crate_note = QLabel("")
        self.crate_note.setProperty("dim", "true")
        crate.addWidget(self.crate_note, 1)
        sv.addLayout(crate)
        self.crate_list = QListWidget()
        self.crate_list.setMinimumHeight(96)
        self.crate_list.setMaximumHeight(150)
        self.crate_list.itemDoubleClicked.connect(lambda _i: self._stage_pick())
        sv.addWidget(self.crate_list)
        self.map_now = TrackMap("A")
        self.map_next = TrackMap("B")
        self.map_c = TrackMap("C")
        self.map_d = TrackMap("D")
        self.deck_maps = {"a": self.map_now, "b": self.map_next, "c": self.map_c, "d": self.map_d}
        for m in (self.map_now, self.map_next, self.map_c, self.map_d):
            sv.addWidget(m)
        self.seam_lbl = QLabel("")
        self.seam_lbl.setWordWrap(True)
        self.seam_lbl.setProperty("dim", "true")
        sv.addWidget(self.seam_lbl)
        self.arc = ArcView()
        sv.addWidget(self.arc)
        sv.addStretch(1)
        split.addWidget(stage)

        deck = QWidget()
        deck.setMinimumWidth(340)
        deck.setMaximumWidth(400)
        dv = QVBoxLayout(deck)
        dv.setContentsMargins(6, 0, 0, 0)
        dv.setSpacing(8)
        self.buttons = {}
        grid = QGridLayout()
        grid.setSpacing(8)

        def big(label, fn, tip, kind=None, row=0, col=0, checkable=False):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setMinimumHeight(52)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if kind:
                b.setProperty("kind", kind)
            if checkable:
                b.setCheckable(True)
            b.clicked.connect(fn)
            grid.addWidget(b, row, col)
            self.buttons[label] = b
            return b
        big("MIX NOW", self._mix_now, "Remix: a move on the next bar.  Automix: the next transition, now.  [Space]", "go", 0, 0)
        big("HOLD", self._hold, "Remix: freeze the lane map.  Automix: one more phrase before the seam.  [H]", None, 0, 1, checkable=True)
        big("DROP", self._drop, "Remix: every lane to the newest song on the next bar.  Automix: the drop moment.  [D]", "hot", 1, 0)
        big("BREAK", self._break, "every lane but one rests four bars, then all come back on the bar  [B]", None, 1, 1)
        big("LOOP 4", lambda: self._loop(4), "every live song holds 4 bars (again: release)  [4]", None, 2, 0, checkable=True)
        big("LOOP 8", lambda: self._loop(8), "every live song holds 8 bars (again: release)  [8]", None, 2, 1, checkable=True)
        big("NEXT DROP", lambda: self._moment("nextdrop"), "land the next track's drop on this one's bar", None, 3, 0)
        big("REROLL", self._reroll, "a different next track", None, 3, 1)
        big("ABORT MIX", self._abort, "recall an armed transition before its point of no return", "hot", 4, 0)
        dv.addLayout(grid)

        # sliders with their values
        sl = QGridLayout()
        sl.setHorizontalSpacing(8)
        sl.setVerticalSpacing(6)
        self._slider_rows = {}

        def slider(row, name, lo, hi, val, tip, fmt):
            lab = QLabel(name)
            lab.setProperty("dim", "true")
            lab.setMinimumWidth(56)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(lo, hi)
            s.setValue(val)
            s.setToolTip(tip)
            v = QLabel(fmt(val))
            v.setMinimumWidth(52)
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            s.valueChanged.connect(lambda x, v=v, fmt=fmt: v.setText(fmt(x)))
            sl.addWidget(lab, row, 0)
            sl.addWidget(s, row, 1)
            sl.addWidget(v, row, 2)
            self._slider_rows[name] = (lab, s, v)
            return s
        self.energy = slider(0, "energy", -40, 40, 0, "lean on the theme's arc target, -0.4 .. +0.4", lambda x: f"{x / 100:+.2f}")
        self.energy.valueChanged.connect(self._energy)
        self.blend = slider(1, "blend", 0, 100, 70, "0: song follows song (a morph) · 1: free recombination of three songs", lambda x: f"{x / 100:.2f}")
        self.blend.valueChanged.connect(lambda x: self.remix and self.remix.set_blend(x / 100.0))
        self.vocals = slider(2, "vocals", 0, 100, 40, "how freely the vocal lane crosses between songs", lambda x: f"{x / 100:.2f}")
        self.vocals.valueChanged.connect(lambda x: self.remix and self.remix.set_vocal_freedom(x / 100.0))
        self.tempo = slider(3, "tempo", 0, 60, 0, "tempo journey: how far the clock may travel with the arc, 0 (fixed) .. 6 %", lambda x: f"±{x / 10:.1f}%")
        self.tempo.valueChanged.connect(lambda x: self.remix and self.remix.set_tempo_span(x / 1000.0))
        self.auto = slider(4, "auto", 0, 100, 0, "autopilot: how often the conductor takes a phrase's move itself - 0 = only you move things (the grid), 100 = every phrase", lambda x: f"{x}%")
        self.auto.valueChanged.connect(lambda x: self.remix and self.remix.set_auto(x / 100.0))
        dv.addLayout(sl)
        combos = QGridLayout()
        combos.setHorizontalSpacing(8)
        self.speed_lbl = QLabel("change")
        self.speed_lbl.setProperty("dim", "true")
        self.speed_box = QComboBox()
        self.speed_box.addItems([s for s, _f in SPEEDS])
        self.speed_box.setCurrentText("normal")
        self.speed_box.currentTextChanged.connect(self._speed)
        combos.addWidget(self.speed_lbl, 0, 0)
        combos.addWidget(self.speed_box, 0, 1)
        self.style_lbl = QLabel("mix type")
        self.style_lbl.setProperty("dim", "true")
        self.style_box = QComboBox()
        self.style_box.addItems(list(STYLE_MENU))
        self.style_box.currentTextChanged.connect(self._style)
        combos.addWidget(self.style_lbl, 1, 0)
        combos.addWidget(self.style_box, 1, 1)
        dv.addLayout(combos)

        verd = QHBoxLayout()
        for label, fn, tip, kind in (("👍 GOOD", lambda: self._verdict(True), "the last move / seam was good - the system learns  [→]", "good"),
                                     ("👎 BAD", lambda: self._verdict(False), "the last move / seam was bad - the system learns  [←]", "bad")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setMinimumHeight(48)
            b.setProperty("kind", kind)
            b.clicked.connect(fn)
            verd.addWidget(b)
            self.buttons[label.split(" ", 1)[1]] = b
        dv.addLayout(verd)
        mem = QHBoxLayout()
        for label, fn, tip in (("SAVE", self._save, "remember this combination  [S]"),
                               ("RECALL", self._recall, "bring the last saved combination back (again: the one before)  [R]"),
                               ("REC", self._rec, "record the output to logs/remix_*.wav with the move log beside it")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setProperty("kind", "small")
            b.clicked.connect(fn)
            mem.addWidget(b)
            self.buttons[label] = b
        self.buttons["REC"].setCheckable(True)
        dv.addLayout(mem)
        self.engine_lbl = QLabel("")
        self.engine_lbl.setWordWrap(True)
        self.engine_lbl.setProperty("dim", "true")
        dv.addWidget(self.engine_lbl)
        dv.addStretch(1)
        split.addWidget(deck)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1100, 360])
        root.addWidget(split, 3)

        # -- the feed and the say line -------------------------------------------------------------------------
        say = QHBoxLayout()
        self.say_edit = QLineEdit()
        self.say_edit.setPlaceholderText("tell the DJ: darker · more vocals · slower changes · wilder · drop it · hold this · that was good")
        self.say_edit.returnPressed.connect(self._say)
        say.addWidget(self.say_edit, 1)
        self.say_btn = QPushButton("SAY")
        self.say_btn.setProperty("kind", "small")
        self.say_btn.setToolTip("the copilot reads the live state and sets the controls your words ask for (Claude Code session, no key)")
        self.say_btn.clicked.connect(self._say)
        say.addWidget(self.say_btn)
        self.say_lbl = QLabel("")
        self.say_lbl.setProperty("dim", "true")
        say.addWidget(self.say_lbl, 2)
        root.addLayout(say)
        self.coming_lbl = QLabel("")
        self.coming_lbl.setProperty("dim", "true")
        root.addWidget(self.coming_lbl)
        self.horizon_list = QListWidget()          # the automixer's coming-up list (Remix uses the label)
        self.horizon_list.setMaximumHeight(72)
        root.addWidget(self.horizon_list)
        self.event_list = QListWidget()            # the feed: moves newest first, thumbs on the move
        self.event_list.setMinimumHeight(120)
        root.addWidget(self.event_list, 2)

        self._bridge = SteerBridge(self)
        self._copilot = None
        self._say_thread = None
        self._say_result = None
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._tick)
        self._keys()
        self._mode_changed(self.mode_box.currentText())

    # -- keys -----------------------------------------------------------------------------------------------
    def _keys(self):
        for seq, fn in (("Space", self._mix_now), ("H", self._hold_toggle), ("D", self._drop), ("B", self._break),
                        ("4", lambda: self._loop(4)), ("8", lambda: self._loop(8)), ("Left", lambda: self._verdict(False)),
                        ("Right", lambda: self._verdict(True)), ("S", self._save), ("R", self._recall)):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # -- data ---------------------------------------------------------------------------------------------------
    def _db(self):
        db = getattr(self.planner, "db", None)
        if db is not None:
            return db
        if self.remix is not None:
            return self.remix.db
        if self._own_db is None:
            try:
                from lib.dj.db import LibraryDB
                self._own_db = LibraryDB(self.planner.music_dir)
            except Exception:
                return None
        return self._own_db

    def _color(self, key):
        if key not in self._song_colors:
            self._song_colors[key] = SONG_PALETTE[len(self._song_colors) % len(SONG_PALETTE)]
        return self._song_colors[key]

    # -- engine ---------------------------------------------------------------------------------------------------
    def _toggle(self):
        if self.system is None and self.remix is None:
            try:
                self._start()
            except Exception as e:  # noqa: BLE001
                self.engine_lbl.setText(f"could not start: {type(e).__name__}: {e}")
                self.close()
        else:
            self.close()
            self.start_btn.setText("▶  START")

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
            self.remix.set_tempo_span(self.tempo.value() / 1000.0)
            self.remix.set_auto(self.auto.value() / 100.0)
            self._energy(self.energy.value())
            self._pool_changed()
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
            self._pool_changed()
        self.mode_box.setEnabled(False)
        self.start_btn.setText("■  STOP")
        self.start_btn.setProperty("kind", "hot")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self._seen_events = 0
        self._feed_sig = None
        self.horizon_list.clear()
        self.event_list.clear()
        self.buttons["HOLD"].setChecked(False)
        self.buttons["REC"].setChecked(False)
        self._sync_loop_buttons()
        self._midi_open()
        self._timer.start()

    def close(self):
        self._timer.stop()
        self._midi_close()
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
        self.start_btn.setText("▶  START")
        self.start_btn.setProperty("kind", "go")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)

    # -- controls -------------------------------------------------------------------------------------------------
    def _theme(self, name):
        if self.system is not None:
            self.system.set_theme(name)
        if self.remix is not None:
            self.remix.set_theme(name)

    def _pool_changed(self, *_):
        """The songs box: a saved setlist as the pool, or the whole library."""
        name = self.songs_box.currentData()
        if self.remix is not None:
            if name:
                from lib.dj.setlist import get_setlist
                sl = get_setlist(self.remix.db, name=name)
                ids = [e["track_id"] for e in (sl or {}).get("entries", [])]
                n = self.remix.set_pool(ids)
                self.remix.pool_name = name
                self.coming_lbl.setText(f"songs: {name} - {n} of its tracks have stems" + ("  (none: the pool is empty, the whole library plays)" if n == 0 else ""))
                if n == 0:
                    self.remix.set_pool(None)
            else:
                self.remix.set_pool(None)
        if self.system is not None:
            self.system.load_setlist(name or "", mode="pool")

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
            self.remix.set_energy_lean(x / 100.0)

    def _mix_now(self):
        if self.system is not None:
            self.system.request_skip()
        if self.remix is not None:
            self.remix.next_move()

    def _hold(self):
        if self.system is not None:
            self.system.request_hold()
            self.buttons["HOLD"].setChecked(False)
        if self.remix is not None:
            self.remix.set_hold(self.buttons["HOLD"].isChecked())

    def _hold_toggle(self):
        b = self.buttons["HOLD"]
        if b.isCheckable():
            b.setChecked(not b.isChecked())
        self._hold()

    def _reroll(self):
        if self.system is not None:
            self.system.request_reroll()

    def _moment(self, flavor):
        if self.system is not None:
            self.system.moment(flavor)

    def _abort(self):
        if self.system is not None:
            self.system.abort_transition()

    def _drop(self):
        if self.remix is not None:
            self.remix.drop()
        elif self.system is not None:
            self.system.moment("drop")

    def _break(self):
        if self.remix is not None:
            self.remix.break_()

    def _loop(self, bars):
        if self.remix is not None:
            self.remix.loop(bars)
            self._sync_loop_buttons()

    def _sync_loop_buttons(self):
        cur = self.remix.user_loop_bars if self.remix is not None else None
        for bars in (4, 8):
            self.buttons[f"LOOP {bars}"].setChecked(cur == bars)

    def _verdict(self, up):
        if self.remix is not None:
            m = self.remix.rate_last(up)
            if m is None:
                self.say_lbl.setText("nothing left to rate - every move has a verdict")
        elif self.system is not None:
            self.system.seam_feedback(up)

    def _rate_move(self, move_id, up):
        """A thumb on a feed row: set, or clear when it is already that verdict."""
        if self.remix is None:
            return
        cur = next((m.get("fb") for m in self.remix.move_log if m.get("id") == move_id), None)
        self.remix.rate_id(move_id, None if cur is up else up)
        self._feed_sig = None

    def _save(self):
        if self.remix is not None:
            self.remix.save_snapshot()
            self._recall_idx = 0

    def _recall(self):
        if self.remix is None or not self.remix.snapshots:
            return
        n = len(self.remix.snapshots)
        self.remix.recall_snapshot(-1 - (self._recall_idx % n))
        self._recall_idx += 1

    def _rec(self):
        if self.remix is None:
            self.buttons["REC"].setChecked(False)
            return
        if self.buttons["REC"].isChecked():
            self.remix.record()
        else:
            self.remix.record_stop()

    # -- the instrument: grid, crate, faders ------------------------------------------------------------------
    def _cell(self, lane, deck):
        if self.remix is None:
            return
        ok, msg = self.remix.assign(lane, deck)
        self.crate_note.setText(msg if ok else f"refused: {msg}")
        self._feed_sig = None

    def _stage_pick(self):
        if self.remix is None:
            self.crate_note.setText("press START first")
            return
        it = self.crate_list.currentItem()
        if it is None:
            self.crate_note.setText("choose a song in the crate")
            return
        ok, msg = self.remix.stage_track(it.data(Qt.ItemDataRole.UserRole))
        self.crate_note.setText(msg)

    def _eject_deck(self, deck):
        if self.remix is not None:
            self.remix.eject(deck)

    def _song_loop(self, deck, bars):
        if self.remix is None:
            return
        s = self.remix.songs.get(deck)
        self.remix.song_loop(deck, None if (s is not None and s.user_loop == bars) else bars)

    def _song_level(self, deck, level):
        if self.remix is not None:
            self.remix.song_gain(deck, level)

    # -- the Perform Copilot -------------------------------------------------------------------------------
    def _say(self):
        text = self.say_edit.text().strip()
        if not text:
            return
        if self._say_thread is not None and self._say_thread.is_alive():
            self.say_lbl.setText("(still thinking about the last one)")
            return
        if self._copilot is None:
            try:
                from tools.dj.planner.perform_copilot import PerformCopilot
                lib = self.remix.library if self.remix is not None else (getattr(self.planner, "library", None) or [])
                self._copilot = PerformCopilot(self._bridge, lib, theme_name=self.theme_box.currentText())
            except Exception as e:  # noqa: BLE001
                self.say_lbl.setText(f"copilot unavailable: {type(e).__name__}: {e}")
                return
        if not self._copilot.available():
            self.say_lbl.setText(f"copilot unavailable: {self._copilot.why_unavailable()}")
            return
        self._bridge.snapshot_controls()
        self.say_edit.clear()
        self.say_lbl.setText(f"… {text}")
        cp = self._copilot

        def work():
            try:
                reply = cp.run_turn(text)
                self._say_result = (text, reply, None)
            except Exception as e:  # noqa: BLE001
                self._say_result = (text, None, f"{type(e).__name__}: {e}")
        self._say_thread = threading.Thread(target=work, daemon=True, name="perform-copilot")
        self._say_thread.start()

    def _apply_steer(self):
        for kind, arg in self._bridge.drain():
            try:
                if kind == "steer":
                    self._apply_controls(arg)
                elif kind == "act":
                    self._apply_action(arg)
                elif kind == "rate":
                    self._verdict(bool(arg))
            except Exception as e:  # noqa: BLE001
                self.say_lbl.setText(f"steer error: {type(e).__name__}: {e}")
        res = self._say_result
        if res is not None:
            self._say_result = None
            text, reply, err = res
            self.say_lbl.setText(f"{text} → {reply if err is None else 'copilot error: ' + err}"[:400])

    def _apply_controls(self, kw):
        if "blend" in kw:
            self.blend.setValue(int(round(100 * max(0.0, min(1.0, kw["blend"])))))
        if "vocals" in kw:
            self.vocals.setValue(int(round(100 * max(0.0, min(1.0, kw["vocals"])))))
        if "energy_lean" in kw:
            self.energy.setValue(int(round(100 * max(-0.4, min(0.4, kw["energy_lean"])))))
        if "tempo" in kw:
            self.tempo.setValue(int(round(1000 * max(0.0, min(0.06, kw["tempo"])))))
        if "auto" in kw:
            self.auto.setValue(int(round(100 * max(0.0, min(1.0, kw["auto"])))))
        if "change_bars" in kw:
            name = next((n for n, b in REMIX_BARS.items() if b == int(kw["change_bars"])), None)
            if name:
                self.speed_box.setCurrentText(name)
        if "mix_speed" in kw and kw["mix_speed"] in dict(SPEEDS):
            self.speed_box.setCurrentText(kw["mix_speed"])
        if "mix_style" in kw and kw["mix_style"] in STYLE_MENU:
            self.style_box.setCurrentText(kw["mix_style"])
        if "theme" in kw:
            from lib.dj.themes import BUILTIN_THEMES
            if kw["theme"] in BUILTIN_THEMES:
                if self.theme_box.findText(kw["theme"]) < 0:
                    self.theme_box.addItem(kw["theme"])
                self.theme_box.setCurrentText(kw["theme"])

    def _apply_action(self, action):
        if action == "next":
            self._mix_now()
        elif action in ("hold", "unhold"):
            b = self.buttons["HOLD"]
            if b.isCheckable():
                b.setChecked(action == "hold")
            self._hold()
        elif action == "drop":
            self._drop()
        elif action == "break":
            self._break()
        elif action in ("loop4", "loop8"):
            bars = 4 if action == "loop4" else 8
            if self.remix is not None and self.remix.user_loop_bars != bars:
                self._loop(bars)
        elif action == "unloop":
            if self.remix is not None and self.remix.user_loop_bars is not None:
                self.remix.loop(None)
                self._sync_loop_buttons()
        elif action == "save":
            self._save()
        elif action == "recall":
            self._recall()
        elif action in ("rec", "rec_stop"):
            self.buttons["REC"].setChecked(action == "rec")
            self._rec()

    # -- nanoKONTROL2 -----------------------------------------------------------------------------------------
    # faders: 1 energy lean, 2 blend, 3 vocals; knobs: 1 change rate / mix speed, 2 tempo.
    # transport: PLAY = MIX NOW, STOP = HOLD, REC = DROP, CYCLE = BREAK, MARKER < > = LOOP 4 / 8,
    # TRACK < > = BAD / GOOD on the last move. Polled from the readout timer; never a thread.
    def _midi_open(self):
        self.midi = None
        try:
            from lib.midi_controller import KorgNanoKontrol2
            ctl = KorgNanoKontrol2(auto_connect=True)
            if ctl.input_device is not None:
                self.midi = ctl
        except Exception as e:  # noqa: BLE001
            self._midi_note = f"nanoKONTROL2: {type(e).__name__}: {e}"
            return
        self._midi_note = "nanoKONTROL2 connected" if self.midi is not None else "nanoKONTROL2 not found"

    def _midi_close(self):
        if getattr(self, "midi", None) is not None:
            try:
                self.midi.disconnect()
            except Exception:
                pass
        self.midi = None

    def _midi_poll(self):
        ctl = getattr(self, "midi", None)
        if ctl is None:
            return
        try:
            changes = ctl.update()
        except Exception as e:  # noqa: BLE001
            self._midi_note = f"nanoKONTROL2: {type(e).__name__}: {e}"
            self.midi = None
            return
        if not changes:
            return
        speeds = [s for s, _f in SPEEDS]
        decks = ("a", "b", "c", "d")
        for name, val in changes.items():
            # channel strips 1-4 = decks A-D: fader = that song's level, S / M / R = drums / bass / other to
            # that song; strips 5-8: S = vocals to deck 1-4, faders 5-8 = energy, blend, vocals, auto
            if name.startswith("slider_"):
                i = int(name.split("_")[1])
                if i <= 4:
                    if self.remix is not None and decks[i - 1] in self.remix.songs:
                        self._song_level(decks[i - 1], 1.5 * val)
                elif i == 5:
                    self.energy.setValue(int(round(-40 + 80 * val)))
                elif i == 6:
                    self.blend.setValue(int(round(100 * val)))
                elif i == 7:
                    self.vocals.setValue(int(round(100 * val)))
                elif i == 8:
                    self.auto.setValue(int(round(100 * val)))
            elif name == "knob_1":
                self.speed_box.setCurrentText(speeds[min(len(speeds) - 1, int(val * len(speeds)))])
            elif name == "knob_2":
                self.tempo.setValue(int(round(60 * val)))
            elif val is True and name[:2] in ("s_", "m_", "r_") and "button" in name:
                i = int(name.rsplit("_", 1)[1])
                if self.remix is not None:
                    if i <= 4:
                        lane = {"s": "drums", "m": "bass", "r": "other"}[name[0]]
                        self._cell(lane, decks[i - 1])
                    elif name[0] == "s":
                        self._cell("vocals", decks[i - 5])
            elif val is True:                                   # transport buttons act on press
                if name == "play":
                    self._mix_now()
                elif name == "stop":
                    self._hold_toggle()
                elif name == "record":
                    self._drop()
                elif name == "cycle":
                    self._break()
                elif name == "marker_prev":
                    self._loop(4)
                elif name == "marker_next":
                    self._loop(8)
                elif name == "track_prev":
                    self._verdict(False)
                elif name == "track_next":
                    self._verdict(True)

    # -- mode -----------------------------------------------------------------------------------------------------
    def _mode_changed(self, mode):
        remix = mode == "Remix"
        for name in ("BREAK", "LOOP 4", "LOOP 8", "SAVE", "RECALL", "REC"):
            self.buttons[name].setVisible(remix)
        for name in ("NEXT DROP", "REROLL", "ABORT MIX"):
            self.buttons[name].setVisible(not remix)
        for name in ("blend", "vocals", "tempo", "auto"):
            for w in self._slider_rows[name]:
                w.setVisible(remix)
        self.style_lbl.setVisible(not remix)
        self.style_box.setVisible(not remix)
        self.speed_lbl.setText("change" if remix else "mix speed")
        self.speed_box.setToolTip("bars between moves: short 4 · normal 8 · long 16 · marathon 32" if remix else "how long the blends run")
        self.buttons["HOLD"].setCheckable(remix)
        self.buttons["DROP"].setToolTip("every lane to the newest song on the next bar  [D]" if remix else "the drop moment: build and land on the bar  [D]")
        for d, m in self.deck_maps.items():
            m.setMinimumHeight(104 if remix else 72)
            m.setVisible(d in ("a", "b") if not remix else d == "a")     # Remix: the tick shows the decks that have songs
        self.arc.setVisible(not remix)
        self.horizon_list.setVisible(not remix)
        for w in (self.pads, self.crate_lbl, self.crate_search, self.crate_stage, self.crate_note, self.crate_list):
            w.setVisible(remix)
        if remix and not self.pads._decks:
            self.pads.rebuild(())
        self.map_now.title, self.map_next.title = ("A", "B") if remix else ("PLAYING", "NEXT")
        self.phrase.set(0.0, "press START" if remix else "press START")
        for m in (self.map_now, self.map_next):
            m.update()

    # -- readout ---------------------------------------------------------------------------------------------------
    def _tick(self):
        # An exception inside a Qt timer slot aborts the whole process with no Python trace. The readout may never do that.
        try:
            self._midi_poll()
            self._apply_steer()
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
        self.state_lbl.setText(f"{st.get('state')}   ·   theme {st.get('theme')}   ·   persona {st.get('persona') or '-'}   ·   "
                               f"mix {st.get('mix_style') or 'auto'} ×{st.get('mix_speed', 1.0):g}   ·   pool {st.get('eligible_pool')}"
                               + (f"   ·   setlist {st['setlist']}" if st.get("setlist") else ""))
        cc = self._color(cur.get("title")) if cur else None
        nc = self._color(nxt.get("title")) if nxt else None
        win_now = (plan["out_s"] - plan["beats"] * 60.0 / max(cur.get("bpm") or 120.0, 1.0), plan["out_s"]) if plan and cur else None
        self.map_now.set(cur, st.get("track_map"), pos=cur.get("pos_s"), window=win_now, color=cc)
        win_next = (plan["in_s"], plan["in_s"] + plan["beats"] * 60.0 / max(nxt.get("bpm") or 120.0, 1.0)) if plan and nxt else None
        self.map_next.set(nxt, st.get("next_map"), window=win_next, color=nc)
        # the seam and the countdown
        bits = []
        if plan:
            bits.append(f"SEAM {plan['style']}  {plan['beats']} beats")
            bits.append(f"A exits {_mmss(plan['out_s'])}, B enters {_mmss(plan['in_s'])}")
            if abs(plan.get("rate", 1.0) - 1.0) > 1e-3 or abs(plan.get("a_rate", 1.0) - 1.0) > 1e-3:
                bits.append(f"tempo B ×{plan['rate']:.3f}" + (f", A ×{plan['a_rate']:.3f}" if abs(plan.get('a_rate', 1.0) - 1.0) > 1e-3 else ""))
            if plan.get("pitch_st"):
                bits.append(f"key shift {plan['pitch_st']:+d} st")
            if plan.get("pair_score") is not None:
                bits.append(f"pair {plan['pair_score']:.2f}")
            if st.get("mix_style"):
                bits.append(("pin honoured" + (f" (crossed {plan['pin_waived']})" if plan.get("pin_waived") else ""))
                            if plan.get("pin") else f"pin NOT honoured ({plan.get('pin_why_not') or 'gated'})")
            if plan.get("speed"):
                bits.append(f"length ×{plan['speed']['factor']:g}: {plan['speed']['beats']} → {plan['speed']['to']} beats")
            if plan.get("morph"):
                bits.append("morph " + ", ".join(f"{s} @{b:g}" for s, b in plan["morph"]))
            if st.get("seam_chips"):
                bits.append("why: " + ", ".join(str(c) for c in st["seam_chips"][:5]))
        if st.get("moment_flavor"):
            bits.append(f"moment {st['moment_flavor']} in {st.get('moment_eta')} s")
        if st.get("moment_denied"):
            bits.append(f"{st['moment_denied']['flavor']} refused: {st['moment_denied']['why']}")
        if st.get("layer"):
            bits.append(f"layer {st['layer']['label']} ({st['layer']['left_s']:.0f} s left)")
        self.seam_lbl.setText("   ·   ".join(bits))
        eta = st.get("blend_in_s")
        if plan and eta is not None:
            total = max(30.0, float(plan.get("beats", 32)) * 60.0 / max(cur.get("bpm") or 120.0, 1.0) + 30.0)
            self.phrase.set(1.0 - min(1.0, eta / total), f"{plan['style']} in {eta:.0f} s" + ("   · armed" if st.get("state") == "armed" else ""),
                            QColor(217, 122, 74) if st.get("state") == "armed" else None)
        elif st.get("state") == "playing":
            self.phrase.set(0.0, "seam not planned yet - the brain plans when the exit comes into range")
        else:
            self.phrase.set(0.0, str(st.get("state") or ""))
        self.buttons["ABORT MIX"].setEnabled(bool(st.get("abortable")))
        self.arc.set(st.get("arc_curve"), st.get("arc_phase"), st.get("arc_heat"), st.get("energy_nudge"))
        hz = st.get("horizon") or []
        if self.horizon_list.count() != len(hz) or any(self.horizon_list.item(i).text().split("  ")[0] != str(h.get("title", ""))[:40] for i, h in enumerate(hz)):
            self.horizon_list.clear()
            for h in hz:
                self.horizon_list.addItem(f"{str(h.get('title', ''))[:40]}  {h.get('bpm', 0) or 0:.0f} bpm {h.get('camelot') or ''}"
                                          + (f"  {h.get('why')}" if h.get("why") else ""))
        self.coming_lbl.setText("coming up" + (f"  ·  setlist {st['setlist']} as the pool" if st.get("setlist") else ""))
        evs = st.get("recent_events") or []
        if len(evs) < self._seen_events:
            self.event_list.clear()
            self._seen_events = 0
        for e in evs[self._seen_events:]:
            kind = e.get("event", "?")
            rest = {k: v for k, v in e.items() if k not in ("t", "clock_s", "event") and not isinstance(v, (dict, list))}
            txt = ", ".join(f"{k} {v}" for k, v in list(rest.items())[:6])
            self.event_list.insertItem(0, f"{time.strftime('%H:%M:%S', time.localtime(e.get('t', time.time())))}  {kind}  {txt}"[:180])
        self._seen_events = len(evs)
        decks = st.get("deck_tel") or {}
        d = "   ".join(f"{k.upper()}{'*' if k == st.get('active_deck') else ''} {'▶' if v.get('playing') else '·'} {_mmss(v.get('time_s'))} g{float(v.get('gain') or 0):.2f} ×{float(v.get('rate') or 1):.3f}"
                       + (" loop" if v.get("loop") else "") for k, v in decks.items())
        sync = st.get("sync") or {}
        ss = st.get("sync_stats") or {}
        lock = ""
        if sync:
            lock = f"\nsync {sync.get('slave')}→{sync.get('master')} bias {sync.get('bias_beats')} beats"
            if sync.get("audible_err_beats") is not None:
                lock += f", audible {sync['audible_err_beats']:+.3f}"
            lock += f", resnaps {ss.get('resnaps', 0)} nudges {ss.get('nudges', 0)}"
        err = getattr(self.system, "last_error", None)
        self.engine_lbl.setText(f"{d}{lock}\nlevel {st.get('level', 0):.2f}" + (f"   {self._midi_note}" if self._midi_note else "")
                                + (f"\nERROR {err}" if err else ""))

    def _tick_remix(self):
        rc = self.remix
        st = rc.status()
        songs = st.get("songs") or {}
        lanes = st.get("lanes") or {}
        ages = st.get("lane_age") or {}
        # the state chips
        self.state_lbl.setText(f"clock {st.get('master_bpm') or 0:.1f} bpm  ·  key {st.get('key_centre') or '?'}"
                               f"  ·  arc {100 * (st.get('arc_phase') or 0):.0f}%  energy {st.get('energy', 0):.2f}"
                               + (f" (lean {st['energy_lean']:+.2f})" if abs(st.get("energy_lean") or 0) > 0.005 else "")
                               + (f"  ·  tempo ±{100 * st['tempo_span']:.1f}% of {st.get('base_bpm') or 0:.1f}" if st.get("tempo_span") else "")
                               + (f"  ·  songs: {st['pool_name']} ({st['pool']})" if st.get("pool_name") else "  ·  songs: whole library")
                               + (f"  ·  {st['n_snapshots']} saved" if st.get("n_snapshots") else "")
                               + ("  ·  RECALL pending" if st.get("recalling") else "")
                               + (f"  ·  REC {st['recording']}" if st.get("recording") else "")
                               + f"  ·  verdicts {st.get('n_verdicts', 0)}")
        # the grid and the crate
        self.pads.render(st, self._color)
        cands = st.get("candidates") or []
        q = self.crate_search.text().strip()
        if q:
            cands = rc.candidates(12, query=q)
        sig = tuple(c["id"] for c in cands)
        if sig != getattr(self, "_crate_sig", None):
            self._crate_sig = sig
            cur = self.crate_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.crate_list.currentItem() else None
            self.crate_list.clear()
            for c in cands:
                it = QListWidgetItem(f"{c['title'][:44]}  ·  {c['artist'][:24] if c.get('artist') else ''}   {c['why']}   fit {c['score']:.2f}")
                it.setData(Qt.ItemDataRole.UserRole, c["id"])
                self.crate_list.addItem(it)
                if c["id"] == cur:
                    self.crate_list.setCurrentItem(it)
        self.crate_lbl.setText(f"CRATE  {len(cands)} fit the clock" + (f" in {st['pool_name']}" if st.get("pool_name") else ""))
        # the phrase bar
        cb = max(1, st.get("change_bars") or 8)
        pb = min(cb, st.get("phrase_bars") or 0)
        left = cb - 1 - pb
        if st.get("breaking"):
            self.phrase.set(1.0, "BREAK - the lanes come back on the bar", QColor(120, 85, 175), cb)
        elif st.get("hold"):
            self.phrase.set(pb / cb, "HOLD - the lane map is frozen (runway moves still happen)", QColor(190, 140, 50), cb)
        else:
            self.phrase.set(pb / cb, f"next move in {max(0, left)} bar{'s' if left != 1 else ''}"
                            + (f"   ·   LOOP {st['user_loop']}" if st.get("user_loop") else ""),
                            QColor(60, 150, 95) if st.get("user_loop") else None, cb)
        # the strips
        for deck, widget in self.deck_maps.items():
            s = songs.get(deck)
            if not s:
                widget.title = deck.upper()
                dec = (st.get("decoding") or {}).get(deck)
                widget.setVisible(bool(dec))
                if dec:
                    widget.set({"title": f"decoding {dec[:40]}…", "artist": "", "bpm": 0, "camelot": "", "duration_s": 1}, {"duration": 1, "sections": [], "energy": []}, color=DIM)
                continue
            widget.setVisible(True)
            col = self._color(s.get("id") or s.get("title"))
            chips = []
            if deck == st.get("master"):
                chips.append(("CLOCK", QColor(200, 200, 210)))
            if s.get("lanes"):
                chips.append((", ".join(s["lanes"]), col))
            elif not s.get("entered"):
                chips.append(("staged", QColor(150, 150, 165)))
            if s.get("section"):
                chips.append((s["section"], KIND_COLORS.get(s["section"], QColor(120, 120, 130)).lighter(140)))
            if s.get("singing"):
                chips.append(("singing", QColor(255, 230, 140)))
            if s.get("loop"):
                chips.append(("loop", QColor(120, 220, 130)))
            if s.get("leaving"):
                chips.append(("leaving", QColor(230, 120, 120)))
            if s.get("eq_low") is not None and s["eq_low"] < 0.6:
                chips.append(("lows cut", QColor(160, 160, 175)))
            if s.get("filter") and s["filter"] != "off":
                chips.append((f"{s['filter']} sweep", QColor(180, 160, 230)))
            if s.get("echo"):
                chips.append(("echo", QColor(180, 160, 230)))
            if s.get("lock_ms") is not None:
                chips.append((f"lock {s['lock_ms']:.0f} ms", QColor(150, 200, 255)))
            widget.title = deck.upper()
            lp = s.get("loop")
            win = (lp[0] / 44100.0, lp[1] / 44100.0) if lp else None
            widget.set(s, s.get("map"), pos=s.get("time_s"), window=win, chips=chips, color=col)
        lr = st.get("last_rated") or {}
        self.seam_lbl.setText((f"last move: {lr['text'][:90]}" + (f"  [{'GOOD' if lr['fb'] else 'BAD'}]" if lr.get("fb") is not None else "   ← rate it: 👍 / 👎")) if lr else "")
        # coming up
        coming = [f"decoding {t[:36]}" for t in (st.get("decoding") or {}).values()]
        coming += [f"{deck.upper()} staged: {songs[deck]['title'][:36]} (×{songs[deck]['rate']:.3f}, {songs[deck]['shift']:+d} st)"
                   for deck in songs if not songs[deck].get("entered")]
        pool = f"songs: {st['pool_name']} ({st['pool']} with stems)" if st.get("pool_name") else "songs: whole library"
        self.coming_lbl.setText(("coming up: " + "   ·   ".join(coming) if coming else "coming up: -") + "      " + pool)
        # the feed with thumbs on the move itself
        feed = st.get("feed") or []
        sig = tuple((m["id"], m.get("fb")) for m in feed)
        if sig != self._feed_sig:
            self._feed_sig = sig
            self.event_list.clear()
            for m in reversed(feed):
                item = QListWidgetItem()
                row = QWidget()
                h = QHBoxLayout(row)
                h.setContentsMargins(6, 1, 6, 1)
                lab = QLabel(f"<span style='color:#9a9aa6'>{m.get('hms') or ''}</span>  {m['text']}")
                lab.setTextFormat(Qt.TextFormat.RichText)
                lab.setStyleSheet("color: #e6e6ea; background: transparent;" + (" font-weight: 600;" if m["kind"] == "manual" else ""))
                h.addWidget(lab, 1)
                if m.get("rateable"):
                    for glyph, up in (("👍", True), ("👎", False)):
                        tb = QToolButton()
                        tb.setText(glyph)
                        tb.setCheckable(True)
                        tb.setChecked(m.get("fb") is up)
                        tb.clicked.connect(lambda _c, mid=m["id"], up=up: self._rate_move(mid, up))
                        h.addWidget(tb)
                item.setSizeHint(row.sizeHint())
                self.event_list.addItem(item)
                self.event_list.setItemWidget(item, row)
        # engine
        parts = []
        for deck, s in songs.items():
            parts.append(f"{deck.upper()} {'▶' if s.get('playing') else '·'} {_mmss(s.get('time_s'))} ×{s.get('rate', 1):.3f} {s.get('shift', 0):+d}st"
                         + (f" fit {s['compat']:.2f}" if s.get("compat") is not None else ""))
        err = st.get("error")
        self.engine_lbl.setText("   ".join(parts) + (f"\n{self._midi_note}" if self._midi_note else "") + (f"\nERROR {err}" if err else ""))
