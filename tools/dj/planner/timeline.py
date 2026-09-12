"""Timeline tab: one continuous run of bars, four lane tracks, songs' parts placed on them.

THE TIMELINE (top): a ruler in bars and the four lanes - drums, bass, other, vocals. A CLIP is a span of
one song's stem placed at a bar, drawn with that stem's real spectrogram tinted in the song's colour. One
clip per lane at a time (placing a clip ends what was there), so levels, beat lock, key shift and
crossfades are the conductor's and never yours. A clip runs until the next clip on its lane unless you
give it an end (drag its right edge). Drag a clip to move it; Delete removes the selected one; the
playhead runs left to right and keeps running - place the next thing when you want a change.

THE MATERIAL (bottom): the songs of your setlist (or a search of the library) and the SONG VIEWER: the
chosen song as four spectrogram rows, the beat grid, sections, the analyser's entry (green) and exit (red)
points. Drag from a stem row onto a lane, or from the title bar to place all four stems; SEND places the
song at the playhead. Everything the timeline plays goes through lib/dj/remix.py.
"""
import json
import time

from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF, QMimeData, QSize
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QPolygonF, QFont, QImage, QDrag, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                             QSlider, QSplitter, QVBoxLayout, QWidget, QSizePolicy)

from tools.dj.planner.perform import STYLE, SONG_PALETTE, KIND_COLORS, SetlistBox, _mmss

LANES = ("drums", "bass", "other", "vocals")
LANE_COLORS = {"drums": QColor(235, 235, 240), "bass": QColor(255, 170, 90), "other": QColor(140, 190, 255), "vocals": QColor(255, 225, 120)}
MIME = "application/x-dj-clip"
BG = QColor(18, 18, 22)
FG = QColor(230, 230, 236)
DIM = QColor(150, 150, 165)


def _spec_image(spec, col, t0, t1, w, h, hop_s):
    """A QImage of the spectrogram band `spec` (uint8 [bands, frames]) between song times t0..t1, tinted."""
    import numpy as np
    f0, f1 = int(max(0, t0 / hop_s)), int(min(spec.shape[1], max(t0 / hop_s + 1, t1 / hop_s)))
    if f1 <= f0 or w <= 0 or h <= 0:
        return None
    sl = spec[:, f0:f1]
    cols = np.linspace(0, sl.shape[1] - 1, max(1, min(w, sl.shape[1]))).astype(int)
    rows = np.linspace(sl.shape[0] - 1, 0, max(1, min(h, sl.shape[0]))).astype(int)      # low bands at the bottom
    a = sl[rows][:, cols].astype(np.float32) / 255.0
    r, g, b = col.red(), col.green(), col.blue()
    img = np.empty((a.shape[0], a.shape[1], 4), dtype=np.uint8)
    img[..., 0] = (18 + (b - 18) * a).astype(np.uint8)      # BGRA
    img[..., 1] = (18 + (g - 18) * a).astype(np.uint8)
    img[..., 2] = (18 + (r - 18) * a).astype(np.uint8)
    img[..., 3] = 255
    q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 4, QImage.Format.Format_ARGB32)
    return q.copy()


# ---------------------------------------------------------------------------------------------------------------
class TimelineCanvas(QWidget):
    """The ruler and the four lanes; clips as spectrogram slices; drag, trim, select, drop."""

    LANE_H = 64
    RULER_H = 22
    LEFT = 64
    readonly = False                     # the Director's picture: drawn, never edited

    def __init__(self, tab, lane_h=None, readonly=False):
        super().__init__()
        self.tab = tab
        if lane_h:
            self.LANE_H = lane_h
        self.readonly = readonly
        self.setMinimumHeight(self.RULER_H + 4 * self.LANE_H + 8)
        self.setAcceptDrops(not readonly)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.px_per_bar = 14.0
        self.first_bar = 0.0             # scroll
        self.selected = None
        self._drag = None                # ("move"|"trim", clip, grab_bar, orig)
        self._img_cache = {}
        self.setMouseTracking(True)

    # geometry
    def x_of(self, bar):
        return self.LEFT + (bar - self.first_bar) * self.px_per_bar

    def bar_at(self, x):
        return self.first_bar + (x - self.LEFT) / self.px_per_bar

    def lane_at(self, y):
        i = int((y - self.RULER_H) // self.LANE_H)
        return LANES[i] if 0 <= i < 4 else None

    def lane_y(self, lane):
        return self.RULER_H + LANES.index(lane) * self.LANE_H

    def visible_bars(self):
        return (self.width() - self.LEFT) / self.px_per_bar

    def follow(self, bar):
        """Keep the playhead in the middle third of the view."""
        vb = self.visible_bars()
        if bar < self.first_bar + vb * 0.1 or bar > self.first_bar + vb * 0.66:
            self.first_bar = max(0.0, bar - vb * 0.33)

    # painting
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, BG)
        tl = self.tab.timeline
        # lanes
        for ln in LANES:
            y = self.lane_y(ln)
            p.fillRect(QRectF(0, y, W, self.LANE_H), QColor(24, 24, 30) if LANES.index(ln) % 2 else QColor(20, 20, 26))
            p.setPen(DIM)
            f = QFont(self.font())
            f.setBold(True)
            f.setPointSizeF(10)
            p.setFont(f)
            p.drawText(QRectF(8, y, self.LEFT - 10, self.LANE_H), Qt.AlignmentFlag.AlignVCenter, ln.upper())
        # ruler + bar lines
        vb = int(self.visible_bars()) + 2
        b0 = int(self.first_bar)
        step = 4 if self.px_per_bar >= 9 else (8 if self.px_per_bar >= 4 else 16)
        p.setFont(QFont(self.font().family(), 8))
        for b in range(b0 - b0 % step, b0 + vb + step, step):
            x = self.x_of(b)
            if x < self.LEFT - 1:
                continue
            p.setPen(QColor(46, 46, 56))
            p.drawLine(QPointF(x, self.RULER_H), QPointF(x, H))
            p.setPen(DIM)
            p.drawText(QRectF(x + 3, 2, 60, self.RULER_H - 4), Qt.AlignmentFlag.AlignVCenter, str(b))
        # clips, as HEARD: per lane the segments where each clip is the one in charge (latest start wins,
        # an earlier clip resumes when a bounded one ends); ghosts (the autopilot's plan) dashed, rests dark
        spec = self.tab.spectro
        for ln in LANES:
            y = self.lane_y(ln)
            for c, s0, s1 in tl.segments(ln, int(self.first_bar), int(self.first_bar) + vb):
                x0, x1 = max(self.x_of(s0), self.LEFT), self.x_of(s1)
                if x1 < self.LEFT or x0 > W:
                    continue
                col = self.tab.color(c.track_id)
                rect = QRectF(x0, y + 3, max(2.0, x1 - x0), self.LANE_H - 6)
                if c.rest:
                    p.fillRect(rect, QColor(10, 10, 12))
                    p.setPen(QPen(QColor(70, 70, 80), 1, Qt.PenStyle.DashLine if c.ghost else Qt.PenStyle.SolidLine))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRect(rect)
                    p.setPen(DIM)
                    p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "rest" if rect.width() > 30 else "")
                    continue
                sp = spec.get(c.track_id)
                drawn = False
                if sp is not None and sp["stems"].get(ln) is not None and rect.width() > 4:
                    bar_s = self.tab.bar_s(c.track_id)
                    # STABLE PICTURES: the image is rendered on a fixed lattice of 8-bar chunks from the clip's
                    # own start, at the current zoom, and only the visible part is painted - a live clip that
                    # grows a pixel a tick used to be re-rendered at a new scale every tick (the spectrogram
                    # visibly "morphed", user 2026-09-12); scrolling and growth now only reveal more of it
                    CH = 8
                    k0 = int((s0 - c.bar) // CH)
                    k1 = int((min(s1, self.bar_at(W)) - c.bar) // CH)
                    p.save()
                    p.setClipRect(rect)
                    p.setOpacity(0.55 if c.ghost else 1.0)
                    for k in range(max(0, k0), max(0, k1) + 1):
                        cb0, cb1 = c.bar + k * CH, c.bar + (k + 1) * CH
                        cx0, cx1 = self.x_of(cb0), self.x_of(cb1)
                        if cx1 < x0 or cx0 > x1:
                            continue
                        w_px = max(1, int(round(cx1 - cx0)))
                        t0 = c.song_time_at(cb0, bar_s)
                        t1 = c.song_time_at(cb1, bar_s)
                        key = (c.id, k, round(self.px_per_bar, 2), int(rect.height()))
                        img = self._img_cache.get(key)
                        if img is None:
                            img = _spec_image(sp["stems"][ln], col, max(0.0, t0), max(0.0, t1), w_px, int(rect.height()), sp["hop_s"])
                            if len(self._img_cache) > 600:
                                self._img_cache.clear()
                            self._img_cache[key] = img
                        if img is not None:
                            p.drawImage(QRectF(cx0, rect.top(), cx1 - cx0, rect.height()), img)
                            drawn = True
                    p.setOpacity(1.0)
                    p.restore()
                if not drawn:
                    fill = QColor(col)
                    fill.setAlpha(40 if c.ghost else 70)
                    p.fillRect(rect, fill)
                pen = QPen(col if c is not self.selected else QColor(255, 255, 255), 2 if c is self.selected else 1)
                if c.ghost:
                    pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRect(rect)
                if c.bar == s0 and c.end_bar is None:
                    p.setPen(QPen(col, 1, Qt.PenStyle.DotLine))
                    p.drawLine(QPointF(rect.right(), rect.top()), QPointF(rect.right(), rect.bottom()))
                t = self.tab.track(c.track_id)
                if t is not None and rect.width() > 40 and c.bar == s0:
                    p.setPen(QColor(255, 255, 255) if not c.ghost else QColor(220, 220, 230))
                    f = QFont(self.font())
                    f.setPointSizeF(9)
                    f.setBold(True)
                    p.setFont(f)
                    p.drawText(QRectF(rect.left() + 4, rect.top() + 1, rect.width() - 8, 16), Qt.AlignmentFlag.AlignVCenter,
                               ("auto · " if c.ghost else "") + f"{t.title[:36]}  @{_mmss(c.start_s)}" + ("" if c.end_bar is None else f"  {c.end_bar - c.bar} bars"))
        # playhead
        ph = self.tab.playhead()
        if ph is not None:
            x = self.x_of(ph)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(QPointF(x, 0), QPointF(x, H))
        # drop hint
        if getattr(self, "_hover_drop", None):
            bar, lanes = self._hover_drop
            x = self.x_of(bar)
            p.setPen(QPen(QColor(120, 220, 120), 2, Qt.PenStyle.DashLine))
            for ln in lanes:
                y = self.lane_y(ln)
                p.drawLine(QPointF(x, y), QPointF(x, y + self.LANE_H))

    # interaction
    def _hit(self, pos):
        ln = self.lane_at(pos.y())
        if ln is None or pos.x() < self.LEFT:
            return None, None
        b = self.bar_at(pos.x())
        c = self.tab.timeline.active(ln, int(b))
        if c is None:
            return None, None
        edge = abs(self.x_of(c.end_bar) - pos.x()) < 6 if c.end_bar is not None else False
        return c, ("trim" if edge else "move")

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or self.readonly:
            return
        c, mode = self._hit(ev.position())
        self.selected = c
        if c is not None:
            self._drag = (mode, c, self.bar_at(ev.position().x()), c.bar, c.end_bar)
        elif ev.position().y() < self.RULER_H:
            self.tab.seek(int(round(self.bar_at(ev.position().x()))))
        self.update()

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            c, mode = self._hit(ev.position())
            self.setCursor(Qt.CursorShape.SizeHorCursor if mode == "trim" else (Qt.CursorShape.OpenHandCursor if c else Qt.CursorShape.ArrowCursor))
            return
        mode, c, grab, ob, oe = self._drag
        b = self.bar_at(ev.position().x())
        if mode == "move":
            nb = max(0, int(round(ob + (b - grab))))
            if nb != c.bar:
                self.tab.timeline.move(c, nb)
        else:
            ne = max(c.bar + 1, int(round(b)))
            c.end_bar = ne
            self.tab.timeline.resolve(c)
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag is not None:
            self._drag = None
            self.tab.changed()
        self.update()

    def mouseDoubleClickEvent(self, ev):
        if self.readonly:
            return
        c, mode = self._hit(ev.position())
        if c is not None:
            c.end_bar = None if c.end_bar is not None else max(c.bar + 4, int(round(self.bar_at(ev.position().x()))))
            self.tab.changed()
            self.update()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and self.selected is not None and not self.readonly:
            self.tab.timeline.remove(self.selected)
            self.selected = None
            self.tab.changed()
            self.update()
        else:
            super().keyPressEvent(ev)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            f = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
            bar_under = self.bar_at(ev.position().x())
            self.px_per_bar = max(2.0, min(80.0, self.px_per_bar * f))
            self.first_bar = max(0.0, bar_under - (ev.position().x() - self.LEFT) / self.px_per_bar)
        else:
            self.first_bar = max(0.0, self.first_bar - ev.angleDelta().y() / 120.0 * 4)
        self._img_cache.clear()
        self.update()

    # drops from the viewer
    def dragEnterEvent(self, ev):
        if ev.mimeData().hasFormat(MIME):
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        if not ev.mimeData().hasFormat(MIME):
            return
        d = json.loads(bytes(ev.mimeData().data(MIME)).decode("utf-8"))
        ln = self.lane_at(ev.position().y())
        lanes = d["lanes"] if len(d["lanes"]) > 1 else ([ln] if ln else [])
        self._hover_drop = (max(0, int(round(self.bar_at(ev.position().x())))), lanes)
        ev.acceptProposedAction()
        self.update()

    def dragLeaveEvent(self, ev):
        self._hover_drop = None
        self.update()

    def dropEvent(self, ev):
        d = json.loads(bytes(ev.mimeData().data(MIME)).decode("utf-8"))
        ln = self.lane_at(ev.position().y())
        lanes = d["lanes"] if len(d["lanes"]) > 1 else ([ln] if ln else [])
        bar = max(0, int(round(self.bar_at(ev.position().x()))))
        self._hover_drop = None
        if lanes:
            self.tab.place(d["track_id"], lanes, bar, d["start_s"])
        ev.acceptProposedAction()
        self.update()


# ---------------------------------------------------------------------------------------------------------------
class SongViewer(QWidget):
    """One song large: four spectrogram rows in song time, the beat grid, sections, entry and exit points.
    Drag from a row (one stem) or the title band (all stems) onto the timeline."""

    ROW_H = 58
    HEAD_H = 24

    def __init__(self, tab):
        super().__init__()
        self.tab = tab
        self.track = None
        self.setMinimumHeight(self.HEAD_H + 4 * self.ROW_H + 16)
        self.setMouseTracking(True)
        self._press = None
        self._img_cache = {}
        self.hover_t = None

    def set_track(self, t):
        self.track = t
        self._img_cache.clear()
        if t is not None:
            self.tab.spectro.request(t)
        self.update()

    def t_of(self, x):
        if self.track is None:
            return 0.0
        return max(0.0, min(self.track.duration_s, (x - 8) / max(1, self.width() - 16) * self.track.duration_s))

    def x_of(self, t):
        return 8 + t / max(self.track.duration_s, 1e-6) * (self.width() - 16)

    def paintEvent(self, ev):
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, BG)
        t = self.track
        if t is None:
            p.setPen(DIM)
            p.drawText(QRectF(10, 0, W - 20, H), Qt.AlignmentFlag.AlignVCenter, "choose a song on the left")
            return
        col = self.tab.color(t.id)
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(11)
        p.setFont(f)
        p.setPen(col)
        p.drawText(QRectF(10, 0, W - 20, self.HEAD_H), Qt.AlignmentFlag.AlignVCenter,
                   f"{t.title}   {t.artist or ''}   {t.bpm:.1f} bpm  {t.camelot or '?'}   {_mmss(t.duration_s)}      ← drag this bar for all stems, a row for one")
        y0 = self.HEAD_H
        sp = self.tab.spectro.get(t.id)
        bar = 4 * t.period_s
        for i, ln in enumerate(LANES):
            y = y0 + i * self.ROW_H
            rect = QRectF(8, y + 2, W - 16, self.ROW_H - 4)
            drawn = False
            if sp is not None and sp["stems"].get(ln) is not None:
                key = (ln, int(rect.width()), int(rect.height()))
                img = self._img_cache.get(key)
                if img is None:
                    img = _spec_image(sp["stems"][ln], LANE_COLORS[ln], 0.0, t.duration_s, int(rect.width()), int(rect.height()), sp["hop_s"])
                    self._img_cache[key] = img
                if img is not None:
                    p.drawImage(rect, img)
                    drawn = True
            if not drawn:
                p.fillRect(rect, QColor(28, 28, 34))
                p.setPen(DIM)
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, "analysing…" if t.id in self.tab.spectro._busy or sp is None else "")
            p.setPen(QColor(20, 20, 24, 200))
            p.setFont(QFont(self.font().family(), 8, QFont.Weight.Bold))
            p.drawText(QRectF(12, y + 2, 80, 14), Qt.AlignmentFlag.AlignVCenter, ln.upper())
        # sections (thin band under the head), the 4-bar grid, entry / exit points
        for s in t.sections or []:
            c = KIND_COLORS.get(s.get("kind"), QColor(90, 90, 100))
            p.fillRect(QRectF(self.x_of(s["start_s"]), y0 - 5, max(1.0, self.x_of(s["end_s"]) - self.x_of(s["start_s"])), 4), c)
        p.setPen(QPen(QColor(255, 255, 255, 35), 1))
        k = 0
        tt = t.nearest_downbeat(0.0)
        while tt < t.duration_s:
            if k % 4 == 0:
                x = self.x_of(tt)
                p.drawLine(QPointF(x, y0), QPointF(x, H - 8))
            tt += bar
            k += 1
        for pt in sorted(t.mix_ins or [], key=lambda q: -q.get("score", 0))[:4]:
            x = self.x_of(pt["time_s"])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(120, 220, 120)))
            p.drawPolygon(QPolygonF([QPointF(x - 6, y0), QPointF(x + 6, y0), QPointF(x, y0 + 9)]))
        for pt in sorted(t.mix_outs or [], key=lambda q: -q.get("score", 0))[:4]:
            x = self.x_of(pt["time_s"])
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(255, 120, 90)))
            p.drawPolygon(QPolygonF([QPointF(x - 6, H - 2), QPointF(x + 6, H - 2), QPointF(x, H - 11)]))
        # the song's position when it is on a deck
        pos = self.tab.song_pos(t.id)
        if pos is not None:
            x = self.x_of(pos)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(QPointF(x, y0), QPointF(x, H))
        if self.hover_t is not None:
            x = self.x_of(self.hover_t)
            p.setPen(QPen(QColor(255, 255, 255, 120), 1))
            p.drawLine(QPointF(x, y0), QPointF(x, H))
            p.setPen(FG)
            p.setFont(QFont(self.font().family(), 8))
            p.drawText(QRectF(x + 4, H - 16, 90, 14), Qt.AlignmentFlag.AlignVCenter, _mmss(self.hover_t))

    def mousePressEvent(self, ev):
        if self.track is None or ev.button() != Qt.MouseButton.LeftButton:
            return
        y = ev.position().y()
        if y < self.HEAD_H:
            lanes = list(LANES)
        else:
            i = int((y - self.HEAD_H) // self.ROW_H)
            if not 0 <= i < 4:
                return
            lanes = [LANES[i]]
        self._press = (ev.position(), lanes)

    def mouseMoveEvent(self, ev):
        if self.track is not None:
            self.hover_t = self.t_of(ev.position().x())
            self.update()
        if self._press is None:
            return
        pos, lanes = self._press
        if (ev.position() - pos).manhattanLength() < 8:
            return
        t_start = self.track.nearest_downbeat(self.t_of(pos.x()))
        self._press = None
        mime = QMimeData()
        mime.setData(MIME, json.dumps({"track_id": self.track.id, "lanes": lanes, "start_s": t_start}).encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)

    def mouseReleaseEvent(self, ev):
        self._press = None

    def leaveEvent(self, ev):
        self.hover_t = None
        self.update()


# ---------------------------------------------------------------------------------------------------------------
class TimelineTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.setObjectName("perform")
        self.setStyleSheet(STYLE)
        self.planner = planner
        from lib.dj.timeline import Timeline, Spectro
        self.timeline = Timeline()
        self.spectro = Spectro(planner.music_dir)
        self.engine = None
        self.rc = None
        self.player = None
        self._colors = {}
        self._own_db = None
        self._lib = None
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        # -- transport ------------------------------------------------------------------------------------
        top = QHBoxLayout()
        self.play_btn = QPushButton("▶  PLAY")
        self.play_btn.setProperty("kind", "go")
        self.play_btn.setMinimumWidth(120)
        self.play_btn.clicked.connect(self._toggle)
        top.addWidget(self.play_btn)
        self.pos_lbl = QLabel("bar 0")
        f = self.pos_lbl.font()
        f.setPointSizeF(13)
        f.setBold(True)
        self.pos_lbl.setFont(f)
        self.pos_lbl.setMinimumWidth(110)
        top.addWidget(self.pos_lbl)
        self.songs_box = SetlistBox(self)
        self.songs_box.setMinimumWidth(220)
        self.songs_box.currentIndexChanged.connect(self._fill_songs)
        top.addWidget(self.songs_box)
        self.search = QLineEdit()
        self.search.setPlaceholderText("search the library")
        self.search.setMaximumWidth(220)
        self.search.textChanged.connect(self._fill_songs)
        top.addWidget(self.search)
        top.addSpacing(10)
        top.addWidget(QLabel("timeline"))
        self.name_edit = QLineEdit("untitled")
        self.name_edit.setMaximumWidth(160)
        top.addWidget(self.name_edit)
        for label, fn in (("SAVE", self._save), ("LOAD", self._load), ("CLEAR", self._clear)):
            b = QPushButton(label)
            b.setProperty("kind", "small")
            b.clicked.connect(fn)
            top.addWidget(b)
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(2, 80)
        self.zoom.setValue(14)
        self.zoom.setMaximumWidth(140)
        self.zoom.setToolTip("zoom (or Ctrl+wheel on the timeline)")
        self.zoom.valueChanged.connect(self._zoom)
        top.addWidget(QLabel("zoom"))
        top.addWidget(self.zoom)
        self.state_lbl = QLabel("")
        self.state_lbl.setProperty("dim", "true")
        top.addWidget(self.state_lbl, 1)
        root.addLayout(top)
        # -- the timeline ---------------------------------------------------------------------------------------
        self.canvas = TimelineCanvas(self)
        root.addWidget(self.canvas)
        self.hint = QLabel("drag a stem row or a song's title bar from the viewer onto a lane · drag a clip to move it · drag its right edge to end it · "
                           "double-click: end here / run until replaced · Delete removes · click the ruler to set where PLAY starts")
        self.hint.setProperty("dim", "true")
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)
        # -- live gestures: the system writes the clips, you say when ------------------------------------------
        live = QHBoxLayout()
        live.setSpacing(8)
        self.gestures = {}
        for label, fn, tip, kind in (
                ("NEXT SONG", self._next_song, "bring the chosen song in the way the conductor would: a lane a phrase from the next phrase  [N]", "go"),
                ("DROP", self._drop, "every lane to the chosen (else the newest) song on the next bar  [D]", "hot"),
                ("BREAK", self._break, "every lane but the most melodic rests four bars from the next bar  [B]", None),
                ("HOLD", self._hold, "no new autopilot plans while held (what is placed still plays)  [H]", None)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setMinimumHeight(40)
            if kind:
                b.setProperty("kind", kind)
            b.clicked.connect(fn)
            live.addWidget(b)
            self.gestures[label] = b
        self.gestures["HOLD"].setCheckable(True)
        live.addSpacing(16)
        lab = QLabel("auto")
        lab.setProperty("dim", "true")
        live.addWidget(lab)
        self.auto = QSlider(Qt.Orientation.Horizontal)
        self.auto.setRange(0, 100)
        self.auto.setValue(100)
        self.auto.setMaximumWidth(180)
        self.auto.setToolTip("autopilot: how often the system plans the next phrase's move itself, written ahead as ghost clips you can delete or move - 0 = only what you place plays")
        self.auto.valueChanged.connect(self._auto)
        live.addWidget(self.auto)
        self.auto_lbl = QLabel("100%")
        self.auto_lbl.setMinimumWidth(40)
        live.addWidget(self.auto_lbl)
        lab2 = QLabel("change")
        lab2.setProperty("dim", "true")
        live.addWidget(lab2)
        self.change_box = QComboBox()
        self.change_box.addItems(["every 4 bars", "every 8 bars", "every 16 bars", "every 32 bars"])
        self.change_box.setCurrentIndex(1)
        self.change_box.currentIndexChanged.connect(self._change)
        live.addWidget(self.change_box)
        live.addStretch(1)
        root.addLayout(live)
        for seq, fn in (("N", self._next_song), ("D", self._drop), ("B", self._break), ("H", lambda: (self.gestures["HOLD"].toggle(), self._hold()))):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        # -- the material ---------------------------------------------------------------------------------------
        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.song_list = QListWidget()
        self.song_list.currentItemChanged.connect(self._song_chosen)
        lv.addWidget(self.song_list)
        row = QHBoxLayout()
        self.send_btn = QPushButton("SEND at playhead")
        self.send_btn.setProperty("kind", "small")
        self.send_btn.setToolTip("place all four stems of the chosen song at the playhead (from its best entry point)")
        self.send_btn.clicked.connect(self._send)
        row.addWidget(self.send_btn)
        lv.addLayout(row)
        split.addWidget(left)
        self.viewer = SongViewer(self)
        split.addWidget(self.viewer)
        split.setSizes([320, 1100])
        root.addWidget(split, 1)
        self.notes_lbl = QLabel("")
        self.notes_lbl.setProperty("dim", "true")
        self.notes_lbl.setWordWrap(True)
        root.addWidget(self.notes_lbl)
        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._play_from = 0
        QTimer.singleShot(1500, self._fill_songs)

    # -- data ---------------------------------------------------------------------------------------------------
    def _db(self):
        db = getattr(self.planner, "db", None)
        if db is not None:
            return db
        if self._own_db is None:
            from lib.dj.db import LibraryDB
            self._own_db = LibraryDB(self.planner.music_dir)
        return self._own_db

    def library(self):
        lib = getattr(self.planner, "library", None)
        if lib:
            return [t for t in lib if getattr(t, "has_stems", False)]
        if self._lib is None:
            from lib.dj import brain as B
            self._lib = [t for t in B.load_library(self._db()) if not t.excluded and getattr(t, "has_stems", False)]
        return self._lib

    def track(self, tid):
        if self.rc is not None:
            t = self.rc._track(tid)
            if t is not None:
                return t
        return next((t for t in self.library() if t.id == tid), None)

    def bar_s(self, tid):
        t = self.track(tid)
        return 4 * t.period_s if t is not None else 2.0

    def color(self, tid):
        if tid not in self._colors:
            self._colors[tid] = SONG_PALETTE[len(self._colors) % len(SONG_PALETTE)]
        return self._colors[tid]

    def playhead(self):
        if self.player is not None and self.player.running and self.rc.master is not None:
            frac = 0.0
            d = self.rc._tel_deck(self.rc.master)
            ms = self.rc.songs.get(self.rc.master)
            if ms is not None and d.get("playing"):
                bar = self.rc._bar_s(ms.track)
                frac = (float(d.get("time_s", 0.0)) % bar) / bar
            return self.player.bar() + frac
        return float(self._play_from)

    def song_pos(self, tid):
        if self.rc is None:
            return None
        d = self.rc.deck_of(tid)
        if d is None:
            return None
        return self.rc._tel_deck(d).get("time_s")

    # -- material ------------------------------------------------------------------------------------------------
    def _fill_songs(self, *_):
        try:
            lib = self.library()
        except Exception as e:  # noqa: BLE001
            self.state_lbl.setText(f"library: {e}")
            return
        name = self.songs_box.currentData()
        ids = None
        if name:
            from lib.dj.setlist import get_setlist
            sl = get_setlist(self._db(), name=name)
            ids = [e["track_id"] for e in (sl or {}).get("entries", [])]
        q = self.search.text().strip().lower()
        by_id = {t.id: t for t in lib}
        rows = [by_id[i] for i in ids if i in by_id] if ids is not None else sorted(lib, key=lambda t: t.title.lower())
        if q:
            rows = [t for t in rows if q in t.title.lower() or q in (t.artist or "").lower()]
        cur = self.song_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.song_list.currentItem() else None
        self.song_list.clear()
        for t in rows[:400]:
            it = QListWidgetItem(f"{t.title[:40]}   {t.bpm:.0f} bpm  {t.camelot or '?'}")
            it.setData(Qt.ItemDataRole.UserRole, t.id)
            self.song_list.addItem(it)
            if t.id == cur:
                self.song_list.setCurrentItem(it)
        self.state_lbl.setText(f"{len(rows)} songs with stems" + (f" in {name}" if name else ""))

    def _song_chosen(self, cur, _prev):
        if cur is None:
            return
        self.viewer.set_track(self.track(cur.data(Qt.ItemDataRole.UserRole)))

    def _send(self):
        t = self.viewer.track
        if t is None:
            return
        start = self._best_entry(t)
        self.place(t.id, list(LANES), int(round(self.playhead())) + 1, start)

    def _best_entry(self, t):
        ins = sorted(t.mix_ins or [], key=lambda q: -q.get("score", 0))
        if ins:
            return t.nearest_downbeat(float(ins[0]["time_s"]))
        for s in t.sections or []:
            if s.get("kind") == "groove":
                return t.nearest_downbeat(s["start_s"])
        return t.nearest_downbeat(0.0)

    def place(self, track_id, lanes, bar, start_s):
        self.timeline.add(track_id, lanes, bar, start_s)
        t = self.track(track_id)
        if t is not None:
            self.spectro.request(t)
        self.changed()

    def changed(self):
        self.canvas.update()

    # -- live gestures ----------------------------------------------------------------------------------------------
    def _chosen_id(self):
        it = self.song_list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def _live_player(self):
        """The player, or a planning-only one before PLAY (gestures write clips either way)."""
        if self.player is not None:
            return self.player
        return None

    def _next_song(self):
        tid = self._chosen_id()
        if tid is None:
            self.state_lbl.setText("choose a song on the left first")
            return
        pl = self._live_player()
        if pl is None:
            from lib.dj.timeline import MORPH_ORDER
            t = self.track(tid)
            land = self._best_entry(t)
            bars = [4, 8, 16, 32][self.change_box.currentIndex()]
            for i, ln in enumerate(MORPH_ORDER):
                self.timeline.add(tid, [ln], self._play_from + (i + 1) * bars, land + i * bars * self.bar_s(tid), ghost=True)
        else:
            pl.next_song(tid)
        self.spectro.request(self.track(tid))
        self.changed()

    def _drop(self):
        pl = self._live_player()
        if pl is not None:
            pl.drop(self._chosen_id() if self._chosen_id() in self.timeline.tracks() or self.rc.deck_of(self._chosen_id() or -1) else None)
            self.changed()

    def _break(self):
        pl = self._live_player()
        if pl is not None:
            pl.break_()
            self.changed()

    def _hold(self):
        if self.player is not None:
            self.player.hold = self.gestures["HOLD"].isChecked()

    def _auto(self, v):
        self.auto_lbl.setText(f"{v}%")
        if self.player is not None:
            self.player.auto = v / 100.0

    def _change(self, i):
        bars = [4, 8, 16, 32][i]
        if self.player is not None:
            self.player.change_bars = bars

    # -- files ------------------------------------------------------------------------------------------------------
    def _save(self):
        path = self.timeline.save(self.planner.music_dir, self.name_edit.text().strip() or "untitled")
        self.state_lbl.setText(f"saved {path}")

    def _load(self):
        from lib.dj.timeline import Timeline
        name = self.name_edit.text().strip()
        names = Timeline.names(self.planner.music_dir)
        if name not in names:
            self.state_lbl.setText("timelines: " + (", ".join(names) if names else "none saved yet") + "  (type a name, then LOAD)")
            return
        self.timeline = Timeline.load(self.planner.music_dir, name)
        for tid in self.timeline.tracks():
            t = self.track(tid)
            if t is not None:
                self.spectro.request(t)
        self.canvas.selected = None
        self.changed()

    def _clear(self):
        from lib.dj.timeline import Timeline
        self.timeline = Timeline(self.name_edit.text().strip() or "untitled")
        self.canvas.selected = None
        self.changed()

    def _zoom(self, v):
        self.canvas.px_per_bar = float(v)
        self.canvas._img_cache.clear()
        self.canvas.update()

    def seek(self, bar):
        if self.player is None or not self.player.running:
            self._play_from = max(0, bar)
            self.canvas.update()

    # -- play ----------------------------------------------------------------------------------------------------------
    def _toggle(self):
        if self.player is None:
            try:
                self._start()
            except Exception as e:  # noqa: BLE001
                self.state_lbl.setText(f"could not start: {type(e).__name__}: {e}")
                self.close()
        else:
            self.close()

    def _start(self):
        from lib.audio_engine import AudioEngine
        from lib.dj.db import LibraryDB
        from lib.dj import brain as B
        from lib.dj.remix import RemixConductor
        from lib.dj.timeline import TimelinePlayer
        if not self.timeline.clips and self.auto.value() <= 0:
            self.state_lbl.setText("place something on the timeline first, or raise the autopilot")
            return
        try:
            self.planner.analysis_tab.player.stop()
        except Exception:
            pass
        self.engine = AudioEngine()
        self.engine.start()
        db = LibraryDB(self.planner.music_dir)
        lib = [t for t in B.load_library(db) if not t.excluded]
        self.rc = RemixConductor(db, self.planner.music_dir, lib, theme="groove")
        self.engine.attach_track("dj_timeline", self.rc.submix)
        self.player = TimelinePlayer(self.rc, self.timeline)
        self.player.auto = self.auto.value() / 100.0
        self.player.change_bars = [4, 8, 16, 32][self.change_box.currentIndex()]
        self.player.hold = self.gestures["HOLD"].isChecked()
        ok, msg = self.player.start(self._play_from)
        if not ok:
            self.state_lbl.setText(msg or "could not start")
            self.close()
            return
        self.play_btn.setText("■  STOP")
        self.play_btn.setProperty("kind", "hot")
        self.play_btn.style().unpolish(self.play_btn)
        self.play_btn.style().polish(self.play_btn)

    def close(self):
        if self.player is not None:
            self.player.running = False
            self.player = None
        if self.rc is not None:
            try:
                self.rc.stop(fade_s=1.0)
                time.sleep(1.2)
            except Exception:
                pass
            self.rc = None
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None
        self.play_btn.setText("▶  PLAY")
        self.play_btn.setProperty("kind", "go")
        self.play_btn.style().unpolish(self.play_btn)
        self.play_btn.style().polish(self.play_btn)

    def _tick(self):
        try:
            if self.player is not None and self.player.running:
                self.player.step()
                b = self.playhead()
                self.canvas.follow(b)
                self.pos_lbl.setText(f"bar {int(b)}.{int((b % 1) * 4) + 1}")
                st = self.rc.status()
                songs = st.get("songs") or {}
                lanes = st.get("lanes") or {}
                self.state_lbl.setText(f"clock {st.get('master_bpm') or 0:.1f} bpm · key {st.get('key_centre') or '?'} · "
                                       + "  ".join(f"{ln}: {(songs.get(d) or {}).get('title', '-')[:18] if d else '-'}" for ln, d in lanes.items()))
                notes = self.player.notes[-2:]
                self.notes_lbl.setText("   ·   ".join(f"{t} {m}" for t, m in notes) + (f"   ERROR {st['error']}" if st.get("error") else ""))
            else:
                self.pos_lbl.setText(f"bar {self._play_from}")
            # every song on the timeline gets its spectrogram (the autopilot's picks included)
            for tid in self.timeline.tracks():
                if self.spectro.get(tid) is None and tid not in self.spectro._busy and tid not in self.spectro.errors:
                    t = self.track(tid)
                    if t is not None:
                        self.spectro.request(t)
            self.canvas.update()
            if self.viewer.track is not None and (self.spectro.get(self.viewer.track.id) is not None or self.rc is not None):
                self.viewer.update()
        except Exception as e:  # noqa: BLE001
            self.notes_lbl.setText(f"readout error: {type(e).__name__}: {e}")
