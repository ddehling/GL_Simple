"""Zoomable waveform view for the Analysis tab.

Shows the whole song's shape and lets you dive to beat level:
    - min/max peak waveform (pyramid-downsampled for speed)
    - section bands colored by kind (top strip)
    - vocal-likely regions (red underline strip)
    - beat grid when zoomed in (downbeats stronger)
    - cue flags (in=green / out=red / interest=yellow; auto ones hollow)
    - playhead; click = seek; wheel = cursor-centered zoom; drag = pan

Signals: seekRequested(float seconds), cueClicked(dict).
"""
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

RATE = 44100

SECTION_COLORS = {
    "intro": QColor(70, 90, 130), "outro": QColor(70, 90, 130),
    "steady": QColor(55, 115, 85), "drop": QColor(185, 80, 55),
    "build": QColor(185, 150, 55), "breakdown": QColor(95, 70, 135),
}
CUE_COLORS = {"in": QColor(90, 220, 120), "out": QColor(240, 100, 90),
              "interest": QColor(240, 210, 90)}


class WaveformView(QWidget):
    seekRequested = pyqtSignal(float)
    cueClicked = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.track = None            # TrackInfo
        self.cues = []
        self.duration = 1.0
        self.view_t0 = 0.0           # visible window (seconds)
        self.view_t1 = 1.0
        self.playhead = 0.0
        self._pyramid = []           # [(spp, min[], max[]), ...] coarse->fine
        self._drag_x = None
        self._drag_t0 = None

    # -- data -------------------------------------------------------------
    def set_track(self, track, samples_mono, cues):
        """samples_mono: float32 mono array (decoded off-thread)."""
        self.track = track
        self.cues = list(cues or [])
        n = len(samples_mono)
        self.duration = max(n / RATE, 0.001)
        self._pyramid = []
        x = samples_mono
        spp = 1
        # Build peak pyramids at 64/512/4096 samples-per-pixel-ish levels.
        for factor in (64, 8, 8):
            spp *= factor
            m = n // spp
            if m < 4:
                break
            r = x[:m * spp].reshape(m, spp) if spp <= 64 else None
            if r is not None:
                mn, mx = r.min(axis=1), r.max(axis=1)
            else:
                prev = self._pyramid[-1]
                pm = len(prev[1]) // factor
                mn = prev[1][:pm * factor].reshape(pm, factor).min(axis=1)
                mx = prev[2][:pm * factor].reshape(pm, factor).max(axis=1)
            self._pyramid.append((spp, mn.astype(np.float32),
                                  mx.astype(np.float32)))
        self.view_t0, self.view_t1 = 0.0, self.duration
        self.playhead = 0.0
        self.update()

    def set_cues(self, cues):
        self.cues = list(cues or [])
        self.update()

    def set_playhead(self, t, follow=True):
        self.playhead = t
        if follow and not (self.view_t0 <= t <= self.view_t1) \
                and self.view_t1 - self.view_t0 < self.duration * 0.999:
            span = self.view_t1 - self.view_t0
            self.view_t0 = max(0.0, t - span * 0.2)
            self.view_t1 = min(self.duration, self.view_t0 + span)
        self.update()

    # -- coords ---------------------------------------------------------------
    def _t2x(self, t):
        span = max(self.view_t1 - self.view_t0, 1e-6)
        return (t - self.view_t0) / span * self.width()

    def _x2t(self, x):
        span = max(self.view_t1 - self.view_t0, 1e-6)
        return self.view_t0 + x / max(self.width(), 1) * span

    # -- painting ---------------------------------------------------------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(18, 18, 22))
        if self.track is None or not self._pyramid:
            p.setPen(QColor(120, 120, 130))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "select a track (Library tab → 'Open in Analysis')")
            return
        W, H = self.width(), self.height()
        strip_h = 16                 # section strip
        voc_h = 8                    # vocal strip
        wave_top = strip_h + 2
        wave_h = H - strip_h - voc_h - 6
        mid_y = wave_top + wave_h / 2

        # Sections strip + translucent wash over the waveform.
        for s in (self.track.sections or []):
            x0, x1 = self._t2x(s["start_s"]), self._t2x(s["end_s"])
            if x1 < 0 or x0 > W:
                continue
            col = SECTION_COLORS.get(s["kind"], QColor(85, 85, 90))
            p.fillRect(QRectF(x0, 0, x1 - x0, strip_h), col)
            wash = QColor(col)
            wash.setAlpha(26)
            p.fillRect(QRectF(x0, wave_top, x1 - x0, wave_h), wash)
            if x1 - x0 > 46:
                p.setPen(QColor(235, 235, 235, 200))
                p.drawText(QRectF(x0 + 3, 0, x1 - x0 - 4, strip_h),
                           Qt.AlignmentFlag.AlignVCenter, s["kind"])
            # Vocal-likely underline.
            if s.get("vocalness", 0) > 0.55:
                p.fillRect(QRectF(x0, H - voc_h - 2, x1 - x0, voc_h),
                           QColor(230, 90, 110, 190))

        # Beat grid when zoomed in enough.
        span = self.view_t1 - self.view_t0
        grid = self.track.grid or []
        if grid and span > 0:
            period = grid[0]["period_s"]
            px_per_beat = W / (span / period)
            if px_per_beat >= 5:
                for g in grid:
                    first_down = g["first_beat_s"] \
                        + self.track.downbeat_offset * g["period_s"]
                    k0 = int((self.view_t0 - g["first_beat_s"])
                             / g["period_s"]) - 1
                    t = g["first_beat_s"] + max(k0, 0) * g["period_s"]
                    while t <= min(self.view_t1, g["end_s"]):
                        if t >= max(self.view_t0, g["start_s"]):
                            x = self._t2x(t)
                            beats_from_down = round(
                                (t - first_down) / g["period_s"])
                            is_down = beats_from_down % 4 == 0
                            p.setPen(QPen(QColor(255, 255, 255,
                                                 70 if is_down else 28), 1))
                            p.drawLine(int(x), wave_top, int(x),
                                       wave_top + wave_h)
                        t += g["period_s"]

        # Waveform from the best-fitting pyramid level.
        spp_needed = span * RATE / max(W, 1)
        level = self._pyramid[0]
        for lv in self._pyramid:
            if lv[0] <= spp_needed:
                level = lv
        spp, mn, mx = level
        p.setPen(QPen(QColor(140, 200, 235, 210), 1))
        i0 = int(self.view_t0 * RATE / spp)
        for x in range(W):
            t = self._x2t(x)
            j0 = int(t * RATE / spp)
            j1 = max(int((t + span / W) * RATE / spp), j0 + 1)
            j0 = np.clip(j0, 0, len(mn) - 1)
            j1 = np.clip(j1, j0 + 1, len(mn))
            lo, hi = float(mn[j0:j1].min()), float(mx[j0:j1].max())
            p.drawLine(x, int(mid_y - hi * wave_h * 0.48),
                       x, int(mid_y - lo * wave_h * 0.48))

        # Cues.
        for c in self.cues:
            x = self._t2x(c["time_s"])
            if x < -4 or x > W + 4:
                continue
            col = CUE_COLORS.get(c["kind"], QColor(200, 200, 200))
            solid = c.get("source") == "user"
            p.setPen(QPen(col, 2))
            p.drawLine(int(x), wave_top, int(x), wave_top + wave_h)
            flag = QRectF(x, wave_top, 46, 14)
            if solid:
                p.fillRect(flag, col)
                p.setPen(QColor(10, 10, 10))
            else:
                p.setPen(QPen(col, 1))
                p.drawRect(flag)
            p.drawText(flag, Qt.AlignmentFlag.AlignCenter,
                       (c.get("label") or c["kind"])[:7])

        # Playhead.
        x = self._t2x(self.playhead)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(int(x), 0, int(x), H)

        # Time ruler labels.
        p.setPen(QColor(160, 160, 170))
        step = _nice_step(span)
        t = (int(self.view_t0 / step) + 1) * step
        while t < self.view_t1:
            x = self._t2x(t)
            p.drawText(int(x) + 3, H - voc_h - 6,
                       f"{int(t // 60)}:{int(t % 60):02d}")
            t += step

    # -- interaction -----------------------------------------------------------
    def wheelEvent(self, ev):
        if self.track is None:
            return
        t_cursor = self._x2t(ev.position().x())
        factor = 0.8 if ev.angleDelta().y() > 0 else 1.25
        span = (self.view_t1 - self.view_t0) * factor
        span = max(min(span, self.duration), 0.5)
        frac = (t_cursor - self.view_t0) / max(self.view_t1 - self.view_t0,
                                               1e-6)
        self.view_t0 = max(0.0, t_cursor - frac * span)
        self.view_t1 = min(self.duration, self.view_t0 + span)
        self.view_t0 = max(0.0, self.view_t1 - span)
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_x = ev.position().x()
            self._drag_t0 = self.view_t0
            self._moved = False

    def mouseMoveEvent(self, ev):
        if self._drag_x is not None:
            dx = ev.position().x() - self._drag_x
            if abs(dx) > 4:
                self._moved = True
                span = self.view_t1 - self.view_t0
                dt = -dx / max(self.width(), 1) * span
                t0 = float(np.clip(self._drag_t0 + dt, 0,
                                   self.duration - span))
                self.view_t0, self.view_t1 = t0, t0 + span
                self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag_x is not None and not self._moved:
            t = self._x2t(ev.position().x())
            # Cue flag hit?
            for c in self.cues:
                if abs(self._t2x(c["time_s"]) - ev.position().x()) < 6:
                    self.cueClicked.emit(c)
                    break
            else:
                self.playhead = t
                self.seekRequested.emit(t)
            self.update()
        self._drag_x = None


def _nice_step(span):
    for s in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if span / s <= 14:
            return s
    return 600
