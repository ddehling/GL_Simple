"""Per-stem lanes for the Analysis tab: WHAT each demucs stem extracted
and WHERE.

Four lanes (drums / bass / other / vocals), each an RMS energy envelope
on the SAME time axis as the waveform/spectrogram above - the view window
follows WaveformView.viewChanged, the playhead follows the transport.
Click = seek (same contract as the main view). Lanes light up where the
stem actually carries material, so separation quality (and bleed - hi-hat
ghosts in the vocals lane) is visible at a glance before any stem style
trusts the files.
"""
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

RATE = 44100
ENV_HOP = 2048                    # ~46 ms per envelope point

LANE_COLORS = {
    "drums": QColor(235, 160, 70),
    "bass": QColor(95, 150, 235),
    "other": QColor(130, 200, 160),
    "vocals": QColor(235, 105, 120),
}
LANE_ORDER = ("drums", "bass", "other", "vocals")


def stem_envelope(arr):
    """(n,2) float array -> per-block mono RMS envelope, normalized to the
    stem's own peak (shape readability over absolute level)."""
    mono = np.asarray(arr, dtype=np.float32).mean(axis=1)
    m = len(mono) // ENV_HOP
    if m < 2:
        return np.zeros(2, dtype=np.float32)
    env = np.sqrt((mono[:m * ENV_HOP].reshape(m, ENV_HOP) ** 2).mean(axis=1))
    peak = float(env.max())
    return (env / peak if peak > 1e-6 else env).astype(np.float32)


class StemLanes(QWidget):
    seekRequested = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.envs = {}               # name -> envelope array
        self.duration = 1.0
        self.view_t0 = 0.0
        self.view_t1 = 1.0
        self.playhead = 0.0
        self.muted = set()           # names currently NOT in the audition mix
        self.model = None            # separator that rendered these stems

    def set_stems(self, envs, duration, model=None):
        self.envs = dict(envs or {})
        self.duration = max(duration, 0.001)
        self.model = model           # which separator rendered these
        self.setVisible(bool(self.envs))
        self.update()

    def clear(self):
        self.envs = {}
        self.setVisible(False)
        self.update()

    def set_view(self, t0, t1):
        self.view_t0, self.view_t1 = t0, t1
        self.update()

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    def set_muted(self, muted):
        self.muted = set(muted)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(14, 14, 18))
        if not self.envs:
            return
        W, H = self.width(), self.height()
        lanes = [n for n in LANE_ORDER if n in self.envs]
        lane_h = H / max(len(lanes), 1)
        span = max(self.view_t1 - self.view_t0, 1e-6)
        env_rate = RATE / ENV_HOP    # envelope points per second
        for k, name in enumerate(lanes):
            y0 = k * lane_h
            base = y0 + lane_h - 2
            col = QColor(LANE_COLORS[name])
            if name in self.muted:
                col.setAlpha(70)
            env = self.envs[name]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(col)
            # One filled bar per pixel column: max of the envelope points
            # under that column (peaks survive any zoom).
            for x in range(W):
                t = self.view_t0 + x / max(W, 1) * span
                j0 = int(t * env_rate)
                j1 = max(int((t + span / W) * env_rate), j0 + 1)
                if j0 >= len(env):
                    break
                v = float(env[j0:min(j1, len(env))].max())
                if v > 0.004:
                    h = v * (lane_h - 6)
                    p.drawRect(QRectF(x, base - h, 1.0, h))
            p.setPen(QPen(QColor(60, 60, 70), 1))
            p.drawLine(0, int(y0), W, int(y0))
            p.setPen(QColor(210, 210, 220)
                     if name not in self.muted else QColor(130, 130, 140))
            p.drawText(6, int(y0 + 14),
                       name + ("  (muted)" if name in self.muted else ""))
        # Which separator produced these files - always visible so a
        # mixed-model library stays legible at a glance.
        if self.model:
            p.setPen(QColor(150, 150, 165))
            p.drawText(self.rect().adjusted(0, 2, -6, 0),
                       Qt.AlignmentFlag.AlignRight
                       | Qt.AlignmentFlag.AlignTop, self.model)
        # Playhead.
        x = (self.playhead - self.view_t0) / span * W
        p.setPen(QPen(QColor(255, 255, 255, 200), 2))
        p.drawLine(int(x), 0, int(x), H)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self.envs:
            span = max(self.view_t1 - self.view_t0, 1e-6)
            t = self.view_t0 + ev.position().x() / max(self.width(), 1) * span
            self.seekRequested.emit(float(np.clip(t, 0, self.duration)))
