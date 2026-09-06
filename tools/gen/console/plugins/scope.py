"""Scope tab: the rack's audio as a waveform and a spectrum, painted with
QPainter at ~30 Hz from the rack's monitor ring (LocalBackend.audio_tap).

Baselines the operator controls:
  waveform  display gain (dB), a reference level line (dBFS) to judge
            headroom against, and the window length
  spectrum  floor and range (dB), smoothing, log/linear frequency, and a
            CAPTURED BASELINE: freeze the current averaged spectrum as a
            reference trace, overlay it, or show the live spectrum as a
            difference against it (what changed since the capture).

Remote backends do not stream audio, so the tab says so and stays idle.
This is a plugin: delete the file and the console loses nothing else."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QPushButton, QSlider,
                             QVBoxLayout, QWidget)

from lib.gen import RATE

FFT_N = 4096


class ScopeMath:
    """Pure numpy so the gate can test it without a window."""

    def __init__(self):
        self.smooth = 0.6
        self._spec = None
        self.baseline = None
        self.freqs = np.fft.rfftfreq(FFT_N, 1.0 / RATE)
        self._win = np.hanning(FFT_N).astype(np.float32)

    def spectrum(self, stereo):
        """dB magnitude spectrum of the last FFT_N samples, smoothed."""
        mono = stereo.mean(axis=1) if stereo.ndim == 2 else stereo
        if len(mono) < FFT_N:
            mono = np.concatenate([np.zeros(FFT_N - len(mono), dtype=np.float32), mono])
        x = mono[-FFT_N:] * self._win
        mag = np.abs(np.fft.rfft(x)) * (2.0 / self._win.sum())
        db = 20.0 * np.log10(mag + 1e-9)
        if self._spec is None or self.smooth <= 0:
            self._spec = db
        else:
            self._spec = self.smooth * self._spec + (1.0 - self.smooth) * db
        return self._spec

    def capture(self):
        self.baseline = None if self._spec is None else self._spec.copy()
        return self.baseline is not None

    def clear(self):
        self.baseline = None

    @staticmethod
    def peak_hz(freqs, db, lo=30.0):
        m = freqs >= lo
        return float(freqs[m][int(np.argmax(db[m]))])


class ScopeCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(360)
        self.wave = None            # (n,2) float32
        self.spec = None            # dB per bin
        self.baseline = None
        self.freqs = None
        self.gain_db = 0.0
        self.ref_db = -6.0
        self.show_ref = True
        self.floor_db = -90.0
        self.range_db = 90.0
        self.log_freq = True
        self.diff = False
        self.show_base = True
        self.message = ""

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor("#0f1116"))
        if self.message:
            p.setPen(QColor("#9aa0ab")); p.drawText(QRectF(0, 0, W, H), Qt.AlignmentFlag.AlignCenter, self.message); return
        hw = int(H * 0.42)
        self._paint_wave(p, 0, 0, W, hw)
        self._paint_spec(p, 0, hw + 6, W, H - hw - 6)

    def _paint_wave(self, p, x0, y0, W, H):
        p.fillRect(x0, y0, W, H, QColor("#12141a"))
        mid = y0 + H / 2
        p.setPen(QPen(QColor("#2a2f3c"), 1)); p.drawLine(x0, int(mid), x0 + W, int(mid))
        g = 10 ** (self.gain_db / 20.0)
        if self.show_ref:
            r = 10 ** (self.ref_db / 20.0) * g
            yy = (H / 2) * min(1.0, r)
            p.setPen(QPen(QColor("#e64"), 1, Qt.PenStyle.DashLine))
            p.drawLine(x0, int(mid - yy), x0 + W, int(mid - yy)); p.drawLine(x0, int(mid + yy), x0 + W, int(mid + yy))
            p.setPen(QColor("#e64")); p.drawText(x0 + 6, int(mid - yy) - 3, f"{self.ref_db:+.0f} dBFS")
        if self.wave is None or len(self.wave) < 2:
            return
        for ch, col in ((0, "#4bd"), (1, "#9ef")):
            y = self.wave[:, ch] * g
            n = len(y)
            cols = min(W, n)
            idx = np.linspace(0, n - 1, cols * 2).astype(int)
            path = QPainterPath()
            xs = np.linspace(x0, x0 + W, len(idx))
            ys = mid - np.clip(y[idx], -1, 1) * (H / 2 - 2)
            path.moveTo(xs[0], ys[0])
            for i in range(1, len(idx)):
                path.lineTo(xs[i], ys[i])
            p.setPen(QPen(QColor(col), 1)); p.drawPath(path)
        peak = float(np.abs(self.wave).max()) if len(self.wave) else 0.0
        p.setPen(QColor("#9aa0ab")); p.drawText(x0 + W - 150, y0 + 14, f"peak {20 * np.log10(peak + 1e-9):+.1f} dBFS  gain {self.gain_db:+.0f} dB")

    def _fx(self, f, W):
        if self.log_freq:
            return W * (np.log10(np.maximum(f, 20.0)) - np.log10(20.0)) / (np.log10(20000.0) - np.log10(20.0))
        return W * f / 20000.0

    def _paint_spec(self, p, x0, y0, W, H):
        p.fillRect(x0, y0, W, H, QColor("#12141a"))
        p.setPen(QPen(QColor("#2a2f3c"), 1))
        for f, lab in ((50, "50"), (100, "100"), (200, "200"), (500, "500"), (1000, "1k"), (2000, "2k"), (5000, "5k"), (10000, "10k")):
            x = x0 + self._fx(np.array([f]), W)[0]
            p.drawLine(int(x), y0, int(x), y0 + H); p.setPen(QColor("#5a6070")); p.drawText(int(x) + 3, y0 + H - 4, lab); p.setPen(QPen(QColor("#2a2f3c"), 1))
        lo, hi = (-self.range_db / 2, self.range_db / 2) if self.diff else (self.floor_db, self.floor_db + self.range_db)
        for d in range(int(lo), int(hi) + 1, 20 if self.range_db > 60 else 10):
            y = y0 + H - H * (d - lo) / (hi - lo)
            p.drawLine(x0, int(y), x0 + W, int(y)); p.setPen(QColor("#5a6070")); p.drawText(x0 + 4, int(y) - 2, f"{d:+d}" if self.diff else str(d)); p.setPen(QPen(QColor("#2a2f3c"), 1))
        if self.spec is None or self.freqs is None:
            return
        f = self.freqs[1:]; xs = x0 + self._fx(f, W)
        def draw(db, col, width=1.5, fill=None):
            ys = y0 + H - H * np.clip((db[1:] - lo) / (hi - lo), 0, 1.05)
            path = QPainterPath(); path.moveTo(xs[0], ys[0])
            step = max(1, len(xs) // (W * 2))
            for i in range(step, len(xs), step):
                path.lineTo(xs[i], ys[i])
            if fill:
                fp = QPainterPath(path); fp.lineTo(xs[-1], y0 + H); fp.lineTo(xs[0], y0 + H); fp.closeSubpath()
                p.fillPath(fp, QColor(fill))
            p.setPen(QPen(QColor(col), width)); p.drawPath(path)
        if self.diff and self.baseline is not None:
            p.setPen(QPen(QColor("#e64"), 1, Qt.PenStyle.DashLine)); ym = y0 + H - H * (0 - lo) / (hi - lo); p.drawLine(x0, int(ym), x0 + W, int(ym))
            draw(self.spec - self.baseline, "#fc6", 1.5)
            p.setPen(QColor("#fc6")); p.drawText(x0 + W - 220, y0 + 14, "live − baseline (dB)")
        else:
            if self.baseline is not None and self.show_base:
                draw(self.baseline, "#e64", 1.2)
            draw(self.spec, "#4bd", 1.5, fill="#1a3a44")
            p.setPen(QColor("#9aa0ab")); p.drawText(x0 + W - 220, y0 + 14, "spectrum (dBFS)" + ("  · baseline" if self.baseline is not None else ""))


class ScopeTab(QWidget):
    def __init__(self, console):
        super().__init__()
        self.console = console
        self.math = ScopeMath()
        self.canvas = ScopeCanvas()
        self.window_s = 0.25
        lay = QVBoxLayout(self); lay.setContentsMargins(12, 8, 12, 8); lay.setSpacing(8)
        lay.addWidget(self.canvas, 1)
        ctl = QHBoxLayout(); ctl.setSpacing(16)
        # waveform baselines
        wf = QFrame(); wf.setObjectName("card"); f1 = QFormLayout(wf); f1.setContentsMargins(10, 8, 10, 8)
        t1 = QLabel("waveform"); t1.setObjectName("cardTitle"); f1.addRow(t1)
        self.gain = self._slider(-24, 24, 0, f1, "display gain (dB)", lambda v: setattr(self.canvas, "gain_db", float(v)))
        self.ref = self._slider(-40, 0, -6, f1, "reference (dBFS)", lambda v: setattr(self.canvas, "ref_db", float(v)))
        self.ref_on = QCheckBox("show reference line"); self.ref_on.setChecked(True); self.ref_on.toggled.connect(lambda on: setattr(self.canvas, "show_ref", on)); f1.addRow(self.ref_on)
        self.win = QComboBox(); [self.win.addItem(l, v) for l, v in (("50 ms", 0.05), ("100 ms", 0.1), ("250 ms", 0.25), ("500 ms", 0.5), ("1 s", 1.0), ("2 s", 2.0))]
        self.win.setCurrentIndex(2); self.win.currentIndexChanged.connect(lambda i: setattr(self, "window_s", self.win.itemData(i))); f1.addRow("window", self.win)
        ctl.addWidget(wf, 1)
        # spectrum baselines
        sp = QFrame(); sp.setObjectName("card"); f2 = QFormLayout(sp); f2.setContentsMargins(10, 8, 10, 8)
        t2 = QLabel("spectrum"); t2.setObjectName("cardTitle"); f2.addRow(t2)
        self.floor = self._slider(-120, -40, -90, f2, "floor (dB)", lambda v: setattr(self.canvas, "floor_db", float(v)))
        self.rng = self._slider(30, 120, 90, f2, "range (dB)", lambda v: setattr(self.canvas, "range_db", float(v)))
        self.smooth = self._slider(0, 95, 60, f2, "smoothing (%)", lambda v: setattr(self.math, "smooth", v / 100.0))
        self.logf = QCheckBox("log frequency"); self.logf.setChecked(True); self.logf.toggled.connect(lambda on: setattr(self.canvas, "log_freq", on)); f2.addRow(self.logf)
        ctl.addWidget(sp, 1)
        # captured baseline
        bl = QFrame(); bl.setObjectName("card"); f3 = QVBoxLayout(bl); f3.setContentsMargins(10, 8, 10, 8)
        t3 = QLabel("baseline"); t3.setObjectName("cardTitle"); f3.addWidget(t3)
        hint = QLabel("Freeze the current spectrum as a reference. Overlay it, or view the live spectrum as a difference against it."); hint.setObjectName("dim"); hint.setWordWrap(True); f3.addWidget(hint)
        cap = QPushButton("⏺ capture baseline"); cap.setProperty("style", "go"); cap.clicked.connect(self.capture); f3.addWidget(cap)
        self.show_base = QCheckBox("overlay baseline"); self.show_base.setChecked(True); self.show_base.toggled.connect(lambda on: setattr(self.canvas, "show_base", on)); f3.addWidget(self.show_base)
        self.diff = QCheckBox("difference mode (live − baseline)"); self.diff.toggled.connect(lambda on: setattr(self.canvas, "diff", on)); f3.addWidget(self.diff)
        clr = QPushButton("clear baseline"); clr.clicked.connect(self.clear_baseline); f3.addWidget(clr)
        self.base_status = QLabel("no baseline"); self.base_status.setObjectName("dim"); f3.addWidget(self.base_status); f3.addStretch(1)
        ctl.addWidget(bl, 1)
        lay.addLayout(ctl)
        self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(33)

    def _slider(self, lo, hi, val, form, label, fn):
        row = QHBoxLayout(); s = QSlider(Qt.Orientation.Horizontal); s.setRange(lo, hi); s.setValue(val)
        v = QLabel(str(val)); v.setFixedWidth(40); v.setAlignment(Qt.AlignmentFlag.AlignRight)
        s.valueChanged.connect(lambda x: (v.setText(str(x)), fn(x))); row.addWidget(s); row.addWidget(v)
        w = QWidget(); w.setLayout(row); form.addRow(label, w); return s

    def capture(self):
        if self.math.capture():
            self.canvas.baseline = self.math.baseline; self.base_status.setText("baseline captured")
        else:
            self.base_status.setText("nothing to capture yet")

    def clear_baseline(self):
        self.math.clear(); self.canvas.baseline = None; self.diff.setChecked(False); self.base_status.setText("no baseline")

    def tick(self):
        if not self.isVisible():
            return
        be = self.console.backend
        tap = be.audio_tap(max(FFT_N, int(self.window_s * RATE))) if (be is not None and hasattr(be, "audio_tap")) else None
        if tap is None:
            self.canvas.message = ("connect to a local backend to see the audio" if be is None or not hasattr(be, "audio_tap")
                                   else ("the show does not stream audio to the console" if be.audio_tap(1) is None and be.__class__.__name__ == "RemoteBackend"
                                         else "start the music to see the audio"))
            self.canvas.update(); return
        self.canvas.message = ""
        self.canvas.wave = tap[-int(self.window_s * RATE):]
        self.canvas.spec = self.math.spectrum(tap)
        self.canvas.freqs = self.math.freqs
        self.canvas.update()

    def refresh(self, state):
        pass


def register(console):
    console.add_tab("Scope", ScopeTab(console))
