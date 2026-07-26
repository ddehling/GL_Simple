"""Live deck monitor for the Mix tab - the DJ-software 'what is actually
happening right now' view, driven from REAL playback state:

  - two scrolling beat-level waveforms (deck A top, deck B bottom) drawn
    from the actual decoded samples, aligned on a shared NOW line and
    scrolled by each deck's true source position from submix telemetry
    (lag-compensated to what the speakers are playing this instant);
  - beat/downbeat markers from each track's analyzed grid, so you can SEE
    the two grids lock (or not) during a blend;
  - live low/mid/high meters per deck: measured band content at the
    playhead x the deck's LIVE EQ and gain automation;
  - a beat-offset readout between the decks while both are audible.

Everything here comes from telemetry + samples, not the plan - if the mix
misbehaves, this view shows it.
"""
import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

RATE = 44100
WINDOW_S = 4.0                   # visible seconds either side of NOW
DECK_COLORS = {"a": QColor(120, 180, 235), "b": QColor(120, 220, 160)}
EQ_COLORS = [QColor(235, 105, 90), QColor(120, 210, 120),
             QColor(110, 170, 240)]


class DeckMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(190)
        self.preview = None
        self.tracks_by_id = {}

    def attach(self, preview, compiled):
        self.preview = preview
        self.tracks_by_id = {}
        if compiled:
            for s in compiled["slots"]:
                self.tracks_by_id[s["track"].id] = s["track"]
        self.update()

    # -- painting -----------------------------------------------------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(12, 12, 16))
        W, H = self.width(), self.height()
        pv = self.preview
        if pv is None or not pv.playing:
            p.setPen(QColor(110, 110, 120))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "live deck monitor - plays when the preview plays")
            return
        tel = pv.telemetry() or {}
        decks = tel.get("decks") or {}
        lag = pv.audible_lag_s()
        row_h = (H - 18) // 2
        meters_w = 96
        wave_w = W - meters_w - 8
        now_x = wave_w * 0.5

        phases = {}
        for row, name in enumerate(("a", "b")):
            y0 = 2 + row * (row_h + 4)
            d = decks.get(name) or {}
            track = self.tracks_by_id.get(d.get("track_id"))
            col = DECK_COLORS[name]
            p.setPen(QColor(70, 70, 80))
            p.drawRect(QRectF(0, y0, wave_w, row_h))
            if not (d.get("ready") and d.get("playing") and track):
                p.setPen(QColor(90, 90, 100))
                p.drawText(QRectF(0, y0, wave_w, row_h),
                           Qt.AlignmentFlag.AlignCenter,
                           f"deck {name.upper()} idle")
                continue
            rate = d.get("rate", 1.0)
            # Lag-compensate: telemetry reflects the producer, which runs
            # ahead of the speakers by the ring-buffer fill.
            pos = max(d.get("time_s", 0.0) - lag * rate, 0.0)
            s16 = pv.cached_samples(d.get("track_id"))
            if s16 is not None:
                self._draw_wave(p, s16, pos, y0, wave_w, row_h, col)
            self._draw_beats(p, track, pos, y0, wave_w, row_h)
            # Label: what this deck is really doing right now.
            gain = d.get("gain", 0.0)
            p.setPen(QColor(235, 235, 245))
            p.drawText(QRectF(4, y0 + 1, wave_w - 8, 14),
                       Qt.AlignmentFlag.AlignLeft,
                       f"{name.upper()}  {track.title[:34]}  "
                       f"{track.bpm * rate:6.1f} bpm  "
                       f"rate {rate:.4f}  gain {gain:.2f}")
            phases[name] = self._phase(track, pos)
            self._draw_meters(p, track, pos, d, W - meters_w, y0,
                              meters_w - 4, row_h)

        # NOW line + beat offset between the decks.
        p.setPen(QPen(QColor(255, 255, 255, 220), 2))
        p.drawLine(int(now_x), 0, int(now_x), H - 16)
        p.setPen(QColor(180, 180, 190))
        if len(phases) == 2:
            err = (phases["b"] - phases["a"] + 0.5) % 1.0 - 0.5
            ok = abs(err) < 0.06
            p.setPen(QColor(120, 235, 140) if ok else QColor(240, 120, 100))
            p.drawText(int(now_x) + 6, H - 4,
                       f"beat offset {err:+.3f} beats "
                       f"{'(locked)' if ok else '(DRIFTING)'}")
        else:
            p.drawText(int(now_x) + 6, H - 4, "single deck")

    def _draw_wave(self, p, s16, pos, y0, w, h, col):
        """Min/max waveform of +/-WINDOW_S around pos from the real samples."""
        i0 = int((pos - WINDOW_S) * RATE)
        i1 = int((pos + WINDOW_S) * RATE)
        n = i1 - i0
        cols = max(w // 2, 64)
        spp = max(n // cols, 1)
        mid_y = y0 + h * 0.55
        amp = h * 0.4
        pen = QPen(col, 1)
        p.setPen(pen)
        for c in range(cols):
            a = i0 + c * spp
            b = a + spp
            if b <= 0 or a >= len(s16):
                continue
            seg = s16[max(a, 0):min(b, len(s16)), 0]
            if not len(seg):
                continue
            lo = float(seg.min()) / 32767.0
            hi = float(seg.max()) / 32767.0
            x = c * (w / cols)
            p.drawLine(int(x), int(mid_y - hi * amp),
                       int(x), int(mid_y - lo * amp))

    def _draw_beats(self, p, track, pos, y0, w, h):
        g = None
        for seg in track.grid or []:
            if seg["start_s"] <= pos <= seg["end_s"]:
                g = seg
                break
        if g is None and track.grid:
            g = track.grid[0]
        if g is None:
            return
        period = g["period_s"]
        first_down = g["first_beat_s"] + track.downbeat_offset * period
        t = pos - WINDOW_S
        t = g["first_beat_s"] + np.ceil(
            (t - g["first_beat_s"]) / period) * period
        while t <= pos + WINDOW_S:
            x = (t - (pos - WINDOW_S)) / (2 * WINDOW_S) * w
            down = round((t - first_down) / period) % 4 == 0
            p.setPen(QPen(QColor(255, 255, 255, 130 if down else 45),
                          2 if down else 1))
            p.drawLine(int(x), y0 + 14, int(x), y0 + h - 2)
            t += period

    def _phase(self, track, pos):
        g = track.grid[0] if track.grid else None
        if g is None or g["period_s"] <= 0:
            return 0.0
        return ((pos - g["first_beat_s"]) / g["period_s"]) % 1.0

    def _draw_meters(self, p, track, pos, d, x0, y0, w, h):
        """Low/mid/high: measured band level at the playhead x live EQ x
        live gain - what this deck contributes to the room right now."""
        bc = track.row.get("band_curve") or {}
        eq = d.get("eq") or [1.0, 1.0, 1.0]
        gain = d.get("gain", 0.0)
        bw = w / 3.0
        for i, key in enumerate(("low", "mid", "high")):
            curve = bc.get(key) or []
            v = 0.0
            if curve:
                ci = min(int(pos * 2), len(curve) - 1)
                v = min(float(curve[ci]), 1.2) * float(eq[i]) * gain
            v = min(v, 1.2)
            bx = x0 + i * bw
            p.fillRect(QRectF(bx + 2, y0 + 14, bw - 4, h - 16),
                       QColor(30, 30, 36))
            c = QColor(EQ_COLORS[i])
            c.setAlpha(230)
            bar_h = (h - 16) * v / 1.2
            p.fillRect(QRectF(bx + 2, y0 + 14 + (h - 16) - bar_h,
                              bw - 4, bar_h), c)
            p.setPen(QColor(150, 150, 160))
            p.drawText(QRectF(bx, y0 + h - 13, bw, 12),
                       Qt.AlignmentFlag.AlignCenter, key)
