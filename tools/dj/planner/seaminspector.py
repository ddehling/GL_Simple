"""Seam inspector: WHY one seam is (or isn't) rhythmically safe, drawn.

Click a seam line in the compiled plan and this panel shows the two
tracks' rhythm signatures aligned the way the blend will sound: the
incoming track's step grid (resampled to the tempo-multiple read and
rotated to the bar alignment the scorer chose) above the outgoing one,
kick/snare/hat rows per track, low-band contradictions highlighted, and a
one-beat "microscope" lane that overlays the two fine onset folds so a
flam risk is a visible pair of near-miss ticks, not just a number.

The point is trust: when the picture obviously matches what the audition
plays, the chips upstairs stop being mystery labels.
"""
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from lib.dj.rhythm import aligned_pattern, region_view

BG = QColor(20, 20, 25)
GRID = QColor(60, 60, 70)
BAR_LINE = QColor(110, 110, 125)
TXT = QColor(160, 160, 170)
TXT_HI = QColor(220, 220, 230)
LOW = QColor(90, 150, 220)       # kick/bass row
MID = QColor(200, 160, 80)       # snare/clap row
HIGH = QColor(150, 150, 160)     # hat row
CLASH = QColor(230, 100, 100)
FLAM = QColor(235, 120, 90)
OK = QColor(80, 180, 130)


def _chip_text(rt):
    sw_a, sw_b = rt.get("swing_a", 0.5), rt.get("swing_b", 0.5)

    def sw(x):
        return "straight" if x < 0.54 else \
            "swung" if x < 0.62 else "shuffle"
    parts = [f"kick agreement {rt.get('kick_agreement', 0):.2f}",
             f"swing {sw(sw_a)} vs {sw(sw_b)}"]
    if rt.get("meter_clash"):
        parts.insert(0, f"METER {rt.get('meter_a', 4)}/4 vs "
                        f"{rt.get('meter_b', 4)}/4")
    fl = rt.get("flam_ms")
    if fl is not None:
        parts.append("hits lock" if fl < 15.0
                     else f"flam risk {fl:.0f}ms" if fl <= 80.0
                     else f"closest hits {fl:.0f}ms")
    if rt.get("conf", 1.0) < 0.5:
        parts.append("low grid confidence - estimates are shaky")
    return "   ".join(parts)


class SeamInspector(QWidget):
    """set_seam(a_track, b_track, plan, seam_info) / clear()."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setMaximumHeight(190)
        self.a = self.b = None
        self.plan = {}
        self.si = {}

    def clear(self):
        self.a = self.b = None
        self.plan, self.si = {}, {}
        self.update()

    def set_seam(self, a, b, plan, seam_info):
        self.a, self.b = a, b
        self.plan = plan or {}
        self.si = seam_info or {}
        self.update()

    # -- painting ----------------------------------------------------------
    def _draw_pattern_block(self, p, sig, rect, mult=1.0, rot=0):
        """One track's 3-band step grid inside rect. Returns the low-band
        pattern actually drawn (for clash marking)."""
        rows = (("high", HIGH, 0.20), ("mid", MID, 0.25), ("low", LOW, 0.55))
        y = rect.y()
        low_drawn = None
        for name, color, frac in rows:
            rh = rect.height() * frac
            pat = aligned_pattern(sig, name, mult, rot)
            if pat is not None:
                n = len(pat)
                sw = rect.width() / n
                for i in range(n):
                    v = float(pat[i])
                    if v <= 0.05:
                        continue
                    h = max(rh * v, 1.5)
                    c = QColor(color)
                    c.setAlpha(int(90 + 165 * min(v, 1.0)))
                    p.fillRect(QRectF(rect.x() + i * sw + 0.5,
                                      y + rh - h, sw - 1.0, h), c)
                if name == "low":
                    low_drawn = pat
            y += rh
        return low_drawn

    def _draw_grid_lines(self, p, rect, n_steps=32):
        sw = rect.width() / n_steps
        for i in range(n_steps + 1):
            x = rect.x() + i * sw
            if i % 16 == 0:
                p.setPen(QPen(BAR_LINE, 1))
            elif i % 4 == 0:
                p.setPen(QPen(GRID, 1))
            else:
                continue
            p.drawLine(int(x), int(rect.y()), int(x),
                       int(rect.y() + rect.height()))

    def _draw_beat_lane(self, p, rect, sig_a, sig_b):
        """One-beat microscope: A's fine onset fold as upward ticks, B's
        downward, near-miss pairs (the audible flam window) bridged red."""
        mid = rect.y() + rect.height() / 2.0
        p.setPen(QPen(GRID, 1))
        p.drawLine(int(rect.x()), int(mid),
                   int(rect.x() + rect.width()), int(mid))

        def peaks(sig, thr=0.5):
            out = []
            for key in ("beat_low", "beat_perc"):
                fold = sig.get(key)
                if fold is None:
                    continue
                n = len(fold)
                for i in range(n):
                    v = float(fold[i])
                    if v >= thr and v >= fold[i - 1] and v >= fold[(i + 1) % n]:
                        out.append((i / n, v))
            return out
        pa, pb = peaks(sig_a), peaks(sig_b)
        period_ms = (60.0 / self.a.bpm * 1000.0) if getattr(
            self.a, "bpm", 0) else 500.0
        half = rect.height() / 2.0 - 2
        for ph, v in pa:
            x = rect.x() + ph * rect.width()
            p.setPen(QPen(LOW, 2))
            p.drawLine(int(x), int(mid), int(x), int(mid - half * v))
        for ph, v in pb:
            x = rect.x() + ph * rect.width()
            p.setPen(QPen(MID, 2))
            p.drawLine(int(x), int(mid), int(x), int(mid + half * v))
        # Flam bridges: each PROMINENT A-hit to its NEAREST B-hit only,
        # when that near-miss falls in the audible flam window. All-pairs
        # bridging turned a noisy fold into a solid orange smear.
        strong_b = [q for q in pb if q[1] >= 0.6]
        for ph_a, va in pa:
            if va < 0.6 or not strong_b:
                continue
            ph_b, _vb = min(strong_b, key=lambda q: min(
                abs(q[0] - ph_a), 1.0 - abs(q[0] - ph_a)))
            d = abs(ph_a - ph_b)
            d = min(d, 1.0 - d)
            if 15.0 <= d * period_ms <= 80.0:
                x0 = rect.x() + min(ph_a, ph_b) * rect.width()
                x1 = rect.x() + max(ph_a, ph_b) * rect.width()
                p.setPen(QPen(FLAM, 2))
                p.drawLine(int(x0), int(mid), int(x1), int(mid))

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, BG)
        if self.a is None or self.b is None:
            p.setPen(TXT)
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter,
                       "seam inspector - select a ↳ seam line in the plan")
            p.end()
            return
        rt = self.si.get("rhythm") or {}
        # Same views the scorer compared: B's intro region over A's exit
        # region when the v2 signatures carry them (whole track otherwise).
        sig_a = region_view(getattr(self.a, "rhythm_sig", None), "out")
        sig_b = region_view(getattr(self.b, "rhythm_sig", None), "in")
        b_reg = " intro" if (sig_b or {}).get("region") == "in" else ""
        a_reg = " exit" if (sig_a or {}).get("region") == "out" else ""
        pad = 8
        p.setPen(TXT_HI)
        style = self.plan.get("style", "?")
        rate = self.plan.get("rate", 1.0)
        p.drawText(pad, 14, f"{self.b.title[:34]}  (in{b_reg}, "
                            f"{'as heard at seam' if rt else 'no signature'})"
                            f"    {style} @ rate {rate:.3f}")
        if sig_a is None or sig_b is None:
            p.setPen(TXT)
            p.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter,
                       "no rhythm signature on "
                       + ("either track" if sig_a is None and sig_b is None
                          else self.a.title[:30] if sig_a is None
                          else self.b.title[:30])
                       + "  -  run the Rhythm pass in the Library tab")
            p.end()
            return

        mult, rot = rt.get("mult", 1.0), rt.get("rot", 0)
        lane_h = 26.0
        block_h = (h - 46 - lane_h - 14) / 2.0
        x0, bw = pad + 20, w - 2 * pad - 20
        b_rect = QRectF(x0, 20, bw, block_h)
        lane = QRectF(x0, 22 + block_h, bw, lane_h)
        a_rect = QRectF(x0, 24 + block_h + lane_h, bw, block_h)

        p.setPen(TXT)
        p.save()
        p.rotate(-90)
        p.drawText(int(-(b_rect.y() + block_h)), pad + 8, "IN")
        p.drawText(int(-(a_rect.y() + block_h)), pad + 8, "OUT")
        p.restore()

        self._draw_grid_lines(p, b_rect)
        self._draw_grid_lines(p, a_rect)
        low_b = self._draw_pattern_block(p, sig_b, b_rect, mult, rot)
        low_a = self._draw_pattern_block(p, sig_a, a_rect)
        self._draw_beat_lane(p, lane, sig_a, sig_b)

        # Low-band contradictions: one track slams a 16th the other leaves
        # empty, in the rows a blend would layer. Marked between the blocks.
        if low_a is not None and low_b is not None:
            n = min(len(low_a), len(low_b))
            sw = bw / n
            for i in range(n):
                va, vb = float(low_a[i]), float(low_b[i])
                if (va > 0.6 and vb < 0.15) or (vb > 0.6 and va < 0.15):
                    x = x0 + i * sw
                    p.setPen(QPen(CLASH, 2))
                    p.drawLine(int(x + sw / 2), int(lane.y() - 2),
                               int(x + sw / 2), int(lane.y() + 3))

        p.setPen(TXT_HI)
        p.drawText(pad, int(a_rect.y() + block_h + 13),
                   f"{self.a.title[:34]}  (out{a_reg})")
        col = OK if rt.get("score", 1.0) >= 0.6 else CLASH
        p.setPen(col if rt else TXT)
        p.drawText(pad + 240, int(a_rect.y() + block_h + 13),
                   _chip_text(rt) if rt else "")
        p.end()
