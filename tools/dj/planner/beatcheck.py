"""Beat Check tab: SEE the beat matching, don't take the engine's word.

Renders a real brain-planned seam through the actual engine (submix,
sync snap, PLL, kick bias - the same path a night plays), taps each
deck's post-EQ audio, and draws the two decks' band-separated envelopes
on one shared render-time axis:

  - three rows: low (<130 Hz), mid (180-2000), high (>4 kHz)
  - deck A drawn UP from each row's centerline, deck B mirrored DOWN,
    so alignment is read straight across the centerline
  - solid ticks: each deck's MEASURED kick positions (grid + phase
    profile, projected through the deck's actual position trace)
  - dashed ticks: the raw stored-grid beats, so the phase correction
    the engine applied is visible as the solid/dashed gap
  - blend start / swap markers, wheel-zoom, drag-pan, click-to-play
    with a live playhead

If the kicks are matched, the low-band attack ridges meet at the
centerline on the solid ticks during the overlap. If they don't, the
engine is wrong and this tab is the proof (2026-08-05, built because
the user - rightly - wanted eyes on it, not another green gate).
"""
import os
import sys

import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout, QWidget)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

RATE = 44100
ENV_HZ = 200                 # drawn envelope resolution (samples/sec)
BANDS = (("low", "lowpass", 130.0),
         ("mid", "bandpass", (180.0, 2000.0)),
         ("high", "highpass", 4000.0))
_LONG_TEST = "(long test blend)"
_STYLE_CHOICES = [_LONG_TEST, "(brain's choice)", "bass_swap",
                  "long_blend", "filter_sweep", "phrase_cut", "echo_out"]
TEST_BEATS = 32              # long-test overlap: 8 measures of dual


def _band_env(x, kind, freq):
    """Smoothed band envelope of (n,2) audio, decimated to ENV_HZ."""
    from scipy.signal import butter, sosfilt
    mono = x.mean(axis=1).astype(np.float64)
    sos = butter(4, freq, btype=kind, fs=RATE, output="sos")
    env = np.abs(sosfilt(sos, mono))
    w = max(int(0.01 * RATE), 1)
    env = np.convolve(env, np.ones(w) / w, mode="same")
    step = RATE // ENV_HZ
    n = len(env) // step * step
    return env[:n].reshape(-1, step).max(axis=1).astype(np.float32)


class _CheckWorker(QThread):
    """Pick a pair, plan, render THROUGH THE REAL ENGINE with per-deck
    taps, measure everything the canvas draws. One at a time."""
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, db, brain, library, rng, want_style):
        super().__init__()
        self.db, self.brain, self.library = db, brain, library
        self.rng, self.want_style = rng, want_style

    def run(self):
        try:
            from lib.dj import beatpower as bp
            long_test = self.want_style == _LONG_TEST
            want = "long_blend" if long_test else self.want_style
            # Pinned styles can legitimately need many picks: the beat-
            # power/band gates make overlapped-drum pairs rare, and
            # legality is checked before any decode so retries are cheap.
            for attempt in range(60):
                cur = self.rng.choice(self.library)
                if not (150 <= cur.duration_s <= 480):
                    continue
                # BEAT-HEAVY MATERIAL ONLY for the long test (user,
                # 2026-08-05: "you keep picking weirdo ambient tracks") -
                # judging beat matching needs beats. Both sides must
                # clear the beat-power blend bar.
                if long_test and bp.blendable(cur.id) is not True:
                    continue
                self.status.emit(f"picking a partner for {cur.title[:28]}...")
                cand, meta = self.brain.choose_next(cur, 0.6, cur.bpm)
                if cand is None or cand.duration_s > 480:
                    continue
                if long_test and bp.blendable(cand.id) is not True:
                    continue
                plan = self.brain.plan_transition(
                    cur, cand, meta,
                    after_s=cur.duration_s * self.rng.uniform(0.35, 0.65),
                    force_style=want, test_gates=bool(want))
                if want and plan["style"] != want:
                    continue             # gates refused the pin - repick
                if long_test and plan["style"] == "long_blend":
                    # A one-measure overlap proves nothing (user, same
                    # day: "you blend the beat over like a single
                    # measure") - stretch the dual to 8 bars so the ear
                    # and the ticks get time to agree or disagree.
                    plan["beats"] = max(int(plan.get("beats") or 0),
                                        TEST_BEATS)
                self.status.emit(f"rendering {plan['style']}: "
                                 f"{cur.title[:22]} -> {cand.title[:22]}...")
                from tools.dj.dj_knobsweep import render_tapped
                mix, decks, marks = render_tapped(self.db, cur, cand,
                                                  dict(plan))
                self.status.emit("measuring bands and beats...")
                out = self._measure(cur, cand, plan, mix, decks, marks)
                self.ready.emit(out)
                return
            self.failed.emit("no renderable seam found in 40 tries - "
                             "try (brain's choice) or another style")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _measure(self, cur, cand, plan, mix, decks, marks):
        from lib.dj import beatpower as bp
        from lib.dj.seamverify import _beats_in_span, _render_times
        blend_s, swap_s = marks["blend_s"], marks["swap_s"]
        t0 = max(0.0, blend_s - 6.0)
        t1 = min(len(mix) / RATE, swap_s + 10.0)
        i0, i1 = int(t0 * RATE), int(t1 * RATE)

        envs = {}
        aud = {}
        for deck in ("a", "b"):
            seg = decks[deck][i0:i1]
            envs[deck] = {name: _band_env(seg, kind, freq)
                          for name, kind, freq in BANDS}
            # Audibility mask at ENV_HZ: beat ticks are only drawn where
            # the deck is actually sounding - ticks marching through a
            # faded-out deck's silence are noise, not information.
            step = RATE // ENV_HZ
            mono = np.abs(seg.mean(axis=1))
            n = len(mono) // step * step
            e = mono[:n].reshape(-1, step).max(axis=1)
            w = max(int(0.5 * ENV_HZ), 1)
            k = np.ones(w) / w
            smooth = np.convolve(e, k, mode="same")
            aud[deck] = (smooth > 0.04 * (float(smooth.max()) or 1.0))

        ticks = {}
        offs = {}
        for deck, track, anchor in (("a", cur, plan["out_s"]),
                                    ("b", cand, plan["in_s"])):
            trace = marks["pos"][deck]
            grid_r = np.array([])
            kick_r = np.array([])
            off = bp.phase_offset(track.id, at_s=anchor) or 0.0
            if len(trace) >= 2:
                s_lo = min(p[1] for p in trace)
                s_hi = max(p[1] for p in trace)
                beats = _beats_in_span(track, s_lo, s_hi)
                if len(beats):
                    grid_r = _render_times(beats, trace)
                    kick_r = _render_times(beats + off, trace)
            ticks[deck] = {"grid": grid_r, "kick": kick_r}
            offs[deck] = off * 1000.0

        # Headline: measured kick-to-kick during the overlap (the number
        # the eye should agree with).
        ka = ticks["a"]["kick"]
        kb = ticks["b"]["kick"]
        sa = ka[(ka >= blend_s) & (ka <= swap_s)]
        sb = kb[(kb >= blend_s) & (kb <= swap_s)]
        k2k = None
        if len(sa) >= 4 and len(sb) >= 4:
            k2k = float(np.median([np.min(np.abs(sb - x)) for x in sa])
                        ) * 1000.0
        return {"a": cur, "b": cand, "plan": plan, "mix": mix,
                "marks": marks, "t0": t0, "t1": t1, "envs": envs,
                "aud": aud, "ticks": ticks, "offs": offs, "k2k_ms": k2k}


class _BandCanvas(QWidget):
    """Three stacked band rows, A up / B down, ticks, zoom, playhead."""
    seekRequested = pyqtSignal(float)    # render seconds

    def __init__(self):
        super().__init__()
        self.data = None
        self.view = None                 # (lo_s, hi_s) visible window
        self.playhead = None
        self.setMinimumHeight(360)
        self.setMouseTracking(False)
        self._drag = None

    def set_data(self, data):
        self.data = data
        pad = 0.0
        self.view = (data["t0"] + pad, data["t1"] - pad)
        self.playhead = None
        self.update()

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    # -- interaction -------------------------------------------------------
    def wheelEvent(self, ev):
        if not self.data:
            return
        lo, hi = self.view
        span = hi - lo
        f = 0.8 if ev.angleDelta().y() > 0 else 1.25
        cx = lo + (ev.position().x() / max(self.width(), 1)) * span
        span = min(max(span * f, 1.0), self.data["t1"] - self.data["t0"])
        lo = max(self.data["t0"], cx - (cx - lo) * f)
        hi = min(self.data["t1"], lo + span)
        self.view = (hi - span, hi) if hi - span < self.data["t0"] \
            else (lo, hi)
        self.update()

    def mousePressEvent(self, ev):
        if not self.data:
            return
        self._drag = (ev.position().x(), self.view)
        self._moved = False

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            return
        x0, (lo, hi) = self._drag
        dx = (ev.position().x() - x0) / max(self.width(), 1) * (hi - lo)
        if abs(ev.position().x() - x0) > 3:
            self._moved = True
        lo2 = lo - dx
        hi2 = hi - dx
        if lo2 < self.data["t0"]:
            lo2, hi2 = self.data["t0"], self.data["t0"] + (hi - lo)
        if hi2 > self.data["t1"]:
            lo2, hi2 = self.data["t1"] - (hi - lo), self.data["t1"]
        self.view = (lo2, hi2)
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag is not None and not getattr(self, "_moved", False):
            lo, hi = self.view
            t = lo + (ev.position().x() / max(self.width(), 1)) * (hi - lo)
            self.seekRequested.emit(t)
        self._drag = None

    # -- painting ----------------------------------------------------------
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(16, 17, 20))
        if not self.data:
            p.setPen(QColor(120, 120, 130))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Generate a seam to inspect its beat matching")
            p.end()
            return
        d = self.data
        lo, hi = self.view
        W, H = self.width(), self.height()
        span = max(hi - lo, 1e-6)

        def X(t):
            return (t - lo) / span * W

        rows = [(name, i) for i, (name, _k, _f) in enumerate(BANDS)]
        row_h = H / len(rows)
        col_a = QColor(255, 176, 32)         # deck A: amber
        col_b = QColor(64, 200, 255)         # deck B: cyan
        env_t0 = d["t0"]

        def audible(deck, t):
            m = d.get("aud", {}).get(deck)
            if m is None:
                return True
            i = int((t - env_t0) * ENV_HZ)
            return bool(m[i]) if 0 <= i < len(m) else False

        for name, ri in rows:
            cy = ri * row_h + row_h / 2
            half = row_h / 2 - 14
            # row frame + centerline
            p.setPen(QPen(QColor(46, 48, 54), 1))
            p.drawLine(0, int(ri * row_h), W, int(ri * row_h))
            p.setPen(QPen(QColor(70, 72, 80), 1))
            p.drawLine(0, int(cy), W, int(cy))
            p.setPen(QColor(150, 152, 160))
            p.drawText(6, int(ri * row_h + 16), name.upper())
            # dashed raw-grid ticks first (under everything)
            for deck, col, sgn in (("a", col_a, -1), ("b", col_b, 1)):
                dim = QColor(col)
                dim.setAlpha(70)
                pen = QPen(dim, 1, Qt.PenStyle.DashLine)
                p.setPen(pen)
                for t in d["ticks"][deck]["grid"]:
                    if lo <= t <= hi and audible(deck, t):
                        x = X(t)
                        p.drawLine(int(x), int(cy),
                                   int(x), int(cy + sgn * half))
            # envelopes: A up, B down, normalized per deck+row over the
            # whole analysis window so relative attack shape is honest
            for deck, col, sgn in (("a", col_a, -1), ("b", col_b, 1)):
                env = d["envs"][deck].get(name)
                if env is None or not len(env):
                    continue
                peak = float(env.max()) or 1.0
                n0 = int(max(0, (lo - env_t0) * ENV_HZ))
                n1 = int(min(len(env), (hi - env_t0) * ENV_HZ + 1))
                if n1 - n0 < 2:
                    continue
                idx = np.arange(n0, n1)
                xs = ((env_t0 + idx / ENV_HZ) - lo) / span * W
                ys = cy + sgn * (env[n0:n1] / peak) * half
                pen = QPen(col, 1.4)
                p.setPen(pen)
                from PyQt6.QtGui import QPolygonF
                poly = QPolygonF([self._pt(xs[i], ys[i])
                                  for i in range(len(idx))])
                p.drawPolyline(poly)
            # solid measured-kick ticks on top
            for deck, col, sgn in (("a", col_a, -1), ("b", col_b, 1)):
                pen = QPen(col, 2)
                p.setPen(pen)
                for t in d["ticks"][deck]["kick"]:
                    if lo <= t <= hi and audible(deck, t):
                        x = X(t)
                        p.drawLine(int(x), int(cy),
                                   int(x), int(cy + sgn * half))

        # blend / swap markers
        for t, label, col in ((d["marks"]["blend_s"], "blend",
                               QColor(120, 255, 120)),
                              (d["marks"]["swap_s"], "swap",
                               QColor(255, 120, 120))):
            if lo <= t <= hi:
                p.setPen(QPen(col, 1, Qt.PenStyle.DotLine))
                p.drawLine(int(X(t)), 0, int(X(t)), H)
                p.setPen(col)
                p.drawText(int(X(t)) + 4, 14, label)

        if self.playhead is not None and lo <= self.playhead <= hi:
            p.setPen(QPen(QColor(240, 240, 245), 2))
            x = int(X(self.playhead))
            p.drawLine(x, 0, x, H)
        p.end()

    @staticmethod
    def _pt(x, y):
        from PyQt6.QtCore import QPointF
        return QPointF(float(x), float(y))


class BeatCheckTab(QWidget):
    """Visual ground truth for beat matching (see module docstring)."""

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        from tools.dj.planner.player import TrackPlayer
        self.player = TrackPlayer()
        self.brain = None
        self.rng = None
        self._worker = None
        self._current = None

        top = QHBoxLayout()
        self.style_box = QComboBox()
        self.style_box.addItems(_STYLE_CHOICES)
        self.gen_btn = QPushButton("New seam")
        self.gen_btn.clicked.connect(self.generate)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#999;")
        top.addWidget(QLabel("Style:"))
        top.addWidget(self.style_box)
        top.addWidget(self.gen_btn)
        top.addWidget(self.play_btn)
        top.addWidget(self.status, 1)

        self.headline = QLabel("")
        self.headline.setStyleSheet("font-weight:bold;")
        self.legend = QLabel(
            "deck A ▲ amber, deck B ▼ cyan · solid tick = measured kick "
            "(grid + phase profile) · dashed = raw stored grid · "
            "wheel zoom, drag pan, click plays from there")
        self.legend.setStyleSheet("color:#888;")

        self.canvas = _BandCanvas()
        self.canvas.seekRequested.connect(self._seek_play)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.headline)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.legend)

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    # -- generation --------------------------------------------------------
    def generate(self):
        if self._worker is not None:
            return
        if not self.planner.library:
            self.status.setText("library is empty - scan first")
            return
        import random
        from lib.dj.brain import Brain
        from lib.dj.themes import get_theme
        if self.rng is None:
            self.rng = random.Random()
        if self.brain is None:
            self.brain = Brain(self.planner.library, get_theme("groove"),
                               seed=self.rng.randrange(1 << 30))
        want = self.style_box.currentText()
        want = None if want.startswith("(") else want
        self.gen_btn.setEnabled(False)
        self._worker = _CheckWorker(self.planner.db, self.brain,
                                    self.planner.library, self.rng, want)
        self._worker.status.connect(self.status.setText)
        self._worker.ready.connect(self._seam_ready)
        self._worker.failed.connect(self._seam_failed)
        self._worker.start()

    def _seam_ready(self, data):
        self._worker.wait()              # never drop a live QThread ref
        self._worker = None
        self.gen_btn.setEnabled(True)
        self._current = data
        self.canvas.set_data(data)
        self.player.pause()
        self.player.load(data["mix"])
        self.player.seek(max(0.0, data["marks"]["blend_s"] - 4.0))
        self.play_btn.setEnabled(True)
        self.play_btn.setText("Play")
        k2k = data["k2k_ms"]
        if data["plan"]["style"] == "long_fade":
            # Fades are not beat-matched BY DESIGN (the pair failed the
            # blend gates) - reporting their kick delta as a defect
            # would teach exactly the wrong lesson.
            k2k_txt = "long_fade: decks are not beat-matched by design"
        elif k2k is not None:
            k2k_txt = f"kick-to-kick (overlap) median {k2k:.1f} ms"
        else:
            k2k_txt = "no overlapping kicks to compare (cut)"
        self.headline.setText(
            f"{data['plan']['style']}:  {data['a'].title[:34]}  →  "
            f"{data['b'].title[:34]}    ·    {k2k_txt}    ·    "
            f"phase applied A {data['offs']['a']:+.1f} ms / "
            f"B {data['offs']['b']:+.1f} ms")
        self.status.setText("")

    def _seam_failed(self, msg):
        self._worker.wait()
        self._worker = None
        self.gen_btn.setEnabled(True)
        self.status.setText(msg)

    # -- playback ----------------------------------------------------------
    def _toggle_play(self):
        if self.player.playing:
            self.player.pause()
            self.play_btn.setText("Play")
            self._timer.stop()
        elif self._current is not None:
            self.player.play()
            self.play_btn.setText("Pause")
            self._timer.start()

    def _seek_play(self, t):
        if self._current is None:
            return
        self.player.seek(t)
        self.player.play()
        self.play_btn.setText("Pause")
        self._timer.start()

    def _tick(self):
        if self._current is None:
            return
        t = self.player.time_s()
        self.canvas.set_playhead(t)
        if not self.player.playing:
            self.play_btn.setText("Play")
            self._timer.stop()
        if t >= self._current["t1"]:
            self.player.pause()          # stop at the analysis window edge

    def shutdown(self):
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        self.player.close()
