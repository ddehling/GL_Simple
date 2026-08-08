"""Layer Lab: hear a drum bed under a track, and tune it by ear.

The loop layer (deck C, DJSystem._do_layer) rides a percussion bed under
the playing track. Whether that is any good is a taste question no
instrument in this repo can answer - total-mix RMS barely moves for a bed
13 dB down, which says nothing about whether it works musically. So this
tab does the one thing that settles it: renders the SAME span of the SAME
track twice, once dry and once with the bed, and lets you flip between
them.

  * pick what it rides under, and which loop rides
  * ONE play button that starts AT the bed (the lead-in is identical in
    both takes by construction, so starting there only ever proves that
    nothing is happening), and a SPACEBAR A/B that swaps dry<->bedded
    under the playhead at the same sample position - the only way a
    -13 dB bed can honestly be judged, since comparing across a stop
    asks you to hold audio in memory (see seamprobe.py)
  * level and length sliders, because those are exactly the two things
    that can only be set by ear (system.LAYER_GAIN / LAYER_BARS are the
    live defaults these start from)
  * the picture: the track's sections and energy with the bed's span
    shaded, and the loop's own waveform tiled underneath it

Rendered through lib.dj.audition.render_layer, which mirrors _do_layer's
event shape - mount early, transport on the downbeat, fade over
LAYER_FADE_BARS - so what you hear is what a live press would make.
"""
import numpy as np
from PyQt6.QtCore import Qt, QRectF, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QVBoxLayout, QWidget)

from lib.dj import looplayer
from lib.dj.system import LAYER_BARS, LAYER_GAIN

RATE = 44100
BG = QColor(20, 20, 25)
TXT = QColor(160, 160, 170)
DIM = QColor(120, 120, 130)
SECTION_COLORS = {
    "intro": QColor(70, 90, 130), "outro": QColor(70, 90, 130),
    "groove": QColor(55, 115, 85), "build": QColor(185, 150, 55),
    "breakdown": QColor(95, 70, 135),
}
BED = QColor(235, 160, 70)          # the layer, everywhere it appears
ENERGY = QColor(120, 150, 200)


class LayerView(QWidget):
    """Top: the under-track with the bed's span shaded. Bottom: the loop
    itself, tiled across that span so the repeat count is visible."""
    seekRequested = pyqtSignal(float)      # seconds into the render

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(230)
        self.job = None
        self.playhead = None

    def set_job(self, job):
        self.job = job
        self.playhead = None
        self.update()

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    def _t2x(self, t, dur):
        w = max(self.width() - 70, 10)
        return 55 + w * max(0.0, min(t / max(dur, 1e-6), 1.0))

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), BG)
        if self.job is None:
            p.setPen(TXT)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Pick a track and a loop, then render.")
            return
        j = self.job
        t = j["track"]
        dur = t.duration_s
        H = self.height()
        top_h = int((H - 30) * 0.62)
        strip = 13
        p.setPen(TXT)
        p.drawText(QRectF(2, 6, 50, strip), Qt.AlignmentFlag.AlignVCenter,
                   "TRACK")

        for sec in t.sections or []:
            x0, x1 = self._t2x(sec["start_s"], dur), self._t2x(sec["end_s"], dur)
            col = SECTION_COLORS.get(sec.get("kind"), QColor(85, 85, 90))
            p.fillRect(QRectF(x0, 6, x1 - x0, strip), col)
            if x1 - x0 > 44:
                p.setPen(QColor(235, 235, 235, 200))
                p.drawText(QRectF(x0 + 3, 6, x1 - x0 - 4, strip),
                           Qt.AlignmentFlag.AlignVCenter, sec.get("kind", ""))
            wash = QColor(col)
            wash.setAlpha(30)
            p.fillRect(QRectF(x0, 6 + strip, x1 - x0, top_h - strip), wash)

        body_top, body_h = 6 + strip, top_h - strip
        self._energy(p, t, dur, body_top, body_h)

        # the bed's span
        x0 = self._t2x(j["at_s"], dur)
        x1 = self._t2x(j["at_s"] + j["span_s"], dur)
        band = QColor(BED)
        band.setAlpha(60)
        p.fillRect(QRectF(x0, body_top, max(x1 - x0, 2.0), body_h), band)
        p.setPen(QPen(BED, 2))
        p.drawLine(int(x0), int(body_top), int(x0), int(body_top + body_h))
        p.drawLine(int(x1), int(body_top), int(x1), int(body_top + body_h))
        p.setPen(BED)
        p.drawText(QRectF(x0 + 4, body_top + 2, 260, 13),
                   Qt.AlignmentFlag.AlignLeft,
                   f"bed · {j['bars']} bars · {j['span_s']:.0f}s")

        # MEASURED bed level (dB under the track) across the span - the
        # answer to "is it even doing anything, and where".
        db = j.get("bed_db")
        if db is not None and len(db):
            hop = j.get("bed_hop_s", 0.25)
            t0 = j["render_t0_s"]
            pts = []
            for i, v in enumerate(db):
                ta = t0 + i * hop
                # -40 dB at the floor, 0 dB at the top of the lane
                y = body_top + body_h - body_h * float(
                    np.clip((v + 40.0) / 40.0, 0.0, 1.0))
                pts.append(QPointF(self._t2x(ta, dur), y))
            p.setPen(QPen(QColor(BED.red(), BED.green(), BED.blue(), 220), 2))
            p.drawPolyline(QPolygonF(pts))
            best = float(np.max(db)) if len(db) else -99.0
            p.setPen(BED)
            p.drawText(QRectF(x0 + 4, body_top + 16, 300, 13),
                       Qt.AlignmentFlag.AlignLeft,
                       f"measured: peaks {best:+.1f} dB under the track")

        # playhead, mapped from render time onto the track's timeline
        if self.playhead is not None:
            ta = j["render_t0_s"] + self.playhead
            if 0 <= ta <= dur:
                xp = self._t2x(ta, dur)
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.drawLine(int(xp), int(body_top), int(xp),
                           int(body_top + body_h))
        self._ruler(p, dur, body_top + body_h)

        # -- the loop itself, tiled across the span --
        ly = top_h + 16
        lh = H - ly - 14
        p.setPen(TXT)
        p.drawText(QRectF(2, ly, 50, 13), Qt.AlignmentFlag.AlignVCenter,
                   "LOOP")
        loop = j.get("loop_wave")
        if loop is None or lh < 10:
            return
        reps = max(j["span_s"] / max(j["loop_s"], 1e-6), 1.0)
        w = int(self.width() - 70)
        mid = ly + lh / 2
        pts_hi, pts_lo = [], []
        for i in range(w):
            k = int(len(loop) * ((i / max(w - 1, 1)) * reps % 1.0))
            v = float(loop[min(k, len(loop) - 1)])
            pts_hi.append(QPointF(55 + i, mid - v * lh * 0.45))
            pts_lo.append(QPointF(55 + i, mid + v * lh * 0.45))
        p.setPen(QPen(BED, 1))
        p.drawPolyline(QPolygonF(pts_hi))
        p.drawPolyline(QPolygonF(pts_lo))
        # repeat boundaries
        p.setPen(QPen(QColor(BED.red(), BED.green(), BED.blue(), 90), 1,
                      Qt.PenStyle.DotLine))
        for r in range(1, int(reps) + 1):
            x = 55 + w * (r / reps)
            p.drawLine(int(x), int(ly), int(x), int(ly + lh))
        p.setPen(DIM)
        p.drawText(int(55), int(ly + lh + 11),
                   f"{j['label']}  ·  {j['loop_s']:.2f}s  ·  "
                   f"{reps:.1f}x  ·  stretch {(j['rate'] - 1) * 100:+.2f}%")

    def _energy(self, p, track, dur, top, h):
        c = np.asarray(track.row.get("energy_curve") or [], dtype=np.float64)
        if len(c) < 8:
            return
        w = int(self.width() - 70)
        pts = []
        for i in range(w):
            t = dur * i / max(w - 1, 1)
            v = float(np.clip(c[min(int(t * 2), len(c) - 1)] / 1.2, 0.0, 1.0))
            pts.append(QPointF(55 + i, top + h - 4 - v * (h - 10)))
        p.setPen(QPen(ENERGY, 1))
        p.drawPolyline(QPolygonF(pts))

    def _ruler(self, p, dur, y):
        p.setPen(DIM)
        step = 60.0 if dur < 600 else 120.0
        t = step
        while t < dur:
            x = self._t2x(t, dur)
            p.drawLine(int(x), int(y) - 4, int(x), int(y))
            p.drawText(int(x) + 2, int(y) - 1, f"{int(t // 60)}:00")
            t += step

    def mousePressEvent(self, ev):
        if self.job is None:
            return
        j = self.job
        w = max(self.width() - 70, 10)
        ta = (ev.position().x() - 55) / w * j["track"].duration_s
        self.seekRequested.emit(max(0.0, ta - j["render_t0_s"]))


class _RenderWorker(QThread):
    """Both takes of one job, off the UI thread."""
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, db, job):
        super().__init__()
        self.db, self.job = db, job

    def run(self):
        try:
            from lib.dj.audition import render_layer
            j = dict(self.job)
            j["dry"] = render_layer(
                self.db, j["track"], j["prep"], j["at_s"], j["bars"],
                j["gain"], with_layer=False,
                status=lambda m: self.status.emit(m))
            j["wet"] = render_layer(
                self.db, j["track"], j["prep"], j["at_s"], j["bars"],
                j["gain"], with_layer=True,
                status=lambda m: self.status.emit(m))
            self.ready.emit(j)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


class LayerLabTab(QWidget):
    """Pick a track, pick a loop, hear it dry vs bedded, tune by ear."""

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        import random
        from tools.dj.planner.player import TrackPlayer
        self.player = TrackPlayer()
        self.rng = random.Random()
        self.track = None
        self.cands = []
        self.job = None
        self._worker = None
        self._playing = "wet"

        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.pick_btn = QPushButton("⟳ Random track")
        self.pick_btn.setToolTip(
            "Pick a track for the bed to ride under, and list the loops "
            "that fit its tempo.")
        self.pick_btn.clicked.connect(self._pick)
        row.addWidget(self.pick_btn)
        row.addWidget(QLabel("Loop:"))
        self.loop_box = QComboBox()
        self.loop_box.setMinimumWidth(280)
        self.loop_box.setToolTip(
            "Drum loops playable at this track's tempo, least stretch "
            "first. 'library' = sliced from a track's drums stem; 'file' = "
            "a curated media/loops/*.wav DJ tool.")
        row.addWidget(self.loop_box)
        row.addStretch(1)
        v.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Level:"))
        self.gain = QSlider(Qt.Orientation.Horizontal)
        self.gain.setRange(5, 100)
        self.gain.setValue(int(LAYER_GAIN * 100))
        self.gain.setFixedWidth(150)
        self.gain.setToolTip(
            f"Bed level. system.LAYER_GAIN is {LAYER_GAIN} - what the live "
            f"button uses. Only your ear can set this.")
        self.gain.valueChanged.connect(self._label_knobs)
        row2.addWidget(self.gain)
        row2.addWidget(QLabel("Bars:"))
        self.bars = QSlider(Qt.Orientation.Horizontal)
        self.bars.setRange(4, 48)
        self.bars.setValue(LAYER_BARS)
        self.bars.setFixedWidth(130)
        self.bars.setToolTip(
            f"How long the bed rides. system.LAYER_BARS is {LAYER_BARS}.")
        self.bars.valueChanged.connect(self._label_knobs)
        row2.addWidget(self.bars)
        self.knobs = QLabel("")
        self.knobs.setStyleSheet("color:#9aa0a8;")
        row2.addWidget(self.knobs)
        self.render_btn = QPushButton("● Render both takes")
        self.render_btn.clicked.connect(self._render)
        self.render_btn.setEnabled(False)
        row2.addWidget(self.render_btn)
        row2.addStretch(1)
        v.addLayout(row2)
        self._label_knobs()

        self.card = QLabel("Press “Random track”, choose a loop, then "
                           "render. You get the same span twice — dry and "
                           "bedded — so the only difference is the layer.")
        self.card.setWordWrap(True)
        self.card.setStyleSheet("font-size: 14px; padding: 8px;")
        v.addWidget(self.card)

        self.view = LayerView()
        self.view.seekRequested.connect(self._seek)
        v.addWidget(self.view, 1)

        row3 = QHBoxLayout()
        # PLAY STARTS AT THE BED, not at the top of the render: the first
        # bars are identical in both takes by construction, so starting
        # there guarantees the first thing you hear is "no difference".
        self.play_btn = QPushButton("▶ Play from the bed")
        self.play_btn.clicked.connect(lambda: self._play(self._playing))
        self.play_btn.setEnabled(False)
        row3.addWidget(self.play_btn)
        # THE INSTRUMENT THAT ACTUALLY WORKS. A -13 dB bed cannot be judged
        # by playing one take, stopping, and playing the other - that asks
        # you to hold audio in memory across a gap, which seamprobe.py
        # already established nobody can do. This swaps the buffer UNDER
        # the playhead at the same sample position, so the bed appears and
        # disappears in your ear with nothing else changing.
        self.ab_btn = QPushButton("⇄  now: WITH BED   (space)")
        self.ab_btn.setStyleSheet("color:#eba046;font-weight:bold;")
        self.ab_btn.clicked.connect(self._swap)
        self.ab_btn.setEnabled(False)
        row3.addWidget(self.ab_btn)
        self.solo_btn = QPushButton("▶ Bed ONLY (solo)")
        self.solo_btn.setToolTip(
            "Play the layer on its own - literally the WITH take minus the "
            "DRY take. Silent here = the bed is not being made. Audible "
            "here but not in the mix = it is only a level problem.")
        self.solo_btn.clicked.connect(lambda: self._play("bed"))
        self.solo_btn.setEnabled(False)
        row3.addWidget(self.solo_btn)
        stop = QPushButton("■ Stop")
        stop.clicked.connect(lambda: self.planner.stop_all_playback())
        row3.addWidget(stop)
        row3.addStretch(1)
        v.addLayout(row3)
        from PyQt6.QtGui import QKeySequence, QShortcut
        sc = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        sc.activated.connect(self._swap)

        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color:#9aa0a8; padding: 0 8px;")
        v.addWidget(self.detail)

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._follow)
        self._tick.start(60)

    def _label_knobs(self):
        # Show the level in dB, not as an opaque 0-1: loops leave
        # looplayer at TARGET_RMS, and a club master sits near 0.2 rms,
        # so gain -> dB-under-the-track is predictable enough to print.
        import math
        g = self.gain.value() / 100.0
        db = 20 * math.log10(max(g * looplayer.TARGET_RMS / 0.2, 1e-6))
        self.knobs.setText(f"gain {g:.2f}  (~{db:+.0f} dB under the "
                           f"track) · {self.bars.value()} bars")

    # -- picking -----------------------------------------------------------
    def _pick(self):
        lib = self.planner.library
        if not lib:
            self.card.setText("No library loaded.")
            return
        pool = [t for t in lib
                if t.duration_s >= 150 and t.sections and t.bpm]
        for _ in range(60):
            t = self.rng.choice(pool)
            cands = looplayer.candidates(
                self.planner.db, lib, t.bpm, exclude_ids={t.id},
                music_root=self.planner.db.music_root, limit=12)
            if cands:
                self.track, self.cands = t, cands
                self.loop_box.clear()
                for c in cands:
                    self.loop_box.addItem(
                        f"[{c['kind']}] {c['label']}  "
                        f"({(c['rate'] - 1) * 100:+.2f}%)")
                self.render_btn.setEnabled(True)
                self.card.setText(
                    f"{t.title}   ·   {t.bpm:.1f} bpm   ·   "
                    f"{int(t.duration_s // 60)}:{int(t.duration_s % 60):02d}"
                    f"   ·   {len(cands)} loops fit this tempo")
                self.detail.setText("Choose a loop and render.")
                return
        self.card.setText(
            "No track found with loops at its tempo. Most of the library "
            "has no stems rendered yet (tools/dj/dj_stems.py), and "
            "media/loops/ may be empty.")

    # -- rendering ---------------------------------------------------------
    def _render(self):
        if self.track is None or not self.cands:
            return
        if self._worker is not None and self._worker.isRunning():
            return
        cand = self.cands[max(self.loop_box.currentIndex(), 0)]
        prep = looplayer.prepare(cand, self.planner.db,
                                 self.planner.db.music_root)
        if prep is None:
            self.detail.setText("that loop failed to load - try another")
            return
        t = self.track
        period = 60.0 / max(t.bpm, 1e-6)
        bars = self.bars.value()
        # Drop the bed somewhere in the track's body, on a downbeat, with
        # room for the whole ride.
        span = bars * 4 * period
        lo, hi = t.duration_s * 0.25, max(t.duration_s * 0.75 - span,
                                          t.duration_s * 0.3)
        at = t.nearest_downbeat(max(min(self.rng.uniform(lo, hi),
                                        t.duration_s - span - 5.0), 5.0))
        from lib.dj.system import LAYER_FADE_BARS
        mono = prep["samples"].mean(axis=1)
        step = max(len(mono) // 1200, 1)
        job = {"track": t, "prep": prep, "at_s": at, "bars": bars,
               "span_s": span, "gain": self.gain.value() / 100.0,
               "fade_s": min(LAYER_FADE_BARS * 4 * period, span * 0.4),
               "label": prep["label"], "rate": prep["rate"],
               "loop_s": prep["loop_s"],
               "loop_wave": np.abs(mono[:len(mono) // step * step]
                                   ).reshape(-1, step).max(axis=1),
               "render_t0_s": max(t.nearest_downbeat(max(at - 6.0, 0.0)),
                                  0.0)}
        self.render_btn.setEnabled(False)
        for b in (self.play_btn, self.ab_btn, self.solo_btn):
            b.setEnabled(False)
        self.detail.setText("rendering...")
        self._worker = _RenderWorker(self.planner.db, job)
        self._worker.ready.connect(self._ready)
        self._worker.failed.connect(self._failed)
        self._worker.status.connect(self.detail.setText)
        self._worker.start()

    def _ready(self, job):
        self.job = job
        self.view.set_job(job)
        self.render_btn.setEnabled(True)
        for b in (self.play_btn, self.ab_btn, self.solo_btn):
            b.setEnabled(True)
        dry, wet = job["dry"], job["wet"]
        n = min(len(dry), len(wet))
        # The bed's OWN level against the track - the number that means
        # something. Total-mix RMS barely moves for a bed 13 dB down, so
        # reporting that instead would read as "nothing happened".
        diff = wet[:n] - dry[:n]
        r_bed = float(np.sqrt((diff.mean(axis=1) ** 2).mean()) + 1e-12)
        # The bed's level against the track, second by second, so the
        # picture SHOWS where it is and how loud. "I hear no difference"
        # is usually "I was listening to the identical lead-in".
        hop = int(0.25 * RATE)
        m = n // hop
        dd = diff[:m * hop].mean(axis=1).reshape(-1, hop)
        tt = dry[:m * hop].mean(axis=1).reshape(-1, hop)
        job["bed_db"] = (20 * np.log10(
            (np.sqrt((dd ** 2).mean(axis=1)) + 1e-9)
            / (np.sqrt((tt ** 2).mean(axis=1)) + 1e-9)))
        job["bed_hop_s"] = 0.25
        # SOLO THE BED. wet-dry IS the layer as rendered - stretch, loop
        # wraps, fades and all - so this needs no extra render and cannot
        # disagree with what the WITH take contains. It exists to settle
        # one question fast: if this is silent the bed is not being made,
        # and if it plays then "I hear no difference" is a level problem.
        job["bed"] = np.ascontiguousarray(diff, dtype=np.float32)
        self.view.set_job(job)
        a = int((job["at_s"] - job["render_t0_s"] + 2.0) * RATE)
        b = min(int(a + job["span_s"] * RATE * 0.6), n)
        r_trk = float(np.sqrt((dry[a:b].mean(axis=1) ** 2).mean()) + 1e-12)
        self.detail.setText(
            f"<b style='color:#eba046'>{job['label']}</b> — "
            f"{job['bars']} bars from "
            f"{int(job['at_s'] // 60)}:{int(job['at_s'] % 60):02d}, "
            f"stretch {(job['rate'] - 1) * 100:+.2f}%, "
            f"gain {job['gain']:.2f}<br>"
            f"bed sits <b>{20 * np.log10(r_bed / r_trk):+.1f} dB</b> "
            f"against the track · peak "
            f"{float(np.abs(wet).max()):.3f} "
            f"({'clip-free' if float(np.abs(wet).max()) < 0.999 else 'CLIPPING'})"
            f" · click the track lane to start from a point")

    def _failed(self, msg):
        self.render_btn.setEnabled(True)
        self.detail.setText("render failed: " + msg)

    # -- playback ----------------------------------------------------------
    def _play(self, key, seek_to=None):
        if self.job is None:
            return
        self._playing = key
        self.planner.claim_playback("layerlab")
        self.player.load(self.job[key])
        if seek_to is None:
            # Land just before the bed is fully up: its fade-in is
            # LAYER_FADE_BARS long, and the lead-in before it is
            # identical in both takes.
            seek_to = max(self.job["at_s"] - self.job["render_t0_s"]
                          + self.job["fade_s"] * 0.8, 0.0)
        self.player.seek(seek_to)
        self.player.play()
        self._label_ab()

    def _swap(self):
        """(Solo is not one of the two sides - flip back to WITH first.)"""
        """Flip dry<->wet AT THE SAME POSITION. Both takes are the same
        span rendered identically apart from deck C, so they are sample
        aligned and the swap is seamless."""
        if self.job is None:
            return
        other = "wet" if self._playing in ("dry", "bed") else "dry"
        was = self.player.playing
        t = self.player.time_s()
        self._playing = other
        self.planner.claim_playback("layerlab")
        self.player.load(self.job[other])
        self.player.seek(t)
        if was:
            self.player.play()
        self._label_ab()

    def _label_ab(self):
        if self._playing == "bed":
            self.ab_btn.setText("⇄  now: BED SOLO   (space)")
            self.ab_btn.setStyleSheet("color:#eba046;font-weight:bold;")
            return
        wet = self._playing == "wet"
        self.ab_btn.setText(f"⇄  now: {'WITH BED' if wet else 'DRY'}"
                            f"   (space)")
        self.ab_btn.setStyleSheet(
            "color:#eba046;font-weight:bold;" if wet
            else "color:#9aa0a8;font-weight:bold;")

    def _seek(self, t):
        if self.job is None or t < 0:
            return
        self._play(self._playing, seek_to=t)

    def _follow(self):
        if self.job is None or not self.player.playing:
            return
        self.view.set_playhead(self.player.time_s())

    def stop_playback(self):
        try:
            self.player.pause()
        except Exception:
            pass

    def close_audio(self):
        try:
            self.player.close()
        except Exception:
            pass
