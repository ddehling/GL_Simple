"""The instrument panel of the Analysis tab: the SONG'S OWN instruments
and what every one of them plays, beat by beat - a tracker view on the
waveform's time axis.

Layout (top to bottom):
    ruler          bar numbers at the bar lines (beat lines fainter)
    stem header    one per stem, in stem colour, carrying the stem's
                   energy envelope (what the separation extracted and
                   where - the old stem lanes folded in) and the count
                   of sounds found; click to collapse / expand
    instrument     one row per sound: a two-line gutter (name = the role
                   read for it, then the measured facts: register,
                   length, hits, level, note range) and its cells:
        pitched      a filled cell per hit with the NOTE NAME written in
                     when the zoom allows; held notes as spans with the
                     chord's names at the start; vertical position =
                     pitch within the sound's own range
        unpitched    a cell per hit, height = strength
    playhead       the beat under the playhead is a lit column

TIME AXIS: identical to the waveform's - the panel maps time over the
waveform widget's full pixel width (axis_widget), never over its own
width minus a margin, so a note sits exactly under its spectrogram
column whatever the scrollbar or the gutter do. The gutter is a
translucent overlay on the left, not a reserved margin.

Only the sounds that matter are shown by default (>= MAIN_SHARE of the
stem's events); set_show_all(True) shows every one. The view window
follows WaveformView.viewChanged, the playhead follows the transport.

THE MIXER. Every stem header and every instrument row carries S and M
boxes like a mixer channel, on two levels with bus semantics:
    stem level   stem_solo / stem_mute: a soloed stem plays alone among
                 the stems, a muted stem silences every row under it
    row level    solo / mute over the instrument ids: a soloed row plays
                 alone among the rows, else everything not muted
A row sounds when its stem sounds AND it sounds among the rows. The
vocals rows have no boxes of their own (the vocal stem is one unit in
the reconstruction: reused phrases, not voices) - the header's boxes are
theirs. mixChanged() fires after every change; the tab reads
stems_audible() / audible() and plays that mix from whichever source it
is on (the stems, or the reconstruction). row_mute_enabled=False draws
the rows' M boxes hollow and refuses them with hint(text): on the stems
source a single sound cannot be dropped from real audio.

Click an instrument's gutter = select it + auditionRequested(instrument)
(its exemplar plays); click ON a note cell = eventRequested(instrument,
event) (that very note plays from the stem); click empty timeline = seek;
click a stem header = collapse. With an instrument selected the panel is
a SAMPLER: the keys z s x d c v g b h n j m play it at C..B of its own
octave, q 2 w 3 e r 5 t 6 y 7 u the octave above, [ ] shift octaves
(noteRequested(instrument, midi) - the exemplar repitched).
"""
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QFont, QFontMetrics, QImage, QLinearGradient
from PyQt6.QtWidgets import QWidget

from tools.dj.planner.stemlanes import LANE_COLORS, ENV_HOP, RATE, draw_sm, sm_hit, SM_W, SM_H, SM_GAP

RULER_H = 16
HEADER_H = 24
ROW_H = 34
GUTTER = 216                # overlay width; the timeline runs underneath it
STEPS = 4
MAIN_SHARE = 0.04           # an instrument under this share of its stem's events is "minor"
MIN_TEXT_STEP_PX = 9        # note names need this many px per 16th step (~8 bars on a wide window)

# Pitched cells are coloured by PITCH CLASS (a hue per semitone, C = red
# round to B), so a line's shape reads in colour as well as height and
# the same note is the same colour in every row; the stem's own colour
# stays on the gutter strip and the unpitched cells.
_PC_HUE = [int(h) for h in np.linspace(0, 330, 12)]


def _pitch_colour(midi, vel, muted=False):
    c = QColor.fromHsl(_PC_HUE[int(midi) % 12], 200, 120 + int(50 * min(1.0, vel)))
    c.setAlpha(60 if muted else 255)
    return c


def _note_name(m):
    from lib.dj.instruments import note_name
    return note_name(m)


class InstrumentLanes(QWidget):
    seekRequested = pyqtSignal(float)
    auditionRequested = pyqtSignal(dict)      # the instrument record (its exemplar)
    eventRequested = pyqtSignal(dict, list)   # instrument, one event: play that note from the stem
    noteRequested = pyqtSignal(dict, int)     # instrument, midi: the sampler
    selectionChanged = pyqtSignal(object)     # instrument | None
    mixChanged = pyqtSignal()                 # any solo / mute changed (stem or row level)
    hint = pyqtSignal(str)                    # a click that could not be honoured, in words

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.selected = None           # instrument id the sampler plays (the last clicked)
        self.chosen = set()            # instrument ids in the SELECTION (ctrl+click adds; the sampler's target)
        self.stem_solo = set()         # stem names soloed / muted (the headers' boxes)
        self.stem_mute = set()
        self.solo = set()              # instrument ids soloed / muted (the rows' boxes)
        self.mute = set()
        self.row_mute_enabled = True   # False on the stems source: a row's M is hollow and refused
        self.octave_shift = 0
        self.result = None
        self.envs = {}                 # stem -> energy envelope (stemlanes.stem_envelope)
        self.collapsed = set()
        self.note_share = {}                     # {stem: share of its sounding bars played as notes} - the hybrid program
        self.show_all = False
        self.beats = np.zeros(0)
        self.period = 0.5
        self.down0 = 0
        self.duration = 1.0
        self.view_t0 = 0.0
        self.view_t1 = 1.0
        self.playhead = 0.0
        self.hover = None              # row index
        self.axis_widget = None        # the WaveformView whose pixel axis this panel shares
        self._cache = None             # (key, QPixmap): the timeline without playhead / lit beat
        self._rows = []                # [("stem", name, n_sounds) | ("inst", inst)]
        self._ev = {}                  # inst id -> dict of arrays
        self._names = {}               # inst id -> (name, facts)
        self.setFixedHeight(0)
        self.hide()

    # -- data -----------------------------------------------------------------
    def set_result(self, result, duration, envs=None):
        from lib.dj import instruments as INS
        self.result = result
        if envs is not None:
            self.envs = dict(envs)
        self.duration = max(float(duration), 0.001)
        self.beats = np.asarray((result or {}).get("beats") or [], dtype=np.float64)
        self.period = float((result or {}).get("period_s") or 0.5)
        self.down0 = int((result or {}).get("down0") or 0)
        self._ev, self._names = {}, {}
        if result:
            for inst in INS.instruments(result):
                ev = inst.get("events") or []
                t0 = np.array([INS.event_time(self.beats, e[0], e[1]) for e in ev], dtype=np.float64)
                t1 = np.array([INS.event_time(self.beats, e[0], e[1]) + e[3] * self.period / STEPS for e in ev], dtype=np.float64)
                midi = np.array([(-1 if e[2] is None else e[2]) for e in ev], dtype=np.int32)
                vel = np.array([e[4] for e in ev], dtype=np.float32)
                beat = np.array([e[0] for e in ev], dtype=np.int64)
                order = np.argsort(t0, kind="stable")
                self._ev[inst["id"]] = {"t0": t0[order], "t1": t1[order], "midi": midi[order], "vel": vel[order], "beat": beat[order]}
                self._names[inst["id"]] = (INS.display_name(inst), INS.facts(inst))
        self._layout()

    def set_envelopes(self, envs):
        self.envs = dict(envs or {})
        self._cache = None
        self.update()

    def set_row_mute_enabled(self, on):
        self.row_mute_enabled = bool(on)
        self._cache = None
        self.update()

    def set_show_all(self, on):
        self.show_all = bool(on)
        self._layout()

    def clear(self):
        self.set_result(None, 1.0)

    def _layout(self):
        from lib.dj import instruments as INS
        self._rows = []
        if self.result:
            for stem in INS.STEM_ORDER:
                ins = (self.result.get("stems") or {}).get(stem, {}).get("instruments") or []
                if not ins and not (stem in self.UNIT_STEMS and stem in (self.result.get("stems") or {})):
                    continue
                # a unit stem (vocals) keeps its header with no rows: the reading carries it as phrases of the
                # recording (READ_VOCALS off), and the header's S / M is how it is heard or muted
                total = sum(i["n"] for i in ins) or 1
                shown = [i for i in ins if self.show_all or i["n"] >= max(6, MAIN_SHARE * total)]
                self._rows.append(("stem", stem, len(ins), len(shown)))
                if stem in self.collapsed:
                    continue
                for inst in shown:
                    self._rows.append(("inst", inst))
        h = RULER_H + sum(HEADER_H if r[0] == "stem" else ROW_H for r in self._rows) + 2
        self.setFixedHeight(h if self._rows else 0)
        self.setVisible(bool(self._rows))
        self._cache = None
        self.update()

    def set_view(self, t0, t1):
        self.view_t0, self.view_t1 = t0, t1
        self.update()                  # the cache key carries the view: it rebuilds itself

    def set_playhead(self, t):
        self.playhead = t
        self.update()

    # -- geometry ------------------------------------------------------------
    def _axis_w(self):
        """Pixels the visible window spans: the waveform's width, so the
        two pictures share one axis (the scroll area's bar narrows THIS
        widget; the mapping must not follow that)."""
        w = self.axis_widget.width() if self.axis_widget is not None else self.width()
        return max(int(w), 1)

    def _x(self, t, W=None):
        span = max(self.view_t1 - self.view_t0, 1e-6)
        return (t - self.view_t0) / span * self._axis_w()

    def _t(self, x, W=None):
        span = max(self.view_t1 - self.view_t0, 1e-6)
        return self.view_t0 + x / self._axis_w() * span

    def _row_tops(self):
        y = RULER_H
        for r in self._rows:
            h = HEADER_H if r[0] == "stem" else ROW_H
            yield r, y, h
            y += h

    # -- paint ---------------------------------------------------------------
    def paintEvent(self, ev):
        """The timeline (ruler, envelopes, cells, gutter) is painted ONCE
        per view/size/state into a pixmap; a playhead move only blits it
        and draws the lit beat + the line on top, so the transport's
        refresh rate costs nothing here."""
        p = QPainter(self)
        if not self._rows:
            p.fillRect(self.rect(), QColor(14, 14, 18))
            return
        W, H = self.width(), self.height()
        key = (round(self.view_t0, 5), round(self.view_t1, 5), W, H, self._axis_w(), self.hover, self.selected, tuple(sorted(self.chosen)),
               tuple(sorted(self.solo)), tuple(sorted(self.mute)), tuple(sorted(self.stem_solo)), tuple(sorted(self.stem_mute)),
               self.row_mute_enabled, tuple(sorted(self.collapsed)), self.show_all, id(self.result))
        if self._cache is None or self._cache[0] != key:
            # a QImage, not a QPixmap: the raster engine draws text into an
            # image in ~10 us a label; the first pixmap text costs 16x that
            im = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
            qp = QPainter(im)
            self._paint_static(qp, W, H)
            qp.end()
            self._cache = (key, im)
        p.drawImage(0, 0, self._cache[1])
        # the beat under the playhead, lit
        if len(self.beats):
            k = int(np.searchsorted(self.beats, self.playhead, side="right") - 1)
            if 0 <= k < len(self.beats):
                nxt = self.beats[k + 1] if k + 1 < len(self.beats) else self.beats[k] + self.period
                xa, xb = self._x(float(self.beats[k])), self._x(float(nxt))
                if xb > 0 and xa < W:
                    p.fillRect(QRectF(max(0.0, xa), 0, max(1.0, xb - max(0.0, xa)), H), QColor(255, 255, 255, 22))
        x = self._x(self.playhead)
        if 0 <= x <= W:
            p.setPen(QPen(QColor(255, 255, 255, 220), 2))
            p.drawLine(int(x), 0, int(x), H)

    def _env_columns(self, env, W):
        """Per-pixel max of an envelope over the view (vectorised)."""
        aw = self._axis_w()
        span = max(self.view_t1 - self.view_t0, 1e-6)
        env_rate = RATE / ENV_HOP
        n = len(env)
        xs = np.arange(W + 1)
        j = np.clip((self.view_t0 + xs / aw * span) * env_rate, 0, n).astype(np.int64)
        j0 = j[:-1]
        out = np.zeros(W, dtype=np.float32)
        ok = j0 < n
        if not ok.any():
            return out
        starts = j0[ok]
        # reduceat over strictly increasing starts; a pixel narrower than one
        # envelope point reads that point (reduceat with equal neighbours
        # returns the element itself)
        vals = np.maximum.reduceat(env, starts)
        out[ok] = vals
        return out

    def _paint_static(self, p, W, H):
        p.fillRect(0, 0, W, H, QColor(14, 14, 18))
        span = max(self.view_t1 - self.view_t0, 1e-6)
        px_beat = self._axis_w() / (span / max(self.period, 1e-6))
        px_step = px_beat / STEPS
        small = QFont(); small.setPointSize(8)
        bold = QFont(); bold.setPointSize(9); bold.setBold(True)
        tiny = QFont(); tiny.setPointSize(7)
        fm_small = QFontMetrics(small)
        # beat / bar lines + ruler
        if len(self.beats) and px_beat >= 0.25:      # the per-line filters below thin the grid
            k0 = max(0, int(np.searchsorted(self.beats, self.view_t0)) - 1)
            k1 = min(len(self.beats), int(np.searchsorted(self.beats, self.view_t1)) + 1)
            p.setFont(tiny)
            for k in range(k0, k1):
                x = self._x(float(self.beats[k]))
                if x < 0 or x > W:
                    continue
                down = (k - self.down0) % 4 == 0
                bar = (k - self.down0) // 4
                if not down and px_beat < 9:
                    continue
                px_bar = px_beat * 4
                if down and ((px_bar < 4 and bar % 16) or (px_bar < 14 and bar % 4)):
                    continue                      # far out: every 4th / 16th bar only
                p.setPen(QPen(QColor(140, 240, 255, 120 if down else 40), 1))
                p.drawLine(int(x), RULER_H, int(x), H)
                if down and px_bar >= 26 or (down and px_bar >= 7 and bar % 4 == 0):
                    p.setPen(QColor(150, 220, 235))
                    p.drawText(int(x) + 3, RULER_H - 4, str(bar))
        p.setPen(QPen(QColor(60, 60, 72), 1))
        p.drawLine(0, RULER_H - 1, W, RULER_H - 1)
        for ri, (row, y0, rh) in enumerate(self._row_tops()):
            if row[0] == "stem":
                _k, stem, n_all, n_shown = row
                col = QColor(LANE_COLORS.get(stem, QColor(200, 200, 200)))
                muted = not self.stem_audible(stem)
                band = QColor(col); band.setAlpha(28 if not muted else 12)
                p.fillRect(QRectF(0, y0, W, rh), band)
                # the stem's energy envelope, folded in from the stem lanes
                env = self.envs.get(stem)
                if env is not None and len(env) > 2:
                    fill = QColor(col); fill.setAlpha(70 if not muted else 25)
                    p.setPen(QPen(fill, 1))
                    base = y0 + rh - 1
                    cols_v = self._env_columns(np.asarray(env, dtype=np.float32), W)
                    for x in np.where(cols_v > 0.004)[0]:
                        hh = float(cols_v[x]) * (rh - 4)
                        p.drawLine(int(x), int(base - hh), int(x), int(base))
                p.setPen(QPen(QColor(70, 70, 82), 1))
                p.drawLine(0, int(y0), W, int(y0))
                continue
            inst = row[1]
            stem = inst["stem"]
            col = QColor(LANE_COLORS.get(stem, QColor(200, 200, 200)))
            silent = not self._audible_now(inst["id"])
            if silent:
                col.setAlpha(90)
            if inst["id"] in self.chosen:
                p.fillRect(QRectF(0, y0, W, rh), QColor(255, 220, 120, 22 if inst["id"] != self.selected else 34))
            if ri == self.hover:
                p.fillRect(QRectF(0, y0, W, rh), QColor(255, 255, 255, 14))
            p.setPen(QPen(QColor(40, 40, 50), 1))
            p.drawLine(0, int(y0 + rh - 1), W, int(y0 + rh - 1))
            # cells (the gutter is painted over them afterwards)
            d = self._ev.get(inst["id"])
            if d is None or not len(d["t0"]):
                continue
            t0, t1, midi, vel, beat = d["t0"], d["t1"], d["midi"], d["vel"], d["beat"]
            i0 = max(0, int(np.searchsorted(t1, self.view_t0)) - 1)
            i1 = min(len(t0), int(np.searchsorted(t0, self.view_t1)) + 1)
            rng = inst.get("range") or [60, 60]
            lo, hi = int(rng[0]), int(rng[1])
            pitched = bool(inst.get("pitched")) and hi >= lo
            sustain = inst["kind"] == "sustain"
            text_ok = px_step >= MIN_TEXT_STEP_PX
            p.setFont(small)
            top, bottom = y0 + 3, y0 + rh - 4
            drawn_text_at = -1e9
            for i in range(i0, i1):
                if t1[i] < self.view_t0 or t0[i] > self.view_t1:
                    continue
                xa, xb = max(0.0, self._x(float(t0[i]))), self._x(float(t1[i]))
                v = float(vel[i])
                c = QColor(col); c.setAlpha(int(110 + 140 * min(1.0, v)) if not silent else 60)
                if pitched and midi[i] >= 0:
                    c = _pitch_colour(int(midi[i]), v, silent)
                p.setPen(Qt.PenStyle.NoPen); p.setBrush(c)
                if pitched and midi[i] >= 0:
                    frac = (int(midi[i]) - lo) / max(hi - lo, 1)
                    yy = bottom - frac * (bottom - top - 6)
                    if sustain:
                        p.drawRect(QRectF(xa, yy - 2, max(2.0, xb - xa - 1), 5))
                    else:
                        # the head marks the strike; the measured ring-out trails it thinner
                        head = max(2.0, min(xb - xa - 1, px_step * 0.85))
                        p.drawRect(QRectF(xa, yy - 3, head, 7))
                        if xb - xa - 1 > head + 2:
                            tail = QColor(c); tail.setAlpha(int(c.alpha() * 0.45))
                            p.setBrush(tail)
                            p.drawRect(QRectF(xa + head, yy - 1, xb - xa - 1 - head, 3))
                            p.setBrush(c)
                    if text_ok:
                        # one label per beat for a chord: the names of every note starting here
                        if sustain and i > 0 and beat[i] == beat[i - 1] and abs(t0[i] - t0[i - 1]) < 1e-6:
                            continue
                        names = [_note_name(int(midi[i]))]
                        if sustain:
                            j = i + 1
                            while j < len(t0) and beat[j] == beat[i] and abs(t0[j] - t0[i]) < 1e-6:
                                names.append(_note_name(int(midi[j]))); j += 1
                        label = " ".join(names)
                        tw = fm_small.horizontalAdvance(label)
                        if xa - drawn_text_at >= tw + 3:
                            p.setPen(QColor(245, 245, 250))
                            ty = int(yy) - 5 if frac < 0.5 else int(yy) + 12
                            p.drawText(int(xa) + 1, max(int(top) + 8, min(int(bottom), ty)), label)
                            drawn_text_at = xa
                else:
                    hpx = 4 + v * (rh - 12)
                    wpx = max(2.0, min(px_step * 0.7, 12.0))
                    p.drawRect(QRectF(xa, bottom - hpx, wpx, hpx))
        # the GUTTER: a translucent overlay on the left (the timeline keeps the
        # waveform's axis underneath it), fading out at its right edge
        p.setPen(Qt.PenStyle.NoPen)
        p.fillRect(QRectF(0, 0, GUTTER - 24, H), QColor(14, 14, 18, 215))
        grad = QLinearGradient(GUTTER - 24, 0, GUTTER, 0)
        grad.setColorAt(0.0, QColor(14, 14, 18, 215))
        grad.setColorAt(1.0, QColor(14, 14, 18, 0))
        p.fillRect(QRectF(GUTTER - 24, 0, 24, H), grad)
        for ri, (row, y0, rh) in enumerate(self._row_tops()):
            if row[0] == "stem":
                _k, stem, n_all, n_shown = row
                col = QColor(LANE_COLORS.get(stem, QColor(200, 200, 200)))
                muted = not self.stem_audible(stem)
                p.fillRect(QRectF(0, y0 + 3, 5, rh - 6), col if not muted else QColor(col.red(), col.green(), col.blue(), 90))
                p.setFont(bold)
                p.setPen(QColor(235, 235, 245) if not muted else QColor(140, 140, 150))
                tri = "▸" if stem in self.collapsed else "▾"
                extra = f"  ({n_all - n_shown} hidden)" if n_shown < n_all else ""
                what = "phrases of the recording" if (n_all == 0 and stem in self.UNIT_STEMS) else f"{n_all} sound{'s' if n_all != 1 else ''}{extra}"
                share = self.note_share.get(stem)
                if share is not None:
                    # the hybrid program: what part of this stem plays as notes, the rest is the recording
                    what += f"  ·  notes on {100 * share:.0f}%"
                p.drawText(QRectF(12, y0, GUTTER - 76, rh), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                           f"{tri} {stem}  ·  {what}")
                draw_sm(p, self.SM_X, y0 + 4, stem in self.stem_solo, stem in self.stem_mute, small)
                continue
            inst = row[1]
            name, facts = self._names.get(inst["id"], (inst["id"], ""))
            silent = not self._audible_now(inst["id"])
            if inst["id"] in self.chosen:
                p.fillRect(QRectF(0, y0 + 3, 5, rh - 6), QColor(255, 220, 120))
                name = ("▶ " if inst["id"] == self.selected else "✓ ") + name
            if silent:
                p.fillRect(QRectF(0, y0, W, rh), QColor(14, 14, 18, 120))
            p.setFont(bold); p.setPen(QColor(235, 235, 245) if not silent else QColor(150, 150, 160))
            name_w = GUTTER - 76
            if inst.get("explained") is not None:
                # how much of the stem this voice accounts for in its own windows (lib/dj/explain.py)
                share = float(inst["explained"])
                col = QColor(120, 200, 130) if share >= 0.6 else (QColor(220, 180, 90) if share >= 0.25 else QColor(220, 110, 100))
                p.setFont(tiny); p.setPen(col if not silent else QColor(col.red(), col.green(), col.blue(), 120))
                p.drawText(QRectF(GUTTER - 112, y0 + 3, 40, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{100 * share:.0f}%")
                name_w = GUTTER - 116
                p.setFont(bold); p.setPen(QColor(235, 235, 245) if not silent else QColor(150, 150, 160))
            p.drawText(QRectF(12, y0 + 2, name_w, 16), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       QFontMetrics(bold).elidedText(name, Qt.TextElideMode.ElideRight, int(name_w)))
            p.setFont(small); p.setPen(QColor(150, 150, 165) if not silent else QColor(105, 105, 115))
            p.drawText(QRectF(12, y0 + 17, GUTTER - 82, 14), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       fm_small.elidedText(facts, Qt.TextElideMode.ElideRight, GUTTER - 84))
            p.setPen(QColor(110, 110, 125)); p.setFont(tiny)
            p.drawText(QRectF(GUTTER - 72, y0 + 18, 46, 12), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, inst["id"])
            if self._row_has_sm(inst):
                draw_sm(p, self.SM_X, y0 + 3, inst["id"] in self.solo, inst["id"] in self.mute, small)
                if not self.row_mute_enabled and inst["id"] not in self.mute:
                    # hollow M: the stems source cannot drop one sound
                    r = QRectF(self.SM_X + SM_W + SM_GAP, y0 + 3, SM_W, SM_H)
                    p.fillRect(r, QColor(22, 22, 28)); p.setPen(QColor(70, 70, 82)); p.setFont(small)
                    p.drawText(r, Qt.AlignmentFlag.AlignCenter, "M")
                    p.setPen(QPen(QColor(70, 70, 82), 1)); p.setBrush(Qt.BrushStyle.NoBrush); p.drawRect(r)
    # -- interaction ------------------------------------------------------------
    def _row_at(self, y):
        for ri, (row, y0, rh) in enumerate(self._row_tops()):
            if y0 <= y < y0 + rh:
                return ri, row
        return None, None

    def mouseMoveEvent(self, ev):
        ri, row = self._row_at(ev.position().y())
        if ri != self.hover:
            self.hover = ri
            if row is not None and row[0] == "inst":
                inst = row[1]
                name, facts = self._names.get(inst["id"], (inst["id"], ""))
                hint = f"role guess: {inst['hint']}\n" if inst.get("hint") else ""
                if inst.get("explained") is not None:
                    hint += (f"explains {100 * inst['explained']:.0f}% of the stem in its own windows"
                             f" (overshoot {100 * inst.get('overshoot', 0):.0f}%, {100 * inst.get('spurious', 0):.0f}% of its events in silence)\n")
                sm = ("S = solo (on the stems source: the recording gated to this sound's hits), M = mute (reconstruction source only)"
                      if self._row_has_sm(inst) else "the vocals are one unit: the stem header's S / M are theirs")
                self.setToolTip(f"{inst['id']}  {name}\n{hint}{facts}\n{sm}; "
                                "click the name to hear its exemplar and select it for the sampler; click a note to hear that note")
            elif row is not None:
                self.setToolTip(f"{row[1]} stem: energy envelope (what the separation extracted); S / M solo or mute the whole stem; "
                                "click the name to collapse")
            self.update()

    def leaveEvent(self, ev):
        self.hover = None
        self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton or not self._rows:
            return
        x, y = ev.position().x(), ev.position().y()
        ri, row = self._row_at(y)
        if row is not None and x < GUTTER:
            # the S / M boxes span the row's whole height for the hit test (an easy target): only x decides
            sm = sm_hit(x, 0, self.SM_X, 0)
            if sm and row[0] == "stem":
                self.toggle_stem(row[1], sm)
                return
            if sm and row[0] == "inst":
                inst = row[1]
                if not self._row_has_sm(inst):
                    self.hint.emit("the vocals are one unit in the reconstruction (reused phrases, not voices): "
                                   "use the vocals header's S / M")
                elif sm == "M" and not self.row_mute_enabled and inst["id"] not in self.mute:
                    self.hint.emit("the stems cannot drop one sound - switch 'hear' to the reconstruction to mute a row "
                                   "(S works here: the recording gated to the sound's hits)")
                else:
                    self.toggle_rows([inst["id"]], sm)
                return
        if row is not None and row[0] == "stem" and x < GUTTER:
            stem = row[1]
            if stem in self.collapsed:
                self.collapsed.discard(stem)
            else:
                self.collapsed.add(stem)
            self._layout()
            return
        if x < GUTTER:
            if row is not None and row[0] == "inst":
                self.select(row[1], add=bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier))
                self.auditionRequested.emit(row[1])
            return
        t = self._t(x, self.width())
        if row is not None and row[0] == "inst":
            ev = self._event_at(row[1], t)
            if ev is not None:
                self.select(row[1])
                self.eventRequested.emit(row[1], list(ev))
                return
        self.seekRequested.emit(float(np.clip(t, 0.0, self.duration)))

    # -- solo / mute --------------------------------------------------------------
    SM_X = GUTTER - 66                        # the two boxes at the gutter's right edge
    UNIT_STEMS = ("vocals",)                  # stems that are one unit: no row-level boxes

    def set_note_share(self, share):
        """{stem: 0..1} from the built program (stats["note_share"]): shown in each stem's header."""
        self.note_share = dict(share or {})
        self.update()

    def _all_ids(self):
        from lib.dj import instruments as INS
        return [i["id"] for i in INS.instruments(self.result)] if self.result else []

    def _stem_ids(self, stem):
        return [i for i in self._all_ids() if i.split(".")[0] == stem]

    def _row_has_sm(self, inst):
        return inst["stem"] not in self.UNIT_STEMS

    def stem_audible(self, stem):
        if self.stem_solo:
            return stem in self.stem_solo
        return stem not in self.stem_mute

    def _row_audible(self, iid):
        if self.solo:
            return iid in self.solo
        return iid not in self.mute

    def _audible_now(self, iid):
        return self.stem_audible(iid.split(".")[0]) and self._row_audible(iid)

    def stems_audible(self):
        """The stems that sound at stem level (in STEM_ORDER)."""
        from lib.dj import instruments as INS
        return [s for s in INS.STEM_ORDER if self.stem_audible(s)]

    def audible(self):
        """The instrument ids that should sound: both levels applied."""
        return [i for i in self._all_ids() if self._audible_now(i)]

    def event_windows(self, iid, min_s=0.15, pre_s=0.01, post_s=0.05):
        """[(a_s, b_s)] where the instrument sounds: each event from just
        before its onset to its measured ring-out (at least min_s) - the
        gate the stems source applies to the recording for a row solo."""
        d = self._ev.get(iid)
        if d is None or not len(d["t0"]):
            return []
        a = d["t0"] - pre_s
        b = np.maximum(d["t1"], d["t0"] + min_s) + post_s
        return list(zip(a.tolist(), b.tolist()))

    def row_solo_ids(self):
        """The soloed rows that sound (their stem is audible) - the stems
        source gates the recording to these sounds' hits."""
        return [i for i in self._all_ids() if i in self.solo and self.stem_audible(i.split(".")[0])]

    def toggle_stem(self, stem, which):
        target = self.stem_solo if which == "S" else self.stem_mute
        if stem in target:
            target.discard(stem)
        else:
            target.add(stem)
        self._changed()

    def toggle_rows(self, ids, which):
        ids = list(ids)
        target = self.solo if which == "S" else self.mute
        if all(i in target for i in ids):
            for i in ids:
                target.discard(i)
        else:
            target.update(ids)
        self._changed()

    def clear_solo_mute(self, emit=True):
        self.solo.clear(); self.mute.clear()
        self.stem_solo.clear(); self.stem_mute.clear()
        self._changed(emit)

    def _changed(self, emit=True):
        self._cache = None
        self.update()
        if emit:
            self.mixChanged.emit()

    def select(self, inst, add=False):
        """Plain click: this instrument alone; ctrl+click: toggle it in the
        selection (the last clicked one is what the sampler plays)."""
        new = inst["id"] if inst else None
        if add and new is not None:
            if new in self.chosen and new == self.selected:
                self.chosen.discard(new)
                new = next(iter(sorted(self.chosen)), None)
            else:
                self.chosen.add(new)
        else:
            self.chosen = {new} if new else set()
        self.selected = new
        self._cache = None
        self.update()
        self.selectionChanged.emit(inst if new else None)
        self.setFocus()

    def chosen_instruments(self):
        return [row[1] for row in self._rows if row[0] == "inst" and row[1]["id"] in self.chosen]

    def _event_at(self, inst, t):
        """The event of `inst` sounding at time t (its cell, or its ring-out), else None."""
        d = self._ev.get(inst["id"])
        if d is None or not len(d["t0"]):
            return None
        span = max(self.view_t1 - self.view_t0, 1e-6)
        slop = span / self._axis_w() * 3            # three pixels of grace either side
        i = int(np.searchsorted(d["t0"], t + slop, side="right")) - 1
        best = None
        while i >= 0 and d["t0"][i] >= t - 4.0:
            if d["t0"][i] - slop <= t <= max(d["t1"][i], d["t0"][i] + self.period / STEPS) + slop:
                best = i
                break
            i -= 1
        if best is None:
            return None
        for inst_ev in inst["events"]:
            if inst_ev[0] == int(d["beat"][best]) and (-1 if inst_ev[2] is None else inst_ev[2]) == int(d["midi"][best])                     and abs(float(d["vel"][best]) - inst_ev[4]) < 1e-6:
                return inst_ev
        return None

    # -- the sampler ------------------------------------------------------------
    _KEYS_LOW = "zsxdcvgbhnjm"
    _KEYS_HIGH = "q2w3er5t6y7u"

    def _selected_inst(self):
        for row in self._rows:
            if row[0] == "inst" and row[1]["id"] == self.selected:
                return row[1]
        return None

    def keyPressEvent(self, ev):
        inst = self._selected_inst()
        txt = ev.text().lower()
        if inst is None or not txt:
            return super().keyPressEvent(ev)
        if txt == "[":
            self.octave_shift -= 1; return
        if txt == "]":
            self.octave_shift += 1; return
        semis = None
        if txt in self._KEYS_LOW:
            semis = self._KEYS_LOW.index(txt)
        elif txt in self._KEYS_HIGH:
            semis = 12 + self._KEYS_HIGH.index(txt)
        if semis is None:
            return super().keyPressEvent(ev)
        base = inst.get("exemplar_midi")
        if base is None:
            rng = inst.get("range") or [60, 60]
            base = int(rng[0])
        root_c = (base // 12) * 12 + 12 * self.octave_shift   # the C below the exemplar's octave
        self.noteRequested.emit(inst, int(root_c + semis))
