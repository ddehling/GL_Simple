"""Analysis tab: ingest a song, read it as the generator's own commands,
recreate it, score the recreation locally and globally, and send the
script to the show.

  DJ track   THE source: pick a track of the DJ library (stems rendered)
             and the planner's instrument reading of it becomes this
             tab's song - lib/dj/gen_link writes script + samples +
             features under logs/analysis/<title>/ (running the pass
             first when the track has none) and the tab opens it. There
             is no file browsing here any more: the library is where
             songs are analysed (tools/dj_planner.py), this tab is where
             they are recreated and varied.
  Replay     the PROGRAMMATIC recreation: every note of the reading
             played by the song's own sampled instruments on its grid
             (lib/dj/gen_link.replay -> replay.wav), nothing composed;
             side B, scorable (Final Voyage: 83 global / 96 structure,
             against 75 for the composed recreation)
  Fidelity   the two honest readouts of the linked song, written by the
             link and shown under the info line (lib/dj/fidelity.py):
             NOTES AS CODE - how much of the reading the program language
             holds (events per pattern+op, share of events carried by
             patterns vs written one by one, verbatim bars, vocal phrase
             reuse, per voice how much of its stem it explains) - and
             WAVEFORM vs THE ORIGINAL - the program rendered per stem
             against the library's real stems (spectral gap in dB, level,
             rhythm correlation, missed/extra 16ths, onset F1, chroma)
             and the mix. The button recomputes it.
  Recreate   the generator's rendition of the script (lib/gen/script.render)
             to logs/analysis/<name>/recreation.wav - the same kit, banks,
             pad, drum bars and lines, interpreted by the composer
  Score      per-phrase + global scores (lib/gen/analysis/score.py),
             drawn as two energy strips (original / recreation) and a
             score bar per phrase; the weakest phrases listed
  Save       script.yaml (edits in the table are applied first)
  Play       send the script to the running show (the "script" action)

All heavy work runs on a worker thread; the tab polls it at the
console's refresh rate. The table edits section / bars / energy /
density / brightness / swing / layers / chords in place."""
from __future__ import annotations

import json
import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QRect
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QImage, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog,
                             QTableWidget, QTableWidgetItem, QPlainTextEdit, QProgressBar, QSplitter, QCheckBox, QSlider,
                             QDialog, QListWidget, QListWidgetItem, QScrollArea)

COLS = ["section", "bars", "energy", "density", "brightness", "swing", "layers", "chords", "lanes"]
SECTION_COLOURS = {"intro": "#4a5a7a", "groove": "#3f7a5a", "build": "#a8772a", "drop": "#b03a3a", "break": "#5a4a8a",
                   "outro": "#4a4a4a", "flow": "#3f7a7a", "swell": "#a85a2a", "calm": "#4a6a8a"}


def _score_colour(v):
    v = max(0.0, min(100.0, float(v))) / 100.0
    r = int(220 * (1.0 - v) + 60 * v)
    g = int(80 * (1.0 - v) + 200 * v)
    return QColor(r, g, 70)


def _tone(good, mid, v, higher_is_better=True):
    """'#rrggbb' for a figure: green past `good`, amber past `mid`, red otherwise."""
    if v is None:
        return "#888"
    ok = (v >= good) if higher_is_better else (v <= good)
    so = (v >= mid) if higher_is_better else (v <= mid)
    return "#6fc47a" if ok else ("#d9b25a" if so else "#d96f64")


def fidelity_html(rep):
    """The two honest readouts of a linked song (lib/dj/fidelity.py) as a
    compact rich-text block: how much of the notes the program language
    holds, and how close its render is to the original stems."""
    n, w = rep.get("notes") or {}, rep.get("waveform") or {}
    if not n and not w:
        return ""
    css = "font-size:11px; color:#ccc"
    td = "style='padding:1px 8px 1px 0'"
    out = [f"<div style='{css}'>"]
    if n:
        out.append("<b>NOTES AS CODE</b> &nbsp;"
                   f"{n['events']} events &rarr; {n['patterns']} patterns over {n['entries']} bar entries + {n['ops']} ops "
                   f"(<b style='color:{_tone(8, 4, n['events_per_entry'])}'>{n['events_per_entry']:.1f}</b> events per pattern+op); "
                   f"<span style='color:{_tone(0.85, 0.6, n['pattern_share'])}'>{100 * n['pattern_share']:.0f}%</span> of events carried by patterns, "
                   f"{100 * n['op_share']:.0f}% written one by one; "
                   f"verbatim <span style='color:{_tone(0.1, 0.4, n['verbatim_share'], False)}'>{n['verbatim_bars']}/{n['bars']} bars</span>; "
                   f"vocals {n['vocal_phrases']} phrases from {n['vocal_distinct']} recordings")
        rows = []
        for v in n.get("voices") or []:
            ex = v.get("explained")
            exs = f"<span style='color:{_tone(0.6, 0.25, ex)}'>{100 * ex:.0f}%</span>" if ex is not None else "<span style='color:#888'>–</span>"
            rows.append(f"<tr><td {td}>{v['id']}</td><td {td}>{v.get('name') or ''}</td><td {td}>{v.get('model') or ''}</td>"
                        f"<td {td} align=right>{v['patterns']}</td><td {td} align=right>{v['units']}</td><td {td} align=right>{v['ops']}</td><td {td} align=right>{exs}</td></tr>")
        for g in n.get("pruned") or []:
            rows.append(f"<tr><td {td} style='color:#999'>{g['id']}</td><td {td} colspan=6 style='color:#999'>dropped: {g['why'][:110]}</td></tr>")
        if rows:
            out.append("<table cellspacing=0 style='margin-top:2px'><tr style='color:#999'><th align=left>voice</th><th align=left>sound</th><th align=left>model</th>"
                       "<th align=right>patterns</th><th align=right>units</th><th align=right>ops</th><th align=right>explains its stem</th></tr>" + "".join(rows) + "</table>")
        ag = n.get("agreement") or {}
        if ag:
            cells = []
            for stem, r in ag.items():
                cells.append(f"{stem} <b style='color:{_tone(0.8, 0.5, r['f1'])}'>{r['f1']:.2f}</b> "
                             f"<span style='color:#999'>(octave-blind {r['f1_octave_blind']:.2f}, onsets {r['onset_f1']:.2f}, {r['n_program']} vs {r['n_transcribed']} notes)</span>")
            out.append("<div style='margin-top:3px'><b>NOTES vs AN INDEPENDENT TRANSCRIPTION</b> &nbsp;F1 at 60 ms and the same pitch &mdash; "
                       "the figure that says whether the notes are the song's: &nbsp;" + " &nbsp;·&nbsp; ".join(cells) + "</div>")
    if w:
        t0, t1 = w.get("window", [0, 0])
        out.append(f"<b style='margin-top:4px'>WAVEFORM vs THE ORIGINAL</b> &nbsp;({t0:.0f}–{t1:.0f} s; gap in dB: 0 identical, 6–8 the same part on another instrument, 12+ a different sound)")
        rows = []
        for stem in ("drums", "bass", "other", "vocals", "mix"):
            r = w.get(stem)
            if not r:
                continue
            gap = r.get("spectral")
            ch = r.get("chroma_r")
            note = ("the stem’s own phrases reused" if r.get("reused") else
                    (f"{r.get('voices', 0)} voices" if stem != "mix" else "all stems together"))
            rows.append(f"<tr><td {td}><b>{stem}</b></td>"
                        f"<td {td} align=right><b style='color:{_tone(6, 12, gap, False)}'>{gap:.1f} dB</b></td>"
                        f"<td {td} align=right>{r['level']:+.1f} dB</td>"
                        f"<td {td} align=right style='color:{_tone(0.8, 0.5, r['activity_r'])}'>{r['activity_r']:.2f}</td>"
                        f"<td {td} align=right>{100 * r['missed']:.0f}%</td><td {td} align=right>{100 * r['spurious']:.0f}%</td>"
                        f"<td {td} align=right style='color:{_tone(0.8, 0.5, r['onset_f1'])}'>{r['onset_f1']:.2f}</td>"
                        f"<td {td} align=right>{(f'{ch:.2f}' if ch is not None else '–')}</td>"
                        f"<td {td} style='color:#999'>{note}</td></tr>")
        out.append("<table cellspacing=0 style='margin-top:2px'><tr style='color:#999'><th align=left>stem</th><th align=right>gap</th><th align=right>level</th>"
                   "<th align=right>rhythm r</th><th align=right>missed</th><th align=right>extra</th><th align=right>onsets F1</th><th align=right>chroma</th><th></th></tr>"
                   + "".join(rows) + "</table>")
    wp = rep.get("waveform_played") or {}
    if wp:
        # the hybrid program: the table above is the voices alone (the reader's figure); this is what plays -
        # notes where the voices explain the recording bar by bar, the recording itself elsewhere
        share = (rep.get("notes") or {}).get("note_share") or {}
        cells = []
        for stem in ("drums", "bass", "other", "vocals", "mix"):
            r = wp.get(stem)
            if not r:
                continue
            ns = (f" <span style='color:{_tone(0.8, 0.4, share[stem])}'>notes on {100 * share[stem]:.0f}%</span>" if stem in share
                  else (" <span style='color:#999'>phrases</span>" if stem == "vocals" else ""))
            cells.append(f"{stem} <b style='color:{_tone(6, 12, r['spectral'], False)}'>{r['spectral']:.1f} dB</b>{ns}")
        out.append("<div style='margin-top:3px'><b>AS PLAYED</b> &nbsp;the hybrid program (notes where the voices explain the recording bar by bar, "
                   "the recording itself elsewhere): &nbsp;" + " &nbsp;·&nbsp; ".join(cells) + "</div>")
    out.append("</div>")
    return "".join(out)


class ScoreStrip(QWidget):
    """Original vs recreation energy per bar, section blocks, and the local score per window."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(90)             # the tab has to fit a laptop screen with everything below it
        self.orig = []
        self.recon = []
        self.report = None
        self.script = None

    def set(self, orig=None, recon=None, report=None, script=None):
        if orig is not None:
            self.orig = orig
        if recon is not None:
            self.recon = recon
        self.report = report
        if script is not None:
            self.script = script
        self.update()

    def paintEvent(self, ev):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QColor("#16181d"))
        n = max(len(self.orig), len(self.recon), 1)
        bw = w / n
        font = QFont(); font.setPointSize(8); qp.setFont(font)
        # sections from the script
        if self.script:
            bar = 0
            for e in self.script.get("sections", []):
                x0, x1 = bar * bw, (bar + e["bars"]) * bw
                col = QColor(SECTION_COLOURS.get(e["section"], "#555")); col.setAlpha(120)
                qp.setPen(QPen(QColor("#0e1013"))); qp.setBrush(QBrush(col))
                qp.drawRect(QRectF(x0, 2, max(1.0, x1 - x0 - 1), 16))
                if x1 - x0 > 40:
                    qp.setPen(QPen(QColor("#f0f2f5"))); qp.drawText(int(x0) + 3, 14, f"{e['section']} {e['bars']}")
                bar += e["bars"]
        # energy curves (dB, normalised to the original's range)
        def curve(feats, y0, y1, colour):
            if not feats:
                return
            vals = np.array([f["energy_db"] for f in feats])
            lo, hi = float(np.percentile(vals, 5)) - 3, float(np.percentile(vals, 98)) + 1
            qp.setPen(QPen(QColor(colour), 2))
            last = None
            for i, v in enumerate(vals):
                y = y1 - (min(max(v, lo), hi) - lo) / max(hi - lo, 1e-6) * (y1 - y0)
                x = (i + 0.5) * bw
                if last is not None:
                    qp.drawLine(int(last[0]), int(last[1]), int(x), int(y))
                last = (x, y)
        curve(self.orig, 24, 70, "#7fd1a8")
        curve(self.recon, 24, 70, "#6cc3ff")
        qp.setPen(QPen(QColor("#7fd1a8"))); qp.drawText(4, 34, "original")
        qp.setPen(QPen(QColor("#6cc3ff"))); qp.drawText(4, 46, "recreation")
        # local scores
        if self.report:
            for r in self.report.get("local", []):
                x0, x1 = r["bar0"] * bw, (r["bar0"] + r["bars"]) * bw
                col = _score_colour(r["score"])
                qp.setPen(QPen(QColor("#0e1013"))); qp.setBrush(QBrush(col))
                hh = 40 * r["score"] / 100.0
                qp.drawRect(QRectF(x0, h - 12 - hh, max(1.0, x1 - x0 - 1), hh))
                if x1 - x0 > 26:
                    qp.setPen(QPen(QColor("#f0f2f5"))); qp.drawText(int(x0) + 2, h - 2, f"{r['score']:.0f}")
            qp.setPen(QPen(QColor("#c8ccd4"))); qp.drawText(4, h - 56, f"global {self.report['global']:.1f}")
        qp.end()


class CompareView(QWidget):
    """Original over recreation: high-resolution log-frequency spectrograms
    (lib/gen/analysis/spectro.py) on one time axis, the recreation shifted
    so its bar 0 sits under the original's first downbeat; bar ticks,
    section labels, the play cursor. Click seeks, the wheel zooms, drag
    scrolls, the view follows the cursor while playing."""

    def __init__(self, on_seek=None):
        super().__init__()
        self.setMinimumHeight(170)
        self.a = None            # {"rgb","fps","seconds"} original
        self.b = None            # recreation
        self.offset_s = 0.0      # recreation display offset (= original first downbeat)
        self.bars = []           # original bar times (s)
        self.sections = []       # [(display_s, label)]
        self.window_s = 60.0
        self.view_start = 0.0
        self.cursor_s = 0.0      # display time
        self.which = "a"
        self.follow = True
        self.on_seek = on_seek
        self._drag = None
        self.setMouseTracking(True)

    def set_sources(self, a=None, b=None, offset_s=None, bars=None, sections=None):
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if offset_s is not None:
            self.offset_s = float(offset_s)
        if bars is not None:
            self.bars = list(bars)
        if sections is not None:
            self.sections = list(sections)
        self.update()

    def total_s(self):
        return max((self.a or {}).get("seconds", 0.0), (self.b or {}).get("seconds", 0.0) + self.offset_s, 1.0)

    def set_cursor(self, display_s, which):
        self.cursor_s = float(display_s)
        self.which = which
        if self.follow and (display_s < self.view_start or display_s > self.view_start + self.window_s * 0.92):
            self.view_start = max(0.0, display_s - self.window_s * 0.15)
        self.update()

    def _x(self, t, w):
        return (t - self.view_start) / self.window_s * w

    def _row(self, qp, src, rect, label, w):
        qp.fillRect(rect, QColor("#0b0c10"))
        if src is not None and src.get("rgb") is not None:
            fps = float(src["fps"])
            shift = self.offset_s if label == "recreation" else 0.0
            f0 = int(max(0.0, (self.view_start - shift) * fps))
            f1 = int(min(src["rgb"].shape[0], (self.view_start + self.window_s - shift) * fps))
            if f1 > f0:
                sl = np.ascontiguousarray(src["rgb"][f0:f1].transpose(1, 0, 2)[::-1])      # (bins, frames, 3), top = high
                img = QImage(sl.data, sl.shape[1], sl.shape[0], sl.shape[1] * 3, QImage.Format.Format_RGB888)
                x0 = self._x(f0 / fps + shift, w)
                x1 = self._x(f1 / fps + shift, w)
                qp.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                qp.drawImage(QRectF(x0, rect.top(), max(1.0, x1 - x0), rect.height()), img)
        qp.setPen(QPen(QColor("#e8eaf0"))); qp.drawText(int(rect.left()) + 6, int(rect.top()) + 12, label)
        # frequency guide
        qp.setPen(QPen(QColor("#5a606c")))
        for hz, frac in ((100, 0.19), (1000, 0.56), (10000, 0.92)):
            y = rect.bottom() - frac * rect.height()
            qp.drawText(int(rect.right()) - 34, int(y), f"{hz // 1000}k" if hz >= 1000 else str(hz))

    def paintEvent(self, ev):
        qp = QPainter(self)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QColor("#16181d"))
        font = QFont(); font.setPointSize(8); qp.setFont(font)
        top_h = 16
        row_h = (h - top_h - 18) / 2.0
        ra = QRectF(0, top_h, w, row_h - 2)
        rb = QRectF(0, top_h + row_h, w, row_h - 2)
        self._row(qp, self.a, ra, "original", w)
        self._row(qp, self.b, rb, "recreation", w)
        # bars and sections
        bar_len = None
        if len(self.bars) > 1:
            bar_len = float(np.median(np.diff(self.bars)))
        if bar_len and bar_len > 0:
            px_per_bar = bar_len / self.window_s * w
            every = 1 if px_per_bar > 40 else (4 if px_per_bar > 10 else 16)
            for i, t in enumerate(self.bars):
                if t < self.view_start - bar_len or t > self.view_start + self.window_s:
                    continue
                if i % every:
                    continue
                x = self._x(t, w)
                qp.setPen(QPen(QColor(255, 255, 255, 70 if i % 4 else 130)))
                qp.drawLine(int(x), top_h, int(x), h - 18)
                if px_per_bar * every > 28:
                    qp.setPen(QPen(QColor("#9aa0ac"))); qp.drawText(int(x) + 2, h - 6, str(i))
        for t, label in self.sections:
            if self.view_start - 1 <= t <= self.view_start + self.window_s:
                x = self._x(t, w)
                qp.setPen(QPen(QColor("#ffd166"), 2)); qp.drawLine(int(x), 0, int(x), top_h)
                qp.drawText(int(x) + 3, 11, label)
        # time ruler
        qp.setPen(QPen(QColor("#5a606c")))
        step = 5 if self.window_s <= 60 else (30 if self.window_s <= 400 else 60)
        t = int(self.view_start // step) * step
        while t < self.view_start + self.window_s:
            x = self._x(t, w)
            qp.drawText(int(x) + 2, top_h + 12 + 0, "")
            t += step
        # cursor
        xc = self._x(self.cursor_s, w)
        qp.setPen(QPen(QColor("#ffffff"), 2)); qp.drawLine(int(xc), top_h, int(xc), h - 18)
        qp.setPen(QPen(QColor("#e8eaf0")))
        qp.drawText(int(xc) + 4, h - 22, f"{int(self.cursor_s // 60)}:{self.cursor_s % 60:05.2f}  ({'A' if self.which == 'a' else 'B'})")
        qp.end()

    # -- interaction ---------------------------------------------------------------
    def _t_at(self, x):
        return self.view_start + x / max(1, self.width()) * self.window_s

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag = (ev.position().x(), self.view_start, False)
            t = self._t_at(ev.position().x())
            row = "b" if ev.position().y() > self.height() / 2 else "a"
            if self.on_seek:
                self.on_seek(t, row)

    def mouseMoveEvent(self, ev):
        if self._drag and (ev.buttons() & Qt.MouseButton.LeftButton):
            x0, vs0, _ = self._drag
            dx = ev.position().x() - x0
            if abs(dx) > 4:
                self.view_start = max(0.0, vs0 - dx / max(1, self.width()) * self.window_s)
                self._drag = (x0, vs0, True)
                self.update()

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def wheelEvent(self, ev):
        t = self._t_at(ev.position().x())
        factor = 0.8 if ev.angleDelta().y() > 0 else 1.25
        new_w = float(max(4.0, min(self.total_s(), self.window_s * factor)))
        frac = (t - self.view_start) / self.window_s
        self.window_s = new_w
        self.view_start = max(0.0, t - frac * new_w)
        self.update()


class BeatGrid(QWidget):
    """The beat as evidence: the selected section's folded onset strengths
    (kick / snare / hat, 16 steps) with the hits the script keeps marked,
    plus the grid facts (tempo, beat length, first downbeat, swing, kind)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(72)
        self.grid = None
        self.hits = None
        self.facts = ""
        self.title = ""

    def set(self, grid=None, hits=None, facts="", title=""):
        self.grid, self.hits, self.facts, self.title = grid, hits, facts, title
        self.update()

    def paintEvent(self, ev):
        qp = QPainter(self)
        w, h = self.width(), self.height()
        qp.fillRect(0, 0, w, h, QColor("#16181d"))
        font = QFont(); font.setPointSize(8); qp.setFont(font)
        qp.setPen(QPen(QColor("#c8ccd4"))); qp.drawText(6, 12, f"beat  {self.title}  {self.facts}")
        if not self.grid:
            qp.setPen(QPen(QColor("#5a606c"))); qp.drawText(6, 30, "ingest a song, then select a section row")
            qp.end(); return
        order = ("kick", "snare", "hat", "ohat", "shaker", "perc", "rim", "tom", "ride")
        rows = tuple(k for k in order if k in (self.grid or {})) or ("kick", "snare", "hat")
        left, top = 48, 18
        cw = max(8.0, (w - left - 8) / 16.0)
        rh = max(10.0, (h - top - 4) / len(rows))
        for r, name in enumerate(rows):
            qp.setPen(QPen(QColor("#9aa0ac"))); qp.drawText(6, int(top + r * rh + rh * 0.7), name)
            vals = (self.grid or {}).get(name) or [0.0] * 16
            hit_steps = {st for st, _ in ((self.hits or {}).get(name) or [])}
            for k in range(16):
                v = max(0.0, min(1.0, float(vals[k]) if k < len(vals) else 0.0))
                x = left + k * cw
                col = QColor(40 + int(190 * v), 60 + int(140 * v), 70)
                qp.setPen(QPen(QColor("#0e1013"))); qp.setBrush(QBrush(col))
                qp.drawRect(QRectF(x, top + r * rh + 1, cw - 1, rh - 2))
                if k in hit_steps:
                    qp.setPen(QPen(QColor("#ffffff"), 2)); qp.setBrush(Qt.BrushStyle.NoBrush)
                    qp.drawRect(QRectF(x + 1, top + r * rh + 2, cw - 3, rh - 4))
                if k % 4 == 0:
                    qp.setPen(QPen(QColor("#ffd166"))); qp.drawLine(int(x), int(top), int(x), int(top + len(rows) * rh))
        qp.end()


class AnalysisPage(QWidget):
    def __init__(self, console):
        super().__init__()
        self.console = console
        self.folder = None
        self.result = None          # ingest result
        self.script = None
        self.recon_feats = None
        self.report = None
        self.busy = False
        self.msg = ""
        self.progress = 0.0
        self._pending = None
        self.b_offset = None            # side B's display offset (None = the source's first downbeat, i.e. a render from bar 0)
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.dj_track = None                               # (id, title, artist) of the library track loaded
        self.dj_btn = QPushButton("♫ DJ track…"); self.dj_btn.clicked.connect(self.from_dj)
        self.dj_btn.setToolTip("Pick a track of the DJ library (stems rendered): its instrument reading becomes this tab's song")
        top.addWidget(self.dj_btn)
        self.track_lbl = QLabel("no track - pick one from the DJ library"); top.addWidget(self.track_lbl, 1)
        for text, fn, tip in (("Replay", self.replay, "The PROGRAMMATIC recreation: every note of the planner's reading played by the "
                                                       "song's own sampled instruments on its grid - nothing composed. Becomes side B."),
                              ("Recreate", self.recreate, "The generator's rendition of the script (its composer interprets the sections, "
                                                           "kit, banks and lines). Becomes side B."),
                              ("Fidelity", self.fidelity_report, "Measure the program: how much of the notes the pattern language holds, and how "
                                                                 "close its render is to the original stems (lib/dj/fidelity.py). Written on link; "
                                                                 "this recomputes it."),
                              ("Score", self.score, "Score side B against the source, per phrase and globally"),
                              ("Tune", self.tune, "Closed-loop tuning of the script against the source"),
                              ("Save", self.save, "Save the (edited) script"), ("Play", self.play, "Send the script to the running show")):
            b = QPushButton(text); b.setToolTip(tip); b.clicked.connect(fn); top.addWidget(b)
        lay.addLayout(top)
        fid = QHBoxLayout()
        fid.addWidget(QLabel("source material"))
        self.fidelity = QSlider(Qt.Orientation.Horizontal); self.fidelity.setRange(0, 100); self.fidelity.setValue(100); self.fidelity.setMaximumWidth(260)
        self.fidelity.setToolTip("0 = generator only; 50 = the song's drum + bass loops under generated melodic layers; 100 = every source loop, the generator fills the rest")
        fid.addWidget(self.fidelity)
        self.fid_lbl = QLabel("100%"); fid.addWidget(self.fid_lbl); fid.addStretch(1)
        self.fidelity.valueChanged.connect(lambda v: self.fid_lbl.setText(f"{v}%"))
        lay.addLayout(fid)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setTextVisible(False); self.bar.setMaximumHeight(6)
        lay.addWidget(self.bar)
        self.info = QLabel(""); lay.addWidget(self.info)
        # the two honest readouts (lib/dj/fidelity.py): notes as code, waveform vs the original
        self.fid_panel = QLabel(""); self.fid_panel.setTextFormat(Qt.TextFormat.RichText); self.fid_panel.setWordWrap(True)
        self.fid_panel.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # in a scroll box of fixed height: the per-voice table grows with the song and must not push
        # the compare view and the section table off the screen
        self.fid_scroll = QScrollArea(); self.fid_scroll.setWidget(self.fid_panel); self.fid_scroll.setWidgetResizable(True)
        self.fid_scroll.setMaximumHeight(150); self.fid_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.fid_scroll.setVisible(False); lay.addWidget(self.fid_scroll)
        # one line by default (it has to fit on a laptop screen with the compare view and the table); the
        # tables open on demand
        fl = QHBoxLayout()
        self.fid_line = QLabel(""); self.fid_line.setTextFormat(Qt.TextFormat.RichText); fl.addWidget(self.fid_line, 1)
        self.fid_more = QPushButton("details ▸"); self.fid_more.setCheckable(True); self.fid_more.setMaximumWidth(90)
        self.fid_more.setToolTip("Show the per-voice and per-stem fidelity tables")
        self.fid_more.toggled.connect(lambda on: (self.fid_scroll.setVisible(on and bool(self.fid_panel.text())),
                                                   self.fid_more.setText("details ▾" if on else "details ▸")))
        fl.addWidget(self.fid_more)
        self.fid_row = QWidget(); self.fid_row.setLayout(fl); self.fid_row.setVisible(False); lay.addWidget(self.fid_row)
        self.strip = ScoreStrip(); lay.addWidget(self.strip)
        # compare: original over recreation, with transport
        tr = QHBoxLayout()
        self.b_a = QPushButton("▶ A original"); self.b_a.clicked.connect(lambda: self.play_src("a")); tr.addWidget(self.b_a)
        self.b_b = QPushButton("▶ B recreation"); self.b_b.clicked.connect(lambda: self.play_src("b")); tr.addWidget(self.b_b)
        self.b_ab = QPushButton("A/B"); self.b_ab.clicked.connect(self.ab); tr.addWidget(self.b_ab)
        self.b_pause = QPushButton("⏸"); self.b_pause.clicked.connect(self.pause); tr.addWidget(self.b_pause)
        self.b_stop = QPushButton("■"); self.b_stop.clicked.connect(self.stop); tr.addWidget(self.b_stop)
        tr.addWidget(QLabel("zoom"))
        self.zoom = QSlider(Qt.Orientation.Horizontal); self.zoom.setRange(4, 600); self.zoom.setValue(60); self.zoom.setMaximumWidth(200)
        self.zoom.valueChanged.connect(self._zoom); tr.addWidget(self.zoom)
        self.follow = QCheckBox("follow"); self.follow.setChecked(True); self.follow.toggled.connect(lambda v: setattr(self.compare, "follow", v)); tr.addWidget(self.follow)
        self.pos_lbl = QLabel("0:00.00"); tr.addWidget(self.pos_lbl); tr.addStretch(1)
        lay.addLayout(tr)
        self.compare = CompareView(on_seek=self.seek); lay.addWidget(self.compare, 2)
        from tools.gen.console.player import WavPlayer
        self.player = WavPlayer()
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_play)
        QShortcut(QKeySequence("Tab"), self.compare, activated=self.ab)
        self.beat = BeatGrid(); lay.addWidget(self.beat)
        split = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, len(COLS)); self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setStretchLastSection(True); self.table.verticalHeader().setVisible(False)
        split.addWidget(self.table)
        self.cmds = QPlainTextEdit(); self.cmds.setReadOnly(True); self.cmds.setPlaceholderText("the command list that regenerates the song")
        split.addWidget(self.cmds)
        split.setSizes([700, 400])
        lay.addWidget(split, 1)
        self.table.itemSelectionChanged.connect(self._show_beat)

    def _beat_facts(self):
        a = (self.result or {}).get("analysis") or {}
        b = a.get("beat") or {}
        if not b:
            return ""
        return (f"{a.get('bpm', 0):.2f} bpm (conf {a.get('bpm_conf', 0):.2f})  beat {b.get('beat_s', 0):.3f} s  "
                f"{b.get('bars', 0)} bars  first downbeat {a.get('first_bar_s', 0):.2f} s (conf {a.get('downbeat_conf', 0):.2f})  "
                f"swing {b.get('swing', 0):.2f}  kind {b.get('drums_kind', '?')}")

    def _show_beat(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        i = rows[0].row() if rows else None
        if self.script and i is not None and i < len(self.script["sections"]):
            e = self.script["sections"][i]
            self.beat.set(e.get("drums_grid"), e.get("drums"), self._beat_facts(), f"section {i} {e['section']}")
        else:
            a = (self.result or {}).get("analysis") or {}
            pat = (a.get("beat") or {}).get("pattern") or {}
            self.beat.set(pat.get("grid"), {k: pat.get(k) for k in ("kick", "snare", "hat")} if pat else None, self._beat_facts(), "whole song")

    # -- helpers --------------------------------------------------------------
    def _run(self, label, fn):
        if self.busy:
            self.msg = "busy"
            return
        self.busy = True
        self.msg = label
        self.progress = 0.0

        def work():
            try:
                fn()
                self.msg = label + " done"
            except Exception as e:  # noqa: BLE001
                self.msg = f"{label} failed: {type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                self.busy = False
                self.progress = 1.0
        threading.Thread(target=work, name=f"analysis-{label}", daemon=True).start()

    def _folder(self):
        if self.folder:
            return self.folder
        if self.dj_track:
            from lib.dj import gen_link as GL
            return GL.folder_for(self.dj_track[1], self.dj_track[0])
        return os.path.join("logs", "analysis", "untitled")

    def _script_from_table(self):
        if self.script is None:
            return None
        from lib.gen import script as S
        sc = dict(self.script)
        rows = []
        for i in range(self.table.rowCount()):
            def cell(j):
                it = self.table.item(i, j)
                return it.text().strip() if it else ""
            try:
                e = {"section": cell(0) or "groove", "bars": int(float(cell(1) or 8))}
                for j, k in ((2, "energy"), (3, "density"), (4, "brightness"), (5, "swing")):
                    if cell(j):
                        e[k] = float(cell(j))
                if cell(6):
                    e["layers"] = [x.strip() for x in cell(6).replace("+", ",").split(",") if x.strip()]
                if cell(7):
                    e["chords"] = [S.parse_chord(x) for x in cell(7).replace(",", " ").split()]
                if cell(8):
                    e["lanes"] = json.loads(cell(8))
                old = self.script["sections"][i] if i < len(self.script["sections"]) else {}
                for keep in ("hook", "bass", "drums", "drums_grid", "drums_phrases", "dyn", "loops", "melody", "bass_line"):
                    if old.get(keep):
                        e[keep] = old[keep]
                rows.append(e)
            except Exception as ex:  # noqa: BLE001
                self.msg = f"row {i + 1}: {ex}"
                return None
        sc["sections"] = rows
        sc["fidelity"] = self.fidelity.value() / 100.0
        return S.normalize(sc)

    def _fill_table(self):
        from lib.gen import script as S
        sc = self.script or {"sections": []}
        if sc.get("fidelity") is not None and any(e.get("loops") for e in sc.get("sections", [])):
            self.fidelity.blockSignals(True); self.fidelity.setValue(int(round(100 * float(sc["fidelity"])))); self.fidelity.blockSignals(False)
            self.fid_lbl.setText(f"{self.fidelity.value()}%")
        self.table.setRowCount(len(sc["sections"]))
        for i, e in enumerate(sc["sections"]):
            vals = [e.get("section", ""), str(e.get("bars", "")), *(f"{e[k]:.2f}" if e.get(k) is not None else "" for k in ("energy", "density", "brightness", "swing")),
                    "+".join(e.get("layers") or []), " ".join(S.chord_str(x) for x in (e.get("chords") or [])),
                    json.dumps(e["lanes"]) if e.get("lanes") else ""]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j == 0:
                    it.setBackground(QColor(SECTION_COLOURS.get(v, "#555")))
                self.table.setItem(i, j, it)
        self.cmds.setPlainText("\n".join(f"bar {b:4d}  {a:10s} {json.dumps(v) if not isinstance(v, str) else v}" for b, a, v in S.to_actions(sc)) if sc.get("sections") else "")

    # -- playback + compare ---------------------------------------------------------
    def _load_compare(self, which, path):
        """Worker: spectrogram + audio for one side into the compare view and the player."""
        from lib.gen.analysis import spectro
        if not path or not os.path.exists(path):
            return
        d = spectro.prepare(path)
        self.player.load(which, d["stereo"])
        first_bar = float(((self.result or {}).get("analysis") or {}).get("first_bar_s", 0.0))
        bars = (self.result or {}).get("bars") or []
        sections = []
        if self.script:
            bar = 0
            for e in self.script["sections"]:
                t = bars[bar] if bar < len(bars) else first_bar + bar * (4 * 60.0 / float(self.script["bpm"]))
                sections.append((float(t), e["section"]))
                bar += e["bars"]
        side = {"rgb": d["rgb"], "fps": d["fps"], "seconds": d["seconds"]}
        offset = first_bar if self.b_offset is None else float(self.b_offset)    # a replay runs in the source's time
        self._compare_pending = (which, side, offset, bars, sections)

    def _source_path(self):
        if self.result and self.result.get("source"):
            return self.result["source"]
        return None

    def play_src(self, which):
        if which not in self.player.sources:
            self.msg = "nothing loaded for " + ("the original" if which == "a" else "the recreation")
            return
        if not self.player.available():
            self.msg = "no output device: " + self.player.error[:60]
            return
        self.player.play(which)

    def ab(self):
        other = "b" if self.player.current == "a" else "a"
        if other in self.player.sources:
            # keep the DISPLAY position: the recreation runs offset by the first downbeat
            t_disp = self._display_pos()
            self.player.switch(other)
            self.player.seek(t_disp - (self.compare.offset_s if other == "b" else 0.0))
            self.compare.which = other

    def pause(self):
        self.player.pause()

    def stop(self):
        self.player.stop()

    def toggle_play(self):
        if self.player.playing:
            self.player.pause()
        elif self.player.current in self.player.sources:
            self.play_src(self.player.current)

    def seek(self, display_s, row):
        which = row if row in self.player.sources else (self.player.current or "a")
        was = self.player.playing
        self.player.switch(which)
        self.player.seek(display_s - (self.compare.offset_s if which == "b" else 0.0))
        self.compare.which = which
        if was:
            self.player.play(which)
        self.compare.set_cursor(display_s, which)

    def _display_pos(self):
        return self.player.position() + (self.compare.offset_s if self.player.current == "b" else 0.0)

    def _zoom(self, v):
        self.compare.window_s = float(v)
        self.compare.update()

    # -- actions --------------------------------------------------------------
    def _show_fidelity(self, folder):
        """Show the folder's fidelity.json (nothing when there is none)."""
        try:
            from lib.dj import fidelity as F
            rep = F.load(folder) if folder else None
        except Exception:  # noqa: BLE001
            rep = None
        html = fidelity_html(rep) if rep else ""
        self.fid_panel.setText(html)
        self.fid_row.setVisible(bool(html))
        self.fid_scroll.setVisible(bool(html) and self.fid_more.isChecked())
        if rep:
            n, w = rep.get("notes") or {}, rep.get("waveform") or {}
            mix = w.get("mix") or {}
            parts = []
            if n:
                parts.append(f"notes as code: <b style='color:{_tone(8, 4, n['events_per_entry'])}'>{n['events_per_entry']:.1f}</b> events per pattern+op, "
                             f"{100 * n['pattern_share']:.0f}% in patterns, verbatim {n['verbatim_bars']}/{n['bars']} bars")
            ag = n.get("agreement") or {}
            if ag:
                parts.append("notes vs transcription: " + " ".join(f"{s} <b style='color:{_tone(0.8, 0.5, r['f1'])}'>{r['f1']:.2f}</b>" for s, r in ag.items()))
            if mix:
                stems = "  ".join(f"{s} {w[s]['spectral']:.0f}" for s in ("drums", "bass", "other", "vocals") if w.get(s))
                parts.append(f"waveform gap: mix <b style='color:{_tone(6, 12, mix['spectral'], False)}'>{mix['spectral']:.1f} dB</b> ({stems})")
            self.fid_line.setText("<span style='font-size:11px'>" + " &nbsp;·&nbsp; ".join(parts) + "</span>")

    def fidelity_report(self):
        """Recompute fidelity.json for the linked folder: the program rendered
        per stem against the library's stems + how programmatic it is."""
        folder = self.folder or self._folder()
        if not os.path.exists(os.path.join(folder, "program.json")):
            self.msg = "no program in this folder - link a DJ track first"
            return

        def work():
            from lib.dj import fidelity as F, resynth as RS, songprogram as SP
            with open(os.path.join(folder, "features.json"), encoding="utf-8") as fh:
                a = (json.load(fh).get("analysis") or {})
            with open(os.path.join(folder, "instruments.json"), encoding="utf-8") as fh:
                res = json.load(fh)
            self.msg = "fidelity: decoding the library's stems"
            stems = RS.load_stems_mono(a.get("dj_library"), a.get("dj_track_id"))
            if stems is None:
                raise FileNotFoundError("the track's stems are not on disk any more")
            prog = SP.load(folder)

            def prog_msg(what):
                self.msg = what
            rep = F.report(prog, stems, res, progress=prog_msg)
            F.save(rep, folder)
            self._pending = ("fidelity", folder)
        self._run("fidelity", work)

    def replay(self):
        """Side B = the reading played back exactly (lib/dj/gen_link.replay)."""
        folder = self.folder or self._folder()
        if not os.path.exists(os.path.join(folder, "instruments.json")):
            self.msg = "pick a DJ track first"
            return

        def work():
            from lib.dj import gen_link as GL
            from lib.gen.analysis import ingest as I

            def prog(what):
                self.msg = f"replay: {what}"
            audio, path = GL.replay(folder, progress=prog)
            a = (self.result or {}).get("analysis") or {}
            bpm = float(a.get("bpm") or (self.script or {}).get("bpm") or 120.0)
            # the replay runs in the SOURCE's time: its bars sit where the original's do
            self.recon_feats = I.features_on_grid(audio.mean(axis=1).astype(np.float32), bpm, float(a.get("first_bar_s", 0.0)))
            self.report = None
            self.b_offset = 0.0
            self._pending = "strip"
            self._load_compare("b", path)
        self._run("replay", work)

    def recreate(self):
        sc = self._script_from_table()
        if sc is None:
            return
        folder = self.folder or self._folder()

        def work():
            from lib.gen import script as S
            from lib.gen.analysis import ingest as I
            self.script = sc
            self.folder = folder
            S.save(sc, os.path.join(folder, "script.yaml"))

            def prog(x):
                self.progress = x
            audio, comp = S.render(sc, out_path=os.path.join(folder, "recreation.wav"), progress=prog, stems=True)
            trims = [e.get("trim_db") for e in (comp.script or {}).get("sections", [])]
            if any(t is not None for t in trims) and not any(e.get("trim_db") is not None for e in sc["sections"]):
                for e, t in zip(sc["sections"], trims):
                    e["trim_db"] = t                  # the level calibration stays with the script (no second pass next time)
                sc["master_db"] = float((comp.script or {}).get("master_db") or 0.0)
                S.save(sc, os.path.join(folder, "script.yaml"))
            self.recon_feats = I.features_on_grid(audio.mean(axis=1).astype(np.float32), sc["bpm"], 0.0)
            self.report = None
            self.b_offset = None                      # the generator starts at bar 0 = t 0
            self._pending = "strip"
            self._load_compare("b", os.path.join(folder, "recreation.wav"))
        self._run("recreate", work)

    def score(self):
        if self.result is None or self.recon_feats is None:
            self.msg = "ingest and recreate first"
            return

        def work():
            from lib.gen.analysis import score as SC
            a = self.result["analysis"]
            self.report = SC.compare(self.result["features"], self.recon_feats, bpm_orig=a["bpm"], bpm_recon=self.script["bpm"],
                                     key_orig=self.script["key"], key_recon=self.script["key"])
            stems = {n: os.path.join(self.folder or "", "stems", n + ".wav") for n in ("drums", "bass", "other", "vocals")}
            if self.folder and all(os.path.exists(p) for p in stems.values()) and os.path.exists(os.path.join(self.folder, "recreation.wav")):
                # the strict measure: the recreation's own stems against the source's, beat by beat; the level
                # differences become the script's per-stem mix trims (the next recreate applies them)
                self.msg = "score: separating the recreation into stems"
                from lib.gen import script as S
                for p in (os.path.join(self.folder, "recon_stems", n + ".wav") for n in stems):
                    if os.path.exists(p):
                        os.remove(p)
                sf_rep = SC.stem_fidelity(stems, os.path.join(self.folder, "recreation.wav"), float(a.get("first_bar_s", 0.0)), self.script["bpm"],
                                          os.path.join(self.folder, "recon_stems"), n_bars=len(self.result["features"]))
                self.report["stems"] = sf_rep
                mix = dict(self.script.get("mix_db") or {})
                for n, v in sf_rep.items():
                    if isinstance(v, dict) and v.get("level_db") is not None and abs(v["level_db"]) >= 0.5:
                        mix[n] = round(float(max(-18.0, min(18.0, mix.get(n, 0.0) - v["level_db"]))), 1)
                if mix != (self.script.get("mix_db") or {}):
                    self.script["mix_db"] = mix
                    S.save(self.script, os.path.join(self.folder, "script.yaml"))
                self.msg = ("stems: " + "  ".join(f"{n} {v['db']:.1f} dB" for n, v in sf_rep.items() if isinstance(v, dict))
                            + f"  (mean {sf_rep['mean_db']:.1f}; mix trims saved, recreate to apply)")
            if self.folder:
                with open(os.path.join(self.folder, "score.json"), "w", encoding="utf-8") as fh:
                    json.dump(self.report, fh, indent=1)
            self._pending = "strip"
        self._run("score", work)

    def tune(self):
        sc = self._script_from_table()
        if sc is None or self.result is None:
            self.msg = "ingest first"
            return
        folder = self.folder or self._folder()

        def work():
            from lib.gen import script as S
            from lib.gen.analysis import ingest as I, score as SC, tune as T

            def prog(x, what):
                self.progress = x; self.msg = f"tune: {what}"
            tuned, rep = T.tune(self.result, sc, rounds=1, progress=prog)
            self.script = tuned
            S.save(tuned, os.path.join(folder, "script_tuned.yaml"))
            audio, _ = S.render(tuned, out_path=os.path.join(folder, "recreation_tuned.wav"))
            self.recon_feats = I.features_on_grid(audio.mean(axis=1).astype(np.float32), tuned["bpm"], 0.0)
            a = self.result["analysis"]
            self.report = SC.compare(self.result["features"], self.recon_feats, bpm_orig=a["bpm"], bpm_recon=tuned["bpm"],
                                     key_orig=tuned["key"], key_recon=tuned["key"])
            self.report["tune"] = rep
            try:
                from lib.gen.feedback import PreferenceMemory
                PreferenceMemory(os.path.join("logs", "gen_prefs.json")).record_scores(tuned["style"], tuned, self.report)
            except Exception:  # noqa: BLE001
                pass
            self._pending = "table+strip"
            self._load_compare("b", os.path.join(folder, "recreation_tuned.wav"))
        self._run("tune", work)

    def save(self):
        sc = self._script_from_table()
        if sc is None:
            return
        from lib.gen import script as S
        folder = self.folder or self._folder()
        self.script = sc
        path = S.save(sc, os.path.join(folder, "script.yaml"))
        self.msg = f"saved {path}"
        self._fill_table()

    def play(self):
        sc = self._script_from_table()
        if sc is None:
            return
        from lib.gen import script as S
        folder = self.folder or self._folder()
        path = os.path.abspath(S.save(sc, os.path.join(folder, "script.yaml")))
        self.console.ctx.emit("script", path)
        self.msg = f"sent {os.path.basename(folder)}/script.yaml to the show"

    def from_dj(self):
        """Open a DJ library track's instrument reading as this tab's song."""
        from lib.dj import resolve_music_dir
        try:
            from lib.dj import gen_link as GL
            root = resolve_music_dir("")
            tracks = GL.tracks_with_stems(root)
        except Exception as e:  # noqa: BLE001
            self.msg = f"DJ library unavailable: {type(e).__name__}: {e}"
            return
        if not tracks:
            self.msg = f"no tracks with stems in {root} (render stems in the DJ planner first)"
            return
        dlg = DJTrackDialog(self, tracks)
        if dlg.exec() != dlg.DialogCode.Accepted or dlg.picked is None:
            return
        tid, title = dlg.picked
        self.dj_track = (tid, title, "")
        self.track_lbl.setText(f"{title}  (DJ track {tid})  - linking...")

        def work():
            from lib.dj import gen_link as GL

            def prog(what):
                self.msg = f"DJ link: {what}"
            folder = GL.link(root, tid, progress=prog)
            self._pending = ("open", folder)
        self._run(f"link '{title[:30]}'", work)

    def open_script(self, cand):
        """Load a linked folder's script.yaml (+ features.json, audio)."""
        if os.path.isdir(cand):
            cand = os.path.join(cand, "script.yaml")
        if not os.path.exists(cand):
            self.msg = f"no script at {cand}"
            return
        from lib.gen import script as S
        self.script = S.load(cand)
        self.folder = os.path.dirname(cand)
        feats = os.path.join(self.folder, "features.json")
        if os.path.exists(feats):
            with open(feats, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.result = {"features": saved["features"], "analysis": dict(saved["analysis"], sections=saved.get("sections", [])),
                           "bars": saved["bars"], "chords": saved["chords"], "script": self.script}
        self._fill_table()
        self.strip.set(orig=(self.result or {}).get("features", []), recon=[], report=None, script=self.script)
        self.msg = f"opened {cand}"
        if self.result is not None:
            self.result["source"] = saved.get("source") if os.path.exists(feats) else None
        src = self._source_path() or os.path.join(self.folder, "original.wav")
        rec = os.path.join(self.folder, "recreation_tuned.wav")
        if not os.path.exists(rec):
            rec = os.path.join(self.folder, "recreation.wav")
        self.b_offset = None
        if not os.path.exists(rec) and os.path.exists(os.path.join(self.folder, "replay.wav")):
            rec = os.path.join(self.folder, "replay.wav")     # the last replay, in the source's time
            self.b_offset = 0.0

        def work():
            self._load_compare("a", src)
            self._load_compare("b", rec)
        self._run("load audio", work)

    # -- refresh (console timer) ----------------------------------------------
    _compare_pending = None

    def refresh(self, state):
        self.bar.setValue(int(100 * self.progress))
        if self._compare_pending is not None:
            which, side, first_bar, bars, sections = self._compare_pending
            self._compare_pending = None
            self.compare.set_sources(a=side if which == "a" else None, b=side if which == "b" else None,
                                     offset_s=first_bar, bars=bars, sections=sections)
            self.compare.window_s = float(min(self.compare.total_s(), 60.0))
        if self.player.sources:
            t = self._display_pos()
            self.compare.set_cursor(t, self.player.current or "a")
            self.pos_lbl.setText(f"{int(t // 60)}:{t % 60:05.2f}  {'A' if self.player.current == 'a' else 'B'}{' ▶' if self.player.playing else ''}")
        if isinstance(self._pending, tuple) and self._pending[0] == "fidelity":
            folder = self._pending[1]
            self._pending = None
            self._show_fidelity(folder)
        if isinstance(self._pending, tuple) and self._pending[0] == "open":
            folder = self._pending[1]
            self._pending = None
            self.open_script(folder)
            self._show_fidelity(folder)
            a = (self.result or {}).get("analysis") or {}
            if self.dj_track:
                self.track_lbl.setText(f"{self.dj_track[1]}  (DJ track {self.dj_track[0]})  -  {folder}")
            if self.script:
                self.info.setText(f"{self.script.get('title')}: DJ track {a.get('dj_track_id', '?')}  {self.script['bpm']:.1f} bpm "
                                  f"{self.script['key']}  {len(self.script['sections'])} sections  "
                                  f"{len(a.get('instruments') or [])} instruments read by the planner")
            if a.get("instruments"):
                head = "DJ instrument reading:" + chr(10) + chr(10).join(a["instruments"])
                try:
                    from lib.dj import songprogram as SP
                    if os.path.exists(os.path.join(folder, "program.json")):
                        head = "SONG PROGRAM (what Replay renders):" + chr(10) + SP.describe(SP.load(folder)) + chr(10) * 2 + head
                except Exception as e:  # noqa: BLE001
                    head += chr(10) + f"(program unreadable: {type(e).__name__}: {e})"
                self.cmds.setPlainText(head + chr(10) * 2 + self.cmds.toPlainText())
        if self._pending == "table":
            self._pending = None
            self._fill_table()
            a = self.result["analysis"]
            mat = ""
            if self.script.get("kit") or self.script.get("vocals"):
                mat = f"   reused: kit {','.join(self.script.get('kit') or {})}  vocals {len(self.script.get('vocals') or [])}" + \
                      ("  hook" if any(e.get("hook") for e in self.script["sections"]) else "")
            self.info.setText(f"{self.script.get('title')}: {self.script['style']} {self.script['bpm']:.1f} bpm {self.script['key']} "
                              f"(key conf {a['key_conf']:.2f}, tempo conf {a['bpm_conf']:.2f}), {len(self.script['sections'])} sections, "
                              f"{a['duration_s']:.0f} s{mat}")
            self.strip.set(orig=self.result["features"], recon=[], report=None, script=self.script)
            self._show_beat()
        elif self._pending == "table+strip":
            self._pending = "strip"
            self._fill_table()
        if self._pending == "strip":
            self._pending = None
            self.strip.set(orig=(self.result or {}).get("features", []), recon=self.recon_feats or [], report=self.report, script=self.script)
            if self.report:
                from lib.gen.analysis import score as SC
                worst = ", ".join(f"bar {r['bar0']} ({r['score']:.0f})" for r in SC.worst(self.report))
                self.info.setText(f"global {self.report['global']:.1f}  local mean {self.report['mean_local']:.1f}  "
                                  f"structure {self.report['structure']:.1f}  tempo {self.report['tempo']:.0f}  key {self.report['key']:.0f}"
                                  f"   weakest: {worst}")
        if self.msg:
            self.console.notify(self.msg, quiet=True) if hasattr(self.console, "notify") else None
            self.msg = ""


class DJTrackDialog(QDialog):
    """Pick a DJ library track (those with stems); a check marks tracks
    that already carry an instrument reading."""

    def __init__(self, parent, tracks):
        super().__init__(parent)
        self.setWindowTitle("DJ track")
        self.resize(560, 480)
        self.tracks = tracks
        self.picked = None
        lay = QVBoxLayout(self)
        self.filter = QLineEdit(); self.filter.setPlaceholderText("filter by title / artist")
        self.filter.textChanged.connect(self._fill)
        lay.addWidget(self.filter)
        self.list = QListWidget(); lay.addWidget(self.list, 1)
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        lay.addWidget(QLabel("✓ = instrument reading on disk (others run the pass first, ~16 s per minute of audio)"))
        row = QHBoxLayout()
        ok = QPushButton("Open"); ok.clicked.connect(self.accept); row.addWidget(ok)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject); row.addWidget(cancel)
        lay.addLayout(row)
        self._fill()

    def _fill(self):
        q = self.filter.text().strip().lower()
        self.list.clear()
        for tid, title, artist, has in self.tracks:
            if q and q not in title.lower() and q not in artist.lower():
                continue
            it = QListWidgetItem(f"{'✓' if has else ' '}  {title} - {artist}")
            it.setData(Qt.ItemDataRole.UserRole, (tid, title))
            self.list.addItem(it)

    def accept(self):
        it = self.list.currentItem()
        if it is not None:
            self.picked = it.data(Qt.ItemDataRole.UserRole)
        super().accept()


def register(console):
    page = AnalysisPage(console)
    console.add_tab("Analysis", page)
