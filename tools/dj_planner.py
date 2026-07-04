"""DJ Set Planner - native PyQt6 desktop app.

Four tabs:
  LIBRARY   scan/rescan the music folder (starts empty, fills as analysis
            runs), search, USER TAGS, auto-classification tags, structure
            strips, multi-select -> add to set, open a song in Analysis.
  ANALYSIS  one song deep: zoomable waveform down to beat level with the
            beat grid, sections, vocal regions and cue flags; play/scrub;
            set IN/OUT/INTEREST points (user cues override the analyzer's
            mix points in every plan).
  SET       build the set: anchors + suggestions, PLAN MODE (theme + target
            length -> suggested set), order optimization, auto-fill between
            timed anchors, compiled plan with warnings, seam audition.
  MIX       DJ-software-style timeline of the compiled set: overlapping
            blocks, beat ticks when zoomed, real gain/EQ automation
            envelopes at every seam; play the whole set live and jump
            between tracks/seams.

Usage: python tools/dj_planner.py [--dir <music_dir>]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtCore import (Qt, QAbstractTableModel, QModelIndex, QRect,
                          QSortFilterProxyModel, QThread, QTimer, QProcess,
                          pyqtSignal)
from PyQt6.QtGui import QColor, QPalette, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QLineEdit, QTableView, QListWidget, QListWidgetItem, QPushButton,
    QComboBox, QStyledItemDelegate, QSplitter, QMessageBox, QInputDialog,
    QAbstractItemView, QDoubleSpinBox, QSpinBox, QStyle, QTabWidget,
    QPlainTextEdit)

from lib.dj import resolve_music_dir
from lib.dj.db import LibraryDB
from lib.dj.brain import Brain, load_library
from lib.dj.themes import BUILTIN_THEMES, get_theme
from lib.dj import setlist as SL
from tools.djplanner.waveform import WaveformView
from tools.djplanner.mixview import MixTimeline
from tools.djplanner.player import TrackPlayer, SetPreview

RATE = 44100
SECTION_COLORS = {
    "intro": QColor(70, 90, 120), "outro": QColor(70, 90, 120),
    "steady": QColor(60, 120, 90), "drop": QColor(190, 80, 60),
    "build": QColor(190, 150, 60), "breakdown": QColor(90, 70, 130),
}
COLS = ["title", "artist", "bpm", "key", "dur", "tags", "structure"]


# ==========================================================================
# Library tab
# ==========================================================================

class LibraryModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.tracks = []

    def set_tracks(self, tracks):
        self.beginResetModel()
        self.tracks = tracks
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.tracks)

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def headerData(self, i, orient, role):
        if role == Qt.ItemDataRole.DisplayRole \
                and orient == Qt.Orientation.Horizontal:
            return COLS[i]
        return None

    def data(self, index, role):
        t = self.tracks[index.row()]
        c = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if c == 0:
                return t.title
            if c == 1:
                return t.artist
            if c == 2:
                return f"{t.bpm:.1f}"
            if c == 3:
                return t.camelot
            if c == 4:
                return f"{int(t.duration_s // 60)}:{int(t.duration_s % 60):02d}"
            if c == 5:
                return " ".join(t.all_tags)
        if role == Qt.ItemDataRole.UserRole:
            return t
        return None


class StripDelegate(QStyledItemDelegate):
    def paint(self, p, option, index):
        t = index.model().data(index, Qt.ItemDataRole.UserRole)
        if t is None:
            return
        if option.state & QStyle.StateFlag.State_Selected:
            p.fillRect(option.rect, option.palette.highlight())
        r = option.rect.adjusted(2, 4, -2, -4)
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        dur = max(t.duration_s, 1.0)
        for s in t.sections:
            x0 = r.x() + r.width() * s["start_s"] / dur
            x1 = r.x() + r.width() * s["end_s"] / dur
            p.fillRect(QRect(int(x0), r.y(), max(int(x1 - x0), 1),
                             r.height()),
                       SECTION_COLORS.get(s["kind"], QColor(80, 80, 80)))
        curve = t.row.get("energy_curve") or []
        if curve:
            p.setPen(QColor(240, 240, 240, 200))
            n = len(curve)
            prev = None
            for i in range(0, n, max(n // 120, 1)):
                x = r.x() + r.width() * i / max(n - 1, 1)
                y = r.y() + r.height() * (1.0 - min(float(curve[i]), 1.0))
                if prev:
                    p.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
                prev = (x, y)
        p.restore()


class LibraryTab(QWidget):
    openAnalysis = pyqtSignal(object)          # TrackInfo
    addTracks = pyqtSignal(list)               # [TrackInfo]

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self._proc = None
        v = QVBoxLayout(self)

        top = QHBoxLayout()
        self.scan_btn = QPushButton("Scan library")
        self.scan_btn.clicked.connect(self.run_scan)
        top.addWidget(self.scan_btn)
        self.scan_lbl = QLabel("")
        top.addWidget(self.scan_lbl, 1)
        self.search = QLineEdit(placeholderText="search title / artist / tag...")
        self.search.textChanged.connect(
            lambda s: self.proxy.setFilterFixedString(s))
        top.addWidget(self.search, 2)
        v.addLayout(top)

        self.model = LibraryModel()
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setItemDelegateForColumn(6, StripDelegate())
        for i, w in enumerate((230, 120, 52, 44, 50, 150, 260)):
            self.table.setColumnWidth(i, w)
        self.table.doubleClicked.connect(
            lambda _: self._open_analysis())
        v.addWidget(self.table)

        bot = QHBoxLayout()
        b1 = QPushButton("Add selected to set →")
        b1.clicked.connect(self._add_selected)
        bot.addWidget(b1)
        b2 = QPushButton("Open in Analysis")
        b2.clicked.connect(self._open_analysis)
        bot.addWidget(b2)
        bot.addSpacing(24)
        self.tag_edit = QLineEdit(placeholderText="tag...")
        self.tag_edit.setMaximumWidth(140)
        bot.addWidget(self.tag_edit)
        bt = QPushButton("Add tag")
        bt.clicked.connect(lambda: self._tag(True))
        bot.addWidget(bt)
        br = QPushButton("Remove tag")
        br.clicked.connect(lambda: self._tag(False))
        bot.addWidget(br)
        bot.addStretch(1)
        self.count_lbl = QLabel("")
        bot.addWidget(self.count_lbl)
        v.addLayout(bot)

        self._scan_timer = QTimer(self)
        self._scan_timer.timeout.connect(self._poll_scan)
        self._last_done = -1

    def selected_tracks(self):
        rows = {i.row() for i in self.table.selectionModel().selectedRows()}
        return [self.proxy.data(self.proxy.index(r, 0),
                                Qt.ItemDataRole.UserRole)
                for r in sorted(rows)]

    def _add_selected(self):
        ts = self.selected_tracks()
        if ts:
            self.addTracks.emit(ts)

    def _open_analysis(self):
        ts = self.selected_tracks()
        if ts:
            self.openAnalysis.emit(ts[0])

    def _tag(self, add):
        tag = self.tag_edit.text().strip().lower()
        ts = self.selected_tracks()
        if not tag or not ts:
            return
        for t in ts:
            (self.planner.db.add_tag if add
             else self.planner.db.remove_tag)(t.id, tag)
        self.planner.reload_library()

    def refresh(self):
        self.model.set_tracks(self.planner.library)
        self.count_lbl.setText(f"{len(self.planner.library)} tracks")

    # -- scanning ------------------------------------------------------------
    def run_scan(self):
        if self._proc is not None:
            return
        script = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "tools", "dj_scan.py")
        self._proc = QProcess(self)
        self._proc.finished.connect(self._scan_done)
        self._proc.start(sys.executable,
                         [script, "--dir", self.planner.music_dir])
        self.scan_btn.setEnabled(False)
        self.scan_lbl.setText("scanning...")
        self._last_done = -1
        self._scan_timer.start(1500)

    def _poll_scan(self):
        from lib.dj.scan import PROGRESS_PATH
        try:
            with open(PROGRESS_PATH) as f:
                pr = json.load(f)
        except Exception:
            return
        self.scan_lbl.setText(
            f"scanning {pr.get('done', 0)}/{pr.get('total', '?')}  "
            f"{pr.get('current', '')[:40]}")
        # New analyses landed -> live-populate the table.
        if pr.get("done", 0) != self._last_done:
            self._last_done = pr.get("done", 0)
            self.planner.reload_library()

    def _scan_done(self, *a):
        self._proc = None
        self._scan_timer.stop()
        self.scan_btn.setEnabled(True)
        self.scan_lbl.setText("scan complete")
        self.planner.reload_library()


# ==========================================================================
# Analysis tab
# ==========================================================================

class DecodeWorker(QThread):
    done = pyqtSignal(object, object)          # track, stereo samples

    def __init__(self, db, track):
        super().__init__()
        self.db, self.track = db, track

    def run(self):
        try:
            from lib.dj.features import decode_file_stereo
            self.done.emit(self.track,
                           decode_file_stereo(self.db.abs(self.track.path)))
        except Exception as e:
            self.done.emit(self.track, f"{type(e).__name__}: {e}")


class AnalysisTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.track = None
        self.player = TrackPlayer()
        self.selected_cue = None
        self._decoder = None
        v = QVBoxLayout(self)

        top = QHBoxLayout()
        self.track_combo = QComboBox()
        self.track_combo.setMinimumWidth(360)
        self.track_combo.currentIndexChanged.connect(self._combo_pick)
        top.addWidget(self.track_combo, 1)
        self.info_lbl = QLabel("")
        top.addWidget(self.info_lbl, 2)
        v.addLayout(top)

        self.wave = WaveformView()
        self.wave.seekRequested.connect(self._seek)
        self.wave.cueClicked.connect(self._cue_clicked)
        v.addWidget(self.wave, 1)

        tr = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        tr.addWidget(self.play_btn)
        self.time_lbl = QLabel("0:00")
        tr.addWidget(self.time_lbl)
        tr.addSpacing(20)
        for kind, label in (("in", "Mark IN"), ("out", "Mark OUT"),
                            ("interest", "Mark INTEREST")):
            b = QPushButton(label)
            b.clicked.connect(lambda _, k=kind: self._add_cue(k))
            tr.addWidget(b)
        self.del_cue_btn = QPushButton("Delete cue")
        self.del_cue_btn.clicked.connect(self._del_cue)
        self.del_cue_btn.setEnabled(False)
        tr.addWidget(self.del_cue_btn)
        tr.addStretch(1)
        self.cue_lbl = QLabel("click a cue flag to select it")
        tr.addWidget(self.cue_lbl)
        v.addLayout(tr)

        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setMaximumHeight(130)
        v.addWidget(self.detail)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(80)

    def refresh_tracklist(self):
        cur = self.track.id if self.track else None
        self.track_combo.blockSignals(True)
        self.track_combo.clear()
        for t in self.planner.library:
            self.track_combo.addItem(f"{t.title} - {t.artist}", t.id)
        if cur is not None:
            i = self.track_combo.findData(cur)
            if i >= 0:
                self.track_combo.setCurrentIndex(i)
        self.track_combo.blockSignals(False)

    def _combo_pick(self, i):
        tid = self.track_combo.itemData(i)
        t = next((x for x in self.planner.library if x.id == tid), None)
        if t is not None and (self.track is None or t.id != self.track.id):
            self.open_track(t)

    def open_track(self, track):
        self.track = track
        self.player.pause()
        self.info_lbl.setText("decoding...")
        i = self.track_combo.findData(track.id)
        if i >= 0:
            self.track_combo.blockSignals(True)
            self.track_combo.setCurrentIndex(i)
            self.track_combo.blockSignals(False)
        self._decoder = DecodeWorker(self.planner.db, track)
        self._decoder.done.connect(self._decoded)
        self._decoder.start()

    def _decoded(self, track, samples):
        if self.track is None or track.id != self.track.id:
            return
        if isinstance(samples, str):
            self.info_lbl.setText("decode failed: " + samples)
            return
        self.player.load(samples)
        mono = samples.mean(axis=1)
        self.wave.set_track(track, mono,
                            self.planner.db.cues_for(track.id))
        lc = track.row.get("live_check") or {}
        axes = track.axes
        self.info_lbl.setText(
            f"{track.bpm:.1f} bpm (conf {track.bpm_conf:.2f})  "
            f"{track.camelot}  {int(track.duration_s // 60)}:"
            f"{int(track.duration_s % 60):02d}   tags: "
            f"{' '.join(track.all_tags) or '-'}")
        secs = "\n".join(
            f"  {s['kind']:9s} {s['start_s']:6.1f}-{s['end_s']:6.1f}s  "
            f"energy {s['energy']:.2f}  busy {s['busyness']:.2f}  "
            f"vocal {s['vocalness']:.2f}  rep {s['repetitiveness']:.2f}"
            for s in track.sections)
        self.detail.setPlainText(
            f"axes: {json.dumps(axes)}   live: bpm {lc.get('live_bpm')} "
            f"agrees={lc.get('agrees')} conf {lc.get('mean_conf')}\n"
            f"sections (v{track.row.get('analysis_version')}):\n{secs}")

    # -- transport ---------------------------------------------------------------
    def _toggle_play(self):
        if self.player.playing:
            self.player.pause()
            self.play_btn.setText("▶ Play")
        else:
            self.player.play()
            self.play_btn.setText("⏸ Pause")

    def _seek(self, t):
        self.player.seek(t)

    def _tick(self):
        if self.player.samples is not None:
            t = self.player.time_s()
            self.time_lbl.setText(f"{int(t // 60)}:{int(t % 60):02d}")
            if self.player.playing:
                self.wave.set_playhead(t)

    # -- cues -------------------------------------------------------------------
    def _add_cue(self, kind):
        if self.track is None:
            return
        t = self.wave.playhead
        # Snap to the nearest downbeat - cues drive bar-aligned mixing.
        t = self.track.nearest_downbeat(t)
        self.planner.db.add_cue(self.track.id, kind, t, label=kind)
        self._cues_changed()

    def _cue_clicked(self, cue):
        self.selected_cue = cue
        editable = cue.get("source") == "user"
        self.del_cue_btn.setEnabled(editable)
        self.cue_lbl.setText(
            f"selected: {cue['kind']} @ {cue['time_s']:.1f}s "
            f"({cue['source']}{'' if editable else ' - not deletable'})")

    def _del_cue(self):
        if self.selected_cue and self.selected_cue.get("id"):
            self.planner.db.remove_cue(self.selected_cue["id"])
            self.selected_cue = None
            self.del_cue_btn.setEnabled(False)
            self._cues_changed()

    def _cues_changed(self):
        self.wave.set_cues(self.planner.db.cues_for(self.track.id))
        self.planner.reload_library(keep_analysis=True)

    def close(self):
        self.player.close()


# ==========================================================================
# Set tab
# ==========================================================================

class CompileWorker(QThread):
    done = pyqtSignal(object)

    def __init__(self, library, entries, theme):
        super().__init__()
        self.library, self.entries, self.theme = library, entries, theme

    def run(self):
        try:
            self.done.emit(SL.compile_plan(self.library, self.entries,
                                           self.theme))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done.emit({"error": f"{type(e).__name__}: {e}"})


class AuditionWorker(QThread):
    done = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, db, a, b, plan):
        super().__init__()
        self.db, self.a, self.b, self.plan = db, a, b, plan

    def run(self):
        try:
            from lib.audio_engine import AudioEngine
            from lib.dj.submix import DJSubmix
            from lib.dj.features import decode_file_stereo
            a, b, plan = self.a, self.b, dict(self.plan)
            self.status.emit("decoding...")
            sa = decode_file_stereo(self.db.abs(a.path))
            sb = decode_file_stereo(self.db.abs(b.path))
            engine = AudioEngine()
            sub = DJSubmix()
            engine.attach_track("dj", sub)
            pre = 12.0
            cue_a = a.nearest_downbeat(max(0.0, plan["out_s"] - pre))
            sub.post_many([
                {"cmd": "load", "deck": "a", "samples": sa, "grid": a.grid,
                 "gain_db": a.gain_db, "cue_s": cue_a},
                {"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01},
                {"cmd": "start", "deck": "a"},
                {"cmd": "load", "deck": "b", "samples": sb, "grid": b.grid,
                 "gain_db": b.gain_db, "cue_s": plan["in_s"]},
            ])
            gen = engine._mixer()
            next(gen)
            gen.send(256)
            brain = Brain([], get_theme("groove"))
            events, swap_at, blend_at = brain.build_events(
                plan, sub.telemetry, "a", "b", a, b)
            sub.post_many(events)
            self.status.emit("rendering seam...")
            total = pre + (swap_at - blend_at) / RATE + 25.0
            out = [np.frombuffer(gen.send(4410), dtype=np.float32)
                   .reshape(-1, 2) for _ in range(int(total * RATE) // 4410)]
            self.done.emit(np.concatenate(out, axis=0))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done.emit(f"{type(e).__name__}: {e}")


class SetTab(QWidget):
    planCompiled = pyqtSignal(object)          # compiled dict

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.entries = []
        self.compiled = None
        self.setlist_id = None
        self._worker = None
        self._aud = None
        self.seam_player = TrackPlayer()

        h = QHBoxLayout(self)
        left = QVBoxLayout()
        srow = QHBoxLayout()
        self.set_combo = QComboBox()
        self.set_combo.currentTextChanged.connect(self._load_set)
        srow.addWidget(self.set_combo, 1)
        for label, fn in (("New", self.new_set), ("Save", self.save_set),
                          ("Delete", self.delete_set)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            srow.addWidget(b)
        left.addLayout(srow)

        # PLAN MODE inputs: how should the night go?
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(BUILTIN_THEMES))
        self.theme_combo.setCurrentText("groove")
        self.theme_combo.currentTextChanged.connect(lambda _: self.recompile())
        prow.addWidget(self.theme_combo)
        prow.addWidget(QLabel("Target:"))
        self.minutes_spin = QSpinBox(minimum=10, maximum=600, value=60)
        self.minutes_spin.setSuffix(" min")
        prow.addWidget(self.minutes_spin)
        left.addLayout(prow)
        prow2 = QHBoxLayout()
        sg = QPushButton("Suggest set (plan mode)")
        sg.clicked.connect(self.suggest)
        prow2.addWidget(sg)
        op = QPushButton("Optimize order")
        op.clicked.connect(self.optimize)
        prow2.addWidget(op)
        af = QPushButton("Auto-fill anchors")
        af.clicked.connect(self.autofill)
        prow2.addWidget(af)
        left.addLayout(prow2)

        left.addWidget(QLabel("Set (drag to reorder, double-click = anchor,"
                              " Del = remove)"))
        self.set_list = QListWidget()
        self.set_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove)
        self.set_list.model().rowsMoved.connect(self._reordered)
        self.set_list.itemDoubleClicked.connect(self._toggle_anchor)
        rm = QAction("remove", self)
        rm.setShortcut("Delete")
        rm.triggered.connect(self._remove_entry)
        self.set_list.addAction(rm)
        left.addWidget(self.set_list, 1)
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Anchor @"))
        self.anchor_min = QDoubleSpinBox(maximum=600.0, decimals=0)
        self.anchor_min.setSuffix(" min")
        arow.addWidget(self.anchor_min)
        pb = QPushButton("Pin selected anchor")
        pb.clicked.connect(self._pin_anchor)
        arow.addWidget(pb)
        arow.addStretch(1)
        left.addLayout(arow)
        h.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(QLabel("Compiled plan (select a ↳ seam, audition):"))
        self.plan_list = QListWidget()
        right.addWidget(self.plan_list, 1)
        brow = QHBoxLayout()
        self.audition_btn = QPushButton("▶ Audition seam")
        self.audition_btn.clicked.connect(self.audition)
        brow.addWidget(self.audition_btn)
        st = QPushButton("■ Stop")
        st.clicked.connect(self.seam_player.close)
        brow.addWidget(st)
        brow.addStretch(1)
        right.addLayout(brow)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        h.addLayout(right, 1)

    # -- entries ------------------------------------------------------------
    def theme(self):
        return get_theme(self.theme_combo.currentText())

    def add_tracks(self, tracks):
        for t in tracks:
            self.entries.append({"track_id": t.id, "pin_type": "suggestion",
                                 "target_offset_min": None,
                                 "style_override": None})
        self._rebuild()
        self.recompile()

    def _entry_label(self, e):
        t = next((x for x in self.planner.library
                  if x.id == e["track_id"]), None)
        name = t.title if t else f"track {e['track_id']}"
        tag = "⚓ " if e["pin_type"] == "anchor" else "• "
        pin = (f"  @{e['target_offset_min']:.0f}min"
               if e.get("target_offset_min") else "")
        bpm = f"  ({t.bpm:.0f} {t.camelot})" if t else ""
        return tag + name + bpm + pin

    def _rebuild(self):
        self.set_list.clear()
        for e in self.entries:
            QListWidgetItem(self._entry_label(e), self.set_list)

    def _reordered(self, *a):
        order = [self.set_list.item(i).text()
                 for i in range(self.set_list.count())]
        labels = {self._entry_label(e): e for e in self.entries}
        self.entries = [labels[t] for t in order if t in labels]
        self.recompile()

    def _toggle_anchor(self, item):
        e = self.entries[self.set_list.row(item)]
        e["pin_type"] = ("anchor" if e["pin_type"] == "suggestion"
                         else "suggestion")
        if e["pin_type"] == "suggestion":
            e["target_offset_min"] = None
        self._rebuild()
        self.recompile()

    def _pin_anchor(self):
        i = self.set_list.currentRow()
        if i < 0:
            return
        self.entries[i]["pin_type"] = "anchor"
        self.entries[i]["target_offset_min"] = self.anchor_min.value() or None
        self._rebuild()
        self.recompile()

    def _remove_entry(self):
        i = self.set_list.currentRow()
        if 0 <= i < len(self.entries):
            self.entries.pop(i)
            self._rebuild()
            self.recompile()

    # -- plan mode ---------------------------------------------------------------
    def suggest(self):
        if self.entries and QMessageBox.question(
                self, "Suggest set", "Replace the current set?") \
                != QMessageBox.StandardButton.Yes:
            return
        self.entries = SL.suggest_set(self.planner.library, self.theme(),
                                      float(self.minutes_spin.value()))
        self._rebuild()
        self.recompile()

    def optimize(self):
        self.entries = SL.optimize_order(self.planner.library, self.entries,
                                         self.theme())
        self._rebuild()
        self.recompile()

    def autofill(self):
        self.entries = SL.autofill(self.planner.library, self.entries,
                                   self.theme())
        self._rebuild()
        self.recompile()

    # -- persistence ------------------------------------------------------------
    def refresh_setlists(self):
        self.set_combo.blockSignals(True)
        cur = self.set_combo.currentText()
        self.set_combo.clear()
        self.set_combo.addItem("(unsaved set)")
        for s in SL.list_setlists(self.planner.db):
            self.set_combo.addItem(s["name"])
        if cur:
            self.set_combo.setCurrentText(cur)
        self.set_combo.blockSignals(False)

    def _load_set(self, name):
        if not name or name.startswith("("):
            return
        sl = SL.get_setlist(self.planner.db, name=name)
        if sl is None:
            return
        self.setlist_id = sl["id"]
        self.entries = [{k: e.get(k) for k in
                         ("track_id", "pin_type", "target_offset_min",
                          "style_override")} for e in sl["entries"]]
        if sl.get("theme") in BUILTIN_THEMES:
            self.theme_combo.setCurrentText(sl["theme"])
        self._rebuild()
        self.recompile()

    def new_set(self):
        name, ok = QInputDialog.getText(self, "New setlist", "Name:")
        if not (ok and name):
            return
        try:
            self.setlist_id = SL.create_setlist(
                self.planner.db, name, theme=self.theme_combo.currentText())
        except Exception as e:
            QMessageBox.warning(self, "New setlist", str(e))
            return
        self.entries = []
        self._rebuild()
        self.refresh_setlists()
        self.set_combo.setCurrentText(name)

    def save_set(self):
        if self.setlist_id is None:
            self.new_set()
            if self.setlist_id is None:
                return
        SL.save_entries(self.planner.db, self.setlist_id, self.entries)
        self.planner.db.conn.execute(
            "UPDATE setlists SET theme = ? WHERE id = ?",
            (self.theme_combo.currentText(), self.setlist_id))
        self.planner.db.conn.commit()
        self.status.setText("saved.")

    def delete_set(self):
        if self.setlist_id is not None and QMessageBox.question(
                self, "Delete", "Delete this setlist?") \
                == QMessageBox.StandardButton.Yes:
            SL.delete_setlist(self.planner.db, self.setlist_id)
            self.setlist_id = None
            self.entries = []
            self._rebuild()
            self.refresh_setlists()
            self.recompile()

    # -- compile + audition --------------------------------------------------------
    def recompile(self):
        if self._worker is not None and self._worker.isRunning():
            self._pending = True
            return
        self._pending = False
        self._worker = CompileWorker(self.planner.library,
                                     list(self.entries), self.theme())
        self._worker.done.connect(self._compiled)
        self._worker.start()

    def _compiled(self, result):
        if getattr(self, "_pending", False):
            self.recompile()
        if "error" in result:
            self.status.setText("compile error: " + result["error"])
            return
        self.compiled = result
        self.plan_list.clear()
        for i, s in enumerate(result["slots"]):
            t = s["track"]
            mins = s["start_offset_s"] / 60.0
            QListWidgetItem(
                f"{int(mins):3d}:{int(s['start_offset_s'] % 60):02d}  "
                f"{t.title}  ({t.bpm:.0f} {t.camelot}, "
                f"{s['play_s'] / 60.0:.1f} min)", self.plan_list)
            p = s["transition"]
            if p:
                nxt = result["slots"][i + 1]["track"]
                warn = ("   ⚠ " + "; ".join(s["warnings"])
                        if s["warnings"] else "")
                item = QListWidgetItem(
                    f"      ↳ {p['style']} @ {p['out_s']:.0f}s "
                    f"(rate {p['rate']:.3f}, seam {p['pair_score']:.2f}) "
                    f"→ {nxt.title}{warn}", self.plan_list)
                item.setData(Qt.ItemDataRole.UserRole, i)
                if s["warnings"]:
                    item.setForeground(QColor(255, 170, 100))
        self.status.setText(
            f"set ~{result['total_s'] / 3600.0:.1f} h, "
            f"{len(result['slots'])} tracks, "
            f"{len(result['warnings'])} warnings")
        self.planCompiled.emit(result)

    def select_seam(self, index):
        for r in range(self.plan_list.count()):
            it = self.plan_list.item(r)
            if it.data(Qt.ItemDataRole.UserRole) == index:
                self.plan_list.setCurrentRow(r)
                return

    def audition(self):
        item = self.plan_list.currentItem()
        i = item.data(Qt.ItemDataRole.UserRole) if item else None
        if i is None or self.compiled is None:
            self.status.setText("select a ↳ seam row first")
            return
        slots = self.compiled["slots"]
        self.audition_btn.setEnabled(False)
        self._aud = AuditionWorker(self.planner.db, slots[i]["track"],
                                   slots[i + 1]["track"],
                                   slots[i]["transition"])
        self._aud.status.connect(self.status.setText)
        self._aud.done.connect(self._audition_done)
        self._aud.start()

    def _audition_done(self, result):
        self.audition_btn.setEnabled(True)
        if isinstance(result, str):
            self.status.setText("audition failed: " + result)
            return
        self.status.setText(f"playing seam ({len(result) / RATE:.0f}s)")
        self.seam_player.load(result)
        self.seam_player.play()


# ==========================================================================
# Mix tab
# ==========================================================================

class MixTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.preview = None
        v = QVBoxLayout(self)
        self.timeline = MixTimeline()
        self.timeline.seamSelected.connect(self._seam_picked)
        v.addWidget(self.timeline, 1)

        row = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play set")
        self.play_btn.clicked.connect(self.play_set)
        row.addWidget(self.play_btn)
        self.pause_btn = QPushButton("⏸")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._pause)
        row.addWidget(self.pause_btn)
        stop = QPushButton("■ Stop")
        stop.clicked.connect(self.stop)
        row.addWidget(stop)
        row.addSpacing(20)
        prev = QPushButton("⏮ Prev track")
        prev.clicked.connect(lambda: self._jump(-1))
        row.addWidget(prev)
        nxt = QPushButton("⏭ Next track")
        nxt.clicked.connect(lambda: self._jump(+1))
        row.addWidget(nxt)
        seam = QPushButton("→ Next seam")
        seam.clicked.connect(self._next_seam)
        row.addWidget(seam)
        row.addStretch(1)
        self.status = QLabel("")
        row.addWidget(self.status)
        v.addLayout(row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(200)

    def set_plan(self, compiled):
        brain = Brain(self.planner.library,
                      self.planner.set_tab.theme())
        self.timeline.set_plan(compiled, brain)

    def play_set(self, from_slot=0):
        st = self.planner.set_tab
        if not st.entries:
            self.status.setText("build a set first")
            return
        self.stop()
        self.preview = SetPreview(self.planner.music_dir, st.entries,
                                  theme_name=st.theme_combo.currentText())
        self.preview.start(from_slot=from_slot)
        self.pause_btn.setChecked(False)
        self.status.setText("playing set...")

    def stop(self):
        if self.preview is not None:
            self.preview.close()
            self.preview = None
        self.timeline.set_playhead(None)
        self.status.setText("stopped")

    def _pause(self, on):
        if self.preview is not None:
            self.preview.pause(on)

    def _jump(self, d):
        if self.preview is not None:
            self.preview.jump_to_slot(self.preview.slot_index + d)

    def _next_seam(self):
        if self.preview is not None:
            self.preview.next_seam()
            self.status.setText("asking for a musical exit...")

    def _seam_picked(self, index):
        self.planner.set_tab.select_seam(index)

    def _tick(self):
        if self.preview is None:
            return
        s = self.preview.status()
        if not s.get("active"):
            if s.get("error"):
                self.status.setText("preview error: " + s["error"])
            return
        cur = s.get("current") or {}
        compiled = self.planner.set_tab.compiled
        slot = min(s.get("slot_index", 0),
                   len(compiled["slots"]) - 1) if compiled else 0
        if compiled and cur.get("pos_s") is not None:
            sl = compiled["slots"][slot]
            in_s = (sl["track"].mix_ins[0]["time_s"]
                    if sl["track"].mix_ins else 0.0)
            self.timeline.set_playhead(
                sl["start_offset_s"] + max(cur["pos_s"] - in_s, 0.0))
        nxt = s.get("next") or {}
        self.status.setText(
            f"[{s.get('state', '?')}] {cur.get('title', '-')}"
            + (f" → {nxt['title']}" if nxt else ""))

    def close(self):
        self.stop()


# ==========================================================================
# Main window
# ==========================================================================

class Planner(QMainWindow):
    def __init__(self, music_dir):
        super().__init__()
        self.music_dir = music_dir
        self.setWindowTitle(f"DJ Set Planner - {music_dir}")
        self.resize(1500, 900)
        os.makedirs(music_dir, exist_ok=True)
        self.db = LibraryDB(music_dir)
        self.library = []

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.library_tab = LibraryTab(self)
        self.analysis_tab = AnalysisTab(self)
        self.set_tab = SetTab(self)
        self.mix_tab = MixTab(self)
        self.tabs.addTab(self.library_tab, "Library")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.set_tab, "Set")
        self.tabs.addTab(self.mix_tab, "Mix")

        self.library_tab.openAnalysis.connect(self._open_analysis)
        self.library_tab.addTracks.connect(self.set_tab.add_tracks)
        self.set_tab.planCompiled.connect(self.mix_tab.set_plan)

        self.reload_library()
        self.set_tab.refresh_setlists()

    def reload_library(self, keep_analysis=False):
        self.library = load_library(self.db)
        self.library_tab.refresh()
        self.analysis_tab.refresh_tracklist()
        if not keep_analysis and self.set_tab.entries:
            self.set_tab.recompile()

    def _open_analysis(self, track):
        self.tabs.setCurrentWidget(self.analysis_tab)
        self.analysis_tab.open_track(track)

    def closeEvent(self, ev):
        self.analysis_tab.close()
        self.mix_tab.close()
        self.set_tab.seam_player.close()
        super().closeEvent(ev)


def main():
    music_dir = resolve_music_dir(
        sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(30, 30, 34))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Base, QColor(24, 24, 28))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(36, 36, 40))
    pal.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Button, QColor(45, 45, 50))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(50, 110, 160))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(120, 120, 130))
    app.setPalette(pal)
    w = Planner(music_dir)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
