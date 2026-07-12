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
from PyQt6.QtCore import (QAbstractItemModel, Qt, QAbstractTableModel, QModelIndex, QRect,
                          QSortFilterProxyModel, QThread, QTimer, QProcess,
                          pyqtSignal)
from PyQt6.QtGui import QColor, QPalette, QAction, QCursor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QLineEdit, QTableView, QListWidget, QListWidgetItem, QPushButton,
    QComboBox, QStyledItemDelegate, QSplitter, QMessageBox, QInputDialog,
    QAbstractItemView, QDoubleSpinBox, QSpinBox, QStyle, QTabWidget,
    QPlainTextEdit, QSlider, QMenu)

from lib.dj import resolve_music_dir
from lib.dj.db import LibraryDB
from lib.dj.brain import Brain, load_library
from lib.dj.themes import BUILTIN_THEMES, get_theme
from lib.dj import setlist as SL
from tools.djplanner.arcstrip import ArcStrip
from tools.djplanner.waveform import WaveformView
from tools.djplanner.mixview import MixTimeline
from tools.djplanner.deckmon import DeckMonitor
from tools.djplanner.player import TrackPlayer, PlanPreview

RATE = 44100
SECTION_COLORS = {
    "intro": QColor(70, 90, 120), "outro": QColor(70, 90, 120),
    "groove": QColor(60, 120, 90), "build": QColor(190, 150, 60),
    "breakdown": QColor(90, 70, 130),
}
COLS = ["title", "artist", "bpm", "key", "dur", "energy", "genre", "tags",
        "structure"]


def track_genre(t):
    """Best single genre label for a track: MusicBrainz genre (from the
    Enrich pass) first, else the embedded file genre tag."""
    g = getattr(t, "genres", None)
    if g:
        return g[0]
    return (getattr(t, "file_genre", "") or "").split(",")[0].split("/")[0].strip()


def energy_glyph(e):
    """Compact energy readout: number + a little bar, sorts as text too."""
    e = max(0.0, min(1.0, float(e or 0.0)))
    return f"{e:.2f} " + chr(9601 + int(e * 7 + 0.5))   # 0.00▁ .. 1.00█
FOLDER_ROLE = Qt.ItemDataRole.UserRole + 1     # folder row -> [TrackInfo]


def track_folder(t):
    """Subdirectory (relative to the music root) a track lives in."""
    import os as _os
    d = _os.path.dirname(t.path).replace("\\", "/")
    return d or "(root)"


# ==========================================================================
# Library tab
# ==========================================================================

class LibraryTreeModel(QAbstractItemModel):
    """Two-level tree: folders (collapsible to one line) -> tracks."""

    def __init__(self):
        super().__init__()
        self.folders = []            # [{"name": str, "tracks": [TrackInfo]}]

    def set_tracks(self, tracks, flat=False):
        self.beginResetModel()
        if flat:
            # One container holding EVERYTHING; the view roots itself at
            # this row, so tracks read as one big globally sortable list.
            self.folders = [{"name": "(all tracks)",
                             "tracks": list(tracks)}]
        else:
            by = {}
            for t in tracks:
                by.setdefault(track_folder(t), []).append(t)
            self.folders = [{"name": k, "tracks": v}
                            for k, v in sorted(by.items())]
        self.endResetModel()

    # -- structure ---------------------------------------------------------
    def index(self, row, col, parent=QModelIndex()):
        if not parent.isValid():
            if 0 <= row < len(self.folders):
                return self.createIndex(row, col)          # folder row
            return QModelIndex()
        f = self.folders[parent.row()]
        if 0 <= row < len(f["tracks"]):
            return self.createIndex(row, col, f)           # track row
        return QModelIndex()

    def parent(self, index):
        f = index.internalPointer()
        if f is None:
            return QModelIndex()
        try:
            return self.createIndex(self.folders.index(f), 0)
        except ValueError:
            return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            return len(self.folders)
        if parent.internalPointer() is None:               # a folder
            return len(self.folders[parent.row()]["tracks"])
        return 0

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def headerData(self, i, orient, role):
        if role == Qt.ItemDataRole.DisplayRole \
                and orient == Qt.Orientation.Horizontal:
            return COLS[i]
        return None

    def folder_name(self, row):
        return self.folders[row]["name"] if 0 <= row < len(self.folders) \
            else ""

    def data(self, index, role):
        f = index.internalPointer()
        c = index.column()
        if f is None:                                      # folder line
            fd = self.folders[index.row()]
            if role == Qt.ItemDataRole.DisplayRole and c == 0:
                return f"{fd['name']}   ({len(fd['tracks'])} tracks)"
            if role == FOLDER_ROLE:
                return fd["tracks"]
            return None
        t = f["tracks"][index.row()]
        if getattr(t, "excluded", False):
            if role == Qt.ItemDataRole.FontRole:
                from PyQt6.QtGui import QFont
                fnt = QFont()
                fnt.setStrikeOut(True)
                fnt.setItalic(True)
                return fnt
            if role == Qt.ItemDataRole.ForegroundRole:
                from PyQt6.QtGui import QColor
                return QColor("#8a8a95")
            if role == Qt.ItemDataRole.DisplayRole and c == 0:
                return "🚫 " + t.title
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
                return energy_glyph(t.energy_proxy())
            if c == 6:
                return track_genre(t)
            if c == 7:
                return " ".join(t.all_tags)
        if role == Qt.ItemDataRole.UserRole:
            return t
        return None


class LibraryProxy(QSortFilterProxyModel):
    """Text search over tracks AND folder names, plus a one-folder filter.
    A folder stays visible while any of its tracks match."""

    def __init__(self):
        super().__init__()
        self.text = ""
        self.folder = None           # None = all folders
        self.tag = None              # None = all tags
        self.flat = False            # flat list: folder filter applies to
                                     # each track's REAL directory

    def set_text(self, text):
        self.text = (text or "").lower()
        self.invalidateFilter()

    def set_folder(self, name):
        self.folder = name
        self.invalidateFilter()

    def set_tag(self, tag):
        self.tag = tag
        self.invalidateFilter()

    def _track_matches(self, m, f_row, t_row):
        fd = m.folders[f_row]
        t = fd["tracks"][t_row]
        if self.flat and self.folder is not None \
                and track_folder(t) != self.folder:
            return False
        if self.tag is not None and self.tag not in t.all_tags:
            return False
        hay = f"{t.title} {t.artist} {' '.join(t.all_tags)}".lower()
        return self.text in hay

    def lessThan(self, left, right):
        # Numeric columns sort numerically (bpm/dur/energy) - string
        # sorting put 99 bpm after 121.
        m = self.sourceModel()
        ta = m.data(left, Qt.ItemDataRole.UserRole)
        tb = m.data(right, Qt.ItemDataRole.UserRole)
        if ta is not None and tb is not None:
            c = left.column()
            if c == 2:
                return (ta.bpm or 0.0) < (tb.bpm or 0.0)
            if c == 4:
                return (ta.duration_s or 0.0) < (tb.duration_s or 0.0)
            if c == 5:
                return ta.energy_proxy() < tb.energy_proxy()
            ka = (m.data(left, Qt.ItemDataRole.DisplayRole) or "").lower()
            kb = (m.data(right, Qt.ItemDataRole.DisplayRole) or "").lower()
            return ka < kb
        return super().lessThan(left, right)

    def filterAcceptsRow(self, row, parent):
        m = self.sourceModel()
        if not parent.isValid():                           # folder row
            if self.flat:
                return True                    # single root-hidden container
            name = m.folder_name(row)
            if self.folder is not None and name != self.folder:
                return False
            if self.tag is None and (not self.text
                                     or self.text in name.lower()):
                return True
            return any(self._track_matches(m, row, r)
                       for r in range(len(m.folders[row]["tracks"])))
        f_row = parent.row()
        if not self.flat:
            if self.folder is not None \
                    and m.folder_name(f_row) != self.folder:
                return False
            if self.tag is None and self.text \
                    and self.text in m.folder_name(f_row).lower():
                return True
        return self._track_matches(m, f_row, row)


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


class EnrichWorker(QThread):
    """MusicBrainz enrichment in-process (no subprocess). Rate-limited to
    ~1 req/s inside the client, so it runs on this thread and streams
    progress. Opens its own DB connection on the worker thread."""
    progress = pyqtSignal(int, int, int, int, str)  # done,total,matched,missed,cur
    finished_run = pyqtSignal(int, int)             # matched, missed

    def __init__(self, music_dir, force=False):
        super().__init__()
        self.music_dir, self.force = music_dir, force
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        from lib.dj.db import LibraryDB
        from lib.dj.enrich import MusicBrainzClient, enrich_track
        db = LibraryDB(self.music_dir)
        rows = db.all_tracks()
        todo = [r for r in rows if self.force or not r.get("enrichment")]
        mb = MusicBrainzClient()
        matched = missed = 0
        for i, r in enumerate(todo):
            if self._stop:
                break
            track = {"title": r.get("title") or "",
                     "artist": r.get("artist") or "",
                     "duration_s": r.get("duration_s")}
            try:
                blob = enrich_track(track, mb=mb)
            except Exception:
                blob = None
            db.set_enrichment(r["id"], blob or {"source": "musicbrainz",
                                                "matched": False})
            if blob:
                matched += 1
            else:
                missed += 1
            self.progress.emit(i + 1, len(todo), matched, missed,
                              (r.get("title") or "")[:36])
        db.close()
        self.finished_run.emit(matched, missed)


class LibraryTab(QWidget):
    openAnalysis = pyqtSignal(object)          # TrackInfo
    addTracks = pyqtSignal(list)               # [TrackInfo]

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self._proc = None
        self._enrich = None
        self._mood_proc = None
        v = QVBoxLayout(self)

        top = QHBoxLayout()
        self.scan_btn = QPushButton("Scan library")
        self.scan_btn.setToolTip("Analyze new/changed tracks and any whose "
                                 "analysis is older than the current feature "
                                 "set (incremental).")
        self.scan_btn.clicked.connect(lambda: self.run_scan(force=False))
        top.addWidget(self.scan_btn)
        self.rescan_btn = QPushButton("Rescan all")
        self.rescan_btn.setToolTip("Force a full re-analysis of EVERY track "
                                   "(ignores the up-to-date check). Use after "
                                   "a big library move or to be certain every "
                                   "track has the latest features.")
        self.rescan_btn.clicked.connect(lambda: self.run_scan(force=True))
        top.addWidget(self.rescan_btn)
        # Enrich = pull genre/year/era/label from MusicBrainz (in-process).
        self.enrich_btn = QPushButton("Enrich (MusicBrainz)")
        self.enrich_btn.setToolTip(
            "Fetch genre, release year/era and label from MusicBrainz for "
            "every track that lacks it. Genres become tags that steer "
            "selection and the copilot. ~1 track/sec; background, resumable.")
        self.enrich_btn.clicked.connect(self.run_enrich)
        top.addWidget(self.enrich_btn)
        # Mood (ML) = score valence/arousal/mood tags with Music2Emo (GPU pass).
        self.mood_btn = QPushButton("Mood (ML)")
        self.mood_btn.setToolTip(
            "Run the Music2Emo model over every un-scored track to get real "
            "valence/arousal and mood tags (dark, party, melancholic, epic...). "
            "Upgrades danceable/dark/uplifting tags and valence steering from "
            "heuristic to ML. Needs a Music2Emotion clone; ~3-5s/track on GPU, "
            "background and resumable.")
        self.mood_btn.clicked.connect(self.run_mood)
        top.addWidget(self.mood_btn)
        self.scan_lbl = QLabel("")
        top.addWidget(self.scan_lbl, 1)
        self.search = QLineEdit(
            placeholderText="search title / artist / folder / tag...")
        self.search.textChanged.connect(
            lambda s: self.proxy.set_text(s))
        top.addWidget(self.search, 2)
        # One-tap folder filter: every subdirectory in the library.
        from PyQt6.QtWidgets import QComboBox
        self.folder_box = QComboBox()
        self.folder_box.addItem("all folders")
        self.folder_box.currentTextChanged.connect(self._folder_filter)
        top.addWidget(self.folder_box, 1)
        # Tag BROWSER (not just search): every tag with its count.
        self.tag_box = QComboBox()
        self.tag_box.addItem("all tags")
        self.tag_box.currentTextChanged.connect(self._tag_filter)
        top.addWidget(self.tag_box, 1)
        v.addLayout(top)

        # Quick-audition transport for the library list.
        self.lib_player = TrackPlayer()
        self._lib_decoder = None
        self._play_list = []
        self._play_idx = -1
        prow = QHBoxLayout()
        for label, cb in (("|< Back", lambda: self._step_play(-1)),
                          ("> Play", self._play_selected),
                          ("|| Pause", self._pause_play),
                          ("[] Stop", lambda: self.planner.stop_all_playback()),
                          (">| Next", lambda: self._step_play(1)),
                          ("<< 15s", lambda: self._seek_rel(-15.0)),
                          ("15s >>", lambda: self._seek_rel(15.0))):
            b = QPushButton(label)
            b.clicked.connect(cb)
            prow.addWidget(b)
        add_btn = QPushButton("Add selected to set →")
        add_btn.clicked.connect(self._add_selected)
        prow.addWidget(add_btn)
        self.flat_btn = QPushButton("View: folders")
        self.flat_btn.setCheckable(True)
        self.flat_btn.setToolTip("Toggle between the directory tree and "
                                 "one big sortable list of every track")
        self.flat_btn.toggled.connect(self._toggle_flat)
        prow.addWidget(self.flat_btn)
        self.play_lbl = QLabel("")
        prow.addWidget(self.play_lbl, 1)
        v.addLayout(prow)

        # SCRUB BAR: skip around inside the playing track (evaluation is
        # mostly 'what do the drop and the middle sound like?' - nobody
        # wants to sit through the intro to find out). Live-seeks while
        # dragging; TrackPlayer's seek is sample-accurate and instant.
        srow = QHBoxLayout()
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self._seek_drag = False
        self.seek_slider.sliderPressed.connect(
            lambda: setattr(self, "_seek_drag", True))
        self.seek_slider.sliderReleased.connect(self._seek_released)
        self.seek_slider.sliderMoved.connect(self._seek_frac)
        srow.addWidget(self.seek_slider, 1)
        self.time_lbl = QLabel("-:-- / -:--")
        srow.addWidget(self.time_lbl)
        v.addLayout(srow)
        self._seek_timer = QTimer(self)
        self._seek_timer.timeout.connect(self._seek_tick)
        self._seek_timer.start(150)

        self.model = LibraryTreeModel()
        self.proxy = LibraryProxy()
        self.proxy.setSourceModel(self.model)
        from PyQt6.QtWidgets import QTreeView
        self.table = QTreeView()
        self.table.setModel(self.proxy)
        self.table.setUniformRowHeights(True)
        self.table.setExpandsOnDoubleClick(False)   # dbl-click = analysis
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setItemDelegateForColumn(8, StripDelegate())
        for i, w in enumerate((250, 120, 52, 44, 50, 62, 100, 150, 260)):
            self.table.setColumnWidth(i, w)
        self.table.doubleClicked.connect(
            lambda _: self._open_analysis())
        self.table.selectionModel().selectionChanged.connect(
            self._extend_folder_selection)
        # Right-click: same do-not-use toggle as the buttons below.
        excl_act = QAction("🚫 Mark do-not-use", self)
        excl_act.triggered.connect(lambda: self._set_excluded(True))
        self.table.addAction(excl_act)
        allow_act = QAction("✓ Allow (clear do-not-use)", self)
        allow_act.triggered.connect(lambda: self._set_excluded(False))
        self.table.addAction(allow_act)
        self.table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.ActionsContextMenu)
        v.addWidget(self.table)

        bot = QHBoxLayout()
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
        bot.addSpacing(16)
        # Do-not-use: hide from ALL auto-selection (set generation, copilot,
        # autoDJ). Track stays in the library, greyed + struck through.
        bx = QPushButton("🚫 Do not use")
        bx.setToolTip("Flag the selected tracks so no system grabs them - "
                      "set generation, the Copilot, and the live autoDJ all "
                      "skip them. They stay in the library; hit Allow to undo.")
        bx.clicked.connect(lambda: self._set_excluded(True))
        bot.addWidget(bx)
        ba = QPushButton("✓ Allow")
        ba.setToolTip("Clear the do-not-use flag on the selected tracks.")
        ba.clicked.connect(lambda: self._set_excluded(False))
        bot.addWidget(ba)
        bot.addStretch(1)
        self.count_lbl = QLabel("")
        bot.addWidget(self.count_lbl)
        v.addLayout(bot)

        self._scan_timer = QTimer(self)
        self._scan_timer.timeout.connect(self._poll_scan)
        self._last_done = -1

    def selected_tracks(self):
        """Selected track rows; selecting a FOLDER line means the whole
        folder (in visible/filtered order)."""
        out, seen = [], set()
        for idx in self.table.selectionModel().selectedRows():
            t = self.proxy.data(idx, Qt.ItemDataRole.UserRole)
            if t is not None:
                if t.id not in seen:
                    seen.add(t.id); out.append(t)
                continue
            for tr in (self.proxy.data(idx, FOLDER_ROLE) or []):
                if tr.id not in seen:
                    seen.add(tr.id); out.append(tr)
        return out

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

    def _set_excluded(self, flag):
        """Flag/unflag the selected tracks 'do not use' (DB v11). Reload so
        they drop out of (or return to) planner.library everywhere at once."""
        ts = self.selected_tracks()
        if not ts:
            return
        for t in ts:
            self.planner.db.set_excluded(t.id, flag)
        self.planner.reload_library()
        n = len(ts)
        self.scan_lbl.setText(
            f"{'flagged do-not-use' if flag else 'allowed'}: {n} track(s)")

    def _extend_folder_selection(self, selected, _deselected):
        """Clicking a folder line selects every track inside it - the
        one-gesture 'select all songs in this directory'."""
        if getattr(self, "_extending_sel", False):
            return
        from PyQt6.QtCore import QItemSelection, QItemSelectionModel
        add = QItemSelection()
        for idx in selected.indexes():
            if idx.column() != 0:
                continue
            if self.proxy.data(idx, Qt.ItemDataRole.UserRole) is None \
                    and self.proxy.data(idx, FOLDER_ROLE) is not None:
                n = self.proxy.rowCount(idx)
                if n:
                    add.select(self.proxy.index(0, 0, idx),
                               self.proxy.index(n - 1, 0, idx))
                self.table.setExpanded(idx, True)
        if not add.isEmpty():
            self._extending_sel = True
            self.table.selectionModel().select(
                add, QItemSelectionModel.SelectionFlag.Select
                | QItemSelectionModel.SelectionFlag.Rows)
            self._extending_sel = False

    # -- quick audition ------------------------------------------------------
    def _visible_tracks(self):
        out = []
        for fr in range(self.proxy.rowCount()):
            fidx = self.proxy.index(fr, 0)
            for tr in range(self.proxy.rowCount(fidx)):
                t = self.proxy.data(self.proxy.index(tr, 0, fidx),
                                    Qt.ItemDataRole.UserRole)
                if t is not None:
                    out.append(t)
        return out

    def _play_track(self, t):
        from tools.djplanner.player import TrackPlayer  # noqa: F401
        self.planner.claim_playback("library")
        self.play_lbl.setText(f"decoding  {t.title[:40]}...")

        class _D(QThread):
            done = pyqtSignal(object)

            def __init__(self, db, tr):
                super().__init__()
                self.db, self.tr = db, tr

            def run(self):
                try:
                    from lib.dj.features import decode_file_stereo
                    self.done.emit(decode_file_stereo(
                        self.db.abs(self.tr.path)))
                except Exception as e:
                    self.done.emit(str(e))
        self._lib_decoder = _D(self.planner.db, t)

        def _go(samples, tr=t):
            if isinstance(samples, str):
                self.play_lbl.setText(f"decode failed: {samples[:60]}")
                return
            self.planner.claim_playback("library")
            self.lib_player.load(samples)
            self.lib_player.play()
            self.play_lbl.setText(f"playing  {tr.title[:44]}")
        self._lib_decoder.done.connect(_go)
        self._lib_decoder.start()

    def _play_selected(self):
        sel = self.selected_tracks()
        self._play_list = sel if len(sel) > 1 else self._visible_tracks()
        t = sel[0] if sel else (self._play_list[0] if self._play_list
                                else None)
        if t is None:
            return
        self._play_idx = next((i for i, x in enumerate(self._play_list)
                               if x.id == t.id), 0)
        self._play_track(t)

    # -- scrubbing ----------------------------------------------------------
    def _dur_s(self):
        p = self.lib_player
        return (len(p.samples) / 44100.0
                if p.samples is not None and len(p.samples) else 0.0)

    def _seek_frac(self, v):
        d = self._dur_s()
        if d > 0:
            self.lib_player.seek(v / 1000.0 * d)

    def _seek_released(self):
        self._seek_drag = False
        self._seek_frac(self.seek_slider.value())

    def _seek_rel(self, dt):
        p = self.lib_player
        if p.samples is not None:
            p.seek(max(0.0, p.time_s() + dt))

    @staticmethod
    def _mmss(t):
        return f"{int(t // 60)}:{int(t % 60):02d}"

    def _seek_tick(self):
        d = self._dur_s()
        if d <= 0:
            self.time_lbl.setText("-:-- / -:--")
            if not self._seek_drag:
                self.seek_slider.setValue(0)
            return
        t = self.lib_player.time_s()
        self.time_lbl.setText(f"{self._mmss(t)} / {self._mmss(d)}")
        if not self._seek_drag:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(int(t / d * 1000))
            self.seek_slider.blockSignals(False)

    def _pause_play(self):
        if self.lib_player.playing:
            self.lib_player.pause()
            self.play_lbl.setText(self.play_lbl.text().replace(
                "playing", "paused"))
        elif self.lib_player.samples is not None:
            self.planner.claim_playback("library")
            self.lib_player.play()
            self.play_lbl.setText(self.play_lbl.text().replace(
                "paused", "playing"))

    def _step_play(self, d):
        if not self._play_list:
            self._play_list = self._visible_tracks()
        if not self._play_list:
            return
        self._play_idx = (self._play_idx + d) % len(self._play_list)
        self._play_track(self._play_list[self._play_idx])

    def stop_playback(self):
        self.lib_player.pause()
        self.lib_player.samples = None
        self.play_lbl.setText("")

    def _folder_filter(self, name):
        self.proxy.set_folder(None if name == "all folders" else name)

    def _tag_filter(self, label):
        tag = None if label == "all tags" else label.split("  (")[0]
        self.proxy.set_tag(tag)

    def _toggle_flat(self, on):
        self.flat_btn.setText("View: flat list" if on else "View: folders")
        self.proxy.flat = bool(on)
        self.refresh()

    def refresh(self):
        flat = getattr(self, "flat_btn", None) is not None             and self.flat_btn.isChecked()
        # Preserve which folders are open across rescans/tag edits.
        open_names = set()
        if not flat:
            for r in range(self.proxy.rowCount()):
                idx = self.proxy.index(r, 0)
                if self.table.isExpanded(idx):
                    src = self.proxy.mapToSource(idx)
                    open_names.add(self.model.folder_name(src.row()))
        first = not self.model.folders
        tracks = self.planner.library_all or self.planner.library
        self.model.set_tracks(tracks, flat=flat)
        n_excl = sum(1 for t in tracks if getattr(t, "excluded", False))
        self.count_lbl.setText(
            f"{len(tracks)} tracks"
            + (f"  ({n_excl} do-not-use)" if n_excl else ""))
        if flat:
            # Root the view AT the single container: pure sortable list.
            src0 = self.model.index(0, 0)
            self.table.setRootIndex(self.proxy.mapFromSource(src0))
        else:
            from PyQt6.QtCore import QModelIndex
            self.table.setRootIndex(QModelIndex())
            for r in range(self.proxy.rowCount()):
                idx = self.proxy.index(r, 0)
                src = self.proxy.mapToSource(idx)
                name = self.model.folder_name(src.row())
                self.table.setExpanded(idx, first or name in open_names)
        counts = {}
        for t in self.planner.library:
            for tag in t.all_tags:
                counts[tag] = counts.get(tag, 0) + 1
        want_tags = ["all tags"] + [f"{k}  ({n})" for k, n in
                                    sorted(counts.items())]
        have_tags = [self.tag_box.itemText(i)
                     for i in range(self.tag_box.count())]
        if have_tags != want_tags:
            cur = self.tag_box.currentText()
            self.tag_box.blockSignals(True)
            self.tag_box.clear()
            self.tag_box.addItems(want_tags)
            if cur in want_tags:
                self.tag_box.setCurrentText(cur)
            self.tag_box.blockSignals(False)
        folders = sorted({track_folder(t) for t in self.planner.library})
        have = [self.folder_box.itemText(i)
                for i in range(self.folder_box.count())]
        want = ["all folders"] + folders
        if have != want:
            cur = self.folder_box.currentText()
            self.folder_box.blockSignals(True)
            self.folder_box.clear()
            self.folder_box.addItems(want)
            if cur in want:
                self.folder_box.setCurrentText(cur)
            self.folder_box.blockSignals(False)

    # -- scanning ------------------------------------------------------------
    def run_scan(self, force=False):
        if self._proc is not None:
            return
        script = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "tools", "dj_scan.py")
        args = [script, "--dir", self.planner.music_dir]
        if force:
            args.append("--force")
        self._proc = QProcess(self)
        self._proc.finished.connect(self._scan_done)
        self._proc.start(sys.executable, args)
        self.scan_btn.setEnabled(False)
        self.rescan_btn.setEnabled(False)
        self.scan_lbl.setText("rescanning all..." if force else "scanning...")
        self._last_done = -1
        self._scan_timer.start(1500)

    def _poll_scan(self):
        from lib.dj.scan import PROGRESS_PATH
        try:
            with open(PROGRESS_PATH) as f:
                pr = json.load(f)
        except Exception:
            return
        if pr.get("phase") == "vocals":
            self.scan_lbl.setText(
                f"vocal pass {pr.get('vocals_done', 0)}"
                f"/{pr.get('vocals_total', '?')}  "
                f"{pr.get('current', '')[:40]}")
            done = pr.get("vocals_done", 0)
        else:
            self.scan_lbl.setText(
                f"scanning {pr.get('done', 0)}/{pr.get('total', '?')}  "
                f"{pr.get('current', '')[:40]}")
            done = pr.get("done", 0)
        # New analyses landed -> live-populate the table.
        if done != self._last_done:
            self._last_done = done
            self.planner.reload_library()

    def _scan_done(self, *a):
        self._proc = None
        self._scan_timer.stop()
        self.scan_btn.setEnabled(True)
        self.rescan_btn.setEnabled(True)
        self.scan_lbl.setText("scan complete")
        self.planner.reload_library()

    # -- enrichment (MusicBrainz, in-process) --------------------------------
    def run_enrich(self):
        if self._enrich is not None and self._enrich.isRunning():
            self._enrich.stop()                      # button toggles to stop
            self.enrich_btn.setText("stopping...")
            return
        self._enrich = EnrichWorker(self.planner.music_dir)
        self._enrich.progress.connect(self._enrich_progress)
        self._enrich.finished_run.connect(self._enrich_done)
        self._enrich.start()
        self.enrich_btn.setText("Stop enrich")

    def _enrich_progress(self, done, total, matched, missed, cur):
        self.scan_lbl.setText(f"enriching {done}/{total}  "
                              f"({matched} matched, {missed} no-match)  {cur}")
        # Live-populate genres into the table + tag browser every 15 tracks.
        if done % 15 == 0:
            self.planner.reload_library()

    def _enrich_done(self, matched, missed):
        self.enrich_btn.setText("Enrich (MusicBrainz)")
        self.scan_lbl.setText(f"enrichment done: {matched} matched, "
                              f"{missed} no-match")
        self.planner.reload_library()

    # -- mood (Music2Emo ML, subprocess) -------------------------------------
    def run_mood(self):
        if self._mood_proc is not None:                  # toggles to stop
            self._mood_proc.kill()
            self.mood_btn.setText("stopping...")
            return
        from lib.dj.mood_ml import find_model_dir
        if find_model_dir() is None:
            self.scan_lbl.setText(
                "Mood model not found - clone github.com/AMAAI-Lab/"
                "Music2Emotion next to GL_Simple or set $DJ_MOOD_MODEL_DIR")
            return
        script = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "tools", "dj_mood.py")
        self._mood_proc = QProcess(self)
        self._mood_proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self._mood_proc.readyReadStandardOutput.connect(self._mood_stdout)
        self._mood_proc.finished.connect(self._mood_done)
        self._mood_proc.start(sys.executable,
                              [script, "--dir", self.planner.music_dir])
        self._mood_done_count = -1
        self.mood_btn.setText("Stop mood")
        self.scan_lbl.setText("mood pass: loading model...")

    def _mood_stdout(self):
        if self._mood_proc is None:
            return
        data = bytes(self._mood_proc.readAllStandardOutput()).decode(
            "utf-8", "replace")
        for line in data.splitlines():
            if not line.startswith("PROGRESS "):
                continue
            parts = line.split(" ", 5)
            if len(parts) < 6:
                continue
            done, total, matched, missed, cur = parts[1:6]
            self.scan_lbl.setText(
                f"mood {done}/{total}  ({matched} scored, {missed} failed)  "
                f"{cur[:40]}")
            # Live-populate mood tags into the table + tag browser periodically.
            try:
                d = int(done)
            except ValueError:
                continue
            if d != self._mood_done_count and d % 15 == 0:
                self._mood_done_count = d
                self.planner.reload_library()

    def _mood_done(self, *a):
        self._mood_proc = None
        self.mood_btn.setText("Mood (ML)")
        self.scan_lbl.setText("mood pass complete")
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
        def _n(v):
            return float(v) if v is not None else 0.0
        secs = "\n".join(
            f"  {s['kind']:9s} {_n(s['start_s']):6.1f}-{_n(s['end_s']):6.1f}s  "
            f"energy {_n(s['energy']):.2f}  busy {_n(s['busyness']):.2f}  "
            f"vocal {_n(s['vocalness']):.2f}  rep {_n(s['repetitiveness']):.2f}"
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
            self.planner.claim_playback("analysis")
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

    def __init__(self, library, entries, theme, pair_memory=None):
        super().__init__()
        self.library, self.entries, self.theme = library, entries, theme
        self.pair_memory = dict(pair_memory or {})

    def run(self):
        try:
            self.done.emit(SL.compile_plan(self.library, self.entries,
                                           self.theme,
                                           pair_memory=self.pair_memory))
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

        # SHAPE the set: order it so tempo/energy follows a curve (the only
        # control that actually makes a set BUILD; keeps seams mixable).
        prow3 = QHBoxLayout()
        prow3.addWidget(QLabel("Shape:"))
        self.shape_metric = QComboBox()
        self.shape_metric.addItems(["tempo", "energy"])
        prow3.addWidget(self.shape_metric)
        self.shape_curve = QComboBox()
        self.shape_curve.addItems(["rise", "peak", "wind_down", "flat"])
        prow3.addWidget(self.shape_curve)
        shb = QPushButton("Apply shape")
        shb.setToolTip("Reorder the set so the chosen metric follows the "
                       "curve — rise = build, peak = up then down. Tempo "
                       "shaping gives the cleanest result.")
        shb.clicked.connect(self.apply_shape)
        prow3.addWidget(shb)
        prow3.addStretch(1)
        left.addLayout(prow3)

        # THE ARC STRIP: the set's shape (energy vs the theme's target,
        # bpm path, seam quality) visible WHILE building, not only after
        # reading the compiled text. Click a bar to select its entry.
        self.arc_strip = ArcStrip()
        self.arc_strip.slotClicked.connect(self._strip_clicked)
        left.addWidget(self.arc_strip)
        left.addWidget(QLabel("Set (drag to reorder, double-click = anchor,"
                              " Del = remove, right-click = repair)"))
        self.set_list = QListWidget()
        self.set_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove)
        self.set_list.model().rowsMoved.connect(self._reordered)
        self.set_list.itemDoubleClicked.connect(self._toggle_anchor)
        rm = QAction("remove", self)
        rm.setShortcut("Delete")
        rm.triggered.connect(self._remove_entry)
        self.set_list.addAction(rm)
        alt = QAction("suggest alternatives for this slot", self)
        alt.triggered.connect(self._suggest_alternatives)
        self.set_list.addAction(alt)
        br = QAction("insert bridge to next track", self)
        br.triggered.connect(self._insert_bridge)
        self.set_list.addAction(br)
        self.set_list.setContextMenuPolicy(
            Qt.ContextMenuPolicy.ActionsContextMenu)
        left.addWidget(self.set_list, 1)
        rrow = QHBoxLayout()
        rm_btn = QPushButton("Remove selected (Del)")
        rm_btn.clicked.connect(self._remove_entry)
        rrow.addWidget(rm_btn)
        dup_btn = QPushButton("Remove duplicates")
        dup_btn.setToolTip("Detects the same song appearing twice - "
                           "including byte-identical copies living in "
                           "other directories (content hash) and re-rips "
                           "(same title/artist, similar length). Keeps "
                           "the anchored or first copy.")
        dup_btn.clicked.connect(self.remove_duplicates)
        rrow.addWidget(dup_btn)
        rrow.addStretch(1)
        left.addLayout(rrow)
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
        st.clicked.connect(lambda: self.planner.stop_all_playback())
        brow.addWidget(st)
        brow.addStretch(1)
        right.addLayout(brow)
        # Play the whole compiled set WITHOUT leaving the Set tab - drives the
        # Mix tab's preview (which already holds the current plan via the
        # planCompiled->set_plan wiring), so audio plays regardless of which
        # tab is showing.
        pbrow = QHBoxLayout()
        self.play_set_btn = QPushButton("▶ Play set")
        self.play_set_btn.setToolTip("Play the whole compiled set from the "
                                     "start (same engine as the Mix tab).")
        self.play_set_btn.clicked.connect(self._play_set)
        pbrow.addWidget(self.play_set_btn)
        self.set_pause_btn = QPushButton("⏸")
        self.set_pause_btn.setCheckable(True)
        self.set_pause_btn.toggled.connect(
            lambda on: self.planner.mix_tab._pause(on))
        pbrow.addWidget(self.set_pause_btn)
        pstop = QPushButton("■ Stop")
        pstop.clicked.connect(lambda: self.planner.stop_all_playback())
        pbrow.addWidget(pstop)
        pnext = QPushButton("⏭ track")
        pnext.clicked.connect(lambda: self.planner.mix_tab._jump(+1))
        pbrow.addWidget(pnext)
        pseam = QPushButton("→ seam")
        pseam.clicked.connect(lambda: self.planner.mix_tab._next_seam())
        pbrow.addWidget(pseam)
        pbrow.addStretch(1)
        right.addLayout(pbrow)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        # Conversational set-builder (Claude tool-loop over this same set).
        from tools.djplanner.copilot_panel import CopilotPanel
        self.copilot_panel = CopilotPanel(planner, self)
        self.copilot_panel.entriesApplied.connect(self._copilot_applied)
        right.addWidget(self.copilot_panel, 1)
        h.addLayout(right, 1)

    # -- entries ------------------------------------------------------------
    def theme(self):
        return get_theme(self.theme_combo.currentText())

    def add_tracks(self, tracks):
        for t in tracks:
            self.entries.append({"track_id": t.id, "pin_type": "suggestion",
                                 "target_offset_min": None,
                                 "style_override": None,
                                 "target_play_s": None})
        self._rebuild()
        self.recompile()

    def _entry_label(self, e):
        t = next((x for x in (self.planner.library_all or self.planner.library)
                  if x.id == e["track_id"]), None)
        name = t.title if t else f"track {e['track_id']}"
        tag = "⚓ " if e["pin_type"] == "anchor" else "• "
        pin = (f"  @{e['target_offset_min']:.0f}min"
               if e.get("target_offset_min") else "")
        bpm = (f"  ({t.bpm:.0f} {t.camelot}  "
               f"{energy_glyph(t.energy_proxy())})" if t else "")
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

    def _dup_keys(self, t):
        """Identity keys for duplicate detection. A byte-identical copy
        in another directory shares the scan's content hash; a re-rip or
        re-encode falls back to normalized title/artist plus a coarse
        length bucket (so a radio edit never collides with the extended
        mix of the same name)."""
        keys = []
        h = ((t.row.get("content_hash") or "").strip()
             if getattr(t, "row", None) else "")
        if h:
            keys.append(("h", h))
        ti = (t.title or "").strip().lower()
        ar = (t.artist or "").strip().lower()
        if ti:
            keys.append(("m", ti, ar, int((t.duration_s or 0.0) // 8)))
        return keys

    def remove_duplicates(self):
        by_id = {t.id: t for t in self.planner.library}
        # Union entries that share ANY identity key.
        owner = {}                    # key -> first entry index
        parent = list(range(len(self.entries)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, e in enumerate(self.entries):
            t = by_id.get(e["track_id"])
            keys = self._dup_keys(t) if t is not None else []
            keys.append(("id", e["track_id"]))     # literal repeats
            for k in keys:
                if k in owner:
                    parent[find(i)] = find(owner[k])
                else:
                    owner[k] = i
        groups = {}
        for i in range(len(self.entries)):
            groups.setdefault(find(i), []).append(i)
        drops = []
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            keep = next((i for i in idxs
                         if self.entries[i]["pin_type"] == "anchor"),
                        idxs[0])
            drops.extend(i for i in idxs if i != keep)
        if not drops:
            self.status.setText("No duplicates found - every entry is a "
                                "distinct song.")
            return
        drops.sort()

        def _title(i):
            t = by_id.get(self.entries[i]["track_id"])
            path = (t.row.get("path", "") if t is not None
                    and getattr(t, "row", None) else "")
            return (t.title if t else f"track {self.entries[i]['track_id']}"
                    ) + (f"   [{path}]" if path else "")
        preview = "\n".join(_title(i) for i in drops[:12])
        if len(drops) > 12:
            preview += f"\n... and {len(drops) - 12} more"
        if QMessageBox.question(
                self, "Remove duplicates",
                f"Remove {len(drops)} duplicate entr"
                f"{'y' if len(drops) == 1 else 'ies'}?\n\n{preview}")                 != QMessageBox.StandardButton.Yes:
            return
        for i in reversed(drops):
            self.entries.pop(i)
        self._rebuild()
        self.recompile()
        self.status.setText(f"Removed {len(drops)} duplicate"
                            f"{'' if len(drops) == 1 else 's'} - kept the "
                            "anchored/first copy of each song.")

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

    def apply_shape(self):
        if len(self.entries) <= 2:
            self.status.setText("build a set first, then shape its curve.")
            return
        self.entries = SL.order_by_shape(
            self.planner.library, self.entries, self.theme(),
            metric=self.shape_metric.currentText(),
            shape=self.shape_curve.currentText())
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
                          "style_override", "target_play_s")}
                        for e in sl["entries"]]
        if sl.get("theme") in BUILTIN_THEMES:
            self.theme_combo.setCurrentText(sl["theme"])
        self._rebuild()
        self.recompile()

    def _create_named_setlist(self, title):
        """Prompt for a name and create an (empty) setlist row. Returns its id
        or None. Does NOT touch self.entries - callers decide what to save."""
        name, ok = QInputDialog.getText(self, title, "Name:")
        if not (ok and name):
            return None
        try:
            return SL.create_setlist(
                self.planner.db, name, theme=self.theme_combo.currentText())
        except Exception as e:
            QMessageBox.warning(self, title, str(e))
            return None

    def _select_combo_silently(self, sid):
        """Reflect a setlist id in the combo without firing _load_set (which
        would reload entries from the DB and clobber the working set)."""
        row = self.planner.db.conn.execute(
            "SELECT name FROM setlists WHERE id = ?", (sid,)).fetchone()
        if row:
            self.set_combo.blockSignals(True)
            self.set_combo.setCurrentText(row["name"])
            self.set_combo.blockSignals(False)

    def new_set(self):
        sid = self._create_named_setlist("New setlist")
        if sid is None:
            return
        self.setlist_id = sid
        self.entries = []                    # New = deliberately start empty
        self._rebuild()
        self.refresh_setlists()
        self._select_combo_silently(sid)

    def save_set(self):
        # An unsaved set: name + create the row, but KEEP the working entries
        # (the bug was calling new_set(), which cleared them before saving).
        if self.setlist_id is None:
            sid = self._create_named_setlist("Save setlist as")
            if sid is None:
                return
            self.setlist_id = sid
        SL.save_entries(self.planner.db, self.setlist_id, self.entries)
        self.planner.db.conn.execute(
            "UPDATE setlists SET theme = ? WHERE id = ?",
            (self.theme_combo.currentText(), self.setlist_id))
        self.planner.db.conn.commit()
        self.refresh_setlists()
        self._select_combo_silently(self.setlist_id)
        self.status.setText(f"saved {len(self.entries)} tracks.")

    def _play_set(self):
        if not self.compiled:
            self.status.setText("build a set first")
            return
        self.set_pause_btn.setChecked(False)
        self.planner.mix_tab.play_set()
        self.status.setText("playing set (Mix tab shows the live timeline).")

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
        self._worker = CompileWorker(
            self.planner.library, list(self.entries), self.theme(),
            pair_memory=getattr(self.planner, "pair_memory", None))
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
                si = s.get("seam_info") or {}
                badges = []
                if si.get("fade"):
                    badges.append("FADE")
                if si.get("floor") is not None and si["floor"] < 0.25:
                    badges.append(f"floor {si['floor']:.2f}")
                if (si.get("d_off") or 0) > 0.035:
                    badges.append(f"Δgroove {si['d_off'] * 1000:.0f}ms")
                pm = si.get("pair_mem")
                if pm:
                    badges.append("★ mixed well before" if pm > 1.0
                                  else "✖ rough before")
                warn = ("   ⚠ " + "; ".join(s["warnings"])
                        if s["warnings"] else "")
                item = QListWidgetItem(
                    f"      ↳ {p['style']} @ {p['out_s']:.0f}s "
                    f"(rate {p['rate']:.3f}, seam {p['pair_score']:.2f}"
                    + ("".join(", " + b for b in badges)) + ") "
                    f"→ {nxt.title}{warn}", self.plan_list)
                item.setData(Qt.ItemDataRole.UserRole, i)
                if si.get("floor") is not None and si["floor"] < 0.15:
                    item.setForeground(QColor(230, 110, 110))
                elif s["warnings"]:
                    item.setForeground(QColor(255, 170, 100))
                elif si.get("fade"):
                    item.setForeground(QColor(160, 160, 170))
        self._update_strip(result)
        self.status.setText(self._report_card(result))
        self.planCompiled.emit(result)

    def _update_strip(self, result):
        theme = self.theme()
        strip = [{"off": s["start_offset_s"], "play": s["play_s"],
                  "energy": s["track"].energy_proxy(),
                  "bpm": s["track"].bpm,
                  "anchor": s["entry"].get("pin_type") == "anchor",
                  "seam": s.get("seam_info"),
                  "warn": bool(s["warnings"])}
                 for s in result["slots"]]
        arc = [(i / 24.0, max(0.0, min(1.0, theme.arc_target(i / 24.0))))
               for i in range(25)]
        self.arc_strip.set_data(strip, arc)

    def _report_card(self, result):
        """One line the whole set can be judged by."""
        slots = result["slots"]
        seams = [s for s in slots[:-1] if s["transition"]]
        n_bm = sum(1 for s in seams
                   if s["transition"]["style"] != "long_fade")
        total = max(result["total_s"], 60.0)
        theme = self.theme()
        fits = [1.0 - abs(s["track"].energy_proxy()
                          - theme.arc_target(min(
                              s["start_offset_s"] / total, 1.0)))
                for s in slots]
        drifts = [abs(s["start_offset_s"]
                      - float(s["entry"]["target_offset_min"]) * 60.0)
                  for s in slots
                  if s["entry"].get("pin_type") == "anchor"
                  and s["entry"].get("target_offset_min")]
        bits = [f"~{result['total_s'] / 3600.0:.1f} h",
                f"{len(slots)} tracks"]
        if seams:
            scores = sorted(s["transition"]["pair_score"] for s in seams)
            bits.append(f"{n_bm}/{len(seams)} beat-matched "
                        f"({n_bm / len(seams):.0%})")
            bits.append(f"median seam {scores[len(scores) // 2]:.2f}")
        if fits:
            bits.append(f"arc fit {sum(fits) / len(fits):.2f}")
        if drifts:
            bits.append(f"anchor drift ≤ {max(drifts) / 60.0:.1f} min")
        if result["warnings"]:
            bits.append(f"{len(result['warnings'])} warnings")
        return "  ·  ".join(bits)

    def _strip_clicked(self, i):
        if 0 <= i < self.set_list.count():
            self.set_list.setCurrentRow(i)
            self.select_seam(i)

    def _copilot_applied(self, entries, theme_name):
        """Adopt the copilot's (or a revert's) entries + theme. Runs on the
        GUI thread via the panel's signal."""
        if theme_name in BUILTIN_THEMES:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentText(theme_name)
            self.theme_combo.blockSignals(False)
        self.entries = [dict(e) for e in entries]
        self._rebuild()
        self.recompile()

    # -- repair: alternatives + bridge -----------------------------------------
    def _slot_alternatives(self, i, n=8):
        """Top replacement candidates for slot i: geometric mean of the
        seam INTO the slot and the seam OUT of it, at that moment's arc -
        the same scoring the live DJ picks with (pair memory included)."""
        if not (0 <= i < len(self.entries)):
            return []
        by_id = {t.id: t for t in self.planner.library}
        cur_ids = {e["track_id"] for e in self.entries}
        prev = by_id.get(self.entries[i - 1]["track_id"]) if i > 0 else None
        nxt = by_id.get(self.entries[i + 1]["track_id"]) \
            if i + 1 < len(self.entries) else None
        brain = Brain(self.planner.library, self.theme())
        brain.pair_memory = dict(
            getattr(self.planner, "pair_memory", {}) or {})
        arc = 0.6
        if self.compiled and i < len(self.compiled["slots"]):
            total = max(self.compiled["total_s"], 60.0)
            arc = self.theme().arc_target(min(
                self.compiled["slots"][i]["start_offset_s"] / total, 1.0))
        scored = []
        for t in self.planner.library:
            if t.id in cur_ids:
                continue
            # Cheap tempo gate before the expensive pair scoring.
            if prev is not None and brain.rate_for(prev.bpm, t)[0] is None:
                continue
            if nxt is not None and brain.rate_for(t.bpm, nxt)[0] is None:
                continue
            s_in = (brain.score(prev, t, arc, prev.bpm)[0]
                    if prev is not None
                    else max(1e-6, 1.0 - abs(t.energy_proxy() - arc)))
            s_out = (brain.score(t, nxt, arc, t.bpm)[0]
                     if nxt is not None else 1.0)
            v = (max(s_in, 1e-9) * max(s_out, 1e-9)) ** 0.5
            scored.append((v, t))
        scored.sort(key=lambda p: -p[0])
        return scored[:n]

    def _suggest_alternatives(self):
        i = self.set_list.currentRow()
        if i < 0:
            return
        if self.entries[i].get("pin_type") == "anchor":
            self.status.setText("that's an anchor - unpin it first if you"
                                " want the machine's opinion")
            return
        self.status.setText("scoring alternatives...")
        QApplication.processEvents()
        alts = self._slot_alternatives(i)
        if not alts:
            self.status.setText("no tempo-reachable alternative for this"
                                " slot")
            return
        menu = QMenu(self)
        for v, t in alts:
            act = menu.addAction(f"{t.title[:44]}  ({t.bpm:.0f} "
                                 f"{t.camelot})  fit {v:.3f}")
            act.setData(t.id)
        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            self.status.setText("")
            return
        self.entries[i]["track_id"] = chosen.data()
        self._rebuild()
        self.recompile()

    def _insert_bridge(self):
        i = self.set_list.currentRow()
        if not (0 <= i < len(self.entries) - 1):
            self.status.setText("select the entry BEFORE the seam you want"
                                " bridged")
            return
        by_id = {t.id: t for t in self.planner.library}
        a = by_id.get(self.entries[i]["track_id"])
        b = by_id.get(self.entries[i + 1]["track_id"])
        if a is None or b is None:
            return
        self.status.setText("searching for a bridge...")
        QApplication.processEvents()
        chain, score = SL.bridge(
            self.planner.library, a, b, self.theme(),
            exclude_ids={e["track_id"] for e in self.entries})
        if not chain:
            self.status.setText(f"no bridge beats the direct seam "
                                f"{a.title[:20]} -> {b.title[:20]}")
            return
        for k, t in enumerate(chain):
            self.entries.insert(i + 1 + k, {
                "track_id": t.id, "pin_type": "suggestion",
                "target_offset_min": None, "style_override": None,
                "target_play_s": None})
        self._rebuild()
        self.recompile()
        self.status.setText(
            "bridged via " + " -> ".join(t.title[:24] for t in chain)
            + f"  (bottleneck {score:.3f})")

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
        self.planner.claim_playback("seam")
        self.seam_player.load(result)
        self.seam_player.play()


# ==========================================================================
# Mix tab
# ==========================================================================

class MixTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.preview = PlanPreview(planner.db)
        v = QVBoxLayout(self)
        self.timeline = MixTimeline()
        self.timeline.seamSelected.connect(self._seam_picked)
        self.timeline.timeClicked.connect(self._time_clicked)
        v.addWidget(self.timeline, 2)

        # LIVE deck monitor: scrolling beat waveforms + band meters + beat
        # offset, from real playback telemetry (lag-compensated).
        self.deckmon = DeckMonitor()
        v.addWidget(self.deckmon, 1)

        def chip(c, label):
            return (f"<span style='color:{c}'>&#9632;</span>"
                    f"<span style='color:#c8c8d0'> {label}&nbsp;&nbsp;</span>")
        legend = QLabel(
            chip("#fac85a", "tempo strip (planned bpm; live bpm at playhead)")
            + chip("#2d4b69", "track block (lane A)")
            + chip("#325f4b", "track block (lane B)")
            + chip("#e6e6f0", "energy curve / GAIN envelope")
            + chip("#eb695a", "low band strip + EQ env")
            + chip("#78d278", "mid") + chip("#6eaaf0", "high")
            + "<span style='color:#8a8a95'>| ticks = beats (bright = "
              "downbeat, appear when zoomed) | ↳ style (blend length) at "
              "each seam | wheel = zoom, drag = pan, click = seek when "
              "playing / select seam</span>")
        legend.setWordWrap(True)
        v.addWidget(legend)

        # Now-playing / up-next readout (mirrors the live web DJ tab).
        self.nowbar = QLabel("Build a set (Set tab), then Play.")
        self.nowbar.setWordWrap(True)
        self.nowbar.setStyleSheet(
            "background:#1c1c24;border:1px solid #33333d;border-radius:6px;"
            "padding:8px;font-size:13px;")
        v.addWidget(self.nowbar)

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
        self._fast = QTimer(self)
        self._fast.timeout.connect(self._fast_tick)
        self._fast.start(40)                    # 25 fps live monitor

    def _fast_tick(self):
        if self.preview.playing:
            self.deckmon.update()

    def set_plan(self, compiled):
        brain = Brain(self.planner.library,
                      self.planner.set_tab.theme())
        self.timeline.set_plan(compiled, brain)
        # A changed plan invalidates the running script - stop, re-arm the
        # preview (kicks the background pre-decode for instant seeks).
        self.preview.stop()
        self.timeline.set_playhead(None)
        self.preview.set_plan(compiled, brain)
        self.timeline.samples_provider = self.preview.cached_samples
        self.deckmon.attach(self.preview, compiled)

    def play_set(self):
        if not self.planner.set_tab.compiled:
            self.status.setText("build a set first")
            return
        self.pause_btn.setChecked(False)
        self.planner.claim_playback("preview")
        self.preview.play_at(0.0)

    def stop(self):
        self.planner.stop_all_playback()
        self.timeline.set_playhead(None)
        self.deckmon.update()
        self.status.setText("stopped")

    def _pause(self, on):
        self.preview.pause(on)

    def _jump(self, d):
        self.preview.jump_slot(d)

    def _next_seam(self):
        self.preview.next_seam()

    def _seam_picked(self, index):
        self.planner.set_tab.select_seam(index)

    def _time_clicked(self, t):
        self.planner.claim_playback("preview")
        """Standard DJ-software click-around: playing or stopped, a click
        on the timeline plays from there (instant via the decode cache)."""
        if not self.planner.set_tab.compiled:
            return
        self.pause_btn.setChecked(False)
        self.preview.play_at(t)
        self.timeline.set_playhead(t)

    def _tick(self):
        pv = self.preview
        compiled = self.planner.set_tab.compiled
        if pv.error:
            self.status.setText("preview error: " + pv.error)
            pv.error = ""
            return
        if not pv.playing:
            if pv.decoding:
                self.status.setText(f"pre-decoding: {pv.decoding[:40]}...")
                self.nowbar.setText(f"pre-decoding {pv.decoding[:40]}… "
                                    "(instant seeks once warm)")
            elif compiled:
                self.nowbar.setText(
                    f"Ready · {len(compiled['slots'])} tracks · "
                    f"~{compiled['total_s']/3600:.1f} h.  Play, or click the "
                    "timeline to start anywhere.")
            return
        t = pv.playhead()
        if t is not None:
            self.timeline.set_playhead(t)
        live_bpm = None
        if compiled and compiled["slots"]:
            slot_i = pv.slot_at_playhead()
            slot = compiled["slots"][slot_i]
            tr = slot["track"]
            # Live tracked tempo: the loudest playing deck's track bpm x rate.
            tel = pv.telemetry() or {}
            by_id = {s["track"].id: s["track"].bpm
                     for s in compiled["slots"]}
            best_gain = -1.0
            for d in (tel.get("decks") or {}).values():
                if d.get("playing") and d.get("gain", 0) > best_gain \
                        and d.get("track_id") in by_id:
                    best_gain = d["gain"]
                    live_bpm = by_id[d["track_id"]] * d["rate"]
            self.timeline.live_bpm = live_bpm
            pos_in = t - slot["start_offset_s"] if t is not None else 0.0
            nxt = (compiled["slots"][slot_i + 1]
                   if slot_i + 1 < len(compiled["slots"]) else None)
            tech = slot["transition"]["style"].replace("_", " ") \
                if slot["transition"] else "—"
            parts = [f"<b>NOW</b> {tr.title[:40]}",
                     f"{tr.bpm:.0f} bpm {tr.camelot}"]
            if live_bpm:
                parts.append(f"playing @ <b>{live_bpm:.1f} bpm</b>")
            parts.append(f"{int(pos_in//60)}:{int(pos_in%60):02d} in")
            line = "  ·  ".join(parts)
            if nxt:
                line += (f"<br><b>NEXT</b> {nxt['track'].title[:40]}  ·  "
                         f"{nxt['track'].bpm:.0f} bpm {nxt['track'].camelot}"
                         f"  ·  via <b>{tech}</b>")
            self.nowbar.setText(line)
            self.status.setText(
                (f"{live_bpm:.1f} bpm | " if live_bpm else "") + tr.title[:30])

    def close(self):
        self.preview.close()


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
        self.library = []            # SELECTABLE tracks (excluded removed)
        self.library_all = []        # everything, for the library browser

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
        # Discover (Beatport) is optional - only if the module imports.
        self.discover_tab = None
        try:
            from tools.djplanner.discover import DiscoverTab
            self.discover_tab = DiscoverTab(self)
            self.tabs.addTab(self.discover_tab, "Discover")
        except Exception as e:
            print(f"[planner] Discover tab unavailable: {e}")

        self.library_tab.openAnalysis.connect(self._open_analysis)
        self.library_tab.addTracks.connect(self.set_tab.add_tracks)
        self.set_tab.planCompiled.connect(self.mix_tab.set_plan)

        self.reload_library()
        self.set_tab.refresh_setlists()

    # ---- playback arbiter: ONE thing plays at a time --------------------
    def claim_playback(self, owner):
        """Starting ANY playback silences every other player first."""
        if getattr(self, "_pb_owner", None) not in (None, owner):
            self._stop_owner(self._pb_owner)
        self._pb_owner = owner

    def _stop_owner(self, owner):
        try:
            if owner == "analysis":
                if self.analysis_tab.player.playing:
                    self.analysis_tab._toggle_play()
            elif owner == "library":
                self.library_tab.stop_playback()
            elif owner == "seam":
                self.set_tab.seam_player.close()
            elif owner == "preview":
                self.mix_tab.preview.stop()
            elif owner == "discover":
                if self.discover_tab is not None:
                    self.discover_tab.stop_preview()
        except Exception:
            pass

    def stop_all_playback(self):
        """ANY stop button stops ANY playing."""
        for o in ("analysis", "library", "seam", "preview", "discover"):
            self._stop_owner(o)
        self._pb_owner = None

    def reload_library(self, keep_analysis=False):
        # library_all = everything (the browser shows + toggles do-not-use);
        # library = what every auto-selector sees (excluded removed), so set
        # generation, the copilot, and the live autoDJ never grab a flagged
        # track.
        self.library_all = load_library(self.db)
        self.library = [t for t in self.library_all if not t.excluded]
        # Cross-night pair memory (thumbs + auto seam assessments) is
        # display/steering input for the Set tab; loaded HERE on the UI
        # thread so the compile worker never touches the shared sqlite
        # connection.
        try:
            _b = Brain([], get_theme("groove"))
            _b.load_pair_memory(self.db)
            self.pair_memory = _b.pair_memory
        except Exception:
            self.pair_memory = {}
        self.library_tab.refresh()
        if not keep_analysis:
            self.analysis_tab.refresh_tracklist()
        # ANY library change (tags added/removed, cues marked) makes a
        # compiled plan stale - tags steer selection and user cues
        # OVERRIDE seam points, so the Set/Mix tabs must recompile.
        if self.set_tab.entries:
            self.set_tab.recompile()

    def _open_analysis(self, track):
        self.tabs.setCurrentWidget(self.analysis_tab)
        self.analysis_tab.open_track(track)

    def closeEvent(self, ev):
        self.analysis_tab.close()
        self.mix_tab.close()
        self.set_tab.seam_player.close()
        self.library_tab.lib_player.close()
        if self.library_tab._enrich is not None \
                and self.library_tab._enrich.isRunning():
            self.library_tab._enrich.stop()
            self.library_tab._enrich.wait(2000)
        if self.library_tab._mood_proc is not None:
            self.library_tab._mood_proc.kill()
            self.library_tab._mood_proc.waitForFinished(2000)
        if self.discover_tab is not None:
            self.discover_tab.close()
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
