"""DJ Set Planner - native PyQt6 desktop app (planning/analysis tools are
PyQt6 by house rule; the web panel is only for live show control).

Build a set in the afternoon, audition every seam, play it at night:

  * Library table over the scanned DB with a per-track STRUCTURE STRIP
    (sections colored by kind + energy curve) so you can see each song's
    shape at a glance; search/sort.
  * Set editor: add tracks, reorder, toggle anchor/suggestion, pin an
    anchor to a time offset; Auto-fill inserts brain-chosen suggestions
    between anchors.
  * The compiler (lib/dj/setlist.py - the SAME Brain the live DJ runs)
    resolves the set: timeline with clock estimates, per-seam cards
    (style, stretch rate, key move, pair score) and warnings.
  * AUDITION any seam: renders the ~50s around it through the real engine
    (offline, faster than realtime) and plays it back in-app.

Usage: python tools/dj_planner.py [--dir <music_dir>]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QRect, \
    QSortFilterProxyModel, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPalette, QPainter, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QLineEdit, QTableView, QListWidget, QListWidgetItem, QPushButton,
    QComboBox, QStyledItemDelegate, QSplitter, QMessageBox, QInputDialog,
    QAbstractItemView, QDoubleSpinBox, QStyle)

from lib.dj import resolve_music_dir
from lib.dj.db import LibraryDB
from lib.dj.brain import load_library
from lib.dj.themes import BUILTIN_THEMES, get_theme
from lib.dj import setlist as SL

RATE = 44100

SECTION_COLORS = {
    "intro": QColor(70, 90, 120), "outro": QColor(70, 90, 120),
    "steady": QColor(60, 120, 90), "drop": QColor(190, 80, 60),
    "build": QColor(190, 150, 60), "breakdown": QColor(90, 70, 130),
}
COLS = ["title", "artist", "bpm", "key", "dur", "structure"]


class LibraryModel(QAbstractTableModel):
    def __init__(self, tracks, db):
        super().__init__()
        self.tracks = tracks
        self.strips = {}                 # track id -> (sections, curve)
        for t in tracks:
            self.strips[t.id] = (t.sections, t.row.get("energy_curve") or [])

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
        if role == Qt.ItemDataRole.UserRole:
            return t
        return None


class StripDelegate(QStyledItemDelegate):
    """Paints the structure strip: section blocks colored by kind with the
    energy curve drawn over them - the song's shape at a glance."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def paint(self, p, option, index):
        t = index.model().data(index, Qt.ItemDataRole.UserRole)
        if t is None:
            return
        if option.state & QStyle.StateFlag.State_Selected:
            p.fillRect(option.rect, option.palette.highlight())
        secs, curve = t.sections, t.row.get("energy_curve") or []
        r = option.rect.adjusted(2, 4, -2, -4)
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        dur = max(t.duration_s, 1.0)
        for s in secs:
            x0 = r.x() + r.width() * s["start_s"] / dur
            x1 = r.x() + r.width() * s["end_s"] / dur
            col = SECTION_COLORS.get(s["kind"], QColor(80, 80, 80))
            p.fillRect(QRect(int(x0), r.y(), max(int(x1 - x0), 1),
                             r.height()), col)
        if curve:
            p.setPen(QColor(240, 240, 240, 200))
            n = len(curve)
            pts = []
            for i, v in enumerate(curve):
                x = r.x() + r.width() * i / max(n - 1, 1)
                y = r.y() + r.height() * (1.0 - min(float(v), 1.0))
                pts.append((x, y))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        p.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        s.setWidth(260)
        return s


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
            self.done.emit({"error": f"{type(e).__name__}: {e}"})


class AuditionWorker(QThread):
    done = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, db, slot_a, slot_b, plan):
        super().__init__()
        self.db, self.a, self.b, self.plan = db, slot_a, slot_b, plan

    def run(self):
        try:
            from lib.audio_engine import AudioEngine
            from lib.dj.submix import DJSubmix
            from lib.dj.brain import Brain
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
            self.status.emit("rendering seam...")
            brain = Brain([], get_theme("groove"))
            events, swap_at, blend_at = brain.build_events(
                plan, sub.telemetry, "a", "b", a, b)
            sub.post_many(events)
            total = pre + (swap_at - blend_at) / RATE + 25.0
            out = [np.frombuffer(gen.send(4410), dtype=np.float32)
                   .reshape(-1, 2) for _ in range(int(total * RATE) // 4410)]
            self.done.emit(np.concatenate(out, axis=0))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done.emit(f"{type(e).__name__}: {e}")


class Player:
    """In-app playback of a rendered seam via miniaudio."""

    def __init__(self):
        self.dev = None
        self.pos = 0
        self.mix = None

    def play(self, mix):
        import miniaudio
        self.stop()
        self.mix = mix
        self.pos = 0

        player = self

        def gen():
            required = yield b""
            while player.pos < len(player.mix):
                chunk = player.mix[player.pos:player.pos + required]
                player.pos += required
                required = yield chunk.astype(np.float32).tobytes()
            while True:
                required = yield b"\x00" * (required * 8)
        g = gen()
        next(g)
        self.dev = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.FLOAT32, nchannels=2,
            sample_rate=RATE)
        self.dev.start(g)

    def stop(self):
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:
                pass
            self.dev = None


class Planner(QMainWindow):
    def __init__(self, music_dir):
        super().__init__()
        self.setWindowTitle(f"DJ Set Planner - {music_dir}")
        self.resize(1400, 860)
        self.db = LibraryDB(music_dir)
        self.library = load_library(self.db)
        self.entries = []                 # working set (entry dicts)
        self.compiled = None
        self.setlist_id = None
        self.player = Player()
        self._worker = None
        self._aud = None
        self._build_ui()
        self._refresh_setlists()
        self.recompile()

    # -- UI scaffolding ----------------------------------------------------
    def _build_ui(self):
        split = QSplitter()
        self.setCentralWidget(split)

        # LEFT: library
        left = QWidget()
        ll = QVBoxLayout(left)
        top = QHBoxLayout()
        self.search = QLineEdit(placeholderText="search title/artist...")
        self.search.textChanged.connect(self._filter)
        top.addWidget(QLabel(f"Library ({len(self.library)} tracks)"))
        top.addWidget(self.search)
        ll.addLayout(top)
        self.model = LibraryModel(self.library, self.db)
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterCaseSensitivity(
            Qt.CaseSensitivity.CaseInsensitive)
        self.proxy.setFilterKeyColumn(-1)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setItemDelegateForColumn(5, StripDelegate(self.model))
        self.table.setColumnWidth(0, 240)
        self.table.setColumnWidth(1, 130)
        self.table.setColumnWidth(2, 52)
        self.table.setColumnWidth(3, 44)
        self.table.setColumnWidth(4, 52)
        self.table.setColumnWidth(5, 280)
        self.table.doubleClicked.connect(lambda _: self.add_selected())
        ll.addWidget(self.table)
        add = QPushButton("Add to set  →")
        add.clicked.connect(self.add_selected)
        ll.addWidget(add)
        split.addWidget(left)

        # RIGHT: set editor + compiled plan
        right = QWidget()
        rl = QVBoxLayout(right)
        srow = QHBoxLayout()
        self.set_combo = QComboBox()
        self.set_combo.currentTextChanged.connect(self._load_selected_set)
        srow.addWidget(self.set_combo, 1)
        for label, fn in (("New", self.new_set), ("Save", self.save_set),
                          ("Delete", self.delete_set)):
            b = QPushButton(label)
            b.clicked.connect(fn)
            srow.addWidget(b)
        rl.addLayout(srow)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(BUILTIN_THEMES))
        self.theme_combo.setCurrentText("groove")
        self.theme_combo.currentTextChanged.connect(lambda _: self.recompile())
        trow.addWidget(self.theme_combo, 1)
        fill = QPushButton("Auto-fill between anchors")
        fill.clicked.connect(self.autofill)
        trow.addWidget(fill)
        rl.addLayout(trow)

        self.set_list = QListWidget()
        self.set_list.setDragDropMode(
            QAbstractItemView.DragDropMode.InternalMove)
        self.set_list.model().rowsMoved.connect(self._reordered)
        rl.addWidget(QLabel("Set (drag to reorder; double-click = "
                            "anchor/suggestion; Del = remove)"))
        rl.addWidget(self.set_list, 2)
        self.set_list.itemDoubleClicked.connect(self._toggle_anchor)
        rm = QAction("remove", self)
        rm.setShortcut("Delete")
        rm.triggered.connect(self._remove_entry)
        self.set_list.addAction(rm)
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Anchor time (min into set):"))
        self.anchor_min = QDoubleSpinBox(maximum=600.0, decimals=0)
        arow.addWidget(self.anchor_min)
        setb = QPushButton("Pin selected anchor")
        setb.clicked.connect(self._pin_anchor)
        arow.addWidget(setb)
        arow.addStretch(1)
        rl.addLayout(arow)

        rl.addWidget(QLabel("Compiled plan (select a seam, then Audition):"))
        self.plan_list = QListWidget()
        rl.addWidget(self.plan_list, 3)
        prow = QHBoxLayout()
        self.audition_btn = QPushButton("▶ Audition selected seam")
        self.audition_btn.clicked.connect(self.audition)
        prow.addWidget(self.audition_btn)
        stop = QPushButton("■ Stop")
        stop.clicked.connect(self.player.stop)
        prow.addWidget(stop)
        prow.addStretch(1)
        rl.addLayout(prow)
        self.status = QLabel("")
        rl.addWidget(self.status)
        split.addWidget(right)
        split.setSizes([760, 640])

    # -- library ---------------------------------------------------------------
    def _filter(self, text):
        self.proxy.setFilterFixedString(text)

    def _selected_track(self):
        idx = self.table.currentIndex()
        if not idx.isValid():
            return None
        return self.proxy.data(self.proxy.index(idx.row(), 0),
                               Qt.ItemDataRole.UserRole)

    # -- set editing --------------------------------------------------------------
    def add_selected(self):
        t = self._selected_track()
        if t is None:
            return
        self.entries.append({"track_id": t.id, "pin_type": "suggestion",
                             "target_offset_min": None,
                             "style_override": None})
        self._rebuild_set_list()
        self.recompile()

    def _entry_label(self, e):
        t = next((x for x in self.library if x.id == e["track_id"]), None)
        name = t.title if t else f"track {e['track_id']}"
        tag = "⚓ " if e["pin_type"] == "anchor" else "• "
        pin = (f"  @{e['target_offset_min']:.0f}min"
               if e.get("target_offset_min") else "")
        bpm = f"  ({t.bpm:.0f} bpm {t.camelot})" if t else ""
        return tag + name + bpm + pin

    def _rebuild_set_list(self):
        self.set_list.clear()
        for e in self.entries:
            QListWidgetItem(self._entry_label(e), self.set_list)

    def _reordered(self, *args):
        # Re-derive entry order from the list widget's current text order.
        order = [self.set_list.item(i).text()
                 for i in range(self.set_list.count())]
        labels = {self._entry_label(e): e for e in self.entries}
        self.entries = [labels[t] for t in order if t in labels]
        self.recompile()

    def _toggle_anchor(self, item):
        i = self.set_list.row(item)
        e = self.entries[i]
        e["pin_type"] = "anchor" if e["pin_type"] == "suggestion" \
            else "suggestion"
        if e["pin_type"] == "suggestion":
            e["target_offset_min"] = None
        self._rebuild_set_list()
        self.recompile()

    def _pin_anchor(self):
        i = self.set_list.currentRow()
        if i < 0:
            return
        e = self.entries[i]
        e["pin_type"] = "anchor"
        e["target_offset_min"] = self.anchor_min.value() or None
        self._rebuild_set_list()
        self.recompile()

    def _remove_entry(self):
        i = self.set_list.currentRow()
        if 0 <= i < len(self.entries):
            self.entries.pop(i)
            self._rebuild_set_list()
            self.recompile()

    def autofill(self):
        theme = get_theme(self.theme_combo.currentText())
        self.entries = SL.autofill(self.library, self.entries, theme)
        self._rebuild_set_list()
        self.recompile()

    # -- setlist persistence ---------------------------------------------------------
    def _refresh_setlists(self):
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        self.set_combo.addItem("(unsaved set)")
        for s in SL.list_setlists(self.db):
            self.set_combo.addItem(s["name"])
        self.set_combo.blockSignals(False)

    def _load_selected_set(self, name):
        if not name or name.startswith("("):
            return
        sl = SL.get_setlist(self.db, name=name)
        if sl is None:
            return
        self.setlist_id = sl["id"]
        self.entries = [{k: e.get(k) for k in
                         ("track_id", "pin_type", "target_offset_min",
                          "style_override")} for e in sl["entries"]]
        if sl.get("theme") in BUILTIN_THEMES:
            self.theme_combo.setCurrentText(sl["theme"])
        self._rebuild_set_list()
        self.recompile()

    def new_set(self):
        name, ok = QInputDialog.getText(self, "New setlist", "Name:")
        if not (ok and name):
            return
        try:
            self.setlist_id = SL.create_setlist(
                self.db, name, theme=self.theme_combo.currentText())
        except Exception as e:
            QMessageBox.warning(self, "New setlist", str(e))
            return
        self.entries = []
        self._rebuild_set_list()
        self._refresh_setlists()
        self.set_combo.setCurrentText(name)

    def save_set(self):
        if self.setlist_id is None:
            self.new_set()
            if self.setlist_id is None:
                return
        SL.save_entries(self.db, self.setlist_id, self.entries)
        self.db.conn.execute("UPDATE setlists SET theme = ? WHERE id = ?",
                             (self.theme_combo.currentText(),
                              self.setlist_id))
        self.db.conn.commit()
        self.status.setText("saved.")

    def delete_set(self):
        if self.setlist_id is None:
            return
        if QMessageBox.question(self, "Delete", "Delete this setlist?") \
                == QMessageBox.StandardButton.Yes:
            SL.delete_setlist(self.db, self.setlist_id)
            self.setlist_id = None
            self.entries = []
            self._rebuild_set_list()
            self._refresh_setlists()
            self.recompile()

    # -- compile + audition ------------------------------------------------------------
    def recompile(self):
        if self._worker is not None and self._worker.isRunning():
            return
        theme = get_theme(self.theme_combo.currentText())
        self._worker = CompileWorker(self.library, list(self.entries), theme)
        self._worker.done.connect(self._compiled)
        self._worker.start()

    def _compiled(self, result):
        if "error" in result:
            self.status.setText("compile error: " + result["error"])
            return
        self.compiled = result
        self.plan_list.clear()
        clock = 0.0
        for i, s in enumerate(result["slots"]):
            t = s["track"]
            mins = s["start_offset_s"] / 60.0
            head = (f"{int(mins):3d}:{int(s['start_offset_s'] % 60):02d}  "
                    f"{t.title}  ({t.bpm:.0f} bpm {t.camelot}, "
                    f"plays {s['play_s'] / 60.0:.1f} min)")
            QListWidgetItem(head, self.plan_list)
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
        total = result["total_s"]
        self.status.setText(
            f"set length ~{total / 3600.0:.1f} h "
            f"({len(result['slots'])} tracks); "
            + (f"{len(result['warnings'])} warnings" if result["warnings"]
               else "no warnings"))

    def audition(self):
        item = self.plan_list.currentItem()
        i = item.data(Qt.ItemDataRole.UserRole) if item else None
        if i is None or self.compiled is None:
            self.status.setText("select a seam row (↳) first")
            return
        slots = self.compiled["slots"]
        a, b = slots[i]["track"], slots[i + 1]["track"]
        plan = slots[i]["transition"]
        self.audition_btn.setEnabled(False)
        self._aud = AuditionWorker(self.db, a, b, plan)
        self._aud.status.connect(self.status.setText)
        self._aud.done.connect(self._audition_done)
        self._aud.start()

    def _audition_done(self, result):
        self.audition_btn.setEnabled(True)
        if isinstance(result, str):
            self.status.setText("audition failed: " + result)
            return
        self.status.setText(f"playing seam ({len(result) / RATE:.0f}s)...")
        self.player.play(result)

    def closeEvent(self, ev):
        self.player.stop()
        super().closeEvent(ev)


def main():
    music_dir = resolve_music_dir(
        sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "")
    if not os.path.isfile(os.path.join(music_dir, "dj_library.sqlite3")):
        print(f"No library DB in {music_dir} - run tools/dj_scan.py first.")
        return 1
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
    app.setPalette(pal)
    w = Planner(music_dir)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
