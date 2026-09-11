"""Stage tab: the stem stage in the planner, played live.

Up to four LANES, each one stem of one track looping a bar-aligned section
on its own deck, all on one clock (lib/dj/stage.py). You choose the track,
the stem, the section, the loop length and the level; the stage does the
rest - the next-bar quantisation, the stretch to the master tempo, the key
shift toward the master key with the harmonic guard, the PLL that holds
every lane on the master's kicks, the hand-over of the clock when the
master leaves. IN and OUT land on the next bar; the button shows "armed"
until they do.

The audio runs through the same AudioEngine + DJSubmix a night runs, in real
time on the default device (Start stage). Nothing is rendered ahead: what
you hear is the engine.
"""
import threading

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSlider, QVBoxLayout, QWidget)

from lib.dj.stage import STEMS

BARS = ("4", "8", "16", "section")


class _Loader(QObject):
    done = pyqtSignal(int, object)      # lane index, None | error str

    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def load(self, i, track):
        def work():
            try:
                self.stage.load(i, track)
                self.done.emit(i, None)
            except Exception as e:  # noqa: BLE001
                self.done.emit(i, f"{type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()


class LaneRow:
    """The widgets of one lane."""

    def __init__(self, tab, i, grid, row0):
        self.tab, self.i = tab, i
        self.search = QLineEdit()
        self.search.setPlaceholderText("track (title / artist)")
        self.search.setMaximumWidth(220)
        self.track_box = QComboBox()
        self.track_box.setMinimumWidth(260)
        self.stem_box = QComboBox()
        self.stem_box.addItems(STEMS)
        self.stem_box.setCurrentText(("drums", "bass", "other", "vocals")[i % 4])
        self.section_box = QComboBox()
        self.section_box.setMinimumWidth(180)
        self.bars_box = QComboBox()
        self.bars_box.addItems(BARS)
        self.bars_box.setCurrentText("8")
        self.gain = QSlider(Qt.Orientation.Horizontal)
        self.gain.setRange(0, 120)
        self.gain.setValue(100)
        self.gain.setMaximumWidth(120)
        self.gain.setToolTip("lane level (100 = the track's own level after loudness matching)")
        self.load_btn = QPushButton("Load")
        self.io_btn = QPushButton("IN")
        self.io_btn.setEnabled(False)
        self.io_btn.setMinimumWidth(70)
        self.allow = QCheckBox("allow clash")
        self.allow.setToolTip("let a melodic stem in although no key shift within 3 semitones makes it fit the master key")
        self.status = QLabel("empty")
        self.status.setMinimumWidth(320)
        r = row0
        grid.addWidget(QLabel(f"lane {i + 1}"), r, 0)
        grid.addWidget(self.search, r, 1)
        grid.addWidget(self.track_box, r, 2)
        grid.addWidget(self.load_btn, r, 3)
        grid.addWidget(self.stem_box, r, 4)
        grid.addWidget(self.section_box, r, 5)
        grid.addWidget(self.bars_box, r, 6)
        grid.addWidget(self.gain, r, 7)
        grid.addWidget(self.io_btn, r, 8)
        grid.addWidget(self.allow, r, 9)
        grid.addWidget(self.status, r + 1, 1, 1, 9)
        self.search.textChanged.connect(self._filter)
        self.track_box.currentIndexChanged.connect(self._track_changed)
        self.load_btn.clicked.connect(lambda: tab._load(i))
        self.io_btn.clicked.connect(lambda: tab._io(i))
        self.gain.valueChanged.connect(lambda v: tab._gain(i, v / 100.0))
        self.bars_box.currentTextChanged.connect(lambda _t: tab._bars(i))
        self._filter("")

    def _filter(self, text):
        text = (text or "").lower()
        self.track_box.blockSignals(True)
        self.track_box.clear()
        n = 0
        for t in self.tab.tracks:
            label = f"{t.title} - {t.artist or ''}  ({t.bpm:.0f} bpm, {t.camelot or '?'})"
            if text and text not in label.lower():
                continue
            self.track_box.addItem(label, t.id)
            n += 1
        self.track_box.blockSignals(False)
        self._track_changed()

    def _track_changed(self):
        tid = self.track_box.currentData()
        t = self.tab.by_id.get(tid)
        self.section_box.clear()
        if t is None:
            return
        for k, s in enumerate(t.sections or []):
            self.section_box.addItem(f"{k + 1}. {s.get('kind', '?')} {int(s['start_s']) // 60}:{int(s['start_s']) % 60:02d} "
                                     f"(bass {s.get('bass_share', 0):.2f}, energy {s.get('energy', 0):.2f})", k)
        # default to the first groove with bass, the stage's body rule
        for k, s in enumerate(t.sections or []):
            if s.get("kind") == "groove" and s.get("bass_share", 0.3) >= 0.28:
                self.section_box.setCurrentIndex(k)
                break

    def track(self):
        return self.tab.by_id.get(self.track_box.currentData())


class StageTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.engine = None
        self.stage = None
        self.tracks = sorted([t for t in (planner.library or []) if getattr(t, "has_stems", False)],
                             key=lambda t: (t.title or "").lower())
        self.by_id = {t.id: t for t in self.tracks}
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        self.start_btn = QPushButton("▶ Start stage")
        self.start_btn.setToolTip("open the audio device and run the stage engine (the same engine a night runs)")
        self.start_btn.clicked.connect(self._toggle_engine)
        top.addWidget(self.start_btn)
        self.master_lbl = QLabel("clock: none - the first lane in sets the tempo and the key")
        top.addWidget(self.master_lbl, 1)
        self.beat_lbl = QLabel("")
        self.beat_lbl.setMinimumWidth(120)
        top.addWidget(self.beat_lbl)
        v.addLayout(top)
        grid = QGridLayout()
        self.rows = [LaneRow(self, i, grid, 2 * i) for i in range(4)]
        v.addLayout(grid)
        self.log_lbl = QLabel("Type part of a title to find a track, Load it on a lane, pick its stem and section, IN "
                              "(the first IN starts the stage). The first lane in is the clock; the others stretch to it, shift "
                              "toward its key, and lock on its kicks. IN/OUT land on the next bar.")
        self.log_lbl.setWordWrap(True)
        v.addWidget(self.log_lbl)
        v.addStretch(1)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._loader = None
        self._new_stage()

    # -- engine ------------------------------------------------------------------
    def _new_stage(self):
        from lib.dj.stage import Stage
        self.stage = Stage(self.planner.db, self.planner.music_dir, n_lanes=4)
        self._loader = _Loader(self.stage)
        self._loader.done.connect(self._loaded)

    def _toggle_engine(self):
        if self.engine is None:
            from lib.audio_engine import AudioEngine
            try:
                self.planner.analysis_tab.player.stop()
            except Exception:
                pass
            self.engine = AudioEngine()
            self.engine.attach_track("stage", self.stage.submix)
            self.engine.start()
            self.start_btn.setText("■ Stop stage")
            self._timer.start()
            self.log_lbl.setText("stage running")
        else:
            self._timer.stop()
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None
            self.start_btn.setText("▶ Start stage")
            self._new_stage()
            for r in self.rows:
                r.status.setText("empty")
                r.io_btn.setText("IN")
                r.io_btn.setEnabled(False)
            self.master_lbl.setText("clock: none - the first lane in sets the tempo and the key")
            self.log_lbl.setText("stage stopped")

    def close(self):
        if self.engine is not None:
            self._timer.stop()
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None

    # -- lane actions --------------------------------------------------------------
    def _load(self, i):
        t = self.rows[i].track()
        if t is None or self.stage is None:
            return
        lane = self.stage.lanes[i]
        if lane.state in ("live", "armed"):
            self.stage.release(i)
        self.rows[i].status.setText(f"loading {t.title}...")
        self.rows[i].load_btn.setEnabled(False)
        self._loader.load(i, t)

    def _loaded(self, i, err):
        row = self.rows[i]
        row.load_btn.setEnabled(True)
        if err:
            row.status.setText("load failed: " + err)
            return
        lane = self.stage.lanes[i]
        row.status.setText(f"{lane.track.title} loaded ({lane.track.bpm:.1f} bpm, {lane.track.camelot or '?'}) - choose the stem and section, then IN")
        row.io_btn.setEnabled(True)
        row.io_btn.setText("IN")

    def _io(self, i):
        if self.stage is None:
            return
        if self.engine is None:
            self._toggle_engine()            # the first IN starts the stage; nothing plays without it
        lane = self.stage.lanes[i]
        row = self.rows[i]
        if lane.state in ("live", "armed"):
            self.stage.release(i)
            row.io_btn.setText("leaving...")
            return
        bars = row.bars_box.currentText()
        at = self.stage.arm(i, row.stem_box.currentText(), int(row.section_box.currentData() or 0),
                            loop_bars=None if bars == "section" else int(bars),
                            gain=row.gain.value() / 100.0, allow_clash=row.allow.isChecked())
        if at is None:
            row.status.setText(self.stage.log[-1] if self.stage.log else "refused")
            return
        row.io_btn.setText("armed...")

    def _gain(self, i, g):
        if self.stage is not None:
            self.stage.set_gain(i, g)

    def _bars(self, i):
        if self.stage is None:
            return
        bars = self.rows[i].bars_box.currentText()
        if bars != "section":
            self.stage.set_loop_bars(i, int(bars))

    # -- the picture -----------------------------------------------------------------
    def _tick(self):
        if self.stage is None:
            return
        self.stage.refresh()
        st = self.stage.status()
        if st["master"]:
            self.master_lbl.setText(f"clock: {st['master']} at {st['master_bpm']:.1f} bpm, key {st['master_camelot'] or '?'}")
        else:
            self.master_lbl.setText("clock: none - the first lane in sets the tempo and the key")
        for i, (row, d) in enumerate(zip(self.rows, st["lanes"])):
            if d["state"] == "empty":
                continue
            state = d["state"]
            if state == "live":
                row.io_btn.setText("OUT")
            elif state == "loaded":
                row.io_btn.setText("IN")
            bits = [f"{d['title']}", f"{d['stem'] or '-'}", state]
            if d.get("loop"):
                bits.append(f"{d['loop_bars']} bars from {d['loop'][0]:.1f}s")
            if state in ("live", "armed", "leaving"):
                bits.append(f"rate {d['rate']:.3f}")
                if d.get("key_shift"):
                    bits.append(f"shift {d['key_shift']:+d} st")
                if d.get("compat") is not None:
                    bits.append(f"key fit {d['compat']:.2f}" + (" CLASH" if d.get("clash") else ""))
                if d.get("audible_err_beats") is not None:
                    bits.append(f"lock {1000 * abs(d['audible_err_beats']) * 60.0 / max(st['master_bpm'] or 120.0, 1):.0f} ms")
            if st["master"] == d["deck"]:
                bits.append("CLOCK")
            row.status.setText("  |  ".join(bits))
        m = next((d for d in st["lanes"] if d["deck"] == st["master"]), None)
        if m and m.get("beat_phase") is not None:
            ph = float(m["beat_phase"])
            self.beat_lbl.setText("●" if ph < 0.15 else "○")
        else:
            self.beat_lbl.setText("")
        if self.stage.log:
            self.log_lbl.setText(self.stage.log[-1])
