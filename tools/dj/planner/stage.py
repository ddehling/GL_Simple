"""Stage tab: two songs, four stem lanes, played live.

Song A is the clock and the key; song B runs beat-locked to it. Each stem
lane - drums, bass, other, vocals - is a three-way switch, A / B / off, that
flips on the next bar with a one-beat crossfade, so any mixture of the two
songs is a state you hold: A's drums under B's bass under A's vocal. MORPH
hands every lane over in the default order at the spacing you set; LOOP
holds a song on a few bars of where it is. The engine underneath is the
one a night runs (lib/dj/pairstage.py over the DJSubmix); the first Play
opens the audio device.
"""
import threading

from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QSlider, QVBoxLayout, QWidget)

from lib.dj.pairstage import STEMS

LOOPS = ("loop off", "4 bars", "8 bars", "16 bars")


class _Loader(QObject):
    done = pyqtSignal(str, object)          # deck, None | error str

    def __init__(self, stage):
        super().__init__()
        self.stage = stage

    def load(self, deck, track):
        def work():
            try:
                self.stage.load(deck, track)
                self.done.emit(deck, None)
            except Exception as e:  # noqa: BLE001
                self.done.emit(deck, f"{type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()


class SongPicker:
    """A search field + box over the library's stem-bearing tracks, read from the planner when used
    (the library loads after the tabs are built)."""

    def __init__(self, tab, deck, grid, row):
        self.tab, self.deck = tab, deck
        self.search = QLineEdit()
        self.search.setPlaceholderText("type part of a title or artist")
        self.search.setMinimumWidth(220)
        self.box = QComboBox()
        self.box.setMinimumWidth(320)
        self.load_btn = QPushButton(f"Load {deck.upper()}")
        self.status = QLabel("empty")
        self.status.setMinimumWidth(300)
        grid.addWidget(QLabel(f"song {deck.upper()}" + ("  (clock + key)" if deck == "a" else "")), row, 0)
        grid.addWidget(self.search, row, 1)
        grid.addWidget(self.box, row, 2)
        grid.addWidget(self.load_btn, row, 3)
        grid.addWidget(self.status, row, 4, 1, 3)
        self.search.textChanged.connect(self.refill)
        self.load_btn.clicked.connect(lambda: tab._load(deck))
        self.refill("")

    def refill(self, text=""):
        text = (text or "").lower()
        cur = self.box.currentData()
        self.box.blockSignals(True)
        self.box.clear()
        for t in self.tab.tracks():
            label = f"{t.title} - {t.artist or ''}  ({t.bpm:.0f} bpm, {t.camelot or '?'})"
            if text and text not in label.lower():
                continue
            self.box.addItem(label, t.id)
        if cur is not None and self.box.findData(cur) >= 0:
            self.box.setCurrentIndex(self.box.findData(cur))
        self.box.blockSignals(False)

    def track(self):
        return self.tab.by_id().get(self.box.currentData())


class StemRow:
    def __init__(self, tab, stem, grid, row):
        self.tab, self.stem = tab, stem
        self.group = QButtonGroup()
        self.btn = {}
        h = QHBoxLayout()
        for key, label in (("a", "A"), ("b", "B"), (None, "off")):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setMinimumWidth(56)
            b.clicked.connect(lambda _c, k=key: tab._switch(stem, k))
            self.group.addButton(b)
            self.btn[key] = b
            h.addWidget(b)
        self.btn[None].setChecked(True)
        w = QWidget()
        w.setLayout(h)
        self.gain = QSlider(Qt.Orientation.Horizontal)
        self.gain.setRange(0, 120)
        self.gain.setValue(100)
        self.gain.setMaximumWidth(140)
        self.gain.valueChanged.connect(lambda v: tab.stage.set_gain(stem, v / 100.0))
        self.status = QLabel("")
        grid.addWidget(QLabel(stem), row, 0)
        grid.addWidget(w, row, 1)
        grid.addWidget(self.gain, row, 2)
        grid.addWidget(self.status, row, 3, 1, 4)

    def show(self, where, pending=False):
        for k, b in self.btn.items():
            b.blockSignals(True)
            b.setChecked(k == where)
            b.blockSignals(False)


class StageTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.engine = None
        self._by_id = {}
        v = QVBoxLayout(self)
        g = QGridLayout()
        self.pick = {"a": SongPicker(self, "a", g, 0), "b": SongPicker(self, "b", g, 1)}
        v.addLayout(g)
        tr = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play A")
        self.play_btn.setToolTip("start both songs at their bodies: A as the clock, B beat-locked and silent until a lane switches to it")
        self.play_btn.clicked.connect(self._play)
        tr.addWidget(self.play_btn)
        self.morph_b = QPushButton("MORPH → B")
        self.morph_b.setToolTip("hand the lanes to B one by one: drums, bass, other, vocals")
        self.morph_b.clicked.connect(lambda: self._morph("b"))
        tr.addWidget(self.morph_b)
        self.morph_a = QPushButton("MORPH → A")
        self.morph_a.clicked.connect(lambda: self._morph("a"))
        tr.addWidget(self.morph_a)
        tr.addWidget(QLabel("every"))
        self.spacing = QComboBox()
        self.spacing.addItems(["4 beats", "8 beats", "16 beats"])
        self.spacing.setCurrentText("8 beats")
        tr.addWidget(self.spacing)
        tr.addWidget(QLabel("   loop A:"))
        self.loop_a = QComboBox(); self.loop_a.addItems(LOOPS)
        self.loop_a.currentTextChanged.connect(lambda t: self._loop("a", t))
        tr.addWidget(self.loop_a)
        tr.addWidget(QLabel("loop B:"))
        self.loop_b = QComboBox(); self.loop_b.addItems(LOOPS)
        self.loop_b.currentTextChanged.connect(lambda t: self._loop("b", t))
        tr.addWidget(self.loop_b)
        tr.addStretch(1)
        self.clock_lbl = QLabel("")
        tr.addWidget(self.clock_lbl)
        v.addLayout(tr)
        g2 = QGridLayout()
        self.rows = {s: StemRow(self, s, g2, k) for k, s in enumerate(STEMS)}
        v.addLayout(g2)
        self.log_lbl = QLabel("Load two songs, Play. Each lane's A / B / off switches on the next bar; MORPH hands them all over in order. "
                              "A is the clock and the key, B is stretched and key-shifted to it and held on A's kicks.")
        self.log_lbl.setWordWrap(True)
        v.addWidget(self.log_lbl)
        v.addStretch(1)
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._new_stage()

    # -- library (read when used: it loads after the tabs are built) --------------------------
    def tracks(self):
        lib = [t for t in (self.planner.library or []) if getattr(t, "has_stems", False)]
        lib.sort(key=lambda t: (t.title or "").lower())
        self._by_id = {t.id: t for t in lib}
        return lib

    def by_id(self):
        if not self._by_id:
            self.tracks()
        return self._by_id

    def showEvent(self, ev):
        super().showEvent(ev)
        for p in self.pick.values():
            if p.box.count() == 0:
                p.refill(p.search.text())

    # -- engine ---------------------------------------------------------------------------------
    def _new_stage(self):
        from lib.dj.pairstage import PairStage
        self.stage = PairStage(self.planner.db, self.planner.music_dir)
        self._loader = _Loader(self.stage)
        self._loader.done.connect(self._loaded)

    def _ensure_engine(self):
        if self.engine is not None:
            return
        from lib.audio_engine import AudioEngine
        try:
            self.planner.analysis_tab.player.stop()
        except Exception:
            pass
        self.engine = AudioEngine()
        self.engine.attach_track("stage", self.stage.submix)
        self.engine.start()
        self._timer.start()

    def close(self):
        if self.engine is not None:
            self._timer.stop()
            try:
                self.stage.stop()
                self.engine.stop()
            except Exception:
                pass
            self.engine = None

    # -- actions --------------------------------------------------------------------------------
    def _load(self, deck):
        p = self.pick[deck]
        t = p.track()
        if t is None:
            p.status.setText("no track chosen - type part of a title, then pick it in the box")
            return
        p.status.setText(f"loading {t.title}...")
        p.load_btn.setEnabled(False)
        self._loader.load(deck, t)

    def _loaded(self, deck, err):
        p = self.pick[deck]
        p.load_btn.setEnabled(True)
        if err:
            p.status.setText("load failed: " + err)
            return
        t = self.stage.track[deck]
        p.status.setText(f"{t.title} ({t.bpm:.1f} bpm, {t.camelot or '?'}) ready")

    def _play(self):
        if self.stage.playing:
            self.stage.stop()
            self.play_btn.setText("▶ Play A")
            for r in self.rows.values():
                r.show(None)
            return
        if self.stage.track["a"] is None:
            self.log_lbl.setText("load song A first")
            return
        self._ensure_engine()
        if self.stage.play("a"):
            self.play_btn.setText("■ Stop")
            for r in self.rows.values():
                r.show("a")

    def _switch(self, stem, where):
        if not self.stage.playing:
            self.rows[stem].show(self.stage.where.get(stem))
            self.log_lbl.setText("press Play first")
            return
        at = self.stage.set_stem(stem, where)
        if at is None:
            self.rows[stem].show(self.stage.where.get(stem))
            self.log_lbl.setText("song B is not loaded" if where == "b" else "refused")
            return
        self.rows[stem].status.setText("switching on the next bar...")

    def _morph(self, to):
        if not self.stage.playing:
            self.log_lbl.setText("press Play first")
            return
        beats = int(self.spacing.currentText().split()[0])
        if self.stage.morph(to, beats_apart=beats) is None:
            self.log_lbl.setText("song B is not loaded" if to == "b" else "refused")

    def _loop(self, deck, text):
        if not self.stage.playing:
            return
        bars = None if text.startswith("loop off") else int(text.split()[0])
        self.stage.loop(deck, bars)

    # -- the picture ------------------------------------------------------------------------------
    def _tick(self):
        st = self.stage.status()
        a, b = st["decks"]["a"], st["decks"]["b"]
        bits = []
        if a["playing"] and self.stage.track["a"] is not None:
            bits.append(f"A {self.stage.track['a'].bpm:.1f} bpm {self.stage.track['a'].camelot or ''}  at {a['time_s'] or 0:.0f}s")
        if b["playing"] and self.stage.track["b"] is not None:
            bits.append(f"B rate {st['rate_b']:.3f}, shift {st['shift_b']:+d} st, key fit {st['compat'] if st['compat'] is not None else '?'}")
            if st["lock_ms"] is not None:
                bits.append(f"lock {st['lock_ms']:.0f} ms")
        ph = a.get("beat_phase")
        if ph is not None and a["playing"]:
            bits.append("●" if float(ph) < 0.15 else "○")
        self.clock_lbl.setText("   ".join(bits))
        for s, r in self.rows.items():
            w = st["where"].get(s)
            r.show(w)
            src = self.stage.track[w] if w else None
            r.status.setText(f"{src.title}" if src else "off")
        if self.stage.log:
            self.log_lbl.setText(self.stage.log[-1])
