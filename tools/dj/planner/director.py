"""Director tab: high-level behaviour over the DJ, for parties and events.

Left, in words: what is playing and how far in, what comes next and when, what the Director intends for
the next few minutes, and the last things that happened. Right, the six dials - songs (theme, playlist,
up next), mixing, layers, energy, pace, loops - and the four moments: NEXT, HOLD, DROP, BREAK, with a
thumbs up or down on whatever just happened. No lanes, no clips, no stems, no levels: the engines
(lib/dj/system.py, lib/dj/remix.py) do that under lib/dj/director.py.
"""
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QPushButton, QSplitter, QVBoxLayout, QWidget, QSizePolicy)

from tools.dj.planner.perform import STYLE, SetlistBox, _mmss

DIAL_LABELS = {
    "mixing": ("MIXING", {"auto": "auto", "blend": "blend", "cut": "cut", "morph": "morph"},
               "how songs join: the dice · long blends · hard cuts · stem morphs (pins through the same gates; a refused pin falls back inside its family)"),
    "layers": ("LAYERS", {"one": "one song", "two": "two songs", "three": "three songs"},
               "one song at a time (the autoDJ as it is) · parts of two songs together · parts of three; the playing song is carried across"),
    "energy": ("ENERGY", {"cool": "cool down", "hold": "hold", "amp": "amp up"},
               "a lean on the night's arc; amped, the clock may also travel with it when layered"),
    "pace": ("PACE", {"short": "short bits", "normal": "normal", "long": "long plays"},
             "how long records play before the next seam; layered, how often the lanes move"),
    "loops": ("LOOPS", {"off": "off", "some": "some", "lots": "lots"},
              "how much the seams loop (loop styles' odds; layered, the clock holds four bars now and then)"),
}


class DirectorTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.setObjectName("perform")
        self.setStyleSheet(STYLE)
        self.planner = planner
        self.engine = None
        self.director = None
        self._own_db = None
        self._lib = None
        self._seen = 0
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)
        # -- top: start, theme, songs ------------------------------------------------------------------------
        top = QHBoxLayout()
        self.start_btn = QPushButton("▶  START")
        self.start_btn.setProperty("kind", "go")
        self.start_btn.setMinimumWidth(150)
        self.start_btn.setMinimumHeight(48)
        self.start_btn.clicked.connect(self._toggle)
        top.addWidget(self.start_btn)
        top.addSpacing(16)
        lab = QLabel("SONGS")
        lab.setProperty("dim", "true")
        top.addWidget(lab)
        self.theme_box = QComboBox()
        from lib.dj.themes import PICKER_THEMES
        self.theme_box.addItems(list(PICKER_THEMES))
        self.theme_box.setCurrentText("groove")
        self.theme_box.setToolTip("theme: what kind of songs, and the shape of the night")
        self.theme_box.currentTextChanged.connect(lambda n: self.director and self.director.set_theme(n))
        top.addWidget(self.theme_box)
        self.songs_box = SetlistBox(self)
        self.songs_box.setMinimumWidth(240)
        self.songs_box.setToolTip("a playlist (the Set tab's saved lists) as the pool of songs; whole library lifts it")
        self.songs_box.currentIndexChanged.connect(lambda _i: self.director and self.director.set_pool(self.songs_box.currentData()))
        top.addWidget(self.songs_box)
        self.state_lbl = QLabel("")
        self.state_lbl.setProperty("dim", "true")
        top.addWidget(self.state_lbl, 1)
        root.addLayout(top)

        # -- the picture: the run as a timeline (read-only) ----------------------------------------------------------
        from lib.dj.timeline import Timeline, Spectro
        from tools.dj.planner.timeline import TimelineCanvas
        self._empty_tl = Timeline("director")
        self.spectro = Spectro(planner.music_dir)
        self._colors = {}
        self.canvas = TimelineCanvas(self, lane_h=46, readonly=True)
        self.canvas.px_per_bar = 9.0
        self.canvas.setToolTip("what played on each lane and when · the playhead · dashed = what the Director plans next · Ctrl+wheel zooms, wheel scrolls")
        root.addWidget(self.canvas)

        split = QSplitter(Qt.Orientation.Horizontal)
        # -- left: what's happening, in words ----------------------------------------------------------------------
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 8, 0)
        lv.setSpacing(6)

        def big(size, dim=False, wrap=True):
            w = QLabel("")
            f = w.font()
            f.setPointSizeF(size)
            f.setBold(not dim)
            w.setFont(f)
            w.setWordWrap(wrap)
            if dim:
                w.setProperty("dim", "true")
            return w
        self.now_h = big(10, dim=True)
        self.now_h.setText("NOW")
        self.now_lbl = big(20)
        self.now_sub = big(12, dim=True)
        self.next_h = big(10, dim=True)
        self.next_h.setText("NEXT")
        self.next_lbl = big(16)
        self.next_sub = big(12, dim=True)
        self.intent_h = big(10, dim=True)
        self.intent_h.setText("THE DIRECTOR INTENDS")
        self.intent_lbl = big(13)
        self.lanes_lbl = big(12, dim=True)
        for w in (self.now_h, self.now_lbl, self.now_sub, self.next_h, self.next_lbl, self.next_sub, self.intent_h, self.intent_lbl, self.lanes_lbl):
            lv.addWidget(w)
        self.recent_h = big(10, dim=True)
        self.recent_h.setText("WHAT JUST HAPPENED")
        lv.addWidget(self.recent_h)
        self.recent = QListWidget()
        lv.addWidget(self.recent, 1)
        split.addWidget(left)

        # -- right: the dials and the moments ---------------------------------------------------------------------------
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        rv.setSpacing(10)
        self.dial_groups = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(10)
        from lib.dj.director import DIALS, DEFAULTS
        for row, (name, options) in enumerate(DIALS.items()):
            title, labels, tip = DIAL_LABELS[name]
            lab = QLabel(title)
            lab.setProperty("dim", "true")
            lab.setToolTip(tip)
            lab.setMinimumWidth(80)
            grid.addWidget(lab, row, 0)
            group = QButtonGroup(self)
            group.setExclusive(True)
            for col, opt in enumerate(options):
                b = QPushButton(labels[opt])
                b.setCheckable(True)
                b.setMinimumHeight(44)
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                b.setToolTip(tip)
                b.setChecked(opt == DEFAULTS[name])
                b.clicked.connect(lambda _c, n=name, o=opt: self._dial(n, o))
                group.addButton(b)
                grid.addWidget(b, row, 1 + col)
            self.dial_groups[name] = group
        rv.addLayout(grid)
        # up next
        un = QLabel("UP NEXT  (optional: the Director honours it when it can)")
        un.setProperty("dim", "true")
        rv.addWidget(un)
        qrow = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("search a song to queue")
        self.search.textChanged.connect(self._fill_search)
        qrow.addWidget(self.search, 1)
        rv.addLayout(qrow)
        lists = QHBoxLayout()
        self.found = QListWidget()
        self.found.setMaximumHeight(120)
        self.found.itemDoubleClicked.connect(lambda it: self._queue(it.data(Qt.ItemDataRole.UserRole)))
        self.found.setToolTip("double-click to queue")
        lists.addWidget(self.found, 1)
        self.queue = QListWidget()
        self.queue.setMaximumHeight(120)
        self.queue.itemDoubleClicked.connect(lambda it: self._unqueue(it.data(Qt.ItemDataRole.UserRole)))
        self.queue.setToolTip("up next, in order · double-click to remove")
        lists.addWidget(self.queue, 1)
        rv.addLayout(lists)
        # the moments
        mom = QGridLayout()
        mom.setSpacing(8)
        self.buttons = {}
        for (label, fn, tip, kind, r, c) in (
                ("NEXT", self._next, "move on now: the next seam / a move on the next bar", "go", 0, 0),
                ("HOLD", self._hold, "stay here a while: one more phrase / the lanes frozen a phrase", None, 0, 1),
                ("DROP", self._drop, "the drop moment / every lane to the newest song", "hot", 1, 0),
                ("BREAK", self._break, "layered only: every lane but one rests four bars", None, 1, 1)):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setMinimumHeight(56)
            if kind:
                b.setProperty("kind", kind)
            b.clicked.connect(fn)
            mom.addWidget(b, r, c)
            self.buttons[label] = b
        rv.addLayout(mom)
        verd = QHBoxLayout()
        for label, up, kind in (("👍 GOOD", True, "good"), ("👎 BAD", False, "bad")):
            b = QPushButton(label)
            b.setMinimumHeight(44)
            b.setProperty("kind", kind)
            b.clicked.connect(lambda _c, u=up: self.director and self.director.rate(u))
            verd.addWidget(b)
        rv.addLayout(verd)
        rv.addStretch(1)
        self.err_lbl = QLabel("")
        self.err_lbl.setProperty("dim", "true")
        self.err_lbl.setWordWrap(True)
        rv.addWidget(self.err_lbl)
        split.addWidget(right)
        split.setSizes([760, 640])
        root.addWidget(split, 1)
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._tick)

    # -- what the canvas asks of its tab ------------------------------------------------------------------------
    @property
    def timeline(self):
        return self.director.tl if self.director is not None else self._empty_tl

    def color(self, tid):
        from tools.dj.planner.perform import SONG_PALETTE
        if tid not in self._colors:
            self._colors[tid] = SONG_PALETTE[len(self._colors) % len(SONG_PALETTE)]
        return self._colors[tid]

    def track(self, tid):
        try:
            return next((t for t in self._library() if t.id == tid), None)
        except Exception:
            return None

    def bar_s(self, tid):
        t = self.track(tid)
        return 4 * t.period_s if t is not None else 2.0

    def playhead(self):
        return self.director.bar() if self.director is not None else 0.0

    def seek(self, bar):
        pass

    def place(self, *a, **k):
        pass

    def changed(self):
        self.canvas.update()

    # -- data -------------------------------------------------------------------------------------------------
    def _db(self):
        db = getattr(self.planner, "db", None)
        if db is not None:
            return db
        if self.director is not None:
            return self.director.db
        if self._own_db is None:
            from lib.dj.db import LibraryDB
            self._own_db = LibraryDB(self.planner.music_dir)
        return self._own_db

    def _library(self):
        if self.director is not None:
            return self.director.library
        lib = getattr(self.planner, "library", None)
        if lib:
            return lib
        if self._lib is None:
            from lib.dj import brain as B
            self._lib = [t for t in B.load_library(self._db()) if not t.excluded]
        return self._lib

    # -- controls ----------------------------------------------------------------------------------------------
    def _toggle(self):
        if self.director is None:
            try:
                self._start()
            except Exception as e:  # noqa: BLE001
                self.err_lbl.setText(f"could not start: {type(e).__name__}: {e}")
                self.close()
        else:
            self.close()

    def _start(self):
        from lib.audio_engine import AudioEngine
        from lib.dj.director import Director
        try:
            self.planner.analysis_tab.player.stop()
        except Exception:
            pass
        self.engine = AudioEngine()
        self.engine.start()
        self.director = Director(self.planner.music_dir, engine=self.engine, theme=self.theme_box.currentText())
        for name, group in self.dial_groups.items():
            for b in group.buttons():
                if b.isChecked():
                    from lib.dj.director import DIALS
                    label = b.text()
                    opt = next(o for o in DIALS[name] if DIAL_LABELS[name][1][o] == label)
                    self.director.dials[name] = opt
        if self.songs_box.currentData():
            self.director.pool_name = self.songs_box.currentData()
        if not self.director.start(threaded=True):
            self.err_lbl.setText(self.director.last_error or "could not start")
            self.close()
            return
        self.start_btn.setText("■  STOP")
        self.start_btn.setProperty("kind", "hot")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self._seen = 0
        self.recent.clear()
        self._timer.start()

    def close(self):
        self._timer.stop()
        if self.director is not None:
            try:
                self.director.stop(fade_s=1.5)
                time.sleep(1.7)
            except Exception:
                pass
            self.director = None
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None
        self.start_btn.setText("▶  START")
        self.start_btn.setProperty("kind", "go")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)

    def _dial(self, name, opt):
        if self.director is not None:
            self.director.set_dial(name, opt)

    def _next(self):
        if self.director is not None:
            self.director.next()

    def _hold(self):
        if self.director is not None:
            self.director.hold()

    def _drop(self):
        if self.director is not None:
            self.director.drop()

    def _break(self):
        if self.director is not None and not self.director.break_():
            self.err_lbl.setText("BREAK is a layered move: set LAYERS to two or three songs")

    def _fill_search(self, text):
        q = text.strip().lower()
        self.found.clear()
        if len(q) < 2:
            return
        try:
            lib = self._library()
        except Exception as e:  # noqa: BLE001
            self.err_lbl.setText(f"library: {e}")
            return
        for t in lib:
            if q in t.title.lower() or q in (t.artist or "").lower():
                it = QListWidgetItem(f"{t.title[:40]}  ·  {t.artist or ''}   {t.bpm:.0f} bpm {t.camelot or ''}")
                it.setData(Qt.ItemDataRole.UserRole, t.id)
                self.found.addItem(it)
                if self.found.count() >= 30:
                    break

    def _queue(self, tid):
        if self.director is not None:
            self.director.queue(tid)
        else:
            self.err_lbl.setText("press START first")
        self._render_queue()

    def _unqueue(self, tid):
        if self.director is not None:
            self.director.unqueue(tid)
        self._render_queue()

    def _render_queue(self):
        self.queue.clear()
        if self.director is None:
            return
        for tid in self.director.up_next:
            it = QListWidgetItem(self.director.title(tid))
            it.setData(Qt.ItemDataRole.UserRole, tid)
            self.queue.addItem(it)

    # -- readout -------------------------------------------------------------------------------------------------
    def _tick(self):
        try:
            self._tick_inner()
        except Exception as e:  # noqa: BLE001
            self.err_lbl.setText(f"readout error: {type(e).__name__}: {e}")

    def _tick_inner(self):
        d = self.director
        if d is None:
            return
        st = d.status()
        now, nxt = st.get("now"), st.get("next")
        mode = "one song at a time" if st.get("mode") == "one" else ("songs layered" if st.get("mode") else "-")
        self.state_lbl.setText(f"{mode}   ·   theme {st.get('theme')}   ·   songs: {st.get('pool') or 'whole library'}"
                               + (f"   ·   switching to {st['switching']}…" if st.get("switching") else ""))
        if now:
            self.now_lbl.setText(f"{now.get('title')}")
            self.now_sub.setText(f"{now.get('artist') or ''}   {_mmss(now.get('pos_s'))} / {_mmss(now.get('duration_s'))}   "
                                 f"{(now.get('bpm') or 0):.0f} bpm  {now.get('camelot') or ''}")
        else:
            self.now_lbl.setText("…")
            self.now_sub.setText("")
        if nxt:
            eta = nxt.get("eta_s")
            self.next_lbl.setText(nxt.get("title") or "")
            self.next_sub.setText(f"{nxt.get('artist') or ''}   " + (f"in about {eta:.0f} s" if eta is not None else "when the exit comes")
                                  + (f"   ·   {nxt.get('how')}" if nxt.get("how") else "") + (f" {nxt['beats']} beats" if nxt.get("beats") else ""))
        else:
            self.next_lbl.setText("not chosen yet")
            self.next_sub.setText("")
        self.intent_lbl.setText(st.get("intent") or "")
        lanes = st.get("lanes")
        if lanes:
            self.lanes_lbl.setText("   ".join(f"{ln}: {t or '-'}" for ln, t in lanes.items()))
        else:
            self.lanes_lbl.setText("")
        rec = list(st.get("recent") or []) + [f"{t}  {m}" for t, m in (st.get("events") or [])[-4:]]
        sig = tuple(rec)
        if sig != getattr(self, "_rec_sig", None):
            self._rec_sig = sig
            self.recent.clear()
            for line in reversed(rec):
                self.recent.addItem(line[:160])
        self._render_queue()
        self.buttons["BREAK"].setEnabled(st.get("mode") == "layered")
        err = st.get("error")
        self.err_lbl.setText(f"ERROR {err}" if err else "")
        # the picture
        self.canvas.follow(d.bar())
        for tid in d.tl.tracks():
            if tid and self.spectro.get(tid) is None and tid not in self.spectro._busy and tid not in self.spectro.errors:
                t = self.track(tid)
                if t is not None and getattr(t, "has_stems", False):
                    self.spectro.request(t)
        self.canvas.update()
