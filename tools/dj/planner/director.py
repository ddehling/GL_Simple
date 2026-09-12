"""Director tab: high-level behaviour over the DJ, for parties and events.

One screen, no scrolling: the picture of the run across the top (the four lanes, what played, the
playhead, what is planned ahead); below it on the left what is playing, what comes next and WHY, and the
songs (the pool ranked by fit, ✓ mixable / ✗ would be rejected, UP NEXT); on the right the ten dials in
two columns and the moments - NEXT, STAY, DROP, BREAK, GOOD, BAD. Explanations live in tooltips. No lanes,
no clips, no stems, no levels: the engines (lib/dj/system.py, lib/dj/remix.py) do that under
lib/dj/director.py.
"""
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QPushButton, QVBoxLayout, QWidget, QSizePolicy)

from tools.dj.planner.perform import STYLE, SetlistBox, _mmss

DIAL_LABELS = {
    "mixing": ("MIXING", {"auto": "auto", "blend": "blend", "cut": "cut", "morph": "morph"},
               "how songs join: the dice · long blends · hard cuts · stem morphs (pins through the same gates; a refused cut becomes a phrase cut)"),
    "layers": ("LAYERS", {"one": "one", "two": "two", "three": "three"},
               "one song at a time (the autoDJ as it is) · parts of two songs together · parts of three; the playing song is carried across"),
    "energy": ("ENERGY", {"cool": "cool", "hold": "hold", "amp": "amp"},
               "a lean on the night's arc; amped, the clock may also travel with it when layered"),
    "tempo": ("TEMPO", {"slower": "slower", "hold": "hold", "faster": "faster"},
              "lean the tempo journey: picks aim 6 bpm lower / higher; layered, the clock travels 3 % that way"),
    "pace": ("PACE", {"short": "short", "normal": "normal", "long": "long"},
             "how long records play before the next seam; layered, how often the lanes move"),
    "seams": ("SEAMS", {"quick": "quick", "normal": "normal", "long": "long"},
              "how long the mixes themselves run: blend lengths × 0.5 / 1 / 2; layered, the lanes' crossfade"),
    "vocals": ("VOCALS", {"none": "none", "some": "some", "lots": "lots"},
               "how much singing: picks lean toward instrumentals or vocal records; layered, how freely the vocal lane crosses"),
    "variety": ("VARIETY", {"close": "close", "varied": "varied", "wild": "wild"},
                "how far the next song may roam from the playing one: close (clean key journeys) · varied · wild (range over pocket)"),
    "loops": ("LOOPS", {"off": "off", "some": "some", "lots": "lots"},
              "how much the seams loop (loop entries' odds; layered, eight-bar holds on grooves at phrase boundaries)"),
    "bass": ("BASS", {"flat": "flat", "boost": "boost", "heavy": "heavy"},
             "the mix bus low band (under 200 Hz): flat · +30 % · +60 %"),
    "tone": ("TONE", {"dark": "dark", "neutral": "neutral", "bright": "bright"},
             "the mix bus high band (over 2.5 kHz): −30 % · flat · +30 %"),
    "level": ("LEVEL", {"quiet": "quiet", "normal": "normal", "loud": "loud"},
              "the mix bus gain: 60 % · 85 % · 100 %"),
    "moments": ("MOMENTS", {"rare": "rare", "some": "some", "lots": "lots"},
                "how often the DJ makes its own moments: one song, a double-drop into the next song once a record is 60 % through; layered, breaks at breakdowns and drops onto a voice"),
    "fx": ("FX", {"none": "none", "some": "some", "lots": "lots"},
           "shapes on moves (filter-out, stutter-out, echo) and the filter / echo seams' odds"),
}
DIAL_ORDER = ("layers", "mixing", "energy", "tempo", "pace", "seams", "vocals", "variety", "loops",
              "moments", "fx", "bass", "tone", "level")


class DirectorTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.setObjectName("perform")
        # the shared dark style, with every button smaller: less padding, a smaller face
        self.setStyleSheet(STYLE + """
QWidget#perform QPushButton { padding: 3px 6px; font-size: 10pt; font-weight: 600; border-radius: 5px; }
QWidget#perform QPushButton[kind="small"] { padding: 2px 5px; font-size: 9.5pt; font-weight: 500; }
QWidget#perform QComboBox, QWidget#perform QLineEdit { padding: 3px 6px; font-size: 10pt; }
QWidget#perform QListWidget { font-size: 9.5pt; }
""")
        self.planner = planner
        self.engine = None
        self.director = None
        self._own_db = None
        self._lib = None
        self._colors = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # -- top: start, theme, songs, one status line ----------------------------------------------------------
        top = QHBoxLayout()
        self.start_btn = QPushButton("▶  START")
        self.start_btn.setProperty("kind", "go")
        self.start_btn.setMinimumWidth(120)
        self.start_btn.setMinimumHeight(30)
        self.start_btn.clicked.connect(self._toggle)
        top.addWidget(self.start_btn)
        top.addSpacing(12)
        self.theme_box = QComboBox()
        from lib.dj.themes import PICKER_THEMES
        self.theme_box.addItems(list(PICKER_THEMES))
        self.theme_box.setCurrentText("groove")
        self.theme_box.setToolTip("theme: what kind of songs, and the shape of the night")
        self.theme_box.currentTextChanged.connect(lambda n: self.director and self.director.set_theme(n))
        top.addWidget(self.theme_box)
        self.songs_box = SetlistBox(self)
        self.songs_box.setMinimumWidth(220)
        self.songs_box.setToolTip("a playlist (the Set tab's saved lists) as the pool of songs; whole library lifts it")
        self.songs_box.currentIndexChanged.connect(self._pool_changed)
        top.addWidget(self.songs_box)
        self.state_lbl = QLabel("")
        self.state_lbl.setProperty("dim", "true")
        top.addWidget(self.state_lbl, 1)
        root.addLayout(top)

        # -- the picture ----------------------------------------------------------------------------------------------
        from lib.dj.timeline import Timeline, Spectro
        from tools.dj.planner.timeline import TimelineCanvas
        self._empty_tl = Timeline("director")
        self.spectro = Spectro(planner.music_dir)
        self.canvas = TimelineCanvas(self, lane_h=40, readonly=True)
        self.canvas.px_per_bar = 9.0
        self.canvas.setToolTip("what played on each lane and when · the playhead · dashed = what the Director plans next · Ctrl+wheel zooms, wheel scrolls")
        root.addWidget(self.canvas)

        body = QHBoxLayout()
        body.setSpacing(14)
        # -- left: now, next, why, the songs ----------------------------------------------------------------------------
        left = QVBoxLayout()
        left.setSpacing(3)

        def lab(size, bold=False, dim=False):
            w = QLabel("")
            f = w.font()
            f.setPointSizeF(size)
            f.setBold(bold)
            w.setFont(f)
            w.setWordWrap(True)
            if dim:
                w.setProperty("dim", "true")
            return w
        self.now_lbl = lab(18, bold=True)
        self.now_sub = lab(10.5, dim=True)
        self.next_lbl = lab(13, bold=True)
        self.next_sub = lab(10, dim=True)
        self.intent_lbl = lab(10.5)
        self.why_lbl = lab(10, dim=True)
        self.why_lbl.setMaximumHeight(70)
        self.lanes_lbl = lab(10, dim=True)
        for w in (self.now_lbl, self.now_sub, self.next_lbl, self.next_sub, self.intent_lbl, self.lanes_lbl, self.why_lbl):
            left.addWidget(w)
        srow = QHBoxLayout()
        self.found_h = QLabel("SONGS")
        self.found_h.setProperty("dim", "true")
        srow.addWidget(self.found_h)
        self.search = QLineEdit()
        self.search.setPlaceholderText("filter songs")
        self.search.setToolTip("empty = the whole pool, ranked by fit to what is playing; ✓ fits from here, ✗ would be rejected (the reason follows); double-click = UP NEXT")
        self.search.textChanged.connect(self._fill_search)
        srow.addWidget(self.search, 1)
        left.addLayout(srow)
        lists = QHBoxLayout()
        self.found = QListWidget()
        self.found.setMinimumHeight(120)
        self.found.itemDoubleClicked.connect(lambda it: self._queue(it.data(Qt.ItemDataRole.UserRole)))
        self.found.setToolTip("✓ mixable from here · ✗ would be rejected · double-click to put it UP NEXT")
        lists.addWidget(self.found, 3)
        qcol = QVBoxLayout()
        un = QLabel("UP NEXT")
        un.setProperty("dim", "true")
        un.setToolTip("your queue, in order; honoured when it can be mixed from where it is; double-click to remove")
        qcol.addWidget(un)
        self.queue = QListWidget()
        self.queue.itemDoubleClicked.connect(lambda it: self._unqueue(it.data(Qt.ItemDataRole.UserRole)))
        qcol.addWidget(self.queue, 1)
        lists.addLayout(qcol, 2)
        left.addLayout(lists, 1)
        body.addLayout(left, 5)

        # -- right: the dials in two columns, the moments ---------------------------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(8)
        from lib.dj.director import DIALS, DEFAULTS
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        self.dial_groups, self._dial_labels = {}, {}
        for i, name in enumerate(DIAL_ORDER):
            options = DIALS[name]
            title, labels, tip = DIAL_LABELS[name]
            cell = QWidget()
            cv = QVBoxLayout(cell)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(2)
            head = QLabel(title)
            head.setProperty("dim", "true")
            head.setToolTip(tip)
            cv.addWidget(head)
            row = QHBoxLayout()
            row.setSpacing(4)
            group = QButtonGroup(self)
            group.setExclusive(True)
            for opt in options:
                b = QPushButton(labels[opt])
                b.setCheckable(True)
                b.setMinimumHeight(22)
                b.setProperty("kind", "small")
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                b.setToolTip(tip)
                b.setChecked(opt == DEFAULTS[name])
                b.clicked.connect(lambda _c, n=name, o=opt: self._dial(n, o))
                group.addButton(b)
                row.addWidget(b)
            cv.addLayout(row)
            grid.addWidget(cell, i // 2, i % 2)
            self.dial_groups[name] = group
            self._dial_labels[name] = head
        right.addLayout(grid)
        # MOOD: the library's own tags as toggles - only songs carrying a lit tag may play (none lit = all)
        mood_h = QLabel("MOOD")
        mood_h.setProperty("dim", "true")
        mood_h.setToolTip("the library's tags; light one or more and only songs carrying at least one of them may play (none lit = everything)")
        right.addWidget(mood_h)
        self.mood_row = QGridLayout()
        self.mood_row.setSpacing(4)
        self.mood_chips = {}
        right.addLayout(self.mood_row)
        QTimer.singleShot(1500, self._fill_tags)
        mom = QGridLayout()
        mom.setSpacing(6)
        self.buttons = {}
        for (label, fn, kind, r, c) in (("NEXT", self._next, "go", 0, 0), ("HOLD", self._hold, None, 0, 1),
                                        ("DROP", self._drop, "hot", 0, 2), ("BREAK", self._break, None, 0, 3),
                                        ("GOOD", lambda: self.director and self.director.rate(True), "good", 1, 0),
                                        ("BAD", lambda: self.director and self.director.rate(False), "bad", 1, 2)):
            b = QPushButton(label)
            b.setMinimumHeight(30 if r == 0 else 26)
            if kind:
                b.setProperty("kind", kind)
            b.clicked.connect(fn)
            mom.addWidget(b, r, c, 1, 1 if r == 0 else 2)
            self.buttons[label] = b
        right.addLayout(mom)
        self.verdict_lbl = QLabel("")
        self.verdict_lbl.setProperty("dim", "true")
        self.verdict_lbl.setWordWrap(True)
        self.verdict_lbl.setToolTip("what your last GOOD / BAD rated, what the DJ learned from it, and what it did about it")
        right.addWidget(self.verdict_lbl)
        self.err_lbl = QLabel("")
        self.err_lbl.setProperty("dim", "true")
        self.err_lbl.setWordWrap(True)
        right.addWidget(self.err_lbl)
        right.addStretch(1)
        rw = QWidget()
        rw.setLayout(right)
        rw.setMinimumWidth(540)
        rw.setMaximumWidth(720)
        body.addWidget(rw, 4)
        root.addLayout(body, 1)
        self._label_moments("one")
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._tick)
        QTimer.singleShot(1200, self._fill_search)

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

    def _label_moments(self, mode):
        """The moments say what they do, in the mode we are in (the detail in the tooltip)."""
        one = mode != "layered"
        texts = {
            "NEXT": ("NEXT SONG", "move on to the next song at the next phrase") if one
                    else ("NEXT MOVE", "the conductor's next lane move on the next bar"),
            "HOLD": ("STAY", "one more phrase of this song before moving on" if one else "freeze the lanes for a phrase"),
            "DROP": ("DROP", "a short build, then the next song's drop, cold" if one else "every part to the newest song on the next bar"),
            "BREAK": ("BREAK", "needs two or three songs layered" if one else "strip to one part for four bars, then all back"),
            "GOOD": ("👍 GOOD", "that was good - the DJ does more of it"),
            "BAD": ("👎 BAD", "that was bad - the DJ does less of it"),
        }
        for k, (t, tip) in texts.items():
            if self.buttons[k].text() != t:
                self.buttons[k].setText(t)
            self.buttons[k].setToolTip(tip)
        self.buttons["BREAK"].setEnabled(not one)
        self._moments_mode = mode

    # -- mood chips -----------------------------------------------------------------------------------------------
    def _fill_tags(self):
        try:
            lib = self._library()
        except Exception:
            return
        counts = {}
        for t in lib:
            for tag in getattr(t, "all_tags", ()) or ():
                counts[tag] = counts.get(tag, 0) + 1
        top = [tag for tag, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= 5][:36]
        for i, tag in enumerate(top):
            if tag in self.mood_chips:
                continue
            b = QPushButton(tag.replace("_", " "))
            b.setCheckable(True)
            b.setProperty("kind", "small")
            b.setMinimumHeight(20)
            b.setMaximumHeight(22)
            b.setToolTip(f"{counts[tag]} songs")
            b.clicked.connect(self._tags_changed)
            self.mood_row.addWidget(b, i // 6, i % 6)
            self.mood_chips[tag] = b

    def _tags_changed(self, *_):
        tags = [tag for tag, b in self.mood_chips.items() if b.isChecked()]
        if self.director is not None:
            self.director.set_tags(tags)
        self._found_sig = None
        self._fill_search()

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
        from lib.dj.director import Director, DIALS
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
                    label = b.text()
                    opt = next(o for o in DIALS[name] if DIAL_LABELS[name][1][o] == label)
                    self.director.dials[name] = opt
        if self.songs_box.currentData():
            self.director.pool_name = self.songs_box.currentData()
        self.director.tags = [tag for tag, b in self.mood_chips.items() if b.isChecked()]
        if not self.director.start(threaded=True):
            self.err_lbl.setText(self.director.last_error or "could not start")
            self.close()
            return
        self.start_btn.setText("■  STOP")
        self.start_btn.setProperty("kind", "hot")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self._found_sig = None
        self._fill_search()
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

    def _pool_changed(self, *_):
        if self.director is not None:
            self.director.set_pool(self.songs_box.currentData())
        self._found_sig = None
        self._fill_search()

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
            self.err_lbl.setText("BREAK is a layered move: set LAYERS to two or three")

    def _fill_search(self, text=None):
        """The song list: with the Director running, the pool ranked by fit to what is playing, each row
        ✓ mixable from here or ✗ would be rejected with the reason; before START, the library or the chosen
        playlist alphabetically. An empty box shows everything."""
        q = (text if text is not None else self.search.text()).strip().lower()
        cur = self.found.currentItem().data(Qt.ItemDataRole.UserRole) if self.found.currentItem() else None
        rows = []
        try:
            if self.director is not None:
                rows = self.director.rank(q, n=60)
            else:
                lib = self._library()
                pool = None
                name = self.songs_box.currentData()
                if name:
                    from lib.dj.setlist import get_setlist
                    sl = get_setlist(self._db(), name=name)
                    pool = {e["track_id"] for e in (sl or {}).get("entries", [])}
                for t in sorted(lib, key=lambda t: t.title.lower()):
                    if pool is not None and t.id not in pool:
                        continue
                    if q and q not in t.title.lower() and q not in (t.artist or "").lower():
                        continue
                    rows.append({"id": t.id, "title": t.title, "artist": t.artist or "", "bpm": t.bpm, "camelot": t.camelot,
                                 "ok": None, "why": ""})
                    if len(rows) >= 60:
                        break
        except Exception as e:  # noqa: BLE001
            self.err_lbl.setText(f"songs: {type(e).__name__}: {e}")
            return
        sig = tuple((r["id"], r["ok"], r["why"]) for r in rows)
        if sig == getattr(self, "_found_sig", None):
            return
        self._found_sig = sig
        self.found.clear()
        for r in rows:
            mark = "" if r["ok"] is None else ("✓  " if r["ok"] else "✗  ")
            it = QListWidgetItem(f"{mark}{r['title'][:38]}  ·  {r['artist'][:22]}   {(r['bpm'] or 0):.0f} bpm {r['camelot'] or ''}"
                                 + (f"   {r['why']}" if r["why"] else ""))
            it.setData(Qt.ItemDataRole.UserRole, r["id"])
            if r["ok"] is False:
                it.setForeground(Qt.GlobalColor.gray)
            self.found.addItem(it)
            if r["id"] == cur:
                self.found.setCurrentItem(it)
        self.found_h.setText(f"SONGS  {len(rows)}" + ("   ✓ fits  ✗ would be rejected" if self.director is not None else ""))

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
        last = (st.get("events") or [("", "")])[-1][1]
        doubled = st.get("doubled") or []
        self.state_lbl.setText(f"{mode}   ·   {st.get('theme')}   ·   {st.get('pool') or 'whole library'}"
                               + (f"   ·   switching to {st['switching']}…" if st.get("switching") else "")
                               + (f"   ·   ⚠ DOUBLED: {', '.join(doubled)} (two songs' {'stems' if doubled != ['mix'] else 'mixes'} at once)" if doubled else "")
                               + (f"   ·   {last[:60]}" if last else ""))
        lv = st.get("last_verdict")
        if lv:
            age = time.time() - lv.get("t", 0)
            txt = ("👍 GOOD" if lv.get("up") else "👎 BAD") + f" ({age:.0f} s ago): {lv.get('what') or ''}"
            if lv.get("learn"):
                txt += f"\n→ {lv['learn']}" + (f" · weight now {lv['weight']:.2f}" if lv.get("weight") is not None else "")
            if lv.get("did"):
                txt += f"\n→ {lv['did']}"
            if self.verdict_lbl.text() != txt:
                self.verdict_lbl.setText(txt)
        else:
            self.verdict_lbl.setText("")
        if now:
            self.now_lbl.setText(f"{now.get('title')}")
            self.now_sub.setText(f"{now.get('artist') or ''}   {_mmss(now.get('pos_s'))} / {_mmss(now.get('duration_s'))}   "
                                 f"{(now.get('bpm') or 0):.0f} bpm  {now.get('camelot') or ''}")
        else:
            self.now_lbl.setText("…")
            self.now_sub.setText("")
        if nxt:
            eta = nxt.get("eta_s")
            self.next_lbl.setText("NEXT  " + (nxt.get("title") or ""))
            self.next_sub.setText(f"{nxt.get('artist') or ''}   " + (f"in about {eta:.0f} s" if eta is not None else "when the exit comes")
                                  + (f"   ·   {nxt.get('how')}" if nxt.get("how") else "") + (f" {nxt['beats']} beats" if nxt.get("beats") else ""))
        else:
            self.next_lbl.setText("NEXT  not chosen yet")
            self.next_sub.setText("")
        self.intent_lbl.setText(st.get("intent") or "")
        lanes = st.get("lanes")
        self.lanes_lbl.setText("   ".join(f"{ln}: {(t or '-')[:22]}" for ln, t in lanes.items()) if lanes else "")
        self.why_lbl.setText("\n".join(st.get("why") or []))
        # the buttons follow the Director's dials (a dial may be turned from outside the tab: the copilot, a script)
        for name, opt in (st.get("dials") or {}).items():
            group = self.dial_groups.get(name)
            if group is None:
                continue
            want = DIAL_LABELS[name][1].get(opt)
            for b in group.buttons():
                if b.text() == want and not b.isChecked():
                    b.setChecked(True)
        for line in st.get("in_effect") or []:
            name = line.split(" ", 1)[0]
            head = self._dial_labels.get(name)
            if head is not None:
                tip = f"{line}\n\n{DIAL_LABELS[name][2]}"
                if head.toolTip() != tip:
                    head.setToolTip(tip)
                    for b in self.dial_groups[name].buttons():
                        b.setToolTip(tip)
        self._render_queue()
        if getattr(self, "_moments_mode", None) != st.get("mode"):
            self._label_moments(st.get("mode"))
        self._songs_tick = getattr(self, "_songs_tick", 0) + 1
        if self._songs_tick % 12 == 0:
            self._fill_search()
        err = st.get("error")
        self.err_lbl.setText(f"ERROR {err}" if err else "")
        self.canvas.follow(d.bar())
        for tid in d.tl.tracks():
            if tid and self.spectro.get(tid) is None and tid not in self.spectro._busy and tid not in self.spectro.errors:
                t = self.track(tid)
                if t is not None and getattr(t, "has_stems", False):
                    self.spectro.request(t)
        self.canvas.update()
