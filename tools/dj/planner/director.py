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
             "how long records play before the next seam (never before the payoff); layered, the phrase length the arrangement moves in: "
             "a new bed settles four of them before the next voice arrives, a voice is heard three before it may take the bed (at its hook or drop)"),
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
                "how often the DJ makes its own moments, spaced at least 8 min (some) / 4 min (lots) apart and only while the arc is warm: "
                "one song, a double-drop into the next song; layered, breaks at breakdowns, drops onto a voice, and every bed change "
                "as a four-bar breakdown before the new drums and bass land (rare = clean crosses only)"),
    "fx": ("FX", {"none": "none", "some": "some", "lots": "lots"},
           "shapes on moves (filter-out, stutter-out, echo) and the filter / echo seams' odds"),
    "arc": ("ARC", {"theme": "theme", "steady": "steady", "build": "build", "waves": "waves", "down": "down", "yours": "yours"},
            "the night's energy plan, drawn on the strip above: the theme's own curve · steady · a build to the end · waves (two swells "
            "with a breather) · a wind-down · yours (drag up / down on the strip to bend the curve; click on it = we are here)"),
    "length": ("LENGTH", {"45m": "45 m", "90m": "90 m", "3h": "3 h", "night": "night"},
               "how long the arc runs; changing it puts you at the same point of the new length"),
}
DIAL_ORDER = ("layers", "mixing", "arc", "length", "energy", "tempo", "pace", "seams", "vocals", "variety", "loops",
              "moments", "fx", "bass", "tone", "level")


class ArcStrip(QWidget):
    """The night's arc as a picture: the plan as a curve (with the ENERGY lean on it), the songs that played
    as dots at their energy, the playhead. Click = 'we are here' (both engines' set clocks move); drag up
    or down = bend the curve there (the ARC dial becomes 'yours')."""
    PAD = 10

    def __init__(self, tab):
        super().__init__(tab)
        self.tab = tab
        self.arc = None
        self._press = None
        self.setMinimumHeight(58)
        self.setMaximumHeight(58)
        self.setToolTip("the night's arc: the plan (curve), what played (dots at each song's energy), where we are (line)\n"
                        "click = we are here on the arc · drag up / down = bend the plan there (ARC becomes 'yours')")
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_arc(self, arc):
        self.arc = arc
        self.update()

    def _xy(self, p, e):
        w, h = self.width(), self.height()
        return self.PAD + p * (w - 2 * self.PAD), (h - 8) - e * (h - 26)

    def _pe(self, x, y):
        w, h = self.width(), self.height()
        return (max(0.0, min(1.0, (x - self.PAD) / max(1.0, w - 2 * self.PAD))),
                max(0.0, min(1.0, ((h - 8) - y) / max(1.0, h - 26))))

    def paintEvent(self, ev):
        from PyQt6.QtGui import QPainter, QColor, QPen, QPainterPath, QPolygonF
        from PyQt6.QtCore import QPointF
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.fillRect(self.rect(), QColor(20, 20, 26))
        a = self.arc or {}
        curve = a.get("curve") or []
        qp.setPen(QColor(154, 154, 166))
        if not curve:
            qp.drawText(self.PAD, 16, "ARC - the night's energy plan appears when the Director starts")
            return
        w, h = self.width(), self.height()
        # quarter ticks with minutes
        length = float(a.get("length_s") or 1.0)
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            x, _ = self._xy(q, 0.0)
            qp.setPen(QColor(50, 50, 60))
            qp.drawLine(int(x), 18, int(x), h - 8)
            qp.setPen(QColor(120, 120, 135))
            qp.drawText(int(x) + 3, h - 1, f"{q * length / 60:.0f}m")
        # the plan
        path = QPainterPath()
        x0, y0 = self._xy(0.0, 0.0)
        path.moveTo(x0, y0)
        for p, e in curve:
            x, y = self._xy(p, e)
            path.lineTo(x, y)
        x1, _ = self._xy(1.0, 0.0)
        path.lineTo(x1, y0)
        path.closeSubpath()
        qp.fillPath(path, QColor(74, 122, 217, 70))
        qp.setPen(QPen(QColor(125, 162, 227), 1.6))
        qp.drawPolyline(QPolygonF([QPointF(*self._xy(p, e)) for p, e in curve]))
        # what played
        qp.setPen(Qt.PenStyle.NoPen)
        for pp, e, _title in a.get("played") or []:
            x, y = self._xy(pp, e)
            qp.setBrush(QColor(240, 200, 90))
            qp.drawEllipse(QPointF(x, y), 3.2, 3.2)
        # where we are
        p = float(a.get("progress") or 0.0)
        x, _ = self._xy(p, 0.0)
        qp.setPen(QPen(QColor(255, 255, 255), 1.2))
        qp.drawLine(int(x), 16, int(x), h - 8)
        tgt = a.get("target")
        if tgt is not None:
            _, y = self._xy(p, float(tgt))
            qp.setBrush(QColor(255, 255, 255))
            qp.drawEllipse(QPointF(x, y), 3.5, 3.5)
        # the words
        el = float(a.get("elapsed_s") or 0.0)
        txt = f"ARC {a.get('shape')} · {a.get('word')} · {el / 60:.0f} of {length / 60:.0f} min"
        if tgt is not None:
            txt += f" · target {tgt:.2f}"
        if a.get("heard") is not None:
            txt += f" · last song {a['heard']:.2f}"
        if a.get("peak_in_s") is not None and a.get("peak_in_s") > 60:
            txt += f" · peak ({a.get('peak_energy', 0):.2f}) in {a['peak_in_s'] / 60:.0f} min"
        qp.setPen(QColor(230, 230, 236))
        qp.drawText(self.PAD, 13, txt)

    def mousePressEvent(self, ev):
        self._press = (ev.position().x(), ev.position().y(), False)

    def mouseMoveEvent(self, ev):
        if self._press is None or self.tab.director is None:
            return
        x0, y0, moved = self._press
        if moved or abs(ev.position().y() - y0) > 4:
            p, e = self._pe(ev.position().x(), ev.position().y())
            self.tab.director.arc_bend(p, e)
            self._press = (x0, y0, True)
            self.set_arc(self.tab.director.arc_status())

    def mouseReleaseEvent(self, ev):
        if self._press is None:
            return
        x0, y0, moved = self._press
        self._press = None
        if not moved and self.tab.director is not None:
            p, _ = self._pe(ev.position().x(), ev.position().y())
            self.tab.director.arc_jump(p)
            self.set_arc(self.tab.director.arc_status())


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
        # -- the arc: the night's energy plan, what played against it, where we are; click / drag steer it ----
        self.arc_strip = ArcStrip(self)
        root.addWidget(self.arc_strip)

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
        mood_h.setToolTip("the tags of the songs in play (the chosen playlist, else the whole library); light one or more and only songs "
                          "carrying at least one of them may play (none lit = everything); the chips follow the playlist box")
        self.mood_h = mood_h
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
    def _pool_tracks(self):
        """The songs in play: the chosen playlist's tracks, else the library (the Director's when running)."""
        lib = self._library()
        name = self.songs_box.currentData()
        if not name:
            return lib, None
        try:
            from lib.dj.setlist import get_setlist
            sl = get_setlist(self._db(), name=name)
            ids = {e["track_id"] for e in (sl or {}).get("entries", [])}
            return [t for t in lib if t.id in ids], name
        except Exception:
            return lib, None

    def _fill_tags(self, force=False):
        """The MOOD chips follow the songs in play: rebuilt from the chosen playlist (else the library) whenever
        the playlist box changes or the library reloads; lit chips stay lit when their tag is still there."""
        try:
            tracks, pool = self._pool_tracks()
        except Exception:
            return
        counts = {}
        for t in tracks:
            for tag in getattr(t, "all_tags", ()) or ():
                counts[tag] = counts.get(tag, 0) + 1
        min_n = 5 if pool is None else 2
        top = [tag for tag, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_n][:36]
        sig = (pool, len(tracks), tuple(top))
        if not force and sig == getattr(self, "_tags_sig", None):
            return
        self._tags_sig = sig
        lit = {tag for tag, b in self.mood_chips.items() if b.isChecked()}
        for b in list(self.mood_chips.values()):
            self.mood_row.removeWidget(b)
            b.setParent(None)
            b.deleteLater()
        self.mood_chips = {}
        for i, tag in enumerate(top):
            b = QPushButton(tag.replace("_", " "))
            b.setCheckable(True)
            b.setProperty("kind", "small")
            b.setMinimumHeight(20)
            b.setMaximumHeight(22)
            b.setToolTip(f"{counts[tag]} of the {len(tracks)} songs in play carry '{tag}'")
            b.setChecked(tag in lit)
            b.clicked.connect(self._tags_changed)
            self.mood_row.addWidget(b, i // 6, i % 6)
            self.mood_chips[tag] = b
        self.mood_h.setText(f"MOOD  ·  {len(tracks)} songs in {('playlist ' + pool) if pool else 'the library'}"
                            + (f"  ·  {len(counts) - len(top)} rarer tags hidden" if len(counts) > len(top) else ""))
        if lit and lit - set(top):
            self._tags_changed()                      # a lit tag left with the pool: the filter follows

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
        self._fill_tags()                             # the Director's library is the one in play now
        self._fill_search()
        self._midi_open()
        self._timer.start()

    # -- nanoKONTROL2: the dials and moments under your hands -------------------------------------------------
    # faders 1-8: ENERGY, TEMPO, PACE, SEAMS, VOCALS, VARIETY, BASS, LEVEL (each fader's travel is cut into the
    # dial's options); knobs 1-8: MIXING, LAYERS, LOOPS, MOMENTS, FX, TONE, ARC, LENGTH.
    # transport: PLAY = NEXT, STOP = HOLD, REC = DROP, CYCLE = BREAK, TRACK < > = BAD / GOOD,
    # MARKER < > = arc back / ahead 10 %. Polled from the readout timer; never a thread.
    MIDI_FADERS = ("energy", "tempo", "pace", "seams", "vocals", "variety", "bass", "level")
    MIDI_KNOBS = ("mixing", "layers", "loops", "moments", "fx", "tone", "arc", "length")

    def _midi_open(self):
        self.midi = None
        self._midi_note = ""
        try:
            from lib.midi_controller import KorgNanoKontrol2
            ctl = KorgNanoKontrol2(auto_connect=True)
            if ctl.input_device is not None:
                self.midi = ctl
        except Exception as e:  # noqa: BLE001
            self._midi_note = f"nanoKONTROL2: {type(e).__name__}: {e}"
            return
        self._midi_note = "nanoKONTROL2 connected" if self.midi is not None else ""

    def _midi_close(self):
        if getattr(self, "midi", None) is not None:
            try:
                self.midi.disconnect()
            except Exception:
                pass
        self.midi = None

    def _midi_poll(self):
        ctl = getattr(self, "midi", None)
        if ctl is None or self.director is None:
            return
        try:
            changes = ctl.update()
        except Exception as e:  # noqa: BLE001
            self._midi_note = f"nanoKONTROL2: {type(e).__name__}: {e}"
            self.midi = None
            return
        if not changes:
            return
        from lib.dj.director import DIALS
        d = self.director

        def pick(name, val):
            opts = DIALS[name]
            opt = opts[min(len(opts) - 1, int(float(val) * len(opts)))]
            if d.dials.get(name) != opt:
                d.set_dial(name, opt)
        for name, val in changes.items():
            if name.startswith("slider_"):
                i = int(name.split("_")[1])
                if 1 <= i <= len(self.MIDI_FADERS):
                    pick(self.MIDI_FADERS[i - 1], val)
            elif name.startswith("knob_"):
                i = int(name.split("_")[1])
                if 1 <= i <= len(self.MIDI_KNOBS):
                    pick(self.MIDI_KNOBS[i - 1], val)
            elif val is True:                                   # transport buttons act on press
                if name == "play":
                    d.next()
                elif name == "stop":
                    d.hold()
                elif name == "record":
                    d.drop()
                elif name == "cycle":
                    d.break_()
                elif name == "track_prev":
                    d.rate(False)
                elif name == "track_next":
                    d.rate(True)
                elif name == "marker_prev":
                    d.arc_jump(max(0.0, d.arc_progress() - 0.1))
                elif name == "marker_next":
                    d.arc_jump(min(0.999, d.arc_progress() + 0.1))

    def close(self):
        self._timer.stop()
        self._midi_close()
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

    def showEvent(self, ev):
        super().showEvent(ev)
        # coming back to the tab: the chips and the list follow whatever the library is NOW (a rescan, new
        # tags, a new playlist), and a cached library from before the planner had loaded its own is dropped
        if getattr(self.planner, "library", None):
            self._lib = None
        QTimer.singleShot(0, self._fill_tags)
        QTimer.singleShot(0, self._fill_search)

    def _pool_changed(self, *_):
        if self.director is not None:
            self.director.set_pool(self.songs_box.currentData())
        self._found_sig = None
        self._fill_tags()
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
        self._midi_poll()
        st = d.status()
        now, nxt = st.get("now"), st.get("next")
        mode = "one song at a time" if st.get("mode") == "one" else ("songs layered" if st.get("mode") else "-")
        last = (st.get("events") or [("", "")])[-1][1]
        doubled = st.get("doubled") or []
        self.state_lbl.setText(f"{mode}   ·   {st.get('theme')}   ·   {st.get('pool') or 'whole library'}"
                               + (f"   ·   switching to {st['switching']}…" if st.get("switching") else "")
                               + (f"   ·   ⚠ DOUBLED: {', '.join(doubled)} (two songs' {'stems' if doubled != ['mix'] else 'mixes'} at once)" if doubled else "")
                               + (f"   ·   {last[:60]}" if last else "")
                               + (f"   ·   {self._midi_note}" if getattr(self, "_midi_note", "") else ""))
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
        self.arc_strip.set_arc(st.get("arc"))
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
        if self._songs_tick % 40 == 0:
            self._fill_tags()                         # the library may have been rescanned or retagged meanwhile
        err = st.get("error")
        self.err_lbl.setText(f"ERROR {err}" if err else "")
        self.canvas.follow(d.bar())
        for tid in d.tl.tracks():
            if tid and self.spectro.get(tid) is None and tid not in self.spectro._busy and tid not in self.spectro.errors:
                t = self.track(tid)
                if t is not None and getattr(t, "has_stems", False):
                    self.spectro.request(t)
        self.canvas.update()
