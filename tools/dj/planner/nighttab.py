"""The NIGHT tab's parts (docs/DJ_SETS.md section 4, 2026-10-29): the theme of a planned night edited in place - its
CRATE (which records may play), its LEANS (weighted words that rank them), its STORY (the chapters) and its KEYS mode -
and the night strip's drawing. The Director tab (tools/dj/planner/director.py) owns the picker, the debounce and the
planner; this file only edits a theme dict and draws a plan status. Nothing here plans: every change is a signal, and
the tab sends the theme to the planner's own process.
"""
import json

from PyQt6.QtCore import QEvent, QPoint, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QCompleter, QDialog, QDialogButtonBox, QDoubleSpinBox,
                             QGridLayout, QHBoxLayout, QLabel, QLayout, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
                             QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from lib.dj.setplanner import CH_DIMS, KEYS_MODES

# the dials that act in RECORD mode (director.set_dial: pace, bass, tone, level; energy on a plan) and LAYERS, which
# switches engines and must stay reachable; ARC is left out - in a planned night the chapters are the arc
NIGHT_DIALS = ("layers", "pace", "energy", "bass", "tone", "level")
KEYS_TIPS = {
    "veto": "veto: the player's own key check, pass / fail (today's planner exactly; every theme that does not set keys)",
    "close": "close: prefer Camelot distance 1 or less (same key, relative, a neighbour) over the diagonal the veto lets through",
    "wander": "wander: no preference beyond the veto",
    "lift": "lift: where a chapter's energy rises, prefer the +1 move on the wheel (the energy boost); +2 is under the "
            "player's veto, so it never blends",
}
KEYS_HEAD_TIP = ("KEYS: how the night moves through keys - a light lean on the planner's choice, never a refusal of its own; "
                 "the player's key veto stands in every mode. The keys were rescanned 2026-10-28 (key v2, 0.825 exact "
                 "against Beatport). The presets plan as before until you pick a mode.")
DIM_TIPS = {"mood": "mood: the record's valence rank (0 dark .. 1 bright)", "vocals": "vocals: the share of the record that sings (its stems)",
            "hypnotic": "hypnotic: the record's hypnotic rank", "hardness": "hardness: the record's hardness rank",
            "mode": "mode: minor 0 .. major 1, from the key reading (key v2, 88 % right on the truth set; an unsure reading "
                    "sits nearer the middle); 0.5 = either",
            "percussive": "percussive: melodic 0 .. percussive 1, the drums' share of drums + melody in the record's stems, "
                          "ranked across the library"}
VOCALS = ("any", "none", "some", "lots")
VOCALS_TIPS = {"any": "any record, sung or not", "none": "only records that do not sing (5 % of the record at most, by the stems)",
               "some": "records that sing at most a third of the time", "lots": "records that sing a quarter of the time or more"}
CHIP_STYLE = """
QPushButton[chip="in"] { background: #2f6b3f; border-color: #43975a; color: #e6e6ea; }
QPushButton[chip="out"] { background: #4a2020; border-color: #8a3a3a; color: #c9a0a0; text-decoration: line-through; }
QPushButton[warn="amber"] { background: #4a3a14; border-color: #b08a2e; color: #f0d48a; }
QPushButton[warn="red"] { background: #5c2424; border-color: #b04444; color: #f0b0b0; }
QPushButton[warn="dim"] { background: #22222a; border-color: #33333e; color: #9a9aa6; }
"""


def _small(b, h=20, w=None):
    b.setProperty("kind", "small")
    b.setMinimumHeight(h)
    b.setMaximumHeight(h + 2)
    if w:
        b.setMaximumWidth(w)
    return b


def _restyle(w):
    w.style().unpolish(w)
    w.style().polish(w)


def _dim(text, tip=None, w=None):
    lb = QLabel(text)
    lb.setProperty("dim", "true")
    if tip:
        lb.setToolTip(tip)
    if w:
        lb.setFixedWidth(w)
    return lb


def _spin01(v, tip):
    s = QDoubleSpinBox()
    s.setRange(0.0, 1.0)
    s.setDecimals(2)
    s.setSingleStep(0.05)
    s.setValue(float(v))
    s.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
    s.setKeyboardTracking(False)             # (typing applies on Enter / leaving the box, the wheel at once)
    s.setFixedWidth(46)
    s.setToolTip(tip)
    return s


class FlowLayout(QLayout):
    """Chips that wrap to the width they are given (a grid of fixed columns clipped them on a narrow column)."""

    def __init__(self, spacing=3):
        super().__init__()
        self._items = []
        self._hfw = {}                                 # width -> height (the layouts above ask the same widths many times)
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)
        self._hfw.clear()

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        self._hfw.clear()
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def invalidate(self):
        self._hfw.clear()
        super().invalidate()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        h = self._hfw.get(w)
        if h is None:
            h = self._hfw[w] = self._lay(QRect(0, 0, w, 0), True)
        return h

    def setGeometry(self, r):
        super().setGeometry(r)
        self._lay(r, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QSize()
        for it in self._items:
            s = s.expandedTo(it.minimumSize())
        return s

    def _lay(self, rect, test):
        x, y, line = rect.x(), rect.y(), 0
        sp = self.spacing()
        for it in self._items:
            sz = it.sizeHint()
            if x > rect.x() and x + sz.width() > rect.right() + 1:
                x, y, line = rect.x(), y + line + sp, 0
            if not test:
                it.setGeometry(QRect(QPoint(x, y), sz))
            x += sz.width() + sp
            line = max(line, sz.height())
        return y + line - rect.y()


class Chip(QPushButton):
    """A word as a small toggle: a click cycles off -> in -> out (a two-way chip: off -> in); right-click = out."""

    def __init__(self, word, tip, tri, on_change):
        super().__init__(word.replace("&", "&&"))      # (a bare & is a shortcut mark: "dance & dj" read "dance  dj")
        self.word, self.tri, self.state = word, tri, ""
        self._polished, self.look = False, None       # (look = the state its style was last polished with)
        self._on_change = on_change
        _small(self)
        self.setToolTip(tip)
        self.clicked.connect(self._cycle)
        if tri:
            self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.customContextMenuRequested.connect(lambda _p: self._set_and_tell("" if self.state == "out" else "out"))

    def set_state(self, s):
        self.state = s
        if s == self.property("chip"):
            return
        self.setProperty("chip", s)
        if self.isVisible():
            _restyle(self)                             # (a chip not shown yet takes its look when it is first polished)
            self.look = s

    def showEvent(self, e):
        # a kept chip hidden with one look and given another state while hidden: Qt does not repolish a dynamic property
        # when the widget is shown again, so a "rock" vetoed by the last theme came back struck through under a theme
        # that does not veto it (2026-10-29) - the crate on screen must be the crate that plans
        super().showEvent(e)
        s = self.property("chip")
        if not self._polished:
            self._polished, self.look = True, s       # (the first show polished it with the state it has)
        elif self.look != s:
            _restyle(self)
            self.look = s

    def _set_and_tell(self, s):
        self.set_state(s)
        self._on_change()

    def _cycle(self):
        order = ("", "in", "out") if self.tri else ("", "in")
        self._set_and_tell(order[(order.index(self.state) + 1) % len(order)] if self.state in order else "")


class DimsDialog(QDialog):
    """A chapter's optional courses (mood, vocals, hypnotic, hardness, mode, percussive): each on or off, from -> to."""

    def __init__(self, parent, dims):
        super().__init__(parent)
        self.setWindowTitle("chapter courses")
        g = QGridLayout(self)
        self.rows = {}
        for i, k in enumerate(CH_DIMS):
            v = dims.get(k)
            cb = QCheckBox(k)
            cb.setChecked(v is not None)
            cb.setToolTip(DIM_TIPS[k])
            a, b = _spin01(v[0] if v else 0.5, f"{k} at the chapter's start"), _spin01(v[-1] if v else 0.5, f"{k} at its end")
            g.addWidget(cb, i, 0)
            g.addWidget(a, i, 1)
            g.addWidget(QLabel("→"), i, 2)
            g.addWidget(b, i, 3)
            self.rows[k] = (cb, a, b)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        g.addWidget(bb, len(CH_DIMS), 0, 1, 4)

    def dims(self):
        return {k: [round(a.value(), 2), round(b.value(), 2)] for k, (cb, a, b) in self.rows.items() if cb.isChecked()}


DIM_SHORT = {"mood": "mood", "vocals": "vox", "hypnotic": "hyp", "hardness": "hard", "mode": "maj", "percussive": "perc"}


def _f2(x):
    return f"{float(x):.2f}".lstrip("0") if float(x) < 1.0 else f"{float(x):.1f}"


def dims_words(dims):
    return " · ".join(f"{DIM_SHORT.get(k, k[:4])} {_f2(v[0])}→{_f2(v[-1])}" for k, v in (dims or {}).items()) or "courses…"


class ChapterRow(QWidget):
    """One chapter on two lines: order, name, share, energy from -> to, tempo from -> to, remove; its leans and courses."""

    def __init__(self, ed, ch):
        super().__init__()
        self.ed = ed
        self.dims = dict(ch.get("dims") or {})
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 2, 0, 4)
        v.setSpacing(2)
        r1 = QHBoxLayout()
        r1.setSpacing(4)
        self.up = _small(QPushButton("▲"), w=24)
        self.dn = _small(QPushButton("▼"), w=24)
        self.up.setToolTip("earlier in the night")
        self.dn.setToolTip("later in the night")
        self.up.clicked.connect(lambda: ed.move_chapter(self, -1))
        self.dn.clicked.connect(lambda: ed.move_chapter(self, +1))
        self.name = QLineEdit(str(ch.get("name") or "chapter"))
        self.name.setToolTip("the chapter's name (the strip and THE PLAN show it)")
        self.name.setMinimumWidth(80)
        self.name.editingFinished.connect(ed.touch)
        self.share = QSpinBox()
        self.share.setRange(1, 100)
        self.share.setSuffix(" %")
        self.share.setValue(int(round(100 * float(ch.get("share") or 0.25))))
        self.share.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.share.setKeyboardTracking(False)
        self.share.setFixedWidth(52)
        self.share.setToolTip("its share of the night (the shares are renormalised to 100 %)")
        self.share.valueChanged.connect(ed.touch)
        e, t = ch.get("energy") or [0.5, 0.5], ch.get("tempo") or [0.5, 0.5]
        self.e0, self.e1 = _spin01(e[0], "energy at the chapter's start (the energy rank, 0..1)"), _spin01(e[-1], "energy at its end")
        self.t0, self.t1 = (_spin01(t[0], "tempo at the chapter's start, as a percentile of the crate's tempos (0 = its slowest)"),
                            _spin01(t[-1], "tempo at its end (a crate percentile)"))
        for s in (self.e0, self.e1, self.t0, self.t1):
            s.valueChanged.connect(ed.touch)
        self.rm = _small(QPushButton("✕"), w=24)
        self.rm.setToolTip("remove this chapter")
        self.rm.clicked.connect(lambda: ed.remove_chapter(self))
        for w in (self.up, self.dn, self.name):
            r1.addWidget(w)
        r1.addWidget(self.share)
        r1.addSpacing(4)
        r1.addWidget(self.e0)
        r1.addWidget(_dim("→"))
        r1.addWidget(self.e1)
        r1.addSpacing(4)
        r1.addWidget(self.t0)
        r1.addWidget(_dim("→"))
        r1.addWidget(self.t1)
        r1.addWidget(self.rm)
        v.addLayout(r1)
        r2 = QHBoxLayout()
        r2.setSpacing(4)
        r2.addSpacing(52)
        self.leans = QLineEdit(", ".join(ch.get("leans") or []))
        self.leans.setPlaceholderText("chapter leans: words, comma separated")
        self.leans.setToolTip("the chapter's own leans: records carrying one of these words fit it better")
        self.leans.editingFinished.connect(ed.touch)
        comp = ed.completer(self.leans)
        if comp is not None:
            self.leans.setCompleter(comp)
        self.dims_btn = _small(QPushButton(dims_words(self.dims)))
        self.dims_btn.setToolTip("optional courses through the chapter: mood, vocals, hypnotic, hardness, mode (minor → major), "
                                 "percussive (melodic → percussive), from → to")
        self.dims_btn.setMaximumWidth(200)
        self.dims_btn.clicked.connect(self._edit_dims)
        # what the plan really asks of the chapter when the ENERGY dial leans it (empty on hold)
        self.eff = _dim("")
        self.eff.setStyleSheet("color: #f0d48a;")
        r2.addWidget(self.leans, 1)
        r2.addWidget(self.eff)
        r2.addWidget(self.dims_btn)
        v.addLayout(r2)

    def set_effective(self, energy, dial):
        txt = f"plans {_f2(energy[0])}→{_f2(energy[-1])}" if (energy and dial and dial != "hold") else ""
        if self.eff.text() != txt:
            self.eff.setText(txt)
            self.eff.setToolTip(f"ENERGY {dial}: every chapter's energy course is moved by {'+' if dial == 'amp' else '−'}0.25 "
                                "(clamped to 0..1); the plan and the strip use these numbers, the boxes keep yours" if txt else "")

    def set(self, ch):
        """Refill in place (a theme picked: the rows are reused, not rebuilt)."""
        self.dims = dict(ch.get("dims") or {})
        self.name.setText(str(ch.get("name") or "chapter"))
        self.share.setValue(int(round(100 * float(ch.get("share") or 0.25))))
        e, t = ch.get("energy") or [0.5, 0.5], ch.get("tempo") or [0.5, 0.5]
        for s, val in ((self.e0, e[0]), (self.e1, e[-1]), (self.t0, t[0]), (self.t1, t[-1])):
            s.setValue(float(val))
        self.leans.setText(", ".join(ch.get("leans") or []))
        self.dims_btn.setText(dims_words(self.dims))

    def _edit_dims(self):
        dlg = DimsDialog(self, self.dims)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.dims = dlg.dims()
            self.dims_btn.setText(dims_words(self.dims))
            self.ed.touch()

    def chapter(self):
        return {"name": self.name.text().strip() or "chapter", "share": self.share.value() / 100.0,
                "energy": [round(self.e0.value(), 3), round(self.e1.value(), 3)], "tempo": [round(self.t0.value(), 3), round(self.t1.value(), 3)],
                "leans": [w.strip().lower() for w in self.leans.text().split(",") if w.strip()], "dims": dict(self.dims)}


class LeanRow(QWidget):
    def __init__(self, ed, word, weight):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.word = QLineEdit(str(word))
        self.word.setToolTip("a word (a genre, a tag, a mood): records carrying it rank higher")
        self.word.editingFinished.connect(ed.touch)
        comp = ed.completer(self.word)
        if comp is not None:
            self.word.setCompleter(comp)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 200)
        self.slider.setValue(int(round(100 * float(weight))))
        self.slider.setFixedWidth(150)
        self.slider.setToolTip("its weight, 0 to 2")
        self.val = _dim(f"{float(weight):.1f}", w=30)
        self.slider.valueChanged.connect(lambda v: (self.val.setText(f"{v / 100:.1f}"), ed.touch()))
        rm = _small(QPushButton("✕"), w=24)
        rm.setToolTip("remove this lean")
        rm.clicked.connect(lambda: ed.remove_lean(self))
        h.addWidget(self.word, 1)
        h.addWidget(self.slider)
        h.addWidget(self.val)
        h.addWidget(rm)

    def set(self, word, weight):
        self.word.setText(str(word))
        self.slider.setValue(int(round(100 * float(weight))))
        self.val.setText(f"{float(weight):.1f}")

    def lean(self):
        w = self.word.text().strip().lower()
        return (w, round(self.slider.value() / 100.0, 2)) if w else None


class NightEditor(QWidget):
    """The theme of the planned night, edited in place: CRATE | LEANS | STORY, plus the KEYS row (placed by the tab
    among the dials). Every edit emits `changed`; theme() reads the widgets back into the planner's theme dict."""
    changed = pyqtSignal()
    keys_changed = pyqtSignal(str)
    LIMITS = {"genres": 10, "moods": 8, "tags": 12, "groups": 8}     # (the floor: the most common words that fit 1280x800 unscrolled;
    #                                                      a taller CRATE shows more - _fit_chips - and "+N" shows the rest)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(CHIP_STYLE)
        self._base = None
        self._filling = False
        self.vocab = {"genres": [], "moods": [], "tags": [], "groups": []}
        self.completer_words = []
        self._more = {"genres": False, "moods": False, "tags": False, "groups": False}
        self._lim = dict(self.LIMITS)                  # the caps the room allows now (never under LIMITS)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(60)                # (a window drag sends many resizes: one refill after it settles)
        self._fit_timer.timeout.connect(self._fit_chips)
        self.chips = {"genres": {}, "moods": {}, "tags": {}, "groups": {}}     # the chips that show, in their order (+ "\0more")
        self._pool = {"genres": {}, "moods": {}, "tags": {}, "groups": {}}      # every chip built, by word (the hidden ones kept)
        self._more_btn = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        # the header: the live count on its own line (the theme's blurb in its tooltip)
        self.count_lbl = _dim("")
        self.count_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._count_tip, self._blurb = "", ""
        v.addWidget(self.count_lbl)
        # the sections, and the planner's warnings beside them as short words (the sentence in the tooltip)
        secs = QHBoxLayout()
        secs.setSpacing(4)
        self.sec_group = QButtonGroup(self)
        self.sec_group.setExclusive(True)
        self.stack = QStackedWidget()
        for i, (label, tip) in enumerate((("CRATE", "which records may play: genres, moods and tags in / out, the tempo and energy ranges, vocals"),
                                          ("LEANS", "weighted words that rank the records the crate lets through"),
                                          ("STORY", "the chapters: a share of the night, an energy and a tempo course each, their own leans"))):
            b = _small(QPushButton(label))
            b.setCheckable(True)
            b.setChecked(i == 0)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c, k=i: self.stack.setCurrentIndex(k))
            self.sec_group.addButton(b, i)
            secs.addWidget(b)
        secs.addStretch(1)
        self.warn_row = QHBoxLayout()
        self.warn_row.setSpacing(3)
        secs.addLayout(self.warn_row)
        self.warn_btns = []
        v.addLayout(secs)
        self._crate_sa = self._scroll(self._build_crate())
        self._crate_sa.viewport().installEventFilter(self)   # (its size decides how many chips show)
        self.stack.addWidget(self._crate_sa)
        self.stack.addWidget(self._scroll(self._build_leans()))
        self.stack.addWidget(self._scroll(self._build_story()))
        v.addWidget(self.stack, 1)
        # KEYS: placed by the tab in the dial grid (one line, like a dial)
        self.keys_cell = QWidget()
        kr = QHBoxLayout(self.keys_cell)
        kr.setContentsMargins(0, 0, 0, 0)
        kr.setSpacing(4)
        kh = _dim("KEYS", KEYS_HEAD_TIP)
        kh.setFixedWidth(64)
        kr.addWidget(kh)
        self.keys_group = QButtonGroup(self)
        self.keys_group.setExclusive(True)
        self.keys_btns = {}
        for m in KEYS_MODES:
            b = _small(QPushButton(m))
            b.setCheckable(True)
            b.setChecked(m == "veto")
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setToolTip(KEYS_TIPS[m])
            b.clicked.connect(lambda _c, k=m: self._keys(k))
            self.keys_group.addButton(b)
            kr.addWidget(b)
            self.keys_btns[m] = b
        self.set_enabled(False)

    # -- building ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _scroll(w):
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(w)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setMinimumHeight(60)                     # (the tab stays one screen high: the section scrolls instead)
        return sa

    def _build_crate(self):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(2, 2, 2, 2)
        g.setHorizontalSpacing(6)
        g.setVerticalSpacing(3)
        self.chip_grids = {}
        # (the ranges first: on a short window they are what shows without scrolling; the chips wrap below)
        r = 0
        rng = QHBoxLayout()
        rng.setSpacing(4)
        self.bpm_lo, self.bpm_hi = QDoubleSpinBox(), QDoubleSpinBox()
        for s, tip in ((self.bpm_lo, "the slowest tempo the crate takes (bpm)"), (self.bpm_hi, "the fastest tempo the crate takes (bpm)")):
            s.setRange(40.0, 220.0)
            s.setDecimals(1)
            s.setSingleStep(1.0)
            s.setKeyboardTracking(False)
            s.setFixedWidth(64)
            s.setToolTip(tip)
            s.valueChanged.connect(self.touch)
        self.en_lo, self.en_hi = QSpinBox(), QSpinBox()
        for s, tip in ((self.en_lo, "the lowest energy rank the crate takes (0..100)"), (self.en_hi, "the highest energy rank (0..100)")):
            s.setRange(0, 100)
            s.setKeyboardTracking(False)
            s.setFixedWidth(56)
            s.setToolTip(tip)
            s.valueChanged.connect(self.touch)
        rng.addWidget(self.bpm_lo)
        rng.addWidget(_dim("–"))
        rng.addWidget(self.bpm_hi)
        rng.addWidget(_dim("bpm"))
        rng.addSpacing(14)
        rng.addWidget(_dim("energy", "the energy range, as the library's energy rank (0 the calmest, 100 the hottest)"))
        rng.addWidget(self.en_lo)
        rng.addWidget(_dim("–"))
        rng.addWidget(self.en_hi)
        rng.addStretch(1)
        g.addWidget(_dim("tempo", "the tempo range (bpm)", w=52), r, 0)
        g.addLayout(rng, r, 1)
        r += 1
        vr = QHBoxLayout()
        vr.setSpacing(4)
        self.voc_group = QButtonGroup(self)
        self.voc_group.setExclusive(True)
        self.voc_btns = {}
        for k in VOCALS:
            b = _small(QPushButton(k))
            b.setCheckable(True)
            b.setToolTip(VOCALS_TIPS[k])
            b.clicked.connect(self.touch)
            self.voc_group.addButton(b)
            self.voc_btns[k] = b
            vr.addWidget(b)
        vr.addSpacing(14)
        # the crate's SOURCE (DJ_SETS.md section 2): the whole library or one of your playlists - applies on pick
        vr.addWidget(_dim("from", "where the crate's records come from: the whole library or one of your playlists (the Set "
                                  "tab's saved lists); the top bar's songs box only feeds the DJ's gap picks in a planned night"))
        from PyQt6.QtWidgets import QComboBox
        self.src_box = QComboBox()
        self.src_box.addItem("whole library", None)
        self.src_box.setToolTip("the crate's source: the whole library or a playlist; applies on pick")
        self.src_box.setMinimumWidth(120)
        self.src_box.activated.connect(self.touch)
        vr.addWidget(self.src_box, 1)
        g.addWidget(_dim("vocals", "how much the records may sing, measured from their stems (never the ML vocal axis)", w=52), r, 0)
        g.addLayout(vr, r, 1)
        r += 1
        self.widened_lbl = _dim("")
        self.widened_lbl.setWordWrap(True)
        self.widened_lbl.setVisible(False)           # (only a "like these" theme has words here: no empty row otherwise)
        g.addWidget(self.widened_lbl, r, 0, 1, 2)
        r += 1
        for key, tip in (("genres", "genres: a click = in (a record must carry one of the lit ones), again = out (no record "
                                    "carrying it), again = off; right-click = out"),
                         ("moods", "moods (any of): only records carrying one of the lit moods; none lit = any mood"),
                         ("tags", "tags: a click = in (a record must carry every lit one), again = out, again = off; right-click = out"),
                         ("groups", "yours: your own categories - the music folders and the setlists, by name (any of): only "
                                    "records in one of the lit ones; none lit = any")):
            g.addWidget(_dim("yours" if key == "groups" else key, tip, w=52), r, 0, Qt.AlignmentFlag.AlignTop)
            cg = FlowLayout(3)
            self.chip_grids[key] = cg
            g.addLayout(cg, r, 1)
            r += 1
        g.setRowStretch(r, 1)
        g.setColumnStretch(1, 1)
        return w

    def _build_leans(self):
        w = QWidget()
        self.lean_box = QVBoxLayout(w)
        self.lean_box.setContentsMargins(2, 2, 2, 2)
        self.lean_box.setSpacing(3)
        add = _small(QPushButton("+ lean"), w=80)
        add.setToolTip("add a weighted word")
        add.clicked.connect(lambda: (self._add_lean("", 1.0), self.touch()))
        self.lean_box.addWidget(add)
        self.lean_box.addStretch(1)
        self.lean_rows = []
        return w

    def _build_story(self):
        w = QWidget()
        self.story_box = QVBoxLayout(w)
        self.story_box.setContentsMargins(2, 2, 2, 2)
        self.story_box.setSpacing(0)
        hd = QHBoxLayout()
        hd.setSpacing(4)
        hd.addSpacing(52)
        hd.addWidget(_dim("chapter"), 1)
        hd.addWidget(_dim("share", w=52))
        hd.addSpacing(4)
        hd.addWidget(_dim("energy from → to", "the energy course through the chapter", w=110))
        hd.addSpacing(4)
        hd.addWidget(_dim("tempo from → to", "the tempo course, as percentiles of the crate's tempos", w=110))
        hd.addSpacing(28)
        self.story_box.addLayout(hd)
        add = _small(QPushButton("+ chapter"), w=100)
        add.setToolTip("add a chapter at the end of the night")
        add.clicked.connect(self._new_chapter)
        self.story_box.addWidget(add)
        self.story_box.addStretch(1)
        self.ch_rows = []
        return w

    # -- the vocabulary (the library's real words) -----------------------------------------------------------------------
    def set_vocab(self, genres, moods, tags, groups=()):
        """[(word, n)] per group, most common first (the library the tab plays from); `groups`: your folders and setlists
        by name (lib/dj/setplanner.vocab)."""
        self.vocab = {"genres": list(genres), "moods": list(moods), "tags": list(tags), "groups": list(groups)}
        seen = []
        for grp in ("genres", "moods", "tags"):
            seen += [w for w, _n in self.vocab[grp]]
        from lib.dj.setplanner import GROUP_MARK
        seen += [GROUP_MARK + w.lower() for w, _n in self.vocab["groups"]]     # (a folder / setlist as a lean: "@groovy")
        self.completer_words = sorted(set(seen))
        from PyQt6.QtCore import QStringListModel
        self.completer_model = QStringListModel(self.completer_words, self)   # (one model for every row's completer)
        if self._base is not None:
            self.set_theme(self.theme(), emit=False)

    def set_sources(self, names):
        """The playlists a crate may come from (the Set tab's saved lists), as [(label, name)]."""
        cur = self.src_box.currentData()
        sig = tuple(names)
        if sig == getattr(self, "_src_sig", None):
            return
        self._src_sig = sig
        self.src_box.blockSignals(True)
        self.src_box.clear()
        self.src_box.addItem("whole library", None)
        for label, name in names:
            self.src_box.addItem(label, name)
        self.src_box.blockSignals(False)
        self._set_source(cur)

    def _set_source(self, name):
        self.src_box.blockSignals(True)
        i = self.src_box.findData(name) if name else 0
        if i < 0:
            self.src_box.addItem(f"{name} (not found)", name)   # (a theme's playlist this library does not hold: said, kept)
            i = self.src_box.count() - 1
        self.src_box.setCurrentIndex(i)
        self.src_box.blockSignals(False)

    def completer(self, parent):
        if not self.completer_words:
            return None
        comp = QCompleter(self.completer_model, parent)
        comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        return comp

    def _fill_chips(self, key, states):
        grid = self.chip_grids[key]
        counts = dict(self.vocab[key])
        words = [w for w, _n in self.vocab[key]]
        lim = self._lim[key] + (2 * self.LIMITS[key] if self._more[key] else 0)
        shown = [w for w in states if w not in words[:lim]] + words[:lim]     # (the theme's own words always show)
        if [w for w in self.chips[key] if w != "\u0000more"] == shown:
            for wd in shown:                           # (the same words: only their states change - no widgets rebuilt)
                self.chips[key][wd].set_state(states.get(wd, ""))
            return
        # a chip once built is kept (hidden when its word does not show): a new chip costs its style polish, and a window
        # resized to and fro, or a theme picked at 1600x1000, would build a hundred of them in one go (a 120 ms stall)
        while grid.count():
            grid.takeAt(0)
        pool = self._pool[key]
        keep = set(shown)
        for wd, c in pool.items():
            if wd not in keep and not c.isHidden():
                c.setVisible(False)
        self.chips[key] = {}
        tri = key not in ("moods", "groups")
        for wd in shown:
            n = counts.get(wd)
            tip = (f"{n} records carry ‘{wd}’" if n is not None else f"‘{wd}’ (not among the library's common words)") + \
                ("\nclick: in → out → off · right-click: out" if tri else "\nclick: in / off")
            c = pool.get(wd)
            if c is None:
                c = pool[wd] = Chip(wd, tip, tri, self.touch)
            else:
                c.setToolTip(tip)
            c.set_state(states.get(wd, ""))
            grid.addWidget(c)
            if c.isHidden():
                c.setVisible(True)
            self.chips[key][wd] = c
        hidden = len(words) - len([w for w in words if w in self.chips[key]])
        m = self._more_btn.get(key)
        if hidden > 0 or self._more[key]:
            if m is None:
                m = self._more_btn[key] = _small(QPushButton())
                m.clicked.connect(lambda _c, k=key: self._toggle_more(k))
            m.setText(f"+{hidden}" if not self._more[key] else "fewer")
            m.setToolTip("more of the library's words" if not self._more[key] else "fewer words")
            grid.addWidget(m)
            if m.isHidden():
                m.setVisible(True)
            self.chips[key]["\u0000more"] = m
        elif m is not None and not m.isHidden():
            m.setVisible(False)
        grid.invalidate()

    def _toggle_more(self, key):
        th = self.theme()
        self._more[key] = not self._more[key]
        self.set_theme(th, emit=False)

    # -- the chips sized to the room (a 1600x1000 CRATE showed 200 px of nothing under "+64") -------------------------------
    def eventFilter(self, obj, ev):
        if obj is self._crate_sa.viewport() and ev.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self._fit_timer.start()
        return super().eventFilter(obj, ev)

    def _flow_height(self, words, width, wd_of, h, sp=3):
        """The height FlowLayout gives `words` at `width` (its own wrapping rule)."""
        x, rows = 0, 1
        for wd in words:
            cw = wd_of(wd)
            if x > 0 and x + cw > width:
                x, rows = 0, rows + 1
            x += cw + sp
        return rows * h + (rows - 1) * sp

    def room_limits(self):
        """The most common words per group that fit the CRATE pane unscrolled, scaled up together from LIMITS (the floor).
        None when the pane is not laid out yet (hidden, or no chips to measure)."""
        sa, grids = self._crate_sa, self.chip_grids
        vp = sa.viewport()
        top, width = grids["genres"].geometry().top(), grids["genres"].geometry().width()
        if top > 0 and width > 40:
            self._room_geo = (vp.size(), top, width)
        elif getattr(self, "_room_geo", None) and self._room_geo[0] == vp.size():
            _vs, top, width = self._room_geo          # (a refill just reset the chip layouts, not laid out again yet: same pane)
        sample = [c for d in self.chips.values() for c in d.values() if isinstance(c, Chip)]
        if not vp.isVisible() or vp.height() <= 0 or width <= 40 or not sample or top <= 0:
            return None
        if not sa.verticalScrollBar().isVisible():
            width -= sa.verticalScrollBar().sizeHint().width()   # (as if the bar were there: a refill that adds it must not flip the count)
        fm = sample[0].fontMetrics()
        hint = [(c.sizeHint().width(), fm.horizontalAdvance(c.word)) for c in sample]
        pad = max(hint, key=lambda t: t[1])
        pad = pad[0] - pad[1]                          # (the widest word: its face is not held up by a minimum width)
        minw = min(t[0] for t in hint)
        h = max(c.sizeHint().height() for c in sample)
        h = max(min(h, 22), 20)                        # (_small: 20..22 px)

        def wd_of(wd):
            return max(minw, fm.horizontalAdvance(wd) + pad)
        room = vp.height() - top - 6                   # (the bottom margin and the grid's two row gaps: 2 + 3 ... one spare)
        states = {k: [w for w, c in d.items() if isinstance(c, Chip) and c.state] for k, d in self.chips.items()}

        def height(lims):
            tot = 0
            for k, cap in lims.items():
                words = [w for w, _n in self.vocab[k]]
                top_w = words[:cap]
                shown = [w for w in states[k] if w not in top_w] + top_w
                hidden = len(words) - len([w for w in words if w in shown])
                if hidden > 0:
                    shown = shown + [f"+{hidden}"]
                tot += self._flow_height(shown, width, wd_of, h) + 3
            return tot - 3
        best = dict(self.LIMITS)
        if height(best) > room:
            return best
        s = 1.0
        while s < 40.0:                                # (all groups grow together; a group the vocabulary runs out of stops)
            s += 0.25
            nxt = {k: min(len(self.vocab[k]), max(b, int(round(b * s)))) for k, b in self.LIMITS.items()}
            if nxt == best:
                if all(nxt[k] >= len(self.vocab[k]) for k in nxt):
                    break
                continue
            if height(nxt) > room:
                break
            best = nxt
        return {k: max(v, self.LIMITS[k]) for k, v in best.items()}

    NEW_PER_PASS = 12                                  # (chips built or re-shown per refill: a pane grown a lot fills over a few passes)

    def _fit_chips(self):
        lims = self.room_limits()
        if lims is None or lims == self._lim:
            return
        budget, again = self.NEW_PER_PASS, False
        for k in lims:
            if lims[k] <= self._lim[k]:
                continue                               # (fewer words: nothing is built, the chips just hide)
            words = [w for w, _n in self.vocab[k]]
            cap = self._lim[k]
            while cap < lims[k]:
                c = self._pool[k].get(words[cap]) if cap < len(words) else None
                if cap < len(words) and (c is None or c.isHidden()):     # (a new chip, or a kept one shown again: ~1 ms each)
                    if budget <= 0:
                        again = True
                        break
                    budget -= 1
                cap += 1
            lims[k] = cap
        if again:
            QTimer.singleShot(0, self._fit_chips)      # (the rest on the next pass: the Qt thread breathes between)
        if lims == self._lim:
            return
        self._lim = lims
        was = self._filling
        self._filling = True                           # (only which words show changes: their states, the theme, stay)
        try:
            for key, d in self.chips.items():
                self._fill_chips(key, {w: c.state for w, c in d.items() if isinstance(c, Chip) and c.state})
        finally:
            self._filling = was

    # -- the theme in and out --------------------------------------------------------------------------------------------
    def set_enabled(self, on):
        self.stack.setEnabled(on)
        for b in self.sec_group.buttons():
            b.setEnabled(on)
        for b in self.keys_btns.values():
            b.setEnabled(on)

    def set_theme(self, theme, emit=False):
        """Fill every widget from `theme` (None: nothing planned - the editor is off)."""
        self._filling = True
        try:
            self._base = json.loads(json.dumps(theme)) if theme else None
            th = self._base or {}
            self.set_enabled(theme is not None)
            c = th.get("crate") or {}
            gs = {w.lower(): "in" for w in c.get("genresIn") or []}
            gs.update({w.lower(): "out" for w in c.get("genresOut") or []})
            self._fill_chips("genres", gs)
            self._fill_chips("moods", {w.lower(): "in" for w in c.get("moodsIn") or []})
            ts = {w.lower(): "in" for w in c.get("tagsIn") or []}
            ts.update({w.lower(): "out" for w in c.get("tagsOut") or []})
            self._fill_chips("tags", ts)
            shown = {w.lower(): w for w, _n in self.vocab["groups"]}          # (your names as the library spells them)
            self._fill_chips("groups", {shown.get(str(w).lower(), str(w)): "in" for w in c.get("groupsIn") or []})
            bpm = (list(c.get("bpm") or [80, 135]) + [80, 135])[:2]
            en = (list(c.get("energy") or [0, 100]) + [0, 100])[:2]
            for s, val in ((self.bpm_lo, bpm[0]), (self.bpm_hi, bpm[1])):
                s.setValue(float(val))
            for s, val in ((self.en_lo, en[0]), (self.en_hi, en[1])):
                s.setValue(int(round(float(val))))
            self.voc_btns.get(c.get("vocals") or "any", self.voc_btns["any"]).setChecked(True)
            self._set_source(c.get("source"))
            # the rows are reused where they exist (a picked theme refills them; only the difference is built or dropped)
            leans = list((th.get("leans") or {}).items())
            for r in self.lean_rows[len(leans):]:
                self._drop_row(self.lean_box, r, self.lean_rows)
            for k, (wd, wt) in enumerate(leans):
                if k < len(self.lean_rows):
                    self.lean_rows[k].set(wd, wt)
                else:
                    self._add_lean(wd, wt)
            chs = list(th.get("chapters") or [])
            for r in self.ch_rows[len(chs):]:
                self._drop_row(self.story_box, r, self.ch_rows)
            for k, ch in enumerate(chs):
                if k < len(self.ch_rows):
                    self.ch_rows[k].set(ch)
                else:
                    self._add_chapter(ch)
            km = str(th.get("keys_mode") or "veto")
            self.keys_btns.get(km, self.keys_btns["veto"]).setChecked(True)
            inf = th.get("inferred") or {}
            wid = inf.get("widened") or []
            if inf:
                vu = int(inf.get("vox_unknown") or 0)
                self.widened_lbl.setText((("like these: " + " → ".join(f"{x['what']} ({x['records']})" for x in wid)) if wid
                                          else f"like these: {inf.get('crate', '?')} records without widening")
                                         + (f" · vocals: {vu} unmeasured" if vu else ""))
                self.widened_lbl.setToolTip(f"the crate around the examples, widened until it holds about twice the {inf.get('need', '?')} "
                                            f"records the night needs (target {inf.get('target', '?')}); {inf.get('story', '')}"
                                            + (f"\nno crate holds: {', '.join(inf['missing'])}" if inf.get("missing") else "")
                                            + (f"\nno stems, so their vocals are not known (the vocal rule ignores them): "
                                               f"{', '.join(inf.get('vox_unknown_titles') or [])}" if vu else ""))
                self.widened_lbl.setVisible(True)
            else:
                self.widened_lbl.setText("")
                self.widened_lbl.setVisible(False)
            self._blurb = th.get("blurb") or ""
            self.set_count(self.count_lbl.text(), self._count_tip)
        finally:
            self._filling = False
        self._fit_timer.start()                        # (the theme's own words and the like-these line take room too)
        if emit:
            self.changed.emit()

    def theme(self):
        """The widgets read back as the planner's theme (the shares renormalised)."""
        th = json.loads(json.dumps(self._base or {}))
        c = dict(th.get("crate") or {})
        st = {k: {w: ch.state for w, ch in d.items() if isinstance(ch, Chip)} for k, d in self.chips.items()}

        def lit(grp, s, field):
            """The words lit `s`, in the theme's own order, then the library's (never the chips' order: which words show
            depends on the room, and the same theme must read back the same at any window size)."""
            pos = {w.lower(): i for i, w in enumerate(c.get(field) or [])}
            rank = {w: i for i, (w, _n) in enumerate(self.vocab[grp])}
            return sorted((w for w, x in st[grp].items() if x == s),
                          key=lambda w: (pos.get(w, len(pos)), rank.get(w, len(rank)), w))
        c["genresIn"], c["genresOut"] = lit("genres", "in", "genresIn"), lit("genres", "out", "genresOut")
        c["moodsIn"] = lit("moods", "in", "moodsIn")
        c["tagsIn"], c["tagsOut"] = lit("tags", "in", "tagsIn"), lit("tags", "out", "tagsOut")
        c["groupsIn"] = lit("groups", "in", "groupsIn")
        lo, hi = sorted((self.bpm_lo.value(), self.bpm_hi.value()))
        c["bpm"] = [round(lo, 1), round(hi, 1)]
        elo, ehi = sorted((self.en_lo.value(), self.en_hi.value()))
        c["energy"] = [elo, ehi]
        c["vocals"] = next((k for k, b in self.voc_btns.items() if b.isChecked()), "any")
        c["source"] = self.src_box.currentData() or None
        th["crate"] = c
        leans = {}
        for r in self.lean_rows:
            x = r.lean()
            if x:
                leans[x[0]] = x[1]
        th["leans"] = leans
        chs = [r.chapter() for r in self.ch_rows]
        tot = sum(ch["share"] for ch in chs) or 1.0
        for ch in chs:
            ch["share"] = round(ch["share"] / tot, 4)
        th["chapters"] = chs
        th["keys_mode"] = next((k for k, b in self.keys_btns.items() if b.isChecked()), "veto")
        return th

    def normalise_shares(self):
        """The shares shown as they are planned (renormalised to 100 %), without an edit."""
        tot = sum(r.share.value() for r in self.ch_rows)
        if not tot or tot == 100:
            return
        for r in self.ch_rows:
            r.share.blockSignals(True)
            r.share.setValue(max(1, int(round(100.0 * r.share.value() / tot))))
            r.share.blockSignals(False)

    # -- rows -----------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _drop_row(box, row, rows):
        box.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        rows.remove(row)

    def _add_lean(self, word, weight):
        r = LeanRow(self, word, weight)
        self.lean_box.insertWidget(len(self.lean_rows), r)
        self.lean_rows.append(r)

    def remove_lean(self, row):
        self._drop_row(self.lean_box, row, self.lean_rows)
        self.touch()

    def _add_chapter(self, ch):
        r = ChapterRow(self, ch)
        self.story_box.insertWidget(1 + len(self.ch_rows), r)
        self.ch_rows.append(r)

    def _new_chapter(self):
        last = self.ch_rows[-1].chapter() if self.ch_rows else {"energy": [0.5, 0.5], "tempo": [0.5, 0.5]}
        e, t = last["energy"][-1], last["tempo"][-1]
        self._add_chapter({"name": f"chapter {len(self.ch_rows) + 1}", "share": 0.2, "energy": [e, e], "tempo": [t, t]})
        self.touch()

    def remove_chapter(self, row):
        if len(self.ch_rows) <= 1:
            return                                   # (a night has at least one chapter)
        self._drop_row(self.story_box, row, self.ch_rows)
        self.touch()

    def move_chapter(self, row, step):
        i = self.ch_rows.index(row)
        j = i + step
        if not 0 <= j < len(self.ch_rows):
            return
        self.ch_rows[i], self.ch_rows[j] = self.ch_rows[j], self.ch_rows[i]
        for r in self.ch_rows:
            self.story_box.removeWidget(r)
        for k, r in enumerate(self.ch_rows):
            self.story_box.insertWidget(1 + k, r)
        self.touch()

    # -- signals ------------------------------------------------------------------------------------------------------------
    def touch(self, *_):
        if not self._filling and self._base is not None:
            self.changed.emit()

    def _keys(self, mode):
        if not self._filling and self._base is not None:
            self.keys_changed.emit(mode)
            self.changed.emit()

    def set_effective(self, chapters, dial):
        """The plan's own chapter courses (NightPlan.status()['chapters']) beside the typed ones while ENERGY leans them."""
        chapters = chapters or []
        for i, r in enumerate(self.ch_rows):
            e = chapters[i].get("energy") if i < len(chapters) and len(chapters) == len(self.ch_rows) else None
            r.set_effective(e, dial)

    # -- the header ---------------------------------------------------------------------------------------------------------
    def set_count(self, text, tip=""):
        if self.count_lbl.text() != text:
            self.count_lbl.setText(text)
        self._count_tip = tip
        self.count_lbl.setToolTip("\n\n".join(x for x in (tip, self._blurb) if x))

    def set_warnings(self, items):
        """[(word, sentence, level)] with level amber | red | dim: small chips, the sentence in the tooltip."""
        sig = tuple(items)
        if sig == getattr(self, "_warn_sig", None):
            return
        self._warn_sig = sig
        for b in self.warn_btns:
            self.warn_row.removeWidget(b)
            b.setParent(None)
            b.deleteLater()
        self.warn_btns = []
        for word, sentence, level in items[:5]:
            b = _small(QPushButton(word))
            b.setProperty("warn", level)
            b.setToolTip(sentence)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            _restyle(b)
            self.warn_row.addWidget(b)
            self.warn_btns.append(b)


def warning_words(warnings, chapters):
    """The planner's warnings as short words (the sentence stays whole in the tooltip): (word, sentence, level)."""
    out = []
    for w in warnings or []:
        text = str(w.get("text") or "")
        bad = bool(w.get("bad"))
        if w.get("chapter") is not None:
            ci = int(w["chapter"])
            nm = (chapters[ci].get("name") if 0 <= ci < len(chapters) else None) or f"chapter {ci + 1}"
            word = f"{nm[:12]} {w['off']}/{w['of']} off" if w.get("off") is not None and w.get("of") else f"{nm[:14]} misses"
            out.append((word, text, "red" if bad else "amber"))
        elif w.get("alts") is not None:
            out.append(("thin crate", text, "amber"))
        elif w.get("widened") is not None:
            out.append(("crate widened", text, "amber"))       # (a "like these" crate the planner widened to finish the night)
        elif text.startswith("The crate holds"):
            out.append(("crate too small", text, "red"))
        elif text.startswith("The route runs out"):
            out.append(("runs out", text, "red"))
        elif text.startswith("Nothing in the crate blends after"):
            out.append(("dead end", text, "red"))
        elif "cannot be the record" in text:
            out.append(("refused", text, "amber"))
        elif "relaxed blend" in text or "plays to its end" in text:
            out.append(("one loose seam", text, "amber"))
        else:
            out.append(("note", text, "red" if bad else "amber"))
    return out


# ---- the night strip (drawn by ArcStrip when a plan exists) ---------------------------------------------------------------
def _hm(s):
    s = max(0, int(round(float(s or 0))))
    return f"{s // 3600}:{(s % 3600) // 60:02d}"


def strip_geometry(sp):
    """(seconds the strip spans, [(slot, x0 fraction, x1 fraction)]) of a plan status."""
    L = float(sp.get("length_s") or 1.0)
    slots = [s for s in sp.get("slots") or [] if s.get("start_s") is not None and s.get("state") != "skipped"]
    span = max([L] + [float(s.get("end_s") or s.get("start_s") or 0.0) for s in slots])
    out = []
    for s in slots:
        a = float(s["start_s"])
        b = float(s.get("end_s") or (a + float(s.get("play_s") or 240.0)))
        out.append((s, a / span, max(a + 30.0, b) / span))
    return span, out


def chapter_marks(sp):
    """{chapter index: 'amber' | 'red'} from the plan's chapter warnings (red only when more than half its records miss)."""
    marks = {}
    for w in sp.get("warnings") or []:
        if w.get("chapter") is None:
            continue
        ci = int(w["chapter"])
        lvl = "red" if w.get("bad") else "amber"
        if marks.get(ci) != "red":
            marks[ci] = lvl
    return marks


def off_marks(sp):
    """{record id: 'amber' | 'red'}: the records a chapter warning names as off what it asks for (outlined on the strip)."""
    out = {}
    for w in sp.get("warnings") or []:
        if w.get("chapter") is None:
            continue
        for i in w.get("off_ids") or []:
            if out.get(int(i)) != "red":
                out[int(i)] = "red" if w.get("bad") else "amber"
    return out


STRIP_TOP = 15.0              # the header line above the bands
STRIP_NAME_H = 12.0           # the chapter names and the warning marks along the top of the bands
BLOCK_HIT_PX = 6.0            # a record's block answers the mouse this close to it (above and below)


def strip_frame(W, H, sp, pad):
    """The strip's geometry, shared by the drawing and the tooltip: (top, bot, span, blocks, X(t), Y(energy))."""
    top, bot = STRIP_TOP, H - 3.0
    span, blocks = strip_geometry(sp)
    xw = W - 2 * pad

    def X(t):
        return pad + xw * max(0.0, min(1.0, float(t) / span))

    def Y(e):
        return bot - 4 - float(e) * (bot - top - STRIP_NAME_H - 4)
    return top, bot, span, blocks, X, Y


def paint_night(widget, qp, sp, pad):
    """Chapters as bands with their names, each record a block at its energy (solid when played, hollow ahead), the
    chapter's energy course, the playhead, amber / red marks on a chapter that misses and an outline on each record
    that misses. While a re-plan runs (sp['planning']: the last plan kept), everything under the header is dimmed."""
    from PyQt6.QtGui import QColor, QPen, QFont, QPolygonF
    from PyQt6.QtCore import QRectF, QPointF
    W, H = widget.width(), widget.height()
    top, bot, span, blocks, X, Y = strip_frame(W, H, sp, pad)
    xw = W - 2 * pad
    planning = bool(sp.get("planning"))
    if planning:
        qp.setOpacity(0.38)
    marks = chapter_marks(sp)
    offs = off_marks(sp)
    small = QFont(widget.font())
    small.setPointSizeF(7.5)
    qp.setFont(small)
    chs = sp.get("chapters") or []
    for i, c in enumerate(chs):
        x0, x1 = X(c.get("start_s") or 0.0), X(c.get("end_s") or span)
        qp.fillRect(QRectF(x0, top, max(1.0, x1 - x0), bot - top), QColor(30, 34, 44) if i % 2 == 0 else QColor(24, 26, 34))
        mk = marks.get(i)
        if mk:
            col = QColor(220, 80, 70) if mk == "red" else QColor(230, 170, 60)
            qp.fillRect(QRectF(x0, top, max(1.0, x1 - x0), 3), col)
            qp.setBrush(col)
            qp.setPen(Qt.PenStyle.NoPen)
            qp.drawPolygon(QPolygonF([QPointF(x1 - 9, top + 3), QPointF(x1 - 2, top + 3), QPointF(x1 - 5.5, top + 9)]))
        qp.setPen(QColor(150, 150, 168))
        qp.drawText(QRectF(x0 + 3, top + 1, max(1.0, x1 - x0 - 6), 11), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    str(c.get("name") or ""))
        # the chapter's energy course, thin
        e = c.get("energy") or [0.5, 0.5]
        qp.setPen(QPen(QColor(110, 140, 200, 160), 1.0, Qt.PenStyle.DashLine))
        qp.drawLine(QPointF(x0, Y(e[0])), QPointF(x1, Y(e[-1])))
    # the energy scale: faint quarter lines (0.25 / 0.5 / 0.75) across the night, labelled at the left edge
    qp.setPen(QPen(QColor(70, 74, 90), 1.0, Qt.PenStyle.DotLine))
    for q in (0.25, 0.5, 0.75):
        qp.drawLine(QPointF(pad, Y(q)), QPointF(W - pad, Y(q)))
    ns = float(sp.get("elapsed_s") or 0.0)
    bh = 8.0 if (bot - top) >= 80 else 6.0
    for s, a, b in blocks:
        x0, x1 = pad + xw * a, pad + xw * b
        e = s.get("energy")
        if e is None:
            e = 0.5
        y = Y(e)
        r = QRectF(x0 + 0.5, y - bh / 2, max(2.0, x1 - x0 - 1.5), bh)
        off = offs.get(int(s["id"])) if s.get("id") is not None else None
        if off:
            col = QColor(220, 80, 70) if off == "red" else QColor(230, 170, 60)
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.setPen(QPen(col, 1.6))
            qp.drawRect(r.adjusted(-2.0, -2.5, 2.0, 2.5))
        st = s.get("state")
        if st in ("played", "now"):
            qp.setPen(Qt.PenStyle.NoPen)
            qp.setBrush(QColor(255, 255, 255) if st == "now" else (QColor(240, 200, 90) if s.get("src") != "brain" else QColor(170, 150, 110)))
            qp.drawRect(r)
        else:
            col = QColor(240, 200, 90) if st == "next" else (QColor(150, 200, 255) if s.get("user") else QColor(125, 162, 227))
            qp.setBrush(Qt.BrushStyle.NoBrush)
            qp.setPen(QPen(col, 1.2))
            qp.drawRect(r)
    x = X(ns)
    qp.setPen(QPen(QColor(255, 255, 255), 1.4))
    qp.drawLine(QPointF(x, top - 2), QPointF(x, bot))
    qp.setOpacity(1.0)
    head = QFont(widget.font())
    head.setPointSizeF(8.5)
    qp.setFont(head)
    L = float(sp.get("length_s") or 0.0)
    if planning:
        # the last plan stays drawn (dimmed) while the new one is planned: the strip never falls back to the arc
        qp.setPen(QColor(240, 200, 90))
        qp.drawText(QRectF(pad, (top + bot) / 2 - 9, W - 2 * pad, 18), Qt.AlignmentFlag.AlignCenter, "planning…")
    qp.setPen(QColor(230, 230, 236))
    seed = sp.get("seed")
    txt = (f"NIGHT {sp.get('theme') or ''} · {_hm(ns)} of {_hm(L)}" + (f" · {sp['chapter']}" if sp.get("chapter") else "")
           + (f" · seed {seed}" if seed is not None else "") + (" · planning…" if (sp.get("pending") or planning) else "")
           + (f" · {sp['error'][:60]}" if sp.get("error") else ""))
    qp.drawText(QRectF(pad, 0, W - 2 * pad, 14), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)


def night_tip(widget, sp, rows, pos, pad):
    """The tooltip under the mouse: a record (its chapter, its time, why it is there) or a chapter (its course, its
    warning)."""
    W, H = widget.width(), widget.height()
    top, bot, span, blocks, X, Y = strip_frame(W, H, sp, pad)
    fx = (pos.x() - pad) / max(1.0, W - 2 * pad)
    hit = None
    # a record answers only near its block (its x span AND its height); the name row and the warning mark along the
    # top of a band, and the band around the blocks, answer with the chapter (the records cover the night end to end)
    if pos.y() > top + STRIP_NAME_H:
        best = None
        for s, a, b in blocks:
            if not (a - 0.004 <= fx <= b + 0.004):
                continue
            e = s.get("energy")
            dy = abs(pos.y() - Y(0.5 if e is None else e))
            if dy > BLOCK_HIT_PX:
                continue
            rank = (dy, 0 if s.get("state") in ("now", "next", "planned") else 1)
            if best is None or rank < best[0]:
                best = (rank, s)
        hit = best[1] if best else None
    offs = off_marks(sp)
    if hit is not None:
        r = (rows or {}).get(hit.get("id")) or {}
        tgt = (r.get("target") or {}).get("energy")
        e = hit.get("energy")
        lines = [f"{hit.get('title') or '?'} — {hit.get('artist') or r.get('artist') or ''}".rstrip(" —"),
                 f"{hit.get('chapter_name') or '?'} · {_hm(hit.get('start_s'))} · {hit.get('state')}"
                 + (f" · energy {float(e):.2f}" if e is not None else "") + (f" (it asks {float(tgt):.2f})" if tgt is not None else "")
                 + (f" · {hit.get('bpm'):.0f} bpm {hit.get('camelot') or ''}" if hit.get("bpm") else "")]
        why = []
        if r.get("matched"):
            why.append("leans: " + ", ".join(r["matched"][:5]))
        sm = r.get("in") or {}
        if sm:
            bits = [sm.get("kind") or ""]
            if sm.get("key_fit") is not None:
                bits.append(f"keys {sm['key_fit']:.2f}")
            if sm.get("tempo_gap_pct") is not None:
                bits.append(f"tempo {sm['tempo_gap_pct']:.1f} %")
            why.append("comes in: " + ", ".join(b for b in bits if b) + (f"; {sm['vox']}" if sm.get("vox") else ""))
        if hit.get("why"):
            why.append(str(hit["why"]))
        elif hit.get("src") == "brain":
            why.append("the DJ filled a gap")
        elif hit.get("user"):
            why.append("your record")
        if r.get("alts"):
            why.append(f"{len(r['alts'])} alternates ready")
        if hit.get("id") is not None and int(hit["id"]) in offs:
            why.append(("⚠ " if offs[int(hit["id"])] == "red" else "▲ ") + "off what its chapter asks for (see the chapter's mark)")
        return "\n".join(lines + why)
    chs = sp.get("chapters") or []
    t = fx * span
    for i, c in enumerate(chs):
        if float(c.get("start_s") or 0.0) <= t < float(c.get("end_s") or span):
            e = c.get("energy") or [0, 0]
            tb = c.get("tempo_bpm") or [None, None]
            words = [f"{c.get('name')}: {_hm(c.get('start_s'))}–{_hm(c.get('end_s'))}, energy {e[0]:.2f}→{e[-1]:.2f}"
                     + (f", {tb[0]:.0f}→{tb[-1]:.0f} bpm" if tb and tb[0] is not None else "")]
            if c.get("leans"):
                words.append("leans: " + ", ".join(c["leans"]))
            for w in sp.get("warnings") or []:
                if w.get("chapter") == i:
                    words.append(("⚠ " if w.get("bad") else "▲ ") + str(w.get("text")))
            return "\n".join(words)
    return ""
