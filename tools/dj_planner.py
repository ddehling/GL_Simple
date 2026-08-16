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
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PyQt6.QtCore import (QAbstractItemModel, Qt, QAbstractTableModel, QModelIndex, QRect,
                          QSize, QSortFilterProxyModel, QThread, QTimer,
                          QProcess, pyqtSignal)
from PyQt6.QtGui import (QColor, QPalette, QAction, QCursor, QFont,
                         QKeySequence, QShortcut)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QLineEdit, QTableView, QListWidget, QListWidgetItem, QPushButton,
    QComboBox, QStyledItemDelegate, QSplitter, QMessageBox, QInputDialog,
    QAbstractItemView, QDoubleSpinBox, QSpinBox, QSizePolicy, QStyle,
    QTabWidget, QPlainTextEdit, QSlider, QMenu, QCheckBox, QTreeWidget,
    QTreeWidgetItem, QHeaderView)

from lib.dj import resolve_music_dir
from lib.dj import setlist as SL
from lib.dj.brain import Brain, load_library
from lib.dj.db import LibraryDB
from lib.dj.planner_util import dup_keys, track_genre, track_sig
from lib.dj.suggest import suggest_followers
from lib.dj.rhythm import seam_chips
from lib.dj.themes import BUILTIN_THEMES, get_theme
from tools.dj.planner.arcstrip import ArcStrip
from tools.dj.planner.seaminspector import SeamInspector
from tools.dj.planner.waveform import WaveformView
from tools.dj.planner.mixview import MixTimeline
from tools.dj.planner.deckmon import DeckMonitor
from tools.dj.planner.player import TrackPlayer, PlanPreview

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The analysis pipeline (stage list, WSL structure handoff, headless
# runner) is shared with tools/dj/dj_analyze.py - see lib/dj/analyze.py.
from lib.dj.analyze import (build_stages as _build_stages,
                            structure_import_results
                            as _structure_import_results_db,
                            structure_mode as _structure_mode,
                            structure_wsl_command as _structure_wsl_command)


def _mono_font():
    """Monospace font for column-aligned list rows."""
    f = QFont("Consolas")
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def _clip(s, w):
    """Fixed-width column: truncate with an ellipsis, pad to w."""
    s = s or ""
    return (s[:w - 1] + "…") if len(s) > w else s.ljust(w)


def _no_width_floor(widget):
    """Stop a status/caption label from setting the WINDOW's minimum
    width. A QLabel's minimum tracks its text, QTabWidget's minimum is
    the max over pages - one long one-line status pinned the whole
    planner wider than a monitor. Ignored horizontal policy: the label
    takes whatever its row has and clips."""
    sp = widget.sizePolicy()
    sp.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
    widget.setSizePolicy(sp)
    return widget


RATE = 44100
SECTION_COLORS = {
    "intro": QColor(70, 90, 120), "outro": QColor(70, 90, 120),
    "groove": QColor(60, 120, 90), "build": QColor(190, 150, 60),
    "breakdown": QColor(90, 70, 130),
}
COLS = ["title", "artist", "bpm", "key", "dur", "energy", "genre", "type",
        "tags", "rhythm", "structure", "stems"]


# How much each transition style EXPOSES the four seam risk channels
# (key overlap, both-lows overlap, long groove overlap, short-window
# precision), 0..1 each. Mirrors plan_transition's steering rules - a
# kick clash bites the open-bass blends, swing clashes bite long
# overlaps, weak grids bite the precision hits, an exposed acapella
# bites on key. DISPLAY heuristic only (the transition-options rating);
# selection and compile use the brain's full score.
_STYLE_EXPOSURE = {
    #                key   lows  groove prec
    "long_blend":    (0.8,  1.0,  0.9,  0.3),
    "bass_swap":     (0.5,  0.2,  0.6,  0.3),
    "filter_sweep":  (0.6,  0.5,  0.7,  0.3),
    "loop_roll_exit": (0.5, 0.7,  0.8,  0.6),
    "echo_out":      (0.3,  0.3,  0.3,  0.7),
    "cut_at_drop":   (0.1,  0.1,  0.1,  1.0),
    "loop_build":    (0.3,  0.7,  0.5,  0.9),
    "stem_drum_swap": (0.5, 0.2,  0.4,  0.5),
    "acapella_out":  (1.0,  0.1,  0.2,  0.4),
    "stem_bass_swap": (0.5, 0.1,  0.6,  0.3),
    "drum_bridge":   (0.05, 0.9,  0.7,  0.8),
    "acapella_in":   (1.0,  0.1,  0.2,  0.4),
    "melody_carry":  (0.9,  0.2,  0.3,  0.3),
    "phrase_cut":    (0.05, 0.05, 0.05, 1.0),
    "spinback_cut":  (0.0,  0.05, 0.05, 0.9),
    "loop_in":       (0.4,  0.5,  0.6,  0.8),
    "breakdown_swap": (0.6, 0.3,  0.4,  0.4),
}


def style_rating(si, p, a, b, style):
    """0..1 predicted seam quality FOR ONE STYLE: the seam's measured
    physics viewed through that style's exposure profile. A fade opts out
    of beat/key physics entirely - safe, never spectacular - so it gets a
    flat 'clean handoff' score."""
    if style == "long_fade":
        return 0.62
    si, p = si or {}, p or {}
    rt = si.get("rhythm") or {}
    key_s = si.get("key_fit", 0.6)
    ka = rt.get("kick_agreement")
    lows_s = 0.7 if ka is None else ka
    if (si.get("d_off") or 0.0) > 0.025:
        lows_s = min(lows_s, 0.45)       # bass placement gap flams the lows
    groove_s = rt.get("score")
    groove_s = 0.75 if groove_s is None else groove_s
    if (rt.get("swing_delta") or 0.0) > 0.055:
        groove_s = min(groove_s, 0.35)   # swung vs straight: nothing fixes it
    conf = min(getattr(a, "bpm_conf", 0.5), getattr(b, "bpm_conf", 0.5))
    prec_s = min(1.0, max(0.0, (conf - 0.4) / 0.5))
    fl = rt.get("flam_ms")
    if fl is not None and 15.0 <= fl <= 80.0:
        prec_s = min(prec_s, 0.35)       # machine-gun near-miss window
    if (si.get("d_off") or 0.0) > 0.028:
        prec_s = min(prec_s, 0.30)       # same gate the brain applies
    ek, el, eg, ep = _STYLE_EXPOSURE[style]
    r = 1.0
    for e, s in ((ek, key_s), (el, lows_s), (eg, groove_s), (ep, prec_s)):
        r *= 1.0 - e * (1.0 - min(max(s, 0.0), 1.0))
    rate = p.get("rate") or 1.0
    r *= math.exp(-(abs(math.log(max(rate, 1e-6))) / 0.06) ** 2)
    floor = si.get("floor")
    if floor is not None and floor < 0.15 \
            and _STYLE_EXPOSURE[style][1] >= 0.5:
        r *= 0.6                         # dead air bites the open overlaps
    return max(0.0, min(1.0, r))


def energy_glyph(e):
    """Compact energy readout: number + a little bar, sorts as text too."""
    e = max(0.0, min(1.0, float(e or 0.0)))
    return f"{e:.2f} " + chr(9601 + int(e * 7 + 0.5))   # 0.00▁ .. 1.00█


def seam_tooltip(a, b, plan, si):
    """The seam EXPLAINER: verdict first, then every factor judged against
    the engine's real thresholds (the same numbers selection and style
    gating use), then what to do about problems. Shared by the set-list
    rows, the compiled plan's ↳ seam lines and the arc strip - one
    vocabulary for 'is this a good mix'."""
    p = plan or {}
    si = si or {}
    rt = si.get("rhythm") or {}
    style = p.get("style", "?")
    fade = bool(si.get("fade") or style == "long_fade")
    rate = p.get("rate") or 1.0
    stretch = abs(rate - 1.0) * 100.0
    issues, cautions = [], []

    # -- judge each factor -------------------------------------------------
    key = si.get("key_fit")
    if key is not None and not fade:
        if key < 0.5:
            issues.append("key clash")
        elif key < 0.62:
            cautions.append("key")
    if not fade:
        if stretch > 5.5:
            issues.append("stretch past the wall")
        elif stretch > 4.0:
            cautions.append("stretch")
    floor = si.get("floor")
    if floor is not None and floor < 0.15 and not fade:
        issues.append("dead air in the overlap")
    if rt and not fade and rt.get("conf", 1.0) >= 0.5:
        if rt.get("meter_clash"):
            issues.append("meter clash")
        if rt.get("kick_agreement", 1.0) < 0.35:
            (issues if style in ("long_blend", "loop_build")
             else cautions).append("kick patterns")
        if rt.get("swing_delta", 0.0) > 0.055:
            cautions.append("swing")
        fl = rt.get("flam_ms")
        if fl is not None and 15.0 <= fl <= 80.0:
            cautions.append("flam")

    lines = [f"{a.title}  →  {b.title}",
             f"style: {style}"
             + (f"  ·  {p['beats']} beats overlap"
                if p.get("beats") and not fade else "")]
    if fade:
        lines.append(
            "DELIBERATE FADE - this pair is outside beat-match physics "
            "(tempo/grid/meter/vocals), so the engine dips one out and "
            "brings the other in clean. Judge it on energy and mood "
            "continuity, not beat terms.")
    elif issues:
        lines.append("ROUGH SEAM - expect audible problems: "
                     + ", ".join(issues)
                     + (".  Also watch: " + ", ".join(cautions)
                        if cautions else "."))
    elif cautions:
        lines.append("WORKABLE - cautions: " + ", ".join(cautions)
                     + ". The style choice below already works around "
                       "what it can.")
    else:
        lines.append("CLEAN - beat-matched and compatible on every "
                     "measured axis.")

    # -- the factors, with the numbers that make them good or bad ----------
    if not fade:
        mult = si.get("mult", rt.get("mult") or 1.0)
        mtxt = {2.0: "B heard double-time", 0.5: "B heard half-time",
                0.75: "3:4 polymeter read", 1.5: "3:2 polymeter read"} \
            .get(mult)
        sp = (rate - 1) * 100
        sp = 0.0 if abs(sp) < 0.05 else sp       # no '-0.0%'
        lines.append(
            f"tempo: stretch {sp:+.1f}%"
            + (f" ({mtxt})" if mtxt else "")
            + "  [under 4% invisible · 4-5.5% audible feel · past 5.5% "
              "the groove drags]")
        if key is not None:
            lines.append(
                f"key: {key:.2f}"
                + ("  - melodies share their notes, blend sings"
                   if key >= 0.8 else
                   "  - workable, keep the overlap's melodies apart (EQ)"
                   if key >= 0.55 else
                   "  - CLASH: two melodies will fight; EQ one side out, "
                   "keep the blend short, or fade")
                + "  [0.8+ great · 0.55 workable · under 0.5 clash]")
        if rt:
            ka = rt.get("kick_agreement", 0.0)
            sw_d = rt.get("swing_delta", 0.0)
            g = rt.get("score", 0.0)
            gl = (f"groove: {g:.2f}  - kick agreement {ka:.2f}"
                  + (", swung-vs-straight clash" if sw_d > 0.055 else "")
                  + (f", closest hits {rt['flam_ms']:.0f}ms"
                     if rt.get("flam_ms") is not None
                     and 15 <= rt["flam_ms"] <= 80 else ""))
            gl += ("  [0.6+ locks together · under 0.45 blends are cut to "
                   "32 beats · kick agreement under 0.35 bans open-bass "
                   "styles]")
            lines.append(gl)
            if rt.get("regions"):
                lines.append(f"   compared {rt['regions']} (A's exit "
                             f"pattern vs B's intro pattern)")
            if rt.get("conf", 1.0) < 0.5:
                lines.append("   ? groove terms are GUESSES here - one "
                             "side's beat grid is low-confidence")
        if floor is not None:
            lines.append(
                f"overlap energy floor: {floor:.2f}"
                + ("  - near-SILENT stretch inside the blend (dead air on "
                   "the floor)" if floor < 0.15 else
                   "  - a dip; acceptable" if floor < 0.25 else
                   "  - overlap stays full")
                + "  [under 0.15 = dead air]")
        d_off = si.get("d_off")
        if d_off is not None and d_off > 0.035:
            lines.append(
                f"groove offset Δ{d_off * 1000:.0f}ms - the basslines sit "
                "differently against their grids; the punchy short styles "
                "are gated off for this pair (they'd flam) - long blends "
                "ride it out  [35ms is the gate]")
        conf = min(a.bpm_conf or 0.0, b.bpm_conf or 0.0)
        if conf < 0.7:
            lines.append(
                f"grid confidence {conf:.2f} - precision styles "
                "(cut/drop/echo) need 0.7+; below 0.5 everything fades. "
                "'Refine grids' in the Library tab can promote tracks.")
    pm = si.get("pair_mem")
    if pm is not None:
        lines.append("history: ★ this exact pair mixed well before"
                     if pm > 1.0 else
                     "history: ✖ this exact pair went rough before "
                     "(measured or thumbed down)")
    if issues and not fade:
        lines.append("fix: right-click the slot for alternatives or a "
                     "bridge track; drag to reorder; or double-click to "
                     "anchor what must stay.")
    return "\n".join(lines)


def groove_glyph(t):
    """Compact groove readout for mono-font list rows: feel + density bar.
    'str▃' = straight sparse, 'shf█' = shuffling busy, '3/4!' = confident
    waltz, ' ·  ' = no rhythm signature yet."""
    sig = getattr(t, "rhythm_sig", None)
    if not sig:
        return " ·  "
    if sig.get("meter") == 3 and sig.get("meter_conf", 0.0) >= 0.4:
        return "3/4!"
    sw = sig.get("swing", 0.5)
    feel = ("str" if sw < 0.54 else "swg" if sw < 0.62 else "shf") \
        if sig.get("swing_conf", 0.0) > 0.3 else "str"
    d = max(0.0, min(1.0, float(sig.get("density") or 0.0)))
    return feel + chr(9601 + int(d * 7 + 0.5))
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

    def __init__(self, music_dir=""):
        super().__init__()
        self.music_dir = music_dir   # stems-column tooltip (model stamp)
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
                return os.path.splitext(t.path)[1].lstrip(".").lower()
            if c == 8:
                return " ".join(t.all_tags)
            if c == 11:
                return "✓" if getattr(t, "has_stems", False) else ""
        if role == Qt.ItemDataRole.ToolTipRole and c == 11:
            if getattr(t, "has_stems", False):
                from lib.dj.stems import stem_model_of
                model = stem_model_of(self.music_dir, t.id) or "htdemucs"
                return (f"stems rendered with {model} (.stems/<id>/) - "
                        "stem_drum_swap and acapella_out can use this "
                        "track; solo them in the Analysis tab")
            return ("no stems - render from the Analysis tab (one "
                    "song) or '+ stems' in Analyze all (library-wide)")
        if role == Qt.ItemDataRole.ToolTipRole and c == 9:
            sig = getattr(t, "rhythm_sig", None)
            if not sig:
                return ("no rhythm signature yet - run the Rhythm pass "
                        "(Library tab, ⚙ Passes)")
            sw = sig.get("swing", 0.5)
            feel = ("straight" if sw < 0.54 else
                    "swung" if sw < 0.62 else "shuffle")
            return (f"groove: {feel} (swing {sw:.2f}), "
                    f"density {sig.get('density', 0.0):.2f}, "
                    f"measured from {'drum stem' if sig.get('source') == 'stem' else 'full mix'}")
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
        self.hide_excluded = False   # hide 🚫 do-not-use tracks
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

    def set_hide_excluded(self, flag):
        self.hide_excluded = bool(flag)
        self.invalidateFilter()

    def _track_matches(self, m, f_row, t_row):
        fd = m.folders[f_row]
        t = fd["tracks"][t_row]
        if self.hide_excluded and getattr(t, "excluded", False):
            return False
        if self.flat and self.folder is not None \
                and track_folder(t) != self.folder:
            return False
        if self.tag is not None and self.tag not in t.all_tags:
            return False
        hay = f"{t.title} {t.artist} {' '.join(t.all_tags)}".lower()
        if self.text in hay:
            return True
        # Tree mode surfaces whole folders whose NAME matches the search.
        return (not self.flat and self.tag is None and bool(self.text)
                and self.text in fd["name"].lower())

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
            if c == 9:
                # rhythm column sorts by pattern density (busy breaks
                # cluster together, sparse 4x4 grooves together)
                def dens(t):
                    sig = getattr(t, "rhythm_sig", None)
                    return sig.get("density", 0.0) if sig else -1.0
                return dens(ta) < dens(tb)
            if c == 11:                  # stems: rendered tracks together
                return (getattr(ta, "has_stems", False)
                        < getattr(tb, "has_stems", False))
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
            if self.tag is None and not self.hide_excluded \
                    and (not self.text or self.text in name.lower()):
                return True
            return any(self._track_matches(m, row, r)
                       for r in range(len(m.folders[row]["tracks"])))
        f_row = parent.row()
        if not self.flat and self.folder is not None \
                and m.folder_name(f_row) != self.folder:
            return False
        return self._track_matches(m, f_row, row)


class RhythmDelegate(QStyledItemDelegate):
    """Library 'rhythm' cell: the kick pattern at a glance. Low-band 16th
    steps as bars (2 bars of pattern), percussion (max of mid/high) as dim
    ticks along the top, a swing dot when the groove is measurably swung.
    Sorts by density (see LibraryProxy.lessThan); numbers live in the
    tooltip."""

    def paint(self, p, option, index):
        t = index.model().data(index, Qt.ItemDataRole.UserRole)
        if t is None:
            return
        if option.state & QStyle.StateFlag.State_Selected:
            p.fillRect(option.rect, option.palette.highlight())
        sig = getattr(t, "rhythm_sig", None)
        if not sig:
            return
        r = option.rect.adjusted(2, 3, -14, -3)
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        low, mid, high = sig["low"], sig["mid"], sig["high"]
        n = len(low)
        sw = r.width() / n
        top_h = r.height() * 0.3
        for i in range(n):
            perc = max(float(mid[i]), float(high[i]))
            if perc > 0.3:
                p.fillRect(QRect(int(r.x() + i * sw), r.y(),
                                 max(int(sw) - 1, 1),
                                 max(int(top_h * perc), 1)),
                           QColor(170, 170, 180, 120))
            v = float(low[i])
            if v > 0.1:
                h = max(int((r.height() - top_h) * v), 2)
                p.fillRect(QRect(int(r.x() + i * sw),
                                 r.y() + r.height() - h,
                                 max(int(sw) - 1, 1), h),
                           QColor(90, 150, 220, 90 + int(160 * min(v, 1.0))))
        swing = sig.get("swing", 0.5)
        if sig.get("swing_conf", 0.0) > 0.3 and swing > 0.56:
            p.setPen(QColor(230, 180, 90))
            p.drawText(option.rect.adjusted(0, 0, -2, 0),
                       Qt.AlignmentFlag.AlignRight
                       | Qt.AlignmentFlag.AlignVCenter, "s")
        p.restore()


def _rhythm_row_payload(t, prev=None, rt=None):
    """Set-row rhythm strip payload: the incoming track's step pattern AS
    HEARD at the seam (in-region view, resampled+rotated to the previous
    track's step frame by the scorer's chosen alignment), each 16th
    classified against the previous track's exit pattern:
        1 = hits coincide (locks)    0 = neutral
       -1 = clash (this track slams where the last one is empty)
       -2 = hole (the last track slams here, this one is silent)
    prev/rt None -> plain pattern, no classification (opener, fade, or
    unmeasured seam)."""
    from lib.dj.rhythm import aligned_pattern, region_view
    sig = getattr(t, "rhythm_sig", None)
    if sig is None:
        return None
    vb = region_view(sig, "in")
    mult = (rt or {}).get("mult") or 1.0
    rot = (rt or {}).get("rot") or 0
    bl = aligned_pattern(vb, "low", mult, rot)
    perc = np.maximum(aligned_pattern(vb, "mid", mult, rot),
                      aligned_pattern(vb, "high", mult, rot))
    n = len(bl)
    cls = [0] * n
    prev_sig = getattr(prev, "rhythm_sig", None) if prev is not None else None
    if prev_sig is not None and rt is not None:
        al = region_view(prev_sig, "out")["low"]
        for i in range(min(n, len(al))):
            b, a = float(bl[i]), float(al[i])
            if b > 0.55 and a < 0.2:
                cls[i] = -1
            elif b > 0.35 and a > 0.35:
                cls[i] = 1
            elif a > 0.55 and b < 0.1:
                cls[i] = -2
    return {"low": [float(x) for x in bl],
            "perc": [float(x) for x in perc], "cls": cls}


class SetRowDelegate(QStyledItemDelegate):
    """The set table's RHYTHM column: the incoming track's step pattern
    colored by HOW IT MIXES with the preceding song at the compiled seam:
    green = kicks coincide, red = clash (a kick where the last track has
    none), red baseline tick = hole (the last track kicks, this one is
    silent), blue = neutral / no seam context. Strip only - the other
    columns are ordinary text with user-adjustable header widths."""

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        return QSize(280, max(s.height(), 24))

    def paint(self, p, option, index):
        if option.state & QStyle.StateFlag.State_Selected:
            p.fillRect(option.rect, option.palette.highlight())
        pay = index.data(SetTab.RHY_ROLE)
        if not pay:
            return
        rr = option.rect.adjusted(3, 3, -3, -3)
        low, perc, cls = pay["low"], pay["perc"], pay["cls"]
        n = max(len(low), 1)
        sw = rr.width() / n
        top_h = rr.height() * 0.3
        NEUT = QColor(90, 150, 220)
        CLASH = QColor(235, 100, 100)
        MATCH = QColor(110, 205, 145)
        p.save()
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(len(low)):
            x = int(rr.x() + i * sw)
            wpx = max(int(sw) - 1, 1)
            pv = float(perc[i])
            if pv > 0.3:
                p.fillRect(QRect(x, rr.y(), wpx, max(int(top_h * pv), 1)),
                           QColor(170, 170, 180, 110))
            v = float(low[i])
            c = cls[i]
            if v > 0.1:
                h = max(int((rr.height() - top_h) * v), 2)
                col = QColor(CLASH if c == -1 else
                             MATCH if c == 1 else NEUT)
                col.setAlpha(100 + int(155 * min(v, 1.0)))
                p.fillRect(QRect(x, rr.y() + rr.height() - h, wpx, h), col)
            elif c == -2:
                p.fillRect(QRect(x, rr.y() + rr.height() - 2, wpx, 2),
                           QColor(235, 100, 100, 170))
        p.restore()


class SetListView(QTreeWidget):
    """The set as a flat table with USER-ADJUSTABLE columns (drag the
    header edges): track columns, the inbound-seam estimate, and the
    rhythm strip (last column, stretches to whatever width is left - drag
    the others smaller to grow it). Rows drag-reorder as whole entries;
    dropping INTO a row is disabled (a set has no nesting)."""
    HEADERS = ["", "title", "artist", "genre", "bpm", "key", "energy",
               "groove", "st", "seam", "rhythm"]
    RHY_COL = 10
    SEAM_COL = 9
    STEM_COL = 8
    reordered = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setColumnCount(len(self.HEADERS))
        self.setHeaderLabels(self.HEADERS)
        self.setRootIsDecorated(False)
        self.setIndentation(0)
        self.setUniformRowHeights(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        hdr = self.header()
        hdr.setStretchLastSection(True)          # rhythm takes the rest
        for i, w in enumerate((26, 230, 120, 96, 44, 40, 64, 52, 30, 64)):
            self.setColumnWidth(i, w)
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

    # QListWidget-flavored helpers so the SetTab call sites stay readable.
    def item(self, i):
        return self.topLevelItem(i)

    def count(self):
        return self.topLevelItemCount()

    def row(self, it):
        return self.indexOfTopLevelItem(it)

    def currentRow(self):
        it = self.currentItem()
        return self.indexOfTopLevelItem(it) if it is not None else -1

    def setCurrentRow(self, i):
        it = self.topLevelItem(i)
        if it is not None:
            self.setCurrentItem(it)

    def dropEvent(self, ev):
        # InternalMove on a QTreeWidget doesn't reliably fire rowsMoved
        # (drops decompose into remove+insert) - signal AFTER the drop
        # lands so the tab rebuilds entries from the item order.
        super().dropEvent(ev)
        self.reordered.emit()


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
        self._structure_proc = None
        self._stems_proc = None
        # One-button pipeline state (Analyze all): remaining stages, the
        # stage subprocess, and the end-of-run report.
        self._pipe = []
        self._pipe_proc = None
        self._pipe_total = 0
        self._pipe_prefix = ""
        self._pipe_stage_name = ""
        self._pipe_skipped = []
        self._pipe_failed = []
        v = QVBoxLayout(self)

        top = QHBoxLayout()
        # THE one-button path: every pass, in order, each skipping work
        # already done - so this is cheap to re-run any time. The
        # individual passes live behind the "Passes" toggle (expert row).
        self.analyze_btn = QPushButton("Analyze all")
        self.analyze_btn.setToolTip(
            "Run the whole analysis pipeline in order: scan (+ vocal pass "
            "for new tracks), chroma backfill, stems (if checked - runs "
            "early so the vocal pass can reuse the separation), fine "
            "vocal-curve upgrade, MusicBrainz enrichment, ML mood, ML "
            "structure. Every stage skips tracks that are already done, "
            "so re-running costs seconds when the library hasn't changed. "
            "Stages whose optional deps aren't installed are skipped and "
            "reported at the end. This is the everyday button; individual "
            "passes are under the gear.")
        self.analyze_btn.clicked.connect(self.run_analyze_all)
        top.addWidget(self.analyze_btn)
        self.stems_chk = QCheckBox("+ stems")
        self.stems_chk.setToolTip(
            "Include the stem render pass in 'Analyze all' (~20 MB of disk "
            "per track under .stems/; unlocks the stem transition styles). "
            "Off by default because of the disk cost.")
        top.addWidget(self.stems_chk)
        self.passes_toggle = QPushButton("⚙ Passes")
        self.passes_toggle.setCheckable(True)
        self.passes_toggle.setToolTip(
            "Show the individual analysis passes - for re-running one "
            "stage alone or forcing re-analysis. 'Analyze all' runs "
            "everything in order and skips finished work.")
        self.passes_toggle.toggled.connect(
            lambda on: self.passes_row.setVisible(on))
        top.addWidget(self.passes_toggle)
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
        self.hide_excl_chk = QCheckBox("Hide 🚫")
        self.hide_excl_chk.setToolTip(
            "Hide do-not-use tracks from the list. They stay in the "
            "library and keep their flag; untick to see them again.")
        self.hide_excl_chk.toggled.connect(
            lambda on: self.proxy.set_hide_excluded(on))
        top.addWidget(self.hide_excl_chk)
        v.addLayout(top)

        # -- expert row: every pass individually (hidden behind ⚙ Passes) --
        self.passes_row = QWidget()
        pr = QHBoxLayout(self.passes_row)
        pr.setContentsMargins(0, 0, 0, 0)
        self.scan_btn = QPushButton("Scan library")
        self.scan_btn.setToolTip("Analyze new/changed tracks and any whose "
                                 "analysis is older than the current feature "
                                 "set (incremental).")
        self.scan_btn.clicked.connect(lambda: self.run_scan(force=False))
        pr.addWidget(self.scan_btn)
        self.rescan_btn = QPushButton("Rescan all")
        self.rescan_btn.setToolTip("Force a full re-analysis of EVERY track "
                                   "(ignores the up-to-date check). Use after "
                                   "a big library move or to be certain every "
                                   "track has the latest features.")
        self.rescan_btn.clicked.connect(lambda: self.run_scan(force=True))
        pr.addWidget(self.rescan_btn)
        self.refine_btn = QPushButton("Refine grids")
        self.refine_btn.setToolTip(
            "Re-run beat-grid analysis on unchanged tracks whose grid "
            "confidence sits below 0.75 (one attempt per analysis version; "
            "stubbornly low tracks aren't re-chewed every press). Promoting "
            "past 0.70 unlocks the precision transition styles.")
        self.refine_btn.clicked.connect(lambda: self.run_scan(refine=True))
        pr.addWidget(self.refine_btn)
        self.chroma_btn = QPushButton("Chroma")
        self.chroma_btn.setToolTip(
            "Backfill the 12-bin harmonic fingerprint for tracks that "
            "lack it (decode + STFT only - fast, no GPU). New scans "
            "compute it inline; this exists for old libraries.")
        self.chroma_btn.clicked.connect(self.run_chroma)
        pr.addWidget(self.chroma_btn)
        self.rhythm_btn = QPushButton("Rhythm")
        self.rhythm_btn.setToolTip(
            "Backfill the beat-sync rhythm signature (kick/snare/hat step "
            "patterns, swing, density) for tracks that lack it, and upgrade "
            "mix-derived signatures to drum-stem-derived ones where stems "
            "exist. Decode + fold only - fast, no GPU. Powers the seam "
            "rhythm chips, the seam inspector and groove-aware selection.")
        self.rhythm_btn.clicked.connect(self.run_rhythm)
        pr.addWidget(self.rhythm_btn)
        self.revocals_btn = QPushButton("Vocal curves")
        self.revocals_btn.setToolTip(
            "Re-measure tracks whose vocal curve is still at the old "
            "coarse 24s resolution (demucs, GPU). Fine curves let seam "
            "planning dodge individual vocal lines.")
        self.revocals_btn.clicked.connect(self.run_revocals)
        pr.addWidget(self.revocals_btn)
        self.enrich_btn = QPushButton("Enrich (MusicBrainz)")
        self.enrich_btn.setToolTip(
            "Fetch genre, release year/era and label from MusicBrainz for "
            "every track that lacks it. Genres become tags that steer "
            "selection and the copilot. ~1 track/sec; background, resumable.")
        self.enrich_btn.clicked.connect(self.run_enrich)
        pr.addWidget(self.enrich_btn)
        self.mood_btn = QPushButton("Mood (ML)")
        self.mood_btn.setToolTip(
            "Run the Music2Emo model over every un-scored track to get real "
            "valence/arousal and mood tags (dark, party, melancholic, epic...). "
            "Upgrades danceable/dark/uplifting tags and valence steering from "
            "heuristic to ML. Needs a Music2Emotion clone; ~3-5s/track on GPU, "
            "background and resumable.")
        self.mood_btn.clicked.connect(self.run_mood)
        pr.addWidget(self.mood_btn)
        self.structure_btn = QPushButton("Structure (ML)")
        self.structure_btn.setToolTip(
            "Run the allin1 structure model over every unlabeled track to "
            "get chorus/verse/bridge/outro segment labels. Seam planning "
            "then exits on real outros and never enters a track mid-chorus. "
            "Runs natively or through WSL (requirements-dj-structure.txt); "
            "a few s/track on GPU, background and resumable.")
        self.structure_btn.clicked.connect(self.run_structure)
        pr.addWidget(self.structure_btn)
        self.stems_btn = QPushButton("Stems (render)")
        self.stems_btn.setToolTip(
            "Pre-separate every track into drums/bass/other/vocals stems "
            "(~20 MB/track under .stems/). Unlocks the stem_drum_swap and "
            "acapella_out transition styles - drums-only entries and vocal "
            "tails over the incoming instrumental. Needs torch + demucs "
            "(requirements-dj-vocals.txt); ~10-30s/track on GPU, background "
            "and resumable.")
        self.stems_btn.clicked.connect(self.run_stems)
        pr.addWidget(self.stems_btn)
        pr.addStretch(1)
        self.passes_row.setVisible(False)
        v.addWidget(self.passes_row)

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

        self.model = LibraryTreeModel(planner.music_dir)
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
        # Delegates MUST be parented (or referenced): the view does NOT
        # take ownership, and an unparented temporary gets GC'd - Qt then
        # calls a dead object and the whole planner dies with an access
        # violation at first paint (bisected 2026-07-21; one lone
        # temporary delegate had only survived here by GC luck).
        self.table.setItemDelegateForColumn(9, RhythmDelegate(self.table))
        self.table.setItemDelegateForColumn(10, StripDelegate(self.table))
        for i, w in enumerate((250, 120, 52, 44, 50, 62, 100, 46, 150, 110,
                               260, 46)):
            self.table.setColumnWidth(i, w)
        self.table.doubleClicked.connect(
            lambda _: self._open_analysis())
        self.table.selectionModel().selectionChanged.connect(
            self._extend_folder_selection)
        # Right-click: add-to-set (same as the button/drag path) + the
        # do-not-use toggle from the buttons below.
        add_act = QAction("➕ Add to set", self)
        add_act.triggered.connect(self._add_selected)
        self.table.addAction(add_act)
        stem_act = QAction("▤ Render stems for selected", self)
        stem_act.triggered.connect(self._stems_selected)
        self.table.addAction(stem_act)
        sep = QAction(self)
        sep.setSeparator(True)
        self.table.addAction(sep)
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
        # ANALYSIS COVERAGE: which passes still have gaps - "is my library
        # ready?" without dropping to each CLI's --stats.
        self.coverage_lbl = _no_width_floor(QLabel(""))
        self.coverage_lbl.setToolTip(
            "Analysis passes with missing tracks. Run '⟳ Analyze all' "
            "(each stage skips what's already done) or the individual "
            "pass under ⚙ Passes. Stems are opt-in ('+ stems') and gate "
            "the stem transition styles.")
        bot.addWidget(self.coverage_lbl)
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

    def _stems_selected(self):
        """Queue a stem render for every selected track (the planner-wide
        queue runs them one at a time; the bottom status bar narrates).
        Tracks that already have stems are skipped - re-render those
        per-song from the Analysis tab if you switched models."""
        ts = self.selected_tracks()
        if not ts:
            return
        todo = [t for t in ts if not getattr(t, "has_stems", False)]
        queued = sum(
            1 for t in todo
            if self.planner.render_stems(t, on_status=self.scan_lbl.setText))
        skipped = len(ts) - len(todo)
        note = (f"queued {queued} track(s) for stems" if todo
                else "all selected tracks already have stems")
        if skipped and todo:
            note += f" ({skipped} already rendered - skipped)"
        self.scan_lbl.setText(note + "; progress in the bottom status bar")

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

    def _select_playing_in_list(self, t):
        """Highlight + scroll to the now-playing track so Back/Next (and Play)
        keep the list selection in sync with what's audible. No-op if the
        track is currently filtered out of view."""
        from PyQt6.QtCore import QItemSelectionModel
        m = self.model
        for f_row, fd in enumerate(m.folders):
            for t_row, tr in enumerate(fd["tracks"]):
                if tr.id == t.id:
                    parent = m.index(f_row, 0)
                    pidx = self.proxy.mapFromSource(m.index(t_row, 0, parent))
                    if pidx.isValid():
                        self._extending_sel = True   # skip folder-expand logic
                        self.table.selectionModel().setCurrentIndex(
                            pidx,
                            QItemSelectionModel.SelectionFlag.ClearAndSelect
                            | QItemSelectionModel.SelectionFlag.Rows)
                        self._extending_sel = False
                        self.table.scrollTo(pidx)
                    return

    def _play_track(self, t, start_s=0.0):
        from tools.dj.planner.player import TrackPlayer  # noqa: F401
        self.planner.claim_playback("library")
        self._select_playing_in_list(t)
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

        def _go(samples, tr=t, s0=start_s):
            if isinstance(samples, str):
                self.play_lbl.setText(f"decode failed: {samples[:60]}")
                return
            self.planner.claim_playback("library")
            self.lib_player.load(samples)
            if s0 > 0:
                dur = len(samples) / 44100.0
                self.lib_player.seek(min(s0, max(dur - 10.0, 0.0)))
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
        try:
            cov = self.planner.db.coverage_counts()
            total = cov.get("tracks", (0, 0))[1]
            gaps = [f"{k} {d}/{tt}" for k, (d, tt) in cov.items()
                    if k != "tracks" and d < tt]
            n_stems = sum(1 for t in tracks
                          if getattr(t, "has_stems", False))
            if total and n_stems < total:
                gaps.append(f"stems {n_stems}/{total}")
            n_err = self.planner.db.error_count()
            if n_err:
                gaps.insert(0, f"⚠ {n_err} failed analysis "
                               "(hidden - Scan library retries them)")
            self.coverage_lbl.setText(
                "analysis ✓ complete" if not gaps
                else "gaps: " + " · ".join(gaps))
        except Exception:
            self.coverage_lbl.setText("")
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
    def run_scan(self, force=False, refine=False):
        if self._proc is not None:
            return
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "dj", "dj_scan.py")
        args = [script, "--dir", self.planner.music_dir]
        if force:
            args.append("--force")
        if refine:
            args.append("--refine-grids")
        self._proc = QProcess(self)
        self._proc.finished.connect(self._scan_done)
        self._proc.start(sys.executable, args)
        self.scan_btn.setEnabled(False)
        self.rescan_btn.setEnabled(False)
        self.refine_btn.setEnabled(False)
        self.scan_lbl.setText(
            "refining low-confidence grids..." if refine
            else "rescanning all..." if force else "scanning...")
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
        self.refine_btn.setEnabled(True)
        try:
            n_err = self.planner.db.error_count()
        except Exception:
            n_err = 0
        self.scan_lbl.setText(
            f"scan complete — ⚠ {n_err} track(s) still failing analysis"
            if n_err else "scan complete")
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
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "dj", "dj_mood.py")
        self._mood_proc = QProcess(self)
        self._mood_proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self._mood_proc.readyReadStandardOutput.connect(self._mood_stdout)
        self._mood_proc.finished.connect(self._mood_done)
        self._mood_proc.start(sys.executable,
                              [script, "--dir", self.planner.music_dir])
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
            # NOTE: no mid-scan reload_library() here. The mood subprocess
            # holds a heavy torch/MERT + decoded-audio footprint; re-hydrating
            # the whole library on top of it OOM'd the GUI (MemoryError in
            # all_tracks). The pass writes to the DB continuously (resumable),
            # so we just show progress and reload ONCE at completion, after the
            # subprocess has exited and freed its memory.

    def _mood_done(self, *a):
        self._mood_proc = None
        self.mood_btn.setText("Mood (ML)")
        self.scan_lbl.setText("mood pass complete")
        try:
            self.planner.reload_library()
        except MemoryError:
            self.scan_lbl.setText("mood pass complete - reopen the planner "
                                  "to see the new tags (low memory)")

    # -- structure (allin1 ML, subprocess; native or via WSL) ----------------
    def _structure_import_results(self):
        """Fold a WSL batch's JSONL results into the DB we own (only the
        Windows side may open it - see lib/dj/analyze.py). Safe to call
        any time; imports leftovers from interrupted runs too."""
        return _structure_import_results_db(self.planner.db,
                                            self.planner.music_dir)

    def run_structure(self):
        if self._structure_proc is not None:             # toggles to stop
            self._structure_proc.kill()
            self.structure_btn.setText("stopping...")
            return
        mode = _structure_mode()
        if mode is None:
            self.scan_lbl.setText(
                "structure model unavailable - no native allin1 AND no "
                "WSL found; see requirements-dj-structure.txt")
            return
        self._structure_wsl = mode == "wsl"
        if self._structure_wsl:
            n = self._structure_import_results()   # interrupted-run leftovers
            cmd, why = _structure_wsl_command(self.planner.music_dir,
                                              self.planner.db)
            if cmd is None:
                self.scan_lbl.setText(why + (f" ({n} imported)" if n else ""))
                if n:
                    self.planner.reload_library()
                return
            program, args = cmd
        else:
            script = os.path.join(_REPO_ROOT, "tools", "dj", "dj_structure.py")
            program, args = sys.executable, [script, "--dir",
                                             self.planner.music_dir]
        self._structure_proc = QProcess(self)
        self._structure_proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self._structure_proc.readyReadStandardOutput.connect(
            self._structure_stdout)
        self._structure_proc.finished.connect(self._structure_done)
        self._structure_proc.start(program, args)
        self.structure_btn.setText("Stop structure")
        self.scan_lbl.setText(f"structure pass ({mode}): loading model...")

    def _structure_stdout(self):
        if self._structure_proc is None:
            return
        data = bytes(self._structure_proc.readAllStandardOutput()).decode(
            "utf-8", "replace")
        for line in data.splitlines():
            if not line.startswith("PROGRESS "):
                continue
            parts = line.split(" ", 5)
            if len(parts) < 6:
                continue
            done, total, matched, missed, cur = parts[1:6]
            self.scan_lbl.setText(
                f"structure {done}/{total}  ({matched} labeled, "
                f"{missed} failed)  {cur[:40]}")
            # Same rule as the mood pass: NO mid-scan reload_library()
            # (torch subprocess + full library re-hydration = OOM); the
            # pass commits per track, reload once on completion.

    def _structure_done(self, code=0, *a):
        self._structure_proc = None
        self.structure_btn.setText("Structure (ML)")
        n = self._structure_import_results() \
            if getattr(self, "_structure_wsl", False) else 0
        if code not in (0, None):
            # Partial results are already imported (resumable) - report
            # the failure honestly instead of a false "complete".
            self.scan_lbl.setText(
                f"structure pass failed (exit {code})"
                + (f" - {n} results imported first" if n else "")
                + " - is the allin1 env installed? See "
                "requirements-dj-structure.txt (WSL: ~/allin1, or set "
                "$DJ_WSL_ALLIN1_PY)")
            if n:
                try:
                    self.planner.reload_library()
                except MemoryError:
                    pass
            return
        self.scan_lbl.setText("structure pass complete"
                              + (f" - {n} imported" if n else ""))
        try:
            self.planner.reload_library()
        except MemoryError:
            self.scan_lbl.setText("structure pass complete - reopen the "
                                  "planner to see the labels (low memory)")

    # -- stems (htdemucs render, subprocess) ---------------------------------
    def run_stems(self):
        if self._stems_proc is not None:                 # toggles to stop
            self._stems_proc.kill()
            self.stems_btn.setText("stopping...")
            return
        from lib.dj import vocals
        if not vocals.available():
            self.scan_lbl.setText(
                "stem renderer unavailable - pip install -r "
                "requirements-dj-vocals.txt (torch + demucs)")
            return
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "dj", "dj_stems.py")
        self._stems_proc = QProcess(self)
        self._stems_proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self._stems_proc.readyReadStandardOutput.connect(self._stems_stdout)
        self._stems_proc.finished.connect(self._stems_done)
        self._stems_proc.start(sys.executable,
                               [script, "--dir", self.planner.music_dir])
        self.stems_btn.setText("Stop stems")
        self.scan_lbl.setText("stem render: loading model...")

    def _stems_stdout(self):
        if self._stems_proc is None:
            return
        data = bytes(self._stems_proc.readAllStandardOutput()).decode(
            "utf-8", "replace")
        for line in data.splitlines():
            if not line.startswith("PROGRESS "):
                continue
            parts = line.split(" ", 5)
            if len(parts) < 6:
                continue
            done, total, matched, missed, cur = parts[1:6]
            self.scan_lbl.setText(
                f"stems {done}/{total}  ({matched} rendered, "
                f"{missed} failed)  {cur[:40]}")
            # Same rule as mood/structure: no mid-pass reload (OOM).

    def _stems_done(self, *a):
        self._stems_proc = None
        self.stems_btn.setText("Stems (render)")
        self.scan_lbl.setText("stem render complete")
        try:
            self.planner.reload_library()
        except MemoryError:
            self.scan_lbl.setText("stem render complete - reopen the "
                                  "planner (low memory)")

    # -- one-button pipeline (Analyze all) -----------------------------------
    def run_analyze_all(self):
        if self._pipe or self._pipe_proc is not None:    # toggles to stop
            self._pipe = []
            if self._pipe_proc is not None:
                self._pipe_proc.kill()
                self._pipe_proc = None
            if self._enrich is not None and self._enrich.isRunning():
                self._enrich.stop()
            self._scan_timer.stop()
            self._pipe_total = 0          # tells stragglers the run is dead
            self.analyze_btn.setText("Analyze all")
            self._pipe_buttons(True)
            self.scan_lbl.setText("analysis pipeline stopped")
            return
        if (any(p is not None for p in (self._proc, self._mood_proc,
                                        self._structure_proc,
                                        self._stems_proc))
                or (self._enrich is not None and self._enrich.isRunning())):
            self.scan_lbl.setText("another pass is running - wait for it "
                                  "(or stop it) first")
            return
        # ONE stage list, shared with the headless CLI (dj_analyze.py) so
        # the two can never drift - see lib/dj/analyze.py for the ordering
        # rationale (stems before vocals, rhythm after stems, no
        # --refine-grids).
        stages = _build_stages(self.planner.music_dir,
                               include_stems=self.stems_chk.isChecked())
        self._pipe = stages
        self._pipe_total = len(stages)
        self._pipe_skipped = []
        self._pipe_failed = []
        self.analyze_btn.setText("Stop analyze")
        self._pipe_buttons(False)
        self._pipe_next()

    def _pipe_buttons(self, on):
        for b in (self.scan_btn, self.rescan_btn, self.refine_btn,
                  self.chroma_btn, self.rhythm_btn, self.revocals_btn,
                  self.enrich_btn, self.mood_btn, self.structure_btn,
                  self.stems_btn):
            b.setEnabled(on)

    def _run_single_stage(self, stage):
        """Run ONE pass through the pipeline runner - progress parsing,
        button state, and the completion reload all come for free."""
        if self._pipe or self._pipe_proc is not None:
            self.scan_lbl.setText("the pipeline is already running")
            return
        if (any(p is not None for p in (self._proc, self._mood_proc,
                                        self._structure_proc,
                                        self._stems_proc))
                or (self._enrich is not None and self._enrich.isRunning())):
            self.scan_lbl.setText("another pass is running - wait for it "
                                  "(or stop it) first")
            return
        self._pipe = [stage]
        self._pipe_total = 1
        self._pipe_skipped = []
        self._pipe_failed = []
        self.analyze_btn.setText("Stop analyze")
        self._pipe_buttons(False)
        self._pipe_next()

    def run_chroma(self):
        self._run_single_stage(
            {"name": "chroma",
             "args": [os.path.join(_REPO_ROOT, "tools", "dj", "dj_chroma.py"),
                      "--dir", self.planner.music_dir]})

    def run_rhythm(self):
        self._run_single_stage(
            {"name": "rhythm",
             "args": [os.path.join(_REPO_ROOT, "tools", "dj", "dj_rhythm.py"),
                      "--dir", self.planner.music_dir]})

    def run_revocals(self):
        from lib.dj import vocals
        if not vocals.available():
            self.scan_lbl.setText("torch/demucs not installed "
                                  "(requirements-dj-vocals.txt)")
            return
        self._run_single_stage(
            {"name": "vocal curves",
             "args": [os.path.join(_REPO_ROOT, "tools", "dj", "dj_scan.py"),
                      "--dir", self.planner.music_dir, "--revocals"]})

    def _pipe_next(self):
        self._scan_timer.stop()
        if not self._pipe:                               # pipeline done
            self.analyze_btn.setText("Analyze all")
            self._pipe_buttons(True)
            notes = []
            if self._pipe_skipped:
                notes.append("skipped: " + ", ".join(self._pipe_skipped))
            if self._pipe_failed:
                notes.append("FAILED: " + ", ".join(self._pipe_failed))
            self.scan_lbl.setText(
                "analysis pipeline complete"
                + (f"  ({'; '.join(notes)})" if notes else ""))
            try:
                self.planner.reload_library()
            except MemoryError:
                self.scan_lbl.setText("pipeline complete - reopen the "
                                      "planner (low memory)")
            return
        st = self._pipe.pop(0)
        self._pipe_stage_name = st["name"]
        k = self._pipe_total - len(self._pipe)
        self._pipe_prefix = f"[{k}/{self._pipe_total}] {st['name']}"
        if st.get("skip"):
            self._pipe_skipped.append(f"{st['name']} ({st['skip']})")
            self._pipe_next()
            return
        self.scan_lbl.setText(self._pipe_prefix + ": starting...")
        if st.get("structure"):
            # Resolve native-vs-WSL now; WSL uses the sqlite-free batch
            # handoff (import happens in _pipe_proc_done).
            mode = _structure_mode()
            self._pipe_structure_import = False
            if mode == "native":
                st["program"] = sys.executable
                st["args"] = [os.path.join(_REPO_ROOT, "tools", "dj",
                                           "dj_structure.py"),
                              "--dir", self.planner.music_dir]
            elif mode == "wsl":
                self._structure_import_results()
                cmd, _why = _structure_wsl_command(self.planner.music_dir,
                                                   self.planner.db)
                if cmd is None:
                    self._pipe_next()          # nothing to label
                    return
                st["program"], st["args"] = cmd
                self._pipe_structure_import = True
            else:
                self._pipe_skipped.append("structure (became unavailable)")
                self._pipe_next()
                return
        if st.get("enrich"):
            # In-process worker (MusicBrainz is rate-limited, not heavy).
            self._enrich = EnrichWorker(self.planner.music_dir)
            self._enrich.progress.connect(self._pipe_enrich_progress)
            self._enrich.finished_run.connect(self._pipe_enrich_done)
            self._enrich.start()
            return
        # A python interpreter handed a path that isn't there exits with
        # code 2 - the SAME code the DJ tools use for "optional deps
        # missing", so a mis-built script path used to report itself as
        # "skipped (deps unavailable)" and the whole pipeline would run
        # start-to-finish in a second having analyzed nothing (real bug
        # after the tools reorg). Name it before we launch.
        if (st.get("program", sys.executable) == sys.executable
                and st.get("args") and not os.path.exists(st["args"][0])):
            self._pipe_failed.append(f"{self._pipe_stage_name} "
                                     f"(script missing: {st['args'][0]})")
            self._pipe_next()
            return
        if st.get("scanfile"):
            # The scanner reports via the progress JSON, not PROGRESS
            # lines - reuse the existing poller (it also live-populates
            # the table, which is safe for the CPU scan stage).
            self._last_done = -1
            self._scan_timer.start(1500)
        self._pipe_proc = QProcess(self)
        self._pipe_proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self._pipe_proc.readyReadStandardOutput.connect(self._pipe_stdout)
        self._pipe_proc.finished.connect(self._pipe_proc_done)
        self._pipe_proc.start(st.get("program", sys.executable), st["args"])

    def _pipe_stdout(self):
        if self._pipe_proc is None:
            return
        data = bytes(self._pipe_proc.readAllStandardOutput()).decode(
            "utf-8", "replace")
        for line in data.splitlines():
            if not line.startswith("PROGRESS "):
                continue
            parts = line.split(" ", 5)
            if len(parts) < 6:
                continue
            done, total, matched, missed, cur = parts[1:6]
            self.scan_lbl.setText(
                f"{self._pipe_prefix} {done}/{total}  ({matched} ok, "
                f"{missed} failed)  {cur[:36]}")
            # No mid-pass reload here - the GPU stages hold big models
            # (same OOM rule as the individual mood/structure buttons).

    def _pipe_proc_done(self, code, *a):
        self._pipe_proc = None
        if getattr(self, "_pipe_structure_import", False):
            self._pipe_structure_import = False
            self._structure_import_results()   # partial imports too
        if code == 2:        # the tools' "deps unavailable" exit
            self._pipe_skipped.append(f"{self._pipe_stage_name} "
                                      "(deps unavailable)")
        elif code != 0:
            self._pipe_failed.append(f"{self._pipe_stage_name} "
                                     f"(exit {code})")
        self._pipe_next()

    def _pipe_enrich_progress(self, done, total, matched, missed, cur):
        self.scan_lbl.setText(
            f"{self._pipe_prefix} {done}/{total}  ({matched} matched, "
            f"{missed} missed)  {cur}")

    def _pipe_enrich_done(self, *a):
        if self._pipe_total:      # 0 = the pipeline was stopped mid-enrich
            self._pipe_next()


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


class SpectroWorker(QThread):
    """Log-frequency spectrogram of the opened track, off the GUI thread
    (~1s of FFT for a 5-minute song)."""
    done = pyqtSignal(int, object)             # track_id, spec dict | None

    def __init__(self, track_id, mono):
        super().__init__()
        self.track_id, self.mono = track_id, mono

    def run(self):
        try:
            from tools.dj.planner.waveform import compute_spectrogram
            self.done.emit(self.track_id, compute_spectrogram(self.mono))
        except Exception as e:
            print(f"[analysis] spectrogram failed: {e}")
            self.done.emit(self.track_id, None)


class StemLoadWorker(QThread):
    """Decode a track's four stem files + their display envelopes."""
    done = pyqtSignal(int, object)   # track_id, {"envs","arrs"} | error str

    def __init__(self, music_dir, track_id, expected_len):
        super().__init__()
        self.music_dir = music_dir
        self.track_id = track_id
        self.expected_len = expected_len

    def run(self):
        try:
            from lib.dj.stems import load_stems
            from tools.dj.planner.stemlanes import stem_envelope
            arrs = load_stems(self.music_dir, self.track_id,
                              expected_len=self.expected_len)
            if arrs is None:
                self.done.emit(self.track_id, "no stems on disk")
                return
            envs = {name: stem_envelope(a) for name, a in arrs.items()}
            self.done.emit(self.track_id, {"envs": envs, "arrs": arrs})
        except Exception as e:
            self.done.emit(self.track_id, f"{type(e).__name__}: {e}")


class AnalysisTab(QWidget):
    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self.track = None
        self.player = TrackPlayer()
        self.selected_cue = None
        self._decoder = None
        self._samples = None             # decoded stereo (playback + stems)
        self._spectro = None             # SpectroWorker
        self._stem_loader = None         # StemLoadWorker
        self._stems = None               # {name: (n,2) float16}
        v = QVBoxLayout(self)

        top = QHBoxLayout()
        self.track_combo = QComboBox()
        self.track_combo.setMinimumWidth(360)
        self.track_combo.currentIndexChanged.connect(self._combo_pick)
        top.addWidget(self.track_combo, 1)
        self.info_lbl = _no_width_floor(QLabel(""))
        top.addWidget(self.info_lbl, 2)
        self.spec_btn = QPushButton("▦ Spectrogram")
        self.spec_btn.setCheckable(True)
        self.spec_btn.setChecked(True)
        self.spec_btn.setToolTip(
            "Toggle spectrogram (log-frequency, 30 Hz-16 kHz) vs the "
            "min/max waveform. Same overlays either way.")
        self.spec_btn.toggled.connect(
            lambda on: self.wave.set_mode("spec" if on else "wave"))
        top.addWidget(self.spec_btn)
        v.addLayout(top)

        self.wave = WaveformView()
        self.wave.seekRequested.connect(self._seek)
        self.wave.cueClicked.connect(self._cue_clicked)
        v.addWidget(self.wave, 1)

        # STEM LANES: what each demucs stem extracted and where, on the
        # same time axis (zoom/pan follows the view above).
        from tools.dj.planner.stemlanes import StemLanes
        self.lanes = StemLanes()
        self.lanes.hide()
        self.wave.viewChanged.connect(self.lanes.set_view)
        self.lanes.seekRequested.connect(self._lane_seek)
        v.addWidget(self.lanes)

        srow = QHBoxLayout()
        srow.addWidget(QLabel("Stems:"))
        self.stems_lbl = _no_width_floor(QLabel(""))
        srow.addWidget(self.stems_lbl, 1)
        self.stem_checks = {}
        for name in ("drums", "bass", "other", "vocals"):
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.setEnabled(False)
            cb.setToolTip(
                f"Include the {name} stem in playback. Uncheck others to "
                "SOLO one stem and hear exactly what the separation "
                "extracted (all four checked = the original mix).")
            cb.toggled.connect(self._stem_mix_changed)
            self.stem_checks[name] = cb
            srow.addWidget(cb)
        self.stem_model_box = QComboBox()
        from lib.dj.stems import DEFAULT_STEM_MODEL, STEM_MODELS
        self.stem_model_box.addItems(list(STEM_MODELS))
        self.stem_model_box.setCurrentText(DEFAULT_STEM_MODEL)
        self.stem_model_box.setToolTip(
            "Separation model:\n"
            "htdemucs_ft - fine-tuned bag of four models (DEFAULT): "
            "audibly cleaner stems, ~1-2 min/track on GPU (weights "
            "download on first use).\n"
            "htdemucs - plain v4: ~10-30s/track, for quick checks.\n"
            "(A RoFormer option can slot in here later if ft isn't "
            "enough.)")
        srow.addWidget(self.stem_model_box)
        self.stem_render_btn = QPushButton("Render stems")
        self.stem_render_btn.setToolTip(
            "Separate just this song with the chosen model (subprocess). "
            "Stems land in .stems/<id>/ and unlock the stem transition "
            "styles for this track. Re-rendering with a different model "
            "overwrites.")
        self.stem_render_btn.clicked.connect(self._render_stems)
        srow.addWidget(self.stem_render_btn)
        self.stem_del_btn = QPushButton("Delete stems")
        self.stem_del_btn.setEnabled(False)
        self.stem_del_btn.setToolTip(
            "Remove this track's stem files from disk. The stem transition "
            "styles (stem_drum_swap / acapella_out) stop considering it "
            "until re-rendered.")
        self.stem_del_btn.clicked.connect(self._delete_stems)
        srow.addWidget(self.stem_del_btn)
        v.addLayout(srow)

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
        self.cue_lbl = _no_width_floor(QLabel("click a cue flag to select it"))
        tr.addWidget(self.cue_lbl, 1)
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
        self._samples = samples
        self.player.load(samples)
        mono = samples.mean(axis=1)
        self.wave.set_track(track, mono,
                            self.planner.db.cues_for(track.id))
        # Spectrogram (off-thread; the waveform shows until it lands).
        self._spectro = SpectroWorker(track.id, mono)
        self._spectro.done.connect(self._spectro_done)
        self._spectro.start()
        # Stems: show the lanes when this track has them.
        self._stems = None
        self.lanes.clear()
        self._sync_stem_row()
        if getattr(track, "has_stems", False):
            self._load_stems()
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
                self.lanes.set_playhead(t)

    # -- spectrogram --------------------------------------------------------
    def _spectro_done(self, track_id, spec):
        if self.track is not None and track_id == self.track.id:
            self.wave.set_spectrogram(spec,
                                      show=self.spec_btn.isChecked())

    # -- stems ----------------------------------------------------------------
    def _lane_seek(self, t):
        self._seek(t)
        self.wave.set_playhead(t, follow=False)
        self.lanes.set_playhead(t)

    def _sync_stem_row(self):
        have = self._stems is not None
        rendering = getattr(self.planner, "_stem_proc", None) is not None
        for cb in self.stem_checks.values():
            cb.setEnabled(have)
        self.stem_del_btn.setEnabled(have and not rendering)
        self.stem_render_btn.setEnabled(self.track is not None
                                        and not rendering)
        if rendering:
            pass                         # label carries live progress
        elif have:
            from lib.dj.stems import stem_model_of
            model = stem_model_of(self.planner.music_dir, self.track.id) \
                if self.track is not None else None
            self.stems_lbl.setText(
                (f"on disk ({model}) - " if model else "on disk - ")
                + "solo/mute to hear what each stem extracted")
        elif self.track is not None \
                and getattr(self.track, "has_stems", False):
            self.stems_lbl.setText("loading stems...")
        else:
            self.stems_lbl.setText("not rendered for this track")

    def _load_stems(self):
        if self.track is None or self._samples is None:
            return
        self._stem_loader = StemLoadWorker(
            self.planner.music_dir, self.track.id, len(self._samples))
        self._stem_loader.done.connect(self._stems_loaded)
        self._stem_loader.start()
        self._sync_stem_row()

    def _stems_loaded(self, track_id, payload):
        if self.track is None or track_id != self.track.id:
            return
        if isinstance(payload, str):
            self.stems_lbl.setText("stem load failed: " + payload)
            return
        self._stems = payload["arrs"]
        from lib.dj.stems import stem_model_of
        self.lanes.set_stems(payload["envs"], len(self._samples) / RATE,
                             model=stem_model_of(self.planner.music_dir,
                                                 track_id))
        self.lanes.set_view(self.wave.view_t0, self.wave.view_t1)
        for cb in self.stem_checks.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self.lanes.set_muted(())
        self._sync_stem_row()

    def _stem_mix_changed(self, *_a):
        """Solo/mute audition: rebuild the player buffer from the checked
        stems (all four checked = the ORIGINAL mix - cleaner than a stem
        sum, which carries separation artifacts)."""
        if self._stems is None or self._samples is None:
            return
        checked = [n for n, cb in self.stem_checks.items()
                   if cb.isChecked()]
        self.lanes.set_muted(n for n in self.stem_checks
                             if n not in checked)
        pos, was = self.player.time_s(), self.player.playing
        if len(checked) == len(self.stem_checks):
            buf = self._samples
        elif not checked:
            buf = np.zeros((len(self._samples), 2), dtype=np.float32)
        else:
            buf = np.zeros((len(self._samples), 2), dtype=np.float32)
            for n in checked:
                buf += self._stems[n].astype(np.float32)
        self.player.load(buf)
        self.player.seek(pos)
        if was:
            self.player.play()

    def _render_stems(self):
        """Kick the planner-wide background render for the open track;
        this tab's status line narrates and the lanes load on finish."""
        if self.track is None:
            return
        self.planner.render_stems(
            self.track, model=self.stem_model_box.currentText(),
            on_status=self.stems_lbl.setText,
            on_done=self._stems_rendered)
        self._sync_stem_row()

    def _stems_rendered(self, track_id, ok):
        if ok and self.track is not None and self.track.id == track_id:
            self._load_stems()

    def _delete_stems(self):
        if self.track is None or self._stems is None:
            return
        if QMessageBox.question(
                self, "Delete stems",
                f"Delete the rendered stems for '{self.track.title}'?\n"
                "The stem transition styles stop considering this track "
                "until re-rendered.") != QMessageBox.StandardButton.Yes:
            return
        import shutil
        from lib.dj.stems import stems_dir
        # If a solo/mute mix is loaded, put the original back first.
        for cb in self.stem_checks.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        if self._samples is not None:
            pos, was = self.player.time_s(), self.player.playing
            self.player.load(self._samples)
            self.player.seek(pos)
            if was:
                self.player.play()
        try:
            shutil.rmtree(stems_dir(self.planner.music_dir, self.track.id))
        except OSError as e:
            self.stems_lbl.setText(f"delete failed: {e}")
            return
        self._stems = None
        self.lanes.clear()
        self.track.has_stems = False
        for t in self.planner.library_all or self.planner.library:
            if t.id == self.track.id:
                t.has_stems = False
        self.planner.library_tab.table.viewport().update()  # stems column
        self._sync_stem_row()

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


class PlanOpWorker(QThread):
    """One long planning operation (suggest / optimize / autofill / shape /
    slot alternatives / bridge) off the GUI thread: the beam searches walk
    the whole library and froze the UI for seconds when run inline."""
    done = pyqtSignal(object)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn())
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
            # One renderer shared with dj_player --audition - see
            # lib/dj/audition.py (dynamic pre-roll and all).
            from lib.dj.audition import render_seam
            self.done.emit(render_seam(self.db, self.a, self.b, self.plan,
                                       status=self.status.emit))
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
        self._op = None                  # running PlanOpWorker, one at a time
        self._plan_btns = []             # greyed out while an op runs
        self.seam_player = TrackPlayer()
        # UNDO for set edits: every mutator snapshots first, so Ctrl+Z
        # walks back anything - including Optimize/Shape/Auto-fill, which
        # replace the WHOLE list and were irreversible before.
        self._undo_stack, self._redo_stack = [], []
        for keys, fn in (("Ctrl+Z", self._undo_edit),
                         ("Ctrl+Y", self._redo_edit),
                         ("Ctrl+Shift+Z", self._redo_edit)):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

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
        # PUSH TO LIVE: save, then hand the set to the running show over
        # the web controller's action queue - the same channel the /dj
        # panel's setlist picker uses. Works whether the DJ is idle (arms;
        # Start plays it) or already playing (replans from the list). The
        # set's saved theme/length/arc ride along via load_setlist.
        push = QPushButton("▶ Push to live")
        menu = QMenu(push)
        menu.addAction("Load in order (play top to bottom)",
                       lambda: self._push_live("order"))
        menu.addAction("Load as pool (brain steers inside the list)",
                       lambda: self._push_live("pool"))
        push.setMenu(menu)
        push.setToolTip(
            "Send this set to the running show (localhost web panel). "
            "Saves first; the show picks up the set's theme and arc "
            "clock too. Override host via DJ_SHOW_URL.")
        left.addLayout(srow)
        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("Notes:"))
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText(
            "set notes (venue, occasion, what worked) - saved with the set")
        nrow.addWidget(self.notes_edit, 1)
        nrow.addWidget(push)             # its own row keeps srow narrow
        left.addLayout(nrow)

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
        self.minutes_spin.setToolTip(
            "Target set length: used by Suggest/Auto-fill AND as the arc "
            "anchor for the next-track suggestions (a 3-track set of a "
            "60-min plan is ~15% in, and suggestions score accordingly).")
        # The target length IS the suggestions' arc anchor AND the arc
        # strip's timeline - re-rank and re-draw on change.
        self.minutes_spin.valueChanged.connect(self._target_changed)
        prow.addWidget(self.minutes_spin)
        left.addLayout(prow)
        # ONE primary ordering flow (the four underlying ops overlapped so
        # much that picking among them was expertise the UI demanded for
        # nothing). Build set = suggest for the theme/length, then shape
        # the chosen metric curve (or just clean the seams for 'flat').
        # The individual ops live under More ▾ - and every one of them is
        # a Ctrl+Z away from being undone now.
        prow2 = QHBoxLayout()
        bs = QPushButton("✦ Build set")
        bs.setToolTip(
            "Generate the whole set in one go: suggest tracks for the "
            "theme + target length, then order them so the shape curve "
            "below actually happens (flat = optimize seams only). "
            "Replaces the current set - Ctrl+Z brings it back.")
        bs.clicked.connect(self.build_set)
        prow2.addWidget(bs, 1)
        more = QPushButton("More ▾")
        mm = QMenu(more)
        mm.addAction("Suggest set (replace)", self.suggest)
        mm.addAction("Optimize order (seams only)", self.optimize)
        mm.addAction("Apply shape to current set", self.apply_shape)
        mm.addAction("Auto-fill timed anchors", self.autofill)
        more.setMenu(mm)
        more.setToolTip(
            "The individual ordering ops, for surgical use:\n"
            "· Suggest - a fresh set from the theme\n"
            "· Optimize - reorder suggestions for cleaner seams (keeps "
            "anchors put; won't create an arc)\n"
            "· Apply shape - impose the curve below on the current set\n"
            "· Auto-fill - insert fills so timed anchors land on their "
            "offsets")
        prow2.addWidget(more)
        left.addLayout(prow2)
        self._plan_btns += [bs, more]

        # SHAPE the set: order it so tempo/energy follows a curve (the only
        # control that actually makes a set BUILD; feeds Build set and the
        # menu's Apply shape).
        prow3 = QHBoxLayout()
        prow3.addWidget(QLabel("Shape:"))
        self.shape_metric = QComboBox()
        self.shape_metric.addItems(["tempo", "energy"])
        prow3.addWidget(self.shape_metric)
        self.shape_curve = QComboBox()
        self.shape_curve.addItems(["rise", "peak", "wind_down", "flat"])
        self.shape_curve.setToolTip(
            "rise = build all set · peak = up then down · wind_down = "
            "start hot, land soft · flat = no curve, best seams only")
        prow3.addWidget(self.shape_curve)
        prow3.addStretch(1)
        left.addLayout(prow3)

        # THE ARC STRIP: the set's shape (energy vs the theme's target,
        # bpm path, seam quality) visible WHILE building, not only after
        # reading the compiled text. Click a bar to select its entry.
        self.arc_strip = ArcStrip()
        self.arc_strip.slotClicked.connect(self._strip_clicked)
        left.addWidget(self.arc_strip)
        left.addWidget(_no_width_floor(
            QLabel("Set (drag to reorder, double-click = anchor,"
                   " Del = remove, right-click = repair)")))
        self.set_list = SetListView()
        # Rhythm-strip column delegate. PARENTED - an unparented delegate
        # gets GC'd and dies natively at first paint (see RhythmDelegate
        # registration above).
        self.set_list.setItemDelegateForColumn(
            SetListView.RHY_COL, SetRowDelegate(self.set_list))
        # Selecting a slot re-anchors the suggestion panel to it
        # ("what should come after THIS one") - debounced. getattr guard:
        # the timer is created later in __init__.
        self.set_list.currentItemChanged.connect(
            lambda *_: (getattr(self, "_suggest_timer", None)
                        and self._suggest_timer.start(400)))
        self.set_list.reordered.connect(self._reordered)
        self.set_list.itemDoubleClicked.connect(
            lambda item, _col: self._toggle_anchor(item))
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
        # PIN the transition style INTO the selected song: the pin goes
        # through plan_transition's real gates (compile AND live), so a
        # vetoed pin shows a 'style pin refused' warning on the ↳ line
        # instead of silently playing something impossible.
        from tools.dj.planner.copilot import STYLES
        pin_menu = QMenu("📌 pin transition INTO this song", self.set_list)
        auto = pin_menu.addAction("auto (clear pin)")
        auto.triggered.connect(lambda: self._pin_style(None))
        pin_menu.addSeparator()
        for s in STYLES:
            a = pin_menu.addAction(s)
            a.triggered.connect(lambda _, st=s: self._pin_style(st))
        self.set_list.addAction(pin_menu.menuAction())
        stem_act = QAction("▤ render stems for this song", self)
        stem_act.triggered.connect(self._render_stems_selected)
        self.set_list.addAction(stem_act)
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
        h.addLayout(left, 3)   # wider than the plan pane: the set rows now
                               # carry the rhythm strip on the right

        right = QVBoxLayout()
        right.addWidget(_no_width_floor(
            QLabel("Compiled plan (select a ↳ seam, audition):")))
        self.plan_list = QListWidget()
        right.addWidget(self.plan_list, 3)   # the plan IS the tab - keep it big
        # SEAM INSPECTOR + TRANSITION OPTIONS side by side: same seam,
        # two views (rhythm grids | style menu), half the vertical cost.
        insp_row = QHBoxLayout()
        self.seam_inspector = SeamInspector()
        insp_row.addWidget(self.seam_inspector, 3)
        opt_col = QVBoxLayout()
        opt_col.addWidget(_no_width_floor(QLabel(
            "Transition options (click = pin, then audition):")))
        self.style_opts = QListWidget()
        self.style_opts.setFont(_mono_font())
        self.style_opts.setMaximumHeight(170)
        self.style_opts.setToolTip(
            "Everything plan_transition considered for the selected seam,\n"
            "playable styles sorted best-first:\n"
            "▶ = what the compiler chose · 📌 = your pin\n"
            "▤stems = plays through rendered stem files on this seam ·\n"
            "▤duck = this blend will mute A's vocal stem through the\n"
            "overlap (two sung passages would otherwise clash)\n"
            "q 0..1 = predicted seam quality for THAT style (the seam's\n"
            "measured physics - key fit, kick agreement, swing, grid\n"
            "confidence - weighted by what the style exposes; a fade is\n"
            "always 'safe but flat'). Display estimate, not the brain's\n"
            "score. odds = dice weight share · 'gated' = a safety rule\n"
            "removed it (render stems, refine grids... to unlock).\n"
            "Click a row to pin the seam to that style - honored live.")
        self.style_opts.itemClicked.connect(self._style_option_clicked)
        opt_col.addWidget(self.style_opts, 1)
        insp_row.addLayout(opt_col, 2)
        right.addLayout(insp_row)
        self.plan_list.currentRowChanged.connect(self._seam_selected)
        brow = QHBoxLayout()
        self.play_set_btn = QPushButton("▶ Play set")
        self.play_set_btn.setToolTip(
            "Play the compiled set from the SELECTED song (from the top "
            "when none selected). Full transport on the Mix tab.")
        self.play_set_btn.clicked.connect(self._play_set)
        brow.addWidget(self.play_set_btn)
        self.audition_btn = QPushButton("▶ Audition seam")
        self.audition_btn.clicked.connect(self.audition)
        brow.addWidget(self.audition_btn)
        st = QPushButton("■ Stop")
        st.clicked.connect(lambda: self.planner.stop_all_playback())
        brow.addWidget(st)
        # RATE the seam you just auditioned: writes the same cross-night
        # seam_feedback the live thumbs use (user weight 1.0; the DJ's own
        # measurements weigh 0.5), so one press teaches pair memory, the
        # feature-class memory (key x offset x grid-conf x groove bucket)
        # AND per-style taste - live selection and every planner op read
        # all three. Enabled once an audition has actually played.
        self._last_aud = None
        self.rate_up_btn = QPushButton("👍")
        self.rate_up_btn.setToolTip(
            "This seam sounded good - remember it. Boosts this exact pair "
            "(cross-night), its feature class, and this transition style "
            "in both the live DJ and the planner's scoring.")
        self.rate_up_btn.clicked.connect(lambda: self._rate_seam(True))
        self.rate_up_btn.setEnabled(False)
        brow.addWidget(self.rate_up_btn)
        self.rate_dn_btn = QPushButton("👎")
        self.rate_dn_btn.setToolTip(
            "This seam sounded rough - remember that. Leans selection away "
            "from this pair, its feature class, and this style. One press "
            "outweighs two of the DJ's own automatic assessments.")
        self.rate_dn_btn.clicked.connect(lambda: self._rate_seam(False))
        self.rate_dn_btn.setEnabled(False)
        brow.addWidget(self.rate_dn_btn)
        # TEMPO ENGINE picker - the no-env-vars way to A/B varispeed vs
        # keylock: pick, recompile, audition the same seam again.
        brow.addStretch(1)
        right.addLayout(brow)
        brow = QHBoxLayout()             # second row: engine + shopping
        brow.addWidget(QLabel("Engine:"))
        self.engine_box = QComboBox()
        self.engine_box.addItems(["vari", "rubberband", "rubberband-crisp",
                                  "wsola", "pv"])
        from lib.dj import stretch_engine_name
        self.engine_box.setCurrentText(stretch_engine_name())
        self.engine_box.setToolTip(
            "Tempo engine for auditions and plans:\n"
            "rubberband - keylock via Rubber Band R3 (DEFAULT; needs "
            "pip install -r requirements-dj-keylock.txt): constant pitch, "
            "warble-free, enables the ±1-semitone key rescue.\n"
            "vari - turntable mode: pitch rides tempo, zero stretch "
            "artifacts; tempo bends split across both decks.\n"
            "rubberband-crisp - Rubber Band R2: sharper attacks, less "
            "CPU, but audible warble on sustained tones.\n"
            "wsola / pv - home-grown keylock engines (A/B by ear).\n"
            "Applies to this planner session; set dj.stretch_engine in "
            "config.yaml to make the live show use it too.")
        self.engine_box.currentTextChanged.connect(self._engine_changed)
        brow.addWidget(self.engine_box)
        shop = QPushButton("🛒 Shop gaps")
        shop.setToolTip(
            "Turn the compiled plan's flagged seams (key clash, weak seam, "
            "energy hole, big stretch) into Beatport connector searches: "
            "pick a gap, the Discover tab shops genre charts inside the "
            "BPM window reachable from BOTH sides, ranked as bridges.")
        shop.clicked.connect(self._shop_gaps)
        brow.addWidget(shop)
        brow.addStretch(1)
        right.addLayout(brow)

        # BOTTOM TABS: suggestions / worst seams / copilot each wanted a
        # permanent slice of the column and the plan list paid for all of
        # them (user: "the plan window is very small now"). One at a time
        # is plenty - they're consultation surfaces, not monitors.
        bottom = QTabWidget()
        sug_page = QWidget()
        sv = QVBoxLayout(sug_page)
        sv.setContentsMargins(4, 4, 4, 4)

        # -- next-track suggestions: what should FOLLOW this set --------------
        self.suggest_hdr = _no_width_floor(QLabel("Suggested next:"))
        sv.addWidget(self.suggest_hdr)
        self.suggest_list = QListWidget()
        self.suggest_list.setFont(_mono_font())   # column-aligned rows
        self.suggest_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.suggest_list.setToolTip(
            "Top candidates to follow the set's last track, scored with "
            "the live brain (seam quality, key, tempo, mood/genre, pair "
            "memory) at the arc position the set has reached, with an "
            "artist-variety lean. Columns: B beat-match, H harmonics, "
            "G groove (kick/swing/flam compatibility), T theme fit. "
            "Double-click to preview; multi-select "
            "and '+ Add' to append. Refreshes as the set changes.")
        self.suggest_list.itemDoubleClicked.connect(self._suggest_play)
        sv.addWidget(self.suggest_list, 1)
        sgrow = QHBoxLayout()
        sg_play = QPushButton("▶ Play")
        sg_play.clicked.connect(self._suggest_play)
        sgrow.addWidget(sg_play)
        sg_stop = QPushButton("■")
        sg_stop.setFixedWidth(32)
        sg_stop.clicked.connect(lambda: self.planner.stop_all_playback())
        sgrow.addWidget(sg_stop)
        sg_add = QPushButton("+ Add to set")
        sg_add.clicked.connect(self._suggest_add)
        sgrow.addWidget(sg_add)
        sg_ref = QPushButton("↻")
        sg_ref.setFixedWidth(32)
        sg_ref.setToolTip("Refresh suggestions now")
        sg_ref.clicked.connect(lambda: self._suggest_timer.start(0))
        sgrow.addWidget(sg_ref)
        # Scrub transport: previews start at the track's MIX-IN point
        # (where it would actually enter the set); drag to scrub live.
        sg_b15 = QPushButton("« 15")
        sg_b15.setFixedWidth(44)
        sg_b15.clicked.connect(lambda: self._sg_seek_rel(-15.0))
        sgrow.addWidget(sg_b15)
        self.sg_seek = QSlider(Qt.Orientation.Horizontal)
        self.sg_seek.setMaximum(1000)
        self.sg_seek.setToolTip("Scrub the previewed suggestion (seeks "
                                "live while dragging)")
        self._sg_drag = False
        self.sg_seek.sliderPressed.connect(
            lambda: setattr(self, "_sg_drag", True))
        self.sg_seek.sliderMoved.connect(self._sg_scrub)
        self.sg_seek.sliderReleased.connect(self._sg_released)
        sgrow.addWidget(self.sg_seek, 1)
        sg_f15 = QPushButton("15 »")
        sg_f15.setFixedWidth(44)
        sg_f15.clicked.connect(lambda: self._sg_seek_rel(15.0))
        sgrow.addWidget(sg_f15)
        self.sg_time = QLabel("-:-- / -:--")
        sgrow.addWidget(self.sg_time)
        sv.addLayout(sgrow)
        bottom.addTab(sug_page, "Suggested next")
        self._sg_tick = QTimer(self)
        self._sg_tick.timeout.connect(self._sg_tick_update)
        self._sg_tick.start(250)
        self._suggest_worker = None
        self._suggest_gen = 0
        self._suggest_timer = QTimer(self)
        self._suggest_timer.setSingleShot(True)
        self._suggest_timer.timeout.connect(self._suggest_start)

        # RANKED WORST SEAMS: the report card's actionable half. The
        # one-line card says "median seam 0.41" - this says WHICH seams
        # drag it down, worst first; click one to select it (inspector +
        # audition), then repair via alternatives/bridge/style pin.
        self.worst_list = QListWidget()
        self.worst_list.setFont(_mono_font())
        self.worst_list.setToolTip(
            "The set's weakest seams, worst first. Click to jump to the "
            "seam; right-click the entry above it for repairs "
            "(alternatives / bridge).")
        self.worst_list.itemClicked.connect(self._worst_clicked)
        bottom.addTab(self.worst_list, "Worst seams")
        # Conversational set-builder (Claude tool-loop over this same set).
        from tools.dj.planner.copilot_panel import CopilotPanel
        self.copilot_panel = CopilotPanel(planner, self)
        self.copilot_panel.entriesApplied.connect(self._copilot_applied)
        bottom.addTab(self.copilot_panel, "Set Copilot")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        right.addWidget(bottom, 1)
        h.addLayout(right, 2)

    # -- entries ------------------------------------------------------------
    def theme(self):
        return get_theme(self.theme_combo.currentText())

    def _snapshot_undo(self):
        """Capture the entry list BEFORE a mutation - one call at the top
        of every mutator is the whole undo system."""
        self._undo_stack.append([dict(e) for e in self.entries])
        del self._undo_stack[:-50]
        self._redo_stack.clear()

    def _undo_edit(self):
        if not self._undo_stack:
            self.status.setText("nothing to undo")
            return
        self._redo_stack.append([dict(e) for e in self.entries])
        self.entries = self._undo_stack.pop()
        self._rebuild()
        self.recompile()
        self.status.setText(
            f"undo ({len(self._undo_stack)} more, Ctrl+Y = redo)")

    def _redo_edit(self):
        if not self._redo_stack:
            self.status.setText("nothing to redo")
            return
        self._undo_stack.append([dict(e) for e in self.entries])
        self.entries = self._redo_stack.pop()
        self._rebuild()
        self.recompile()
        self.status.setText(f"redo ({len(self._redo_stack)} more)")

    def add_tracks(self, tracks, at=None):
        """Append tracks, or insert them at index `at` (in order)."""
        self._snapshot_undo()
        new = [{"track_id": t.id, "pin_type": "suggestion",
                "target_offset_min": None,
                "style_override": None,
                "target_play_s": None} for t in tracks]
        if at is None or not (0 <= at <= len(self.entries)):
            self.entries.extend(new)
        else:
            self.entries[at:at] = new
        self._rebuild()
        self.recompile()

    def _target_changed(self, _v):
        """Target-length spinner moved: re-anchor suggestions and rescale
        the arc strip's timeline."""
        self._suggest_timer.start(600)
        if self.compiled:
            self._update_strip(self.compiled)

    # -- next-track suggestions ------------------------------------------
    def _suggest_anchor_idx(self):
        """Selected slot index, or None (= follow the set's last track)."""
        row = self.set_list.currentRow()
        return row if 0 <= row < len(self.entries) else None

    def _suggest_start(self):
        if self._suggest_worker is not None \
                and self._suggest_worker.isRunning():
            self._suggest_timer.start(400)       # retry after the current
            return
        self._suggest_gen += 1
        gen = self._suggest_gen
        entries = list(self.entries)
        library = list(self.planner.library)
        theme = self.theme()
        compiled = self.compiled
        pair_mem = dict(getattr(self.planner, "pair_memory", {}) or {})

        target_s = float(self.minutes_spin.value()) * 60.0
        anchor = self._suggest_anchor_idx()
        # Header says WHAT the suggestions follow - selection or set end.
        by_id = {t.id: t for t in library}
        aidx = anchor if anchor is not None else len(entries) - 1
        at = (by_id.get(entries[aidx]["track_id"])
              if 0 <= aidx < len(entries) else None)
        if at is None:
            self.suggest_hdr.setText("Suggested next (openers):")
        elif anchor is not None and anchor < len(entries) - 1:
            self.suggest_hdr.setText(
                f"Suggested next (follows SELECTED slot {anchor + 1}: "
                f"{at.title[:30]}):")
        else:
            self.suggest_hdr.setText(
                f"Suggested next (follows the last track: "
                f"{at.title[:30]}):")

        def fn():
            return suggest_followers(library, entries, theme, compiled,
                                     pair_mem, target_s=target_s,
                                     anchor_idx=anchor)

        def done(res):
            if gen != self._suggest_gen:
                return                           # stale: set changed since
            self._suggest_apply(res)
        w = PlanOpWorker(fn)
        w.done.connect(done)
        self._suggest_worker = w
        w.start()

    @staticmethod
    def _q_bar(x):
        """0..1 quality -> a one-char bar (▁ weak .. █ excellent)."""
        if x is None:
            return "·"
        return "▁▂▃▅▆█"[min(5, int(max(0.0, min(1.0, x)) * 5.999))]

    def _suggest_apply(self, res):
        if isinstance(res, dict) and "error" in res:
            return                               # keep the old list
        self.suggest_list.clear()
        fade_header_done = False
        for r in res:
            if r.get("tier") == "fade" and not fade_header_done:
                hdr = QListWidgetItem(
                    "── fade-reachable (outside beat-match range; enters "
                    "via a deliberate fade) ──", self.suggest_list)
                hdr.setFlags(Qt.ItemFlag.NoItemFlags)
                hdr.setForeground(QColor(140, 140, 155))
                fade_header_done = True
            if r.get("fit") is not None:
                note = f"{r['fit']:.2f} ·top {r.get('top_pct', 0):d}%"
            else:
                note = r.get("why", "")
            quality = (f"B{self._q_bar(r.get('beat'))}"
                       f"H{self._q_bar(r.get('key'))}"
                       f"G{self._q_bar(r.get('groove'))}"
                       f"T{self._q_bar(r.get('theme'))}")
            ago = self._played_ago(r["id"])
            it = QListWidgetItem(
                f"{_clip(r['title'], 30)} {_clip(r['artist'], 18)} "
                f"{_clip(r.get('genre', ''), 14)} "
                f"{r['bpm']:3.0f} {r['camelot']:>3s}  {quality}  {note}"
                + (f"  ·{ago}" if ago else ""),
                self.suggest_list)
            it.setData(Qt.ItemDataRole.UserRole, r["id"])
            if r.get("tier") == "fade":
                it.setForeground(QColor(165, 170, 185))
            tip = []
            if r.get("tier") == "fade":
                tip.append(f"FADE-REACHABLE: outside ±8% tempo reach of "
                           f"the last track - would enter via the dipped "
                           f"fade, so beat/key don't apply. Rank: top "
                           f"{r.get('top_pct', 0)}% of {r.get('n_viable')} "
                           f"fade candidates (energy/mood/genre fit).")
            elif r.get("n_viable"):
                tip.append(f"Rank: top {r.get('top_pct', 0)}% of "
                           f"{r['n_viable']} beat-matchable candidates "
                           f"(fit is a many-factor product; ~0.4 is the "
                           f"practical ceiling)")
            if r.get("beat") is not None:
                sp = abs(r.get("stretch_pct") or 0.0)
                tip.append(f"Beat: {r['beat']:.2f}  "
                           f"(stretch {r.get('stretch_pct', 0):+.1f}% - "
                           + ("invisible" if sp <= 4.0 else
                              "audible feel change" if sp <= 5.5 else
                              "past the wall, groove will drag")
                           + ")")
            if r.get("key") is not None:
                tip.append(f"Harmonics: {r['key']:.2f}  - "
                           + ("melodies will sing together"
                              if r["key"] >= 0.8 else
                              "workable; keep overlapping melodies apart"
                              if r["key"] >= 0.55 else
                              "CLASH - two fighting melodies unless one "
                              "side is EQ'd out")
                           + "  [0.8+ great · 0.55 workable · <0.5 clash]")
            if r.get("groove") is not None:
                g = r["groove"]
                tip.append(f"Groove: {g:.2f}  - "
                           + ("grooves lock together" if g >= 0.6 else
                              "half-agrees; the blend will be kept short"
                              if g >= 0.45 else
                              "grooves fight - expect a one-low-bed style "
                              "or a short decisive swap")
                           + (f"  ({', '.join(r['groove_chips'])})"
                              if r.get("groove_chips") else "")
                           + "  [0.6+ locks · <0.45 short blends only]")
            if r.get("theme") is not None:
                tip.append(f"Theme: {r['theme']:.2f}  "
                           f"(energy {r.get('energy', 0):.2f} vs arc "
                           f"target {r.get('arc', 0):.2f} + mood match - "
                           f"how well it serves the NIGHT, not the seam)")
            it.setToolTip("\n".join(tip))

    def _suggest_selected_tracks(self):
        ids = [it.data(Qt.ItemDataRole.UserRole)
               for it in self.suggest_list.selectedItems()]
        by_id = {t.id: t for t in self.planner.library}
        return [by_id[i] for i in ids if i in by_id]

    def _suggest_play(self, *a):
        tracks = self._suggest_selected_tracks()
        if not tracks:
            return
        t = tracks[0]
        # Start where the track would actually ENTER the set - its first
        # mix-in point - not the cold intro. Scrub from there.
        start = t.mix_ins[0]["time_s"] if t.mix_ins else 0.0
        # Reuse the library tab's whole decode/transport pipeline.
        self.planner.library_tab._play_track(t, start_s=start)

    # -- suggestion scrub transport (drives the shared library player) ----
    def _sg_player(self):
        return self.planner.library_tab.lib_player

    def _sg_dur(self):
        p = self._sg_player()
        return (len(p.samples) / 44100.0
                if p.samples is not None and len(p.samples) else 0.0)

    def _sg_scrub(self, v):
        d = self._sg_dur()
        if d > 0:
            self._sg_player().seek(v / 1000.0 * d)

    def _sg_released(self):
        self._sg_drag = False
        self._sg_scrub(self.sg_seek.value())

    def _sg_seek_rel(self, dt):
        p = self._sg_player()
        if p.samples is not None:
            p.seek(max(0.0, min(p.time_s() + dt, self._sg_dur() - 1.0)))

    @staticmethod
    def _sg_mmss(t):
        return f"{int(t // 60)}:{int(t % 60):02d}"

    def _sg_tick_update(self):
        d = self._sg_dur()
        if d <= 0:
            self.sg_time.setText("-:-- / -:--")
            if not self._sg_drag:
                self.sg_seek.setValue(0)
            return
        t = self._sg_player().time_s()
        self.sg_time.setText(f"{self._sg_mmss(t)} / {self._sg_mmss(d)}")
        if not self._sg_drag:
            self.sg_seek.blockSignals(True)
            self.sg_seek.setValue(int(t / d * 1000))
            self.sg_seek.blockSignals(False)

    def _suggest_add(self):
        tracks = self._suggest_selected_tracks()
        if not tracks:
            self.status.setText("select suggestion(s) to add first")
            return
        # Follow the anchoring: a mid-set selection means "insert AFTER
        # that slot", not "append at the end".
        anchor = self._suggest_anchor_idx()
        at = anchor + 1 if (anchor is not None
                            and anchor < len(self.entries) - 1) else None
        self.add_tracks(tracks, at=at)  # rebuild+recompile -> auto-refresh

    # Set-table item roles: the entry dict rides column 0 (reorder reads
    # THIS, never display text); the rhythm-strip payload rides the
    # rhythm column itself (its delegate reads index.data directly).
    ENTRY_ROLE = Qt.ItemDataRole.UserRole
    RHY_ROLE = Qt.ItemDataRole.UserRole + 2

    def _entry_track(self, e):
        return next((x for x in (self.planner.library_all
                                 or self.planner.library)
                     if x.id == e["track_id"]), None)

    def _rebuild(self):
        self.set_list.clear()
        for e in self.entries:
            t = self._entry_track(e)
            tag = "⚓" if e["pin_type"] == "anchor" else "•"
            if e.get("target_offset_min"):
                tag += f"@{e['target_offset_min']:.0f}m"
            if e.get("style_override"):
                tag += "📌"              # seam INTO this song is pinned
            if t is None:
                cols = [tag, f"track {e['track_id']}", "", "", "", "",
                        "", "", "", "", ""]
            else:
                cols = [tag, t.title, t.artist, track_genre(t),
                        f"{t.bpm:.0f}", t.camelot,
                        energy_glyph(t.energy_proxy()), groove_glyph(t),
                        "▤" if getattr(t, "has_stems", False) else "",
                        "", ""]
            it = QTreeWidgetItem(cols)
            it.setToolTip(SetListView.STEM_COL,
                          "▤ = stems rendered (stem transition styles "
                          "available)" if getattr(t, "has_stems", False)
                          else "no stems - right-click to render")
            # Rows drag as WHOLE entries; never droppable INTO (no nesting)
            it.setFlags(Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsDragEnabled)
            it.setData(0, self.ENTRY_ROLE, e)
            # Plain pattern until the compiler colors it against its
            # predecessor.
            it.setData(SetListView.RHY_COL, self.RHY_ROLE,
                       _rhythm_row_payload(t) if t is not None else None)
            self.set_list.addTopLevelItem(it)

    @staticmethod
    def _seam_estimate(si, p):
        """One 0..1 'how well will these two work together' number for the
        set list, blending the seam's physics: section-pair quality, key
        fit, groove compatibility, stretch cost, blend floor. DISPLAY
        blend only - selection/compile use the brain's full score; this
        exists so adjacency quality is readable per row at a glance."""
        si = si or {}
        p = p or {}
        key = si.get("key_fit", 0.6)
        rt = si.get("rhythm") or {}
        groove = rt.get("score", 0.75)       # unmeasured -> mildly neutral
        # pair_score's practical ceiling is ~0.6 on real material - rescale
        # so a genuinely good seam reads near the top of the bar.
        pairq = min(1.0, (p.get("pair_score") or 0.3) / 0.6)
        rate = p.get("rate") or 1.0
        stretch = math.exp(-((abs(math.log(max(rate, 1e-6)))) / 0.045) ** 2)
        est = 0.30 * pairq + 0.25 * groove + 0.25 * key + 0.20 * stretch
        floor = si.get("floor")
        if floor is not None and floor < 0.15:
            est *= 0.6                       # dead air in the overlap
        return max(0.0, min(1.0, est))

    def _color_set_list(self, result):
        """Color each set entry by its INBOUND transition (how the set
        arrives AT this track): green = clean seam (or the opener),
        orange = compiler-warned, red = energy hole in the blend,
        grey = enters via a fade. Same palette as the compiled plan.
        Also appends the inbound seam ESTIMATE (bar + number) to the row
        and puts the chips/warnings in its tooltip - adjacency quality
        readable in the set list itself, not only in the compiled plan."""
        GOOD = QColor(120, 200, 140)
        WARN = QColor(255, 170, 100)
        BAD = QColor(230, 110, 110)
        FADE = QColor(160, 160, 170)
        slots = result.get("slots") or []
        # Compile drops unknown/do-not-use ids, so slot k may not be
        # entry k - align by walking track_ids in order.
        e_idx, rows = 0, []
        for s in slots:
            while e_idx < len(self.entries) and \
                    self.entries[e_idx]["track_id"] != s["track"].id:
                e_idx += 1
            if e_idx >= len(self.entries):
                break
            rows.append(e_idx)
            e_idx += 1
        ncol = len(SetListView.HEADERS)

        def paint_row(item, col, seam_text, tip):
            for c in range(ncol):
                item.setForeground(c, col)
                item.setToolTip(c, tip)
            item.setText(SetListView.SEAM_COL, seam_text)

        for k, s in enumerate(slots):
            if k >= len(rows):
                break
            item = self.set_list.item(rows[k])
            if item is None:
                continue
            if k == 0:
                paint_row(item, GOOD, "", "opener - no inbound seam")
                continue
            prev = slots[k - 1]                  # the seam INTO this track
            si = prev.get("seam_info") or {}
            p = prev.get("transition") or {}
            style = p.get("style")
            fade = bool(si.get("fade") or style == "long_fade")
            if si.get("floor") is not None and si["floor"] < 0.15:
                col = BAD
            elif fade:
                # A fade opts out of beat/key physics - tempo-clash and
                # key-clash warnings just restate WHY it's a fade.
                col = FADE
            elif prev.get("warnings"):
                col = WARN
            else:
                col = GOOD
            # Inbound adjacency estimate in the 'seam' column + the words
            # behind it in the tooltip (fades show as 'fade' - the
            # estimate would just restate why it's a fade).
            if fade:
                seam_text = "fade"
            else:
                est = self._seam_estimate(si, p)
                seam_text = f"{self._q_bar(est)}{est:.2f}"
            tip = seam_tooltip(prev["track"], s["track"], p, si)
            extra = list(prev.get("warnings") or [])
            if extra:
                tip += "\ncompiler notes: " + "; ".join(extra)
            paint_row(item, col, seam_text, tip)
            # Rhythm strip vs the PRECEDING song: clash-color the steps at
            # the compiled seam alignment; fades keep the plain pattern
            # (beat physics don't apply there).
            rt = si.get("rhythm")
            item.setData(
                SetListView.RHY_COL, self.RHY_ROLE,
                _rhythm_row_payload(s["track"], prev["track"], rt)
                if rt and not fade else _rhythm_row_payload(s["track"]))

    def _reordered(self, *a):
        # Entries ride the items via ENTRY_ROLE (column 0) - never map
        # back through display text.
        self._snapshot_undo()            # self.entries still pre-drag here
        self.entries = [e for e in
                        (self.set_list.item(i).data(0, self.ENTRY_ROLE)
                         for i in range(self.set_list.count()))
                        if e is not None]
        self.recompile()

    def _toggle_anchor(self, item):
        self._snapshot_undo()
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
        self._snapshot_undo()
        self.entries[i]["pin_type"] = "anchor"
        self.entries[i]["target_offset_min"] = self.anchor_min.value() or None
        self._rebuild()
        self.recompile()

    def _pin_style(self, style):
        """Pin (or clear) the transition style of the seam INTO the
        selected song - stored on its entry (style_override), the same
        field compile_plan and the live order-mode honor."""
        i = self.set_list.currentRow()
        if not (0 <= i < len(self.entries)):
            return
        if i == 0 and style:
            self.status.setText(
                "the first song has no incoming transition to pin")
            return
        self._snapshot_undo()
        self.entries[i]["style_override"] = style
        self._rebuild()
        self.set_list.setCurrentRow(i)
        self.recompile()
        self.status.setText(
            f"seam into slot {i + 1} pinned to {style} - check the ↳ line "
            "(a 'style pin refused' warning = a gate vetoed it)"
            if style else "style pin cleared - the brain rolls again")

    def _render_stems_selected(self):
        """Single-song stem render, IN PLACE: the job runs in the
        background, this tab's status line narrates, and the compiled
        plan refreshes on finish (stem styles + pins unlock) - no tab
        switch (focus theft, user-reported)."""
        i = self.set_list.currentRow()
        if not (0 <= i < len(self.entries)):
            return
        t = self._entry_track(self.entries[i])
        if t is None:
            self.status.setText("that entry's track is missing from the "
                                "library")
            return
        if getattr(t, "has_stems", False):
            self.status.setText(f"'{t.title[:30]}' already has stems - "
                                "re-rendering anyway")
        self.planner.render_stems(t, on_status=self.status.setText)

    def _remove_entry(self):
        i = self.set_list.currentRow()
        if 0 <= i < len(self.entries):
            self._snapshot_undo()
            self.entries.pop(i)
            self._rebuild()
            self.recompile()

    def remove_duplicates(self):
        self._snapshot_undo()
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
            keys = dup_keys(t) if t is not None else []
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
    # -- long planning ops (all off the GUI thread) ----------------------------
    def _run_plan_op(self, label, fn, apply_fn):
        """Run one planning operation on a worker thread. The beam searches
        (optimize/shape), whole-library chains (suggest/autofill) and pair
        scoring (alternatives/bridge) take seconds on a big library - inline
        they froze the whole window. One op at a time; the plan buttons grey
        out and the status line says what's cooking. `fn` runs on the worker
        (must only touch snapshots it captured); `apply_fn(result)` runs back
        on the GUI thread."""
        if self._op is not None and self._op.isRunning():
            self.status.setText("another planning operation is running...")
            return
        self.status.setText(label + "...")
        for b in self._plan_btns:
            b.setEnabled(False)

        def _done(result):
            for b in self._plan_btns:
                b.setEnabled(True)
            if isinstance(result, dict) and "error" in result:
                self.status.setText(f"{label} failed: {result['error']}")
                return
            apply_fn(result)
        self._op = PlanOpWorker(fn)
        self._op.done.connect(_done)
        self._op.start()

    def _apply_entries(self, entries):
        self._snapshot_undo()
        self.entries = entries
        self._rebuild()
        self.recompile()
        self.status.setText("")

    def build_set(self):
        """The one-button ordering flow: suggest a set for the theme +
        target length, then impose the chosen shape (order_by_shape keeps
        seams beat-matchable while it curves), or - for 'flat' - clean the
        seams with the beam optimizer instead. Composing the ops here
        replaces the old which-of-four-buttons ritual; the pieces stay
        available under More for surgical edits."""
        if self.entries and QMessageBox.question(
                self, "Build set", "Replace the current set?") \
                != QMessageBox.StandardButton.Yes:
            return
        lib, theme = self.planner.library, self.theme()
        minutes = float(self.minutes_spin.value())
        metric = self.shape_metric.currentText()
        shape = self.shape_curve.currentText()

        def run():
            entries = SL.suggest_set(lib, theme, minutes)
            if len(entries) > 2:
                if shape == "flat":
                    entries = SL.optimize_order(lib, entries, theme)
                else:
                    entries = SL.order_by_shape(lib, entries, theme,
                                                metric=metric, shape=shape)
            return entries
        self._run_plan_op(
            f"building the set (suggest → "
            f"{'optimize' if shape == 'flat' else f'{metric} {shape}'})",
            run, self._apply_entries)

    def suggest(self):
        if self.entries and QMessageBox.question(
                self, "Suggest set", "Replace the current set?") \
                != QMessageBox.StandardButton.Yes:
            return
        lib, theme = self.planner.library, self.theme()
        minutes = float(self.minutes_spin.value())
        self._run_plan_op("suggesting a set",
                          lambda: SL.suggest_set(lib, theme, minutes),
                          self._apply_entries)

    def optimize(self):
        lib, theme = self.planner.library, self.theme()
        entries = list(self.entries)
        self._run_plan_op("optimizing order (beam search)",
                          lambda: SL.optimize_order(lib, entries, theme),
                          self._apply_entries)

    def autofill(self):
        lib, theme = self.planner.library, self.theme()
        entries = list(self.entries)
        self._run_plan_op("solving anchor timing + fills",
                          lambda: SL.autofill(lib, entries, theme),
                          self._apply_entries)

    def apply_shape(self):
        if len(self.entries) <= 2:
            self.status.setText("build a set first, then shape its curve.")
            return
        lib, theme = self.planner.library, self.theme()
        entries = list(self.entries)
        metric = self.shape_metric.currentText()
        shape = self.shape_curve.currentText()
        self._run_plan_op(
            f"shaping ({metric} {shape})",
            lambda: SL.order_by_shape(lib, entries, theme,
                                      metric=metric, shape=shape),
            self._apply_entries)

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
        self.notes_edit.setText(sl.get("notes") or "")
        self._undo_stack.clear()         # a loaded set is a fresh timeline
        self._redo_stack.clear()
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
        self._undo_stack.clear()
        self._redo_stack.clear()
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
        # Plan-level metadata rides with the set: theme + compiled length
        # (the live system runs the set's own arc clock off total_s) and
        # the operator's notes.
        self.planner.db.set_setlist_meta(
            self.setlist_id, theme=self.theme_combo.currentText(),
            notes=self.notes_edit.text(),
            total_s=(self.compiled or {}).get("total_s"))
        self.refresh_setlists()
        self._select_combo_silently(self.setlist_id)
        self.status.setText(f"saved {len(self.entries)} tracks.")

    def _push_live(self, mode="order"):
        """Save, then load this set into the running show via the web
        controller's HTTP action route (planner and show are separate
        processes; the DB is the payload, this is just the trigger)."""
        self.save_set()
        if self.setlist_id is None:
            return                       # user cancelled the save prompt
        row = self.planner.db.conn.execute(
            "SELECT name FROM setlists WHERE id = ?",
            (self.setlist_id,)).fetchone()
        if row is None:
            return
        name = row["name"]
        import urllib.request
        base = os.environ.get("DJ_SHOW_URL", "http://localhost:5000")
        action = "setlist" if mode == "order" else "setlist_pool"
        req = urllib.request.Request(
            base + "/api/dj/action",
            data=json.dumps({"action": action, "value": name}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=2.0):
                pass
            self.status.setText(
                f"pushed '{name}' to the live show ({mode} mode).")
        except Exception:
            self.status.setText(
                f"live show not reachable at {base} - set saved; "
                "load it from the web panel when the show is up.")

    def _selected_slot_start(self):
        """Wall-clock offset of the SELECTED entry's compiled slot, so Play
        starts from the highlighted song. 0.0 when nothing/first selected."""
        row = self.set_list.currentRow()
        slots = (self.compiled or {}).get("slots") or []
        if row <= 0 or not slots:
            return 0.0
        if row < len(self.entries):
            target = self.entries[row]
            for sl in slots:          # compiler may skip missing tracks -
                if sl.get("entry") is target:      # match the entry itself
                    return float(sl.get("start_offset_s") or 0.0)
        if row < len(slots):          # fallback: positional
            return float(slots[row].get("start_offset_s") or 0.0)
        return 0.0

    def _play_set(self):
        if not self.compiled:
            self.status.setText("build a set first")
            return
        start_s = self._selected_slot_start()
        self.planner.mix_tab.play_set(start_s)
        which = "from the selected song" if start_s > 0.0 else "from the top"
        self.status.setText(
            f"playing set {which} (Mix tab shows the live timeline).")

    def delete_set(self):
        if self.setlist_id is not None and QMessageBox.question(
                self, "Delete", "Delete this setlist?") \
                == QMessageBox.StandardButton.Yes:
            SL.delete_setlist(self.planner.db, self.setlist_id)
            self.setlist_id = None
            self.entries = []
            self._undo_stack.clear()
            self._redo_stack.clear()
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
        # A recompile rebuilds the plan list, which used to DROP the
        # selected seam line - the highlight vanished and the inspector
        # blanked, which read as "my seam turned grey" (user-reported
        # right after rating a seam, since rating recompiles). Capture the
        # selected seam's track-id pair before clearing and re-select the
        # matching seam after (identity survives reorders, not just
        # index).
        sel_pair = None
        cur = self.plan_list.currentItem()
        if cur is not None and self.compiled:
            ci = cur.data(Qt.ItemDataRole.UserRole)
            old = self.compiled.get("slots") or []
            if ci is not None and 0 <= ci < len(old) - 1:
                sel_pair = (old[ci]["track"].id, old[ci + 1]["track"].id)
        self.compiled = result
        self._suggest_timer.start(400)   # set changed -> refresh suggestions
        self._color_set_list(result)
        self.plan_list.clear()
        for i, s in enumerate(result["slots"]):
            t = s["track"]
            mins = s["start_offset_s"] / 60.0
            ago = self._played_ago(t.id)
            QListWidgetItem(
                f"{int(mins):3d}:{int(s['start_offset_s'] % 60):02d}  "
                f"{t.title}  ({t.bpm:.0f} {t.camelot}, "
                f"{s['play_s'] / 60.0:.1f} min)"
                + (f"  · played {ago}" if ago else ""), self.plan_list)
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
                # Rhythm chips (word-first, worst term only, '?' = shaky
                # grid). A fade opts out of beat physics - the chips would
                # just restate why it's a fade.
                if not si.get("fade"):
                    badges.extend(seam_chips(p, si))
                pm = si.get("pair_mem")
                if pm:
                    badges.append("★ mixed well before" if pm > 1.0
                                  else "✖ rough before")
                # NIGHT EVIDENCE: this exact pairing was measured live
                # (seam_quality in logs/dj_*.jsonl) - the visibility the
                # numeric pair memory never had.
                nv = getattr(self.planner, "night_verdicts", {}) \
                    .get((t.title, nxt.title))
                if nv:
                    last = nv[-1]
                    d = f"{last['date'][4:6]}-{last['date'][6:8]}"
                    if any(v["rough"] for v in nv):
                        badges.append(f"✖ flammed live {d}")
                    else:
                        badges.append(f"✓ clean live {d}"
                                      + (f" ×{len(nv)}" if len(nv) > 1
                                         else ""))
                warn = ("   ⚠ " + "; ".join(s["warnings"])
                        if s["warnings"] else "")
                item = QListWidgetItem(
                    f"      ↳ {p['style']} @ {p['out_s']:.0f}s "
                    f"(rate {p['rate']:.3f}, seam {p['pair_score']:.2f}"
                    + ("".join(", " + b for b in badges)) + ") "
                    f"→ {nxt.title}{warn}", self.plan_list)
                item.setData(Qt.ItemDataRole.UserRole, i)
                item.setToolTip(seam_tooltip(t, nxt, p, si))
                if si.get("floor") is not None and si["floor"] < 0.15:
                    item.setForeground(QColor(230, 110, 110))
                elif si.get("fade"):
                    item.setForeground(QColor(160, 160, 170))
                elif s["warnings"]:
                    item.setForeground(QColor(255, 170, 100))
        # Re-select the same seam (by track-id pair) so the highlight and
        # the seam inspector survive the rebuild.
        if sel_pair:
            slots = result["slots"]
            for r in range(self.plan_list.count()):
                it = self.plan_list.item(r)
                ci = it.data(Qt.ItemDataRole.UserRole)
                if ci is not None and ci + 1 < len(slots) and \
                        (slots[ci]["track"].id,
                         slots[ci + 1]["track"].id) == sel_pair:
                    self.plan_list.setCurrentRow(r)
                    break
        self._update_strip(result)
        self.status.setText(self._report_card(result))
        self._update_worst(result)
        self.planCompiled.emit(result)

    def _engine_changed(self, name):
        """Switch the tempo engine for this planner session: auditions,
        previews and plans all read it dynamically. Recompiles because
        planning SEMANTICS follow the engine (varispeed splits tempo
        bends across decks; keylock enables the key-shift rescue)."""
        import os as _os
        from lib.dj import stretch_engine_name
        if name.startswith("rubberband"):
            _os.environ["DJ_STRETCH_ENGINE"] = "rubberband"
            _os.environ["DJ_RB_ENGINE"] = \
                "faster" if name.endswith("crisp") else "finer"
        else:
            _os.environ["DJ_STRETCH_ENGINE"] = name
        resolved = stretch_engine_name()
        if resolved == "rubberband":
            resolved = name              # variant is ours to report
        if resolved != name:
            self.status.setText(
                f"{name} unavailable - pip install -r "
                f"requirements-dj-keylock.txt (using {resolved})")
        else:
            self.status.setText(
                f"tempo engine: {name} - re-audition a seam to hear it; "
                f"set dj.stretch_engine in config.yaml for the live show")
        self.recompile()

    def _seam_selected(self, _row=None):
        """Feed the seam inspector + transition options from the selected
        ↳ seam line; any other row (or a stale index) clears them."""
        item = self.plan_list.currentItem()
        idx = item.data(Qt.ItemDataRole.UserRole) if item else None
        slots = (self.compiled or {}).get("slots") or []
        if idx is None or not (0 <= idx < len(slots) - 1):
            self.seam_inspector.clear()
            self.style_opts.clear()
            return
        s = slots[idx]
        self.seam_inspector.set_seam(s["track"], slots[idx + 1]["track"],
                                     s.get("transition"),
                                     s.get("seam_info"))
        self._update_style_options(idx)

    def _update_style_options(self, idx):
        """One row per style for the seam slots[idx] -> slots[idx+1]:
        chosen/pinned marker, dice-odds share, or the gate that removed
        it. Data straight from the compiled plan's diag - the same record
        the armed log carries at night."""
        self.style_opts.clear()
        slots = (self.compiled or {}).get("slots") or []
        if not (0 <= idx < len(slots) - 1):
            return
        p = slots[idx].get("transition") or {}
        diag = p.get("diag") or {}
        menu = diag.get("menu") or {}
        gated = diag.get("gated") or {}
        fade_reason = diag.get("fade_reason")
        pin_rec = diag.get("style_pin") or {}
        entry_in = slots[idx + 1]["entry"]
        pin = entry_in.get("style_override")
        chosen = p.get("style")
        total = sum(menu.values()) or 1.0

        def add(text, style_key, color=None):
            it = QListWidgetItem(text, self.style_opts)
            it.setData(Qt.ItemDataRole.UserRole, idx)
            it.setData(Qt.ItemDataRole.UserRole + 1, style_key)
            if color:
                it.setForeground(color)
            return it

        if pin_rec and not pin_rec.get("honored"):
            add(f"⚠ pin '{pin_rec.get('want')}' refused "
                f"({pin_rec.get('why_not')}) - plays {chosen}", "__info__",
                QColor(255, 170, 100))
        add(("→ " if pin is None else "  ") + "auto (let the dice roll)",
            "__auto__")
        # Per-style predicted quality: the seam's physics through each
        # style's exposure profile (style_rating). Playable styles sort
        # best-first; the odds stay so you can see quality and dice
        # disagree (a great-rated style the theme rarely rolls = pin it).
        from tools.dj.planner.copilot import STYLES
        si = slots[idx].get("seam_info") or {}
        a_t, b_t = slots[idx]["track"], slots[idx + 1]["track"]

        def q(st):
            v = style_rating(si, p, a_t, b_t, st)
            return f"q {v:.2f} " + chr(9601 + int(v * 7 + 0.5))

        # ▤ marks the styles that PLAY THROUGH STEMS on this seam (which
        # sides have files decides which styles qualify); the duck note
        # shows when a blend will sidestep a vocal-over-vocal clash.
        _STEM_SIDES = {"stem_drum_swap": "ab", "stem_bass_swap": "ab",
                       "drum_bridge": "ab", "acapella_out": "a",
                       "melody_carry": "a", "acapella_in": "b"}
        a_has = getattr(a_t, "has_stems", False)
        b_has = getattr(b_t, "has_stems", False)

        def stem_mark(st):
            need = _STEM_SIDES.get(st)
            if not need:
                return ""
            ok = (("a" not in need or a_has)
                  and ("b" not in need or b_has))
            return "▤" if ok else ""

        duck = bool(p.get("duck_vocal_a"))
        rows = []
        for st in STYLES:
            mark = "▶" if st == chosen else ("📌" if st == pin else " ")
            if st in menu:
                extra = ""
                if stem_mark(st):
                    extra = " ▤stems"
                elif duck and st == chosen:
                    extra = " ▤duck"
                rows.append((0, -style_rating(si, p, a_t, b_t, st),
                             f"{mark} {st:16s} {q(st)}  "
                             f"{100.0 * menu[st] / total:3.0f}% odds"
                             + extra,
                             st, None))
            elif st == "long_fade":
                note = (f"forced ({fade_reason})" if fade_reason
                        else "pin only (deliberate fade)")
                rows.append((1, 0.0, f"{mark} {st:16s} {q(st)}  {note}",
                             st, None if fade_reason
                             else QColor(150, 150, 160)))
            elif st in gated:
                rows.append((2, 0.0,
                             f"{mark} {st:16s} q  -    gated: {gated[st]}",
                             st, QColor(150, 150, 160)))
            elif fade_reason:
                rows.append((3, 0.0, f"{mark} {st:16s} q  -    "
                             f"unavailable (fade forced: {fade_reason})",
                             st, QColor(150, 150, 160)))
            else:
                rows.append((3, 0.0, f"{mark} {st:16s} q  -    "
                             "not in this theme's menu",
                             st, QColor(150, 150, 160)))
        for _grp, _neg, text, st, color in sorted(rows,
                                                  key=lambda r: r[:2]):
            add(text, st, color)

    def _style_option_clicked(self, item):
        style = item.data(Qt.ItemDataRole.UserRole + 1)
        idx = item.data(Qt.ItemDataRole.UserRole)
        if style == "__info__" or idx is None:
            return
        slots = (self.compiled or {}).get("slots") or []
        if not (0 <= idx < len(slots) - 1):
            return
        entry_in = slots[idx + 1]["entry"]     # same dict as in self.entries
        want = None if style == "__auto__" else style
        if entry_in.get("style_override") == want:
            return                             # no-op click
        self._snapshot_undo()
        entry_in["style_override"] = want
        self._rebuild()
        self.recompile()                       # reselects this seam after
        self.status.setText(
            f"seam pinned to {want} - audition to hear it" if want
            else "pin cleared - the dice roll again")

    def _update_strip(self, result):
        theme = self.theme()
        slots = result["slots"]
        strip = []
        for i, s in enumerate(slots):
            d = {"off": s["start_offset_s"], "play": s["play_s"],
                 "energy": s["track"].energy_proxy(),
                 "bpm": s["track"].bpm,
                 "anchor": s["entry"].get("pin_type") == "anchor",
                 "seam": s.get("seam_info"),
                 "title": s["track"].title,
                 "chips": seam_chips(s.get("transition"),
                                     s.get("seam_info")),
                 "warn": bool(s["warnings"]),
                 "warnings": list(s["warnings"])}
            if i + 1 < len(slots) and s.get("transition"):
                d["tip"] = seam_tooltip(s["track"], slots[i + 1]["track"],
                                        s["transition"], s.get("seam_info"))
            strip.append(d)
        arc = [(i / 24.0, max(0.0, min(1.0, theme.arc_target(i / 24.0))))
               for i in range(25)]
        self.arc_strip.set_data(strip, arc,
                                target_s=float(self.minutes_spin.value())
                                * 60.0)

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

    def _update_worst(self, result, n=5):
        """Rank the set's seams worst-first (warnings outrank raw score)
        and list the bottom N as clickable repair targets."""
        self.worst_list.clear()
        slots = result["slots"]
        seams = []
        for i, s in enumerate(slots[:-1]):
            p = s.get("transition")
            if p:
                seams.append((len(s["warnings"]),
                              -(p.get("pair_score") or 0.0), i, s, p))
        seams.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for nw, negs, i, s, p in seams[:n]:
            if nw == 0 and -negs >= 0.15:
                continue                 # healthy seam - nothing to repair
            nxt = slots[i + 1]["track"]
            why = "; ".join(s["warnings"][:2]) \
                or ("deliberate fade" if p["style"] == "long_fade"
                    else f"weak seam {-negs:.2f}")
            it = QListWidgetItem(
                f"{i + 1:2d}→{i + 2:<2d} {_clip(s['track'].title, 20)}→"
                f"{_clip(nxt.title, 20)} {p['style']:14} {why}",
                self.worst_list)
            it.setData(Qt.ItemDataRole.UserRole, i)
            it.setForeground(QColor(230, 110, 110) if nw
                             else QColor(255, 170, 100))

    def _worst_clicked(self, item):
        i = item.data(Qt.ItemDataRole.UserRole)
        if i is not None:
            self._strip_clicked(i)

    def _played_ago(self, track_id):
        """'today' / 'Nd ago' from play_history, or None if never played
        live. The recency chip that keeps a set from accidentally leaning
        on last Saturday's exact records."""
        ts = getattr(self.planner, "last_played", {}).get(track_id)
        if not ts:
            return None
        d = (time.time() - ts) / 86400.0
        if d < 1.0:
            return "today"
        return f"{d:.0f}d ago"

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
        self._snapshot_undo()
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
        self._run_plan_op("scoring alternatives",
                          lambda: self._slot_alternatives(i),
                          lambda alts: self._show_alternatives(i, alts))

    def _shop_gaps(self):
        """Weak seams -> a Beatport shopping trip. Every seam the compiler
        flagged becomes a pickable gap; the chosen one runs a connector
        search in the Discover tab (genre charts + both-sides BPM window +
        bridge-fit ranking). The honest answer to 'my library can't make
        this set work'."""
        dt = self.planner.discover_tab
        if dt is None:
            self.status.setText("Discover tab unavailable (Beatport module "
                                "didn't load)")
            return
        if not self.compiled or not self.compiled.get("slots"):
            self.status.setText("compile a set first")
            return
        slots = self.compiled["slots"]
        gaps = []
        for i, s in enumerate(slots[:-1]):
            if not s.get("transition"):
                continue
            si = s.get("seam_info") or {}
            reasons = list(s.get("warnings") or [])
            if si.get("key_fit", 1.0) < 0.55 and not any(
                    w.startswith("key clash") for w in reasons):
                reasons.append(f"key fit {si['key_fit']}")
            if si.get("floor") is not None and si["floor"] < 0.15:
                reasons.append(f"energy hole {si['floor']}")
            if reasons:
                gaps.append((i, s["track"], slots[i + 1]["track"],
                             "; ".join(reasons)))
        if not gaps:
            self.status.setText("no flagged seams - nothing to shop")
            return
        menu = QMenu(self)
        for i, a, b, why in gaps[:12]:
            act = menu.addAction(f"{i + 1}. {a.title[:24]} → {b.title[:24]}"
                                 f"   ({why[:52]})")
            act.setData((a.id, b.id))
        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            return
        aid, bid = chosen.data()
        by_id = {t.id: t for t in self.planner.library}
        a, b = by_id.get(aid), by_id.get(bid)
        if a is None or b is None:
            self.status.setText("set changed - recompile and retry")
            return
        dt.shop_gap(a, b, f"{a.title[:20]} → {b.title[:20]}")
        self.planner.tabs.setCurrentWidget(dt)

    def _show_alternatives(self, i, alts):
        if not alts:
            self.status.setText("no tempo-reachable alternative for this"
                                " slot")
            return
        self.status.setText("")
        menu = QMenu(self)
        for v, t in alts:
            act = menu.addAction(f"{t.title[:44]}  ({t.bpm:.0f} "
                                 f"{t.camelot})  fit {v:.3f}")
            act.setData(t.id)
        chosen = menu.exec(QCursor.pos())
        if chosen is None:
            self.status.setText("")
            return
        if not (0 <= i < len(self.entries)):     # list edited while scoring
            self.status.setText("the set changed while scoring - try again")
            return
        self._snapshot_undo()
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
        lib, theme = self.planner.library, self.theme()
        excl = {e["track_id"] for e in self.entries}
        self._run_plan_op(
            "searching for a bridge",
            lambda: SL.bridge(lib, a, b, theme, exclude_ids=excl),
            lambda res: self._apply_bridge(i, a, b, res))

    def _apply_bridge(self, i, a, b, res):
        chain, score = res
        if not chain:
            self.status.setText(f"no bridge beats the direct seam "
                                f"{a.title[:20]} -> {b.title[:20]}")
            return
        if not (0 <= i < len(self.entries) - 1
                and self.entries[i].get("track_id") == a.id):
            self.status.setText("the set changed while searching - try again")
            return
        self._snapshot_undo()
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
        # Remember WHAT is being auditioned so a thumb afterwards charges
        # the right pair+style (the list may recompile while audio plays).
        self._last_aud = {"a": slots[i]["track"], "b": slots[i + 1]["track"],
                          "style": (slots[i]["transition"] or {}).get("style")}
        self.rate_up_btn.setEnabled(False)
        self.rate_dn_btn.setEnabled(False)
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
        self.rate_up_btn.setEnabled(True)
        self.rate_dn_btn.setEnabled(True)

    def _rate_seam(self, up):
        """Store a user thumb for the last-auditioned seam and fold it
        into the session immediately: pair/class/style memory reload and
        the plan recompiles, so the ★/✖ badge and scoring shift right
        away instead of on the next launch."""
        la = self._last_aud
        if not la:
            self.status.setText("audition a seam first, then rate it")
            return
        try:
            self.planner.db.add_seam_feedback(
                la["a"].id, la["b"].id, la["style"], up, source="user")
        except Exception as e:
            self.status.setText(f"could not store rating: {e}")
            return
        try:
            _b = Brain([], get_theme("groove"))
            _b.load_pair_memory(self.planner.db)
            self.planner.pair_memory = _b.pair_memory
        except Exception:
            pass
        self.status.setText(
            f"{'👍 good' if up else '👎 rough'} seam remembered: "
            f"{la['a'].title[:22]} → {la['b'].title[:22]} "
            f"({la['style'] or 'seam'}) - steers this pair, its class, "
            f"and the style from now on")
        self.recompile()


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

    def play_set(self, start_s=0.0):
        if not self.planner.set_tab.compiled:
            self.status.setText("build a set first")
            return
        self.pause_btn.setChecked(False)
        self.planner.claim_playback("preview")
        self.preview.play_at(float(start_s or 0.0))

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
# Nights tab: post-mortem of what the live DJ actually did
# ==========================================================================

class NightsTab(QWidget):
    """Reads logs/dj_*.jsonl - the same evidence tools/dj/dj_review.py
    reports on - and shows each night's play-by-play plus the engine's own
    per-seam verdicts (seam_quality events), so last weekend's flams are
    visible WHILE building next weekend's set. Read-only: the numeric
    learning already happens via seam_feedback/pair memory; this tab is
    the visibility that memory never had."""

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        self._nights = []              # [(date, events)] newest first
        h = QHBoxLayout(self)
        left = QVBoxLayout()
        rb = QPushButton("↻ Reload logs")
        rb.setToolTip("Re-read logs/dj_*.jsonl (e.g. after a show) and "
                      "refresh the Set tab's night badges.")
        rb.clicked.connect(self.reload_logs)
        left.addWidget(rb)
        self.night_list = QListWidget()
        self.night_list.setFont(_mono_font())
        self.night_list.currentRowChanged.connect(self._night_selected)
        left.addWidget(self.night_list, 1)
        h.addLayout(left, 2)
        right = QVBoxLayout()
        right.addWidget(_no_width_floor(QLabel(
            "Measured seams (engine verdicts; red = rough by the same bar "
            "that charges pair memory; 'rated' = your thumbs that night):")))
        self.seam_tree = QTreeWidget()
        self.seam_tree.setHeaderLabels(
            ["out → in", "style", "verdict", "rated",
             "flam (beats)", "hole (s)"])
        self.seam_tree.setRootIsDecorated(False)
        self.seam_tree.setFont(_mono_font())
        self.seam_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        right.addWidget(self.seam_tree, 3)
        right.addWidget(QLabel("Play-by-play:"))
        self.track_list = QListWidget()
        self.track_list.setFont(_mono_font())
        right.addWidget(self.track_list, 2)
        h.addLayout(right, 3)

    def reload_logs(self):
        self.planner.reload_night_data(force=True)
        self.refresh()
        # The Set tab's seam badges read night_verdicts - refresh them.
        if self.planner.set_tab.entries:
            self.planner.set_tab.recompile()

    def refresh(self):
        from lib.dj.review import night_summary
        self._nights = list(reversed(
            getattr(self.planner, "night_logs", [])))
        self.night_list.clear()
        for s in reversed(night_summary(
                getattr(self.planner, "night_logs", []))):
            d = s["date"]
            line = (f"{d[:4]}-{d[4:6]}-{d[6:]}  {s['hours']:4.1f}h  "
                    f"{s['plays']:3d} tracks  {s['seams']:3d} seams  "
                    f"{s['rough']:2d} rough  {s['skips']:3d} skips")
            if s.get("fb_up") or s.get("fb_down"):
                line += f"  {s['fb_up']}👍 {s['fb_down']}👎"
            if s.get("starved"):
                line += (f"  ⚠{s['starved']} starved"
                         + (f" ({s['starved_stem']} on stem seams)"
                            if s.get("starved_stem") else ""))
            if s["themes"]:
                line += "  [" + "/".join(s["themes"][:4]) + "]"
            it = QListWidgetItem(line, self.night_list)
            if s["rough"]:
                it.setForeground(QColor(255, 170, 100))
        if self.night_list.count():
            self.night_list.setCurrentRow(0)

    def _night_selected(self, row):
        from lib.dj.review import night_seam_rows, night_tracks
        self.seam_tree.clear()
        self.track_list.clear()
        if not (0 <= row < len(self._nights)):
            return
        _, evs = self._nights[row]
        for r in night_seam_rows(evs):
            rated = ("" if r["rated"] is None
                     else ("👍 good" if r["rated"] else "👎 bad"))
            it = QTreeWidgetItem([
                f"{r['a'][:34]} → {r['b'][:34]}", r["style"], r["verdict"]
                + (" (urgent)" if r["urgent"] else ""),
                rated, f"{r['max_err_beats']:.3f}", f"{r['hole_s']:.2f}"])
            if r["rough"]:
                for c in range(6):
                    it.setForeground(c, QColor(230, 110, 110))
            elif r["verdict"] != "clean":
                for c in range(6):
                    it.setForeground(c, QColor(255, 170, 100))
            if r["rated"] is not None:      # your ear outranks the meter
                it.setForeground(3, QColor(120, 210, 120) if r["rated"]
                                 else QColor(240, 100, 100))
            self.seam_tree.addTopLevelItem(it)
        for t in night_tracks(evs):
            mark = ("  " if t["rated"] is None
                    else ("👍" if t["rated"] else "👎"))
            it = QListWidgetItem(
                f"{mark} {_clip(t['title'], 42)} {_clip(t['artist'], 24)} "
                f"via {t['via']}", self.track_list)
            if t["rated"] is not None:
                it.setForeground(QColor(120, 210, 120) if t["rated"]
                                 else QColor(240, 100, 100))


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
        self._stem_proc = None       # ONE stem-render subprocess, planner-wide
        self._stem_job = None        # (track_id, title, on_status, on_done)
        self._stem_queue = []        # waiting (track, model, cb, cb) FIFO

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.library_tab = LibraryTab(self)
        self.analysis_tab = AnalysisTab(self)
        self.set_tab = SetTab(self)
        self.mix_tab = MixTab(self)
        self.nights_tab = NightsTab(self)
        self.tabs.addTab(self.library_tab, "Library")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.set_tab, "Set")
        self.tabs.addTab(self.mix_tab, "Mix")
        # Lab: ONE seam, rendered once, seen through three lenses -
        # Scope (what the blend does), Beat (per-deck bands + measured
        # kicks) and Exit (where the track is left, vs the old engine) -
        # plus the good/passable/bad rating treadmill that feeds
        # seam_feedback and logs/seam_lab_ratings.jsonl.
        #
        # Replaced four tabs on 2026-08-08 (Seam Lab, Beat Check, Exit
        # Compare, Gate Check). They shared a spine - pick a seam, render,
        # play - and differed only in what they drew, so each held its own
        # seam and nothing seen in one could be checked against another.
        # Gate Check's trial loop was retired rather than ported: its
        # verdicts had already done their job (they retired the 20ms kick
        # screen - see tools/tests/_dj_kickdelta_test.py), and
        # lib/dj/gateprobe.py plus logs/gate_ratings.jsonl are still on
        # disk if it is ever wanted back. Seam Lab's single-knob probe
        # staircase went the same way; tools/dj/planner/seamprobe.py
        # remains.
        from tools.dj.planner.lab import LabTab
        self.lab_tab = LabTab(self)
        self.tabs.addTab(self.lab_tab, "Lab")
        # Layer Lab (tools/dj/planner/layerlab.py) is SHELVED, not
        # deleted - see docs/DJ_README.md "Loop layer (SHELVED)". The
        # engine capability is intact; only the tab is unregistered.
        # Re-enable by restoring these three lines:
        #   from tools.dj.planner.layerlab import LayerLabTab
        #   self.layer_tab = LayerLabTab(self)
        #   self.tabs.addTab(self.layer_tab, "Layer Lab")
        self.layer_tab = None
        self.tabs.addTab(self.nights_tab, "Nights")
        # Discover (Beatport) is optional - only if the module imports.
        self.discover_tab = None
        try:
            from tools.dj.planner.discover import DiscoverTab
            self.discover_tab = DiscoverTab(self)
            self.tabs.addTab(self.discover_tab, "Discover")
        except Exception as e:
            print(f"[planner] Discover tab unavailable: {e}")

        self.library_tab.openAnalysis.connect(self._open_analysis)
        self.library_tab.addTracks.connect(self.set_tab.add_tracks)
        self.set_tab.planCompiled.connect(self.mix_tab.set_plan)

        # WINDOW WIDTH FLOOR: QTabWidget's minimum is the MAX over every
        # page, and a QComboBox's default size hint grows with its LONGEST
        # item - track titles, setlist names and folder paths were pinning
        # the whole window wider than a monitor (same failure the
        # narrative editor fixed 2026-07-26). Cap every combo's hint; they
        # still stretch with their layouts.
        for combo in self.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy
                .AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)

        # Planner-wide background-job readout (bottom status bar): what
        # the stem renderer is doing and what's waiting in its queue -
        # visible from EVERY tab, no focus changes.
        self.stem_status = _no_width_floor(QLabel(""))
        self.statusBar().addPermanentWidget(self.stem_status, 1)

        self.reload_library()
        self.set_tab.refresh_setlists()
        self.nights_tab.refresh()

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
            elif owner == "lab":
                self.lab_tab.stop_playback()
            elif owner == "preview":
                self.mix_tab.preview.stop()
            elif owner == "discover":
                if self.discover_tab is not None:
                    self.discover_tab.stop_preview()
        except Exception:
            pass

    def stop_all_playback(self):
        """ANY stop button stops ANY playing."""
        for o in ("analysis", "library", "seam", "lab", "preview",
                  "discover"):
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
        # Night-log evidence (parsed once; the Nights tab's reload button
        # forces a re-read) + last-played recency for the chips.
        self.reload_night_data()
        try:
            self.last_played = {
                r["track_id"]: r["last"] for r in self.db.conn.execute(
                    "SELECT track_id, MAX(started_at) AS last"
                    " FROM play_history GROUP BY track_id")}
        except Exception:
            self.last_played = {}
        self.library_tab.refresh()
        if not keep_analysis:
            self.analysis_tab.refresh_tracklist()
        # ANY library change (tags added/removed, cues marked) makes a
        # compiled plan stale - tags steer selection and user cues
        # OVERRIDE seam points, so the Set/Mix tabs must recompile.
        if self.set_tab.entries:
            self.set_tab.recompile()     # _compiled refreshes suggestions
        else:
            self.set_tab._suggest_timer.start(600)   # opener suggestions

    # ---- stem rendering: one background job, callable from ANY tab ------
    def render_stems(self, track, model=None, on_status=None, on_done=None):
        """Render ONE track's stems in a background subprocess WITHOUT
        stealing focus - the caller passes callbacks and stays where it
        is. One job runs at a time (demucs owns the GPU); further requests
        QUEUE and start automatically as each render finishes. Returns
        True when the job started or was queued."""
        if self._stem_proc is not None:
            running = self._stem_job[0] if self._stem_job else None
            if track.id == running or any(
                    t.id == track.id for t, *_ in self._stem_queue):
                if on_status:
                    on_status(f"'{track.title[:30]}' is already "
                              "rendering/queued")
                return False
            self._stem_queue.append((track, model, on_status, on_done))
            if on_status:
                on_status(f"queued '{track.title[:30]}' for stems "
                          f"(#{len(self._stem_queue)} in line)")
            self._update_stem_status()
            return True
        from lib.dj import vocals
        if not vocals.available():
            if on_status:
                on_status("stem renderer unavailable - pip install -r "
                          "requirements-dj-vocals.txt (torch + demucs)")
            return False
        model = model or self.analysis_tab.stem_model_box.currentText()
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._stem_out)
        proc.finished.connect(self._stem_done)
        proc.start(sys.executable,
                   [os.path.join(_REPO_ROOT, "tools", "dj", "dj_stems.py"),
                    "--dir", self.music_dir, "--track", str(track.id),
                    "--model", model])
        self._stem_proc = proc
        self._stem_job = (track.id, track.title, on_status, on_done)
        if on_status:
            on_status(f"separating '{track.title[:30]}' ({model})... "
                      "loading model")
        self._update_stem_status("loading model")
        self.analysis_tab._sync_stem_row()
        return True

    def _update_stem_status(self, line=""):
        """The status-bar readout: current render + waiting queue."""
        if self._stem_proc is None and not self._stem_queue:
            self.stem_status.setText("")
            return
        cur = self._stem_job[1][:28] if self._stem_job else "?"
        txt = f"▤ stems: rendering '{cur}'"
        if line:
            txt += f" - {line[:60]}"
        if self._stem_queue:
            txt += ("  ·  queued: " + ", ".join(
                t.title[:18] for t, *_ in self._stem_queue[:4]))
            if len(self._stem_queue) > 4:
                txt += f" +{len(self._stem_queue) - 4} more"
        self.stem_status.setText(txt)

    def _stem_out(self):
        if self._stem_proc is None or self._stem_job is None:
            return
        data = bytes(self._stem_proc.readAllStandardOutput()).decode(
            "utf-8", "replace")
        _tid, title, on_status, _cb = self._stem_job
        for line in data.splitlines():
            line = line.strip()
            if line and not line.startswith("PROGRESS"):
                if on_status:
                    on_status(f"separating '{title[:24]}': {line[:70]}")
                self._update_stem_status(line)

    def _stem_done(self, code, *_a):
        self._stem_proc = None
        tid, title, on_status, on_done = self._stem_job
        self._stem_job = None
        from lib.dj.stems import has_stems as _hs
        ok = code == 0 and _hs(self.music_dir, tid)
        for t in self.library_all or self.library:
            if t.id == tid:
                t.has_stems = ok or getattr(t, "has_stems", False)
        self.library_tab.table.viewport().update()   # ✓ column
        if on_status:
            on_status(f"stems for '{title[:30]}': "
                      + ("rendered ✓" if ok else f"failed (exit {code}) - "
                         "see console"))
        if on_done:
            on_done(tid, ok)
        self.analysis_tab._sync_stem_row()
        # Fresh stems change what the style gates allow: a pinned stem
        # style that compiled as "refused (no_stems)" must clear on its
        # own, not wait for a manual edit (user-reported).
        if ok and self.set_tab.entries:
            self.set_tab._rebuild()          # ▤ column in the set list
            self.set_tab.recompile()
        if self._stem_queue:                 # next in line starts itself
            nt, nm, ns, nd = self._stem_queue.pop(0)
            self.render_stems(nt, model=nm, on_status=ns, on_done=nd)
        else:
            self._update_stem_status()       # clears the readout

    def reload_night_data(self, force=False):
        """Parse logs/dj_*.jsonl into night_logs + per-pair verdicts (the
        Set tab's 'flammed live' badges). Cached - the logs only change
        when a show plays - unless force."""
        if not force and getattr(self, "night_logs", None) is not None:
            return
        try:
            from lib.dj.review import load_nights, pair_verdicts
            self.night_logs = load_nights()
            self.night_verdicts = pair_verdicts(self.night_logs)
        except Exception as e:
            print(f"[planner] night logs unavailable: {e}")
            self.night_logs, self.night_verdicts = [], {}

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
        self._stem_queue = []    # BEFORE the kill: finished-signal fires
        for proc in (self.library_tab._structure_proc,
                     self.library_tab._stems_proc,
                     self.library_tab._pipe_proc,
                     self._stem_proc):
            if proc is not None:
                proc.kill()
                proc.waitForFinished(2000)
        self.library_tab._pipe = []
        self.library_tab._pipe_total = 0
        if self.set_tab._op is not None and self.set_tab._op.isRunning():
            self.set_tab._op.wait(3000)      # a beam search can't be killed
        # A render can't be killed, so shutdown waits it out.
        self.lab_tab.shutdown()
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
