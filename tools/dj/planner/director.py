"""Director tab: high-level behaviour over the DJ, for parties and events.

One screen, no scrolling: the picture of the run across the top (the four lanes, what played, the
playhead, what is planned ahead); below it on the left what is playing, what comes next and WHY, and the
songs (the pool ranked by fit, ✓ mixable / ✗ would be rejected, UP NEXT); on the right the ten dials in
two columns and the moments - NEXT, STAY, the row of five composed moments (SLAM, BREAK, SING, TAKEOVER,
BRIDGE: press = arm with a countdown, again = go now, right-click = cancel), GOOD, BAD. Explanations live
in tooltips. No lanes, no clips, no stems, no levels: the engines (lib/dj/system.py, lib/dj/remix.py) do
that under lib/dj/director.py.
"""
import os
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QPushButton, QVBoxLayout, QWidget, QSizePolicy, QTabWidget)

from tools.dj.planner.perform import STYLE, SetlistBox, _mmss
from tools.dj.planner.stemlanes import LANE_COLORS

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
# (FX is off the surface - the moments carry their own shapes now; the dial still exists for the controller and the web page)
DIAL_ORDER = ("layers", "mixing", "arc", "length", "energy", "tempo", "pace", "seams", "vocals", "variety", "loops",
              "moments", "bass", "tone", "level")
# the composed moments, in the order they sit on the row: key -> (label, one-song label or None)
MOMENT_ROW = (("slam", "SLAM"), ("break", "BREAK"), ("sing", "SING"), ("takeover", "TAKEOVER"), ("bridge", "BRIDGE"), ("build", "BUILD"), ("again", "AGAIN"))


class SongMap(QWidget):
    """One record's structure across its whole length: sections by colour (the scanner's intro / groove /
    build / breakdown / outro), the drops as white ticks, the measured hook in gold, where it sings in green,
    and the playhead. Under NOW and NEXT - the shape of the song at hand without zooming the picture."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.track = None
        self.pos_s = None
        self.setMinimumHeight(16)
        self.setMaximumHeight(16)

    def set(self, track, pos_s=None):
        if track is not self.track:
            self.track = track
            if track is not None:
                secs = track.sections or []
                parts = []
                for i, s in enumerate(secs):
                    k = s.get("kind") or "?"
                    if k == "groove" and i > 0 and secs[i - 1].get("kind") in ("build", "breakdown", "intro"):
                        k = "DROP"
                    parts.append(f"{k} {_mmss(s['start_s'])}")
                hk = getattr(track, "hook", None)
                tip = [f"{track.title} — {track.artist or ''}   {_mmss(track.duration_s)}", "  ·  ".join(parts)]
                if hk:
                    tip.append("hook at " + ", ".join(_mmss(x) for x in (hk.get("starts") or [hk["start_s"]])[:6]) + f"  ({_mmss(hk['end_s'] - hk['start_s'])} long)")
                self.setToolTip("\n".join(tip))
            else:
                self.setToolTip("")
        self.pos_s = pos_s
        self.update()

    def paintEvent(self, ev):
        from PyQt6.QtGui import QPainter, QColor, QPen
        from PyQt6.QtCore import QRectF, QPointF
        from tools.dj.planner.timeline import SECTION_COLORS
        qp = QPainter(self)
        W, H = self.width(), self.height()
        qp.fillRect(self.rect(), QColor(20, 20, 26))
        t = self.track
        if t is None or not (t.duration_s or 0):
            return
        dur = float(t.duration_s)
        x = lambda s: max(0.0, min(W, W * float(s) / dur))  # noqa: E731
        secs = t.sections or []
        for i, s in enumerate(secs):
            col = QColor(SECTION_COLORS.get(s.get("kind"), QColor(90, 90, 100)))
            e = s.get("energy")
            if e is not None:
                col.setAlpha(int(110 + 145 * max(0.0, min(1.0, float(e)))))
            qp.fillRect(QRectF(x(s["start_s"]), 2, max(1.0, x(s["end_s"]) - x(s["start_s"])), H - 8), col)
            if i > 0 and s.get("kind") == "groove" and secs[i - 1].get("kind") in ("build", "breakdown"):
                qp.setPen(QPen(QColor(255, 255, 255), 2))
                qp.drawLine(QPointF(x(s["start_s"]), 0), QPointF(x(s["start_s"]), H - 4))
        vc = (getattr(t, "axes", None) or {}).get("vc") or []
        hop = float((t.axes or {}).get("vc_hop") or 8.0)
        qp.setPen(QPen(QColor(120, 220, 140), 2))
        for tt, v in vc:
            if v is not None and float(v) >= 0.03:
                qp.drawLine(QPointF(x(tt), H - 4), QPointF(x(float(tt) + hop), H - 4))
        hk = getattr(t, "hook", None)
        if hk:
            qp.setPen(QPen(QColor(240, 200, 90), 3))
            length = float(hk.get("end_s", 0) - hk.get("start_s", 0)) or 15.0
            for st in (hk.get("starts") or [hk["start_s"]]):
                qp.drawLine(QPointF(x(st), H - 1), QPointF(x(st + length), H - 1))
        if self.pos_s is not None:
            qp.setPen(QPen(QColor(255, 255, 255), 2))
            qp.drawLine(QPointF(x(self.pos_s), 0), QPointF(x(self.pos_s), H))


class StemViewer(QWidget):
    """SONG PARTS: a floating window that follows the highlighted song (SONGS or UP NEXT) - its four stems as
    spectrogram rows in song time, the sections along the top, the drops as white ticks, the hook in gold,
    where it sings in green, its live position when it is on a deck - and a line of plain words per stem
    saying where that part is usable ("bass 84 % · out 2:41-3:02 · loudest 1:04-1:36"). Right-click a row:
    the same commands as the lists (that stem in, TAKEOVER / BRIDGE with it, UP NEXT).
    (user 2026-09-13: "the songs list doesn't really provide enough information about what stems can be used")"""
    ROW_H = 54
    HEAD_H = 30
    SEC_H = 7
    WORDS_W = 250

    def __init__(self, tab, embedded=False):
        # embedded: a TAB of the Director's right column, swappable with the dials (user 2026-09-17); else the
        # old tool window
        if embedded:
            super().__init__(tab)
        else:
            super().__init__(tab, Qt.WindowType.Tool)
            self.setWindowTitle("Song parts")
            self.resize(960, self.HEAD_H + self.SEC_H + 4 * self.ROW_H + 30)
        self.tab = tab
        self.embedded = embedded
        self.setMinimumSize(520, self.HEAD_H + self.SEC_H + 4 * self.ROW_H + 30)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #121216; color: #e6e6ea;")
        self.track = None
        self.hover_t = None
        self._img_cache = {}
        self._words = {}
        self._words_for = None

    # -- data -------------------------------------------------------------------------------------------------
    def set_track(self, t):
        if t is self.track:
            return
        self.track = t
        self._img_cache.clear()
        self._words, self._words_for = {}, None
        self.hover_t = None
        if t is not None:
            self.setWindowTitle(f"Song parts - {t.title[:48]}")
            if getattr(t, "has_stems", False):
                self.tab.spectro.request(t)
        self.update()

    def _x(self, t_s, W):
        return 10 + float(t_s) / max(float(self.track.duration_s), 1e-6) * (W - 20 - self.WORDS_W)

    def _t(self, x, W):
        return max(0.0, min(float(self.track.duration_s), (x - 10) / max(1.0, W - 20 - self.WORDS_W) * float(self.track.duration_s)))

    def _compute_words(self, sp):
        """Per stem: share of the body where the part sounds, its longest silence, its loudest half minute."""
        import numpy as np
        t = self.track
        hop = float(sp.get("hop_s") or 0.1)
        secs = t.sections or []
        body = [s for s in secs if s.get("kind") not in ("intro", "outro")]
        b0 = float(body[0]["start_s"]) if body else 0.0
        b1 = float(body[-1]["end_s"]) if body else float(t.duration_s)
        out = {}
        for ln, spec in (sp.get("stems") or {}).items():
            if spec is None or spec.size == 0:
                continue
            e = spec.astype(np.float32).mean(axis=0)               # per frame, 0..255 dB scale
            top = float(np.percentile(e, 95)) or 1.0
            lvl = np.clip(e / top, 0.0, 1.0)
            per_s = int(round(1.0 / hop)) or 1
            n = len(lvl) // per_s
            sec_lvl = lvl[:n * per_s].reshape(n, per_s).mean(axis=1)
            i0, i1 = int(b0), min(n, int(b1))
            seg = sec_lvl[i0:i1] if i1 > i0 else sec_lvl
            on = seg >= 0.35
            share = float(on.mean()) if len(seg) else 0.0
            # longest silence inside the body
            best, cur, start, best_start = 0, 0, 0, 0
            for k, v in enumerate(on):
                if not v:
                    if cur == 0:
                        start = k
                    cur += 1
                    if cur > best:
                        best, best_start = cur, start
                else:
                    cur = 0
            quiet = (i0 + best_start, i0 + best_start + best) if best >= 4 else None
            # loudest half minute
            win = 30
            loud = None
            if len(sec_lvl) > win:
                cs = np.cumsum(np.insert(sec_lvl, 0, 0.0))
                means = (cs[win:] - cs[:-win]) / win
                k = int(np.argmax(means))
                loud = (k, k + win)
            words = [f"{int(round(100 * share))} % of the body"]
            if quiet:
                words.append(f"out {_mmss(quiet[0])}-{_mmss(quiet[1])}")
            if loud:
                words.append(f"loudest {_mmss(loud[0])}-{_mmss(loud[1])}")
            out[ln] = {"words": " · ".join(words), "share": share, "lvl": sec_lvl}
        # the pairings (docs/DJ_PAIRINGS.md): which vocals sit over this record, which basslines swap under it,
        # and where this song's own vocal and bassline sit on other records
        try:
            pr = self.tab._pairings_for(t.id)
            if pr:
                vo = [x["title"][:18] for x in (pr.get("vocal_over") or [])[:3]]
                bu = [x["title"][:18] for x in (pr.get("bass_under") or [])[:3]]
                if vo and "other" in out:
                    out["other"]["words"] += f" · vocals that sit over it: {', '.join(vo)}"
                if bu and "drums" in out:
                    out["drums"]["words"] += f" · basslines that swap under it: {', '.join(bu)}"
            inv = self.tab._pairings_inverse(t.id)
            if inv.get("vocal") and "vocals" in out:
                out["vocals"]["words"] += f" · sits over: {', '.join(inv['vocal'][:3])}"
            if inv.get("bass") and "bass" in out:
                out["bass"]["words"] += f" · swaps under: {', '.join(inv['bass'][:3])}"
        except Exception:
            pass
        return out

    def _song_pos(self):
        d = self.tab.director
        try:
            if d is not None and d.rc is not None and self.track is not None:
                deck = d.rc.deck_of(self.track.id)
                if deck is not None:
                    tel = d.rc._tel_deck(deck)
                    if tel.get("playing"):
                        return float(tel.get("time_s") or 0.0), d.rc.songs[deck].held(d.rc.lanes)
        except Exception:
            pass
        return None, []

    # -- paint ------------------------------------------------------------------------------------------------
    def paintEvent(self, ev):
        from PyQt6.QtGui import QPainter, QColor, QPen, QFont
        from PyQt6.QtCore import QRectF, QPointF
        from tools.dj.planner.timeline import SECTION_COLORS, LANE_COLORS, _spec_image
        p = QPainter(self)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(18, 18, 22))
        t = self.track
        if t is None:
            p.setPen(QColor(150, 150, 165))
            p.drawText(QRectF(12, 0, W - 24, H), Qt.AlignmentFlag.AlignVCenter, "highlight a song in SONGS or UP NEXT to see its parts")
            return
        f = QFont(self.font())
        f.setBold(True)
        f.setPointSizeF(10.5)
        p.setFont(f)
        p.setPen(QColor(230, 230, 236))
        hk = getattr(t, "hook", None)
        head = f"{t.title}   {t.artist or ''}   {(t.bpm or 0):.1f} bpm  {t.camelot or '?'}   {_mmss(t.duration_s)}"
        if hk:
            head += f"   hook {_mmss(hk['start_s'])}"
        if not getattr(t, "has_stems", False):
            head += "   (no stems rendered: the autoDJ can play it whole, the conductor cannot use its parts)"
        p.drawText(QRectF(12, 0, W - 24, self.HEAD_H), Qt.AlignmentFlag.AlignVCenter, head)
        # the sections band with drop ticks, the hook in gold, singing in green
        ys = self.HEAD_H
        secs = t.sections or []
        for i, s in enumerate(secs):
            col = QColor(SECTION_COLORS.get(s.get("kind"), QColor(90, 90, 100)))
            p.fillRect(QRectF(self._x(s["start_s"], W), ys, max(1.0, self._x(s["end_s"], W) - self._x(s["start_s"], W)), self.SEC_H), col)
        y0 = ys + self.SEC_H + 2
        sp = self.tab.spectro.get(t.id) if getattr(t, "has_stems", False) else None
        if sp is not None and self._words_for != t.id:
            try:
                self._words = self._compute_words(sp)
            except Exception:
                self._words = {}
            self._words_for = t.id
        pos, held = self._song_pos()
        small = QFont(self.font())
        small.setPointSizeF(8.5)
        words_x = W - self.WORDS_W - 4
        for i, ln in enumerate(("drums", "bass", "other", "vocals")):
            y = y0 + i * self.ROW_H
            rect = QRectF(10, y + 2, W - 20 - self.WORDS_W, self.ROW_H - 6)
            drawn = False
            if sp is not None and (sp.get("stems") or {}).get(ln) is not None:
                key = (ln, int(rect.width()), int(rect.height()))
                img = self._img_cache.get(key)
                if img is None:
                    img = _spec_image(sp["stems"][ln], LANE_COLORS[ln], 0.0, float(t.duration_s), int(rect.width()), int(rect.height()), sp["hop_s"])
                    self._img_cache[key] = img
                if img is not None:
                    p.drawImage(rect, img)
                    drawn = True
            if not drawn:
                p.fillRect(rect, QColor(28, 28, 34))
                p.setPen(QColor(150, 150, 165))
                p.setFont(small)
                msg = ("no stems" if not getattr(t, "has_stems", False) else
                       ("analysing the stems…" if (t.id in self.tab.spectro._busy or sp is None) and t.id not in self.tab.spectro.errors else
                        f"could not read the stems: {self.tab.spectro.errors.get(t.id, '')[:60]}"))
                p.drawText(rect, Qt.AlignmentFlag.AlignCenter, msg)
            # the lane name, lit when this song holds the lane right now
            p.setFont(QFont(self.font().family(), 8, QFont.Weight.Bold))
            p.setPen(QColor(255, 255, 255) if ln in held else QColor(200, 200, 210, 200))
            p.drawText(QRectF(14, y + 3, 90, 14), Qt.AlignmentFlag.AlignVCenter, {"other": "MELODY", "vocals": "VOCAL"}.get(ln, ln.upper()) + ("  ● in play" if ln in held else ""))
            # the words
            p.setFont(small)
            p.setPen(QColor(220, 220, 228))
            w = self._words.get(ln)
            p.drawText(QRectF(words_x, y + 2, self.WORDS_W, self.ROW_H - 6), Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
                       w["words"] if w else ("" if not drawn else "…"))
        # drops, hook and singing over the rows
        Hrows = y0 + 4 * self.ROW_H
        for i, s in enumerate(secs):
            if i > 0 and s.get("kind") == "groove" and secs[i - 1].get("kind") in ("build", "breakdown", "intro"):
                x = self._x(s["start_s"], W)
                p.setPen(QPen(QColor(255, 255, 255, 170), 1, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(x, ys), QPointF(x, Hrows))
        if hk:
            length = float(hk.get("end_s", 0) - hk.get("start_s", 0)) or 15.0
            p.setPen(QPen(QColor(240, 200, 90), 3))
            for st in (hk.get("starts") or [hk["start_s"]]):
                p.drawLine(QPointF(self._x(st, W), Hrows + 3), QPointF(self._x(st + length, W), Hrows + 3))
        vc = (getattr(t, "axes", None) or {}).get("vc") or []
        hop = float((getattr(t, "axes", None) or {}).get("vc_hop") or 8.0)
        p.setPen(QPen(QColor(120, 220, 140), 2))
        yv = y0 + 3 * self.ROW_H + self.ROW_H - 5
        for tt, v in vc:
            if v is not None and float(v) >= 0.03:
                p.drawLine(QPointF(self._x(tt, W), yv), QPointF(self._x(float(tt) + hop, W), yv))
        # the record's instrumental grooves (the pairing finder's: 12 bars and more with no singing) in blue under
        # the MELODY row - where another song's vocal can sit (docs/DJ_PAIRINGS.md)
        try:
            pr = self.tab._pairings_for(t.id)
        except Exception:
            pr = None
        if pr and pr.get("grooves"):
            p.setPen(QPen(QColor(90, 170, 240), 2))
            yg = y0 + 2 * self.ROW_H + self.ROW_H - 5
            for ga, gb in pr["grooves"]:
                p.drawLine(QPointF(self._x(float(ga), W), yg), QPointF(self._x(float(gb), W), yg))
        # the live position and the hover time
        if pos is not None:
            x = self._x(pos, W)
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(QPointF(x, ys), QPointF(x, Hrows))
        if self.hover_t is not None:
            x = self._x(self.hover_t, W)
            p.setPen(QPen(QColor(255, 255, 255, 120), 1))
            p.drawLine(QPointF(x, ys), QPointF(x, Hrows))
            p.setPen(QColor(230, 230, 236))
            p.setFont(small)
            sec = t.section_at(self.hover_t) or {}
            p.drawText(QRectF(min(x + 4, W - self.WORDS_W - 130), Hrows + 6, 130, 14), Qt.AlignmentFlag.AlignVCenter, f"{_mmss(self.hover_t)}  {sec.get('kind') or ''}")
        p.setPen(QColor(150, 150, 165))
        p.setFont(small)
        p.drawText(QRectF(12, H - 18, W - 24, 14), Qt.AlignmentFlag.AlignVCenter,
                   "sections along the top · dashed = a drop · gold = the hook · green = singing · blue = an instrumental groove (a vocal can sit here) · right-click a row: that part in, TAKEOVER / BRIDGE with it, UP NEXT")

    def mouseMoveEvent(self, ev):
        if self.track is not None:
            self.hover_t = self._t(ev.position().x(), self.width())
            self.update()

    def leaveEvent(self, ev):
        self.hover_t = None
        self.update()

    def contextMenuEvent(self, ev):
        if self.track is None or self.tab.director is None:
            return
        self.tab._song_actions(self.track.id, from_queue=False).exec(ev.globalPos())


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
        self.setMinimumHeight(48)
        self.setMaximumHeight(48)
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
    # every command the operator gives is logged as "ui: ..." BEFORE it acts, in the Director's event trail and
    # the night's log - twice a direction was "ignored" and the log held no trace of the click at all
    _LOGGED = ("_next", "_hold", "_moment", "_moment_cancel", "_bad", "_flag", "_dial", "_stem", "_program",
               "_queue", "_queue_selected", "_unqueue", "_queue_top", "_toggle")

    def _wrap_ui(self, fn, name):
        import inspect
        try:
            params = list(inspect.signature(fn).parameters.values())
            varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
            n_pos = sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        except (TypeError, ValueError):
            varargs, n_pos = True, 0

        def wrapped(*a, **k):
            # Qt hands `clicked` a `checked` bool: a handler that takes no arguments must not see it (a bound
            # method connected directly is trimmed by PyQt; a wrapper is not - START broke on it)
            if not varargs:
                a = a[:n_pos]
            try:
                d = self.director
                if d is not None:
                    words = " ".join(str(x)[:40] for x in a if not hasattr(x, "pos") and not isinstance(x, bool))
                    d._event(f"ui: {name.lstrip('_')} {words}".strip())
                else:
                    from lib.dj.session import log_row
                    log_row({"event": "ui", "text": f"{name.lstrip('_')} (no director)"})
            except Exception:
                pass
            return fn(*a, **k)
        return wrapped

    def __init__(self, planner):
        super().__init__()
        self.setObjectName("perform")
        for _nm in self._LOGGED:
            setattr(self, _nm, self._wrap_ui(getattr(self, _nm), _nm))
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
        # OUTPUTS (user 2026-09-20): where the floor plays and where the cue (the PAIRS preview) plays - two devices
        self.outputs_btn = QPushButton("OUTPUTS")
        self.outputs_btn.setToolTip("choose the MAIN output (the floor) and the CUE output (the headphone preview of PAIRS); "
                                    "kept per machine in logs/dj_outputs.json; the main output takes effect at the next START")
        self.outputs_btn.clicked.connect(self._outputs_dialog)
        top.addWidget(self.outputs_btn)
        self.songs_box = SetlistBox(self)
        self.songs_box.setMinimumWidth(220)
        self.songs_box.setToolTip("a playlist (the Set tab's saved lists) as the pool of songs; whole library lifts it")
        self.songs_box.currentIndexChanged.connect(self._pool_changed)
        top.addWidget(self.songs_box)
        self.state_lbl = QLabel("")
        self.state_lbl.setProperty("dim", "true")
        self.state_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)   # a long line never widens the tab
        top.addWidget(self.state_lbl, 1)
        # the picture's controls: zoom out / in, back to now, follow the playhead
        self.follow_btn = None
        for label, tip, fn in (("−", "zoom out (or Ctrl+wheel on the picture)", lambda: self.canvas.zoom(1 / 1.3)),
                               ("+", "zoom in (or Ctrl+wheel on the picture)", lambda: self.canvas.zoom(1.3)),
                               ("now", "back to the playhead, following again", self._back_to_now)):
            b = QPushButton(label)
            b.setProperty("kind", "small")
            b.setMinimumHeight(24)
            b.setMaximumWidth(44 if label != "now" else 52)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            top.addWidget(b)
        self.follow_btn = QPushButton("follow")
        self.follow_btn.setCheckable(True)
        self.follow_btn.setChecked(True)
        self.follow_btn.setProperty("kind", "small")
        self.follow_btn.setMinimumHeight(24)
        self.follow_btn.setMaximumWidth(64)
        self.follow_btn.setToolTip("keep the playhead in view (scrolling the picture by hand turns this off)")
        top.addWidget(self.follow_btn)
        root.addLayout(top)
        # -- the arc: the night's energy plan, what played against it, where we are; click / drag steer it ----
        self.arc_strip = ArcStrip(self)
        root.addWidget(self.arc_strip)

        # -- the picture ----------------------------------------------------------------------------------------------
        from lib.dj.timeline import Timeline, Spectro
        from tools.dj.planner.timeline import TimelineCanvas
        self._empty_tl = Timeline("director")
        self.spectro = Spectro(planner.music_dir)
        self.canvas = TimelineCanvas(self, lane_h=34, readonly=True)
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
        # word-wrapped labels drive the tab's MINIMUM height up as the window narrows (a 1600-wide window
        # demanded 977 px and the whole tab was squeezed unreadable): cap them, one line where one line does
        for w, h in ((self.now_lbl, 30), (self.now_sub, 20), (self.next_lbl, 24), (self.next_sub, 20), (self.intent_lbl, 40), (self.lanes_lbl, 20)):
            w.setMaximumHeight(h)
        for w in (self.now_lbl, self.now_sub, self.next_lbl, self.next_sub, self.lanes_lbl):
            w.setWordWrap(False)
        self.now_map = SongMap()
        self.now_map.setToolTip("the playing record's shape")
        self.next_map = SongMap()
        # THE PLAN (user 2026-09-17: "the auto system just doesn't display its plan"): an ordered list of what
        # comes, each row with the seconds from now - NOW, a MOMENT, the EVENT, the SEAM (how, why, the drop, the
        # rehearsal), the RECORD after it, UP NEXT; a row that just moved says why. It replaces the intent line
        # and the WHY panel (their words live on in tooltips and the log).
        self.plan_h = QLabel("THE PLAN")
        self.plan_h.setProperty("dim", "true")
        self.plan_h.setToolTip("what will happen next, in order, with the time from now: the record and its next section, an armed moment, the event, "
                               "the seam into the next record (how it comes in, why there, whether its drop lands, the rehearsal's verdict), "
                               "the record after it, your queue. A row changes only when the plan moves, and then says why.")
        self.plan_list = QListWidget()
        self.plan_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.plan_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.plan_list.setWordWrap(True)
        # THE SET's rows (THEN, a ready REQUEST) take commands: put it UP NEXT, not tonight, show its parts
        self.plan_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.plan_list.customContextMenuRequested.connect(self._plan_menu)
        # ...as the third tab of the right column (DIALS | PARTS | PLAN): on the left it squeezed the song lists
        self.plan_tab = QWidget()
        pl = QVBoxLayout(self.plan_tab)
        pl.setContentsMargins(6, 6, 6, 6)
        pl.addWidget(self.plan_h)
        pl.addWidget(self.plan_list, 1)
        # THE PAIRS (docs/DJ_MORE_THAN_A_MIXER.md 1): the mashup finder - the best pairs across the library, HEAR
        # one in the headphones (the cue output), DROP IT into the set
        self.pairs_tab = QWidget()
        pv = QVBoxLayout(self.pairs_tab)
        pv.setContentsMargins(6, 6, 6, 6)
        self.pairs_h = QLabel("PAIRS")
        self.pairs_h.setProperty("dim", "true")
        self.pairs_h.setToolTip("the best pairs across the whole library, from the pairing file: a vocal that sits over a record's groove, a bassline that "
                                "swaps under a record's drums. Double-click = HEAR eight bars in the headphones (the cue output); right-click = HEAR, "
                                "DROP IT (the record goes UP NEXT and the pair fires when it plays), show parts")
        pv.addWidget(self.pairs_h)
        prow = QHBoxLayout()
        self.pairs_clock = QPushButton("fits the clock")
        self.pairs_clock.setCheckable(True)
        self.pairs_clock.setChecked(True)
        self.pairs_clock.setToolTip("only pairs whose record is inside the tempo wall of the clock playing now")
        self.pairs_with = QPushButton("with the highlighted song")
        self.pairs_with.setCheckable(True)
        self.pairs_with.setToolTip("only pairs that involve the song highlighted in SONGS / UP NEXT")
        self.pairs_stop = QPushButton("■ stop")
        self.pairs_stop.setToolTip("stop the headphone preview")
        for b in (self.pairs_clock, self.pairs_with):
            b.toggled.connect(lambda _c: self._fill_pairs(force=True))
        self.pairs_stop.clicked.connect(self._pairs_stop)
        prow.addWidget(self.pairs_clock)
        prow.addWidget(self.pairs_with)
        prow.addStretch(1)
        prow.addWidget(self.pairs_stop)
        pv.addLayout(prow)
        self.pairs = QListWidget()
        self.pairs.setWordWrap(True)
        self.pairs.itemDoubleClicked.connect(lambda it: self._hear_pair(it.data(Qt.ItemDataRole.UserRole)) if it is not None else None)
        self.pairs.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pairs.customContextMenuRequested.connect(self._pairs_menu)
        pv.addWidget(self.pairs, 1)
        self.pairs_lbl = QLabel("")
        self.pairs_lbl.setProperty("dim", "true")
        self.pairs_lbl.setWordWrap(True)
        pv.addWidget(self.pairs_lbl)
        self._pairs_rows = []
        self._pairs_sig = None
        self._cue = None
        self._preview_cache = {}
        for w in (self.now_lbl, self.now_sub, self.now_map, self.next_lbl, self.next_sub, self.next_map, self.intent_lbl, self.lanes_lbl, self.why_lbl):
            left.addWidget(w)
        self.intent_lbl.setVisible(False)
        self.why_lbl.setVisible(False)
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
        # STEERING the list: what kind of song, relative to the reference (the last queued song, else the
        # playing one), and how to sort it
        frow = QHBoxLayout()
        frow.setSpacing(4)
        self.filter_groups = {}

        def fgroup(name, opts, tips):
            grp = QButtonGroup(self)
            grp.setExclusive(True)
            for opt, tip in zip(opts, tips):
                b = QPushButton(opt)
                b.setCheckable(True)
                b.setProperty("kind", "small")
                b.setMinimumHeight(20)
                b.setMaximumHeight(22)
                b.setToolTip(tip)
                b.setChecked(opt == opts[0])
                b.clicked.connect(lambda _c: (setattr(self, "_found_sig", None), self._fill_search()))
                grp.addButton(b)
                frow.addWidget(b)
            self.filter_groups[name] = grp
            frow.addSpacing(8)
        fgroup("voice", ("any", "inst", "vocal"), ("any song", "instrumentals only", "songs that sing"))
        fgroup("energy", ("any", "calmer", "same", "hotter"),
               ("any energy", "a step calmer than the reference (the last queued song, else the playing one): 0.04-0.30 below",
                "about the same energy as the reference", "a step hotter than the reference: 0.04-0.30 above (a chain of these is a build; "
                "when nothing hotter fits, the list falls back to the same energy and says so)"))
        fgroup("tempo", ("any", "slower", "same", "faster"), ("any tempo", "slower than the reference", "within 2 bpm of the reference", "faster than the reference"))
        self.f_hook = QPushButton("hook")
        self.f_hook.setCheckable(True)
        self.f_hook.setProperty("kind", "small")
        self.f_hook.setMinimumHeight(20)
        self.f_hook.setMaximumHeight(22)
        self.f_hook.setToolTip("only songs with a measured hook (a sung chorus that repeats)")
        self.f_hook.clicked.connect(lambda _c: (setattr(self, "_found_sig", None), self._fill_search()))
        frow.addWidget(self.f_hook)
        self.f_unplayed = QPushButton("unheard")
        self.f_unplayed.setCheckable(True)
        self.f_unplayed.setChecked(True)
        self.f_unplayed.setProperty("kind", "small")
        self.f_unplayed.setMinimumHeight(20)
        self.f_unplayed.setMaximumHeight(22)
        self.f_unplayed.setToolTip("hide what already played tonight")
        self.f_unplayed.clicked.connect(lambda _c: (setattr(self, "_found_sig", None), self._fill_search()))
        frow.addWidget(self.f_unplayed)
        frow.addSpacing(8)
        self.sort_box = QComboBox()
        for label, key in (("by fit", "fit"), ("calm → hot", "energy"), ("hot → calm", "energy_desc"), ("by bpm", "bpm"), ("by title", "title"), ("by hook", "hook")):
            self.sort_box.addItem(label, key)
        self.sort_box.setToolTip("how the list is ordered; ✓ always above ✗")
        self.sort_box.currentIndexChanged.connect(lambda _i: (setattr(self, "_found_sig", None), self._fill_search()))
        frow.addWidget(self.sort_box)
        frow.addStretch(1)
        # an explicit button beside the double-click: a list that refilled between two clicks swallowed the
        # double-click and the queue was "ignored" (flags 2026-09-13)
        self.queue_btn = QPushButton("▶ UP NEXT")
        self.queue_btn.setProperty("kind", "small")
        self.queue_btn.setMinimumHeight(20)
        self.queue_btn.setMaximumHeight(22)
        self.queue_btn.setToolTip("put the selected song UP NEXT (Enter or a double-click on the row does the same)")
        self.queue_btn.clicked.connect(self._queue_selected)
        frow.addWidget(self.queue_btn)
        # SONG PARTS: the floating viewer of the highlighted song's stems
        self.viewer = StemViewer(self, embedded=True)
        self.view_btn = QPushButton("PARTS")
        self.view_btn.setCheckable(True)
        self.view_btn.setProperty("kind", "small")
        self.view_btn.setMinimumHeight(20)
        self.view_btn.setMaximumHeight(22)
        self.view_btn.setToolTip("the PARTS tab (swaps with the dials): the highlighted song's four stems over time, its sections, hook and singing, "
                                 "and where each part is usable - follows whatever you highlight in SONGS or UP NEXT")
        self.view_btn.toggled.connect(self._toggle_viewer)
        frow.addWidget(self.view_btn)
        left.addLayout(frow)
        lists = QHBoxLayout()
        self.found = QListWidget()
        self.found.setMinimumHeight(80)
        # (the item can be None: Enter on an empty list, or a row the refill replaced between the key and the signal)
        self.found.itemDoubleClicked.connect(lambda it: self._queue(it.data(Qt.ItemDataRole.UserRole)) if it is not None else None)
        self.found.itemActivated.connect(lambda it: self._queue(it.data(Qt.ItemDataRole.UserRole)) if it is not None else None)   # Enter
        self.found.itemPressed.connect(lambda _it: setattr(self, "_found_click_t", time.time()))          # no refill right after a click
        self.found.currentItemChanged.connect(lambda cur, _prev: self._highlighted(cur))
        self.found.setToolTip("✓ mixable from here · ✗ would be rejected · double-click, Enter or ▶ UP NEXT to queue it · hover a row for the song's shape")
        lists.addWidget(self.found, 3)
        qcol = QVBoxLayout()
        # PLAYING: the songs on the decks right now - the record, the parts in play, the staged ones - so the
        # MIXES WITH list and the parts window have the playing song as their context (user 2026-09-17)
        self.decks_h = QLabel("PLAYING")
        self.decks_h.setProperty("dim", "true")
        self.decks_h.setToolTip("the songs on the decks now: ▶ the record, ▸ a part of it in play, ○ staged and waiting, ↘ leaving; "
                                "highlight one and MIXES WITH and the parts window follow it; right-click for the commands")
        qcol.addWidget(self.decks_h)
        self.decks = QListWidget()
        self.decks.setMaximumHeight(96)
        self.decks.currentItemChanged.connect(lambda cur, _prev: self._highlighted(cur))
        self.decks.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.decks.customContextMenuRequested.connect(self._decks_menu)
        self.decks.itemDoubleClicked.connect(lambda it: self._show_parts(it.data(Qt.ItemDataRole.UserRole)) if it is not None else None)
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self.decks, activated=lambda: self._eject(self.decks.currentItem().data(Qt.ItemDataRole.UserRole)) if self.decks.currentItem() else None,
                  context=Qt.ShortcutContext.WidgetShortcut)
        qcol.addWidget(self.decks)
        un = QLabel("UP NEXT")
        un.setProperty("dim", "true")
        un.setToolTip("your queue, in order; honoured when it can be mixed from where it is; double-click to remove")
        qcol.addWidget(un)
        self.queue = QListWidget()
        self.queue.itemDoubleClicked.connect(lambda it: self._unqueue(it.data(Qt.ItemDataRole.UserRole)) if it is not None else None)
        self.queue.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.queue.customContextMenuRequested.connect(self._queue_menu)
        self.queue.currentItemChanged.connect(lambda cur, _prev: self._highlighted(cur))
        self.queue.setToolTip("your queue, in order, with each song's state (○ ready on a deck, ▸ a part of it in play); "
                              "right-click a song for the same commands as in SONGS; double-click removes it")
        qcol.addWidget(self.queue, 1)
        # MIXES WITH: for the highlighted song, what goes with it and how (user 2026-09-17: "present more
        # information about what songs are good mixes together") - as the next record (key, tempo, energy and
        # whether its stems can arrive through a seam), whose vocal sits over it, whose bassline swaps under it,
        # and where its own parts sit; double-click a row to highlight that song in SONGS
        self.mix_h = QLabel("MIXES WITH")
        self.mix_h.setProperty("dim", "true")
        self.mix_h.setToolTip("for the highlighted song: what goes after it and how it would arrive, whose vocal sits over it, whose bassline "
                              "swaps under it, and where its own vocal and bassline sit - from the stems (docs/DJ_PAIRINGS.md)")
        qcol.addWidget(self.mix_h)
        self.mixes = QListWidget()
        self.mixes.setMinimumHeight(70)
        self.mixes.setToolTip("double-click a song here to find it in SONGS; right-click for the same commands")
        self.mixes.itemDoubleClicked.connect(lambda it: self._find_song(it.data(Qt.ItemDataRole.UserRole)) if it is not None else None)
        self.mixes.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.mixes.customContextMenuRequested.connect(self._mixes_menu)
        qcol.addWidget(self.mixes, 1)
        # the stem-by-stem grid (READY) is OFF the main surface (the composed moments replaced it - "a painful
        # mishmash"); a song's stem still comes in from the right-click menu on the list, loud and on the bar
        self.loud_btn = QPushButton("LOUD")
        self.loud_btn.setCheckable(True)
        self.loud_btn.setChecked(True)
        self.loud_btn.setVisible(False)
        self.phrase_btn = QPushButton("PHRASE")
        self.phrase_btn.setCheckable(True)
        self.phrase_btn.setVisible(False)
        self.ready_grid = QGridLayout()
        self._ready_rows = {}
        lists.addLayout(qcol, 2)
        self.found.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.found.customContextMenuRequested.connect(self._song_menu)
        left.addLayout(lists, 1)
        body.addLayout(left, 5)

        # -- right: the dials in two columns, the moments ---------------------------------------------------------------
        right = QVBoxLayout()
        right.setSpacing(8)
        from lib.dj.director import DIALS, DEFAULTS
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(3)
        self.dial_groups, self._dial_labels = {}, {}
        for i, name in enumerate(DIAL_ORDER):
            options = DIALS[name]
            title, labels, tip = DIAL_LABELS[name]
            # one line per dial - the heading to the LEFT of its buttons (two rows per dial put the right
            # column at 679 px minimum and a 1080 screen squeezed everything unreadable)
            cell = QWidget()
            row = QHBoxLayout(cell)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            head = QLabel(title)
            head.setProperty("dim", "true")
            head.setToolTip(tip)
            head.setFixedWidth(64)
            row.addWidget(head)
            group = QButtonGroup(self)
            group.setExclusive(True)
            for opt in options:
                b = QPushButton(labels[opt])
                b.setCheckable(True)
                b.setMinimumHeight(20)
                b.setMaximumHeight(22)
                b.setProperty("kind", "small")
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                b.setToolTip(tip)
                b.setChecked(opt == DEFAULTS[name])
                b.clicked.connect(lambda _c, n=name, o=opt: self._dial(n, o))
                group.addButton(b)
                row.addWidget(b)
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
        # OVER THE NEXT SONGS: one PROGRAM box (the 2×4 grid of programs left the surface), a song count, stop
        from lib.dj.director import PROGRAMS
        prow = QHBoxLayout()
        prow.setSpacing(4)
        ph = QLabel("PROGRAM")
        ph.setProperty("dim", "true")
        ph.setFixedWidth(64)
        ph.setToolTip("a program over the next few songs: the dials it names step at every song change (layered: every bed change), "
                      "the rest stay yours, and your dials come back when it ends; pick how many songs, then the program")
        prow.addWidget(ph)
        self.prog_box = QComboBox()
        self.prog_box.addItem("none", None)
        for key, (label, tip, _fn) in PROGRAMS.items():
            self.prog_box.addItem(label, key)
            self.prog_box.setItemData(self.prog_box.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
        self.prog_box.setMinimumHeight(22)
        self.prog_box.setToolTip(ph.toolTip())
        self.prog_box.activated.connect(lambda i: self._program(self.prog_box.itemData(i)))
        prow.addWidget(self.prog_box, 1)
        self.prog_n = QButtonGroup(self)
        self.prog_n.setExclusive(True)
        for n in (2, 3, 4):
            b = QPushButton(str(n))
            b.setCheckable(True)
            b.setChecked(n == 3)
            b.setProperty("kind", "small")
            b.setMinimumHeight(20)
            b.setMaximumHeight(22)
            b.setMaximumWidth(30)
            b.setToolTip(f"over {n} songs")
            self.prog_n.addButton(b, n)
            prow.addWidget(b)
        self.prog_cancel = QPushButton("stop")
        self.prog_cancel.setProperty("kind", "small")
        self.prog_cancel.setMinimumHeight(20)
        self.prog_cancel.setMaximumHeight(22)
        self.prog_cancel.setMaximumWidth(48)
        self.prog_cancel.setToolTip("end the program now: your dials come back")
        self.prog_cancel.clicked.connect(lambda: self.director and self.director.cancel_program())
        self.prog_cancel.setVisible(False)
        prow.addWidget(self.prog_cancel)
        right.addLayout(prow)
        self.prog_btns = {}
        self.prog_lbl = QLabel("")
        self.prog_lbl.setProperty("dim", "true")
        self.prog_lbl.setWordWrap(True)
        self.prog_lbl.setMaximumHeight(34)
        right.addWidget(self.prog_lbl)
        # THE MOMENTS: NEXT / STAY, then the five composed moments (press = arm: the picture draws the choreography
        # in gold and the button counts down to the one; press again = go now; right-click = cancel), GOOD / BAD
        mom = QGridLayout()
        mom.setSpacing(6)
        self.buttons = {}
        for (label, fn, kind, c) in (("NEXT", self._next, "go", 0), ("HOLD", self._hold, None, 1)):
            b = QPushButton(label)
            b.setMinimumHeight(28)
            if kind:
                b.setProperty("kind", kind)
            b.clicked.connect(fn)
            mom.addWidget(b, 0, c * 5, 1, 5)
            self.buttons[label] = b
        self.moment_btns = {}
        self._moment_tips = {}
        from lib.dj.remix import MOMENTS
        for i, (key, label) in enumerate(MOMENT_ROW):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setMinimumHeight(30)
            b.setProperty("kind", "hot")
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self._moment_tips[key] = (f"{label}: {MOMENTS[key][2]}\n\npress = arm (lands on the drop or hook ahead, else the next phrase; the picture shows it in gold, "
                                      "the button counts down) · press again = go now · right-click = cancel")
            b.setToolTip(self._moment_tips[key])
            b.clicked.connect(lambda _c, k=key: self._moment(k))
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _p, k=key: self._moment_cancel(k))
            mom.addWidget(b, 1, i * 2, 1, 2)
            self.moment_btns[key] = b
        for (label, fn, kind, c, span) in (("GOOD", lambda: self.director and self.director.rate(True), "good", 0, 3),
                                           ("BAD", self._bad, "bad", 3, 3),
                                           ("GOOD FLAG", lambda: self._flag(True), None, 6, 2),
                                           ("BAD FLAG", lambda: self._flag(False), None, 8, 2)):
            b = QPushButton(label)
            b.setMinimumHeight(26)
            if kind:
                b.setProperty("kind", kind)
            b.clicked.connect(fn)
            mom.addWidget(b, 2, c, 1, span)
            self.buttons[label] = b
        self.buttons["GOOD FLAG"].setToolTip("✓ this worked - say what: kept for LATER as a case to protect when a rule changes; the whole room is "
                                             "written to logs/dj_flags.jsonl and marked in the story row; nothing is learned. Review: python tools/dj/dj_flags.py --good")
        self.buttons["BAD FLAG"].setToolTip("⚑ this is not working - say what kind, for LATER: the whole room is written to logs/dj_flags.jsonl "
                                            "(songs, positions, lanes, gains, the plan, the moment, the last moves) and marked in the story row; "
                                            "nothing is learned and nothing moves - unlike BAD, which punishes a move and a song's tags. "
                                            "Review: python tools/dj/dj_flags.py --bad")
        right.addLayout(mom)
        self.moment_lbl = QLabel("")
        self.moment_lbl.setProperty("dim", "true")
        self.moment_lbl.setWordWrap(True)
        self.moment_lbl.setMaximumHeight(34)
        self.moment_lbl.setToolTip("the armed moment: whose, where it lands, and what refused the last one")
        right.addWidget(self.moment_lbl)
        self.verdict_lbl = QLabel("")
        self.verdict_lbl.setProperty("dim", "true")
        self.verdict_lbl.setWordWrap(True)
        self.verdict_lbl.setMaximumHeight(54)
        self.verdict_lbl.setToolTip("what your last GOOD / BAD rated, what the DJ learned from it, and what it did about it")
        right.addWidget(self.verdict_lbl)
        self.err_lbl = QLabel("")
        self.err_lbl.setProperty("dim", "true")
        self.err_lbl.setWordWrap(True)
        right.addWidget(self.err_lbl)
        right.addStretch(1)
        rw = QWidget()
        rw.setLayout(right)
        # the right column is TWO TABS: the dials and the moments, and the Song parts (the highlighted song's
        # four stems over time with the words) - swappable, the same width (user 2026-09-17)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(rw, "DIALS")
        self.right_tabs.addTab(self.viewer, "PARTS")
        self.right_tabs.addTab(self.plan_tab, "PLAN")
        self.right_tabs.addTab(self.pairs_tab, "PAIRS")
        self.right_tabs.setToolTip("DIALS: how it plays; PARTS: the highlighted song's stems, sections, hook, singing and grooves; "
                                   "PLAN: what will happen next, in order, with the time from now")
        self.right_tabs.currentChanged.connect(self._right_tab_changed)
        self.right_tabs.setMinimumWidth(540)
        self.right_tabs.setMaximumWidth(760)
        body.addWidget(self.right_tabs, 4)
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

    def marks(self):
        """The story for the picture: what happened where, and what is planned (ghost)."""
        return self.director.all_marks() if self.director is not None else []

    def lane_gains(self):
        return self.director.lane_gains() if self.director is not None else {}

    def on_user_scroll(self):
        if self.follow_btn is not None:
            self.follow_btn.setChecked(False)

    def _back_to_now(self):
        if self.follow_btn is not None:
            self.follow_btn.setChecked(True)
        if self.director is not None:
            vb = self.canvas.visible_bars()
            self.canvas.first_bar = max(0.0, self.director.bar() - vb * 0.33)
            self.canvas._img_cache.clear()
            self.canvas.update()

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

    def _pairings(self):
        """The pairing lists (docs/DJ_PAIRINGS.md) from <music>/.stems/pairings.json, re-read when the file changes."""
        import json
        import os
        path = os.path.join(self.planner.music_dir, ".stems", "pairings.json")
        try:
            mt = os.path.getmtime(path)
        except OSError:
            return None
        if getattr(self, "_pair_mtime", None) != mt:
            try:
                self._pair_data = json.load(open(path, encoding="utf-8"))
                self._pair_mtime = mt
                self._pair_inv = None
            except Exception:  # noqa: BLE001
                self._pair_data = None
        return getattr(self, "_pair_data", None)

    def _pairings_for(self, tid):
        p = self._pairings()
        return ((p.get("records") or {}).get(str(int(tid)))) if p else None

    def _pairings_inverse(self, tid):
        """Where this song's own vocal and bassline sit: {"vocal": [record titles], "bass": [...]}."""
        p = self._pairings()
        if not p:
            return {}
        if getattr(self, "_pair_inv", None) is None:
            inv = {}
            for rid, rec in (p.get("records") or {}).items():
                for x in rec.get("vocal_over") or []:
                    inv.setdefault(int(x["id"]), {"vocal": [], "bass": []})["vocal"].append((x["score"], rec["title"][:18]))
                for x in rec.get("bass_under") or []:
                    inv.setdefault(int(x["id"]), {"vocal": [], "bass": []})["bass"].append((x["score"], rec["title"][:18]))
            for v in inv.values():
                v["vocal"] = [t for _, t in sorted(v["vocal"], key=lambda st: -st[0])]
                v["bass"] = [t for _, t in sorted(v["bass"], key=lambda st: -st[0])]
            self._pair_inv = inv
        return self._pair_inv.get(int(tid), {})

    def _mix_mode(self, tid, fits):
        """(lane key, words): how this song would PRIMARILY mix from here - its vocal over the record playing, its
        bassline under it, or, as the next record, drums first / a bass swap / the cut (the pairing file's
        seam-ability); None when nothing is known. The SONGS list colours each row by it (user 2026-09-20)."""
        w = fits.get(int(tid)) if fits else None
        if w:
            return ("vocals", "its vocal over the record") if w.startswith("its vocal") else ("bass", "its bassline under the record")
        rec = self._pairings_for(tid) or {}
        sm = rec.get("seam") or {}
        if not sm:
            return None
        if sm.get("drums"):
            return "drums", "as the next record: drums first"
        if sm.get("bass"):
            return "bass", "as the next record: a bass swap"
        return "cut", "as the next record: the cut"

    def _fits_now(self):
        """{song id: words} for the songs whose parts sit on the RECORD playing now."""
        out = {}
        try:
            rc = self.director.rc if self.director is not None else None
            if rc is None or getattr(rc, "record_deck", None) not in rc.songs:
                return out
            rec = self._pairings_for(rc.songs[rc.record_deck].track.id)
            if not rec:
                return out
            for x in rec.get("vocal_over") or []:
                out[int(x["id"])] = f"its vocal sits over this record ({x['fit']})"
            for x in rec.get("bass_under") or []:
                out.setdefault(int(x["id"]), f"its bassline swaps under this record ({x['fit']})")
        except Exception:
            pass
        return out

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
        """The moments say what they do, in the mode we are in (the detail in the tooltip). One song at a time
        only the first moment works: the autoDJ's DROP (a build into the next song's drop)."""
        one = mode != "layered"
        texts = {
            "NEXT": ("NEXT SONG", "move on to the next song at the next phrase") if one
                    else ("NEXT MOVE", "the conductor's next lane move on the next bar"),
            "HOLD": ("STAY", "one more phrase of this song before moving on" if one else "freeze the lanes for a phrase"),
            "GOOD": ("👍 GOOD", "that was good - the DJ does more of it"),
            "BAD": ("👎 BAD", "that was bad - the DJ does less of it; on a moment it asks which of five things hurt and moves that one number"),
        }
        for k, (t, tip) in texts.items():
            if self.buttons[k].text() != t:
                self.buttons[k].setText(t)
            self.buttons[k].setToolTip(tip)
        for key, label in MOMENT_ROW:
            b = self.moment_btns[key]
            if one:
                b.setEnabled(key == "slam")
                if key == "slam" and b.text() != "DROP":
                    b.setText("DROP")
            else:
                b.setEnabled(True)
                if key == "slam" and b.text() == "DROP":
                    b.setText(label)
        self._moments_mode = mode

    def _moment(self, key):
        """Press = arm; the armed one pressed again = go now."""
        if self.director is None:
            self.err_lbl.setText("press START first")
            self.moment_btns[key].setChecked(False)
            return
        ok, msg = self.director.arm_moment(key)
        self.moment_lbl.setText(msg if ok else f"{key.upper()} refused: {msg}")
        if not ok:
            self.moment_btns[key].setChecked(False)

    def _moment_cancel(self, key):
        if self.director is None:
            return
        ok, msg = self.director.moment_cancel()
        self.moment_lbl.setText(msg)

    def _flag(self, good):
        """GOOD FLAG / BAD FLAG: what worked, or what is not working, for later - a kind from the menu, or a
        kind with a note. No learning either way."""
        if self.director is None:
            self.err_lbl.setText("press START first")
            return
        from PyQt6.QtWidgets import QMenu, QInputDialog
        from lib.dj.director import FLAG_BAD, FLAG_GOOD
        kinds, other, head_text, prompt = ((FLAG_GOOD, "good_other", "✓ this worked (a case to protect)", "what worked? (a sentence; the room's state is captured with it)")
                                           if good else (FLAG_BAD, "other", "⚑ this is not working (kept for later, nothing learned)",
                                                         "what is not working? (a sentence; the room's state is captured with it)"))
        m = QMenu(self)
        head = m.addAction(head_text)
        head.setEnabled(False)
        for key, label in kinds.items():
            if key == other:
                continue
            a = m.addAction(label)
            a.triggered.connect(lambda _c, k=key: self._flag_done(self.director.flag(k)))
        m.addSeparator()
        note = m.addAction("with a note…")

        def _with_note():
            text, ok = QInputDialog.getText(self, "GOOD FLAG" if good else "BAD FLAG", prompt)
            if ok:
                self._flag_done(self.director.flag(other, text))
        note.triggered.connect(lambda _c: _with_note())
        b = self.buttons["GOOD FLAG" if good else "BAD FLAG"]
        m.exec(b.mapToGlobal(b.rect().topLeft()))

    def _flag_done(self, msg):
        self.verdict_lbl.setText("⚑ " + msg)
        self.moment_lbl.setText("")

    def _bad(self):
        """BAD: on a moment, the ear's five words first (one word moves one number of that moment); on anything
        else, plain BAD."""
        if self.director is None:
            return
        tgt = getattr(self, "_rate_target", None)
        if not tgt or not tgt.get("moment"):
            self.director.rate(False)
            return
        from PyQt6.QtWidgets import QMenu
        from lib.dj.remix import MOMENT_WORDS
        m = QMenu(self)
        head = m.addAction(f"BAD on {tgt['text'][:44]} - what hurt?")
        head.setEnabled(False)
        for word, text in MOMENT_WORDS.items():
            a = m.addAction(text)
            a.triggered.connect(lambda _c, w=word: self.director and self.director.rate(False, word=w))
        m.addSeparator()
        plain = m.addAction("just bad (less of this moment)")
        plain.triggered.connect(lambda _c: self.director and self.director.rate(False))
        b = self.buttons["BAD"]
        m.exec(b.mapToGlobal(b.rect().topLeft()))

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
        ranked = [tag for tag, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if n >= min_n]
        limit = 36 if getattr(self, "_mood_more", False) else 18        # three rows fit a 1080 screen; "more…" opens six
        top = ranked[:limit]
        hidden = len(ranked) - len(top)
        sig = (pool, len(tracks), tuple(top), hidden)
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
        # the last cell toggles more / fewer chips (not a mood, never lit)
        old = getattr(self, "_mood_toggle", None)
        if old is not None:
            self.mood_row.removeWidget(old)
            old.setParent(None)
            old.deleteLater()
            self._mood_toggle = None
        if hidden > 0 or getattr(self, "_mood_more", False):
            t = QPushButton(f"+{hidden} more…" if hidden > 0 else "fewer")
            t.setProperty("kind", "small")
            t.setMinimumHeight(20)
            t.setMaximumHeight(22)
            t.setToolTip("show more of the library's tags (or fewer)")

            def _toggle():
                self._mood_more = not getattr(self, "_mood_more", False)
                self._fill_tags(force=True)
            t.clicked.connect(_toggle)
            self.mood_row.addWidget(t, len(top) // 6, len(top) % 6)
            self._mood_toggle = t
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
        # the engine is STARTED LAST: loading the library and building the brain hold the GIL for seconds,
        # and an engine already running underran on that ("the render thread cannot keep up", user's log
        # 2026-09-12); nothing is heard before the first engine is ready anyway
        self.engine = AudioEngine()
        try:
            from lib.dj.cue import main_device_name
            self.engine.device_name = main_device_name()  # the MAIN output chosen under OUTPUTS ("" = the system default)
        except Exception:
            pass
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
        self.engine.start()
        self.start_btn.setText("■  STOP")
        self.start_btn.setProperty("kind", "hot")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self._found_sig = None
        self._fill_tags()                             # the Director's library is the one in play now
        self._fill_search()
        self._midi_open()
        self._timer.start()
        self._dropbox_start()

    # -- THE FLOOR: a request becomes playable during the night (docs/DJ_THE_SET.md rule 5) --------------------------
    # Two ways in: "make it playable" on a library song without stems, and a file dropped into the music folder
    # while the night runs. Both go analysis (a new file only) → stems (the planner's one-slot GPU queue, the
    # fast model) → pairings (one CPU pass, one at a time) → the Director adopts the fresh library entry. Each
    # step is a REQUEST row in THE PLAN.
    def _make_playable(self, tid):
        t = self.track(tid)
        d = self.director
        if t is None or d is None:
            return
        if getattr(t, "has_stems", False):
            self.moment_lbl.setText(f"{t.title[:30]} already has its stems")
            return
        key = f"stems:{t.id}"
        d.note_request(key, t.title, "waiting for the stem renderer", track_id=t.id)
        ok = self.planner.render_stems(t, model="htdemucs",
                                       on_status=lambda s, k=key, tt=t: self._floor_status(k, tt.title, s),
                                       on_done=lambda tid2, ok2, k=key, tt=t: self._floor_stems_done(k, tt.title, tid2, ok2))
        if not ok:
            d.note_request(key, t.title, "stems", error="the stem renderer is busy with it already, or unavailable (torch + demucs)")
        else:
            self.moment_lbl.setText(f"{t.title[:30]}: making it playable (stems, then pairings)")

    def _floor_status(self, key, title, words):
        d = self.director
        if d is None:
            return
        w = str(words or "")
        stage = "separating" if "separating" in w or "loading" in w else ("queued for stems" if "queued" in w else w[:40])
        pct = None
        try:
            import re
            m = re.search(r"(\d{1,3})\s*%", w)
            if m:
                pct = float(m.group(1))
        except Exception:
            pct = None
        d.note_request(key, title, stage, pct=pct)

    def _floor_stems_done(self, key, title, tid, ok):
        d = self.director
        if d is None:
            return
        if not ok:
            d.note_request(key, title, "stems", error="the stems did not render (see the console)")
            return
        d.note_request(key, title, "pairings (its envelope and lists)", track_id=tid)
        if not hasattr(self, "_floor_jobs"):
            self._floor_jobs, self._floor_ready = [], []
        self._floor_jobs.append(("pairings", key, title, tid))
        self._floor_next()

    def _floor_next(self):
        """One CPU job at a time: a new file's analysis, or the pairings pass after stems."""
        if getattr(self, "_floor_proc", None) is not None or not self._floor_jobs:
            return
        from PyQt6.QtCore import QProcess
        import sys as _sys
        job = self._floor_jobs.pop(0)
        kind, key, title = job[0], job[1], job[2]
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._floor_out = []
        proc.readyReadStandardOutput.connect(lambda p=proc: self._floor_out.extend(bytes(p.readAllStandardOutput()).decode("utf-8", "replace").splitlines()))
        if kind == "pairings":
            args = [os.path.join(repo, "tools", "dj", "dj_pairings.py"), "--music", self.planner.music_dir, "--all", "--workers", "1"]
        else:   # ("add", key, title, path)
            args = [os.path.join(repo, "tools", "dj", "dj_scan.py"), "--dir", self.planner.music_dir, "--add", job[3]]
        proc.finished.connect(lambda code, _st, j=job: self._floor_proc_done(j, code))
        self._floor_proc = proc
        proc.start(_sys.executable, args)
        if self.director is not None:
            self.director.note_request(key, title, "pairings running" if kind == "pairings" else "analyzing the file", track_id=(job[3] if kind == "pairings" else None))

    def _floor_proc_done(self, job, code):
        self._floor_proc = None
        d = self.director
        kind, key, title = job[0], job[1], job[2]
        out = list(getattr(self, "_floor_out", []) or [])
        if kind == "add":
            tid = None
            for ln in out:
                if ln.startswith("ADDED "):
                    try:
                        tid = int(ln.split()[1])
                    except Exception:
                        tid = None
            if code != 0 or tid is None:
                if d is not None:
                    d.note_request(key, title, "analysis", error=(next((ln for ln in reversed(out) if ln.strip()), f"exit {code}"))[:90])
            else:
                if d is not None:
                    d.note_request(key, title, "waiting for the stem renderer", track_id=tid)
                from types import SimpleNamespace
                stub = SimpleNamespace(id=tid, title=title)
                ok = self.planner.render_stems(stub, model="htdemucs",
                                               on_status=lambda s, k=key, tt=title: self._floor_status(k, tt, s),
                                               on_done=lambda tid2, ok2, k=key, tt=title: self._floor_stems_done(k, tt, tid2, ok2))
                if not ok and d is not None:
                    d.note_request(key, title, "stems", error="the stem renderer is busy or unavailable")
        else:
            tid = job[3]
            if code != 0 and d is not None:
                d.note_request(key, title, "pairings", error=f"the pairings pass failed (exit {code}); the song plays without its lists")
            self._floor_ready.append((key, title, tid))
            self._floor_reload()
        self._floor_next()

    def _floor_reload(self):
        """A fresh hydration of the library on a worker (its own DB handle), adopted by the Director on this
        thread: the new song joins SONGS, the stems it gained make it layerable."""
        if getattr(self, "_floor_reloading", False):
            return
        self._floor_reloading = True
        import threading
        music = self.planner.music_dir
        box = {}

        def work():
            try:
                from lib.dj.db import LibraryDB
                from lib.dj import brain as B
                box["lib"] = B.load_library(LibraryDB(music))
            except Exception as e:  # noqa: BLE001
                box["err"] = f"{type(e).__name__}: {e}"

        def poll():
            if "lib" not in box and "err" not in box:
                QTimer.singleShot(250, poll)
                return
            self._floor_reloading = False
            d = self.director
            ready, self._floor_ready = list(self._floor_ready), []
            if d is None:
                return
            if "err" in box:
                for key, title, _tid in ready:
                    d.note_request(key, title, "library", error=box["err"][:90])
                return
            try:
                d.adopt_library(box["lib"])
            except Exception as e:  # noqa: BLE001
                self.err_lbl.setText(f"adopt: {type(e).__name__}: {e}")
            for key, title, tid in ready:
                d.note_request(key, title, "ready", ready=True, track_id=tid)
            self._lib = None
            self._found_sig = None
            self._fill_search()
        threading.Thread(target=work, daemon=True, name="director-floor-reload").start()
        QTimer.singleShot(250, poll)

    def _dropbox_start(self):
        """Files that appear under the music folder while the night runs are found within twenty seconds."""
        self._floor_jobs = getattr(self, "_floor_jobs", [])
        self._floor_ready = getattr(self, "_floor_ready", [])
        self._known_paths = None
        if getattr(self, "_drop_timer", None) is None:
            self._drop_timer = QTimer(self)
            self._drop_timer.setInterval(20000)
            self._drop_timer.timeout.connect(self._dropbox_poll)
        self._drop_timer.start()

    def _dropbox_stop(self):
        t = getattr(self, "_drop_timer", None)
        if t is not None:
            t.stop()

    def _dropbox_poll(self):
        if self.director is None or getattr(self, "_drop_busy", False):
            return
        import threading
        music = self.planner.music_dir
        box = {}
        self._drop_busy = True

        def work():
            try:
                from lib.dj.scan import list_audio_files
                box["files"] = [os.path.normcase(os.path.abspath(p)) for p in list_audio_files(music)]
            except Exception as e:  # noqa: BLE001
                box["err"] = str(e)

        def poll():
            if "files" not in box and "err" not in box:
                QTimer.singleShot(200, poll)
                return
            self._drop_busy = False
            files = box.get("files")
            if files is None:
                return
            if self._known_paths is None:
                # the first sweep after START only learns what is there
                self._known_paths = set(files)
                return
            new = [p for p in files if p not in self._known_paths]
            self._known_paths.update(files)
            for p in new[:6]:
                title = os.path.splitext(os.path.basename(p))[0]
                key = f"file:{p}"
                if self.director is not None:
                    self.director.note_request(key, title, "new file: waiting for analysis")
                self._floor_jobs.append(("add", key, title, p))
            if new:
                self._floor_next()
        threading.Thread(target=work, daemon=True, name="director-dropbox").start()
        QTimer.singleShot(200, poll)

    def _outputs_dialog(self):
        """Two combo boxes over the machine's playback devices: MAIN (the floor) and CUE (the preview). Saved to
        logs/dj_outputs.json; the cue player is rebuilt at once, the main output is used at the next START."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel as _QL
        from lib.dj.cue import playback_devices, load_outputs, save_outputs
        devs = [n for n, _i in playback_devices()]
        cur = load_outputs()
        dlg = QDialog(self)
        dlg.setWindowTitle("OUTPUTS")
        form = QFormLayout(dlg)
        main_box, cue_box = QComboBox(), QComboBox()
        main_box.addItem("system default", "")
        cue_box.addItem("none (no headphone preview)", "")
        for n in devs:
            main_box.addItem(n, n)
            cue_box.addItem(n, n)

        def select(box, name):
            if not name:
                box.setCurrentIndex(0)
                return
            for i in range(box.count()):
                if name.lower() in str(box.itemData(i) or "").lower():
                    box.setCurrentIndex(i)
                    return
            box.addItem(f"{name} (not found now)", name)
            box.setCurrentIndex(box.count() - 1)
        select(main_box, cur.get("main", ""))
        select(cue_box, cur.get("cue", ""))
        form.addRow("MAIN output (the floor)", main_box)
        form.addRow("CUE output (headphones: the PAIRS preview)", cue_box)
        note = _QL("the main output takes effect at the next START; the cue output at once.\n"
                   "Pick two different devices, or the preview would play on the floor.")
        note.setWordWrap(True)
        form.addRow(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        main, cue = str(main_box.currentData() or ""), str(cue_box.currentData() or "")
        if main and cue and main == cue:
            self.err_lbl.setText("OUTPUTS: the cue output is the main output - the preview would play on the floor; cue set to none")
            cue = ""
        save_outputs(main, cue)
        self._cue = None                              # rebuilt with the new device on the next HEAR
        words = f"OUTPUTS: main = {main or 'system default'} · cue = {cue or 'none'}"
        if self.director is not None and self.engine is not None and (getattr(self.engine, "device_name", "") or "") != main:
            words += "  (the main output changes at the next START)"
        self.moment_lbl.setText(words)
        try:
            self._fill_pairs(force=True)
        except Exception:
            pass

    # -- THE PAIRS: the mashup finder and the headphone preview (docs/DJ_MORE_THAN_A_MIXER.md 1) ---------------------
    def _fill_pairs(self, force=False):
        d = self.director
        if d is None:
            if self.pairs.count():
                self.pairs.clear()
            self.pairs_h.setText("PAIRS   (press START: the pairs are ranked against the clock)")
            return
        with_tid = None
        if self.pairs_with.isChecked():
            it = self.found.currentItem() or self.queue.currentItem()
            with_tid = it.data(Qt.ItemDataRole.UserRole) if it is not None else None
        try:
            rows = d.pairs(n=80, fit_clock=self.pairs_clock.isChecked(), with_tid=with_tid)
        except Exception as e:  # noqa: BLE001
            self.pairs_lbl.setText(f"pairs: {type(e).__name__}: {e}")
            return
        sig = tuple((p["kind"], p["r"].id, p["s"].id, p["played"]) for p in rows)
        if sig == self._pairs_sig and not force:
            return
        self._pairs_sig = sig
        self._pairs_rows = rows
        self.pairs.clear()
        for i, p in enumerate(rows):
            it = QListWidgetItem(("· " if not p["played"] else "(played) ") + p["words"])
            it.setData(Qt.ItemDataRole.UserRole, i)
            it.setToolTip(f"{p['words']}\nscore {p['score']:.2f}\ndouble-click: HEAR eight bars in the headphones · right-click: DROP IT")
            col = LANE_COLORS.get("vocals" if p["kind"] == "vocal" else "bass")
            if p["played"]:
                it.setForeground(Qt.GlobalColor.gray)
            elif col is not None:
                it.setForeground(col)
            self.pairs.addItem(it)
        self.pairs_h.setText(f"PAIRS  {len(rows)}" + ("  ·  inside the wall of the clock" if self.pairs_clock.isChecked() else "  ·  the whole library")
                             + (f"  ·  with {d.title(int(with_tid))[:22]}" if with_tid is not None else ""))

    def _pairs_menu(self, pos):
        it = self.pairs.itemAt(pos)
        if it is None or self.director is None:
            return
        i = it.data(Qt.ItemDataRole.UserRole)
        if i is None or i >= len(self._pairs_rows):
            return
        p = self._pairs_rows[i]
        from PyQt6.QtWidgets import QMenu
        m = QMenu(self)
        a = m.addAction("HEAR: eight bars in the headphones")
        a.triggered.connect(lambda _c, k=i: self._hear_pair(k))
        a = m.addAction(f"DROP IT: {p['r'].title[:22]} UP NEXT, the pair when it plays")
        a.triggered.connect(lambda _c, k=i: self._drop_pair(k))
        m.addSeparator()
        for t in (p["r"], p["s"]):
            a = m.addAction(f"show the parts of {t.title[:26]}")
            a.triggered.connect(lambda _c, tt=t.id: self._show_parts(tt))
        m.exec(self.pairs.mapToGlobal(pos))

    def _cue_player(self):
        if self._cue is None:
            from lib.dj.cue import CuePlayer
            self._cue = CuePlayer()
        return self._cue

    def _hear_pair(self, i):
        """Render the pair on a worker (the decode in the helper process, RubberBand offline) and play it on the cue
        output; never on the floor while the Director runs."""
        if i is None or i >= len(self._pairs_rows) or self.director is None:
            return
        p = self._pairs_rows[i]
        cue = self._cue_player()
        if not cue.available:
            self.pairs_lbl.setText(cue.why)
            return
        if getattr(self, "_hear_busy", False):
            self.pairs_lbl.setText("a preview is rendering - a moment")
            return
        self._hear_busy = True
        self.pairs_lbl.setText(f"rendering: {p['words'][:70]} …")
        import threading
        from lib.dj.stemload import decode_song
        from lib.dj.preview import render_pair
        music, db = self.planner.music_dir, self._db()
        cache = self._preview_cache
        box = {}

        def loader(tid):
            if tid in cache:
                return cache[tid]
            t = self.track(tid)
            _s, stems, _l = decode_song(db.abs(t.path), music, tid, with_levels=False)
            cache[tid] = stems
            if len(cache) > 6:
                cache.pop(next(iter(cache)))
            return stems

        def work():
            try:
                box["buf"] = render_pair(music, p["kind"], p["r"], p["s"], p["r_start"], p["s_start"], bars=8, shift=p["shift"], cache=cache, loader=loader)
            except Exception as e:  # noqa: BLE001
                box["err"] = f"{type(e).__name__}: {e}"

        def poll():
            if "buf" not in box and "err" not in box:
                QTimer.singleShot(200, poll)
                return
            self._hear_busy = False
            if "err" in box:
                self.pairs_lbl.setText(f"the preview could not be rendered: {box['err'][:90]}")
                return
            from lib.dj.submix import RATE
            ok, words = cue.play(box["buf"], RATE)
            self.pairs_lbl.setText((f"hearing {words}: " if ok else "") + (p["words"][:70] if ok else words))
        threading.Thread(target=work, daemon=True, name="director-preview").start()
        QTimer.singleShot(200, poll)

    def _drop_pair(self, i):
        if i is None or i >= len(self._pairs_rows) or self.director is None:
            return
        msg = self.director.drop_pair(self._pairs_rows[i])
        self.pairs_lbl.setText(msg)
        self.moment_lbl.setText(msg[:90])
        self._render_queue()

    def _pairs_stop(self):
        if self._cue is not None:
            self._cue.stop()
        self.pairs_lbl.setText("preview stopped")

    def _skip_in_set(self, tid):
        d = self.director
        if d is None:
            return
        d.skip_in_set(tid)
        self.moment_lbl.setText(f"not tonight: {d.title(int(tid))[:30]}")
        self._found_sig = None
        self._fill_search()

    def _plan_menu(self, pos):
        """THE PLAN's own commands: a THEN row (the chain) - put it UP NEXT, not tonight, its parts; a ready
        REQUEST - put it UP NEXT."""
        it = self.plan_list.itemAt(pos)
        d = self.director
        if it is None or d is None:
            return
        data = it.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, tuple) or data[1] is None:
            return
        kind, tid = data
        from PyQt6.QtWidgets import QMenu
        m = QMenu(self)
        title = d.title(int(tid))[:30]
        if kind == "THEN":
            a = m.addAction(f"put {title} UP NEXT")
            a.triggered.connect(lambda _c, t=tid: self._queue(t))
            a = m.addAction(f"not tonight: {title} leaves the set")
            a.triggered.connect(lambda _c, t=tid: self._skip_in_set(t))
        elif kind == "IDEA":
            a = m.addAction(f"do it: {title}")
            a.triggered.connect(lambda _c: self.moment_lbl.setText(d.idea_take()))
            a = m.addAction("skip this idea")
            a.triggered.connect(lambda _c: self.moment_lbl.setText(d.idea_skip()))
        elif kind == "REQUEST":
            a = m.addAction(f"put {title} UP NEXT")
            a.triggered.connect(lambda _c, t=tid: self._queue(t))
        else:
            return
        v = m.addAction("show its parts (the PARTS tab)")
        v.triggered.connect(lambda _c, t=tid: self._show_parts(t))
        m.exec(self.plan_list.mapToGlobal(pos))

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
        if os.environ.get("DJ_NO_MIDI"):
            return                                    # headless smokes never take the controller from a live planner
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
                    d.arm_moment("slam")               # REC = SLAM (one song: DROP); again = go now
                elif name == "cycle":
                    d.arm_moment("break")              # CYCLE = BREAK routine; again = go now
                elif name == "track_prev":
                    d.rate(False)
                elif name == "track_next":
                    d.rate(True)
                elif name == "marker_set":
                    self._flag_done(d.flag("other", "flagged from the controller (SET)"))
                elif name == "marker_prev":
                    d.arc_jump(max(0.0, d.arc_progress() - 0.1))
                elif name == "marker_next":
                    d.arc_jump(min(0.999, d.arc_progress() + 0.1))

    def close(self):
        self._timer.stop()
        self._dropbox_stop()
        if getattr(self, "_cue", None) is not None:
            self._cue.stop()
        try:
            self.viewer.hide()
        except Exception:
            pass
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

    def _stem(self, tid, lane, held):
        if self.director is None:
            self.err_lbl.setText("press START first")
            return
        if held:
            msg = self.director.stem_out(lane)
        else:
            msg = self.director.stem_in(tid, lane, loud=self.loud_btn.isChecked(), when="phrase" if self.phrase_btn.isChecked() else "bar")
        self.err_lbl.setText(msg or "")

    def _song_menu(self, pos):
        it = self.found.itemAt(pos)
        if it is None or self.director is None:
            return
        self._song_actions(it.data(Qt.ItemDataRole.UserRole), from_queue=False).exec(self.found.mapToGlobal(pos))

    def _queue_menu(self, pos):
        it = self.queue.itemAt(pos)
        if it is None or self.director is None:
            return
        self._song_actions(it.data(Qt.ItemDataRole.UserRole), from_queue=True).exec(self.queue.mapToGlobal(pos))

    def _song_actions(self, tid, from_queue):
        """The same commands for a song wherever it is listed (SONGS or UP NEXT): its stems in, TAKEOVER /
        BRIDGE with it, and the queue itself."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QFont
        m = QMenu(self)
        title = self.director.title(tid)[:28]
        # THE MENU IS TRIMMED TO WHAT MAKES SENSE FOR THIS PAIR (user 2026-09-17: "too many options, most of
        # them useless for a given match"): a line saying how the pair fits; the vocal only when the song
        # sings; the bassline (the TAKEOVER's manual form); TAKEOVER; BRIDGE only while the record sings;
        # drums and melody under "more" (the drum swap is off by design, a melody alone is a texture); a song
        # outside the tempo wall gets the reason and its parts window, nothing else. What FITS is ★ and bold.
        good, why = self._fit_words(tid)
        bold = QFont(self.font())
        bold.setBold(True)
        pair = self._pair_line(tid)
        if pair:
            h = m.addAction(pair)
            h.setEnabled(False)
        layered = self.director.mode == "layered"
        walled = bool(why and why.startswith("outside the tempo wall"))
        halftime = bool(why and (why.startswith("half-time part") or why.startswith("double-time part")))
        if halftime:
            h = m.addAction(why)
            h.setEnabled(False)
        if layered and not walled:
            rec = self._pairings_for(tid) or {}
            sings = rec.get("sings")
            if sings is None:
                t = self.track(tid)
                sings = bool(getattr(t, "hook", None)) or bool(rec.get("sung"))
            when = 'at the phrase' if self.phrase_btn.isChecked() else 'now'
            loud = " · loud" if self.loud_btn.isChecked() else ""
            shown = []
            if sings or good.get("vocals"):
                shown.append(("vocals", "vocals"))
            shown.append(("bass", "bassline"))
            for lane, word in shown:
                star = good.get(lane)
                a = m.addAction(("★ " if star else "") + f"{word} of {title} in {when}{loud}" + (f"   ({star})" if star else ""))
                if star:
                    a.setFont(bold)
                a.triggered.connect(lambda _c, t=tid, ln=lane: self._stem(t, ln, False))
            more = m.addMenu("more parts…")
            for lane, word in (("drums", "drums"), ("other", "melody")) + (() if sings or good.get("vocals") else (("vocals", "vocals (it does not sing by its stem)"),)):
                a = more.addAction(f"{word} of {title} in {when}{loud}")
                a.triggered.connect(lambda _c, t=tid, ln=lane: self._stem(t, ln, False))
            m.addSeparator()
            star = good.get("bass")
            a = m.addAction(("★ " if star else "") + "TAKEOVER with this (its bassline on the one, its drums at the phrase)" + (f"   ({star})" if star else ""))
            if star:
                a.setFont(bold)
            a.triggered.connect(lambda _c, t=tid: self.moment_lbl.setText(self.director.moment_with(t, "takeover", when="phrase" if self.phrase_btn.isChecked() else "bar")))
            rc = self.director.rc
            try:
                rec_sings = rc is not None and rc.lanes.get("vocals") == rc.record_deck and rc.record_deck in rc.songs \
                    and rc._singing(rc.songs[rc.record_deck], rc._song_time_at(rc.record_deck, rc._next_bar_clock())) is True
            except Exception:
                rec_sings = False
            if rec_sings:
                a = m.addAction("BRIDGE with this (the record fades under its own vocal, this one's bass + drums slam)")
                a.triggered.connect(lambda _c, t=tid: self.moment_lbl.setText(self.director.moment_with(t, "bridge", when="phrase" if self.phrase_btn.isChecked() else "bar")))
        elif walled:
            h = m.addAction(why)
            h.setEnabled(False)
        elif layered and not getattr(self.track(tid), "has_stems", False):
            # THE FLOOR (docs/DJ_THE_SET.md rule 5): a song without stems can be made playable tonight
            h = m.addAction("no stems yet: it cannot be layered")
            h.setEnabled(False)
            a = m.addAction("▤ make it playable (its stems, about a minute on the GPU; then its pairings)")
            a.triggered.connect(lambda _c, t=tid: self._make_playable(t))
        m.addSeparator()
        v = m.addAction("show its parts (the PARTS tab)")
        v.triggered.connect(lambda _c, t=tid: self._show_parts(t))
        if walled:
            # THE SET rule 4: a far song is a route, not a refusal - when a chain of records reaches it
            hops = None
            try:
                hops = self.director.route_to(tid)
            except Exception as e:  # noqa: BLE001
                self.err_lbl.setText(f"route: {type(e).__name__}: {e}")
            if hops:
                a = m.addAction(f"put it UP NEXT via {len(hops)} record{'s' if len(hops) > 1 else ''}: " + ", ".join(f"{h['title'][:18]} ({h['bpm']:.0f})" for h in hops))
                a.setFont(bold)
                a.triggered.connect(lambda _c, t=tid: self._queue(t))
            elif hops == []:
                a = m.addAction("put it UP NEXT (it follows the last queued record)")
                a.triggered.connect(lambda _c, t=tid: self._queue(t))
            else:
                h = m.addAction("no chain of records reaches it tonight")
                h.setEnabled(False)
            return m
        if from_queue:
            up = m.addAction("▲ play this next (to the top)")
            up.triggered.connect(lambda _c, t=tid: self._queue_top(t))
            rm = m.addAction("✕ remove from UP NEXT")
            rm.triggered.connect(lambda _c, t=tid: self._unqueue(t))
        else:
            q = m.addAction("put it UP NEXT")
            q.triggered.connect(lambda _c, t=tid: self._queue(t))
        return m

    def _pair_line(self, tid):
        """One line on how this song and the record playing fit: keys and their fit, the tempo, the energy."""
        try:
            from lib.dj.brain import camelot_compat
            rc = self.director.rc if self.director is not None else None
            if rc is None or rc.master is None or rc.record_deck not in rc.songs:
                return None
            rec, t = rc.songs[rc.record_deck].track, self.track(tid)
            if t is None or t.id == rec.id:
                return None
            parts = []
            if rec.camelot and t.camelot:
                parts.append(f"{rec.camelot} → {t.camelot} fit {camelot_compat(rec.camelot, t.camelot):.2f}")
            if t.bpm and rc.master_bpm:
                parts.append(f"tempo {100 * (rc.master_bpm / float(t.bpm) - 1):+.0f}%")
            try:
                e0, e1 = float(rec.energy_proxy()), float(t.energy_proxy())
                parts.append("same energy" if abs(e1 - e0) < 0.12 else ("hotter" if e1 > e0 else "calmer"))
            except Exception:
                pass
            return f"with {rec.title[:22]}:  " + " · ".join(parts) if parts else None
        except Exception:
            return None

    def _fit_words(self, tid):
        """({lane: why it is a good option now}, a reason it is not): the pairings of the record playing (its
        vocal sits over it / its bassline swaps under it, with the fit), the tempo wall, a key clash."""
        good, why = {}, None
        d = self.director
        try:
            rc = d.rc if d is not None else None
            if rc is None or rc.master is None:
                return good, None
            rec = rc.songs.get(rc.record_deck)
            t = self.track(tid)
            if rec is None or t is None:
                return good, None
            pr = self._pairings_for(rec.track.id) or {}
            for x in pr.get("vocal_over") or []:
                if int(x["id"]) == int(tid):
                    good["vocals"] = f"its vocal sits over {rec.track.title[:18]}, fit {x['fit']}"
            for x in pr.get("bass_under") or []:
                if int(x["id"]) == int(tid):
                    good["bass"] = f"its bassline swaps under these drums, fit {x['fit']}"
            if t.bpm and rc.master_bpm:
                from lib.dj.remix import RATE_MIN, RATE_MAX
                rate = rc.master_bpm / float(t.bpm)
                if not (RATE_MIN <= rate <= RATE_MAX):
                    _r, mult = rc._part_rate(t)
                    if mult in (2.0, 0.5):
                        # a HALF-TIME / DOUBLE-TIME part (docs/DJ_MORE_THAN_A_MIXER.md 3): its parts come in, it is never the record
                        return {}, f"{'half-time' if mult == 2.0 else 'double-time'} part: {t.bpm:.0f} bpm under a {rc.master_bpm:.0f} clock - its parts only, never the record"
                    return {}, f"outside the tempo wall: {t.bpm:.0f} bpm against {rc.master_bpm:.0f}"
            if not good and t.camelot and rec.track.camelot:
                from lib.dj.brain import camelot_compat
                if camelot_compat(rec.track.camelot, t.camelot) < 0.55:
                    why = f"key {t.camelot} against {rec.track.camelot}: a clash, said and done if you ask"
        except Exception:
            pass
        return good, why

    def _highlighted(self, item):
        """The Song parts window and the MIXES WITH list follow the highlighted row of SONGS or UP NEXT."""
        if item is None:
            return
        tid = item.data(Qt.ItemDataRole.UserRole)
        if tid is None:
            return
        t = self.track(tid)
        if t is not None:
            self.viewer.set_track(t)
        try:
            self._fill_mixes(tid)
        except Exception as e:  # noqa: BLE001
            self.err_lbl.setText(f"mixes: {type(e).__name__}: {e}")

    def _mixes_menu(self, pos):
        it = self.mixes.itemAt(pos)
        if it is None or self.director is None or it.data(Qt.ItemDataRole.UserRole) is None:
            return
        self._song_actions(it.data(Qt.ItemDataRole.UserRole), from_queue=False).exec(self.mixes.mapToGlobal(pos))

    def _find_song(self, tid):
        if tid is None:
            return
        for i in range(self.found.count()):
            if self.found.item(i).data(Qt.ItemDataRole.UserRole) == tid:
                self.found.setCurrentRow(i)
                return
        t = self.track(tid)
        if t is not None:
            self.viewer.set_track(t)
            self._fill_mixes(tid)

    def _next_partners(self, tid, n=5):
        """Songs that follow this one well AS THE NEXT RECORD: key fit, tempo inside the wall, the energy rank
        close, and (from the pairing file, version 3) whether their stems can arrive through a seam - the
        drums first or the bass swap - and carry on after the landmark."""
        from lib.dj.brain import camelot_compat
        from lib.dj.remix import RATE_MIN, RATE_MAX
        t = self.track(tid)
        if t is None or not t.bpm:
            return []
        p = self._pairings() or {}
        recs = p.get("records") or {}
        try:
            e0 = float(t.energy_proxy())
        except Exception:
            e0 = 0.5
        played = set()
        try:
            if self.director is not None:
                played = set(self.director._played_ids())
        except Exception:
            played = set()
        out = []
        for c in self._library():
            if c.id == t.id or not getattr(c, "has_stems", False) or not c.bpm or c.id in played:
                continue
            rate = float(t.bpm) / float(c.bpm)
            if not (RATE_MIN <= rate <= RATE_MAX):
                continue
            kf = camelot_compat(t.camelot, c.camelot) if (t.camelot and c.camelot) else 0.6
            if kf < 0.55:
                continue
            try:
                e1 = float(c.energy_proxy())
            except Exception:
                e1 = 0.5
            rec = recs.get(str(c.id)) or {}
            sm = rec.get("seam") or {}
            ways = [w for w, ok in (("drums first", sm.get("drums")), ("bass swap", sm.get("bass"))) if ok]
            arrive = 1.0 if ways else (0.85 if sm.get("after", True) else 0.6)
            score = kf * (1.0 - abs(1.0 - rate) / 0.1 * 0.3) * (1.0 - 0.3 * abs(e1 - e0)) * arrive
            words = [f"key {c.camelot or '?'} {kf:.2f}", f"tempo {100 * (rate - 1):+.0f}%",
                     ("same energy" if abs(e1 - e0) < 0.12 else ("hotter" if e1 > e0 else "calmer"))]
            words.append(("arrives " + " or ".join(ways)) if ways else ("the cut" if sm else ""))
            mode = "drums" if "drums first" in ways else ("bass" if "bass swap" in ways else "cut")
            out.append((score, c.id, c.title, " · ".join(w for w in words if w), mode))
        out.sort(key=lambda x: -x[0])
        return out[:n]

    def _fill_mixes(self, tid):
        """The MIXES WITH list for one song."""
        from PyQt6.QtGui import QFont
        t = self.track(tid)
        self.mixes.clear()
        if t is None:
            self.mix_h.setText("MIXES WITH")
            return
        self.mix_h.setText(f"MIXES WITH  {t.title[:30]}")
        bold = QFont(self.font())
        bold.setBold(True)

        def head(text):
            it = QListWidgetItem(text)
            it.setFont(bold)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.mixes.addItem(it)

        def row(cid, text, tip=None, mode=None):
            it = QListWidgetItem("    " + text)
            it.setData(Qt.ItemDataRole.UserRole, cid)
            if tip:
                it.setToolTip(tip)
            col = LANE_COLORS.get(mode) if mode else None   # THE COLOUR IS THE MIX, as in SONGS (user 2026-09-20)
            if col is not None:
                it.setForeground(col)
            self.mixes.addItem(it)
        pr = self._pairings_for(tid) or {}
        nxt = self._next_partners(tid)
        head("after it, as the next record" + ("" if nxt else ": nothing inside the wall"))
        for score, cid, title, words, mode in nxt:
            row(cid, f"{title[:28]}   {words}", f"{title}\n{words}\nscore {score:.2f}", mode=mode)
        vo = (pr.get("vocal_over") or [])[:4]
        head("a vocal that sits over its grooves" + ("" if vo else ": none found" + ("" if pr.get("grooves") else " (no instrumental groove of 12 bars)")))
        for x in vo:
            row(int(x["id"]), f"{x['title'][:28]}   fit {x['fit']}" + (f", shift {x['shift']:+d}" if x.get("shift") else "") + (" · at its hook" if x.get("hook") else ""),
                f"{x['title']}\nits sung stretch {_mmss(x['sung'][0])}-{_mmss(x['sung'][1])} over this song's groove {_mmss(x['groove'][0])}-{_mmss(x['groove'][1])}", mode="vocals")
        bu = (pr.get("bass_under") or [])[:4]
        head("a bassline that swaps under its drums (then it is the record)" + ("" if bu else ": none found"))
        for x in bu:
            row(int(x["id"]), f"{x['title'][:28]}   fit {x['fit']}" + (f", shift {x['shift']:+d}" if x.get("shift") else "") + f" · from {_mmss(x['landmark'])}",
                f"{x['title']}\nits bass and melody from its drop at {_mmss(x['landmark'])}, {x.get('runway_s', 0):.0f} s after", mode="bass")
        inv = self._pairings_inverse(tid)
        if inv.get("vocal") or inv.get("bass"):
            head("its own parts sit on")
            if inv.get("vocal"):
                row(None, "vocal over: " + ", ".join(inv["vocal"][:4]))
            if inv.get("bass"):
                row(None, "bassline under: " + ", ".join(inv["bass"][:4]))

    def _toggle_viewer(self, on):
        """PARTS (the button) = the PARTS tab of the right column; off = back to the dials."""
        self.right_tabs.setCurrentWidget(self.viewer if on else self.right_tabs.widget(0))
        if on:
            cur = self.decks.currentItem() or self.found.currentItem() or self.queue.currentItem()
            if cur is not None:
                self._highlighted(cur)

    def _right_tab_changed(self, idx):
        on = self.right_tabs.widget(idx) is self.viewer
        if self.view_btn.isChecked() != on:
            self.view_btn.blockSignals(True)
            self.view_btn.setChecked(on)
            self.view_btn.blockSignals(False)
        if self.right_tabs.widget(idx) is self.pairs_tab:
            try:
                self._fill_pairs(force=True)
            except Exception as e:  # noqa: BLE001
                self.pairs_lbl.setText(f"pairs: {type(e).__name__}: {e}")

    def _show_parts(self, tid):
        """A song's parts, in the PARTS tab."""
        t = self.track(tid) if tid is not None else None
        if t is not None:
            self.viewer.set_track(t)
        self.view_btn.setChecked(True)
        self.right_tabs.setCurrentWidget(self.viewer)

    def _queue_top(self, tid):
        if self.director is None:
            return
        self.director.up_next = [int(tid)] + [t for t in self.director.up_next if t != int(tid)]
        self.director._event(f"up next, first: {self.director.title(int(tid))}")
        self._render_queue()

    def _fill_ready(self, rows):
        """The READY rows follow the decks: one line per song, four stem buttons, lit where it holds the lane."""
        # rebuild the widgets only when a song appears, leaves or changes which lanes it HOLDS; the guard's
        # bar-by-bar opinion (why / loud) changes in place - rebuilding on it made the row bounce every bar
        sig = tuple((r["id"], r["bed"], r["staged"], tuple((ln, v["held"]) for ln, v in r["lanes"].items())) for r in rows)
        if sig == getattr(self, "_ready_sig", None):
            for r in rows[:4]:
                ws = self._ready_rows.get(r["id"])
                if not ws:
                    continue
                for j, (lane, word) in enumerate((("bass", "bass"), ("drums", "drums"), ("other", "melody"), ("vocals", "vocal"))):
                    v, b = r["lanes"][lane], ws[1 + j]
                    text = word + (" ♪" if v["loud"] else "")
                    if b.text() != text:
                        b.setText(text)
                    on = r["staged"] and (v["held"] or not v["why"])
                    if b.isEnabled() != on:
                        b.setEnabled(on)
                    tip = ((f"{r['title'][:28]} holds the {lane}" + (" (loud)" if v["loud"] else "") + " - press = back to the bed") if v["held"]
                           else (f"refused: {v['why']}" if v["why"] else f"the {lane} of {r['title'][:28]} in"))
                    if b.toolTip() != tip:
                        b.setToolTip(tip)
            return
        self._ready_sig = sig
        for w in list(self._ready_rows.values()):
            for x in w:
                self.ready_grid.removeWidget(x)
                x.setParent(None)
                x.deleteLater()
        self._ready_rows = {}
        for i, r in enumerate(rows[:4]):
            name = QLabel(("● " if r["bed"] else ("○ " if r["staged"] else "… ")) + r["title"][:20])
            name.setProperty("dim", "true")
            name.setToolTip(("the bed (drums + bass)" if r["bed"] else ("staged and ready" if r["staged"] else "decoding")) + f": {r['title']}"
                            + (f"\nhook in {r['hook_in_bars']:.0f} bars" if r.get("hook_in_bars") is not None else "")
                            + (f"\ndrop in {r['drop_in_bars']:.0f} bars" if r.get("drop_in_bars") is not None else ""))
            name.setMaximumWidth(150)
            self.ready_grid.addWidget(name, i, 0)
            ws = [name]
            for j, (lane, word) in enumerate((("bass", "bass"), ("drums", "drums"), ("other", "melody"), ("vocals", "vocal"))):
                v = r["lanes"][lane]
                b = QPushButton(word + (" ♪" if v["loud"] else ""))
                b.setCheckable(True)
                b.setChecked(v["held"])
                b.setProperty("kind", "small")
                b.setMinimumHeight(20)
                b.setMaximumHeight(22)
                b.setEnabled(r["staged"] and (v["held"] or not v["why"]))
                b.setToolTip((f"{r['title'][:28]} holds the {lane}" + (" (loud)" if v["loud"] else "") + " - press = back to the bed") if v["held"]
                             else (f"refused: {v['why']}" if v["why"] else f"the {lane} of {r['title'][:28]} in"))
                b.clicked.connect(lambda _c, t=r["id"], ln=lane, h=v["held"]: self._stem(t, ln, h))
                self.ready_grid.addWidget(b, i, 1 + j)
                ws.append(b)
            self._ready_rows[r["id"]] = ws

    def _program(self, key):
        if self.director is None:
            self.err_lbl.setText("press START first")
            self.prog_box.setCurrentIndex(0)
            return
        if key is None:
            self.director.cancel_program()            # "none" = stop
            return
        self.director.start_program(key, n=self.prog_n.checkedId() or 3)

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
        filters = {}
        try:
            for name, grp in self.filter_groups.items():
                b = grp.checkedButton()
                if b is not None and b.text() != "any":
                    filters[name] = b.text()
            if self.f_hook.isChecked():
                filters["hook"] = True
            if self.f_unplayed.isChecked():
                filters["unplayed"] = True
        except Exception:
            filters = {}
        sort = self.sort_box.currentData() if hasattr(self, "sort_box") else "fit"
        try:
            if self.director is not None:
                rows = self.director.rank(q, n=80, filters=filters, sort=sort)
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
        fits = self._fits_now()
        sig = tuple((r["id"], r["ok"], r["why"], fits.get(r["id"])) for r in rows)
        if sig == getattr(self, "_found_sig", None):
            return
        self._found_sig = sig
        scroll = self.found.verticalScrollBar().value()
        self.found.clear()
        for r in rows:
            mark = "" if r["ok"] is None else ("✓  " if r["ok"] else "✗  ")
            if r["id"] in fits:
                mark = "★ " + mark                    # a PAIRING with the record playing: its vocal sits over it, or its bassline swaps under it
                r["why"] = (fits[r["id"]] + ("; " + r["why"] if r.get("why") else ""))
            if "energy" in r:
                # two lines: who it is and what it is like; then why it fits or not
                e = r.get("energy") or 0.0
                # words, not glyphs: a block or music-note glyph pulls in a fallback font that breaks the
                # item's multi-line layout (the third line vanished)
                eword = "calm" if e < 0.35 else ("warm" if e < 0.55 else ("hot" if e < 0.75 else "peak"))
                # ONE line, the few facts that decide a pick (user: "flooding me with details"): tempo and
                # key, the energy word, sings or not, two moods, the hook; everything else in the tooltip
                short = [f"{(r['bpm'] or 0):.0f} {r['camelot'] or ''}".strip(), eword, "sings" if r.get("sings") else "inst"]
                # HOW IT FITS, in the line (user 2026-09-17): the key fit and the tempo step from the reference
                if r["ok"] and r.get("why"):
                    fitw = [w for w in r["why"].split(", ") if w.startswith(("tempo ", "key "))]
                    if fitw:
                        short.append(" ".join(w.replace("key ", "").replace("tempo ", "") for w in fitw))
                if r.get("tags"):
                    short.append(", ".join(r["tags"][:2]))
                if r.get("hook_s") is not None:
                    short.append(f"hook {int(r['hook_s']) // 60}:{int(r['hook_s']) % 60:02d}")
                line = f"{mark}{r['title'][:34]}  ·  {r['artist'][:18]}   " + " · ".join(short)
                if r["ok"] is False and r.get("why"):
                    reason = r["why"].split(", ")[0] if r["why"].startswith(("played", "already", "no stems")) else next(
                        (w for w in r["why"].split(", ") if "out of reach" in w or "clash" in w or "loose" in w or "outside" in w), r["why"][:40])
                    line += f"   —  {reason}"
                elif r["ok"] and "(via " in (r.get("why") or ""):
                    line += "   —  " + r["why"][r["why"].index("(via ") + 1:].split(")")[0]     # THE SET rule 4: reachable through other records
                it = QListWidgetItem(line)
                full = [f"{(r['bpm'] or 0):.0f} bpm {r['camelot'] or ''}".strip(), f"energy {e:.2f} ({eword})", "sings" if r.get("sings") else "instrumental"]
                if r.get("valence"):
                    full.append(r["valence"])
                if r.get("genre"):
                    full.append(r["genre"])
                if r.get("year"):
                    full.append(str(r["year"]))
                if r.get("tags"):
                    full.append(", ".join(r["tags"][:5]))
                if r.get("duration_s"):
                    full.append(_mmss(r["duration_s"]))
                tip = [f"{r['title']} — {r['artist']}", "  ·  ".join(full)]
                if r.get("hook_s") is not None:
                    tip.append(f"hook at {int(r['hook_s']) // 60}:{int(r['hook_s']) % 60:02d}")
                if r.get("shape"):
                    tip.append(f"shape: {r['shape']}")
                if r.get("why"):
                    tip.append(("fits: " if r["ok"] else "would be rejected: ") + r["why"])
                if r.get("last_played_min") is not None:
                    tip.append(f"played {r['last_played_min']} min ago")
                it.setToolTip("\n".join(tip))
            else:
                it = QListWidgetItem(f"{mark}{r['title'][:38]}  ·  {r['artist'][:22]}   {(r['bpm'] or 0):.0f} bpm {r['camelot'] or ''}"
                                     + (f"   {r['why']}" if r["why"] else ""))
            it.setData(Qt.ItemDataRole.UserRole, r["id"])
            if r["ok"] is False:
                it.setForeground(Qt.GlobalColor.gray)
            elif self.director is not None and self.director.rc is not None:
                # THE COLOUR IS THE MIX: the lane's colour from the picture - a vocal over the record, a bassline
                # under it, drums first as the next record; the cut and the unknown stay plain
                mode = self._mix_mode(r["id"], fits)
                if mode is not None:
                    key, words = mode
                    col = LANE_COLORS.get(key)
                    if col is not None:
                        it.setForeground(col)
                    it.setToolTip((it.toolTip() + "\n" if it.toolTip() else "") + f"mixes as: {words}")
            self.found.addItem(it)
            if r["id"] == cur:
                self.found.setCurrentItem(it)
        self.found.verticalScrollBar().setValue(scroll)   # the list stays where the eye left it
        ref = getattr(self.director, "rank_ref", None) if self.director is not None else None
        note = getattr(self.director, "rank_note", None) if self.director is not None else None
        legend = ""
        if self.director is not None and self.director.rc is not None:
            legend = "   ·   " + "  ".join(f"<span style='color:{LANE_COLORS[k].name()}'>●</span> {w}" for k, w in (("vocals", "vocal over"), ("bass", "bassline under / bass swap"), ("drums", "drums first")))
        self.found_h.setText(f"SONGS  {len(rows)}" + ((f"   ✓ fits after {ref[:22]} (last in UP NEXT)  ✗ would be rejected" if ref
                                                     else "   ✓ fits  ✗ would be rejected") if self.director is not None else "")
                             + legend + (f"   ·   {note}" if note else ""))

    def _queue_selected(self):
        it = self.found.currentItem()
        if it is None:
            self.err_lbl.setText("select a song in the list first")
            return
        self._queue(it.data(Qt.ItemDataRole.UserRole))

    def _queue(self, tid):
        if tid is None:
            return
        if self.director is not None:
            ok, msg = self.director.queue(tid)
            if ok:
                via = [t for t in self.director.up_next if t in self.director.via_ids]
                self.moment_lbl.setText(f"UP NEXT: {self.director.title(int(tid))[:40]}" + (f"  (after {len(self.director.up_next) - 1} more)" if len(self.director.up_next) > 1 else "")
                                        + (f"  ·  via {', '.join(self.director.title(t)[:16] for t in via)}" if via else ""))
            else:
                self.moment_lbl.setText(msg)
        else:
            self.err_lbl.setText("press START first")
        self._render_queue()
        self._found_sig = None
        self._fill_search()                           # the list now ranks after the last queued song

    def _unqueue(self, tid):
        if self.director is not None:
            self.director.unqueue(tid)
        self._render_queue()
        self._found_sig = None
        self._fill_search()

    def _render_plan(self, rows):
        """THE PLAN list: rebuilt only when its rows change (their keys and words); the clocks tick in place."""
        import sys
        from PyQt6.QtGui import QFont

        def clock(r):
            return ("now" if (r.get("t") is not None and r["t"] < 1.0) else (_mmss(r["t"]) if r.get("t") is not None else "")).rjust(5)

        def line(r):
            head = f"{clock(r)}  {r['kind']:<8} {r['text']}"
            return head + (f"\n{'':7}{r['sub']}" if r.get("sub") else "")
        sig = tuple((str(r.get("key")), r["kind"], r["text"], r.get("sub", "")) for r in rows)
        if sig != getattr(self, "_plan_sig", None):
            self._plan_sig = sig
            self.plan_list.clear()
            mono = QFont("Consolas" if sys.platform.startswith("win") else "Menlo", 9)
            for r in rows:
                it = QListWidgetItem(line(r))
                it.setFont(mono)
                it.setToolTip(f"{r['kind']}: {r['text']}\n{r.get('sub', '')}" + (f"\nbar {r['bar']}" if r.get("bar") is not None else "")
                              + ("\nright-click: put it UP NEXT · not tonight · show its parts" if r["kind"] == "THEN" else "")
                              + ("\nright-click: do it · skip" if r["kind"] == "IDEA" else "")
                              + ("\nright-click: put it UP NEXT" if r["kind"] == "REQUEST" and r.get("id") is not None else ""))
                it.setData(Qt.ItemDataRole.UserRole, (r["kind"], r.get("id")))
                if r["kind"] in ("SEAM", "MOMENT", "EVENT") and r.get("t") is not None and r["t"] < 20.0:
                    it.setForeground(Qt.GlobalColor.yellow)
                elif r["kind"] in ("THEN", "CLOCK"):
                    it.setForeground(Qt.GlobalColor.gray)
                elif r["kind"] == "IDEA":
                    it.setForeground(Qt.GlobalColor.magenta)
                elif r["kind"] == "REQUEST":
                    it.setForeground(Qt.GlobalColor.cyan)
                self.plan_list.addItem(it)
        else:
            for i, r in enumerate(rows):
                it = self.plan_list.item(i)
                if it is not None:
                    txt = line(r)
                    if it.text() != txt:
                        it.setText(txt)

    def _decks_menu(self, pos):
        it = self.decks.itemAt(pos)
        if it is None or self.director is None or it.data(Qt.ItemDataRole.UserRole) is None:
            return
        tid = it.data(Qt.ItemDataRole.UserRole)
        m = self._song_actions(tid, from_queue=False)
        first = m.actions()[0] if m.actions() else None
        is_rec = it.text().startswith("▶")
        off = m.addAction("✕ take it off the decks" + ("  (the record: NEXT or DROP instead)" if is_rec else "  (its parts back to the record; Delete)"))
        off.setEnabled(not is_rec)
        off.triggered.connect(lambda _c, t=tid: self._eject(t))
        m.removeAction(off)
        m.insertAction(first, off)
        m.insertSeparator(first)
        m.exec(self.decks.mapToGlobal(pos))

    def _eject(self, tid):
        if self.director is None or tid is None:
            return
        self.moment_lbl.setText(self.director.eject(tid))
        self._decks_sig = None
        self._render_queue()

    def _render_decks(self):
        """PLAYING: what is on the decks. Rebuilt only when it changes; when the RECORD changes, its row is
        highlighted so MIXES WITH and the parts window show the playing song's context."""
        d = self.director
        rows = []
        try:
            if d is not None and d.rc is not None and d.rc.master is not None:
                rc = d.rc
                words = {"bass": "bass", "drums": "drums", "other": "melody", "vocals": "vocal"}
                sm = rc.seam
                for deck, s in rc.songs.items():
                    held = s.held(rc.lanes)
                    if deck == rc.record_deck:
                        since = (rc.bar_n - s.play_from_k) if s.play_from_k is not None else 0
                        from lib.dj.submix import RATE
                        bar_s = rc._bar_clock() / RATE
                        rows.append((0, s.track.id, f"▶ {s.track.title[:34]}   the record · {_mmss(since * bar_s)} in"))
                    elif s.leaving:
                        rows.append((3, s.track.id, f"↘ {s.track.title[:34]}   leaving"))
                    elif held:
                        rows.append((1, s.track.id, f"▸ {s.track.title[:34]}   in: " + ", ".join(words[ln] for ln in held)))
                    elif s.staged_at is not None:
                        what = "next record" if (sm is not None and sm.get("b") == deck) else ("your pick" if getattr(s, "user_pick", False) else ("a part on its way" if getattr(s, "part_only", False) else "staged"))
                        rows.append((2, s.track.id, f"○ {s.track.title[:34]}   {what}"))
                    else:
                        rows.append((2, s.track.id, f"… {s.track.title[:34]}   decoding"))
                rows.sort(key=lambda r: r[0])
            elif d is not None and d.system is not None and d.system.current is not None:
                cur = d.system.current
                rows.append((0, cur.id, f"▶ {cur.title[:34]}   playing"))
                nxt = d.system.next_track
                if nxt is not None:
                    rows.append((2, nxt.id, f"○ {nxt.title[:34]}   next"))
        except Exception as e:  # noqa: BLE001
            self.err_lbl.setText(f"decks: {type(e).__name__}: {e}")
            return
        sig = tuple((tid, text.split("   ")[0], text.split("·")[-1][:6] if "·" not in text else "") for _o, tid, text in rows)
        rec_id = rows[0][1] if rows and rows[0][0] == 0 else None
        if sig != getattr(self, "_decks_sig", None):
            self._decks_sig = sig
            cur = self.decks.currentItem().data(Qt.ItemDataRole.UserRole) if self.decks.currentItem() else None
            self.decks.blockSignals(True)
            self.decks.clear()
            for _o, tid, text in rows:
                it = QListWidgetItem(text)
                it.setData(Qt.ItemDataRole.UserRole, tid)
                it.setToolTip(self.director.title(tid) if self.director is not None else "")
                self.decks.addItem(it)
                if tid == cur:
                    self.decks.setCurrentItem(it)
            self.decks.blockSignals(False)
        elif rows:
            # the record's "in" time ticks: refresh the first row's text without rebuilding
            it0 = self.decks.item(0)
            if it0 is not None and it0.text() != rows[0][2]:
                it0.setText(rows[0][2])
        if rec_id is not None and rec_id != getattr(self, "_decks_rec", None):
            self._decks_rec = rec_id
            it0 = self.decks.item(0)
            if it0 is not None:
                self.decks.setCurrentItem(it0)        # (fires _highlighted: MIXES WITH and the parts window follow the new record)

    def _render_queue(self):
        """UP NEXT with each song's state: ○ staged and ready, ▸ in play (which parts), … decoding, nothing =
        waiting for a deck. Rebuilt only when it changes (right-click a row for the same commands as SONGS)."""
        try:
            self._render_decks()
        except Exception as e:  # noqa: BLE001
            self.err_lbl.setText(f"decks: {type(e).__name__}: {e}")
        if self.director is None:
            if self.queue.count():
                self.queue.clear()
            return
        rows = self.director.up_next_status()
        sig = tuple((r["id"], r["state"], tuple(r["lanes"])) for r in rows)
        if sig == getattr(self, "_queue_sig", None):
            return
        self._queue_sig = sig
        cur = self.queue.currentItem().data(Qt.ItemDataRole.UserRole) if self.queue.currentItem() else None
        self.queue.clear()
        words = {"bass": "bass", "drums": "drums", "other": "melody", "vocals": "vocal"}
        for r in rows:
            mark = {"staged": "○ ", "in play": "▸ ", "decoding": "… ", "leaving": "↘ ", "next": "▸ "}.get(r["state"], "")
            tail = ("   in: " + ", ".join(words[ln] for ln in r["lanes"])) if r["lanes"] else ({"staged": "   ready on a deck", "next": "   next record"}.get(r["state"], ""))
            it = QListWidgetItem(mark + r["title"][:40] + tail)
            it.setData(Qt.ItemDataRole.UserRole, r["id"])
            it.setToolTip(f"{r['title']} - {r['state']}" + (f", its {', '.join(r['lanes'])} sounding now" if r["lanes"] else "")
                          + "\nright-click: its stems in, TAKEOVER / BRIDGE with it, play next, remove")
            self.queue.addItem(it)
            if r["id"] == cur:
                self.queue.setCurrentItem(it)

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
        do = st.get("last_dropout")
        drop_note = ""
        if do and time.time() - do.get("t", 0) < 180:
            drop_note = f"   ·   ⚠ DROPOUT {time.time() - do['t']:.0f} s ago" + (f" ({do['length_s']:.1f} s)" if do.get("length_s") else " (still silent)") \
                + (f" during {do['moment']}" if do.get("moment") else "") + (f" while switching to {do['switching']}" if do.get("switching") else "")
        self.state_lbl.setText(f"{mode}   ·   {st.get('theme')}   ·   {st.get('pool') or 'whole library'}" + drop_note
                               + (f"   ·   switching to {st['switching']}…" if st.get("switching") else "")
                               + (f"   ·   ⚠ DOUBLED: {', '.join(doubled)} (two songs' {'stems' if doubled != ['mix'] else 'mixes'} at once)" if doubled else "")
                               + (f"   ·   {last[:60]}" if last else "")
                               + (f"   ·   {self._midi_note}" if getattr(self, "_midi_note", "") else ""))
        # the moments: the armed one lit with its countdown, the rest plain; the tooltip carries how often each
        # was armed or refused tonight (and why) and the numbers the ear's words have moved
        mo = st.get("moment")
        self._rate_target = st.get("rate_target")
        stats, tune = st.get("moment_stats") or {}, st.get("moment_tune") or {}
        for key, label in MOMENT_ROW:
            b = self.moment_btns[key]
            armed = bool(mo and mo.get("kind") == key)
            base = "DROP" if (key == "slam" and st.get("mode") != "layered") else label
            text = (f"{label} {'▶' if mo['started'] else 'in'} {max(0, mo['in_bars'])}" if armed else base)
            if b.text() != text:
                b.setText(text)
            if b.isChecked() != armed:
                b.setChecked(armed)
            tip = self._moment_tips.get(key, "")
            s = stats.get(key)
            if s:
                refused = sorted(s.get("refused", {}).items(), key=lambda kv: -kv[1])
                tip += f"\n\ntonight: armed {s.get('armed', 0)}, refused {sum(n for _, n in refused)}" + (" (" + ", ".join(f"{k} ×{n}" for k, n in refused[:3]) + ")" if refused else "")
            t = tune.get(key)
            if t:
                tip += "\nnumbers: " + ", ".join(f"{k} {v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k} {v}" for k, v in t.items())
            if b.toolTip() != tip:
                b.setToolTip(tip)
        if mo:
            words = f"{mo['label']} ({mo['who']}): {'running' if mo['started'] else 'armed'}, the one in {mo['in_bars']} bars on {mo['what']}"
            if mo.get("other") and mo["kind"] in ("takeover", "bridge"):
                words += f" of {mo['other'][:24]}"
            if self.moment_lbl.text() != words:
                self.moment_lbl.setText(words)
        elif getattr(self, "_moment_was", None):
            self.moment_lbl.setText("")
        self._moment_was = bool(mo)
        # the program over the next songs: which is lit, where it stands
        ps = st.get("program")
        want_i = self.prog_box.findData(ps.get("key")) if ps else 0
        if self.prog_box.currentIndex() != max(0, want_i):
            self.prog_box.blockSignals(True)
            self.prog_box.setCurrentIndex(max(0, want_i))
            self.prog_box.blockSignals(False)
        self.prog_cancel.setVisible(bool(ps))
        if ps:
            steps = ps.get("steps") or []
            words = "  →  ".join((("▶ " if s["now"] else ("✓ " if s["done"] else "")) + f"song {s['i'] + 1}: {s['text']}") for s in steps)
            self.prog_lbl.setText(f"{ps['label']} over {ps['n']} songs:  {words}")
        else:
            self.prog_lbl.setText("")
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
            exit_in = nxt.get("exit_in_s")
            self.next_lbl.setText("NEXT  " + (nxt.get("title") or ""))
            if exit_in is not None:
                # the record's exit is the fixed point; the seam's own length says how much earlier it begins
                when = f"this record ends in {exit_in:.0f} s"
                if nxt.get("how"):
                    when += f"   ·   {nxt['how']}" + (f", {nxt['beats']:.0f} beats" if nxt.get("beats") else "") + (f", begins in about {eta:.0f} s" if eta is not None else "")
                else:
                    when += "   ·   the seam is planned when the exit comes into range"
            else:
                when = (f"in about {eta:.0f} s" if eta is not None else "when the exit comes") + (f"   ·   {nxt.get('how')}" if nxt.get("how") else "") \
                    + (f" {nxt['beats']} beats" if nxt.get("beats") else "")
            self.next_sub.setText(f"{nxt.get('artist') or ''}   " + when)
        else:
            self.next_lbl.setText("NEXT  not chosen yet")
            self.next_sub.setText("")
        self.intent_lbl.setText(st.get("intent") or "")
        try:
            self._render_plan(st.get("plan_rows") or [])
            # the PAIRS list follows the clock and what has played: refreshed every thirty seconds while its tab shows
            self._pairs_tick = getattr(self, "_pairs_tick", 0) + 1
            if self.right_tabs.currentWidget() is self.pairs_tab and (self._pairs_tick % 120 == 1):
                try:
                    self._fill_pairs()
                except Exception as e:  # noqa: BLE001
                    self.pairs_lbl.setText(f"pairs: {type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            self.err_lbl.setText(f"plan: {type(e).__name__}: {e}")
        # the song maps: the playing record (by id when the status carries one, else by title) and the next
        try:
            lib_by_title = {t.title: t for t in d.library}
            self.now_map.set(lib_by_title.get((now or {}).get("title")), (now or {}).get("pos_s"))
            self.next_map.set(lib_by_title.get((nxt or {}).get("title")), None)
        except Exception:
            pass
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
        # the SONGS list re-ranks when its REFERENCE changes (the record, or the last song UP NEXT) and
        # otherwise every 30 s - re-ranking every 3 s reordered the rows under the hand ("things bounce")
        ref_key = ((now or {}).get("title"), tuple(d.up_next[-1:]), (nxt or {}).get("title"))
        if (ref_key != getattr(self, "_songs_ref", None) or self._songs_tick % 120 == 0) \
                and time.time() - getattr(self, "_found_click_t", 0.0) > 2.0:      # never under a hand that just clicked
            self._songs_ref = ref_key
            self._fill_search()
        if self._songs_tick % 40 == 0:
            self._fill_tags()                         # the library may have been rescanned or retagged meanwhile
        err = st.get("error")
        self.err_lbl.setText(f"ERROR {err}" if err else "")
        if self.follow_btn is None or self.follow_btn.isChecked():
            self.canvas.follow(d.bar())
        if self.viewer.isVisible():
            self.viewer.update()                      # its spectrogram arrives in the background; its playhead moves
        for tid in d.tl.tracks():
            if tid and self.spectro.get(tid) is None and tid not in self.spectro._busy and tid not in self.spectro.errors:
                t = self.track(tid)
                if t is not None and getattr(t, "has_stems", False):
                    self.spectro.request(t)
        self.canvas.update()
