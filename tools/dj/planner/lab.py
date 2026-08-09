"""Lab: one seam, several lenses.

Replaces four separate tabs (Seam Lab, Beat Check, Exit Compare, Gate
Check) that all did the same three things - get a brain-planned seam,
render it, play it - and differed only in what they DREW afterwards. Four
tabs meant four seam pickers, four renders and four playbacks of four
DIFFERENT seams, so nothing you saw in one could be checked against
another.

Here the seam is picked and rendered ONCE, and the lens switches over it:

  Scope   what the blend does to the sound - the automation the render
          actually ran (tools/dj/planner/seamscope.py)
  Beat    per-deck band envelopes with measured kick ticks, deck A up /
          deck B down about a shared centreline, so beat alignment is
          read straight across (tools/dj/planner/beatcheck.py)
  Exit    where the outgoing track is left, on a picture of the whole
          song, against the pre-2026-08-07 engine's choice
          (tools/dj/planner/exitcompare.py)

One render feeds all three: `render_tapped` returns the mix (playback,
Exit), the per-deck taps (Beat) and - since 2026-08-08 - the same `info`
automation dict `audition.render_seam` fills (Scope).

Rating is unchanged from Seam Lab: good / passable / bad / skip, appended
to logs/seam_lab_ratings.jsonl, with good and bad additionally writing the
cross-night seam_feedback the live thumbs use (source "lab").

NOT carried over (2026-08-08, deliberately):
  * Gate Check's trial loop. It did its job - its verdicts are what
    retired the 20ms kick screen (see tools/tests/_dj_kickdelta_test.py).
    The panel is deleted; lib/dj/gateprobe.py (which reads its bars from
    the gates themselves, and is used by brain.py) and the 42 logged
    verdicts in logs/gate_ratings.jsonl both remain.
  * Seam Lab's single-knob probe staircase. The plain verdict is what the
    420 logged ratings mostly are. The panel is deleted, but the staircase
    itself lives in tools/dj/planner/seamprobe.py and is still used by
    dj_knobsweep and seamstats, so reviving it means new UI, not new
    logic.

Both panels are recoverable from git (deleted in the commit that added
this file).
"""
import json
import os
import random
import time

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QShortcut, QKeySequence
from PyQt6.QtWidgets import (QButtonGroup, QComboBox, QHBoxLayout, QLabel,
                             QPushButton, QRadioButton, QStackedWidget,
                             QTextBrowser, QVBoxLayout, QWidget)

from lib.dj import stretch_engine_name
from lib.dj.version import engine_version
from lib.dj.brain import Brain
from lib.dj.themes import BUILTIN_THEMES, get_theme

RATE = 44100

_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "logs",
    "seam_lab_ratings.jsonl")

# Source of seams. "long blend" is Beat Check's old filter: retry until the
# brain itself plans a real overlapped blend on beat-heavy material, which
# is the only case where beat alignment can be judged at all.
_SRC_ANY = "any seam"
_SRC_LONG = "long blends only"
_SOURCES = (_SRC_ANY, _SRC_LONG)

_BLEND_STYLES = ("long_blend", "bass_swap", "filter_sweep", "stem_bass_swap")

_VERDICTS = (("good", "Good (1)"), ("passable", "Passable (2)"),
             ("bad", "Bad (3)"), ("skip", "Skip (4)"))

# COVERAGE BALANCE (carried over from Seam Lab). Share of seams that get a
# style pinned toward the least-evidenced style. The rest are the brain's
# own choice - forcing a starved style onto an arbitrary pair rates WORSE
# than letting the brain pick (measured: 0.26 vs 0.35 mean verdict), so the
# lab buys evidence on some seams and stays representative on others.
_PIN_SHARE = 0.45
_STYLES = sorted(set(get_theme("groove").style_weights) | {
    "stem_drum_swap", "acapella_out", "stem_bass_swap", "drum_bridge",
    "acapella_in", "melody_carry", "phrase_cut", "spinback_cut",
    "loop_in", "breakdown_swap"})


class _SeamWorker(QThread):
    """Pick a pair, plan it, render it tapped - off the UI thread.

    ONE worker for every lens. The payload carries everything all three
    draw from, so switching lens never re-renders and never re-picks:
      audio  - the mix, for playback and the Exit lane's timing
      decks  - per-deck post-EQ taps, for the Beat lens
      marks  - blend/swap in render seconds, plus each deck's position
               trace, for the Beat lens
      info   - the automation the render ran, for the Scope lens
      bands  - the measured band/kick data, for the Beat lens
    """
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, db, brain, library, rng, want_style, source,
                 used_ids, relaxed_ids):
        super().__init__()
        self.db, self.brain, self.library = db, brain, library
        self.rng, self.want_style, self.source = rng, want_style, source
        self.used_ids = used_ids
        self.relaxed_ids = relaxed_ids

    def run(self):
        try:
            from lib.dj import beatpower as bp
            long_only = self.source == _SRC_LONG
            # NO GATE OVERRIDES, EVER (inherited from Beat Check, learned
            # the hard way 2026-08-05): a pinned style still goes through
            # every gate. Rendering a seam the engine would REFUSE teaches
            # nothing about the engine.
            #
            # But a refused pin is NOT a reason to throw the seam away. The
            # coverage pin is an ATTEMPT at a starved style, and a style is
            # starved precisely because it rarely survives the gates - so
            # requiring the pin to land made ~45% of seams unrenderable and
            # the treadmill died on "no renderable seam" (2026-08-08). The
            # fallback the gates chose is a real seam worth rating; the card
            # says which pin was refused and the log keeps want_style beside
            # style, which is how the analysis sees pins that never land.
            #
            # The long-blend source is the exception: it exists to judge
            # beat alignment, which needs an actual overlap, so there it
            # retries until the brain plans one on beat-heavy material.
            for veto in (self.used_ids, self.relaxed_ids):
                self.brain.veto_ids = set(veto)
                tries = 300 if long_only else 24
                for attempt in range(tries):
                    cur = self.rng.choice(self.library)
                    # Long test wants a workable render window on both
                    # sides; a plain seam only needs enough track to arm in.
                    if cur.id in veto or cur.duration_s < 120:
                        continue
                    if long_only and not (150 <= cur.duration_s <= 480):
                        continue
                    if long_only and bp.blendable(cur.id) is not True:
                        continue
                    if attempt % 10 == 0:
                        self.status.emit(
                            f"looking for a long blend... ({attempt})"
                            if long_only else
                            f"picking a partner for {cur.title[:28]}...")
                    self.brain.note_played(cur)
                    cand, meta = self.brain.choose_next(cur, 0.6, cur.bpm)
                    if cand is None:
                        continue
                    if long_only and (cand.duration_s > 480
                                      or bp.blendable(cand.id) is not True):
                        continue
                    self.brain.note_played(cand)
                    after_s = cur.duration_s * self.rng.uniform(0.35, 0.65)
                    plan = self.brain.plan_transition(
                        cur, cand, meta, after_s=after_s,
                        force_style=self.want_style)
                    if long_only and not (plan["style"] in _BLEND_STYLES
                                          and int(plan.get("beats") or 0) >= 16):
                        continue          # not a real overlap - repick
                    self.status.emit(f"rendering {plan['style']}: "
                                     f"{cur.title[:22]} -> {cand.title[:22]}...")
                    from tools.dj.dj_knobsweep import render_tapped
                    info = {}
                    mix, decks, marks = render_tapped(self.db, cur, cand,
                                                      dict(plan), info=info)
                    self.status.emit("measuring bands and beats...")
                    bands = None
                    try:
                        from tools.dj.planner.beatcheck import measure_bands
                        bands = measure_bands(cur, cand, plan, mix, decks,
                                              marks)
                    except Exception as e:
                        # The Beat lens is the only casualty; Scope and
                        # Exit still work, so don't lose the whole seam.
                        self.status.emit(f"band measure failed: {e}")
                    self.ready.emit({
                        "a": cur, "b": cand,
                        "plan": info.get("plan", plan),
                        "after_s": after_s, "audio": mix,
                        "decks": decks, "marks": marks,
                        "info": info, "bands": bands,
                        "want_style": self.want_style})
                    return
            self.failed.emit(
                "no long overlapped blend found on beat-heavy material - "
                "switch to 'any seam'" if long_only else
                "no renderable seam found - the library may be too small "
                "or every candidate is vetoed this session")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


class LabTab(QWidget):
    """One seam, three lenses, one verdict. Keys: 1 good, 2 passable,
    3 bad, 4 skip, R replay."""

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        from tools.dj.planner.player import TrackPlayer
        self.player = TrackPlayer()
        self.rng = random.Random()
        self.brain = None
        self._worker = None
        self._next = None                # pre-rendered seam waiting its turn
        self._current = None
        self._session = False
        self._used = set()               # every track heard this session
        self._recent = []                # the tail, as a relaxed fallback
        self._tally = {"good": 0, "passable": 0, "bad": 0, "skip": 0}
        self._mem_brain = None
        self._style_counts = {}          # style -> ratings so far
        self._style_dead = set()         # asked for repeatedly, never landed

        v = QVBoxLayout(self)

        # ---- source + transport (shared by every lens) -------------------
        row = QHBoxLayout()
        row.addWidget(QLabel("Theme:"))
        self.theme_box = QComboBox()
        self.theme_box.addItems(sorted(BUILTIN_THEMES))
        self.theme_box.setCurrentText("groove")
        self.theme_box.setToolTip(
            "Theme the generator's brain runs under - steers candidate "
            "choice and the style dice exactly like a live night.")
        self.theme_box.currentTextChanged.connect(self._reset_brain)
        row.addWidget(self.theme_box)
        row.addWidget(QLabel("Seams:"))
        self.source_box = QComboBox()
        self.source_box.addItems(_SOURCES)
        self.source_box.setToolTip(
            "'long blends only' retries until the brain plans a real "
            "overlapped blend on beat-heavy material - the only case where "
            "the Beat lens has anything to judge.")
        row.addWidget(self.source_box)
        self.start_btn = QPushButton("▶ Start session")
        self.start_btn.setToolTip(
            "Generate seams one after another: the next renders while you "
            "listen to this one. Rate (or skip) to advance.")
        self.start_btn.clicked.connect(self._toggle_session)
        row.addWidget(self.start_btn)
        self.replay_btn = QPushButton("↻ Replay (R)")
        self.replay_btn.clicked.connect(self._replay)
        self.replay_btn.setEnabled(False)
        row.addWidget(self.replay_btn)
        stop = QPushButton("■ Stop audio")
        stop.clicked.connect(lambda: self.planner.stop_all_playback())
        row.addWidget(stop)
        row.addStretch(1)
        v.addLayout(row)

        self.card = QLabel(
            "Press Start - the lab generates a seam the way a live night "
            "would, renders it once, and every lens below shows THAT seam.")
        self.card.setWordWrap(True)
        self.card.setStyleSheet("font-size: 15px; padding: 10px;")
        v.addWidget(self.card)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #9aa0a8; padding: 0 10px;")
        v.addWidget(self.detail)

        # ---- lens switcher ------------------------------------------------
        lens_row = QHBoxLayout()
        lens_row.addWidget(QLabel("Lens:"))
        self.lens_group = QButtonGroup(self)
        self._lens_names = ("Scope", "Beat", "Exit")
        for i, (name, tip) in enumerate((
                ("Scope", "What the blend does to the sound - the exact "
                          "automation this render ran."),
                ("Beat", "Per-deck band envelopes and measured kicks. Deck A "
                         "up, deck B down: matched kicks meet at the "
                         "centreline."),
                ("Exit", "Where this seam leaves the outgoing track, on a "
                         "picture of the whole song, against the old "
                         "engine's exit."))):
            rb = QRadioButton(name)
            rb.setToolTip(tip)
            if i == 0:
                rb.setChecked(True)
            self.lens_group.addButton(rb, i)
            lens_row.addWidget(rb)
        self.lens_group.idClicked.connect(self._set_lens)
        self.legend = QLabel("")
        self.legend.setStyleSheet("color:#888;")
        lens_row.addWidget(self.legend, 1)
        v.addLayout(lens_row)

        # ---- the lenses ---------------------------------------------------
        from tools.dj.planner.seamscope import SeamScope
        from tools.dj.planner.beatcheck import _BandCanvas
        from tools.dj.planner.exitcompare import ExitLanes
        self.scope = SeamScope()
        self.scope.seekRequested.connect(self._seek)
        self.bands = _BandCanvas()
        self.bands.seekRequested.connect(self._seek)
        self.lanes = ExitLanes()
        self.stack = QStackedWidget()
        for w in (self.scope, self.bands, self.lanes):
            self.stack.addWidget(w)
        self.stack.setMinimumHeight(260)
        v.addWidget(self.stack, 1)

        self.note_lbl = QLabel("")
        self.note_lbl.setWordWrap(True)
        self.note_lbl.setStyleSheet("color: #7a8b99; padding: 0 4px;")
        v.addWidget(self.note_lbl)

        # ---- verdict ------------------------------------------------------
        row = QHBoxLayout()
        self.rate_btns = {}
        for verdict, label in _VERDICTS:
            b = QPushButton(label)
            b.setEnabled(False)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _, w=verdict: self._rate(w))
            self.rate_btns[verdict] = b
            row.addWidget(b)
        v.addLayout(row)

        self.status = QLabel("")
        v.addWidget(self.status)
        self.tally_lbl = QLabel("")
        v.addWidget(self.tally_lbl)

        # WHAT THE RATINGS HAVE TAUGHT, across every session - the log is
        # only worth keeping if its findings are visible while you rate.
        self.learn_lbl = QTextBrowser()
        self.learn_lbl.setOpenExternalLinks(False)
        self.learn_lbl.setStyleSheet(
            "QTextBrowser { background: palette(alternate-base);"
            " border: 1px solid palette(mid); border-radius: 4px;"
            " padding: 6px; font-size: 11px; }")
        self.learn_lbl.setToolTip(
            "Aggregated from logs/seam_lab_ratings.jsonl (every session, "
            "including passable) plus the live cross-night memory.\n\n"
            "Percentages are the GOOD share of DECIDED verdicts - "
            "passable is counted in the totals but abstains. Buckets with "
            "fewer than 5 ratings are dropped rather than shown as "
            "confident numbers, and thin ones are marked.")
        self.learn_lbl.setMinimumHeight(150)
        v.addWidget(self.learn_lbl, 1)
        self._refresh_learning()

        self._head_timer = QTimer(self)
        self._head_timer.setInterval(80)
        self._head_timer.timeout.connect(self._tick_playhead)
        self._head_timer.start()

        for i, (verdict, _l) in enumerate(_VERDICTS, start=1):
            QShortcut(QKeySequence(str(i)), self,
                      activated=lambda w=verdict: self._rate(w))
        QShortcut(QKeySequence("R"), self, activated=self._replay)

        self._set_lens(0)

    # ---- lens ------------------------------------------------------------
    def _set_lens(self, idx):
        self.stack.setCurrentIndex(idx)
        self.legend.setText((
            "click to play from there",
            "deck A ▲ amber, deck B ▼ cyan · solid tick = measured kick, "
            "dashed = stored grid · wheel zoom, drag pan, click plays",
            "red = the pre-2026-08-07 engine's exit, green = this build · "
            "hatched = the play-time budget's floor",
        )[idx])
        if self._current is not None:
            self._show(self._current, lens_only=True)

    # ---- brain -----------------------------------------------------------
    def _reset_brain(self, *_):
        self.brain = None

    def _starved_style(self):
        """A style to ATTEMPT, biased hard toward the least-evidenced.

        Diagnosis needs samples: a style sitting at 0-of-4 cannot tell you
        whether it is broken everywhere or only on loose grids. This only
        affects what the lab generates - the live night's own style dice
        are untouched. A pin that the gates refuse just repicks, so a
        starved style never forces an illegal seam."""
        if self.rng.random() > _PIN_SHARE:
            return None                  # let the brain choose this one
        counts = self._style_counts or {}
        pool = [(k, counts.get(k, 0)) for k in _STYLES
                if k not in self._style_dead] or \
               [(k, counts.get(k, 0)) for k in _STYLES]
        weights = [1.0 / (1.0 + n) for _k, n in pool]
        r = self.rng.random() * sum(weights)
        for (k, _n), w in zip(pool, weights):
            r -= w
            if r <= 0:
                return k
        return pool[-1][0]

    def _ensure_brain(self):
        if self.brain is None:
            self.brain = Brain(self.planner.library,
                               get_theme(self.theme_box.currentText()),
                               seed=self.rng.randrange(1 << 30))
        return self.brain

    # ---- session ---------------------------------------------------------
    def _toggle_session(self):
        self._session = not self._session
        self.start_btn.setText("■ Stop session" if self._session
                               else "▶ Start session")
        if self._session:
            self._pump()

    def _pump(self):
        """Keep exactly one render in flight and at most one queued."""
        if not self._session or self._worker is not None:
            return
        if self._next is not None and self._current is not None:
            return
        if not self.planner.library:
            self.status.setText("library is empty - scan first")
            return
        want = self._starved_style()
        brain = self._ensure_brain()
        self._worker = _SeamWorker(
            self.planner.db, brain, self.planner.library, self.rng, want,
            self.source_box.currentText(), set(self._used),
            set(self._recent[-12:]))
        self._worker.status.connect(self.status.setText)
        self._worker.ready.connect(self._seam_ready)
        self._worker.failed.connect(self._seam_failed)
        self._worker.finished.connect(self._worker_done)
        self._worker.start()

    def _worker_done(self):
        self._worker = None
        self._pump()                     # prefetch the next one

    def _seam_ready(self, seam):
        for t in (seam["a"], seam["b"]):
            self._used.add(t.id)
            self._recent.append(t.id)
        if self._current is None:
            self._show(seam)
        else:
            self._next = seam            # waits until this one is rated

    def _seam_failed(self, msg):
        self.status.setText(msg)
        self._session = False
        self.start_btn.setText("▶ Start session")

    # ---- present ---------------------------------------------------------
    def _show(self, seam, lens_only=False):
        """Draw `seam` into whichever lens is showing. `lens_only` re-draws
        after a lens switch without touching playback."""
        self._current = seam
        a, b, plan = seam["a"], seam["b"], seam["plan"]
        render_s = len(seam["audio"]) / RATE
        idx = self.lens_group.checkedId()
        if idx == 0:
            self.scope.set_seam(a, b, seam.get("info"), seam["after_s"],
                                render_s)
        elif idx == 1:
            if seam.get("bands"):
                self.bands.set_data(seam["bands"])
            else:
                self.note_lbl.setText(
                    "Beat lens: no band measurement for this seam "
                    "(measuring failed, or the render carried no taps).")
        else:
            # ExitLanes wants its OWN seam shape, not the Lab payload: an
            # exit dict per variant keyed "cur"/"new", plus the blend length
            # in seconds. In simple mode it draws one exit ("cur"), but
            # _render_t0 can still be asked for "new", so both are supplied.
            exit_at = {"out_s": plan.get("out_s", 0.0), "after_s": 0.0,
                       "fallback": False}
            self.lanes.set_seam(
                {"a": a, "b": b, "cur": exit_at, "new": dict(exit_at),
                 "in_s": plan.get("in_s", 0.0),
                 "beats": plan.get("beats") or 0,
                 "blend_s": (plan.get("beats") or 0) * a.period_s or 14.0},
                simple=True)
        if lens_only:
            return

        want = seam["want_style"]
        pin_note = ""
        if want and plan["style"] != want:
            why = ((plan.get("diag") or {}).get("style_pin") or {})
            pin_note = (f"   (wanted {want}: refused "
                        f"{why.get('why_not') or 'by gates'})")
        self.card.setText(
            f"<b>{a.title}</b> → <b>{b.title}</b><br>"
            f"{plan['style']} @ rate {plan['rate']:.3f}{pin_note}")
        self.detail.setText(
            f"{a.bpm:.0f}→{b.bpm:.0f} bpm · keys {a.camelot or '?'}→"
            f"{b.camelot or '?'} · pair score {plan.get('pair_score', 0):.2f}"
            f" · {plan.get('beats', 0)} beats · pitch "
            f"{plan.get('pitch_st', 0) or 0:+g} st · armed at "
            f"{seam['after_s']:.0f}s of {a.duration_s:.0f}s · "
            f"{render_s:.0f}s render")
        self.planner.claim_playback("lab")
        self.player.load(seam["audio"])
        self.player.play()
        for b_ in self.rate_btns.values():
            b_.setEnabled(True)
        self.replay_btn.setEnabled(True)
        n = sum(self._tally.values())
        self.status.setText(f"playing - rate to advance ({n} rated this "
                            f"session)")

    def _replay(self):
        if self._current is None:
            return
        self.planner.claim_playback("lab")
        self.player.load(self._current["audio"])
        self.player.play()

    def _seek(self, t):
        if self._current is None:
            return
        self.planner.claim_playback("lab")
        self.player.load(self._current["audio"])
        self.player.play()
        try:
            self.player.seek(float(t))
        except Exception:
            pass

    def _tick_playhead(self):
        if self._current is None:
            return
        t = self.player.time_s()
        idx = self.lens_group.checkedId()
        try:
            if idx == 0:
                self.scope.set_playhead(t)
            elif idx == 1:
                self.bands.set_playhead(t)
            else:
                self.lanes.set_playhead(t)
        except Exception:
            pass

    # ---- rating ----------------------------------------------------------
    def _memory_brain(self):
        if self._mem_brain is None:
            self._mem_brain = Brain(self.planner.library, get_theme("groove"))
        return self._mem_brain

    def _refresh_learning(self):
        """Re-aggregate every rating ever logged and redraw the readout.
        Called at startup and after each verdict - the dataset just grew."""
        from tools.dj.planner import seamstats
        brain = None
        try:
            brain = self._memory_brain()
        except Exception:
            pass
        try:
            rows = seamstats.read_ratings()
            # Evidence per style drives the balance-coverage picker.
            counts, attempts, landed = {}, {}, set()
            for r in rows:
                s = r.get("style")
                if s:
                    counts[s] = counts.get(s, 0) + 1
                    landed.add(s)
                wnt = r.get("want_style")
                if wnt:
                    attempts[wnt] = attempts.get(wnt, 0) + 1
            self._style_counts = counts
            # Retired or ungateable styles refuse every time (cut_at_drop
            # was retired 2026-08-02). Attempting them forever would burn
            # the balance rotation on seams that always fall back, so a
            # style asked for repeatedly and NEVER landed drops out.
            self._style_dead = {s for s, n in attempts.items()
                                if n >= 5 and s not in landed}
            sm = seamstats.analyze(rows, self.planner.library)
            html = seamstats.report_html(sm, brain)
        except Exception as e:
            import traceback
            traceback.print_exc()
            html = f"<p>could not build the ratings report: {e}</p>"
        pos = self.learn_lbl.verticalScrollBar().value()
        self.learn_lbl.setHtml(html)
        self.learn_lbl.verticalScrollBar().setValue(pos)   # keep the view

    def _rate(self, verdict):
        seam = self._current
        if seam is None or not self.rate_btns[verdict].isEnabled():
            return
        a, b, plan = seam["a"], seam["b"], seam["plan"]
        listened = self.player.time_s()
        self.player.pause()
        self._tally[verdict] = self._tally.get(verdict, 0) + 1
        if verdict != "skip":
            rec = {"t": time.time(), "verdict": verdict,
                   "a_id": a.id, "a_title": a.title,
                   "b_id": b.id, "b_title": b.title,
                   "style": plan["style"], "want_style": seam["want_style"],
                   "rate": plan["rate"],
                   "pitch_st": plan.get("pitch_st", 0),
                   "beats": plan.get("beats"),
                   "pair_score": plan.get("pair_score"),
                   "out_s": plan.get("out_s"), "in_s": plan.get("in_s"),
                   "after_s": seam["after_s"],
                   "theme": self.theme_box.currentText(),
                   "engine": stretch_engine_name(),
                   "ver": engine_version(),
                   "camelot_a": a.camelot, "camelot_b": b.camelot,
                   "bpm_a": round(a.bpm, 2), "bpm_b": round(b.bpm, 2),
                   "conf_a": round(a.bpm_conf or 0.0, 2),
                   "conf_b": round(b.bpm_conf or 0.0, 2),
                   "stems_a": bool(getattr(a, "has_stems", False)),
                   "stems_b": bool(getattr(b, "has_stems", False)),
                   "pair_class": list(Brain._pair_class(a, b)),
                   "rhythm": {k: v for k, v in (plan.get("rhythm") or {})
                              .items()
                              if k in ("score", "flam_ms", "conf",
                                       "kick_agreement", "meter_clash",
                                       "swing_a", "swing_b")},
                   "pin_why": ((plan.get("diag") or {}).get("style_pin")
                               or {}).get("why_not"),
                   "tune": plan.get("tune") or {},
                   "gate_test": (plan.get("diag") or {}).get("gate_test"),
                   "listened_s": round(listened, 1),
                   "render_s": round(len(seam["audio"]) / RATE, 1),
                   # Which lens was up when the verdict was given - the
                   # consolidated tab's one new field, so a later reading
                   # of the log can tell a by-ear verdict from one given
                   # while staring at the kick ticks.
                   "lens": self._lens_names[self.lens_group.checkedId()]}
            try:
                os.makedirs(os.path.dirname(_LOG), exist_ok=True)
                with open(_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                self.status.setText(f"could not log rating: {e}")
        if verdict in ("good", "bad"):
            # The same cross-night memory the live thumbs teach - pair,
            # feature class AND per-style taste (passable stays neutral).
            try:
                self.planner.db.add_seam_feedback(
                    a.id, b.id, plan["style"], verdict == "good",
                    source="lab")
                self.planner.pair_memory = self._memory_brain().pair_memory
            except Exception as e:
                self.status.setText(f"could not store feedback: {e}")
        self._current = None
        self.scope.clear()
        for b_ in self.rate_btns.values():
            b_.setEnabled(False)
        self.replay_btn.setEnabled(False)
        t = self._tally
        self.tally_lbl.setText("session: " + "   ".join(
            f"{k} {v}" for k, v in sorted(t.items()) if v))
        if verdict != "skip":
            self._refresh_learning()     # the dataset just grew
        if self._next is not None:
            nxt, self._next = self._next, None
            self._show(nxt)
        self._pump()

    # ---- lifecycle -------------------------------------------------------
    def stop_playback(self):
        try:
            self.player.pause()
        except Exception:
            pass

    def shutdown(self):
        self._session = False
        self.stop_playback()
        w = self._worker
        if w is not None and w.isRunning():
            try:
                w.wait(10000)            # a render can't be killed
            except Exception:
                pass
