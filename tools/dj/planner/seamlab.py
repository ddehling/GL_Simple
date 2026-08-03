"""Seam Lab: a rating treadmill for brain-planned seams.

Generates transitions the way a real night does (random arm point in a
random track, the real Brain choosing the next track and planning the
style), renders each through the shared offline audition renderer, plays
it, and asks for one verdict: good / passable / bad. Rating advances to
the next seam - the next render runs WHILE you listen, so the loop is
listen, press, listen.

Every verdict is appended to logs/seam_lab_ratings.jsonl with the full
plan context (pair, style, rate, pitch, pair score, arm point, engine,
how long you listened) - the analyzable dataset. Good/bad additionally
write the same cross-night seam_feedback the live thumbs use (source
"lab"), so the DJ learns from every session; passable is recorded but
steers nothing.
"""
import json
import os
import random
import time

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QTextBrowser, QVBoxLayout, QWidget)

from lib.dj import stretch_engine_name
from lib.dj.version import engine_version
from lib.dj.brain import Brain
from lib.dj.themes import BUILTIN_THEMES, get_theme

RATE = 44100
_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "logs",
    "seam_lab_ratings.jsonl")

# The rateable style vocabulary for the filter combo: theme keys plus the
# styles plan_transition defaults in when a theme dict predates them.
_BALANCE = "(balance coverage)"
_MODE_SEAM = "Rate the seam"
_MODE_PROBE = "Probe one parameter"
# Button slots, reused by both modes so the row never rebuilds.
_BUTTONS = ("a", "b", "c", "d", "skip")
_LABELS = {
    _MODE_SEAM: {
        "a": ("👍 Good (1)", "good",
              "Sounded like a DJ. Teaches the cross-night pair / class / "
              "style memory the live selection reads."),
        "b": ("😐 Passable (2)", "passable",
              "Not rough, not memorable. Counted in the totals; steers "
              "nothing."),
        "c": ("👎 Bad (3)", "bad",
              "Rough seam. Leans the memory away from this pair, its "
              "feature class and this style."),
        "d": ("", None, ""),
        "skip": ("⏭ Skip (4)", "skip", "No verdict - advance without "
                 "guessing."),
    },
    _MODE_PROBE: {
        "a": ("👍 Better (1)", "good",
              "The change described above IMPROVES the seam. The baseline "
              "moves halfway toward the tested value and re-checks."),
        "b": ("👎 Worse (2)", "bad",
              "The change makes it worse. Two of these in both directions "
              "and the parameter is called settled."),
        "c": ("🤷 Can't tell (3)", "cant_tell",
              "No audible difference. The MOST useful answer: the next "
              "test of this parameter uses a bigger change, and if even "
              "the extreme is inaudible the parameter is parked as "
              "imperceptible (its range may be too tight to matter)."),
        "d": ("❓ Don't get it (4)", "dont_get_it",
              "The description doesn't tell you what to listen for. Two "
              "of these and the parameter is parked as needing a better "
              "explanation - the fault is the wording, not you."),
        "skip": ("⏭ Skip (5)", "skip", "Wasn't listening - no answer "
                 "recorded."),
    },
}
# Track-slots (both sides of a seam) excluded from re-selection. ~40 seams
# of history: enough that nothing recurs while you still remember it,
# small enough to leave the candidate pool healthy.
_VETO_WINDOW = 80
# Share of seams that get a coverage pin. The rest are the brain's own
# choice - forcing a starved style onto an arbitrary pair rates WORSE
# than letting the brain pick (measured: 0.26 vs 0.35 mean verdict), so
# the lab buys evidence on some seams and stays representative on others.
_PIN_SHARE = 0.45
_STYLES = sorted(set(get_theme("groove").style_weights) | {
    "stem_drum_swap", "acapella_out", "stem_bass_swap", "drum_bridge",
    "acapella_in", "melody_carry", "phrase_cut", "spinback_cut",
    "loop_in", "breakdown_swap"})


class _GenWorker(QThread):
    """Pick a pair, plan it, render it - off the UI thread. The brain and
    rng are owned by the tab but only ever touched from ONE running
    worker at a time (the tab enforces that)."""
    ready = pyqtSignal(object)           # dict with audio + context
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, db, brain, library, rng, want_style, used_ids,
                 relaxed_ids, probe_mode=False, baseline=None,
                 probe_doc=None):
        super().__init__()
        self.db, self.brain, self.library = db, brain, library
        self.rng, self.want_style = rng, want_style
        self.probe_mode = probe_mode     # pick ONE parameter to test
        self.baseline = baseline or {}   # knob values in force
        self.probe_doc = probe_doc       # staircase state (read-only here)
        self.used_ids = used_ids         # every track heard this session
        self.relaxed_ids = relaxed_ids   # just the recent tail, as a fallback

    def run(self):
        try:
            # SAMPLE WITHOUT REPLACEMENT: a rating session should spend its
            # time on material it has not heard yet, so BOTH sides of every
            # seam are vetoed for the rest of the session. The second pass
            # relaxes to the recent tail only - a narrow tempo/key pocket
            # can genuinely run out of partners, and stalling the treadmill
            # would be worse than a spaced repeat.
            for veto in (self.used_ids, self.relaxed_ids):
                self.brain.veto_ids = set(veto)
                for attempt in range(8):
                    cur = self.rng.choice(self.library)
                    if cur.id in veto or cur.duration_s < 120:
                        continue
                    self.status.emit(f"picking a seam from "
                                     f"{cur.title[:28]}...")
                    self.brain.note_played(cur)
                    cand, meta = self.brain.choose_next(cur, 0.6, cur.bpm)
                    if cand is None:
                        continue
                    # A lab seam IS a play: without this the brain's
                    # distinct-song no-repeat never saw the incoming side,
                    # so its favourites won choose_next over and over.
                    self.brain.note_played(cand)
                    after_s = cur.duration_s * self.rng.uniform(0.35, 0.65)
                    plan = self.brain.plan_transition(
                        cur, cand, meta, after_s=after_s,
                        force_style=self.want_style)
                    from tools.dj.planner.seamtune import (RANGES,
                                                            apply_plan_knobs,
                                                            knobs_for)
                    # ONE parameter, or none. Random multi-knob jitter is
                    # gone: measurement showed a whole-seam verdict cannot
                    # separate simultaneous variables at any realistic
                    # rating volume.
                    #
                    # The probe is chosen HERE, after the style is known,
                    # and only from knobs that style's automation actually
                    # reads. Choosing it earlier probed things like the
                    # brake length on a long_fade - inaudible by
                    # construction, which the staircase would have read as
                    # "imperceptible" and parked a knob that was simply
                    # never exercised.
                    probe = None
                    if self.probe_mode:
                        from tools.dj.planner import seamprobe
                        usable = {k: RANGES[k] for k in knobs_for(
                            plan["style"], bool(plan.get("duck_vocal_a")))
                            if k in RANGES}
                        probe = seamprobe.next_probe(
                            usable, self.baseline, doc=self.probe_doc,
                            rng=self.rng)
                    if probe is not None:
                        plan["tune"] = {probe["knob"]: probe["value"]}
                        apply_plan_knobs(plan)   # beats_scale lives in plan
                    self.status.emit(
                        f"rendering {plan['style']}: {cur.title[:22]} -> "
                        f"{cand.title[:22]}...")
                    from lib.dj.audition import render_seam
                    info = {}            # exact automation, for the scope
                    audio = render_seam(self.db, cur, cand, plan, info=info)
                    self.ready.emit({"a": cur, "b": cand,
                                     "probe": probe,
                                     "plan": info.get("plan", plan),
                                     "after_s": after_s, "audio": audio,
                                     "info": info,
                                     "want_style": self.want_style})
                    return
            self.failed.emit("no renderable seam found in 16 tries")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")


class SeamLabTab(QWidget):
    """Generate -> listen -> rate -> next. Keyboard: 1 good, 2 passable,
    3 bad, 4 skip, R replay."""

    def __init__(self, planner):
        super().__init__()
        self.planner = planner
        from tools.dj.planner.player import TrackPlayer
        self.player = TrackPlayer()
        self.rng = random.Random()
        self.brain = None
        self._gen = None                 # running _GenWorker, one at a time
        self._next = None                # pre-rendered seam waiting its turn
        self._current = None             # the seam being played/rated
        self._recent = []                # recent cur ids - no instant repeats
        self._session = False
        self._played_at = None
        self._tally = {"good": 0, "passable": 0, "bad": 0, "skip": 0}
        self._mem_brain = None           # cached, for the learning panel
        self._ended = False              # playback hit the scope's end
        self._style_counts = {}          # style -> ratings so far
        self._style_dead = set()         # asked for repeatedly, never landed
        self.probe = None                # parameter under test, probe mode

        v = QVBoxLayout(self)

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
        row.addWidget(QLabel("Mode:"))
        self.mode_box = QComboBox()
        self.mode_box.addItems([_MODE_PROBE, _MODE_SEAM])
        self.mode_box.setToolTip(
            f"{_MODE_PROBE} — hold every setting at its baseline, move "
            "exactly ONE, and say which one moved so you can judge that "
            "rather than the seam as a whole. Four answers; 'can't tell' "
            "is the one that drives the search.\n\n"
            f"{_MODE_SEAM} — the plain treadmill: no parameter is "
            "touched, and your verdict teaches the pair / style / "
            "condition memory the live engine selects with.")
        self.mode_box.currentTextChanged.connect(self._mode_changed)
        row.addWidget(self.mode_box)
        row.addWidget(QLabel("Style:"))
        self.style_box = QComboBox()
        self.style_box.addItems([_BALANCE, "(brain's choice)"])
        self.style_box.addItems(_STYLES)
        self.style_box.setToolTip(
            "What style each generated seam attempts.\n\n"
            f"{_BALANCE} (default) — bias hard toward the styles with the "
            "least evidence. You cannot learn WHY a style fails, or where "
            "it works, from four seams; left to the brain's own dice the "
            "rare styles stay rare and stay unexplained.\n\n"
            "(brain's choice) — the distribution a real night produces.\n\n"
            "A named style pins every seam to it. Pins and balance picks "
            "both go through the REAL gates: when a pair can't carry the "
            "style the fallback plays, and the log records wanted vs got.")
        row.addWidget(self.style_box)
        self.start_btn = QPushButton("▶ Start session")
        self.start_btn.setToolTip(
            "Generate seams one after another: the next one renders while "
            "you listen to this one. Rate (or skip) to advance.")
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

        self.card = QLabel("Press Start - the lab generates a seam the "
                           "way a live night would, plays it, and moves "
                           "on as soon as you rate it.")
        self.card.setWordWrap(True)
        self.card.setStyleSheet("font-size: 15px; padding: 12px;")
        v.addWidget(self.card)
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #9aa0a8; padding: 0 12px;")
        v.addWidget(self.detail)

        from tools.dj.planner.seamscope import SeamScope
        self.strip = SeamScope()
        self.strip.seekRequested.connect(self._seek)
        # Bounded, NOT stretched: the scope is a glance at the mechanics,
        # the analysis pane below is where the session's meaning lives.
        self.strip.setMinimumHeight(150)
        self.strip.setMaximumHeight(210)
        v.addWidget(self.strip)
        self._head_timer = QTimer(self)
        self._head_timer.setInterval(80)
        self._head_timer.timeout.connect(self._tick_playhead)
        self._head_timer.start()

        # WHAT IS BEING TESTED - only meaningful in probe mode, where the
        # whole point is that you know which single thing moved.
        self.probe_lbl = QLabel("")
        self.probe_lbl.setWordWrap(True)
        self.probe_lbl.setStyleSheet(
            "QLabel { background: palette(highlight); color: palette("
            "highlighted-text); border-radius: 5px; padding: 9px;"
            " font-size: 14px; }")
        self.probe_lbl.hide()
        v.addWidget(self.probe_lbl)
        # What the LAST answer did. Separate from the status line, which
        # is immediately overwritten by "rendering the next seam".
        self.note_lbl = QLabel("")
        self.note_lbl.setWordWrap(True)
        self.note_lbl.setStyleSheet("color: #7a8b99; padding: 0 4px;")
        v.addWidget(self.note_lbl)

        row = QHBoxLayout()
        self.rate_btns = {}
        for verdict in _BUTTONS:
            b = QPushButton("")
            b.setEnabled(False)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _, w=verdict: self._rate(w))
            self.rate_btns[verdict] = b
            row.addWidget(b)
        v.addLayout(row)
        self._relabel_buttons()

        self.status = QLabel("")
        v.addWidget(self.status)
        self.tally_lbl = QLabel("")
        v.addWidget(self.tally_lbl)

        # WHAT THE RATINGS HAVE TAUGHT, across every session - the log is
        # only worth keeping if its findings are visible while you rate.
        # Scrollable rich text: this is the tab's main readout, not a
        # footnote, so it gets the stretch the scope gave up.
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
        v.addWidget(self.learn_lbl, 1)
        self._refresh_learning()

        for key, slot in (("1", "a"), ("2", "b"), ("3", "c"),
                          ("4", "d"), ("5", "skip")):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda w=slot: self._rate(w))
        sc = QShortcut(QKeySequence("R"), self)
        sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc.activated.connect(self._replay)

    # ---- session flow ----------------------------------------------------
    def _reset_brain(self, *_):
        self.brain = None                # rebuilt on next generation

    def _ensure_brain(self):
        if self.brain is None:
            self.brain = Brain(self.planner.library,
                               get_theme(self.theme_box.currentText()),
                               seed=self.rng.randrange(1 << 30))
            try:
                # Same memory the live night runs under (UI thread owns
                # the sqlite connection - workers never touch it).
                self.brain.load_pair_memory(self.planner.db)
            except Exception:
                pass
            # Re-seed the no-repeat memory across a theme switch - a
            # fresh brain otherwise reached for the same favorites the
            # session just heard.
            heard = set(self._recent[-60:])
            for t in self.planner.library:
                if t.id in heard:
                    self.brain.note_played(t)

    def _toggle_session(self):
        if self._session:
            self._session = False
            self.start_btn.setText("▶ Start session")
            self.status.setText("session paused - current seam still "
                                "rateable")
            return
        if not self.planner.library:
            self.status.setText("no library loaded")
            return
        self._session = True
        self.start_btn.setText("⏸ Pause session")
        self._pump()

    def _mode(self):
        return self.mode_box.currentText()

    def _relabel_buttons(self):
        spec = _LABELS[self._mode()]
        for slot, b in self.rate_btns.items():
            label, verdict, tip = spec[slot]
            b.setText(label)
            b.setToolTip(tip)
            b.setVisible(bool(label))

    def _mode_changed(self, *_):
        self._relabel_buttons()
        self.probe_lbl.setVisible(self._mode() == _MODE_PROBE)
        self.style_box.setEnabled(self._mode() == _MODE_SEAM)
        self._refresh_learning()

    def _want_style(self):
        if self._mode() == _MODE_PROBE:
            # A probe must isolate ONE variable; pinning a style on top
            # would confound the answer with a style change.
            return None
        s = self.style_box.currentText()
        if s == _BALANCE:
            return self._starved_style()
        return None if s.startswith("(") else s

    def _starved_style(self):
        """A style to ATTEMPT, biased hard toward the least-evidenced.

        Diagnosis needs samples: a style sitting at 0-of-4 cannot tell you
        whether it is broken everywhere or only on loose grids. This only
        affects what the lab generates - the live night's own style dice
        are untouched."""
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

    def _pump(self):
        """Keep exactly one render in flight whenever a seam is needed:
        the one being listened to came from the previous pump."""
        if not self._session or self._gen is not None:
            return
        if self._next is not None:
            return                       # one pre-rendered seam is plenty
        self._ensure_brain()
        # ROLLING veto window, not the whole session. Vetoing every track
        # heard starves choose_next: the eligible pool for a given outgoing
        # track is already narrowed by tempo and key, so a session-long
        # veto strips most of it and the brain settles for worse and worse
        # partners (measured over 92 rated seams: mean pair score fell
        # 0.31 -> 0.26 and the verdict mean fell 0.40 -> 0.22 across the
        # session). A window still kills the near-repeats that prompted
        # this - those were within a handful of seams.
        used = set(self._recent[-_VETO_WINDOW:])
        tail = set(self._recent[-20:])
        probe_mode = self._mode() == _MODE_PROBE
        baseline, probe_doc, want = {}, None, self._want_style()
        if probe_mode:
            from tools.dj.planner import seamprobe
            from tools.dj.planner.seamtune import RANGES, styles_reading
            from lib.dj import tuning
            from lib.dj.brain import TUNE_DEFAULTS
            baseline = tuning.current(TUNE_DEFAULTS)
            probe_doc = seamprobe.load()
            # UNBIASED COVERAGE: choose the least-answered knob FIRST, then
            # ask for a style that reads it. Left to the brain's own style
            # dice, the blend family would answer its knobs many times over
            # while echo, spinback and loop-roll knobs were never exercised
            # at all. The pin still goes through the real gates, so an
            # unsuitable pair simply falls back and the worker picks a
            # probe from whatever style actually played.
            openk = seamprobe.open_knobs(RANGES, probe_doc)
            if openk:
                fewest = min(seamprobe.trials_of(probe_doc, k)
                             for k in openk)
                target = self.rng.choice(
                    [k for k in openk
                     if seamprobe.trials_of(probe_doc, k) == fewest])
                cands = [st for st in styles_reading(target)
                         if st not in self._style_dead]
                if cands:
                    want = self.rng.choice(cands)
        self._gen = _GenWorker(self.planner.db, self.brain,
                               self.planner.library, self.rng,
                               want, used, tail,
                               probe_mode, baseline, probe_doc)
        self._gen.status.connect(self.status.setText)
        self._gen.ready.connect(self._seam_ready)
        self._gen.failed.connect(self._seam_failed)
        self._gen.start()

    def _seam_ready(self, seam):
        # ready/failed fire from INSIDE run() (queued): the thread can
        # still be winding down when this slot runs, and dropping the last
        # Python reference then destroys a live QThread - Qt qFatals the
        # whole app (0xc0000409 in Qt6Core, seen 2026-08-02). wait() is
        # microseconds here and closes the gap for good.
        self._gen.wait()
        self._gen = None
        self._recent.append(seam["a"].id)
        self._recent.append(seam["b"].id)   # B sides count as heard too
        if self._current is None:
            self._play(seam)
        else:
            self._next = seam            # waits for the current verdict
        self._pump()

    def _seam_failed(self, msg):
        self._gen.wait()                 # same lifetime story as _seam_ready
        self._gen = None
        self.status.setText(f"generation failed: {msg} - retrying")
        if self._session and self._current is None and self._next is None:
            self._pump()

    def _play(self, seam):
        self._current = seam
        plan, a, b = seam["plan"], seam["a"], seam["b"]
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
            f"{len(seam['audio']) / RATE:.0f}s render")
        self.probe = seam.get("probe")
        if self.probe:
            self.probe_lbl.setText(
                f"<b>Listen for one thing:</b> {self.probe['description']}"
                f" &nbsp;<span style='opacity:0.75'>({self.probe['knob']} "
                f"{self.probe['baseline']:g} → {self.probe['value']:g}"
                f", trial {self.probe['trials'] + 1})</span>")
            self.probe_lbl.show()
        else:
            self.probe_lbl.setVisible(False)
        self.strip.set_seam(a, b, seam.get("info"), seam["after_s"],
                            len(seam["audio"]) / RATE)
        self.planner.claim_playback("seamlab")
        self.player.load(seam["audio"])
        self.player.play()
        self._ended = False
        self._played_at = time.time()
        for b_ in self.rate_btns.values():
            b_.setEnabled(True)
        self.replay_btn.setEnabled(True)
        n = sum(self._tally.values())
        self.status.setText(f"playing - rate to advance ({n} rated this "
                            f"session)")

    def _replay(self):
        if self._current is None:
            return
        self.planner.claim_playback("seamlab")
        self.player.load(self._current["audio"])
        self.player.play()
        self._ended = False

    def _seek(self, t):
        """Click/drag on the scope - jump the audition there."""
        if self._current is None:
            return
        end = self.strip.window_end()
        if end is not None:
            t = min(t, max(end - 0.1, 0.0))
        self.planner.claim_playback("seamlab")
        self.player.seek(t)
        self.player.play()
        self._ended = False

    def _tick_playhead(self):
        if self._current is None:
            self.strip.set_playhead(None)
            return
        t = self.player.time_s()
        end = self.strip.window_end()
        # Stop where the scope stops: past the analysis region the render
        # is just the incoming track playing on, and letting it run buried
        # the verdict under a minute of unrelated music.
        if end is not None and t >= end:
            t = end
            if self.player.playing:
                self.player.pause()
            if not self._ended:
                self._ended = True
                self.status.setText(
                    "end of the analysed seam - R replays, or click the "
                    "scope to hear any part of it again")
        self.strip.set_playhead(t)

    # ---- what the ratings have taught -----------------------------------
    def _memory_brain(self):
        """Brain over the real library with cross-night memory loaded -
        the same read the live selection does. UI thread only (it touches
        the shared sqlite connection); the library is only walked once."""
        if self._mem_brain is None:
            self._mem_brain = Brain(self.planner.library,
                                    get_theme("groove"))
        self._mem_brain.load_pair_memory(self.planner.db)
        return self._mem_brain

    def _refresh_learning(self):
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
                w = r.get("want_style")
                if w:
                    attempts[w] = attempts.get(w, 0) + 1
            self._style_counts = counts
            # Retired or ungateable styles refuse every time (cut_at_drop
            # was retired 2026-08-02). Attempting them forever would burn
            # the balance rotation on seams that always fall back, so a
            # style that has been asked for repeatedly and NEVER landed
            # drops out of the rotation by itself.
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

    def _answer_probe(self, seam, verdict, listened):
        """Fold one parameter answer into the staircase, and apply the
        baseline move if the answer earned one."""
        probe = seam.get("probe")
        if probe is None or verdict == "skip":
            return
        from tools.dj.planner import seamprobe
        from tools.dj.planner.seamtune import RANGES
        from lib.dj import tuning
        from lib.dj.brain import TUNE_DEFAULTS
        try:
            doc, note = seamprobe.record(probe, verdict, RANGES)
            moves = seamprobe.pending_moves(doc)
            for knob, val in moves.items():
                tuning.set_value(knob, val,
                                 why=f"probe: {verdict} at {probe['value']:g}")
                seamprobe.reopen(doc, knob)
            if note:
                self.note_lbl.setText("→ " + note)
            elif moves:
                self.note_lbl.setText("→ baseline updated: " + ", ".join(
                    f"{k} → {v:g}" for k, v in moves.items()))
            else:
                self.note_lbl.setText(
                    f"→ noted ({probe['knob']}, "
                    f"{probe['trials'] + 1} answer"
                    f"{'s' if probe['trials'] else ''} so far)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status.setText(f"could not record the answer: {e}")

    # ---- rating ----------------------------------------------------------
    def _rate(self, slot):
        """`slot` is a button position; what it MEANS depends on the mode."""
        seam = self._current
        spec = _LABELS[self._mode()]
        if seam is None or slot not in spec:
            return
        _label, verdict, _tip = spec[slot]
        if verdict is None or not self.rate_btns[slot].isEnabled():
            return
        a, b, plan = seam["a"], seam["b"], seam["plan"]
        listened = self.player.time_s()
        self.player.pause()
        self._tally[verdict] = self._tally.get(verdict, 0) + 1
        if self._mode() == _MODE_PROBE:
            self._answer_probe(seam, verdict, listened)
        elif verdict != "skip":
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
                   # WHICH ENGINE judged this - so a later transition or
                   # knob change can be segmented instead of silently
                   # mixed into the same statistics.
                   "ver": engine_version(),
                   # The diagnosis fields: WHY a seam was the way it was.
                   # Logged per rating so the analysis panel can say what
                   # is failing and how, not just how often.
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
                   # The randomised execution nudge this seam carried -
                   # the causal half of the learning dataset.
                   "tune": plan.get("tune") or {},
                   "gate_test": (plan.get("diag") or {}).get("gate_test"),
                   "listened_s": round(listened, 1),
                   "render_s": round(len(seam["audio"]) / RATE, 1)}
            try:
                os.makedirs(os.path.dirname(_LOG), exist_ok=True)
                with open(_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                self.status.setText(f"could not log rating: {e}")
        if verdict in ("good", "bad") and self._mode() == _MODE_SEAM:
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
        self.strip.clear()
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
            self._play(nxt)
            self._pump()
        elif self._session:
            self.status.setText("rendering the next seam...")
            self._pump()
        else:
            self.status.setText("session paused")

    def stop_playback(self):
        self.player.close()
