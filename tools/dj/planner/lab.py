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

from lib.dj import gateprobe, stretch_engine_name
from lib.dj.gateprobe import gate_doc, gate_names, gate_styles
from lib.dj.version import engine_version
from lib.dj.brain import Brain
from lib.dj.themes import BUILTIN_THEMES, get_theme

RATE = 44100

_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "logs",
    "seam_lab_ratings.jsonl")
# Gate trials keep their OWN log, the one Gate Check wrote before it was
# folded away - 42 verdicts are already in it and a new trial has to be
# readable next to them, so the schema is theirs, not this tab's.
_GATE_LOG = os.path.join(os.path.dirname(_LOG), "gate_ratings.jsonl")

# Source of seams. "long blend" is Beat Check's old filter: retry until the
# brain itself plans a real overlapped blend on beat-heavy material, which
# is the only case where beat alignment can be judged at all.
_SRC_ANY = "any seam"
_SRC_LONG = "long blends only"
# The selection A/B source shows ONLY seams where the 2026-08-12 selection
# patches (anchor-true blendability mirror + exit-chain lookahead) changed
# which partner wins the pick. The plan gates are untouched by those
# patches, so every OTHER seam is identical to the old engine's - the
# changed picks are the entire audible surface of the change, and rating
# ~15 of them is a night's worth of validation in under an hour.
_SRC_SELAB = "selection A/B (changed picks)"
# The batch A/B source is the same idea aimed at the 2026-08-12/13 plan
# changes (deep entries, second-chance exits, the beat-power bar coming
# off the blends): it serves ONLY seams that exercise one of those
# surfaces - a seam yesterday's engine could not or would not have
# played. Everything else the batch left byte-identical, so these ARE
# its audible surface, pre-filtered so no listen is wasted on a seam
# both engines agree about.
_SRC_BATCH = "batch A/B (new-behavior seams)"
_SOURCES = (_SRC_ANY, _SRC_LONG, _SRC_SELAB, _SRC_BATCH)

# What a lab session may restrict itself to. "whole library" is the
# pre-2026-08-13 behavior; a saved setlist name runs the session the way
# a live night in POOL mode would (brain.pool_ids), which is how the
# uniform-pool claim ("progressive trance", 242 tracks, measured
# long_fade 72%->35%) gets listened to rather than only simulated.
_POOL_ALL = "whole library"


def _batch_surface(cand, plan):
    """Which 2026-08-12/13 surface this seam exercises, or None.

    Returns {"kind", "note"} - kind is the log/analysis key, note is the
    human line for the card. Ordering matters only for attribution when a
    seam exercises two surfaces at once (rare); the retry is checked
    first because it is the rarest and the most deliberate."""
    d = plan.get("diag") or {}
    if d.get("exit_retry"):
        r = d["exit_retry"]
        return {"kind": "exit_retry",
                "note": (f"second-chance exit: the first pick at "
                         f"{r.get('from_out_s', 0):.0f}s died to A's own "
                         f"exit (breakdown/collapse); retried at "
                         f"{r.get('to_out_s', 0):.0f}s")}
    if cand.duration_s > 0 \
            and plan.get("in_s", 0.0) >= 0.45 * cand.duration_s:
        f = plan["in_s"] / cand.duration_s
        return {"kind": "deep_entry",
                "note": (f"deep entry: B enters at {plan['in_s']:.0f}s "
                         f"({100 * f:.0f}% of the track) - unreachable "
                         f"before the 45% cap came off")}
    if plan.get("style") in ("long_blend", "bass_swap", "filter_sweep"):
        from lib.dj import beatpower as bp
        pw = bp.power_at(cand.id, plan.get("in_s", 0.0) + 10.0)
        if pw is not None and pw < bp.BLEND_MIN:
            return {"kind": "power_rescue",
                    "note": (f"bar-removal rescue: B's entry beat power "
                             f"{pw:.2f} < {bp.BLEND_MIN:.2f} - yesterday "
                             f"this seam was a forced fade")}
    return None

_BLEND_STYLES = ("long_blend", "bass_swap", "filter_sweep", "stem_bass_swap")

_VERDICTS = (("good", "Good (1)"), ("passable", "Passable (2)"),
             ("bad", "Bad (3)"), ("skip", "Skip (4)"))

# COVERAGE BALANCE (carried over from Seam Lab). Share of seams that get a
# style pinned toward the least-evidenced style. The rest are the brain's
# own choice - forcing a starved style onto an arbitrary pair rates WORSE
# than letting the brain pick (measured: 0.26 vs 0.35 mean verdict), so the
# lab buys evidence on some seams and stays representative on others.
_PIN_SHARE = 0.45
_STYLE_AUTO = "auto (coverage)"
_GATE_NONE = "none (rate seams)"
# Styles that claim to hit a STRUCTURAL moment in the incoming track. The
# claim is checkable, so the card checks it: a drop-anchored style whose
# cut lands 40 beats from any drop is landing somewhere arbitrary, and
# without this the only way to know was to trust the style's name.
_DROP_ANCHORED = ("cut_at_drop", "loop_build")


def _anchor_note(b, plan):
    """'· ON B's drop' / '· 12.0 beats before B's drop' for the styles that
    promise to land on one, '' for everything else."""
    if plan.get("style") not in _DROP_ANCHORED:
        return ""
    try:
        from lib.dj.features import drop_moments
        drops = drop_moments(b.sections)
        per = max(b.period_s, 1e-6)
        ahead = [d for d in drops if d >= plan["in_s"] - 2 * per]
        if not ahead:
            return "  ·  <b>no drop in B after the entry</b>"
        d = (min(ahead) - plan["in_s"]) / per
        where = ("lands <b>ON</b> B's drop" if abs(d) < 0.5
                 else f"lands <b>{d:+.1f} beats</b> from B's drop")
        # ...and HOW HARD it hits. A labelled drop is the segmenter's
        # opinion; this is the music's. Without it the card said "ON B's
        # drop" for seams that never dropped at all.
        from lib.dj.brain import drop_step
        st = plan.get("drop_step") or drop_step(b, plan["in_s"])
        if st:
            how = ("a slam" if st >= 3.0 else "a real drop" if st >= 2.0
                   else "only a lift" if st >= 1.15 else "NO audible step")
            where += f", ×{st:.2f} — {how}"
        return "  ·  " + where
    except Exception:
        return ""
_STYLES = sorted(set(get_theme("groove").style_weights) | {
    "stem_drum_swap", "acapella_out", "stem_bass_swap", "drum_bridge",
    "acapella_in", "melody_carry", "phrase_cut", "spinback_cut",
    "loop_in", "breakdown_swap"})


def _pre_patch_pick(brain, cur):
    """choose_next as the PRE-2026-08-12 selection ran it: the track-level
    blendability mirror and no exit-chain lookahead. A frozen reference,
    exitcompare-style - it must NOT track the live code, that is the point.
    Shadows the two methods on the instance and restores them in finally;
    plan gates are shared with the live engine, so a pick that comes back
    identical means the patches changed nothing for this seam."""
    from lib.dj import beatpower as _bp

    def _old_pb(self, cur_, cand, pair=None):
        bs_b = _bp.band_scores(cand.id, region="in") or {}
        ev_b = [v for v in (bs_b.get("low"), _bp.scores().get(cand.id))
                if v is not None]
        if ev_b and max(ev_b) < _bp.BLEND_MIN:
            return False
        bs_a = _bp.band_scores(cur_.id, region="out") or {}
        ev_a = [v for v in (bs_a.get("low"), _bp.scores().get(cur_.id))
                if v is not None]
        if ev_a and max(ev_a) < _bp.BLEND_MIN_EXIT:
            return False
        for t in (cur_, cand):
            if (t.bpm_conf or 0.0) < 0.7 \
                    and _bp.profile_coverage(t.id) < 0.6:
                return False
        ratio = cur_.bpm / max(cand.bpm, 1e-6)
        while ratio > 1.5:
            ratio /= 2.0
        while ratio < 0.67:
            ratio *= 2.0
        if not (0.945 <= ratio <= 1.058):
            return False
        if cur_.bpm / max(cand.bpm, 1e-6) > 1.5 \
                or cand.bpm / max(cur_.bpm, 1e-6) > 1.5:
            return False
        return True

    import types
    brain._pair_blendable = types.MethodType(_old_pb, brain)
    brain._exit_blendable = types.MethodType(lambda self, t: True, brain)
    try:
        return brain.choose_next(cur, 0.6, cur.bpm)
    finally:
        del brain.__dict__["_pair_blendable"]
        del brain.__dict__["_exit_blendable"]


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
                 used_ids, relaxed_ids, require_style=False,
                 trial_gate="", pool_ids=None):
        super().__init__()
        # Cooperative cancel (2026-08-15): a rare-gate search runs for
        # minutes, and Stop/gate-switch used to leave it grinding to
        # completion while the UI looked dead ("stopping and starting a
        # session doesn't work well, and its hard to switch between
        # gates" - operator, mid-rating). Checked between attempts and
        # before delivery; a cancelled worker exits fast and emits
        # nothing.
        self._cancelled = False
        self.db, self.brain, self.library = db, brain, library
        self.rng, self.want_style, self.source = rng, want_style, source
        self.used_ids = used_ids
        self.relaxed_ids = relaxed_ids
        # Session pool (a saved setlist's track ids), or None for the whole
        # library. The brain side is already set (brain.pool_ids) by the
        # tab; this copy narrows the A pick below.
        self.pool_ids = pool_ids
        # COVERAGE PIN vs AUDITION PIN. The coverage pin is an attempt at a
        # starved style and takes whatever the gates allow instead (see
        # run()). When the operator picks a style by hand they want THAT
        # seam, so the fallback is exactly the wrong answer - retry until
        # the pin lands, and say so plainly if it never does.
        self.require_style = bool(require_style and want_style)
        # GATE TRIAL: put one screen on trial by hearing what it refuses.
        # The seam must be one this gate actually killed, and the pin is
        # let through it (test_gates) so there is something to listen to.
        # Everything else still refuses normally - crossing one threshold
        # is the experiment, crossing all of them is just a broken engine.
        self.trial_gate = trial_gate or ""
        # What this worker is hunting - the tab drops any delivery whose
        # signature no longer matches the current controls (a render can
        # complete in the same instant the operator retargets).
        self.sig = (want_style, self.trial_gate, source)

    def cancel(self):
        self._cancelled = True

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
            # TARGETED SEARCH for a hand-picked style. Sampling the whole
            # library at random is fine for the workhorses and hopeless for
            # a rare one: breakdown_swap survives 0.5% of random pairs, so
            # a 300-try budget finds ~1 and each attempt costs a full
            # candidate scoring pass (measured 46s per seam found). Narrow
            # BOTH sides to tracks that can structurally serve the style -
            # the A pool directly, the B side through the brain's own
            # veto_ids so choose_next can only return a viable partner -
            # and the same style lands in 4.3s. No gate is relaxed; this
            # only stops the search wasting its budget on pairs that were
            # never going to pass. (Measured 1.8% vs 0.5%.)
            pool, style_veto = self.library, set()
            if self.require_style:
                from lib.dj.brain import audition_pools
                pool, style_veto = audition_pools(self.library,
                                                  self.want_style,
                                                  trial_gate=self.trial_gate)
                self.status.emit(
                    f"searching {len(pool)} usable A tracks x "
                    f"{len(self.library) - len(style_veto)} usable B "
                    f"for {self.want_style}...")
            if self.pool_ids is not None:
                pool = [t for t in pool if t.id in self.pool_ids]
                if not pool:
                    self.failed.emit("the selected pool has no usable "
                                     "A tracks for this search")
                    return
            ab_only = self.source == _SRC_SELAB
            batch_only = self.source == _SRC_BATCH
            for veto in (self.used_ids, self.relaxed_ids):
                self.brain.veto_ids = set(veto) | style_veto
                # A/B needs its own budget: only ~1 pick in a few is
                # patch-changed, and each attempt costs TWO full candidate
                # scoring passes.
                tries = 300 if (long_only or self.require_style) \
                    else (120 if (ab_only or batch_only) else 24)
                for attempt in range(tries):
                    if self._cancelled:
                        return
                    cur = self.rng.choice(pool)
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
                    sel_ab = None
                    if ab_only:
                        # SAME DICE, TWO RULEBOOKS. Both selectors roll from
                        # an identical rng state, so a differing winner is
                        # the patches' doing, never the explore jitter's.
                        _seed = self.rng.getrandbits(32)
                        self.brain.rng = random.Random(_seed)
                        cand, meta = self.brain.choose_next(
                            cur, 0.6, cur.bpm)
                        self.brain.rng = random.Random(_seed)
                        old_cand, old_meta = _pre_patch_pick(self.brain, cur)
                        if cand is None:
                            continue
                        if old_cand is not None and old_cand.id == cand.id:
                            if attempt % 10 == 0:
                                self.status.emit(
                                    f"pick unchanged by the patches, "
                                    f"repicking... ({attempt})")
                            continue      # nothing the patch touched
                    else:
                        cand, meta = self.brain.choose_next(
                            cur, 0.6, cur.bpm)
                        if cand is None:
                            continue
                    if long_only and (cand.duration_s > 480
                                      or bp.blendable(cand.id) is not True):
                        continue
                    self.brain.note_played(cand)
                    after_s = cur.duration_s * self.rng.uniform(0.35, 0.65)
                    if ab_only and old_cand is not None:
                        # What would the OLD night have played here? Same
                        # gates (the patches never touched them), same
                        # after_s - so the card can put the counterfactual
                        # next to what you are hearing.
                        try:
                            old_plan = self.brain.plan_transition(
                                cur, old_cand, old_meta, after_s=after_s)
                            _menu = ((old_plan.get("diag") or {})
                                     .get("menu") or {})
                            sel_ab = {
                                "old_b": old_cand.title,
                                "old_b_id": old_cand.id,
                                "old_style": old_plan["style"],
                                "old_fade_only": not [s for s in _menu
                                                      if s != "long_fade"]}
                        except Exception:
                            sel_ab = {"old_b": old_cand.title,
                                      "old_b_id": old_cand.id,
                                      "old_style": "?",
                                      "old_fade_only": None}
                    elif ab_only:
                        sel_ab = {"old_b": None, "old_b_id": None,
                                  "old_style": None, "old_fade_only": None}
                    plan = self.brain.plan_transition(
                        cur, cand, meta, after_s=after_s,
                        force_style=self.want_style,
                        # AUDITION BENCH: styles held off the live menu
                        # pending a listen are playable HERE and only
                        # here - that is what the Lab is for. Threshold
                        # gates are untouched (test_gates stays off), so
                        # a benched style still has to pass every screen
                        # the DJ would apply to it on the night.
                        allow_benched=True,
                        # ...unless a gate is explicitly ON TRIAL, which is
                        # the one sanctioned reason to cross a threshold.
                        test_gates=bool(self.trial_gate))
                    if long_only and not (plan["style"] in _BLEND_STYLES
                                          and int(plan.get("beats") or 0) >= 16):
                        continue          # not a real overlap - repick
                    if self.require_style and plan["style"] != self.want_style:
                        continue          # hand-picked style - repick
                    batch_surface = None
                    if batch_only:
                        batch_surface = _batch_surface(cand, plan)
                        if batch_surface is None:
                            if attempt % 10 == 0:
                                self.status.emit(
                                    f"seam identical under the old "
                                    f"engine, repicking... ({attempt})")
                            continue      # both engines agree - no signal
                    if self.trial_gate:
                        # A TRIAL SEAM IS ONE THIS GATE ACTUALLY REFUSED.
                        # brain only honours the pin when EVERY refusal is
                        # testable, and it stamps which ones it crossed in
                        # diag['gate_test'] - so requiring our gate to
                        # appear there is what makes the verdict evidence
                        # about THAT screen rather than about whichever
                        # pile-up happened to include it.
                        tested = ((plan.get("diag") or {}).get("gate_test")
                                  or "")
                        if self.trial_gate not in tested:
                            continue

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
                    if self._cancelled:
                        return           # retargeted while rendering
                    self.ready.emit({
                        "sig": self.sig,
                        "a": cur, "b": cand,
                        "plan": info.get("plan", plan),
                        "after_s": after_s, "audio": mix,
                        "decks": decks, "marks": marks,
                        "info": info, "bands": bands,
                        "want_style": self.want_style,
                        "trial_gate": self.trial_gate,
                        "sel_ab": sel_ab,
                        "batch_surface": batch_surface,
                        # What the screens MEASURED on this seam, so the
                        # card can say how far it missed the bar - a
                        # verdict without the number is unreadable later.
                        "gate_rows": (gateprobe.probe(cur, cand, plan)
                                      if self.trial_gate else [])})
                    return
            self.failed.emit(
                "no long overlapped blend found on beat-heavy material - "
                "switch to 'any seam'" if long_only else
                (f"no pair in {tries} tries let the brain plan "
                 f"{self.want_style} - its gates refused every one. Switch "
                 f"Style back to '{_STYLE_AUTO}', or check the Gate lens "
                 f"for which screen is doing it.") if self.require_style else
                (f"no new-behavior seam in {tries} tries - the batch "
                 f"surfaces are ~15-20% of seams, so this can happen on "
                 f"a small pool; try again or widen the pool")
                if batch_only else
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
        self._active_sig = None          # what the live worker hunts
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
        row.addWidget(QLabel("Pool:"))
        self.pool_box = QComboBox()
        self.pool_box.addItem(_POOL_ALL)
        try:
            for r in self.planner.db.conn.execute(
                    "SELECT name FROM setlists ORDER BY name"):
                self.pool_box.addItem(r["name"])
        except Exception:
            pass
        self.pool_box.setToolTip(
            "Restrict the session to a saved setlist, exactly like a live "
            "night in POOL mode (both decks come from inside it). 'whole "
            "library' is the old behavior.")
        self.pool_box.currentTextChanged.connect(self._reset_brain)
        row.addWidget(self.pool_box)
        row.addWidget(QLabel("Seams:"))
        self.source_box = QComboBox()
        self.source_box.addItems(_SOURCES)
        self.source_box.setToolTip(
            "'long blends only' retries until the brain plans a real "
            "overlapped blend on beat-heavy material - the only case where "
            "the Beat lens has anything to judge. 'selection A/B' shows "
            "ONLY seams where the 2026-08-12 selection patches changed "
            "which partner wins the pick - the entire audible surface of "
            "that change; the card names what the old engine would have "
            "played. Style/Gate boxes are ignored there.")
        row.addWidget(self.source_box)
        row.addWidget(QLabel("Style:"))
        self.style_box = QComboBox()
        self.style_box.addItems([_STYLE_AUTO] + list(_STYLES))
        self.style_box.setToolTip(
            "'auto' balances coverage toward the least-rated style and "
            "accepts whatever the gates allow instead when a pin is "
            "refused. Pick a style by name to audition THAT technique: "
            "the lab keeps repicking pairs until it plans one, so a "
            "rarely-legal style takes longer to find. Styles held off "
            "the live menu pending a listen are playable here.")
        row.addWidget(self.style_box)
        row.addWidget(QLabel("Gate:"))
        self.gate_box = QComboBox()
        self.gate_box.addItems([_GATE_NONE] + list(gate_names()))
        self.gate_box.setToolTip(
            "Put ONE screen on trial. The lab then only shows you seams "
            "that this gate refused, with the gate crossed so there is "
            "something to hear - every other gate still refuses normally. "
            "Rate 'bad' if the gate was right to refuse it; 'good' or "
            "'passable' means it was not. Verdicts append to "
            "logs/gate_ratings.jsonl beside the earlier 42.")
        self.gate_box.currentTextChanged.connect(self._retarget)
        self.style_box.currentTextChanged.connect(self._retarget)
        self.source_box.currentTextChanged.connect(self._retarget)
        row.addWidget(self.gate_box)
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

        # WHAT AM I RATING? The gate names are engine shorthand; the
        # person mid-listening shouldn't have to remember what
        # no_beat_power_B measures or which styles it still governs.
        # Text lives in gateprobe (with the names and bars) so it can
        # never describe a gate the engine no longer has.
        self.gate_doc_lbl = QLabel("")
        self.gate_doc_lbl.setWordWrap(True)
        self.gate_doc_lbl.setStyleSheet(
            "color: #9ab; font-size: 12px; padding: 2px 6px;")
        self.gate_doc_lbl.setVisible(False)
        v.addWidget(self.gate_doc_lbl)
        self._update_gate_doc()

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
            "deck A/out ▲ amber, deck B/in ▼ cyan · solid tick = measured "
            "kick, "
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

    def _pool_ids(self):
        """Track ids of the selected pool setlist, or None (whole library).
        Resolved fresh each pump - a setlist edited in another tab is
        picked up without restarting the session."""
        name = self.pool_box.currentText()
        if name == _POOL_ALL:
            return None
        try:
            from lib.dj import setlist as SL
            sl = SL.get_setlist(self.planner.db, name=name)
            ids = {e["track_id"] for e in (sl or {}).get("entries", [])}
            return ids or None
        except Exception:
            return None

    def _ensure_brain(self):
        if self.brain is None:
            self.brain = Brain(self.planner.library,
                               get_theme(self.theme_box.currentText()),
                               seed=self.rng.randrange(1 << 30))
        # Live pool-mode parity: candidate selection filters on pool_ids
        # exactly as DJSystem does when a setlist plays as a pool.
        self.brain.pool_ids = self._pool_ids()
        return self.brain

    # ---- session ---------------------------------------------------------
    def _toggle_session(self):
        self._session = not self._session
        self.start_btn.setText("■ Stop session" if self._session
                               else "▶ Start session")
        if self._session:
            self._pump()
        else:
            # Stop means STOP (2026-08-15): the in-flight search used to
            # grind on for minutes after this button, eating CPU and
            # blocking the next Start (pump refuses while a worker
            # lives). Cancel it and drop the prefetched seam - a fresh
            # Start hunts under whatever the controls say THEN.
            if self._worker is not None:
                self._worker.cancel()
            self._next = None
            self.status.setText("session stopped")

    def _retarget(self, *_):
        """Gate/style/source changed: abandon the old hunt and (if a
        session is running) start hunting the new target immediately.
        The old flow kept the previous target's search running to
        completion and then SHOWED its seam - switching gates felt
        broken because it was ('its hard to switch between gates')."""
        self._update_gate_doc()
        self._next = None                    # queued seam is stale
        if self._worker is not None:
            self._worker.cancel()            # _worker_done re-pumps
        elif self._session:
            self._pump()
        if self._session:
            self.status.setText("retargeting...")

    def _update_gate_doc(self):
        gate = self.gate_box.currentText()
        self.gate_doc_lbl.setText(
            "" if gate == _GATE_NONE else gate_doc(gate))
        self.gate_doc_lbl.setVisible(gate != _GATE_NONE)

    def _pump(self):
        """Keep exactly one render in flight and at most one queued."""
        if not self._session or self._worker is not None:
            return
        if self._next is not None and self._current is not None:
            return
        if not self.planner.library:
            self.status.setText("library is empty - scan first")
            return
        sel = self.style_box.currentText()
        gate = self.gate_box.currentText()
        gate = "" if gate == _GATE_NONE else gate
        require = sel != _STYLE_AUTO
        want = sel if require else self._starved_style()
        if self.source_box.currentText() in (_SRC_SELAB, _SRC_BATCH):
            # The A/B question is "does the night's OWN pick get better or
            # worse under the patches" - a style pin or gate trial would
            # answer a different question on the same audio. Brain's
            # choice, nothing crossed. (Same logic for the batch source:
            # its surfaces must arise naturally or the verdict is about
            # the pin, not the batch.)
            want, require, gate = None, False, ""
        if gate and not require:
            # A trial needs a PINNED style - brain only crosses a threshold
            # for a pin (there is no "let anything through" mode, by
            # design). Pin one the gate actually kills, else the trial
            # would rate seams this screen never touched.
            kills = [s for s in gate_styles(gate) if s in _STYLES]
            want = ("long_blend" if "long_blend" in kills
                    else (kills[0] if kills else want))
            require = bool(kills)
        brain = self._ensure_brain()
        self._worker = _SeamWorker(
            self.planner.db, brain, self.planner.library, self.rng, want,
            self.source_box.currentText(), set(self._used),
            set(self._recent[-12:]), require_style=require, trial_gate=gate,
            pool_ids=brain.pool_ids)
        self._active_sig = self._worker.sig
        self._worker.status.connect(self.status.setText)
        self._worker.ready.connect(self._seam_ready)
        self._worker.failed.connect(self._seam_failed)
        self._worker.finished.connect(self._worker_done)
        self._worker.start()

    def _worker_done(self):
        self._worker = None
        self._pump()                     # prefetch the next one

    def _seam_ready(self, seam):
        # A render can complete in the same instant the operator
        # retargets or stops - a seam built for the OLD controls must
        # never show under the new ones.
        if not self._session or seam.get("sig") != self._active_sig:
            return
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
        ab_note = ""
        ab = seam.get("sel_ab")
        if ab:
            if ab.get("old_b"):
                _of = (" - a forced fade" if ab.get("old_fade_only")
                       else "")
                ab_note = (
                    f"<br><span style='color:#50b0e0'>SELECTION A/B: the "
                    f"pre-patch engine would have played "
                    f"<b>{ab['old_b']}</b> via {ab['old_style']}{_of}."
                    f"</span>"
                    f"<br><span style='color:#9aa0a8'>You are hearing the "
                    f"NEW pick - rate it on its own merits.</span>")
            else:
                ab_note = ("<br><span style='color:#50b0e0'>SELECTION A/B: "
                           "the pre-patch engine found NO partner here."
                           "</span>")
        batch_note = ""
        bs = seam.get("batch_surface")
        if bs:
            batch_note = (
                f"<br><span style='color:#60c080'>BATCH A/B "
                f"({bs['kind']}): {bs['note']}</span>"
                f"<br><span style='color:#9aa0a8'>this seam only exists "
                f"under the 2026-08-12/13 changes - rate it on its own "
                f"merits.</span>")
        gate_note = ""
        if seam.get("trial_gate"):
            g = seam["trial_gate"]
            row_ = next((r for r in (seam.get("gate_rows") or [])
                         if r.get("gate") == g), {})
            gate_note = (
                f"<br><span style='color:#e0a030'>ON TRIAL: <b>{g}</b> "
                f"refused this seam</span>"
                f"<br><span style='color:#9aa0a8'>measured "
                f"{row_.get('detail', '?')} &nbsp;·&nbsp; bar "
                f"{row_.get('bar', '?')}</span>"
                f"<br><span style='color:#9aa0a8'>rate <b>bad</b> if the "
                f"gate was right to refuse it</span>")
        self.card.setText(
            f"<b>{a.title}</b> → <b>{b.title}</b><br>"
            f"{plan['style']} @ rate {plan['rate']:.3f}"
            f"{_anchor_note(b, plan)}{pin_note}{ab_note}{batch_note}"
            f"{gate_note}")
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
            # Retired or ungateable styles refuse every time (breakdown_swap
            # is benched; the loop family and the cut styles other than
            # cut_at_drop are retired). Attempting them forever would burn
            # the balance rotation on seams that always fall back, so a
            # style asked for repeatedly and NEVER landed drops out. Note
            # this only governs the AUTO rotation - a hand-picked style in
            # the Style box is always attempted.
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
                   # Selection A/B context, when this seam came from that
                   # source: what the pre-2026-08-12 selection would have
                   # played instead. Lets the analysis split "new picks"
                   # from the ordinary sample.
                   "sel_ab": seam.get("sel_ab"),
                   # Batch A/B context: WHICH 2026-08-12/13 surface this
                   # seam exercised (deep_entry / exit_retry /
                   # power_rescue), so per-surface tallies fall straight
                   # out of the log.
                   "batch_surface": (seam.get("batch_surface") or {}
                                     ).get("kind"),
                   "pool": (self.pool_box.currentText()
                            if self.pool_box.currentText() != _POOL_ALL
                            else None),
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
        # GATE TRIAL VERDICT, in the schema the 42 existing ones use, so a
        # new trial reads alongside the old ones. The question a trial
        # answers is not "is this seam good" but "was the screen RIGHT to
        # refuse it" - only `bad` vindicates the gate. `skip` is not a
        # verdict about the gate and is not logged.
        gate = seam.get("trial_gate")
        if gate and verdict != "skip":
            row = next((r for r in (seam.get("gate_rows") or [])
                        if r.get("gate") == gate), {})
            grec = {"t": time.time(), "gate": gate,
                    "gate_was_right": verdict == "bad",
                    "style": plan["style"], "beats": plan.get("beats"),
                    "rate": plan["rate"],
                    "a": a.title, "b": b.title, "a_id": a.id, "b_id": b.id,
                    "measured": row.get("detail"), "bar": row.get("bar"),
                    "also_tested": (plan.get("diag") or {}).get("gate_test"),
                    "solo": ((plan.get("diag") or {}).get("gate_test") or ""
                             ).strip() == gate,
                    "chips": [], "listened_s": round(listened, 1)}
            try:
                with open(_GATE_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(grec) + "\n")
            except Exception as e:
                self.status.setText(f"could not log gate verdict: {e}")
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
