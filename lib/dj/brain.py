"""The DJ brain: what to play next, and exactly how to get there.

Selection couples SONG choice to MIX quality: a candidate scores on tempo
fit (small stretch ratios preferred, half/double time considered), Camelot
key compatibility, energy fit to the theme's arc target, mood/spectral
match, recency penalties - AND the best available section-pair for the
transition. Two busy/vocal sections never blend over each other; if two
tracks have no quiet seam between them, that pairing loses points.

plan_transition() returns a fully-resolved plan (style, A's exit point, B's
entry point, blend length); build_events() compiles it into sample-stamped
submix automation using a telemetry snapshot for the clock mapping.
"""
import math
import random
import re
import time

import numpy as np

from lib.dj import stretch_engine_name
from lib.dj.features import _finite, drop_moments, hardness_raw
from lib.dj import tuning as _tuning
from lib.dj.rhythm import (prep_signature, rhythm_terms, seam_rhythm,
                           tempo_mult_for)
from lib.dj.themes import adapt_theme

RATE = 44100

# How much EARLINESS the play-time budget tolerates in best_pair, in
# seconds of exponential decay. The budget is a preference, not a filter
# (see best_pair): an exit this far below the drawn budget keeps ~37% of
# its score, so it still wins when the on-time alternatives are dead.
# Smaller = the budget's pacing wins more often and exits ride later;
# larger = exit quality wins more often and records play shorter.
BUDGET_TAU_S = 60.0
# ...and how much LATENESS it tolerates (2026-08-13). For a year the
# budget only penalised leaving EARLY: an exit past it scored a flat 1.0,
# so nothing anywhere bounded how late a seam could land. EXIT_MAX_FRAC
# and EXIT_HARD_MAX_FRAC bound the BUDGET, and the budget is only a floor
# - `after_s` refuses candidates below it and the scorer then took the
# best-fitting out point in the entire rest of the record.
#
# Measured on the night of 2026-08-13 (showman, 10 seams): the median exit
# landed at 0.90 of the record, 6 of 10 past the "absolute" 0.85 ceiling,
# and twice inside the outro - one at section energy 0.34 (operator: "it
# rolls into the fucking outro"). It also made play_len_x a DEAD LEVER:
# shortening a persona's budget only lowers the floor, which WIDENS the
# unpenalised late region, so showman rode records exactly as far as monk.
#
# Deliberately gentler than the early side (90s vs 60s): the early decay
# exists so a good exit a minute early can beat a dead one on time, and
# that argument runs both ways. The HARD fraction is the backstop the
# system's EXIT_HARD_MAX_FRAC always claimed to be - it now applies to
# the exit itself, not just to the budget that suggests one.
LATE_TAU_S = 90.0
EXIT_LATE_HARD_FRAC = 0.85
# Ceiling the drawn play budget may ask of a record's REMAINDER after its
# entry point. Lives HERE (not system.py, which imports it as
# EXIT_MAX_FRAC) for the same one-number reason as EXIT_LATE_HARD_FRAC:
# best_pair's entry-runway floor below and system._draw_exit's budget cap
# must agree, or the floor reasons about a cap that isn't the one applied.
EXIT_BUDGET_FRAC = 0.72
# THE SEAM WINDOW ITSELF MUST STAY ALIVE (2026-08-17). The fade-crater
# class - a third of census fades lurching or dying (Mukadderat ->
# 06_DEADLIFE 35dB / 0% floor, the Dunes set, Symmetry) - is an
# EXIT-ANCHOR defect, not a fade-shape one: the blessed v3 fade exposes
# A's own curve from ~8s before the boundary (recede start) to ~6s
# after (stop_lead), with B not fully present until ~7s past it. Two
# arm-time timing patches were tried and reverted (see
# docs/DJ_VERIFICATION.md) - the softness starts BEFORE the anchor, so
# only moving the anchor helps. best_pair's existing energy damp reads
# the SECTION MEAN, and section means hide exactly these holes (Dunes'
# winning anchor: section 0.52, curve minimum 0.01 six seconds later).
# So each exit anchor is also judged by its own 2 Hz curve across the
# exposure window, low-quantile against the track's groove body: an
# anchor whose window carries >= EXIT_LIFE_BODY_FRAC of body is
# untouched (healthy outros are quieter than the body ON PURPOSE - the
# golden rule wants them), a dying one decays toward EXIT_LIFE_FLOOR
# so a live anchor a phrase earlier can win. The statistic is the
# 1s-smoothed MINIMUM over the window (see _exit_life for why a
# quartile lost this argument twice on Dunes): the dead-air gate
# (rms_min_ratio) is a min, and a 1-2s hush notch a quantile smooths
# over IS the crater. The ratio is SQUARED before the damp because
# crater depth
# is heard in dB: a half-energy window is a -6dB dip (fine), a
# tenth-energy window is a -20dB hole (the Mukadderat crater) -
# linear-in-energy under-punished exactly the deep tails this exists
# for, and the measured margins showed it (dead anchors surviving by
# 2-16% against live ones).
EXIT_LIFE_PRE_S = 10.0
EXIT_LIFE_POST_S = 6.0
EXIT_LIFE_BODY_FRAC = 0.6
EXIT_LIFE_FLOOR = 0.2
# The ENTRY-side mirror: B's first ~20s after in_s is what carries the
# room alone once A stops (fade_stop_lead ~6s past the seam). Used only
# to decide whether a cramped clean entry deserves the roomy-veto
# RESCUE (see best_pair) - not as a global lean; full-room dead entries
# (the Condor-class B-gap) are a separate, undiagnosed lever.
ENTRY_LIFE_SPAN_S = 20.0
ENTRY_LIFE_RESCUE_MIN = 0.25

# GATE BARS, named so the Gate Check panel can show what a seam was judged
# against without re-typing the numbers (lib/dj/gateprobe.py reads these).
# Changing one here changes both the gate and what the panel reports.
# breakdown_swap's entry window: how far after B's build start its drop may
# sit and still count as "the build leads to it". 4 beats is the floor -
# closer than that and B enters essentially ON the drop, with nothing to
# ride. 40 beats is a blend (32) plus a phrase of grace.
_BDSWAP_DROP_MIN_BEATS = 4.0
_BDSWAP_DROP_MAX_BEATS = 40.0
# ...and the clearance the low/mid restore must keep from that drop, so the
# EQ move and the drop onset are heard as two events, not one slam.
_BDSWAP_RESTORE_CLEAR_BEATS = 4.0
# cut_at_drop: how hard B's drop must actually HIT, as an energy ratio
# across the entry on the dense curve. drop_moments() labels a boundary
# from SECTION MEANS (>0.25 jump, landing >=0.65 of peak) - that is the
# segmenter's opinion, not the music's. Measured over 1534 labelled drops
# on this library: median x1.57, and 13% have no audible step at all,
# which is why the style kept cutting into songs that never dropped
# (operator, 2026-08-12: "giving me songs that aren't dropping"). The
# style's whole premise is the slam, so it now requires a measured one.
# RATED BY EAR 2026-08-14 (25 cut_drop_shape Gate Check trials): the
# strict bars (step 1.8 / land 0.65 / run-up 0.55 / kick 1.15x) were
# wrong 20/25 - refused seams sounded fine at the ~20% background bad
# rate, and NO measured quantity separated the 5 bad from the 20 fine
# (bad step range 1.67-5.33 inside fine 1.40-6.72, bad KICK median 0.95
# with a fine seam at 0.06, bad entries off DEEPER dips). step solo:
# 0/5 right. Sixth per-track scalar family rated non-predictive. The
# bars moved to the FLOORS of the rated band - below them nothing was
# ever rendered, so there is no verdict to loosen on - and the kick
# kill came off entirely (rated across its whole range). Solo tallies
# were thin (3-5 per bar); operator acted on the joint count.
_CUT_DROP_MIN_STEP = 1.5
_CUT_DROP_WIN_BEATS = 8.0
# ...and RECALL: where we are allowed to look for one. The style used to
# enter only at a `pre_drop` MIX-IN hint, and mix-in points are proposed
# where you would bring a track IN - measured on this library, 61% of them
# sit in the first quarter of the song and none past three quarters, while
# the real >=x2.0 step-ups have a median position of 0.47. The door was in
# the wrong part of the building, which is why the style kept entering on
# a gentle intro build. B's own curve is scanned instead.
_CUT_DROP_SCAN_FROM = 20.0       # skip the intro proper
_CUT_DROP_RUNWAY_S = 60.0        # ...and leave this much of B left to ride
_CUT_DROP_SCAN_BEATS = 4.0       # scan resolution
# (step 2.0 -> 1.8 and runway 120 -> 60 on 2026-08-13, swept against the
# night's 17 labelled entries: at 2.0/120 only 37 of 982 tracks could serve
# as B with ONE usable entry each, so the style repeated the same few songs
# at the same moment - which reads as a bug long before it reads as rare.
# Loosening to 1.8/60 gives 63 tracks at identical verdict agreement. No
# setting in the sweep separated good from bad, so the strictness was
# buying nothing - and the 2026-08-14 ear rating above then showed the
# strictness was actively wrong.)
# SHAPE, not just size. A ratio cannot tell a drop from a rise - quiet ->
# mid scores the same as breakdown -> slam. These landed at 0.65/0.55 by
# construction; the 2026-08-14 rating (see _CUT_DROP_MIN_STEP) moved
# them to the floors of the heard band. The KICK-RETURN kill
# (power >= 1.15x the dip) is GONE from the same rating: it was the
# least-wrong bar (2/5 solo right) but still sorted nothing - a fine
# seam measured x0.06 and a bad one x0.95. drop_kick_levels() remains a
# MEASUREMENT (gateprobe row, cutdrop test) so the number stays next to
# every verdict; do not re-promote it to a kill without a new rating.
# (Its history is itself a lesson: the first read sampled power_at()
# +/-4 beats against a 20s-bucketed profile - below the instrument's
# resolution, ~1.75x implicit bar, 482 of 1923 candidates killed. Fixed
# to bucket-resolution dip->landing the same day the rating then
# retired the bar. Same defect class as the v1 beat-power trap.)
_CUT_DROP_MIN_AFTER = 0.50       # ...the drop must LAND hot
_CUT_DROP_MAX_BEFORE = 0.65      # ...off something genuinely down
# NEAR-MISS band for the Gate Check trial (`cut_drop_shape`): entries
# that clear these floors but fail >=1 strict bar above are cached
# separately so a pinned trial seam can be HEARD and the strict bars
# rated by ear. The floors keep the trial honest - below them the entry
# is unlikely to be a drop by any argument, so rendering it would rate
# little. The 2026-08-14 round rated the 1.5/0.50/0.65 band and the
# bars moved to its floors; these floors open the NEXT unheard band so
# the new bars stay testable ("a threshold nobody may cross can never
# be shown wrong").
_CUT_DROP_TRIAL_MIN_STEP = 1.25
_CUT_DROP_TRIAL_MIN_AFTER = 0.40
_CUT_DROP_TRIAL_MAX_BEFORE = 0.75
KICK_SCREEN_BLEND_S = 0.020   # overlapped-drum styles: max kick-placement
                              # delta, sits below the ~25ms audibility line
                              # on purpose (stored grids are themselves only
                              # ~25ms onset-accurate)
KICK_SCREEN_CUT_S = 0.028     # ...and for the short-dual cut/echo/loop tier
BAND_CLASH_HI = 1.5           # one side this rhythmic in a band...
BAND_CLASH_LO = 1.2           # ...against the other below this = a clash

STYLES = ("long_blend", "bass_swap", "cut_at_drop", "loop_roll_exit",
          "loop_build", "long_fade",
          "stem_drum_swap", "acapella_out")
# TEMPO WALL. Widened 0.92-1.08 -> 0.90-1.10 (2026-08-06, operator's
# call) on the strength of the beat-matching work: verified phase
# profiles, kick-true anchors and the sync bias now hold a lock that the
# 8% wall predates. 10% is not arbitrary - it is exactly where
# deck.set_rate already clamps (deck.py, np.clip(target, 0.90, 1.10)),
# so no plan can ask for a rate the deck would silently refuse, and the
# soak test's 0.90..1.101 assertion still holds.
# Reachable 1:1 pairs on the real library: ~35% -> ~40%.
# Note the wall is only the OUTER limit. Deep stretch is still leaned
# against by s_rate (Gaussian, sigma 0.045) and capped per side at 6% by
# rate_for. It was ALSO hard-gated at plan time by stretch>5.5%_risky
# until 2026-08-13, when 13 ear trials rated that gate right zero times
# and it came off - so this widening is finally reachable rather than
# nominal (it had been capping 72% of deep pairs at 5.5% for a week).
STRETCH_MIN, STRETCH_MAX = 0.90, 1.10
# Rate-gradient speeds. Both were tuned in the WSOLA era when a tempo ramp
# had NO pitch consequence; with the varispeed engine every gradient IS a
# pitch glide, and glides (not static offsets) are what ears catch (user:
# 'noticeable tonal shifts due to the speed shift gradients'). 0.0008/s =
# 1.4 cents/s - under the slow-drift noticing threshold on tonal material.
# 2026-07-22: A-ramp slowed 0.0015 -> 0.0008 (user: traditional practice;
# both ends of a seam now glide at the same proven-inaudible rate). The
# compiler's exit-skew clock model (setlist.compile_plan) shares THIS
# constant - a divergent hardcoded rate there drifted slot boundaries.
GLIDE_PER_S = 0.0008             # post-transition rate->1.0 glide speed
ARATE_RAMP_PER_S = 0.0008        # outgoing deck's pre-blend meet-tempo ramp
DEEP_ENTRIES = True              # entry candidates past the analyzer's 45%
                                 # cap (TrackInfo derives them from stored
                                 # sections at load; see the block there)


# --------------------------------------------------------------------------
# Track wrapper
# --------------------------------------------------------------------------

class TrackInfo:
    """One library track hydrated with sections/loops/mix points."""

    def __init__(self, row, sections, loops, mix_points, cues=None,
                 user_tags=None):
        self.row = row
        self.id = row["id"]
        self.path = row["path"]
        self.title = row.get("title") or row["path"]
        self.artist = row.get("artist") or ""
        self.duration_s = row["duration_s"] or 0.0
        self.bpm = row["bpm"] or 0.0
        self.bpm_conf = row["bpm_conf"] or 0.0
        self.downbeat_offset = row["downbeat_offset"] or 0
        self.downbeat_conf = row["downbeat_conf"] or 0.0
        self.camelot = row["camelot"] or ""
        self.grid = row.get("beat_grid") or []
        self.gain_db = row.get("loudness_gain_db") or 0.0
        self.kick_offset_s = float(row.get("kick_offset_s") or 0.0)
        self.phrase_beats = int(row.get("phrase_beats") or 0)
        self.phrase_start_s = float(row.get("phrase_start_s") or 0.0)
        self.phrase_conf = float(row.get("phrase_conf") or 0.0)
        self.mood_hist = row.get("mood_hist") or {}
        self.rhythm_density = row.get("rhythm_density") or 0.0
        self.spectral = row.get("spectral") or {}
        self.key_mode = row.get("key_mode")
        # 12-bin A-origin pitch-class profile (DB v12) - the continuous
        # harmonic fingerprint chroma_key_compat rotates by the planned
        # playback rate's true pitch shift. None until backfilled/rescanned.
        ch = row.get("chroma")
        self.chroma = ch if isinstance(ch, list) and len(ch) == 12 else None
        # ML structure segments (DB v12, allin1 pass): [[start_s, end_s,
        # label], ...] with labels intro/verse/chorus/bridge/inst/solo/
        # break/outro. Empty until the structure pass has run.
        st = row.get("structure") or {}
        self.ml_segments = st.get("segments") or []
        # Beat-sync rhythm signature (DB v13): step patterns + swing +
        # density, numpy-hydrated once here (seam scoring touches it per
        # candidate). None until scanned/backfilled - every consumer is
        # evidence-gated on that.
        self.rhythm_sig = prep_signature(row.get("rhythm"))
        # Character (danceability/valence) is library-ranked in load_library;
        # None until then (ghosts/tests fall back to raw via character.py).
        self.danceability = None
        self.valence = None
        self.arousal = None
        # Music2Emo ML descriptors when the mood pass has run (tracks.mood_ml):
        # normalized valence/arousal (0..1) + predicted mood tags. character.py
        # PREFERS ml_valence over its heuristic; moods fold into all_tags.
        mm = row.get("mood_ml") or {}
        self.ml_valence = mm.get("valence")
        self.ml_arousal = mm.get("arousal")
        self.ml_moods = mm.get("moods") or []
        self.sections = sections
        self.loops = loops
        self.axes = row.get("axes") or {}
        # Library-percentile overlay for the axes whose RAW values saturate
        # (hypnotic/hardness/energy) - filled by load_library; empty on
        # tracks built outside a library context (beatport previews).
        self.axes_rank = {}
        self.auto_tags = row.get("auto_tags") or []
        self.user_tags = list(user_tags or [])
        # MusicBrainz enrichment (genre/year/era), when present. Merged into
        # all_tags so flavor steering + the copilot see genres for free.
        enr = row.get("enrichment") or {}
        self.genres = enr.get("genres") or []
        self.year = enr.get("year")
        self.decade = enr.get("decade")
        self.enrichment = enr
        # Embedded container genre tag (free, from the file itself). Populated
        # on (re)scan; complements MusicBrainz genres for un-enriched tracks.
        self.file_genre = row.get("file_genre") or ""
        # Genre identity for the brain's coherence lean: MusicBrainz genres +
        # the embedded file genre ONLY (moods/user/character tags aren't
        # genres). Precomputed - score() touches this per candidate.
        gs = {g.strip().lower() for g in self.genres if g and g.strip()}
        for part in self.file_genre.replace("/", ",").replace(";", ",") \
                .split(","):
            part = part.strip().lower()
            if part:
                gs.add(part)
        self.genre_set = gs
        # 'do not use' flag (DB v11). Kept on the object so the library
        # browser can show + toggle it; callers that auto-select filter it out.
        self.excluded = bool(row.get("excluded"))
        # Pre-rendered stems on disk (tools/dj/dj_stems.py)? Stamped by
        # load_library (needs the music root); gates the stem styles.
        self.has_stems = False
        self.cues = list(cues or [])
        self.mix_ins = [p for p in mix_points if p["kind"] == "in"]
        self.mix_outs = [p for p in mix_points if p["kind"] == "out"]
        # USER-authored in/out cues override the analyzer's guesses: if any
        # exist for a direction, they become the only candidates (score 1.0,
        # so pair selection favors what the human marked).
        user_ins = [c for c in self.cues
                    if c["kind"] == "in" and c["source"] == "user"]
        user_outs = [c for c in self.cues
                     if c["kind"] == "out" and c["source"] == "user"]
        if user_ins:
            self.mix_ins = [{"kind": "in", "time_s": c["time_s"],
                             "score": 1.0, "style_hint": c.get("label")
                             or "blend"} for c in user_ins]
        if user_outs:
            self.mix_outs = [{"kind": "out", "time_s": c["time_s"],
                              "score": 1.0, "style_hint": c.get("label")
                              or "blend"} for c in user_outs]
        # DEEP ENTRIES (2026-08-12). find_mix_points only bookmarks "in"
        # candidates in the first 45% of a track, so the back half of
        # every record was unreachable as an entry BY CONSTRUCTION -
        # measured on this library that discards 39.4% of all
        # entry-quality material (543 sections of the same kind/vocal/
        # busyness profile and boundary strength as the eligible ones:
        # post-breakdown re-entries, second drops, late grooves). The
        # protections the cap stood in for all live in best_pair and are
        # better informed there: in_fit scores section kind + ML label,
        # vocals are walked point-accurate, and the earliness lean
        # (early_b, e^-(t-20)/120) already pays a deep entry down ~50%
        # at 3 minutes - so depth wins only when everything shallow is
        # genuinely worse. Same scoring formula as the analyzer's pass;
        # derived from stored sections at LOAD so no rescan is needed.
        # Guards: never past the outro, and >=90s of record must remain
        # (system._draw_exit budgets against the REMAINDER from entry).
        # The stored primary stays at [0] - beatpower's labeled regions,
        # the rhythm anchors and gateprobe's region_for all key on it.
        if DEEP_ENTRIES and not user_ins and self.duration_s > 0 \
                and self.sections:
            deep = []
            for i, s in enumerate(self.sections):
                t0 = s.get("start_s") or 0.0
                if i == 0 or t0 < self.duration_s * 0.45:
                    continue
                if s.get("kind") == "outro":
                    continue
                if self.duration_s - t0 < 90.0:
                    continue
                strength = max(s.get("boundary_strength") or 0.05, 0.05)
                quiet = 1.0 - 0.7 * (s.get("busyness") or 0.0)
                deep.append({"kind": "in", "time_s": round(t0, 3),
                             "score": round(strength * quiet, 3),
                             "style_hint": "blend", "deep": True})
            if deep and self.mix_ins:
                head, rest = self.mix_ins[0], self.mix_ins[1:]
                self.mix_ins = [head] + sorted(
                    rest + deep, key=lambda p: -p["score"])
            elif deep:
                self.mix_ins = sorted(deep, key=lambda p: -p["score"])

    @property
    def all_tags(self):
        # CACHED: selection scores every track and calls set(all_tags) several
        # times per candidate per horizon rebuild; recomputing this (character
        # tags, genre/mood folding) each access was a live-CPU cost that showed
        # as a visual slowdown when tag steering was on. Everything here is set
        # once at load except user_tags (mutated by _refresh_tags) and
        # danceability (set by rank_library) - so key the cache on those.
        key = (tuple(self.user_tags), self.danceability is not None)
        if getattr(self, "_all_tags_key", None) == key:
            return self._all_tags_cache
        extra = {g.lower() for g in self.genres}
        if self.decade:
            extra.add(self.decade)
        # embedded file genre (e.g. "Electronic", "Rock/Pop") -> split parts
        for part in self.file_genre.replace("/", ",").replace(";", ",").split(","):
            part = part.strip().lower()
            if part:
                extra.add(part)
        # ML mood tags (dark, party, melancholic, epic, ...) steer flavor,
        # copilot and search for free once the mood pass has run.
        extra.update(m.lower() for m in self.ml_moods)
        if self.danceability is not None:
            from lib.dj.character import character_tags
            extra.update(character_tags(self))
        result = sorted(set(self.auto_tags) | set(self.user_tags) | extra)
        self._all_tags_cache, self._all_tags_key = result, key
        return result

    @property
    def period_s(self):
        return 60.0 / self.bpm if self.bpm > 0 else 0.5

    def section_at(self, t):
        for s in self.sections:
            if s["start_s"] <= t < s["end_s"]:
                return s
        return self.sections[-1] if self.sections else None

    def ml_segment_at(self, t):
        """ML structure label at time t ('' when the pass hasn't run)."""
        for s, e, label in self.ml_segments:
            if s <= t < e:
                return label
        return ""

    def nearest_downbeat(self, t):
        """Downbeat time closest to t (from the main grid segment)."""
        g = None
        for seg in self.grid:
            if seg["start_s"] <= t <= seg["end_s"]:
                g = seg
                break
        if g is None and self.grid:
            g = self.grid[0]
        if g is None:
            return t
        bar = 4 * g["period_s"]
        first_down = g["first_beat_s"] + self.downbeat_offset * g["period_s"]
        k = round((t - first_down) / bar)
        return first_down + k * bar

    def nearest_phrase(self, t):
        """Nearest 16/32-beat PHRASE start to t; falls back to the nearest
        downbeat when the hypermeter wasn't confidently detected. Blends
        that start/end on phrase boundaries land where the MUSIC breathes,
        not just on a bar line."""
        # Gate calibrated on the real library: conf>=0.1 detections align
        # with section boundaries ~5x chance (0.39 vs 0.08); below that
        # it's noise. Worst case is a bar-snap - same as no phrase data.
        if self.phrase_beats <= 0 or self.phrase_conf < 0.1:
            return self.nearest_downbeat(t)
        span = self.phrase_beats * self.period_s
        k = round((t - self.phrase_start_s) / max(span, 1e-6))
        cand = self.phrase_start_s + k * span
        # Keep it inside the track and grid-honest.
        if cand < 0 or cand > self.duration_s:
            return self.nearest_downbeat(t)
        return self.nearest_downbeat(cand)

    def _raw_energy(self):
        """Absolute intensity from MEASURED loudness + sustain + rhythm.
        The old mood_hist/density/bass blend saturated on club material
        (everything read 0.5-0.72); integrated loudness and the 2 Hz curve
        were already stored but only used for level-matching. Loudness is
        the strongest separator (a quiet ambient piece and a slammed peak
        master differ by dB, not by mood buckets)."""
        # loudness_gain_db = gain that brings this track TO the target
        # (clipped +/-9): negative = hotter than target. Map to 0..1.
        loud = max(0.0, min(1.0, 0.5 - self.gain_db / 18.0))
        # SUSTAIN: mean of the per-track p95-normalized 2 Hz curve - a
        # track that sits near its own peak the whole time (steady club
        # groove) vs one that spends most bars far below it (long ambient
        # valleys, huge breakdowns). Neutral when unanalyzed (ghosts).
        curve = self.row.get("energy_curve") or []
        sustain = min(sum(curve) / len(curve), 1.0) if curve else 0.55
        dens = min(self.rhythm_density / 3.0, 1.0)
        mh = self.mood_hist
        mood_e = mh.get("peak", 0.0) + 0.6 * mh.get("groove", 0.0) \
            + 0.25 * mh.get("chill", 0.0)
        bass = self.spectral.get("bass_share", 0.33)
        return max(0.0, min(1.0, 0.34 * loud + 0.22 * dens + 0.16 * sustain
                            + 0.16 * mood_e + 0.12 * bass * 2.0))

    def energy_proxy(self):
        """Cross-track comparable energy 0..1. LIBRARY-RELATIVE when the
        track was loaded via load_library: the raw proxy clusters ~0.5-0.72
        on real libraries (mood_hist + density + bass all saturate), which
        washes out energy arcs and energy-based selection. load_library
        percentile-RANKS the raw values so they actually span 0..1 - a
        chill breakdown and a peak now sit at opposite ends. Falls back to
        raw for tracks built outside a library (ghosts, tests)."""
        r = getattr(self, "energy_rank", None)
        return r if r is not None else self._raw_energy()

    def _raw_drive(self):
        """Rhythmic drive from mood/density/bass ONLY - no loudness term.
        This feeds the LIVE visual energy (system.live_energy): the submix
        loudness-COMPENSATES mastering at playback, so a quietly-mastered
        banger sounds as loud as anything in the room - judging it by its
        master level dimmed the visuals on music that was audibly driving
        (user-heard, 2026-07-13). Selection/arcs keep the loudness-aware
        energy_proxy; the floor's lights follow the groove."""
        mh = self.mood_hist
        mood_e = mh.get("peak", 0.0) + 0.6 * mh.get("groove", 0.0) \
            + 0.25 * mh.get("chill", 0.0)
        dens = min(self.rhythm_density / 3.0, 1.0)
        bass = self.spectral.get("bass_share", 0.33)
        return max(0.0, min(1.0, 0.45 * mood_e + 0.35 * dens + 0.2 * bass * 2.0))

    def drive(self):
        """Library-ranked rhythmic drive 0..1 (see _raw_drive)."""
        r = getattr(self, "drive_rank", None)
        return r if r is not None else self._raw_drive()


# CROSSFADE LAW. A gain ramp this long or longer is a musical FADE and
# gets the constant-power curve; anything shorter is a cut, a de-click or
# a level set, and stays linear.
#
# Measured 2026-08-08 on real long blends: the two decks are uncorrelated
# in every band, |rho| < 0.06 - including the low band, because beat
# ALIGNMENT is not waveform CORRELATION. Uncorrelated signals add as
# power, so two linear ramps crossing at 0.5 amplitude land at
# 0.25 + 0.25 = -3.01 dB. The constant-power curve (Deck._ramp_gain)
# interpolates g^2 instead, so the pair holds its level across the cross.
#
# Applied as a post-pass over the finished event list rather than at each
# of the ~30 gain sites, so a new style cannot forget to opt in. An event
# that sets "curve" explicitly keeps what it asked for.
_POWER_RAMP_MIN_S = 1.0
# ...and ONLY for styles where the two decks genuinely overlap. A solo
# fade to silence (long_fade, echo_out, the cut styles) is heard on its
# own, where constant-power is the wrong shape: sqrt(1-u) sits 3 dB above
# linear at the halfway point and then drops away steeply, so a lone
# outro would hang and then fall off a cliff.
_POWER_STYLES = frozenset((
    "long_blend", "bass_swap", "filter_sweep", "stem_bass_swap",
    "stem_drum_swap", "drum_bridge", "breakdown_swap", "melody_carry",
    "acapella_in", "acapella_out", "loop_in"))


def _apply_fade_curve(ev, style):
    """Mark the long gain ramps of an OVERLAPPED blend as constant-power.

    Measured 2026-08-08: on the blends sampled this moved the mid-blend
    level by less than 0.2 dB, because the automation staggers the two
    fades rather than crossing them - deck A is already 7.5 dB down two
    seconds into an 11 s blend while B is still 15 dB below its final
    level. The curve is still the correct law for the stretch where they
    DO overlap; it is simply not what causes the mid-blend sag. That is a
    timing question between fade_out_ramp and fade_b_ramp1/2.
    """
    if style not in _POWER_STYLES:
        return
    for e in ev:
        if (e.get("cmd") == "gain" and "curve" not in e
                and float(e.get("ramp_s", 0.0)) >= _POWER_RAMP_MIN_S):
            e["curve"] = "power"


def load_library(db):
    """Hydrate every playable track from the DB into TrackInfo objects."""
    out = []
    for row in db.all_tracks():
        if not row.get("bpm") or row["bpm"] <= 0 or not row.get("duration_s"):
            continue
        if row["duration_s"] < 90.0:
            continue
        out.append(TrackInfo(row, db.sections_for(row["id"]),
                             db.loops_for(row["id"]),
                             db.mix_points_for(row["id"]),
                             cues=db.cues_for(row["id"]),
                             user_tags=db.tags_for(row["id"])))
    # LIBRARY-RELATIVE ENERGY: percentile-rank each track's raw energy so
    # the values span 0..1 (the raw proxy clusters ~0.5-0.72). This is what
    # makes energy arcs and energy-based selection discriminate.
    if len(out) >= 4:
        import bisect
        raws = sorted(t._raw_energy() for t in out)
        drvs = sorted(t._raw_drive() for t in out)
        denom = max(len(raws) - 1, 1)
        for t in out:
            t.energy_rank = bisect.bisect_left(raws, t._raw_energy()) / denom
            t.drive_rank = bisect.bisect_left(drvs, t._raw_drive()) / denom
        # FLAVOR-AXIS RANKING: hypnotic/hardness are stored RAW and
        # saturate on a real library (measured 2026-07-24: hardness mean
        # ~0.98, hypnotic ~0.88 across every theme's pool), which made
        # every axis_target on them a dead lever - gentle_organic's
        # hardness-0.2 target had nothing below ~0.9 to find, and
        # hypnotic_deep matched the whole library. Same cure energy got:
        # percentile-rank across THIS library, so a 0.95 target means
        # "top of this collection". Ranked values live in axes_rank;
        # vocal and speed stay raw (they carry absolute meaning: demucs
        # duration fraction / a fixed bpm mapping). "energy" rides the
        # existing energy_rank so axis_targets on it rank too.
        # MID-RANK on ties: bisect_left hands a tied blob the
        # first-occurrence rank, which for a degenerate axis is a
        # statistical lie ("everyone is soft"); ties belong mid-
        # distribution. Kept as the honest behavior for any axis that
        # still ties heavily.
        #
        # HARDNESS is recomputed from its INGREDIENTS rather than read
        # from the stored axis. features.hardness_raw is unbounded by
        # design, but libraries scanned before 2026-07-24 hold the old
        # 0..1-CLIPPED value, where 81% of a real 649-track library sat
        # at exactly 1.0 - ranking that blob gave four fifths of the
        # collection the identical rank 0.60, so hard_drive's
        # hardness-0.9 target and gentle_organic's hardness-0.2 target
        # resolved to the SAME score for most candidates. Recomputing
        # here means every existing library gets a live lever back
        # without re-analyzing a single file; a rescan just makes the
        # stored column agree.
        for t in out:
            t._hard_raw = hardness_raw(t.spectral, t.mood_hist,
                                       t.rhythm_density,
                                       _finite(t.axes.get("speed"), 0.5))
        # VOCAL ranks too. The raw axis is a demucs duration fraction with
        # a MEDIAN OF ZERO on a real library (57% of tracks are wholly
        # instrumental) and a p90 of 0.44, so vocal_journey's authored
        # `axis_targets={"vocal": 0.6}` sat above the 90th percentile -
        # nothing to find, and every instrumental scored nearly the same as
        # every singer. Ranked, a target of 0.9 means "the most sung
        # material this collection has". The RAW value stays what every
        # vocal-clash gate reads (_vocal_at, the acapella premise, the
        # persona pull): those need absolute presence, not a ranking.
        for axis, getter in (("hypnotic",
                              lambda t: _finite(t.axes.get("hypnotic"), 0.5)),
                             ("vocal",
                              lambda t: _finite(t.axes.get("vocal"), 0.0)),
                             ("hardness", lambda t: t._hard_raw)):
            vals = sorted(getter(t) for t in out)
            for t in out:
                v = getter(t)
                t.axes_rank[axis] = (bisect.bisect_left(vals, v)
                                     + bisect.bisect_right(vals, v) - 1) \
                    / 2.0 / denom
        for t in out:
            t.axes_rank["energy"] = t.energy_rank
    # DERIVED CHARACTER: danceability + valence, library-ranked (see
    # lib/dj/character.py). Adds mood/danceability the scanner never exposed.
    from lib.dj.character import rank_library
    rank_library(out)
    # STEMS ON DISK: one isdir/isfile sweep per load; gates the stem
    # transition styles (stem_drum_swap / acapella_out).
    try:
        from lib.dj.stems import has_stems
        root = getattr(db, "music_root", None)
        if root:
            for t in out:
                t.has_stems = has_stems(root, t.id)
                # ...and remember where to LOOK, so a session that
                # outlives a stem render can re-check (see
                # Brain._stems_refresh - 11 tracks rendered mid-session
                # on 2026-08-12 stayed False until restart).
                t._music_root = root
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Key compatibility (Camelot wheel)
# --------------------------------------------------------------------------

def _kick_delta_s(a, b, rate=1.0):
    """How far apart the two tracks' kicks sit against their grids AS THEY
    WILL SOUND, in seconds.

    `kick_offset_s` is folded bass placement measured at each track's
    NATURAL tempo, but B is stretched to meet A: b's offset occupies
    `kick_offset_s / rate` of wall time once it plays. Comparing the raw
    stored values (which every kick screen did until 2026-08-07) judges
    the pair at a tempo neither deck is playing - measured over 700 real
    seams it shifts the delta by a median 1.9ms (p95 11.1ms) and flips
    the 20ms screen's verdict on 6.6% of them, in both directions.

    Expressed in A's timebase, so it is independent of how the varispeed
    dual-bend splits the stretch across the decks: that split scales both
    sides by sqrt(rate) and cannot change which side is early. Under
    keylock (the rubberband engine) a_rate is 1.0 and this is exact.

    CIRCULAR (2026-08-07). kick_offset_s is a BEAT PHASE - features.
    measure_kick_offset folds the envelope onto ((t - first_beat) /
    period) % 1.0 and wraps the result to [-p/2, p/2). Subtracting two
    such phases linearly can report most of a beat for placements that are
    actually adjacent: measured on the real library, Visitors -> Activation
    read 317.6ms linearly and 133.7ms the short way round. Once B is
    stretched to A's tempo both share A's period, so the distance is
    modulo that."""
    ka = a.kick_offset_s or 0.0
    kb = (b.kick_offset_s or 0.0) / (rate or 1.0)
    d = abs(ka - kb)
    p = getattr(a, "period_s", 0.0) or 0.0
    if p > 0:
        d %= p
        d = min(d, p - d)
    return d


def camelot_compat(c1, c2):
    if not c1 or not c2:
        return 0.6                      # unknown keys: mildly neutral
    try:
        n1, m1 = int(c1[:-1]), c1[-1]
        n2, m2 = int(c2[:-1]), c2[-1]
    except ValueError:
        return 0.6
    dn = min((n1 - n2) % 12, (n2 - n1) % 12)
    if dn == 0:
        return 1.0 if m1 == m2 else 0.92    # same / relative major-minor
    if dn == 1:
        # Neighbour on the wheel; the diagonal (letter switch, e.g. 8A->9B)
        # shares the same 6-of-7 notes but relates the tonal centers more
        # loosely, so it prices between neighbour and the dn=2 tier.
        return 0.9 if m1 == m2 else 0.7
    if dn == 2 and m1 == m2:
        return 0.55
    return 0.3


def drop_step(track, at_s, beats=_CUT_DROP_WIN_BEATS):
    """How hard the music actually steps up across `at_s`, as an energy
    ratio on the dense curve (after/before, +/-`beats`). None when the
    track carries no curve.

    This is the INDEPENDENT check on a labelled drop: drop_moments() reads
    section means, so it answers "did the segmenter see a boundary", and
    this answers "does it hit". They disagree often enough to matter -
    13% of labelled drops measure no audible step at all."""
    curve = (track.row or {}).get("energy_curve") or []
    if not curve:
        return None
    w = max(beats * getattr(track, "period_s", 0.5), 4.0)
    i0, i1 = int(max(at_s - w, 0.0) * 2), min(int(at_s * 2), len(curve))
    j0, j1 = int(at_s * 2), min(int((at_s + w) * 2), len(curve))
    if i1 - i0 < 2 or j1 - j0 < 2:
        return None
    before = sum(curve[i0:i1]) / (i1 - i0)
    after = sum(curve[j0:j1]) / (j1 - j0)
    return after / max(before, 1e-4)


def drop_levels(track, at_s):
    """(before, after) energy across `at_s`, each as a fraction of the
    track's OWN p95 - the two quantities _CUT_DROP_MAX_BEFORE and
    _CUT_DROP_MIN_AFTER are compared against.

    Module-level and public for the same reason gateprobe imports its bars
    from the gates: a test that recomputes 'the same' number its own way
    disagrees in the last digit and reports a failure nobody can act on
    (measured: np.percentile vs a sorted index put a landing at 0.6499
    against a 0.65 bar the engine had passed).
    """
    curve = (track.row or {}).get("energy_curve") or []
    if not curve:
        return None
    peak = max(sorted(curve)[int(0.95 * (len(curve) - 1))], 1e-4)
    win = max(_CUT_DROP_WIN_BEATS * max(track.period_s, 0.3), 4.0)
    i0, i1 = int(max(at_s - win, 0.0) * 2), min(int(at_s * 2), len(curve))
    j0, j1 = int(at_s * 2), min(int((at_s + win) * 2), len(curve))
    if i1 - i0 < 2 or j1 - j0 < 2:
        return None
    return (sum(curve[i0:i1]) / (i1 - i0) / peak,
            sum(curve[j0:j1]) / (j1 - j0) / peak)


def drop_kick_levels(track_id, db_s):
    """(dip, landing) beat power around a drop downbeat, or None for
    either side that has no measurement. Module-level and public for the
    same reason drop_levels is: the cutdrop test and gateprobe must read
    the SAME windows the gate enforces, never recompute them.

    Read at the dense profile's OWN resolution (PROF_BUCKET_S buckets,
    linearly interpolated - see the note above _CUT_DROP_MIN_AFTER). The
    dip is the QUIETEST the kick got across the bucket before the drop,
    because that is what "the kick returns" returns from - a point read
    20s early lands on the tail of the previous groove and calls the
    return a wash. The landing is read half a bucket past the downbeat,
    the nearest position where the interpolated line is past the step."""
    from lib.dj import beatpower as _bp
    dips = [_bp.power_at(track_id, max(db_s - o, 0.0))
            for o in (5.0, 10.0, 15.0, _bp.PROF_BUCKET_S)]
    dips = [p for p in dips if p is not None]
    return (min(dips) if dips else None,
            _bp.power_at(track_id, db_s + _bp.PROF_BUCKET_S / 2))


def off_meter_span(track, lo_s, hi_s, tol=0.05):
    """True when any stored-grid SEGMENT overlapping [lo_s, hi_s] claims
    a bpm more than `tol` off the track's nominal meter.

    Stored grids are segmented, and the scanner can emit a segment
    through a breakdown that describes nothing (found 2026-08-14: a
    conf-0.99 track whose 220-240s segment claims 72.46 bpm at score
    0.25 between solid 107.9 segments). Every sync layer OBEYS the
    stored lattice - the PLL locks it, the phrase snap lands on it - so
    where the lattice is fiction, the decks hold a lock the music
    audibly walks away from ("both beatlines... well out of phase" -
    operator, on a hypnotic-heavy night; 38% of the library carries at
    least one such segment, median 11% of the track). 5% also catches
    half/double-time segments, whose bars are the wrong LENGTH.
    Same defect class the cut entry guard in _drop_entries covers;
    module-level so gateprobe/tests read the SAME rule."""
    for sg in (track.grid or []):
        if sg["end_s"] < lo_s or sg["start_s"] > hi_s:
            continue
        if abs(sg["bpm"] / max(track.bpm, 1e-6) - 1.0) > tol:
            return True
    return False


def audition_pools(library, style, trial_gate=""):
    """(a_pool, b_veto_ids) for auditioning `style`: the tracks that can
    STRUCTURALLY serve each side, so a search does not spend its whole
    budget on pairs the gates were always going to refuse.

    A SEARCH AID, NOT A GATE. It removes only pairs that could never pass,
    and every safety screen still runs on whatever survives - so a seam
    found this way is one the live engine would also have accepted.

    Measured on a 982-track library: breakdown_swap goes from 0.5% of
    random pairs (46s to find one) to 1.8% (4.3s), which is the difference
    between auditionable and not. The structural rules mirror
    plan_transition's, and read the SAME window constants the gate reads,
    so the two cannot drift apart.
    """
    lib = list(library or [])
    # NEVER PRE-FILTER ON THE GATE UNDER TRIAL. These pools encode the same
    # confidence bars some gates enforce, so narrowing by them while that
    # gate is on trial removes exactly the seams the trial exists to hear -
    # measured: a cut_needs_grid_conf>=0.8 trial found 0 seams in 300 tries
    # because the pool had already dropped every track under 0.8. The
    # structural clauses (has a breakdown, has a pre-drop entry) stay: no
    # threshold debate makes a missing section appear.
    _conf_bar = 0.0 if (trial_gate or "").startswith(
        ("cut_needs_grid_conf", "grid_conf")) else 1.0

    if style == "breakdown_swap":
        def _a(t):
            return (t.bpm_conf >= 0.7 * _conf_bar
                    and any(s["kind"] == "breakdown"
                            for s in (t.sections or [])))

        def _b(t):
            if t.bpm_conf < 0.7 * _conf_bar:
                return False
            per = max(t.period_s, 1e-6)
            drops = drop_moments(t.sections)
            lo, hi = (_BDSWAP_DROP_MIN_BEATS * per,
                      _BDSWAP_DROP_MAX_BEATS * per)
            return any(any(s["start_s"] + lo <= d <= s["start_s"] + hi
                           for d in drops)
                       for s in (t.sections or []) if s["kind"] == "build")
    elif style == "cut_at_drop":
        # Both sides carry the short-dual tier's grid bar (0.8 until
        # 2026-08-13, when the extra strictness was rated away), and B must
        # own a drop that measurably hits - the mix-in hint this used to
        # ask for lives in the intro, not where the drops are. The conf bar
        # drops out when it is itself on trial; the drop never does,
        # because no verdict conjures one. When the SHAPE bars are on
        # trial the scan floor drops to the trial tier for the same
        # reason the conf bar does: pre-filtering at the strict step bar
        # would remove exactly the near-miss tracks the trial exists to
        # hear.
        _step_bar = (_CUT_DROP_TRIAL_MIN_STEP
                     if (trial_gate or "").startswith("cut_drop_shape")
                     else _CUT_DROP_MIN_STEP)

        def _a(t):
            return t.bpm_conf >= 0.7 * _conf_bar

        def _b(t):
            if t.bpm_conf < 0.7 * _conf_bar:
                return False
            per = max(t.period_s, 0.3)
            at = max(_CUT_DROP_SCAN_FROM, 8 * per)
            stop = t.duration_s - _CUT_DROP_RUNWAY_S
            while at < stop:
                s = drop_step(t, at)
                if s is not None and s >= _step_bar:
                    return True
                at += _CUT_DROP_SCAN_BEATS * per
            return False
    else:
        _a = _b = None

    # TRIAL TARGETING is the INVERSE of the usual narrowing. Normally these
    # pools remove pairs that could never pass; a trial needs pairs that
    # will TRIP the gate, and the beat-power screens almost never get to be
    # the blocker on a random pair - the exit gates (a_exit_collapses,
    # a_exits_through_breakdown, unstable_phase_*) refuse first and are not
    # testable, so the override stands down. Measured: 0 trial seams in 300
    # tries for no_beat_power_A until the pool was biased this way.
    if (trial_gate or "").startswith("no_beat_power"):
        from lib.dj import beatpower as _bpm
        sc = _bpm.scores() or {}
        if trial_gate.endswith("_A"):
            def _a(t, _p=_a):                      # noqa: ANN001
                s = sc.get(t.id)
                return (s is not None and s < _bpm.BLEND_MIN_EXIT
                        and (_p is None or _p(t)))
        else:
            def _b(t, _p=_b):                      # noqa: ANN001
                s = sc.get(t.id)
                return (s is not None and s < _bpm.BLEND_MIN
                        and (_p is None or _p(t)))

    if _a is None and _b is None:
        return lib, set()
    # Never hand back an EMPTY A pool - a library with no usable A should
    # search badly, not crash the caller into an infinite retry.
    a_pool = ([t for t in lib if _a(t)] or lib) if _a else lib
    return a_pool, ({t.id for t in lib if not _b(t)} if _b else set())


def _shift_camelot(cam, semitones):
    """Camelot code after a pitch shift (+1 semitone = +7 on the wheel)."""
    try:
        num, letter = int(cam[:-1]), cam[-1]
    except (ValueError, IndexError):
        return cam
    return f"{((num - 1 + 7 * semitones) % 12) + 1}{letter}"


def _rot_chroma(p, semitones):
    """Rotate a 12-bin profile by a possibly-FRACTIONAL semitone shift
    (pitch up by s moves energy from bin i to bin i+s; fractional parts
    interpolate between adjacent bins)."""
    k = math.floor(semitones)
    f = semitones - k
    out = [0.0] * 12
    for i in range(12):
        v = p[i]
        out[(i + k) % 12] += v * (1.0 - f)
        out[(i + k + 1) % 12] += v * f
    return out


# Pearson-r -> score map, calibrated on Krumhansl-Schmuckler key
# templates so clean profiles reproduce the camelot_compat tiers:
# measured r on templates - same 1.0, relative 0.65, fifth 0.34,
# diagonals 0.24/0.54, two-steps -0.16, semitone -0.39, tritone -0.67.
# Piecewise-linear through those anchors: relative maps to 0.92, fifth
# 0.85, diagonals ~0.78-0.89, two-steps 0.55, semitone-and-worse 0.3 -
# and real (messier) profiles + fractional varispeed detunes land on the
# continuum in between instead of falling off an integer cliff.
_CHROMA_MAP = ((-0.35, 0.3), (-0.15, 0.55), (0.35, 0.85), (1.0, 1.0))


def chroma_key_compat(pa, pb, semitones=0.0):
    """Continuous harmonic compatibility of two 12-bin A-origin chroma
    profiles, with pb rotated by the TRUE sounded pitch offset between the
    decks in semitones (fractional under varispeed: 12*log2(rate); the
    keylock transpose's pitch_st otherwise). Pearson correlation of the
    profiles mapped onto camelot_compat's 0.3..1.0 scale. Returns None
    when either profile is missing/degenerate - callers keep the Camelot
    tier in that case."""
    if not pa or not pb or len(pa) != 12 or len(pb) != 12:
        return None
    if abs(semitones) > 1e-9:
        pb = _rot_chroma(pb, semitones)
    ma = sum(pa) / 12.0
    mb = sum(pb) / 12.0
    va = [x - ma for x in pa]
    vb = [x - mb for x in pb]
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na < 1e-9 or nb < 1e-9:
        return None                     # flat profile: no harmonic identity
    r = sum(x * y for x, y in zip(va, vb)) / (na * nb)
    if r <= _CHROMA_MAP[0][0]:
        return _CHROMA_MAP[0][1]
    for (r0, s0), (r1, s1) in zip(_CHROMA_MAP, _CHROMA_MAP[1:]):
        if r <= r1:
            return s0 + (s1 - s0) * (r - r0) / (r1 - r0)
    return _CHROMA_MAP[-1][1]


# --------------------------------------------------------------------------
# Brain
# --------------------------------------------------------------------------

_VERSION_WORDS = ("mix", "remix", "edit", "version", "extended", "original",
                  "radio", "club", "dub", "instrumental", "remaster",
                  "rework", "rmx", "bootleg", "vip")


def _title_root(title):
    """A song's identity with version/mix decorations stripped: 'Dunkel
    (Hobin Rude Remix)' and 'Dunkel (Original Mix)' -> 'dunkel'. Feat
    clauses go too. Empty when nothing is left (all-decoration titles)."""
    t = (title or "").strip().lower()
    t = re.sub(r"[(\[][^)\]]*\b(%s)\b[^)\]]*[)\]]" % "|".join(_VERSION_WORDS),
               " ", t)
    t = re.sub(r"\s*[-(\[]?\s*(feat\.|ft\.|featuring)\s.*$", " ", t)
    return re.sub(r"\s+", " ", t).strip(" -_")


# EVIDENCE WEIGHT for the broad memories (style, feature class). Without
# it a handful of votes moved a multiplier as far as a hundred would:
# measured 2026-08-02 on the real feedback table, phrase_cut sat at the
# 0.60 FLOOR on five votes while long_fade - the can't-beat-match FALLBACK
# - was the only boosted style. That is a rich-get-richer collapse: a
# penalised style gets chosen less, so it never earns the evidence that
# would clear it, and the night drifts toward wall-to-wall fades. Pull the
# raw multiplier back toward neutral in proportion to how much evidence
# stands behind it; a genuinely bad style still gets there, it just has to
# earn it. Pair memory is deliberately NOT shrunk - "these two exact songs
# clashed" is strong evidence from one hearing.
_SHRINK_K = 8.0

_COND_RT_MEMO = {}                  # (a_id, b_id) -> rhythm terms

# LEARNABLE EXECUTION KNOBS. Each was a hand-picked constant inside
# build_events; naming them lets the lab jitter one seam at a time and
# learn the direction each wants to move, WITHOUT repeating a pair - the
# jitter is independent of the music, so across enough seams it separates
# from it. Defaults reproduce the previous behaviour exactly.
TUNE_DEFAULTS = {
    "swap_pos": 0.5,        # where the swap sits between blend start/end
    "swap_beats_long": 6.0,  # crossfade width, staged long blend
    "swap_beats": 4.0,      # crossfade width, single-swap styles
    "b_mid0_hot": 0.3,      # B's entry mid shelf when A's mids are busy
    "b_mid0": 0.45,         # B's entry mid shelf otherwise
    "b_high0_hot": 0.7,     # B's entry high shelf when A's highs are busy
    "b_high0": 1.0,         # B's entry high shelf otherwise
    "b_high0_long": 0.5,    # ...capped this low on a staged long blend
    "b_mid0_long": 0.3,
    "trim_cap": 1.41,       # quiet-intro entry trim ceiling (+3 dB)
    # Fade shape: v3, the user's own. One evening (2026-08-05) churned it
    # through "sustained" (v4) and "tight cross" (v5), each validated on
    # two seams and a pair of invented metrics; v5 reproduced the fast
    # handoff the user had ALREADY rejected in July ("songs slamming into
    # each other") and drew "you have utterly ruined the long fade".
    # Everything from that evening is reverted EXCEPT two deltas the user
    # asked for by name: the kick-carve ("the kick clash is terrible")
    # and urgent compression on skips. Verified 2026-08-06 by diffing
    # this block against 98e92ef: for a non-urgent fade the only
    # behavioral difference from the known-good week-ago version is the
    # kick-carve.
    #
    # DO NOT "improve" the overlap again without the user asking. Three
    # separate attempts made it audibly worse, and the measurements that
    # justified them were wrong: a 2026-08-06 study of 40 week-ago fades
    # vs 40 current ones found the SAFE version had MORE rhythmic
    # collision (density 7.4 vs 3.7, 90% both-sides-live vs 85%). What
    # the ear objects to in a fade is not measured by any instrument in
    # this repo yet - so changes here need ears, not numbers.
    "fade_recede": 0.5,     # long_fade: level A recedes to
    "fade_lead_a": 8.0,     # long_fade: A starts receding this early
    "fade_lead_b": 4.0,     # long_fade: B arrives this early
    "fade_b_stage1": 0.6,   # long_fade: B's "present" level before full
    "fade_b_ramp1": 3.5,    # long_fade: seconds to reach it
    "fade_b_ramp2": 8.0,    # long_fade: seconds from there to full
    "fade_out_ramp": 5.0,   # long_fade: A's final fade length
    "fade_stop_lead": 6.0,  # long_fade: A stops this long after the seam
    # ONE KICK AT A TIME (2026-08-06). Both knobs carve the OUTGOING
    # track, which is what makes them safe: A is leaving and has already
    # had its full presence, while B's identity must arrive whole and
    # B's quiet entry is what masks the mismatch (see the regime note in
    # build_events). Neither moves a gain event.
    # The low-band baton pass, as two independently tunable halves. They
    # share a default so the handover is complementary out of the box;
    # separate knobs let the lab explore an asymmetric one (a gentler B
    # arrival, say) and let a harness neutralize one side alone.
    "fade_a_low_out": 1.2,  # long_fade: A's low leaves this fast
    "fade_b_low_in": 1.2,   # long_fade: ...and B's arrives this fast
    # ONE BEAT PATTERN AT A TIME (2026-08-14). The baton above hands
    # over the KICK, but a fade between two rhythm-dense tracks still
    # plays both PERCUSSION lines - snares, toms, mid-hats live in A's
    # mids, which deliberately carry until the exit fade completes -
    # free-running through each other for the dual ("both beatlines
    # over each other, well out of phase" - operator, hypnotic-heavy
    # night where nearly every fade pair measured rhythm-dense with
    # predicted kick_agreement 0.1-0.5: the planner PREDICTED each
    # clash and the fade path never consumed the number). When the
    # seam's own rhythm prediction says the patterns fight
    # (kick_agreement < 0.6 at conf >= 0.5 - evidence-gated, no
    # measurement means no carve), A's mids leave on their own clock
    # from the seam instead of riding the gain fade. Slower than the
    # low baton on purpose: mids also carry A's melody, and this is a
    # fade-to-darkness, not a mute. Clash pairs only - a fade against
    # an ambient B keeps A's melody to the end, exactly as before.
    "fade_a_mid_out": 2.5,  # long_fade: A's mids leave this fast on clash
    # The carve TRIGGER is rhythm DENSITY on both sides, not
    # kick_agreement (2026-08-14, second revision same day). The first
    # trigger was kick_agreement < 0.6 - "the patterns will fight" -
    # and it was rated wrong within hours: As the Day Rises ->
    # Pulsacions measured ka 0.991, carve stood down, and the operator
    # heard "really bad overlapped beats". On an UNSYNCED fade, pattern
    # agreement is not safety - two near-identical kick/perc lines at
    # 122.9 vs 120.9 bpm drift through EVERY phase relationship during
    # the dual, and identical-patterns-phasing is the classic
    # trainwreck, worse than different patterns interleaving. The real
    # risk factor is simply two dense rhythm beds coexisting, whatever
    # their patterns. Density is per-track and stored, so this also
    # closes the loose-grid hole the ka trigger had (grid_conf<0.5
    # fades carried untrusted rhythm predictions and got no carve -
    # exactly the fades most likely to need one). At this library's
    # distribution (p5 0.93, median 1.75) the floor of 1.0 exempts
    # only genuinely sparse/ambient sides - the baton is the NORM for
    # dance-material fades now, which is what three ear reports in one
    # day said it should be.
    "fade_clash_density": 1.0,  # long_fade: carve when BOTH sides'
                                # rhythm_density >= this (set high to
                                # disable the clash carve)
    "fade_clash_lead_x": 0.5,  # long_fade: clash pairs shrink B's lead
                               # (co-presence is TIME x depth; EQ can
                               # only touch half the percussion band)
    # THE CARVE GOES DEEPER WHEN B LANDS HOT (2026-08-15). The deepest
    # settings measured (lead 0.25 / mid 1.2: co-presence 0.5-1.25s on
    # the reported trainwreck pairs) failed the render gate as BLANKET
    # defaults - a late B on a quiet entry leaves the room empty (dead
    # air + lurch on the conf<0.45 fade population). The split is B's
    # ENTRY HEAT vs its own body (gain comp makes that ~room-relative):
    # measured, dense clash entries run 0.77-0.90 while the dead-air
    # cases measure 0.63-0.70 - so above the 0.75 bar the entry can
    # carry a tight handover and the carve uses the deep tier; below
    # it, the gentle tier that every fade population survives.
    "fade_clash_hot_heat": 0.75,   # entry-heat bar for the deep tier
    "fade_clash_lead_hot_x": 0.25,  # deep tier: B's lead shrink
    "fade_a_mid_out_hot": 1.2,      # deep tier: A's mids leave this fast
    # A's air once B is in the room. DEFAULT 1.0 = OFF, and the default
    # is a measurement, not caution: at 0.6 it moved transient
    # co-presence by nothing on average (4.31 -> 4.25s over 4 pairs) and
    # made it WORSE on one (6.25 -> 7.25s). Trimming the louder deck 4 dB
    # does not move its transients out of the way, it moves them DOWN
    # TOWARD the other deck's, so more of the overlap sits inside the
    # ~12 dB window where both kits are audible. Killing transient clash
    # needs a decisive cut, which costs A's character. Left as a knob for
    # the lab to disprove me with ears.
    "fade_a_high": 1.0,     # long_fade: A's air, once B is in the room
    "stage1_gain": 0.92,    # staged blend: B's stage-1 level (x entry trim)
    "stage1_frac": 0.35,    # staged blend: fraction of the span to reach it
    "high_swap_at": 0.22,   # staged blend: when the highs start migrating
    "beats_scale": 1.0,     # global multiplier on the planned blend length
    "pre_dip_at": 0.5,      # staged blend: when A starts its glide down
    "pre_dip_gain": 0.85,   # ...and the level it glides to
    "exit_res": 8.0,        # beats of A reserved after the swap
    "exit_res_long": 16.0,
    "duck_depth": 0.0,      # vocal duck: A's vocal stem level
    "duck_beats": 2.0,      # ...over this many beats
    "echo_lead_beats": 12.0,   # echo_out: B arrives this many beats early
    "echo_b_gain": 0.9,     # echo_out: B's level under the tail
    "echo_delay_beats": 0.75,  # echo_out: delay time (dotted eighth)
    "echo_feedback": 0.62,
    "echo_wet": 0.8,
    "echo_tail_s": 2.5,     # echo_out: how long the tail rings on
    "spinback_s": 1.4,      # spinback_cut: platter wind-down length
    "brake_s": 0.9,         # phrase_cut's optional brake, when it fires
    "brake_chance": 0.5,    # ...how often it fires
    "roll_shrink1": 16.0,   # loop_roll_exit: beats before the first halving
    "roll_shrink2": 24.0,   # ...and the second
}


def _shrink(v_raw, votes, lo, hi):
    """Evidence-proportional multiplier: neutral at no votes, approaching
    the raw value as votes accumulate past _SHRINK_K."""
    n = max(float(votes), 0.0)
    return max(lo, min(hi, 1.0 + (v_raw - 1.0) * n / (n + _SHRINK_K)))


def seam_conditions(a, b, rt=None):
    """The handful of conditions a style's competence is learned PER.

    This is what turns "phrase_cut is bad" (which can only ever delete the
    style) into "phrase_cut is bad on LOOSE GRIDS" (which tells the engine
    when to reach for it and tells a human what to fix). Deliberately
    coarse - each bucket has to collect enough votes to mean something
    within a few sessions."""
    out = []
    conf = min(getattr(a, "bpm_conf", 0.0) or 0.0,
               getattr(b, "bpm_conf", 0.0) or 0.0)
    out.append("grid:precise" if conf >= 0.7 else "grid:loose")
    kc = camelot_compat(getattr(a, "camelot", None),
                        getattr(b, "camelot", None))
    out.append("key:good" if kc >= 0.9 else
               "key:ok" if kc >= 0.55 else "key:clash")
    if rt:
        s = rt.get("score")
        if s is not None:
            out.append("groove:lock" if s >= 0.6 else
                       "groove:half" if s >= 0.45 else "groove:fight")
        f = rt.get("flam_ms")
        if f is not None:
            out.append("flam:lock" if f < 15.0 else
                       "flam:risk" if f <= 80.0 else "flam:far")
    return tuple(out)


class Brain:
    def __init__(self, library, theme, seed=None, stretch_max=STRETCH_MAX):
        self.library = list(library)
        # ML MOOD STEERING kicks in ONLY when most of the library is mood-
        # scored (lib/dj/mood_ml). On a partially-scored library the few
        # scored tracks would be judged on valence/danceability while the
        # unscored majority stay neutral - an asymmetry that measurably
        # demoted good picks. All-or-(mostly)-nothing avoids that.
        scored = sum(1 for t in self.library
                     if getattr(t, "ml_valence", None) is not None)
        self._use_mood = bool(self.library) and scored >= 0.8 * len(self.library)
        # HARD tag filter (live panel "only these tags play"): a candidate must
        # carry at least one of these tags to be eligible. Empty = no filter.
        # Independent of pool_ids (setlist) and flavor (soft lean), so all
        # three compose (steer WITHIN a pool, within the required tags).
        self.require_tags = set()
        # STYLE PACING: the last few chosen transition styles (anti-streak -
        # the same style twice in a row is halved, three times zeroed), and
        # the clock of the last engineered MOMENT (spectacle-tier seam at an
        # arc peak; cooldown keeps them landmarks, not wallpaper).
        self.recent_styles = []
        self.last_moment_t = 0.0
        # CONTENT IDENTITY: the real library holds dozens of byte-identical
        # copies under different track ids (plus re-rips). For recency and
        # queue purposes a copy IS the song - per-id memory let 'the same
        # song' reappear via its twin (user-seen in the next-3 queue).
        def _ck(t):
            # VERSION FAMILIES: 'Dunkel (Original Mix)' and 'Dunkel (Hobin
            # Rude Remix)' are the same SONG to a listener - hash/title keys
            # left them unlinked and they played back-to-back (2026-07-12
            # night log). Key on the stripped title root when there is one;
            # hash only rescues untitled files.
            root = _title_root(t.title)
            artist = (t.artist or "").strip().lower()
            if root:
                return "m:" + root + "|" + artist
            h = (t.row.get("content_hash") or "").strip()
            if h:
                return "h:" + h
            return ("m:" + (t.title or "").strip().lower() + "|" + artist)
        self.ckey = {t.id: _ck(t) for t in library}
        self._lib_bpms = [t.bpm for t in self.library]
        self.theme = adapt_theme(theme, self._lib_bpms)
        if self.theme.bpm_range != theme.bpm_range:
            print(f"[DJ] theme '{theme.name}' tempo window fitted to this "
                  f"library: {theme.bpm_range[0]:.0f}-{theme.bpm_range[1]:.0f}"
                  f" -> {self.theme.bpm_range[0]:.0f}-"
                  f"{self.theme.bpm_range[1]:.0f} bpm")
        self.rng = random.Random(seed)
        # PERSONA: the night's mixing temperament (lib/dj/persona). Neutral
        # is all-identity - every persona read below must reduce to exactly
        # the pre-persona arithmetic at the defaults.
        from lib.dj.persona import NEUTRAL
        self.persona = NEUTRAL
        self.stretch_max = min(stretch_max, STRETCH_MAX)
        self.stretch_min = max(2.0 - self.stretch_max, STRETCH_MIN)
        self.recent = []                # (wall_time, track_id, artist)
        self._ds_cache = {}             # ckey -> distinct songs since last play
        self._ds_cache_key = None
        # NO-REPEAT DEPTH: how many DISTINCT other songs must play before one
        # can return. Scales with the library so a big collection never
        # replays soon, while a small pool just round-robins its members.
        self.norepeat_n = max(6, min(len(self.library) // 4, 35))
        # SETLIST POOL: when set, selection is confined to these track ids
        # (the operator's list as a POOL - the brain steers the order via
        # arc/flavor/nudge). System drains ids as they play; None = free.
        self.pool_ids = None
        self._recent_skips = {}         # track_id -> skip count (tonight)
        # LIVE FLAVOR overrides: same shape as the theme's flavor fields,
        # set from the web/planner mid-set ("more hypnotic", "no vocals").
        # Merged OVER the theme - the theme is the default, the operator
        # is the boss.
        self.flavor = {}                # {prefer_tags, avoid_tags, axis_targets}
        self.veto_ids = set()           # transient 'not this one' (reroll)
        self.last_scored_n = None       # candidates the last pick drew from
        self.style_fb = {}              # style -> tonight multiplier (thumbs)
        self.pair_memory = {}           # (a_id,b_id) -> cross-night multiplier
        self.class_memory = {}          # (key,off,conf) bucket -> multiplier
        self.style_memory = {}          # style -> cross-night multiplier
        self.style_cond_memory = {}     # (style, condition) -> multiplier
        self._rhythm_cache = {}         # (a_id,b_id,mult) -> rhythm score
        self._drop_entry_cache = {}     # track id -> [(downbeat_s, step)]
        self._drop_near_cache = {}      # track id -> [(db_s, step, tags)]

    @staticmethod
    def _pair_class(a, b):
        """Feature-space bucket a seam belongs to: key fit x groove-offset
        gap x grid confidence. This is how one night's feedback GENERALIZES -
        the exact A->B pair rarely recurs on a big library, but 'clash-key
        wide-offset loose-grid seams keep flamming' is a lesson every future
        pick can use."""
        kc = camelot_compat(a.camelot, b.camelot)
        key = "good" if kc >= 0.9 else ("ok" if kc >= 0.55 else "clash")
        d_off = abs(a.kick_offset_s - b.kick_offset_s)
        off = "tight" if d_off <= 0.035 else ("mid" if d_off <= 0.09
                                              else "wide")
        conf = "precise" if min(a.bpm_conf or 0.0,
                                b.bpm_conf or 0.0) >= 0.7 else "loose"
        # Groove bucket from the rhythm signatures - cheap scalars only
        # (this runs per candidate per pick), the full pattern math stays
        # in _rhythm_fit. Lets one measured swing-clash train-wreck teach
        # the brain about the whole class of swung-vs-straight pairings.
        sa = getattr(a, "rhythm_sig", None)
        sb = getattr(b, "rhythm_sig", None)
        if sa is None or sb is None:
            groove = "unmeasured"
        else:
            sd = abs(sa.get("swing", 0.5) - sb.get("swing", 0.5)) \
                * min(sa.get("swing_conf", 0.0), sb.get("swing_conf", 0.0))
            groove = "swingclash" if sd > 0.055 else "swingok"
        return (key, off, conf, groove)

    def load_pair_memory(self, db, days=90.0):
        """CROSS-NIGHT TASTE: thumbs on seams and bail-out skips persist
        as pair-level multipliers - a seam that worked last Saturday gets
        a lasting bonus, one that got skipped carries a lasting caution.
        Bounded 0.4..1.6: memory is a lean, never a law.

        The same feedback also aggregates into CLASS memory (feature
        buckets, see _pair_class) and STYLE memory (cross-night style
        multipliers applied in plan_transition) - both bounded tighter
        than pair memory because they apply broadly, and both gated on
        >=3 weighted votes so one odd night can't tilt a whole class."""
        fb, skips = db.pair_stats(days=days)
        mem = {}
        for k, (ups, downs) in fb.items():
            mem[k] = (1.15 ** min(ups, 4)) * (0.75 ** min(downs, 4))
        for k, n in skips.items():
            mem[k] = mem.get(k, 1.0) * (0.85 ** min(n, 3))
        self.pair_memory = {k: max(0.4, min(1.6, v))
                            for k, v in mem.items() if abs(v - 1.0) > 0.01}
        by_id = {t.id: t for t in self.library}
        cls, sty = {}, {}
        try:
            rows = db.seam_feedback_rows(days=days)
        except Exception:
            rows = []
        for r in rows:
            w = 0.5 if (r["source"] or "user") == "auto" else 1.0
            up = 1 if r["up"] else 0
            if r.get("style"):
                u, d = sty.get(r["style"], (0.0, 0.0))
                sty[r["style"]] = (u + w * up, d + w * (1 - up))
            a, b = by_id.get(r["a_id"]), by_id.get(r["b_id"])
            if a is not None and b is not None:
                k = self._pair_class(a, b)
                u, d = cls.get(k, (0.0, 0.0))
                cls[k] = (u + w * up, d + w * (1 - up))
        self.class_memory = {}
        for k, (u, d) in cls.items():
            if u + d >= 3.0:
                v = _shrink((1.04 ** min(u, 8)) * (0.93 ** min(d, 8)),
                            u + d, 0.75, 1.25)
                if abs(v - 1.0) > 0.01:
                    self.class_memory[k] = v
        self.style_memory = {}
        for s, (u, d) in sty.items():
            if u + d >= 3.0:
                v = _shrink((1.06 ** min(u, 8)) * (0.90 ** min(d, 8)),
                            u + d, 0.6, 1.4)
                if abs(v - 1.0) > 0.01:
                    self.style_memory[s] = v
        # CONDITIONAL COMPETENCE: the same votes re-aggregated per (style,
        # condition). This is the memory that can IMPROVE a style instead
        # of retiring it - a technique that fails on loose grids but works
        # on precise ones keeps its place in the box and gets reached for
        # where it belongs, and the planner panel can say so in words.
        cnd = {}
        for r in rows:
            a, b = by_id.get(r["a_id"]), by_id.get(r["b_id"])
            if a is None or b is None or not r.get("style"):
                continue
            ck = (r["a_id"], r["b_id"])
            rt = _COND_RT_MEMO.get(ck)
            if rt is None:
                try:                 # the plan's rate is not stored; the
                    rt = seam_rhythm(a, b, 1.0) or {}   # tempo read is
                except Exception:                       # near enough for
                    rt = {}                             # a coarse bucket
                _COND_RT_MEMO[ck] = rt
            w = 0.5 if (r["source"] or "user") == "auto" else 1.0
            up = 1 if r["up"] else 0
            for c in seam_conditions(a, b, rt):
                k2 = (r["style"], c)
                u, d = cnd.get(k2, (0.0, 0.0))
                cnd[k2] = (u + w * up, d + w * (1 - up))
        self.style_cond_memory = {}
        for k2, (u, d) in cnd.items():
            if u + d >= 3.0:
                v = _shrink((1.06 ** min(u, 8)) * (0.90 ** min(d, 8)),
                            u + d, 0.6, 1.4)
                if abs(v - 1.0) > 0.01:
                    self.style_cond_memory[k2] = v
        return len(self.pair_memory)

    def style_multiplier(self, style, conds):
        """What the memory says about `style` FOR THESE CONDITIONS.

        Conditional evidence leads when it exists (it describes this seam);
        the global average is only a weak prior, and is halved when no
        conditional evidence applies - a global drawn from situations that
        do not hold here should not condemn the style here.

        A style that is bad in EVERY condition still lands at the bottom of
        the same 0.6-1.4 band the flat memory always used: this localises
        blame, it does not protect anything."""
        g = self.style_memory.get(style, 1.0)
        terms = [self.style_cond_memory[(style, c)] for c in conds
                 if (style, c) in self.style_cond_memory]
        if terms:
            prod = 1.0
            for t in terms:
                prod *= t
            cond = prod ** (1.0 / len(terms))     # geometric mean
            v = (cond ** 0.75) * (g ** 0.25)
        else:
            v = 1.0 + (g - 1.0) * 0.5
        return max(0.6, min(1.4, v))

    def set_flavor(self, flavor):
        self.flavor = dict(flavor or {})

    def explain_pick(self, cur, cand, meta):
        """One human line of WHY this track won - keys, stretch, flavor
        tags that matched, remembered seams. Steering needs reasons."""
        bits = []
        if cur is not None and cur.camelot and cand.camelot:
            c = camelot_compat(cur.camelot, cand.camelot)
            bits.append(f"{cur.camelot}->{cand.camelot}"
                        + ("" if c >= 0.9 else " ok" if c >= 0.55 else " !"))
        r = (meta or {}).get("rate")
        if r:
            bits.append(f"{(r - 1) * 100:+.1f}%")
        prefer = dict(self.theme.prefer_tags)
        prefer.update(self.flavor.get("prefer_tags") or {})
        hit = [t for t in prefer if t in cand.all_tags][:3]
        if hit:
            bits.append("+" + "+".join(hit))
        pm = self.pair_memory.get((getattr(cur, "id", None), cand.id))
        if pm:
            bits.append("mixed well before" if pm > 1.0 else "rough before")
        if (meta or {}).get("forced_fade"):
            bits.append("fade seam")
        return "  ".join(bits)

    def plan_horizon(self, current, arc_fn, out_bpm, n=3, preplayed=None):
        """PROVISIONAL next-n chain for the trajectory display: what would
        play if nothing changes. Pure lookahead - recency/skip state is
        snapshotted and restored, and a fixed-seed rng keeps the preview
        stable between recomputes (a queue that reshuffles every tick
        reads as indecision)."""
        saved_recent = list(self.recent)
        saved_rng = self.rng
        self.rng = random.Random(1234)
        out = []
        cur = current
        try:
            for t in (preplayed or []):      # queue items already shown
                self.note_played(t)
            for i in range(n):
                cand, meta = self.choose_next(
                    cur, arc_fn(i + 1), cur.bpm if cur else out_bpm)
                if cand is None:
                    break
                self.note_played(cand)
                out.append({"id": cand.id, "title": cand.title,
                            "artist": cand.artist,
                            "bpm": round(cand.bpm, 1),
                            "energy": round(cand.energy_proxy(), 2),
                            "tags": cand.all_tags[:4],
                            "why": self.explain_pick(cur, cand, meta)})
                cur = cand
        finally:
            self.recent = saved_recent
            self.rng = saved_rng
        return out

    def _arc_energy(self, cand):
        """The value the energy ARC chases for a candidate. Blends real ML
        AROUSAL (a cleaner intensity measure than the compressed energy proxy)
        in when the track is mood-scored; falls back to energy_proxy alone
        otherwise. Both are library percentiles, so they share the arc's 0..1
        scale."""
        e = cand.energy_proxy()
        ar = getattr(cand, "arousal_rank", None)
        if ar is None or not self._use_mood:
            return e
        return 0.6 * e + 0.4 * ar

    def _flavor_score(self, cand):
        """0.15..1.0 preference multiplier from theme flavor + live
        overrides: tag leans and axis-target pulls."""
        prefer = dict(self.theme.prefer_tags)
        avoid = dict(self.theme.avoid_tags)
        axes_t = dict(self.theme.axis_targets)
        prefer.update(self.flavor.get("prefer_tags") or {})
        avoid.update(self.flavor.get("avoid_tags") or {})
        axes_t.update(self.flavor.get("axis_targets") or {})
        if not (prefer or avoid or axes_t):
            return 1.0
        tags = set(cand.all_tags)
        s = 1.0
        for tag, w in prefer.items():
            s *= (1.0 + 0.9 * w) if tag in tags else (1.0 - 0.35 * w)
        for tag, w in avoid.items():
            if tag in tags:
                s *= 1.0 - 0.75 * w
        for axis, target in axes_t.items():
            # Ranked value when the library overlay has one (hypnotic/
            # hardness/vocal/energy saturate raw - see load_library), raw
            # else.
            v = cand.axes_rank.get(axis, cand.axes.get(axis))
            if v is None:
                continue
            # SIGMA 0.25, FLOOR 0.40 (was 0.35 / 0.50). At sigma 0.35 on a
            # 0..1 axis a candidate sitting a full 0.5 from the target -
            # the opposite half of the library - still scored 0.68, so an
            # axis target was at most a 1.5x preference against a product
            # of twenty other terms. The themes measurably converged
            # (hypnotic_deep's picks read 0.76 hypnotic against groove's
            # 0.69 with a 0.95 target). Now that the axes are honest
            # percentiles, a tighter pull cannot strand selection: material
            # exists at every point of a uniform distribution by
            # construction - the same argument that took the energy arc's
            # sigma from 0.3 to 0.21.
            s *= math.exp(-((float(v) - float(target)) / 0.25) ** 2) \
                * 0.6 + 0.4
        return max(s, 0.15)             # lean hard, never blacklist

    def seam_feedback(self, style, up):
        """Operator thumbs on the LAST transition: nudge tonight's style
        weighting (bounded - taste input, not a kill switch)."""
        cur = self.style_fb.get(style, 1.0)
        cur *= 1.15 if up else 0.7
        self.style_fb[style] = max(0.3, min(2.0, cur))

    def replay_style_fb(self, verdicts):
        """Rebuild tonight's thumb-driven style weighting from scratch as a
        pure function of the SURVIVING verdicts [(style, up), ...]. Called
        on every verdict add/change/clear: an edited or removed rating
        actually lets go of its nudge (the clamp in seam_feedback makes
        single presses non-invertible, so replay beats undo)."""
        self.style_fb = {}
        for style, up in verdicts:
            if style:
                self.seam_feedback(style, up)

    def note_skipped(self, track):
        """Operator skipped it - a labeled 'not tonight' the scorer uses."""
        tid = getattr(track, "id", track)
        self._recent_skips[tid] = self._recent_skips.get(tid, 0) + 1

    # -- memory --------------------------------------------------------------
    def note_played(self, track, when=None):
        self.recent.append((when or time.time(),
                            self.ckey.get(track.id, track.id),
                            track.artist))
        cutoff = (when or time.time()) - 10 * 3600
        self.recent = [r for r in self.recent if r[0] > cutoff]

    def _distinct_since_map(self):
        """{ckey -> number of DISTINCT other songs played since it last
        played}. A song not in recent memory is absent (== infinitely far).
        Cached per recent-list state (rebuilt on note_played)."""
        key = (len(self.recent), self.recent[-1][0] if self.recent else None)
        if self._ds_cache_key == key:
            return self._ds_cache
        seen_after, ds = set(), {}
        for _, tid, _ in reversed(self.recent):     # newest -> oldest
            if tid not in ds:                        # this is its LAST play
                ds[tid] = len(seen_after)            # distinct songs after it
            seen_after.add(tid)
        self._ds_cache, self._ds_cache_key = ds, key
        return ds

    def _recency_penalty(self, track, now=None):
        """DISTINCT-SONG no-repeat (tempo/duration independent): a song is a
        near-wall until norepeat_n OTHER distinct songs have played, then
        forgiven on a quadratic ramp. This is what stops 'the same song a
        lot' on small/steered pools - it round-robins the members instead of
        letting a wall-clock timer bring one back every hour. Nonzero floor so
        a genuinely dry pool still degrades to spaced repeats, not a stall.
        Same-artist spacing stays time-based."""
        now = now or time.time()
        ck = self.ckey.get(track.id, track.id)
        pen = 1.0
        ds = self._distinct_since_map().get(ck)
        if ds is not None and ds < self.norepeat_n:
            pen *= 0.003 + 0.997 * (ds / self.norepeat_n) ** 2
        # Same-artist spacing is BOUNDED (min x0.25 total): stacking it per
        # recent play drove EVERY candidate in a few-artist pool below the
        # old 0.01 floor, the floor flattened them equal, and the oldest-
        # first ordering vanished (measured: ds=1 repeats won in a 27-song
        # genre pool). The distinct-song term must stay the dominant order.
        art = 1.0
        for when, tid, artist in self.recent:
            if tid != ck and artist and artist == track.artist:
                age_h = (now - when) / 3600.0
                art *= min(1.0, 0.6 + age_h / 2.0)      # ~1h for an artist
        return max(pen * max(art, 0.25), 1e-4)

    # -- tempo ---------------------------------------------------------------
    def _grid_verified(self, current, cand):
        """Is this pair's beat matching MEASURED-good on both sides?

        The cheap half of the plan-time `_risky` predicate: whole-track
        grid confidence plus phase-profile coverage. Selection skips the
        swing_delta term deliberately - it needs the pair's rhythm
        signatures, which is real work per candidate, and the plan gate
        re-checks the FULL predicate before any blend is built. The worst
        case of that mismatch is a deep-stretch pair surviving selection
        and then being faded at plan time, which is exactly what happened
        to it before the wall widened.

        Only called for candidates past 5.5%, so the profile lookups stay
        off the hot path for the ordinary pick."""
        if current is None:
            return False
        if min(current.bpm_conf or 0.0, cand.bpm_conf or 0.0) < 0.8:
            return False
        from lib.dj import beatpower as _bpv
        # Candidate first (it varies, and it is the one that usually
        # fails), then the outgoing track's coverage MEMOIZED - it is
        # constant across a selection pass but the lookup stats the
        # profile file every call, which is ~18us x every deep candidate.
        # A background beatpower scan can leave this stale for at most
        # one track's play time, on a soft selection lean.
        if _bpv.profile_coverage(cand.id) < 0.8:
            return False
        memo = getattr(self, "_cov_cur", None)
        if memo is None or memo[0] != current.id:
            memo = self._cov_cur = (current.id,
                                    _bpv.profile_coverage(current.id))
        return memo[1] >= 0.8

    def rate_for(self, out_bpm, cand):
        """Best stretch rate to bring cand to out_bpm, allowing half/double
        time reads. Returns (rate, effective_bpm) or (None, 0)."""
        best = None
        for mult in (1.0, 2.0, 0.5):
            eff = cand.bpm * mult
            if eff <= 0:
                continue
            r = out_bpm / eff
            if self.stretch_min <= r <= self.stretch_max:
                if best is None or abs(math.log(r)) < abs(math.log(best[0])):
                    best = (r, eff)
        return best if best else (None, 0.0)

    def rate_for_dual(self, out_bpm, cand):
        """MEET-IN-THE-MIDDLE: when no single-deck stretch reaches, bend
        BOTH decks toward a meeting tempo (geometric mean, clamped inside
        each deck's stretch range) - extends the reachable tempo gap to
        ~12% (more with half/double reads). Returns (rate_b, eff_bpm,
        a_rate) or (None, 0, 1.0). a_rate is the OUTGOING deck's ramp
        target before the blend; the incoming glides home after.

        Per-deck cap 6% (2026-07-22, was 8.1%): riding a whole blend 8%
        off natural tempo audibly drags the groove - traditional practice
        cuts or fades a gap that wide instead of stretching through it.
        Beyond the cap the seam falls to a deliberate long_fade."""
        best = None
        for mult in (1.0, 2.0, 0.5):
            eff = cand.bpm * mult
            if eff <= 0:
                continue
            m = math.sqrt(out_bpm * eff)
            m = min(max(m, out_bpm * self.stretch_min,
                        eff * self.stretch_min),
                    out_bpm * self.stretch_max, eff * self.stretch_max)
            ra, rb = m / out_bpm, m / eff
            if not (self.stretch_min <= ra <= self.stretch_max
                    and self.stretch_min <= rb <= self.stretch_max):
                continue
            if abs(m / out_bpm - 1.0) > 0.06 or abs(m / eff - 1.0) > 0.06:
                continue
            cost = abs(math.log(ra)) + abs(math.log(rb))
            if best is None or cost < best[3]:
                best = (rb, eff, ra, cost)
        return (best[0], best[1], best[2]) if best else (None, 0.0, 1.0)

    def _rhythm_fit(self, current, cand, rate):
        """Bounded groove-compatibility lean 0.78..1.0 from the stored
        rhythm signatures (lib/dj/rhythm). Evidence-gated: either side
        without a signature is neutral 1.0. Cached per (pair, tempo-read) -
        selection scores every candidate on every horizon rebuild."""
        sa = getattr(current, "rhythm_sig", None)
        sb = getattr(cand, "rhythm_sig", None)
        if sa is None or sb is None:
            return 1.0
        mult = tempo_mult_for(current.bpm, cand.bpm, rate)
        key = (current.id, cand.id, mult)
        v = self._rhythm_cache.get(key)
        if v is None:
            rt = rhythm_terms(sa, sb, mult=mult,
                              period_s=60.0 / max(current.bpm, 1.0))
            v = rt["score"] if rt else 1.0
            self._rhythm_cache[key] = v
        return 0.78 + 0.22 * v

    def _body_energy(self, track):
        """Median section energy of a track's groove body (cached) - the
        reference for 'is this exit still alive or already comedown'."""
        cache = getattr(self, "_body_e_cache", None)
        if cache is None:
            cache = self._body_e_cache = {}
        v = cache.get(track.id)
        if v is None:
            es = [s.get("energy") or 0.0 for s in (track.sections or [])
                  if s.get("kind") not in ("intro", "outro")]
            v = float(np.median(es)) if es else 0.0
            cache[track.id] = v
        return v

    def _exit_life(self, track, out_s, stat="min"):
        """How alive the track's OWN 2 Hz curve is through the seam's
        exposure window around an exit anchor (see the EXIT_LIFE_*
        constants), against EXIT_LIFE_BODY_FRAC of the groove body,
        squared, clamped 0..1. 1.0 when there is no evidence (no curve /
        no body) - no penalty.

        Both consumers - the score damp and the phrase-snap guard - use
        the 'min' statistic (1s-smoothed hole depth). A q25 was tried
        for the damp on the theory that a min would over-punish short
        breathers; the renders said otherwise TWICE on Dunes alone: the
        quartile called both the 187s and the 228s windows alive (0.68,
        1.0) and both rendered floors of 0.02-0.07 - the dead-air gate
        is a min, and a 1-2s hush notch in the exposure window IS what
        it flags. 'q25' stays computed for diagnostics and any future
        consumer that genuinely wants broad liveness.
        Memoized per (track, anchor): anchors
        belong to the A side, which is shared by every candidate in a
        pick (~640 best_pair calls), so this computes ~8 windows per
        pick, not ~1650."""
        cache = getattr(self, "_exit_life_cache", None)
        if cache is None:
            cache = self._exit_life_cache = {}
        key = (track.id, int(out_s * 2))
        v = cache.get(key)
        if v is None:
            v = {"q25": 1.0, "min": 1.0}
            body = self._body_energy(track)
            arr = self._energy_arr(track)
            if body > 0.2 and len(arr):
                i0 = max(int((out_s - EXIT_LIFE_PRE_S) * 2), 0)
                i1 = min(int((out_s + EXIT_LIFE_POST_S) * 2), len(arr))
                if i1 > i0:
                    seg = arr[i0:i1]
                    sm = (seg[1:] + seg[:-1]) * 0.5 if len(seg) >= 2 \
                        else seg
                    ref = EXIT_LIFE_BODY_FRAC * body
                    v = {"q25": min(float(np.quantile(seg, 0.25))
                                    / ref, 1.0) ** 2,
                         "min": min(float(sm.min()) / ref, 1.0) ** 2}
            cache[key] = v
        return v[stat]

    def _entry_life(self, track, in_s):
        """_exit_life's B-side mirror: how alive the incoming track's
        curve is through [in_s, in_s + ENTRY_LIFE_SPAN_S] - the stretch
        B carries alone once A stops. Same quantile/body/square shape,
        same no-evidence-no-penalty default, memoized per anchor."""
        cache = getattr(self, "_entry_life_cache", None)
        if cache is None:
            cache = self._entry_life_cache = {}
        key = (track.id, int(in_s * 2))
        v = cache.get(key)
        if v is None:
            v = 1.0
            body = self._body_energy(track)
            arr = self._energy_arr(track)
            if body > 0.2 and len(arr):
                i0 = max(int(in_s * 2), 0)
                i1 = min(int((in_s + ENTRY_LIFE_SPAN_S) * 2), len(arr))
                if i1 > i0:
                    q = float(np.quantile(arr[i0:i1], 0.25))
                    v = min(q / (EXIT_LIFE_BODY_FRAC * body), 1.0) ** 2
            cache[key] = v
        return v

    def _pair_blendable(self, cur, cand, pair=None):
        """Could this pair support an overlapped blend? The CHEAP mirror
        of the plan-time screens (cached file-backed lookups only, no
        audio - score() runs this ~640x per pick): beat power at the
        seam-relevant regions (asymmetric bars: B's intro must groove,
        A's exit only hands off), and a trusted grid on both sides.
        Used as a selection LEAN so the DJ picks partners it can
        actually mix into - not as a gate.

        `pair` (best_pair's dict, when the caller has it) anchors the
        checks to the seam that would actually be planned. A TRUTHFUL
        mirror needs it: measured over 150 diagnosed picks (2026-08-12),
        21 winners this function called blendable were then faded by the
        plan gates, every one a track-level-vs-anchor-level disagreement.
        A mirror that flatters the winner is worse than none - it spends
        the s_blend lean promoting exactly the pairs the night cannot
        mix, past rivals it could."""
        from lib.dj import beatpower as _bp
        # Region the way the plan gate resolves it (_reg_for): the stored
        # 'in'/'out' band scores describe the PRIMARY mix points, but
        # best_pair may anchor the seam mid-track - judging a mid-groove
        # anchor by the intro/outro score was 5 of the 21 misses.
        def _reg(track, at_s, kind):
            try:
                pts = track.mix_outs if kind == "out" else track.mix_ins
                ref = pts[0]["time_s"] if pts else None
            except Exception:
                ref = None
            return kind if (ref is not None and at_s is not None
                            and abs(at_s - ref) <= 45.0) else "mid"
        out_s = pair["out_s"] if pair else None
        in_s = pair["in_s"] if pair else None
        bs_b = _bp.band_scores(cand.id, region=_reg(cand, in_s, "in")) or {}
        ev_b = [v for v in (bs_b.get("low"), _bp.scores().get(cand.id))
                if v is not None]
        if ev_b and max(ev_b) < _bp.BLEND_MIN:
            return False
        bs_a = _bp.band_scores(cur.id, region=_reg(cur, out_s, "out")) or {}
        ev_a = [v for v in (bs_a.get("low"), _bp.scores().get(cur.id))
                if v is not None]
        if ev_a and max(ev_a) < _bp.BLEND_MIN_EXIT:
            return False
        # GRID TRUST, the plan's own predicates (2026-08-12; was a
        # track-level coverage>=0.6 stand-in, the other 16 misses):
        #   * a PATCHY profile (0 < coverage < 0.6) is an unstable_phase
        #     kill regardless of conf - the phase wanders mid-blend;
        #   * the conf wall stands down only on trusted phase buckets AT
        #     THE ANCHORS (_local_ok), not on whole-track coverage.
        for t in (cur, cand):
            _cov = _bp.profile_coverage(t.id)
            if 0.0 < _cov < 0.6:
                return False
        if min(cur.bpm_conf or 0.0, cand.bpm_conf or 0.0) < 0.7:
            # Region-specific: the standdown must ask the correction's own
            # question (see _local_ok's 2026-08-16 note).
            if pair is None \
                    or _bp.phase_offset(cur.id, region="out",
                                        at_s=out_s) is None \
                    or _bp.phase_offset(cand.id, region="in",
                                        at_s=in_s) is None:
                return False
        # TEMPO SANITY (2026-08-05): a half/double-time pairing is a
        # legitimate FOLLOW but never a blend - the gate caught the lean
        # boosting a 160->85bpm pair into a "blend" whose grids can't
        # lock 1:1 (170ms structural delta). Same wall as selection's
        # stretch discipline.
        ratio = cur.bpm / max(cand.bpm, 1e-6)
        while ratio > 1.5:
            ratio /= 2.0
        while ratio < 0.67:
            ratio *= 2.0
        if not (0.945 <= ratio <= 1.058):
            return False
        if cur.bpm / max(cand.bpm, 1e-6) > 1.5 \
                or cand.bpm / max(cur.bpm, 1e-6) > 1.5:
            return False
        return True

    def _exit_blendable(self, t):
        """Can the night blend OUT of this track once it plays? The
        forward half of the chain: _pair_blendable judges the seam INTO
        a candidate, but 37 of 150 diagnosed picks (2026-08-12) faded
        with ZERO blendable partners on the table, and every one traced
        to the CURRENT track failing the A-side screens - the fade was
        decided one pick earlier, when the night entered a track it
        could not blend out of. Track-level only (the eventual partner
        doesn't exist yet): exit-region beat power against the exit bar,
        and a grid the gates could trust. Same cheap file-backed lookups
        as _pair_blendable."""
        from lib.dj import beatpower as _bp
        bs = _bp.band_scores(t.id, region="out") or {}
        evid = [v for v in (bs.get("low"), _bp.scores().get(t.id))
                if v is not None]
        if evid and max(evid) < _bp.BLEND_MIN_EXIT:
            return False
        _cov = _bp.profile_coverage(t.id)
        if 0.0 < _cov < 0.6:
            return False
        if (t.bpm_conf or 0.0) < 0.7 and _cov < 0.6:
            return False
        return True

    # -- selection -----------------------------------------------------------
    def score(self, current, cand, arc_target, out_bpm, now=None,
              bpm_target=None, relax=False, allow_repeat=False):
        if cand.id == getattr(current, "id", None)                 or cand.id in self.veto_ids:
            return 0.0, None
        if self.pool_ids is not None and cand.id not in self.pool_ids:
            return 0.0, None
        if not self._tag_ok(cand):          # hard "only these tags play"
            return 0.0, None
        # HARD NO-REPEAT (user: "repeats within the bulk of the set should
        # be deeply disallowed"): inside the norepeat window a song scores
        # ZERO - the soft penalty still let beat-similar recents outscore
        # fresh-but-unreachable songs, funneling the night onto the same
        # mixable cluster. choose_next's cascade prefers a FADE into fresh
        # music before ever setting allow_repeat.
        if not allow_repeat:
            ds = self._distinct_since_map().get(
                self.ckey.get(cand.id, cand.id))
            if ds is not None and ds < self.norepeat_n:
                return 0.0, None
        rate, eff_bpm = self.rate_for(out_bpm, cand)
        if rate is None:
            return 0.0, None
        lo, hi = self.theme.bpm_range
        # bpm_widen is persona TASTE (crate-digger blur at the edges), not
        # safety - the stretch wall and rate gates below are untouched.
        wide = (1.12 if relax else 1.0) * self.persona.bpm_widen
        if not (lo * 0.93 / wide <= eff_bpm <= hi * 1.07 * wide):
            return 0.0, None
        s_rate = math.exp(-((abs(math.log(rate))) / 0.045) ** 2)
        s_key = camelot_compat(getattr(current, "camelot", ""), cand.camelot)
        # KEY-SHIFT RESCUE: a clashing pair may become compatible with the
        # candidate pitched +/-1 semitone (deck does it tempo-neutrally).
        # Only when the shift direction keeps the COMBINED stretch sane.
        # NOT with the varispeed engine: pitch already rides tempo there, so
        # an independent transpose is impossible (the deck's stretch+resample
        # pitch path nets out to a plain rate change).
        pitch_st = 0
        if s_key < 0.5 and cand.camelot and getattr(current, "camelot", "") \
                and stretch_engine_name() != "vari":
            for st in (1, -1):
                shifted = _shift_camelot(cand.camelot, st)
                comb = abs(math.log(rate) - st * math.log(2.0) / 12.0)
                if camelot_compat(current.camelot, shifted) >= 0.8 \
                        and comb <= math.log(1.075):
                    # Rescued - but priced BELOW an honestly-ok key (0.55
                    # tier at dn=2): real DJs transpose ~2.5% of seams, and
                    # once the fade-avoidance lean landed, 0.7 made rescues
                    # win 10% of the time. A last resort, not a habit.
                    s_key = 0.62
                    pitch_st = st
                    break
        # CHROMA REFINEMENT: when both harmonic fingerprints exist, compare
        # them at the TRUE sounded pitch offset. Under varispeed the blend's
        # relative detune is 12*log2(rate) regardless of how the dual-deck
        # split shares it (rate_b/a_rate = rate); under keylock, tempo is
        # pitch-neutral so only the rescue transpose shifts anything. This
        # catches what integer Camelot can't: fractional detune, mislabeled
        # keys, and modal color the label doesn't carry. Averaged with the
        # Camelot tier, not replacing it - the label still anchors clean
        # cases, the measurement corrects the messy ones.
        cur_chroma = getattr(current, "chroma", None)
        if cur_chroma and cand.chroma:
            if stretch_engine_name() == "vari":
                semis = 12.0 * math.log(rate) / math.log(2.0)
            else:
                semis = float(pitch_st)
            sc = chroma_key_compat(cur_chroma, cand.chroma, semis)
            if sc is not None:
                s_key = 0.45 * s_key + 0.55 * sc
        # PERSONA key strictness: an exponent, so a perfect key stays 1.0
        # for everyone and only the gray zone moves - the purist punishes a
        # 0.6-tier key hard, the crate-digger barely notices it. Applied
        # after the chroma refinement AND the rescue tier, so a strict
        # persona also uses transposition rescues more sparingly.
        if self.persona.key_strictness != 1.0:
            s_key = s_key ** self.persona.key_strictness
        # Sigma 0.21 (was 0.3): the arc is chased on a library-PERCENTILE
        # scale, so a tight pull can't strand selection - material exists
        # near any target. At 0.3 a candidate 0.3 off-target still scored
        # 0.37 and the theme arcs measurably flattened (hard_drive's rise
        # materialized at 41% of its plan; user: "themes seem flat").
        s_energy = math.exp(-((self._arc_energy(cand) - arc_target) / 0.21) ** 2)
        s_mood = 0.25 + sum(self.theme.mood_weights.get(m, 0.0) * f
                            for m, f in cand.mood_hist.items())
        # ML MOOD AWARENESS (Music2Emo). All three are no-ops until tracks are
        # mood-scored (ml_valence/arousal present), so pre-mood behavior is
        # unchanged. (a) VALENCE CONTINUITY: don't cut a dark track into a
        # bubbly one - a big mood jump reads as jarring as a key clash. Soft
        # lean (0.6..1.0), both sides must be scored.
        s_valence = 1.0
        cv = getattr(current, "ml_valence", None) if current is not None else None
        if self._use_mood and cv is not None and cand.ml_valence is not None:
            s_valence = 0.6 + 0.4 * math.exp(
                -((cand.ml_valence - cv) / 0.35) ** 2)
        # (b) DANCEABILITY TARGET: club themes pull toward high danceability,
        # downtempo toward low. Only when this track is mood-scored and the
        # theme sets a target.
        s_dance = 1.0
        if self._use_mood and self.theme.dance_target is not None \
                and cand.ml_valence is not None \
                and getattr(cand, "danceability", None) is not None:
            s_dance = 0.5 + 0.5 * math.exp(
                -((cand.danceability - self.theme.dance_target) / 0.35) ** 2)
        # GENRE/ERA COHERENCE: free-play nights should hang together without
        # the operator lighting chips. Adjacent tracks sharing a genre get a
        # mild edge, fully disjoint ones a mild drag; a big release-year jump
        # (decades apart) leans down a touch. EVIDENCE-GATED like _blend_floor:
        # missing genre/year on either side is neutral - no evidence, no
        # penalty. Soft on purpose (min ~0.77 combined): a deliberate genre
        # pivot must stay one good seam away, and the operator's chips/flavor
        # always outrank this.
        s_cohere = 1.0
        if current is not None:
            ga = getattr(current, "genre_set", None) or set()
            gb = getattr(cand, "genre_set", None) or set()
            if ga and gb:
                frac = len(ga & gb) / min(len(ga), len(gb))
                s_cohere *= 0.84 + 0.16 * frac
            ya = getattr(current, "year", None)
            yb = getattr(cand, "year", None)
            if ya and yb:
                s_cohere *= 0.92 + 0.08 * math.exp(
                    -((float(ya) - float(yb)) / 15.0) ** 2)
        s_spec = 1.0
        if self.theme.spectral_lean == "bass":
            s_spec = 0.7 + 0.6 * cand.spectral.get("bass_share", 0.33) * 2.0
        elif self.theme.spectral_lean == "high":
            s_spec = 0.7 + 0.6 * cand.spectral.get("high_share", 0.2) * 3.0
        pair = self.best_pair(current, cand) if current is not None else None
        s_pair = pair["score"] if pair else (0.5 if current is None else 0.15)
        # TRANSITION-AWARE SELECTION: a candidate whose best seam would be
        # FORCED to long_fade (loose grid on either side, beatless seam, or
        # two sung passages over each other - the same gates plan_transition
        # applies) drags the night toward fades even though other candidates
        # would keep the real repertoire open. Lean away, never reject: on a
        # loose-gridded pool the fade is still the correct seam.
        s_style = 1.0
        forced_fade = False
        if current is not None and pair is not None:
            conf_min = min(current.bpm_conf or 0.0, cand.bpm_conf or 0.0)
            forced_fade = (conf_min < 0.5
                           or not pair.get("beaty", True)
                           or (self._vocal_at(current, pair["out_s"]) > 0.5
                               and self._vocal_at(cand, pair["in_s"]) > 0.5))
            if forced_fade:
                s_style = 0.55
            elif conf_min < 0.7:
                s_style = 0.9        # blends fine, precision styles gated off
            # GROOVE-OFFSET LEAN: grid-phase sync still flams by the two
            # tracks' kick-offset difference (the PLL's target is the grid,
            # not the kicks). Soft: a big offset gap only costs ~half.
            # Do NOT try to compensate this at sync instead: kick_offset_s
            # is bass PLACEMENT (median 0.35 beats on the real library),
            # not grid skew - shifting grids to align it pulled the real
            # kicks apart (user-heard double beats, 2026-07-13, reverted).
            # Sigma 38ms (was 45): a 31ms gap audibly double-beat a long
            # blend - selection should lose more ground to clean pairs in
            # the 25-40ms band.
            d_off = abs(current.kick_offset_s - cand.kick_offset_s)
            lean = 0.55 + 0.45 * math.exp(-((d_off / 0.038) ** 2))
            # PERSONA groove tolerance divides the PENALTY, not the score -
            # a tolerant persona shrugs at half-matched grooves (the style
            # gates still protect the seam); 1.0 is exact legacy behavior.
            if self.persona.groove_tolerance != 1.0:
                lean = 1.0 - (1.0 - lean) / self.persona.groove_tolerance
            s_style *= lean
        # VARIETY: two near-identical-sounding tracks back to back is the
        # classic auto-DJ tell. Penalize spectral+character similarity to
        # the current track (never to zero - a coherent run is fine, a
        # clone parade is not).
        s_var = 1.0
        if current is not None:
            s_var = 1.0 - 0.45 * self._similarity(current, cand)
        # GENERALIZED SEAM MEMORY: exact-pair memory only bites when the same
        # A->B recurs (rare on a 500+ library). class_memory buckets the same
        # feedback by pair FEATURES (key fit / groove-offset gap / grid conf),
        # so one measured train-wreck teaches the brain about the whole class.
        # Bounded tighter than pair memory - it applies broadly.
        s_class = 1.0
        if current is not None and self.class_memory:
            s_class = self.class_memory.get(
                self._pair_class(current, cand), 1.0)
        # GROOVE COMPATIBILITY (DB v13 rhythm signatures): kick-pattern
        # agreement + swing clash + flam risk at the planned read. Soft and
        # evidence-gated like s_cohere - unscanned tracks stay neutral, and
        # EQ discipline survives most kick clashes, so this leans (0.78x
        # floor), never vetoes.
        s_rhythm = self._rhythm_fit(current, cand, rate)
        if self.persona.groove_tolerance != 1.0:
            s_rhythm = 1.0 - (1.0 - s_rhythm) / self.persona.groove_tolerance
        # KICK-OFFSET FLAM LEAN (user-heard: "a bit too much flam",
        # 2026-08-04). The sync is grid-primary, so two tracks whose kicks
        # sit differently against their own grids flam by the offset
        # difference for the whole overlap - and measurement showed 54% of
        # chosen pairs sat past the 28ms audibility line, with the blend
        # family exposed on 37% of all seams. A lean, not a veto (x0.6 at
        # worst): half the library would be unreachable otherwise. Only
        # confident grids vote - a shaky offset is not evidence.
        if current is not None \
                and min(current.bpm_conf or 0.0, cand.bpm_conf or 0.0) >= 0.7:
            d_off = abs(current.kick_offset_s - cand.kick_offset_s)
            if d_off > 0.028:
                s_rhythm *= max(0.6, 1.0 - 6.0 * (d_off - 0.028))
        # PERSONA vocal pull: a soft lean toward (storyteller) or away from
        # (monk) singers, on the library-ranked vocal axis. +-0.35 at the
        # extremes - the same order as the theme's own axis leans, so the
        # operator's chips and the theme always outrank it.
        s_pers = 1.0
        if self.persona.vocal_pull:
            # RANKED, as the comment above always claimed and the code did
            # not (fixed 2026-08-13). It read the RAW demucs fraction,
            # whose median on this library is 0.0 - 57% of tracks are
            # wholly instrumental - so (2v-1) was -1.0 for most of the
            # library and the pull became a flat scalar that ordered
            # nothing. Measured across a night: storyteller 0.055, monk
            # 0.053, neutral 0.055 - a lever moving three thousandths.
            # The rank spreads the same material over 0..1, which is what
            # the theme's own vocal target already uses. (The RAW value
            # stays what the vocal-CLASH gates read - _vocal_at, the
            # acapella premise - those need absolute presence, not an
            # ordering.)
            v = cand.axes_rank.get("vocal")
            if v is None:
                v = cand.axes.get("vocal", 0.5)
            s_pers = 1.0 + 0.35 * self.persona.vocal_pull \
                * (2.0 * (v if v is not None else 0.5) - 1.0)
        # PERSONA exploration widens the selection dice (neutral 0.9..1.1).
        dice_w = 0.1 * self.persona.explore
        s_recency = self._recency_penalty(cand, now)
        s_skip = self._skip_penalty(cand)
        s_flavor = self._flavor_score(cand)
        s_pairmem = self.pair_memory.get(
            (getattr(current, "id", None), cand.id), 1.0)
        total = (s_rate * s_key * s_energy * s_mood * s_spec * s_var * s_style
                 * s_valence * s_dance * s_cohere * s_class * s_rhythm
                 * s_pers
                 * s_recency
                 * s_skip * s_pair
                 * s_flavor
                 * s_pairmem
                 * self.rng.uniform(1.0 - dice_w, 1.0 + dice_w))
        # STRETCH WALL: beyond ~5.5% the time-stretch is audible as feel
        # (WSOLA stays clean but the groove drags/rushes). Soft, not zero -
        # a dry pool may still cross it rather than strand the set.
        #
        # CONDITIONAL ON A VERIFIED GRID (2026-08-06), the rule the
        # plan-time gate used before it was rated away on 2026-08-13:
        # "deep stretch is only fatal on RISKY material". This SELECTION
        # lean is the surviving half of that idea and is deliberately kept
        # - it discourages depth without forbidding it, which is what the
        # verdicts said the hard gate should have been doing. A blanket
        # cliff here
        # made the widened wall cosmetic - it is a 20x penalty, so a 6-8%
        # candidate scored ~0.007 against a tempo-clean rival and could
        # never win the finalist dice however good the pair was.
        # Verified-grid pairs now face only the smooth s_rate lean;
        # everything unverified keeps the old cliff exactly.
        _deep = abs(math.log(rate)) > math.log(1.055)
        s_wall = 1.0
        if _deep and not self._grid_verified(current, cand):
            s_wall = 0.05
        total *= s_wall
        # Confident grids keep the CHAIN mixable: a low-conf pick forces
        # long_fade on both its seams. Mild lean only - flavor can still
        # bring in a loose-gridded track it really wants.
        s_conf = 0.75 + 0.25 * min(cand.bpm_conf, 1.0)
        total *= s_conf
        # BLENDABILITY LEAN (2026-08-05). A live night played FIVE fades
        # in a row ("holy shit this is bad" - user, skipping seam after
        # seam): selection knew nothing about what decides blend-vs-fade
        # (beat power, verified grids), so it happily chained partners
        # the gates could only fade between. The DJ's craft is picking
        # the next record so the MIX works - candidates this pair could
        # genuinely blend into score full; fade-bound ones pay hard.
        # Cheap (cached file-backed lookups only); not a hard filter, so
        # an arc/flavor pick the night really wants can still win.
        s_blend = 1.0 if self._pair_blendable(current, cand, pair) else 0.45
        total *= s_blend
        # CHAIN LOOKAHEAD (2026-08-12). s_blend judges the seam INTO the
        # candidate; nothing judged the seam OUT of it, and the selection
        # diagnosis measured the cost: every zero-blendable-rival fade
        # (37 of 150 picks) was decided one pick EARLIER, when the night
        # entered a track whose own exit fails the A-side screens.
        # Softer than s_blend (0.6 vs 0.45): the candidate's own seam is
        # still good, the cost is deferred - and flavor/arc can still
        # bring in a dead-end track it really wants, which will then
        # leave by a fade that is honestly the right seam.
        s_exit_chain = 1.0 if self._exit_blendable(cand) else 0.6
        total *= s_exit_chain
        # TEMPO ARC: the night has a planned BPM journey, not just a range.
        # (Weight raised 0.45->0.60: at 0.45 a rise-theme night walked only
        # ~40% of its planned tempo climb - seam-quality terms outvoted it.)
        s_bpm_arc = 1.0
        if bpm_target:
            s_bpm_arc = 0.40 + 0.60 * math.exp(
                -((eff_bpm - bpm_target) / 7.0) ** 2)
            total *= s_bpm_arc
        # TERM BREAKDOWN rides the winning candidate's meta into the `armed`
        # log line, so tools/dj/dj_review.py can correlate every term against
        # what the seam MEASURED. Eighteen constants were each tuned in
        # isolation by ear or sim and never checked against outcomes - this
        # is the join that makes that checkable. Cheap: a dict of floats per
        # candidate, ~640 per pick, once every few minutes.
        terms = {"rate": s_rate, "key": s_key, "energy": s_energy,
                 "mood": s_mood, "spec": s_spec, "var": s_var,
                 "style": s_style, "valence": s_valence, "dance": s_dance,
                 "cohere": s_cohere, "class": s_class, "rhythm": s_rhythm,
                 "persona": s_pers, "recency": s_recency, "skip": s_skip,
                 "pair": s_pair, "flavor": s_flavor, "pairmem": s_pairmem,
                 "wall": s_wall, "conf": s_conf, "blend": s_blend,
                 "exit_chain": s_exit_chain, "bpm_arc": s_bpm_arc}
        return total, {"rate": rate, "eff_bpm": eff_bpm, "pair": pair,
                       "pitch_st": pitch_st, "forced_fade": forced_fade,
                       "terms": terms}

    def set_theme(self, theme):
        """Live retheme - same library fitting as construction."""
        self.theme = adapt_theme(theme, self._lib_bpms)

    def set_require_tags(self, tags):
        """Set the HARD tag filter (live panel). Only tracks carrying at least
        one of these tags may play; empty clears the filter."""
        self.require_tags = {str(t) for t in (tags or [])}

    def _tag_ok(self, cand):
        """True if the candidate satisfies the hard tag filter (has at least
        one required tag), or no filter is set."""
        return (not self.require_tags
                or bool(self.require_tags & set(cand.all_tags)))

    def eligible_pool_size(self):
        """How many songs the selectors can actually draw from right now
        (setlist pool ∩ hard tag filter). Surfaced on the live panel: a
        tiny pool makes repeats ARITHMETICALLY forced ('only these tags' +
        4 matching songs = a repeat every ~4 songs), and the night reads
        wrong unless the operator can see why."""
        return sum(1 for t in self.library
                   if (self.pool_ids is None or t.id in self.pool_ids)
                   and self._tag_ok(t))

    @staticmethod
    def _similarity(a, b):
        """0..1 sameness of two tracks (spectral shares + character axes)."""
        d = (abs(a.spectral.get("bass_share", .33) - b.spectral.get("bass_share", .33))
             + abs(a.spectral.get("mid_share", .33) - b.spectral.get("mid_share", .33))
             + abs(a.spectral.get("high_share", .25) - b.spectral.get("high_share", .25))
             + 0.5 * abs(a.energy_proxy() - b.energy_proxy())
             + 0.5 * abs(a.axes_rank.get("hypnotic",
                                         a.axes.get("hypnotic", .5) or .5)
                         - b.axes_rank.get("hypnotic",
                                           b.axes.get("hypnotic", .5) or .5)))
        return max(0.0, 1.0 - d * 2.0)

    def _skip_penalty(self, cand):
        """HISTORY LEARNING: tracks the operator skipped recently score
        lower - a skip is a labeled 'not tonight'."""
        n = self._recent_skips.get(cand.id, 0)
        return 1.0 / (1.0 + 0.8 * n)

    def choose_next(self, current, arc_target, out_bpm, now=None,
                    bpm_target=None):
        """Returns (TrackInfo, meta) or (None, None) when the library is dry.

        LOOKAHEAD: greedy picking paints into corners (a great seam into a
        track nothing mixes out of). The top few candidates also get judged
        by their own best successor, so the pick keeps the set OPEN."""
        scored = []
        for cand in self.library:
            s, meta = self.score(current, cand, arc_target, out_bpm, now,
                                 bpm_target=bpm_target)
            if s > 0.0:
                scored.append((s, cand, meta))
        # Surfaced on the live panel: the pool a pick ACTUALLY drew from
        # (tag filter ∩ tempo-reachable from the current track). A narrow
        # style + skipping through it shrinks this to a handful, and then
        # repeats are forced no matter what the wall says.
        self.last_scored_n = len(scored)
        if not scored:
            # DEAD-END RESCUE: from tempo-extreme tracks the theme window
            # can be unreachable (measured: 69/573 on the real library).
            # Widen the gate rather than strand the set - the stretch wall
            # still keeps rates sane, and a long_fade is always available.
            for cand in self.library:
                s, meta = self.score(current, cand, arc_target, out_bpm,
                                     now, bpm_target=bpm_target, relax=True)
                if s > 0.0:
                    scored.append((s, cand, meta))
        if not scored:
            # FRESH-OVER-REPEAT: every tempo-reachable candidate is inside
            # the no-repeat window. A FADE into a fresh song (any tempo in
            # the theme window) beats a beatmatched repeat - otherwise the
            # night funnels onto the same beat-similar cluster forever.
            pick = self._fresh_fade_pick(current, arc_target, now)
            if pick is not None:
                return pick, {"rate": 1.0, "eff_bpm": pick.bpm, "pair": None,
                              "tempo_clash": True}
        if not scored:
            # LAST RESORT - repeats allowed, and OLDEST-FIRST IS ABSOLUTE:
            # take the oldest cohort of eligible songs (within 4 distinct
            # of the oldest available), prefer a beatmatched seam INSIDE
            # it, fade otherwise. Ranking by score let seam quality win -
            # a 2-song tempo island then flip-flopped A-B-A-B forever
            # while 20-song-old material sat one fade away (measured).
            # Returns (None, None) only on an empty/vetoed pool, which is
            # how planned sets end short instead of repeating.
            ds_map = self._distinct_since_map()
            elig = [t for t in self.library
                    if t.id != getattr(current, "id", None)
                    and t.id not in self.veto_ids
                    and (self.pool_ids is None or t.id in self.pool_ids)
                    and self._tag_ok(t)]
            if not elig:
                return None, None

            def _age(t):
                d = ds_map.get(self.ckey.get(t.id, t.id))
                return self.norepeat_n if d is None else d
            oldest = max(_age(t) for t in elig)
            cohort = [t for t in elig if _age(t) >= oldest - 4]
            best = None
            for cand in cohort:
                s, meta = self.score(current, cand, arc_target, out_bpm,
                                     now, bpm_target=bpm_target, relax=True,
                                     allow_repeat=True)
                if s > 0.0 and (best is None or s > best[0]):
                    best = (s, cand, meta)
            if best is not None:
                return best[1], best[2]
            pick = max(cohort, key=lambda t: (
                math.exp(-((self._arc_energy(t) - arc_target) / 0.21) ** 2)
                * self._flavor_score(t) * self._skip_penalty(t)))
            return pick, {"rate": 1.0, "eff_bpm": pick.bpm, "pair": None,
                          "tempo_clash": True}
        scored.sort(key=lambda x: -x[0])
        finalists = []
        for s, cand, meta in scored[:5]:
            succ = 0.0
            for s2, cand2, _ in scored[:12]:
                if cand2.id == cand.id:
                    continue
                p = self.best_pair(cand, cand2)
                if p and p["score"] > succ:
                    succ = p["score"]
            finalists.append((s * (0.75 + 0.25 * succ), cand, meta))
        # SAMPLE among the finalists (cubed weights keep it quality-biased)
        # instead of argmax: a deterministic winner-take-all played the
        # same tracks in the same order every single night.
        finalists.sort(key=lambda x: -x[0])
        top = finalists[:3]
        # PERSONA exploration flattens (crate-digger) or sharpens (monk)
        # the finalist sampling; neutral keeps the tuned cubed weights.
        weights = [t[0] ** (3.0 / self.persona.explore) for t in top]
        pick = self.rng.choices(top, weights=weights, k=1)[0]
        return pick[1], pick[2]

    def _fresh_fade_pick(self, current, arc_target, now=None):
        """A FRESH song to fade into when every tempo-reachable candidate is
        a near repeat. Stays inside the theme's bpm window (half/double
        reads count) so the night keeps its character; the seam will be a
        deliberate long_fade (caller passes tempo_clash meta)."""
        lo, hi = self.theme.bpm_range
        ds_map = self._distinct_since_map()
        pool = []
        for t in self.library:
            if t.id == getattr(current, "id", None) or t.id in self.veto_ids:
                continue
            if self.pool_ids is not None and t.id not in self.pool_ids:
                continue
            if not self._tag_ok(t):
                continue
            ds = ds_map.get(self.ckey.get(t.id, t.id))
            if ds is not None and ds < self.norepeat_n:
                continue
            if not any(lo * 0.93 <= t.bpm * m <= hi * 1.07
                       for m in (1.0, 2.0, 0.5)):
                continue
            pool.append(t)
        if not pool:
            return None
        ranked = [(math.exp(-((self._arc_energy(t) - arc_target) / 0.21) ** 2)
                   * self._flavor_score(t) * self._skip_penalty(t)
                   * self._recency_penalty(t, now)
                   * self.rng.uniform(0.9, 1.1), t) for t in pool]
        ranked.sort(key=lambda x: -x[0])
        return ranked[0][1]

    def emergency_pick(self, current, arc_target, now=None):
        """CONTINUITY OUTRANKS POLISH: the watchdog's last resort when even
        the relaxed selection came up empty with the current track about to
        run out. Every gate except recency is ignored - closest energy fit
        wins and the seam will be a fade. (None, None) only on an empty
        library."""
        pool = [t for t in self.library
                if t.id != getattr(current, "id", None)
                and t.id not in self.veto_ids
                and (self.pool_ids is None or t.id in self.pool_ids)
                and self._tag_ok(t)]
        if not pool:
            pool = [t for t in self.library
                    if t.id != getattr(current, "id", None)
                    and t.id not in self.veto_ids]
        if not pool:
            return None, None
        best = max(pool, key=lambda t: (
            math.exp(-((self._arc_energy(t) - arc_target) / 0.21) ** 2)
            * self._recency_penalty(t, now)))
        return best, {"rate": 1.0, "eff_bpm": best.bpm, "pair": None,
                      "tempo_clash": True}

    def choose_first(self, arc_target, now=None):
        cands = []
        for cand in self.library:
            if self.pool_ids is not None and cand.id not in self.pool_ids:
                continue
            if not self._tag_ok(cand):
                continue
            lo, hi = self.theme.bpm_range
            if not (lo * 0.93 <= cand.bpm <= hi * 1.07):
                continue
            s = math.exp(-((self._arc_energy(cand) - arc_target) / 0.21) ** 2) \
                * (0.25 + sum(self.theme.mood_weights.get(m, 0.0) * f
                              for m, f in cand.mood_hist.items())) \
                * self._recency_penalty(cand, now) \
                * self._flavor_score(cand) * self.rng.uniform(0.9, 1.1)
            cands.append((s, cand))
        if not cands and self.pool_ids is not None:
            # The operator's pool outranks the theme's tempo taste: open
            # with the best-fitting pool track even off-window.
            for cand in self.library:
                if cand.id in self.pool_ids:
                    cands.append((math.exp(-((self._arc_energy(cand)
                                              - arc_target) / 0.21) ** 2)
                                  * self.rng.uniform(0.9, 1.1), cand))
        if not cands:
            return None
        # Open with ANY strong fit, not always the same single winner.
        cands.sort(key=lambda x: -x[0])
        top = cands[:6]
        return self.rng.choices(top, weights=[s ** 2 for s, _ in top],
                                k=1)[0][1]

    # -- section-pair mixability (the anti-garbage rule) -------------------------
    def best_pair(self, cur, cand, after_s=None, exclude_out_s=None):
        """Best (A-exit, B-entry) combination, or None. Never lets two
        busy/vocal sections blend over each other.

        exclude_out_s: skip A-exits within ±2s of this point - the
        exit-retry's tool for asking "and without THAT exit?" when the
        top pair's out point alone killed every blend."""
        # THE PLAY-TIME BUDGET IS A PREFERENCE, NOT A FILTER (2026-08-07).
        # As a filter this deleted every candidate below after_s - and
        # Theme.min_play_s/max_play_s are ABSOLUTE seconds with no idea how
        # long the record is, so on the real library groove's valley draw
        # medians 330s against a median track of 332s: the budget asks for
        # ~99% of the song. Nothing survived on 25% of seams and
        # plan_transition fell back to `duration - 35s`, which is the
        # comedown by construction - the user heard it as "songs mix out
        # deep into their comedown, just noodling around for a while".
        # Measured over 360 seams: fallback 25% -> 0%, exits landing past
        # the track's last groove section 19.2% -> 9.4%, at a cost of a
        # median 3% of track length in play time. Earliness is now paid
        # for in the score (BUDGET_TAU_S) instead of being fatal.
        outs = list(cur.mix_outs)
        if exclude_out_s is not None:
            outs = [o for o in outs
                    if abs(o["time_s"] - exclude_out_s) > 2.0]
        if not outs:
            return None
        # PERSONA PACING REACHES THE EXIT (2026-08-13). play_len_x scaled
        # only the drawn BUDGET, which is a floor - and shortening a floor
        # WIDENS the unpenalised region above it, so showman rode records
        # exactly as far as monk (measured 3.34 vs 3.47 min against a 62%
        # ask; the simulator's old 3.26-vs-5.50 spread was its own artefact
        # - it advanced by the budget rather than the planned exit). The
        # lever has to move the LATE side too: a patient persona tolerates
        # overrunning its budget, an impatient one does not.
        _plx = float(getattr(self.persona, "play_len_x", 1.0) or 1.0)
        _late_tau = LATE_TAU_S * _plx
        # ...and it may pull the hard ceiling IN, never past it: the 0.85 is
        # a safety about what a record has left, not a taste knob, so monk's
        # 1.30 clamps back to it while showman's 0.80 genuinely leaves
        # earlier.
        _hard_frac = min(EXIT_LATE_HARD_FRAC * _plx, EXIT_LATE_HARD_FRAC)
        # How good a section is to mix OUT of / IN to. The golden rule of
        # melodic-house mixing: bring the new track's INTRO (drums, no lead)
        # in over the old track's OUTRO/breakdown, so two lead melodies never
        # play at once. Kind drives this; busyness/vocalness refine it.
        # ML structure labels (allin1, tracks.structure) refine the
        # internal kinds when present: the SSM sectionizer can't tell a
        # chorus from a groove, so without them a seam could exit A mid-hook
        # or drop B's entry into the middle of its chorus. Averaged with
        # the internal base - either source alone can be wrong.
        ml_out = {"outro": 1.0, "end": 1.0, "break": 0.9, "bridge": 0.75,
                  "inst": 0.7, "solo": 0.6, "verse": 0.5, "start": 0.5,
                  "chorus": 0.35, "intro": 0.4}
        ml_in = {"intro": 1.0, "start": 1.0, "break": 0.8, "inst": 0.75,
                 "verse": 0.6, "bridge": 0.6, "solo": 0.5, "chorus": 0.45,
                 "outro": 0.2, "end": 0.2}

        def out_fit(sec, voc, ml):
            k = sec.get("kind", "groove")
            base = {"outro": 1.0, "breakdown": 0.85, "groove": 0.6,
                    "build": 0.3, "intro": 0.4}.get(k, 0.5)
            if ml:
                base = 0.5 * base + 0.5 * ml_out.get(ml, 0.5)
            return base * (1.0 - 0.5 * voc)

        def in_fit(sec, voc, ml):
            k = sec.get("kind", "groove")
            base = {"intro": 1.0, "breakdown": 0.8, "groove": 0.55,
                    "build": 0.6, "outro": 0.2}.get(k, 0.5)
            if ml:
                base = 0.5 * base + 0.5 * ml_in.get(ml, 0.5)
            return base * (1.0 - 0.6 * voc)

        # PREPASS both axes once (the old o x i loop recomputed every
        # per-i quantity - section, ml label, 24s vocal walk - for every
        # o, an 8x waste; and called _blend_floor per combo, ~21 calls
        # per candidate = 60% of all selection time. Perf audit
        # 2026-07-31; scores are arithmetic-identical, just hoisted.)
        outs = outs[:8]
        o_pre = []
        for o in outs:
            sec_a = cur.section_at(min(o["time_s"] + 1.0,
                                       cur.duration_s - 1.0))
            # POINT-ACCURATE vocals: the seam only overlaps A's last ~24s
            # before out_s and B's first ~24s after in_s. The per-section
            # MEAN dilutes a hook that sits exactly there (and credits one
            # that doesn't); the fine demucs curve, walked across the
            # actual overlap window, judges what will really sound.
            voc_a = max(sec_a.get("vocalness") or 0.0,
                        self._vocal_span_max(cur, o["time_s"] - 24.0,
                                             o["time_s"])) \
                if sec_a is not None else 0.0
            ml_a = cur.ml_segment_at(min(o["time_s"] + 1.0,
                                         cur.duration_s - 1.0))
            o_pre.append((o, sec_a, voc_a, ml_a))
        # AN ENTRY MUST LEAVE ROOM FOR A SONG (2026-08-16). The remainder
        # cap (system._draw_exit) budgets play as a fraction of what is
        # left AFTER the entry point, so a deep entry that leaves ~110s of
        # record caps the dwell near 80s - silently bypassing the theme's
        # min_play_s floor. Measured with the persona sim on this library
        # (groove): EVERY persona played 15-29% of its songs for under two
        # minutes, monk included - structural, not taste. Judge each entry
        # by the dwell the cap will actually ALLOW: entries that cannot
        # host even the persona-scaled minimum draw are skipped while a
        # roomier entry exists, and merely leaned against (x room^2) when
        # the whole record is too short to host it - short records still
        # play, they just stop winning ties. play_len_x keeps the persona
        # spread: monk demands more runway than showman.
        _floor_s = self.theme.min_play_s * _plx \
            * float(getattr(self.persona, "entry_floor_x", 1.0) or 1.0)
        _room_frac = min(EXIT_BUDGET_FRAC * _plx, EXIT_LATE_HARD_FRAC)
        i_pre = []
        for i in cand.mix_ins[:8]:
            sec_b = cand.section_at(min(i["time_s"] + 1.0,
                                        cand.duration_s - 1.0))
            if sec_b is None:
                i_pre.append((i, None, 0.0, "", 0.0, 0.0, True))
                continue
            # Off-meter is a property of the ENTRY, not the combo -
            # hoisted out of the o x i loop (it re-asked per o), and
            # _roomy below must not count a vetoed entry: Seed's roomy
            # intros are off-meter, and counting them vetoed its one
            # clean entry for cramped room - every combo died and the
            # pair fell to plan_transition's blind fallback exit.
            om_b = bool(off_meter_span(cand, i["time_s"] - 5.0,
                                       i["time_s"] + 30.0))
            room = 1.0
            if cand.duration_s > 0 and _floor_s > 0:
                runway = max(cand.duration_s - i["time_s"], 0.0) * _room_frac
                room = min(runway / _floor_s, 1.0)
            voc_b = max(sec_b.get("vocalness") or 0.0,
                        self._vocal_span_max(cand, i["time_s"],
                                             i["time_s"] + 24.0))
            ml_b = cand.ml_segment_at(min(i["time_s"] + 1.0,
                                          cand.duration_s - 1.0))
            # Prefer mixing the incoming in EARLIER (nearer its groove
            # start) over a deep point, but only a gentle lean - the
            # mix-in must still land where the track has energy, or the
            # blend goes quiet as the outgoing leaves.
            early_b = math.exp(-max(i["time_s"] - 20.0, 0.0) / 120.0)
            i_pre.append((i, sec_b, voc_b, ml_b, early_b, room, om_b))
        # Does ANY entry have full room? If so the cramped ones are vetoed
        # outright below; if not (a short record), the lean decides.
        # Only entries that survive the off-meter veto count - a vetoed
        # entry must not set the bar the survivors are judged by.
        _roomy = any(p[5] >= 1.0 for p in i_pre
                     if p[1] is not None and not p[6])
        # NO HOLES IN THE BLEND: section MEANS hide a near-silent
        # stretch inside the overlap (a breakdown bar, a stripped
        # intro) - the render then dips to nothing mid-blend
        # (measured rms 0.04, user-audible dead air). Walk the two
        # 2 Hz energy curves through the aligned overlap window and
        # punish any moment where BOTH sides are quiet at once.
        # One vectorized pass for the whole out x in grid.
        holes = self._blend_floor_grid(
            cur, [o["time_s"] for o in outs],
            cand, [ip[0]["time_s"] for ip in i_pre])
        best = None
        # LEAVE WHILE THE MUSIC IS STILL ALIVE (2026-08-05, user: "the dj
        # seems to like to go almost to the end to mix, but the tail of
        # a lot of songs is often just empty comedown"). The analyzer's
        # bookmarked mix-outs mostly sit at the END - past the comedown -
        # so the night rode every outro into dead air before mixing.
        # Score each exit by its section's energy AGAINST THE TRACK'S
        # BODY: the last strong phrase boundary beats a tail bookmark.
        _body_e = self._body_energy(cur)
        # ANCHOR TRUST STEERS THE PICK (2026-08-14). Two facts the seam
        # gates judge at the CHOSEN anchors, which selection used to
        # ignore - so it kept choosing anchors that doomed winnable
        # seams:
        #  - OFF-METER GRID SEGMENTS: plan_transition diverts any seam
        #    whose overlap crosses a segment >5% off the track's meter
        #    to the deliberate fade (~12% of seams). Candidates inside
        #    those spans are vetoed HERE, with the same off_meter_span
        #    windows the diversion uses, so a clean anchor wins when one
        #    exists and the fade only happens when the track genuinely
        #    offers nothing honest.
        #  - PHASE TRUST on low-conf pairs: the grid_conf<0.7 wall
        #    stands down only when BOTH anchors carry a measured,
        #    trusted phase bucket (_local_ok). Pulsacions (conf 0.67)
        #    exited at 413s - IQR ~130ms, untrusted - when its 370s
        #    bucket measured IQR 8ms: the pick alone forced the whole
        #    night around it into unsynced fades. When the pair is
        #    low-conf AND both tracks have usable coverage, untrusted-
        #    anchor combos now score x0.35: a lean, not a kill, so a
        #    track with no trusted anchors still exits somewhere.
        _lowconf = min(cur.bpm_conf or 0.0, cand.bpm_conf or 0.0) < 0.7
        _trust_matters = False
        if _lowconf:
            from lib.dj import beatpower as _bpt
            if (_bpt.profile_coverage(cur.id) >= 0.6
                    and _bpt.profile_coverage(cand.id) >= 0.6):
                _trust_matters = True
        _trust_o = {}
        _trust_i = {}
        for oi, (o, sec_a, voc_a, ml_a) in enumerate(o_pre):
            if sec_a is None:
                continue
            if off_meter_span(cur, o["time_s"] - 5.0, o["time_s"] + 30.0):
                continue
            if _trust_matters and oi not in _trust_o:
                _trust_o[oi] = _bpt.phase_offset(
                    cur.id, region="out", at_s=o["time_s"]) is not None
            of = out_fit(sec_a, voc_a, ml_a)
            # What leaving EARLY costs (see the `outs` comment above): a
            # candidate below the drawn budget stays on the table, decaying
            # with how early it is, so a good exit ~a minute early can beat
            # a dead one that merely lands on time.
            bud = 1.0
            if after_s is not None and o["time_s"] < after_s:
                bud = math.exp(-(after_s - o["time_s"]) / BUDGET_TAU_S)
            elif after_s is not None and o["time_s"] > after_s:
                # LATE COSTS TOO (2026-08-13, see LATE_TAU_S). Without this
                # the budget was a floor with nothing above it and the
                # scorer rode every record into its last groove.
                bud = math.exp(-(o["time_s"] - after_s) / _late_tau)
            # ...and past the hard fraction the record has nothing left to
            # leave ON. Not a lean - a refusal, the way the constant in
            # system.py always described itself.
            if cur.duration_s > 0 and o["time_s"] > _hard_frac * cur.duration_s:
                continue
            busy_a = sec_a.get("busyness") or 0.0
            ra = sec_a.get("rhythm_density") or 0.0
            ea = sec_a.get("energy") or 0.0
            if _body_e > 0.2:
                of *= 0.25 + 0.75 * min(ea / _body_e, 1.0)
            # ...and by the CURVE through the seam's exposure window,
            # which the section mean above cannot see (the fade-crater
            # class - see the EXIT_LIFE_* constants). NOT folded into
            # `of`: the weighted-sum fit floor below (0.25 + 0.75*fit)
            # caps everything routed through it at a 4x swing, and the
            # dead Mukadderat anchor beat that cap through rf * clash
            # alone. Like `bud` and `room` - single-anchor facts - the
            # life factor multiplies the score directly. Memoized per
            # anchor, so this costs ~8 quantiles per pick.
            xlife = EXIT_LIFE_FLOOR + (1.0 - EXIT_LIFE_FLOOR) \
                * self._exit_life(cur, o["time_s"])
            for ii, (i, sec_b, voc_b, ml_b, early_b, room,
                     om_b) in enumerate(i_pre):
                if sec_b is None or om_b:
                    continue
                if room < 1.0 and _roomy:
                    continue        # a roomier entry exists - take that one
                if room < 1.0 and not _roomy \
                        and self._entry_life(cand, i["time_s"]) \
                        < ENTRY_LIFE_RESCUE_MIN:
                    # A cramped entry competes only when no roomy clean
                    # entry exists (_roomy fix above) AND it is actually
                    # ALIVE. Rescuing a cramped DEAD entry trades the
                    # None->fallback forced fade (which enters B where
                    # the hints say B works) for a planned crater:
                    # Father King -> The Road Back's only clean entries
                    # sit in a dying breakdown (entry-life 0.12/0.16),
                    # and rescuing one measured floor 0.50 -> 0.11.
                    continue
                if _trust_matters and ii not in _trust_i:
                    _trust_i[ii] = _bpt.phase_offset(
                        cand.id, region="in", at_s=i["time_s"]) is not None
                busy_b = sec_b.get("busyness") or 0.0
                fit = of * in_fit(sec_b, voc_b, ml_b)
                quiet = 1.0 - 0.5 * min(busy_a + busy_b, 1.6) / 1.6
                # BLEND WHERE THE BEATS ARE: a beat-matched blend is only
                # audible as beat-matched if BOTH sides carry rhythm and
                # comparable energy - otherwise it just reads as a fade.
                # ASYMMETRIC since 2026-08-17 (the fade-crater class):
                # B's entries are quiet intros by the golden rule, and a
                # symmetric match term rewarded the A-exit that DIED to
                # meet them - on Mukadderat the live groove anchor took
                # 0.14 from this line while the dead one took 0.38, and
                # that ratio outvoted every energy damp upstream. B
                # hotter than A's exit is the real lurch (B slams in
                # over a receding A: tight 0.4 sigma stays); A carrying
                # over a quiet entry is the blessed shape itself
                # (recede 0.5 holds the room while B rises), so that
                # side only leans, it never drags the exit into the
                # comedown to "match".
                rb = sec_b.get("rhythm_density") or 0.0
                eb = sec_b.get("energy") or 0.0
                beaty = ra >= 1.2 and rb >= 1.2
                _esig = 0.4 if eb > ea else 0.8
                rhythm_fit = (1.3 if beaty else 0.55) \
                    * math.exp(-((ea - eb) ** 2) / (2 * _esig ** 2))
                # Two lead-carrying sections over each other = clash: heavy
                # penalty (not a hard reject, so there's always a best pair).
                clash = 1.0
                if busy_a > 0.6 and busy_b > 0.6:
                    clash *= 0.3
                if voc_a > 0.5 and voc_b > 0.5:
                    clash *= 0.25
                mp = 0.5 + 0.5 * max(o["score"], 0.0) * max(i["score"], 0.0)
                hole = holes[oi][ii]
                # Weighted-sum form so a mediocre pair stays ~0.05-1, never
                # collapsing to ~0 (which would zero the whole selection).
                score = ((0.25 + 0.75 * fit) * (0.6 + 0.4 * quiet)
                         * (0.4 + 0.6 * early_b) * rhythm_fit * clash * mp
                         * (0.25 + 0.75 * min(hole / 0.25, 1.0)) * bud
                         * room * room * xlife)
                if _trust_matters and not (_trust_o.get(oi)
                                           and _trust_i.get(ii)):
                    # See the anchor-trust note above the loop: on a
                    # low-conf pair, untrusted anchors force the fade at
                    # plan time - prefer the anchors that keep the
                    # blend alive.
                    score *= 0.35
                if best is None or score > best["score"]:
                    best = {"out_s": o["time_s"], "in_s": i["time_s"],
                            "out_hint": o.get("style_hint", "blend"),
                            "in_hint": i.get("style_hint", "blend"),
                            "score": round(score, 5), "beaty": beaty,
                            "kinds": (sec_a.get("kind"), sec_b.get("kind")),
                            "busy": (round(busy_a, 2), round(busy_b, 2)),
                            "voc": (round(voc_a, 2), round(voc_b, 2)),
                            "room": round(room, 2)}
        return best

    @staticmethod
    def _energy_arr(track):
        """The track's 2 Hz energy curve as a cached numpy array. The
        curve is scanned ~12x per CANDIDATE during selection - list
        indexing in a Python loop made _blend_floor 60% of all scoring
        time (perf audit 2026-07-31)."""
        arr = getattr(track, "_ec_arr", None)
        if arr is None:
            arr = np.asarray(track.row.get("energy_curve") or [],
                             dtype=np.float64)
            track._ec_arr = arr
        return arr

    @staticmethod
    def _blend_floor_grid(cur, out_ss, cand, in_ss,
                          span_s=15.0, carry_s=40.0):
        """Quietest combined moment of each (out, in) seam, walked through
        the stored 2 Hz energy curves (0..1.2, per-track p95-normalized):
        during the ~span_s overlap A ([out_s-span, out_s]) still carries
        the room, so the floor is max(A, B); PAST the seam B is alone - a
        mix-in at the top of a long dying breakdown reads fine as a
        section mean but leaves the room empty for 20s once A is gone
        (measured rms 0.04). Returns a (len(out_ss), len(in_ss)) list of
        floors; 1.0 entries when either curve is missing (no evidence,
        no penalty).

        One broadcasted pass for the whole grid: best_pair used to call
        the scalar walk per combo, ~21 calls per candidate = 60% of all
        selection time (perf audit 2026-07-31). Numerically identical to
        the original loop: int() truncation toward zero matches
        .astype(int64) for the values involved, and the loop's
        break-at-first-overrun equals the prefix mask because the index
        is non-decreasing in tau (verified exact over the library)."""
        no, ni = len(out_ss), len(in_ss)
        ca = Brain._energy_arr(cur)
        cb = Brain._energy_arr(cand)
        if not len(ca) or not len(cb) or not no or not ni:
            return [[1.0] * ni for _ in range(no)]
        tau = np.arange(int(carry_s * 2) + 1, dtype=np.float64) * 0.5
        # B side: (ni, K) - what the incoming curve does after each in_s.
        ib = ((np.asarray(in_ss, dtype=np.float64)[:, None] + tau)
              * 2.0).astype(np.int64)
        valid = ib < len(cb)                   # prefix per row (see above)
        eb = np.where(ib >= 0, cb[np.clip(ib, 0, len(cb) - 1)], 0.0)
        # A side: (no, K) - what the outgoing curve still carries.
        ia = ((np.asarray(out_ss, dtype=np.float64)[:, None]
               - span_s + tau) * 2.0).astype(np.int64)
        ok_a = (tau <= span_s) & (ia >= 0) & (ia < len(ca))
        ea = np.where(ok_a, ca[np.clip(ia, 0, len(ca) - 1)], 0.0)
        # (no, ni, K): the seam floor is min over the VALID taus of
        # max(A, B); invalid taus (past B's end) sit at +inf so they
        # never win the min, and an all-invalid row keeps floor 1.0.
        m = np.maximum(ea[:, None, :], eb[None, :, :])
        m = np.where(valid[None, :, :], m, np.inf)
        floors = np.minimum(m.min(axis=2), 1.0)
        return floors.tolist()

    @staticmethod
    def _blend_floor(cur, out_s, cand, in_s, span_s=15.0, carry_s=40.0):
        """Scalar convenience wrapper over _blend_floor_grid."""
        return Brain._blend_floor_grid(cur, [out_s], cand, [in_s],
                                       span_s, carry_s)[0][0]

    # (near_tempo_veto lived here on 2026-08-13 for a few hours: it vetoed
    # tempo-near partners so a deep-stretch gate trial could find seams at
    # all - selection's s_rate lean meant only 4 in 300 tries otherwise.
    # It went out with the gate it served. If a "deep stretch only" Lab
    # source is ever wanted, it is a dozen lines: for each `cur`, veto
    # every track whose rate_for read sits within 5.5% of it.)

    def _drop_entries(self, track):
        """[(downbeat_s, step)] - every place B's own energy curve actually
        SLAMS up, strongest first. Cached per track (pair-independent, and
        plan_transition asks once per candidate).

        This replaces reading `pre_drop` mix-in hints, which answered a
        different question: mix points are proposed where a track can be
        BROUGHT IN, so they cluster in the intro (61% inside the first
        quarter, none past three quarters) while real drops sit mid-song
        (median 0.47). Scanning the curve is the only way to reach them.

        Guards that the mix-in hints used to provide implicitly, and which
        therefore have to be explicit here:
          - the landing is snapped to a DOWNBEAT (a drop lands on one; a
            cut two beats off it is just a mistake),
          - B keeps _CUT_DROP_RUNWAY_S of track after the entry, so a
            mid-song entry does not arm the next seam moments later,
          - B is not SINGING through the 16-beat run-in, which is the part
            that plays under A.

        And the SHAPE test, which a step ratio alone fails: the landing has
        to be hot (>=_CUT_DROP_MIN_AFTER of the track's own p95), the
        run-up has to be genuinely down, and the KICK has to come back -
        beat power rising is what separates a drop from a swell.
        """
        cached = self._drop_entry_cache.get(track.id)
        if cached is not None:
            return cached
        per = max(track.period_s, 0.3)
        if not ((track.row or {}).get("energy_curve") or []):
            self._drop_entry_cache[track.id] = []
            self._drop_near_cache[track.id] = []
            return []

        out = []
        t = max(_CUT_DROP_SCAN_FROM, 8 * per)
        stop = track.duration_s - _CUT_DROP_RUNWAY_S
        step_s = _CUT_DROP_SCAN_BEATS * per
        while t < stop:
            s = drop_step(track, t)
            # Collected at the TRIAL floor: candidates between the floors
            # and the strict bars feed the near-miss cache below, so the
            # Gate Check trial has something to render.
            if s is not None and s >= _CUT_DROP_TRIAL_MIN_STEP:
                out.append((t, s))
            t += step_s
        # Evaluate each candidate ONCE, then select in two passes - the
        # near-miss tier must never influence which strict entries win
        # their 16-beat neighbourhood.
        evals = []
        for at, s_raw in sorted(out, key=lambda r: -r[1]):
            # The run-in is B playing UNDER A - a vocal there fights A's
            # outro, and the mix-in hints had already been vetted for it.
            if self._vocal_at(track, max(at - 8 * per, 0.0)) >= 0.5:
                continue
            db = track.nearest_downbeat(at)
            if db <= 0 or db >= track.duration_s - _CUT_DROP_RUNWAY_S:
                continue
            # THE LATTICE UNDER THE CUT MUST BE THE TRACK'S METER. Stored
            # grids are SEGMENTED, and a breakdown can carry a garbage
            # segment the scanner itself distrusted while the track-level
            # bpm_conf stays high (found 2026-08-14: Red Lotus, conf
            # 0.99, whose 220-240s segment claims 72.46 bpm at score
            # 0.25 between solid 107.9 segments - the cut's run-in AND
            # its "downbeat" were scheduled on that fiction, and the
            # render gate measured a 159ms sawtooth grid delta as the
            # deck synced to beats that do not exist). A cut hard-
            # schedules on the lattice with no dual for the PLL to hide
            # the lie, so both the landing and the run-in start must sit
            # in a segment whose bpm IS the track's meter. 5% also
            # rejects half/double-time segments, whose bars are the
            # wrong LENGTH for the 16-beat run-in. Structural, not a
            # taste bar: the quantity is the scanner's own segment
            # tempo, and no ear verdict can make a 72-bpm lattice
            # describe 108-bpm music.
            _bad_seg = False
            for _t in (db, max(db - 16 * per, 0.0)):
                for _sg in (track.grid or []):
                    if _sg["start_s"] <= _t <= _sg["end_s"]:
                        if abs(_sg["bpm"] / max(track.bpm, 1e-6)
                               - 1.0) > 0.05:
                            _bad_seg = True
                        break
            if _bad_seg:
                continue
            # JUDGE THE SEAM WHERE IT PLAYS. The scan steps every 4 beats
            # and the landing then snaps to a downbeat, which slides both
            # measurement windows by up to two beats - so the shape has to
            # be re-tested at `db`, not at the scan position that found it.
            lv = drop_levels(track, db)
            if lv is None:
                continue
            # (The kick-return kill that used to run here - beat power
            # rising across the drop - was rated away 2026-08-14: see
            # the note above _CUT_DROP_MIN_AFTER. drop_kick_levels()
            # still measures it for the gateprobe row and the cutdrop
            # test; it just no longer refuses anything.)
            s_db = drop_step(track, db) or s_raw
            evals.append((at, s_raw, db, s_db, lv))

        def _tags(s_db, lv):
            """Which STRICT bars this candidate fails ([] = strict pass).
            Every bar is judged at the SNAPPED DOWNBEAT - the seam that
            plays - not the scan position that found the candidate. The
            first version enforced the step bar at the scan position and
            recorded the downbeat's value in the plan, so entries could
            play at x1.71 against an enforced x1.8 (caught by
            _dj_cutdrop_test, which judges where it plays)."""
            t = []
            if s_db < _CUT_DROP_MIN_STEP:
                t.append("step")
            if lv[1] < _CUT_DROP_MIN_AFTER:
                t.append("after")
            if lv[0] > _CUT_DROP_MAX_BEFORE:
                t.append("before")
            return t

        # One entry per drop: keep the strongest in each 16-beat
        # neighbourhood, else a single slam yields six near-identical
        # candidates and crowds out the track's other drops.
        picked, seen = [], []
        for at, s_raw, db, s_db, lv in evals:
            if any(abs(at - p) <= 16 * per for p in seen):
                continue
            if _tags(s_db, lv):
                continue
            seen.append(at)
            picked.append((db, s_db))
        # Near-miss pass: strict failures inside the trial floors, spaced
        # away from the strict picks AND each other. [(db, step, tags)].
        near, near_seen = [], list(seen)
        for at, s_raw, db, s_db, lv in evals:
            if any(abs(at - p) <= 16 * per for p in near_seen):
                continue
            tags = _tags(s_db, lv)
            if not tags:
                continue
            if lv[1] < _CUT_DROP_TRIAL_MIN_AFTER \
                    or lv[0] > _CUT_DROP_TRIAL_MAX_BEFORE:
                continue
            near_seen.append(at)
            near.append((db, s_db, tags))
        self._drop_entry_cache[track.id] = picked
        self._drop_near_cache[track.id] = near
        return picked

    @staticmethod
    def _stems_refresh(track):
        """Re-stat the stems dir for a track whose load-time has_stems
        stamp says False, and flip the stamp when the files have since
        appeared. The cheap half of fixing the stale-stamp problem: a
        session that outlives a stem render sees the new stems on the
        next planned seam. (True never re-checks - stems are not
        deleted out from under a live night, and the planner's own
        delete action runs in-process and can restamp if it ever needs
        to.)"""
        root = getattr(track, "_music_root", None)
        if not root:
            return False
        try:
            from lib.dj.stems import has_stems as _hs
            if _hs(root, track.id):
                track.has_stems = True
                return True
        except Exception:
            pass
        return False

    def _drop_near_entries(self, track):
        """Near-miss drop entries for the `cut_drop_shape` trial:
        [(downbeat_s, step, [failed strict bars])], strongest first.
        Filled by the same scan as _drop_entries."""
        if track.id not in self._drop_near_cache:
            self._drop_entries(track)
        return self._drop_near_cache.get(track.id, [])

    def _drop_after(self, track, after_s):
        """First DROP MOMENT (energy slams up at a boundary) at/after
        after_s, else the earliest one, or None."""
        from lib.dj.features import drop_moments
        drops = drop_moments(track.sections)
        if not drops:
            return None
        ahead = [t for t in drops if t >= after_s]
        return min(ahead) if ahead else min(drops)

    # -- transition planning -----------------------------------------------------
    def plan_transition(self, cur, cand, meta, after_s=None, arc=None,
                        force_style=None, test_gates=False,
                        allow_benched=False):
        """Resolve style + timing. Returns a plan dict (see build_events).
        `arc` (0..1, optional) couples style choice to the night's energy
        position - valleys breathe, the climb commits, the peak spends the
        spectacle tier. `force_style` (a planner style pin) collapses the
        dice roll to that style - but only if every safety gate left it on
        the menu; a gated-off pin falls back to the normal roll and the
        refusal is recorded in the plan's diag (style_pin.honored).

        `test_gates` (OFFLINE USE - the Seam Lab) lets a pin through a
        gate that is a tuned THRESHOLD rather than a structural
        requirement, so the threshold can be judged by ear instead of
        taken on faith. The override is recorded in diag['gate_test'].
        Never set this on a live night: the gates are there because the
        thresholds are mostly right.

        `allow_benched` (OFFLINE USE - the Lab) admits styles that are off
        the live menu pending an AUDITION rather than on a taste verdict.
        Deliberately separate from `test_gates`: that one crosses a tuned
        threshold on a style the DJ already plays, this one puts a style
        back on the table at all. Live nights leave it False, so a bench
        stays a bench until somebody has actually listened."""
        pair = meta.get("pair") if meta else None
        # The exit-retry (below) hands back an alternative pair whose out
        # may sit before after_s - the budget is a soft preference and
        # best_pair already paid for the earliness. Re-deriving here
        # would silently rediscover the pair the retry just excluded.
        _exit_retry = bool((meta or {}).get("_exit_retry"))
        if pair is None or (not _exit_retry and after_s is not None
                            and pair["out_s"] < after_s):
            pair = self.best_pair(cur, cand, after_s=after_s)
        if pair is None:
            # Last resort: exit on the last downbeat-aligned half minute -
            # but NEVER before the requested after-point (a late entry on
            # a short tail inverted out<in, and the seam fired seconds
            # after the song started). Capped at the same late-exit hard
            # fraction best_pair enforces (2026-08-14): `duration - 35`
            # on a long record lands at 0.9x+ of the track - the exact
            # comedown territory the ceiling exists to refuse - and a
            # fallback that ignores the house rule is a bypass, not a
            # fallback.
            out_fb = max(cur.duration_s - 35.0, cur.duration_s * 0.6)
            out_fb = min(out_fb, EXIT_LATE_HARD_FRAC * cur.duration_s)
            if after_s is not None:
                out_fb = min(max(out_fb, after_s),
                             max(cur.duration_s - 8.0, out_fb))
            # ...but never INTO a dead stretch when a live bookmark
            # exists (2026-08-17, the fade-crater class): the blind
            # half-minute point knows nothing about the curve, and on
            # Take Me Home it landed where the record was already
            # rolling off (census floor 0.078). Rank the track's own
            # mix-out bookmarks by the same exit-life the scorer uses,
            # paying for earliness/lateness on the same taus best_pair
            # does; the blind point competes on equal terms, so when
            # everything is equally dead (or there are no bookmarks)
            # the behavior is exactly the old one.
            def _fb_score(t):
                bud = 1.0
                if after_s is not None:
                    d = t - after_s
                    bud = math.exp(d / BUDGET_TAU_S) if d < 0 \
                        else math.exp(-d / LATE_TAU_S)
                return bud * (EXIT_LIFE_FLOOR + (1.0 - EXIT_LIFE_FLOOR)
                              * self._exit_life(cur, t))
            best_fb = (_fb_score(out_fb), out_fb)
            for o in cur.mix_outs:
                t = o["time_s"]
                # Half the record must have played, and the late-exit
                # ceiling holds here the way it does in the scorer.
                if not (cur.duration_s * 0.5 <= t
                        <= EXIT_LATE_HARD_FRAC * cur.duration_s):
                    continue
                s_fb = _fb_score(t)
                if s_fb > best_fb[0]:
                    best_fb = (s_fb, t)
            out_fb = best_fb[1]
            pair = {"out_s": out_fb,
                    "in_s": cand.mix_ins[0]["time_s"] if cand.mix_ins else 0.0,
                    "out_hint": "blend", "in_hint": "blend", "score": 0.1}
        rate = meta["rate"] if meta else 1.0
        pst = (meta or {}).get("pitch_st", 0)
        # Groove terms for THIS seam (region-aware, evidence-gated None).
        # Steers the style menu below and rides the plan so the live seam
        # self-assessment can compare prediction against measurement.
        rt = seam_rhythm(cur, cand, rate)
        rt_sure = rt is not None and rt.get("conf", 0.0) >= 0.5

        # Style menu, gated by analysis confidence. Weighted by tonight's
        # thumbs (style_fb) AND cross-night learned style taste - which is
        # read FOR THIS SEAM'S CONDITIONS (grid/key/groove/flam), not as a
        # flat per-style verdict, so a technique that struggles on loose
        # grids is still reached for when the grids are precise.
        conds = seam_conditions(cur, cand, rt)
        weights = {k: w * self.style_fb.get(k, 1.0)
                   * self.style_multiplier(k, conds)
                   for k, w in self.theme.style_weights.items()}
        # PERSONA: signature-move bias + the theatricality scale on the
        # whole punchy accent tier. Menu weighting only - every confidence/
        # flam/vocal gate below still zeroes what it zeroes for everyone.
        pers = self.persona
        if pers.style_bias or pers.theatrics != 1.0:
            _accent = ("cut_at_drop", "loop_build",
                       "loop_roll_exit", "echo_out",
                       "stem_drum_swap", "drum_bridge", "acapella_in",
                       "phrase_cut", "spinback_cut", "loop_in")
            for k in list(weights):
                w = weights[k] * pers.style_bias.get(k, 1.0)
                if k in _accent:
                    w *= pers.theatrics
                weights[k] = w
        # Stem styles: accent-tier defaults when the theme dict predates
        # them (hard-gated on rendered stems below, so a default here is
        # inert without tools/dj/dj_stems.py output on disk).
        for k, dflt in (("stem_drum_swap", 0.3), ("acapella_out", 0.2),
                        ("stem_bass_swap", 0.3), ("drum_bridge", 0.2),
                        ("acapella_in", 0.15), ("melody_carry", 0.2),
                        ("phrase_cut", 0.25), ("spinback_cut", 0.15),
                        ("loop_in", 0.2), ("breakdown_swap", 0.2)):
            if k not in weights:
                weights[k] = (dflt * self.style_fb.get(k, 1.0)
                              * self.style_multiplier(k, conds))
        # GATE ATTRIBUTION: every style that gets ZEROED below records which
        # gate did it. Without this the only observable was the outcome -
        # four styles carried 93% of 560 real seams and nothing said whether
        # the elaborate techniques lose the dice roll or never reach the
        # table at all. kill() is the ONLY path that zeroes a weight from
        # here down, so the record is complete by construction; it rides the
        # plan into the `armed` log line for tools/dj/dj_review.py --gates.
        gated = {}
        gated_all = {}              # style -> EVERY reason it was killed

        def kill(styles, reason):
            for k in ((styles,) if isinstance(styles, str) else styles):
                if weights.get(k, 0.0) > 0.0:
                    gated.setdefault(k, reason)
                # EVERY reason, including ones recorded after the weight
                # already hit zero: `gated` keeps the first for the gate
                # report, but an override must see them all or it can let
                # a style through a threshold while a STRUCTURAL kill it
                # never saw still stands (measured: loop_build killed by
                # kick_offset then by no_drop_in_A, overridden on the
                # first, crashed on the missing drop).
                gated_all.setdefault(k, set()).add(reason)
                weights[k] = 0.0
        # Reasons a pin MAY override under test_gates. These are tuned
        # THRESHOLDS - somebody's estimate of where a technique starts to
        # sound bad - and a threshold nobody is allowed to cross can never
        # be shown wrong. Everything absent from this set is structural
        # (no stems, no drop, no breakdown to blend over): overriding
        # those doesn't test a belief, it builds a plan out of parts that
        # aren't there.
        testable = ("grid_conf<0.7",
                    # (cut_needs_grid_conf>=0.8 removed 2026-08-13 - the
                    # gate it named no longer exists, rated away.)
                    "grid_conf<0.5", "downbeat_conf", "kick_offset>28ms",
                    "key_fit<0.8", "anti_streak", "kick_clash",
                    "swing_clash", "meter_clash", "half_time",
                    # ADDED 2026-08-07 after measuring what the stack
                    # costs JOINTLY. long_blend carries the highest weight
                    # in the groove theme (1.7) and reaches the dice on
                    # 19% of seams - not out-rolled, gated out by eleven
                    # conjunctive screens. The three biggest are these,
                    # ~30% of seams between them, and none of them was
                    # reachable from the Seam Lab, so nobody could ever
                    # hear whether their thresholds are right. All three
                    # are TUNED BARS on present evidence, never structural
                    # (each short-circuits when the measurement is
                    # missing: band_clash skips on None, no_beat_power
                    # needs a non-empty `evid`, the kick screen compares
                    # two stored offsets) - so they belong here by the
                    # rule stated above: a threshold nobody may cross can
                    # never be shown wrong. Every band is listed because
                    # the override needs EVERY reason to be testable.
                    "band_clash_low", "band_clash_mid", "band_clash_high",
                    # (stretch>5.5%_risky was listed here 2026-08-12 so it
                    # could be rated at all; 13 trials later - right zero
                    # times - the gate came off the blend family, so the
                    # entry went with it. Making it testable was the whole
                    # point: a threshold nobody may cross can never be
                    # shown wrong, and this one was wrong. Its echo-only
                    # survivor `stretch>5.5%_echo` is NOT listed - it is
                    # unrated, and the render gate says it is load-bearing.)
                    "no_beat_power_A", "no_beat_power_B",
                    "kick_offset>20ms",
                    # ADDED 2026-08-14: cut_at_drop's strict entry-shape
                    # bars (step/after/before), fired only when B holds
                    # a near-miss inside the trial floors - so the
                    # override never invents a drop, it promotes one the
                    # tuned bars refused. "No drop at all" stays
                    # structural as no_real_drop_in_B. Rated the same
                    # day (25 trials, wrong 20): bars moved to the rated
                    # band's floors, kick kill removed. Still listed
                    # because the NEW bars are unrated - the trial now
                    # serves the next unheard band.
                    "cut_drop_shape")

        # RETIRED (2000-pair audit + live record, 2026-08-02):
        # cut_at_drop reached 2% of menus, won 0/2000 rolls, and measured
        # 2/5 rough live - phrase_cut does its job without the drop
        # dependency (choreography kept so old pinned sets degrade
        # politely: a pin shows 'refused (retired)'). bassline_layer
        # (10% of menus, 3 live plays ever) and double_drop (the fx
        # one-shot holdout; loop_build carries the drop spectacle) are
        # REMOVED outright - the kill keeps their old pins refusing
        # politely, the choreography is gone.
        kill(("bassline_layer", "double_drop"), "retired")
        # cut_at_drop is BENCHED FOR AUDITION, not retired (2026-08-12).
        # Its 2026-08-02 retirement rested on three findings and none of
        # them survived re-measurement on this library:
        #   - "won 0/2000 rolls" is arithmetic, not a verdict. At 0.08 of
        #     5.22 it is 1.5% of the dice and reaches the menu on ~2% of
        #     seams, so the EXPECTED wins in 2000 rolls is about 0.6.
        #   - "reached 2% of menus" is one gate: cut_needs_grid_conf>=0.8
        #     on BOTH sides, and the library's median bpm_conf is 0.79 -
        #     the bar sits in the middle of the distribution and costs 76%
        #     of pairs before any other screen runs.
        #   - "2/5 rough live" and the flam gate's "0.247 beats, 4x every
        #     other style" were both measured BEFORE 2026-08-04, when the
        #     grid-phase profiles and the sync-drag fix landed. Flam is the
        #     whole case against this style, and it is the style most
        #     exposed to that defect - it hard-cuts with zero overlap for
        #     the PLL to settle in. Re-measured over 6 renders on the
        #     current stack: median 0.017 beats, max 0.030 - better than
        #     the workhorse blends (0.061-0.068), 0/6 failing the lurch
        #     gate, no clipping.
        # AUDITIONED AND REINSTATED 2026-08-12 (operator, after listening
        # in the Lab): "cut at drop seems fine". Back on the live menu as a
        # RARE accent - see themes.style_weights, and note that it keeps
        # the strictest grid bar of any style (cut_needs_grid_conf>=0.8,
        # both sides) deliberately: the bar was raised for flam, and even
        # though the flam is gone the bar is also what keeps a hard cut on
        # material whose grid is actually trustworthy.
        # spinback_cut retired 2026-08-04: the platter wind-down IS the
        # style, and the user's verdict on the slowdown-into-cut mechanic
        # is "cheesy and overdone" (phrase_cut's optional brake is off for
        # the same reason, via the brake_chance knob). phrase_cut carries
        # the clean-cut job. Old pins refuse politely, as with cut_at_drop.
        kill("spinback_cut", "retired")
        # phrase_cut retired 2026-08-05: "I have never heard a good
        # phrase cut" (user) - after the grid-phase fix put every cut on
        # the audible kick, so this is a verdict on the STYLE, not the
        # timing. The open-format slam (A hard-cut, B full in 40ms) is
        # simply wrong for this library's flowing material; a punctual
        # slam is still a slam. echo_out keeps the leave-without-a-fade
        # job with a musical exit. Old pins refuse politely.
        kill("phrase_cut", "retired")
        # The LOOP-ROLL family retired the same day: "I don't like the
        # loop rolls at all" (user). loop_roll_exit, loop_in and
        # loop_build are all the same stutter-a-shrinking-loop trick worn
        # three ways; the quality gate had also just caught loop_in
        # lurching 7.8 dB. Drop arrivals belong to the nextdrop MOMENT,
        # on the music alone.
        kill(("loop_roll_exit", "loop_in", "loop_build"), "retired")
        # breakdown_swap UN-BENCHED 2026-08-14. Benched 2026-08-04 when
        # its EQ restore stacked on B's drop (9.1 dB slam); the
        # 2026-08-13 rebuild picks the build whose drop actually arrives
        # and clears the restore >=4 beats before it (lurch 3.3 dB
        # median, 0/5 failing). Heard 2026-08-14 via the audition bench:
        # 12 good / 1 passable / 1 bad on 14 Lab seams - the bench's
        # stated exit (measured, then heard) is met, so no kill remains.
        # (The allow_benched hatch stays for whatever gets benched next.)
        # melody_carry + acapella_out BENCHED 2026-08-16 (operator, Lab
        # session: 0 good / 5 passable / 8 bad over 13 seams, and the
        # verdicts track NOTHING measurable - bad at every grid
        # confidence and kick agreement - so this is not an admission
        # problem). The operator's verdict on the mechanic itself: "I
        # just mostly didn't like them... it just felt pointless. For
        # them to work well you need to be purposeful." A lingering
        # vocal/melody tail is a deliberate gesture; a dice roll cannot
        # mean it. Off the live menu until a design gives the tail a
        # PURPOSE (or a revival listen via the Lab's allow_benched
        # hatch, which still plays them). acapella_in stays live:
        # different mechanic (the voice ENTERS and B's own full mix
        # resolves it), unrated in that session, no live thumbs-down.
        if not allow_benched:
            kill(("melody_carry", "acapella_out"), "benched")

        # ANTI-STREAK: one weighted dice roll per seam is blind to what it
        # rolled last time - nights ran long_blend x4 by pure chance and
        # read as monotone. Same style as last seam: halved; as the last
        # TWO: off the menu this round.
        if self.recent_styles:
            last = self.recent_styles[-1]
            if last in weights:
                weights[last] *= 0.5
                if len(self.recent_styles) >= 2 \
                        and self.recent_styles[-2] == last:
                    kill(last, "anti_streak")
        # ARC-COUPLED PACING: valleys lean into the long workhorse blends,
        # the climb favors decisive single-swaps, the peak unlocks the
        # punchy tier. Dynamics get SHAPE instead of uniform dice noise.
        _punchy = ("cut_at_drop", "loop_build",
                   "loop_roll_exit", "echo_out",
                   "stem_drum_swap", "drum_bridge", "acapella_in",
                   "phrase_cut", "spinback_cut", "loop_in")
        moment = False
        if arc is not None:
            if arc < 0.35:
                weights["long_blend"] = weights.get("long_blend", 0) * 1.6
                weights["filter_sweep"] = weights.get("filter_sweep", 0) * 1.2
                for k in _punchy:
                    weights[k] = weights.get(k, 0) * 0.45
            elif arc > 0.7:
                for k in _punchy:
                    weights[k] = weights.get(k, 0) * 1.8
                weights["long_blend"] = weights.get("long_blend", 0) * 0.6
            else:
                weights["bass_swap"] = weights.get("bass_swap", 0) * 1.25
            # ENGINEERED MOMENT: at a genuine peak, once per cooldown, the
            # spectacle styles get a decisive boost - the night gets 2-3
            # landmarks people remember instead of leaving its rarest
            # techniques to dice. Stamped only if one actually wins.
            if arc >= 0.82 and time.time() - self.last_moment_t \
                    > 2400.0 * self.persona.moment_cooldown_x:
                moment = True
                # DECISIVE, not merely boosted. A 4x nudge on a 0.08 base
                # still lost to long_blend's 1.02 nine times in ten, so
                # "engineered moment" produced a workhorse blend at most
                # peaks - 3 double_drops and 4 cut_at_drops in 560 real
                # seams. Once the system has decided THIS is a landmark
                # (a genuine peak, and not for another cooldown), the
                # spectacle tier has to actually win: boost it hard AND
                # stand the workhorses down for this one seam. They are
                # not removed - if every spectacle style is gated off by
                # grid confidence or a missing drop, the blend still
                # carries the seam, which is the correct fallback.
                for k in ("loop_build", "cut_at_drop", "stem_drum_swap",
                          "acapella_out", "acapella_in", "drum_bridge"):
                    weights[k] = weights.get(k, 0) * 12.0
                for k in ("long_blend", "bass_swap", "filter_sweep"):
                    weights[k] = weights.get(k, 0) * 0.25
        low_conf = (cur.bpm_conf < 0.5 or cand.bpm_conf < 0.5)
        # PROFILE-VERIFIED GRID outranks the stale conf scalar
        # (2026-08-05): bpm_conf is the ORIGINAL SCAN's whole-track fit
        # score; the phase profile is a direct per-20s measurement of
        # attacks locking the lattice (a wrong period smears a bucket
        # ~100ms and fails its trust bar, so trusted buckets pin the
        # local tempo too). When BOTH sides are trusted at the seam
        # anchors AND >=60% of each track's buckets pass, the grid is
        # measured-good and conf<0.5 was condemning it on stale
        # evidence - these fades were 17% of ALL seams.
        from lib.dj import beatpower as _bpv
        # THE WAIVER IS THE CORRECTION'S OWN LOOKUP (2026-08-16). This
        # standdown exists because "the kick-true anchors already correct
        # the placement" - so it must ask the EXACT question build_events
        # will ask (region 'out'/'in' at the anchor), never a looser one.
        # On profile-format entries the region argument is ignored and
        # nothing changes; on legacy/prof-less entries the region-default
        # ('mid') lookup could pass on a track-body measurement while the
        # region-specific correction then found nothing and applied 0.0 -
        # gates stood down on evidence the seam never received. Lived
        # consequence 2026-08-16: stem_drum_swap on bpm_conf 0.69/0.66
        # with phase_a_ms 0.0 in the armed log - grids "locked" to 5ms
        # while the rendered kicks flammed 125ms median, self-assessed
        # clean, operator-heard terrible. Same rule as the sync-bias
        # lesson (2026-08-14): never two lookups for one seam.
        _local_ok = (
            _bpv.phase_offset(cur.id, region="out",
                              at_s=pair["out_s"]) is not None
            and _bpv.phase_offset(cand.id, region="in",
                                  at_s=pair["in_s"]) is not None)
        if low_conf and _local_ok \
                and _bpv.profile_coverage(cur.id) >= 0.6 \
                and _bpv.profile_coverage(cand.id) >= 0.6:
            low_conf = False
        # A tempo-clash pair (user-ordered set beyond the stretch range,
        # rate fell back to 1.0) can NEVER beat-match - a "blend" there is
        # two grids sliding past each other. Deliberate fade, always.
        if (meta or {}).get("tempo_clash"):
            low_conf = True
        if (meta or {}).get("a_rate", 1.0) not in (1.0, None)                 and not low_conf and pair.get("beaty", True):
            # dual-bend ramp is implemented in the blend path only
            kill([k for k in list(weights)
                  if k not in ("long_blend", "bass_swap", "filter_sweep",
                               "stem_drum_swap", "acapella_out",
                               "stem_bass_swap", "drum_bridge",
                               "acapella_in", "melody_carry")],
                 "dual_bend_blend_only")
        fade_reason = None
        rolled = False              # did a style dice roll actually happen?
        gate_tested = None          # threshold this seam was let through
        cut_trial_tags = None       # cut_drop_shape trial: bars failed
        # cut_at_drop's vetted entry - set by the gate below when it runs,
        # and NEEDED at plan time, so it cannot live only in that scope.
        cut_pd, cut_step = None, 0.0
        # THE LATTICE UNDER THE OVERLAP MUST BE THE TRACK'S METER
        # (2026-08-14, blend-side twin of the cut entry guard - see
        # off_meter_span). The windows cover where the synced styles
        # actually play both decks: A from just before its exit through
        # the longest dual, B from its run-in through the same. A fade
        # never claims sync, so it is the honest style here - and the
        # fade's own clash carve (fade_a_mid_out) knows these pairs
        # fight. If the night logs show this reason often, the next
        # investment is teaching best_pair to MOVE the anchors off the
        # bad span instead of conceding the blend.
        _off_meter = (off_meter_span(cur, pair["out_s"] - 5.0,
                                     pair["out_s"] + 30.0)
                      or off_meter_span(cand, pair["in_s"] - 5.0,
                                        pair["in_s"] + 30.0))
        if low_conf or _off_meter or not pair.get("beaty", True):
            # No confident grid, a fictitious grid segment under the
            # overlap, or the best seam is BEATLESS on one side: a
            # beat-matched blend there is inaudible as such (or locks a
            # lie) and just smears - deliberate clean fade on the phrase.
            style = "long_fade"
            fade_reason = ("tempo_clash" if (meta or {}).get("tempo_clash")
                           else "grid_conf<0.5" if low_conf
                           else "off_meter_segment" if _off_meter
                           else "beatless_seam")
        else:
            if (cur.downbeat_conf < 0.15 or cand.downbeat_conf < 0.15):
                kill("cut_at_drop", "downbeat_conf")
            # BAR TRUTH IS drum_bridge's PREMISE (2026-08-14). The bridge
            # interleaves two bare rhythm skeletons for 8+ beats with all
            # harmonic cover stripped - beat-level lock can be perfect
            # (rendered: 4ms grid delta) while a wrong downbeat estimate
            # offsets the PATTERNS by a beat or two, snare against kick,
            # which the ear reads as "beat matching isn't working"
            # (operator, Entertain Us -> Organa Baumel: downbeat_conf
            # 0.20 / 0.25, the two least bar-sure tracks imaginable,
            # interleaved naked). Both sides must know where the bar is.
            # 0.3 is a first bar set from that one seam - it is in the
            # `testable` set, so Gate Check can rate it.
            if (cur.downbeat_conf < 0.3 or cand.downbeat_conf < 0.3):
                kill("drum_bridge", "downbeat_conf")
            # Short-dual precision styles (a few bars of overlap, no time
            # for the PLL to settle) demand STRONG grids on both sides -
            # at conf ~0.6 the stored grid itself wobbles 25-50ms
            # (measured on the real library) and the seam audibly flams.
            # LOCAL GRID VERIFICATION outranks the global conf scalar
            # (2026-08-05): a trusted phase-profile bucket at the seam
            # anchor is direct MEASUREMENT that the grid locks the music
            # there - 12+ beats with attack peaks on the lattice and a
            # tight IQR verify both local period and phase (a 0.5% tempo
            # error alone would smear a 20s bucket ~100ms and fail the
            # trust bar). bpm_conf is a whole-track fit score; plenty of
            # tracks score 0.5-0.7 globally while their groove sections
            # are metronomic. Only when the seam's own neighborhood is
            # UNVERIFIED does the conf wall stand. (_local_ok computed
            # above, before the low-conf fade branch, which uses it too.)
            if min(cur.bpm_conf, cand.bpm_conf) < 0.7 and not _local_ok:
                kill(("cut_at_drop", "echo_out",
                      "phrase_cut", "spinback_cut",
                      "loop_in"), "grid_conf<0.7")
                # 2026-08-04, user: "the beats are fundamentally off a
                # lot of the time... I hear a double beat". Root cause:
                # BLENDS were allowed down to conf 0.5, where the stored
                # grid wobbles 25-50ms and the BPM itself can be wrong -
                # a wrong tempo ratio drifts through the whole overlap as
                # a periodic double beat NO PLL authority can hold (the
                # gate measured one such pair at med 158ms / p95 331ms).
                # No overlapped drums on a grid the analyzer doesn't
                # trust: these pairs play the dipped fade or an acapella
                # path until --refine-grids (or a verified local phase
                # profile, above) promotes their tracks.
                kill(("long_blend", "bass_swap", "filter_sweep",
                      "stem_bass_swap", "melody_carry", "breakdown_swap",
                      "stem_drum_swap", "drum_bridge"), "grid_conf<0.7")
            # cut_at_drop's EXTRA grid bar is GONE (2026-08-13, Gate
            # Check). It required bpm_conf>=0.8 on both sides, against the
            # 0.7 its tier uses, on the strength of a median flam of 0.247
            # beats measured over 560 seams (n=4) - four times any other
            # style. Two things killed it:
            #   - that flam was measured BEFORE the 2026-08-04 grid-phase
            #     and sync-drag fixes. Re-measured after: 0.017 beats,
            #     better than the workhorse blends.
            #   - rated by ear on 14 refused seams: the gate was right 3
            #     times (21%), and on the 11 solo-gate trials 3 (27%). The
            #     measurement does not order the verdicts either - wrong at
            #     conf 0.71/0.75/0.73, right at 0.76.
            # It also cost more than anything else this style faced: only
            # 48.6% of the library clears 0.8 and it applied to BOTH sides,
            # ~76% of pairs, with the library's own median sitting at 0.79.
            # cut_at_drop now sits on the same grid_conf<0.7 bar as the
            # rest of the short-dual tier (above) - which, unlike this one,
            # stands down when a verified local phase profile says the grid
            # locks the music at the seam. Fourth per-track scalar rated
            # non-predictive; see the note in the beat-power block.
            # Short-dual styles are exposed to raw GROOVE-OFFSET flam: the
            # PLL is grid-primary, so two tracks whose basslines sit
            # differently against their own grids flam by the OFFSET
            # DIFFERENCE for the whole overlap - measured 170ms deltas on
            # confident grids. (Sync-side compensation was tried and
            # REVERTED - the offset is bass placement, not grid skew.)
            # Gate 28ms (was 35): a 31ms offset rode a long blend as an
            # audible double beat (user-heard, 'Shahoor's Palace' ->
            # 'State the Obvious', 2026-07-22) - percussive flam turns
            # audible ~25ms. loop_roll_exit joined the list: the roll
            # repeats material, which showcases the flam.
            # OBSOLETE WHERE PHASE IS MEASURED (2026-08-05): "nothing can
            # align two kick patterns that sit differently against their
            # own grids" was true when sync could only lock grids - the
            # phase-profile bias now measures where each track's music
            # actually hits and aligns THAT (validated 2-17ms kick-to-
            # kick on rendered seams). kick_offset_s is folded bass
            # placement, and these two screens were benching ~23% of
            # confident pairs on it. They stand only where the seam's
            # local phase is UNMEASURED (_local_ok, same evidence rule
            # as the conf wall).
            if _kick_delta_s(cur, cand, rate) > KICK_SCREEN_CUT_S \
                    and not _local_ok:
                kill(("cut_at_drop", "echo_out",
                      "loop_build", "loop_roll_exit",
                      "loop_in"), "kick_offset>28ms")
            # OFFSET PAIRS -> NO OVERLAPPED DRUMS AT ALL (2026-08-04). The
            # user was condemning perfectly good PAIRINGS because the
            # blend exposed their bassline/kick mismatch - a >28ms offset
            # is structural (the PLL locks transients, EQ carves
            # basslines, but nothing can align two kick patterns that sit
            # differently against their own grids; measured as a constant
            # ~27-30ms grid residual that no settle time closes, and the
            # 2026-07-22 double-beat complaint was a 31ms offset riding a
            # long blend). ~23% of confident pairs. They keep phrase_cut,
            # echo_out, the acapella paths and the fade - the PAIRING
            # survives, played without overlapping drums. The screen sits
            # at 20ms, not the 25-28ms audibility line: stored grids are
            # themselves only ~25ms onset-accurate, and pairs whose stored
            # offset read 26ms measured 27-48ms in rendered audio - the
            # screen needs headroom for its own measurement error.
            # NOT THE PLAIN BLENDS (2026-08-07, Gate Check verdicts). Rated
            # by ear on 16 distinct seams it refused: 13 sounded fine, 3
            # sounded bad - and the three bad ones measured 23.1 / 24.0 /
            # 29.3ms while everything from 42ms to 186ms sounded FINE (bad
            # median 24.0ms vs fine median 44.4ms). Above 30ms: 9 pairs,
            # none bad. The screen's premise - bigger delta is worse - is
            # backwards for this quantity, and in hindsight obviously so:
            # this measures BASS PLACEMENT (median 0.35 beats, see
            # features.measure_kick_offset), so a large delta means one
            # track's bass is offbeat and the two read as separate
            # rhythmic layers, while 20-35ms is the flam zone where two
            # hits fuse into one smeared event. A window there is a
            # hypothesis, not a gate: the 15-30ms band is 3 bad / 4 fine
            # on this data, a coin flip. The stem/mid-running styles keep
            # the screen - unrated, and they overlap whole kits.
            if _kick_delta_s(cur, cand, rate) > KICK_SCREEN_BLEND_S \
                    and not _local_ok:
                kill(("stem_bass_swap", "melody_carry", "breakdown_swap",
                      "stem_drum_swap", "drum_bridge"),
                     "kick_offset>20ms")
            # GROOVE-AWARE STYLE STEERING (rhythm signatures, trusted
            # grids only). Selection already leaned away from these pairs;
            # when one plays anyway (setlist order, thin pool), pick the
            # technique that hides the clash instead of exposing it.
            # KICKS MUST INTERLEAVE ONE-TO-ONE, or drums never overlap
            # (2026-08-04, user: "you're matching the wrong part of the
            # music... I hear a double beat"; measured in rendered audio -
            # a half-time pair blended at 3.2x kick density). A half or
            # double tempo READ means the densities cannot match by
            # construction; contradicting kick patterns mean they do not
            # in practice. Neither is a styling problem - EQ carves
            # basslines, not kick transients - so these are hard vetoes,
            # not weight leans. The pair keeps the fade, the cut and the
            # acapella paths.
            _overlap = ("long_blend", "bass_swap", "filter_sweep",
                        "stem_bass_swap", "melody_carry",
                        "breakdown_swap", "stem_drum_swap", "drum_bridge")
            # THE 5.5% STRETCH WALL IS GONE (2026-08-13, Gate Check).
            # Added 2026-08-05 as a hard plan-time wall - a duplicate
            # Swing Star analysed at 79.7bpm paired with 85bpm at 6.2%
            # stretch and rendered a 187ms-median wander - then made
            # conditional the same evening when the blanket form faded a
            # clean rescue-tier pair the operator heard ("why isn't this
            # resolved"). It refused deep stretch on "risky" material:
            # swing, bpm_conf<0.8, or phase coverage <0.8 on either side.
            #
            # Rated by ear on 13 refused seams: the gate was right ZERO
            # times, including 0 of 6 solo-gate trials. That is the most
            # one-sided verdict any screen here has drawn.
            #
            # Two things had changed under it. The deck wall itself was
            # widened 0.92-1.08 -> 0.90-1.10 on 2026-08-06 on the strength
            # of the beat-matching work, so this gate spent a week capping
            # 72% of deep pairs at 5.5% and making that widening
            # unreachable - the exact blanket-wall failure its own comment
            # warned about. And its risk test was mostly a statement about
            # the ANALYSIS, not the music: profile_coverage returns 0.0
            # when a track was never scanned, so absence counted as risk.
            # The founding case was a track whose BPM was simply WRONG,
            # which grid_conf<0.7 and unstable_phase_* now catch directly -
            # neither existed in this form when the wall was written.
            #
            # What still bounds stretch: deck.set_rate clips to
            # [0.90, 1.10], rate_for caps each side at 6% before that, and
            # s_rate leans hard against depth in selection. Fifth per-track
            # scalar rated non-predictive - see the beat-power block.
            #
            # ECHO_OUT KEEPS THE WALL. Every one of the 13 trials pinned
            # long_blend, so the verdicts are about the OVERLAP family and
            # say nothing about echo - and echo is different physics, the
            # point its original note already made: a brief beat-matched
            # run-in with no dual for the PLL to settle in, the same
            # argument that earns cut_at_drop its own tier. Removing it
            # here too was over-broad and the render gate caught it
            # immediately - echo_out locked at 179ms median / 343ms p95
            # against a 35ms bar, which is the 187ms wander this wall was
            # built for, reproduced. Rate echo's own deep-stretch seams
            # before touching this line.
            if abs(rate - 1.0) > 0.055:
                kill("echo_out", "stretch>5.5%_echo")
            # BEAT POWER (2026-08-04): grid confidence measures whether a
            # lattice FITS; it never asked whether the music actually
            # thumps on it. 38% of this library carries confident grids
            # over diffuse grooves - beat-matching those is matching air,
            # sample-perfect sync and an audible mess ("the beats are
            # fundamentally off... I hear a double beat"). Overlapped
            # drums require BOTH tracks to concentrate low-band attack on
            # their own beats (lib/dj/beatpower.py, measured from the raw
            # audio). Unmeasured tracks pass - the scan fills in.
            from lib.dj import beatpower as _bp
            # REGION-AWARE (2026-08-05): the whole-track score samples
            # ~30s at the MIDPOINT, but a blend overlaps A's EXIT with
            # B's INTRO - a track that thumps where the seam actually
            # plays was being benched by the wrong 30 seconds (this kill
            # alone blocked 36% of all seams). Judge each side by its
            # seam-relevant region's low-band score when measured; either
            # instrument clearing the bar is evidence of a real beat bed
            # there, and the phase-corrected sync + downstream gates
            # handle the rest.
            # ASYMMETRIC BARS (2026-08-05): B becomes the mix's
            # foundation - its intro must carry a real groove
            # (BLEND_MIN). A is LEAVING - it only needs enough pulse to
            # hand off, and "A dissolves while B's groove takes over" is
            # textbook mixing; only a truly beatless exit forces the
            # fade (BLEND_MIN_EXIT). The symmetric 1.5 bar was the
            # single largest blend killer (22% of all seams) on exits
            # that were merely softening, not beatless.
            # Region BY POSITION (2026-08-05): the 'out'/'in' band scores
            # are measured at the PRIMARY mix points, but an urgent
            # (skip) exit leaves mid-track - judging a mid-groove exit
            # by the outro's diffuseness faded 7 of 9 skip seams in one
            # sitting ("not beat matched at all" - it was a fade). When
            # the actual anchor is far from the measured point, the
            # track BODY ('mid', measured at the midpoint) is the
            # closer evidence.
            def _reg_for(track, at_s, kind):
                try:
                    pts = (track.mix_outs if kind == "out"
                           else track.mix_ins)
                    ref = pts[0]["time_s"] if pts else None
                except Exception:
                    ref = None
                return kind if (ref is not None
                                and abs(at_s - ref) <= 45.0) else "mid"
            _reg_a = _reg_for(cur, pair["out_s"], "out")
            _reg_b = _reg_for(cand, pair["in_s"], "in")
            # UNSTABLE PHASE = NO OVERLAPPED DRUMS (2026-08-05): a PATCHY
            # profile - some trusted buckets amid wild ones (IQRs >100ms)
            # - means the track's phase WANDERS. One clean anchor bucket
            # can open the conf wall while the material drifts a quarter
            # beat mid-blend (gate-measured: Swing Star -> Power Core,
            # 187ms median wander, p95 357ms - no PLL holds swing).
            # Coverage <0.6 with a profile present is that fingerprint.
            for t, side in ((cur, "A"), (cand, "B")):
                _cov = _bp.profile_coverage(t.id)
                if 0.0 < _cov < 0.6:
                    kill(_overlap, f"unstable_phase_{side}")
            # +/-10s off the boundary, matching the legacy windows
            # (regions["in"]/["out"] center at primary +/-10): out_s and
            # in_s are SECTION BOUNDARIES, so the profile read exactly
            # there is half the wrong section - A's outro it never plays
            # into, B's quiet bar before its drums arrive. Sampling at
            # the boundary inflated A-side refusals 5.9% -> 11.9% on the
            # paired 700-seam run; the deck's audible material during
            # the overlap is [out_s-15, out_s] and [in_s, in_s+30].
            for t, side, reg, _at, bar in (
                    (cur, "A", _reg_a, pair["out_s"] - 10.0,
                     _bp.BLEND_MIN_EXIT),
                    (cand, "B", _reg_b, pair["in_s"] + 10.0,
                     _bp.BLEND_MIN)):
                # SCORE THE SEAM'S OWN POSITION (2026-08-12). The labeled
                # regions are three 30s windows - the midpoint, and one
                # around each PRIMARY mix point - and _reg_for falls back
                # to the MIDPOINT whenever the seam lands >45s from the
                # primary. Measured over 700 planned seams that fallback
                # fired on 35.4%, judging the incoming deck by a stretch
                # of music the blend never touches. It is not a harmless
                # approximation: 46.7% of tracks fail the B bar at their
                # midpoint vs 27.2% at their entry, and 35.3% straddle
                # the bar between regions - beat power is LOCAL, exactly
                # as phase turned out to be on 2026-08-04, and this is
                # the same fix (a dense ~20s profile, read at the seam).
                # The dense value stands ALONE when present: max()-ing it
                # with the whole-track scalar would re-admit the very
                # midpoint reading this replaces, letting a track with a
                # thumping body enter on a dead one.
                pw = _bp.power_at(t.id, _at)
                if pw is not None:
                    evid = [pw]
                else:
                    bs = _bp.band_scores(t.id, region=reg) or {}
                    evid = [v for v in (bs.get("low"),
                                        _bp.scores().get(t.id))
                            if v is not None]
                if evid and max(evid) < bar:
                    # NEITHER BEAT-POWER BAR GOVERNS THE PLAIN BLENDS.
                    # A-side (2026-08-07, Gate Check): 6 fine / 2 bad on 8
                    # refused seams (p~0.001 against a 20%-false-alarm
                    # baseline), bad ones (1.01, 0.98, 1.01) INSIDE the
                    # range of the fine ones - no threshold sorts them. A
                    # is the deck LEAVING: it only has to hand off, and
                    # the EQ staging carves its low end at the swap.
                    # B-side (2026-08-13, Gate Check, rated the morning
                    # after the dense phase-corrected profile made the
                    # measurement honest at the seam's own position): 12
                    # fine / 2 bad on 14 refused seams (p~1e-7; the strict
                    # solo-only tally is 2 wrong / 1 right on 3 solo
                    # trials - thin, the operator chose to act on the
                    # joint count). Same non-discrimination fingerprint:
                    # bad ones measured 1.16 and 1.25, fine ones 0.92-1.28.
                    # Seams at 0.92 - barely any on-beat dominance -
                    # blended fine; every refused seam with a DEEP B entry
                    # (74-280s) sounded fine (0 bad of 8), so low on-beat
                    # dominance mid-track (offbeat basslines, rolling
                    # grooves) is not the defect this bar assumed. Third
                    # scalar in a row rated non-predictive (band_clash,
                    # kick_offset, now beat power) - see the memory rule:
                    # per-track scalars keep failing to predict what the
                    # ear objects to.
                    # The stem styles keep BOTH bars: they run whole kits
                    # together, and no stem seam has been rated. echo's
                    # own B bar (below) is separate and also unrated.
                    kill(tuple(s for s in _overlap
                               if s not in ("long_blend", "bass_swap",
                                            "filter_sweep")),
                         f"no_beat_power_{side}")
                # echo_out BEAT-MATCHES ITS RUN-IN into B, but had no
                # B-side beat requirement at all - syncing into a
                # beatless track locks onto NOTHING (gate-measured twice
                # at 116ms med: Oh Joy, power 1.03 = 'no beat to match'
                # by the instrument's own definition). Echo's dual is
                # brief, so the bar sits below the blend bar.
                if side == "B" and evid and max(evid) < 1.15:
                    kill("echo_out", "echo_needs_b_beat")
            # A's EXIT MUST CARRY THROUGH THE OVERLAP (2026-08-05). The
            # blend-floor scan takes max(A, B), so B's rise HIDES a
            # collapsing A - the rendered result is a hole-then-slam
            # (measured: Negev -> Neptunes bass_swap, A's pre-out
            # breakdown fell 25 dB mid-overlap while B surged, a 6.6 dB
            # mix lurch the floor scan waved through). If A's own energy
            # curve dies to under 30% of its exit-start level inside the
            # overlap span, the pair gets the fade - which is the
            # arrangement A itself is performing.
            _ea = Brain._energy_arr(cur)
            _i0 = int((pair["out_s"] - 15.0) * 2)
            _i1 = int(pair["out_s"] * 2)
            if len(_ea) and 0 <= _i0 < _i1 <= len(_ea):
                _seg = _ea[_i0:_i1]
                _start = float(np.median(_seg[:6]))
                if _start > 0.15 and len(_seg) > 6 \
                        and float(np.min(_seg[6:])) < 0.3 * _start:
                    kill(_overlap, "a_exit_collapses")
            # ... and the exit must be a steady GROOVE. Measured 2026-08-05
            # (Negev -> Neptunes): the pair scan parked out_s inside A's
            # BREAKDOWN - stored broadband energy only sags 5 dB there, so
            # the collapse screen above stays quiet, but the section has
            # rhythm_density halving and the style's EQ migration guts
            # what's left (rendered: A -25 dB mid-overlap, then B slams
            # in +6.6 dB). An overlap needs a drum bed on BOTH sides for
            # its whole span; a breakdown exit hands the room over with a
            # lurch no choreography can hide.
            _sx = cur.section_at(pair["out_s"] - 4.0) or {}
            if _sx.get("kind") == "breakdown" \
                    or (_sx.get("energy") or 1.0) < 0.35:
                kill(_overlap, "a_exits_through_breakdown")
            # BAND-AWARE STYLE ELIGIBILITY (user, 2026-08-04: "different
            # portions of the frequency bands can also mismatch - that's
            # why different mix styles exist"). Each style overlaps a
            # known set of bands; in any overlapped band, RHYTHMIC over
            # DIFFUSE is the smear the ear rejects (one track articulates
            # the lattice, the other washes over it). Both-rhythmic is
            # normal DJ material (hats over hats) - the alignment gates
            # above govern that - and both-diffuse is an ambient wash.
            # Activates per-track as the --bands scan lands; unmeasured
            # bands pass.
            # STAGED GEOMETRY UPDATE (2026-08-05): the EQ-carved blends
            # no longer RUN mids together - B's mid rides a 0.25-0.3
            # shelf (~-11 dB) through the dual and the swap is a HANDOFF,
            # so a rhythmic-vs-diffuse mid mismatch never stacks
            # audibly. Their true simultaneous band is the HIGH end
            # (B's highs enter half-open and migrate mid-blend). The
            # mid clash still governs the styles that genuinely run
            # mids together: melody_carry (carries A's melody stem over
            # B) and the stem drum paths (full-bodied kits). This kill
            # was blocking 15% of ALL seams on a mismatch the staging
            # already hides.
            # BAND CLASH NO LONGER GOVERNS THE PLAIN BLENDS (2026-08-07,
            # Gate Check verdicts). long_blend/bass_swap/filter_sweep were
            # screened on the HIGH band alone; rated by ear on 11 solo
            # seams it refused, 9 sounded fine and 2 sounded bad (p~2e-5).
            # The measurement does not discriminate anywhere near the bar:
            # a 0.24-vs-8.34 mismatch was fine and a 1.00-vs-3.08 was bad,
            # so loosening 1.5/1.2 would not sort them either - there is no
            # threshold on this quantity that matches the ear. Their true
            # simultaneous band is the high end and B's highs enter half
            # open (see the staging note below), which is evidently enough.
            # The styles that genuinely run mids or full kits together keep
            # the screen; those were never rated and their premise is
            # different.
            _style_bands = {
                # stem_bass_swap enters WHOLE minus the bass stem - its
                # mids are fully open from bar one, so the mid clash
                # absolutely applies (dropping it admitted a pair that
                # rendered a low-end hole, spectral 2026-08-05).
                "stem_bass_swap": ("mid", "high"),
                "melody_carry": ("mid", "high"),
                "breakdown_swap": ("mid", "high"),
                "stem_drum_swap": ("low", "mid", "high"),
                "drum_bridge": ("low", "mid", "high"),
            }
            # A exits, B enters: judge each track's bands in the REGION
            # the blend actually overlaps.
            ba_ = _bp.band_scores(cur.id, region=_reg_a)
            bb_ = _bp.band_scores(cand.id, region=_reg_b)
            if ba_ and bb_:
                for st_, bands_ in _style_bands.items():
                    for bd in bands_:
                        va, vb = ba_.get(bd), bb_.get(bd)
                        if va is None or vb is None:
                            continue
                        hi_, lo_ = max(va, vb), min(va, vb)
                        if hi_ >= BAND_CLASH_HI and lo_ < BAND_CLASH_LO:
                            kill(st_, f"band_clash_{bd}")
                            break
            # SWING CLASH RUNS ON ITS OWN CONFIDENCE, IN BOTH BRANCHES
            # (2026-08-14). This screen used to live only in the
            # UNTRUSTED-signature branch below - a seam whose rhythm was
            # confidently measured skipped it entirely, so the better
            # the measurement the less it was used. Dunes -> Beam Me Up
            # (swing 0.534 vs 0.606, delta 0.072 at swing_conf 0.994,
            # overall conf 0.75 -> trusted branch) sailed through and
            # the operator rated it "pretty bad" live. And the old
            # handling BOOSTED stem_drum_swap x2 on clash, on the theory
            # it "removes one percussion bed" - its own event builder
            # says otherwise: B enters on its drum stem UNDER A's full
            # mix, and after the swap A's drum stem rides OVER B's full
            # mix - two beds on both sides of the seam, the most exposed
            # style there is. Same failure shape as drum_bridge's
            # key-clash boost, retired the same day: a steer built on
            # theory, rated wrong the first time its flagship case
            # played. Now every kit-overlay style is DAMPED on a
            # confident swing clash; swing_conf gates it (the swing
            # measurement carries its own confidence, independent of
            # the overall rhythm conf).
            if (rt is not None and rt.get("swing_conf", 0.0) >= 0.5
                    and rt["swing_delta"] > 0.055):
                for k in ("long_blend", "filter_sweep", "loop_roll_exit",
                          "drum_bridge", "stem_drum_swap"):
                    weights[k] = weights.get(k, 0.0) * 0.3
            if rt_sure:
                if abs(rt.get("mult", 1.0) - 1.0) > 1e-6:
                    kill(_overlap, "tempo_multiple_read")
                if rt["kick_agreement"] < 0.35:
                    kill(_overlap, "kick_clash")
                elif rt["kick_agreement"] < 0.6:
                    # KIT-OVERLAY DAMP IN THE 0.35-0.6 BAND (2026-08-16,
                    # Lab verdicts): above the kill bar but the kit-
                    # exposing styles rated mean 0.25 there (4 seams -
                    # drum_bridge bad at 0.49 and 0.59, stem_bass_swap
                    # bad at 0.60). n=4 is a lean, not a wall - same
                    # shape as the swing-clash damp above: these styles
                    # EXPOSE both kits, so they compete at 0.3x instead
                    # of being handed pairs whose kicks half-disagree.
                    for k in ("stem_drum_swap", "drum_bridge",
                              "stem_bass_swap"):
                        weights[k] = weights.get(k, 0.0) * 0.3
            else:
                # No trusted signatures: screen on what IS stored. Tempo
                # classes an octave apart force a multiple read, and a
                # 1.8x kick-density gap doubles beats even at 1:1 tempo.
                ratio = cur.bpm / max(cand.bpm, 1e-6)
                if ratio > 1.43 or ratio < 0.70:
                    kill(_overlap, "tempo_multiple_read")
                da, db_ = cur.rhythm_density, cand.rhythm_density
                if da > 0 and db_ > 0 and max(da / db_, db_ / da) >= 1.8:
                    kill(_overlap, "kick_density_mismatch")
                fl = rt.get("flam_ms") if rt is not None else None
                if fl is not None and 15.0 <= fl <= 80.0:
                    # Machine-gun near-misses: the short punchy styles
                    # expose them raw (same reasoning as the groove-offset
                    # gate above, measured one level finer).
                    for k in ("cut_at_drop", "echo_out", "loop_build"):
                        weights[k] = weights.get(k, 0.0) * 0.3
            # cut_at_drop ENTERS AT B'S REAL DROP (2026-08-12, rebuilt).
            # It used to require a `pre_drop` MIX-IN hint and take the
            # highest-scoring one, trusting the section label to mean a
            # slam. Both halves were wrong, and measurement separated them:
            #   PRECISION - the label is the segmenter's opinion. 13% of
            #     labelled drops measure no audible step; the median is
            #     x1.57, "a strong lift".
            #   RECALL - mix-in points answer "where can this track be
            #     brought IN", so 61% sit in the first quarter of the song
            #     and none past three quarters, while the real >=x2.0 steps
            #     have a median position of 0.47. The style could not reach
            #     a mid-song drop at all, so it settled for an intro build.
            # Together: the operator heard it "giving me songs that aren't
            # dropping". _drop_entries scans B's own curve instead, and
            # carries the downbeat/runway/vocal guards the mix-in hints
            # used to supply implicitly.
            _entries = self._drop_entries(cand)
            if _entries:
                cut_pd, cut_step = _entries[0]
            elif self._drop_near_entries(cand):
                # B has a candidate drop inside the trial floors that the
                # strict shape bars refused - a TUNED refusal, so it gets
                # its own testable reason instead of hiding inside the
                # structural "no drop at all". The trial override below
                # supplies the near-miss entry when this is crossed.
                kill("cut_at_drop", "cut_drop_shape")
            else:
                kill("cut_at_drop", "no_real_drop_in_B")
            # (loop_roll_exit rolls the 16 beats just before out_s - its
            # window is derived, so no after_s restriction needed here.)
            loop_ok = any(l["start_s"] < pair["out_s"] for l in cur.loops)
            if not loop_ok:
                kill("loop_roll_exit", "no_loop_before_exit")
            # loop_build stutters A into its own drop as a tension build,
            # then B cuts in on the drop. Needs a drop in A to build toward.
            if self._drop_after(cur, pair["out_s"] - 8 * cur.period_s) is None:
                kill("loop_build", "no_drop_in_A")
            # STEM STYLES need pre-rendered stems on disk (dj_stems.py).
            # The stamp is load-time; a False re-checks the disk here so
            # a mid-session stem render unlocks the styles on the very
            # next seam instead of after a restart (2026-08-12: 11
            # tracks rendered mid-session stayed False all night). Only
            # False tracks pay the stat, and True never re-checks.
            a_stems = getattr(cur, "has_stems", False) \
                or self._stems_refresh(cur)
            b_stems = getattr(cand, "has_stems", False) \
                or self._stems_refresh(cand)
            if not (a_stems and b_stems):
                kill(("stem_drum_swap", "stem_bass_swap", "drum_bridge"),
                     "no_stems")
            if not a_stems:
                kill(("acapella_out", "melody_carry"), "no_stems")
            if not b_stems:
                kill("acapella_in", "no_stems")
            # drum_bridge runs BOTH grids fully exposed for bars -
            # precision-tier grid bar applies.
            if min(cur.bpm_conf, cand.bpm_conf) < 0.7:
                kill("drum_bridge", "grid_conf<0.7")
            # The key-sensitive stem premises share one PRECISE key fit
            # (chroma-aware, corrected for any pitch-shift rescue).
            kf_precise = None
            if any(weights.get(k, 0.0) > 0.0 for k in
                   ("acapella_out", "acapella_in", "melody_carry")):
                kf_precise = camelot_compat(cur.camelot, cand.camelot)
                sc = chroma_key_compat(
                    getattr(cur, "chroma", None), cand.chroma,
                    12.0 * math.log(max(rate, 1e-6)) / math.log(2.0)
                    if stretch_engine_name() == "vari" else float(pst))
                if sc is not None:
                    kf_precise = sc
            if weights.get("acapella_out", 0.0) > 0.0:
                # acapella_out's premise: A actually SINGS around its exit
                # (the tail IS its vocal riding B's instrumental), B stays
                # vocal-free under it, and the keys sit close - an exposed
                # voice over an off-key bed is the worst clash there is.
                tail_v = self._vocal_span_max(
                    cur, pair["out_s"] - 16.0, pair["out_s"] + 16.0)
                b_under = self._vocal_span_max(
                    cand, pair["in_s"] + 20.0, pair["in_s"] + 60.0)
                if tail_v < 0.35 or b_under > 0.5 or kf_precise < 0.8:
                    kill("acapella_out", "acapella_premise")
            if weights.get("acapella_in", 0.0) > 0.0:
                # THE MIRROR: B's intro actually sings (its isolated vocal
                # rides A's bed before the full mix lands), A's outro
                # stays out of its way, keys tight.
                b_in_v = self._vocal_span_max(
                    cand, pair["in_s"], pair["in_s"] + 30.0)
                a_out_v = self._vocal_span_max(
                    cur, pair["out_s"] - 20.0, pair["out_s"] + 4.0)
                if b_in_v < 0.35 or a_out_v > 0.35 or kf_precise < 0.8:
                    kill("acapella_in", "acapella_premise")
            if weights.get("melody_carry", 0.0) > 0.0 \
                    and kf_precise < 0.8:
                # A's melodic bed sustains under B - key fit IS the premise.
                kill("melody_carry", "key_fit<0.8")
            # (The KEY-CLASH x2.5 BOOST for drum_bridge lived here from
            # its birth until 2026-08-14 - "both tracks strip to
            # percussion while the harmony resets", so clash-key pairs
            # were STEERED into bridges. The theory's flagship seam
            # (Entertain Us -> Organa Baumel, key fit 0.30, four camelot
            # steps) was rated "fairly bad" by the operator the first
            # time it played live. The theory ignores what the loop-
            # layer post-mortem already measured: demucs drum stems are
            # bleedy - tonal content rides them, so the old key keeps
            # sounding through the bridge and the alien key lands on its
            # tail. The boost is gone: drum_bridge stays AVAILABLE at
            # any key the other gates allow, it just competes on merit
            # instead of being handed the worst-key pairs. Whether a
            # hard key FLOOR is also warranted is a Lab question - one
            # seam does not set a bar. spinback_cut's twin boost went
            # with it (retired style, the weight was already zero).
            # breakdown_swap needs the sections to exist: A must have a
            # breakdown ahead of the exit region, B a build to enter on.
            bd_a = next((s for s in (cur.sections or [])
                         if s["kind"] == "breakdown"
                         and s["end_s"] > (after_s or 0.0)), None)
            # THE BUILD HAS TO LEAD SOMEWHERE (2026-08-12). This used to
            # take the FIRST build section in B and nothing checked that a
            # drop followed it. Measured over 12 legal pairs on this
            # library, the first build sat a MEDIAN 318 beats from the
            # nearest drop and 4 of 12 had no drop after it at all - so the
            # style's whole premise ("ride A's breakdown carrying B's
            # build, the drop that follows is the payoff") was not what the
            # code did, and the payoff usually did not exist. Pick the
            # build whose drop arrives soonest instead, and require it
            # inside a musical window: the drop must land within the blend
            # or just past it, or there is nothing to build toward and this
            # is just a mid-carving bass_swap with extra steps.
            # (Library check: 72% of tracks have both a build and a drop;
            # best-build gets the drop within 32 beats on 65% of them,
            # against 43% for the first build.)
            _bl_max = _BDSWAP_DROP_MAX_BEATS * max(cand.period_s, 1e-6)
            _bl_min = _BDSWAP_DROP_MIN_BEATS * max(cand.period_s, 1e-6)
            bl_b, b_drop_s = None, None
            _drops_b = drop_moments(cand.sections)
            for _s in (cand.sections or []):
                if _s["kind"] != "build":
                    continue
                _ahead = [d for d in _drops_b
                          if _s["start_s"] + _bl_min <= d
                          <= _s["start_s"] + _bl_max]
                if not _ahead:
                    continue
                if bl_b is None or min(_ahead) - _s["start_s"] < \
                        b_drop_s - bl_b["start_s"]:
                    bl_b, b_drop_s = _s, min(_ahead)
            if bd_a is None or bl_b is None:
                kill("breakdown_swap", "no_breakdown_or_build")
            else:
                # VIOLENT DYNAMICS stay solo (2026-08-04): a build whose
                # crescendo jumps >~8 dB, or a breakdown that collapses
                # that far, slams the blend however clean the sync is
                # (gate-measured 9.1 dB blend step vs 5.3 solo; an A-side
                # breakdown probed at an 11x collapse). The style wants a
                # gentle hollow and a rising build, not cliffs.
                def _dyn_ratio(track, sec_):
                    curve = (track.row or {}).get("energy_curve") or []
                    i0 = int(sec_["start_s"] * 2)
                    i1 = min(int(sec_["end_s"] * 2) + 1, len(curve))
                    if not curve or not (0 <= i0 < i1):
                        return 1.0
                    seg = curve[i0:i1]
                    return max(seg) / max(min(seg), 1e-4)
                if _dyn_ratio(cand, bl_b) > 2.5                         or _dyn_ratio(cur, bd_a) > 2.5:
                    kill("breakdown_swap", "violent_dynamics")
            weights["long_fade"] = 0.0
            # ECHO IS PUNCTUATION, NOT A WORKHORSE (2026-08-05, user:
            # "you are way overusing echo out"). When every blend is
            # gated, the menu used to hold only echo_out - it won those
            # seams by DEFAULT, 40% of the night. For blend-less menus
            # the deliberate fade rejoins the dice at 3x echo's weight:
            # most such seams take the fade the material asked for, and
            # the echo lands as an occasional accent.
            if not any(weights.get(k, 0) > 0 for k in
                       ("long_blend", "bass_swap", "filter_sweep",
                        "stem_bass_swap", "melody_carry")):
                # SECOND-CHANCE EXIT (2026-08-12). a_exit_collapses and
                # a_exits_through_breakdown are properties of the chosen
                # out POINT alone - the pair scorer optimizes overall
                # fit, so it happily parks the exit on a breakdown and
                # thereby kills every overlapped style, dropping the
                # seam to a dice fade even when A has other exits whose
                # energy carries. Measured on the progressive-cluster
                # pool: exit-anchored kills were the FIRST reason on
                # ~22% of dice-fade seams. When the workhorse blends
                # died to exit-anchored reasons ONLY (any whole-track
                # reason - beat power, grid, kick pattern - means a new
                # exit cannot save the pair), re-ask best_pair for its
                # best pair WITHOUT that exit and re-plan once. The
                # _exit_retry flag bounds the recursion at one level and
                # keeps the retried pair from being re-derived (see the
                # pair block at the top).
                _exit_only = {"a_exit_collapses",
                              "a_exits_through_breakdown"}
                if not _exit_retry and all(
                        gated_all.get(s) and gated_all[s] <= _exit_only
                        for s in ("long_blend", "bass_swap",
                                  "filter_sweep")):
                    alt = self.best_pair(cur, cand, after_s=after_s,
                                         exclude_out_s=pair["out_s"])
                    if alt is not None:
                        # The alt exit must itself SURVIVE the two
                        # exit-anchored kills that triggered the retry,
                        # or the trade gains nothing and loses the
                        # scorer's anchor: Wild Window's 240.9s exit was
                        # breakdown-killed, the retry took 178.8s - whose
                        # section energy 0.28 fails the same 0.35 bar -
                        # so the blends died again and the fade played
                        # the dying groove best_pair had just paid to
                        # avoid (rendered floor 0.16 vs 0.23).
                        _sx2 = cur.section_at(alt["out_s"] - 4.0) or {}
                        if _sx2.get("kind") == "breakdown" \
                                or (_sx2.get("energy") or 1.0) < 0.35:
                            alt = None
                        else:
                            _ea2 = Brain._energy_arr(cur)
                            _j0 = int((alt["out_s"] - 15.0) * 2)
                            _j1 = int(alt["out_s"] * 2)
                            if len(_ea2) and 0 <= _j0 < _j1 <= len(_ea2):
                                _sg2 = _ea2[_j0:_j1]
                                _st2 = float(np.median(_sg2[:6]))
                                if _st2 > 0.15 and len(_sg2) > 6 \
                                        and float(np.min(_sg2[6:])) \
                                        < 0.3 * _st2:
                                    alt = None
                    if alt is not None:
                        # rate is read as meta["rate"] behind an `if
                        # meta` truthiness guard - the retry meta is
                        # never empty, so carry the default explicitly.
                        _m = dict(meta or {}, pair=alt, _exit_retry=True)
                        _m.setdefault("rate", 1.0)
                        plan = self.plan_transition(
                            cur, cand, _m,
                            after_s=after_s, arc=arc,
                            force_style=force_style,
                            test_gates=test_gates,
                            allow_benched=allow_benched)
                        plan.setdefault("diag", {})["exit_retry"] = {
                            "from_out_s": round(float(pair["out_s"]), 3),
                            "to_out_s": round(float(alt["out_s"]), 3)}
                        return plan
                # THE FADE IS A LAST RESORT, NOT A DEFAULT (2026-08-14,
                # operator: "it should mostly be last resort picks").
                # The blend-less check above only watches the five
                # BLEND styles - when a cut, a stem style or a bridge
                # survived the gates, the fade still re-entered at 0.8
                # against their 0.2-0.3 accent weights and won those
                # menus ~4:1. Now: if ANY synced style beyond echo is
                # still standing, the fade re-enters at HALF weight and
                # the surviving styles get real odds; only when the
                # menu is echo-or-nothing does the 2026-08-05 2:1
                # fade:echo rule apply unchanged (echo as punctuation,
                # the fade carrying what the material asked for).
                _other_synced = any(
                    weights.get(k, 0) > 0 for k in
                    ("cut_at_drop", "stem_drum_swap", "drum_bridge",
                     "acapella_out", "acapella_in", "breakdown_swap",
                     "loop_build", "loop_roll_exit"))
                weights["long_fade"] = (1.0 if _other_synced else 2.0) \
                    * max(weights.get("echo_out", 0.0), 0.4)
            # GATE UNDER TEST: a pin refused only by a tuned threshold is
            # let through so the threshold itself can be rated. Structural
            # refusals (no stems, no drop, retired) still stand.
            reasons = gated_all.get(force_style) or set()
            if test_gates and force_style \
                    and weights.get(force_style, 0.0) <= 0.0 \
                    and reasons and all(r in testable for r in reasons):
                weights[force_style] = 1.0
                gate_tested = ", ".join(sorted(reasons))
                gated.pop(force_style, None)
                if force_style == "cut_at_drop" and cut_pd is None \
                        and "cut_drop_shape" in reasons:
                    # The trial crossed the shape bars, so the entry the
                    # bars refused is what plays: B's strongest near-miss.
                    # Its failed-bar tags ride diag (stamped with
                    # gate_test below) so per-bar tallies fall out of the
                    # ratings log.
                    _near = self._drop_near_entries(cand)
                    if _near:
                        cut_pd, cut_step, _tags = _near[0]
                        cut_trial_tags = list(_tags)
            menu = [(s, w) for s, w in weights.items() if w > 0]
            if force_style == "long_fade":
                # Operator pinned the deliberate fade - always available.
                style = "long_fade"
                fade_reason = "style_pin"
            elif force_style and weights.get(force_style, 0.0) > 0.0:
                # Style pin: only reachable when every gate above left it
                # on the menu - safety gates outrank the pin.
                style = force_style
                rolled = True
            elif menu:
                styles, ws = zip(*menu)
                style = self.rng.choices(styles, weights=ws, k=1)[0]
                rolled = True
            else:
                # EVERY style gated -> the deliberate fade, nothing else.
                # The old fallback here was `bass_swap` - the MOST
                # demanding overlap style handed to exactly the pairs
                # that failed every gate. User-caught in Beat Check
                # (2026-08-05, Birds Mind -> No Ending, conf 0.665):
                # min-conf 0.5-0.7 kills all blends AND all cuts, this
                # branch zeroes long_fade for the roll, no stems -> empty
                # menu -> unmatched bass_swap, gates bypassed. ~9% of the
                # library sits in that conf band, so roughly a sixth of
                # all seams could fall through - and tightening the walls
                # (2026-08-04) enlarged the trapdoor instead of closing
                # it. A pair nothing is safe for gets the fade that was
                # DESIGNED for unmixable pairs.
                style = "long_fade"
                fade_reason = "all_styles_gated"

        # NEVER BLEND TWO SUNG PASSAGES: the swap-slot scan protects the
        # swap beat, but if A exits THROUGH a vocal passage while B enters
        # on one, the whole overlap is two voices fighting. With stems on
        # A a blend can DUCK A's vocal stem instead of surrendering the
        # seam to a fade (vocal_over_vocal was a top logged fade reason);
        # without stems the dipped fade stays the graceful answer.
        duck_vocal = False
        if style != "long_fade":
            sa_v = (cur.section_at(pair["out_s"] - 1.0) or {})
            sb_v = (cand.section_at(pair["in_s"] + 1.0) or {})
            both_pt = (self._vocal_at(cur, pair["out_s"]) > 0.5
                       and self._vocal_at(cand, pair["in_s"]) > 0.5)
            both_sec = ((sa_v.get("vocalness") or 0) > 0.5
                        and (sb_v.get("vocalness") or 0) > 0.5)
            if both_pt or both_sec:
                if getattr(cur, "has_stems", False) and style in (
                        "long_blend", "bass_swap", "filter_sweep",
                        "stem_bass_swap", "melody_carry", "loop_in",
                        "breakdown_swap"):
                    duck_vocal = True
                else:
                    style = "long_fade"
                    fade_reason = "vocal_over_vocal"

        # METER CLASH: a confidently-3/4 track against a 4/4 one cannot
        # beat-match musically whatever the style - every bar line drifts.
        # Deliberate fade, same rule as a tempo clash.
        if style != "long_fade" and rt_sure and rt.get("meter_clash"):
            style = "long_fade"
            fade_reason = "meter_clash"

        # What the dice actually chose from, and what never made the table.
        # Rides the plan into the `armed` log (tools/dj/dj_review.py --gates).
        # An empty menu when `rolled` is False is the truth, not a bug: a
        # low-confidence or beatless seam takes the fade WITHOUT building a
        # menu, so reporting the untouched theme weights there would count
        # every style as "offered" on seams where no style was ever on the
        # table - inflating the menu share of exactly the techniques the
        # gate report exists to explain.
        diag = {"gated": gated,
                "menu": ({k: round(w, 3) for k, w in weights.items()
                          if w > 0} if rolled else {}),
                # The anchors the GATES were evaluated at, before the
                # phrase snap and the kick-true offset moved out_s/in_s.
                # phase_offset() is time-bucketed, so re-checking a gate
                # at the plan's final anchors can land in a different
                # bucket and disagree with what actually happened
                # (measured: 9 of 152 seams). Anything re-deriving a gate
                # verdict - lib/dj/gateprobe.py, the gate report - has to
                # use these, not out_s/in_s.
                "pair_out_s": round(float(pair["out_s"]), 3),
                "pair_in_s": round(float(pair["in_s"]), 3),
                "fade_reason": fade_reason}
        if gate_tested:
            # Which threshold this seam was allowed to cross, so the
            # verdict can be read as evidence about that threshold.
            diag["gate_test"] = gate_tested
            if cut_trial_tags:
                # ...and for cut_drop_shape, WHICH strict bars the played
                # entry failed - per-bar tallies fall out of the log.
                diag["cut_drop_trial"] = cut_trial_tags
        if force_style:
            # Planned-set pin outcome: honored, or refused by which
            # gate(s). EVERY reason, not just the first (2026-08-15):
            # `gated` keeps only the first kill for the gate report, so
            # a pin refusal used to surface whichever screen happened to
            # run earliest - _dj_setlist_test's "drum_bridge gated
            # without stems" failed for months because the synthetic
            # pair tripped kick_offset>20ms BEFORE the stems kill and
            # the warning never mentioned no_stems at all. gated_all is
            # already collected (the trial override needs the complete
            # set); the pin verdict reads it too, so the planner's
            # warning tells the operator everything that stands between
            # the pin and the night.
            _reasons = sorted(gated_all.get(force_style) or [])
            diag["style_pin"] = {
                "want": force_style, "honored": style == force_style,
                "why_not": (None if style == force_style
                            else ", ".join(_reasons) or fade_reason
                            or "lost_menu")}

        # Pacing memory (anti-streak reads this next seam) + moment stamp.
        self.recent_styles = (self.recent_styles + [style])[-4:]
        if moment and style in ("loop_build", "cut_at_drop",
                                "stem_drum_swap", "acapella_out"):
            self.last_moment_t = time.time()

        # House blends BREATHE, and real mixes cluster transition lengths
        # at MULTIPLES OF 32 BEATS (ISMIR20, 1557 mixes) - 64 for the
        # workhorse blend, 32 for the decisive ones. Short punchy exits
        # (echo/cut) are accents, weighted rare in themes - a night of
        # 8-second slams reads harsh and amateur (user-confirmed).
        beats = {"long_blend": 64, "bass_swap": 32, "cut_at_drop": 16,
                 "loop_roll_exit": 32, "loop_build": 16, "long_fade": 0,
                 "filter_sweep": 32, "echo_out": 8,
                 "stem_drum_swap": 32, "acapella_out": 32,
                 "stem_bass_swap": 32, "drum_bridge": 16,
                 "acapella_in": 32, "melody_carry": 32,
                 "phrase_cut": 16, "spinback_cut": 16,
                 "loop_in": 32, "breakdown_swap": 32}[style]
        if style == "long_blend":
            # LENGTH VARIETY: the workhorse mostly runs 64 beats; some of
            # the time it stretches to a 96-beat marathon (still a
            # 32-multiple). One fixed length for every blend read as
            # uniform pacing (user: "there should be some variety").
            # PERSONA patience owns the marathon odds (neutral 0.35, and
            # the draw maps exactly like the legacy `< 0.65 -> 64` so a
            # neutral night is seed-for-seed identical to pre-persona).
            beats = 64 if self.rng.random() < (1.0 - self.persona.p96) \
                else 96
        # GROOVE-COUPLED LENGTH: when the grooves only half-agree, a 32-beat
        # blend is fine where a 64-beat one grates - don't ride a known
        # clash through two extra phrases. Groove-OFFSET counts too: the
        # EQ staging hides the second BASSLINE, not the second set of
        # percussion transients, so a >25ms placement gap flams the whole
        # dual - keep the exposure to one phrase.
        d_off_p = _kick_delta_s(cur, cand, rate)
        # kick_offset_s is a folded whole-track energy profile dominated
        # by BASS PLACEMENT (median 0.35-beat lies, see _sync_bias_beats
        # history). When the seam's phase profile is measured on BOTH
        # sides, kick placement is corrected by the sync bias and this
        # scalar has nothing left to predict - blends were being halved
        # and capped off stale data (user, 2026-08-05: "this is so
        # fucking short"). Pattern-level evidence (rhythm score, flam_ms
        # from the signatures) still shortens - phase correction aligns
        # lattices to music, it cannot fix two different grooves.
        from lib.dj import beatpower as _bpl
        _pv = (_bpl.phase_offset(cur.id, region="out",
                                 at_s=pair["out_s"]) is not None
               and _bpl.phase_offset(cand.id, region="in",
                                     at_s=pair["in_s"]) is not None)
        if beats > 32 and style in ("long_blend", "bass_swap",
                                    "filter_sweep", "stem_bass_swap") \
                and ((rt is not None and rt.get("score", 1.0) < 0.45)
                     or (d_off_p > 0.025 and not _pv)):
            beats = 32
        # PREDICTED-AUDIBLE FLAM -> HALVE THE EXPOSURE (user-heard "too
        # much flam"). The 28ms kick-offset gate only ever protected the
        # short-dual styles; blends were assumed safe behind EQ staging,
        # which hides basslines but not percussive flam. Measured: 37% of
        # seams carried a predicted-audible flam into a matched blend,
        # nearly all at 28 beats of dual. 16 beats keeps the blend a
        # blend while the two kick patterns coexist half as long.
        if style in ("long_blend", "bass_swap", "filter_sweep",
                     "stem_bass_swap", "melody_carry", "breakdown_swap",
                     "stem_drum_swap", "drum_bridge"):
            fl = (rt or {}).get("flam_ms")
            fl_sure = (fl is not None and 15.0 <= fl <= 80.0
                       and (rt or {}).get("conf", 0.0) >= 0.5)
            if fl_sure or (d_off_p > 0.028 and not _pv):
                beats = min(beats, 16)
        # LEARNED BLEND LENGTH: a global multiplier on however many beats
        # the rules above arrived at, snapped back to a whole bar. This is
        # the one execution knob that lives in the plan rather than in
        # build_events, because the audition pre-roll and the drawn
        # timeline both size themselves from plan["beats"].
        _bs = _tuning.value("beats_scale", TUNE_DEFAULTS["beats_scale"])
        if abs(_bs - 1.0) > 1e-6:
            beats = max(8, int(round(beats * _bs / 4.0)) * 4)
        # ANCHOR CONTEXT FOR THE NIGHT LOG (2026-08-16). Everything here
        # was already computed to make the choice; keeping it is what lets
        # a bad seam be diagnosed later - which section kinds blended, how
        # vocal/busy each side was, the kick-placement delta, the grid
        # confidences, and how much runway the entry left (room). Without
        # these the log records a verdict with no scene: "clean by the
        # meter, bad by ear" seams were undiagnosable.
        diag["anchors"] = {
            "kinds": list(pair.get("kinds") or ()),
            "busy": list(pair.get("busy") or ()),
            "voc": list(pair.get("voc") or ()),
            "room": pair.get("room"),
            "grid_conf": [round(cur.bpm_conf or 0.0, 2),
                          round(cand.bpm_conf or 0.0, 2)],
            "kick_delta_beats": round(float(d_off_p), 4),
        }
        if style == "loop_build":
            # Exit ON A's drop; the stutter build fills the bars before it.
            a_drop = self._drop_after(cur, pair["out_s"] - 8 * cur.period_s)
            out_s = cur.nearest_downbeat(a_drop)
            in_s = cand.nearest_downbeat(pair["in_s"])
            return {"style": style, "rate": rate, "out_s": out_s,
                    "in_s": in_s, "beats": beats, "rhythm": rt,
                    "pair_score": pair["score"], "cand_id": cand.id,
                    "pitch_st": pst, "a_rate": (meta or {}).get("a_rate", 1.0),
                    "diag": diag}
        # Blend-family styles anchor to PHRASE boundaries (16/32 beats) when
        # the hypermeter was confidently detected - the blend then completes
        # where the music breathes. Drop-anchored styles keep the drop.
        out_s = cur.nearest_phrase(pair["out_s"])
        in_s = cand.nearest_phrase(pair["in_s"])
        # The phrase snap is content-blind and can move the exit up to
        # half a phrase - far enough to land back in the very hole the
        # scorer just paid to avoid (Dunes: scorer 195.4s, snap 187.2s,
        # straight into the dead breakdown tail; rendered floor 0.023
        # with the anchor fix defeated). When the snapped window is
        # materially deader than the scored one, keep the scorer's
        # anchor on its own downbeat: phrasing is a preference, the
        # crater is a defect. Judged on the hole-sensitive 'min' stat -
        # see _exit_life: the snap's failure mode is a short notch that
        # the score damp's quartile deliberately ignores.
        if self._exit_life(cur, out_s, stat="min") \
                < 0.5 * self._exit_life(cur, pair["out_s"], stat="min"):
            out_s = cur.nearest_downbeat(pair["out_s"])
        plan_b_drop = None            # breakdown_swap's payoff, if any
        plan_drop_step = None         # cut_at_drop's measured slam size
        if style == "breakdown_swap" and bd_a is not None \
                and bl_b is not None:
            # Blend over A's BREAKDOWN carrying B's BUILD: exit a phrase
            # into the breakdown, enter at the build's start - the drop
            # that follows is the payoff.
            out_s = cur.nearest_phrase(min(
                bd_a["start_s"] + 8 * cur.period_s,
                max(bd_a["end_s"] - 4 * cur.period_s,
                    bd_a["start_s"])))
            in_s = cand.nearest_phrase(bl_b["start_s"])
            # The payoff, carried to build_events so the low/mid restore can
            # get out of its way. Phrase-snapping in_s moves the entry, so
            # the drop is stamped RELATIVE to the snapped entry, not the
            # raw build start - build_events maps it through rate_b.
            plan_b_drop = b_drop_s
        if style == "cut_at_drop":
            # Enter exactly where the gate above vetted: B's strongest
            # measured drop, already snapped to a downbeat. cut_pd is a
            # TIME here, not a mix-in dict - the hints are no longer
            # consulted, so the seam that plays is the seam that was
            # checked, with no second opinion in between.
            if cut_pd is not None:
                in_s = cut_pd
                plan_drop_step = cut_step or None
        plan = {"style": style, "rate": rate,
                "out_s": out_s, "in_s": in_s, "beats": beats, "rhythm": rt,
                "pair_score": pair["score"], "cand_id": cand.id,
                "duck_vocal_a": duck_vocal, "b_drop_s": plan_b_drop,
                "drop_step": plan_drop_step,
                    "pitch_st": pst, "a_rate": (meta or {}).get("a_rate", 1.0),
                    "diag": diag}
        if arc is not None:
            plan["arc"] = round(arc, 3)   # riser gating in the fade path
        # VARISPEED MEET-IN-THE-MIDDLE: with the varispeed engine pitch rides
        # tempo, so a single-sided match puts the whole pitch shift on B.
        # Split the bend across BOTH decks instead - A ramps to a_rate before
        # the blend (build_events schedules it), B enters at sqrt(rate) and
        # glides home from HALF the distance. Blend-family styles only: they
        # are the ones whose event builder implements the outgoing ramp.
        _blend_split = (stretch_engine_name() == "vari"
                        and style in ("long_blend", "bass_swap",
                                      "filter_sweep", "stem_drum_swap",
                                      "acapella_out", "stem_bass_swap",
                                      "drum_bridge", "acapella_in",
                                      "melody_carry"))
        # THE CUT SPLITS UNDER KEYLOCK TOO (2026-08-12, operator's call).
        # For the blend family the split exists to halve the PITCH shift,
        # which is why it is varispeed-only - under R3 there is no pitch
        # shift to halve. The cut splits for a second reason keylock does
        # NOT remove: B is asked to play its DROP - the entire payoff of
        # the move - up to 10% off its own tempo, and then slides home for
        # two minutes afterwards. Halving that costs the departing track a
        # bend it will not live to regret. (Its event builder schedules
        # the outgoing ramp; that is what the whitelist really gated on.)
        _cut_split = style == "cut_at_drop"
        if ((_blend_split or _cut_split)
                and plan["a_rate"] in (1.0, None) and not pst
                and abs(math.log(max(plan["rate"], 1e-6))) > 0.010):
            plan["rate"] = math.sqrt(plan["rate"])
            plan["a_rate"] = 1.0 / plan["rate"]
        if style == "loop_roll_exit":
            # Loop the 16 bars-worth just before the exit point: with the
            # window pinned to out_s the first wrap and both shrink moments
            # all land exactly on the grid (elapsed beats stay multiples of
            # the shrinking span).
            plan["loop_start_s"] = max(0.0, out_s - 16 * cur.period_s)
        if style in ("acapella_out", "melody_carry"):
            plan["tail_beats"] = 16   # A's exposed vocal/melody rides B
        return plan

    @staticmethod
    def _vc_arrs(track):
        """The fine demucs vocal curve as cached (xs, ys) numpy arrays,
        or None when the track has no stored curve. Sampled ~14x per
        candidate during selection (perf audit 2026-07-31)."""
        cached = getattr(track, "_vc_arrs", None)
        if cached is None:
            vc = (track.axes or {}).get("vc")
            if vc and len(vc) >= 2:
                cached = (np.asarray([p[0] for p in vc], dtype=np.float64),
                          np.asarray([p[1] for p in vc], dtype=np.float64))
            else:
                cached = False
            track._vc_arrs = cached
        return cached or None

    @classmethod
    def _vocal_span_max(cls, track, t0, t1, step_s=6.0):
        """Peak vocal presence inside [t0, t1] of a track's timeline,
        sampled from the fine demucs curve. The clash question is 'is
        there a vocal line ANYWHERE in this window', so max, not mean.

        Memoized per track: the windows come from stored mix points, so
        the same few spans are re-walked for every candidate the picker
        scores (~7000 redundant walks per selection pass, perf audit
        2026-07-31)."""
        cache = getattr(track, "_vspan_cache", None)
        if cache is None:
            cache = track._vspan_cache = {}
        key = (round(t0, 4), round(t1, 4), step_s)
        hit = cache.get(key)
        if hit is not None:
            return hit
        ct0 = max(0.0, min(t0, track.duration_s))
        ct1 = max(ct0, min(t1, track.duration_s))
        n = max(2, int((ct1 - ct0) / step_s) + 1)
        arrs = cls._vc_arrs(track)
        if arrs is not None:
            xs, ys = arrs
            ts = ct0 + (ct1 - ct0) * np.arange(n, dtype=np.float64) / (n - 1)
            # np.interp clamps outside [xs0, xs-1] to the end values -
            # the same behavior as the scalar path's edge branches.
            out = float(np.interp(ts, xs, ys).max())
        else:
            out = max(cls._vocal_at(track, ct0 + (ct1 - ct0) * k / (n - 1))
                      for k in range(n))
        if len(cache) > 256:            # a night touches ~16 spans/track
            cache.clear()
        cache[key] = out
        return out

    @classmethod
    def _vocal_at(cls, track, t):
        """Vocal presence at a source time: fine demucs curve when stored
        (axes['vc']), per-section mean otherwise."""
        arrs = cls._vc_arrs(track)
        if arrs is not None:
            xs, ys = arrs
            return float(np.interp(t, xs, ys))
        sec = track.section_at(t)
        return (sec.get("vocalness") or 0.0) if sec else 0.0

    # -- automation compilation ------------------------------------------------
    def build_events(self, plan, snapshot, active, incoming, cur, cand):
        """Compile a plan into submix events. `snapshot` is submix telemetry;
        `active`/`incoming` are deck names; `cur`/`cand` TrackInfos.

        Returns (events, swap_at_clock, blend_start_clock).

        TUNABLE EXECUTION: `plan["tune"]` may override the geometry and EQ
        constants below by name (see TUNE_DEFAULTS). Every one of them was
        a hand-picked number; exposing them lets the Seam Lab jitter them
        per seam and learn which way each one wants to move. Absent or
        empty, every value is exactly the old constant."""
        tune = plan.get("tune") or {}

        def K(name):
            # Per-seam override (the lab's jitter) > LEARNED value > the
            # original constant. This is where a rating finally changes
            # how the engine mixes.
            if name in tune:
                return float(tune[name])
            return float(_tuning.value(name, TUNE_DEFAULTS[name]))

        # KICK-TRUE ANCHORS (2026-08-04). out_s/in_s are grid-quantized,
        # but the stored grid PHASE misses the audible kicks by ~48ms
        # median in seam regions (see submix._sync_bias_beats). A cut
        # scheduled on a grid beat therefore lands mid-flam, and B's cue
        # opens ~50ms off its own downbeat - user-heard on every style,
        # loudest on phrase_cut where nothing overlaps to blur it. Shift
        # both anchors from lattice time to MUSIC time using the measured
        # per-region offsets. Synced styles are unaffected on the B side
        # (the snap re-anchors), cuts are fixed on both sides. Guarded:
        # build_events may recompile a plan (abort/recall).
        if "phase_applied" not in plan:
            from lib.dj import beatpower as _bp
            gf = plan.get("grid_fixed") or {}
            off_a = 0.0 if gf.get("a") else \
                _bp.phase_offset(cur.id, region="out",
                                 at_s=plan["out_s"]) or 0.0
            off_b = 0.0 if gf.get("b") else \
                _bp.phase_offset(cand.id, region="in",
                                 at_s=plan["in_s"]) or 0.0
            plan["out_s"] += off_a
            plan["in_s"] += off_b
            plan["phase_applied"] = {"a_ms": round(off_a * 1000, 1),
                                     "b_ms": round(off_b * 1000, 1)}
        # ONE source of truth for the kick bias: computed HERE from the
        # same offsets that shifted the anchors, and shipped inside the
        # sync events. (v1 let the submix re-look-up the offsets by deck
        # position; validation caught it landing on a different profile
        # bucket than the anchors - a 25ms intended bias rendered as
        # 151ms of kick error. Never two lookups for one seam.)
        _pa = plan["phase_applied"]
        sync_bias = float(np.clip(
            _pa["b_ms"] / 1000.0 / max(cand.period_s, 1e-6)
            - _pa["a_ms"] / 1000.0 / max(cur.period_s, 1e-6),
            -0.25, 0.25))
        # Ship the sync picture to the night log too (diag rides the
        # `armed` event, logged after this compile): the anchor phase
        # shifts, the bias the PLL will hold, and which sides ran on a
        # live-fixed grid - the fields a flam post-mortem needs first.
        plan.setdefault("diag", {})["sync"] = {
            "phase_a_ms": _pa["a_ms"], "phase_b_ms": _pa["b_ms"],
            "bias_beats": round(sync_bias, 4),
            "grid_fixed": dict(plan.get("grid_fixed") or {})}
        # Audio-PLL stays ON for all pairs. Measured both ways on the
        # weak-kick outlier pairs (2026-08-04): with the single-reading
        # jump replaced by the 3-stable bar, the audio path IMPROVED
        # them (97/65ms audio-off vs 74/52 audio-on) - on unstable-phase
        # material the deck's actual output transients beat any stored
        # lattice, corrected or not. The flag stays plumbed for future
        # experiments; the real exposure on weak-agreement pairs is run-
        # in loudness, handled below.
        _rt = plan.get("rhythm") or {}
        _ka = _rt.get("kick_agreement")
        sync_audio = True
        # Weak or unknown kick agreement: the cut styles' beat-matched
        # run-in cannot truly lock (flam risk is material, not
        # execution) - ride the incoming deck quieter until the cut so
        # the overlap stays texture, not a competing beat.
        runin_gain = 0.8 if (_ka is not None and _ka >= 0.5) else 0.5
        tel = snapshot["decks"][active]
        clock = snapshot["clock"]
        rate_a = max(tel["rate"], 1e-6)

        def clock_at(src_time_s):
            return clock + int((src_time_s - tel["time_s"]) / rate_a * RATE)

        # Nothing may schedule in the past: past events all fire in one
        # flush (run-ins vanish, sync snaps at full gain, loop windows can
        # land entirely behind the cursor and never wrap).
        now_guard = clock + int(0.3 * RATE)
        # Each style stamps plan["no_return_at"]: the decisive clock beyond
        # which aborting sounds worse than finishing (the bass/melody swap,
        # the cut, the drop). DJSystem._do_abort recalls the transition only
        # BEFORE this point; past it, the mix is committed.

        beat_out = cur.period_s / rate_a          # output-domain beat of A
        style = plan["style"]
        rate_b = plan["rate"]
        ev = []

        if style == "long_fade":
            # DIPPED HANDOFF, not a symmetric wash. This style exists
            # precisely because the pair CANNOT be beat-matched (loose
            # grid / tempo clash) - so a long full-range overlap is two
            # unrelated songs fighting (measured ~half of all seams on an
            # eclectic library; user-heard as 'terribly mixed'). Radio
            # rule instead: the outgoing track is mostly GONE before the
            # incoming one rises, with a deliberate low-level dip between
            # chapters. Overlap where both are loud: ~2s instead of 12.
            S0 = clock_at(plan["out_s"])
            # Shape (v3): the outgoing song RECEDES to ~half level first
            # (still carrying the room), the incoming arrives underneath
            # over ~9s, and only then does the outgoing leave. v2's
            # 5.5s-out/4s-in handoff was heard as songs slamming into
            # each other - at ~half of all seams on an eclectic library
            # the fade IS the mix, it has to breathe. v1's symmetric 12s
            # full-level wash (mud) and v2's -22dB hole (lag 3/rise 6
            # into a quiet intro) both stay dead: A holds 0.45 through
            # B's rise, so the room never empties and never doubles.
            # The shape lives mostly BEFORE the out point: past out_s the
            # outgoing track's own arrangement often collapses (that IS
            # the boundary), so v3's hold-A-past-the-seam dug a -23 dB
            # hole. Recede through A's final LOUD phrase instead, get B
            # half-up by the boundary, and let A leave just after it.
            # URGENT (skip / mix-now) fades COMPRESS (2026-08-05): a skip
            # means "move on" - the live log showed skip after skip each
            # dragged the full 8s recede + slow arrival out of wherever
            # the song happened to be ("now the fades are terrible").
            # The deliberate fade keeps its breathing room; the urgent
            # one gets A out and B present in half the time.
            _ug = 0.45 if plan.get("urgent") else 1.0
            # THE FADE IS DELIBERATELY UNSYNCED, AND THAT IS ITS SAFETY
            # (2026-08-06). Three "improvements" were reverted here after
            # the user heard what they did ("I can hear things mismatch
            # on the long fade"; a week earlier: "generic but safe"):
            #   1. TEMPO MATCH within the stretch wall. Matching tempo
            #      WITHOUT sync is the worst of both worlds: B's beats
            #      hold a near-constant offset from A's for the entire
            #      overlap - a sustained flam the ear locks onto. At
            #      NATIVE tempo the two grids drift past each other and
            #      no single relationship is audible long enough to
            #      annoy. B plays at 1.0, always.
            #   2. QUIET-INTRO TRIM. B's sparse intro sitting ~10dB
            #      under A was MASKING its unsynced percussion; lifting
            #      it (up to +4.6dB) made the mismatch plainly audible.
            #      A crater at a dead tail is the accepted trade - that
            #      is what "generic but safe" sounded like.
            #   3. BOUNDARY-ALIGNED CUE. Landing B's anchor downbeat on
            #      the seam only matters if the beats relate; unsynced,
            #      it just re-picks which misaligned beats collide.
            # B is cued at in_s SOUNDING at B0, native tempo, nominal
            # fader. Do not re-add these without the user asking.
            # ONE BEAT PATTERN AT A TIME (2026-08-14). The low baton
            # below hands over the KICK, but a fade between two
            # rhythm-dense tracks still played both PERCUSSION lines -
            # B's hats/click arrive at full EQ under A's still-full kit,
            # free-running through each other from B0 ("both beatlines
            # over each other, well out of phase" - operator, on a
            # hypnotic-heavy night whose fades all measured predicted
            # kick_agreement 0.1-0.5: the planner predicted every clash
            # and the fade path never consumed the number). Measured on
            # rendered clash pairs, in three attempts:
            #   - carving A's mids AFTER the seam moved perc co-presence
            #     by nothing: the clash lives BEFORE the seam;
            #   - closing B's HIGH at entry moved it barely: the EQ
            #     splits at 2500 Hz and half the percussion band
            #     (snare body, 1-2.5k) rides B's MID, which is also its
            #     identity and stays whole by design.
            # So the lever that remains is TIME, exactly as
            # perc_overlap's own note predicts ("a shelf or a different
            # in-point"): on clash pairs B's entry moves closer to the
            # seam (fade_clash_lead_x), its high waits with its low,
            # and at S0 the top end is handed over on the baton clocks
            # while A's mids leave decisively. The TRIGGER is rhythm
            # DENSITY on both sides - see the fade_clash_density knob
            # note for why kick_agreement was the wrong question on
            # unsynced decks (rated wrong the day it shipped: identical
            # patterns PHASING are the trainwreck, not a mismatch).
            # Evidence-gated per track: no density measurement, no
            # carve - an ambient/sparse side keeps the arrive-whole
            # entry.
            _da = getattr(cur, "rhythm_density", None)
            _db_ = getattr(cand, "rhythm_density", None)
            _clash = (_da is not None and _db_ is not None
                      and min(_da, _db_) >= K("fade_clash_density"))
            # DEPTH TIER: how hard the carve may squeeze depends on
            # whether B's entry can CARRY a tight handover - see the
            # fade_clash_hot_heat knob note. Entry heat = B's first 15s
            # after in_s against its own body median (evidence-gated:
            # no curve, no deep tier).
            _hot = False
            if _clash:
                _bc = self._energy_arr(cand)
                if len(_bc):
                    _i0 = int(plan["in_s"] * 2)
                    _i1 = min(int((plan["in_s"] + 15.0) * 2), len(_bc))
                    if 0 <= _i0 < _i1:
                        _body = float(np.median(_bc))
                        _heat = float(_bc[_i0:_i1].mean()) / max(_body,
                                                                 1e-4)
                        _hot = _heat >= K("fade_clash_hot_heat")
                plan.setdefault("diag", {})["fade_clash_carve"] = [
                    round(_da, 2), round(_db_, 2),
                    "hot" if _hot else "gentle"]
            _lead_x = (K("fade_clash_lead_hot_x") if _hot
                       else K("fade_clash_lead_x")) if _clash else 1.0
            _mid_out = (K("fade_a_mid_out_hot") if _hot
                        else K("fade_a_mid_out"))
            A0 = max(S0 - int(K("fade_lead_a") * _ug * RATE), now_guard)
            B0 = max(S0 - int(K("fade_lead_b") * _ug * _lead_x * RATE), A0)
            ev += [
                {"at": A0, "cmd": "gain", "deck": active,
                 "value": K("fade_recede"),
                 "ramp_s": K("fade_lead_a") * _ug},
                {"at": B0, "cmd": "cue", "deck": incoming,
                 "time_s": plan["in_s"]},
                {"at": B0, "cmd": "rate", "deck": incoming, "value": 1.0},
                # Mids/highs full from the first beat (a fade is not a
                # carve - B's identity arrives whole), but the LOW waits
                # until A has left (2026-08-05): a fade overlaps two
                # UNSYNCED tracks, and at similar tempos their kick
                # drums phase against each other through the whole dip -
                # rare when fades were rare, constant once they carried
                # half the night ("the kick clash is terrible" - user).
                # Atmosphere may overlap; unsynced KICKS never do.
                # (On predicted-clash pairs the HIGH waits with it - the
                # percussion baton above.)
                {"at": B0, "cmd": "eq", "deck": incoming, "low": 0.0,
                 "mid": 1.0, "high": 0.0 if _clash else 1.0,
                 "ramp_s": 0.01},
                # A GIVES UP ITS AIR AS B ARRIVES (2026-08-06). The carve
                # above removes B's kick FUNDAMENTAL and nothing else -
                # the 200 Hz LR4 split leaves B's beater click, snare and
                # hats at unity, and those transients are what the ear
                # locks onto (the PLL correlates exactly them; low-band
                # envelopes were rejected for the job). So from the
                # moment B is in the room, the DEPARTING track starts
                # giving up its top end - a fade-out getting darker,
                # which is what fade-outs do. A keeps its low until the
                # seam, its mids, its melody and its voice.
                # Carving A, never B: B's identity arrives whole and its
                # quiet entry is the masking the regime note protects.
                # Set fade_a_high to 1.0 to revert this exactly.
                {"at": B0, "cmd": "eq", "deck": active,
                 "high": K("fade_a_high"), "ramp_s": 2.0},
                # Lows return AS A exits (not after), so the room never
                # goes drumless-on-drumless ("the last mix was terrible" -
                # comedown tail + dip + kickless arrival stacked into
                # three kinds of empty). This timing is LOAD-BEARING: do
                # not push it later.
                # The original rationale here also claimed A was "too
                # quiet to clash" by this point. It was not - A recedes
                # to 0.5 and takes fade_out_ramp to reach zero, so the
                # two low bands crossed at ~-10 dB with both kicks plainly
                # audible. That is fixed on A's side below, not by moving
                # this event.
                # B's low rises on the SAME clock A's leaves (2026-08-06),
                # so the low band crossfades instead of leaving a gap.
                # Tied to fade_a_low_out rather than fade_out_ramp: A's
                # kick was gone in 1.2s while B's took the full 5s to
                # arrive, and the ~4s of missing low measured as a
                # halved rms_min (0.39 -> 0.19 on Imagine -> Got to
                # Change). B's low still STARTS at S0 and now reaches
                # full sooner than it used to - earlier low end, not
                # later, so the drumless-on-drumless warning above is
                # respected by a wider margin than before.
                {"at": S0, "cmd": "eq", "deck": incoming, "low": 1.0,
                 "mid": 1.0,
                 "ramp_s": max(K("fade_b_low_in") * _ug, 1.2)},
                # A's KICK LEAVES BEFORE B's ARRIVES (2026-08-06). The
                # line above and A's exit fade below used to ramp THROUGH
                # each other across fade_out_ramp - A receding from 0.5,
                # B rising from 0 - crossing around -10 dB with both kick
                # fundamentals plainly audible for 2-3s ("we keep getting
                # awful kick clashes"). Hand the low band over instead of
                # crossing it: A's kick is gone in ~1.2s, by which point
                # B's low is still under -17 dB.
                # NOT by delaying B's low - that is the one thing the
                # note below forbids (it emptied the room). The
                # separation is won from A's side, and A's mids/highs
                # keep carrying until its fade completes, so nothing goes
                # drumless-on-drumless.
                {"at": S0, "cmd": "eq", "deck": active, "low": 0.0,
                 "ramp_s": K("fade_a_low_out") * _ug},
                {"at": B0, "cmd": "gain", "deck": incoming, "value": 0.0,
                 "ramp_s": 0.01},
                {"at": B0, "cmd": "start", "deck": incoming},
                # Arrive in two stages: present quickly, full gently.
                # Nominal fader - no intro trim (see the regime note
                # above: the quiet entry is what masks the mismatch).
                {"at": B0, "cmd": "gain", "deck": incoming,
                 "value": K("fade_b_stage1"),
                 "ramp_s": K("fade_b_ramp1") * _ug},
                {"at": B0 + int(K("fade_b_ramp1") * _ug * RATE),
                 "cmd": "gain", "deck": incoming, "value": 1.0,
                 "ramp_s": K("fade_b_ramp2") * _ug},
                {"at": S0, "cmd": "gain", "deck": active, "value": 0.0,
                 "ramp_s": K("fade_out_ramp") * _ug},
                {"at": S0 + int(K("fade_stop_lead") * RATE), "cmd": "stop",
                 "deck": active},
            ]
            if _clash:
                # THE PERCUSSION BATON'S SEAM HALF (see the note above
                # A0): B's high arrives on the low-baton clock as A's
                # whole kit leaves - high fast with the low, mids on
                # their own slightly gentler clock (they also carry A's
                # melody; this is a fade-to-darkness, not a mute).
                ev += [
                    {"at": S0, "cmd": "eq", "deck": incoming, "high": 1.0,
                     "ramp_s": max(K("fade_b_low_in") * _ug, 1.2)},
                    {"at": S0, "cmd": "eq", "deck": active, "high": 0.0,
                     "ramp_s": K("fade_a_low_out") * _ug},
                    {"at": S0, "cmd": "eq", "deck": active, "mid": 0.0,
                     "ramp_s": _mid_out * _ug},
                ]
            # (A riser-through-the-dip variant was tried and REMOVED
            # 2026-08-02: synthesized whooshes read as cheesy - user. The
            # fade stays clean; drama comes from the music, brake or echo.)
            plan["no_return_at"] = S0        # A starts leaving at the seam
            # The seam clock itself, for anything that must not play past
            # it - the percussion bed sizes its tail from this (its
            # returned swap_at is S0+6s, which had it tiling A's drums
            # straight through B's arrival).
            plan["seam_at"] = S0
            return ev, S0 + int(6.0 * RATE), A0

        nb = plan["beats"]
        if style == "echo_out":
            # Throw A's last beat into a tempo-synced delay and cut: the
            # tail decays over B, which arrives beat-locked underneath -
            # the clean way to LEAVE a track without a long fade.
            S_out = clock_at(plan["out_s"])
            lead = int(K("echo_lead_beats") * cand.period_s / rate_b
                       * RATE)
            S0 = max(S_out - lead, now_guard)
            cue_b = max(0.0, plan["in_s"] - (S_out - S0) / RATE * rate_b)
            ev += [
                {"at": S0, "cmd": "cue", "deck": incoming, "time_s": cue_b},
                {"at": S0, "cmd": "rate", "deck": incoming, "value": rate_b},
                {"at": S0, "cmd": "eq", "deck": incoming, "low": 0.0,
                 "mid": 0.55, "ramp_s": 0.01},
                {"at": S0, "cmd": "gain", "deck": incoming, "value": 0.0,
                 "ramp_s": 0.01},
                {"at": S0, "cmd": "start", "deck": incoming},
                {"at": S0, "cmd": "sync", "slave": incoming, "master": active,
                 "bias_beats": sync_bias, "audio_pll": sync_audio},
                {"at": S0, "cmd": "gain", "deck": incoming,
                 "value": K("echo_b_gain"),
                 "ramp_s": max((S_out - S0) / RATE, 0.1)},
                # The throw: dotted-eighth echo engages one beat out.
                {"at": S_out - int(beat_out * RATE), "cmd": "echo",
                 "deck": active, "active": True,
                 "delay_s": K("echo_delay_beats") * beat_out,
                 "feedback": K("echo_feedback"),
                 "wet": K("echo_wet")},
                {"at": S_out, "cmd": "gain", "deck": active, "value": 0.0,
                 "ramp_s": 0.03},
                {"at": S_out, "cmd": "eq", "deck": incoming, "low": 1.0,
                 "mid": 1.0, "ramp_s": 0.25},
                {"at": S_out, "cmd": "gain", "deck": incoming, "value": 1.0,
                 "ramp_s": 2 * beat_out},
                {"at": S_out + int(K("echo_tail_s") * RATE), "cmd": "stop",
                 "deck": active},
                {"at": S_out + int(K("echo_tail_s") * RATE),
                 "cmd": "end_sync"},
            ]
            swap_at = S_out + int(K("echo_tail_s") * RATE)
            plan["no_return_at"] = S_out - int(beat_out * RATE)  # echo throw
            self._glide_home(ev, incoming, rate_b, swap_at)
            return ev, swap_at, S0

        if style in ("cut_at_drop", "phrase_cut", "spinback_cut"):
            # The cut lands on B's drop downbeat (cut_at_drop) or a shared
            # phrase boundary (phrase_cut - the open-format slam for music
            # WITHOUT drops; spinback_cut - the platter dies backward-
            # feeling into a cold slam, total harmonic reset); B rides in
            # underneath first.
            S_cut = clock_at(plan["out_s"])
            # 16 B-beats of run-up, measured in OUTPUT time (period/rate) so
            # the launch lands 16 matched beats before the cut, not 16 source
            # beats (up to 8% off - a beat-and-a-third the PLL can't absorb).
            lead = int(16 * cand.period_s / rate_b * RATE)
            # MEET IN THE MIDDLE ON A CUT (2026-08-12). This path used to
            # leave A at its natural rate and stretch B the whole way, so
            # the incoming DROP - the only reason the style exists -
            # arrived bent by the entire tempo difference and then slid
            # home over ~2 minutes. Measured over 6 renders before this
            # change: a_rate 1.0 on every pair, B carrying a median 0.24
            # and a worst 1.76 semitones AT THE DROP. Bend both decks
            # instead, exactly as the blend family does.
            a_rate = plan.get("a_rate", 1.0) or 1.0
            if abs(a_rate - 1.0) > 1e-4:
                # AFFORD THE SPLIT, NEVER RUSH IT. The ramp has to finish
                # before the run-in (a master still moving gives the PLL a
                # target it cannot lock) and has to run at
                # ARATE_RAMP_PER_S, the gradient measured inaudible.
                # Armed PLAN_LEAD_S ahead that buys ~4% of bend; deeper
                # pairs take the split they can afford rather than a
                # faster, audible glide. a_rate -> 1.0 degrades to exactly
                # the old single-sided behaviour.
                full = rate_b / a_rate       # B's rate with A left alone
                budget = max(0.0, (S_cut - lead - now_guard) / RATE
                             * ARATE_RAMP_PER_S)
                d = min(abs(a_rate - 1.0), budget)
                a_rate = 1.0 + (d if a_rate > 1.0 else -d)
                rate_b = full * a_rate       # keeps the beat match exact
                lead = int(16 * cand.period_s / rate_b * RATE)
                plan["rate"], plan["a_rate"] = rate_b, a_rate
            if abs(a_rate - 1.0) > 1e-4:
                ramp_wall = abs(a_rate - 1.0) / ARATE_RAMP_PER_S
                # clock_at() maps A's source to wall assuming a CONSTANT
                # rate_a. The ramp and the run-in at a_rate both break
                # that, and on a CUT the error is not cosmetic - it lands
                # the slam off the drop. A runs (a_rate - 1) fast across
                # the run-in and, on average, half of that across the
                # ramp: take exactly that much source back. (The blend
                # path's own correction is an approximation it can
                # absorb; a cut cannot, so this one is derived.)
                S_cut -= int((a_rate - 1.0)
                             * (lead / RATE + ramp_wall / 2.0) * RATE)
                ev.append({"at": max(S_cut - lead - int(ramp_wall * RATE),
                                     now_guard),
                           "cmd": "rate", "deck": active, "value": a_rate,
                           "ramp_s": ramp_wall})
            S0 = max(S_cut - lead, now_guard)
            # B must still ARRIVE at in_s exactly at the cut, however much
            # run-in survives the clamp.
            cue_b = max(0.0, plan["in_s"]
                        - (S_cut - S0) / RATE * rate_b)
            ev += [
                {"at": S0, "cmd": "cue", "deck": incoming, "time_s": cue_b},
                {"at": S0, "cmd": "rate", "deck": incoming, "value": rate_b},
                {"at": S0, "cmd": "eq", "deck": incoming, "low": 0.0,
                 "mid": 0.5, "ramp_s": 0.01},
                {"at": S0, "cmd": "gain", "deck": incoming, "value": 0.0,
                 "ramp_s": 0.01},
                {"at": S0, "cmd": "start", "deck": incoming},
                {"at": S0, "cmd": "sync", "slave": incoming, "master": active,
                 "bias_beats": sync_bias, "audio_pll": sync_audio},
                {"at": S0, "cmd": "gain", "deck": incoming,
                 "value": runin_gain,
                 "ramp_s": 12 * cand.period_s},
                {"at": S_cut, "cmd": "end_sync"},
                {"at": S_cut, "cmd": "gain", "deck": active, "value": 0.0,
                 "ramp_s": 0.04},
                {"at": S_cut, "cmd": "eq", "deck": incoming, "low": 1.0,
                 "mid": 1.0, "high": 1.0, "ramp_s": 0.04},
                {"at": S_cut, "cmd": "gain", "deck": incoming, "value": 1.0,
                 "ramp_s": 0.04},
                {"at": S_cut + int(0.5 * RATE), "cmd": "stop", "deck": active},
            ]
            # Half the time, A leaves with a vinyl BRAKE into the drop
            # instead of a plain cut - the platter winds down through the
            # last bar and B's drop slams in. spinback_cut ALWAYS brakes,
            # longer - the dying platter IS the style.
            if style == "spinback_cut":
                _sb = K("spinback_s")
                ev.append({"at": S_cut - int(_sb * RATE), "cmd": "brake",
                           "deck": active, "duration_s": _sb})
            elif self.rng.random() < K("brake_chance"):
                _br = K("brake_s")
                ev.append({"at": S_cut - int(_br * RATE), "cmd": "brake",
                           "deck": active, "duration_s": _br})
            # (The synthesized riser into the cut was REMOVED 2026-08-02:
            # user verdict - cheesy. The brake and the cold landing carry
            # the moment on the music alone.)
            swap_at = S_cut + int(0.5 * RATE)
            plan["no_return_at"] = S_cut - int(1.2 * RATE)   # covers the brake
            self._glide_home(ev, incoming, rate_b, swap_at)
            return ev, swap_at, S0

        if style == "loop_build":
            # Tension build: A stutters a loop that shrinks 8->4->2->1 beats
            # (all ending on its drop) accelerating into it, releases ON the
            # drop, and B cuts in - the loop-build-into-drop move. A's loop
            # end is pinned to the drop so release lands exactly on it.
            drop_s = plan["out_s"]
            per = cur.period_s
            # (beats_len, output beats to hold that stage)
            stages = [(8, 8), (4, 4), (2, 2), (1, 2)]
            S0 = max(clock_at(drop_s - stages[0][0] * per), now_guard)
            t = S0
            for length, hold in stages:
                ls = drop_s - length * per
                # (a late-fired loop is safe: the window END is the drop,
                # still ahead of the cursor, so the wrap engages normally)
                ev.append({"at": max(t, now_guard), "cmd": "loop",
                           "deck": active, "start_s": ls, "end_s": drop_s})
                # Filter up as it builds (rising tension), trim lows late.
                ev.append({"at": t, "cmd": "eq", "deck": active,
                           "high": 1.0, "mid": 1.0,
                           "low": 1.0 if length > 2 else 0.6, "ramp_s": 0.1})
                t += int(hold * beat_out * RATE)
            S_drop = t                                   # release = the drop
            cue_b = max(0.0, plan["in_s"] - 8 * cand.period_s)
            out = S_drop + int(8 * beat_out * RATE)
            ev += [
                # Pre-run B under the tail of the build, bass-cut + synced.
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "cue",
                 "deck": incoming, "time_s": cue_b},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "rate",
                 "deck": incoming, "value": rate_b},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "eq",
                 "deck": incoming, "low": 0.0, "mid": 0.5, "high": 0.6,
                 "ramp_s": 0.05},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "gain",
                 "deck": incoming, "value": 0.0, "ramp_s": 0.05},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "start",
                 "deck": incoming},
                {"at": S_drop - int(8 * beat_out * RATE), "cmd": "sync",
                 "slave": incoming, "master": active, "bias_beats": sync_bias, "audio_pll": sync_audio},
                # THE DROP: release A's loop into it, B slams in full, A ducks.
                {"at": S_drop, "cmd": "release_loop", "deck": active},
                {"at": S_drop, "cmd": "eq", "deck": incoming, "low": 1.0,
                 "mid": 1.0, "high": 1.0, "ramp_s": 0.06},
                {"at": S_drop, "cmd": "gain", "deck": incoming, "value": 1.0,
                 "ramp_s": 0.06},
                {"at": S_drop, "cmd": "eq", "deck": active, "low": 0.0,
                 "ramp_s": 0.06},
                {"at": S_drop, "cmd": "gain", "deck": active, "value": 0.0,
                 "ramp_s": 4 * beat_out},
                {"at": out, "cmd": "stop", "deck": active},
                {"at": out, "cmd": "clear_loop", "deck": active},
                {"at": out, "cmd": "end_sync"},
            ]
            # (Riser + impact "production polish" REMOVED 2026-08-02:
            # synthesized FX at transitions read as cheesy - user. The
            # shrinking loop is its own build; the drop is its own impact.)
            plan["no_return_at"] = S_drop        # the loop releases into it
            self._glide_home(ev, incoming, rate_b, out)
            return ev, out, S0

        # (double_drop and bassline_layer were REMOVED 2026-08-02 with
        # their choreography: double_drop was the last fx_play one-shot
        # holdout - the nextdrop MOMENT in system.py owns the synced-drop
        # spectacle now, on the music alone - and bassline_layer won 3
        # rolls in 2000. Old pins refuse politely via the retired kill.)

        # Clean bass-swap EQ blend (long_blend / bass_swap / loop_roll_exit).
        # The golden rule: ONLY ONE BASSLINE AT A TIME. The incoming track
        # comes in with its low end fully cut and rides on top (we mix into
        # its intro/breakdown, so that's drums + atmosphere, not a clashing
        # lead); at the midpoint downbeat the bass swaps decisively in one
        # move; the outgoing track then leaves with its bass already gone.
        # No two-bass mud, no dueling low mids - the reliable pro default.
        # The blend COMPLETES at out_s - A's out point is the boundary where
        # its groove ends (that's why the seam scored there), so playing
        # 16-32 beats PAST it means A's own outro collapse lands mid-blend
        # (measured as 8-9 dB level lurches). Real DJs finish the blend ON
        # the boundary, riding A's last full-groove phrase.
        end = clock_at(plan["out_s"])
        # DUAL-DECK TEMPO MEET: the OUTGOING deck ramps to the meeting tempo
        # BEFORE the blend (ARATE_RAMP_PER_S - a slow pitch glide under
        # varispeed, so it must stay below the drift-noticing threshold),
        # the blend runs at that tempo, and the incoming glides home after
        # the swap. clock_at assumes constant rate, so the ramp's source-vs-
        # wall skew is compensated in `end`.
        a_rate = plan.get("a_rate", 1.0) or 1.0
        if abs(a_rate - 1.0) > 1e-4:
            ramp_wall = abs(a_rate - 1.0) / ARATE_RAMP_PER_S
            end -= int(ramp_wall * (a_rate - 1.0) / 2.0 / a_rate * RATE)
            beat_out = cur.period_s / a_rate
            S0 = max(end - int(nb * beat_out * RATE), now_guard)
            ev.append({"at": max(S0 - int(ramp_wall * RATE), now_guard),
                       "cmd": "rate", "deck": active, "value": a_rate,
                       "ramp_s": ramp_wall})
        else:
            S0 = max(end - int(nb * beat_out * RATE), now_guard)
        mid = S0 + int((end - S0) * K("swap_pos"))
        half = (end - S0) / RATE / 2.0
        # STAGED BEATS-TOGETHER FOR EVERY REAL BLEND (2026-08-05). The
        # staging comment below records that unstaged spans "perceptually
        # collapsed to the ~4-beat swap crossfade" - and bass_swap/
        # filter_sweep kept exactly that geometry in the name of style
        # variety. The user heard precisely the predicted collapse ("it's
        # literally 4 beats with most being reduced - this isn't how you
        # are supposed to mix"). Variety between blend styles lives in
        # the swap choreography, not in whether the beats actually run
        # together; every blend with room for it (>=24 beats) now rides
        # B near-full through the dual. Shorter blends keep the simple
        # ramp - their 12-beat high-migration wouldn't fit.
        # (stem_bass_swap stays on its proven single-swap geometry: its
        # bass is STEM-gated, and staging it measured a pair-dependent
        # low-end hole - the stem restore and the staged gain/EQ moves
        # fight over the handover moment. Rare style, old shape.)
        long_stage = style == "long_blend" or (
            style in ("bass_swap", "filter_sweep") and nb >= 24)
        # EXIT RESERVATION: how much of the blend the swap can NEVER eat.
        # A's whole audible exit lives between the swap and the blend end,
        # and late-arriving bass in B (intro entries) plus the vocal-phrase
        # scan routinely pin the swap at this ceiling - at 8 beats the
        # two-stage exit dropped -9 dB in ~2s and the workhorse blend
        # "slammed down" (user-heard, 2026-07-22). The staged long blend
        # reserves a quarter of its span; the decisive styles keep 8.
        exit_res = K("exit_res_long") if long_stage else K("exit_res")
        # Never swap the bass into a BASSLESS stretch of B: cutting A's low
        # while B enters on intro atmosphere collapses the mix floor ~8 dB
        # (measured). Time the swap to where B's content actually carries
        # bass, clamped inside the blend.
        b_bassy = None
        for sec in (cand.sections or []):
            if sec["end_s"] <= plan["in_s"] + 0.5:
                continue
            if sec.get("bass_share", 0.3) >= 0.28:
                b_bassy = max(sec["start_s"], plan["in_s"])
                break
        if b_bassy is not None:
            k = round((b_bassy - plan["in_s"]) / max(cand.period_s, 1e-6))
            # B's bass arrival is a FLOOR on the swap, never a pull-forward:
            # the old form replaced the halfway default outright, so a track
            # entering already-bassy (most club mix-ins) swapped 4 BEATS in -
            # A spent the whole blend as a bassless ghost and the two songs
            # audibly coexisted for seconds (user: 'blending seems real
            # short'; measured median swap at 8% of the blend). Keep the swap
            # no earlier than halfway, no later than exit_res beats before
            # the end (the exit fade must survive as a real fade, not a cut).
            mid = min(max(mid, S0 + int(k * beat_out * RATE),
                          S0 + int(4 * beat_out * RATE)),
                      max(end - int(exit_res * beat_out * RATE), S0 + 1))
        else:
            # B NEVER gets bassy - no section after in_s carries real
            # bass, so there is no good moment to hand it the low end.
            # The halfway default here cut A's low with NOTHING arriving
            # to replace it: a measured 42 dB low-end hole for seconds
            # (spectral QA, 'Grand Bazaar' -> 'Faina'', 2026-08-02).
            # Swap as LATE as the exit reservation allows instead - A's
            # bass carries the room until its exit fade must begin.
            mid = max(end - int(exit_res * beat_out * RATE), S0 + 1)
        # VOCAL-PHRASE AWARENESS: the swap is the loudest EQ moment of the
        # blend - never land it on a sung line. Scan 4-beat slots from the
        # bass-ready point; take the first where BOTH decks are vocal-free
        # (fine demucs curve when stored, section means otherwise).
        lo_c = mid
        hi_c = max(end - int(exit_res * beat_out * RATE), lo_c + 1)
        c = lo_c
        step = int(4 * beat_out * RATE)
        while c <= hi_c and step > 0:
            b_src = plan["in_s"] + (c - S0) / RATE * rate_b
            a_src = plan["out_s"] - (end - c) / RATE * a_rate
            if self._vocal_at(cand, b_src) < 0.5                     and self._vocal_at(cur, a_src) < 0.5:
                mid = c
                break
            c += step
        # Harmonic clash makes overlap unforgivable: with incompatible
        # keys (after any pitch-shift rescue), B's melody waits until A is
        # essentially gone before opening. (Hoisted above the swap clamp
        # 2026-08-12: breakdown_swap's restore-clearance has to know
        # whether the mid opens at the swap or three-quarters later.)
        b_cam = _shift_camelot(cand.camelot, plan.get("pitch_st", 0) or 0)
        key_ok = camelot_compat(cur.camelot, b_cam) >= 0.55
        # Swap crossfade width: an instant low swap is a measured 8 dB
        # step; 4 beats stays decisive but spreads it. The staged long
        # blend widens to 6 - by then the highs have already migrated, so
        # the swap is the SECOND move, not the whole transition.
        swap_beats = K("swap_beats_long") if long_stage else K("swap_beats")
        # THE RESTORE GETS OUT OF THE DROP'S WAY (2026-08-12). breakdown_swap
        # parks B's entry on a build whose drop lands inside the blend, so
        # the low+mid restore and the drop onset are two big upward moves
        # that can land on the same beat - the 9.1 dB step that benched the
        # style (vs 5.3 dB solo). They have to be heard as two events: pull
        # the swap early enough that the restore FINISHES a phrase-quarter
        # before the drop. Everything downstream (A's exit fade, the mid
        # open, no_return_at) is derived from `mid`, so moving it here moves
        # the whole choreography coherently.
        if style == "breakdown_swap" and plan.get("b_drop_s"):
            drop_at = S0 + int((plan["b_drop_s"] - plan["in_s"])
                               / max(rate_b, 1e-6) * RATE)
            # The restore's LAST moment: B's low ramps over swap_beats from
            # `mid`, and its mid opens at mid_open_at over 4-8 beats. On-key
            # pairs open the mid at the swap, so the low ramp is the tail;
            # off-key ones defer the mid and that becomes the tail instead.
            tail = max(swap_beats, 4.0) * beat_out
            if not key_ok:
                tail = 0.75 * (end - mid) / RATE + 8 * beat_out
            latest = drop_at - int((_BDSWAP_RESTORE_CLEAR_BEATS * beat_out
                                    + tail) * RATE)
            # Never drag the swap in front of B's bass arrival or the blend
            # start - a swap into a bassless stretch is the 42 dB low-end
            # hole this file already learned about the hard way.
            floor_c = max(S0 + int(4 * beat_out * RATE),
                          S0 + int((b_bassy - plan["in_s"])
                                   / max(rate_b, 1e-6) * RATE)
                          if b_bassy is not None else S0)
            if latest < mid:
                mid = max(min(mid, latest), min(floor_c, mid))
        # A's exit fade spans swap -> blend end however late the swap lands.
        half_exit = max((end - mid) / RATE, 4 * beat_out)
        plan["no_return_at"] = mid               # the bass/mid handover
        # ONE MELODY AT A TIME - the mid-range twin of the one-bassline
        # rule. Both tracks' melodic content lives in the mids; letting
        # the incoming open its mids at the swap while the outgoing was
        # still audible for 12+ beats stacked clashing notes (user: 'lots
        # of overlap of clashing notes'). The mids now HAND OVER like the
        # bass does: B rides in on drums/air with mids shelved, the swap
        # crossfades low AND mid decisively, A keeps only a shadow of its
        # mids through its exit fade.
        sec_a = cur.section_at(plan["out_s"] - 1.0) or {}
        b_mid0 = (K("b_mid0_hot") if sec_a.get("mid_share", 0.33) > 0.42
                  else K("b_mid0"))
        b_high0 = (K("b_high0_hot") if sec_a.get("high_share", 0.25) > 0.30
                   else K("b_high0"))
        # LONG_BLEND = the STAGED MIGRATION (the classic technique): the
        # beats run together at near-full presence for bars, then the HIGH
        # END hands over subtly, and only then the mid/bass commitment.
        # Without the staging, even a 64-beat span perceptually collapsed
        # to the ~4-beat swap crossfade - the whole song identity flipped
        # in ~2s and every blend read fast (user). bass_swap/filter_sweep
        # keep the decisive single-swap geometry - that contrast IS the
        # style variety. (long_stage defined above the swap clamp - the
        # exit reservation depends on it.)
        if long_stage:
            b_high0 = min(b_high0, K("b_high0_long"))   # enter carved
            # B rides the long dual at near-FULL gain (that's the point),
            # so its mid shelf must sit lower than the ramping-gain case
            # or the shelf x 0.92 puts a second melody under A for bars.
            b_mid0 = min(b_mid0, K("b_mid0_long"))
        # (key_ok and swap_beats are settled ABOVE, before the swap clamp -
        # breakdown_swap's restore-clearance needs both to know how long the
        # restore takes. mid_open_at stays HERE because it reads the final
        # `mid`.)
        mid_open_at = mid if key_ok else \
            min(mid + int(0.75 * (end - mid)), end)
        # STEM_DRUM_SWAP / DRUM_BRIDGE enter on the DRUMS STEM alone: the
        # stems already strip B's bassline/melody/vocals, so the EQ carve
        # would only gut the drums themselves. Low sits at 0.55 (two
        # beat-locked kicks reinforce; full double-sub would pump the
        # limiter). ACAPELLA_IN enters on B's VOCAL stem alone - the voice
        # needs its mids/air open from the first bar.
        stem_entry = style in ("stem_drum_swap", "drum_bridge")
        vox_entry = style == "acapella_in"
        b_low0 = 0.55 if stem_entry else 0.0
        if stem_entry or vox_entry:
            b_mid0, b_high0 = 1.0, 1.0
        # QUIET-INTRO ENTRY TRIM: loudness comp (gain_db) levels whole
        # TRACKS, but the blend plays B's entry REGION against A's outro -
        # an atmospheric intro at full fader still sits ~10 dB under A's
        # final groove (measured: 'The Way' -> 'Ouahe', an 11s -12 dB hole
        # through the handover). Ride the incoming channel HOT by up to
        # +3 dB while its intro carries the blend - the DJ's gain-knob
        # move, automated - releasing to nominal as its own body arrives.
        b_trim = 1.0
        curve = (cand.row or {}).get("energy_curve") or []
        if curve:
            i0 = int(plan["in_s"] * 2)
            i1 = min(int((plan["in_s"] + (end - S0) / RATE * rate_b) * 2)
                     + 1, len(curve))
            if 0 <= i0 < i1:
                intro = sum(curve[i0:i1]) / (i1 - i0)
                med = sorted(curve)[len(curve) // 2]
                if intro > 0.05 and med > intro:
                    b_trim = min(med / intro, K("trim_cap"))
        # SILENT SETTLE: launch + sync the incoming deck FOUR BEATS before
        # anything is audible. The snap lands within the grids' own onset
        # accuracy (~25-35ms) and the PLL needs a few bars to trim that to
        # lock - with the audible blend starting at S0, its first bars
        # rode the raw snap error as audible flam (user-heard "a bit too
        # much flam" 2026-08-04; the flam-capped 16-beat blends then
        # measured med 33ms grid delta because the unsettled head
        # dominated). Four beats at gain zero puts the settling where
        # nobody can hear it. Cue rolls back the same four beats so the
        # musical entry still lands at in_s exactly at S0.
        settle = int(8 * beat_out * RATE)
        Sq = max(S0 - settle, now_guard)
        cue_b = max(0.0, plan["in_s"]
                    - (S0 - Sq) / RATE * rate_b)
        ev += [
            {"at": Sq, "cmd": "cue", "deck": incoming, "time_s": cue_b},
            {"at": Sq, "cmd": "rate", "deck": incoming, "value": rate_b},
            # Incoming: bass cut, mids shelved, highs carved - drums + air.
            {"at": Sq, "cmd": "eq", "deck": incoming, "low": b_low0,
             "mid": b_mid0, "high": b_high0, "ramp_s": 0.01},
            {"at": Sq, "cmd": "gain", "deck": incoming, "value": 0.0,
             "ramp_s": 0.01},
            {"at": Sq, "cmd": "start", "deck": incoming},
            {"at": Sq, "cmd": "sync", "slave": incoming, "master": active,
                 "bias_beats": sync_bias, "audio_pll": sync_audio},
        ]
        if stem_entry:
            ev.append({"at": S0, "cmd": "stem_gains", "deck": incoming,
                       "gains": {"drums": 1.0, "bass": 0.0, "other": 0.0,
                                 "vocals": 0.0}, "ramp_s": 0.01})
        elif vox_entry:
            ev.append({"at": S0, "cmd": "stem_gains", "deck": incoming,
                       "gains": {"vocals": 1.0, "drums": 0.0, "bass": 0.0,
                                 "other": 0.0}, "ramp_s": 0.01})
        elif style == "stem_bass_swap":
            # B enters whole EXCEPT its bassline - the stem-clean version
            # of the EQ carve: no spill, drums/melody untouched.
            ev.append({"at": S0, "cmd": "stem_gains", "deck": incoming,
                       "gains": {"drums": 1.0, "bass": 0.0, "other": 1.0,
                                 "vocals": 1.0}, "ramp_s": 0.01})
        if plan.get("duck_vocal_a"):
            # VOCAL DUCK: two sung passages overlap - A's voice steps
            # aside for the blend instead of the whole seam fading.
            ev.append({"at": S0, "cmd": "stem_gains", "deck": active,
                       "gains": {"vocals": K("duck_depth"), "drums": 1.0,
                                 "bass": 1.0, "other": 1.0},
                       "ramp_s": K("duck_beats") * beat_out})
        if long_stage:
            span_s = (end - S0) / RATE
            # KNOWN DEFECT, 16-BEAT BLENDS (found 2026-08-08, NOT fixed -
            # fixing it changes how those seams sound and wants ears, per
            # the tuning note above). When the swap lands at the blend
            # start, `mid` == S0 (+/- a sample), so the ramp to 1.0 below
            # REPLACES this stage-1 ramp in the same instant - set_gain
            # overwrites any pending ramp. B then arrives at full
            # immediately instead of riding under A, and stage1_gain /
            # stage1_frac are dead knobs on those seams.
            #
            # Measured over 40 planned long_blends: stage-1 survives 17.2s
            # at 64 beats and 25.4s at 96, but 0.000s on every one of the
            # four 16-beat seams. 36/40 are healthy; the collapse is
            # confined to the short end.
            #
            # If fixed: either skip this event when `mid` is within a beat
            # of S0, or keep the swap off the blend start so a short blend
            # still gets a staged entry. Either way, A/B it by ear first.
            #
            # Stage 1 - BEATS TOGETHER: B rises to near-full presence over
            # the first third and RIDES there (drums+air under A, EQ keeps
            # one bassline / one melody), instead of still creeping up
            # when the swap arrives.
            ev.append({"at": S0, "cmd": "gain", "deck": incoming,
                       "value": K("stage1_gain") * b_trim,
                       "ramp_s": K("stage1_frac") * span_s})
            # TRIM RELEASES AT THE BASS HANDOVER (2026-08-05): the hot
            # entry trim exists to lift a CARVED, bass-less entry over
            # A's outro. Once B's low opens, B is the mix's foundation
            # and must sit at nominal - the trim riding ~18s past the
            # handover measured as a +3.6 dB 'double bass' that was
            # really B's own bass at +2.6 dB over its post-swap self
            # (King of the Streets -> Fever Dream, gate-caught).
            ev.append({"at": mid, "cmd": "gain", "deck": incoming,
                       "value": 1.0,
                       "ramp_s": max(swap_beats, 4) * beat_out})
            # Stage 2 - THE SUBTLE HIGH SWAP: hats/air hand over across 12
            # beats, ending before the earliest possible swap (the EQ ramp
            # clock is shared per deck - overlapping ramps stretch each
            # other).
            hi_at = S0 + int(K("high_swap_at") * (end - S0))
            ev.append({"at": hi_at, "cmd": "eq", "deck": incoming,
                       "high": 1.0, "ramp_s": 12 * beat_out})
            ev.append({"at": hi_at, "cmd": "eq", "deck": active,
                       "high": 0.35, "ramp_s": 12 * beat_out})
            # Stage 2.5 - A starts LEANING OUT before the swap: a gentle
            # glide to 0.85 from mid-blend, so however late the swap lands
            # the handover reads as one continuous slope instead of full-
            # presence-then-cliff (B is already riding at 0.92 with highs
            # migrating - the room never thins).
            pre = S0 + int(K("pre_dip_at") * (end - S0))
            if pre < mid:
                ev.append({"at": pre, "cmd": "gain", "deck": active,
                           "value": K("pre_dip_gain"),
                           "ramp_s": max((mid - pre) / RATE, 2 * beat_out)})
        else:
            ev.append({"at": S0, "cmd": "gain", "deck": incoming,
                       "value": b_trim, "ramp_s": half})
            # Same trim-release rule as the staged path: nominal once
            # B's low opens at the swap.
            ev.append({"at": mid, "cmd": "gain", "deck": incoming,
                       "value": 1.0,
                       "ramp_s": max(swap_beats, 4) * beat_out})
        if style == "drum_bridge":
            # The generic low/mid handover GUTTED the bridge: A's drums
            # minus lows minus mids is a transient shadow, and the room
            # emptied for 8 beats ('everything dropped' - user, Vagrant ->
            # Undress). A percussion break needs BOTH kits full-bodied:
            # keep mids/highs open, split the low end instead of cutting.
            ev += [
                {"at": mid, "cmd": "eq", "deck": active, "low": 0.45,
                 "mid": 1.0, "high": 1.0, "ramp_s": swap_beats * beat_out},
                {"at": mid, "cmd": "eq", "deck": incoming, "low": 0.75,
                 "mid": 1.0, "high": 1.0, "ramp_s": swap_beats * beat_out},
            ]
        else:
            ev += [
            # Stage 3 - the swap downbeat: low AND mid hand over. The low
            # ramps are STAGGERED (2026-08-05): simultaneous linear
            # crossfades put both basslines at ~70% mid-swap and bass-
            # heavy pairs summed +3.6 dB (gate-measured, King of the
            # Streets -> Fever Dream). A's low leaves on the downbeat;
            # B's arrives from 40% in, so the overlap integral stays
            # under the double-bass bar while the handoff still spreads.
            {"at": mid, "cmd": "eq", "deck": active, "low": 0.0,
             "mid": 0.25, "ramp_s": swap_beats * beat_out},
            # (stem styles skip the stagger: their B low is ALSO gated
            # by the bass-stem restore at the swap - delaying the EQ on
            # top measured as a low-end hole, spectral gate 2026-08-05.)
            {"at": mid + (int(0.4 * swap_beats * beat_out * RATE)
                          if style in ("long_blend", "bass_swap",
                                       "filter_sweep") else 0),
             "cmd": "eq", "deck": incoming, "low": 1.0,
             "ramp_s": (0.6 if style in ("long_blend", "bass_swap",
                                         "filter_sweep") else 1.0)
             * swap_beats * beat_out},
            {"at": mid, "cmd": "eq", "deck": incoming,
             "high": 1.0, "ramp_s": swap_beats * beat_out},
            # A key-clash-delayed mid open happens with B carrying the mix
            # alone - opening the shelf (up to +10 dB of mid band) over 4
            # beats measured as a 6.6 dB mix lurch. A is ~gone by then, so
            # take 8 beats; on-key opens stay at 4 (they cross A's mids).
            {"at": mid_open_at, "cmd": "eq", "deck": incoming, "mid": 1.0,
             "ramp_s": (4 if key_ok else 8) * beat_out},
        ]
        if style == "loop_in":
            # B enters LOOPING its first bar under A - the stutter builds
            # tension - tightens to half a bar, then releases into the
            # full track at the swap.
            lp = 4 * cand.period_s
            ev += [
                {"at": S0, "cmd": "loop", "deck": incoming,
                 "start_s": plan["in_s"], "end_s": plan["in_s"] + lp},
                {"at": S0 + int(16 * beat_out * RATE), "cmd": "loop",
                 "deck": incoming, "start_s": plan["in_s"],
                 "end_s": plan["in_s"] + lp / 2.0},
                {"at": mid, "cmd": "clear_loop", "deck": incoming},
            ]
        if style == "stem_drum_swap":
            # The swap opens B's full stem set with its bass/EQ arrival,
            # and A collapses to its DRUMS stem - what survives A's EQ
            # (low cut, mid shelf) is pure percussion riding over B: the
            # classic percussion tail, then the normal fade takes it out.
            ev += [
                {"at": mid, "cmd": "stem_gains", "deck": incoming,
                 "gains": {"drums": 1.0, "bass": 1.0, "other": 1.0,
                           "vocals": 1.0}, "ramp_s": swap_beats * beat_out},
                {"at": mid, "cmd": "stem_gains", "deck": active,
                 "gains": {"drums": 1.0, "bass": 0.0, "other": 0.0,
                           "vocals": 0.0}, "ramp_s": swap_beats * beat_out},
            ]
        elif style == "drum_bridge":
            # THE PERCUSSION BREAK: at the swap A also collapses to drums;
            # both rhythm sections ride together for bridge_beats with all
            # harmonic content gone (the key-clash rescue), then B opens
            # its full mix and takes the floor.
            bridge = int(plan.get("bridge_beats", 8) * beat_out * RATE)
            # B's full mix opens TWO BEATS BEFORE the bridge ends, so the
            # room refills while A's drums are still carrying - the
            # handoff overlaps instead of leaving a gap.
            reopen = mid + bridge - int(2 * beat_out * RATE)
            ev += [
                {"at": mid, "cmd": "stem_gains", "deck": active,
                 "gains": {"drums": 1.0, "bass": 0.0, "other": 0.0,
                           "vocals": 0.0}, "ramp_s": swap_beats * beat_out},
                {"at": reopen, "cmd": "stem_gains", "deck": incoming,
                 "gains": {"drums": 1.0, "bass": 1.0, "other": 1.0,
                           "vocals": 1.0}, "ramp_s": 4 * beat_out},
                {"at": reopen, "cmd": "eq", "deck": incoming, "low": 1.0,
                 "mid": 1.0, "high": 1.0, "ramp_s": 4 * beat_out},
            ]
        elif style == "stem_bass_swap":
            # The swap trades the actual BASS STEMS - one bassline at all
            # times, zero crossover spill.
            ev += [
                {"at": mid, "cmd": "stem_gains", "deck": incoming,
                 "gains": {"drums": 1.0, "bass": 1.0, "other": 1.0,
                           "vocals": 1.0}, "ramp_s": swap_beats * beat_out},
                {"at": mid, "cmd": "stem_gains", "deck": active,
                 "gains": {"drums": 1.0, "bass": 0.0, "other": 1.0,
                           "vocals": 1.0}, "ramp_s": swap_beats * beat_out},
            ]
        elif style == "acapella_in":
            # B's full mix lands at the swap - the voice that rode A's
            # bed gets its own instrumental underneath it.
            ev.append({"at": mid, "cmd": "stem_gains", "deck": incoming,
                       "gains": {"drums": 1.0, "bass": 1.0, "other": 1.0,
                                 "vocals": 1.0}, "ramp_s": 2 * beat_out})
        if style == "acapella_out":
            # A's exit is NOT a fade-to-nothing: at the blend boundary A
            # collapses to its VOCAL stem and rides B's full instrumental
            # for tail_beats - the acapella tail. Gated upstream on A
            # actually singing there, B staying instrumental under it,
            # and tight key fit. The voice sits at 0.8 gain with mids
            # open; drums/bass/other are gone via stems, so no EQ fight.
            tail_beats = plan.get("tail_beats", 16)
            tail_end = end + int(tail_beats * beat_out * RATE)
            ev += [
                {"at": mid, "cmd": "gain", "deck": active, "value": 0.5,
                 "ramp_s": 0.6 * half_exit},
                {"at": end, "cmd": "stem_gains", "deck": active,
                 "gains": {"vocals": 1.0, "drums": 0.0, "bass": 0.0,
                           "other": 0.12}, "ramp_s": 2 * beat_out},
                {"at": end, "cmd": "eq", "deck": active, "low": 0.0,
                 "mid": 1.0, "high": 1.0, "ramp_s": 2 * beat_out},
                {"at": end, "cmd": "gain", "deck": active, "value": 0.8,
                 "ramp_s": 2 * beat_out},
                {"at": tail_end, "cmd": "gain", "deck": active,
                 "value": 0.0, "ramp_s": 8 * beat_out},
            ]
        elif style == "melody_carry":
            # A's exit is its MELODY BED: at the blend boundary A
            # collapses to its 'other' stem (pads/leads, no drums/bass/
            # voice) and sustains under B's full mix for tail_beats -
            # harmonic glue for tight-key pairs, then it breathes out.
            tail_beats = plan.get("tail_beats", 16)
            tail_end = end + int(tail_beats * beat_out * RATE)
            ev += [
                {"at": mid, "cmd": "gain", "deck": active, "value": 0.5,
                 "ramp_s": 0.6 * half_exit},
                {"at": end, "cmd": "stem_gains", "deck": active,
                 "gains": {"other": 1.0, "vocals": 0.0, "drums": 0.0,
                           "bass": 0.0}, "ramp_s": 2 * beat_out},
                {"at": end, "cmd": "eq", "deck": active, "low": 0.0,
                 "mid": 1.0, "high": 1.0, "ramp_s": 2 * beat_out},
                {"at": end, "cmd": "gain", "deck": active, "value": 0.7,
                 "ramp_s": 2 * beat_out},
                {"at": tail_end, "cmd": "gain", "deck": active,
                 "value": 0.0, "ramp_s": 8 * beat_out},
            ]
        elif style == "drum_bridge":
            # A holds NEAR-FULL through the percussion bridge (its drums
            # are half the point), then leaves fast once B's full mix
            # lands.
            bridge = int(plan.get("bridge_beats", 8) * beat_out * RATE)
            ev += [
                {"at": mid, "cmd": "gain", "deck": active, "value": 0.9,
                 "ramp_s": 2 * beat_out},
                {"at": mid + bridge, "cmd": "gain", "deck": active,
                 "value": 0.0, "ramp_s": 4 * beat_out},
            ]
        else:
            ev += [
                # Outgoing leaves over the rest of the blend (bass already
                # gone). TWO-STAGE fade ~ equal-power: a single linear ramp
                # loses most of its dB in its final second (measured 6.6 dB
                # mix steps at the fade tail once the swap moved mid-blend);
                # dropping to -9 dB first means the terminal collapse
                # happens with A already buried under B.
                {"at": mid, "cmd": "gain", "deck": active, "value": 0.35,
                 "ramp_s": 0.6 * half_exit},
                {"at": mid + int(0.6 * half_exit * RATE), "cmd": "gain",
                 "deck": active, "value": 0.0, "ramp_s": 0.4 * half_exit},
            ]
        # Release the quiet-intro trim once B's own body carries the room.
        if b_trim > 1.02:
            ev.append({"at": end, "cmd": "gain", "deck": incoming,
                       "value": 1.0, "ramp_s": 16 * beat_out})
        # Glue the overlap: duck B a few dB on A's kicks until the swap.
        # DEEPER when the groove offsets differ: B's hits land 20-30ms off
        # A's, and pushing them down on A's kicks masks the flam the EQ
        # staging can't touch (percussion transients live in the open
        # mids/highs).
        d_off_ev = abs(cur.kick_offset_s - cand.kick_offset_s)
        duck_depth = 0.18 if d_off_ev <= 0.02 else \
            min(0.32, 0.18 + 5.0 * (d_off_ev - 0.02))
        ev += [{"at": S0, "cmd": "duck", "on": True, "depth": duck_depth},
               {"at": mid, "cmd": "duck", "on": False}]
        if style == "filter_sweep":
            # A leaves through a rising resonant high-pass instead of a
            # plain fade: its weight thins musically over the second half
            # while B (bass now in) carries the floor.
            ev += [
                {"at": mid, "cmd": "filter", "deck": active, "mode": "hp",
                 "cutoff_hz": 60.0, "q": 2.2},
                {"at": mid, "cmd": "filter", "deck": active,
                 "cutoff_hz": 3200.0,
                 "ramp_s": max((end - mid) / RATE, 0.5)},
            ]
        if style == "loop_roll_exit":
            ls = plan["loop_start_s"]
            ev += [
                {"at": S0, "cmd": "loop", "deck": active,
                 "start_s": ls, "end_s": ls + 16 * cur.period_s},
                {"at": S0 + int(K("roll_shrink1") * beat_out * RATE),
                 "cmd": "loop", "deck": active, "start_s": ls,
                 "end_s": ls + 8 * cur.period_s},
                {"at": S0 + int(K("roll_shrink2") * beat_out * RATE),
                 "cmd": "loop", "deck": active, "start_s": ls,
                 "end_s": ls + 4 * cur.period_s},
            ]
        stop_at = end + int(4 * beat_out * RATE)
        if style in ("acapella_out", "melody_carry"):
            # the vocal/melody tail plays past `end`
            stop_at = end + int((plan.get("tail_beats", 16) + 12)
                                * beat_out * RATE)
        ev += [{"at": stop_at, "cmd": "stop", "deck": active},
               {"at": stop_at, "cmd": "end_sync"},
               {"at": stop_at, "cmd": "clear_loop", "deck": active}]
        self._glide_home(ev, incoming, rate_b, stop_at)
        _apply_fade_curve(ev, style)
        return ev, stop_at, S0

    def preview_events(self, plan, cur, cand):
        """The EXACT automation a transition will run, timed from a zeroed
        clock with deck A cued at the blend start - for the planner's mix
        view and offline auditions. Returns (events, swap_at, blend_at) with
        'at' in samples where blend_at corresponds to plan['out_s']."""
        # The run-up must EXCEED the style's whole pre-roll (blend/run-in),
        # or the no-past-events guard clamps the transition into a
        # degenerate splice (snapshot clock is treated as NOW).
        pre = (plan.get("beats", 16) + 24) * cur.period_s
        snapshot = {"clock": 0,
                    "decks": {"a": {"time_s": plan["out_s"] - pre,
                                    "rate": 1.0}}}
        return self.build_events(plan, snapshot, "a", "b", cur, cand)

    @staticmethod
    def _glide_home(ev, deck, rate, at):
        """After the swap the new dominant deck glides to its natural rate."""
        if abs(rate - 1.0) > 1e-4:
            ev.append({"at": at, "cmd": "rate", "deck": deck, "value": 1.0,
                       "ramp_s": abs(rate - 1.0) / GLIDE_PER_S})
