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
from lib.dj.features import _finite, hardness_raw
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
# against by s_rate (Gaussian, sigma 0.045) and still HARD-GATED on
# risky material at plan time (stretch>5.5%_risky) - widening here does
# not hand a shaky grid an 10% blend.
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
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# Key compatibility (Camelot wheel)
# --------------------------------------------------------------------------

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

    def _pair_blendable(self, cur, cand):
        """Could this pair support an overlapped blend? The CHEAP mirror
        of the plan-time screens (cached file-backed lookups only, no
        audio - score() runs this ~640x per pick): beat power at the
        seam-relevant regions (asymmetric bars: B's intro must groove,
        A's exit only hands off), and a trusted grid on both sides
        (conf, or profile-verified). Used as a selection LEAN so the DJ
        picks partners it can actually mix into - not as a gate."""
        from lib.dj import beatpower as _bp
        bs_b = _bp.band_scores(cand.id, region="in") or {}
        ev_b = [v for v in (bs_b.get("low"), _bp.scores().get(cand.id))
                if v is not None]
        if ev_b and max(ev_b) < _bp.BLEND_MIN:
            return False
        bs_a = _bp.band_scores(cur.id, region="out") or {}
        ev_a = [v for v in (bs_a.get("low"), _bp.scores().get(cur.id))
                if v is not None]
        if ev_a and max(ev_a) < _bp.BLEND_MIN_EXIT:
            return False
        for t in (cur, cand):
            if (t.bpm_conf or 0.0) < 0.7 \
                    and _bp.profile_coverage(t.id) < 0.6:
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
        # CONDITIONAL ON A VERIFIED GRID (2026-08-06), the same rule the
        # plan-time gate already uses: "deep stretch is only fatal on
        # RISKY material" (see stretch>5.5%_risky). A blanket cliff here
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
        s_blend = 1.0 if self._pair_blendable(current, cand) else 0.45
        total *= s_blend
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
                 "wall": s_wall, "conf": s_conf, "bpm_arc": s_bpm_arc}
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
    def best_pair(self, cur, cand, after_s=None):
        """Best (A-exit, B-entry) combination, or None. Never lets two
        busy/vocal sections blend over each other."""
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
        if not outs:
            return None
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
        i_pre = []
        for i in cand.mix_ins[:8]:
            sec_b = cand.section_at(min(i["time_s"] + 1.0,
                                        cand.duration_s - 1.0))
            if sec_b is None:
                i_pre.append((i, None, 0.0, "", 0.0))
                continue
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
            i_pre.append((i, sec_b, voc_b, ml_b, early_b))
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
        for oi, (o, sec_a, voc_a, ml_a) in enumerate(o_pre):
            if sec_a is None:
                continue
            of = out_fit(sec_a, voc_a, ml_a)
            # What leaving EARLY costs (see the `outs` comment above): a
            # candidate below the drawn budget stays on the table, decaying
            # with how early it is, so a good exit ~a minute early can beat
            # a dead one that merely lands on time.
            bud = 1.0
            if after_s is not None and o["time_s"] < after_s:
                bud = math.exp(-(after_s - o["time_s"]) / BUDGET_TAU_S)
            busy_a = sec_a.get("busyness") or 0.0
            ra = sec_a.get("rhythm_density") or 0.0
            ea = sec_a.get("energy") or 0.0
            if _body_e > 0.2:
                of *= 0.25 + 0.75 * min(ea / _body_e, 1.0)
            for ii, (i, sec_b, voc_b, ml_b, early_b) in enumerate(i_pre):
                if sec_b is None:
                    continue
                busy_b = sec_b.get("busyness") or 0.0
                fit = of * in_fit(sec_b, voc_b, ml_b)
                quiet = 1.0 - 0.5 * min(busy_a + busy_b, 1.6) / 1.6
                # BLEND WHERE THE BEATS ARE: a beat-matched blend is only
                # audible as beat-matched if BOTH sides carry rhythm and
                # comparable energy - otherwise it just reads as a fade.
                rb = sec_b.get("rhythm_density") or 0.0
                eb = sec_b.get("energy") or 0.0
                beaty = ra >= 1.2 and rb >= 1.2
                rhythm_fit = (1.3 if beaty else 0.55) \
                    * math.exp(-((ea - eb) ** 2) / (2 * 0.4 ** 2))
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
                         * (0.25 + 0.75 * min(hole / 0.25, 1.0)) * bud)
                if best is None or score > best["score"]:
                    best = {"out_s": o["time_s"], "in_s": i["time_s"],
                            "out_hint": o.get("style_hint", "blend"),
                            "in_hint": i.get("style_hint", "blend"),
                            "score": round(score, 5), "beaty": beaty,
                            "kinds": (sec_a.get("kind"), sec_b.get("kind")),
                            "busy": (round(busy_a, 2), round(busy_b, 2))}
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
                        force_style=None, test_gates=False):
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
        thresholds are mostly right."""
        pair = meta.get("pair") if meta else None
        if pair is None or (after_s is not None
                            and pair["out_s"] < after_s):
            pair = self.best_pair(cur, cand, after_s=after_s)
        if pair is None:
            # Last resort: exit on the last downbeat-aligned half minute -
            # but NEVER before the requested after-point (a late entry on
            # a short tail inverted out<in, and the seam fired seconds
            # after the song started).
            out_fb = max(cur.duration_s - 35.0, cur.duration_s * 0.6)
            if after_s is not None:
                out_fb = min(max(out_fb, after_s),
                             max(cur.duration_s - 8.0, out_fb))
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
        testable = ("grid_conf<0.7", "cut_needs_grid_conf>=0.8",
                    "grid_conf<0.5", "downbeat_conf", "kick_offset>28ms",
                    "key_fit<0.8", "anti_streak", "kick_clash",
                    "swing_clash", "meter_clash", "half_time")

        # RETIRED (2000-pair audit + live record, 2026-08-02):
        # cut_at_drop reached 2% of menus, won 0/2000 rolls, and measured
        # 2/5 rough live - phrase_cut does its job without the drop
        # dependency (choreography kept so old pinned sets degrade
        # politely: a pin shows 'refused (retired)'). bassline_layer
        # (10% of menus, 3 live plays ever) and double_drop (the fx
        # one-shot holdout; loop_build carries the drop spectacle) are
        # REMOVED outright - the kill keeps their old pins refusing
        # politely, the choreography is gone.
        kill(("cut_at_drop", "bassline_layer", "double_drop"), "retired")
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
        # BENCHED pending a choreography fix (2026-08-04): breakdown_swap
        # restores B's low+mid exactly at the swap, which the style
        # deliberately parks just before B's drop - EQ restore and drop
        # onset stack into a gate-measured 9.1 dB slam on pairs that pass
        # every material screen. Fix = complete the restore >=4 beats
        # pre-drop; until then the style stays off the menu.
        kill("breakdown_swap", "benched_lurch_fix_pending")

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
        _local_ok = (
            _bpv.phase_offset(cur.id, at_s=pair["out_s"]) is not None
            and _bpv.phase_offset(cand.id,
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
        if low_conf or not pair.get("beaty", True):
            # No confident grid, or the best seam is BEATLESS on one side:
            # a beat-matched blend there is inaudible as such and just
            # smears - do a deliberate clean fade on the phrase instead.
            style = "long_fade"
            fade_reason = ("tempo_clash" if (meta or {}).get("tempo_clash")
                           else "grid_conf<0.5" if low_conf
                           else "beatless_seam")
        else:
            if (cur.downbeat_conf < 0.15 or cand.downbeat_conf < 0.15):
                kill("cut_at_drop", "downbeat_conf")
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
            # cut_at_drop earns a STRICTER bar than the rest of its tier.
            # Measured over 560 logged seams (tools/dj/dj_review.py): median
            # flam 0.247 beats against 0.061-0.068 for every blend style
            # and 0.034-0.056 for the other short ones - four times worse
            # than anything else the DJ plays. It is the only technique
            # that hard-cuts with zero overlap for the PLL to settle in,
            # and it enters at a pre-drop point picked OUTSIDE the pair
            # scan, so nothing else has vetted that landing. Small sample
            # (n=4), so this tightens the gate rather than removing the
            # style: on a strong grid the cut is the right move.
            if min(cur.bpm_conf, cand.bpm_conf) < 0.8:
                kill("cut_at_drop", "cut_needs_grid_conf>=0.8")
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
            if abs(cur.kick_offset_s - cand.kick_offset_s) > 0.028 \
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
            if abs(cur.kick_offset_s - cand.kick_offset_s) > 0.020 \
                    and not _local_ok:
                kill(("long_blend", "bass_swap", "filter_sweep",
                      "stem_bass_swap", "melody_carry", "breakdown_swap",
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
            # HARD STRETCH WALL AT PLAN TIME (2026-08-05). Selection's
            # 5.5% wall is a relative lean - a tempo-outlier track with
            # no better partners still pairs beyond it (a duplicate
            # Swing Star analyzed at 79.7bpm paired with 85bpm at 6.2%
            # stretch and rendered a 187ms-median wander; gate-caught
            # three times before this landed). A blend's PLL cannot hold
            # material stretched past the wall - the plan must refuse
            # absolutely what selection only discourages. echo_out's
            # beat-matched run-in is the same physics (its lock failed
            # next, same pair, once the blends were walled).
            # CONDITIONAL (2026-08-05, same evening): the blanket wall
            # immediately faded a clean 5.5-8% rescue-tier pair the user
            # heard ("why isn't this resolved") - while Negev->Neptunes
            # at 6.2% stretch measured 5.6ms kick-to-kick. Deep stretch
            # is only fatal on RISKY material: swing, patchy phase, or
            # a grid nobody verified. Steady verified tracks keep their
            # rescue-tier blends.
            _risky = ((rt or {}).get("swing_delta", 0.0) > 0.05
                      or min(cur.bpm_conf, cand.bpm_conf) < 0.8
                      or _bpv.profile_coverage(cur.id) < 0.8
                      or _bpv.profile_coverage(cand.id) < 0.8)
            if abs(rate - 1.0) > 0.055 and _risky:
                kill(_overlap + ("echo_out",), "stretch>5.5%_risky")
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
            for t, side, reg, bar in (
                    (cur, "A", _reg_a, _bp.BLEND_MIN_EXIT),
                    (cand, "B", _reg_b, _bp.BLEND_MIN)):
                bs = _bp.band_scores(t.id, region=reg) or {}
                evid = [v for v in (bs.get("low"),
                                    _bp.scores().get(t.id))
                        if v is not None]
                if evid and max(evid) < bar:
                    kill(_overlap, f"no_beat_power_{side}")
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
            _style_bands = {
                "long_blend": ("high",), "bass_swap": ("high",),
                "filter_sweep": ("high",),
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
                        if hi_ >= 1.5 and lo_ < 1.2:
                            kill(st_, f"band_clash_{bd}")
                            break
            if rt_sure:
                if abs(rt.get("mult", 1.0) - 1.0) > 1e-6:
                    kill(_overlap, "tempo_multiple_read")
                if rt["kick_agreement"] < 0.35:
                    kill(_overlap, "kick_clash")
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
                if rt is not None and rt["swing_delta"] > 0.055:
                    # Swung vs straight flams every offbeat for the whole
                    # overlap; only removing one percussion bed fixes it.
                    # stem_drum_swap does exactly that; drum_bridge keeps
                    # BOTH percussion beds - it showcases the clash.
                    for k in ("long_blend", "filter_sweep",
                              "loop_roll_exit", "drum_bridge"):
                        weights[k] = weights.get(k, 0.0) * 0.3
                    weights["stem_drum_swap"] = \
                        weights.get("stem_drum_swap", 0.0) * 2.0
                fl = rt.get("flam_ms") if rt is not None else None
                if fl is not None and 15.0 <= fl <= 80.0:
                    # Machine-gun near-misses: the short punchy styles
                    # expose them raw (same reasoning as the groove-offset
                    # gate above, measured one level finer).
                    for k in ("cut_at_drop", "echo_out", "loop_build"):
                        weights[k] = weights.get(k, 0.0) * 0.3
            # cut_at_drop needs a pre-drop entry in B - ANY of B's pre_drop
            # mix-ins qualifies, not just the best-scoring pair's (gating on
            # pair["in_hint"] starved the style to literally zero uses
            # across a 125-track library: pre_drop points rarely win the
            # generic pair scoring).
            pre_drops = [p for p in cand.mix_ins
                         if p.get("style_hint") == "pre_drop"]
            if not pre_drops:
                kill("cut_at_drop", "no_pre_drop_in_B")
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
            a_stems = getattr(cur, "has_stems", False)
            b_stems = getattr(cand, "has_stems", False)
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
            # KEY CLASH is drum_bridge's home turf: both tracks strip to
            # percussion while the harmony resets - boost it exactly where
            # everything harmonic struggles.
            if camelot_compat(cur.camelot, cand.camelot) < 0.5:
                # Key clash: the harmonic-reset styles are the honest
                # answers - percussion bridge (stems) or spinback (none).
                if weights.get("drum_bridge", 0.0) > 0.0:
                    weights["drum_bridge"] *= 2.5
                if weights.get("spinback_cut", 0.0) > 0.0:
                    weights["spinback_cut"] *= 2.0
            # breakdown_swap needs the sections to exist: A must have a
            # breakdown ahead of the exit region, B a build to enter on.
            bd_a = next((s for s in (cur.sections or [])
                         if s["kind"] == "breakdown"
                         and s["end_s"] > (after_s or 0.0)), None)
            bl_b = next((s for s in (cand.sections or [])
                         if s["kind"] == "build"), None)
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
                weights["long_fade"] = 2.0 * max(
                    weights.get("echo_out", 0.0), 0.4)
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
                "fade_reason": fade_reason}
        if gate_tested:
            # Which threshold this seam was allowed to cross, so the
            # verdict can be read as evidence about that threshold.
            diag["gate_test"] = gate_tested
        if force_style:
            # Planned-set pin outcome: honored, or refused by which gate.
            diag["style_pin"] = {
                "want": force_style, "honored": style == force_style,
                "why_not": (None if style == force_style
                            else gated.get(force_style) or fade_reason
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
        d_off_p = abs(cur.kick_offset_s - cand.kick_offset_s)
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
        _pv = (_bpl.phase_offset(cur.id, at_s=pair["out_s"]) is not None
               and _bpl.phase_offset(cand.id,
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
        if style == "cut_at_drop":
            # Enter at B's best PRE-DROP point (the style's whole premise),
            # not the generic pair in-point.
            pd = max((p for p in cand.mix_ins
                      if p.get("style_hint") == "pre_drop"),
                     key=lambda p: p.get("score", 0.0), default=None)
            if pd is not None:
                in_s = cand.nearest_downbeat(pd["time_s"])
        plan = {"style": style, "rate": rate,
                "out_s": out_s, "in_s": in_s, "beats": beats, "rhythm": rt,
                "pair_score": pair["score"], "cand_id": cand.id,
                "duck_vocal_a": duck_vocal,
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
        if (stretch_engine_name() == "vari"
                and style in ("long_blend", "bass_swap", "filter_sweep",
                              "stem_drum_swap", "acapella_out",
                              "stem_bass_swap", "drum_bridge",
                              "acapella_in", "melody_carry")
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
            A0 = max(S0 - int(K("fade_lead_a") * _ug * RATE), now_guard)
            B0 = max(S0 - int(K("fade_lead_b") * _ug * RATE), A0)
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
                {"at": B0, "cmd": "eq", "deck": incoming, "low": 0.0,
                 "mid": 1.0, "high": 1.0, "ramp_s": 0.01},
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
        # Harmonic clash makes overlap unforgivable: with incompatible
        # keys (after any pitch-shift rescue), B's melody waits until A is
        # essentially gone before opening.
        b_cam = _shift_camelot(cand.camelot, plan.get("pitch_st", 0) or 0)
        key_ok = camelot_compat(cur.camelot, b_cam) >= 0.55
        mid_open_at = mid if key_ok else \
            min(mid + int(0.75 * (end - mid)), end)
        # Swap crossfade width: an instant low swap is a measured 8 dB
        # step; 4 beats stays decisive but spreads it. The staged long
        # blend widens to 6 - by then the highs have already migrated, so
        # the swap is the SECOND move, not the whole transition.
        swap_beats = K("swap_beats_long") if long_stage else K("swap_beats")
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
