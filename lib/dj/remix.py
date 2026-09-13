"""Remix mode: the system plays PARTS of songs together and keeps changing them.

Two or three songs are live at once, each on its own deck, all locked to
one clock (the master song's tempo; the submix PLL holds every other deck)
and shifted toward one key. The four stem lanes - drums, bass, other,
vocals - are each assigned to one live song. Every phrase the conductor
makes a MOVE: a lane crosses to another live song, a lane rests for a
phrase, a song that holds no lane leaves, a new song (the brain's pick, to
the theme and the night's energy arc) comes in through one lane. Songs
enter and leave lane by lane, never whole, so what comes out is a
continuous recombination rather than a sequence of tracks.

STRUCTURE. A staged song is cued so that its landmark - the groove after
its first build, else its body - lands on the move that lets it in. A
planned move waits up to two bars for a section boundary of the master.
The vocal lane only crosses to a song that is singing there (section
vocalness, when the vocal pass has run), and leaves a song whose singing
stops. When the song holding most lanes reaches a breakdown, the
conductor may break with it (learned).

HYGIENE. Songs holding neither the bass nor the drums lane lose their lows
(EQ low 0.25 below 200 Hz) so one kick and one bass are heard; the `other`
lane sits at 0.8 under another song's vocal; every lane is trimmed to the
first song's stem levels (+-6 dB) so a louder song's drums do not shout.

SHAPES. A move is planned a bar early, so it has a bar to shape: a song
giving up its last lane leaves through a low-pass sweep or a one-beat drum
stutter, a lane that rests throws an echo. Deck FX are per deck, so a shape
runs only when the song holds exactly the lane that moves.

The harmonic guard runs on every move: a tonal lane (bass, other, vocals)
only crosses to a song whose shifted key fits the songs on the other tonal
lanes; drums are free. A song a lane still needs but whose body is running
out is looped on its last bars, and evicted after two phrases of looping.
Nothing here opens a device: the caller mounts `conductor.submix` on an
AudioEngine (the Perform tab) or pulls the mixer by hand (the gate).

Steering (the sliders): blend (0 = every move hands a lane to the newest
song, i.e. song follows song as a morph; 1 = every move recombines freely
across three songs), change_bars (a move every 4 / 8 / 16 / 32 bars),
vocal_freedom (how freely the vocal lane crosses), energy lean (on the
theme's arc target), theme, hold (freeze the lane map; runway rules still
apply), next (a move now).

Performing (the buttons): DROP (every lane to the newest song on the next
bar - the morph all at once), BREAK (every lane but one rests for a few
bars and they all come back on the bar), LOOP 4 / 8 (every live song loops
where it is; press again to release).

Learning: GOOD / BAD on the last move is stored (seam_feedback, style
"remix:<kind>:<lane>") and every conductor reads the record at start:
kinds of move you dislike happen less, kinds you like happen more, with a
Laplace prior so a handful of verdicts nudges and does not dictate.
"""
import json
import math
import os
import random
import threading
import time

import numpy as np

from lib.dj.submix import DJSubmix, RATE

STEMS = ("drums", "bass", "other", "vocals")
TONAL = ("bass", "other", "vocals")
MORPH_ORDER = ("drums", "bass", "other", "vocals")
DECKS = ("a", "b", "c", "d")     # three heard + one staged is the DJ's norm; the fourth is the operator's slot
LEAD_S = 0.25
RATE_MIN, RATE_MAX = 0.90, 1.10
CLASH_BELOW = 0.55
XFADE_BEATS = 2.0                # a lane crosses over this many beats
DROP_BEATS = 0.5                 # ...and on a DROP over this many
RUNWAY_S = 40.0                  # a song under this much playable time ahead is looped on its last bars
LOOP_BARS = 8
EXPIRE_PHRASES = 2               # ...and evicted after this many phrases of looping
MIN_STAY_PHRASES = 3             # a song's LAST lane cannot be taken before it has been heard this long
MIN_RUNWAY_S = 90.0              # a pick must have this much body from its entry point (one looped 2 s after entering)
REST_CHANCE = 0.12               # a free move that rests a lane for a phrase instead of crossing it
BREAK_BARS = 4                   # BREAK: every lane but one rests this long
SET_CYCLE_S = 90 * 60.0          # the theme's arc runs over this, as in the automixer
SNAP_BARS = 2                    # a planned move waits up to this long for a section boundary of the master
LOW_CARVE = 0.25                 # EQ low on songs holding neither bass nor drums
OTHER_DUCK = 0.6                 # the `other` lane under another song's vocal (-4.4 dB: a melody under a singer steps back)
VOCAL_MIN = 0.35                 # section vocalness for the vocal lane to cross INTO a song
VOCAL_GONE = 0.2                 # ...and below this the vocal lane leaves it
TRIM_MIN, TRIM_MAX = 0.5, 2.0    # per-stem level trim toward the first song's levels (-6 .. +6 dB)
AUTO_BREAK_CHANCE = 0.5          # a breakdown of the song holding most lanes may become a BREAK (learned)
TEMPO_STEP = 0.005               # the clock moves at most this much per bar on a tempo journey
TEMPO_SPAN_MAX = 0.06            # ...and at most this far from the opener's tempo
RECALL_PHRASES = 8               # a RECALL whose songs cannot come back within this many phrases is abandoned
VOICE_PHRASES = 3                # the arrangement policy: a voice is heard this many phrases before it MAY take the bed
VOICE_MAX_PHRASES = 4            # ...and takes it by then even if its drop never comes (a voice is not a home; six
                                 # phrases of a lone melody over the bed was flagged "nothing happening" 2026-09-13)
VOCAL_REST_BARS = 4              # the vocal lane rests only when the singing stops for this long, and comes back only
                                 # when it carries on this long (two-bar reads flapped rest/return every few seconds)
OTHER_VOICE = 0.5                # a VOICE's melody (another song's `other` over the bed) plays at this level: a texture, not
                                 # a second song (ten "clash" flags were a full melody over the bed, 2026-09-13)
CLASH_DUCK = 0.5                 # a voice whose chords clash with the bed HERE (fit under CLASH_BELOW) is held at this
                                 # level until the sections agree again (the guard only judged arrivals; three clash
                                 # flags came 20-40 s after a clean arrival when a section changed)
SETTLE_PHRASES = 4               # a new bed is heard for this many phrases before the next voice arrives (let it breathe)
ENTRY_FIT = 0.68                 # a song ARRIVING on a tonal lane must fit the room's keys this well (CLASH_BELOW keeps what plays)
KEY_SHIFTS = (1, -1, 2, -2)      # a staged song is shifted at most two semitones to fit the room (three sounded wrong)
SING_ON, SING_OFF = 0.12, 0.05   # the vocal stem's own envelope (of its loud level): singing above, silent below
STRIP_BARS = 4                   # the breakdown before a bed lands: the old bed's drums + bass out for this many bars
DROP_LOOK_BARS = 2               # a voice's drop counts as "now" when its groove starts within this many bars
RATED_KINDS = ("enter", "cross", "cross_new", "rest", "return", "drop", "break", "break_auto",
               "voice_in", "bed_to", "voice_out", "moment_in", "moment")
# MOMENTS: composed shapes the conductor runs as scripts of bar offsets from THE ONE (the literature's break
# routine: take away -> build -> a hole with the downbeat delayed -> bass drum and bass back on the one of a
# phrase). key -> (label, lead bars before the one, what it does)
MOMENTS = {
    "slam": ("SLAM", 2, "bass and drums out for the last two bars, a beat of nothing, everything back louder on the one"),
    "break": ("BREAK", 16, "drums and bass out for eight bars (the melody stays); the drums back thin and their lows return over eight; "
                          "half a bar of nothing; everything back louder on the one"),
    "sing": ("SING", 2, "at the hook: bass and drums out for two bars, the vocal alone for the room to sing; back louder on the third bar"),
    "takeover": ("TAKEOVER", 1, "the other song's bassline takes over on the one, loud, at its drop or hook; its drums follow at the phrase"),
    "bridge": ("BRIDGE", 8, "the bed fades out under this song's vocal; the other song's drums come in under it; a beat of nothing; "
                            "its bass and drums slam on the one, the vocal carries over"),
}
MOMENT_BOOST = 1.25              # bass and drums come back this much louder for a phrase (the peak guard keeps the bus honest)
MOMENT_LOOK_PHRASES = 4          # a moment lands on the target song's drop or hook when one comes within this many phrases
MOMENT_TIGHTEN = False           # the BREAK's build may tighten the bed's loop 2 -> 1 -> half a bar (the drum-roll effect) - OFF: the
                                 # user heard the first live BREAK as "painful in its execution"; a stutter build stays rejected
MOMENT_SWEEP = False             # ...or high-pass sweep the whole bed deck over the build - OFF for the same reason (it thins the
                                 # melody and vocal too); the build is the drums back thin, their lows returning (the deck's low shelf)
# the ear's five words on a moment (BAD on a moment asks which) and the ONE number each word moves - see tune_moment
MOMENT_WORDS = {"hole": "the hole was too long", "thin": "it sounded thin", "long": "it went on too long",
                "beat": "the return was off the beat", "place": "it landed in the wrong place"}
MOMENT_TUNE_DEFAULT = {
    "slam": {"hole_beats": 1.0, "out_bars": 2, "boost": MOMENT_BOOST, "payoff_only": False},
    "break": {"hole_beats": 2.0, "take_bars": 8, "build_low": LOW_CARVE, "boost": MOMENT_BOOST, "payoff_only": False},
    "sing": {"out_bars": 2, "boost": MOMENT_BOOST, "payoff_only": False},
    "takeover": {"hole_beats": 1.0, "boost": MOMENT_BOOST, "payoff_only": False},
    "bridge": {"hole_beats": 1.0, "fade_bars": 4, "boost": MOMENT_BOOST, "payoff_only": False},
}
ARR_EVERY_PHRASES = 1            # the arrangement moves at most once per this many phrases (the Director asks for 2: calm between moments)


def _compat(c1, c2):
    from lib.dj.brain import camelot_compat
    return camelot_compat(c1, c2)


def _shift(cam, s):
    from lib.dj.brain import _shift_camelot
    return _shift_camelot(cam, s)


class Song:
    def __init__(self, deck, track):
        self.deck = deck
        self.track = track
        self.rate = 1.0
        self.shift = 0
        self.compat = None
        self.staged_at = None            # submix clock it started playing (silent)
        self.entered = False             # has held a lane
        self.entered_k = None            # bar count at entry (order of arrival)
        self.leaving = False
        self.leave_clock = None
        self.idle_k = None               # bar count when it last lost its last lane
        self.loop = None                 # the runway loop (start_s, end_s)
        self.loop_k = None
        self.evicted = False
        self.stem_rms = {}               # per-stem RMS over the body (level trims)
        self.vox_sections = None         # the vocals stem's RMS per section: the measured singing map
        self.env = None                  # per-stem half-second envelopes (the strips' stem picture)
        self.vocal_data = bool(track.sections) and any((s.get("vocalness") or 0) > 0 for s in track.sections)
        self.auto_broke = False
        self.level = 1.0                 # the operator's fader for this song (deck gain)
        self.user_loop = None            # the operator's loop on this song, bars
        self.fx_until = None             # a shape's FX is running on this deck until this clock
        self.ready_k = None              # bar count from which a move may let it in (its landmark lands there)
        self.was_bed = False             # arrangement policy: it held the bed and is now a voice on its way out
        self.voice_since = None          # bar count since it has been a voice
        self.bed_since = None            # bar count since it holds the bed (drums + bass)
        self.bed_lands_k = None          # bar count when a planned bed change lands on it (the strip is running)

    def held(self, lanes):
        return [s for s in STEMS if lanes.get(s) == self.deck]

    def key(self):
        return _shift(self.track.camelot, self.shift) if self.track.camelot else ""


class RemixConductor:
    def __init__(self, db, music_root, library, theme=None, seed=None):
        from lib.dj.brain import Brain
        from lib.dj.themes import get_theme
        self.db, self.music_root = db, music_root
        self.library = [t for t in library if getattr(t, "has_stems", False) and not getattr(t, "excluded", False)]
        self.brain = Brain(list(self.library), get_theme(theme or "groove"), seed=seed) if self.library else None
        if self.brain is not None:
            try:
                self.brain.load_play_counts(db)       # cross-night freshness: the worn songs lean away here too
            except Exception:
                pass
        self.rng = random.Random(seed)
        self.submix = DJSubmix(deck_names=DECKS)
        self.songs = {}                  # deck -> Song
        self.lanes = {s: None for s in STEMS}
        self.lane_k = {}                 # lane -> bar count of its last cross (anti ping-pong)
        self.master = None               # deck name holding the clock
        self.master_bpm = None
        self.key_centre = None           # the session's key: every song is shifted toward it
        self.ref_rms = None              # the first song's stem levels: every later song is trimmed toward them
        self.pool_name = None            # the setlist the songs are confined to (None = the whole stem library)
        # steering
        self.blend = 0.7
        self.change_bars = 8
        self.vocal_freedom = 0.4
        self.energy_lean = 0.0
        self.hold = False
        self._land_k = None              # arrangement: bar count when a planned bed change lands (a strip is running)
        self._land_at = None             # ...and its clock (hygiene waits for it)
        self.wait_why = None             # arrangement: why the last phrase passed without a move, in words
        self.strip_before_bed = False    # the Director's MOMENTS: a four-bar breakdown before a new bed lands
        self.bass_moments = False        # the Director's MOMENTS: a bed change is bass-first (the bassline takes over, drums follow)
        self.lane_boost = {}             # lane -> (gain multiplier, until bar): a tactical "loud" on a lane, then back to its trim
        self._drums_follow = None        # deck whose drums complete a bass-first bed change when _land_k comes
        self._moment = None              # a composed MOMENT armed or running (see arm_moment); the arrangement pauses meanwhile
        self.moment_tune = self._load_moment_tune()   # per kind: the numbers the ear's words have moved (persisted per library)
        self.moment_stats = {}           # kind -> {"armed": n, "refused": {reason: n}}: how often each moment could be used
        self.calm_phrases = ARR_EVERY_PHRASES         # the arrangement moves at most once per this many phrases
        self._arr_k = None               # bar of the arrangement's last move
        self._calm_until = None          # no arrangement move before this bar (a phrase of plain play after a moment)
        self.plan = None                 # THE PLAN: the one next arrangement event, decided once and kept (see _plan_check)
        self._user_picks = set()         # track ids the operator staged (UP NEXT, a right-click): they arrive before the conductor's own picks
        self.reserve_pick = False        # the Director holds an UP NEXT that is not on a deck yet: the conductor stages no pick of its own
        self._verdict_forced_until = 0.0 # moves made because of a BAD are not rateable until this wall time (hear them first)
        self._force_move = False
        self.auto = 1.0                  # autopilot amount: 1 = the conductor moves every phrase, 0 = only the operator moves
        self.cross_beats = XFADE_BEATS   # how a lane crosses (the Director's mixing dial: cut / blend / morph)
        self.avoid_ids = set()           # songs played tonight by any engine (the Director's memory): never picked again
        self.policy = "arrangement"      # "arrangement" (a bed and voices, one change at a time) | "free" (any lane, any song)
        self.fx_level = 1.0              # the Director's FX dial: the chance a move gets a shape (0 = plain crossfades)
        self._cands_cache = (0.0, None, [])
        self._cue_req = {}               # deck -> song time to open / stage at (the timeline player's exact starts)
        self._open_lanes = None          # lanes the opener comes up on (None = all)
        # performing
        self.user_loop_bars = None       # LOOP 4 / 8 in force
        self._break = None               # (restore_clock, {lane: deck}) while a BREAK runs
        self.tempo_span = 0.0            # tempo journey: 0 = fixed clock
        self.tempo_lean = 0.0            # the tempo dial: a fixed lean on the clock
        self.base_bpm = None             # the opener's tempo, the journey's centre
        self._tempo_k = None
        self.snapshots = []              # saved combinations
        self._recall = None              # a recall in progress
        self.record_path = None
        self._rec_thread = None
        self._rec_stop = False
        self._rec_from = 0
        # learning
        self.move_w = {}                 # (kind, lane) -> 0..1 from stored verdicts (0.5 = no opinion)
        self._tally = {}
        self._load_verdicts()
        # bookkeeping
        self.running = False
        self._lock = threading.Lock()
        self._pending = {}               # deck -> (track, samples, stems, rms) decoded, waiting to mount
        self._decoding = {}              # deck -> track
        self.phrase_bars = 0             # bars of the master since the last move
        self.bar_k = None                # the master's bar index, as last seen (wraps inside a loop)
        self.bar_n = 0                   # bars counted since the start: the monotonic clock every age is measured on
        self.start_clock = None
        self.moves = []                  # (time, text) what the conductor did
        self.move_log = []               # dicts: {t, kind, lane, from_id, to_id, text, fb, shape}
        self.log = []
        self.last_error = None
        self._thread = None

    # -- lifecycle -----------------------------------------------------------------------
    def start(self, first_track=None, threaded=True, cue_s=None, lanes=None, hold_open=False):
        """Open. `first_track` picks the opener (else the conductor does); `cue_s` opens it at that song
        time instead of its body and `lanes` opens only those lanes (the timeline player's exact starts);
        `hold_open` decodes but waits for open_pending() (the Director's handover on a chosen bar)."""
        if self.brain is None:
            self.last_error = "no stem-bearing tracks in the library (run tools/dj/dj_stems.py)"
            return False
        first = first_track or self._pick_first()
        self.running = True
        self.start_clock = self.submix.clock
        if cue_s is not None:
            self._cue_req["a"] = float(max(0.0, cue_s))
        self._open_lanes = set(lanes) if lanes is not None else None
        self._hold_open = bool(hold_open)
        self._open_at = None
        self._decode(first, "a")
        if threaded:
            self._thread = threading.Thread(target=self._run, daemon=True, name="remix")
            self._thread.start()
        return True

    def stop(self, fade_s=1.0):
        self.running = False
        if self._rec_thread is not None:
            self.record_stop()
        now = self.submix.clock + int(0.05 * RATE)
        ev = []
        for d in DECKS:
            ev += [{"at": now, "cmd": "gain", "deck": d, "value": 0.0, "ramp_s": fade_s},
                   {"at": now + int((fade_s + 0.1) * RATE), "cmd": "stop", "deck": d},
                   {"at": now + int((fade_s + 0.1) * RATE), "cmd": "clear_loop", "deck": d}]
        ev.append({"at": now + int((fade_s + 0.1) * RATE), "cmd": "end_sync"})
        self.submix.post_many(ev)

    def _run(self):
        while self.running:
            try:
                self.step()
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
                import traceback
                traceback.print_exc()
            time.sleep(0.2)

    # -- the brain's picks and the arc --------------------------------------------------------
    def arc_progress(self):
        if self.start_clock is None:
            return 0.0
        elapsed = (self.submix.clock - self.start_clock) / RATE
        length = float(getattr(self, "arc_len_s", 0) or 0) or SET_CYCLE_S
        theme = self.brain.theme
        if getattr(theme, "arc", "") == "all_night" and not getattr(self, "arc_waypoints", None):
            return min(1.0, elapsed / max(length, 6 * 3600.0 if length == SET_CYCLE_S else length))
        return (elapsed % length) / length

    def _arc_base(self, progress):
        """The theme's arc, or the Director's drawn / chosen curve (waypoints) when one is set."""
        pts = getattr(self, "arc_waypoints", None)
        if pts:
            xs = [p for p, _ in pts]
            ys = [e for _, e in pts]
            if progress <= xs[0]:
                return ys[0]
            if progress >= xs[-1]:
                return ys[-1]
            for i in range(len(xs) - 1):
                if xs[i] <= progress <= xs[i + 1]:
                    f = (progress - xs[i]) / max(xs[i + 1] - xs[i], 1e-6)
                    return ys[i] + f * (ys[i + 1] - ys[i])
        return self.brain.theme.arc_target(progress) if self.brain else 0.6

    def energy_target(self):
        """The arc at this point of the set, plus the lean."""
        return max(0.05, min(0.95, self._arc_base(self.arc_progress()) + self.energy_lean))

    # -- the arc: the Director's hands ---------------------------------------------------------------------------
    def set_arc_waypoints(self, pts):
        self.arc_waypoints = sorted((max(0.0, min(1.0, float(p))), max(0.0, min(1.0, float(e)))) for p, e in (pts or []))[:16]

    def set_arc_length(self, seconds):
        self.arc_len_s = float(max(1800.0, min(12 * 3600.0, seconds)))

    def set_arc_progress(self, p):
        """'We are HERE on the arc': move the set's start so that progress reads `p` now."""
        if self.start_clock is None:
            return
        length = float(getattr(self, "arc_len_s", 0) or 0) or SET_CYCLE_S
        self.start_clock = int(self.submix.clock - max(0.0, min(0.999, float(p))) * length * RATE)

    def _pick_first(self):
        """The opener: from the pool when one is set, inside the theme's tempo window (a 70 bpm half-time
        read once opened a session nothing could join), a confident grid, room to play."""
        lo, hi = self.brain.theme.bpm_range
        avoid = set(getattr(self, "avoid_ids", ()) or ())
        lib = [t for t in self.library if (self.brain.pool_ids is None or t.id in self.brain.pool_ids) and t.id not in avoid] or self.library
        cands = [t for t in lib if (t.bpm_conf or 0) >= 0.7 and t.duration_s >= 150 and lo <= t.bpm <= hi]
        if not cands:
            cands = [t for t in lib if (t.bpm_conf or 0) >= 0.7 and t.duration_s >= 150]
        return self.rng.choice(cands) if cands else self.rng.choice(lib)

    def set_pool(self, track_ids):
        """Confine the songs to a pool (a saved setlist's tracks); None = the whole stem library. Songs
        already live finish their part; every pick from now on comes from the pool."""
        if track_ids is None:
            self.brain.pool_ids = None
            self.pool_name = None
            return 0
        have = {t.id for t in self.library}
        self.brain.pool_ids = {int(i) for i in track_ids if int(i) in have}
        return len(self.brain.pool_ids)

    def _pick_next(self):
        """The brain's choice against the master song, to the theme and the arc; must run inside the
        tempo wall against the clock, have body ahead of its entry, and not already be live or on its way."""
        ms = self.songs.get(self.master)
        if ms is None:
            return None
        busy = {s.track.id for s in self.songs.values()} | {t.id for t in self._decoding.values()} | {p[0].id for p in self._pending.values()}
        busy |= set(getattr(self, "avoid_ids", ()) or ())        # the Director's played list: no repeats across engines
        saved = set(self.brain.veto_ids)
        try:
            # the cheap gates go FIRST, for the whole library, so the brain scores once: a candidate must be
            # inside the tempo wall, have runway from its entry, and be shiftable (two semitones at most) to
            # fit the room's key at ENTRY_FIT. (Re-running choose_next up to eight times - a full scoring pass
            # each - held the GIL for seconds and starved the audio thread; the user heard the underrun.)
            veto = set(busy)
            for t in self.brain.library:
                if t.id in veto or not t.bpm:
                    continue
                rate = self.master_bpm / max(t.bpm, 1e-6)
                if not (RATE_MIN <= rate <= RATE_MAX):
                    veto.add(t.id)
                    continue
                if (t.duration_s - self._body_start(t)) / rate < MIN_RUNWAY_S:
                    veto.add(t.id)
                    continue
                _shift, fit = self._key_fit(t)
                if fit is not None and fit < ENTRY_FIT:
                    veto.add(t.id)
            self.brain.veto_ids |= veto
            cand, meta = self.brain.choose_next(ms.track, self.energy_target(), self.master_bpm)
            if cand is None or cand.id in veto:
                return None
            return cand
        finally:
            self.brain.veto_ids = saved

    # -- decoding (off the conductor thread) ----------------------------------------------------
    def _decode(self, track, deck):
        if deck in self._decoding or deck in self._pending:
            return
        self._decoding[deck] = track

        def work():
            try:
                # decoded and measured in the helper PROCESS (lib/dj/stemload): five seconds of PyAV and
                # numpy that starved the audio producer when done in this process
                from lib.dj.stemload import decode_song
                samples, stems, rms = decode_song(self.db.abs(track.path), self.music_root, track.id,
                                                  self._body_start(track), track.sections)
                with self._lock:
                    self._pending[deck] = (track, samples, stems, rms)
            except Exception as e:  # noqa: BLE001
                self.last_error = f"decode {track.title}: {type(e).__name__}: {e}"
                self._note(self.last_error)
            finally:
                self._decoding.pop(deck, None)
        threading.Thread(target=work, daemon=True, name=f"remix-decode-{deck}").start()

    def _stem_levels(self, track, stems):
        """Per-stem RMS over up to 120 s of the body (chunked: stems are float16, whole tracks), plus the
        vocals stem's RMS per SECTION - the measured singing map (the ML vocal pass covers only part of the
        library; the stem itself never lies). Returned under the key "_vox_sections"."""
        # one pass per stem (lib/dj/stemload.stem_levels, shared with the helper process): the mean square
        # per half-second hop, from which the body RMS, the envelope and the singing map all derive
        from lib.dj.stemload import stem_levels
        return stem_levels(stems, self._body_start(track), track.sections or [])

    def _mount(self, deck):
        with self._lock:
            item = self._pending.pop(deck, None)
        if item is None:
            return None
        track, samples, stems, rms = item
        self.submix.post_many([
            {"cmd": "unload", "deck": deck},
            {"cmd": "load", "deck": deck, "samples": samples, "grid": track.grid, "track_id": track.id,
             "gain_db": track.gain_db, "stems": stems, "cue_s": 0.0},
            {"cmd": "gain", "deck": deck, "value": 0.0, "ramp_s": 0.01},
            {"cmd": "stem_gains", "deck": deck, "gains": {s: 0.0 for s in STEMS}, "ramp_s": 0.01},
            {"cmd": "eq", "deck": deck, "low": 1.0, "mid": 1.0, "high": 1.0, "ramp_s": 0.01},
            {"cmd": "filter", "deck": deck, "mode": "off"},
            {"cmd": "echo", "deck": deck, "active": False},
        ])
        song = Song(deck, track)
        song.user_pick = track.id in self._user_picks
        rms = dict(rms)
        song.env = rms.pop("_env", None)
        song.vox_sections = rms.pop("_vox_sections", None)
        if song.vox_sections:
            song.vocal_data = True                    # measured from the stem: every song has a singing map
        song.stem_rms = rms
        # the level trim reference: the LIBRARY's median stem levels (remembered per track as they are measured,
        # so it firms up night by night); until eight tracks are known, the first song's (a quiet first song
        # trimmed every later melody to half - flag 'level' 2026-09-13)
        self._remember_stem_rms(track, rms)
        if self.ref_rms is None:
            self.ref_rms = self._library_ref_levels() or dict(rms)
        self.songs[deck] = song
        return song

    def _remember_stem_rms(self, track, rms):
        """Store a track's measured stem levels in its axes (the scanner keeps the key) - the reference grows."""
        vals = {ln: round(float(rms.get(ln, 0.0)), 5) for ln in STEMS}
        try:
            axes = dict(getattr(track, "axes", None) or {})
            if all(abs(float(axes.get("stem_rms", {}).get(ln, -1)) - vals[ln]) < 1e-4 for ln in STEMS):
                return
            axes["stem_rms"] = vals
            track.axes = axes
            row = self.db.conn.execute("SELECT axes FROM tracks WHERE id = ?", (track.id,)).fetchone()
            stored = json.loads(row["axes"]) if row and row["axes"] else {}
            stored["stem_rms"] = vals
            self.db.conn.execute("UPDATE tracks SET axes = ? WHERE id = ?", (json.dumps(stored), track.id))
            self.db.conn.commit()
        except Exception:
            pass

    def _library_ref_levels(self, min_n=8):
        """Per-lane median of the stem levels remembered across the library; None until `min_n` tracks have
        them. The VOCAL reference is the median over songs that SING (a vocal stem at least a fifth of the
        drums' median): over a half-instrumental library the plain median was 0.032 and every real singer
        was trimmed toward a mostly silent lane, up to the x2 cap (flag 'level', 'vocals' 2026-09-13)."""
        per = {ln: [] for ln in STEMS}
        for t in self.library:
            sr = (getattr(t, "axes", None) or {}).get("stem_rms")
            if not isinstance(sr, dict):
                continue
            for ln in STEMS:
                v = sr.get(ln)
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(f) and f > 0.0:
                    per[ln].append(f)
        if any(len(v) < min_n for v in per.values()):
            return None
        ref = {ln: float(np.median(v)) for ln, v in per.items()}
        sung = [v for v in per["vocals"] if v >= 0.2 * ref["drums"]]
        if len(sung) >= min_n:
            ref["vocals"] = float(np.median(sung))
        return ref

    # -- geometry ------------------------------------------------------------------------------------
    def _seg(self, track, t):
        for seg in track.grid or []:
            if seg["start_s"] <= t <= seg["end_s"]:
                return seg
        return (track.grid or [None])[0]

    def _bar_s(self, track, t=0.0):
        seg = self._seg(track, t)
        return 4 * (seg["period_s"] if seg else track.period_s)

    def _body_start(self, track):
        secs = track.sections or []
        for s in secs:
            if s.get("kind") == "groove" and s.get("bass_share", 0.3) >= 0.28:
                return track.nearest_downbeat(s["start_s"])
        for s in secs:
            if s.get("kind") not in ("intro", "outro"):
                return track.nearest_downbeat(s["start_s"])
        return track.nearest_downbeat(0.0)

    def _landmark(self, track):
        """Where a song should be heard first: the groove after its first build (the drop, in this
        library's vocabulary), if it comes within 90 s of the body and leaves runway; else the body."""
        body = self._body_start(track)
        secs = track.sections or []
        for i in range(1, len(secs)):
            if secs[i - 1].get("kind") == "build" and secs[i].get("kind") == "groove" \
                    and body <= secs[i]["start_s"] <= body + 90.0 \
                    and track.duration_s - secs[i]["start_s"] >= MIN_RUNWAY_S:
                return track.nearest_downbeat(secs[i]["start_s"])
        return body

    def _tel(self):
        return self.submix.telemetry or {}

    def _tel_deck(self, deck):
        return (self._tel().get("decks") or {}).get(deck) or {}

    def _clock(self):
        return int(self._tel().get("clock", self.submix.clock))

    def _song_time_at(self, deck, at):
        """The song's own time when the submix clock reads `at`."""
        d = self._tel_deck(deck)
        time_s, rate = float(d.get("time_s", 0.0)), max(float(d.get("rate", 1.0)), 1e-6)
        return time_s + (at - self._clock()) / RATE * rate

    def _section_at_clock(self, deck, at):
        s = self.songs.get(deck)
        if s is None:
            return None
        return s.track.section_at(self._song_time_at(deck, at))

    def _next_bar_clock(self, lead_s=LEAD_S):
        now = self._clock()
        ms = self.songs.get(self.master)
        d = self._tel_deck(self.master) if self.master else {}
        if ms is None or not d.get("playing"):
            return now + int(lead_s * RATE)
        time_s, rate = float(d.get("time_s", 0.0)), max(float(d.get("rate", 1.0)), 1e-6)
        seg = self._seg(ms.track, time_s)
        if seg is None:
            return now + int(lead_s * RATE)
        bar = 4 * seg["period_s"]
        first_down = seg["first_beat_s"] + ms.track.downbeat_offset * seg["period_s"]
        t_min = time_s + lead_s * rate
        k = math.ceil((t_min - first_down) / bar - 1e-6)
        t_next = first_down + k * bar
        lp = d.get("loop")
        if lp and t_next * RATE >= lp[1] - 1:
            t_next = lp[1] / RATE
        return now + int((t_next - time_s) / rate * RATE)

    def _beat_s(self):
        """One beat of the clock in OUTPUT seconds (the master's source beat over its playback rate)."""
        ms = self.songs.get(self.master)
        return self._bar_s(ms.track) / 4.0 / max(ms.rate, 1e-6) if ms else 0.5

    def _bar_clock(self):
        return int(4 * self._beat_s() * RATE)

    def _master_bar_index(self):
        ms = self.songs.get(self.master)
        d = self._tel_deck(self.master) if self.master else {}
        if ms is None or not d.get("playing"):
            return None
        return int(float(d.get("time_s", 0.0)) / self._bar_s(ms.track))

    def _snap_to_section(self, at):
        """A planned move waits up to SNAP_BARS for the master's next section boundary."""
        ms = self.songs.get(self.master)
        if ms is None or ms.loop is not None or self.user_loop_bars:
            return at
        t_at = self._song_time_at(self.master, at)
        bar = self._bar_s(ms.track, t_at)
        for s in ms.track.sections or []:
            b = s["start_s"]
            if t_at - 0.05 * bar < b <= t_at + SNAP_BARS * bar + 0.05 * bar:
                b = ms.track.nearest_downbeat(b)
                return at + int((b - t_at) / max(ms.rate, 1e-6) * RATE)
        return at

    # -- the harmonic guard and the vocal rule ----------------------------------------------------------
    def _key_fit(self, track):
        """(shift, compat) that puts `track` nearest the session key."""
        if not self.key_centre or not track.camelot:
            return 0, None
        best = (0, _compat(self.key_centre, track.camelot))
        for s in KEY_SHIFTS:
            c = _compat(self.key_centre, _shift(track.camelot, s))
            if c > best[1] + 1e-9:
                best = (s, c)
            if best[1] >= 0.9:
                break
        return best

    def _pair_fit(self, a, b, at=None):
        """Harmonic fit 0.3..1 of two live songs as they sound NOW: the Camelot tier of their (shifted) keys,
        refined by the measured chroma of the sections each is in at clock `at` when both have one (a bed in
        its breakdown and a voice at its hook are chords, not keys). None when neither is known."""
        cam = _compat(a.key(), b.key()) if (a.track.camelot and b.track.camelot) else None
        sc = None
        if at is not None and a.track.sec_chroma and b.track.sec_chroma:
            pa = a.track.sec_chroma_at(self._song_time_at(a.deck, at))
            pb = b.track.sec_chroma_at(self._song_time_at(b.deck, at))
            if pa and pb:
                from lib.dj.brain import chroma_key_compat
                sc = chroma_key_compat(pa, pb, semitones=float(b.shift - a.shift))
        if cam is None:
            return sc
        if sc is None:
            return cam
        return 0.45 * cam + 0.55 * sc

    def _tonal_ok(self, deck, lane, at=None):
        """May `lane` cross to the song on `deck` next to the songs on the other tonal lanes?"""
        if lane not in TONAL:
            return True
        song = self.songs.get(deck)
        if song is None:
            return False
        # a song ARRIVING in the room (no tonal lane yet) must fit better than one already heard: the ear
        # forgives a combination it knows, not a new clash
        arriving = not any(self.lanes.get(ln) == deck for ln in TONAL)
        floor = ENTRY_FIT if arriving else CLASH_BELOW
        for other in TONAL:
            if other == lane:
                continue
            od = self.lanes.get(other)
            if od is None or od == deck:
                continue
            o = self.songs.get(od)
            if o is None:
                continue
            fit = self._pair_fit(o, song, at)
            if fit is not None and fit < floor:
                return False
        return True

    def _singing(self, song, t, span_s=None):
        """Is the song singing at its own time `t`? First the vocal stem's own half-second ENVELOPE over the
        next two bars (SING_ON of its loud level = singing, SING_OFF = silent; a quiet verse is still singing -
        the per-section read below called whole verses silent, user 2026-09-12: "large sections of vocal
        rest"); then the section map (RMS per section); then the ML vocalness; None when nothing is known."""
        env = (song.env or {}).get("vocals") if song.env else None
        if env is not None and len(env) > 4:
            span = span_s if span_s is not None else 2 * self._bar_s(song.track, t)
            i0 = max(0, int(t * 2))
            i1 = min(len(env), int((t + span) * 2) + 1)
            if i1 > i0:
                peak = float(np.max(np.asarray(env[i0:i1], dtype=np.float32)))
                if peak >= SING_ON:
                    return True
                if peak < SING_OFF:
                    return False
        secs = song.track.sections or []
        idx = next((i for i, s in enumerate(secs) if s["start_s"] <= t < s["end_s"]), len(secs) - 1 if secs else None)
        if idx is None:
            return None
        vs = song.vox_sections
        if vs and len(vs) == len(secs):
            peak = max(vs)
            if peak <= 10 ** (-45 / 20):
                return False                          # an instrumental: no section sings
            return vs[idx] >= max(0.25 * peak, 10 ** (-45 / 20))
        v = secs[idx].get("vocalness")
        if not song.vocal_data or v is None:
            return None
        return v >= VOCAL_MIN

    def _vocal_ok(self, deck, at):
        """The vocal lane only crosses INTO a song that is singing there (nothing known = allowed)."""
        s = self.songs.get(deck)
        if s is None:
            return False
        sing = self._singing(s, self._song_time_at(deck, at))
        return sing is None or sing

    def _vocal_gone(self, deck, at, bars=None):
        """The song on `deck` has stopped singing for the next `bars` bars (default VOCAL_REST_BARS)."""
        s = self.songs.get(deck)
        if s is None:
            return False
        t = self._song_time_at(deck, at)
        span = (bars if bars is not None else VOCAL_REST_BARS) * self._bar_s(s.track, t)
        return self._singing(s, t, span_s=span) is False

    def _sings_through(self, deck, at, bars):
        """Does the song on `deck` keep singing through the next `bars` bars (from the vocal stem's envelope:
        most of the half-seconds above the silent line)? True / False, None when unmeasured."""
        s = self.songs.get(deck)
        env = (s.env or {}).get("vocals") if (s is not None and s.env) else None
        if env is None or len(env) < 4:
            return None
        t = self._song_time_at(deck, at)
        span = bars * self._bar_s(s.track, t)
        i0, i1 = max(0, int(t * 2)), min(len(env), int((t + span) * 2) + 1)
        if i1 - i0 < 2:
            return None
        seg = np.asarray(env[i0:i1], dtype=np.float32)
        return bool(np.mean(seg >= SING_OFF) >= 0.6 and float(seg.max()) >= SING_ON)

    def _lane_ok(self, deck, lane, at):
        return self._tonal_ok(deck, lane, at) and (lane != "vocals" or self._vocal_ok(deck, at))

    def _takeable(self, lane):
        """May a move take `lane` from its holder? Not when it is the holder's last lane and the holder is
        fresh (a song that entered must be heard for MIN_STAY_PHRASES before it can be pushed out); songs
        looping their last bars or evicted give anything."""
        d = self.lanes.get(lane)
        if d is None or d not in self.songs:
            return True
        s = self.songs[d]
        if s.loop is not None or s.evicted or len(s.held(self.lanes)) > 1:
            return True
        return (self.bar_n - (s.entered_k or 0)) >= MIN_STAY_PHRASES * self.change_bars

    # -- hygiene --------------------------------------------------------------------------------------------
    def _trim(self, deck, lane):
        """The level trim of one stem. A SONG KEEPS ITS OWN BALANCE: the song carrying the drums (the bed -
        or a song alone) has ONE factor for all its stems, its drums brought to the library's drum level,
        so a record through the conductor sounds like the record. A stem sounding against another song's
        stems (a voice's melody or vocal over a foreign bed, a bassline swapped under foreign drums) is
        trimmed per lane toward the library's level for that lane - the two songs' stems meet there.
        (Per-lane trims on a lone song had rebalanced its own mix by up to 6 dB - user 2026-09-13.)"""
        s = self.songs.get(deck)
        if s is None or not self.ref_rms or not s.stem_rms:
            return 1.0

        def k(ln):
            a, b = self.ref_rms.get(ln), s.stem_rms.get(ln)
            if not a or not b:
                return None
            return float(min(TRIM_MAX, max(TRIM_MIN, a / b)))
        if self.lanes.get("drums") == deck:
            f = k("drums")
            if f is None:
                vals = [x for x in (k(ln) for ln in STEMS) if x is not None]
                f = float(np.exp(np.mean(np.log(vals)))) if vals else 1.0
            return f
        return k(lane) or 1.0

    def _lane_gain(self, deck, lane, at=None):
        g = self._trim(deck, lane)
        bed = self.lanes.get("drums")
        if bed is not None and deck != bed:
            g = min(g, 1.0)                           # a stem set against another song is never trimmed UP past unity
        if lane == "other":
            if bed is not None and deck != bed:
                g *= OTHER_VOICE                      # a voice's melody is a texture over the bed, not a second song
            vd = self.lanes.get("vocals")
            if vd is not None and vd != deck and vd in self.songs:
                vs = self.songs[vd]
                # the duck only under a vocal that IS singing (unknown used to duck too: thin rooms got thinner)
                if self._singing(vs, self._song_time_at(vd, at if at is not None else self._clock())) is True:
                    g *= OTHER_DUCK
        b = self.lane_boost.get(lane)
        if b is not None and self.lanes.get(lane) == deck and self.bar_n < b[1]:
            if len(b) > 2 and b[2] == "user":
                g = max(g, 1.0)                       # YOU asked for it loud: never under unity before the boost (a vocal you brought in landed at 0.4)
            g *= b[0]                                 # a tactical "loud" on this lane (the peak guard keeps the bus honest)
        if getattr(self, "_clash_duck", {}).get(lane) == deck:
            g *= CLASH_DUCK                           # its chords clash with the bed here (see _clash_tick)
        return min(g, 1.6)                            # (a quiet stem trimmed up and then boosted reached 1.6 x - flag 'clash' 2026-09-13)

    def _clash_tick(self):
        """Every bar: a VOICE whose chords clash with the bed in the sections both are in now (the pair fit
        under CLASH_BELOW, two bars ahead) is held at CLASH_DUCK until they agree again. The guard judged the
        arrival; the songs moved on into other sections."""
        if getattr(self, "_ctick_k", None) == self.bar_n:
            return
        self._ctick_k = self.bar_n
        if not hasattr(self, "_clash_duck"):
            self._clash_duck = {}
        bed = self._bed_deck()
        if bed is None or bed not in self.songs:
            return
        at = self._next_bar_clock()
        ahead = at + 2 * self._bar_clock()
        bs = self.songs[bed]
        for lane in ("other", "vocals"):
            d = self.lanes.get(lane)
            ducked = self._clash_duck.get(lane)
            if d is None or d == bed or d not in self.songs or self.songs[d].leaving:
                if ducked is not None:
                    self._clash_duck.pop(lane, None)
                continue
            fits = [f for f in (self._pair_fit(bs, self.songs[d], at), self._pair_fit(bs, self.songs[d], ahead)) if f is not None]
            fit = min(fits) if fits else None
            if fit is None:
                continue
            if ducked != d and fit < CLASH_BELOW:
                self._clash_duck[lane] = d
                self.submix.post({"at": at, "cmd": "stem_gains", "deck": d, "gains": {lane: self._lane_gain(d, lane, at)}, "ramp_s": self._beat_s()})
                self._move("duck", lane, self._song_id(d), self._song_id(bed),
                           f"the {lane} of {self.songs[d].track.title[:26]} held down: its chords clash with {bs.track.title[:22]} here (fit {fit:.2f})")
            elif ducked == d and fit >= CLASH_BELOW + 0.05:
                self._clash_duck.pop(lane, None)
                self.submix.post({"at": at, "cmd": "stem_gains", "deck": d, "gains": {lane: self._lane_gain(d, lane, at)}, "ramp_s": 2 * self._beat_s()})
                self._move("unduck", lane, self._song_id(d), self._song_id(bed), f"the {lane} of {self.songs[d].track.title[:26]} back up: the chords agree again (fit {fit:.2f})")

    def _boost_step(self):
        """A lane's "loud" ends: back to its trim over a bar."""
        for lane, (mult, until) in list(self.lane_boost.items()):
            if self.bar_n >= until:
                del self.lane_boost[lane]
                d = self.lanes.get(lane)
                if d is not None and d in self.songs:
                    self.submix.post({"at": self._next_bar_clock(), "cmd": "stem_gains", "deck": d,
                                      "gains": {lane: self._lane_gain(d, lane)}, "ramp_s": 4 * self._beat_s()})

    def _hygiene(self, at):
        """After a move lands: lows only on the songs carrying the bass or the drums lane; every held stem
        re-posted at its level (a song that just became the bed switches from per-lane trims to its own
        balance; the `other` lane under another song's vocal at OTHER_DUCK); posted at `at`, a one-beat ramp."""
        beat = self._beat_s()
        ev = []
        keep = {self.lanes.get("bass"), self.lanes.get("drums")}
        for d, s in self.songs.items():
            if s.leaving or s.staged_at is None:
                continue
            low = 1.0 if d in keep else LOW_CARVE
            ev.append({"at": at, "cmd": "eq", "deck": d, "low": low, "ramp_s": beat})
            held = {ln: self._lane_gain(d, ln, at) for ln in STEMS if self.lanes.get(ln) == d}
            if held:
                ev.append({"at": at, "cmd": "stem_gains", "deck": d, "gains": held, "ramp_s": beat})
        if ev:
            self.submix.post_many(ev)

    # -- shapes ------------------------------------------------------------------------------------------------
    def _shape_exit(self, deck, at):
        """The song on `deck` gives up its last lane at `at`: a low-pass sweep or a one-beat stutter over the
        bar before. Returns the shape's name for the move text, or None when there is no time for one."""
        s = self.songs.get(deck)
        if s is None:
            return None
        now = self._clock()
        t0 = max(now + int(LEAD_S * RATE), at - self._bar_clock())
        dur = (at - t0) / RATE
        beat = self._beat_s()
        if dur < 1.5 * beat:
            return None
        if dur >= 3.5 * beat and s.loop is None and not self.user_loop_bars and self.rng.random() < 0.5:
            ts = self._song_time_at(deck, t0)
            per = self._bar_s(s.track, ts) / 4.0
            start = s.track.nearest_downbeat(ts)
            if start < ts:
                start += per * math.ceil((ts - start) / per - 1e-6)
            self.submix.post_many([{"at": t0, "cmd": "loop", "deck": deck, "start_s": start, "end_s": start + per},
                                   {"at": at, "cmd": "clear_loop", "deck": deck}])
            s.fx_until = at
            return "stutter out"
        self.submix.post_many([{"at": t0, "cmd": "filter", "deck": deck, "mode": "lp", "cutoff_hz": 16000.0, "ramp_s": 0.0, "q": 1.0},
                               {"at": t0 + int(0.02 * RATE), "cmd": "filter", "deck": deck, "cutoff_hz": 260.0, "ramp_s": dur - 0.02},
                               {"at": at + int(2 * beat * RATE), "cmd": "filter", "deck": deck, "mode": "off"}])
        s.fx_until = at + int(2 * beat * RATE)
        return "filter out"

    def _shape_rest(self, deck, at):
        """A lane rests at `at` and its song holds nothing else: an echo throw rings it out."""
        s = self.songs.get(deck)
        if s is None:
            return None
        beat = self._beat_s()
        t0 = max(self._clock() + int(LEAD_S * RATE), at - int(beat * RATE))
        off = at + 2 * self._bar_clock()
        self.submix.post_many([{"at": t0, "cmd": "echo", "deck": deck, "active": True, "delay_s": 0.75 * beat, "feedback": 0.55, "wet": 0.45},
                               {"at": off, "cmd": "echo", "deck": deck, "active": False}])
        s.fx_until = off
        return "echo"

    # -- songs arriving and leaving ---------------------------------------------------------------------
    def _open_first(self, deck):
        song = self._mount(deck)
        if song is None:
            return
        t = song.track
        at = self.submix.clock + int(LEAD_S * RATE)
        if getattr(self, "_open_at", None) is not None:
            at = max(at, int(self._open_at))           # the Director's handover bar
            self._open_at = None
        self.master, self.master_bpm, self.key_centre = deck, float(t.bpm), t.camelot
        self.base_bpm = float(t.bpm)
        song.rate, song.staged_at, song.entered, song.entered_k = 1.0, at, True, 0
        cue = self._cue_req.pop(deck, None)
        open_lanes = set(self._open_lanes) if self._open_lanes is not None else set(STEMS)
        # the vocal lane opens only where the song is singing (it opened on a silent passage, rested two
        # seconds later and came back ten seconds after that - flag 'busy' 2026-09-13)
        if "vocals" in open_lanes and self._singing(song, cue if cue is not None else self._body_start(t)) is False:
            open_lanes.discard("vocals")
        self.lanes = {s: (deck if s in open_lanes else None) for s in STEMS}
        song.awake = True
        self.submix.post_many([
            {"at": at, "cmd": "cue", "deck": deck, "time_s": cue if cue is not None else self._body_start(t)},
            {"at": at, "cmd": "rate", "deck": deck, "value": 1.0},
            # the opener at the library's level, its own balance kept (one factor for every stem: _trim)
            {"at": at, "cmd": "stem_gains", "deck": deck, "gains": {s: (self._lane_gain(deck, s, at) if s in open_lanes else 0.0) for s in STEMS}, "ramp_s": 0.01},
            {"at": at, "cmd": "gain", "deck": deck, "value": 1.0, "ramp_s": 0.05},
            {"at": at, "cmd": "start", "deck": deck},
        ])
        # the bed has been the bed since the SONG started playing, not since this open: a record handed over
        # from the autoDJ two minutes in has settled two minutes already (user: "long chunks when very little
        # is going on" - the first minute and a half of a layered set was the settle rule counting from zero)
        played_bars = int((cue or 0.0) / max(self._bar_s(t, cue or 0.0), 1e-6)) if cue is not None else 0
        # ...but the room still gets two phrases of the bed alone before anything arrives (a voice came 15 s
        # after the handover - flag 'busy' 2026-09-13)
        played_bars = min(played_bars, max(0, (SETTLE_PHRASES - 2) * self.change_bars))
        song.bed_since = self.bar_n - played_bars
        self.brain.note_played(t)
        self._log_play(t)
        self._move("open", None, None, t.id, f"{t.title} opens on {'every lane' if len(open_lanes) == 4 else ', '.join(sorted(open_lanes)) or 'no lane'}: the clock, {t.bpm:.1f} bpm, key {t.camelot or '?'}")

    def _stage_song(self, deck):
        """Mount a decoded song, start it beat-locked with every lane closed, cued so its LANDMARK lands on
        the move that lets it in. It plays silently until then, so the PLL has settled when it is heard."""
        song = self._mount(deck)
        if song is None:
            return False
        t = song.track
        rate = (self.master_bpm or t.bpm) / max(t.bpm, 1e-6)
        if not (RATE_MIN <= rate <= RATE_MAX):
            self._note(f"{t.title} refused at rate {rate:.3f}")
            self.songs.pop(deck, None)
            self.submix.post({"cmd": "unload", "deck": deck})
            return False
        song.rate = rate
        song.shift, song.compat = self._key_fit(t)
        at = self._next_bar_clock()
        song.staged_at = at
        req = self._cue_req.pop(deck, None)
        if req is not None:
            # the timeline player's exact start: song time `song_time` must arrive at timeline bar `at_bar`;
            # the deck starts on the NEXT bar (bar_n + 1) and then advances one song bar per clock bar
            if isinstance(req, tuple):
                song_time, at_bar = req
                bar_song = self._bar_s(t, song_time)
                cue = song_time - (at_bar - self.bar_n - 1) * bar_song
                if cue < 0.0:
                    cue = song_time - bar_song * math.floor(song_time / bar_song)
            else:
                cue = float(req)
            song.ready_k = self.bar_n
            landmark, to_move = cue, 1
        else:
            # bars until the move that can let it in (moves are planned a bar early; the PLL wants two bars)
            to_move = self.change_bars - self.phrase_bars
            if to_move < 3:
                to_move += self.change_bars
            song.ready_k = self.bar_n + to_move - 1                  # the move that may let it in, in bars
            landmark = self._landmark(t)
            bar_song = self._bar_s(t, landmark)
            cue = landmark - (to_move - 1) * bar_song              # it starts on the NEXT bar
            if cue < 0.0:
                cue = landmark - bar_song * math.floor(landmark / bar_song)      # the same phase, as early as it goes
        self.submix.post_many([
            {"at": at, "cmd": "cue", "deck": deck, "time_s": t.nearest_downbeat(cue)},
            {"at": at, "cmd": "rate", "deck": deck, "value": rate},
            {"at": at, "cmd": "pitch", "deck": deck, "semitones": float(song.shift)},
            {"at": at, "cmd": "eq", "deck": deck, "low": LOW_CARVE, "mid": 1.0, "high": 1.0, "ramp_s": 0.01},
            # the deck sleeps at gain 0 until its first lane opens (_wake): the PLL re-snaps only a near-silent
            # deck, and a staged song at gain 1 with every stem closed could never snap after a cue - it slewed
            # 0.3 beat away for fifteen seconds and arrived inside that (the gate, 2026-09-13)
            {"at": at, "cmd": "gain", "deck": deck, "value": 0.0, "ramp_s": 0.01},
            {"at": at, "cmd": "start", "deck": deck},
            {"at": at, "cmd": "sync", "slave": deck, "master": self.master, "bias_beats": 0.0, "audio_pll": False},
        ])
        self.brain.note_played(t)
        self._log_play(t)
        fit = f", key fit {song.compat:.2f}" if song.compat is not None else ""
        self._note(f"{t.title} staged on {deck.upper()} (rate {rate:.3f}, shift {song.shift:+d} st{fit}; landmark {landmark:.0f}s in {to_move} bars)")
        return True

    def _leave_song(self, deck, at):
        song = self.songs.get(deck)
        if song is None or song.leaving:
            return
        beat = self._beat_s()
        # an echo or a sweep still running rings out before the deck stops
        tail = max(2 * beat * RATE, (song.fx_until - at) if song.fx_until else 0)
        stop_at = at + int(tail)
        ev = [{"at": at, "cmd": "gain", "deck": deck, "value": 0.0, "ramp_s": 1.5 * beat},
              {"at": stop_at, "cmd": "stop", "deck": deck}, {"at": stop_at, "cmd": "clear_loop", "deck": deck},
              {"at": stop_at, "cmd": "echo", "deck": deck, "active": False}, {"at": stop_at, "cmd": "filter", "deck": deck, "mode": "off"}]
        if deck == self.master:
            others = [d for d in self.songs if d != deck and not self.songs[d].leaving]
            if others:
                new = max(others, key=lambda d: (len(self.songs[d].held(self.lanes)), -(self.songs[d].entered_k or 0)))
                ev.append({"at": at, "cmd": "end_sync", "slave": new})
                for d in others:
                    if d != new:
                        ev.append({"at": at, "cmd": "sync", "slave": d, "master": new, "bias_beats": 0.0, "audio_pll": False})
                self.master = new
                self.master_bpm = float(self.songs[new].track.bpm) * self.songs[new].rate
                self.bar_k = None
                self._move("clock", None, None, self.songs[new].track.id, f"{self.songs[new].track.title} takes the clock")
            else:
                self.master = None
        else:
            ev.append({"at": stop_at, "cmd": "end_sync", "slave": deck})
        self.submix.post_many(ev)
        song.leaving, song.leave_clock = True, stop_at + int(0.5 * RATE)
        self._move("leave", None, song.track.id, None, f"{song.track.title} leaves")

    def _wake(self, deck, at):
        """A sleeping (staged, gain 0) deck comes up to its level a beat before its first lane opens."""
        s = self.songs.get(deck)
        if s is None or getattr(s, "awake", False):
            return
        s.awake = True
        # posted AT the bar (not a beat before), after anything already queued for it: a stem brought in the
        # instant its song was staged had its wake applied before the staging's gain 0 - the lane map said
        # "vocals in" and nothing sounded (2026-09-13)
        self.submix.post({"at": max(self._clock(), at, int(s.staged_at or 0)), "cmd": "gain", "deck": deck, "value": float(s.level), "ramp_s": 0.05})

    # SYNC IN THE ARRANGEMENT IS GRID-ONLY. The audio PLL aligns KICKS; in the arrangement only the clock
    # song carries drums, so every slave's sounding stems are melody, vocals or a bassline - against those
    # the kick xcorr measures pattern offset and nudged decks 0.3 beat off for twenty seconds after every
    # clock pass (the gate, 2026-09-13). The grid holds them; the master carries the beat.
    def _pass_clock(self, new, at):
        """The clock belongs to the song carrying the DRUMS: when the drums move to `new`, it becomes the
        master at `at` (every other live deck re-locks to it) instead of waiting until the old master leaves
        - a master without drums starved the audio PLL of its reference and the new bed drifted 0.3 beat for
        twenty seconds (the gate, 2026-09-13)."""
        if new is None or new == self.master or new not in self.songs or self.songs[new].leaving:
            return
        ev = [{"at": at, "cmd": "end_sync", "slave": new}]
        for d, s in self.songs.items():
            if d != new and not s.leaving and s.staged_at is not None:
                ev.append({"at": at, "cmd": "sync", "slave": d, "master": new, "bias_beats": 0.0, "audio_pll": False})
        self.submix.post_many(ev)
        self.master = new
        self.master_bpm = float(self.songs[new].track.bpm) * self.songs[new].rate
        self.bar_k = None
        self._move("clock", None, None, self.songs[new].track.id, f"{self.songs[new].track.title} takes the clock (it carries the drums)")

    def _cross(self, lane, to_deck, at, beats=None, shape=True):
        """Lane `lane` goes to `to_deck` (None = rest) at clock `at`, crossing over `beats` (default: the
        mixing dial's cross_beats). When the song giving the lane holds nothing else, the move gets a
        shape. Returns the shape's name."""
        if beats is None:
            beats = self.cross_beats
        xf = beats * self._beat_s()
        frm = self.lanes.get(lane)
        ev, tag = [], None
        if frm is not None and frm in self.songs and frm != to_deck:
            if shape and len(self.songs[frm].held(self.lanes)) == 1 and not self.songs[frm].leaving \
                    and self.rng.random() < getattr(self, "fx_level", 1.0):
                tag = self._shape_rest(frm, at) if to_deck is None else self._shape_exit(frm, at)
            ev.append({"at": at, "cmd": "stem_gains", "deck": frm, "gains": {lane: 0.0}, "ramp_s": xf})
        self.lanes[lane] = to_deck                     # the map first: the level of a stem depends on who holds the drums now
        self.lane_k[lane] = self.bar_n
        if to_deck is not None and to_deck in self.songs:
            self._wake(to_deck, at)
            ev.append({"at": at, "cmd": "stem_gains", "deck": to_deck, "gains": {lane: self._lane_gain(to_deck, lane, at)}, "ramp_s": xf})
            s = self.songs[to_deck]
            if not s.entered:
                s.entered, s.entered_k = True, self.bar_n
        if ev:
            self.submix.post_many(ev)
        return tag

    def _song_id(self, deck):
        s = self.songs.get(deck) if deck else None
        return s.track.id if s else None

    # -- the choreography ------------------------------------------------------------------------------------
    def step(self):
        if not self.running:
            return
        if self.master is None:
            if "a" in self._pending and not getattr(self, "_hold_open", False):
                self._open_first("a")
            return
        # songs that finished leaving free their decks
        clock = self._clock()
        for d, s in list(self.songs.items()):
            if s.leaving and s.leave_clock is not None and clock >= s.leave_clock:
                self.songs.pop(d, None)
        # a BREAK ends: the lanes are back where they were - or, when that song has left meanwhile, on the
        # song that holds the drums now (a lane must never come back to nobody)
        if self._break is not None and clock >= self._break[0]:
            fallback = self.lanes.get("drums") or self.master
            at_back = clock + int(LEAD_S * RATE)
            for lane, d in self._break[1].items():
                if d in self.songs and not self.songs[d].leaving:
                    self.lanes[lane] = d
                elif fallback in self.songs and not self.songs[fallback].leaving and self._lane_ok(fallback, lane, at_back):
                    self._cross(lane, fallback, at_back, beats=1.0, shape=False)
            self._break = None
            self._hygiene(clock + int(LEAD_S * RATE))
        # decoded songs come in silently at once; the brain's next pick decodes onto a free deck
        for d in DECKS:
            if d in self._pending and d not in self.songs:
                self._stage_song(d)
        live = [s for s in self.songs.values() if not s.leaving]
        cap = 2 if self.blend < 0.5 else 3
        free = [d for d in DECKS if d not in self.songs and d not in self._decoding and d not in self._pending]
        # ONE ON DECK: with autopilot on there is always a next song staged and waiting (decoded, locked,
        # silent), so a NEXT or a dial turn is heard on the next bar, not after a decode. The cap governs
        # how many are HEARD at once (the entry in _one_move), not how many are ready.
        staged_waiting = [s for s in live if not s.entered] + list(self._decoding.values()) + list(self._pending.values())
        if self.auto > 0.0 and free and not staged_waiting and not self.reserve_pick:
            nxt = self._pick_next()
            if nxt is not None:
                self._decode(nxt, free[0])
        # the phrase clock: the master's bars
        k = self._master_bar_index()
        if k is None:
            return
        if self.bar_k is None:
            self.bar_k = k
        elif k != self.bar_k:
            n = (k - self.bar_k) if k > self.bar_k else 1          # a loop wraps the index: one bar
            self.phrase_bars += n
            self.bar_n += n
            self.bar_k = k
        # a bed change that was in flight has landed: the arrangement is free again, a deferred LOOP comes now;
        # a bass-first change still owes its drums - that move comes NOW, not at the next phrase
        if self._land_k is not None and self.bar_n >= self._land_k:
            if self._drums_follow is not None:
                self._force_move = True               # _arrangement_move's landing block crosses the drums
            else:
                self._land_k = None
                b = getattr(self, "_loop_after_land", None)
                if b:
                    self._loop_after_land = None
                    self.loop(b)
        self._keep_runway(self.bar_n)
        self._tempo_step()
        self._boost_step()
        if self._moment_step():
            return                                    # a MOMENT is armed or running: nothing else moves
        if self._break is not None:
            return
        self._recall_step()
        if self._recall is not None:
            return                                    # a recall in progress owns the lanes
        if getattr(self, "policy", "arrangement") == "arrangement" and not self.hold:
            self._vocal_tick()
            try:
                self._clash_tick()
            except Exception as e:  # noqa: BLE001
                self.last_error = f"clash watch: {type(e).__name__}: {e}"
            if getattr(self, "_plan_k", None) != self.bar_n:
                self._plan_k = self.bar_n             # the plan is decided as soon as there is something to decide, once a bar
                try:
                    self._plan_check(self._next_bar_clock())
                except Exception as e:  # noqa: BLE001
                    self.last_error = f"plan: {type(e).__name__}: {e}"
        self._auto_break()
        # a move is PLANNED a bar early (shapes need the bar) and lands on the phrase boundary - or on the
        # PLAN's own bar (an arrival or a fade-out decided for a bar inside the phrase is made due there)
        due = self.phrase_bars >= self.change_bars - 1
        p = self.plan
        if p is not None and p["what"] not in ("bed_to", "land") and self.bar_n >= p["bar"] - 1 and not self.hold and self.auto > 0.0:
            due = True                                # ("land" is the picture of a change in flight, not a move to make)
        if self._force_move:
            self._force_move = False
            self.phrase_bars = 0
            self.plan = None                          # NEXT: whatever was planned gives way to a move now
            self._forced = True                       # NEXT: the arrangement's waits are waived for this move
            try:
                self._one_move(self._next_bar_clock())
            finally:
                self._forced = False
        elif due and not self.hold:
            self.phrase_bars = 0
            # the autopilot amount: the chance this phrase's move is the conductor's (0 = never: the
            # operator's grid is the only thing that moves; 1 = every phrase, the autonomous mode)
            if self.auto >= 1.0 or self.rng.random() < self.auto:
                self._one_move(self._snap_to_section(self._next_bar_clock()))

    def _auto_break(self):
        """The song holding most lanes reaches a breakdown: the conductor may break with it, once per song -
        never while loops hold the room, within two phrases of a DROP (peaks are spaced) or while a bed
        change is in flight."""
        if self._break is not None or self.hold or self.user_loop_bars or self._land_k is not None:
            return
        if getattr(self, "_drop_k", None) is not None and self.bar_n - self._drop_k < 4 * self.change_bars:
            return                                    # a DROP is a peak: four phrases of plain play before another moment
        counts = {}
        for d in self.lanes.values():
            if d is not None:
                counts[d] = counts.get(d, 0) + 1
        if not counts:
            return
        big = max(counts, key=counts.get)
        s = self.songs.get(big)
        if s is None or s.auto_broke or counts[big] < 2:
            return
        sec = self._section_at_clock(big, self._clock()) or {}
        if sec.get("kind") != "breakdown":
            return
        s.auto_broke = True
        if self.rng.random() < AUTO_BREAK_CHANCE * 2 * self._w("break_auto") * getattr(self, "auto_break_scale", 1.0):
            self.break_(bars=BREAK_BARS, kind="break_auto")

    def _keep_runway(self, k):
        """A song whose body is running out loops its last bars; one that has looped for two phrases
        gives its lanes away; one with no lanes leaves a bar later."""
        for deck, song in list(self.songs.items()):
            if song.leaving:
                continue
            d = self._tel_deck(deck)
            if not d.get("playing"):
                continue
            held = song.held(self.lanes)
            if not held:
                if not song.entered or self._break is not None or (self._land_k is not None and k < self._land_k):
                    continue                          # staged and waiting, resting through a BREAK, or the old bed still sounding until the new one lands
                if getattr(song, "user_k", None) is not None and k - song.user_k < 4 * self.change_bars:
                    continue                          # a song YOU brought in stays on its deck four phrases even while its lane rests (one left 2 s after its vocal paused)
                if song.idle_k is None:
                    song.idle_k = k
                elif k - song.idle_k >= 1:
                    self._leave_song(deck, self._next_bar_clock())
                continue
            song.idle_k = None
            time_s, rate = float(d.get("time_s", 0.0)), max(float(d.get("rate", 1.0)), 1e-6)
            if song.loop is None and not d.get("loop"):
                left = (song.track.duration_s - time_s) / rate
                sec = song.track.section_at(time_s) or {}
                # an "outro" label with minutes of body left is a mis-segmentation, not an ending (the gate
                # caught a voice looping "its last 8 bars" with 217 s to go - the loop wrap threw its lock)
                if left < RUNWAY_S or (sec.get("kind") == "outro" and left < MIN_RUNWAY_S):
                    bar = self._bar_s(song.track, time_s)
                    start = song.track.nearest_downbeat(max(0.0, time_s - LOOP_BARS * bar))
                    self.submix.post({"at": self._next_bar_clock(), "cmd": "loop", "deck": deck,
                                      "start_s": start, "end_s": start + LOOP_BARS * bar})
                    song.loop, song.loop_k = (start, start + LOOP_BARS * bar), k
                    self._move("loop", None, song.track.id, None, f"{song.track.title} loops its last {LOOP_BARS} bars ({left:.0f} s of body left)")
            elif song.loop_k is not None and not song.evicted and k - song.loop_k >= EXPIRE_PHRASES * self.change_bars:
                song.evicted = True
                self._evict(deck)

    def _evict(self, deck):
        """Every lane the song holds crosses to another live song (or rests), one bar apart."""
        song = self.songs[deck]
        others = [d for d, s in self.songs.items() if d != deck and not s.leaving and s.staged_at is not None]
        at = self._next_bar_clock()
        bar = self._bar_clock()
        for i, lane in enumerate([ln for ln in MORPH_ORDER if self.lanes.get(ln) == deck]):
            tgt = [d for d in others if self._lane_ok(d, lane, at + i * bar)]
            to = self.rng.choice(tgt) if tgt else None
            self._cross(lane, to, at + i * bar)
            self._hygiene(at + i * bar + int(XFADE_BEATS * self._beat_s() * RATE))
        self._move("evict", None, song.track.id, None, f"{song.track.title} has looped long enough: its lanes move on")

    def _w(self, kind, lane=None):
        """Learned weight of a kind of move (0.5 = no verdicts yet)."""
        return self.move_w.get((kind, lane), self.move_w.get((kind, None), 0.5))

    def _one_move(self, at):
        tag = None
        self.wait_why = None
        n0 = len(self.move_log)
        try:
            tag = self._one_move_inner(at)
        finally:
            # hygiene (the low carve, the duck) reads the lane map, which already shows a bed change that
            # LANDS later: it waits for the landing rather than thinning the bed that still plays - and it
            # runs only when something MOVED (a phrase that passed without a move queued a hygiene pass at
            # a future landing with the pre-landing lane map, which reopened lows just carved)
            moved = any(m["kind"] != "plan" for m in self.move_log[n0:])
            t_h = at
            if self._land_at is not None and self._land_at > at:
                t_h = self._land_at
            self._land_at = None
            if moved:
                self._hygiene(t_h + int(XFADE_BEATS * self._beat_s() * RATE))
        return tag

    # -- the arrangement's reading of a song's structure --------------------------------------------------------
    def _kind_at(self, deck, at):
        sec = self._section_at_clock(deck, at) or {}
        return sec.get("kind")

    def _stem_level_span(self, deck, lane, at0, at1):
        """The song's own stem envelope (0..1 of its 95th percentile, half-second steps) - its MINIMUM over the
        span of clocks at0..at1, so a stem that drops out mid-span reads as quiet; None when unknown."""
        s = self.songs.get(deck)
        env = (s.env or {}).get(lane) if s is not None else None
        if env is None or len(env) == 0:
            return None
        t0, t1 = self._song_time_at(deck, at0), self._song_time_at(deck, at1)
        i0, i1 = int(max(0.0, min(t0, t1)) * 2), int(max(t0, t1) * 2) + 1
        i0, i1 = min(i0, len(env) - 1), min(max(i1, i0 + 1), len(env))
        return float(np.percentile(np.asarray(env[i0:i1], dtype=np.float32), 25))

    def _drop_in_bars(self, deck, at, look_bars, min_groove_bars=0):
        """Bars of the song on `deck` from clock `at` to its next GROOVE that follows a build, a breakdown or
        the intro (its drop, in this library's vocabulary) - 0 when the drop is right there, None when none
        comes within `look_bars` bars. `min_groove_bars`: the groove after the drop must last this long (a
        bed change landed on a drop whose groove ended 30 s later and the room fell into the new bed's build -
        flags 'dead air' and 'wrong song' 2026-09-13)."""
        s = self.songs.get(deck)
        if s is None:
            return None
        t_at = self._song_time_at(deck, at)
        secs = s.track.sections or []
        bar = self._bar_s(s.track, t_at)
        for i in range(1, len(secs)):
            if secs[i].get("kind") == "groove" and secs[i - 1].get("kind") in ("build", "breakdown", "intro"):
                if min_groove_bars and (float(secs[i]["end_s"]) - float(secs[i]["start_s"])) < min_groove_bars * bar - 0.05:
                    continue
                b = s.track.nearest_downbeat(secs[i]["start_s"])
                d = (b - t_at) / max(bar, 1e-6)
                if -DROP_LOOK_BARS <= d <= look_bars + 0.05:
                    return max(0.0, d)
        return None

    def _odd_grid_bars(self, deck, at):
        """Bars from clock `at` until the song on `deck` is OUT of a mis-tracked grid segment (its tempo off
        the song's by more than 3 %: a half-time breakdown read at 66 bpm) - 0 when it is on the song's grid
        now. A bed change or an arrival is not landed inside one: the beat phase there is meaningless."""
        s = self.songs.get(deck)
        g = s.track.grid if s is not None else None
        if not isinstance(g, list) or len(g) < 2:
            return 0
        t = self._song_time_at(deck, at)
        dom = max(g, key=lambda x: float(x.get("end_s", 0.0)) - float(x.get("start_s", 0.0)))
        p0 = float(dom.get("period_s") or s.track.period_s)
        for x in g:
            if float(x.get("start_s", 0.0)) <= t < float(x.get("end_s", 0.0)):
                if abs(float(x.get("period_s") or p0) / max(p0, 1e-6) - 1.0) > 0.03:
                    return int(math.ceil((float(x["end_s"]) - t) / max(4 * p0, 1e-6))) + 1
                return 0
        return 0

    def _groove_through(self, deck, at, bars, strict=False):
        """Is the song on `deck` carrying steadily at clock `at` and still `bars` bars later - not in a build,
        a breakdown or an outro (`strict`: a groove proper, no intro either - what an ARRIVING song must be
        in)? None when it has no section map. A bed change or an arrival lands only where both songs carry
        through the coming phrase (five flags on 2026-09-13 were a bassline taking over as its song entered
        its breakdown, an arrival six seconds before the bed's build, and two builds at once). A bed's long
        "intro" is what is playing and counts as steady - refusing it moved a plan five times in a row."""
        s = self.songs.get(deck)
        if s is None or not s.track.sections:
            return None
        bad = ("build", "breakdown", "outro") + (("intro",) if strict else ())
        t0 = self._song_time_at(deck, at)
        t1 = t0 + bars * self._bar_s(s.track, t0)
        for t in (t0, t0 + 0.5 * (t1 - t0), t1):
            sec = s.track.section_at(t) or {}
            if sec.get("kind") in bad:
                return False
        return True

    def _groove_start_bars(self, deck, at, look_bars=None, strict=False):
        """Bars from clock `at` to the start of the song's next steady stretch (a groove proper when `strict`)
        that lasts at least two phrases; 0 when it carries now; None when none comes before the body ends.
        `look_bars` caps the search; unbounded by default, so a plan is moved ONCE to the real bar."""
        s = self.songs.get(deck)
        if s is None or not s.track.sections:
            return 0
        t = self._song_time_at(deck, at)
        bar = self._bar_s(s.track, t)
        if self._groove_through(deck, at, 2 * self.change_bars, strict=strict):
            return 0
        ok_kinds = ("groove",) if strict else ("groove", "intro")
        for x in s.track.sections:
            if x.get("kind") in ok_kinds and float(x["end_s"]) > t and (float(x["end_s"]) - max(float(x["start_s"]), t)) >= 2 * self.change_bars * bar - 0.05:
                d = max(0.0, (s.track.nearest_downbeat(float(x["start_s"])) - t) / max(bar, 1e-6))
                if look_bars is None or d <= look_bars + 0.05:
                    return int(round(d))
                break
        return None

    def _body_left_s(self, deck, at=None):
        s = self.songs.get(deck)
        if s is None:
            return 0.0
        t = self._song_time_at(deck, at if at is not None else self._clock())
        return max(0.0, float(s.track.duration_s) - t) / max(float(s.rate), 1e-6)

    def _on_way_out(self, deck, at=None):
        """A song that must not take the bed or receive a lane: looping its last bars, evicted, leaving, or
        with less body left than a bed change needs (a bass-first change into a song looping its last eight
        bars tangled the room for a minute - flags 'seam', 'clash', 'timing' 2026-09-13)."""
        s = self.songs.get(deck)
        if s is None or s.leaving or s.evicted or s.loop is not None:
            return True
        return self._body_left_s(deck, at) < MIN_RUNWAY_S

    def _bed_dies_in_bars(self, bed, at):
        """Bars from the bar at `at` to the point where the bed song stops carrying: its OUTRO, or the end
        of its body less the runway a change needs; None when nothing comes. A breakdown is not a death - it
        is a passage the bed rides through (counting it as one chained a bed change every phrase in the gate:
        every song has a breakdown coming)."""
        s = self.songs.get(bed)
        if s is None:
            return None
        t = self._song_time_at(bed, at)
        bar = self._bar_s(s.track, t)
        cands = []
        for x in s.track.sections or []:
            if x.get("kind") == "outro" and float(x["start_s"]) > t + 0.5 * bar:
                cands.append((s.track.nearest_downbeat(float(x["start_s"])) - t) / max(bar, 1e-6))
                break
        end = (float(s.track.duration_s) - MIN_RUNWAY_S - t) / max(bar, 1e-6)
        if end > 0:
            cands.append(end)
        return int(round(min(cands))) if cands else None

    def _hook_in_bars(self, deck, at, look_bars):
        """Bars of the song on `deck` until its next HOOK occurrence starts (0 while inside one that has just
        begun), or None without a measured hook or none within `look_bars`."""
        s = self.songs.get(deck)
        h = s.track.hook if s is not None else None
        if not h:
            return None
        t_at = self._song_time_at(deck, at)
        bar = self._bar_s(s.track, t_at)
        for st in (h.get("starts") or [h["start_s"]]):
            d = (st - t_at) / max(bar, 1e-6)
            if -DROP_LOOK_BARS <= d <= look_bars + 0.05:
                return max(0.0, d)
        return None

    def _payoff_in_bars(self, deck, at, look_bars, min_groove_bars=0):
        """(bars, what) to the song's nearest payoff within the window: its drop or its hook; (None, None)."""
        d = self._drop_in_bars(deck, at, look_bars, min_groove_bars=min_groove_bars)
        h = self._hook_in_bars(deck, at, look_bars)
        if h is not None and (d is None or h <= d):
            return h, "hook"
        return d, ("drop" if d is not None else None)

    def _bed_to(self, d, at, d_bars, what="drop", bass_first=None):
        """The BED (drums + bass together) passes to the voice on `d`. With `strip_before_bed`, the old
        bed's drums and bass drop out STRIP_BARS before the landing - a breakdown of the room - and the new
        drums and bass land as a cut on the bar (the release); else the two lanes cross on the bar. When
        `d_bars` is given the landing is put on the voice's own drop. `bass_first` forces (or forbids) the
        bassline-first path regardless of the MOMENTS dial - an ARRIVAL through the bass is one."""
        s, old = self.songs[d], self._bed_deck()
        bar_c = self._bar_clock()
        land = at + int(round(d_bars)) * bar_c if d_bars else at
        strip = bool(self.strip_before_bed) and old is not None and old in self.songs and old != d
        arriving = not s.entered
        # the text says what is true: a payoff only when one was found (a change "at its drop" that landed on
        # a phrase boundary read as a wrong drop)
        why = f"at its {what}" if (d_bars is not None and what in ("drop", "hook")) else ("as it arrives" if arriving else "at the phrase, after its turn as a voice")
        tag = None
        use_bass_first = (self.bass_moments if bass_first is None else bool(bass_first))
        # THE BASS MOMENT (MOMENTS on, no strip): the bassline takes over first - at the payoff, loud for a
        # phrase, the old bed's bass gone - and the drums follow at the next phrase. The classic bass swap:
        # the room feels the new song's low end before it knows the song has changed.
        if (not strip or arriving) and use_bass_first and old is not None and old in self.songs and old != d:
            self.lane_boost["bass"] = (1.25, self.bar_n + 2 * self.change_bars)
            tag = self._cross("bass", d, land, beats=1.0, shape=False)
            bars_to_land = int(round((land - at) / max(bar_c, 1))) + 1
            self._land_k = self.bar_n + bars_to_land + self.change_bars
            self._land_at = land
            self._drums_follow = d
            s.bed_since = self.bar_n + bars_to_land
            self._move("voice_in" if arriving else "bed_to", "bass", self._song_id(old), s.track.id,
                       f"the bassline of {s.track.title} takes over {why}, loud; its drums follow at the phrase", shape=tag)
            return tag
        if strip:
            strip_at = land - STRIP_BARS * bar_c
            if strip_at < at:
                strip_at, land = at, at + STRIP_BARS * bar_c
            # the breakdown must be HEARD: something the voice plays through the strip (its lane's stem over
            # the span), else the bed crosses plainly - a silent breakdown is dead air, not tension
            voice_lanes = [ln for ln in ("other", "vocals") if self.lanes.get(ln) == d]
            lv = max([self._stem_level_span(d, ln, strip_at, land) or 0.0 for ln in voice_lanes] or [0.0])
            if lv < 0.3:                              # a third of the stem's own loud level, through the whole strip
                strip = False
                land = at + int(round(d_bars)) * bar_c if d_bars else at
                self._note(f"no breakdown before {s.track.title}'s bed: its voice would be too quiet ({lv:.2f})")
        if strip:
            beats = DROP_BEATS if self.cross_beats <= 1.0 else 1.0
            ev = [{"at": strip_at, "cmd": "stem_gains", "deck": old, "gains": {"drums": 0.0, "bass": 0.0}, "ramp_s": 1.0 * self._beat_s()}]
            for ln in ("drums", "bass"):
                ev.append({"at": land, "cmd": "stem_gains", "deck": d, "gains": {ln: self._lane_gain(d, ln, land)}, "ramp_s": beats * self._beat_s()})
            self.submix.post_many(ev)
            for ln in ("drums", "bass"):
                self.lanes[ln] = d
                self.lane_k[ln] = self.bar_n
            text = (f"the bed passes to {s.track.title} {why}: {self.songs[old].track.title}'s drums and bass drop out for "
                    f"{STRIP_BARS} bars, then {s.track.title}'s land")
        else:
            tag = self._cross("drums", d, land)
            self._cross("bass", d, land)
            if old is not None and old in self.songs:
                self.submix.post({"at": land, "cmd": "eq", "deck": old, "low": LOW_CARVE, "ramp_s": self._beat_s()})
            text = f"the bed passes to {s.track.title} {why}: its drums and bass under the voice"
        # bars until the landing, counted from NOW: the move is planned a bar before `at`
        bars_to_land = int(round((land - at) / max(bar_c, 1))) + 1
        self._land_k = self.bar_n + bars_to_land if bars_to_land > 1 else None
        self._land_at = land if bars_to_land > 1 else None
        s.bed_since = self.bar_n + bars_to_land
        s.was_bed = False
        if old is not None and old in self.songs:
            self.songs[old].was_bed = True
            self.songs[old].voice_since = self.bar_n + bars_to_land
            self.songs[old].bed_since = None
        self._move("bed_to", "drums", self._song_id(old), s.track.id, text, shape=tag)
        return tag

    # -- the ARRANGEMENT policy: a bed and voices -----------------------------------------------------------------
    # Drums and bass belong together: the BED. Other and vocals are VOICES. One thing changes per move, and
    # every song makes the same journey: it arrives as a voice over the current bed; when it has been heard
    # for VOICE_PHRASES it takes the bed (drums and bass together, under its voice - the classic morph); the
    # old bed song is now a voice and leaves after a phrase. Layers "three" allows a second voice.
    def _bed_deck(self):
        return self.lanes.get("drums")

    def _arrangement_move(self, at):
        live = [d for d, s in self.songs.items() if not s.leaving and s.staged_at is not None]
        entered = [d for d in live if self.songs[d].entered]
        bed = self._bed_deck()
        voices = {d for ln in ("other", "vocals") for d in [self.lanes.get(ln)] if d is not None and d != bed}
        max_voices = 2 if self.blend >= 0.5 else 1
        bar = self.bar_n
        # a bed change in flight (the strip, then the landing; or the bassline in and the drums to follow)
        # owns the arrangement until it lands - and never for longer than four phrases (a stuck landing
        # froze the room once)
        if self._land_k is not None:
            if bar < self._land_k and self._land_k - bar <= 4 * self.change_bars:
                self.wait_why = ("the bassline has taken over: the drums follow at the phrase" if self._drums_follow
                                 else "the bed is changing: the breakdown, then the new drums and bass land")
                return None
            self._land_k = None
            d = self._drums_follow
            self._drums_follow = None
            if d is not None and d in self.songs and self._on_way_out(d, at):
                # the song whose bassline took over is running out (looping its last bars, evicted): the change
                # is abandoned - its bass goes back to the drums' song, nothing follows
                old = self.lanes.get("drums")
                if old is not None and old in self.songs and not self.songs[old].leaving and self.lanes.get("bass") == d:
                    tag = self._cross("bass", old, at, beats=1.0, shape=False)
                    self._move("return", "bass", self._song_id(d), self._song_id(old),
                               f"the bass back on {self.songs[old].track.title}: {self.songs[d].track.title[:24]} is running out, its drums never follow", shape=tag)
                    return tag
                d = None
            if d is not None and d in self.songs and not self.songs[d].leaving:
                old = self.lanes.get("drums")
                tag = self._cross("drums", d, at)
                if old is not None and old in self.songs and old != d:
                    self.songs[old].was_bed, self.songs[old].voice_since, self.songs[old].bed_since = True, bar, None
                    # its lows go with its last low lane - now, not at the hygiene pass two beats later
                    self.submix.post({"at": at, "cmd": "eq", "deck": old, "low": LOW_CARVE, "ramp_s": self._beat_s()})
                # (the clock stays with the old song until it leaves - _leave_song passes it; passing it here
                #  re-synced the audible old bed with a snap and it sat 0.3 beat off for twenty seconds)
                self._move("bed_to", "drums", self._song_id(old), self._song_id(d), f"the drums of {self.songs[d].track.title} follow: the bed is its", shape=tag)
                return tag
        # THE PLAN: the one next event, decided once (calm and the settle baked into its bar) and kept until it
        # is done or a rule moves it (logged) - the countdown the operator sees counts to THIS bar
        forced = getattr(self, "_forced", False)
        if not forced:
            self._plan_check(at)
        p = self.plan
        # CALM between moments (user: "busy"): with the bed whole, the arrangement moves at most once per
        # `calm_phrases` phrases and not within a phrase after a moment; NEXT (forced), the plan's own bar
        # and an empty bed lane (housekeeping below) are exempt
        bed_whole = bed is not None and self.lanes.get("bass") is not None
        plan_due = p is not None and p["bar"] - (bar + 1) <= (self.change_bars - 1 if p["what"] == "bed_to" else 0)
        if not forced and bed_whole and not plan_due:
            if self._calm_until is not None and bar < self._calm_until:
                self.wait_why = f"a phrase of plain play after the moment ({self._calm_until - bar} bars)"
                return None
            if self.calm_phrases > 1 and self._arr_k is not None and bar - self._arr_k < self.calm_phrases * self.change_bars:
                self.wait_why = f"calm: one move every {self.calm_phrases} phrases ({self.calm_phrases * self.change_bars - (bar - self._arr_k)} bars to the next)"
                return None
        # 0. housekeeping first. THE BED IS NEVER EMPTY: a drums or bass lane left with nobody (a break whose
        #    song left before it ended, an eviction with no target, a landing that never came - the user heard
        #    "the bass track empty for minutes") goes back to the bed song at once, or to the song holding most
        #    lanes / the clock when there is no bed; a rested `other` of the bed comes back after a phrase; the
        #    VOCAL lane is kept bar by bar in _vocal_tick
        if bed is None:
            holders = {}
            for ln, dd in self.lanes.items():
                if dd is not None and dd in self.songs and self.songs[dd].entered and not self.songs[dd].leaving:
                    holders[dd] = holders.get(dd, 0) + 1
            cand = max(holders, key=holders.get) if holders else (self.master if self.master in self.songs else None)
            if cand is not None:
                tag = None
                for ln in ("drums", "bass"):
                    if self.lanes.get(ln) != cand and self._lane_ok(cand, ln, at):
                        tag = self._cross(ln, cand, at) or tag
                self.songs[cand].bed_since = bar
                self._move("return", "drums", None, self._song_id(cand), f"{self.songs[cand].track.title} takes the bed (drums and bass had nobody)", shape=tag)
                return tag
        else:
            for ln in ("bass", "drums"):
                if self.lanes.get(ln) is None and self._lane_ok(bed, ln, at):
                    tag = self._cross(ln, bed, at)
                    self._move("return", ln, None, self._song_id(bed), f"{ln} back on {self.songs[bed].track.title} (the bed is never empty)", shape=tag)
                    return tag
            ln = "other"
            if self.lanes.get(ln) is None and bar - self.lane_k.get(ln, -10 ** 6) >= self.change_bars and self._lane_ok(bed, ln, at):
                tag = self._cross(ln, bed, at)
                self._move("return", ln, None, self._song_id(bed), f"{ln} returns on {self.songs[bed].track.title} (the bed is whole again)", shape=tag)
                return tag
        # 1. the old bed song, now only a voice, leaves after a phrase - on the plan's bar
        for d in list(voices):
            s = self.songs[d]
            if s.voice_since is None:
                s.voice_since = bar                   # a lane came to it outside the policy (an evict, a recall, your grid)
            planned_out = p is not None and p["what"] == "voice_out" and p["deck"] == d
            due_out = planned_out and bar + 1 >= p["bar"]
            running_out = due_out and self._on_way_out(d, at) and not getattr(s, "was_bed", False)
            if (getattr(s, "was_bed", False) and bar - s.voice_since >= self.change_bars and (forced or p is None or due_out)) or running_out:
                tag = None
                for ln in ("other", "vocals"):
                    if self.lanes.get(ln) == d:
                        to = bed if (bed is not None and self._lane_ok(bed, ln, at)) else None
                        tag = self._cross(ln, to, at) or tag
                self._move("voice_out", None, s.track.id, self._song_id(bed),
                           f"{s.track.title} fades out as a voice" + (" (it is running out)" if running_out else "; the bed carries on"), shape=tag)
                return tag
        # 2. a voice that has been heard long enough takes the bed (drums + bass together, under its voice) -
        #    AT ITS OWN DROP when one comes within the window (the payoff lands the change), by
        #    VOICE_MAX_PHRASES regardless, at once when the bed is running out
        bed_s = self.songs.get(bed) if bed is not None else None
        bed_kind = self._kind_at(bed, at) if bed is not None else None
        bed_out = bed_s is not None and (bed_s.loop is not None or bed_kind == "outro") or forced
        for d in sorted(voices, key=lambda x: getattr(self.songs[x], "voice_since", 0)):
            s = self.songs[d]
            if getattr(s, "was_bed", False) or d == self._drums_follow:
                continue
            if self._on_way_out(d, at):
                self.wait_why = f"{s.track.title[:28]} is running out: it never takes the bed"
                continue
            heard = bar - (s.voice_since if s.voice_since is not None else bar)
            planned = p is not None and p["deck"] == d and p["what"] in ("bed_to", "voice_out")
            plan_now = planned and p["bar"] - (bar + 1) <= self.change_bars - 1
            # the plan already weighed how long the voice has been heard (a phrase at least when the bed is
            # dying): its bar is not held back by the three-phrase rule (two changes planned "before the
            # breakdown" came a phrase late and the bed sat in its breakdown with a melody on top)
            if heard < (1 if (forced or plan_now) else VOICE_PHRASES) * self.change_bars:
                continue
            if not self._lane_ok(d, "bass", at):
                if heard >= VOICE_MAX_PHRASES * self.change_bars or (planned and p["what"] == "voice_out" and bar + 1 >= p["bar"]):
                    # it has had its phrases and can never take the bed (its bass clashes with the room):
                    # it fades out as a voice so the next song can come
                    tag = None
                    for ln in ("other", "vocals"):
                        if self.lanes.get(ln) == d:
                            to = bed if (bed is not None and self._lane_ok(bed, ln, at)) else None
                            tag = self._cross(ln, to, at) or tag
                    s.was_bed = True
                    self._move("voice_out", None, s.track.id, self._song_id(bed), f"{s.track.title} fades out as a voice (its bass would clash with the room)", shape=tag)
                    return tag
                if planned and p["what"] == "bed_to":
                    self._plan_move("voice_out", d, max(bar + 1, (s.voice_since or bar) + VOICE_MAX_PHRASES * self.change_bars),
                                    "its bass would clash with the room: it fades out instead of taking the bed")
                self.wait_why = f"{s.track.title[:28]}'s bass would clash with the room: it stays a voice"
                continue
            if forced or p is None:
                d_bars, what = self._payoff_in_bars(d, at, self.change_bars)
                return self._bed_to(d, at, d_bars, what)
            if not planned or p["what"] != "bed_to":
                self.wait_why = f"{s.track.title[:28]} is heard; the plan is {self._plan_words()} first"
                continue
            left = p["bar"] - (bar + 1)
            if left > self.change_bars - 1 and not bed_out:
                self.wait_why = f"{s.track.title[:28]} takes the bed at bar {p['bar']} ({left} bars): {p['why']}"
                continue
            return self._bed_to(d, at, max(0, left), p.get("payoff"))
        # 3. a staged song arrives as a voice over the bed - on the PLAN's bar (the settle, the calm and the
        #    bed's build already in it); a breakdown of the bed or a bed running out opens the door early
        waiting = [d for d in live if not self.songs[d].entered and (self.songs[d].ready_k is None or bar + 1 >= self.songs[d].ready_k)]
        if waiting and len(voices) < max_voices and bed_s is not None and not forced:
            planned_in = p is not None and p["what"] == "voice_in" and p["deck"] in waiting
            if not planned_in:
                self.wait_why = f"{self.songs[waiting[0]].track.title[:28]} is ready; the plan is {self._plan_words()} first" if p is not None else self.wait_why
                waiting = []
            else:
                waiting = [p["deck"]]
                left = p["bar"] - (bar + 1)
                if left > 0 and not bed_out and bed_kind != "breakdown":
                    self.wait_why = f"{self.songs[p['deck']].track.title[:28]} arrives at bar {p['bar']} ({left} bars): {p['why']}"
                    waiting = []
                elif self._body_left_s(p["deck"], at) < MIN_RUNWAY_S:
                    # it would arrive with no body left (one arrived and looped its last bars 16 s later): it is
                    # cued back so its landmark lands a phrase from now, and arrives then
                    self._plan_move("voice_in", p["deck"], bar + 1 + self.change_bars, "cued back: it would have arrived with no body left")
                    self._recue_for(p["deck"], bar + 1 + self.change_bars)
                    waiting = []
                elif bed_kind == "build" and not bed_out:
                    # the plan reckoned without this build (a section map can be off): it moves once to the build's end
                    t_now = self._song_time_at(bed, at)
                    sec = bed_s.track.section_at(t_now) or {}
                    end_bars = int(math.ceil((float(sec.get("end_s", t_now)) - t_now) / max(self._bar_s(bed_s.track, t_now), 1e-6))) if sec else self.change_bars
                    self._plan_move("voice_in", p["deck"], bar + 1 + max(1, end_bars), f"{bed_s.track.title[:24]} is building to its drop: nothing arrives on top of it")
                    waiting = []
        if waiting and len(voices) < max_voices:
            d = waiting[0]
            s = self.songs[d]
            # HOW A SONG ARRIVES (the user's yes, 2026-09-13, after ten "clash" flags that no key or chord
            # measure could hear - what they shared was another song's full melody over the bed, mostly while
            # it sang): 1. through its VOCAL - an acapella over an instrumental bed - when it sings through the
            # coming phrase and the bed does not; 2. through its MELODY over an INSTRUMENTAL bed, at a texture
            # level (OTHER_VOICE); 3. when the bed sings, through its BASS - the bass swap, the change the
            # operator asked for from the start: its bassline takes over under the bed's drums, the drums follow
            # at the phrase. Never a melody over a singer. (Bass first for everything made every arrival a bed
            # change and no song was ever a voice - the gate's first run of this rule.)
            bed_sings = bed is not None and (self.lanes.get("vocals") == bed or self._sings_through(bed, at, self.change_bars) is True
                                             or self._singing(self.songs[bed], self._song_time_at(bed, at)) is True)
            # both songs must GROOVE through the coming phrase: no arrival on top of the bed's build or
            # breakdown, none from a song in its own (two builds at once was flagged twice)
            bed_grooves = bed is None or self._groove_through(bed, at, self.change_bars) is not False
            if not bed_grooves and not bed_out:
                g = self._groove_start_bars(bed, at)
                self._plan_move("voice_in", d, bar + 1 + (g if g else self.change_bars), f"{self.songs[bed].track.title[:24]} is building or breaking down: it arrives when that is over")
                waiting = []
            elif self._groove_through(d, at, self.change_bars, strict=True) is False:
                g = self._groove_start_bars(d, at, strict=True)
                self._plan_move("voice_in", d, bar + 1 + (g if g else self.change_bars), f"{s.track.title[:24]} is not in a groove itself: it arrives at its next one")
                waiting = []
        if waiting and len(voices) < max_voices:
            d = waiting[0]
            s = self.songs[d]

            def _free(ln):
                # a voice lane may be taken from the bed or from a voice on its way out - never from a voice
                # that has not had its turn (a song is heard until it takes the bed or fades out)
                h = self.lanes.get(ln)
                return h is None or h == bed or h not in self.songs or self.songs[h].was_bed or self.songs[h].leaving
            lane, how = None, None
            if (self.vocal_freedom > 0.1 and not bed_sings and self._vocal_ok(d, at) and self._sings_through(d, at, self.change_bars) is not False
                    and _free("vocals") and self._lane_ok(d, "vocals", at)):
                lane, how = "vocals", f"{s.track.title} arrives through its vocal over the instrumental bed"
            elif not bed_sings and _free("other") and self._lane_ok(d, "other", at):
                lane, how = "other", f"{s.track.title} arrives through its melody over the instrumental bed, held low"
            elif bed is not None and bed in self.songs and not self._on_way_out(d, at) and self._lane_ok(d, "bass", at) \
                    and self._land_k is None and self._drums_follow is None:
                # the bed sings: the bass swap - a bed change from the first bar (the plan's arrival bar was its bar)
                s.voice_since = bar
                tag = self._bed_to(d, at, None, None, bass_first=True)
                return tag
            else:
                self.wait_why = (f"{s.track.title[:28]} waits: the bed is singing - no melody over a singer, and its bass would clash"
                                 if bed_sings else f"{s.track.title[:28]} waits: no lane it can take yet")
            if lane is not None:
                frm = self.lanes.get(lane)
                tag = self._cross(lane, d, at)
                s.voice_since = bar
                self._move("voice_in", lane, self._song_id(frm), s.track.id, how, shape=tag)
                return tag
        # (no random rests: "too busy, and too happy to have large sections of vocal rest" - user 2026-09-12;
        #  a phrase with nothing to do is a phrase of music)
        return None

    # -- THE PLAN: one next arrangement event, decided once and kept ----------------------------------------------
    # "the system seems unable to make up its mind when planning. it keeps changing the length of how long
    # something will play" (user 2026-09-13): every countdown was recomputed each tick from the state. Now the
    # next event - a voice arriving, a voice taking the bed, the old bed fading out - is decided ONCE, with the
    # settle, the calm and the song's own payoff in its bar, and kept until it is done; a rule that must move
    # it says so in the story (`plan` moves).
    def _plan_words(self, p=None):
        p = p or self.plan
        if p is None:
            return "nothing"
        s = self.songs.get(p["deck"])
        title = s.track.title[:24] if s else "?"
        return {"voice_in": f"{title} arriving as a voice", "bed_to": f"{title} taking the bed",
                "voice_out": f"{title} fading out", "land": f"the drums of {title} following (the bed is its)"}.get(p["what"], p["what"]) + f" at bar {p['bar']}"

    def _plan_set(self, what, deck, bar, why, payoff=None):
        self._plan_seq = getattr(self, "_plan_seq", 0) + 1
        self.plan = {"what": what, "deck": deck, "bar": int(bar), "why": why, "payoff": payoff, "set_k": self.bar_n, "moved": 0, "seq": self._plan_seq}
        self._note(f"plan: {self._plan_words()} ({int(bar) - self.bar_n} bars): {why}")
        if what == "voice_in":
            self._recue_for(deck, int(bar))

    def _plan_move(self, what, deck, bar, why):
        p = self.plan
        if p is None:
            return self._plan_set(what, deck, bar, why)
        p.update({"what": what, "deck": deck, "bar": int(bar), "why": why, "moved": p.get("moved", 0) + 1})
        self._move("plan", None, None, self._song_id(deck), f"plan moved: {self._plan_words()} - {why}")
        if what == "voice_in":
            self._recue_for(deck, int(bar))

    def _recue_for(self, deck, arrive_bar):
        """A staged song plays silently until its arrival: cue it so its LANDMARK sounds on `arrive_bar` (the
        staging cue assumed the next move; the plan may be minutes later, and one song arrived with 75 s of
        body left and looped at once). Nobody hears the jump."""
        s = self.songs.get(deck)
        if s is None or s.entered or s.leaving or s.staged_at is None:
            return
        n = arrive_bar - (self.bar_n + 1)
        if n < 2:
            return                                    # too close: a jump now would arrive before its phase has re-locked (20 s off in the gate)
        t = s.track
        landmark = self._landmark(t)
        bar_song = self._bar_s(t, landmark)
        cue = landmark - n * bar_song
        if cue < 0.0:
            cue = landmark - bar_song * math.floor(landmark / bar_song)
        at = self._next_bar_clock()
        now_t = self._song_time_at(deck, at)
        if abs(now_t - cue) < 0.5 * bar_song:
            return                                    # already there
        # the cue is a jump: the PLL is re-snapped on it (a re-cued song that arrived a phrase later was 0.4 beat
        # off for twenty seconds in the gate while the loop pulled it back)
        # (the re-snap a beat after the cue: read in the same instant, the deck's phase was still the old position)
        self.submix.post_many([{"at": at, "cmd": "cue", "deck": deck, "time_s": t.nearest_downbeat(cue)},
                               {"at": at + int(self._beat_s() * RATE), "cmd": "sync", "slave": deck, "master": self.master, "bias_beats": 0.0, "audio_pll": False}])
        s.ready_k = arrive_bar
        self.recue_log = (getattr(self, "recue_log", []) + [(self._tel().get("clock_s"), deck, round(cue, 1), arrive_bar)])[-40:]
        self._note(f"{t.title[:24]} cued to {cue:.0f} s: its landmark lands at bar {arrive_bar}")

    def _plan_check(self, at):
        """Keep the plan while its subject stands; when there is none, decide the next event once."""
        bar = self.bar_n
        bed = self._bed_deck()
        p = self.plan
        # a bed change in flight IS the plan: the drums follow (or the new bed lands) at _land_k - shown as
        # such, not as a stale "takes the bed" with a countdown gone negative (flag 'plan' 2026-09-13)
        if self._land_k is not None and bar < self._land_k:
            d = self._drums_follow if self._drums_follow is not None else self.lanes.get("bass")
            if d is not None and d in self.songs and not (p is not None and p["what"] == "land" and p["deck"] == d and p["bar"] == self._land_k):
                self._plan_set("land", d, self._land_k, "the bassline has taken over; its drums follow at the phrase" if self._drums_follow else "the new bed lands")
            return
        if p is not None:
            s = self.songs.get(p["deck"])
            stale = (s is None or s.leaving
                     or p["what"] == "land"
                     or (p["what"] == "voice_in" and s.entered)
                     or (p["what"] == "bed_to" and p["deck"] == bed)
                     or (p["what"] in ("bed_to", "voice_out") and not s.held(self.lanes))
                     # MISSED: its bar is a phrase gone and it did not happen (an arrival decided before your
                     # vocal took the only voice slot sat at "bar 16 (0 bars)" for eighty bars while the voice
                     # never got a plan to take the bed) - decided afresh from the room as it is now
                     or bar > p["bar"] + self.change_bars)
            if not stale:
                return
            if bar > p["bar"] + self.change_bars:
                self._note(f"plan missed: {self._plan_words(p)} - decided afresh")
            self.plan = None
        earliest = bar + 1
        if self.calm_phrases > 1 and self._arr_k is not None:
            earliest = max(earliest, self._arr_k + self.calm_phrases * self.change_bars)
        if self._calm_until is not None:
            earliest = max(earliest, self._calm_until)
        voices = {d for ln in ("other", "vocals") for d in [self.lanes.get(ln)] if d is not None and d != bed and d in self.songs and not self.songs[d].leaving}
        bar_c = self._bar_clock()
        # (a) the old bed, now a voice, fades out a phrase after it became one
        for d in sorted(voices, key=lambda x: self.songs[x].voice_since or 0):
            s = self.songs[d]
            if s.was_bed:
                self._plan_set("voice_out", d, max(earliest, (s.voice_since if s.voice_since is not None else bar) + self.change_bars),
                               "the old bed fades out as a voice")
                return
        # (b) a voice takes the bed: at its payoff once it has had its phrases, else by its limit - and before
        #     the CURRENT bed dies (its breakdown, its outro, its body's end): a bed change waits for nothing
        #     when the bed is about to stop carrying
        bed_dead = self._bed_dies_in_bars(bed, at) if bed is not None else None
        for d in sorted(voices, key=lambda x: self.songs[x].voice_since or 0):
            s = self.songs[d]
            if d == self._drums_follow:
                continue
            if self._on_way_out(d, at):
                self._plan_set("voice_out", d, earliest, "it is running out: it fades out instead of taking the bed")
                return
            since = s.voice_since if s.voice_since is not None else bar
            first = max(earliest, since + VOICE_PHRASES * self.change_bars)
            last = max(first, since + VOICE_MAX_PHRASES * self.change_bars)
            deadline = bed_dead is not None and bar + 1 + bed_dead < last
            if deadline:
                # the bed dies first: the change lands a bar before that, once the voice has had a phrase
                first = max(earliest, min(first, max(since + self.change_bars, bar + 1)))
                last = max(first, bar + bed_dead)
            at_first = at + max(0, first - (bar + 1)) * bar_c
            pay, what = self._payoff_in_bars(d, at_first, max(1, last - first), min_groove_bars=2 * self.change_bars)
            if not self._lane_ok(d, "bass", at):
                self._plan_set("voice_out", d, last, "its bass would clash with the room: it fades out instead of taking the bed")
                return
            if pay is not None:
                K, why = first + int(round(pay)), f"at its {what}"
            elif deadline:
                K, why, what = last, f"before {self.songs[bed].track.title[:20]}'s breakdown", None
            else:
                K, why, what = last, f"by its phrases ({'no hook or drop' if s.track.hook else 'no drop'} in sight)", None
            odd = self._odd_grid_bars(d, at + max(0, K - (bar + 1)) * bar_c)
            if odd:
                K, why, what = K + odd, f"{why}, after a mis-tracked stretch of its grid", None
            # the new bed must GROOVE from its landing through two phrases (a bassline took over as its song
            # entered its breakdown): a landing in an intro, build or breakdown moves to its next groove
            if what is None:
                g = self._groove_start_bars(d, at + max(0, K - (bar + 1)) * bar_c, 4 * self.change_bars, strict=True)
                if g:
                    K, why = K + g, f"{why}, at its next groove"
            self._plan_set("bed_to", d, K, why, payoff=what)
            return
        # (c) a staged song arrives as a voice once the bed has settled (and the song is ready)
        max_voices = 2 if self.blend >= 0.5 else 1
        bed_s = self.songs.get(bed) if bed is not None else None
        waiting = [d for d, s in self.songs.items() if not s.leaving and s.staged_at is not None and not s.entered and not s.evicted]
        waiting.sort(key=lambda x: not getattr(self.songs[x], "user_pick", False))   # YOUR pick first, whatever staged earlier
        if waiting and len(voices) < max_voices and bed_s is not None:
            d = waiting[0]
            s = self.songs[d]
            since = bed_s.bed_since if bed_s.bed_since is not None else bar
            A = max(earliest, since + SETTLE_PHRASES * self.change_bars)
            why = "once the bed has settled" if A > bar + 1 else "the bed has settled"
            kind_now = self._kind_at(bed, at)
            if bed_s.loop is not None or kind_now in ("outro", "breakdown"):
                A, why = earliest, ("the bed is running out" if kind_now == "outro" or bed_s.loop is not None else "the bed is in a breakdown: room for a voice")
            elif bed_dead is not None and bar + 1 + bed_dead < A + self.change_bars:
                # the bed dies before a voice would have had a phrase over it: the voice comes a phrase before that
                A = max(earliest, bar + 1 + bed_dead - self.change_bars)
                why = f"a phrase before {bed_s.track.title[:20]}'s breakdown"
            else:
                # never on top of the bed's build: if the bar falls in one, the plan takes the build's end
                t_A = self._song_time_at(bed, at + (A - bar - 1) * bar_c)
                sec = bed_s.track.section_at(t_A) or {}
                if sec.get("kind") == "build":
                    A += int(math.ceil((float(sec.get("end_s", t_A)) - t_A) / max(self._bar_s(bed_s.track, t_A), 1e-6)))
                    why = f"after {bed_s.track.title[:20]}'s build"
            odd = self._odd_grid_bars(d, at + (A - bar - 1) * bar_c)
            if odd:
                A, why = A + odd, f"{why}, once its grid is steady"
            self._plan_set("voice_in", d, A, why)

    def _vocal_tick(self):
        """Every bar, the vocal lane follows the singing: it rests when the song on it falls silent for the
        next two bars, and comes back on the bed the moment the bed sings again - not at the next phrase
        move. Never during a break or a recall, never while a bed change is in flight."""
        if self._break is not None or self._recall is not None or (self._land_k is not None and self.bar_n < self._land_k):
            return
        if getattr(self, "_vtick_k", None) == self.bar_n:
            return
        self._vtick_k = self.bar_n
        at = self._next_bar_clock()
        bed = self._bed_deck()
        vd = self.lanes.get("vocals")
        since = self.bar_n - self.lane_k.get("vocals", -10 ** 6)
        # dwell: a lane that just came back holds four bars before it may rest, one that just rested holds two
        # before it may come back (a sparse vocal flipped rest / back within seconds in the gate)
        if vd is not None and vd in self.songs:
            # on for a phrase before it may rest, and only when the singing stops for VOCAL_REST_BARS ahead;
            # never while YOUR loud on it runs (you asked for that vocal: it stays through its phrase)
            ub = self.lane_boost.get("vocals")
            if ub is not None and len(ub) > 2 and ub[2] == "user" and self.bar_n < ub[1]:
                return
            if since >= self.change_bars and self._vocal_gone(vd, at):
                self._cross("vocals", None, at, beats=2.0, shape=False)
                self._move("rest", "vocals", self._song_id(vd), None, f"vocals of {self.songs[vd].track.title[:30]} rest (the singing stops)")
                self._reduck(at)
            return
        if since < VOCAL_REST_BARS:
            return
        # back only when a bed song sings NOW and keeps singing through the coming bars (a sparse vocal flapped
        # rest / return every few seconds - four pairs in five minutes on 2026-09-13). The bed is the drums'
        # song - or, during a bass-first change, the song whose bassline has taken over: its hook may be
        # the reason it was chosen (Need You Now's vocal missed its own hook while its bass played)
        for cand in (self.lanes.get("bass"), bed):
            if cand is None or cand not in self.songs or not self.songs[cand].entered or self.songs[cand].leaving:
                continue
            # (back after two sung bars; the REST needs four silent ones - a vocal with short gaps never rests, so it cannot flap)
            if self._vocal_ok(cand, at) and self._singing(self.songs[cand], self._song_time_at(cand, at)) is True \
                    and self._sings_through(cand, at, 2) is not False \
                    and self._tonal_ok(cand, "vocals", at):
                self._cross("vocals", cand, at, beats=1.0, shape=False)
                self._move("return", "vocals", None, self._song_id(cand), f"vocals back on {self.songs[cand].track.title[:30]} (singing again)")
                self._reduck(at)
                return

    def _reduck(self, at):
        """The `other` lane's level follows the vocal lane: ducked under another song's singing, full when
        the vocal rests (a returning vocal at x2 once sat over an unducked melody - flag 'clash' 2026-09-13)."""
        od = self.lanes.get("other")
        if od is None or od not in self.songs or self.songs[od].leaving:
            return
        self.submix.post({"at": at, "cmd": "stem_gains", "deck": od, "gains": {"other": self._lane_gain(od, "other", at)}, "ramp_s": self._beat_s()})

    def _one_move_inner(self, at):
        if getattr(self, "policy", "arrangement") == "arrangement":
            n0 = len(self.move_log)
            tag = self._arrangement_move(at)
            if any(m["kind"] != "plan" for m in self.move_log[n0:]):
                self._arr_k = self.bar_n
                self.plan = None                      # done (or overtaken): the next event is decided on the next step
            return tag
        live = [d for d, s in self.songs.items() if not s.leaving and s.staged_at is not None]
        # a staged song comes in through one lane - on the move its landmark was cued for (or any later one),
        # and only while fewer songs are heard than the cap (the blend dial: two, or three)
        cap = 2 if self.blend < 0.5 else 3
        n_entered = len([d for d in live if self.songs[d].entered])
        waiting = [d for d in live if not self.songs[d].entered
                   and (self.songs[d].ready_k is None or self.bar_n + 1 >= self.songs[d].ready_k)]
        if waiting and n_entered >= cap:
            waiting = []
        if waiting:
            d = waiting[0]
            lane = self._lane_to_give(d, at)
            if lane is not None:
                frm = self.lanes.get(lane)
                tag = self._cross(lane, d, at)
                s = self.songs[d]
                self._move("enter", lane, self._song_id(frm), s.track.id,
                           f"{s.track.title} enters through {lane} (rate {s.rate:.3f}, shift {s.shift:+d} st)", shape=tag)
                return tag
        entered = [d for d in live if self.songs[d].entered]
        # songs looping their last bars are on the way out: they give lanes, they do not take them
        fresh = [d for d in entered if self.songs[d].loop is None] or entered
        # the vocal lane leaves a song whose singing has stopped
        vd = self.lanes.get("vocals")
        if vd is not None and vd in self.songs and self._vocal_gone(vd, at):
            tgt = [d for d in fresh if d != vd and self._lane_ok(d, "vocals", at)]
            if tgt:
                d = self.rng.choice(tgt)
                tag = self._cross("vocals", d, at)
                self._move("cross", "vocals", self._song_id(vd), self._song_id(d), f"vocals cross to {self.songs[d].track.title} (the singing stopped)", shape=tag)
            else:
                tag = self._cross("vocals", None, at)
                self._move("rest", "vocals", self._song_id(vd), None, "vocals rest (the singing stopped)", shape=tag)
            return tag
        # a resting lane comes back first
        for lane in STEMS:
            if self.lanes.get(lane) is None:
                tgt = [d for d in fresh if self._lane_ok(d, lane, at)]
                if tgt:
                    d = self.rng.choice(tgt)
                    tag = self._cross(lane, d, at)
                    self._move("return", lane, None, self._song_id(d), f"{lane} returns on {self.songs[d].track.title}", shape=tag)
                    return tag
        if len(entered) < 2:
            return None
        # the blend: a consolidating move (the newest song takes its next lane) or a free recombination;
        # the verdict record tilts the choice (2 x weight: 0.5 = as set, 1.0 = twice as often)
        p_cons = (1.0 - self.blend) * 2 * self._w("cross_new")
        if self.rng.random() < p_cons:
            newest = max(fresh, key=lambda d: self.songs[d].entered_k or 0)
            for lane in MORPH_ORDER:
                if self.lanes.get(lane) != newest and self._lane_ok(newest, lane, at) and self._takeable(lane):
                    frm = self.lanes.get(lane)
                    tag = self._cross(lane, newest, at)
                    self._move("cross_new", lane, self._song_id(frm), self._song_id(newest),
                               f"{lane} crosses to {self.songs[newest].track.title} (toward the new song)", shape=tag)
                    return tag
        lanes = list(STEMS)
        self.rng.shuffle(lanes)
        # liked lanes first (stable on ties); a lane that moved within the last two phrases goes last, so
        # `other` does not ping-pong between two songs phrase after phrase (seen in the tab smoke)
        recent = 2 * self.change_bars
        lanes.sort(key=lambda ln: (self.bar_n - self.lane_k.get(ln, -10 ** 6) < recent, -self._w("cross", ln)))
        for lane in lanes:
            if lane == "vocals" and self.rng.random() > min(1.0, self.vocal_freedom * 2 * self._w("cross", "vocals")):
                continue
            if not self._takeable(lane):
                continue
            cur = self.lanes.get(lane)
            targets = [d for d in fresh if d != cur and self._lane_ok(d, lane, at)]
            if cur is not None and self.rng.random() < REST_CHANCE * 2 * self._w("rest", lane):
                tag = self._cross(lane, None, at)
                self._move("rest", lane, self._song_id(cur), None, f"{lane} rests for a phrase", shape=tag)
                return tag
            if targets:
                d = self.rng.choice(targets)
                tag = self._cross(lane, d, at)
                self._move("cross", lane, self._song_id(cur), self._song_id(d), f"{lane} crosses to {self.songs[d].track.title}", shape=tag)
                return tag
        return None

    def _lane_to_give(self, deck, at):
        """The lane a new song enters through: a resting lane it fits, else - in the morph order - a
        lane from the OLDEST song still holding lanes (songs arrive and push the oldest out), guarded."""
        order = sorted(MORPH_ORDER, key=lambda ln: (-self._w("enter", ln), MORPH_ORDER.index(ln)))
        for lane in order:
            if self.lanes.get(lane) is None and self._lane_ok(deck, lane, at):
                return lane
        holders = {d for d in self.lanes.values() if d is not None and d in self.songs}
        for old in sorted(holders, key=lambda d: (self.songs[d].entered_k or 0)):
            for lane in order:
                if self.lanes.get(lane) == old and self._lane_ok(deck, lane, at) and self._takeable(lane):
                    return lane
        for lane in order:
            if self._lane_ok(deck, lane, at):
                return lane
        return None

    # -- the instrument: the operator's own moves ---------------------------------------------------------------
    def set_auto(self, x):
        """Autopilot amount 0..1: the chance the conductor takes each phrase's move (0 = only you move)."""
        self.auto = float(max(0.0, min(1.0, x)))

    def candidates(self, n=12, query=None):
        """The crate: songs that fit the clock NOW, ranked - tempo inside the wall, key fit to the session
        key (with the shift it would take), energy against the arc - with the reasons. From the pool when
        one is set. Cached for two seconds; a `query` filters titles and artists."""
        now = time.time()
        if self._cands_cache[1] == query and now - self._cands_cache[0] < 2.0:
            return self._cands_cache[2][:n]
        out = []
        if self.master is not None and self.master_bpm:
            busy = {s.track.id for s in self.songs.values()} | {t.id for t in self._decoding.values()} | {p[0].id for p in self._pending.values()}
            target = self.energy_target()
            q = (query or "").strip().lower()
            for t in self.library:
                if t.id in busy or (self.brain.pool_ids is not None and t.id not in self.brain.pool_ids):
                    continue
                if q and q not in (t.title or "").lower() and q not in (t.artist or "").lower():
                    continue
                rate = self.master_bpm / max(t.bpm, 1e-6)
                if not (RATE_MIN <= rate <= RATE_MAX) or (t.bpm_conf or 0) < 0.5:
                    continue
                if (t.duration_s - self._body_start(t)) / rate < MIN_RUNWAY_S:
                    continue
                shift, compat = self._key_fit(t)
                try:
                    e = float(self.brain._arc_energy(t))
                except Exception:
                    e = 0.5
                fit_key = compat if compat is not None else 0.6
                fit_tempo = max(0.0, 1.0 - abs(math.log(rate)) / 0.1)
                fit_e = max(0.0, 1.0 - abs(e - target) / 0.5)
                score = 0.5 * fit_key + 0.3 * fit_tempo + 0.2 * fit_e
                why = []
                why.append(f"{'+' if rate >= 1 else ''}{100 * (rate - 1):.1f}% tempo")
                if compat is not None:
                    why.append(f"key {t.camelot}{f' shifted {shift:+d}' if shift else ''} fit {compat:.2f}")
                else:
                    why.append("key unknown")
                why.append(f"energy {e:.2f} vs {target:.2f}")
                out.append({"id": t.id, "title": t.title, "artist": t.artist, "bpm": t.bpm, "camelot": t.camelot,
                            "rate": rate, "shift": shift, "compat": compat, "energy": e, "score": score, "why": ", ".join(why)})
            out.sort(key=lambda c: -c["score"])
        self._cands_cache = (now, query, out)
        return out[:n]

    def stage_track(self, track_id):
        """The operator picks a song from the crate: it decodes and stages on a free deck (beat-locked,
        key-fitted, silent) and appears as a grid column. Returns (ok, message)."""
        t = self._track(int(track_id))
        if t is None:
            return False, "not in the stem library"
        if self.master is None:
            return False, "nothing is playing yet"
        busy = {s.track.id for s in self.songs.values()} | {x.id for x in self._decoding.values()} | {p[0].id for p in self._pending.values()}
        if t.id in busy:
            return False, "already on a deck"
        rate = self.master_bpm / max(t.bpm, 1e-6)
        if not (RATE_MIN <= rate <= RATE_MAX):
            return False, f"{t.bpm:.0f} bpm is outside the tempo wall for this clock ({rate:.3f})"
        free = [d for d in DECKS if d not in self.songs and d not in self._decoding and d not in self._pending]
        if not free:
            return False, "no free deck - eject one first"
        self._user_picks.add(t.id)                    # yours: it arrives before anything the conductor picked itself
        self._decode(t, free[0])
        self._note(f"{t.title} staging on {free[0].upper()} (your pick)")
        return True, f"{t.title} → deck {free[0].upper()}"

    def why_not(self, lane, deck):
        """Why the grid cell (lane, deck) is refused right now; None when it is allowed."""
        s = self.songs.get(deck)
        if s is None:
            return "no song on that deck"
        if s.leaving:
            return "leaving"
        if s.staged_at is None:
            return "still decoding"
        at = self._next_bar_clock()
        if lane in TONAL:
            for other in TONAL:
                if other == lane:
                    continue
                od = self.lanes.get(other)
                if od is None or od == deck or od not in self.songs:
                    continue
                o = self.songs[od]
                fit = self._pair_fit(o, s, at)
                if fit is not None and fit < CLASH_BELOW:
                    cam = _compat(o.key(), s.key()) if (o.track.camelot and s.track.camelot) else None
                    if cam is not None and cam >= CLASH_BELOW:
                        return f"the chords clash right now with the {other} of {o.track.title[:24]} (keys {o.key()} / {s.key()} agree, the sections do not)"
                    return f"key clash with the {other} of {o.track.title[:24]} ({o.key()} vs {s.key()})"
        if lane == "vocals" and self._singing(s, self._song_time_at(deck, at)) is False:
            return "not singing there now"
        return None

    def assign(self, lane, deck, force=False):
        """The operator puts `lane` on `deck` (None = rest) on the next bar, through the same crossfade,
        shapes and hygiene the conductor uses. Refused with the reason when the guard says no - unless
        `force` (the timeline: the operator placed it; only a missing or leaving song still refuses)."""
        if lane not in STEMS or self.master is None:
            return False, "no clock"
        if deck is not None:
            why = self.why_not(lane, deck)
            if why and (not force or why in ("no song on that deck", "leaving", "still decoding")):
                return False, why
        if self.lanes.get(lane) == deck:
            return True, "already there"
        at = self._next_bar_clock()
        frm = self.lanes.get(lane)
        if self._break is not None:
            self._break = None
        tag = self._cross(lane, deck, at)
        self._hygiene(at + int(XFADE_BEATS * self._beat_s() * RATE))
        self.phrase_bars = 0
        to_title = self.songs[deck].track.title if deck in self.songs else None
        self._move("manual", lane, self._song_id(frm), self._song_id(deck),
                   f"you: {lane} → {to_title}" if deck is not None else f"you: {lane} rests", shape=tag)
        return True, f"{lane} → {to_title or 'rest'} on the next bar"

    def bring(self, lane, deck, boost=1.0, bars=8, when="bar"):
        """THE TACTICAL MOVE: this song's `lane` in, on the next bar (or the next phrase), `boost` loud for
        `bars` bars, the previous holder's stem gone in a beat. The guard still speaks (a key clash refuses)
        but the arrangement's waits do not apply - you asked. Returns (ok, message)."""
        if lane not in STEMS or self.master is None or deck not in self.songs:
            return False, "no such song on a deck"
        s = self.songs[deck]
        if s.leaving or s.staged_at is None:
            return False, "still decoding" if s.staged_at is None else "leaving"
        if self._clock() < int(s.staged_at):
            return False, "still staging: on the next bar"   # its cue, start and sync land on the coming bar; the bring follows them
        if self._on_way_out(deck):
            return False, f"{s.track.title[:24]} is running out ({self._body_left_s(deck):.0f} s of body left)"
        why = self.why_not(lane, deck)
        if why:
            return False, why
        at = self._next_bar_clock()
        if when == "phrase":
            left = max(0, self.change_bars - 1 - self.phrase_bars)
            at += left * self._bar_clock()
        if self._break is not None:
            self._break = None
        if boost and boost != 1.0:
            self.lane_boost[lane] = (float(boost), self.bar_n + int(bars) + (0 if when != "phrase" else max(0, self.change_bars - 1 - self.phrase_bars)), "user")
        frm = self.lanes.get(lane)
        tag = self._cross(lane, deck, at, beats=1.0, shape=False)
        s.user_k = self.bar_n                         # your pick: it stays on its deck a while even when its lane rests (see _keep_runway, _vocal_tick)
        if s.voice_since is None:
            s.voice_since = self.bar_n
        self._hygiene(at + int(self._beat_s() * RATE))
        self.phrase_bars = 0
        self._move("moment_in", lane, self._song_id(frm), s.track.id,
                   f"you: the {lane} of {s.track.title} in" + (", loud" if boost and boost != 1.0 else "") + (" at the phrase" if when == "phrase" else " now"), shape=tag)
        return True, f"the {lane} of {s.track.title[:24]} comes in {'at the phrase' if when == 'phrase' else 'on the next bar'}"

    def release(self, lane):
        """The lane back to the bed (the song holding the drums) on the next bar; the boost ends."""
        if lane not in STEMS or self.master is None:
            return False, "no clock"
        bed = self.lanes.get("drums") if lane != "drums" else None
        if bed is None:
            bed = max((d for d, s in self.songs.items() if s.entered and not s.leaving), key=lambda d: len(self.songs[d].held(self.lanes)), default=None)
        if bed is None or self.lanes.get(lane) == bed:
            return False, "already the bed's"
        self.lane_boost.pop(lane, None)
        at = self._next_bar_clock()
        if lane == "vocals" and not self._vocal_ok(bed, at):
            tag = self._cross(lane, None, at, beats=1.0, shape=False)
            self._move("moment_out", lane, None, None, f"you: the {lane} rests (the bed is not singing there)", shape=tag)
            return True, f"{lane} rests"
        frm = self.lanes.get(lane)
        tag = self._cross(lane, bed, at, beats=1.0, shape=False)
        self._move("moment_out", lane, self._song_id(frm), self._song_id(bed), f"you: the {lane} back to {self.songs[bed].track.title}", shape=tag)
        return True, f"{lane} back to the bed"

    # -- MOMENTS: composed shapes over 1-16 bars, landing on the one ------------------------------------------------
    # The literature's break routine (Solberg 2014/2019): take away (the beat and bass go, something melodic
    # stays), build (layers return in 2/4/8-bar steps, the frequency range opens), the hole (a beat to a bar
    # of nothing, the downbeat delayed), the payoff (bass drum and bass slam back on the one of a phrase,
    # louder). Each moment is a script of bar offsets from THE ONE; the arrangement's own moves pause while
    # one is armed or running, so nothing else moves under it. Armed = the landing is chosen (the target
    # song's drop or hook within four phrases, else its next phrase far enough away) and counted down.
    def _second_song(self, bed):
        """The other song for a two-song moment: the newest VOICE heard, else a song staged and waiting."""
        live = [(d, s) for d, s in self.songs.items() if d != bed and not s.leaving and s.staged_at is not None and not self._on_way_out(d)]
        voices = [(d, s) for d, s in live if s.entered]
        if voices:
            return max(voices, key=lambda ds: ds[1].entered_k or 0)[0]
        staged = [(d, s) for d, s in live if not s.entered]
        return staged[0][0] if staged else None

    def _phrase_ahead(self, deck, at, min_bars):
        """Bars from the bar at clock `at` to the song's next 8-bar phrase boundary (counted from the start of
        its section) at least `min_bars` away - or its next section start when that is nearer and far enough."""
        s = self.songs.get(deck)
        if s is None:
            return int(min_bars)
        t = self._song_time_at(deck, at)
        bar = self._bar_s(s.track, t)
        sec = s.track.section_at(t) or {}
        s0 = s.track.nearest_downbeat(float(sec.get("start_s", 0.0))) if sec else 0.0
        ph = 8 * bar
        n = math.ceil((t - s0 + min_bars * bar - 0.05 * bar) / ph)
        cand = (s0 + n * ph - t) / bar
        for x in s.track.sections or []:
            d = (s.track.nearest_downbeat(float(x["start_s"])) - t) / bar
            if min_bars - 0.05 <= d < cand:
                cand = d
        return int(round(max(min_bars, cand)))

    @staticmethod
    def _refusal_key(msg):
        for key in ("key clash", "chords clash", "no hook", "not singing", "no second song", "still decoding", "no vocal on a lane",
                    "nothing would be left", "not measured", "silent there", "no drop or hook", "neither drums nor bass", "not on the lane"):
            if key in msg:
                return key
        return msg[:40]

    def arm_moment(self, kind, auto=False, other=None):
        """Arm a composed moment: choose where it lands and count down. Pressing the same moment again while
        it is armed = go now (the script starts on the next bar). `other`: the deck of the second song for
        TAKEOVER / BRIDGE (else the newest voice or a staged song). Every arm and every refusal is counted
        per kind (moment_stats) so an unusable moment shows itself. Returns (ok, sentence)."""
        ok, msg = self._arm_moment_inner(kind, auto, other)
        if kind in MOMENTS:
            st = self.moment_stats.setdefault(kind, {"armed": 0, "refused": {}})
            if ok:
                st["armed"] += 1
            else:
                key = self._refusal_key(msg)
                st["refused"][key] = st["refused"].get(key, 0) + 1
                try:
                    from lib.dj.session import log_row
                    log_row({"event": "moment_refused", "kind": kind, "auto": bool(auto), "why": msg[:120]})
                except Exception:
                    pass
        return ok, msg

    def _arm_moment_inner(self, kind, auto, other):
        if kind not in MOMENTS:
            return False, "no such moment"
        if self.master is None:
            return False, "no clock"
        label = MOMENTS[kind][0]
        lead = self._moment_lead(kind)
        tune = self.moment_tune.get(kind, {})
        if self._moment is not None:
            if self._moment["kind"] == kind and not self._moment["started"]:
                return self.moment_go()
            self.moment_cancel()
        at = self._next_bar_clock()
        bar_c = self._bar_clock()
        B = self._bed_deck() or self.master
        if B not in self.songs or self.songs[B].leaving:
            return False, "no bed"
        held = {ln: d for ln, d in self.lanes.items() if d is not None and d in self.songs and not self.songs[d].leaving}
        V, T, what, hb = None, B, None, None
        if kind in ("slam", "break") and not any(held.get(ln) == B for ln in ("drums", "bass")):
            return False, "the bed holds neither drums nor bass"
        if kind == "break":
            stays = [ln for ln in ("other", "vocals") if ln in held]
            if not stays:
                return False, "nothing would be left playing (no melody or vocal on a lane)"
            if all(self.songs[held[ln]].loop is not None for ln in stays):
                return False, "the melody left is looping its last bars"
        if kind == "sing":
            S = held.get("vocals")
            if S is None:
                return False, "no vocal on a lane"
            hb = self._hook_in_bars(S, at, MOMENT_LOOK_PHRASES * self.change_bars)
            if hb is None:
                return False, f"no hook of {self.songs[S].track.title[:24]} within {MOMENT_LOOK_PHRASES} phrases"
            T, what = S, "hook"
        if kind in ("takeover", "bridge"):
            V = other if (other in self.songs and other != B and not self.songs[other].leaving and self.songs[other].staged_at is not None) else None
            if V is None and other is not None:
                return False, "that song is not ready on a deck" if other in self.songs else "that song is not on a deck"
            V = V or self._second_song(B)
            if V is None:
                return False, "no second song on a deck (put one UP NEXT)"
            why = self.why_not("bass", V)
            if why:
                return False, f"{self.songs[V].track.title[:24]}: {why}"
            if kind == "bridge":
                if held.get("vocals") != B:
                    return False, "the bed's vocal is not on the lane"
                if self._sings_through(B, at, 4) is False:
                    return False, f"{self.songs[B].track.title[:24]} is not singing here"
            T = V
        # the landing: the target song's drop or hook within four phrases, else its next phrase far enough away
        if kind == "sing":
            L = self.bar_n + 1 + int(round(hb)) + int(tune.get("out_bars", 2))   # the bed out at the hook, back `out_bars` later
        else:
            pay, what = self._payoff_in_bars(T, at, MOMENT_LOOK_PHRASES * self.change_bars)
            if pay is not None and pay >= lead - 0.05:
                L = self.bar_n + 1 + int(round(pay))
            elif tune.get("payoff_only"):
                return False, f"no drop or hook of {self.songs[T].track.title[:24]} far enough ahead (this moment lands only on one now)"
            else:
                L = self.bar_n + 1 + self._phrase_ahead(T, at, lead)
                what = "the phrase"
        at_L = at + (L - self.bar_n - 1) * bar_c
        if kind == "break":
            # the melody or vocal left alone must actually sound through the take-away (measured from its stem)
            stays = [ln for ln in ("other", "vocals") if ln in held]
            lvs = [self._stem_level_span(held[ln], ln, at_L - lead * bar_c, at_L - 8 * bar_c) for ln in stays]
            if not any(lv is not None for lv in lvs):
                return False, "the melody's level there is not measured"      # unknown is not a yes: eight bars of nothing is the risk
            if max(lv for lv in lvs if lv is not None) < 0.12:
                return False, "the melody left alone is silent there"
        if self._break is not None:                   # a BREAK running gives way: its lanes come back with the script
            for lane, d in self._break[1].items():
                if d in self.songs and not self.songs[d].leaving:
                    self.lanes[lane] = d
            self._break = None
        self._loop_after_land = None
        self._moment = {"kind": kind, "label": label, "lead": lead, "L": L, "at_L": at_L, "B": B, "V": V, "what": what or "the phrase",
                        "fired": set(), "started": False, "end_clock": None, "t_L": self._song_time_at(B, at_L), "auto": bool(auto),
                        "script": self._moment_script(kind), "who": "the DJ" if auto else "you"}
        self._moment["script_marks"] = getattr(self, "_pending_marks", [])
        tt = self.songs[T].track.title[:24]
        self._move("moment_arm", None, self._song_id(B), self._song_id(V),
                   f"{'the DJ' if auto else 'you'}: {label} armed, lands in {L - self.bar_n} bars on {tt}'s {what or 'phrase'}")
        return True, f"{label} in {L - self.bar_n} bars, on {tt}'s {what or 'phrase'}"

    def moment_go(self):
        """The armed moment starts on the next bar (its landing moves to the next bar plus its lead)."""
        m = self._moment
        if m is None:
            return False, "nothing armed"
        if m["started"]:
            return True, f"{m['label']} is already running"
        at = self._next_bar_clock()
        m["L"] = self.bar_n + 1 + m["lead"]
        m["at_L"] = at + m["lead"] * self._bar_clock()
        m["t_L"] = self._song_time_at(m["B"], m["at_L"])
        m["what"] = "now"
        self._move("moment_arm", None, self._song_id(m["B"]), self._song_id(m["V"]), f"{m['who']}: {m['label']} now - lands in {m['lead']} bars")
        return True, f"{m['label']} now: lands in {m['lead']} bars"

    def moment_cancel(self):
        """Disarm; a running script is undone on the next bar (every lane back at its level, loops and
        filters off, no boost)."""
        m = self._moment
        if m is None:
            return False, "nothing armed"
        self._moment = None
        if m["started"]:
            at = self._next_bar_clock()
            ev = []
            for lane, d in self.lanes.items():
                if d is not None and d in self.songs and not self.songs[d].leaving:
                    self.lane_boost.pop(lane, None)
                    ev.append(self._m_gain(d, lane, at, 1.0))
            ev += [{"at": at, "cmd": "clear_loop", "deck": m["B"]}, {"at": at, "cmd": "filter", "deck": m["B"], "mode": "off"}]
            self.submix.post_many(ev)
            self._hygiene(at + int(self._beat_s() * RATE))
            self.phrase_bars = 0
        self._move("moment_cancel", None, self._song_id(m["B"]), None, f"{m['label']} cancelled" + (" - everything back on the bar" if m["started"] else ""))
        return True, f"{m['label']} cancelled"

    def moment_status(self):
        """The armed or running moment for the readout and the picture: what, whose, bars to the one, and the
        choreography as (offset, end offset, lanes, words) from the one."""
        m = self._moment
        if m is None:
            return None
        marks = [{"off": off, "end_off": end, "lanes": lanes, "text": text} for off, end, lanes, text in m.get("script_marks") or []]
        return {"kind": m["kind"], "label": m["label"], "in_bars": m["L"] - self.bar_n, "started": m["started"], "what": m["what"],
                "who": m["who"], "auto": m["auto"], "bed": self.songs[m["B"]].track.title if m["B"] in self.songs else None,
                "other": (self.songs[m["V"]].track.title if m["V"] in self.songs else None) if m["V"] else None,
                "marks": marks, "desc": MOMENTS[m["kind"]][2]}

    def _m_gain(self, deck, lane, at, ramp_beats, mult=1.0):
        return {"at": at, "cmd": "stem_gains", "deck": deck, "gains": {lane: self._lane_gain(deck, lane, at) * mult}, "ramp_s": ramp_beats * self._beat_s()}

    def _m_mute(self, deck, lane, at, ramp_beats):
        return {"at": at, "cmd": "stem_gains", "deck": deck, "gains": {lane: 0.0}, "ramp_s": ramp_beats * self._beat_s()}

    def _m_held(self):
        """(lane, deck) pairs sounding by the lane map."""
        out = []
        for lane, d in self.lanes.items():
            if d is None or d not in self.songs or self.songs[d].leaving:
                continue
            out.append((lane, d))
        return out

    def _m_hole(self, at, beats, keep_vocal=False):
        """Everything out at clock `at` for `beats` beats (a hard mute, 0.1 beat): the micro-vacuum before the
        one. `keep_vocal`: the vocal carries through (a bridge's thread). No beats = no hole (the ear's word)."""
        if beats <= 0:
            return
        ev = []
        for lane, d in self._m_held():
            if keep_vocal and lane == "vocals":
                continue
            ev.append(self._m_mute(d, lane, at, 0.1))
        if ev:
            self.submix.post_many(ev)
        self._note(f"the hole: {beats:g} beat{'s' if beats != 1 else ''} of nothing")

    def _m_payoff(self, m, at, lanes_to=None, boost_lanes=("drums", "bass")):
        """THE ONE: every held lane back at its level, bass and drums MOMENT_BOOST louder for a phrase; loops
        cleared and the bed put back where it would have been; the sweep off. `lanes_to`: {lane: deck} moves
        applied first (a two-song payoff)."""
        for lane, d in (lanes_to or {}).items():
            self.lanes[lane] = d
            s = self.songs.get(d)
            if s is not None and not s.entered:
                s.entered, s.entered_k = True, self.bar_n
            self._wake(d, at)
        boost = float(self.moment_tune.get(m["kind"], {}).get("boost", MOMENT_BOOST))
        for lane in boost_lanes:
            if self.lanes.get(lane) is not None:
                self.lane_boost[lane] = (boost, self.bar_n + self.change_bars)
        ev = [{"at": at, "cmd": "clear_loop", "deck": m["B"]}, {"at": at, "cmd": "filter", "deck": m["B"], "mode": "off"}]
        if m.get("looped"):
            ev.append({"at": at, "cmd": "jump", "deck": m["B"], "time_s": float(m["t_L"])})
        for lane, d in self._m_held():
            if lane == "vocals" and not self._vocal_ok(d, at):
                continue
            ev.append(self._m_gain(d, lane, at, 0.05))
        # the previous holders of moved lanes go quiet on the one
        for lane, d in (lanes_to or {}).items():
            for od, s in self.songs.items():
                if od != d and not s.leaving and s.staged_at is not None:
                    ev.append(self._m_mute(od, lane, at, 0.25))
        self.submix.post_many(ev)
        self._hygiene(at + int(self._beat_s() * RATE))
        m["end_clock"] = at + self._bar_clock()

    def _m_roles(self, V, B):
        """After a two-song payoff the other song is the bed; the old bed is a voice on its way out."""
        v, b = self.songs.get(V), self.songs.get(B)
        if v is not None:
            v.bed_since, v.was_bed, v.voice_since = self.bar_n, False, None
        if b is not None:
            b.was_bed, b.voice_since, b.bed_since = True, self.bar_n, None
        self._land_k, self._land_at, self._drums_follow = None, None, None

    def _moment_script(self, kind):
        """The script: [(offset in bars from the one, fn(at_clock_of_that_bar))]; and the marks for the picture."""
        beat_c = int(self._beat_s() * RATE)
        tune = self.moment_tune.get(kind, {})
        hb = float(tune.get("hole_beats", 1.0))
        out_bars = int(tune.get("out_bars", 2))

        def hole_at(at):
            """The hole sits on the LAST `hb` beats of the bar starting at `at`."""
            return at + int((4.0 - hb) * beat_c)

        def M():
            return self._moment

        def out_bed(at, lanes, ramp_beats, text):
            mm = M()
            ev = [self._m_mute(mm["B"], ln, at, ramp_beats) for ln in lanes if self.lanes.get(ln) == mm["B"]]
            if ev:
                self.submix.post_many(ev)
            self._move("moment", None, self._song_id(mm["B"]), None, f"{mm['label']}: {text}")

        def payoff_one(at):
            mm = M()
            self._m_payoff(mm, at)
            self._move("moment", None, self._song_id(mm["B"]), None, f"{mm['label']}: on the one - everything back, bass and drums loud")

        if kind == "slam":
            script = [(-out_bars, lambda at: out_bed(at, ("drums", "bass"), 1.0, f"bass and drums out for {out_bars} bar{'s' if out_bars != 1 else ''}")),
                      (-1, lambda at: self._m_hole(hole_at(at), hb)),
                      (0, payoff_one)]
            marks = [(-out_bars, 0, ["drums", "bass"], "SLAM: bass + drums out")] + ([(-1, 0, list(STEMS), "hole")] if hb > 0 else []) \
                + [(0, None, list(STEMS), "SLAM on the one")]
        elif kind == "sing":
            script = [(-out_bars, lambda at: out_bed(at, ("drums", "bass"), 1.0, "the hook - bass and drums out, the room sings")),
                      (0, payoff_one)]
            marks = [(-out_bars, 0, ["drums", "bass"], "SING: the vocal alone"), (0, None, list(STEMS), "back, loud")]
        elif kind == "break":
            take = int(tune.get("take_bars", 8))
            build_low = float(tune.get("build_low", LOW_CARVE))
            def build(at):
                mm = M()
                B = mm["B"]
                ev = []
                if self.lanes.get("drums") == B:
                    ev.append(self._m_gain(B, "drums", at, 1.0))
                if MOMENT_SWEEP:
                    ev += [{"at": at, "cmd": "filter", "deck": B, "mode": "hp", "cutoff_hz": 50.0, "ramp_s": 0.0, "q": 0.9},
                           {"at": at + int(0.05 * RATE), "cmd": "filter", "deck": B, "cutoff_hz": 420.0, "ramp_s": 7.5 * 4 * self._beat_s()}]
                else:
                    # the drums come back THIN (the deck's low shelf at the carve) and their lows return over the
                    # build - the kick fattens toward the one; the bass stem itself waits for the one
                    ev += [{"at": at, "cmd": "eq", "deck": B, "low": build_low, "ramp_s": 0.01},
                           {"at": at + int(0.05 * RATE), "cmd": "eq", "deck": B, "low": 1.0, "ramp_s": 7.5 * 4 * self._beat_s()}]
                self.submix.post_many(ev)
                self._move("moment", None, self._song_id(B), None, f"BREAK: the build - drums back{', the lows sweep away' if MOMENT_SWEEP else ' thin, their lows returning'}")

            def tighten(at):
                mm = M()
                B = mm["B"]
                s = self.songs.get(B)
                bar_c = self._bar_clock()
                at_L = at + 4 * bar_c
                mm["at_L"] = at_L
                if MOMENT_TIGHTEN and s is not None and s.loop is None and not self.user_loop_bars:
                    t0 = self._song_time_at(B, at)
                    start = s.track.nearest_downbeat(t0)
                    bar = self._bar_s(s.track, start)
                    mm["looped"] = True
                    self.submix.post_many([
                        {"at": at, "cmd": "loop", "deck": B, "start_s": start, "end_s": start + 2 * bar},
                        {"at": at + 2 * bar_c, "cmd": "loop", "deck": B, "start_s": start, "end_s": start + bar},
                        {"at": at + 3 * bar_c, "cmd": "loop", "deck": B, "start_s": start, "end_s": start + 0.5 * bar},
                    ])
                    self._move("moment", None, self._song_id(B), None, "BREAK: the drums tighten - two bars, one, half")
                # the last beats are the hole, the one is posted now (a looping master stalls the bar count)
                self._m_hole(at_L - int(hb * beat_c), hb, keep_vocal=True)
                mm["fired"].update({-1, 0})
                self._m_payoff(mm, at_L)
                self._move("moment", None, self._song_id(B), None, "BREAK: on the one - everything back, bass and drums loud")

            script = [(-(take + 8), lambda at: out_bed(at, ("drums", "bass"), 2.0, "drums and bass out, the melody stays")),
                      (-8, build), (-4, tighten),
                      (-1, lambda at: self._m_hole(hole_at(at), hb, keep_vocal=True)),
                      (0, payoff_one)]
            marks = [(-(take + 8), -8, ["drums", "bass"], "BREAK: drums + bass out"), (-8, 0, ["drums"], "build: drums back thin, lows return")]
            if MOMENT_TIGHTEN:
                marks.append((-4, 0, ["drums", "bass"], "tighten 2·1·½"))
            if hb > 0:
                marks.append((-1, 0, list(STEMS), "hole"))
            marks.append((0, None, list(STEMS), "on the one, loud"))
        elif kind == "takeover":
            def hole_bass(at):
                mm = M()
                B = mm["B"]
                if self.lanes.get("bass") == B and hb > 0:
                    self.submix.post(self._m_mute(B, "bass", hole_at(at), 0.1))
                    self._note(f"the hole: the old bassline out for {hb:g} beat{'s' if hb != 1 else ''}")

            def bass_in(at):
                mm = M()
                V = mm["V"]
                self._m_payoff(mm, at, lanes_to={"bass": V}, boost_lanes=("bass",))
                s = self.songs[V]
                if s.voice_since is None:
                    s.voice_since = self.bar_n
                self._move("moment", "bass", self._song_id(mm["B"]), self._song_id(V), f"TAKEOVER: the bassline of {s.track.title} takes over on the one, loud")
                mm["end_clock"] = None                # the drums still owe their move

            def drums_follow(at):
                mm = M()
                V, B = mm["V"], mm["B"]
                if V in self.songs and not self.songs[V].leaving:
                    self._cross("drums", V, at, beats=1.0, shape=False)
                    self._m_roles(V, B)
                    self._hygiene(at + int(self._beat_s() * RATE))
                    self._move("bed_to", "drums", self._song_id(B), self._song_id(V), f"TAKEOVER: the drums of {self.songs[V].track.title} follow - it is the bed")
                mm["end_clock"] = at + self._bar_clock()

            script = [(-1, hole_bass), (0, bass_in), (self.change_bars, drums_follow)]
            marks = ([(-1, 0, ["bass"], "hole")] if hb > 0 else []) + [(0, None, ["bass"], "TAKEOVER: new bassline, loud"), (self.change_bars, None, ["drums"], "its drums follow")]
        elif kind == "bridge":
            fade = int(tune.get("fade_bars", 4))

            def bed_out(at):
                mm = M()
                B = mm["B"]
                ev = [self._m_mute(B, ln, at, 4.0 * fade) for ln in ("drums", "bass", "other") if self.lanes.get(ln) == B]
                if ev:
                    self.submix.post_many(ev)
                self._move("moment", None, self._song_id(B), None, f"BRIDGE: the bed fades out under the vocal of {self.songs[B].track.title}")

            def drums_under(at):
                mm = M()
                V = mm["V"]
                if V in self.songs and not self.songs[V].leaving:
                    self.lanes["drums"] = V
                    s = self.songs[V]
                    if not s.entered:
                        s.entered, s.entered_k = True, self.bar_n
                    self._wake(V, at)
                    self.submix.post(self._m_gain(V, "drums", at, 1.0))
                    self._move("moment", "drums", self._song_id(mm["B"]), self._song_id(V), f"BRIDGE: the drums of {s.track.title} under the vocal")

            def slam(at):
                mm = M()
                V, B = mm["V"], mm["B"]
                to = {"drums": V, "bass": V}
                if self.lanes.get("other") in (B, None) and self._tonal_ok(V, "other", at):
                    to["other"] = V
                self._m_payoff(mm, at, lanes_to=to)
                self._m_roles(V, B)
                self._move("moment", "bass", self._song_id(B), self._song_id(V), f"BRIDGE: bass and drums of {self.songs[V].track.title} slam on the one - it is the bed")

            script = [(-(fade + 4), bed_out), (-4, drums_under),
                      (-1, lambda at: self._m_hole(hole_at(at), hb, keep_vocal=True)),
                      (0, slam)]
            marks = [(-(fade + 4), -4, ["drums", "bass", "other"], "BRIDGE: the bed fades under the vocal"), (-4, 0, ["drums"], "new drums under")] \
                + ([(-1, 0, ["drums"], "hole")] if hb > 0 else []) + [(0, None, ["drums", "bass", "other"], "slam on the one")]
        else:
            script, marks = [], []
        # the marks ride on the moment dict (set right after this returns)
        self._pending_marks = marks
        return script

    def _moment_step(self):
        """Run the armed moment: an action at bar L+off fires when the bar before it begins (posted at the
        next bar's clock), late if a bar was skipped. Returns True while a moment owns the room."""
        m = self._moment
        if m is None:
            return False
        clock = self._clock()
        if m["end_clock"] is not None and clock >= m["end_clock"]:
            self._moment_end(m)
            return False
        if m["B"] not in self.songs or self.songs[m["B"]].leaving:
            self._moment = None
            self._note(f"{m['label']} dropped: the bed left")
            return False
        if m["started"]:
            # the watchdog: a planned hole is two beats at most - the bus silent for longer than a bar and a
            # half means the take-away left nothing audible (a melody stem quieter than its map said, a
            # vocal that stopped): the moment is undone and everything comes back on the next bar
            rms = float(self._tel().get("rms") or 0.0)
            if rms < 0.004:
                if m.get("quiet_since") is None:
                    m["quiet_since"] = clock
                elif clock - m["quiet_since"] > 1.5 * self._bar_clock():
                    self.moment_cancel()
                    self._move("moment_cancel", None, self._song_id(m["B"]), None, f"{m['label']} undone: the room went silent for more than a bar")
                    return False
            else:
                m["quiet_since"] = None
        for off, fn in m["script"]:
            if off in m["fired"]:
                continue
            if self.bar_n >= m["L"] + off - 1:
                m["fired"].add(off)
                m["started"] = True
                at = self._next_bar_clock()
                m["at_L"] = at - off * self._bar_clock()
                try:
                    fn(at)
                except Exception as e:  # noqa: BLE001
                    self.last_error = f"moment {m['kind']} @{off}: {type(e).__name__}: {e}"
        return True

    def _moment_end(self, m):
        self._moment = None
        self.plan = None                              # the room changed: the next event is decided afresh
        self.phrase_bars = 0                          # a phrase of plain play before the arrangement moves again
        self._calm_until = self.bar_n + self.change_bars
        self._drop_k = self.bar_n                     # a peak: the auto break keeps its distance
        self._note(f"{m['label']} done")

    # -- the ear's words on a moment: one word moves one number ------------------------------------------------------
    def _tune_path(self):
        try:
            return os.path.join(str(self.music_root), ".dj_moment_tune.json")
        except Exception:
            return None

    def _load_moment_tune(self):
        tune = {k: dict(v) for k, v in MOMENT_TUNE_DEFAULT.items()}
        p = self._tune_path()
        try:
            if p and os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for k, v in (saved or {}).items():
                    if k in tune and isinstance(v, dict):
                        tune[k].update({kk: vv for kk, vv in v.items() if kk in tune[k]})
        except Exception:
            pass
        return tune

    def _save_moment_tune(self):
        if os.environ.get("DJ_NO_PLAY_LOG", "") == "1":
            return                                    # gates never move the user's numbers
        p = self._tune_path()
        try:
            if p:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(self.moment_tune, f, indent=1)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"moment tune store: {e}"

    def tune_moment(self, kind, word):
        """BAD on a moment, with the word: `hole` halves the hole (to none), `thin` keeps more of the bed
        (a shorter take-away for SLAM/SING, a fuller low shelf in the BREAK's build), `long` halves the
        take-away / the fade, `beat` and `place` make the moment land only on a measured drop or hook.
        Returns the sentence of what changed."""
        t = self.moment_tune.get(kind)
        if t is None or word not in MOMENT_WORDS:
            return None
        before = dict(t)
        if word == "hole" and "hole_beats" in t:
            t["hole_beats"] = 0.0 if t["hole_beats"] <= 0.5 else t["hole_beats"] / 2.0
        elif word == "thin":
            if "build_low" in t:
                t["build_low"] = min(1.0, t["build_low"] + 0.25)
            elif "out_bars" in t:
                t["out_bars"] = max(1, t["out_bars"] - 1)
            else:
                t["boost"] = min(1.6, t["boost"] + 0.1)
        elif word == "long":
            if "take_bars" in t:
                t["take_bars"] = max(2, t["take_bars"] // 2)
            elif "fade_bars" in t:
                t["fade_bars"] = max(1, t["fade_bars"] // 2)
            elif "out_bars" in t:
                t["out_bars"] = max(1, t["out_bars"] - 1)
        elif word in ("beat", "place"):
            t["payoff_only"] = True
        changed = [f"{k} {before[k]:g} → {v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k} → {v}"
                   for k, v in t.items() if before.get(k) != v]
        self._save_moment_tune()
        label = MOMENTS[kind][0]
        msg = f"{label}: {MOMENT_WORDS[word]} → " + (", ".join(changed) if changed else "nothing left to move there")
        self._note(msg)
        return msg

    def _moment_lead(self, kind):
        t = self.moment_tune.get(kind, {})
        if kind in ("slam", "sing"):
            return int(t.get("out_bars", 2))
        if kind == "break":
            return int(t.get("take_bars", 8)) + 8
        if kind == "bridge":
            return int(t.get("fade_bars", 4)) + 4
        return MOMENTS[kind][1]

    def eject(self, deck):
        """The operator takes a song off: its lanes move to the other songs (or rest), it leaves a bar later."""
        s = self.songs.get(deck)
        if s is None or s.leaving:
            return False
        if s.held(self.lanes):
            s.evicted = True
            self._evict(deck)
        else:
            self._leave_song(deck, self._next_bar_clock())
        self._move("eject", None, s.track.id, None, f"you: {s.track.title} out")
        return True

    def song_loop(self, deck, bars):
        """Loop one song `bars` bars from its next downbeat (None releases)."""
        s = self.songs.get(deck)
        if s is None or s.leaving:
            return False
        at = self._next_bar_clock()
        if bars is None:
            self.submix.post({"at": at, "cmd": "release_loop", "deck": deck})
            s.user_loop = None
            self._move("song_unloop", None, s.track.id, None, f"you: {s.track.title} loop released")
            return True
        start = s.track.nearest_downbeat(self._song_time_at(deck, at))
        bar = self._bar_s(s.track, start)
        self.submix.post({"at": at, "cmd": "loop", "deck": deck, "start_s": start, "end_s": start + int(bars) * bar})
        s.user_loop = int(bars)
        self._move("song_loop", None, s.track.id, None, f"you: {s.track.title} loops {bars} bars")
        return True

    def song_gain(self, deck, level):
        """One song's level, 0..1.5 (the deck gain; lanes keep their trims)."""
        s = self.songs.get(deck)
        if s is None:
            return False
        s.level = float(max(0.0, min(1.5, level)))
        if getattr(s, "awake", False):                # a sleeping (staged) deck keeps gain 0 until its first lane opens
            self.submix.post({"cmd": "gain", "deck": deck, "value": s.level, "ramp_s": 0.1})
        return True

    # -- the Director's hands ---------------------------------------------------------------------------------------
    def decoded(self):
        """The opener is decoded and waiting (start(hold_open=True))."""
        return "a" in self._pending

    def preload_opener(self, track, samples, stems, rms=None):
        """The Director hands over a song it already decoded: no decode wait, the opener is pending at
        once (levels and the singing map measured here unless given, as a decode would). Safe to call
        from a warming thread."""
        if rms is None:
            rms = self._stem_levels(track, stems)
        with self._lock:
            self._pending["a"] = (track, samples, stems, rms)
        self._decoding.pop("a", None)

    def drop_preload(self):
        with self._lock:
            self._pending.pop("a", None)

    def open_pending(self, at_clock, cue_s):
        """Open the held opener at submix clock `at_clock`, cued so song time `cue_s` sounds then."""
        self._cue_req["a"] = float(max(0.0, cue_s))
        self._open_at = int(at_clock)
        self._hold_open = False
        if "a" in self._pending and self.master is None:
            self._open_first("a")

    def set_cross_beats(self, beats):
        """How a lane crosses: 0.5 beat = a cut, 2 = the house crossfade, 4 = a long blend."""
        self.cross_beats = float(max(0.1, min(16.0, beats)))

    # -- the timeline player's hands ----------------------------------------------------------------------------
    def stage_for_bar(self, track_id, song_time_s, at_bar):
        """Stage a song so that its time `song_time_s` arrives exactly at clock bar `at_bar` (bars counted
        as bar_n). Decodes onto a free deck; the cue is computed when the deck actually starts."""
        t = self._track(int(track_id))
        if t is None or self.master is None:
            return False, "no such song / no clock"
        busy = {s.track.id for s in self.songs.values()} | {x.id for x in self._decoding.values()} | {p[0].id for p in self._pending.values()}
        if t.id in busy:
            return True, "already on a deck"
        rate = self.master_bpm / max(t.bpm, 1e-6)
        if not (RATE_MIN <= rate <= RATE_MAX):
            return False, f"{t.bpm:.0f} bpm is outside the tempo wall ({rate:.3f})"
        free = [d for d in DECKS if d not in self.songs and d not in self._decoding and d not in self._pending]
        if not free:
            return False, "no free deck"
        self._cue_req[free[0]] = (float(song_time_s), int(at_bar))
        self._decode(t, free[0])
        return True, free[0]

    def jump(self, deck, song_time_s, at_bar):
        """Re-cue a live song so that `song_time_s` lands on clock bar `at_bar` (a clip that starts
        elsewhere in a song already playing). Posted on the bar; the PLL re-locks from there."""
        s = self.songs.get(deck)
        if s is None or self.master is None:
            return False
        at = self._next_bar_clock() + (int(at_bar) - self.bar_n - 1) * self._bar_clock()
        if at < self._clock():
            return False
        self.submix.post({"at": at, "cmd": "cue", "deck": deck, "time_s": s.track.nearest_downbeat(float(song_time_s))})
        return True

    def deck_of(self, track_id):
        return next((d for d, s in self.songs.items() if s.track.id == track_id and not s.leaving), None)

    # -- steering ------------------------------------------------------------------------------------------------
    def set_blend(self, x):
        self.blend = float(max(0.0, min(1.0, x)))

    def set_change_bars(self, bars):
        self.change_bars = int(max(1, bars))

    def set_vocal_freedom(self, x):
        self.vocal_freedom = float(max(0.0, min(1.0, x)))

    def set_energy_lean(self, x):
        self.energy_lean = float(max(-0.4, min(0.4, x)))

    def set_hold(self, on):
        self.hold = bool(on)

    def next_move(self):
        self._force_move = True

    def nudge_move(self):
        """A dial turned: the next bar is a phrase decision (with the arrangement's waits intact) instead of
        waiting out the phrase - not a forced move."""
        self.phrase_bars = max(self.phrase_bars, self.change_bars - 1)

    def set_theme(self, name):
        from lib.dj.themes import get_theme
        if self.brain is not None:
            self.brain.set_theme(get_theme(name))

    # -- performing ------------------------------------------------------------------------------------------------
    def drop(self):
        """Every lane to the newest entered song on the next bar - the morph all at once, as a cut."""
        if self.master is None:
            return False
        live = [d for d, s in self.songs.items() if not s.leaving and s.staged_at is not None]
        cands = [d for d in live if self.songs[d].entered and self.songs[d].loop is None] or [d for d in live if self.songs[d].entered]
        waiting = [d for d in live if not self.songs[d].entered]
        if waiting:
            cands = waiting                                   # a staged song drops in whole
        if not cands:
            return False
        newest = max(cands, key=lambda d: (self.songs[d].entered_k if self.songs[d].entered_k is not None else self.bar_n + 1))
        at = self._next_bar_clock()
        if self._break is not None:
            self._break = None
        for lane in STEMS:
            # the vocal lane follows only where the song is singing; a silent vocal stem left open is
            # dead air on that lane, so it rests until the singing comes (the vocal rule brings it back)
            to = newest if (lane != "vocals" or self._vocal_ok(newest, at)) else None
            if self.lanes.get(lane) != to:
                self._cross(lane, to, at, beats=DROP_BEATS, shape=False)
        self._hygiene(at)
        self.phrase_bars = 0
        # the arrangement: the song that took every lane is the bed now, freshly settled; nothing is in flight
        self._land_k, self._land_at = None, None
        self._drop_k = self.bar_n
        for d, s in self.songs.items():
            if d == newest:
                s.bed_since, s.was_bed, s.voice_since = self.bar_n, False, None
            elif s.entered and not s.leaving:
                s.was_bed, s.voice_since, s.bed_since = True, self.bar_n, None
        self.plan = None
        self._move("drop", None, None, self._song_id(newest), f"DROP: every lane to {self.songs[newest].track.title}")
        return True

    def break_(self, bars=BREAK_BARS, kind="break"):
        """Every lane but one rests for `bars` bars, then they all come back on the bar. The lane that
        stays is the most melodic one held (vocals, else other, else bass, else drums)."""
        if self.master is None:
            return False
        if self._break is not None:
            if kind != "break":
                return False                          # an auto break never interrupts a break
            # YOUR break during an auto break: the running one ends now (its lanes come back), yours starts
            for lane, d in self._break[1].items():
                if d in self.songs and not self.songs[d].leaving:
                    self.lanes[lane] = d
            self._break = None
        held = {ln: d for ln, d in self.lanes.items() if d is not None and d in self.songs}
        if len(held) < 2:
            return False
        at = self._next_bar_clock()
        # the lane that stays must be AUDIBLE for the whole break: vocals only while that song is singing
        # (a silent vocal stem alone is dead air - one bar of it showed in the gate), else other, bass, drums
        # `other` first: a melodic stem is rarely silent for a bar; vocals only when singing at both ends
        # of the break (sparse vocals still left dead bars in the gate when kept alone)
        order = [ln for ln in ("other", "vocals", "bass", "drums") if ln in held]
        back_at = at + int(bars) * self._bar_clock()
        # ...and MEASURED: the stem kept must actually sound through the break (a voice that just arrived at
        # a sparse bar, a pad that drops out - the gate heard dead bars from a kept `other`): the loudest
        # held stem over the break's span wins, melodic ones with a small edge; nothing loud enough = no break
        best, best_lv = None, 0.0
        for ln in order:
            if ln == "vocals" and not (self._vocal_ok(held[ln], at) and self._vocal_ok(held[ln], back_at)):
                continue
            lv = self._stem_level_span(held[ln], ln, at, back_at)
            lv = lv * (1.15 if ln in ("other", "vocals") else 1.0) if lv is not None else (0.5 if ln in ("other", "bass", "drums") else 0.0)
            if lv > best_lv:
                best, best_lv = ln, lv
        if best is None or best_lv < 0.12:
            return False
        keep = best
        back = at + int(bars) * self._bar_clock()
        xf = 1.0 * self._beat_s()
        ev, restore = [], {}
        for ln, d in held.items():
            if ln == keep:
                continue
            ev.append({"at": at, "cmd": "stem_gains", "deck": d, "gains": {ln: 0.0}, "ramp_s": xf})
            ev.append({"at": back, "cmd": "stem_gains", "deck": d, "gains": {ln: self._lane_gain(d, ln, back)}, "ramp_s": 0.25 * self._beat_s()})
            restore[ln] = d
            self.lanes[ln] = None
        self.submix.post_many(ev)
        self._break = (back + int(0.1 * RATE), restore)
        self.phrase_bars = 0
        who = self.songs[held[keep]].track.title if keep in held else "?"
        self._move(kind, keep, None, self._song_id(held.get(keep)),
                   f"{'BREAK' if kind == 'break' else 'breakdown'}: {keep} of {who} alone for {bars} bars, the rest come back on the bar")
        return True

    def loop(self, bars):
        """LOOP 4 / 8: every live song loops `bars` bars from its own next downbeat, on the master's next
        bar (the whole combination holds); the same bars again, or None, releases every loop."""
        if self.master is None:
            return False
        if bars and self._land_k is not None and self.bar_n < self._land_k:
            # a bed change is landing (the strip is running, the new drums and bass are scheduled): a loop
            # now would wrap the songs under those events - the hold comes on the bar the bed lands
            self._loop_after_land = bars
            self._note(f"LOOP {bars} waits for the new bed to land ({self._land_k - self.bar_n} bars)")
            return True
        at = self._next_bar_clock()
        if bars is None or self.user_loop_bars == bars:
            for d, s in self.songs.items():
                if s.loop is None:
                    self.submix.post({"at": at, "cmd": "release_loop", "deck": d})
            self.user_loop_bars = None
            self._move("unloop", None, None, None, "loops released")
            return True
        for d, s in self.songs.items():
            if s.leaving or s.loop is not None:
                continue
            t = self._tel_deck(d)
            if not t.get("playing"):
                continue
            start = s.track.nearest_downbeat(self._song_time_at(d, at))
            bar = self._bar_s(s.track, start)
            self.submix.post({"at": at, "cmd": "loop", "deck": d, "start_s": start, "end_s": start + int(bars) * bar})
        self.user_loop_bars = int(bars)
        self._move("user_loop", None, None, None, f"LOOP {bars}: every song holds {bars} bars")
        return True

    # -- tempo journey ---------------------------------------------------------------------------------------------------
    def set_tempo_span(self, x):
        """How far the clock may travel with the arc: 0 = a fixed tempo, 0.06 = +-6 % of the opener's tempo."""
        self.tempo_span = float(max(0.0, min(TEMPO_SPAN_MAX, x)))

    def set_tempo_lean(self, x):
        """The Director's tempo dial: a fixed lean on the clock, -0.06 .. +0.06 of the opener's tempo
        (slower / faster), taken in the same half-percent steps."""
        self.tempo_lean = float(max(-TEMPO_SPAN_MAX, min(TEMPO_SPAN_MAX, x)))

    def _tempo_step(self):
        """Once a bar, one small step of the clock toward the arc's tempo, every deck re-rated together so the
        PLL never has to absorb more than its trim window; no step that would push a song past the wall."""
        lean = getattr(self, "tempo_lean", 0.0)
        if self.master is None or (self.tempo_span <= 0.0 and lean == 0.0) or self.base_bpm is None or self._tempo_k == self.bar_n:
            return
        self._tempo_k = self.bar_n
        target = self.base_bpm * (1.0 + self.tempo_span * (2.0 * self.energy_target() - 1.0) + lean)
        ratio = target / max(self.master_bpm, 1e-6)
        if abs(ratio - 1.0) < 0.0015:
            return
        new_bpm = self.master_bpm * max(1.0 - TEMPO_STEP, min(1.0 + TEMPO_STEP, ratio))
        rates = {d: new_bpm / max(s.track.bpm, 1e-6) for d, s in self.songs.items() if not s.leaving}
        if not rates or any(not (RATE_MIN <= r <= RATE_MAX) for r in rates.values()):
            return
        at = self._next_bar_clock()
        ramp = 4 * self._beat_s()
        self.submix.post_many([{"at": at, "cmd": "rate", "deck": d, "value": r, "ramp_s": ramp} for d, r in rates.items()])
        for d, r in rates.items():
            self.songs[d].rate = r
        self.master_bpm = new_bpm

    # -- snapshots ---------------------------------------------------------------------------------------------------------
    def save_snapshot(self):
        """Remember this combination: which song plays each lane (and the songs' titles for the readout)."""
        held = dict(self.lanes)
        if self._break is not None:
            held.update(self._break[1])               # a SAVE during a break remembers the combination the break interrupted
        lanes = {ln: self._song_id(d) for ln, d in held.items()}
        if not any(lanes.values()):
            return None
        titles = {self._song_id(d): self.songs[d].track.title for d in held.values() if d in self.songs}
        snap = {"t": time.strftime("%H:%M:%S"), "lanes": lanes, "titles": titles, "bar_n": self.bar_n}
        self.snapshots.append(snap)
        self._move("save", None, None, None, "SAVE: " + ", ".join(f"{ln} {titles.get(t, '-')[:18]}" for ln, t in lanes.items() if t))
        return snap

    def recall_snapshot(self, index=-1):
        """Bring a saved combination back: songs that left are decoded and staged again (a live song not in
        the snapshot gives up its lanes to make a deck free), then the lanes cross on one bar."""
        if not self.snapshots:
            return False
        snap = dict(self.snapshots[index])
        snap["since"] = self.bar_n
        self._recall = snap
        self._move("recall_armed", None, None, None, f"RECALL {snap['t']} armed")
        return True

    def _track(self, tid):
        return next((t for t in self.library if t.id == tid), None)

    def _log_play(self, t):
        """Every song the conductor brings in counts as PLAYED in the library's history (the autoDJ seeds
        its no-repeat memory from it) - the two engines share one night."""
        try:
            self.db.log_play_start(t.id, transition_style="remix", theme=getattr(self.brain.theme, "name", None))
        except Exception:
            pass

    def _recall_step(self):
        snap = self._recall
        if snap is None or self.master is None:
            return
        want = {tid for tid in snap["lanes"].values() if tid and self._track(tid) is not None}
        by_id = {s.track.id: d for d, s in self.songs.items() if not s.leaving}
        on_way = {t.id for t in self._decoding.values()} | {p[0].id for p in self._pending.values()}
        missing = [tid for tid in want if tid not in by_id and tid not in on_way]
        if self.bar_n - snap["since"] > RECALL_PHRASES * self.change_bars:
            self._recall = None
            self._move("recall_failed", None, None, None,
                       f"RECALL {snap['t']} abandoned: {len(want - set(by_id))} of its songs could not be brought back in time")
            return
        if missing or (want - set(by_id)):
            if missing:
                free = [d for d in DECKS if d not in self.songs and d not in self._decoding and d not in self._pending]
                if free:
                    for tid, d in zip(missing, free):          # every missing song at once, one per free deck
                        self._decode(self._track(tid), d)
                else:
                    # make a deck free: a live song outside the snapshot gives its lanes up now, leaves in a bar
                    extra = [d for d, s in self.songs.items() if not s.leaving and s.track.id not in want]
                    if extra:
                        victim, at = extra[0], self._next_bar_clock()
                        for ln, d in list(self.lanes.items()):
                            if d == victim:
                                tgt = [dd for dd in by_id.values() if dd != victim and self._lane_ok(dd, ln, at)]
                                self._cross(ln, self.rng.choice(tgt) if tgt else None, at, shape=False)
                        self._hygiene(at)
            return
        if any(self.songs[by_id[tid]].staged_at is None for tid in want):
            return
        at = self._next_bar_clock()
        for ln, tid in snap["lanes"].items():
            d = by_id.get(tid) if tid else None
            if ln == "vocals" and d is not None and not self._vocal_ok(d, at):
                d = None                                  # not singing there now: the lane rests until it is
            if self.lanes.get(ln) != d:
                self._cross(ln, d, at, shape=False)
        self._hygiene(at + int(XFADE_BEATS * self._beat_s() * RATE))
        self._recall = None
        self.phrase_bars = 0
        self._move("recall", None, None, None, f"RECALL {snap['t']}: " + ", ".join(
            f"{ln} {snap['titles'].get(t, '-')[:18]}" for ln, t in snap["lanes"].items() if t))

    # -- recording ---------------------------------------------------------------------------------------------------------
    def record(self, path=None):
        """Tap the submix into a WAV (the move log lands beside it as JSON when the recording stops)."""
        import os
        import queue
        import wave
        if self._rec_thread is not None:
            return self.record_path
        if path is None:
            repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_dir = os.path.join(repo, "logs")
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, time.strftime("remix_%Y%m%d_%H%M.wav"))
        self.record_path = path
        q = queue.Queue(maxsize=400)
        self._rec_stop = False

        def _writer():
            w = wave.open(path, "wb")
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(RATE)
            try:
                while not self._rec_stop or not q.empty():
                    try:
                        blk = q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    w.writeframes((np.clip(blk, -1, 1) * 32767).astype(np.int16).tobytes())
            finally:
                w.close()
        self._rec_thread = threading.Thread(target=_writer, daemon=True, name="remix-recorder")
        self._rec_thread.start()
        self.submix.record_q = q
        self._rec_from = len(self.move_log)
        self._move("record", None, None, None, f"recording to {os.path.basename(path)}")
        return path

    def record_stop(self):
        import json
        if self._rec_thread is None:
            return None
        self.submix.record_q = None
        self._rec_stop = True
        self._rec_thread.join(timeout=3.0)
        self._rec_thread = None
        path = self.record_path
        try:
            log = [{k: m.get(k) for k in ("t", "clock_s", "kind", "lane", "text", "fb", "shape")} for m in self.move_log[self._rec_from:]]
            with open(path[:-4] + ".json", "w", encoding="utf-8") as f:
                json.dump({"wav": path, "moves": log, "snapshots": self.snapshots}, f, indent=1)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"move log: {e}"
        self._move("record_stop", None, None, None, "recording stopped")
        return path

    # -- learning ------------------------------------------------------------------------------------------------------
    def _load_verdicts(self):
        """Per (kind, lane) Laplace rate from every stored remix verdict: (ups + 1) / (n + 2)."""
        self.move_w = {}
        try:
            rows = self.db.conn.execute("SELECT style, up FROM seam_feedback WHERE style LIKE 'remix:%' AND source = 'user'").fetchall()
        except Exception:
            rows = []
        tally = {}
        for style, up in rows:
            parts = str(style).split(":")
            kind = parts[1] if len(parts) > 1 else None
            lane = parts[2] if len(parts) > 2 and parts[2] else None
            for key in ((kind, lane), (kind, None)):
                n, u = tally.get(key, (0, 0))
                tally[key] = (n + 1, u + (1 if up else 0))
        self._tally = tally
        for key, (n, u) in tally.items():
            self.move_w[key] = (u + 1.0) / (n + 2.0)
        self.n_verdicts = len(rows)

    def rate_last(self, up, max_age_s=None):
        """GOOD / BAD on the last rated kind of move (an entry, a cross, a rest, a return, a drop, a break).
        Stored as seam_feedback style 'remix:<kind>:<lane>' between the songs involved; the running
        weights update at once and every later conductor starts from the record. `max_age_s`: a move older
        than this is not what the verdict is about (a BAD in a dead stretch condemned a vocal rest from two
        minutes earlier); a move made because of a BAD is not rateable until it has been heard."""
        now = time.time()
        for m in reversed(self.move_log):
            if m["kind"] in RATED_KINDS and m.get("fb") is None:
                if m.get("unrateable_until", 0.0) > now:
                    continue
                if max_age_s is not None and now - m["t"] > max_age_s:
                    return None
                break
        else:
            return None
        return self._rate(m, up)

    def rate_id(self, move_id, up):
        """GOOD / BAD (or None to clear) on one specific move, by the id the feed shows."""
        m = next((x for x in self.move_log if x.get("id") == move_id), None)
        if m is None or m["kind"] not in RATED_KINDS:
            return None
        if up is None:
            if m.get("fb") is not None:
                self._unrate(m)
            return m
        if m.get("fb") is not None:
            self._unrate(m)
        return self._rate(m, up)

    def _unrate(self, m):
        up = m.pop("fb")
        row = m.pop("fb_row", None)
        if row:
            try:
                self.db.delete_seam_feedback(row)
            except Exception:
                pass
        for key in ((m["kind"], m.get("lane")), (m["kind"], None)):
            n, u = self._tally.get(key, (0, 0))
            n, u = max(0, n - 1), max(0, u - (1 if up else 0))
            self._tally[key] = (n, u)
            self.move_w[key] = (u + 1.0) / (n + 2.0)
        self.n_verdicts = max(0, getattr(self, "n_verdicts", 0) - 1)

    def _rate(self, m, up):
        m["fb"] = bool(up)
        style = f"remix:{m['kind']}:{m.get('lane') or ''}"
        a = m.get("from_id") or m.get("to_id") or 0
        b = m.get("to_id") or m.get("from_id") or 0
        try:
            m["fb_row"] = self.db.add_seam_feedback(a, b, style, up, source="user")
        except Exception as e:  # noqa: BLE001
            self.last_error = f"verdict store: {e}"
        for key in ((m["kind"], m.get("lane")), (m["kind"], None)):
            n, u = self._tally.get(key, (0, 0))
            self._tally[key] = (n + 1, u + (1 if up else 0))
            self.move_w[key] = (u + (1 if up else 0) + 1.0) / (n + 3.0)
        self.n_verdicts = getattr(self, "n_verdicts", 0) + 1
        self._note(f"{'GOOD' if up else 'BAD'}: {m['text']}")
        return m

    # -- state ------------------------------------------------------------------------------------------------------
    def status(self):
        tel = self._tel()
        decks = tel.get("decks") or {}
        sync = tel.get("sync") or {}
        views = dict(sync.get("slaves") or ({sync["slave"]: sync} if sync else {}))
        m = decks.get(self.master) or {}
        last_rated = next((x for x in reversed(self.move_log) if x["kind"] in RATED_KINDS), None)
        out = {"master": self.master, "master_bpm": self.master_bpm, "key_centre": self.key_centre,
               "lanes": dict(self.lanes), "phrase_bars": self.phrase_bars, "change_bars": self.change_bars,
               "blend": self.blend, "vocal_freedom": self.vocal_freedom, "energy": self.energy_target(),
               "energy_lean": self.energy_lean, "arc_phase": self.arc_progress(), "hold": self.hold,
               "user_loop": self.user_loop_bars, "breaking": self._break is not None,
               "songs": {}, "moves": list(self.moves[-40:]), "error": self.last_error,
               "decoding": {d: t.title for d, t in self._decoding.items()}, "ready": sorted(self._pending),
               "clock_s": tel.get("clock_s"), "n_verdicts": getattr(self, "n_verdicts", 0),
               "tempo_span": self.tempo_span, "base_bpm": self.base_bpm, "n_snapshots": len(self.snapshots),
               "auto": self.auto,
               "arrangement": self._arrangement_status(),
               "moment": self.moment_status(),
               "moment_stats": {k: {"armed": v["armed"], "refused": dict(v["refused"])} for k, v in self.moment_stats.items()},
               "moment_tune": {k: dict(v) for k, v in self.moment_tune.items()},
               "grid": {d: {ln: self.why_not(ln, d) for ln in STEMS} for d, s in self.songs.items() if not s.leaving},
               "candidates": self.candidates(10),
               "pool": (None if self.brain is None or self.brain.pool_ids is None else len(self.brain.pool_ids)),
               "pool_name": getattr(self, "pool_name", None),
               "lane_age": {ln: (self.bar_n - self.lane_k[ln]) if ln in self.lane_k else None for ln in STEMS},
               "feed": [{"id": x.get("id"), "hms": x.get("hms"), "kind": x["kind"], "lane": x.get("lane"), "text": x["text"],
                         "fb": x.get("fb"), "shape": x.get("shape"), "rateable": x["kind"] in RATED_KINDS}
                        for x in self.move_log[-40:]],
               "recalling": self._recall is not None,
               "recording": (self.record_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] if self._rec_thread is not None and self.record_path else None),
               "last_rated": ({"text": last_rated["text"], "fb": last_rated.get("fb")} if last_rated else None),
               "move_w": {f"{k}:{ln or ''}": round(w, 2) for (k, ln), w in self.move_w.items()}}
        for d, s in self.songs.items():
            t = decks.get(d) or {}
            lock = None
            if d != self.master and t.get("playing") and m.get("playing") and t.get("beat_phase") is not None and m.get("beat_phase") is not None:
                v = views.get(d) or {}
                err = (float(t["beat_phase"]) - float(m["beat_phase"]) - float(v.get("bias_beats") or 0.0) + 0.5) % 1.0 - 0.5
                lock = abs(err) * self._beat_s() * 1000.0
            sec = s.track.section_at(float(t.get("time_s") or 0.0)) or {}
            out["songs"][d] = {"id": s.track.id, "title": s.track.title, "artist": s.track.artist, "bpm": s.track.bpm, "camelot": s.track.camelot,
                               "duration_s": s.track.duration_s, "rate": s.rate, "shift": s.shift, "compat": s.compat,
                               "playing": bool(t.get("playing")), "time_s": t.get("time_s"), "loop": t.get("loop"),
                               "lanes": s.held(self.lanes), "entered": s.entered, "leaving": s.leaving,
                               "level": s.level, "user_loop": s.user_loop, "staged": s.staged_at is not None,
                               "lock_ms": lock, "map": self._map(s.track, s), "section": sec.get("kind"),
                               "singing": self._singing(s, float(t.get("time_s") or 0.0)),
                               "eq_low": (t.get("eq") or [None])[0], "filter": t.get("filter"), "echo": t.get("echo"),
                               "trims": {ln: round(self._trim(d, ln), 2) for ln in STEMS}}
        return out

    def _arrangement_status(self):
        """The arrangement in words and numbers: who is the bed and for how long, who is a voice and what
        it waits for, what is in flight, and why the last phrase passed without a move."""
        now = time.time()
        p = self.plan
        plan = None
        if p is not None:
            ps = self.songs.get(p["deck"])
            plan = {"what": p["what"], "title": ps.track.title if ps else "?", "bar": p["bar"], "in_bars": p["bar"] - self.bar_n,
                    "why": p["why"], "payoff": p.get("payoff"), "moved": p.get("moved", 0), "words": self._plan_words(p)}
        out = {"bed": None, "voices": [], "landing_in_bars": None, "wait_why": self.wait_why, "plan": plan,
               "settle_phrases": SETTLE_PHRASES, "voice_phrases": VOICE_PHRASES, "strip": bool(self.strip_before_bed),
               "calm_phrases": self.calm_phrases,
               # the measure of "busy": lane moves in the last ten minutes (entries, crosses, rests, returns, bed changes)
               "moves_10min": sum(1 for x in self.move_log if now - x["t"] <= 600.0
                                  and x["kind"] in ("enter", "cross", "cross_new", "rest", "return", "voice_in", "bed_to", "voice_out", "manual", "moment_in", "moment_out"))}
        try:
            bar = self.bar_n
            at = self._next_bar_clock()
            bed = self._bed_deck()
            if bed is not None and bed in self.songs:
                s = self.songs[bed]
                since = bar - s.bed_since if s.bed_since is not None else None
                out["bed"] = {"deck": bed, "title": s.track.title, "since_bars": since,
                              "settle_left": (max(0, SETTLE_PHRASES * self.change_bars - since) if since is not None else 0),
                              "kind": self._kind_at(bed, at)}
            for ln in ("other", "vocals"):
                d = self.lanes.get(ln)
                if d is None or d == bed or d not in self.songs:
                    continue
                s = self.songs[d]
                heard = bar - s.voice_since if s.voice_since is not None else 0
                drop, what = self._payoff_in_bars(d, at, 4 * self.change_bars)
                out["voices"].append({"deck": d, "title": s.track.title, "lane": ln, "heard_bars": heard, "leaving": bool(s.was_bed),
                                      "may_take_bed_in": max(0, VOICE_PHRASES * self.change_bars - heard),
                                      "must_take_bed_in": max(0, VOICE_MAX_PHRASES * self.change_bars - heard),
                                      "drop_in_bars": (None if drop is None else round(drop, 1)), "payoff": what,
                                      "kind": self._kind_at(d, at), "hook": bool(s.track.hook)})
            if self._land_k is not None:
                out["landing_in_bars"] = max(0, self._land_k - bar)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
        return out

    def web_status(self):
        """The show page's picture: status without the track maps, plus the moves as text."""
        st = self.status()
        for s in st["songs"].values():
            s.pop("map", None)
            s.pop("trims", None)
        st.pop("move_w", None)
        return st

    def outstate_keys(self):
        """Published into the show's outstate each tick - the visuals' coupling, in the automixer's
        vocabulary: the arc, the energy target, the next move's ETA as the 'blend' ETA, a DROP or an
        entry as a stamped drop."""
        eta = None
        if self.master is not None and self.change_bars:
            bars_left = max(0, self.change_bars - 1 - self.phrase_bars)
            eta = bars_left * 4 * self._beat_s() + (self._next_bar_clock() - self._clock()) / RATE
        last = self.move_log[-1] if self.move_log else None
        stamp = getattr(self, "_drop_wall", None)
        if last is not None and last["kind"] in ("drop", "enter", "recall") and last.get("t") != getattr(self, "_drop_seen", None):
            self._drop_seen = last["t"]
            self._drop_wall = stamp = time.time()
        return {"dj_active": self.running, "dj_arc_phase": self.arc_progress(), "dj_arc_heat": self.energy_target(),
                "dj_energy": self.energy_target(), "dj_drop_t": stamp,
                "dj_drop_hard": bool(last is not None and last["kind"] == "drop"),
                "dj_next_drop_eta": None, "dj_moment_eta": None, "dj_moment_hole": self._break is not None,
                "dj_blend_eta": eta, "dj_swap_eta": None, "dj_style": "remix"}

    def _map(self, t, song=None):
        """A song's geography for the strips: sections, the energy curve, the analyser's entry and exit
        points, and - once the song is on a deck - each stem's half-second envelope (up to 240 points)."""
        secs = [[round(x["start_s"], 1), round(x["end_s"], 1), x["kind"], round(x.get("vocalness") or 0.0, 2)] for x in (t.sections or [])][:40]
        curve = t.row.get("energy_curve") or []
        if curve:
            idx = [int(i * (len(curve) - 1) / 23) for i in range(24)]
            curve = [round(float(curve[i]), 2) for i in idx]
        out = {"duration": round(t.duration_s, 1), "sections": secs, "energy": curve,
               "ins": [round(float(p["time_s"]), 1) for p in sorted(t.mix_ins or [], key=lambda p: -p.get("score", 0))[:3]],
               "outs": [round(float(p["time_s"]), 1) for p in sorted(t.mix_outs or [], key=lambda p: -p.get("score", 0))[:3]]}
        env = getattr(song, "env", None) if song is not None else None
        if env:
            stems = {}
            for name, arr in env.items():
                n = len(arr)
                if n == 0:
                    continue
                m = min(240, n)
                idx = np.linspace(0, n - 1, m).astype(int)
                stems[name] = [round(float(v), 2) for v in np.asarray(arr, dtype=np.float32)[idx]]
            out["stems"] = stems
        return out

    def _move(self, kind, lane, from_id, to_id, msg, shape=None):
        if shape:
            msg = f"{msg} · {shape}"
        self.moves.append((time.strftime("%H:%M:%S"), msg))
        self._move_seq = getattr(self, "_move_seq", 0) + 1
        self.move_log.append({"id": self._move_seq, "t": time.time(), "clock_s": self._tel().get("clock_s"), "kind": kind,
                              "lane": lane, "from_id": from_id, "to_id": to_id, "text": msg, "fb": None, "shape": shape,
                              "hms": time.strftime("%H:%M:%S")})
        if getattr(self, "_verdict_forced_until", 0.0) > time.time():
            self.move_log[-1]["unrateable_until"] = self._verdict_forced_until   # a BAD made this move: hear it before judging it
        if len(self.moves) > 200:
            del self.moves[:-200]
            del self.move_log[:-200]
        self._note(msg)
        try:
            from lib.dj.session import log_row
            log_row({"event": "remix_move", "kind": kind, "lane": lane, "bar": self.bar_n, "text": msg[:160]})
        except Exception:
            pass

    def _note(self, msg):
        self.log.append(msg)
        if len(self.log) > 400:
            del self.log[:-400]
