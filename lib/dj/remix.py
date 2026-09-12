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
import math
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
OTHER_DUCK = 0.8                 # the `other` lane under another song's vocal
VOCAL_MIN = 0.35                 # section vocalness for the vocal lane to cross INTO a song
VOCAL_GONE = 0.2                 # ...and below this the vocal lane leaves it
TRIM_MIN, TRIM_MAX = 0.5, 2.0    # per-stem level trim toward the first song's levels (-6 .. +6 dB)
AUTO_BREAK_CHANCE = 0.5          # a breakdown of the song holding most lanes may become a BREAK (learned)
TEMPO_STEP = 0.005               # the clock moves at most this much per bar on a tempo journey
TEMPO_SPAN_MAX = 0.06            # ...and at most this far from the opener's tempo
RECALL_PHRASES = 8               # a RECALL whose songs cannot come back within this many phrases is abandoned
VOICE_PHRASES = 2                # the arrangement policy: a voice is heard this many phrases before it MAY take the bed
VOICE_MAX_PHRASES = 5            # ...and takes it by then even if its drop never comes (a voice is not a home)
SETTLE_PHRASES = 3               # a new bed is heard for this many phrases before the next voice arrives (let it breathe)
STRIP_BARS = 4                   # the breakdown before a bed lands: the old bed's drums + bass out for this many bars
DROP_LOOK_BARS = 2               # a voice's drop counts as "now" when its groove starts within this many bars
RATED_KINDS = ("enter", "cross", "cross_new", "rest", "return", "drop", "break", "break_auto",
               "voice_in", "bed_to", "voice_out")


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
            self.brain.veto_ids |= busy
            for _ in range(6):
                cand, meta = self.brain.choose_next(ms.track, self.energy_target(), self.master_bpm)
                if cand is None:
                    return None
                rate = self.master_bpm / max(cand.bpm, 1e-6)
                runway = (cand.duration_s - self._body_start(cand)) / rate
                if RATE_MIN <= rate <= RATE_MAX and cand.id not in busy and runway >= MIN_RUNWAY_S:
                    return cand
                self.brain.veto_ids.add(cand.id)
        finally:
            self.brain.veto_ids = saved
        return None

    # -- decoding (off the conductor thread) ----------------------------------------------------
    def _decode(self, track, deck):
        if deck in self._decoding or deck in self._pending:
            return
        self._decoding[deck] = track

        def work():
            try:
                from lib.dj.features import decode_file_stereo
                from lib.dj.stems import load_stems
                samples = decode_file_stereo(self.db.abs(track.path))
                stems = load_stems(self.music_root, track.id, expected_len=len(samples))
                if not stems:
                    raise ValueError("no stems on disk")
                rms = self._stem_levels(track, stems)
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
        i0 = int(self._body_start(track) * RATE)
        out = {}
        for name, arr in stems.items():
            a = np.asarray(arr)
            n = len(a)
            i1 = min(n, i0 + 120 * RATE)
            if i1 - i0 < RATE:
                i0, i1 = 0, min(n, 120 * RATE)
            acc, cnt = 0.0, 0
            for j in range(i0, i1, 10 * RATE):
                blk = a[j:min(i1, j + 10 * RATE)].astype(np.float32)
                acc += float(np.sum(blk * blk))
                cnt += blk.size
            out[name] = math.sqrt(acc / max(cnt, 1))
        # per-stem ENVELOPES at half-second resolution (RMS per 0.5 s, each stem scaled by its own 95th
        # percentile): the strips' stem-level picture - where each stem is present, finer than sections
        env = {}
        hop = RATE // 2
        for name, arr in stems.items():
            a = np.asarray(arr)
            n = len(a) // hop
            if n <= 0:
                continue
            vals = np.empty(n, dtype=np.float32)
            for j in range(0, n, 200):
                j1 = min(n, j + 200)
                blk = a[j * hop:j1 * hop].astype(np.float32)
                blk = blk.reshape(j1 - j, hop, -1)
                vals[j:j1] = np.sqrt(np.mean(blk * blk, axis=(1, 2)))
            top = float(np.percentile(vals, 95)) or 1e-6
            env[name] = np.clip(vals / top, 0.0, 1.0).astype(np.float16)
        out["_env"] = env
        vox = stems.get("vocals")
        if vox is not None and track.sections:
            a = np.asarray(vox)
            per = []
            for s in track.sections:
                j0, j1 = int(s["start_s"] * RATE), min(len(a), int(s["end_s"] * RATE))
                if j1 - j0 < RATE // 2:
                    per.append(0.0)
                    continue
                acc, cnt = 0.0, 0
                for j in range(j0, j1, 10 * RATE):
                    blk = a[j:min(j1, j + 10 * RATE)].astype(np.float32)
                    acc += float(np.sum(blk * blk))
                    cnt += blk.size
                per.append(math.sqrt(acc / max(cnt, 1)))
            out["_vox_sections"] = per
        return out

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
        rms = dict(rms)
        song.env = rms.pop("_env", None)
        song.vox_sections = rms.pop("_vox_sections", None)
        if song.vox_sections:
            song.vocal_data = True                    # measured from the stem: every song has a singing map
        song.stem_rms = rms
        # the level trim reference: the stems of the first song, scaled by its loudness gain like every deck's
        if self.ref_rms is None:
            self.ref_rms = dict(rms)
        self.songs[deck] = song
        return song

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
        for s in (1, -1, 2, -2, 3, -3):
            c = _compat(self.key_centre, _shift(track.camelot, s))
            if c > best[1] + 1e-9:
                best = (s, c)
            if best[1] >= 0.9:
                break
        return best

    def _tonal_ok(self, deck, lane):
        """May `lane` cross to the song on `deck` next to the songs on the other tonal lanes?"""
        if lane not in TONAL:
            return True
        song = self.songs.get(deck)
        if song is None:
            return False
        for other in TONAL:
            if other == lane:
                continue
            od = self.lanes.get(other)
            if od is None or od == deck:
                continue
            o = self.songs.get(od)
            if o is None or not o.track.camelot or not song.track.camelot:
                continue
            if _compat(o.key(), song.key()) < CLASH_BELOW:
                return False
        return True

    def _singing(self, song, t):
        """Is the song singing at its own time `t`? Measured from the vocals stem per section (RMS above a
        quarter of the song's loudest singing section and above -45 dBFS); the ML vocalness when the stem
        map is missing; None when nothing is known."""
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

    def _vocal_gone(self, deck, at):
        s = self.songs.get(deck)
        if s is None:
            return False
        return self._singing(s, self._song_time_at(deck, at)) is False

    def _lane_ok(self, deck, lane, at):
        return self._tonal_ok(deck, lane) and (lane != "vocals" or self._vocal_ok(deck, at))

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
        s = self.songs.get(deck)
        if s is None or not self.ref_rms or not s.stem_rms.get(lane) or not self.ref_rms.get(lane):
            return 1.0
        return float(min(TRIM_MAX, max(TRIM_MIN, self.ref_rms[lane] / s.stem_rms[lane])))

    def _lane_gain(self, deck, lane, at=None):
        g = self._trim(deck, lane)
        if lane == "other":
            vd = self.lanes.get("vocals")
            if vd is not None and vd != deck and vd in self.songs:
                vs = self.songs[vd]
                if self._singing(vs, self._song_time_at(vd, at if at is not None else self._clock())) is not False:
                    g *= OTHER_DUCK
        return g

    def _hygiene(self, at):
        """After a move lands: lows only on the songs carrying the bass or the drums lane; the `other` lane
        under another song's vocal at OTHER_DUCK; posted at `at` with a one-beat ramp."""
        beat = self._beat_s()
        ev = []
        keep = {self.lanes.get("bass"), self.lanes.get("drums")}
        for d, s in self.songs.items():
            if s.leaving or s.staged_at is None:
                continue
            low = 1.0 if d in keep else LOW_CARVE
            ev.append({"at": at, "cmd": "eq", "deck": d, "low": low, "ramp_s": beat})
            if self.lanes.get("other") == d:
                ev.append({"at": at, "cmd": "stem_gains", "deck": d, "gains": {"other": self._lane_gain(d, "other", at)}, "ramp_s": beat})
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
        open_lanes = self._open_lanes if self._open_lanes is not None else set(STEMS)
        self.submix.post_many([
            {"at": at, "cmd": "cue", "deck": deck, "time_s": cue if cue is not None else self._body_start(t)},
            {"at": at, "cmd": "rate", "deck": deck, "value": 1.0},
            {"at": at, "cmd": "stem_gains", "deck": deck, "gains": {s: (1.0 if s in open_lanes else 0.0) for s in STEMS}, "ramp_s": 0.01},
            {"at": at, "cmd": "gain", "deck": deck, "value": 1.0, "ramp_s": 0.05},
            {"at": at, "cmd": "start", "deck": deck},
        ])
        self.lanes = {s: (deck if s in open_lanes else None) for s in STEMS}
        song.bed_since = self.bar_n
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
            {"at": at, "cmd": "gain", "deck": deck, "value": 1.0, "ramp_s": 0.05},
            {"at": at, "cmd": "start", "deck": deck},
            {"at": at, "cmd": "sync", "slave": deck, "master": self.master, "bias_beats": 0.0, "audio_pll": True},
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
                        ev.append({"at": at, "cmd": "sync", "slave": d, "master": new, "bias_beats": 0.0, "audio_pll": True})
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
        if to_deck is not None and to_deck in self.songs:
            ev.append({"at": at, "cmd": "stem_gains", "deck": to_deck, "gains": {lane: self._lane_gain(to_deck, lane, at)}, "ramp_s": xf})
            s = self.songs[to_deck]
            if not s.entered:
                s.entered, s.entered_k = True, self.bar_n
        if ev:
            self.submix.post_many(ev)
        self.lanes[lane] = to_deck
        self.lane_k[lane] = self.bar_n
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
        # a BREAK ends: the lanes are back where they were
        if self._break is not None and clock >= self._break[0]:
            for lane, d in self._break[1].items():
                if d in self.songs and not self.songs[d].leaving:
                    self.lanes[lane] = d
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
        if self.auto > 0.0 and free and not staged_waiting:
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
        self._keep_runway(self.bar_n)
        self._tempo_step()
        if self._break is not None:
            return
        self._recall_step()
        if self._recall is not None:
            return                                    # a recall in progress owns the lanes
        self._auto_break()
        # a move is PLANNED a bar early (shapes need the bar) and lands on the phrase boundary
        due = self.phrase_bars >= self.change_bars - 1
        if self._force_move:
            self._force_move = False
            self.phrase_bars = 0
            self._one_move(self._next_bar_clock())
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
        try:
            tag = self._one_move_inner(at)
        finally:
            # hygiene (the low carve, the duck) reads the lane map, which already shows a bed change that
            # LANDS later: it waits for the landing rather than thinning the bed that still plays
            t_h = at
            if self._land_at is not None and self._land_at > at:
                t_h = self._land_at
            self._land_at = None
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

    def _drop_in_bars(self, deck, at, look_bars):
        """Bars of the song on `deck` from clock `at` to its next GROOVE that follows a build, a breakdown or
        the intro (its drop, in this library's vocabulary) - 0 when the drop is right there, None when none
        comes within `look_bars` bars."""
        s = self.songs.get(deck)
        if s is None:
            return None
        t_at = self._song_time_at(deck, at)
        secs = s.track.sections or []
        bar = self._bar_s(s.track, t_at)
        for i in range(1, len(secs)):
            if secs[i].get("kind") == "groove" and secs[i - 1].get("kind") in ("build", "breakdown", "intro"):
                b = s.track.nearest_downbeat(secs[i]["start_s"])
                d = (b - t_at) / max(bar, 1e-6)
                if -DROP_LOOK_BARS <= d <= look_bars + 0.05:
                    return max(0.0, d)
        return None

    def _bed_to(self, d, at, d_bars):
        """The BED (drums + bass together) passes to the voice on `d`. With `strip_before_bed`, the old
        bed's drums and bass drop out STRIP_BARS before the landing - a breakdown of the room - and the new
        drums and bass land as a cut on the bar (the release); else the two lanes cross on the bar. When
        `d_bars` is given the landing is put on the voice's own drop."""
        s, old = self.songs[d], self._bed_deck()
        bar_c = self._bar_clock()
        land = at + int(round(d_bars)) * bar_c if d_bars else at
        strip = bool(self.strip_before_bed) and old is not None and old in self.songs and old != d
        why = "at its drop" if d_bars is not None else "after its phrases as a voice"
        tag = None
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
        # 0. housekeeping first: a vocal lane whose song stopped singing rests; a rested lane of the bed
        #    comes back after a phrase (the bed is whole again)
        vd = self.lanes.get("vocals")
        if vd is not None and vd in self.songs and self._vocal_gone(vd, at):
            tag = self._cross("vocals", None, at)
            self._move("rest", "vocals", self._song_id(vd), None, "vocals rest (the singing stopped)", shape=tag)
            return tag
        if bed is not None:
            for ln in ("other", "vocals"):
                if self.lanes.get(ln) is None and bar - self.lane_k.get(ln, -10 ** 6) >= self.change_bars and self._lane_ok(bed, ln, at):
                    tag = self._cross(ln, bed, at)
                    self._move("return", ln, None, self._song_id(bed), f"{ln} returns on {self.songs[bed].track.title} (the bed is whole again)", shape=tag)
                    return tag
        # a bed change in flight (the strip, then the landing) owns the arrangement until it lands
        if self._land_k is not None:
            if bar < self._land_k:
                self.wait_why = "the bed is changing: the breakdown, then the new drums and bass land"
                return None
            self._land_k = None
        # 1. the old bed song, now only a voice, leaves after a phrase
        for d in list(voices):
            s = self.songs[d]
            if s.voice_since is None:
                s.voice_since = bar                   # a lane came to it outside the policy (an evict, a recall, your grid)
            if getattr(s, "was_bed", False) and bar - s.voice_since >= self.change_bars:
                tag = None
                for ln in ("other", "vocals"):
                    if self.lanes.get(ln) == d:
                        to = bed if (bed is not None and self._lane_ok(bed, ln, at)) else None
                        tag = self._cross(ln, to, at) or tag
                self._move("voice_out", None, s.track.id, self._song_id(bed), f"{s.track.title} fades out as a voice; the bed carries on", shape=tag)
                return tag
        # 2. a voice that has been heard long enough takes the bed (drums + bass together, under its voice) -
        #    AT ITS OWN DROP when one comes within the window (the payoff lands the change), by
        #    VOICE_MAX_PHRASES regardless, at once when the bed is running out
        bed_s = self.songs.get(bed) if bed is not None else None
        bed_kind = self._kind_at(bed, at) if bed is not None else None
        bed_out = bed_s is not None and (bed_s.loop is not None or bed_kind == "outro")
        for d in sorted(voices, key=lambda x: getattr(self.songs[x], "voice_since", 0)):
            s = self.songs[d]
            if getattr(s, "was_bed", False):
                continue
            heard = bar - (s.voice_since if s.voice_since is not None else bar)
            if heard < VOICE_PHRASES * self.change_bars:
                continue
            if not self._lane_ok(d, "bass", at):
                if heard >= VOICE_MAX_PHRASES * self.change_bars:
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
                self.wait_why = f"{s.track.title[:28]}'s bass would clash with the room: it stays a voice"
                continue
            d_bars = self._drop_in_bars(d, at, self.change_bars)
            if d_bars is None and heard < VOICE_MAX_PHRASES * self.change_bars and not bed_out:
                self.wait_why = f"{s.track.title[:28]} waits for its drop before it takes the bed"
                continue
            return self._bed_to(d, at, d_bars)
        # 3. a staged song arrives as a voice over the bed - once the bed has SETTLED (heard as itself for
        #    SETTLE_PHRASES), never while the bed is building to its drop; a breakdown of the bed or a bed
        #    running out opens the door early
        waiting = [d for d in live if not self.songs[d].entered and (self.songs[d].ready_k is None or bar + 1 >= self.songs[d].ready_k)]
        if waiting and len(voices) < max_voices and bed_s is not None:
            since = bar - bed_s.bed_since if bed_s.bed_since is not None else 10 ** 6
            settle_left = SETTLE_PHRASES * self.change_bars - since
            if bed_kind == "build" and not bed_out:
                self.wait_why = f"{bed_s.track.title[:28]} is building to its drop: nothing arrives on top of it"
                waiting = []
            elif settle_left > 0 and not bed_out and bed_kind != "breakdown":
                self.wait_why = f"the bed settles {settle_left} more bars before the next song arrives"
                waiting = []
        if waiting and len(voices) < max_voices:
            d = waiting[0]
            s = self.songs[d]
            lane = "vocals" if (self._vocal_ok(d, at) and self.rng.random() < self.vocal_freedom + 0.3
                                and (bed is None or self._singing(self.songs[bed], self._song_time_at(bed, at)) is not True)) else "other"
            if not self._lane_ok(d, lane, at):
                lane = "other" if lane == "vocals" else None

            def _free(ln):
                # a voice lane may be taken from the bed or from a voice on its way out - never from a voice
                # that has not had its turn (a song is heard until it takes the bed or fades out)
                h = self.lanes.get(ln)
                return h is None or h == bed or h not in self.songs or self.songs[h].was_bed or self.songs[h].leaving
            if lane is not None and not _free(lane):
                alt = "other" if lane == "vocals" else "vocals"
                lane = alt if (_free(alt) and self._lane_ok(d, alt, at) and (alt != "vocals" or self._vocal_ok(d, at))) else None
                if lane is None:
                    self.wait_why = "the voice lanes are taken by a song that has not had its turn yet"
            if lane is not None and self._lane_ok(d, lane, at):
                frm = self.lanes.get(lane)
                tag = self._cross(lane, d, at)
                s.voice_since = bar
                self._move("voice_in", lane, self._song_id(frm), s.track.id, f"{s.track.title} arrives as a voice ({lane}) over the bed", shape=tag)
                return tag
        # 4. with nothing to do, an occasional breath: the bed's own voice rests a phrase (rarely)
        if bed is not None and len(entered) >= 2 and self.rng.random() < 0.08:
            for ln in ("vocals", "other"):
                if self.lanes.get(ln) == bed:
                    tag = self._cross(ln, None, at)
                    self._move("rest", ln, self._song_id(bed), None, f"the bed's {ln} rests a phrase", shape=tag)
                    return tag
        return None

    def _one_move_inner(self, at):
        if getattr(self, "policy", "arrangement") == "arrangement":
            return self._arrangement_move(at)
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
                if o.track.camelot and s.track.camelot and _compat(o.key(), s.key()) < CLASH_BELOW:
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

    def rate_last(self, up):
        """GOOD / BAD on the last rated kind of move (an entry, a cross, a rest, a return, a drop, a break).
        Stored as seam_feedback style 'remix:<kind>:<lane>' between the songs involved; the running
        weights update at once and every later conductor starts from the record."""
        for m in reversed(self.move_log):
            if m["kind"] in RATED_KINDS and m.get("fb") is None:
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
        out = {"bed": None, "voices": [], "landing_in_bars": None, "wait_why": self.wait_why,
               "settle_phrases": SETTLE_PHRASES, "voice_phrases": VOICE_PHRASES, "strip": bool(self.strip_before_bed)}
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
                drop = self._drop_in_bars(d, at, 4 * self.change_bars)
                out["voices"].append({"deck": d, "title": s.track.title, "lane": ln, "heard_bars": heard, "leaving": bool(s.was_bed),
                                      "may_take_bed_in": max(0, VOICE_PHRASES * self.change_bars - heard),
                                      "must_take_bed_in": max(0, VOICE_MAX_PHRASES * self.change_bars - heard),
                                      "drop_in_bars": (None if drop is None else round(drop, 1)), "kind": self._kind_at(d, at)})
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
        if len(self.moves) > 200:
            del self.moves[:-200]
            del self.move_log[:-200]
        self._note(msg)

    def _note(self, msg):
        self.log.append(msg)
        if len(self.log) > 400:
            del self.log[:-400]
