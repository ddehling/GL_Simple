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

from lib.dj.submix import DJSubmix, RATE

STEMS = ("drums", "bass", "other", "vocals")
TONAL = ("bass", "other", "vocals")
MORPH_ORDER = ("drums", "bass", "other", "vocals")
DECKS = ("a", "b", "c")
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
RATED_KINDS = ("enter", "cross", "cross_new", "rest", "return", "drop", "break")


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
        self.master = None               # deck name holding the clock
        self.master_bpm = None
        self.key_centre = None           # the session's key: every song is shifted toward it
        # steering
        self.blend = 0.7
        self.change_bars = 8
        self.vocal_freedom = 0.4
        self.energy_lean = 0.0
        self.hold = False
        self._force_move = False
        # performing
        self.user_loop_bars = None       # LOOP 4 / 8 in force
        self._break = None               # (restore_clock, {lane: deck}) while a BREAK runs
        # learning
        self.move_w = {}                 # (kind, lane) -> 0..1 from stored verdicts (0.5 = no opinion)
        self._load_verdicts()
        # bookkeeping
        self.running = False
        self._lock = threading.Lock()
        self._pending = {}               # deck -> (track, samples, stems) decoded, waiting to mount
        self._decoding = {}              # deck -> track
        self.phrase_bars = 0             # bars of the master since the last move
        self.bar_k = None                # the master's bar index, as last seen (wraps inside a loop)
        self.bar_n = 0                   # bars counted since the start: the monotonic clock every age is measured on
        self.start_clock = None
        self.moves = []                  # (time, text) what the conductor did
        self.move_log = []               # dicts: {t, kind, lane, from_id, to_id, text, fb}
        self.log = []
        self.last_error = None
        self._thread = None

    # -- lifecycle -----------------------------------------------------------------------
    def start(self, first_track=None, threaded=True):
        if self.brain is None:
            self.last_error = "no stem-bearing tracks in the library (run tools/dj/dj_stems.py)"
            return False
        first = first_track or self._pick_first()
        self.running = True
        self.start_clock = self.submix.clock
        self._decode(first, "a")
        if threaded:
            self._thread = threading.Thread(target=self._run, daemon=True, name="remix")
            self._thread.start()
        return True

    def stop(self, fade_s=1.0):
        self.running = False
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
        theme = self.brain.theme
        if getattr(theme, "arc", "") == "all_night":
            return min(1.0, elapsed / (6 * 3600.0))
        return (elapsed % SET_CYCLE_S) / SET_CYCLE_S

    def energy_target(self):
        """The theme's arc at this point of the set, plus the lean."""
        base = self.brain.theme.arc_target(self.arc_progress()) if self.brain else 0.6
        return max(0.05, min(0.95, base + self.energy_lean))

    def _pick_first(self):
        cands = [t for t in self.library if (t.bpm_conf or 0) >= 0.7 and t.duration_s >= 150]
        return self.rng.choice(cands) if cands else self.rng.choice(self.library)

    def _pick_next(self):
        """The brain's choice against the master song, to the theme and the arc; must run inside the
        tempo wall against the clock, have body ahead of its entry, and not already be live or on its way."""
        ms = self.songs.get(self.master)
        if ms is None:
            return None
        busy = {s.track.id for s in self.songs.values()} | {t.id for t in self._decoding.values()} | {p[0].id for p in self._pending.values()}
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
                with self._lock:
                    self._pending[deck] = (track, samples, stems)
            except Exception as e:  # noqa: BLE001
                self.last_error = f"decode {track.title}: {type(e).__name__}: {e}"
                self._note(self.last_error)
            finally:
                self._decoding.pop(deck, None)
        threading.Thread(target=work, daemon=True, name=f"remix-decode-{deck}").start()

    def _mount(self, deck):
        with self._lock:
            item = self._pending.pop(deck, None)
        if item is None:
            return None
        track, samples, stems = item
        self.submix.post_many([
            {"cmd": "unload", "deck": deck},
            {"cmd": "load", "deck": deck, "samples": samples, "grid": track.grid, "track_id": track.id,
             "gain_db": track.gain_db, "stems": stems, "cue_s": 0.0},
            {"cmd": "gain", "deck": deck, "value": 0.0, "ramp_s": 0.01},
            {"cmd": "stem_gains", "deck": deck, "gains": {s: 0.0 for s in STEMS}, "ramp_s": 0.01},
            {"cmd": "eq", "deck": deck, "low": 1.0, "mid": 1.0, "high": 1.0, "ramp_s": 0.01},
        ])
        song = Song(deck, track)
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

    def _tel(self):
        return self.submix.telemetry or {}

    def _tel_deck(self, deck):
        return (self._tel().get("decks") or {}).get(deck) or {}

    def _clock(self):
        return int(self._tel().get("clock", self.submix.clock))

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
        ms = self.songs.get(self.master)
        return self._bar_s(ms.track) / 4.0 if ms else 0.5

    def _bar_clock(self):
        return int(4 * self._beat_s() * RATE)

    def _master_bar_index(self):
        ms = self.songs.get(self.master)
        d = self._tel_deck(self.master) if self.master else {}
        if ms is None or not d.get("playing"):
            return None
        return int(float(d.get("time_s", 0.0)) / self._bar_s(ms.track))

    # -- the harmonic guard --------------------------------------------------------------------------
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

    # -- songs arriving and leaving ---------------------------------------------------------------------
    def _open_first(self, deck):
        song = self._mount(deck)
        if song is None:
            return
        t = song.track
        at = self.submix.clock + int(LEAD_S * RATE)
        self.master, self.master_bpm, self.key_centre = deck, float(t.bpm), t.camelot
        song.rate, song.staged_at, song.entered, song.entered_k = 1.0, at, True, 0
        self.submix.post_many([
            {"at": at, "cmd": "cue", "deck": deck, "time_s": self._body_start(t)},
            {"at": at, "cmd": "rate", "deck": deck, "value": 1.0},
            {"at": at, "cmd": "stem_gains", "deck": deck, "gains": {s: 1.0 for s in STEMS}, "ramp_s": 0.01},
            {"at": at, "cmd": "gain", "deck": deck, "value": 1.0, "ramp_s": 0.05},
            {"at": at, "cmd": "start", "deck": deck},
        ])
        self.lanes = {s: deck for s in STEMS}
        self.brain.note_played(t)
        self._move("open", None, None, t.id, f"{t.title} opens on every lane: the clock, {t.bpm:.1f} bpm, key {t.camelot or '?'}")

    def _stage_song(self, deck):
        """Mount a decoded song, start it beat-locked at its body with every lane closed. It plays
        silently until a move gives it a lane, so the PLL has settled by the time it is heard."""
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
        self.submix.post_many([
            {"at": at, "cmd": "cue", "deck": deck, "time_s": self._body_start(t)},
            {"at": at, "cmd": "rate", "deck": deck, "value": rate},
            {"at": at, "cmd": "pitch", "deck": deck, "semitones": float(song.shift)},
            {"at": at, "cmd": "gain", "deck": deck, "value": 1.0, "ramp_s": 0.05},
            {"at": at, "cmd": "start", "deck": deck},
            {"at": at, "cmd": "sync", "slave": deck, "master": self.master, "bias_beats": 0.0, "audio_pll": True},
        ])
        self.brain.note_played(t)
        fit = f", key fit {song.compat:.2f}" if song.compat is not None else ""
        self._note(f"{t.title} staged on {deck.upper()} (rate {rate:.3f}, shift {song.shift:+d} st{fit})")
        return True

    def _leave_song(self, deck, at):
        song = self.songs.get(deck)
        if song is None or song.leaving:
            return
        beat = self._beat_s()
        stop_at = at + int(2 * beat * RATE)
        ev = [{"at": at, "cmd": "gain", "deck": deck, "value": 0.0, "ramp_s": 1.5 * beat},
              {"at": stop_at, "cmd": "stop", "deck": deck}, {"at": stop_at, "cmd": "clear_loop", "deck": deck}]
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

    def _cross(self, lane, to_deck, at, beats=XFADE_BEATS):
        """Lane `lane` goes to `to_deck` (None = rest) at clock `at`, crossing over `beats`."""
        xf = beats * self._beat_s()
        frm = self.lanes.get(lane)
        ev = []
        if frm is not None and frm in self.songs:
            ev.append({"at": at, "cmd": "stem_gains", "deck": frm, "gains": {lane: 0.0}, "ramp_s": xf})
        if to_deck is not None and to_deck in self.songs:
            ev.append({"at": at, "cmd": "stem_gains", "deck": to_deck, "gains": {lane: 1.0}, "ramp_s": xf})
            s = self.songs[to_deck]
            if not s.entered:
                s.entered, s.entered_k = True, self.bar_n
        if ev:
            self.submix.post_many(ev)
        self.lanes[lane] = to_deck

    def _song_id(self, deck):
        s = self.songs.get(deck) if deck else None
        return s.track.id if s else None

    # -- the choreography ------------------------------------------------------------------------------------
    def step(self):
        if not self.running:
            return
        if self.master is None:
            if "a" in self._pending:
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
        # decoded songs come in silently at once; the brain's next pick decodes onto a free deck
        for d in DECKS:
            if d in self._pending and d not in self.songs:
                self._stage_song(d)
        live = [s for s in self.songs.values() if not s.leaving]
        cap = 2 if self.blend < 0.5 else 3
        free = [d for d in DECKS if d not in self.songs and d not in self._decoding and d not in self._pending]
        if free and len(live) + len(self._decoding) + len(self._pending) < cap:
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
        if self._break is not None:
            return
        due = self.phrase_bars >= self.change_bars
        if (due and not self.hold) or self._force_move:
            self._force_move = False
            self.phrase_bars = 0
            self._one_move()

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
                if not song.entered or self._break is not None:
                    continue                          # staged and waiting, or resting through a BREAK
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
                if left < RUNWAY_S or sec.get("kind") == "outro":
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
            tgt = [d for d in others if self._tonal_ok(d, lane)]
            to = self.rng.choice(tgt) if tgt else None
            self._cross(lane, to, at + i * bar)
        self._move("evict", None, song.track.id, None, f"{song.track.title} has looped long enough: its lanes move on")

    def _w(self, kind, lane=None):
        """Learned weight of a kind of move (0.5 = no verdicts yet)."""
        return self.move_w.get((kind, lane), self.move_w.get((kind, None), 0.5))

    def _one_move(self):
        at = self._next_bar_clock()
        live = [d for d, s in self.songs.items() if not s.leaving and s.staged_at is not None]
        waiting = [d for d in live if not self.songs[d].entered]
        # a staged song comes in through one lane
        if waiting:
            d = waiting[0]
            lane = self._lane_to_give(d)
            if lane is not None:
                frm = self.lanes.get(lane)
                self._cross(lane, d, at)
                s = self.songs[d]
                self._move("enter", lane, self._song_id(frm), s.track.id,
                           f"{s.track.title} enters through {lane} (rate {s.rate:.3f}, shift {s.shift:+d} st)")
                return
        entered = [d for d in live if self.songs[d].entered]
        # songs looping their last bars are on the way out: they give lanes, they do not take them
        fresh = [d for d in entered if self.songs[d].loop is None] or entered
        # a resting lane comes back first
        for lane in STEMS:
            if self.lanes.get(lane) is None:
                tgt = [d for d in fresh if self._tonal_ok(d, lane)]
                if tgt:
                    d = self.rng.choice(tgt)
                    self._cross(lane, d, at)
                    self._move("return", lane, None, self._song_id(d), f"{lane} returns on {self.songs[d].track.title}")
                    return
        if len(entered) < 2:
            return
        # the blend: a consolidating move (the newest song takes its next lane) or a free recombination;
        # the verdict record tilts the choice (2 x weight: 0.5 = as set, 1.0 = twice as often)
        p_cons = (1.0 - self.blend) * 2 * self._w("cross_new")
        if self.rng.random() < p_cons:
            newest = max(fresh, key=lambda d: self.songs[d].entered_k or 0)
            for lane in MORPH_ORDER:
                if self.lanes.get(lane) != newest and self._tonal_ok(newest, lane) and self._takeable(lane):
                    frm = self.lanes.get(lane)
                    self._cross(lane, newest, at)
                    self._move("cross_new", lane, self._song_id(frm), self._song_id(newest),
                               f"{lane} crosses to {self.songs[newest].track.title} (toward the new song)")
                    return
        lanes = list(STEMS)
        self.rng.shuffle(lanes)
        lanes.sort(key=lambda ln: -self._w("cross", ln))          # liked lanes first (stable on ties)
        for lane in lanes:
            if lane == "vocals" and self.rng.random() > min(1.0, self.vocal_freedom * 2 * self._w("cross", "vocals")):
                continue
            if not self._takeable(lane):
                continue
            cur = self.lanes.get(lane)
            targets = [d for d in fresh if d != cur and self._tonal_ok(d, lane)]
            if cur is not None and self.rng.random() < REST_CHANCE * 2 * self._w("rest", lane):
                self._cross(lane, None, at)
                self._move("rest", lane, self._song_id(cur), None, f"{lane} rests for a phrase")
                return
            if targets:
                d = self.rng.choice(targets)
                self._cross(lane, d, at)
                self._move("cross", lane, self._song_id(cur), self._song_id(d), f"{lane} crosses to {self.songs[d].track.title}")
                return

    def _lane_to_give(self, deck):
        """The lane a new song enters through: a resting lane it fits, else - in the morph order - a
        lane from the OLDEST song still holding lanes (songs arrive and push the oldest out), guarded."""
        order = sorted(MORPH_ORDER, key=lambda ln: (-self._w("enter", ln), MORPH_ORDER.index(ln)))
        for lane in order:
            if self.lanes.get(lane) is None and self._tonal_ok(deck, lane):
                return lane
        holders = {d for d in self.lanes.values() if d is not None and d in self.songs}
        for old in sorted(holders, key=lambda d: (self.songs[d].entered_k or 0)):
            for lane in order:
                if self.lanes.get(lane) == old and self._tonal_ok(deck, lane) and self._takeable(lane):
                    return lane
        for lane in order:
            if self._tonal_ok(deck, lane):
                return lane
        return None

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
            if self.lanes.get(lane) != newest:
                self._cross(lane, newest, at, beats=DROP_BEATS)
        self.phrase_bars = 0
        self._move("drop", None, None, self._song_id(newest), f"DROP: every lane to {self.songs[newest].track.title}")
        return True

    def break_(self, bars=BREAK_BARS):
        """Every lane but one rests for `bars` bars, then they all come back on the bar. The lane that
        stays is the most melodic one held (vocals, else other, else bass, else drums)."""
        if self.master is None or self._break is not None:
            return False
        held = {ln: d for ln, d in self.lanes.items() if d is not None and d in self.songs}
        if len(held) < 2:
            return False
        keep = next((ln for ln in ("vocals", "other", "bass", "drums") if ln in held), None)
        at = self._next_bar_clock()
        back = at + int(bars) * self._bar_clock()
        xf = 1.0 * self._beat_s()
        ev, restore = [], {}
        for ln, d in held.items():
            if ln == keep:
                continue
            ev.append({"at": at, "cmd": "stem_gains", "deck": d, "gains": {ln: 0.0}, "ramp_s": xf})
            ev.append({"at": back, "cmd": "stem_gains", "deck": d, "gains": {ln: 1.0}, "ramp_s": 0.25 * self._beat_s()})
            restore[ln] = d
            self.lanes[ln] = None
        self.submix.post_many(ev)
        self._break = (back + int(0.1 * RATE), restore)
        self.phrase_bars = 0
        self._move("break", keep, None, self._song_id(held.get(keep)), f"BREAK: {keep} alone for {bars} bars, the rest come back on the bar")
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
        now = self._clock()
        for d, s in self.songs.items():
            if s.leaving or s.loop is not None:
                continue
            t = self._tel_deck(d)
            if not t.get("playing"):
                continue
            time_s, rate = float(t.get("time_s", 0.0)), max(float(t.get("rate", 1.0)), 1e-6)
            start = s.track.nearest_downbeat(time_s + (at - now) / RATE * rate)
            bar = self._bar_s(s.track, start)
            self.submix.post({"at": at, "cmd": "loop", "deck": d, "start_s": start, "end_s": start + int(bars) * bar})
        self.user_loop_bars = int(bars)
        self._move("user_loop", None, None, None, f"LOOP {bars}: every song holds {bars} bars")
        return True

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
               "last_rated": ({"text": last_rated["text"], "fb": last_rated.get("fb")} if last_rated else None),
               "move_w": {f"{k}:{ln or ''}": round(w, 2) for (k, ln), w in self.move_w.items()}}
        for d, s in self.songs.items():
            t = decks.get(d) or {}
            lock = None
            if d != self.master and t.get("playing") and m.get("playing") and t.get("beat_phase") is not None and m.get("beat_phase") is not None:
                v = views.get(d) or {}
                err = (float(t["beat_phase"]) - float(m["beat_phase"]) - float(v.get("bias_beats") or 0.0) + 0.5) % 1.0 - 0.5
                lock = abs(err) * self._beat_s() * 1000.0
            out["songs"][d] = {"title": s.track.title, "artist": s.track.artist, "bpm": s.track.bpm, "camelot": s.track.camelot,
                               "duration_s": s.track.duration_s, "rate": s.rate, "shift": s.shift, "compat": s.compat,
                               "playing": bool(t.get("playing")), "time_s": t.get("time_s"), "loop": t.get("loop"),
                               "lanes": s.held(self.lanes), "entered": s.entered, "leaving": s.leaving,
                               "lock_ms": lock, "map": self._map(s.track)}
        return out

    def _map(self, t):
        secs = [[round(x["start_s"], 1), round(x["end_s"], 1), x["kind"], round(x.get("vocalness") or 0.0, 2)] for x in (t.sections or [])][:40]
        curve = t.row.get("energy_curve") or []
        if curve:
            idx = [int(i * (len(curve) - 1) / 23) for i in range(24)]
            curve = [round(float(curve[i]), 2) for i in idx]
        return {"duration": round(t.duration_s, 1), "sections": secs, "energy": curve}

    def _move(self, kind, lane, from_id, to_id, msg):
        self.moves.append((time.strftime("%H:%M:%S"), msg))
        self.move_log.append({"t": time.time(), "kind": kind, "lane": lane, "from_id": from_id, "to_id": to_id,
                              "text": msg, "fb": None})
        if len(self.moves) > 200:
            del self.moves[:-200]
            del self.move_log[:-200]
        self._note(msg)

    def _note(self, msg):
        self.log.append(msg)
        if len(self.log) > 400:
            del self.log[:-400]
