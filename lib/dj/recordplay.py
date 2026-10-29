"""THE RECORD IS THE RECORD, and the records are MIXED - docs/DJ_RESCOPE.md Phase 1 + docs/DJ_MIXING.md A and C.

One record plays WHOLE at its own tempo, untouched: the original mix, rate 1.0, deck EQ flat, no filter, no
echo, no stems. The only per-record gain is the library's loudness trim (a DJ's trim knob). It plays at least
`play_min_s`; nothing but NEXT shortens that.

The transition is a MIX, not a cut (the user, 2026-10-24: "go look up what actual mixing is. It's not just
cutting the last song off for a beat then starting another"). The next record is picked and decoded the
moment a record opens and waits on its deck. As soon as it is ready THE BLEND is planned, once, in the
record's own bars (docs/DJ_MIXING.md, section 5 A):

  K      the next record STARTS, inaudible, on a phrase boundary of the record at or after its minimum play,
         beat-matched to it (the keylock stretcher and the PLL; never varispeed - the user's B is NO), cued so
         that its landmark lands on the swap bar when its intro allows, bass KILLED, mids and highs dipped;
         its fader rises over the first half of the lead
  K+L/2  the highs trade over L/4 bars (the next record's to full, the record's down to half)
  K+3L/4 the mids trade over L/4 bars
  S=K+L  THE BASS SWAP on the phrase: the record's low band to zero, the next record's to full, two beats -
         one bassline at a time; the next record is the record from here
  E=S+T  the record's fader has come down over T bars (constant power) and it stops; the new record glides
         back to its own tempo over a few seconds

  L, T = 16, 8 bars (the 32-bar house blend, the default); 32, 16 (long: the record has the room and the next
  record's intro is 32 bars or more); 8, 4 (short: NEXT, or the record has little room). At 116 bpm a bar is
  2.07 s: the standard blend is 50 s of two records together.

THE BREAKDOWN SWAP (section 5 C) when the record offers it: a breakdown of 8+ bars at or after its minimum
play, and the next record has a build into its landmark. The next record starts 4 bars before the breakdown,
bass killed, full fader by the breakdown; the record's mids dip to make room; on the bar where the record's
breakdown resolves (its own drop, or the breakdown's end) the next record's bass comes in and the record goes
out in one beat: the next record's drop replaces the record's.

THE CUT survives only as the fallback: the record runs out before any blend fits, or the tempo gap is beyond
the stretcher (8 %). Every event is posted to the submix when the plan is decided, tagged with a transaction,
and cancelled and re-posted when it moves (NEXT: the short blend from the next bar + 2, or the swap brought
forward when the blend is under way; STAY: the whole blend a phrase later while it has not started) - the plan
shown is the plan played. No dice, no learning. Deterministic given the same picks. Nothing here opens a
device: the caller mounts `player.submix` on an AudioEngine (the Director) or pulls the mixer by hand (the
gate). Deck c is unused.
"""
import math
import threading
import time

from lib.dj.submix import DJSubmix, RATE

DECKS = ("a", "b", "c")
PHRASE_BARS = 8
LEAD_S = 0.3                      # an event is posted at least this far ahead of the render head
CUT_MIN_BARS = 2                  # a planned start is at least this many bars ahead
NEXT_BARS = 2                     # NEXT: the blend (or the swap) at the next bar plus this many
STOP_AFTER_S = 2.0                # the old deck stops this long after it is silent
LANDMARK_LOOK_S = 90.0            # the drop counts as the landmark when it comes within this of the body
RUNWAY_SLACK_S = 20.0             # a record must have play_min + this much body from its landmark
SIZES = {"short": (8, 4), "standard": (16, 8), "long": (32, 16)}   # (lead L, tail T) in bars
GLIDE_S = 8.0                     # the new record glides back to its own tempo over this after the old one stops
MAX_TEMPO_GAP = 0.08              # beyond this rate ratio the stretcher cannot match: the cut
BREAK_MIN_BARS = 8                # a breakdown the swap can use
# PLACEMENT BY THE RECORDS (2026-10-26, the user: vocals over vocals and two records at full were "very noticeable"):
VOX_QUIET = 0.15                  # a vocal stem under this much of its own loud level, over a window, is quiet (SING_ON in the conductor is 0.12)
LAST_FROM = 0.6                   # a blend starts no earlier than this fraction of the record (mix out of its last third, never its peak)
UNDER_GAIN = 0.6                  # the next record sits under the record at this fader until the bass swap ("cut, don't boost")
TAIL_BARS = 4                     # the record comes down over this after the swap...
TAIL_OUTRO_BARS = 8               # ...or this when the swap is in its outro
KEY_BLEND = 0.68                  # two records overlap for phrases only when their keys fit this well (the conductor's ENTRY_FIT); under it, the cut
KEY_LONG = 0.90                   # ...and only a near-perfect fit earns more than the short blend (Inevitable Ending -> Dunes at 0.55 got 36 bars, 2026-10-26)
ALLOWED_WHY =("still decoding", "nothing planned yet", "is the record playing", "not on a deck", "no such song", "press START")


def _hms():
    return time.strftime("%H:%M:%S")


def _mmss(s):
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"


VOX_FLOOR = 0.06                  # a vocal stem whose loud level is under this much of the mix's level (-24 dB) is bleed, not singing


def vocal_envelope(music_root, track_id, mix_rms=None):
    """The vocal stem's level per half second, scaled to its 95th percentile (1.0 = its loud level), float16 - or
    None without stems. Decoded in the helper process like the mix (one file, ~1 s); the stems themselves are not
    kept: the player plays the original mix, the envelope only says WHERE the record sings. `mix_rms`: the mix's
    own level; a stem whose loud level is under VOX_FLOOR of it is an instrumental's bleed (the breakdown smoke's
    pair was refused as "singing" over its instrumental intro, 2026-10-26) and reads as silent everywhere."""
    import numpy as np
    from lib.dj.stems import stem_paths
    paths = stem_paths(music_root, track_id)
    if not paths or "vocals" not in paths:
        return None
    from lib.dj.stemload import decode_mix
    v = decode_mix(paths["vocals"])
    hop = RATE // 2
    n = len(v) // hop
    if n < 2:
        return None
    e = np.empty(n, dtype=np.float32)
    step = 240
    for j in range(0, n, step):
        j1 = min(n, j + step)
        blk = np.asarray(v[j * hop:j1 * hop], dtype=np.float32).reshape(j1 - j, hop, -1)
        e[j:j1] = np.sqrt((blk * blk).mean(axis=(1, 2)))
    top = float(np.percentile(e, 95)) or 1e-6
    if mix_rms is not None and top < VOX_FLOOR * float(mix_rms):
        return np.zeros(n, dtype=np.float16)
    return np.clip(e / top, 0.0, 4.0).astype(np.float16)


def mix_level(samples):
    """The mix's RMS over its middle half, from every 50th frame (cheap; the reference for VOX_FLOOR)."""
    import numpy as np
    n = len(samples)
    if n < RATE:
        return 0.0
    x = np.asarray(samples[n // 4:3 * n // 4:50], dtype=np.float32)
    return float(np.sqrt((x * x).mean())) if x.size else 0.0


def mix_envelope(samples, gain_db=0.0):
    """The mix's RMS per half second (absolute, with the library's loudness trim applied), float16: the record's OWN
    quiet passages, so a breakdown at -50 dBFS is not logged as a DROPOUT (Ruhi's, twice, 2026-10-26)."""
    import numpy as np
    hop = RATE // 2
    n = len(samples) // hop
    if n < 2:
        return None
    e = np.empty(n, dtype=np.float32)
    step = 240
    for j in range(0, n, step):
        j1 = min(n, j + step)
        blk = np.asarray(samples[j * hop:j1 * hop], dtype=np.float32).reshape(j1 - j, hop, -1)
        e[j:j1] = np.sqrt((blk * blk).mean(axis=(1, 2)))
    return (e * (10.0 ** (float(gain_db or 0.0) / 20.0))).astype(np.float16)


class Rec:
    def __init__(self, deck, track):
        self.deck = deck
        self.track = track
        self.state = "decoding"       # decoding | ready | playing | leaving
        self.cue_t = 0.0              # where it starts (the opener's first downbeat; a blend sets it)
        self.open_clock = None        # submix clock when it started sounding
        self.open_bar = None          # its bar index (by its grid) at cue_t: phrases count from here
        self.next_pressed = False     # NEXT was pressed while it was the record (the gate's min-play exemption)
        self.rate = 1.0               # the rate it was started at (beat-matched to the record before it)
        self.vox = None               # the VOCAL stem's envelope, one value per half second, 1.0 = its loud level (measured at decode; nothing is played from stems)
        self.mix_env = None           # the mix's own RMS per half second (its quiet passages are not dropouts)


class RecordPlayer:
    def __init__(self, db, music_root, library, theme="groove", seed=None):
        from lib.dj.brain import Brain
        from lib.dj.themes import get_theme
        self.db, self.music_root = db, music_root
        self.library = [t for t in library if t.bpm and t.grid and not getattr(t, "excluded", False)]
        self.brain = Brain(list(self.library), get_theme(theme or "groove"), seed=seed) if self.library else None
        self.submix = DJSubmix(deck_names=DECKS)
        self.play_min_s = 180.0
        self.hole_beats = 1.0
        self.pick_hook = None
        self.avoid_ids = set()
        self.running = False
        self.record = None            # deck name of the record playing
        self.songs = {}               # deck -> Rec
        self.cut = None               # THE PLAN: see _plan_blend / _plan_cut ("kind": blend | breakdown | cut)
        self.moves = []               # {t, kind, text, from_id, to_id}
        self.wait_why = "press START"
        self.last_error = None
        self._lock = threading.Lock()
        self._pending = {}            # deck -> (track, samples) decoded, waiting to mount
        self._decoding = {}           # deck -> track
        self._seq = 0
        self._next_asap = False
        self._first = None
        self._thread = None
        self._open_pending = None
        self._settled_clock = 0       # the record is back at its own tempo from this clock (plans wait for it)
        self._unfit = set()           # track ids found unmixable with the record playing (vocals / keys / tempo): picked again
        self._repicks = 0             # how often the next record was picked again for this record (two at most)
        self._user_ids = set()        # track ids the user asked for (UP NEXT): never picked again, the relaxed blend applies

    # -- lifecycle -----------------------------------------------------------------------------------------------------
    def prepare(self, first_track=None):
        """READY BEFORE START: the decode helper warmed and the opener decoded onto deck A, so START plays within a
        bar (the Director's tab calls this while the window sits open). Returns the opener, or None."""
        try:
            from lib.dj.stemload import warm, decode_mix
            warm()
        except Exception:
            return None
        first = first_track or self._pick(None)
        if first is None:
            return None
        try:
            samples = decode_mix(self.db.abs(first.path))
        except Exception as e:  # noqa: BLE001
            self.last_error = f"decode {first.title}: {type(e).__name__}: {e}"
            return None
        try:
            vox = vocal_envelope(self.music_root, first.id, mix_rms=mix_level(samples))
        except Exception:
            vox = None
        try:
            menv = mix_envelope(samples, first.gain_db)
        except Exception:
            menv = None
        with self._lock:
            self._pending["a"] = (first, samples, vox, menv)
        self.songs["a"] = Rec("a", first)
        self._first = first
        self._move("ready", f"ready before START: {first.title[:30]} is decoded on deck A", to_id=first.id)
        return first

    def start(self, first_track=None, threaded=False):
        try:
            from lib.dj.stemload import warm
            warm()
        except Exception:
            pass
        prepared = self._first if ("a" in self._pending and self._first is not None and (first_track is None or first_track.id == self._first.id)) else None
        if prepared is None and "a" in self._pending:
            with self._lock:
                self._pending.pop("a", None)          # (prepared for another opener: decoded afresh)
            self.songs.pop("a", None)
        first = prepared or first_track or self._pick(None)
        if first is None:
            self.last_error = "no first record: the library is empty or all of it is avoided"
            return False
        self._first = first
        self.running = True
        self.wait_why = "the first record is decoding"
        if prepared is None:
            self._decode(first, "a", opener=True)
        if threaded:
            self._thread = threading.Thread(target=self._run, daemon=True, name="recordplay")
            self._thread.start()
        return True

    def stop(self, fade_s=1.0):
        self.running = False
        try:
            self.submix.post({"cmd": "mix_gain", "value": 0.0, "ramp_s": max(0.05, float(fade_s))})
        except Exception:
            pass

    def _run(self):
        while self.running:
            try:
                self.step()
            except Exception as e:  # noqa: BLE001
                self.last_error = f"{type(e).__name__}: {e}"
            time.sleep(0.2)

    # -- the clock of the record ------------------------------------------------------------------------------------------
    def _tel(self):
        return self.submix.telemetry or {}

    def _tel_deck(self, deck):
        return (self._tel().get("decks") or {}).get(deck) or {}

    def _clock(self):
        tel = self._tel()
        return int(tel.get("clock", self.submix.clock))

    @staticmethod
    def _seg(track, t):
        for seg in track.grid or []:
            if seg["start_s"] <= t <= seg["end_s"]:
                return seg
        return (track.grid or [None])[0]

    def bar_s(self, track=None, t=0.0):
        track = track or (self.songs[self.record].track if self.record in self.songs else None)
        if track is None:
            return 2.0
        seg = self._seg(track, t)
        return 4.0 * (seg["period_s"] if seg else track.period_s)

    def _first_down(self, track, t=0.0):
        seg = self._seg(track, t)
        if seg is None:
            return 0.0
        return float(seg["first_beat_s"]) + float(track.downbeat_offset or 0) * float(seg["period_s"])

    def _bar_index(self, track, t, t_ref=None):
        """The bar index of song time `t`, counted on the grid segment at `t_ref` (default: at `t`). Every bar
        computation about one plan uses ONE segment - the one the record is in now - so an index and the time it is
        turned back into agree (a song with two tempo segments read a cut 41 s ahead as 131 bars, 2026-10-24)."""
        t_ref = t if t_ref is None else t_ref
        return int(math.floor((t - self._first_down(track, t_ref)) / self.bar_s(track, t_ref) + 1e-6))

    def _bar_time(self, track, k, t_ref=0.0):
        return self._first_down(track, t_ref) + k * self.bar_s(track, t_ref)

    def _song_time_now(self, deck):
        return float(self._tel_deck(deck).get("time_s") or 0.0)

    def _rate_now(self, deck):
        r = self._tel_deck(deck).get("rate")
        try:
            r = float(r) if r is not None else 1.0
        except (TypeError, ValueError):
            r = 1.0
        return r if r > 0.5 else 1.0

    def _clock_of_song_time(self, deck, t):
        """The submix clock at which `deck` reaches its song time `t` (at the rate it plays now)."""
        return self._clock() + int(round((t - self._song_time_now(deck)) / self._rate_now(deck) * RATE))

    @property
    def bar_n(self):
        r = self.songs.get(self.record) if self.record else None
        if r is None or r.open_bar is None:
            return 0
        t_now = self._song_time_now(r.deck)
        return max(0, self._bar_index(r.track, t_now) - self._bar_index(r.track, r.cue_t, t_ref=t_now))

    # -- where a record is heard first ------------------------------------------------------------------------------------
    @staticmethod
    def body_start(track):
        secs = track.sections or []
        cands = [s for s in secs if s.get("kind") == "groove" and s.get("bass_share", 0.3) >= 0.28]
        cands += [s for s in secs if s.get("kind") not in ("intro", "outro") and s not in cands]
        if cands:
            return max(0.0, track.nearest_downbeat(float(cands[0]["start_s"])))
        return max(0.0, track.nearest_downbeat(0.0))

    def landmark(self, track):
        """The groove after the first build (the drop) within LANDMARK_LOOK_S of the body, when it leaves the
        record its minimum play; else the body; else the first downbeat."""
        body = self.body_start(track)
        secs = track.sections or []
        need = self.play_min_s + RUNWAY_SLACK_S
        for i in range(1, len(secs)):
            if secs[i - 1].get("kind") == "build" and secs[i].get("kind") == "groove" \
                    and body <= float(secs[i]["start_s"]) <= body + LANDMARK_LOOK_S \
                    and float(track.duration_s) - float(secs[i]["start_s"]) >= need:
                return max(0.0, track.nearest_downbeat(float(secs[i]["start_s"])))
        if float(track.duration_s) - body >= need:
            return body
        return max(0.0, track.nearest_downbeat(0.0))

    def landmark_has_build(self, track):
        """True when the section before the landmark is a build (the breakdown swap wants a drop to land)."""
        lm = self.landmark(track)
        secs = track.sections or []
        for i in range(1, len(secs)):
            if abs(float(secs[i]["start_s"]) - lm) < 0.5 * self.bar_s(track, lm) and secs[i - 1].get("kind") == "build":
                return True
        return False

    def fits_as_record(self, track):
        """None when the track can be a record (long enough from its landmark), else the reason."""
        if not track.bpm or not track.grid:
            return "no beat grid"
        if float(track.duration_s or 0.0) - self.landmark(track) < self.play_min_s + RUNWAY_SLACK_S:
            return "too short for a record"
        return None

    # -- picking and decoding --------------------------------------------------------------------------------------------
    def _busy_ids(self):
        ids = {r.track.id for r in self.songs.values()} | {t.id for t in self._decoding.values()} | {p[0].id for p in self._pending.values()}
        return ids

    def _pick(self, current):
        busy = self._busy_ids()
        hook = self.pick_hook
        if hook is not None:
            try:
                t = hook()
            except Exception as e:  # noqa: BLE001
                self.last_error = f"pick hook: {type(e).__name__}: {e}"
                t = None
            if t is not None and t.id not in busy:
                why = self.fits_as_record(t)
                if why is None:
                    self._user_ids.add(t.id)
                    return t
                self._move("pick", f"{t.title[:30]} cannot be the record ({why}): the DJ picks instead", to_id=t.id)
        if self.brain is None:
            return None
        self.brain.veto_ids = set(self.avoid_ids) | busy | set(self._unfit) | {t.id for t in self.library if self.fits_as_record(t) is not None}
        if current is None:
            cands = [t for t in self.library if t.id not in self.brain.veto_ids]
            if not cands:
                return None
            # the opener: the brain's own first pick when it offers one, else the most central tempo
            try:
                pick = self.brain.pick_first() if hasattr(self.brain, "pick_first") else None
            except Exception:
                pick = None
            if pick is not None and pick.id not in self.brain.veto_ids:
                return pick
            lo, hi = self.brain.theme.bpm_range
            mid = 0.5 * (lo + hi)
            return min(cands, key=lambda t: abs(float(t.bpm) - mid))
        # THE PICK MUST BE MIXABLE (2026-10-26, "just had a mix hard cut?": the brain picked 4A after 7B and 7B after 4A,
        # fit 0.30 twice, and the key gate rightly refused to blend them - so the room got two cuts in a row). A record
        # whose key clashes with the one playing is vetoed at the pick while any mixable record is left; the cut is the
        # fallback for a plan that cannot be placed, not a way of life
        base_veto = set(self.brain.veto_ids)
        cur_bpm = float(current.bpm) * (self._rate_now(self.record) if self.record else 1.0)
        clash = {t.id for t in self.library if t.id not in base_veto
                 and ((self.key_fit(current, t) or 1.0) < KEY_BLEND or abs(cur_bpm / max(float(t.bpm), 1e-6) - 1.0) > MAX_TEMPO_GAP)}
        pick = None
        for veto in (base_veto | clash, base_veto):
            self.brain.veto_ids = veto
            try:
                pick, _meta = self.brain.choose_next(current, 0.6, float(current.bpm))
            except Exception as e:  # noqa: BLE001
                self.last_error = f"pick: {type(e).__name__}: {e}"
                pick = None
            if pick is not None:
                if veto is not base_veto or not clash:
                    break
                self._move("pick", f"no record in a key that mixes with {current.title[:24]} ({current.camelot or '?'}) is left tonight: {pick.title[:24]} ({pick.camelot or '?'}) will be a cut", to_id=pick.id)
                break
        self.brain.veto_ids = base_veto
        return pick

    def _decode(self, track, deck, opener=False):
        if deck in self._decoding or deck in self._pending:
            return
        self._decoding[deck] = track
        self.songs[deck] = Rec(deck, track)
        self._move("decoding", f"{track.title[:30]} is decoding onto deck {deck.upper()}" + (" (the opener)" if opener else ""), to_id=track.id)

        def work():
            try:
                from lib.dj.stemload import decode_mix
                samples = decode_mix(self.db.abs(track.path))
                try:
                    vox = vocal_envelope(self.music_root, track.id, mix_rms=mix_level(samples))
                except Exception as e:  # noqa: BLE001
                    vox = None
                    self._move("error", f"vocal envelope of {track.title[:24]}: {type(e).__name__}: {e} (the blend will not know where it sings)", to_id=track.id)
                try:
                    menv = mix_envelope(samples, track.gain_db)
                except Exception:
                    menv = None
                # THE PICK MUST BLEND (the user, 2026-10-26: never a cut the DJ decided): with its vocal envelope known, a next
                # record that cannot be placed under the record playing by the rules is picked again, twice at most; a record
                # the user asked for is kept (the relaxed blend applies)
                if not opener and self.record in self.songs and track.id not in self._user_ids and self._repicks < 2:
                    probe = Rec(deck, track)
                    probe.vox, probe.state = vox, "ready"
                    try:
                        feasible = self._find_blend_strict(self.songs[self.record], probe) is not None
                    except Exception:
                        feasible = True
                    if not feasible:
                        self._unfit.add(track.id)
                        self._repicks += 1
                        self.songs.pop(deck, None)
                        del samples
                        self._move("pick", f"{track.title[:28]} cannot be blended with {self.songs[self.record].track.title[:22]} by the rules "
                                           f"({getattr(self, '_no_blend_why', 'no placement')}): picked again", to_id=track.id)
                        return
                with self._lock:
                    self._pending[deck] = (track, samples, vox, menv)
            except Exception as e:  # noqa: BLE001
                self.last_error = f"decode {track.title}: {type(e).__name__}: {e}"
                self._move("error", self.last_error, to_id=track.id)
                self.songs.pop(deck, None)
            finally:
                self._decoding.pop(deck, None)
        threading.Thread(target=work, daemon=True, name=f"record-decode-{deck}").start()

    def _mount(self, deck):
        with self._lock:
            item = self._pending.pop(deck, None)
        if item is None:
            return None
        track, samples, vox, menv = (tuple(item) + (None, None))[:4]
        r = self.songs.get(deck)
        if r is None or r.track.id != track.id:
            r = self.songs[deck] = Rec(deck, track)
        r.vox, r.mix_env = vox, menv
        opener = self.record is None and not any(x.state in ("playing", "leaving") for x in self.songs.values())
        r.cue_t = max(0.0, track.nearest_downbeat(0.0)) if opener else self.landmark(track)
        self.submix.post_many([
            {"cmd": "unload", "deck": deck},
            {"cmd": "load", "deck": deck, "samples": samples, "grid": track.grid, "track_id": track.id, "gain_db": track.gain_db},
            {"cmd": "gain", "deck": deck, "value": 0.0, "ramp_s": 0.0},
            {"cmd": "eq", "deck": deck, "low": 1.0, "mid": 1.0, "high": 1.0, "ramp_s": 0.0},
            {"cmd": "filter", "deck": deck, "mode": "off"},
            {"cmd": "echo", "deck": deck, "active": False},
            {"cmd": "rate", "deck": deck, "value": 1.0},
            {"cmd": "stretch_hold", "deck": deck, "value": False},      # (a fresh record plays bit-exact until it is synced)
            {"cmd": "cue", "deck": deck, "time_s": r.cue_t},
        ])
        r.state = "ready"
        self._move("ready", f"{track.title[:30]} is ready on deck {deck.upper()}" + (f", cued to its first downbeat" if opener else f" (its landmark at {_mmss(r.cue_t)})"), to_id=track.id)
        return r

    def _free_deck(self):
        for d in ("a", "b"):
            if d not in self.songs and d not in self._decoding and d not in self._pending:
                return d
        return None

    # -- the step ------------------------------------------------------------------------------------------------------------
    def step(self):
        if not self.running:
            return
        for d in DECKS:
            if d in self._pending:
                self._mount(d)
        clock = self._clock()
        # a record with no room to open into (the first one; or the room went empty): it opens on its own
        if self.record is None:
            ready = [r for r in self.songs.values() if r.state == "ready"]
            if ready:
                self._open(ready[0], clock + int(LEAD_S * RATE))
            else:
                self.wait_why = "the first record is decoding" if self.songs else "nothing to play: no record could be picked"
            return
        rec = self.songs.get(self.record)
        if rec is None:
            self.record = None
            return
        # the plan advances: the next record starts / the bass swaps / the old record is gone
        if self.cut is not None:
            self._advance(clock)
            rec = self.songs.get(self.record)
            if rec is None:
                return
        # the old record's deck leaves
        for d, r in list(self.songs.items()):
            if r.state == "leaving" and r.open_clock is not None and clock >= r.open_clock:
                self.songs.pop(d, None)
        # the next record: picked the moment a record opens
        if not self._has_next() and self._free_deck() is not None:
            nxt = self._pick(rec.track)
            if nxt is not None:
                self._decode(nxt, self._free_deck())
            else:
                self.wait_why = "nothing planned yet: no next record fits (the wall, the length, what has played)"
        nxt = self._next_rec()
        t_now = self._song_time_now(rec.deck)
        played = t_now - rec.cue_t
        left = float(rec.track.duration_s) - t_now
        # the record runs out with nothing under way: the cut now (the next record whole), or the room goes quiet
        if left <= 2.0 * self.bar_s(rec.track, t_now) + 1.0:
            if self.cut is not None and self.cut["kind"] != "cut" and self.cut["state"] != "planned":
                return                                     # (a blend under way ends before the record does)
            if nxt is not None and nxt.state == "ready":
                if self.cut is None or self.cut["clock"] > self._clock_of_song_time(rec.deck, float(rec.track.duration_s)) - int(0.5 * RATE):
                    self._plan_cut(nxt, at_song_t=min(t_now + 1.0 * self.bar_s(rec.track, t_now), float(rec.track.duration_s) - 0.3),
                                   why=f"{rec.track.title[:24]} runs out: the cut now", phrase=False)
            elif left <= 0.3 or self._tel_deck(rec.deck).get("finished"):
                self._move("ended", f"{rec.track.title[:30]} ran out with no next record ready: the room is quiet until one is", from_id=rec.track.id)
                self.submix.post({"cmd": "stop", "deck": rec.deck})
                self.songs.pop(rec.deck, None)
                self.record = None
                self.cut = None
                self.wait_why = "the next record is decoding; it opens the moment it is ready"
            return
        if self.cut is None:
            if nxt is None:
                self.wait_why = "nothing planned yet: the next record is being picked"
            elif nxt.state != "ready":
                self.wait_why = f"nothing planned yet: {nxt.track.title[:24]} is decoding"
            elif clock < self._settled_clock:
                self.wait_why = f"nothing planned yet: {rec.track.title[:24]} is settling to its own tempo ({(self._settled_clock - clock) // RATE + 1} s)"
            elif self._next_asap:
                self._plan_blend(nxt, why="NEXT: the moment it was ready", size="short", asap=True)
            else:
                # THE PLAN, decided the moment the next record is ready: the blend from the record's minimum play
                if not self._plan_breakdown(nxt):
                    self._plan_blend(nxt, why="from its minimum play")
                if self.cut is None:
                    self.wait_why = f"{rec.track.title[:24]} plays to its minimum ({_mmss(self.play_min_s - played)} more)"
        else:
            self.wait_why = ""

    def _has_next(self):
        return any(d != self.record and r.state in ("decoding", "ready") for d, r in self.songs.items()) or bool(self._decoding) or bool(self._pending)

    def _next_rec(self):
        for d, r in self.songs.items():
            if d != self.record and r.state in ("decoding", "ready"):
                return r
        return None

    def _open(self, r, at):
        """The room is empty: `r` opens on its own at clock `at`, no blend."""
        self.submix.post_many([
            {"at": at, "cmd": "cue", "deck": r.deck, "time_s": r.cue_t},
            {"at": at, "cmd": "gain", "deck": r.deck, "value": 1.0, "ramp_s": 0.005},
            {"at": at, "cmd": "start", "deck": r.deck},
        ])
        r.state, r.open_clock = "playing", at
        r.open_bar = self._bar_index(r.track, r.cue_t)
        self.record = r.deck
        self.cut = None
        self._next_asap = False
        self._settled_clock = 0
        self._repicks = 0
        self.wait_why = ""
        if self.brain is not None:
            try:
                self.brain.note_played(r.track)
            except Exception:
                pass
        self._move("open", f"{r.track.title} opens whole at {r.track.bpm:.1f} bpm, key {r.track.camelot or '?'}, from {_mmss(r.cue_t)}", to_id=r.track.id)

    # -- the plan's geometry ------------------------------------------------------------------------------------------------
    def _geometry(self, rec):
        """(track, t_now, bar_s, k_now, open_bar, k_min, k_end): the record's bars as the plan counts them."""
        track = rec.track
        t_now = self._song_time_now(rec.deck)
        bar_s = self.bar_s(track, t_now)
        k_now = self._bar_index(track, t_now)
        open_bar = self._bar_index(track, rec.cue_t, t_ref=t_now)
        k_min = int(math.ceil((rec.cue_t + self.play_min_s - self._first_down(track, t_now)) / bar_s - 1e-6))
        k_end = int(math.floor((float(track.duration_s) - 0.3 - self._first_down(track, t_now)) / bar_s))
        return track, t_now, bar_s, k_now, open_bar, k_min, k_end

    @staticmethod
    def _phrase_up(k, open_bar):
        while (k - open_bar) % PHRASE_BARS:
            k += 1
        return k

    def _section_kind(self, track, t):
        try:
            s = track.section_at(t)
        except Exception:
            s = None
        return (s or {}).get("kind") if s else None

    def _tempo_ratio(self, rec, nxt):
        """The next record's rate so its beat matches the record's (the record at the rate it plays now)."""
        return float(rec.track.bpm) * self._rate_now(rec.deck) / float(nxt.track.bpm)

    @staticmethod
    def key_fit(a, b):
        """Camelot fit of two records' keys, 0..1; None when a key is unknown."""
        try:
            from lib.dj.brain import camelot_compat
            if not a.camelot or not b.camelot:
                return None
            return float(camelot_compat(a.camelot, b.camelot))
        except Exception:
            return None

    # -- where the records SING (the vocal envelopes measured at decode) ------------------------------------------------------
    @staticmethod
    def _vox_quiet(r, t0, t1):
        """True when `r`'s vocal stem stays under VOX_QUIET of its loud level from song time t0 to t1; False when it
        sings there; None when the record has no envelope (no stems on disk: nothing is known, nothing is refused)."""
        if r is None or r.vox is None:
            return None
        i0, i1 = max(0, int(t0 * 2)), min(len(r.vox), int(math.ceil(t1 * 2)) + 1)
        if i1 - i0 < 1:
            return None
        return bool(float(max(r.vox[i0:i1])) < VOX_QUIET)

    def _cue_for_lead(self, nxt, L, bar_b, prefer_landmark=True, quiet_bars=None):
        """Where the next record is cued so that it does NOT sing for `quiet_bars` bars (default: the L bars of the
        lead, until the swap) from its cue: its landmark minus L bars when that window is vocal-quiet (its drop lands
        on the swap), else the latest phrase boundary of its intro whose window is quiet, else its first downbeat when
        quiet. Returns (cue, on_landmark); (None, False): it sings everywhere it could enter."""
        t = nxt.track
        lm = self.landmark(t)
        span = L * bar_b
        quiet_span = (quiet_bars if quiet_bars is not None else L) * bar_b
        cands = []
        if prefer_landmark and lm - span >= -0.5 * bar_b:
            cands.append(max(0.0, t.nearest_downbeat(lm - span)))
        first = max(0.0, t.nearest_downbeat(0.0))
        k = int(math.floor((lm - span - first) / max(PHRASE_BARS * bar_b, 1e-6)))
        while k >= 0:
            c = max(0.0, t.nearest_downbeat(first + k * PHRASE_BARS * bar_b))
            if c not in cands:
                cands.append(c)
            k -= 1
        if first not in cands:
            cands.append(first)
        for c in cands:
            q = self._vox_quiet(nxt, c, c + quiet_span)
            if q is not False:
                return c, (abs(c - (lm - span)) < 0.5 * bar_b)
        return None, False

    def _quietest_cue(self, nxt, L, bar_b):
        """When no window of the next record is vocal-quiet for the lead: the phrase boundary of its intro (or its
        landmark minus the lead) whose lead window sings LEAST - the relaxed plan's cue, never None."""
        t = nxt.track
        lm = self.landmark(t)
        span = L * bar_b
        first = max(0.0, t.nearest_downbeat(0.0))
        cands = [max(0.0, t.nearest_downbeat(lm - span))] if lm - span >= -0.5 * bar_b else []
        k = int(math.floor((lm - span - first) / max(PHRASE_BARS * bar_b, 1e-6)))
        while k >= 0:
            cands.append(max(0.0, t.nearest_downbeat(first + k * PHRASE_BARS * bar_b)))
            k -= 1
        cands.append(first)
        if nxt.vox is None:
            return cands[0], False
        best, best_v = cands[0], None
        for c in cands:
            i0, i1 = max(0, int(c * 2)), min(len(nxt.vox), int(math.ceil((c + span) * 2)) + 1)
            v = float(max(nxt.vox[i0:i1])) if i1 > i0 else 0.0
            if best_v is None or v < best_v - 1e-6:
                best, best_v = c, v
        return best, abs(best - (lm - span)) < 0.5 * bar_b

    def _find_blend(self, rec, nxt, relax=False):
        """... `relax` (2026-10-26, the user: "we should never be just ripping a playing song out of the mix unless I hit
        the next key"): when no placement keeps the rules, the blend is placed anyway at the LAST phrases of the record
        - its outro, where records are built to be mixed out of - as the short blend, with the next record's least-
        singing intro window; the keys may only get along. A blend that bends a rule beats a cut the user did not ask for."""
        if relax:
            track, t_now, bar_s, k_now, open_bar, k_min, k_end = self._geometry(rec)
            L, _T0 = SIZES["short"]
            T = TAIL_BARS
            K = self._phrase_up(max(k_now + CUT_MIN_BARS, k_min), open_bar)
            K_last = K
            while K_last + PHRASE_BARS + L + T <= k_end - 1:
                K_last += PHRASE_BARS
            K = K_last
            if K + L + T > k_end - 1:
                return None                           # no phrase of the record is left: it runs out (the cut at its end)
            bar_b = self.bar_s(nxt.track, self.landmark(nxt.track))
            cue_b, on_lm = self._cue_for_lead(nxt, L, bar_b, quiet_bars=L)
            if cue_b is None:
                cue_b, on_lm = self._quietest_cue(nxt, L, bar_b)
            fit = self.key_fit(track, nxt.track)
            why = (f"no placement kept every rule, so the last phrases of {track.title[:22]} (its way out), the short blend"
                   + (f"; keys {track.camelot}→{nxt.track.camelot} fit {fit:.2f}" if fit is not None else "") + (", its drop on the swap" if on_lm else ", from its intro"))
            return {"K": K, "size": "short", "L": L, "T": T, "cue_b": cue_b, "on_landmark": on_lm, "why": why, "relaxed": True}
        return self._find_blend_strict(rec, nxt)

    def _find_blend_strict(self, rec, nxt):
        """PLACEMENT BY THE RECORDS (docs/DJ_MIXING.md 7): the earliest phrase boundary K of the record at or after its
        minimum play AND its last stretch (LAST_FROM of its length), in a groove or an outro (a build or a breakdown is
        the breakdown swap's), with a size (long when the next record's intro allows it, else standard, else short)
        that fits the record's room, and NEVER TWO SINGERS: either the next record is cued where its vocal stays quiet
        through the whole overlap (the record may sing over its intro - the DJ's standard move), or the record's own
        vocal is quiet through the overlap and the next record's until the swap. Returns
        {"K","size","L","T","cue_b","on_landmark","why"} or None."""
        track, t_now, bar_s, k_now, open_bar, k_min, k_end = self._geometry(rec)
        fit = self.key_fit(track, nxt.track)
        self._no_blend_why = "no place where a blend keeps the two records' vocals apart over a steady stretch of the record"
        if fit is not None and fit < KEY_BLEND:
            self._no_blend_why = f"keys {track.camelot}→{nxt.track.camelot} clash (fit {fit:.2f}): no phrases of the two together"
            return None                               # the keys clash: two records never overlap for phrases (the cut does not care)
        k_last = int(math.ceil((LAST_FROM * float(track.duration_s) - self._first_down(track, t_now)) / bar_s))
        K0 = self._phrase_up(max(k_now + CUT_MIN_BARS, k_min, k_last), open_bar)
        bar_b = self.bar_s(nxt.track, self.landmark(nxt.track))
        # the DJ's own sizes: the standard blend and the short one - never the long one (2026-10-26: a 32-bar lead held
        # two records together for a minute, with the record's own build and drop inside it; BAD); long is for a dial
        sizes = ["short"] if (fit is not None and fit < KEY_LONG) else ["standard", "short"]
        best, best_score = None, None
        K = K0
        while K + SIZES["short"][0] + 1 <= k_end - 1:
            tK = self._bar_time(track, K, t_now)
            kind = self._section_kind(track, tK + 0.25 * bar_s)
            for nm in sizes:
                L, _T0 = SIZES[nm]
                T = TAIL_OUTRO_BARS if self._section_kind(track, self._bar_time(track, K + L, t_now) + 0.25 * bar_s) == "outro" else TAIL_BARS
                if K + L + T > k_end - 1:
                    continue
                t1 = self._bar_time(track, K + L + T, t_now)
                # THE RECORD STAYS STEADY through the overlap: no build, no breakdown, no drop of its own under the next record
                # (Dissolution built and dropped inside a lead, then the next record dropped on the swap: two drops, BAD)
                if not self._steady_span(track, tK, t1):
                    continue
                a_quiet = self._vox_quiet(rec, tK, t1)
                score = 0
                # (1) the next record silent for the whole overlap: the record may sing over its intro
                cue_b, on_lm = self._cue_for_lead(nxt, L, bar_b, quiet_bars=L + T)
                how = f"{nxt.track.title[:22]}'s vocal silent through the mix"
                if cue_b is not None:
                    score = 3 if a_quiet is not False else 1
                    if a_quiet is not False:
                        how = "neither record sings through the mix"
                else:
                    # (2) the record quiet through the overlap, the next record quiet until the swap
                    if a_quiet is False:
                        continue                      # both would sing: not this place, not this size
                    cue_b, on_lm = self._cue_for_lead(nxt, L, bar_b, quiet_bars=L)
                    if cue_b is None:
                        continue                      # the next record sings everywhere it could enter for this lead
                    score = 2
                    how = f"{track.title[:22]}'s vocal quiet through the mix, {nxt.track.title[:22]}'s until the swap"
                if on_lm:
                    score += 0.5                      # its drop on the swap: the payoff lands on the one
                if kind == "outro":
                    score += 0.5                      # the record's way out is where it is built to be mixed out of
                why = (f"{track.title[:22]}'s {kind or 'groove'} from bar {self.bar_n + (K - k_now)}; {how}"
                       + (", its drop on the swap" if on_lm else ", from its intro") + (f"; keys {track.camelot}→{nxt.track.camelot} fit {fit:.2f}" if fit is not None else ""))
                if best_score is None or score > best_score + 1e-9:
                    best, best_score = {"K": K, "size": nm, "L": L, "T": T, "cue_b": cue_b, "on_landmark": on_lm, "why": why}, score
                if best_score >= 3.5:
                    return best                      # nothing beats both quiet with the drop on the swap: the earliest such place
            K += PHRASE_BARS
        return best

    def _steady_span(self, track, t0, t1):
        """The record is steady from t0 to t1: no build or breakdown section inside, and no DROP (a groove that follows a
        build, a breakdown or an intro) starting inside - the next record is the only thing that moves under a blend."""
        secs = track.sections or []
        eps = 0.25 * self.bar_s(track, t0)
        for i, s in enumerate(secs):
            s0, s1 = float(s["start_s"]), float(s["end_s"])
            if s1 <= t0 + eps or s0 >= t1 - eps:
                continue
            if s.get("kind") in ("build", "breakdown"):
                return False
            if s.get("kind") == "groove" and i > 0 and secs[i - 1].get("kind") in ("build", "breakdown", "intro") and t0 + eps < s0 < t1 - eps:
                return False
        return True

    def _cancel_plan(self):
        if self.cut is not None:
            self.submix.post({"cmd": "cancel", "txn": self.cut["txn"]})

    # -- THE BLEND ----------------------------------------------------------------------------------------------------------
    def _plan_blend(self, nxt, why, size=None, asap=False, k_start=None):
        """Decide the blend ONCE (docs/DJ_MIXING.md 5 A). `size`: short | standard | long (None: the largest that fits
        the record's room and the next record's intro); `asap`: from the next bar + NEXT_BARS, no phrase alignment;
        `k_start`: a bar of the record to start on (STAY). The old plan, if any, is cancelled and re-posted."""
        rec = self.songs.get(self.record)
        if rec is None or nxt is None or nxt.state != "ready":
            return False
        ratio = self._tempo_ratio(rec, nxt)
        fit = self.key_fit(rec.track, nxt.track)
        if asap and fit is not None and fit < KEY_BLEND:
            # NEXT into a clashing key: the cut now, not a phrase of two keys fighting
            return self._plan_cut(nxt, why=f"{why} - keys {rec.track.camelot}→{nxt.track.camelot} clash (fit {fit:.2f}): the cut", phrase=False, min_bars=NEXT_BARS, honor_min=False)
        if abs(ratio - 1.0) > MAX_TEMPO_GAP:
            if k_start is not None:
                return False
            if asap:
                return self._plan_cut(nxt, why=f"{why} - a tempo gap of {abs(ratio - 1.0) * 100:.0f}% the stretcher cannot match: the cut",
                                      phrase=False, min_bars=NEXT_BARS, honor_min=False)
            # the DJ's own plan never rips the record: a tempo gap no blend can bridge means the record plays to its END and the
            # next record comes in whole on its last downbeat (the pick should not have offered it: see _pick)
            return self._plan_cut(nxt, why=f"{why} - a tempo gap of {abs(ratio - 1.0) * 100:.0f}% no blend can bridge: {rec.track.title[:22]} plays to its end",
                                  at_song_t=float(rec.track.duration_s) - 0.3)
        found = None
        with self._lock:
            track, t_now, bar_s, k_now, open_bar, k_min, k_end = self._geometry(rec)
            lm_b = self.landmark(nxt.track)
            bar_b = self.bar_s(nxt.track, lm_b)
            chosen, T, cue_b = None, None, None
            if k_start is not None or asap:
                # an ORDER (NEXT: from the next bar + 2; STAY: a phrase later): the place is yours, the next record's cue
                # still keeps its vocal out until the swap when any cue of it can
                K = max(k_now + 1, int(k_start)) if k_start is not None else k_now + NEXT_BARS
                lm_bars_b = lm_b / max(bar_b, 1e-6)
                if size == "long":
                    size = "standard"                 # (the DJ never chooses the long blend; a STAY carried it forward - the gate, 2026-10-27)
                order = [size] if size else ["standard", "short"]
                for nm in order + [x for x in ("standard", "short") if x not in order]:
                    L, _T0 = SIZES[nm]
                    if nm == "long" and not (size == "long" or lm_bars_b >= 32 - 1e-6):
                        continue
                    T = TAIL_OUTRO_BARS if self._section_kind(track, self._bar_time(track, K + L, t_now) + 0.25 * bar_s) == "outro" else TAIL_BARS
                    if K + L + T > k_end - 1 and T > TAIL_BARS:
                        T = TAIL_BARS                 # (the long outro tail does not fit: the short one)
                    if K + L + T <= k_end - 1:
                        chosen = nm
                        break
                if chosen is not None:
                    L, _T0 = SIZES[chosen]
                    cue_b, _on = self._cue_for_lead(nxt, L, bar_b)
                    if cue_b is None:
                        cue_b = max(0.0, nxt.track.nearest_downbeat(lm_b - L * bar_b)) if lm_b - L * bar_b >= 0.0 else max(0.0, nxt.track.nearest_downbeat(0.0))
            else:
                # THE PLAN: placement by the records (vocals apart, the record's last stretch, the sizes that fit); when
                # no placement keeps every rule, the relaxed one at the record's way out - NEVER a cut the user did not press
                found = self._find_blend(rec, nxt)
                if found is None:
                    found = self._find_blend(rec, nxt, relax=True)
                if found is not None:
                    K, chosen, T, cue_b = found["K"], found["size"], found["T"], found["cue_b"]
                    why = f"{why}: {found['why']}"
        if chosen is None:
            if k_start is not None:
                return False                          # STAY with no room a phrase later: the plan stands (never an earlier cut)
            if not asap:
                # not a phrase of the record is left for any blend: it plays to its end and the next record comes in whole there
                return self._plan_cut(nxt, why=f"{why} - no phrase of {track.title[:22]} is left for a blend: it plays to its end", at_song_t=float(track.duration_s) - 0.3)
            return self._plan_cut(nxt, why=f"{why} - {track.title[:24]} has no room left for a blend: the cut", phrase=False, min_bars=NEXT_BARS, honor_min=False)
        with self._lock:
            L, _T0 = SIZES[chosen]
            S, E = K + L, K + L + T
            T_K = self._clock_of_song_time(rec.deck, self._bar_time(track, K, t_now))
            T_K = max(T_K, self._clock() + int((LEAD_S + 0.6) * RATE))
            wall_bar = int(round(bar_s / self._rate_now(rec.deck) * RATE))          # one bar of the record, in frames
            beat = wall_bar // 4
            T_S, T_E = T_K + L * wall_bar, T_K + (L + T) * wall_bar
            nxt.cue_t, nxt.rate = cue_b, ratio
            self._cancel_plan()
            self._seq += 1
            txn = f"blend{self._seq}"
            a, b = rec.deck, nxt.deck
            half, quarter = (L * wall_bar) // 2, (L * wall_bar) // 4
            pre = max(self._clock() + int(0.1 * RATE), T_K - int(0.5 * RATE))
            ev = [
                {"at": pre, "cmd": "cue", "deck": b, "time_s": cue_b, "txn": txn},
                {"at": pre, "cmd": "rate", "deck": b, "value": ratio, "txn": txn},
                {"at": pre, "cmd": "eq", "deck": b, "low": 0.0, "mid": 0.8, "high": 0.8, "ramp_s": 0.0, "txn": txn},
                {"at": pre, "cmd": "gain", "deck": b, "value": 0.0, "ramp_s": 0.0, "txn": txn},
                {"at": T_K, "cmd": "start", "deck": b, "txn": txn},
                {"at": T_K, "cmd": "sync", "slave": b, "master": a, "bias_beats": 0.0, "audio_pll": True, "txn": txn},
                # the next record sits UNDER the record until the swap (UNDER_GAIN): two records at full was the mud
                {"at": T_K, "cmd": "gain", "deck": b, "value": UNDER_GAIN, "ramp_s": half / RATE, "curve": "power", "txn": txn},
                # the highs trade, then the mids (top-down)
                {"at": T_K + half, "cmd": "eq", "deck": b, "high": 1.0, "ramp_s": quarter / RATE, "txn": txn},
                {"at": T_K + half, "cmd": "eq", "deck": a, "high": 0.5, "ramp_s": quarter / RATE, "txn": txn},
                {"at": T_K + half + quarter, "cmd": "eq", "deck": b, "mid": 1.0, "ramp_s": quarter / RATE, "txn": txn},
                {"at": T_K + half + quarter, "cmd": "eq", "deck": a, "mid": 0.6, "ramp_s": quarter / RATE, "txn": txn},
                # THE BASS SWAP on the phrase, two beats: one bassline at a time; the next record to full with its bass
                {"at": T_S, "cmd": "eq", "deck": a, "low": 0.0, "ramp_s": 2 * beat / RATE, "txn": txn},
                {"at": T_S, "cmd": "eq", "deck": b, "low": 1.0, "ramp_s": 2 * beat / RATE, "txn": txn},
                {"at": T_S, "cmd": "gain", "deck": b, "value": 1.0, "ramp_s": 2 * beat / RATE, "curve": "power", "txn": txn},
                # the record comes down over the tail and stops
                {"at": T_S, "cmd": "gain", "deck": a, "value": 0.0, "ramp_s": (T * wall_bar) / RATE, "curve": "power", "txn": txn},
                {"at": T_E, "cmd": "end_sync", "txn": txn},
                {"at": T_E, "cmd": "rate", "deck": b, "value": 1.0, "ramp_s": GLIDE_S, "txn": txn},
                {"at": T_E + int(0.3 * RATE), "cmd": "stop", "deck": a, "txn": txn},
                {"at": T_E + int(0.4 * RATE), "cmd": "unload", "deck": a, "txn": txn},
            ]
            self.submix.post_many(ev)
            moved = self.cut is not None
            self.cut = {"seq": self._seq, "txn": txn, "kind": "blend", "size": chosen, "a": a, "b": b,
                        "clock": T_K, "swap_clock": T_S, "end_clock": T_E, "stop_clock": T_E + int(0.4 * RATE),
                        "bar": self.bar_n + (K - k_now), "swap_bar": self.bar_n + (S - k_now), "end_bar": self.bar_n + (E - k_now),
                        "L": L, "T": T, "song_t": self._bar_time(track, K, t_now), "cue_b": cue_b, "ratio": ratio,
                        "why": why, "state": "planned"}
            self._next_asap = False
            self.wait_why = ""
        in_s = (T_K - self._clock()) / RATE
        self._move("blend_move" if moved else "blend_plan",
                   f"the blend {'moves' if moved else 'is planned'}: {nxt.track.title[:28]} starts in {_mmss(in_s)} ({self.cut['bar'] - self.bar_n} bars), bass out, "
                   f"beat-matched {'+' if ratio >= 1 else ''}{(ratio - 1) * 100:.1f}%; highs then mids over the lead; THE BASS SWAP {L} bars in; "
                   f"{track.title[:22]} out over {T} more ({chosen}, {L + T} bars together) - {why}",
                   from_id=rec.track.id, to_id=nxt.track.id)
        return True

    # -- THE BREAKDOWN SWAP ----------------------------------------------------------------------------------------------------
    def _plan_breakdown(self, nxt, why="the record's breakdown"):
        """docs/DJ_MIXING.md 5 C: the record has a breakdown of BREAK_MIN_BARS+ bars resolving at or after its minimum
        play, the next record has a build into its landmark - the next record's intro and build go under the
        breakdown, its drop lands where the record's breakdown resolves, the record goes out there in one beat."""
        rec = self.songs.get(self.record)
        if rec is None or nxt is None or nxt.state != "ready":
            return False
        if not self.landmark_has_build(nxt.track):
            return False
        ratio = self._tempo_ratio(rec, nxt)
        if abs(ratio - 1.0) > MAX_TEMPO_GAP:
            return False
        track, t_now, bar_s, k_now, open_bar, k_min, k_end = self._geometry(rec)
        secs = track.sections or []
        pick = None
        for i, s in enumerate(secs):
            if s.get("kind") != "breakdown":
                continue
            bs = self._bar_index(track, float(s["start_s"]), t_ref=t_now)
            be = self._bar_index(track, float(s["end_s"]), t_ref=t_now)
            if be - bs < BREAK_MIN_BARS or bs < k_now + CUT_MIN_BARS + 4:
                continue
            # where it RESOLVES: the record's own drop (the first groove within 16 bars after the breakdown) else its end
            D = be
            j = i + 1
            while j < len(secs) and secs[j].get("kind") in ("build", "breakdown"):
                j += 1
            if j < len(secs) and secs[j].get("kind") == "groove":
                dj = self._bar_index(track, float(secs[j]["start_s"]), t_ref=t_now)
                if dj - be <= 16:
                    D = dj
            D = self._phrase_up(D, open_bar) if (D - open_bar) % PHRASE_BARS <= 1 else D
            if D < k_min or D > k_end - 2:
                continue
            pick = (bs, be, D)
            break
        if pick is None:
            return False
        bs, be, D = pick
        with self._lock:
            lm = self.landmark(nxt.track)
            bar_b = self.bar_s(nxt.track, lm)
            K = bs - 4
            lm_bars = int(math.floor(lm / max(bar_b, 1e-6) + 1e-6))
            if lm_bars < D - K:
                K = D - lm_bars                       # its whole intro and build fit under what is left
            if K < k_now + CUT_MIN_BARS:
                return False
            cue_b = max(0.0, nxt.track.nearest_downbeat(lm - (D - K) * bar_b))
            # the next record's intro and build go under the breakdown: not if it SINGS there (the breakdown is the
            # record's quiet; a vocal over it is the flag 'clash'); the blend search answers instead
            if self._vox_quiet(nxt, cue_b, lm) is False:
                return False
            nxt.cue_t, nxt.rate = cue_b, ratio
            T_K = self._clock_of_song_time(rec.deck, self._bar_time(track, K, t_now))
            T_K = max(T_K, self._clock() + int((LEAD_S + 0.6) * RATE))
            wall_bar = int(round(bar_s / self._rate_now(rec.deck) * RATE))
            beat = wall_bar // 4
            T_B, T_D = T_K + (bs - K) * wall_bar, T_K + (D - K) * wall_bar
            self._cancel_plan()
            self._seq += 1
            txn = f"break{self._seq}"
            a, b = rec.deck, nxt.deck
            pre = max(self._clock() + int(0.1 * RATE), T_K - int(0.5 * RATE))
            ev = [
                {"at": pre, "cmd": "cue", "deck": b, "time_s": cue_b, "txn": txn},
                {"at": pre, "cmd": "rate", "deck": b, "value": ratio, "txn": txn},
                {"at": pre, "cmd": "eq", "deck": b, "low": 0.0, "mid": 1.0, "high": 1.0, "ramp_s": 0.0, "txn": txn},
                {"at": pre, "cmd": "gain", "deck": b, "value": 0.0, "ramp_s": 0.0, "txn": txn},
                {"at": T_K, "cmd": "start", "deck": b, "txn": txn},
                {"at": T_K, "cmd": "sync", "slave": b, "master": a, "bias_beats": 0.0, "audio_pll": True, "txn": txn},
                {"at": T_K, "cmd": "gain", "deck": b, "value": 1.0, "ramp_s": max(1, bs - K) * wall_bar / RATE, "curve": "power", "txn": txn},
                {"at": T_B, "cmd": "eq", "deck": a, "mid": 0.7, "ramp_s": wall_bar / RATE, "txn": txn},
                # the drop: the next record's bass in, the record out, one beat
                {"at": T_D, "cmd": "eq", "deck": b, "low": 1.0, "ramp_s": beat / RATE, "txn": txn},
                {"at": T_D, "cmd": "gain", "deck": a, "value": 0.0, "ramp_s": beat / RATE, "txn": txn},
                {"at": T_D + beat, "cmd": "end_sync", "txn": txn},
                {"at": T_D + beat, "cmd": "rate", "deck": b, "value": 1.0, "ramp_s": GLIDE_S, "txn": txn},
                {"at": T_D + int(STOP_AFTER_S * RATE), "cmd": "stop", "deck": a, "txn": txn},
                {"at": T_D + int(STOP_AFTER_S * RATE) + int(0.1 * RATE), "cmd": "unload", "deck": a, "txn": txn},
            ]
            self.submix.post_many(ev)
            moved = self.cut is not None
            self.cut = {"seq": self._seq, "txn": txn, "kind": "breakdown", "size": "breakdown", "a": a, "b": b,
                        "clock": T_K, "swap_clock": T_D, "end_clock": T_D + beat, "stop_clock": T_D + int(STOP_AFTER_S * RATE) + int(0.1 * RATE),
                        "bar": self.bar_n + (K - k_now), "swap_bar": self.bar_n + (D - k_now), "end_bar": self.bar_n + (D - k_now),
                        "L": D - K, "T": 0, "song_t": self._bar_time(track, K, t_now), "cue_b": cue_b, "ratio": ratio,
                        "why": why, "state": "planned"}
            self._next_asap = False
            self.wait_why = ""
        in_s = (T_K - self._clock()) / RATE
        self._move("blend_move" if moved else "blend_plan",
                   f"the breakdown swap {'moves' if moved else 'is planned'}: {nxt.track.title[:28]} starts in {_mmss(in_s)} ({self.cut['bar'] - self.bar_n} bars), bass out, "
                   f"under {track.title[:22]}'s breakdown ({be - bs} bars from bar {self.bar_n + (bs - k_now)}); its drop lands at bar {self.cut['swap_bar']} "
                   f"where the breakdown resolves and {track.title[:22]} goes out in a beat - {why}",
                   from_id=rec.track.id, to_id=nxt.track.id)
        return True

    # -- THE CUT (the fallback) -----------------------------------------------------------------------------------------------
    def _plan_cut(self, nxt, why, at_song_t=None, phrase=True, min_bars=CUT_MIN_BARS, honor_min=True):
        """The record's gain to 0 one beat before the one, the next record whole at its landmark downbeat on the one.
        Only when no blend fits (the record runs out; a tempo gap beyond the stretcher; no place that keeps the two
        records' vocals apart). `honor_min`: not before the record's minimum play (a cut planned "from its minimum
        play" fired six bars after the record opened - the record gate, 2026-10-26: 47 s of Alone); an order (NEXT,
        the record running out) passes False. The bar is the first phrase boundary where the record is not singing
        in the bar before it (a cut mid-word), within four phrases, else the first."""
        rec = self.songs.get(self.record)
        if rec is None or nxt is None or nxt.state != "ready":
            return False
        with self._lock:
            track, t_now, bar_s, k_now, open_bar, k_min, k_end = self._geometry(rec)
            if at_song_t is not None:
                k = max(k_now + 1, int(math.ceil((at_song_t - self._first_down(track, t_now)) / bar_s - 1e-6)))
            else:
                k = k_now + max(1, int(min_bars))
                if honor_min:
                    k = max(k, k_min)
                if phrase:
                    k = self._phrase_up(k, open_bar)
                    if honor_min:
                        for j in range(5):
                            kj = k + j * PHRASE_BARS
                            if kj > k_end - 1:
                                break
                            if self._vox_quiet(rec, self._bar_time(track, kj, t_now) - bar_s, self._bar_time(track, kj, t_now)) is not False:
                                k = kj
                                break
            k = min(k, max(k_now + 1, k_end))
            lm = self.landmark(nxt.track)
            nxt.cue_t, nxt.rate = lm, 1.0
            T = self._clock_of_song_time(rec.deck, self._bar_time(track, k, t_now))
            T = max(T, self._clock() + int(LEAD_S * RATE) + int(self.hole_beats * 60.0 / float(track.bpm) * RATE))
            hole = int(self.hole_beats * 60.0 / float(track.bpm) * RATE)
            self._cancel_plan()
            self._seq += 1
            txn = f"cut{self._seq}"
            stop_at = T + int(STOP_AFTER_S * RATE)
            self.submix.post_many([
                {"at": T - hole, "cmd": "gain", "deck": rec.deck, "value": 0.0, "ramp_s": 0.01, "txn": txn},
                {"at": max(self._clock() + int(0.1 * RATE), T - int(0.5 * RATE)), "cmd": "cue", "deck": nxt.deck, "time_s": lm, "txn": txn},
                {"at": max(self._clock() + int(0.1 * RATE), T - int(0.5 * RATE)), "cmd": "rate", "deck": nxt.deck, "value": 1.0, "txn": txn},
                {"at": max(self._clock() + int(0.1 * RATE), T - int(0.5 * RATE)), "cmd": "eq", "deck": nxt.deck, "low": 1.0, "mid": 1.0, "high": 1.0, "ramp_s": 0.0, "txn": txn},
                {"at": T, "cmd": "gain", "deck": nxt.deck, "value": 1.0, "ramp_s": 0.005, "txn": txn},
                {"at": T, "cmd": "start", "deck": nxt.deck, "txn": txn},
                {"at": stop_at, "cmd": "stop", "deck": rec.deck, "txn": txn},
                {"at": stop_at + int(0.1 * RATE), "cmd": "unload", "deck": rec.deck, "txn": txn},
            ])
            moved = self.cut is not None
            self.cut = {"seq": self._seq, "txn": txn, "kind": "cut", "size": "cut", "a": rec.deck, "b": nxt.deck,
                        "clock": T, "swap_clock": T, "end_clock": T, "hole_clock": T - hole, "stop_clock": stop_at + int(0.1 * RATE),
                        "bar": self.bar_n + (k - k_now), "swap_bar": self.bar_n + (k - k_now), "end_bar": self.bar_n + (k - k_now),
                        "L": 0, "T": 0, "song_t": self._bar_time(track, k, t_now), "cue_b": lm, "ratio": 1.0,
                        "why": why, "state": "planned"}
            self._next_asap = False
            self.wait_why = ""
        in_s = (T - self._clock()) / RATE
        self._move("cut_move" if moved else "cut_plan",
                   f"the cut {'moves' if moved else 'is planned'}: {nxt.track.title[:28]} comes in whole at its landmark in {_mmss(in_s)} ({self.cut['bar'] - self.bar_n} bars) - {why}",
                   from_id=rec.track.id, to_id=nxt.track.id)
        return True

    # -- the plan advancing ------------------------------------------------------------------------------------------------------
    def _advance(self, clock):
        c = self.cut
        if c is None:
            return
        a, b = self.songs.get(c["a"]), self.songs.get(c["b"])
        if c["state"] == "planned" and clock >= c["clock"]:
            if c["kind"] == "cut":
                self._swapped(clock)
                return
            c["state"] = "started"
            if b is not None:
                b.state, b.open_clock = "playing", c["clock"]
                b.open_bar = self._bar_index(b.track, b.cue_t)
                self._move("blend_start", f"{b.track.title[:28]} is in under {a.track.title[:24] if a else 'the record'}: bass out, rising"
                           + (f"; the breakdown swap lands in {c['swap_bar'] - self.bar_n} bars" if c["kind"] == "breakdown" else f"; THE BASS SWAP in {c['swap_bar'] - self.bar_n} bars"),
                           from_id=a.track.id if a else None, to_id=b.track.id)
        if c["state"] == "started" and clock >= c["swap_clock"]:
            self._swapped(clock)
            return
        if c["state"] == "swapped" and clock >= c["end_clock"]:
            self._done(clock)

    def _swapped(self, clock):
        """The next record is the record: the bass swap (a blend), the drop (a breakdown swap), the one (a cut)."""
        c = self.cut
        a, b = self.songs.get(c["a"]), self.songs.get(c["b"])
        c["state"] = "swapped"
        if a is not None:
            a.state = "leaving"
            a.open_clock = c["stop_clock"] + int(0.5 * RATE)   # (reused: when its deck is forgotten)
        if b is not None:
            if b.state != "playing":
                b.state, b.open_clock = "playing", c["clock"]
            b.open_bar = self._bar_index(b.track, b.cue_t)
            self.record = b.deck
            self._repicks = 0
            if self.brain is not None:
                try:
                    self.brain.note_played(b.track)
                except Exception:
                    pass
            if c["kind"] == "blend":
                self._move("swap", f"THE BASS SWAP: {b.track.title} is the record" + (f"; {a.track.title[:24]} comes down over {c['T']} bars" if a else ""),
                           from_id=a.track.id if a else None, to_id=b.track.id)
            elif c["kind"] == "breakdown":
                self._move("swap", f"THE DROP: {b.track.title} is the record" + (f" where {a.track.title[:24]}'s breakdown resolved; it is out" if a else ""),
                           from_id=a.track.id if a else None, to_id=b.track.id)
            else:
                self._move("cut", f"CUT on the one: {b.track.title} is the record, whole, from its landmark" + (f" (after {a.track.title[:24]})" if a else ""),
                           from_id=a.track.id if a else None, to_id=b.track.id)
        else:
            self.record = None
        self._settled_clock = c["end_clock"] + int((GLIDE_S + 0.5) * RATE) if c["kind"] != "cut" else 0
        if c["kind"] == "cut" or c["end_clock"] <= clock:
            self._done(clock)

    def _done(self, clock):
        c = self.cut
        if c is None:
            return
        a, b = self.songs.get(c["a"]), self.songs.get(c["b"])
        if c["kind"] != "cut" and b is not None:
            self._move("blend_done", f"{b.track.title[:28]} plays whole" + (f"; {a.track.title[:24]} is gone" if a else "") + ", gliding back to its own tempo",
                       from_id=a.track.id if a else None, to_id=b.track.id)
        self.cut = None
        self._next_asap = False

    # -- the controls (orders) -----------------------------------------------------------------------------------------------
    def next(self):
        if not self.running or self.record is None:
            return False, "press START" if not self.running else "still decoding: the first record"
        rec = self.songs.get(self.record)
        nxt = self._next_rec()
        if rec is not None:
            rec.next_pressed = True
        c = self.cut
        if c is not None and c["state"] == "swapped":
            return True, "the blend is finishing now"
        if c is not None and c["kind"] == "cut" and self._clock() >= c.get("hole_clock", c["clock"]) - int(LEAD_S * RATE):
            return True, "the cut is firing now"
        if c is not None and c["state"] == "started":
            # under way: the swap brought forward to the next bar + NEXT_BARS, the tail short
            return self._hurry()
        if nxt is None or nxt.state != "ready":
            self._next_asap = True
            self._move("next", "NEXT: " + (f"{nxt.track.title[:24]} is still decoding - the blend comes the moment it is ready" if nxt else "the next record is being picked - the blend comes the moment it is ready"))
            return False, "still decoding: the blend comes the moment it is ready"
        if self._clock() < self._settled_clock:
            self._next_asap = True
            self._move("next", f"NEXT: {rec.track.title[:24]} is settling to its own tempo - the blend comes the moment it has")
            return False, "nothing planned yet: the record is settling to its own tempo - the blend comes the moment it has"
        ok = self._plan_blend(nxt, why="NEXT", size="short", asap=True)
        if ok:
            kind = "blend" if self.cut["kind"] != "cut" else "cut"
            self._move("next", f"NEXT: the {kind} starts in {self.cut['bar'] - self.bar_n} bars")
            return True, f"NEXT: {nxt.track.title[:28]} comes in in {self.cut['bar'] - self.bar_n} bars, the bass swap {self.cut['L']} bars later" if kind == "blend" \
                else f"NEXT: the cut in {self.cut['bar'] - self.bar_n} bars, {nxt.track.title[:28]} whole at its landmark"
        return False, "still decoding: the blend comes the moment it is ready"

    def bad(self):
        """BAD on a blend under way (the user, 2026-10-26: the room sat through 24 bars of two records fighting): the
        bass swap at the next bar and the record out in a beat. Before the blend starts, or with nothing planned, a
        BAD on the record is a NEXT. Returns (ok, msg)."""
        c = self.cut
        if c is not None and c["state"] == "started":
            return self._hurry(tail_bars=0, why="BAD")
        return self.next()

    def _hurry(self, tail_bars=None, why="NEXT"):
        """NEXT while the next record is already in under the record: the bass swap at the next bar + NEXT_BARS and the
        record out over the short tail (`tail_bars` 0: out in a beat - BAD). The queued events are recalled; what has
        sounded stands."""
        c = self.cut
        rec, nxt = self.songs.get(c["a"]), self.songs.get(c["b"])
        if rec is None or nxt is None:
            return False, "nothing planned yet: the record plays on"
        with self._lock:
            track, t_now, bar_s, k_now, open_bar, k_min, k_end = self._geometry(rec)
            k = min(k_now + (1 if tail_bars == 0 else NEXT_BARS), max(k_now + 1, k_end - 1))
            T_S = max(self._clock_of_song_time(rec.deck, self._bar_time(track, k, t_now)), self._clock() + int((LEAD_S + 0.3) * RATE))
            wall_bar = int(round(bar_s / self._rate_now(rec.deck) * RATE))
            beat = wall_bar // 4
            T_tail = SIZES["short"][1] if tail_bars is None else int(tail_bars)
            T_E = T_S + max(beat, T_tail * wall_bar)
            if k + T_tail > k_end - 1:
                T_E = T_S + max(beat, (k_end - 1 - k) * wall_bar)
            self.submix.post({"cmd": "cancel", "txn": c["txn"]})
            self._seq += 1
            txn = f"hurry{self._seq}"
            a, b = rec.deck, nxt.deck
            lead = max(T_S - self._clock(), beat)
            ev = [
                {"cmd": "gain", "deck": b, "value": 1.0, "ramp_s": lead / RATE, "curve": "power", "txn": txn},
                {"cmd": "eq", "deck": b, "mid": 1.0, "high": 1.0, "ramp_s": lead / RATE, "txn": txn},
                {"cmd": "eq", "deck": a, "mid": 0.6, "high": 0.5, "ramp_s": lead / RATE, "txn": txn},
                {"at": T_S, "cmd": "eq", "deck": a, "low": 0.0, "ramp_s": 2 * beat / RATE, "txn": txn},
                {"at": T_S, "cmd": "eq", "deck": b, "low": 1.0, "ramp_s": 2 * beat / RATE, "txn": txn},
                {"at": T_S, "cmd": "gain", "deck": a, "value": 0.0, "ramp_s": (T_E - T_S) / RATE, "curve": "power", "txn": txn},
                {"at": T_E, "cmd": "end_sync", "txn": txn},
                {"at": T_E, "cmd": "rate", "deck": b, "value": 1.0, "ramp_s": GLIDE_S, "txn": txn},
                {"at": T_E + int(0.3 * RATE), "cmd": "stop", "deck": a, "txn": txn},
                {"at": T_E + int(0.4 * RATE), "cmd": "unload", "deck": a, "txn": txn},
            ]
            self.submix.post_many(ev)
            swap_bar = self.bar_n + (k - k_now)
            c.update({"seq": self._seq, "txn": txn, "swap_clock": T_S, "end_clock": T_E, "stop_clock": T_E + int(0.4 * RATE),
                      "swap_bar": swap_bar, "end_bar": swap_bar + T_tail, "why": why, "size": "short",
                      "L": max(1, swap_bar - c["bar"]), "T": T_tail})     # (the lead as it now is: from its start to the hurried swap)
            self._next_asap = False
        out = (f"{why}: the bass swap in {c['swap_bar'] - self.bar_n} bars, {rec.track.title[:24]} out "
               + (f"over {T_tail} more" if T_tail else "in a beat"))
        self._move("next" if why == "NEXT" else "bad", out)
        return True, out

    def stay(self):
        if not self.running or self.record is None:
            return False, "press START" if not self.running else "nothing planned yet: the first record is decoding"
        c = self.cut
        if c is None:
            return False, "nothing planned yet: the record plays on"
        if c["state"] != "planned" or self._clock() >= c["clock"] - int(2 * LEAD_S * RATE):
            return False, "nothing planned yet: the blend is under way - it stands" if c["kind"] != "cut" else "the cut is firing now"
        nxt = self.songs.get(c["b"])
        rec = self.songs.get(self.record)
        if nxt is None or rec is None:
            return False, "nothing planned yet: the record plays on"
        track, t_now, bar_s, k_now, open_bar, k_min, k_end = self._geometry(rec)
        k_start = self._bar_index(track, c["song_t"], t_ref=t_now) + PHRASE_BARS
        if c["kind"] == "cut":
            want_t = c["song_t"] + PHRASE_BARS * bar_s
            if want_t > float(track.duration_s) - 0.5:
                return False, "nothing planned yet: the record has no phrase left to stay for - the cut stands"
            ok = self._plan_cut(nxt, why="STAY: a phrase later", at_song_t=want_t)
        else:
            L, T = SIZES.get(c.get("size"), SIZES["standard"]) if c["kind"] == "blend" else SIZES["short"]
            if k_start + SIZES["short"][0] + SIZES["short"][1] > k_end - 1:
                return False, "nothing planned yet: the record has no phrase left to stay for - the blend stands"
            size = c.get("size") if c["kind"] == "blend" and k_start + L + T <= k_end - 1 else None
            ok = self._plan_blend(nxt, why="STAY: a phrase later", size=size, k_start=k_start)
            if not ok:
                return False, "nothing planned yet: the record has no phrase left to stay for - the blend stands"
        if ok:
            self._move("stay", f"STAY: the {'cut' if self.cut['kind'] == 'cut' else 'blend'} moved a phrase later, in {self.cut['bar'] - self.bar_n} bars")
            return True, f"STAY: the {'cut' if self.cut['kind'] == 'cut' else 'blend'} in {self.cut['bar'] - self.bar_n} bars"
        return False, "nothing planned yet: the record plays on"

    def deck_of(self, track_id):
        for d, r in self.songs.items():
            if r.track.id == int(track_id):
                return d
        return None

    def room_quiet_by_design(self, thr=0.004):
        """True when every record SOUNDING is itself under `thr` (RMS) at its cursor by its own envelope: the record's
        breakdown, not a dropout. None when nothing is known."""
        known = False
        tel = (self._tel().get("decks") or {})
        for dk, r in self.songs.items():
            t = tel.get(dk) or {}
            if not t.get("playing") or float(t.get("gain") or 0.0) <= 0.1 or r.mix_env is None:
                continue
            known = True
            i = int(float(t.get("time_s") or 0.0) * 2)
            if i < len(r.mix_env) and float(r.mix_env[i]) * float(t.get("gain") or 1.0) >= thr:
                return False
        return True if known else None

    def eject(self, track_id):
        d = self.deck_of(track_id)
        if d is None:
            return "not on a deck"
        r = self.songs[d]
        if d == self.record:
            return f"{r.track.title[:30]} is the record playing (press NEXT to move on)"
        if self.cut is not None and self.cut["b"] == d:
            if self.cut["state"] != "planned":
                return f"{r.track.title[:30]} is the record coming in (the blend is under way)"
            self.submix.post({"cmd": "cancel", "txn": self.cut["txn"]})
            self.cut = None
        self.songs.pop(d, None)
        self.submix.post({"cmd": "unload", "deck": d})
        self._move("eject", f"{r.track.title[:30]} off deck {d.upper()}: another next record is picked", from_id=r.track.id)
        return f"{r.track.title[:30]} off the decks"

    # -- the room: what the lights are told (lib/dj/vis.py consumes these; the same vocabulary as DJSystem) ------------------------
    render_lead = None            # callable -> frames the renderer is ahead of the speakers; None = no render-ahead ring

    def _vis_lead(self):
        fn = self.render_lead
        if fn is None:
            return 0
        try:
            return int(fn() or 0)
        except Exception:
            return 0

    def _vis_tel(self):
        """Telemetry as HEARD (the render head runs ahead of the speakers)."""
        try:
            return self.submix.telemetry_delayed(self._vis_lead()) or {}
        except Exception:
            return self._tel()

    def _vis_clock(self):
        return self.submix.clock - self._vis_lead()

    def live_beat(self):
        """The record's beat as the room hears it: bpm, beat phase, bar and phrase phase (phrases counted from where the
        record opened, so they are the blend's phrases), the section's bass share and rhythmic drive. None when nothing
        trustworthy plays (the same bpm_conf bar the brain uses)."""
        r = self.songs.get(self.record) if self.record else None
        if r is None:
            return None
        d = (self._vis_tel().get("decks") or {}).get(r.deck)
        t = r.track
        if not d or not d.get("playing") or not d.get("ready") or (t.bpm_conf or 0.0) < 0.5:
            return None
        pos = float(d.get("time_s") or 0.0)
        rate = max(float(d.get("rate") or 1.0), 1e-6)
        seg = self._seg(t, pos)
        per = float((seg or {}).get("period_s") or t.period_s or 0.0)
        if per <= 0.0:
            return None
        fb = float((seg or {}).get("first_beat_s") or 0.0)
        off = int(t.downbeat_offset or 0)
        idx = (pos - fb) / per - off                      # beats since the first downbeat
        k = int(math.floor(idx / 4.0 + 1e-6))
        bar_phase = (idx / 4.0) - k
        open_bar = r.open_bar if r.open_bar is not None else self._bar_index(t, r.cue_t, t_ref=pos)
        phrase_phase = (((k - open_bar) % PHRASE_BARS) + bar_phase) / float(PHRASE_BARS)
        sec = self._section_at(t, pos)
        drive = max(0.0, min(1.0, float(sec.get("rhythm_density") or 0.0) / 4.0))
        return {"bpm": 60.0 * rate / per, "phase": float(d.get("beat_phase") or 0.0), "bar_phase": bar_phase,
                "phrase_phase": phrase_phase, "bass_share": float(sec.get("bass_share") or 0.33), "drive": drive}

    @staticmethod
    def _section_at(t, pos):
        try:
            return t.section_at(pos) or {}
        except Exception:
            return {}

    def live_energy(self):
        """Ground-truth energy 0..1 of what plays now: each record's library-ranked drive shaped by its own energy curve at
        the playhead, gain-weighted across the decks that sound (a blend reads as the mix of the two). None when nothing plays."""
        tel = self._vis_tel()
        by_id = {r.track.id: r.track for r in self.songs.values()}
        num = den = 0.0
        for d in (tel.get("decks") or {}).values():
            g = float(d.get("gain") or 0.0)
            if not d.get("playing") or g <= 0.02:
                continue
            t = by_id.get(d.get("track_id"))
            if t is None:
                continue
            shape = 1.0
            curve = (t.row.get("energy_curve") or []) if hasattr(t, "row") else []
            if curve:
                i = float(d.get("time_s") or 0.0) * 2.0
                i0 = max(0, min(int(i), len(curve) - 1))
                i1 = min(i0 + 1, len(curve) - 1)
                f = i - i0
                shape = float(curve[i0]) * (1.0 - f) + float(curve[i1]) * f
            try:
                drv = float(t.drive())
            except Exception:
                drv = 0.5
            num += g * drv * shape
            den += g
        if den <= 0.0:
            return None
        lo, hi = self._energy_span()
        return max(0.12, min(1.0, (num / den - lo) / max(hi - lo, 1e-6)))

    def _energy_span(self):
        span = getattr(self, "_energy_lo_hi", None)
        if span is None:
            vals = []
            for t in self.library:
                try:
                    vals.append(float(t.drive()))
                except Exception:
                    pass
            if len(vals) >= 8:
                vals.sort()
                span = (vals[int(0.05 * (len(vals) - 1))], vals[int(0.95 * (len(vals) - 1))])
                if span[1] - span[0] < 0.05:
                    span = (0.0, 1.0)
            else:
                span = (0.0, 1.0)
            self._energy_lo_hi = span
        return span

    def outstate_keys(self):
        """Published into outstate each tick: the room knows the future. dj_blend_eta counts down to the next record
        entering, dj_swap_eta to the bass swap (a blend), the drop (a breakdown swap) or the one (a cut); the record's
        own drops and the swap are stamped as drops (a breakdown swap's and a cut's as HARD ones); a cut's hole is the
        breath-hold."""
        vclock = self._vis_clock()
        c = self.cut
        eta = swap_eta = None
        hole = False
        if c is not None and c["state"] in ("planned", "started", "swapped"):
            if c["state"] == "planned" and c["clock"] > vclock:
                eta = (c["clock"] - vclock) / RATE
            if c["state"] in ("planned", "started") and c["swap_clock"] > vclock:
                swap_eta = (c["swap_clock"] - vclock) / RATE
            if c["kind"] == "cut":
                hole = c.get("hole_clock", c["clock"]) <= vclock < c["clock"]
            # the swap lands: one stamp per plan
            if c["swap_clock"] <= vclock and getattr(self, "_swap_stamped", None) != c["seq"]:
                self._swap_stamped = c["seq"]
                self._dj_drop_wall = time.time()
                self._dj_drop_hard = c["kind"] in ("breakdown", "cut")
        # the record's own drops (the DB knows every one): the next within 20 s, and a stamp when the playhead crosses one
        drop_eta = None
        r = self.songs.get(self.record) if self.record else None
        d = (self._vis_tel().get("decks") or {}).get(r.deck) if r is not None else None
        if r is not None and d and d.get("playing"):
            pos = float(d.get("time_s") or 0.0)
            cid, moments = getattr(self, "_drops_cache", (None, []))
            if cid != r.track.id:
                try:
                    from lib.dj.features import drop_moments
                    moments = drop_moments(r.track.sections or [])
                except Exception:
                    moments = []
                self._drops_cache = (r.track.id, moments)
            prev_id, prev_pos = getattr(self, "_drop_scan_prev", (None, None))
            rate = max(float(d.get("rate") or 1.0), 1e-6)
            for st in moments:
                dd = (st - pos) / rate
                if 0.0 < dd < 20.0 and (drop_eta is None or dd < drop_eta):
                    drop_eta = dd
                if prev_id == r.track.id and prev_pos is not None and prev_pos < st <= pos and pos - prev_pos < 2.0:
                    self._dj_drop_wall = time.time()
                    self._dj_drop_hard = False
            self._drop_scan_prev = (r.track.id, pos)
        ndrop = swap_eta
        if drop_eta is not None:
            ndrop = drop_eta if ndrop is None else min(ndrop, drop_eta)
        return {"dj_active": self.running,
                "dj_energy": self.live_energy(),
                "dj_drop_t": getattr(self, "_dj_drop_wall", None),
                "dj_drop_hard": bool(getattr(self, "_dj_drop_hard", False)),
                "dj_next_drop_eta": ndrop,
                "dj_moment_eta": None,
                "dj_moment_hole": hole,
                "dj_blend_eta": eta,
                "dj_swap_eta": swap_eta,
                "dj_style": c["kind"] if c is not None else None}

    # -- words ------------------------------------------------------------------------------------------------------------------
    def _move(self, kind, text, from_id=None, to_id=None):
        self.moves.append({"t": time.time(), "kind": kind, "text": text, "from_id": from_id, "to_id": to_id})
        if len(self.moves) > 400:
            del self.moves[:-400]
        try:
            # the night log, so a BAD can be read the next day (the night of 2026-10-26 left only two BADs and two
            # 'played' rows; the transition had to be replayed to be understood); silent under DJ_NO_PLAY_LOG
            from lib.dj.session import log_row
            log_row({"event": "record_move", "kind": kind, "text": text[:200], "from_id": from_id, "to_id": to_id})
        except Exception:
            pass

    def plan_words(self, c=None):
        """One line saying what the plan does (the tab's row)."""
        c = c or self.cut
        if c is None:
            return ""
        if c["kind"] == "cut":
            return "comes in whole at its landmark · the cut on the one, a beat of nothing before it"
        if c["kind"] == "breakdown":
            return f"starts under the breakdown, bass out · its drop lands at bar {c['swap_bar']} where the breakdown resolves; the record goes out in a beat"
        return (f"starts bass-out at bar {c['bar']}, fader up, highs then mids · THE BASS SWAP at bar {c['swap_bar']} · "
                f"the record out by bar {c['end_bar']} ({c['L'] + c['T']} bars together, {c['size']})")

    def status(self):
        out = {"record": None, "next": None, "cut": None, "bar": self.bar_n, "bar_s": self.bar_s(), "wait_why": self.wait_why,
               "moves": [(time.strftime("%H:%M:%S", time.localtime(m["t"])), m["text"]) for m in self.moves[-40:]], "error": self.last_error}
        rec = self.songs.get(self.record) if self.record else None
        if rec is not None:
            t_now = self._song_time_now(rec.deck)
            played = max(0.0, t_now - rec.cue_t)
            out["record"] = {"title": rec.track.title, "artist": rec.track.artist or "", "track_id": rec.track.id, "time_s": t_now,
                             "duration_s": float(rec.track.duration_s), "bpm": float(rec.track.bpm), "camelot": rec.track.camelot or "",
                             "played_s": played, "min_left_s": max(0.0, self.play_min_s - played), "deck": rec.deck, "rate": self._rate_now(rec.deck)}
        nxt = self._next_rec()
        c = self.cut
        if nxt is None and c is not None and c["state"] == "started":
            nxt = self.songs.get(c["b"])                 # (in under the record, not yet the record)
        if nxt is not None:
            out["next"] = {"title": nxt.track.title, "artist": nxt.track.artist or "", "track_id": nxt.track.id, "state": nxt.state, "deck": nxt.deck,
                           "landmark_s": nxt.cue_t}
        if c is not None:
            in_s = max(0.0, (c["clock"] - self._clock()) / RATE)
            bdeck = c["b"]
            out["cut"] = {"in_s": in_s, "in_bars": c["bar"] - self.bar_n, "bar": c["bar"], "why": c["why"], "seq": c["seq"],
                          "state": c["state"], "kind": c["kind"], "size": c.get("size"), "bars": c["L"] + c["T"],
                          "lead_bars": c["L"], "tail_bars": c["T"],
                          "swap_in_s": max(0.0, (c["swap_clock"] - self._clock()) / RATE), "swap_bar": c["swap_bar"], "end_bar": c["end_bar"],
                          "end_in_s": max(0.0, (c["end_clock"] - self._clock()) / RATE),
                          "ratio": c.get("ratio", 1.0), "words": self.plan_words(c),
                          "a_track_id": self.songs[c["a"]].track.id if c["a"] in self.songs else None,
                          "b_track_id": self.songs[bdeck].track.id if bdeck in self.songs else None}
        elif rec is not None and nxt is not None:
            out["cut"] = self._project(rec, nxt)
        # THE DECKS as heard (the page's meters): each deck's fader, its three bands, what it holds and its role
        decks = {}
        tel = (self._tel().get("decks") or {})
        for dk in ("a", "b"):
            t = tel.get(dk) or {}
            r = self.songs.get(dk)
            role = None
            if r is not None:
                role = ("record" if dk == self.record else
                        ("leaving" if r.state == "leaving" else ("in" if r.state == "playing" else r.state)))
            eq = t.get("eq") or [1.0, 1.0, 1.0]
            decks[dk] = {"title": r.track.title if r is not None else None, "track_id": r.track.id if r is not None else None,
                         "role": role, "playing": bool(t.get("playing")), "gain": round(float(t.get("gain") or 0.0), 3),
                         "low": round(float(eq[0]), 3), "mid": round(float(eq[1]), 3), "high": round(float(eq[2]), 3),
                         "rate": round(float(t.get("rate") or 1.0), 4), "time_s": float(t.get("time_s") or 0.0)}
        out["decks"] = decks
        return out

    def _project(self, rec, nxt):
        """THE PLAN before it is decided (the next record still decoding; the tempo settling): where the blend WILL start
        if nothing is pressed - the record's first phrase boundary at or after its minimum play, two bars ahead at least."""
        try:
            track, t_now, bar_s, k_now, open_bar, k_min, k_end = self._geometry(rec)
            found = self._find_blend(rec, nxt) if (nxt.state == "ready" or nxt.vox is not None) else None
            if found is None and (nxt.state == "ready" or nxt.vox is not None):
                found = self._find_blend(rec, nxt, relax=True)
            if found is not None:
                K, L, T = found["K"], found["L"], found["T"]
                in_s = max(0.0, self._bar_time(track, K, t_now) - t_now)
                return {"in_s": in_s, "in_bars": K - k_now, "bar": self.bar_n + (K - k_now), "seq": None, "state": "projected",
                        "kind": "blend", "size": found["size"], "bars": L + T,
                        "swap_in_s": in_s + L * bar_s, "swap_bar": self.bar_n + (K - k_now) + L, "end_bar": self.bar_n + (K - k_now) + L + T,
                        "ratio": 1.0, "words": f"the {L + T}-bar blend ({found['size']}): {found['why']}" + ("" if nxt.state == "ready" else " (decided when it is ready)"),
                        "why": "placed by the records" + ("" if nxt.state == "ready" else f" ({nxt.track.title[:20]} still decoding)"),
                        "b_track_id": nxt.track.id}
            k_last = int(math.ceil((LAST_FROM * float(track.duration_s) - self._first_down(track, t_now)) / bar_s))
            K = self._phrase_up(max(k_now + CUT_MIN_BARS, k_min, k_last), open_bar)
            K = min(K, max(k_now + 1, k_end))
            in_s = max(0.0, self._bar_time(track, K, t_now) - t_now)
            known = rec.vox is not None and (nxt.vox is not None or nxt.state == "ready")
            return {"in_s": in_s, "in_bars": K - k_now, "bar": self.bar_n + (K - k_now), "seq": None, "state": "projected",
                    "kind": "cut", "size": "cut", "bars": 0, "swap_in_s": in_s, "swap_bar": self.bar_n + (K - k_now), "end_bar": self.bar_n + (K - k_now),
                    "ratio": 1.0,
                    "words": ("the cut: no place keeps the two records' vocals apart for a blend" if known else "the blend is placed when both records' vocals are measured"),
                    "why": ("its minimum play is done then" if k_min > k_now else "its minimum play is done") + ("" if nxt.state == "ready" else f" ({nxt.track.title[:20]} still decoding)"),
                    "b_track_id": nxt.track.id}
        except Exception:
            return None
