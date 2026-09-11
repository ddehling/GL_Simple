"""The pair stage: two songs, four stem lanes, each lane A / B / off.

The simplest form of the stem stage, and the one a DJ actually plays: song
A is the clock and the key, song B runs beat-locked to it (stretched inside
the deck's wall, key-shifted toward A's key within three semitones), and
each of the four stems - drums, bass, other, vocals - is switched between A,
B or nothing on the next bar with a one-beat crossfade. Every combination is
a state you can hold: A's drums under B's bass under A's vocal. MORPH runs
the default handover order (drums, bass, other, vocals) at a chosen spacing;
each song can loop a few bars of where it is.

Both songs play on the submix's two decks (one PLL slave: the path every
night runs), stems on the decks themselves (Deck.set_stem_gains), so
nothing here is new machinery - only the choreography is.
"""
import math

from lib.dj.submix import DJSubmix, RATE

STEMS = ("drums", "bass", "other", "vocals")
LEAD_S = 0.25
RATE_MIN, RATE_MAX = 0.90, 1.10
KEY_SHIFT_MAX = 3
XFADE_BEATS = 1.0                 # a stem switch crosses over this many beats
MORPH_ORDER = ("drums", "bass", "other", "vocals")


def _camelot_compat(c1, c2):
    from lib.dj.brain import camelot_compat
    return camelot_compat(c1, c2)


def _shift_camelot(cam, s):
    from lib.dj.brain import _shift_camelot as f
    return f(cam, s)


class PairStage:
    def __init__(self, db, music_root):
        self.db = db
        self.music_root = music_root
        self.submix = DJSubmix(deck_names=("a", "b"))
        self.track = {"a": None, "b": None}
        self.n = {"a": 0, "b": 0}
        self.where = {s: None for s in STEMS}     # which song each stem plays from: "a" | "b" | None
        self.gain = {s: 1.0 for s in STEMS}
        self.loop_bars = {"a": None, "b": None}
        self.playing = False
        self.rate_b = 1.0
        self.shift_b = 0
        self.compat = None
        self.log = []

    # -- material --------------------------------------------------------------
    def load(self, deck, track):
        """Decode `track` and its stems onto deck a or b (call off the audio thread)."""
        from lib.dj.features import decode_file_stereo
        from lib.dj.stems import load_stems
        samples = decode_file_stereo(self.db.abs(track.path))
        stems = load_stems(self.music_root, track.id, expected_len=len(samples)) if getattr(track, "has_stems", False) else None
        if not stems:
            raise ValueError(f"{track.title}: no stems on disk")
        self.submix.post_many([
            {"cmd": "unload", "deck": deck},
            {"cmd": "load", "deck": deck, "samples": samples, "grid": track.grid, "track_id": track.id,
             "gain_db": track.gain_db, "stems": stems, "cue_s": 0.0},
            {"cmd": "gain", "deck": deck, "value": 0.0, "ramp_s": 0.01},
            {"cmd": "stem_gains", "deck": deck, "gains": {s: 0.0 for s in STEMS}, "ramp_s": 0.01},
        ])
        self.track[deck], self.n[deck] = track, len(samples)
        self._note(f"{deck.upper()}: {track.title} loaded ({track.bpm:.1f} bpm, {track.camelot or '?'})")

    # -- clock helpers -----------------------------------------------------------
    def _seg(self, track, t):
        for seg in track.grid or []:
            if seg["start_s"] <= t <= seg["end_s"]:
                return seg
        return (track.grid or [None])[0]

    def bar_s(self, track, t=0.0):
        seg = self._seg(track, t)
        return 4 * (seg["period_s"] if seg else track.period_s)

    def body_start(self, track):
        """Where a song's body begins: its first groove with bass, else its first non-intro section, else 0."""
        secs = track.sections or []
        for s in secs:
            if s.get("kind") == "groove" and s.get("bass_share", 0.3) >= 0.28:
                return track.nearest_downbeat(s["start_s"])
        for s in secs:
            if s.get("kind") not in ("intro", "outro"):
                return track.nearest_downbeat(s["start_s"])
        return track.nearest_downbeat(0.0)

    def next_bar_clock(self):
        tel = self.submix.telemetry or {}
        now = int(tel.get("clock", self.submix.clock))
        d = (tel.get("decks") or {}).get("a") or {}
        ta = self.track["a"]
        if not (self.playing and d.get("playing") and ta is not None):
            return now + int(LEAD_S * RATE)
        time_s, rate = float(d.get("time_s", 0.0)), max(float(d.get("rate", 1.0)), 1e-6)
        seg = self._seg(ta, time_s)
        if seg is None:
            return now + int(LEAD_S * RATE)
        bar = 4 * seg["period_s"]
        first_down = seg["first_beat_s"] + ta.downbeat_offset * seg["period_s"]
        t_min = time_s + LEAD_S * rate
        k = math.ceil((t_min - first_down) / bar - 1e-6)
        t_next = first_down + k * bar
        lp = d.get("loop")
        if lp and t_next * RATE >= lp[1] - 1:
            t_next = lp[1] / RATE
        return now + int((t_next - time_s) / rate * RATE)

    def beat_s(self):
        ta = self.track["a"]
        return (self.bar_s(ta) / 4.0) if ta is not None else 0.5

    # -- transport ---------------------------------------------------------------
    def play(self, stems_from="a"):
        """Start both songs at their bodies, A as the clock, B beat-locked and silent (its stems closed);
        the four lanes come up from `stems_from` ("a" / "b")."""
        ta, tb = self.track["a"], self.track["b"]
        if ta is None:
            return False
        at = self.submix.clock + int(LEAD_S * RATE)
        ev = [
            {"at": at, "cmd": "cue", "deck": "a", "time_s": self.body_start(ta)},
            {"at": at, "cmd": "rate", "deck": "a", "value": 1.0},
            {"at": at, "cmd": "eq", "deck": "a", "low": 1.0, "mid": 1.0, "high": 1.0, "ramp_s": 0.01},
            {"at": at, "cmd": "stem_gains", "deck": "a", "gains": {s: (1.0 if stems_from == "a" else 0.0) for s in STEMS}, "ramp_s": 0.01},
            {"at": at, "cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.05},
            {"at": at, "cmd": "start", "deck": "a"},
        ]
        if tb is not None:
            rate = ta.bpm / max(tb.bpm, 1e-6)
            if not (RATE_MIN <= rate <= RATE_MAX):
                self._note(f"B refused: {tb.bpm:.1f} bpm against {ta.bpm:.1f} needs rate {rate:.3f}")
                tb = None
        if tb is not None:
            self.rate_b = rate
            self.shift_b, self.compat = self._key_shift(ta, tb)
            ev += [
                {"at": at, "cmd": "cue", "deck": "b", "time_s": self.body_start(tb)},
                {"at": at, "cmd": "rate", "deck": "b", "value": rate},
                {"at": at, "cmd": "pitch", "deck": "b", "semitones": float(self.shift_b)},
                {"at": at, "cmd": "eq", "deck": "b", "low": 1.0, "mid": 1.0, "high": 1.0, "ramp_s": 0.01},
                {"at": at, "cmd": "stem_gains", "deck": "b", "gains": {s: (1.0 if stems_from == "b" else 0.0) for s in STEMS}, "ramp_s": 0.01},
                {"at": at, "cmd": "gain", "deck": "b", "value": 1.0, "ramp_s": 0.05},
                {"at": at, "cmd": "start", "deck": "b"},
                {"at": at, "cmd": "sync", "slave": "b", "master": "a", "bias_beats": 0.0, "audio_pll": True},
            ]
        self.submix.post_many(ev)
        self.playing = True
        self.where = {s: (stems_from if (stems_from == "a" or tb is not None) else "a") for s in STEMS}
        self._note(f"playing: A {ta.title}" + (f" + B {tb.title} at rate {self.rate_b:.3f}, shift {self.shift_b:+d} st, key fit {self.compat:.2f}" if tb is not None else ""))
        return True

    def stop(self):
        now = self.submix.clock + int(0.05 * RATE)
        ev = []
        for d in ("a", "b"):
            ev += [{"at": now, "cmd": "gain", "deck": d, "value": 0.0, "ramp_s": 0.3},
                   {"at": now + int(0.4 * RATE), "cmd": "stop", "deck": d},
                   {"at": now + int(0.4 * RATE), "cmd": "clear_loop", "deck": d}]
        ev.append({"at": now + int(0.4 * RATE), "cmd": "end_sync"})
        self.submix.post_many(ev)
        self.playing = False
        self._note("stopped")

    # -- the lanes -----------------------------------------------------------------
    def set_stem(self, stem, where, at=None):
        """Switch one stem lane to song "a", "b" or None (off) on the next bar, crossing over XFADE_BEATS."""
        if stem not in STEMS or where not in ("a", "b", None):
            return None
        if where == "b" and self.track["b"] is None:
            return None
        at = self.next_bar_clock() if at is None else at
        xf = XFADE_BEATS * self.beat_s()
        g = self.gain[stem]
        ev = []
        for d in ("a", "b"):
            if self.track[d] is None:
                continue
            ev.append({"at": at, "cmd": "stem_gains", "deck": d, "gains": {stem: (g if where == d else 0.0)}, "ramp_s": xf})
        self.submix.post_many(ev)
        self.where[stem] = where
        self._note(f"{stem} -> {where.upper() if where else 'off'} at bar clock {at}")
        return at

    def set_gain(self, stem, g):
        self.gain[stem] = float(g)
        w = self.where.get(stem)
        if w and self.playing:
            self.submix.post({"cmd": "stem_gains", "deck": w, "gains": {stem: float(g)}, "ramp_s": 0.1})

    def morph(self, to="b", order=MORPH_ORDER, beats_apart=8):
        """Hand the lanes over to song `to` one by one, `beats_apart` beats between them, from the next bar."""
        if to not in ("a", "b") or self.track[to] is None:
            return None
        at = self.next_bar_clock()
        step = int(beats_apart * self.beat_s() * RATE)
        for k, stem in enumerate(order):
            if self.where.get(stem) != to:
                self.set_stem(stem, to, at=at + k * step)
        self._note(f"morph -> {to.upper()}: {', '.join(order)} every {beats_apart} beats")
        return at

    def loop(self, deck, bars):
        """Loop song `deck` on `bars` bars from its current bar (None = release the loop on the next bar)."""
        t = self.track[deck]
        if t is None or not self.playing:
            return None
        at = self.next_bar_clock()
        if bars is None:
            self.submix.post({"at": at, "cmd": "release_loop", "deck": deck})
            self.loop_bars[deck] = None
            self._note(f"{deck.upper()}: loop released")
            return at
        tel = self.submix.telemetry or {}
        d = (tel.get("decks") or {}).get(deck) or {}
        time_s, rate = float(d.get("time_s", 0.0)), max(float(d.get("rate", 1.0)), 1e-6)
        # the loop starts on this song's own next downbeat (its time, its grid)
        start = t.nearest_downbeat(time_s + (at - int(tel.get("clock", self.submix.clock))) / RATE * rate)
        bar = self.bar_s(t, start)
        end = start + int(bars) * bar
        self.submix.post({"at": at, "cmd": "loop", "deck": deck, "start_s": start, "end_s": end})
        self.loop_bars[deck] = int(bars)
        self._note(f"{deck.upper()}: loop {bars} bars from {start:.1f}s")
        return at

    # -- the harmonic guard --------------------------------------------------------
    def _key_shift(self, ta, tb):
        if not ta.camelot or not tb.camelot:
            return 0, None
        best = (0, _camelot_compat(ta.camelot, tb.camelot))
        for s in (1, -1, 2, -2, 3, -3):
            c = _camelot_compat(ta.camelot, _shift_camelot(tb.camelot, s))
            if c > best[1] + 1e-9:
                best = (s, c)
            if best[1] >= 0.9:
                break
        return best

    # -- state -----------------------------------------------------------------------
    def status(self):
        tel = self.submix.telemetry or {}
        decks = tel.get("decks") or {}
        sync = tel.get("sync") or {}
        out = {"playing": self.playing, "where": dict(self.where), "gain": dict(self.gain), "loop_bars": dict(self.loop_bars),
               "rate_b": self.rate_b, "shift_b": self.shift_b, "compat": self.compat,
               "lock_ms": None, "decks": {}}
        for d in ("a", "b"):
            t = decks.get(d) or {}
            out["decks"][d] = {"title": self.track[d].title if self.track[d] else None, "playing": bool(t.get("playing")),
                               "time_s": t.get("time_s"), "beat_phase": t.get("beat_phase"), "loop": t.get("loop")}
        # the lock the PLL holds: B's grid phase against A's, the kick bias removed (the wide audible
        # meter in the telemetry measures rhythm-pattern offset and is not this)
        a, b = decks.get("a") or {}, decks.get("b") or {}
        if sync and a.get("playing") and b.get("playing") and a.get("beat_phase") is not None and b.get("beat_phase") is not None:
            err = (float(b["beat_phase"]) - float(a["beat_phase"]) - float(sync.get("bias_beats") or 0.0) + 0.5) % 1.0 - 0.5
            out["lock_ms"] = abs(err) * self.beat_s() * 1000.0
        return out

    def _note(self, msg):
        self.log.append(msg)
        if len(self.log) > 200:
            self.log = self.log[-200:]
