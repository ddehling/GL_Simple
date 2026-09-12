"""The timeline: one continuous run of bars with four lane tracks - drums, bass, other, vocals - and
CLIPS on them: a span of one song's stem placed at a bar. One clip per lane at a time, so there is never
a mix of stems to balance: levels, beat lock, key shift and crossfades are the conductor's (lib/dj/remix.py).

    Clip      one lane of one song from song time `start_s`, starting at timeline bar `bar`, until
              `end_bar` (None = until the next clip on that lane takes over - the DJ's "until further
              notice"). Precedence is IMPLICIT: at any bar the latest-starting clip that covers it is
              heard, so a clip dropped over a running one takes over, and when a bounded clip ends the
              earlier one resumes; drag a clip away and nothing is left trimmed. `ghost` marks a clip the
              AUTOPILOT planned ahead of the playhead - played like any other unless you delete or move it.
    Timeline  the clips, segments() for drawing, save / load
    Spectro   per-stem spectrograms (48 log bands, 0.1 s hop, uint8) computed from the stems once and
              cached beside them - the viewer's and the clips' picture of the music
    TimelinePlayer  walks the timeline on the conductor (autopilot 0 inside the conductor: the timeline is
              the only thing that moves): stages songs so a clip's song time lands on its bar, puts lanes on
              decks at the bar, re-cues a live song for a clip elsewhere in it, ejects songs no clip needs.
              Its own AUTOPILOT plans ghost clips a horizon ahead - the conductor's move policy (new song
              through one lane, a lane crossing, a rest) written onto the timeline where you can see it
              coming and change it - and its GESTURES (next song, drop, break) write clips the same way.

Timeline bars are the conductor's `bar_n` plus the bar play started from. Every song advances one song
bar per timeline bar (they are beat-locked), so the song time at bar b of a clip is
start_s + (b - clip.bar) * bar_length(song).
"""
import json
import os
import random
import threading
import time

import numpy as np

LANES = ("drums", "bass", "other", "vocals")
MORPH_ORDER = ("drums", "bass", "other", "vocals")
STAGE_AHEAD_BARS = 10           # a clip's song is staged this many bars before the clip (decode + PLL settle)
EJECT_AFTER_BARS = 8            # a song with no clip for this long ahead leaves
HORIZON_PHRASES = 2             # the autopilot plans this many phrases ahead of the playhead


class Clip:
    _seq = 0

    def __init__(self, track_id, lane, bar, start_s, end_bar=None, cid=None, ghost=False, rest=False):
        Clip._seq += 1
        self.id = cid if cid is not None else Clip._seq
        Clip._seq = max(Clip._seq, self.id)
        self.track_id = int(track_id)
        self.lane = lane
        self.bar = int(bar)
        self.start_s = float(start_s)
        self.end_bar = None if end_bar is None else int(end_bar)
        self.ghost = bool(ghost)
        self.rest = bool(rest)           # a REST: the lane is silent for this span (the song id is who rests)

    def active_at(self, b):
        return self.bar <= b and (self.end_bar is None or b < self.end_bar)

    def song_time_at(self, b, bar_s):
        return self.start_s + (b - self.bar) * bar_s

    def to_dict(self):
        return {"id": self.id, "track_id": self.track_id, "lane": self.lane, "bar": self.bar,
                "start_s": round(self.start_s, 3), "end_bar": self.end_bar, "ghost": self.ghost, "rest": self.rest}

    @classmethod
    def from_dict(cls, d):
        return cls(d["track_id"], d["lane"], d["bar"], d["start_s"], d.get("end_bar"), cid=d.get("id"),
                   ghost=d.get("ghost", False), rest=d.get("rest", False))


class Timeline:
    def __init__(self, name="untitled"):
        self.name = name
        self.clips = []

    # -- rules -----------------------------------------------------------------------------------------------
    def on_lane(self, lane):
        return sorted((c for c in self.clips if c.lane == lane), key=lambda c: (c.bar, c.id))

    def active(self, lane, b):
        """The clip heard on `lane` at bar b: the latest-starting clip that covers b (ties: the newest)."""
        best = None
        for c in self.on_lane(lane):
            if c.active_at(b) and (best is None or (c.bar, c.id) >= (best.bar, best.id)):
                best = c
        return best

    def segments(self, lane, b0, b1):
        """[(clip, from_bar, to_bar)] as heard on `lane` between bars b0 and b1 (bar resolution)."""
        out = []
        cur, start = None, b0
        for b in range(int(b0), int(b1) + 1):
            c = self.active(lane, b)
            if c is not cur:
                if cur is not None:
                    out.append((cur, start, b))
                cur, start = c, b
        if cur is not None:
            out.append((cur, start, int(b1) + 1))
        return out

    def add(self, track_id, lanes, bar, start_s, end_bar=None, ghost=False):
        """Place a span of a song on lanes at a bar. It takes over from `bar` (implicit precedence); a
        clip already starting at the very same bar on that lane is replaced."""
        made = []
        for ln in lanes:
            for c in list(self.on_lane(ln)):
                if c.bar == int(bar):
                    self.clips.remove(c)
            clip = Clip(track_id, ln, bar, start_s, end_bar, ghost=ghost)
            self.clips.append(clip)
            made.append(clip)
        return made

    def resolve(self, clip):
        """Kept for callers: with implicit precedence there is nothing to trim; only a duplicate start on
        the lane goes."""
        for c in list(self.on_lane(clip.lane)):
            if c is not clip and c.bar == clip.bar:
                self.clips.remove(c)

    def move(self, clip, new_bar):
        """Shift a clip (and its end) to start at `new_bar`, in place."""
        delta = int(new_bar) - clip.bar
        if delta == 0 or clip not in self.clips:
            return
        clip.bar += delta
        if clip.end_bar is not None:
            clip.end_bar += delta
        self.resolve(clip)

    def remove(self, clip):
        if clip in self.clips:
            self.clips.remove(clip)

    def clear_ghosts(self, after_bar=None):
        self.clips = [c for c in self.clips if not (c.ghost and (after_bar is None or c.bar >= after_bar))]

    def last_bar(self):
        ends = [c.end_bar if c.end_bar is not None else c.bar + 1 for c in self.clips]
        return max(ends) if ends else 0

    def tracks(self):
        return sorted({c.track_id for c in self.clips})

    # -- files -------------------------------------------------------------------------------------------------
    @staticmethod
    def folder(music_root):
        p = os.path.join(music_root, "timelines")
        os.makedirs(p, exist_ok=True)
        return p

    def save(self, music_root, name=None):
        self.name = name or self.name
        path = os.path.join(self.folder(music_root), f"{self.name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": self.name, "clips": [c.to_dict() for c in self.clips], "saved": time.time()}, f, indent=1)
        return path

    @classmethod
    def load(cls, music_root, name):
        path = os.path.join(cls.folder(music_root), f"{name}.json")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        tl = cls(d.get("name") or name)
        tl.clips = [Clip.from_dict(c) for c in d.get("clips", [])]
        return tl

    @classmethod
    def names(cls, music_root):
        p = cls.folder(music_root)
        return sorted(f[:-5] for f in os.listdir(p) if f.endswith(".json"))


# ---------------------------------------------------------------------------------------------------------------
# spectrograms
# ---------------------------------------------------------------------------------------------------------------
class Spectro:
    """Per-stem spectrograms: 48 log-spaced bands from 40 Hz to 16 kHz, a frame every 0.1 s, uint8 dB scale
    (top = the stem's 99th percentile, 60 dB below = 0). Computed once per song from the stems on disk and
    cached as spec.npz beside them; the viewer and the clips draw from it."""
    BANDS = 48
    HOP_S = 0.1
    NFFT = 4096
    RATE = 44100

    def __init__(self, music_root):
        self.music_root = music_root
        self.cache = {}                  # track_id -> {"hop_s", stems: {name: uint8 [bands, frames]}}
        self._lock = threading.Lock()
        self._queue = []
        self._busy = set()
        self._thread = None
        self.errors = {}

    def path(self, track_id):
        return os.path.join(self.music_root, ".stems", str(track_id), "spec.npz")

    def get(self, track_id):
        return self.cache.get(track_id)

    def request(self, track):
        """Have the song's spectrogram ready (disk cache, else computed in the background)."""
        tid = track.id
        if tid in self.cache or tid in self._busy:
            return
        with self._lock:
            if track not in self._queue:
                self._queue.append(track)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._work, daemon=True, name="spectro")
                self._thread.start()

    def _work(self):
        while True:
            with self._lock:
                if not self._queue:
                    return
                track = self._queue.pop(0)
                self._busy.add(track.id)
            try:
                spec = self._load_or_compute(track)
                self.cache[track.id] = spec
            except Exception as e:  # noqa: BLE001
                self.errors[track.id] = f"{type(e).__name__}: {e}"
            finally:
                self._busy.discard(track.id)

    def _load_or_compute(self, track):
        p = self.path(track.id)
        if os.path.exists(p):
            z = np.load(p)
            return {"hop_s": float(z["hop_s"]), "stems": {k[5:]: z[k] for k in z.files if k.startswith("stem_")}}
        from lib.dj.stems import load_stems
        stems = load_stems(self.music_root, track.id)
        if not stems:
            raise ValueError("no stems on disk")
        out = {}
        for name, arr in stems.items():
            out[name] = self.compute(np.asarray(arr))
        try:
            np.savez_compressed(p, hop_s=self.HOP_S, **{f"stem_{k}": v for k, v in out.items()})
        except Exception:
            pass
        return {"hop_s": self.HOP_S, "stems": out}

    @classmethod
    def compute(cls, samples):
        """uint8 [BANDS, frames] for a (n, 2) or (n,) array at 44.1 kHz."""
        x = samples.astype(np.float32)
        if x.ndim == 2:
            x = x.mean(axis=1)
        hop = int(cls.HOP_S * cls.RATE)
        n = max(0, (len(x) - cls.NFFT) // hop + 1)
        if n <= 0:
            return np.zeros((cls.BANDS, 1), dtype=np.uint8)
        win = np.hanning(cls.NFFT).astype(np.float32)
        freqs = np.fft.rfftfreq(cls.NFFT, 1.0 / cls.RATE)
        edges = np.geomspace(40.0, 16000.0, cls.BANDS + 1)
        idx = np.searchsorted(freqs, edges)
        out = np.zeros((cls.BANDS, n), dtype=np.float32)
        step = 400                                   # frames per batch (memory)
        for f0 in range(0, n, step):
            f1 = min(n, f0 + step)
            frames = np.stack([x[i * hop:i * hop + cls.NFFT] for i in range(f0, f1)]) * win
            mag = np.abs(np.fft.rfft(frames, axis=1)) ** 2
            for b in range(cls.BANDS):
                lo, hi = idx[b], max(idx[b] + 1, idx[b + 1])
                out[b, f0:f1] = mag[:, lo:hi].mean(axis=1)
        db = 10.0 * np.log10(out + 1e-10)
        top = float(np.percentile(db, 99.0))
        img = np.clip((db - (top - 60.0)) / 60.0, 0.0, 1.0) * 255.0
        return img.astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------
# the player
# ---------------------------------------------------------------------------------------------------------------
class TimelinePlayer:
    """Walks a Timeline on a RemixConductor, and - with its autopilot up - writes the conductor's next
    moves onto the timeline as ghost clips a horizon ahead, where they can be seen and changed."""

    def __init__(self, rc, timeline, seed=None):
        self.rc = rc
        self.tl = timeline
        self.play_from = 0
        self.last_bar = None
        self.notes = []
        self.running = False
        self._jumped = set()             # clip ids re-cued already
        self.auto = 0.0                  # the autopilot amount: chance a phrase gets a planned ghost move
        self.hold = False                # no new ghosts while held
        self.change_bars = 8             # a phrase
        self._planned_to = None          # the bar the autopilot has planned up to
        self.rng = random.Random(seed)

    # -- geometry ---------------------------------------------------------------------------------------------
    def bar_s(self, track_id):
        t = self.rc._track(track_id)
        return self.rc._bar_s(t) if t is not None else 2.0

    def bar(self):
        return self.play_from + self.rc.bar_n

    def _phrase_after(self, b):
        """The next phrase boundary strictly after bar b (phrases counted from play_from)."""
        rel = b - self.play_from
        return self.play_from + (rel // self.change_bars + 1) * self.change_bars

    # -- lifecycle ---------------------------------------------------------------------------------------------
    def start(self, play_from=0):
        """Open on the song that holds most lanes at `play_from`, cued so the timeline's clips are exact;
        with nothing placed at all, the autopilot opens on the conductor's own pick."""
        self.play_from = int(play_from)
        active = {ln: self.tl.active(ln, self.play_from) for ln in LANES}
        held = [c for c in active.values() if c is not None]
        if not held:
            first = min(self.tl.clips, key=lambda c: (c.bar, c.id), default=None)
            if first is None:
                if self.auto <= 0.0:
                    return False, "the timeline is empty - place something, or raise the autopilot"
                t = self.rc._pick_first()
                start_s = self.rc._landmark(t)
                self.tl.add(t.id, list(LANES), self.play_from, start_s, ghost=True)
                ok = self.rc.start(first_track=t, threaded=False, cue_s=start_s, lanes=set(LANES))
            else:
                t = self.rc._track(first.track_id)
                cue = max(0.0, first.start_s - (first.bar - self.play_from) * self.bar_s(first.track_id))
                ok = self.rc.start(first_track=t, threaded=False, cue_s=cue, lanes=set())
        else:
            counts = {}
            for c in held:
                counts[c.track_id] = counts.get(c.track_id, 0) + 1
            tid = max(counts, key=counts.get)
            c0 = next(c for c in held if c.track_id == tid)
            t = self.rc._track(tid)
            cue = c0.song_time_at(self.play_from, self.bar_s(tid))
            lanes = {ln for ln, c in active.items() if c is not None and c.track_id == tid}
            ok = self.rc.start(first_track=t, threaded=False, cue_s=max(0.0, cue), lanes=lanes)
        self.rc.set_auto(0.0)
        self.running = bool(ok)
        return ok, (self.rc.last_error if not ok else "playing")

    # -- the walk ------------------------------------------------------------------------------------------------
    def step(self):
        """Call every tick (the conductor's own step runs too): the autopilot's plan, staging ahead, the
        lane map at each bar, re-cues, ejects."""
        rc = self.rc
        rc.step()
        if rc.master is None or not self.running:
            return
        b = self.bar()
        self._plan(b)
        # stage songs whose clips are coming (or already due) and are not on a deck
        wanted = {}
        for c in self.tl.clips:
            if c.bar <= b + STAGE_AHEAD_BARS and (c.end_bar is None or c.end_bar > b):
                if rc.deck_of(c.track_id) is None:
                    prev = wanted.get(c.track_id)
                    if prev is None or c.bar < prev.bar:
                        wanted[c.track_id] = c
        for tid, c in wanted.items():
            at_bar = max(c.bar, b + 2)
            st = c.song_time_at(at_bar, self.bar_s(tid))
            ok, msg = rc.stage_for_bar(tid, st, at_bar - self.play_from)
            if ok and msg not in ("already on a deck",):
                self._note(f"staging {rc._track(tid).title[:30]} for bar {at_bar}")
        if self.last_bar == b:
            return
        self.last_bar = b
        # re-cue live songs for clips that start next bar elsewhere in the song
        for c in self.tl.clips:
            if c.bar == b + 1 and c.id not in self._jumped:
                d = rc.deck_of(c.track_id)
                if d is not None and rc.songs[d].staged_at is not None and rc._tel_deck(d).get("playing"):
                    have = rc._song_time_at(d, rc._next_bar_clock())
                    if abs(have - c.start_s) > 0.6 * self.bar_s(c.track_id):
                        rc.jump(d, c.start_s, c.bar - self.play_from)
                        self._note(f"re-cue {rc._track(c.track_id).title[:24]} to {c.start_s:.0f}s for bar {c.bar}")
                self._jumped.add(c.id)
        # the lane map at this bar
        for ln in LANES:
            c = self.tl.active(ln, b)
            if c is not None and c.ghost and c.bar <= b:
                c.ghost = False                      # reached: the plan became the music
            want = rc.deck_of(c.track_id) if (c is not None and not c.rest) else None
            if c is not None and not c.rest and want is None:
                continue                             # not staged yet: the lane keeps what it has
            if rc.lanes.get(ln) != want:
                ok, msg = rc.assign(ln, want, force=True)
                if not ok:
                    self._note(f"{ln} → {c.track_id if c else 'rest'}: {msg}")
        # songs no clip needs for a while leave
        for d, s in list(rc.songs.items()):
            if s.leaving or not s.entered:
                continue
            needed = any(c.track_id == s.track.id and (c.end_bar is None or c.end_bar > b) and c.bar <= b + EJECT_AFTER_BARS
                         for c in self.tl.clips)
            if not needed and not s.held(rc.lanes):
                rc.eject(d)

    # -- the autopilot: the conductor's policy written ahead as ghosts ---------------------------------------------
    def _plan(self, b):
        if self.auto <= 0.0 or self.hold:
            return
        horizon = self._phrase_after(b + (HORIZON_PHRASES - 1) * self.change_bars)
        if self._planned_to is None:
            self._planned_to = self._phrase_after(b)
        while self._planned_to <= horizon:
            target = self._planned_to
            self._planned_to += self.change_bars
            if self.rng.random() >= self.auto:
                continue
            if any(not c.ghost and c.bar == target for c in self.tl.clips):
                continue                             # you placed something there: yours stands
            self._plan_one(target)

    def _songs_at(self, b):
        """{track_id: set(lanes)} heard at bar b according to the timeline."""
        out = {}
        for ln in LANES:
            c = self.tl.active(ln, b)
            if c is not None and not c.rest:
                out.setdefault(c.track_id, set()).add(ln)
        return out

    def _plan_one(self, target):
        """One move at bar `target`, by the conductor's policy: a new song through one lane when there is
        room, else a lane crossing to the newest song (consolidating) or to any other (free), or a rest."""
        rc = self.rc
        songs = self._songs_at(target - 1)
        held_by = {ln: self.tl.active(ln, target - 1) for ln in LANES}
        n_live = len(songs)
        cap = 2 if rc.blend < 0.5 else 3
        # a new song, when there is room: the brain's pick, through a lane of the oldest song
        if n_live < cap or (n_live < 3 and self.rng.random() < 0.3):
            saved = set(rc.brain.veto_ids)
            try:
                rc.brain.veto_ids |= set(songs)
                cand = rc._pick_next()
            finally:
                rc.brain.veto_ids = saved
            if cand is not None and cand.id not in songs:
                lane = self._entry_lane(held_by, songs)
                if lane is not None:
                    self.tl.add(cand.id, [lane], target, rc._landmark(cand), ghost=True)
                    self._note(f"plan bar {target}: {cand.title[:26]} in through {lane}")
                    return
        if n_live < 2:
            return
        # a crossing: toward the newest song (1 - blend) or free
        order = sorted(songs, key=lambda tid: min(c.bar for c in self.tl.clips if c.track_id == tid))
        newest = order[-1]
        lanes = list(LANES)
        self.rng.shuffle(lanes)
        if self.rng.random() >= rc.blend:
            for ln in MORPH_ORDER:
                c = held_by.get(ln)
                if c is None or c.track_id != newest:
                    if self._lane_fits(ln, newest, held_by):
                        self._ghost_cross(ln, newest, target, c)
                        return
        for ln in lanes:
            if ln == "vocals" and self.rng.random() > rc.vocal_freedom:
                continue
            c = held_by.get(ln)
            if c is not None and not c.rest and self.rng.random() < 0.12:
                gap = Clip(c.track_id, ln, target, c.song_time_at(target, self.bar_s(c.track_id)),
                           end_bar=target + self.change_bars, ghost=True, rest=True)
                self.tl.clips.append(gap)
                self._note(f"plan bar {target}: {ln} rests a phrase")
                return
            targets = [tid for tid in songs if (c is None or tid != c.track_id) and self._lane_fits(ln, tid, held_by)]
            if targets:
                tid = self.rng.choice(targets)
                self._ghost_cross(ln, tid, target, c)
                return

    def _entry_lane(self, held_by, songs):
        oldest = None
        if songs:
            oldest = min(songs, key=lambda tid: min(c.bar for c in self.tl.clips if c.track_id == tid))
        for ln in MORPH_ORDER:
            c = held_by.get(ln)
            if c is None:
                return ln
        for ln in MORPH_ORDER:
            c = held_by.get(ln)
            if c is not None and c.track_id == oldest and len([x for x in held_by.values() if x is not None and x.track_id == oldest]) > 1:
                return ln
        return MORPH_ORDER[0]

    def _lane_fits(self, lane, tid, held_by):
        """The harmonic guard on the plan: a tonal lane goes to a song whose key fits the other tonal lanes."""
        if lane == "drums":
            return True
        from lib.dj.remix import CLASH_BELOW, _compat, TONAL
        t = self.rc._track(tid)
        if t is None or not t.camelot:
            return True
        for other in TONAL:
            if other == lane:
                continue
            c = held_by.get(other)
            if c is None or c.track_id == tid:
                continue
            o = self.rc._track(c.track_id)
            if o is not None and o.camelot and _compat(o.camelot, t.camelot) < CLASH_BELOW:
                return False
        return True

    def _ghost_cross(self, lane, tid, target, current):
        """The lane crosses to `tid` at `target`: the song continues from where its other clips put it
        (same song time), else from its landmark."""
        t = self.rc._track(tid)
        ref = next((c for c in self.tl.clips if c.track_id == tid and c.bar <= target), None)
        start = ref.song_time_at(target, self.bar_s(tid)) if ref is not None else self.rc._landmark(t)
        self.tl.add(tid, [lane], target, start, ghost=True)
        self._note(f"plan bar {target}: {lane} → {t.title[:26]}")

    # -- gestures --------------------------------------------------------------------------------------------------
    def next_song(self, track_id, from_bar=None):
        """Bring a song in the way the conductor would: one lane per phrase in the morph order, starting at
        the next phrase, each clip from the song's landmark onward (so the four lanes agree in time)."""
        t = self.rc._track(int(track_id))
        if t is None:
            return False
        b = self.bar() if self.running else self.play_from
        start_bar = from_bar if from_bar is not None else self._phrase_after(b)
        land = self.rc._landmark(t)
        bar_s = self.bar_s(t.id)
        for i, ln in enumerate(MORPH_ORDER):
            at = start_bar + i * self.change_bars
            self.tl.add(t.id, [ln], at, land + i * self.change_bars * bar_s, ghost=True)
        self._note(f"NEXT: {t.title[:30]} from bar {start_bar}, a lane a phrase")
        return True

    def drop(self, track_id=None):
        """Every lane to a song at the next bar: the given song, else the newest on the timeline."""
        b = self.bar() if self.running else self.play_from
        at = b + 1
        if track_id is None:
            songs = self._songs_at(b)
            if not songs:
                return False
            track_id = max(songs, key=lambda tid: max(c.bar for c in self.tl.clips if c.track_id == tid))
        t = self.rc._track(int(track_id))
        ref = next((c for c in sorted(self.tl.clips, key=lambda c: -c.bar) if c.track_id == t.id and c.bar <= at), None)
        start = ref.song_time_at(at, self.bar_s(t.id)) if ref is not None else self.rc._landmark(t)
        self.tl.add(t.id, list(LANES), at, start, ghost=True)
        self._note(f"DROP: {t.title[:30]} at bar {at}")
        return True

    def break_(self, bars=4):
        """Every lane but the most melodic one rests `bars` bars from the next bar, then they resume."""
        b = self.bar() if self.running else self.play_from
        at = b + 1
        held = {ln: self.tl.active(ln, b) for ln in LANES}
        keep = next((ln for ln in ("vocals", "other", "bass", "drums") if held.get(ln) is not None), None)
        for ln in LANES:
            c = held.get(ln)
            if ln == keep or c is None:
                continue
            gap = Clip(c.track_id, ln, at, c.song_time_at(at, self.bar_s(c.track_id)), end_bar=at + bars, ghost=True, rest=True)
            self.tl.clips.append(gap)
        self._note(f"BREAK: {keep} alone for {bars} bars")
        return True

    def _note(self, msg):
        self.notes.append((time.strftime("%H:%M:%S"), msg))
        if len(self.notes) > 200:
            del self.notes[:-200]
