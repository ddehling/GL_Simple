"""The timeline: one continuous run of bars with four lane tracks - drums, bass, other, vocals - and
CLIPS on them: a span of one song's stem placed at a bar. One clip per lane at a time, so there is never
a mix of stems to balance: levels, beat lock, key shift and crossfades are the conductor's (lib/dj/remix.py).

    Clip      one lane of one song from song time `start_s`, starting at timeline bar `bar`, until
              `end_bar` (None = until the next clip on that lane replaces it - the DJ's "until further
              notice"; the song's own end still loops its last bars through the conductor's runway rule)
    Timeline  the clips, with the rules (a new clip ends whatever it overlaps on its lane), save / load
    Spectro   per-stem spectrograms (48 log bands, 0.1 s hop, uint8) computed from the stems once and
              cached beside them - the viewer's and the clips' picture of the music
    TimelinePlayer  walks the timeline on the conductor: stages songs so a clip's song time lands on its
              bar, puts lanes on decks at the bar, re-cues a live song for a clip elsewhere in it, ejects
              songs no clip needs; nothing here opens a device

Timeline bars are the conductor's `bar_n` plus the bar play started from. Every song advances one song
bar per timeline bar (they are beat-locked), so the song time at bar b of a clip is
start_s + (b - clip.bar) * bar_length(song).
"""
import json
import math
import os
import threading
import time

import numpy as np

LANES = ("drums", "bass", "other", "vocals")
STAGE_AHEAD_BARS = 10           # a clip's song is staged this many bars before the clip (decode + PLL settle)
EJECT_AFTER_BARS = 8            # a song with no clip for this long ahead leaves


class Clip:
    _seq = 0

    def __init__(self, track_id, lane, bar, start_s, end_bar=None, cid=None):
        Clip._seq += 1
        self.id = cid if cid is not None else Clip._seq
        Clip._seq = max(Clip._seq, self.id)
        self.track_id = int(track_id)
        self.lane = lane
        self.bar = int(bar)
        self.start_s = float(start_s)
        self.end_bar = None if end_bar is None else int(end_bar)

    def active_at(self, b):
        return self.bar <= b and (self.end_bar is None or b < self.end_bar)

    def song_time_at(self, b, bar_s):
        return self.start_s + (b - self.bar) * bar_s

    def to_dict(self):
        return {"id": self.id, "track_id": self.track_id, "lane": self.lane, "bar": self.bar,
                "start_s": round(self.start_s, 3), "end_bar": self.end_bar}

    @classmethod
    def from_dict(cls, d):
        return cls(d["track_id"], d["lane"], d["bar"], d["start_s"], d.get("end_bar"), cid=d.get("id"))


class Timeline:
    def __init__(self, name="untitled"):
        self.name = name
        self.clips = []

    # -- rules -----------------------------------------------------------------------------------------------
    def on_lane(self, lane):
        return sorted((c for c in self.clips if c.lane == lane), key=lambda c: (c.bar, c.id))

    def active(self, lane, b):
        """The clip heard on `lane` at bar b: the latest-starting clip that covers b."""
        best = None
        for c in self.on_lane(lane):
            if c.active_at(b) and (best is None or c.bar >= best.bar):
                best = c
        return best

    def add(self, track_id, lanes, bar, start_s, end_bar=None):
        """Place a span of a song on lanes at a bar. Whatever those lanes held that runs past `bar` now
        ends there (one clip per lane at a time)."""
        made = []
        for ln in lanes:
            for c in self.on_lane(ln):
                if c.bar < bar and (c.end_bar is None or c.end_bar > bar):
                    c.end_bar = bar
                elif c.bar >= bar and (end_bar is None or c.bar < end_bar):
                    self.clips.remove(c)
            clip = Clip(track_id, ln, bar, start_s, end_bar)
            self.clips.append(clip)
            made.append(clip)
        return made

    def move(self, clip, new_bar):
        delta = int(new_bar) - clip.bar
        if delta == 0:
            return
        length = None if clip.end_bar is None else clip.end_bar - clip.bar
        self.clips.remove(clip)
        self.add(clip.track_id, [clip.lane], clip.bar + delta, clip.start_s, None if length is None else clip.bar + delta + length)

    def remove(self, clip):
        if clip in self.clips:
            self.clips.remove(clip)

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
    """Walks a Timeline on a RemixConductor (autopilot 0: the timeline is the only thing that moves)."""

    def __init__(self, rc, timeline):
        self.rc = rc
        self.tl = timeline
        self.play_from = 0
        self.last_bar = None
        self.notes = []
        self.running = False
        self._jumped = set()             # (clip id) re-cued already

    def bar_s(self, track_id):
        t = self.rc._track(track_id)
        return self.rc._bar_s(t) if t is not None else 2.0

    def bar(self):
        return self.play_from + self.rc.bar_n

    def start(self, play_from=0):
        """Open on the song that holds most lanes at `play_from`, cued so the timeline's clips are exact."""
        self.play_from = int(play_from)
        active = {ln: self.tl.active(ln, self.play_from) for ln in LANES}
        held = [c for c in active.values() if c is not None]
        if not held:
            # nothing at this bar: open on the earliest clip's song, silent, and let the walk place it
            first = min(self.tl.clips, key=lambda c: (c.bar, c.id), default=None)
            if first is None:
                return False, "the timeline is empty"
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

    def step(self):
        """Call every tick (the conductor's own step runs too): staging ahead, the lane map at each bar,
        re-cues, ejects."""
        rc = self.rc
        rc.step()
        if rc.master is None or not self.running:
            return
        b = self.bar()
        # stage songs whose clips are coming (or already due) and are not on a deck
        wanted = {}
        for c in self.tl.clips:
            if c.bar <= b + STAGE_AHEAD_BARS and (c.end_bar is None or c.end_bar > b):
                if rc.deck_of(c.track_id) is None:
                    prev = wanted.get(c.track_id)
                    if prev is None or c.bar < prev.bar:
                        wanted[c.track_id] = c
        for tid, c in wanted.items():
            if c.bar - b <= STAGE_AHEAD_BARS:
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
            want = rc.deck_of(c.track_id) if c is not None else None
            if c is not None and want is None:
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

    def _note(self, msg):
        self.notes.append((time.strftime("%H:%M:%S"), msg))
        if len(self.notes) > 200:
            del self.notes[:-200]
