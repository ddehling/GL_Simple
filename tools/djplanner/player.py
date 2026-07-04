"""Audio playback for the planner.

TrackPlayer   - one decoded track, instant seek/scrub (Analysis tab).
SetPreview    - plays a COMPILED SET live: a producer thread runs the real
                DJSystem + engine mixer (the exact code path of a show),
                rendering ~2s ahead into a ring buffer that a miniaudio
                device drains. Jumping to a slot rebuilds the system with
                the remaining setlist; 'next seam' asks the brain for a
                musical exit right now (request_skip).
"""
import queue
import threading
import time

import numpy as np

RATE = 44100


class _Device:
    """Shared miniaudio pull device around a ring-ish queue of blocks."""

    def __init__(self, fetch):
        """fetch(n) -> float32 (n,2) called on the audio thread."""
        import miniaudio
        self._fetch = fetch

        def gen():
            required = yield b""
            while True:
                try:
                    chunk = self._fetch(required)
                except Exception:
                    chunk = np.zeros((required, 2), dtype=np.float32)
                required = yield chunk.astype(np.float32).tobytes()
        g = gen()
        next(g)
        self.dev = miniaudio.PlaybackDevice(
            output_format=miniaudio.SampleFormat.FLOAT32, nchannels=2,
            sample_rate=RATE)
        self.dev.start(g)

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass


class TrackPlayer:
    """Play a (n,2)/(n,) float32 array with sample-accurate seek."""

    def __init__(self):
        self.samples = None
        self.pos = 0
        self.playing = False
        self._dev = None

    def load(self, samples):
        s = np.asarray(samples, dtype=np.float32)
        if s.ndim == 1:
            s = np.stack([s, s], axis=1)
        self.samples = s
        self.pos = 0

    def _fetch(self, n):
        if not self.playing or self.samples is None:
            return np.zeros((n, 2), dtype=np.float32)
        a = self.samples[self.pos:self.pos + n]
        self.pos += len(a)
        if len(a) < n:
            self.playing = False
            a = np.concatenate([a, np.zeros((n - len(a), 2),
                                            dtype=np.float32)])
        return a

    def play(self):
        if self.samples is None:
            return
        if self._dev is None:
            self._dev = _Device(self._fetch)
        self.playing = True

    def pause(self):
        self.playing = False

    def seek(self, seconds):
        if self.samples is not None:
            self.pos = int(np.clip(seconds * RATE, 0,
                                   len(self.samples) - 1))

    def time_s(self):
        return self.pos / RATE

    def close(self):
        self.playing = False
        if self._dev is not None:
            self._dev.close()
            self._dev = None


class SetPreview:
    """Live playback of a planned set through the REAL DJ system."""

    def __init__(self, music_dir, entries, theme_name="groove", log_dir=None):
        self.music_dir = music_dir
        self.entries = list(entries)
        self.theme_name = theme_name
        self.log_dir = log_dir
        # ~8s of 2205-frame blocks: must outlast the synchronous next-track
        # decode inside dj.step() (1-3s) or transitions would underrun.
        self._buf = queue.Queue(maxsize=160)
        self._leftover = np.zeros((0, 2), dtype=np.float32)
        self._dev = None
        self._producer = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self.dj = None
        self.slot_index = 0                     # which entry is playing
        self._pending_cue = None
        self.error = ""

    # -- lifecycle ------------------------------------------------------------
    def start(self, from_slot=0, cue_s=None):
        """Begin playback at slot `from_slot`; optional cue_s jumps to that
        track-time once the deck is rolling (timeline click-to-seek)."""
        self.stop()
        self._stop.clear()
        self._pause.clear()
        self.slot_index = from_slot
        self._pending_cue = cue_s
        self._producer = threading.Thread(target=self._produce,
                                          args=(from_slot,), daemon=True)
        self._producer.start()
        if self._dev is None:
            self._dev = _Device(self._fetch)

    def stop(self):
        self._stop.set()
        if self._producer is not None:
            self._producer.join(timeout=2.0)
            self._producer = None
        try:
            while True:
                self._buf.get_nowait()
        except queue.Empty:
            pass
        if self.dj is not None:
            try:
                self.dj._running = False
            except Exception:
                pass
            self.dj = None

    def close(self):
        self.stop()
        if self._dev is not None:
            self._dev.close()
            self._dev = None

    def pause(self, on):
        (self._pause.set if on else self._pause.clear)()

    def next_seam(self):
        if self.dj is not None:
            self.dj.request_skip()

    def jump_to_slot(self, i):
        self.start(from_slot=max(0, min(i, len(self.entries) - 1)))

    def seek(self, slot, track_time_s):
        """Jump playback to a specific moment. Seeking inside the CURRENT
        track is instant (just re-cues the live deck); another slot means a
        clean restart of the system from that entry."""
        dj = self.dj
        if (dj is not None and slot == self.slot_index
                and dj.state in ("playing", "armed") and dj.current):
            deck = dj.active_deck
            dj.submix.post({"cmd": "cue", "deck": deck,
                            "time_s": float(track_time_s)})
            return
        self.start(from_slot=max(0, min(slot, len(self.entries) - 1)),
                   cue_s=track_time_s)

    # -- internals --------------------------------------------------------------
    def _fetch(self, n):
        if self._pause.is_set():
            return np.zeros((n, 2), dtype=np.float32)
        out = np.zeros((n, 2), dtype=np.float32)
        got = 0
        if len(self._leftover):
            take = min(n, len(self._leftover))
            out[:take] = self._leftover[:take]
            self._leftover = self._leftover[take:]
            got = take
        while got < n:
            try:
                blk = self._buf.get_nowait()
            except queue.Empty:
                break                            # underrun -> brief silence
            take = min(n - got, len(blk))
            out[got:got + take] = blk[:take]
            self._leftover = blk[take:]
            got += take
        return out

    def _produce(self, from_slot):
        try:
            from lib.audio_engine import AudioEngine
            from lib.dj.system import DJSystem
            engine = AudioEngine()
            dj = DJSystem(self.music_dir, engine=engine,
                          theme=self.theme_name, threaded=False,
                          log_dir=self.log_dir, seed=1234)
            if not dj.start():
                self.error = dj.last_error
                return
            dj._setlist_name = "(preview)"
            dj._setlist_queue = [dict(e) for e in self.entries[from_slot:]]
            self.dj = dj
            gen = engine._mixer()
            next(gen)
            block = 2205
            prev_id = None
            while not self._stop.is_set():
                if self._pause.is_set():
                    time.sleep(0.05)
                    continue
                buf = gen.send(block)
                dj.step()
                cur = (dj.status()["current"] or {}).get("id")
                if cur is not None and cur != prev_id:
                    if prev_id is not None:
                        self.slot_index += 1
                    prev_id = cur
                    # Deferred click-seek: cue once the first track rolls.
                    cue = getattr(self, "_pending_cue", None)
                    if cue is not None:
                        self._pending_cue = None
                        dj.submix.post({"cmd": "cue", "deck": dj.active_deck,
                                        "time_s": float(cue)})
                arr = np.frombuffer(buf, dtype=np.float32).reshape(-1, 2)
                while not self._stop.is_set():
                    try:
                        self._buf.put(arr, timeout=0.2)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"

    def status(self):
        if self.dj is None:
            return {"active": False, "error": self.error}
        s = self.dj.status()
        s["active"] = True
        s["slot_index"] = self.slot_index
        return s
