"""A/B wav player for the console (sounddevice): play the original or
the recreation from any position, switch between them without losing
the place, seek, stop. Independent of the show's audio engine (WASAPI
shared mode lets both run).

    p = WavPlayer()
    p.load("a", stereo_array); p.load("b", stereo_array)
    p.play("a"); p.seek(42.0); p.switch("b"); p.pause(); p.position(); p.playing
No output device -> available() is False and every call is a no-op."""
from __future__ import annotations

import threading

import numpy as np

from lib.gen import RATE


class WavPlayer:
    def __init__(self):
        self.sources = {}
        self.current = None
        self.pos = 0                 # frames into the current source
        self.playing = False
        self.gain = 0.9
        self._stream = None
        self._lock = threading.Lock()
        self.error = ""

    def available(self) -> bool:
        try:
            import sounddevice as sd
            sd.query_devices(kind="output")
            return True
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            return False

    def load(self, name, stereo):
        x = np.asarray(stereo, dtype=np.float32)
        if x.ndim == 1:
            x = np.stack([x, x], axis=1)
        with self._lock:
            self.sources[name] = np.ascontiguousarray(x)
            if self.current is None:
                self.current = name

    def seconds(self, name=None):
        src = self.sources.get(name or self.current)
        return 0.0 if src is None else src.shape[0] / RATE

    def position(self) -> float:
        return self.pos / RATE

    def _ensure_stream(self):
        if self._stream is not None:
            return True
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(samplerate=RATE, channels=2, dtype="float32", blocksize=1024, callback=self._callback)
            self._stream.start()
            return True
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self._stream = None
            return False

    def _callback(self, out, frames, time_info, status):
        with self._lock:
            src = self.sources.get(self.current)
            if not self.playing or src is None or self.pos >= src.shape[0]:
                out[:] = 0.0
                if src is not None and self.pos >= src.shape[0]:
                    self.playing = False
                return
            end = min(src.shape[0], self.pos + frames)
            n = end - self.pos
            out[:n] = src[self.pos:end] * self.gain
            if n < frames:
                out[n:] = 0.0
            self.pos = end

    def play(self, name=None):
        if name is not None:
            self.switch(name)
        if not self._ensure_stream():
            return False
        with self._lock:
            src = self.sources.get(self.current)
            if src is None:
                return False
            if self.pos >= src.shape[0]:
                self.pos = 0
            self.playing = True
        return True

    def pause(self):
        with self._lock:
            self.playing = False

    def stop(self):
        with self._lock:
            self.playing = False
            self.pos = 0

    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float):
        with self._lock:
            src = self.sources.get(self.current)
            n = src.shape[0] if src is not None else 0
            self.pos = int(max(0, min(n, seconds * RATE)))

    def switch(self, name):
        """Change source, keeping the position (in seconds)."""
        with self._lock:
            if name in self.sources and name != self.current:
                self.current = name
                src = self.sources[name]
                self.pos = min(self.pos, src.shape[0])

    def close(self):
        self.pause()
        try:
            if self._stream is not None:
                self._stream.stop(); self._stream.close()
        except Exception:
            pass
        self._stream = None
