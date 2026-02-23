"""Streaming audio engine using miniaudio.

All audio streams directly from disk — no files are preloaded into RAM.
Looping, crossfades, and mixing happen inside a single generator that feeds
miniaudio's PlaybackDevice callback thread.

Thread safety: all mutable track state lives inside _mixer(), which runs on
the audio thread. Callers communicate via a SimpleQueue of command tuples —
no locks needed anywhere.
"""

import queue
import numpy as np
import miniaudio
from pathlib import Path

SAMPLE_RATE   = 44100
CHANNELS      = 2
FORMAT        = miniaudio.SampleFormat.FLOAT32
CHUNK_FRAMES  = 1024   # frames per stream_file read; buffered in _Track


class _Track:
    """Single streaming audio source with loop and linear fade support."""

    def __init__(self, path: Path, *, loop: bool = False, skip: float = 0.0,
                 fade_in: float = 0.0, volume: float = 1.0,
                 duration: float = 0.0, is_ambient: bool = False):
        self.is_ambient       = is_ambient
        self.done             = False
        self._path            = path
        self._loop            = loop
        self._skip            = skip
        self.volume           = volume
        self._gen             = self._open(skip)
        self._buf             = np.zeros((0, CHANNELS), dtype=np.float32)
        self._fade_in_frames  = int(fade_in * SAMPLE_RATE)
        self._max_frames      = int(duration * SAMPLE_RATE) if duration > 0 else 0
        self._pos             = 0   # total frames read (drives fade-in and duration cap)
        self._fading_out      = False
        self._fade_out_frames = 0
        self._fade_out_pos    = 0

    def _open(self, skip: float):
        return miniaudio.stream_file(
            str(self._path),
            output_format=FORMAT,
            nchannels=CHANNELS,
            sample_rate=SAMPLE_RATE,
            frames_to_read=CHUNK_FRAMES,
            seek_frame=int(skip * SAMPLE_RATE),
        )

    def fade_out(self, duration: float):
        self._fading_out      = True
        self._fade_out_frames = max(1, int(duration * SAMPLE_RATE))
        self._fade_out_pos    = 0
        self._loop            = False

    def read(self, n_frames: int):
        # Fill internal buffer until we have enough frames (or hit EOF).
        while len(self._buf) < n_frames and not self.done:
            try:
                raw = next(self._gen)
                new = np.frombuffer(raw, dtype=np.float32).reshape(-1, CHANNELS)
                self._buf = np.concatenate([self._buf, new])
            except StopIteration:
                if self._loop:
                    self._gen = self._open(self._skip)
                else:
                    self.done = True

        if len(self._buf) == 0:
            return None

        # Cap to duration limit if set.
        if self._max_frames > 0:
            n_frames = min(n_frames, self._max_frames - self._pos)
            if n_frames <= 0:
                self.done = True
                return None

        n     = min(n_frames, len(self._buf))
        chunk = self._buf[:n].copy()
        self._buf = self._buf[n:]

        # Fade-in: linear ramp from 0→1 over the first _fade_in_frames samples.
        if self._fade_in_frames > 0 and self._pos < self._fade_in_frames:
            start = self._pos
            end   = min(start + n, self._fade_in_frames)
            ramp  = np.linspace(start / self._fade_in_frames,
                                end   / self._fade_in_frames,
                                end - start, dtype=np.float32)
            chunk[:end - start] *= ramp[:, np.newaxis]
        self._pos += n

        # Fade-out: linear ramp from 1→0 over _fade_out_frames samples.
        if self._fading_out:
            remaining = self._fade_out_frames - self._fade_out_pos
            if remaining <= 0:
                self.done = True
                return None
            fade_len = min(n, remaining)
            s    = 1.0 - self._fade_out_pos / self._fade_out_frames
            e    = 1.0 - (self._fade_out_pos + fade_len) / self._fade_out_frames
            ramp = np.linspace(s, e, fade_len, dtype=np.float32)
            chunk[:fade_len] *= ramp[:, np.newaxis]
            if fade_len < n:
                chunk[fade_len:] = 0.0
            self._fade_out_pos += fade_len

        return chunk * self.volume



class AudioEngine:
    """Streaming multi-track audio mixer — no files loaded into RAM.

    All public methods are thread-safe: they post command tuples to a
    SimpleQueue that is drained on the audio thread inside _mixer().
    """

    FADE_IN  = 5.0   # seconds
    FADE_OUT = 5.0   # seconds

    def __init__(self):
        self._cmds: queue.SimpleQueue = queue.SimpleQueue()
        self._device = None

    # ------------------------------------------------------------------
    # Public API (safe to call from any thread)
    # ------------------------------------------------------------------

    def start(self):
        self._device = miniaudio.PlaybackDevice(
            output_format=FORMAT,
            nchannels=CHANNELS,
            sample_rate=SAMPLE_RATE,
            buffersize_msec=200,
        )
        gen = self._mixer()
        next(gen)   # advance to first yield so miniaudio can send(framecount)
        self._device.start(gen)

    def stop(self):
        if self._device:
            self._device.stop()
            self._device.close()
            self._device = None

    def play_ambient(self, path, skip_seconds: float = 0.0,
                     fade_in: float = FADE_IN, fade_out: float = FADE_OUT):
        """Cross-fade to a new looping ambient track."""
        self._cmds.put(("ambient", Path(path), skip_seconds, fade_in, fade_out))

    def stop_ambient(self, duration: float = FADE_OUT):
        """Fade out the current ambient track."""
        self._cmds.put(("stop_ambient", duration))

    def schedule_event(self, path, volume: float = 1.0, duration: float = 0.0):
        """Play a sound file once. duration>0 caps playback to that many seconds."""
        self._cmds.put(("oneshot", Path(path), volume, duration))


    # ------------------------------------------------------------------
    # Mixer — runs entirely on miniaudio's audio callback thread.
    # `tracks` is only ever mutated here, so no locking is needed.
    # ------------------------------------------------------------------

    def _mixer(self):
        tracks: dict = {}
        required_frames = yield b""   # miniaudio handshake

        while True:
            # Drain pending commands from other threads.
            try:
                while True:
                    cmd  = self._cmds.get_nowait()
                    kind = cmd[0]

                    if kind == "ambient":
                        _, path, skip, fi, fo = cmd
                        for t in tracks.values():
                            if t.is_ambient:
                                t.fade_out(fo)
                        tracks[f"ambient_{path.name}"] = _Track(
                            path, loop=True, skip=skip,
                            fade_in=fi, is_ambient=True)
                        print(f"[AudioEngine] Ambient → {path.name}")

                    elif kind == "stop_ambient":
                        _, fo = cmd
                        for t in tracks.values():
                            if t.is_ambient:
                                t.fade_out(fo)

                    elif kind == "oneshot":
                        _, path, vol, dur = cmd
                        tracks[f"os_{id(cmd)}"] = _Track(path, volume=vol,
                                                         duration=dur)


            except queue.Empty:
                pass

            # Mix all active tracks.
            buf  = np.zeros((required_frames, CHANNELS), dtype=np.float32)
            dead = []
            for key, track in tracks.items():
                chunk = track.read(required_frames)
                if chunk is None or track.done:
                    dead.append(key)
                else:
                    buf[:len(chunk)] += chunk
            for key in dead:
                del tracks[key]

            peak = np.max(np.abs(buf))
            if peak > 1.0:
                buf /= peak

            required_frames = yield buf.tobytes()


# Backward-compatibility alias — render_pipeline.py instantiates this name.
ThreadedAudioEngine = AudioEngine
