"""Streaming audio engine using miniaudio.

All audio streams directly from disk — no files are preloaded into RAM.
Looping, crossfades, and mixing happen inside a single generator that feeds
miniaudio's PlaybackDevice callback thread.

Thread safety: all mutable track state lives inside _mixer(), which runs on
the audio thread. Callers communicate via a SimpleQueue of command tuples —
no locks needed anywhere.
"""

import queue
import time
import numpy as np
import miniaudio
from pathlib import Path

SAMPLE_RATE   = 44100
CHANNELS      = 2
FORMAT        = miniaudio.SampleFormat.FLOAT32
CHUNK_FRAMES  = 1024   # frames per stream_file read; buffered in _Track


class _MemTrack:
    """Memory-decoded audio track — entire file loaded into RAM.

    Used for oneshot sounds (narrative, BART sounds) to avoid I/O contention
    during mixing.  No streaming, no disk access after init.
    """

    def __init__(self, samples: np.ndarray, path: Path, *, volume: float = 1.0,
                 duration: float = 0.0, is_narrative: bool = False):
        self.is_ambient   = False
        self.is_narrative = is_narrative
        self.done         = False
        self._path        = path
        self.volume       = volume
        self._pos         = 0
        self._fading_out  = False
        self._fade_out_frames = 0
        self._fade_out_pos    = 0

        self._samples = samples
        # Note: no duration trim. The in-memory array is the ground truth
        # for length; trimming based on a separately-measured duration would
        # silently truncate VBR MP3s when mp3_get_file_info() under-reports.

    def fade_out(self, duration: float):
        self._fading_out      = True
        self._fade_out_frames = max(1, int(duration * SAMPLE_RATE))
        self._fade_out_pos    = 0

    def read(self, n_frames: int):
        if self.done or self._pos >= len(self._samples):
            self.done = True
            return None

        end = min(self._pos + n_frames, len(self._samples))
        chunk = self._samples[self._pos:end].copy()
        self._pos = end

        # Fade-out
        if self._fading_out:
            remaining = self._fade_out_frames - self._fade_out_pos
            if remaining <= 0:
                self.done = True
                return None
            fade_len = min(len(chunk), remaining)
            s = 1.0 - self._fade_out_pos / self._fade_out_frames
            e = 1.0 - (self._fade_out_pos + fade_len) / self._fade_out_frames
            ramp = np.linspace(s, e, fade_len, dtype=np.float32)
            chunk[:fade_len] *= ramp[:, np.newaxis]
            if fade_len < len(chunk):
                chunk[fade_len:] = 0.0
            self._fade_out_pos += fade_len

        return chunk * self.volume


class _Track:
    """Single streaming audio source with loop and linear fade support."""

    def __init__(self, path: Path, *, loop: bool = False, skip: float = 0.0,
                 fade_in: float = 0.0, volume: float = 1.0,
                 duration: float = 0.0, loop_length: float = 0.0,
                 is_ambient: bool = False, is_narrative: bool = False):
        self.is_ambient       = is_ambient
        self.is_narrative     = is_narrative
        self.done             = False
        self._path            = path
        self._loop            = loop
        self._skip            = skip
        self.volume           = volume
        self._gen             = self._open(skip)
        self._buf             = np.zeros((0, CHANNELS), dtype=np.float32)
        self._fade_in_frames  = int(fade_in * SAMPLE_RATE)
        self._max_frames      = int(duration * SAMPLE_RATE) if duration > 0 else 0
        # loop_length: if >0, restart from skip every N seconds instead of at EOF.
        self._loop_frames     = int(loop_length * SAMPLE_RATE) if loop_length > 0 else 0
        self._pos             = 0   # monotonic total frames (drives fade-in and duration cap only)
        self._loop_pos        = 0   # position within the current ARI window (resets on each loop)
        self._fading_out      = False
        self._fade_out_frames = 0
        self._fade_out_pos    = 0
        # ARI crossfade: second stream blended in near the loop boundary.
        self._xfade_gen       = None
        self._xfade_buf       = np.zeros((0, CHANNELS), dtype=np.float32)

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

    # Duration of the crossfade applied at ARI loop boundaries (seconds).
    # During this window the tail of the current loop fades out linearly
    # while the beginning of the next loop fades in, producing a smooth
    # overlap identical to what weather-state transitions get.
    _ARI_XFADE_SEC = 3.0

    def read(self, n_frames: int):
        # ARI loop boundary with crossfade.
        # When we're within _ARI_XFADE_SEC of the loop end, open a second
        # stream from skip_time and blend it in while fading the current
        # stream out. When the loop boundary is reached, the new stream
        # becomes primary.
        if self._loop_frames > 0:
            remaining = self._loop_frames - self._loop_pos
            xfade_frames = int(self._ARI_XFADE_SEC * SAMPLE_RATE)

            if remaining <= 0:
                # Loop boundary reached.
                if self._xfade_gen is not None:
                    # The xfade stream has been read in parallel for
                    # xfade_frames samples during the crossfade — it is now
                    # at file pos `_skip + xfade_frames`. Account for that
                    # in _loop_pos so the next loop window ends at the same
                    # absolute file offset every time. Without this, each
                    # loop would drift forward by xfade_frames worth of
                    # audio, eventually playing parts of the file far past
                    # the intended ARI window.
                    self._gen = self._xfade_gen
                    self._buf = self._xfade_buf
                    self._xfade_gen = None
                    self._xfade_buf = np.zeros((0, CHANNELS), dtype=np.float32)
                    self._loop_pos = xfade_frames
                else:
                    # Edge case: no crossfade was set up (e.g. n_frames was
                    # large enough to skip past the xfade trigger). Fresh
                    # stream from _skip — no drift to compensate.
                    self._gen = self._open(self._skip)
                    self._buf = np.zeros((0, CHANNELS), dtype=np.float32)
                    self._loop_pos = 0
                remaining = self._loop_frames - self._loop_pos

            elif remaining <= xfade_frames and self._xfade_gen is None:
                # Approaching the loop boundary — open the next stream.
                self._xfade_gen = self._open(self._skip)
                self._xfade_buf = np.zeros((0, CHANNELS), dtype=np.float32)

            n_frames = min(n_frames, remaining)

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

        # ARI crossfade blend: if a next-loop stream is open, read from it
        # and mix into the current chunk with a linear crossfade ramp.
        if self._xfade_gen is not None and self._loop_frames > 0:
            xfade_frames = int(self._ARI_XFADE_SEC * SAMPLE_RATE)
            remaining_in_loop = self._loop_frames - self._loop_pos
            # How far into the crossfade window are we? (0 = just entered, 1 = done)
            xf_pos = xfade_frames - remaining_in_loop

            # Fill the crossfade buffer from the next-loop stream.
            while len(self._xfade_buf) < n:
                try:
                    raw = next(self._xfade_gen)
                    new = np.frombuffer(raw, dtype=np.float32).reshape(-1, CHANNELS)
                    self._xfade_buf = np.concatenate([self._xfade_buf, new])
                except StopIteration:
                    break

            xf_n = min(n, len(self._xfade_buf))
            if xf_n > 0:
                xf_chunk = self._xfade_buf[:xf_n]
                self._xfade_buf = self._xfade_buf[xf_n:]

                # Linear crossfade ramp: current fades 1→0, next fades 0→1.
                t_start = max(0, xf_pos) / max(1, xfade_frames)
                t_end   = max(0, xf_pos + xf_n) / max(1, xfade_frames)
                t_start = min(t_start, 1.0)
                t_end   = min(t_end, 1.0)
                ramp_in  = np.linspace(t_start, t_end, xf_n, dtype=np.float32)[:, np.newaxis]
                ramp_out = 1.0 - ramp_in

                chunk[:xf_n] = chunk[:xf_n] * ramp_out + xf_chunk * ramp_in

        # Fade-in: linear ramp from 0→1 over the first _fade_in_frames samples.
        if self._fade_in_frames > 0 and self._pos < self._fade_in_frames:
            start = self._pos
            end   = min(start + n, self._fade_in_frames)
            ramp  = np.linspace(start / self._fade_in_frames,
                                end   / self._fade_in_frames,
                                end - start, dtype=np.float32)
            chunk[:end - start] *= ramp[:, np.newaxis]
        self._pos      += n
        self._loop_pos += n

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
        self.master_volume = 1.0    # scales all audio output
        self.narrative_volume = 1.0  # scales non-ambient (oneshot) tracks in real time
        # Scales is_ambient tracks at mix time. Stories_OGL pushes the
        # active weather state's ``Sound_volume`` parameter here each
        # frame; weather editor edits to Sound_volume therefore take
        # effect immediately without restart. 1.0 = no attenuation.
        self.ambient_volume = 1.0

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
                     fade_in: float = FADE_IN, fade_out: float = FADE_OUT,
                     ari: float = 0.0):
        """Cross-fade to a new looping ambient track.

        ari: seconds to play from skip_seconds before looping back. 0 = loop at EOF.
        """
        self._cmds.put(("ambient", Path(path), skip_seconds, fade_in, fade_out, ari))

    def stop_ambient(self, duration: float = FADE_OUT):
        """Fade out the current ambient track."""
        self._cmds.put(("stop_ambient", duration))

    def stop_all(self, duration: float = FADE_OUT):
        """Fade out all tracks (ambient + oneshots)."""
        self._cmds.put(("stop_all", duration))

    def schedule_event(self, path, volume: float = 1.0, duration: float = 0.0,
                       narrative: bool = False):
        """Play a sound file once. duration>0 caps playback to that many seconds.

        Decodes in a background thread to avoid blocking both the main thread
        and the audio callback thread.
        """
        import threading
        p = Path(path)
        def _decode_and_queue():
            try:
                decoded = miniaudio.decode_file(
                    str(p), output_format=FORMAT,
                    nchannels=CHANNELS, sample_rate=SAMPLE_RATE)
                # Copy raw bytes first to own the memory independently of
                # miniaudio's C buffer, preventing GC race conditions
                raw_bytes = bytes(decoded.samples)
                samples = np.frombuffer(raw_bytes,
                                        dtype=np.float32).reshape(-1, CHANNELS)
                self._cmds.put(("oneshot_mem", samples, p, volume, duration, narrative))
            except Exception as e:
                print(f"[AudioEngine] Failed to decode {p.name}: {e}")
        threading.Thread(target=_decode_and_queue, daemon=True).start()


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
                        _, path, skip, fi, fo, ari = cmd
                        for t in tracks.values():
                            if t.is_ambient:
                                t.fade_out(fo)
                        tracks[f"ambient_{path.name}_{time.monotonic():.6f}"] = _Track(
                            path, loop=True, skip=skip,
                            fade_in=fi, loop_length=ari, is_ambient=True)
                        print(f"[AudioEngine] Ambient → {path.name}")

                    elif kind == "stop_ambient":
                        _, fo = cmd
                        for t in tracks.values():
                            if t.is_ambient:
                                t.fade_out(fo)

                    elif kind == "stop_all":
                        _, fo = cmd
                        for t in tracks.values():
                            t.fade_out(fo)

                    elif kind == "oneshot_mem":
                        _, samples, path, vol, dur, narr = cmd
                        # Unique key — id(cmd) was unsafe because Python can
                        # reuse the address of a freed tuple, silently
                        # overwriting a still-playing track in the dict.
                        tracks[f"os_{path.name}_{time.monotonic_ns()}"] = _MemTrack(
                            samples, path, volume=vol, duration=dur,
                            is_narrative=narr)
                        if not narr:
                            track_dur = len(samples) / SAMPLE_RATE
                            finish_at = time.strftime(
                                "%H:%M:%S", time.localtime(time.time() + track_dur))
                            print(f"[AudioEngine] Oneshot ▶ {path.name}  "
                                  f"dur={track_dur:.2f}s  finishes ~{finish_at}")


            except queue.Empty:
                pass

            # Mix tracks into separate buses: narrative vs everything else.
            # The limiter only applies to non-narrative audio so narrative
            # is never ducked by other sounds.
            narr_buf = np.zeros((required_frames, CHANNELS), dtype=np.float32)
            other_buf = np.zeros((required_frames, CHANNELS), dtype=np.float32)
            dead = []
            narr_vol = self.narrative_volume
            amb_vol = self.ambient_volume
            for key, track in tracks.items():
                try:
                    chunk = track.read(required_frames)
                except Exception:
                    dead.append(key)
                    continue
                if chunk is None or track.done:
                    dead.append(key)
                else:
                    if track.is_narrative:
                        narr_buf[:len(chunk)] += chunk * narr_vol
                    elif track.is_ambient:
                        # Ambient tracks scale by the per-state
                        # Sound_volume parameter pushed in from the
                        # weather scheduler — editing Sound_volume in
                        # the weather editor takes effect at the next
                        # frame, no restart needed.
                        other_buf[:len(chunk)] += chunk * amb_vol
                    else:
                        other_buf[:len(chunk)] += chunk
            for key in dead:
                del tracks[key]

            # Limit only the non-narrative bus
            peak = np.max(np.abs(other_buf))
            if peak > 2.0:
                other_buf /= (peak / 2.0)

            buf = (narr_buf + other_buf) * self.master_volume

            required_frames = yield buf.tobytes()


# Backward-compatibility alias — render_pipeline.py instantiates this name.
ThreadedAudioEngine = AudioEngine
