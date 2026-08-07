"""Streaming audio engine using miniaudio.

All audio streams directly from disk — no files are preloaded into RAM.
Looping, crossfades, and mixing happen inside a single generator that feeds
miniaudio's PlaybackDevice callback thread.

Thread safety: all mutable track state lives inside _mixer(), which runs on
the audio thread. Callers communicate via a SimpleQueue of command tuples —
no locks needed anywhere.
"""

import itertools
import queue
import sys
import threading
import time
import numpy as np
import miniaudio
from pathlib import Path

# DEFAULT device/mix rate. The engine's ACTUAL rate is per-instance
# (``AudioEngine(sample_rate=...)``, ``engine.sample_rate``) so different
# projects can run at different rates — WoL's masters are 48 kHz, Fan's
# library and the whole DJ stack are 44.1 kHz. This constant is only the
# fallback for callers that don't specify one; nothing outside this module
# reads it, so track/ring maths below all go through the instance rate.
SAMPLE_RATE   = 44100
CHANNELS      = 2
FORMAT        = miniaudio.SampleFormat.FLOAT32
CHUNK_FRAMES  = 1024   # frames per stream_file read; buffered in _Track

# RENDER-AHEAD RING. The device callback used to RENDER the mix inline, so
# it had to finish a whole buffer within one buffer-duration or the DAC ran
# dry - and it is Python, so it also had to win the GIL against the render
# loop, the web preview and the DMX threads to get started at all. Measured
# in the show on a 4-core N150: 6.3% of blocks starved, up to 191ms of that
# spent purely waiting to be scheduled.
#
# Now a producer thread renders ahead into this ring and the callback only
# copies bytes out of it, so a late producer eats into slack instead of
# punching a hole in the output. The cost is latency: a control action
# (skip, cue, a scheduled DJ event) is heard RING_TARGET_MS later than
# before, on top of the device's own buffer.
RING_TARGET_MS   = 400        # how far ahead we render = the jitter we absorb
RING_CAPACITY_MS = 700        # ring size; > target so the producer has room
RENDER_BLOCK     = 2048       # frames per producer pass (~46ms)
# Ring sizes are FRAME counts, so they depend on the rate — derived per
# instance in AudioEngine (self._ring_target / _ring_capacity) and re-derived
# by set_sample_rate. These module-level values are the 44.1 kHz defaults.
RING_TARGET   = SAMPLE_RATE * RING_TARGET_MS // 1000
RING_CAPACITY = SAMPLE_RATE * RING_CAPACITY_MS // 1000


class _MemTrack:
    """Memory-decoded audio track — entire file loaded into RAM.

    Used for oneshot sounds (narrative, BART sounds) to avoid I/O contention
    during mixing.  No streaming, no disk access after init.
    """

    def __init__(self, samples: np.ndarray, path: Path, *, volume: float = 1.0,
                 duration: float = 0.0, is_narrative: bool = False,
                 fade_in: float = 0.0, is_soundpool: bool = False,
                 rate: int = SAMPLE_RATE):
        self.is_ambient   = False
        self.is_narrative = is_narrative
        self.is_soundpool = is_soundpool
        self.done         = False
        self._path        = path
        self.volume       = volume
        self._pos         = 0
        # Rate of the samples we were handed == the engine's rate, since
        # schedule_event decoded them at it. All frame maths below uses it.
        self._rate        = int(rate)
        self._fade_in_frames  = int(fade_in * self._rate)
        self._fading_out  = False
        self._fade_out_frames = 0
        self._fade_out_pos    = 0

        self._samples = samples
        # Note: no duration trim. The in-memory array is the ground truth
        # for length; trimming based on a separately-measured duration would
        # silently truncate VBR MP3s when mp3_get_file_info() under-reports.

    def fade_out(self, duration: float):
        self._fading_out      = True
        self._fade_out_frames = max(1, int(duration * self._rate))
        self._fade_out_pos    = 0

    def read(self, n_frames: int):
        if self.done or self._pos >= len(self._samples):
            self.done = True
            return None

        start = self._pos
        end = min(self._pos + n_frames, len(self._samples))
        chunk = self._samples[start:end].copy()
        self._pos = end

        # Fade-in: linear ramp from 0->1 over the first _fade_in_frames samples.
        if self._fade_in_frames > 0 and start < self._fade_in_frames:
            fi_end = min(end, self._fade_in_frames)
            ramp = np.linspace(start / self._fade_in_frames,
                               fi_end / self._fade_in_frames,
                               fi_end - start, dtype=np.float32)
            chunk[:fi_end - start] *= ramp[:, np.newaxis]

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
                 is_ambient: bool = False, is_narrative: bool = False,
                 rate: int = SAMPLE_RATE):
        self.is_ambient       = is_ambient
        self.is_narrative     = is_narrative
        self.is_soundpool     = False   # soundpool clips are always _MemTracks
        self.done             = False
        self._path            = path
        self._loop            = loop
        self._skip            = skip
        self.volume           = volume
        # Set BEFORE _open(): the stream is opened at this rate, so miniaudio
        # resamples the file to it on the fly.
        self._rate            = int(rate)
        self._gen             = self._open(skip)
        self._buf             = np.zeros((0, CHANNELS), dtype=np.float32)
        self._fade_in_frames  = int(fade_in * self._rate)
        self._max_frames      = int(duration * self._rate) if duration > 0 else 0
        # loop_length: if >0, restart from skip every N seconds instead of at EOF.
        self._loop_frames     = int(loop_length * self._rate) if loop_length > 0 else 0
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
            sample_rate=self._rate,
            frames_to_read=CHUNK_FRAMES,
            seek_frame=int(skip * self._rate),
        )

    def fade_out(self, duration: float):
        self._fading_out      = True
        self._fade_out_frames = max(1, int(duration * self._rate))
        self._fade_out_pos    = 0
        self._loop            = False

    # Duration of the crossfade applied at ARI loop boundaries (seconds).
    # During this window the tail of the current loop fades out while the
    # beginning of the next loop fades in, on an equal-power (sin/cos) ramp —
    # see the blend in read(). The weather-state transition fades
    # (_fade_in_frames / _fade_out_frames) are still linear.
    _ARI_XFADE_SEC = 3.0

    def read(self, n_frames: int):
        # ARI loop boundary with crossfade.
        # When we're within _ARI_XFADE_SEC of the loop end, open a second
        # stream from skip_time and blend it in while fading the current
        # stream out. When the loop boundary is reached, the new stream
        # becomes primary.
        if self._loop_frames > 0:
            remaining = self._loop_frames - self._loop_pos
            xfade_frames = int(self._ARI_XFADE_SEC * self._rate)

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
            xfade_frames = int(self._ARI_XFADE_SEC * self._rate)
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

                # EQUAL-POWER crossfade ramp (sin/cos quarter-wave).
                # The two streams here are DIFFERENT audio from the same file
                # (the tail of the ARI window vs the head at _skip), so they
                # are uncorrelated and sum in POWER, not amplitude. A linear
                # amplitude ramp gives a**2 + (1-a)**2, which bottoms out at
                # 0.5 mid-fade — a -3 dB sag on every loop (measured -1.9 dB
                # mean / -3.5 dB worst on two different ambient beds). sin/cos
                # holds sin**2 + cos**2 = 1 exactly, and unlike a sqrt ramp it
                # has zero slope at both ends so it joins the un-faded audio
                # either side without a kink.
                # NOTE: correct here precisely BECAUSE the streams are
                # uncorrelated. Do not copy this to a fade between two copies
                # of the SAME phase-aligned signal — that case wants linear.
                t_start = max(0, xf_pos) / max(1, xfade_frames)
                t_end   = max(0, xf_pos + xf_n) / max(1, xfade_frames)
                t_start = min(t_start, 1.0)
                t_end   = min(t_end, 1.0)
                t        = np.linspace(t_start, t_end, xf_n, dtype=np.float32)[:, np.newaxis]
                ramp_in  = np.sin(t * (np.pi / 2.0))
                ramp_out = np.cos(t * (np.pi / 2.0))

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

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        # Device + mix rate for THIS engine. Per-project (see
        # project.yaml ``audio.sample_rate``); switchable at runtime via
        # set_sample_rate() so a project swap can change it without a
        # restart. Every track built by the mixer inherits it.
        self.sample_rate = int(sample_rate)
        self._ring_target = self.sample_rate * RING_TARGET_MS // 1000
        self._ring_capacity = self.sample_rate * RING_CAPACITY_MS // 1000
        self._cmds: queue.SimpleQueue = queue.SimpleQueue()
        self._device = None
        self.master_volume = 1.0    # scales all audio output
        self.narrative_volume = 1.0  # scales non-ambient (oneshot) tracks in real time
        # Scales is_ambient tracks at mix time. Stories_OGL pushes the
        # active weather state's ``Sound_volume`` parameter here each
        # frame; weather editor edits to Sound_volume therefore take
        # effect immediately without restart. 1.0 = no attenuation.
        self.ambient_volume = 1.0
        # Scales is_soundpool tracks (random sound-pool clips) at mix time.
        # Stories_OGL pushes the web UI's soundpool_volume global modifier
        # here each frame, so the slider takes effect live and independently
        # of master / narrative / ambient. 1.0 = no attenuation.
        self.soundpool_volume = 1.0
        # Stereo balance, -1.0 (left only) .. +1.0 (right only), 0 = centered.
        # Panning is a linear attenuation of the OPPOSITE channel only — the
        # favored channel is never boosted, so balance can't clip a mix that
        # was otherwise in range. Pushed from the web UI's audio_balance
        # global modifier each frame by Stories_OGL.
        self.balance = 0.0
        # Monotonic counter feeding unique oneshot track keys. Generated on
        # the caller's thread in schedule_event (before the async decode) so
        # the caller can reference the track later via fade_out_event.
        self._evt_seq = itertools.count()
        # STOP GENERATION: stop_all() can only fade tracks that already
        # exist - a oneshot whose background DECODE was still in flight
        # would materialize AFTER the stop and play right through a set
        # switch (user-heard: the old set's music under the new set / the
        # DJ). stop_all bumps this; an in-flight decode captured the old
        # value and silently drops its post.
        self._stop_gen = 0
        # While True, schedule_event is a no-op (the DJ owns the
        # soundtrack: weather events keep driving visuals but must not
        # layer their sounds over the mix). Set by Stories_OGL._dj_start.
        self.oneshots_muted = False
        # Optional monitor tap: a callable(buf) invoked with each mixed output
        # block (shape (frames, CHANNELS) float32 @ self.sample_rate) on the audio
        # thread. Lets the audio analyzer react to the show's OWN output (the
        # "internal" source) without a device. None = no tap (zero cost).
        self._monitor_tap = None
        # Audio-callback deadline telemetry (see _mixer). Read via
        # callback_stats() to check whether this machine can actually
        # render the current mix in realtime.
        self._cb_count = 0
        self._cb_starved = 0
        self._cb_short_frames = 0
        self._cb_min_depth = self._ring_capacity
        self._cb_last_warn = 0.0
        # Render-ahead ring (see RING_TARGET_MS). Written by the producer
        # thread, read by the device callback; _ring_w / _ring_r are
        # monotonic frame counts, so depth is just w - r.
        self._ring = None
        self._ring_w = 0
        self._ring_r = 0
        self._ring_lock = threading.Lock()
        self._ring_stop = threading.Event()
        self._producer = None

    def callback_stats(self):
        """Render-ahead health since start:
        {callbacks, starved, short_frames, min_depth_ms}.
        `starved` counts callbacks the ring could not fill completely -
        i.e. real holes in the output. `min_depth_ms` is how close the ring
        ever came to running dry; if it stays well above zero the producer
        is keeping up comfortably."""
        return {"callbacks": self._cb_count,
                "starved": self._cb_starved,
                "short_frames": self._cb_short_frames,
                "min_depth_ms": self._cb_min_depth * 1000.0 / self.sample_rate}

    def render_lead_frames(self):
        """How far AHEAD of the speakers the renderer currently is, in
        frames - i.e. how much finished audio is sitting in the ring.

        Anything that drives visuals from render-time state (the DJ's deck
        telemetry, its transition/drop ETAs) must subtract this, or it fires
        a beat early. The monitor tap does NOT need it: that already runs on
        the consumer side.

        Deliberately excludes the device's own buffer. That lead existed
        before the ring and the visuals were tuned against it; this reports
        only what the ring ADDED, so compensating restores the previous
        relationship exactly rather than introducing a new one."""
        if self._ring is None:
            return 0
        with self._ring_lock:
            return max(0, self._ring_w - self._ring_r)

    def set_monitor_tap(self, fn):
        """Register (or clear with None) a callable invoked with each mixed
        output block. Must be cheap and non-blocking — it runs on the audio
        callback thread."""
        self._monitor_tap = fn

    # ------------------------------------------------------------------
    # Render-ahead ring: producer thread renders, device callback copies.
    # ------------------------------------------------------------------

    def _render_ahead(self):
        """Keep RING_TARGET frames of finished mix queued up.

        Runs on its own thread, so a slow pass here (a heavy DJ blend, a GIL
        stall) drains the ring instead of dropping audio. Only falls behind
        audibly if it cannot sustain realtime on average."""
        gen = self._mixer()
        next(gen)
        while not self._ring_stop.is_set():
            with self._ring_lock:
                depth = self._ring_w - self._ring_r
            if depth >= self._ring_target:
                # Full enough; nap briefly rather than spin (which would
                # burn a core and hold the GIL for no reason).
                self._ring_stop.wait(0.004)
                continue
            try:
                raw = gen.send(RENDER_BLOCK)
            except StopIteration:
                return
            chunk = np.frombuffer(raw, dtype=np.float32).reshape(-1, CHANNELS)
            n = len(chunk)
            with self._ring_lock:
                w = self._ring_w % self._ring_capacity
                k = min(n, self._ring_capacity - w)
                self._ring[w:w + k] = chunk[:k]
                if k < n:
                    self._ring[:n - k] = chunk[k:]
                self._ring_w += n

    def _device_feed(self):
        """The actual miniaudio callback: copy out of the ring, nothing else.

        No mixing, no allocation beyond one output block - so it needs the
        GIL for microseconds instead of the ~150ms the old inline renderer
        held it for, and it makes its deadline even while the show's Python
        threads are busy."""
        required = yield b""
        while True:
            out = np.zeros((required, CHANNELS), dtype=np.float32)
            with self._ring_lock:
                depth = self._ring_w - self._ring_r
                n = min(required, depth)
                if n > 0:
                    r = self._ring_r % self._ring_capacity
                    k = min(n, self._ring_capacity - r)
                    out[:k] = self._ring[r:r + k]
                    if k < n:
                        out[k:n] = self._ring[:n - k]
                    self._ring_r += n
                left = depth - n
            self._cb_count += 1
            if left < self._cb_min_depth:
                self._cb_min_depth = left
            if n < required:
                # The ring ran dry: this block is part silence. The only
                # remaining cause is the producer failing to sustain
                # realtime, which is a real "this machine cannot render
                # this mix" signal rather than a scheduling accident.
                self._cb_starved += 1
                self._cb_short_frames += required - n
                now = time.perf_counter()
                if now - self._cb_last_warn > 10.0:
                    self._cb_last_warn = now
                    print(f"[AudioEngine] ring underrun: filled {n}/{required} "
                          f"frames ({self._cb_starved}/{self._cb_count} "
                          f"callbacks, {self._cb_short_frames} frames lost) - "
                          f"the render thread cannot keep up")
            # Tap on the CONSUMER side so the analyzer - and every
            # audio-reactive shader - sees what is being played now, not
            # what was rendered RING_TARGET_MS ago.
            tap = self._monitor_tap
            if tap is not None:
                try:
                    # Pass the rate we are actually mixing at — the analyzer
                    # resamples to its own fixed target, and this engine's
                    # rate is per-project and can change at runtime.
                    tap(out, self.sample_rate)
                except Exception:
                    pass
            required = yield out.tobytes()

    # ------------------------------------------------------------------
    # Public API (safe to call from any thread)
    # ------------------------------------------------------------------

    def start(self):
        # GIL LATENCY: _mixer runs on miniaudio's callback thread and has a
        # hard deadline (one buffer, nothing rendered ahead - being late IS
        # a dropout). It competes for the GIL with pure-Python worker
        # threads (the DJ brain's step(), the planner, web handlers), and
        # CPython hands the GIL out with no priority: at the 5ms default a
        # memcpy-sized callback measured 264ms late with 3 pure-Python
        # threads running, and a rendering callback far worse (every
        # mid-render release re-queues behind them). Shortening the switch
        # interval bounds that wait - measured on this box, worst-case
        # lateness fell 510ms -> 32ms going from 5ms to 0.5ms. The cost is
        # more frequent thread switches; the mix loop is numpy-heavy (which
        # drops the GIL anyway), so it is not measurably slower.
        if sys.getswitchinterval() > 0.0005:
            sys.setswitchinterval(0.0005)

        self._ring = np.zeros((self._ring_capacity, CHANNELS), dtype=np.float32)
        self._ring_w = self._ring_r = 0
        self._ring_stop.clear()
        self._producer = threading.Thread(target=self._render_ahead,
                                          daemon=True, name="audio-render")
        self._producer.start()
        # PRE-FILL before the device opens: starting with an empty ring
        # would hand the DAC a ring-target's worth of silence (and count
        # every one of those blocks as an underrun).
        prefill_start = time.perf_counter()
        deadline = prefill_start + 5.0
        while time.perf_counter() < deadline:
            with self._ring_lock:
                if self._ring_w - self._ring_r >= self._ring_target:
                    break
            time.sleep(0.01)
        # start() runs on the caller's thread — the main render thread when a
        # project swap changes the rate — so a slow prefill stalls the show.
        # Normal is ~the ring target (400 ms); anything near the 5 s deadline
        # means the producer could not keep up and should be visible.
        prefill = time.perf_counter() - prefill_start
        if prefill > 1.0:
            print(f"[AudioEngine] ring prefill took {prefill:.2f}s "
                  f"(expected ~{RING_TARGET_MS / 1000:.1f}s)")

        self._device = miniaudio.PlaybackDevice(
            output_format=FORMAT,
            nchannels=CHANNELS,
            sample_rate=self.sample_rate,
            buffersize_msec=200,
        )
        gen = self._device_feed()
        next(gen)   # advance to first yield so miniaudio can send(framecount)
        self._device.start(gen)

    def set_sample_rate(self, rate: int) -> bool:
        """Switch the device + mix rate at runtime. Returns True if the
        engine is running at ``rate`` when this returns.

        Used by project swap: WoL runs at 48 kHz, Fan (and the whole DJ
        stack) at 44.1. A no-op when the rate already matches, so same-rate
        swaps cost nothing and never interrupt audio.

        This STOPS ALL AUDIO. Every track lives inside the _mixer generator,
        which dies with the producer thread, so the caller is expected to
        have faded things out already (Stories_OGL's swap does) and to
        re-start whatever should be playing afterwards.
        """
        rate = int(rate)
        if rate == self.sample_rate:
            return True
        prev = self.sample_rate
        print(f"[AudioEngine] sample rate {prev} -> {rate} Hz")

        # Invalidate decodes in flight BEFORE stopping: they were decoded at
        # the OLD rate and would otherwise be posted to the new mixer and
        # play at the wrong pitch. Same guard stop_all uses.
        self._stop_gen += 1
        self.stop()

        # Commands already queued carry old-rate payloads (an "oneshot_mem"
        # holds decoded samples). The queue survives stop/start, so the new
        # mixer would consume them — drop them.
        dropped = self._drain_cmds()
        if dropped:
            print(f"[AudioEngine]   dropped {dropped} queued command(s) "
                  f"built for {prev} Hz")

        def _apply(r):
            self.sample_rate = r
            self._ring_target = r * RING_TARGET_MS // 1000
            self._ring_capacity = r * RING_CAPACITY_MS // 1000
            self._cb_min_depth = self._ring_capacity

        _apply(rate)
        try:
            self.start()
            return True
        except Exception as e:
            # The old device is already closed, so failing here means
            # silence for the rest of the run. Fall back to the rate we
            # know worked rather than leave the show mute.
            print(f"[AudioEngine] device would not open at {rate} Hz ({e}); "
                  f"falling back to {prev} Hz")
            _apply(prev)
            try:
                self.start()
            except Exception as e2:
                print(f"[AudioEngine] FALLBACK ALSO FAILED ({e2}) - "
                      f"audio is now stopped")
            return False

    def _drain_cmds(self) -> int:
        """Discard every pending mixer command. Returns how many were tossed."""
        n = 0
        try:
            while True:
                self._cmds.get_nowait()
                n += 1
        except queue.Empty:
            pass
        return n

    def stop(self):
        # Device first: once it is closed nothing reads the ring, so the
        # producer can exit without racing a callback.
        if self._device:
            self._device.stop()
            self._device.close()
            self._device = None
        self._ring_stop.set()
        if self._producer is not None:
            self._producer.join(timeout=2.0)
            self._producer = None

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
        """Fade out all tracks (ambient + oneshots) AND invalidate any
        oneshot whose decode is still in flight (it would otherwise
        materialize after this stop and play right through)."""
        self._stop_gen += 1
        self._cmds.put(("stop_all", duration))

    def schedule_event(self, path, volume: float = 1.0, duration: float = 0.0,
                       narrative: bool = False, fade_in: float = 0.0,
                       soundpool: bool = False) -> str:
        """Play a sound file once. duration>0 caps playback to that many seconds.

        fade_in: seconds of linear 0->1 ramp at the start (0 = hard start).

        Decodes in a background thread to avoid blocking both the main thread
        and the audio callback thread. Returns the track's key, which can be
        passed to ``fade_out_event`` to fade this oneshot out later (e.g. to
        crossfade into the next clip). The key is valid even though decoding
        is still in flight — the matching track appears once decode finishes.
        """
        import threading
        p = Path(path)
        # Generate the key now, on the caller's thread, so it can be returned
        # before the async decode posts the track to the mixer.
        key = f"os_{p.name}_{time.monotonic_ns()}_{next(self._evt_seq)}"
        if self.oneshots_muted:
            return key            # DJ owns the soundtrack; key stays valid
        gen = self._stop_gen      # captured BEFORE the async decode
        # Decode at the rate that is current NOW. If set_sample_rate lands
        # while this decode is in flight it bumps _stop_gen, so the post
        # below is dropped rather than handing the new mixer old-rate audio.
        rate = self.sample_rate

        def _decode_and_queue():
            try:
                decoded = miniaudio.decode_file(
                    str(p), output_format=FORMAT,
                    nchannels=CHANNELS, sample_rate=rate)
                # Copy raw bytes first to own the memory independently of
                # miniaudio's C buffer, preventing GC race conditions
                raw_bytes = bytes(decoded.samples)
                samples = np.frombuffer(raw_bytes,
                                        dtype=np.float32).reshape(-1, CHANNELS)
                if gen != self._stop_gen or self.oneshots_muted:
                    # A stop_all (set switch / DJ takeover) landed while
                    # this decode was in flight - the show that scheduled
                    # this sound is gone. Posting it would play the OLD
                    # set's audio under the new one (the intermittent
                    # "previous set keeps playing" bug).
                    print(f"[AudioEngine] Dropped stale oneshot {p.name} "
                          "(stopped while decoding)")
                    return
                self._cmds.put(("oneshot_mem", samples, p, volume, duration,
                                narrative, fade_in, key, soundpool))
            except Exception as e:
                print(f"[AudioEngine] Failed to decode {p.name}: {e}")
        threading.Thread(target=_decode_and_queue, daemon=True).start()
        return key

    def fade_out_event(self, key: str, duration: float = FADE_OUT):
        """Fade out a previously scheduled oneshot, addressed by its key.

        No-op if the key is unknown (e.g. the clip already finished or its
        decode failed). Used by the sound pool to crossfade the outgoing clip
        as the next one fades in.
        """
        self._cmds.put(("fade_out_event", key, duration))

    def attach_track(self, key: str, track_obj):
        """Attach an externally-managed track object to the mixer.

        The object must implement the track protocol the mixer already
        consumes: ``read(n) -> (n, 2) float32 | None``, ``done``,
        ``fade_out(duration)`` and the ``is_narrative / is_ambient /
        is_soundpool`` flags. Used by the DJ submix (lib/dj/submix.py) to
        mount its whole two-deck mix as ONE track that rides the existing
        limiter, master volume and monitor tap.
        """
        self._cmds.put(("attach", key, track_obj))


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
                            fade_in=fi, loop_length=ari, is_ambient=True,
                            rate=self.sample_rate)
                        print(f"[AudioEngine] Ambient -> {path.name}")

                    elif kind == "stop_ambient":
                        _, fo = cmd
                        for t in tracks.values():
                            if t.is_ambient:
                                t.fade_out(fo)

                    elif kind == "stop_all":
                        _, fo = cmd
                        for t in tracks.values():
                            t.fade_out(fo)

                    elif kind == "fade_out_event":
                        _, key, fo = cmd
                        t = tracks.get(key)
                        if t is not None:
                            t.fade_out(fo)

                    elif kind == "oneshot_mem":
                        _, samples, path, vol, dur, narr, fade_in, key, sp = cmd
                        # Key is generated by schedule_event on the caller's
                        # thread so the caller can reference this track (e.g.
                        # to fade it out for a crossfade). id(cmd) was unsafe
                        # because Python can reuse the address of a freed
                        # tuple, silently overwriting a still-playing track.
                        tracks[key] = _MemTrack(
                            samples, path, volume=vol, duration=dur,
                            is_narrative=narr, fade_in=fade_in, is_soundpool=sp,
                            rate=self.sample_rate)
                        if not narr:
                            track_dur = len(samples) / self.sample_rate
                            finish_at = time.strftime(
                                "%H:%M:%S", time.localtime(time.time() + track_dur))
                            print(f"[AudioEngine] Oneshot > {path.name}  "
                                  f"dur={track_dur:.2f}s  finishes ~{finish_at}")

                    elif kind == "attach":
                        _, key, obj = cmd
                        tracks[key] = obj
                        print(f"[AudioEngine] Attached track '{key}'")


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
            sp_vol = self.soundpool_volume
            for key, track in tracks.items():
                try:
                    # _Track.read() is allowed to return a SHORT chunk: at an
                    # ARI loop boundary it caps the read at the frames left in
                    # the window (see `n_frames = min(n_frames, remaining)`).
                    # Reading once and mixing chunk[:len(chunk)] left the rest
                    # of the block as ZEROS — digital silence in the output at
                    # every single loop boundary. Harmless when the block was
                    # the device buffer (2-9 ms), but the render-ahead ring
                    # fixed it at RENDER_BLOCK=2048 and the hole became 25-45 ms:
                    # clearly audible as a blip in any looping ambient bed.
                    # Keep reading until the block is full or the track ends.
                    parts, need = [], required_frames
                    while need > 0:
                        part = track.read(need)
                        if part is None or len(part) == 0:
                            break
                        parts.append(part)
                        need -= len(part)
                        if track.done:
                            break
                    chunk = np.concatenate(parts) if parts else None
                except Exception:
                    dead.append(key)
                    continue
                # Mix whatever we got BEFORE reaping. The old order tested
                # `track.done` first and dropped the chunk entirely, so a
                # final short read that arrives together with done=True was
                # discarded rather than played.
                if chunk is not None and len(chunk):
                    if track.is_narrative:
                        narr_buf[:len(chunk)] += chunk * narr_vol
                    elif track.is_ambient:
                        # Ambient tracks scale by the per-state
                        # Sound_volume parameter pushed in from the
                        # weather scheduler — editing Sound_volume in
                        # the weather editor takes effect at the next
                        # frame, no restart needed.
                        other_buf[:len(chunk)] += chunk * amb_vol
                    elif track.is_soundpool:
                        # Random sound-pool clips scale by the live
                        # soundpool_volume slider (web global_modifiers),
                        # independent of master / narrative / ambient.
                        other_buf[:len(chunk)] += chunk * sp_vol
                    else:
                        other_buf[:len(chunk)] += chunk
                if chunk is None or track.done:
                    dead.append(key)
            for key in dead:
                del tracks[key]

            # Limit only the non-narrative bus
            peak = np.max(np.abs(other_buf))
            if peak > 2.0:
                other_buf /= (peak / 2.0)

            buf = (narr_buf + other_buf) * self.master_volume

            # Stereo balance: attenuate only the channel opposite the pan
            # direction (column 0 = left, 1 = right). Applied after master
            # volume so it acts on the final mix, including narrative.
            bal = self.balance
            if bal > 0.0:
                buf[:, 0] *= max(0.0, 1.0 - bal)
            elif bal < 0.0:
                buf[:, 1] *= max(0.0, 1.0 + bal)

            # NOTE: the monitor tap is NOT fired here. This generator runs on
            # the render-ahead thread, up to RING_TARGET_MS before the audio
            # is actually heard; tapping here would make every audio-reactive
            # shader lead the speakers by that much. _device_feed taps the
            # block it hands the device instead.
            required_frames = yield buf.tobytes()


# Backward-compatibility alias — render_pipeline.py instantiates this name.
ThreadedAudioEngine = AudioEngine
