"""One DJ playback deck: RAM-decoded track -> loop -> WSOLA -> EQ -> gain.

The deck owns a VIRTUAL source cursor that grows monotonically; an active
loop is a mapping from virtual position to source position (with a short
equal-power crossfade at the seam), so the stretcher upstream never notices
the wrap. Decode happens on a daemon thread; `ready` flips when the samples
are in RAM (a 5-minute stereo track is ~106 MB float32 - fine on the PCs
this targets).
"""
import os
import threading

import numpy as np

from lib.dj import stretch_engine_name
from lib.dj.eq import ThreeBandEQ
from lib.dj.fx import EchoDelay, SweepFilter
from lib.dj.stretch import WSOLAStretcher
from lib.dj.pv_stretch import PhaseVocoderStretcher
from lib.dj.varispeed import VarispeedStretcher

# Tempo engine (env DJ_STRETCH_ENGINE, see lib.dj.stretch_engine_name).
# 'vari' (default): turntable mode - pitch rides tempo, zero stretch
# artifacts; the brain meets both decks in the middle so each song shifts
# half as far. 'wsola'/'pv': keylock engines (constant pitch, each with its
# own artifact tradeoff) for A/B by ear.
_ENGINES = {"pv": PhaseVocoderStretcher, "wsola": WSOLAStretcher,
            "vari": VarispeedStretcher}


def _stretch_engine():
    return _ENGINES.get(stretch_engine_name(), VarispeedStretcher)

RATE = 44100
LOOP_XFADE = 128                 # frames, equal-power seam blend
ENV_FPS = 200                    # output kick-envelope rate (5ms bins)
ENV_LEN = 2 * ENV_FPS            # ring holds the last 2s
_ENV_GROUP = RATE // ENV_FPS     # samples per envelope bin


class Deck:
    def __init__(self, name="deck"):
        self.name = name
        self.samples = None      # (n, 2) float32
        self.ready = False
        self.playing = False
        self.finished = False    # cursor ran off the end while playing
        self.track_id = None
        self.grid = []           # beat-grid segments from the DB
        self.rate = 1.0          # base rate (automation ramps this)
        self.rate_trim = 0.0     # PLL micro-correction, +/- ~0.003
        self._rate_ramp = None   # (target, per_second)
        self.gain = 0.0          # current linear gain
        self.kick_offset_s = 0.0 # per-track groove offset (analysis v6)
        self._gain_ramp = None   # (target, per_second)
        self.eq = ThreeBandEQ()
        self.filter = SweepFilter()      # resonant sweep (post-EQ)
        self.echo = EchoDelay()          # echo-out tail (post-gain)
        self._brake = None       # (speed, decel_per_s, src_pos) - vinyl stop
        self._fade_in = 0        # frames of fade-in after a jump cut
        # KEY SHIFT: pitch factor f = 2^(st/12). The stretcher runs at
        # rate/f and a streaming resampler scales the block by f - net
        # tempo unchanged, pitch shifted, and the deck's TIMELINE (grid,
        # cues, beat phase) stays in original track time throughout.
        self._pitch_f = 1.0
        self._rs_buf = np.zeros((0, 2), dtype=np.float32)
        self._rs_pos = 0.0
        self.loop = None         # (start_frame, end_frame) in source domain
        self._virt = 0           # virtual cursor (frames, int, output of map)
        self.stretch = _stretch_engine()(self._fetch)
        self._lock = threading.Lock()
        # Running FULL-BAND amplitude envelope of this deck's OUTPUT (pre-
        # EQ/pre-gain). The submix correlates the POSITIVE DIFFERENCE of two
        # decks' rings - i.e. their TRANSIENTS (kick/hat attacks), which sit
        # on the grid. (Low-band envelopes were tried and align the OFFBEAT
        # bass stabs this genre loves, actively fighting kick alignment.)
        self.out_env = np.zeros(ENV_LEN, dtype=np.float32)
        self._env_rem = np.zeros(0, dtype=np.float64)

    # -- loading -------------------------------------------------------------
    def set_pitch(self, semitones):
        self._pitch_f = float(2.0 ** (semitones / 12.0))
        self._rs_buf = self._rs_buf[:0]
        self._rs_pos = 0.0

    def load(self, samples, track_id=None, grid=None, gain_db=0.0,
             kick_offset_s=0.0, pitch_st=0.0):
        self.samples = np.asarray(samples, dtype=np.float32)
        if self.samples.ndim == 1:
            self.samples = np.repeat(self.samples[:, None], 2, axis=1)
        self.track_id = track_id
        self.grid = grid or []
        self.loudness_gain = float(10.0 ** (gain_db / 20.0))
        self.kick_offset_s = float(kick_offset_s or 0.0)
        self.set_pitch(pitch_st or 0.0)
        self.finished = False
        self.reset_fx()                          # no stale FX across tracks
        self.ready = True

    def load_file_async(self, path, track_id=None, grid=None, gain_db=0.0):
        self.ready = False

        def _work():
            try:
                from lib.dj.features import decode_file_stereo
                samples = decode_file_stereo(path)
                with self._lock:
                    self.load(samples, track_id, grid, gain_db)
            except Exception as e:
                print(f"[DJ deck {self.name}] decode failed for {path}: {e}")
        threading.Thread(target=_work, daemon=True).start()

    def unload(self):
        self.playing = False
        self.ready = False
        self.samples = None
        self.track_id = None
        self.loop = None
        self.finished = False
        self.reset_fx()

    def reset_fx(self):
        self.filter.set(mode="off")
        self.echo.set(active=False)
        # EQ must not leak across tracks: a deck that finished as the
        # OUTGOING side of a bass swap sits at low=0 / mid=0.25, and any
        # style that doesn't explicitly restate EQ (long_fade set gain
        # only) played the next track with its whole bass stripped
        # (user-heard). Every load starts flat; styles shape from there.
        self.eq.set_gains(1.0, 1.0, 1.0, ramp_s=0.01)
        self._brake = None
        self._fade_in = 0

    # -- transport -----------------------------------------------------------
    def cue(self, time_s):
        """Position the cursor (deck stopped or playing)."""
        frame = int(time_s * RATE)
        self._virt = frame
        self.stretch.seek(frame)
        self._rs_buf = self._rs_buf[:0]
        self._rs_pos = 0.0
        self.out_env[:] = 0.0                    # stale rhythm history
        self._env_rem = self._env_rem[:0]

    def start(self):
        if self.ready:
            self.playing = True

    def stop(self):
        self.playing = False

    def set_loop(self, start_s, end_s):
        if end_s > start_s:
            self.loop = (int(start_s * RATE), int(end_s * RATE))

    def clear_loop(self):
        self.loop = None

    def release_loop(self):
        """Exit an active loop AT its end so playback continues PAST it
        (into the drop), instead of jumping to the run-away virtual cursor
        a bare clear would leave. No-op if not looping."""
        if self.loop is not None:
            _, le = self.loop
            self.stretch.seek(le)
            self.loop = None

    def brake(self, duration_s=1.5):
        """Vinyl brake: playback decelerates to a stop WITH the pitch
        falling (variable-speed resample, not WSOLA - a brake IS a pitch
        drop). The deck stops itself when the platter reaches zero."""
        if self.playing:
            self._brake = [1.0, 1.0 / max(duration_s, 0.1),
                           float(self.stretch.source_pos)]

    def jump_cut(self, time_s):
        """Phrase-jump edit: hard relocate at a downbeat with a short
        fade-in so the discontinuity never clicks. Used by the live
        micro-arrangement (skip a dull phrase, extend a breakdown)."""
        self.stretch.seek(int(time_s * RATE))
        self._fade_in = 192

    def set_gain(self, target, ramp_s=0.05):
        ramp_s = max(ramp_s, 1e-3)
        self._gain_ramp = (float(target),
                           abs(float(target) - self.gain) / ramp_s)

    def set_rate(self, target, ramp_s=0.0):
        target = float(np.clip(target, 0.90, 1.10))
        if ramp_s <= 0.0:
            self.rate = target
            self._rate_ramp = None
        else:
            self._rate_ramp = (target, abs(target - self.rate) / ramp_s)

    # -- position / telemetry --------------------------------------------------
    def _map_source(self, virt):
        """Virtual (monotonic) frame -> source frame through the loop."""
        if self.loop is not None:
            ls, le = self.loop
            if virt >= le:
                return ls + (virt - ls) % (le - ls)
        return virt

    def source_time_s(self):
        return self._map_source(int(self.stretch.source_pos)) / RATE

    def beat_period_s(self):
        t = self.source_time_s()
        for g in self.grid:
            if g["start_s"] <= t <= g["end_s"]:
                return g["period_s"]
        return self.grid[0]["period_s"] if self.grid else 0.0

    def _current_seg(self):
        t = self.source_time_s()
        for seg in self.grid:
            if seg["start_s"] <= t <= seg["end_s"]:
                return seg
        return self.grid[0] if self.grid else None

    def beat_phase(self):
        """Playback-domain beat phase in [0,1), from the DB grid."""
        g = self._current_seg()
        if g is None or g["period_s"] <= 0:
            return 0.0
        return ((self.source_time_s() - g["first_beat_s"])
                / g["period_s"]) % 1.0

    def nudge_seconds(self, dt_s):
        """Shift the play cursor by dt_s (source-domain seconds); used by
        onset-anchored sync to slide the deck onto the master's kicks."""
        self.stretch.seek(int(round(self.stretch.source_pos + dt_s * RATE)))

    def phase_snap(self, target_phase):
        """Instantly shift the play cursor so beat_phase == target_phase -
        the DJ 'sync' snap. Moves by the SHORTEST direction (<=half a beat)
        and only safe while inaudible (gain ~0); the stretcher reseeks, so
        callers must gate on low gain. Returns beats shifted."""
        g = self._current_seg()
        if g is None or g["period_s"] <= 0:
            return 0.0
        err = (self.beat_phase() - target_phase + 0.5) % 1.0 - 0.5
        shift_frames = -err * g["period_s"] * RATE      # move back if ahead
        self.stretch.seek(int(round(self.stretch.source_pos + shift_frames)))
        return err

    def effective_rate(self):
        return self.rate * (1.0 + self.rate_trim)

    # -- audio ---------------------------------------------------------------
    def _fetch(self, pos, n):
        """Stretcher pull: n frames at virtual position pos, loop-mapped,
        zero-padded past the end (equal-power blend at the loop seam)."""
        src = self.samples
        if src is None:
            return np.zeros((n, 2), dtype=np.float32)
        out = np.empty((n, 2), dtype=np.float32)
        if self.loop is None:
            a = max(0, min(pos, len(src)))
            b = max(0, min(pos + n, len(src)))
            got = b - a
            k0 = a - pos
            out[:k0] = 0.0
            out[k0:k0 + got] = src[a:b]
            out[k0 + got:] = 0.0
            return out
        ls, le = self.loop
        span = le - ls
        v = pos + np.arange(n)
        s = np.where(v < le, v, ls + (v - ls) % span)
        s_ok = np.clip(s, 0, len(src) - 1)
        out[:] = src[s_ok]
        out[(s < 0) | (s >= len(src))] = 0.0
        # Equal-power blend across the seam: during the first LOOP_XFADE
        # frames after each wrap, mix in the audio from just before the seam.
        seam = (v >= le) & (s - ls < LOOP_XFADE)
        if np.any(seam):
            f = ((s[seam] - ls) / LOOP_XFADE)[:, None]
            pre = np.clip(le - LOOP_XFADE + (s[seam] - ls), 0, len(src) - 1)
            out[seam] = (src[np.clip(s[seam], 0, len(src) - 1)] * np.sqrt(f)
                         + src[pre] * np.sqrt(1.0 - f))
        return out

    def _feed_env(self, block):
        """Append this output block's amplitude envelope to the ring."""
        mono = block.mean(axis=1).astype(np.float64)
        buf = np.concatenate([self._env_rem, np.abs(mono)])
        nbin = len(buf) // _ENV_GROUP
        if nbin:
            vals = buf[:nbin * _ENV_GROUP].reshape(nbin, _ENV_GROUP).max(1)
            if nbin >= ENV_LEN:
                self.out_env[:] = vals[-ENV_LEN:]
            else:
                self.out_env[:-nbin] = self.out_env[nbin:]
                self.out_env[-nbin:] = vals
        self._env_rem = buf[nbin * _ENV_GROUP:]

    def _read_brake(self, n):
        """Variable-speed interpolated read during a vinyl brake."""
        speed, decel, pos = self._brake
        dt = 1.0 / RATE
        idx = np.empty(n)
        for i in range(n):                       # 44k simple ops/s max: fine
            idx[i] = pos
            pos += speed
            speed -= decel * dt
            if speed <= 0.0:
                idx[i + 1:] = pos
                self._brake = None
                self.playing = False
                break
        else:
            self._brake = [speed, decel, pos]
        src = self.samples
        i0 = np.clip(idx.astype(np.int64), 0, len(src) - 2)
        fr = (idx - i0)[:, None]
        block = src[i0] * (1 - fr) + src[i0 + 1] * fr
        if self._brake is None:                  # fade the final grains out
            block *= np.linspace(1.0, 0.0, n)[:, None] ** 0.5
        return block.astype(np.float64)

    def _read_pitched(self, n):
        """Streaming linear-interp resample of the stretcher output by
        _pitch_f - the second half of the key shift (see __init__)."""
        f = self._pitch_f
        need = int(np.ceil(self._rs_pos + n * f)) + 2 - len(self._rs_buf)
        if need > 0:
            self._rs_buf = np.concatenate(
                [self._rs_buf, self.stretch.read(need)])
        idx = self._rs_pos + np.arange(n) * f
        i0 = idx.astype(np.int64)
        fr = (idx - i0)[:, None]
        out = (self._rs_buf[i0] * (1 - fr)
               + self._rs_buf[i0 + 1] * fr).astype(np.float32)
        nxt = idx[-1] + f
        k = int(nxt)
        self._rs_buf = self._rs_buf[k:]
        self._rs_pos = nxt - k
        return out

    def read(self, n):
        """n output frames through the whole chain. Zeros when idle."""
        if not (self.playing and self.ready and self.samples is not None):
            if self.echo.ringing:                # echo tail outlives the deck
                tail = self.echo.process(np.zeros((n, 2), dtype=np.float32))
                return (tail * getattr(self, "loudness_gain", 1.0)
                        ).astype(np.float32)
            return np.zeros((n, 2), dtype=np.float32)
        dt = n / RATE
        if self._rate_ramp is not None:
            target, speed = self._rate_ramp
            step = speed * dt
            if abs(target - self.rate) <= step:
                self.rate = target
                self._rate_ramp = None
            else:
                self.rate += step if target > self.rate else -step
        self.stretch.rate = self.effective_rate() / self._pitch_f
        with self._lock:
            if self._brake is not None:
                block = self._read_brake(n)
            elif self._pitch_f != 1.0:
                block = self._read_pitched(n)
            else:
                block = self.stretch.read(n).copy()
        if self.loop is None and self.stretch.source_pos >= len(self.samples):
            self.finished = True
        if self._fade_in > 0:                    # de-click after a jump cut
            total = 192.0
            k = min(self._fade_in, n)
            f0 = 1.0 - self._fade_in / total
            f1 = 1.0 - (self._fade_in - k) / total
            block[:k] *= np.linspace(f0, f1, k)[:, None]
            self._fade_in -= k
        self._feed_env(block)                    # pre-EQ/gain rhythm tap
        block = self.eq.process(block)
        block = self.filter.process(block)
        g0 = self.gain
        if self._gain_ramp is not None:
            target, speed = self._gain_ramp
            step = speed * dt
            if abs(target - self.gain) <= step:
                self.gain = target
                self._gain_ramp = None
            else:
                self.gain += step if target > self.gain else -step
        if g0 == self.gain:
            if self.gain != 1.0:
                block = block * self.gain
        else:
            block = block * np.linspace(g0, self.gain, n)[:, None]
        # Echo sits AFTER the gain stage: cut the deck and the captured
        # tail keeps ringing over the incoming track (the echo-out exit).
        if self.echo.active:
            block = self.echo.process(block)
        return (block * getattr(self, "loudness_gain", 1.0)).astype(np.float32)
