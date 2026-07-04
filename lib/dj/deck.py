"""One DJ playback deck: RAM-decoded track -> loop -> WSOLA -> EQ -> gain.

The deck owns a VIRTUAL source cursor that grows monotonically; an active
loop is a mapping from virtual position to source position (with a short
equal-power crossfade at the seam), so the stretcher upstream never notices
the wrap. Decode happens on a daemon thread; `ready` flips when the samples
are in RAM (a 5-minute stereo track is ~106 MB float32 - fine on the PCs
this targets).
"""
import threading

import numpy as np

from lib.dj.eq import ThreeBandEQ
from lib.dj.stretch import WSOLAStretcher

RATE = 44100
LOOP_XFADE = 128                 # frames, equal-power seam blend


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
        self._gain_ramp = None   # (target, per_second)
        self.eq = ThreeBandEQ()
        self.loop = None         # (start_frame, end_frame) in source domain
        self._virt = 0           # virtual cursor (frames, int, output of map)
        self.stretch = WSOLAStretcher(self._fetch)
        self._lock = threading.Lock()

    # -- loading -------------------------------------------------------------
    def load(self, samples, track_id=None, grid=None, gain_db=0.0):
        self.samples = np.asarray(samples, dtype=np.float32)
        if self.samples.ndim == 1:
            self.samples = np.repeat(self.samples[:, None], 2, axis=1)
        self.track_id = track_id
        self.grid = grid or []
        self.loudness_gain = float(10.0 ** (gain_db / 20.0))
        self.finished = False
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

    # -- transport -----------------------------------------------------------
    def cue(self, time_s):
        """Position the cursor (deck stopped or playing)."""
        frame = int(time_s * RATE)
        self._virt = frame
        self.stretch.seek(frame)

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

    def read(self, n):
        """n output frames through the whole chain. Zeros when idle."""
        if not (self.playing and self.ready and self.samples is not None):
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
        self.stretch.rate = self.effective_rate()
        with self._lock:
            block = self.stretch.read(n).copy()
        if self.loop is None and self.stretch.source_pos >= len(self.samples):
            self.finished = True
        block = self.eq.process(block)
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
        return (block * getattr(self, "loudness_gain", 1.0)).astype(np.float32)
