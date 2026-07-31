"""Three-band DJ EQ: Linkwitz-Riley 4th-order crossovers at 200 / 2500 Hz.

LR4 = squared 2nd-order Butterworth, so the three bands sum back to (almost)
flat when all gains are 1.0 - killing the low band during a bass swap leaves
mids/highs untouched. Stateful (scipy sosfilt zi per channel), block-based,
with per-block linear gain ramps so automation never zipper-clicks.
"""
import numpy as np
from scipy.signal import butter, sosfilt

RATE = 44100
LOW_HZ = 200.0
HIGH_HZ = 2500.0


def _lr4(freq, kind):
    """4th-order Linkwitz-Riley as a cascade of two 2nd-order Butterworths."""
    sos = butter(2, freq, kind, fs=RATE, output="sos")
    return np.vstack([sos, sos])


class ThreeBandEQ:
    def __init__(self, channels=2):
        self.channels = channels
        # One cascade per band, both channels filtered in a single
        # sosfilt call (axis=0). scipy's per-call Python overhead (~90us
        # of arg validation) dominated the actual filtering at 256-frame
        # sub-blocks - the old per-channel/per-stage layout was 16 calls
        # per process(), this is 3 (perf audit 2026-07-31). The merged
        # mid cascade (band-pass = HP then LP) is the same math: a
        # cascade of cascades is one longer cascade.
        self._sos = {
            "low": _lr4(LOW_HZ, "low"),
            "mid": np.vstack([_lr4(LOW_HZ, "high"), _lr4(HIGH_HZ, "low")]),
            "high": _lr4(HIGH_HZ, "high"),
        }
        self._zi = {k: np.zeros((len(s), 2, channels))
                    for k, s in self._sos.items()}
        self.gains = np.ones(3)          # current low/mid/high
        self._targets = np.ones(3)
        self._ramp_left_s = 0.0
        self._active = False

    def set_gain(self, band, gain, ramp_s=0.05):
        """band: 0=low 1=mid 2=high; gain is linear (0..~1.5)."""
        self._targets[band] = float(gain)
        self._ramp_left_s = max(self._ramp_left_s, float(ramp_s))

    def set_gains(self, low=None, mid=None, high=None, ramp_s=0.05):
        for i, g in enumerate((low, mid, high)):
            if g is not None:
                self._targets[i] = float(g)
        self._ramp_left_s = max(self._ramp_left_s, float(ramp_s))

    @property
    def is_flat(self):
        return (np.all(np.abs(self.gains - 1.0) < 1e-3)
                and np.all(np.abs(self._targets - 1.0) < 1e-3))

    def _run(self, key, x):
        y, self._zi[key] = sosfilt(self._sos[key], x, axis=0,
                                   zi=self._zi[key])
        return y

    def process(self, block):
        """block: (n, channels) float32. Returns the EQ'd block.

        Bit-exact bypass while flat. The LR4 band sum is an ALLPASS (phase
        rotates near the crossovers), so bypass<->active switches crossfade
        over one block instead of hard-cutting - a hard cut would click."""
        n = len(block)
        if n == 0:
            return block
        dt = n / RATE
        if self._ramp_left_s <= 0.0 and np.all(self._targets == self.gains):
            g0 = g1 = self.gains
        else:
            frac = 1.0 if self._ramp_left_s <= dt else dt / self._ramp_left_s
            g0 = self.gains.copy()
            g1 = g0 + (self._targets - g0) * frac
            self.gains = g1
            self._ramp_left_s = max(0.0, self._ramp_left_s - dt)
        flat = np.all(g0 == 1.0) and np.all(g1 == 1.0)
        active = getattr(self, "_active", False)
        if flat and not active:
            return block                                  # true bypass
        x = block.astype(np.float64)
        low = self._run("low", x)
        mid = self._run("mid", x)
        high = self._run("high", x)
        ramp = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        gl = g0[0] + (g1[0] - g0[0]) * ramp
        gm = g0[1] + (g1[1] - g0[1]) * ramp
        gh = g0[2] + (g1[2] - g0[2]) * ramp
        y = low * gl + mid * gm + high * gh
        if not active:                                    # fade processed in
            y = x + (y - x) * ramp
            self._active = True
        elif flat:                                        # fade back to raw
            y = y + (x - y) * ramp
            self._active = False
            for k in self._zi:                            # cold state is fine:
                self._zi[k][:] = 0.0                      # next engage fades in
        return y.astype(np.float32)
