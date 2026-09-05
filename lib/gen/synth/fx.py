"""Block-streaming effects with state carried across calls: tempo-synced
ping-pong delay, Freeverb-style reverb, kick sidechain duck, soft clip."""
from __future__ import annotations

import numpy as np

from lib.gen import RATE
from lib.gen.synth import dsp


class PingPongDelay:
    def __init__(self, time_samples: int, feedback: float = 0.42, damp_hz: float = 3500.0):
        self.L = max(int(time_samples), 64)
        self.buf = np.zeros((self.L, 2), dtype=np.float32)
        self.idx = 0
        self.fb = float(feedback)
        self.coef = float(1.0 - np.exp(-2 * np.pi * damp_hz / RATE))
        self._lp = [0.0, 0.0]

    def process(self, x):
        n = x.shape[0]
        out = np.empty_like(x)
        i = 0
        while i < n:
            m = min(n - i, self.L - self.idx, self.L)
            seg = self.buf[self.idx:self.idx + m]
            out[i:i + m] = seg
            # cross-feed: left tap feeds right line and vice versa
            fbk = np.empty_like(seg)
            fbk[:, 0] = seg[:, 1]
            fbk[:, 1] = seg[:, 0]
            inp = x[i:i + m]
            wl, self._lp[0] = dsp.onepole_lp((inp[:, 0] + fbk[:, 0] * self.fb).astype(np.float32), self.coef, self._lp[0])
            wr, self._lp[1] = dsp.onepole_lp((inp[:, 1] + fbk[:, 1] * self.fb).astype(np.float32), self.coef, self._lp[1])
            self.buf[self.idx:self.idx + m, 0] = wl
            self.buf[self.idx:self.idx + m, 1] = wr
            self.idx = (self.idx + m) % self.L
            i += m
        return out


class Reverb:
    """Freeverb: 8 lowpass-feedback combs + 4 allpasses per channel, the
    right channel offset by 23 samples for width."""
    _COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
    _APS = (556, 441, 341, 225)

    def __init__(self, room: float = 0.82, damp: float = 0.35, size_scale: float = 1.0):
        self.fb = float(room)
        self.damp = float(damp)
        self.combs = [[np.zeros(int(c * size_scale) + off, dtype=np.float32) for c in self._COMBS] for off in (0, 23)]
        self.cidx = [[0] * len(self._COMBS) for _ in range(2)]
        self.clp = [[0.0] * len(self._COMBS) for _ in range(2)]
        self.aps = [[np.zeros(a + off, dtype=np.float32) for a in self._APS] for off in (0, 23)]
        self.aidx = [[0] * len(self._APS) for _ in range(2)]

    def process(self, x):
        n = x.shape[0]
        out = np.empty_like(x)
        mono = ((x[:, 0] + x[:, 1]) * 0.5).astype(np.float32)
        for ch in range(2):
            acc = np.zeros(n, dtype=np.float32)
            for k in range(len(self._COMBS)):
                y, self.cidx[ch][k], self.clp[ch][k] = dsp.comb(mono, self.combs[ch][k], self.cidx[ch][k], self.fb, self.damp, self.clp[ch][k])
                acc += y
            acc *= 0.125
            for k in range(len(self._APS)):
                acc, self.aidx[ch][k] = dsp.allpass(acc, self.aps[ch][k], self.aidx[ch][k], 0.5)
            out[:, ch] = acc
        return out


def duck_curve(n, clock, kick_times, depth=0.5, tau=0.11, hold=0.01):
    """Gain curve (n,) for a block starting at `clock`, from kick onsets
    (absolute samples). Sidechain pump: dip to 1-depth then recover."""
    g = np.ones(n, dtype=np.float32)
    if not kick_times:
        return g
    t = np.arange(n, dtype=np.float32) + clock
    for k in kick_times:
        dt = (t - k) / RATE
        m = dt >= 0
        if not m.any():
            continue
        env = np.where(dt < hold, 1.0, np.exp(-(dt - hold) / tau))
        g = np.minimum(g, np.where(m, 1.0 - depth * env, 1.0)).astype(np.float32)
    return g


def softclip(x, drive=1.0):
    """Bounded to (-1, 1): tanh with a pre-gain (unity slope near 0 is
    restored by 1/drive so quiet material is untouched)."""
    return (np.tanh(x * drive) / drive).astype(np.float32) if drive <= 1.0 else np.tanh(x * drive).astype(np.float32)
