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


class Limiter:
    """Master peak limiter (stereo-linked, no lookahead): holds transients
    under `ceiling` with a ~0.3 ms attack and a ~120 ms release, so the
    kick keeps its crest instead of being flattened by a hard tanh."""

    def __init__(self, ceiling: float = 0.92, attack_s: float = 0.0003, release_s: float = 0.12):
        self.ceiling = float(ceiling)
        self.att = float(np.exp(-1.0 / max(attack_s * RATE, 1.0)))
        self.rel = float(np.exp(-1.0 / max(release_s * RATE, 1.0)))
        self.gain = 1.0

    def process(self, x):
        out, self.gain = dsp.peak_limiter(np.ascontiguousarray(x, dtype=np.float32), self.ceiling, self.att, self.rel, self.gain)
        return out


class Compressor:
    """Bus compressor (stereo-linked peak detector). thresh_db, ratio,
    attack/release in seconds, makeup_db."""

    def __init__(self, thresh_db=-12.0, ratio=3.0, attack_s=0.005, release_s=0.1, makeup_db=3.0):
        self.thresh = float(10 ** (thresh_db / 20.0))
        self.ratio = float(ratio)
        self.att = float(np.exp(-1.0 / max(attack_s * RATE, 1.0)))
        self.rel = float(np.exp(-1.0 / max(release_s * RATE, 1.0)))
        self.makeup = float(10 ** (makeup_db / 20.0))
        self.env = 0.0

    def process(self, x):
        out, self.env = dsp.compressor(np.ascontiguousarray(x, dtype=np.float32), self.thresh, self.ratio,
                                       self.att, self.rel, self.makeup, self.env)
        return out


class Biquad:
    """Stereo biquad: kind = highpass | lowpass | highshelf | lowshelf |
    peak. Coefficients from the Audio EQ Cookbook."""

    def __init__(self, kind, freq_hz, q=0.707, gain_db=0.0):
        self.set(kind, freq_hz, q, gain_db)
        self.z = [[0.0, 0.0], [0.0, 0.0]]

    def set(self, kind, freq_hz, q=0.707, gain_db=0.0):
        w0 = 2.0 * np.pi * float(freq_hz) / RATE
        cw, sw = np.cos(w0), np.sin(w0)
        alpha = sw / (2.0 * q)
        A = 10 ** (gain_db / 40.0)
        if kind == "highpass":
            b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
            a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
        elif kind == "lowpass":
            b0, b1, b2 = (1 - cw) / 2, 1 - cw, (1 - cw) / 2
            a0, a1, a2 = 1 + alpha, -2 * cw, 1 - alpha
        elif kind == "highshelf":
            sa = 2 * np.sqrt(A) * alpha
            b0 = A * ((A + 1) + (A - 1) * cw + sa)
            b1 = -2 * A * ((A - 1) + (A + 1) * cw)
            b2 = A * ((A + 1) + (A - 1) * cw - sa)
            a0 = (A + 1) - (A - 1) * cw + sa
            a1 = 2 * ((A - 1) - (A + 1) * cw)
            a2 = (A + 1) - (A - 1) * cw - sa
        elif kind == "lowshelf":
            sa = 2 * np.sqrt(A) * alpha
            b0 = A * ((A + 1) - (A - 1) * cw + sa)
            b1 = 2 * A * ((A - 1) - (A + 1) * cw)
            b2 = A * ((A + 1) - (A - 1) * cw - sa)
            a0 = (A + 1) + (A - 1) * cw + sa
            a1 = -2 * ((A - 1) + (A + 1) * cw)
            a2 = (A + 1) + (A - 1) * cw - sa
        else:  # peak
            b0, b1, b2 = 1 + alpha * A, -2 * cw, 1 - alpha * A
            a0, a1, a2 = 1 + alpha / A, -2 * cw, 1 - alpha / A
        self.c = tuple(float(v) for v in (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0))

    def process(self, x):
        out = np.empty_like(x)
        b0, b1, b2, a1, a2 = self.c
        for ch in range(2):
            y, z1, z2 = dsp.biquad(np.ascontiguousarray(x[:, ch], dtype=np.float32), b0, b1, b2, a1, a2,
                                   self.z[ch][0], self.z[ch][1])
            out[:, ch] = y
            self.z[ch] = [z1, z2]
        return out


class Saturator:
    """tanh drive with gain compensation: harmonics on the bass without
    a level jump."""

    def __init__(self, drive=1.8):
        self.drive = float(drive)
        self.comp = float(1.0 / np.tanh(min(self.drive, 4.0) * 0.5) * 0.5)

    def process(self, x):
        return (np.tanh(x * self.drive) * self.comp).astype(np.float32)



class FDNReverb:
    """Modulated 8-line feedback delay network behind two diffusion
    allpasses: a smooth, wide tail with no metallic ring. size scales
    the line lengths (0.5 small room .. 2 hall); decay is the loop gain;
    damp rolls the top off in the tail."""
    _LENS = (1487, 1723, 2011, 2371, 2689, 2971, 3299, 3607)   # ~34-82 ms, primes
    _APS = (347, 113)

    def __init__(self, size: float = 1.0, decay: float = 0.78, damp: float = 0.3, mod_depth: float = 9.0):
        self.size = float(size)
        self.decay = float(decay)
        self.damp = float(damp)
        self.mod_depth = float(mod_depth)
        self.lens = np.array([int(L * self.size) for L in self._LENS], dtype=np.int64)
        lmax = int(self.lens.max() + mod_depth + 4)
        self.lines = np.zeros((8, lmax), dtype=np.float32)
        self.idx = np.zeros(8, dtype=np.int64)
        self.lp = np.zeros(8, dtype=np.float64)
        self.mod_rates = np.array([0.11, 0.13, 0.17, 0.19, 0.23, 0.29, 0.31, 0.37], dtype=np.float64)
        self.phase = 0.0
        self.aps = [[np.zeros(a + off, dtype=np.float32) for a in self._APS] for off in (0, 17)]
        self.aidx = [[0, 0], [0, 0]]

    def set(self, decay=None, damp=None):
        if decay is not None:
            self.decay = float(max(0.0, min(0.97, decay)))
        if damp is not None:
            self.damp = float(max(0.0, min(0.95, damp)))

    def process(self, x):
        n = x.shape[0]
        pre = np.empty_like(x)
        for ch in range(2):
            v = np.ascontiguousarray(x[:, ch], dtype=np.float32)
            for k in range(len(self._APS)):
                v, self.aidx[ch][k] = dsp.allpass(v, self.aps[ch][k], self.aidx[ch][k], 0.6)
            pre[:, ch] = v
        out = np.empty_like(x)
        self.phase = dsp.fdn_reverb(pre, self.lines, self.lens, self.idx, self.lp, self.decay, self.damp,
                                    self.mod_depth, self.mod_rates, self.phase, float(RATE), out)
        return out


class Chorus:
    """Stereo chorus send: modulated delay, LFOs a quarter turn apart."""

    def __init__(self, rate_hz: float = 0.6, depth_ms: float = 6.0, base_ms: float = 12.0):
        self.rate = float(rate_hz)
        self.depth = float(depth_ms * RATE / 1000.0)
        self.base = float(base_ms * RATE / 1000.0)
        L = int(self.base + self.depth + 64)
        self.buf = np.zeros((L, 2), dtype=np.float32)
        self.idx = 0
        self.phase = 0.0

    def process(self, x):
        out = np.empty_like(x)
        self.idx, self.phase = dsp.chorus(np.ascontiguousarray(x, dtype=np.float32), self.buf, self.idx, self.base,
                                          self.depth, self.rate, self.phase, float(RATE), out)
        return out


class LookaheadLimiter:
    """True-peak-style limiter: the signal is delayed by `lookahead`
    samples and the gain is computed from the peak over the samples that
    are about to play, so nothing ever exceeds `ceiling` (no overshoot,
    no need for a hard clip behind it). Release is a one-pole."""

    def __init__(self, ceiling: float = 0.95, lookahead: int = 64, release_s: float = 0.08):
        self.ceiling = float(ceiling)
        self.L = int(lookahead)
        self.rel = float(np.exp(-1.0 / max(release_s * RATE, 1.0)))
        self.delay = np.zeros((self.L, 2), dtype=np.float32)
        self.gain = 1.0

    def process(self, x):
        n = x.shape[0]
        L = self.L
        joined = np.concatenate([self.delay, x], axis=0)           # (L+n, 2)
        peak = np.abs(joined).max(axis=1)                          # (L+n,)
        # for each output sample i (= joined[i]) look at joined[i .. i+L]
        win = np.lib.stride_tricks.sliding_window_view(np.concatenate([peak, np.zeros(L, dtype=peak.dtype)]), L + 1)[:n]
        pk = win.max(axis=1)
        target = np.where(pk > self.ceiling, self.ceiling / np.maximum(pk, 1e-9), 1.0).astype(np.float32)
        g, self.gain = dsp.gain_smooth(target, self.gain, self.rel)
        out = joined[:n] * g[:, None]
        self.delay = joined[n:].copy()
        return out.astype(np.float32)


class Loudness:
    """Program loudness, K-weighted-ish (high shelf +4 dB at 1.5 kHz over
    a 38 Hz high-pass), integrated by a slow one-pole (~3 s). lufs()
    returns the running estimate; a gate below -45 keeps silence from
    dragging it. Used by the rack to hold a per-style target."""

    def __init__(self, tau_s: float = 3.0):
        self.hp = Biquad("highpass", 38.0, 0.5)
        self.shelf = Biquad("highshelf", 1500.0, 0.7, 4.0)
        self.coef = float(np.exp(-1.0 / max(tau_s * RATE / 1024.0, 1.0)))   # per 1024-sample block
        self.ms = 1e-6
        self.blocks = 0

    def feed(self, x):
        y = self.shelf.process(self.hp.process(np.ascontiguousarray(x, dtype=np.float32)))
        ms = float(np.mean(y.astype(np.float64) ** 2))
        k = self.coef ** (x.shape[0] / 1024.0)
        if ms > 1e-9:
            self.ms = ms + (self.ms - ms) * k
        self.blocks += 1

    def lufs(self):
        return -0.691 + 10.0 * np.log10(self.ms + 1e-12)
