"""Per-sample DSP kernels (numba, nogil). Everything with a recursion -
filters, envelopes, delay lines - lives here; block-level maths stays in
numpy in the voices. If numba is missing the kernels still run (pure
Python, ~100x slower) so imports never fail; the gate checks speed."""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                       # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f


@njit(cache=True, nogil=True)
def polyblep_saw(n, phase0, freq, rate):
    """Anti-aliased sawtooth. freq: per-sample Hz array. Returns (out, phase)."""
    out = np.empty(n, dtype=np.float32)
    ph = phase0
    for i in range(n):
        dt = freq[i] / rate
        v = 2.0 * ph - 1.0
        # polyBLEP correction at the discontinuity
        if ph < dt:
            t = ph / dt
            v -= (t + t - t * t - 1.0)
        elif ph > 1.0 - dt:
            t = (ph - 1.0) / dt
            v -= (t * t + t + t + 1.0)
        out[i] = v
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
    return out, ph


@njit(cache=True, nogil=True)
def polyblep_pulse(n, phase0, freq, rate, width):
    out = np.empty(n, dtype=np.float32)
    ph = phase0
    for i in range(n):
        dt = freq[i] / rate
        v = 1.0 if ph < width else -1.0
        if ph < dt:
            t = ph / dt
            v += (t + t - t * t - 1.0)
        elif ph > 1.0 - dt:
            t = (ph - 1.0) / dt
            v += (t * t + t + t + 1.0)
        p2 = ph - width
        if p2 < 0.0:
            p2 += 1.0
        if p2 < dt:
            t = p2 / dt
            v -= (t + t - t * t - 1.0)
        elif p2 > 1.0 - dt:
            t = (p2 - 1.0) / dt
            v -= (t * t + t + t + 1.0)
        out[i] = v
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
    return out, ph


@njit(cache=True, nogil=True)
def svf(x, cutoff, res, rate, mode):
    """Cytomic/Chamberlin-style state-variable filter with per-sample
    cutoff (Hz array). res 0..1. mode 0 = LP, 1 = BP, 2 = HP. 2x
    oversampled Chamberlin for stability up to ~rate/4."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    lp = 0.0
    bp = 0.0
    q = 2.0 - 1.9 * min(max(res, 0.0), 1.0)     # damping
    for i in range(n):
        fc = cutoff[i]
        if fc > rate * 0.22:
            fc = rate * 0.22
        f = 2.0 * math.sin(math.pi * fc / (rate * 2.0))
        xi = x[i]
        for _ in range(2):
            hp = xi - lp - q * bp
            bp += f * hp
            lp += f * bp
        if mode == 0:
            out[i] = lp
        elif mode == 1:
            out[i] = bp
        else:
            out[i] = hp
    return out


@njit(cache=True, nogil=True)
def adsr(n, gate, a, d, s, r, rate):
    """Exponential-ish ADSR: gate samples of on-time, then release. Returns
    n samples; a/d/r in seconds, s 0..1."""
    out = np.empty(n, dtype=np.float32)
    ca = math.exp(-1.0 / max(a * rate, 1.0))
    cd = math.exp(-1.0 / max(d * rate, 1.0))
    cr = math.exp(-1.0 / max(r * rate, 1.0))
    v = 0.0
    stage = 0
    for i in range(n):
        if i >= gate:
            v = v * cr
        elif stage == 0:
            v = 1.0 + (v - 1.0) * ca
            if v > 0.995:
                v = 1.0
                stage = 1
        else:
            v = s + (v - s) * cd
        out[i] = v
    return out


@njit(cache=True, nogil=True)
def exp_decay(n, tau, rate):
    out = np.empty(n, dtype=np.float32)
    c = math.exp(-1.0 / max(tau * rate, 1.0))
    v = 1.0
    for i in range(n):
        out[i] = v
        v *= c
    return out


@njit(cache=True, nogil=True)
def comb(x, buf, idx, fb, damp, lp_state):
    """Lowpass-feedback comb (Freeverb). buf is the delay line; returns
    (out, new_idx, new_lp_state)."""
    n = x.shape[0]
    L = buf.shape[0]
    out = np.empty(n, dtype=np.float32)
    lp = lp_state
    for i in range(n):
        y = buf[idx]
        lp = y * (1.0 - damp) + lp * damp
        buf[idx] = x[i] + lp * fb
        out[i] = y
        idx += 1
        if idx >= L:
            idx = 0
    return out, idx, lp


@njit(cache=True, nogil=True)
def allpass(x, buf, idx, g):
    n = x.shape[0]
    L = buf.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        b = buf[idx]
        v = x[i] + (-g) * b
        buf[idx] = v
        out[i] = b + g * v
        idx += 1
        if idx >= L:
            idx = 0
    return out, idx


@njit(cache=True, nogil=True)
def onepole_lp(x, coef, state):
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    s = state
    for i in range(n):
        s = s + coef * (x[i] - s)
        out[i] = s
    return out, s
