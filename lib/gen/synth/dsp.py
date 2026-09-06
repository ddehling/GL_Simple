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


@njit(cache=True, nogil=True)
def peak_limiter(x, ceiling, att, rel, state):
    """Stereo-linked peak limiter without lookahead: the gain computer
    follows |x| with an instantaneous-ish attack (att coef) and a slow
    release (rel coef). Returns (out, new_state). state = current gain."""
    n = x.shape[0]
    out = np.empty_like(x)
    g = state
    for i in range(n):
        pk = abs(x[i, 0])
        r = abs(x[i, 1])
        if r > pk:
            pk = r
        target = 1.0
        if pk * 1.0 > ceiling:
            target = ceiling / pk
        if target < g:
            g = target + (g - target) * att
        else:
            g = target + (g - target) * rel
        out[i, 0] = x[i, 0] * g
        out[i, 1] = x[i, 1] * g
    return out, g


@njit(cache=True, nogil=True)
def smooth_noise(n, rate_hz, sr, seed_vals):
    """Slow random modulation in [-1, 1]: a random walk of breakpoints
    every 1/rate_hz seconds, cosine-interpolated. seed_vals: uniform
    [-1,1] samples, one per breakpoint (len >= n*rate_hz/sr + 2)."""
    out = np.empty(n, dtype=np.float32)
    seg = max(int(sr / max(rate_hz, 1e-3)), 1)
    for i in range(n):
        k = i // seg
        f = (i - k * seg) / seg
        f = 0.5 - 0.5 * math.cos(math.pi * f)
        out[i] = seed_vals[k] * (1.0 - f) + seed_vals[k + 1] * f
    return out


@njit(cache=True, nogil=True)
def svf_tpt(x, cutoff, res, rate, mode, drive):
    """Zavalishin/Cytomic trapezoidal-integrator SVF (zero-delay feedback):
    stable to ~0.45 * rate, resonance that actually sings, a tanh drive at
    the input so pushing it adds harmonics instead of numbers. cutoff:
    per-sample Hz. res 0..1. mode 0 LP, 1 BP, 2 HP. drive >= 1."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    ic1 = 0.0
    ic2 = 0.0
    r = max(min(res, 1.0), 0.0)
    k = 2.0 - 1.95 * r                     # damping: 2 (none) .. 0.05 (screaming)
    lim = rate * 0.45
    inv = 1.0 / drive
    for i in range(n):
        fc = cutoff[i]
        if fc > lim:
            fc = lim
        if fc < 5.0:
            fc = 5.0
        g = math.tan(math.pi * fc / rate)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2
        v0 = math.tanh(x[i] * drive) * inv if drive > 1.0 else x[i]
        v3 = v0 - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if mode == 0:
            out[i] = v2
        elif mode == 1:
            out[i] = v1
        else:
            out[i] = v0 - k * v1 - v2
    return out


@njit(cache=True, nogil=True)
def compressor(x, thresh, ratio, att, rel, makeup, state):
    """Stereo-linked feed-forward peak compressor. thresh linear, ratio
    >= 1, att/rel one-pole coefs on the detector, makeup linear gain.
    state = detector envelope. Returns (out, new_state)."""
    n = x.shape[0]
    out = np.empty_like(x)
    env = state
    for i in range(n):
        pk = abs(x[i, 0])
        r = abs(x[i, 1])
        if r > pk:
            pk = r
        if pk > env:
            env = pk + (env - pk) * att
        else:
            env = pk + (env - pk) * rel
        g = 1.0
        if env > thresh:
            g = (thresh + (env - thresh) / ratio) / env
        g *= makeup
        out[i, 0] = x[i, 0] * g
        out[i, 1] = x[i, 1] * g
    return out, env


@njit(cache=True, nogil=True)
def biquad(x, b0, b1, b2, a1, a2, z1, z2):
    """Transposed direct form II biquad on a mono block. Returns
    (out, z1, z2)."""
    n = x.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        xi = x[i]
        y = b0 * xi + z1
        z1 = b1 * xi - a1 * y + z2
        z2 = b2 * xi - a2 * y
        out[i] = y
    return out, z1, z2


@njit(cache=True, nogil=True)
def fm_sine(n, phase0, freq, ratio, index_env, rate):
    """2-operator FM: carrier at freq (per-sample Hz), modulator at
    freq*ratio, modulation index per sample. Returns the carrier."""
    out = np.empty(n, dtype=np.float32)
    pc = phase0
    pm = phase0 * ratio
    twopi = 2.0 * math.pi
    for i in range(n):
        out[i] = math.sin(twopi * pc + index_env[i] * math.sin(twopi * pm))
        d = freq[i] / rate
        pc += d
        pm += d * ratio
        if pc >= 1.0:
            pc -= 1.0
        if pm >= 1.0:
            pm -= math.floor(pm)
    return out



@njit(cache=True, nogil=True)
def fdn_reverb(x, lines, lens, idx, lp_state, fb, damp, mod_depth, mod_rates, phase, rate, out):
    """8-line feedback delay network with a Hadamard mixing matrix,
    one-pole damping per line and slow sinusoidal modulation of each
    line's read position (kills the metallic ring of static combs).
    x: (n,2) input (pre-diffused), lines: (8, Lmax) buffers, lens: (8,)
    nominal lengths, idx: (8,) write positions, lp_state: (8,) damping
    state, fb: loop gain, damp: 0..1, mod_depth in samples, mod_rates
    (8,) Hz, phase: running LFO phase (seconds). Returns new phase; the
    wet signal is written into out (n,2)."""
    n = x.shape[0]
    N = 8
    h = 1.0 / math.sqrt(N)
    y = np.empty(N, dtype=np.float32)
    v = np.empty(N, dtype=np.float32)
    for i in range(n):
        t = phase + i / rate
        mono = 0.5 * (x[i, 0] + x[i, 1])
        for k in range(N):
            L = lens[k]
            m = mod_depth * (0.5 + 0.5 * math.sin(2.0 * math.pi * mod_rates[k] * t + k))
            rp = idx[k] - L + m
            while rp < 0.0:
                rp += lines.shape[1]
            r0 = int(rp)
            fr = rp - r0
            r1 = r0 + 1
            if r1 >= lines.shape[1]:
                r1 = 0
            s = lines[k, r0] * (1.0 - fr) + lines[k, r1] * fr
            lp_state[k] = s * (1.0 - damp) + lp_state[k] * damp
            y[k] = lp_state[k]
        # Hadamard 8x8 (Sylvester construction), scaled
        for k in range(N):
            acc = 0.0
            for j in range(N):
                # entry sign = parity of popcount(k & j)
                b = k & j
                c = 0
                while b:
                    c += b & 1
                    b >>= 1
                acc += y[j] if (c & 1) == 0 else -y[j]
            v[k] = acc * h
        outl = 0.0
        outr = 0.0
        for k in range(N):
            lines[k, idx[k]] = mono + fb * v[k]
            idx[k] += 1
            if idx[k] >= lines.shape[1]:
                idx[k] = 0
            if k % 2 == 0:
                outl += y[k]
            else:
                outr += y[k]
        out[i, 0] = outl * 0.35
        out[i, 1] = outr * 0.35
    return phase + n / rate


@njit(cache=True, nogil=True)
def chorus(x, buf, idx, base, depth, rate_hz, phase, sr, out):
    """Stereo chorus: one modulated delay per channel, LFOs a quarter
    turn apart. base/depth in samples. Writes wet into out, returns
    (idx, phase)."""
    n = x.shape[0]
    L = buf.shape[0]
    for i in range(n):
        t = phase + i / sr
        for ch in range(2):
            buf[idx, ch] = x[i, ch]
            lfo = math.sin(2.0 * math.pi * rate_hz * t + (0.0 if ch == 0 else 1.5707963))
            d = base + depth * (0.5 + 0.5 * lfo)
            rp = idx - d
            while rp < 0.0:
                rp += L
            r0 = int(rp)
            fr = rp - r0
            r1 = r0 + 1
            if r1 >= L:
                r1 = 0
            out[i, ch] = buf[r0, ch] * (1.0 - fr) + buf[r1, ch] * fr
        idx += 1
        if idx >= L:
            idx = 0
    return idx, phase + n / sr


@njit(cache=True, nogil=True)
def karplus(n, period, decay, brightness, excite):
    """Karplus-Strong plucked string. period: samples (fractional, via a
    first-order allpass); decay: loop gain 0..1; brightness: 0..1 blend
    between the averaging low-pass (dark) and the raw feedback (bright);
    excite: excitation burst (len <= n)."""
    out = np.empty(n, dtype=np.float32)
    L = int(period)
    if L < 2:
        L = 2
    frac = period - L
    if frac < 0.0:
        frac = 0.0
    buf = np.zeros(L + 1, dtype=np.float32)
    ap_x = 0.0
    ap_y = 0.0
    prev = 0.0
    w = 0
    ne = excite.shape[0]
    # allpass coefficient for the fractional delay
    c = (1.0 - frac) / (1.0 + frac)
    for i in range(n):
        r = w - L
        if r < 0:
            r += L + 1
        s = buf[r]
        # fractional delay (allpass)
        ap_y = c * (s - ap_y) + ap_x
        ap_x = s
        d = ap_y
        # damping: average with the previous sample (dark) vs raw (bright)
        fbk = (0.5 * (d + prev)) * (1.0 - brightness) + d * brightness
        prev = d
        e = excite[i] if i < ne else 0.0
        v = e + fbk * decay
        buf[w] = v
        w += 1
        if w >= L + 1:
            w = 0
        out[i] = v
    return out


@njit(cache=True, nogil=True)
def gain_smooth(target, g0, rel):
    """Limiter gain smoothing: instant when the target is lower (attack
    handled by lookahead), one-pole release when it rises."""
    n = target.shape[0]
    out = np.empty(n, dtype=np.float32)
    g = g0
    for i in range(n):
        t = target[i]
        if t < g:
            g = t
        else:
            g = t + (g - t) * rel
        out[i] = g
    return out, g
