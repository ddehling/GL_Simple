"""DJ FX primitives: resonant sweep filter, tempo-synced echo, synthesized
one-shots (riser / impact).

SweepFilter and EchoDelay are STREAMING (per-block, state carried across
calls, coefficients ramped block-wise) and cheap enough for the audio
callback: everything is scipy sosfilt / numpy slices, no per-sample Python.
The one-shots are synthesized OFFLINE on the control thread at plan time
and shipped to the submix as ready buffers.
"""
import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

RATE = 44100


def _rbj_sos(kind, fc, q):
    """One RBJ-cookbook biquad as an sos row. kind: 'lp' | 'hp'."""
    fc = float(np.clip(fc, 20.0, 18000.0))
    w0 = 2.0 * np.pi * fc / RATE
    alpha = np.sin(w0) / (2.0 * max(q, 0.1))
    cw = np.cos(w0)
    if kind == "lp":
        b0 = (1 - cw) / 2
        b1 = 1 - cw
        b2 = (1 - cw) / 2
    else:
        b0 = (1 + cw) / 2
        b1 = -(1 + cw)
        b2 = (1 + cw) / 2
    a0 = 1 + alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0,
                      1.0, (-2 * cw) / a0, (1 - alpha) / a0]])


class SweepFilter:
    """Ramped resonant LP/HP per deck - THE house-DJ transition tool.

    mode 'off' is a true bypass (no state, no cost). Cutoff ramps linearly
    in LOG-frequency (how sweeps are heard); coefficients update once per
    block with filter state carried, which is inaudible at ~23 ms blocks."""

    def __init__(self, channels=2):
        self.channels = channels
        self.mode = "off"
        self.q = 1.2
        self._fc = 1000.0
        self._target = 1000.0
        self._ramp_left_s = 0.0
        self._zi = None

    def set(self, mode=None, cutoff_hz=None, ramp_s=0.0, q=None):
        if mode is not None and mode != self.mode:
            self.mode = mode
            self._zi = None                      # fresh state on mode flip
            if mode != "off" and cutoff_hz is not None and ramp_s > 0.0:
                self._fc = float(cutoff_hz) if self._ramp_left_s <= 0 \
                    else self._fc
        if q is not None:
            self.q = float(np.clip(q, 0.5, 6.0))
        if cutoff_hz is not None:
            self._target = float(np.clip(cutoff_hz, 20.0, 18000.0))
            self._ramp_left_s = max(float(ramp_s), 0.0)
            if ramp_s <= 0.0:
                self._fc = self._target

    def process(self, block):
        if self.mode == "off":
            return block
        n = len(block)
        dt = n / RATE
        if self._ramp_left_s > 0.0:
            frac = min(dt / self._ramp_left_s, 1.0)
            lf = np.log(self._fc) + (np.log(self._target)
                                     - np.log(self._fc)) * frac
            self._fc = float(np.exp(lf))
            self._ramp_left_s -= dt
        else:
            self._fc = self._target
        sos = _rbj_sos(self.mode, self._fc, self.q)
        if self._zi is None:
            self._zi = np.stack([sosfilt_zi(sos) * 0.0
                                 for _ in range(self.channels)])
        out = np.empty_like(block)
        for c in range(self.channels):
            out[:, c], self._zi[c] = sosfilt(sos, block[:, c],
                                             zi=self._zi[c])
        return out


class EchoDelay:
    """Tempo-synced feedback delay for echo-out exits.

    Engage with a beat-length delay and healthy feedback, then cut the
    deck's gain: the echo sits AFTER the gain stage, so the captured tail
    keeps ringing and decays over the incoming track. Block-exact as long
    as delay >= block size (min delay clamped to 1024 frames)."""

    MAX_S = 2.5

    def __init__(self, channels=2):
        self.channels = channels
        self.active = False
        self.delay = RATE // 2
        self.feedback = 0.55
        self.wet = 0.7
        self._ring = np.zeros((int(self.MAX_S * RATE), channels),
                              dtype=np.float32)
        self._pos = 0

    def set(self, active=None, delay_s=None, feedback=None, wet=None):
        if delay_s is not None:
            self.delay = int(np.clip(delay_s * RATE, 1024,
                                     len(self._ring) - 1))
        if feedback is not None:
            self.feedback = float(np.clip(feedback, 0.0, 0.85))
        if wet is not None:
            self.wet = float(np.clip(wet, 0.0, 1.0))
        if active is not None:
            if active and not self.active:
                self._ring[:] = 0.0              # clean capture
            self.active = bool(active)

    def process(self, block):
        if not self.active:
            return block
        n = len(block)
        ring = self._ring
        idx_w = (self._pos + np.arange(n)) % len(ring)
        idx_r = (idx_w - self.delay) % len(ring)
        delayed = ring[idx_r]
        ring[idx_w] = block + delayed * self.feedback
        self._pos = (self._pos + n) % len(ring)
        return block + delayed * self.wet

    @property
    def ringing(self):
        """True while a meaningful tail is still sounding."""
        return self.active and float(np.abs(self._ring).max()) > 1e-3


# --------------------------------------------------------------------------
# One-shot synthesis (offline, control thread)
# --------------------------------------------------------------------------

def make_riser(dur_s=8.0, gain=0.5, seed=0):
    """White-noise riser: band-passed noise whose center sweeps up with a
    squared amplitude swell - the standard pre-drop tension tool."""
    n = int(dur_s * RATE)
    rng = np.random.RandomState(seed)
    noise = rng.randn(n).astype(np.float64)
    out = np.zeros(n)
    step = 2048
    filt = SweepFilter(channels=1)
    filt.set(mode="lp", cutoff_hz=400.0, q=2.2)
    for i in range(0, n, step):
        frac = i / n
        filt.set(cutoff_hz=400.0 * (20.0 ** frac))     # 400 -> 8 kHz
        seg = noise[i:i + step][:, None]
        out[i:i + len(seg)] = filt.process(seg)[:, 0]
    amp = (np.linspace(0.0, 1.0, n) ** 2) * gain
    # quick release so the riser never smears past its landing downbeat
    rel = min(int(0.02 * RATE), n)
    amp[-rel:] *= np.linspace(1.0, 0.0, rel)
    x = (out * amp).astype(np.float32)
    return np.stack([x, x], axis=1)


def make_impact(gain=0.7, seed=1):
    """Impact hit: pitch-dropping sub boom + short noise burst + LP tail."""
    dur = 1.4
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    f = 110.0 * np.exp(-t / 0.06) + 42.0
    phase = 2 * np.pi * np.cumsum(f) / RATE
    boom = np.sin(phase) * np.exp(-t / 0.35)
    rng = np.random.RandomState(seed)
    burst = rng.randn(n) * np.exp(-t / 0.05) * 0.6
    filt = SweepFilter(channels=1)
    filt.set(mode="lp", cutoff_hz=3000.0, q=0.9)
    mix = filt.process((boom + burst)[:, None] * gain)[:, 0]
    x = np.clip(mix, -1.0, 1.0).astype(np.float32)
    return np.stack([x, x], axis=1)
