"""DJ FX primitives: resonant sweep filter, tempo-synced echo, synthesized
one-shots (riser / impact).

SweepFilter and EchoDelay are STREAMING (per-block, state carried across
calls, coefficients ramped block-wise) and cheap enough for the audio
callback: everything is scipy sosfilt / numpy slices, no per-sample Python.
The one-shots are synthesized OFFLINE on the control thread at plan time
and shipped to the submix as ready buffers.
"""
import numpy as np
from scipy.signal import sosfilt

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
            self._zi = np.zeros((1, 2, self.channels))
        # Both channels in ONE sosfilt call (axis=0): scipy's per-call
        # Python overhead dominated the 1-section filter at 256-frame
        # sub-blocks (perf audit 2026-07-31).
        out, self._zi = sosfilt(sos, block, axis=0, zi=self._zi)
        return out.astype(block.dtype)


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

def make_riser(dur_s=8.0, gain=0.5, seed=0, beat_s=None):
    """Pre-drop riser, three layers wide instead of one mono hiss:

      - STEREO band-swept noise (independent L/R seeds - decorrelated
        noise is what makes a riser sound big instead of like a leak),
      - a detuned tone stack rising one octave underneath (the pitch
        climb the ear actually tracks as 'something is coming'),
      - when beat_s is given, an accelerating sidechain-style pump that
        deepens and speeds up toward the landing - tension you can feel
        in the rhythm, not just the spectrum.

    Squared amplitude swell as before; quick release so it never smears
    past its landing downbeat."""
    n = int(dur_s * RATE)
    t = np.arange(n) / RATE
    frac_t = t / max(dur_s, 1e-6)
    rng = np.random.RandomState(seed)
    ch = []
    for _c in range(2):
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
        ch.append(out)
    # Tone stack: three detuned partials climbing an octave. Quiet next
    # to the noise, but pitch is what sells the climb.
    tone = np.zeros(n)
    for det in (-0.008, 0.0, 0.009):
        f = 110.0 * (1.0 + det) * (2.0 ** frac_t)
        tone += np.sin(2 * np.pi * np.cumsum(f) / RATE)
    tone *= 0.22
    amp = (frac_t ** 2) * gain
    if beat_s and beat_s > 0.05:
        f_pump = (1.0 / beat_s) * (1.0 + frac_t)       # 1x -> 2x per beat
        ph = 2 * np.pi * np.cumsum(f_pump) / RATE
        amp *= 1.0 - 0.4 * frac_t * (0.5 + 0.5 * np.cos(ph))
    # quick release so the riser never smears past its landing downbeat
    rel = min(int(0.02 * RATE), n)
    amp[-rel:] *= np.linspace(1.0, 0.0, rel)
    return np.stack([((ch[0] + tone) * amp), ((ch[1] + tone) * amp)],
                    axis=1).astype(np.float32)


def at_peak(buf, peak):
    """Rescale a one-shot to an exact PEAK amplitude.

    The `gain` args below are pre-filter amplitudes, not peaks: filtered
    noise has a crest factor near 5, so make_riser(gain=0.16) actually
    peaks at 0.67 and gain=0.26 clips past full scale. Anything layered
    over a live mix needs its headroom stated in peaks, or the master
    limiter decides the balance for you."""
    m = float(np.abs(buf).max())
    return buf * (float(peak) / m) if m > 1e-9 else buf


def at_tail(buf, rms=0.13, peak=0.6):
    """Rescale a SWELLING one-shot by the RMS of its final quarter, with a
    soft cap on the peak.

    at_peak is the wrong tool for a riser: a squared swell spends its whole
    peak budget on its last instants (crest factor ~11), so peaking a riser
    at 0.45 leaves its body at -28 dBFS RMS - 13 dB under a club master,
    i.e. the buried-build failure all over again, just via a different
    knob. Loudness against a mix is an RMS fact, so state it in RMS where
    it matters (the tail that rides into the landing) and tame the final
    spike with a soft clip instead of letting it set the level of
    everything before it."""
    tail = buf[-max(1, len(buf) // 4):]
    r = float(np.sqrt(np.mean(tail.astype(np.float64) ** 2)))
    if r < 1e-9:
        return buf
    out = buf * (float(rms) / r)
    m = float(np.abs(out).max())
    if m > peak:
        out = np.tanh(out / peak) * peak
    return out.astype(np.float32)


def make_roll(beat_s, beats=4, gain=0.3, seed=2):
    """Accelerating snare/clap ROLL into a landing: hits subdivide 8ths ->
    16ths -> 32nds over `beats` beats with a swelling envelope, the last
    hit one subdivision BEFORE the end so the landing itself is free for
    the impact.

    `beat_s` is the OUTPUT beat (source period / playback rate) - the roll
    has to lock to what the room hears, not to the file's natural tempo.
    A riser alone reads as a synth swell; it's the rhythmic acceleration
    that tells a crowd exactly which downbeat to expect."""
    beat_s = max(float(beat_s), 0.08)
    total = beat_s * max(int(beats), 1)
    hn = int(0.09 * RATE)                        # one hit, decay included
    n = int(total * RATE) + hn
    rng = np.random.RandomState(seed)
    t = np.arange(hn) / RATE
    filt = SweepFilter(channels=1)
    filt.set(mode="hp", cutoff_hz=900.0, q=0.9)  # crack, no sub content
    crack = filt.process((rng.randn(hn) * np.exp(-t / 0.022))[:, None])[:, 0]
    # A real snare is crack + BODY: a pitch-thunking tone burst around
    # 195 Hz. Noise alone read as a typewriter, not a drum.
    body = np.sin(2 * np.pi * 195.0 * t + 3.0 * np.exp(-t / 0.008)) \
        * np.exp(-t / 0.05)
    hit = 0.75 * crack + 0.6 * body
    hit /= max(float(np.abs(hit).max()), 1e-9)
    out = np.zeros((n, 2))
    at, k = 0.0, 0
    while at < total - 1e-6:
        frac = at / total
        div = 2.0 if frac < 0.5 else (4.0 if frac < 0.8 else 8.0)
        i0 = int(at * RATE)
        seg = hit[:max(0, min(hn, n - i0))]
        g = gain * (0.35 + 0.65 * frac ** 2)
        # Alternate hits lean L/R, narrowing as the roll tightens -
        # motion that collapses to center right before the landing.
        pan = 0.22 * (1.0 - frac) * (1 if k % 2 else -1)
        out[i0:i0 + len(seg), 0] += seg * g * (1.0 - pan)
        out[i0:i0 + len(seg), 1] += seg * g * (1.0 + pan)
        at += beat_s / div
        k += 1
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def make_impact(gain=0.7, seed=1):
    """Landing impact: saturated pitch-dropping sub boom + noise burst +
    a STEREO crash wash that hangs over the first bar of the drop.

    The old version was a 0.35s clean sine thud - it read as 'PHHHT'
    (user) next to a full-scale master. Weight comes from saturation
    (tanh fattens the boom's harmonics), and size comes from the
    decorrelated crash tail: a real drop is a hit AND a wash."""
    dur = 2.2
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    f = 120.0 * np.exp(-t / 0.07) + 40.0
    phase = 2 * np.pi * np.cumsum(f) / RATE
    boom = np.tanh(1.8 * np.sin(phase) * np.exp(-t / 0.5))
    rng = np.random.RandomState(seed)
    burst = rng.randn(n) * np.exp(-t / 0.04) * 0.7
    lp = SweepFilter(channels=1)
    lp.set(mode="lp", cutoff_hz=3500.0, q=0.9)
    center = lp.process((boom + burst)[:, None])[:, 0]
    out = np.zeros((n, 2))
    for c in range(2):
        crash = rng.randn(n) * np.exp(-t / 0.55) * 0.30
        hp = SweepFilter(channels=1)
        hp.set(mode="hp", cutoff_hz=2500.0, q=0.7)
        out[:, c] = center + hp.process(crash[:, None])[:, 0]
    return np.clip(out * gain, -1.0, 1.0).astype(np.float32)
