"""Streaming phase-vocoder time-stretcher (constant pitch) for the DJ decks.

WHY (2026-07-12): the WSOLA stretcher (lib/dj/stretch.py) reuses whole
waveform chunks, so it keeps drum transients crisp but its frame-to-frame
overlap-add of not-quite-matching windows leaves a ~43 Hz "warble/chopping"
roughness whenever a deck is tempo-stretched - most audible in the long
post-transition glide. This engine works in the FREQUENCY domain instead:

STFT (N=2048, 75% overlap) with per-bin PHASE CONTINUATION: the stretched
output stays phase-coherent, which removes ~25% of WSOLA's steady-state
tonal warble (measured, clean fixed-rate A/B on a real track).

STATUS - OPT-IN, NOT the default (2026-07-12). Findings from a full gate sweep:
  - Straight phase continuation (default here) passes the stretch gate
    (tempo/onsets/confidence/bypass/block, CPU ~3.4%) and mix + brain gates,
    and beats WSOLA on steady warble.
  - BUT at SEAMS (cut_at_drop/echo_out/long_fade) it dips lower than WSOLA
    (real quality-gate dead-air fails, min/median RMS 0.10-0.12 vs WSOLA's
    clean pass) - a frequency-domain reconstruction is simply less graceful
    than WSOLA's time-domain chunk reuse at cuts/transients.
  - Loose phase locking (DJ_PV_LOCK=1, Laroche neighbour coupling) fixes
    tonal phasiness but makes the SEAM dead-air WORSE (coupling smears energy
    at discontinuities: quality fails 11 -> 15) and pushes CPU to ~5%. Hard
    transient phase-reset clicks at the OLA seam. So neither cleanly wins.
Making this default-quality needs research-grade seam/transient handling
(PGHI-style phase reconstruction); until then WSOLA stays the default and
this is $DJ_STRETCH_ENGINE=pv for ear A/B. Env knobs: DJ_PV_LOCK / DJ_PV_TRANS
/ DJ_PV_LOCK_ITERS / DJ_PV_ALWAYS.

Same pull interface as WSOLAStretcher (drop-in): fetch(pos, n) -> (n, ch),
read any number of frames, produce in Hs quanta. rate/seek/source_pos/
phase_trim/no_bypass behave identically (submix PLL unchanged). Per-sample OLA
normalization by the ACTUAL summed window makes output full-amplitude from
frame 1 (no seek/mode-switch fade-in). Bit-exact bypass at rate 1.
"""
import os

import numpy as np

# Loose phase locking ON by default (restores mainlobe coherence -> no
# amplitude modulation). Hard transient reset OFF (clicks at the OLA seam).
_LOCK = os.environ.get("DJ_PV_LOCK", "0") != "0"
_TRANS = os.environ.get("DJ_PV_TRANS", "0") != "0"    # skip lock on onsets
_LOCK_ITERS = int(os.environ.get("DJ_PV_LOCK_ITERS", "1"))  # neighbour passes
_ALWAYS = os.environ.get("DJ_PV_ALWAYS", "0") != "0"  # diag: never bypass

N = 2048
HS = 512                         # synthesis hop (75% overlap - PV needs it)
BYPASS_EPS = 0.001
XFADE = 128                      # frames, for bypass<->pv switches
TWO_PI = 2.0 * np.pi
# Transient gate: a frame is an onset when its spectral flux both spikes vs
# the running average AND is a real fraction of the frame energy.
TRANS_K = 1.8
TRANS_RATIO = 0.22
PHASE_TRIM_CLIP = HS / 8.0       # frames/hop of PLL correction (== WSOLA's 64)


def _princarg(x):
    """Wrap radians to (-pi, pi]."""
    return x - TWO_PI * np.round(x / TWO_PI)


class PhaseVocoderStretcher:
    def __init__(self, fetch, channels=2):
        self.fetch = fetch
        self.channels = channels
        self.rate = 1.0
        self.no_bypass = False
        self.phase_trim = 0.0
        self._pos = 0.0                  # nominal (rate-integrated) source frame
        self._win = np.hanning(N).astype(np.float64)[:, None]
        self._omega = (TWO_PI * np.arange(N // 2 + 1) / N)[:, None]  # rad/sample
        # OLA gain of the analysis*synthesis window (Hann^2) at 75% overlap.
        self._w2 = (self._win[:, 0] ** 2)     # analysis*synthesis window
        self._reset_state()
        self._bypassed = True

    # -- public ------------------------------------------------------------
    def seek(self, source_frame):
        self._pos = float(source_frame)
        self.phase_trim = 0.0
        self._reset_state()
        self._fifo = self._fifo[:0]

    @property
    def source_pos(self):
        return self._pos

    def read(self, n_out):
        while len(self._fifo) < n_out:
            self._produce()
        out, self._fifo = self._fifo[:n_out], self._fifo[n_out:]
        return out

    # -- internals ---------------------------------------------------------
    def _reset_state(self):
        self._last_frame = None
        self._prev_phase = None
        self._prev_mag = None
        self._synth_phase = None
        self._flux_ema = 0.0
        self._ola = np.zeros((N, self.channels), dtype=np.float64)
        # Per-sample window-overlap accumulator: dividing the OLA by the ACTUAL
        # summed window (not a steady-state constant) makes the output full-
        # amplitude from the very first frame - otherwise every seek/mode-
        # switch starts with a quiet ramp-up that reads as dead air at a seam.
        self._wacc = np.zeros(N, dtype=np.float64)
        if not hasattr(self, "_fifo"):
            self._fifo = np.zeros((0, self.channels), dtype=np.float32)

    def _produce(self):
        want_bypass = (abs(self.rate - 1.0) < BYPASS_EPS
                       and not self.no_bypass and not _ALWAYS)
        if want_bypass != self._bypassed:
            old = self._render_bypass(HS) if self._bypassed \
                else self._render_pv()
            self._bypassed = want_bypass
            if want_bypass and self._last_frame is not None:
                self._pos = float(self._last_frame + HS)
            self._reset_state()          # fresh accumulators for the new mode
            new = self._render_bypass(HS) if want_bypass else self._render_pv()
            f = np.linspace(0.0, 1.0, min(XFADE, HS))[:, None]
            blk = new.copy()
            blk[:len(f)] = old[:len(f)] * (1 - f) + new[:len(f)] * f
            self._push(blk)
            return
        self._push(self._render_bypass(HS) if want_bypass
                   else self._render_pv())

    def _push(self, blk):
        self._fifo = np.concatenate(
            [self._fifo, blk.astype(np.float32)], axis=0)

    def _render_bypass(self, n):
        p = int(round(self._pos))
        blk = self.fetch(p, n).astype(np.float64)
        self._pos = p + n
        return blk

    def _phase_lock(self, synth, mag):
        """Loose phase locking (Laroche & Dolson): the locked output phase of
        each bin is the angle of the sum of that bin's complex value and its
        two neighbours' (using the free-running synthesis phases). This couples
        the bins across a sinusoid's mainlobe so they stay coherent - the
        overlapping frames then reconstruct at constant amplitude instead of
        beating against each other. No peak finding, so it never jumps as
        peaks drift. One or more relaxation passes."""
        z = mag * np.exp(1j * synth)
        out = synth
        for _ in range(max(_LOCK_ITERS, 1)):
            zc = z.copy()
            zc[1:] += z[:-1]
            zc[:-1] += z[1:]
            out = np.angle(zc)
            z = mag * np.exp(1j * out)
        return out

    def _render_pv(self):
        Ra = self.rate * HS
        frame = int(round(self._pos))
        block = self.fetch(frame, N).astype(np.float64)
        X = np.fft.rfft(block * self._win, axis=0)
        mag = np.abs(X)
        phase = np.angle(X)
        if self._last_frame is None:                 # first frame: seed
            self._synth_phase = phase.copy()
            out_phase = self._synth_phase
        else:
            actual_Ra = max(frame - self._last_frame, 1)
            expected = self._omega * actual_Ra
            dphi = _princarg(phase - self._prev_phase - expected)
            true_freq = self._omega + dphi / actual_Ra
            mono = mag.sum(axis=1)
            flux = float(np.sum(np.maximum(mono - self._prev_mag.sum(axis=1),
                                           0.0)))
            energy = float(np.sum(mono)) + 1e-9
            transient = (_TRANS and flux > TRANS_K * self._flux_ema
                         and flux / energy > TRANS_RATIO)
            self._flux_ema = 0.9 * self._flux_ema + 0.1 * flux
            # ACCUMULATOR is always the free-running PV phase; locking is an
            # OUTPUT-only transform (feeding it back corrupts the estimate).
            self._synth_phase = self._synth_phase + true_freq * HS
            # SKIP locking on a transient frame: a kick/snare is broadband, so
            # coupling its bins smears the attack (drops the onset count at big
            # ratios). Leaving that one frame free-running keeps it sharp - and
            # unlike a hard phase RESET it introduces no OLA-seam click.
            out_phase = (self._synth_phase if (transient or not _LOCK)
                         else self._phase_lock(self._synth_phase, mag))
        y = np.fft.irfft(mag * np.exp(1j * out_phase), n=N, axis=0) \
            * self._win
        self._ola[:N] += y
        self._wacc[:N] += self._w2
        out = (self._ola[:HS]
               / np.maximum(self._wacc[:HS], 1e-6)[:, None]).copy()
        self._ola[:N - HS] = self._ola[HS:N]
        self._ola[N - HS:N] = 0.0
        self._wacc[:N - HS] = self._wacc[HS:N]
        self._wacc[N - HS:N] = 0.0
        self._prev_phase = phase
        self._prev_mag = mag
        self._last_frame = frame
        step = float(np.clip(self.phase_trim, -PHASE_TRIM_CLIP,
                             PHASE_TRIM_CLIP))
        self.phase_trim -= step
        self._pos += Ra + step
        return out
