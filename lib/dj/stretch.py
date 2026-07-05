"""Streaming WSOLA time-stretcher (constant pitch) for the DJ decks.

Waveform-Similarity Overlap-Add: each synthesis frame overlap-adds a source
window chosen near the nominal position but shifted (+/- search) to best
match the natural continuation of the previous window - transients stay
crisp because whole waveform chunks are reused, which is why WSOLA beats a
phase vocoder on percussive dance music at DJ-sized ratios.

    window N=2048, synthesis hop Hs=1024 (Hann, 50% OLA), search +/-512
    rate clamp 0.90..1.10 upstream (brain prefers <=+/-5%)
    bit-exact bypass when |rate-1| < 0.001, crossfaded mode switches

Pull model: the deck hands us fetch(pos, n) -> (n, 2) float32 (loop wrap and
end-of-track zero-fill live in the deck), and reads any number of output
frames; we produce internally in Hs quanta and buffer the remainder.
"""
import numpy as np
from scipy.signal import correlate

N = 2048
HS = 1024
SEARCH = 512
BYPASS_EPS = 0.001
XFADE = 128                      # frames, for bypass<->wsola switches


class WSOLAStretcher:
    def __init__(self, fetch, channels=2):
        self.fetch = fetch
        self.channels = channels
        self.rate = 1.0
        # While beat-synced, the rate trim dances around 1.0; letting the
        # stretcher flap between bypass and WSOLA shifts output timing by up
        # to the search width each flip (a random-walk phase kick that
        # defeats the sync). Pin WSOLA mode during sync.
        self.no_bypass = False
        self._win = np.hanning(N).astype(np.float64)[:, None]
        # Pending phase correction in SOURCE FRAMES (+ = advance = play
        # earlier material sooner). Absorbed a few frames per hop inside
        # the WSOLA search, so beat-phase micro-corrections are click-free
        # and pitch-true - this is how the sync PLL follows a track's
        # bar-to-bar swing instead of chasing it with slow rate trims.
        self.phase_trim = 0.0
        self._pos = 0.0          # nominal (rate-integrated) source frame
        self._q_prev = None      # source pos of the previous chosen window
        self._tail = np.zeros((N - HS, channels), dtype=np.float64)
        self._fifo = np.zeros((0, channels), dtype=np.float32)
        self._bypassed = True

    # -- public ------------------------------------------------------------
    def seek(self, source_frame):
        self._pos = float(source_frame)
        self.phase_trim = 0.0
        self._q_prev = None
        self._tail[:] = 0.0
        self._fifo = self._fifo[:0]

    @property
    def source_pos(self):
        """Fractional source cursor (frames) - the PLL's measurement."""
        return self._pos

    def read(self, n_out):
        """Produce n_out stretched frames (float32 (n_out, ch))."""
        while len(self._fifo) < n_out:
            self._produce()
        out, self._fifo = self._fifo[:n_out], self._fifo[n_out:]
        return out

    # -- internals ---------------------------------------------------------
    def _produce(self):
        want_bypass = (abs(self.rate - 1.0) < BYPASS_EPS
                       and not self.no_bypass)
        if want_bypass != self._bypassed:
            # Render one block in the NEW mode and crossfade from the old
            # mode's continuation so the seam never clicks.
            old = self._render_bypass(HS) if self._bypassed \
                else self._render_wsola()
            self._bypassed = want_bypass
            if want_bypass:
                self._pos = (self._q_prev + HS) if self._q_prev is not None \
                    else self._pos
                self._q_prev = None
                self._tail[:] = 0.0
            new = self._render_bypass(HS) if want_bypass \
                else self._render_wsola()
            # The two renders both advanced the cursor; keep the new mode's.
            f = np.linspace(0.0, 1.0, min(XFADE, HS))[:, None]
            blk = new.copy()
            blk[:len(f)] = old[:len(f)] * (1 - f) + new[:len(f)] * f
            self._push(blk)
            return
        self._push(self._render_bypass(HS) if want_bypass
                   else self._render_wsola())

    def _push(self, blk):
        self._fifo = np.concatenate(
            [self._fifo, blk.astype(np.float32)], axis=0)

    def _render_bypass(self, n):
        p = int(round(self._pos))
        blk = self.fetch(p, n).astype(np.float64)
        self._pos = p + n
        return blk

    def _render_wsola(self):
        ha = self.rate * HS
        if self._q_prev is None:
            q = int(round(self._pos))
            w = self.fetch(q, N).astype(np.float64) * self._win
            # First block: top up the Hann fade-in with raw signal so output
            # starts at unity amplitude instead of swelling from silence.
            out = w[:HS] + self.fetch(q, HS).astype(np.float64) \
                * (1.0 - self._win[:HS])
            self._tail = w[HS:].copy()
            self._q_prev = q
            self._pos += ha
            return out
        # Natural continuation of the last chosen window.
        target = self.fetch(self._q_prev + HS, HS).astype(np.float64)
        t_mono = target.mean(axis=1)
        base = int(round(self._pos))
        lo = max(0, base - SEARCH)
        region = self.fetch(lo, (base - lo) + SEARCH + HS).astype(np.float64)
        r_mono = region.mean(axis=1)
        # Normalized cross-correlation: plain correlation drifts toward the
        # loudest patch; dividing by sliding energy keeps it about SHAPE.
        num = correlate(r_mono, t_mono, mode="valid", method="fft")
        csum = np.concatenate([[0.0], np.cumsum(r_mono * r_mono)])
        energy = csum[HS:] - csum[:-HS]
        ncc = num / np.sqrt(np.maximum(energy, 1e-9) *
                            max(np.dot(t_mono, t_mono), 1e-9))
        q = lo + int(np.argmax(ncc))
        w = self.fetch(q, N).astype(np.float64) * self._win
        out = self._tail + w[:HS]           # 50% Hann OLA sums to unity
        self._tail = w[HS:].copy()
        self._q_prev = q
        # Bleed pending phase correction into the cursor, bounded per hop
        # (64 frames/hop ~ 62 ms/s of correction) so the search always
        # re-aligns within its window and the OLA stays seamless.
        step = float(np.clip(self.phase_trim, -64.0, 64.0))
        self.phase_trim -= step
        self._pos += ha + step
        return out
