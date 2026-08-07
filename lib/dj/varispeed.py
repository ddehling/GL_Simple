"""Varispeed "turntable" tempo engine for the DJ decks - pitch rides tempo.

WHY (2026-07-12, user decision): both keylock engines trade artifacts -
WSOLA's chunk overlap-add warbles (~43 Hz roughness while stretched, the
"subtle chopping" during transitions) and the phase vocoder swaps that for
seam thinning. A plain resampler has NO time-stretch artifact at all: it is
what a turntable's pitch fader does. The cost is that pitch follows tempo
(rate 1.04 = +0.68 semitone), which the user explicitly accepted - and the
brain halves it by bending BOTH decks to a meeting tempo (each moves half
the log-distance; see Brain.plan_transition varispeed split), so a typical
3-4% gap is under a fifth of a semitone per song.

Implementation: streaming 4-point Catmull-Rom interpolation over the deck's
fetch() pull - full quality for DJ-sized rates (0.90..1.10, the wall in
brain.STRETCH_MIN/MAX), vectorized,
far cheaper than either keylock engine.

Same pull interface as WSOLAStretcher (drop-in): fetch(pos, n) -> (n, ch)
float32; rate / seek() / source_pos / phase_trim / no_bypass identical, so
the submix sync PLL is unchanged. Two behavioral notes:
  - source_pos is EXACT (WSOLA's real output lags its nominal cursor by up
    to the +/-512-frame search), so PLL measurements get cleaner.
  - phase_trim (PLL micro-correction, source frames) is bled at <=1.5% of
    realtime: with varispeed a position slew IS a momentary pitch bend, so
    it must stay gentle - exactly a DJ's finger-drag on the platter.
Bit-exact bypass at rate 1.0 (crossfaded mode switches, like WSOLA).
"""
import numpy as np

BYPASS_EPS = 0.001
XFADE = 128                      # frames, for bypass<->interp switches
# Max phase-trim bleed per output frame. Under varispeed a position slew IS
# a momentary pitch bend: 0.006 = ~10 cents at full tilt (was 0.015 = 26
# cents, which read as a tonal wobble during PLL nudges). A typical 24 ms
# nudge now absorbs in ~4 s - still well inside the 1.6 s-spaced nudge
# cadence because corrections queue and bleed continuously.
TRIM_FRAC = 0.006


class VarispeedStretcher:
    def __init__(self, fetch, channels=2):
        self.fetch = fetch
        self.channels = channels
        self.rate = 1.0
        self.no_bypass = False
        self.phase_trim = 0.0
        self._pos = 0.0              # fractional source cursor (frames)
        self._bypassed = None        # None = first read, no switch xfade

    # -- public ------------------------------------------------------------
    def seek(self, source_frame):
        self._pos = float(source_frame)
        self.phase_trim = 0.0

    @property
    def source_pos(self):
        """Fractional source cursor (frames) - the PLL's measurement."""
        return self._pos

    def read(self, n_out):
        """Produce n_out frames (float32 (n_out, ch)) at the current rate."""
        if n_out <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        want_bypass = (abs(self.rate - 1.0) < BYPASS_EPS
                       and not self.no_bypass)
        if self._bypassed is None:
            self._bypassed = want_bypass
        if want_bypass != self._bypassed:
            # Render one block in the old mode and one in the new from the
            # same cursor, crossfade - the seam never clicks. Keep the new
            # mode's cursor advance.
            pos0 = self._pos
            old = self._render(n_out, self._bypassed)
            self._pos = pos0
            self._bypassed = want_bypass
            if want_bypass:
                self._pos = float(int(round(self._pos)))
            new = self._render(n_out, want_bypass)
            k = min(XFADE, n_out)
            f = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
            out = new.copy()
            out[:k] = old[:k] * (1.0 - f) + new[:k] * f
            return out
        return self._render(n_out, want_bypass)

    # -- internals -----------------------------------------------------------
    def _render(self, n, bypass):
        if bypass:
            p = int(round(self._pos))
            self._pos = float(p + n)
            return self.fetch(p, n)
        # Bleed pending PLL phase correction gently (a slew IS a pitch bend).
        step = float(np.clip(self.phase_trim, -TRIM_FRAC * n, TRIM_FRAC * n))
        self.phase_trim -= step
        adv = self.rate + step / n
        p = self._pos + adv * np.arange(n, dtype=np.float64)
        self._pos += adv * n
        lo = float(p.min())
        hi = float(p.max())
        base = int(np.floor(lo)) - 1
        span = int(np.floor(hi)) + 2 - base + 1
        block = self.fetch(base, span).astype(np.float64)
        idx = p - base
        i0 = np.floor(idx).astype(np.int64)
        t = (idx - i0)[:, None]
        p0 = block[i0 - 1]
        p1 = block[i0]
        p2 = block[i0 + 1]
        p3 = block[i0 + 2]
        # Catmull-Rom (4-point, 3rd order) - transparent at DJ rates.
        out = p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2
                                             - p3 + t * (3.0 * (p1 - p2)
                                                         + p3 - p0)))
        return out.astype(np.float32)
