"""Realtime biquad cascades for the DJ decks: scipy's sosfilt, minus scipy.

``scipy.signal.sosfilt`` spends ~45 us per call validating and reshaping
its arguments before it filters a sample (moveaxis, _validate_sos, dtype
probing). The deck EQ and sweep filter call it on 256-frame sub-blocks -
three cascades per deck, eight sub-blocks per render block - so on a
two-deck blend that wrapper alone was ~3 ms of every 46 ms of audio
(profile 2026-10-23), all of it holding the GIL against the render loop.

``sosfilt`` here is a drop-in for ``scipy.signal.sosfilt(sos, x, axis=0,
zi=zi)`` on ``(n, channels)`` blocks: the same transposed-direct-form-II
recurrence scipy's C loop runs, in the same float64 arithmetic, so the
output is BIT-IDENTICAL (checked over whole rendered DJ scenarios, not
just a block). It runs under Numba with the GIL released and no fastmath
(no FMA contraction, no reassociation - either would change bits).

Only the realtime paths use it (lib/dj/eq.py, lib/dj/fx.py). Analysis and
measurement code keeps scipy so the verification instruments are
untouched. Falls back to scipy when Numba is unavailable.
"""
import numpy as np
from scipy.signal import sosfilt as _scipy_sosfilt

try:
    from numba import njit

    @njit(nogil=True, cache=True)
    def _cascade(sos, y, zi):
        # y: (n, ch) float64, filtered in place. zi: (sections, 2, ch) in place.
        n, ch = y.shape
        for s in range(sos.shape[0]):
            b0 = sos[s, 0]
            b1 = sos[s, 1]
            b2 = sos[s, 2]
            a1 = sos[s, 4]
            a2 = sos[s, 5]
            for c in range(ch):
                z0 = zi[s, 0, c]
                z1 = zi[s, 1, c]
                for i in range(n):
                    x_cur = y[i, c]
                    x_new = b0 * x_cur + z0
                    z0 = b1 * x_cur - a1 * x_new + z1
                    z1 = b2 * x_cur - a2 * x_new
                    y[i, c] = x_new
                zi[s, 0, c] = z0
                zi[s, 1, c] = z1

    # Compile (or load the on-disk cache) NOW, on whatever thread imports
    # the decks - never on the audio render thread's first EQ'd block.
    _cascade(np.zeros((1, 6)), np.zeros((1, 2), order="F"), np.zeros((1, 2, 2)))
    HAVE_FAST = True
except Exception:                                   # no numba: scipy it is
    HAVE_FAST = False


def sosfilt(sos, x, zi):
    """``scipy.signal.sosfilt(sos, x, axis=0, zi=zi)`` for (n, ch) blocks.

    Returns ``(y, zf)`` as scipy does: new float64 arrays, inputs untouched.
    ``y`` is COLUMN-major like scipy's (it filters a transposed view): the
    layout survives ``astype`` into the mix, and numpy's pairwise sums
    (the block RMS in submix telemetry) add in layout order - a C-order
    result moved the dead-air RMS by 7e-9. Same layout, same bits.
    """
    if not HAVE_FAST:
        return _scipy_sosfilt(sos, x, axis=0, zi=zi)
    y = np.array(x, dtype=np.float64, order="F")     # copy, as scipy's result is
    zf = np.array(zi, dtype=np.float64, order="C")
    _cascade(np.ascontiguousarray(sos, dtype=np.float64), y, zf)
    return y, zf
