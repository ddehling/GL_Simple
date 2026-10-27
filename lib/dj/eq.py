"""Three-band DJ EQ: Linkwitz-Riley 4th-order crossovers at 200 / 2500 Hz.

LR4 = squared 2nd-order Butterworth, so the three bands sum back to (almost)
flat when all gains are 1.0 - killing the low band during a bass swap leaves
mids/highs untouched. Stateful (sosfilt zi per channel), block-based,
with per-block linear gain ramps so automation never zipper-clicks.
Filtering runs through lib/dj/sosfast.py: scipy's exact recurrence without
scipy's per-call wrapper (bit-identical output, GIL released).
"""
import numpy as np
from scipy.signal import butter

from lib.dj.sosfast import sosfilt

RATE = 44100
LOW_HZ = 200.0
HIGH_HZ = 2500.0
MODE_XFADE_S = 0.08              # a topology change (bypass / a shortcut / the full split) crossfades over this
MODE_XFADE = int(MODE_XFADE_S * RATE)
WARM_FRAMES = int(0.06 * RATE)   # input history a cold cascade is settled on before it joins a crossfade


def _lr4(freq, kind):
    """4th-order Linkwitz-Riley as a cascade of two 2nd-order Butterworths."""
    sos = butter(2, freq, kind, fs=RATE, output="sos")
    return np.vstack([sos, sos])


class ThreeBandEQ:
    def __init__(self, channels=2):
        self.channels = channels
        # One cascade per band, both channels filtered in a single
        # sosfilt call (axis=0). scipy's per-call Python overhead (~90us
        # of arg validation) dominated the actual filtering at 256-frame
        # sub-blocks - the old per-channel/per-stage layout was 16 calls
        # per process(), this is 3 (perf audit 2026-07-31). The merged
        # mid cascade (band-pass = HP then LP) is the same math: a
        # cascade of cascades is one longer cascade.
        self._sos = {
            "low": _lr4(LOW_HZ, "low"),
            "mid": np.vstack([_lr4(LOW_HZ, "high"), _lr4(HIGH_HZ, "low")]),
            "high": _lr4(HIGH_HZ, "high"),
        }
        self._zi = {k: np.zeros((len(s), 2, channels))
                    for k, s in self._sos.items()}
        self.gains = np.ones(3)          # current low/mid/high
        self._targets = np.ones(3)
        self._ramp_left_s = 0.0
        self._active = False

    def set_gain(self, band, gain, ramp_s=0.05):
        """band: 0=low 1=mid 2=high; gain is linear (0..~1.5)."""
        self._targets[band] = float(gain)
        self._ramp_left_s = max(self._ramp_left_s, float(ramp_s))

    def set_gains(self, low=None, mid=None, high=None, ramp_s=0.05):
        for i, g in enumerate((low, mid, high)):
            if g is not None:
                self._targets[i] = float(g)
        self._ramp_left_s = max(self._ramp_left_s, float(ramp_s))

    @property
    def is_flat(self):
        """Nothing to process: at unity, and no topology crossfade still running (a caller that skips process()
        while flat must not cut a fade short)."""
        return (np.all(np.abs(self.gains - 1.0) < 1e-3)
                and np.all(np.abs(self._targets - 1.0) < 1e-3)
                and getattr(self, "_xf", None) is None
                and getattr(self, "_mode", "bypass") == "bypass")

    def _run(self, key, x):
        y, self._zi[key] = sosfilt(self._sos[key], x, self._zi[key])
        return y

    def process(self, block):
        """block: (n, channels) float32. Returns the EQ'd block.

        Bit-exact bypass while flat. The LR4 band sum is an ALLPASS (phase rotates near the crossovers), so the
        processed signal and the raw one differ in PHASE even at unity gain - and the one-band shortcuts differ
        from the full split the same way. Every change of TOPOLOGY (bypass <-> a shortcut <-> the full split)
        is therefore crossfaded over MODE_XFADE_S across blocks, both paths rendered meanwhile. The old one-block
        (6-23 ms) crossfade was a thump on every return to flat: at the end of a bass swap the sub-bass of the two
        versions sits half a cycle apart (measured 2026-10-24: a 0.18 full-scale step where the music's own jumps
        were 0.03 - "definitely some clicks")."""
        n = len(block)
        if n == 0:
            return block
        dt = n / RATE
        if self._ramp_left_s <= 0.0 and np.all(self._targets == self.gains):
            g0 = g1 = self.gains
        else:
            frac = 1.0 if self._ramp_left_s <= dt else dt / self._ramp_left_s
            g0 = self.gains.copy()
            g1 = g0 + (self._targets - g0) * frac
            self.gains = g1
            self._ramp_left_s = max(0.0, self._ramp_left_s - dt)
        flat = np.all(g0 == 1.0) and np.all(g1 == 1.0)
        # ONE-BAND SHORTCUT: when only the low (or only the high) band moves - the low carve on a layered
        # voice, the Director's BASS or TONE - the other two bands are the input minus that band, so a
        # single cascade does it: y = x + (g - 1) * band. Three cascades cost 28 % of the layered mixer
        # (profile 2026-09-12); this is a third of that.
        only_low = np.all(g0[1:] == 1.0) and np.all(g1[1:] == 1.0)
        only_high = np.all(g0[:2] == 1.0) and np.all(g1[:2] == 1.0)
        want = "bypass" if flat else ("low" if only_low else ("high" if only_high else "full"))
        mode = getattr(self, "_mode", None)
        xf = getattr(self, "_xf", None)
        if mode is None:
            mode = self._mode = want
        # the last WARM_FRAMES of input: a cascade the new topology needs that the old one did not run is
        # settled on them first (output discarded), so the crossfade starts from a warm filter, not a step response
        hist = getattr(self, "_hist", None)
        if xf is None and want != mode:
            needs = {"bypass": (), "low": ("low",), "high": ("high",), "full": ("low", "mid", "high")}
            cold = [k for k in needs[want] if k not in needs[mode]]
            if cold and hist is not None and len(hist):
                for k in cold:
                    self._zi[k][:] = 0.0
                    self._run(k, hist.astype(np.float64))
            xf = self._xf = {"from": mode, "to": want, "done": 0}
        if xf is None and want == "bypass":
            self._active = False
            self._hist = block if n >= WARM_FRAMES else (block if hist is None else np.concatenate([hist, block])[-WARM_FRAMES:])
            return block                                  # true bypass, bit-exact
        self._hist = block if n >= WARM_FRAMES else (block if hist is None else np.concatenate([hist, block])[-WARM_FRAMES:])
        x = block.astype(np.float64)
        ramp = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        gl = g0[0] + (g1[0] - g0[0]) * ramp
        gm = g0[1] + (g1[1] - g0[1]) * ramp
        gh = g0[2] + (g1[2] - g0[2]) * ramp
        rendered, bands = {}, {}

        def band(k):
            # each cascade runs ONCE per block: two topologies in a crossfade may share one (running it twice
            # would feed the block through its state twice - garbage, measured 4x jumps)
            if k not in bands:
                bands[k] = self._run(k, x)
            return bands[k]

        def render(m):
            if m in rendered:
                return rendered[m]
            if m == "bypass":
                y = x
            elif m == "low":
                y = x + (gl - 1.0) * band("low")
            elif m == "high":
                y = x + (gh - 1.0) * band("high")
            else:
                y = band("low") * gl + band("mid") * gm + band("high") * gh
            rendered[m] = y
            return y
        self._active = True
        if xf is None:
            y = render(mode)
        else:
            # both topologies, crossfaded over MODE_XFADE frames across blocks (the new path's filters start cold:
            # their settling is hidden under the fade)
            a, b = render(xf["from"]), render(xf["to"])
            w = np.clip((xf["done"] + np.arange(n)) / float(MODE_XFADE), 0.0, 1.0)[:, None]
            y = a * (1.0 - w) + b * w
            xf["done"] += n
            if xf["done"] >= MODE_XFADE:
                self._mode = mode = xf["to"]
                self._xf = xf = None
                if mode == "bypass":
                    self._active = False
                    for k in self._zi:                    # cold state is fine: the next engage fades in
                        self._zi[k][:] = 0.0
        return y.astype(np.float32)
