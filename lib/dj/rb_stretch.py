"""Streaming Rubber Band keylock engine for the DJ decks.

Rubber Band (Breakfast Quay, the library behind Mixxx's high-quality
keylock) via the `pylibrb` binding - prebuilt wheels for Windows, Linux
x86_64 and macOS, so `pip install pylibrb` is the whole install on the
platforms we deploy to (other arches build from source; the deck falls
back to varispeed with a printed warning when the import fails).

Engine R3 ("finer") by default: R2 ("faster") passed the synthetic gate
but WARBLED on sustained tones on the real library (user-heard,
2026-07-22) - and audible warble is the artifact keylock exists to avoid.
R3 is warble-free at real-world rates; its one measured weakness is
transient duplication at the 8%-slowdown extreme (5.02 onsets/s vs a
4.72 bound), which the dual-bend cap (6%/deck) makes rare in practice.
DJ_RB_ENGINE=faster re-selects R2 ("crisp": sharper attacks, ~2% CPU vs
~7%, tonal warble) - also exposed as 'rubberband-crisp' in the planner's
engine picker for A/B.

Same pull interface as WSOLAStretcher (drop-in): fetch(pos, n) -> (n, ch),
read any number of frames. rate/seek/source_pos/phase_trim/no_bypass
behave identically so the submix PLL needs no changes. Differences vs the
home-grown engines:

  - Rubber Band wants CONTIGUOUS input, so phase_trim is absorbed as a
    momentary TIME-RATIO bias (<= 64 source frames per 1024-out block,
    same authority as WSOLA's search absorption) instead of a cursor
    splice - under keylock a ratio bias is pitch-free by definition, so
    the correction is inaudible.
  - No bit-exact bypass at rate 1.0: R3 at ratio 1.0 is near-transparent
    but always processes. (The PLL pins no_bypass during sync anyway, so
    the engines behave the same where it matters.)
  - Startup latency is swallowed at seek() by the documented pad/drop
    dance (feed preferred_start_pad zeros, drop start_delay output), so
    read() returns the cue material from the first frame - seam event
    timing stays sample-honest.

Key shifts ride the existing deck path (stretcher at rate/pitch_f +
resample) - engine-agnostic, unchanged here.
"""
import os

import numpy as np

SR = 44100
BLOCK_OUT = 1024                 # output quantum per produce cycle
TRIM_CLIP = 64.0                 # source frames of PLL correction per block
BYPASS_EPS = 0.001
XFADE = 128                      # frames, for bypass<->rb switches


def available():
    try:
        import pylibrb                                    # noqa: F401
        return True
    except Exception:
        return False


class RubberBandDeckStretcher:
    def __init__(self, fetch, channels=2):
        from pylibrb import Option, RubberBandStretcher
        self.fetch = fetch
        self.channels = channels
        self.rate = 1.0
        self.no_bypass = False       # API compat; RB never bypasses
        self.phase_trim = 0.0
        self._pos = 0.0              # nominal (rate-integrated) source frame
        opts = (Option.PROCESS_REALTIME | Option.CHANNELS_TOGETHER
                | Option.TRANSIENTS_CRISP | Option.THREADING_NEVER)
        if os.environ.get("DJ_RB_ENGINE", "finer").lower() == "faster":
            opts |= Option.ENGINE_FASTER     # R2 "crisp": sharp attacks,
        else:                                # tonal warble (user-heard)
            opts |= Option.ENGINE_FINER      # R3: warble-free (default)
        det = os.environ.get("DJ_RB_DETECTOR", "compound").lower()
        if det == "percussive":
            opts |= Option.DETECTOR_PERCUSSIVE
        elif det == "soft":
            opts |= Option.DETECTOR_SOFT
        self._st = RubberBandStretcher(sample_rate=SR, channels=channels,
                                       options=opts)
        self._src = 0                # next source frame to FEED (contiguous)
        self._drop = 0               # start-delay output frames to discard
        self._fifo = np.zeros((0, channels), dtype=np.float32)
        self._bypassed = True        # bit-exact passthrough at rate ~1.0

    # -- public (same contract as WSOLAStretcher) --------------------------
    def seek(self, source_frame):
        self.phase_trim = 0.0
        self._pos = float(source_frame)
        self._fifo = self._fifo[:0]
        self._bypassed = True        # re-primes on first stretched block
        self._prime()

    def _prime(self):
        """Reset RB and swallow its startup latency (documented pad/drop
        dance) so the next stretched output frame corresponds to the
        cursor - seam event timing stays sample-honest."""
        self._st.reset()
        self._src = int(round(self._pos))
        self._st.time_ratio = 1.0 / max(self.rate, 0.05)
        pad = int(self._st.get_preferred_start_pad())
        if pad > 0:
            self._st.process(np.zeros((self.channels, pad),
                                      dtype=np.float32), False)
        self._drop = int(self._st.get_start_delay())

    @property
    def source_pos(self):
        return self._pos

    def read(self, n_out):
        while len(self._fifo) < n_out:
            self._produce()
        out, self._fifo = self._fifo[:n_out], self._fifo[n_out:]
        return out

    # -- internals ---------------------------------------------------------
    def _produce(self):
        # Bit-exact BYPASS at rate ~1.0 (same contract as WSOLA): most of
        # the night a deck plays un-stretched, and passthrough is both
        # exact and free. Crossfade the mode switches; re-entering RB mode
        # re-primes the engine at the cursor.
        want_bypass = (abs(self.rate - 1.0) < BYPASS_EPS
                       and not self.no_bypass
                       and abs(self.phase_trim) < 1.0)
        if want_bypass != self._bypassed:
            old = self._render_rb() if not self._bypassed \
                else self._render_bypass(BLOCK_OUT)
            self._bypassed = want_bypass
            if not want_bypass:
                self._prime()            # RB restarts at the cursor
            new = self._render_bypass(BLOCK_OUT) if want_bypass \
                else self._render_rb()
            f = np.linspace(0.0, 1.0, min(XFADE, len(new)))[:, None]
            blk = new.copy()
            blk[:len(f)] = old[:len(f)] * (1 - f) + new[:len(f)] * f
            self._fifo = np.concatenate(
                [self._fifo, blk.astype(np.float32)], axis=0)
            return
        if want_bypass:
            blk = self._render_bypass(BLOCK_OUT)
            self._fifo = np.concatenate(
                [self._fifo, blk.astype(np.float32)], axis=0)
            return
        blk = self._render_rb()
        self._fifo = np.concatenate(
            [self._fifo, blk.astype(np.float32)], axis=0)

    def _render_bypass(self, n):
        p = int(round(self._pos))
        blk = self.fetch(p, n).astype(np.float32)
        self._pos = p + n
        return blk

    def _render_rb(self):
        # Absorb pending PLL phase correction as a ratio bias this block:
        # consuming `step` extra source frames over BLOCK_OUT output frames
        # advances beat phase exactly like WSOLA's cursor step, with no
        # input splice (keylock makes the momentary tempo bias pitch-free).
        step = float(np.clip(self.phase_trim, -TRIM_CLIP, TRIM_CLIP))
        self.phase_trim -= step
        r_eff = max(self.rate + step / BLOCK_OUT, 0.05)
        self._st.time_ratio = 1.0 / r_eff
        chunks, got = [], 0
        while got < BLOCK_OUT:
            avail = int(self._st.available())
            if avail <= 0:
                need = max(int(self._st.get_samples_required()), 256)
                blk = self.fetch(self._src, need)     # (need, ch) float32
                self._src += need
                self._st.process(
                    np.ascontiguousarray(blk.astype(np.float32).T), False)
                continue
            out = self._st.retrieve(
                min(avail, BLOCK_OUT - got + self._drop))   # (ch, n)
            if self._drop > 0:
                d = min(self._drop, out.shape[1])
                out = out[:, d:]
                self._drop -= d
            if out.shape[1]:
                chunks.append(out.T)
                got += out.shape[1]
        blk = np.concatenate(chunks, axis=0).astype(np.float32)
        self._pos += r_eff * len(blk)
        return blk
