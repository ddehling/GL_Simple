"""
Beat / tempo detector for the audio-reactive shader system.

A pure CONSUMER of the dict produced by ``MicrophoneAnalyzer.get_extended_analysis()``
- it does NOT touch the audio capture thread. Call ``update(sound_dict, dt)`` once per
render frame from ``Stories_OGL.send_variables()``; it publishes onset / decay / bpm /
phase values into ``outstate`` so any shader can sync visuals to the beat.

Approach (tuned against real 4-on-the-floor dance music, where a naive
threshold-on-flux detector over-triggers on off-beat hats and the 8th-note
bassline):

  1. Onset envelope: bass-weighted spectral flux per frame.
  2. Tempo: autocorrelation of a ~8s onset-envelope window, harmonic-summed
     and biased toward the dance range, so it locks the beat period instead
     of a half/double-time octave.
  3. Beat emission: a phase-locked oscillator runs at the estimated tempo and
     emits one beat per cycle; strong onsets nudge its phase (PLL) so it stays
     aligned with the kick. Beats therefore come at the musical tempo (~2/s at
     120 BPM), not on every transient.

Design notes:
  * Deterministic DSP only - NO ``np.random`` (the engine's weather-transition
    RNG stream must not be perturbed; see np_random_global_state_pollution).
  * Cheap: per-frame work is O(32) for onset; the autocorrelation runs only
    every few frames over a fixed ~320-sample buffer.
  * ``update()`` is wrapped so any internal error returns a zeroed result and
    never breaks the render loop.

Published keys (set in send_variables):
  beat            bool  - True only on the frame a beat fires
  beat_decay      float - 1 -> 0 envelope after each beat (fast flash ramp)
  bpm             float - smoothed tempo estimate (0 when no audio)
  beat_phase      float - 0..1 metronome, locked to the beat
  beat_intensity  float - raw onset strength, for scaling flash amount
"""

import math
from collections import deque

import numpy as np

FPS = 40.0                     # analysis frame rate the detector assumes


class BeatDetector:
    """Onset + tempo (autocorrelation) + phase-locked beat oscillator."""

    def __init__(self):
        # --- Onset (bass-weighted spectral flux) ---
        self._bass_ema = 0.0       # fast baseline for sub/low bass
        self._ema_alpha = 0.25     # ~0.25s baseline at 40fps
        self._flux_hist = deque(maxlen=40)   # ~1s of onset strengths

        # --- Onset envelope for tempo estimation (~8s) ---
        self._env = deque(maxlen=320)

        # --- Tempo (BPM) ---
        self._bpm = 120.0
        self._bpm_alpha = 0.12
        self._min_bpm = 80.0
        self._max_bpm = 170.0
        self._pref_lo = 110.0      # dance-range bias (deep/melodic house)
        self._pref_hi = 132.0
        self._tempo_decim = 8      # recompute tempo every N frames
        self._frame = 0
        self._have_tempo = False

        # --- Phase-locked oscillator ---
        self._phase = 0.0          # 0..1
        self._pll_gain = 0.10      # how hard a strong onset pulls the phase
        self._decay = 0.0
        self._decay_tau = 0.12     # flash-envelope time constant
        self._last_strength = 0.0

    # ------------------------------------------------------------------
    def update(self, sound, dt):
        """Advance one frame. ``sound`` is outstate['sound'] (or None)."""
        try:
            return self._update(sound, dt)
        except Exception:
            return {"onset": False, "decay": self._decay, "bpm": 0.0,
                    "phase": self._phase, "strength": 0.0}

    def _idle(self, dt):
        """No usable audio: bleed the envelope, hold the phase, no tempo."""
        self._decay *= math.exp(-dt / self._decay_tau)
        return {"onset": False, "decay": self._decay, "bpm": 0.0,
                "phase": self._phase, "strength": 0.0}

    def _update(self, sound, dt):
        dt = float(dt) if dt and dt > 0.0 else (1.0 / FPS)
        if dt > 0.5:
            dt = 0.5

        if sound is None:
            return self._idle(dt)
        bands = sound.get("norm_short")
        if bands is None or len(bands) == 0:
            return self._idle(dt)

        row = np.asarray(bands[0], dtype=np.float32)
        n = row.shape[0]

        # Onset from SUB/LOW BASS only (the kick lives ~40-150 Hz, bands 0..5).
        # Restricting to bass keeps off-beat hats / claps from firing the PLL.
        bass = float(np.mean(row[0:min(6, n)]))
        a = self._ema_alpha
        self._bass_ema = (1.0 - a) * self._bass_ema + a * bass
        onset_strength = max(0.0, bass - self._bass_ema)

        self._flux_hist.append(onset_strength)
        self._env.append(onset_strength)
        self._last_strength = onset_strength

        # Re-estimate tempo periodically from the onset envelope.
        self._frame += 1
        if self._frame % self._tempo_decim == 0:
            self._estimate_tempo()

        # Adaptive "strong onset" gate for PLL phase correction.
        if len(self._flux_hist) >= 8:
            med = float(np.median(np.asarray(self._flux_hist, dtype=np.float32)))
        else:
            med = 0.0
        strong = onset_strength > (med * 1.8 + 0.02)

        # Advance the phase-locked oscillator at the current tempo.
        self._phase += dt * (self._bpm / 60.0)

        # PLL: a strong onset nudges the phase toward the nearest beat (0/1).
        if strong and self._have_tempo:
            err = self._phase - round(self._phase)   # signed distance to a tick, [-0.5,0.5]
            self._phase -= self._pll_gain * err

        # Emit a beat when the oscillator crosses a cycle boundary.
        onset = False
        if self._phase >= 1.0:
            self._phase -= math.floor(self._phase)
            if self._have_tempo:
                onset = True
                self._decay = 1.0
        if not onset:
            self._decay *= math.exp(-dt / self._decay_tau)

        return {"onset": onset, "decay": self._decay,
                "bpm": (self._bpm if self._have_tempo else 0.0),
                "phase": self._phase % 1.0, "strength": onset_strength}

    # ------------------------------------------------------------------
    def _estimate_tempo(self):
        """Autocorrelate the onset envelope and pick the beat period, biased
        toward the dance range and harmonic-summed to avoid octave errors."""
        if len(self._env) < 120:                  # need ~3s before trusting tempo
            return
        x = np.asarray(self._env, dtype=np.float64)
        x = x - x.mean()
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        if ac[0] <= 1e-9:
            return
        ac = ac / ac[0]

        min_lag = int(round(60.0 * FPS / self._max_bpm))
        max_lag = int(round(60.0 * FPS / self._min_bpm))
        max_lag = min(max_lag, len(ac) - 1)

        best_lag, best_score = 0, -1.0
        for lag in range(min_lag, max_lag + 1):
            # Comb / harmonic sum: a true beat period also has peaks at 2x/3x
            # the lag, so summing them favours the fundamental over a random
            # off-beat lag and over the half-time octave.
            score = ac[lag]
            if 2 * lag < len(ac):
                score += 0.5 * ac[2 * lag]
            if 3 * lag < len(ac):
                score += 0.33 * ac[3 * lag]
            bpm = 60.0 * FPS / lag
            if self._pref_lo <= bpm <= self._pref_hi:
                score *= 1.15                      # gentle dance-range bias
            if score > best_score:
                best_score, best_lag = score, lag

        if best_lag <= 0:
            return
        new_bpm = 60.0 * FPS / best_lag
        if not self._have_tempo:
            self._bpm = new_bpm
            self._have_tempo = True
        else:
            self._bpm = (1.0 - self._bpm_alpha) * self._bpm + self._bpm_alpha * new_bpm
