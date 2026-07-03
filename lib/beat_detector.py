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
  beat_confidence float - 0..1 tempo+phase lock quality; gate beat-locked
                          visuals on this so they degrade to free-run
                          instead of misfiring on garbage phase
  beat_count      int   - monotonic beat index (never resets while running)
  bar_phase       float - 0..1 over a 4-beat bar. The bar ORIGIN IS
  phrase_phase    float - 0..1 over a 16-beat phrase.  ARBITRARY - we count
                          beats from detector start, we do not detect the
                          musical downbeat. Use for slow cyclic evolution,
                          not "flash on the One".
"""

import math
from collections import deque

import numpy as np

FPS = 40.0                     # analysis frame rate the detector assumes


class BeatDetector:
    """Onset + tempo (autocorrelation) + phase-locked beat oscillator."""

    def __init__(self):
        # --- Onset (bass-weighted spectral flux + a percussive mid channel) ---
        self._bass_ema = 0.0       # fast baseline for sub/low bass
        self._perc_ema = 0.0       # fast baseline for low-mids (soft kicks,
                                   # snares - carries the pulse in chill music
                                   # where sustained bass drones bury the kick)
        self._ema_alpha = 0.25     # ~0.25s baseline at 40fps
        self._flux_hist = deque(maxlen=40)   # ~1s of onset strengths

        # --- Onset envelope for tempo estimation (~8s) ---
        self._env = deque(maxlen=320)

        # --- Tempo (BPM) ---
        self._bpm = 120.0
        self._bpm_alpha = 0.12
        self._min_bpm = 65.0       # downtempo/chill lives at 65-100
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

        # --- Confidence + musical counting ---
        self._count = 0            # monotonic beat index
        self._q_tempo = 0.0        # how much the winning autocorr lag stands out
        self._q_phase = 0.0        # PLL lock quality (from onset phase error)
        self._phase_err_ema = 0.25 # EMA of |phase error| at strong onsets
        self._confidence = 0.0     # smoothed q_tempo * q_phase
        self._conf_tau = 1.0
        # Lock quality COASTS while music is playing (a kick resting for 8
        # bars is a breakdown, not a lost beat - the oscillator is still on
        # grid) and only dies fast in true silence. Real-track measurement:
        # the old flat 2.5s bleed lost lock on 6/6 club tracks and spent up
        # to 31% of a steady track below the consumers' conf gate.
        self._q_phase_coast_tau = 15.0  # music present, kick resting
        self._q_phase_bleed_tau = 1.2   # true silence: die fast
        self._level_peak = 1e-9         # slow AGC peak for presence detection
        # Escape hatches: the coast + syncopation gate + continuity bonus
        # are individually right but collectively SELF-CONFIRMING - after a
        # track change the new kicks land off the old grid and get rejected
        # as "syncopation" while confidence coasts on nothing. So rejected
        # onsets count against the grid, and sustained tempo disagreement
        # forces a hard re-lock.
        self._rejects = 0               # consecutive off-grid strong onsets
        self._tempo_disagree = 0        # consecutive raw-winner disagreements

    # ------------------------------------------------------------------
    def update(self, sound, dt):
        """Advance one frame. ``sound`` is outstate['sound'] (or None)."""
        try:
            return self._update(sound, dt)
        except Exception:
            return self._result(False, 0.0, bpm=0.0)

    def _result(self, onset, strength, bpm):
        phase = self._phase % 1.0
        return {"onset": onset, "decay": self._decay, "bpm": bpm,
                "phase": phase, "strength": strength,
                "confidence": self._confidence,
                "count": self._count,
                "bar_phase": ((self._count % 4) + phase) / 4.0,
                "phrase_phase": ((self._count % 16) + phase) / 16.0}

    def _idle(self, dt):
        """No usable audio: bleed the envelope, hold the phase, no tempo."""
        self._decay *= math.exp(-dt / self._decay_tau)
        self._q_phase *= math.exp(-dt / self._q_phase_bleed_tau)
        self._confidence += (0.0 - self._confidence) * (1.0 - math.exp(-dt / self._conf_tau))
        return self._result(False, 0.0, bpm=0.0)

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

        # Musical presence: raw band power vs a slow rolling peak. Separates
        # "the kick is resting" (coast the lock) from "the music stopped"
        # (let the lock die). Missing raw_bands -> assume present.
        raw = sound.get("raw_bands")
        if raw is not None and len(raw) > 0:
            lvl = float(np.mean(np.asarray(raw[0], dtype=np.float32)))
            self._level_peak = max(self._level_peak * math.exp(-dt / 60.0),
                                   lvl, 1e-9)
            present = lvl > 0.03 * self._level_peak
        else:
            present = True

        # Onset: bass flux first (the kick, bands 0..5), plus a downweighted
        # low-mid percussive channel (bands 6..15: soft kicks, snares, toms).
        # Chill/downtempo tracks often ride a sustained bass drone that
        # raises the bass baseline and buries a soft kick's flux - the
        # percussive channel still sees the pulse. Hats (bands 20+) stay
        # excluded so off-beat cymbals don't fire the PLL.
        bass = float(np.mean(row[0:min(6, n)]))
        perc = float(np.mean(row[6:min(16, n)])) if n > 6 else bass
        a = self._ema_alpha
        self._bass_ema = (1.0 - a) * self._bass_ema + a * bass
        self._perc_ema = (1.0 - a) * self._perc_ema + a * perc
        onset_strength = max(max(0.0, bass - self._bass_ema),
                             0.55 * max(0.0, perc - self._perc_ema))

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
        no_evidence = True
        if strong and self._have_tempo:
            err = self._phase - round(self._phase)   # signed distance to a tick, [-0.5,0.5]
            # Once locked, an onset far from the grid is almost always
            # syncopation (a bassline hit between kicks), not evidence that
            # the grid is wrong - letting it pull the phase and poison the
            # lock statistic is what made confidence flap on real tracks.
            if abs(err) <= 0.25 or self._q_phase <= 0.25:
                no_evidence = False
                self._rejects = 0
                self._phase -= self._pll_gain * err
                # Lock quality: small |err| at onsets means the oscillator is
                # sitting on the kicks. |err| is in [0, 0.5]; 1/6 cycle off -> 0.
                self._phase_err_ema = 0.9 * self._phase_err_ema + 0.1 * abs(err)
                self._q_phase = max(0.0, min(1.0, 1.0 - 3.0 * self._phase_err_ema))
            else:
                # An off-grid strong onset was rejected as syncopation. A
                # few of those are normal; a steady stream means the GRID
                # is wrong (track change) - surrender the lock so the next
                # onset re-seats the phase.
                self._rejects += 1
                if self._rejects >= 20:
                    # ~5s of nothing-but-off-grid hits: surrender softly
                    # (just below the gate, so the next onset re-seats us).
                    self._rejects = 0
                    self._q_phase = min(self._q_phase, 0.24)
                    self._phase_err_ema = 0.25
        if no_evidence:
            # Nothing said about the grid this frame. While music is still
            # playing this is a breakdown / syncopation - COAST (the tempo
            # grid is almost certainly still valid). In true silence, die fast.
            tau = self._q_phase_coast_tau if present else self._q_phase_bleed_tau
            self._q_phase *= math.exp(-dt / tau)

        target_conf = (self._q_tempo * self._q_phase) if self._have_tempo else 0.0
        self._confidence += (target_conf - self._confidence) * (1.0 - math.exp(-dt / self._conf_tau))

        # Emit a beat when the oscillator crosses a cycle boundary.
        onset = False
        if self._phase >= 1.0:
            self._phase -= math.floor(self._phase)
            if self._have_tempo:
                onset = True
                self._decay = 1.0
                self._count += 1
        if not onset:
            self._decay *= math.exp(-dt / self._decay_tau)

        return self._result(onset, onset_strength,
                            bpm=(self._bpm if self._have_tempo else 0.0))

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
        raw_lag, raw_score = 0, -1.0    # winner WITHOUT the continuity bonus
        score_sum, score_n = 0.0, 0
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
                score *= 1.08                      # gentle dance-range bias
                # (softened when chill support landed: tempo continuity now
                # provides the stability this prior used to; at 1.15 it was
                # dragging ~100 BPM downtempo up to a 126 BPM half-truth)
            if score > raw_score:
                raw_score, raw_lag = score, lag
            # Tempo continuity: once locked, the incumbent tempo (and its
            # neighbourhood) gets a bonus so soft material with several
            # plausible related lags doesn't flap between them.
            if self._have_tempo and abs(bpm - self._bpm) / self._bpm < 0.08:
                score *= 1.25
            score_sum += score
            score_n += 1
            if score > best_score:
                best_score, best_lag = score, lag

        # Hard re-lock: when the continuity-free winner keeps disagreeing
        # with the incumbent AND keeps agreeing with ITSELF (a real tempo
        # change is self-consistent; soft-material noise wanders), snap to
        # it instead of letting the bonus pin us to a dead track.
        raw_bpm = 60.0 * FPS / raw_lag if raw_lag > 0 else 0.0
        if (self._have_tempo and raw_bpm > 0.0
                and abs(raw_bpm - self._bpm) / self._bpm > 0.10
                and abs(raw_bpm - getattr(self, '_last_raw_bpm', 0.0))
                    / max(raw_bpm, 1.0) < 0.06):
            self._tempo_disagree += 1
            if self._tempo_disagree >= 15:     # ~3s of a CONSISTENT new tempo
                self._bpm = raw_bpm
                best_lag = raw_lag
                self._tempo_disagree = 0
                self._q_phase = min(self._q_phase, 0.2)   # re-seat the phase
                self._phase_err_ema = 0.25
        else:
            self._tempo_disagree = 0
        self._last_raw_bpm = raw_bpm

        if best_lag <= 0:
            return
        # Tempo quality: how much the winning lag stands out above the comb-sum
        # floor. Scores are sums of autocorrelation coefficients, so a real
        # groove peaks around 0.5-1.5 while noise sits near 0; an absolute
        # margin over the (non-negative) mean is robust where a ratio blows
        # up on a near-zero denominator.
        mean_score = score_sum / max(1, score_n)
        # Tempo quality: RELATIVE prominence of the winning lag (how much it
        # stands out of the comb-sum floor as a fraction of itself), scaled
        # by an absolute-evidence factor so near-silent noise can't fake a
        # prominent peak. The old absolute margin (/0.5) was tuned for hard
        # dance music and starved soft grooves whose autocorrelation runs
        # at half the amplitude.
        prominence = (best_score - max(mean_score, 0.0)) / max(best_score, 1e-6)
        evidence = max(0.0, min(1.0, best_score / 0.25))
        q = max(0.0, min(1.0, prominence * 1.8)) * evidence
        # Asymmetric smoothing: tempo trust builds quickly on good evidence
        # but decays slowly - a breakdown weakens the autocorrelation for a
        # few seconds without meaning the tempo actually changed.
        alpha = 0.2 if q > self._q_tempo else 0.03
        self._q_tempo = (1.0 - alpha) * self._q_tempo + alpha * q
        new_bpm = 60.0 * FPS / best_lag
        if not self._have_tempo:
            self._bpm = new_bpm
            self._have_tempo = True
        else:
            self._bpm = (1.0 - self._bpm_alpha) * self._bpm + self._bpm_alpha * new_bpm
