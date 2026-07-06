"""
Song-structure / band-energy signals for the audio-reactive shader system.

Companion to ``lib/beat_detector.py`` and, like it, a pure CONSUMER of the
dict from ``MicrophoneAnalyzer.get_extended_analysis()``. Call
``update(sound_dict, beat_result, dt)`` once per render frame from
``Stories_OGL.send_variables()`` (right after the BeatDetector update, with
the same wall-clock dt); it publishes scalar band energies and coarse song
structure (energy level, build-ups, drops) into ``outstate``.

Where the BeatDetector answers "where is the beat RIGHT NOW", this answers
"what is the song DOING" over seconds - the signals an auto-VJ or a
slow-evolving shader wants.

Silence gating: the analyzer's norm_* bands are self-normalized (AGC), so
during true silence tiny-noise / tiny-mean ~= 1.0 - "average" - which would
fake a mid-level energy forever. We therefore track the absolute raw level
against a slow-decaying peak of itself and gate every scalar by that ratio:
music sits near its own recent peak (gate ~1), silence sits orders of
magnitude below it (gate 0), independent of input gain.

Design notes (same contract as BeatDetector):
  * Deterministic DSP only - NO ``np.random``.
  * Cheap: O(32) per frame plus a small ring buffer.
  * ``update()`` is exception-wrapped and returns a safe zeroed dict on any
    internal error or when ``sound`` is None, so the render loop never breaks.

Published keys (set in send_variables):
  audio_bass    float - smoothed bass energy vs recent average (~0..2+,
  audio_mid             1.0 = at the 0.5s running average; 0 in silence).
  audio_high            Bands 0:8 / 8:20 / 20:32 of the 32-band spectrum.
                        NOTE: these HOVER AROUND 1.0 during steady music
                        (that's what AGC normalization means) - mapping
                        them linearly into brightness gives a static wash.
                        For visible reactivity use the *_punch keys below.
  bass_punch    float - 0..1 TRANSIENT envelopes: 0 between hits, snaps to
  mid_punch             ~1 on a hit in that range, releases in ~0.25s.
  high_punch            The signal to drive flashes/swells/pops with -
  audio_punch           real temporal contrast for free. audio_punch is
                        the broadband max of the three.
  audio_energy  float - 0..1 slow overall loudness vs the 2.5s baseline;
                        ~0.5 at a typical steady groove, 0 in silence.
  build_level   float - 0..1 riser detector (sustained energy climb x rising
                        highs). Peaks during a build-up before a drop.
  drop          bool  - True only on the frame a drop fires: bass went quiet
                        (breakdown, or true silence), then slams back.
  drop_decay    float - 1 -> 0 envelope after each drop (punch flashes).
"""

import math
from collections import deque

import numpy as np

# --- Band slices over the 32 log-spaced bands (40 Hz .. 16 kHz) ------------
BASS_SLICE = slice(0, 8)      # ~40-300 Hz
MID_SLICE = slice(8, 20)      # ~300-2000 Hz
HIGH_SLICE = slice(20, 32)    # ~2-16 kHz

# --- Smoothing time constants (seconds) -------------------------------------
BAND_TAU = 0.12               # bass/mid/high scalar smoothing
ENERGY_TAU = 1.5              # overall energy smoothing
DROP_DECAY_TAU = 0.35         # drop punch envelope

# --- Punch envelopes ---------------------------------------------------------
# A hit = the instantaneous norm_short mean of a range jumping above its
# steady-state ~1.0. Deviation above PUNCH_FLOOR maps to 0..1 over
# PUNCH_SPAN (norm 1.1 -> 0, norm 1.9 -> 1), attack instant, release
# exponential. This is what gives visuals temporal contrast: 0 between
# hits, ~1 on the hit.
PUNCH_FLOOR = 1.1
PUNCH_SPAN = 0.8
PUNCH_RELEASE_TAU = 0.16      # fast enough to reach near-zero between
                              # 4-on-the-floor kicks (0.47s apart)

# --- Silence gate (raw level vs slow-decaying peak of itself) ---------------
LEVEL_TAU = 0.4               # raw-level smoothing before the peak/ratio
PEAK_DECAY_TAU = 60.0         # peak forgets old loudness over ~1 min
GATE_LO = 0.05                # level <= 5% of peak  -> gate 0 (silence)
GATE_HI = 0.25                # level >= 25% of peak -> gate 1 (music)
GATE_TAU = 0.25               # gate smoothing

# --- Build detector ---------------------------------------------------------
BUILD_WINDOW_S = 4.0          # slope measured over this window
BUILD_SLOPE_GAIN = 8.0        # slope (energy units / s) -> 0..1

# --- Drop detector -----------------------------------------------------------
# A "quiet episode" (breakdown / silence) ARMS the detector for a window;
# the first bass slam while armed fires the drop. Quiet/slam are judged on
# the ABSOLUTE raw bass level vs its own slow-decaying peak - the AGC-
# normalized bass is useless mid-breakdown (tiny/tiny ~ noise), but the norm
# rising-cross is still required at slam time so a slow fade-in doesn't fire.
BASS_LEVEL_TAU = 0.1          # raw bass level smoothing (fast - a drop must
                              # register on the FIRST kick, not a second late)
BASS_PEAK_TAU = 60.0          # bass peak memory
DROP_QUIET_FRAC = 0.15        # bass below 15% of its peak counts as quiet...
DROP_QUIET_MIN_S = 1.5        # ...sustained at least this long to arm
DROP_ARM_WINDOW_S = 20.0      # armed for this long after quiet begins
DROP_SLAM_FRAC = 0.2          # RISING CROSS of bass above 20% of its peak,
DROP_SLAM_LEVEL = 1.25        # while norm bass is above this -> drop.
                              # (The edge must be on the absolute level: the
                              # AGC-normalized bass jitters above 1.25 all
                              # through a breakdown - tiny/tiny - so it has
                              # no clean edge at the actual return.)
DROP_REFRACTORY_S = 8.0       # min time between drops

# --- Mood classifier ---------------------------------------------------------
# Coarse music character for the director: what KIND of music is playing,
# not just how loud. Judged from percussive activity (time-average of the
# punch envelopes - a steady groove keeps them busy, ambient pads leave
# them at zero), overall energy, beat confidence, and the silence gate.
# Hysteresis: a candidate mood must persist before we switch, so a quiet
# bar can't flicker the classification.
PERC_TAU = 3.0                # leaky window for the hit counter
PERC_HIT_LEVEL = 0.55         # a punch rising past this counts as one hit
MOOD_AMBIENT_HITS = 0.8       # fewer than ~1 hit per 4s = no percussion
                              # (a steady 2 Hz kick holds the counter ~6)
MOOD_CHILL_BPM = 106.0        # slower than this (or soft) = chill
MOOD_CHILL_ENERGY = 0.30
MOOD_PEAK_ENERGY = 0.58
MOOD_PEAK_DROP_S = 20.0       # a drop counts as peak music for this long
                              # (drop_decay alone dies in ~0.4s - far shorter
                              # than the 4s mood dwell, so it never flipped
                              # the mood by itself)
MOOD_DWELL_S = 4.0            # candidate must persist this long
MOOD_SILENT_DWELL_S = 1.5

# --- Rhythm density ----------------------------------------------------------
# Punch hits per BEAT (not per second): ~0.5 = sparse half-time, ~1-1.5 =
# driving 4/4, 2+ = busy breaks. Lets the director prefer slow-motion rooms
# for sparse grooves and busy rooms for dense ones at the same energy.
DENSITY_TAU = 5.0
DENSITY_MAX = 3.0

FPS = 40.0                    # analysis frame rate assumed for buffer sizing


class AudioStructure:
    """Scalar band energies + build/drop tracking from the analyzer dict."""

    def __init__(self):
        self._bass = 0.0
        self._mid = 0.0
        self._high = 0.0
        self._bass_punch = 0.0
        self._mid_punch = 0.0
        self._high_punch = 0.0
        self._energy = 0.0
        self._gate = 0.0
        self._level = 0.0             # smoothed absolute raw level
        self._peak = 1e-9             # slow-decaying peak of the level
        # (t, energy) ring buffer for the build slope.
        self._energy_hist = deque(maxlen=int(BUILD_WINDOW_S * FPS) + 8)
        self._t = 0.0
        self._build = 0.0
        # Drop state machine.
        self._bass_level = 0.0        # smoothed absolute raw bass level
        self._bass_peak = 1e-9        # slow-decaying peak of it
        self._quiet_time = 0.0        # how long bass has been quiet
        self._armed_until = -1.0      # drop may fire until this time
        self._since_drop = 1e9
        self._prev_slam_ok = False    # last frame's absolute-slam condition
        self._drop_decay = 0.0
        # Mood classifier.
        self._hits = 0.0              # leaky count of punch TRANSIENTS -
                                      # pads park a punch at a low constant,
                                      # percussion crosses it repeatedly
        self._prev_punch = 0.0
        self._mood = "silent"
        self._mood_cand = "silent"
        self._mood_cand_t = 0.0
        self._density = 0.0           # punch hits per beat (rhythm busyness)

    # ------------------------------------------------------------------
    def update(self, sound, beat, dt):
        """Advance one frame. ``sound`` is outstate['sound'] (or None);
        ``beat`` is the BeatDetector result dict (used by the mood
        classifier for tempo/confidence)."""
        try:
            result = self._update(sound, dt)
            self._classify(beat or {}, dt)
            result["mood"] = self._mood
            result["perc"] = round(min(self._hits / 6.0, 1.0), 3)
            result["density"] = round(self._density, 2)
            return result
        except Exception:
            return self._result(False)

    def _classify(self, beat, dt):
        """Coarse music character: silent / ambient / chill / groove / peak."""
        punch = max(self._bass_punch, self._mid_punch, self._high_punch)
        self._hits *= math.exp(-dt / PERC_TAU)
        if punch >= PERC_HIT_LEVEL and self._prev_punch < PERC_HIT_LEVEL:
            self._hits += 1.0
        self._prev_punch = punch

        bpm = float(beat.get("bpm", 0.0) or 0.0)
        conf = float(beat.get("confidence", 0.0) or 0.0)

        # Rhythm density: hit rate normalized by tempo (hits per beat).
        da = 1.0 - math.exp(-dt / DENSITY_TAU)
        if bpm > 1.0 and self._hits > 0.3:
            inst = (self._hits / PERC_TAU) / (bpm / 60.0)
            self._density += (min(inst, DENSITY_MAX) - self._density) * da
        else:
            self._density += (0.0 - self._density) * da * 0.5

        if self._gate < 0.15:
            cand = "silent"
        elif self._hits < MOOD_AMBIENT_HITS:
            cand = "ambient"        # music present, but nothing percussive
        elif (self._energy > MOOD_PEAK_ENERGY
                or self._since_drop < MOOD_PEAK_DROP_S):
            cand = "peak"
        elif ((0.0 < bpm < MOOD_CHILL_BPM and conf > 0.25)
                or self._energy < MOOD_CHILL_ENERGY):
            cand = "chill"
        else:
            cand = "groove"

        if cand != self._mood_cand:
            self._mood_cand = cand
            self._mood_cand_t = self._t
        dwell = MOOD_SILENT_DWELL_S if cand == "silent" else MOOD_DWELL_S
        if cand != self._mood and self._t - self._mood_cand_t >= dwell:
            self._mood = cand

    def _result(self, drop):
        return {"bass": self._bass, "mid": self._mid, "high": self._high,
                "bass_punch": self._bass_punch, "mid_punch": self._mid_punch,
                "high_punch": self._high_punch,
                "punch": max(self._bass_punch, self._mid_punch, self._high_punch),
                "energy": self._energy, "build": self._build,
                "drop": drop, "drop_decay": self._drop_decay,
                "mood": self._mood,
                "perc": round(min(self._hits / 6.0, 1.0), 3),
                "density": round(self._density, 2)}

    def _update(self, sound, dt):
        dt = float(dt) if dt and dt > 0.0 else (1.0 / FPS)
        if dt > 0.5:
            dt = 0.5
        self._t += dt
        self._since_drop += dt
        self._drop_decay *= math.exp(-dt / DROP_DECAY_TAU)

        if sound is None:
            return self._idle(dt)
        short = sound.get("norm_short")
        long_ = sound.get("norm_long")
        raw = sound.get("raw_bands")
        if (short is None or len(short) == 0 or long_ is None
                or len(long_) == 0 or raw is None or len(raw) == 0):
            return self._idle(dt)

        srow = np.asarray(short[0], dtype=np.float32)
        lrow = np.asarray(long_[0], dtype=np.float32)
        rrow = np.asarray(raw[0], dtype=np.float32)

        # --- Silence gate: smoothed raw level vs its own slow-decaying peak --
        raw_level = float(np.mean(rrow))
        self._level += (raw_level - self._level) * (1.0 - math.exp(-dt / LEVEL_TAU))
        self._peak = max(self._peak * math.exp(-dt / PEAK_DECAY_TAU), self._level, 1e-9)
        ratio = self._level / self._peak
        gate_now = (ratio - GATE_LO) / (GATE_HI - GATE_LO)
        gate_now = max(0.0, min(1.0, gate_now))
        self._gate += (gate_now - self._gate) * (1.0 - math.exp(-dt / GATE_TAU))
        gate = self._gate

        # --- Scalar band energies (fast EMA of the 0.5s-normalized bands) ---
        # Each range is additionally gated by whether it holds REAL power:
        # AGC norms of near-empty bands are tiny/tiny noise ratios that fire
        # phantom punches (a pure pad "hitting" in the cymbal bands through
        # spectral leakage). A range must carry at least ~0.3% of the
        # dominant range's raw level to count at all.
        lv_b = float(np.sum(rrow[BASS_SLICE]))
        lv_m = float(np.sum(rrow[MID_SLICE]))
        lv_h = float(np.sum(rrow[HIGH_SLICE]))
        dom = max(lv_b, lv_m, lv_h, 1e-12)
        sig_b = min(1.0, lv_b / (0.003 * dom))
        sig_m = min(1.0, lv_m / (0.003 * dom))
        sig_h = min(1.0, lv_h / (0.003 * dom))
        a = 1.0 - math.exp(-dt / BAND_TAU)
        bass = gate * sig_b * float(np.clip(np.mean(srow[BASS_SLICE]), 0.0, 4.0))
        mid = gate * sig_m * float(np.clip(np.mean(srow[MID_SLICE]), 0.0, 4.0))
        high = gate * sig_h * float(np.clip(np.mean(srow[HIGH_SLICE]), 0.0, 4.0))
        self._bass += (bass - self._bass) * a
        self._mid += (mid - self._mid) * a
        self._high += (high - self._high) * a

        # --- Punch envelopes: instant attack on a hit, ~0.25s release -------
        rel = math.exp(-dt / PUNCH_RELEASE_TAU)
        for attr, raw_v in (("_bass_punch", bass), ("_mid_punch", mid),
                            ("_high_punch", high)):
            hit = max(0.0, min(1.0, (raw_v - PUNCH_FLOOR) / PUNCH_SPAN))
            setattr(self, attr, max(getattr(self, attr) * rel, hit))

        # --- Slow overall energy (2.5s baseline x absolute level) ----------
        # The AGC-normalized mean hovers at ~1.0 during ANY steady music, so
        # on its own it reads chill and peak both as ~0.5 - the club could
        # never tell them apart (user-visible: rooms didn't react to the
        # set's energy). Scale it by how loud the room actually is vs its
        # own recent peak (the gate's ratio, volume-calibrated by the 60s
        # peak decay): full-tilt music ~1.0, laid-back passages ~0.7,
        # breakdowns ~0.55 - honest dynamic range for the director's
        # percentile calibration to chew on.
        ae = 1.0 - math.exp(-dt / ENERGY_TAU)
        amp = 0.4 + 0.6 * min(ratio * 1.25, 1.0)
        energy_now = gate * amp * float(np.clip(np.mean(lrow) * 0.5, 0.0, 1.0))
        self._energy += (energy_now - self._energy) * ae
        self._energy_hist.append((self._t, self._energy))

        # --- Build: sustained energy climb x rising highs -------------------
        t0, e0 = self._energy_hist[0]
        span = self._t - t0
        slope = (self._energy - e0) / span if span > 1.0 else 0.0
        high_long = gate * float(np.clip(np.mean(lrow[HIGH_SLICE]), 0.0, 1.0))
        self._build = float(np.clip(slope * BUILD_SLOPE_GAIN, 0.0, 1.0) * high_long)

        # --- Drop: quiet episode arms, first bass slam while armed fires ----
        raw_bass = float(np.mean(rrow[BASS_SLICE]))
        ab = 1.0 - math.exp(-dt / BASS_LEVEL_TAU)
        self._bass_level += (raw_bass - self._bass_level) * ab
        self._bass_peak = max(self._bass_peak * math.exp(-dt / BASS_PEAK_TAU),
                              self._bass_level, 1e-9)
        bass_long = gate * float(np.mean(lrow[BASS_SLICE]))
        drop = False
        if self._bass_level < DROP_QUIET_FRAC * self._bass_peak:
            self._quiet_time += dt
            if self._quiet_time >= DROP_QUIET_MIN_S:
                self._armed_until = self._t + DROP_ARM_WINDOW_S
        else:
            self._quiet_time = 0.0
        slam_ok = self._bass_level >= DROP_SLAM_FRAC * self._bass_peak
        if (slam_ok and not self._prev_slam_ok
                and bass_long >= DROP_SLAM_LEVEL
                and self._t <= self._armed_until
                and self._since_drop >= DROP_REFRACTORY_S):
            drop = True
            self._since_drop = 0.0
            self._armed_until = -1.0
            self._drop_decay = 1.0
        self._prev_slam_ok = slam_ok

        return self._result(drop)

    def _idle(self, dt):
        """No usable audio data: bleed everything toward silence."""
        a = 1.0 - math.exp(-dt / BAND_TAU)
        self._bass += (0.0 - self._bass) * a
        self._mid += (0.0 - self._mid) * a
        self._high += (0.0 - self._high) * a
        rel = math.exp(-dt / PUNCH_RELEASE_TAU)
        self._bass_punch *= rel
        self._mid_punch *= rel
        self._high_punch *= rel
        ae = 1.0 - math.exp(-dt / ENERGY_TAU)
        self._energy += (0.0 - self._energy) * ae
        self._energy_hist.append((self._t, self._energy))
        self._build = 0.0
        self._gate = 0.0
        return self._result(False)


class HarmonicTracker:
    """Key-center tracking from the analyzer's 12-bin chroma, for palette
    steering. Pitch classes map through the CIRCLE OF FIFTHS onto a hue
    circle, so musically adjacent keys land on adjacent hues; the tracked
    center is the circular mean of the smoothed chroma on that circle.

    update(chroma, dt) -> {"center": 0..1 hue-like key position,
                           "strength": 0..1 how tonal the music is,
                           "changed": True once when the key visibly moves}
    Same contract as the other trackers: exception-wrapped, safe zeros.
    """

    # pitch class -> position on the circle of fifths (C G D A E B F# ...)
    _CIRCLE = np.exp(2j * np.pi * np.array([(pc * 7) % 12 for pc in range(12)]) / 12.0)

    def __init__(self):
        self._slow = 0j            # ~10s harmonic center (what we publish)
        self._fast = 0j            # ~2.5s center (detects key changes)
        self._diff_since = None
        self._t = 0.0
        self._last_change_t = -1e9

    def update(self, chroma, dt):
        try:
            return self._update(chroma, dt)
        except Exception:
            return {"center": 0.0, "strength": 0.0, "changed": False}

    def _update(self, chroma, dt):
        dt = float(dt) if dt and dt > 0.0 else 0.025
        self._t += dt
        if chroma is None or len(chroma) < 12 or float(np.sum(chroma)) < 0.05:
            # Atonal / silent: strength bleeds away, center holds.
            self._slow *= math.exp(-dt / 6.0)
            self._fast *= math.exp(-dt / 2.0)
            return self._pack(False)

        v = complex(np.sum(np.asarray(chroma[:12], dtype=np.float64) * self._CIRCLE))
        self._fast += (v - self._fast) * (1.0 - math.exp(-dt / 2.5))

        # Key change: the published center only pursues the fast one while
        # they roughly agree. When the fast center pulls a real distance
        # away (a single-fifth modulation moves 1/12 = 30deg of the circle)
        # the published center FREEZES, and if the gap persists it snaps
        # over and reports the change once. The published pursuit must be
        # STATELY (tau 20s) - a faster chase closes a single-fifth gap to
        # ~12deg before any usable threshold can catch it.
        changed = False
        d = 0.0
        if abs(self._slow) > 0.15 and abs(self._fast) > 0.15:
            d = abs(math.remainder(
                (np.angle(self._fast) - np.angle(self._slow)) / (2 * math.pi), 1.0))
        if d > 1.0 / 24.0:
            if self._diff_since is None:
                self._diff_since = self._t
            elif (self._t - self._diff_since > 2.5
                    and self._t - self._last_change_t > 25.0):
                self._slow = self._fast
                self._last_change_t = self._t
                self._diff_since = None
                changed = True
        else:
            self._diff_since = None
            self._slow += (self._fast - self._slow) * (1.0 - math.exp(-dt / 20.0))
        return self._pack(changed)

    def _pack(self, changed):
        strength = min(abs(self._slow), 1.0)
        center = float((np.angle(self._slow) / (2 * math.pi)) % 1.0)
        return {"center": center, "strength": round(strength, 3),
                "changed": changed}
