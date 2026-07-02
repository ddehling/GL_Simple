"""Offline test for the club-set rhythm/structure signals (no hardware).

Synthesizes a 128-BPM dance track in numpy (kick + hats + pad, with a
scripted 6s breakdown, a drop, then trailing silence), replicates
MicrophoneAnalyzer's exact per-frame pipeline (4096 Hann FFT -> RMS
log-bands -> 0.95/0.05 smoothing -> rolling norm_short/norm_long), and
feeds BeatDetector + AudioStructure each frame at 40 fps - exactly what
they see live. Then asserts the published signals behave as specified.

Track timeline (58 s):
    0 - 30 s   steady groove (kick on quarters, hats on 8ths, pad)
   30 - 36 s   breakdown (kick removed - bass goes quiet, hats/pad stay)
   36 s        bass slams back -> the one scripted DROP
   36 - 50 s   steady groove again
   50 - 58 s   digital silence

Usage: python tools/_club_signals_test.py
Exit code 0 + "ALL PASS" when every assertion holds.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.beat_detector import BeatDetector
from lib.audio_signals import AudioStructure

RATE = 44100
FPS = 40
CHUNK = 4096
HOP = RATE // FPS
WIN_SHORT = 20                 # analyzer defaults (frames)
WIN_LONG = 100
NUM_BANDS = 32

BPM = 128.0
BEAT_S = 60.0 / BPM
GROOVE1_END = 30.0
BREAK_END = 36.0               # drop lands here
GROOVE2_END = 50.0
TRACK_END = 58.0

# Analyzer's log band edges (40 Hz .. 16 kHz) and masks.
freq_bins = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
band_edges = np.logspace(np.log10(40), np.log10(16000), NUM_BANDS + 1)
band_masks = []
for i in range(NUM_BANDS):
    m = (freq_bins >= band_edges[i]) & (freq_bins < band_edges[i + 1])
    if not np.any(m):
        m = (freq_bins >= band_edges[i] * 0.9) & (freq_bins < band_edges[i + 1] * 1.1)
    band_masks.append(m)
hann = np.hanning(CHUNK)


def synth_track():
    n = int(TRACK_END * RATE)
    x = np.zeros(n, dtype=np.float64)
    rng = np.random.RandomState(12345)   # local generator (house rule)

    def add_kick(t0):
        i0 = int(t0 * RATE)
        dur = int(0.10 * RATE)
        if i0 + dur > n:
            return
        tl = np.arange(dur) / RATE
        x[i0:i0 + dur] += 0.9 * np.sin(2 * np.pi * 55.0 * tl) * np.exp(-tl / 0.04)

    def add_hat(t0):
        i0 = int(t0 * RATE)
        dur = int(0.03 * RATE)
        if i0 + dur > n:
            return
        burst = rng.randn(dur) * np.exp(-np.arange(dur) / (0.008 * RATE))
        x[i0:i0 + dur] += 0.10 * np.diff(np.concatenate([[0.0], burst]))  # HPF-ish

    # Kicks on quarters, except during the breakdown.
    t = 0.0
    while t < GROOVE2_END:
        in_break = GROOVE1_END <= t < BREAK_END
        if not in_break:
            add_kick(t)
        t += BEAT_S
    # Hats on 8ths through everything non-silent.
    t = 0.0
    while t < GROOVE2_END:
        add_hat(t)
        t += BEAT_S / 2.0
    # Pad (mid content) through all non-silent sections.
    tt = np.arange(int(GROOVE2_END * RATE)) / RATE
    x[:len(tt)] += 0.03 * np.sin(2 * np.pi * 440.0 * tt) + 0.02 * np.sin(2 * np.pi * 660.0 * tt)
    return x


def band_powers_for_window(win):
    mags = np.abs(np.fft.rfft(win * hann))
    mags[0] = 0.0
    bp = np.zeros(NUM_BANDS, dtype=np.float64)
    for i, m in enumerate(band_masks):
        if np.any(m):
            bp[i] = np.sqrt(np.mean(mags[m] ** 2))
    return bp


def run():
    samples = synth_track()
    bd = BeatDetector()
    st = AudioStructure()
    dt = 1.0 / FPS

    prev_bp = None
    recent = []                    # newest-first rolling raw frames
    log = []                       # per-frame dicts for assertions
    drops = []
    t = 0.0
    pos = 0
    while pos + CHUNK <= len(samples):
        win = samples[pos:pos + CHUNK]
        bp = band_powers_for_window(win)
        if prev_bp is None:
            prev_bp = bp.copy()
        else:
            bp = 0.95 * bp + 0.05 * prev_bp
            prev_bp = bp.copy()

        recent.insert(0, bp)
        if len(recent) > WIN_LONG:
            recent.pop()
        mean_short = np.mean(recent[:WIN_SHORT], axis=0)
        mean_long = np.mean(recent, axis=0)
        mean_short = np.where(mean_short < 1e-10, 1e-10, mean_short)
        mean_long = np.where(mean_long < 1e-10, 1e-10, mean_long)
        sound = {
            "raw_bands": np.array([bp]),
            "norm_short": np.array([bp / mean_short]),
            "norm_long": np.array([bp / mean_long]),
            "norm_long_relu": np.array([np.maximum(0.0, bp / mean_long - 1.0)]),
        }

        beat = bd.update(sound, dt)
        sig = st.update(sound, beat, dt)
        if sig["drop"]:
            drops.append(t)
        log.append(dict(t=t, bpm=beat["bpm"], conf=beat["confidence"],
                        count=beat["count"], bar=beat["bar_phase"],
                        phrase=beat["phrase_phase"], onset=beat["onset"],
                        bass=sig["bass"], energy=sig["energy"],
                        build=sig["build"], bass_punch=sig["bass_punch"],
                        high_punch=sig["high_punch"]))
        t += dt
        pos += HOP

    return log, drops


def at(log, t_target):
    return min(log, key=lambda r: abs(r["t"] - t_target))


def main():
    log, drops = run()
    failures = []

    def check(name, cond, detail):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if not cond:
            failures.append(name)

    print("Club signals offline test (synthetic 128 BPM track)\n")

    r25 = at(log, 25.0)
    check("bpm lock", abs(r25["bpm"] - BPM) / BPM < 0.03,
          f"bpm={r25['bpm']:.1f} at t=25s (target {BPM:.0f} +/- 3%)")

    check("confidence locked", r25["conf"] > 0.5,
          f"confidence={r25['conf']:.2f} at t=25s (want > 0.5)")

    # bar_phase periodicity: wraps every 4 beats -> 2 or 3 wraps in 20..25s.
    seg = [r for r in log if 20.0 <= r["t"] <= 25.0]
    wraps = sum(1 for a, b in zip(seg, seg[1:]) if b["bar"] < a["bar"] - 0.5)
    in_range = all(0.0 <= r["bar"] < 1.0 and 0.0 <= r["phrase"] < 1.0 for r in log)
    check("bar_phase periodic", wraps in (2, 3) and in_range,
          f"{wraps} bar wraps in 20-25s (expect 2-3), phases in [0,1)={in_range}")

    beats_5s = at(log, 25.0)["count"] - at(log, 20.0)["count"]
    check("beat_count rate", 9 <= beats_5s <= 12,
          f"{beats_5s} beats counted in 20-25s (128 BPM -> ~10.7)")

    check("one drop", len(drops) == 1,
          f"drops fired at {['%.1f' % d for d in drops]} (expect exactly 1)")
    if drops:
        check("drop timing", abs(drops[0] - BREAK_END) <= 1.0,
              f"drop at t={drops[0]:.1f}s (scripted return at {BREAK_END:.0f}s +/- 1s)")
    steady_drops = [d for d in drops if 5.0 <= d <= 28.0]
    check("no drop during steady groove", len(steady_drops) == 0,
          f"drops in 5-28s: {steady_drops}")

    check("energy during groove", r25["energy"] > 0.2,
          f"audio_energy={r25['energy']:.2f} at t=25s (want > 0.2)")

    # Punch envelopes must actually PULSE during a steady groove - high on
    # kicks, near zero between them - or visuals have no temporal contrast.
    seg = [r for r in log if 20.0 <= r["t"] <= 25.0]
    bp_max = max(r["bass_punch"] for r in seg)
    bp_min = min(r["bass_punch"] for r in seg)
    check("bass_punch pulses", bp_max > 0.6 and bp_min < 0.2,
          f"bass_punch range {bp_min:.2f}..{bp_max:.2f} in 20-25s "
          f"(want max>0.6, min<0.2)")
    sil = [r for r in log if 55.0 <= r["t"] <= 57.5]
    check("punch silent in silence", max(r["bass_punch"] for r in sil) < 0.05
          and max(r["high_punch"] for r in sil) < 0.05,
          f"max punches in silence: bass={max(r['bass_punch'] for r in sil):.2f} "
          f"high={max(r['high_punch'] for r in sil):.2f}")

    r57 = at(log, 57.5)
    check("confidence dies in silence", r57["conf"] < 0.15,
          f"confidence={r57['conf']:.2f} at t=57.5s, 7.5s into silence (want < 0.15)")
    check("energy dies in silence", r57["energy"] < 0.05,
          f"audio_energy={r57['energy']:.2f} at t=57.5s (want < 0.05)")

    # Informational timeline
    print("\n  t     bpm   conf  energy  bass  build")
    for tt in (5, 15, 25, 32, 35, 37, 45, 52, 57):
        r = at(log, float(tt))
        print(f"  {r['t']:4.0f} {r['bpm']:6.1f}  {r['conf']:.2f}   {r['energy']:.2f}  "
              f"{r['bass']:.2f}   {r['build']:.2f}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
