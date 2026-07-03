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


def run(samples=None):
    if samples is None:
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
                        high_punch=sig["high_punch"],
                        mood=sig["mood"], density=sig["density"],
                        decay_amp=beat["decay"] if beat["onset"] else 0.0))
        t += dt
        pos += HOP

    return log, drops


def at(log, t_target):
    return min(log, key=lambda r: abs(r["t"] - t_target))


def finite(log):
    return all(np.isfinite([r["bpm"], r["conf"], r["energy"], r["bass"],
                            r["build"], r["bass_punch"], r["high_punch"]]).all()
               for r in log)


def stress_suites(check):
    """Adversarial input suites: the pipeline must stay sane (finite, no
    drop storms, beat lock survives) under real-world input damage."""

    # --- Clipping: a preamp slammed into the rails -------------------------
    clipped = np.clip(synth_track() * 6.0, -1.0, 1.0)
    log, drops = run(clipped)
    r = at(log, 25.0)
    check("clip: signals finite", finite(log), "no NaN/inf anywhere")
    check("clip: bpm survives", abs(r["bpm"] - BPM) / BPM < 0.05,
          f"bpm={r['bpm']:.1f} on hard-clipped input (want {BPM:.0f} +/- 5%)")
    seg = [x for x in log if 20.0 <= x["t"] <= 25.0]
    check("clip: punches still pulse", max(x["bass_punch"] for x in seg) > 0.5,
          f"max bass_punch={max(x['bass_punch'] for x in seg):.2f}")
    check("clip: no drop storm", len(drops) <= 2, f"{len(drops)} drops (allow <= 2)")

    # --- Dropouts: 40ms of zeros every 2s (flaky USB / bluetooth) ----------
    holey = synth_track()
    hole = int(0.040 * RATE)
    for t0 in np.arange(2.0, GROOVE2_END, 2.0):
        i = int(t0 * RATE)
        holey[i:i + hole] = 0.0
    log, drops = run(holey)
    r = at(log, 25.0)
    check("dropout: signals finite", finite(log), "no NaN/inf anywhere")
    check("dropout: bpm survives", abs(r["bpm"] - BPM) / BPM < 0.05,
          f"bpm={r['bpm']:.1f} with 40ms holes every 2s")
    steady = [d for d in drops if 5.0 <= d <= 28.0]
    check("dropout: holes aren't drops", len(steady) == 0,
          f"drops during steady groove: {steady}")

    # --- Cold start: long silence, then music (AGC wind-up) ----------------
    groove = synth_track()[:int(20.0 * RATE)]
    cold = np.concatenate([np.zeros(int(20.0 * RATE)), groove])
    log, drops = run(cold)
    check("cold start: signals finite", finite(log), "no NaN/inf anywhere")
    seg = [x for x in log if 20.0 <= x["t"] <= 23.0]
    check("cold start: punches bounded", max(x["bass_punch"] for x in seg) <= 1.05,
          f"max bass_punch={max(x['bass_punch'] for x in seg):.2f} in first 3s "
          f"of music (AGC wind-up must not explode)")
    early = [d for d in drops if d < 22.0]
    check("cold start: at most one wake-up drop", len(early) <= 1,
          f"drops near music start: {early}")
    r = at(log, 38.0)
    check("cold start: lock recovers", r["conf"] > 0.4 and abs(r["bpm"] - BPM) / BPM < 0.05,
          f"18s after music starts: bpm={r['bpm']:.1f} conf={r['conf']:.2f}")

    # --- Track change: the DJ mixes 128 -> 100 BPM ---------------------------
    def groove_at(bpm, seconds):
        n2 = int(seconds * RATE)
        x2 = np.zeros(n2)
        rng2 = np.random.RandomState(7)
        beat2 = 60.0 / bpm
        tg = 0.0
        while tg < seconds:
            i0 = int(tg * RATE); dur = int(0.10 * RATE)
            if i0 + dur <= n2:
                tl = np.arange(dur) / RATE
                x2[i0:i0+dur] += 0.9 * np.sin(2*np.pi*55.0*tl) * np.exp(-tl/0.04)
            tg += beat2
        tg = 0.0
        while tg < seconds:
            i0 = int(tg * RATE); dur = int(0.03 * RATE)
            if i0 + dur <= n2:
                burst = rng2.randn(dur) * np.exp(-np.arange(dur)/(0.008*RATE))
                x2[i0:i0+dur] += 0.10 * np.diff(np.concatenate([[0.0], burst]))
            tg += beat2 / 2
        tt2 = np.arange(n2) / RATE
        return x2 + 0.03*np.sin(2*np.pi*440*tt2)
    mix = np.concatenate([groove_at(128.0, 40.0),
                          np.zeros(int(1.5 * RATE)),
                          groove_at(100.0, 45.0)])
    log, _ = run(mix)
    r0 = at(log, 38.0)
    check("track change: first tempo locked", abs(r0["bpm"] - 128.0) / 128.0 < 0.05,
          f"bpm={r0['bpm']:.1f} before the change")
    r1 = at(log, 55.0)
    check("track change: re-locks new tempo", abs(r1["bpm"] - 100.0) / 100.0 < 0.05,
          f"bpm={r1['bpm']:.1f} 13.5s after the change (want ~100)")
    r2 = at(log, 62.0)
    check("track change: confidence recovers", r2["conf"] > 0.4,
          f"conf={r2['conf']:.2f} 20s after the change")

    # --- DC offset: a miswired line input ----------------------------------
    log, drops = run(synth_track() + 0.3)
    r = at(log, 25.0)
    check("dc offset: signals finite", finite(log), "no NaN/inf anywhere")
    check("dc offset: bpm survives", abs(r["bpm"] - BPM) / BPM < 0.05,
          f"bpm={r['bpm']:.1f} with +0.3 DC (want {BPM:.0f} +/- 5%)")


def wav_report(path):
    """Real-track mode: run the full pipeline over a WAV file and print the
    signal timeline - no assertions, just evidence for human judgment.

    Usage: python tools/_club_signals_test.py track.wav
    """
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    if sr != RATE:
        from math import gcd
        from scipy import signal as sps
        g = gcd(RATE, int(sr))
        data = sps.resample_poly(data, RATE // g, int(sr) // g)
    dur = len(data) / RATE
    print(f"Real-track report: {os.path.basename(path)} ({dur:.0f}s @ {sr} Hz)\n")

    log, drops = run(data)
    print(f"  drops fired at: {['%.1f' % d for d in drops] or 'none'}")
    confs = [r["conf"] for r in log if r["t"] > 15.0]
    bpms = [r["bpm"] for r in log if r["t"] > 15.0 and r["conf"] > 0.4]
    print(f"  confidence after 15s: mean {np.mean(confs):.2f}, "
          f"below 0.3 for {np.mean(np.array(confs) < 0.3) * 100:.0f}% of the track")
    if bpms:
        print(f"  bpm while confident: median {np.median(bpms):.1f} "
              f"(p10 {np.percentile(bpms, 10):.1f} / p90 {np.percentile(bpms, 90):.1f})")
    print(f"  signals finite: {finite(log)}")
    print("\n  t     bpm   conf  energy  bass  build  bass_punch")
    step = max(5, int(dur // 20 / 5) * 5)
    for tt in range(5, int(dur), step):
        r = at(log, float(tt))
        print(f"  {r['t']:4.0f} {r['bpm']:6.1f}  {r['conf']:.2f}   {r['energy']:.2f}  "
              f"{r['bass']:.2f}   {r['build']:.2f}   {r['bass_punch']:.2f}")


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

    # Beats must be GROUNDED in audio: none in silence (the oscillator is a
    # prediction, not a metronome), gentler flashes through the kickless
    # breakdown than in the full groove.
    # Beats taper out of silence: a few gentle coasting pulses in the first
    # seconds (confidence dying), then full stop - never a light show.
    c52 = at(log, 52.0)["count"]
    c55 = at(log, 55.0)["count"]
    c57 = at(log, 57.5)["count"]
    check("beats stop in silence", c57 - c55 == 0 and c57 - c52 <= 8,
          f"{c57 - c52} beats 2-7.5s in, {c57 - c55} beats 5-7.5s in (want 0)")
    tail = [r["decay_amp"] for r in log if 52.5 <= r["t"] <= 55.0 and r["decay_amp"] > 0]
    check("silence-coast flashes are gentle",
          (max(tail) < 0.7) if tail else True,
          f"max coasting flash={max(tail):.2f}" if tail else "no coasting beats")
    groove_flash = max(r["decay_amp"] for r in log if 20.0 <= r["t"] <= 25.0)
    check("groove flashes at full strength", groove_flash > 0.9,
          f"groove flash={groove_flash:.2f} (real kicks = full flash)")

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

    print("\nMood classifier:\n")
    check("groove reads as music", at(log, 25.0)["mood"] in ("groove", "peak", "chill"),
          f"mood={at(log, 25.0)['mood']} at t=25s (steady groove)")
    check("silence reads as silent", at(log, 57.0)["mood"] == "silent",
          f"mood={at(log, 57.0)['mood']} at t=57s (7s into silence)")
    check("drop forces peak", at(log, 45.0)["mood"] == "peak",
          f"mood={at(log, 45.0)['mood']} at t=45s (9s after the drop)")
    # Pad-only synthetic: sustained tones, zero percussion -> ambient.
    n_pad = int(40.0 * RATE)
    tt = np.arange(n_pad) / RATE
    pad = (0.20 * np.sin(2 * np.pi * 220.0 * tt) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.1 * tt))
           + 0.12 * np.sin(2 * np.pi * 331.0 * tt))
    log_a, _ = run(pad)
    r_a = at(log_a, 30.0)
    check("pads read as ambient", r_a["mood"] == "ambient",
          f"mood={r_a['mood']} at t=30s of beatless pads")
    check("groove rhythm density sane", 0.4 <= at(log, 25.0)["density"] <= 3.0,
          f"density={at(log, 25.0)['density']} hits/beat at t=25s (kick+hats 4/4)")
    check("pads have no rhythm density", r_a["density"] < 0.3,
          f"density={r_a['density']} on beatless pads")

    # Harmonic tracker: key center stabilizes, then a key change fires once.
    from lib.audio_signals import HarmonicTracker
    ht = HarmonicTracker()
    a_maj = np.zeros(12); a_maj[[9, 1, 4]] = [0.5, 0.25, 0.25]   # A C# E
    d_maj = np.zeros(12); d_maj[[2, 6, 9]] = [0.5, 0.25, 0.25]   # D F# A
    for _ in range(int(30 * 40)):
        r1 = ht.update(a_maj, 1 / 40.0)
    c1 = r1["center"]
    changed = []
    for k in range(int(20 * 40)):
        r2 = ht.update(d_maj, 1 / 40.0)
        if r2["changed"]:
            changed.append(k / 40.0)
    check("key center is tonal", r1["strength"] > 0.4,
          f"strength={r1['strength']} after 30s of A major")
    check("key change fires once", len(changed) == 1,
          f"changed at {changed} (expect exactly one)")
    d_center = abs((r2["center"] - c1 + 0.5) % 1.0 - 0.5)
    check("key center moved", d_center > 0.05,
          f"center {c1:.2f} -> {r2['center']:.2f}")


    print("\nAdversarial input suites (clip / dropout / cold start / DC):\n")
    stress_suites(check)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        wav_report(sys.argv[1])
    else:
        main()
