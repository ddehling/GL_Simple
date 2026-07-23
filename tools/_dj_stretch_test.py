"""Phase B gate #1: the WSOLA stretcher moves tempo exactly as commanded.

Synthesizes a 128-BPM groove, plays it through a real Deck at rates
0.92 / 0.96 / 1.0 / 1.04 / 1.08, and runs OUR OWN live BeatDetector pipeline
(tools/_club_signals_test.run) on the output: detected BPM must equal
128 x rate, confidence must survive stretching, onsets must not smear away,
output must be block-size invariant, and CPU must stay far below realtime.

Usage:
    python tools/_dj_stretch_test.py           # ALL PASS gate
    python tools/_dj_stretch_test.py --wav     # also write ear-check WAVs
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.dj.deck import Deck

RATE = 44100
BPM = 128.0
BEAT = 60.0 / BPM
DUR = 50.0

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def synth_groove(seconds=DUR):
    n = int(seconds * RATE)
    x = np.zeros(n)
    rng = np.random.RandomState(3)
    t = 0.0
    while t < seconds:
        i0 = int(t * RATE)
        d = int(0.10 * RATE)
        if i0 + d <= n:
            tl = np.arange(d) / RATE
            x[i0:i0 + d] += 0.9 * np.sin(2 * np.pi * 54 * tl) * np.exp(-tl / 0.045)
        t += BEAT
    t = 0.0
    while t < seconds:
        i0 = int(t * RATE)
        d = int(0.03 * RATE)
        if i0 + d <= n:
            b = rng.randn(d) * np.exp(-np.arange(d) / (0.008 * RATE))
            x[i0:i0 + d] += 0.10 * np.diff(np.concatenate([[0.0], b]))
        t += BEAT / 2
    tt = np.arange(n) / RATE
    x += 0.04 * np.sin(2 * np.pi * 220 * tt) + 0.02 * np.sin(2 * np.pi * 330 * tt)
    x = (x / np.max(np.abs(x)) * 0.8).astype(np.float32)
    return np.stack([x, x], axis=1)


GRID = [{"start_s": 0.0, "end_s": DUR, "period_s": BEAT, "first_beat_s": 0.0,
         "bpm": BPM}]


def render(samples, rate, out_seconds, block=1024):
    deck = Deck("t")
    deck.load(samples, grid=GRID)
    deck.gain = 1.0
    deck.set_rate(rate)
    deck.cue(0.0)
    deck.start()
    n_out = int(out_seconds * RATE)
    chunks = []
    t0 = time.perf_counter()
    done = 0
    while done < n_out:
        m = min(block, n_out - done)
        chunks.append(deck.read(m))
        done += m
    elapsed = time.perf_counter() - t0
    return np.concatenate(chunks, axis=0), elapsed


def detect(mono):
    """BPM from the offline grid estimator (ms precision - the live
    BeatDetector quantizes to integer 40fps lags, +/-2.5%, and would blur
    exactly what this test measures); confidence from the live pipeline;
    transient count straight off the onset envelope, detector-free."""
    from lib.dj import features as F
    from tools import _club_signals_test as CS
    bands, _ = F.frame_track(mono.astype(np.float32))
    ob, obass, operc, nov = F._onset_channels(bands)
    grid, bpm, gconf, beats = F.estimate_beat_grid(ob + 0.5 * operc)
    onset = ob + 0.5 * operc
    from scipy.signal import find_peaks
    idx, _ = find_peaks(onset, height=0.5 * np.percentile(onset, 90),
                        distance=3)      # 75ms dedup of envelope jitter
    peaks = len(idx)
    log, drops = CS.run(mono.astype(np.float64))
    confs = np.array([r["conf"] for r in log if r["t"] > 15.0])
    return {"bpm": bpm, "grid_conf": gconf,
            "conf": float(np.mean(confs)), "onsets": peaks,
            "drops": len(drops)}


def main():
    write_wav = "--wav" in sys.argv
    samples = synth_groove()
    print("WSOLA stretch test (128-BPM groove through a real Deck)\n")

    base = None
    results = {}
    for rate in (1.0, 0.92, 0.96, 1.04, 1.08):
        out_seconds = min((DUR - 2.0) / rate, 45.0)
        out, elapsed = render(samples, rate, out_seconds)
        mono = out.mean(axis=1)
        r = detect(mono)
        r["elapsed"] = elapsed
        r["out_seconds"] = out_seconds
        results[rate] = r
        if rate == 1.0:
            base = r
        want = BPM * rate
        check(f"rate {rate:.2f} bpm", abs(r["bpm"] - want) / want < 0.01,
              f"detected {r['bpm']:.2f} (want {want:.2f} +/- 1%), "
              f"conf {r['conf']:.2f}, {elapsed:.2f}s for {out_seconds:.0f}s audio")
        if write_wav:
            from scipy.io import wavfile
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "logs", f"dj_stretch_{rate:.2f}.wav")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            wavfile.write(p, RATE, (out * 32767).astype(np.int16))
            print(f"        wrote {os.path.normpath(p)}")

    print()
    for rate, r in results.items():
        if rate == 1.0:
            continue
        check(f"rate {rate:.2f} confidence survives",
              r["conf"] >= 0.9 * base["conf"],
              f"conf {r['conf']:.2f} vs baseline {base['conf']:.2f}")
        # Onset RATE per output second must match the baseline's scaled by
        # the tempo ratio (transients preserved, none smeared away/doubled).
        r_rate = r["onsets"] / r["out_seconds"]
        b_rate = base["onsets"] / base["out_seconds"] * rate
        # WSOLA legitimately duplicates/skips the odd chunk at ratio
        # extremes (why the brain prefers <=5% stretches); 12% bounds real
        # damage (smearing halves the count, doubling doubles it).
        # Rubber Band R3 ("finer", the default variant): +22% at the 8%-
        # slowdown extreme only. DELIBERATE trade (2026-07-22): R2 passed
        # 12% everywhere but WARBLED on sustained tones (user-heard, the
        # artifact keylock exists to avoid); R3 is warble-free at real-
        # world rates and 8% slowdowns are rare since the dual-bend cap
        # (6%/deck). DJ_RB_ENGINE=faster re-selects R2 (crisp) for A/B.
        tol = 0.12
        from lib.dj import stretch_engine_name
        if stretch_engine_name() == "rubberband" and rate <= 0.94 \
                and os.environ.get("DJ_RB_ENGINE",
                                   "finer").lower() != "faster":
            tol = 0.22
        check(f"rate {rate:.2f} onsets preserved",
              abs(r_rate - b_rate) / b_rate < tol,
              f"{r_rate:.2f} onsets/s vs expected {b_rate:.2f} "
              f"(+/- {tol * 100:.0f}%)")

    # Bypass really is bit-exact at rate 1.0.
    out1, _ = render(samples, 1.0, 20.0)
    check("rate 1.0 is bit-exact", np.array_equal(out1, samples[:len(out1)]),
          f"max deviation {np.max(np.abs(out1 - samples[:len(out1)])):.2e}")

    # Block-size invariance: the FIFO must make output independent of how
    # callers slice their reads.
    a, _ = render(samples, 1.04, 12.0, block=512)
    b, _ = render(samples, 1.04, 12.0, block=4410)
    check("block-size invariant", np.allclose(a, b, atol=1e-6),
          f"max diff {np.max(np.abs(a - b)):.2e} between 512 and 4410 reads")

    # CPU: stretching must cost well under 5% of realtime per deck.
    # Rubber Band R3 runs ~7% - acceptable for the opt-in premium engine
    # (decks bypass at rate 1.0, so this only applies while stretching).
    from lib.dj import stretch_engine_name as _sen
    budget = 0.10 if _sen() == "rubberband" else 0.05
    frac = results[1.04]["elapsed"] / results[1.04]["out_seconds"]
    check("cpu budget", frac < budget,
          f"{frac * 100:.1f}% of realtime at rate 1.04 "
          f"(want < {budget * 100:.0f}%)")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
