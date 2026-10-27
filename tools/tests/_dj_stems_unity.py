"""STEMS UNITY - the Phase 2 precondition of docs/DJ_RESCOPE.md: do the four stems sum back to the mix?

For N random library records with stems: decode the mix, load the four stems, sum them in float32 and over the
BODY (sections of kind intro/outro skipped; with no sections 10 % to 90 % of the duration) compare the RMS of the
stem sum against the mix in dB, and the RMS of the residual (mix - sum) relative to the mix in dB.
A record passes when the level difference is within 0.5 dB.

    python tools/tests/_dj_stems_unity.py --music D:/Devel/music [--n 12] [--seed 1]
"""
import gc
import os
import random
import sys

import numpy as np

os.environ["DJ_NO_PLAY_LOG"] = "1"
os.environ["DJ_NO_MIDI"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.tests._low_priority import PRIORITY  # noqa: E402,F401  (idle priority)

TOL_DB = 0.5
RATE = 44100


def arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def body_ranges(t, n_samples):
    """sample ranges of the body: every section not intro/outro; else 10 % to 90 %."""
    dur = n_samples / RATE
    rng = []
    for s in (getattr(t, "sections", None) or []):
        if s.get("kind") in ("intro", "outro"):
            continue
        a, z = max(0, int(float(s["start_s"]) * RATE)), min(n_samples, int(float(s["end_s"]) * RATE))
        if z > a:
            rng.append((a, z))
    if not rng:
        rng = [(int(0.1 * dur * RATE), int(0.9 * dur * RATE))]
    return rng


def db(x):
    return 20.0 * np.log10(max(float(x), 1e-12))


def measure(t, music, dbh):
    from lib.dj.features import decode_file_stereo
    from lib.dj.stems import load_stems
    samples = decode_file_stereo(dbh.abs(t.path))
    samples = np.asarray(samples, dtype=np.float32)
    stems = load_stems(music, t.id, expected_len=len(samples))
    if not stems:
        return None
    n = min(len(samples), min(len(v) for v in stems.values()))
    ranges = body_ranges(t, n)
    # the stems' offset against the mix (a constant delay shows as a huge residual with a fine level): FFT cross-correlation on 10 s
    a0, z0 = ranges[0][0], min(n, ranges[0][0] + RATE * 10)
    x = samples[a0:z0, 0].astype(np.float64)
    y = sum(v[a0:z0, 0].astype(np.float64) for v in stems.values())
    c = np.fft.irfft(np.fft.rfft(x) * np.conj(np.fft.rfft(y)))
    k = int(np.argmax(c))
    lag = (k - len(x) if k > len(x) // 2 else k)        # x[n] ~ y[n - lag]
    L = -lag if abs(lag) < 8192 else 0                    # stems delayed by L samples
    e_mix = e_sum = e_res = e_ral = 0.0
    cnt = 0
    for a, z in ranges:
        z = min(z, n)
        if z <= a:
            continue
        # chunked so no second 300 MB array is built
        for i in range(a, z, RATE * 20):
            j = min(z, i + RATE * 20)
            mix = samples[i:j]
            tot = np.zeros_like(mix)
            for v in stems.values():
                tot += v[i:j].astype(np.float32)
            res = mix - tot
            e_mix += float(np.sum(mix.astype(np.float64) ** 2))
            e_sum += float(np.sum(tot.astype(np.float64) ** 2))
            e_res += float(np.sum(res.astype(np.float64) ** 2))
            if L and 0 <= i + L and j + L <= n:
                tl = np.zeros_like(mix)
                for v in stems.values():
                    tl += v[i + L:j + L].astype(np.float32)
                e_ral += float(np.sum((mix - tl).astype(np.float64) ** 2))
            else:
                e_ral += float(np.sum(res.astype(np.float64) ** 2))
            cnt += mix.size
    del samples, stems
    gc.collect()
    if cnt == 0 or e_mix <= 0:
        return None
    rms_mix, rms_sum, rms_res, rms_ral = (e_mix / cnt) ** 0.5, (e_sum / cnt) ** 0.5, (e_res / cnt) ** 0.5, (e_ral / cnt) ** 0.5
    return db(rms_sum) - db(rms_mix), db(rms_res) - db(rms_mix), db(rms_mix), (db(rms_ral) - db(rms_mix), L)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    music = arg("--music", "D:/Devel/music")
    n = arg("--n", 12, int)
    seed = arg("--seed", 1, int)
    from lib.dj.db import LibraryDB
    from lib.dj import brain as B
    dbh = LibraryDB(music)
    lib = [t for t in B.load_library(dbh) if getattr(t, "has_stems", False) and not t.excluded]
    if not lib:
        print("no records with stems in", music)
        return 2
    picks = random.Random(seed).sample(lib, min(n, len(lib)))
    print(f"{len(lib)} records with stems; checking {len(picks)} (seed {seed})")
    print(f"{'title':34s} {'level diff dB':>14s} {'residual dB':>12s} {'aligned res dB':>15s} {'lag smp':>8s} {'mix dBFS':>9s}  result")
    rows = []
    for t in picks:
        title = f"{t.title[:30]}"
        try:
            r = measure(t, music, dbh)
        except Exception as e:  # noqa: BLE001
            print(f"{title:34s} {'-':>14s} {'-':>12s} {'-':>15s} {'-':>8s} {'-':>9s}  FAIL  {type(e).__name__}: {e}", flush=True)
            rows.append(None)
            continue
        if r is None:
            print(f"{title:34s} {'-':>14s} {'-':>12s} {'-':>15s} {'-':>8s} {'-':>9s}  FAIL  no stems / silent body", flush=True)
            rows.append(None)
            continue
        ld, rs, mx, (ral, L) = r
        ok = abs(ld) <= TOL_DB
        rows.append((ld, rs, ok, ral))
        print(f"{title:34s} {ld:+14.2f} {rs:12.1f} {ral:15.1f} {L:8d} {mx:9.1f}  {'PASS' if ok else 'FAIL'}", flush=True)
    good = [r for r in rows if r]
    bad = len(rows) - sum(1 for r in good if r[2])
    if good:
        worst = max(good, key=lambda r: abs(r[0]))[0]
        med = float(np.median([r[1] for r in good]))
        print(f"\nworst level diff {worst:+.2f} dB · median residual {med:.1f} dB, aligned {float(np.median([r[3] for r in good])):.1f} dB (relative to the mix) · tolerance {TOL_DB} dB")
    print("ALL OK" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code or 0)
