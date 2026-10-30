"""Key-reader truth gate (key v2, 2026-10-28).

Scores lib/dj/features.estimate_key_audio against published keys
(tools/tests/data/dj_key_truth.json: Beatport / Tunebat / Mixgraph lookups
for real library tracks) with the MIREX weighting - exact 1.0, fifth 0.5,
relative 0.3, parallel 0.2 - and fails below thresholds set a little under
the measured numbers (holdout n=40: exact 0.825 / MIREX 0.863; all 72:
0.847 / 0.889; the old frame_track reader was 0.31 / 0.39). Also checks the
synthetic anchors (A440 -> A-origin bin 0, A major chord -> 11B, silence).

Tracks resolve by PATH in this machine's library DB (opened read-only), so
the gate runs on any machine holding the same files. Audio comes from a
key-path cache of 11025 Hz mono .npy files named <track id>.npy when
--cache is given and the file exists, else from decode_file (~1 s/track).

Usage:
    python tools/tests/_dj_key_truth.py [--music D:/Devel/music] [--cache DIR] [-v]
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
           "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("DJ_NO_PLAY_LOG", "1")
os.environ.setdefault("DJ_NO_MIDI", "1")

import argparse
import json
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
TRUTH = os.path.join(ROOT, "tools", "tests", "data", "dj_key_truth.json")

# measured minus a small margin (decode vs cache, a relabelled truth entry)
MIN_HOLDOUT_EXACT, MIN_HOLDOUT_MIREX = 0.75, 0.80
MIN_ALL_EXACT, MIN_ALL_MIREX = 0.78, 0.83
MIN_SCORED = 50            # fewer resolvable tracks = the gate can't judge
MAX_KEY_SHARE = 0.20       # no single predicted key may own the set (old: 7B)

failures = []


def check(name, ok, detail):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def _parse(c):
    c = str(c).strip().upper()
    return int(c[:-1]), c[-1]


def mirex(est, ref):
    """MIREX weighting on Camelot codes: exact 1, fifth 0.5, relative 0.3,
    parallel 0.2, else 0."""
    try:
        en, el = _parse(est)
        rn, rl = _parse(ref)
    except (ValueError, IndexError):
        return 0.0
    if en == rn and el == rl:
        return 1.0
    if el == rl and (en - rn) % 12 in (1, 11):
        return 0.5
    if en == rn:
        return 0.3                       # relative major/minor share the number
    # parallel: a minor key's tonic major sits 3 Camelot steps up (7A Dm, 10B D)
    if el == "A" and rl == "B" and (rn - en) % 12 == 3:
        return 0.2
    if el == "B" and rl == "A" and (en - rn) % 12 == 3:
        return 0.2
    return 0.0


def test_mirex():
    print("MIREX scoring rules:\n")
    cases = [("7B", "7B", 1.0), ("7B", "8B", 0.5), ("12B", "1B", 0.5),
             ("8A", "8B", 0.3), ("7A", "10B", 0.2), ("8A", "11B", 0.2),
             ("7B", "4A", 0.2), ("7B", "5A", 0.0)]
    bad = [(e, r) for e, r, w in cases if mirex(e, r) != w]
    check("mirex table", not bad, f"{len(cases)} cases, wrong: {bad}")


def test_synthetic():
    import numpy as np
    from lib.dj import features as F
    print("\nSynthetic anchors (44.1k path, includes the 4:1 decimation):\n")
    rate = F.RATE
    tt = np.arange(int(12.0 * rate)) / rate
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * tt)).astype(np.float32)
    v = F.key_chroma(tone, rate)
    a = F.key_chroma_profile(v)
    check("A440 -> C-origin pc 9 / A-origin bin 0",
          int(np.argmax(v)) == 9 and int(np.argmax(a)) == 0,
          f"pc {int(np.argmax(v))}, A-origin bin {int(np.argmax(a))}, "
          f"sum {sum(a):.4f}")
    chord = (0.4 * np.sin(2 * np.pi * 440.0 * tt)
             + 0.3 * np.sin(2 * np.pi * 554.37 * tt)
             + 0.3 * np.sin(2 * np.pi * 659.25 * tt)).astype(np.float32)
    pc, mode, cam, conf, _ = F.estimate_key_audio(chord, rate)
    check("A major chord -> 11B", cam == "11B" and pc == 9,
          f"{cam} pc {pc} {mode} conf {conf:.2f}")
    r = F.estimate_key_audio(np.zeros(rate * 5, np.float32), rate)
    check("silence -> neutral, no chroma", r[3] == 0.0 and r[4] is None,
          f"{r[:4]}")


def _audio(row_id, path, cache):
    import numpy as np
    from lib.dj.features import RATE, decode_file
    if cache:
        p = os.path.join(cache, f"{row_id}.npy")
        if os.path.isfile(p):
            return np.load(p).astype(np.float32), 11025, "cache"
    return decode_file(path), RATE, "decode"


def test_truth(music, cache, verbose):
    import numpy as np
    from lib.dj.db import DB_FILENAME, LibraryDB
    from lib.dj.features import estimate_key_audio
    print("\nPublished-key truth set:\n")
    doc = json.load(open(TRUTH, encoding="utf-8"))
    ents = [e for e in doc["tracks"] if e["agreement"] != "conflicting"]
    dbp = os.path.join(music, DB_FILENAME)
    if not os.path.isfile(dbp):
        check("library present", False, f"no {dbp} (pass --music)")
        return
    db = LibraryDB(music, readonly=True)
    rows = []
    src = Counter()
    t0 = time.time()
    for e in ents:
        row = db.conn.execute(
            "SELECT id FROM tracks WHERE path = ? AND missing = 0",
            (e["path"],)).fetchone()
        path = db.abs(e["path"])
        if row is None or not os.path.isfile(path):
            if verbose:
                print(f"    missing: {e['title']}")
            continue
        x, sr, how = _audio(row["id"], path, cache)
        src[how] += 1
        _, _, cam, conf, _ = estimate_key_audio(x, sr)
        rows.append((e, cam, conf))
        if verbose:
            print(f"    {e['split']:7s} {e['camelot']:>4s} read {cam:<4s} "
                  f"conf {conf:.2f} mirex {mirex(cam, e['camelot']):.1f}  "
                  f"{e['title'][:44]}")
    db.close()
    print(f"  {len(rows)}/{len(ents)} scored tracks resolved "
          f"({dict(src)}) in {time.time() - t0:.0f}s")
    check("enough tracks", len(rows) >= MIN_SCORED,
          f"{len(rows)} resolvable (want >= {MIN_SCORED})")
    if not rows:
        return
    for split, lo_x, lo_m in (("holdout", MIN_HOLDOUT_EXACT, MIN_HOLDOUT_MIREX),
                              ("all", MIN_ALL_EXACT, MIN_ALL_MIREX)):
        rs = [r for r in rows if split == "all" or r[0]["split"] == split]
        if not rs:
            check(f"{split} present", False, "no tracks")
            continue
        ex = float(np.mean([cam == e["camelot"] for e, cam, _ in rs]))
        mx = float(np.mean([mirex(cam, e["camelot"]) for e, cam, _ in rs]))
        check(f"{split} exact", ex >= lo_x,
              f"{ex:.3f} (n={len(rs)}, want >= {lo_x})")
        check(f"{split} MIREX", mx >= lo_m,
              f"{mx:.3f} (n={len(rs)}, want >= {lo_m})")
    top, cnt = Counter(cam for _, cam, _ in rows).most_common(1)[0]
    check("no key collapse", cnt / len(rows) <= MAX_KEY_SHARE,
          f"most-read key {top} = {cnt}/{len(rows)} "
          f"(want <= {MAX_KEY_SHARE:.0%})")
    # confidence must mean something: the confident half beats the rest
    hi = [cam == e["camelot"] for e, cam, c in rows if c >= 0.5]
    lo = [cam == e["camelot"] for e, cam, c in rows if c < 0.5]
    if hi and lo:
        check("conf separates", np.mean(hi) > np.mean(lo),
              f"exact {np.mean(hi):.2f} at conf >= 0.5 (n={len(hi)}) vs "
              f"{np.mean(lo):.2f} below (n={len(lo)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", default="D:/Devel/music")
    ap.add_argument("--cache", default=os.environ.get("DJ_KEY_CACHE"),
                    help="dir of <track id>.npy mono 11025 Hz audio")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS
                              if os.name == "nt" else 10)
    except Exception:
        pass
    print("DJ key reader truth gate\n" + "=" * 40 + "\n")
    test_mirex()
    test_synthetic()
    test_truth(args.music, args.cache, args.verbose)
    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    main()
