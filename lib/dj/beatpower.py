"""Beat power: does this track actually THUMP on its own beats?

The missing column that explained a day of chasing ghosts (2026-08-04).
Grid confidence measures periodicity - whether a steady lattice fits the
audio. It never asked whether the music puts low-band ATTACK energy ON
that lattice. Measured across the library: 38% of tracks score below
1.2 - confident grids over diffuse grooves (organic percussion, rolling
basslines, tribal walls). Beat-matching those is matching air: the sync
can be sample-perfect while the ear hears an unrelated mess, which is
exactly what the user reported ("the beats are fundamentally off...
double beat... all the time").

Score = mean on-beat low-band attack peak / mean off-beat (half-beat
later) attack peak, over ~30s at the track's midpoint. Density-neutral:
a busy full mix scores high if its kicks land on its beats, low only
when the off-beats carry just as much attack. ~1.0 = no beat to match;
>=2 = clean four-on-floor.

CLI (fills logs/beat_power.json incrementally, resumable):
    python -m lib.dj.beatpower --music D:/Devel/music
"""
import json
import os

import numpy as np

RATE = 44100
BLEND_MIN = 1.5        # both sides must clear this for overlapped drums


def path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "logs", "beat_power.json")


_CACHE = {"mtime": None, "scores": {}}


def scores():
    """{track_id: score} as last scanned. Cheap mtime-cached re-read."""
    p = path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return {}
    if _CACHE["mtime"] != m:
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            _CACHE["scores"] = {int(k): float(v)
                                for k, v in doc.get("scores", {}).items()}
            _CACHE["mtime"] = m
        except (OSError, ValueError):
            return {}
    return _CACHE["scores"]


def blendable(track_id):
    """True / False / None (not yet measured - existing gates decide)."""
    s = scores().get(track_id)
    return None if s is None else s >= BLEND_MIN


def compute(track, db):
    """Score one track from its raw audio + stored grid, or None."""
    from lib.dj.features import decode_file_stereo
    from scipy.signal import butter, sosfilt
    try:
        x = decode_file_stereo(db.abs(track.path))
    except Exception:
        return None
    mid = track.duration_s * 0.5
    lo = int(max(mid - 15, 0) * RATE)
    hi = int(min(mid + 15, track.duration_s) * RATE)
    seg = x[lo:hi].mean(axis=1).astype(np.float64)
    if len(seg) < 5 * RATE:
        return None
    sos = butter(4, 110.0, btype="lowpass", fs=RATE, output="sos")
    env = np.abs(sosfilt(sos, seg))
    w = max(int(0.01 * RATE), 1)
    env = np.convolve(env, np.ones(w) / w, mode="same")
    att = np.diff(env)
    att[att < 0] = 0.0
    beats = []
    for g in (track.grid or []):
        per = g.get("period_s") or 0
        if per <= 0:
            continue
        b = g["first_beat_s"]
        span_lo, span_hi = lo / RATE, hi / RATE
        if b < span_lo:
            b += np.ceil((span_lo - b) / per) * per
        while b <= min(span_hi, g["end_s"]):
            beats.append((b - span_lo, per))
            b += per
    if len(beats) < 16:
        return None
    on = off = 0.0
    n_on = n_off = 0
    w2 = int(0.03 * RATE)
    for b, per in beats:
        i = int(b * RATE)
        if i + w2 >= len(att):
            break
        on += float(np.max(att[max(i - w2, 0):i + w2]))
        n_on += 1
        j = int((b + per * 0.5) * RATE)
        if j + w2 < len(att):
            off += float(np.max(att[j - w2:j + w2]))
            n_off += 1
    if not n_on or not n_off or off <= 0:
        return None
    return round((on / n_on) / (off / n_off), 3)


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", required=True)
    args = ap.parse_args()
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from lib.dj.brain import load_library
    from lib.dj.db import LibraryDB
    db = LibraryDB(args.music)
    lib = load_library(db)
    done = {}
    try:
        with open(path(), encoding="utf-8") as f:
            done = json.load(f).get("scores", {})
    except (OSError, ValueError):
        pass
    print(f"{len(lib)} tracks, {len(done)} already scored", flush=True)
    for i, t in enumerate(lib):
        if str(t.id) in done:
            continue
        s = compute(t, db)
        if s is not None:
            done[str(t.id)] = s
        if i % 10 == 0 or i == len(lib) - 1:
            os.makedirs(os.path.dirname(path()), exist_ok=True)
            with open(path(), "w", encoding="utf-8") as f:
                json.dump({"t": time.time(), "scores": done}, f)
            print(f"[{i + 1}/{len(lib)}] scored {len(done)}", flush=True)
        time.sleep(0.3)                  # breathing room - desktop first
    with open(path(), "w", encoding="utf-8") as f:
        json.dump({"t": time.time(), "scores": done}, f)
    vals = np.array(list(map(float, done.values())))
    print(f"done: {len(done)} scored, median {np.median(vals):.2f}, "
          f"{np.mean(vals < BLEND_MIN):.0%} below the blend bar", flush=True)


if __name__ == "__main__":
    main()
