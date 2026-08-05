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
            _CACHE["scores"] = {
                int(k): float(v["score"] if isinstance(v, dict) else v)
                for k, v in doc.get("scores", {}).items()
                if (v["score"] if isinstance(v, dict) else v) is not None}
            _CACHE["mtime"] = m
        except (OSError, ValueError):
            return {}
    return _CACHE["scores"]


def band_scores(track_id, region="mid"):
    """Per-band rhythmicity for a track region ('mid' body, 'in' intro,
    'out' exit), or None until the --bands pass has scored it. Falls back
    to the body when a region was unmeasurable."""
    p = path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return None
    if _CACHE.get("bands_mtime") != m:
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            _CACHE["bands"] = {int(k): v.get("bands")
                               for k, v in doc.get("scores", {}).items()
                               if isinstance(v, dict) and v.get("bands")}
            # accept both flat {band: s} and region {region: {band: s}}
            _CACHE["bands_mtime"] = m
        except (OSError, ValueError):
            return None
    b = _CACHE.get("bands", {}).get(track_id)
    if b is None:
        return None
    if "low" in b:                      # old flat format
        return b
    return b.get(region) or b.get("mid")


def blendable(track_id):
    """True / False / None (not yet measured - existing gates decide)."""
    s = scores().get(track_id)
    return None if s is None else s >= BLEND_MIN


BANDS = {"low": ("lowpass", 110.0),
         "mid": ("bandpass", (250.0, 2000.0)),
         "high": ("highpass", 4000.0)}


def compute(track, db, bands=False):
    """Score one track from its raw audio + stored grid.

    Returns a scalar (low-band score, the blend gate) or, with
    bands=True, {"low":s,"mid":s,"high":s} - the per-band rhythmicity
    the band-aware style selection consumes: different frequency bands
    mismatch independently, and each mix style is a strategy for which
    bands may overlap (user, 2026-08-04)."""
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
    if bands:
        # REGION-AWARE (user, 2026-08-04: timing per band, where it
        # matters): a blend overlaps A's EXIT with B's INTRO, and a track
        # that thumps in its body can be a diffuse wash in its intro. One
        # score set per region: "mid" (body), "in" (around the primary
        # mix-in), "out" (around the primary mix-out).
        def _span(center):
            l = int(max(center - 15, 0) * RATE)
            h = int(min(center + 15, track.duration_s) * RATE)
            return x[l:h].mean(axis=1).astype(np.float64), l
        regions = {"mid": (seg, lo)}
        try:
            if track.mix_ins:
                regions["in"] = _span(track.mix_ins[0]["time_s"] + 10)
            if track.mix_outs:
                regions["out"] = _span(track.mix_outs[0]["time_s"] - 10)
        except Exception:
            pass
        out = {}
        for rname, (rseg, rlo) in regions.items():
            if len(rseg) < 5 * RATE:
                continue
            out[rname] = {name: _band_score(track, rseg, rlo, kind, freq)
                          for name, (kind, freq) in BANDS.items()}
        return out or None
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


def _band_score(track, seg, lo, kind, freq):
    from scipy.signal import butter, sosfilt
    sos = butter(4, freq, btype=kind, fs=RATE, output="sos")
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
        span_lo = lo / RATE
        span_hi = span_lo + len(seg) / RATE
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
    ap.add_argument("--bands", action="store_true",
                    help="also fill per-band scores (for band-aware "
                         "style selection)")
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
        key = str(t.id)
        have = done.get(key)
        need_bands = args.bands and not isinstance(have, dict) or             (args.bands and isinstance(have, dict) and "bands" not in have)
        if have is not None and not need_bands:
            continue
        if args.bands:
            b = compute(t, db, bands=True)
            if b is not None:
                sc = (have.get("score") if isinstance(have, dict)
                      else have) or b.get("low")
                done[key] = {"score": sc if sc is not None else b.get("low"),
                             "bands": b}
        else:
            s = compute(t, db)
            if s is not None:
                done[key] = s
        if i % 10 == 0 or i == len(lib) - 1:
            os.makedirs(os.path.dirname(path()), exist_ok=True)
            with open(path(), "w", encoding="utf-8") as f:
                json.dump({"t": time.time(), "scores": done}, f)
            print(f"[{i + 1}/{len(lib)}] scored {len(done)}", flush=True)
        time.sleep(0.3)                  # breathing room - desktop first
    with open(path(), "w", encoding="utf-8") as f:
        json.dump({"t": time.time(), "scores": done}, f)
    vals = np.array([float(v["score"] if isinstance(v, dict) else v)
                     for v in done.values()
                     if (v["score"] if isinstance(v, dict) else v)
                     is not None])
    print(f"done: {len(done)} scored, median {np.median(vals):.2f}, "
          f"{np.mean(vals < BLEND_MIN):.0%} below the blend bar", flush=True)


if __name__ == "__main__":
    main()
