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
# Bars lowered 2026-08-05 (were 1.5/1.2): they were set while sync could
# only align GRIDS - blending borderline-groove material then smeared
# ("matching air"). With kicks now measured and phase-locked (validated
# 2-17ms on rendered seams) and the pattern screens (density mismatch,
# 1:1 interleave, swing clash) still standing, the bars only need to
# exclude genuinely beatless material. A week of over-gating turned
# nights into fade marathons ("a week ago it was pretty good. now every
# dj mix sucks" - user; the live log: five fades in a row).
BLEND_MIN = 1.3        # the INCOMING side: it becomes the foundation
BLEND_MIN_EXIT = 1.05  # the OUTGOING side: it only hands off


def path():
    # DJ_BEATPOWER_PATH: hermetic override for test harnesses. The
    # synthetic-fleet e2e uses tiny track ids that COLLIDE with real
    # library ids in the production file - its fake tracks inherited
    # real tracks' scores and lost their blends to another song's beat
    # power (caught 2026-08-05 when phrase_cut's retirement unmasked it).
    env = os.environ.get("DJ_BEATPOWER_PATH")
    if env:
        return env
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


def profile_coverage(track_id):
    """Fraction of a track's phase-profile buckets that pass the trust
    bar - i.e. how much of the track has a MEASURED-good grid (attacks
    on the lattice, consistent phase, which also pins the local period).
    0.0 when unscanned. A high coverage outranks a stale whole-track
    bpm_conf scalar: the profile is direct evidence, the scalar a fit
    score from the original scan."""
    p = path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return 0.0
    if _CACHE.get("phase_mtime") != m:
        phase_offset(track_id)           # refresh the shared cache
    d = _CACHE.get("phase", {}).get(track_id)
    prof = (d or {}).get("prof")
    if not prof:
        return 0.0
    ok = sum(1 for r in prof
             if r.get("n", 0) >= 10 and r.get("iqr", 999.0) <= 55.0)
    return ok / len(prof)


def phase_offset(track_id, region="mid", at_s=None):
    """Measured music-vs-grid offset in SECONDS for a track region, or
    None when unmeasured or unreliable. Positive = the audible low-band
    attacks land LATE relative to the stored grid.

    Pass at_s (source seconds of the seam anchor) to select the NEAREST
    measured region by POSITION - a seam forced mid-track (the lab, the
    knob sweep) must not read the offset measured at the primary mix-out
    100s away; phase is a local property. Without at_s, the region label
    ('in'/'mid'/'out') selects, with 'mid' fallback.

    This is the column the 2026-08-04 beat-match hunt ended on: stored
    grids are periodically right (high confidence) but their PHASE misses
    the actual kicks by ~48ms median in seam regions, with signs
    differing per track - so grid-to-grid sync aligns lattices while the
    ear hears flam. The sync bias consumes the DIFFERENCE of these
    offsets to put kicks, not grids, in register."""
    p = path()
    try:
        m = os.path.getmtime(p)
    except OSError:
        return None
    if _CACHE.get("phase_mtime") != m:
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
            _CACHE["phase"] = {int(k): v.get("phase")
                               for k, v in doc.get("scores", {}).items()
                               if isinstance(v, dict) and v.get("phase")}
            _CACHE["phase_mtime"] = m
        except (OSError, ValueError):
            return None
    d = _CACHE.get("phase", {}).get(track_id)
    if not d:
        return None

    def _ok(rec):
        # Trust bar: enough beats with real attacks, and a consistent
        # story. A wide IQR means the bucket has no single phase
        # (organic/rubato) - correcting with its median is a coin flip.
        return (rec is not None and rec.get("n", 0) >= 10
                and rec.get("iqr", 999.0) <= 55.0)

    prof = d.get("prof")
    if prof is not None:
        if at_s is None:
            at_s = prof[len(prof) // 2]["at_s"]    # label-less: track body
        pts = [(r["at_s"], r["ms"]) for r in prof if _ok(r)]
        if not pts:
            return None
        # If the bucket AT this position was measured but failed the
        # trust bar, the phase HERE is genuinely unstable (fill, break,
        # rubato) - interpolating neighbors across it is a guess dressed
        # as a measurement (validation: -44ms miss on exactly this case).
        # Decline; no correction beats a wrong one.
        for r in prof:
            if abs(r["at_s"] - at_s) <= PROF_BUCKET_S / 2 and not _ok(r):
                return None
        xs = np.asarray([p[0] for p in pts])
        nearest = float(np.min(np.abs(xs - at_s)))
        if nearest > PROF_REACH_S:
            return None       # nothing trustworthy near this position
        # INTERPOLATE, don't stair-step: measured profiles are smooth
        # RAMPS (~-0.44 ms/s on most of this library - the stored grid
        # period is systematically ~0.04% off the music, a scanner tempo
        # quantization artifact), so the local phase at a seam is the
        # line through the neighboring buckets, not the nearest median.
        return float(np.interp(at_s, xs,
                               np.asarray([p[1] for p in pts]))) / 1000.0
    # legacy labeled-region records (pre-profile format)
    if at_s is not None:
        best, dist = None, None
        for rec in d.values():
            if not _ok(rec) or rec.get("at_s") is None:
                continue
            dd = abs(rec["at_s"] - at_s)
            if dist is None or dd < dist:
                best, dist = rec, dd
        if best is not None:
            return best["ms"] / 1000.0
    p_ = d.get(region) or d.get("mid")
    if not _ok(p_):
        return None
    return p_["ms"] / 1000.0


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


PROF_BUCKET_S = 20.0     # phase profile resolution along the track
PROF_REACH_S = 30.0      # farthest a bucket may be from the asked position


def compute_phase(track, db):
    """DENSE music-vs-grid phase profile from the raw audio.

    One full-track pass: low-band attack envelope, then for EVERY grid
    beat find the strongest attack peak within +/-75ms (capped at 0.3
    beat) and record peak_time - beat_time. Offsets are bucketed every
    ~20s; each bucket keeps median/iqr/count. Returns
    {"prof": [{"at_s", "ms", "iqr", "n"}, ...]}.

    Dense on purpose (2026-08-04): the first cut measured 3 labeled
    regions (in/mid/out) and validation immediately caught seams landing
    60-100s from the nearest measurement, where the local phase had
    drifted 40-90ms away - phase is a LOCAL property, so the profile
    must cover wherever a seam can land. Same instrument that DIAGNOSED
    the 48ms defect (rendered attacks vs trace-projected beats) applied
    at the source, so correction and diagnosis can never disagree about
    what a kick is."""
    from lib.dj.features import decode_file_stereo
    from scipy.signal import butter, sosfilt
    try:
        x = decode_file_stereo(db.abs(track.path))
    except Exception:
        return None
    mono = x.mean(axis=1).astype(np.float32)
    del x
    sos = butter(4, 110.0, btype="lowpass", fs=RATE, output="sos")
    env = np.abs(sosfilt(sos, mono)).astype(np.float32)
    del mono
    w = max(int(0.01 * RATE), 1)
    env = np.convolve(env, np.ones(w, dtype=np.float32) / w, mode="same")
    att = np.diff(env)
    del env
    att[att < 0] = 0.0

    peaks = []                       # (beat_s, peak_value, offset_s)
    for g in (track.grid or []):
        per = g.get("period_s") or 0
        if per <= 0:
            continue
        half = int(min(0.075, 0.3 * per) * RATE)
        b = g["first_beat_s"]
        if b < 0:
            b += np.ceil(-b / per) * per
        while b <= g["end_s"]:
            i = int(b * RATE)
            if i - half >= 0 and i + half < len(att):
                wseg = att[i - half:i + half]
                k = int(np.argmax(wseg))
                peaks.append((b, float(wseg[k]), (k - half) / RATE))
            b += per
    del att
    if len(peaks) < 24:
        return None
    ref = float(np.median([p for _, p, _ in peaks]))
    prof = []
    n_buckets = int(track.duration_s / PROF_BUCKET_S) + 1
    for bi in range(n_buckets):
        lo, hi = bi * PROF_BUCKET_S, (bi + 1) * PROF_BUCKET_S
        offs = [o for b, p, o in peaks
                if lo <= b < hi and p > 0 and p >= 0.3 * ref]
        if len(offs) < 10:
            continue
        q1, q3 = np.percentile(offs, [25, 75])
        prof.append({"at_s": round(lo + PROF_BUCKET_S / 2, 1),
                     "ms": round(float(np.median(offs)) * 1000.0, 1),
                     "iqr": round(float(q3 - q1) * 1000.0, 1),
                     "n": len(offs)})
    return {"prof": prof} if prof else None


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
    ap.add_argument("--phase", action="store_true",
                    help="also fill per-region music-vs-grid phase "
                         "offsets (kick alignment bias)")
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
        need_bands = args.bands and (not isinstance(have, dict)
                                     or "bands" not in have)
        need_phase = args.phase and (not isinstance(have, dict)
                                     or "phase" not in have
                                     or "prof" not in have["phase"])
        if have is not None and not need_bands and not need_phase:
            continue
        if args.bands or args.phase:
            rec = dict(have) if isinstance(have, dict) else \
                ({"score": have} if have is not None else {"score": None})
            if need_bands:
                b = compute(t, db, bands=True)
                if b is not None:
                    rec["bands"] = b
                    if rec.get("score") is None:
                        rec["score"] = b.get("low")
            if need_phase:
                p = compute_phase(t, db)
                if p is not None:
                    rec["phase"] = p
            if rec:
                done[key] = rec
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
