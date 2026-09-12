"""Structure measured from the STEMS - what the DJ hears in a song's parts, finer than the scanner's
section labels:

* SECTION CHROMA: a 12-bin pitch-class profile per section, from the harmonic stems (bass + other +
  vocals, never drums). The whole-track chroma says what key a song is in; this says what CHORDS sound
  in each part, so a bed and a voice can be checked for harmonic fit at the bars they actually overlap.
* THE HOOK: the loudest, most repeated sung passage - the payoff a crowd waits for. From the vocal stem:
  per-bar loudness and a per-bar timbre / pitch signature; four-bar windows that recur (self-similarity)
  and are loud score highest; an ML "chorus" label on the window is a bonus. Instrumentals have none.

Stored in tracks.axes ("secchroma", "hook", "measure_v") - a free-form JSON column the scanner preserves
across rescans (see scan._KEEP_AXES_KEYS). Consumers: the conductor's tonal guard and payoff timing, the
brain's seam scoring, the Director's song list. Everything is evidence-gated: a track without the
measurement behaves exactly as before.

Run:  python tools/dj/dj_scan.py --dir D:/Devel/music --measure        (resumable, commits per track)
"""
import json
import math
import os

import numpy as np

MEASURE_V = 1
RATE = 44100
HOOK_BARS = 4                     # the window a hook is measured over
HOOK_SIM = 0.80                   # windows this alike count as repeats
SUNG_FRAC = 0.35                  # a window is "sung" when its vocal loudness is this share of the loudest window
SILENT_DBFS = -45.0               # a vocal stem never above this is an instrumental
HOOK_MIN_DBFS = -32.0             # ...and a hook window quieter than this is bleed, not a chorus
MIN_SECTION_S = 2.0


def _mono(path):
    from lib.dj.features import decode_file_stereo
    x = decode_file_stereo(path)
    return np.asarray(x, dtype=np.float32).mean(axis=1)


def _bars_of(track, n_samples):
    """Bar boundaries (seconds) over the track from the beat grid: a list of downbeat times, at least 2."""
    dur = n_samples / RATE
    g = track.grid[0] if track.grid else None
    if g is None or not g.get("period_s"):
        bar = 4 * 60.0 / max(track.bpm or 120.0, 1.0)
        return [k * bar for k in range(int(dur // bar) + 1)]
    bar = 4 * g["period_s"]
    first = g["first_beat_s"] + (track.downbeat_offset or 0) * g["period_s"]
    while first - bar >= 0.0:
        first -= bar
    out = []
    t = first
    while t < dur:
        out.append(t)
        t += bar
    return out if len(out) >= 2 else [0.0, dur]


def section_chroma(track, harmonic_mono):
    """[12 floats or None per section] from the harmonic stems' frame chroma (40 fps)."""
    from lib.dj.features import frame_track, FPS
    if len(harmonic_mono) < RATE * 2:
        return None
    _bands, chroma = frame_track(np.ascontiguousarray(harmonic_mono))
    # frame energy gates silence out of the means (a rest is not a chord)
    frame_e = np.sqrt(np.mean(np.square(harmonic_mono[:len(chroma) * (RATE // FPS)].reshape(-1, RATE // FPS)), axis=1)) if len(harmonic_mono) >= len(chroma) * (RATE // FPS) else None
    out = []
    for s in track.sections or []:
        f0, f1 = int(s["start_s"] * FPS), int(s["end_s"] * FPS)
        f0, f1 = max(0, min(f0, len(chroma))), max(0, min(f1, len(chroma)))
        if f1 - f0 < MIN_SECTION_S * FPS:
            out.append(None)
            continue
        seg = chroma[f0:f1]
        if frame_e is not None:
            e = frame_e[f0:f1]
            keep = e >= 0.25 * (float(np.percentile(e, 90)) or 1e-9)
            if keep.sum() >= 0.5 * FPS:
                seg = seg[keep]
        m = seg.mean(axis=0)
        tot = float(m.sum())
        if tot <= 1e-9:
            out.append(None)
            continue
        out.append([round(float(v / tot), 4) for v in m])
    return out


def find_hook(track, vocal_mono):
    """The hook: {"start_s", "end_s", "score", "repeats", "loud_dbfs", "chorus"} or None (instrumental)."""
    from lib.dj.features import frame_track, FPS
    if len(vocal_mono) < RATE * 8:
        return None
    peak_rms = float(np.sqrt(np.mean(np.square(vocal_mono.astype(np.float32)))))
    bars = _bars_of(track, len(vocal_mono))
    if len(bars) < HOOK_BARS + 2:
        return None
    bands, chroma = frame_track(np.ascontiguousarray(vocal_mono))
    lb = np.log10(bands + 1e-6)
    # per bar: loudness (RMS), a 44-dim signature (12 chroma + 32 log bands), unit length
    n = len(bars) - 1
    loud = np.zeros(n, dtype=np.float64)
    sig = np.zeros((n, 44), dtype=np.float64)
    for i in range(n):
        a, b = int(bars[i] * RATE), int(bars[i + 1] * RATE)
        seg = vocal_mono[a:b]
        if len(seg) < RATE // 4:
            continue
        loud[i] = float(np.sqrt(np.mean(np.square(seg))))
        f0, f1 = int(bars[i] * FPS), max(int(bars[i] * FPS) + 1, int(bars[i + 1] * FPS))
        f0, f1 = min(f0, len(chroma) - 1), min(f1, len(chroma))
        v = np.concatenate([chroma[f0:f1].mean(axis=0) * 3.0, lb[f0:f1].mean(axis=0)])
        v = v - v.mean()
        nv = float(np.linalg.norm(v))
        if nv > 1e-9:
            sig[i] = v / nv
    top = float(loud.max()) if n else 0.0
    if top <= 0.0 or 20 * math.log10(top + 1e-12) < SILENT_DBFS:
        return None                                   # an instrumental: no hook
    W = HOOK_BARS
    nw = n - W + 1
    if nw < 2:
        return None
    wl = np.array([loud[i:i + W].mean() for i in range(nw)])
    sung = wl >= SUNG_FRAC * wl.max()
    # similarity of windows: mean per-bar cosine over the W bars
    wsig = np.stack([sig[i:i + W] for i in range(nw)])              # [nw, W, 44]
    flat = wsig.reshape(nw, -1)
    sim = flat @ flat.T / W
    chorus = np.zeros(nw, dtype=bool)
    for i in range(nw):
        mid = 0.5 * (bars[i] + bars[min(i + W, n)])
        if track.ml_segment_at(mid) == "chorus":
            chorus[i] = True
    # the ML vocal curve (axes["vc"]: [(t, vocalness)]) tells bleed from singing - a whistle, a choir pad or
    # a synth the separator left in the vocal stem is loud but reads as no vocal there
    vc = (track.axes or {}).get("vc") or []
    vx = np.asarray([p[0] for p in vc], dtype=np.float64) if len(vc) >= 2 else None
    vy = np.asarray([p[1] for p in vc], dtype=np.float64) if len(vc) >= 2 else None

    def vocalness(i):
        if vx is None:
            return None
        t0, t1 = bars[i], bars[min(i + W, n)]
        ts = np.linspace(t0, t1, 5)
        return float(np.max(np.interp(ts, vx, vy)))
    best, best_score, best_rep = None, -1.0, 0
    for i in range(nw):
        if not sung[i]:
            continue
        v = vocalness(i)
        if v is not None and v < 0.03:
            continue                                  # loud in the stem, but nobody is singing (the ML curve is a
            #                                           demucs vocal FRACTION: 0.1 is a sung chorus, instrumentals read 0.0)
        rep = int(sum(1 for j in range(nw) if abs(j - i) >= W and sung[j] and sim[i, j] >= HOOK_SIM))
        if rep < 1 and not chorus[i]:
            continue                                  # a hook recurs (or the ML pass calls it a chorus)
        score = (wl[i] / wl.max()) * (1.0 + 0.5 * rep) * (1.3 if chorus[i] else 1.0)
        if score > best_score:
            best, best_score, best_rep = i, score, rep
    if best is None or 20 * math.log10(float(wl[best]) + 1e-12) < HOOK_MIN_DBFS:
        return None
    # every occurrence of the hook (windows alike to the best one, loud enough), merged into runs: the FIRST
    # occurrence is where the payoff first comes - what the DJ waits for; later ones are the next chances
    chorus_any = bool(chorus.any())
    alike = [j for j in range(nw) if sung[j] and sim[best, j] >= HOOK_SIM and wl[j] >= 0.75 * wl[best]
             and (not chorus_any or chorus[j])] + [best]        # a verse in the same voice is not the hook
    alike = sorted(set(alike))
    runs = []
    for j in alike:
        if runs and j - runs[-1][1] <= 1:
            runs[-1][1] = j
        else:
            runs.append([j, j])
    starts = [round(float(bars[lo]), 3) for lo, _ in runs][:8]
    lo, hi = runs[0]
    hi = min(hi, lo + 4 * W - 1)                       # at most 16 bars
    return {"start_s": starts[0], "end_s": round(float(bars[min(hi + W, n)]), 3), "starts": starts,
            "score": round(float(best_score), 3), "repeats": best_rep,
            "loud_dbfs": round(20 * math.log10(float(wl[best]) + 1e-12), 1), "chorus": bool(chorus[best]),
            "vocalness": (None if vocalness(best) is None else round(vocalness(best), 2))}


def measure_track(track, music_root):
    """Both measurements for one track from its stems on disk; None when it has no stems."""
    from lib.dj.stems import stem_paths
    paths = stem_paths(music_root, track.id)
    if paths is None:
        return None
    bass, other, vocals = _mono(paths["bass"]), _mono(paths["other"]), _mono(paths["vocals"])
    n = min(len(bass), len(other), len(vocals))
    harmonic = (bass[:n] + other[:n] + vocals[:n]) / 3.0
    out = {"measure_v": MEASURE_V}
    try:
        out["secchroma"] = section_chroma(track, harmonic)
    except Exception as e:  # noqa: BLE001
        out["secchroma_error"] = f"{type(e).__name__}: {e}"
    try:
        out["hook"] = find_hook(track, vocals[:n])
    except Exception as e:  # noqa: BLE001
        out["hook_error"] = f"{type(e).__name__}: {e}"
    return out


def measure_pass(db, music_root, progress_cb=None, force=False, only_ids=None):
    """Measure every stem-bearing track that has no current measurement; commits per track (resumable)."""
    from lib.dj import brain as B
    from lib.dj.stems import has_stems
    lib = [t for t in B.load_library(db) if not t.excluded and has_stems(music_root, t.id)]
    if only_ids:
        lib = [t for t in lib if t.id in set(only_ids)]
    todo = [t for t in lib if force or int((t.axes or {}).get("measure_v") or 0) < MEASURE_V]
    done = errors = 0
    for i, t in enumerate(todo):
        if progress_cb:
            progress_cb(i, len(todo), t.title)
        try:
            m = measure_track(t, music_root)
            if m is None:
                continue
            row = db.conn.execute("SELECT axes FROM tracks WHERE id = ?", (t.id,)).fetchone()
            axes = json.loads(row["axes"]) if row and row["axes"] else {}
            axes.update(m)
            db.conn.execute("UPDATE tracks SET axes = ? WHERE id = ?", (json.dumps(axes), t.id))
            db.conn.commit()
            done += 1
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  measure failed on {t.title}: {type(e).__name__}: {e}", flush=True)
    if progress_cb:
        progress_cb(len(todo), len(todo), "done")
    return {"status": "done", "measured": done, "errors": errors, "todo": len(todo), "with_stems": len(lib)}
