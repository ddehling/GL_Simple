"""Score a recreation against the original, locally and globally.

Both sides are per-bar feature tracks (ingest.bar_features): energy dB,
band shares, low/high onset density, chroma. Local score per window of
`window` bars (default 4 = one phrase) blends four distances:

  energy    |dB difference|            exp(-d / 6)
  spectrum  L1 of the three band shares  1 - 0.5 * L1
  rhythm    |low hits| + |high hits| per s   exp(-d / 2)
  harmony   cosine of mean chroma        (cos + 1) / 2 -> clipped 0..1

local = 100 * (0.35 energy + 0.2 spectrum + 0.2 rhythm + 0.25 harmony)

Global blends the mean local score with STRUCTURE (correlation of the two
energy envelopes over the song, so sections rise and fall in the same
places even where the timbre differs) and the tempo / key agreement.
Everything is 0..100; 100 is the song against itself."""
from __future__ import annotations

import numpy as np

WEIGHTS = {"energy": 0.35, "spectrum": 0.2, "rhythm": 0.2, "harmony": 0.25}


def _window(feats, i0, i1):
    seg = feats[i0:i1]
    if not seg:
        return None
    return {"energy_db": float(np.mean([f["energy_db"] for f in seg])),
            "shares": np.array([[f["bass"], f["mid"], f["high"]] for f in seg]).mean(axis=0),
            "low_hits": float(np.mean([f["low_hits"] for f in seg])),
            "high_hits": float(np.mean([f["high_hits"] for f in seg])),
            "chroma": np.array([f["chroma"] for f in seg]).mean(axis=0)}


def _local(a, b):
    d_e = abs(a["energy_db"] - b["energy_db"])
    s_e = float(np.exp(-d_e / 6.0))
    s_s = float(max(0.0, 1.0 - 0.5 * np.abs(a["shares"] - b["shares"]).sum()))
    d_r = abs(a["low_hits"] - b["low_hits"]) + abs(a["high_hits"] - b["high_hits"])
    s_r = float(np.exp(-d_r / 2.0))
    ca, cb = a["chroma"], b["chroma"]
    na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
    cos = float(ca @ cb / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
    s_h = float(min(1.0, max(0.0, cos)))
    total = 100.0 * (WEIGHTS["energy"] * s_e + WEIGHTS["spectrum"] * s_s + WEIGHTS["rhythm"] * s_r + WEIGHTS["harmony"] * s_h)
    return {"score": round(total, 1), "energy": round(100 * s_e, 1), "spectrum": round(100 * s_s, 1),
            "rhythm": round(100 * s_r, 1), "harmony": round(100 * s_h, 1), "d_energy_db": round(d_e, 2)}


def compare(orig, recon, window: int = 4, bpm_orig=None, bpm_recon=None, key_orig=None, key_recon=None):
    """orig / recon: per-bar feature lists. Returns
    {"global", "local": [{"bar0", "t", ...}], "structure", "tempo", "key", "n_bars"}"""
    n = min(len(orig), len(recon))
    local = []
    for i0 in range(0, n, window):
        i1 = min(n, i0 + window)
        a, b = _window(orig, i0, i1), _window(recon, i0, i1)
        if a is None or b is None:
            break
        rec = _local(a, b)
        rec.update({"bar0": i0, "bars": i1 - i0, "t": orig[i0]["t"]})
        local.append(rec)
    mean_local = float(np.mean([r["score"] for r in local])) if local else 0.0
    # structure: the energy envelopes (normalised) should rise and fall together
    ea = np.array([f["energy_db"] for f in orig[:n]])
    eb = np.array([f["energy_db"] for f in recon[:n]])
    if n >= 8 and ea.std() > 1e-6 and eb.std() > 1e-6:
        r = float(np.corrcoef(ea, eb)[0, 1])
        structure = 100.0 * max(0.0, (r + 1.0) / 2.0)
    else:
        structure = 50.0
    tempo = 100.0
    if bpm_orig and bpm_recon:
        rel = abs(float(bpm_orig) - float(bpm_recon)) / max(float(bpm_orig), 1e-6)
        tempo = 100.0 * float(np.exp(-rel / 0.02))
    key = 100.0 if (key_orig is None or key_recon is None or str(key_orig) == str(key_recon)) else 40.0
    coverage = min(1.0, n / max(1, max(len(orig), len(recon))))
    glob = (0.65 * mean_local + 0.25 * structure + 0.05 * tempo + 0.05 * key) * (0.7 + 0.3 * coverage)
    return {"global": round(float(glob), 1), "mean_local": round(mean_local, 1), "structure": round(structure, 1),
            "tempo": round(tempo, 1), "key": round(key, 1), "coverage": round(coverage, 3), "n_bars": n, "local": local,
            "weights": dict(WEIGHTS)}


def worst(report, k=3):
    return sorted(report.get("local", []), key=lambda r: r["score"])[:k]
