"""Score a recreation against the original, locally and globally.

Both sides are per-bar feature tracks (ingest.bar_features): energy dB,
band shares, low/high onset density, chroma, and a 32-band spectral
PROFILE. Local score per window of `window` bars (default 4 = one
phrase) blends five distances:

  energy    |dB difference|                 exp(-d / 6)
  spectrum  L1 of the three band shares     1 - 0.5 * L1
  timbre    correlation of the 32-band log profiles   (r + 1) / 2
  rhythm    |low hits| + |high hits| per s  exp(-d / 2)
  harmony   cosine of mean chroma           clipped 0..1

local = 100 * (0.3 energy + 0.15 spectrum + 0.15 timbre + 0.15 rhythm + 0.25 harmony)

Before comparing, the two energy envelopes are ALIGNED with dynamic time
warping (a band of +-8 bars), so a section that lands a bar late in the
recreation is compared with the right bars rather than dragging every
later window down. Global blends the mean local score with STRUCTURE
(correlation of the aligned envelopes) and the tempo / key agreement.
Everything is 0..100; 100 is the song against itself."""
from __future__ import annotations

import numpy as np

WEIGHTS = {"energy": 0.3, "spectrum": 0.15, "timbre": 0.15, "rhythm": 0.15, "harmony": 0.25}


def dtw_align(a, b, band: int = 8):
    """Map each index of a (original bars) to an index of b (recreation
    bars) by DTW over the two sequences (energy dB), within a band.
    Returns a list `m` with m[i] = j."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    scale = m / max(n, 1)
    for i in range(1, n + 1):
        jc = int(round(i * scale))
        for j in range(max(1, jc - band), min(m, jc + band) + 1):
            c = abs(a[i - 1] - b[j - 1])
            D[i, j] = c + min(D[i - 1, j - 1], D[i - 1, j] + 0.5, D[i, j - 1] + 0.5)
    # backtrack
    i, j = n, m
    if D[n, m] >= INF:            # band too narrow at the end: fall back to the diagonal
        return [min(m - 1, int(round(k * scale))) for k in range(n)]
    path = {}
    while i > 0 and j > 0:
        path.setdefault(i - 1, j - 1)
        steps = [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1)]
        _, i, j = min(steps, key=lambda t: t[0])
    out = []
    last = 0
    for k in range(n):
        last = path.get(k, last)
        out.append(min(m - 1, last))
    return out


def _window(feats, idx):
    seg = [feats[i] for i in idx if 0 <= i < len(feats)]
    if not seg:
        return None
    out = {"energy_db": float(np.mean([f["energy_db"] for f in seg])),
           "shares": np.array([[f["bass"], f["mid"], f["high"]] for f in seg]).mean(axis=0),
           "low_hits": float(np.mean([f["low_hits"] for f in seg])),
           "high_hits": float(np.mean([f["high_hits"] for f in seg])),
           "chroma": np.array([f["chroma"] for f in seg]).mean(axis=0)}
    if all("profile" in f for f in seg):
        out["profile"] = np.array([f["profile"] for f in seg]).mean(axis=0)
    if all("pattern" in f for f in seg):
        out["pattern"] = np.concatenate([np.array([f["pattern"][k] for f in seg]).mean(axis=0) for k in ("kick", "snare", "hat")])
    return out


def _local(a, b):
    d_e = abs(a["energy_db"] - b["energy_db"])
    s_e = float(np.exp(-d_e / 6.0))
    s_s = float(max(0.0, 1.0 - 0.5 * np.abs(a["shares"] - b["shares"]).sum()))
    d_r = abs(a["low_hits"] - b["low_hits"]) + abs(a["high_hits"] - b["high_hits"])
    s_r = float(np.exp(-d_r / 2.0))
    if "pattern" in a and "pattern" in b:
        pa, pb = a["pattern"], b["pattern"]
        na, nb = np.linalg.norm(pa), np.linalg.norm(pb)
        s_p = float(pa @ pb / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
        s_r = 0.5 * s_r + 0.5 * max(0.0, s_p)          # the beat itself, not just how busy it is
    ca, cb = a["chroma"], b["chroma"]
    na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
    cos = float(ca @ cb / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
    s_h = float(min(1.0, max(0.0, cos)))
    if "profile" in a and "profile" in b and a["profile"].std() > 1e-9 and b["profile"].std() > 1e-9:
        r = float(np.corrcoef(a["profile"], b["profile"])[0, 1])
        s_t = float(max(0.0, (r + 1.0) / 2.0))
    else:
        s_t = s_s
    total = 100.0 * (WEIGHTS["energy"] * s_e + WEIGHTS["spectrum"] * s_s + WEIGHTS["timbre"] * s_t
                     + WEIGHTS["rhythm"] * s_r + WEIGHTS["harmony"] * s_h)
    return {"score": round(total, 1), "energy": round(100 * s_e, 1), "spectrum": round(100 * s_s, 1), "timbre": round(100 * s_t, 1),
            "rhythm": round(100 * s_r, 1), "harmony": round(100 * s_h, 1), "d_energy_db": round(d_e, 2)}


def compare(orig, recon, window: int = 4, bpm_orig=None, bpm_recon=None, key_orig=None, key_recon=None, align: bool = True):
    """orig / recon: per-bar feature lists. Returns
    {"global", "local": [{"bar0", "t", ...}], "structure", "tempo", "key", "n_bars", "mapping"}"""
    n = min(len(orig), len(recon))
    if align and n >= 8:
        mapping = dtw_align([f["energy_db"] for f in orig], [f["energy_db"] for f in recon])
    else:
        mapping = list(range(n))
    local = []
    for i0 in range(0, len(orig), window):
        idx_o = list(range(i0, min(len(orig), i0 + window)))
        idx_r = [mapping[i] for i in idx_o if i < len(mapping)]
        a, b = _window(orig, idx_o), _window(recon, idx_r)
        if a is None or b is None:
            break
        rec = _local(a, b)
        rec.update({"bar0": i0, "bars": len(idx_o), "t": orig[i0]["t"], "recon_bar0": idx_r[0] if idx_r else None})
        local.append(rec)
    mean_local = float(np.mean([r["score"] for r in local])) if local else 0.0
    # structure: the aligned energy envelopes should rise and fall together
    ea = np.array([f["energy_db"] for f in orig[: len(mapping)]])
    eb = np.array([recon[j]["energy_db"] for j in mapping])
    if len(ea) >= 8 and ea.std() > 1e-6 and eb.std() > 1e-6:
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
            "weights": dict(WEIGHTS), "mapping": mapping}


def worst(report, k=3):
    return sorted(report.get("local", []), key=lambda r: r["score"])[:k]


def section_scores(report, script):
    """Mean local score per script section: [(index, section, score)]."""
    out = []
    bar = 0
    for i, e in enumerate(script.get("sections", [])):
        rs = [r for r in report.get("local", []) if bar <= r["bar0"] < bar + e["bars"]]
        out.append((i, e["section"], round(float(np.mean([r["score"] for r in rs])), 1) if rs else None))
        bar += e["bars"]
    return out
