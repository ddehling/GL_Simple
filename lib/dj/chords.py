"""A chord track for a song program: one chord per bar (or half bar)
from the harmonic content of the melodic stems, with the bass note as
the root when the bass holds one.

chord_track(stems_mono, beats, down0, period) -> [{"bar": b, "root": pc,
"quality": "maj"|"min"|"dom7"|"min7"|"sus", "bass": pc | None, "conf": 0..1}]

Templates are matched on a beat-synchronous chroma of the `other` stem's
harmonic part (HPSS) summed with the bass stem's chroma; a Viterbi pass
prefers staying on a chord (songs change chords on bars, not on beats),
and the root reads from the bass when the bass chroma is peaked.
"""
import numpy as np

from lib.dj import instruments as INS

RATE = INS.RATE
NAMES = INS.NOTE_NAMES
_TEMPLATES = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "dom7": [0, 4, 7, 10], "min7": [0, 3, 7, 10], "sus": [0, 5, 7],
}
STAY = 0.35                    # Viterbi: the cost of changing chord between bars (in chroma-match units)


def _chroma_bars(y, beats, down0, period, per="bar"):
    import librosa
    C = librosa.feature.chroma_cqt(y=np.ascontiguousarray(y, dtype=np.float32), sr=RATE, hop_length=1024)
    n_bars = (len(beats) - down0) // 4
    out = []
    for b in range(n_bars):
        t0 = beats[down0 + b * 4]
        t1 = t0 + 4 * period
        f0, f1 = int(t0 * RATE / 1024), int(min(C.shape[1], t1 * RATE / 1024))
        out.append(C[:, f0:max(f1, f0 + 1)].mean(axis=1) if f0 < C.shape[1] else np.zeros(12))
    return np.array(out)


def chord_track(stems_mono, beats, down0, period, progress=None):
    beats = np.asarray(beats, dtype=np.float64)
    other = stems_mono.get("other")
    bass = stems_mono.get("bass")
    if other is None and bass is None:
        return []
    import librosa
    H = librosa.effects.hpss(other, margin=(1.0, 3.0), kernel_size=17)[0] if other is not None else None
    Co = _chroma_bars(H, beats, down0, period) if H is not None else None
    Cb = _chroma_bars(bass, beats, down0, period) if bass is not None else None
    n_bars = len(Co) if Co is not None else len(Cb)
    templates, labels = [], []
    for root in range(12):
        for q, ivs in _TEMPLATES.items():
            t = np.zeros(12)
            for iv in ivs:
                t[(root + iv) % 12] = 1.0
            t[root] = 1.3                                   # the root weighs more
            templates.append(t / np.linalg.norm(t))
            labels.append((root, q))
    T = np.array(templates)
    scores = np.zeros((n_bars, len(labels)))
    bass_pc = [None] * n_bars
    for b in range(n_bars):
        c = np.zeros(12)
        if Co is not None:
            c += Co[b] / (np.linalg.norm(Co[b]) + 1e-9)
        if Cb is not None:
            cb = Cb[b] / (np.linalg.norm(Cb[b]) + 1e-9)
            c += 0.6 * cb
            if cb.max() > 0.6:
                bass_pc[b] = int(np.argmax(cb))
        c = c / (np.linalg.norm(c) + 1e-9)
        scores[b] = T @ c
        if bass_pc[b] is not None:
            for k, (root, _q) in enumerate(labels):
                if root == bass_pc[b]:
                    scores[b, k] += 0.15                     # the bass names the root
    # Viterbi: stay unless the change is worth it
    n = len(labels)
    best = scores[0].copy()
    back = np.zeros((n_bars, n), dtype=int)
    for b in range(1, n_bars):
        stay = best
        change = best.max() - STAY
        arg_change = int(np.argmax(best))
        new = np.where(stay >= change, stay, change)
        back[b] = np.where(stay >= change, np.arange(n), arg_change)
        best = new + scores[b]
    path = [int(np.argmax(best))]
    for b in range(n_bars - 1, 0, -1):
        path.append(int(back[b, path[-1]]))
    path.reverse()
    out = []
    for b, k in enumerate(path):
        root, q = labels[k]
        out.append({"bar": b, "root": root, "quality": q, "bass": bass_pc[b],
                    "conf": round(float(scores[b, k]), 3), "name": NAMES[root] + ("m" if q.startswith("min") else "") + ("7" if "7" in q else "")})
    if progress:
        changes = sum(1 for i in range(1, len(out)) if out[i]["name"] != out[i - 1]["name"])
        progress(f"chords: {len(out)} bars, {changes} changes, {len(set(o['name'] for o in out))} distinct")
    return out
