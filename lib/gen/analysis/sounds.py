"""Sound identification: find the SOUNDS in a stem first, then find the
NOTES by explaining the stem with those sounds.

Drums (drum_sounds): every onset of the drum stem gets a timbre vector
(mel spectrum of its attack and body, decay, centroid); the onsets are
clustered, near-identical clusters merged, tiny ones dropped -> the
song's percussion palette (up to nine sounds). Each sound gets an
exemplar one-shot (its most isolated hit) and a drum SLOT (kick = the
lowest, snare = the mid sound on the backbeat, hat = the busiest high
one, the rest by register). Then template NMF over the whole stem gives
each sound's activation per 6 ms frame, and peak picking + folding onto
the 16th grid gives, per bar, a strength grid PER SOUND - so a conga is
not read as a snare and a shaker is not read as a hat, which is what the
three-band reader did.

Melodic (pluck_instruments): the "other" stem is split into a sustained
part (the drone / pad) and a plucked part (HPSS). Plucked onsets get a
pitch (pyin on the onset's own decay) and a timbre vector; clustered ->
instruments, each with a multisample from its own cleanest hit per pitch
and a note line on the bar grid. The sustained part becomes a pad
sample (its steadiest two bars) pitched by its dominant note; its chords
come from the chord reader. basic-pitch read the drone's harmonics as a
melody at every octave; the plucks - the actual tune - it mostly missed.

Everything here is evidence and material for the generator: the lines
seed motifs, the grids are templates, the samples are what plays."""
from __future__ import annotations

import os

import numpy as np

from lib.gen import RATE

reasons = []

N_MELS = 48
FMIN, FMAX = 30.0, 16000.0
HOP = 256                      # 5.8 ms: onset timing for the grids
N_FFT = 1024
DRUM_SLOT_ORDER = ("kick", "snare", "hat", "ohat", "shaker", "perc", "rim", "tom", "ride")
MAX_SOUNDS = 9
PITCH_CONF = 0.22              # harmonic-summation confidence for a plucked onset to count as a note
MERGE_COS = 0.93               # refined templates this alike (cosine of sqrt spectra) are one sound
NMF_REFINE = 50                # joint W/H iterations after the fixed-template fit


def _mono(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    y = x.mean(axis=1)
    if sr != RATE:
        idx = np.arange(0, len(y), sr / RATE)
        y = np.interp(idx, np.arange(len(y)), y).astype(np.float32)
    return y


def _mel(y, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
    import librosa
    return librosa.feature.melspectrogram(y=np.ascontiguousarray(y, dtype=np.float32), sr=RATE, n_fft=n_fft, hop_length=hop,
                                          n_mels=n_mels, fmin=FMIN, fmax=FMAX, power=1.0)


def _mel_freqs(n_mels=N_MELS):
    import librosa
    return librosa.mel_frequencies(n_mels=n_mels, fmin=FMIN, fmax=FMAX)


def onsets(y, delta=0.12, wait_s=0.03, hop=HOP):
    """Onset sample positions in y (librosa spectral flux, no backtracking)."""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=RATE, hop_length=hop)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=RATE, hop_length=hop, units="frames", backtrack=False,
                                    delta=delta, wait=max(1, int(wait_s * RATE / hop)))
    return [int(f * hop) for f in fr], env


def timbre(y, starts, attack_s=0.02, body_s=0.06, tail_s=0.11):
    """Per onset: (feature vector, centroid Hz, decay dB, energy dB).
    Feature = attack + body mel spectra (dB, relative to the hit's max)
    with the decay and log-centroid appended."""
    freqs = _mel_freqs()
    n_tail = int(tail_s * RATE)
    feats, meta = [], []
    for s in starts:
        seg = y[s:s + n_tail]
        if len(seg) < n_tail:
            seg = np.concatenate([seg, np.zeros(n_tail - len(seg), dtype=np.float32)])
        M = _mel(seg, hop=128)
        L = 20.0 * np.log10(M + 1e-6)
        fa = max(1, int(attack_s * RATE / 128))
        fb = max(fa + 1, int(body_s * RATE / 128))
        attack = L[:, :fa].mean(axis=1)
        body = L[:, fa:fb].mean(axis=1)
        tail = L[:, fb:].mean(axis=1) if L.shape[1] > fb else body
        ref = float(L.max())
        attack, body, tail = attack - ref, body - ref, tail - ref
        w = 10 ** (body / 20.0)
        centroid = float((w * freqs).sum() / max(float(w.sum()), 1e-9))
        decay = float((tail - body).mean())
        energy = float(20.0 * np.log10(np.sqrt(np.mean(seg[: int(body_s * RATE)] ** 2)) + 1e-9))
        feats.append(np.concatenate([attack, body, [decay / 6.0, np.log2(max(centroid, 30.0) / 30.0) * 6.0]]))
        meta.append((centroid, decay, energy))
    return np.array(feats, dtype=np.float32), meta


def cluster(X, k_max=8, min_share=0.012, merge_corr=0.985):
    """K-means at k_max, then merge clusters whose mean spectra correlate
    above merge_corr and drop those under min_share of the onsets.
    Returns labels (-1 = dropped) and the kept label ids."""
    from sklearn.cluster import KMeans
    if len(X) < 8:
        return np.zeros(len(X), dtype=int), [0]
    k = int(min(k_max, max(2, len(X) // 12)))
    km = KMeans(k, n_init=6, random_state=0).fit(X)
    labels = km.labels_.copy()
    merged = True
    while merged:
        merged = False
        ids = sorted(set(labels.tolist()))
        cents = {i: X[labels == i].mean(axis=0) for i in ids}
        best = None
        for a in ids:
            for b in ids:
                if b <= a:
                    continue
                ca, cb = cents[a][:-2], cents[b][:-2]
                r = float(np.corrcoef(ca, cb)[0, 1])
                d_extra = float(np.abs(cents[a][-2:] - cents[b][-2:]).max())
                if r >= merge_corr and d_extra < 1.0 and (best is None or r > best[0]):
                    best = (r, a, b)
        if best is not None:
            labels[labels == best[2]] = best[1]
            merged = True
    n = len(labels)
    keep = [i for i in sorted(set(labels.tolist())) if (labels == i).sum() >= max(6, min_share * n)]
    labels = np.where(np.isin(labels, keep), labels, -1)
    return labels, keep


def _exemplar(y, starts, energies, all_starts=None, min_gap_s=0.12, max_len_s=0.6):
    """The most isolated loud hit among starts -> (start, end) samples.
    Isolation is measured against ALL onsets of the stem (all_starts):
    the cut must end before the next hit of ANY sound, or the one-shot
    carries its neighbours."""
    order = np.argsort(starts)
    starts_s = [starts[i] for i in order]
    energies_s = [energies[i] for i in order]
    alls = np.asarray(sorted(all_starts) if all_starts is not None else starts_s, dtype=np.int64)
    best, best_score = None, -1e9
    n_pre = int(0.05 * RATE)
    for s, en in zip(starts_s, energies_s):
        if s < n_pre or s + int(0.06 * RATE) >= len(y):
            continue
        k = int(np.searchsorted(alls, s, side="right"))
        nxt = int(alls[k]) if k < len(alls) else len(y)
        gap = (nxt - s) / RATE
        if gap < min_gap_s:
            continue
        k0 = int(np.searchsorted(alls, s, side="left"))
        prev_gap = (s - int(alls[k0 - 1])) / RATE if k0 > 0 else 1.0
        pre = float(np.sqrt(np.mean(y[s - n_pre:s - 64] ** 2)) + 1e-6)
        peak = float(np.abs(y[s:s + int(0.03 * RATE)]).max())
        score = 20 * np.log10(peak / pre) + 0.5 * en + 8.0 * min(gap, 0.6) + 4.0 * min(prev_gap, 0.3)
        if score > best_score:
            best, best_score = (s, min(nxt - int(0.005 * RATE), s + int(max_len_s * RATE))), score
    if best is None:
        s = starts_s[0]
        best = (s, min(len(y), s + int(0.3 * RATE)))
    s, e = best
    seg = y[s:e]
    w = int(0.005 * RATE)
    if len(seg) > int(0.06 * RATE) + w:
        env = np.sqrt(np.convolve(seg.astype(np.float64) ** 2, np.ones(w) / w, mode="same"))     # 5 ms rms envelope
        pk = float(env[: int(0.03 * RATE)].max()) + 1e-9
        thr = pk * 10 ** (-42 / 20)
        below = np.where(env[int(0.06 * RATE):] < thr)[0]
        if len(below):
            e = s + int(0.06 * RATE) + int(below[0]) + int(0.01 * RATE)
    return s, min(e, len(y))


def _cut(y, s, e, path, peak=0.9):
    import soundfile as sf
    seg = y[s:e].astype(np.float32).copy()
    if len(seg) < 64:
        return None
    fi = min(int(0.002 * RATE), len(seg) // 4)
    seg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    fo = min(int(0.02 * RATE), len(seg) // 3)
    seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    seg = seg / max(float(np.abs(seg).max()), 1e-6) * peak
    sf.write(path, seg, RATE, subtype="PCM_16")
    return path


def _clean_templates(W, sounds, same=0.12, mixed=0.6, passes=2):
    """W (n_mels, K) linear-magnitude templates from onset clusters. For
    each template, a non-negative fit on the OTHER templates (in the
    sqrt-magnitude domain, so a few loud bands do not decide): residual
    energy < `same` -> it is another sound (merge into the best-fitting
    one); < `mixed` -> it is a coincidence (keep the residual as the
    sound). Returns (W, sounds) with merged entries combined."""
    from scipy.optimize import nnls
    W = np.array(W, dtype=np.float64)
    sounds = list(sounds)
    for _ in range(passes):
        i = 0
        while i < len(sounds) and len(sounds) > 1:
            others = [j for j in range(len(sounds)) if j != i]
            A = np.sqrt(W[:, others])
            target = np.sqrt(W[:, i])
            coef, _ = nnls(A, target)
            fit = A @ coef
            res_c = target - fit
            r = float((res_c ** 2).sum() / max(float((target ** 2).sum()), 1e-12))
            res = np.maximum(res_c, 0.0) ** 2
            j = others[int(np.argmax(coef))] if coef.sum() > 0 else None
            same_kind = (j is not None and abs(sounds[i]["decay"] - sounds[j]["decay"]) <= 4.0
                         and max(sounds[i]["centroid"], sounds[j]["centroid"]) <= 1.6 * max(30.0, min(sounds[i]["centroid"], sounds[j]["centroid"])))
            if r < same and same_kind:
                sounds[j]["idx"] = np.concatenate([sounds[j]["idx"], sounds[i]["idx"]])
                sounds[j]["n"] = int(len(sounds[j]["idx"]))
                if sounds[i].get("level", -99) > sounds[j].get("level", -99):
                    sounds[j]["level"] = sounds[i]["level"]
                W = np.delete(W, i, axis=1)
                del sounds[i]
                continue
            if r < mixed and coef.sum() > 0:
                W[:, i] = np.maximum(res, 0.0) + 1e-9
            i += 1
    return W, sounds


def nmf_activations(V, W, iters=40, refine=12):
    """Template NMF (KL): V (n_mels, T) ~ W (n_mels, K) H (K, T). W starts
    from the clustered onset spectra; `refine` joint iterations let the
    templates shed what belongs to a sound that plays at the same time
    (a clap's template learned from kick+clap onsets carries the kick's
    low end until the kick template, active there too, takes it back).
    Returns (H, W)."""
    W = np.maximum(np.asarray(W, dtype=np.float64), 1e-9)
    W = W / W.sum(axis=0, keepdims=True)
    V = np.maximum(np.asarray(V, dtype=np.float64), 1e-9)
    K = W.shape[1]
    H = np.full((K, V.shape[1]), float(V.mean()) / K + 1e-6)
    ones = np.ones_like(V)
    for it in range(iters + refine):
        WH = np.maximum(W @ H, 1e-9)
        H *= (W.T @ (V / WH)) / np.maximum(W.T @ ones, 1e-9)
        if it >= iters:
            WH = np.maximum(W @ H, 1e-9)
            W *= ((V / WH) @ H.T) / np.maximum(ones @ H.T, 1e-9)
            W = W / np.maximum(W.sum(axis=0, keepdims=True), 1e-9)
    return H, W


def _hits_from_activation(h, hop, thr_rel=0.25, min_gap_s=0.04):
    """Peak-pick one sound's activation -> [(sample, strength 0..1)]."""
    if len(h) < 3:
        return []
    ref = float(np.percentile(h, 99.5)) + 1e-9
    d = np.diff(h, prepend=h[0])
    hits = []
    last = -10 ** 9
    gap = max(1, int(min_gap_s * RATE / hop))
    thr = thr_rel * ref
    for i in range(1, len(h) - 1):
        if h[i] >= thr and h[i] >= h[i - 1] and h[i] >= h[i + 1] and d[i] > 0 and i - last >= gap:
            hits.append((i * hop, float(min(1.0, h[i] / ref))))
            last = i
    return hits


def grids_from_hits(hits_by_slot, bars, steps=16):
    """[(sample, strength)] per slot -> per bar {slot: [steps floats]} (max strength per step)."""
    bars = np.asarray(bars, dtype=np.float64) * RATE
    out = [{slot: [0.0] * steps for slot in hits_by_slot} for _ in range(max(0, len(bars) - 1))]
    for slot, hits in hits_by_slot.items():
        for s, v in hits:
            k = int(np.searchsorted(bars, s, side="right") - 1)
            if not (0 <= k < len(bars) - 1):
                continue
            step = int(round((s - bars[k]) / (bars[k + 1] - bars[k]) * steps))
            if step >= steps:
                if k + 1 < len(out):
                    out[k + 1][slot][0] = max(out[k + 1][slot][0], v)
                continue
            out[k][slot][step] = max(out[k][slot][step], v)
    for g in out:
        for slot in g:
            g[slot] = [round(float(x), 3) for x in g[slot]]
    return out


def _assign_slots(sounds, grids_raw):
    """Sounds (centroid, decay, n, dur) -> drum slot per sound."""
    free = list(DRUM_SLOT_ORDER)
    assign = {}
    order = sorted(range(len(sounds)), key=lambda i: -sounds[i]["n"])
    # kick: the sound whose (refined) template lives below 200 Hz the most, if any does
    n_all = max(1, sum(sd["n"] for sd in sounds))
    low = [i for i in range(len(sounds)) if sounds[i].get("low", 0.0) >= 0.25 and sounds[i]["n"] >= 0.03 * n_all]
    if not low:
        low = [i for i in range(len(sounds)) if sounds[i]["centroid"] < 220.0]
    if low:
        i = max(low, key=lambda i: sounds[i].get("low", 0.0) * np.sqrt(sounds[i]["n"]))     # the low sound that plays the most
        assign[i] = "kick"; free.remove("kick")
    # snare: the mid sound whose hits sit on the backbeat (steps 4, 12) the most
    mids = [i for i in range(len(sounds)) if i not in assign and 220.0 <= sounds[i]["centroid"] < 3500.0]
    if mids and grids_raw:
        def backbeat(i):
            prof = np.zeros(16)
            for g in grids_raw:
                prof += np.asarray(g[i])
            tot = prof.sum() + 1e-9
            return (prof[4] + prof[12]) / tot - (prof[0] + prof[8]) / tot
        i = max(mids, key=backbeat)
        if backbeat(i) > 0.05:
            assign[i] = "snare"; free.remove("snare")
    # hat: the busiest high short sound
    highs = [i for i in range(len(sounds)) if i not in assign and sounds[i]["centroid"] >= 2500.0]
    if highs:
        i = max(highs, key=lambda i: sounds[i]["n"])
        assign[i] = "hat"; free.remove("hat")
    for i in order:
        if i in assign or not free:
            continue
        c, dur, dec = sounds[i]["centroid"], sounds[i]["dur"], sounds[i]["decay"]
        if c >= 2500.0:
            pref = ["shaker", "ohat", "ride", "perc", "rim"] if dur < 0.12 else ["ohat", "ride", "shaker", "perc", "rim"]
        elif c >= 700.0:
            pref = ["perc", "rim", "snare", "shaker", "tom"]
        elif c >= 220.0:
            pref = ["tom", "perc", "rim", "snare"]
        else:
            pref = ["tom", "perc", "kick"]
        for p in pref + free:
            if p in free:
                assign[i] = p; free.remove(p)
                break
    return assign


def drum_sounds(drums_wav, out_dir, bars=None, progress=None):
    """-> {"kit": {slot: wav}, "sounds": [{slot, file, centroid, decay, n, dur}],
    "grids": [per bar {slot: [16 strengths]}] (when bars are given)}."""
    os.makedirs(out_dir, exist_ok=True)
    y = _mono(drums_wav)
    if len(y) < RATE:
        return {"kit": {}, "sounds": [], "grids": []}
    starts, env = onsets(y)
    if len(starts) < 8:
        reasons.append("drum stem: too few onsets to identify sounds")
        return {"kit": {}, "sounds": [], "grids": []}
    X, meta = timbre(y, starts)
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    labels, keep = cluster(Xn, k_max=min(8, MAX_SOUNDS))
    if progress:
        progress(0.4, "drum sounds clustered")
    sounds = []
    for lab in keep:
        idx = np.where(labels == lab)[0]
        cen = float(np.median([meta[i][0] for i in idx]))
        dec = float(np.median([meta[i][1] for i in idx]))
        lvl = float(np.percentile([meta[i][2] for i in idx], 75))          # how loud this sound's hits are (dBFS)
        s, e = _exemplar(y, [starts[i] for i in idx], [meta[i][2] for i in idx], all_starts=starts)
        sounds.append({"idx": idx, "centroid": round(cen, 1), "decay": round(dec, 1), "n": int(len(idx)),
                       "dur": round((e - s) / RATE, 3), "cut": (s, e), "level": lvl})
    top = max(sd["level"] for sd in sounds)
    for sd in sounds:
        sd["level_db"] = round(sd["level"] - top, 1)                         # relative to the loudest sound
    # templates: the mean attack+body mel spectrum of each sound's hits (linear magnitude)
    V = _mel(y)
    fr = lambda s: int(s / HOP)
    W = []
    for sd in sounds:
        cols = []
        for i in sd["idx"][:300]:
            f0 = fr(starts[i])
            cols.append(V[:, f0:f0 + 6].mean(axis=1))
        W.append(np.median(cols, axis=0))              # the median: a hit that coincided with another sound does not pull it
    W = np.array(W).T
    freqs = _mel_freqs()
    for i, sd in enumerate(sounds):
        col = W[:, i]
        sd["low"] = round(float(col[freqs < 200.0].sum() / max(float(col.sum()), 1e-9)), 3)     # low-band share of the template
    # onsets that coincide (a kick under a clap) make clusters whose template is a SUM of sounds: explain each
    # template by the others - a template that is (nearly) another one of the same kind merges into it, one
    # that is another plus something keeps only the something
    W, sounds = _clean_templates(W, sounds)
    H, W_ref = nmf_activations(V, W, refine=0)
    if progress:
        progress(0.7, "drum activations")
    hits = {i: _hits_from_activation(H[i], HOP) for i in range(len(sounds))}
    grids_raw = grids_from_hits(hits, bars) if bars is not None and len(bars) > 1 else []
    assign = _assign_slots(sounds, grids_raw)
    kit, out_sounds = {}, []
    for i, sd in enumerate(sounds):
        slot = assign.get(i)
        if slot is None:
            continue
        s, e = sd["cut"]
        path = _cut(y, s, e, os.path.join(out_dir, f"{slot}_song.wav"))
        if path is None:
            continue
        kit[slot] = path
        out_sounds.append({"slot": slot, "file": path, "centroid": sd["centroid"], "decay": sd["decay"], "n": sd["n"], "dur": sd["dur"],
                           "level_db": sd["level_db"], "low": sd.get("low", 0.0)})
    grids = [{assign[i]: g[i] for i in g if i in assign} for g in grids_raw]
    return {"kit": kit, "sounds": out_sounds, "grids": grids, "n_onsets": len(starts),
            "kit_db": {sd["slot"]: sd["level_db"] for sd in out_sounds}}


# -- melodic -------------------------------------------------------------------
_HPS_N = 8192


def _pitch_hps(y, s, midi_lo=40, midi_hi=96, pre_s=0.2, post_s=0.13):
    """Pitch of the sound that STARTS at sample s: the spectrum after the
    onset minus the spectrum before it (the drone / pad / everything
    already sounding), then harmonic summation over a semitone grid.
    Returns (midi or None, confidence 0..1)."""
    a0, a1 = max(0, s - int(pre_s * RATE)), max(0, s - int(0.005 * RATE))
    b0, b1 = s + int(0.005 * RATE), min(len(y), s + int(post_s * RATE))
    if b1 - b0 < 1024 or a1 - a0 < 1024:
        return None, 0.0
    pre, post = y[a0:a1].astype(np.float64), y[b0:b1].astype(np.float64)
    Pre = np.abs(np.fft.rfft(pre * np.hanning(len(pre)), _HPS_N)) / len(pre)
    Post = np.abs(np.fft.rfft(post * np.hanning(len(post)), _HPS_N)) / len(post)
    R = np.maximum(Post - 1.2 * Pre, 0.0)
    freqs = np.fft.rfftfreq(_HPS_N, 1.0 / RATE)
    band = (freqs >= 60.0) & (freqs <= 5000.0)
    total = float(R[band].sum()) + 1e-12
    if total < 1e-7:
        return None, 0.0
    df = freqs[1]

    def peak(f):
        i = int(round(f / df))
        lo, hi = max(0, i - 2), min(len(R), i + 3)
        return float(R[lo:hi].max()) if hi > lo else 0.0

    best, best_sal, sal = None, 0.0, {}
    for m in range(midi_lo, midi_hi + 1):
        f0 = 440.0 * 2 ** ((m - 69) / 12.0)
        v = sum(peak(h * f0) / (h ** 0.6) for h in range(1, 7))
        sal[m] = v
        if v > best_sal:
            best, best_sal = m, v
    if best is None:
        return None, 0.0
    # octave check: energy on the odd harmonics of the candidate says it is the true fundamental
    f0 = 440.0 * 2 ** ((best - 69) / 12.0)
    odd = peak(f0) + peak(3 * f0) + peak(5 * f0)
    even = peak(2 * f0) + peak(4 * f0) + peak(6 * f0)
    if odd < 0.3 * even and best + 12 <= midi_hi:
        best = best + 12
    conf = float(best_sal / total)
    return best, min(1.0, conf * 3.0)


def _pitch_of(seg, fmin=70.0, fmax=2200.0):
    """MIDI pitch of a short decaying segment via pyin (median of voiced frames), or None."""
    import librosa
    if len(seg) < 2048:
        return None, 0.0
    try:
        f0, voiced, prob = librosa.pyin(seg, fmin=fmin, fmax=fmax, sr=RATE, frame_length=2048, hop_length=256)
    except Exception:  # noqa: BLE001
        return None, 0.0
    ok = voiced & np.isfinite(f0)
    if ok.sum() < 2:
        return None, 0.0
    midi = 69 + 12 * np.log2(np.median(f0[ok]) / 440.0)
    return int(round(midi)), float(ok.mean())


def pluck_instruments(other_wav, bars, out_dir, max_instruments=3, max_pitches=24, progress=None):
    """-> {"instruments": [{"bank": [{"file", "base_midi"}], "line": [(bar, step, midi, dur_steps, vel)], "n", "centroid"}],
    "pad": {"file", "base_midi", "seconds"} | None, "n_onsets"}"""
    import librosa
    import soundfile as sf
    os.makedirs(out_dir, exist_ok=True)
    y = _mono(other_wav)
    bars = np.asarray(bars, dtype=np.float64)
    if len(y) < 2 * RATE or len(bars) < 2:
        return {"instruments": [], "pad": None, "n_onsets": 0}
    Hh, P = librosa.effects.hpss(y, margin=(1.0, 3.0))
    starts, env = onsets(P, delta=0.2, wait_s=0.05)
    if progress:
        progress(0.2, f"{len(starts)} plucked onsets")
    out = {"instruments": [], "pad": None, "n_onsets": len(starts)}
    # the pad: the steadiest loud two bars of the sustained part, pitched by pyin
    bar_len = float(np.median(np.diff(bars)))
    best, best_score = None, -1e9
    frame = int(0.1 * RATE)
    for k in range(0, len(bars) - 2, 2):
        a, b = int(bars[k] * RATE), int(bars[k + 2] * RATE)
        seg = Hh[a:b]
        if len(seg) < RATE:
            continue
        rms = np.array([np.sqrt(np.mean(seg[i:i + frame] ** 2)) for i in range(0, len(seg) - frame, frame)]) + 1e-9
        lvl = 20 * np.log10(rms.mean())
        steadiness = -float(np.std(20 * np.log10(rms)))
        n_pluck = int(np.sum((np.asarray(starts) >= a) & (np.asarray(starts) < b)))    # plucked hits inside: not a pad
        score = lvl + 2.0 * steadiness - 1.5 * n_pluck
        if score > best_score:
            best, best_score = (a, b), score
    if best is not None:
        a, b = best
        seg = Hh[a:b].astype(np.float32).copy()
        midi, conf = _pitch_of(seg[: 4 * RATE], fmin=40.0, fmax=600.0)
        if midi is not None and conf > 0.3:
            f = int(0.05 * RATE)
            seg[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            seg[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
            seg = seg / max(float(np.abs(seg).max()), 1e-6) * 0.8
            path = os.path.join(out_dir, "pad_song.wav")
            sf.write(path, seg, RATE, subtype="PCM_16")
            out["pad"] = {"file": path, "base_midi": int(midi), "seconds": round(len(seg) / RATE, 2),
                          "bar": int(np.searchsorted(bars, a / RATE, side="right") - 1)}
    if len(starts) < 8:
        return out
    # timbre per plucked onset on the plucked part; pitch on the full stem with what was already
    # sounding (the drone) subtracted
    X, meta = timbre(P, starts, attack_s=0.02, body_s=0.08, tail_s=0.16)
    pitches = [_pitch_hps(y, s) for s in starts]
    if progress:
        progress(0.6, "plucks pitched")
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    labels, keep = cluster(Xn, k_max=6, min_share=0.03)
    insts = []
    for lab in keep:
        idx = np.where(labels == lab)[0]
        pitched = [i for i in idx if pitches[i][0] is not None and pitches[i][1] >= PITCH_CONF]
        if len(pitched) < max(6, 0.3 * len(idx)):
            continue                                   # a non-pitched cluster (percussion leaked into the stem)
        cen = float(np.median([meta[i][0] for i in idx]))
        insts.append({"idx": idx, "pitched": pitched, "centroid": cen})
    insts.sort(key=lambda d: -len(d["pitched"]))
    for n_i, inst in enumerate(insts[:max_instruments]):
        # bank: the most isolated hit per pitch, cut from the ORIGINAL stem (the tone with its body)
        by_pitch = {}
        for i in inst["pitched"]:
            m = pitches[i][0]
            s = starts[i]
            nxt = starts[i + 1] if i + 1 < len(starts) else len(y)     # the next plucked onset of ANY instrument
            gap = (nxt - s) / RATE
            score = meta[i][2] + 8.0 * min(gap, 0.6) + 4.0 * pitches[i][1]
            if m not in by_pitch or score > by_pitch[m][0]:
                by_pitch[m] = (score, i, min(gap, 0.8))
        top = sorted(by_pitch.items(), key=lambda kv: -kv[1][0])[:max_pitches]
        bank = []
        for m, (score, i, gap) in sorted(top):
            s = starts[i]
            e = min(len(y), s + int(max(0.12, gap - 0.005) * RATE))       # ends before the next hit
            path = _cut(y, s, e, os.path.join(out_dir, f"pluck{n_i}_{m}.wav"), peak=0.8)
            if path:
                bank.append({"file": path, "base_midi": int(m)})
        line = []
        for i in inst["pitched"]:
            s = starts[i] / RATE
            k = int(np.searchsorted(bars, s, side="right") - 1)
            if not (0 <= k < len(bars) - 1):
                continue
            step = int(round((s - bars[k]) / (bars[k + 1] - bars[k]) * 16))
            if step >= 16:
                k, step = k + 1, 0
            vel = float(np.clip(0.4 + (meta[i][2] + 30.0) / 30.0, 0.3, 1.0))
            line.append((int(k), int(step), int(pitches[i][0]), 1.0, round(vel, 2)))
        dedup = {}
        for bar, step, midi, dur, vel in line:
            if (bar, step) not in dedup or vel > dedup[(bar, step)][4]:
                dedup[(bar, step)] = (bar, step, midi, dur, vel)
        out["instruments"].append({"bank": bank, "line": sorted(dedup.values()), "n": int(len(inst["pitched"])),
                                   "centroid": round(inst["centroid"], 1)})
    if progress:
        progress(1.0, f"{len(out['instruments'])} plucked instruments")
    return out
