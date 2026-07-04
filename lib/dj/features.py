"""Offline per-track music analysis for the DJ subsystem.

Everything the DJ brain needs to mix two songs well, computed once per file
by the library scanner (tools/dj_scan.py) and stored in the sqlite library:

    tempo + beat grid   windowed autocorrelation -> comb-refined period/phase
                        per stable-tempo segment (handles tempo changes)
    downbeat            bar phase scored by bass-onset + novelty at beats
    key                 Krumhansl-Schmuckler on energy-weighted chroma -> Camelot
    structure           beat-synchronous self-similarity matrix -> checkerboard
                        novelty -> sections with fingerprints (energy, busyness,
                        vocalness, repetitiveness, spectral balance)
    loops               downbeat-aligned 8/16/32-beat windows inside repetitive
                        low-vocal sections, scored by half-vs-half similarity
    mix points          section boundaries scored for entering/leaving the track
    loudness            gain to a common target so mixed tracks sit level
    live cross-check    the REAL BeatDetector + AudioStructure run over the
                        track (what the visuals will actually see at showtime)

The framing pipeline (4096 Hann FFT -> 32 log bands -> 0.95/0.05 smoothing at
40 fps) replicates MicrophoneAnalyzer exactly, so offline conclusions transfer
to the live system. Pure functions, multiprocessing-safe, no GL/GLFW imports.
"""
import json
import math
import os

import numpy as np

ANALYSIS_VERSION = 1

RATE = 44100
FPS = 40
CHUNK = 4096
HOP = RATE // FPS               # 1102 samples between analysis frames
NUM_BANDS = 32

MIN_BPM, MAX_BPM = 65.0, 200.0
TEMPO_WIN_S = 20.0              # autocorr window
TEMPO_HOP_S = 10.0
SSM_KERNEL_BEATS = 16           # checkerboard novelty kernel (beats)
# Spectral-flux peaks LEAD the true transient: energy starts rising while
# the 4096-sample Hann window is still approaching the hit. Constant for our
# fixed framing; calibrated on synthetic kicks (see _dj_features_test).
ONSET_LATENCY_S = 0.028
LOUDNESS_TARGET_DBFS = -14.0

_KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_NOTE_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


# --------------------------------------------------------------------------
# Decode + metadata
# --------------------------------------------------------------------------

def decode_file(path):
    """Decode any audio file to mono float32 at 44100 Hz.

    miniaudio handles mp3/wav/flac/ogg natively; PyAV catches the rest
    (m4a/aac/opus/wma...)."""
    try:
        import miniaudio
        dec = miniaudio.decode_file(
            path, output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1, sample_rate=RATE)
        return np.asarray(dec.samples, dtype=np.float32)
    except Exception:
        pass
    import av
    from av.audio.resampler import AudioResampler
    resampler = AudioResampler(format="flt", layout="mono", rate=RATE)
    chunks = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):        # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("no audio decoded")
    return np.concatenate(chunks).astype(np.float32)


def decode_file_stereo(path):
    """Stereo variant for playback decks (Phase B)."""
    try:
        import miniaudio
        dec = miniaudio.decode_file(
            path, output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=2, sample_rate=RATE)
        return np.asarray(dec.samples, dtype=np.float32).reshape(-1, 2)
    except Exception:
        pass
    import av
    from av.audio.resampler import AudioResampler
    resampler = AudioResampler(format="flt", layout="stereo", rate=RATE)
    chunks = []
    with av.open(path) as container:
        for frame in container.decode(container.streams.audio[0]):
            for out in resampler.resample(frame):
                a = out.to_ndarray()
                chunks.append(a.T if a.shape[0] == 2 else
                              np.repeat(a.reshape(-1, 1), 2, axis=1))
        for out in resampler.resample(None):
            a = out.to_ndarray()
            chunks.append(a.T if a.shape[0] == 2 else
                          np.repeat(a.reshape(-1, 1), 2, axis=1))
    if not chunks:
        raise ValueError("no audio decoded")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def read_metadata(path):
    """Title/artist/album from container tags (PyAV); filename fallback."""
    title = os.path.splitext(os.path.basename(path))[0]
    artist = album = ""
    try:
        import av
        with av.open(path) as c:
            meta = {k.lower(): v for k, v in (c.metadata or {}).items()}
            for s in c.streams:
                for k, v in (s.metadata or {}).items():
                    meta.setdefault(k.lower(), v)
        title = meta.get("title", title) or title
        artist = meta.get("artist", meta.get("album_artist", "")) or ""
        album = meta.get("album", "") or ""
    except Exception:
        pass
    return {"title": title, "artist": artist, "album": album}


# --------------------------------------------------------------------------
# Framing: the live analyzer's band/chroma pipeline, vectorized
# --------------------------------------------------------------------------

def _band_matrix():
    freq_bins = np.fft.rfftfreq(CHUNK, 1.0 / RATE)
    edges = np.logspace(np.log10(40), np.log10(16000), NUM_BANDS + 1)
    mat = np.zeros((NUM_BANDS, len(freq_bins)))
    for i in range(NUM_BANDS):
        m = (freq_bins >= edges[i]) & (freq_bins < edges[i + 1])
        if not np.any(m):
            m = (freq_bins >= edges[i] * 0.9) & (freq_bins < edges[i + 1] * 1.1)
        if np.any(m):
            mat[i, m] = 1.0 / np.count_nonzero(m)
    return mat, freq_bins


_BAND_MAT, _FREQ_BINS = _band_matrix()
_HANN = np.hanning(CHUNK)
_CHROMA_MASK = (_FREQ_BINS >= 80.0) & (_FREQ_BINS <= 5000.0)
# NOTE: A-origin pitch classes, exactly like MicrophoneAnalyzer.get_chroma.
_CHROMA_PCS = (np.round(12.0 * np.log2(np.maximum(_FREQ_BINS[_CHROMA_MASK], 1.0)
                                       / 440.0)) % 12).astype(np.int64)


def frame_track(samples):
    """Per-frame features at 40 fps: smoothed band powers [T,32] (identical
    to the live analyzer's raw_bands) and A-origin chroma [T,12]."""
    from scipy.signal import lfilter
    n_frames = max(0, (len(samples) - CHUNK) // HOP + 1)
    if n_frames == 0:
        raise ValueError("track shorter than one analysis window")
    bands = np.empty((n_frames, NUM_BANDS), dtype=np.float64)
    chroma = np.empty((n_frames, 12), dtype=np.float64)
    batch = 512
    for b0 in range(0, n_frames, batch):
        b1 = min(b0 + batch, n_frames)
        idx = (np.arange(b0, b1)[:, None] * HOP + np.arange(CHUNK)[None, :])
        wins = samples[idx].astype(np.float64) * _HANN[None, :]
        mags = np.abs(np.fft.rfft(wins, axis=1))
        mags[:, 0] = 0.0
        power = mags ** 2
        bands[b0:b1] = np.sqrt(power @ _BAND_MAT.T)
        cp = power[:, _CHROMA_MASK]
        ch = np.zeros((b1 - b0, 12))
        for pc in range(12):
            sel = _CHROMA_PCS == pc
            if np.any(sel):
                ch[:, pc] = cp[:, sel].sum(axis=1)
        tot = ch.sum(axis=1, keepdims=True)
        chroma[b0:b1] = np.where(tot > 1e-12, ch / np.maximum(tot, 1e-12), 0.0)
    # The analyzer's light temporal smoothing: y[t] = 0.95 x[t] + 0.05 y[t-1]
    bands = lfilter([0.95], [1.0, -0.05], bands, axis=0)
    return bands, chroma


def _onset_channels(bands):
    """Normalized positive spectral flux: broadband / bass / percussive,
    plus an L2 novelty curve on the log band envelope."""
    mean = np.maximum(bands.mean(axis=0, keepdims=True), 1e-10)
    nb = bands / mean
    flux = np.maximum(0.0, np.diff(nb, axis=0, prepend=nb[:1]))
    onset_broad = flux[:, 0:24].mean(axis=1)
    onset_bass = flux[:, 0:6].mean(axis=1)
    onset_perc = flux[:, 6:16].mean(axis=1)
    logb = np.log1p(bands / mean)
    novelty = np.linalg.norm(np.diff(logb, axis=0, prepend=logb[:1]), axis=1)
    return onset_broad, onset_bass, onset_perc, novelty


# --------------------------------------------------------------------------
# Tempo + beat grid
# --------------------------------------------------------------------------

def _comb_score(onset, period_f, phase_f, start_f, end_f):
    """Mean interpolated onset at beat positions phase + k*period (frames)."""
    if period_f <= 1.0:
        return 0.0, 0.0
    ks = np.arange(0, int((end_f - start_f - phase_f) / period_f) + 1)
    pos = start_f + phase_f + ks * period_f
    pos = pos[pos < end_f - 1]
    if len(pos) < 4:
        return 0.0, 0.0
    i = pos.astype(np.int64)
    fr = pos - i
    vals = onset[i] * (1 - fr) + onset[i + 1] * fr
    return float(vals.mean()), float(len(pos))


def _best_phase(onset, period_f, start_f, end_f, n_phase=64):
    best_s, best_p = -1.0, 0.0
    for p in np.linspace(0, period_f, n_phase, endpoint=False):
        s, _ = _comb_score(onset, period_f, p, start_f, end_f)
        if s > best_s:
            best_s, best_p = s, p
    return best_p, best_s

def _refine_period(onset, period_f, start_f, end_f, span=0.02, steps=25):
    cands = period_f * np.linspace(1 - span, 1 + span, steps)
    scores = []
    for pf in cands:
        p, s = _best_phase(onset, pf, start_f, end_f, n_phase=32)
        scores.append(s)
    k = int(np.argmax(scores))
    best = cands[k]
    if 0 < k < steps - 1:                     # parabolic touch-up
        y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            best += 0.5 * (y0 - y2) / denom * (cands[1] - cands[0])
    return float(best)


def _polish_grid(onset, first_beat_f, period_f, start_f, end_f):
    """Millisecond-precision grid: locate the local onset peak near each
    predicted beat (parabolic sub-frame), then least-squares fit
    t_k = phase + k * period over the confident beats. Two rounds."""
    for _ in range(2):
        n_beats = int((end_f - first_beat_f) / period_f)
        if n_beats < 8:
            return first_beat_f, period_f
        ks, ts, ws = [], [], []
        half = max(2, int(period_f * 0.3))
        for k in range(n_beats):
            c = first_beat_f + k * period_f
            i0 = int(max(start_f, c - half))
            i1 = int(min(end_f - 1, c + half + 1))
            if i1 - i0 < 3:
                continue
            seg = onset[i0:i1]
            j = int(np.argmax(seg))
            peak = seg[j]
            if peak <= 0:
                continue
            pos = i0 + j
            if 0 < j < len(seg) - 1:            # parabolic sub-frame
                y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]
                den = y0 - 2 * y1 + y2
                if abs(den) > 1e-12:
                    pos += 0.5 * (y0 - y2) / den
            ks.append(k)
            ts.append(pos)
            ws.append(peak)
        if len(ks) < 8:
            return first_beat_f, period_f
        ks, ts, ws = np.array(ks, float), np.array(ts), np.array(ws)
        thr = np.percentile(ws, 50)
        keep = ws >= thr * 0.8                  # confident beats only
        if keep.sum() < 8:
            keep = np.ones(len(ks), bool)
        w = ws[keep]
        A = np.stack([np.ones(keep.sum()), ks[keep]], axis=1) * w[:, None]
        coef, *_ = np.linalg.lstsq(A, ts[keep] * w, rcond=None)
        first_beat_f, period_f = float(coef[0]), float(coef[1])
    return first_beat_f, period_f


def _window_bpm(onset, start_f, end_f):
    """Best BPM for one window: autocorr peaks -> octave candidates -> comb."""
    x = onset[start_f:end_f] - onset[start_f:end_f].mean()
    if len(x) < FPS * 4 or np.all(x == 0):
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / max(ac[0], 1e-12)
    lo = int(math.floor(60.0 / MAX_BPM * FPS))
    hi = int(math.ceil(60.0 / MIN_BPM * FPS))
    hi = min(hi, len(ac) - 2)
    if hi <= lo:
        return 0.0, 0.0
    seg = ac[lo:hi + 1]
    order = np.argsort(seg)[::-1][:6]
    cands = set()
    for k in order:
        lag = lo + int(k)
        for mult in (1.0, 2.0, 0.5):
            bpm = 60.0 * FPS / (lag * mult)
            if MIN_BPM <= bpm <= MAX_BPM:
                cands.add(round(bpm, 2))
    best_bpm, best_score = 0.0, -1.0
    for bpm in cands:
        # Integer autocorr lags quantize the period; a true period between
        # lags drifts off-beat across the window and under-scores, letting
        # wrong-but-integer periods win. Refine locally BEFORE judging.
        pf = _refine_period(onset, 60.0 / bpm * FPS, start_f, end_f,
                            span=0.03, steps=13)
        rbpm = 60.0 * FPS / pf
        if not (MIN_BPM <= rbpm <= MAX_BPM):
            continue
        _, s = _best_phase(onset, pf, start_f, end_f, n_phase=32)
        prior = math.exp(-0.5 * (math.log2(rbpm / 120.0) / 0.9) ** 2)
        s *= (0.7 + 0.3 * prior)
        if s > best_score:
            best_bpm, best_score = rbpm, s
    return best_bpm, best_score


def estimate_beat_grid(onset):
    """Windowed tempo -> stable segments -> comb-refined grid per segment.

    Returns (grid, bpm, bpm_conf, beats) where grid is a list of
    {start_s, end_s, period_s, first_beat_s} and beats is every beat time."""
    n = len(onset)
    win = min(int(TEMPO_WIN_S * FPS), n)
    hop = max(1, int(TEMPO_HOP_S * FPS))
    wins = []
    for s0 in range(0, max(1, n - win + 1), hop):
        bpm, score = _window_bpm(onset, s0, min(s0 + win, n))
        if bpm > 0:
            wins.append((s0, min(s0 + win, n), bpm, score))
    if not wins:
        return [], 0.0, 0.0, np.array([])

    # Group consecutive windows agreeing within 2.5% into segments.
    groups, cur = [], [wins[0]]
    for w in wins[1:]:
        if abs(w[2] - np.median([c[2] for c in cur])) / w[2] < 0.025:
            cur.append(w)
        else:
            groups.append(cur)
            cur = [w]
    groups.append(cur)
    # Absorb single-window blips into the larger neighbour.
    merged = []
    for g in groups:
        if len(g) == 1 and merged and len(merged[-1]) >= 2:
            merged[-1].extend(g)
        else:
            merged.append(g)
    groups = [g for g in merged if g]

    grid, all_beats = [], []
    for gi, g in enumerate(groups):
        seg_s = g[0][0]
        seg_e = g[-1][1] if gi == len(groups) - 1 else groups[gi + 1][0][0]
        seg_e = min(seg_e, n)
        bpm0 = float(np.median([w[2] for w in g]))
        pf = _refine_period(onset, 60.0 / bpm0 * FPS, seg_s, seg_e)
        phase, score = _best_phase(onset, pf, seg_s, seg_e, n_phase=96)
        fb_f, pf = _polish_grid(onset, seg_s + phase, pf, seg_s, seg_e)
        period_s = pf / FPS
        first_beat = fb_f / FPS + ONSET_LATENCY_S
        while first_beat - period_s >= seg_s / FPS:
            first_beat -= period_s
        grid.append({"start_s": seg_s / FPS, "end_s": seg_e / FPS,
                     "period_s": period_s, "first_beat_s": first_beat,
                     "bpm": 60.0 / period_s, "score": score})
        t = first_beat
        while t < seg_e / FPS:
            all_beats.append(t)
            t += period_s

    # Track BPM = the longest segment's; confidence blends window agreement
    # (how many windows voted for it) with comb sharpness vs the mean onset.
    main = max(grid, key=lambda s: s["end_s"] - s["start_s"])
    bpm = main["bpm"]
    votes = [w[2] for w in wins]
    agree = np.mean([1.0 if min(abs(v - bpm) / bpm, abs(v - 2 * bpm) / (2 * bpm),
                                abs(v - 0.5 * bpm) / (0.5 * bpm)) < 0.04 else 0.0
                     for v in votes])
    base = max(np.mean(onset), 1e-9)
    sharp = np.clip((main["score"] / base - 1.0) / 3.0, 0.0, 1.0)
    bpm_conf = float(np.clip(0.6 * agree + 0.4 * sharp, 0.0, 1.0))
    return grid, float(bpm), bpm_conf, np.array(all_beats)


def _sample_curve(curve, times_s):
    pos = np.clip(np.asarray(times_s) * FPS, 0, len(curve) - 1.001)
    i = pos.astype(np.int64)
    fr = pos - i
    return curve[i] * (1 - fr) + curve[np.minimum(i + 1, len(curve) - 1)] * fr


def estimate_downbeat(beats, onset_bass, novelty):
    """Which of the 4 beat-phase classes carries the bar: bass onsets land on
    the 1, and section novelty concentrates at bar boundaries."""
    if len(beats) < 16:
        return 0, 0.0
    b = _sample_curve(onset_bass, beats)
    nv = _sample_curve(novelty, beats)
    scores = np.zeros(4)
    for o in range(4):
        scores[o] = b[o::4].mean() + 0.6 * nv[o::4].mean()
    order = np.argsort(scores)[::-1]
    best, second = scores[order[0]], scores[order[1]]
    conf = float(np.clip((best - second) / max(best, 1e-9), 0.0, 1.0))
    return int(order[0]), conf


# --------------------------------------------------------------------------
# Key
# --------------------------------------------------------------------------

def camelot_of(pc, mode):
    """pc is C-origin (C=0). A minor -> 8A, C major -> 8B."""
    if mode == "minor":
        pc = (pc + 3) % 12
    num = ((pc * 7) % 12 + 7) % 12 + 1
    return f"{num}{'A' if mode == 'minor' else 'B'}"


def estimate_key(chroma_a_origin, frame_energy):
    """Krumhansl-Schmuckler over the energy-weighted mean chroma.

    Analyzer chroma is A-origin; rotate (i+3)%12 to C-origin first."""
    w = np.maximum(frame_energy, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    mean_a = (chroma_a_origin * w[:, None]).sum(axis=0) / w.sum()
    mean_c = np.zeros(12)
    for i in range(12):
        mean_c[(i + 9) % 12] = mean_a[i]          # A-origin bin i == pc (9+i)%12
    if mean_c.sum() <= 1e-12:
        return 0, "major", "8B", 0.0
    x = mean_c - mean_c.mean()
    best = []
    for mode, prof in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        p = prof - prof.mean()
        for pc in range(12):
            r = float(np.dot(x, np.roll(p, pc)) /
                      max(np.linalg.norm(x) * np.linalg.norm(p), 1e-12))
            best.append((r, pc, mode))
    best.sort(reverse=True)
    r0, pc, mode = best[0]
    r1 = best[1][0]
    conf = float(np.clip((r0 - r1) * 4.0 + max(r0, 0.0) * 0.5, 0.0, 1.0))
    return pc, mode, camelot_of(pc, mode), conf


# --------------------------------------------------------------------------
# Structure: SSM -> novelty -> sections -> fingerprints
# --------------------------------------------------------------------------

def _beat_sync_features(bands, chroma, beats):
    """Average band env (log) + chroma per beat -> unit-norm vectors."""
    idx = np.clip((np.asarray(beats) * FPS).astype(np.int64), 0, len(bands) - 1)
    idx = np.append(idx, len(bands))
    feats = []
    mean = np.maximum(bands.mean(axis=0), 1e-10)
    for k in range(len(idx) - 1):
        a, b = idx[k], max(idx[k] + 1, idx[k + 1])
        bb = np.log1p(bands[a:b] / mean).mean(axis=0)
        cc = chroma[a:b].mean(axis=0)
        v = np.concatenate([bb, cc * 3.0])        # chroma weighted up
        feats.append(v / max(np.linalg.norm(v), 1e-12))
    return np.array(feats)


def _novelty_curve(feats, kernel_beats=SSM_KERNEL_BEATS):
    """Checkerboard-kernel novelty along the SSM diagonal."""
    nb = len(feats)
    K = min(kernel_beats, max(4, nb // 4))
    K -= K % 2
    if nb < K + 2 or K < 4:
        return np.zeros(nb)
    ssm = feats @ feats.T
    half = K // 2
    g = np.exp(-0.5 * ((np.arange(K) - (K - 1) / 2) / (K / 4)) ** 2)
    kern = np.outer(g, g)
    sign = np.ones((K, K))
    sign[:half, half:] = -1
    sign[half:, :half] = -1
    kern *= sign
    nov = np.zeros(nb)
    for c in range(half, nb - half):
        nov[c] = float((ssm[c - half:c + half, c - half:c + half] * kern).sum())
    nov = np.maximum(nov, 0.0)
    if nov.max() > 0:
        nov /= nov.max()
    return nov


def _pick_boundaries(nov, min_gap_beats=8):
    """Peaks above mean+0.5*std with a minimum spacing."""
    if len(nov) < 4:
        return []
    thr = nov.mean() + 0.5 * nov.std()
    cand = [i for i in range(1, len(nov) - 1)
            if nov[i] >= thr and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    cand.sort(key=lambda i: -nov[i])
    picked = []
    for i in cand:
        if all(abs(i - j) >= min_gap_beats for j in picked):
            picked.append(i)
    return sorted(picked)


def build_sections(bands, chroma, beats, downbeat_offset,
                   onset_perc, novelty_frames):
    """Section list with fingerprints; boundaries snapped to downbeats."""
    nb = len(beats)
    if nb < 16:
        return [], np.zeros(0)
    feats = _beat_sync_features(bands, chroma, beats)
    nov = _novelty_curve(feats)
    bounds_b = _pick_boundaries(nov)
    # Snap to downbeats (beat index % 4 == downbeat_offset).
    def snap(i):
        d = i - ((i - downbeat_offset) % 4)
        return int(np.clip(d if (i - d) <= 2 else d + 4, 0, nb - 1))
    bounds = sorted({0, nb} | {snap(i) for i in bounds_b})
    bounds = [b for b in bounds if b == 0 or b == nb or 4 <= b <= nb - 4]
    # Merge runts (<8 beats).
    clean = [bounds[0]]
    for b in bounds[1:]:
        if b - clean[-1] < 8 and b != nb:
            continue
        clean.append(b)
    if clean[-1] != nb:
        clean.append(nb)
    bounds = clean

    mean = np.maximum(bands.mean(axis=0), 1e-10)
    total_pow = (bands / mean).mean(axis=1)
    p95 = max(np.percentile(total_pow, 95), 1e-9)
    energy_curve = np.clip(total_pow / p95, 0.0, 1.5)
    ssm = feats @ feats.T

    sections = []
    for k in range(len(bounds) - 1):
        b0, b1 = bounds[k], bounds[k + 1]
        t0, t1 = float(beats[b0]), float(beats[b1 - 1]) + (
            float(beats[1] - beats[0]) if nb > 1 else 0.5)
        f0, f1 = int(t0 * FPS), min(int(t1 * FPS), len(bands))
        if f1 <= f0:
            continue
        seg_bands = bands[f0:f1] / mean
        tot = seg_bands.mean(axis=1)
        bass = seg_bands[:, 0:6].mean()
        mid = seg_bands[:, 6:20].mean()
        high = seg_bands[:, 20:32].mean()
        ssum = max(bass + mid + high, 1e-9)
        perc = onset_perc[f0:f1]
        hits = float((perc > np.percentile(onset_perc, 75)).mean()) * FPS
        beat_rate = 1.0 / max(float(beats[1] - beats[0]), 1e-6) if nb > 1 else 2.0
        chroma_seg = chroma[f0:f1].mean(axis=0)
        clarity = float(chroma_seg.max() / max(chroma_seg.mean(), 1e-9) / 12.0)
        rep = float(ssm[b0:b1, b0:b1].mean())
        nov_strength = float(nov[b0]) if b0 < len(nov) else 0.0
        sections.append({
            "start_s": round(t0, 3), "end_s": round(t1, 3),
            "start_beat": int(b0), "end_beat": int(b1),
            "energy": round(float(np.clip(tot.mean() / max(energy_curve.mean(), 1e-9)
                                          * energy_curve.mean(), 0, 1.5)), 3),
            "bass_share": round(float(bass / ssum), 3),
            "mid_share": round(float(mid / ssum), 3),
            "high_share": round(float(high / ssum), 3),
            "rhythm_density": round(hits / max(beat_rate, 1e-6), 3),
            "repetitiveness": round(rep, 3),
            "busyness": round(float(perc.mean() + seg_bands[:, 6:20].std()), 4),
            "vocalness": round(float((mid / ssum) * clarity * 4.0), 4),
            "boundary_strength": round(nov_strength, 3),
        })

    # Normalize busyness/vocalness/energy to the track's own range.
    for key in ("busyness", "vocalness"):
        vals = np.array([s[key] for s in sections])
        vmax = max(vals.max(), 1e-9)
        for s in sections:
            s[key] = round(float(s[key] / vmax), 3)
    evals = np.array([s["energy"] for s in sections])
    emax = max(evals.max(), 1e-9)
    for s in sections:
        s["energy"] = round(float(s["energy"] / emax), 3)

    _classify_sections(sections)
    return sections, nov


def _classify_sections(sections):
    if not sections:
        return
    energies = np.array([s["energy"] for s in sections])
    hi = np.percentile(energies, 70)
    lo = np.percentile(energies, 30)
    for i, s in enumerate(sections):
        kind = "steady"
        e = s["energy"]
        if i == 0 and e <= np.median(energies):
            kind = "intro"
        elif i == len(sections) - 1 and e <= np.median(energies):
            kind = "outro"
        elif e >= hi:
            kind = "drop"
        elif e <= lo:
            kind = "breakdown"
        elif i + 1 < len(sections) and sections[i + 1]["energy"] >= hi \
                and e < sections[i + 1]["energy"]:
            kind = "build"
        s["kind"] = kind
    # A build is only a build if it precedes a drop; re-check breakdown->drop.
    for i, s in enumerate(sections):
        if s["kind"] == "steady" and i + 1 < len(sections) \
                and sections[i + 1]["kind"] == "drop" \
                and s["energy"] < sections[i + 1]["energy"] - 0.1:
            s["kind"] = "build"


def find_loops(sections, feats_beats, beats):
    """Downbeat-aligned 8/16/32-beat loop candidates inside repetitive,
    low-vocal sections, scored by half-vs-half self-similarity."""
    loops = []
    if len(beats) < 20 or feats_beats is None or not len(feats_beats):
        return loops
    for s in sections:
        if s["vocalness"] > 0.6 or s["repetitiveness"] < 0.5:
            continue
        b0, b1 = s["start_beat"], s["end_beat"]
        for nb_ in (32, 16, 8):
            for st in range(b0, b1 - nb_ + 1, 4):
                a = feats_beats[st:st + nb_ // 2]
                b = feats_beats[st + nb_ // 2:st + nb_]
                if len(a) != len(b) or not len(a):
                    continue
                sim = float(np.mean(np.sum(a * b, axis=1)))
                score = sim * (0.5 + 0.5 * s["repetitiveness"])
                loops.append({"start_s": round(float(beats[st]), 3),
                              "beats": nb_, "score": round(score, 3)})
    loops.sort(key=lambda l: -l["score"])
    out, seen = [], []
    for l in loops:                                # spread picks around
        if any(abs(l["start_s"] - t) < 8.0 and l["beats"] == b_
               for t, b_ in seen):
            continue
        out.append(l)
        seen.append((l["start_s"], l["beats"]))
        if len(out) >= 6:
            break
    return out


def find_mix_points(sections, duration_s):
    """Entry/exit points anchored to section boundaries, scored by boundary
    strength x low busyness x usefulness of position."""
    pts = []
    for i, s in enumerate(sections):
        t = s["start_s"]
        strength = max(s["boundary_strength"], 0.05)
        quiet = 1.0 - 0.7 * s["busyness"]
        if i > 0 and t < duration_s * 0.45:
            hint = "pre_drop" if s["kind"] == "drop" else "blend"
            pts.append({"kind": "in", "time_s": round(t, 3),
                        "score": round(strength * quiet, 3),
                        "style_hint": hint})
        if i > 0 and t > duration_s * 0.45:
            prev = sections[i - 1]
            hint = "outro" if s["kind"] == "outro" else "blend"
            pts.append({"kind": "out", "time_s": round(t, 3),
                        "score": round(strength * (1.0 - 0.7 * prev["busyness"]), 3),
                        "style_hint": hint})
    # Always offer a default entry (first non-intro section or t=0).
    if not any(p["kind"] == "in" for p in pts):
        t0 = sections[1]["start_s"] if len(sections) > 1 else 0.0
        pts.append({"kind": "in", "time_s": round(t0, 3),
                    "score": 0.2, "style_hint": "blend"})
    if not any(p["kind"] == "out" for p in pts) and sections:
        pts.append({"kind": "out", "time_s": round(sections[-1]["start_s"], 3),
                    "score": 0.2, "style_hint": "outro"})
    pts.sort(key=lambda p: (p["kind"], -p["score"]))
    return pts


# --------------------------------------------------------------------------
# Loudness + energy curve
# --------------------------------------------------------------------------

def estimate_loudness_gain(samples):
    """Gain (dB) that brings the loudest 60% of 0.5s frames to the target."""
    fl = RATE // 2
    n = len(samples) // fl
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(samples[:n * fl].reshape(n, fl) ** 2, axis=1))
    rms = np.sort(rms)[int(n * 0.4):]
    loud = float(np.mean(rms))
    if loud <= 1e-9:
        return 0.0
    gain = LOUDNESS_TARGET_DBFS - 20.0 * math.log10(loud)
    return float(np.clip(gain, -9.0, 9.0))


def energy_curve_2hz(bands):
    mean = np.maximum(bands.mean(axis=0), 1e-10)
    tot = (bands / mean).mean(axis=1)
    blk = FPS // 2
    n = len(tot) // blk
    if n == 0:
        return []
    curve = tot[:n * blk].reshape(n, blk).mean(axis=1)
    p95 = max(np.percentile(curve, 95), 1e-9)
    return [round(float(v), 3) for v in np.clip(curve / p95, 0.0, 1.2)]


def _spectral_shares(bands):
    m = bands.mean(axis=0)
    bass, mid, high = m[0:6].sum(), m[6:20].sum(), m[20:32].sum()
    tot = max(bass + mid + high, 1e-12)
    return {"bass_share": round(float(bass / tot), 3),
            "mid_share": round(float(mid / tot), 3),
            "high_share": round(float(high / tot), 3)}


# --------------------------------------------------------------------------
# Live-pipeline cross-check (the real BeatDetector + AudioStructure)
# --------------------------------------------------------------------------

def live_crosscheck(bands, grid_bpm):
    """Run the actual live detector over the framed track: what showtime
    visuals will see. Returns bpm agreement + mood histogram + density."""
    from lib.beat_detector import BeatDetector
    from lib.audio_signals import AudioStructure
    WIN_SHORT, WIN_LONG = 20, 100
    bd, st = BeatDetector(), AudioStructure()
    dt = 1.0 / FPS
    csum = np.cumsum(bands, axis=0)
    moods, densities, bpms, confs = {}, [], [], []
    for t in range(len(bands)):
        s0 = max(0, t - WIN_SHORT + 1)
        l0 = max(0, t - WIN_LONG + 1)
        ms = (csum[t] - (csum[s0 - 1] if s0 > 0 else 0)) / (t - s0 + 1)
        ml = (csum[t] - (csum[l0 - 1] if l0 > 0 else 0)) / (t - l0 + 1)
        ms = np.maximum(ms, 1e-10)
        ml = np.maximum(ml, 1e-10)
        bp = bands[t]
        sound = {"raw_bands": np.array([bp]),
                 "norm_short": np.array([bp / ms]),
                 "norm_long": np.array([bp / ml]),
                 "norm_long_relu": np.array([np.maximum(0.0, bp / ml - 1.0)])}
        beat = bd.update(sound, dt)
        sig = st.update(sound, beat, dt)
        if t > FPS * 15:
            moods[sig["mood"]] = moods.get(sig["mood"], 0) + 1
            densities.append(sig["density"])
            if beat["confidence"] > 0.4:
                bpms.append(beat["bpm"])
            confs.append(beat["confidence"])
    total = max(sum(moods.values()), 1)
    mood_hist = {k: round(v / total, 3) for k, v in sorted(moods.items())}
    live_bpm = float(np.median(bpms)) if bpms else 0.0
    agrees = False
    if live_bpm > 0 and grid_bpm > 0:
        agrees = any(abs(live_bpm - m * grid_bpm) / (m * grid_bpm) < 0.05
                     for m in (0.5, 1.0, 2.0))
    return {"live_bpm": round(live_bpm, 2), "agrees": bool(agrees),
            "mean_conf": round(float(np.mean(confs)) if confs else 0.0, 3),
            "mood_hist": mood_hist,
            "rhythm_density": round(float(np.median(densities))
                                    if densities else 0.0, 3)}


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

def analyze_samples(samples, deep=True):
    """Full analysis of mono float32 44.1k samples -> plain-JSON-able dict."""
    duration_s = len(samples) / RATE
    bands, chroma = frame_track(samples)
    onset_broad, onset_bass, onset_perc, novelty = _onset_channels(bands)
    onset_mix = onset_broad + 0.5 * onset_perc

    grid, bpm, bpm_conf, beats = estimate_beat_grid(onset_mix)
    downbeat, db_conf = estimate_downbeat(beats, onset_bass, novelty)

    frame_energy = (bands / np.maximum(bands.mean(axis=0), 1e-10)).mean(axis=1)
    key_pc, key_mode, camelot, key_conf = estimate_key(chroma, frame_energy)

    sections, nov = build_sections(bands, chroma, beats, downbeat,
                                   onset_perc, novelty)
    feats_beats = _beat_sync_features(bands, chroma, beats) if len(beats) >= 16 else None
    loops = find_loops(sections, feats_beats, beats)
    mix_points = find_mix_points(sections, duration_s)

    result = {
        "analysis_version": ANALYSIS_VERSION,
        "duration_s": round(duration_s, 3),
        "bpm": round(bpm, 3), "bpm_conf": round(bpm_conf, 3),
        "beat_grid": [{k: (round(v, 5) if isinstance(v, float) else v)
                       for k, v in g.items()} for g in grid],
        "downbeat_offset": downbeat, "downbeat_conf": round(db_conf, 3),
        "key_pc": key_pc, "key_mode": key_mode,
        "camelot": camelot, "key_conf": round(key_conf, 3),
        "key_name": f"{_NOTE_NAMES[key_pc]} {key_mode}",
        "loudness_gain_db": round(estimate_loudness_gain(samples), 2),
        "energy_curve": energy_curve_2hz(bands),
        "sections": sections, "loops": loops, "mix_points": mix_points,
        "spectral": _spectral_shares(bands),
        "mood_hist": {}, "rhythm_density": 0.0,
    }
    if deep:
        live = live_crosscheck(bands, bpm)
        result["live_check"] = live
        result["mood_hist"] = live["mood_hist"]
        result["rhythm_density"] = live["rhythm_density"]
        if not live["agrees"] and live["live_bpm"] > 0:
            result["bpm_conf"] = round(result["bpm_conf"] * 0.5, 3)
    return result


def analyze_file(path, deep=True):
    samples = decode_file(path)
    result = analyze_samples(samples, deep=deep)
    result.update(read_metadata(path))
    return result


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    r = analyze_file(sys.argv[1])
    r.pop("energy_curve", None)
    print(json.dumps(r, indent=1))
    print(f"\nanalyzed in {time.time() - t0:.1f}s", file=sys.stderr)
