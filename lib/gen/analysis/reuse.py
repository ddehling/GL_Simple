"""Reuse the source song's own material in the recreation: separate it
into stems (demucs, the same stack the DJ planner uses), cut a drum kit
of one-shots from the drum stem, chop the vocal stem into phrases placed
on the bar grid, and transcribe the melodic stem into the hook the
generator will develop.

    stems     = separate(samples_stereo, out_dir)          # {"drums","bass","other","vocals"} -> wav paths
    kit       = drum_kit(stems["drums"], out_dir)           # {"kick": path, "snare": path, "hat": path, ...}
    vocals    = vocal_chops(stems["vocals"], bars, out_dir) # [{"bar": 12.25, "file": path, "seconds": 1.8}]
    hook      = transcribe_hook(stems["other"], bars, key_pc, mode)   # {"steps", "degrees", "contour", "name"}

All optional: available() says whether demucs is importable; every
function degrades to None / [] and reports why in `reasons`, so ingest
works without the heavy stack. CPU separation of a 4-minute track takes
a few minutes on a laptop; the analysis tab runs it on a worker thread."""
from __future__ import annotations

import os

import numpy as np

from lib.gen import RATE

reasons = []


def available() -> bool:
    try:
        import torch  # noqa: F401
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def separate(samples_stereo, out_dir, model="htdemucs", progress=None):
    """demucs on (n,2) float32 @ 44.1k -> {source: wav path} in out_dir."""
    import soundfile as sf
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    os.makedirs(out_dir, exist_ok=True)
    have = {name: os.path.join(out_dir, f"{name}.wav") for name in ("drums", "bass", "other", "vocals")}
    if all(os.path.exists(p) for p in have.values()):
        return have
    m = get_model(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(device).eval()
    x = np.ascontiguousarray(np.asarray(samples_stereo, dtype=np.float32).T)
    if x.shape[0] == 1:
        x = np.repeat(x, 2, axis=0)
    audio = torch.from_numpy(x).unsqueeze(0)
    ref = audio.mean(0)
    std = float(ref.std()) + 1e-8
    audio = audio / std
    with torch.no_grad():
        out = apply_model(m, audio, device=device, shifts=0, split=True, overlap=0.1, progress=False)[0].cpu().numpy() * std
    paths = {}
    for i, name in enumerate(m.sources):
        p = os.path.join(out_dir, f"{name}.wav")
        sf.write(p, np.clip(out[i].T, -1, 1), RATE, subtype="PCM_16")
        paths[name] = p
    return paths


def _mono(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    y = x.mean(axis=1)
    if sr != RATE:
        idx = np.arange(0, len(y), sr / RATE)
        y = np.interp(idx, np.arange(len(y)), y).astype(np.float32)
    return y


def _onsets(env, hop, thr_rel=0.35, min_gap_s=0.08):
    """Peak-pick an onset envelope (per hop): indices of hits."""
    if len(env) < 3:
        return []
    d = np.maximum(env[1:] - env[:-1], 0.0)
    thr = thr_rel * (np.percentile(d, 99) + 1e-9)
    gap = max(1, int(min_gap_s * RATE / hop))
    hits = []
    last = -gap
    for i in range(1, len(d) - 1):
        if d[i] > thr and d[i] >= d[i - 1] and d[i] >= d[i + 1] and i - last >= gap:
            hits.append(i + 1)
            last = i
    return hits


def _band_env(y, lo, hi, hop=441):
    """Energy envelope of a band (Hz) per hop, via STFT magnitude."""
    n_fft = 2048
    win = np.hanning(n_fft).astype(np.float32)
    n = max(0, (len(y) - n_fft) // hop)
    env = np.zeros(n, dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / RATE)
    m = (freqs >= lo) & (freqs < hi)
    for i in range(n):
        seg = y[i * hop:i * hop + n_fft] * win
        spec = np.abs(np.fft.rfft(seg))
        env[i] = float((spec[m] ** 2).sum())
    return env, hop


def drum_kit(drums_wav, out_dir, seconds=None):
    """Cut a kick, snare/clap and hat one-shot from the drum stem: the
    cleanest hit of each class (loud, with a quiet run-up). Returns
    {"kick": path, "snare": path, "hat": path} for what was found."""
    import soundfile as sf
    y = _mono(drums_wav)
    if seconds:
        y = y[: int(seconds * RATE)]
    os.makedirs(out_dir, exist_ok=True)
    classes = {"kick": ((30.0, 150.0), 0.45), "snare": ((150.0, 900.0), 0.35), "hat": ((5000.0, 16000.0), 0.15)}
    kit = {}
    for name, ((lo, hi), length) in classes.items():
        env, hop = _band_env(y, lo, hi)
        hits = _onsets(env, hop)
        best, best_score = None, -1.0
        for h in hits:
            i0 = h * hop
            i1 = i0 + int(length * RATE)
            if i1 >= len(y) or i0 < int(0.05 * RATE):
                continue
            pre = float(np.abs(y[i0 - int(0.03 * RATE):i0 - 32]).mean()) + 1e-6
            peak = float(np.abs(y[i0:i0 + int(0.02 * RATE)]).max())
            score = peak / pre * env[h]
            if score > best_score:
                best, best_score = i0, score
        if best is None:
            reasons.append(f"no {name} hit found in the drum stem")
            continue
        seg = y[best:best + int(length * RATE)].copy()
        fade = int(0.01 * RATE)
        seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        seg = seg / max(float(np.abs(seg).max()), 1e-6) * 0.9
        p = os.path.join(out_dir, f"{name}_song.wav")
        sf.write(p, seg, RATE, subtype="PCM_16")
        kit[name] = p
    return kit


def vocal_chops(vocals_wav, bars, out_dir, min_s=0.35, max_s=8.0, thr_db=-32.0, limit=48):
    """Phrases of the vocal stem placed on the bar grid: [{"bar": float
    (bar index + fraction, snapped to 16ths), "file": path, "seconds": len}]."""
    import soundfile as sf
    y = _mono(vocals_wav)
    bars = np.asarray(bars, dtype=np.float64)
    if len(bars) < 2:
        return []
    os.makedirs(out_dir, exist_ok=True)
    hop = int(0.02 * RATE)
    n = len(y) // hop
    rms = np.array([float(np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2)) + 1e-9) for i in range(n)])
    db = 20 * np.log10(rms)
    if db.max() < -50:
        reasons.append("vocal stem is silent")
        return []
    on = db > max(thr_db, float(np.percentile(db, 70)) - 12.0)
    # smooth: close gaps under 250 ms
    gap = int(0.25 / 0.02)
    i = 0
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]:
                j += 1
            if 0 < i and j < n and j - i <= gap:
                on[i:j] = True
            i = j
        else:
            i += 1
    chops = []
    i = 0
    while i < n and len(chops) < limit:
        if on[i]:
            j = i
            while j < n and on[j] and (j - i) * 0.02 < max_s:
                j += 1
            t0, t1 = i * 0.02, j * 0.02
            if t1 - t0 >= min_s:
                # snap the start to the nearest 16th on the bar grid
                k = int(np.searchsorted(bars, t0, side="right") - 1)
                if 0 <= k < len(bars) - 1:
                    bar_len = bars[k + 1] - bars[k]
                    frac = round((t0 - bars[k]) / bar_len * 16) / 16.0
                    start = bars[k] + frac * bar_len
                    a = max(0, int((start - 0.01) * RATE))
                    b = min(len(y), int(t1 * RATE) + int(0.05 * RATE))
                    seg = y[a:b].copy()
                    fade = min(len(seg) // 4, int(0.02 * RATE))
                    if fade > 0:
                        seg[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                        seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
                    p = os.path.join(out_dir, f"vox_{len(chops):02d}.wav")
                    sf.write(p, seg, RATE, subtype="PCM_16")
                    chops.append({"bar": round(k + frac, 4), "file": p, "seconds": round((b - a) / RATE, 3)})
            i = j
        else:
            i += 1
    return chops


def transcribe_hook(stem_wav, bars, key_pc, mode, min_notes=4):
    """Transcribe the melodic stem and return the most repeated two-bar
    cell as a hook (steps 0..31, degrees from the tonic, contour), or
    None. Uses basic-pitch when installed, else librosa pyin (mono)."""
    bars = np.asarray(bars, dtype=np.float64)
    if len(bars) < 3:
        return None
    notes = _notes_basic_pitch(stem_wav)
    if notes is None:
        notes = _notes_pyin(stem_wav)
    if not notes:
        reasons.append("no notes transcribed from the melodic stem")
        return None
    from lib.gen.theory import Key
    key = Key(key_pc, "minor" if mode == "minor" else "major")
    pcs = [key.degree_pc(d) for d in range(7)]
    # place notes on the 16th grid: (bar_index, step, degree index)
    placed = []
    for t, midi in notes:
        k = int(np.searchsorted(bars, t, side="right") - 1)
        if not (0 <= k < len(bars) - 1):
            continue
        bar_len = bars[k + 1] - bars[k]
        step = int(round((t - bars[k]) / bar_len * 16))
        if step >= 16:
            continue
        pc = int(round(midi)) % 12
        if pc not in pcs:
            # nearest scale tone
            pc = min(pcs, key=lambda p: min((p - pc) % 12, (pc - p) % 12))
        deg = pcs.index(pc)
        octave_off = (int(round(midi)) - (12 * 4 + key.root)) // 12     # relative to the tonic in octave 4
        placed.append((k, step, deg + 7 * octave_off))
    if len(placed) < min_notes:
        return None
    # two-bar windows -> signature of (step, degree mod 7); the most repeated one wins
    from collections import Counter, defaultdict
    wins = defaultdict(list)
    for k, step, deg in placed:
        w = k // 2
        wins[w].append(((k % 2) * 16 + step, deg))
    sigs = Counter()
    rep = {}
    for w, cell in wins.items():
        cell = sorted(set(cell))
        if len(cell) < min_notes:
            continue
        sig = tuple((s, d % 7) for s, d in cell)
        sigs[sig] += 1
        rep.setdefault(sig, cell)
    if not sigs:
        return None
    sig, count = sigs.most_common(1)[0]
    cell = rep[sig][:12]
    steps = [s for s, _ in cell]
    degs = [max(-4, min(9, d)) for _, d in cell]
    span = max(degs) - min(degs)
    third = max(1, len(degs) // 3)
    a, b, c = np.mean(degs[:third]), np.mean(degs[third:2 * third] or degs), np.mean(degs[-third:])
    contour = "flat" if span == 0 else ("arch" if b > a and b > c else ("wave" if b < a and b < c else ("rise" if c > a else "fall")))
    return {"steps": steps, "degrees": degs, "contour": contour, "name": f"transcribed x{count}"}


def _notes_basic_pitch(path):
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
    except Exception:
        return None
    try:
        _, midi_data, note_events = predict(path, ICASSP_2022_MODEL_PATH)
        out = [(float(s), float(p)) for s, e, p, amp, *_ in note_events if e - s >= 0.08 and amp >= 0.3]
        out.sort()
        return out
    except Exception as e:  # noqa: BLE001
        reasons.append(f"basic-pitch failed: {type(e).__name__}: {e}")
        return None


def _notes_pyin(path):
    try:
        import librosa
    except Exception:
        reasons.append("neither basic-pitch nor librosa available for transcription")
        return []
    y = _mono(path)
    if len(y) < RATE:
        return []
    f0, voiced, _ = librosa.pyin(y, fmin=80.0, fmax=1200.0, sr=RATE, frame_length=2048, hop_length=512)
    out = []
    cur = None
    for i, (f, v) in enumerate(zip(f0, voiced)):
        t = i * 512 / RATE
        if v and f and np.isfinite(f):
            midi = 69 + 12 * np.log2(f / 440.0)
            if cur is None or abs(midi - cur[1]) > 0.7:
                if cur is not None and t - cur[0] >= 0.08:
                    out.append((cur[0], cur[1]))
                cur = (t, midi)
        else:
            if cur is not None and t - cur[0] >= 0.08:
                out.append((cur[0], cur[1]))
            cur = None
    return out


def bass_line(bass_wav, bars, key_pc, mode):
    """Transcribe the bass stem (monophonic, librosa pyin 30-300 Hz) ->
    {"pcs": [pitch class or None per bar], "cells": {phrase_bar0: {"steps": [...], "degrees": [...]}}}.
    The per-bar pitch class feeds the chord reader (the bass IS the root);
    the per-phrase cell (16-step onset grid + degree offsets from the
    tonic) is what the recreation's bass plays."""
    try:
        import librosa
    except Exception:
        reasons.append("librosa missing: no bass line")
        return None
    y = _mono(bass_wav)
    bars = np.asarray(bars, dtype=np.float64)
    if len(y) < RATE or len(bars) < 2:
        return None
    hop = 512
    f0, voiced, _ = librosa.pyin(y, fmin=30.0, fmax=300.0, sr=RATE, frame_length=4096, hop_length=hop)
    env = np.array([float(np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2)) + 1e-9) for i in range(len(f0))])
    thr = np.percentile(env, 60)
    from lib.gen.theory import Key
    key = Key(key_pc, "minor" if mode == "minor" else "major")
    pcs_key = [key.degree_pc(d) for d in range(7)]
    per_bar = []
    notes = []          # (bar_index, step, midi)
    for k in range(len(bars) - 1):
        a = int(bars[k] * RATE / hop)
        b = int(bars[k + 1] * RATE / hop)
        seg = [(i, f0[i]) for i in range(a, min(b, len(f0))) if voiced[i] and np.isfinite(f0[i]) and env[i] > thr]
        if not seg:
            per_bar.append(None)
            continue
        midis = np.array([69 + 12 * np.log2(f / 440.0) for _, f in seg])
        pc = int(np.round(np.median(midis))) % 12
        per_bar.append(pc)
        # onsets within the bar: where the pitch starts or jumps
        last = None
        bar_len = b - a
        for i, f in seg:
            m = int(round(69 + 12 * np.log2(f / 440.0)))
            if last is None or abs(m - last[1]) >= 1 or i - last[0] > int(0.5 * RATE / hop):
                step = int(round((i - a) / max(bar_len, 1) * 16))
                if step < 16:
                    notes.append((k, step, m))
            last = (i, m)
    cells = {}
    for ph0 in range(0, len(bars) - 1, 4):
        ns = [(st, m) for (k, st, m) in notes if k == ph0]           # the phrase's first bar as the cell
        if len(ns) < 2:
            continue
        seen = {}
        for st, m in ns:
            seen.setdefault(st, m)
        steps = sorted(seen)
        tonic = 12 * 2 + key.root                                       # bass octave reference
        degs = []
        for st in steps:
            m = seen[st]
            pc = m % 12
            if pc not in pcs_key:
                pc = min(pcs_key, key=lambda p: min((p - pc) % 12, (pc - p) % 12))
            d = pcs_key.index(pc) + 7 * ((m - tonic) // 12)
            degs.append(int(max(-7, min(14, d))))
        cells[ph0] = {"steps": steps[:8], "degrees": degs[:8]}
    return {"pcs": per_bar, "cells": cells}


def melodic_bank(other_wav, out_dir, max_slices=6, length_s=0.6):
    """Slice the melodic stem at its strongest onsets into short samples
    with their pitch, so the recreation's keys / arp can play the song's
    own tone: [{"file", "base_midi"}]."""
    import soundfile as sf
    try:
        import librosa
    except Exception:
        reasons.append("librosa missing: no melodic bank")
        return []
    y = _mono(other_wav)
    if len(y) < RATE:
        return []
    os.makedirs(out_dir, exist_ok=True)
    env, hop = _band_env(y, 150.0, 4000.0)
    hits = _onsets(env, hop, thr_rel=0.5, min_gap_s=0.25)
    if not hits:
        reasons.append("no melodic onsets found")
        return []
    hits = sorted(hits, key=lambda h: -env[h])[: max_slices * 3]
    bank = []
    for h in hits:
        i0 = h * hop
        i1 = i0 + int(length_s * RATE)
        if i1 >= len(y):
            continue
        seg = y[i0:i1].copy()
        f0, voiced, _ = librosa.pyin(seg, fmin=80.0, fmax=1500.0, sr=RATE, frame_length=2048, hop_length=256)
        good = f0[np.isfinite(f0) & voiced] if voiced is not None else f0[np.isfinite(f0)]
        if len(good) < 4:
            continue
        midi = int(round(69 + 12 * np.log2(float(np.median(good)) / 440.0)))
        fade = int(0.02 * RATE)
        seg[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        seg[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        seg = seg / max(float(np.abs(seg).max()), 1e-6) * 0.8
        pth = os.path.join(out_dir, f"tone_{len(bank):02d}_{midi}.wav")
        sf.write(pth, seg, RATE, subtype="PCM_16")
        bank.append({"file": pth, "base_midi": midi})
        if len(bank) >= max_slices:
            break
    return bank


def reuse(samples_stereo, bars, key_pc, mode, out_dir, progress=None, want=("kit", "vocals", "hook", "bass", "bank")):
    """Everything above in one call -> {"stems", "kit", "vocals", "hook", "reasons"}."""
    reasons.clear()
    if not available():
        reasons.append("demucs/torch not installed (pip install -r requirements-dj-vocals.txt)")
        return {"stems": None, "kit": {}, "vocals": [], "hook": None, "bass_pcs": None, "bass_cells": {}, "bank": [], "reasons": list(reasons)}
    if progress:
        progress(0.05, "separating stems")
    stems = separate(samples_stereo, os.path.join(out_dir, "stems"))
    out = {"stems": stems, "kit": {}, "vocals": [], "hook": None, "bass_pcs": None, "bass_cells": {}, "bank": []}
    if "kit" in want:
        if progress:
            progress(0.6, "cutting the drum kit")
        out["kit"] = drum_kit(stems["drums"], os.path.join(out_dir, "kit"))
    if "vocals" in want:
        if progress:
            progress(0.75, "chopping vocals")
        out["vocals"] = vocal_chops(stems["vocals"], bars, os.path.join(out_dir, "vox"))
    if "hook" in want:
        if progress:
            progress(0.85, "transcribing the hook")
        out["hook"] = transcribe_hook(stems["other"], bars, key_pc, mode) or transcribe_hook(stems["vocals"], bars, key_pc, mode)
    if "bass" in want:
        if progress:
            progress(0.9, "transcribing the bass")
        bl = bass_line(stems["bass"], bars, key_pc, mode)
        if bl:
            out["bass_pcs"] = bl["pcs"]
            out["bass_cells"] = bl["cells"]
    if "bank" in want:
        if progress:
            progress(0.95, "slicing melodic tones")
        out["bank"] = melodic_bank(stems["other"], os.path.join(out_dir, "bank"))
    out["reasons"] = list(reasons)
    if progress:
        progress(1.0, "reuse done")
    return out
