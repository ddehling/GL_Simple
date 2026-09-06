"""Ingest a song and infer the SongScript that would have generated it.

On top of the DJ analysis (lib/dj/features.py: beat grid, downbeats, key,
sections, rhythm signature) this adds what the generator needs:

  * style       from tempo + the kick/snare step pattern (four / breakbeat /
                halftime / broken) and the spectral tilt
  * sections    DJ kinds -> generator sections (intro, groove, drop, build,
                break, outro), lengths on the 4-bar grid, energy 0..1
  * layers      which slots play, from band shares and onset activity
  * chords      per-bar chroma matched against the key's diatonic triads,
                one 4-bar loop per section
  * levers      density (onset rate), brightness (high share), swing

and the per-bar FEATURE track used to score a recreation (score.py):
energy dB, band shares, low/high onset density, chroma - on the song's
own bar grid so original and recreation compare bar for bar.

Melody transcription is out of scope: hooks are not inferred (the
generator writes its own); the recreation matches structure, harmony,
rhythm, spectrum and dynamics, not the tune."""
from __future__ import annotations

import os

import numpy as np

from lib.gen import RATE
from lib.dj import features as F

STYLE_BPM = [("ambient", 0, 80), ("hiphop", 80, 100), ("downtempo", 80, 112), ("groove", 112, 128),
             ("techno", 126, 136), ("trance", 134, 150), ("dnb", 150, 200)]
SECTION_MAP = {"intro": "intro", "groove": "groove", "breakdown": "break", "build": "build", "outro": "outro"}


def _bar_grid(beats, downbeat):
    """Bar start times (s) from the beat list and the downbeat offset."""
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 8:
        return np.zeros(0)
    return beats[int(downbeat) % 4::4]


def _chroma_c(chroma_a):
    """The DJ chroma is A-origin (index 0 = A); roll so index 0 = C."""
    return np.roll(chroma_a, 9, axis=1)


def bar_chroma(samples, t0, t1, lo_hz=55.0, hi_hz=2500.0):
    """Chroma (C-origin, sums to 1) and bass chroma (55-220 Hz) of one bar
    from a single long FFT: harmonic content, not the DJ frame chroma,
    which is tuned for key finding over a whole track."""
    a, b = int(t0 * RATE), int(t1 * RATE)
    x = samples[a:b].astype(np.float64)
    n = len(x)
    if n < 2048:
        return np.ones(12) / 12.0, np.ones(12) / 12.0
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / RATE)
    out = np.zeros(12)
    bass = np.zeros(12)
    m = (freqs >= lo_hz) & (freqs <= hi_hz)
    f = freqs[m]
    pc = (np.round(12.0 * np.log2(f / 261.6256)) % 12).astype(int)
    w = spec[m] / f                                   # 1/f: the fundamental region counts most
    np.add.at(out, pc, w)
    mb = (freqs >= lo_hz) & (freqs <= 220.0)
    pcb = (np.round(12.0 * np.log2(freqs[mb] / 261.6256)) % 12).astype(int)
    np.add.at(bass, pcb, spec[mb])
    out = out / max(float(out.sum()), 1e-12)
    bass = bass / max(float(bass.sum()), 1e-12)
    return out, bass


def bar_features(samples, bars, bands=None, chroma=None):
    """Per-bar feature dicts on the given bar grid (times in s)."""
    if bands is None or chroma is None:
        bands, chroma = F.frame_track(samples)
    chroma = _chroma_c(chroma)
    onset_broad, onset_bass, onset_perc, _ = F._onset_channels(bands)
    mean = np.maximum(bands.mean(axis=0), 1e-10)
    out = []
    thr_low = np.percentile(onset_bass, 80)
    thr_high = np.percentile(onset_perc, 80)
    for i in range(len(bars) - 1):
        f0, f1 = int(bars[i] * F.FPS), int(bars[i + 1] * F.FPS)
        if f1 <= f0 or f1 > len(bands):
            break
        seg = bands[f0:f1]
        tot = float(seg.mean())
        rel = seg / mean
        bass, mid, high = rel[:, 0:6].mean(), rel[:, 6:20].mean(), rel[:, 20:32].mean()
        ssum = max(bass + mid + high, 1e-9)
        c, cb = bar_chroma(samples, bars[i], bars[i + 1])
        dur = (bars[i + 1] - bars[i])
        out.append({"t": float(bars[i]), "energy_db": float(10 * np.log10(tot + 1e-12)),
                    "bass": float(bass / ssum), "mid": float(mid / ssum), "high": float(high / ssum),
                    "low_hits": float((onset_bass[f0:f1] > thr_low).sum() / dur),
                    "high_hits": float((onset_perc[f0:f1] > thr_high).sum() / dur),
                    "chroma": [round(float(x), 5) for x in c],
                    "bass_chroma": [round(float(x), 5) for x in cb]})
    return out


def _triad_templates(key_pc, mode):
    from lib.gen.theory import Key
    k = Key(key_pc, "minor" if mode == "minor" else "major")
    temps = []
    for deg in range(7):
        pcs = [m % 12 for m in k.chord(deg, octave=3, size=3)]
        t = np.zeros(12)
        t[pcs[0]] = 1.0
        t[pcs[1]] = 0.8
        t[pcs[2]] = 0.8
        temps.append(t / np.linalg.norm(t))
    return temps


def chords_per_bar(feats, key_pc, mode):
    """Diatonic degree per bar: triad template fit on the bar chroma, with
    the bass note (the root in this music) weighted in."""
    from lib.gen.theory import Key
    k = Key(key_pc, "minor" if mode == "minor" else "major")
    temps = _triad_templates(key_pc, mode)
    roots = [k.degree_pc(d) for d in range(7)]
    out = []
    for f in feats:
        c = np.asarray(f["chroma"])
        n = np.linalg.norm(c)
        if n < 1e-9:
            out.append(0)
            continue
        c = c / n
        cb = np.asarray(f.get("bass_chroma") or np.zeros(12))
        cb = cb / max(float(cb.max()), 1e-9)
        scores = [float(c @ t) + 0.5 * float(cb[roots[d]]) for d, t in enumerate(temps)]
        out.append(int(np.argmax(scores)))
    return out


def _refine_key(feats, key_pc, mode):
    """The KS estimate confuses a key with its relative / neighbours on
    chord-loop music; pick, among those candidates, the key whose diatonic
    triads fit the per-bar chroma best."""
    cands = [(pc, md) for pc in range(12) for md in ("minor", "major")]
    best, best_fit = (key_pc, mode), -1.0
    for pc, md in cands:
        temps = _triad_templates(pc, md)
        fit = 0.0
        n = 0
        for f in feats:
            c = np.asarray(f["chroma"])
            nrm = np.linalg.norm(c)
            if nrm < 1e-9:
                continue
            c = c / nrm
            fit += max(float(c @ t) for t in temps)
            n += 1
        fit = fit / max(n, 1)
        # tie-breaks: the KS estimate itself, and a tonic that actually sounds
        if (pc, md) == (key_pc, mode):
            fit += 0.002
        tonic_w = float(np.mean([np.asarray(f["chroma"])[pc] for f in feats])) if feats else 0.0
        fit += 0.2 * tonic_w
        if fit > best_fit:
            best, best_fit = (pc, md), fit
    return best


def _style_from(bpm, sig, sections):
    low = np.asarray(sig.get("low") or [], dtype=float)
    mid = np.asarray(sig.get("mid") or [], dtype=float)
    kind = "four"
    if len(low) >= 16:
        L = low.reshape(-1, 16).mean(axis=0) if len(low) % 16 == 0 else low[:16]
        M = mid.reshape(-1, 16).mean(axis=0) if len(mid) >= 16 and len(mid) % 16 == 0 else (mid[:16] if len(mid) >= 16 else np.zeros(16))
        on_beats = L[[0, 4, 8, 12]].mean()
        off = np.delete(L, [0, 4, 8, 12]).mean() if L.size == 16 else 0.0
        if on_beats > 1.6 * (off + 1e-6) and L[[4, 8, 12]].min() > 0.5 * L[0]:
            kind = "four"
        elif M.size == 16 and M[8] > 1.3 * max(M[4], M[12], 1e-6):
            kind = "halftime"
        elif M.size == 16 and M[4] > 0.6 * M.max() and M[12] > 0.6 * M.max() and L[10] > 0.5 * L[0]:
            kind = "breakbeat"
        else:
            kind = "broken"
    if bpm >= 150:
        return "dnb"
    if kind == "halftime" or (bpm < 100 and kind != "four"):
        return "hiphop" if bpm >= 80 else "ambient"
    if bpm < 80:
        return "ambient"
    if kind in ("broken", "breakbeat") and bpm < 112:
        return "downtempo"
    for name, lo, hi in STYLE_BPM:
        if lo <= bpm < hi and name in ("groove", "techno", "trance"):
            return name
    return "groove"


def _layers(sec, kind, emax, high_mean, bass_mean):
    e = sec["energy"] / max(emax, 1e-6)
    lay = set()
    if kind in ("groove", "drop", "build", "intro", "outro") and sec["bass_share"] > 0.6 * bass_mean:
        lay |= {"kick"}
    if sec["bass_share"] > 0.8 * bass_mean and kind not in ("intro", "outro"):
        lay |= {"bass"}
    if sec["high_share"] > 0.7 * high_mean:
        lay |= {"hat", "shaker"}
    if sec["high_share"] > 1.05 * high_mean and e > 0.7:
        lay |= {"ohat", "ride"}
    if e > 0.45 and kind not in ("intro",):
        lay |= {"snare", "perc"}
    lay |= {"pad"}
    if e > 0.5:
        lay |= {"keys"}
    if e > 0.65 or kind in ("build", "drop"):
        lay |= {"lead", "arp"}
    if kind == "break":
        lay -= {"kick", "snare"}
    return sorted(lay)


def ingest(path, progress=None, deep=False, reuse=False, out_dir=None, want=("kit", "vocals", "hook")):
    """Analyse an audio file -> {"script": SongScript, "features": [bar feats], "analysis": dict, "bars": [s]}.
    reuse=True also separates the song into stems and puts its own drums,
    vocal phrases and transcribed hook into the script (lib/gen/analysis/reuse.py)."""
    samples = F.decode_file(path)
    title = os.path.splitext(os.path.basename(path))[0]
    res = ingest_samples(samples, title=title, progress=progress, deep=deep)
    if reuse:
        from lib.gen.analysis import reuse as R
        out_dir = out_dir or os.path.join("logs", "analysis", title)
        stereo = F.decode_file_stereo(path)
        a = res["analysis"]
        from lib.gen.theory import parse_key
        key = parse_key(res["script"]["key"])
        mat = R.reuse(stereo, res["bars"], key.root, "minor" if key.mode != "major" else "major", out_dir,
                      progress=(lambda p, what: progress(0.5 + 0.5 * p, what)) if progress else None, want=want)
        res["material"] = mat
        sc = res["script"]
        if mat.get("kit"):
            sc["kit"] = mat["kit"]
        if mat.get("vocals"):
            sc["vocals"] = mat["vocals"]
        if mat.get("hook"):
            # the hook is the theme from the first loud section on
            for e in sc["sections"]:
                if e["section"] in ("drop", "groove", "build") and e.get("energy", 0) >= 0.7:
                    e["hook"] = mat["hook"]
                    break
            else:
                if sc["sections"]:
                    sc["sections"][0]["hook"] = mat["hook"]
        a["reuse_reasons"] = mat.get("reasons", [])
    return res


def ingest_samples(samples, title="ingested", progress=None, deep=False):
    if progress:
        progress(0.05, "framing")
    bands, chroma = F.frame_track(samples)
    onset_broad, onset_bass, onset_perc, novelty = F._onset_channels(bands)
    onset_mix = onset_broad + 0.5 * onset_perc
    if progress:
        progress(0.25, "beat grid")
    grid, bpm, bpm_conf, beats = F.estimate_beat_grid(onset_mix)
    downbeat, db_conf = F.estimate_downbeat(beats, bands, chroma, onset_bass, onset_broad)
    frame_energy = (bands / np.maximum(bands.mean(axis=0), 1e-10)).mean(axis=1)
    key_pc, key_mode, camelot, key_conf = F.estimate_key(chroma, frame_energy)
    if progress:
        progress(0.5, "sections")
    sections, _nov = F.build_sections(bands, chroma, beats, downbeat, onset_perc, novelty)
    try:
        from lib.dj.rhythm import rhythm_signature
        sig = rhythm_signature(bands, grid, downbeat) or {}
    except Exception:
        sig = {}
    bars = _bar_grid(beats, downbeat)
    if progress:
        progress(0.7, "features")
    feats = bar_features(samples, bars, bands, chroma)
    key_pc, key_mode = _refine_key(feats, key_pc, key_mode)
    chords = chords_per_bar(feats, key_pc, key_mode)
    emax = max([s["energy"] for s in sections] + [1e-6])
    high_mean = float(np.mean([s["high_share"] for s in sections])) if sections else 0.2
    bass_mean = float(np.mean([s["bass_share"] for s in sections])) if sections else 0.3
    style = _style_from(bpm, sig, sections)
    # DJ swing = the offbeat position 0.5 (straight) .. 0.67 (triplet); the composer wants 0 .. 0.33
    swing = float(np.clip((float(sig.get("swing", 0.5) or 0.5) - 0.5) * 2.0, 0.0, 0.33))
    entries = []
    bar_t = list(bars)
    for i, sec in enumerate(sections):
        kind = SECTION_MAP.get(sec.get("kind", "groove"), "groove")
        e = sec["energy"] / emax
        prev_kind = sections[i - 1].get("kind") if i else None
        if kind == "groove" and e >= 0.85 and prev_kind in ("breakdown", "build"):
            kind = "drop"
        nbars = max(4, int(round((sec["end_beat"] - sec["start_beat"]) / 16.0)) * 4)
        # bars of this section on the grid -> its chord loop (first 4 bars, most common per position)
        b0 = int(np.searchsorted(bars, sec["start_s"] - 1e-3))
        loop = chords[b0:b0 + 4] if b0 + 4 <= len(chords) else chords[b0:]
        if len(loop) < 4:
            loop = (loop + [0, 0, 0, 0])[:4]
        dens = float(np.clip(0.35 + sec.get("rhythm_density", 3.0) / 6.0, 0.4, 1.3))
        e_lever = float(np.clip(1.3 * e - 0.3, 0.1, 1.0))      # DJ energy (RMS-ish) -> the form lever
        bright = float(np.clip(1.0 + (sec["high_share"] - high_mean) * 3.0, 0.6, 1.5))
        entry = {"section": kind, "bars": nbars, "energy": round(e_lever, 3),
                 "density": round(dens, 2), "brightness": round(bright, 2), "swing": round(swing, 3),
                 "layers": _layers(sec, kind, emax, high_mean, bass_mean), "chords": loop}
        if kind == "break" and sec["high_share"] < 0.6 * high_mean:
            entry["lanes"] = {"lp": 2500.0}
        entries.append(entry)
    style_mode = {"groove": "minor", "techno": "phrygian", "trance": "minor", "dnb": "minor", "hiphop": "dorian",
                  "downtempo": "dorian", "ambient": "lydian"}[style]
    from lib.gen.theory import Key
    key = Key(key_pc, "minor" if key_mode == "minor" else "major")
    key_txt = key.camelot if key.camelot != "?" else key.name
    script = {"title": title, "style": style, "bpm": round(float(bpm), 2), "key": key_txt, "seed": 1, "humanize": 1.0,
              "end": True, "sections": entries}
    if progress:
        progress(1.0, "done")
    analysis = {"bpm": float(bpm), "bpm_conf": float(bpm_conf), "key": key.name, "camelot": key.camelot, "key_conf": float(key_conf),
                "downbeat_conf": float(db_conf), "duration_s": len(samples) / RATE, "n_sections": len(sections),
                "sections": sections, "rhythm": {k: sig.get(k) for k in ("swing", "swing_conf", "w_low", "w_mid", "w_high")},
                "style_mode": style_mode, "first_bar_s": float(bars[0]) if len(bars) else 0.0}
    return {"script": script, "features": feats, "analysis": analysis, "bars": [float(b) for b in bars], "chords": chords}


def features_on_grid(samples, bpm, first_bar_s=0.0):
    """Feature track for audio whose bars are known (a recreation): bars
    every 4 beats from first_bar_s."""
    bar_len = 4 * 60.0 / float(bpm)
    n = int((len(samples) / RATE - first_bar_s) // bar_len)
    bars = np.array([first_bar_s + i * bar_len for i in range(n + 1)])
    return bar_features(samples, bars)
