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


CHROMA_N, CHROMA_HOP = 8192, 4096
_CHROMA_CACHE = {}


def _chroma_bins(lo_hz, hi_hz):
    key = (lo_hz, hi_hz)
    if key not in _CHROMA_CACHE:
        freqs = np.fft.rfftfreq(CHROMA_N, 1.0 / RATE)
        m = (freqs >= lo_hz) & (freqs <= hi_hz)
        pc = (np.round(12.0 * np.log2(freqs[m] / 261.6256)) % 12).astype(int)
        _CHROMA_CACHE[key] = (m, pc, freqs[m])
    return _CHROMA_CACHE[key]


def bar_chroma(samples, t0, t1, lo_hz=80.0, hi_hz=2500.0):
    """Chroma (C-origin, sums to 1) and bass chroma (40-250 Hz) of one bar:
    the MEDIAN over 186 ms frames, so the sustained (harmonic) content
    counts and transients do not - a long single FFT let the kick's
    fundamental read as the tonic on every bar of a dance record."""
    a, b = int(t0 * RATE), int(t1 * RATE)
    x = samples[a:b].astype(np.float64)
    n = len(x)
    if n < 2048:
        return np.ones(12) / 12.0, np.ones(12) / 12.0
    if n < CHROMA_N:
        x = np.concatenate([x, np.zeros(CHROMA_N - n)])
    win = np.hanning(CHROMA_N)
    m, pc, f = _chroma_bins(lo_hz, hi_hz)
    mb, pcb, fb = _chroma_bins(40.0, 250.0)
    starts = range(0, max(1, len(x) - CHROMA_N + 1), CHROMA_HOP)
    idx = np.array([np.arange(s, s + CHROMA_N) for s in starts])
    spec = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
    rows = np.zeros((len(starts), 12))
    rows_b = np.zeros((len(starts), 12))
    for j in range(len(starts)):
        np.add.at(rows[j], pc, spec[j, m] / f)          # 1/f: the fundamental region counts most
        np.add.at(rows_b[j], pcb, spec[j, mb])
    rows /= np.maximum(rows.sum(axis=1, keepdims=True), 1e-12)
    rows_b /= np.maximum(rows_b.sum(axis=1, keepdims=True), 1e-12)
    out = np.median(rows, axis=0)
    bass = np.median(rows_b, axis=0)
    out = out / max(float(out.sum()), 1e-12)
    bass = bass / max(float(bass.sum()), 1e-12)
    return out, bass


def bar_features(samples, bars, bands=None, chroma=None):
    """Per-bar feature dicts on the given bar grid (times in s)."""
    if bands is None or chroma is None:
        bands, chroma = F.frame_track(samples)
    chroma = _chroma_c(chroma)
    mean = np.maximum(bands.mean(axis=0), 1e-10)
    pats = bar_patterns(bands, bars)
    out = []
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
        prof = 10.0 * np.log10(seg.mean(axis=0) + 1e-12)          # 32-band log profile (timbre)
        pat = pats[i] if i < len(pats) else {k: [0.0] * STEPS for k in DRUM_BANDS}
        low_hits, high_hits = _busyness(pat, dur)
        out.append({"t": float(bars[i]), "energy_db": float(10 * np.log10(tot + 1e-12)),
                    "bass": float(bass / ssum), "mid": float(mid / ssum), "high": float(high / ssum),
                    "low_hits": low_hits, "high_hits": high_hits,
                    "chroma": [round(float(x), 5) for x in c],
                    "bass_chroma": [round(float(x), 5) for x in cb],
                    "profile": [round(float(x), 2) for x in prof],
                    "pattern": pat})
    return out


ACTIVE_STEP = 0.5


def _busyness(pat, dur):
    """Hits per second from a bar's folded pattern: the 16th steps whose
    onset strength clears half the bar's max - low = kick, high = the mean
    of snare and hat. Level-independent, unlike counting onsets above a
    whole-track percentile (which measured where the loudest onsets fell)."""
    dur = max(float(dur), 1e-6)
    low = sum(1 for v in pat.get("kick", []) if v >= ACTIVE_STEP) / dur
    high = 0.5 * (sum(1 for v in pat.get("snare", []) if v >= ACTIVE_STEP) + sum(1 for v in pat.get("hat", []) if v >= ACTIVE_STEP)) / dur
    return float(low), float(high)


DRUM_BANDS = {"kick": (0, 4), "snare": (8, 16), "hat": (24, 32)}      # of the 32 DJ bands
HIT_THR = {"kick": 0.55, "snare": 0.65, "hat": 0.55}                   # of the folded pattern's max
STEPS = 16


def _band_onsets(bands):
    """Positive spectral flux per drum class, per analysis frame."""
    out = {}
    for name, (lo, hi) in DRUM_BANDS.items():
        e = bands[:, lo:hi].mean(axis=1)
        e = np.log1p(e / max(float(e.mean()), 1e-9))
        d = np.maximum(np.diff(e, prepend=e[0]), 0.0)
        out[name] = d
    return out


def bar_patterns(bands, bars):
    """Per bar: {"kick": [16], "snare": [16], "hat": [16]} - onset strength
    folded onto the 16th grid, normalised to the bar's max (0..1)."""
    ons = _band_onsets(bands)
    out = []
    for i in range(len(bars) - 1):
        t0, t1 = float(bars[i]), float(bars[i + 1])
        f0, f1 = int(t0 * F.FPS), int(t1 * F.FPS)
        if f1 <= f0 + STEPS or f1 > len(bands):
            break
        pat = {}
        edges = np.linspace(f0, f1, STEPS + 1)
        for name, d in ons.items():
            v = np.array([float(d[int(edges[k]):max(int(edges[k]) + 1, int(edges[k + 1]))].max()) for k in range(STEPS)])
            pat[name] = [round(float(x), 3) for x in (v / max(float(v.max()), 1e-9))]
        out.append(pat)
    return out


def section_pattern(bar_pats, b0, b1, thr=0.45):
    """Average pattern over bars [b0, b1) -> {"kick": [(step, vel)], "snare": [...], "hat": [...],
    "grid": {"kick": [16 floats], ...}} - the hits are the steps whose mean strength clears thr."""
    seg = bar_pats[b0:b1]
    if not seg:
        return None
    out = {"grid": {}}
    for name in seg[0].keys():                       # kick/snare/hat from the band reader, or every identified sound
        m = np.array([bp.get(name, [0.0] * STEPS) for bp in seg]).mean(axis=0)
        m = m / max(float(m.max()), 1e-9)
        out["grid"][name] = [round(float(x), 3) for x in m]
        hits = [(k, round(float(m[k]), 2)) for k in range(STEPS) if m[k] >= max(thr, HIT_THR.get(name, thr))]
        out[name] = hits[:12]
    return out


def phrase_templates(bar_pats, b0, b1, thr=0.45):
    """One drum template per 4-bar phrase of bars [b0, b1): the phrase's
    mean pattern (as section_pattern) plus "fill" - the phrase's last bar
    on its own when its kick/snare differ from the phrase mean (a fill)."""
    out = []
    for p in range(b0, b1, 4):
        tpl = section_pattern(bar_pats, p, min(p + 4, b1), thr)
        if tpl is None:
            break
        names = [k for k in tpl if k != "grid"]
        entry = {k: tpl[k] for k in names}
        last = min(p + 4, b1) - 1
        if last > p:
            lb = section_pattern(bar_pats, last, last + 1, thr)
            if lb:
                diff = 0.0
                for k in [k for k in ("kick", "snare", "tom", "perc") if k in tpl["grid"]]:
                    a, b = np.asarray(tpl["grid"][k]), np.asarray(lb["grid"][k])
                    na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
                    diff = max(diff, 1.0 - (float(a @ b / (na * nb_)) if na > 1e-9 and nb_ > 1e-9 else 1.0))
                if diff > 0.25:
                    entry["fill"] = {k: lb[k] for k in names}
        out.append(entry)
    return out


DB_PER_UNIT = 2.0        # energy_db is 10 log10 of a mean AMPLITUDE (the DJ bands are sqrt(power)): 1 unit = 2 dB of gain


def _dynamics(feats, b0, nbars):
    """Per-bar dB (real dB of gain) relative to the section's mean level (the "dyn" entry)."""
    seg = [f["energy_db"] for f in feats[b0:b0 + nbars]]
    if len(seg) < 2:
        return None
    m = float(np.mean(seg))
    out = [round(float(np.clip(DB_PER_UNIT * (x - m), -12.0, 12.0)), 1) for x in seg]
    if max(abs(x) for x in out) < 0.3:
        return None
    return out


QUALITIES = {"maj": (0, 4, 7), "min": (0, 3, 7), "sus2": (0, 2, 7), "sus4": (0, 5, 7)}


def _quality_fit(chroma, root_pc):
    """Cosine of a bar's chroma against the four triad qualities on a root."""
    c = np.asarray(chroma, dtype=float)
    n = np.linalg.norm(c)
    if n < 1e-9:
        return {}
    c = c / n
    out = {}
    for q, ivs in QUALITIES.items():
        t = np.zeros(12)
        for w, iv in zip((1.0, 0.8, 0.8), ivs):
            t[(root_pc + iv) % 12] = w
        out[q] = float(c @ (t / np.linalg.norm(t)))
    return out


ROOT_W = {"harm": 1.0, "bass_chroma": 0.6, "bass_pc": 0.25, "hold": 0.04}


def read_chords(feats, key_pc, mode, bass_pcs=None, harm_chroma=None, bass_chroma=None, margin=0.08):
    """A chord per bar. The ROOT is the diatonic degree that the evidence
    agrees on: the harmonic chroma's fit to the degree's triad, the bar's
    bass chroma at the degree's root, the transcribed bass pitch class
    when there is one, and a small preference for holding the previous
    chord (loops hold). The QUALITY is a chroma check on that root
    (major / minor / sus2 / sus4); one that is not the key's own on that
    degree is written as {"deg", "third", "sus"} so the harmony spells
    what the song plays. Returns [int | dict per bar]."""
    from lib.gen.theory import Key
    k = Key(key_pc, "minor" if mode == "minor" else "major")
    temps = _triad_templates(key_pc, mode)
    roots = [k.degree_pc(d) for d in range(7)]
    own_third = {d: (k.degree_pc(d + 2) - k.degree_pc(d)) % 12 for d in range(7)}
    out = []
    prev = None
    for i, f in enumerate(feats):
        harm = np.asarray(harm_chroma[i] if harm_chroma is not None and i < len(harm_chroma) else f["chroma"], dtype=float)
        cb = np.asarray(bass_chroma[i] if bass_chroma is not None and i < len(bass_chroma) else (f.get("bass_chroma") or np.zeros(12)), dtype=float)
        n = np.linalg.norm(harm)
        hn = harm / n if n > 1e-9 else harm
        fits = np.array([float(hn @ t) for t in temps])
        cbn = cb / max(float(cb.max()), 1e-9)
        bpc = bass_pcs[i] if bass_pcs is not None and i < len(bass_pcs) else None
        score = ROOT_W["harm"] * fits + ROOT_W["bass_chroma"] * np.array([cbn[r] for r in roots])
        if bpc is not None:
            for d, r in enumerate(roots):
                if r == int(bpc) % 12:
                    score[d] += ROOT_W["bass_pc"]
        if prev is not None:
            score[prev] += ROOT_W["hold"]
        deg = int(np.argmax(score)) if n > 1e-9 or cb.sum() > 1e-9 else (prev if prev is not None else 0)
        prev = deg
        # the quality on that root (only when the chroma clearly prefers it over the key's own)
        q = _quality_fit(harm, roots[deg])
        entry = deg
        if q:
            own = "min" if own_third[deg] == 3 else "maj"
            qbest = max(q, key=q.get)
            others = max(v for kq, v in q.items() if kq != qbest)
            if qbest != own and q[qbest] >= q[own] + margin and q[qbest] >= others + 0.5 * margin:
                if qbest in ("maj", "min"):
                    entry = {"deg": deg, "third": qbest}
                else:
                    entry = {"deg": deg, "sus": int(qbest[-1])}
        out.append(entry)
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


def chords_from_bass(feats, bass_pcs, key_pc, mode, harm_chroma=None, bass_chroma=None):
    """Chords rooted on the transcribed bass note, quality from the chroma
    (read_chords with the stems' evidence)."""
    return read_chords(feats, key_pc, mode, bass_pcs=bass_pcs, harm_chroma=harm_chroma, bass_chroma=bass_chroma)


def _section_chords(chords, b0, nbars):
    """The section's chord list, one per bar (cycled / padded to nbars)."""
    seg = list(chords[b0:b0 + nbars])
    if not seg:
        return [0, 0, 0, 0]
    while len(seg) < nbars:
        seg.append(seg[len(seg) % max(1, min(4, len(seg)))])
    return seg[:nbars]


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


def _drums_kind(sig):
    low = np.asarray(sig.get("low") or [], dtype=float)
    mid = np.asarray(sig.get("mid") or [], dtype=float)
    if len(low) < 16:
        return "unknown"
    L = low.reshape(-1, 16).mean(axis=0) if len(low) % 16 == 0 else low[:16]
    M = mid.reshape(-1, 16).mean(axis=0) if len(mid) >= 16 and len(mid) % 16 == 0 else (mid[:16] if len(mid) >= 16 else np.zeros(16))
    on_beats = L[[0, 4, 8, 12]].mean()
    off = np.delete(L, [0, 4, 8, 12]).mean()
    if on_beats > 1.6 * (off + 1e-6) and L[[4, 8, 12]].min() > 0.5 * L[0]:
        return "four"
    if M.size == 16 and M[8] > 1.3 * max(M[4], M[12], 1e-6):
        return "halftime"
    if M.size == 16 and M[4] > 0.6 * M.max() and M[12] > 0.6 * M.max() and L[10] > 0.5 * L[0]:
        return "breakbeat"
    return "broken"


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


def ingest(path, progress=None, deep=False, reuse=False, out_dir=None, want=("kit", "vocals", "hook", "bass", "bank", "loops")):
    """Analyse an audio file -> {"script": SongScript, "features": [bar feats], "analysis": dict, "bars": [s]}.
    reuse=True also separates the song into stems and puts its own drums,
    vocal phrases and transcribed hook into the script (lib/gen/analysis/reuse.py)."""
    samples = F.decode_file(path)
    title = os.path.splitext(os.path.basename(path))[0]
    structure = structure_allin1(path) if os.environ.get("GEN_STRUCTURE", "1") != "0" else None
    res = ingest_samples(samples, title=title, progress=progress, deep=deep, structure=structure)
    res["analysis"]["structure_model"] = "allin1" if structure else "dj"
    if reuse:
        from lib.gen.analysis import reuse as R
        out_dir = out_dir or os.path.join("logs", "analysis", title)
        stereo = F.decode_file_stereo(path)
        a = res["analysis"]
        from lib.gen.theory import parse_key
        key = parse_key(res["script"]["key"])
        mat = R.reuse(stereo, res["bars"], key.root, "minor" if key.mode != "major" else "major", out_dir,
                      progress=(lambda p, what: progress(0.5 + 0.5 * p, what)) if progress else None, want=want,
                      sections=res["script"]["sections"])
        res["material"] = mat
        sc = res["script"]
        if mat.get("bass_pcs") or mat.get("harm_chroma"):
            rich = chords_from_bass(res["features"], mat.get("bass_pcs"), key.root, "minor" if key.mode != "major" else "major",
                                    harm_chroma=mat.get("harm_chroma"), bass_chroma=mat.get("bass_chroma"))
            from lib.gen.script import chord_deg
            res["chords"] = [chord_deg(c) for c in rich]
            res["chords_rich"] = rich
            b = 0
            for e in sc["sections"]:
                e["chords"] = _section_chords(rich, b, e["bars"])
                b += e["bars"]
        if mat.get("bass_cells"):
            sc["bass_cells"] = R.bass_cell_library(mat["bass_cells"])       # a library the bass generator draws from
        if mat.get("bank"):
            sc["bank"] = mat["bank"]
        if mat.get("bass_bank"):
            sc["bass_bank"] = mat["bass_bank"]
        for key_name, mat_key in (("melody", "melody"), ("bass_line", "bass_line")):
            if mat.get(mat_key):
                b = 0
                for e in sc["sections"]:
                    b0, b1 = b, b + e["bars"]
                    notes = [[bar - b0, st, midi, dur, vel] for bar, st, midi, dur, vel in mat[mat_key] if b0 <= bar < b1]
                    if notes:
                        e[key_name] = notes
                    b = b1
        if mat.get("melody"):
            motifs = R.melody_motifs(mat["melody"], res["chords"], key.root, "minor" if key.mode != "major" else "major")
            if motifs:
                sc["motifs"] = motifs                     # the generator's motif memory, seeded from the song
                for e in sc["sections"]:
                    if e["section"] in ("drop", "groove", "build") and e.get("energy", 0) >= 0.7 and not e.get("hook"):
                        e["hook"] = {k: motifs[0][k] for k in ("steps", "degrees", "contour", "name")}
                        break
        if mat.get("loops"):
            n_loops = 0
            for e, lp in zip(sc["sections"], mat["loops"]):
                files = {k: v for k, v in lp.items() if not k.startswith("_")}
                if files:
                    e["loops"] = files
                    n_loops += 1
            if n_loops:
                sc["fidelity"] = 0.0          # programmatic by default: the song's MATERIAL through the generator's mechanisms;
                                              # the slider adds the source loops as a reference (1.0 = the loops themselves)
        if mat.get("kit"):
            sc["kit"] = mat["kit"]
            if mat.get("kit_db"):
                sc["kit_db"] = mat["kit_db"]              # each sound's level in the song, relative to the loudest
        if mat.get("pad"):
            sc["pad"] = dict(mat["pad"])
            # the texture is the chord itself: pitch it by the ROOT sounding in the bars it was cut from, so the
            # drone plays untransposed on the tonic (pyin heard its fifth)
            pb = mat["pad"].get("bar")
            if pb is not None and res.get("chords_rich") and 0 <= pb < len(res["chords_rich"]):
                from lib.gen.script import chord_deg
                root_pc = key.degree_pc(chord_deg(res["chords_rich"][pb]))
                heard = int(sc["pad"].get("base_midi", 60))
                sc["pad"]["base_midi"] = min((m for m in range(24, 96) if m % 12 == root_pc), key=lambda m: abs(m - heard))
        if mat.get("instruments"):
            # the melodic layers follow the identified instruments: lead = the first, keys = the second,
            # arp only when the song has a third; each plays in the sections where its notes are
            lines = [inst.get("line") or [] for inst in mat["instruments"]]
            if mat.get("melody") and lines and not lines[0]:
                lines[0] = mat["melody"]
            slot_of = ("lead", "keys", "arp")
            b = 0
            for e in sc["sections"]:
                b0, b1 = b, b + e["bars"]
                b += e["bars"]
                on = set()
                for k, line in enumerate(lines[:3]):
                    if sum(1 for n in line if b0 <= n[0] < b1) >= 2:
                        on.add(slot_of[k])
                e["layers"] = sorted((set(e.get("layers") or []) - {"lead", "keys", "arp"}) | on)
        if mat.get("bank_keys"):
            sc["bank_keys"] = mat["bank_keys"]
        if mat.get("drum_grids"):
            # the beat per IDENTIFIED SOUND (template NMF on the drum stem) replaces the three-band reading:
            # templates per section / phrase for every slot the song's kit has, and the layers follow the song
            grids = mat["drum_grids"]
            from lib.gen.events import DRUM_SLOTS
            b = 0
            for e in sc["sections"]:
                b0, b1 = b, min(b + e["bars"], len(grids))
                b += e["bars"]
                if b1 <= b0:
                    continue
                drums = section_pattern(grids, b0, b1)
                if not drums:
                    continue
                slots_on = [k for k in drums["grid"] if any(np.asarray(g.get(k, [0.0])).max() >= 0.5 for g in grids[b0:b1])]
                e["drums"] = {k: drums[k] for k in drums["grid"]}
                e["drums_grid"] = drums["grid"]
                phrases = phrase_templates(grids, b0, b1)
                if len(phrases) > 1:
                    e["drums_phrases"] = phrases
                # and the beat bar by bar: every hit above 0.2 of the sound's max (the kit plays the strong ones always)
                e["drums_bars"] = [{k: [[st, round(float(v), 2)] for st, v in enumerate(g.get(k) or []) if v >= 0.2] for k in slots_on}
                                   for g in grids[b0:b1]]
                e["layers"] = sorted((set(e.get("layers") or []) - set(DRUM_SLOTS)) | set(slots_on))
            a["drum_sounds"] = mat.get("drum_sounds", [])
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


def _sections_from_structure(segments, beats, bands, onset_perc):
    """DJ-style section dicts (energy, shares, rhythm_density, kind) on the
    structure model's segments, snapped to beats."""
    beats = np.asarray(beats, dtype=np.float64)
    mean = np.maximum(bands.mean(axis=0), 1e-10)
    total_pow = (bands / mean).mean(axis=1)
    p95 = max(np.percentile(total_pow, 95), 1e-9)
    out = []
    thr = np.percentile(onset_perc, 75)
    beat_rate = 1.0 / max(float(beats[1] - beats[0]), 1e-6) if len(beats) > 1 else 2.0
    for sg in segments:
        b0 = int(np.searchsorted(beats, sg["start_s"]))
        b1 = int(np.searchsorted(beats, sg["end_s"]))
        if b1 - b0 < 4:
            continue
        f0, f1 = int(sg["start_s"] * F.FPS), min(int(sg["end_s"] * F.FPS), len(bands))
        if f1 <= f0:
            continue
        seg_bands = bands[f0:f1] / mean
        bass, mid, high = seg_bands[:, 0:6].mean(), seg_bands[:, 6:20].mean(), seg_bands[:, 20:32].mean()
        ssum = max(bass + mid + high, 1e-9)
        hits = float((onset_perc[f0:f1] > thr).mean()) * F.FPS
        out.append({"start_s": float(sg["start_s"]), "end_s": float(sg["end_s"]), "start_beat": b0, "end_beat": b1,
                    "energy": round(float(np.clip(total_pow[f0:f1].mean() / p95, 0, 1.5)), 3),
                    "bass_share": round(float(bass / ssum), 3), "mid_share": round(float(mid / ssum), 3),
                    "high_share": round(float(high / ssum), 3), "rhythm_density": round(hits / max(beat_rate, 1e-6), 3),
                    "kind": {"drop": "groove", "break": "breakdown"}.get(sg["kind"], sg["kind"]), "label": sg.get("label")})
    return out


_ALLIN1_BROKEN = False


def structure_allin1(path):
    """Beats, downbeats and functional sections from the allin1 model
    (optional; the DJ planner uses it too). None when unavailable."""
    try:
        import allin1
    except Exception:
        return None
    global _ALLIN1_BROKEN
    if _ALLIN1_BROKEN:
        return None
    try:
        r = allin1.analyze(path, include_activations=False, include_embeddings=False)
    except Exception as e:  # noqa: BLE001
        _ALLIN1_BROKEN = True
        print(f"[analysis] allin1 unavailable ({type(e).__name__}: {str(e)[:80]}); DJ segmentation used")
        return None
    label_map = {"chorus": "drop", "verse": "groove", "inst": "groove", "solo": "groove", "bridge": "break", "break": "break",
                 "intro": "intro", "outro": "outro", "start": "intro", "end": "outro"}
    segs = [{"start_s": float(sg.start), "end_s": float(sg.end), "label": str(sg.label), "kind": label_map.get(str(sg.label), "groove")}
            for sg in r.segments]
    return {"bpm": float(r.bpm), "beats": [float(b) for b in r.beats], "downbeats": [float(b) for b in r.downbeats], "segments": segs}


def ingest_samples(samples, title="ingested", progress=None, deep=False, structure=None):
    if progress:
        progress(0.05, "framing")
    bands, chroma = F.frame_track(samples)
    onset_broad, onset_bass, onset_perc, novelty = F._onset_channels(bands)
    onset_mix = onset_broad + 0.5 * onset_perc
    if progress:
        progress(0.25, "beat grid")
    grid, bpm, bpm_conf, beats = F.estimate_beat_grid(onset_mix)
    downbeat, db_conf = F.estimate_downbeat(beats, bands, chroma, onset_bass, onset_broad)
    if structure and structure.get("downbeats") and len(structure["downbeats"]) >= 4:
        # the structure model knows the downbeats: rebuild the beat list from them
        dbs = np.asarray(structure["downbeats"], dtype=np.float64)
        beats = np.concatenate([np.linspace(dbs[i], dbs[i + 1], 4, endpoint=False) for i in range(len(dbs) - 1)] + [dbs[-1:]])
        downbeat, db_conf = 0, 0.99
        if structure.get("bpm"):
            bpm, bpm_conf = float(structure["bpm"]), max(bpm_conf, 0.9)
    frame_energy = (bands / np.maximum(bands.mean(axis=0), 1e-10)).mean(axis=1)
    key_pc, key_mode, camelot, key_conf = F.estimate_key(chroma, frame_energy)
    if progress:
        progress(0.5, "sections")
    sections, _nov = F.build_sections(bands, chroma, beats, downbeat, onset_perc, novelty)
    if structure and structure.get("segments"):
        sections = _sections_from_structure(structure["segments"], beats, bands, onset_perc)
    try:
        from lib.dj.rhythm import rhythm_signature
        sig = rhythm_signature(bands, grid, downbeat) or {}
    except Exception:
        sig = {}
    bars = _bar_grid(beats, downbeat)
    if progress:
        progress(0.7, "features")
    feats = bar_features(samples, bars, bands, chroma)
    bar_pats = bar_patterns(bands, bars)
    for f, bp in zip(feats, bar_pats):
        f["pattern"] = bp
    key_pc, key_mode = _refine_key(feats, key_pc, key_mode)
    rich = read_chords(feats, key_pc, key_mode)              # root from the bar's bass chroma, quality checked on the chroma
    from lib.gen.script import chord_deg
    chords = [chord_deg(c) for c in rich]
    emax = max([s["energy"] for s in sections] + [1e-6])
    high_mean = float(np.mean([s["high_share"] for s in sections])) if sections else 0.2
    bass_mean = float(np.mean([s["bass_share"] for s in sections])) if sections else 0.3
    style = _style_from(bpm, sig, sections)
    # DJ swing = the offbeat position 0.5 (straight) .. 0.67 (triplet); the composer wants 0 .. 0.33
    swing = float(np.clip((float(sig.get("swing", 0.5) or 0.5) - 0.5) * 2.0, 0.0, 0.33))
    entries = []
    bar_t = list(bars)
    song_db = float(np.mean([f["energy_db"] for f in feats])) if feats else 0.0
    for i, sec in enumerate(sections):
        kind = SECTION_MAP.get(sec.get("kind", "groove"), "groove")
        e = sec["energy"] / emax
        prev_kind = sections[i - 1].get("kind") if i else None
        if kind == "groove" and e >= 0.85 and prev_kind in ("breakdown", "build"):
            kind = "drop"
        nbars = max(4, int(round((sec["end_beat"] - sec["start_beat"]) / 16.0)) * 4)
        # bars of this section on the grid -> its chord loop (first 4 bars, most common per position)
        b0 = int(np.searchsorted(bars, sec["start_s"] - 1e-3))
        loop = _section_chords(rich, b0, nbars)
        dens = float(np.clip(0.35 + sec.get("rhythm_density", 3.0) / 6.0, 0.4, 1.3))
        e_lever = float(np.clip(1.3 * e - 0.3, 0.1, 1.0))      # DJ energy (RMS-ish) -> the form lever
        bright = float(np.clip(1.0 + (sec["high_share"] - high_mean) * 3.0, 0.6, 1.5))
        b1 = int(np.searchsorted(bars, sec["end_s"] - 1e-3))
        drums = section_pattern(bar_pats, b0, max(b0 + 1, b1))
        entry = {"section": kind, "bars": nbars, "energy": round(e_lever, 3),
                 "density": round(dens, 2), "brightness": round(bright, 2), "swing": round(swing, 3),
                 "layers": _layers(sec, kind, emax, high_mean, bass_mean), "chords": loop}
        if kind == "break" and sec["high_share"] < 0.6 * high_mean:
            entry["lanes"] = {"lp": 2500.0}
        if drums and (drums["kick"] or drums["snare"] or drums["hat"]):
            entry["drums"] = {k: drums[k] for k in ("kick", "snare", "hat")}
            entry["drums_grid"] = drums["grid"]
            phrases = phrase_templates(bar_pats, b0, min(b0 + nbars, len(bar_pats)))
            if len(phrases) > 1:
                entry["drums_phrases"] = phrases            # the beat phrase by phrase, fills where the song has them
        dyn = _dynamics(feats, b0, nbars)
        if dyn:
            entry["dyn"] = dyn                               # the section's bar-by-bar dynamics
        seg_db = [f["energy_db"] for f in feats[b0:b0 + nbars]]
        if seg_db and feats:
            entry["level"] = round(DB_PER_UNIT * float(np.mean(seg_db) - song_db), 1)   # the section's level (dB) against the song's mean
        entries.append(entry)
    style_mode = {"groove": "minor", "techno": "phrygian", "trance": "minor", "dnb": "minor", "hiphop": "dorian",
                  "downtempo": "dorian", "ambient": "lydian"}[style]
    from lib.gen.theory import Key
    key = Key(key_pc, "minor" if key_mode == "minor" else "major")
    key_txt = key.camelot if key.camelot != "?" else key.name
    script = {"title": title, "style": style, "bpm": round(float(bpm), 2), "bpm_src": round(float(bpm), 2), "key": key_txt, "seed": 1,
              "humanize": 1.0, "end": True, "sections": entries, "fx": False,      # the form's risers/impacts are not the song's
              "level_ref_db": round(float(20.0 * np.log10(float(np.sqrt(np.mean(np.asarray(samples, dtype=np.float32) ** 2))) + 1e-9)), 1)}
    if progress:
        progress(1.0, "done")
    beat_s = float(np.median(np.diff(np.asarray(beats)))) if len(beats) > 2 else 60.0 / max(bpm, 1e-6)
    analysis = {"bpm": float(bpm), "bpm_conf": float(bpm_conf), "key": key.name, "camelot": key.camelot, "key_conf": float(key_conf),
                "downbeat_conf": float(db_conf), "duration_s": len(samples) / RATE, "n_sections": len(sections),
                "beat": {"beat_s": round(beat_s, 4), "beats": len(beats), "bars": len(bars), "first_beat_s": round(float(beats[0]), 3) if len(beats) else 0.0,
                         "downbeat_offset": int(downbeat), "swing": swing, "drums_kind": _drums_kind(sig),
                         "pattern": section_pattern(bar_pats, 0, len(bar_pats)) if bar_pats else None},
                "sections": sections, "rhythm": {k: sig.get(k) for k in ("swing", "swing_conf", "w_low", "w_mid", "w_high")},
                "style_mode": style_mode, "first_bar_s": float(bars[0]) if len(bars) else 0.0}
    return {"script": script, "features": feats, "analysis": analysis, "bars": [float(b) for b in bars], "chords": chords, "chords_rich": rich}


def features_on_grid(samples, bpm, first_bar_s=0.0):
    """Feature track for audio whose bars are known (a recreation): bars
    every 4 beats from first_bar_s."""
    bar_len = 4 * 60.0 / float(bpm)
    n = int((len(samples) / RATE - first_bar_s) // bar_len)
    bars = np.array([first_bar_s + i * bar_len for i in range(n + 1)])
    bands, chroma = F.frame_track(samples)
    feats = bar_features(samples, bars, bands, chroma)
    for f, bp in zip(feats, bar_patterns(bands, bars)):
        f["pattern"] = bp
    return feats
