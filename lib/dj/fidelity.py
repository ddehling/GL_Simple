"""How close is a reconstruction to the song, and how much of the song
did the program language capture? One place for both measures, so the
evaluator (tools/tests/_dj_recon_eval.py), the reading's own check
(lib/dj/explain.py) and the gen console's Analysis tab quote the same
numbers.

WAVEFORM (compare / evaluate): a rendered stem against the real stem
    level       rms difference (dB)
    spectral    mean |dB| over a 64-band log-mel, per beat on beats where
                the stem sounds, over the cells that carry the beat
                (within 40 dB of its loudest); 0 = identical, 6-8 = the
                same part on another instrument, 12+ = a different sound
    activity_r  Pearson r between 16th-note rms envelopes (the rhythm)
    missed /    share of active 16ths the render misses / adds
    spurious
    onset_f1    onset times within 40 ms (drums, plucked parts)
    onset_off   signed offset of matched onsets, ms (+ = late), and its sd
    chroma_r    per-beat chroma correlation on active beats (pitched
                stems: the notes, octave-blind)

NOTES (notes_report): how programmatic the program is
    events, patterns, ops, bar entries, events per pattern+op, the share
    of events that patterns carry vs. ops (deviations written one by
    one) vs. verbatim bars, vocal phrases vs. distinct recordings, and
    per voice the share of the stem it explains (lib/dj/explain.py).
"""
import json
import os

import numpy as np

RATE = 44100
HOP = 512


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)) + 1e-9)


def mel_db(y):
    import librosa
    M = librosa.feature.melspectrogram(y=np.ascontiguousarray(y, dtype=np.float32), sr=RATE, n_fft=2048, hop_length=HOP, n_mels=64,
                                       fmin=30.0, fmax=16000.0, power=2.0)
    return 10 * np.log10(M + 1e-10)


def env16(y, beats, period):
    """rms per 16th-note step on the grid -> (n_steps,)"""
    steps = []
    for k in range(len(beats)):
        nxt = beats[k + 1] if k + 1 < len(beats) else beats[k] + period
        for s in range(4):
            a = int((beats[k] + s / 4 * (nxt - beats[k])) * RATE)
            b = int((beats[k] + (s + 1) / 4 * (nxt - beats[k])) * RATE)
            seg = y[a:b]
            steps.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    return np.array(steps)


def onsets(y):
    import librosa
    env = librosa.onset.onset_strength(y=np.ascontiguousarray(y, dtype=np.float32), sr=RATE, hop_length=256)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=RATE, hop_length=256, units="frames", backtrack=False, delta=0.15, wait=5)
    return np.array(fr) * 256 / RATE


def f1(a, b, tol=0.04):
    if not len(a) or not len(b):
        return 0.0
    b = np.sort(b)
    hit = 0
    for t in a:
        k = np.searchsorted(b, t)
        d = min(abs(b[min(k, len(b) - 1)] - t), abs(b[max(k - 1, 0)] - t))
        hit += d <= tol
    prec = hit / len(a)
    hit2 = 0
    a_s = np.sort(a)
    for t in b:
        k = np.searchsorted(a_s, t)
        d = min(abs(a_s[min(k, len(a_s) - 1)] - t), abs(a_s[max(k - 1, 0)] - t))
        hit2 += d <= tol
    rec = hit2 / len(b)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def chroma_beats(y, beats, period):
    import librosa
    C = librosa.feature.chroma_cqt(y=np.ascontiguousarray(y, dtype=np.float32), sr=RATE, hop_length=HOP)
    out = []
    for k in range(len(beats)):
        nxt = beats[k + 1] if k + 1 < len(beats) else beats[k] + period
        f0, f1_ = int(beats[k] * RATE / HOP), int(nxt * RATE / HOP)
        out.append(C[:, f0:max(f1_, f0 + 1)].mean(axis=1) if f0 < C.shape[1] else np.zeros(12))
    return np.array(out)


def lag_ms(e_ref, e_rec, step_s):
    if e_ref.std() < 1e-9 or e_rec.std() < 1e-9:
        return 0.0
    a = (e_ref - e_ref.mean()) / e_ref.std()
    b = (e_rec - e_rec.mean()) / e_rec.std()
    best, best_l = -2, 0
    for lag in range(-3, 4):                       # +-3 sixteenths
        if lag >= 0:
            r = float(np.mean(a[lag:] * b[: len(b) - lag])) if lag < len(a) else -2
        else:
            r = float(np.mean(a[: lag] * b[-lag:]))
        if r > best:
            best, best_l = r, lag
    return -best_l * step_s * 1000.0


def spectral_gap(ref, rec, beats, period, Mr=None):
    """The spectral figure alone (fast; Mr = mel_db(ref) may be passed in
    when the reference is compared several times). -> (gap dB, active beats)"""
    Mr = mel_db(ref) if Mr is None else Mr
    Mc = mel_db(rec)
    n = min(Mr.shape[1], Mc.shape[1])
    active, diffs = [], []
    ref_floor = np.percentile(Mr.max(axis=0), 95) - 30
    for k in range(len(beats)):
        f0, f1_ = int(beats[k] * RATE / HOP), int((beats[k] + period) * RATE / HOP)
        if f1_ > n:
            break
        if Mr[:, f0:f1_].max() < ref_floor:
            continue
        # cells that CARRY the beat: within 40 dB of its loudest cell in either signal. Near-silent
        # cells (leakage, floor) were dominating the mean and punished a spectrally right render with
        # different sidelobes by 10 dB (found on the synthetic gate with an additive render: level and
        # chroma exact, "13 dB")
        seg_r, seg_c = Mr[:, f0:f1_], Mc[:, f0:f1_]
        floor = max(float(seg_r.max()), float(seg_c.max())) - 40.0
        cells = (seg_r > floor) | (seg_c > floor)
        diffs.append(float(np.mean(np.abs(seg_r[cells] - seg_c[cells]))))
        active.append(k)
    return (float(np.mean(diffs)) if diffs else float("nan")), active


def compare(ref, rec, beats, period, pitched):
    """All the waveform figures for one rendered stem (rec) against the
    real one (ref); beats relative to the start of both."""
    out = {"level": db(rec) - db(ref)}
    out["spectral"], active = spectral_gap(ref, rec, beats, period)
    e_ref, e_rec = env16(ref, beats, period), env16(rec, beats, period)
    thr_ref = np.percentile(e_ref, 95) * 10 ** (-25 / 20)
    thr_rec = np.percentile(e_rec, 95) * 10 ** (-25 / 20) if e_rec.max() > 0 else 1e9
    on_ref, on_rec = e_ref > thr_ref, e_rec > max(thr_rec, thr_ref * 0.5)
    out["activity_r"] = float(np.corrcoef(e_ref, e_rec)[0, 1]) if e_ref.std() > 0 and e_rec.std() > 0 else 0.0
    out["missed"] = float((on_ref & ~on_rec).sum() / max(on_ref.sum(), 1))
    out["spurious"] = float((on_rec & ~on_ref).sum() / max(on_rec.sum(), 1))
    out["lag_ms"] = lag_ms(e_ref, e_rec, period / 4)
    o_rec, o_ref = onsets(rec), onsets(ref)
    out["onset_f1"] = f1(o_rec, o_ref)
    # signed offset of matched onsets (recon - stem), ms: + = the recon is late
    diffs = []
    if len(o_rec) and len(o_ref):
        o_ref_s = np.sort(o_ref)
        for t in o_rec:
            k = np.searchsorted(o_ref_s, t)
            cand = [o_ref_s[j] for j in (k - 1, k) if 0 <= j < len(o_ref_s)]
            if cand:
                d = min(cand, key=lambda c: abs(c - t))
                if abs(t - d) <= 0.06:
                    diffs.append((t - d) * 1000.0)
    out["onset_off_ms"] = float(np.median(diffs)) if diffs else float("nan")
    out["onset_off_sd"] = float(np.std(diffs)) if len(diffs) > 3 else float("nan")
    if pitched:
        Cr, Cc = chroma_beats(ref, beats, period), chroma_beats(rec, beats, period)
        rs = []
        for k in active:
            if k < len(Cr) and Cr[k].std() > 1e-6 and Cc[k].std() > 1e-6:
                rs.append(float(np.corrcoef(Cr[k], Cc[k])[0, 1]))
        out["chroma_r"] = float(np.mean(rs)) if rs else float("nan")
    return out


def _roughness(mag, freqs, n_peaks=24):
    """Sensory dissonance of one spectrum (Plomp-Levelt curve as parametrised by Sethares): the
    roughness of every pair of its strongest partials, weighted by their amplitudes, as a share
    of the pair-amplitude mass - 0 = every pair consonant, ~1 = every pair inside the critical
    band. Level-free, so a render can be compared with the recording it stands for."""
    from scipy.signal import find_peaks
    idx, _ = find_peaks(mag, distance=3)
    if len(idx) < 2:
        return 0.0
    idx = idx[np.argsort(mag[idx])[::-1][:n_peaks]]
    a, f = mag[idx], freqs[idx]
    order = np.argsort(f)
    a, f = a[order], f[order]
    fi, fj = f[:, None], f[None, :]
    lo = np.minimum(fi, fj)
    s = 0.24 / (0.021 * lo + 19.0)
    x = s * np.abs(fi - fj)
    d = np.exp(-3.5 * x) - np.exp(-5.75 * x)
    w = a[:, None] * a[None, :]
    iu = np.triu_indices(len(a), 1)
    return float(np.sum(w[iu] * d[iu]) / max(float(np.sum(w[iu])), 1e-12))


def discord(ref, rec, beats, period, pc_floor=0.3):
    """How discordant a render is against the recording it stands for, per beat the recording
    is active on, two ways:
      wrong_pc  - the share of the render's chroma energy on pitch classes the recording does NOT
                  sound on that beat (below pc_floor x its strongest class); the recording's own
                  figure at the same floor is the baseline
      roughness - sensory dissonance (see _roughness) of each signal's beat spectrum; the render's
                  excess over the recording is what a listener hears as clash
    -> {"wrong_pc_rec", "wrong_pc_ref", "roughness_rec", "roughness_ref", "n_beats"}"""
    import librosa
    n = min(len(ref), len(rec))
    ref, rec = np.asarray(ref[:n], dtype=np.float32), np.asarray(rec[:n], dtype=np.float32)
    Mr = mel_db(ref)
    ref_floor = np.percentile(Mr.max(axis=0), 95) - 30
    Cr, Cc = chroma_beats(ref, beats, period), chroma_beats(rec, beats, period)
    n_fft = 4096
    Sr = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=HOP))
    Sc = np.abs(librosa.stft(rec, n_fft=n_fft, hop_length=HOP))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / RATE)
    band = (freqs >= 60) & (freqs <= 6000)
    wrong_rec, wrong_ref, r_rec, r_ref = [], [], [], []
    for k in range(len(beats)):
        f0, f1_ = int(beats[k] * RATE / HOP), int((beats[k] + period) * RATE / HOP)
        if f1_ > min(Mr.shape[1], Sr.shape[1], Sc.shape[1]) or k >= len(Cr):
            break
        if Mr[:, f0:f1_].max() < ref_floor:
            continue
        cr, cc = Cr[k], Cc[k]
        if cr.max() <= 0 or cc.max() <= 0:
            continue
        present = cr >= pc_floor * cr.max()
        wrong_rec.append(float(cc[~present].sum() / max(cc.sum(), 1e-9)))
        wrong_ref.append(float(cr[~present].sum() / max(cr.sum(), 1e-9)))
        r_ref.append(_roughness(Sr[band, f0:f1_].mean(axis=1), freqs[band]))
        r_rec.append(_roughness(Sc[band, f0:f1_].mean(axis=1), freqs[band]))
    med = lambda v: float(np.median(v)) if v else float("nan")
    return {"wrong_pc_rec": med(wrong_rec), "wrong_pc_ref": med(wrong_ref), "roughness_rec": med(r_rec), "roughness_ref": med(r_ref),
            "n_beats": len(wrong_rec)}


# --------------------------------------------------------------------------
# whole programs
# --------------------------------------------------------------------------

def window(result, stems_mono, t0=30.0, span=120.0):
    """The evaluation window: 120 s from 30 s in, or the whole track when
    it is shorter than that."""
    dur = min(len(y) for y in stems_mono.values()) / RATE
    if dur <= t0 + span:
        return 0.0, dur
    return t0, min(dur, t0 + span)


def evaluate(prog, stems_mono, result, t0=None, t1=None, use_residual=False, progress=None):
    """The program's voices rendered per stem against the real stems +
    the mix -> {stem: compare-dict | None, "mix": ..., "window": [t0, t1]}.
    Vocals (phrases / verbatim) render from the stem itself and are marked
    "reused"; use_residual=False keeps the verbatim residual bars of
    `other` out so the figure is the voices' own."""
    from lib.dj import instruments as INS, songprogram as SP
    beats = np.asarray(result["beats"], dtype=np.float64)
    period = float(result["period_s"])
    if t0 is None or t1 is None:
        t0, t1 = window(result, stems_mono)
    sel = (beats >= t0) & (beats < t1 - period)
    bts = beats[sel] - t0
    rows = {"window": [float(t0), float(t1)]}
    n = int((t1 - t0) * RATE)
    mix_ref = np.zeros(n, dtype=np.float32)
    mix_rec = np.zeros(n, dtype=np.float32)
    for stem in INS.STEM_ORDER:
        if stem not in stems_mono:
            rows[stem] = None
            continue
        ref = stems_mono[stem][int(t0 * RATE):int(t1 * RATE)]
        ids = [v for v, voice in prog["voices"].items() if voice["stem"] == stem]
        reused = stem in {v["stem"] for v in prog.get("verbatim", [])} or stem in (prog.get("phrases") or {})
        if (not ids and not reused) or db(ref) < -60:
            rows[stem] = None
            continue
        if progress:
            progress(f"fidelity: rendering {stem}")
        # the stem's voices plus the stem's own name: the bare name selects its phrases / verbatim slices (the
        # hybrid program's recording parts), the voice ids its notes outside them
        # limit=None: the levels as rendered - the default peak clamp scaled a whole stem down by its one
        # hottest voice and made the level column lie (found by the mixer work, 2026-09-09)
        rec = SP.render(prog, {stem: stems_mono[stem]}, ids=ids + [stem], t0=t0, t1=t1, room=False,
                        use_residual=use_residual, limit=None)[:, 0][: len(ref)]
        if len(rec) < len(ref):
            rec = np.concatenate([rec, np.zeros(len(ref) - len(rec), dtype=np.float32)])
        mix_ref[: len(ref)] += ref
        mix_rec[: len(rec)] += rec
        r = compare(ref, rec, bts, period, pitched=stem != "drums")
        r["voices"] = len(ids)
        r["reused"] = bool(reused and not ids)
        rows[stem] = r
    rows["mix"] = compare(mix_ref, mix_rec, bts, period, pitched=True)
    return rows


def notes_report(prog, result=None):
    """How much of the notes the program language holds -> dict."""
    s = prog.get("stats") or {}
    n_events = int(s.get("events", 0))
    # events carried by patterns vs written as ops (deviations)
    op_events = 0
    for vid, seq in (prog.get("sequence") or {}).items():
        for e in seq:
            op_events += sum(1 for op in e.get("ops", []) if op[0] in ("add", "nudge"))
    n_bars = int(s.get("bars", 0) or 0)
    verb = int(s.get("verbatim_bars", 0) or 0)
    ph_n, ph_d = int(s.get("vocal_phrases", 0) or 0), int(s.get("vocal_distinct", 0) or 0)
    voices = []
    # the per-voice figures: from the build's own check when it ran, else from the reading (v7+ readings carry them)
    ex = dict(prog.get("explain") or {})
    if result is not None:
        from lib.dj import instruments as INS
        for inst in INS.instruments(result):
            if inst.get("explained") is not None and inst["id"] not in ex:
                ex[inst["id"]] = {"explained": inst["explained"], "overshoot": inst.get("overshoot")}
    for vid, v in prog["voices"].items():
        pats = prog.get("patterns", {}).get(vid) or {}
        seq = prog.get("sequence", {}).get(vid) or []
        r = ex.get(vid) or {}
        voices.append({"id": vid, "name": v.get("name"), "stem": v["stem"], "model": v.get("model"),
                       "patterns": len(pats), "units": len(seq), "ops": sum(len(e.get("ops", [])) for e in seq),
                       "explained": r.get("explained"), "overshoot": r.get("overshoot")})
    return {"events": n_events, "patterns": int(s.get("patterns", 0)), "entries": int(s.get("sequence", 0)), "ops": int(s.get("ops", 0)),
            "events_per_entry": float(s.get("events_per_entry", 0.0)),
            "pattern_share": (n_events - op_events) / max(n_events, 1), "op_share": op_events / max(n_events, 1),
            "verbatim_bars": verb, "bars": n_bars, "verbatim_share": verb / max(n_bars, 1),
            "vocal_phrases": ph_n, "vocal_distinct": ph_d, "vocal_reuse": 1.0 - ph_d / max(ph_n, 1) if ph_n else 0.0,
            "voices": voices, "pruned": prog.get("pruned") or (result or {}).get("pruned") or []}


def _match_notes(a, b, tol=0.06, octave_blind=False):
    """a, b: [(t, midi)] -> (precision, recall, f1) of a against b: onset within tol AND the pitch."""
    if not a or not b:
        return 0.0, 0.0, 0.0
    ref = sorted(b)
    rt = np.array([t for t, _m in ref]); rm = np.array([m for _t, m in ref])
    used = np.zeros(len(ref), dtype=bool)
    hit = 0
    for t, m in sorted(a):
        lo, hi = np.searchsorted(rt, t - tol), np.searchsorted(rt, t + tol)
        for j in range(lo, hi):
            if used[j]:
                continue
            if (rm[j] == m) if not octave_blind else ((rm[j] - m) % 12 == 0):
                used[j] = True
                hit += 1
                break
    p, r = hit / len(a), hit / len(ref)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def notes_agreement(prog, stems_mono, result, t0=None, t1=None, progress=None):
    """THE NOTE-LEVEL figure: the program's notes against an independent
    transcription of the stem (basic-pitch) - onset within 60 ms and the
    same pitch (F1), octave-blind, and onsets alone. Neither side is the
    truth on a real song (on the synthetic gate the reader scores 1.00 on
    the bass line and 0.84 on the melody, basic-pitch 0.96 and 0.40), but
    a low agreement says the notes are not the song's, whatever the
    spectral gap says (Sussudio 2026-09-09: mix gap 6.4 dB, bass notes
    F1 0.19, and it sounded like nothing). -> {stem: {...}} or {}."""
    from lib.dj import instruments as INS, songprogram as SP
    if t0 is None or t1 is None:
        t0, t1 = window(result, stems_mono)
    out = {}
    for stem in ("bass", "other"):
        if stem not in stems_mono or not any(v["stem"] == stem for v in prog["voices"].values()):
            continue
        if progress:
            progress(f"fidelity: transcribing {stem} for the note check")
        y = stems_mono[stem][int(t0 * RATE):int(t1 * RATE)]
        tr = INS.transcribe(y)
        if tr is None:
            continue
        ref = [(n[0], int(n[2])) for n in tr if n[3] >= 0.3 and (n[1] - n[0]) >= 0.05]
        pn = [(t - t0, int(m)) for vid, v in prog["voices"].items() if v["stem"] == stem
              for t, m, _d, _vel in SP.expand(prog, vid) if m is not None and t0 <= t < t1]
        p, r, f = _match_notes(pn, ref)
        _p2, _r2, f2 = _match_notes(pn, ref, octave_blind=True)
        _p3, _r3, f3 = _match_notes([(t, 0) for t, _m in pn], [(t, 0) for t, _m in ref])
        out[stem] = {"f1": f, "precision": p, "recall": r, "f1_octave_blind": f2, "onset_f1": f3, "n_program": len(pn), "n_transcribed": len(ref)}
    return out


def report(prog, stems_mono, result, progress=None):
    """Both halves + the note-level agreement, ready to save."""
    notes = notes_report(prog, result)
    try:
        notes["agreement"] = notes_agreement(prog, stems_mono, result, progress=progress)
    except Exception as e:  # noqa: BLE001
        notes["agreement"] = {}
        if progress:
            progress(f"note check skipped ({type(e).__name__}: {str(e)[:60]})")
    wave = evaluate(prog, stems_mono, result, progress=progress)
    rep = {"version": 3, "notes": notes, "waveform": wave}
    # the hybrid program: what PLAYS is the voices where they explain the recording and the recording where
    # they do not; "waveform" above is the voices alone (the reader's figure), "waveform_played" the program
    # as heard, with the share of each stem's sounding bars that is notes
    if any(v.get("residual") for v in prog.get("verbatim") or []):
        rep["waveform_played"] = evaluate(prog, stems_mono, result, use_residual=True, progress=progress)
        rep["notes"]["note_share"] = (prog.get("stats") or {}).get("note_share") or {}
    return rep


def save(rep, folder):
    path = os.path.join(folder, "fidelity.json")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 4)
        if isinstance(o, np.integer):
            return int(o)
        return o
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean(rep), fh, indent=1)
    return path


def load(folder):
    path = os.path.join(folder, "fidelity.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def describe(rep):
    """Readable lines for both halves."""
    n, w = rep.get("notes") or {}, rep.get("waveform") or {}
    lines = []
    if n:
        lines.append(f"NOTES AS CODE: {n['events']} events -> {n['patterns']} patterns over {n['entries']} bar entries + {n['ops']} ops "
                     f"({n['events_per_entry']:.1f} events per pattern+op); {100 * n['pattern_share']:.0f}% of events carried by patterns, "
                     f"{100 * n['op_share']:.0f}% written one by one; verbatim {n['verbatim_bars']}/{n['bars']} bars "
                     f"({100 * n['verbatim_share']:.0f}%); vocals {n['vocal_phrases']} phrases from {n['vocal_distinct']} recordings")
        for v in n.get("voices") or []:
            ex = f"explains {100 * v['explained']:.0f}% of its stem" if v.get("explained") is not None else "not measured"
            lines.append(f"   {v['id']:10s} {str(v['name']):10s} {str(v['model']):8s} {v['patterns']:3d} patterns / {v['units']:3d} units / {v['ops']:3d} ops; {ex}")
        for g in n.get("pruned") or []:
            lines.append(f"   {g['id']:10s} dropped: {g['why']}")
        ag = n.get("agreement") or {}
        if ag:
            lines.append("NOTES vs AN INDEPENDENT TRANSCRIPTION (onset within 60 ms and the same pitch; the figure that says whether the notes are the song's):")
            for stem, r in ag.items():
                lines.append(f"   {stem:6s} F1 {r['f1']:.2f} (precision {r['precision']:.2f}, recall {r['recall']:.2f}); octave-blind {r['f1_octave_blind']:.2f}; "
                             f"onsets alone {r['onset_f1']:.2f}; {r['n_program']} program notes vs {r['n_transcribed']} transcribed")
    if w:
        t0, t1 = w.get("window", [0, 0])
        lines.append(f"WAVEFORM vs THE ORIGINAL ({t0:.0f}-{t1:.0f} s): spectral gap dB (0 = identical, 6-8 = same part other instrument, 12+ = different sound)")
        for stem in ("drums", "bass", "other", "vocals", "mix"):
            r = w.get(stem)
            if not r:
                continue
            ch = f"  chroma {r['chroma_r']:.2f}" if r.get("chroma_r") is not None else ""
            tag = "  (the stem's own phrases reused)" if r.get("reused") else ""
            lines.append(f"   {stem:6s} gap {r['spectral']:5.1f} dB  level {r['level']:+5.1f} dB  rhythm r {r['activity_r']:.2f}  "
                         f"missed {100 * r['missed']:.0f}%  extra {100 * r['spurious']:.0f}%  onsets F1 {r['onset_f1']:.2f}{ch}{tag}")
    wp = rep.get("waveform_played") or {}
    if wp:
        share = (rep.get("notes") or {}).get("note_share") or {}
        lines.append("AS PLAYED (the hybrid program: notes where the voices explain the recording bar by bar, the recording itself elsewhere):")
        for stem in ("drums", "bass", "other", "vocals", "mix"):
            r = wp.get(stem)
            if not r:
                continue
            ns = f"  notes on {100 * share[stem]:.0f}% of its sounding bars" if stem in share else ("  (the recording's phrases)" if stem == "vocals" else "")
            lines.append(f"   {stem:6s} gap {r['spectral']:5.1f} dB  level {r['level']:+5.1f} dB{ns}")
    return lines
