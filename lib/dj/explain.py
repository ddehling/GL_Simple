"""Does each claimed instrument explain the stem? Analysis by synthesis, per
voice, on the song itself.

For every voice of a program its own render is compared with the stem
INSIDE ITS OWN EVENT WINDOWS (the onset and the note's first 300 ms):

    explained   share of the stem's spectral energy there that the voice
                accounts for (1 = all of it; a voice sharing the window
                with others explains its part; a wrong sample or pitch
                explains little)
    overshoot   energy the voice puts where the stem has none, relative
                to the stem's energy there (a too-loud or wrong-timbre
                voice)
    spurious    share of the voice's events whose window holds no stem
                energy at all (events that are not there)
    share_db    the voice render's total energy against the stem's

suggest() turns that into candidates: events in silence go (the whole
voice when nearly all of them are); a voice that explains almost nothing
goes; two voices whose exemplars sound alike and whose events never
coincide are one instrument split by the clustering - IF the keeper's
sound, played at the absorbed voice's events, still explains the stem
there (a kick and a snare also alternate and never coincide; the kick
sample at the snare's hits explains little, so that pair stays two).

verify() is the closed loop: every candidate is kept only when the
WHOLE STEM, rendered from the program, gets closer to the recording
(lib/dj/fidelity.spectral_gap on the evaluation window). Measured
2026-09-09 without it: a voice with 60% of its events in silence still
carried the other 40%, and dropping it cost 7 dB on that stem; a merge
that explained more in its own windows was louder over the stem and
cost 1.7 dB. With it, only what helps sticks.

Everything here is measured on the song; nothing is assumed about what
an instrument "should" be.
"""
import numpy as np

from lib.dj import instruments as INS

RATE = INS.RATE
STEPS = INS.STEPS
WIN_S = 0.3
N_FFT = 2048
SILENT_DB = -55.0              # a window under this (rms dBFS) holds no stem energy
DROP_SPURIOUS = 0.1            # more than this share of events in silence: those events go
PRUNE_SPURIOUS = 0.9           # nearly all events in silence: the voice goes
PRUNE_EXPLAINED = 0.08         # explains under this share of its windows: noise
MERGE_COS = 0.95               # exemplar spectra this alike...
MERGE_COINCIDE = 0.05          # ...with under this share of coinciding events: one sound...
MERGE_KEEP_SHARE = 0.9         # ...if the keeper's sound explains at least this share of what the absorbed one's own did
MERGE_OVERSHOOT_ADD = 0.25     # ...without adding more than this much overshoot
VERIFY_TOL_DB = 0.1            # a candidate must not widen the stem's spectral gap by more than this
OFFPITCH_SHARE = 0.25          # a read pitch with under this share of the strongest pitch's salience over the note is not there
OFFPITCH_MIN = 3               # ...and a voice needs at least this many such events for the candidate
GAIN_STEP_DB = 3.0             # a voice's level moves in steps of this...
GAIN_MAX_DB = 12.0             # ...up to this (beyond it the voice is the wrong sound, not the wrong level)
GAIN_STEPS = False             # OFF (measured 2026-09-09 against the synthetic truth): the stem's spectral gap is not a
                               # level-faithful objective - it shrank 9.3 -> 6.4 dB while the hat and shaker were pushed
                               # 4-6 dB ABOVE their true levels and the kick moved away from its; the mel measure rewards
                               # filling the stem's low-level high-band cells. A level objective that cannot be gamed
                               # that way is needed before this candidate goes back on.


def _spec(x):
    n = len(x)
    if n < 512:
        return None
    return np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) * np.hanning(n), N_FFT))


def _win_level(y, t):
    a = int(t * RATE)
    b = int(min(len(y), (t + WIN_S) * RATE))
    if b - a < 512:
        return -120.0
    seg = y[a:b].astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9)


def explain_voice(prog, stems_mono, vid, max_events=400, seed=0):
    from lib.dj import songprogram as SP
    voice = prog["voices"][vid]
    y = stems_mono.get(voice["stem"])
    if y is None:
        return None
    events = SP.expand(prog, vid)
    if not events:
        return None
    rng = np.random.default_rng(seed)
    if len(events) > max_events:
        events = [events[i] for i in sorted(rng.choice(len(events), size=max_events, replace=False))]
    t0 = max(0.0, min(e[0] for e in events) - 0.5)
    t1 = min(len(y) / RATE, max(e[0] for e in events) + 3.0)
    # fast: the check measures the voice's energy in its own windows, not its articulation (the additive holds and
    # tails were the reading's cost: a real track's self-check tripled with them, 2026-09-09)
    rec = SP.render(prog, {voice["stem"]: y}, ids=[vid], t0=t0, t1=t1, room=False, use_residual=False, limit=None, fast=True)[:, 0]
    explained, overshoot, fits, silent = [], [], [], 0
    for t, midi, dur, vel in events:
        a = int(t * RATE)
        b = int(min(len(y), (t + WIN_S) * RATE))
        ra = int((t - t0) * RATE)
        rb = ra + (b - a)
        if b - a < 512 or rb > len(rec):
            continue
        if _win_level(y, t) < SILENT_DB:
            silent += 1
            continue
        S, R = _spec(y[a:b]), _spec(rec[ra:rb])
        if S is None or R is None:
            continue
        e_s = float(np.sum(S ** 2)) + 1e-12
        explained.append(1.0 - float(np.sum(np.maximum(S - R, 0.0) ** 2)) / e_s)
        overshoot.append(float(np.sum(np.maximum(R - S, 0.0) ** 2)) / e_s)
        # the level the stem has where THIS voice puts its energy (bins within 20 dB of the render's
        # peak): positive = the render is too quiet there. An upper bound when other voices share the
        # band; the stem-level verification decides
        band = R >= R.max() * 10 ** (-20 / 20)
        if band.any() and float(np.sum(R[band] ** 2)) > 0:
            fits.append(10 * np.log10((float(np.sum(S[band] ** 2)) + 1e-12) / (float(np.sum(R[band] ** 2)) + 1e-12)))
    out = {"n": len(events), "spurious": silent / max(len(events), 1),
           "explained": float(np.median(explained)) if explained else 0.0,
           "overshoot": float(np.median(overshoot)) if overshoot else 0.0,
           "share_db": float(20 * np.log10((np.sqrt(np.mean(rec ** 2)) + 1e-9) / (np.sqrt(np.mean(y[int(t0 * RATE):int(t1 * RATE)] ** 2)) + 1e-9))),
           "level_fit_db": float(np.median(fits)) if fits else 0.0,
           "scored": len(explained)}
    return out


def explain_all(prog, stems_mono, progress=None):
    out = {}
    for vid in prog["voices"]:
        try:
            r = explain_voice(prog, stems_mono, vid)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {str(e)[:60]}"}
        if r:
            out[vid] = r
        if progress and r and "error" not in r:
            progress(f"{vid}: explains {100 * r['explained']:.0f}% of its windows, overshoot {100 * r['overshoot']:.0f}%, "
                     f"{100 * r['spurious']:.0f}% events in silence")
    return out


def offpitch_events(inst, y, result):
    """Indices of pitched events whose read pitch has little salience in the stem over the
    note itself (the onset reader's short window and drone subtraction misread; kick bleed
    at the range floor reads as low notes). The closed loop decides whether dropping them
    helps - Sussudio's worst bass beats were all wrong pitch classes, mostly C1-D1 readings."""
    beats = np.asarray(result["beats"], dtype=np.float64)
    period = float(result["period_s"])
    lo, hi = INS.PITCH_RANGE.get(inst["stem"], (24, 96))
    out = []
    for k, e in enumerate(inst["events"]):
        if e[2] is None:
            continue
        t = INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
        a = int((t + 0.03) * RATE)
        b = int(min(len(y), (t + min(0.8, max(0.15, e[3] * period / STEPS))) * RATE))
        if b - a < 2048:
            continue
        R, df = INS._spectrum(y[a:b])
        sal, _ = INS._harmonic_salience(R, df, lo, hi)
        if sal.max() <= 0:
            continue
        m = int(e[2])
        s_read = float(sal[m - lo]) if lo <= m <= hi else 0.0
        if s_read < OFFPITCH_SHARE * float(sal.max()):
            out.append(k)
    return out


def silent_events(inst, y, result):
    """Indices of the instrument's events whose window holds no stem energy."""
    beats = np.asarray(result["beats"], dtype=np.float64)
    out = []
    for k, e in enumerate(inst["events"]):
        t = INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
        if _win_level(y, t) < SILENT_DB:
            out.append(k)
    return out


def _exemplar_spectrum(inst, stems_mono):
    import librosa
    from lib.dj import resynth as RS
    ex = RS.exemplar_audio(inst, stems_mono)
    if ex is None or len(ex) < 1024:
        return None
    M = librosa.feature.melspectrogram(y=np.ascontiguousarray(ex[: int(0.2 * RATE)]), sr=RATE, n_fft=1024, hop_length=256, n_mels=48,
                                       fmin=30.0, fmax=16000.0, power=1.0).mean(axis=1)
    v = np.sqrt(M)
    return v / (np.linalg.norm(v) + 1e-9)


def _with_sound_of(prog, absorb, keep):
    """A shallow variant of the program where `absorb` plays with `keep`'s sound and model."""
    p2 = dict(prog)
    p2["_sounds"] = dict(prog["_sounds"])
    p2["_sounds"][absorb] = prog["_sounds"][keep]
    p2["voices"] = dict(prog["voices"])
    p2["voices"][absorb] = dict(prog["voices"][absorb], model=prog["voices"][keep]["model"],
                                level_dbfs=prog["voices"][keep].get("level_dbfs"), exemplar_midi=prog["voices"][keep].get("exemplar_midi"))
    return p2


def suggest(result, stems_mono, explanation, prog=None):
    """-> list of candidates, each {"kind": "drop"|"prune"|"merge", "id", ("keep"), ("events"), "why"}.
    prog: the program the explanation was measured on; with it every
    merge candidate is checked by synthesis at the absorbed voice's events."""
    insts = {i["id"]: i for i in INS.instruments(result)}
    cands = []
    pruned = set()
    for vid, inst in insts.items():
        y = stems_mono.get(inst["stem"])
        if y is None:
            continue
        sil = silent_events(inst, y, result)
        share = len(sil) / max(len(inst["events"]), 1)
        r = explanation.get(vid) or {}
        if share >= PRUNE_SPURIOUS:
            cands.append({"kind": "prune", "id": vid, "why": f"{100 * share:.0f}% of its events fall where the stem is silent"})
            pruned.add(vid)
        elif share >= DROP_SPURIOUS:
            cands.append({"kind": "drop", "id": vid, "events": sil, "why": f"{len(sil)} of its {len(inst['events'])} events fall where the stem is silent"})
        if vid not in pruned and inst.get("pitched"):
            off = offpitch_events(inst, y, result)
            if len(off) >= OFFPITCH_MIN:
                cands.append({"kind": "drop", "id": vid, "events": off,
                              "why": f"{len(off)} of its {len(inst['events'])} notes have no salience at their read pitch over the note"})
        if vid not in pruned and "explained" in r and r["explained"] < PRUNE_EXPLAINED and r["scored"] >= 8:
            cands.append({"kind": "prune", "id": vid, "why": f"explains {100 * r['explained']:.0f}% of the stem in its own windows"})
            pruned.add(vid)
        # level: a step in each direction (the closed loop keeps the one that helps and verify()
        # keeps stepping the same way while it does). The stem's level in the voice's own band is
        # only a hint: another voice on the same harmonics (a melody an octave over a pad) makes
        # the stem read louder than the voice is, so the hint's direction is tried first, not alone
        if GAIN_STEPS and vid not in pruned and r.get("scored", 0) >= 8:
            hint = float(r.get("level_fit_db") or 0.0)
            first = GAIN_STEP_DB if hint >= 0 else -GAIN_STEP_DB
            for g in (first, -first):
                cands.append({"kind": "gain", "id": vid, "db": float(g),
                              "why": f"level {'+' if g > 0 else '-'}{abs(g):.0f} dB (the stem reads {hint:+.0f} dB against this voice in its own band)"})
    by_stem = {}
    for vid, inst in insts.items():
        if vid not in pruned:
            by_stem.setdefault(inst["stem"], []).append(vid)
    from lib.dj import voices as V
    for stem, vids in by_stem.items():
        if stem not in stems_mono:
            continue
        specs = {v: _exemplar_spectrum(insts[v], stems_mono) for v in vids}
        times = {v: V.event_times(insts[v], result) for v in vids}
        taken = set()
        for a in vids:
            for b in vids:
                if b <= a or a in taken or b in taken or specs[a] is None or specs[b] is None:
                    continue
                if insts[a]["kind"] != insts[b]["kind"] or bool(insts[a].get("pitched")) != bool(insts[b].get("pitched")):
                    continue
                cos = float(specs[a] @ specs[b])
                if cos < MERGE_COS:
                    continue
                ta, tb = times[a], times[b]
                if not len(ta) or not len(tb):
                    continue
                k = np.searchsorted(tb, ta)
                near = np.minimum(np.abs(tb[np.clip(k, 0, len(tb) - 1)] - ta), np.abs(tb[np.clip(k - 1, 0, len(tb) - 1)] - ta))
                coincide = float(np.mean(near < 0.02))
                if coincide > MERGE_COINCIDE:
                    continue
                keep, absorb = (a, b) if insts[a]["n"] >= insts[b]["n"] else (b, a)
                verdict = ""
                if prog is not None and keep in prog.get("voices", {}) and absorb in prog.get("voices", {}):
                    own = explanation.get(absorb) or {}
                    try:
                        swapped = explain_voice(_with_sound_of(prog, absorb, keep), stems_mono, absorb)
                    except Exception:  # noqa: BLE001
                        swapped = None
                    if swapped is None or "explained" not in own:
                        continue
                    ok = swapped["explained"] >= MERGE_KEEP_SHARE * own["explained"] and swapped["overshoot"] <= own["overshoot"] + MERGE_OVERSHOOT_ADD
                    verdict = f"; {keep}'s sound at its hits explains {100 * swapped['explained']:.0f}% vs its own {100 * own['explained']:.0f}%"
                    if not ok:
                        continue
                taken.add(absorb)
                cands.append({"kind": "merge", "id": absorb, "keep": keep,
                              "why": f"the same sound as {keep} (spectra {cos:.2f} alike, events never coincide{verdict})"})
    return cands


def apply(result, cands):
    """A copy of the reading with the candidates applied: dropped events,
    pruned voices, merged voices (events united, the keeper's sound).
    What went is listed in result["pruned"] = [{id, name, stem, n, why}]."""
    import copy
    res = copy.deepcopy(result)
    insts = {i["id"]: i for i in INS.instruments(res)}
    gone = list(res.get("pruned") or [])
    for c in cands:
        vid = c["id"]
        if vid not in insts or insts[vid].get("_drop"):
            continue
        inst = insts[vid]
        if c["kind"] == "gain":
            inst["gain_db"] = round(float(inst.get("gain_db") or 0.0) + float(c["db"]), 1)
            gone.append({"id": vid, "name": INS.display_name(inst), "stem": inst["stem"], "n": 0, "kind": "gain", "db": float(c["db"]),
                         "why": c["why"]})
        elif c["kind"] == "drop":
            drop = set(c["events"])
            inst["events"] = [e for k, e in enumerate(inst["events"]) if k not in drop]
            inst["n"] = len(inst["events"])
            inst["label"], inst["detail"] = INS.describe(inst)
            inst["dropped_events"] = int(inst.get("dropped_events", 0)) + len(drop)
            gone.append({"id": vid, "name": INS.display_name(inst), "stem": inst["stem"], "n": len(drop), "kind": "events", "why": c["why"]})
        elif c["kind"] == "merge" and c["keep"] in insts and not insts[c["keep"]].get("_drop"):
            k = insts[c["keep"]]
            k["events"] = sorted(k["events"] + inst["events"], key=lambda e: (e[0], e[1], e[2] or 0))
            k["n"] = len(k["events"])
            ms = [e[2] for e in k["events"] if e[2] is not None]
            if ms:
                k["range"] = [int(min(ms)), int(max(ms))]
            k["label"], k["detail"] = INS.describe(k)
            k["merged"] = (k.get("merged") or []) + [vid]
            inst["_drop"] = True
            gone.append({"id": vid, "name": INS.display_name(inst), "stem": inst["stem"], "n": inst["n"], "kind": "merged", "why": c["why"]})
        elif c["kind"] == "prune":
            inst["_drop"] = True
            gone.append({"id": vid, "name": INS.display_name(inst), "stem": inst["stem"], "n": inst["n"], "kind": "voice", "why": c["why"]})
    for stem, rec in res["stems"].items():
        rec["instruments"] = [i for i in rec.get("instruments", []) if not i.get("_drop")]
    res["pruned"] = gone
    return res


def annotate(result, explanation):
    """Write each voice's figures into its instrument record (explained,
    overshoot, spurious, share_db) - what the planner's gutter shows."""
    for inst in INS.instruments(result):
        r = explanation.get(inst["id"])
        if r and "explained" in r:
            inst["explained"] = round(float(r["explained"]), 3)
            inst["overshoot"] = round(float(r["overshoot"]), 3)
            inst["spurious"] = round(float(r["spurious"]), 3)
            inst["share_db"] = round(float(r["share_db"]), 1)
    return result


def verify(one, y, name, prog, cands, progress=None):
    """Greedy closed loop on ONE stem: apply each candidate in turn to the
    reading, rebuild, render the stem, keep the candidate only when the
    spectral gap to the recording does not widen. -> (reading, kept, gap0, gap1)"""
    from lib.dj import fidelity as F, songprogram as SP
    beats = np.asarray(one["beats"], dtype=np.float64)
    period = float(one["period_s"])
    t0, t1 = F.window(one, {name: y})
    ref = y[int(t0 * RATE):int(t1 * RATE)]
    Mr = F.mel_db(ref)
    sel = (beats >= t0) & (beats < t1 - period)
    bts = beats[sel] - t0

    def gap_of(p):
        rec = SP.render(p, {name: y}, t0=t0, t1=t1, room=False, use_residual=False, limit=None, fast=True)[:, 0][: len(ref)]
        if len(rec) < len(ref):
            rec = np.concatenate([rec, np.zeros(len(ref) - len(rec), dtype=np.float32)])
        return F.spectral_gap(ref, rec, bts, period, Mr=Mr)[0]
    base = gap_of(prog)
    gap0 = base
    kept = []
    cur = one
    cur_prog = prog
    queue = list(cands)
    moved = {}                                             # voice -> total gain moved so far
    while queue:
        c = queue.pop(0)
        if c["kind"] == "gain":
            if c["id"] not in cur_prog.get("voices", {}) or abs(moved.get(c["id"], 0.0) + c["db"]) > GAIN_MAX_DB:
                continue
            # the opposite direction is pointless once this one helped
            if moved.get(c["id"], 0.0) * c["db"] < 0:
                continue
        trial = apply(cur, [c])
        try:
            if c["kind"] == "gain":
                # a level change needs no rebuild: the same program with the voice's gain moved
                p2 = dict(cur_prog, voices=dict(cur_prog["voices"]))
                p2["voices"][c["id"]] = dict(p2["voices"][c["id"]], gain_db=float(p2["voices"][c["id"]].get("gain_db") or 0.0) + float(c["db"]))
            else:
                p2 = SP.build(trial, {name: y}, verbatim_stems=(), chords=False, residual=False)
            g = gap_of(p2)
        except Exception as e:  # noqa: BLE001
            if progress:
                progress(f"{c['id']}: check failed ({type(e).__name__})")
            continue
        ok = np.isfinite(g) and (g <= base + VERIFY_TOL_DB if c["kind"] != "gain" else g < base - VERIFY_TOL_DB)
        if progress:
            progress(f"{c['id']} {c['kind']}: stem gap {base:.1f} -> {g:.1f} dB, {'kept' if ok else 'rejected'} ({c['why'][:70]})")
        if ok:
            c = dict(c, gap_before=round(float(base), 2), gap_after=round(float(g), 2))
            kept.append(c)
            cur, base, cur_prog = trial, g, p2
            if c["kind"] == "gain":
                moved[c["id"]] = moved.get(c["id"], 0.0) + c["db"]
                queue.insert(0, dict(c, why=c["why"]))    # keep stepping the same way while it helps
            cur["pruned"] = cur["pruned"][:-1] + [dict(cur["pruned"][-1], gap=f"{c['gap_before']:.1f} -> {c['gap_after']:.1f} dB")]
    return cur, kept, gap0, base


BLEED_DB = -40.0               # the other stems louder than this inside an exemplar cut: the separation bled into it
BLEED_GAIN_DB = 15.0           # ...and a hit where they are at least this much quieter replaces it


def clean_exemplars(result, stems_mono, progress=None):
    """Exemplar cuts are taken from one stem, but the separation bleeds the
    loud moments of the OTHER stems into it: Sussudio's kick, hat and perc
    were cut under −20 dBFS of singing and the reconstruction "sang" on
    every hit. For every instrument with a cut, when the other stems are
    loud inside it, the loudest own hit with the other stems quiet (and no
    other event of its stem within 80 ms) becomes the exemplar instead.
    Not a closed-loop candidate: the bled stem is the reference the loop
    measures against, so it cannot see the bleed."""
    beats = np.asarray(result["beats"], dtype=np.float64)
    changed = 0
    for inst in INS.instruments(result):
        ex = inst.get("exemplar")
        stem = inst["stem"]
        if not ex or stem not in stems_mono:
            continue
        others = [n for n in stems_mono if n != stem]
        if not others:
            continue

        def other_level(a, b):
            segs = [stems_mono[n][int(a * RATE):int(b * RATE)] for n in others]
            return max((20 * np.log10(np.sqrt(np.mean(s.astype(np.float64) ** 2)) + 1e-9) for s in segs if len(s)), default=-120.0)
        cur = other_level(ex[0], ex[1])
        if cur < BLEED_DB:
            continue
        span = max(0.12, float(ex[1]) - float(ex[0]))
        # every event of the stem, to keep the new cut isolated
        all_t = np.sort(np.concatenate([V_event_times(i, beats) for i in INS.instruments(result) if i["stem"] == stem]))
        best = None
        for e in sorted(inst["events"], key=lambda e: -e[4])[:40]:
            if e[5] < 0.99:
                continue
            t = INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
            k = np.searchsorted(all_t, t)
            near = [all_t[j] for j in (k - 1, k, k + 1) if 0 <= j < len(all_t) and abs(all_t[j] - t) > 1e-4]
            if any(abs(x - t) < 0.08 for x in near):
                continue
            lv = other_level(t, t + span)
            if best is None or lv < best[0]:
                best = (lv, t, e)
        if best is None or best[0] > cur - BLEED_GAIN_DB:
            continue
        inst["exemplar"] = [round(float(best[1]), 4), round(float(best[1] + span), 4)]
        if inst.get("pitched") and best[2][2] is not None:
            inst["exemplar_midi"] = int(best[2][2])
        inst["exemplar_moved"] = f"other stems {cur:.0f} -> {best[0]:.0f} dBFS in the cut"
        changed += 1
        if progress:
            progress(f"{inst['id']}: exemplar moved to {best[1]:.2f} s ({inst['exemplar_moved']})")
    return changed


def V_event_times(inst, beats):
    return np.array([INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0) for e in inst["events"]], dtype=np.float64)


PRUNE = True                   # explain_pass drops/merges on the evidence, each step verified on the stem


def explain_pass(result, stems, progress=None, prune=PRUNE):
    """The reading's own check, one stem at a time (the program build
    only needs the stem the voice lives in): measure every instrument,
    annotate it, and - with prune - drop the events and voices that are
    not there and join the voices that are one sound, each step kept only
    if the rendered stem gets closer to the recording. stems: {name:
    array | loader} as identify() takes them. Returns the (possibly
    reduced) result; result["pruned"] says what went and why,
    result["stems"][name]["gap_db"] the stem's spectral gap after."""
    from lib.dj import songprogram as SP
    # first the exemplar cuts, against the other stems (one pass with every stem decoded, mono)
    try:
        allm = {n: INS._mono(s() if callable(s) else s) for n, s in stems.items() if n in result.get("stems", {})}
        n_moved = clean_exemplars(result, allm, progress=progress)
        if n_moved and progress:
            progress(f"{n_moved} exemplar cuts moved away from the other stems' loud moments")
        del allm
    except Exception as e:  # noqa: BLE001
        result.setdefault("reasons", []).append(f"exemplar check failed ({type(e).__name__}: {str(e)[:80]})")
    for name, rec in list(result.get("stems", {}).items()):
        if not rec.get("instruments") or name not in stems:
            continue
        y = INS._mono(stems[name]() if callable(stems[name]) else stems[name])
        if name == "bass" and "drums" in stems and rec.get("kick_bleed_gain"):
            d = INS._mono(stems["drums"]() if callable(stems["drums"]) else stems["drums"])
            y, _g = INS.clean_bass(y, d)
            del d
        one = dict(result, stems={name: rec}, pruned=[])
        mono = {name: y}
        if progress:
            progress(f"{name}: checking each instrument against the stem")
        try:
            prog = SP.build(one, mono, verbatim_stems=(), chords=False, residual=False)
            ex = explain_all(prog, mono)
            cands = suggest(one, mono, ex, prog=prog) if prune else []
        except Exception as e:  # noqa: BLE001
            result.setdefault("reasons", []).append(f"{name}: instrument check failed ({type(e).__name__}: {str(e)[:80]})")
            continue
        annotate(one, ex)
        try:
            if cands:
                one, kept, gap0, gap1 = verify(one, y, name, prog, cands, progress=progress)
            else:
                from lib.dj import fidelity as F
                one, kept, gap0, gap1 = verify(one, y, name, prog, [], progress=None)
        except Exception as e:  # noqa: BLE001
            result.setdefault("reasons", []).append(f"{name}: instrument check could not be verified ({type(e).__name__}: {str(e)[:80]})")
            kept, gap0, gap1 = [], float("nan"), float("nan")
        if any(c["kind"] == "merge" for c in kept):
            # the joined voices, measured again with their united events
            try:
                prog2 = SP.build(one, mono, verbatim_stems=(), chords=False, residual=False)
                keeps = {c["keep"] for c in kept if c["kind"] == "merge"}
                annotate(one, {k: v for k, v in explain_all(prog2, mono).items() if k in keeps})
            except Exception:  # noqa: BLE001
                pass
        one["stems"][name]["gap_db"] = None if not np.isfinite(gap1) else round(float(gap1), 2)
        one["stems"][name]["gap_db_before"] = None if not np.isfinite(gap0) else round(float(gap0), 2)
        result["pruned"] = (result.get("pruned") or []) + one["pruned"]
        result["stems"][name] = one["stems"][name]
        if progress:
            kept_s = one["stems"][name]["instruments"]
            progress(f"{name}: gap {gap0:.1f} -> {gap1:.1f} dB; " + ", ".join(f"{i['id']} explains {100 * i.get('explained', 0):.0f}%" for i in kept_s))
        del y
    return result
