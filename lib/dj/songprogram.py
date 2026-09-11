"""The SongProgram: a song as a REDUCED PROGRAM - voices, a small pattern
vocabulary per voice, a groove template, a chord track, a per-bar
sequence with ops, verbatim slices - built from the instrument reading
and rendered deterministically. docs/RECONSTRUCTION_PLAN.md.

Why patterns are inferred here and not compressed afterwards: the music
repeats, the reading does not (a spurious hit here, a missed one there).
Bars are clustered by similarity, each cluster's CONSENSUS is the
pattern, and every bar's deviation from its pattern is classified: a
loud extra hit is a variation (kept as an `add` op), a quiet one is
noise (dropped); a missing hit that the pattern nearly always has is a
missed detection (played), one it has only usually is a real gap (a
`drop` op). Bars unlike every pattern are one-off patterns (fills).

Program (in memory; save() writes JSON + the samples as wavs):
    grid       {t0, period_s, down0, n_beats}  the DJ grid, uniform
    voices     {id: {stem, kind, pitched, name, hint, level_dbfs, model}}
               model: "kit" (layers with tails), "pitched" (per-pitch
               samples), "chord" (the exemplar played as a unit),
               "additive" (harmonics from the voice's own profile,
               lib/dj/additive.py - chosen when it explains the stem
               better than the samples over the voice's notes),
               "verbatim" (a stem); every pitched voice also carries its
               additive profile (`profile`), the morphable form
    patterns   {voice: {pid: {"items": [[step16, rel_pitch|None, dur_steps, vel, offset_s], ...], "share": [..]}}}
               offset_s = the item's timing against its grid step, the pattern's own feel
    groove     {voice: {step16: [offset_s, vel]}}   (the voice's mean feel per step; used for `add` ops)
    units      {voice: 1|2|4}  bars per pattern (the unit that compresses the voice best)
    sequence   {voice: [{"bar": b, "pat": pid, "root": midi|None, "ops": [
                   ["add", step, pitch, dur, vel, offset] | ["drop", step, pitch] | ["nudge", step, pitch, offset]]}]}
    rooms      {stem: {rt60_s, wet_db}}
    verbatim   [{"stem": name, "bar0": b, "bars": n}]   (a stem played as audio over those bars)
    stats      events explained, patterns, sequence entries, ops
"""
import json
import os

import numpy as np

from lib.dj import instruments as INS
from lib.dj import voices as V

RATE = INS.RATE
STEPS = INS.STEPS
BAR = 16
PROGRAM_VERSION = 1
SIM_DRUMS = 0.75               # Jaccard for two drum bars to be one pattern
SIM_PITCHED = 0.7
PROTO_SHARE = 0.6              # an item is in the pattern when this share of the cluster's bars have it
SURE_SHARE = 0.95              # ...and when it is this reliable, a bar missing it is a missed detection, not a gap
LOUD_ADD = 0.5                 # an extra hit must be this loud to count as a variation (else reading noise)
NUDGE_S = 0.03                 # a hit this far from its pattern's timing keeps its own (a `nudge` op)
MAX_SAMPLE_S = 2.5
RELEASE_S = 0.12
LEN_STEPS = 2                  # an occurrence whose length differs from its pattern item by more than this gets a "len" op
RESIDUAL_DB = 14.0             # a bar whose voices leave this much (mean |dB| on a log-mel) unexplained is carried verbatim
RESIDUAL_STEMS = ("drums", "bass", "other")   # THE HYBRID PROGRAM (2026-09-10, after the discord measurements): every
                               # note stem is notes where its voices explain the recording bar by bar and the recording's
                               # own audio where they do not; the voices are silent inside those bars (they doubled the
                               # recording before). What plays as notes is reported per stem (stats["note_share"]) - the
                               # honest figure for what the program IS. Drums and bass were added to `other` here: the
                               # user's three tracks measured 13-26 dB off on those stems too
RESIDUAL_DB_STEM = {"drums": 14.0, "bass": 14.0, "other": 14.0}   # per-stem thresholds (RESIDUAL_DB when absent)
RESIDUAL_CHROMA_MIN = 0.6      # a pitched stem's bar also needs the render's chroma to follow the recording's, beat by
                               # beat, at least this well (the user 2026-09-10 on the spectral test alone: "an angry child
                               # slamming piano keys" - wrong piano notes on the right piano sample pass a log-mel mean)
RESIDUAL_ONSET_RATIO = 2.0     # ...and the render may not strike more than this many times the recording's onsets in the
RESIDUAL_ONSET_SLACK = 2       # bar (plus this many): a sustained texture read as a hail of piano hits fails here
RESIDUAL_MIN_RUN = 2           # a run of note bars shorter than this between recording bars is the recording too: the
                               # recording and a different-sounding render must not trade places every bar
FAMILY_SAMPLES = False         # kit samples cut from the drumsep family stems (lib/dj/drumsep.py) when the reading has
                               # families; off = the mixed stem's cut, isolated against every family's onsets (see build).
                               # OFF: the render gate 2026-09-09 (true hits through the kit vs the true kit) - family cuts
                               # 8.2 dB median, mixed-stem cuts 7.0 (house 15.6 vs 9.6, pop 7.7 vs 5.9): a separated family
                               # stem is a softer, smeared version of the hit; the mixed stem's isolated hit is the recording
HOLD_ADDITIVE = True           # a pitched-sample voice plays a note far longer than its recording through its additive profile
HOLD_ADDITIVE_S = 0.6          # ...when the note is at least this long...
HOLD_ADDITIVE_X = 1.5          # ...and this many times the recording's length
HOLD_MIN_DECAY_DB_S = -6.0     # ...with its per-harmonic decay clamped to a sustain (a struck voice's profile dies in a second)
TAIL_ADDITIVE = True           # a struck note longer than its recording continues through the additive profile from the
                               # recording's end level (see _extend_with_profile); off = the recording's last 40 ms looped.
                               # Render gate 2026-09-09: neutral (every part within 0.4 dB); kept as the more honest tail
CALIBRATE_ADDITIVE = True      # the additive profile's level tied to the voice's exemplar recording (see build)
CALIBRATE_WIN_S = 0.25         # ...over the attack window the profile is fitted on


# --------------------------------------------------------------------------
# bars and patterns
# --------------------------------------------------------------------------

def _bars_of(inst, result):
    d0 = int(result.get("down0", 0))
    bars = {}
    for ev in inst["events"]:
        bar = (ev[0] - d0) // 4
        if bar < 0:
            continue
        step = ((ev[0] - d0) % 4) * STEPS + ev[1]
        bars.setdefault(bar, []).append((int(step), ev[2], int(ev[3]), float(ev[4]), float(ev[6]) if len(ev) > 6 else 0.0))
    return bars


def _key(items, pitched, root):
    return frozenset((s, (m - root) if (pitched and m is not None and root is not None) else None) for s, m, _d, _v, _o in items)


def _jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def infer_patterns(bars, pitched, sim):
    """bars {bar: [(step, midi, dur, vel, off)]} -> (patterns {pid: {items, share}},
    sequence [{bar, pat, root, ops}], groove {step: [off, vel]})."""
    roots = {}
    keys = {}
    for b, items in bars.items():
        ms = [m for _s, m, _d, _v, _o in items if m is not None]
        roots[b] = min(ms) if (pitched and ms) else None
        keys[b] = _key(items, pitched, roots[b])
    # greedy clustering: the bar that covers the most others at >= sim becomes a prototype
    remaining = sorted(bars)
    clusters = []
    while remaining:
        best, best_c = None, []
        for i in remaining:
            c = [j for j in remaining if _jaccard(keys[i], keys[j]) >= sim]
            if len(c) > len(best_c):
                best, best_c = i, c
        clusters.append(best_c)
        remaining = [j for j in remaining if j not in set(best_c)]
    patterns, sequence = {}, []
    groove_acc = {}
    for ci, members in enumerate(clusters):
        pid = f"P{ci + 1}"
        counts, vel_by, dur_by, off_by = {}, {}, {}, {}
        for b in members:
            for s, m, d, v, o in bars[b]:
                k = (s, (m - roots[b]) if (pitched and m is not None and roots[b] is not None) else None)
                counts[k] = counts.get(k, 0) + 1
                vel_by.setdefault(k, []).append(v)
                dur_by.setdefault(k, []).append(d)
                off_by.setdefault(k, []).append(o)
        n = len(members)
        proto = {k: c / n for k, c in counts.items() if c / n >= PROTO_SHARE}
        if not proto:                                    # a lone odd bar: itself is the pattern
            proto = {k: 1.0 for k in counts}
        items = sorted((k[0], k[1], int(np.median(dur_by[k])), round(float(np.median(vel_by[k])), 2),
                        round(float(np.median(off_by[k])), 4)) for k in proto)
        patterns[pid] = {"items": [list(it) for it in items], "share": [round(proto[(it[0], it[1])], 2) for it in items]}
        proto_keys = set(proto)
        proto_off = {(it[0], it[1]): it[4] for it in items}
        proto_dur = {(it[0], it[1]): it[2] for it in items}
        for b in members:
            bar_keys = {}
            for s, m, d, v, o in bars[b]:
                k = (s, (m - roots[b]) if (pitched and m is not None and roots[b] is not None) else None)
                bar_keys[k] = (d, v, o)
                groove_acc.setdefault(s, []).append((o, v))
            ops = []
            for k, (d, v, o) in bar_keys.items():
                if k not in proto_keys:
                    if v >= LOUD_ADD:
                        ops.append(["add", k[0], k[1], int(d), round(float(v), 2), round(float(o), 4)])
                    continue
                if abs(o - proto_off[k]) > NUDGE_S:
                    ops.append(["nudge", k[0], k[1], round(float(o), 4)])
                if abs(int(d) - proto_dur[k]) > LEN_STEPS:
                    ops.append(["len", k[0], k[1], int(d)])          # this occurrence holds longer / shorter than the pattern
            for k in proto_keys:
                if k not in bar_keys and proto[k] < SURE_SHARE:
                    ops.append(["drop", k[0], k[1]])
            sequence.append({"bar": int(b), "pat": pid, "root": (int(roots[b]) if roots[b] is not None else None), "ops": ops})
    sequence.sort(key=lambda e: e["bar"])
    groove = {int(s) % BAR: [round(float(np.median([o for o, _v in lst])), 4), round(float(np.mean([v for _o, v in lst])), 3)]
              for s, lst in groove_acc.items()}
    return patterns, sequence, groove


def _unit_bars(bars, unit):
    """bars -> units of `unit` bars: steps continue past 16 into the next bar."""
    if unit == 1:
        return bars
    out = {}
    for b, items in bars.items():
        u = (b // unit) * unit
        k = b - u
        out.setdefault(u, []).extend((s + BAR * k, m, d, v, o) for s, m, d, v, o in items)
    return out


CHOOSER_NOTES = 30             # _additive_better judges on this many notes...
CHOOSER_BY = "isolated"        # ..."isolated": the longest clear run after the onset (any sound) | "loud": the loudest (the
                               # old rule - the loudest notes are the chord moments and coincidences)
CHOOSER_LEVEL_FREE = True      # ...with one gain fitted per model (a right timbre at the wrong level lost: the render gate
                               # 2026-09-09 found the chooser wrong on the rock and pop bass and the dnb sub)
CHOOSER_HARMONIC = False       # ...over the mel bands at the note's own harmonics only (the stem's bleed is not the voice).
                               # OFF: measured on the render gate 2026-09-09 - it flipped the dnb sub back to samples (18.8 vs
                               # additive 9.7 dB against the true part) and left the rock bass on samples (21.5 vs 8.2); on
                               # the demucs stem the samples reproduce the bleed and the chooser cannot see past it


def _additive_better(inst, y, result, snd, model, period, n=None):
    """Render the voice's n best-isolated notes both ways over their own
    span and compare each with the stem there (energy-weighted mel |dB|,
    phase-free, one gain per model): True when the additive render explains
    the stem better than the sample-based one."""
    import librosa
    from lib.dj import additive as AD, resynth as RS
    n = n or CHOOSER_NOTES
    beats = np.asarray(result["beats"], dtype=np.float64)
    evs_all = [e for e in inst["events"] if e[2] is not None]
    src = V.sample_source_times(snd.get("pitches"))
    if len(src):
        # not the events the samples were cut from: there the sample render is the stem itself
        def is_source(e):
            t = INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
            k = int(np.searchsorted(src, t))
            return min(abs(src[min(k, len(src) - 1)] - t), abs(src[max(k - 1, 0)] - t)) < 0.05
        rest = [e for e in evs_all if not is_source(e)]
        if len(rest) >= 3:
            evs_all = rest
    if CHOOSER_BY == "isolated" and len(evs_all) > n:
        all_on = V.stem_onsets(result, inst["stem"])
        def clear(e):
            t = INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
            k = int(np.searchsorted(all_on, t + 0.005, side="right"))
            return (float(all_on[k]) - t) if k < len(all_on) else 2.0
        events = sorted(evs_all, key=lambda e: -clear(e))[:n]
    else:
        events = sorted(evs_all, key=lambda e: -e[4])[:n]
    if len(events) < 3:
        return False
    prof = snd["profile"]

    def sample_render(e, dur):
        want = int(dur * RATE)
        if model == "chord" and snd.get("chord") is not None:
            base = inst.get("exemplar_midi")
            key = (int(e[2]) - int(base)) if base is not None else 0
            return RS.looped(RS.repitch(snd["chord"], key), want) * np.float32(10 ** (_gain_db(e[4], "sustain") / 20.0))
        pitches = snd.get("pitches") or {}
        if not pitches:
            return None
        avail = np.array(sorted(pitches))
        m = int(e[2])
        near = int(avail[np.argmin(np.abs(avail - m))])
        rec, gain = V.pick_layer(pitches[near], _gain_db(e[4], "hit"))
        s = V.sustain_sample(rec, int(want * 2 ** ((m - near) / 12.0)) + 16)
        s = RS.repitch(s, m - near) if near != m else s
        return s[:want] * np.float32(10 ** (gain / 20.0))

    mel_f = librosa.mel_frequencies(n_mels=48, fmin=30, fmax=12000)

    def gap(ref, rec, midi=None):
        n_ = min(len(ref), len(rec))
        if n_ < 2048:
            return None
        M = lambda x: 10 * np.log10(librosa.feature.melspectrogram(y=np.ascontiguousarray(x[:n_], dtype=np.float32), sr=RATE, n_fft=2048,
                                                                  hop_length=512, n_mels=48, fmin=30, fmax=12000, power=2.0) + 1e-10)
        A, B = M(ref), M(rec)
        floor = max(float(A.max()), float(B.max())) - 40.0
        cells = (A > floor) | (B > floor)
        if CHOOSER_HARMONIC and midi is not None:
            # only the bands where THIS note lives: the stem's other content (a guitar's low end bled into the
            # bass stem) is not the voice, and a sample that reproduces it must not win for that
            f0 = 440.0 * 2 ** ((int(midi) - 69) / 12.0)
            near = np.zeros(len(mel_f), dtype=bool)
            for h in range(1, 13):
                near |= np.abs(mel_f - h * f0) <= 0.06 * h * f0 + 0.5 * np.gradient(mel_f)
            cells &= near[:, None]
        if not cells.any():
            return None
        d = A[cells] - B[cells]
        if CHOOSER_LEVEL_FREE:
            d = d - float(np.median(d))
        return float(np.mean(np.abs(d)))
    g_s, g_a = [], []
    for e in events:
        t = INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
        dur = min(1.5, max(0.12, e[3] * period / STEPS))
        a, b = int(t * RATE), int(min(len(y), (t + dur) * RATE))
        if b - a < 2048:
            continue
        ref = y[a:b]
        rs = sample_render(e, dur)
        if rs is not None and model != "chord":
            rs = rs[int(V.PRE_S * RATE):]              # the sample carries a 10 ms pre-roll the renderer places early
        ra = AD.render_note(prof, e[2], dur, e[4])
        gs, ga = (gap(ref, rs, e[2]) if rs is not None else None), gap(ref, ra, e[2])
        if gs is not None and ga is not None:
            g_s.append(gs); g_a.append(ga)
    if len(g_a) < 3:
        return False
    snd["chooser"] = {"samples_db": round(float(np.median(g_s)), 2), "additive_db": round(float(np.median(g_a)), 2), "n": len(g_a)}
    return float(np.median(g_a)) < float(np.median(g_s)) - 1.0     # the song's own recording wins ties


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(result, stems, sections=None, verbatim_stems=("vocals",), progress=None, chords=True, residual=True, room=False,
          fit_levels=False, exemplar_kit=True, clean_mixtures=False, explain=False):
    """chords / residual: the chord track (~20 s of chroma) and the
    unexplained-residual test (a full render of `other`) can be skipped
    for an interactive build (the planner's play-selected-only).
    room: the SYNTHETIC drum room (a decaying-noise reverb tuned to the
    measured decay) - off by default: with the kit subtracted, the tails
    are the recording's own.
    explain: measure every voice against the stem by synthesis
    (lib/dj/explain.py), drop the voices that are not there and join the
    ones that are one sound, rebuild; the measurements stay in
    prog["explain"] (per voice: explained, overshoot, spurious, share_db)
    with prog["pruned"] / prog["merged"] / prog["why"]."""
    beats = np.asarray(result["beats"], dtype=np.float64)
    period = float(result.get("period_s") or np.median(np.diff(beats)))
    prog = {"version": PROGRAM_VERSION, "reading_version": int(result.get("version") or 0),
            "grid": {"t0": float(beats[0]), "period_s": period, "down0": int(result.get("down0", 0)), "n_beats": int(len(beats))},
            "bpm": round(60.0 / period, 3), "sections": sections or [], "voices": {}, "patterns": {}, "groove": {}, "sequence": {},
            "rooms": {}, "verbatim": [], "_sounds": {}, "stats": {}}
    n_bars = max(1, (len(beats) - prog["grid"]["down0"]) // 4)
    insts = INS.instruments(result)
    stems_mono = {n: V._mono(a) for n, a in stems.items()}
    onsets_by_stem = {s: V.stem_onsets(result, s) for s in INS.STEM_ORDER}
    n_events = n_pat = n_seq = n_ops = 0
    # DRUM LEVELS FIT TO THE STEM: every drum event's gain is solved on the stem's own 60 ms window
    # (all coincident sounds jointly, NNLS on the waveform) against the sample as extracted at its
    # recording level; the reading's cluster-derived velocities were up to 12 dB off (measured on the
    # synthetic gate) and are replaced here. The sample IS the nominal level (gain 1).
    fitted = {}
    kit_insts = [i for i in insts if i["stem"] == "drums" and "drums" in stems_mono] if fit_levels else []
    if kit_insts:
        samples = {}
        for inst in kit_insts:
            snd0 = V.kit_sound(inst, stems_mono["drums"], result, onsets_by_stem["drums"], scale=False)
            if snd0 and snd0["layers"]:
                loud = max(snd0["layers"], key=lambda l: l["db"])
                samples[inst["id"]] = {"times": V.event_times(inst, result), "sample": loud["audio"]}
                fitted[inst["id"]] = snd0
        if samples:
            gains = V.fit_levels(stems_mono["drums"], samples)
            new_insts = []
            for inst in insts:
                if inst["id"] in gains:
                    g = gains[inst["id"]]
                    evs = []
                    for k, e in enumerate(inst["events"]):
                        gk = g.get(k)
                        e = list(e)
                        if gk is not None and gk > 0:
                            e[4] = float(np.clip(1.0 + 20.0 * np.log10(gk) / 24.0, 0.05, 1.7))
                        evs.append(e)
                    inst = dict(inst, events=evs)
                new_insts.append(inst)
            insts = new_insts
            if progress:
                progress(f"drums: levels of {sum(len(g) for g in gains.values())} events fitted to the stem")
    prog["phrases"] = {}
    for stem in verbatim_stems:
        if stem not in stems_mono:
            continue
        if stem != "vocals" and not any(i["stem"] == stem for i in insts):
            continue
        if stem == "vocals":
            # the phrases come from the stem audio, whether or not its notes were read (READ_VOCALS)
            if float(np.abs(stems_mono[stem]).max()) < 1e-3:
                continue
            # phrases with reuse: a chorus recorded once, placed wherever it recurs
            lib, placements = V.phrase_library(stems_mono[stem], beats, period)
            if lib:
                prog["phrases"][stem] = {"library": [[float(a), float(b)] for a, b in lib], "placements": placements}
                if progress:
                    progress(f"{stem}: {len(placements)} phrases, {len(lib)} distinct ({len(placements) - len(lib)} reused)")
                continue
        prog["verbatim"].append({"stem": stem, "bar0": 0, "bars": int(n_bars)})
    # kit samples from the drumsep FAMILY stems when the reading was made per family: the exemplar cut of a
    # kick then holds the kick alone (the mixed stem's cut carried the hat and whatever rang under it)
    family_audio = {}
    if FAMILY_SAMPLES and "drums" in stems_mono and any(i.get("family") for i in insts if i["stem"] == "drums"):
        try:
            from lib.dj import drumsep as DS
            family_audio = DS.separate(stems_mono["drums"], progress=progress)
            if progress:
                progress("drums: kit samples cut from the separated families")
        except Exception as e:  # noqa: BLE001
            if progress:
                progress(f"drums: family stems unavailable ({type(e).__name__}: {str(e)[:60]}); samples from the mixed stem")
    for inst in insts:
        stem = inst["stem"]
        if stem in verbatim_stems:
            continue
        vid = inst["id"]
        pitched = bool(inst.get("pitched"))
        model = "kit" if stem == "drums" else ("chord" if inst["kind"] == "sustain" else ("pitched" if pitched else "kit"))
        src_stems = stems_mono
        if stem == "drums" and inst.get("family") in family_audio:
            src_stems = dict(stems_mono, drums=family_audio[inst["family"]])
        prog["voices"][vid] = {"stem": stem, "kind": inst["kind"], "pitched": pitched, "name": INS.display_name(inst),
                               "hint": inst.get("hint"), "level_dbfs": inst.get("level_dbfs"), "model": model,
                               "exemplar": inst.get("exemplar"), "exemplar_midi": inst.get("exemplar_midi"), "range": inst.get("range"),
                               "gain_db": float(inst.get("gain_db") or 0.0)}
        # sound model (kit voices get a second pass below, with the other sounds subtracted)
        if model == "kit":
            if exemplar_kit:
                from lib.dj import resynth as RS
                snd = {"layers": [{"db": 0.0, "audio": RS.exemplar_audio(inst, src_stems)}], "decay_db_s": None}
                prog.setdefault("_kit_pending", []).append(vid)
            elif vid in fitted:
                # the fitted sample at its recording level; one layer (velocity is the fitted gain)
                loud = max(fitted[vid]["layers"], key=lambda l: l["db"])
                snd = {"layers": [{"db": 0.0, "audio": loud["audio"]}], "decay_db_s": fitted[vid].get("decay_db_s")}
            else:
                snd = V.kit_sound(inst, stems_mono[stem], result, onsets_by_stem[stem])
            prog["_sounds"][vid] = snd
        elif model == "pitched":
            snd = {"pitches": V.pitch_sounds(inst, stems_mono[stem], result, onsets_by_stem[stem], period=period)}
            # chords: where the voice strikes several notes at once, the song's own chord recordings
            steps = {}
            for e in inst["events"]:
                if e[2] is not None:
                    steps.setdefault((e[0], e[1]), set()).add(int(e[2]))
            poly_steps = sum(1 for v in steps.values() if len(v) > 1)
            if steps and poly_steps >= max(4, 0.15 * len(steps)):
                snd["chords"] = V.chord_sounds(inst, stems_mono[stem], result, onsets_by_stem[stem], period=period)
                if progress:
                    progress(f"{vid}: {poly_steps} chord hits, {len(snd['chords'])} distinct chord recordings")
            prog["_sounds"][vid] = snd
        else:
            from lib.dj import resynth as RS
            prog["_sounds"][vid] = {"chord": RS.exemplar_audio(inst, stems_mono)}
        if pitched or inst["kind"] == "sustain":
            prog.setdefault("_profile_pending", []).append((vid, inst, model))
        # patterns
        # held chords stay one item PER TONE (step, interval from the bar's root, length): the chord's
        # shape is in the program, so the additive model plays every tone and a shape change is a
        # pattern difference, not lost. (Before 2026-09-09 a held chord was one item with the lowest
        # note: the additive render played the root alone - the synthetic pads explained 13-30%.)
        bars = _bars_of(inst, result)
        # the unit (1, 2 or 4 bars) whose patterns + ops explain the voice with the fewest entries
        best = None
        for unit in (1, 2, 4):
            ub = _unit_bars(bars, unit)
            pats, seq, groove = infer_patterns(ub, pitched, SIM_DRUMS if stem == "drums" else SIM_PITCHED)
            cost = len(pats) + sum(len(e["ops"]) for e in seq)
            if best is None or cost < best[0]:
                best = (cost, unit, pats, seq, groove)
        _cost, unit, pats, seq, groove = best
        prog.setdefault("units", {})[vid] = unit
        prog["patterns"][vid], prog["sequence"][vid], prog["groove"][vid] = pats, seq, groove
        n_events += len(inst["events"]); n_pat += len(pats); n_seq += len(seq); n_ops += sum(len(e["ops"]) for e in seq)
        if progress:
            progress(f"{vid}: {len(bars)} bars -> {len(pats)} patterns, {sum(len(e['ops']) for e in seq)} ops")
    # the additive profile (always kept) and the model choice, by what explains the stem better -
    # measured on the stem with the OTHER voices' renders as a mask (analysis by synthesis: the
    # program so far, sample models). Measured on the synthetic gate 2026-09-09 without this: the
    # melody's fundamentals sit on the pad's second harmonics, the pad profile came out 7 dB too
    # loud and far too bright (pads vs the true pad: 31 dB)
    for vid, inst, model in prog.pop("_profile_pending", []):
        stem = inst["stem"]
        try:
            from lib.dj import additive as AD
            others = [v for v, voice in prog["voices"].items() if voice["stem"] == stem and v != vid and v in prog["patterns"]]
            y = stems_mono[stem]
            oy = None
            if others:
                # the other voices' render is a spectral MASK for the profile (where they own a harmonic,
                # it is not read), not a subtraction: a sample render is not phase-aligned to the recording
                rec = render(prog, {stem: y}, ids=others, t0=0.0, t1=len(y) / RATE, room=False, use_residual=False, fast=True)[:, 0][: len(y)]
                oy = np.zeros_like(y)
                oy[: len(rec)] = rec
            prof = AD.analyze_voice(inst, y, result, others_y=oy)
            if prof is not None:
                if CALIBRATE_ADDITIVE:
                    # the profile's level (a low percentile over harmonics, under the other voices' mask)
                    # under-reads a real voice by up to 12 dB (the render gate 2026-09-09: ballad piano and
                    # strings -12.5 dB). Tie it to the voice's OWN exemplar recording: the additive render of
                    # the exemplar's note at the exemplar's velocity must have the exemplar's level
                    try:
                        from lib.dj import resynth as RS
                        ex = RS.exemplar_audio(inst, stems_mono)
                        em = inst.get("exemplar_midi")
                        if ex is not None and em is not None and len(ex) > int(0.1 * RATE):
                            # exemplar_audio scales the cut to level_dbfs, the level of the voice's LOUD hits (velocity
                            # 1.0), so the additive note is rendered at velocity 1.0 too
                            head_n = min(len(ex), int(CALIBRATE_WIN_S * RATE))
                            piece = AD.render_note(prof, int(em), max(CALIBRATE_WIN_S, len(ex) / RATE), 1.0)
                            r_ex = float(np.sqrt(np.mean(ex[:head_n].astype(np.float64) ** 2)) + 1e-9)
                            r_ad = float(np.sqrt(np.mean(piece[:head_n].astype(np.float64) ** 2)) + 1e-9)
                            adj = float(np.clip(20 * np.log10(r_ex / r_ad), -20.0, 20.0))
                            prof["peak_db"] = float(prof["peak_db"] + adj)
                            prof["calibration_db"] = adj
                    except Exception as e:  # noqa: BLE001
                        if progress:
                            progress(f"{vid}: additive level calibration skipped ({type(e).__name__}: {str(e)[:60]})")
                prog["_sounds"][vid]["profile"] = prof
                # a HELD voice plays additive whenever it has a profile: the alternative is the exemplar
                # chunk looped over the hold, which carries whatever else was sounding in that cut (the
                # synthetic gate: the melody's plucks repeating inside the pad - onset F1 0.98 -> 0.70)
                # and is not notes. Struck voices keep the measured choice (samples vs additive).
                better = inst["kind"] == "sustain" or _additive_better(inst, y, result, prog["_sounds"][vid], model, period)
                if better:
                    prog["voices"][vid]["model"] = "additive"
                if progress:
                    ch = prog["_sounds"][vid].get("chooser") or {}
                    progress(f"{vid}: additive profile from {prof['n_notes']} notes (other voices masked"
                             f"{', level %+.1f dB' % prof['calibration_db'] if 'calibration_db' in prof else ''}) -> "
                             f"{'additive' if better else model} plays"
                             + (f" (stem gap samples {ch['samples_db']:.1f} / additive {ch['additive_db']:.1f} dB over {ch['n']} notes)" if ch else ""))
        except Exception as e:  # noqa: BLE001
            if progress:
                progress(f"{vid}: additive profile skipped ({type(e).__name__}: {str(e)[:60]})")
    # a sound whose exemplar hit has other sounds' events on it (it never plays alone: a clap that
    # always lands with the kick) gets those sounds' exemplars - clean, isolated, at the level the
    # reading gives them there - subtracted from its cut; sounds with a clean exemplar are left alone
    kit_vids = prog.pop("_kit_pending", [])
    # (measured on the synthetic gate 2026-09-09: the cascade made the mixture exemplar LOUDER - the
    # credited events' velocities it subtracts with are not exact enough - so it is opt-in)
    if clean_mixtures and len(kit_vids) > 1 and "drums" in stems_mono:
        from lib.dj import resynth as RS
        by_id = {i["id"]: i for i in insts}
        ex_t = {v: float(by_id[v]["exemplar"][0]) + V.PRE_S for v in kit_vids}
        times = {v: V.event_times(by_id[v], result) for v in kit_vids}
        vels = {v: np.array([e[4] for e in by_id[v]["events"]]) for v in kit_vids}
        conf = {v: np.array([e[5] for e in by_id[v]["events"]]) for v in kit_vids}
        # which other sounds sit on each exemplar hit (within 15 ms of it, or ringing into its cut)
        on = {v: [o for o in kit_vids if o != v and len(times[o])
                  and np.any((times[o] > ex_t[v] - 0.015) & (times[o] < ex_t[v] + float(by_id[v]["exemplar"][1]) - float(by_id[v]["exemplar"][0])))]
              for v in kit_vids}
        clean = {v: not on[v] for v in kit_vids}
        n_sub = 0
        # cascade: a sound is cleaned once every sound on its exemplar is clean (the hat alone first,
        # then the kick with the hat subtracted, then the clap with both subtracted)
        for _pass in range(len(kit_vids)):
            progressed = False
            for v in kit_vids:
                if clean[v] or not all(clean[o] for o in on[v]):
                    continue
                others = [(times[o], vels[o], prog["_sounds"][o]["layers"]) for o in on[v] if prog["_sounds"][o]]
                a_s, b_s = by_id[v]["exemplar"]
                seg = V.subtract_others(stems_mono["drums"], a_s, b_s, others, lambda x: (float(x) - 1.0) * 24.0)
                if len(seg) > 64:
                    fo = min(int(0.02 * RATE), len(seg) // 3)
                    seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
                    prog["_sounds"][v] = {"layers": [{"db": 0.0, "audio": seg}], "decay_db_s": None}
                    n_sub += 1
                clean[v] = True
                progressed = True
            if not progressed:
                break
        if progress and n_sub:
            progress(f"drums: {n_sub} mixture exemplars cleaned by subtracting the isolated sounds")
    for stem in ("drums",) if room else ():           # only a transient stem exposes its room between hits
        if stem in stems_mono and len(onsets_by_stem[stem]) > 8 and stem not in verbatim_stems:
            rm = V.room(stems_mono[stem], onsets_by_stem[stem])
            if rm:
                prog["rooms"][stem] = rm
    # the chord track: one chord per bar from the melodic stems, the bass naming the root
    prog["chords"] = []
    if chords:
        try:
            from lib.dj import chords as CH
            prog["chords"] = CH.chord_track(stems_mono, beats, prog["grid"]["down0"], period, progress=progress)
        except Exception as e:  # noqa: BLE001
            if progress:
                progress(f"chords skipped ({type(e).__name__}: {str(e)[:60]})")
    # the unexplained-residual test: bars the voices cannot explain are the stem's own audio
    n_verbatim_bars = 0
    note_share, bar_gaps = {}, {}
    for stem in (RESIDUAL_STEMS if residual else ()):
        if stem in verbatim_stems or stem not in stems_mono:
            continue
        vids = [v for v, d in prog["voices"].items() if d["stem"] == stem]
        if not vids:
            # no voice read at all: the stem is the recording throughout
            prog["verbatim"].append({"stem": stem, "bar0": 0, "bars": int(n_bars), "residual": True})
            note_share[stem] = 0.0
            n_verbatim_bars += int(n_bars)
            continue
        bad, gaps = residual_bars(prog, stems_mono, stem, vids)
        n_verbatim_bars += len(bad)
        for b0, nb in _runs(bad):
            prog["verbatim"].append({"stem": stem, "bar0": int(b0), "bars": int(nb), "residual": True})
        active = len(gaps)                                   # bars the stem sounds on
        note_share[stem] = round(1.0 - len(bad) / active, 3) if active else 1.0
        bar_gaps[stem] = round(float(np.median(list(gaps.values()))), 1) if gaps else None
        if progress:
            progress(f"{stem}: {len(bad)} of {active} sounding bars unexplained by the voices -> the recording there; "
                     f"notes on {100 * note_share[stem]:.0f}% (median bar gap {bar_gaps[stem]} dB)")
    ph = (prog.get("phrases") or {}).get("vocals") or {}
    prog["stats"] = {"events": n_events, "patterns": n_pat, "sequence": n_seq, "ops": n_ops,
                     "events_per_entry": round(n_events / max(n_pat + n_ops, 1), 1),
                     "verbatim_bars": n_verbatim_bars, "bars": int(n_bars), "note_share": note_share, "bar_gap_db": bar_gaps,
                     "vocal_phrases": len(ph.get("placements") or []), "vocal_distinct": len(ph.get("library") or [])}
    if explain:
        # the reading's own check (measure, prune, merge - each step verified on the stem), then this
        # build again on what survived; readings made by identify() already carry it
        import copy
        from lib.dj import explain as EX
        result2 = EX.explain_pass(copy.deepcopy(result), stems_mono, progress=progress)
        if result2.get("pruned"):
            prog = build(result2, stems, sections=sections, verbatim_stems=verbatim_stems, progress=progress, chords=chords,
                         residual=residual, room=room, fit_levels=fit_levels, exemplar_kit=exemplar_kit, clean_mixtures=clean_mixtures)
        prog["explain"] = {i["id"]: {k: i[k] for k in ("explained", "overshoot", "spurious", "share_db") if k in i}
                           for i in INS.instruments(result2) if "explained" in i}
        prog["pruned"] = result2.get("pruned") or []
    return prog


def _runs(bars):
    """sorted bar indices -> [(bar0, n)] runs"""
    out = []
    for b in sorted(bars):
        if out and out[-1][0] + out[-1][1] == b:
            out[-1][1] += 1
        else:
            out.append([b, 1])
    return [tuple(r) for r in out]


def residual_bars(prog, stems_mono, stem, vids):
    """Bars of `stem` where the rendered voices leave more than
    RESIDUAL_DB unexplained (mean |dB| over a 48-band log-mel, on the
    bands where the stem has energy)."""
    import librosa
    y = stems_mono[stem]
    rec = render(prog, {stem: y}, ids=vids, t0=0.0, t1=len(y) / RATE, room=False)[:, 0][: len(y)]
    hop = 1024
    M = lambda x: 10 * np.log10(librosa.feature.melspectrogram(y=np.ascontiguousarray(x), sr=RATE, n_fft=2048, hop_length=hop,
                                                              n_mels=48, fmin=30.0, fmax=12000.0, power=2.0) + 1e-10)
    Mr, Mc = M(y), M(rec)
    beats = _beat_times(prog)
    g = prog["grid"]
    floor = float(np.percentile(Mr.max(axis=0), 95)) - 35.0
    bad = []
    n_bars = (len(beats) - g["down0"]) // 4
    thr = RESIDUAL_DB_STEM.get(stem, RESIDUAL_DB)
    gaps = {}
    # the note-level checks: the render's chroma must follow the recording's beat by beat (pitched stems), and it
    # may not strike far more onsets than the recording - a spectral mean cannot tell wrong notes on the right
    # instrument, nor a sustained texture read as a hail of hits
    from lib.dj import fidelity as F
    period = g["period_s"]
    pitched = stem != "drums"
    Cr = Cc = None
    if pitched:
        Cr, Cc = F.chroma_beats(y, beats, period), F.chroma_beats(rec, beats, period)
    on_r, on_c = np.asarray(F.onsets(y)), np.asarray(F.onsets(rec))
    why = {}
    for b in range(n_bars):
        k0 = g["down0"] + b * 4
        t_a = float(beats[k0])
        t_b = t_a + 4 * period
        f0 = int(t_a * RATE / hop)
        f1 = int(min(Mr.shape[1], t_b * RATE / hop))
        if f1 <= f0 + 1:
            continue
        seg_r, seg_c = Mr[:, f0:f1], Mc[:, f0:f1]
        if seg_r.max() < floor:
            continue                                     # the stem is quiet here: nothing to explain
        mask = seg_r > floor
        if not mask.any():
            continue
        gap = float(np.mean(np.abs(seg_r[mask] - seg_c[mask])))
        gaps[b] = gap
        reasons = []
        if gap > thr:
            reasons.append(f"gap {gap:.0f} dB")
        if pitched and Cr is not None:
            rs = []
            for k in range(k0, min(k0 + 4, len(Cr), len(Cc))):
                if Cr[k].std() > 1e-6 and Cc[k].std() > 1e-6:
                    rs.append(float(np.corrcoef(Cr[k], Cc[k])[0, 1]))
            if rs and float(np.mean(rs)) < RESIDUAL_CHROMA_MIN:
                reasons.append(f"chroma {np.mean(rs):.2f}")
        n_r = int(np.sum((on_r >= t_a) & (on_r < t_b))) if len(on_r) else 0
        n_c = int(np.sum((on_c >= t_a) & (on_c < t_b))) if len(on_c) else 0
        if n_c > RESIDUAL_ONSET_RATIO * n_r + RESIDUAL_ONSET_SLACK:
            reasons.append(f"{n_c} hits for {n_r}")
        if reasons:
            bad.append(b)
            why[b] = ", ".join(reasons)
    # hysteresis: a short run of note bars between recording bars is the recording too
    if bad and RESIDUAL_MIN_RUN > 1:
        badset = set(bad)
        sounding = sorted(gaps)
        run = []
        for b in sounding + [None]:
            if b is not None and b not in badset:
                run.append(b)
                continue
            if run and len(run) < RESIDUAL_MIN_RUN:
                before = run[0] - 1 in badset
                after = b is not None and b in badset
                if before and after:
                    for r in run:
                        badset.add(r)
                        why[r] = "a lone note bar between recording bars"
            run = []
        bad = sorted(badset)
    gaps["_why"] = why
    return bad, gaps


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def _beat_times(prog):
    g = prog["grid"]
    return g["t0"] + np.arange(g["n_beats"]) * g["period_s"]


def expand(prog, vid):
    """The voice's sequence + patterns + ops + groove -> events [t_s, midi|None, dur_steps, vel]."""
    beats = _beat_times(prog)
    g = prog["grid"]
    period = g["period_s"]
    pats, groove = prog["patterns"][vid], prog["groove"].get(vid, {})
    out = []
    for e in prog["sequence"][vid]:
        p = pats[e["pat"]]
        items = {(it[0], it[1]): (it[2], it[3], (it[4] if len(it) > 4 else None)) for it in p["items"]}
        for op in e["ops"]:
            if op[0] == "add":
                items[(op[1], op[2])] = (op[3], op[4], (op[5] if len(op) > 5 else None))
            elif op[0] == "drop":
                items.pop((op[1], op[2]), None)
            elif op[0] == "nudge" and (op[1], op[2]) in items:
                d, v, _o = items[(op[1], op[2])]
                items[(op[1], op[2])] = (d, v, op[3])
            elif op[0] == "len" and (op[1], op[2]) in items:
                _d, v, o = items[(op[1], op[2])]
                items[(op[1], op[2])] = (int(op[3]), v, o)
        for (step, rel), (dur, vel, off) in items.items():
            beat = g["down0"] + e["bar"] * 4 + step // STEPS          # steps past 16 run into the unit's next bars
            if beat >= len(beats):
                continue
            if off is None:
                gr = groove.get(step % BAR) or groove.get(str(step % BAR)) or [0.0, vel]
                off = float(gr[0])
            t = INS.event_time(beats, beat, step % STEPS) + float(off)
            midi = (e["root"] + rel) if (rel is not None and e["root"] is not None) else None
            out.append([t, midi, dur, vel])
    out.sort()
    return out


def _gain_db(vel, kind):
    return (vel - 1.0) * (30.0 if kind == "sustain" else 24.0)


def verbatim_spans(prog, use_residual=True):
    """{stem: [(t_a, t_b)]} - the spans a stem plays as its own recording (the hybrid program). Its voices
    are silent there: the recording already holds them, and a note on top of its own recording doubled it."""
    beats = _beat_times(prog)
    g = prog["grid"]
    spans = {}
    for vb in prog.get("verbatim") or []:
        if vb.get("residual") and not use_residual:
            continue
        b0 = g["down0"] + vb["bar0"] * 4
        b1 = min(len(beats) - 1, b0 + vb["bars"] * 4)
        ta = float(beats[b0]) if b0 < len(beats) else 0.0
        tb = float(beats[b1]) if b1 < len(beats) else (float(beats[-1]) + g["period_s"])
        if vb["bar0"] == 0 and vb["bars"] >= (len(beats) - g["down0"]) // 4:
            ta, tb = 0.0, 1e9
        spans.setdefault(vb["stem"], []).append((ta, tb))
    return spans


def outside_verbatim(events, spans):
    """The (t, midi, dur, vel) events not starting inside any of the spans."""
    if not spans:
        return events
    return [e for e in events if not any(a <= e[0] < b for a, b in spans)]


def render(prog, stems, ids=None, t0=0.0, t1=None, room=True, progress=None, use_residual=True, limit="peak", voices=True, fast=False):
    """(n,2) float32: every voice in ids (default all) + the verbatim stems.
    limit: "peak" scales the whole render down when its single loudest
    sample exceeds 0.98 (the evaluator's convention); None leaves the
    levels as rendered (a caller that sums voices and limits the sum -
    the planner - wants the raw voice: one transient must not scale a
    whole song down, measured 7-15 dB on real tracks)."""
    from lib.dj import resynth as RS
    stems_mono = {n: V._mono(a) for n, a in stems.items()}
    period = prog["grid"]["period_s"]
    if t1 is None:
        t1 = max(len(y) / RATE for y in stems_mono.values())
    n = int((t1 - t0) * RATE) + 1
    buses = {}
    spans = verbatim_spans(prog, use_residual)
    vids = [v for v in prog["voices"] if ids is None or v in ids] if voices else []
    for q, vid in enumerate(vids):
        voice = prog["voices"][vid]
        stem_bus = buses.setdefault(voice["stem"], np.zeros(n, dtype=np.float32))
        events = outside_verbatim(expand(prog, vid), spans.get(voice["stem"]))
        stem_bus += _render_voice(prog, vid, events, n, t0, t1, fast=fast)
        if progress:
            progress(q + 1, len(vids), vid)
    out = np.zeros(n, dtype=np.float32)
    for stem, bus in buses.items():
        rm = prog["rooms"].get(stem) if room else None
        if rm:
            bus = V.reverb(bus, rm["rt60_s"], rm["wet_db"])
        out += bus
    beats = _beat_times(prog)
    g = prog["grid"]
    # the phrases and verbatim slices of a stem play when ids is None or names the stem itself (the planner's
    # per-stem "recording" unit; a voice id alone renders the voice only, or a mixer summing voice units
    # would carry the recording once per voice)
    def _stem_wanted(stem):
        return ids is None or stem in ids
    for stem, ph in (prog.get("phrases") or {}).items():
        y = stems_mono.get(stem)
        if y is None or not _stem_wanted(stem):
            continue
        for pl in ph["placements"]:
            a, b = ph["library"][pl["phrase"]]
            if pl["t"] >= t1 or pl["t"] + (b - a) <= t0:
                continue
            piece = y[int(a * RATE):int(b * RATE)].copy() * np.float32(10 ** (pl["gain_db"] / 20.0))
            f = min(int(0.02 * RATE), len(piece) // 4)
            if f > 0:
                piece[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
                piece[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
            _place(out, piece, pl["t"] - t0, n)
    for vb in prog["verbatim"]:
        y = stems_mono.get(vb["stem"])
        if y is None:
            continue
        if not _stem_wanted(vb["stem"]):
            continue
        if vb.get("residual") and not use_residual:
            continue
        b0 = g["down0"] + vb["bar0"] * 4
        b1 = min(len(beats) - 1, b0 + vb["bars"] * 4)
        ta = float(beats[b0]) if b0 < len(beats) else t0
        tb = float(beats[b1]) if b1 < len(beats) else (float(beats[-1]) + g["period_s"])
        if vb["bar0"] == 0 and vb["bars"] >= (len(beats) - g["down0"]) // 4:
            ta, tb = 0.0, len(y) / RATE                  # the whole stem
        a, b = int(max(ta, t0) * RATE), int(min(len(y), min(tb, t1) * RATE))
        if b <= a:
            continue
        piece = y[a:b].copy()
        f = min(int(0.01 * RATE), len(piece) // 4)
        if f > 0:
            piece[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
            piece[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        _place(out, piece, a / RATE - t0, n)
    if limit == "peak":
        peak = float(np.abs(out).max())
        if peak > 0.98:
            out *= 0.98 / peak
    return np.stack([out, out], axis=1)


STEM_MATCH_DB = 6.0            # mixdown: a stem's bus is moved at most this far to match the stem's level where its voices play


def limit(out, ceiling=0.98, attack_s=0.005, release_s=0.08):
    """A look-ahead peak limiter over a (n,) or (n,2) buffer, in place: the
    gain is down before a transient and back over release_s. Only the
    samples over the ceiling are touched - scaling a whole buffer by its one
    loudest sample took a real track down 15 dB for one hot transient."""
    n = len(out)
    blk = max(int(0.001 * RATE), 1)
    m = n // blk
    if m < 4:
        np.clip(out, -ceiling, ceiling, out)
        return out
    width = blk * (out.shape[1] if out.ndim == 2 else 1)
    peaks = np.abs(out[: m * blk]).reshape(m, width).max(axis=1)
    if float(peaks.max()) <= ceiling:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    k = max(int(attack_s * 1000), 1)
    ahead = sliding_window_view(np.concatenate([peaks, np.full(k - 1, peaks[-1])]), k).max(axis=1)
    req = np.minimum(1.0, ceiling / np.maximum(ahead, 1e-9))
    step = 1.0 / max(release_s * 1000, 1.0)
    g = np.empty(m, dtype=np.float64)
    cur = 1.0
    for i in range(m):
        cur = min(req[i], cur + step)
        g[i] = cur
    gain = np.interp(np.arange(n), (np.arange(m) + 0.5) * blk, g).astype(np.float32)
    out *= gain[:, None] if out.ndim == 2 else gain
    np.clip(out, -1.0, 1.0, out)
    return out


def mixdown(prog, stems, ids=None, t0=0.0, t1=None, progress=None, use_residual=True, stem_match=True):
    """The program rendered the way it should be HEARD: every voice raw (no
    whole-track clamp), capped where it renders louder than its stem inside
    its own event windows (voice_ceiling_db, one-sided), each stem's bus then
    matched to the stem's level over those windows (+-STEM_MATCH_DB, an rms
    match, not the spectral gap), the verbatim slices and phrases added, and a
    look-ahead limiter on the sum. -> (n,2) float32. The planner's Analysis
    tab does the same per cached voice; Replay and the CLI render used to go
    through render(limit="peak") and came out 7-15 dB down under one hot voice."""
    stems_mono = {n: V._mono(a) for n, a in stems.items()}
    if t1 is None:
        t1 = max(len(y) / RATE for y in stems_mono.values())
    n = int((t1 - t0) * RATE) + 1
    vids = [v for v in prog["voices"] if ids is None or v in ids]
    buses, masks = {}, {}
    w = int(0.3 * RATE)
    spans = verbatim_spans(prog, use_residual)
    for q, vid in enumerate(vids):
        voice = prog["voices"][vid]
        events = outside_verbatim(expand(prog, vid), spans.get(voice["stem"]))
        y = _render_voice(prog, vid, events, n, t0, t1)
        db = voice_ceiling_db(prog, stems_mono, vid, y, t0=t0)
        if db < -0.5:
            y *= np.float32(10 ** (db / 20.0))
        bus = buses.setdefault(voice["stem"], np.zeros(n, dtype=np.float32))
        bus += y
        mask = masks.setdefault(voice["stem"], np.zeros(n, dtype=bool))
        for t, _m, _d, _v in events:
            a = int((t - t0) * RATE)
            if 0 <= a < n:
                mask[a: min(n, a + w)] = True
        if progress:
            progress(q + 1, len(vids), vid)
    out = np.zeros(n, dtype=np.float32)
    for stem, bus in buses.items():
        ref = stems_mono.get(stem)
        if stem_match and ref is not None and masks[stem].any():
            off = int(t0 * RATE)
            m = min(n, len(ref) - off)
            if m > RATE:
                sel = masks[stem][:m]
                r_ref = float(np.sqrt(np.mean(ref[off: off + m][sel].astype(np.float64) ** 2)) + 1e-9)
                r_bus = float(np.sqrt(np.mean(bus[:m][sel].astype(np.float64) ** 2)) + 1e-9)
                g = float(np.clip(20 * np.log10(r_ref / r_bus), -STEM_MATCH_DB, STEM_MATCH_DB))
                bus *= np.float32(10 ** (g / 20.0))
        out += bus
    # the verbatim slices and vocal phrases (the recording's own audio, at its own level)
    if prog.get("verbatim") or prog.get("phrases"):
        extra = render(prog, stems, ids=None, t0=t0, t1=t1, room=False, use_residual=use_residual, limit=None, voices=False)[:, 0]
        out[: min(n, len(extra))] += extra[:n]
    return limit(np.stack([out, out], axis=1))


def _extend_with_profile(sample, prof, midi, want, vel, gain_db):
    """sample (the recording, ends early) -> `want` samples: the recording, then the additive note's
    continuation from the recording's end, level-matched over a 30 ms crossfade."""
    from lib.dj import additive as AD
    n_s = len(sample)
    xf = min(int(0.03 * RATE), n_s // 3)
    if xf < 64:
        return sample
    piece = AD.render_note(prof, midi, want / RATE, vel)
    if len(piece) <= n_s:
        return sample
    # the additive note's level where the recording ends, matched to the recording's last stretch (the
    # recording carries the event gain applied by the caller; the additive piece must too)
    g_lin = 10 ** (gain_db / 20.0)
    a = sample[n_s - xf:].astype(np.float64) * g_lin
    b = piece[n_s - xf: n_s].astype(np.float64)
    r_a = float(np.sqrt(np.mean(a ** 2)) + 1e-9)
    r_b = float(np.sqrt(np.mean(b ** 2)) + 1e-9)
    tail = piece[n_s - xf: want] * np.float32(min(30.0, r_a / r_b) / max(g_lin, 1e-6))
    out = np.zeros(max(n_s, min(want, n_s - xf + len(tail))), dtype=np.float32)
    out[:n_s] = sample
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    out[n_s - xf: n_s] = sample[n_s - xf:] * (1.0 - ramp) + tail[:xf] * ramp
    rest = tail[xf: xf + (len(out) - n_s)]
    out[n_s: n_s + len(rest)] = rest
    return out


def render_events(prog, vid, events, t0=0.0, t1=None):
    """One voice's SOUND MODEL playing an explicit event list [[t_s, midi|None,
    dur_steps, vel], ...] -> mono float32 from t0 to t1 (the render gate plays
    a song's TRUE notes through its extracted voices, so the sound model is
    measured apart from the reading)."""
    if t1 is None:
        t1 = max((e[0] for e in events), default=t0) + MAX_SAMPLE_S
    n = int((t1 - t0) * RATE) + 1
    return _render_voice(prog, vid, sorted(events), n, t0, t1)


def _chord_vel(prog, vels):
    """The velocity a chord recording plays at for tones struck together: their combined level on
    split-level readings (voices.GROUP_LEVEL_REF: the recording's reference is the chord's level), the
    loudest tone's on older readings (where every tone carried the chord's level already)."""
    vels = [float(v) for v in vels] or [1.0]
    return V.group_vel(vels) if V.grouped_levels(prog.get("reading_version")) else max(vels)


_NOTE_CACHES = {}              # id(profile) -> (profile, {note key: audio}); see _note_cache


def _note_cache(prof):
    """Additive notes memoised per profile: repetitive music plays the same pitch at the same length
    and velocity over and over, and render_note is the render's cost (26-47 s per voice on a real
    track before this)."""
    from lib.dj import additive as AD
    # the cache lives as long as the profile object does (a build renders every voice several times
    # for the profile masks, the planner renders the same program over and over): one per profile,
    # identity-checked, the table bounded
    ent = _NOTE_CACHES.get(id(prof))
    if ent is None or ent[0] is not prof:
        if len(_NOTE_CACHES) >= 64:
            _NOTE_CACHES.pop(next(iter(_NOTE_CACHES)))
        ent = _NOTE_CACHES[id(prof)] = (prof, {})
    cache = ent[1]

    def note(midi, length_s, vel, min_decay_db_s=None, phase_seed=0):
        # the level is in the key: the build calibrates peak_db after the profile exists
        key = (int(midi), round(float(length_s), 2), round(float(vel), 2), min_decay_db_s, phase_seed, round(float(prof["peak_db"]), 3))
        piece = cache.get(key)
        if piece is None:
            piece = AD.render_note(prof, int(midi), length_s, vel, min_decay_db_s=min_decay_db_s, phase_seed=phase_seed)
            if len(cache) < 4096:
                cache[key] = piece
        return piece
    return note


def _render_voice(prog, vid, events, n, t0, t1, fast=False):
    """The voice's events through its sound model -> its bus (n samples from t0), the voice's
    verified gain (lib/dj/explain.py "gain" candidates) applied. fast: no additive holds or tails
    for sample voices (the build's masking render needs the energy, not the articulation)."""
    from lib.dj import resynth as RS
    voice = prog["voices"][vid]
    snd = prog["_sounds"].get(vid) or {}
    period = prog["grid"]["period_s"]
    bus = np.zeros(n, dtype=np.float32)
    note_of = _note_cache(snd["profile"]) if snd.get("profile") else None
    for _once in (0,):                                   # the body's `continue`s leave the bus empty
        if voice["model"] == "additive" and snd.get("profile"):
            prof = snd["profile"]
            kind = "sustain" if voice["kind"] == "sustain" else "hit"
            for t, midi, dur, vel in events:
                if t < t0 - 60 or t >= t1 or midi is None:
                    continue
                # a held note plays its whole measured length (a pad held 30 s is 30 s of sinusoids, cheap);
                # a struck note's tail is capped
                length = dur * period / STEPS if kind == "sustain" else min(MAX_SAMPLE_S * 2, dur * period / STEPS)
                piece = note_of(int(midi), length, vel)
                _place(bus, piece, t - t0, n)
        elif voice["model"] == "chord":
            ex = snd.get("chord")
            if ex is None:
                continue
            base = voice.get("exemplar_midi")
            cache = {}
            # the tones held from one start play as ONE chord recording (the exemplar, repitched to the
            # lowest tone): the program holds every tone, the sample is the voice's chord sound
            by_t = {}
            for t, midi, dur, vel in events:
                by_t.setdefault(round(t, 4), []).append((t, midi, dur, vel))
            for _tt, grp in sorted(by_t.items()):
                t = grp[0][0]
                if t < t0 - 60 or t >= t1:
                    continue
                ms = [int(m) for _t, m, _d, _v in grp if m is not None]
                midi = min(ms) if ms else None
                dur, vel = max(d for _t, _m, d, _v in grp), _chord_vel(prog, [v for _t, _m, _d, v in grp])
                key = (int(midi) - int(base)) if (midi is not None and base is not None) else 0
                if key not in cache:
                    cache[key] = RS.repitch(ex, key)
                want = int((dur * period / STEPS + RELEASE_S) * RATE)
                piece = RS.looped(cache[key], want) * np.float32(10 ** (_gain_db(vel, "sustain") / 20.0))
                _place(bus, piece, t - t0, n)
        elif voice["model"] == "pitched":
            pitches = snd.get("pitches") or {}
            if not pitches:
                continue
            avail = np.array(sorted(pitches))
            cache = {}
            fb_m = int(voice["exemplar_midi"]) if voice.get("exemplar_midi") in pitches else int(avail[len(avail) // 2])
            chords = snd.get("chords") or {}
            if chords:
                # chord hits: one recording per struck chord; the single notes of a chord step are
                # consumed by it (a chord is not its notes stacked)
                by_t = {}
                for t, midi, dur, vel in events:
                    by_t.setdefault(round(t, 4), []).append((t, midi, dur, vel))
                single = []
                for tt, grp in by_t.items():
                    ms = sorted(set(int(m) for _t, m, _d, _v in grp if m is not None))
                    if len(ms) < 2:
                        single.extend(grp)
                        continue
                    root, shape = ms[0], tuple(m - ms[0] for m in ms)
                    rec, semis = V.chord_lookup(chords, root, shape)
                    if rec is None:
                        single.extend(grp)
                        continue
                    t = grp[0][0]
                    if t < t0 - 4 or t >= t1:
                        continue
                    if semis:
                        key = ("chord", root, shape)
                        if key not in cache:
                            cache[key] = RS.repitch(rec, semis)
                        rec = cache[key]
                    dur, vel = max(d for _t, _m, d, _v in grp), _chord_vel(prog, [v for _t, _m, _d, v in grp])
                    length_s = dur * period / STEPS
                    want = int(min(MAX_SAMPLE_S, length_s + RELEASE_S) * RATE)
                    g_lin = np.float32(10 ** (_gain_db(vel, "hit") / 20.0))
                    if HOLD_ADDITIVE and not fast and length_s >= HOLD_ADDITIVE_S and length_s * RATE > HOLD_ADDITIVE_X * len(rec):
                        # a held chord far longer than its recording (a pad; the recording is the voice's struck
                        # chord and would die): every tone sustained by the additive profile, the sum at the
                        # recording's level at this gain - else the recording's steady part looped
                        head_n = int(0.15 * RATE)
                        ref_rms = float(np.sqrt(np.mean((rec[:head_n].astype(np.float64) * g_lin) ** 2)) + 1e-9)
                        prof_c = snd.get("profile")
                        if prof_c is not None:
                            tones = [note_of(m_, length_s, vel, min_decay_db_s=HOLD_MIN_DECAY_DB_S, phase_seed=k_) for k_, m_ in enumerate(ms)]
                            L = max(len(x) for x in tones)
                            piece = np.zeros(L, dtype=np.float32)
                            for x in tones:
                                piece[: len(x)] += x
                            p_rms = float(np.sqrt(np.mean(piece[:head_n].astype(np.float64) ** 2)) + 1e-9)
                            piece = piece * np.float32(min(100.0, ref_rms / p_rms))
                        else:
                            piece = RS.looped(rec, int(length_s * RATE)) * g_lin
                            fo = min(int(0.05 * RATE), len(piece) // 4)
                            if fo > 0:
                                piece = piece.copy()
                                piece[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
                        _place(bus, piece, t - t0, n)
                        continue
                    piece = rec[:want] * g_lin
                    _place(bus, piece, t - t0 - V.PRE_S, n)
                events = sorted(single)
            prof = snd.get("profile") if (HOLD_ADDITIVE and not fast) else None
            for t, midi, dur, vel in events:
                if t < t0 - 4 or t >= t1:
                    continue
                length_s = dur * period / STEPS
                want = int(min(MAX_SAMPLE_S, length_s + RELEASE_S) * RATE)
                db = _gain_db(vel, "hit")
                if midi is None:                        # an unpitched hit of a pitched voice: a note without a read pitch
                    rec, gain = V.pick_layer(pitches[fb_m], db)
                    sample = rec["short"]
                elif int(midi) in pitches or prof is not None:
                    near_m = int(midi) if int(midi) in pitches else int(avail[np.argmin(np.abs(avail - int(midi)))])
                    rec, gain = V.pick_layer(pitches[near_m], db)
                    have = rec["held"] if rec.get("held") is not None else rec["short"]
                    if length_s >= HOLD_ADDITIVE_S and length_s * RATE > HOLD_ADDITIVE_X * len(have):
                        # a note far longer than any recording of it: the sample would stop and leave silence (the
                        # render gate 2026-09-09: pads and organ through hit samples measured 50-57 dB from the true
                        # part). The voice's additive profile holds the note for its whole length, SUSTAINED (a held
                        # articulation does not die the way the voice's struck notes do) and at the level the
                        # recording has at the same event gain; without a profile the recording's steady part loops
                        g_lin = np.float32(10 ** (gain / 20.0))
                        head_n = int(0.15 * RATE)
                        ref_rms = float(np.sqrt(np.mean((have[:head_n].astype(np.float64) * g_lin) ** 2)) + 1e-9)
                        if prof is not None:
                            piece = note_of(int(midi), length_s, vel, min_decay_db_s=HOLD_MIN_DECAY_DB_S)
                            p_rms = float(np.sqrt(np.mean(piece[:head_n].astype(np.float64) ** 2)) + 1e-9)
                            piece = piece * np.float32(min(100.0, ref_rms / p_rms))
                        else:
                            piece = RS.looped(have if int(midi) == near_m else RS.repitch(have, int(midi) - near_m), int(length_s * RATE)) * g_lin
                            fo = min(int(0.05 * RATE), len(piece) // 4)
                            if fo > 0:
                                piece = piece.copy()
                                piece[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
                        _place(bus, piece, t - t0, n)
                        continue
                    if int(midi) not in pitches:
                        m = int(midi)
                        key = (near_m, m, want > len(rec["short"]) * 0.8, round(rec["db"], 1) if "db" in rec else None)
                        if key not in cache:
                            cache[key] = RS.repitch(V.sustain_sample(rec, int(want * 2 ** ((m - near_m) / 12.0)) + 16), m - near_m)
                        sample = cache[key]
                    else:
                        sample = V.sustain_sample(rec, want)
                    if TAIL_ADDITIVE and prof is not None and not fast and want > len(sample) + int(0.05 * RATE) and rec.get("held") is None:
                        # the recording ends before the note does (cut at the next onset of any sound): keep its
                        # attack and continue from its end level through the voice's additive profile, crossfaded -
                        # instead of the recording's last 40 ms looped under a fitted decay
                        sample = _extend_with_profile(sample, prof, int(midi), want, vel, gain)
                else:
                    m = int(midi)
                    near = int(avail[np.argmin(np.abs(avail - m))])
                    rec, gain = V.pick_layer(pitches[near], db)
                    key = (near, m, want > len(rec["short"]) * 0.8, round(rec["db"], 1) if "db" in rec else None)
                    if key not in cache:
                        cache[key] = RS.repitch(V.sustain_sample(rec, int(want * 2 ** ((m - near) / 12.0)) + 16), m - near)
                    sample = cache[key]
                piece = sample[:want] * np.float32(10 ** (gain / 20.0))
                _place(bus, piece, t - t0 - V.PRE_S, n)
        else:                                            # kit: velocity layers
            if not snd or not snd.get("layers"):
                continue
            layers = snd["layers"]
            ldb = np.array([l["db"] for l in layers])
            for t, midi, dur, vel in events:
                if t < t0 - 4 or t >= t1:
                    continue
                db = _gain_db(vel, "hit")
                li = int(np.argmin(np.abs(ldb - db)))
                sample = layers[li]["audio"]
                piece = sample * np.float32(10 ** ((db - ldb[li]) / 20.0))
                _place(bus, piece, t - t0 - V.PRE_S, n)
    g = float(voice.get("gain_db") or 0.0)
    return bus if g == 0.0 else bus * np.float32(10 ** (g / 20.0))


def voice_ceiling_db(prog, stems, vid, y, t0=0.0, win_s=0.3):
    """How far (dB, <= 0) a voice's render `y` (mono, from t0) must come
    down so it is no louder than its stem inside its own event windows
    (onset .. +win_s). A voice is part of its stem, so a render above the
    stem there is a levels artifact of the reading (a per-pitch sample
    'un-velocitied' by 20 dB, then an event at +8 dB: measured 15-20 dB
    hot on a real track), not music - and one such voice used to scale
    a whole track down through the peak clamp. 0.0 when it is within."""
    voice = prog["voices"].get(vid)
    if voice is None:
        return 0.0
    ref = stems.get(voice["stem"])
    if ref is None:
        return 0.0
    ref = V._mono(ref)
    y = np.asarray(y, dtype=np.float32)
    n = min(len(y), len(ref) - int(t0 * RATE))
    if n <= 0:
        return 0.0
    mask = np.zeros(n, dtype=bool)
    w = int(win_s * RATE)
    for t, _m, _d, _v in expand(prog, vid):
        a = int((t - t0) * RATE)
        if 0 <= a < n:
            mask[a: min(n, a + w)] = True
    if not mask.any():
        return 0.0
    off = int(t0 * RATE)
    r_ref = float(np.sqrt(np.mean(ref[off: off + n][mask].astype(np.float64) ** 2)) + 1e-9)
    r_y = float(np.sqrt(np.mean(y[:n][mask].astype(np.float64) ** 2)) + 1e-9)
    return min(0.0, 20.0 * np.log10(r_ref / r_y))


def _place(bus, piece, at_s, n):
    a = int(at_s * RATE)
    if a < 0:
        piece = piece[-a:]
        a = 0
    b = min(n, a + len(piece))
    if b > a:
        fo = min(int(0.02 * RATE), len(piece) // 3)
        if fo > 0 and b - a == len(piece):
            piece = piece.copy()
            piece[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
        bus[a:b] += piece[: b - a]


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def save(prog, folder):
    """program.json + one wav per sample under folder/sounds/ (32-bit float: a kit cut can peak at
    -36 dBFS, and a 16-bit store left it nine bits - the planner's stored program then rendered 0.8 %
    of peak away from the fresh build; ~100 MB per 3-minute track, on a par with its stems)."""
    import soundfile as sf
    os.makedirs(os.path.join(folder, "sounds"), exist_ok=True)
    sounds_ref = {}
    for vid, snd in (prog.get("_sounds") or {}).items():
        if not snd:
            continue
        ref = {}
        if "layers" in snd:
            ref["layers"] = []
            for i, l in enumerate(snd["layers"]):
                p = os.path.join(folder, "sounds", f"{vid}_L{i}.wav")
                sf.write(p, l["audio"], RATE, subtype="FLOAT")
                ref["layers"].append({"db": l["db"], "file": p})
            ref["decay_db_s"] = snd.get("decay_db_s")
        if "pitches" in snd:
            ref["pitches"] = {}
            for m, rec in snd["pitches"].items():
                p = os.path.join(folder, "sounds", f"{vid}_{m}.wav")
                sf.write(p, rec["short"], RATE, subtype="FLOAT")
                entry = {"short": p}
                if rec.get("held") is not None:
                    ph = os.path.join(folder, "sounds", f"{vid}_{m}_held.wav")
                    sf.write(ph, rec["held"], RATE, subtype="FLOAT")
                    entry["held"] = ph
                if rec.get("layers"):
                    entry["layers"] = []
                    for li, l in enumerate(rec["layers"]):
                        pl = os.path.join(folder, "sounds", f"{vid}_{m}_v{li}.wav")
                        sf.write(pl, l["short"], RATE, subtype="FLOAT")
                        le = {"db": l["db"], "short": pl}
                        if l.get("held") is not None:
                            plh = os.path.join(folder, "sounds", f"{vid}_{m}_v{li}_held.wav")
                            sf.write(plh, l["held"], RATE, subtype="FLOAT")
                            le["held"] = plh
                        entry["layers"].append(le)
                ref["pitches"][str(m)] = entry
        if snd.get("chord") is not None:
            p = os.path.join(folder, "sounds", f"{vid}_chord.wav")
            sf.write(p, snd["chord"], RATE, subtype="FLOAT")
            ref["chord"] = p
        if snd.get("profile"):
            prof = snd["profile"]
            ref["profile"] = {"attack": prof["attack"].tolist(), "decay_db_s": prof["decay_db_s"].tolist(), "shape": prof["shape"].tolist(),
                              "release_s": prof["release_s"], "peak_db": prof["peak_db"], "n_notes": prof["n_notes"]}
        if snd.get("chords"):
            ref["chords"] = []
            for i, ((root, shape), a) in enumerate(snd["chords"].items()):
                p = os.path.join(folder, "sounds", f"{vid}_ch{i}_{root}.wav")
                sf.write(p, a, RATE, subtype="FLOAT")
                ref["chords"].append({"root": int(root), "shape": list(shape), "file": p})
        sounds_ref[vid] = ref
    d = {k: v for k, v in prog.items() if k != "_sounds"}
    d["sounds"] = sounds_ref
    path = os.path.join(folder, "program.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1)
    return path


def load(folder):
    import soundfile as sf
    with open(os.path.join(folder, "program.json"), encoding="utf-8") as fh:
        prog = json.load(fh)
    prog["_sounds"] = {}
    for vid, ref in (prog.pop("sounds", None) or {}).items():
        snd = {}
        if "layers" in ref:
            snd["layers"] = [{"db": l["db"], "audio": sf.read(l["file"], dtype="float32")[0]} for l in ref["layers"]]
            snd["decay_db_s"] = ref.get("decay_db_s")
        if "pitches" in ref:
            snd["pitches"] = {}
            for m, entry in ref["pitches"].items():
                if isinstance(entry, str):
                    entry = {"short": entry}
                rec = {"short": sf.read(entry["short"], dtype="float32")[0],
                       "held": sf.read(entry["held"], dtype="float32")[0] if entry.get("held") else None}
                if entry.get("layers"):
                    rec["layers"] = [{"db": l["db"], "short": sf.read(l["short"], dtype="float32")[0],
                                      "held": sf.read(l["held"], dtype="float32")[0] if l.get("held") else None} for l in entry["layers"]]
                snd["pitches"][int(m)] = rec
        if "chord" in ref:
            snd["chord"] = sf.read(ref["chord"], dtype="float32")[0]
        if ref.get("chords"):
            snd["chords"] = {(int(c["root"]), tuple(c["shape"])): sf.read(c["file"], dtype="float32")[0] for c in ref["chords"]}
        if ref.get("profile"):
            pr = ref["profile"]
            snd["profile"] = {"attack": np.array(pr["attack"], dtype=np.float32), "decay_db_s": np.array(pr["decay_db_s"], dtype=np.float32),
                              "shape": np.array(pr["shape"], dtype=np.float32), "release_s": pr["release_s"], "peak_db": pr["peak_db"],
                              "n_notes": pr["n_notes"]}
        prog["_sounds"][vid] = snd
    # json turned the groove's int keys into strings
    prog["groove"] = {v: {int(k): val for k, val in g.items()} for v, g in prog["groove"].items()}
    return prog


def describe(prog):
    s = prog["stats"]
    ch = prog.get("chords") or []
    changes = sum(1 for i in range(1, len(ch)) if ch[i]["name"] != ch[i - 1]["name"])
    lines = [f"{len(prog['voices'])} voices, {s['patterns']} patterns, {s['sequence']} bar entries, {s['ops']} ops "
             f"for {s['events']} events ({s['events_per_entry']} events per pattern+op); chords {len(set(c['name'] for c in ch))} distinct, "
             f"{changes} changes; "
             f"the recording on {s.get('verbatim_bars', 0)} stem-bars of {s.get('bars', '?')} bars"
             + (" (notes on " + ", ".join(f"{st} {100 * v:.0f}%" for st, v in (s.get('note_share') or {}).items()) + ")" if s.get("note_share") else "")
             + f"; vocal phrases {s.get('vocal_phrases', 0)} "
             f"({s.get('vocal_distinct', 0)} distinct); rooms {prog['rooms']}"]
    for vid, pats in prog["patterns"].items():
        seq = prog["sequence"][vid]
        used = {}
        for e in seq:
            used[e["pat"]] = used.get(e["pat"], 0) + 1
        top = sorted(used.items(), key=lambda kv: -kv[1])[:3]
        ex = (prog.get("explain") or {}).get(vid) or {}
        exs = f"; explains {100 * ex['explained']:.0f}%, overshoot {100 * ex['overshoot']:.0f}%" if "explained" in ex else ""
        lines.append(f"  {vid:10s} {prog['voices'][vid]['name']:10s} {len(pats):3d} patterns x{prog.get('units', {}).get(vid, 1)} bars over {len(seq):3d} units; "
                     f"top {', '.join(f'{p}x{k}' for p, k in top)}; ops {sum(len(e['ops']) for e in seq)}{exs}")
    for g in prog.get("pruned") or []:
        what = {"events": f"{g['n']} events DROPPED", "merged": "MERGED", "voice": "PRUNED", "gain": "LEVEL MOVED"}.get(g.get("kind"), "DROPPED")
        lines.append(f"  {g['id']:10s} {what}: {g['why']}" + (f" [{g['gap']}]" if g.get("gap") else ""))
    return "\n".join(lines)
