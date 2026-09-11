"""Play a song back from its instrument reading: notes + the song's own
samples, programmatically.

The instrument pass (lib/dj/instruments.py) leaves, per track, the
sounds the song is built from and what each plays per beat. This module
turns that back into audio without the original mix:

    render(result, stems, ids)      every event of the chosen instruments
                                    played by that instrument's EXEMPLAR
                                    cut from the stem - repitched for a
                                    pitched sound, looped for a held one,
                                    at the event's level - on the DJ grid
    export_gen(result, stems, ...)  the same material in the generative
                                    console's SongScript form (kit
                                    one-shots, note banks, pad sample,
                                    per-bar drum grids, melody and bass
                                    lines, sections), so lib/gen/script
                                    can recreate and vary the song

render() is what the Analysis tab's "play selected only" plays: it is
the reconstruction, not a separation - hearing it against the stem is
the honest check of the reading (a spurious note is a wrong note here).

Levels: exemplars are used at their ORIGINAL level in the stem (not
normalised), and an event's velocity is the dB it sat under the
instrument's loud hits, so the reconstruction lands at the song's own
balance. stems: {name: (n,) or (n,2) float array @44100}.
"""
import os

import numpy as np

from lib.dj import instruments as INS

RATE = INS.RATE
STEPS = INS.STEPS
RELEASE_S = 0.12               # tail added past an event's measured length
PRE_S = 0.01                   # a sample is cut this much BEFORE its detected onset (the detector marks the
                               # flux peak, ~10 ms into the attack) and placed that much earlier: the attack
                               # survives and the replay lands on the stem's own transients (measured +12 ms late before)
XFADE_S = 0.02                 # loop crossfade for held sounds
MAX_HIT_S = 3.0
GEN_DRUM_SLOTS = ("kick", "snare", "hat", "ohat", "shaker", "perc", "rim", "tom", "ride")


def _mono(a):
    a = np.asarray(a)
    return a.astype(np.float32).mean(axis=1) if a.ndim == 2 else a.astype(np.float32)


def exemplar_audio(inst, stems):
    """The instrument's exemplar cut from its stem, mono float32 at the
    stem's own level (None without audio)."""
    y = stems.get(inst.get("stem"))
    if y is None or not inst.get("exemplar"):
        return None
    y = _mono(y)
    a, b = inst["exemplar"]
    a, b = int(a * RATE) - (int(PRE_S * RATE) if inst["kind"] == "hit" else 0), int(min(len(y), b * RATE))
    a = max(0, a)
    if b - a < 64:
        return None
    seg = y[a:b].copy()
    # the exemplar is the most ISOLATED hit, not the loudest: put it at the level the
    # instrument's loud hits had (what every event's velocity is measured against)
    ref = inst.get("level_dbfs")
    if ref is not None:
        head = seg[int(PRE_S * RATE): int((PRE_S + 0.06) * RATE)] if inst["kind"] == "hit" else seg
        rms = float(np.sqrt(np.mean(head.astype(np.float64) ** 2)) + 1e-9)
        seg *= np.float32(min(30.0, 10 ** (float(ref) / 20.0) / rms))
    fi = min(int(0.003 * RATE), len(seg) // 4)
    seg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    fo = min(int(0.02 * RATE), len(seg) // 3)
    seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return seg


def repitch(seg, semitones):
    """Resample: +12 = an octave up (shorter), the sample's character kept."""
    if abs(semitones) < 1e-6:
        return seg
    rate = 2.0 ** (semitones / 12.0)
    n_out = int(len(seg) / rate)
    if n_out < 16:
        return seg[:16]
    xs = np.arange(n_out) * rate
    return np.interp(xs, np.arange(len(seg)), seg).astype(np.float32)


def looped(seg, n_samples):
    """seg extended to n_samples by looping its middle with crossfades."""
    if len(seg) >= n_samples:
        return seg[:n_samples]
    xf = min(int(XFADE_S * RATE), len(seg) // 4)
    if xf < 8:
        return np.resize(seg, n_samples)
    body = seg[xf:-xf] if len(seg) > 4 * xf else seg
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
    out = seg.copy()
    while len(out) < n_samples + xf:
        piece = body.copy()
        piece[:xf] = out[-xf:] * (1.0 - ramp) + piece[:xf] * ramp
        out = np.concatenate([out[:-xf], piece])
    return out[:n_samples]


def _event_gain(inst, ev):
    vel = float(ev[4])
    if inst["kind"] == "sustain":
        return 10 ** (((vel - 1.0) * 30.0) / 20.0)
    return 10 ** (((vel - 1.0) * 24.0) / 20.0)


def _event_t(beats, ev):
    """The event's real onset: its grid step plus the offset the reading kept."""
    t = INS.event_time(beats, ev[0], ev[1])
    return t + (float(ev[6]) if len(ev) > 6 else 0.0)


def pitch_samples(inst, stems, result, max_pitches=32):
    """A sample PER PITCH for a pitched hit instrument: the loudest,
    best-isolated event of each pitch cut from the stem at its own
    level - so a note plays its own recording, not a repitched copy of
    one exemplar. {midi: mono float32}; pitches without one fall back
    to the repitched exemplar in render()."""
    y = stems.get(inst["stem"])
    if y is None or not inst.get("pitched"):
        return {}
    y = _mono(y)
    beats = np.asarray(result["beats"], dtype=np.float64)
    period = float(result.get("period_s") or 0.5)
    all_t = np.array(sorted(_event_t(beats, e) for i in INS.instruments(result) if i["stem"] == inst["stem"] for e in i["events"]))
    best = {}
    for ev in inst["events"]:
        if ev[2] is None:
            continue
        t = _event_t(beats, ev)
        k = int(np.searchsorted(all_t, t + 0.005, side="right"))
        nxt = float(all_t[k]) if k < len(all_t) else t + 2.0
        gap = max(0.0, nxt - t)
        score = ev[4] + 0.6 * min(gap, 1.0)
        if ev[2] not in best or score > best[ev[2]][0]:
            best[ev[2]] = (score, t, min(max(gap, 0.08), ev[3] * period / STEPS + RELEASE_S, MAX_HIT_S), ev[4])
    out = {}
    for midi, (_s, t, length, vel) in sorted(best.items(), key=lambda kv: -kv[1][0])[:max_pitches]:
        a, b = max(0, int((t - PRE_S) * RATE)), int(min(len(y), (t + length) * RATE))
        if b - a < 256:
            continue
        seg = y[a:b].copy()
        fi = min(int(0.002 * RATE), len(seg) // 4)
        seg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
        fo = min(int(0.02 * RATE), len(seg) // 3)
        seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
        # the sample was recorded at THAT event's velocity: undo it, so the event gain re-applies it
        g = _event_gain(inst, [0, 0, midi, 1, vel, 1.0])
        out[int(midi)] = seg / np.float32(max(g, 0.05))
    return out


def render(result, stems, ids=None, t0=None, t1=None, progress=None, use_stems=("vocals",), per_pitch=True):
    """(n,2) float32 covering [t0, t1) (the whole track by default):
    the chosen instruments' events played by their samples. Stems named
    in use_stems are mixed in AS AUDIO instead of being re-synthesised
    (vocals: no sample-and-notes model sings) whenever any of their
    instruments is chosen."""
    beats = np.asarray(result["beats"], dtype=np.float64)
    period = float(result.get("period_s") or np.median(np.diff(beats)))
    insts = [i for i in INS.instruments(result) if ids is None or i["id"] in ids]
    if t0 is None:
        t0 = 0.0
    if t1 is None:
        t1 = float(max(len(_mono(y)) / RATE for y in stems.values())) if stems else float(beats[-1] + period)
    n = int(max(0.0, t1 - t0) * RATE) + 1
    out = np.zeros(n, dtype=np.float32)
    for stem in use_stems or ():
        if any(i["stem"] == stem for i in insts) and stems.get(stem) is not None:
            y = _mono(stems[stem])
            a, b = int(t0 * RATE), int(min(len(y), t1 * RATE))
            if b > a:
                out[: b - a] += y[a:b]
            insts = [i for i in insts if i["stem"] != stem]
    for q, inst in enumerate(insts):
        ex = exemplar_audio(inst, stems)
        if ex is None:
            continue
        base = inst.get("exemplar_midi")
        sustain = inst["kind"] == "sustain"
        samples = pitch_samples(inst, stems, result) if (per_pitch and inst["kind"] == "hit" and inst.get("pitched")) else {}
        cache = {}
        events = inst["events"]
        if sustain:
            # a held CHORD is one sample: the exemplar (a chord itself) moved so its lowest note lands
            # on the group's lowest note, for the group's longest hold - not one copy per note
            groups = {}
            for ev in events:
                groups.setdefault((ev[0], ev[1]), []).append(ev)
            events = []
            for key in sorted(groups):
                g = groups[key]
                lo = min(e[2] for e in g if e[2] is not None) if any(e[2] is not None for e in g) else None
                events.append([key[0], key[1], lo, max(e[3] for e in g), max(e[4] for e in g), g[0][5]] + ([g[0][6]] if len(g[0]) > 6 else []))
        for ev in events:
            t = _event_t(beats, ev)
            if t < t0 - 4.0 or t >= t1:
                continue
            length = ev[3] * period / STEPS
            midi = ev[2]
            if midi is not None and int(midi) in samples:
                sample = samples[int(midi)]
            elif inst.get("pitched") and midi is not None and base is not None:
                key = int(midi) - int(base)
                if key not in cache:
                    cache[key] = repitch(ex, key)
                sample = cache[key]
            else:
                sample = ex
            if sustain:
                want = int((length + RELEASE_S) * RATE)
                sample = looped(sample, want)
            else:
                want = int(min(MAX_HIT_S, length + RELEASE_S) * RATE)
                sample = sample[:want] if len(sample) > want else sample
            if len(sample) < 8:
                continue
            piece = sample * np.float32(_event_gain(inst, ev))
            fo = min(int(0.03 * RATE), len(piece) // 3)
            if fo > 0:
                piece = piece.copy()
                piece[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
            a = int((t - t0 - (PRE_S if not sustain else 0.0)) * RATE)
            if a < 0:
                piece = piece[-a:]
                a = 0
            b = min(n, a + len(piece))
            if b > a:
                out[a:b] += piece[: b - a]
        if progress:
            progress(q + 1, len(insts), inst["id"])
    peak = float(np.abs(out).max())
    if peak > 0.98:
        out *= 0.98 / peak
    return np.stack([out, out], axis=1)


# --------------------------------------------------------------------------
# The generative console's SongScript
# --------------------------------------------------------------------------

def _write_wav(path, seg, peak=None):
    import soundfile as sf
    seg = np.asarray(seg, dtype=np.float32)
    if peak is not None:
        m = float(np.abs(seg).max())
        if m > 1e-6:
            seg = seg / m * peak
    sf.write(path, seg, RATE, subtype="PCM_16")
    return path


def _bar_of(result, beat):
    return (int(beat) - int(result.get("down0", 0))) // 4


def _step16(result, beat, step):
    return ((int(beat) - int(result.get("down0", 0))) % 4) * STEPS + int(step)


def _note_bank(inst, stems, result, out_dir, prefix, max_pitches=24):
    """Per-pitch samples for a pitched hit instrument: the loudest,
    best-isolated event of each pitch cut from the stem."""
    y = stems.get(inst["stem"])
    if y is None:
        return []
    y = _mono(y)
    beats = np.asarray(result["beats"], dtype=np.float64)
    period = float(result.get("period_s") or 0.5)
    all_t = np.array(sorted(INS.event_time(beats, e[0], e[1]) for i in INS.instruments(result)
                            if i["stem"] == inst["stem"] for e in i["events"]))
    best = {}
    for ev in inst["events"]:
        if ev[2] is None:
            continue
        t = INS.event_time(beats, ev[0], ev[1])
        k = int(np.searchsorted(all_t, t, side="right"))
        nxt = float(all_t[k]) if k < len(all_t) else t + 2.0
        gap = max(0.0, nxt - t)
        score = ev[4] + 0.5 * min(gap, 1.0)
        if ev[2] not in best or score > best[ev[2]][0]:
            best[ev[2]] = (score, t, min(gap, ev[3] * period / STEPS + RELEASE_S, MAX_HIT_S))
    top = sorted(best.items(), key=lambda kv: -kv[1][0])[:max_pitches]
    bank = []
    for midi, (_s, t, length) in sorted(top):
        a, b = int(t * RATE), int(min(len(y), (t + max(0.12, length)) * RATE))
        if b - a < 256:
            continue
        seg = y[a:b].copy()
        fo = min(int(0.02 * RATE), len(seg) // 3)
        seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
        path = _write_wav(os.path.join(out_dir, f"{prefix}_{int(midi)}.wav"), seg, peak=0.8)
        bank.append({"file": path, "base_midi": int(midi)})
    return bank


def _chords_for_section(chords, key_text, b0, b1):
    """The chord track's bars b0..b1 as gen chord entries: scale degree of the
    root in the song's key (the nearest degree when the root is outside the
    scale) with the third's quality."""
    if not chords:
        return None
    from lib.gen.theory import parse_key
    try:
        key = parse_key(key_text)
    except Exception:  # noqa: BLE001
        return None
    pcs = [key.degree_pc(d) for d in range(7)]
    out = []
    for c in chords:
        if not (b0 <= c["bar"] < b1):
            continue
        root = int(c["root"])
        deg = min(range(7), key=lambda d: min((root - pcs[d]) % 12, (pcs[d] - root) % 12))
        q = c["quality"]
        entry = {"deg": deg}
        if q.startswith("min"):
            entry["third"] = "min"
        elif q in ("maj", "dom7"):
            entry["third"] = "maj"
        elif q == "sus":
            entry["sus"] = 4
        out.append(entry if len(entry) > 1 else deg)
    return out or None


def export_gen(result, stems, out_dir, title="song", bpm=None, key="8A", sections=None, progress=None, chords=None):
    """Write the reading as gen material + script.yaml under out_dir.
    sections: the DJ track's [{start_s, end_s, kind, energy}] (one
    section over the whole track when None). Returns the script path."""
    from lib.gen import script as S
    os.makedirs(out_dir, exist_ok=True)
    beats = np.asarray(result["beats"], dtype=np.float64)
    period = float(result.get("period_s") or np.median(np.diff(beats)))
    bpm = float(bpm or 60.0 / period)
    bar_len = 4.0 * period
    down0 = int(result.get("down0", 0))
    n_bars = max(1, (len(beats) - down0) // 4)
    insts = INS.instruments(result)
    by_stem = {s: [i for i in insts if i["stem"] == s] for s in INS.STEM_ORDER}
    # -- sections on the bar grid
    kinds = {"intro": "intro", "outro": "outro", "groove": "groove", "build": "build", "breakdown": "break", "break": "break",
             "drop": "drop", "verse": "groove", "chorus": "drop"}
    secs = []
    if sections:
        for s in sections:
            b0 = int(round((float(s["start_s"]) - beats[down0]) / bar_len))
            b1 = int(round((float(s["end_s"]) - beats[down0]) / bar_len))
            if b1 - b0 >= 2:
                secs.append({"section": kinds.get(str(s.get("kind", "groove")), "groove"), "bar0": max(0, b0),
                             "bars": b1 - b0, "energy": float(s.get("energy") or 0.6)})
    if not secs:
        secs = [{"section": "groove", "bar0": 0, "bars": n_bars, "energy": 0.7}]
    # -- kit: the drum sounds by slot
    kit, kit_db = {}, {}
    drums = sorted(by_stem["drums"], key=lambda i: -i["n"])
    for inst in drums:
        slot = inst.get("hint")
        if slot not in GEN_DRUM_SLOTS or slot in kit:
            continue
        ex = exemplar_audio(inst, stems)
        if ex is None:
            continue
        kit[slot] = _write_wav(os.path.join(out_dir, f"{slot}_song.wav"), ex, peak=0.9)
        kit_db[slot] = float(inst.get("level_db") or 0.0)
        inst["_slot"] = slot
    # -- melodic banks: lead = the busiest pitched hit sound of `other`, keys = the second
    pitched_other = sorted([i for i in by_stem["other"] if i["kind"] == "hit" and i.get("pitched")], key=lambda i: -i["n"])
    bank = _note_bank(pitched_other[0], stems, result, out_dir, "pluck0") if pitched_other else []
    bank_keys = _note_bank(pitched_other[1], stems, result, out_dir, "pluck1") if len(pitched_other) > 1 else []
    pitched_bass = sorted([i for i in by_stem["bass"] if i["kind"] == "hit" and i.get("pitched")], key=lambda i: -i["n"])
    bass_bank = _note_bank(pitched_bass[0], stems, result, out_dir, "bass") if pitched_bass else []
    # -- pad: the largest held sound of `other`
    pad = None
    held = sorted([i for i in by_stem["other"] if i["kind"] == "sustain"], key=lambda i: -i["n"])
    if held:
        ex = exemplar_audio(held[0], stems)
        if ex is not None and held[0].get("exemplar_midi") is not None:
            pad = {"file": _write_wav(os.path.join(out_dir, "pad_song.wav"), ex, peak=0.8),
                   "base_midi": int(held[0]["exemplar_midi"]), "seconds": round(len(ex) / RATE, 2)}
    if progress:
        progress("samples written")
    # -- per section: drum bars, lines, layers
    def line_of(inst, b0, b1):
        out = []
        for ev in inst["events"]:
            bar = _bar_of(result, ev[0])
            if b0 <= bar < b1 and ev[2] is not None:
                out.append([bar - b0, _step16(result, ev[0], ev[1]), int(ev[2]), int(ev[3]), float(ev[4])])
        return out
    sc_sections = []
    for s in secs:
        b0, b1 = s["bar0"], s["bar0"] + s["bars"]
        e = {"section": s["section"], "bars": s["bars"], "energy": round(min(1.0, s["energy"]), 2)}
        bars = [{} for _ in range(s["bars"])]
        for inst in drums:
            slot = inst.get("_slot")
            if not slot:
                continue
            for ev in inst["events"]:
                bar = _bar_of(result, ev[0])
                if b0 <= bar < b1:
                    bars[bar - b0].setdefault(slot, []).append([_step16(result, ev[0], ev[1]), round(float(ev[4]), 2)])
        if any(bars):
            e["drums_bars"] = bars
        layers = {slot for b in bars for slot in b}
        if pitched_other:
            e["melody"] = line_of(pitched_other[0], b0, b1)
            if e["melody"]:
                layers.add("lead")
        if pitched_bass:
            e["bass_line"] = line_of(pitched_bass[0], b0, b1)
            if e["bass_line"]:
                layers.add("bass")
        if pad and any(b0 <= _bar_of(result, ev[0]) < b1 for ev in held[0]["events"]):
            layers.add("pad")
        e["layers"] = sorted(layers)
        ch = _chords_for_section(chords, key, b0, b1)
        if ch:
            e["chords"] = ch
        sc_sections.append(e)
    for inst in drums:
        inst.pop("_slot", None)
    script = {"title": title, "style": "groove", "bpm": round(bpm, 2), "key": key, "kit": kit or None, "kit_db": kit_db,
              "bank": bank, "bank_keys": bank_keys, "bass_bank": bass_bank, "pad": pad, "fidelity": 0.0,
              "sections": sc_sections}
    path = S.save(S.normalize(script), os.path.join(out_dir, "script.yaml"))
    if progress:
        progress(f"script: {len(sc_sections)} sections, kit {list(kit)}, {len(bank)} lead notes, {len(bass_bank)} bass notes")
    return path


def load_stems_mono(music_root, track_id):
    """{stem: mono float32} decoded from disk."""
    from lib.dj.stems import stem_paths
    from lib.dj.features import decode_file_stereo
    paths = stem_paths(music_root, track_id)
    if paths is None:
        return None
    out = {n: _mono(decode_file_stereo(p)) for n, p in paths.items()}
    if "bass" in out and "drums" in out:
        out["bass"], _g = INS.clean_bass(out["bass"], out["drums"])      # the reading was made on the cleaned bass
    return out
