"""Sound models for a song program: the samples and room a voice plays
with, extracted from the song's own stems.

    kit_sound(inst, ...)     a drum sound as VELOCITY LAYERS (2-3 samples
                             by hit level), each cut WITH its tail - the
                             longest-isolated instance of the layer, its
                             decay extrapolated where the next hit cut
                             it short - plus the sound's decay time
    pitch_sounds(inst, ...)  per-pitch samples with tails for a pitched
                             hit voice (each pitch its own recording)
    room(stem, onsets)       the stem's room: decay time and wet level
                             measured after isolated hits, so the
                             renderer can add it as a send instead of
                             baking it into every sample
    reverb(audio, rt_s, wet) apply that room

Everything here is measured, nothing assumed: a dry kit gets a dry room,
a hall gets a hall.
"""
import numpy as np

from lib.dj import instruments as INS

RATE = INS.RATE
STEPS = INS.STEPS
PRE_S = 0.01                   # cut this much before the onset (the detector marks the flux peak)
TAIL_FLOOR_DB = -36.0          # a sample ends where its level falls this far under its peak (-48 ran hats into whatever rang next)
MAX_SAMPLE_S = 2.5
MAX_EXTEND_S = 0.5             # a truncated tail is extrapolated at most this far (long loops read as extra onsets)
LAYER_MIN_DB = 4.0             # velocity layers must sit this far apart
LAYER_MIN_N = 6


def _mono(a):
    a = np.asarray(a)
    return a.astype(np.float32).mean(axis=1) if a.ndim == 2 else a.astype(np.float32)


def _env_db(y, win_s=0.005):
    w = max(8, int(win_s * RATE))
    m = len(y) // w
    if m < 2:
        return np.full(2, -120.0), w
    e = np.sqrt((y[: m * w].reshape(m, w).astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(e + 1e-9), w


def event_times(inst, result):
    beats = np.asarray(result["beats"], dtype=np.float64)
    return np.array([INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0) for e in inst["events"]])


def stem_onsets(result, stem):
    """Every event time of every instrument of a stem, sorted."""
    ts = []
    for inst in INS.instruments(result):
        if inst["stem"] == stem:
            ts.extend(event_times(inst, result).tolist())
    return np.array(sorted(ts))


def _cut_with_tail(y, t, gap_s, decay_db_s=None):
    """Cut a hit at t: from PRE_S before it until its level has fallen
    TAIL_FLOOR_DB under the peak or the next hit (gap_s) comes. If the
    next hit cut it short, extend the tail by looping its last stretch
    under the fitted (or given) decay. -> (audio, decay dB/s, truncated)."""
    a = max(0, int((t - PRE_S) * RATE))
    b = min(len(y), int((t + min(gap_s, MAX_SAMPLE_S)) * RATE))
    seg = y[a:b].astype(np.float32).copy()
    if len(seg) < int(0.03 * RATE):
        return None, None, True
    env, w = _env_db(seg)
    pk_i = int(np.argmax(env[: max(1, int(0.05 * RATE / w))]))
    pk = float(env[pk_i])
    below = np.where(env[pk_i:] < pk + TAIL_FLOOR_DB)[0]
    if len(below):
        end = min(len(seg), (pk_i + int(below[0]) + 2) * w)
        seg = seg[:end]
        truncated = False
    else:
        truncated = True
    # the decay rate from the second half of what we have
    n = len(env[pk_i:])
    slope = None
    if n >= 8:
        xs = np.arange(n // 2, n) * w / RATE
        ys = env[pk_i + n // 2: pk_i + n]
        if len(xs) >= 4 and ys.std() > 0.5:
            slope = float(np.polyfit(xs, ys, 1)[0])
    if slope is None or slope > -3.0:
        slope = decay_db_s if decay_db_s is not None else -60.0
    if truncated:
        # loop the last 40 ms with crossfades under the exponential decay until the floor
        cur = float(env[-1]) - pk
        need_db = TAIL_FLOOR_DB - cur
        add_s = min(MAX_EXTEND_S, need_db / min(slope, -3.0))
        loop = seg[-int(0.04 * RATE):]
        if add_s > 0.01 and len(loop) > 64:
            xf = len(loop) // 4
            ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
            tail = seg.copy()
            while len(tail) < len(seg) + int(add_s * RATE):
                piece = loop.copy()
                piece[:xf] = tail[-xf:] * (1.0 - ramp) + piece[:xf] * ramp
                tail = np.concatenate([tail[:-xf], piece])
            gain = 10 ** ((np.arange(len(tail) - len(seg)) / RATE * slope) / 20.0).astype(np.float32)
            tail[len(seg):] *= gain
            seg = tail
    fo = min(int(0.02 * RATE), len(seg) // 3)
    seg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    fi = min(int(0.002 * RATE), len(seg) // 8)
    seg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    return seg, slope, truncated


def _layers(vels):
    """Velocity classes: 1-3 clusters of the events' levels (dB under the
    loud hits), at least LAYER_MIN_DB apart and LAYER_MIN_N strong."""
    db = np.array([(v - 1.0) * 24.0 for v in vels])
    if len(db) < 2 * LAYER_MIN_N:
        return [(float(np.median(db)), np.arange(len(db)))]
    best = [(float(np.median(db)), np.arange(len(db)))]
    for k in (2, 3):
        cents = np.percentile(db, np.linspace(15, 85, k))
        for _ in range(12):
            lab = np.argmin(np.abs(db[:, None] - cents[None, :]), axis=1)
            cents = np.array([db[lab == i].mean() if (lab == i).any() else cents[i] for i in range(k)])
        ok = all((lab == i).sum() >= LAYER_MIN_N for i in range(k)) and np.all(np.diff(np.sort(cents)) >= LAYER_MIN_DB)
        if ok:
            best = [(float(cents[i]), np.where(lab == i)[0]) for i in np.argsort(cents)]
    return best


def subtract_others(y, t0, t1, others, gain_db_fn):
    """The stem between t0 and t1 with every OTHER sound's known sample
    subtracted at its event times (the events that fall in or ring into
    the window), each at its event level. others: [(times, vels, layers)]
    with layers as kit_sound() returns them. What remains is the sound
    under study plus whatever the subtraction missed."""
    a, b = int(t0 * RATE), int(min(len(y), t1 * RATE))
    seg = y[a:b].astype(np.float32).copy()
    for times, vels, layers in others:
        if not layers:
            continue
        ldb = np.array([l["db"] for l in layers])
        for t, v in zip(times, vels):
            if t >= t1 or t < t0 - 3.0:
                continue
            db = gain_db_fn(v)
            li = int(np.argmin(np.abs(ldb - db)))
            sample = layers[li]["audio"] * np.float32(10 ** ((db - ldb[li]) / 20.0))
            s0 = int((t - PRE_S) * RATE) - a
            if s0 + len(sample) <= 0 or s0 >= len(seg):
                continue
            lo, hi = max(0, s0), min(len(seg), s0 + len(sample))
            seg[lo:hi] -= sample[lo - s0: hi - s0]
    return seg


def kit_sound(inst, stem_y, result, all_onsets=None, others=None, scale=True):
    """{"layers": [{"db": level under the loud hits, "audio": mono}], "decay_db_s": float,
    "level_dbfs": the loud hits' level}.

    others: the OTHER drum voices' samples ([(times, vels, layers)]); when
    given, every candidate hit is cut from the stem with those sounds
    subtracted at their event times, so a hat that only ever plays under
    the kick comes out as the hat alone, with its whole tail (the next
    hit no longer ends the cut when it is subtracted well)."""
    y = _mono(stem_y)
    ts_all = event_times(inst, result)
    # samples come from the sound's OWN onsets only (conf 1.0); a hit credited from a coincidence
    # (conf < 1) is another sound's recording with this one under it
    own_idx = [i for i, e in enumerate(inst["events"]) if e[5] >= 0.99] or list(range(len(inst["events"])))
    ts = ts_all[own_idx]
    vels = np.array([inst["events"][i][4] for i in own_idx], dtype=np.float64)
    if all_onsets is None:
        all_onsets = stem_onsets(result, inst["stem"])
    own_t = np.sort(ts)
    layers = []
    decay = None
    gain_db = lambda v: (float(v) - 1.0) * 24.0
    for lvl_db, idx in _layers(vels):
        # the best-isolated hit of the layer, preferring loud and well before the next hit of ANY
        # sound; with subtraction available, "next hit" means the next hit of THIS sound
        cands = []
        for i in idx:
            t = float(ts[i])
            ref_t = own_t if others else all_onsets
            k = int(np.searchsorted(ref_t, t + 0.005, side="right"))
            gap = (float(ref_t[k]) - t) if k < len(ref_t) else 2.0
            k2 = int(np.searchsorted(all_onsets, t + 0.005, side="right"))
            gap_any = (float(all_onsets[k2]) - t) if k2 < len(all_onsets) else 2.0
            cands.append((min(gap, 2.0) + 0.3 * float(vels[i]) + 0.2 * min(gap_any, 1.0), t, gap, gap_any))
        cands.sort(reverse=True)
        chosen = None
        for _score, t, gap, gap_any in cands[:4]:
            if others:
                win = subtract_others(y, t - 1.0, t + min(gap, MAX_SAMPLE_S) + 0.05, others, gain_db)
                seg, slope, trunc = _cut_with_tail(win, 1.0, gap, decay)
                # the subtraction may have missed the coincident hit: if a transient of another sound
                # still stands where it was, end the cut there after all
                if seg is not None and gap_any < gap:
                    env, w = _env_db(seg)
                    at = int((PRE_S + gap_any) * RATE / w)
                    if 2 < at < len(env) - 2 and env[at + 1] - env[at - 2] > 6.0:
                        seg, slope, trunc = _cut_with_tail(y, t, gap_any, decay)
            else:
                seg, slope, trunc = _cut_with_tail(y, t, gap_any, decay)
            if seg is not None:
                if decay is None and not trunc and slope is not None:
                    decay = slope
                chosen = (seg, slope, trunc)
                if not trunc:
                    break
        if chosen is None:
            continue
        seg = chosen[0]
        # restore the layer's level: the sample's first 60 ms rms -> level_dbfs + layer offset
        # (scale=False keeps the recording's own level: the program fits event gains to the stem)
        ref = inst.get("level_dbfs") if scale else None
        if ref is not None:
            head = seg[int(PRE_S * RATE): int((PRE_S + 0.06) * RATE)]
            rms = float(np.sqrt(np.mean(head.astype(np.float64) ** 2)) + 1e-9)
            seg = seg * np.float32(min(30.0, 10 ** ((float(ref) + lvl_db) / 20.0) / rms))
        layers.append({"db": lvl_db, "audio": seg})
    if not layers:
        return None
    return {"layers": layers, "decay_db_s": decay if decay is not None else -60.0, "level_dbfs": inst.get("level_dbfs")}


HELD_S = 0.45                  # an event at least this long is a held articulation
MULTI_SAMPLE = True            # per pitch, velocity ZONES (1-3, LAYER_MIN_DB apart), each its own recording at its RECORDED
                               # level; the renderer plays the zone nearest the event and moves it by the difference only.
                               # Before: one recording per pitch, divided by its own event's velocity gain (floor 0.05:
                               # up to +26 dB) and then pushed by the event gain - the "hot voice" and the thin, buzzy
                               # samples the reconstructions were heard to have
ZONE_MIN_N = 3
MAX_LAYERS = 3
SOLO_S = 0.03                  # a per-pitch recording comes from an onset with no other onset within this (see pitch_sounds_multi)
SOLO_WEIGHT = 0.0              # ...preferred by this much in the candidate score. OFF: measured on the render gate 2026-09-09
                               # (true notes through the extracted voices vs the true parts): `other` median 11.9 -> 14.3 dB
                               # (e-piano 11.1 -> 14.5, clav 9.7 -> 10.6) and it flipped the dnb bass to the wrong model -
                               # the solo onsets are the quiet, short ones; isolation AFTER the onset was the better rule


def _zones(dbs):
    """Velocity zones over the events' dB-under-loud values -> [(zone_db, idx array)], loud first."""
    dbs = np.asarray(dbs, dtype=np.float64)
    if len(dbs) < 2 * ZONE_MIN_N:
        return [(float(np.median(dbs)), np.arange(len(dbs)))]
    best = [(float(np.median(dbs)), np.arange(len(dbs)))]
    for k in (2, 3):
        if k > MAX_LAYERS:
            break
        cents = np.percentile(dbs, np.linspace(15, 85, k))
        for _ in range(12):
            lab = np.argmin(np.abs(dbs[:, None] - cents[None, :]), axis=1)
            cents = np.array([dbs[lab == i].mean() if (lab == i).any() else cents[i] for i in range(k)])
        ok = all((lab == i).sum() >= ZONE_MIN_N for i in range(k)) and np.all(np.diff(np.sort(cents)) >= LAYER_MIN_DB)
        if ok:
            best = [(float(cents[i]), np.where(lab == i)[0]) for i in np.argsort(-cents)]
    return best


GROUP_LEVEL_REF = True         # a recording cut at one tone of a chord CONTAINS the chord: its level reference is the
                               # chord's (the voice's simultaneous tones summed in energy), not the tone's. The partner of
                               # polyreader.SPLIT_CHORD_LEVEL (readings v13+: each tone carries 1/n of the chord's energy):
                               # without it the un-velocity gain inflated every sample voice by the split (render gate
                               # 2026-09-09: house piano level +4.4 -> +10.1 dB, rock organ +4.1 -> +14.0, `other` 14.9 -> 16.3)
GROUP_LEVEL_VERSION = 13       # readings older than this carried the whole chord's level on every tone: no summing


def group_vel(vels):
    """The combined velocity of tones sounding together (velocity = 1 + dB/24; energies add)."""
    vels = [float(v) for v in vels]
    if len(vels) <= 1:
        return vels[0] if vels else 1.0
    e = sum(10 ** ((v - 1.0) * 24.0 / 10.0) for v in vels)
    return 1.0 + 10.0 * np.log10(max(e, 1e-12)) / 24.0


def grouped_levels(version):
    """Whether a reading (or program) of this version splits chord levels per tone, so a cut's reference is the group."""
    return GROUP_LEVEL_REF and int(version or 0) >= GROUP_LEVEL_VERSION


def event_group_vels(inst, result):
    """Per event of `inst`: the velocity of its simultaneous group in this voice (same beat and step) when
    the reading splits chord levels, else the event's own velocity."""
    evs = inst["events"]
    if not grouped_levels(result.get("version")):
        return [float(e[4]) for e in evs]
    by_key = {}
    for e in evs:
        if e[2] is not None:
            by_key.setdefault((e[0], e[1]), []).append(float(e[4]))
    gv = {k: group_vel(v) for k, v in by_key.items()}
    return [gv.get((e[0], e[1]), float(e[4])) if e[2] is not None else float(e[4]) for e in evs]


def pick_layer(rec, db):
    """(the recording to play, the gain in dB to apply) for an event at `db` under the voice's loud hits:
    the nearest velocity layer moved by the difference; a recording without layers is the old
    un-velocitied one and takes the whole event gain."""
    layers = rec.get("layers")
    if not layers:
        return rec, db
    li = int(np.argmin([abs(l["db"] - db) for l in layers]))
    return layers[li], db - layers[li]["db"]


MASK_SAMPLES = False           # a per-pitch (or chord) recording keeps only the energy near its own harmonics: in a dense
                               # stem the cut at a note's onset is the whole mix at that moment (the render gate 2026-09-09:
                               # the funk horns 35 dB from the true part with every note present, 6 dB hot). OFF: measured on
                               # the render gate - always masked: rock bass 21.5 -> 9.9 (the guitar's bleed), lead 14.5 ->
                               # 11.3, pluck 16.1 -> 13.6, but slap 7.3 -> 15.1, distorted guitar 6.0 -> 10.0, clav 9.7 ->
                               # 11.2, stab 11.3 -> 14.3, the horns 36 either way (same-pitch overlap); with MASK_MIN_RATIO
                               # (adaptive) the bass median still 9.9 -> 11.2 and `other` unchanged. Medians flat at 11.1
MASK_HARMONICS = 16
MASK_WIDTH = 0.03              # +- this fraction of the harmonic's frequency is kept
MASK_FLOOR_DB = -24.0          # everything else attenuated by this
MASK_MIN_RATIO = 0.6           # the masked cut is kept only when it holds at least this share of the cut's energy: the
                               # instrument was harmonic and what went was bleed (rock bass under the guitar: 21.5 -> 9.9 dB);
                               # under it the instrument IS the broadband part (slap pops, distortion, a clav's click) and
                               # the cut stays whole (slap 7.3 -> 15.1 dB, distorted guitar 6.0 -> 10.0 when masked anyway)


def masked_or_whole(seg, midis):
    """harmonic_mask when the instrument is harmonic enough for the mask to remove only bleed (see MASK_MIN_RATIO)."""
    m = harmonic_mask(seg, midis)
    e_all = float(np.sum(seg.astype(np.float64) ** 2)) + 1e-12
    e_m = float(np.sum(m.astype(np.float64) ** 2))
    return m if e_m / e_all >= MASK_MIN_RATIO else seg


def harmonic_mask(seg, midis, floor_db=None, width=None, n_harm=None):
    """seg (mono) filtered to the harmonics of the given midi notes (an STFT mask, soft edges): what is not
    within `width` of a harmonic of one of the notes is attenuated by floor_db."""
    import librosa
    floor_db = MASK_FLOOR_DB if floor_db is None else floor_db
    width = MASK_WIDTH if width is None else width
    n_harm = MASK_HARMONICS if n_harm is None else n_harm
    if len(seg) < 2048 or not midis:
        return seg
    n_fft, hop = 2048, 256
    S = librosa.stft(np.ascontiguousarray(seg, dtype=np.float32), n_fft=n_fft, hop_length=hop)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / RATE)
    keep = np.zeros(len(freqs), dtype=np.float32)
    for m in set(int(x) for x in midis):
        f0 = 440.0 * 2 ** ((m - 69) / 12.0)
        for h in range(1, n_harm + 1):
            f = h * f0
            if f > RATE / 2:
                break
            w = max(width * f, 1.5 * RATE / n_fft)
            keep = np.maximum(keep, np.exp(-0.5 * ((freqs - f) / (0.5 * w)) ** 2).astype(np.float32))
    floor = 10 ** (floor_db / 20.0)
    gain = floor + (1.0 - floor) * keep
    out = librosa.istft(S * gain[:, None], hop_length=hop, length=len(seg))
    return out.astype(np.float32)


def pitch_sounds(inst, stem_y, result, all_onsets=None, max_pitches=36, period=None):
    if MULTI_SAMPLE:
        return pitch_sounds_multi(inst, stem_y, result, all_onsets=all_onsets, max_pitches=max_pitches, period=period)
    return _pitch_sounds_single(inst, stem_y, result, all_onsets=all_onsets, max_pitches=max_pitches, period=period)


def pitch_sounds_multi(inst, stem_y, result, all_onsets=None, max_pitches=36, period=None):
    """{midi: {"short", "held", "layers": [{"db", "short", "held"}]}}: per pitch,
    per velocity zone, the best-isolated event cut with its whole tail at the
    level it was recorded; "short"/"held" mirror the loudest layer for the
    callers that predate layers."""
    y = _mono(stem_y)
    ts = event_times(inst, result)
    if all_onsets is None:
        all_onsets = stem_onsets(result, inst["stem"])
    period = period or float(result.get("period_s") or 0.5)
    by_pitch = {}
    own_t = np.sort(ts)
    gvels = event_group_vels(inst, result)               # the cut's level reference: what the cut contains (see GROUP_LEVEL_REF)
    for i, ev in enumerate(inst["events"]):
        if ev[2] is None:
            continue
        t = float(ts[i])
        k = int(np.searchsorted(all_onsets, t + 0.005, side="right"))
        gap = (float(all_onsets[k]) - t) if k < len(all_onsets) else 2.0
        length = ev[3] * period / STEPS
        # SOLO: no other note of any sound within SOLO_S of this onset. A per-pitch recording cut at one
        # tone of a chord IS the chord, and a chord rendered from three such recordings stacks three
        # chords (the render gate 2026-09-09: sample voices 4-8 dB hot on polyphonic parts)
        k0 = int(np.searchsorted(all_onsets, t - SOLO_S))
        k1 = int(np.searchsorted(all_onsets, t + SOLO_S))
        solo = (k1 - k0) <= 1
        by_pitch.setdefault(int(ev[2]), []).append((t, gap, length, float(gvels[i]), solo))
    # the pitches the voice plays most first
    order = sorted(by_pitch, key=lambda m: -len(by_pitch[m]))[:max_pitches]
    out = {}
    for midi in order:
        evs = by_pitch[midi]
        dbs = [(v - 1.0) * 24.0 for _t, _g, _l, v, _s in evs]
        layers = []
        for zone_db, idx in _zones(dbs):
            # the best-isolated event of the zone: SOLO first, then the longest clear run after it (any
            # sound), loud ties last
            cands = sorted((SOLO_WEIGHT * float(evs[i][4]) + min(evs[i][1], MAX_SAMPLE_S) + 0.1 * evs[i][3], i) for i in idx)
            layer = None
            for _score, i in cands[::-1][:4]:
                t, gap, length, vel, _solo = evs[i]
                seg, _slope, _tr = _cut_with_tail(y, t, max(gap, 0.08))
                if seg is None or len(seg) < int(0.03 * RATE):
                    continue
                if MASK_SAMPLES:
                    seg = masked_or_whole(seg, [midi])
                layer = {"db": float(zone_db), "short": seg, "held": None, "t": float(t)}
                if length >= HELD_S and gap >= 0.8 * length:
                    a, b = max(0, int((t - PRE_S) * RATE)), int(min(len(y), (t + min(gap, length, MAX_SAMPLE_S)) * RATE))
                    hseg = y[a:b].astype(np.float32).copy()
                    if len(hseg) > int(0.2 * RATE):
                        fo = min(int(0.02 * RATE), len(hseg) // 3)
                        hseg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
                        layer["held"] = masked_or_whole(hseg, [midi]) if MASK_SAMPLES else hseg
                break
            if layer is not None:
                layers.append(layer)
        if not layers:
            continue
        # a held take anywhere in the pitch serves every layer that has none (moved by the level difference)
        with_held = [l for l in layers if l["held"] is not None]
        if with_held:
            src = with_held[0]
            for l in layers:
                if l["held"] is None:
                    l["held"] = src["held"] * np.float32(10 ** ((l["db"] - src["db"]) / 20.0))
        loud = layers[0]
        out[midi] = {"short": loud["short"], "held": loud["held"], "layers": layers, "t": loud["t"]}
    return out


def sample_source_times(pitches):
    """Every onset a pitched voice's recordings were cut at (the chooser must not judge a model on
    the very events its samples are: the sample render IS the stem there)."""
    ts = []
    for rec in (pitches or {}).values():
        if rec.get("t") is not None:
            ts.append(float(rec["t"]))
        for l in rec.get("layers") or ():
            if l.get("t") is not None:
                ts.append(float(l["t"]))
    return np.array(sorted(set(ts)))


def _pitch_sounds_single(inst, stem_y, result, all_onsets=None, max_pitches=36, period=None):
    """{midi: {"short": audio, "held": audio | None}} for a pitched hit
    voice, every recording the song's own: per pitch the loudest,
    best-isolated SHORT event and, where the song has one, the longest
    clean HELD event (sustained by looping its own steady part when the
    renderer needs more)."""
    y = _mono(stem_y)
    ts = event_times(inst, result)
    if all_onsets is None:
        all_onsets = stem_onsets(result, inst["stem"])
    period = period or float(result.get("period_s") or 0.5)
    best, held = {}, {}
    for i, ev in enumerate(inst["events"]):
        if ev[2] is None:
            continue
        t = float(ts[i])
        k = int(np.searchsorted(all_onsets, t + 0.005, side="right"))
        gap = (float(all_onsets[k]) - t) if k < len(all_onsets) else 2.0
        length = ev[3] * period / STEPS
        score = ev[4] + 0.6 * min(gap, 1.0)
        if ev[2] not in best or score > best[ev[2]][0]:
            best[ev[2]] = (score, t, gap, ev[4])
        if length >= HELD_S and gap >= 0.8 * length:
            hs = min(gap, length) + 0.3 * ev[4]
            if ev[2] not in held or hs > held[ev[2]][0]:
                held[ev[2]] = (hs, t, min(gap, length, MAX_SAMPLE_S), ev[4])
    out = {}
    for midi, (_s, t, gap, vel) in sorted(best.items(), key=lambda kv: -kv[1][0])[:max_pitches]:
        seg, _slope, _tr = _cut_with_tail(y, t, max(gap, 0.08))
        if seg is None:
            continue
        # the recording is at the stem's own level, made at velocity `vel`: undo that velocity so the
        # renderer's per-event gain puts every note where it was
        g = 10 ** (((float(vel) - 1.0) * 24.0) / 20.0)
        rec = {"short": seg / np.float32(max(g, 0.05)), "held": None}
        if midi in held:
            _hs, th, length, hv = held[midi]
            a, b = max(0, int((th - PRE_S) * RATE)), int(min(len(y), (th + length) * RATE))
            hseg = y[a:b].astype(np.float32).copy()
            if len(hseg) > int(0.2 * RATE):
                fo = min(int(0.02 * RATE), len(hseg) // 3)
                hseg[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
                gh = 10 ** (((float(hv) - 1.0) * 24.0) / 20.0)
                rec["held"] = hseg / np.float32(max(gh, 0.05))
        out[int(midi)] = rec
    return out


def sustain_sample(rec, n_samples):
    """The recording for an event of n_samples: the held take when the
    song has one (its steady middle looped past its end), else the short."""
    from lib.dj.resynth import looped
    held = rec.get("held")
    if held is not None and n_samples > len(rec["short"]) * 0.8:
        if n_samples <= len(held):
            return held[:n_samples]
        return looped(held, n_samples)
    return rec["short"][:n_samples] if len(rec["short"]) > n_samples else rec["short"]


def room(stem_y, onsets, min_gap_s=0.5, max_hits=60):
    """The stem's room after isolated hits: decay (dB/s -> RT60) and the
    wet share (level 150-400 ms after the hit relative to its peak).
    -> {"rt60_s", "wet_db"} or None when nothing is isolated enough."""
    y = _mono(stem_y)
    onsets = np.asarray(onsets, dtype=np.float64)
    slopes, wets = [], []
    for i in range(len(onsets) - 1):
        t, gap = onsets[i], onsets[i + 1] - onsets[i]
        if gap < min_gap_s:
            continue
        a, b = int(t * RATE), int(min(len(y), (t + min(gap, 1.2)) * RATE))
        seg = y[a:b]
        env, w = _env_db(seg)
        pk_i = int(np.argmax(env[: max(1, int(0.05 * RATE / w))]))
        pk = float(env[pk_i])
        i0, i1 = int(0.15 * RATE / w), int(min(gap, 1.2) * RATE / w) - 2
        if i1 - i0 < 6 or pk < -50:
            continue
        xs = np.arange(i0, i1) * w / RATE
        ys = env[i0:i1]
        if ys.max() < pk - 55:
            continue
        slope, icpt = np.polyfit(xs, ys, 1)
        pred = slope * xs + icpt
        r2 = 1.0 - float(np.sum((ys - pred) ** 2) / max(np.sum((ys - ys.mean()) ** 2), 1e-9))
        # a room DECAYS monotonically and fast enough: a sustained note or a busy stem does not fit
        if -400 < slope < -24 and r2 >= 0.8:
            slopes.append(float(slope))
            wets.append(float(np.mean(env[int(0.15 * RATE / w): int(0.4 * RATE / w)]) - pk))
        if len(slopes) >= max_hits:
            break
    if len(slopes) < 4:
        return None
    rt = float(-60.0 / np.median(slopes))
    wet = float(np.median(wets))
    if rt > 2.5 or wet > -8.0:
        return None                                   # not a room: something else was ringing
    return {"rt60_s": round(rt, 3), "wet_db": round(wet, 1), "n": len(slopes)}


def reverb_ir(rt60_s, seconds=None, seed=0):
    """An exponentially decaying noise impulse response with a gentle
    high-frequency roll-off, RT60 as measured."""
    seconds = seconds or min(3.0, max(0.1, rt60_s * 1.2))
    n = int(seconds * RATE)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32)
    # one-pole lowpass ~4 kHz: rooms darken tails
    a = np.exp(-2 * np.pi * 4000.0 / RATE)
    lp = np.zeros_like(noise)
    acc = 0.0
    for i in range(n):
        acc = a * acc + (1 - a) * noise[i]
        lp[i] = acc
    t = np.arange(n) / RATE
    env = 10 ** ((-60.0 * t / max(rt60_s, 0.05)) / 20.0)
    ir = lp * env.astype(np.float32)
    ir[: int(0.005 * RATE)] = 0.0                    # no direct path in the wet signal
    return ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9)


def reverb(dry, rt60_s, wet_db, ir=None):
    """dry + a wet copy at wet_db (relative), the room measured by room()."""
    from scipy.signal import fftconvolve
    if rt60_s is None or rt60_s <= 0.05:
        return dry
    ir = reverb_ir(rt60_s) if ir is None else ir
    wet = fftconvolve(dry, ir, mode="full")[: len(dry)].astype(np.float32)
    # scale the wet copy so its level 150-400 ms after a typical hit sits wet_db under the dry peak:
    # the IR is unit-energy, so match by the dry/wet rms ratio
    g = 10 ** (wet_db / 20.0) * (np.sqrt(np.mean(dry ** 2)) + 1e-9) / (np.sqrt(np.mean(wet ** 2)) + 1e-9)
    return dry + wet * np.float32(min(g, 4.0))


# --------------------------------------------------------------------------
# Vocal phrases: the stem cut into phrases, repeats found, one recording reused
# --------------------------------------------------------------------------
PHRASE_FLOOR_DB = -32.0        # under the stem's loud level: silence between phrases
PHRASE_GAP_S = 0.35            # a shorter gap does not end a phrase
PHRASE_MIN_S = 0.6
PHRASE_SIM = 0.82              # cosine on (log-mel + chroma) shape for two phrases to be the same (warped reuse)
PHRASE_WARP = False            # reuse near-repeats through a DTW alignment (fewer recordings; measured 2026-09-09 on Sussudio:
                               # 17.9 dB from the stem against 11.3 with exact phrases). OFF since 2026-09-09 at the user's
                               # word - larger samples from the vocal stem are fine: only EXACT repeats (frame-aligned
                               # cosine >= PHRASE_SIM_EXACT, lengths within PHRASE_LEN_TOL) share a recording
PHRASE_SIM_EXACT = 0.95
PHRASE_LEN_TOL = 0.05


def phrases(y, beats, period):
    """[(t0, t1)] phrases of a vocal stem, starts snapped to the 16th grid."""
    y = _mono(y)
    win = int(0.05 * RATE)
    m = len(y) // win
    env = 20 * np.log10(np.sqrt((y[: m * win].reshape(m, win) ** 2).mean(axis=1)) + 1e-9)
    if m < 4:
        return []
    ref = float(np.percentile(env, 95))
    on = env > ref + PHRASE_FLOOR_DB
    out = []
    i = 0
    while i < m:
        if not on[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j < m and gap * win / RATE < PHRASE_GAP_S:
            gap = gap + 1 if not on[j] else 0
            j += 1
        t0, t1 = i * win / RATE, (j - gap) * win / RATE
        if t1 - t0 >= PHRASE_MIN_S:
            pos = INS.beat_step_of(np.asarray(beats), t0, period)
            if pos is not None:
                snapped = INS.event_time(np.asarray(beats), pos[0], pos[1])
                if abs(snapped - t0) < 0.08:
                    t0 = snapped
            out.append((round(t0, 4), round(t1, 4)))
        i = j
    return out


def _phrase_frames(y, t0, t1):
    """Per-frame features of a phrase (log-mel + chroma, ~46 ms frames), unit-normalised per frame."""
    import librosa
    seg = y[int(t0 * RATE):int(t1 * RATE)]
    if len(seg) < 4096:
        return None
    M = librosa.feature.melspectrogram(y=np.ascontiguousarray(seg), sr=RATE, n_fft=2048, hop_length=2048, n_mels=32, power=2.0)
    L = np.clip(10 * np.log10(M + 1e-10) - 10 * np.log10(M.max() + 1e-10), -60, 0) / 60.0
    C = librosa.feature.chroma_stft(y=np.ascontiguousarray(seg), sr=RATE, n_fft=2048, hop_length=2048)
    F = np.concatenate([L, C], axis=0).T
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def _phrase_sim(Fa, Fb):
    """Similarity after DTW alignment: the mean frame cosine along the
    best path (1 = the same phrase, timing aside), 0 when lengths differ
    by more than 25%."""
    import librosa
    if Fa is None or Fb is None:
        return 0.0
    if not PHRASE_WARP:
        # exact repeats only: the same length (within PHRASE_LEN_TOL) and frame-for-frame alike
        n = min(len(Fa), len(Fb))
        if abs(len(Fa) - len(Fb)) > PHRASE_LEN_TOL * max(len(Fa), len(Fb)) or n < 2:
            return 0.0
        return float(np.mean(np.sum(Fa[:n] * Fb[:n], axis=1)))
    if abs(len(Fa) - len(Fb)) > 0.25 * max(len(Fa), len(Fb)):
        return 0.0
    D = 1.0 - Fa @ Fb.T
    _acc, path = librosa.sequence.dtw(C=D, step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]))
    cos = [1.0 - D[i, j] for i, j in path]
    return float(np.mean(cos))


def phrase_library(y, beats, period):
    """Phrases + reuse: every phrase is compared with the earlier distinct
    ones; a repeat (cosine >= PHRASE_SIM, length within 20%) becomes a
    PLACEMENT of the earlier recording. -> (library [(t0, t1)],
    placements [{"phrase": k, "t": start, "gain_db": level vs the recording}])."""
    y = _mono(y)
    ph = phrases(y, beats, period)
    lib, shapes, placements = [], [], []
    for t0, t1 in ph:
        shape = _phrase_frames(y, t0, t1)
        lvl = 20 * np.log10(np.sqrt(np.mean(y[int(t0 * RATE):int(t1 * RATE)] ** 2)) + 1e-9)
        best, best_k = 0.0, None
        if shape is not None:
            for k, (s, (a, b)) in enumerate(zip(shapes, lib)):
                c = _phrase_sim(s, shape)
                if c > best:
                    best, best_k = c, k
        if best_k is not None and best >= (PHRASE_SIM if PHRASE_WARP else PHRASE_SIM_EXACT):
            a, b = lib[best_k]
            ref_lvl = 20 * np.log10(np.sqrt(np.mean(y[int(a * RATE):int(b * RATE)] ** 2)) + 1e-9)
            placements.append({"phrase": best_k, "t": t0, "gain_db": round(float(lvl - ref_lvl), 1), "sim": round(best, 3)})
        else:
            lib.append((t0, t1))
            shapes.append(shape)
            placements.append({"phrase": len(lib) - 1, "t": t0, "gain_db": 0.0, "sim": 1.0})
    return lib, placements


# --------------------------------------------------------------------------
# Chord hits: the song's own chord recordings
# --------------------------------------------------------------------------
def chord_sounds(inst, stem_y, result, all_onsets=None, period=None, max_chords=40):
    """{(root_midi, intervals tuple): audio} - for every distinct chord the
    voice strikes, the best-isolated occurrence cut from the stem (all
    its notes together, as recorded)."""
    y = _mono(stem_y)
    period = period or float(result.get("period_s") or 0.5)
    if all_onsets is None:
        all_onsets = stem_onsets(result, inst["stem"])
    beats = np.asarray(result["beats"], dtype=np.float64)
    groups = {}
    for ev in inst["events"]:
        if ev[2] is None:
            continue
        groups.setdefault((ev[0], ev[1]), []).append(ev)
    best = {}
    for (b, s), evs in groups.items():
        ms = sorted(set(int(e[2]) for e in evs))
        if len(ms) < 2:
            continue
        root = ms[0]
        shape = tuple(m - root for m in ms)
        t = INS.event_time(beats, b, s) + (evs[0][6] if len(evs[0]) > 6 else 0.0)
        k = int(np.searchsorted(all_onsets, t + 0.005, side="right"))
        gap = (float(all_onsets[k]) - t) if k < len(all_onsets) else 2.0
        # the chord recording's level reference is the chord's (its tones summed) on split-level readings
        vel = group_vel([e[4] for e in evs]) if grouped_levels(result.get("version")) else max(e[4] for e in evs)
        length = max(e[3] for e in evs) * period / STEPS
        score = vel + 0.6 * min(gap, 1.0)
        key = (root, shape)
        if key not in best or score > best[key][0]:
            best[key] = (score, t, min(max(gap, 0.1), length + 0.15, MAX_SAMPLE_S), vel)
    out = {}
    for key, (_s, t, length, vel) in sorted(best.items(), key=lambda kv: -kv[1][0])[:max_chords]:
        seg, _slope, _tr = _cut_with_tail(y, t, length)
        if seg is None:
            continue
        if MASK_SAMPLES:
            root, shape = key
            seg = masked_or_whole(seg, [int(root) + int(iv) for iv in shape])
        g = 10 ** (((float(vel) - 1.0) * 24.0) / 20.0)
        out[key] = seg / np.float32(max(g, 0.05))
    return out


def chord_lookup(sounds, root, shape):
    """The recording for (root, shape): exact, else the same shape at the
    nearest root (to be repitched by the difference), else None."""
    if (root, shape) in sounds:
        return sounds[(root, shape)], 0
    cands = [(abs(r - root), r) for (r, sh) in sounds if sh == shape]
    if cands:
        _d, r = min(cands)
        return sounds[(r, shape)], root - r
    return None, 0


# --------------------------------------------------------------------------
# Masked extraction: one sound out of a coincidence, by its own template
# --------------------------------------------------------------------------
_N_FFT, _HOP, _N_MELS = 1024, 256, 48


def _mel_filter():
    import librosa
    return librosa.filters.mel(sr=RATE, n_fft=_N_FFT, n_mels=_N_MELS, fmin=30.0, fmax=16000.0)


def masked_cut(y, t, templates, which, length_s, iters=30):
    """The sound `which` alone, cut from the stem around the hit at t
    (PRE_S before it, length_s after): the stem's STFT is explained by
    the sounds' templates (fixed-template NMF on the window's mel
    magnitude, the reading's 3-block templates collapsed to their first
    block), the Wiener mask of `which` is mapped back to STFT bins and
    the masked STFT inverted. templates: {name: 48*3 floats}."""
    import librosa
    a = max(0, int((t - PRE_S) * RATE))
    b = min(len(y), int((t + length_s) * RATE))
    seg = np.asarray(y[a:b], dtype=np.float32)
    if len(seg) < _N_FFT:
        return None
    S = librosa.stft(seg, n_fft=_N_FFT, hop_length=_HOP)
    mag = np.abs(S)
    fb = _mel_filter()                                  # (48, 513)
    Vmel = fb @ mag                                     # (48, T)
    names = list(templates)
    W = np.array([np.asarray(templates[n], dtype=np.float64)[:_N_MELS] for n in names]).T      # (48, K)
    W = np.maximum(W, 1e-9)
    W = W / W.sum(axis=0, keepdims=True)
    V = np.maximum(Vmel, 1e-9)
    K = W.shape[1]
    H = np.full((K, V.shape[1]), float(V.mean()) / K + 1e-6)
    ones = np.ones_like(V)
    for _ in range(iters):
        WH = np.maximum(W @ H, 1e-9)
        H *= (W.T @ (V / WH)) / np.maximum(W.T @ ones, 1e-9)
    j = names.index(which)
    est = np.maximum(W @ H, 1e-9)
    mask_mel = (W[:, [j]] @ H[[j]]) / est                # (48, T)
    # mel mask -> bin mask (weighted by the filterbank), a floor keeps the sound's transient
    fb_n = fb / np.maximum(fb.sum(axis=0, keepdims=True), 1e-9)
    mask_bin = fb_n.T @ mask_mel                        # (513, T)
    out = librosa.istft(S * mask_bin.astype(np.float32), hop_length=_HOP, length=len(seg))
    return out.astype(np.float32)


def fit_levels(y, voices, win_s=0.06):
    """Per-event gains fitted to the stem: at every event, the stem's energy in
    a 60 ms window is explained by the sounds that start in that window
    (NNLS with each sound's sample energy over the same window) -> for each
    voice, {event index: gain (linear, 1 = the sample as extracted)}.
    voices: {name: {"times": [...], "sample": mono at nominal level}}."""
    from scipy.optimize import nnls
    y = np.asarray(y, dtype=np.float32)
    w = int(win_s * RATE)
    all_ev = []                                          # (t, name, idx)
    for name, v in voices.items():
        for i, t in enumerate(v["times"]):
            all_ev.append((float(t), name, i))
    all_ev.sort()
    gains = {name: {} for name in voices}
    i = 0
    n = len(all_ev)
    while i < n:
        t0 = all_ev[i][0]
        group = [all_ev[i]]
        j = i + 1
        while j < n and all_ev[j][0] - t0 < win_s * 0.5:
            group.append(all_ev[j]); j += 1
        a = int(t0 * RATE); bwin = min(len(y), a + w)
        target = y[a:bwin].astype(np.float64)
        if bwin - a < w // 2:
            i = j; continue
        cols = []
        for (t, name, k) in group:
            s = voices[name]["sample"]
            off = int((t - t0) * RATE) + int(PRE_S * RATE)
            col = np.zeros(bwin - a)
            seg = s[off:off + (bwin - a)] if off < len(s) else np.zeros(0)
            col[: len(seg)] = seg[: bwin - a]
            cols.append(col)
        A = np.array(cols).T
        # energy-domain fit: least squares on the waveform (phases of the extracted sample match the
        # stem's at its own hit, less so elsewhere) - fall back to an rms ratio when the fit degenerates
        try:
            g, _ = nnls(A, target)
        except Exception:  # noqa: BLE001
            g = np.zeros(len(group))
        for (t, name, k), gk in zip(group, g):
            s_rms = float(np.sqrt(np.mean(A[:, group.index((t, name, k))] ** 2)) + 1e-9)
            t_rms = float(np.sqrt(np.mean(target ** 2)) + 1e-9)
            if gk <= 1e-4 or gk > 30:
                gk = min(30.0, t_rms / s_rms / max(len(group), 1) ** 0.5)
            gains[name][k] = float(gk)
        i = j
    return gains
