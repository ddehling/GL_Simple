"""The bass stem read as ONE monophonic line by pitch tracking, instead of
onset clustering followed by a pitch guess per onset.

Why (2026-09-09, the truth set): the onset-first reader scores 1.00 on
the additive synthetic gate and 0.04-0.19 (notes, 60 ms + pitch) against
an independent transcription on real bass stems; the disagreement is in
timing first and pitch second. A bass line is monophonic: a pitch track
over the whole stem gives every note's pitch from its whole length, and
its boundaries from pitch changes and energy onsets (repeated notes).

    read(y, beats, period) -> [instrument] in the reading's format

The line becomes one pitched hit voice ("bass") with events
[beat, step, midi, dur_steps, vel, conf, offset_s]; held notes are simply
long notes (no separate sustain voice). The sample models, the pattern
language and the closed-loop check are unchanged.
"""
import numpy as np

from lib.dj import instruments as INS

RATE = INS.RATE
STEPS = INS.STEPS
FMIN, FMAX = 30.0, 420.0
FRAME, HOP = 4096, 512
MIN_NOTE_S = 0.05
GAP_S = 0.04                   # an unvoiced gap this long ends the note
ONSET_DELTA = 0.12             # onset-strength peak threshold (relative to its max) for a repeated note
ONSET_MIN_S = 0.07             # a new onset closer than this to the note start is the same note
VEL_RANGE_DB = 24.0


DECIMATE = 4                   # pyin on the stem decimated by this (11 kHz for a line under 420 Hz): the same frames in
                               # time (frame and hop scale with it), a quarter of the samples; same notes (measured), but
                               # pyin's cost is its Viterbi decode, not the frames, so this alone changed nothing
PYIN_RESOLUTION = 0.15         # pitch states per semitone = 1/this (librosa's default 0.1 = ten; the reader rounds to
                               # whole semitones): the decode is 49 of the reader's 51 s on a 100 s stem at 0.1. Truth set
                               # 2026-09-09: 0.15 -> the same notes (median 0.87, funk 0.87) in half the time (14-26 s a
                               # song for 28-49); 0.2 -> funk 0.85 in a third; 0.25 -> funk 0.83 (the slap pops go) in a sixth


def _f0_track(y, frame=None, hop=None, fmin=None):
    import librosa
    frame, hop, fmin = frame or FRAME, hop or HOP, fmin or FMIN
    y = np.ascontiguousarray(y, dtype=np.float32)
    sr = RATE
    if DECIMATE > 1 and frame % DECIMATE == 0 and hop % DECIMATE == 0 and FMAX * 2.2 < RATE / DECIMATE:
        from scipy.signal import resample_poly
        y = resample_poly(y, 1, DECIMATE).astype(np.float32)
        sr, frame, hop = RATE // DECIMATE, frame // DECIMATE, hop // DECIMATE
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=FMAX, sr=sr, frame_length=frame, hop_length=hop, fill_na=np.nan,
                                   resolution=PYIN_RESOLUTION)
    midi = np.full(len(f0), np.nan)
    ok = np.isfinite(f0) & voiced
    midi[ok] = 69.0 + 12.0 * np.log2(f0[ok] / 440.0)
    return midi, prob


FINE_FRAME, FINE_HOP_PY, FINE_FMIN = 2048, 256, 70.0    # the second, faster pitch pass (above 70 Hz)
FINE_SPLIT_SEMIS = 5           # a stretch of the fine track this far from the coarse note...
FINE_SPLIT_S = 0.05            # ...for this long is a note of its own inside it (a fast octave pop)


def _fine_splits(notes, y):
    """Two resolutions: the 93 ms frame that the low notes need merges a fast
    octave pop into the root note (pop truth track: recall 0.59). A 46 ms
    pass above 70 Hz finds, inside each coarse note, stretches at another
    pitch and carves them out."""
    fm, _p = _f0_track(y, FINE_FRAME, FINE_HOP_PY, FINE_FMIN)
    t_fine = np.arange(len(fm)) * FINE_HOP_PY / RATE
    out = []
    min_frames = max(2, int(FINE_SPLIT_S * RATE / FINE_HOP_PY))
    for t0, t1, m, r in notes:
        a, b = np.searchsorted(t_fine, t0), np.searchsorted(t_fine, t1)
        seg = fm[a:b]
        if len(seg) < 2 * min_frames:
            out.append((t0, t1, m, r))
            continue
        far = np.isfinite(seg) & (np.abs(seg - m) >= FINE_SPLIT_SEMIS)
        # contiguous far stretches
        pieces, k = [], 0
        while k < len(far):
            if far[k]:
                j = k
                while j < len(far) and far[j]:
                    j += 1
                if j - k >= min_frames:
                    pieces.append((k, j))
                k = j
            else:
                k += 1
        if not pieces:
            out.append((t0, t1, m, r))
            continue
        cur = t0
        for k, j in pieces:
            ta, tb = t_fine[a + k], t_fine[min(a + j, len(t_fine) - 1)]
            if ta - cur >= MIN_NOTE_S:
                out.append((cur, ta, m, r))
            mm = int(round(float(np.median(seg[k:j]))))
            if tb - ta >= MIN_NOTE_S:
                out.append((ta, tb, mm, r))
            cur = tb
        if t1 - cur >= MIN_NOTE_S:
            out.append((cur, t1, m, r))
    return sorted(out)


def _onsets(y):
    import librosa
    env = librosa.onset.onset_strength(y=np.ascontiguousarray(y, dtype=np.float32), sr=RATE, hop_length=HOP, aggregate=np.median)
    if env.max() <= 0:
        return np.zeros(0)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=RATE, hop_length=HOP, units="frames", backtrack=False,
                                    delta=ONSET_DELTA * float(env.max()), wait=int(ONSET_MIN_S * RATE / HOP))
    return np.asarray(fr) * HOP / RATE


FINE_HOP = 256
RISE_DB = 6.0                  # a rise of this much within RISE_S after a dip, inside one pitch, is a repeated note
RISE_S = 0.04
DIP_DB = 3.0                   # ...after the envelope fell at least this much against the previous DIP_S
DIP_S = 0.12
REFINE_S = 0.08                # a note's start is moved to the steepest rise of the fine envelope within this
OUTLIER_SEMIS = 7              # a short note this far from BOTH neighbours is a tracking slip...
OUTLIER_MAX_S = 0.15           # ...when shorter than this
FLOOR_DB = -36.0               # a note this far under the line's loud notes is not one
FINE_PASS = False              # the second, faster pitch pass (_fine_splits): measured WORSE on the truth set (median 0.84 -> 0.67,
                               # house 0.93 -> 0.22 - the 46 ms pass fragments sub-bass notes); the pop track's fast octave pops stay open
OCTAVE_CHECK = False           # per-note pitch re-check (INS.verify_pitches): measured neutral on the truth set (0.84 -> 0.84)
OCTAVE_ODD = True              # per-note OCTAVE test on the odd harmonics (_octave_fix): pyin tracks a slap pop an octave
                               # low (funk truth track: 31 of 36 pops read at the root) and a D2 line as D1 (28 notes at
                               # midi 26); the fundamental of the read pitch is then absent and so are its odd harmonics
OCTAVE_ODD_RATIO = 0.55        # odd harmonics (1, 3, 5) under this share of the even ones (2, 4, 6): the note is the octave up
                               # (truth set: 0.3 -> funk 0.76, 0.55 -> 0.88, 0.7 -> 0.88, 0.85 -> 0.81 with rock and pop slipping;
                               # the other four songs do not move)
OCTAVE_DOWN_RATIO = 0.5        # the octave BELOW's odd harmonics (f0/2, 3f0/2, 5f0/2) at this share of the read pitch's
                               # first three: the note is the octave down (pyin on a strong second harmonic)
OCTAVE_DOWN = False


def _octave_fix(notes, y, lo, hi):
    """Each note's pitch checked one octave either way against the stem's
    spectrum over the note itself (20 ms in, at most 0.4 s). -> (notes, n_changed)"""
    out = []
    changed = 0
    for t0, t1, m, r in notes:
        a, b = int((t0 + 0.02) * RATE), int(min(len(y), min(t1, t0 + 0.4) * RATE))
        if b - a < 1024:
            out.append((t0, t1, m, r))
            continue
        R, df = INS._spectrum(y[a:b], INS._N_HPS)
        from scipy.ndimage import maximum_filter1d
        Rm = maximum_filter1d(R, 5)
        pk = lambda f: float(Rm[int(round(f / df))]) if 0 < int(round(f / df)) < len(Rm) else 0.0
        f0 = 440.0 * 2 ** ((m - 69) / 12.0)
        odd = pk(f0) + pk(3 * f0) + pk(5 * f0)
        even = pk(2 * f0) + pk(4 * f0) + pk(6 * f0)
        if odd < OCTAVE_ODD_RATIO * even and m + 12 <= hi:
            m += 12
            changed += 1
        elif OCTAVE_DOWN and m - 12 >= lo:
            below = pk(f0 / 2) + pk(1.5 * f0) + pk(2.5 * f0)
            if below >= OCTAVE_DOWN_RATIO * (pk(f0) + pk(2 * f0) + pk(3 * f0)):
                m -= 12
                changed += 1
        out.append((t0, t1, m, r))
    return out, changed


def _fine_env(y):
    """log-rms envelope at FINE_HOP (5.8 ms) -> (times, dB)"""
    n = len(y) // FINE_HOP
    frames = y[: n * FINE_HOP].astype(np.float64).reshape(n, FINE_HOP)
    win = 4
    e = np.sqrt(np.convolve(np.mean(frames ** 2, axis=1), np.ones(win) / win, mode="same"))
    return np.arange(n) * FINE_HOP / RATE, 20 * np.log10(e + 1e-9)


def _refine_start(t, env_t, env_db):
    """The steepest rise of the envelope within +-REFINE_S of t."""
    a, b = np.searchsorted(env_t, t - REFINE_S), np.searchsorted(env_t, t + REFINE_S)
    if b - a < 3:
        return t
    d = np.diff(env_db[a:b])
    k = int(np.argmax(d))
    return float(env_t[a + k])


def _rises(env_t, env_db, ta, tb):
    """Times inside (ta, tb) where the envelope rises RISE_DB within RISE_S after a dip: re-attacks."""
    a, b = np.searchsorted(env_t, ta), np.searchsorted(env_t, tb)
    out = []
    w = max(1, int(RISE_S * RATE / FINE_HOP))
    back = max(1, int(DIP_S * RATE / FINE_HOP))
    k = a + 1
    while k < b - w:
        seg = env_db[k: k + w + 1]
        # a re-attack: the envelope FELL (a dip of DIP_DB against the last DIP_S) and then rises RISE_DB;
        # a rise alone is a swell or the sound's own wobble (a filtered sub read 429 notes for 200)
        if seg[-1] - seg[0] >= RISE_DB and env_db[k] <= env_db[k - 1] and float(env_db[max(0, k - back): k].max()) - env_db[k] >= DIP_DB:
            out.append(float(env_t[k]))
            k += w
        else:
            k += 1
    return out


def segments(y):
    """-> [(t0, t1, midi, rms)] notes of the line."""
    midi, prob = _f0_track(y)
    env_t, env_db = _fine_env(y)
    n = len(midi)
    t_of = lambda k: k * HOP / RATE
    rms = np.array([np.sqrt(np.mean(y[k * HOP: k * HOP + FRAME].astype(np.float64) ** 2)) if k * HOP < len(y) else 0.0 for k in range(n)])
    raw = _pitch_runs(midi, n, t_of, rms)
    # repeated notes: a run split at every re-attack of the fine envelope; starts refined to the rise
    notes = []
    for t0, t1, m, r in raw:
        cuts = [t for t in _rises(env_t, env_db, t0 + ONSET_MIN_S, t1 - MIN_NOTE_S)]
        bounds = [t0] + cuts + [t1]
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            s2 = _refine_start(s, env_t, env_db) if i == 0 else s
            if e - s2 >= MIN_NOTE_S:
                seg_r = rms[int(s2 * RATE / HOP): max(int(e * RATE / HOP), int(s2 * RATE / HOP) + 1)]
                notes.append((s2, e, m, float(seg_r.max()) if len(seg_r) else r))
    return notes


def _pitch_runs(midi, n, t_of, rms):
    """-> [(t0, t1, midi, rms)] runs of one pitch in the track (no re-attack splitting)."""
    notes = []
    k = 0
    gap_frames = max(1, int(GAP_S * RATE / HOP))
    while k < n:
        if not np.isfinite(midi[k]):
            k += 1
            continue
        start = k
        cur = int(round(midi[k]))
        j = k + 1
        unvoiced = 0
        while j < n:
            if np.isfinite(midi[j]):
                unvoiced = 0
                if int(round(midi[j])) != cur:
                    # a pitch change that holds for 3 frames ends the note
                    ahead = midi[j: j + 3]
                    if np.all(np.isfinite(ahead)) and np.all(np.round(ahead) != cur):
                        break
            else:
                unvoiced += 1
                if unvoiced >= gap_frames:
                    break
            j += 1
        end = j - (unvoiced if unvoiced >= gap_frames else 0)
        if t_of(end) - t_of(start) >= MIN_NOTE_S:
            seg = midi[start:end]
            seg = seg[np.isfinite(seg)]
            body = seg[max(1, len(seg) // 6):] if len(seg) > 6 else seg          # the pitch after the attack settles
            notes.append((t_of(start), t_of(end), int(round(float(np.median(body)))), float(np.max(rms[start:end]) if end > start else 0.0)))
        k = max(j, start + 1)
    return notes


def read(y, beats, period, progress=None):
    """The bass stem -> [instrument dict] (one voice) in the reading's format."""
    y = INS._mono(y)
    notes = segments(y)
    if FINE_PASS:
        notes = _fine_splits(notes, y)
    # an isolated short note far below (or above) both its neighbours is the tracker slipping on an
    # attack (the synthetic bass line read F#1 under its E2-B2 line), not a note of the song
    kept = []
    for i, (t0, t1, m, r) in enumerate(notes):
        prev_m = notes[i - 1][2] if i else None
        next_m = notes[i + 1][2] if i + 1 < len(notes) else None
        far = [abs(m - x) > OUTLIER_SEMIS for x in (prev_m, next_m) if x is not None]
        if (t1 - t0) < OUTLIER_MAX_S and far and all(far):
            continue
        kept.append((t0, t1, m, r))
    notes = kept
    n_oct = 0
    if OCTAVE_ODD and notes:
        lo, hi = INS.PITCH_RANGE["bass"]
        notes, n_oct = _octave_fix(notes, y, lo, hi)
    if progress:
        progress(f"bass: pitch track -> {len(notes)} notes ({n_oct} octaves corrected)")
    if len(notes) < 4:
        return []
    beats = np.asarray(beats, dtype=np.float64)
    loud = np.array([r for _a, _b, _m, r in notes])
    ref = float(np.percentile(loud[loud > 0], 95)) if np.any(loud > 0) else 1.0
    # notes far under the line's level are the tracker voicing the floor (the synthetic bass read F#1 in
    # its silences), not notes
    notes = [n for n in notes if n[3] >= ref * 10 ** (FLOOR_DB / 20.0)]
    events = []
    for t0, t1, m, r in notes:
        pos = INS.beat_step_of(beats, t0, period)
        if pos is None:
            continue
        b, st = pos
        grid_t = INS.event_time(beats, b, st)
        dur_steps = max(1, int(round((t1 - t0) / (period / STEPS))))
        vel = float(np.clip(1.0 + 20 * np.log10(max(r, 1e-9) / ref) / VEL_RANGE_DB, 0.15, 1.0))
        events.append([int(b), int(st), int(m), int(dur_steps), round(vel, 3), 0.9, round(float(t0 - grid_t), 4)])
    events.sort(key=lambda e: (e[0], e[1]))
    if OCTAVE_CHECK:
        # every note's pitch re-checked over its own length against the stem's harmonic salience
        # (the old reader's verify_pitches: another pitch must be 1.3x as salient, an octave with odd
        # harmonics present wins) - slap and synth bass octave errors (funk truth track: 0.74 exact vs
        # 0.83 octave-blind)
        lo, hi = INS.PITCH_RANGE["bass"]
        events, _n = INS.verify_pitches(y, events, beats, period, lo, hi)
    ms = [e[2] for e in events]
    # the exemplar: the loudest note that is isolated (no neighbour within 80 ms) - the cut plays as the sound
    best = None
    for i, (t0, t1, m, r) in enumerate(notes):
        prev_end = notes[i - 1][1] if i else -1.0
        nxt = notes[i + 1][0] if i + 1 < len(notes) else t0 + 10.0
        if t0 - prev_end >= 0.08 and nxt - t0 >= 0.25 and (best is None or r > best[0]):
            best = (r, t0, min(t1, t0 + 1.0), m)
    if best is None:
        r, t0, t1, m = max((n[3], n[0], n[1], n[2]) for n in notes)
        best = (r, t0, min(t1, t0 + 1.0), m)
    import librosa
    ex = y[int(best[1] * RATE):int(best[2] * RATE)]
    cen = float(librosa.feature.spectral_centroid(y=np.ascontiguousarray(ex), sr=RATE).mean()) if len(ex) > 2048 else 100.0
    durs = [t1 - t0 for t0, t1, _m, _r in notes]
    inst = {"stem": "bass", "kind": "hit", "pitched": True, "range": [int(min(ms)), int(max(ms))],
            "centroid_hz": round(cen, 1), "decay_db": 0.0, "dur_s": round(float(np.median(durs)), 3), "n": len(events),
            "level_db": 0.0, "level_dbfs": round(float(20 * np.log10(ref + 1e-9)), 1),
            "exemplar": [round(float(best[1]), 4), round(float(best[2]), 4)], "exemplar_midi": int(best[3]),
            "hint": "bass", "reader": "pitchtrack", "events": events}
    return [inst]
